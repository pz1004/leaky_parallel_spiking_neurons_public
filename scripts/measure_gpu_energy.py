"""Serial gross-GPU energy and component timing on frozen validation inputs."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from spikingjelly.activation_based import functional

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.config import load_config, file_sha256, utc_now, environment_summary
from experiments.cli import _load_checkpoint
from experiments.data import build_dataset
from experiments.engine import seed_everything
from lpsn_neuron import LPSpikingNeuron, soft_reset_correction
from experiments.release_paths import ORIGINAL, EVIDENCE, resolve_path

OUTPUT = EVIDENCE / 'energy'


def resolved_config(path, **overrides):
    """Load a release config without depending on the caller's working directory."""
    cfg = load_config(path)
    paths = {name: str(resolve_path(getattr(cfg, name)))
             for name in ('data_root', 'split_manifest', 'output_root', 'frame_cache_root')
             if getattr(cfg, name) is not None}
    return replace(cfg, **{**paths, **overrides})


def source_snapshot(output, script):
    """Freeze this fresh invocation's executable source before measurements."""
    script = Path(script).resolve()
    sources = [script, PROJECT / 'lpsn_neuron.py', PROJECT / 'lpsn_variants.py']
    sources += sorted((PROJECT / 'experiments').glob('*.py'))
    sources += sorted((PROJECT / 'scripts').glob('*.py'))
    manifest = {'created_at': utc_now(), 'scope': 'fresh release invocation',
                'script_sha256': file_sha256(script),
                'source_sha256': {str(p.relative_to(PROJECT)): file_sha256(p) for p in sources}}
    target = output / (script.stem + '_source_manifest.json')
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite an existing invocation: {target}; choose a fresh LPSN_EVIDENCE_ROOT')
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / (script.stem + '_executed.py')).write_bytes(script.read_bytes())
    return manifest


def energy_per_sample(start_mj, end_mj, calls, batch):
    if end_mj <= start_mj or calls <= 0 or batch <= 0:
        raise ValueError('Invalid energy counter delta or sample count')
    return (end_mj - start_mj) / 1000.0 / (calls * batch)


class EnergyCounter:
    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle = nvml_handle_for_visible_device(self.lib)
        name = ctypes.create_string_buffer(96)
        self.check(self.lib.nvmlDeviceGetName(self.handle, name, len(name)))
        if b'A100' not in name.value:
            raise RuntimeError(f'Expected A100, found {name.value}')
        self.name = name.value.decode()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    def read(self):
        value = ctypes.c_ulonglong()
        self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)))
        return value.value


def nvml_handle_for_visible_device(lib):
    """Match NVML's physical device to PyTorch logical CUDA device zero."""
    handle = ctypes.c_void_p()
    identifier = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if identifier:
        identifier = str(identifier)
        if not identifier.startswith(('GPU-', 'MIG-')):
            identifier = 'GPU-' + identifier
        code = lib.nvmlDeviceGetHandleByUUID(identifier.encode(), ctypes.byref(handle))
    else:
        identifier = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
        if identifier.startswith(('GPU-', 'MIG-')):
            code = lib.nvmlDeviceGetHandleByUUID(identifier.encode(), ctypes.byref(handle))
        elif identifier.isdigit():
            code = lib.nvmlDeviceGetHandleByIndex_v2(int(identifier), ctypes.byref(handle))
        else:
            raise RuntimeError('Cannot determine the NVML device for logical CUDA device zero')
    EnergyCounter.check(code)
    return handle


def config_for(dataset, model, seed=42, t=None):
    t = t or (16 if dataset == 'hardvs' else 32)
    study = 'hardvs_main' if dataset == 'hardvs' else 'temporal_native'
    root = ORIGINAL / 'generated_configs'
    protocol = EVIDENCE / 'protocol.json'
    if protocol.exists():
        root = resolve_path(json.loads(protocol.read_text()).get('baseline_configs_root', root))
    path = root / f'{study}/{dataset}/{model}/T{t}/seed_{seed}.yml'
    return resolved_config(path, num_workers=0, device='cuda:0'), path


def check_no_training():
    offenders = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        if b'experiments.cli' in args and b'train' in args:
            offenders.append(path.parent.name)
    if offenders:
        raise RuntimeError(f'Concurrent training detected: {offenders}')


def component_profile(model, inputs):
    rows = []
    def hook(name):
        def measure(node, args):
            x = args[0]
            w = node.effective_weight()
            flat = x.reshape(x.shape[0], -1)
            membrane = (w @ flat).reshape_as(x)
            history = node._exact_detached_soft_reset_spikes(membrane)
            tasks = [('temporal_matrix_multiply', lambda: w @ flat),
                     ('sequential_history_scan', lambda: node._exact_detached_soft_reset_spikes(membrane)),
                     ('reset_correction_matrix_and_apply', lambda: soft_reset_correction(history, node.log_tau, node.beta))]
            for component, fn in tasks:
                for _ in range(5): fn()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20): fn()
                end.record(); end.synchronize()
                rows.append(dict(layer=name, shape=list(x.shape), component=component, milliseconds=begin.elapsed_time(end)/20))
        return measure
    hooks = [m.register_forward_pre_hook(hook(name)) for name, m in model.named_modules()
             if isinstance(m, LPSpikingNeuron) and m.soft_reset_strategy == 'exact_detached']
    model(inputs); functional.reset_net(model)
    for h in hooks: h.remove()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Outputs: LPSN_EVIDENCE_ROOT/energy (default: reproduction_outputs/energy). This command performs GPU inference when invoked.')
    parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'protocol.json').exists() or any(OUTPUT.glob('*_seed42_B*.json')):
        raise RuntimeError('Energy evidence already exists; choose a fresh LPSN_EVIDENCE_ROOT')
    lock = (EVIDENCE / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_no_training()
    source = source_snapshot(OUTPUT, __file__)
    seed_everything(42)
    meter = EnergyCounter()
    idle = []
    for _ in range(10):
        before = meter.read(); start = time.perf_counter(); time.sleep(1)
        after = meter.read(); seconds = time.perf_counter() - start
        idle.append(dict(start_mj=before, end_mj=after, seconds=seconds, watts=(after-before)/1000/seconds))
    protocol = dict(created_at=utc_now(), counter='nvmlDeviceGetTotalEnergyConsumption', counter_units='mJ',
                    source='https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html',
                    idle_blocks=idle, environment=environment_summary(), source_manifest=source,
                    gpu_process_snapshot=subprocess.check_output(['nvidia-smi'], text=True),
                    scope='gross GPU board energy, including idle and background graphics; excludes CPU, acquisition, preprocessing; dense GPU execution',
                    inference='whole model forward on device-resident validation tensors; reset_net between calls; float32, inference_mode')
    (OUTPUT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for dataset in ['cifar10_dvs', 'dvs_gesture', 'hardvs']:
        cfg, _ = config_for(dataset, 'psn')
        # Raw preprocessing is identical and writes no cache if HARDVS lacks a cache.
        ds, split = build_dataset(replace(cfg, cache_frames=False), 'val')
        first = [ds[i] for i in range(16)]
        tensor = torch.stack([v[0] for v in first])
        sample_ids = [v[2] for v in first]
        digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        for name in ['psn', 'lpsn_a1_lif_init', 'lpsn_a2_soft'] + (['lif'] if dataset == 'hardvs' else []):
            cfg, path = config_for(dataset, name)
            model, checkpoint, checkpoint_path = _load_checkpoint(cfg)
            for batch in [1, 16]:
                target = OUTPUT / f'{dataset}_{name}_seed42_B{batch}.json'
                if target.exists():
                    continue
                inputs = tensor[:batch].to(cfg.device)
                with torch.inference_mode():
                    for _ in range(100):
                        model(inputs); functional.reset_net(model)
                    torch.cuda.synchronize()
                    blocks = []
                    for block in range(10):
                        check_no_training()
                        torch.cuda.reset_peak_memory_stats()
                        start_mj = meter.read(); started = time.perf_counter(); calls = 0
                        elapsed = 0.
                        while calls < 100 or elapsed < 5.:
                            for _ in range(10):
                                model(inputs); functional.reset_net(model)
                            calls += 10
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter()-started
                        end_mj = meter.read()
                        blocks.append(dict(block=block, start_mj=start_mj, end_mj=end_mj, seconds=elapsed,
                            forward_calls=calls, joules_per_sample=energy_per_sample(start_mj,end_mj,calls,batch),
                            latency_ms_per_batch=elapsed*1000/calls, throughput=calls*batch/elapsed,
                            peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2))
                    components = component_profile(model, inputs) if name == 'lpsn_a2_soft' else []
                result = dict(dataset=dataset, model=name, architecture=cfg.architecture, time_steps=cfg.time_steps,
                              seed=42, batch_size=batch, selected_epoch=checkpoint['epoch'],
                              checkpoint=str(checkpoint_path), checkpoint_sha256=file_sha256(checkpoint_path),
                              split_hash=split['split_hash'], config_sha256=file_sha256(path),
                              validation_ids=sample_ids[:batch], validation_input_16_sha256=digest,
                              warmup=100, blocks=blocks, component_microbenchmarks=components,
                              component_note='isolated per-layer CUDA event timings, 5 warmups and 20 calls; not an additive end-to-end decomposition')
                target.write_text(json.dumps(result, indent=2)+'\n')
                print(json.dumps({'complete':target.name,'joules_per_sample':np.mean([r['joules_per_sample'] for r in blocks])}),flush=True)
                del inputs
            del model
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
