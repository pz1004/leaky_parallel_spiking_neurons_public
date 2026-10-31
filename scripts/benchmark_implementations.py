"""Read-only checkpoint execution audit; writes only execution_comparison evidence.

No optimization of checkpoint parameters: Adam learning rate is zero.  Training
mode is used solely to measure the forward/backward/optimizer workload.  Mutable
normalization buffers are restored between blocks, and no checkpoint is written.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import file_sha256, environment_summary, utc_now
from experiments.cli import _load_checkpoint
from experiments.data import build_dataset
from experiments.engine import seed_everything
from experiments.neurons import FixedTemporalOperator
from lpsn_neuron import LPSpikingNeuron, _effective_tau
from lpsn_variants import build_lpsn_neuron
from dataclasses import replace
from experiments.release_paths import EVIDENCE
from scripts.measure_gpu_energy import resolved_config, source_snapshot, check_no_training, nvml_handle_for_visible_device, config_for as baseline_config_for

OUT = EVIDENCE / "execution_comparison"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Meter:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.handle = nvml_handle_for_visible_device(self.lib)
        name = ctypes.create_string_buffer(96)
        code = self.lib.nvmlDeviceGetName(self.handle, name, len(name))
        if code:
            raise RuntimeError(f'NVML could not identify the measured GPU name (error {code})')
        uuid = ctypes.create_string_buffer(96)
        uuid_code = self.lib.nvmlDeviceGetUUID(self.handle, uuid, len(uuid))
        self.device_metadata = {
            'logical_cuda_device': 0,
            'nvml_name': name.value.decode(),
            'nvml_uuid': uuid.value.decode() if uuid_code == 0 else None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        }

    def read(self):
        result = ctypes.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(result)) == 0
        return result.value


class RetainedCorrectionA2(nn.Module):
    """Fixed A2: retain the pre-update correction from the detached history scan."""
    def __init__(self, original):
        super().__init__()
        assert not any(p.requires_grad for p in original.parameters())
        self.original = original
        self.step_mode = "m"

    def forward(self, x):
        n = self.original
        membrane = (n.effective_weight() @ x.flatten(1)).view_as(x)
        with torch.no_grad():
            lam = 1 - 1 / _effective_tau(n.log_tau)
            state = torch.zeros_like(membrane[0])
            corrections = []
            for t in range(n.T):
                corrections.append(state)
                spike = n.surrogate_function(membrane[t] - state - n.v_threshold)
                state = lam * (state + n.beta * spike)
            correction = torch.stack(corrections)
        return n.surrogate_function(membrane - correction + n._bias_view(membrane))


class ElementwiseDiagonal(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("diagonal", original.weight.diagonal().clone())
        self.register_buffer("bias", original.bias.flatten().clone())
        self.surrogate_function = copy.deepcopy(original.surrogate_function)
        self.step_mode = "m"

    def forward(self, x):
        dims = (x.shape[0],) + (1,) * (x.ndim - 1)
        return self.surrogate_function(self.diagonal.view(dims) * x + self.bias.view(dims))


def lif():
    return neuron.LIFNode(tau=2.0, decay_input=True, v_threshold=1.0,
                          v_reset=None, detach_reset=True,
                          surrogate_function=surrogate.ATan(alpha=2.0),
                          step_mode="m", backend="torch")


def transform(model, implementation):
    for name, child in list(model.named_children()):
        if isinstance(child, LPSpikingNeuron) and implementation == "a2_retained":
            setattr(model, name, RetainedCorrectionA2(child))
        elif isinstance(child, LPSpikingNeuron) and implementation == "lif_soft_detached":
            setattr(model, name, lif())
        elif isinstance(child, FixedTemporalOperator) and implementation == "diagonal_elementwise":
            setattr(model, name, ElementwiseDiagonal(child))
        else:
            transform(child, implementation)
    return model


def comparison(a, b):
    delta = (a.double() - b.double())
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-30)),
            "different_elements": int(torch.count_nonzero(delta)), "elements": a.numel()}


def exactness():
    records = []
    for t in [8, 16, 32]:
        for pattern in ["mixed", "nonnegative", "subthreshold"]:
            torch.manual_seed(1729 + t)
            x = torch.randn(t, 3, 11, dtype=torch.float64)
            x = {"mixed": x * 1.7 + 1.2, "nonnegative": x.abs() * 2.5,
                 "subthreshold": x * 0.15}[pattern]
            weight = torch.randn_like(x)
            base = build_lpsn_neuron(T=t, variant="lpsn_a2_soft").double()
            # Construct exactly the intended scalar tau=2 in float64, rather than
            # promoting an already rounded float32 log(2).
            with torch.no_grad(): base.log_tau.fill_(math.log(2.0))
            outputs = {}
            for name, node in [("a2_original", base),
                               ("a2_retained", RetainedCorrectionA2(copy.deepcopy(base))),
                               ("lif_soft_detached", lif().double())]:
                node.train(); inp = x.clone().requires_grad_(True)
                y = node(inp); grad = torch.autograd.grad((y * weight).sum(), inp)[0]
                outputs[name] = (y.detach(), grad.detach())
                functional.reset_net(node)
            for name in ["a2_retained", "lif_soft_detached"]:
                yo, go = outputs["a2_original"]; y, g = outputs[name]
                records.append({"T": t, "pattern": pattern, "reference": "a2_original",
                    "implementation": name, "output": comparison(yo, y), "gradient": comparison(go, g),
                    "output_exact": torch.equal(yo, y), "gradient_pass": torch.allclose(go, g, atol=1e-11, rtol=1e-10),
                    "reference_gradient_norm": float(go.norm()), "spike_fraction": float(yo.mean())})
            diag = FixedTemporalOperator(t, "fixed_diagonal").double()
            vals=[]
            for node in [diag, ElementwiseDiagonal(diag)]:
                inp=x.clone().requires_grad_(True); y=node(inp)
                vals.append((y.detach(),torch.autograd.grad((y*weight).sum(), inp)[0]))
            records.append({"T":t,"pattern":pattern,"reference":"diagonal_dense",
                "implementation":"diagonal_elementwise","output":comparison(vals[0][0],vals[1][0]),
                "gradient":comparison(vals[0][1],vals[1][1]),"output_exact":torch.equal(vals[0][0],vals[1][0]),
                "gradient_pass":torch.allclose(vals[0][1],vals[1][1],atol=1e-11,rtol=1e-10),
                "reference_gradient_norm":float(vals[0][1].norm())})
    save("exactness.json", {"dtype":"float64", "gradient_objective":"fixed random weighted sum of spikes",
         "tolerance":{"atol":1e-11,"rtol":1e-10},"records":records,
         "all_pass":all(r["output_exact"] and r["gradient_pass"] for r in records)})
    if not all(r["output_exact"] and r["gradient_pass"] for r in records):
        raise RuntimeError("Float64 operator equivalence failed; inspect exactness.json")


def config_for(family, t):
    if family == "diagonal":
        protocol_path=EVIDENCE/'protocol.json'
        if protocol_path.exists():
            protocol=json.loads(protocol_path.read_text())
            matches=[r for r in protocol['runs'] if r['dataset']=='cifar10_dvs'
                     and r['model']=='fixed_diagonal' and int(r['seed'])==42
                     and r['architecture']=='vgg_small']
            if len(matches)!=1:raise ValueError('Expected one diagonal seed-42 declaration in the extension protocol')
            from experiments.release_paths import resolve_path
            path=resolve_path(matches[0]['config'])
            if file_sha256(path)!=matches[0]['sha256']:
                raise RuntimeError('Extension config changed since protocol preparation')
        else:
            path=ROOT/"configs/extension/components/cifar10_dvs/vgg_small/fixed_diagonal/seed_42.yml"
        return resolved_config(path, device="cpu", num_workers=0, cache_frames=False),path
    model="lpsn_a1_lif_init" if family=="a1" else "lpsn_a2_soft"
    cfg,path=baseline_config_for('cifar10_dvs',model,42,t=t)
    return replace(cfg,device="cpu",num_workers=0,cache_frames=False),path


def full_check(base, implementations, inp_cpu, labels_cpu):
    values={}; results=[]
    for implementation in implementations:
        model=transform(copy.deepcopy(base),implementation).cuda().train()
        # LIF's inference fast path intentionally has no surrogate derivative.
        # Keep neuron training paths active, while freezing spatial normalization
        # and dropout to compare meaningful gradients at identical activations.
        for node in model.modules():
            if isinstance(node,(nn.modules.batchnorm._BatchNorm,nn.Dropout,layer.Dropout)):
                node.eval()
        seed_everything(8675309)
        x=inp_cpu[:1].cuda().requires_grad_(True); y=labels_cpu[:1].cuda()
        model.zero_grad(set_to_none=True)
        output=model(x); loss=F.cross_entropy(output.mean(0),y); loss.backward()
        gradients={n:p.grad.detach().cpu().flatten() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        values[implementation] = (output.detach().cpu(), x.grad.detach().cpu(), gradients)
        functional.reset_net(model); del model,x,output,loss; torch.cuda.empty_cache()
    ref=values[implementations[0]]
    for name in implementations[1:]:
        out,inp,grads=values[name]
        assert ref[2].keys()==grads.keys()
        a=torch.cat([ref[2][k] for k in sorted(grads)])
        b=torch.cat([grads[k] for k in sorted(grads)])
        results.append({"reference":implementations[0],"implementation":name,
            "output":comparison(ref[0],out),"input_gradient":comparison(ref[1],inp),
            "parameter_gradients":comparison(a,b),"output_exact":torch.equal(ref[0],out),
            "gradient_pass":torch.allclose(a,b,atol=1e-4,rtol=1e-3),
            "reference_gradient_norm":float(a.norm()),"dtype":"float32","mode":"neuron train paths; batchnorm/dropout eval",
            "gradient_objective":"cross entropy of time-averaged output spikes on fixed validation labels"})
    return results


def train_call(model, optimizer, x, y):
    torch.cuda.synchronize(); begun=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    out=model(x); loss=F.cross_entropy(out.mean(0),y)
    torch.cuda.synchronize(); forward=time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(); backward=time.perf_counter()
    optimizer.step(); functional.reset_net(model)
    torch.cuda.synchronize(); ended=time.perf_counter()
    return {"forward_loss_ms":(forward-begun)*1000,"backward_ms":(backward-forward)*1000,
            "optimizer_reset_ms":(ended-backward)*1000,"total_step_ms":(ended-begun)*1000}


def run_block(base, implementation, x_cpu,y_cpu,batch,block,meter,seconds,steps):
    model=transform(copy.deepcopy(base),implementation).cuda()
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    x=x_cpu[:batch].cuda(); y=y_cpu[:batch].cuda()
    model.eval()
    with torch.inference_mode():
        for _ in range(10):model(x);functional.reset_net(model)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start_mj=meter.read();start=time.perf_counter();calls=0
        while calls<20 or time.perf_counter()-start<seconds:
            for _ in range(5):model(x);functional.reset_net(model)
            calls+=5;torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;end_mj=meter.read()
        infer={"start_mj":start_mj,"end_mj":end_mj,"seconds":elapsed,"calls":calls,
            "latency_ms":1000*elapsed/calls,"joules_per_sample":(end_mj-start_mj)/1000/(calls*batch),
            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
    model.train();seed_everything(8090+block)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=0.0,weight_decay=0)
    for _ in range(2):train_call(model,optimizer,x,y)
    model.load_state_dict(snapshot);functional.reset_net(model)
    # Warmed Adam state allocations are retained; state values and step counters
    # are reset. Learning rate zero leaves every learned parameter unchanged.
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):v.zero_()
    seed_everything(9000+block)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    train=[train_call(model,optimizer,x,y) for _ in range(steps)]
    peak=torch.cuda.max_memory_allocated()/2**20
    parameter_unchanged=all(torch.equal(v.detach().cpu(),snapshot[n]) for n,v in model.named_parameters())
    model.load_state_dict(snapshot);functional.reset_net(model)
    restored=all(torch.equal(v.detach().cpu(),snapshot[n]) for n,v in model.state_dict().items())
    result={"implementation":implementation,"batch":batch,"block":block,"inference":infer,
       "adam_zero_lr_steps":train,"training_peak_allocated_mib":peak,
       "learned_parameters_unchanged":parameter_unchanged,"all_state_restored":restored}
    del model,optimizer,x,y;torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__, epilog='Outputs: LPSN_EVIDENCE_ROOT/execution_comparison. The default runs GPU benchmarks; --checks-only runs CPU operator checks.')
    parser.add_argument("--seconds",type=float,default=5.0,help='Minimum inference seconds per block (reference protocol: 5).')
    parser.add_argument("--blocks",type=int,default=3,help='Distinct sampled validation batches/measurement blocks (reference protocol: 3).')
    parser.add_argument("--steps",type=int,default=5,help='Timed zero-update Adam steps per block (reference protocol: 5).')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--checks-only",action="store_true",help='Run the 27 float64 operator contrasts only.')
    modes.add_argument("--full-checks-only",action="store_true",help='Repeat full-model checks for a completed run, preserving the prior check record.')
    args=parser.parse_args()
    if args.seconds <= 0 or args.blocks < 2 or args.steps < 1:
        parser.error('--seconds must be positive, --blocks at least 2, and --steps at least 1')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(EVIDENCE/'gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_no_training()
    if args.full_checks_only:
        protocol=json.loads((OUT/"protocol.json").read_text())
        if "completed_at" not in protocol:raise RuntimeError("Wait for the timed benchmark to finish before repeating checks")
        check_dir=OUT/'rechecks'/str(time.time_ns())
        check_dir.mkdir(parents=True)
        source_snapshot(check_dir,__file__)
        records=[]
        for meta in protocol["families"]:
            cfg,path=config_for(meta["family"],meta["T"])
            if file_sha256(path)!=meta['config_sha256']:
                raise RuntimeError('Config changed since the timed run')
            base,_,checkpoint_path= _load_checkpoint(cfg)
            if file_sha256(checkpoint_path)!=meta['checkpoint_sha256_before']:
                raise RuntimeError('Checkpoint changed since the timed run')
            ds,_=build_dataset(cfg,"val")
            samples=[ds[int(i)] for i in meta["validation_indices"][:16]]
            x=torch.stack([v[0] for v in samples]);y=torch.tensor([v[1] for v in samples])
            assert tensor_hash(x)==meta["input_batch_sha256"][0]
            records.extend({"family":meta["family"],"T":meta["T"],**r} for r in full_check(base,meta["implementations"],x,y))
            del base,ds,samples,x,y
        old=OUT/"full_model_checks.json"
        if old.exists(): shutil.copy2(old,check_dir/'previous_full_model_checks.json')
        save("full_model_checks.json",{"records":records,"atol":1e-4,"rtol":1e-3,
            "check_script_sha256":file_sha256(Path(__file__)),"created_at":utc_now(),
            "check_source_file":str((check_dir/(Path(__file__).stem+'_executed.py')).relative_to(OUT)),
            "note":"Checks freeze torch and SpikingJelly dropout plus batchnorm; same RNG before every implementation; all numerical deviations retained. Earlier check records are retained under rechecks/."})
        return
    if any((OUT/name).exists() for name in ('protocol.json','raw_blocks.jsonl','exactness.json')):
        raise RuntimeError("Refusing to overwrite existing benchmark evidence; choose a fresh LPSN_EVIDENCE_ROOT")
    source=source_snapshot(OUT,__file__)
    shutil.copy2(__file__,OUT/'benchmark_source_executed.py')
    seed_everything(42);torch.set_num_threads(8)
    exactness()
    if args.checks_only:return
    meter=Meter()
    protocol={"created_at":utc_now(),"script_sha256":file_sha256(Path(__file__)),"source_manifest":source,"environment":environment_summary(),
       "gpu_snapshot":subprocess.check_output(["nvidia-smi"],text=True),"measured_device":meter.device_metadata,"blocks":args.blocks,
       "inference_min_seconds":args.seconds,"inference_min_calls":20,"inference_warmups":10,
       "workload_steps_per_block":args.steps,"workload_warmups":2,"adam_learning_rate":0.0,
       "scope":"implementation comparison on frozen validation inputs; no accuracy evaluation, no validation fitting, no checkpoint writes",
       "training_scope":"train mode, cross entropy, Adam lr0; weights unchanged, BN buffers restored; separate synchronized phase timings",
       "energy_scope":"gross GPU board counter energy including idle/background activity; device-resident inference only",
       "order_rng_seed":314159,"validation_sampling_seed":20261029,"families":[]}
    rng=random.Random(314159);checks=[]
    for family,t,implementations in [("a2",16,["a2_original","a2_retained","lif_soft_detached"]),
                                   ("a2",32,["a2_original","a2_retained","lif_soft_detached"]),
                                   ("diagonal",32,["diagonal_dense","diagonal_elementwise"]),
                                   ("a1",32,["a1_dense"])]:
        cfg,path=config_for(family,t);base,checkpoint,checkpoint_path=_load_checkpoint(cfg)
        ds,split=build_dataset(cfg,"val")
        indices=np.random.default_rng(20261029).permutation(len(ds))[:16*args.blocks]
        if len(indices)!=16*args.blocks:raise ValueError('Not enough validation examples for the requested distinct batches')
        samples=[ds[int(i)] for i in indices]
        batches=[]
        for block in range(args.blocks):
            selected=samples[16*block:16*(block+1)]
            batches.append((torch.stack([v[0] for v in selected]),torch.tensor([v[1] for v in selected])))
        meta={"family":family,"T":t,"checkpoint":str(checkpoint_path),"checkpoint_sha256_before":file_sha256(checkpoint_path),
              "config":str(path),"config_sha256":file_sha256(path),"split_hash":split["split_hash"],
              "validation_indices":indices.tolist(),"validation_ids":[v[2] for v in samples],
              "input_batch_sha256":[tensor_hash(x) for x,y in batches],"labels":[y.tolist() for x,y in batches],
              "implementations":implementations}
        protocol["families"].append(meta);save("protocol.json",protocol)
        checks.extend({"family":family,"T":t,**r} for r in full_check(base,implementations,*batches[0]))
        save("full_model_checks.json",{"records":checks,"atol":1e-4,"rtol":1e-3,
             "check_script_sha256":file_sha256(Path(__file__)),"check_source_file":"benchmark_source_executed.py",
             "note":"All float32 deviations are retained; neuron-level float64 checks separately test algebraic equivalence."})
        for block in range(args.blocks):
            schedule=[(b,imp) for b in [1,16] for imp in implementations];rng.shuffle(schedule)
            for batch,implementation in schedule:
                check_no_training()
                result=run_block(base,implementation,*batches[block],batch,block,meter,args.seconds,args.steps)
                result.update(family=family,T=t,input_batch_sha256=meta["input_batch_sha256"][block],order_in_block=schedule.index((batch,implementation)))
                with (OUT/"raw_blocks.jsonl").open("a") as handle:handle.write(json.dumps(result)+"\n")
                print(json.dumps({"family":family,"T":t,"batch":batch,"implementation":implementation,
                    "block":block,"forward_ms":result["inference"]["latency_ms"],
                    "step_ms":statistics.mean(r["total_step_ms"] for r in result["adam_zero_lr_steps"])}),flush=True)
        meta["checkpoint_sha256_after"]=file_sha256(checkpoint_path)
        assert meta["checkpoint_sha256_before"]==meta["checkpoint_sha256_after"]
        save("protocol.json",protocol)
        del base,ds,samples,batches
    protocol["completed_at"]=utc_now();save("protocol.json",protocol)


if __name__ == "__main__":main()
