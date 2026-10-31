import unittest
from dataclasses import replace
import numpy as np
import torch
from experiments.config import load_config
from experiments.model import build_model, SharedVGGSNN
from experiments.neurons import build_neuron
from experiments.statistics import crossed_estimates


class ExtensionTests(unittest.TestCase):
    def test_architecture_state_and_run_identity(self):
        kwargs = dict(neuron_name='psn', time_steps=8, num_classes=3, image_size=32)
        torch.manual_seed(42)
        old = SharedVGGSNN(**kwargs)
        torch.manual_seed(42)
        new = build_model('vgg_small', **kwargs)
        for k, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[k]), k)
        config = load_config('configs/smoke.yml')
        paths = {replace(config, architecture=a).run_dir for a in ['vgg_small', 'vgg_small_half', 'spiking_resnet18']}
        self.assertEqual(len(paths), 3)
        x = torch.randn(2, 8, 2, 32, 32)
        for architecture in ['vgg_small_half', 'spiking_resnet18']:
            model = build_model(architecture, **kwargs)
            convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
            if architecture == 'vgg_small_half':
                self.assertEqual(convolutions[0].out_channels, 32)
                hidden = next(module for module in model.modules() if isinstance(module, torch.nn.Linear))
                self.assertEqual(hidden.out_features, 256)
            else:
                self.assertEqual(convolutions[0].kernel_size, (7, 7))
                self.assertEqual(convolutions[0].stride, (2, 2))
            y = model(x)
            self.assertEqual(y.shape, (8, 2, 3))
            y.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_unknown_architecture_fails_in_config_and_model_dispatch(self):
        config = replace(load_config('configs/smoke.yml'), architecture='unknown_architecture')
        with self.assertRaisesRegex(ValueError, 'architecture'):
            config.validate()
        with self.assertRaisesRegex(ValueError, 'architecture'):
            build_model('unknown_architecture', neuron_name='psn', time_steps=8, num_classes=3)

    def test_operator_controls_and_residual_trainability(self):
        anchor = build_neuron('lpsn_a1_lif_init', 8)
        for kind in ['fixed_diagonal', 'fixed_causal_uniform']:
            n = build_neuron(kind, 8)
            torch.testing.assert_close(n.weight.sum(1), anchor.kernel_weight().sum(1))
            self.assertFalse(any(p.requires_grad for p in n.parameters()))
            self.assertEqual(torch.count_nonzero(torch.triu(n.weight, diagonal=1)).item(), 0)
        for name, kernel, residual in [('lpsn_a3_kernel', True, False), ('lpsn_fixed_kernel_residual', False, True), ('lpsn_a4a_frozen_bias', True, True)]:
            n = build_neuron(name, 8)
            self.assertEqual(n.log_tau.requires_grad, kernel)
            self.assertEqual(n.alpha.requires_grad, kernel)
            self.assertEqual(n.delta_W.requires_grad, residual)
            self.assertFalse(n.bias.requires_grad)
            self.assertEqual(n.delta_W.count_nonzero().item(), 0)
            torch.testing.assert_close(n.effective_weight(), anchor.effective_weight())
            n(torch.randn(8, 2, 3, requires_grad=True)).sum().backward()
            if residual:
                self.assertEqual(torch.triu(n.delta_W.grad, diagonal=1).count_nonzero().item(), 0)
                self.assertEqual(n.residual_l2, 1e-3)

    def test_crossed_draws_preserve_shared_uncertainty(self):
        rows = [dict(seed=s, draw=d, v=v) for s in range(5) for d, v in enumerate([0., 1.])]
        estimates = crossed_estimates(rows, 'v')
        self.assertTrue(set(estimates) <= {0., .5, 1.})
        self.assertGreater(estimates.std(), .3)
        with self.assertRaises(ValueError):
            crossed_estimates(rows[:-1], 'v')
        with self.assertRaises(ValueError):
            crossed_estimates(rows + [rows[0]], 'v')

    def test_exact_a1_a2_forward(self):
        torch.manual_seed(0)
        x = torch.randn(8, 3, 5) * 3
        for name, reset in [('lpsn_a1_lif_init', False), ('lpsn_a2_soft', True)]:
            n = build_neuron(name, 8)
            v = torch.zeros_like(x[0]); pre = []; spikes = []
            for t in range(8):
                h = .5 * v + .5 * x[t]
                s = (h >= 1).float()
                pre.append(h); spikes.append(s)
                v = h - s if reset else h
            torch.testing.assert_close(n.forward_logits(x, include_bias=False), torch.stack(pre), atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(n(x), torch.stack(spikes)))

    def test_energy_counter_units(self):
        from scripts.measure_gpu_energy import energy_per_sample
        self.assertEqual(energy_per_sample(1000, 17000, 2, 4), 2.)
        with self.assertRaises(ValueError): energy_per_sample(1000,999,2,4)

    def test_checkpoint_rejects_architecture_mismatch_before_loading(self):
        import tempfile
        from pathlib import Path
        from experiments.cli import _load_checkpoint
        config=load_config('configs/smoke.yml')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            torch.save({'config':{'architecture':'vgg_small_half'}},path)
            with self.assertRaisesRegex(ValueError,'architecture'):
                _load_checkpoint(config,path)

    def test_interrupted_test_evaluation_cannot_be_reopened(self):
        import tempfile,json
        from experiments.cli import evaluate_checkpoint
        from experiments.config import file_sha256
        with tempfile.TemporaryDirectory() as d:
            cfg=replace(load_config('configs/smoke.yml'),experiment_id='extension_test',output_root=d)
            cfg.run_dir.mkdir(parents=True)
            ck=cfg.run_dir/'checkpoint_best.pt';ck.write_bytes(b'not-read-by-this-guard')
            (cfg.run_dir/'train_result.json').write_text(json.dumps({'config_hash':cfg.config_hash,'checkpoint_sha256':file_sha256(ck)}))
            (cfg.run_dir/'test_evaluation_started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'refusing another test read'):
                evaluate_checkpoint(cfg)

    def test_aggregation_keeps_architectures_separate(self):
        import csv,json,tempfile
        from argparse import Namespace
        from pathlib import Path
        from experiments.cli import command_analyze
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for a,v in [('vgg_small',.2),('vgg_small_half',.8)]:
                rows.append(dict(experiment_id='temporal_native',dataset='synthetic',architecture=a,model='psn',
                                 time_steps=8,condition='clean',severity=0,seed=42,draw=0,accuracy=v,macro_f1=v))
            (root/'locked_test_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            output=root/'summary.csv'
            command_analyze(Namespace(runs_root=root,output=output))
            result=list(csv.DictReader(output.open()))
            self.assertEqual({r['architecture']:float(r['accuracy_mean']) for r in result},
                             {'vgg_small':.2,'vgg_small_half':.8})

    def test_corrected_mathematical_boundaries(self):
        import math
        # Exact cycle enumeration at an arbitrary integer origin, L=4.
        for k in range(7):
            enumerated=np.mean([all((origin+offset)%4!=0 for offset in range(k)) for origin in range(4)])
            self.assertEqual(enumerated,max(4-k,0)/4)
        # Matrix powers verify the seven-layer finite-horizon expression.
        T=8;g=lam=.5
        K=np.array([[g*lam**(t-s) if t>=s else 0. for s in range(T)] for t in range(T)])
        r=np.linalg.matrix_power(K,7).sum(axis=0)/T
        analytic=[g**7/T*sum(math.comb(k+6,6)*lam**k for k in range(T-s)) for s in range(T)]
        np.testing.assert_allclose(r,analytic)
        self.assertEqual(r[-1],g**7/T)
        # Superposition fails with hard reset, while each separate input is subthreshold.
        def h2(x):
            h1=.5*x
            return .5*h1*(h1<1)
        self.assertAlmostEqual(h2(1.2)+h2(1.2),.6)
        self.assertEqual(h2(2.4),0.)
        # Continuous stationary residual-life branches agree at the refractory boundary.
        for r0 in [0.,1.,3.]:
            nu=.7;left=1-r0/(r0+1/nu);right=1/(1+nu*r0)
            self.assertAlmostEqual(left,right)

if __name__ == '__main__':
    unittest.main()
