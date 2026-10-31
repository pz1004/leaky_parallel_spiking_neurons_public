from __future__ import annotations

import unittest

import torch
from spikingjelly.activation_based import neuron, surrogate

from experiments.neurons import RPSU, build_neuron


class NeuronAdapterTests(unittest.TestCase):
    def test_lpsn_rejects_single_step_before_mixing_batch_items(self) -> None:
        for name in ("lpsn_a1_lif_init", "lpsn_a2_soft"):
            with self.subTest(name=name):
                node = build_neuron(name, 3)
                self.assertEqual(node.supported_step_mode(), ("m",))
                with self.assertRaises(ValueError):
                    node.step_mode = "s"
                self.assertEqual(node.step_mode, "m")

    def test_fixed_a2_matches_soft_reset_input_and_upstream_gradients(self) -> None:
        torch.manual_seed(127)
        source = torch.randn(12, 3, 4) * 1.3 + 1.0
        input_a = source.clone().requires_grad_()
        input_b = source.clone().requires_grad_()
        weight_a = torch.eye(4, requires_grad=True)
        weight_b = weight_a.detach().clone().requires_grad_()
        a2 = build_neuron("lpsn_a2_soft", 12)
        reference = neuron.LIFNode(
            tau=2.0, decay_input=True, v_reset=None, detach_reset=True,
            surrogate_function=surrogate.ATan(alpha=2.0), step_mode="m", backend="torch",
        )
        actual = a2(input_a @ weight_a)
        expected = reference(input_b @ weight_b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        coefficients = torch.randn_like(actual)
        (actual * coefficients).sum().backward()
        (expected * coefficients).sum().backward()
        torch.testing.assert_close(input_a.grad, input_b.grad, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(weight_a.grad, weight_b.grad, atol=2e-6, rtol=2e-6)

    def test_every_adapter_preserves_sequence_shape(self) -> None:
        inputs = torch.randn(8, 2, 3)
        for name in ("lif", "psn", "masked_psn", "sliding_psn", "rpsu", "lpsn_a1_lif_init", "lpsn_a2_soft"):
            with self.subTest(name=name):
                output = build_neuron(name, 8)(inputs)
                self.assertEqual(tuple(output.shape), tuple(inputs.shape))

    def test_rpsu_learned_reset_matrix_is_strictly_causal_after_masking(self) -> None:
        node = RPSU(8)
        effective = node.weight.detach() * node.causal_mask
        self.assertEqual(int(torch.count_nonzero(torch.triu(effective, diagonal=1))), 0)

    def test_native_length_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            RPSU(8)(torch.randn(16, 2, 3))


if __name__ == "__main__":
    unittest.main()
