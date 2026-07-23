from __future__ import annotations

import unittest

import torch

from experiments.neurons import RPSU, build_neuron


class NeuronAdapterTests(unittest.TestCase):
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
