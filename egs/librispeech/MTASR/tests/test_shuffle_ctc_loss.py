#!/usr/bin/env python3
"""Tests for the recipe-independent graph/shuffle CTC criterion."""

import unittest

import k2
import torch

from shuffle_ctc import GraphCTCLoss, ShuffleCTCLoss, graph_ctc_loss


class ShuffleCTCLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 6, 4, requires_grad=True)
        self.log_probs = self.logits.log_softmax(dim=-1)
        self.lengths = torch.tensor([6, 5])
        self.graphs = k2.ctc_graph(
            [[1, 2], [2, 1]], modified=False, device="cpu"
        )

    def test_reductions_are_consistent(self):
        losses = graph_ctc_loss(
            self.log_probs,
            self.graphs,
            self.lengths,
            reduction="none",
            output_beam=20,
        )
        summed = graph_ctc_loss(
            self.log_probs,
            self.graphs,
            self.lengths,
            reduction="sum",
            output_beam=20,
        )
        mean = graph_ctc_loss(
            self.log_probs,
            self.graphs,
            self.lengths,
            reduction="mean",
            output_beam=20,
        )
        self.assertTrue(torch.allclose(summed, losses.sum()))
        self.assertTrue(torch.allclose(mean, losses.mean()))

    def test_loss_has_finite_gradients(self):
        criterion = ShuffleCTCLoss(reduction="sum", output_beam=20)
        loss = criterion(self.log_probs, self.graphs, self.lengths)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(self.logits.grad.abs().sum().item(), 0)

    def test_module_and_function_match(self):
        module = GraphCTCLoss(reduction="none", output_beam=20)
        actual = module(self.log_probs, self.graphs, self.lengths)
        expected = graph_ctc_loss(
            self.log_probs,
            self.graphs,
            self.lengths,
            reduction="none",
            output_beam=20,
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "log_probs"):
            graph_ctc_loss(
                self.log_probs[0], self.graphs, self.lengths, output_beam=20
            )


if __name__ == "__main__":
    unittest.main()
