#!/usr/bin/env python3
"""Numerical equivalence with the expanded-logit k2 reference path."""

import unittest

import k2
import torch

import shuffle_ctc.factorized as factorized_module
from shuffle_ctc.factorized import factorized_graph_ctc_loss
from shuffle_ctc.loss import graph_ctc_loss


def modified_topology(symbols):
    symbols = sorted(set(symbols))
    final = len(symbols) + 1
    arcs = [f"0 0 0 0 0"]
    for state, symbol in enumerate(symbols, 1):
        arcs.append(f"0 {state} {symbol} {symbol} 0")
    arcs.append(f"0 {final} -1 -1 0")
    for state, _ in enumerate(symbols, 1):
        arcs += [f"{state} 0 0 0 0", f"{state} {final} -1 -1 0"]
    arcs.append(str(final))
    return k2.arc_sort(k2.Fsa.from_str("\n".join(arcs), acceptor=False))


class FactorizedIntersectDenseTest(unittest.TestCase):
    def _run(self, device):
        torch.manual_seed(1234)
        vocab, speakers, frames = 7, 3, 8
        token_logits = torch.randn(1, frames, vocab, device=device,
                                   requires_grad=True)
        speaker_logits = torch.randn(1, frames, speakers, device=device,
                                     requires_grad=True)
        token = token_logits.log_softmax(-1)
        speaker = speaker_logits.log_softmax(-1)
        joint = token[..., 1:].unsqueeze(-1) + speaker.unsqueeze(-2)
        joint = joint.permute(0, 1, 3, 2).flatten(2)
        expanded = torch.cat([token[..., :1], joint], -1)
        labels = list(range(1, 1 + (vocab - 1) * speakers))
        graph = k2.create_fsa_vec([modified_topology(labels)]).to(device)
        lengths = torch.tensor([frames], dtype=torch.int32)

        reference = graph_ctc_loss(
            expanded, graph, lengths, reduction="sum", output_beam=100,
            use_double_scores=True,
        )
        ref_grads = torch.autograd.grad(
            reference, (token_logits, speaker_logits), retain_graph=True
        )
        actual = factorized_graph_ctc_loss(
            token, speaker, graph, lengths, reduction="sum", output_beam=100,
            use_double_scores=True,
        )
        actual_grads = torch.autograd.grad(actual, (token_logits, speaker_logits))

        self.assertTrue(torch.allclose(actual, reference, atol=2e-5, rtol=2e-5),
                        (actual, reference))
        for got, expected in zip(actual_grads, ref_grads):
            self.assertTrue(torch.allclose(got, expected, atol=3e-5, rtol=3e-5),
                            (got - expected).abs().max())

    def test_cpu_matches_expanded_k2(self):
        self._run(torch.device("cpu"))

    def test_unsafe_fsavec_is_split_without_changing_loss_or_gradients(self):
        """Exercise the int32 preflight without constructing a huge graph."""
        torch.manual_seed(20260914)
        batch, frames, vocab, speakers = 2, 8, 7, 3
        token_logits = torch.randn(
            batch, frames, vocab, requires_grad=True
        )
        speaker_logits = torch.randn(
            batch, frames, speakers, requires_grad=True
        )
        token = token_logits.log_softmax(-1)
        speaker = speaker_logits.log_softmax(-1)
        labels = list(range(1, 1 + (vocab - 1) * speakers))
        graph = k2.create_fsa_vec(
            [modified_topology(labels) for _ in range(batch)]
        )
        lengths = torch.full((batch,), frames, dtype=torch.int32)

        reference = factorized_graph_ctc_loss(
            token, speaker, graph, lengths, reduction="mean", output_beam=100
        )
        reference_grads = torch.autograd.grad(
            reference, (token_logits, speaker_logits), retain_graph=True
        )
        old_limit = factorized_module._K2_DENSE_INT32_MAX
        try:
            # The combined FsaVec exceeds this artificial limit; either graph
            # on its own remains below it.
            states_per_graph = int(graph.arcs.shape().row_splits(1)[1].item())
            factorized_module._K2_DENSE_INT32_MAX = (
                (frames + 1) * states_per_graph + 2
            )
            split = factorized_graph_ctc_loss(
                token, speaker, graph, lengths,
                reduction="mean", output_beam=100,
            )
            split_grads = torch.autograd.grad(
                split, (token_logits, speaker_logits)
            )
        finally:
            factorized_module._K2_DENSE_INT32_MAX = old_limit

        self.assertTrue(torch.allclose(split, reference, atol=2e-5, rtol=2e-5))
        for got, expected in zip(split_grads, reference_grads):
            self.assertTrue(
                torch.allclose(got, expected, atol=3e-5, rtol=3e-5),
                (got - expected).abs().max(),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_expanded_k2(self):
        self._run(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_repeated_beam_pruning_matches_expanded_k2(self):
        """Force FormatOutput to retry with progressively smaller beams."""
        torch.manual_seed(8675309)
        device = torch.device("cuda")
        batch, frames, vocab, speakers = 3, 80, 16, 4
        token_logits = torch.randn(
            batch, frames, vocab, device=device, requires_grad=True
        )
        speaker_logits = torch.randn(
            batch, frames, speakers, device=device, requires_grad=True
        )
        token = token_logits.log_softmax(-1)
        speaker = speaker_logits.log_softmax(-1)
        joint = token[..., 1:].unsqueeze(-1) + speaker.unsqueeze(-2)
        expanded = torch.cat(
            [token[..., :1], joint.permute(0, 1, 3, 2).flatten(2)], -1
        )
        labels = list(range(1, 1 + (vocab - 1) * speakers))
        graph = k2.create_fsa_vec(
            [modified_topology(labels) for _ in range(batch)]
        ).to(device)
        lengths = torch.full((batch,), frames, dtype=torch.int32)
        kwargs = dict(
            reduction="sum", output_beam=30, max_states=1000,
            use_double_scores=True,
        )
        reference = graph_ctc_loss(expanded, graph, lengths, **kwargs)
        ref_grads = torch.autograd.grad(
            reference, (token_logits, speaker_logits), retain_graph=True
        )
        actual = factorized_graph_ctc_loss(
            token, speaker, graph, lengths, **kwargs
        )
        actual_grads = torch.autograd.grad(
            actual, (token_logits, speaker_logits)
        )
        self.assertTrue(
            torch.allclose(actual, reference, atol=3e-4, rtol=3e-4),
            (actual, reference),
        )
        for got, expected in zip(actual_grads, ref_grads):
            self.assertTrue(
                torch.allclose(got, expected, atol=5e-4, rtol=5e-4),
                (got - expected).abs().max(),
            )


if __name__ == "__main__":
    unittest.main()
