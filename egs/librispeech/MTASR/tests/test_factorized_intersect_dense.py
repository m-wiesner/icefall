#!/usr/bin/env python3
"""Numerical equivalence with the expanded-logit k2 reference path."""

import unittest

import k2
import torch

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_matches_expanded_k2(self):
        self._run(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
