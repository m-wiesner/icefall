#!/usr/bin/env python3

import unittest

import k2
import torch

from shuffle_ctc.viterbi import (
    factorized_viterbi,
    factorized_viterbi_batched,
    token_viterbi_batched,
)


def graph():
    # Two legal orders: speaker 0/token 1 then speaker 1/token 2, or vice versa.
    # Joint labels for V=4 are (s0,t1)=1 and (s1,t2)=5.
    text = "\n".join([
        "0 0 0 0 0",
        "0 1 1 1 0",
        "0 2 5 2 0",
        "1 1 0 0 0",
        "1 3 5 2 0",
        "2 2 0 0 0",
        "2 3 1 1 0",
        "3 3 0 0 0",
        "3 4 -1 -1 0",
        "4",
    ])
    return k2.create_fsa_vec([
        k2.arc_sort(k2.Fsa.from_str(text, acceptor=False))
    ])


def token_graph():
    text = "\n".join([
        "0 0 0 0 0", "0 1 1 1 0", "0 2 2 2 0",
        "1 1 0 0 0", "1 3 2 2 0", "2 2 0 0 0",
        "2 3 1 1 0", "3 3 0 0 0", "3 4 -1 -1 0", "4",
    ])
    return k2.arc_sort(k2.Fsa.from_str(text, acceptor=False))


def repeated_token_segment_graph():
    # The identical token occurs in two distinct source segments.  Aux labels
    # 1 and 2 are what make occurrence-specific time gating possible.
    text = "\n".join([
        "0 0 0 0 0",
        "0 1 1 1 0",
        "1 1 0 0 0",
        "1 2 1 2 0",
        "2 2 0 0 0",
        "2 3 -1 -1 0",
        "3",
    ])
    return k2.create_fsa_vec([
        k2.arc_sort(k2.Fsa.from_str(text, acceptor=False))
    ])


class FactorizedViterbiTest(unittest.TestCase):
    def _run(self, device):
        torch.manual_seed(20260903)
        token = torch.randn(1, 7, 4, device=device).log_softmax(-1)
        speaker = torch.randn(1, 7, 2, device=device).log_softmax(-1)
        g = graph().to(device)
        actual = factorized_viterbi(g, token[0], speaker[0])

        joint = token[..., 1:].unsqueeze(-1) + speaker.unsqueeze(-2)
        expanded = torch.cat([token[..., :1], joint.permute(0, 1, 3, 2).flatten(2)], -1)
        dense = k2.DenseFsaVec(expanded, torch.tensor([[0, 0, 7]], dtype=torch.int32))
        lattice = k2.intersect_dense(g, dense, output_beam=100)
        best = k2.shortest_path(lattice, use_double_scores=True)
        keep = best.labels > 0
        self.assertTrue(torch.equal(actual.labels, best.labels[keep]))
        self.assertTrue(torch.equal(actual.aux_labels, best.aux_labels[keep]))
        self.assertAlmostEqual(
            actual.score.item(),
            best.get_tot_scores(log_semiring=False, use_double_scores=True).item(),
            places=5,
        )

        # A hard joint mask can force the opposite legal ordering without
        # expanding joint floating-point logits inside the Viterbi routine.
        allowed = torch.ones((7, 7), dtype=torch.bool, device=device)
        allowed[:3, 5] = False
        masked = factorized_viterbi(
            g, token[0], speaker[0], joint_allowed=allowed
        )
        self.assertTrue(torch.isfinite(masked.score))
        self.assertEqual(masked.labels.numel(), 2)

        pruned = factorized_viterbi(
            g, token[0], speaker[0], search_beam=100.0,
            min_active_states=0, max_active_states=1000,
        )
        self.assertTrue(torch.equal(pruned.labels, actual.labels))
        self.assertTrue(torch.equal(pruned.frames, actual.frames))
        self.assertAlmostEqual(pruned.score.item(), actual.score.item(), places=5)
        self.assertEqual(pruned.active_state_counts.numel(), token.size(1))

        # Shuffle-only alignment deliberately ignores the speaker branch.
        token_only = factorized_viterbi(
            g, token[0], speaker[0], score_speakers=False
        )
        reversed_speaker = factorized_viterbi(
            g, token[0], speaker[0].flip(-1), score_speakers=False
        )
        self.assertTrue(torch.equal(token_only.labels, reversed_speaker.labels))
        self.assertTrue(torch.equal(token_only.frames, reversed_speaker.frames))
        self.assertAlmostEqual(
            token_only.score.item(), reversed_speaker.score.item(), places=5
        )

        batched_graph = k2.create_fsa_vec([token_graph(), token_graph()]).to(device)
        batched_scores = token[0].unsqueeze(0).expand(2, -1, -1).contiguous()
        batched = token_viterbi_batched(batched_graph, batched_scores)
        expected_token = factorized_viterbi(
            k2.create_fsa_vec([token_graph()]).to(device), token[0], speaker[0],
            score_speakers=False,
        )
        for result in batched:
            self.assertTrue(torch.equal(result.labels, expected_token.labels))
            self.assertTrue(torch.equal(result.frames, expected_token.frames))
            self.assertAlmostEqual(result.score.item(), expected_token.score.item(), places=5)

        # The general batched implementation supports attributed graphs and
        # independently stops each member at its unpadded acoustic length.
        second_length = 5
        joint_graph = graph()[0]
        attributed_graphs = k2.create_fsa_vec(
            [joint_graph, joint_graph]
        ).to(device)
        token_batch = token.expand(2, -1, -1).contiguous()
        speaker_batch = speaker.expand(2, -1, -1).contiguous()
        lengths = torch.tensor([7, second_length], device=device)
        attributed_batch = factorized_viterbi_batched(
            attributed_graphs, token_batch, speaker_batch, lengths
        )
        expected = [
            factorized_viterbi(graph().to(device), token[0], speaker[0]),
            factorized_viterbi(
                graph().to(device), token[0, :second_length],
                speaker[0, :second_length],
            ),
        ]
        for actual_item, expected_item in zip(attributed_batch, expected):
            self.assertTrue(torch.equal(actual_item.labels, expected_item.labels))
            self.assertTrue(torch.equal(actual_item.aux_labels, expected_item.aux_labels))
            self.assertTrue(torch.equal(actual_item.frames, expected_item.frames))
            self.assertAlmostEqual(
                actual_item.score.item(), expected_item.score.item(), places=5
            )

        # Token-only batches ignore speaker probabilities while using the
        # same graph-vector and variable-length machinery.
        token_graphs = k2.create_fsa_vec([token_graph(), token_graph()]).to(device)
        token_batch_results = factorized_viterbi_batched(
            token_graphs, token_batch, speaker_batch, lengths,
            score_speakers=False,
        )
        for index, actual_item in enumerate(token_batch_results):
            expected_item = factorized_viterbi(
                k2.create_fsa_vec([token_graph()]).to(device),
                token[0, : int(lengths[index])],
                speaker[0, : int(lengths[index])],
                score_speakers=False,
            )
            self.assertTrue(torch.equal(actual_item.labels, expected_item.labels))
            self.assertTrue(torch.equal(actual_item.frames, expected_item.frames))
            self.assertAlmostEqual(
                actual_item.score.item(), expected_item.score.item(), places=5
            )

        # A hard segment window distinguishes repeated instances of the same
        # token. Blank remains legal outside each segment's window.
        repeated_graph = repeated_token_segment_graph().to(device)
        repeated_token = torch.tensor(
            [[[-3.0, -0.1], [-3.0, -0.1], [-3.0, -0.1],
              [-3.0, -0.1], [-3.0, -0.1], [-3.0, -0.1]]],
            device=device,
        )
        repeated_speaker = torch.zeros((1, 6, 1), device=device)
        windows = torch.tensor([[1, 1], [4, 4]], device=device)
        windowed = factorized_viterbi(
            repeated_graph,
            repeated_token[0],
            repeated_speaker[0],
            segment_windows=windows,
        )
        self.assertTrue(torch.equal(
            windowed.frames, torch.tensor([1, 4], device=device)
        ))
        self.assertTrue(torch.equal(
            windowed.aux_labels, torch.tensor([1, 2], device=device)
        ))

        # After blank removal the gate uses original, not compacted, frame
        # indexes, while traceback continues to return compacted indexes for
        # the caller's existing timestamp restoration code.
        compact_token = repeated_token[:, :3]
        compact_speaker = repeated_speaker[:, :3]
        original_frames = torch.tensor([0, 4, 7], device=device)
        compact_windows = torch.tensor([[4, 4], [7, 7]], device=device)
        compacted = factorized_viterbi(
            repeated_graph,
            compact_token[0],
            compact_speaker[0],
            segment_windows=compact_windows,
            original_frame_indices=original_frames,
        )
        self.assertTrue(torch.equal(
            compacted.frames, torch.tensor([1, 2], device=device)
        ))

        repeated_graphs = k2.create_fsa_vec([
            repeated_token_segment_graph()[0],
            repeated_token_segment_graph()[0],
        ]).to(device)
        batched_windowed = factorized_viterbi_batched(
            repeated_graphs,
            repeated_token.expand(2, -1, -1).contiguous(),
            repeated_speaker.expand(2, -1, -1).contiguous(),
            torch.tensor([6, 6], device=device),
            segment_windows=windows.unsqueeze(0).expand(2, -1, -1),
        )
        for result in batched_windowed:
            self.assertTrue(torch.equal(
                result.frames, torch.tensor([1, 4], device=device)
            ))
            self.assertTrue(torch.equal(
                result.aux_labels, torch.tensor([1, 2], device=device)
            ))

    def test_cpu(self):
        self._run(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        self._run(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
