#!/usr/bin/env python3
"""Regression tests for speaker-independent token shuffle constraints."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = ("wavlm_shuffle_ctc", "wavlm_spashl_ctc")


class FakeSentencePiece:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def encode(self, texts):
        return [self.transcripts[text] for text in texts]

    def vocab_size(self):
        return 16


def make_compiler(implementation):
    path = ROOT / implementation / "mdctc_graph_compiler.py"
    spec = importlib.util.spec_from_file_location(
        f"{implementation}_graph_compiler", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiler = module.MDCTCGraphCompiler.__new__(module.MDCTCGraphCompiler)
    compiler.device = "cpu"
    compiler.sp = FakeSentencePiece({"A": [1, 2], "B": [3, 4]})
    return compiler


def count_accepting_paths(graph):
    """Count paths in the acyclic graph without requiring an FsaVec."""
    paths_to_state = {0: 1}
    accepting_paths = 0
    for source, destination, label, _ in graph.arcs.values().tolist():
        paths = paths_to_state.get(source, 0)
        if label == -1:
            accepting_paths += paths
        else:
            paths_to_state[destination] = paths_to_state.get(destination, 0) + paths
    return accepting_paths


class TokenShuffleConstraintTest(unittest.TestCase):
    def test_brownian_bridge_width_shape(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                width = compiler.token_half_width
                self.assertEqual(width(0.0, 100, 20, "brownian_bridge"), 10)
                self.assertEqual(width(1.0, 100, 20, "brownian_bridge"), 10)
                self.assertEqual(width(0.5, 100, 20, "brownian_bridge"), 50)
                self.assertAlmostEqual(
                    width(0.25, 100, 20, "brownian_bridge"),
                    width(0.75, 100, 20, "brownian_bridge"),
                )

    def test_equal_bridge_collars_reproduce_fixed_constraints(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                fixed = compiler.get_seqs_and_constraints_no_spk(
                    ["A", "B"], [0, 75], [100, 100], collar=50
                )
                bridge = compiler.get_seqs_and_constraints_no_spk(
                    ["A", "B"], [0, 75], [100, 100], collar=50,
                    edge_collar=50, interval_model="brownian_bridge"
                )
                self.assertEqual(fixed, bridge)

    def test_speaker_pairs_are_counted_once(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                _, constraints, ambiguous, _ = compiler.get_seqs_and_constraints(
                    ["A", "B"], [0, 0], [100, 100], ["a", "b"],
                    {"a": 0, "b": 1}, collar=1000,
                    allow_self_overlap=True,
                )
                self.assertEqual(constraints, [])
                self.assertEqual(ambiguous, 4)

    def test_two_overlapping_two_token_utterances_have_six_shuffles(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                sequences, constraints, ambiguous, _ = (
                    compiler.get_seqs_and_constraints_no_spk(
                        ["A", "B"], [0, 0], [100, 100], collar=1000
                    )
                )

                self.assertEqual(constraints, [])
                self.assertEqual(ambiguous, 4)
                graph = compiler.build_topo_sort_fsa(
                    sequences, constraints, use_speaker=False
                )
                self.assertEqual(count_accepting_paths(graph), 6)

    def test_equal_timestamps_do_not_create_a_cycle(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                sequences, constraints, ambiguous, _ = (
                    compiler.get_seqs_and_constraints_no_spk(
                        ["A", "B"], [0, 0], [100, 100], collar=0
                    )
                )

                self.assertNotIn((sequences[1][0], sequences[0][0]), constraints)
                self.assertNotIn((sequences[0][0], sequences[1][0]), constraints)
                self.assertEqual(ambiguous, 2)
                graph = compiler.build_topo_sort_fsa(
                    sequences, constraints, use_speaker=False
                )
                self.assertGreater(count_accepting_paths(graph), 0)

    def test_tokens_outside_collar_receive_chronological_precedence(self):
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation):
                compiler = make_compiler(implementation)
                sequences, constraints, ambiguous, _ = (
                    compiler.get_seqs_and_constraints_no_spk(
                        ["A", "B"], [0, 1000], [100, 100], collar=100
                    )
                )

                expected = {(left, right) for left in sequences[0] for right in sequences[1]}
                self.assertEqual(set(constraints), expected)
                self.assertEqual(len(constraints), 4)
                self.assertEqual(ambiguous, 0)


if __name__ == "__main__":
    unittest.main()
