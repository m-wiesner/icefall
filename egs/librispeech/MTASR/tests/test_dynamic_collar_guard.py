#!/usr/bin/env python3
"""Termination tests for dynamic collar reduction."""

import unittest

from wavlm_spashl_ctc.mdctc_graph_compiler import MDCTCGraphCompiler


class DynamicCollarGuardTest(unittest.TestCase):
    def test_speaker_graph_fails_when_zero_collar_cannot_meet_cap(self):
        compiler = object.__new__(MDCTCGraphCompiler)
        compiler.get_seqs_and_constraints = lambda *args, **kwargs: (
            [], [], 2, {}
        )
        with self.assertRaisesRegex(RuntimeError, "at zero collar"):
            compiler.compile(
                [["A", "B"]], [[0, 0]], [[16000, 16000]], [["s", "s"]],
                collar=1, max_overlaps=1, allow_self_overlap=True,
            )

    def test_no_speaker_graph_fails_when_zero_collar_cannot_meet_cap(self):
        compiler = object.__new__(MDCTCGraphCompiler)
        compiler.get_seqs_and_constraints_no_spk = lambda *args, **kwargs: (
            [], [], 2, {}
        )
        with self.assertRaisesRegex(RuntimeError, "at zero collar"):
            compiler.compile_nospk(
                [["A", "B"]], [[0, 0]], [[16000, 16000]],
                collar=1, max_overlaps=1,
            )


if __name__ == "__main__":
    unittest.main()
