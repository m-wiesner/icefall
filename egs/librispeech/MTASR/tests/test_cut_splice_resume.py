#!/usr/bin/env python3
"""Regression tests for curriculum generator discovery and resume state."""

import unittest

from lhotse import CutSet

from vendor.lhotse.cut_splice3 import (
    ApproximateResumeDynamicBucketingSampler,
    CutSpliceIterable,
)


class CutSpliceResumeTest(unittest.TestCase):
    def test_finds_generator_through_training_filter_wrapper(self):
        generator = CutSpliceIterable(
            [], seed=42, max_duration=20, final_max_duration=30
        )
        filtered = CutSet(generator).filter(lambda cut: True)
        sampler = ApproximateResumeDynamicBucketingSampler.__new__(
            ApproximateResumeDynamicBucketingSampler
        )
        sampler.cuts = (filtered,)
        self.assertIs(sampler._cut_splice_iterable(), generator)

    def test_checkpoint_state_through_training_filter_wrapper(self):
        generator = CutSpliceIterable([], seed=42)
        filtered = CutSet(generator).filter(lambda cut: True)
        sampler = ApproximateResumeDynamicBucketingSampler(
            filtered,
            max_duration=300,
            num_buckets=10,
            duration_bins=list(range(10, 100, 10)),
            buffer_size=100,
        )
        state = sampler.state_dict()
        self.assertEqual(state["resume_policy"], "fresh_buffer_rewind")
        self.assertEqual(
            state["cut_splice_curriculum"]["num_examples_generated"], 0
        )

    def test_buffer_rewind_is_clamped_at_zero(self):
        generator = CutSpliceIterable([], seed=42)
        generator.set_approximate_resume_state(
            {"num_examples_generated": 10}, buffer_size=20
        )
        self.assertEqual(generator.resume_generation_index, 0)


if __name__ == "__main__":
    unittest.main()
