import sys
import unittest
from pathlib import Path

import torch


RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR / "wavlm_spashl_ctc"))

from model import split_joint_labels  # noqa: E402


class TestWavlmJointLabels(unittest.TestCase):
    def test_split_joint_labels_including_block_boundaries(self):
        vocab_size = 5
        labels = torch.tensor([1, 4, 5, 8, 9, 12])

        speakers, tokens = split_joint_labels(labels, vocab_size)

        self.assertEqual(speakers.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(tokens.tolist(), [1, 4, 1, 4, 1, 4])
        self.assertTrue(
            torch.equal(labels, tokens + speakers * (vocab_size - 1))
        )

    def test_split_joint_labels_rejects_blank(self):
        with self.assertRaisesRegex(ValueError, "nonblank"):
            split_joint_labels(torch.tensor([0, 1]), token_vocab_size=5)


if __name__ == "__main__":
    unittest.main()
