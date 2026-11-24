import sys
from pathlib import Path

import torch


RECIPE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE_DIR / "wavlm_spashl_ctc"))

from align_token import collapse_max_speaker_scores  # noqa: E402


def test_collapse_max_speaker_scores_preserves_blank_and_scores_emissions():
    token = torch.tensor(
        [[[-0.1, -1.0, -2.0], [-0.2, -3.0, -4.0]]]
    )
    speaker = torch.tensor(
        [[[-0.7, -0.2, -0.01], [-0.4, -0.5, -0.001]]]
    )

    # Only the first two speaker slots are valid.  The attractive third slot
    # must not influence alignment for this two-speaker recording.
    actual = collapse_max_speaker_scores(token, speaker, [2])
    expected = torch.tensor(
        [[[-0.1, -1.2, -2.2], [-0.2, -3.4, -4.4]]]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(token[..., 0], actual[..., 0])
