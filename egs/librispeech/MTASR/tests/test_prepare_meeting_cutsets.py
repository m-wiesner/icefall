from lhotse import SupervisionSegment
from lhotse.supervision import AlignmentItem

from local.prepare_ami_notsofar import _alignment_fragment


def test_boundary_words_are_assigned_exactly_once():
    supervision = SupervisionSegment(
        id="sup",
        recording_id="rec",
        start=29.0,
        duration=2.0,
        channel=0,
        speaker="speaker",
        text="hello, world!",
        alignment={
            "word": [
                AlignmentItem("hello,", 29.0, 0.8),
                AlignmentItem("world!", 30.0, 0.8),
            ]
        },
    )
    first = _alignment_fragment(supervision, 0.0, 30.0, False)
    second = _alignment_fragment(supervision, 30.0, 60.0, True)
    assert first[0] == "HELLO"
    assert second[0] == "WORLD"
    assert len(first[3]["word"]) + len(second[3]["word"]) == 2
