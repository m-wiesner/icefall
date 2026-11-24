from types import SimpleNamespace

from local.rechunk_supervision_groups import overlap_components, pack_components


def _sup(start, duration):
    return SimpleNamespace(start=start, end=start + duration)


def test_overlap_components_are_never_split():
    components = overlap_components(
        [_sup(0, 10), _sup(9, 10), _sup(20, 2), _sup(25, 2)]
    )
    assert [len(c) for c in components] == [2, 1, 1]
    groups = list(pack_components(components, max_duration=22, max_pause=4))
    assert [len(g) for g in groups] == [3, 1]
