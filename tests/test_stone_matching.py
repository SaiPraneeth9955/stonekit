import numpy as np

from stonekit.eval.stone_matching import group_pieces, kind_of


def labels(*pieces, shape=(1, 30)):
    out = np.zeros(shape, dtype=np.uint16)
    for label, (start, stop) in pieces:
        out[0, start:stop] = label
    return out


def kinds(predicted, annotated):
    groups = group_pieces(predicted, annotated, set(np.unique(predicted)) - {0}, set(np.unique(annotated)) - {0})
    return sorted((g.kind, tuple(g.annotated), tuple(g.predicted)) for g in groups)


def test_every_relationship_by_hand():
    annotated = labels((1, (0, 3)), (2, (5, 9)), (3, (11, 13)), (4, (14, 16)), (5, (20, 22)))
    predicted = labels((1, (1, 3)), (2, (5, 6)), (3, (8, 9)), (4, (11, 16)), (5, (25, 27)))
    assert kinds(predicted, annotated) == [
        ("false_positive", (), (5,)),
        ("merged", (3, 4), (4,)),
        ("missed", (5,), ()),
        ("one_to_one", (1,), (1,)),
        ("split", (2,), (2, 3)),
    ]


def test_complex_group():
    annotated = labels((1, (0, 4)), (2, (6, 10)))
    predicted = labels((1, (2, 7)), (2, (8, 10)))
    assert kinds(predicted, annotated) == [("complex", (1, 2), (1, 2))]


def test_empty_masks_give_no_groups():
    empty = np.zeros((1, 5), dtype=np.uint16)
    assert kinds(empty, empty) == []


def test_kind_names():
    assert kind_of(1, 0) == "missed"
    assert kind_of(0, 1) == "false_positive"
    assert kind_of(2, 1) == "merged"
