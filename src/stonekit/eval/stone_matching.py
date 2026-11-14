"""Group predicted and annotated stone pieces that overlap, and name each group's relationship.

Two pieces belong to one group when they share a voxel, directly or through other pieces. Groups:
- one_to_one: one annotated piece, one predicted piece (measurement agreement is computed only here)
- split: one annotated piece covered by several predicted pieces
- merged: several annotated pieces covered by one predicted piece
- complex: several of each
- missed: an annotated piece with no overlapping prediction
- false_positive: a predicted piece with no overlapping annotation
Annotations are references, not physical truth.
"""

from dataclasses import dataclass, field

from stonekit.eval.seg_metrics import overlap_pairs


@dataclass
class Group:
    kind: str
    annotated: list[int] = field(default_factory=list)
    predicted: list[int] = field(default_factory=list)


def kind_of(n_annotated: int, n_predicted: int) -> str:
    if n_annotated and not n_predicted:
        return "missed"
    if n_predicted and not n_annotated:
        return "false_positive"
    if n_annotated == 1 and n_predicted == 1:
        return "one_to_one"
    if n_annotated == 1:
        return "split"
    if n_predicted == 1:
        return "merged"
    return "complex"


def group_pieces(predicted_labels, annotated_labels, predicted_ids, annotated_ids) -> list[Group]:
    """Every listed piece appears in exactly one group."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in predicted_ids:
        parent[("p", int(i))] = ("p", int(i))
    for i in annotated_ids:
        parent[("a", int(i))] = ("a", int(i))
    for p, a in overlap_pairs(predicted_labels, annotated_labels):
        if ("p", p) in parent and ("a", a) in parent:
            parent[find(("p", p))] = find(("a", a))
    members: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in parent:
        members.setdefault(find(node), []).append(node)
    groups = []
    for nodes in members.values():
        annotated = sorted(i for side, i in nodes if side == "a")
        predicted = sorted(i for side, i in nodes if side == "p")
        groups.append(Group(kind_of(len(annotated), len(predicted)), annotated, predicted))
    return sorted(groups, key=lambda g: (g.annotated[:1] or [10**9], g.predicted[:1] or [10**9]))
