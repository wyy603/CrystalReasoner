"""Bucket scoring helper (from ``crysreas.metric.metrics.inseg``)."""

from __future__ import annotations


def inseg(x: list, a: list) -> list[int]:
    scores = []
    for v, r in zip(x, a):
        if r is None or len(r) < 2:
            scores.append(0)
            continue
        scores.append(int(v >= r[0] and v <= r[1]))
    return scores
