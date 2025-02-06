#!/usr/bin/env python

"""Selects subsets based on duration in hours."""


from lhotse.cut import CutSet


def compute_duration(cuts: CutSet) -> float:
    """Returns duration of the CutSet in hours."""
    return sum([c.duration for c in cuts]) / 3600.0


def select_subset(cuts: CutSet, duration: float, shuffle: bool = False) -> \
        CutSet:
    """Returns a subset of the specified CutSet of the specified duration."""
    num_cuts = sum(1 for c in cuts)
    if compute_duration(cuts) <= duration or num_cuts in [0, 1] or \
            duration == 0:
        return cuts

    if shuffle:
        cuts = cuts.shuffle()

    if cuts[0].duration / 3600.0 > duration:
        return cuts.subset(first=1)

    split_at = num_cuts // 2
    split_size = max(split_at // 2, 1)
    evaluated = set()

    while split_at and split_at not in evaluated:
        evaluated.add(split_at)
        if compute_duration(cuts.subset(first=split_at)) < duration:
            split_at += split_size
        else:
            split_at -= split_size
        split_size = max(split_size // 2, 1)

    split_at = min(split_at + split_size, num_cuts)
    return cuts.subset(first=split_at)
