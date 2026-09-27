"""Paired mother/source-cluster percentile bootstrap (proposal Appendix E.2)."""

from __future__ import annotations

from collections import Counter
from math import floor, isfinite
import random
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    left = floor(position)
    right = min(left + 1, len(sorted_values) - 1)
    return sorted_values[left] + (position - left) * (sorted_values[right] - sorted_values[left])


def paired_cluster_bootstrap(
    problem_ids: Sequence[str],
    statistic: Callable[[Mapping[str, int]], Mapping[str, float | None]],
    *, n_resamples: int = 2000, seed: int = 2027, confidence: float = 0.95,
    cluster_ids: Mapping[str, str] | None = None,
) -> dict:
    """Recompute full statistics with one shared mother multiplicity map.

    The callback must use the SAME frozen EvaluationPlan for every resample.
    Use evaluate(..., problem_multiplicities=counts), option_diagnostic(...),
    or solving_association(...). For model differences compute both models in
    this callback with the same counts, then subtract their statistics.

    cluster_ids can group near-duplicate mothers into larger source clusters.
    Any undefined replicate withholds that statistic's interval; invalid draws
    are counted and never silently discarded or renormalized.
    """
    problems = tuple(sorted(problem_ids))
    if not problems or len(problems) != len(set(problems)):
        raise ValueError("Pass distinct mother IDs, not one ID per probe")
    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if type(seed) is not int or not 0 < confidence < 1:
        raise ValueError("Record an integer seed and confidence in (0,1)")
    groups = dict(zip(problems, problems)) if cluster_ids is None else dict(cluster_ids)
    if set(groups) != set(problems) or any(not isinstance(g, str) or not g for g in groups.values()):
        raise ValueError("Cluster mapping must exactly cover mothers with nonempty IDs")
    clusters = sorted(set(groups.values()))
    point = dict(statistic(MappingProxyType(dict.fromkeys(problems, 1))))
    if not point:
        raise ValueError("Statistic callback must return at least one named statistic")

    def validate(values: Mapping) -> None:
        if set(values) != set(point):
            raise ValueError("Statistic names changed across bootstrap draws")
        if any(v is not None and not isfinite(v) for v in values.values()):
            raise ValueError("Use None, not NaN or infinity, for undefined statistics")

    validate(point)
    draws = {key: [] for key in point}
    invalid = dict.fromkeys(point, 0)
    rng = random.Random(seed)
    for _ in range(n_resamples):
        sampled = Counter(rng.choice(clusters) for _ in clusters)
        counts = MappingProxyType({p: sampled[groups[p]] for p in problems})
        values = statistic(counts)
        validate(values)
        for key, value in values.items():
            if value is None:
                invalid[key] += 1
            else:
                draws[key].append(value)
    tail = (1 - confidence) / 2
    intervals = {}
    for key in point:
        samples = sorted(draws[key])
        usable = point[key] is not None and invalid[key] == 0
        intervals[key] = {"estimate": point[key],
                          "lower": _percentile(samples, tail) if usable else None,
                          "upper": _percentile(samples, 1 - tail) if usable else None,
                          "valid_resamples": len(samples), "undefined_resamples": invalid[key],
                          "status": "ok" if usable else "withheld_undefined_statistic"}
    return {"intervals": intervals, "n_resamples": n_resamples, "seed": seed,
            "confidence": confidence, "n_clusters": len(clusters),
            "method": "paired_cluster_percentile_linear_interpolation"}
