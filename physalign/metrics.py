"""Equations (4)-(9), (11): item events -> mother means -> type summary.

The pool and type weights are frozen from probe metadata, never from model
successes. Missing infrastructure outcomes are explicit None values; they do
not shrink the denominator. None in a returned metric means undefined.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import fsum, isclose, isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from .scoring import Probe, ProbeScore


def _multiplicities(probes: Sequence[Probe], counts: Mapping[str, int] | None) -> dict[str, int]:
    problems = {p.problem_id for p in probes}
    if counts is None:
        return dict.fromkeys(problems, 1)
    if not problems <= counts.keys():
        raise ValueError("Multiplicity must include every planned mother, including zeros")
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Mother multiplicities must be nonnegative integers")
    return {p: counts[p] for p in problems}


@dataclass(frozen=True)
class Pool:
    """One eligibility pool with its OWN predeclared type weights."""

    probes: tuple[Probe, ...]
    type_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        probes = tuple(self.probes)
        if len({p.probe_id for p in probes}) != len(probes):
            raise ValueError("Duplicate probe ID in pool")
        types = sorted({p.probe_type for p in probes})
        weights = ({t: 1 / len(types) for t in types} if self.type_weights is None
                   else dict(self.type_weights))
        if set(weights) != set(types):
            raise ValueError("Type weights must exactly cover the metadata-defined pool")
        if any(not isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Type weights must be finite and nonnegative")
        if types and not isclose(fsum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Type weights must sum to one; they are not auto-normalized")
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "type_weights", MappingProxyType(weights))

    def estimate(self, values: Mapping[str, float | None], *,
                 problem_multiplicities: Mapping[str, int] | None = None) -> dict:
        """Eq. (4), then type weighting. Bounds replace missing values by 0/1.

        Multiplicity implements a mother-cluster bootstrap WITHOUT collapsing
        repeated mothers or resampling their probes separately. Missing a required
        type makes the fixed-weight summary undefined; weights never change.
        """
        if set(values) != {p.probe_id for p in self.probes}:
            raise ValueError("Values must exactly match the frozen pool; use explicit None for missing")
        if any(v is not None and (not isfinite(v) or not 0 <= v <= 1) for v in values.values()):
            raise ValueError("Pool values must be finite scores in [0, 1], or None")
        counts = _multiplicities(self.probes, problem_multiplicities)
        groups = defaultdict(lambda: defaultdict(list))
        for p in self.probes:
            if counts[p.problem_id]:
                groups[p.probe_type][p.problem_id].append(values[p.probe_id])
        by_type = {}
        for t in self.type_weights:
            mothers = groups[t]
            n_problems = sum(counts[i] for i in mothers)
            n_probes = sum(counts[i] * len(v) for i, v in mothers.items())
            n_missing = sum(counts[i] * sum(x is None for x in v) for i, v in mothers.items())
            # First divide by m_it WITHIN each mother, then by N_t.
            lower = (fsum(counts[i] * fsum(x for x in v if x is not None) / len(v)
                          for i, v in mothers.items()) / n_problems if n_problems else None)
            upper = (fsum(counts[i] * fsum(1.0 if x is None else x for x in v) / len(v)
                          for i, v in mothers.items()) / n_problems if n_problems else None)
            coverage = (fsum(counts[i] * sum(x is not None for x in v) / len(v)
                             for i, v in mothers.items()) / n_problems if n_problems else None)
            by_type[t] = {
                "value": lower if not n_missing else None, "lower": lower, "upper": upper,
                "n_probes": n_probes, "n_problems": n_problems, "missing_probes": n_missing,
                "weighted_service_coverage": coverage,
            }

        def combine(key: str) -> float | None:
            # Even zero-weight required types keep their declared support contract.
            if not by_type or any(d[key] is None for d in by_type.values()):
                return None
            return fsum(self.type_weights[t] * d[key] for t, d in by_type.items())

        return {"value": combine("value"), "lower": combine("lower"), "upper": combine("upper"),
                "weighted_service_coverage": combine("weighted_service_coverage"), "by_type": by_type}


@dataclass(frozen=True)
class EvaluationPlan:
    """Build BEFORE looking at model outputs. All defaults are metadata-only."""

    probes: tuple[Probe, ...]
    binding_type_weights: Mapping[str, float] | None = None
    joint_type_weights: Mapping[str, float] | None = None
    packet_type_weights: Mapping[str, float] | None = None
    binding_only_type_weights: Mapping[str, float] | None = None
    binding_pool: Pool = field(init=False)
    joint_pool: Pool = field(init=False)
    packet_pool: Pool = field(init=False)
    binding_only_pool: Pool = field(init=False)

    def __post_init__(self) -> None:
        probes = tuple(self.probes)
        if not probes:
            raise ValueError("An evaluation plan needs at least one eligible binding probe")
        object.__setattr__(self, "probes", probes)
        for name, selected, weights_name in (
            ("binding_pool", probes, "binding_type_weights"),
            ("joint_pool", tuple(p for p in probes if p.joint_eligible), "joint_type_weights"),
            ("packet_pool", tuple(p for p in probes if p.packet_nonempty), "packet_type_weights"),
            ("binding_only_pool", tuple(p for p in probes if not p.joint_eligible), "binding_only_type_weights"),
        ):
            pool = Pool(selected, getattr(self, weights_name))
            object.__setattr__(self, name, pool)
            object.__setattr__(self, weights_name, pool.type_weights)


def _validate_scores(plan: EvaluationPlan, scores: Mapping[str, ProbeScore | None], condition: str) -> None:
    if set(scores) != {p.probe_id for p in plan.probes}:
        raise ValueError("Score IDs must exactly match the plan; missing outcomes must be explicit None")
    for p in plan.probes:
        score = scores[p.probe_id]
        if score is not None and (score.probe != p or score.condition != condition):
            raise ValueError(f"Probe specification/condition mismatch for {p.probe_id}")


def _values(pool: Pool, scores: Mapping[str, ProbeScore | None], metric: str) -> dict:
    return {p.probe_id: None if scores[p.probe_id] is None else getattr(scores[p.probe_id], metric)
            for p in pool.probes}


def _support(pool: Pool, scores: Mapping[str, ProbeScore | None], counts: Mapping[str, int],
             *, recognition: bool = False) -> dict:
    probes = [p for p in pool.probes if counts[p.problem_id]]
    total = sum(counts[p.problem_id] for p in probes)
    served = [p for p in probes if scores[p.probe_id] is not None]
    n_served = sum(counts[p.problem_id] for p in served)
    result = {
        "planned_probes": total,
        "planned_problems": sum(counts[i] for i in {p.problem_id for p in probes}),
        "served_probes": n_served,
        "served_problems": sum(counts[i] for i in {p.problem_id for p in served}),
        "missing_probes": total - n_served,
        "service_coverage": n_served / total if total else None,
    }
    if recognition:
        correct = [p for p in served if scores[p.probe_id].C == 1]
        n_correct = sum(counts[p.problem_id] for p in correct)
        result.update(correctly_read_probes=n_correct,
                      correctly_read_problems=sum(counts[i] for i in {p.problem_id for p in correct}),
                      unweighted_recognition_coverage=n_correct / total if total == n_served and total else None)
    return result


def _invalidity(pool: Pool, scores: Mapping[str, ProbeScore | None], counts: Mapping[str, int]) -> dict:
    served = [p for p in pool.probes if counts[p.problem_id] and scores[p.probe_id] is not None]
    n_served = sum(counts[p.problem_id] for p in served)
    invalid_objects = sum(counts[p.problem_id] for p in served if not scores[p.probe_id].object_valid)
    fields = defaultdict(lambda: [0, 0])
    for p in served:
        for key, valid in scores[p.probe_id].field_valid.items():
            fields[key][0] += counts[p.problem_id]
            fields[key][1] += counts[p.problem_id] * int(not valid)
    object_estimate = pool.estimate({p.probe_id: None if scores[p.probe_id] is None
                                  else int(not scores[p.probe_id].object_valid) for p in pool.probes},
                                  problem_multiplicities=counts)
    return {"served_outputs": n_served, "invalid_objects": invalid_objects,
            "object_invalid_rate_served": invalid_objects / n_served if n_served else None,
            "weighted_object_invalidity": object_estimate,
            "fields": {key: {"required_served_fields": n, "invalid_fields": bad,
                             "invalid_rate_served": bad / n} for key, (n, bad) in fields.items()}}


def _joint_diagnostics(c: float | None, b: float | None, j: float | None) -> dict:
    ratio = j / c if c is not None and j is not None and c > 0 else None
    quadrants = {"q11": None, "q10": None, "q01": None, "q00": None}
    if c is not None and b is not None and j is not None:
        quadrants = {"q11": j, "q10": c - j, "q01": b - j, "q00": 1 - c - b + j}
        if any(v < -1e-12 for v in quadrants.values()) or not isclose(fsum(quadrants.values()), 1, abs_tol=1e-12):
            raise ArithmeticError("Joint quadrant invariant failed")
        quadrants = {key: max(0.0, value) for key, value in quadrants.items()}
    return {"BAcc_given_C": ratio, "BindErr_given_C": None if ratio is None else 1 - ratio,
            "quadrants": quadrants}


def evaluate(plan: EvaluationPlan, raw: Mapping[str, ProbeScore | None],
             gold: Mapping[str, ProbeScore | None] | None = None, *,
             problem_multiplicities: Mapping[str, int] | None = None) -> dict:
    """Four primary scores and matched diagnostics. Scores are fractions, not %.

    gold=None means the gold experiment was not supplied. Otherwise both mappings
    must cover the SAME planned probes; an explicit None item is an infrastructure
    missing response. Headline scores with missing support are withheld, with
    fixed-denominator lower/upper bounds. Wrong or malformed outputs remain zeros.
    """
    _validate_scores(plan, raw, "raw")
    if gold is not None:
        _validate_scores(plan, gold, "gold")
    counts = _multiplicities(plan.probes, problem_multiplicities)
    B, L = plan.binding_pool, plan.joint_pool

    def estimate(pool: Pool, scores: Mapping, metric: str) -> dict:
        return pool.estimate(_values(pool, scores, metric), problem_multiplicities=counts)

    summaries = {"CAcc": estimate(L, raw, "C"), "BAcc": estimate(B, raw, "B"),
                 "JAcc": estimate(L, raw, "J"), "BAcc_L": estimate(L, raw, "B")}
    if gold is not None:
        summaries["BAcc_gold"] = estimate(B, gold, "B")
    metrics = {key: value["value"] for key, value in summaries.items()}
    metrics.setdefault("BAcc_gold", None)
    metrics["delta_BAcc_gold_raw"] = (metrics["BAcc_gold"] - metrics["BAcc"]
                                      if metrics["BAcc_gold"] is not None and metrics["BAcc"] is not None else None)
    joint = _joint_diagnostics(metrics["CAcc"], metrics["BAcc_L"], metrics["JAcc"])
    metrics.update({k: joint[k] for k in ("BAcc_given_C", "BindErr_given_C")})
    by_type = {}
    for t in B.type_weights:
        row = {name: summary["by_type"].get(t, {}).get("value") for name, summary in summaries.items()}
        row.setdefault("BAcc_gold", None)
        row.update(_joint_diagnostics(row.get("CAcc"), row.get("BAcc_L"), row.get("JAcc")))
        row["delta_BAcc_gold_raw"] = (row["BAcc_gold"] - row["BAcc"]
                                     if row["BAcc_gold"] is not None and row["BAcc"] is not None else None)
        type_binding = Pool(tuple(p for p in B.probes if p.probe_type == t))
        type_joint = Pool(tuple(p for p in L.probes if p.probe_type == t))
        row["support"] = {"binding_raw": _support(type_binding, raw, counts),
                          "joint_raw": _support(type_joint, raw, counts, recognition=True)}
        if gold is not None:
            row["support"]["binding_gold"] = _support(type_binding, gold, counts)
        by_type[t] = row
    packet = {"type_weights": dict(plan.packet_pool.type_weights),
              "support": _support(plan.packet_pool, raw, counts),
              "BAcc": estimate(plan.packet_pool, raw, "B"),
              "BAcc_gold": estimate(plan.packet_pool, gold, "B") if gold is not None else None}
    packet["delta_BAcc_gold_raw"] = (packet["BAcc_gold"]["value"] - packet["BAcc"]["value"]
                                    if packet["BAcc_gold"] is not None and packet["BAcc_gold"]["value"] is not None
                                    and packet["BAcc"]["value"] is not None else None)
    packet["weighted_packet_coverage"] = B.estimate(
        {p.probe_id: int(p.packet_nonempty) for p in B.probes}, problem_multiplicities=counts)["value"]
    delta_bounds = None
    if gold is not None:
        lo, hi = summaries["BAcc"], summaries["BAcc_gold"]
        if lo["lower"] is not None and hi["lower"] is not None:
            delta_bounds = {"lower": hi["lower"] - lo["upper"], "upper": hi["upper"] - lo["lower"]}
    support = {"binding_raw": _support(B, raw, counts),
               "joint_raw": _support(L, raw, counts, recognition=True)}
    invalidity = {"raw": _invalidity(B, raw, counts)}
    if gold is not None:
        support["binding_gold"] = _support(B, gold, counts)
        invalidity["gold"] = _invalidity(B, gold, counts)
    return {"metrics": metrics, "by_type": by_type, "quadrants": joint["quadrants"],
            "summaries": summaries, "delta_BAcc_bounds": delta_bounds, "support": support,
            "type_weights": {"binding": dict(B.type_weights), "joint": dict(L.type_weights)},
            "nonempty_packets": packet, "invalidity": invalidity,
            "gold_status": "not_supplied" if gold is None else "supplied"}


def uniform_candidate_baseline(pool: Pool, *,
                               problem_multiplicities: Mapping[str, int] | None = None) -> dict:
    """Aggregate item-level chance, never 1 / mean(K).

    Uniform sampling is over public aliases. If approved canonical identities
    make several aliases correct, all of them count in the exact probability.
    Sets/relations require a separately declared random distribution.
    """
    probabilities = {}
    for p in pool.probes:
        if p.binding.kind != "single":
            raise ValueError("Uniform single-target baseline does not define a set/relation distribution")
        domain = next(iter(p.binding.domains.values()))
        probabilities[p.probe_id] = sum(c == p.binding.gold[0] for c in domain.values()) / len(domain)
    return pool.estimate(probabilities, problem_multiplicities=problem_multiplicities)


def solving_association(plan: EvaluationPlan, raw: Mapping[str, ProbeScore | None],
                        answers: Mapping[str, int], *,
                        problem_multiplicities: Mapping[str, int] | None = None) -> dict:
    """Eq. (8): b_i is the plain mean of ALL binding probes in mother i.

    answers holds independently graded original-problem correctness, not model
    solution text. SolveAcc uses every supplied answer; association uses the
    mother intersection, disclosed explicitly. Missing probe service is rejected
    instead of silently deleting hard mothers or changing each b_i denominator.
    """
    _validate_scores(plan, raw, "raw")
    if any(type(a) is not int or a not in (0, 1) for a in answers.values()):
        raise ValueError("Original answer correctness must be binary")
    problems = {p.problem_id for p in plan.probes} | set(answers)
    counts = dict.fromkeys(problems, 1) if problem_multiplicities is None else dict(problem_multiplicities)
    if not problems <= counts.keys() or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Multiplicity must cover probe and answer mother IDs")
    by_mother = defaultdict(list)
    for p in plan.probes:
        if p.problem_id in answers and counts[p.problem_id]:
            if raw[p.probe_id] is None:
                raise ValueError("Solving association requires complete binding service on eligible mothers")
            by_mother[p.problem_id].append(raw[p.probe_id].B)
    binding = {i: fsum(values) / len(values) for i, values in by_mother.items()}
    group_means, group_counts = {}, {}
    for label, correct in (("correct", 1), ("wrong", 0)):
        group = [i for i in binding if answers[i] == correct]
        n = sum(counts[i] for i in group)
        group_counts[label] = n
        group_means[label] = fsum(counts[i] * binding[i] for i in group) / n if n else None
    n_answers = sum(counts[i] for i in answers)
    return {"SolveAcc": fsum(counts[i] * a for i, a in answers.items()) / n_answers if n_answers else None,
            "answer_problems": n_answers, "association_problems": sum(counts[i] for i in binding),
            "binding_by_problem": binding, "group_counts": group_counts,
            "mean_binding_correct": group_means["correct"], "mean_binding_wrong": group_means["wrong"],
            "delta_assoc": group_means["correct"] - group_means["wrong"]
            if all(v is not None for v in group_means.values()) else None}


def option_diagnostic(plan: EvaluationPlan, without: Mapping[str, ProbeScore | None],
                      with_options: Mapping[str, ProbeScore | None], *,
                      problem_multiplicities: Mapping[str, int] | None = None) -> dict:
    """Eq. (9): form harm/repair PER ITEM before ANY averaging.

    Score both experimental conditions with condition='raw' (no gold reading).
    Joint correctness on L and B on the binding-only pool remain separate.
    """
    _validate_scores(plan, without, "raw")
    _validate_scores(plan, with_options, "raw")
    counts = _multiplicities(plan.probes, problem_multiplicities)
    result = {}
    for name, pool, metric in (("joint", plan.joint_pool, "J"),
                               ("binding_only", plan.binding_only_pool, "B")):
        before, after = _values(pool, without, metric), _values(pool, with_options, metric)
        if any(counts[p.problem_id] and (before[p.probe_id] is None or after[p.probe_id] is None)
               for p in pool.probes):
            raise ValueError("Option transitions require complete paired service on the declared pool")
        events = {key: {} for key in ("H", "R", "unchanged_correct", "unchanged_wrong", "accuracy_without", "accuracy_with")}
        for p in pool.probes:
            a, b = before[p.probe_id], after[p.probe_id]
            a, b = (0, 0) if not counts[p.problem_id] else (a, b)
            for key, value in zip(events, (a * (1 - b), (1 - a) * b, a * b,
                                           (1 - a) * (1 - b), a, b)):
                events[key][p.probe_id] = value
        summaries = {key: pool.estimate(values, problem_multiplicities=counts) for key, values in events.items()}
        scores = {key: summary["value"] for key, summary in summaries.items()}
        scores["net_change"] = scores["R"] - scores["H"] if scores["R"] is not None else None
        result[name] = {"metrics": scores, "summaries": summaries,
                        "type_weights": dict(pool.type_weights), "support": _support(pool, without, counts)}
    return result
