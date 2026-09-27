"""Item scores: proposal equations (2), (3), (10) and Appendix D.

Only frozen, approved probe specifications belong here. This module does not
infer eligibility, repair annotations, or judge a free-form physics solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .ocr_normalizers import NORMALIZER_IDS, normalize


class ReadingRule(Protocol):
    def normalize(self, value: Any) -> Any: ...
    def equivalent(self, prediction: Any, gold: Any) -> bool: ...


@dataclass(frozen=True)
class OCRRule:
    """Use the sample package's byte-identical frozen OCR implementation."""

    normalizer_id: str = "ocr_label_v2_1"

    def __post_init__(self) -> None:
        if self.normalizer_id not in NORMALIZER_IDS:
            raise ValueError(f"Unknown OCR normalizer: {self.normalizer_id}")

    def normalize(self, value: Any) -> str | None:
        return normalize(value, self.normalizer_id)

    def equivalent(self, prediction: str, gold: str) -> bool:
        return prediction == gold


@dataclass(frozen=True)
class QuantityRule:
    """Explicit quantity-comparison task, NEVER an OCR relaxation.

    Declare compatible units and their factors into one common reference unit.
    Factors and absolute tolerance use exact rational arithmetic. For example,
    {"kg": "1", "g": "0.001"}. No inferred dimensions, symbolic evaluation,
    case folding, or implicit numerical tolerance is provided.
    """

    unit_scales: Mapping[str, str]
    absolute_tolerance: str = "0"
    _scales: Mapping[str, Fraction] = field(init=False, repr=False)
    _tolerance: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        scales = {unit: Fraction(scale) for unit, scale in self.unit_scales.items()}
        if not scales or any(not isinstance(u, str) or not u.strip() or u != u.strip()
                             or scale <= 0 for u, scale in scales.items()):
            raise ValueError("Quantity units must be nonempty, with positive scales")
        tolerance = Fraction(self.absolute_tolerance)
        if tolerance < 0:
            raise ValueError("Quantity tolerance must be nonnegative")
        object.__setattr__(self, "unit_scales", MappingProxyType(dict(self.unit_scales)))
        object.__setattr__(self, "_scales", MappingProxyType(scales))
        object.__setattr__(self, "_tolerance", tolerance)

    def normalize(self, value: Any) -> Fraction | None:
        if not isinstance(value, str):
            return None
        match = re.fullmatch(
            r"([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*(\S+)",
            value.strip(),
        )
        if not match or match[2] not in self._scales:
            return None
        return Fraction(match[1]) * self._scales[match[2]]

    def equivalent(self, prediction: Fraction, gold: Fraction) -> bool:
        return abs(prediction - gold) <= self._tolerance


@dataclass(frozen=True)
class ReadTarget:
    """One output field tied to one fixed visual/textual evidence anchor."""

    field: str
    anchor_id: str
    expected: str
    rule: ReadingRule = field(default_factory=OCRRule)

    def __post_init__(self) -> None:
        if not self.field or not self.anchor_id:
            raise ValueError("Each readable field needs an output key and an anchor")
        if self.rule.normalize(self.expected) is None:
            raise ValueError(f"Unsupported gold reading at {self.anchor_id}")


@dataclass(frozen=True)
class BindingTarget:
    """Per-field alias domains and approved canonical targets.

    kind='single' or 'set': exactly one output field/domain.
    kind='relation': three fields/domains, ordered subject, predicate, object.
    Relation equivalences must already be encoded in the canonical maps.
    """

    kind: str
    domains: Mapping[str, Mapping[str, str]]
    gold: tuple[str, ...]
    symmetric: bool = False
    field_order: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in {"single", "set", "relation"}:
            raise ValueError("Binding kind must be single, set or relation")
        domains = {}
        for key, candidates in self.domains.items():
            if not key or not candidates:
                raise ValueError("Binding fields need nonempty candidate domains")
            for alias, canonical in candidates.items():
                if not isinstance(alias, str) or not alias or alias.strip() != alias:
                    raise ValueError("Candidate aliases must be nonempty and trimmed")
                if not isinstance(canonical, str) or not canonical:
                    raise ValueError("Canonical targets must be nonempty strings")
            domains[key] = MappingProxyType(dict(candidates))
        gold = tuple(self.gold)
        if not gold or any(not isinstance(g, str) or not g for g in gold):
            raise ValueError("A binding target must have nonempty canonical gold")
        if self.kind == "relation":
            if len(domains) != 3 or len(gold) != 3:
                raise ValueError("Relations require three ordered fields and targets")
            if any(g not in d.values() for g, d in zip(gold, domains.values())):
                raise ValueError("A gold relation endpoint/predicate is absent")
        else:
            if len(domains) != 1 or (self.kind == "single" and len(gold) != 1):
                raise ValueError("Single/set bindings require one field")
            if not set(gold) <= set(next(iter(domains.values())).values()):
                raise ValueError("Every gold target must occur in the candidate domain")
            if self.symmetric:
                raise ValueError("Only explicit relation tasks may be symmetric")
            if self.kind == "set":
                gold = tuple(sorted(set(gold)))
        object.__setattr__(self, "domains", MappingProxyType(domains))
        object.__setattr__(self, "gold", gold)
        # Dict equality ignores insertion order, but relation field order is
        # semantic. Include it explicitly in raw/gold specification equality.
        object.__setattr__(self, "field_order", tuple(domains))


@dataclass(frozen=True)
class Probe:
    probe_id: str
    problem_id: str
    probe_type: str
    binding: BindingTarget
    readings: tuple[ReadTarget, ...] = ()
    packet_nonempty: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(s, str) or not s for s in
               (self.probe_id, self.problem_id, self.probe_type)):
            raise ValueError("Probe, problem and type IDs must be nonempty strings")
        readings = tuple(self.readings)
        keys = [r.field for r in readings] + list(self.binding.domains)
        if len(keys) != len(set(keys)):
            raise ValueError("Reading and binding output fields must be distinct")
        if type(self.packet_nonempty) is not bool:
            raise ValueError("packet_nonempty must be an audited boolean")
        object.__setattr__(self, "readings", readings)

    @property
    def joint_eligible(self) -> bool:
        return bool(self.readings)


@dataclass(frozen=True)
class ProbeScore:
    probe: Probe
    condition: str
    C: int | None
    B: int
    object_valid: bool
    field_valid: Mapping[str, bool]
    canonical_binding: tuple[str, ...] | None = None
    set_f1: float | None = None

    def __post_init__(self) -> None:
        if self.condition not in {"raw", "gold"}:
            raise ValueError("Condition must be raw or gold")
        requires_content = self.condition == "raw" and self.probe.joint_eligible
        if requires_content and (type(self.C) is not int or self.C not in (0, 1)):
            raise ValueError("Raw joint scores require binary C")
        if not requires_content and self.C is not None:
            raise ValueError("Gold and binding-only scores must not report recognition")
        if type(self.B) is not int or self.B not in (0, 1):
            raise ValueError("B must be binary")
        keys = {r.field for r in self.probe.readings} | set(self.probe.binding.domains)
        if set(self.field_valid) != keys or any(type(v) is not bool for v in self.field_valid.values()):
            raise ValueError("Validity must cover every required field")
        if type(self.object_valid) is not bool:
            raise ValueError("Object validity must be boolean")
        if self.C == 1 and not all(self.field_valid[r.field] for r in self.probe.readings):
            raise ValueError("Correct content requires valid readable fields")
        if self.B == 1 and not all(self.field_valid[key] for key in self.probe.binding.domains):
            raise ValueError("Correct binding requires valid binding fields")
        if not self.object_valid and (any(self.field_valid.values()) or self.B or self.C):
            raise ValueError("Unparseable objects cannot receive correct fields")
        object.__setattr__(self, "field_valid", MappingProxyType(dict(self.field_valid)))

    @property
    def J(self) -> int | None:
        # Never accept a separately supplied J that could disagree with C * B.
        return None if self.C is None else self.C * self.B


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _parse_response(response: str) -> dict[str, Any] | None:
    if not isinstance(response, str):
        raise TypeError("Pass the original response string, not an already repaired dict")
    text = response.strip()
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text)
        if not fence:
            return None
        text = fence[1]
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _canonical_relation(values: tuple[str, ...], symmetric: bool) -> tuple[str, ...]:
    if symmetric and values[2] < values[0]:
        return values[2], values[1], values[0]
    return values


def score_response(probe: Probe, response: str, *, condition: str = "raw") -> ProbeScore:
    """Score independent fields. Gold's copied read is NEVER fresh recognition.

    Aliases are case-sensitive with outer whitespace stripped. One JSON object,
    optionally fenced, is accepted. Additional keys are ignored; missing/invalid
    required fields fail individually. No semantic or syntactic repair is made.
    An infrastructure failure is represented by None OUTSIDE this function.
    """
    if condition not in {"raw", "gold"}:
        raise ValueError("Condition must be raw or gold")
    obj = _parse_response(response)
    valid = {}
    read_correct = []
    for target in probe.readings:
        predicted = target.rule.normalize(None if obj is None else obj.get(target.field))
        valid[target.field] = predicted is not None
        gold = target.rule.normalize(target.expected)
        read_correct.append(predicted is not None and target.rule.equivalent(predicted, gold))
    binding = probe.binding
    canonical = []
    for key, domain in binding.domains.items():
        value = None if obj is None else obj.get(key)
        aliases = value if binding.kind == "set" and isinstance(value, list) else [value]
        field_ok = isinstance(value, list) if binding.kind == "set" else isinstance(value, str)
        field_ok = field_ok and all(isinstance(a, str) and a.strip() in domain for a in aliases)
        valid[key] = field_ok
        if field_ok:
            canonical.extend(domain[a.strip()] for a in aliases)
    binding_valid = all(valid[key] for key in binding.domains)
    canonical_binding = None
    set_f1 = None
    B = 0
    if binding_valid:
        canonical_binding = tuple(canonical)
        if binding.kind == "set":
            predicted_set, gold_set = set(canonical), set(binding.gold)
            canonical_binding = tuple(sorted(predicted_set))
            B = int(predicted_set == gold_set)
            set_f1 = 2 * len(predicted_set & gold_set) / (len(predicted_set) + len(gold_set))
        elif binding.kind == "relation":
            canonical_binding = _canonical_relation(canonical_binding, binding.symmetric)
            B = int(canonical_binding == _canonical_relation(binding.gold, binding.symmetric))
        else:
            B = int(canonical_binding == binding.gold)
    elif binding.kind == "set":
        set_f1 = 0.0
    C = int(all(read_correct)) if condition == "raw" and probe.joint_eligible else None
    return ProbeScore(probe, condition, C, B, obj is not None, valid, canonical_binding, set_f1)
