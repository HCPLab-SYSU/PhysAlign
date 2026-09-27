"""Output shape validation only; no model runner or response scoring."""
import re
from typing import Any
from reference_normalizers import NORMALIZER_IDS, normalize
class ScoreContractError(ValueError):
    pass
def need(condition, message):
    if not condition:
        raise ScoreContractError(message)

def validate_contract(key: dict[str, Any]) -> None:
    c = key["response_contract"]
    need(set(c) == {"binding_key", "cardinality", "read_shape", "read_anchor_ids"}, "RESPONSE_CONTRACT_KEYS")
    need(c["binding_key"] in ("owner", "referent", "role", "subjects", "endpoint", "evidence"), "BINDING_KEY")
    need(c["cardinality"] in ("one", "set"), "CARDINALITY")
    need(c["read_shape"] in ("none", "string", "object_by_anchor"), "READ_SHAPE")
    ids = c["read_anchor_ids"]
    need(isinstance(ids, list) and all(isinstance(x, str) and re.fullmatch(r"R[1-9][0-9]*", x) for x in ids), "READ_ANCHORS")
    need(len(ids) == len(set(ids)), "DUPLICATE_READ_ANCHOR")
    need((c["read_shape"] == "none" and not ids) or
         (c["read_shape"] == "string" and len(ids) == 1) or
         (c["read_shape"] == "object_by_anchor" and len(ids) >= 2), "READ_SHAPE_ARITY")
    need(type(key["joint_eligible"]) is bool and key["joint_eligible"] == bool(ids), "JOINT_CONTRACT_MISMATCH")
    targets = key["read_targets"]
    need(isinstance(targets, list) and [x["anchor_alias"] for x in targets] == ids, "READ_TARGET_ORDER_OR_COVERAGE")
    for t in targets:
        need(t["normalizer_id"] in NORMALIZER_IDS, "UNKNOWN_NORMALIZER")
        need(normalize(t["expected"], t["normalizer_id"]) is not None, "UNSUPPORTED_GOLD_READING")
    candidates = key["candidate_map"]
    need(isinstance(candidates, dict) and len(candidates) >= 2, "CANDIDATE_MAP")
    for a, t in candidates.items():
        need(isinstance(a, str) and re.fullmatch(r"E[1-9][0-9]*", a) is not None, "ALIAS_SCHEMA")
        need(set(t) == {"kind", "id"} and t["kind"] in ("entity", "role", "evidence") and isinstance(t["id"], str) and bool(t["id"]), "CANONICAL_TARGET")
    universe = {(t["kind"], t["id"]) for t in candidates.values()}
    need(len(universe) == len(candidates), "DUPLICATE_CANONICAL_CANDIDATE")
    gold = key["gold_targets"]
    need(isinstance(gold, list) and bool(gold), "EMPTY_GOLD")
    need(all(isinstance(t, dict) and set(t) == {"kind", "id"} for t in gold), "GOLD_SCHEMA")
    gs = {(t["kind"], t["id"]) for t in gold}
    need(len(gs) == len(gold) and gs <= universe, "GOLD_CANDIDATE_COVERAGE")
    need(c["cardinality"] != "one" or len(gs) == 1, "SINGLE_TARGET_COUNT")


