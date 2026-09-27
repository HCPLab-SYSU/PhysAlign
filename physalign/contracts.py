"""Lossless JSON representation of frozen scoring contracts (never model input)."""

from .scoring import BindingTarget, OCRRule, Probe, QuantityRule, ReadTarget


def encode_probe(p):
    readings = []
    for r in p.readings:
        rule = ({"kind": "ocr", "normalizer_id": r.rule.normalizer_id} if isinstance(r.rule, OCRRule)
                else {"kind": "quantity", "unit_scales": dict(r.rule.unit_scales),
                      "absolute_tolerance": r.rule.absolute_tolerance})
        readings.append({"field": r.field, "anchor_id": r.anchor_id, "expected": r.expected, "rule": rule})
    return {"probe_id": p.probe_id, "problem_id": p.problem_id, "probe_type": p.probe_type,
            "binding": {"kind": p.binding.kind, "domains": [
                {"field": f, "candidates": dict(d)} for f, d in p.binding.domains.items()],
                "gold": list(p.binding.gold), "symmetric": p.binding.symmetric},
            "readings": readings, "packet_nonempty": p.packet_nonempty}


def decode_probe(row):
    b = row["binding"]
    readings = []
    for r in row["readings"]:
        spec = r["rule"]
        rule = (OCRRule(spec["normalizer_id"]) if spec["kind"] == "ocr" else
                QuantityRule(spec["unit_scales"], spec["absolute_tolerance"]))
        readings.append(ReadTarget(r["field"], r["anchor_id"], r["expected"], rule))
    return Probe(row["probe_id"], row["problem_id"], row["probe_type"],
                 BindingTarget(b["kind"], {d["field"]: d["candidates"] for d in b["domains"]},
                               tuple(b["gold"]), b["symmetric"]), tuple(readings), row["packet_nonempty"])


def remap_probe(probe, old_to_new):
    row = encode_probe(probe)
    for d in row["binding"]["domains"]:
        d["candidates"] = {old_to_new[a]: identity for a, identity in d["candidates"].items()}
    return decode_probe(row)
