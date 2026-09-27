"""Offline scoring: frozen public requests + immutable response logs + private truth."""

from __future__ import annotations

from pathlib import Path

from .adapters import ModelRequest
from .bootstrap import paired_cluster_bootstrap
from .dataset import load_truth, require
from .metrics import evaluate, uniform_candidate_baseline
from .planning import metric_plan, verify_public_plan
from .runner import collect_records, request_id, validate_manifest
from .scoring import _parse_response, score_response
from .storage import atomic_json, atomic_text, canonical, file_hash, fingerprint, outside_bundle, read_json, run_lock


def _intervals(plan, probes, metric_spec, raw, gold):
    config = plan["bootstrap"]
    if config["n_resamples"] == 0:
        return {"status": "disabled_in_frozen_plan"}
    probe_ids = {p.probe_id for p in probes}
    clusters = {r["problem_id"]: r["cluster_id"] for r in plan["probes"] if r["instance_id"] in probe_ids}
    # Each scope resamples its contributing mothers. Within this callback both
    # Raw and Gold use the SAME counts, and the full type aggregation is redone.
    def statistic(counts):
        m = evaluate(metric_spec, raw, gold, problem_multiplicities=counts)["metrics"]
        keys = ["CAcc", "BAcc", "JAcc", "BAcc_L", "BAcc_given_C", "BindErr_given_C"]
        if gold is not None:
            keys.extend(("BAcc_gold", "delta_BAcc_gold_raw"))
        return {key: m[key] for key in keys}
    return paired_cluster_bootstrap(sorted(clusters), statistic, cluster_ids=clusters, **config)


def _chance(spec):
    if any(p.binding.kind != "single" for p in spec.probes):
        return {"status": "not_defined_for_mixed_set_or_relation_pool",
                "reason": "A set/relation sampling distribution must be declared separately; the pool is not silently restricted."}
    return {"status": "ok", "distribution": "uniform_over_public_aliases_per_probe",
            "binding": uniform_candidate_baseline(spec.binding_pool)}


def _parsed_log(text):
    """A diagnostic copy must never make a served response unscorable.

    For example JSON 1e999 overflows Python's float in an irrelevant field.
    The scorer still evaluates the requested fields; the untouched response
    remains in the authoritative log even if this convenience copy is omitted.
    """
    parsed = _parse_response(text)
    try:
        canonical(parsed)
    except (ValueError, RecursionError):
        return None, "not_serializable_original_response_preserved"
    return parsed, "parsed" if parsed is not None else "invalid_object"


def score_run(run_directory: str | Path, *, dataset_root=None, output=None,
              allow_incomplete: bool = False) -> dict:
    directory = Path(run_directory).resolve()
    require((directory / "manifest.json").is_file(), "No frozen run manifest")
    with run_lock(directory):
        manifest = read_json(directory / "manifest.json")
        validate_manifest(manifest)
        plan = manifest["plan"]
        dataset = verify_public_plan(plan, dataset_root)
        # Source/private keys are read for the first time on this offline path.
        dataset.verify_inventory()
        for relative, expected in plan["source_hashes"].items():
            require(file_hash(dataset.root / relative) == expected, "Scoring truth/source changed since preparation")
        truth, metadata = load_truth(dataset)
        selected = []
        for row in plan["probes"]:
            p = truth[row["instance_id"]]
            require((p.problem_id, p.probe_type, p.joint_eligible, p.packet_nonempty) ==
                    (row["problem_id"], row["probe_type"], row["joint_eligible"], row["packet_nonempty"]), "Scoring eligibility or type changed")
            require(metadata[p.probe_id]["cluster_id"] == row["cluster_id"], "Duplicate/source cluster changed")
            selected.append(p)
        records = collect_records(directory, manifest, allow_incomplete=allow_incomplete)
        raw, gold, scored_rows = {}, {}, []
        pending = []
        for row, record in zip(plan["requests"], records):
            iid, condition = row["instance_id"], row["condition"]
            item = dataset.items[iid, condition]
            rid = request_id(manifest, row)
            if record is not None:
                inp = item.record["input"]
                expected = ModelRequest(rid, inp["messages"]["system"], inp["messages"]["user"],
                                        dataset.images(item), canonical(manifest["adapter"]["settings"])).public_snapshot()
                require(read_json(directory / "requests" / (rid + ".json")) == expected, "Logged model input differs from the frozen public input")
                require(fingerprint(expected) == record["request_hash"], "Prediction belongs to a different model input")
            else:
                pending.append({"instance_id": iid, "condition": condition})
            score = (score_response(truth[iid], record["response"]["text"], condition=condition)
                     if record is not None and record["status"] == "completed" else None)
            parsed, parsed_status = (_parsed_log(record["response"]["text"])
                                     if score is not None else (None, "no_completed_response"))
            (raw if condition == "raw" else gold)[iid] = score
            scored_rows.append({"instance_id": iid, "condition": condition, "problem_id": truth[iid].problem_id,
                                "probe_type": truth[iid].probe_type, "request_id": rid,
                                "interface": item.record["interface"],
                                "status": "pending" if record is None else record["status"],
                                "C": None if score is None else score.C, "B": None if score is None else score.B,
                                "J": None if score is None else score.J,
                                "parsed_output": parsed, "parsed_output_status": parsed_status,
                                "object_valid": None if score is None else score.object_valid,
                                "field_valid": None if score is None else dict(score.field_valid),
                                "canonical_binding": None if score is None else score.canonical_binding,
                                "set_f1": None if score is None else score.set_f1})
        full_spec = metric_plan(selected, plan["weights"]["full"])
        full_gold = gold if len(plan["paired_ids"]) == len(selected) else None
        raw_all = evaluate(full_spec, raw, full_gold)
        paired_ids = set(plan["paired_ids"])
        paired_probes = [p for p in selected if p.probe_id in paired_ids]
        paired, paired_intervals = None, None
        if paired_probes:
            paired_spec = metric_plan(paired_probes, plan["weights"]["paired"])
            paired_raw = {p.probe_id: raw[p.probe_id] for p in paired_probes}
            paired_gold = {p.probe_id: gold[p.probe_id] for p in paired_probes}
            paired = evaluate(paired_spec, paired_raw, paired_gold)
            paired_intervals = _intervals(plan, paired_probes, paired_spec, paired_raw, paired_gold)
        intervals = _intervals(plan, selected, full_spec, raw, full_gold)
        prediction_path = directory / "predictions.jsonl"
        if prediction_path.exists():
            expected_text = "".join(canonical(r) + "\n" for r in records if r is not None)
            require(prediction_path.read_text(encoding="utf-8") == expected_text, "Prediction export differs from authoritative per-request logs")
        responses = [r["response"] for r in records if r is not None and r["status"] == "completed"]
        returned_versions = sorted({r["returned_model_version"] for r in responses if r["returned_model_version"] is not None})
        report = {"schema_version": "physalign_evaluation_report_v1", "run_id": manifest["run_id"],
                  "plan_hash": plan["plan_hash"], "scientific_run": manifest["adapter"]["scientific_run"],
                  "data_split": plan["split"], "adapter": manifest["adapter"],
                  "raw_all": raw_all, "paired": paired,
                  "uniform_candidate_baseline": {"raw_all": _chance(full_spec),
                                                 "paired": _chance(paired_spec) if paired_probes else None},
                  "intervals": {"raw_all": intervals, "paired": paired_intervals},
                  "pairing": {"planned_raw_probes": len(selected), "planned_paired_probes": len(paired_probes),
                              "paired_ids": plan["paired_ids"], "raw_only_ids": plan["raw_only_ids"],
                              "full_binding_gold_available": full_gold is not None,
                              "pair_selection": "frozen_export_availability_before_inference"},
                  "service": {"planned_requests": len(records), "completed_outputs": len(responses),
                              "infrastructure_missing": sum(r is not None and r["status"] == "infrastructure_missing" for r in records),
                              "pending_requests": pending},
                  "provenance": {"returned_model_versions": returned_versions,
                                 "multiple_returned_versions": len(returned_versions) > 1,
                                 "response_version_unavailable": sum(r["returned_model_version"] is None for r in responses),
                                 "human_reviewed_probes": plan["interpretation"]["human_reviewed_probes"],
                                 "scorer_hashes": plan["scorer_hashes"]}}
        destination = Path(output).resolve() if output is not None else directory / "report.json"
        require(destination.suffix == ".json", "Report output must be a .json file")
        outside_bundle(destination, dataset.root)
        if destination.exists():
            prior = read_json(destination)
            require(prior.get("schema_version") == report["schema_version"] and prior.get("run_id") == report["run_id"], "Refusing to overwrite an unrelated artifact")
        atomic_json(destination, report)
        atomic_text(directory / "scores.jsonl", "".join(canonical(r) + "\n" for r in scored_rows))
        return report
