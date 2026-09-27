"""Prepare a reviewable evaluation manifest before any model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .dataset import PublicDataset, load_truth, require
from .metrics import EvaluationPlan
from .storage import file_hash, fingerprint, outside_bundle, read_json, write_new

PLAN_SCHEMA = "physalign_run_plan_v1"
SCORER_FILES = ("scoring.py", "ocr_normalizers.py", "metrics.py", "bootstrap.py")


def code_hashes(*, all_modules: bool = False) -> dict:
    directory = Path(__file__).resolve().parent
    files = sorted(directory.glob("*.py")) if all_modules else [directory / name for name in SCORER_FILES]
    return {p.name: file_hash(p) for p in files}


def weights_for(probes, override: Mapping | None = None) -> dict:
    overrides = {} if override is None else dict(override)
    allowed = {"binding", "joint", "packet", "binding_only"}
    require(set(overrides) <= allowed, "Unknown type-weight pool")
    if not probes:
        require(not overrides, "Cannot supply weights for an empty paired pool")
        return {name: {} for name in sorted(allowed)}
    plan = EvaluationPlan(tuple(probes), overrides.get("binding"), overrides.get("joint"),
                          overrides.get("packet"), overrides.get("binding_only"))
    return {name: dict(getattr(plan, name + "_pool").type_weights) for name in sorted(allowed)}


def metric_plan(probes, weights: dict) -> EvaluationPlan:
    return EvaluationPlan(tuple(probes), weights["binding"], weights["joint"], weights["packet"], weights["binding_only"])


def prepare_plan(dataset_root: str | Path, *, split: str, conditions=("raw", "gold"),
                 problem_ids=None, type_weights=None, bootstrap_resamples: int = 2000,
                 bootstrap_seed: int = 2027, confidence: float = .95,
                 max_retries: int = 2, reservation_files=()) -> dict:
    """Selection uses approved metadata only. No filtering by model performance."""
    require(tuple(conditions) in {("raw",), ("raw", "gold")}, "Core evaluation requires raw, optionally paired gold")
    require(type(max_retries) is int and 0 <= max_retries <= 2, "At most two infrastructure retries are allowed")
    require(type(bootstrap_resamples) is int and (bootstrap_resamples == 0 or bootstrap_resamples >= 2), "Bootstrap count must be 0 or at least 2")
    require(type(bootstrap_seed) is int and 0 < confidence < 1, "Invalid bootstrap settings")
    dataset = PublicDataset(dataset_root)
    dataset.verify_inventory()
    probes, metadata = load_truth(dataset)
    by_mother, by_cluster, mother_clusters = {}, {}, {}
    for iid, p in probes.items():
        meta = metadata[iid]
        by_mother.setdefault(p.problem_id, set()).add(meta["split"])
        by_cluster.setdefault(meta["cluster_id"], set()).add(meta["split"])
        mother_clusters.setdefault(p.problem_id, set()).add(meta["cluster_id"])
    require(all(len(v) == 1 for v in by_mother.values()), "A mother crosses dataset splits")
    require(all(len(v) == 1 for v in by_cluster.values()), "A duplicate/source cluster crosses dataset splits")
    require(all(len(v) == 1 for v in mother_clusters.values()), "One mother was assigned to multiple bootstrap clusters")
    reservations, reserved_cluster_ids, extra_reservation_hashes = set(), set(), {}

    def add_reservations(reservation):
        require(isinstance(reservation.get("problem_ids"), list), "Invalid development reservation")
        cluster_ids = reservation.get("cluster_ids", [])
        require(isinstance(cluster_ids, list), "Invalid reserved cluster list")
        require(all(isinstance(s, str) and s for s in reservation["problem_ids"] + cluster_ids), "Invalid reserved identity")
        reservations.update(reservation["problem_ids"])
        reserved_cluster_ids.update(cluster_ids)

    if "DEVELOPMENT_RESERVATION.json" in dataset.inventory:
        add_reservations(dataset.read_verified("DEVELOPMENT_RESERVATION.json"))
    # Reservations belong to the dataset or explicit caller-supplied files.
    paths = [Path(p).resolve() for p in reservation_files]
    for path in sorted(set(paths)):
        add_reservations(read_json(path))
        extra_reservation_hashes[str(path)] = file_hash(path)
    selected = [p for iid, p in probes.items() if metadata[iid]["split"] == split]
    if problem_ids is not None:
        requested = set(problem_ids)
        require(requested and requested <= {p.problem_id for p in selected}, "Unknown/empty problem selection for this split")
        selected = [p for p in selected if p.problem_id in requested]
    require(bool(selected), "No probes in the requested split")
    if split == "test":
        require(dataset.manifest["schema_version"] != "physalign_eval_starter_samples_v1", "Starter probes are development data, not a formal test export")
        require(dataset.manifest.get("formal_release_eligibility_checked") is True, "Formal test export has no completed eligibility declaration")
        require(not ({p.problem_id for p in selected} & reservations), "Development-reserved mother selected for unseen test")
        reserved_clusters = reserved_cluster_ids | {metadata[iid]["cluster_id"] for iid, p in probes.items() if p.problem_id in reservations}
        require(not {metadata[p.probe_id]["cluster_id"] for p in selected} & reserved_clusters, "A selected test cluster contains a development-reserved mother")
    selected.sort(key=lambda p: p.probe_id)
    rows, probe_rows = [], []
    for p in selected:
        iid = p.probe_id
        meta = metadata[iid]
        exported = [c for c in ("raw", "gold") if (iid, c) in dataset.items]
        scheduled = [c for c in conditions if c in exported]
        probe_rows.append({"instance_id": iid, "problem_id": p.problem_id, "probe_type": p.probe_type,
                           "logical_probe_id": meta["logical_probe_id"], "split": split,
                           "interface": dataset.items[iid, "raw"].record["interface"],
                           "cluster_id": meta["cluster_id"], "joint_eligible": p.joint_eligible,
                           "packet_nonempty": p.packet_nonempty, "exported_conditions": exported,
                           "scheduled_conditions": scheduled, "review_status": meta["review_status"],
                           "human_reviewed": meta["human_reviewed"]})
        for c in scheduled:
            item = dataset.items[iid, c]
            images = dataset.images(item)
            rows.append({"instance_id": iid, "logical_probe_id": meta["logical_probe_id"],
                         "problem_id": p.problem_id, "condition": c, "input_hash": item.input_hash,
                         "record_hash": fingerprint(item.record),
                         "images": [{"asset_id": a.asset_id, "sha256": a.sha256, "mime_type": a.mime_type,
                                     "width": a.width, "height": a.height} for a in images]})
    paired_ids = [r["instance_id"] for r in probe_rows if "gold" in r["scheduled_conditions"]]
    paired_set = set(paired_ids)
    paired = [p for p in selected if p.probe_id in paired_set]
    overrides = type_weights or {}
    require(set(overrides) <= {"full", "paired"}, "Type weights must specify full and/or paired pools")
    source_hashes = dict(dataset.inventory)
    for relative in ("manifest.json", "FILE_MANIFEST.json"):
        source_hashes[relative] = file_hash(dataset.root / relative)
    payload = {"schema_version": PLAN_SCHEMA, "dataset_hint": str(dataset.root), "split": split,
               "dataset_schema": dataset.manifest["schema_version"],
               "purpose": "local_probe_evaluation", "conditions": list(conditions),
               "source_hashes": source_hashes, "scorer_hashes": code_hashes(),
               "probes": probe_rows, "requests": rows, "paired_ids": paired_ids,
               "raw_only_ids": [p.probe_id for p in selected if p.probe_id not in paired_set],
               "weights": {"full": weights_for(selected, overrides.get("full")), "paired": weights_for(paired, overrides.get("paired"))},
               "bootstrap": {"n_resamples": bootstrap_resamples, "seed": bootstrap_seed, "confidence": confidence},
               "retry_policy": {"max_retries": max_retries, "retry_only": "infrastructure_without_usable_response"},
               "reserved_problem_ids": sorted(reservations), "reserved_cluster_ids": sorted(reserved_cluster_ids),
               "extra_reservation_hashes": extra_reservation_hashes,
               "interpretation": {"gold_copied_read_is_recognition": False,
                                  "missing_gold_view_is_infrastructure_failure": False,
                                  "binding_type_source": "task_id" if dataset.manifest["schema_version"] == "physalign_eval_starter_samples_v1" else "private.probe_type",
                                  "human_reviewed_probes": sum(r["human_reviewed"] for r in probe_rows)}}
    return {**payload, "plan_hash": fingerprint(payload)}


def validate_plan(plan: dict) -> None:
    require(plan.get("schema_version") == PLAN_SCHEMA, "Unknown run plan schema")
    require(plan.get("plan_hash") == fingerprint({k: v for k, v in plan.items() if k != "plan_hash"}), "Run plan hash mismatch")
    require(plan["scorer_hashes"] == code_hashes(), "Scoring implementation changed after the protocol was frozen; prepare a new plan")
    keys = [(r["instance_id"], r["condition"]) for r in plan["requests"]]
    require(len(keys) == len(set(keys)), "Duplicate scheduled request")


def save_plan(plan: dict, path: str | Path) -> None:
    validate_plan(plan)
    outside_bundle(Path(path), Path(plan["dataset_hint"]))
    write_new(Path(path), plan)


def load_plan(path: str | Path) -> dict:
    plan = read_json(Path(path))
    validate_plan(plan)
    return plan


def verify_public_plan(plan: dict, dataset_root=None) -> PublicDataset:
    validate_plan(plan)
    dataset = PublicDataset(dataset_root or plan["dataset_hint"])
    for name in ("manifest.json", "FILE_MANIFEST.json"):
        require(file_hash(dataset.root / name) == plan["source_hashes"][name], "Bundle manifest changed after preparation")
    for row in plan["requests"]:
        item = dataset.items.get((row["instance_id"], row["condition"]))
        require(item is not None and fingerprint(item.record) == row["record_hash"] and item.input_hash == row["input_hash"], "Frozen public request changed")
    return dataset
