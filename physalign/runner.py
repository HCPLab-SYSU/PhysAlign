"""Fresh-context inference, durable attempt logs and conservative resumption.

This module only loads PublicDataset. It never opens scoring keys or calls the
scorer. Model responses cannot alter future prompts, candidate sets or retries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import inspect
import time
import uuid

from .adapters import InfrastructureError, ModelAdapter, ModelRequest, ModelResponse
from .dataset import require
from .planning import code_hashes, verify_public_plan
from .storage import atomic_json, atomic_text, canonical, file_hash, fingerprint, outside_bundle, read_json, run_lock, write_new

RUN_SCHEMA = "physalign_run_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seal(value: dict) -> dict:
    return {**value, "record_hash": fingerprint(value)}


def checked_record(path: Path) -> dict:
    value = read_json(path)
    require(value.get("record_hash") == fingerprint({k: v for k, v in value.items() if k != "record_hash"}), f"Corrupt run record: {path.name}")
    return value


def request_id(manifest: dict, row: dict) -> str:
    return fingerprint({"run_id": manifest["run_id"], "plan_hash": manifest["plan"]["plan_hash"],
                        "instance_id": row["instance_id"], "condition": row["condition"],
                        "input_hash": row["input_hash"], "adapter": manifest["adapter"]})


def validate_manifest(manifest: dict) -> None:
    require(manifest.get("schema_version") == RUN_SCHEMA, "Unknown run schema")
    fields = ("plan", "adapter", "adapter_provenance", "implementation_hashes")
    if "protocol_metadata" in manifest:
        fields += ("protocol_metadata",)
    require(manifest.get("definition_hash") == fingerprint({k: manifest[k] for k in fields}), "Run definition hash mismatch")
    require(isinstance(manifest.get("run_id"), str) and len(manifest["run_id"]) == 32, "Invalid run identity")


def _terminal(manifest: dict, row: dict, request_hash: str, attempts: list[dict], *,
              indeterminate: bool = False) -> dict:
    require(bool(attempts) or indeterminate, "Terminal outcome has no attempts")
    if not indeterminate and attempts[-1]["status"] != "completed":
        require(not attempts[-1]["retryable"] or len(attempts) == 1 + manifest["plan"]["retry_policy"]["max_retries"],
                "Retryable request was finalized before exhausting the frozen retry policy")
    completed = bool(attempts) and attempts[-1]["status"] == "completed" and not indeterminate
    return _seal({"request_id": request_id(manifest, row), "instance_id": row["instance_id"],
                  "logical_probe_id": row["logical_probe_id"], "problem_id": row["problem_id"],
                  "condition": row["condition"], "input_hash": row["input_hash"], "request_hash": request_hash,
                  "status": "completed" if completed else "infrastructure_missing",
                  "reason": None if completed else "interrupted_request_unknown_response" if indeterminate else attempts[-1]["error_code"],
                  "attempt_count": len(attempts) + int(indeterminate),
                  "attempt_record_hashes": [a["record_hash"] for a in attempts],
                  "response": attempts[-1]["response"] if completed else None})


def _attempt_history(directory: Path, rid: str, expected_hash: str, budget: int) -> tuple[list[dict], bool]:
    """Validate a contiguous log; a started unfinished call is indeterminate."""
    history = []
    saw_gap = False
    unknown = False
    for i in range(budget):
        started = directory / "attempts" / f"{rid}.{i}.started.json"
        finished = directory / "attempts" / f"{rid}.{i}.finished.json"
        if not started.exists():
            require(not finished.exists(), "Finished attempt has no durable start record")
            saw_gap = True
            continue
        require(not saw_gap and not unknown, "Non-contiguous attempt history")
        if history:
            require(history[-1]["status"] == "infrastructure_error" and history[-1]["retryable"], "An answered/nonretryable request was retried")
        start = checked_record(started)
        require((start["request_id"], start["request_hash"], start["attempt"]) == (rid, expected_hash, i), "Attempt start does not match frozen request")
        if not finished.exists():
            unknown = True
            continue
        result = checked_record(finished)
        require((result["request_id"], result["request_hash"], result["attempt"]) == (rid, expected_hash, i), "Attempt result does not match frozen request")
        require(result["status"] in {"completed", "infrastructure_error"}, "Unknown attempt status")
        if history:
            require(history[-1]["status"] == "infrastructure_error" and history[-1]["retryable"], "An answered/nonretryable request was retried")
        if result["status"] == "completed":
            require(result["response"] is not None, "Completed attempt lost its response")
        else:
            require(result["response"] is None, "Usable response incorrectly marked as infrastructure failure")
        history.append(result)
    actual = {p.name for p in (directory / "attempts").glob(rid + ".*.json")}
    allowed = {f"{rid}.{i}.{state}.json" for i in range(budget) for state in ("started", "finished")}
    require(actual <= allowed, "Attempt budget exceeded or unknown attempt log")
    return history, unknown


def _run_one(adapter: ModelAdapter, request: ModelRequest, manifest: dict, row: dict, directory: Path) -> dict:
    rid = request.request_id
    snapshot = request.public_snapshot()
    req_hash = fingerprint(snapshot)
    request_path = directory / "requests" / (rid + ".json")
    if request_path.exists():
        require(read_json(request_path) == snapshot, "Public request changed during resume")
    else:
        write_new(request_path, snapshot)
    budget = 1 + manifest["plan"]["retry_policy"]["max_retries"]
    result_path = directory / "results" / (rid + ".json")
    history, unknown = _attempt_history(directory, rid, req_hash, budget)
    if result_path.exists():
        record = checked_record(result_path)
        expected = _terminal(manifest, row, req_hash, history, indeterminate=unknown)
        require(record == expected, "Final prediction disagrees with its authoritative attempt history")
        return record
    if unknown:
        record = _terminal(manifest, row, req_hash, history, indeterminate=True)
        atomic_json(result_path, record)
        return record
    if history and (history[-1]["status"] == "completed" or not history[-1]["retryable"] or len(history) == budget):
        record = _terminal(manifest, row, req_hash, history)
        atomic_json(result_path, record)
        return record
    for index in range(len(history), budget):
        require(adapter.info.manifest() == manifest["adapter"], "Adapter settings changed within a paired run")
        write_new(directory / "attempts" / f"{rid}.{index}.started.json",
                  _seal({"request_id": rid, "request_hash": req_hash, "attempt": index, "started_at": _now()}))
        started_at = time.perf_counter()
        try:
            response = adapter.generate(request)
        except InfrastructureError as exc:
            result = {"status": "infrastructure_error", "response": None, "error_code": exc.code, "retryable": exc.retryable}
        else:
            require(isinstance(response, ModelResponse), "Adapter returned an invalid response object")
            # No answer parsing or correctness feedback exists on this path.
            result = {"status": "completed", "response": response.record(), "error_code": None, "retryable": False}
        result = _seal({**result, "request_id": rid, "request_hash": req_hash, "attempt": index,
                        "finished_at": _now(), "duration_seconds": time.perf_counter() - started_at})
        atomic_json(directory / "attempts" / f"{rid}.{index}.finished.json", result)
        history.append(result)
        if result["status"] == "completed" or not result["retryable"]:
            break
    record = _terminal(manifest, row, req_hash, history)
    atomic_json(result_path, record)
    return record


def collect_records(directory: str | Path, manifest: dict, *, allow_incomplete: bool = False) -> list[dict | None]:
    """Read-only audit of final records against durable attempts and input hashes."""
    directory = Path(directory)
    validate_manifest(manifest)
    budget = 1 + manifest["plan"]["retry_policy"]["max_retries"]
    expected_names, records = set(), []
    for row in manifest["plan"]["requests"]:
        rid = request_id(manifest, row)
        expected_names.add(rid + ".json")
        result_path = directory / "results" / (rid + ".json")
        if not result_path.exists():
            require(allow_incomplete, "Run is incomplete; resume it or explicitly request an incomplete diagnostic")
            records.append(None)
            continue
        snapshot = read_json(directory / "requests" / (rid + ".json"))
        require(snapshot["request_id"] == rid, "Request snapshot identity mismatch")
        req_hash = fingerprint(snapshot)
        history, unknown = _attempt_history(directory, rid, req_hash, budget)
        require(bool(history) or unknown, "Final prediction has no model attempt")
        record = checked_record(result_path)
        require(record == _terminal(manifest, row, req_hash, history, indeterminate=unknown), "Prediction/attempt inconsistency")
        records.append(record)
    require({p.name for p in (directory / "results").glob("*.json")} <= expected_names, "Run contains predictions outside its frozen plan")
    return records


def run_evaluation(plan: dict, adapter: ModelAdapter, output: str | Path, *,
                   dataset_root=None, resume: bool = False, progress=None,
                   adapter_configuration_hash: str | None = None) -> dict:
    dataset = verify_public_plan(plan, dataset_root)
    def make_request(row, rid, settings):
        item = dataset.items[row["instance_id"], row["condition"]]
        images = dataset.images(item)
        require([{"asset_id": a.asset_id, "sha256": a.sha256, "mime_type": a.mime_type,
                  "width": a.width, "height": a.height} for a in images] == row["images"], "Image data differ from frozen plan")
        messages = item.record["input"]["messages"]
        return ModelRequest(rid, messages["system"], messages["user"], images, settings)
    return run_requests(plan, adapter, output, make_request, source_root=dataset.root,
                        resume=resume, progress=progress, adapter_configuration_hash=adapter_configuration_hash)


def run_requests(plan: dict, adapter: ModelAdapter, output: str | Path, request_factory, *,
                 source_root: Path, resume=False, progress=None, adapter_configuration_hash=None,
                 protocol_metadata=None):
    """Shared first-response journal; caller validates its frozen experiment plan.

    request_factory only returns public ModelRequests. A suite can use the same
    journal for original solving and controls without opening keys at inference.
    """
    directory = Path(output).resolve()
    outside_bundle(directory, source_root)
    info = adapter.info.manifest()
    try:
        adapter_file = Path(inspect.getfile(type(adapter)))
        implementation = file_hash(adapter_file) if adapter_file.is_file() else None
    except TypeError:
        implementation = None
    provenance = {"class": type(adapter).__module__ + "." + type(adapter).__qualname__,
                  "module_hash": implementation, "configuration_hash": adapter_configuration_hash}
    definition = {"plan": plan, "adapter": info, "adapter_provenance": provenance,
                  "implementation_hashes": code_hashes(all_modules=True)}
    if protocol_metadata is not None:
        definition['protocol_metadata'] = protocol_metadata
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            require(resume, "Run already exists; use resume to preserve its first outcomes")
            manifest = read_json(manifest_path)
            validate_manifest(manifest)
            require(manifest["definition_hash"] == fingerprint(definition), "Dataset/protocol/adapter/code changed; cannot resume this run")
        else:
            require(not resume, "No existing run to resume")
            require(not [p for p in directory.iterdir() if p.name != ".lock"], "Refusing to mix with a nonempty output directory")
            manifest = {"schema_version": RUN_SCHEMA, "run_id": uuid.uuid4().hex, "created_at": _now(),
                        **definition, "definition_hash": fingerprint(definition)}
            atomic_json(manifest_path, manifest)
        for name in ("requests", "attempts", "results"):
            (directory / name).mkdir(exist_ok=True)
        try:
            for index, row in enumerate(plan["requests"]):
                request = request_factory(row, request_id(manifest, row), canonical(info["settings"]))
                require(isinstance(request, ModelRequest), "Request factory did not return public input")
                record = _run_one(adapter, request, manifest, row, directory)
                if progress:
                    progress({"completed": index + 1, "total": len(plan["requests"]),
                              "instance_id": row["instance_id"], "condition": row["condition"], "status": record["status"]})
        finally:
            records = collect_records(directory, manifest, allow_incomplete=True)
            atomic_text(directory / "predictions.jsonl", "".join(canonical(r) + "\n" for r in records if r is not None))
            atomic_json(directory / "status.json", {"planned_requests": len(records),
                        "completed_outputs": sum(r is not None and r["status"] == "completed" for r in records),
                        "infrastructure_missing": sum(r is not None and r["status"] == "infrastructure_missing" for r in records),
                        "pending_requests": sum(r is None for r in records), "updated_at": _now()})
        return read_json(directory / "status.json")
