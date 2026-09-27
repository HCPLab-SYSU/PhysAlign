"""Release regressions using generated data; no archived experiments or API calls."""
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_release import audit, inspect_content
from physgraph_annotation_lib import (
    STAGES, blank_document, json_sha256, read_json, read_jsonl,
    validate_document, write_json_atomic, write_jsonl_atomic,
)
from physgraph_api_client import ExclusiveRunLock
from physgraph_pipeline.adapters import _safe_problem_id, PhysicsDatasetAdapter
from physgraph_pipeline.config import ConfigError, load_pipeline_config
from physgraph_pipeline.doctor import diagnose_workspace
from physgraph_pipeline.workspace import export_approved_graphs, migrate_workspace, prepare_workspace
from run_physgraph_annotation_review import AnnotationServer, AnnotationState, Handler
from workspace_fixtures import example_workspace


@pytest.fixture
def approved(tmp_path):
    workspace = example_workspace(tmp_path)
    problem = read_jsonl(workspace / "blind/manifest.jsonl")[0]
    pid = problem["problem_id"]
    problem.update(raw_question="A block is shown. Find its acceleration.",
                   segments={"stem": "A block is shown.", "query": "Find its acceleration.", "options": []},
                   segmentation={"confidence": "high", "method": "synthetic_test", "review_status": "approved"})
    write_jsonl_atomic(workspace / "blind/manifest.jsonl", [problem])
    docs = {stage: blank_document(stage, pid) for stage in STAGES}
    visuals = [{"id": "v001", "type": "object_shape", "subtype": "block", "text": "",
                "image_id": problem["images"][0]["image_id"], "bbox_1000": [328, 417, 547, 717],
                "keypoints_1000": [], "center_1000": [-1, -1], "radius_1000": -1, "confidence": "high"}]
    physical = [{"id": "p001", "type": "body", "subtype": "block", "name": "block", "symbol": "",
                 "visual_anchor_ids": ["v001"], "text_mention_ids": [], "provenance": ["IMAGE"], "confidence": "high"}]
    bindings = [{"id": "b001", "type": "represents", "from_id": "v001", "to_id": "p001",
                 "provenance": ["IMAGE"], "evidence_visual_ids": ["v001"], "evidence_mention_ids": []}]
    mentions = [{"id": "m001", "section": "query", "quote": "acceleration", "occurrence": 1, "role": "query_target"}]
    target = {**docs["pass3"]["query_target"], "mention_id": "m001", "target_kind": "acceleration", "target_node_ids": ["p001"]}
    docs["pass1"]["visual_nodes"] = deepcopy(visuals)
    docs["pass2"].update(physical_nodes=deepcopy(physical), bindings=deepcopy(bindings))
    docs["pass3"].update(text_mentions=deepcopy(mentions), query_target=deepcopy(target))
    docs["pass4"].update(visual_nodes=deepcopy(visuals), physical_nodes=deepcopy(physical),
                         bindings=deepcopy(bindings), text_mentions=deepcopy(mentions), query_target=deepcopy(target))
    state = AnnotationState(workspace, ROOT / "physgraph_review_app", workspace / "prompts")
    previous = {}
    for stage, document in docs.items():
        assert not [i for i in validate_document(stage, document, problem, previous) if i["level"] == "error"]
        payload = state.save_stage(pid, stage, {"action": "approve", "reviewer": "SYNTHETIC_TEST_NOT_HUMAN", "note": "", "document": document})
        assert payload["stages"][stage]["status"] == "approved"
        previous[stage] = document
    return workspace, pid


def test_export_accepts_complete_unchanged_approval(approved, tmp_path):
    workspace, pid = approved
    result = export_approved_graphs(workspace, tmp_path / "export.jsonl")
    assert result["records"] == 1
    assert read_jsonl(tmp_path / "export.jsonl")[0]["problem_id"] == pid


@pytest.mark.parametrize("change", ["document", "approval", "invalid_graph", "media", "segmentation", "excluded"])
def test_export_refuses_untrusted_approval(approved, tmp_path, change):
    workspace, pid = approved
    state_path = workspace / "reviews/state.json"
    state = read_json(state_path)
    if change in {"document", "invalid_graph"}:
        path = workspace / "passes/pass4" / (pid + ".json")
        document = read_json(path)
        document["query_target"]["target_node_ids"] = ["p999"]
        write_json_atomic(path, document)
        if change == "invalid_graph":
            state["problems"][pid]["stages"]["pass4"]["document_sha256"] = json_sha256(document)
    elif change == "approval":
        state["problems"][pid]["stages"]["pass2"]["status"] = "draft"
    elif change == "excluded":
        state["problems"][pid]["exclusion"] = {"status": "excluded"}
    elif change == "media":
        image = tmp_path / "dataset/images/block_on_surface.png"
        image.write_bytes(image.read_bytes() + b"changed")
    else:
        manifest = read_jsonl(workspace / "blind/manifest.jsonl")
        manifest[0]["segmentation"]["review_status"] = "needs_review"
        write_jsonl_atomic(workspace / "blind/manifest.jsonl", manifest)
    write_json_atomic(state_path, state)
    output = tmp_path / "export.jsonl"
    with pytest.raises(ConfigError):
        export_approved_graphs(workspace, output)
    assert not output.exists()


def test_doctor_detects_changed_media(tmp_path):
    workspace = example_workspace(tmp_path)
    image = tmp_path / "dataset/images/block_on_surface.png"
    image.write_bytes(image.read_bytes() + b"changed")
    assert not diagnose_workspace(workspace)["ok"]


def test_force_refuses_changed_source_and_keeps_workspace(tmp_path):
    workspace = example_workspace(tmp_path)
    config = load_pipeline_config(tmp_path / "config.json")
    prepare_workspace(config, force=True)
    manifest = (workspace / "blind/manifest.jsonl").read_bytes()
    path = tmp_path / "dataset/annotations.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["stem"] = "Changed question"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ConfigError):
        prepare_workspace(config, force=True)
    assert (workspace / "blind/manifest.jsonl").read_bytes() == manifest


def test_migrate_infers_source_after_dataset_move(tmp_path):
    workspace = example_workspace(tmp_path)
    moved = tmp_path / "moved_dataset"
    shutil.move(str(tmp_path / "dataset"), moved)
    migrate_workspace(workspace, moved)
    assert diagnose_workspace(workspace)["ok"]
    config = read_json(workspace / "workspace_config.json")
    assert "legacy_paths" not in config


@pytest.mark.parametrize("pid", ["CON", "nul.txt", "LPT9", "com1.json", None, {}, "../escape"])
def test_reject_nonportable_ids(pid):
    with pytest.raises(ConfigError):
        _safe_problem_id(pid, 0)


def test_reject_case_colliding_ids(tmp_path):
    example_workspace(tmp_path)
    path = tmp_path / "dataset/annotations.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    other = {**record, "id": record["id"].upper()}
    path.write_text("\n".join(json.dumps(r) for r in (record, other)), encoding="utf-8")
    with pytest.raises(ConfigError):
        PhysicsDatasetAdapter(load_pipeline_config(tmp_path / "config.json")).load()


def test_pid_probe_does_not_terminate_another_process():
    child = subprocess.Popen([sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert ExclusiveRunLock._pid_is_running(child.pid)
        assert child.poll() is None
        assert ExclusiveRunLock._pid_is_running(os.getpid())
    finally:
        child.communicate(timeout=10)
    assert not ExclusiveRunLock._pid_is_running(child.pid)


def test_http_rejects_foreign_origin_host_and_form_without_changes(tmp_path):
    workspace = example_workspace(tmp_path)
    state_path = workspace / "reviews/state.json"
    before = state_path.read_bytes()
    server = AnnotationServer(("127.0.0.1", 0), Handler)
    server.state = AnnotationState(workspace, ROOT / "physgraph_review_app", workspace / "prompts")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = "127.0.0.1:" + str(server.server_port)
    try:
        for method, path, headers, expected in [
            ("GET", "/api/health", {}, 200),
            ("GET", "/api/catalog", {"Host": "foreign.invalid:" + str(server.server_port)}, 400),
            ("POST", "/api/problem/demo_force_001/segmentation", {"Origin": "https://foreign.invalid", "Content-Type": "application/json"}, 400),
            ("POST", "/api/problem/demo_force_001/segmentation", {"Origin": "http://" + host, "Content-Type": "text/plain"}, 400),
        ]:
            conn = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                conn.request(method, path, body="{}" if method == "POST" else None, headers=headers)
                response = conn.getresponse()
                assert response.status == expected, response.read()
                response.read()
            finally:
                conn.close()
        assert state_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("name", ["data/private.bin", ".env.local", "settings.local.json", "credentials.json", "notes.docx"])
def test_audit_rejects_sensitive_file_names_even_binary(name):
    assert inspect_content(name, b"\x00\xff")


def test_audit_accepts_empty_template_and_flags_home_paths():
    assert not inspect_content(".env.example", b"OPENAI_API_KEY=\n")
    home = "C:" + "/Users/" + "example-person/project"
    assert inspect_content("config.json", json.dumps({"path": home}).encode())


def test_audit_refuses_unreadable_git_index(tmp_path, monkeypatch):
    import audit_release
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(audit_release, "_git", lambda *args: None)
    _, findings = audit(tmp_path)
    assert any("Git index is unreadable" in label for _, _, label in findings)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git executable is optional")
def test_audit_checks_untracked_and_staged_bytes(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", f"safe.directory={tmp_path.as_posix()}", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init")
    token = "sk-" + "x" * 30
    staged = tmp_path / "staged.json"
    staged.write_text(json.dumps({"key": token}), encoding="utf-8")
    git("add", "staged.json")
    staged.write_text("{}", encoding="utf-8")
    (tmp_path / "untracked.json").write_text(json.dumps({"key": token}), encoding="utf-8")
    private = tmp_path / "data/private.bin"
    private.parent.mkdir()
    private.write_bytes(b"\x00\xff")
    git("add", "data/private.bin")
    _, findings = audit(tmp_path)
    paths = {name for name, _, _ in findings}
    assert "staged.json [index]" in paths
    assert "untracked.json" in paths
    assert "data/private.bin" in paths
