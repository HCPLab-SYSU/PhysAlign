"""Load exported probes without changing their prompts, images or candidates.

PublicDataset never opens scoring keys. load_truth is used only by preparation
and offline scoring, never by the inference runner or the model adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Mapping

from .scoring import BindingTarget, OCRRule, Probe, QuantityRule, ReadTarget, score_response
from .storage import canonical, confined, digest, file_hash, fingerprint, loads, read_json

PUBLIC_SCHEMA = "physalign_public_qa_v1"
BUNDLE_SCHEMAS = {"physalign_eval_starter_samples_v1", "physalign_eval_bundle_v1"}
MARKERS = ("Original statement", "Original question", "Original answer options",
           "Evidence locations and candidates", "Images and locator views", "Local question",
           "Output format", "Local reading information")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def image_metadata(data: bytes) -> tuple[str, int, int]:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33 and data[12:16] == b"IHDR":
        return "image/png", *struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        position = 2
        while position < len(data):
            require(data[position] == 0xFF, "Invalid JPEG marker")
            while position < len(data) and data[position] == 0xFF:
                position += 1
            require(position < len(data), "Truncated JPEG marker")
            marker = data[position]
            position += 1
            if marker in {0xD9, 0xDA}:
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            require(position + 2 <= len(data), "Truncated JPEG segment")
            length = int.from_bytes(data[position:position + 2], "big")
            require(length >= 2 and position + length <= len(data), "Invalid JPEG segment length")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                require(length >= 8, "Truncated JPEG frame")
                height, width = struct.unpack(">HH", data[position + 3:position + 7])
                return "image/jpeg", width, height
            position += length
    raise ValueError("Unsupported or malformed image; this export loader accepts PNG/JPEG")


def sections(user: str) -> dict[str, str]:
    positions = []
    for marker in MARKERS:
        hits = list(re.finditer(r"(?m)^\[" + re.escape(marker) + r"\]\n", user))
        require(len(hits) == 1, f"Expected one exact [{marker}] section")
        positions.append((hits[0].start(), hits[0].end()))
    require(positions[0][0] == 0 and positions == sorted(positions), "Public prompt section order changed")
    return {marker: user[end:positions[i + 1][0] if i + 1 < len(positions) else len(user)].removesuffix("\n")
            for i, (marker, (_, end)) in enumerate(zip(MARKERS, positions))}


@dataclass(frozen=True)
class ImageData:
    asset_id: str
    mime_type: str
    width: int
    height: int
    sha256: str
    data: bytes


@dataclass(frozen=True)
class PublicItem:
    """Only immutable serialized PUBLIC input is held here."""
    serialized: str

    @property
    def record(self) -> dict:
        return loads(self.serialized)

    @property
    def key(self) -> tuple[str, str]:
        r = self.record
        return r["instance_id"], r["condition"]

    @property
    def input_hash(self) -> str:
        return fingerprint(self.record["input"])

    @property
    def parts(self) -> dict:
        return sections(self.record["input"]["messages"]["user"])

    @property
    def packet(self) -> list:
        text = self.parts["Local reading information"]
        return [] if text == "None." else loads(text)


class PublicDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        require(self.manifest.get("schema_version") in BUNDLE_SCHEMAS, "Unsupported bundle schema")
        require(self.manifest.get("image_paths_relative_to") == "public", "Image paths must be relative to public/")
        self.inventory = read_json(self.root / "FILE_MANIFEST.json")
        require(isinstance(self.inventory, dict), "FILE_MANIFEST must map relative paths to SHA-256")
        for relative, sha in self.inventory.items():
            confined(self.root, relative)
            require(isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha) is not None, "Invalid inventory hash")
        self.items: dict[tuple[str, str], PublicItem] = {}
        for condition in ("raw", "gold"):
            relative = f"public/qa_{condition}.json"
            if not (self.root / relative).exists():
                require(condition != "raw", "Missing public/qa_raw.json")
                continue
            records = self.read_verified(relative)
            require(isinstance(records, list), f"{relative} must be a JSON array")
            for record in records:
                self._validate_record(record, condition)
                item = PublicItem(canonical(record))
                require(item.key not in self.items, f"Duplicate public item: {item.key}")
                self.items[item.key] = item
        require(any(c == "raw" for _, c in self.items), "No raw probes")
        logical = {}
        for (iid, condition), item in self.items.items():
            r = item.record
            if condition == "raw":
                require(r["logical_probe_id"] not in logical, "One logical probe has multiple raw IDs")
                logical[r["logical_probe_id"]] = iid
            else:
                require((iid, "raw") in self.items, "Gold has no matching Raw probe")
                raw = self.items[iid, "raw"].record
                require({k: v for k, v in raw.items() if k not in {"condition", "input"}} ==
                        {k: v for k, v in r.items() if k not in {"condition", "input"}}, "Raw/Gold metadata differ")
                require(raw["input"]["attachments"] == r["input"]["attachments"], "Raw/Gold attachments/order differ")
                require(raw["input"]["messages"]["system"] == r["input"]["messages"]["system"], "Raw/Gold instructions differ")
                marker = "[Local reading information]\n"
                require(raw["input"]["messages"]["user"].rsplit(marker, 1)[0] ==
                        r["input"]["messages"]["user"].rsplit(marker, 1)[0], "Raw/Gold differ outside the reading packet")

    def verify_file(self, relative: str) -> Path:
        require(relative in self.inventory, f"File absent from frozen inventory: {relative}")
        path = confined(self.root, relative)
        require(file_hash(path) == self.inventory[relative], f"File hash mismatch: {relative}")
        return path

    def read_verified(self, relative: str):
        return read_json(self.verify_file(relative))

    def verify_inventory(self) -> None:
        for relative in self.inventory:
            self.verify_file(relative)

    def _validate_record(self, r: dict, condition: str) -> None:
        allowed = {"schema_version", "instance_id", "logical_probe_id", "task_id", "interface", "language", "split", "condition", "input"}
        require(isinstance(r, dict) and set(r) == allowed, "Unexpected/missing fields in public record")
        require(r["schema_version"] == PUBLIC_SCHEMA and r["condition"] == condition, "Public schema/condition mismatch")
        for key in allowed - {"input"}:
            require(isinstance(r[key], str) and bool(r[key]), f"Invalid public {key}")
        inp = r["input"]
        require(isinstance(inp, dict) and set(inp) == {"messages", "attachments"}, "Unexpected public input fields")
        require(isinstance(inp["messages"], dict) and set(inp["messages"]) == {"system", "user"}, "Exactly one system and one user message required")
        require(all(isinstance(v, str) and v for v in inp["messages"].values()), "Messages must be nonempty strings")
        require(isinstance(inp["attachments"], list) and inp["attachments"], "Multimodal probes require actual image attachments")
        ids = []
        for a in inp["attachments"]:
            require(isinstance(a, dict) and set(a) == {"asset_id", "path", "sha256", "width", "height"}, "Unexpected attachment fields")
            require(isinstance(a["asset_id"], str) and bool(a["asset_id"]), "Invalid asset ID")
            confined(self.root / "public", a["path"])
            relative = "public/" + a["path"]
            require(self.inventory.get(relative) == a["sha256"], "Attachment hash differs from inventory")
            require(all(type(a[k]) is int and a[k] > 0 for k in ("width", "height")), "Invalid declared image dimensions")
            ids.append(a["asset_id"])
        require(len(ids) == len(set(ids)), "Duplicate attachment asset ID")
        parts = sections(inp["messages"]["user"])
        evidence = loads(parts["Evidence locations and candidates"])
        require(isinstance(evidence, dict) and set(evidence) == {"anchors", "candidates"}, "Invalid public evidence structure")
        require(isinstance(evidence["anchors"], dict) and bool(evidence["anchors"]), "At least one explicit evidence anchor is required")
        require(isinstance(evidence["candidates"], list) and len(evidence["candidates"]) >= 2, "At least two candidates required")
        aliases = []
        for candidate in evidence["candidates"]:
            require(set(candidate) == {"alias", "locators"} and isinstance(candidate["alias"], str), "Invalid public candidate")
            require(isinstance(candidate["locators"], list) and candidate["locators"], "Candidate has no public locator")
            aliases.append(candidate["alias"])
            for locator in candidate["locators"]:
                self._validate_locator(locator, ids, parts)
        require(len(aliases) == len(set(aliases)), "Duplicate candidate aliases")
        for locator in evidence["anchors"].values():
            self._validate_locator(locator, ids, parts)
        require(isinstance(loads(parts["Output format"]), dict), "Output format must declare JSON fields")
        packet = [] if parts["Local reading information"] == "None." else loads(parts["Local reading information"])
        require(isinstance(packet, list), "Reading packet must be an array")
        seen = set()
        for entry in packet:
            require(isinstance(entry, dict) and set(entry) == {"anchor_id", "text"}, "Reading packet may contain only anchored transcriptions")
            require(entry["anchor_id"] in evidence["anchors"] and entry["anchor_id"] not in seen, "Unknown/duplicate packet anchor")
            require(isinstance(entry["text"], str) and bool(entry["text"].strip()), "Empty packet reading")
            seen.add(entry["anchor_id"])
        require(condition != "raw" or not packet, "Raw condition already contains a reading packet")

    @staticmethod
    def _validate_locator(locator: dict, asset_ids: list, parts: dict) -> None:
        require(isinstance(locator, dict), "Locator must be an object")
        if locator.get("kind") == "visual":
            require(set(locator) == {"kind", "image_id", "geometry"}, "Unexpected visual locator fields")
            require(locator["image_id"] in asset_ids, "Locator refers to an unattached original image")
            g = locator["geometry"]
            require(set(g) == {"type", "bbox_1000"} and g["type"] == "bbox", "Unsupported geometry contract")
            b = g["bbox_1000"]
            require(isinstance(b, list) and len(b) == 4 and all(type(v) in (int, float) and 0 <= v <= 1000 for v in b), "Invalid normalized bounding box")
            require(b[0] <= b[2] and b[1] <= b[3], "Reversed bounding box")
        elif locator.get("kind") == "text":
            legacy = set(locator) == {"kind", "section", "start", "end", "text"}
            hashed_quote = set(locator) == {"kind", "section", "start", "end", "quote", "section_sha256"}
            require(legacy or hashed_quote, f"Unsupported exact-text locator contract: {sorted(locator)}")
            names = ({"statement": "Original statement", "question": "Original question", "options": "Original answer options"}
                     if legacy else {"stem": "Original statement", "query": "Original question", "options": "Original answer options"})
            require(locator["section"] in names, "Unknown text source section")
            text = parts[names[locator["section"]]]
            if hashed_quote:
                require(locator['section_sha256'] == digest(text.encode('utf-8')), "Text locator section SHA-256 mismatch")
            a, b = locator["start"], locator["end"]
            require(type(a) is int and type(b) is int and 0 <= a < b <= len(text), "Invalid exact text offsets")
            require(text[a:b] == locator["text" if legacy else "quote"], "Text span does not match the unchanged source string")
        else:
            raise ValueError("Unsupported locator kind; an explicit export contract is required")

    def images(self, item: PublicItem) -> tuple[ImageData, ...]:
        result = []
        for a in item.record["input"]["attachments"]:
            data = self.verify_file("public/" + a["path"]).read_bytes()
            require(digest(data) == a["sha256"], "Image changed during loading")
            mime, width, height = image_metadata(data)
            require((width, height) == (a["width"], a["height"]), "Actual image dimensions differ from declared coordinates")
            result.append(ImageData(a["asset_id"], mime, width, height, a["sha256"], data))
        return tuple(result)


def _index(records: list, name: str) -> dict:
    require(isinstance(records, list), f"{name} must be an array")
    result = {}
    for r in records:
        require(isinstance(r, dict) and isinstance(r.get("instance_id"), str), f"Invalid record in {name}")
        require(r["instance_id"] not in result, f"Duplicate instance in {name}")
        result[r["instance_id"]] = r
    return result


def _target(value: dict) -> str:
    require(set(value) == {"kind", "id"} and all(isinstance(v, str) and v for v in value.values()), "Invalid canonical target")
    return canonical([value["kind"], value["id"]])


def load_truth(dataset: PublicDataset) -> tuple[dict[str, Probe], dict[str, dict]]:
    """Scoring-only keys, reconciled with public IDs and final review records.

    Starter records use their final result.json rather than the earlier exporter
    snapshot's pending_review fields. A native bundle can supply private/probes.json
    with explicit scoring contracts; see docs/EVALUATION.md.
    """
    raw = {iid: item for (iid, c), item in dataset.items.items() if c == "raw"}
    if "private/probes.json" in dataset.inventory:
        records = _index(dataset.read_verified("private/probes.json"), "private/probes.json")
        require(set(records) == set(raw), "Native scoring keys and raw instances differ")
        probes, metadata = {}, {}
        for iid, row in records.items():
            item = raw[iid]
            b = row["binding"]
            domains = {d["field"]: d["candidates"] for d in b["domains"]}
            require(len(domains) == len(b["domains"]), "Duplicate binding output fields")
            readings = []
            for r in row["readings"]:
                rule = r["rule"]
                if rule["kind"] == "ocr":
                    normalizer = OCRRule(rule["normalizer_id"])
                elif rule["kind"] == "quantity":
                    normalizer = QuantityRule(rule["unit_scales"], rule.get("absolute_tolerance", "0"))
                else:
                    raise ValueError("Unknown reading rule")
                readings.append(ReadTarget(r["field"], r["anchor_id"], r["expected"], normalizer))
            gold_item = dataset.items.get((iid, "gold"))
            require(row["logical_probe_id"] == item.record["logical_probe_id"], "Native logical probe ID mismatch")
            require(row["split"] == item.record["split"], "Private/public split mismatch")
            require(row["probe_type"] == item.record["task_id"], "Private/public probe type mismatch")
            require(row["review_status"] in {"model_approved", "human_approved"}, "Unapproved native probe")
            require(row["review_status"] != "human_approved" or row["human_reviewed"] is True, "Human approval contradicts review flag")
            require(row["permitted_gold_packet"] == (gold_item.packet if gold_item else None), "Gold packet differs from its frozen permissions")
            probes[iid] = Probe(iid, row["problem_id"], row["probe_type"], BindingTarget(b["kind"], domains, tuple(b["gold"]), b.get("symmetric", False)), tuple(readings), bool(gold_item and gold_item.packet))
            metadata[iid] = {"problem_id": row["problem_id"], "logical_probe_id": row["logical_probe_id"],
                             "review_status": row["review_status"], "human_reviewed": row["human_reviewed"],
                             "cluster_id": row["cluster_id"], "split": row["split"]}
    else:
        require(dataset.manifest["schema_version"] == "physalign_eval_starter_samples_v1", "Native bundle requires private/probes.json")
        answers = _index(dataset.read_verified("private/answers.json"), "answers")
        membership = _index(dataset.read_verified("private/membership.json"), "membership")
        require(set(answers) == set(membership) == set(raw), "Public/private probe sets differ")
        probes, metadata = {}, {}
        for iid, item in raw.items():
            base = "private/evidence/" + iid + "/"
            key = dataset.read_verified(base + "private_key.json")
            review = dataset.read_verified(base + "result.json")
            audit = dataset.read_verified(base + "audit.json")
            blind = dataset.read_verified(base + "blind.json")
            answer, member = answers[iid], membership[iid]
            require(all(r["logical_probe_id"] == item.record["logical_probe_id"] for r in (key, review, answer, member)), "Logical probe identity mismatch")
            require(key["problem_id"] == review["problem_id"] == member["problem_id"], "Mother identity mismatch")
            require(review["status"] == answer["review_status"] == "model_approved", "Unapproved development probe")
            require(review["review_target_root"] == member["review_target_root"] and not review["flags"], "Review scope mismatch or unresolved flags")
            require(audit["record_root"] == review["audit_root"] and audit["response"]["verdict"] == "pass", "Final audit evidence mismatch")
            require(blind["record_root"] == review["blind_root"] and blind["response"]["ambiguous"] is False, "Final blind evidence mismatch or unresolved ambiguity")
            require(answer["interface"] == member["interface"] == item.record["interface"], "Interface mismatch")
            contract = key["response_contract"]
            require(contract["cardinality"] in {"one", "set"}, "Unsupported source cardinality")
            domain = {a: _target(t) for a, t in key["candidate_map"].items()}
            require(set(answer["candidate_ids"]) == set(domain) and len(answer["candidate_ids"]) == len(domain), "Simplified candidate list differs from canonical map")
            readings = tuple(ReadTarget(r.get("field", "read"), r["anchor_alias"], r["expected"], OCRRule(r["normalizer_id"])) for r in key["read_targets"])
            gold_item = dataset.items.get((iid, "gold"))
            if gold_item is not None:
                require(audit["response"]["checks"]["packet_no_binding_leakage"]["status"] == "pass", "No passed packet audit")
                expected = [{"anchor_id": r.anchor_id, "text": r.expected} for r in readings]
                require(gold_item.packet == expected, "Starter Gold packet is not exactly the approved target transcription")
            p = Probe(iid, member["problem_id"], item.record["task_id"],
                      BindingTarget("single" if contract["cardinality"] == "one" else "set", {contract["binding_key"]: domain}, tuple(_target(t) for t in key["gold_targets"])),
                      readings, bool(gold_item and gold_item.packet))
            check = score_response(p, canonical(answer["answer"]))
            require(check.B == 1 and (not p.joint_eligible or check.C == 1), "Simplified answer disagrees with canonical scoring key")
            probes[iid] = p
            metadata[iid] = {"problem_id": p.problem_id, "logical_probe_id": item.record["logical_probe_id"],
                             "review_status": review["status"], "human_reviewed": review["human_reviewed"],
                             "cluster_id": p.problem_id, "split": item.record["split"]}
    for iid, p in probes.items():
        parts = raw[iid].parts
        evidence = loads(parts["Evidence locations and candidates"])
        aliases = {c["alias"] for c in evidence["candidates"]}
        require(aliases == set().union(*(set(d) for d in p.binding.domains.values())), "Private candidate map does not cover exactly the public candidates")
        output = loads(parts["Output format"])
        require(set(output) == set(p.binding.domains) | {r.field for r in p.readings}, "Public/private required output fields differ")
        for r in p.readings:
            require(r.anchor_id in evidence["anchors"], "Private read target has no public anchor")
        for field in p.binding.domains:
            require(isinstance(output[field], list) if p.binding.kind == "set" else isinstance(output[field], str), "Public output cardinality differs from scoring key")
        require(isinstance(metadata[iid]["cluster_id"], str) and bool(metadata[iid]["cluster_id"]), "Missing frozen duplicate/source cluster")
        require(type(metadata[iid]["human_reviewed"]) is bool, "Invalid human review flag")
    if "counts" in dataset.manifest:
        counts = dataset.manifest["counts"]
        actual = {"probes": len(probes), "raw": len(raw),
                  "gold": sum(c == "gold" for _, c in dataset.items),
                  "mothers": len({p.problem_id for p in probes.values()})}
        require(all(counts.get(k) == v for k, v in actual.items()), "Bundle declared counts disagree with actual probe support")
    return probes, metadata
