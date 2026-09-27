"""Tested safety primitives for PhysAlign v2; NOT a complete production exporter.

Only stdlib is needed for most functions. verify_image_assets additionally uses
Pillow already installed by the caller. SourceAdapter, image rendering, semantic
adjudication and authenticated review storage must be implemented in the real
repository. Structural checks cannot establish physical correctness/no leakage.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class ContractError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ContractError(code, detail)


def exact_keys(value: Any, required: Iterable[str], optional: Iterable[str] = ()) -> None:
    require(isinstance(value, dict), "OBJECT_REQUIRED")
    req, opt = set(required), set(optional)
    require(req <= set(value) and set(value) <= req | opt,
            "PUBLIC_FIELD_FORBIDDEN", f"expected {sorted(req)}, got {sorted(value)}")


def canonical_bytes(value: Any) -> bytes:
    # This is a named Python serialization contract, not a claim of RFC 8785.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_matches(text: str, quote: str) -> list[list[int]]:
    require(isinstance(text, str) and isinstance(quote, str) and bool(quote), "EMPTY_OR_INVALID_QUOTE")
    result, pos = [], 0
    while True:
        start = text.find(quote, pos)
        if start < 0:
            return result
        end = start + len(quote)
        result.append([start, end])
        pos = end  # literal_nonoverlap_1based_v1


def resolve_span(text: str, quote: str, occurrence: int, *,
                 unique_policy_approved: bool = False) -> dict[str, Any]:
    """Do not mutate input. Policy approval must be verified by the real caller."""
    require(type(occurrence) is int and occurrence > 0, "OCCURRENCE_SCHEMA_INVALID")
    matches = exact_matches(text, quote)
    base = {"section_sha256": text_digest(text), "quote": quote,
            "original_occurrence": occurrence, "all_matches": matches,
            "match_policy": "literal_nonoverlap_1based_v1", "span": None}
    if occurrence <= len(matches):
        base.update(status="resolved_exact", span=matches[occurrence - 1])
    elif len(matches) == 1:
        base.update(status="repaired_unique_exact" if unique_policy_approved else "unique_exact_repair_proposed",
                    proposed_span=matches[0], corrected_occurrence=1,
                    repair_policy="unique_exact_sidecar_v1")
        if unique_policy_approved:
            base["span"] = matches[0]
    else:
        base["status"] = "ambiguous_occurrence" if matches else "quote_not_found"
    base["decision_content_hash"] = digest(base)
    return base


def text_locator_view(text: str, start: int, end: int, alias: str,
                      *, context_chars: int = 50) -> dict[str, str]:
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "TEXT_RANGE_INVALID")
    require(bool(re.fullmatch(r"[REH][1-9][0-9]*", alias)), "INVALID_ALIAS")
    require(type(context_chars) is int and context_chars >= 0, "INVALID_CONTEXT_WINDOW")
    # Structured segments avoid relying on model-side counting or HTML execution.
    return {"alias": alias, "prefix": text[max(0, start-context_chars):start],
            "selected": text[start:end], "suffix": text[end:end+context_chars]}


def group_resolved_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Input: all resolved refers_to rows in the considered scope.

    canonical_target and identity_state must already follow approved identities.
    This routine does NOT infer identity or prove annotations are exhaustive.
    Unresolved related rows must block the call/scope, not be filtered out first.
    """
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(row.get("closure_state") == "resolved", "TARGET_CLOSURE_UNRESOLVED")
        require(row.get("identity_state") == "resolved", "IDENTITY_UNRESOLVED")
        key = {k: row[k] for k in ("problem_id", "section_sha256", "section", "start", "end", "public_scope")}
        require(type(key["start"]) is int and type(key["end"]) is int and 0 <= key["start"] < key["end"], "TEXT_RANGE_INVALID")
        token = digest(key)
        if token not in groups:
            groups[token] = {"query_key": key, "source_binding_ids": [], "source_mention_ids": [], "targets": []}
        g = groups[token]
        g["source_binding_ids"].append(row["binding_id"])
        g["source_mention_ids"].append(row["mention_id"])
        g["targets"].append(row["canonical_target"])
    result = []
    for token in sorted(groups):
        g = groups[token]
        for field in ("source_binding_ids", "source_mention_ids", "targets"):
            g[field] = sorted(set(g[field]))
        require(bool(g["targets"]), "EMPTY_TARGET_SET")
        g["cardinality"] = "one" if len(g["targets"]) == 1 else "set"
        result.append(g)
    return result


def validate_context(context: dict[str, Any]) -> None:
    exact_keys(context, ("stem", "query", "options", "image_ids"))
    require(isinstance(context["stem"], str) and isinstance(context["query"], str), "CONTEXT_TEXT_INVALID")
    require(isinstance(context["options"], list), "OPTIONS_INVALID")
    seen = set()
    for option in context["options"]:
        exact_keys(option, ("id", "text"))
        require(isinstance(option["id"], str) and bool(option["id"]) and isinstance(option["text"], str), "OPTIONS_INVALID")
        require(option["id"] not in seen, "DUPLICATE_OPTION_ID")
        seen.add(option["id"])
    images = context["image_ids"]
    require(isinstance(images, list) and all(isinstance(i, str) and i for i in images), "IMAGE_IDS_INVALID")
    require(len(images) == len(set(images)), "DUPLICATE_IMAGE_ID")


def finite_number(x: Any) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def point(p: Any) -> None:
    require(isinstance(p, list) and len(p) == 2 and all(finite_number(x) and 0 <= x <= 1000 for x in p), "GEOMETRY_INVALID")


def validate_geometry(g: dict[str, Any], width: int, height: int) -> None:
    require(isinstance(g, dict), "GEOMETRY_INVALID")
    kind = g.get("type")
    if kind == "bbox":
        exact_keys(g, ("type", "bbox_1000"))
        b = g["bbox_1000"]
        require(isinstance(b, list) and len(b) == 4 and all(finite_number(x) for x in b), "GEOMETRY_INVALID")
        require(0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000, "GEOMETRY_INVALID")
    elif kind == "point":
        exact_keys(g, ("type", "xy_1000")); point(g["xy_1000"])
    elif kind in ("polyline", "polygon"):
        exact_keys(g, ("type", "points_1000"))
        pts = g["points_1000"]
        require(isinstance(pts, list) and len(pts) >= (2 if kind == "polyline" else 3), "GEOMETRY_INVALID")
        for p in pts:
            point(p)
        require(len(set(map(tuple, pts))) >= (2 if kind == "polyline" else 3), "GEOMETRY_INVALID")
        if kind == "polygon":
            area2 = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(pts, pts[1:]+pts[:1]))
            require(abs(area2) > 1e-9, "GEOMETRY_INVALID")
            # Self-intersection and semantic shape validity require the renderer/audit.
    elif kind == "circle":
        exact_keys(g, ("type", "center_1000", "radius_1000_min_dim"))
        point(g["center_1000"])
        r = g["radius_1000_min_dim"]
        require(finite_number(r) and r > 0, "GEOMETRY_INVALID")
        cx, cy = g["center_1000"]
        rx, ry = r*min(width, height)/width, r*min(width, height)/height
        require(0 <= cx-rx < cx+rx <= 1000 and 0 <= cy-ry < cy+ry <= 1000, "GEOMETRY_INVALID")
    else:
        raise ContractError("UNSUPPORTED_GEOMETRY", str(kind))


def validate_locator(loc: dict[str, Any], context: dict[str, Any], assets: dict[str, Any]) -> None:
    require(isinstance(loc, dict), "LOCATOR_INVALID")
    if loc.get("kind") == "visual":
        exact_keys(loc, ("kind", "image_id", "geometry"))
        image = loc["image_id"]
        require(image in context["image_ids"] and image in assets, "ASSET_MISSING_OR_UNLISTED", str(image))
        a = assets[image]
        require(type(a.get("width")) is int and type(a.get("height")) is int and a["width"] > 0 and a["height"] > 0, "ASSET_DIMENSIONS_INVALID")
        validate_geometry(loc["geometry"], a["width"], a["height"])
    elif loc.get("kind") == "text":
        exact_keys(loc, ("kind", "section", "start", "end", "quote", "section_sha256"))
        section = loc["section"]
        require(section in ("stem", "query"), "TEXT_SECTION_INVALID")
        text = context[section]; s, e = loc["start"], loc["end"]
        require(type(s) is int and type(e) is int and 0 <= s < e <= len(text), "TEXT_RANGE_INVALID")
        require(text_digest(text) == loc["section_sha256"], "SOURCE_TEXT_STALE")
        require(isinstance(loc["quote"], str) and text[s:e] == loc["quote"], "TEXT_QUOTE_MISMATCH")
    else:
        raise ContractError("LOCATOR_INVALID")


def view_fingerprint(locators: list[dict[str, Any]]) -> str:
    # This detects exact duplicate logical displays, not all perceptually
    # indistinguishable renderings. Rendered-pixel review is still required.
    normalized = sorted({canonical_bytes(loc).decode("utf-8") for loc in locators})
    return digest(normalized)


def validate_generated_question_references(question: str, references: dict[str, Any]) -> None:
    """Call ONLY on compiler-generated question text, not original source text."""
    used = set(re.findall(r"(?<![A-Za-z0-9_])H[1-9][0-9]*(?![A-Za-z0-9_])", question))
    require(used == set(references), "UNBOUND_REFERENCE")


def validate_view_bundle(bundle: dict[str, Any], assets: dict[str, Any],
                         role_alias_labels: dict[str, str] | None = None) -> None:
    """Validate the declared view sub-contract, NOT an entire FrozenProbe.

    role_alias_labels is independently looked up from an approved registry by
    the caller; it must not be derived from the untrusted candidate labels.
    """
    exact_keys(bundle, ("task_id", "language", "context", "anchors", "query_anchor_ids", "read_anchor_ids",
                        "reference_views", "used_reference_ids", "candidates", "recognition_packet"))
    require(bundle["language"] in ("en", "zh"), "LANGUAGE_UNRESOLVED")
    require(bundle["task_id"] in ("T01", "T02", "T03", "T04", "T05", "T06"), "UNKNOWN_TASK")
    context = bundle["context"]; validate_context(context)
    anchors = bundle["anchors"]
    require(isinstance(anchors, dict), "ANCHORS_INVALID")
    for alias, loc in anchors.items():
        require(bool(re.fullmatch(r"R[1-9][0-9]*", alias)), "INVALID_ALIAS")
        validate_locator(loc, context, assets)
    for field in ("query_anchor_ids", "read_anchor_ids"):
        refs = bundle[field]
        require(isinstance(refs, list) and all(isinstance(x, str) and x in anchors for x in refs)
                and len(refs) == len(set(refs)), "ANCHOR_REFERENCE_INVALID")
    refs = bundle["reference_views"]
    require(isinstance(refs, dict), "REFERENCES_INVALID")
    used = bundle["used_reference_ids"]
    require(isinstance(used, list) and all(isinstance(x, str) for x in used)
            and len(used) == len(set(used)) and set(used) == set(refs), "UNBOUND_REFERENCE")
    for alias, locs in refs.items():
        require(bool(re.fullmatch(r"H[1-9][0-9]*", alias)), "INVALID_ALIAS")
        require(isinstance(locs, list) and bool(locs), "REFERENCE_LOCATOR_MISSING")
        for loc in locs:
            validate_locator(loc, context, assets)
    candidates = bundle["candidates"]
    require(isinstance(candidates, list) and len(candidates) >= 2, "CANDIDATE_COUNT_INVALID")
    aliases, fingerprints = set(), set()
    for c in candidates:
        require(isinstance(c, dict), "CANDIDATE_INVALID")
        domain = "role" if bundle["task_id"] == "T01" else ("evidence" if bundle["task_id"] == "T06" else "entity")
        fields = ("alias", "kind", "label") if domain == "role" else ("alias", "kind", "locators")
        exact_keys(c, fields)
        alias = c["alias"]
        require(isinstance(alias, str) and bool(re.fullmatch(r"E[1-9][0-9]*", alias)) and alias not in aliases, "INVALID_OR_DUPLICATE_ALIAS")
        aliases.add(alias)
        require(c["kind"] == domain, "CANDIDATE_DOMAIN_INVALID")
        if domain == "role":
            require(role_alias_labels is not None and alias in role_alias_labels
                    and c["label"] == role_alias_labels[alias], "ROLE_LABEL_MISMATCH")
            fingerprint = digest(c["label"])
        else:
            locs = c["locators"]
            require(isinstance(locs, list) and bool(locs), "CANDIDATE_LOCATOR_MISSING")
            for loc in locs:
                validate_locator(loc, context, assets)
            fingerprint = view_fingerprint(locs)
        require(fingerprint not in fingerprints, "INDISTINGUISHABLE_CANDIDATES")
        fingerprints.add(fingerprint)
    packet = bundle["recognition_packet"]
    require(isinstance(packet, list), "PACKET_INVALID")
    seen = set()
    for entry in packet:
        exact_keys(entry, ("anchor", "text"))
        require(entry["anchor"] in anchors and entry["anchor"] not in seen
                and isinstance(entry["text"], str) and bool(entry["text"].strip()), "PACKET_INVALID")
        seen.add(entry["anchor"])
    if bundle["task_id"] == "T06":
        require(not anchors and not refs and not bundle["query_anchor_ids"] and not bundle["read_anchor_ids"] and not packet,
                "T06_PUBLIC_SELECTION_CUE")


def build_packet(plan: dict[str, Any], observations: dict[str, Any],
                 observation_to_public_anchor: dict[str, str], *,
                 public_bundle: dict[str, Any], assets: dict[str, Any],
                 anchor_map: dict[str, Any], preview: bool = False) -> list[dict[str, str]]:
    """Copies exact text only after source-version and R-position equality checks.

    The caller verifies selection symmetry/semantics and that its review_ref is
    valid for this content. A nonempty string review_ref alone is not proof.
    """
    exact_keys(plan, ("state", "observation_ids", "empty_reason", "review_ref"))
    state, ids = plan["state"], plan["observation_ids"]
    require(state in ("approved_nonempty", "approved_empty") or (preview and state == "pending_review"), "PACKET_PENDING_OR_REJECTED")
    if state != "pending_review":
        require(isinstance(plan["review_ref"], str) and bool(plan["review_ref"]), "PACKET_REVIEW_MISSING")
    require(isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)), "PACKET_INVALID")
    if preview and state == "pending_review" and not ids:
        return []
    if state == "approved_empty":
        require(not ids and plan["empty_reason"] in (
            "no_independent_readable_primitive", "only_unsafe_binding_statement", "family_policy_no_packet"), "PACKET_EMPTY_REASON_INVALID")
        return []
    require(bool(ids) and plan["empty_reason"] is None, "PACKET_STATE_CONTENT_MISMATCH")
    result, seen_anchors = [], set()
    for oid in ids:
        require(oid in observations and oid in observation_to_public_anchor, "OBSERVATION_MISSING")
        obs = observations[oid]
        require(obs.get("state") == "approved" and isinstance(obs.get("text"), str), "OBSERVATION_UNAPPROVED")
        anchor = observation_to_public_anchor[oid]
        from reference_integrity import validate_observation_location
        validate_observation_location(oid, obs, anchor, public_bundle, assets, anchor_map)
        require(anchor not in seen_anchors, "DUPLICATE_PACKET_ANCHOR")
        seen_anchors.add(anchor)
        result.append({"anchor": anchor, "text": obs["text"]})
    return result


def validate_pair(raw: dict[str, Any], gold: dict[str, Any], expected_packet: list[dict[str, str]]) -> None:
    require(raw.get("recognition_packet") == [], "RAW_HAS_PACKET")
    r, g = copy.deepcopy(raw), copy.deepcopy(gold)
    r.pop("recognition_packet", None); g.pop("recognition_packet", None)
    require(r == g, "RAW_GOLD_UNMATCHED")
    require(gold.get("recognition_packet") == expected_packet, "PACKET_TEXT_MISMATCH")


def validate_relation_registration(registry: dict[str, Any], source_namespace: str,
                                   raw_predicate: str) -> dict[str, Any]:
    # Schema+direction checks are mandatory even for a helper lookup. Review
    # authentication is performed by the integrated path with a trusted ledger.
    from reference_registry import resolve_relation
    return resolve_relation(registry, source_namespace, raw_predicate)


def validate_review_binding(review: dict[str, Any], content_root: str) -> None:
    # Authenticating/authorizing the reviewer is the real ledger's responsibility.
    require(review.get("status") == "approved" and review.get("content_root") == content_root,
            "AUDIT_STALE_OR_UNAPPROVED")
    require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"]), "REVIEWER_MISSING")


def logical_id(identity: dict[str, Any]) -> str:
    # The real compiler supplies only stable source-query identity, not content.
    return "PAQ-" + digest(identity)


def instance_id(content_core: dict[str, Any], approved_review_snapshot: dict[str, Any]) -> str:
    root = digest(content_core)
    validate_review_binding(approved_review_snapshot, root)
    return "PAI-" + digest({"content_root": root, "review": approved_review_snapshot})


def validate_batch_identity(items: list[dict[str, Any]]) -> None:
    seen = set()
    for item in items:
        require(isinstance(item, dict) and all(isinstance(item.get(k), str) and bool(item[k].strip()) for k in ("logical_probe_id", "variant_id", "language")), "BATCH_IDENTITY_FIELDS_MISSING")
        require(item["language"] in ("en", "zh"), "LANGUAGE_UNRESOLVED")
        key = (item["logical_probe_id"], item["variant_id"], item["language"])
        require(key not in seen, "DUPLICATE_LOGICAL_ID")
        seen.add(key)


def verify_image_assets(assets: dict[str, Any], root: Path) -> None:
    """Verify actual file bytes/decoded pixels. Orientation here is 'stored_pixels'.

    A repository using EXIF-transposed pixels must register a different contract
    and transform all anchors consistently, rather than silently transposing.
    """
    from PIL import Image
    root = root.resolve()
    for image_id, record in assets.items():
        path = (root / record["relative_path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "ASSET_MISSING", image_id)
        require(record.get("orientation_policy") == "stored_pixels", "ORIENTATION_POLICY_UNREGISTERED")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["bytes_sha256"], "ASSET_HASH_MISMATCH")
        with Image.open(path) as img:
            img.load()
            require(img.size == (record["width"], record["height"]) and img.mode == record["mode"], "ASSET_DIMENSIONS_INVALID")
            pixel_blob = canonical_bytes({"mode": img.mode, "size": list(img.size)}) + b"\x00" + img.tobytes()
            require(hashlib.sha256(pixel_blob).hexdigest() == record["pixels_sha256"], "ASSET_PIXEL_HASH_MISMATCH")


def validate_attachment_coverage(required: dict[str, str], supplied: dict[str, str]) -> None:
    """asset_id -> actual attached bytes SHA256. All extras require declaration."""
    require(required == supplied, "ATTACHMENT_MANIFEST_MISMATCH")


def select_atomic_observations(observations: dict[str, Any], presented_diagram_ids: set[str],
                               policy: dict[str, Any]) -> dict[str, Any]:
    """Target-independent lexical selection, NOT semantic leakage approval.

    No gold targets, owners, labels-to-entity mappings or candidate correctness
    are accepted by this function. Zero selections do not imply approved_empty.
    """
    grammar = [re.compile(p) for p in policy["patterns"]]
    max_entries = policy["max_entries"]
    require(type(max_entries) is int and max_entries > 0, "PACKET_BUDGET_INVALID")
    selected, excluded = [], {}
    for oid in sorted(observations):
        obs = observations[oid]
        if obs.get("asset_role") != "diagram" or obs.get("image_id") not in presented_diagram_ids or obs.get("source_kind") != "text_glyph":
            excluded[oid] = "outside_declared_observation_scope"
        elif obs.get("state") != "approved":
            excluded[oid] = "observation_not_approved"
        elif not isinstance(obs.get("text"), str) or not any(p.fullmatch(obs["text"].strip()) for p in grammar):
            excluded[oid] = "policy_lexical_filter"
        else:
            selected.append(oid)
    require(len(selected) <= max_entries, "PACKET_BUDGET_EXCEEDED")
    return {"observation_ids": selected, "excluded": excluded,
            "packet_state": "pending_review", "selection_only_not_semantic_approval": True}

