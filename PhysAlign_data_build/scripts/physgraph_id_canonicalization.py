"""Deterministic, reference-safe ID canonicalization for PhysGraph Pass 2--4."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

from physgraph_annotation_lib import MENTION_SECTIONS, PROVENANCE, json_sha256


def _ordered_map(records: Any, prefix: str) -> tuple[dict[str, str], str | None]:
    if not isinstance(records, list):
        return {}, "records_not_array"
    identifiers: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            return {}, "record_or_id_invalid"
        identifiers.append(record["id"])
    if len(identifiers) != len(set(identifiers)):
        return {}, "duplicate_source_ids"
    return {old: f"{prefix}{index:03d}" for index, old in enumerate(identifiers, start=1)}, None


def _map_scalar(value: Any, mapping: dict[str, str]) -> Any:
    return mapping.get(value, value) if isinstance(value, str) else value


def _map_list(value: Any, mapping: dict[str, str]) -> Any:
    if not isinstance(value, list):
        return value
    return [_map_scalar(item, mapping) for item in value]


def _combined(*mappings: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for mapping in mappings:
        for old, new in mapping.items():
            if old not in result or result[old] == new:
                result[old] = new
    return result


def _mention_candidates(
    mention: Any,
    problem: dict[str, Any],
) -> tuple[int, list[int], int] | None:
    """Return section, all exact quote positions, and the declared occurrence."""

    if not isinstance(mention, dict):
        return None
    section = mention.get("section")
    quote = mention.get("quote")
    occurrence = mention.get("occurrence")
    segments = problem.get("segments")
    if (
        section not in MENTION_SECTIONS
        or not isinstance(quote, str)
        or not quote
        or not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or occurrence < 1
        or not isinstance(segments, dict)
    ):
        return None
    text = str(segments.get(section, ""))
    positions: list[int] = []
    cursor = 0
    while True:
        start = text.find(quote, cursor)
        if start < 0:
            break
        positions.append(start)
        cursor = start + len(quote)
    if not positions:
        return None
    return MENTION_SECTIONS.index(section), positions, occurrence


def _sort_mentions_by_source(
    records: Any,
    problem: dict[str, Any] | None,
) -> dict[str, Any]:
    """Sort mentions only when every source position is exact and auditable."""

    audit: dict[str, Any] = {
        "status": "not_requested" if problem is None else "not_array",
        "count": len(records) if isinstance(records, list) else 0,
        "changed": False,
        "occurrence_repairs": [],
    }
    if problem is None or not isinstance(records, list):
        return audit
    candidates: list[tuple[int, list[int], int]] = []
    for index, record in enumerate(records):
        candidate = _mention_candidates(record, problem)
        if candidate is None:
            audit["status"] = "skipped_unresolvable_source_position"
            audit["unresolvable_index"] = index
            return audit
        candidates.append(candidate)

    resolved_positions: list[tuple[int, int] | None] = [None] * len(records)
    repaired_occurrences: list[int | None] = [None] * len(records)
    repair_bases: list[str | None] = [None] * len(records)
    for index, (section_index, positions, occurrence) in enumerate(candidates):
        if occurrence <= len(positions):
            resolved_positions[index] = (section_index, positions[occurrence - 1])
        elif len(positions) == 1:
            # A unique exact quote has only one possible occurrence. This repairs
            # a common model error where occurrence was used as a global index.
            resolved_positions[index] = (section_index, positions[0])
            repaired_occurrences[index] = 1
            repair_bases[index] = "unique_exact_quote"

    # A repeated quote with an invalid occurrence is repaired only if its place is
    # uniquely bracketed by already resolved neighboring mentions in the same
    # section. This handles phrases such as two "released from rest" occurrences
    # without silently defaulting every ambiguous quote to its first occurrence.
    for index, resolved in enumerate(resolved_positions):
        if resolved is not None:
            continue
        section_index, positions, _ = candidates[index]
        previous = next(
            (
                position[1]
                for position in reversed(resolved_positions[:index])
                if position is not None and position[0] == section_index
            ),
            None,
        )
        following = next(
            (
                position[1]
                for position in resolved_positions[index + 1:]
                if position is not None and position[0] == section_index
            ),
            None,
        )
        bracketed = [
            (occurrence_index, position)
            for occurrence_index, position in enumerate(positions, start=1)
            if (previous is None or position >= previous)
            and (following is None or position <= following)
        ]
        if len(bracketed) == 1 and (previous is not None or following is not None):
            occurrence_index, position = bracketed[0]
            resolved_positions[index] = (section_index, position)
            repaired_occurrences[index] = occurrence_index
            repair_bases[index] = "unique_neighbor_bracket"

    unresolved_index = next((index for index, value in enumerate(resolved_positions) if value is None), None)
    if unresolved_index is not None:
        audit["status"] = "skipped_unresolvable_source_position"
        audit["unresolvable_index"] = unresolved_index
        return audit

    positioned: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
    for index, (record, position, repaired_occurrence, repair_basis) in enumerate(
        zip(records, resolved_positions, repaired_occurrences, repair_bases)
    ):
        assert position is not None
        if repaired_occurrence is not None:
            before = record.get("occurrence")
            record["occurrence"] = repaired_occurrence
            audit["occurrence_repairs"].append({
                "index": index,
                "before": before,
                "after": repaired_occurrence,
                "basis": repair_basis,
            })
        positioned.append((position, index, record))
    ordered = [record for _, _, record in sorted(positioned, key=lambda item: (item[0], item[1]))]
    order_changed = ordered != records
    if order_changed:
        records[:] = ordered
    changed = order_changed or bool(audit["occurrence_repairs"])
    audit.update(
        status="sorted" if order_changed else "already_in_source_order",
        changed=changed,
        order_changed=order_changed,
    )
    return audit


_MENTION_LIST_REFERENCE_FIELDS = {
    "text_mention_ids", "evidence_mention_ids", "candidate_ids",
}


def _equivalent_mention_reference_usage(
    value: Any,
    group_ids: set[str],
    *,
    key: str | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Require duplicate mention IDs to be used as one inseparable evidence set.

    Scalar references (for example query_target.mention_id or binding endpoints)
    make a group semantically distinguishable and therefore block deduplication.
    For schema-defined mention-ID arrays, every use must contain either none or
    all members of the group. Unknown string occurrences also block the repair.
    """

    usages: list[dict[str, Any]] = []

    def visit(current: Any, path: str, field: str | None) -> bool:
        if isinstance(current, dict):
            for child_key, child in current.items():
                if child_key == "text_mentions":
                    continue
                if not visit(child, f"{path}.{child_key}", child_key):
                    return False
            return True
        if isinstance(current, list):
            present = [item for item in current if isinstance(item, str) and item in group_ids]
            if present:
                if field not in _MENTION_LIST_REFERENCE_FIELDS or set(present) != group_ids:
                    return False
                usages.append({"path": path, "ids": present})
            for index, child in enumerate(current):
                if isinstance(child, (dict, list)) and not visit(child, f"{path}[{index}]", field):
                    return False
            return True
        if isinstance(current, str) and current in group_ids:
            return False
        return True

    return visit(value, "$", key), usages


def _rewrite_equivalent_mention_references(
    value: Any,
    group_ids: set[str],
    keeper_id: str,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "text_mentions":
                continue
            if key in _MENTION_LIST_REFERENCE_FIELDS and isinstance(child, list):
                rewritten: list[Any] = []
                for item in child:
                    replacement = keeper_id if isinstance(item, str) and item in group_ids else item
                    if replacement not in rewritten:
                        rewritten.append(replacement)
                value[key] = rewritten
            else:
                _rewrite_equivalent_mention_references(child, group_ids, keeper_id)
    elif isinstance(value, list):
        for child in value:
            _rewrite_equivalent_mention_references(child, group_ids, keeper_id)


def _deduplicate_equivalent_mentions(
    document: dict[str, Any],
    problem: dict[str, Any] | None,
) -> dict[str, Any]:
    """Remove only provably redundant invalid-occurrence mention copies.

    Models occasionally emit several byte-identical mentions and give the extra
    copies impossible occurrence numbers. They may be collapsed only when one
    member has a valid exact occurrence and every downstream use treats the
    whole group as a single inseparable evidence set. This deliberately does not
    merge valid repeated textual occurrences.
    """

    records = document.get("text_mentions")
    audit: dict[str, Any] = {
        "status": "not_requested" if problem is None else "not_array",
        "changed": False,
        "groups": [],
    }
    if problem is None or not isinstance(records, list):
        return audit

    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        groups[(record.get("section"), record.get("quote"), record.get("role"))].append(record)

    removed_ids: set[str] = set()
    for signature, group in groups.items():
        if len(group) < 2:
            continue
        ids = [record.get("id") for record in group]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            continue
        candidates = [_mention_candidates(record, problem) for record in group]
        if any(candidate is None for candidate in candidates):
            continue
        valid = [
            index
            for index, candidate in enumerate(candidates)
            if candidate is not None and candidate[2] <= len(candidate[1])
        ]
        invalid = [index for index in range(len(group)) if index not in valid]
        # One valid member provides the only auditable keeper. More than one
        # valid occurrence may represent distinct textual evidence and is kept.
        if len(valid) != 1 or not invalid:
            continue
        group_ids = set(ids)
        safe, usages = _equivalent_mention_reference_usage(document, group_ids)
        if not safe:
            continue
        keeper = group[valid[0]]
        keeper_id = keeper["id"]
        removed = [value for value in ids if value != keeper_id]
        _rewrite_equivalent_mention_references(document, group_ids, keeper_id)
        removed_ids.update(removed)
        audit["groups"].append({
            "section": signature[0],
            "quote": signature[1],
            "role": signature[2],
            "keeper_id": keeper_id,
            "removed_ids": removed,
            "basis": "single_valid_occurrence_and_equivalent_reference_usage",
            "reference_usages": usages,
        })

    if removed_ids:
        records[:] = [
            record for record in records
            if not isinstance(record, dict) or record.get("id") not in removed_ids
        ]
        audit["changed"] = True
        audit["status"] = "deduplicated"
    else:
        audit["status"] = "no_safe_duplicates"
    return audit


def _repair_provenance(
    records: Any,
    *,
    path: str,
    visual_field: str,
    mention_field: str,
    repairs: list[dict[str, Any]],
) -> None:
    """Add only provenance labels that are directly entailed by existing evidence IDs."""

    if not isinstance(records, list):
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("provenance"), list):
            continue
        before = list(record["provenance"])
        required: list[str] = []
        if isinstance(record.get(visual_field), list) and record[visual_field]:
            required.append("IMAGE")
        if isinstance(record.get(mention_field), list) and record[mention_field]:
            required.append("TEXT")
        after = list(before)
        for source in required:
            if source not in after:
                after.append(source)
        if after == before:
            continue
        # Keep invalid or duplicate source labels visible to the validator.  When
        # every label is valid, use the schema order for deterministic output.
        if all(source in PROVENANCE for source in after) and len(after) == len(set(after)):
            after = [source for source in PROVENANCE if source in after]
        record["provenance"] = after
        repairs.append({
            "path": f"{path}[{index}].provenance",
            "before": before,
            "after": after,
            "added": [source for source in after if source not in before],
        })


def _repair_payload_provenance(payload: dict[str, Any]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    specs = {
        "pass2": (
            ("physical_nodes", "visual_anchor_ids", "text_mention_ids"),
            ("bindings", "evidence_visual_ids", "evidence_mention_ids"),
        ),
        "pass3": (
            ("new_or_updated_physical_nodes", "visual_anchor_ids", "text_mention_ids"),
            ("quantities", "visual_anchor_ids", "text_mention_ids"),
            ("bindings", "evidence_visual_ids", "evidence_mention_ids"),
            ("constraints", "evidence_visual_ids", "evidence_mention_ids"),
        ),
        "pass4": (
            ("physical_nodes", "visual_anchor_ids", "text_mention_ids"),
            ("quantities", "visual_anchor_ids", "text_mention_ids"),
            ("bindings", "evidence_visual_ids", "evidence_mention_ids"),
            ("relations", "evidence_visual_ids", "evidence_mention_ids"),
            ("constraints", "evidence_visual_ids", "evidence_mention_ids"),
        ),
    }
    for stage, stage_specs in specs.items():
        document = payload.get(stage)
        if not isinstance(document, dict):
            continue
        for field, visual_field, mention_field in stage_specs:
            _repair_provenance(
                document.get(field),
                path=f"$.{stage}.{field}",
                visual_field=visual_field,
                mention_field=mention_field,
                repairs=repairs,
            )
    return repairs


def _rewrite_physical_nodes(records: Any, p_map: dict[str, str], m_map: dict[str, str]) -> None:
    if not isinstance(records, list):
        return
    for node in records:
        if not isinstance(node, dict):
            continue
        node["id"] = _map_scalar(node.get("id"), p_map)
        node["text_mention_ids"] = _map_list(node.get("text_mention_ids"), m_map)


def _rewrite_mentions(records: Any, m_map: dict[str, str]) -> None:
    if not isinstance(records, list):
        return
    for mention in records:
        if isinstance(mention, dict):
            mention["id"] = _map_scalar(mention.get("id"), m_map)


def _rewrite_bindings(
    records: Any,
    b_map: dict[str, str],
    p_map: dict[str, str],
    m_map: dict[str, str],
) -> None:
    if not isinstance(records, list):
        return
    reference_map = _combined(p_map, m_map)
    for binding in records:
        if not isinstance(binding, dict):
            continue
        binding["id"] = _map_scalar(binding.get("id"), b_map)
        binding["from_id"] = _map_scalar(binding.get("from_id"), reference_map)
        binding["to_id"] = _map_scalar(binding.get("to_id"), reference_map)
        binding["evidence_mention_ids"] = _map_list(binding.get("evidence_mention_ids"), m_map)


def _rewrite_quantities(
    records: Any,
    q_map: dict[str, str],
    p_map: dict[str, str],
    m_map: dict[str, str],
) -> None:
    if not isinstance(records, list):
        return
    for quantity in records:
        if not isinstance(quantity, dict):
            continue
        quantity["id"] = _map_scalar(quantity.get("id"), q_map)
        quantity["owner_id"] = _map_scalar(quantity.get("owner_id"), p_map)
        quantity["text_mention_ids"] = _map_list(quantity.get("text_mention_ids"), m_map)


def _rewrite_constraints(
    records: Any,
    c_map: dict[str, str],
    p_map: dict[str, str],
    m_map: dict[str, str],
) -> None:
    if not isinstance(records, list):
        return
    for constraint in records:
        if not isinstance(constraint, dict):
            continue
        constraint["id"] = _map_scalar(constraint.get("id"), c_map)
        constraint["subject_ids"] = _map_list(constraint.get("subject_ids"), p_map)
        constraint["evidence_mention_ids"] = _map_list(constraint.get("evidence_mention_ids"), m_map)


def _rewrite_relations(
    records: Any,
    r_map: dict[str, str],
    p_map: dict[str, str],
    q_map: dict[str, str],
    m_map: dict[str, str],
) -> None:
    if not isinstance(records, list):
        return
    for relation in records:
        if not isinstance(relation, dict):
            continue
        relation["id"] = _map_scalar(relation.get("id"), r_map)
        relation["subject_id"] = _map_scalar(relation.get("subject_id"), p_map)
        relation["object_id"] = _map_scalar(relation.get("object_id"), p_map)
        relation["quantity_id"] = _map_scalar(relation.get("quantity_id"), q_map)
        relation["evidence_mention_ids"] = _map_list(relation.get("evidence_mention_ids"), m_map)


def _rewrite_query_target(target: Any, p_map: dict[str, str], m_map: dict[str, str]) -> None:
    if not isinstance(target, dict):
        return
    target["mention_id"] = _map_scalar(target.get("mention_id"), m_map)
    for field in ("target_node_ids", "location_node_ids", "time_or_event_node_ids"):
        target[field] = _map_list(target.get(field), p_map)


def _rewrite_ambiguities(
    records: Any,
    u_map: dict[str, str],
    p_map: dict[str, str],
    m_map: dict[str, str],
    q_map: dict[str, str],
) -> None:
    if not isinstance(records, list):
        return
    candidate_map = _combined(p_map, m_map, q_map)
    for ambiguity in records:
        if not isinstance(ambiguity, dict):
            continue
        ambiguity["id"] = _map_scalar(ambiguity.get("id"), u_map)
        ambiguity["candidate_ids"] = _map_list(ambiguity.get("candidate_ids"), candidate_map)
        ambiguity["evidence_mention_ids"] = _map_list(ambiguity.get("evidence_mention_ids"), m_map)


def _physical_signature(node: Any) -> tuple[Any, ...] | None:
    if not isinstance(node, dict):
        return None
    anchors = node.get("visual_anchor_ids")
    if not isinstance(anchors, list):
        return None
    return (
        node.get("type"),
        node.get("subtype"),
        node.get("name"),
        node.get("symbol"),
        tuple(anchors),
    )


def _build_physical_maps(
    pass2: dict[str, Any],
    pass3: dict[str, Any],
    pass4: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, Any]]:
    pass2_records = pass2.get("physical_nodes")
    pass3_records = pass3.get("new_or_updated_physical_nodes")
    pass4_records = pass4.get("physical_nodes")
    pass2_map, pass2_error = _ordered_map(pass2_records, "p")
    audit: dict[str, Any] = {"pass2_error": pass2_error, "pass4_match": "not_attempted"}
    if pass2_error is not None or not isinstance(pass3_records, list) or not isinstance(pass4_records, list):
        audit["pass4_match"] = "skipped_invalid_collections"
        return pass2_map, {}, {}, audit

    pass3_ids = [node.get("id") for node in pass3_records if isinstance(node, dict)]
    if len(pass3_ids) != len(pass3_records) or any(not isinstance(value, str) or not value for value in pass3_ids):
        audit["pass3_error"] = "record_or_id_invalid"
        return pass2_map, {}, {}, audit
    if len(pass3_ids) != len(set(pass3_ids)):
        audit["pass3_error"] = "duplicate_source_ids"
        return pass2_map, {}, {}, audit

    pass3_map: dict[str, str] = {}
    next_index = len(pass2_map) + 1
    for old in pass3_ids:
        if old in pass2_map:
            pass3_map[old] = pass2_map[old]
        else:
            pass3_map[old] = f"p{next_index:03d}"
            next_index += 1

    # Build the expected merged entity order. Pass 3 records with an existing
    # Pass 2 ID are updates; otherwise they append a new entity.
    expected: dict[str, dict[str, Any]] = {}
    expected_order: list[str] = []
    if isinstance(pass2_records, list):
        for node in pass2_records:
            canonical_id = pass2_map[node["id"]]
            expected[canonical_id] = deepcopy(node)
            expected[canonical_id]["id"] = canonical_id
            expected_order.append(canonical_id)
    for node in pass3_records:
        canonical_id = pass3_map[node["id"]]
        expected[canonical_id] = deepcopy(node)
        expected[canonical_id]["id"] = canonical_id
        if canonical_id not in expected_order:
            expected_order.append(canonical_id)

    signature_queues: dict[tuple[Any, ...], deque[str]] = defaultdict(deque)
    for canonical_id in expected_order:
        signature = _physical_signature(expected[canonical_id])
        if signature is None:
            audit["pass4_match"] = "skipped_invalid_expected_signature"
            return pass2_map, pass3_map, {}, audit
        signature_queues[signature].append(canonical_id)

    pass4_ids = [node.get("id") for node in pass4_records if isinstance(node, dict)]
    if len(pass4_ids) != len(pass4_records) or any(not isinstance(value, str) or not value for value in pass4_ids):
        audit["pass4_match"] = "skipped_invalid_pass4_ids"
        return pass2_map, pass3_map, {}, audit
    if len(pass4_ids) != len(set(pass4_ids)):
        audit["pass4_match"] = "skipped_duplicate_pass4_ids"
        return pass2_map, pass3_map, {}, audit

    pass4_map: dict[str, str] = {}
    for node in pass4_records:
        signature = _physical_signature(node)
        queue = signature_queues.get(signature) if signature is not None else None
        if not queue:
            audit["pass4_match"] = "skipped_unmatched_physical_signature"
            return pass2_map, pass3_map, {}, audit
        pass4_map[node["id"]] = queue.popleft()
    if any(queue for queue in signature_queues.values()):
        audit["pass4_match"] = "skipped_missing_pass4_entities"
        return pass2_map, pass3_map, {}, audit
    audit["pass4_match"] = "matched_by_semantic_signature_and_occurrence"
    return pass2_map, pass3_map, pass4_map, audit


def _record_map(audit: dict[str, Any], stage: str, kind: str, mapping: dict[str, str], error: str | None = None) -> None:
    stage_entry = audit.setdefault("id_maps", {}).setdefault(stage, {})
    stage_entry[kind] = {
        "changed": {old: new for old, new in mapping.items() if old != new},
        "source_count": len(mapping),
        "error": error or "",
    }


def canonicalize_downstream_ids(
    payload: Any,
    problem: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Canonicalize downstream bookkeeping without guessing semantic content.

    Besides IDs and schema-defined references, the function may sort valid text
    mentions by their exact source positions and add provenance labels that are
    directly entailed by existing evidence IDs. It never edits visual IDs, text,
    geometry, semantic fields, or evidence arrays. Unresolvable input remains for
    the strict validators to reject.
    """
    if not isinstance(payload, dict) or any(not isinstance(payload.get(stage), dict) for stage in ("pass2", "pass3", "pass4")):
        return payload, None
    normalized = deepcopy(payload)
    pass2, pass3, pass4 = (normalized[stage] for stage in ("pass2", "pass3", "pass4"))
    audit: dict[str, Any] = {
        "rule": "schema_aware_deterministic_id_canonicalization",
        "before_sha256": json_sha256(payload),
        "id_maps": {},
    }

    audit["mention_deduplication"] = {
        "pass3": _deduplicate_equivalent_mentions(pass3, problem),
        "pass4": _deduplicate_equivalent_mentions(pass4, problem),
    }
    audit["mention_ordering"] = {
        "pass3": _sort_mentions_by_source(pass3.get("text_mentions"), problem),
        "pass4": _sort_mentions_by_source(pass4.get("text_mentions"), problem),
    }

    p2_map, p3_node_map, p4_map, physical_audit = _build_physical_maps(pass2, pass3, pass4)
    audit["physical_matching"] = physical_audit

    p3_ref_map = _combined(p2_map, p3_node_map)
    collection_specs = {
        "pass2": {"bindings": "b", "ambiguities": "u"},
        "pass3": {"text_mentions": "m", "quantities": "q", "bindings": "b", "constraints": "c", "ambiguities": "u"},
        "pass4": {"text_mentions": "m", "quantities": "q", "bindings": "b", "relations": "r", "constraints": "c", "ambiguities": "u"},
    }
    maps: dict[str, dict[str, dict[str, str]]] = {stage: {} for stage in collection_specs}
    errors: dict[str, dict[str, str | None]] = {stage: {} for stage in collection_specs}
    for stage, specs in collection_specs.items():
        document = normalized[stage]
        for field, prefix in specs.items():
            mapping, error = _ordered_map(document.get(field), prefix)
            maps[stage][prefix] = mapping
            errors[stage][prefix] = error
            _record_map(audit, stage, prefix, mapping, error)

    _record_map(audit, "pass2", "p", p2_map, physical_audit.get("pass2_error"))
    _record_map(audit, "pass3", "p", p3_node_map, physical_audit.get("pass3_error"))
    _record_map(audit, "pass4", "p", p4_map, None if p4_map else physical_audit.get("pass4_match"))

    # Pass 2
    _rewrite_physical_nodes(pass2.get("physical_nodes"), p2_map, {})
    _rewrite_bindings(pass2.get("bindings"), maps["pass2"].get("b", {}), p2_map, {})
    _rewrite_ambiguities(pass2.get("ambiguities"), maps["pass2"].get("u", {}), p2_map, {}, {})

    # Pass 3
    m3, q3 = maps["pass3"].get("m", {}), maps["pass3"].get("q", {})
    _rewrite_mentions(pass3.get("text_mentions"), m3)
    _rewrite_physical_nodes(pass3.get("new_or_updated_physical_nodes"), p3_node_map, m3)
    if isinstance(pass3.get("new_or_updated_physical_nodes"), list):
        pass3["new_or_updated_physical_nodes"].sort(key=lambda node: node.get("id", "") if isinstance(node, dict) else "")
    _rewrite_quantities(pass3.get("quantities"), q3, p3_ref_map, m3)
    _rewrite_bindings(pass3.get("bindings"), maps["pass3"].get("b", {}), p3_ref_map, m3)
    _rewrite_constraints(pass3.get("constraints"), maps["pass3"].get("c", {}), p3_ref_map, m3)
    _rewrite_query_target(pass3.get("query_target"), p3_ref_map, m3)
    _rewrite_ambiguities(pass3.get("ambiguities"), maps["pass3"].get("u", {}), p3_ref_map, m3, q3)

    # Pass 4
    m4, q4 = maps["pass4"].get("m", {}), maps["pass4"].get("q", {})
    _rewrite_mentions(pass4.get("text_mentions"), m4)
    _rewrite_physical_nodes(pass4.get("physical_nodes"), p4_map, m4)
    if p4_map and isinstance(pass4.get("physical_nodes"), list):
        pass4["physical_nodes"].sort(key=lambda node: node.get("id", "") if isinstance(node, dict) else "")
    _rewrite_quantities(pass4.get("quantities"), q4, p4_map, m4)
    _rewrite_bindings(pass4.get("bindings"), maps["pass4"].get("b", {}), p4_map, m4)
    _rewrite_relations(pass4.get("relations"), maps["pass4"].get("r", {}), p4_map, q4, m4)
    _rewrite_constraints(pass4.get("constraints"), maps["pass4"].get("c", {}), p4_map, m4)
    _rewrite_query_target(pass4.get("query_target"), p4_map, m4)
    _rewrite_ambiguities(pass4.get("ambiguities"), maps["pass4"].get("u", {}), p4_map, m4, q4)

    audit["provenance_repairs"] = _repair_payload_provenance(normalized)

    audit["after_sha256"] = json_sha256(normalized)
    audit["changed"] = normalized != payload
    return (normalized, audit) if normalized != payload else (payload, None)
