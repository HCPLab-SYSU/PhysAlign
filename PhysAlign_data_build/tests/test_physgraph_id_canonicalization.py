from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_id_canonicalization import canonicalize_downstream_ids  # noqa: E402
from physgraph_annotation_lib import read_json, read_jsonl  # noqa: E402
from run_high_confidence_physgraph_closed_loop import (  # noqa: E402
    force_verified_pass1_into_pass4,
    validate_downstream,
)


def physical(identifier: str, name: str, anchor: str, mentions: list[str] | None = None) -> dict:
    return {
        "id": identifier, "type": "body", "subtype": name, "name": name, "symbol": "",
        "visual_anchor_ids": [anchor], "text_mention_ids": mentions or [],
        "provenance": ["IMAGE", "TEXT"] if mentions else ["IMAGE"], "confidence": "high",
    }


def binding(identifier: str, from_id: str, to_id: str, mention_ids: list[str] | None = None) -> dict:
    mentions = mention_ids or []
    return {
        "id": identifier, "type": "refers_to" if from_id.startswith("m") else "represents",
        "from_id": from_id, "to_id": to_id,
        "provenance": ["TEXT"] if mentions else ["IMAGE"],
        "evidence_visual_ids": [] if mentions else [from_id],
        "evidence_mention_ids": mentions,
    }


class CanonicalizationTests(unittest.TestCase):
    def test_equivalent_invalid_mentions_are_safely_collapsed_and_references_rewritten(self) -> None:
        problem = {"segments": {"stem": "ions then ions", "query": "find mass"}}
        duplicates = [
            {"id": "m006", "section": "stem", "quote": "ions", "occurrence": 2, "role": "entity"},
            {"id": "m007", "section": "stem", "quote": "ions", "occurrence": 3, "role": "entity"},
            {"id": "m008", "section": "stem", "quote": "ions", "occurrence": 4, "role": "entity"},
        ]
        query = {
            "id": "m009", "section": "query", "quote": "mass",
            "occurrence": 1, "role": "query_target",
        }
        referenced_node = physical("p001", "ions", "v001", ["m006", "m007", "m008"])
        payload = {
            "pass2": {"physical_nodes": [], "bindings": [], "ambiguities": []},
            "pass3": {
                "new_or_updated_physical_nodes": [deepcopy(referenced_node)],
                "text_mentions": [*deepcopy(duplicates), deepcopy(query)],
                "quantities": [], "bindings": [], "constraints": [],
                "query_target": {
                    "mention_id": "m009", "target_kind": "mass", "target_symbol_latex": "m",
                    "target_node_ids": ["p001"], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [],
            },
            "pass4": {
                "physical_nodes": [deepcopy(referenced_node)],
                "text_mentions": [*deepcopy(duplicates), deepcopy(query)],
                "quantities": [], "bindings": [], "relations": [], "constraints": [],
                "query_target": {
                    "mention_id": "m009", "target_kind": "mass", "target_symbol_latex": "m",
                    "target_node_ids": ["p001"], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [],
            },
        }

        normalized, audit = canonicalize_downstream_ids(payload, problem)
        self.assertIsNotNone(audit)
        self.assertEqual(len(normalized["pass3"]["text_mentions"]), 2)
        self.assertEqual(
            normalized["pass3"]["new_or_updated_physical_nodes"][0]["text_mention_ids"],
            ["m001"],
        )
        self.assertEqual(normalized["pass4"]["physical_nodes"][0]["text_mention_ids"], ["m001"])
        self.assertEqual(audit["mention_deduplication"]["pass3"]["status"], "deduplicated")
        self.assertEqual(
            audit["mention_deduplication"]["pass3"]["groups"][0]["removed_ids"],
            ["m007", "m008"],
        )

    def test_equivalent_mentions_are_not_collapsed_when_scalar_usage_distinguishes_them(self) -> None:
        problem = {"segments": {"stem": "ions then ions", "query": "ions"}}
        duplicates = [
            {"id": "m006", "section": "stem", "quote": "ions", "occurrence": 2, "role": "entity"},
            {"id": "m007", "section": "stem", "quote": "ions", "occurrence": 3, "role": "entity"},
        ]
        payload = {
            "pass2": {"physical_nodes": [], "bindings": [], "ambiguities": []},
            "pass3": {
                "new_or_updated_physical_nodes": [], "text_mentions": deepcopy(duplicates),
                "quantities": [], "bindings": [], "constraints": [],
                "query_target": {
                    "mention_id": "m007", "target_kind": "other", "target_symbol_latex": "",
                    "target_node_ids": [], "location_node_ids": [], "time_or_event_node_ids": [],
                }, "ambiguities": [],
            },
            "pass4": {
                "physical_nodes": [], "text_mentions": deepcopy(duplicates),
                "quantities": [], "bindings": [], "relations": [], "constraints": [],
                "query_target": {
                    "mention_id": "m007", "target_kind": "other", "target_symbol_latex": "",
                    "target_node_ids": [], "location_node_ids": [], "time_or_event_node_ids": [],
                }, "ambiguities": [],
            },
        }
        normalized, audit = canonicalize_downstream_ids(payload, problem)
        self.assertEqual(len(normalized["pass3"]["text_mentions"]), 2)
        self.assertEqual(audit["mention_deduplication"]["pass3"]["status"], "no_safe_duplicates")


    def test_mentions_and_provenance_are_repaired_only_from_local_evidence(self) -> None:
        problem = {
            "segments": {
                "stem": "alpha appears before beta",
                "query": "find beta",
            }
        }
        stem_alpha = {
            "id": "m008", "section": "stem", "quote": "alpha",
            "occurrence": 1, "role": "entity",
        }
        stem_beta = {
            "id": "m009", "section": "stem", "quote": "beta",
            "occurrence": 9, "role": "entity",
        }
        query_beta = {
            "id": "m010", "section": "query", "quote": "beta",
            "occurrence": 4, "role": "query_target",
        }
        relation = {
            "id": "r006", "predicate": "before", "subject_id": "p001", "object_id": "p001",
            "quantity_id": "", "provenance": ["TEXT"], "evidence_visual_ids": ["v001"],
            "evidence_mention_ids": ["m008", "m009"], "confidence": "high",
        }
        payload = {
            "pass2": {"physical_nodes": [], "bindings": [], "ambiguities": []},
            "pass3": {
                "new_or_updated_physical_nodes": [],
                "text_mentions": [deepcopy(stem_beta), deepcopy(stem_alpha), deepcopy(query_beta)],
                "quantities": [], "bindings": [], "constraints": [],
                "query_target": {
                    "mention_id": "m010", "target_kind": "other", "target_symbol_latex": "",
                    "target_node_ids": [], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [],
            },
            "pass4": {
                "physical_nodes": [],
                "text_mentions": [deepcopy(stem_beta), deepcopy(stem_alpha), deepcopy(query_beta)],
                "quantities": [], "bindings": [], "relations": [relation], "constraints": [],
                "query_target": {
                    "mention_id": "m010", "target_kind": "other", "target_symbol_latex": "",
                    "target_node_ids": [], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [],
            },
        }

        normalized, audit = canonicalize_downstream_ids(payload, problem)
        self.assertIsNotNone(audit)
        self.assertEqual(
            [(item["id"], item["quote"], item["occurrence"]) for item in normalized["pass3"]["text_mentions"]],
            [("m001", "alpha", 1), ("m002", "beta", 1), ("m003", "beta", 1)],
        )
        self.assertEqual(normalized["pass3"]["query_target"]["mention_id"], "m003")
        self.assertEqual(normalized["pass4"]["relations"][0]["evidence_mention_ids"], ["m001", "m002"])
        self.assertEqual(normalized["pass4"]["relations"][0]["provenance"], ["IMAGE", "TEXT"])
        self.assertEqual(
            audit["mention_ordering"]["pass3"]["occurrence_repairs"][0]["basis"],
            "unique_exact_quote",
        )
        self.assertEqual(len(audit["provenance_repairs"]), 1)


    def test_stage_local_physical_collision_is_resolved_by_semantic_identity(self) -> None:
        object_node = physical("p001", "object", "v001")
        force_node = physical("p003", "force", "v002")
        system_node = physical("p002", "system", "v003", ["m003"])
        payload = {
            "pass2": {
                "problem_id": "p_test",
                "physical_nodes": [deepcopy(object_node), deepcopy(force_node)],
                "bindings": [binding("b004", "v002", "p003")],
                "ambiguities": [{
                    "id": "u005", "scope": "physical", "description": "force",
                    "candidate_ids": ["p003"], "evidence_visual_ids": ["v002"],
                    "evidence_mention_ids": [],
                }],
            },
            "pass3": {
                "problem_id": "p_test",
                "new_or_updated_physical_nodes": [deepcopy(system_node)],
                "text_mentions": [{
                    "id": "m003", "section": "query", "quote": "system",
                    "occurrence": 1, "role": "query_target",
                }],
                "quantities": [{
                    "id": "q005", "kind": "mass", "symbol_latex": "m", "value_latex": "",
                    "unit_latex": "", "owner_id": "p002", "visual_anchor_ids": [],
                    "text_mention_ids": ["m003"], "provenance": ["TEXT"],
                }],
                "bindings": [binding("b006", "m003", "p002", ["m003"])],
                "constraints": [{
                    "id": "c004", "kind": "scope", "subject_ids": ["p002"], "value_text": "whole",
                    "provenance": ["TEXT"], "evidence_visual_ids": [], "evidence_mention_ids": ["m003"],
                }],
                "query_target": {
                    "mention_id": "m003", "target_kind": "mass", "target_symbol_latex": "m",
                    "target_node_ids": ["p002"], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [],
            },
            "pass4": {
                "schema_version": "physgraph-0.1", "problem_id": "p_test", "domain": "mechanics",
                "visual_nodes": [],
                "physical_nodes": [deepcopy(object_node), deepcopy(system_node), deepcopy(force_node)],
                "text_mentions": [{
                    "id": "m003", "section": "query", "quote": "system",
                    "occurrence": 1, "role": "query_target",
                }],
                "quantities": [{
                    "id": "q005", "kind": "mass", "symbol_latex": "m", "value_latex": "",
                    "unit_latex": "", "owner_id": "p002", "visual_anchor_ids": [],
                    "text_mention_ids": ["m003"], "provenance": ["TEXT"],
                }],
                "bindings": [binding("b009", "m003", "p002", ["m003"])],
                "relations": [{
                    "id": "r009", "predicate": "acts_on", "subject_id": "p003", "object_id": "p002",
                    "quantity_id": "q005", "provenance": ["TEXT"], "evidence_visual_ids": [],
                    "evidence_mention_ids": ["m003"], "confidence": "high",
                }],
                "constraints": [{
                    "id": "c004", "kind": "scope", "subject_ids": ["p002"], "value_text": "whole",
                    "provenance": ["TEXT"], "evidence_visual_ids": [], "evidence_mention_ids": ["m003"],
                }],
                "query_target": {
                    "mention_id": "m003", "target_kind": "mass", "target_symbol_latex": "m",
                    "target_node_ids": ["p002"], "location_node_ids": [], "time_or_event_node_ids": [],
                },
                "ambiguities": [{
                    "id": "u005", "scope": "physical", "description": "force",
                    "candidate_ids": ["p003", "q005"], "evidence_visual_ids": ["v002"],
                    "evidence_mention_ids": ["m003"],
                }],
            },
        }

        normalized, audit = canonicalize_downstream_ids(payload)
        self.assertIsNotNone(audit)
        self.assertEqual([node["id"] for node in normalized["pass2"]["physical_nodes"]], ["p001", "p002"])
        self.assertEqual(normalized["pass2"]["bindings"][0]["to_id"], "p002")
        self.assertEqual(normalized["pass3"]["new_or_updated_physical_nodes"][0]["id"], "p003")
        self.assertEqual(normalized["pass3"]["query_target"]["target_node_ids"], ["p003"])
        self.assertEqual(normalized["pass3"]["quantities"][0]["owner_id"], "p003")
        self.assertEqual(
            [(node["id"], node["name"]) for node in normalized["pass4"]["physical_nodes"]],
            [("p001", "object"), ("p002", "force"), ("p003", "system")],
        )
        self.assertEqual(normalized["pass4"]["relations"][0]["subject_id"], "p002")
        self.assertEqual(normalized["pass4"]["relations"][0]["object_id"], "p003")
        self.assertEqual(normalized["pass4"]["relations"][0]["quantity_id"], "q001")
        self.assertEqual(normalized["pass4"]["ambiguities"][0]["candidate_ids"], ["p002", "q001"])
        self.assertEqual(normalized["pass4"]["query_target"]["mention_id"], "m001")
        self.assertEqual(payload["pass2"]["physical_nodes"][1]["id"], "p003", "input must not mutate")

    def test_ambiguous_or_unmatched_pass4_physical_nodes_are_not_guessed(self) -> None:
        payload = {
            "pass2": {"physical_nodes": [physical("p002", "object", "v001")], "bindings": [], "ambiguities": []},
            "pass3": {"new_or_updated_physical_nodes": [], "text_mentions": [], "quantities": [], "bindings": [], "constraints": [], "query_target": {}, "ambiguities": []},
            "pass4": {"physical_nodes": [physical("p009", "different", "v009")], "text_mentions": [], "quantities": [], "bindings": [], "relations": [], "constraints": [], "query_target": {}, "ambiguities": []},
        }
        normalized, audit = canonicalize_downstream_ids(payload)
        self.assertEqual(normalized["pass4"]["physical_nodes"][0]["id"], "p009")
        self.assertEqual(audit["physical_matching"]["pass4_match"], "skipped_unmatched_physical_signature")


if __name__ == "__main__":
    unittest.main()
