import hashlib
import json
from pathlib import Path
import unittest

from physalign import (
    BindingTarget, EvaluationPlan, OCRRule, Probe, QuantityRule, ReadTarget,
    evaluate, score_response,
)


def ownership(probe_id="p", problem_id="mother", probe_type="quantity", *, readings=None):
    return Probe(probe_id, problem_id, probe_type,
                 BindingTarget("single", {"owner": {"E6": "object_a", "E2": "object_b"}}, ("object_a",)),
                 (ReadTarget("read", "R1", "2 kg"),) if readings is None else readings,
                 packet_nonempty=True)


def outcome(probe, c=1, b=1, *, condition="raw"):
    response = {"owner": "E6" if b else "E2"}
    for r in probe.readings:
        response[r.field] = r.expected if c else "wrong"
    return score_response(probe, json.dumps(response), condition=condition)


class ScoringTests(unittest.TestCase):
    def test_proposal_table_5(self):
        p = ownership()
        cases = [
            ({"read": "2 kg", "owner": "E6"}, (1, 1, 1)),
            ({"read": "2 kg", "owner": "E2"}, (1, 0, 0)),
            ({"read": "3 kg", "owner": "E6"}, (0, 1, 0)),
            ({"read": "3 kg", "owner": "E2"}, (0, 0, 0)),
            ({"read": "2 kg", "owner": "unknown"}, (1, 0, 0)),
            ({"owner": "E6"}, (0, 1, 0)),
            ({"read": "2000 g", "owner": "E6"}, (0, 1, 0)),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                s = score_response(p, json.dumps(response))
                self.assertEqual((s.C, s.B, s.J), expected)

    def test_quantity_conversion_is_an_explicit_separate_task(self):
        rule = QuantityRule({"kg": "1", "g": "0.001"})
        p = ownership(readings=(ReadTarget("read", "R1", "2 kg", rule),))
        self.assertEqual(score_response(p, '{"read":"2000 g","owner":"E6"}').J, 1)
        self.assertEqual(score_response(p, '{"read":"2000 G","owner":"E6"}').J, 0)
        self.assertEqual(score_response(p, '{"read":"2.00001 kg","owner":"E6"}').J, 0)

    def test_exact_quantity_arithmetic_and_declared_tolerance(self):
        rule = QuantityRule({"m": "1", "cm": "0.01"}, "0.001")
        p = ownership(readings=(ReadTarget("read", "R1", "0.3 m", rule),))
        self.assertEqual(score_response(p, '{"read":"30.1 cm","owner":"E6"}').C, 1)
        self.assertEqual(score_response(p, '{"read":"30.1001 cm","owner":"E6"}').C, 0)

    def test_multiple_fields_must_all_pass_at_their_own_anchors(self):
        p = ownership(readings=(ReadTarget("left", "R1", "2 kg"), ReadTarget("right", "R2", "3 kg")))
        good = score_response(p, '{"left":"2 kg","right":"3 kg","owner":"E6"}')
        swapped = score_response(p, '{"left":"3 kg","right":"2 kg","owner":"E6"}')
        missing = score_response(p, '{"left":"2 kg","owner":"E6"}')
        self.assertEqual((good.C, swapped.C, missing.C), (1, 0, 0))
        self.assertEqual((swapped.B, missing.B), (1, 1))

    def test_gold_never_reports_recognition_or_joint_score(self):
        p = ownership()
        for text in ('{"read":"2 kg","owner":"E6"}', '{"owner":"E6"}'):
            s = score_response(p, text, condition="gold")
            self.assertEqual((s.C, s.B, s.J), (None, 1, None))

    def test_binding_only_has_no_vacuously_correct_content(self):
        s = outcome(ownership(readings=()))
        self.assertEqual((s.C, s.B, s.J), (None, 1, None))

    def test_read_failure_does_not_invalidate_binding(self):
        for read in (None, 2, [], {}, "", "\\frac{2}{1} kg"):
            s = score_response(ownership(), json.dumps({"read": read, "owner": "E6"}))
            self.assertEqual((s.C, s.B, s.J), (0, 1, 0))
            self.assertFalse(s.field_valid["read"])

    def test_wrong_but_supported_read_is_not_a_format_failure(self):
        s = score_response(ownership(), '{"read":"3 kg","owner":"E6"}')
        self.assertEqual(s.C, 0)
        self.assertTrue(s.object_valid)
        self.assertTrue(s.field_valid["read"])

    def test_alias_case_is_fixed_and_outer_whitespace_is_allowed(self):
        self.assertEqual(score_response(ownership(), '{"read":"2kg","owner":" E6 "}').J, 1)
        self.assertEqual(score_response(ownership(), '{"read":"2kg","owner":"e6"}').B, 0)

    def test_frozen_ocr_does_not_erase_semantics(self):
        for text in ("-2 kg", "2 Kg", "2.0 kg", "2000 g", "2kg extra", "2 KG"):
            self.assertEqual(score_response(ownership(), json.dumps({"read": text, "owner": "E6"})).C, 0)
        text = json.dumps({"read": "$2\\,\\mathrm{kg}$", "owner": "E6"})
        self.assertEqual(score_response(ownership(), text).C, 1)

    def test_one_optional_markdown_fence(self):
        for fence in ("```json", "```"):
            s = score_response(ownership(), fence + '\n{"read":"2 kg","owner":"E6"}\n```')
            self.assertEqual(s.J, 1)

    def test_invalid_objects_fail_all_fields(self):
        for response in ("I think E6", "[]", "null", "", "{} {}", '{"owner":"E6",}',
                         '{"owner":"E6","owner":"E2"}',
                         '{"read":"2kg","owner":"E6","extra":{"x":1,"x":2}}',
                         '{"read":NaN,"owner":"E6"}',
                         'Answer: {"read":"2kg","owner":"E6"}',
                         '```json\n{"owner":"E6"}\n```\n{"owner":"E2"}'):
            with self.subTest(response=response):
                s = score_response(ownership(), response)
                self.assertEqual((s.C, s.B, s.J, s.object_valid), (0, 0, 0, False))
                self.assertFalse(any(s.field_valid.values()))

    def test_parseable_missing_fields_are_field_failures(self):
        s = score_response(ownership(), "{}")
        self.assertTrue(s.object_valid)
        self.assertEqual((s.C, s.B), (0, 0))

    def test_unrelated_fields_do_not_change_requested_fact(self):
        s = score_response(ownership(), '{"read":"2kg","owner":"E6","unrelated_edge":"wrong"}')
        self.assertEqual(s.J, 1)

    def test_raw_response_required_not_repaired_dict(self):
        with self.assertRaises(TypeError):
            score_response(ownership(), {"owner": "E6"})

    def test_set_exactness_and_f1(self):
        p = Probe("s", "m", "scope", BindingTarget("set", {"subjects": {"E1": "a", "E2": "b", "E3": "c"}}, ("a", "b")))
        for selection, expected, f1 in ((["E2", "E1", "E1"], 1, 1), (["E1"], 0, 2/3),
                                        (["E1", "E2", "E3"], 0, 4/5), ([], 0, 0),
                                        (["E1", "unknown"], 0, 0), ("E1", 0, 0)):
            s = score_response(p, json.dumps({"subjects": selection}))
            self.assertEqual(s.B, expected)
            self.assertAlmostEqual(s.set_f1, f1)

    def test_empty_gold_set_is_not_inferred_from_missing_annotation(self):
        with self.assertRaises(ValueError):
            BindingTarget("set", {"subjects": {"E1": "a", "E2": "b"}}, ())

    def test_direction_and_explicit_symmetry(self):
        entities = {"E1": "a", "E2": "b"}
        domains = {"subject": entities, "predicate": {"R1": "contact", "R2": "contact"}, "object": entities}
        response = '{"subject":"E2","predicate":"R2","object":"E1"}'
        for symmetric in (False, True):
            p = Probe("r", "m", "relation", BindingTarget("relation", domains, ("a", "contact", "b"), symmetric))
            self.assertEqual(score_response(p, response).B, int(symmetric))

    def test_relation_field_order_is_part_of_the_frozen_specification(self):
        entities = {"E1": "a", "E2": "b"}
        forward = {"subject": entities, "predicate": {"R1": "contact"}, "object": entities}
        reordered = {"object": entities, "predicate": {"R1": "contact"}, "subject": entities}
        a = Probe("r", "m", "relation", BindingTarget("relation", forward, ("a", "contact", "b")))
        b = Probe("r", "m", "relation", BindingTarget("relation", reordered, ("a", "contact", "b")))
        self.assertNotEqual(a, b)
        raw = score_response(a, '{"subject":"E1","predicate":"R1","object":"E2"}')
        gold = score_response(b, '{"subject":"E1","predicate":"R1","object":"E2"}', condition="gold")
        with self.assertRaises(ValueError):
            evaluate(EvaluationPlan((a,)), {"r": raw}, {"r": gold})

    def test_alias_permutation_preserves_canonical_prediction(self):
        p = ownership()
        q = Probe("renamed", "mother", "quantity", BindingTarget("single", {"owner": {"E9": "object_a", "E1": "object_b"}}, ("object_a",)), p.readings)
        a = score_response(p, '{"read":"2kg","owner":"E6"}')
        b = score_response(q, '{"read":"2kg","owner":"E9"}')
        self.assertEqual((a.C, a.B, a.J, a.canonical_binding), (b.C, b.B, b.J, b.canonical_binding))

    def test_invalid_gold_is_an_error_not_a_model_failure(self):
        with self.assertRaises(ValueError):
            ReadTarget("read", "R", "", OCRRule())
        with self.assertRaises(ValueError):
            BindingTarget("single", {"owner": {"E1": "a"}}, ("missing",))
        with self.assertRaises(ValueError):
            ownership(readings=(ReadTarget("owner", "R", "2kg"),))


class SyntheticSampleTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "examples/synthetic"

    def test_reference_ocr_is_byte_identical(self):
        local = Path(__file__).resolve().parents[1] / "physalign" / "ocr_normalizers.py"
        self.assertEqual(hashlib.sha256(local.read_bytes()).digest(),
                         hashlib.sha256((self.root / "reference" / "ocr_normalizers.py").read_bytes()).digest())

    def test_all_six_synthetic_sample_keys_and_three_matched_gold_conditions(self):
        answers = json.loads((self.root / "private" / "answers.json").read_text(encoding="utf-8"))
        membership = {m["instance_id"]: m for m in json.loads((self.root / "private" / "membership.json").read_text(encoding="utf-8"))}
        probes, raw, gold = [], {}, {}
        for answer in answers:
            pid = answer["instance_id"]
            key = json.loads((self.root / "private" / "evidence" / pid / "private_key.json").read_text(encoding="utf-8"))
            domain = {a: v["id"] for a, v in key["candidate_map"].items()}
            binding = BindingTarget("single", {key["response_contract"]["binding_key"]: domain},
                                    tuple(t["id"] for t in key["gold_targets"]))
            readings = tuple(ReadTarget("read", t["anchor_alias"], t["expected"], OCRRule(t["normalizer_id"]))
                             for t in key["read_targets"])
            p = Probe(pid, membership[pid]["problem_id"], answer["interface"], binding, readings, bool(readings))
            probes.append(p)
            raw[pid] = score_response(p, json.dumps(answer["answer"]))
            if readings:
                gold[pid] = score_response(p, json.dumps(answer["answer"]), condition="gold")
        report = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertEqual([report["metrics"][k] for k in ("CAcc", "BAcc", "JAcc")], [1, 1, 1])
        self.assertEqual(report["support"]["binding_raw"]["planned_problems"], 3)
        self.assertEqual(report["support"]["joint_raw"]["planned_probes"], 3)
        matched_plan = EvaluationPlan(tuple(p for p in probes if p.joint_eligible))
        matched = evaluate(matched_plan, {p.probe_id: raw[p.probe_id] for p in matched_plan.probes}, gold)
        self.assertEqual(matched["metrics"]["BAcc_gold"], 1)
        self.assertEqual(matched["metrics"]["delta_BAcc_gold_raw"], 0)


if __name__ == "__main__":
    unittest.main()
