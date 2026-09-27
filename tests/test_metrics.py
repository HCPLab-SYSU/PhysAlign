from fractions import Fraction
import json
import random
import unittest

from physalign import (
    BindingTarget, EvaluationPlan, Pool, Probe, evaluate, option_diagnostic,
    score_response, solving_association, uniform_candidate_baseline,
)
from test_scoring import ownership, outcome


class MetricsTests(unittest.TestCase):
    def test_100_item_worked_example(self):
        probes = [ownership(str(i)) for i in range(100)]
        raw, gold = {}, {}
        for i, p in enumerate(probes):
            c, b = (1, 1) if i < 70 else (1, 0) if i < 90 else (0, 1) if i < 95 else (0, 0)
            raw[p.probe_id] = outcome(p, c, b)
            gold[p.probe_id] = outcome(p, 1, int(i < 85), condition="gold")
        report = evaluate(EvaluationPlan(tuple(probes)), raw, gold)
        m = report["metrics"]
        for key, expected in {"CAcc": .9, "BAcc": .75, "JAcc": .7, "BAcc_gold": .85,
                              "BAcc_L": .75, "delta_BAcc_gold_raw": .1,
                              "BAcc_given_C": 7/9, "BindErr_given_C": 2/9}.items():
            self.assertAlmostEqual(m[key], expected)
        self.assertNotAlmostEqual(m["JAcc"], m["CAcc"] * m["BAcc"])
        for key, expected in {"q11": .7, "q10": .2, "q01": .05, "q00": .05}.items():
            self.assertAlmostEqual(report["quadrants"][key], expected)
        self.assertEqual(report["support"]["joint_raw"]["correctly_read_probes"], 90)

    def test_20_versus_4_probes_are_two_equal_mothers(self):
        probes = [ownership(str(i), "many" if i < 20 else "few") for i in range(24)]
        raw = {p.probe_id: outcome(p, int(i < 20), int(i < 20)) for i, p in enumerate(probes)}
        result = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertEqual(result["metrics"]["BAcc"], .5)
        self.assertEqual(result["metrics"]["JAcc"], .5)
        self.assertNotEqual(result["metrics"]["BAcc"], 20/24)

    def test_type_then_mother_order_with_unequal_type_support(self):
        probes = [ownership("a1", "m1", "a"), ownership("a2", "m2", "a")]
        probes += [ownership("b" + str(i), "m1", "b") for i in range(9)]
        raw = {p.probe_id: outcome(p, 1, int(p.probe_id == "a1")) for p in probes}
        result = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertEqual(result["metrics"]["BAcc"], .25)
        self.assertEqual(result["by_type"]["a"]["BAcc"], .5)
        self.assertEqual(result["by_type"]["b"]["BAcc"], 0)

    def test_custom_type_weights_are_used_without_renormalization(self):
        probes = (ownership("a", "m", "a"), ownership("b", "m", "b"))
        raw = {p.probe_id: outcome(p, 1, int(p.probe_type == "a")) for p in probes}
        result = evaluate(EvaluationPlan(probes, {"a": .8, "b": .2}, {"a": .3, "b": .7}), raw)
        self.assertAlmostEqual(result["metrics"]["BAcc"], .8)
        self.assertAlmostEqual(result["metrics"]["BAcc_L"], .3)
        self.assertAlmostEqual(result["metrics"]["JAcc"], .3)

    def test_conditional_ratio_after_aggregation_not_average_of_ratios(self):
        probes = [ownership("a" + str(i), "m1", "a") for i in range(10)]
        probes += [ownership("b" + str(i), "m2", "b") for i in range(10)]
        raw = {p.probe_id: outcome(p, int(p.probe_type == "b" or p.probe_id == "a0"),
                                  int(p.probe_type == "a")) for p in probes}
        m = evaluate(EvaluationPlan(tuple(probes)), raw)["metrics"]
        self.assertAlmostEqual(m["CAcc"], .55)
        self.assertAlmostEqual(m["JAcc"], .05)
        self.assertAlmostEqual(m["BAcc_given_C"], 1/11)
        self.assertNotAlmostEqual(m["BAcc_given_C"], .5)
        report = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertEqual(report["by_type"]["a"]["support"]["joint_raw"]["correctly_read_probes"], 1)
        self.assertEqual(report["by_type"]["b"]["support"]["joint_raw"]["correctly_read_probes"], 10)

    def test_binding_only_items_cannot_pollute_joint_quadrants(self):
        probes = [ownership("joint")]
        probes += [ownership(str(i), readings=()) for i in range(9)]
        raw = {p.probe_id: outcome(p, 1, int(not p.joint_eligible)) for p in probes}
        report = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertAlmostEqual(report["metrics"]["BAcc"], .9)
        self.assertEqual(report["metrics"]["BAcc_L"], 0)
        self.assertEqual(report["quadrants"], {"q11": 0, "q10": 1, "q01": 0, "q00": 0})

    def test_zero_content_denominator_is_undefined(self):
        p = ownership()
        report = evaluate(EvaluationPlan((p,)), {p.probe_id: outcome(p, 0, 1)})
        self.assertIsNone(report["metrics"]["BAcc_given_C"])
        self.assertIsNone(report["metrics"]["BindErr_given_C"])
        self.assertEqual(report["quadrants"]["q01"], 1)

    def test_empty_joint_pool_is_not_zero_accuracy(self):
        p = ownership(readings=())
        report = evaluate(EvaluationPlan((p,)), {p.probe_id: outcome(p)})
        self.assertEqual(report["metrics"]["BAcc"], 1)
        self.assertIsNone(report["metrics"]["CAcc"])
        self.assertIsNone(report["metrics"]["JAcc"])
        self.assertEqual(report["type_weights"]["joint"], {})

    def test_invalid_outputs_remain_in_denominator(self):
        a, b = ownership("a", "m1"), ownership("b", "m2")
        report = evaluate(EvaluationPlan((a, b)), {"a": outcome(a), "b": score_response(b, "invalid")})
        self.assertEqual(report["metrics"]["BAcc"], .5)
        self.assertEqual(report["invalidity"]["raw"]["object_invalid_rate_served"], .5)
        self.assertEqual(report["support"]["binding_raw"]["missing_probes"], 0)

    def test_missing_infrastructure_preserves_weights_and_bounds(self):
        probes = [ownership(str(i), "many" if i < 20 else "few") for i in range(21)]
        raw = {p.probe_id: None if i < 20 else outcome(p) for i, p in enumerate(probes)}
        report = evaluate(EvaluationPlan(tuple(probes)), raw)
        self.assertIsNone(report["metrics"]["BAcc"])
        self.assertEqual(report["summaries"]["BAcc"]["lower"], .5)
        self.assertEqual(report["summaries"]["BAcc"]["upper"], 1)
        self.assertEqual(report["summaries"]["BAcc"]["weighted_service_coverage"], .5)
        self.assertEqual(report["support"]["binding_raw"]["service_coverage"], 1/21)

    def test_missing_gold_pair_is_not_silently_intersected(self):
        a, b = ownership("a", "m1"), ownership("b", "m2")
        raw = {"a": outcome(a), "b": outcome(b, 0, 0)}
        gold = {"a": None, "b": outcome(b, 1, 1, condition="gold")}
        report = evaluate(EvaluationPlan((a, b)), raw, gold)
        self.assertIsNone(report["metrics"]["BAcc_gold"])
        self.assertIsNone(report["metrics"]["delta_BAcc_gold_raw"])
        self.assertEqual(report["delta_BAcc_bounds"], {"lower": 0, "upper": .5})

    def test_raw_gold_specification_or_condition_mismatch_rejected(self):
        p = ownership()
        plan = EvaluationPlan((p,))
        with self.assertRaises(ValueError):
            evaluate(plan, {p.probe_id: outcome(p)}, {})
        with self.assertRaises(ValueError):
            evaluate(plan, {p.probe_id: outcome(p)}, {p.probe_id: outcome(p)})
        q = ownership(problem_id="different")
        with self.assertRaises(ValueError):
            evaluate(plan, {p.probe_id: outcome(q)})

    def test_nonempty_packet_subset_has_its_own_matched_report(self):
        a = ownership("a", "m1")
        b = Probe("b", "m2", "role", a.binding)
        plan = EvaluationPlan((a, b))
        raw = {"a": outcome(a, 1, 0), "b": outcome(b, 1, 1)}
        gold = {p.probe_id: outcome(p, 1, 1, condition="gold") for p in plan.probes}
        report = evaluate(plan, raw, gold)
        self.assertEqual(report["metrics"]["delta_BAcc_gold_raw"], .5)
        self.assertEqual(report["nonempty_packets"]["delta_BAcc_gold_raw"], 1)
        self.assertEqual(report["nonempty_packets"]["weighted_packet_coverage"], .5)

    def test_frozen_weights_and_duplicate_ids(self):
        p = ownership()
        weights = {"quantity": 1.0}
        plan = EvaluationPlan((p,), weights)
        weights["quantity"] = .2
        self.assertEqual(plan.binding_pool.type_weights["quantity"], 1)
        with self.assertRaises(ValueError):
            EvaluationPlan((p, p))
        for wrong in ({}, {"other": 1}, {"quantity": .2}, {"quantity": -1}, {"quantity": float("nan")}):
            with self.assertRaises(ValueError):
                EvaluationPlan((p,), wrong)

    def test_random_baseline_averages_reciprocals_with_mother_weights(self):
        probes = []
        for i in range(21):
            k = 2 if i < 20 else 10
            domain = {"E" + str(j): "target" + str(j) for j in range(k)}
            probes.append(Probe(str(i), "many" if i < 20 else "few", "ref",
                                BindingTarget("single", {"owner": domain}, ("target0",))))
        self.assertAlmostEqual(uniform_candidate_baseline(Pool(tuple(probes)))["value"], .3)

    def test_random_baseline_respects_canonical_alias_equivalence(self):
        p = Probe("p", "m", "ref", BindingTarget("single", {"owner": {"E1": "a", "E2": "b", "E3": "a"}}, ("a",)))
        self.assertAlmostEqual(uniform_candidate_baseline(Pool((p,)))["value"], 2/3)

    def test_set_chance_requires_its_own_declared_distribution(self):
        p = Probe("p", "m", "scope", BindingTarget("set", {"subjects": {"E1": "a", "E2": "b"}}, ("a",)))
        with self.assertRaises(ValueError):
            uniform_candidate_baseline(Pool((p,)))

    def test_missing_type_in_resample_does_not_renormalize(self):
        a, b = ownership("a", "m1", "a"), ownership("b", "m2", "b")
        plan = EvaluationPlan((a, b))
        report = evaluate(plan, {"a": outcome(a), "b": outcome(b)}, problem_multiplicities={"m1": 2, "m2": 0})
        self.assertIsNone(report["metrics"]["BAcc"])
        self.assertEqual(report["by_type"]["a"]["BAcc"], 1)
        self.assertIsNone(report["by_type"]["b"]["BAcc"])

    def test_repeated_mother_in_bootstrap_is_not_collapsed(self):
        a, b = ownership("a", "m1"), ownership("b", "m2")
        report = evaluate(EvaluationPlan((a, b)), {"a": outcome(a), "b": outcome(b, 0, 0)},
                          problem_multiplicities={"m1": 2, "m2": 1})
        self.assertAlmostEqual(report["metrics"]["BAcc"], 2/3)

    def test_fraction_oracle_for_unequal_mothers_types_and_eligibility(self):
        rng = random.Random(28741)
        for trial in range(80):
            probes, raw, gold = [], {}, {}
            for i in range(1, rng.randint(3, 6)):
                for t in range(3):
                    for j in range(rng.randint(0, 5)):
                        p = ownership(f"{i}-{t}-{j}", f"m{i}", f"t{t}", readings=() if rng.random() < .3 else None)
                        probes.append(p)
                        raw[p.probe_id] = outcome(p, rng.randrange(2), rng.randrange(2))
                        gold[p.probe_id] = outcome(p, 1, rng.randrange(2), condition="gold")
            plan = EvaluationPlan(tuple(probes))
            actual = evaluate(plan, raw, gold)
            # Independent one-pass oracle uses exact rational ITEM weights:
            # alpha_t / (N_t * m_it). It does not call the aggregation code.
            expected = {}
            for name, selected, scores, field in (("CAcc", plan.joint_pool.probes, raw, "C"),
                                                  ("JAcc", plan.joint_pool.probes, raw, "J"),
                                                  ("BAcc_L", plan.joint_pool.probes, raw, "B"),
                                                  ("BAcc", plan.probes, raw, "B"),
                                                  ("BAcc_gold", plan.probes, gold, "B")):
                types = {p.probe_type for p in selected}
                total = Fraction(0)
                for p in selected:
                    nt = len({q.problem_id for q in selected if q.probe_type == p.probe_type})
                    mit = sum(q.problem_id == p.problem_id and q.probe_type == p.probe_type for q in selected)
                    total += Fraction(getattr(scores[p.probe_id], field), len(types) * nt * mit)
                expected[name] = total if selected else None
                if selected:
                    self.assertAlmostEqual(actual["metrics"][name], float(total), places=14)
                else:
                    self.assertIsNone(actual["metrics"][name])
            if expected["CAcc"]:
                self.assertAlmostEqual(actual["metrics"]["BAcc_given_C"], float(expected["JAcc"] / expected["CAcc"]), places=14)
            self.assertAlmostEqual(actual["metrics"]["delta_BAcc_gold_raw"], float(expected["BAcc_gold"] - expected["BAcc"]), places=14)


class SecondaryMetricsTests(unittest.TestCase):
    def test_solving_four_mother_example(self):
        probes = [ownership(f"{i}-{j}", f"m{i}") for i in range(4) for j in range(4)]
        n_correct = [4, 2, 3, 1]
        raw = {p.probe_id: outcome(p, 1, int(int(p.probe_id[-1]) < n_correct[int(p.problem_id[-1])])) for p in probes}
        report = solving_association(EvaluationPlan(tuple(probes)), raw, {"m0": 1, "m1": 1, "m2": 0, "m3": 0})
        self.assertEqual(report["SolveAcc"], .5)
        self.assertEqual(report["mean_binding_correct"], .75)
        self.assertEqual(report["mean_binding_wrong"], .5)
        self.assertEqual(report["delta_assoc"], .25)

    def test_solving_bi_is_plain_probe_mean_not_type_balanced(self):
        probes = [ownership(str(i), "m", "a" if i < 9 else "b") for i in range(10)]
        raw = {p.probe_id: outcome(p, 1, int(p.probe_type == "a")) for p in probes}
        report = solving_association(EvaluationPlan(tuple(probes)), raw, {"m": 1, "answer_only": 0})
        self.assertEqual(report["binding_by_problem"]["m"], .9)
        self.assertEqual(report["SolveAcc"], .5)
        self.assertIsNone(report["mean_binding_wrong"])
        self.assertIsNone(report["delta_assoc"])

    def test_missing_service_cannot_silently_improve_association(self):
        p = ownership()
        with self.assertRaises(ValueError):
            solving_association(EvaluationPlan((p,)), {p.probe_id: None}, {p.problem_id: 1})

    def test_harm_repair_and_unchanged_outcomes(self):
        probes = [ownership(str(i)) for i in range(4)]
        before_bits, after_bits = [1, 0, 1, 0], [0, 1, 1, 0]
        before = {p.probe_id: outcome(p, 1, before_bits[i]) for i, p in enumerate(probes)}
        after = {p.probe_id: outcome(p, 1, after_bits[i]) for i, p in enumerate(probes)}
        m = option_diagnostic(EvaluationPlan(tuple(probes)), before, after)["joint"]["metrics"]
        self.assertEqual([m[k] for k in ("H", "R", "unchanged_correct", "unchanged_wrong")], [.25] * 4)
        self.assertEqual(m["net_change"], m["accuracy_with"] - m["accuracy_without"])

    def test_transitions_are_item_events_not_products_of_aggregate_scores(self):
        a, b = ownership("a"), ownership("b")
        scores = {"a": outcome(a), "b": outcome(b, 0, 0)}
        m = option_diagnostic(EvaluationPlan((a, b)), scores, scores)["joint"]["metrics"]
        self.assertEqual((m["H"], m["R"]), (0, 0))
        self.assertEqual(m["accuracy_without"] * (1 - m["accuracy_with"]), .25)

    def test_options_use_joint_on_L_and_binding_only_separately(self):
        a, b = ownership("a"), ownership("b", readings=())
        before = {"a": outcome(a, 1, 1), "b": outcome(b, 1, 0)}
        after = {"a": outcome(a, 0, 1), "b": outcome(b, 1, 1)}
        r = option_diagnostic(EvaluationPlan((a, b)), before, after)
        self.assertEqual(r["joint"]["metrics"]["H"], 1)
        self.assertEqual(r["binding_only"]["metrics"]["R"], 1)


if __name__ == "__main__":
    unittest.main()
