import unittest

from physalign import EvaluationPlan, evaluate, paired_cluster_bootstrap, solving_association
from test_scoring import ownership, outcome


class BootstrapTests(unittest.TestCase):
    def test_reproducible_paired_raw_gold_and_model_differences(self):
        probes = tuple(ownership(str(i), "m" + str(i // 3)) for i in range(12))
        plan = EvaluationPlan(probes)
        raw = {p.probe_id: outcome(p, 1, int(int(p.probe_id) % 3 == 0)) for p in probes}
        gold = {p.probe_id: outcome(p, 1, 1, condition="gold") for p in probes}
        seen = []

        def statistic(counts):
            seen.append(dict(counts))
            m = evaluate(plan, raw, gold, problem_multiplicities=counts)["metrics"]
            return {"gain": m["delta_BAcc_gold_raw"], "identical_model_difference": m["BAcc"] - m["BAcc"]}

        first = paired_cluster_bootstrap(["m0", "m1", "m2", "m3"], statistic, n_resamples=120, seed=19)
        second = paired_cluster_bootstrap(["m3", "m2", "m1", "m0"], statistic, n_resamples=120, seed=19)
        self.assertEqual(first, second)
        self.assertTrue(all(sum(c.values()) == 4 for c in seen))
        self.assertTrue(any(max(c.values()) > 1 for c in seen))
        self.assertAlmostEqual(first["intervals"]["gain"]["lower"], 2/3)
        self.assertAlmostEqual(first["intervals"]["gain"]["upper"], 2/3)
        self.assertEqual(first["intervals"]["identical_model_difference"]["lower"], 0)
        self.assertEqual(first["intervals"]["identical_model_difference"]["upper"], 0)

    def test_lost_type_support_withholds_interval(self):
        a, b = ownership("a", "m1", "a"), ownership("b", "m2", "b")
        plan = EvaluationPlan((a, b))
        raw = {"a": outcome(a), "b": outcome(b, 0, 0)}
        result = paired_cluster_bootstrap(["m1", "m2"], lambda counts: {
            "BAcc": evaluate(plan, raw, problem_multiplicities=counts)["metrics"]["BAcc"]
        }, n_resamples=100, seed=2027)
        interval = result["intervals"]["BAcc"]
        self.assertGreater(interval["undefined_resamples"], 0)
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])
        self.assertEqual(interval["estimate"], .5)

    def test_duplicate_source_groups_are_sampled_together(self):
        seen = []
        def statistic(counts):
            seen.append(dict(counts))
            return {"x": float(counts["a"])}
        result = paired_cluster_bootstrap(["a", "b", "c"], statistic, n_resamples=40,
                                          cluster_ids={"a": "source1", "b": "source1", "c": "source2"})
        self.assertEqual(result["n_clusters"], 2)
        self.assertTrue(all(c["a"] == c["b"] for c in seen))
        self.assertTrue(all(c["a"] + c["c"] == 2 for c in seen))

    def test_solving_empty_correctness_group_is_undefined(self):
        a, b = ownership("a", "m1"), ownership("b", "m2")
        plan = EvaluationPlan((a, b))
        raw = {"a": outcome(a), "b": outcome(b, 0, 0)}
        result = paired_cluster_bootstrap(["m1", "m2"], lambda counts: {
            "association": solving_association(plan, raw, {"m1": 1, "m2": 0}, problem_multiplicities=counts)["delta_assoc"]
        }, n_resamples=100)
        self.assertGreater(result["intervals"]["association"]["undefined_resamples"], 0)
        self.assertIsNone(result["intervals"]["association"]["lower"])

    def test_percentile_interpolation_and_recorded_settings(self):
        r = paired_cluster_bootstrap(["m"], lambda counts: {"constant": .25}, n_resamples=20, confidence=.9, seed=7)
        self.assertEqual(r["intervals"]["constant"]["lower"], .25)
        self.assertEqual(r["intervals"]["constant"]["upper"], .25)
        self.assertEqual((r["confidence"], r["seed"], r["n_resamples"]), (.9, 7, 20))

    def test_reject_per_probe_ids_and_nonfinite_statistics(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(["m", "m"], lambda counts: {"x": 1})
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap(["m"], lambda counts: {"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
