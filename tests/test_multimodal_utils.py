"""Hand-calculated edge cases for classification and selective prediction."""

import unittest

import numpy as np

from multimodal_utils import (
    classification_metrics, choose_method, coverage_threshold,
    paired_accuracy_bootstrap, selective_metrics, softmax,
)


def scores_for(predictions, classes=3):
    """Create certain predictions for cases where only class identity matters."""
    return np.eye(classes)[predictions]


class MultimodalMetricTests(unittest.TestCase):
    def test_softmax_stability_and_translation_invariance(self):
        logits = np.array([[10000.0, 10001.0], [-10000.0, -9999.0]])
        actual = softmax(logits)
        expected = [1 / (1 + np.e), np.e / (1 + np.e)]
        np.testing.assert_allclose(actual, [expected, expected])
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_allclose(actual.sum(axis=1), 1)

    def test_softmax_extreme_finite_logits(self):
        np.testing.assert_array_equal(softmax([[1e308, -1e308]]), [[1.0, 0.0]])

    def test_known_confusion_and_macro_f1(self):
        labels = [0, 0, 0, 1, 1, 2]
        result = classification_metrics(labels, scores_for([0, 0, 1, 1, 2, 0]))
        self.assertEqual(result["confusion_matrix"], [[2, 1, 0], [0, 1, 1], [1, 0, 0]])
        self.assertEqual(result["n_examples"], 6)
        self.assertEqual(result["n_errors"], 3)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.5 + 0) / 3)
        self.assertEqual(result["per_class_recall"], [2 / 3, 0.5, 0])

    def test_absent_class_has_undefined_recall_but_fixed_class_macro_f1(self):
        result = classification_metrics([0], [[1, 0, 0]])
        self.assertEqual(result["per_class_recall"], [1.0, None, None])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)

    def test_argmax_tie_handling_is_consistent(self):
        probs = [[0.5, 0.5], [0.5, 0.5]]
        self.assertEqual(classification_metrics([0, 1], probs)["confusion_matrix"], [[1, 0], [1, 0]])
        self.assertEqual(selective_metrics([0, 1], probs, 0.5)["selective_accuracy"], 0.5)

    def test_choose_method_uses_fixed_priority_for_ties(self):
        metrics = {"ensemble": {"accuracy": 0.8}, "single": {"accuracy": 0.8}, "raw": {"accuracy": 0.7}}
        self.assertEqual(choose_method(metrics, ["raw", "single", "ensemble"]), "single")
        self.assertEqual(choose_method(metrics, ["ensemble", "single", "raw"]), "ensemble")

    def test_coverage_threshold_uses_ceiling_and_includes_ties(self):
        values = [0.9, 0.8, 0.8, 0.2]
        threshold = coverage_threshold(values, 0.5)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(float(np.mean(np.asarray(values) >= threshold)), 0.75)
        self.assertEqual(coverage_threshold(values, 0.01), 0.9)
        self.assertEqual(coverage_threshold(values, 1), 0.2)
        self.assertEqual(coverage_threshold([0.9, 0.8, 0.7], 0.5), 0.8)

    def test_selective_prediction_exposes_confident_errors_and_denominators(self):
        # Retained: one correct, one confidently wrong. Referred: one correct,
        # one wrong. Deferring does not magically resolve either referred case.
        labels = [0, 1, 0, 1]
        probs = [[0.95, 0.05], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]]
        result = selective_metrics(labels, probs, 0.8)
        self.assertEqual(result["accepted_count"], 2)
        self.assertEqual(result["referred_count"], 2)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_accuracy"], 0.5)
        self.assertEqual(result["selective_error_rate"], 0.5)
        self.assertEqual(result["accepted_errors"], 1)
        self.assertEqual(result["referred_errors"], 1)
        self.assertEqual(result["caught_error_fraction"], 0.5)
        self.assertEqual(result["correctly_accepted_fraction_total"], 0.25)
        self.assertEqual(result["per_class_coverage"], [0.5, 0.5])
        self.assertEqual(result["per_class_accepted_accuracy"], [1.0, 0.0])

    def test_empty_retained_set_uses_none_instead_of_fabricated_accuracy(self):
        result = selective_metrics([0, 1], [[0.8, 0.2], [0.2, 0.8]], 1.0)
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["selective_accuracy"])
        self.assertIsNone(result["selective_error_rate"])
        self.assertIsNone(result["caught_error_fraction"])
        self.assertEqual(result["per_class_accepted_accuracy"], [None, None])

    def test_absent_selective_class_and_no_errors_are_undefined(self):
        result = selective_metrics([0], [[0.8, 0.2]], 0)
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["per_class_coverage"], [1, None])
        self.assertEqual(result["per_class_accepted_accuracy"], [1, None])
        self.assertIsNone(result["caught_error_fraction"])

    def test_paired_bootstrap_keeps_identical_predictions_together(self):
        # Independent resampling of the two methods would incorrectly create
        # uncertainty here. Proper paired resampling yields exactly zero.
        labels = [0, 0, 1, 1]
        probs = scores_for([0, 1, 0, 1], 2)
        result = paired_accuracy_bootstrap(labels, probs, probs, resamples=500)
        self.assertEqual(result["mean_delta_b_minus_a"], 0)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual(result["tied_count"], 4)

    def test_paired_bootstrap_direction_counts_and_reproducibility(self):
        labels = [0, 0, 1, 1]
        a = scores_for([0, 1, 0, 0], 2)
        b = scores_for([1, 0, 1, 0], 2)
        result = paired_accuracy_bootstrap(labels, a, b)
        self.assertEqual(result["mean_delta_b_minus_a"], 0.25)
        self.assertEqual((result["a_better_count"], result["b_better_count"], result["tied_count"]), (1, 2, 1))
        self.assertEqual(result, paired_accuracy_bootstrap(labels, a, b))
        self.assertLessEqual(result["ci95"][0], result["mean_delta_b_minus_a"])
        self.assertGreaterEqual(result["ci95"][1], result["mean_delta_b_minus_a"])

    def test_paired_bootstrap_constant_improvement(self):
        result = paired_accuracy_bootstrap([0, 1], [[0, 1], [1, 0]], [[1, 0], [0, 1]])
        self.assertEqual(result["mean_delta_b_minus_a"], 1)
        self.assertEqual(result["ci95"], [1, 1])

    def test_invalid_logits(self):
        for invalid in [[], [[]], [1, 2], [[np.nan, 0]], [[np.inf, 0]], [[[1]]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                softmax(invalid)

    def test_invalid_probabilities(self):
        for invalid in [[], [[]], [[np.nan, 0]], [[np.inf, 0]], [[-0.1, 1.1]], [[0.2, 0.3]], [0.2, 0.8]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                classification_metrics([0], invalid)

    def test_invalid_labels(self):
        for invalid in [[], [2], [-1], [0.0], [True], [[0]], ["0"], [np.nan]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                classification_metrics(invalid, [[0.2, 0.8]])

    def test_invalid_coverage_inputs(self):
        for values, target in [([], 0.5), ([np.nan], 0.5), ([1.1], 0.5), ([[0.5]], 0.5), ([0.5], 0), ([0.5], 1.1), ([0.5], np.nan), ([0.5], True)]:
            with self.subTest(values=values, target=target), self.assertRaises(ValueError):
                coverage_threshold(values, target)

    def test_invalid_selection_and_threshold(self):
        for metrics, order in [({}, []), ({"a": {"accuracy": 0.5}}, ["a", "a"]), ({}, ["a"]), ({"a": {"accuracy": np.nan}}, ["a"])]:
            with self.subTest(metrics=metrics, order=order), self.assertRaises(ValueError):
                choose_method(metrics, order)
        for threshold in [-0.1, 1.1, np.nan, np.inf]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                selective_metrics([0], [[0.5, 0.5]], threshold)

    def test_bootstrap_rejects_misalignment_and_invalid_controls(self):
        with self.assertRaises(ValueError):
            paired_accuracy_bootstrap([0], [[0.5, 0.5]], [[1, 0, 0]])
        for options in [{"resamples": 0}, {"resamples": 2.5}, {"seed": -1}, {"seed": True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired_accuracy_bootstrap([0], [[0.5, 0.5]], [[0.5, 0.5]], **options)


if __name__ == "__main__":
    unittest.main()
