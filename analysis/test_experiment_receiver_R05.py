"""CPU tests for spatial alignment, fixed selection, targets and full evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import experiment_receiver_R05 as a


class R05Tests(unittest.TestCase):
    def test_flip_inverse_axes(self):
        x = torch.arange(8 * 12 * 16).reshape(8, 12, 16)
        for axis in range(3):
            self.assertTrue(torch.equal(torch.flip(torch.flip(x, [axis]), [axis]), x))
            asymmetric = torch.flip(x, [axis])
            self.assertFalse(torch.equal(asymmetric, x))

    def test_direction_definitions_and_historical_D(self):
        p = np.full((8, 8, 8), .25, np.float32)
        aligned = np.stack([p + .05, p + .10, p + .20])
        maps = a.direction_maps(p, aligned)
        np.testing.assert_allclose(maps["Dx"], .05, atol=1e-7)
        np.testing.assert_allclose(maps["Dxy"], .075, atol=1e-7)
        old, _, _, _, _ = a.r0.construct_risks(p, aligned, 42)
        self.assertTrue(np.array_equal(maps["Dxyz"], old["disagreement"]))

    def test_pooling_coordinates_and_trilinear_centers(self):
        indices = np.indices((8, 12, 16)).astype(np.float32)
        ramp = indices[0] * 100 + indices[1] * 10 + indices[2]
        grid_indices = np.indices((2, 3, 4)) * 4 + 1.5
        expected = grid_indices[0] * 100 + grid_indices[1] * 10 + grid_indices[2]
        for mapping in ("average", "trilinear"):
            np.testing.assert_allclose(a.map_to_grid(ramp, (2, 3, 4), mapping), expected, atol=1e-5)
        expected_max = (np.indices((2, 3, 4)) * 4 + 3)
        np.testing.assert_array_equal(a.map_to_grid(ramp, (2, 3, 4), "max"), expected_max[0] * 100 + expected_max[1] * 10 + expected_max[2])
        impulse = np.zeros((8, 8, 8), np.float32)
        impulse[4, 0, 4] = 1
        self.assertEqual(a.map_to_grid(impulse, (2, 2, 2), "average")[1, 0, 1], 1 / 64)
        self.assertEqual(a.map_to_grid(impulse, (2, 2, 2), "max")[1, 0, 1], 1)
        self.assertEqual(a.map_to_grid(impulse, (2, 2, 2), "trilinear").sum(), 0)
        with self.assertRaises(ValueError):
            a.map_to_grid(impulse, (3, 2, 2), "max")

    def test_error_count_thresholds_and_lift(self):
        e = np.zeros((8, 8, 8), bool)
        # Counts at different corner blocks: 1,6,7,16.
        for (i, j, k), count in zip(((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1)), (1, 6, 7, 16)):
            block = np.zeros(64, bool)
            block[:count] = True
            e[i*4:i*4+4, j*4:j*4+4, k*4:k*4+4] = block.reshape(4, 4, 4)
        counts = a.block_count(e)
        self.assertEqual(counts.sum(), e.sum())
        self.assertEqual((counts / 64 >= .10).sum(), 2)
        self.assertEqual((counts / 64 >= .25).sum(), 1)
        selected = counts >= 7
        self.assertEqual(a.lift_grid(selected).sum(), 64 * selected.sum())

    def test_fixed_budgets_and_independent_random_state(self):
        maps = {method: np.zeros((4, 4, 4), np.float32) for method in a.METHODS}
        domain = np.zeros((4, 4, 4), bool)
        domain[1:3, :, :] = True
        np.random.seed(76)
        before = np.random.get_state()
        payload, checks = a.selections(maps, domain, "x3", "average", 42)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        for check in checks:
            selected = a.unpack(payload, "x3", "average", check["domain"], check["method"], check["budget"], domain.shape)
            self.assertEqual(selected.sum(), check["selected_positions"])
            self.assertEqual(check["selected_positions"], a.r0.budget_count(check["domain_positions"], check["budget"]))
            if check["domain"] != "global":
                self.assertFalse((selected & ~domain).any())

    def test_GT_lock_and_missing_target(self):
        history = dict(data_sources=[], sampling=dict(patch_coordinates=[]), configuration=dict(patch_size=[8, 8, 8]))
        reader = a.r0.HistoricalReader(history)
        with self.assertRaises(RuntimeError):
            reader.read("forbidden", 0, "label")
        score = np.zeros((2, 2, 2), np.float32)
        ranked = a.ranked_metrics(score, np.ones_like(score, bool), np.zeros_like(score, int), 64)
        self.assertIsNone(ranked["ap_any"])
        self.assertEqual(ranked["ap_any_undefined_reason"], "no_positive_errors")
        self.assertIsNone(ranked["spearman_fraction"])
        self.assertEqual(ranked["spearman_undefined_reason"], "constant_score_or_target")
        a.r0.safe(ranked)

    def test_existing_directory_and_absolute_code_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                a.create_output(tmp)
        self.assertTrue(all(p.is_absolute() and p.is_file() for p in a.NEW_CODE))

    def test_output_overwrite_refused_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(a, "ROOT", root):
                out = a.create_output(root / "analysis/results/experiment_receiver_R05/run_20261013_150000")
                sentinel = out / "sentinel.txt"
                sentinel.write_text("keep existing data")
                before = a.r0.sha256(sentinel)
                with self.assertRaises(FileExistsError):
                    a.create_output(out)
                self.assertEqual(a.r0.sha256(sentinel), before)

    def test_AP_ties_and_target_severity(self):
        scores = np.array([.5, .5, .2, .1]).reshape(1, 1, 4)
        counts = np.array([1, 0, 7, 0]).reshape(1, 1, 4)
        rank = a.ranked_metrics(scores, np.ones(scores.shape, bool), counts, 64)
        self.assertAlmostEqual(rank["ap_any"], .5 * .5 + .5 * (2 / 3))
        self.assertAlmostEqual(rank["ap_severe10"], 1 / 3)
        self.assertEqual(rank["prevalence_any"], .5)
        self.assertEqual(rank["prevalence_severe10"], .25)

    def test_case_bootstrap_not_voxels(self):
        rows = [dict(case_id=c, patch_index=p, method="Dxyz", metric=value)
                for c, values in (("A", (0, 0, 0)), ("B", (1, 1, 1))) for p, value in enumerate(values)]
        cases = a.case_aggregate(rows, ["A", "B"], ("method",), ("metric",))
        stat = a.bootstrap([r["metric"] for r in cases])
        self.assertEqual(stat["n_cases"], 2)
        self.assertEqual(stat["mean"], .5)
        self.assertEqual(stat["unit"], "case")
        self.assertEqual(stat["repetitions"], 10000)
        with self.assertRaises(AssertionError):
            a.case_aggregate(rows[:-1], ["A", "B"], ("method",), ("metric",))

    def test_interior_strict_distance(self):
        mask = a.interior_domain((40, 40, 40), 4)
        self.assertFalse(mask[4].any())
        self.assertTrue(mask[5, 5, 5])
        self.assertEqual(mask.sum(), 30 ** 3)

    def test_actual_VNet_frozen_eval_features(self):
        torch.set_num_threads(2)
        model = a.VNet(n_channels=1, n_classes=2, n_filters=2, normalization="batchnorm", has_dropout=True).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        image = np.random.RandomState(42).normal(size=(32, 32, 32)).astype(np.float32)
        p, aligned, check = a.r0.teacher_predictions(model, image, "cpu")
        with torch.no_grad():
            x = torch.from_numpy(image)[None, None]
            features = model.encoder(x)
            self.assertTrue(torch.equal(model.decoder(features), model(x)))
        self.assertEqual(features[2].shape[2:], (8, 8, 8))
        self.assertFalse(features[2].requires_grad)
        self.assertTrue(check["no_aug_forward_exact"])
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))
        a.r0.finite(aligned, "test predictions")

    def test_small_complete_evaluation_pipeline_and_gain_sign(self):
        # Synthetic wrong foreground at one voxel; highest risk selects it.
        p = np.full((8, 8, 8), .05, np.float32)
        p[2:6, 2:6, 2:6] = .95
        aligned = np.stack([p, np.flip(p, 0).copy(), p * .9])
        base, b, _, neighborhood, _ = a.r0.construct_risks(p, aligned, 42)
        original = dict(probability=p, aligned_flip_probabilities=aligned, pseudo_boundary=b,
                        pseudo_boundary_neighborhood=neighborhood, **{"risk_" + k: v for k, v in base.items()})
        maps = a.risk_maps(original)
        payload, _ = a.selections(maps, neighborhood, "original", "native", 42)
        grid_domain = a.block_count(neighborhood) > 0
        payload.update(grid_shape=np.array((2, 2, 2)), common_grid_domain=grid_domain)
        for mapping in a.MAPPINGS:
            mapped = {k: a.map_to_grid(v, (2, 2, 2), mapping) for k, v in maps.items()}
            payload.update({"risk_" + mapping + "_" + k: v for k, v in mapped.items()})
            selected, _ = a.selections(mapped, grid_domain, "x3", mapping, 42)
            payload.update(selected)
        gt = (p > .5).astype(np.uint8)
        gt[0, 0, 0] = 1
        rows, direction, originals, failures, status, targets, counts = a.evaluate_patch("case", 0, payload, original, gt)
        self.assertEqual(len(rows), 3 * 2 * 12 * 4)
        self.assertEqual(status["error_voxels"], 1)
        self.assertEqual(counts.sum(), 1)
        self.assertEqual(status["severe10_grids"], 0)
        self.assertTrue(all(r["ap_severe10"] is None for r in rows))
        self.assertTrue(all(r["ap_severe10_undefined_reason"] == "no_positive_errors" for r in rows))
        a.r0.safe(dict(rows=rows, direction=direction, originals=originals, failures=failures))
        # Coverage improvement means MORE true errors captured; no inverted sign.
        selected = counts > 0
        loc = a.localization(selected, np.ones(counts.shape, bool), counts, targets,
                             np.zeros(p.shape, bool), neighborhood)
        self.assertEqual(loc["error_coverage"], 1)
        self.assertEqual(loc["coverage_change_vs_original"], 1)
        self.assertEqual(loc["new_error_voxels"], 1)
        self.assertEqual(loc["lost_original_error_voxels"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
