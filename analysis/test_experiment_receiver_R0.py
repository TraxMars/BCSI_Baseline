"""Meaningful R0 tests: sealed GT access, exact coordinates and fair ranking."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import experiment_receiver_R0 as r0


class EquivariantTeacher(torch.nn.Module):
    def forward(self, image):
        return torch.cat((-image, image), dim=1)


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flip_and_inverse_exact_coordinates_and_zero_disagreement(self):
        image = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9) / 100
        p, aligned, checks = r0.teacher_predictions(EquivariantTeacher().eval(), image, "cpu")
        self.assertTrue(all(np.array_equal(p, a) for a in aligned))
        maps, *_ = r0.construct_risks(p, aligned, 42)
        self.assertEqual(float(maps["disagreement"].max()), 0)
        self.assertTrue(checks["no_aug_forward_exact"])
        for axis in range(3):
            self.assertTrue(np.array_equal(image, np.flip(np.flip(image, axis), axis)))
        self.assertFalse(np.array_equal(p, np.flip(p, 0)))

    def test_entropy_limits_and_fixed_combination_without_GT(self):
        p = np.array([0, 0.5, 1], dtype=np.float32).reshape(1, 1, 3)
        aligned = np.stack([p, p, p])
        maps, *_ = r0.construct_risks(p, aligned, 42)
        self.assertTrue(np.array_equal(maps["entropy"], np.array([0, 1, 0], dtype=np.float32).reshape(p.shape)))
        self.assertTrue(np.array_equal(maps["combined"], maps["entropy"] / 2))
        self.assertNotIn("gt", inspect.signature(r0.construct_risks).parameters)
        self.assertNotIn("gt", inspect.signature(r0.build_candidates).parameters)

    def test_pseudo_boundary_distance_units_and_empty_boundary(self):
        mask = np.zeros((15, 15, 15), dtype=bool)
        mask[5:10, 5:10, 5:10] = True
        boundary, distance, prior, valid = r0.pseudo_boundary_distance(mask)
        self.assertTrue(valid)
        self.assertTrue(np.all(distance[boundary] == 0))
        self.assertTrue(np.all(prior[boundary] == 1))
        self.assertAlmostEqual(float(distance[0, 7, 7]), 4)
        _, distance, prior, valid = r0.pseudo_boundary_distance(np.zeros(mask.shape, dtype=bool))
        self.assertFalse(valid)
        self.assertTrue(np.isfinite(distance).all())
        self.assertEqual(float(prior.sum()), 0)

    def test_morphology_matches_original_pooling_edge_rule(self):
        mask = torch.zeros(1, 1, 8, 9, 10)
        mask[:, :, :5, 2:7, 3:8] = 1
        dilated = torch.nn.functional.max_pool3d(mask, 3, 1, 1) > 0.5
        eroded = (1 - torch.nn.functional.max_pool3d(1 - mask, 3, 1, 1)) > 0.5
        self.assertTrue(np.array_equal(r0.boundary(mask[0, 0].numpy()), (dilated & ~eroded)[0, 0].numpy()))

    def test_GT_reader_is_locked_until_all_score_files_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.h5"
            with h5py.File(path, "w") as f:
                f["image"] = np.ones((8, 8, 8), dtype=np.float32)
                f["label"] = np.ones((8, 8, 8), dtype=np.uint8)
            coords = dict(case_id="case", patch_index=0, original_shape=[8, 8, 8], padding=[0, 0, 0],
                          padded_shape=[8, 8, 8], start_padded=[1, 1, 1], start_original=[1, 1, 1])
            history = dict(data_sources=[dict(case_id="case", path=str(path))], sampling=dict(patch_coordinates=[coords]),
                           configuration=dict(patch_size=[4, 4, 4]))
            reader = r0.HistoricalReader(history)
            self.assertEqual(reader.read("case", 0, "image").shape, (4, 4, 4))
            with self.assertRaises(RuntimeError):
                reader.read("case", 0, "label")
            with self.assertRaises(RuntimeError):
                reader.seal([])
            artifact = Path(tmp) / "scores.npz"
            np.savez(str(artifact), entropy=np.zeros((4, 4, 4)))
            reader.seal([dict(path=str(artifact), sha256=r0.sha256(artifact))])
            self.assertEqual(reader.read("case", 0, "label").sum(), 64)
            self.assertEqual([e["phase"] for e in reader.events], ["A", "B"])

    def test_replay_coordinates_are_exact_and_invalid_history_rejected(self):
        image = np.arange(8 * 10 * 12).reshape(8, 10, 12)
        coords = dict(original_shape=[8, 10, 12], padding=[3, 2, 1], padded_shape=[14, 14, 14],
                      start_padded=[2, 3, 4], start_original=[-1, 1, 3])
        patch = r0.replay_array(image, coords, [8, 8, 8], np.float32)
        expected = np.pad(image, [(3, 3), (2, 2), (1, 1)])[2:10, 3:11, 4:12]
        self.assertTrue(np.array_equal(patch, expected))
        with self.assertRaises(ValueError):
            r0.replay_array(image, dict(coords, start_original=[0, 1, 3]), [8, 8, 8], np.float32)

    def test_budget_counts_equal_ties_GT_free_and_domain_fairness(self):
        maps = {m: np.zeros((8, 8, 8), dtype=np.float32) for m in r0.METHODS}
        domain = np.zeros((8, 8, 8), dtype=bool)
        domain[2:6, 2:6, 2:6] = True
        candidates, checks = r0.build_candidates(maps, domain, 42)
        second, _ = r0.build_candidates(maps, domain, 42)
        self.assertTrue(all(np.array_equal(candidates[k], second[k]) for k in candidates))
        for d in r0.DOMAINS:
            for budget in r0.BUDGETS:
                sizes = [c["selected_voxels"] for c in checks if c["domain"] == d and c["budget"] == budget]
                self.assertEqual(len(set(sizes)), 1)
                selected = r0.unpack_candidate(candidates, d, "entropy", budget, domain.shape)
                self.assertEqual(selected.sum(), sizes[0])
                if d != "global":
                    self.assertFalse((selected & ~domain).any())

    def test_AP_exact_tie_grouping_matches_sklearn_and_missing_targets(self):
        rng = np.random.RandomState(42)
        for _ in range(5):
            scores = rng.randint(0, 5, size=100).astype(float)
            target = rng.randint(0, 2, size=100).astype(bool)
            order = np.argsort(-scores)
            actual, reason = r0.average_precision_sorted(scores[order], target[order])
            self.assertAlmostEqual(actual, average_precision_score(target, scores), places=12)
        self.assertEqual(r0.average_precision_sorted(np.zeros(5), np.zeros(5, dtype=bool)), (None, "no_positive_errors"))
        self.assertAlmostEqual(r0.average_precision_sorted(np.ones(4), np.array([0, 1, 0, 1]))[0], 0.5)

    def test_precision_recall_enrichment_signs_and_empty_target(self):
        domain = np.ones(100, dtype=bool)
        target = np.zeros(100, dtype=bool)
        target[:10] = True
        selected = np.zeros(100, dtype=bool)
        selected[:5] = True
        metrics = r0.selection_metrics(selected, domain, target)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["enrichment"], 10)
        metrics = r0.selection_metrics(selected, domain, np.zeros(100, dtype=bool))
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["enrichment"])
        self.assertEqual(metrics["undefined_reason"], "no_positive_errors")

    def test_far_FP_truncation_and_empty_GT_explicit(self):
        gt = np.zeros((65, 65, 65), dtype=np.uint8)
        gt[1, 1, 1] = 1
        p = np.zeros(gt.shape, dtype=np.float32)
        p[32, 32, 32] = 0.99
        p[64, 64, 64] = 0.99
        targets = r0.evaluate_errors(p, gt, r0.boundary(p > 0.5))
        self.assertEqual(targets["far_fp"].sum(), 2)
        self.assertEqual(targets["far_fp_core"].sum(), 1)
        self.assertEqual(targets["far_fp_crop_sensitive"].sum(), 1)
        self.assertEqual(targets["high_confidence_far_fp"].sum(), 2)
        targets = r0.evaluate_errors(p, np.zeros(gt.shape), r0.boundary(p > 0.5))
        self.assertFalse(targets["far_defined"])
        self.assertTrue(np.isfinite(targets["foreground_distance"]).all())

    def test_case_bootstrap_clusters_and_nonfinite_rejection(self):
        stats = r0.bootstrap_cases([0.1, 0.9], 42)
        self.assertEqual(stats["n_cases"], 2)
        self.assertEqual(stats["resampling_unit"], "case")
        self.assertAlmostEqual(stats["mean"], 0.5)
        self.assertIsNone(r0.bootstrap_cases([None, None], 42)["mean"])
        with self.assertRaises(FloatingPointError):
            r0.finite(np.array([np.nan]), "case/patch")
        with self.assertRaises(FloatingPointError):
            r0.safe(float("inf"))

    def test_no_error_patch_kept_with_explicit_missing_AP_and_empty_domain(self):
        p = np.ones((8, 8, 8), dtype=np.float32) * 0.99
        maps, b, distance, neighborhood, valid = r0.construct_risks(p, np.stack([p] * 3), 42)
        candidates, _ = r0.build_candidates(maps, neighborhood, 42)
        archive = dict(probability=p, pseudo_boundary=b, pseudo_boundary_neighborhood=neighborhood,
                       pseudo_boundary_present=np.array(valid), **{"risk_" + k: v for k, v in maps.items()}, **candidates)
        rows, far_rows, targets, status = r0.evaluate_patch("case", 0, archive, np.ones(p.shape, dtype=np.uint8))
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(r["ap_all_errors"] is None for r in rows))
        self.assertTrue(all(r["far_fp_recall"] is None for r in rows))
        self.assertTrue(all(r["far_core_recall_undefined_reason"] == "no_far_FP_core" for r in rows))
        self.assertTrue(all(r["all_errors_undefined_reason"] == "empty_evaluation_domain" for r in rows if r["domain"] != "global"))
        repeated = [dict(row, patch_index=i) for i in range(3) for row in rows]
        cases, paired, summary = r0.aggregate(repeated, ["case"], 42)
        self.assertEqual(summary["methods"]["global"]["entropy"]["5"]["ap_all_errors"]["n_cases"], 0)
        self.assertEqual(r0.choose_recommendation(summary, [status])["choice"], "C")

    def test_original_frozen_VNet_eval_and_buffers(self):
        model = r0.VNet(n_filters=2).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        image = np.random.RandomState(42).normal(size=(32, 32, 32)).astype(np.float32)
        p, aligned, check = r0.teacher_predictions(model, image, "cpu")
        self.assertTrue(check["no_aug_forward_exact"])
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(np.isfinite(aligned).all())

    def test_existing_output_is_refused_before_any_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                r0.main(["--output_dir", tmp, "--reference_dir", "/missing", "--device", "cpu"])

    def test_source_manifest_paths_are_absolute_and_repository_relative(self):
        self.assertTrue(all(p.is_absolute() and p.exists() for p in r0.NEW_CODE))
        self.assertTrue(all(str(p.relative_to(r0.ROOT)).startswith("analysis/") for p in r0.NEW_CODE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
