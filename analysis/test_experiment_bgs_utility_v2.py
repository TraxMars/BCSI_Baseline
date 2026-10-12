"""CPU checks: exact replay, donor-only construction, RNG and case clustering."""
import copy
import inspect
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch import nn
import experiment_bgs_utility_v2 as audit


class ToyEncoder(nn.Module):
    def forward(self, image):
        return [torch.cat([image, 2 * image, 3 * image, -image], dim=1)] * 5


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()
        self.decoder = ToyDecoder()
        self.register_buffer("unchanged", torch.ones(1))
        self.eval()

    def forward(self, image):
        return self.decoder(self.encoder(image))


class ToyDecoder(nn.Module):
    def forward(self, features):
        return features[2][:, :2]


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def donor(self, case_id, invalid=False):
        image = np.arange(12 ** 3, dtype=np.float32).reshape(12, 12, 12) / 1000
        label = np.zeros_like(image, dtype=np.int64)
        if not invalid:
            label[4:8, 4:8, 4:8] = 1
        return dict(case_id=case_id, patch_index=0, image=image, label=label)

    def test_fixed_donors_all_receivers_are_distinct_and_wrap(self):
        ids = ["case{}".format(i) for i in range(8)]
        mapping = audit.donor_mapping(ids)
        self.assertEqual(mapping["case7"], ("case0", "case1"))
        self.assertTrue(all(r not in d and d[0] != d[1] for r, d in mapping.items()))

    def test_exact_padding_and_coordinates_and_rejection(self):
        image = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
        label = (image % 3 == 0).astype(np.int64)
        original = audit.legacy.crop_volume(image, label, (8, 8, 8), np.random.RandomState(42))
        im, gt = audit.replay_crop(image, label, original[2], (8, 8, 8))
        self.assertTrue(np.array_equal(im, original[0]))
        self.assertTrue(np.array_equal(gt, original[1]))
        bad = copy.deepcopy(original[2])
        bad["start_original"][0] += 1
        with self.assertRaises(ValueError):
            audit.replay_crop(image, label, bad, (8, 8, 8))
        with self.assertRaises(ValueError):
            audit.require_metadata({"status": "complete"})

    def test_signed_average_before_relu_normalization(self):
        fake_bgs = [torch.tensor([[-0.8, 0.2, 0.4, 0.1]], dtype=torch.float64),
                    torch.tensor([[0.4, 0.6, -0.2, 0.3]], dtype=torch.float64)]
        signature = torch.stack(fake_bgs).mean(0)[0]
        expected = (signature.relu() / (signature.relu().max() + 1e-6)).float()
        def reference(*args, **kwargs):
            return args[0], expected, {"valid_samples": 2}
        with patch.object(audit, "compute_bgs", side_effect=[(b, torch.tensor([True])) for b in fake_bgs]), \
                patch.object(audit, "apply_labeled_bgs_guidance", side_effect=reference):
            w, diagnostics, _ = audit.donor_signature(ToyModel(), [self.donor("d1"), self.donor("d2")], "receiver", "cpu")
        self.assertTrue(torch.equal(w, expected))
        individual_then_mean = torch.stack([b[0].relu() / (b[0].relu().max() + 1e-6) for b in fake_bgs]).mean(0).float()
        self.assertFalse(torch.allclose(w, individual_then_mean))
        self.assertEqual(float(w[0]), 0)

    def test_actual_A_weight_match_and_receiver_gt_not_an_argument(self):
        donors = [self.donor("d1"), self.donor("d2")]
        received_labels = []
        original = audit.compute_bgs
        def capture(feature, labels, min_voxels):
            received_labels.append(labels.clone())
            return original(feature, labels, min_voxels)
        with patch.object(audit, "compute_bgs", side_effect=capture):
            w, diagnostics, _ = audit.donor_signature(ToyModel(), donors, "receiver", "cpu")
        self.assertEqual(len(received_labels), 2)
        self.assertTrue(diagnostics["weights_match_experiment_A_exactly"])
        self.assertFalse(w.requires_grad)
        self.assertNotIn("receiver_label", inspect.signature(audit.donor_signature).parameters)
        self.assertNotIn("utility", inspect.signature(audit.make_strategy_plan).parameters)
        with self.assertRaises(ValueError):
            audit.donor_signature(ToyModel(), [self.donor("receiver"), self.donor("d2")], "receiver", "cpu")

    def test_invalid_donors_are_skipped_without_receiver_rescue(self):
        w, diagnostics, _ = audit.donor_signature(ToyModel(), [self.donor("d1", True), self.donor("d2", True)], "r", "cpu")
        self.assertIsNone(w)
        self.assertEqual(diagnostics["reason"], "no_valid_donor")
        w, diagnostics, _ = audit.donor_signature(ToyModel(), [self.donor("d1", True), self.donor("d2")], "r", "cpu")
        self.assertIsNotNone(w)
        self.assertEqual(diagnostics["valid_donors"], 1)

    def test_shuffle_independent_reproducible_and_all_permutations_preserved(self):
        w = torch.linspace(0, 1, 64)
        torch_state = torch.get_rng_state().clone()
        np_state = np.random.get_state()
        py_state = random.getstate()
        plan = audit.make_strategy_plan(w, 42)
        same = audit.make_strategy_plan(w, 42)
        self.assertEqual(audit.plan_digest(plan), audit.plan_digest(same))
        shuffles = [p for p in plan if p.name == "shuffled_bgs"]
        self.assertEqual([p.repetition for p in shuffles], list(range(20)))
        for p in shuffles:
            self.assertTrue(torch.equal(w.sort().values, p.weight.sort().values))
            self.assertLessEqual(p.shuffle_mean_delta, 1e-12)
            self.assertLessEqual(p.shuffle_norm_delta, 1e-12)
            self.assertEqual(sorted(p.permutation), list(range(64)))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(py_state, random.getstate())

    def test_only_x3_and_uniform_matches_mean_not_energy(self):
        features = [torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 2, 2).clone() for _ in range(5)]
        snapshots = [f.clone() for f in features]
        plan = audit.make_strategy_plan(torch.tensor([0, 0.1, 0.4, 1.0]), 42)
        for p in plan:
            changed, relative = audit.apply_strategy(features, p)
            self.assertTrue(all(changed[i] is features[i] for i in (0, 1, 3, 4)))
            self.assertTrue(all(torch.equal(f, s) for f, s in zip(features, snapshots)))
            self.assertEqual(changed[2].shape, features[2].shape)
        true = next(p for p in plan if p.name == "true_bgs")
        uniform = next(p for p in plan if p.name == "uniform_gain")
        self.assertAlmostEqual(float(true.weight.mean()), float(uniform.weight.mean()), places=6)
        self.assertNotAlmostEqual(float(true.weight.norm()), float(uniform.weight.norm()), places=5)

    def test_loss_gain_sign_and_prediction_metrics(self):
        label = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        label[:, 2:6, 2:6, 2:6] = 1
        inner, outer = audit.legacy.boundary_sides(label)
        base_logits = torch.zeros(1, 2, 8, 8, 8)
        perfect = torch.stack(((label == 0).float() * 10, (label == 1).float() * 10), dim=1)
        base = audit.evaluate_logits(base_logits, label, inner, outer)
        better = audit.evaluate_logits(perfect, label, inner, outer)
        gains = audit.calculate_gains(base, better)
        self.assertGreater(gains["boundary_ce_gain"], 0)
        self.assertGreater(gains["patch_dice_gain"], 0)
        self.assertEqual(better["fp_voxels"], 0)
        self.assertEqual(better["fn_voxels"], 0)
        self.assertEqual(better["patch_dice"], 1)
        self.assertEqual(better["patch_jaccard"], 1)
        self.assertGreater(audit.true_advantage(better, base), 0)
        self.assertLess(audit.true_advantage(base, better), 0)

    def test_case_clustering_and_twenty_shuffle_averages(self):
        cases = ["a", "b", "c", "d", "e", "f", "g", "h"]
        comparison, repetitions = [], []
        for i, case in enumerate(cases):
            for patch_index in range(3):
                for strategy in audit.STRATEGIES:
                    comparison.append(dict(checkpoint="model", receiver_case=case, patch_index=patch_index,
                                           strategy=strategy, boundary_ce=1.0, boundary_ce_gain=float(i),
                                           true_vs_shuffle_boundary_advantage=0.0, true_vs_uniform_boundary_advantage=0.0,
                                           S=0.0, negative_utility_weight_mass_fraction=0.5))
                for repetition in range(20):
                    repetitions.append(dict(checkpoint="model", receiver_case=case, patch_index=patch_index,
                                            repetition=repetition, boundary_ce=1.0 + repetition / 100,
                                            boundary_ce_gain=float(i), full_ce=0.5))
        rows, case_shuffle, summary = audit.summarize_cases(comparison, repetitions, cases, ["model"], 42)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(case_shuffle), 160)
        self.assertEqual(summary["model"]["primary_metrics"]["true_gain"]["n_cases"], 8)
        self.assertAlmostEqual(summary["model"]["primary_metrics"]["true_gain"]["mean"], 3.5)
        self.assertEqual(summary["model"]["primary_metrics"]["true_gain"]["resampling_unit"], "case")
        with self.assertRaises(AssertionError):
            audit.summarize_cases(comparison, repetitions[:-1], cases, ["model"], 42)

    def test_alignment_negative_weights_and_sign(self):
        u = np.array([-1, 1, -2, 2])
        w = torch.tensor([1.0, 0, 1.0, 0])
        aligned = audit.alignment_metrics(w, u)
        self.assertEqual(aligned["S"], -3)
        self.assertAlmostEqual(aligned["linear_predicted_boundary_gain"], -0.3)
        self.assertEqual(aligned["negative_utility_weight_mass_fraction"], 1)
        uniform = audit.alignment_metrics(torch.ones(4) * 0.5, u)
        self.assertIsNone(uniform["w_utility_spearman"])
        self.assertEqual(uniform["undefined_reason"], "constant_vector")
        self.assertEqual(uniform["negative_utility_weight_mass_fraction"], 0.5)

    def test_real_eval_no_gain_and_frozen_buffers(self):
        model = audit.legacy.VNet(n_filters=2).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = audit.legacy.snapshot_state(model)
        image = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            features = model.encoder(image)
            strategy = audit.Strategy("no_gain", -1, torch.zeros(features[2].shape[1]))
            unchanged, relative = audit.apply_strategy(features, strategy)
            self.assertTrue(torch.equal(model(image), model.decoder(unchanged)))
            self.assertEqual(relative, 0)
        self.assertTrue(audit.legacy.verify_state(model, before)["weights_and_buffers_exactly_unchanged"])

    def test_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                audit.main(["--output_dir", directory, "--reference_dir", "/missing", "--device", "cpu"])

    def test_end_to_end_receiver_loop_retains_every_shuffle_and_skips_unverified_U(self):
        ids = ["a", "b", "c"]
        patches = {}
        for case in ids:
            for index in range(3):
                sample = self.donor(case)
                sample["patch_index"] = index
                gt = torch.from_numpy(sample["label"])[None]
                inner, outer = audit.legacy.boundary_sides(gt)
                sample["coords"] = dict(inner_voxels=int(inner.sum()), outer_voxels=int(outer.sum()))
                patches[case, index] = sample
        comp, reps, alignment, weights, mapping, invalid, checks = audit.analyze_model(
            ToyModel(), "toy", 0, patches, ids, {}, SimpleNamespace(seed=42, device="cpu"), 8)
        self.assertEqual(len(comp), 9 * 4)
        self.assertEqual(len(reps), 9 * 20)
        self.assertEqual(len(invalid), 0)
        self.assertTrue(all(row["no_gain_forward_exact"] for row in checks["receivers"]))
        self.assertTrue(all(not row["historical_U_verified"] for row in checks["receivers"]))
        self.assertTrue(checks["weights_and_buffers_exactly_unchanged"])
        self.assertTrue(all(row["global_rng_unchanged"] for row in checks["receivers"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
