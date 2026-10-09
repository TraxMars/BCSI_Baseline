"""Focused CPU checks for frozen-model gain utility and sampling/statistics."""
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiment_bgs_utility as audit


class ToyDecoder(nn.Module):
    def forward(self, features):
        f = features[2]
        # Positive a0 amplifies the correct foreground logit; a1 the background.
        return torch.cat((f[:, 1:2], f[:, 0:1]), 1)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = ToyDecoder()
        self.dummy = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer("fixed_buffer", torch.ones(1))
        self.eval()


class UtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def toy_data(self):
        target = torch.ones(1, 4, 4, 4, dtype=torch.long)
        feat = torch.ones(1, 2, 4, 4, 4)
        feat[:, 0] = 2
        return [feat.clone() for _ in range(5)], target

    def test_gradient_sign_analytical_and_central_difference(self):
        features, target = self.toy_data()
        model = ToyModel()
        inner = torch.zeros_like(target, dtype=torch.bool)
        outer = ~inner
        inner[:, :2] = True
        outer = ~inner
        before = audit.snapshot_state(model)
        u, losses, logits = audit.gain_utility(model, features, target, inner, outer)
        expected = torch.tensor([2 / (1 + np.exp(1)), -1 / (1 + np.exp(1))], dtype=u.dtype)
        self.assertTrue(torch.allclose(u, expected, atol=1e-6))
        self.assertGreater(float(u[0]), 0)
        self.assertLess(float(u[1]), 0)
        args = SimpleNamespace(delta=0.001, sign_epsilon=1e-6, fd_atol=1e-4, fd_rtol=0.01)
        rows, groups = audit.finite_differences(model, features, target, inner, outer, losses,
                                               np.array([0.5, -0.5]), u.numpy(), args)
        self.assertTrue(all(r["sign_agreement"] and r["magnitude_agreement"] for r in rows))
        self.assertGreater(rows[0]["boundary_ce_gain_plus"], 0)
        self.assertLess(rows[1]["boundary_ce_gain_plus"], 0)
        self.assertTrue(torch.equal(logits, model.decoder(features)))
        self.assertTrue(audit.verify_state(model, before)["weights_and_buffers_exactly_unchanged"])

    def test_balanced_boundary_ce_does_not_weight_region_size(self):
        labels = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        logits = torch.randn(1, 2, 2, 2, 2)
        inner = torch.zeros_like(labels, dtype=torch.bool)
        inner.flatten()[0] = True
        outer = ~inner
        ce = F.cross_entropy(logits, labels, reduction="none")
        losses = audit.loss_values(logits, labels, inner, outer)
        expected = 0.5 * (ce[inner].double().mean() + ce[outer].double().mean())
        self.assertTrue(torch.equal(losses["boundary_ce"], expected))
        self.assertTrue(torch.equal(losses["full_ce"], F.cross_entropy(logits, labels)))

    def test_only_x3_scales_and_cached_features_are_unchanged(self):
        features, _ = self.toy_data()
        snapshots = [f.clone() for f in features]
        changed = audit.scale_x3(features, torch.tensor([1.1, 0.9]))
        self.assertTrue(all(changed[i] is features[i] for i in (0, 1, 3, 4)))
        self.assertTrue(all(torch.equal(s, f) for s, f in zip(snapshots, features)))
        self.assertEqual(changed[2].shape, features[2].shape)

    def test_bgs_reproduces_existing_helper_with_full_channel_weights(self):
        torch.manual_seed(42)
        features = torch.randn(1, 64, 12, 12, 8)
        labels = torch.zeros(1, 48, 48, 32, dtype=torch.long)
        labels[:, 16:32, 16:32, 8:24] = 1
        bgs, gb, gn, w, valid = audit.bgs_details(features, labels)
        reference, flags = audit.compute_bgs(features, labels)
        self.assertTrue(valid.item())
        self.assertTrue(torch.equal(flags, valid) and torch.equal(bgs, reference))
        self.assertEqual(w.shape, (1, 64))
        self.assertFalse(w.requires_grad)
        self.assertTrue(torch.all((w >= 0) & (w <= 1)))
        self.assertTrue(torch.equal(bgs, (gb - gn) / (gb + gn + audit.EPS)))

    def test_real_vnet_unit_gain_frozen_buffers_and_autograd(self):
        torch.manual_seed(123)
        model = audit.VNet(n_channels=1, n_classes=2, n_filters=2).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = audit.snapshot_state(model)
        image = torch.randn(1, 1, 32, 32, 32)
        labels = torch.zeros(1, 32, 32, 32, dtype=torch.long)
        labels[:, 10:22, 10:22, 10:22] = 1
        with torch.no_grad():
            features = model.encoder(image)
            original = model(image)
        inner, outer = audit.boundary_sides(labels)
        u, losses, logits = audit.gain_utility(model, features, labels, inner, outer)
        self.assertTrue(torch.equal(logits, original))
        self.assertTrue(torch.isfinite(u).all())
        self.assertEqual(len(u), features[2].shape[1])
        self.assertTrue(audit.verify_state(model, before)["no_model_parameter_gradients"])

    def test_crop_seed_coordinates_and_original_padding_rule(self):
        image = np.arange(8 * 10 * 12).reshape(8, 10, 12).astype(np.float32)
        label = (image % 3 == 0).astype(np.int64)
        first = audit.crop_volume(image, label, (8, 8, 8), np.random.RandomState(42))
        second = audit.crop_volume(image, label, (8, 8, 8), np.random.RandomState(42))
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]["padding"], [3, 2, 1])
        self.assertEqual(first[0].shape, (8, 8, 8))

    def test_groups_use_signed_bgs_and_record_overlapping_small_models(self):
        bgs = np.linspace(-1, 1, 64)
        groups = audit.select_bgs_groups(bgs)
        self.assertEqual(groups, {"high": [63, 62, 61, 60], "middle": [30, 31, 32, 33], "low": [0, 1, 2, 3]})
        self.assertEqual(sum(map(len, groups.values())), 12)
        tied = audit.select_bgs_groups(np.zeros(64))
        self.assertEqual(tied["low"], [0, 1, 2, 3])
        small = audit.select_bgs_groups(np.arange(3))
        self.assertLess(len(set(sum(small.values(), []))), sum(map(len, small.values())))

    def test_case_bootstrap_undefined_correlations_and_nonfinite_rejected(self):
        summary = audit.bootstrap_cases([0.1, 0.9], np.random.RandomState(42), 1000)
        self.assertEqual(summary["n_cases"], 2)
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertEqual(summary["resampling_unit"], "case")
        self.assertEqual(audit.safe_spearman(np.zeros(64), np.arange(64)), (None, "constant_vector"))
        with self.assertRaises(FloatingPointError):
            audit.assert_finite(np.array([0, np.nan]), "case/layer/channel")
        with self.assertRaises(FloatingPointError):
            audit.json_safe({"bad": float("inf")})

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            audit.create_output(path)
            (path / "sentinel.txt").write_text("preserve")
            with self.assertRaises(FileExistsError):
                audit.create_output(path)
            self.assertEqual((path / "sentinel.txt").read_text(), "preserve")

    def test_actual_labeled_split_excludes_test_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.list").write_text("\n".join("case{}".format(i) for i in range(20)))
            (root / "test.list").write_text("test0\n")
            args = SimpleNamespace(data_path=str(root), dataset="LA", labeled_num=10, patch_size=[112, 112, 80])
            cases, split = audit.labeled_split(args)
            self.assertEqual([c[1] for c in cases], ["case{}".format(i) for i in range(8)])
            self.assertFalse(split["test_gt_loaded"])
            (root / "test.list").write_text("case0\n")
            with self.assertRaises(ValueError):
                audit.labeled_split(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
