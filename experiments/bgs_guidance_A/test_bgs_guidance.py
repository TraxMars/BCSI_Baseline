"""Experiment A checks: exact baseline recovery, detached signature and BGS."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainer import Trainer
from model.vnet import VNet
from utils.boundary_guidance import (compute_bgs, spatial_gradient_3d,
                                     build_boundary_and_nonboundary,
                                     apply_labeled_bgs_guidance)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BGSChecks(unittest.TestCase):
    def args(self, enabled=False, alpha=0.1):
        return SimpleNamespace(in_channels=1, num_classes=2, device="cpu", base_lr=0.01,
                               max_iterations=30000, labeled_bs=2, ema_decay=0.9,
                               consistency=0.1, consistency_rampup=200,
                               use_bgs_guidance=enabled, bgs_alpha=alpha)

    def labels(self, batch=2):
        labels = torch.zeros(batch, 32, 32, 32, dtype=torch.long)
        labels[:, 12:20, 12:20, 12:20] = 1
        return labels

    def test_01_disabled_baseline_and_alpha_zero_full_step_exact(self):
        original = load_file("original_trainer", Path(__file__).parent / "baseline_source/trainer.py")
        torch.manual_seed(91)
        baseline = original.Trainer(self.args())
        torch.manual_seed(91)
        disabled = Trainer(self.args())
        torch.manual_seed(91)
        zero = Trainer(self.args(True, 0))
        batch = dict(image=torch.randn(4, 1, 32, 32, 32), label=self.labels(4))
        # Unlabeled GT must be completely irrelevant even when present in batch.
        batch["label"][2:] = 999
        losses = []
        for trainer in (baseline, disabled, zero):
            captured = []
            def record(fmt, *args):
                if fmt.startswith("iteration %d : loss"):
                    captured.append(tuple(float(x) for x in args[1:4]))
            torch.manual_seed(731)
            with mock.patch("logging.info", side_effect=record):
                if trainer is disabled:
                    with mock.patch("trainer.apply_labeled_bgs_guidance", side_effect=AssertionError("Disabled guidance called")):
                        with mock.patch.object(trainer.model, "forward", wraps=trainer.model.forward) as forward:
                            trainer.train(batch, 1000, "/tmp")
                            self.assertEqual(forward.call_count, 1)
                else:
                    with mock.patch.object(trainer.ema_model, "forward", wraps=trainer.ema_model.forward) as teacher:
                        trainer.train(batch, 1000, "/tmp")
                        self.assertEqual(teacher.call_count, 1)
            self.assertEqual(len(captured), 1)
            losses.append(captured[0])
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in trainer.ema_model.parameters()))
        self.assertEqual(losses[0], losses[1])
        self.assertEqual(losses[0], losses[2])
        for other in (disabled, zero):
            for key, expected in baseline.model.state_dict().items():
                self.assertTrue(torch.equal(expected, other.model.state_dict()[key]), key)
            for key, expected in baseline.ema_model.state_dict().items():
                self.assertTrue(torch.equal(expected, other.ema_model.state_dict()[key]), key)
            self.assertEqual(baseline.optimizer.param_groups[0]["lr"], other.optimizer.param_groups[0]["lr"])
            self.assertEqual(baseline.scheduler.last_epoch, other.scheduler.last_epoch)
        self.assertFalse(zero.bgs_weights.requires_grad)

    def test_02_formula_matches_experiment_05(self):
        audit = load_file("specificity_reference", ROOT / "analysis/boundary_specificity_audit.py")
        torch.manual_seed(2)
        feature = torch.randn(2, 7, 8, 8, 8, requires_grad=True)
        labels = self.labels()
        bgs, valid = compute_bgs(feature, labels)
        self.assertTrue(bool(valid.all()))
        self.assertFalse(bgs.requires_grad)
        for i in range(2):
            regions = audit.make_regions(labels[i:i+1, None], feature.shape[2:])
            reference, _, _ = audit.bgs_vector(feature[i:i+1].detach(), regions)
            np.testing.assert_allclose(bgs[i].numpy(), reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(spatial_gradient_3d(feature.detach()), audit.spatial_gradient(feature.detach()))

    def test_03_labeled_unchanged_detached_weights_and_unlabeled_shape(self):
        x3 = torch.randn(4, 9, 8, 8, 8, requires_grad=True)
        result, w, diag = apply_labeled_bgs_guidance(x3, self.labels(), 2, 0.1)
        self.assertTrue(torch.equal(result[:2], x3[:2]))
        self.assertEqual(result[2:].shape, x3[2:].shape)
        self.assertFalse(w.requires_grad)
        self.assertTrue(bool(((w >= 0) & (w <= 1)).all()))
        expected = x3[2:] + 0.1 * w[None, :, None, None, None] * x3[2:]
        self.assertTrue(torch.equal(result[2:], expected))
        self.assertLessEqual(diag["relative_change"], 0.100001)
        result[2:].square().sum().backward()
        self.assertTrue(torch.equal(x3.grad[:2], torch.zeros_like(x3.grad[:2])))
        self.assertTrue(bool(x3.grad[2:].abs().sum() > 0))

    def test_04_no_valid_sample_skips_with_original_tensor(self):
        x3 = torch.randn(4, 6, 8, 8, 8, requires_grad=True)
        result, w, diag = apply_labeled_bgs_guidance(x3, torch.zeros_like(self.labels()), 2)
        self.assertIs(result, x3)
        self.assertTrue(diag["skipped"])
        self.assertEqual(diag["valid_samples"], 0)
        self.assertEqual(diag["relative_change"], 0)
        self.assertTrue(torch.equal(w, torch.zeros_like(w)))

    def test_05_one_valid_sample_and_positive_signature_formula(self):
        x3 = torch.randn(4, 8, 8, 8, 8)
        labels = self.labels()
        labels[1] = 0
        bgs, valid = compute_bgs(x3[:2], labels)
        result, w, diag = apply_labeled_bgs_guidance(x3, labels, 2)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(diag["valid_samples"], 1)
        positive = bgs[0].relu()
        expected = (positive / (positive.max() + 1e-6)).to(x3.dtype)
        self.assertTrue(torch.equal(w, expected))
        self.assertFalse(diag["skipped"])
        self.assertTrue(torch.isfinite(result).all())

    def test_06_logits_and_teacher_no_grad_with_active_guidance(self):
        trainer = Trainer(self.args(True))
        inputs = torch.randn(4, 1, 32, 32, 32)
        features = trainer.model.encoder(inputs)
        others = [features[i] for i in (0, 1, 3, 4)]
        features[2], w, diag = apply_labeled_bgs_guidance(features[2], self.labels(), 2)
        self.assertEqual(diag["valid_samples"], 2)
        self.assertTrue(all(features[i] is original for i, original in zip((0, 1, 3, 4), others)))
        logits = trainer.model.decoder(features)
        with torch.no_grad():
            teacher = trainer.ema_model(inputs)
        self.assertEqual(logits.shape, teacher.shape)
        self.assertEqual(tuple(logits.shape), (4, 2, 32, 32, 32))
        loss = torch.nn.functional.mse_loss(logits.softmax(1)[2:], teacher.softmax(1)[2:])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in trainer.ema_model.parameters()))
        self.assertFalse(w.requires_grad)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))

    def test_07_nonfinite_feature_fails_explicitly(self):
        x3 = torch.ones(4, 3, 8, 8, 8)
        x3[3, 0, 0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            apply_labeled_bgs_guidance(x3, self.labels(), 2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
