"""Numerical checks for the scientific measurements, independent of training."""
import importlib.util
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("boundary_representation_audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BoundaryAuditTests(unittest.TestCase):
    def test_feature_resolution_bands(self):
        label = torch.zeros(1, 1, 7, 7, 7)
        label[:, :, 2:5, 2:5, 2:5] = 1
        inside, outside = audit.boundary_bands(label, (7, 7, 7))
        self.assertEqual(int(inside.sum()), 26)
        self.assertEqual(int(outside.sum()), 98)
        self.assertFalse(bool((inside & outside).any()))
        upsampled = torch.nn.functional.interpolate(label, scale_factor=2, mode="nearest")
        actual = audit.boundary_bands(upsampled, (7, 7, 7))
        self.assertTrue(torch.equal(actual[0], inside))
        self.assertTrue(torch.equal(actual[1], outside))
        # An all-foreground crop has no outside band and must be rejected.
        _, outside_full = audit.boundary_bands(torch.ones_like(label), (7, 7, 7))
        self.assertEqual(int(outside_full.sum()), 0)

    def test_signed_standardization_population_variance(self):
        feature = torch.tensor([[[[[1., 3., 5., 7.]]], [[[7., 5., 3., 1.]]]]])
        inside = torch.tensor([[[[[True, True, False, False]]]]])
        outside = ~inside
        d = audit.transition_vector(feature, inside, outside)
        np.testing.assert_allclose(d, np.array([-4., 4.]) / np.sqrt(1 + audit.EPS))
        constant = audit.transition_vector(torch.ones_like(feature), inside, outside)
        np.testing.assert_array_equal(constant, [0., 0.])

    def test_cosine_retains_sign_and_topk_uses_absolute_magnitude(self):
        d = np.array([-5., 3., 1., 0.])
        self.assertEqual(audit.topk(d, 0.5), {0, 1})
        self.assertEqual(audit.jaccard({0, 1}, {1, 2}), 1 / 3)
        self.assertLess(audit.signed_cosine(audit.normalize(d), audit.normalize(-d)), -0.999)
        self.assertAlmostEqual(audit.spearman(d, -d), 1)
        self.assertTrue(np.isnan(audit.signed_cosine(np.zeros(4), audit.normalize(d))))
        self.assertTrue(np.isnan(audit.spearman(np.zeros(4), d)))
        # Mean raw patch vectors before normalization is distinct from
        # averaging normalized patches when their magnitudes differ.
        patch_d = np.array([[10., 0.], [0., 1.]])
        case_norm = audit.normalize(patch_d.mean(axis=0))
        self.assertGreater(case_norm[0], 0.99)
        self.assertLess(case_norm[1], 0.1)

    def test_augmentations_preserve_image_label_alignment(self):
        label = (torch.arange(2 * 3 * 4).reshape(1, 1, 2, 3, 4) % 2).float()
        image = 3 * label + 7
        for name in audit.AUGMENTATIONS:
            x, y = audit.geometric_view(image, label, name)
            self.assertTrue(torch.equal(x, 3 * y + 7))
            self.assertEqual(int(y.sum()), int(label.sum()))
        x, _ = audit.geometric_view(image, label, "rot90_axes01")
        self.assertEqual(tuple(x.shape), (1, 1, 3, 2, 4))

    def test_case_bootstrap_excludes_self_pairs(self):
        m = np.array([[np.nan, 0.4, 0.4], [np.nan, np.nan, 0.4], [np.nan, np.nan, np.nan]])
        weights = np.array([[1, 1, 1], [2, 1, 0], [3, 0, 0]])
        low, high = audit.case_pair_ci(m, weights)
        self.assertAlmostEqual(low, 0.4)
        self.assertAlmostEqual(high, 0.4)
        low, high = audit.cluster_ci([0.2, 0.2, 0.8], [0, 0, 1], np.array([[1, 1], [2, 0], [0, 2]]))
        self.assertLess(low, 0.4)
        self.assertGreater(high, 0.6)

    def test_nonfinite_d_reports_layer_case_channel_and_fails(self):
        class Dummy(torch.nn.Module):
            def encoder(self, image):
                f = torch.ones(1, 3, 7, 7, 7)
                f[:, 1] = float("nan")
                return [f] * 5
        label = torch.zeros(1, 1, 7, 7, 7)
        label[:, :, 2:5, 2:5, 2:5] = 1
        stream = io.StringIO()
        logger = logging.getLogger("test_nonfinite")
        logger.addHandler(logging.StreamHandler(stream))
        logger.setLevel(logging.INFO)
        events = []
        with self.assertRaises(FloatingPointError):
            audit.extract(Dummy().eval(), torch.ones_like(label), label, SimpleNamespace(min_band_voxels=8),
                          logger, {layer: [] for layer in audit.LAYERS}, events, "labeled_case", 1, "identity")
        message = stream.getvalue()
        self.assertIn("NONFINITE_D layer=x2 case=labeled_case", message)
        self.assertIn("channel=1", message)
        self.assertEqual(events[-1]["status"], "error")

    def test_no_forbidden_analysis_calls(self):
        audit.check_analysis_source()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
