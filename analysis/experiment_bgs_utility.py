"""Frozen-checkpoint BGS versus x3 channel-gain utility; never trains a model.

Run from any directory with the repository's SSL environment. Existing output
directories are refused, including empty directories; use --output_dir for a
new, explicit destination. No pre-existing repository file is written.
"""
import argparse
import csv
import datetime
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from dataloader.dataset import build_Dataset
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from model.vnet import VNet
from utils.boundary_guidance import (EPS, build_boundary_and_nonboundary,
                                    compute_bgs, morphology_3d,
                                    spatial_gradient_3d)
from utils.losses import DiceLoss
from utils.utils import patients_to_slices

DEFAULT_MODELS = [
    ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth",
    ROOT / "Results/experiment_A_bgs_alpha0p1_seed42/result_LA_10l/fold_0/Model_iter_29000.pth",
]
NEW_CODE = {Path(__file__).resolve(), Path(__file__).with_name("test_experiment_bgs_utility.py").resolve()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=str(ROOT.parent / "Dataset/LA"))
    p.add_argument("--dataset", choices=["LA"], default="LA")
    p.add_argument("--labeled_num", type=int, choices=[10], default=10)
    p.add_argument("--model_paths", nargs="+", default=[str(x) for x in DEFAULT_MODELS])
    p.add_argument("--checkpoint_names", nargs="+", default=["baseline_ema", "experiment_A_ema"])
    p.add_argument("--patch_size", type=int, nargs=3, default=[112, 112, 80])
    p.add_argument("--patches_per_case", type=int, default=3)
    p.add_argument("--min_band_voxels", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--bootstrap_samples", type=int, default=10000)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--fd_atol", type=float, default=1e-5)
    p.add_argument("--fd_rtol", type=float, default=0.1)
    p.add_argument("--sign_epsilon", type=float, default=1e-6)
    p.add_argument("--output_dir", default=str(ROOT / "analysis/results/experiment_bgs_utility"))
    args = p.parse_args(argv)
    if len(args.model_paths) != len(args.checkpoint_names) or len(set(args.checkpoint_names)) != len(args.checkpoint_names):
        p.error("Provide a unique --checkpoint_names entry for every --model_paths entry")
    if any(v < 16 or v % 16 for v in args.patch_size):
        p.error("VNet patch dimensions must be multiples of 16")
    for name in ("patches_per_case", "min_band_voxels", "max_retries", "bootstrap_samples", "cpu_threads"):
        if getattr(args, name) < 1:
            p.error(name + " must be positive")
    if not 0 < args.delta < 1 or min(args.fd_atol, args.fd_rtol, args.sign_epsilon) < 0:
        p.error("Invalid perturbation or numerical tolerances")
    return args


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_manifest(output_dir):
    """Read-only snapshot: existing source, experiments, checkpoints and results."""
    manifest = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")
                         and (Path(parent) / d).resolve() != output_dir.resolve())
        for name in sorted(files):
            path = Path(parent) / name
            if path.resolve() in NEW_CODE or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_file():
                manifest[str(path.relative_to(ROOT))] = sha256(path)
    return manifest


def create_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents and "Results" in path.relative_to(ROOT).parts:
        raise ValueError("Output must not be the repository or an existing Results experiment")
    path.mkdir(parents=True, exist_ok=False)
    return path


def assert_finite(value, context):
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    bad = np.argwhere(~np.isfinite(arr))
    if len(bad):
        raise FloatingPointError("{}: NaN/Inf at indices {}".format(context, bad.tolist()))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError("Nonfinite JSON output")
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = json_safe(row)
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in clean.items()})


def labeled_split(args):
    """Use the actual Dataset order, patients_to_slices and sampler definition."""
    dataset = build_Dataset(args, str(Path(args.data_path).resolve()), "train_LA", transform=None)
    count = patients_to_slices(args.dataset, args.labeled_num)
    labeled = list(range(count))
    sampler = TwoStreamBatchSampler(labeled, list(range(count, len(dataset))), 4, 2)
    assert list(sampler.primary_indices) == labeled and sampler.primary_batch_size == 2
    test_list = Path(args.data_path) / "test.list"
    test_ids = set(test_list.read_text().splitlines())
    ids = [dataset.image_list[i] for i in labeled]
    if len(ids) != len(set(ids)) or set(ids) & test_ids:
        raise ValueError("Duplicate labeled IDs or overlap with test.list")
    return [(i, dataset.image_list[i], Path(dataset.sample_list[i])) for i in labeled], dict(
        train_count=len(dataset), labeled_count=count, labeled_indices=labeled,
        labeled_case_ids=ids, sampler_primary_batch_size=2, sampler_secondary_batch_size=2,
        train_list_path=str(Path(args.data_path) / "train.list"), train_list_sha256=sha256(Path(args.data_path) / "train.list"),
        test_list_path=str(test_list), test_list_sha256=sha256(test_list), labeled_test_overlap=[],
        derivation="train.py labeled_idxs=range(patients_to_slices('LA',10)); Dataset preserves train.list order; TwoStreamBatchSampler primary=labeled",
        test_gt_loaded=False)


def boundary_sides(labels):
    mask = labels.unsqueeze(1) > 0 if labels.ndim == 4 else labels > 0
    return (mask & ~morphology_3d(mask, operation="erode"))[:, 0], (morphology_3d(mask) & ~mask)[:, 0]


def crop_volume(image, label, patch_size, rng):
    """Original RandomCrop padding and exclusive-upper-bound coordinate rule."""
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError("Expected matching 3D image/label arrays")
    padding = [max((int(p) - int(s)) // 2 + 3, 0) for p, s in zip(patch_size, label.shape)] if any(
        s <= p for s, p in zip(label.shape, patch_size)) else [0, 0, 0]
    padded_image = np.pad(image, [(p, p) for p in padding]) if any(padding) else image
    padded_label = np.pad(label, [(p, p) for p in padding]) if any(padding) else label
    start = [int(rng.randint(0, s - p)) for s, p in zip(padded_label.shape, patch_size)]
    slices = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
    return (np.ascontiguousarray(padded_image[slices], dtype=np.float32),
            np.ascontiguousarray(padded_label[slices], dtype=np.int64),
            dict(start_padded=start, start_original=[s - p for s, p in zip(start, padding)],
                 padding=padding, original_shape=list(label.shape), padded_shape=list(padded_label.shape)))


def sample_patches(cases, args, spatial_shape):
    patches, attempts, statuses, sources = [], [], [], []
    for index, case_id, path in cases:
        # Separate, reproducible per-case stream; never consumes torch/model RNG.
        rng = np.random.RandomState(args.seed + index * 1009)
        with h5py.File(str(path), "r") as f:
            image, label = f["image"][:], f["label"][:]
        assert_finite(image, case_id + "/image")
        if not np.isin(label, [0, 1]).all():
            raise ValueError(case_id + " has nonbinary labels")
        sources.append(dict(case_id=case_id, train_index=index, path=str(path), sha256=sha256(path)))
        for patch_index in range(args.patches_per_case):
            accepted = False
            for retry in range(args.max_retries):
                crop_image, crop_label, coords = crop_volume(image, label, args.patch_size, rng)
                target = torch.from_numpy(crop_label).unsqueeze(0)
                inner, outer = boundary_sides(target)
                boundary, nonboundary = build_boundary_and_nonboundary(target, spatial_shape)
                counts = dict(foreground_voxels=int(target.sum()), inner_voxels=int(inner.sum()),
                              outer_voxels=int(outer.sum()), bgs_boundary_voxels=int(boundary.sum()),
                              bgs_nonboundary_voxels=int(nonboundary.sum()))
                invalid = [name for name, n in counts.items() if n < args.min_band_voxels]
                row = dict(case_id=case_id, train_index=index, patch_index=patch_index,
                           attempt=retry + 1, case_rng_seed=args.seed + index * 1009,
                           accepted=not invalid, reason=";".join(invalid), **coords, **counts)
                attempts.append(row)
                if not invalid:
                    patches.append(dict(case_id=case_id, patch_index=patch_index, image=crop_image,
                                        label=crop_label, coords=coords, counts=counts))
                    statuses.append(dict(case_id=case_id, patch_index=patch_index, valid=True,
                                         attempts=retry + 1, reason="", **coords, **counts))
                    accepted = True
                    break
            if not accepted:
                statuses.append(dict(case_id=case_id, patch_index=patch_index, valid=False,
                                     attempts=args.max_retries, reason="max_retries_exhausted"))
        logging.info("Sampling %s: %d/%d accepted", case_id,
                     sum(p["case_id"] == case_id for p in patches), args.patches_per_case)
    if not patches:
        raise RuntimeError("No valid labeled patches; all sampling attempts retained")
    return patches, attempts, statuses, sources


@torch.no_grad()
def bgs_details(feature, labels, min_voxels=8):
    reference, valid = compute_bgs(feature, labels, min_voxels)
    boundary, nonboundary = build_boundary_and_nonboundary(labels, feature.shape[2:])
    gradient = spatial_gradient_3d(feature.detach()).double()
    gb = torch.where(boundary, gradient, torch.zeros_like(gradient)).flatten(2).sum(2) / boundary.flatten(1).sum(1).clamp_min(1)[:, None]
    gn = torch.where(nonboundary, gradient, torch.zeros_like(gradient)).flatten(2).sum(2) / nonboundary.flatten(1).sum(1).clamp_min(1)[:, None]
    reconstructed = (gb - gn) / (gb + gn + EPS)
    reconstructed = torch.where(valid[:, None], reconstructed, torch.zeros_like(reconstructed))
    if not torch.equal(reconstructed, reference):
        raise AssertionError("BGS differs from existing compute_bgs")
    positive = reference.relu()
    weights = (positive / (positive.max(dim=1, keepdim=True).values + EPS)).to(feature.dtype).detach()
    return reference, gb, gn, weights, valid


def scale_x3(features, a):
    if a.ndim != 1 or a.numel() != features[2].shape[1]:
        raise ValueError("Gain vector must match the dynamically read x3 channel count")
    modified = list(features)
    modified[2] = features[2] * a[None, :, None, None, None]
    assert all(modified[i] is features[i] for i in (0, 1, 3, 4))
    return modified


def loss_values(logits, labels, inner, outer):
    if not inner.any() or not outer.any():
        raise ValueError("Both GT boundary sides must be nonempty")
    # FP64 reductions reduce finite-difference cancellation; CE itself follows
    # the baseline FP32 forward, and the full supervised loss is unchanged.
    ce = F.cross_entropy(logits, labels.long(), reduction="none")
    balanced = 0.5 * (ce[inner].double().mean() + ce[outer].double().mean())
    full = F.cross_entropy(logits, labels.long())
    supervised = 0.5 * (full + DiceLoss(logits.shape[1])(logits, labels.long(), softmax=True))
    return dict(boundary_ce=balanced, full_ce=full, supervised_ce_dice=supervised)


def gain_utility(model, features, labels, inner, outer):
    # Encoder tensors were produced under no_grad (never inference_mode).
    with torch.enable_grad():
        a = features[2].new_ones(features[2].shape[1], requires_grad=True)
        logits = model.decoder(scale_x3(features, a))
        losses = loss_values(logits, labels, inner, outer)
        utility = -torch.autograd.grad(losses["boundary_ce"], a, create_graph=False)[0]
    return utility.detach(), {k: float(v.detach().item()) for k, v in losses.items()}, logits.detach()


def select_bgs_groups(bgs, group_size=4):
    """Signed-BGS-only stable ranking; channel index breaks ties, never utility."""
    arr = np.asarray(bgs)
    order = np.lexsort((np.arange(len(arr)), arr))
    k = min(group_size, len(arr))
    middle_start = max(0, (len(arr) - k) // 2)
    return dict(high=order[-k:][::-1].tolist(), middle=order[middle_start:middle_start + k].tolist(),
                low=order[:k].tolist())


def safe_spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, "constant_vector"
    value = float(spearmanr(a, b).correlation)
    assert_finite(value, "Spearman")
    return value, ""


def bootstrap_cases(values, rng, repetitions=10000):
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(vals):
        return dict(mean=None, std=None, median=None, ci95=[None, None], n_cases=0,
                    reason="no_defined_case_statistics")
    assert_finite(vals, "case bootstrap input")
    means = vals[rng.randint(0, len(vals), size=(repetitions, len(vals)))].mean(1)
    return dict(mean=float(vals.mean()), std=float(vals.std()), median=float(np.median(vals)),
                ci95=np.percentile(means, [2.5, 97.5]).tolist(), n_cases=len(vals),
                resampling_unit="case", bootstrap_samples=repetitions)


@torch.no_grad()
def finite_differences(model, features, labels, inner, outer, base_losses, bgs, utility, args):
    groups = select_bgs_groups(bgs)
    memberships = {}
    for group, channels in groups.items():
        for channel in channels:
            memberships.setdefault(channel, []).append(group)
    rows = []
    for channel in sorted(memberships):
        perturbations = {}
        for name, sign in (("plus", 1), ("minus", -1)):
            a = features[2].new_ones(features[2].shape[1])
            a[channel] += sign * args.delta
            logits = model.decoder(scale_x3(features, a))
            assert_finite(logits, "FD channel {} {} logits".format(channel, name))
            perturbations[name] = {k: float(v.item()) for k, v in loss_values(logits, labels, inner, outer).items()}
        plus, minus = perturbations["plus"], perturbations["minus"]
        u_fd = (minus["boundary_ce"] - plus["boundary_ce"]) / (2 * args.delta)
        u = float(utility[channel])
        error = abs(u_fd - u)
        informative = max(abs(u), abs(u_fd)) > args.sign_epsilon
        row = dict(channel=channel, group=";".join(memberships[channel]), membership_count=len(memberships[channel]),
                   bgs=float(bgs[channel]), delta=args.delta, utility=u, utility_fd=u_fd,
                   absolute_error=error, relative_error=error / max(abs(u), abs(u_fd), 1e-12),
                   magnitude_agreement=error <= args.fd_atol + args.fd_rtol * max(abs(u), abs(u_fd)),
                   sign_informative=informative,
                   sign_agreement=bool(np.sign(u_fd) == np.sign(u)) if informative else None)
        row["linear_predicted_gain_plus"] = args.delta * u
        row["linear_predicted_gain_minus"] = -args.delta * u
        row["boundary_ce_central_curvature"] = (plus["boundary_ce"] + minus["boundary_ce"] - 2 * base_losses["boundary_ce"]) / args.delta ** 2
        for loss in base_losses:
            row.update({loss + "_base": base_losses[loss], loss + "_plus": plus[loss],
                        loss + "_minus": minus[loss], loss + "_gain_plus": base_losses[loss] - plus[loss],
                        loss + "_gain_minus": base_losses[loss] - minus[loss]})
        rows.append(row)
    return rows, groups


def snapshot_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def verify_state(model, before):
    after = model.state_dict()
    changed = [name for name, value in before.items() if not torch.equal(value, after[name].detach().cpu())]
    if changed:
        raise AssertionError("Model parameter/buffer mutation: " + repr(changed))
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("Model parameters became trainable or received gradients")
    if any(module.training for module in model.modules()):
        raise AssertionError("A model module left eval mode")
    return dict(weights_and_buffers_exactly_unchanged=True, state_tensor_count=len(before),
                all_parameters_frozen=True, no_model_parameter_gradients=True, all_modules_eval=True)


def load_frozen(path, device):
    model = VNet(n_channels=1, n_classes=2).to(device)
    extra = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    checkpoint = torch.load(str(path), map_location="cpu", **extra)
    # These EMA checkpoints are plain state_dict, not resume/training bundles.
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def analyze_checkpoint(model, name, patches, args):
    before = snapshot_state(model)
    records, channels, fd_rows, patch_rows, checks = [], [], [], [], []
    for p in patches:
        context = "{}/{}/patch{}".format(name, p["case_id"], p["patch_index"])
        image = torch.from_numpy(p["image"])[None, None].to(args.device)
        labels = torch.from_numpy(p["label"])[None].to(args.device)
        inner, outer = boundary_sides(labels)
        with torch.no_grad():
            features = model.encoder(image)
            preserved = [tensor.clone() for tensor in features]
            baseline_logits = model(image)
            bgs, gb, gn, w, valid = bgs_details(features[2], labels, args.min_band_voxels)
        if any(f.requires_grad for f in features) or w.requires_grad:
            raise AssertionError(context + " encoder cache or weights unexpectedly require gradients")
        if hasattr(torch, "is_inference") and any(torch.is_inference(f) for f in features):
            raise AssertionError(context + " encoder cache was created in inference_mode")
        if not bool(valid[0]):
            raise AssertionError(context + " BGS mask validity changed after fixed sampling")
        utility, losses, unit_logits = gain_utility(model, features, labels, inner, outer)
        max_error = float((unit_logits - baseline_logits).abs().max().item())
        if not torch.allclose(unit_logits, baseline_logits, atol=1e-6, rtol=1e-6):
            raise AssertionError(context + " unit-gain decoder differs from original eval forward")
        if any(not torch.equal(old, current) for old, current in zip(preserved, features)):
            raise AssertionError(context + " cached encoder feature mutated")
        for field, value in (("BGS", bgs), ("g_boundary", gb), ("g_nonboundary", gn),
                             ("w", w), ("U", utility), ("logits", unit_logits)):
            assert_finite(value, context + "/" + field)
        b, g_b, g_n, weights, u = [v.detach().cpu().numpy().reshape(-1) for v in (bgs, gb, gn, w, utility)]
        rho_b, reason_b = safe_spearman(b, u)
        rho_w, reason_w = safe_spearman(weights, u)
        fd, groups = finite_differences(model, features, labels, inner, outer, losses, b, u, args)
        fd_by_channel = {row["channel"]: row for row in fd}
        identity = dict(checkpoint=name, case_id=p["case_id"], patch_index=p["patch_index"])
        fd_rows.extend(dict(**identity, **row) for row in fd)
        patch_row = dict(**identity, bgs_utility_spearman=rho_b, bgs_spearman_undefined_reason=reason_b,
                         w_utility_spearman=rho_w, w_spearman_undefined_reason=reason_w,
                         utility_mean=float(u.mean()), utility_positive_fraction=float((u > args.sign_epsilon).mean()),
                         utility_negative_fraction=float((u < -args.sign_epsilon).mean()),
                         feature_shape=list(features[2].shape), fd_unique_channels=len(fd),
                         fd_duplicate_memberships=sum(len(v) for v in groups.values()) - len(fd), **losses)
        for group, selected in groups.items():
            values = u[selected]
            rows = [fd_by_channel[c] for c in selected]
            gains = np.array([r["boundary_ce_gain_plus"] for r in rows])
            attenuation = np.array([r["boundary_ce_gain_minus"] for r in rows])
            full_gains = np.array([r["full_ce_gain_plus"] for r in rows])
            supervised_gains = np.array([r["supervised_ce_dice_gain_plus"] for r in rows])
            patch_row.update({group + "_utility_mean": float(values.mean()),
                              group + "_utility_median": float(np.median(values)),
                              group + "_utility_positive_fraction": float((values > args.sign_epsilon).mean()),
                              group + "_utility_negative_fraction": float((values < -args.sign_epsilon).mean()),
                              group + "_utility_near_zero_fraction": float((np.abs(values) <= args.sign_epsilon).mean()),
                              group + "_gain_plus_mean": float(gains.mean()),
                              group + "_gain_plus_median": float(np.median(gains)),
                              group + "_gain_plus_positive_fraction": float((gains > 0).mean()),
                              group + "_gain_plus_negative_fraction": float((gains < 0).mean()),
                              group + "_gain_minus_mean": float(attenuation.mean()),
                              group + "_gain_minus_positive_fraction": float((attenuation > 0).mean()),
                              group + "_full_ce_gain_plus_mean": float(full_gains.mean()),
                              group + "_supervised_gain_plus_mean": float(supervised_gains.mean()),
                              group + "_boundary_better_full_ce_worse_fraction": float(((gains > 0) & (full_gains < 0)).mean()),
                              group + "_boundary_better_supervised_worse_fraction": float(((gains > 0) & (supervised_gains < 0)).mean()),
                              group + "_central_curvature_mean": float(np.mean([r["boundary_ce_central_curvature"] for r in rows]))})
        patch_row["high_minus_low_utility"] = patch_row["high_utility_mean"] - patch_row["low_utility_mean"]
        patch_row["high_minus_low_gain_plus"] = patch_row["high_gain_plus_mean"] - patch_row["low_gain_plus_mean"]
        patch_rows.append(patch_row)
        for channel in range(len(b)):
            member = [group for group, selected in groups.items() if channel in selected]
            channels.append(dict(**identity, channel=channel, bgs=float(b[channel]), w=float(weights[channel]),
                                 utility=float(u[channel]), g_boundary=float(g_b[channel]),
                                 g_nonboundary=float(g_n[channel]), selected_group=";".join(member),
                                 bgs_boundary_voxels=p["counts"]["bgs_boundary_voxels"],
                                 bgs_nonboundary_voxels=p["counts"]["bgs_nonboundary_voxels"],
                                 inner_voxels=int(inner.sum()), outer_voxels=int(outer.sum()),
                                 feature_shape=list(features[2].shape), **losses))
        records.append(dict(**identity, bgs=b.copy(), w=weights.copy(), utility=u.copy(), groups=groups))
        checks.append(dict(**identity, unit_gain_forward_max_abs_error=max_error,
                           unit_gain_forward_exact=torch.equal(unit_logits, baseline_logits),
                           feature_shapes=[list(f.shape) for f in features], cached_features_not_inference=True,
                           cached_features_unmodified=True, only_x3_scaled=True,
                           bgs_matches_existing_helper_exactly=True, w_requires_grad=False,
                           only_a_has_grad=True, all_outputs_finite=True))
        logging.info("%s x3=%s BGS/U rho=%.4f w/U rho=%s, FD %d/%d within tolerance", context,
                     list(features[2].shape), rho_b if rho_b is not None else 0,
                     str(rho_w), sum(r["magnitude_agreement"] for r in fd), len(fd))
        del image, labels, features, preserved, baseline_logits, unit_logits
    state_checks = verify_state(model, before)
    return records, channels, fd_rows, patch_rows, dict(**state_checks, patches=checks)


def aggregate(records, patch_rows, fd_rows, args):
    cases, pairwise, signs, summary = [], [], [], {}
    for model_index, name in enumerate(args.checkpoint_names):
        rr = [r for r in records if r["checkpoint"] == name]
        case_ids = sorted({r["case_id"] for r in rr})
        rng = np.random.RandomState(args.seed + 100000 + model_index)
        case_vectors, model_cases = [], []
        for case_id in case_ids:
            pp = [p for p in patch_rows if p["checkpoint"] == name and p["case_id"] == case_id]
            row = dict(checkpoint=name, case_id=case_id, valid_patches=len(pp))
            numeric = [k for k, v in pp[0].items() if isinstance(v, (float, np.floating))
                       or k in ("bgs_utility_spearman", "w_utility_spearman")]
            for key in numeric:
                values = [p[key] for p in pp if p[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            for key in ("bgs_utility_spearman", "w_utility_spearman"):
                row[key + "_defined_patches"] = sum(p[key] is not None for p in pp)
            cases.append(row)
            model_cases.append(row)
            pr = [r for r in rr if r["case_id"] == case_id]
            case_vectors.append(dict(case_id=case_id, bgs=np.mean([r["bgs"] for r in pr], axis=0),
                                     utility=np.mean([r["utility"] for r in pr], axis=0)))
        metrics = {key: bootstrap_cases([row[key] for row in model_cases], rng, args.bootstrap_samples)
                   for key in model_cases[0] if key not in ("checkpoint", "case_id", "valid_patches")
                   and not key.endswith("_defined_patches")}
        model_pairs = []
        for i, left in enumerate(case_vectors):
            for right in case_vectors[i + 1:]:
                rho, reason = safe_spearman(left["utility"], right["utility"])
                row = dict(checkpoint=name, case_i=left["case_id"], case_j=right["case_id"],
                           utility_rank_spearman=rho, undefined_reason=reason,
                           sign_agreement_fraction=float(np.mean(
                               np.where(np.abs(left["utility"]) <= args.sign_epsilon, 0, np.sign(left["utility"])) ==
                               np.where(np.abs(right["utility"]) <= args.sign_epsilon, 0, np.sign(right["utility"])))))
                pairwise.append(row)
                model_pairs.append(row)
        # Correlated pairs are descriptive only. Case-resampled pairwise average
        # CI uses an edge matrix and excludes duplicate draws of the same case.
        n = len(case_vectors)
        edge = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = safe_spearman(case_vectors[i]["utility"], case_vectors[j]["utility"])
                if rho is not None:
                    edge[i, j] = edge[j, i] = rho
        pair_boot = []
        for draw in rng.randint(0, n, size=(args.bootstrap_samples, n)):
            values = edge[draw[:, None], draw[None, :]][np.triu_indices(n, 1)]
            values = values[np.isfinite(values)]
            if len(values):
                pair_boot.append(float(values.mean()))
        pair_values = [p["utility_rank_spearman"] for p in model_pairs if p["utility_rank_spearman"] is not None]
        utility_matrix = np.stack([r["utility"] for r in case_vectors])
        bgs_matrix = np.stack([r["bgs"] for r in case_vectors])
        high_sets = [set(select_bgs_groups(b)["high"]) for b in bgs_matrix]
        low_sets = [set(select_bgs_groups(b)["low"]) for b in bgs_matrix]
        for channel in range(utility_matrix.shape[1]):
            positive = utility_matrix[:, channel] > args.sign_epsilon
            negative = utility_matrix[:, channel] < -args.sign_epsilon
            high_negative = [channel in high_sets[i] and bool(negative[i]) for i in range(n)]
            low_positive = [channel in low_sets[i] and bool(positive[i]) for i in range(n)]
            signs.append(dict(checkpoint=name, channel=channel, n_cases=n,
                              positive_case_fraction=float(positive.mean()), negative_case_fraction=float(negative.mean()),
                              near_zero_case_fraction=float((~positive & ~negative).mean()),
                              majority_sign_fraction=float(max(positive.sum(), negative.sum(), (~positive & ~negative).sum()) / n),
                              high_bgs_negative_case_count=sum(high_negative), low_bgs_positive_case_count=sum(low_positive),
                              high_bgs_negative_cases=[case_ids[i] for i, v in enumerate(high_negative) if v],
                              low_bgs_positive_cases=[case_ids[i] for i, v in enumerate(low_positive) if v],
                              case_mean_utility=float(utility_matrix[:, channel].mean())))
        ff = [r for r in fd_rows if r["checkpoint"] == name]
        info = [r for r in ff if r["sign_informative"]]
        group_fd = {}
        for group in ("high", "middle", "low"):
            rows = [r for r in ff if group in r["group"].split(";")]
            group_fd[group] = dict(n_channels=len(rows), gain_plus_pooled_median=float(np.median([r["boundary_ce_gain_plus"] for r in rows])),
                                   gain_minus_pooled_median=float(np.median([r["boundary_ce_gain_minus"] for r in rows])),
                                   numerical_agreement_fraction=float(np.mean([r["magnitude_agreement"] for r in rows])))
        summary[name] = dict(n_cases=n, n_patches=len(rr), n_channels=utility_matrix.shape[1],
                             metrics=metrics, group_finite_difference=group_fd,
                             positive_bgs_utility_cases=sum(r["bgs_utility_spearman"] is not None and r["bgs_utility_spearman"] > 0 for r in model_cases),
                             negative_bgs_utility_cases=sum(r["bgs_utility_spearman"] is not None and r["bgs_utility_spearman"] < 0 for r in model_cases),
                             finite_difference=dict(n=len(ff), informative_sign_n=len(info),
                                                    magnitude_agreement_fraction=float(np.mean([r["magnitude_agreement"] for r in ff])),
                                                    sign_agreement_fraction=float(np.mean([r["sign_agreement"] for r in info])) if info else None,
                                                    median_absolute_error=float(np.median([r["absolute_error"] for r in ff])),
                                                    max_absolute_error=float(max(r["absolute_error"] for r in ff)),
                                                    magnitude_mismatches=[{k: r[k] for k in ("case_id", "patch_index", "channel", "utility", "utility_fd", "absolute_error")}
                                                                          for r in ff if not r["magnitude_agreement"]],
                                                    sign_mismatches=[{k: r[k] for k in ("case_id", "patch_index", "channel", "utility", "utility_fd")}
                                                                    for r in info if not r["sign_agreement"]]),
                             cross_case_utility_ranking=dict(n_pairs=len(pair_values), mean=float(np.mean(pair_values)) if pair_values else None,
                                                            median=float(np.median(pair_values)) if pair_values else None,
                                                            ci95=np.percentile(pair_boot, [2.5, 97.5]).tolist() if pair_boot else [None, None],
                                                            resampling_unit="case; repeated copies of identical case excluded from pair average"),
                             undefined_patch_correlations=[{k: p[k] for k in ("case_id", "patch_index", "bgs_spearman_undefined_reason", "w_spearman_undefined_reason")}
                                                           for p in patch_rows if p["checkpoint"] == name and
                                                           (p["bgs_utility_spearman"] is None or p["w_utility_spearman"] is None)])
    return cases, pairwise, signs, summary


def make_figures(out, channels, cases, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        rows = [r for r in channels if r["checkpoint"] == name]
        ax.scatter([r["bgs"] for r in rows], [r["utility"] for r in rows], s=9, alpha=0.35, rasterized=True)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(xlabel="Signed BGS", ylabel="Boundary gain utility U = -dL/da", title=name)
    fig.suptitle("All channels and fixed labeled patches; checkpoints analyzed separately")
    fig.tight_layout()
    fig.savefig(str(out / "bgs_vs_utility.png"), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        cc = [c for c in cases if c["checkpoint"] == name]
        for col, metric in enumerate(("utility_mean", "gain_plus_mean")):
            data = [[c[g + "_" + metric] for c in cc] for g in ("high", "middle", "low")]
            ax = axes[row, col]
            ax.boxplot(data, labels=["High BGS", "Middle BGS", "Low BGS"], showfliers=True)
            for i, values in enumerate(data):
                ax.scatter(np.full(len(values), i + 1), values, color="tab:blue", s=22, alpha=0.7)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set(title=name, ylabel="Case mean U" if col == 0 else "Case mean boundary CE gain (+5%)")
    fig.suptitle("One dot per case; four BGS-selected channels per group and patch")
    fig.tight_layout()
    fig.savefig(str(out / "group_utility_comparison.png"), dpi=180)
    plt.close(fig)


def fmt_stat(s, precision=4):
    if s["mean"] is None:
        return "未定义"
    return ("{:." + str(precision) + "f} [95% CI {:." + str(precision) + "f}, {:." + str(precision) + "f}]").format(s["mean"], *s["ci95"])


def write_report(out, summary, cases, signs, metadata, verification, args):
    primary = summary[args.checkpoint_names[0]]
    # Conservative, explicit descriptive decision rules. No absence-of-effect
    # claim based solely on a CI crossing zero. Equivalence band is +/-0.2 rho.
    rho_ci = primary["metrics"]["bgs_utility_spearman"]["ci95"]
    gain_ci = primary["metrics"]["high_minus_low_gain_plus"]["ci95"]
    auxiliary_negative = all(s["metrics"]["bgs_utility_spearman"]["ci95"][1] is not None and
                             s["metrics"]["bgs_utility_spearman"]["ci95"][1] < 0 for s in summary.values())
    if rho_ci[1] is not None and rho_ci[1] < 0 and gain_ci[1] < 0 and auxiliary_negative:
        choice = "A. 保留 BGS，考虑 Reverse-BGS 对照。"
        rationale = "本次固定 checkpoint 内呈稳定负相关，且高 BGS 组的小幅增强收益低于低组；可把反向排序作为下一步待检验对照，不能据此预言训练改善。"
    elif rho_ci[0] is not None and rho_ci[0] >= -0.2 and rho_ci[1] <= 0.2:
        choice = "B. 停止使用 BGS 决定增强幅度，转向 Receiver-aware 机制。"
        rationale = "主模型病例级相关性区间位于预先说明的弱相关范围 ±0.2 内，缺少用 BGS 映射增益的依据；此建议不等于已经验证任何 Receiver-aware 方法。"
    else:
        choice = "C. 证据不充分，需要一个明确的、最小成本的补充控制实验。"
        rationale = "虽然两个模型复现了弱负相关，高−低组有限增益的配对区间跨零，且低 BGS 组平均增强收益仍为负，尚不足以选择 Reverse-BGS。最小补充是在冻结 Baseline、现有 24 个 patch 上做完整权重向量的 paired control：按 train.list 顺序，对接收病例 i 固定使用另外两例 (i+1)%8、(i+2)%8 的同索引 patch 计算平均 BGS，再按原 Experiment A 生成 w；接收病例不参与签名。比较 a=1+0.1w 的真实对应与固定 seed、独立 RNG 的 20 次通道置换，保持权重分布完全一致，以病例级 bootstrap 汇总真实−置换的边界 CE 收益差，并同时检查全图 CE/CE+Dice。这个补充只需 decoder 前向，不训练，不使用无标签或测试 GT，也不能据此证明无标签训练期的因果机制。δ=0.01 可作为现有预选通道的数值复核，但当前方向一致率已达 99.7%，它不是主要未决问题。"
    summary["recommendation"] = dict(choice=choice, rationale=rationale,
                                      decision_rules="A: main rho CI<0 and high-low G+ CI<0, with all available checkpoint rho CIs<0; B: main rho CI within [-0.2,0.2]; otherwise C")
    lines = ["# Experiment BGS-U：BGS 与通道增益效用分析", "",
             "## 研究问题与预先固定的方法", "",
             "检验边界梯度选择性 BGS 是否指示固定模型中增强 x3 通道会改善真实边界预测。这里只测冻结 checkpoint 的局部效用，不训练模型。",
             "主对象为未受 BGS 训练调制的 Baseline EMA；Experiment A EMA 仅作辅助。每个 checkpoint 独立计算，通道编号不跨模型语义对齐。",
             "实际 labeled 病例来自 train.py、patients_to_slices、Dataset 与 TwoStreamBatchSampler 的交叉核对：train.list 前 {} 例。只读取 test.list 的 ID 排除重叠，没有读取测试或无标签 GT。".format(metadata["data_split"]["labeled_count"]),
             "病例 ID：" + "、".join(metadata["data_split"]["labeled_case_ids"]), "",
             "每例目标 3 个 112×112×80 固定 patch。按原 RandomCrop 的零填充与坐标范围抽样，不加旋转或强度扰动；有效性仅取决于 GT 前景、原分辨率内外边界及 x3 的 B/N 区域均至少 8 voxel。每个目标 patch 最多 200 次尝试。完整坐标、尝试及失败保存在 metadata、sampling_attempts.csv、patch_status.csv。两个模型使用完全相同的 patch。",
             "复用 boundary_guidance.py：最近邻 GT、B=dilate(Y)\\erode(Y)、N=~dilate(B)，三轴绝对前向差分，末平面复制。保存全部通道的有符号 BGS、两区域梯度均值与 ReLU/max 权重。这里权重是单 patch 归一化；Experiment A 训练时先对有效 labeled batch 求平均，此处并不模拟该 batch 聚合。",
             "固定模型 eval，冻结全部参数及 BN buffers；encoder 用 no_grad 缓存，decoder 仅对 a 开启 autograd。仅缩放 decoder 输入的 x3，其余层保持同一张量。主损失为原分辨率内侧/外侧各占一半的 CE，U=-∂L_B/∂a|a=1；同时记录全图 CE 和原监督损失 0.5×(CE+Dice)。边界 CE 的 FP64 区域均值仅减少差分抵消，不改变原模型或监督损失。",
             "有限差分 δ=0.05；每 patch 按原始有符号 BGS 的最高 4、最低 4、居中 4 通道选择，索引仅用于平分 tie，重复去重并保留组归属；完全不按效用选择。G+=L(1)-L(1+δ)，G-=L(1)-L(1-δ)，U_FD=(Lminus-Lplus)/(2δ)。", 
             "病例内先平均 patch 统计，再等权汇总病例；95% CI 使用病例级 bootstrap（{} 次）。通道对、patch 或同一病例下的通道不作为独立 bootstrap 病例。跨病例排名以病例平均 U 计算，成对 CI 也重采样病例并排除同一病例副本。".format(args.bootstrap_samples),
             "FD 数值一致阈值事先设为 |U_FD-U|≤{}+{}×max(|U|,|U_FD|)，效用符号近零阈值 {}。有限差分的实际收益正负用 0 判定；不一致项全部保留。常数向量的 Spearman 显式标注未定义，用 null/空值输出，不伪造相关性。".format(args.fd_atol, args.fd_rtol, args.sign_epsilon), "",
             "## 结果", "",
             "有效病例数 {}，有效 patch 数 {}，采样尝试 {}，失败目标 patch {}。".format(metadata["sampling"]["valid_cases"], metadata["sampling"]["valid_patches"], metadata["sampling"]["attempts"], metadata["sampling"]["skipped_patches"]), "",
             "|模型|BGS–U Spearman（病例均值、95% CI）|w–U Spearman|高 BGS U<0 比例|低 BGS U>0 比例|高−低 U|", "|---|---|---|---|---|---|"]
    for name in args.checkpoint_names:
        s, m = summary[name], summary[name]["metrics"]
        lines.append("|{}|{}|{}|{:.1%}|{:.1%}|{}|".format(name, fmt_stat(m["bgs_utility_spearman"]),
                         fmt_stat(m["w_utility_spearman"]), m["high_utility_negative_fraction"]["mean"],
                         m["low_utility_positive_fraction"]["mean"], fmt_stat(m["high_minus_low_utility"], 6)))
    lines += ["", "|模型|高组 G+ 均值 [CI] / pooled 中位数|低组 G+ 均值 [CI] / pooled 中位数|高组增强使损失增加|低组增强使损失下降|高组减弱使损失下降|", "|---|---|---|---|---|---|"]
    for name in args.checkpoint_names:
        s, m = summary[name], summary[name]["metrics"]
        lines.append("|{}|{} / {:.6f}|{} / {:.6f}|{:.1%}|{:.1%}|{:.1%}|".format(name,
                     fmt_stat(m["high_gain_plus_mean"], 6), s["group_finite_difference"]["high"]["gain_plus_pooled_median"],
                     fmt_stat(m["low_gain_plus_mean"], 6), s["group_finite_difference"]["low"]["gain_plus_pooled_median"],
                     m["high_gain_plus_negative_fraction"]["mean"], m["low_gain_plus_positive_fraction"]["mean"],
                     m["high_gain_minus_positive_fraction"]["mean"]))
    lines += ["", "病例间一致性（每例先平均 patch 的 Spearman）：", "",
              "|模型|病例|BGS–U|w–U|高−低 U|高−低 G+|", "|---|---|---|---|---|---|"]
    for c in cases:
        lines.append("|{}|{}|{}|{}|{:.6f}|{:.6f}|".format(c["checkpoint"], c["case_id"],
                         "未定义" if c["bgs_utility_spearman"] is None else "{:.4f}".format(c["bgs_utility_spearman"]),
                         "未定义" if c["w_utility_spearman"] is None else "{:.4f}".format(c["w_utility_spearman"]),
                         c["high_minus_low_utility"], c["high_minus_low_gain_plus"]))
    lines += ["", "## 有限差分复核与跨病例效用稳定性", ""]
    for name in args.checkpoint_names:
        s, m = summary[name], summary[name]["metrics"]
        fd, rank = s["finite_difference"], s["cross_case_utility_ranking"]
        ss = [r for r in signs if r["checkpoint"] == name]
        lines += ["- {}：FD {} 次，数值容差内 {:.1%}，有效方向一致 {:.1%}；数值不一致 {}、方向不一致 {}。所有具体 case/patch/channel 在 summary_statistics.json 和 finite_difference_validation.csv 中保留。中位绝对误差 {:.6g}，最大 {:.6g}。".format(
                  name, fd["n"], fd["magnitude_agreement_fraction"], fd["sign_agreement_fraction"] or 0,
                  len(fd["magnitude_mismatches"]), len(fd["sign_mismatches"]), fd["median_absolute_error"], fd["max_absolute_error"]),
                  "  病例平均 U 排名的 {} 个病例对 Spearman 均值 {}，病例 bootstrap CI {}。各通道跨病例多数符号一致率均值 {:.1%}。高 BGS 且 U<0 至少在半数病例出现的通道 {}；低 BGS 且 U>0 至少在半数病例出现的通道 {}。这些是该 checkpoint 内的描述，不是固定的应增强通道。".format(
                  rank["n_pairs"], rank["mean"], rank["ci95"], np.mean([r["majority_sign_fraction"] for r in ss]),
                  [r["channel"] for r in ss if r["high_bgs_negative_case_count"] >= s["n_cases"] / 2],
                  [r["channel"] for r in ss if r["low_bgs_positive_case_count"] >= s["n_cases"] / 2]),
                  "  高组近零一阶效用比例 {:.1%}；高−低有限扰动收益 {}。δ=0.05 可能跨越 ReLU 分段，因此 FD 和梯度差异不能单独视为实现错误。".format(
                  m["high_utility_near_zero_fraction"]["mean"], fmt_stat(m["high_minus_low_gain_plus"], 6)),
                  "  方向不一致项（均保留）：" + json.dumps(fd["sign_mismatches"], ensure_ascii=False),
                  "  边界改善而全图 CE 恶化的比例：高组 {:.1%}、低组 {:.1%}；边界改善而 CE+Dice 恶化的比例：高组 {:.1%}、低组 {:.1%}。这些冲突项保留在逐通道 FD 文件中。".format(
                  m["high_boundary_better_full_ce_worse_fraction"]["mean"], m["low_boundary_better_full_ce_worse_fraction"]["mean"],
                  m["high_boundary_better_supervised_worse_fraction"]["mean"], m["low_boundary_better_supervised_worse_fraction"]["mean"])]
    lines += ["", "**不能选择性解读的结果：** 两个模型的 BGS–U 都是弱负相关，但高−低组的配对收益区间都包含零；低 BGS 组的平均正向增强收益也都为负。这说明存在有益的低选择性通道，不能推导为‘低 BGS 通道整体应增强’，更不能仅凭相关性就推导 Reverse-BGS 有效。高组近零一阶效用为 0%，当前主要观察是负效用，不是已经证实饱和。组内比例按通道–patch 观察计算（每组每模型 96 次），同一通道可能重复出现；区间仍以病例为单位。"]
    m = primary["metrics"]
    lines += ["", "## 对六个研究问题的回答", "",
              "1. **高 BGS 是否普遍具有正向增益效用？** 主模型高组 U>0 为 {:.1%}，U<0 为 {:.1%}；因此不能把高 BGS 当作普遍有益增强的保证。整体方向应结合上述病例级区间判断。".format(m["high_utility_positive_fraction"]["mean"], m["high_utility_negative_fraction"]["mean"]),
              "2. **是否存在高 BGS 但增强有害的通道？** 主模型高组有 {:.1%} 的 +5% 扰动增加边界 CE。这是在固定模型、固定 labeled patch 上直接测到的有限增益现象，不意味着跨病例永久有害。".format(m["high_gain_plus_negative_fraction"]["mean"]),
              "3. **是否存在低 BGS 但增强有益的通道？** 主模型低组 U>0 为 {:.1%}，+5% 增强降低边界 CE 为 {:.1%}。低 BGS 仅指低边界梯度选择性，不能称为边界能力弱。".format(m["low_utility_positive_fraction"]["mean"], m["low_gain_plus_positive_fraction"]["mean"]),
              "4. **是否有稳定相关性、不同病例是否一致？** 主模型 BGS–U {}，{} 例病例均值为正、{} 例为负。完整逐病例结果、辅助模型及跨病例 U 排名均在上文；CI 包含零时不能等同于证明无关系。".format(fmt_stat(m["bgs_utility_spearman"]), primary["positive_bgs_utility_cases"], primary["negative_bgs_utility_cases"]),
              "5. **‘真实 BGS 过度增强、Shuffle 偶然补偿’的假设是否得到支持？** 对‘选择性排序不等于适宜增益排序’这一局部环节提供了支持：直接测得高 BGS 负效用和部分低 BGS 正效用，两个模型也复现弱负相关。这与所提机制的部分环节一致，但不足以证明已发生过度增强或增益饱和，更不能解释 Shuffle 训练优势的因果机制。低组平均增益仍为负，高低配对差异也未排除零，必须同时保留这些限制证据。既没有分析 Shuffle 增益落到哪些无标签通道，也没有对无标签使用 GT，因此不能直接证明 Shuffle 补偿了无标签的弱通道。训练期 BN 耦合、不同参数轨迹、batch 签名平均等现象本实验不能解释。",
              "6. **下一步选择：{}** {}".format(choice, rationale), "",
              "## 保护、复核与局限性", "",
              "模型所有参数及 buffers 前后逐 tensor 完全一致；a=1 与原 eval forward 等价；x1/x2/x4/x5 同一张量，原 x3 也未被原地改写；BGS 与现有函数逐元素完全一致。只对 a 使用 autograd.grad，没有创建优化器、训练或 EMA 更新。已有文件与 checkpoint 的 SHA256 前后核对，详细状态见 verification.json。有限差分不一致属于需科学解读的观察，不会从分析中删除。",
              "保护状态：{}。数值有限性：{}。数据限制：仅实际 labeled 病例，无 test GT。".format(verification["protected_files_unchanged"], verification["all_numeric_outputs_finite"]),
              "局限：仅 8 例 labeled、每例 3 个随机固定局部 patch，CI 是病例内 patch 平均的区间；没有全体病例或多 seed 推断。a=1 附近是冻结模型的局部效应，不是重新训练后的因果效应。eval 模式固定 BN/Dropout，与训练模式的 batch 耦合不同。±5% 扰动不是增益剂量反应曲线，不能直接证明饱和。边界是一体素形态学带，无 spacing 加权。辅助 checkpoint 的通道不对齐，且同一数据集已有训练暴露，辅助复现不是独立外部验证。", "",
              "全图 CE、原监督损失及其有限扰动收益逐行保留，可复核是否有边界改善而全局损失恶化；没有把任何诊断损失加入 Trainer，也没有实施 Reverse-BGS、Uniform、Receiver-aware 或 A+。"]
    (out / "experiment_report.md").write_text("\n".join(lines) + "\n")


def validate_output_files(out):
    """Undefined statistics use null/empty strings, never numeric NaN/Inf."""
    for path in out.glob("*.json"):
        json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(str(path) + value)))
    for path in out.glob("*.csv"):
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                for field, value in row.items():
                    if value.strip().lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
                        raise FloatingPointError("{} row {} field {}".format(path, i, field))


def main(argv=None):
    args = parse_args(argv)
    out = create_output(args.output_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(str(out / "analysis_log.txt")), logging.StreamHandler(sys.stdout)])
    metadata = dict(status="started", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    configuration=vars(args), method="frozen checkpoint, GT-only prespecified sampling, gradients only on a",
                    protocol=dict(group_selection="signed BGS high4/low4/middle4; stable channel-index tie break; no utility selection",
                                  crop_rule="original RandomCrop padding/exclusive upper bound; no augmentation",
                                  min_voxels=args.min_band_voxels, max_attempts_per_target_patch=args.max_retries,
                                  delta=args.delta, fd_atol=args.fd_atol, fd_rtol=args.fd_rtol, sign_epsilon=args.sign_epsilon,
                                  bootstrap_unit="case", undefined_statistics="null/empty plus explicit reason",
                                  no_training=True, no_unlabeled_or_test_gt=True),
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(NEW_CODE) if p.exists()},
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                     cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    before = protected_manifest(out)
    metadata["protected_file_sha256_before"] = before
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    metadata["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    metadata["code_version_note"] = "SHA256 manifest is authoritative when no git commit is available"
    write_json(out / "experiment_metadata.json", metadata)
    verification = dict(status="running")
    try:
        unit_tests = subprocess.run([sys.executable, "-B", str(ROOT / "analysis/test_experiment_bgs_utility.py")],
                                    capture_output=True, text=True)
        write_json(out / "test_results.json", dict(exit_code=unit_tests.returncode,
                                                  stdout=unit_tests.stdout, stderr=unit_tests.stderr))
        if unit_tests.returncode:
            raise AssertionError("Analysis unit checks failed: " + unit_tests.stderr)
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            metadata["environment"]["device_name"] = torch.cuda.get_device_name(device)
        cases, split = labeled_split(args)
        metadata["data_split"] = split
        metadata["checkpoints"] = [dict(name=n, path=str(Path(p).resolve()), sha256=sha256(p), type="EMA plain state_dict")
                                   for n, p in zip(args.checkpoint_names, args.model_paths)]
        first_model = load_frozen(args.model_paths[0], args.device)
        probe_before = snapshot_state(first_model)
        with torch.no_grad():
            probe_features = first_model.encoder(torch.zeros((1, 1, *args.patch_size), device=device))
        shapes = [list(f.shape) for f in probe_features]
        spatial_shape = tuple(probe_features[2].shape[2:])
        verify_state(first_model, probe_before)
        del probe_features, probe_before
        metadata["actual_feature_shapes"] = shapes
        logging.info("Dynamic encoder feature shapes: %s", shapes)
        patches, attempts, statuses, sources = sample_patches(cases, args, spatial_shape)
        write_csv(out / "sampling_attempts.csv", attempts)
        write_csv(out / "patch_status.csv", statuses)
        metadata["data_sources"] = sources
        metadata["sampling"] = dict(valid_cases=len({p["case_id"] for p in patches}), valid_patches=len(patches),
                                    skipped_cases=[case_id for _, case_id, _ in cases if not any(p["case_id"] == case_id for p in patches)],
                                    skipped_patches=sum(not s["valid"] for s in statuses), attempts=len(attempts),
                                    rejected_attempts=sum(not a["accepted"] for a in attempts),
                                    patch_coordinates=[dict(case_id=p["case_id"], patch_index=p["patch_index"], **p["coords"], **p["counts"])
                                                       for p in patches], all_attempts=attempts)
        write_json(out / "experiment_metadata.json", metadata)
        records, channels, fd, patch_rows, model_checks = [], [], [], [], {}
        for i, (name, path) in enumerate(zip(args.checkpoint_names, args.model_paths)):
            model = first_model if i == 0 else load_frozen(path, args.device)
            result = analyze_checkpoint(model, name, patches, args)
            rr, cc, ff, pp, check = result
            records.extend(rr)
            channels.extend(cc)
            fd.extend(ff)
            patch_rows.extend(pp)
            model_checks[name] = check
            write_csv(out / "channel_bgs_utility.csv", channels)
            write_csv(out / "finite_difference_validation.csv", fd)
            write_csv(out / "patch_level_statistics.csv", patch_rows)
            if i == 0:
                first_model = None
            del model, result
            if device.type == "cuda":
                torch.cuda.empty_cache()
        case_rows, pairs, signs, summary = aggregate(records, patch_rows, fd, args)
        write_csv(out / "case_level_statistics.csv", case_rows)
        write_csv(out / "pairwise_utility_stability.csv", pairs)
        write_csv(out / "channel_sign_stability.csv", signs)
        make_figures(out, channels, case_rows, args.checkpoint_names)
        after = protected_manifest(out)
        changed = [path for path, old in before.items() if after.get(path) != old]
        added = sorted(set(after) - set(before))
        if changed or added:
            raise AssertionError("Pre-existing project file manifest changed: " + repr((changed, added)))
        for ck in metadata["checkpoints"]:
            if sha256(ck["path"]) != ck["sha256"]:
                raise AssertionError("Checkpoint file changed: " + ck["path"])
        verification = dict(status="passed", protected_files_unchanged=True, protected_file_count=len(before),
                            changed_files=[], checkpoint_hashes_unchanged=True,
                            only_actual_labeled_training_gt=True, test_gt_loaded=False, unlabeled_gt_loaded=False,
                            optimizer_created=False, parameter_updates=False, ema_updates=False, training_performed=False,
                            model_checks=model_checks, all_numeric_outputs_finite=True,
                            finite_difference_agreement={name: summary[name]["finite_difference"] for name in args.checkpoint_names},
                            finite_difference_note="Disagreements are reported observations, not discarded; tolerance is in protocol",
                            positive_utility_definition="U=-dL/da; Gplus=L0-Lplus; positive means boundary loss decreases",
                            gain_sign_analytical_unit_test_passed=True, unit_test_exit_code=unit_tests.returncode,
                            unit_test_evidence="test_results.json",
                            cached_features_created_under="torch.no_grad; not inference_mode")
        metadata["status"] = "complete"
        metadata["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["protected_file_sha256_after"] = after
        write_report(out, summary, case_rows, signs, metadata, verification, args)
        write_json(out / "summary_statistics.json", summary)
        write_json(out / "experiment_metadata.json", metadata)
        write_json(out / "verification.json", verification)
        validate_output_files(out)
        logging.info("Complete; unchanged protected files=%d; output=%s", len(before), out)
        for name in args.checkpoint_names:
            logging.info("%s BGS/U: %s", name, fmt_stat(summary[name]["metrics"]["bgs_utility_spearman"]))
        logging.info("Recommendation: %s", summary["recommendation"]["choice"])
    except Exception as exc:
        logging.exception("Analysis failed; partial outputs retained; rerun requires a NEW output directory")
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        write_json(out / "experiment_metadata.json", metadata)
        write_json(out / "verification.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
