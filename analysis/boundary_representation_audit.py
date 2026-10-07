#!/usr/bin/env python3
"""Read-only boundary audit of an existing MT VNet checkpoint.

Only the first patients_to_slices(dataset, labeled_num) training-list entries
are opened. No training or prediction path is called. See README.md for the
sampling, bootstrap, random-null and centered-PCA conventions.
"""

import argparse
import ast
import csv
import hashlib
import inspect
import itertools
import json
import logging
from pathlib import Path
import random
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataloader.dataset import build_Dataset
from model.vnet import VNet
from utils.transforms import RandomCrop, ToTensor
from utils.utils import patients_to_slices

LAYERS = ("x2", "x3", "x4")
EPS = 1e-6
PROTECTED = ("train.py", "trainer.py", "model/vnet.py", "dataloader/dataset.py",
             "utils/transforms.py", "utils/utils.py", "prediction.py")
AUGMENTATIONS = ("identity", "flip_axis0", "flip_axis1", "rot90_axes01")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", required=True)
    p.add_argument("--dataset", required=True, choices=["LA", "Pancreas", "BraTS2019", "Lung",
                   "/2018LA_Seg_Training Set", "/Pancreas", "/BraTS2019", "/Lung"])
    p.add_argument("--labeled_num", type=int, required=True, help="Baseline labeled percentage, not case count")
    p.add_argument("--model_path", required=True)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--device", default="cpu")
    p.add_argument("--patches_per_case", type=int, default=3)
    p.add_argument("--topk_ratios", type=float, nargs="+", default=[0.125, 0.25])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--min_band_voxels", type=int, default=8)
    p.add_argument("--min_foreground_voxels", type=int, default=32)
    p.add_argument("--max_retries", type=int, default=50, help="Maximum candidate crops per case")
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--random_samples", type=int, default=5000, help="Independent random channel-subset pairs")
    p.add_argument("--shuffle_repeats", type=int, default=200, help="Channel permutations per real case pair")
    p.add_argument("--cpu_threads", type=int, default=4)
    args = p.parse_args(argv)
    for name in ("patches_per_case", "min_band_voxels", "min_foreground_voxels", "max_retries",
                 "bootstrap_samples", "random_samples", "shuffle_repeats", "cpu_threads"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.max_retries < args.patches_per_case:
        p.error("--max_retries must be >= --patches_per_case")
    if any(s < 16 or s % 16 for s in args.patch_size):
        p.error("VNet patch dimensions must be positive multiples of 16")
    if any(not 0 < r <= 1 for r in args.topk_ratios):
        p.error("--topk_ratios must be in (0, 1]")
    args.topk_ratios = sorted(set(args.topk_ratios))
    if args.in_channels != 1 or args.num_classes != 2:
        p.error("This baseline audit uses single-channel images and binary foreground labels")
    return args


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_analysis_source():
    tree = ast.parse(Path(__file__).read_text())
    forbidden_calls = {"backward", "train", "update_ema_variables"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in forbidden_calls:
                raise RuntimeError(f"Forbidden analysis call: {name}")
        if isinstance(node, ast.Attribute) and node.attr in {"optim", "optimizer", "scheduler"}:
            raise RuntimeError(f"Forbidden analysis attribute: {node.attr}")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def boundary_bands(label, spatial_shape):
    mask = TF.interpolate(label.float(), size=tuple(spatial_shape), mode="nearest") > 0.5
    mask_f = mask.float()
    dilated = TF.max_pool3d(mask_f, 3, 1, 1)
    eroded = 1 - TF.max_pool3d(1 - mask_f, 3, 1, 1)
    return (mask_f - eroded) > 0.5, (dilated - mask_f) > 0.5


def transition_vector(feature, inside, outside):
    """Population variance (correction=0), independently for each channel."""
    if feature.shape[0] != 1:
        raise ValueError("Patch extraction uses batch size 1")
    v_in = feature[0, :, inside[0, 0]].double()
    v_out = feature[0, :, outside[0, 0]].double()
    mu_in, mu_out = v_in.mean(dim=1), v_out.mean(dim=1)
    var_in, var_out = v_in.var(dim=1, unbiased=False), v_out.var(dim=1, unbiased=False)
    d = (mu_in - mu_out) / torch.sqrt((var_in + var_out) / 2 + EPS)
    return d.cpu().numpy()


def normalize(d):
    return d / (np.linalg.norm(d) + EPS)


def topk(d, ratio):
    k = max(1, min(len(d), round(len(d) * ratio)))
    # Stable sorting makes exact ties reproducible; tie frequency is logged.
    return set(np.argsort(-np.abs(d), kind="stable")[:k].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b)


def spearman(a, b):
    a, b = rankdata(np.abs(a)), rankdata(np.abs(b))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def signed_cosine(a, b):
    if np.linalg.norm(a) <= EPS or np.linalg.norm(b) <= EPS:
        return float("nan")
    return float(np.clip(np.dot(a, b), -1, 1))


def geometric_view(image, label, name):
    if name == "identity":
        return image, label
    if name.startswith("flip_axis"):
        dim = 2 + int(name[-1])
        return torch.flip(image, [dim]), torch.flip(label, [dim])
    if name == "rot90_axes01":
        return torch.rot90(image, 1, [2, 3]), torch.rot90(label, 1, [2, 3])
    raise ValueError(name)


@torch.no_grad()
def extract(model, image, label, args, logger, shapes, events, case_id, patch_id, view):
    if torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise RuntimeError("Analysis requires every module in eval and gradients disabled")
    features = model.encoder(image)
    if len(features) != 5:
        raise RuntimeError("Expected encoder outputs [x1, x2, x3, x4, x5]")
    vectors = {}
    for layer, feature in zip(LAYERS, features[1:4]):
        shape = tuple(feature.shape)
        if shape not in shapes[layer]:
            shapes[layer].append(shape)
            logger.info("FEATURE_SHAPE layer=%s shape=%s view=%s", layer, shape, view)
        inside, outside = boundary_bands(label, feature.shape[2:])
        n_in, n_out = int(inside.sum()), int(outside.sum())
        event = dict(case_id=case_id, patch_id=patch_id, augmentation=view, layer=layer,
                     feature_shape=str(shape), band_in_voxels=n_in, band_out_voxels=n_out)
        if min(n_in, n_out) < args.min_band_voxels:
            event.update(status="skipped", reason="insufficient_band_voxels")
            logger.info("SKIP layer=%s case=%s patch=%s view=%s band_in=%d band_out=%d",
                        layer, case_id, patch_id, view, n_in, n_out)
        else:
            d = transition_vector(feature, inside, outside)
            bad = np.flatnonzero(~np.isfinite(d))
            if bad.size:
                for channel in bad:
                    logger.error("NONFINITE_D layer=%s case=%s patch=%s view=%s channel=%d value=%s",
                                 layer, case_id, patch_id, view, channel, d[channel])
                event.update(status="error", reason=f"nonfinite_d_channels={bad.tolist()}")
                events.append(event)
                raise FloatingPointError("Nonfinite transition vector; see analysis_log.txt")
            vectors[layer] = d
            event.update(status="valid", reason="")
        events.append(event)
    return vectors


def distribution_summary(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    result = dict(n_observations=len(values), n_defined=len(valid), n_undefined=len(values) - len(valid),
                  mean=float("nan"), std=float("nan"), median=float("nan"),
                  ci95_low=float("nan"), ci95_high=float("nan"))
    if len(valid):
        result.update(mean=float(valid.mean()), std=float(valid.std()), median=float(np.median(valid)))
    return result


def case_pair_ci(matrix, weights):
    """Case bootstrap: weight off-diagonal original-case pairs, never self pairs."""
    upper = np.triu_indices(len(matrix), 1)
    vals = matrix[upper]
    valid = np.isfinite(vals)
    if not np.any(valid):
        return float("nan"), float("nan")
    pair_weights = weights[:, upper[0][valid]] * weights[:, upper[1][valid]]
    denom = pair_weights.sum(axis=1)
    estimates = (pair_weights @ vals[valid])[denom > 0] / denom[denom > 0]
    if not len(estimates):
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def cluster_ci(values, groups, weights):
    sums, counts = np.zeros(weights.shape[1]), np.zeros(weights.shape[1])
    for value, group in zip(values, groups):
        if np.isfinite(value):
            sums[group] += value
            counts[group] += 1
    denom = weights @ counts
    estimates = (weights @ sums)[denom > 0] / denom[denom > 0]
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5])) if len(estimates) else (np.nan, np.nan)


def iid_ci(values, samples, rng):
    # Bounded memory for thousands of random subset pairs.
    estimates = []
    for _ in range(samples):
        estimates.append(np.mean(rng.choice(values, len(values), replace=True)))
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def summary_row(layer, metric, scope, source, values, ci, n_cases, ratio="", k="", method=""):
    row = dict(layer=layer, metric=metric, scope=scope, source=source, topk_ratio=ratio, k=k,
               n_cases=n_cases, ci_method=method, **distribution_summary(values))
    row["ci95_low"], row["ci95_high"] = ci
    return row


def matrix_for(n):
    return np.full((n, n), np.nan)


def compare_cases(case_vectors, args, rng, logger):
    summaries, jaccard_rows, cosine_rows, ranking_rows, null_rows, pca_rows = [], [], [], [], [], []
    for layer in LAYERS:
        cases = sorted(case_vectors[layer])
        n = len(cases)
        if not n:
            logger.warning("NO_VALID_CASES layer=%s", layer)
            continue
        raw = np.stack([case_vectors[layer][c] for c in cases])
        norms = np.stack([normalize(d) for d in raw])
        c = raw.shape[1]
        weights = rng.multinomial(n, np.full(n, 1 / n), size=args.bootstrap_samples)
        pairs = list(itertools.combinations(range(n), 2))
        cosine_m, shuffled_m, rank_m = matrix_for(n), matrix_for(n), matrix_for(n)
        shuffled_values = []
        for i, j in pairs:
            cosine = signed_cosine(norms[i], norms[j])
            rank = spearman(raw[i], raw[j])
            cosine_m[i, j], rank_m[i, j] = cosine, rank
            if not np.isfinite(cosine) or not np.isfinite(rank):
                logger.warning("UNDEFINED_PAIR layer=%s cases=%s,%s cosine=%s spearman=%s",
                               layer, cases[i], cases[j], cosine, rank)
            shuffled = [signed_cosine(norms[i], rng.permutation(norms[j]))
                        for _ in range(args.shuffle_repeats)]
            defined = [v for v in shuffled if np.isfinite(v)]
            shuffled_m[i, j] = np.mean(defined) if defined else np.nan
            shuffled_values.extend(shuffled)
            cosine_rows.append(dict(layer=layer, case_i=cases[i], case_j=cases[j], cosine=cosine,
                               shuffled_mean=shuffled_m[i, j], shuffled_std=np.std(defined) if defined else np.nan))
            ranking_rows.append(dict(layer=layer, case_i=cases[i], case_j=cases[j], spearman_abs_d=rank))
            for repeat, value in enumerate(shuffled):
                null_rows.append(dict(layer=layer, metric="signed_cosine", topk_ratio="", k="",
                                      case_i=cases[i], case_j=cases[j], repeat=repeat, value=value))
        upper = np.triu_indices(n, 1)
        method = "case_bootstrap_offdiagonal_mean"
        summaries.append(summary_row(layer, "signed_cosine", "cross_case", "real", cosine_m[upper],
                         case_pair_ci(cosine_m, weights), n, method=method))
        summaries.append(summary_row(layer, "signed_cosine", "cross_case", "channel_shuffle", shuffled_values,
                         case_pair_ci(shuffled_m, weights), n, method=method))
        summaries.append(summary_row(layer, "signed_cosine", "cross_case", "real_minus_shuffle",
                         (cosine_m - shuffled_m)[upper], case_pair_ci(cosine_m - shuffled_m, weights), n, method=method))
        summaries.append(summary_row(layer, "spearman_abs_d", "cross_case", "real", rank_m[upper],
                         case_pair_ci(rank_m, weights), n, method=method))
        for ratio in args.topk_ratios:
            sets = [topk(d, ratio) for d in raw]
            k = len(sets[0])
            for case_id, d in zip(cases, raw):
                ordered = np.sort(np.abs(d))[::-1]
                if k < c and ordered[k - 1] == ordered[k]:
                    logger.warning("TOPK_CUTOFF_TIE layer=%s case=%s ratio=%s k=%d", layer, case_id, ratio, k)
            jm = matrix_for(n)
            for i, j in pairs:
                jm[i, j] = jaccard(sets[i], sets[j])
                jaccard_rows.append(dict(layer=layer, topk_ratio=ratio, channels=c, k=k,
                                        case_i=cases[i], case_j=cases[j], jaccard=jm[i, j]))
            random_j = np.array([jaccard(set(rng.choice(c, k, replace=False)),
                                        set(rng.choice(c, k, replace=False))) for _ in range(args.random_samples)])
            for repeat, value in enumerate(random_j):
                null_rows.append(dict(layer=layer, metric="topk_jaccard", topk_ratio=ratio, k=k,
                                      case_i="", case_j="", repeat=repeat, value=value))
            summaries.append(summary_row(layer, "topk_jaccard", "cross_case", "real", jm[upper],
                             case_pair_ci(jm, weights), n, ratio, k, method))
            summaries.append(summary_row(layer, "topk_jaccard", "cross_case", "random_subsets", random_j,
                             iid_ci(random_j, args.bootstrap_samples, rng), n, ratio, k, "iid_random_draw_bootstrap_mean"))
            delta = jm - random_j.mean()
            summaries.append(summary_row(layer, "topk_jaccard", "cross_case", "real_minus_random",
                             delta[upper], case_pair_ci(delta, weights), n, ratio, k, "case_bootstrap_fixed_MC_null_mean"))
        # Centered PCA measures variability about the common direction. Also
        # report uncentered singular energy to retain that common direction.
        for mode, matrix in (("centered_pca", norms - norms.mean(axis=0)), ("uncentered_svd", norms)):
            singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
            energy = singular ** 2
            total = float(energy.sum())
            limit = min(c, n - 1 if mode == "centered_pca" else n)
            ratios = energy[:limit] / total if total > 1e-20 else np.full(limit, np.nan)
            cumulative = np.cumsum(ratios)
            for component, (r, cum) in enumerate(zip(ratios, cumulative), 1):
                pca_rows.append(dict(layer=layer, mode=mode, n_cases=n, channels=c, component=component,
                                     explained_variance_ratio=float(r), cumulative_explained_variance=float(cum)))
            for requested in (1, 2, 4, 8):
                used = min(requested, limit)
                logger.info("SUBSPACE layer=%s mode=%s requested=%d used=%d cumulative=%s", layer, mode,
                            requested, used, cumulative[used - 1] if used else "undefined")
            if total <= 1e-20:
                logger.warning("DEGENERATE_SUBSPACE layer=%s mode=%s", layer, mode)
    return summaries, jaccard_rows, cosine_rows, ranking_rows, null_rows, pca_rows


def compare_augmentations(records, args, rng, logger):
    rows, summaries = [], []
    for record in records:
        case_id, patch_id, layer, views = record
        for a, b in itertools.combinations(AUGMENTATIONS, 2):
            if a not in views or b not in views:
                logger.info("SKIP_AUG_PAIR layer=%s case=%s patch=%s views=%s,%s", layer, case_id, patch_id, a, b)
                continue
            cosine = signed_cosine(normalize(views[a]), normalize(views[b]))
            rank = spearman(views[a], views[b])
            if not np.isfinite(cosine) or not np.isfinite(rank):
                logger.warning("UNDEFINED_AUG_PAIR layer=%s case=%s patch=%s views=%s,%s", layer, case_id, patch_id, a, b)
            for ratio in args.topk_ratios:
                ta, tb = topk(views[a], ratio), topk(views[b], ratio)
                rows.append(dict(layer=layer, case_id=case_id, patch_id=patch_id, augmentation_i=a,
                                 augmentation_j=b, topk_ratio=ratio, k=len(ta),
                                 jaccard=jaccard(ta, tb), spearman_abs_d=rank, signed_cosine=cosine))
    for layer in LAYERS:
        cases = sorted({r["case_id"] for r in rows if r["layer"] == layer})
        if not cases:
            continue
        weights = rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)), size=args.bootstrap_samples)
        for ratio in args.topk_ratios:
            selected = [r for r in rows if r["layer"] == layer and r["topk_ratio"] == ratio]
            groups = [cases.index(r["case_id"]) for r in selected]
            for metric, field in (("topk_jaccard", "jaccard"), ("spearman_abs_d", "spearman_abs_d"),
                                  ("signed_cosine", "signed_cosine")):
                if metric != "topk_jaccard" and ratio != args.topk_ratios[0]:
                    continue
                values = [r[field] for r in selected]
                summaries.append(summary_row(layer, metric, "intra_sample_augmentation", "real", values,
                                 cluster_ci(values, groups, weights), len(cases),
                                 ratio if metric == "topk_jaccard" else "",
                                 selected[0]["k"] if metric == "topk_jaccard" else "", "case_cluster_bootstrap_mean"))
    return rows, summaries


def save_vectors(out, case_vectors, patches, augmentation_records):
    payload = {"eps": np.array(EPS), "layer_names": np.array(LAYERS)}
    for layer in LAYERS:
        cases = sorted(case_vectors[layer])
        channels = len(next(iter(case_vectors[layer].values()))) if cases else 0
        d = np.stack([case_vectors[layer][c] for c in cases]) if cases else np.empty((0, channels))
        payload.update({f"{layer}_case_ids": np.array(cases, dtype=str), f"{layer}_case_d": d,
                        f"{layer}_case_abs_d": np.abs(d),
                        f"{layer}_case_d_norm": np.stack([normalize(x) for x in d]) if cases else d.copy()})
        selected = [p for p in patches if p["layer"] == layer]
        payload[f"{layer}_patch_case_ids"] = np.array([p["case_id"] for p in selected], dtype=str)
        payload[f"{layer}_patch_ids"] = np.array([p["patch_id"] for p in selected], dtype=int)
        payload[f"{layer}_patch_d"] = np.stack([p["d"] for p in selected]) if selected else np.empty((0, channels))
        payload[f"{layer}_patch_abs_d"] = np.abs(payload[f"{layer}_patch_d"])
        payload[f"{layer}_case_patch_counts"] = np.array([sum(p["case_id"] == c for p in selected) for c in cases])
        aug = [(case_id, patch_id, name, d) for case_id, patch_id, l, views in augmentation_records if l == layer
               for name, d in views.items()]
        payload[f"{layer}_aug_case_ids"] = np.array([a[0] for a in aug], dtype=str)
        payload[f"{layer}_aug_patch_ids"] = np.array([a[1] for a in aug], dtype=int)
        payload[f"{layer}_aug_names"] = np.array([a[2] for a in aug], dtype=str)
        payload[f"{layer}_aug_d"] = np.stack([a[3] for a in aug]) if aug else np.empty((0, channels))
    np.savez_compressed(out / "case_transition_vectors.npz", **payload)


def plot_results(out, summaries, pca_rows, ratios):
    def select(layer, metric, source, ratio=None):
        return next((r for r in summaries if r["layer"] == layer and r["scope"] == "cross_case"
                     and r["metric"] == metric and r["source"] == source
                     and (ratio is None or r["topk_ratio"] == ratio)), None)

    def bars(ax, metric, ratio, sources):
        width = 0.8 / len(sources)
        for si, source in enumerate(sources):
            rows = [select(l, metric, source, ratio) for l in LAYERS]
            means = np.array([r["mean"] if r else np.nan for r in rows])
            x = np.arange(3) - 0.4 + width / 2 + si * width
            ax.bar(x, means, width, label=source)
            # Percentile CIs need not contain the point estimate; draw bounds directly.
            for xx, r in zip(x, rows):
                if r and np.isfinite(r["ci95_low"]) and np.isfinite(r["ci95_high"]):
                    ax.vlines(xx, r["ci95_low"], r["ci95_high"], color="black", linewidth=1)
        ax.set_xticks(np.arange(3), LAYERS)
        ax.set_ylabel("Mean similarity (95% CI of mean)")
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, len(ratios), figsize=(5 * len(ratios), 4), squeeze=False)
    for ax, ratio in zip(axes[0], ratios):
        bars(ax, "topk_jaccard", ratio, ["real"])
        ax.set_title(f"Cross-case Top-K Jaccard: ratio={ratio:g}")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "mean_topk_jaccard.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars(ax, "signed_cosine", None, ["real"])
    ax.set_title("Cross-case signed boundary transition cosine")
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(out / "mean_btv_cosine.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(ratios) + 1, figsize=(5 * (len(ratios) + 1), 4))
    for ax, ratio in zip(axes[:-1], ratios):
        bars(ax, "topk_jaccard", ratio, ["real", "random_subsets"])
        ax.set_title(f"Top-K Jaccard: ratio={ratio:g}")
        ax.set_ylim(0, 1)
    bars(axes[-1], "signed_cosine", None, ["real", "channel_shuffle"])
    axes[-1].set_title("Signed BTV cosine")
    axes[-1].set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(out / "real_vs_random.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("centered_pca", "uncentered_svd")):
        for layer in LAYERS:
            rows = [r for r in pca_rows if r["layer"] == layer and r["mode"] == mode]
            if rows:
                ax.plot([r["component"] for r in rows], [r["cumulative_explained_variance"] for r in rows],
                        marker="o", label=layer)
        ax.set(xlabel="Components", ylabel="Cumulative explained variance / energy", ylim=(0, 1.02), title=mode)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pca_cumulative_explained_variance.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def audit(args, logger, metadata, events, case_status):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; explicitly use --device cpu")
    model = VNet(n_channels=args.in_channels, n_classes=args.num_classes)
    # SSL uses PyTorch 1.10, which predates weights_only. Newer releases use
    # the restricted loader; older releases load the user's baseline weights.
    load_kwargs = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    checkpoint = torch.load(args.model_path, map_location="cpu", **load_kwargs)
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    state_before = model_hash(model)
    metadata["model_state_sha256_before"] = state_before
    canonical = {"/2018LA_Seg_Training Set": "LA", "/Pancreas": "Pancreas",
                 "/BraTS2019": "BraTS2019", "/Lung": "Lung"}.get(args.dataset, args.dataset)
    data_dir = args.data_path if not args.dataset.startswith("/") else args.data_path + args.dataset
    dataset = build_Dataset(args, data_dir=data_dir, split="train_" + canonical, transform=None)
    labeled_count = patients_to_slices(args.dataset, args.labeled_num)
    if labeled_count > len(dataset.sample_list):
        raise ValueError("Labeled split exceeds the training list")
    labeled_paths = dataset.sample_list[:labeled_count]
    if len(set(labeled_paths)) != labeled_count:
        raise ValueError("Duplicate case paths in the labeled split")
    metadata.update(labeled_cases_requested=labeled_count, labeled_paths=labeled_paths,
                    total_training_entries=len(dataset.sample_list),
                    split_rule="first patients_to_slices(dataset, labeled_num) entries in baseline train.list")
    logger.info("SPLIT labeled=%d total_training_entries=%d; only labeled HDF5 files will be opened",
                labeled_count, len(dataset.sample_list))
    crop, tensorize = RandomCrop(args.patch_size), ToTensor()
    shapes = {layer: [] for layer in LAYERS}
    patches, augmentation_records = [], []
    case_vectors = {layer: {} for layer in LAYERS}
    for case_index, path in enumerate(labeled_paths):
        case_id = dataset.image_list[case_index]
        per_layer = {layer: [] for layer in LAYERS}
        attempts, accepted = 0, 0
        logger.info("CASE_START index=%d/%d case=%s path=%s", case_index + 1, labeled_count, case_id, path)
        try:
            with h5py.File(path, "r") as f:
                image_np, label_np = f["image"][:], f["label"][:]
        except (OSError, KeyError) as e:
            logger.error("SKIP_CASE case=%s reason=%s", case_id, e)
            for layer in LAYERS:
                case_status.append(dict(case_id=case_id, layer=layer, attempts=0, valid_patches=0,
                                        target_patches=args.patches_per_case, status="skipped", reason=str(e)))
            continue
        if image_np.ndim != 3 or image_np.shape != label_np.shape:
            raise ValueError(f"Unexpected image/label shapes in {path}")
        if not np.isfinite(image_np).all() or not np.isfinite(label_np).all():
            logger.error("NONFINITE_INPUT case=%s", case_id)
            raise FloatingPointError(f"Nonfinite input in case {case_id}")
        labels = np.unique(label_np)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Expected binary label; case={case_id} values={labels}")
        while attempts < args.max_retries and any(len(v) < args.patches_per_case for v in per_layer.values()):
            attempts += 1
            patch = tensorize(crop({"image": image_np, "label": label_np}))
            foreground = int(patch["label"].sum())
            if foreground < args.min_foreground_voxels:
                events.append(dict(case_id=case_id, patch_id=attempts, augmentation="identity", layer="input",
                                   status="skipped", reason="insufficient_foreground", foreground_voxels=foreground))
                logger.info("SKIP_PATCH case=%s patch=%d foreground=%d", case_id, attempts, foreground)
                continue
            image = patch["image"].unsqueeze(0).to(device)
            label = patch["label"].unsqueeze(0).unsqueeze(0).to(device)
            identity = extract(model, image, label, args, logger, shapes, events, case_id, attempts, "identity")
            selected = [l for l in LAYERS if l in identity and len(per_layer[l]) < args.patches_per_case]
            if not selected:
                logger.info("SKIP_PATCH case=%s patch=%d reason=no_valid_layer_needing_patches", case_id, attempts)
                continue
            accepted += 1
            views = {"identity": identity}
            for name in AUGMENTATIONS[1:]:
                aug_image, aug_label = geometric_view(image, label, name)
                views[name] = extract(model, aug_image, aug_label, args, logger, shapes, events, case_id, attempts, name)
            for layer in selected:
                per_layer[layer].append(identity[layer])
                patches.append(dict(case_id=case_id, patch_id=attempts, layer=layer, d=identity[layer]))
                augmentation_records.append((case_id, attempts, layer,
                                             {name: vectors[layer] for name, vectors in views.items() if layer in vectors}))
            logger.info("PATCH_ACCEPTED case=%s patch=%d layer_counts=%s", case_id, attempts,
                        {l: len(v) for l, v in per_layer.items()})
        for layer, vectors in per_layer.items():
            if vectors:
                case_d = np.mean(vectors, axis=0)
                bad = np.flatnonzero(~np.isfinite(case_d))
                if len(bad):
                    logger.error("NONFINITE_CASE_D layer=%s case=%s channels=%s", layer, case_id, bad.tolist())
                    raise FloatingPointError("Nonfinite case mean")
                if np.linalg.norm(case_d) <= EPS:
                    logger.warning("ZERO_CASE_VECTOR layer=%s case=%s; cosine is undefined", layer, case_id)
                case_vectors[layer][case_id] = case_d
            status = "complete" if len(vectors) == args.patches_per_case else "partial" if vectors else "skipped"
            reason = "" if status == "complete" else "max_retries_exhausted"
            case_status.append(dict(case_id=case_id, layer=layer, attempts=attempts, valid_patches=len(vectors),
                                    target_patches=args.patches_per_case, status=status, reason=reason))
            logger.info("CASE_DONE layer=%s case=%s status=%s valid_patches=%d attempts=%d",
                        layer, case_id, status, len(vectors), attempts)
    rng = np.random.default_rng(args.seed + 10000)
    summaries, jr, cr, rr, nr, pr = compare_cases(case_vectors, args, rng, logger)
    ar, aug_summaries = compare_augmentations(augmentation_records, args, rng, logger)
    summaries += aug_summaries
    out = Path(args.output_dir)
    save_vectors(out, case_vectors, patches, augmentation_records)
    write_csv(out / "summary.csv", summaries, ["layer", "scope", "metric", "source", "topk_ratio", "k", "n_cases",
              "n_observations", "n_defined", "n_undefined", "mean", "std", "median", "ci95_low", "ci95_high", "ci_method"])
    write_csv(out / "pairwise_channel_jaccard.csv", jr, ["layer", "topk_ratio", "channels", "k", "case_i", "case_j", "jaccard"])
    write_csv(out / "pairwise_transition_cosine.csv", cr, ["layer", "case_i", "case_j", "cosine", "shuffled_mean", "shuffled_std"])
    write_csv(out / "pairwise_channel_spearman.csv", rr, ["layer", "case_i", "case_j", "spearman_abs_d"])
    write_csv(out / "augmentation_stability.csv", ar, ["layer", "case_id", "patch_id", "augmentation_i", "augmentation_j",
              "topk_ratio", "k", "jaccard", "spearman_abs_d", "signed_cosine"])
    write_csv(out / "random_baselines.csv", nr, ["layer", "metric", "topk_ratio", "k", "case_i", "case_j", "repeat", "value"])
    write_csv(out / "pca_explained_variance.csv", pr, ["layer", "mode", "n_cases", "channels", "component",
              "explained_variance_ratio", "cumulative_explained_variance"])
    plot_results(out, summaries, pr, args.topk_ratios)
    metadata["feature_shapes"] = shapes
    metadata["layer_counts"] = {}
    for layer in LAYERS:
        statuses = [s for s in case_status if s["layer"] == layer]
        counts = dict(valid_cases=len(case_vectors[layer]), skipped_cases=sum(s["status"] == "skipped" for s in statuses),
                      partial_cases=sum(s["status"] == "partial" for s in statuses),
                      valid_identity_patches=sum(s["valid_patches"] for s in statuses))
        metadata["layer_counts"][layer] = counts
        logger.info("LAYER_TOTAL layer=%s %s", layer, counts)
    valid_union = set().union(*(set(case_vectors[l]) for l in LAYERS))
    metadata.update(valid_cases_any_layer=len(valid_union), skipped_cases_all_layers=labeled_count - len(valid_union),
                    nonfinite_d_count=0, model_state_sha256_after=model_hash(model))
    if state_before != metadata["model_state_sha256_after"]:
        raise RuntimeError("Model parameters or buffers changed during analysis")
    logger.info("MODEL_STATE_UNCHANGED eval=%s no_grad=%s no_parameter_grads=%s",
                all(not m.training for m in model.modules()), not torch.is_grad_enabled(),
                all(p.grad is None for p in model.parameters()))
    if not valid_union:
        raise RuntimeError("No valid labeled cases; diagnostics were saved but no conclusions can be drawn")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty --output_dir to preserve earlier audits")
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("boundary_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.FileHandler(out / "analysis_log.txt"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    protected = [ROOT / f for f in PROTECTED] + [Path(args.model_path).resolve()]
    before = {str(p): file_hash(p) for p in protected}
    metadata = dict(arguments=vars(args), protected_sha256_before=before,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    script_sha256=file_hash(__file__), status="running")
    events, case_status = [], []
    try:
        check_analysis_source()
        logger.info("ARGUMENTS %s", json.dumps(vars(args), sort_keys=True))
        logger.info("CHECKS source contains no training/gradient/update calls; protected SHA256 captured")
        audit(args, logger, metadata, events, case_status)
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "failed"
        logger.exception("AUDIT_FAILED")
        raise
    finally:
        after = {str(p): file_hash(p) for p in protected}
        metadata["protected_sha256_after"] = after
        metadata["protected_files_unchanged"] = before == after
        write_csv(out / "patch_status.csv", events, ["case_id", "patch_id", "augmentation", "layer", "status", "reason",
                  "foreground_voxels", "feature_shape", "band_in_voxels", "band_out_voxels"])
        write_csv(out / "case_status.csv", case_status, ["case_id", "layer", "attempts", "valid_patches",
                  "target_patches", "status", "reason"])
        (out / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("PROTECTED_FILES_UNCHANGED=%s (train, trainer, VNet, data, transforms, utils, prediction, checkpoint)", before == after)
        if before != after:
            for path in before:
                if before[path] != after[path]:
                    logger.error("PROTECTED_FILE_CHANGED path=%s", path)
            raise RuntimeError("Protected file integrity check failed")
    logger.info("AUDIT_COMPLETE output=%s", out.resolve())


if __name__ == "__main__":
    main()
