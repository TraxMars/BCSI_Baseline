#!/usr/bin/env python3
"""Experiment 0.5: read-only x3 boundary specificity audit of an existing VNet.

Default matching_policy follows the four-band minimum in the supplied protocol.
all_six_min is an explicit alternative for cores too small to match that count.
Experiment 0 and every baseline source/checkpoint are protected by SHA256.
"""
import argparse
import ast
from collections import Counter
import inspect
import itertools
import json
import logging
from pathlib import Path
import random
import sys

import h5py
import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as TF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import boundary_representation_audit as base

EPS = 1e-6
REGIONS = ("near_in", "near_out", "far_in", "far_out", "foreground_core", "background_core")
KINDS = ("near", "far", "global", "bgs")
COMPARISONS = (("near", "far"), ("near", "global"), ("far", "global"))
# These are descriptive report targets only, never used to choose model features.
REPORT_CHANNELS = (12, 24, 35, 49)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=str(ROOT.parent / "Dataset/LA"))
    p.add_argument("--dataset", choices=["LA"], default="LA")
    p.add_argument("--labeled_num", type=int, default=10)
    p.add_argument("--model_path", default=str(ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth"))
    p.add_argument("--patch_size", type=int, nargs=3, default=[112, 112, 80])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--patches_per_case", type=int, default=3)
    p.add_argument("--min_band_voxels", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=200)
    p.add_argument("--matching_policy", choices=["strict_four_bands", "all_six_min"], default="strict_four_bands")
    p.add_argument("--topk_ratios", type=float, nargs="+", default=[0.125, 0.25])
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--random_samples", type=int, default=5000)
    p.add_argument("--shuffle_repeats", type=int, default=200)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--reference_output_dir", default=str(HERE / "results/LA_10pct_seed42"))
    p.add_argument("--output_dir", default=str(HERE / "results/LA_10pct_seed42_specificity"))
    args = p.parse_args(argv)
    for name in ("patches_per_case", "min_band_voxels", "max_retries", "bootstrap_samples",
                 "random_samples", "shuffle_repeats", "cpu_threads"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.max_retries < args.patches_per_case:
        p.error("--max_retries must be >= --patches_per_case")
    if any(s < 16 or s % 16 for s in args.patch_size):
        p.error("VNet patch dimensions must be multiples of 16")
    if any(not 0 < r <= 1 for r in args.topk_ratios):
        p.error("--topk_ratios must lie in (0,1]")
    args.topk_ratios = sorted(set(args.topk_ratios))
    if args.in_channels != 1 or args.num_classes != 2:
        p.error("This experiment uses the existing single-channel binary LA baseline")
    return args


def morphology(mask, iterations, erode=False):
    result = mask.float()
    for _ in range(iterations):
        result = 1 - TF.max_pool3d(1 - result, 3, 1, 1) if erode else TF.max_pool3d(result, 3, 1, 1)
    return result > 0.5


def make_regions(label, spatial_shape):
    mask = TF.interpolate(label.float(), size=tuple(spatial_shape), mode="nearest") > 0.5
    e1, e2, e3 = (morphology(mask, n, erode=True) for n in (1, 2, 3))
    d1, d2, d3 = (morphology(mask, n) for n in (1, 2, 3))
    regions = dict(near_in=mask & ~e1, near_out=d1 & ~mask,
                   far_in=e2 & ~e3, far_out=d3 & ~d2,
                   foreground_core=e3, background_core=~d3)
    for a, b in itertools.combinations(REGIONS, 2):
        if bool((regions[a] & regions[b]).any()):
            raise RuntimeError(f"Overlapping transition regions: {a}, {b}")
    boundary = d1 & ~e1
    regions["boundary"] = boundary
    regions["nonboundary"] = ~morphology(boundary, 1)
    return regions


def matched_indices(regions, rng, policy, minimum):
    counts = {name: int(regions[name].sum()) for name in REGIONS}
    n_four = min(counts[name] for name in REGIONS[:4])
    n = min(counts.values()) if policy == "all_six_min" else n_four
    if n < minimum:
        return None, counts, n, "insufficient_region_voxels"
    if min(counts[name] for name in REGIONS[4:]) < n:
        return None, counts, n, "global_core_cannot_match_four_band_min"
    indices = {name: rng.choice(torch.nonzero(regions[name].reshape(-1), as_tuple=False).reshape(-1).cpu().numpy(),
                                size=n, replace=False) for name in REGIONS}
    return indices, counts, n, ""


def standardized_samples(feature, inside_indices, outside_indices):
    flat = feature[0].reshape(feature.shape[1], -1)
    inside = flat.index_select(1, torch.as_tensor(inside_indices, device=feature.device)).double()
    outside = flat.index_select(1, torch.as_tensor(outside_indices, device=feature.device)).double()
    moments = dict(mu_in=inside.mean(dim=1), mu_out=outside.mean(dim=1),
                   var_in=inside.var(dim=1, unbiased=False), var_out=outside.var(dim=1, unbiased=False))
    d = (moments["mu_in"] - moments["mu_out"]) / torch.sqrt((moments["var_in"] + moments["var_out"]) / 2 + EPS)
    return d.cpu().numpy(), {key: value.cpu().numpy() for key, value in moments.items()}


def spatial_gradient(feature):
    if min(feature.shape[2:]) < 2:
        raise ValueError("Finite differences require each feature dimension >=2")
    # Forward differences anchored at their lower-index voxel. Replicate the
    # last real difference at each terminal plane instead of inventing zeros.
    gx = TF.pad((feature[:, :, 1:] - feature[:, :, :-1]).abs(), (0, 0, 0, 0, 0, 1), mode="replicate")
    gy = TF.pad((feature[:, :, :, 1:] - feature[:, :, :, :-1]).abs(), (0, 0, 0, 1, 0, 0), mode="replicate")
    gz = TF.pad((feature[:, :, :, :, 1:] - feature[:, :, :, :, :-1]).abs(), (0, 1, 0, 0, 0, 0), mode="replicate")
    return (gx + gy + gz) / 3


def bgs_vector(feature, regions):
    gradient = spatial_gradient(feature)
    boundary = gradient[0, :, regions["boundary"][0, 0]].double().mean(dim=1)
    control = gradient[0, :, regions["nonboundary"][0, 0]].double().mean(dim=1)
    bgs = (boundary - control) / (boundary + control + EPS)
    return bgs.cpu().numpy(), boundary.cpu().numpy(), control.cpu().numpy()


def finite_vector(value, kind, case_id, patch_id, view, logger):
    for channel in np.flatnonzero(~np.isfinite(value)):
        logger.error("NONFINITE_VECTOR kind=%s case=%s patch=%s view=%s channel=%d value=%s",
                     kind, case_id, patch_id, view, channel, value[channel])
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite scientific measurement; see specificity_log.txt")


def signed_spearman(a, b):
    # BGS ranks use their signed values; abs(BGS) would favor negative selectivity.
    a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else float("nan")


def score_topk(scores, ratio):
    k = max(1, min(len(scores), round(len(scores) * ratio)))
    return set(np.argsort(-scores, kind="stable")[:k].tolist())


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denom, -1, 1)) if denom > EPS * EPS else float("nan")


def case_weights(n, samples, rng):
    return rng.multinomial(n, np.full(n, 1 / n), size=samples)


def mean_ci(values, samples, rng):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan
    estimates = rng.choice(vals, size=(samples, len(vals)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.percentile(estimates, [2.5, 97.5]))


def summary_row(scope, metric, source, values, ci, n_cases, ratio="", k="", method="case_bootstrap_mean"):
    row = dict(layer="x3", scope=scope, metric=metric, source=source, topk_ratio=ratio, k=k,
               n_cases=n_cases, ci_method=method, **base.distribution_summary(values))
    row["ci95_low"], row["ci95_high"] = ci
    return row


@torch.no_grad()
def extract_x3(model, image, logger, metadata, view):
    if torch.is_grad_enabled() or any(m.training for m in model.modules()):
        raise RuntimeError("Every encoder call must use eval and no_grad")
    features = model.encoder(image)
    if len(features) != 5:
        raise RuntimeError("Expected original encoder outputs [x1,x2,x3,x4,x5]")
    feature = features[2]
    shape = list(feature.shape)
    if shape not in metadata["feature_shapes"]:
        metadata["feature_shapes"].append(shape)
        logger.info("FEATURE_SHAPE layer=x3 shape=%s view=%s", shape, view)
    return feature


def compare_transitions(cases, case_vectors, patches, args, rng, logger):
    rows, summaries = [], []
    for a, b in COMPARISONS:
        metric = f"{a}_{b}_cosine"
        case_means, vector_cosines = [], []
        for case_id in cases:
            selected = [p for p in patches if p["case_id"] == case_id]
            values = [cosine(p[a], p[b]) for p in selected]
            for patch, value in zip(selected, values):
                if not np.isfinite(value):
                    logger.warning("UNDEFINED_COSINE comparison=%s case=%s patch=%s", metric, case_id, patch["patch_id"])
                rows.append(dict(unit="patch", case_id=case_id, patch_id=patch["patch_id"], comparison=metric,
                                 cosine=value, matched_voxels=patch["sample_n"], valid_patches=""))
            defined = [v for v in values if np.isfinite(v)]
            mean = float(np.mean(defined)) if defined else np.nan
            vector_cos = cosine(case_vectors[a][case_id], case_vectors[b][case_id])
            case_means.append(mean)
            vector_cosines.append(vector_cos)
            rows.append(dict(unit="case_mean_of_patch_cosines", case_id=case_id, patch_id="", comparison=metric,
                             cosine=mean, matched_voxels="", valid_patches=len(defined)))
            rows.append(dict(unit="cosine_of_case_mean_vectors", case_id=case_id, patch_id="", comparison=metric,
                             cosine=vector_cos, matched_voxels="", valid_patches=len(selected)))
        summaries.append(summary_row("transition_case_mean_of_patch_cosines", metric, "real", case_means,
                         mean_ci(case_means, args.bootstrap_samples, rng), len(cases)))
        summaries.append(summary_row("transition_cosine_of_case_mean_vectors", metric, "real", vector_cosines,
                         mean_ci(vector_cosines, args.bootstrap_samples, rng), len(cases)))
    return rows, summaries


def compare_bgs(cases, case_vectors, patches, args, rng, logger):
    n = len(cases)
    if not n:
        return [], [], []
    vectors = [case_vectors["bgs"][c] for c in cases]
    channels = len(vectors[0])
    weights = case_weights(n, args.bootstrap_samples, rng)
    pairs = list(itertools.combinations(range(n), 2))
    rank_matrix, null_matrix = base.matrix_for(n), base.matrix_for(n)
    pair_rows, null_rows, summaries = [], [], []
    upper = np.triu_indices(n, 1)
    for i, j in pairs:
        rank_matrix[i, j] = signed_spearman(vectors[i], vectors[j])
        shuffled = [signed_spearman(vectors[i], rng.permutation(vectors[j])) for _ in range(args.shuffle_repeats)]
        defined = [v for v in shuffled if np.isfinite(v)]
        null_matrix[i, j] = np.mean(defined) if defined else np.nan
        if not np.isfinite(rank_matrix[i, j]):
            logger.warning("UNDEFINED_BGS_SPEARMAN cases=%s,%s", cases[i], cases[j])
        for repeat, value in enumerate(shuffled):
            null_rows.append(dict(metric="bgs_spearman", topk_ratio="", k="", case_i=cases[i], case_j=cases[j], repeat=repeat, value=value))
    for source, matrix in (("real", rank_matrix), ("channel_shuffle", null_matrix), ("real_minus_shuffle", rank_matrix-null_matrix)):
        summaries.append(summary_row("bgs_cross_case", "bgs_spearman", source, matrix[upper],
                         base.case_pair_ci(matrix, weights), n, method="case_bootstrap_offdiagonal_mean"))
    for ratio in args.topk_ratios:
        sets = [score_topk(v, ratio) for v in vectors]
        k = len(sets[0])
        matrix = base.matrix_for(n)
        for i, j in pairs:
            matrix[i, j] = base.jaccard(sets[i], sets[j])
            pair_rows.append(dict(case_i=cases[i], case_j=cases[j], topk_ratio=ratio, k=k,
                                  spearman_bgs=rank_matrix[i, j], jaccard_bgs=matrix[i, j],
                                  shuffled_spearman_mean=null_matrix[i, j],
                                  topk_positive_fraction_i=float(np.mean([vectors[i][c] > 0 for c in sets[i]])),
                                  topk_positive_fraction_j=float(np.mean([vectors[j][c] > 0 for c in sets[j]]))))
        random_j = np.array([base.jaccard(set(rng.choice(channels, k, replace=False)),
                                         set(rng.choice(channels, k, replace=False))) for _ in range(args.random_samples)])
        for repeat, value in enumerate(random_j):
            null_rows.append(dict(metric="bgs_topk_jaccard", topk_ratio=ratio, k=k, case_i="", case_j="", repeat=repeat, value=value))
        summaries.append(summary_row("bgs_cross_case", "bgs_topk_jaccard", "real", matrix[upper],
                         base.case_pair_ci(matrix, weights), n, ratio, k, "case_bootstrap_offdiagonal_mean"))
        summaries.append(summary_row("bgs_cross_case", "bgs_topk_jaccard", "random_subsets", random_j,
                         base.iid_ci(random_j, args.bootstrap_samples, rng), n, ratio, k, "iid_random_draw_bootstrap_mean"))
        delta = matrix - random_j.mean()
        summaries.append(summary_row("bgs_cross_case", "bgs_topk_jaccard", "real_minus_random", delta[upper],
                         base.case_pair_ci(delta, weights), n, ratio, k, "case_bootstrap_fixed_MC_null_mean"))
    mean_bgs = [float(case_vectors["bgs"][case_id].mean()) for case_id in cases]
    positive = [float(np.mean(case_vectors["bgs"][case_id] > 0)) for case_id in cases]
    for metric, values in (("channel_mean_bgs", mean_bgs), ("positive_bgs_channel_fraction", positive)):
        summaries.append(summary_row("bgs_selectivity", metric, "real", values,
                         mean_ci(values, args.bootstrap_samples, rng), n))
    return pair_rows, null_rows, summaries


def compare_relation(cases, case_vectors, args, rng, logger):
    rows, summaries = [], []
    correlations = []
    for case_id in cases:
        near, bgs = np.abs(case_vectors["near"][case_id]), case_vectors["bgs"][case_id]
        corr = signed_spearman(near, bgs)
        if not np.isfinite(corr):
            logger.warning("UNDEFINED_BTV_BGS_RELATION case=%s", case_id)
        correlations.append(corr)
        for ratio in args.topk_ratios:
            a, b = score_topk(near, ratio), score_topk(bgs, ratio)
            rows.append(dict(case_id=case_id, topk_ratio=ratio, k=len(a), spearman_abs_near_bgs=corr,
                             topk_jaccard=base.jaccard(a,b), near_topk=json.dumps(sorted(a)), bgs_topk=json.dumps(sorted(b))))
    if cases:
        summaries.append(summary_row("btv_bgs_relation", "spearman_abs_near_bgs", "real", correlations,
                         mean_ci(correlations, args.bootstrap_samples, rng), len(cases)))
        for ratio in args.topk_ratios:
            selected = [r for r in rows if r["topk_ratio"] == ratio]
            values = [r["topk_jaccard"] for r in selected]
            summaries.append(summary_row("btv_bgs_relation", "topk_overlap", "real", values,
                             mean_ci(values, args.bootstrap_samples, rng), len(cases), ratio, selected[0]["k"]))
    return rows, summaries


def compare_augmentation(cases, patches, args, rng, logger):
    rows, summaries = [], []
    for patch in patches:
        for a,b in itertools.combinations(base.AUGMENTATIONS,2):
            if a not in patch["aug_bgs"] or b not in patch["aug_bgs"]:
                logger.info("SKIP_AUG_PAIR case=%s patch=%s views=%s,%s", patch["case_id"], patch["patch_id"],a,b)
                continue
            va,vb=patch["aug_bgs"][a],patch["aug_bgs"][b]
            rank, cos = signed_spearman(va,vb), cosine(va,vb)
            if not np.isfinite(rank) or not np.isfinite(cos):
                logger.warning("UNDEFINED_AUG_BGS case=%s patch=%s views=%s,%s",patch["case_id"],patch["patch_id"],a,b)
            for ratio in args.topk_ratios:
                sa,sb=score_topk(va,ratio),score_topk(vb,ratio)
                rows.append(dict(case_id=patch["case_id"],patch_id=patch["patch_id"],augmentation_i=a,augmentation_j=b,
                                 topk_ratio=ratio,k=len(sa),spearman_bgs=rank,jaccard_bgs=base.jaccard(sa,sb),cosine_bgs=cos))
    if cases:
        weights=case_weights(len(cases),args.bootstrap_samples,rng)
        for ratio in args.topk_ratios:
            selected=[r for r in rows if r["topk_ratio"]==ratio]
            groups=[cases.index(r["case_id"]) for r in selected]
            for metric,field in (("bgs_spearman","spearman_bgs"),("bgs_topk_jaccard","jaccard_bgs"),("bgs_cosine","cosine_bgs")):
                if metric!="bgs_topk_jaccard" and ratio!=args.topk_ratios[0]:
                    continue
                values=[r[field] for r in selected]
                summaries.append(summary_row("bgs_intra_sample_augmentation",metric,"real",values,
                                 base.cluster_ci(values,groups,weights),len(cases),
                                 ratio if metric=="bgs_topk_jaccard" else "",
                                 selected[0]["k"] if metric=="bgs_topk_jaccard" and selected else "",
                                 "case_cluster_bootstrap_mean"))
    return rows,summaries


def channel_descriptions(cases, case_vectors, args):
    rows=[]
    if not cases:
        return rows
    near=np.stack([abs(case_vectors["near"][c]) for c in cases])
    bgs=np.stack([case_vectors["bgs"][c] for c in cases])
    mean_near,mean_bgs=near.mean(axis=0),bgs.mean(axis=0)
    rank_near,rank_bgs=rankdata(-mean_near),rankdata(-mean_bgs)
    for channel in range(near.shape[1]):
        rows.append(dict(channel=channel,report_focus=channel in REPORT_CHANNELS,
                         mean_abs_d_near=float(mean_near[channel]),mean_bgs=float(mean_bgs[channel]),
                         d_near_rank=float(rank_near[channel]),bgs_rank=float(rank_bgs[channel]),
                         case_mean_d_near_rank=float(np.mean([rankdata(-v)[channel] for v in near])),
                         case_mean_bgs_rank=float(np.mean([rankdata(-v)[channel] for v in bgs])),
                         positive_bgs_case_fraction=float(np.mean(bgs[:,channel]>0))))
    return rows


def save_vectors(out,cases,case_vectors,patches):
    channels=len(case_vectors["near"][cases[0]]) if cases else 0
    for kind in KINDS:
        vectors=np.stack([case_vectors[kind][c] for c in cases]) if cases else np.empty((0,channels))
        payload=dict(case_ids=np.array(cases,dtype=str),case_vectors=vectors,
                     patch_case_ids=np.array([p["case_id"] for p in patches],dtype=str),
                     patch_ids=np.array([p["patch_id"] for p in patches],dtype=int),
                     patch_vectors=np.stack([p[kind] for p in patches]) if patches else np.empty((0,channels)),
                     case_patch_counts=np.array([sum(p["case_id"]==c for p in patches) for c in cases]),
                     sample_counts=np.array([p["sample_n"] for p in patches]),eps=np.array(EPS))
        if kind!="bgs":
            payload["case_vectors_norm"]=np.stack([base.normalize(v) for v in vectors]) if cases else vectors.copy()
            for key in ("mu_in","mu_out","var_in","var_out"):
                payload[f"patch_{key}"]=np.stack([p[f"{kind}_moments"][key] for p in patches]) if patches else np.empty((0,channels))
        if kind=="near":
            payload["patch_full_band_vectors"]=np.stack([p["near_full"] for p in patches]) if patches else np.empty((0,channels))
        if kind=="bgs":
            for key in ("g_boundary","g_nonboundary"):
                payload[f"patch_{key}"]=np.stack([p[key] for p in patches]) if patches else np.empty((0,channels))
            aug=[(p["case_id"],p["patch_id"],name,v) for p in patches for name,v in p["aug_bgs"].items()]
            payload.update(aug_case_ids=np.array([a[0] for a in aug],dtype=str),aug_patch_ids=np.array([a[1] for a in aug],dtype=int),
                           aug_names=np.array([a[2] for a in aug],dtype=str),
                           aug_vectors=np.stack([a[3] for a in aug]) if aug else np.empty((0,channels)))
        np.savez_compressed(out/f"case_{kind}_vectors.npz",**payload)
    sampling=dict(patch_case_ids=np.array([p["case_id"] for p in patches],dtype=str),
                  patch_ids=np.array([p["patch_id"] for p in patches],dtype=int),
                  offsets=np.r_[0,np.cumsum([p["sample_n"] for p in patches])],
                  spatial_shapes=np.array([p["spatial_shape"] for p in patches],dtype=int))
    for name in REGIONS:
        sampling[f"{name}_indices"]=np.concatenate([p["indices"][name] for p in patches]) if patches else np.empty(0,dtype=int)
    np.savez_compressed(out/"sampled_voxel_indices.npz",**sampling)


def plot_results(out,summaries,channels,args):
    plt=base.plt
    def find(scope,metric,source="real",ratio=None):
        return next((r for r in summaries if r["scope"]==scope and r["metric"]==metric and r["source"]==source
                     and (ratio is None or r["topk_ratio"]==ratio)),None)
    def bar(ax,x,row,label,color):
        if row:
            ax.bar(x,row["mean"],label=label,color=color,width=0.35)
            ax.vlines(x,row["ci95_low"],row["ci95_high"],color="black",linewidth=1)
    fig,ax=plt.subplots(figsize=(7,4))
    names=[f"{a}_{b}_cosine" for a,b in COMPARISONS]
    for i,name in enumerate(names):
        bar(ax,i,find("transition_case_mean_of_patch_cosines",name),None,"tab:blue")
    ax.set(xticks=range(3),xticklabels=["near vs far","near vs global","far vs global"],ylim=(-1,1),ylabel="Mean signed cosine (95% CI)")
    fig.tight_layout();fig.savefig(out/"near_far_global_cosine.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,1+len(args.topk_ratios),figsize=(5*(1+len(args.topk_ratios)),4),squeeze=False)
    panels=[("bgs_spearman",None,"channel_shuffle")]+[("bgs_topk_jaccard",r,"random_subsets") for r in args.topk_ratios]
    for ax,(metric,ratio,null) in zip(axes[0],panels):
        bar(ax,0,find("bgs_cross_case",metric,ratio=ratio),"real","tab:blue")
        bar(ax,1,find("bgs_cross_case",metric,null,ratio),null,"tab:orange")
        ax.set(xticks=[0,1],xticklabels=["real","random"],title=metric+(f" {ratio:g}" if ratio else ""),ylim=(-1,1) if ratio is None else (0,1))
    fig.tight_layout();fig.savefig(out/"bgs_cross_case_stability.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for i,ratio in enumerate(args.topk_ratios):
        bar(ax,i,find("btv_bgs_relation","topk_overlap",ratio=ratio),None,"tab:blue")
    ax.set(xticks=range(len(args.topk_ratios)),xticklabels=[f"Top {r*100:g}%" for r in args.topk_ratios],ylim=(0,1),ylabel="BTV / BGS Top-K Jaccard (95% CI)")
    fig.tight_layout();fig.savefig(out/"btv_bgs_topk_overlap.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter([r["mean_abs_d_near"] for r in channels],[r["mean_bgs"] for r in channels],s=22,alpha=.7)
    for row in channels:
        if row["report_focus"]:
            ax.annotate(str(row["channel"]),(row["mean_abs_d_near"],row["mean_bgs"]),xytext=(5,5),textcoords="offset points")
    ax.axhline(0,color="gray",linewidth=.8)
    ax.set(xlabel="Mean over cases of abs(case d_near)",ylabel="Mean case BGS",title="x3: transition strength and boundary selectivity")
    fig.tight_layout();fig.savefig(out/"near_strength_bgs_scatter.png",dpi=160);plt.close(fig)


def write_report(out,args,metadata,summaries,channels):
    def find(scope,metric,source="real",ratio=None):
        return next((r for r in summaries if r["scope"]==scope and r["metric"]==metric and r["source"]==source
                     and (ratio is None or r["topk_ratio"]==ratio)),None)
    def value(row):
        return f"{row['mean']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]" if row else "不可计算"
    lines=["# Experiment 0.5 — Boundary Specificity Audit", "",
           "本报告提供数据和 A/B/C 假设的解释依据，**不自动宣称 BTV 是 boundary-specific**。", "",
           f"沿用 Experiment 0 的 LA 10%、seed={args.seed}、checkpoint `{Path(args.model_path).name}`、"
           f"patch={args.patch_size}；只分析 x3，实际 feature shape={metadata['feature_shapes']}。",
           f"有效病例={metadata.get('valid_cases',0)}，跳过病例={metadata.get('skipped_cases',0)}，"
           f"partial 病例={metadata.get('partial_cases',0)}，有效 identity patch={metadata.get('valid_patches',0)}。", ""]
    if args.matching_policy=="all_six_min":
        lines += ["采样规则使用明确选择的 `all_six_min`：四条 near/far band 和两个 global core 的 voxel 数统一取最小值，"
                  "六个区域各自无放回等量抽样。原协议的四条 band 最小值在当前 LA/x3 上超过 foreground core 容量；"
                  "预检查每病例 200 次 crop，8 个病例均无法按原数量匹配。该调整保持六组统计的样本数一致，"
                  "不改变 ring/core 定义；实际 counts 和 sampled indices 均保存。", ""]
    else:
        lines += ["严格使用原四条 band 的最小 voxel 数；global core 不够时跳过 patch，不使用重复采样或改变 ring 定义。", ""]
    lines += ["以下 CI 都是均值的 95% bootstrap CI。先在每个 patch 计算 signed cosine，再对同一病例的 patch cosine 求均值，"
              "最后按病例汇总和 bootstrap；`cosine_of_case_mean_vectors` 是独立保存的补充统计，不能与主统计混用。", "",
              "| 主比较 | mean [95% CI] |", "|---|---:|"]
    for a,b in COMPARISONS:
        lines.append(f"| {a} vs {b} | {value(find('transition_case_mean_of_patch_cosines',f'{a}_{b}_cosine'))} |")
    lines += ["", "BGS 使用 signed 值排名，正值表示 boundary 的平均 feature gradient 高于 non-boundary control。"
              "按 abs(BGS) 选 Top-K 会误选强负值，本实现不这样做。BGS 的 spatial gradient 是三个方向 forward difference"
              "绝对值的平均，终端平面复制最后一个有效差分；使用全部 boundary/nonboundary voxel 求均值。", "",
              "| BGS 指标 | real mean [95% CI] | random mean [95% CI] |", "|---|---:|---:|"]
    lines.append(f"| 跨病例 signed BGS Spearman | {value(find('bgs_cross_case','bgs_spearman'))} | {value(find('bgs_cross_case','bgs_spearman','channel_shuffle'))} |")
    for ratio in args.topk_ratios:
        lines.append(f"| 跨病例 Top-{ratio*100:g}% Jaccard | {value(find('bgs_cross_case','bgs_topk_jaccard',ratio=ratio))} | {value(find('bgs_cross_case','bgs_topk_jaccard','random_subsets',ratio))} |")
    lines += ["",f"mean BGS：{value(find('bgs_selectivity','channel_mean_bgs'))}；"
              f"正 BGS channel 比例：{value(find('bgs_selectivity','positive_bgs_channel_fraction'))}。", "",
              "| 同一 patch 的 BGS augmentation 指标 | mean [95% CI] |", "|---|---:|"]
    for metric in ("bgs_spearman","bgs_cosine"):
        lines.append(f"| {metric} | {value(find('bgs_intra_sample_augmentation',metric))} |")
    for ratio in args.topk_ratios:
        lines.append(f"| Top-{ratio*100:g}% Jaccard | {value(find('bgs_intra_sample_augmentation','bgs_topk_jaccard',ratio=ratio))} |")
    lines += ["",f"病例级 Spearman(abs(d_near), BGS)：{value(find('btv_bgs_relation','spearman_abs_near_bgs'))}。"]
    for ratio in args.topk_ratios:
        lines.append(f"BTV / BGS Top-{ratio*100:g}% Jaccard：{value(find('btv_bgs_relation','topk_overlap',ratio=ratio))}。")
    lines += ["", "以下 12/24/35/49 只作为 Experiment 0 的解释对象，未用于采样、feature 选择或任何模型操作。"
              "rank 按跨病例平均后的 channel score 排名，1 为最高；mean abs(d_near) 取 mean(abs(case_d))，"
              "case_d 先对原始 patch d 求均值。", "",
              "| channel | mean abs(d_near) | mean BGS | d_near rank | BGS rank | 正 BGS 病例比例 |", "|---|---:|---:|---:|---:|---:|"]
    for row in channels:
        if row["report_focus"]:
            lines.append(f"| {row['channel']} | {row['mean_abs_d_near']:.4f} | {row['mean_bgs']:.4f} | {row['d_near_rank']:g} | {row['bgs_rank']:g} | {row['positive_bgs_case_fraction']:.3f} |")
    lines += ["", "三种可能都需要讨论：", "",
              "A. near–global similarity 不高且 BGS 稳定，支持 boundary-specific representation。",
              "B. near–global similarity 高且 BGS 稳定，说明 BTV 更像 semantic class contrast，同时存在 boundary-selective channels。",
              "C. near–global similarity 高且 BGS 不稳定，global channel-level boundary representation 缺乏支持，可在后续研究考虑 local boundary modeling。", "",
              "上述‘高/稳定’没有预注册硬阈值，不应仅凭一个绝对数值自动分类。应结合完整 ranking、Top-K 相对随机基线、"
              "增强稳定性和正 BGS 响应判断。具体研究判断将在核验结果后补充。", "",
              "该审计限于一个 checkpoint 和 8 个 labeled 病例。等量 sampling 的样本数受 foreground core 限制，"
              "可能提高 d 的采样噪声；near/far/core 采用 pooling 的离散网格距离，并非物理距离。"
              "forward difference 的锚点及 nearest resize 相位也会影响几何增强比较。"
              "BGS 的全局 non-boundary control 包含前景和背景不同组织，本身也不是边界专属机制的因果证明。", "",
              "保存原始 patch/case vectors、moments、BGS 分子所需的梯度均值和 voxel indices。"
              "NaN/Inf 按 case/patch/view/channel 明确报错；常数 ranking/零向量产生的无定义相关性单独记录。"
              "baseline 代码、checkpoint 和 Experiment 0 目录以执行前后 SHA256 校验；model 参数与 buffers 也校验。"
              "全程 eval/no_grad，不引入训练或 segmentation module。", "",
              "文件：specificity_summary.csv、near_far_global_cosine.csv、case_*_vectors.npz、"
              "bgs_pairwise_stability.csv、btv_bgs_relation.csv、augmentation_bgs_stability.csv、"
              "channel_specificity.csv、case_status.csv、patch_status.csv、sampled_voxel_indices.npz、"
              "random_baselines.csv、specificity_metadata.json、specificity_log.txt 及四张 PNG。", "",
              "复跑使用新 output_dir，命令示例：", "", "```bash",
              "conda run --no-capture-output -n SSL python analysis/boundary_specificity_audit.py \\",
              f"  --matching_policy {args.matching_policy} --device {args.device} \\",
              "  --output_dir analysis/results/LA_10pct_seed42_specificity_rerun", "```", ""]
    (out/"specificity_report.md").write_text("\n".join(lines))


@torch.no_grad()
def audit(args,logger,metadata,events,case_status):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; explicitly use --device cpu")
    model=base.VNet(n_channels=args.in_channels,n_classes=args.num_classes)
    kwargs={"weights_only":True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    model.load_state_dict(torch.load(args.model_path,map_location="cpu",**kwargs),strict=True)
    model.to(device);model.eval();model.requires_grad_(False)
    metadata["model_state_sha256_before"]=base.model_hash(model)
    ds=base.build_Dataset(args,data_dir=args.data_path,split="train_LA",transform=None)
    labeled_count=base.patients_to_slices(args.dataset,args.labeled_num)
    if labeled_count>len(ds.sample_list) or len(set(ds.sample_list[:labeled_count]))!=labeled_count:
        raise ValueError("Invalid labeled training prefix")
    metadata.update(labeled_paths=ds.sample_list[:labeled_count],labeled_cases_requested=labeled_count,feature_shapes=[])
    logger.info("SPLIT labeled=%d total=%d; only labeled HDF5 files are opened",labeled_count,len(ds.sample_list))
    crop,tensorize=base.RandomCrop(args.patch_size),base.ToTensor()
    sampling_rng=np.random.default_rng(args.seed)
    patches=[]
    case_vectors={kind:{} for kind in KINDS}
    for case_id,path in zip(ds.image_list[:labeled_count],ds.sample_list[:labeled_count]):
        logger.info("CASE_START case=%s path=%s",case_id,path)
        try:
            with h5py.File(path,"r") as f: image_np,label_np=f["image"][:],f["label"][:]
        except (OSError,KeyError) as e:
            case_status.append(dict(case_id=case_id,attempts=0,valid_patches=0,status="skipped",reason=str(e)))
            logger.error("SKIP_CASE case=%s reason=%s",case_id,e);continue
        if image_np.ndim!=3 or image_np.shape!=label_np.shape or not np.isin(np.unique(label_np),[0,1]).all():
            raise ValueError(f"Unexpected image/label in {case_id}")
        if not np.isfinite(image_np).all() or not np.isfinite(label_np).all():
            raise FloatingPointError(f"Nonfinite input in {case_id}")
        selected=[]
        for attempt in range(1,args.max_retries+1):
            patch=tensorize(crop({"image":image_np,"label":label_np}))
            image=patch["image"][None].to(device);label=patch["label"][None,None].to(device)
            feature=extract_x3(model,image,logger,metadata,"identity")
            regions=make_regions(label,feature.shape[2:])
            indices,counts,sample_n,reason=matched_indices(regions,sampling_rng,args.matching_policy,args.min_band_voxels)
            event=dict(case_id=case_id,patch_id=attempt,augmentation="identity",feature_shape=str(list(feature.shape)),
                       sample_n=sample_n,**counts,boundary=int(regions["boundary"].sum()),nonboundary=int(regions["nonboundary"].sum()))
            if not reason and min(event["boundary"],event["nonboundary"])<args.min_band_voxels:
                reason="insufficient_bgs_region_voxels"
            event.update(status="skipped" if reason else "valid",reason=reason);events.append(event)
            if reason:
                logger.info("SKIP_PATCH case=%s patch=%d reason=%s counts=%s",case_id,attempt,reason,counts);continue
            record=dict(case_id=case_id,patch_id=attempt,sample_n=sample_n,indices=indices,spatial_shape=tuple(feature.shape[2:]))
            for kind,a,b in (("near","near_in","near_out"),("far","far_in","far_out"),("global","foreground_core","background_core")):
                record[kind],record[f"{kind}_moments"]=standardized_samples(feature,indices[a],indices[b])
                finite_vector(record[kind],kind,case_id,attempt,"identity",logger)
            record["near_full"]=base.transition_vector(feature,regions["near_in"],regions["near_out"])
            finite_vector(record["near_full"],"near_full",case_id,attempt,"identity",logger)
            record["bgs"],record["g_boundary"],record["g_nonboundary"]=bgs_vector(feature,regions)
            finite_vector(record["bgs"],"bgs",case_id,attempt,"identity",logger)
            record["aug_bgs"]={"identity":record["bgs"]}
            for view in base.AUGMENTATIONS[1:]:
                x,y=base.geometric_view(image,label,view)
                f=extract_x3(model,x,logger,metadata,view);r=make_regions(y,f.shape[2:])
                nb,nc=int(r["boundary"].sum()),int(r["nonboundary"].sum())
                ev=dict(case_id=case_id,patch_id=attempt,augmentation=view,feature_shape=str(list(f.shape)),boundary=nb,nonboundary=nc)
                if min(nb,nc)<args.min_band_voxels:
                    ev.update(status="skipped",reason="insufficient_bgs_region_voxels")
                    logger.info("SKIP_AUG case=%s patch=%d view=%s",case_id,attempt,view)
                else:
                    bgs,_,_=bgs_vector(f,r);finite_vector(bgs,"bgs",case_id,attempt,view,logger)
                    record["aug_bgs"][view]=bgs;ev.update(status="valid",reason="")
                events.append(ev)
            patches.append(record);selected.append(record)
            logger.info("PATCH_ACCEPTED case=%s patch=%d valid=%d sample_n=%d counts=%s",case_id,attempt,len(selected),sample_n,counts)
            if len(selected)==args.patches_per_case:
                break
        for kind in KINDS:
            if selected:
                value=np.mean([p[kind] for p in selected],axis=0)
                finite_vector(value,f"case_{kind}",case_id,"mean","identity",logger)
                case_vectors[kind][case_id]=value
        status="complete" if len(selected)==args.patches_per_case else "partial" if selected else "skipped"
        case_status.append(dict(case_id=case_id,attempts=attempt,valid_patches=len(selected),status=status,
                                reason="" if status=="complete" else "max_retries_exhausted"))
        logger.info("CASE_DONE case=%s status=%s valid_patches=%d attempts=%d",case_id,status,len(selected),attempt)
    cases=sorted(case_vectors["near"])
    metadata.update(valid_cases=len(cases),skipped_cases=sum(r["status"]=="skipped" for r in case_status),
                    partial_cases=sum(r["status"]=="partial" for r in case_status),valid_patches=len(patches),
                    patch_skip_reasons=dict(Counter(r["reason"] for r in events if r["status"]=="skipped")),
                    matched_voxel_counts=[p["sample_n"] for p in patches],nonfinite_vector_elements=0)
    rng=np.random.default_rng(args.seed+10000)
    transition_rows,summaries=compare_transitions(cases,case_vectors,patches,args,rng,logger)
    pair_rows,null_rows,s=compare_bgs(cases,case_vectors,patches,args,rng,logger);summaries+=s
    relation_rows,s=compare_relation(cases,case_vectors,args,rng,logger);summaries+=s
    aug_rows,s=compare_augmentation(cases,patches,args,rng,logger);summaries+=s
    channels=channel_descriptions(cases,case_vectors,args)
    out=Path(args.output_dir);save_vectors(out,cases,case_vectors,patches)
    base.write_csv(out/"specificity_summary.csv",summaries,["layer","scope","metric","source","topk_ratio","k","n_cases",
                   "n_observations","n_defined","n_undefined","mean","std","median","ci95_low","ci95_high","ci_method"])
    base.write_csv(out/"near_far_global_cosine.csv",transition_rows,["unit","case_id","patch_id","comparison","cosine","matched_voxels","valid_patches"])
    base.write_csv(out/"bgs_pairwise_stability.csv",pair_rows,["case_i","case_j","topk_ratio","k","spearman_bgs","jaccard_bgs",
                   "shuffled_spearman_mean","topk_positive_fraction_i","topk_positive_fraction_j"])
    base.write_csv(out/"btv_bgs_relation.csv",relation_rows,["case_id","topk_ratio","k","spearman_abs_near_bgs","topk_jaccard","near_topk","bgs_topk"])
    base.write_csv(out/"augmentation_bgs_stability.csv",aug_rows,["case_id","patch_id","augmentation_i","augmentation_j","topk_ratio","k","spearman_bgs","jaccard_bgs","cosine_bgs"])
    base.write_csv(out/"channel_specificity.csv",channels,["channel","report_focus","mean_abs_d_near","mean_bgs","d_near_rank","bgs_rank",
                   "case_mean_d_near_rank","case_mean_bgs_rank","positive_bgs_case_fraction"])
    base.write_csv(out/"random_baselines.csv",null_rows,["metric","topk_ratio","k","case_i","case_j","repeat","value"])
    plot_results(out,summaries,channels,args);write_report(out,args,metadata,summaries,channels)
    metadata["model_state_sha256_after"]=base.model_hash(model)
    if metadata["model_state_sha256_after"]!=metadata["model_state_sha256_before"]:
        raise RuntimeError("Model parameters or buffers changed")
    logger.info("MODEL_STATE_UNCHANGED eval=%s no_grad=%s no_parameter_grads=%s",all(not m.training for m in model.modules()),
                not torch.is_grad_enabled(),all(p.grad is None for p in model.parameters()))
    logger.info("TOTALS valid_cases=%d skipped_cases=%d partial_cases=%d valid_patches=%d",metadata["valid_cases"],
                metadata["skipped_cases"],metadata["partial_cases"],metadata["valid_patches"])
    return bool(cases)


def protected_paths(args):
    paths=[ROOT/f for f in ("train.py","trainer.py","model/vnet.py","dataloader/dataset.py","prediction.py")]
    paths+=sorted(p for p in (ROOT/"utils").rglob("*") if p.is_file())
    paths+=[Path(args.model_path).resolve(),HERE/"boundary_representation_audit.py",HERE/"test_boundary_representation_audit.py",HERE/"README.md"]
    reference=Path(args.reference_output_dir).resolve()
    if not (reference/"audit_metadata.json").is_file():
        raise FileNotFoundError("Experiment 0 reference metadata is required")
    paths+=sorted(p for p in reference.rglob("*") if p.is_file())
    return sorted(set(paths))


def check_source():
    tree=ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,"id","")
            if name in {"backward","train","update_ema_variables"}:
                raise RuntimeError(f"Forbidden scientific analysis call: {name}")
        if isinstance(node,ast.Attribute) and node.attr in {"optim","optimizer","scheduler"}:
            raise RuntimeError("Forbidden training object in analysis")


def main(argv=None):
    args=parse_args(argv)
    reference=Path(args.reference_output_dir).resolve();out=Path(args.output_dir).resolve()
    if out==reference or reference in out.parents or out in reference.parents:
        raise ValueError("Output and Experiment 0 directories must be separate")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output_dir")
    protected=protected_paths(args)
    before={str(p):base.file_hash(p) for p in protected}
    ref=json.loads((reference/"audit_metadata.json").read_text())
    for key in ("dataset","labeled_num","patch_size","seed","patches_per_case","in_channels","num_classes"):
        if getattr(args,key)!=ref["arguments"][key]:
            raise ValueError(f"Experiment 0 setting mismatch: {key}")
    if Path(args.model_path).resolve()!=(ROOT/ref["arguments"]["model_path"]).resolve():
        raise ValueError("Checkpoint differs from Experiment 0")
    if Path(args.data_path).resolve()!=Path(ref["arguments"]["data_path"]).resolve():
        raise ValueError("Data path differs from Experiment 0")
    out.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger("specificity_audit");logger.setLevel(logging.INFO);logger.handlers.clear()
    for handler in (logging.FileHandler(out/"specificity_log.txt"),logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"));logger.addHandler(handler)
    metadata=dict(arguments=vars(args),protected_sha256_before=before,script_sha256=base.file_hash(__file__),
                  torch_version=torch.__version__,numpy_version=np.__version__,status="running")
    events,case_status=[],[]
    try:
        check_source();base.check_analysis_source()
        logger.info("ARGUMENTS %s",json.dumps(vars(args),sort_keys=True))
        success=audit(args,logger,metadata,events,case_status)
        metadata["status"]="complete" if success else "no_valid_patches"
    except BaseException:
        metadata["status"]="failed";logger.exception("SPECIFICITY_AUDIT_FAILED");raise
    finally:
        after={str(p):base.file_hash(p) for p in protected}
        metadata["protected_sha256_after"]=after;metadata["protected_files_unchanged"]=before==after
        base.write_csv(out/"case_status.csv",case_status,["case_id","attempts","valid_patches","status","reason"])
        base.write_csv(out/"patch_status.csv",events,["case_id","patch_id","augmentation","status","reason","feature_shape","sample_n",
                       *REGIONS,"boundary","nonboundary"])
        (out/"specificity_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        logger.info("PROTECTED_FILES_UNCHANGED=%s files=%d; includes baseline, checkpoint, all Experiment 0 artifacts",before==after,len(before))
        if before!=after:
            for path in before:
                if before[path]!=after[path]:logger.error("PROTECTED_FILE_CHANGED path=%s",path)
            raise RuntimeError("Protected file integrity failed")
    logger.info("SPECIFICITY_AUDIT_DONE status=%s output=%s",metadata["status"],out)


if __name__=="__main__":
    main()
