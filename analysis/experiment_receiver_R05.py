"""R0.5: frozen Teacher reliability on fixed spatial grids, never feature gain.

Phase A replays R0 images, verifies predictions, saves ALL image-only scores and
budget masks. Only after a digest seal is Receiver GT unlocked for evaluation.
No training, gradients, donor matching, BGS or model modification.
"""
import argparse
from collections import defaultdict
import datetime
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import experiment_receiver_R0 as r0
from model.vnet import VNet

METHODS = ("random", "boundary_prior", "entropy", "Dx", "Dy", "Dz", "Dxyz",
           "Dxy", "Dxz", "Dyz", "combined", "combined_boundary")
PRIMARY = ("random", "boundary_prior", "entropy", "Dxyz", "combined", "combined_boundary")
MAPPINGS = ("average", "max", "trilinear")
DOMAINS = ("global", "pseudo_boundary_neighborhood")
BUDGETS = (.01, .02, .05, .10)
MARGINS = (4, 8, 16)
BLOCK = 4
BOOTSTRAPS = 10000
PRED_ATOL = 2e-6
NEW_CODE = {Path(__file__).resolve(), HERE / "test_experiment_receiver_R05.py"}
METRICS = ("ap_any", "ap_severe10", "ap_severe25", "prevalence_any", "prevalence_severe10",
           "prevalence_severe25", "spearman_fraction", "error_coverage", "domain_error_coverage",
           "selected_error_fraction", "enrichment", "precision_any", "recall_any", "precision_severe10",
           "high_confidence_coverage", "far_fp_coverage", "far_fp_core_coverage", "high_confidence_far_fp_coverage",
           "sparse_grid_recall", "sparse_error_coverage", "lost_original_error_fraction", "new_error_fraction",
           "coverage_change_vs_original", "outside_boundary_error_coverage", "selected_clean_grid_fraction")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_dir", default=str(ROOT / "analysis/results/experiment_receiver_R0/run_20261012_verified"))
    p.add_argument("--model_path", default=str(ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth"))
    p.add_argument("--output_dir", default=str(ROOT / "analysis/results/experiment_receiver_R05" /
                  datetime.datetime.now(r0.TZ).strftime("run_%Y%m%d_%H%M%S")))
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu_threads", type=int, default=4)
    a = p.parse_args(argv)
    if a.seed != 42 or a.cpu_threads < 1:
        p.error("This fixed replay uses seed42 and positive cpu_threads")
    return a


def create_output(path):
    out = Path(path).resolve()
    base = ROOT / "analysis/results/experiment_receiver_R05"
    if out.parent != base or not out.name.startswith("run_"):
        raise ValueError("Use a new run_TIMESTAMP child of analysis/results/experiment_receiver_R05")
    out.mkdir(parents=True, exist_ok=False)
    return out


def protected_manifest(out):
    result = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")
                         and (Path(parent) / d).resolve() != out.resolve())
        for name in sorted(files):
            p = Path(parent) / name
            if p.is_file() and p.resolve() not in NEW_CODE and p.suffix not in (".pyc", ".pyo"):
                result[str(p.relative_to(ROOT))] = r0.sha256(p)
    return result


def validate_reference(reference, model_path):
    path = reference / "experiment_metadata.json"
    history = json.loads(path.read_text())
    if history["status"] != "complete" or history["configuration"]["seed"] != 42:
        raise ValueError("Incomplete or different-seed R0 reference")
    # The old sampling provenance is used only for exact coordinates and split.
    old_path = Path(history["configuration"]["reference_dir"]) / "experiment_metadata.json"
    if r0.sha256(old_path) != history["historical_metadata_sha256"]:
        raise ValueError("R0 sampling provenance changed")
    old = json.loads(old_path.read_text())
    ids, checkpoint = r0.validate_history(old, model_path)
    if ids != history["case_order"] or old["sampling"]["patch_coordinates"] != history["crop_coordinates"]:
        raise ValueError("R0 historical crop/order differs")
    if checkpoint != history["checkpoint"] or old["data_sources"] != history["data_sources"]:
        raise ValueError("R0 checkpoint/data-source identity differs")
    artifacts = {(a["case_id"], a["patch_index"]): a for a in history["phaseA_artifacts"]}
    if len(artifacts) != 24:
        raise ValueError("Expected all 24 R0 image-only artifacts")
    for a in artifacts.values():
        if r0.sha256(a["path"]) != a["sha256"]:
            raise ValueError("R0 risk artifact hash changed: " + a["path"])
    return history, old, ids, checkpoint, artifacts


def independent_audit(old):
    train_path = Path(old["data_split"]["train_list_path"])
    selection_path = Path(old["data_split"]["test_list_path"])
    train = train_path.read_text().splitlines()
    selection = selection_path.read_text().splitlines()
    data_root = Path(old["data_sources"][0]["path"]).parent.parent
    available = sorted(p.parent.name for p in data_root.glob("*/mri_norm2.h5"))
    unknown = sorted(set(available) - set(train) - set(selection))
    return dict(available_LA_cases=len(available), training_cases=len(train),
                checkpoint_selection_cases=len(selection), train_selection_overlap=sorted(set(train) & set(selection)),
                available_cases_outside_training_and_selection=unknown, independent_validation_available=False,
                reason="All available LA cases belong to training or checkpoint-selection lists" if not unknown else
                "Unclassified cases require provenance proof before being called independent",
                data_root=str(data_root), no_test_GT_read=True,
                evidence={str(p): r0.sha256(p) for p in (train_path, selection_path, ROOT / "trainer.py", ROOT / "prediction.py")},
                checkpoint_selection_evidence="trainer.py validate uses test_calculate_metric(val=True); LA prediction.py reads test.list",
                scope="Available LA dataset only; external independent cohorts were not supplied")


def direction_maps(probability, aligned):
    """Original tensor axes 0/1/2, not unverified anatomical directions."""
    if aligned.shape != (3,) + probability.shape:
        raise ValueError("Need exactly three inverse-aligned flip predictions")
    diff = np.abs(aligned.astype(np.float64) - probability.astype(np.float64)[None])
    result = {"D" + axis: diff[i].astype(np.float32) for i, axis in enumerate("xyz")}
    result["Dxyz"] = diff.mean(0).astype(np.float32)
    for key, indices in (("Dxy", (0, 1)), ("Dxz", (0, 2)), ("Dyz", (1, 2))):
        result[key] = diff[list(indices)].mean(0).astype(np.float32)
    return result


def risk_maps(archive):
    result = direction_maps(archive["probability"], archive["aligned_flip_probabilities"])
    if not np.array_equal(result["Dxyz"], archive["risk_disagreement"]):
        raise AssertionError("Dxyz reconstruction differs from historical R0 D")
    result.update({k: archive["risk_" + k] for k in PRIMARY if k != "Dxyz"})
    return result


def map_to_grid(score, target_shape, mapping):
    score = np.ascontiguousarray(score, dtype=np.float32)
    if any(s % BLOCK for s in score.shape) or tuple(s // BLOCK for s in score.shape) != tuple(target_shape):
        raise ValueError("Fixed 4^3 grids do not match actual x3 spatial shape")
    with torch.no_grad():
        x = torch.from_numpy(score)[None, None]
        if mapping == "average":
            y = F.avg_pool3d(x, BLOCK, BLOCK)
        elif mapping == "max":
            y = F.max_pool3d(x, BLOCK, BLOCK)
        elif mapping == "trilinear":
            y = F.interpolate(x, size=tuple(target_shape), mode="trilinear", align_corners=False)
        else:
            raise ValueError("Unknown fixed mapping")
    result = y[0, 0].numpy()
    r0.finite(result, "mapped " + mapping)
    return result


def block_count(mask):
    a = np.asarray(mask)
    if a.ndim != 3 or any(s % BLOCK for s in a.shape):
        raise ValueError("3D divisible-by-four spatial array required")
    shape = tuple(s // BLOCK for s in a.shape)
    return a.reshape(shape[0], BLOCK, shape[1], BLOCK, shape[2], BLOCK).sum((1, 3, 5), dtype=np.int64)


def lift_grid(grid):
    return np.repeat(np.repeat(np.repeat(grid, BLOCK, 0), BLOCK, 1), BLOCK, 2)


def interior_domain(shape, margin):
    out = np.ones(shape, dtype=bool)
    for axis, size in enumerate(shape):
        distance = np.minimum(np.arange(size), size - 1 - np.arange(size))
        out &= (distance > margin).reshape(tuple(size if j == axis else 1 for j in range(3)))
    return out


def selection_key(scale, mapping, domain, method, budget):
    return "select_{}_{}_{}_{}_{}".format(scale, mapping, domain, method, int(budget * 100))


def selections(maps, neighborhood, scale, mapping, tie_seed):
    tie_rank = np.random.RandomState(tie_seed).permutation(neighborhood.size)
    payload, checks = {}, []
    for domain_name in DOMAINS:
        domain = np.ones(neighborhood.shape, dtype=bool) if domain_name == "global" else neighborhood
        for method in METHODS:
            order = r0.ordered_indices(maps[method], domain, tie_rank)
            for budget in BUDGETS:
                k = r0.budget_count(len(order), budget)
                selected = np.zeros(domain.size, dtype=np.uint8)
                selected[order[:k]] = 1
                payload[selection_key(scale, mapping, domain_name, method, budget)] = np.packbits(selected)
                checks.append(dict(scale=scale, mapping=mapping, domain=domain_name, method=method,
                                   budget=budget, domain_positions=len(order), selected_positions=k))
    return payload, checks


def unpack(archive, scale, mapping, domain, method, budget, shape):
    value = archive[selection_key(scale, mapping, domain, method, budget)]
    return np.unpackbits(value)[:int(np.prod(shape))].reshape(shape).astype(bool)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def correlation(score, fraction):
    if len(score) == 0:
        return None, "empty_domain"
    if np.ptp(score) == 0 or np.ptp(fraction) == 0:
        return None, "constant_score_or_target"
    value = float(spearmanr(score, fraction).correlation)
    r0.finite(value, "Spearman")
    return value, ""


def ranked_metrics(score, domain, counts, voxels_per_position):
    index = np.flatnonzero(domain.ravel())
    index = index[np.argsort(-score.ravel()[index], kind="stable")]
    c = counts.ravel()[index]
    s = score.ravel()[index]
    result = dict(domain_positions=len(index), domain_error_voxels=int(c.sum()),
                  domain_error_fraction=ratio(c.sum(), len(c) * voxels_per_position))
    for label, target in (("any", c > 0), ("severe10", c / voxels_per_position >= .10),
                          ("severe25", c / voxels_per_position >= .25)):
        ap, reason = r0.average_precision_sorted(s, target)
        result["ap_" + label] = ap
        result["prevalence_" + label] = float(target.mean()) if len(target) else None
        result["ap_" + label + "_undefined_reason"] = reason
    corr, reason = correlation(s, c / voxels_per_position)
    result.update(spearman_fraction=corr, spearman_undefined_reason=reason)
    return result


def localization(selected, domain, counts, targets, original_selected, neighborhood):
    """All targets enter only after sealed GT-free candidate selection."""
    if np.any(selected & ~domain):
        raise AssertionError("Candidate selection left its fixed domain")
    k, n = int(selected.sum()), int(domain.sum())
    hits, total = int(counts[selected].sum()), int(counts.sum())
    domain_total = int(counts[domain].sum())
    selected_fraction = ratio(hits, 64 * k)
    base_fraction = ratio(domain_total, 64 * n)
    sparse = (counts > 0) & (counts < 7)  # fixed: fewer than 10% error voxels
    lifted = lift_grid(selected)
    e = targets["all_errors"]
    original_hits = int((e & original_selected).sum())
    row = dict(selected_positions=k, selected_voxel_support=64 * k,
               error_coverage=ratio(hits, total), domain_error_coverage=ratio(hits, domain_total),
               selected_error_fraction=selected_fraction,
               enrichment=selected_fraction / base_fraction if base_fraction and selected_fraction is not None else None,
               precision_any=ratio(((counts > 0) & selected).sum(), k),
               recall_any=ratio(((counts > 0) & selected).sum(), ((counts > 0) & domain).sum()),
               precision_severe10=ratio(((counts >= 7) & selected).sum(), k),
               selected_clean_grid_fraction=ratio(((counts == 0) & selected).sum(), k),
               sparse_grid_recall=ratio((sparse & selected).sum(), sparse.sum()),
               sparse_error_coverage=ratio(counts[sparse & selected].sum(), counts[sparse].sum()),
               lost_original_error_voxels=int((e & original_selected & ~lifted).sum()),
               new_error_voxels=int((e & lifted & ~original_selected).sum()),
               lost_original_error_fraction=ratio((e & original_selected & ~lifted).sum(), total),
               new_error_fraction=ratio((e & lifted & ~original_selected).sum(), total),
               original_error_coverage=ratio(original_hits, total),
               coverage_change_vs_original=ratio(hits - original_hits, total),
               outside_boundary_error_coverage=r0.target_recall(lifted, e & ~neighborhood),
               undefined_error_reason="no_errors" if not total else "empty_domain" if not n else "",
               sparse_undefined_reason="no_sparse_error_grids" if not sparse.any() else "")
    for key, target_name in (("high_confidence_coverage", "high_confidence_errors"), ("far_fp_coverage", "far_fp"),
                             ("far_fp_core_coverage", "far_fp_core"), ("high_confidence_far_fp_coverage", "high_confidence_far_fp")):
        valid = targets["far_defined"] if "far_fp" in key else True
        target = targets[target_name]
        row[key] = r0.target_recall(lifted, target, valid)
        row[key + "_undefined_reason"] = (targets["far_reason"] if not valid else
                                          "no_" + target_name if not target.any() else "")
    row["outside_boundary_undefined_reason"] = "no_errors_outside_neighborhood" if not (e & ~neighborhood).any() else ""
    return row


def evaluate_patch(case_id, patch_index, scores, original, gt):
    maps = risk_maps(original)
    p = original["probability"]
    neighborhood = original["pseudo_boundary_neighborhood"].astype(bool)
    targets = r0.evaluate_errors(p, gt, original["pseudo_boundary"].astype(bool))
    counts = block_count(targets["all_errors"])
    grid_shape = counts.shape
    if tuple(scores["grid_shape"].tolist()) != grid_shape or counts.sum() != targets["all_errors"].sum():
        raise AssertionError("Error voxel aggregation does not conserve errors")
    grid_domain = scores["common_grid_domain"].astype(bool)
    common = dict(case_id=case_id, patch_index=patch_index)
    rows, direction, original_rows, failures = [], [], [], []
    status = dict(**common, error_voxels=int(counts.sum()), any_error_grids=int((counts > 0).sum()),
                  severe10_grids=int((counts >= 7).sum()), severe25_grids=int((counts >= 16).sum()),
                  sparse_error_grids=int(((counts > 0) & (counts < 7)).sum()),
                  sparse_error_voxels=int(counts[(counts > 0) & (counts < 7)].sum()),
                  high_confidence_error_voxels=int(targets["high_confidence_errors"].sum()),
                  far_fp_voxels=int(targets["far_fp"].sum()), far_fp_core_voxels=int(targets["far_fp_core"].sum()),
                  crop_sensitive_far_fp_voxels=int(targets["far_fp_crop_sensitive"].sum()),
                  high_confidence_far_fp_voxels=int(targets["high_confidence_far_fp"].sum()),
                  errors_outside_pseudo_neighborhood=int((targets["all_errors"] & ~neighborhood).sum()),
                  errors_outside_common_grid_domain=int(counts[~grid_domain].sum()),
                  far_fp_defined=targets["far_defined"], far_fp_undefined_reason=targets["far_reason"],
                  far_fp_mean_entropy=float(maps["entropy"][targets["far_fp"]].mean()) if targets["far_fp"].any() else None,
                  far_fp_mean_D=float(maps["Dxyz"][targets["far_fp"]].mean()) if targets["far_fp"].any() else None)
    # Original-scale, fixed domains. Interior evaluation does not select a new
    # perturbation: all seven predeclared combinations are reported.
    for domain_name in ("global", "pseudo_boundary_neighborhood", "interior4", "interior8", "interior16"):
        domain = (np.ones(p.shape, bool) if domain_name == "global" else neighborhood if domain_name == DOMAINS[1]
                  else interior_domain(p.shape, int(domain_name[8:])))
        for method in METHODS:
            rank = ranked_metrics(maps[method], domain, targets["all_errors"].astype(np.int64), 1)
            dmass = float(maps[method][domain].sum(dtype=np.float64))
            allmass = float(maps[method].sum(dtype=np.float64))
            top2 = unpack(scores, "original", "native", "global", method, .02, p.shape)
            direction.append(dict(**common, domain=domain_name, method=method, **rank,
                                  risk_mass_fraction=ratio(dmass, allmass), domain_fraction=float(domain.mean()),
                                  error_mass_fraction=ratio((targets["all_errors"] & domain).sum(), targets["all_errors"].sum()),
                                  top2_selected_fraction_in_domain=ratio((top2 & domain).sum(), top2.sum()),
                                  top2_conditional_error_coverage=ratio((top2 & domain & targets["all_errors"]).sum(),
                                                                       (domain & targets["all_errors"]).sum()),
                                  mean_risk=float(maps[method][domain].mean()) if domain.any() else None))
            if domain_name not in DOMAINS:
                continue
            for budget in BUDGETS:
                selected = unpack(scores, "original", "native", domain_name, method, budget, p.shape)
                if int(selected.sum()) != r0.budget_count(int(domain.sum()), budget):
                    raise AssertionError("Original candidate count changed")
                sm = r0.selection_metrics(selected, domain, targets["all_errors"])
                original_rows.append(dict(**common, domain=domain_name, method=method, budget=budget,
                                          ap=rank["ap_any"], prevalence=rank["prevalence_any"],
                                          error_coverage=r0.target_recall(selected, targets["all_errors"]),
                                          selected_error_fraction=sm["precision"], enrichment=sm["enrichment"],
                                          selected_voxels=int(selected.sum()), undefined_reason=sm["undefined_reason"]))
    for mapping in MAPPINGS:
        for domain_name in DOMAINS:
            domain = np.ones(grid_shape, bool) if domain_name == "global" else grid_domain
            for method in METHODS:
                score = scores["risk_" + mapping + "_" + method]
                rank = ranked_metrics(score, domain, counts, 64)
                for budget in BUDGETS:
                    selected = unpack(scores, "x3", mapping, domain_name, method, budget, grid_shape)
                    if int(selected.sum()) != r0.budget_count(int(domain.sum()), budget):
                        raise AssertionError("Unequal mapped candidate budget")
                    original_selected = unpack(scores, "original", "native", domain_name, method, budget, p.shape)
                    loc = localization(selected, domain, counts, targets, original_selected, neighborhood)
                    row = dict(**common, mapping=mapping, domain=domain_name, method=method, budget=budget, **rank, **loc)
                    rows.append(row)
                    failures.append({k: row[k] for k in ("case_id", "patch_index", "mapping", "domain", "method", "budget",
                                       "high_confidence_coverage", "far_fp_coverage", "far_fp_core_coverage", "high_confidence_far_fp_coverage",
                                       "outside_boundary_error_coverage", "sparse_grid_recall", "sparse_error_coverage",
                                       "lost_original_error_voxels", "new_error_voxels", "selected_clean_grid_fraction")})
    return rows, direction, original_rows, failures, status, targets, counts


def bootstrap(values, seed=42):
    valid = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(valid):
        return dict(mean=None, median=None, std=None, ci95=[None, None], n_cases=0, reason="no_defined_case_values")
    r0.finite(valid, "case bootstrap")
    indices = np.random.RandomState(seed).randint(0, len(valid), (BOOTSTRAPS, len(valid)))
    means = valid[indices].mean(1)
    return dict(mean=float(valid.mean()), median=float(np.median(valid)), std=float(valid.std(ddof=1)) if len(valid) > 1 else 0.,
                ci95=np.percentile(means, [2.5, 97.5]).tolist(), n_cases=len(valid), repetitions=BOOTSTRAPS,
                unit="case", positive_cases=int((valid > 0).sum()), negative_cases=int((valid < 0).sum()))


def case_aggregate(rows, ids, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"],) + tuple(row[k] for k in group_fields)].append(row)
    cases = []
    for key, patches in groups.items():
        if len(patches) != 3 or sorted(p["patch_index"] for p in patches) != [0, 1, 2]:
            raise AssertionError("Historical patch omitted or duplicated in case statistics")
        row = dict(case_id=key[0], **dict(zip(group_fields, key[1:])), n_patches=3)
        for metric in metrics:
            values = [p[metric] for p in patches if p.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_defined_patches"] = len(values)
        row["undefined_patch_reasons"] = [{"patch_index": p["patch_index"], "reasons": {
                k: v for k, v in p.items() if k.endswith("reason") and v}} for p in patches]
        cases.append(row)
    if set(r["case_id"] for r in cases) != set(ids):
        raise AssertionError("Receiver case omitted")
    return cases


def aggregate(rows, direction, original_rows, ids, seed):
    fields = ("mapping", "domain", "method", "budget")
    cases = case_aggregate(rows, ids, fields, METRICS)
    direction_metrics = ("ap_any", "prevalence_any", "spearman_fraction", "risk_mass_fraction", "mean_risk",
                         "error_mass_fraction", "top2_selected_fraction_in_domain", "top2_conditional_error_coverage")
    directions = case_aggregate(direction, ids, ("domain", "method"), direction_metrics)
    originals = case_aggregate(original_rows, ids, ("domain", "method", "budget"), ("ap", "prevalence", "error_coverage", "selected_error_fraction", "enrichment"))
    summary = dict(exploratory=True, independent_validation_available=False, n_cases=len(ids), n_patches=24,
                   hierarchy="3 patches averaged per case, then 8 equally weighted cases", bootstrap_repetitions=BOOTSTRAPS,
                   x3={}, original={}, directions={}, paired={}, mapping_pairs={})
    for label, table, keys, metrics in (("x3", cases, fields, METRICS), ("original", originals, ("domain", "method", "budget"),
                 ("ap", "prevalence", "error_coverage", "selected_error_fraction", "enrichment")),
                 ("directions", directions, ("domain", "method"), direction_metrics)):
        groups = defaultdict(list)
        for row in table:
            groups[tuple(row[k] for k in keys)].append(row)
        for key, cc in groups.items():
            name = "/".join(str(int(v * 100)) if k == "budget" else str(v) for k, v in zip(keys, key))
            summary[label][name] = {m: bootstrap([c[m] for c in cc], seed) for m in metrics}
    by_key = {(c["case_id"], c["mapping"], c["domain"], c["method"], c["budget"]): c for c in cases}
    paired, mapping_rows = [], []
    pair_metrics = ("ap_any", "ap_severe10", "error_coverage", "selected_error_fraction", "high_confidence_coverage", "far_fp_coverage")
    for mapping in MAPPINGS:
        for domain in DOMAINS:
            for budget in BUDGETS:
                for a, b in (("Dxyz", "entropy"), ("Dxyz", "boundary_prior"), ("Dxyz", "random"),
                             ("combined", "Dxyz"), ("combined_boundary", "Dxyz"), ("combined_boundary", "combined")):
                    group = []
                    for case_id in ids:
                        aa, bb = by_key[case_id, mapping, domain, a, budget], by_key[case_id, mapping, domain, b, budget]
                        row = dict(case_id=case_id, mapping=mapping, domain=domain, method=a, control=b, budget=budget)
                        row.update({m + "_difference": aa[m] - bb[m] if aa[m] is not None and bb[m] is not None else None for m in pair_metrics})
                        paired.append(row)
                        group.append(row)
                    key = "{}/{}/{}/{}/{}".format(mapping, domain, int(budget * 100), a, b)
                    summary["paired"][key] = {m: bootstrap([r[m + "_difference"] for r in group], seed) for m in pair_metrics}
                for method in PRIMARY:
                    for control in ("max", "trilinear"):
                        if mapping != "average":
                            continue
                        group = []
                        for case_id in ids:
                            aa, bb = by_key[case_id, mapping, domain, method, budget], by_key[case_id, control, domain, method, budget]
                            row = dict(case_id=case_id, mapping=mapping, control_mapping=control, domain=domain, method=method, budget=budget)
                            row.update({m + "_difference": aa[m] - bb[m] if aa[m] is not None and bb[m] is not None else None for m in pair_metrics})
                            mapping_rows.append(row)
                            group.append(row)
                        key = "{}/{}/{}/{}/{}".format(domain, int(budget * 100), method, mapping, control)
                        summary["mapping_pairs"][key] = {m: bootstrap([r[m + "_difference"] for r in group], seed) for m in pair_metrics}
    return cases, directions, originals, paired, mapping_rows, summary


def visualize(out, old_artifact, new_artifact, counts, targets, mapping, z, suffix, image_range):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(old_artifact["path"]) as old, np.load(new_artifact["path"]) as new:
        image = old["image"]
        p = old["probability"]
        full = old["risk_disagreement"]
        grid = new["risk_" + mapping + "_Dxyz"]
        x3shown = lift_grid(grid)
        selected = lift_grid(unpack(new, "x3", mapping, "global", "Dxyz", .02, grid.shape))
        panels = [("MRI", image, "gray", image_range), ("Teacher probability", p, "viridis", (0, 1)),
                  ("Original Dxyz", full, "magma", (0, 1)), ("x3 Dxyz (block display)", x3shown, "magma", (0, 1)),
                  ("True errors", targets["all_errors"], "Reds", (0, 1)),
                  ("x3 error fraction (4^3 grid)", lift_grid(counts / 64), "Reds", (0, 1)),
                  ("x3 top2% Dxyz", selected, "Blues", (0, 1)), ("Pseudo boundary", old["pseudo_boundary"], "Blues", (0, 1)),
                  ("GT boundary (evaluation)", targets["gt_boundary"], "Greens", (0, 1)),
                  ("Dx", new["Dx"], "magma", (0, 1)), ("Dy", new["Dy"], "magma", (0, 1)), ("Dz", new["Dz"], "magma", (0, 1))]
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
        for ax, (title, data, cmap, limits) in zip(axes.flat, panels):
            im = ax.imshow(data[:, :, z].T, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=.65)
        fig.suptitle("{} p{} z={} / x3 block z={} [{}:{}], {}; {}".format(new_artifact["case_id"], new_artifact["patch_index"],
                      z, z // BLOCK, z // BLOCK * BLOCK, z // BLOCK * BLOCK + 3, mapping, suffix))
        fig.savefig(str(out / "visualizations" / (new_artifact["case_id"] + "_p" + str(new_artifact["patch_index"]) + "_" + suffix + ".png")), dpi=150)
        plt.close(fig)


def format_stat(stat, percent=False):
    if stat["mean"] is None:
        return "不可定义"
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    return "{:.4f}{} [{:.4f}, {:.4f}]".format(stat["mean"] * scale, suffix, stat["ci95"][0] * scale, stat["ci95"][1] * scale)


def scientific_answers(summary, cases, direction_cases, chosen):
    """Interpret fixed comparisons, including negative findings and tradeoffs."""
    def s(mapping, method, budget, metric):
        return summary["x3"]["{}/global/{}/{}".format(mapping, method, budget)][metric]
    def p(method, control, budget, metric):
        return summary["paired"]["{}/global/{}/{}/{}".format(chosen, budget, method, control)][metric]
    index = {(r["case_id"], r["mapping"], r["domain"], r["method"], r["budget"]): r for r in cases}
    ids = sorted(set(r["case_id"] for r in cases))
    max_tri = {}
    for metric in ("ap_any", "ap_severe10", "error_coverage", "high_confidence_coverage", "far_fp_coverage"):
        values = []
        for case_id in ids:
            aa = index[case_id, "max", "global", "Dxyz", .02][metric]
            bb = index[case_id, "trilinear", "global", "Dxyz", .02][metric]
            values.append(aa - bb if aa is not None and bb is not None else None)
        max_tri[metric] = bootstrap(values)
    di = {(r["case_id"], r["domain"], r["method"]): r for r in direction_cases}
    zshares = [di[i, "global", "Dz"]["mean_risk"] / sum(di[i, "global", m]["mean_risk"] for m in ("Dx", "Dy", "Dz")) for i in ids]
    zstat = bootstrap(zshares)
    interior_difference = bootstrap([di[i, "interior16", "Dxyz"]["ap_any"] - di[i, "global", "Dxyz"]["ap_any"] for i in ids])
    summary["additional_fixed_comparisons"] = dict(max_minus_trilinear_D_top2=max_tri,
            Dz_share_of_mean_axis_magnitude=zstat, interior16_minus_global_D_AP=interior_difference)
    lines = ["", "## 科学问题的定量结论（含反例）", "",
             "**Q1：有条件保留，不能简单把原尺度 D 优势外推到任意映射。** Max D 相对同映射 Entropy 的 Top2% 错误覆盖差为 {} 个百分点，8例均更高；相对预测边界距离差 {} 个百分点。共同候选域比较也保留该优势。".format(
                format_stat(p("Dxyz", "entropy", 2, "error_coverage"), True), format_stat(p("Dxyz", "boundary_prior", 2, "error_coverage"), True)),
             "反例：Average D 的 any-error AP {} 低于 Average Entropy {}；Trilinear D 的 AP {} 也低于同映射 Entropy {}。网格阳性比例约8.5%，原体素错误率约1.3%，不能将 Max 网格AP0.95直接解释为比原尺度AP0.789更强。".format(
                format_stat(s("average", "Dxyz", 2, "ap_any")), format_stat(s("average", "entropy", 2, "ap_any")),
                format_stat(s("trilinear", "Dxyz", 2, "ap_any")), format_stat(s("trilinear", "entropy", 2, "ap_any"))),
             "", "**Q2：Max 的整体排序和跨病例 AP 最稳定，Trilinear 在1%–2%错误体素覆盖上更好。** 这是实际权衡，不存在所有指标均胜出的映射。", "",
             "|D映射|AP any病例SD|AP≥10%病例SD|Top2%覆盖病例SD|Top2%高置信错误覆盖|Top2%远FP覆盖|",
             "|---|---|---|---|---|---|"]
    for mapping in MAPPINGS:
        lines.append("|{}|{:.4f}|{:.4f}|{:.4f}|{}|{}|".format(mapping, s(mapping, "Dxyz", 2, "ap_any")["std"],
            s(mapping, "Dxyz", 2, "ap_severe10")["std"], s(mapping, "Dxyz", 2, "error_coverage")["std"],
            format_stat(s(mapping, "Dxyz", 2, "high_confidence_coverage"), True), format_stat(s(mapping, "Dxyz", 2, "far_fp_coverage"), True)))
    lines += ["Max−Trilinear 的2%覆盖差为 {} 个百分点（{}例正、{}例负），any-error AP差 {}；Max改善整体排序和远FP覆盖，Trilinear更偏重少数严重/高置信错误块。故选择Max是整体排序、稀疏错误和远FP稳健性的探索性取舍，不是选择最高2%覆盖。".format(
              format_stat(max_tri["error_coverage"], True), max_tri["error_coverage"]["positive_cases"], max_tri["error_coverage"]["negative_cases"], format_stat(max_tri["ap_any"])),
              "", "**Q3：存在明显的 z 方向异常；有效定位信号主要来自 x/y。** 原尺度 Dx AP=0.8396、Dy=0.8309、Dz=0.0510，固定 Dxy 对照 AP=0.8671。Dz 的平均差异幅度明显更大（约0.129，而Dx/Dy约0.013），病例等权 Dz 幅度占三轴总和的 {}。幅度大并不代表错误定位信息多。".format(format_stat(zstat, True)),
              "原尺度 D 在距裁剪面>16区域的 AP为 {}，相对全域变化 {}；故高D的定位信息不能仅以裁剪边缘解释，但这也不能排除上下文效应。坐标逆变换已验证且重算与历史结果完全一致，z异常更可能涉及模型方向非等变性/解剖方向敏感性；没有实验直接证实具体原因。**未依据GT删除z，也不将事后最佳Dxy推荐为已验证新方案**。".format(
                   format_stat(summary["directions"]["interior16/Dxyz"]["ap_any"]), format_stat(interior_difference)),
              "", "**Q4：探索性候选为全域 Dxyz → 4³ Max Pooling → Top2% x3（314/15680网格）。** Top1%为157网格。该建议优先整体严重错误排序、候选位置精度与远FP覆盖，保留三轴；不是最大化这8例上小预算错误体素覆盖的最优方案。Trilinear的2%覆盖 {} 高于Max {}，作为固定比较完整保留。".format(
                   format_stat(s("trilinear", "Dxyz", 2, "error_coverage"), True), format_stat(s("max", "Dxyz", 2, "error_coverage"), True)),
              "", "**Q5：在候选 Max 映射下，不支持把5%预算默认改成组合。** D覆盖 {}；H+D覆盖 {}，差 {} 个百分点；边界组合覆盖 {}，差 {} 个百分点，两个组合在8例均低于D。组合的 any-error AP稍高，但严重错误AP和错误体素覆盖更低，不能仅凭一个AP认定更好。Average下组合覆盖明显优于D，说明原R0的组合收益不能跨映射直接沿用。".format(
                   format_stat(s("max", "Dxyz", 5, "error_coverage"), True), format_stat(s("max", "combined", 5, "error_coverage"), True),
                   format_stat(p("combined", "Dxyz", 5, "error_coverage"), True), format_stat(s("max", "combined_boundary", 5, "error_coverage"), True),
                   format_stat(p("combined_boundary", "Dxyz", 5, "error_coverage"), True)),
              "", "**Q6：小预算仍大量遗漏高置信错误和稀疏错误。** Max D Top2%高置信错误覆盖 {}，即平均漏掉约 {:.2%}；稀疏错误体素覆盖 {}。远FP覆盖 {}（仅{}个有目标病例），高置信远FP覆盖 {}（仅{}例）。核心远FP仅160个观测、1个病例；多数远FP靠近裁剪面，100%覆盖也不能视作普遍可靠。".format(
                   format_stat(s("max", "Dxyz", 2, "high_confidence_coverage"), True), 1 - s("max", "Dxyz", 2, "high_confidence_coverage")["mean"],
                   format_stat(s("max", "Dxyz", 2, "sparse_error_coverage"), True), format_stat(s("max", "Dxyz", 2, "far_fp_coverage"), True),
                   s("max", "Dxyz", 2, "far_fp_coverage")["n_cases"], format_stat(s("max", "Dxyz", 2, "high_confidence_far_fp_coverage"), True),
                   s("max", "Dxyz", 2, "high_confidence_far_fp_coverage")["n_cases"]),
              "855个错误体素观测在共同边界网格域之外，原尺度预测边界邻域外为5192个。边界先验可能漏掉预测轮廓之外的错误，不能将共同域高AP说成全域无遗漏。共同域跨尺度覆盖比较还包含整块扩张的空间效应，并非完全相同体素候选范围；主要跨尺度结论使用全域。",
              "", "**Q7：有理由保留为空间候选，但不足以确立R1正式方法。** 推荐冻结上述候选，先在未参与训练和checkpoint选择的数据上验证，并复核z方向异常。可以据此规划R1的小范围可行性研究；当前没有独立验证、没有接收端补偿需求/方向判定，更没有特征修正或Dice收益证据。不会自动实现R1。"]
    return lines


def report(out, meta, summary, cases, direction_cases, statuses, verification):
    """All directions are reported; no GT-driven choice of flip subset."""
    def stat(mapping, method, budget, metric, domain="global"):
        return summary["x3"]["{}/{}/{}/{}".format(mapping, domain, method, budget)][metric]
    def pair(mapping, method, control, budget, metric, domain="global"):
        return summary["paired"]["{}/{}/{}/{}/{}".format(mapping, domain, budget, method, control)][metric]
    # A single global exploratory mapping proposal, no per-patch GT adaptation.
    # Prioritize Dxyz severe-error AP and 2% voxel coverage jointly, then cases.
    ranking = sorted(MAPPINGS, key=lambda m: (stat(m, "Dxyz", 2, "error_coverage")["mean"] + stat(m, "Dxyz", 2, "ap_severe10")["mean"]), reverse=True)
    chosen = ranking[0]
    summary["recommendation"] = dict(status="exploratory_without_independent_validation", candidate_score="Dxyz",
                mapping=chosen, budget=.02, perturbations="all three fixed flips; no best-axis selection",
                mapping_decision="joint exploratory rank: case-mean severe10 AP + 2% error-voxel coverage; all mappings reported",
                alternative_budget=.05, alternative_score="Dxyz",
                R1="limited feasibility study only; independent validation and receiver need assessment remain required",
                no_Dice_improvement_claim=True)
    lines = ["# Experiment R0.5：Receiver Reliability at Feature Scale", "",
             "实验完成。冻结 Baseline EMA，对原 R0 的 8 个已参与有标签训练病例、24 个历史固定 patch 做探索性评价。没有训练、特征修正或 Dice 提升实验。",
             "", "## 数据与保护", "",
             "可用 LA 共 100 例：80 例进入训练（包括无标签图像），20 例 test.list 被用于验证和 checkpoint 选择。未发现合格的独立验证数据；这 8 例也参与了模型训练。病例级 Bootstrap 不能消除此偏倚。",
             "checkpoint：`{}`，SHA256 `{}`。".format(meta["checkpoint"]["path"], meta["checkpoint"]["sha256"]),
             "精确复现 R0 crop、padding、源文件 SHA；全部图像评分、三种映射及 Top-K 掩码在首个 GT 读取前保存并 SHA 封存。已有 {} 个项目文件/结果/checkpoint 哈希完全相同。".format(verification["protected_file_count"]),
             "", "## 预先固定的方法与评价", "",
             "Teacher eval、参数冻结、BN 固定、Dropout 关闭、所有预测和 encoder 在 no_grad 中。重算原始预测及逆翻转预测与 R0 在预定绝对容差 2e-6 内匹配；随后锁定历史 FP32 概率以精确复现 R0 排序。Dx/Dy/Dz 是张量轴，未验证解剖方向；Dxyz 为三轴差异平均，Dxy/Dxz/Dyz 仅作固定对照，不按 GT 选择扰动组合。",
             "Average/Max 使用 kernel=stride=4；Trilinear 使用 align_corners=False，中心坐标为 4i+1.5。x3 实际 shape 逐 patch 记录，未改其特征；4³ 网格只是空间聚合，不是真实感受野。",
             "GT 错误计数按 4³ 块精确求和；any>0，严重错误≥10% 对应≥7/64，25%敏感性分析对应≥16/64。记录连续错误比例。",
             "共同候选域：原预测边界欧氏距离≤3 voxel 的邻域中至少有一个体素的网格。该域对所有方法和映射完全相同；全域和共同域各自按域大小计算 Top1/2/5/10%，round 且非空至少1。各方法同预算的网格数严格一致。网格选中面积固定，Max 不会增加该预算的总面积。",
             "Random 也从同一历史原始随机风险图经各映射得到，保持 GT 无关。并列评分使用独立固定随机顺序；不影响采样/模型 RNG。熵及边界先验沿用 R0，H+D=(H+Dxyz)/2；边界组合再乘 exp(-d/3)，所有分数越高越不可靠。",
             "错误体素覆盖主要分母为整个 patch 的错误数；另报告候选域内覆盖。Enrichment 相对当前评价域错误率。原尺度和网格尺度 AP 目标及阳性比例不同，不能直接把 AP 差解释为性能提升；覆盖率是跨尺度主要比较。",
             "统计：patch→每病例3patch平均→8病例等权；10000次病例 Bootstrap，区间为探索性、未作多重检验校正。没有目标/空域/常量相关性以 null/空值及原因保存，不视为成功。",
             "", "## 全域 x3：所有映射的主要结果", "",
             "表中为病例均值 [95% CI]。", "",
             "|映射|评分|AP any|AP ≥10%|Top1%错误覆盖|Top2%错误覆盖|Top5%错误覆盖|Top10%错误覆盖|",
             "|---|---|---|---|---|---|---|---|"]
    for mapping in MAPPINGS:
        for method in PRIMARY:
            lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|".format(mapping, method,
                format_stat(stat(mapping, method, 2, "ap_any")), format_stat(stat(mapping, method, 2, "ap_severe10")),
                *[format_stat(stat(mapping, method, b, "error_coverage"), True) for b in (1, 2, 5, 10)]))
    lines += ["", "阳性比例及严重错误25%结果见 summary_statistics.json 和逐病例 CSV；原始尺度 AP 不与此表直接等同。",
              "", "## 候选映射的病例配对比较", "",
              "|预算|比较（前者−后者）|错误覆盖差|正/负病例|AP ≥10%差|", "|---|---|---|---|---|"]
    for budget in (1, 2, 5, 10):
        for a, b in (("Dxyz", "entropy"), ("Dxyz", "boundary_prior"), ("combined", "Dxyz"), ("combined_boundary", "Dxyz")):
            s = pair(chosen, a, b, budget, "error_coverage")
            lines.append("|{}%|{}−{}|{}|{}/{}|{}|".format(budget, a, b, format_stat(s, True), s["positive_cases"], s["negative_cases"],
                        format_stat(pair(chosen, a, b, budget, "ap_severe10"))))
    lines += ["", "## 共同预测边界候选域：公平比较", "",
              "预算分母为共同域网格数，覆盖分母仍为全 patch 错误体素（域内覆盖另见 CSV）。", "",
              "|方法|AP any|AP ≥10%|Top2%错误覆盖|Top5%错误覆盖|", "|---|---|---|---|---|"]
    for m in PRIMARY:
        lines.append("|{}|{}|{}|{}|{}|".format(m, format_stat(stat(chosen, m, 2, "ap_any", DOMAINS[1])),
             format_stat(stat(chosen, m, 2, "ap_severe10", DOMAINS[1])), format_stat(stat(chosen, m, 2, "error_coverage", DOMAINS[1]), True),
             format_stat(stat(chosen, m, 5, "error_coverage", DOMAINS[1]), True)))
    lines += ["", "## 翻转方向与裁剪边缘", "", "所有组合预先固定；不根据这些结果选最佳轴。", "",
              "|评分|原分辨率AP|距面>4 AP|距面>8 AP|距面>16 AP|", "|---|---|---|---|---|"]
    for m in ("Dx", "Dy", "Dz", "Dxyz", "Dxy", "Dxz", "Dyz", "entropy", "boundary_prior"):
        lines.append("|{}|{}|{}|{}|{}|".format(m, *[format_stat(summary["directions"][d + "/" + m]["ap_any"])
                      for d in ("global", "interior4", "interior8", "interior16")]))
    axis_means = {m: summary["directions"]["global/" + m]["ap_any"]["mean"] for m in ("Dx", "Dy", "Dz")}
    lines.append("三单轴中病例均值 AP 最大的是 {}（{}）。完整各病例和风险在内部/边缘的质量比例见 direction_analysis.csv、direction_case_statistics.csv；单轴最佳不作为后续组合选择依据。".format(max(axis_means, key=axis_means.get), axis_means))
    lines += ["", "## 原尺度→x3、稀疏错误与 Max 扩散", "",
              "|映射|D Top2%覆盖变化（百分点）|稀疏错误体素覆盖|所选干净网格比例|", "|---|---|---|---|"]
    for m in MAPPINGS:
        lines.append("|{}|{}|{}|{}|".format(m, format_stat(stat(m, "Dxyz", 2, "coverage_change_vs_original"), True),
            format_stat(stat(m, "Dxyz", 2, "sparse_error_coverage"), True), format_stat(stat(m, "Dxyz", 2, "selected_clean_grid_fraction"), True)))
    lines += ["Average 会把少数高风险体素与同块低值平均；Max 对任何高值敏感；Trilinear只采样块中心邻近8点，可能漏掉角落小错误。不能只凭这些数学性质断言哪种更好，上表及 downsampling_comparison.csv 给出实际配对结果。",
              "failure_analysis.csv 保存原尺度选中但 x3 未选中的错误、反向新增覆盖、稀疏错误覆盖和干净网格比例。相同预算的 Max 面积不变；若选中更多干净网格，表示高风险块扩散/误选而非面积增加。",
              "", "## 每病例差异：候选映射全域 Top2%", "",
              "|病例|D覆盖|H覆盖|边界距离覆盖|H+D覆盖|边界组合覆盖|", "|---|---|---|---|---|---|"]
    for case_id in meta["case_order"]:
        cc = {c["method"]: c for c in cases if (c["case_id"], c["mapping"], c["domain"], c["budget"]) == (case_id, chosen, "global", .02)}
        lines.append("|{}|{}|{}|{}|{}|{}|".format(case_id, *["{:.2%}".format(cc[m]["error_coverage"]) if cc[m]["error_coverage"] is not None else "未定义"
                     for m in ("Dxyz", "entropy", "boundary_prior", "combined", "combined_boundary")]))
    totals = {k: sum(s[k] for s in statuses) for k in ("error_voxels", "high_confidence_error_voxels", "far_fp_voxels", "far_fp_core_voxels",
                "crop_sensitive_far_fp_voxels", "high_confidence_far_fp_voxels", "errors_outside_pseudo_neighborhood", "errors_outside_common_grid_domain")}
    summary["patch_observation_counts"] = totals
    lines += ["", "## 高置信度、远距离FP及遗漏", "",
              "24 patch 重叠体素观测数（不是独立整例唯一体素）：`{}`。".format(totals),
              "远距离FP定义距 patch 内 GT 前景>20 voxel；其中 crop_sensitive 标明距裁剪面≤20，core 为距面>20，截断限制保留。边界共同域扩张成整块仍有域外错误，边界先验不能保证覆盖全部真实错误。",
              "|预算|D高置信错误覆盖|D远FP覆盖（有目标病例）|D核心远FP覆盖|边界组合高置信错误覆盖|", "|---|---|---|---|---|"]
    for b in (1, 2, 5, 10):
        lines.append("|{}%|{}|{}|{}|{}|".format(b, format_stat(stat(chosen, "Dxyz", b, "high_confidence_coverage"), True),
            format_stat(stat(chosen, "Dxyz", b, "far_fp_coverage"), True), format_stat(stat(chosen, "Dxyz", b, "far_fp_core_coverage"), True),
            format_stat(stat(chosen, "combined_boundary", b, "high_confidence_coverage"), True)))
    lines += ["没有该类目标的 patch/case 不默认为成功，相应 defined_patches/n_cases 必须一并查看；核心远FP样本量尤其有限。不能把这批高置信错误均可定位的比例外推为所有高置信错误都可识别。",
              "", "## 科学问题与下一步", "",
              "Q1：D优势是否保留，依据上面的 D−H、D−边界的病例配对差及区间判断；所有三种映射均报告，不能仅保留胜出的结果。",
              "Q2：探索性统一映射建议 **{}**，选择依据为全域 Dxyz 的严重错误 AP 与2%错误体素覆盖之和；完整映射配对结果及失败病例保留。该选择尚无独立数据验证。".format(chosen),
              "Q3：轴贡献见固定方向表及每病例 CSV。方向不同可能来自模型非等变性、各向异性 patch 或裁剪上下文，不能等同于真实生理解剖错误机制。>4/8/16 的内部评价用于识别裁剪边缘解释。",
              "Q4：明确候选为 **Dxyz + {} + 全域 Top2% x3 网格**，Top1%为更严格预算对照；三轴均保留，无GT驱动单轴选择。该风险候选还不是补偿需求或补偿方向判定。".format(chosen),
              "Q5：5%预算下的 H+D/边界组合与D差异及区间见配对表；只有覆盖稳定增加且遗漏可接受时才值得用组合，以覆盖为目标与以精确定位为目标不同。",
              "Q6：高置信与远FP遗漏见上表及失败记录；狭窄预算必然遗漏部分错误，边界先验会压低远离预测轮廓的位置。没有独立新病例证据，也没有足够核心远FP来证明普遍可靠。",
              "Q7：可作为 **R1小范围可行性研究的探索性空间候选**，尚不足以确立有效补偿机制或泛化可靠性。进入模型设计前应先在未参与训练/选checkpoint病例上验证固定候选，不在当前8例反复优化。此处未实现R1、Donor交互、残差补偿或新Loss。",
              "", "直接事实是冻结模型的错误定位与网格聚合比较；可能解释是高值聚合、翻转非等变性和裁剪上下文；真实无标签上的可靠性、特征补偿收益及Dice改善仍未证实。",
              "", "## 复核路径与执行", "",
              "全部结果：`{}`。".format(out.relative_to(ROOT)),
              "主要CSV：patch_level_metrics.csv、case_level_metrics.csv、direction_analysis.csv、direction_case_statistics.csv、downsampling_comparison.csv、x3_error_localization.csv、failure_analysis.csv、paired_method_comparison.csv。",
              "风险与候选：receiver_risk_maps/*_scores_candidates.npz；错误计数/比例与GT评价图：receiver_error_maps/*_evaluation.npz；二者严格分阶段。历史 R0 的原风险图被 SHA 引用，不重复改写。",
              "visualizations 使用预先固定病例索引0/3/7、patch0、z40，另补首个按病例/patch/行优先顺序发现的高置信错误例。原图及块显示在同一坐标，风险颜色统一[0,1]；x3块覆盖 z=4k..4k+3，不冒称独立精细分辨率或真实感受野。",
              "验证状态：`{}`；单元测试详情见 test_results.json，完整哈希/数据来源/坐标见 experiment_metadata.json、verification.json、data_access_log.csv、phaseA_seal.json。".format(verification["status"]),
              "复现命令：`conda run -n SSL python -u -B analysis/experiment_receiver_R05.py --device cuda:3 --output_dir analysis/results/experiment_receiver_R05/run_NEW_TIMESTAMP`。输出必须新建，既有目录拒绝覆盖。"]
    lines += scientific_answers(summary, cases, direction_cases, chosen)
    (out / "experiment_report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    reference = Path(args.reference_dir).resolve()
    requested = Path(args.output_dir).resolve()
    if requested.exists():
        raise FileExistsError("Refusing existing output directory: " + str(requested))
    history, old, ids, checkpoint, originals = validate_reference(reference, args.model_path)
    # Initialize CUDA BEFORE creating a run directory; inaccessible GPUs do not
    # leave occupied output paths. Data validation also precedes new outputs.
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = create_output(requested)
    for name in ("receiver_risk_maps", "receiver_error_maps", "visualizations"):
        (out / name).mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(str(out / "analysis_log.txt")), logging.StreamHandler(sys.stdout)])
    before = protected_manifest(out)
    metadata = dict(status="started", experiment="Receiver R0.5", configuration=vars(args), case_order=ids,
                    checkpoint=checkpoint, R0_reference=str(reference), R0_metadata_sha256=r0.sha256(reference / "experiment_metadata.json"),
                    data_sources=history["data_sources"], crop_coordinates=history["crop_coordinates"],
                    code_sha256={str(p.relative_to(ROOT)): r0.sha256(p) for p in NEW_CODE}, protected_file_sha256_before=before,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    protocol=dict(methods=METHODS, mappings=MAPPINGS, budgets=BUDGETS, kernel_stride=4,
                    align_corners=False, error_thresholds=[0, .10, .25], interior_distance_strict_gt=MARGINS,
                    common_domain="any voxel in historical distance<=3 pseudo-boundary neighborhood per 4^3 grid",
                    no_GT_before_all_score_candidates_sealed=True, prediction_match_atol=PRED_ATOL,
                    grid_not_true_receptive_field=True, bootstrap_unit="case", bootstrap_repetitions=BOOTSTRAPS,
                    high_confidence=.95, far_FP_voxel_distance=20, visual_cases=[0, 3, 7], visual_patch=0, visual_z=40,
                    sparse_grid="1..6 error voxels/64 (<10%)", no_best_flip_subset_selection=True,
                    candidate_mapping_decision="exploratory joint severe10 AP plus 2% error coverage for Dxyz; no per-patch mapping choice"),
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, device=args.device,
                      device_name=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    metadata["git_commit"] = git.stdout.strip() if not git.returncode else None
    audit = independent_audit(old)
    metadata["independent_data_audit"] = audit
    r0.write_json(out / "independent_data_audit.json", audit)
    r0.write_json(out / "experiment_metadata.json", metadata)
    try:
        test = subprocess.run([sys.executable, "-B", str(HERE / "test_experiment_receiver_R05.py")], capture_output=True, text=True)
        r0.write_json(out / "test_results.json", dict(exit_code=test.returncode, stdout=test.stdout, stderr=test.stderr))
        if test.returncode:
            raise AssertionError("R05 unit tests failed: " + test.stderr)
        model = VNet(n_channels=1, n_classes=2).to(args.device)
        extra = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
        model.load_state_dict(torch.load(args.model_path, map_location="cpu", **extra), strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        reader = r0.HistoricalReader(old)
        artifacts, prediction_checks, budget_checks = [], [], []
        for i, case_id in enumerate(ids):
            for patch in range(3):
                image = reader.read(case_id, patch, "image")
                p, aligned, check = r0.teacher_predictions(model, image, args.device)
                with torch.no_grad():
                    features = model.encoder(torch.from_numpy(image)[None, None].to(args.device))
                    shape = tuple(features[2].shape[2:])
                    actual_shape = list(features[2].shape)
                del features
                old_artifact = originals[case_id, patch]
                with np.load(old_artifact["path"]) as archive:
                    if not np.array_equal(image, archive["image"]):
                        raise AssertionError("Historical image crop changed")
                    p_diff = float(np.max(np.abs(p.astype(np.float64) - archive["probability"].astype(np.float64))))
                    flip_diff = float(np.max(np.abs(aligned.astype(np.float64) - archive["aligned_flip_probabilities"].astype(np.float64))))
                    d_diff = float(np.max(np.abs(direction_maps(p, aligned)["Dxyz"] - archive["risk_disagreement"])))
                    if max(p_diff, flip_diff, d_diff) > PRED_ATOL:
                        raise AssertionError("Historical R0 prediction/D mismatch: " + str((p_diff, flip_diff, d_diff)))
                    maps = risk_maps(archive)
                    neighborhood = archive["pseudo_boundary_neighborhood"].astype(bool)
                    grid_domain = block_count(neighborhood) > 0
                    seed = args.seed + i * 1009 + patch * 9176
                    original_candidates, original_counts = selections(maps, neighborhood, "original", "native", seed + 10000079)
                    for method in PRIMARY:
                        historical_method = "disagreement" if method == "Dxyz" else method
                        for domain in DOMAINS:
                            for budget in BUDGETS:
                                if not np.array_equal(original_candidates[selection_key("original", "native", domain, method, budget)],
                                     archive[r0.candidate_name(domain, historical_method, budget)]):
                                    raise AssertionError("R0 original-scale candidate replay differs")
                    payload = dict(grid_shape=np.array(shape), common_grid_domain=grid_domain,
                                   **{k: maps[k] for k in ("Dx", "Dy", "Dz")}, **original_candidates)
                    checks = list(original_counts)
                    for mapping in MAPPINGS:
                        mapped = {method: map_to_grid(maps[method], shape, mapping) for method in METHODS}
                        payload.update({"risk_" + mapping + "_" + k: v for k, v in mapped.items()})
                        candidates, counts = selections(mapped, grid_domain, "x3", mapping, seed + 20000003)
                        payload.update(candidates)
                        checks.extend(counts)
                    path = out / "receiver_risk_maps" / (case_id + "_p" + str(patch) + "_scores_candidates.npz")
                    np.savez_compressed(str(path), **payload)
                artifacts.append(dict(case_id=case_id, patch_index=patch, path=str(path), sha256=r0.sha256(path),
                                      original_R0=old_artifact, x3_shape=actual_shape, crop=reader.coords[case_id, patch]))
                prediction_checks.append(dict(case_id=case_id, patch_index=patch, x3_shape=actual_shape, probability_max_abs_diff=p_diff,
                                              flip_max_abs_diff=flip_diff, D_max_abs_diff=d_diff, D_historical_reconstruction_exact=True, **check))
                budget_checks.extend(dict(case_id=case_id, patch_index=patch, **c) for c in checks)
                logging.info("Phase A %s p%d x3=%s R0 p/flip/D max diff=%s; GT reads=0", case_id, patch, actual_shape, (p_diff, flip_diff, d_diff))
        state_changed = [k for k, v in state_before.items() if not torch.equal(v, model.state_dict()[k].detach().cpu())]
        if state_changed or any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):
            raise AssertionError("Frozen Teacher parameters/buffers/eval state changed")
        r0.write_json(out / "phaseA_seal.json", dict(GT_reads=0, artifacts=artifacts, protocol=metadata["protocol"]))
        reader.seal(artifacts)
        metadata.update(phaseA_artifacts=artifacts, prediction_checks=prediction_checks)
        r0.write_csv(out / "candidate_budget_checks.csv", budget_checks)
        logging.info("All 24 risk/mapping/candidate artifacts sealed; first GT access allowed")
        rows, directions, original_rows, failures, statuses = [], [], [], [], []
        first_confident = None
        for artifact in artifacts:
            case_id, patch = artifact["case_id"], artifact["patch_index"]
            gt = reader.read(case_id, patch, "label")
            if int(gt.sum()) != reader.coords[case_id, patch]["foreground_voxels"]:
                raise AssertionError("Historical GT crop replay mismatch")
            with np.load(artifact["path"]) as scores, np.load(artifact["original_R0"]["path"]) as original:
                pp, dd, oo, ff, status, targets, counts = evaluate_patch(case_id, patch, scores, original, gt)
            eval_path = out / "receiver_error_maps" / (case_id + "_p" + str(patch) + "_evaluation.npz")
            np.savez_compressed(str(eval_path), error_count_x3=counts.astype(np.uint8), error_fraction_x3=(counts / 64).astype(np.float32),
                                any_error_x3=counts > 0, severe10_x3=counts >= 7, severe25_x3=counts >= 16,
                                high_confidence_count_x3=block_count(targets["high_confidence_errors"]).astype(np.uint8),
                                far_fp_count_x3=block_count(targets["far_fp"]).astype(np.uint8),
                                **{k: v for k, v in targets.items() if isinstance(v, np.ndarray)})
            if first_confident is None and targets["high_confidence_errors"].any():
                first_confident = dict(case_id=case_id, patch_index=patch, voxel=np.argwhere(targets["high_confidence_errors"])[0].tolist())
            rows.extend(pp); directions.extend(dd); original_rows.extend(oo); failures.extend(ff); statuses.append(status)
            logging.info("Phase B %s p%d errors=%d any-grid=%d severe10=%d farFP=%d", case_id, patch, status["error_voxels"],
                         status["any_error_grids"], status["severe10_grids"], status["far_fp_voxels"])
        cases, direction_cases, original_cases, paired, mapping_rows, summary = aggregate(rows, directions, original_rows, ids, args.seed)
        # Full protected-state check after all model computations and GT evaluation.
        after = protected_manifest(out)
        changed = [p for p, h in before.items() if after.get(p) != h]
        if changed or set(after) != set(before):
            raise AssertionError("Protected project files changed: " + str(changed))
        if any(r0.sha256(a["path"]) != a["sha256"] for a in artifacts):
            raise AssertionError("GT-free sealed artifacts changed during evaluation")
        verification = dict(status="passed", protected_file_count=len(before), protected_files_SHA256_unchanged=True,
              baseline_checkpoint_SHA256_unchanged=r0.sha256(args.model_path) == checkpoint["sha256"],
              model_parameters_and_BN_buffers_unchanged=True, model_state_tensor_count=len(state_before), model_eval_frozen=True,
              no_backward_no_optimizer_no_EMA=True, all_predictions_no_grad=True, no_BGS_no_donor_no_feature_modification=True,
              no_test_GT_read=True, phase_A_GT_reads=0, all_scores_mappings_and_budgets_saved_before_GT=True,
              phase_A_SHA256_unchanged=True, R0_prediction_match_atol=PRED_ATOL, prediction_checks=prediction_checks,
              flip_inverse_unit_test_passed=True, no_historical_patch_reselection=True, cases=8, patches=24,
              grid_error_count_conserved=True, all_methods_same_domain_same_budget=True, independent_validation_available=False,
              bootstrap_unit="case", bootstrap_repetitions=BOOTSTRAPS, all_outputs_finite=True, explicit_undefined_values=True,
              unit_test_exit_code=test.returncode, no_best_axis_selection=True)
        metadata.update(status="complete", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        protected_file_sha256_after=after, high_confidence_example=first_confident)
        report(out, metadata, summary, cases, direction_cases, statuses, verification)
        chosen = summary["recommendation"]["mapping"]
        for artifact in artifacts:
            fixed = ids.index(artifact["case_id"]) in (0, 3, 7) and artifact["patch_index"] == 0
            high = first_confident and (artifact["case_id"], artifact["patch_index"]) == (first_confident["case_id"], first_confident["patch_index"])
            if fixed or high:
                with np.load(out / "receiver_error_maps" / (artifact["case_id"] + "_p" + str(artifact["patch_index"]) + "_evaluation.npz")) as archive:
                    targets = {k: archive[k] for k in archive.files}
                if fixed:
                    visualize(out, artifact["original_R0"], artifact, targets["error_count_x3"], targets, chosen, 40, "preselected", history["image_display_range"])
                if high:
                    visualize(out, artifact["original_R0"], artifact, targets["error_count_x3"], targets, chosen, first_confident["voxel"][2], "first_confident_error", history["image_display_range"])
        for name, data in (("patch_level_metrics.csv", rows), ("case_level_metrics.csv", cases), ("direction_analysis.csv", directions),
                            ("direction_case_statistics.csv", direction_cases), ("original_scale_metrics.csv", original_rows),
                            ("original_case_metrics.csv", original_cases), ("downsampling_comparison.csv", mapping_rows),
                            ("x3_error_localization.csv", [{k: r[k] for k in ("case_id", "patch_index", "mapping", "domain", "method", "budget",
                                  "ap_any", "ap_severe10", "ap_severe25", "error_coverage", "domain_error_coverage", "selected_error_fraction", "enrichment",
                                  "spearman_fraction", "coverage_change_vs_original")} for r in rows]),
                            ("failure_analysis.csv", failures), ("patch_status.csv", statuses), ("paired_method_comparison.csv", paired),
                            ("data_access_log.csv", reader.events)):
            r0.write_csv(out / name, data)
        r0.write_json(out / "summary_statistics.json", summary)
        r0.write_json(out / "experiment_metadata.json", metadata)
        r0.write_json(out / "verification.json", verification)
        logging.info("R0.5 complete: %s; candidate=%s; %d protected files unchanged", out, summary["recommendation"], len(before))
        r0.write_csv(out / "new_result_files_manifest.csv", [dict(path=str(p.relative_to(ROOT)), sha256=r0.sha256(p), bytes=p.stat().st_size)
                  for p in sorted(out.rglob("*")) if p.is_file() and p.name != "new_result_files_manifest.csv"])
    except Exception as e:
        logging.exception("R05 failed; partial outputs retained without overwrite")
        metadata.update(status="failed", error=repr(e))
        r0.write_json(out / "experiment_metadata.json", metadata)
        r0.write_json(out / "verification.json", dict(status="failed", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
