"""R0: image-only frozen EMA risk construction, then sealed-map GT evaluation.

No BGS, feature modulation, student imitation, training, backward or EMA update.
Output directories must be new. Default outputs use an Asia/Shanghai timestamp.
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

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.special import xlogy
import torch
from model.vnet import VNet

METHODS = ("random", "boundary_prior", "entropy", "disagreement", "combined", "combined_boundary")
DOMAINS = ("global", "pseudo_boundary_neighborhood")
BUDGETS = (0.01, 0.02, 0.05, 0.10)
RADIUS = 3
TAU = 3.0
CONFIDENCE = 0.95
FAR_DISTANCE = 20
BOOTSTRAPS = 10000
NEW_CODE = {Path(__file__).resolve(), Path(__file__).resolve().with_name("test_experiment_receiver_R0.py")}
TZ = datetime.timezone(datetime.timedelta(hours=8))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_dir", default=str(ROOT / "analysis/results/experiment_bgs_utility/run_cuda_20261009_1713"))
    p.add_argument("--model_path", default=str(ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth"))
    p.add_argument("--output_dir", default=str(ROOT / "analysis/results/experiment_receiver_R0" /
                                               datetime.datetime.now(TZ).strftime("run_%Y%m%d_%H%M%S")))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu_threads", type=int, default=4)
    a = p.parse_args(argv)
    if a.seed < 0 or a.cpu_threads < 1:
        p.error("Invalid seed or cpu_threads")
    return a


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def finite(value, context):
    a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    bad = np.argwhere(~np.isfinite(a))
    if len(bad):
        raise FloatingPointError("{} NaN/Inf at {}".format(context, bad.tolist()))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError("Nonfinite serialized output")
    return value


def write_json(path, value):
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in safe(row).items()})


def protected_manifest(out):
    result = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")
                         and (Path(parent) / d).resolve() != out.resolve())
        for name in sorted(files):
            p = Path(parent) / name
            if p.is_file() and p.resolve() not in NEW_CODE and p.suffix not in (".pyc", ".pyo"):
                result[str(p.relative_to(ROOT))] = sha256(p)
    return result


def create_output(path):
    p = Path(path).resolve()
    if p == ROOT or (ROOT in p.parents and "Results" in p.relative_to(ROOT).parts):
        raise ValueError("Output cannot be the repository or original Results")
    p.mkdir(parents=True, exist_ok=False)
    return p


def validate_history(history, model_path):
    try:
        ids = history["data_split"]["labeled_case_ids"]
        config = history["configuration"]
        coords = history["sampling"]["patch_coordinates"]
        assert history["status"] == "complete"
        assert len(ids) == len(set(ids)) == 8 and config["patch_size"] == [112, 112, 80]
        assert len(coords) == 24 and {(c["case_id"], c["patch_index"]) for c in coords} == {(i, j) for i in ids for j in range(3)}
        for c in coords:
            for key in ("start_padded", "start_original", "padding", "original_shape", "padded_shape"):
                assert len(c[key]) == 3
        sources = history["data_sources"]
        assert {s["case_id"] for s in sources} == set(ids) and len(sources) == 8
        train = Path(history["data_split"]["train_list_path"])
        test = Path(history["data_split"]["test_list_path"])
        assert train.read_text().splitlines()[:8] == ids
        assert not set(ids) & set(test.read_text().splitlines())
        assert sha256(train) == history["data_split"]["train_list_sha256"]
        assert sha256(test) == history["data_split"]["test_list_sha256"]
        checkpoint = next(c for c in history["checkpoints"] if c["name"] == "baseline_ema")
        assert Path(model_path).resolve() == Path(checkpoint["path"]).resolve()
        assert sha256(model_path) == checkpoint["sha256"]
        for s in sources:
            assert sha256(s["path"]) == s["sha256"]
    except (AssertionError, KeyError, StopIteration, TypeError) as e:
        raise ValueError("Historical crop/source/checkpoint provenance invalid; no new sampling allowed") from e
    return ids, checkpoint


def replay_array(array, coords, patch_size, dtype):
    if list(array.shape) != coords["original_shape"]:
        raise ValueError("Historical source shape changed")
    padding = [max((p - s) // 2 + 3, 0) for p, s in zip(patch_size, array.shape)] if any(
        s <= p for s, p in zip(array.shape, patch_size)) else [0, 0, 0]
    if padding != coords["padding"] or [s + 2 * p for s, p in zip(array.shape, padding)] != coords["padded_shape"]:
        raise ValueError("Historical padding differs")
    start = coords["start_padded"]
    if [s - p for s, p in zip(start, padding)] != coords["start_original"]:
        raise ValueError("Original and padded coordinates disagree")
    if any(s < 0 or s + p > d for s, p, d in zip(start, patch_size, coords["padded_shape"])):
        raise ValueError("Historical crop outside source")
    slices = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
    return np.ascontiguousarray(np.pad(array, [(p, p) for p in padding])[slices], dtype=dtype)


class HistoricalReader:
    """The label API is locked until ALL image-only candidate files are sealed."""
    def __init__(self, history):
        self.history = history
        self.sources = {s["case_id"]: s for s in history["data_sources"]}
        self.coords = {(c["case_id"], c["patch_index"]): c for c in history["sampling"]["patch_coordinates"]}
        self.events = []
        self.sealed = False

    def read(self, case_id, patch_index, dataset):
        if dataset not in ("image", "label") or dataset == "label" and not self.sealed:
            raise RuntimeError("Receiver GT access prohibited until all Phase-A artifacts are saved")
        source = self.sources[case_id]
        with h5py.File(source["path"], "r") as f:
            array = f[dataset][:]
        self.events.append(dict(case_id=case_id, patch_index=patch_index, dataset=dataset,
                                phase="B" if self.sealed else "A", event_index=len(self.events)))
        result = replay_array(array, self.coords[case_id, patch_index], self.history["configuration"]["patch_size"],
                              np.float32 if dataset == "image" else np.int64)
        finite(result, case_id + "/" + dataset)
        if dataset == "label" and not np.isin(result, [0, 1]).all():
            raise ValueError("Nonbinary GT")
        return result

    def seal(self, artifacts):
        if len(artifacts) != len(self.coords):
            raise RuntimeError("Incomplete Phase-A artifacts")
        if any(not Path(a["path"]).is_file() or sha256(a["path"]) != a["sha256"] for a in artifacts):
            raise RuntimeError("Phase-A artifact digest mismatch")
        self.sealed = True


def dilate(mask, iterations=1):
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        result = ndi.maximum_filter(result, size=3, mode="constant", cval=0).astype(bool)
    return result


def boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    eroded = ~ndi.maximum_filter(~mask, size=3, mode="constant", cval=0).astype(bool)
    return dilate(mask) & ~eroded


def pseudo_boundary_distance(pseudo):
    b = boundary(pseudo)
    valid = bool(b.any())
    # Missing boundary has no meaningful distance. Finite sentinel + explicit
    # flag ensures no Inf files; its prior/domain are identically zero/empty.
    distance = ndi.distance_transform_edt(~b) if valid else np.full(b.shape, np.linalg.norm(np.array(b.shape) - 1) + 1)
    prior = np.exp(-distance / TAU) if valid else np.zeros(b.shape)
    return b, distance.astype(np.float32), prior.astype(np.float32), valid


def construct_risks(probability, aligned_perturbations, seed):
    """No image labels, errors, GT boundary, utility or Dice accepted."""
    p = np.asarray(probability, dtype=np.float64)
    finite(p, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probability out of range")
    entropy = -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / np.log(2)
    disagreement = np.mean(np.abs(np.asarray(aligned_perturbations, dtype=np.float64) - p), axis=0)
    finite(disagreement, "aligned disagreement")
    b, distance, prior, valid = pseudo_boundary_distance(p > 0.5)
    combined = (entropy + disagreement) / 2  # D_n=D: its theoretical range is [0,1].
    random_score = np.random.RandomState(seed).uniform(size=p.shape).astype(np.float32)
    maps = dict(random=random_score, boundary_prior=prior, entropy=np.clip(entropy, 0, 1).astype(np.float32),
                disagreement=disagreement.astype(np.float32), combined=combined.astype(np.float32),
                combined_boundary=(combined * prior).astype(np.float32))
    for name, value in maps.items():
        finite(value, name)
        if np.any((value < 0) | (value > 1 + 1e-6)):
            raise ValueError(name + " risk out of [0,1]")
    return maps, b, distance, (distance <= RADIUS) & valid, valid


@torch.no_grad()
def teacher_predictions(model, image, device):
    x = torch.from_numpy(image)[None, None].to(device)
    original = torch.softmax(model(x), dim=1)[0, 1]
    repeated = torch.softmax(model(x), dim=1)[0, 1]
    if not torch.equal(original, repeated):
        raise AssertionError("No-augmentation EMA eval forward is not deterministic")
    aligned = []
    for axis in (0, 1, 2):
        transformed = torch.flip(x, [axis + 2])
        probability = torch.softmax(model(transformed), dim=1)[0, 1]
        restored = torch.flip(probability, [axis])
        aligned.append(restored.cpu().numpy())
    finite(original, "teacher probability")
    return original.cpu().numpy(), np.stack(aligned), dict(no_aug_forward_exact=True, transforms=["flip_x", "flip_y", "flip_z"])


def ordered_indices(score, domain, tie_rank):
    indices = np.flatnonzero(domain.ravel())
    return indices[np.lexsort((tie_rank[indices], -score.ravel()[indices]))]


def budget_count(size, ratio):
    return min(size, max(1, int(round(size * ratio)))) if size else 0


def candidate_name(domain, method, budget):
    return "candidate_{}_{}_{}".format(domain, method, int(budget * 100))


def build_candidates(maps, neighborhood, tie_seed):
    shape = neighborhood.shape
    tie_rank = np.random.RandomState(tie_seed).permutation(neighborhood.size)
    payload, counts = {}, []
    for domain_name in DOMAINS:
        domain = np.ones(shape, dtype=bool) if domain_name == "global" else neighborhood
        for method in METHODS:
            ordered = ordered_indices(maps[method], domain, tie_rank)
            for ratio in BUDGETS:
                k = budget_count(len(ordered), ratio)
                selected = np.zeros(neighborhood.size, dtype=np.uint8)
                selected[ordered[:k]] = 1
                payload[candidate_name(domain_name, method, ratio)] = np.packbits(selected)
                counts.append(dict(domain=domain_name, method=method, budget=ratio, domain_voxels=len(ordered), selected_voxels=k))
    return payload, counts


def unpack_candidate(archive, domain, method, budget, shape):
    return np.unpackbits(archive[candidate_name(domain, method, budget)])[:int(np.prod(shape))].reshape(shape).astype(bool)


def evaluate_errors(p, gt, b_p):
    pseudo, foreground = p > 0.5, gt > 0
    b_gt = boundary(foreground)
    evaluation_band = dilate(b_gt, RADIUS) | dilate(b_p, RADIUS)
    errors = pseudo ^ foreground
    boundary_errors = errors & evaluation_band
    confidence = np.maximum(p, 1 - p)
    high_conf_errors = errors & (confidence >= CONFIDENCE)
    if foreground.any():
        fg_distance = ndi.distance_transform_edt(~foreground).astype(np.float32)
        far = pseudo & ~foreground & (fg_distance > FAR_DISTANCE)
        far_defined, far_reason = True, ""
    else:
        fg_distance = np.full(p.shape, np.linalg.norm(np.array(p.shape) - 1) + 1, dtype=np.float32)
        far = np.zeros(p.shape, dtype=bool)
        far_defined, far_reason = False, "empty_GT_foreground_far_distance_undefined"
    border_distance = np.minimum.reduce(np.broadcast_arrays(*[
        np.minimum(np.arange(s), s - 1 - np.arange(s)).reshape(tuple(s if j == i else 1 for j in range(3)))
        for i, s in enumerate(p.shape)]))
    far_core = far & (border_distance > FAR_DISTANCE)
    return dict(gt=gt.astype(np.uint8), pseudo=pseudo, gt_boundary=b_gt, evaluation_band=evaluation_band,
                all_errors=errors, boundary_errors=boundary_errors, high_confidence_errors=high_conf_errors,
                far_fp=far, far_fp_core=far_core, far_fp_crop_sensitive=far & ~far_core,
                high_confidence_far_fp=far & (p >= CONFIDENCE), foreground_distance=fg_distance,
                far_defined=far_defined, far_reason=far_reason)


def average_precision_sorted(sorted_scores, sorted_target):
    positive = int(np.count_nonzero(sorted_target))
    if len(sorted_target) == 0:
        return None, "empty_evaluation_domain"
    if not positive:
        return None, "no_positive_errors"
    # Group tied scores exactly, matching sklearn's average_precision_score.
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp = np.cumsum(sorted_target, dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(precision * np.diff(np.r_[0, tp]) / positive)), ""


def selection_metrics(selected, domain, target):
    n, positives, k = int(domain.sum()), int((target & domain).sum()), int(selected.sum())
    if np.any(selected & ~domain):
        raise AssertionError("Selection outside fixed candidate domain")
    hits = int((selected & target).sum())
    return dict(precision=hits / k if k else None, recall=hits / positives if positives else None,
                enrichment=(hits / k) / (positives / n) if k and positives else None,
                prevalence=positives / n if n else None, positive_voxels=positives,
                selected_error_voxels=hits, selected_voxels=k,
                undefined_reason="empty_evaluation_domain" if not n else "no_positive_errors" if not positives else "")


def target_recall(selected, target, valid=True):
    n = int(target.sum())
    return float((selected & target).sum() / n) if n and valid else None


def evaluate_patch(case_id, patch_index, archive, gt):
    p = archive["probability"]
    targets = evaluate_errors(p, gt, archive["pseudo_boundary"].astype(bool))
    neighborhood = archive["pseudo_boundary_neighborhood"].astype(bool)
    shape = p.shape
    rows, far_rows = [], []
    common = dict(case_id=case_id, patch_index=patch_index, all_error_voxels=int(targets["all_errors"].sum()),
                  boundary_error_voxels=int(targets["boundary_errors"].sum()),
                  high_confidence_error_voxels=int(targets["high_confidence_errors"].sum()),
                  errors_outside_predicted_neighborhood=int((targets["all_errors"] & ~neighborhood).sum()),
                  boundary_errors_outside_predicted_neighborhood=int((targets["boundary_errors"] & ~neighborhood).sum()),
                  far_fp_voxels=int(targets["far_fp"].sum()), far_fp_core_voxels=int(targets["far_fp_core"].sum()),
                  far_fp_crop_sensitive_voxels=int(targets["far_fp_crop_sensitive"].sum()),
                  high_confidence_far_fp_voxels=int(targets["high_confidence_far_fp"].sum()),
                  far_fp_defined=targets["far_defined"], far_fp_undefined_reason=targets["far_reason"],
                  pseudo_boundary_present=bool(archive["pseudo_boundary_present"]),
                  patch_dice=float(2 * np.sum((p > 0.5) & (gt > 0)) / (np.sum(p > 0.5) + np.sum(gt > 0)))
                  if np.sum(p > 0.5) + np.sum(gt > 0) else 1.0)
    entropy, disagreement = archive["risk_entropy"], archive["risk_disagreement"]
    far, high = targets["far_fp"], targets["high_confidence_far_fp"]
    far_entropy = float(entropy[far].mean()) if far.any() else None
    far_disagreement = float(disagreement[far].mean()) if far.any() else None
    high_far_low_disagreement = int((high & (disagreement <= 0.05)).sum())
    for domain_name in DOMAINS:
        domain = np.ones(shape, dtype=bool) if domain_name == "global" else neighborhood
        for method in METHODS:
            score = archive["risk_" + method]
            order = np.flatnonzero(domain.ravel())
            order = order[np.argsort(-score.ravel()[order], kind="stable")]
            aps = {}
            for target_name in ("all_errors", "boundary_errors"):
                ap, reason = average_precision_sorted(score.ravel()[order], targets[target_name].ravel()[order])
                aps["ap_" + target_name] = ap
                aps["ap_" + target_name + "_undefined_reason"] = reason
            # GT-defined domain is used ONLY for retrospective ranking, never
            # for constructing/selecting a candidate mask.
            restricted = order[targets["evaluation_band"].ravel()[order]]
            ap_band, band_reason = average_precision_sorted(score.ravel()[restricted], targets["all_errors"].ravel()[restricted])
            aps.update(ap_within_evaluation_band=ap_band, ap_within_band_undefined_reason=band_reason,
                       evaluation_band_prevalence=float(targets["all_errors"].ravel()[restricted].mean()) if len(restricted) else None)
            for ratio in BUDGETS:
                selected = unpack_candidate(archive, domain_name, method, ratio, shape)
                if int(selected.sum()) != budget_count(int(domain.sum()), ratio):
                    raise AssertionError("Top-K count differs from precommitted budget")
                row = dict(**common, domain=domain_name, method=method, budget=ratio,
                           domain_voxels=int(domain.sum()), selected_voxels=int(selected.sum()), **aps)
                for target_name in ("all_errors", "boundary_errors"):
                    metrics = selection_metrics(selected, domain, targets[target_name])
                    row.update({target_name + "_" + k: v for k, v in metrics.items() if k != "selected_voxels"})
                row["high_confidence_error_recall"] = target_recall(selected, targets["high_confidence_errors"])
                row["far_fp_recall"] = target_recall(selected, targets["far_fp"], targets["far_defined"])
                row["far_fp_core_recall"] = target_recall(selected, targets["far_fp_core"], targets["far_defined"])
                row["high_confidence_far_fp_recall"] = target_recall(selected, targets["high_confidence_far_fp"], targets["far_defined"])
                row["high_confidence_error_undefined_reason"] = "no_high_confidence_errors" if not common["high_confidence_error_voxels"] else ""
                row["far_recall_undefined_reason"] = targets["far_reason"] or ("no_far_FP" if not common["far_fp_voxels"] else "")
                row["far_core_recall_undefined_reason"] = targets["far_reason"] or ("no_far_FP_core" if not common["far_fp_core_voxels"] else "")
                row["high_confidence_far_recall_undefined_reason"] = targets["far_reason"] or ("no_high_confidence_far_FP" if not common["high_confidence_far_fp_voxels"] else "")
                rows.append(row)
                far_rows.append(dict(**common, domain=domain_name, method=method, budget=ratio,
                                     far_fp_mean_entropy=far_entropy,
                                     far_fp_mean_disagreement=far_disagreement,
                                     far_fp_recall=row["far_fp_recall"], high_confidence_far_fp_recall=row["high_confidence_far_fp_recall"],
                                     high_confidence_far_fp_miss_fraction=1 - row["high_confidence_far_fp_recall"] if row["high_confidence_far_fp_recall"] is not None else None,
                                     high_confidence_far_fp_low_D_voxels=high_far_low_disagreement,
                                     undefined_reason=row["far_recall_undefined_reason"]))
    return rows, far_rows, targets, common


def bootstrap_cases(values, seed):
    values = np.array([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(mean=None, median=None, std=None, ci95=[None, None], n_cases=0,
                    reason="no_defined_case_values", resampling_unit="case")
    finite(values, "bootstrap values")
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, len(values), (BOOTSTRAPS, len(values)))].mean(1)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                ci95=np.percentile(means, [2.5, 97.5]).tolist(), n_cases=len(values),
                resampling_unit="case", repetitions=BOOTSTRAPS)


def aggregate(rows, case_ids, seed):
    identity = {"case_id", "patch_index", "domain", "method", "budget"}
    numeric = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in identity and not k.endswith("reason") and (
               k in ("ap_all_errors", "ap_boundary_errors", "ap_within_evaluation_band") or
               k.endswith(("_recall", "_precision", "_enrichment", "_prevalence")) or
               any(isinstance(r.get(k), (float, int, np.floating, np.integer)) and not isinstance(r.get(k), bool) for r in rows))]
    cases, summary, paired = [], {}, []
    for case_id in case_ids:
        for domain in DOMAINS:
            for method in METHODS:
                for ratio in BUDGETS:
                    pp = [r for r in rows if (r["case_id"], r["domain"], r["method"], r["budget"]) == (case_id, domain, method, ratio)]
                    if len(pp) != 3:
                        raise AssertionError("A historical patch was silently omitted")
                    row = dict(case_id=case_id, domain=domain, method=method, budget=ratio, n_patches=3)
                    for k in numeric:
                        vals = [p[k] for p in pp if p.get(k) is not None]
                        row[k] = float(np.mean(vals)) if vals else None
                        row[k + "_defined_patches"] = len(vals)
                    row["undefined_patch_reasons"] = [dict(patch_index=p["patch_index"], reasons={k: v for k, v in p.items()
                                                                                               if k.endswith("reason") and v}) for p in pp]
                    cases.append(row)
    metrics_to_pair = ("ap_all_errors", "ap_boundary_errors", "all_errors_precision", "all_errors_recall",
                       "boundary_errors_precision", "boundary_errors_recall", "high_confidence_error_recall", "far_fp_recall")
    for domain in DOMAINS:
        summary[domain] = {}
        for method in METHODS:
            summary[domain][method] = {}
            for ratio in BUDGETS:
                cc = [r for r in cases if (r["domain"], r["method"], r["budget"]) == (domain, method, ratio)]
                key = str(int(ratio * 100))
                summary[domain][method][key] = {k: bootstrap_cases([c[k] for c in cc], seed + i * 17)
                                               for i, k in enumerate(numeric)}
    by_key = {(c["case_id"], c["domain"], c["method"], c["budget"]): c for c in cases}
    pair_summary = {}
    for domain in DOMAINS:
        for ratio in BUDGETS:
            for method in METHODS:
                for control in ("boundary_prior", "random", "entropy", "disagreement"):
                    if method == control:
                        continue
                    group_key = "{}/{}/{}/{}".format(domain, int(ratio * 100), method, control)
                    group = []
                    for case_id in case_ids:
                        a = by_key[case_id, domain, method, ratio]
                        b = by_key[case_id, domain, control, ratio]
                        row = dict(case_id=case_id, domain=domain, budget=ratio, method=method, control=control)
                        for metric in metrics_to_pair:
                            row[metric + "_difference"] = a[metric] - b[metric] if a.get(metric) is not None and b.get(metric) is not None else None
                        group.append(row)
                        paired.append(row)
                    pair_summary[group_key] = {}
                    for i, metric in enumerate(metrics_to_pair):
                        vals = [r[metric + "_difference"] for r in group]
                        stat = bootstrap_cases(vals, seed + i * 19)
                        stat.update(improved_cases=sum(v is not None and v > 0 for v in vals),
                                    worsened_cases=sum(v is not None and v < 0 for v in vals),
                                    tied_cases=sum(v is not None and v == 0 for v in vals))
                        pair_summary[group_key][metric] = stat
    return cases, paired, dict(methods=summary, paired_comparisons=pair_summary)


def choose_recommendation(summary, patch_status):
    pairs = summary["paired_comparisons"]
    def pair(method, control, domain="pseudo_boundary_neighborhood", metric="ap_all_errors", budget=5):
        return pairs["{}/{}/{}/{}".format(domain, budget, method, control)][metric]
    def metric(method, name, domain="global", budget=10):
        return summary["methods"][domain][method][str(budget)][name]
    selected = "entropy"
    d = pair("disagreement", "entropy")
    if d["n_cases"] and d["ci95"][0] > 0 and d["improved_cases"] >= 6 and pair(
            "disagreement", "entropy", "global", "boundary_errors_recall")["mean"] >= 0:
        selected = "disagreement"
    combination_supported = False
    for combo in ("combined", "combined_boundary"):
        advantages = [pair(combo, s) for s in ("entropy", "disagreement")]
        if all(a["n_cases"] and a["ci95"][0] > 0 and a["improved_cases"] >= 6 for a in advantages) and all(
                pair(combo, s, "global", "boundary_errors_recall")["mean"] >= 0 for s in ("entropy", "disagreement")):
            selected, combination_supported = combo, True
    confident = sum(p["high_confidence_error_voxels"] for p in patch_status)
    far_confident = sum(p["high_confidence_far_fp_voxels"] for p in patch_status)
    hc = metric(selected, "high_confidence_error_recall")
    hcf = metric(selected, "high_confidence_far_fp_recall")
    baseline_advantage = pair(selected, "boundary_prior")
    poor = (not baseline_advantage["n_cases"] or baseline_advantage["ci95"][0] <= 0 or
            (confident >= 50 and (hc["mean"] is None or hc["mean"] < 0.5)) or
            (far_confident >= 50 and (hcf["mean"] is None or hcf["mean"] < 0.5)))
    choice = "C" if poor else "A" if combination_supported else "B"
    return dict(choice=choice, candidate_method=selected, combination_supported=combination_supported,
                confident_error_observations=confident, confident_far_fp_observations=far_confident,
                rule="Prefer simple entropy unless D or combination improves common-domain AP with case CI>0 and >=6/8 positive paired cases, without lowering global boundary recall at5%. C if no stable advantage over boundary prior or top10% recall<50% for >=50 high-confidence errors/farFP. These are exploratory gates, not clinical thresholds.",
                interpretation={"A": "组合稳定且未触发主要高置信度漏检门槛，可以设计R1的独立验证。",
                                "B": "优先采用单指标作小预算排序；没有组合全面优于它的证据。组合在较大预算下的覆盖优势仍需保留，R1须按病例独立验证。",
                                "C": "暂不进入特征交互R1，先改进Receiver错误分析和高置信度漏检。"}[choice])


def visualize(out, artifact, targets, selected_method, suffix, z_index, image_range, focus_voxel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(artifact["path"]) as a:
        image, p = a["image"], a["probability"]
        selection = unpack_candidate(a, "global", selected_method, 0.05, p.shape)
        panels = [(image, "MRI", "gray", image_range[0], image_range[1]),
                  (p, "Teacher foreground p", "viridis", 0, 1),
                  (a["pseudo_boundary"], "Pseudo boundary", "gray", 0, 1),
                  (targets["gt_boundary"], "GT boundary (evaluation)", "gray", 0, 1),
                  (targets["all_errors"], "Actual error", "Reds", 0, 1),
                  (a["risk_entropy"], "Entropy H", "magma", 0, 1),
                  (a["risk_disagreement"], "Aligned disagreement D", "magma", 0, 1),
                  (selection, selected_method + " top5% candidate", "Reds", 0, 1)]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for ax, (data, title, cmap, low, high) in zip(axes.ravel(), panels):
            im = ax.imshow(data[:, :, z_index].T, origin="lower", cmap=cmap, vmin=low, vmax=high)
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
            if focus_voxel is not None:
                ax.plot(focus_voxel[0], focus_voxel[1], marker="+", markersize=12, markeredgewidth=2, color="cyan")
            if title.startswith(("Teacher", "Entropy", "Aligned")):
                fig.colorbar(im, ax=ax, fraction=0.045)
        fig.suptitle("{} patch{} z={} | {} | fixed spatial coordinates/scales".format(artifact["case_id"], artifact["patch_index"], z_index, suffix))
        if focus_voxel is not None:
            point = tuple(focus_voxel)
            fig.text(0.5, 0.015, "Cyan +: voxel {} | p={:.5f}, H={:.5f}, D={:.5f}, GT={}".format(
                focus_voxel, float(p[point]), float(a["risk_entropy"][point]), float(a["risk_disagreement"][point]), int(targets["gt"][point])), ha="center")
        fig.tight_layout(rect=(0, 0.04 if focus_voxel is not None else 0, 1, 0.96))
        fig.savefig(str(out / "visualizations" / (artifact["case_id"] + "_p" + str(artifact["patch_index"]) + "_" + suffix + ".png")), dpi=150)
        plt.close(fig)
        # Always also display the predeclared boundary combination, independent
        # of whether it is statistically recommended as a future mask.
        combo = unpack_candidate(a, "global", "combined_boundary", 0.05, p.shape)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image[:, :, z_index].T, origin="lower", cmap="gray", vmin=image_range[0], vmax=image_range[1])
        ax.contour(combo[:, :, z_index].T, levels=[0.5], colors=["cyan"], linewidths=0.7)
        ax.set_title("Predeclared R_B top5% (GT-free) | " + suffix)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(out / "visualizations" / (artifact["case_id"] + "_p" + str(artifact["patch_index"]) + "_" + suffix + "_RB.png")), dpi=150)
        plt.close(fig)


def stat_text(stat):
    return "{:.4f} [{:.4f}, {:.4f}]".format(stat["mean"], *stat["ci95"]) if stat["mean"] is not None else "未定义(n=0)"


def report(out, metadata, summary, case_rows, patch_status, verification):
    out = Path(out).resolve()
    names = {"random": "随机", "boundary_prior": "预测边界距离", "entropy": "Entropy", "disagreement": "Disagreement",
             "combined": "H+D", "combined_boundary": "(H+D)×边界先验"}
    recommendation = summary["recommendation"]
    lines = ["# Experiment R0：Receiver边界可靠性分析", "", "## 研究问题、数据与保护", "",
             "仅根据Receiver图像和同一冻结Teacher的概率及翻转稳定性，检验是否可定位真实分割错误。本轮没有Donor、BGS增强、特征补偿或训练。使用Baseline EMA iter27000，不把两个EMA副本冒称Student–Teacher。",
             "精确复现BGS-U的train.list前8个labeled病例、每例3个112×112×80 patch；核对H5 SHA256、裁剪与padding，不重新抽样。病例已用于模型有标签训练，因此结果仅为探索性，不是未见病例验证。",
             "Phase A只访问H5的image dataset；完成全部24个风险图和所有预算候选mask的保存及SHA封存后，Phase B才解锁label dataset。对源H5做字节哈希仅用于历史来源核验，不解析GT；dataset访问审计见data_access_log.csv。GT边界、错误图与Dice仅用于B。", "",
             "## 实验前固定的方法与评价域", "",
             "H为归一化二分类熵，使用xlogy避免log(0)。D为原预测与x/y/z翻转后逆变换概率的平均绝对差；Dn=D（理论范围[0,1]），不做GT或每patch最大值归一化。R=(H+D)/2，P_B=exp(−d/3)，R_B=R×P_B。预测边界为26邻域一体素内外带，距离为欧氏voxel；共同预测边界域d≤3。", 
             "计算成本：Entropy只需1次Teacher前向；D与两个组合需原图及3次翻转共4次前向，边界prior/RB还需一次距离变换。没有可学习参数。分析中重复原图前向仅作一致性检查，不是Student预测。",
             "评价边界域B_eval=dilate(B_GT,3)∪dilate(Bp,3)，膨胀用3×3×3重复3次（Chebyshev邻域）；它只用于后验目标与域内AP，不用于GT筛选Top-K。全域E_all与E_boundary分别评价；共同候选域内E_all是主要公平排序对照，各方法在该域用相同1/2/5/10%预算。全域预算分母为整个patch，共同域预算分母为共同域voxel数，不混淆两者。",
             "连续距离prior是位置基线，不是错误概率。随机分数和同分时随机tie rank都用独立固定seed，所有方法同预算数量严格相同。AUPRC为average precision，准确处理同分组，并同时报告阳性率。无错误时AP/Recall/Enrichment未定义，保留空值、原因与有效patch/case数，不记为成功。",
             "高置信度固定max(p,1−p)≥0.95。远距离FP为预测前景且GT背景、距patch内GT前景>20 voxel；空GT时此目标未定义。裁剪可能漏掉patch外更近的GT或patch外错误：保留所有patch定义的farFP，另列距所有crop面>20的保守core与crop-sensitive部分，不能把后者删掉后宣称安全。",
             "先patch→病例3patch均值→8例等权；10000次病例bootstrap，配对比较也只重采样病例。CI为探索性，未做多重比较校正，不能把小样本多指标比较称为无偏选型或因果证明。",
             "可视化预定病例序号0/3/7、patch0、z=40；另取固定病例/patch顺序中第一个高置信度错误的首个voxel所在切片，仅用于事后错误展示。所有概率/风险色标为[0,1]，MRI用全部图像共同范围，不挑表现最好的示例。", "", "## 定量结果", "",
             "均值[95%病例CI]；全域和共同预测边界域AP_all，以及全域Top5%的边界错误定位：", "",
             "|方法|全域AP_all|全域AP_boundary|共同边界域AP_all|共同域阳性率|Top5%边界Precision|Top5%边界Recall|Top5%边界富集|", "|---|---|---|---|---|---|---|---|"]
    for method in METHODS:
        g = summary["methods"]["global"][method]["5"]
        b = summary["methods"]["pseudo_boundary_neighborhood"][method]["5"]
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|".format(names[method], stat_text(g["ap_all_errors"]), stat_text(g["ap_boundary_errors"]),
                      stat_text(b["ap_all_errors"]), stat_text(b["all_errors_prevalence"]), stat_text(g["boundary_errors_precision"]),
                      stat_text(g["boundary_errors_recall"]), stat_text(g["boundary_errors_enrichment"])))
    base = summary["methods"]["global"]["random"]["5"]
    lines += ["", "全域错误阳性率 {}；边界错误阳性率 {}。全部1/2/5/10%结果和缺失原因见patch/case CSV及summary。".format(
              stat_text(base["all_errors_prevalence"]), stat_text(base["boundary_errors_prevalence"])), "",
              "三个错误目标不互斥：远FP在其自己的预测轮廓附近时也属于E_boundary，因此不能把边界高覆盖解释为远FP不存在。", "",
              "### 预算改变时的取舍", "", "|方法|Top1%边界Recall|Top2%边界Recall|Top5%边界Recall|Top10%边界Recall|", "|---|---|---|---|---|"]
    for method in ("entropy", "disagreement", "combined", "combined_boundary"):
        lines.append("|{}|{}|".format(names[method], "|".join(stat_text(summary["methods"]["global"][method][b]["boundary_errors_recall"]) for b in ("1", "2", "5", "10"))))
    lines += ["", "**不能省略的组合收益：** D在1–2%预算下明显更强；但5%预算下H+D和RB对D的边界Recall配对增量分别是 {}、{}。二者均在8/8例增加覆盖，完整排序AP却都低于D。因此‘没有组合普遍优于D的证据’不等于‘组合没有价值’，最合适的信号取决于交互预算和precision/recall目标。".format(
              stat_text(summary["paired_comparisons"]["global/5/combined/disagreement"]["boundary_errors_recall"]),
              stat_text(summary["paired_comparisons"]["global/5/combined_boundary/disagreement"]["boundary_errors_recall"])), "",
              "## 同域配对比较：不是只击败全域随机", "", "|方法|共同域AP相对边界prior差 [CI]|更好/更差病例|全域Top5%边界Recall相对prior差 [CI]|", "|---|---|---|---|"]
    for method in METHODS:
        if method == "boundary_prior":
            continue
        p = summary["paired_comparisons"]["pseudo_boundary_neighborhood/5/" + method + "/boundary_prior"]["ap_all_errors"]
        recall = summary["paired_comparisons"]["global/5/" + method + "/boundary_prior"]["boundary_errors_recall"]
        lines.append("|{}|{}|{}/{}|{}|".format(names[method], stat_text(p), p["improved_cases"], p["worsened_cases"], stat_text(recall)))
    lines += ["", "|组合/差异|共同域AP配对差 [CI]|更好/更差病例|", "|---|---|---|"]
    for method, control in (("disagreement", "entropy"), ("combined", "entropy"), ("combined", "disagreement"),
                            ("combined_boundary", "entropy"), ("combined_boundary", "disagreement")):
        p = summary["paired_comparisons"]["pseudo_boundary_neighborhood/5/" + method + "/" + control]["ap_all_errors"]
        lines.append("|{}−{}|{}|{}/{}|".format(names[method], names[control], stat_text(p), p["improved_cases"], p["worsened_cases"]))
    lines += ["", "## Receiver差异与未覆盖错误", "", "|病例|Entropy共同域AP|D共同域AP|H+D共同域AP|RB共同域AP|错误voxel总观察数|预测边界邻域外错误|高置信度错误|farFP / core / crop-sensitive|", "|---|---|---|---|---|---|---|---|---|"]
    for case_id in metadata["case_order"]:
        cc = [c for c in case_rows if c["case_id"] == case_id and c["domain"] == "pseudo_boundary_neighborhood" and c["budget"] == 0.05]
        values = {c["method"]: c for c in cc}
        pp = [p for p in patch_status if p["case_id"] == case_id]
        values_text = ["{:.4f}".format(values[m]["ap_all_errors"]) if values[m]["ap_all_errors"] is not None else "未定义" for m in ("entropy", "disagreement", "combined", "combined_boundary")]
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{} / {} / {}|".format(case_id, *values_text,
                      sum(p["all_error_voxels"] for p in pp), sum(p["errors_outside_predicted_neighborhood"] for p in pp),
                      sum(p["high_confidence_error_voxels"] for p in pp), sum(p["far_fp_voxels"] for p in pp),
                      sum(p["far_fp_core_voxels"] for p in pp), sum(p["far_fp_crop_sensitive_voxels"] for p in pp)))
    totals = {k: sum(p[k] for p in patch_status) for k in ("all_error_voxels", "boundary_error_voxels", "high_confidence_error_voxels",
              "errors_outside_predicted_neighborhood", "boundary_errors_outside_predicted_neighborhood", "far_fp_voxels", "far_fp_core_voxels",
              "far_fp_crop_sensitive_voxels", "high_confidence_far_fp_voxels")}
    far_statistics = summary["methods"]["global"]["disagreement"]["10"]
    with (out / "far_false_positive_analysis.csv").open() as f:
        far_low_D = sum(int(row["high_confidence_far_fp_low_D_voxels"]) for row in csv.DictReader(f)
                        if row["domain"] == "global" and row["method"] == "disagreement" and float(row["budget"]) == 0.1)
    lines += ["", "总数是24个裁剪patch的voxel观察数，patch之间可重叠，不能当作独立voxel或整例去重计数：" + json.dumps(totals, ensure_ascii=False), "",
              "## 远距离与高置信度漏检", "", "|方法|全域Top10%全部高置信错误Recall [CI]|farFP Recall [CI]|高置信farFP Recall [CI]|", "|---|---|---|---|"]
    for method in METHODS:
        g = summary["methods"]["global"][method]["10"]
        lines.append("|{}|{}|{}|{}|".format(names[method], stat_text(g["high_confidence_error_recall"]), stat_text(g["far_fp_recall"]),
                      stat_text(g["high_confidence_far_fp_recall"])))
    lines += ["", "far_false_positive_analysis.csv保留每patch远FP的平均H、D、各预算覆盖及高置信farFP漏检比例。若该类为0，Recall未定义，不能声称成功定位。高置信度错误示例存在状态：" + str(metadata["high_confidence_example"]), "",
              "farFP覆盖仅在{}个有该类错误的病例定义；高置信farFP仅{}例、core farFP仅{}例。观察到的Top10%覆盖只说明这批裁剪内的已观察目标；本批高置信farFP中D≤0.05的数量为{}。{}/{}个farFP属于crop-sensitive，外推整例远距离失效仍缺证据。".format(
              far_statistics["far_fp_recall"]["n_cases"], far_statistics["high_confidence_far_fp_recall"]["n_cases"],
              far_statistics["far_fp_core_recall"]["n_cases"], far_low_D, totals["far_fp_crop_sensitive_voxels"], totals["far_fp_voxels"]),
              "## 七个科学问题", "",
              "Q1. Entropy定位边界错误的全域Top5%Recall为 {}；共同域AP相对边界prior为 {}。后者才检验是否超越仅靠预测边界位置。".format(
              stat_text(summary["methods"]["global"]["entropy"]["5"]["boundary_errors_recall"]),
              stat_text(summary["paired_comparisons"]["pseudo_boundary_neighborhood/5/entropy/boundary_prior"]["ap_all_errors"])),
              "Q2. D−H共同域AP差 {}。其区间跨零时没有稳定额外排序价值；正向区间只能说明本固定样本上的增量，不是真实错误概率。".format(
              stat_text(summary["paired_comparisons"]["pseudo_boundary_neighborhood/5/disagreement/entropy"]["ap_all_errors"])),
              "Q3. 组合改善5–10%预算覆盖，却降低相对D的AP与1–2%覆盖，因此并非全面优于两个单指标。不能依据一个未校正CI宣称组合普遍显著更好；固定参数、不在GT上重复调参。预定组合升级条件是否满足：{}。".format(recommendation["combination_supported"]),
              "Q4. 预测边界邻域外有 {} 个错误、其中 {} 个属于评价边界错误。候选域限制会直接漏掉这些错误，风险高低也无法把域外voxel选入。".format(totals["errors_outside_predicted_neighborhood"], totals["boundary_errors_outside_predicted_neighborhood"]),
              "Q5. 高置信度错误 {}，其中patch定义的高置信farFP {}；保守core farFP {}。以上表格完整报告漏检；没有把这些目标排除后夸大定位效果。".format(
              totals["high_confidence_error_voxels"], totals["high_confidence_far_fp_voxels"], totals["far_fp_core_voxels"]),
              "Q6. 当前探索性候选为 **{}**，优先用于1–2%小预算排序，依据共同域AP与prior差、≥6/8病例配对方向、全域边界Recall、Top10%高置信度覆盖以及实现复杂度。预定规则先选简单H；只有D或组合对其共同域AP配对CI为正、至少6例更好且不降低5%边界Recall时升级；若相对prior不稳定，或至少50个高置信错误/远FP的Top10%平均覆盖不足50%，则选择C。这是探索门槛，不是临床阈值。组合的5–10%覆盖优势应作为未来预算对照保留，这一候选尚不是独立验证的训练mask。".format(names[recommendation["candidate_method"]]),
              "Q7. 选择 **{}**：{}。本轮不实现R1或任何特征匹配模块。".format(recommendation["choice"], recommendation["interpretation"]), "",
              "## 事实、解释与局限", "",
              "事实是冻结Teacher在已见过的24个固定patch上的错误排序、预算富集与漏检。可能的解释包括边界概率不确定、翻转相位/卷积采样敏感，以及过度自信的错误；这些不是经过证明的失效机制。后续区域交互是否提升半监督训练仍是假设。",
              "距离为voxel，无未经验证的physical spacing。crop造成的上下文缺失和farFP定义偏差已分开记录；共同预测边界域只覆盖模型当前轮廓附近。单seed、8病例、训练数据复用、patch重叠及探索性多指标比较限制外推。离线指标选择不能替代按病例独立开发/验证，bootstrap无法消除数据复用。",
              "所有定义明确的统计使用病例级CI；缺失目标保留case/patch记录与原因，有效病例数可小于8；不把不定义的远FPRecall填0/1。",
              "保护检查：{}；已有{}个文件SHA256未变，checkpoint/参数/BN buffers未变；所有评分及候选文件在读取GT前封存且之后哈希不变。全部24patch、8病例保留；单元测试及审计见verification.json、test_results.json、data_access_log.csv。".format(verification["status"], verification["protected_file_count"]), "",
              "## 新增仓库路径", "",
              "代码：analysis/experiment_receiver_R0.py、analysis/test_experiment_receiver_R0.py。结果相对仓库根路径：" + str(out.relative_to(ROOT)) + "/。",
              "该目录下experiment_report.md、patch_level_metrics.csv、case_level_metrics.csv、summary_statistics.json、error_localization_comparison.csv、far_false_positive_analysis.csv、receiver_error_maps/、visualizations/、experiment_metadata.json、verification.json、analysis_log.txt、test_results.json，以及phaseA_seal.json、data_access_log.csv、patch_status.csv。",
              "这里只生成本地仓库文件，没有声称已经推送或发布到GitHub。"]
    (out / "experiment_report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError("Refusing existing output directory: " + str(out))
    history_path = Path(args.reference_dir) / "experiment_metadata.json"
    history = json.loads(history_path.read_text())
    ids, checkpoint = validate_history(history, args.model_path)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(args.device)  # Fail before creating outputs if CUDA denied.
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = create_output(out)
    for name in ("receiver_error_maps", "visualizations"):
        (out / name).mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(str(out / "analysis_log.txt")), logging.StreamHandler(sys.stdout)])
    before = protected_manifest(out)
    metadata = dict(status="started", experiment="Receiver R0", configuration=vars(args),
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), case_order=ids,
                    checkpoint=checkpoint, historical_metadata_sha256=sha256(history_path), data_sources=history["data_sources"],
                    crop_coordinates=history["sampling"]["patch_coordinates"], code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in NEW_CODE},
                    protected_file_sha256_before=before, protocol=dict(methods=METHODS, budgets=BUDGETS,
                    r_voxels=RADIUS, tau_voxels=TAU, far_fp_distance_voxels=FAR_DISTANCE, high_confidence=CONFIDENCE,
                    perturbations=["flip_x", "flip_y", "flip_z"], D_normalization="identity: D already in [0,1]",
                    bootstrap_repetitions=BOOTSTRAPS, bootstrap_unit="case", visual_cases=[0, 3, 7], visual_patch=0, visual_z=40,
                    morphology="26-neighbor cube; erode outside treated foreground, dilate outside background; no artificial crop-face transition",
                    no_GT_in_phase_A=True, no_BGS_no_modulation=True, missing_targets="null/empty with reasons, never automatic success"),
                    environment=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version,
                                     device=args.device, device_name=torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else "CPU"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    metadata["git_commit"] = git.stdout.strip() if not git.returncode else None
    write_json(out / "experiment_metadata.json", metadata)
    try:
        tests = subprocess.run([sys.executable, "-B", str(ROOT / "analysis/test_experiment_receiver_R0.py")], capture_output=True, text=True)
        write_json(out / "test_results.json", dict(exit_code=tests.returncode, stdout=tests.stdout, stderr=tests.stderr))
        if tests.returncode:
            raise AssertionError("R0 unit tests failed: " + tests.stderr)
        model = VNet(n_channels=1, n_classes=2).to(args.device)
        extra = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
        model.load_state_dict(torch.load(args.model_path, map_location="cpu", **extra), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        reader = HistoricalReader(history)
        artifacts, predictions_checks, budget_checks = [], [], []
        image_min, image_max = float("inf"), -float("inf")
        for i, case_id in enumerate(ids):
            for patch_index in range(3):
                image = reader.read(case_id, patch_index, "image")
                image_min, image_max = min(image_min, float(image.min())), max(image_max, float(image.max()))
                p, aligned, check = teacher_predictions(model, image, args.device)
                seed = args.seed + i * 1009 + patch_index * 9176
                maps, b, distance, neighborhood, present = construct_risks(p, aligned, seed)
                candidates, counts = build_candidates(maps, neighborhood, seed + 10000079)
                path = out / "receiver_error_maps" / (case_id + "_p" + str(patch_index) + "_scores_candidates.npz")
                np.savez_compressed(str(path), image=image, probability=p, aligned_flip_probabilities=aligned,
                                    pseudo=p > 0.5, pseudo_boundary=b, pseudo_boundary_distance=distance,
                                    pseudo_boundary_neighborhood=neighborhood, pseudo_boundary_present=np.array(present),
                                    **{"risk_" + k: v for k, v in maps.items()}, **candidates)
                artifacts.append(dict(case_id=case_id, patch_index=patch_index, path=str(path), sha256=sha256(path),
                                      crop=reader.coords[case_id, patch_index], probability_shape=list(p.shape), pseudo_boundary_present=present))
                predictions_checks.append(dict(case_id=case_id, patch_index=patch_index, **check))
                budget_checks.extend(dict(case_id=case_id, patch_index=patch_index, **c) for c in counts)
                logging.info("Phase A saved %s patch%d: pseudo-neighborhood=%d, GT reads=%d", case_id, patch_index,
                             int(neighborhood.sum()), sum(e["dataset"] == "label" for e in reader.events))
        changed_state = [k for k, v in state_before.items() if not torch.equal(v, model.state_dict()[k].detach().cpu())]
        if changed_state or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise AssertionError("Frozen EMA parameters/buffers/eval state changed")
        write_json(out / "phaseA_seal.json", dict(phase="A_complete", GT_dataset_reads=0, artifacts=artifacts, protocol=metadata["protocol"]))
        reader.seal(artifacts)
        metadata["phaseA_artifacts"] = artifacts
        metadata["image_display_range"] = [image_min, image_max]
        metadata["no_aug_prediction_checks"] = predictions_checks
        write_csv(out / "candidate_budget_checks.csv", budget_checks)
        logging.info("All 24 score/candidate maps sealed; first GT access now permitted")
        rows, far_rows, patch_status, first_confident = [], [], [], None
        for artifact in artifacts:
            case_id, patch_index = artifact["case_id"], artifact["patch_index"]
            gt = reader.read(case_id, patch_index, "label")
            if int(gt.sum()) != reader.coords[case_id, patch_index]["foreground_voxels"]:
                raise AssertionError("Historical GT foreground count differs")
            with np.load(artifact["path"]) as archive:
                metrics, far, targets, status = evaluate_patch(case_id, patch_index, archive, gt)
                if first_confident is None and targets["high_confidence_errors"].any():
                    voxel = np.argwhere(targets["high_confidence_errors"])[0].tolist()
                    first_confident = dict(case_id=case_id, patch_index=patch_index, voxel=voxel,
                                           rule="first case/patch in fixed order, then first row-major high-confidence error")
                np.savez_compressed(str(out / "receiver_error_maps" / (case_id + "_p" + str(patch_index) + "_evaluation.npz")),
                                    **{k: v for k, v in targets.items() if isinstance(v, np.ndarray)})
            rows.extend(metrics)
            far_rows.extend(far)
            patch_status.append(status)
            logging.info("Phase B %s patch%d: errors=%d confident=%d farFP=%d", case_id, patch_index,
                         status["all_error_voxels"], status["high_confidence_error_voxels"], status["far_fp_voxels"])
        cases, paired, summary = aggregate(rows, ids, args.seed)
        summary["recommendation"] = choose_recommendation(summary, patch_status)
        metadata["high_confidence_example"] = first_confident
        for artifact in artifacts:
            fixed = ids.index(artifact["case_id"]) in (0, 3, 7) and artifact["patch_index"] == 0
            high = first_confident and (artifact["case_id"], artifact["patch_index"]) == (first_confident["case_id"], first_confident["patch_index"])
            if fixed or high:
                with np.load(out / "receiver_error_maps" / (artifact["case_id"] + "_p" + str(artifact["patch_index"]) + "_evaluation.npz")) as ev:
                    targets = {k: ev[k] for k in ev.files}
                if fixed:
                    visualize(out, artifact, targets, summary["recommendation"]["candidate_method"], "preselected", 40, [image_min, image_max])
                if high:
                    visualize(out, artifact, targets, summary["recommendation"]["candidate_method"], "first_confident_error", first_confident["voxel"][2], [image_min, image_max], first_confident["voxel"])
        for filename, data in (("patch_level_metrics.csv", rows), ("case_level_metrics.csv", cases),
                               ("error_localization_comparison.csv", paired), ("far_false_positive_analysis.csv", far_rows),
                               ("data_access_log.csv", reader.events), ("patch_status.csv", patch_status)):
            write_csv(out / filename, data)
        after = protected_manifest(out)
        changed_files = [p for p, h in before.items() if after.get(p) != h]
        if changed_files or set(after) != set(before):
            raise AssertionError("Existing project files changed: " + str(changed_files))
        if any(sha256(a["path"]) != a["sha256"] for a in artifacts):
            raise AssertionError("Image-only Phase-A artifacts changed after GT evaluation")
        verification = dict(status="passed", protected_files_unchanged=True, protected_file_count=len(before),
                            checkpoint_sha256_unchanged=sha256(args.model_path) == checkpoint["sha256"],
                            model_state_tensor_count=len(state_before), model_parameters_and_BN_buffers_unchanged=True,
                            all_modules_eval=True, all_model_parameters_frozen=True, no_model_parameter_gradients=True,
                            no_training_no_optimizer_no_EMA=True, no_backward=True, no_BGS_no_feature_modulation=True,
                            no_fake_student_teacher_comparison=True, test_GT_read=False, phase_A_GT_reads=0,
                            all_phase_A_maps_saved_before_first_GT=True, phase_A_map_hashes_unchanged=True,
                            no_reselection_of_patches=True, patches=24, cases=8,
                            same_domain_budgets_across_methods=True, bootstrap_unit="case", bootstrap_repetitions=BOOTSTRAPS,
                            all_outputs_finite=True, undefined_values_explicit=True, unit_test_exit_code=tests.returncode,
                            no_aug_forward_checks=predictions_checks, flip_inverse_unit_test_passed=True)
        metadata.update(status="complete", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        protected_file_sha256_after=after)
        report(out, metadata, summary, cases, patch_status, verification)
        write_json(out / "summary_statistics.json", summary)
        write_json(out / "experiment_metadata.json", metadata)
        write_json(out / "verification.json", verification)
        logging.info("R0 complete: %s; %d protected files unchanged; recommendation=%s", out, len(before), summary["recommendation"])
    except Exception as e:
        logging.exception("R0 failed; partial artifacts retained; no overwrite on rerun")
        metadata.update(status="failed", error=repr(e))
        write_json(out / "experiment_metadata.json", metadata)
        write_json(out / "verification.json", dict(status="failed", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
