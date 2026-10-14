"""R1-S0: frozen, fold-sealed, cross-case local prototype transfer analysis.

Only decoder-input x3 is changed. No VNet training, BGS gain or GT-driven
Receiver selection. Literal label swap is invariant under competitive cosine
matching; an explicitly authorized opposite-prototype control tests direction.
"""
import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import h5py
import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
import experiment_bgs_utility as u
import experiment_receiver_R0 as r0
import experiment_receiver_R05 as r05

SEED = 42
ALPHAS = (.02, .05)
MAPPINGS = ("max", "trilinear")
REPETITIONS = 20
BOOTSTRAPS = 10000
PURITY = .75
RADIUS = 3.
CAP = .10
Q_SCALE = .20
Q_MIN = .001
EPS = 1e-6
MIN_CANDIDATES = 8
NEW_CODE = {Path(__file__).resolve(), HERE / "test_experiment_R1_S0.py"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default=str(ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth"))
    p.add_argument("--reference_dir", default=str(ROOT / "analysis/results/experiment_bgs_utility/run_cuda_20261009_1713"))
    p.add_argument("--R05_dir", default=str(ROOT / "analysis/results/experiment_receiver_R05/run_20261013_150525"))
    p.add_argument("--D0_dir", default=str(ROOT / "analysis/results/experiment_donor_D0/run_20261013_174351"))
    p.add_argument("--output_dir", default=str(ROOT / "analysis/results/experiment_R1_S0" / datetime.datetime.now(r0.TZ).strftime("run_%Y%m%d_%H%M%S")))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    return p.parse_args(argv)


def protected_manifest(out):
    result = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache") and (Path(parent) / d).resolve() != out.resolve())
        for name in sorted(files):
            path = Path(parent) / name
            if path.is_file() and path.resolve() not in NEW_CODE and path.suffix not in (".pyc", ".pyo"):
                result[str(path.relative_to(ROOT))] = r0.sha256(path)
    return result


def create_output(path):
    out = Path(path).resolve()
    if out.parent != ROOT / "analysis/results/experiment_R1_S0" or not out.name.startswith("run_"):
        raise ValueError("A fresh run_TIMESTAMP child is required")
    out.mkdir(parents=True, exist_ok=False)
    return out


class FoldReader:
    """No Receiver label access before every strategy in that fold is sealed.

    Earlier Donor access in another fold never bypasses the current role gate.
    Labels are not cached; only image-derived features may be cached globally.
    """
    def __init__(self, history):
        self.sources = {s["case_id"]: s for s in history["data_sources"]}
        self.coords = {(c["case_id"], c["patch_index"]): c for c in history["sampling"]["patch_coordinates"]}
        self.events = []
        self.fold = None
        self.receiver = None
        self.donors = []
        self.sealed = False

    def start_fold(self, fold, receiver, donors):
        if receiver in donors or len(set(donors)) != 7:
            raise ValueError("LOCO must have seven distinct other-case donors")
        self.fold, self.receiver, self.donors, self.sealed = fold, receiver, list(donors), False

    def read(self, case, patch, field):
        if field not in ("image", "label"):
            raise ValueError("Only image and label datasets allowed")
        if field == "label":
            if self.fold is None:
                raise RuntimeError("No label access before explicit fold roles")
            if case == self.receiver and not self.sealed:
                raise RuntimeError("Receiver GT locked until ALL fold Phase-A predictions are sealed")
            if case != self.receiver and case not in self.donors:
                raise RuntimeError("Label source is outside current fold")
        with h5py.File(self.sources[case]["path"], "r") as f:
            array = f[field][:]
        arr = r0.replay_array(array, self.coords[case, patch], [112, 112, 80], np.float32 if field == "image" else np.int64)
        r0.finite(arr, case + "/" + field)
        if field == "label" and not np.isin(arr, [0, 1]).all():
            raise ValueError("Nonbinary GT")
        self.events.append(dict(event_index=len(self.events), fold=self.fold, case_id=case, patch_index=patch,
            dataset=field, role="receiver" if case == self.receiver else "donor" if case in self.donors else "image_cache",
            phase="B" if self.sealed else "A", fold_sealed=self.sealed))
        return arr

    def seal(self, artifacts, expected_predictions):
        if not artifacts or sum(a.get("kind") == "prediction" for a in artifacts) != expected_predictions:
            raise RuntimeError("Incomplete Phase-A prediction set")
        for a in artifacts:
            if not Path(a["path"]).is_file() or r0.sha256(a["path"]) != a["sha256"]:
                raise RuntimeError("Phase-A digest mismatch")
        self.sealed = True


def grid_min(array):
    a = np.asarray(array)
    if a.ndim != 3 or any(s % 4 for s in a.shape):
        raise ValueError("Original spatial shape must match fixed 4-cubed grids")
    return a.reshape(a.shape[0] // 4, 4, a.shape[1] // 4, 4, a.shape[2] // 4, 4).min((1, 3, 5))


def donor_candidates(mask):
    """Only Donor GT. Boundary proximity means minimum distance in each grid."""
    mask = np.asarray(mask, dtype=bool)
    b = r0.boundary(mask)
    fraction = r05.block_count(mask) / 64.
    if b.any():
        distance = ndi.distance_transform_edt(~b)
    else:
        distance = np.full(mask.shape, np.linalg.norm(np.array(mask.shape)) + 1.)
    minimum = grid_min(distance)
    near = (minimum <= RADIUS) & bool(b.any())
    pure = (fraction >= PURITY, fraction <= 1 - PURITY)
    return dict(foreground_fraction=fraction.astype(np.float32), minimum_boundary_distance=minimum.astype(np.float32),
        intersects_boundary=r05.block_count(b) > 0, boundary_fg=pure[0] & near, boundary_bg=pure[1] & near,
        nonboundary_fg=pure[0] & ~near, nonboundary_bg=pure[1] & ~near, boundary_present=bool(b.any()))


def cpu_generator(seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def sample_rows(array, n, seed):
    idx = torch.randperm(len(array), generator=cpu_generator(seed))[:n].numpy()
    return array[idx]


def make_case_library(case, case_index, xs, masks):
    pools = defaultdict(list)
    grids = []
    for patch, (x, mask) in enumerate(zip(xs, masks)):
        candidates = donor_candidates(mask)
        vectors = x.reshape(x.shape[0], -1).T
        if tuple(x.shape[1:]) != candidates["foreground_fraction"].shape:
            raise ValueError("GT grid and actual x3 coordinates disagree")
        for key in ("boundary_fg", "boundary_bg", "nonboundary_fg", "nonboundary_bg"):
            pools[key].append(vectors[candidates[key].ravel()])
        grids.append(candidates)
    pools = {k: np.concatenate(v, 0) for k, v in pools.items()}
    counts = {k: len(v) for k, v in pools.items()}
    valid = all(counts[k] >= MIN_CANDIDATES for k in ("boundary_fg", "boundary_bg"))
    nonvalid = all(counts[k] >= MIN_CANDIDATES for k in pools)
    prototypes = {}
    if valid:
        prototypes["boundary"] = np.stack([pools["boundary_" + side].mean(0, dtype=np.float64) for side in ("fg", "bg")]).astype(np.float32)
    matched_counts = []
    if nonvalid:
        boundary_matched, nonboundary = [], []
        for ci, side in enumerate(("fg", "bg")):
            n = min(counts["boundary_" + side], counts["nonboundary_" + side])
            matched_counts.append(n)
            for key, destination in (("boundary_", boundary_matched), ("nonboundary_", nonboundary)):
                selection = sample_rows(pools[key + side], n, SEED + case_index * 1009 + ci * 97 + (70001 if key == "boundary_" else 80021))
                destination.append(selection.mean(0, dtype=np.float64))
        prototypes["boundary_count_matched"] = np.array(boundary_matched, dtype=np.float32)
        prototypes["nonboundary"] = np.array(nonboundary, dtype=np.float32)
    all_vectors = np.concatenate([x.reshape(x.shape[0], -1).T for x in xs], 0).astype(np.float64)
    return dict(case_id=case, counts=counts, matched_counts=matched_counts, valid=valid, nonboundary_valid=nonvalid,
        prototypes=prototypes, mean=all_vectors.mean(0), second=(all_vectors ** 2).mean(0), grids=grids,
        raw_channel_std=all_vectors.std(0))


def combine_libraries(libraries, receiver):
    if receiver in libraries or len(libraries) != 7:
        raise ValueError("Receiver contamination of prototype/scaler")
    mean = np.mean([v["mean"] for v in libraries.values()], 0)
    second = np.mean([v["second"] for v in libraries.values()], 0)
    scale = np.sqrt(np.maximum(second - mean ** 2, 0)).clip(EPS)
    result = dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32), prototypes={}, sources={})
    for kind in ("boundary", "boundary_count_matched", "nonboundary"):
        valid = [v for v in libraries.values() if kind in v["prototypes"]]
        # Do not quietly replace the prescribed seven-case prototype by fewer cases.
        if len(valid) == 7:
            result["prototypes"][kind] = np.mean([v["prototypes"][kind] for v in valid], 0).astype(np.float32)
        result["sources"][kind] = [v["case_id"] for v in valid]
    return result


def self_prototypes(x, probability):
    # Reliable-side purity and pseudo-boundary neighborhood use prediction only.
    c = donor_candidates(probability > .5)
    v = x.reshape(x.shape[0], -1).T
    selected = [v[c["boundary_" + side].ravel()] for side in ("fg", "bg")]
    if any(len(a) < MIN_CANDIDATES for a in selected):
        return None, [len(a) for a in selected]
    return np.stack([a.mean(0, dtype=np.float64) for a in selected]).astype(np.float32), [len(a) for a in selected]


def match(x, prototypes, mean, scale):
    """No Receiver labels accepted; raw targets and matching space are separate."""
    shape = x.shape[1:]
    if prototypes is None:
        return dict(target=np.zeros_like(x), q=np.zeros(shape, np.float32), chosen=np.zeros(shape, np.uint8),
            fg_similarity=np.zeros(shape, np.float32), bg_similarity=np.zeros(shape, np.float32),
            margin=np.zeros(shape, np.float32), valid=np.zeros(shape, bool), reason="invalid_prototype_library")
    v = x.reshape(x.shape[0], -1).T.astype(np.float64)
    p = prototypes.astype(np.float64)
    z = (v - mean) / scale
    zp = (p - mean) / scale
    vn, pn = np.linalg.norm(z, axis=1), np.linalg.norm(zp, axis=1)
    valid = (vn > EPS) & (np.linalg.norm(v, axis=1) > EPS) & bool(np.all(pn > EPS))
    sims = z @ zp.T / (np.maximum(vn[:, None], EPS) * np.maximum(pn[None], EPS))
    chosen = np.argmax(sims, axis=1)  # 0 = foreground side, 1 = background side
    margin = np.abs(sims[:, 0] - sims[:, 1])
    q = np.clip(margin / Q_SCALE, 0, 1)
    valid &= q > Q_MIN
    q[~valid] = 0
    target = p[chosen].T.reshape(x.shape).astype(np.float32)
    result = dict(target=target, q=q.reshape(shape).astype(np.float32), chosen=chosen.reshape(shape).astype(np.uint8),
        fg_similarity=sims[:, 0].reshape(shape).astype(np.float32), bg_similarity=sims[:, 1].reshape(shape).astype(np.float32),
        margin=margin.reshape(shape).astype(np.float32), valid=valid.reshape(shape), reason=None)
    for key, value in result.items():
        if isinstance(value, np.ndarray): r0.finite(value, "match/" + key)
    return result


def apply_residual(x, target, mask, q, alpha, energy_target=None, channels=None):
    """Raw-space correction, then per-position cap; energy matching has no GT.

    Equal-energy controls preserve their own Q=0 skips and residual direction.
    A scalar matches total residual L2 to the true method when feasible. The
    cap may make that energy unattainable; this is explicitly reported.
    """
    if x.shape != target.shape or x.shape[1:] != mask.shape or q.shape != mask.shape:
        raise ValueError("Residual coordinate/shape mismatch")
    raw = alpha * mask[None] * q[None] * (target - x)
    if channels is not None:
        keep = np.zeros(x.shape[0], bool)
        keep[np.asarray(channels, int)] = True
        raw *= keep[:, None, None, None]
    norms = np.linalg.norm(x.astype(np.float64), axis=0)
    delta_norm = np.linalg.norm(raw.astype(np.float64), axis=0)
    limit = CAP * norms
    limit[norms <= EPS] = 0
    factor = np.ones(mask.shape, dtype=np.float64)
    nonzero = delta_norm > 0
    factor[nonzero] = np.minimum(1., limit[nonzero] / delta_norm[nonzero])
    multiplier = 1.
    feasible = True
    if energy_target is not None:
        active = nonzero & (limit > 0)
        maximum = float(np.linalg.norm(limit[active]))
        feasible = float(energy_target) <= maximum + 1e-7
        wanted = min(float(energy_target), maximum)
        if wanted == 0:
            multiplier = 0.
        elif not active.any():
            multiplier = 0.
        else:
            low, high = 0., max(1., float(np.max(limit[active] / delta_norm[active])))
            for _ in range(60):
                mid = (low + high) / 2
                energy = np.linalg.norm(np.minimum(mid * delta_norm[active], limit[active]))
                if energy < wanted: low = mid
                else: high = mid
            multiplier = (low + high) / 2
        factor[nonzero] = np.minimum(multiplier, limit[nonzero] / delta_norm[nonzero])
        factor[~nonzero] = 0
    delta = (raw.astype(np.float64) * factor[None]).astype(np.float32)
    new = x + delta
    # Statistics and cap refer to the actual float32 feature modification.
    actual = new.astype(np.float64) - x.astype(np.float64)
    actual_norm = np.linalg.norm(actual, axis=0)
    relative = actual_norm / np.maximum(norms, EPS)
    if np.any(relative > CAP + 2e-6) or np.any(actual[:, ~mask] != 0):
        raise AssertionError("Residual escaped mask or L2 cap")
    r0.finite(new, "modified x3")
    n = int(mask.sum())
    stats = dict(mask_positions=n, changed_positions=int(np.count_nonzero(actual_norm)),
        residual_l2=float(np.linalg.norm(actual)), relative_l2=float(np.linalg.norm(actual) / (np.linalg.norm(x) + EPS)),
        maximum_position_relative_l2=float(relative.max()), mean_position_relative_l2=float(relative[mask].mean()) if n else 0.,
        clipped_fraction=float(np.mean(delta_norm[mask] * multiplier > limit[mask])) if n else 0.,
        energy_target=energy_target, energy_match_feasible=feasible, energy_multiplier=multiplier,
        energy_match_relative_error=float(abs(np.linalg.norm(actual) - energy_target) / (energy_target + EPS)) if energy_target is not None else None,
        q_mean=float(q[mask].mean()) if n else 0., q_max=float(q[mask].max()) if n else 0.,
        q_near_zero=int(np.count_nonzero(q[mask] <= Q_MIN)), match_success_fraction=float(np.mean(q[mask] > Q_MIN)) if n else 0.)
    return new, stats


def replay_features(features, changed_x3):
    modified = list(features)
    modified[2] = torch.as_tensor(changed_x3, device=features[2].device, dtype=features[2].dtype)[None]
    if modified[2].shape != features[2].shape:
        raise ValueError("Decoder x3 replacement shape differs")
    assert all(modified[i] is features[i] for i in (0, 1, 3, 4))
    return modified


def random_mask(shape, count, seed):
    flat = np.zeros(int(np.prod(shape)), bool)
    flat[torch.randperm(len(flat), generator=cpu_generator(seed))[:count].numpy()] = True
    return flat.reshape(shape)


def risk_mask(score, case_index, patch):
    tie_seed = SEED + case_index * 1009 + patch * 9176 + 20000003
    ties = np.random.RandomState(tie_seed).permutation(score.size)
    order = r0.ordered_indices(score, np.ones(score.shape, bool), ties)
    count = r0.budget_count(len(order), .02)
    flat = np.zeros(score.size, bool)
    flat[order[:count]] = True
    return flat.reshape(score.shape)


def check_historical_risk(image, probability, aligned, x3_shape, artifact, case_index, patch):
    d = r05.direction_maps(probability, aligned)["Dxyz"]
    original = artifact["original_R0"]
    if r0.sha256(original["path"]) != original["sha256"] or r0.sha256(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Historical image-only archive changed")
    with np.load(original["path"]) as old:
        if not np.array_equal(image, old["image"]): raise AssertionError("Historical image crop differs")
        if not np.array_equal(probability, old["probability"]): raise AssertionError("Teacher p0 differs from R0")
        if not np.array_equal(aligned, old["aligned_flip_probabilities"]): raise AssertionError("Flip outputs differ from R0")
        if not np.array_equal(d, old["risk_disagreement"]): raise AssertionError("Dxyz differs from R0")
    maps, masks = {}, {}
    with np.load(artifact["path"]) as old:
        for mapping in MAPPINGS:
            maps[mapping] = r05.map_to_grid(d, x3_shape, mapping)
            masks[mapping] = risk_mask(maps[mapping], case_index, patch)
            if not np.array_equal(maps[mapping], old["risk_" + mapping + "_Dxyz"]):
                raise AssertionError("Mapped D differs from R0.5")
            if not np.array_equal(masks[mapping], r05.unpack(old, "x3", mapping, "global", "Dxyz", .02, x3_shape)):
                raise AssertionError("Top2 mask differs from R0.5")
    return d, maps, masks


def artifact(path, kind):
    return dict(path=str(path.resolve()), sha256=r0.sha256(path), kind=kind)


def summarize(values, seed=42):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(a): return dict(n_cases=0, mean=None, median=None, std=None, ci95=None, reason="no_defined_case_values")
    bootstrap = np.random.RandomState(seed).choice(a, (BOOTSTRAPS, len(a)), replace=True).mean(1)
    return dict(n_cases=len(a), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=1)) if len(a)>1 else 0.,
        ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(), positive_cases=int((a>0).sum()), negative_cases=int((a<0).sum()),
        zero_cases=int((a==0).sum()), minimum=float(a.min()), maximum=float(a.max()))


def mean_defined(values):
    a = [v for v in values if v is not None]
    return float(np.mean(a)) if a else None


def evaluation_context(gt):
    y = torch.from_numpy(gt)[None]
    inner, outer = u.boundary_sides(y)
    truth = gt.astype(bool)
    face = np.ones(truth.shape, bool)
    for axis, size in enumerate(truth.shape):
        d = np.minimum(np.arange(size), size-1-np.arange(size))
        face &= (d > 20).reshape(tuple(size if j==axis else 1 for j in range(3)))
    far_domain = ~truth & (ndi.distance_transform_edt(~truth) > 20) if truth.any() else np.zeros(truth.shape, bool)
    return dict(y=y,inner=inner,outer=outer,truth=truth,far_domain=far_domain,face=face,fraction=r05.block_count(truth)/64.)


@torch.no_grad()
def evaluate_prediction(logits, probability, pred, gt, baseline_pred, baseline_probability, mask, matching, context=None):
    """Phase B only. Original loss helpers and explicit correct/harm signs."""
    context = evaluation_context(gt) if context is None else context
    y,inner,outer = context['y'],context['inner'],context['outer']
    if not inner.any() or not outer.any():
        raise ValueError("Historical patch no longer has both boundary sides")
    z = torch.from_numpy(logits)[None]
    losses = u.loss_values(z, y, inner, outer)
    ce = F.cross_entropy(z, y, reduction="none")
    result = {k: float(v.item()) for k, v in losses.items()}
    result.update(inner_ce=float(ce[inner].double().mean()), outer_ce=float(ce[outer].double().mean()),
                  ce_plus_dice_sum=2 * result["supervised_ce_dice"])
    truth = context['truth']
    errors = baseline_pred != truth
    changed = pred != truth
    corrected = errors & ~changed
    harmed = ~errors & changed
    intersection = int(np.count_nonzero(pred & truth))
    union = int(np.count_nonzero(pred | truth))
    result.update(dice=2*intersection / (int(pred.sum()) + int(truth.sum())) if pred.any() or truth.any() else 1.,
        jaccard=intersection/union if union else 1., fp_voxels=int(np.count_nonzero(pred & ~truth)),
        fn_voxels=int(np.count_nonzero(~pred & truth)), corrected=int(corrected.sum()), harmed=int(harmed.sum()),
        net_correction=int(corrected.sum())-int(harmed.sum()), corrected_rate=r05.ratio(corrected.sum(), errors.sum()),
        harmed_rate=r05.ratio(harmed.sum(), (~errors).sum()), probability_mean_absolute_change=float(np.mean(np.abs(probability-baseline_probability))),
        probability_max_absolute_change=float(np.max(np.abs(probability-baseline_probability))))
    lifted = r05.lift_grid(mask)
    result.update(mask_error_voxels=int(np.count_nonzero(errors & lifted)), mask_correct_voxels=int(np.count_nonzero(~errors & lifted)))
    for location, domain in (("inside_mask", lifted), ("outside_mask", ~lifted), ("inner_boundary", inner[0].numpy()), ("outer_boundary", outer[0].numpy())):
        result["corrected_"+location] = int(np.count_nonzero(corrected & domain))
        result["harmed_"+location] = int(np.count_nonzero(harmed & domain))
        result["probability_change_"+location] = float(np.mean(np.abs(probability-baseline_probability)[domain])) if domain.any() else None
    far_valid = bool(truth.any())
    if far_valid:
        far_domain,face = context['far_domain'],context['face']
        for name, domain in (("far_fp",far_domain), ("far_fp_core",far_domain & face), ("far_fp_crop_sensitive",far_domain & ~face)):
            result[name] = int(np.count_nonzero(pred & domain))
            result[name+"_change"] = result[name] - int(np.count_nonzero(baseline_pred & domain))
        result["high_confidence_far_fp"] = int(np.count_nonzero(pred & far_domain & (probability >= .95)))
        result["high_confidence_far_fp_change"] = result["high_confidence_far_fp"] - int(np.count_nonzero(baseline_pred & far_domain & (baseline_probability >= .95)))
    else:
        for name in ("far_fp", "far_fp_core", "far_fp_crop_sensitive", "high_confidence_far_fp"):
            result[name] = result[name+"_change"] = None
    result["far_fp_reason"] = None if far_valid else "empty_patch_GT_distance_undefined"
    result["high_confidence_fp"] = int(np.count_nonzero(pred & ~truth & (probability >= .95)))
    fraction = context['fraction']
    pseudo_fraction = r05.block_count(baseline_pred) / 64.
    chosen_fg = matching["chosen"] == 0
    conflict = chosen_fg != (pseudo_fraction >= .5)
    for name, cells in (("match_conflict", conflict & mask), ("match_agreement", ~conflict & mask)):
        domain = r05.lift_grid(cells)
        result[name+"_positions"] = int(cells.sum())
        result[name+"_original_error_fraction"] = r05.ratio(np.count_nonzero(errors & domain), domain.sum())
        result[name+"_corrected"] = int(np.count_nonzero(corrected & domain))
        result[name+"_harmed"] = int(np.count_nonzero(harmed & domain))
    pure = (fraction >= .75) | (fraction <= .25)
    result["match_class_accuracy_pure_grids"] = r05.ratio(np.count_nonzero((chosen_fg == (fraction >= .75)) & pure & mask), np.count_nonzero(pure & mask))
    result["match_class_evaluation_pure_grids"] = int(np.count_nonzero(pure & mask))
    for lo, hi in ((0.,.25),(.25,.5),(.5,.75),(.75,1.000001)):
        domain = r05.lift_grid(mask & (matching["q"] >= lo) & (matching["q"] < hi))
        key = "qbin_{}".format(int(lo*100))
        result[key+"_voxels"] = int(domain.sum())
        result[key+"_corrected"] = int(np.count_nonzero(corrected & domain))
        result[key+"_harmed"] = int(np.count_nonzero(harmed & domain))
        result[key+"_original_errors"] = int(np.count_nonzero(errors & domain))
    for k,v in result.items():
        if isinstance(v,(int,float)): r0.finite(v,"metric/"+k)
    return result


def aggregate(rows, ids):
    numeric = [k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ("fold","patch_index","repetition","alpha")]
    case_rows = []
    grouped = defaultdict(list)
    for row in rows: grouped[row["case_id"],row["mapping"],row["method"],row["alpha"]].append(row)
    for (case,mapping,method,alpha), rr in grouped.items():
        patches = defaultdict(list)
        for r in rr: patches[r["patch_index"]].append(r)
        record = dict(case_id=case,mapping=mapping,method=method,alpha=alpha,n_patches=len(patches),n_raw_results=len(rr))
        for metric in numeric:
            record[metric] = mean_defined([mean_defined([r.get(metric) for r in pp]) for pp in patches.values()])
        case_rows.append(record)
    methods = defaultdict(list)
    for r in case_rows: methods[r["mapping"],r["method"],r["alpha"]].append(r)
    summary = {}
    for (mapping,method,alpha), rr in methods.items():
        key = "{}|{}|{}".format(mapping,method,alpha)
        summary[key] = {metric:summarize([r[metric] for r in rr]) for metric in numeric}
        summary[key]["case_ids"] = [r["case_id"] for r in rr]
    lookup = {(r["case_id"],r["mapping"],r["method"],r["alpha"]):r for r in case_rows}
    comparisons = []
    controls = sorted(set(r["method"] for r in case_rows if r["method"] != "true_boundary"))
    for mapping in MAPPINGS:
        for alpha in ALPHAS:
            for control in controls:
                pairs = [(lookup.get((case,mapping,"true_boundary",alpha)),lookup.get((case,mapping,control,alpha))) for case in ids]
                if not all(a is not None and b is not None for a,b in pairs): continue
                for metric,direction in (("boundary_ce",-1),("full_ce",-1),("dice",1),("net_correction",1)):
                    values = [direction*(a[metric]-b[metric]) for a,b in pairs]
                    s = summarize(values)
                    comparisons.append(dict(mapping=mapping,alpha=alpha,method="true_boundary",control=control,metric=metric,
                        advantage_direction="positive favors true_boundary",**s,per_case_advantage={c:v for c,v in zip(ids,values)}))
    return case_rows, summary, comparisons


def load_references(args):
    old_path = Path(args.reference_dir)/"experiment_metadata.json"
    history = json.loads(old_path.read_text())
    ids, checkpoint = r0.validate_history(history,args.model_path)
    rpath = Path(args.R05_dir)/"experiment_metadata.json"
    rmeta = json.loads(rpath.read_text())
    dpath = Path(args.D0_dir)/"experiment_metadata.json"
    dmeta = json.loads(dpath.read_text())
    for name, m in (("R0.5",rmeta),("D0-A",dmeta)):
        if m["status"] != "complete" or m["case_order"] != ids or m["checkpoint"] != checkpoint or m["crop_coordinates"] != history["sampling"]["patch_coordinates"] or m["data_sources"] != history["data_sources"]:
            raise ValueError(name+" differs in checkpoint/crops/source/case identity")
    artifacts = {(a["case_id"],a["patch_index"]):a for a in rmeta["phaseA_artifacts"]}
    if len(artifacts) != 24: raise ValueError("Incomplete historical R0.5 risk maps")
    return history,ids,checkpoint,artifacts,{str(p):r0.sha256(p) for p in (old_path,rpath,dpath)}


def save_prediction(path, logits, baseline_array=None, baseline_path=None):
    """Lossless FP32 bit patterns and native decisions; never quantization.

    Sparse bitwise XOR relative to the sealed No-Correction logits reduces
    disk usage without floating-point subtraction or rounding. Every restored
    bit is identical. Baseline and delta artifacts are both in the fold seal.
    """
    a = logits[0].cpu().numpy()
    p = torch.softmax(logits,dim=1)[0,1]
    pred = (p>.5).cpu().numpy()
    r0.finite(a,"prediction logits")
    if baseline_array is None:
        np.savez_compressed(path,logits=a,prediction_bits=np.packbits(pred))
    else:
        xor=np.bitwise_xor(a.view(np.uint32),np.asarray(baseline_array,dtype=np.float32).view(np.uint32))
        np.savez_compressed(path,logits_xor=xor,baseline_file=np.array(Path(baseline_path).name),prediction_bits=np.packbits(pred))


def read_prediction(path, baseline_array=None):
    path=Path(path)
    with np.load(path) as a:
        if "logits" in a.files:
            logits = a["logits"]
        else:
            if baseline_array is None:
                with np.load(path.parent/str(a["baseline_file"].item())) as b: baseline_array=b['logits']
            logits=np.bitwise_xor(a['logits_xor'],baseline_array.view(np.uint32)).view(np.float32)
        pred = np.unpackbits(a["prediction_bits"])[:int(np.prod(logits.shape[1:]))].reshape(logits.shape[1:]).astype(bool)
    probability = torch.softmax(torch.from_numpy(logits),dim=0)[1].numpy()
    return logits,probability,pred


@torch.no_grad()
def receiver_phaseA(model, reader, images, xs, donor, case, ci, patch, fold_dir, reference, channel_sets=None, repetitions=REPETITIONS):
    image = images[case,patch]
    tensor = torch.from_numpy(image)[None,None].to(next(model.parameters()).device)
    features = model.encoder(tensor)
    original = model(tensor)
    baseline = model.decoder(features)
    if not torch.equal(original,baseline): raise AssertionError("Original forward/decoder replay not exactly equal")
    preserved = [v.clone() for v in features]
    x = features[2][0].cpu().numpy()
    if not np.array_equal(x,xs[case,patch]): raise AssertionError("Image-only feature cache differs")
    p,aligned,_ = r0.teacher_predictions(model,image,tensor.device)
    if not np.array_equal(p,torch.softmax(baseline,1)[0,1].cpu().numpy()): raise AssertionError("p0/replay differs")
    cpu_probability = torch.softmax(baseline[0].cpu(),0)[1].numpy()
    cpu_probability_difference = float(np.abs(p-cpu_probability).max())
    if cpu_probability_difference > 2e-7: raise AssertionError("CPU/GPU replay softmax exceeds storage tolerance")
    disagreement,grids,masks = check_historical_risk(image,p,aligned,x.shape[1:],reference,ci,patch)
    selfp,selfcounts = self_prototypes(x,p)
    protos = dict(donor["prototypes"])
    protos["self"] = selfp
    matches = {kind:match(x,protos.get(kind),donor["mean"],donor["scale"]) for kind in ("boundary","nonboundary","boundary_count_matched","self")}
    true = matches["boundary"]
    if "boundary" in protos:
        swapped = match(x,protos["boundary"][::-1].copy(),donor["mean"],donor["scale"])
        if not np.array_equal(true["target"][:,true['q']>0],swapped["target"][:,true['q']>0]) or not np.array_equal(true["q"],swapped["q"]):
            raise AssertionError("Literal label swap should preserve target/Q (except exact ties)")
    records,artifacts = [],[]
    patch_dir = fold_dir/"phaseA"/(case+"_p"+str(patch))
    patch_dir.mkdir(parents=True)
    basepath = patch_dir/"no_correction.npz"
    save_prediction(basepath,baseline)
    artifacts.append(artifact(basepath,"prediction"))
    payload = dict(image=image,probability=p,aligned_flip_probabilities=aligned,Dxyz=disagreement,x3=x,
                   self_counts=np.array(selfcounts))
    for mapping in MAPPINGS:
        payload["risk_"+mapping]=grids[mapping]
        payload["mask_"+mapping]=masks[mapping]
    for kind,m in matches.items():
        for key in ("q","chosen","fg_similarity","bg_similarity","margin","valid"):
            payload[kind+"_"+key]=m[key]
    matchpath = patch_dir/"matching_maps.npz"
    np.savez_compressed(matchpath,**payload)
    artifacts.append(artifact(matchpath,"image_only_maps"))
    for mapping in MAPPINGS:
        mask = masks[mapping]
        for alpha in ALPHAS:
            common = dict(fold=ci,case_id=case,patch_index=patch,mapping=mapping,alpha=alpha)
            records.append(dict(**common,method="no_correction",repetition=None,path=str(basepath),matching_kind="boundary",
                                mask_seed=None,mask_positions=int(mask.sum()),residual_l2=0.,relative_l2=0.,changed_positions=0))
            true_new,true_stats = apply_residual(x,true["target"],mask,true["q"],alpha)
            target_energy = true_stats["residual_l2"]
            specs = [("true_boundary","boundary",None,False),("opposite_match","boundary",None,False),
                     ("opposite_match_energy_matched","boundary",None,True)]
            for kind in ("nonboundary","boundary_count_matched","self"):
                specs.extend([(kind,kind,None,False),(kind+"_energy_matched",kind,None,True)])
            for repetition in range(repetitions):
                for method in ("random_mask","random_match"):
                    specs.extend([(method,"boundary",repetition,False),(method+"_energy_matched","boundary",repetition,True)])
            if channel_sets:
                specs.extend([(method,"boundary",None,True) for method in channel_sets])
            truepath = None
            for method,kind,repetition,equal_energy in specs:
                m = matches[kind]
                selected = mask
                target = m["target"]
                seed = None
                channels = channel_sets.get(method) if channel_sets else None
                if method.startswith("opposite_match"):
                    if "boundary" in protos: target = protos["boundary"][1-true["chosen"]].transpose(3,0,1,2)
                elif method.startswith("random_mask"):
                    seed = SEED+ci*100003+patch*9176+repetition*101+110000003
                    selected = random_mask(mask.shape,int(mask.sum()),seed)
                elif method.startswith("random_match"):
                    seed = SEED+ci*100003+patch*9176+repetition*101+220000013
                    labels = torch.randint(0,2,mask.shape,generator=cpu_generator(seed)).numpy()
                    if "boundary" in protos: target = protos["boundary"][labels].transpose(3,0,1,2)
                changed,stats = apply_residual(x,target,selected,m["q"],alpha,target_energy if equal_energy else None,channels)
                modified = replay_features(features,changed)
                logits = model.decoder(modified)
                filename = "{}_{}_a{}_r{}.npz".format(mapping,method,str(alpha).replace(".","p"),repetition if repetition is not None else "single")
                path = patch_dir/filename
                save_prediction(path,logits,baseline[0].cpu().numpy(),basepath)
                artifacts.append(artifact(path,"prediction"))
                records.append(dict(**common,method=method,repetition=repetition,path=str(path),matching_kind=kind,
                    mask_seed=seed,matching_map_path=str(matchpath),energy_matched=equal_energy,**stats))
                if method=="true_boundary": truepath=path
            # Save no duplicate prediction: the swapped-label intervention is identical.
            records.append(dict(**common,method="label_swap_invariant",repetition=None,path=str(truepath),matching_kind="boundary",
                                mask_seed=None,matching_map_path=str(matchpath),**true_stats))
    if any(not torch.equal(a,b) for a,b in zip(preserved,features)):
        raise AssertionError("Encoder feature cache was mutated")
    return records,artifacts,dict(case_id=case,patch_index=patch,forward_replay_exact=True,all_other_features_unchanged=True,
        historical_probability_and_D_and_masks_exact=True,x3_shape=list(features[2].shape),label_swap_invariant=True,
        self_candidate_counts=selfcounts,nonfinite_outputs=False,cpu_gpu_probability_max_difference=cpu_probability_difference)


def fold_prototypes(reader, xs, ids, receiver):
    libs = {}
    for case in reader.donors:
        masks = [reader.read(case,p,"label") for p in range(3)]
        libs[case] = make_case_library(case,ids.index(case),[xs[case,p] for p in range(3)],masks)
    return libs,combine_libraries(libs,receiver)


def case_library_rows(libs,donor,fold,receiver):
    rows = []
    for case,lib in libs.items():
        for kind,ps in lib["prototypes"].items():
            for ci,side in enumerate(("fg","bg")):
                rows.append(dict(fold=fold,receiver_case=receiver,donor_case=case,kind=kind,side=side,
                    candidates=lib["counts"]["boundary_"+side] if kind.startswith("boundary") else lib["counts"]["nonboundary_"+side],
                    sampled_count=lib["matched_counts"][ci] if kind!="boundary" else lib["counts"]["boundary_"+side],
                    prototype_l2=float(np.linalg.norm(ps[ci])),prototype=ps[ci].tolist(),valid=True))
        if not lib["valid"]: rows.append(dict(fold=fold,receiver_case=receiver,donor_case=case,valid=False,reason="fewer_than_8_boundary_candidates",counts=lib["counts"]))
    for kind,ps in donor["prototypes"].items():
        for ci,side in enumerate(("fg","bg")):
            rows.append(dict(fold=fold,receiver_case=receiver,donor_case="seven_case_equal_mean",kind=kind,side=side,
                prototype_l2=float(np.linalg.norm(ps[ci])),prototype=ps[ci].tolist(),source_case_count=len(donor["sources"][kind]),valid=True))
    return rows


def evaluate_fold(reader,records,case,images):
    rows = []
    base_cache = {}
    for patch in sorted(set(r["patch_index"] for r in records)):
        gt = reader.read(case,patch,"label")
        context=evaluation_context(gt)
        rr = [r for r in records if r["patch_index"]==patch]
        baseline = read_prediction(next(r["path"] for r in rr if r["method"]=="no_correction"))
        map_path = next(r["matching_map_path"] for r in rr if r["method"]=="true_boundary")
        with np.load(map_path) as a:
            maps={k:a[k] for k in a.files if k!="aligned_flip_probabilities" and k!="x3"}
        metric_cache = {}
        baseline_losses = None
        for r in rr:
            selected = maps["mask_"+r["mapping"]]
            if r["method"].startswith("random_mask"):
                selected = random_mask(selected.shape,int(selected.sum()),r["mask_seed"])
            kind=r["matching_kind"]
            if kind+"_q" not in maps: kind="boundary"
            matching={key:maps[kind+"_"+key] for key in ("q","chosen")}
            if r["method"].startswith("opposite_match"): matching["chosen"]=1-matching["chosen"]
            elif r["method"].startswith("random_match"):
                matching["chosen"]=torch.randint(0,2,selected.shape,generator=cpu_generator(r["mask_seed"])).numpy().astype(np.uint8)
            z,p,pred = baseline if r["method"]=="no_correction" else read_prediction(r["path"],baseline[0])
            if r["method"]=="no_correction" and baseline_losses is None:
                baseline_losses=evaluate_prediction(z,p,pred,gt,baseline[2],baseline[1],selected,matching,context)
            values=evaluate_prediction(z,p,pred,gt,baseline[2],baseline[1],selected,matching,context)
            values.update(boundary_ce_gain=baseline_losses["boundary_ce"]-values["boundary_ce"],
                full_ce_gain=baseline_losses["full_ce"]-values["full_ce"],dice_gain=values["dice"]-baseline_losses["dice"])
            rows.append({**{k:v for k,v in r.items() if k not in ("path","matching_map_path")},**values})
        # Pre-fixed visualizations: cases 0 and 3, patch 0, middle z slice.
        if patch==0 and reader.fold in (0,3):
            representative=next(r for r in rr if r["method"]=="true_boundary" and r["mapping"]=="max" and r["alpha"]==.05)
            _,after,after_pred=read_prediction(representative["path"])
            visualize(Path(map_path).parents[3]/"visualizations",case,images[case,patch],gt,baseline[1],after,
                maps["mask_max"],maps["boundary_chosen"],maps["boundary_q"],baseline[2],after_pred)
    return rows


def visualize(directory,case,image,gt,before,after,mask,chosen,q,pred0,pred1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    z=image.shape[2]//2
    error=pred0!=gt.astype(bool)
    corrected=error & (pred1==gt.astype(bool))
    harmed=~error & (pred1!=gt.astype(bool))
    panels=[("MRI",image[:,:,z],"gray",None,None),("Teacher p0",before[:,:,z],"viridis",0,1),
        ("Risk mask",r05.lift_grid(mask)[:,:,z],"gray",0,1),("Matched side: fg=0",r05.lift_grid(chosen)[:,:,z],"coolwarm",0,1),
        ("Matching Q",r05.lift_grid(q)[:,:,z],"viridis",0,1),("After p",after[:,:,z],"viridis",0,1),
        ("Probability change",(after-before)[:,:,z],"coolwarm",-.1,.1),("GT",gt[:,:,z],"gray",0,1),
        ("Corrected",corrected[:,:,z],"Greens",0,1),("Harmed",harmed[:,:,z],"Reds",0,1)]
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for ax,(title,array,cmap,lo,hi) in zip(axes.flat,panels):
        ax.imshow(array.T,origin="lower",cmap=cmap,vmin=lo,vmax=hi);ax.set_title(title);ax.axis("off")
    fig.suptitle(case+" patch0 fixed middle slice z="+str(z)+"; max, alpha=.05")
    fig.tight_layout();fig.savefig(directory/(case+"_p0_fixed_slice.png"),dpi=140);plt.close(fig)


def write_report(out,summary,comparisons,case_rows,verification,stageF):
    def fmt(x): return "不可定义" if x is None else "{:.6g}".format(x)
    lines=["# Experiment R1-S0：冻结 Donor–Receiver 边界特征传递可行性", "", "## 研究范围与协议", "",
        "使用原始 Baseline EMA iter27000，8个已参与有标签训练病例、24个历史固定 patch。LOCO 仅隔离当前 Donor 构造和匹配中的 Receiver GT；这不是未见病例的独立泛化验证。所有预测在 eval、冻结参数及 no_grad 下完成，修改的只有原 Decoder 输入 x3 skip；x4/x5 仍来自原始 Encoder。",
        "主方案固定 Dxyz→Max Pooling→Top2%，次要方案 Trilinear→Top2%；alpha=0.02/0.05、Q=clip(|cos_fg−cos_bg|/0.2)、每位置相对L2上限0.10。Donor原型由每病例边界纯度≥0.75且网格最小边界距离≤3voxel的样本均值，再7病例等权平均。非边界为最小距离>3voxel；count-matched boundary 与 nonboundary 使用逐病例逐类别相同样本数。",
        "匹配尺度统计来自7个Donor全x3网格的等病例均值/二阶矩。余弦仅在标准化空间计算，残差仍在原始特征空间。Receiver自身原型只使用伪标签。无效原型、异常范数、Q≤0.001的格点不强制修正。",
        "原协议的标签交换在竞争匹配+绝对margin下数学等价，不能识别语义作用。保留等价检查；经用户明确授权增加反向匹配（改选另一类原型并保留Q）。随机Mask及随机前景/背景目标各20次、独立CPU Generator；全部结果保留。等能量对照以GT无关标量及相同逐位置cap匹配True residual总L2，不能匹配时明确记录。",
        "每折所有策略、3patch预测及配置先保存并SHA256封存，才解锁该Receiver GT。跨折同病例可换角色。原始两类FP32 logits无损保存，二值预测采用GPU原始softmax>0.5的封存位图；Phase B CPU重放softmax/原CE+Dice，数值容差单独测试。Random先patch内20次平均，再case内3patch平均，最后8病例等权、病例级Bootstrap10000次。", "", "## 主要定量结果", "",
        "Gain_B=BoundaryCE(无修正)−BoundaryCE(修正)，正值有益。Net=纠正错误−破坏正确，正值有益。下表CI为病例级探索性95%区间。", "",
        "|映射|alpha|方法|Boundary CE收益均值 [CI]|正收益病例|净纠正均值|Full CE收益|Dice变化|", "|---|---:|---|---|---:|---:|---:|---:|"]
    display=("true_boundary","opposite_match","nonboundary","boundary_count_matched","self","random_mask","random_match")
    for mapping in MAPPINGS:
        for alpha in ALPHAS:
            for method in display:
                key="{}|{}|{}".format(mapping,method,alpha)
                if key not in summary: continue
                s=summary[key];g=s["boundary_ce_gain"]
                lines.append("|{}|{}|{}|{} [{}, {}]|{}/8|{}|{}|{}|".format(mapping,alpha,method,fmt(g["mean"]),fmt(g["ci95"][0]),fmt(g["ci95"][1]),g["positive_cases"],fmt(s["net_correction"]["mean"]),fmt(s["full_ce_gain"]["mean"]),fmt(s["dice_gain"]["mean"])))
    lines += ["", "## 配对比较与扰动公平性", "", "正差值表示True边界优于对照；同时审查原始和等能量对照。", "",
              "|映射|alpha|对照|Boundary CE优势均值 [CI]|优势病例|", "|---|---:|---|---|---:|"]
    for r in comparisons:
        if r["metric"]!="boundary_ce" or r["control"]=="label_swap_invariant": continue
        lines.append("|{}|{}|{}|{} [{},{}]|{}/8|".format(r["mapping"],r["alpha"],r["control"],fmt(r["mean"]),fmt(r["ci95"][0]),fmt(r["ci95"][1]),r["positive_cases"]))
    lines += ["", "## 全部Receiver及失败结果", "", "|病例|alpha|Max True收益|纠正|误伤|净纠正|远FP变化|核心远FP变化|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in case_rows:
        if r["method"]=="true_boundary" and r["mapping"]=="max":
            lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|".format(r["case_id"],r["alpha"],fmt(r["boundary_ce_gain"]),fmt(r["corrected"]),fmt(r["harmed"]),fmt(r["net_correction"]),fmt(r["far_fp_change"]),fmt(r["far_fp_core_change"])))
    lines += ["", "## 科学问题Q1–Q10", ""]
    primary=[summary["max|true_boundary|{}".format(a)] for a in ALPHAS]
    positive=any(s["boundary_ce_gain"]["mean"]>0 and s["boundary_ce_gain"]["positive_cases"]>=5 and s["net_correction"]["mean"]>0 for s in primary)
    for alpha in ALPHAS:
        s=summary["max|true_boundary|{}".format(alpha)]
        lines.append("alpha={}：Max True纠正均值{}、误伤{}；Mask内误伤{}、Mask外误伤{}；纯侧网格匹配正确率{}；匹配与伪标签冲突位置原错误率{}。".format(alpha,fmt(s["corrected"]["mean"]),fmt(s["harmed"]["mean"]),fmt(s["harmed_inside_mask"]["mean"]),fmt(s["harmed_outside_mask"]["mean"]),fmt(s["match_class_accuracy_pure_grids"]["mean"]),fmt(s["match_conflict_original_error_fraction"]["mean"])))
    lines += ["", "Q1：风险定位的价值通过True+D Mask与20次Random Mask配对比较判断，不能由R0.5的错误覆盖率直接推导修正价值。具体差值和CI见上表；Mask外变化来自原Decoder空间卷积，已分别统计。",
        "Q2：真实边界知识的增量需同时优于非边界、Self、随机匹配及反向匹配，且等能量比较方向一致。标签交换完全等价，不能作为支持语义知识的证据。count-matched比较控制每Donor每类样本数，仍不能消除原型分布和范数差异。",
        "Q3：余弦margin/Q反映特征原型可分性，不能直接称为正确匹配概率。纯侧GT匹配准确率及与伪标签冲突的实际错误率均仅在Phase B评估。混合网格不强制赋真实前景/背景类别。",
        "Q4：Q的固定四档[0,.25)、[.25,.5)、[.5,.75)、[.75,1]分别保存原错误量、纠正量及误伤量，见correction_harm_analysis.csv。Q高未必代表Decoder需要相应原型残差。",
        "Q5：全部纠正、误伤和净纠正见上表及每patch/病例CSV；不得只比较Boundary CE。高D但原本正确的区域误伤分母为Mask内原正确体素，Decoder影响范围另报。",
        "Q6：Full CE、监督0.5×(CE+Dice)、Dice/Jaccard、FP/FN均保留；Boundary CE收益若伴随负净纠正或全局退化，不能判断交互有效。",
        "Q7："+stageF["reason"],
        "Q8：Max与Trilinear使用预定同一2%预算，未按GT改变主方案。其收益/误伤差异仅作为敏感性分析，不能在本数据上选最好配置并宣称独立性能。",
        "Q9："+("仅出现探索性可行信号，仍须审查对照增量和全局副作用，不能直接启动完整Mean Teacher训练。" if positive else "目前不支持进入完整Mean Teacher训练；正向残差的错误纠正和误伤需要先得到可靠证据。"),
        "Q10：若定位能覆盖错误但所有目标方向都无益，更可能是原型表达、竞争匹配或残差方向/置信度失配；若True与Random Mask相近，风险定位未转化为修正收益；若Self/非边界与True相近，则不能归因于跨样本真实边界知识。该区分是机制解释而非因果证明。", "", "## 事实、解释和未证实假设", "",
        "直接事实是本冻结checkpoint下各预定修正的实际Decoder重放损失、纠正与误伤及其跨病例配对差异。风险高不等于需要替换为Donor均值；候选库的边界纯侧筛选也不保证局部形态对应。全局两类均值可能丢失空间结构，这是合理解释，但本实验未单独证明。",
        "所有Receiver已被Baseline有标签训练，Donor集合高度重叠；8病例Bootstrap不能消除训练复用、选择checkpoint和小样本偏倚。patch裁剪使距离>20voxel的远FP定义敏感，保守核心与裁剪敏感部分单独保存。4³网格不是x3实际感受野。不能宣称提高整例/未见病例Dice或已实现半监督知识迁移。", "", "## 工程与保护验证", "",
        json.dumps(verification,ensure_ascii=False,indent=2), "", "## 可复核文件", "",
        "本目录包含全部原始随机重复、每patch/病例指标、配对CI、原型与匹配图、全FP32预测、逐折封存SHA、数据访问日志、预定病例0/3中间切片可视化和测试记录。原有源码、checkpoint、历史结果的SHA256前后相同；新文件尚不代表已推送GitHub。"]
    (out/"experiment_report.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    if str(args.device).startswith("cuda"):
        torch.ones(1,device=args.device)  # Fail before creating any run output.
    history,ids,checkpoint,references,reference_hashes=load_references(args)
    out=create_output(args.output_dir)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[logging.FileHandler(out/"analysis_log.txt"),logging.StreamHandler()])
    metadata=dict(status="running",experiment="R1-S0",configuration=vars(args),case_order=ids,checkpoint=checkpoint,
        crop_coordinates=history["sampling"]["patch_coordinates"],data_sources=history["data_sources"],reference_sha256=reference_hashes,
        code_sha256={str(p):r0.sha256(p) for p in NEW_CODE},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=args.device),
        protocol=dict(alphas=ALPHAS,mappings=MAPPINGS,top_budget=.02,purity=PURITY,boundary_min_distance_voxels=RADIUS,
            relative_L2_cap=CAP,q_scale=Q_SCALE,q_skip_threshold=Q_MIN,minimum_candidates=MIN_CANDIDATES,
            random_repetitions=REPETITIONS,bootstrap_case_repetitions=BOOTSTRAPS,
            label_swap_is_invariant=True,opposite_matching_explicitly_authorized=True,
            donor_equal_case_weight=True,matching_statistics="seven Donor full grids, equal case moments",
            energy_control="GT-free scalar matching total residual L2 with own Q-zero skips and per-position cap",
            no_receiver_GT_in_phaseA=True,original_decoder_batch_size=1,
            stageF_gate="Max True any fixed alpha: positive mean Boundary gain, >=5/8 positive cases, positive mean Net, nonnegative mean FullCE/Dice gain within1e-6",
            receiver_cases_seen_during_baseline_training=True,independent_validation=False))
    try:
        logging.info("Hashing all pre-existing project files and checkpoints")
        before=protected_manifest(out);metadata["protected_file_sha256_before"]=before
        metadata["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        r0.write_json(out/"experiment_metadata.json",metadata)
        test=subprocess.run([sys.executable,"-B",str(HERE/"test_experiment_R1_S0.py")],cwd=HERE,text=True,capture_output=True)
        r0.write_json(out/"test_results.json",dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,command=[sys.executable,"-B","analysis/test_experiment_R1_S0.py"]))
        if test.returncode: raise RuntimeError("Unit tests failed")
        model=u.load_frozen(args.model_path,args.device)
        state=u.snapshot_state(model)
        reader=FoldReader(history)
        images,xs={},{}
        with torch.no_grad():
            for case in ids:
                for patch in range(3):
                    image=reader.read(case,patch,"image");images[case,patch]=image
                    fs=model.encoder(torch.from_numpy(image)[None,None].to(args.device))
                    xs[case,patch]=fs[2][0].cpu().numpy().copy()
                logging.info("Image-only feature cache %s x3=%s",case,(1,)+xs[case,0].shape)
        all_rows,proto_rows,perturbations,checks,seals=[],[],[],[],[]
        folds=(0,3) if args.mode=="smoke" else range(8)
        for ci in folds:
            case=ids[ci];reader.start_fold(ci,case,[i for i in ids if i!=case])
            fold_dir=out/("fold_"+str(ci));fold_dir.mkdir()
            libs,donor=fold_prototypes(reader,xs,ids,case)
            logging.info("Fold %d receiver=%s boundary sources=%d; per-case counts=%s",ci,case,len(donor["sources"]["boundary"]),{k:v["counts"] for k,v in libs.items()})
            proto_rows.extend(case_library_rows(libs,donor,ci,case))
            prototype_path=fold_dir/"donor_prototypes.npz"
            payload={"mean":donor["mean"],"scale":donor["scale"]}
            payload.update(donor["prototypes"])
            for dc,lib in libs.items():
                payload[dc+"_channel_std"]=lib["raw_channel_std"]
                for kind,proto in lib["prototypes"].items(): payload[dc+"_"+kind]=proto
                for patch,g in enumerate(lib["grids"]):
                    for key,v in g.items(): payload[dc+"_p"+str(patch)+"_"+key]=v
            np.savez_compressed(prototype_path,**payload)
            artifacts=[artifact(prototype_path,"donor_library")];records=[]
            patches=(0,) if args.mode=="smoke" else range(3)
            for patch in patches:
                with torch.no_grad():
                    rr,aa,check=receiver_phaseA(model,reader,images,xs,donor,case,ci,patch,fold_dir,references[case,patch],repetitions=1 if args.mode=="smoke" else REPETITIONS)
                records.extend(rr);artifacts.extend(aa);checks.append(check)
                logging.info("Fold %d patch %d Phase A complete: %d strategies; Receiver GT still locked",ci,patch,len(rr))
            plan=fold_dir/"phaseA_plan.json"
            r0.write_json(plan,dict(receiver=case,donor_cases=reader.donors,configuration=metadata["protocol"],records=records))
            artifacts.append(artifact(plan,"predefined_strategy_plan"))
            seal=dict(fold=ci,receiver_case=case,donor_cases=reader.donors,artifacts=artifacts,
                prediction_count=sum(a["kind"]=="prediction" for a in artifacts),receiver_GT_read=False,sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            seal_path=fold_dir/"phaseA_seal.json";r0.write_json(seal_path,seal)
            expected_predictions=len(patches)*(1+len(MAPPINGS)*len(ALPHAS)*(9+4*(1 if args.mode=="smoke" else REPETITIONS)))
            reader.seal(artifacts,expected_predictions);seals.append(dict(fold=ci,path=str(seal_path),sha256=r0.sha256(seal_path)))
            perturbations.extend({k:v for k,v in r.items() if k not in ("path","matching_map_path")} for r in records)
            if args.mode=="full":
                logging.info("Fold %d sealed %d predictions; now Phase B GT evaluation",ci,seal["prediction_count"])
                all_rows.extend(evaluate_fold(reader,records,case,images))
            r0.write_csv(out/"data_access_log.csv",reader.events)
            r0.write_csv(out/"feature_perturbation_statistics.csv",perturbations)
            r0.write_csv(out/"donor_prototype_statistics.csv",proto_rows)
            if all_rows: r0.write_csv(out/"patch_level_metrics.csv",all_rows)
            r0.write_json(out/"phaseA_seal.json",dict(folds=seals,scope="per-fold: all3patch/allstrategies sealed before that Receiver GT",mode=args.mode))
            logging.info("Fold %d finished",ci)
        verify=u.verify_state(model,state)
        after=protected_manifest(out)
        if before!=after: raise AssertionError("Pre-existing project file hashes changed")
        verify.update(protected_file_sha256_unchanged=True,protected_file_count=len(before),all_receiver_GT_reads_after_fold_seal=all(e["fold_sealed"] for e in reader.events if e["dataset"]=="label" and e["role"]=="receiver"),
            no_test_GT_access=True,no_training_or_backward_or_optimizer_or_EMA=True,no_BGS_gain=True,
            exact_historical_risk_and_masks=True,forward_replay_exact=True,other_encoder_features_unchanged=True,
            label_swap_invariant=True,opposite_match_authorized=True,random_repetitions_all_saved=True,
            no_nonfinite_outputs=True,bootstrap_unit="case",all_8_cases_24patches=args.mode=="full",patch_checks=checks)
        stageF=dict(executed=False,reason="完整64维未满足预定收益与副作用门槛；按协议停止TopK复杂化，不把D0-A重要性等同可迁移性。")
        if args.mode=="full":
            case_rows,summary,comparisons=aggregate(all_rows,ids)
            # Gate is recorded before any Receiver evaluation; no alpha/mask tuning.
            gate=any(summary["max|true_boundary|{}".format(a)]["boundary_ce_gain"]["mean"]>0 and
                summary["max|true_boundary|{}".format(a)]["boundary_ce_gain"]["positive_cases"]>=5 and
                summary["max|true_boundary|{}".format(a)]["net_correction"]["mean"]>0 and
                summary["max|true_boundary|{}".format(a)]["full_ce_gain"]["mean"]>=-1e-6 and
                summary["max|true_boundary|{}".format(a)]["dice_gain"]["mean"]>=-1e-6 for a in ALPHAS)
            stageF["gate_passed"]=gate
            if gate:
                stageF["reason"]="主实验达到预定门槛，需要继续开发集独立TopK后续消融；当前不得宣称该步骤已完成。"
                stageF["pending"]=True
            r0.write_csv(out/"case_level_metrics.csv",case_rows)
            r0.write_csv(out/"method_comparison.csv",comparisons)
            r0.write_csv(out/"correction_harm_analysis.csv",[{k:v for k,v in r.items() if k in ("case_id","patch_index","mapping","method","alpha","repetition") or k.startswith(("corrected","harmed","net_","mask_","match_","qbin_","far_","high_confidence"))} for r in all_rows])
            r0.write_csv(out/"random_control_results.csv",[r for r in all_rows if r["repetition"] is not None])
            r0.write_csv(out/"receiver_mask_statistics.csv",[{k:r.get(k) for k in ("case_id","patch_index","mapping","mask_positions","mask_error_voxels","mask_correct_voxels")} for r in all_rows if r["method"]=="no_correction" and r["alpha"]==.02])
            r0.write_csv(out/"donor_matching_statistics.csv",[{k:v for k,v in r.items() if k in ("case_id","patch_index","mapping","method","alpha","repetition","q_mean","q_max","q_near_zero","match_success_fraction") or k.startswith("match_")} for r in all_rows if r["repetition"] is None])
            r0.write_csv(out/"channel_selection_ablation.csv",[stageF])
            r0.write_json(out/"summary_statistics.json",dict(methods=summary,paired_comparisons=comparisons,stageF=stageF,statistical_unit="case",bootstrap_repetitions=BOOTSTRAPS))
            r0.write_json(out/"verification.json",verify)
            write_report(out,summary,comparisons,case_rows,verify,stageF)
        else:
            r0.write_json(out/"verification.json",verify)
            (out/"experiment_report.md").write_text("# R1-S0 工程冒烟测试\n\n固定病例0/3各patch0，仅Phase A；没有读取Receiver GT，没有计算效果或调整超参数。完整主实验尚未执行。\n")
        metadata.update(status="stageF_pending" if stageF.get("pending") else "complete" if args.mode=="full" else "smoke_complete",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protected_file_sha256_after=after,phaseA_seals=seals,patch_checks=checks,stageF=stageF)
        r0.write_json(out/"experiment_metadata.json",metadata)
        logging.info("%s output=%s",metadata["status"],out)
    except Exception as exc:
        logging.exception("R1-S0 stopped without changing scientific parameters")
        metadata.update(status="failed",error=repr(exc))
        r0.write_json(out/"experiment_metadata.json",metadata)
        raise


if __name__=="__main__": main()
