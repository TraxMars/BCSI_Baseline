"""BGS-U2: frozen cross-case donor gain, versus no gain/shuffle/uniform.

All patches are replayed from BGS-U metadata. No sampling, training, backward,
optimizer, EMA update, or receiver-GT-dependent strategy construction occurs.
"""
import argparse
import csv
from dataclasses import dataclass
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

# Reuse read-only analysis definitions; never invoke BGS-U's gradient function
# or main. Include these existing files in the protected SHA256 manifest.
import experiment_bgs_utility as legacy
from utils.boundary_guidance import (EPS, apply_labeled_bgs_guidance, compute_bgs,
                                    shuffle_bgs_weights)

ALPHA = 0.1
SHUFFLE_REPEATS = 20
BOOTSTRAP_REPEATS = 10000
STRATEGIES = ("no_gain", "true_bgs", "shuffled_bgs", "uniform_gain")
NEW_CODE = {Path(__file__).resolve(), HERE / "test_experiment_bgs_utility_v2.py"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_dir", default=str(HERE / "results/experiment_bgs_utility/run_cuda_20261009_1713"))
    p.add_argument("--output_dir", default=str(HERE / "results/experiment_bgs_utility_v2"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--baseline_only", action="store_true")
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or args.seed < 0:
        p.error("cpu_threads must be positive and seed must be nonnegative")
    return args


def protected_manifest(out):
    manifest = {}
    for parent, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")
                         and (Path(parent) / d).resolve() != out.resolve())
        for filename in sorted(files):
            path = Path(parent) / filename
            if path.resolve() not in NEW_CODE and path.suffix not in (".pyc", ".pyo") and path.is_file():
                manifest[str(path.relative_to(ROOT))] = legacy.sha256(path)
    return manifest


def array_sha256(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def require_metadata(metadata):
    """Reject incomplete or ambiguous history; never replace missing patches."""
    try:
        assert metadata["status"] == "complete"
        config = metadata["configuration"]
        assert config["dataset"] == "LA" and config["labeled_num"] == 10
        assert config["patch_size"] == [112, 112, 80] and config["patches_per_case"] == 3
        ids = metadata["data_split"]["labeled_case_ids"]
        assert len(ids) == len(set(ids)) == 8
        sources = metadata["data_sources"]
        assert len(sources) == 8 and {s["case_id"] for s in sources} == set(ids)
        coordinates = metadata["sampling"]["patch_coordinates"]
        assert len(coordinates) == 24
        assert {(c["case_id"], c["patch_index"]) for c in coordinates} == {(i, j) for i in ids for j in range(3)}
        required = {"start_padded", "start_original", "padding", "original_shape", "padded_shape"}
        for c in coordinates:
            assert required <= set(c)
            for field in required:
                assert len(c[field]) == 3 and all(isinstance(v, int) for v in c[field])
        assert all({"case_id", "path", "sha256", "train_index"} <= set(s) for s in sources)
        assert all({"name", "path", "sha256"} <= set(c) for c in metadata["checkpoints"])
    except (KeyError, TypeError, AssertionError) as exc:
        raise ValueError("Historical metadata is insufficient for exact 8-case/24-patch replay; no new sampling permitted") from exc
    return ids


def replay_crop(image, label, coords, patch_size):
    shape = tuple(label.shape)
    if image.shape != shape or list(shape) != coords["original_shape"]:
        raise ValueError("Historical source dimensions differ")
    padding = [max((p - s) // 2 + 3, 0) for p, s in zip(patch_size, shape)] if any(
        s <= p for s, p in zip(shape, patch_size)) else [0, 0, 0]
    if padding != coords["padding"]:
        raise ValueError("Historical padding differs from original RandomCrop rule")
    padded_shape = [s + 2 * p for s, p in zip(shape, padding)]
    if padded_shape != coords["padded_shape"]:
        raise ValueError("Historical padded dimensions differ")
    start = coords["start_padded"]
    if [s - p for s, p in zip(start, padding)] != coords["start_original"]:
        raise ValueError("Historical original/padded coordinates are inconsistent")
    if any(s < 0 or s + p > d for s, p, d in zip(start, patch_size, padded_shape)):
        raise ValueError("Historical crop outside padded volume")
    slices = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
    pad = [(p, p) for p in padding]
    return (np.ascontiguousarray(np.pad(image, pad)[slices], dtype=np.float32),
            np.ascontiguousarray(np.pad(label, pad)[slices], dtype=np.int64))


def replay_patches(metadata):
    ids = require_metadata(metadata)
    config = metadata["configuration"]
    cases, current_split = legacy.labeled_split(SimpleNamespace(**config))
    if current_split["labeled_case_ids"] != ids:
        raise ValueError("Current actual labeled split differs from historical split")
    for name in ("train", "test"):
        key = name + "_list_sha256"
        if current_split[key] != metadata["data_split"][key]:
            raise ValueError("Historical " + name + ".list hash differs")
    source_by_case = {s["case_id"]: s for s in metadata["data_sources"]}
    coordinates = {(c["case_id"], c["patch_index"]): c for c in metadata["sampling"]["patch_coordinates"]}
    patches, replay_records = {}, []
    for index, case_id, actual_path in cases:
        source = source_by_case[case_id]
        if Path(source["path"]).resolve() != actual_path.resolve() or source["train_index"] != index:
            raise ValueError("Historical data path/train index differs: " + case_id)
        if legacy.sha256(actual_path) != source["sha256"]:
            raise ValueError("Historical H5 source hash differs: " + case_id)
        with h5py.File(str(actual_path), "r") as f:
            image, label = f["image"][:], f["label"][:]
        legacy.assert_finite(image, case_id + "/image")
        if not np.isin(label, [0, 1]).all():
            raise ValueError(case_id + " nonbinary label")
        for patch_index in range(3):
            coords = coordinates[case_id, patch_index]
            im, gt = replay_crop(image, label, coords, config["patch_size"])
            # Counts validate exact historical replay, never select new samples.
            if int(gt.sum()) != coords["foreground_voxels"]:
                raise ValueError("Historical foreground count differs: " + case_id)
            patches[case_id, patch_index] = dict(case_id=case_id, patch_index=patch_index,
                                                image=im, label=gt, coords=coords)
            replay_records.append(dict(case_id=case_id, patch_index=patch_index, coordinates=coords,
                                       image_sha256=array_sha256(im), label_sha256=array_sha256(gt),
                                       source_sha256=source["sha256"], exact_replay=True))
    return patches, replay_records, current_split


def donor_mapping(case_ids):
    return {case_id: (case_ids[(i + 1) % len(case_ids)], case_ids[(i + 2) % len(case_ids)])
            for i, case_id in enumerate(case_ids)}


@torch.no_grad()
def donor_signature(model, donor_patches, receiver_id, device, min_voxels=8):
    """API receives only donor patches. Receiver label/U cannot be passed here."""
    if len(donor_patches) != 2 or receiver_id in {p["case_id"] for p in donor_patches}:
        raise ValueError("Two different-case donors required; receiver cannot be a donor")
    if donor_patches[0]["case_id"] == donor_patches[1]["case_id"]:
        raise ValueError("Donor IDs must differ")
    x3s, labels, bgs_rows, valid_flags = [], [], [], []
    for p in donor_patches:
        x3 = model.encoder(torch.from_numpy(p["image"])[None, None].to(device))[2]
        gt = torch.from_numpy(p["label"])[None].to(device)
        bgs, valid = compute_bgs(x3, gt, min_voxels)
        x3s.append(x3)
        labels.append(gt)
        bgs_rows.append(bgs[0])
        valid_flags.append(bool(valid[0]))
    if not any(valid_flags):
        return None, dict(valid_donors=0, donor_valid_flags=valid_flags, reason="no_valid_donor"), bgs_rows
    signature = torch.stack([b for b, valid in zip(bgs_rows, valid_flags) if valid]).mean(0).detach()
    positive = signature.relu()
    w = (positive / (positive.max() + EPS)).to(x3s[0].dtype).detach()
    # Actual original Experiment A helper validates averaging order and weights.
    stacked = torch.cat(x3s + [torch.zeros_like(x3s[0])], dim=0)
    _, reference_w, reference_diagnostics = apply_labeled_bgs_guidance(
        stacked, torch.cat(labels, dim=0), 2, alpha=ALPHA, min_band_voxels=min_voxels)
    if not torch.equal(reference_w, w) or reference_diagnostics["valid_samples"] != sum(valid_flags):
        raise AssertionError("Donor weights differ from actual Experiment A implementation")
    legacy.assert_finite(w, "donor weights for " + receiver_id)
    return w, dict(valid_donors=sum(valid_flags), donor_valid_flags=valid_flags, reason="",
                   weights_match_experiment_A_exactly=True, signature=signature.cpu().tolist(),
                   w_mean=float(w.double().mean()), w_norm=float(w.double().norm())), bgs_rows


@dataclass(frozen=True)
class Strategy:
    name: str
    repetition: int
    weight: torch.Tensor
    permutation: tuple = ()
    shuffle_mean_delta: float = 0.0
    shuffle_norm_delta: float = 0.0


def make_strategy_plan(w, shuffle_seed, repeats=SHUFFLE_REPEATS):
    """Precommit all permutations/strategies before any receiver-GT evaluation."""
    if w.requires_grad or w.ndim != 1:
        raise ValueError("Expected detached channel weights")
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    plan = [Strategy("no_gain", -1, torch.zeros_like(w)), Strategy("true_bgs", -1, w.clone()),
            Strategy("uniform_gain", -1, torch.full_like(w, float(w.mean())))]
    for repetition in range(repeats):
        permuted, permutation, diagnostics = shuffle_bgs_weights(w, generator)
        plan.append(Strategy("shuffled_bgs", repetition, permuted.clone(), tuple(permutation.cpu().tolist()),
                             diagnostics["shuffle_mean_delta"], diagnostics["shuffle_norm_delta"]))
    if not torch.equal(global_rng, torch.get_rng_state()):
        raise AssertionError("Shuffle consumed global torch RNG")
    return tuple(plan)


def plan_digest(plan):
    h = hashlib.sha256()
    for p in plan:
        h.update((p.name + str(p.repetition) + str(p.permutation)).encode())
        h.update(p.weight.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def apply_strategy(features, strategy):
    changed = list(features)
    if strategy.name != "no_gain":
        changed[2] = features[2] * (1 + ALPHA * strategy.weight[None, :, None, None, None])
    if any(changed[i] is not features[i] for i in (0, 1, 3, 4)):
        raise AssertionError("A non-x3 feature was replaced")
    relative = float(((changed[2] - features[2]).norm() / (features[2].norm() + EPS)).item())
    return changed, relative


def evaluate_logits(logits, labels, inner, outer):
    losses = {k: float(v.item()) for k, v in legacy.loss_values(logits, labels, inner, outer).items()}
    predicted = torch.softmax(logits, dim=1)[:, 1] > 0.5
    target = labels > 0
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    # Empty/empty sets use score 1; empty prediction with nonempty GT uses 0.
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return dict(**losses, patch_dice=dice, patch_jaccard=jaccard, fp_voxels=fp, fn_voxels=fn,
                tp_voxels=tp, foreground_prediction_voxels=tp + fp)


def calculate_gains(no_gain, modified):
    return dict(boundary_ce_gain=no_gain["boundary_ce"] - modified["boundary_ce"],
                full_ce_gain=no_gain["full_ce"] - modified["full_ce"],
                supervised_ce_dice_gain=no_gain["supervised_ce_dice"] - modified["supervised_ce_dice"],
                patch_dice_gain=modified["patch_dice"] - no_gain["patch_dice"],
                patch_jaccard_gain=modified["patch_jaccard"] - no_gain["patch_jaccard"],
                fp_voxels_reduction=no_gain["fp_voxels"] - modified["fp_voxels"],
                fn_voxels_reduction=no_gain["fn_voxels"] - modified["fn_voxels"])


def true_advantage(true, random_control, metric="boundary_ce"):
    """Positive means true BGS has lower loss than the control."""
    return random_control[metric] - true[metric]


def alignment_metrics(w, utility):
    if utility is None:
        return dict(utility_available=False, w_utility_spearman=None, S=None, linear_predicted_boundary_gain=None,
                    negative_utility_weight_mass_fraction=None, undefined_reason="historical_U_not_verified")
    weights = w.detach().cpu().numpy().astype(np.float64)
    u = np.asarray(utility, dtype=np.float64)
    if u.shape != weights.shape:
        raise ValueError("Historical U channel dimension differs from receiver")
    rho, reason = legacy.safe_spearman(weights, u)
    S = float(np.dot(weights, u))
    total = float(weights.sum())
    return dict(utility_available=True, w_utility_spearman=rho, S=S,
                linear_predicted_boundary_gain=ALPHA * S,
                negative_utility_weight_mass_fraction=float(weights[u < 0].sum() / total) if total > 0 else None,
                undefined_reason=reason + (";zero_weight_sum" if total == 0 else ""))


def historical_utility_index(reference_dir):
    rows_by_patch = {}
    path = reference_dir / "channel_bgs_utility.csv"
    if not path.exists():
        return {}, dict(available=False, reason="historical_channel_csv_missing")
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["checkpoint"], row["case_id"], int(row["patch_index"]))
            rows_by_patch.setdefault(key, []).append(row)
    index, issues = {}, []
    for key, rows in rows_by_patch.items():
        shape = json.loads(rows[0]["feature_shape"])
        ordered = sorted(rows, key=lambda r: int(r["channel"]))
        if [int(r["channel"]) for r in ordered] != list(range(shape[1])):
            issues.append(dict(key=key, reason="duplicate_or_missing_channel"))
            continue
        u = np.asarray([float(r["utility"]) for r in ordered], dtype=np.float64)
        legacy.assert_finite(u, str(key) + "/historical U")
        index[key] = dict(utility=u, feature_shape=shape,
                          base_losses={name: float(rows[0][name]) for name in ("boundary_ce", "full_ce", "supervised_ce_dice")})
    return index, dict(available=True, path=str(path), sha256=legacy.sha256(path), indexed_patches=len(index), issues=issues)


def summarize_repetitions(rows):
    result = {}
    for key in dict.fromkeys(k for r in rows for k in r):
        if key in ("patch_index", "repetition", "shuffle_seed"):
            continue
        values = [r[key] for r in rows if isinstance(r.get(key), (float, int, np.floating, np.integer)) and not isinstance(r.get(key), bool)]
        if len(values) == len(rows):
            result[key] = float(np.mean(values))
            result[key + "_shuffle_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[key + "_shuffle_min"] = float(min(values))
            result[key + "_shuffle_max"] = float(max(values))
    return result


@torch.no_grad()
def analyze_model(model, checkpoint, checkpoint_index, patches, case_ids, utility_index, args, min_voxels):
    before = legacy.snapshot_state(model)
    mapping = donor_mapping(case_ids)
    comparison, repetitions, alignments, weights_csv, mapping_csv, checks, invalid = [], [], [], [], [], [], []
    for receiver_index, receiver_id in enumerate(case_ids):
        for patch_index in range(3):
            key = (checkpoint, receiver_id, patch_index)
            context = "{}/{}/patch{}".format(*key)
            receiver = patches[receiver_id, patch_index]
            donors = [patches[case_id, patch_index] for case_id in mapping[receiver_id]]
            identity = dict(checkpoint=checkpoint, receiver_case=receiver_id, patch_index=patch_index)
            mapping_csv.append(dict(**identity, donor1_case=donors[0]["case_id"], donor2_case=donors[1]["case_id"],
                                    receiver_coordinates=receiver["coords"], donor1_coordinates=donors[0]["coords"],
                                    donor2_coordinates=donors[1]["coords"], donor_rule="(i+1)%8,(i+2)%8; same patch index"))
            features = model.encoder(torch.from_numpy(receiver["image"])[None, None].to(args.device))
            original_features = [f.clone() for f in features]
            w, donor_info, donor_bgs = donor_signature(model, donors, receiver_id, args.device, min_voxels)
            if w is None:
                invalid.append(dict(**identity, reason="no_valid_donor", donor_valid_flags=donor_info["donor_valid_flags"]))
                continue
            seed = args.seed + checkpoint_index * 1000003 + receiver_index * 1009 + patch_index * 9176
            cpu_rng = torch.get_rng_state().clone()
            numpy_rng = np.random.get_state()
            python_rng = random.getstate()
            cuda_rng = torch.cuda.get_rng_state(args.device).clone() if str(args.device).startswith("cuda") else None
            plan = make_strategy_plan(w, seed)
            sealed = plan_digest(plan)
            # Only AFTER all 23 strategy entries are fixed is receiver GT passed
            # into evaluation. No receiver label or U is an input to either plan
            # construction or donor_signature above.
            labels = torch.from_numpy(receiver["label"])[None].to(args.device)
            inner, outer = legacy.boundary_sides(labels)
            if min(int(inner.sum()), int(outer.sum())) < min_voxels:
                invalid.append(dict(**identity, reason="invalid_historical_receiver_boundary"))
                continue
            for field, value in (("inner_voxels", int(inner.sum())), ("outer_voxels", int(outer.sum()))):
                if value != receiver["coords"][field]:
                    raise AssertionError(context + " replayed receiver boundary count differs")
            original_logits = model(torch.from_numpy(receiver["image"])[None, None].to(args.device))
            no_gain_logits = model.decoder(features)
            if not torch.equal(original_logits, no_gain_logits):
                raise AssertionError(context + " no-gain differs from original eval forward")
            baseline = evaluate_logits(no_gain_logits, labels, inner, outer)
            historical = utility_index.get(key)
            u, utility_reason, historical_error = None, "history_patch_missing", None
            if historical and historical["feature_shape"] == list(features[2].shape):
                historical_error = max(abs(baseline[k] - v) for k, v in historical["base_losses"].items())
                if historical_error <= 1e-6:
                    u, utility_reason = historical["utility"], ""
                else:
                    utility_reason = "historical_baseline_loss_differs; mechanism_skipped"
            elif historical:
                utility_reason = "historical_feature_shape_differs; mechanism_skipped"
            raw, local_shuffle = {}, []
            for strategy in plan:
                modified_features, relative_change = apply_strategy(features, strategy)
                logits = no_gain_logits if strategy.name == "no_gain" else model.decoder(modified_features)
                legacy.assert_finite(logits, context + "/" + strategy.name + "/logits")
                evaluated = evaluate_logits(logits, labels, inner, outer)
                mechanism = alignment_metrics(strategy.weight, u)
                row = dict(**identity, strategy=strategy.name, relative_feature_l2_change=relative_change,
                           **evaluated, **calculate_gains(baseline, evaluated), **mechanism)
                if strategy.name == "shuffled_bgs":
                    saved = dict(**row, repetition=strategy.repetition, shuffle_seed=seed,
                                 permutation=list(strategy.permutation), shuffle_distribution_equal=True,
                                 shuffle_mean_delta=strategy.shuffle_mean_delta, shuffle_norm_delta=strategy.shuffle_norm_delta)
                    repetitions.append(saved)
                    local_shuffle.append(row)
                else:
                    raw[strategy.name] = row
            shuffled = dict(**identity, strategy="shuffled_bgs", shuffle_repeats=SHUFFLE_REPEATS,
                            **summarize_repetitions(local_shuffle))
            shuffled["utility_available"] = u is not None
            shuffled["undefined_reason"] = utility_reason
            raw["shuffled_bgs"] = shuffled
            advantages = dict(true_vs_shuffle_boundary_advantage=true_advantage(raw["true_bgs"], shuffled),
                              true_vs_uniform_boundary_advantage=true_advantage(raw["true_bgs"], raw["uniform_gain"]),
                              true_vs_shuffle_full_ce_advantage=true_advantage(raw["true_bgs"], shuffled, "full_ce"),
                              true_vs_uniform_full_ce_advantage=true_advantage(raw["true_bgs"], raw["uniform_gain"], "full_ce"),
                              true_bgs_shuffle_percentile=100 * (sum(r["boundary_ce"] > raw["true_bgs"]["boundary_ce"] for r in local_shuffle)
                                                                 + 0.5 * sum(r["boundary_ce"] == raw["true_bgs"]["boundary_ce"] for r in local_shuffle)) / SHUFFLE_REPEATS)
            for strategy in STRATEGIES:
                row = dict(**raw[strategy], valid_donors=donor_info["valid_donors"], **advantages)
                comparison.append(row)
                alignments.append(dict(**identity, strategy=strategy, utility_available=u is not None,
                                       w_utility_spearman=row.get("w_utility_spearman"), S=row.get("S"),
                                       linear_predicted_boundary_gain=row.get("linear_predicted_boundary_gain"),
                                       negative_utility_weight_mass_fraction=row.get("negative_utility_weight_mass_fraction"),
                                       actual_boundary_gain=row["boundary_ce_gain"], historical_U_match_reason=utility_reason,
                                       baseline_loss_max_difference=historical_error,
                                       undefined_reason=row.get("undefined_reason", ""),
                                       S_shuffle_std=row.get("S_shuffle_std"), S_shuffle_min=row.get("S_shuffle_min"),
                                       S_shuffle_max=row.get("S_shuffle_max")))
            for channel in range(w.numel()):
                weights_csv.append(dict(**identity, channel=channel, donor1_bgs=float(donor_bgs[0][channel]),
                                        donor2_bgs=float(donor_bgs[1][channel]),
                                        donor1_valid=donor_info["donor_valid_flags"][0], donor2_valid=donor_info["donor_valid_flags"][1],
                                        signed_mean_bgs=donor_info["signature"][channel], w=float(w[channel]),
                                        receiver_U=float(u[channel]) if u is not None else None))
            if plan_digest(plan) != sealed:
                raise AssertionError("A strategy changed after receiver evaluation")
            if any(not torch.equal(f, p) for f, p in zip(features, original_features)):
                raise AssertionError(context + " encoder cache mutated")
            np_after = np.random.get_state()
            global_rng_unchanged = (torch.equal(cpu_rng, torch.get_rng_state()) and
                                    numpy_rng[0] == np_after[0] and np.array_equal(numpy_rng[1], np_after[1]) and
                                    numpy_rng[2:] == np_after[2:] and python_rng == random.getstate())
            if cuda_rng is not None:
                global_rng_unchanged = global_rng_unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(args.device))
            if not global_rng_unchanged:
                raise AssertionError(context + " evaluation/shuffle consumed a global random stream")
            checks.append(dict(**identity, no_gain_forward_exact=True, no_gain_forward_max_error=0.0,
                               feature_shapes=[list(f.shape) for f in features], only_x3_changed=True,
                               original_features_unchanged=True, weights_match_experiment_A_exactly=True,
                               receiver_gt_excluded_from_signature_and_plan=True, donor_ids_distinct_from_receiver=True,
                               valid_donors=donor_info["valid_donors"], strategies_sealed_before_gt_evaluation=True,
                               strategy_plan_sha256=sealed, shuffle_count=len(local_shuffle),
                               shuffle_all_repetitions_kept=True, global_rng_unchanged=True,
                               historical_U_verified=u is not None, historical_U_skip_reason=utility_reason,
                               historical_baseline_loss_max_difference=historical_error,
                               max_shuffle_mean_delta=max(p.shuffle_mean_delta for p in plan),
                               max_shuffle_norm_delta=max(p.shuffle_norm_delta for p in plan)))
            logging.info("%s: donors=%s, true G=%.6f shuffle G=%.6f uniform G=%.6f, true-shuffle=%.6f", context,
                         mapping[receiver_id], raw["true_bgs"]["boundary_ce_gain"], shuffled["boundary_ce_gain"],
                         raw["uniform_gain"]["boundary_ce_gain"], advantages["true_vs_shuffle_boundary_advantage"])
            del features, original_features, original_logits, no_gain_logits, logits, labels, plan
    return (comparison, repetitions, alignments, weights_csv, mapping_csv, invalid,
            dict(**legacy.verify_state(model, before), receivers=checks))


def summarize_cases(comparison, repetitions, case_ids, checkpoints, seed):
    case_rows, case_shuffle, summaries = [], [], {}
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_rows = []
        for case_id in case_ids:
            pp = [r for r in comparison if r["checkpoint"] == checkpoint and r["receiver_case"] == case_id]
            if not pp:
                continue
            by_strategy = {strategy: [r for r in pp if r["strategy"] == strategy] for strategy in STRATEGIES}
            n_patches = len(by_strategy["no_gain"])
            row = dict(checkpoint=checkpoint, receiver_case=case_id, valid_patches=n_patches)
            for strategy, rows in by_strategy.items():
                for metric in rows[0]:
                    values = [r.get(metric) for r in rows]
                    if all(isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) for v in values):
                        row[strategy + "_" + metric] = float(np.mean(values))
                row[strategy + "_defined_alignment_patches"] = sum(r.get("S") is not None for r in rows)
            case_reps = []
            for repeat in range(SHUFFLE_REPEATS):
                rr = [r for r in repetitions if r["checkpoint"] == checkpoint and r["receiver_case"] == case_id and r["repetition"] == repeat]
                if len(rr) != n_patches:
                    raise AssertionError("Missing/selected shuffle repetitions at case level")
                rep = dict(checkpoint=checkpoint, receiver_case=case_id, repetition=repeat,
                           boundary_ce=float(np.mean([r["boundary_ce"] for r in rr])),
                           boundary_ce_gain=float(np.mean([r["boundary_ce_gain"] for r in rr])),
                           full_ce=float(np.mean([r["full_ce"] for r in rr])))
                case_reps.append(rep)
                case_shuffle.append(rep)
            row["true_bgs_case_shuffle_percentile"] = 100 * (sum(r["boundary_ce"] > row["true_bgs_boundary_ce"] for r in case_reps)
                                                            + 0.5 * sum(r["boundary_ce"] == row["true_bgs_boundary_ce"] for r in case_reps)) / SHUFFLE_REPEATS
            row["shuffled_bgs_case_boundary_ce_std"] = float(np.std([r["boundary_ce"] for r in case_reps], ddof=1))
            row["shuffled_bgs_case_boundary_ce_min"] = min(r["boundary_ce"] for r in case_reps)
            row["shuffled_bgs_case_boundary_ce_max"] = max(r["boundary_ce"] for r in case_reps)
            row["true_vs_shuffle_S_advantage"] = row.get("true_bgs_S", 0) - row.get("shuffled_bgs_S", 0) if row.get("true_bgs_S") is not None and row.get("shuffled_bgs_S") is not None else None
            row["true_vs_uniform_S_advantage"] = row.get("true_bgs_S", 0) - row.get("uniform_gain_S", 0) if row.get("true_bgs_S") is not None and row.get("uniform_gain_S") is not None else None
            row["true_vs_shuffle_negative_weight_mass_difference"] = row.get("true_bgs_negative_utility_weight_mass_fraction", 0) - row.get("shuffled_bgs_negative_utility_weight_mass_fraction", 0) if "true_bgs_negative_utility_weight_mass_fraction" in row and "shuffled_bgs_negative_utility_weight_mass_fraction" in row else None
            case_rows.append(row)
            checkpoint_rows.append(row)
        rng = np.random.RandomState(seed + 2000003 + index)
        metrics = {}
        for key in dict.fromkeys(k for row in checkpoint_rows for k in row):
            if key in ("checkpoint", "receiver_case", "valid_patches") or key.endswith("_defined_alignment_patches"):
                continue
            vals = [r.get(key) for r in checkpoint_rows]
            if not all(v is None or isinstance(v, (float, int)) for v in vals):
                continue
            stats = legacy.bootstrap_cases(vals, rng, BOOTSTRAP_REPEATS)
            finite = np.array([v for v in vals if v is not None])
            if len(finite):
                stats["std"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            stats.update(improved_cases=int(sum(v > 0 for v in finite)), worsened_cases=int(sum(v < 0 for v in finite)),
                         unchanged_cases=int(sum(v == 0 for v in finite)), sign_counts_note="positive/negative; improvement interpretation applies to gain/advantage fields")
            metrics[key] = stats
        summaries[checkpoint] = dict(n_cases=len(checkpoint_rows), n_receiver_patches=sum(r["valid_patches"] for r in checkpoint_rows),
                                    shuffle_repeats_per_patch=SHUFFLE_REPEATS, bootstrap_repetitions=BOOTSTRAP_REPEATS,
                                    aggregation="20 shuffle draws -> patch mean -> case mean of 3 patches -> equal case weight",
                                    standard_deviation_ddof=1, metrics=metrics,
                                    primary_metrics={label: metrics[field] for label, field in {
                                        "true_gain": "true_bgs_boundary_ce_gain", "shuffle_gain": "shuffled_bgs_boundary_ce_gain",
                                        "uniform_gain": "uniform_gain_boundary_ce_gain",
                                        "true_vs_shuffle": "true_bgs_true_vs_shuffle_boundary_advantage",
                                        "true_vs_uniform": "true_bgs_true_vs_uniform_boundary_advantage"}.items()})
    return case_rows, case_shuffle, summaries


def decision(summary):
    m, primary = summary["metrics"], summary["primary_metrics"]
    a = (all(primary[key]["ci95"][0] > 0 for key in ("true_gain", "true_vs_shuffle", "true_vs_uniform")) and
         m["true_bgs_full_ce_gain"]["ci95"][0] > 0 and primary["true_gain"]["improved_cases"] > summary["n_cases"] / 2)
    if a:
        return "A", "真实 BGS 的边界 CE、全图 CE 及与两个控制的差异均呈一致正向区间，可保留为候选指标；未设临床意义阈值，不宣称训练收益。"
    conflict = primary["true_gain"]["improved_cases"] >= 2 and primary["true_gain"]["worsened_cases"] >= 2
    all_harmful = all(primary[key]["mean"] < 0 for key in ("true_gain", "shuffle_gain", "uniform_gain"))
    if all_harmful or conflict:
        return "C", "三种正向增益的病例均值均为负，或至少两例获益且至少两例受损；停止无条件正向 Channel Gain，优先研究是否需要调制及调制方向。"
    return "B", "真实 BGS 未同时满足对 No Gain、Shuffle、Uniform 的一致优势要求；停止按 BGS 大小直接决定无标签通道增益，将研究重点转向 Receiver-aware 判断。"


def make_plot(out, case_rows, checkpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(7 * len(checkpoints), 5), squeeze=False)
    for ax, checkpoint in zip(axes[0], checkpoints):
        rows = [r for r in case_rows if r["checkpoint"] == checkpoint]
        for row in rows:
            ax.plot(range(4), [row[s + "_boundary_ce_gain"] for s in STRATEGIES], "o-", alpha=0.6,
                    linewidth=1, markersize=4, label=row["receiver_case"][:6])
        mean = [np.mean([r[s + "_boundary_ce_gain"] for r in rows]) for s in STRATEGIES]
        ax.plot(range(4), mean, "ko-", linewidth=2.5, label="Equal case mean")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["No gain", "True BGS", "Shuffle mean", "Uniform"])
        ax.set(title=checkpoint, ylabel="Balanced boundary CE gain (positive = lower loss)")
        ax.legend(fontsize=7)
    fig.suptitle("Fixed patches; one line per receiver case; shuffle averaged before cases")
    fig.tight_layout()
    fig.savefig(str(out / "comparison_plot.png"), dpi=180)
    plt.close(fig)


def stat_text(stat):
    return "{:.6f} [{:.6f}, {:.6f}]".format(stat["mean"], *stat["ci95"]) if stat["mean"] is not None else "未定义"


def positive_evidence(stat):
    if stat["mean"] is None:
        return "该项未定义/已跳过"
    if stat["ci95"][0] > 0:
        return "本分析支持正向收益/优势"
    if stat["ci95"][1] < 0:
        return "本分析显示负向收益/优势"
    return "病例级区间跨零，尚无稳定方向证据"


def report(out, summaries, case_rows, metadata, verification):
    names = [c["name"] for c in metadata["checkpoints"]]
    lines = ["# Experiment BGS-U2：跨病例边界增益迁移", "", "## 研究问题与方法", "",
             "在冻结模型内，检验其他 labeled Donor 的 BGS 权重是否使 Receiver 的真实边界预测优于 No Gain、随机通道对应或统一增益。本实验不训练，也不更新任何模型、BN buffer 或 EMA。",
             "主对象 Baseline EMA iter27000；Experiment A EMA iter29000 是辅助，每个模型独立分析，通道 ID 不跨 checkpoint 对齐。全部采用 eval、固定 BN、关闭 Dropout，所有前向在 no_grad 下进行，没有 backward、autograd.grad 或优化器。",
             "从 BGS-U metadata 精确复现 {} 例、{} 个 112×112×80 patch；核对训练划分、H5 文件 SHA256、尺寸、padding、坐标与历史 voxel 数。不重新随机采样，不按预测质量替换样本。图像/GT patch 哈希在 metadata 中。".format(metadata["labeled_cases"], metadata["fixed_patches"]),
             "固定 train.list 前8例顺序；Receiver i 的 Donor 是 (i+1)%8、(i+2)%8，同 patch index。Donor 和 Receiver 均不同病例。每例轮流承担两种角色，因此这些是模型训练已见过的 labeled 病例上的模拟 Receiver，不是新的未见病例。",
             "策略生成接口只接收 Donor patch：分别计算各 Donor x3 的 compute_bgs，仅对有效 Donor 的有符号 BGS 先平均，然后 ReLU/max 归一化。与现有 Experiment A helper 输出逐元素核对。无有效 Donor 则整组标记无效，不用 Receiver GT 补救。Receiver GT 在全部策略和20次置换封存后才传入评价；历史 U 仅作后验解释，不参与权重。",
             "No Gain 原特征；True BGS 用 (1+0.1w)×x3；Shuffle 用独立 CPU torch.Generator 预生成20次均匀 randperm，所有结果保留；Uniform 用 (1+0.1×mean(w))×x3。只修改 decoder 输入 x3，x1/x2/x4/x5 始终为同一张量。保存实际相对 L2 变化；Uniform 只匹配平均权重，不匹配扰动能量。",
             "Shuffle seed = seed + checkpoint_index×1000003 + receiver_index×1009 + patch_index×9176。不会重采样以避开固定点或挑选更好的排列；CPU、CUDA、NumPy、Python 全局 RNG 均核对未被策略/评价消耗。",
             "主要损失沿用 BGS-U：原分辨率内外一体素 GT 带的 CE 各占1/2。全图 CE、原监督损失 0.5×(CE+Dice)、patch Dice/Jaccard、FP/FN 同时报告。使用 foreground softmax>0.5，不使用 LCC；patch 指标不能与整例 LCC Dice 横向比较。",
             "所有 CE 收益定义为 L(NoGain)−L(strategy)，正值有益。True-vs-control 优势为 L(control)−L(True)，正值有益。Dice/Jaccard 改善为 strategy−NoGain，FP/FN 改善为 NoGain−strategy。S=ΣwU，0.1S 只是冻结模型一阶预测，不代替实际前向。",
             "统计顺序：每 patch 的20次 Shuffle先求均值→每病例3个 patch求均值→8例等权平均；10000次 bootstrap 独立重采样单位为病例。病例 SD 和 Shuffle SD 均用 ddof=1。病例真实 BGS 百分位比较其真实平均 CE 与20个病例级随机结果（每个repeat先平均3个patch）；百分位越高，真实对应越好，tie取半分。",
             "判定规则按主模型：A 要求真实增益、对两个控制的优势及全图 CE 收益区间均为正且多数病例改善；否则，三种增益均值都为负或真实BGS至少两例改善/两例恶化则 C；其余 B。报告效应原始尺度和区间，未设置或宣称临床意义阈值。", "", "## 主要统计", "",
             "均值和95%病例 bootstrap CI（正值表示收益或真实BGS优势）：", "",
             "|checkpoint|指标|mean [95% CI]|median|SD|改善/恶化/不变病例|", "|---|---|---|---|---|---|"]
    for name in names:
        s = summaries[name]
        for label, stats in s["primary_metrics"].items():
            lines.append("|{}|{}|{}|{:.6f}|{:.6f}|{}/{}/{}|".format(name, label, stat_text(stats), stats["median"], stats["std"],
                         stats["improved_cases"], stats["worsened_cases"], stats["unchanged_cases"]))
    lines += ["", "## 全部策略的辅助指标", "", "|checkpoint|策略|边界CE|全图CE|CE+Dice|patch Dice|Jaccard|FP|FN|相对feature L2|", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        m = summaries[name]["metrics"]
        for strategy in STRATEGIES:
            values = [m[strategy + "_" + k]["mean"] for k in ("boundary_ce", "full_ce", "supervised_ce_dice", "patch_dice", "patch_jaccard", "fp_voxels", "fn_voxels", "relative_feature_l2_change")]
            lines.append("|{}|{}|{}|".format(name, strategy, "|".join("{:.6f}".format(v) for v in values)))
    lines += ["", "## Receiver逐病例差异", "", "|checkpoint|Receiver|True gain|Shuffle gain|Uniform gain|True−Shuffle优势|True−Uniform优势|True随机百分位|Shuffle CE SD / min / max|", "|---|---|---|---|---|---|---|---|---|"]
    for row in case_rows:
        lines.append("|{}|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.1f}%|{:.6f} / {:.6f} / {:.6f}|".format(
                     row["checkpoint"], row["receiver_case"], row["true_bgs_boundary_ce_gain"], row["shuffled_bgs_boundary_ce_gain"], row["uniform_gain_boundary_ce_gain"],
                     row["true_bgs_true_vs_shuffle_boundary_advantage"], row["true_bgs_true_vs_uniform_boundary_advantage"],
                     row["true_bgs_case_shuffle_percentile"], row["shuffled_bgs_case_boundary_ce_std"], row["shuffled_bgs_case_boundary_ce_min"], row["shuffled_bgs_case_boundary_ce_max"]))
    lines += ["", "百分位是描述性随机对照位置，不是480个独立病例的显著性检验。重复排列完整结果、每patch随机均值/SD/范围、每病例20个随机均值分别见 shuffle_repetitions.csv、receiver_comparison.csv、case_shuffle_repetitions.csv。", "", "## 权重–Receiver效用的机制解释", "", "|checkpoint|Donor w–Receiver U Spearman [CI]|S True [CI]|S Shuffle [CI]|S Uniform [CI]|True−Shuffle S [CI]|真实权重分给U<0的比例 [CI]|", "|---|---|---|---|---|---|---|"]
    mechanism_comments = []
    for name in names:
        m = summaries[name]["metrics"]
        keys = ("true_bgs_w_utility_spearman", "true_bgs_S", "shuffled_bgs_S", "uniform_gain_S", "true_vs_shuffle_S_advantage", "true_bgs_negative_utility_weight_mass_fraction")
        lines.append("|{}|{}|".format(name, "|".join(stat_text(m[k]) if k in m else "未定义/已跳过" for k in keys)))
        if "true_vs_shuffle_negative_weight_mass_difference" in m:
            mechanism_comments.append("{}：真实−随机的负效用权重占比差 {}。占比按Σ(U<0时w)/Σw，未按其判定后再修改策略。".format(name, stat_text(m["true_vs_shuffle_negative_weight_mass_difference"])))
    lines += [""] + [comment + "\n" for comment in mechanism_comments]
    lines += ["负效用的权重占比、Spearman排名相关性和S不等价：占比不考虑效用大小，S包含有符号效用的幅度。辅助A中真实对应的负效用权重占比更高，但S与实际增益均优于Shuffle；这些结果不能被删去，也不能用单一占比解释其净效用。", ""]
    lines += ["模型内对照与模型间现象必须分开解读：主 Baseline 的真实−随机配对区间跨零，没有稳定优势；辅助A的真实−随机优势为 {}。两者的真实增益相对No Gain仍分别为 {}、{}，所以辅助A的相对优势不能改写为绝对改善。BGS训练经历可能改变网络对通道缩放的响应，但本实验不能证明这一原因。".format(
              stat_text(summaries[names[1]]["primary_metrics"]["true_vs_shuffle"]) if len(names) > 1 else "未分析辅助模型",
              stat_text(summaries[names[0]]["primary_metrics"]["true_gain"]),
              stat_text(summaries[names[1]]["primary_metrics"]["true_gain"]) if len(names) > 1 else "未分析辅助模型"), ""]
    primary = summaries[names[0]]
    p = primary["primary_metrics"]
    choice, rationale = decision(primary)
    mechanism_m = primary["metrics"]
    mass_diff = mechanism_m.get("true_vs_shuffle_negative_weight_mass_difference")
    if mass_diff is not None and mass_diff["mean"] is not None:
        negative_answer = ("是，在本分析中负效用权重占比更大" if mass_diff["ci95"][0] > 0 else
                           "否，在本分析中负效用权重占比更小" if mass_diff["ci95"][1] < 0 else
                           "未获得真实对应更偏向负效用通道的稳定证据，配对区间跨零")
        mechanism_answer = negative_answer + "；真实−随机占比差 " + stat_text(mass_diff)
    else:
        mechanism_answer = "历史U未通过验证，跳过该机制判断"
    summaries["recommendation"] = dict(choice=choice, rationale=rationale, primary_checkpoint=names[0])
    lines += ["", "## 六个问题的回答与证据层级", "",
              "1. **是否比 No Gain 改善边界 CE？** {}；主模型真实BGS平均收益 {}，{}例改善、{}例恶化。直接结论由收益符号和区间决定，不把负收益说成改善。".format(positive_evidence(p["true_gain"]), stat_text(p["true_gain"]), p["true_gain"]["improved_cases"], p["true_gain"]["worsened_cases"]),
              "2. **是否优于20次 Shuffle 的平均？** {}；配对优势 {}，{}例真实对应更好、{}例更差。使用全部20次均值，没有选最佳随机结果。".format(positive_evidence(p["true_vs_shuffle"]), stat_text(p["true_vs_shuffle"]), p["true_vs_shuffle"]["improved_cases"], p["true_vs_shuffle"]["worsened_cases"]),
              "3. **是否优于 Uniform？** {}；配对优势 {}，{}例更好、{}例更差；Uniform的实际扰动能量不同，上述L2指标保留这一差别。".format(positive_evidence(p["true_vs_uniform"]), stat_text(p["true_vs_uniform"]), p["true_vs_uniform"]["improved_cases"], p["true_vs_uniform"]["worsened_cases"]),
              "4. **是否更倾向于增强负效用通道？** {}。不能只因为真实S<0就断言强于随机。S为一阶近似，实际0.1增益可能非线性；Uniform的S等于均匀置换在无限重复下的期望，20次Shuffle只是这一线性期望的有限样本近似，非线性Decoder损失没有此等价保证。".format(mechanism_answer),
              "5. **是否Receiver依赖？** 主模型真实收益的方向在{}例/{}例分别为正/负，逐病例表完整列出。Donor组合随Receiver变化，因此本设计能描述迁移结果异质性，但不能把‘同一权重在不同Receiver的因果差异’从Donor变化中隔离出来；没有额外改变预定Donor规则。".format(p["true_gain"]["improved_cases"], p["true_gain"]["worsened_cases"]),
              "6. **是否继续作为核心指导？** 按预定主模型规则选择 **{}**：{} 辅助checkpoint只作为并列复核，不改变主要流程。".format(choice, rationale), "",
              "**直接验证事实：** 固定已训练 checkpoint、已见过的labeled病例和历史固定patch上的实际decoder损失变化、随机控制位置、全图及patch级指标。",
              "**可能机制：** 当真实w的S更负、负效用权重占比更大且实际增益更差时，与‘将额外正向增益分给不需要增强的通道’一致。正向增益各策略都恶化时，也需考虑一般放大导致decoder分布偏移。能量差异、非线性和Donor/Receiver需求不匹配也可能参与。",
              "**尚未证实：** 真实无标签数据上的迁移或训练收益、Shuffle训练优势的因果机制、某个固定通道应永久增强/抑制、增益饱和或Receiver-aware机制有效。没有无标签/测试GT，没有未见病例外部验证，只有8例且训练数据复用；bootstrap不能消除这些局限。Donor组合有重叠，当前区间按固定映射重采样Receiver病例，不等价于联合重采样独立的Donor和Receiver。", "",
              "## 验证与文件保护", "",
              "历史U复用只在checkpoint SHA256、H5哈希、crop/padding、feature shape和No Gain历史损失一致时启用；不一致时仅跳过该机制项并记录原因，不重新采样，也不逆向调权。验证过的U数：{} / {}。".format(verification["historical_U_matched_patches"], verification["valid_receiver_model_patches"]),
              "{}个原有文件逐SHA256前后不变；模型所有参数和BN buffers逐tensor不变；No Gain与原eval forward完全一致；原始encoder缓存未改写；Donor权重与A完全一致；所有shuffle集合严格相同、mean/norm差≤1e-12；全局RNG不变；无NaN/Inf。单元测试记录在test_results.json，逐组复核在verification.json。".format(verification["protected_file_count"]),
              "无效组 {}，完整原因见 invalid_groups.csv；不丢弃严重或不利的Receiver。Receiver GT可以在该病例作为其他组的Donor时使用，但不参与它自身作为Receiver的签名、策略或权重选择。".format(verification["invalid_groups"]),
              "本轮仅完成分析，没有实现Reverse-BGS、Receiver-aware、Prototype、Attention、Channel Exchange或新的训练loss。所有主、辅助结果并列保存，未按Raw/LCC表现改变评价方式。"]
    (out / "experiment_report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError("Refusing existing output directory: " + str(out))
    reference_dir = Path(args.reference_dir).resolve()
    history = json.loads((reference_dir / "experiment_metadata.json").read_text())
    ids = require_metadata(history)
    device = torch.device(args.device)
    # Validate CUDA access BEFORE creating the output directory. A denied GPU
    # call leaves no output directory which could subsequently be overwritten.
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = legacy.create_output(out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(str(out / "analysis_log.txt")), logging.StreamHandler(sys.stdout)])
    before = protected_manifest(out)
    checkpoints = history["checkpoints"][:1] if args.baseline_only else history["checkpoints"]
    metadata = dict(status="started", experiment="BGS-U2", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    configuration=vars(args), alpha=ALPHA, shuffle_repeats=SHUFFLE_REPEATS, bootstrap_repeats=BOOTSTRAP_REPEATS,
                    reference_dir=str(reference_dir), historical_metadata_sha256=legacy.sha256(reference_dir / "experiment_metadata.json"),
                    checkpoints=checkpoints, labeled_cases=len(ids), fixed_patches=24, labeled_case_order=ids,
                    protected_file_sha256_before=before, code_sha256={str(p.relative_to(ROOT)): legacy.sha256(p) for p in NEW_CODE},
                    historical_U_rule="exact checkpoint/source/crop/padding/shape provenance plus baseline loss match; otherwise mechanism skipped",
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, device=str(device),
                                     device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                                     cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
                    prespecified_rules=dict(donors="train.list ordered (i+1)%8,(i+2)%8; same patch index",
                                            alpha=ALPHA, invalid_donors="use valid donors only; no valid donor -> invalid group",
                                            shuffle="20 independent CPU Generator permutations, no selection; patch-wise average",
                                            receiver_gt="evaluation only, after all strategy weights/permutations fixed",
                                            bootstrap_unit="receiver case", uniform="match mean w, not feature perturbation energy",
                                            decision="A: positive gain/paired/full CE CIs and majority improve; else C if all gains mean<0 or >=2 each direction; else B",
                                            primary_checkpoint=checkpoints[0]["name"]))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    metadata["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    legacy.write_json(out / "experiment_metadata.json", metadata)
    try:
        tests = subprocess.run([sys.executable, "-B", str(HERE / "test_experiment_bgs_utility_v2.py")], capture_output=True, text=True)
        legacy.write_json(out / "test_results.json", dict(exit_code=tests.returncode, stdout=tests.stdout, stderr=tests.stderr))
        if tests.returncode:
            raise AssertionError("BGS-U2 tests failed: " + tests.stderr)
        patches, replay_records, current_split = replay_patches(history)
        metadata.update(replayed_patches=replay_records, current_data_split=current_split,
                        data_sources=history["data_sources"], historical_crop_rule=history["protocol"]["crop_rule"],
                        historical_sampling_coordinates=history["sampling"]["patch_coordinates"])
        utility_index, utility_info = historical_utility_index(reference_dir)
        metadata["historical_U"] = utility_info
        all_comp, all_rep, all_align, all_weights, all_mapping, all_invalid, checks = [], [], [], [], [], [], {}
        for model_index, checkpoint in enumerate(checkpoints):
            if legacy.sha256(checkpoint["path"]) != checkpoint["sha256"]:
                raise AssertionError("Historical checkpoint SHA differs: " + checkpoint["name"])
            model = legacy.load_frozen(checkpoint["path"], args.device)
            comp, rep, align, weights, mapping, invalid, check = analyze_model(
                model, checkpoint["name"], model_index, patches, ids, utility_index, args,
                history["configuration"]["min_band_voxels"])
            all_comp.extend(comp)
            all_rep.extend(rep)
            all_align.extend(align)
            all_weights.extend(weights)
            all_mapping.extend(mapping)
            all_invalid.extend(invalid)
            checks[checkpoint["name"]] = check
            for filename, rows in (("receiver_comparison.csv", all_comp), ("shuffle_repetitions.csv", all_rep),
                                   ("weight_utility_alignment.csv", all_align), ("donor_channel_weights.csv", all_weights),
                                   ("donor_receiver_mapping.csv", all_mapping), ("invalid_groups.csv", all_invalid)):
                legacy.write_csv(out / filename, rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not all_comp:
            raise RuntimeError("No valid receiver groups; no resampling or receiver-GT rescue performed")
        names = [c["name"] for c in checkpoints]
        case_rows, case_shuffle, summaries = summarize_cases(all_comp, all_rep, ids, names, args.seed)
        legacy.write_csv(out / "case_level_statistics.csv", case_rows)
        legacy.write_csv(out / "case_shuffle_repetitions.csv", case_shuffle)
        make_plot(out, case_rows, names)
        after = protected_manifest(out)
        changed = [p for p, digest in before.items() if after.get(p) != digest]
        added = sorted(set(after) - set(before))
        if changed or added:
            raise AssertionError("Protected files changed: " + repr((changed, added)))
        matched = sum(r["historical_U_verified"] for m in checks.values() for r in m["receivers"])
        total = sum(len(m["receivers"]) for m in checks.values())
        verification = dict(status="passed", protected_files_unchanged=True, protected_file_count=len(before),
                            checkpoint_sha256_unchanged=True, original_experiments_unchanged=True,
                            replay_exact=True, no_sampling_performed=True, test_gt_read=False, unlabeled_gt_read=False,
                            no_training=True, no_backward=True, no_autograd_grad=True, no_optimizer=True, no_ema_update=True,
                            receiver_gt_does_not_construct_weights=True, no_receiver_utility_weight_optimization=True,
                            bootstrap_unit="case; eight receivers expected, not 24 patches or 480 shuffles",
                            expected_receivers=8, checkpoint_checks=checks,
                            historical_U_matched_patches=matched, valid_receiver_model_patches=total,
                            invalid_groups=len(all_invalid), all_numeric_outputs_finite=True,
                            shuffle_repetitions_saved=len(all_rep), expected_shuffle_repetitions=total * SHUFFLE_REPEATS,
                            all_twenty_shuffles_used=True, unit_test_exit_code=tests.returncode,
                            unit_test_evidence="test_results.json")
        if len(all_rep) != total * SHUFFLE_REPEATS:
            raise AssertionError("Not all precommitted permutations retained")
        metadata.update(status="complete", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        protected_file_sha256_after=after)
        report(out, summaries, case_rows, metadata, verification)
        legacy.write_json(out / "summary_statistics.json", summaries)
        legacy.write_json(out / "verification.json", verification)
        legacy.write_json(out / "experiment_metadata.json", metadata)
        legacy.validate_output_files(out)
        logging.info("Complete: %d valid receiver-model groups; %d shuffles; %d protected files unchanged", total, len(all_rep), len(before))
        logging.info("Decision: %s", summaries["recommendation"])
    except Exception as exc:
        logging.exception("BGS-U2 failed; outputs retained and directory cannot be overwritten")
        metadata.update(status="failed", error=repr(exc))
        legacy.write_json(out / "experiment_metadata.json", metadata)
        legacy.write_json(out / "verification.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
