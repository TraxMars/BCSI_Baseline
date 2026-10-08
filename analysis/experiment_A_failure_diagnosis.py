#!/usr/bin/env python3
"""Experiment A-D: diagnose Experiment A without training or model changes."""

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label as connected_components
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from model.vnet import VNet
from prediction import getLargestCC, test_single_case
from utils.boundary_guidance import apply_labeled_bgs_guidance
from utils.utils import calculate_metric_percase


METRICS = ("dice", "jaccard", "hd95", "asd")
HIGHER_IS_BETTER = {"dice": True, "jaccard": True, "hd95": False, "asd": False}
PROTECTED_SOURCES = (
    "train.py",
    "trainer.py",
    "model/vnet.py",
    "utils/boundary_guidance.py",
    "prediction.py",
)
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BGS_PATTERN = re.compile(
    rf"BGS iteration (?P<iteration>\d+) : valid (?P<valid>\d+) "
    rf"mean (?P<mean>{FLOAT}) max (?P<max>{FLOAT}) "
    rf"positive_fraction (?P<positive>{FLOAT}) w_mean (?P<w_mean>{FLOAT}) "
    rf"w_max (?P<w_max>{FLOAT}) relative_change (?P<relative>{FLOAT}) "
    r"skip (?P<skip>True|False) nonfinite (?P<nonfinite>\d+)"
)


def parse_args():
    run_dir = ROOT / "Results/experiment_A_bgs_alpha0p1_seed42"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT.parent / "Dataset/LA")
    parser.add_argument("--run-dir", type=Path, default=run_dir)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=ROOT / "Results/seed_42/result_LA_10l/fold_0/Model_iter_27000.pth",
    )
    parser.add_argument(
        "--guidance-checkpoint",
        type=Path,
        default=run_dir / "result_LA_10l/fold_0/Model_iter_29000.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results/experiment_A_diagnosis",
    )
    parser.add_argument("--anomaly-case", default="WSJB9P4JCXUVHBOYFVWL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[112, 112, 80])
    parser.add_argument("--bgs-alpha", type=float, default=0.1)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_case_metrics(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate case in {path}: {case_id}")
        result[case_id] = {metric: float(row[metric]) for metric in METRICS}
    return result


def paired_case_analysis(baseline_path, guidance_path, anomaly_case, output_path):
    baseline = read_case_metrics(baseline_path)
    guidance = read_case_metrics(guidance_path)
    if baseline.keys() != guidance.keys() or len(baseline) != 20:
        raise RuntimeError("Expected the same 20 cases in baseline and guidance tables")

    case_ids = list(baseline)
    utility_deltas = {
        metric: np.array(
            [
                (guidance[case][metric] - baseline[case][metric])
                * (1.0 if HIGHER_IS_BETTER[metric] else -1.0)
                for case in case_ids
            ],
            dtype=float,
        )
        for metric in METRICS
    }
    rows = []
    summaries = {}
    tolerance = 1e-12
    for index, case_id in enumerate(case_ids):
        row = {"case_id": case_id, "is_prespecified_anomaly": int(case_id == anomaly_case)}
        for metric in METRICS:
            base = baseline[case_id][metric]
            guided = guidance[case_id][metric]
            raw_delta = guided - base
            utility_delta = utility_deltas[metric][index]
            if utility_delta > tolerance:
                outcome = "improved"
            elif utility_delta < -tolerance:
                outcome = "worsened"
            else:
                outcome = "tie"
            leave_one_out = np.delete(utility_deltas[metric], index)
            row.update(
                {
                    f"baseline_{metric}": base,
                    f"guidance_{metric}": guided,
                    f"delta_{metric}": raw_delta,
                    f"utility_delta_{metric}": utility_delta,
                    f"outcome_{metric}": outcome,
                    f"loo_mean_utility_delta_{metric}": float(leave_one_out.mean()),
                    f"loo_median_utility_delta_{metric}": float(np.median(leave_one_out)),
                }
            )
        rows.append(row)

    for metric in METRICS:
        values = utility_deltas[metric]
        raw_values = values if HIGHER_IS_BETTER[metric] else -values
        loo_means = np.array(
            [float(np.delete(values, index).mean()) for index in range(len(values))]
        )
        worst_index = int(np.argmin(values))
        influential_index = int(np.argmax(np.abs(loo_means - values.mean())))
        summaries[metric] = {
            "improved": int((values > tolerance).sum()),
            "worsened": int((values < -tolerance).sum()),
            "ties": int((np.abs(values) <= tolerance).sum()),
            "paired_mean_raw_delta": float(raw_values.mean()),
            "paired_median_raw_delta": float(np.median(raw_values)),
            "paired_mean_utility_delta": float(values.mean()),
            "paired_median_utility_delta": float(np.median(values)),
            "worst_case": case_ids[worst_index],
            "worst_utility_delta": float(values[worst_index]),
            "loo_mean_utility_delta_min": float(loo_means.min()),
            "loo_mean_utility_delta_max": float(loo_means.max()),
            "most_influential_case": case_ids[influential_index],
            "most_influential_loo_mean": float(loo_means[influential_index]),
        }

    write_csv(output_path, rows)
    return rows, summaries


def linear_slope_per_1000(iterations, values):
    if len(values) < 2 or np.ptp(iterations) == 0:
        return 0.0
    return float(np.polyfit(np.asarray(iterations) / 1000.0, values, 1)[0])


def aggregate_bgs(records, phase, start, end):
    selected = [record for record in records if start <= record["iteration"] <= end]
    if not selected:
        raise RuntimeError(f"No BGS records for phase {phase}")
    row = {
        "phase": phase,
        "iteration_start": selected[0]["iteration"],
        "iteration_end": selected[-1]["iteration"],
        "records": len(selected),
        "skip_rate": float(np.mean([record["skip"] for record in selected])),
        "nonfinite_count": int(sum(record["nonfinite"] for record in selected)),
    }
    iterations = [record["iteration"] for record in selected]
    fields = (
        "valid_samples",
        "mean_bgs",
        "max_bgs",
        "positive_fraction",
        "w_mean",
        "w_max",
        "relative_change",
    )
    for field in fields:
        values = np.asarray([record[field] for record in selected], dtype=float)
        row.update(
            {
                f"{field}_mean": float(values.mean()),
                f"{field}_median": float(np.median(values)),
                f"{field}_std": float(values.std()),
                f"{field}_min": float(values.min()),
                f"{field}_max": float(values.max()),
                f"{field}_first": float(values[0]),
                f"{field}_last": float(values[-1]),
                f"{field}_change": float(values[-1] - values[0]),
                f"{field}_slope_per_1000_iterations": linear_slope_per_1000(
                    iterations, values
                ),
            }
        )
    return row


def bgs_training_statistics(log_path, output_path):
    records = []
    for line in Path(log_path).read_text().splitlines():
        match = BGS_PATTERN.search(line)
        if not match:
            continue
        records.append(
            {
                "iteration": int(match["iteration"]),
                "valid_samples": int(match["valid"]),
                "mean_bgs": float(match["mean"]),
                "max_bgs": float(match["max"]),
                "positive_fraction": float(match["positive"]),
                "w_mean": float(match["w_mean"]),
                "w_max": float(match["w_max"]),
                "relative_change": float(match["relative"]),
                "skip": match["skip"] == "True",
                "nonfinite": int(match["nonfinite"]),
            }
        )
    if len(records) != 30000 or [record["iteration"] for record in records] != list(range(30000)):
        raise RuntimeError("Expected exactly one ordered BGS record for iterations 0..29999")

    rows = []
    for start in range(0, 30000, 5000):
        end = start + 4999
        rows.append(aggregate_bgs(records, f"{start:05d}-{end:05d}", start, end))
    rows.append(aggregate_bgs(records, "overall", 0, 29999))
    write_csv(output_path, rows)
    return rows


def torch_load_state(path, device):
    kwargs = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    state = torch.load(path, map_location=device, **kwargs)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Invalid model state: {path}")
    return state


def load_model(checkpoint, device):
    model = VNet(n_channels=1, n_classes=2).to(device)
    model.load_state_dict(torch_load_state(checkpoint, device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def resolve_case_path(data_path, case_id):
    root = Path(data_path)
    candidate = root / "2018LA_Seg_TrainingSet" / case_id / "mri_norm2.h5"
    if not candidate.is_file():
        candidate = root / case_id / "mri_norm2.h5"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def component_row(name, mask):
    mask = np.asarray(mask, dtype=bool)
    labels = connected_components(mask, connectivity=mask.ndim)
    sizes = np.bincount(labels.reshape(-1))[1:]
    sizes = np.sort(sizes)[::-1]
    total = int(mask.sum())
    return {
        "mask": name,
        "connectivity": 26,
        "component_count": int(len(sizes)),
        "foreground_voxels": total,
        "largest_component_voxels": int(sizes[0]) if len(sizes) else 0,
        "second_component_voxels": int(sizes[1]) if len(sizes) > 1 else 0,
        "largest_component_fraction": float(sizes[0] / total) if total else 0.0,
        "components_lt_100_voxels": int((sizes < 100).sum()),
    }


def select_key_slices(gt, baseline, guidance):
    baseline_error = (baseline != gt).sum(axis=(0, 1))
    guidance_error = (guidance != gt).sum(axis=(0, 1))
    guidance_fn = (gt & ~guidance).sum(axis=(0, 1))
    gt_area = gt.sum(axis=(0, 1))
    priorities = [
        int(np.argmax(guidance_error - baseline_error)),
        int(np.argmax(guidance_fn)),
        int(np.argmax(gt_area)),
    ]
    selected = []
    for index in priorities + list(np.argsort(-(guidance_error - baseline_error))):
        if index not in selected:
            selected.append(int(index))
        if len(selected) == 3:
            break
    return selected


def mask_overlay(image, positive, color):
    low, high = np.percentile(image, [1, 99])
    base = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat(base[..., None], 3, axis=2)
    color = np.asarray(color, dtype=float)
    rgb[positive] = 0.35 * rgb[positive] + 0.65 * color
    return rgb


def error_overlay(image, gt, prediction):
    low, high = np.percentile(image, [1, 99])
    base = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat(base[..., None], 3, axis=2)
    false_positive = prediction & ~gt
    false_negative = gt & ~prediction
    rgb[false_positive] = 0.25 * rgb[false_positive] + 0.75 * np.array([1.0, 0.1, 0.1])
    rgb[false_negative] = 0.25 * rgb[false_negative] + 0.75 * np.array([0.1, 0.4, 1.0])
    return rgb


def save_key_slice_figure(path, image, gt, baseline, guidance, slices):
    columns = ("MRI", "GT", "Baseline", "Guidance", "Baseline FP/FN", "Guidance FP/FN")
    figure, axes = plt.subplots(len(slices), len(columns), figsize=(18, 4.6 * len(slices)))
    axes = np.atleast_2d(axes)
    for row, z_index in enumerate(slices):
        image_slice = image[:, :, z_index]
        gt_slice = gt[:, :, z_index]
        baseline_slice = baseline[:, :, z_index]
        guidance_slice = guidance[:, :, z_index]
        panels = (
            image_slice,
            mask_overlay(image_slice, gt_slice, [0.1, 1.0, 0.2]),
            mask_overlay(image_slice, baseline_slice, [1.0, 0.85, 0.1]),
            mask_overlay(image_slice, guidance_slice, [0.9, 0.1, 1.0]),
            error_overlay(image_slice, gt_slice, baseline_slice),
            error_overlay(image_slice, gt_slice, guidance_slice),
        )
        for column, panel in enumerate(panels):
            axes[row, column].imshow(np.rot90(panel), cmap="gray" if column == 0 else None)
            axes[row, column].set_title(f"{columns[column]} | z={z_index}")
            axes[row, column].axis("off")
    figure.suptitle("Red = false positive; blue = false negative", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def infer_anomaly(args, output_dir, expected_metrics):
    case_id = args.anomaly_case
    test_cases = (args.data_path / "test.list").read_text().splitlines()
    if case_id not in test_cases:
        raise RuntimeError(f"Anomaly case is not in test.list: {case_id}")
    with h5py.File(resolve_case_path(args.data_path, case_id), "r") as handle:
        image = handle["image"][:].astype(np.float32)
        gt = handle["label"][:].astype(bool)

    predictions = {}
    raw_predictions = {}
    measured = {}
    for name, checkpoint in (
        ("baseline", args.baseline_checkpoint),
        ("guidance", args.guidance_checkpoint),
    ):
        model = load_model(checkpoint, args.device)
        with torch.inference_mode():
            raw, _ = test_single_case(
                args,
                model,
                image,
                stride_xy=18,
                stride_z=4,
                patch_size=args.patch_size,
                num_classes=2,
            )
        final = getLargestCC(raw).astype(bool)
        raw_predictions[name] = raw.astype(bool)
        predictions[name] = final
        values = calculate_metric_percase(torch.from_numpy(gt), final.astype(np.uint8))
        measured[name] = dict(zip(METRICS, map(float, values)))
        for metric in METRICS:
            if not math.isclose(
                measured[name][metric], expected_metrics[name][case_id][metric], rel_tol=0, abs_tol=1e-6
            ):
                raise RuntimeError(f"Raw inference disagrees with saved {name} {metric}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = predictions["baseline"]
    guidance = predictions["guidance"]
    failure_dir = output_dir / "failure_visualizations"
    failure_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        failure_dir / f"{case_id}_masks.npz",
        image=image,
        gt_mask=gt.astype(np.uint8),
        baseline_raw_prediction=raw_predictions["baseline"].astype(np.uint8),
        guidance_raw_prediction=raw_predictions["guidance"].astype(np.uint8),
        baseline_prediction=baseline.astype(np.uint8),
        guidance_prediction=guidance.astype(np.uint8),
        baseline_false_positive=(baseline & ~gt).astype(np.uint8),
        baseline_false_negative=(gt & ~baseline).astype(np.uint8),
        guidance_false_positive=(guidance & ~gt).astype(np.uint8),
        guidance_false_negative=(gt & ~guidance).astype(np.uint8),
    )

    component_rows = [component_row("gt", gt)]
    for name in ("baseline", "guidance"):
        component_rows.append(component_row(f"{name}_raw", raw_predictions[name]))
        component_rows.append(component_row(f"{name}_final", predictions[name]))
    write_csv(failure_dir / f"{case_id}_component_statistics.csv", component_rows)

    slices = select_key_slices(gt, baseline, guidance)
    slice_rows = []
    for z_index in slices:
        slice_rows.append(
            {
                "z_index": z_index,
                "gt_voxels": int(gt[:, :, z_index].sum()),
                "baseline_error_voxels": int((baseline[:, :, z_index] != gt[:, :, z_index]).sum()),
                "guidance_error_voxels": int((guidance[:, :, z_index] != gt[:, :, z_index]).sum()),
                "guidance_minus_baseline_error_voxels": int(
                    (guidance[:, :, z_index] != gt[:, :, z_index]).sum()
                    - (baseline[:, :, z_index] != gt[:, :, z_index]).sum()
                ),
            }
        )
    write_csv(failure_dir / f"{case_id}_key_slices.csv", slice_rows)
    save_key_slice_figure(
        failure_dir / f"{case_id}_key_slices.png", image, gt, baseline, guidance, slices
    )

    gt_volume = int(gt.sum())
    summary = {
        "case_id": case_id,
        "shape": list(gt.shape),
        "metrics": measured,
        "gt_foreground_voxels": gt_volume,
        "baseline_foreground_voxels": int(baseline.sum()),
        "guidance_foreground_voxels": int(guidance.sum()),
        "baseline_to_gt_volume_ratio": float(baseline.sum() / gt_volume),
        "guidance_to_gt_volume_ratio": float(guidance.sum() / gt_volume),
        "baseline_false_positive_voxels": int((baseline & ~gt).sum()),
        "baseline_false_negative_voxels": int((gt & ~baseline).sum()),
        "guidance_false_positive_voxels": int((guidance & ~gt).sum()),
        "guidance_false_negative_voxels": int((gt & ~guidance).sum()),
        "components": component_rows,
        "key_slices": slice_rows,
    }
    (failure_dir / f"{case_id}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def center_crop(array, output_shape):
    padding = []
    for size, target in zip(array.shape, output_shape):
        total = max(target - size, 0)
        padding.append((total // 2, total - total // 2))
    if any(before or after for before, after in padding):
        array = np.pad(array, padding, mode="constant")
    starts = [(size - target) // 2 for size, target in zip(array.shape, output_shape)]
    slices = tuple(slice(start, start + target) for start, target in zip(starts, output_shape))
    return array[slices]


def fixed_bn_batch(data_path, patch_size, device):
    case_ids = (Path(data_path) / "train.list").read_text().splitlines()
    indices = (0, 1, 8, 9)
    images, labels = [], []
    for index in indices:
        with h5py.File(resolve_case_path(data_path, case_ids[index]), "r") as handle:
            images.append(center_crop(handle["image"][:], patch_size))
            labels.append(center_crop(handle["label"][:], patch_size))
    image_tensor = torch.from_numpy(np.stack(images).astype(np.float32)).unsqueeze(1).to(device)
    label_tensor = torch.from_numpy(np.stack(labels).astype(np.int64)).to(device)
    return case_ids, indices, image_tensor, label_tensor


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def clone_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def bn_state(module):
    return {
        name: {
            "running_mean": layer.running_mean.detach().cpu().clone(),
            "running_var": layer.running_var.detach().cpu().clone(),
            "num_batches_tracked": int(layer.num_batches_tracked.item()),
        }
        for name, layer in module.named_modules()
        if isinstance(layer, nn.BatchNorm3d)
    }


def norm(value):
    return float(torch.linalg.vector_norm(value.double()).item())


def decoder_branch(decoder, state, features, mode, rng_state):
    decoder.load_state_dict(state, strict=True)
    decoder.train(mode == "train")
    restore_rng(rng_state)
    with torch.no_grad():
        logits = decoder(features).detach().cpu()
    return logits, bn_state(decoder)


def batchnorm_coupling(args, output_path):
    caller_rng = capture_rng()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = load_model(args.guidance_checkpoint, args.device)
    original_model_state = clone_state(model)
    original_training = model.training
    try:
        case_ids, indices, images, labels = fixed_bn_batch(
            args.data_path, tuple(args.patch_size), args.device
        )
        model.eval()
        with torch.no_grad():
            original_features = model.encoder(images)
            guided_x3, weights, diagnostics = apply_labeled_bgs_guidance(
                original_features[2], labels[:2], labeled_bs=2, alpha=args.bgs_alpha
            )
        if diagnostics["skipped"]:
            raise RuntimeError("Fixed BatchNorm diagnostic batch produced no valid BGS sample")
        guided_features = list(original_features)
        guided_features[2] = guided_x3
        decoder_state = clone_state(model.decoder)
        initial_bn = bn_state(model.decoder)
        paired_rng = capture_rng()
        rows = []
        summaries = {}
        for mode in ("train", "eval"):
            original_logits, original_bn = decoder_branch(
                model.decoder, decoder_state, original_features, mode, paired_rng
            )
            guided_logits, guided_bn = decoder_branch(
                model.decoder, decoder_state, guided_features, mode, paired_rng
            )
            labeled_delta = guided_logits[:2] - original_logits[:2]
            unlabeled_delta = guided_logits[2:] - original_logits[2:]
            summary = {
                "mode": mode,
                "labeled_logits_mean_abs_diff": float(labeled_delta.abs().mean().item()),
                "labeled_logits_max_abs_diff": float(labeled_delta.abs().max().item()),
                "labeled_logits_relative_l2": float(
                    norm(labeled_delta) / (norm(original_logits[:2]) + 1e-12)
                ),
                "unlabeled_logits_mean_abs_diff": float(unlabeled_delta.abs().mean().item()),
                "unlabeled_logits_max_abs_diff": float(unlabeled_delta.abs().max().item()),
                "unlabeled_logits_relative_l2": float(
                    norm(unlabeled_delta) / (norm(original_logits[2:]) + 1e-12)
                ),
            }
            summaries[mode] = summary
            rows.append(
                {
                    **summary,
                    "layer": "__logits__",
                    "original_running_mean_change_l2": "",
                    "guided_running_mean_change_l2": "",
                    "branch_running_mean_difference_l2": "",
                    "original_running_var_change_l2": "",
                    "guided_running_var_change_l2": "",
                    "branch_running_var_difference_l2": "",
                    "original_num_batches_increment": "",
                    "guided_num_batches_increment": "",
                    "valid_bgs_samples": diagnostics["valid_samples"],
                    "w_mean": diagnostics["w_mean"],
                    "w_max": diagnostics["w_max"],
                    "relative_x3_change": diagnostics["relative_change"],
                }
            )
            for layer in initial_bn:
                initial = initial_bn[layer]
                branch_original = original_bn[layer]
                branch_guided = guided_bn[layer]
                rows.append(
                    {
                        **summary,
                        "layer": layer,
                        "original_running_mean_change_l2": norm(
                            branch_original["running_mean"] - initial["running_mean"]
                        ),
                        "guided_running_mean_change_l2": norm(
                            branch_guided["running_mean"] - initial["running_mean"]
                        ),
                        "branch_running_mean_difference_l2": norm(
                            branch_guided["running_mean"] - branch_original["running_mean"]
                        ),
                        "original_running_var_change_l2": norm(
                            branch_original["running_var"] - initial["running_var"]
                        ),
                        "guided_running_var_change_l2": norm(
                            branch_guided["running_var"] - initial["running_var"]
                        ),
                        "branch_running_var_difference_l2": norm(
                            branch_guided["running_var"] - branch_original["running_var"]
                        ),
                        "original_num_batches_increment": branch_original[
                            "num_batches_tracked"
                        ]
                        - initial["num_batches_tracked"],
                        "guided_num_batches_increment": branch_guided["num_batches_tracked"]
                        - initial["num_batches_tracked"],
                        "valid_bgs_samples": diagnostics["valid_samples"],
                        "w_mean": diagnostics["w_mean"],
                        "w_max": diagnostics["w_max"],
                        "relative_x3_change": diagnostics["relative_change"],
                    }
                )
        write_csv(output_path, rows)
        result = {
            "fixed_batch_case_ids": [case_ids[index] for index in indices],
            "fixed_batch_indices": list(indices),
            "diagnostics": diagnostics,
            "summaries": summaries,
            "max_train_bn_mean_branch_difference": max(
                float(row["branch_running_mean_difference_l2"])
                for row in rows
                if row["mode"] == "train" and row["layer"] != "__logits__"
            ),
            "max_train_bn_var_branch_difference": max(
                float(row["branch_running_var_difference_l2"])
                for row in rows
                if row["mode"] == "train" and row["layer"] != "__logits__"
            ),
        }
    finally:
        model.load_state_dict(original_model_state, strict=True)
        model.train(original_training)
        restored = clone_state(model)
        if any(not torch.equal(original_model_state[key], restored[key]) for key in original_model_state):
            raise RuntimeError("BatchNorm diagnostic did not restore model state exactly")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("BatchNorm diagnostic unexpectedly created gradients")
        restore_rng(caller_rng)
    return result


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def generate_report(
    path,
    args,
    paired_summary,
    bgs_rows,
    anomaly,
    bn_result,
    protected_unchanged,
):
    overall = next(row for row in bgs_rows if row["phase"] == "overall")
    phase_rows = [row for row in bgs_rows if row["phase"] != "overall"]
    lines = [
        "# Experiment A-D — Failure Mechanism Diagnosis",
        "",
        "本诊断没有训练、调参或实现 A+。正式配对汇总保留全部 20 个病例，"
        "异常病例没有被排除。Baseline 与 Experiment A 的 checkpoint 分别为 iteration 27000 和 29000。",
        "",
        "## 已经直接验证的事实",
        "",
        f"- 五个受保护源文件在诊断前后 SHA256 均未变化：`{protected_unchanged}`。",
        "- 异常病例重新执行了原始滑窗推理、0.5 阈值和最大连通域后处理；重算指标与已有逐病例 CSV 一致。",
        "- BatchNorm 检查全程使用 `torch.no_grad()`，没有反向传播或参数更新；"
        "完成后模型参数与 buffers 逐张量恢复。",
        "",
        "### 全部 20 例 paired metric delta",
        "",
        "`delta = guidance - baseline`；HD95/ASD 的负 delta 表示改善。"
        "`utility delta` 已统一为正值表示改善。",
        "",
    ]
    metric_rows = []
    for metric in METRICS:
        summary = paired_summary[metric]
        metric_rows.append(
            (
                metric,
                summary["improved"],
                summary["worsened"],
                summary["ties"],
                f"{summary['paired_mean_raw_delta']:.6f}",
                f"{summary['paired_median_raw_delta']:.6f}",
                summary["worst_case"],
                f"{summary['worst_utility_delta']:.6f}",
                f"[{summary['loo_mean_utility_delta_min']:.6f}, {summary['loo_mean_utility_delta_max']:.6f}]",
            )
        )
    lines += markdown_table(
        [
            "metric",
            "improved",
            "worsened",
            "tie",
            "mean raw delta",
            "paired median raw delta",
            "worst case",
            "worst utility delta",
            "LOO mean utility range",
        ],
        metric_rows,
    )
    lines += [
        "",
        "leave-one-out 只用于敏感性分析；正式均值、median 和病例计数均使用完整 20 例。",
        "",
        f"### 预先指定异常病例 `{anomaly['case_id']}`",
        "",
        f"- GT 前景体积：{anomaly['gt_foreground_voxels']} voxels。",
        f"- Baseline 预测体积：{anomaly['baseline_foreground_voxels']} voxels "
        f"（GT 比例 {anomaly['baseline_to_gt_volume_ratio']:.4f}）。",
        f"- Guidance 预测体积：{anomaly['guidance_foreground_voxels']} voxels "
        f"（GT 比例 {anomaly['guidance_to_gt_volume_ratio']:.4f}）。",
        f"- Baseline FP/FN：{anomaly['baseline_false_positive_voxels']} / "
        f"{anomaly['baseline_false_negative_voxels']} voxels。",
        f"- Guidance FP/FN：{anomaly['guidance_false_positive_voxels']} / "
        f"{anomaly['guidance_false_negative_voxels']} voxels。",
        "",
    ]
    anomaly_metric_rows = []
    for metric in METRICS:
        baseline = anomaly["metrics"]["baseline"][metric]
        guidance = anomaly["metrics"]["guidance"][metric]
        anomaly_metric_rows.append((metric, f"{baseline:.6f}", f"{guidance:.6f}", f"{guidance-baseline:.6f}"))
    lines += markdown_table(["metric", "baseline", "guidance", "delta"], anomaly_metric_rows)
    lines += [
        "",
        "3D component 明细、原始阈值 mask、最大连通域后 mask、FP/FN 与关键切片图保存在 "
        "`failure_visualizations/`。数据没有 voxel spacing，因此体积只报告 voxel 数，不伪造物理体积。",
        "",
        "### BGS 训练日志",
        "",
        f"- 共解析 {overall['records']} 个 iteration；guidance skip rate="
        f"{overall['skip_rate']:.6f}，nonfinite count={overall['nonfinite_count']}。",
        f"- valid labeled samples：mean={overall['valid_samples_mean']:.6f}，"
        f"min={overall['valid_samples_min']:.0f}，max={overall['valid_samples_max']:.0f}。",
        f"- mean BGS={overall['mean_bgs_mean']:.6f}；positive fraction="
        f"{overall['positive_fraction_mean']:.6f}。",
        f"- w.mean={overall['w_mean_mean']:.6f}；w.max={overall['w_max_mean']:.6f}；"
        f"relative change={overall['relative_change_mean']:.6f}。",
        "",
    ]
    lines += markdown_table(
        ["phase", "w.mean", "w.mean change", "w.max", "relative change", "skip rate"],
        [
            (
                row["phase"],
                f"{row['w_mean_mean']:.6f}",
                f"{row['w_mean_change']:.6f}",
                f"{row['w_max_mean']:.6f}",
                f"{row['relative_change_mean']:.6f}",
                f"{row['skip_rate']:.6f}",
            )
            for row in phase_rows
        ],
    )
    lines += ["", "每个阶段固定为 5,000 iterations；完整分布、首尾变化和线性斜率见 `bgs_training_statistics.csv`。", ""]

    bn_rows = []
    for mode in ("train", "eval"):
        summary = bn_result["summaries"][mode]
        bn_rows.append(
            (
                mode,
                f"{summary['labeled_logits_mean_abs_diff']:.10f}",
                f"{summary['labeled_logits_max_abs_diff']:.10f}",
                f"{summary['labeled_logits_relative_l2']:.10f}",
                f"{summary['unlabeled_logits_mean_abs_diff']:.10f}",
            )
        )
    lines += ["### Decoder BatchNorm coupling", ""]
    lines += markdown_table(
        ["decoder mode", "labeled mean abs diff", "labeled max abs diff", "labeled relative L2", "unlabeled mean abs diff"],
        bn_rows,
    )
    lines += [
        "",
        f"固定 batch：`{', '.join(bn_result['fixed_batch_case_ids'])}`。BGS valid samples="
        f"{bn_result['diagnostics']['valid_samples']}，w.mean={bn_result['diagnostics']['w_mean']:.6f}，"
        f"x3 relative change={bn_result['diagnostics']['relative_change']:.6f}。",
        f"train mode 下两分支最大的 BN running-mean 差异 L2="
        f"{bn_result['max_train_bn_mean_branch_difference']:.10f}，running-var 差异 L2="
        f"{bn_result['max_train_bn_var_branch_difference']:.10f}。逐层数据见 `batchnorm_coupling.csv`。",
        "",
        "## 尚未验证的原因假设",
        "",
        "- 固定 batch 上已直接观察到 train mode 的 labeled logits 差异，而 eval mode 差异为 0，"
        "验证了混合 batch statistics 的耦合存在；但本诊断没有做反事实重训练，因此不能断言它造成了最终 Dice 下降。",
        "- 异常病例的体积偏差、FP/FN 和连通域变化描述了失败形态；它们不能单独区分"
        "训练期优化偏移、病例解剖差异或阈值/后处理敏感性。",
        "- BGS 权重随训练阶段的漂移是相关性证据。没有不同 alpha、冻结 BN 或独立 seed 对照，"
        "不能把该漂移解释为因果机制。",
        "- 本实验 validation 与最终 test 使用同一 `test.list`，因此不能据此声称独立测试集泛化机制。",
        "",
        "## 产物",
        "",
        "- `paired_case_analysis.csv`：20 例完整 paired delta 与逐例 leave-one-out。",
        "- `bgs_training_statistics.csv`：每 5,000 iterations 及 overall 的 BGS/weight 统计。",
        "- `batchnorm_coupling.csv`：train/eval decoder logits 差异与逐层 BN buffers 变化。",
        "- `failure_visualizations/`：异常病例 masks、FP/FN、3D components 和关键切片。",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main():
    args = parse_args()
    args.data_path = args.data_path.resolve()
    args.run_dir = args.run_dir.resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    args.guidance_checkpoint = args.guidance_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnosis: {args.output_dir}")
    for path in (args.baseline_checkpoint, args.guidance_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    protected_before = {relative: sha256(ROOT / relative) for relative in PROTECTED_SOURCES}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    baseline_table = args.run_dir / "baseline_test.csv"
    guidance_table = args.run_dir / "guidance_test.csv"
    expected = {
        "baseline": read_case_metrics(baseline_table),
        "guidance": read_case_metrics(guidance_table),
    }
    try:
        _, paired_summary = paired_case_analysis(
            baseline_table,
            guidance_table,
            args.anomaly_case,
            temporary / "paired_case_analysis.csv",
        )
        bgs_rows = bgs_training_statistics(
            args.run_dir / "result_LA_10l/fold_0/log.txt",
            temporary / "bgs_training_statistics.csv",
        )
        anomaly = infer_anomaly(args, temporary, expected)
        bn_result = batchnorm_coupling(args, temporary / "batchnorm_coupling.csv")
        protected_after = {relative: sha256(ROOT / relative) for relative in PROTECTED_SOURCES}
        if protected_before != protected_after:
            raise RuntimeError("A protected training/inference source changed during diagnosis")
        generate_report(
            temporary / "diagnosis_report.md",
            args,
            paired_summary,
            bgs_rows,
            anomaly,
            bn_result,
            protected_unchanged=True,
        )
        metadata = {
            "experiment": "A-D Failure Mechanism Diagnosis",
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "baseline_checkpoint_sha256": sha256(args.baseline_checkpoint),
            "guidance_checkpoint": str(args.guidance_checkpoint),
            "guidance_checkpoint_sha256": sha256(args.guidance_checkpoint),
            "device": str(args.device),
            "anomaly_case": args.anomaly_case,
            "all_20_cases_included": True,
            "training_performed": False,
            "backpropagation_performed": False,
            "protected_sources_sha256": protected_after,
            "batchnorm_fixed_batch": bn_result["fixed_batch_case_ids"],
        }
        (temporary / "diagnosis_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        os.replace(temporary, args.output_dir)
    except BaseException:
        print(f"Partial diagnostic output retained at {temporary}", file=sys.stderr)
        raise
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
