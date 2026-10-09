"""Labeled BGS signature and unlabeled x3 gain; no learnable parameters."""
import torch
import torch.nn.functional as F

EPS = 1e-6


def resolve_guidance_mode(args):
    mode = getattr(args, "guidance_mode", None)
    if mode is None:
        return "bgs" if getattr(args, "use_bgs_guidance", False) else "none"
    if mode not in ("none", "bgs", "shuffled_bgs"):
        raise ValueError("Unknown guidance mode: " + str(mode))
    if mode == "none" and getattr(args, "use_bgs_guidance", False):
        raise ValueError("Conflicting guidance mode and legacy flag")
    return mode


@torch.no_grad()
def shuffle_bgs_weights(w, generator):
    """Permute weights using only a dedicated CPU RNG, without renormalization."""
    if generator is None or generator.device.type != "cpu":
        raise ValueError("Shuffling requires a dedicated CPU torch.Generator")
    permutation = torch.randperm(w.numel(), generator=generator, device="cpu").to(w.device)
    shuffled = w.index_select(0, permutation).detach()
    same_distribution = torch.equal(w.sort().values, shuffled.sort().values)
    # Reordering can change the last bit of floating-point reductions. The
    # multiset check is exact; direct FP64 reductions are checked at 1e-12.
    mean_delta = float((w.double().mean() - shuffled.double().mean()).abs().item())
    norm_delta = float((w.double().norm() - shuffled.double().norm()).abs().item())
    if not same_distribution or mean_delta > 1e-12 or norm_delta > 1e-12:
        raise AssertionError("BGS permutation changed the weight distribution")
    return shuffled, permutation, dict(shuffle_distribution_equal=same_distribution,
                                      shuffle_mean_delta=mean_delta, shuffle_norm_delta=norm_delta)


def morphology_3d(mask, iterations=1, operation="dilate"):
    if operation not in ("dilate", "erode") or iterations < 0:
        raise ValueError("Expected nonnegative iterations and dilate/erode")
    result = mask.float()
    for _ in range(iterations):
        if operation == "erode":
            result = 1 - F.max_pool3d(1 - result, 3, 1, 1)
        else:
            result = F.max_pool3d(result, 3, 1, 1)
    return result > 0.5


def build_boundary_and_nonboundary(labels, spatial_shape):
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("Labels must have shape [B,D,H,W] or [B,1,D,H,W]")
    mask = F.interpolate(labels.float(), size=tuple(spatial_shape), mode="nearest") > 0.5
    e1 = morphology_3d(mask, operation="erode")
    d1 = morphology_3d(mask)
    boundary = d1 & ~e1
    nonboundary = ~morphology_3d(boundary)
    return boundary, nonboundary


def spatial_gradient_3d(feature):
    if feature.ndim != 5 or min(feature.shape[2:]) < 2:
        raise ValueError("Expected [B,C,D,H,W] with spatial dimensions >=2")
    gx = F.pad((feature[:, :, 1:] - feature[:, :, :-1]).abs(), (0, 0, 0, 0, 0, 1), mode="replicate")
    gy = F.pad((feature[:, :, :, 1:] - feature[:, :, :, :-1]).abs(), (0, 0, 0, 1, 0, 0), mode="replicate")
    gz = F.pad((feature[:, :, :, :, 1:] - feature[:, :, :, :, :-1]).abs(), (0, 1, 0, 0, 0, 0), mode="replicate")
    return (gx + gy + gz) / 3


@torch.no_grad()
def compute_bgs(feature, labels, min_band_voxels=8):
    """Return detached [B,C] BGS and valid [B] flags; invalid rows are zero.

    Float64 region means reproduce Experiment 0.5. No analysis code is imported.
    """
    if feature.shape[0] != labels.shape[0]:
        raise ValueError("Feature/label batch sizes differ")
    if not torch.isfinite(feature).all():
        raise FloatingPointError("Nonfinite labeled feature in BGS computation")
    boundary, nonboundary = build_boundary_and_nonboundary(labels, feature.shape[2:])
    boundary_count = boundary.flatten(1).sum(dim=1)
    nonboundary_count = nonboundary.flatten(1).sum(dim=1)
    valid = (boundary_count >= min_band_voxels) & (nonboundary_count >= min_band_voxels)
    gradient = spatial_gradient_3d(feature.detach()).double()
    g_boundary = torch.where(boundary, gradient, torch.zeros_like(gradient)).flatten(2).sum(dim=2)
    g_nonboundary = torch.where(nonboundary, gradient, torch.zeros_like(gradient)).flatten(2).sum(dim=2)
    g_boundary = g_boundary / boundary_count.clamp_min(1).unsqueeze(1)
    g_nonboundary = g_nonboundary / nonboundary_count.clamp_min(1).unsqueeze(1)
    bgs = (g_boundary - g_nonboundary) / (g_boundary + g_nonboundary + EPS)
    bgs = torch.where(valid.unsqueeze(1), bgs, torch.zeros_like(bgs))
    if not torch.isfinite(bgs).all():
        raise FloatingPointError("Nonfinite BGS")
    return bgs.detach(), valid


def apply_labeled_bgs_guidance(x3, labeled_labels, labeled_bs, alpha=0.1, min_band_voxels=8,
                               shuffle_generator=None):
    """Use only labeled GT/features for detached weights; rescale unlabeled x3."""
    if not 0 < labeled_bs < x3.shape[0] or labeled_labels.shape[0] != labeled_bs:
        raise ValueError("Guidance needs labeled prefix and unlabeled suffix")
    if not torch.isfinite(x3).all():
        raise FloatingPointError("Nonfinite x3 feature")
    bgs, valid = compute_bgs(x3[:labeled_bs], labeled_labels, min_band_voxels)
    with torch.no_grad():
        valid_count = int(valid.sum().item())
        if valid_count:
            signature = bgs[valid].mean(dim=0).detach()
            positive = signature.relu()
            w = (positive / (positive.max() + EPS)).to(dtype=x3.dtype).detach()
            diagnostics = dict(valid_samples=valid_count, mean_bgs=float(signature.mean().item()),
                               max_bgs=float(signature.max().item()),
                               positive_fraction=float((signature > 0).float().mean().item()),
                               w_mean=float(w.mean().item()), w_max=float(w.max().item()),
                               skipped=False, nonfinite=0)
        else:
            w = x3.new_zeros(x3.shape[1])
            diagnostics = dict(valid_samples=0, mean_bgs=0.0, max_bgs=0.0,
                               positive_fraction=0.0, w_mean=0.0, w_max=0.0,
                               skipped=True, nonfinite=0)
    if not torch.isfinite(w).all():
        raise FloatingPointError("Nonfinite detached BGS weight")
    if not valid_count:
        diagnostics["relative_change"] = 0.0
        return x3, w, diagnostics
    if shuffle_generator is not None:
        w, _, shuffle_diagnostics = shuffle_bgs_weights(w, shuffle_generator)
        diagnostics.update(shuffle_diagnostics)
    fu = x3[labeled_bs:]
    fu_new = fu + alpha * w[None, :, None, None, None] * fu
    x3_new = torch.cat((x3[:labeled_bs], fu_new), dim=0)
    with torch.no_grad():
        if not torch.isfinite(fu_new).all():
            raise FloatingPointError("Nonfinite guided unlabeled feature")
        diagnostics["relative_change"] = float(((fu_new - fu).norm() / (fu.norm() + EPS)).item())
    return x3_new, w, diagnostics
