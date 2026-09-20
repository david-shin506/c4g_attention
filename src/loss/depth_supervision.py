from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PoseAlignedDepthLabels:
    """VGGT depth expressed in the same units as the renderer output."""

    depth: Tensor
    mask: Tensor
    scale: Tensor
    pose_relative_error: Tensor
    alignment_valid: Tensor


def _quat_xyzw_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert VGGT's scalar-last (x, y, z, w) quaternion to a matrix."""

    i, j, k, r = quaternion.unbind(dim=-1)
    two_s = 2.0 / quaternion.square().sum(dim=-1).clamp_min(1e-12)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(*quaternion.shape[:-1], 3, 3)


def vggt_pose_encoding_to_w2c(pose_encoding: Tensor) -> Tensor:
    """Decode VGGT ``absT_quaR_FoV`` poses into OpenCV world-to-camera matrices.

    VGGT stores W2C translation first and an XYZW quaternion second.  The final
    two FOV values are intentionally not used here.
    """

    if pose_encoding.shape[-1] != 9:
        raise ValueError(f"Expected VGGT pose encoding [..., 9], got {pose_encoding.shape}")
    rotation = _quat_xyzw_to_matrix(pose_encoding[..., 3:7])
    result = torch.eye(
        4, dtype=pose_encoding.dtype, device=pose_encoding.device
    ).expand(*pose_encoding.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = pose_encoding[..., :3]
    return result


def estimate_pose_scale(
    predicted_centers: Tensor,
    dataset_centers: Tensor,
    *,
    min_baseline: float = 1e-4,
    min_pair_fraction: float = 0.2,
    max_relative_error: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate the Sim(3) scale from corresponding camera-center baselines.

    Pairwise distances remove the unknown global rotation and translation.  A
    median over every usable pair is more robust than selecting one baseline.
    Static-camera sequences are deliberately marked invalid: their metric scale
    cannot be recovered from pose alignment without an additional scale cue.
    """

    if predicted_centers.shape != dataset_centers.shape:
        raise ValueError(
            "Predicted and dataset camera centers must have the same shape, got "
            f"{predicted_centers.shape} and {dataset_centers.shape}"
        )
    if predicted_centers.ndim != 3 or predicted_centers.shape[-1] != 3:
        raise ValueError(f"Expected [B, V, 3] camera centers, got {predicted_centers.shape}")
    if not 0.0 <= min_pair_fraction < 1.0:
        raise ValueError("min_pair_fraction must be in [0, 1)")

    batch_size, num_views, _ = predicted_centers.shape
    scales = predicted_centers.new_ones(batch_size)
    errors = predicted_centers.new_full((batch_size,), float("inf"))
    valid_alignments = torch.zeros(batch_size, dtype=torch.bool, device=predicted_centers.device)

    if num_views < 2:
        return scales, errors, valid_alignments

    for batch_index in range(batch_size):
        predicted_distances = torch.pdist(predicted_centers[batch_index].float())
        dataset_distances = torch.pdist(dataset_centers[batch_index].float())
        pair_mask = (
            torch.isfinite(predicted_distances)
            & torch.isfinite(dataset_distances)
            & (predicted_distances > min_baseline)
            & (dataset_distances > min_baseline)
        )
        if not pair_mask.any():
            continue

        # Very short baselines amplify small pose errors into extreme scale
        # ratios. Estimate scale only from pairs that are well separated in
        # both trajectories, while retaining every view for the depth labels.
        predicted_max = predicted_distances[pair_mask].max()
        dataset_max = dataset_distances[pair_mask].max()
        pair_mask = (
            pair_mask
            & (predicted_distances >= min_pair_fraction * predicted_max)
            & (dataset_distances >= min_pair_fraction * dataset_max)
        )
        if not pair_mask.any():
            continue

        predicted_distances = predicted_distances[pair_mask]
        dataset_distances = dataset_distances[pair_mask]
        scale = torch.median(dataset_distances / predicted_distances)
        relative_error = torch.median(
            (scale * predicted_distances - dataset_distances).abs()
            / dataset_distances.clamp_min(min_baseline)
        )
        if torch.isfinite(scale) & (scale > 0):
            scales[batch_index] = scale.to(scales.dtype)
        if torch.isfinite(relative_error):
            errors[batch_index] = relative_error.to(errors.dtype)
        is_valid = (
            torch.isfinite(scale)
            & (scale > 0)
            & torch.isfinite(relative_error)
            & (relative_error <= max_relative_error)
        )
        if is_valid:
            valid_alignments[batch_index] = True

    return scales, errors, valid_alignments


def build_pose_aligned_depth_labels(
    *,
    vggt_depth: Tensor,
    vggt_depth_confidence: Tensor,
    vggt_pose_encoding: Tensor,
    dataset_c2w: Tensor,
    near: Tensor,
    far: Tensor,
    renderer_scale_invariant: bool,
    confidence_quantile: float = 0.2,
    min_baseline: float = 1e-4,
    min_pair_fraction: float = 0.2,
    max_pose_relative_error: float = 0.3,
) -> PoseAlignedDepthLabels:
    """Turn scale-ambiguous VGGT output into fixed-scale depth pseudo-labels."""

    if vggt_depth.ndim == 5 and vggt_depth.shape[-1] == 1:
        vggt_depth = vggt_depth.squeeze(-1)
    if vggt_depth.ndim != 4:
        raise ValueError(f"Expected VGGT depth [B, V, H, W], got {vggt_depth.shape}")
    if not 0.0 <= confidence_quantile < 1.0:
        raise ValueError("confidence_quantile must be in [0, 1)")

    predicted_w2c = vggt_pose_encoding_to_w2c(vggt_pose_encoding.float())
    predicted_c2w = torch.linalg.inv(predicted_w2c)
    predicted_centers = predicted_c2w[..., :3, 3]
    dataset_centers = dataset_c2w.float()[..., :3, 3]
    scale, pose_error, alignment_valid = estimate_pose_scale(
        predicted_centers,
        dataset_centers,
        min_baseline=min_baseline,
        min_pair_fraction=min_pair_fraction,
        max_relative_error=max_pose_relative_error,
    )

    metric_depth = vggt_depth.float() * scale[:, None, None, None]
    base_mask = (
        torch.isfinite(metric_depth)
        & (metric_depth > 0)
        & torch.isfinite(vggt_depth_confidence)
        & (metric_depth >= near[..., None, None])
        & (metric_depth <= far[..., None, None])
        & alignment_valid[:, None, None, None]
    )

    confidence_mask = torch.zeros_like(base_mask)
    for batch_index in range(metric_depth.shape[0]):
        valid_confidence = vggt_depth_confidence[batch_index][base_mask[batch_index]]
        if valid_confidence.numel() == 0:
            continue
        threshold = torch.quantile(valid_confidence.float(), confidence_quantile)
        confidence_mask[batch_index] = (
            torch.isfinite(vggt_depth_confidence[batch_index])
            & (vggt_depth_confidence[batch_index] >= threshold)
        )
    mask = base_mask & confidence_mask

    if renderer_scale_invariant:
        depth = metric_depth / near[..., None, None].clamp_min(1e-8)
    else:
        depth = metric_depth
    depth = torch.where(mask, depth, torch.zeros_like(depth))

    return PoseAlignedDepthLabels(
        depth=depth,
        mask=mask,
        scale=scale,
        pose_relative_error=pose_error,
        alignment_valid=alignment_valid,
    )


class StrictDepthLoss(nn.Module):
    """Fixed-scale depth loss; unlike SSI this never fits scale or shift."""

    def __init__(self, mode: str = "log_l1", epsilon: float = 1e-6) -> None:
        super().__init__()
        if mode not in {"log_l1", "relative_l1", "l1"}:
            raise ValueError(f"Unknown strict depth loss mode: {mode}")
        self.mode = mode
        self.epsilon = epsilon
        self.name = f"StrictDepthLoss({mode})"

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        interpolate: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if prediction.shape[-2:] != target.shape[-2:] and interpolate:
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        if prediction.shape != target.shape:
            raise ValueError(f"Depth shape mismatch: {prediction.shape} vs. {target.shape}")

        # Label validity determines supervision coverage. In particular, do not
        # mask prediction <= 0: an empty/zero renderer depth must be penalized,
        # not become an escape hatch from strict supervision.
        valid = torch.isfinite(prediction) & torch.isfinite(target) & (target > 0)
        if mask is not None:
            valid = valid & mask.bool()
        if not valid.any():
            zero = prediction.sum() * 0.0
            return zero, {
                "valid_fraction": valid.float().mean().detach(),
                "median_prediction_ratio": prediction.new_tensor(float("nan")),
            }

        prediction_valid = prediction[valid]
        target_valid = target[valid]
        if self.mode == "log_l1":
            per_pixel = (
                (prediction_valid + self.epsilon).clamp_min(self.epsilon).log()
                - (target_valid + self.epsilon).log()
            ).abs()
        elif self.mode == "relative_l1":
            per_pixel = (prediction_valid - target_valid).abs() / target_valid.clamp_min(self.epsilon)
        else:
            per_pixel = (prediction_valid - target_valid).abs()

        return per_pixel.mean(), {
            "valid_fraction": valid.float().mean().detach(),
            "median_prediction_ratio": torch.median(
                (prediction_valid.detach() / target_valid.detach().clamp_min(self.epsilon))
            ),
        }
