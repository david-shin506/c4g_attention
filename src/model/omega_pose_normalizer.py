from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from .encoder.backbone.vggt_omega.models.vggt_omega import VGGTOmega
from .encoder.backbone.vggt_omega.utils.pose_enc import encoding_to_camera


@dataclass
class OmegaPoseNormalizerCfg:
    enabled: bool = False
    weights: str = ""
    scale_min: float = 1e-3
    scale_max: float = 1e3
    baseline_epsilon: float = 1e-6
    # Omega-unit camera baseline (max distance of any context camera to the
    # first one, as predicted by Omega) below which the window is treated as a
    # static camera. Omega's translation noise floor on truly static input is
    # ~1e-3..3e-3, so the scale ratio is meaningless below roughly 1e-2.
    # 0.0 disables the check (old behaviour).
    min_pred_baseline: float = 0.0
    # What to do with windows whose scale is invalid (static GT, tiny Omega
    # baseline, or ratio outside [scale_min, scale_max]):
    #   "keep":     keep the GT poses in raw dataset units with scale 1 (old behaviour)
    #   "identity": treat the window as a static camera: every context/target
    #               camera becomes the identity pose
    invalid_pose_mode: str = "keep"
    # If set, near/far are fixed to these values (in Omega units) instead of
    # being multiplied by the estimated scale. The Gaussians live in Omega
    # units regardless of the scale estimate, so scaling near/far only couples
    # the clipping planes to the (noisy) scale.
    fixed_near: Optional[float] = None
    fixed_far: Optional[float] = None
    # Where the scale comes from:
    #   "pose":  median ratio of Omega / GT camera-centre distances (needs camera motion)
    #   "depth": median ratio of Omega dense depth / GT depth (defined for static cameras)
    #   "auto":  depth when the batch carries valid GT depth, otherwise the pose ratio
    scale_source: str = "pose"
    depth_min_valid_pixels: int = 500
    # Omega's camera translations are ~0.8x its dense-depth units (kubric
    # large-baseline windows: pose/depth ratio 0.85; the C3G head, trained with
    # the pose ratio, renders depth at 0.75-0.8x the dense depth). Multiplying
    # the depth-anchored scale by this keeps the units continuous with the
    # pretrained head; within-scene consistency does not depend on it.
    depth_scale_multiplier: float = 0.8


@dataclass
class OmegaPoseNormalizerOutput:
    context_extrinsics: Tensor
    target_extrinsics: Tensor
    scale: Tensor
    scale_valid: Tensor
    gt_baseline: Tensor
    pred_baseline: Tensor
    scale_from_depth: Tensor


class FrozenOmegaPoseNormalizer(nn.Module):
    """Match C3G's Omega-derived camera canonicalization and scene scale."""

    def __init__(self, cfg: OmegaPoseNormalizerCfg) -> None:
        super().__init__()
        if not 0 < cfg.scale_min <= cfg.scale_max:
            raise ValueError(
                "Expected 0 < omega_pose_normalizer.scale_min <= scale_max, got "
                f"{cfg.scale_min} and {cfg.scale_max}"
            )
        if cfg.min_pred_baseline < 0:
            raise ValueError(
                "omega_pose_normalizer.min_pred_baseline must be >= 0, got "
                f"{cfg.min_pred_baseline}"
            )
        if cfg.invalid_pose_mode not in ("keep", "identity"):
            raise ValueError(
                "omega_pose_normalizer.invalid_pose_mode must be 'keep' or "
                f"'identity', got {cfg.invalid_pose_mode!r}"
            )
        if (cfg.fixed_near is None) != (cfg.fixed_far is None):
            raise ValueError(
                "omega_pose_normalizer.fixed_near and fixed_far must be set together"
            )
        if cfg.fixed_near is not None and not 0 < cfg.fixed_near < cfg.fixed_far:
            raise ValueError(
                "Expected 0 < omega_pose_normalizer.fixed_near < fixed_far, got "
                f"{cfg.fixed_near} and {cfg.fixed_far}"
            )
        if cfg.scale_source not in ("pose", "depth", "auto"):
            raise ValueError(
                "omega_pose_normalizer.scale_source must be 'pose', 'depth' or "
                f"'auto', got {cfg.scale_source!r}"
            )
        self.cfg = cfg
        self.model = VGGTOmega(enable_camera=True, enable_depth=self.uses_depth)
        self._load_weights(Path(cfg.weights))
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        return self

    @property
    def uses_fixed_bounds(self) -> bool:
        return self.cfg.fixed_near is not None

    @property
    def uses_depth(self) -> bool:
        return self.cfg.scale_source != "pose"

    @staticmethod
    def _read_checkpoint(path: Path):
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    def _load_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"VGGT-Omega teacher checkpoint not found: {path}")

        checkpoint = self._read_checkpoint(path)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid VGGT-Omega checkpoint format: {path}")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid VGGT-Omega checkpoint format: {path}")

        prefixes = ("aggregator.", "camera_head.") + (("dense_head.",) if self.uses_depth else ())
        selected = {}
        for key, value in state_dict.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                continue
            for prefix in ("module.", "model.", "encoder.", "backbone."):
                key = key.removeprefix(prefix)
            if key.startswith(prefixes):
                selected[key] = value

        for prefix in prefixes:
            if not any(key.startswith(prefix) for key in selected):
                raise ValueError(f"Checkpoint {path} has no weights with prefix {prefix!r}")

        missing, unexpected = self.model.load_state_dict(selected, strict=False)
        critical_missing = [key for key in missing if key.startswith(prefixes)]
        if critical_missing or unexpected:
            raise ValueError(
                "Invalid selected VGGT-Omega teacher weights: "
                f"missing={critical_missing[:8]}, unexpected={unexpected[:8]}"
            )
        del checkpoint, state_dict, selected

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        context_extrinsics: Tensor,
        target_extrinsics: Tensor,
        context_depth: Optional[Tensor] = None,
    ) -> OmegaPoseNormalizerOutput:
        want_depth = self.uses_depth and context_depth is not None
        predictions = self.model(images, predict_camera=True, predict_depth=want_depth)
        pred_depth = predictions["depth"].float()[..., 0] if want_depth else None  # [b, v, h, w]
        pred_w2c_34, _ = encoding_to_camera(
            predictions["pose_enc"].float(),
            image_size_hw=images.shape[-2:],
            build_intrinsics=False,
        )
        pred_w2c = _canonicalize_w2c(_pad_extrinsics(pred_w2c_34))
        pred_c2w = torch.linalg.inv(pred_w2c)

        context_c2w, target_c2w = _canonicalize_gt_c2w(
            context_extrinsics.float(),
            target_extrinsics.float(),
        )
        return apply_scale_policy(
            self.cfg,
            context_c2w,
            target_c2w,
            pred_c2w[..., :3, 3],
            pred_depth=pred_depth,
            gt_depth=context_depth.float() if want_depth else None,
        )


def apply_scale_policy(
    cfg: OmegaPoseNormalizerCfg,
    context_c2w: Tensor,
    target_c2w: Tensor,
    pred_centers: Tensor,
    pred_depth: Optional[Tensor] = None,
    gt_depth: Optional[Tensor] = None,
) -> OmegaPoseNormalizerOutput:
    """Estimate the GT->Omega scale and apply the invalid-window policy.

    Pure function of (canonicalized GT c2w, Omega-predicted camera centers) so
    the policy can be tested without running the Omega model.
    """
    gt_centers = context_c2w[..., :3, 3]
    gt_baseline = _max_baseline(gt_centers)
    pred_baseline = _max_baseline(pred_centers)

    scale, scale_valid = _estimate_scale(
        gt_centers,
        pred_centers,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        epsilon=cfg.baseline_epsilon,
    )
    if cfg.min_pred_baseline > 0:
        near_static = ~torch.isfinite(pred_baseline) | (
            pred_baseline < cfg.min_pred_baseline
        )
        scale_valid = scale_valid & ~near_static
        scale = torch.where(scale_valid, scale, torch.ones_like(scale))

    scale_from_depth = torch.zeros_like(scale_valid)
    if cfg.scale_source != "pose" and pred_depth is not None and gt_depth is not None:
        depth_scale, depth_valid = _estimate_depth_scale(
            pred_depth,
            gt_depth,
            scale_min=cfg.scale_min,
            scale_max=cfg.scale_max,
            min_valid_pixels=cfg.depth_min_valid_pixels,
        )
        depth_scale = depth_scale * cfg.depth_scale_multiplier
        if cfg.scale_source == "depth":
            scale, scale_valid = depth_scale, depth_valid
        else:  # auto: the depth ratio wins whenever it is available
            scale = torch.where(depth_valid, depth_scale, scale)
            scale_valid = scale_valid | depth_valid
        scale_from_depth = depth_valid
    elif cfg.scale_source == "depth":
        scale_valid = torch.zeros_like(scale_valid)
        scale = torch.ones_like(scale)

    if cfg.invalid_pose_mode == "identity":
        context_c2w = _replace_invalid_with_identity(context_c2w, scale_valid)
        target_c2w = _replace_invalid_with_identity(target_c2w, scale_valid)

    return OmegaPoseNormalizerOutput(
        context_extrinsics=_scale_c2w_translations(context_c2w, scale),
        target_extrinsics=_scale_c2w_translations(target_c2w, scale),
        scale=scale,
        scale_valid=scale_valid,
        gt_baseline=gt_baseline,
        pred_baseline=pred_baseline,
        scale_from_depth=scale_from_depth,
    )


def _estimate_depth_scale(
    pred_depth: Tensor,
    gt_depth: Tensor,
    scale_min: float,
    scale_max: float,
    min_valid_pixels: int,
) -> tuple[Tensor, Tensor]:
    """Per-batch median of Omega depth / GT depth over valid pixels (GT > 0)."""
    batch = pred_depth.shape[0]
    scales = pred_depth.new_ones(batch)
    valid_scenes = torch.zeros(batch, dtype=torch.bool, device=pred_depth.device)
    for batch_index in range(batch):
        gt = gt_depth[batch_index]
        pred = pred_depth[batch_index]
        valid = torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred) & (pred > 0)
        if valid.sum() < min_valid_pixels:
            continue
        raw_scale = (pred[valid] / gt[valid]).median()
        if not torch.isfinite(raw_scale) or not scale_min <= raw_scale.item() <= scale_max:
            continue
        scales[batch_index] = raw_scale
        valid_scenes[batch_index] = True
    return scales, valid_scenes


def _max_baseline(centers: Tensor) -> Tensor:
    """Largest distance of any camera to the first camera, per batch element."""
    distances = torch.linalg.norm(centers - centers[:, :1], dim=-1)
    return distances.amax(dim=1)


def _replace_invalid_with_identity(extrinsics: Tensor, valid: Tensor) -> Tensor:
    identity = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    identity = identity.expand_as(extrinsics)
    return torch.where(valid[:, None, None, None], extrinsics, identity)


def _pad_extrinsics(extrinsics: Tensor) -> Tensor:
    bottom = extrinsics.new_zeros(*extrinsics.shape[:-2], 1, 4)
    bottom[..., 0, 3] = 1
    return torch.cat([extrinsics, bottom], dim=-2)


def _canonicalize_w2c(extrinsics: Tensor) -> Tensor:
    reference_c2w = torch.linalg.inv(extrinsics[:, 0])
    return torch.einsum("bvij,bjk->bvik", extrinsics, reference_c2w)


def _canonicalize_gt_c2w(
    context_extrinsics: Tensor,
    target_extrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    world_to_reference = torch.linalg.inv(context_extrinsics[:, 0])
    context = torch.einsum("bij,bvjk->bvik", world_to_reference, context_extrinsics)
    target = torch.einsum("bij,bvjk->bvik", world_to_reference, target_extrinsics)
    return context, target


def _estimate_scale(
    gt_centers: Tensor,
    pred_centers: Tensor,
    scale_min: float,
    scale_max: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    gt_distances = torch.linalg.norm(gt_centers - gt_centers[:, :1], dim=-1)
    pred_distances = torch.linalg.norm(pred_centers - pred_centers[:, :1], dim=-1)
    scales = gt_centers.new_ones(gt_centers.shape[0])
    valid_scenes = torch.zeros(
        gt_centers.shape[0], dtype=torch.bool, device=gt_centers.device
    )

    for batch_index in range(gt_centers.shape[0]):
        valid = (
            torch.isfinite(gt_distances[batch_index])
            & torch.isfinite(pred_distances[batch_index])
            & (gt_distances[batch_index] > epsilon)
            & (pred_distances[batch_index] > epsilon)
        )
        ratios = pred_distances[batch_index, valid] / gt_distances[batch_index, valid]
        if ratios.numel() == 0:
            continue
        raw_scale = ratios.median()
        if not torch.isfinite(raw_scale):
            continue
        if not scale_min <= raw_scale.item() <= scale_max:
            continue
        scales[batch_index] = raw_scale
        valid_scenes[batch_index] = True
    return scales, valid_scenes


def _scale_c2w_translations(extrinsics: Tensor, scale: Tensor) -> Tensor:
    extrinsics = extrinsics.clone()
    extrinsics[..., :3, 3] *= scale[:, None, None]
    return extrinsics
