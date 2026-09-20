#!/usr/bin/env python3
"""Camera-specific pose alignment shared across all target timestamps."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from tqdm import tqdm


def _gather_camera_parameter(parameter: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, C, ...]`` camera parameters using ``[B, V]`` indices."""
    suffix = parameter.shape[2:]
    gather_indices = indices.reshape(*indices.shape, *((1,) * len(suffix)))
    gather_indices = gather_indices.expand(*indices.shape, *suffix)
    return parameter.gather(1, gather_indices)


def _camera_layout(batch, masks_per_timestamp):
    """Return sorted physical camera IDs and per-timestamp camera indices."""
    target_cameras = batch["target"]["camera"]
    if target_cameras.ndim != 2:
        raise ValueError(f"Expected target camera tensor [B, V], got {target_cameras.shape}")

    camera_ids = torch.unique(target_cameras[0], sorted=True)
    if camera_ids.numel() == 0:
        raise ValueError("No target cameras available for static pose alignment")
    for batch_index in range(1, target_cameras.shape[0]):
        other_ids = torch.unique(target_cameras[batch_index], sorted=True)
        if not torch.equal(camera_ids, other_ids):
            raise ValueError("All batch elements must contain the same physical cameras")

    timestamp_camera_indices = []
    for mask in masks_per_timestamp:
        timestamp_camera_ids = target_cameras[:, mask]
        matches = timestamp_camera_ids.unsqueeze(-1) == camera_ids.view(1, 1, -1)
        if not matches.any(dim=-1).all() or not (matches.sum(dim=-1) == 1).all():
            raise ValueError("Could not uniquely map a target view to a physical camera")
        timestamp_camera_indices.append(matches.to(torch.long).argmax(dim=-1))
    return camera_ids, timestamp_camera_indices


def align_static_camera_poses(
    self,
    batch,
    gaussians_per_timestamp,
    masks_per_timestamp,
    h,
    w,
    initial_extrinsics=None,
):
    """Jointly optimize one extrinsic per physical camera across all times.

    This is a drop-in replacement for ModelWrapper's current shared-target-pose
    routine. Unlike the original routine, it does not collapse a multi-camera
    target set onto one extrinsic.
    """
    from src.misc.cam_utils import update_pose

    self.encoder.eval()
    for parameter in self.encoder.parameters():
        parameter.requires_grad = False

    camera_ids, timestamp_camera_indices = _camera_layout(
        batch, masks_per_timestamp
    )
    target_cameras = batch["target"]["camera"]
    first_positions = torch.stack([
        torch.nonzero(target_cameras[0] == camera_id, as_tuple=False)[0, 0]
        for camera_id in camera_ids
    ])
    shared_extrinsics = batch["target"]["extrinsics"][:, first_positions].clone()
    if initial_extrinsics is not None:
        if initial_extrinsics.shape != shared_extrinsics.shape:
            raise ValueError(
                "Initial target poses must provide one extrinsic per physical "
                f"camera: expected {shared_extrinsics.shape}, got "
                f"{initial_extrinsics.shape}"
            )
        shared_extrinsics = initial_extrinsics.clone()

    batch_size, num_cameras = shared_extrinsics.shape[:2]
    num_timestamps = len(masks_per_timestamp)
    print(
        "Static camera pose align: "
        f"{num_cameras} physical camera poses shared across "
        f"{num_timestamps} timestamps"
    )

    with torch.set_grad_enabled(True):
        camera_rotation_delta = nn.Parameter(
            torch.zeros(batch_size, num_cameras, 3, device=self.device)
        )
        camera_translation_delta = nn.Parameter(
            torch.zeros(batch_size, num_cameras, 3, device=self.device)
        )
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_rotation_delta], "lr": self.test_cfg.rot_opt_lr},
            {"params": [camera_translation_delta], "lr": self.test_cfg.trans_opt_lr},
        ])
        previous_loss = None
        patience_counter = 0
        patience_limit = 10

        with self.benchmarker.time("optimize"):
            for iteration in tqdm(
                range(self.test_cfg.pose_align_steps * 10),
                desc="Static camera pose align",
            ):
                pose_optimizer.zero_grad()
                total_loss_value = torch.zeros((), device=self.device)
                for gaussians, mask, camera_indices in zip(
                    gaussians_per_timestamp,
                    masks_per_timestamp,
                    timestamp_camera_indices,
                ):
                    timestamp_extrinsics = _gather_camera_parameter(
                        shared_extrinsics, camera_indices
                    )
                    timestamp_rotation_delta = _gather_camera_parameter(
                        camera_rotation_delta, camera_indices
                    )
                    timestamp_translation_delta = _gather_camera_parameter(
                        camera_translation_delta, camera_indices
                    )
                    output = self.decoder.forward(
                        gaussians,
                        timestamp_extrinsics,
                        batch["target"]["intrinsics"][:, mask],
                        batch["target"]["near"][:, mask],
                        batch["target"]["far"][:, mask],
                        (h, w),
                        cam_rot_delta=timestamp_rotation_delta,
                        cam_trans_delta=timestamp_translation_delta,
                    )
                    target_image = batch["target"]["image"][:, mask]
                    timestamp_loss = 0
                    for loss_function in self.losses:
                        timestamp_loss = timestamp_loss + loss_function.forward(
                            output,
                            batch,
                            gaussians,
                            self.global_step,
                            target_image=target_image,
                        )
                    total_loss_value = total_loss_value + timestamp_loss.detach()
                    (timestamp_loss / num_timestamps).backward()

                current_loss = (total_loss_value / num_timestamps).item()
                with torch.no_grad():
                    pose_optimizer.step()
                    updated_extrinsics = update_pose(
                        cam_rot_delta=rearrange(
                            camera_rotation_delta, "b c i -> (b c) i"
                        ),
                        cam_trans_delta=rearrange(
                            camera_translation_delta, "b c i -> (b c) i"
                        ),
                        extrinsics=rearrange(
                            shared_extrinsics, "b c i j -> (b c) i j"
                        ),
                    )
                    shared_extrinsics = rearrange(
                        updated_extrinsics,
                        "(b c) i j -> b c i j",
                        b=batch_size,
                        c=num_cameras,
                    )
                    camera_rotation_delta.zero_()
                    camera_translation_delta.zero_()

                if previous_loss is not None:
                    if abs(current_loss - previous_loss) < 0.00001:
                        patience_counter += 1
                        if patience_counter >= patience_limit and iteration >= 100:
                            break
                    else:
                        patience_counter = 0
                previous_loss = current_loss

    outputs = []
    with torch.no_grad():
        for gaussians, mask, camera_indices in zip(
            gaussians_per_timestamp,
            masks_per_timestamp,
            timestamp_camera_indices,
        ):
            outputs.append(self.decoder.forward(
                gaussians,
                _gather_camera_parameter(shared_extrinsics, camera_indices),
                batch["target"]["intrinsics"][:, mask],
                batch["target"]["near"][:, mask],
                batch["target"]["far"][:, mask],
                (h, w),
            ))
    del pose_optimizer
    return outputs
