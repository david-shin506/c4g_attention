import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .intrinsics import maybe_apply_davis_intrinsics
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


DEFAULT_NVIDIA_SCENES = [
    "Balloon1",
    "Balloon2",
    "Jumping",
    "Playground",
    "Skating",
    "Truck",
    "Umbrella",
]


@dataclass
class DatasetNvidiaCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    scenes: list[str] | None = None
    num_context_views: int = 8
    gap: int = 0
    use_original_size: bool = True
    re10k_intrinsic: bool = False


@dataclass
class DatasetNvidiaCfgWrapper:
    nvidia: DatasetNvidiaCfg


class DatasetNvidia(Dataset):
    cfg: DatasetNvidiaCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetNvidiaCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Discover scenes
        self.data_list = []
        scenes_filter = cfg.scenes if cfg.scenes is not None else DEFAULT_NVIDIA_SCENES
        for root in cfg.roots:
            root = str(root)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if name not in scenes_filter:
                    continue
                scene_path = os.path.join(root, name)
                poses_path = os.path.join(scene_path, "poses_bounds.npy")
                if os.path.isdir(scene_path) and os.path.exists(poses_path):
                    self.data_list.append(scene_path)

        # Load metadata for all scenes
        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.load_data, sp): sp for sp in self.data_list
            }
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"NVIDIA: {self.stage}: loaded {len(self.scene_ids)} scenes")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, scene_path):
        """Parse LLFF poses_bounds.npy and prepare per-frame dicts."""
        scene_id = os.path.basename(scene_path)

        poses_bounds = np.load(
            os.path.join(scene_path, "poses_bounds.npy")
        )  # (N, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        bounds = poses_bounds[:, -2:]  # (N, 2) near/far

        N = poses.shape[0]

        # Extract c2w in LLFF convention and HWF
        c2w_llff = poses[:, :3, :4]  # (N, 3, 4)
        hwf = poses[:, :, 4]  # (N, 3) -> [H, W, focal]
        H_orig, W_orig, focal = hwf[0]
        H_orig, W_orig = int(H_orig), int(W_orig)

        # Convert LLFF [down, right, backwards] to OpenCV [right, down, forward]
        c2w_cv = np.zeros((N, 4, 4), dtype=np.float32)
        c2w_cv[:, :3, 0] = c2w_llff[:, :, 1]    # right
        c2w_cv[:, :3, 1] = c2w_llff[:, :, 0]    # down
        c2w_cv[:, :3, 2] = -c2w_llff[:, :, 2]   # forward (negate backward)
        c2w_cv[:, :3, 3] = c2w_llff[:, :, 3]    # translation
        c2w_cv[:, 3, 3] = 1.0

        # w2c for storage (will be inverted to c2w in getitem)
        w2c = np.linalg.inv(c2w_cv).astype(np.float32)

        # Intrinsics: normalized by image size
        # Image folder depends on use_original_size
        if self.cfg.use_original_size:
            img_dir = os.path.join(scene_path, "images_original_size")
            img_H, img_W = H_orig, W_orig
        else:
            img_dir = os.path.join(scene_path, "images")
            # Resized images are half resolution
            img_H, img_W = H_orig // 2, W_orig // 2

        # Scale focal length if using resized images
        if not self.cfg.use_original_size:
            focal_scaled = focal / 2.0
        else:
            focal_scaled = focal

        intr = np.eye(3, dtype=np.float32)
        intr[0, 0] = focal_scaled / img_W   # fx normalized
        intr[1, 1] = focal_scaled / img_H   # fy normalized
        intr[0, 2] = 0.5                    # cx at center (normalized)
        intr[1, 2] = 0.5                    # cy at center (normalized)

        # Build per-frame dicts
        frames = []
        for i in range(N):
            image_path = os.path.join(img_dir, f"{i:03d}.png")
            frames.append({
                "image_path": image_path,
                "extrinsics": w2c[i],    # w2c, inverted to c2w in getitem
                "intrinsics": intr,       # normalized (same for all frames)
                "near": float(bounds[i, 0]),
                "far": float(bounds[i, 1]),
            })
        return frames, scene_id

    def load_frames(self, frames):
        """Load PNG images and return [N, 3, H, W] tensor in [0, 1]."""
        images = []
        for frame in frames:
            img = Image.open(frame["image_path"]).convert("RGB")
            images.append(self.to_tensor(img))
        return torch.stack(images)

    # ------------------------------------------------------------------
    # getitem
    # ------------------------------------------------------------------

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene = self.scene_ids[index]
        example = self.scenes[scene]

        N = len(example)  # 12 views

        # Collect all poses
        extrinsics = np.array([f["extrinsics"] for f in example])
        intrinsics = np.array([f["intrinsics"] for f in example])

        # w2c -> c2w
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        # Optionally replace with hard-coded RE10K intrinsic
        if self.cfg.re10k_intrinsic:
            re10k_K = [[0.4836, 0.0000, 0.5000],
                        [0.0000, 0.8597, 0.5000],
                        [0.0000, 0.0000, 1.0000]]
            intrinsics = torch.tensor(re10k_K, dtype=torch.float32).unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)

        # View sampling: context and target split
        max_idx = N - 1

        if self.cfg.gap > 0:
            # Gap-based: context at stride=gap, target at midpoints
            context_indices = torch.arange(0, N, self.cfg.gap, dtype=torch.long)
            context_indices = context_indices[:self.cfg.num_context_views]

            target_indices = torch.tensor(
                [context_indices[i] + self.cfg.gap // 2
                 for i in range(len(context_indices) - 1)],
                dtype=torch.long,
            )
            target_indices = target_indices[target_indices < N]
        else:
            # Use num_context_views as context, rest as target
            num_ctx = min(self.cfg.num_context_views, N)
            context_indices = torch.linspace(0, max_idx, num_ctx).long()
            context_indices = torch.unique(context_indices)

            # Target: all views not in context
            ctx_set = set(context_indices.tolist())
            target_indices = torch.tensor(
                [i for i in range(N) if i not in ctx_set],
                dtype=torch.long,
            )
            if len(target_indices) == 0:
                target_indices = context_indices

        # is_ctx_time: for multi-view (no temporal), mark all as same time
        ctx_set = set(context_indices.tolist())
        is_ctx_time = torch.tensor(
            [i.item() in ctx_set for i in target_indices],
            dtype=torch.bool,
        )

        # Check FOV
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        # Load images
        input_frames = [example[i] for i in context_indices]
        target_frames = [example[i] for i in target_indices]

        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frames)

        # Check shapes
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise Exception("Bad example image shape")

        # Baseline normalization
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a = context_extrinsics[0, :3, 3]
            b_all = context_extrinsics[1:, :3, 3]
            diff = b_all - a
            norms = diff.norm(dim=1)
            scale = norms.max()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(f"Skipped {scene}: baseline out of range ({scale:.6f})")
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
            context_extrinsics = extrinsics[context_indices]
        else:
            scale = 1

        # Relative pose normalization
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(
                context_extrinsics[0:1], extrinsics
            )

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        target_extrinsics = extrinsics[target_indices]
        target_intrinsics = intrinsics[target_indices]

        overlap = torch.tensor([1.0])  # placeholder

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "camera": torch.zeros(len(context_indices), dtype=torch.long),
                "overlap": overlap,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
                "camera": torch.zeros(len(target_indices), dtype=torch.long),
                "is_ctx_time": is_ctx_time,
            },
            "scene": "nvidia_" + scene,
            "dataset_name": self.cfg.name,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14))
        return maybe_apply_davis_intrinsics(example, self.cfg.force_davis_intrinsics)

    def __getitem__(self, index: int) -> dict:
        num_context_views = self.view_sampler.num_context_views
        patchsize_h, patchsize_w = self.cfg.input_image_shape
        patchsize_h = patchsize_h // 14
        patchsize_w = patchsize_w // 14
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            index = np.random.randint(len(self))
            return self.__getitem__(index)

    def __len__(self) -> int:
        return len(self.scene_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
