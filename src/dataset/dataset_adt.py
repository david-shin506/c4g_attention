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


DEFAULT_ADT_SCENES = [
    "Apartment_release_multiuser_cook_seq141_M1292",
    "Apartment_release_multiskeleton_party_seq114_M1292",
    "Apartment_release_meal_skeleton_seq135_M1292",
    "Apartment_release_work_skeleton_seq137_M1292",
]


@dataclass
class DatasetADTCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    num_frames: int = 300
    scenes: list[str] | None = None
    num_context_frames: int = 32
    gap: int = 0
    max_temporal: int = 300
    num_target_frames: int | None = None
    rectified_subdir: str = "synthetic_video/camera-rgb-rectified-600-h1000"


@dataclass
class DatasetADTCfgWrapper:
    adt: DatasetADTCfg


class DatasetADT(Dataset):
    cfg: DatasetADTCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetADTCfg,
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
        scenes_filter = cfg.scenes if cfg.scenes is not None else DEFAULT_ADT_SCENES
        for root in cfg.roots:
            root = str(root)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if name not in scenes_filter:
                    continue
                scene_path = os.path.join(root, name)
                rect_dir = os.path.join(scene_path, cfg.rectified_subdir)
                transforms_path = os.path.join(rect_dir, "transforms.json")
                if os.path.isdir(rect_dir) and os.path.exists(transforms_path):
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
        print(f"ADT: {self.stage}: loaded {len(self.scene_ids)} scenes")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, scene_path):
        """Parse rectified transforms.json and prepare per-frame dicts."""
        scene_id = os.path.basename(scene_path)
        rect_dir = os.path.join(scene_path, self.cfg.rectified_subdir)
        transforms_path = os.path.join(rect_dir, "transforms.json")

        with open(transforms_path, "r") as f:
            data = json.load(f)

        frames_raw = data["frames"]
        num_frames = min(self.cfg.num_frames, len(frames_raw))
        frames_raw = frames_raw[:num_frames]

        frames = []
        for fr in frames_raw:
            # Intrinsics: normalized by image size
            w, h = fr["w"], fr["h"]
            intr = np.eye(3, dtype=np.float32)
            intr[0, 0] = fr["fx"] / w
            intr[1, 1] = fr["fy"] / h
            intr[0, 2] = fr["cx"] / w
            intr[1, 2] = fr["cy"] / h

            # Extrinsics: transform_matrix is c2w -> invert to w2c
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            w2c = np.linalg.inv(c2w).astype(np.float32)

            # Image path
            image_path = os.path.join(rect_dir, fr["image_path"])

            frames.append({
                "image_path": image_path,
                "extrinsics": w2c,   # w2c, will be inverted to c2w in getitem
                "intrinsics": intr,  # normalized
                "timestamp": fr["timestamp"],
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

        T = len(example)

        # Collect all poses
        extrinsics = np.array([f["extrinsics"] for f in example])
        intrinsics = np.array([f["intrinsics"] for f in example])

        # w2c -> c2w
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        # Temporal sampling for context
        max_t = min(T - 1, self.cfg.max_temporal - 1)

        if self.cfg.gap > 0:
            # Gap-based sequential: context at stride=gap, target at midpoints
            context_indices = torch.arange(0, max_t + 1, self.cfg.gap, dtype=torch.long)
            context_indices = context_indices[:self.cfg.num_context_frames]

            if self.cfg.num_target_frames is not None:
                target_time_indices = torch.arange(
                    self.cfg.gap // 2, max_t + 1, self.cfg.gap, dtype=torch.long
                )
                target_time_indices = target_time_indices[:self.cfg.num_target_frames]
                target_time_indices = target_time_indices[target_time_indices < T]
            else:
                target_time_indices = torch.tensor(
                    [context_indices[i] + self.cfg.gap // 2
                     for i in range(len(context_indices) - 1)],
                    dtype=torch.long,
                )
                target_time_indices = target_time_indices[target_time_indices < T]
        else:
            # Linspace-based uniform sampling
            num_ctx = min(self.cfg.num_context_frames, max_t + 1)
            context_indices = torch.linspace(0, max_t, num_ctx).long()
            context_indices = torch.unique(context_indices)

            if self.cfg.num_target_frames is not None:
                num_tgt_t = min(self.cfg.num_target_frames, max_t + 1)
                target_time_indices = torch.linspace(0, max_t, num_tgt_t).long()
                target_time_indices = torch.unique(target_time_indices)
            else:
                target_time_indices = context_indices

        # Boolean mask: which target frames share a timestamp with context
        ctx_set = set(context_indices.tolist())
        is_ctx_time = torch.tensor(
            [i.item() in ctx_set for i in target_time_indices],
            dtype=torch.bool,
        )

        # Check FOV
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        # Load images
        input_frames = [example[i] for i in context_indices]
        target_frames = [example[i] for i in target_time_indices]

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

        target_extrinsics = extrinsics[target_time_indices]
        target_intrinsics = intrinsics[target_time_indices]

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
                "near": self.get_bound("near", len(target_time_indices)) / scale,
                "far": self.get_bound("far", len(target_time_indices)) / scale,
                "index": target_time_indices,
                "camera": torch.zeros(len(target_time_indices), dtype=torch.long),
                "is_ctx_time": is_ctx_time,
            },
            "scene": "adt_" + scene,
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
