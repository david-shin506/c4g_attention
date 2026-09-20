import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import cv2
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


@dataclass
class DatasetTUMCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    num_frames: int = 50
    scenes: list[str] | None = None
    test_target_mode: Literal["dense", "middle", "midpoints"] = "dense"


@dataclass
class DatasetTUMCfgWrapper:
    tum: DatasetTUMCfg


class DatasetTUM(Dataset):
    cfg: DatasetTUMCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetTUMCfg,
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
        for root in cfg.roots:
            root = str(root)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if cfg.scenes is not None and name not in cfg.scenes:
                    continue
                scene_path = os.path.join(root, name)
                if os.path.isdir(scene_path) and os.path.exists(
                    os.path.join(scene_path, "metadata.json")
                ):
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
        print(f"TUM: {self.stage}: loaded {len(self.scene_ids)} scenes")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, scene_path):
        """Parse metadata.json and prepare per-frame dicts."""
        scene_id = os.path.basename(scene_path)
        meta_path = os.path.join(scene_path, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Limit to first num_frames frames
        num_frames = min(self.cfg.num_frames, len(meta))
        meta = meta[:num_frames]

        video_path = os.path.join(scene_path, f"{scene_id}.mp4")
        H, W = self.cfg.original_image_shape

        data = []
        for i, entry in enumerate(meta):
            # Extrinsic: 3x4 → pad to 4x4
            extr_34 = np.array(entry["extrinsic"], dtype=np.float32)
            extr_44 = np.eye(4, dtype=np.float32)
            extr_44[:3, :] = extr_34

            # Intrinsic: 3x3 → normalize by image dims
            intr = np.array(entry["intrinsic"], dtype=np.float32)
            intr[0, :] /= W
            intr[1, :] /= H

            frame = {
                "video_path": video_path,
                "frame_index": i,
                "extrinsics": extr_44,
                "intrinsics": intr,
                "timestamp": entry["timestamp_ns"],
            }
            data.append(frame)
        return data, scene_id

    def load_frames(self, frames):
        """Decode video frames and return [N, 3, H, W] tensor in [0, 1]."""
        H, W = self.cfg.original_image_shape
        images = []
        for frame in frames:
            try:
                raw = iio.imread(
                    frame["video_path"], index=frame["frame_index"], plugin="pyav"
                )
            except ImportError:
                cap = cv2.VideoCapture(str(frame["video_path"]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame["frame_index"]))
                ok, raw_bgr = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError(
                        f"Failed to read frame {frame['frame_index']} from {frame['video_path']}"
                    )
                raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(raw).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H), Image.BILINEAR)
            images.append(self.to_tensor(img))
        return torch.stack(images)

    # ------------------------------------------------------------------
    # getitem
    # ------------------------------------------------------------------

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene = self.scene_ids[index]
        example = self.scenes[scene]

        # Collect poses
        extrinsics = np.array([f["extrinsics"] for f in example])
        intrinsics = np.array([f["intrinsics"] for f in example])

        # MG preprocessed version already stores c2w (no inversion needed)
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        # Replace with re10K intrinsic
        # re10k_K = [[0.8597, 0.0000, 0.5000],
        #             [0.0000, 0.8597, 0.5000],
        #             [0.0000, 0.0000, 1.0000]]
        # intrinsics = torch.tensor(re10k_K, dtype=torch.float32).unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)

        # Sample context / target views
        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            raise Exception("Not enough frames")

        # Test target sampling can be dense video, stride midpoints, or one middle frame.
        if self.stage == "test":
            ctx_set = set(context_indices.tolist())
            ctx_min = int(context_indices.min())
            ctx_max = int(context_indices.max())
            if self.cfg.test_target_mode == "midpoints":
                pass
            elif self.cfg.test_target_mode == "middle":
                middle = (ctx_min + ctx_max) // 2
                if middle in ctx_set:
                    candidates = [i for i in range(ctx_min, ctx_max + 1) if i not in ctx_set]
                    if candidates:
                        middle = min(candidates, key=lambda i: abs(i - middle))
                if middle not in ctx_set:
                    target_indices = torch.tensor([middle], dtype=torch.int64)
            elif self.cfg.test_target_mode == "dense":
                all_target = [i for i in range(ctx_min, ctx_max + 1) if i not in ctx_set]
                if all_target:
                    target_indices = torch.tensor(all_target, dtype=torch.int64)
            else:
                raise ValueError(f"Unknown TUM test_target_mode: {self.cfg.test_target_mode}")

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
                print(f"Skipped {scene} because of baseline out of range: {scale:.6f}")
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        # Relative pose normalization
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        # Masks: all-ones (TUM has no precomputed masks)
        num_total = len(example)
        H, W = self.cfg.original_image_shape
        masks = torch.ones(num_total, H, W, dtype=torch.float32)
        context_masks = masks[context_indices]
        target_masks = masks[target_indices]

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "camera": torch.zeros_like(context_indices),
                "mask": context_masks,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
                "camera": torch.zeros_like(target_indices),
                "mask": target_masks,
            },
            "scene": "tum_" + scene,
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
