import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest import mock

import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .intrinsics import davis_intrinsics_like, maybe_apply_davis_intrinsics
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


@dataclass
class DatasetEgoExo4DCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    view_drop_prob: float = 0.3
    min_views_per_timestamp: int = 2
    frame_stride: int = 10


@dataclass
class DatasetEgoExo4DCfgWrapper:
    egoexo4d: DatasetEgoExo4DCfg


class DatasetEgoExo4D(Dataset):
    """EgoExo4D dataset loader for exo (static) camera views.

    Sampling strategy:
      - view_sampler picks context & target *timestamps*
      - Context: all exo cameras per timestamp, with random view dropping
        (some timestamps lose a few cameras, at least min_views_per_timestamp kept)
      - Target: all exo cameras at novel timestamps between contexts
    """

    cfg: DatasetEgoExo4DCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetEgoExo4DCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.takes: list[dict] = []

        for root in self.cfg.roots:
            root = Path(root)
            self._load_root(root)

        print(f"egoexo4d: {self.stage}: {len(self.takes)} takes")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_to_split(stage: Stage) -> str:
        if stage in ("train", "val", "test"):
            return stage
        return "train"

    def _load_root(self, root: Path) -> None:
        split = self._stage_to_split(self.stage)

        # Load splits  (take_uid -> "train"/"val"/"test")
        with open(root / "annotations" / "splits.json") as f:
            uid_to_split: dict[str, str] = json.load(f)["take_uid_to_split"]

        # Load takes metadata  (list of dicts)
        with open(root / "takes.json") as f:
            takes_list: list[dict] = json.load(f)
        uid_to_take = {t["take_uid"]: t for t in takes_list}

        cam_pose_dir = root / "annotations" / "ego_pose" / split / "camera_pose"

        for take_uid, take_split in uid_to_split.items():
            if take_split != split:
                continue
            if take_uid not in uid_to_take:
                continue

            take_meta = uid_to_take[take_uid]
            take_name = take_meta["take_name"]

            cam_pose_path = cam_pose_dir / f"{take_uid}.json"
            if not cam_pose_path.exists():
                continue

            video_dir = root / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448"
            if not video_dir.exists():
                continue

            with open(cam_pose_path) as f:
                cam_data: dict = json.load(f)

            exo_cams, extrinsics, intrinsics = self._parse_cameras(cam_data, video_dir, take_name)
            if len(exo_cams) < 2:
                continue

            # Get frame count from aria per-frame extrinsics (fast, no I/O)
            num_frames = self._get_frame_count_from_pose(cam_data)
            if num_frames is None:
                # Fallback: duration from takes.json
                dur = take_meta.get("duration_sec", 0)
                num_frames = int(dur * 30) if dur > 0 else 0
            if num_frames < 2:
                continue

            self.takes.append({
                "take_uid": take_uid,
                "take_name": take_name,
                "video_dir": video_dir,
                "exo_cams": exo_cams,
                "extrinsics": extrinsics,  # dict[cam_name -> np.ndarray (4,4) c2w]
                "intrinsics": intrinsics,  # dict[cam_name -> np.ndarray (3,3) normalised]
                "num_frames": num_frames,
            })

    @staticmethod
    def _parse_cameras(
        cam_data: dict,
        video_dir: Path,
        take_name: str = "",
    ) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return (exo_cam_names, extrinsics_c2w, intrinsics_normalised)."""
        # Skip cam05+ for dance takes (top-down overhead cameras)
        is_dance = "dance" in take_name.lower()

        exo_cams: list[str] = []
        extrinsics: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}

        for key in sorted(cam_data.keys()):
            if key in ("metadata",) or key.startswith("aria"):
                continue
            # Skip cam05+ for dance takes
            if is_dance and key >= "cam05":
                continue
            # Must have a matching video file
            if not (video_dir / f"{key}.mp4").exists():
                continue

            entry = cam_data[key]
            ext_34 = np.array(entry["camera_extrinsics"], dtype=np.float32)
            K = np.array(entry["camera_intrinsics"], dtype=np.float32)

            # 3×4 W2C → 4×4 C2W
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = ext_34
            c2w = np.linalg.inv(w2c).astype(np.float32)

            # Normalise intrinsics to [0, 1] using principal point to infer native resolution
            native_w = K[0, 2] * 2  # cx = W/2
            native_h = K[1, 2] * 2  # cy = H/2
            K[0, :] /= native_w
            K[1, :] /= native_h

            exo_cams.append(key)
            extrinsics[key] = c2w
            intrinsics[key] = K

        return exo_cams, extrinsics, intrinsics

    @staticmethod
    def _get_frame_count_from_pose(cam_data: dict) -> int | None:
        """Get frame count from the aria camera's per-frame extrinsics."""
        for key in cam_data:
            if not key.startswith("aria"):
                continue
            ext = cam_data[key].get("camera_extrinsics", {})
            if isinstance(ext, dict) and len(ext) > 0:
                return len(ext)
        return None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.takes)

    def __getitem__(self, idx: int) -> dict:
        max_retries = 50
        for attempt in range(max_retries):
            try:
                return self._getitem_impl(idx)
            except Exception:
                idx = np.random.randint(len(self))
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _getitem_impl(self, idx: int) -> dict:
        take = self.takes[idx]
        video_dir: Path = take["video_dir"]
        exo_cams: list[str] = take["exo_cams"]
        num_cams = len(exo_cams)
        total_frames: int = take["num_frames"]

        stride = self.cfg.frame_stride
        num_timestamps = total_frames // stride
        if num_timestamps < self.view_sampler.num_context_views + 1:
            raise Exception(
                f"Not enough timestamps ({num_timestamps}) for "
                f"{self.view_sampler.num_context_views} context views"
            )

        # view_sampler expects (num_views, 4, 4) extrinsics – only num_views matters
        ref_ext = torch.tensor(take["extrinsics"][exo_cams[0]], dtype=torch.float32)
        dummy_ext = ref_ext.unsqueeze(0).expand(num_timestamps, -1, -1)
        dummy_int = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(num_timestamps, -1, -1)

        ctx_ts, tgt_ts, overlap = self.view_sampler.sample("", dummy_ext, dummy_int)

        ctx_frame_indices = ctx_ts * stride
        tgt_frame_indices = tgt_ts * stride

        # ---- Build context (with view dropping) ----
        ctx_cams_ids: list[int] = []
        ctx_frames: list[tuple[str, int]] = []  # (cam_name, frame_idx)
        ctx_ext_list: list[np.ndarray] = []
        ctx_int_list: list[np.ndarray] = []
        ctx_ts_list: list[int] = []

        # Dropped context views will be added to target
        dropped_cams_ids: list[int] = []
        dropped_frames: list[tuple[str, int]] = []
        dropped_ext_list: list[np.ndarray] = []
        dropped_int_list: list[np.ndarray] = []
        dropped_ts_list: list[int] = []

        for i, ts in enumerate(ctx_ts):
            frame_idx = ctx_frame_indices[i].item()
            ts_val = ts.item()

            if i == 0:
                # First context timestamp always keeps all views
                keep = list(range(num_cams))
            else:
                keep = [c for c in range(num_cams) if torch.rand(1).item() > self.cfg.view_drop_prob]
                # Guarantee minimum
                if len(keep) < min(self.cfg.min_views_per_timestamp, num_cams):
                    perm = torch.randperm(num_cams).tolist()
                    needed = min(self.cfg.min_views_per_timestamp, num_cams)
                    keep = sorted(set(keep + perm[:needed]))

            dropped = [c for c in range(num_cams) if c not in keep]

            for c in keep:
                cam = exo_cams[c]
                ctx_cams_ids.append(c)
                ctx_frames.append((cam, frame_idx))
                ctx_ext_list.append(take["extrinsics"][cam])
                ctx_int_list.append(take["intrinsics"][cam])
                ctx_ts_list.append(ts_val)

            for c in dropped:
                cam = exo_cams[c]
                dropped_cams_ids.append(c)
                dropped_frames.append((cam, frame_idx))
                dropped_ext_list.append(take["extrinsics"][cam])
                dropped_int_list.append(take["intrinsics"][cam])
                dropped_ts_list.append(ts_val)

        # ---- Build target (novel timestamps + dropped context views) ----
        tgt_cams_ids: list[int] = []
        tgt_frames: list[tuple[str, int]] = []
        tgt_ext_list: list[np.ndarray] = []
        tgt_int_list: list[np.ndarray] = []
        tgt_ts_list: list[int] = []

        # Novel timestamps: all exo cams
        for i, ts in enumerate(tgt_ts):
            frame_idx = tgt_frame_indices[i].item()
            ts_val = ts.item()
            for c, cam in enumerate(exo_cams):
                tgt_cams_ids.append(c)
                tgt_frames.append((cam, frame_idx))
                tgt_ext_list.append(take["extrinsics"][cam])
                tgt_int_list.append(take["intrinsics"][cam])
                tgt_ts_list.append(ts_val)

        # Dropped context views become additional targets
        tgt_cams_ids.extend(dropped_cams_ids)
        tgt_frames.extend(dropped_frames)
        tgt_ext_list.extend(dropped_ext_list)
        tgt_int_list.extend(dropped_int_list)
        tgt_ts_list.extend(dropped_ts_list)

        # ---- Read frames ----
        ctx_images = self._load_images(video_dir, ctx_frames)
        tgt_images = self._load_images(video_dir, tgt_frames)

        if self.cfg.skip_bad_shape:
            expected = (3, *self.cfg.original_image_shape)
            if ctx_images.shape[1:] != expected or tgt_images.shape[1:] != expected:
                raise Exception("Bad image shape")

        # ---- Tensors ----
        ctx_ext = torch.tensor(np.stack(ctx_ext_list), dtype=torch.float32)
        tgt_ext = torch.tensor(np.stack(tgt_ext_list), dtype=torch.float32)

        # Override intrinsics with RE10k values (model is overfitted to re10k)
        re10k_int = torch.tensor(
            [[0.4836, 0.0, 0.5], [0.0, 0.8597, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        n_ctx_total = len(ctx_cams_ids)
        n_tgt_total = len(tgt_cams_ids)
        ctx_int = re10k_int.unsqueeze(0).expand(n_ctx_total, -1, -1).clone()
        tgt_int = re10k_int.unsqueeze(0).expand(n_tgt_total, -1, -1).clone()
        if self.cfg.force_davis_intrinsics:
            ctx_int = davis_intrinsics_like(ctx_int)
            tgt_int = davis_intrinsics_like(tgt_int)
        ctx_index = torch.tensor(ctx_ts_list, dtype=torch.int64)
        tgt_index = torch.tensor(tgt_ts_list, dtype=torch.int64)
        ctx_camera = torch.tensor(ctx_cams_ids, dtype=torch.int64)
        tgt_camera = torch.tensor(tgt_cams_ids, dtype=torch.int64)

        # ---- Baseline normalization ----
        scale = 1.0
        if self.cfg.make_baseline_1:
            positions = ctx_ext[:, :3, 3]
            diffs = positions[1:] - positions[0:1]
            norms = diffs.norm(dim=1)
            if norms.numel() > 0:
                scale = norms.max().item()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                raise Exception(f"Baseline {scale:.3f} out of range")
            if scale > 0:
                ctx_ext[:, :3, 3] /= scale
                tgt_ext[:, :3, 3] /= scale

        # ---- Relative pose normalization ----
        if self.cfg.relative_pose:
            ref = ctx_ext[0:1]
            ctx_ext = camera_normalization(ref, ctx_ext)
            tgt_ext = camera_normalization(ref, tgt_ext)

        n_ctx = len(ctx_cams_ids)
        n_tgt = len(tgt_cams_ids)
        scene = f"egoexo4d/{take['take_name']}"

        example = {
            "context": {
                "extrinsics": ctx_ext,
                "intrinsics": ctx_int,
                "image": ctx_images,
                "near": self.get_bound("near", n_ctx) / scale,
                "far": self.get_bound("far", n_ctx) / scale,
                "index": ctx_index,
                "camera": ctx_camera,
                "overlap": overlap,
            },
            "target": {
                "extrinsics": tgt_ext,
                "intrinsics": tgt_int,
                "image": tgt_images,
                "near": self.get_bound("near", n_tgt) / scale,
                "far": self.get_bound("far", n_tgt) / scale,
                "index": tgt_index,
                "camera": tgt_camera,
            },
            "scene": scene,
            "dataset_name": self.cfg.name,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
        return maybe_apply_davis_intrinsics(example, self.cfg.force_davis_intrinsics)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_images(
        self,
        video_dir: Path,
        frames: list[tuple[str, int]],
    ) -> Float[Tensor, "batch 3 height width"]:
        images_np = [
            self._read_frame(str(video_dir / f"{cam}.mp4"), idx)
            for cam, idx in frames
        ]
        return self._convert_numpy_images(images_np)

    def _read_frame(self, video_path: str, index: int) -> np.ndarray:
        if iio is None:
            raise ImportError("imageio.v3 is required to read video frames.")
        mock.patch("imageio_ffmpeg._io.subprocess.Popen.kill").start()
        frame = iio.imread(video_path, index=index)  # H W 3 uint8
        H, W = self.cfg.original_image_shape
        img = Image.fromarray(frame)
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        return np.asarray(img)

    def _convert_numpy_images(
        self,
        images: list[np.ndarray],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for im in images:
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[:, :, :3]
            torch_images.append(self.to_tensor(Image.fromarray(im)))
        return torch.stack(torch_images)

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
            return "train"
        return "train" if self.stage in ("train", "val") else self.stage
