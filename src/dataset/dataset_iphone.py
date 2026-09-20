import json
import os
import struct
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .intrinsics import maybe_apply_davis_intrinsics
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


# ---- Colmap binary reading utilities (from flow3d/data/colmap.py) ----

@dataclass(frozen=True)
class ColmapCameraModel:
    model_id: int
    model_name: str
    num_params: int

@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

@dataclass(frozen=True)
class ColmapBaseImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

class ColmapImage(ColmapBaseImage):
    pass

COLMAP_CAMERA_MODELS = {
    ColmapCameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    ColmapCameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    ColmapCameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    ColmapCameraModel(model_id=3, model_name="RADIAL", num_params=5),
    ColmapCameraModel(model_id=4, model_name="OPENCV", num_params=8),
    ColmapCameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    ColmapCameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    ColmapCameraModel(model_id=7, model_name="FOV", num_params=5),
    ColmapCameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    ColmapCameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    ColmapCameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
COLMAP_CAMERA_MODEL_IDS = {cm.model_id: cm for cm in COLMAP_CAMERA_MODELS}

def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file: Union[str, Path]) -> Dict[int, ColmapCamera]:
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = COLMAP_CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = COLMAP_CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file: Union[str, Path]) -> Dict[int, ColmapImage]:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2D = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = _read_next_bytes(
                fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            ) if num_points2D > 0 else np.zeros((0, 2))
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3]))) if num_points2D > 0 else np.zeros(0, dtype=int)
            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=image_name, xys=xys, point3D_ids=point3D_ids,
            )
    return images

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2],
    ])

# ---- End colmap utilities ----


@dataclass
class DatasetIphoneCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    context_cameras: list[int] | None = None
    target_cameras: list[int] | None = None
    num_target_frames: int | None = None
    max_temporal: int = 51


@dataclass
class DatasetIphoneCfgWrapper:
    iphone: DatasetIphoneCfg

class DatasetIphone(Dataset):
    cfg: DatasetIphoneCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetIphoneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        
        if stage != "test":
            raise NotImplementedError("Only test stage is implemented for DatasetIphone")

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        
        # load data
        self.data_root = cfg.roots[0]
        self.data_list = ['apple', 'block', 'spin', 'paper-windmill', 'teddy']
        # self.data_list = ['apple', 'block', 'space-out', 'spin', 'paper-windmill', 'teddy']
        self.data_list = [f"{d}/{d}" for d in self.data_list]

        self.scene_ids = {}
        self.scenes = {}
        index = 0
        
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_jsons, os.path.join(self.data_root, scene_path)) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        print(f"IPhone: {self.stage}: loaded {len(self.scene_ids)} scenes")
        
    def convert_intrinsics(self, meta_data):
        store_h, store_w = meta_data["image_size"][1], meta_data["image_size"][0]
        fx, fy, cx, cy = (
            meta_data["focal_length"],
            meta_data["focal_length"],
            meta_data["principal_point"][0],
            meta_data["principal_point"][1],
        )
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = float(fx) / float(store_w)
        intrinsics[1, 1] = float(fy) / float(store_h)
        intrinsics[0, 2] = float(cx) / float(store_w)
        intrinsics[1, 2] = float(cy) / float(store_h)
        return intrinsics

    def convert_extrinsics(self, meta_data):
        r = np.array(meta_data["orientation"])
        t = np.array(meta_data["position"])
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = r.T
        extr[:3, 3] = -r.T @ t
        return extr

    def load_cam_data(self, scene_path, frame_name):
        json_path = os.path.join(scene_path, 'camera', frame_name + ".json")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def load_jsons(self, scene_path):
        json_path = os.path.join(scene_path, "metadata.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        # Load colmap poses
        colmap_sparse_dir = os.path.join(scene_path, "flow3d_preprocessed/colmap/sparse")
        colmap_cameras = read_cameras_binary(os.path.join(colmap_sparse_dir, "cameras.bin"))
        colmap_images = read_images_binary(os.path.join(colmap_sparse_dir, "images.bin"))
        colmap_img_lookup = {v.name: v for v in colmap_images.values()}
        scale = np.load(os.path.join(scene_path, "flow3d_preprocessed/colmap/scale.npy")).item()

        scene_frames = {}
        scene_id = scene_path.split("/")[-1].split(".")[0]
        for frame_name, frame_data in data.items():
            assert frame_data["appearance_id"] == frame_data["warp_id"], f"Something is wrong with the data! {scene_path}, {frame_name}, {frame_data['appearance_id']}, {frame_data['warp_id']}"
            frame_tmp = {}
            if int(frame_data["camera_id"]) not in scene_frames:
                scene_frames[int(frame_data["camera_id"])] = {}

            img_name = frame_name + ".png"
            if img_name not in colmap_img_lookup:
                continue  # skip frames not registered in colmap

            colmap_img = colmap_img_lookup[img_name]
            cam = colmap_cameras[colmap_img.camera_id]

            # Intrinsics (normalized by image size)
            if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx, fy = cam.params[0], cam.params[1]
                cx, cy = cam.params[2], cam.params[3]
            else:
                raise Exception(f"Unsupported camera model: {cam.model}")

            intrinsics = np.eye(3, dtype=np.float32)
            intrinsics[0, 0] = fx / cam.width
            intrinsics[1, 1] = fy / cam.height
            intrinsics[0, 2] = cx / cam.width
            intrinsics[1, 2] = cy / cam.height

            # Extrinsics: colmap gives w2c, apply scale to translation
            R = qvec2rotmat(colmap_img.qvec)
            t = colmap_img.tvec
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            c2w = np.linalg.inv(w2c)
            c2w[:3, 3] *= scale
            w2c = np.linalg.inv(c2w).astype(np.float32)

            frame_tmp["intrinsics"] = intrinsics
            frame_tmp["extrinsics"] = w2c
            frame_tmp["file_path"] = os.path.join(scene_path, "rgb", "1x", frame_name + ".png")
            scene_frames[int(frame_data["camera_id"])][int(frame_data["appearance_id"])] = frame_tmp

        # remove appearance_id only appeared in few cameras
        frame_id_key_list = [list(frames.keys()) for frames in scene_frames.values()]
        common_frame_ids = set(frame_id_key_list[0])
        for frame_ids in frame_id_key_list[1:]:
            common_frame_ids = common_frame_ids & set(frame_ids)
        common_frame_ids = sorted(list(common_frame_ids))
        scene_frames_final = []
        for frames in scene_frames.values():
            scene_frames_tmp = []
            for frame_id in common_frame_ids:
                scene_frames_tmp.append(frames[frame_id])
            scene_frames_final.append(scene_frames_tmp)
        return scene_frames_final, scene_id

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                file_path = file_path["file_path"]
                futures_with_idx.append(
                    (
                        idx,
                        executor.submit(
                            lambda p: self.to_tensor(Image.open(p).convert("RGB")),
                            file_path,
                        ),
                    )
                )
            
            # Pre-allocate list with correct size to maintain order
            torch_images = [None] * len(frames)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size
            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        # Return as list if images have different sizes
        return torch_images
        
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]
        
    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        
        scene = self.scene_ids[index]
        
        example = self.scenes[scene]
        # load poses
        extrinsics = []
        intrinsics = []
        for cam in example:
            extrinsic_cam = []
            intrinsic_cam = []
            for frame in cam:
                extrinsic = frame["extrinsics"]
                intrinsic = frame["intrinsics"]
                extrinsic_cam.append(extrinsic)
                intrinsic_cam.append(intrinsic)
            extrinsics.append(extrinsic_cam)
            intrinsics.append(intrinsic_cam)
        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics = torch.linalg.inv(extrinsics)
        
        V, T, _, _ = extrinsics.shape

        # Context cameras (default: camera 0 only)
        if self.cfg.context_cameras is not None:
            context_cameras_list = [c for c in self.cfg.context_cameras if c < V]
        else:
            context_cameras_list = [0]

        # Sample temporal indices for context
        num_context = min(32, T)
        max_t = min(T - 1, self.cfg.max_temporal - 1)
        context_indices = torch.linspace(0, max_t, num_context).long()

        # Target cameras
        if self.cfg.target_cameras is not None:
            target_cameras = [c for c in self.cfg.target_cameras if c < V]
        else:
            target_cameras = list(range(1, V))

        # Target temporal sampling
        if self.stage == "test":
            # Dense video: all frames between first and last context
            target_time_indices = torch.arange(0, max_t + 1, dtype=torch.long)
        elif self.cfg.num_target_frames is not None:
            num_tgt_t = min(self.cfg.num_target_frames, max_t + 1)
            target_time_indices = torch.linspace(0, max_t, num_tgt_t).long()
            target_time_indices = torch.unique(target_time_indices)
        else:
            target_time_indices = context_indices

        # Boolean mask: which target frames share a timestamp with context
        ctx_set = set(context_indices.tolist())
        is_ctx_time = torch.tensor(
            [i.item() in ctx_set for _ in target_cameras for i in target_time_indices],
            dtype=torch.bool,
        )

        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics.view(-1, 3, 3)).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        # Load the images.
        input_frames = [example[c][i] for c in context_cameras_list for i in context_indices]
        target_frame = [example[c][i] for c in target_cameras for i in target_time_indices]

        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)

        # Load covisible masks for target frames
        target_visibility = []
        for frame in target_frame:
            covisible_path = frame["file_path"].replace("rgb/1x", "covisible/2x/val")
            if os.path.exists(covisible_path):
                mask = self.to_tensor(Image.open(covisible_path).convert("L"))  # [1, h, w]
            else:
                mask = torch.ones(1, target_images.shape[-2], target_images.shape[-1])
            target_visibility.append(mask)
        target_visibility = torch.stack(target_visibility)  # [num_targets, 1, h, w]
        # Resize to match target image resolution if needed
        if target_visibility.shape[-2:] != target_images.shape[-2:]:
            target_visibility = F.interpolate(
                target_visibility, size=target_images.shape[-2:], mode='nearest'
            )
        target_visibility = (target_visibility > 0.5).float()

        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise Exception("Bad example image shape")
        
        # print(extrinsics.shape, intrinsics.shape, context_indices, target_indices)
        # print('---------------------------------------------')
        # Resize the world to make the baseline 1.
        # Use the first context camera as reference for scale computation
        ref_cam = context_cameras_list[0]
        context_extrinsics_ref = extrinsics[ref_cam]
        if self.cfg.make_baseline_1:
            a = context_extrinsics_ref[0, :3, 3]
            b_all = context_extrinsics_ref[1:, :3, 3]
            diff = b_all - a
            norms = diff.norm(dim=1)
            scale = norms.max()
            extrinsics[:, :, :3, 3] /= scale
        else:
            scale = 1

        # Build context tensors across all context cameras
        context_extrinsics = torch.cat(
            [extrinsics[c][context_indices] for c in context_cameras_list], dim=0
        )  # [len(ctx_cams) * num_ctx, 4, 4]
        context_intrinsics_tensor = torch.cat(
            [intrinsics[c][context_indices] for c in context_cameras_list], dim=0
        )
        context_indices_rep = context_indices.repeat(len(context_cameras_list))
        context_cameras_rep = torch.cat(
            [torch.full((len(context_indices),), c, dtype=torch.long) for c in context_cameras_list], dim=0
        )

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(context_extrinsics[0:1], extrinsics.view(-1, 4, 4)).view(V, T, 4, 4)
            context_extrinsics = torch.cat(
                [extrinsics[c][context_indices] for c in context_cameras_list], dim=0
            )
            context_intrinsics_tensor = torch.cat(
                [intrinsics[c][context_indices] for c in context_cameras_list], dim=0
            )

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        target_extrinsics = torch.stack([extrinsics[c][i] for c in target_cameras for i in target_time_indices], dim=0)
        target_intrinsics = torch.stack([intrinsics[c][i] for c in target_cameras for i in target_time_indices], dim=0)
        target_indices_rep = torch.tensor([i for c in target_cameras for i in target_time_indices], dtype=torch.long)
        target_cameras_rep = torch.tensor([c for c in target_cameras for i in target_time_indices], dtype=torch.long)
        N_ctx_total = len(context_cameras_list) * len(context_indices)
        overlap = torch.tensor([1.0])  # placeholder
        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics_tensor,
                "image": context_images,
                "near": self.get_bound("near", N_ctx_total) / scale,
                "far": self.get_bound("far", N_ctx_total) / scale,
                "index": context_indices_rep,
                "camera": context_cameras_rep,
                "overlap": overlap,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "visibility": target_visibility,
                "near": self.get_bound("near", len(target_indices_rep)) / scale,
                "far": self.get_bound("far", len(target_indices_rep)) / scale,
                "index": target_indices_rep,
                "camera": target_cameras_rep,
                "is_ctx_time": is_ctx_time,
            },
            "scene": "iphone_"+scene,
            "dataset_name": self.cfg.name,
        }
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14))
        return maybe_apply_davis_intrinsics(example, self.cfg.force_davis_intrinsics)
        
    def __getitem__(self, index: int) -> dict:
        num_context_views = self.view_sampler.num_context_views
        patchsize_h, patchsize_w = self.cfg.input_image_shape
        patchsize_h = patchsize_h // 14
        patchsize_w = patchsize_w // 14
        # return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            index = np.random.randint(len(self))
            return self.__getitem__(index)

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
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
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)
