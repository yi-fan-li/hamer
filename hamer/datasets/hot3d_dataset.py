"""
HOT3DDataset — reads frames directly from HOT3D VRS files at eval time.
No image extraction to disk required.

Uses the bundled third-party/hot3d code and projectaria_tools from the
hamer conda environment (Python 3.10), same as prepare_hot3d_eval.py.
"""

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from yacs.config import CfgNode

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio

# ---------------------------------------------------------------------------
# Put the bundled hot3d code on the path (third-party/hot3d in the hamer repo)
# ---------------------------------------------------------------------------
_HAMER_ROOT = Path(__file__).parent.parent.parent
_HOT3D_CODE_DIR = str(_HAMER_ROOT / "third-party" / "hot3d")
if _HOT3D_CODE_DIR not in sys.path:
    sys.path.insert(0, _HOT3D_CODE_DIR)

# Module-level VRS provider cache — one provider per VRS path, per worker process.
# Each DataLoader worker is a separate process, so workers don't share state.
_provider_cache: Dict[str, Any] = {}

FLIP_KEYPOINT_PERMUTATION = list(range(21))


def _get_provider(vrs_path: str, seq_dir: str):
    """
    Return a cached (device_provider, calib_cache) for vrs_path.
    Opens the VRS file lazily on first access within a worker.

    calib_cache: dict mapping stream_label → (stream_id, T_dev_cam, fisheye_cal, linear_cal)
    """
    if vrs_path in _provider_cache:
        return _provider_cache[vrs_path]

    from data_loaders.headsets import Headset
    from data_loaders.io_utils import load_json
    from projectaria_tools.core.calibration import FISHEYE624, LINEAR

    metadata = load_json(os.path.join(seq_dir, "metadata.json"))
    headset = Headset[metadata["headset"]]

    if headset == Headset.Aria:
        from data_loaders.AriaDataProvider import AriaDataProvider
        # mps_folder_path=None: skip online calibration (not needed for eval)
        provider = AriaDataProvider(vrs_path, mps_folder_path=None)
    else:
        from data_loaders.QuestDataProvider import QuestDataProvider
        camera_models_path = os.path.join(seq_dir, "camera_models.json")
        provider = QuestDataProvider(vrs_path, camera_models_path)

    # Pre-compute linear calibrations for all image streams (avoid per-frame overhead)
    calib_cache = {}
    for sid in provider.get_image_stream_ids():
        label = provider.get_image_stream_label(sid)
        try:
            T_dev_cam, fisheye_cal = provider.get_camera_calibration(sid, FISHEYE624)
            _, linear_cal = provider.get_camera_calibration(sid, LINEAR)
            calib_cache[label] = (sid, T_dev_cam, fisheye_cal, linear_cal)
        except Exception:
            pass

    _provider_cache[vrs_path] = (provider, calib_cache)
    return _provider_cache[vrs_path]


def _load_frame(vrs_path: str, seq_dir: str, stream_label: str, timestamp_ns: int):
    """
    Load and undistort a single frame from a VRS file.
    Returns an RGB uint8 numpy array, or None on failure.
    Uses factory calibration (no MPS/online calibration required).
    """
    from projectaria_tools.core.calibration import distort_by_calibration
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    try:
        provider, calib_cache = _get_provider(vrs_path, seq_dir)
    except Exception:
        return None

    if stream_label not in calib_cache:
        return None
    sid, _, fisheye_cal, linear_cal = calib_cache[stream_label]

    try:
        raw = provider.get_image(timestamp_ns, sid)
    except Exception:
        return None
    if raw is None:
        return None

    # Undistort using factory calibration (tutorial pattern)
    try:
        img = distort_by_calibration(raw, linear_cal, fisheye_cal)
    except Exception:
        img = raw  # fall back to distorted image

    # Normalise to RGB uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # Aria/Quest images come out as RGB — no channel swap needed

    return img


class HOT3DDataset(Dataset):
    """
    HaMeR evaluation dataset for HOT3D.

    Reads frames directly from VRS recordings on-the-fly — no image extraction
    required.  Uses the NPZ produced by prepare_hot3d_eval.py which stores VRS
    paths, timestamps, bounding boxes, and projected GT keypoints.

    datasets_eval.yaml entry (IMG_DIR is unused but required by the factory):
        HOT3D-VAL:
            TYPE: HOT3DDataset
            DATASET_FILE: hamer_evaluation_data/hot3d_train.npz
            IMG_DIR: ""
            KEYPOINT_LIST: [0]
    """

    def __init__(
        self,
        cfg: CfgNode,
        dataset_file: str,
        img_dir: str = "",       # unused; images come from VRS
        train: bool = False,
        rescale_factor: float = 2,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
        self.rescale_factor = rescale_factor
        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        data = np.load(dataset_file, allow_pickle=True)
        num_pose = 3 * (cfg.MANO.NUM_HAND_JOINTS + 1)

        self.vrs_paths     = data["vrs_path"]
        self.seq_dirs      = data["seq_dir"]
        self.timestamps    = data["timestamp_ns"].astype(np.int64)
        self.stream_labels = data["stream_label"]

        self.center = data["center"].astype(np.float32)
        self.scale  = data["scale"].reshape(len(self.center), -1) / 200.0
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))

        try:
            self.right = data["right"].astype(np.float32)
        except KeyError:
            self.right = np.ones(len(self.center), dtype=np.float32)

        try:
            self.keypoints_2d = data["hand_keypoints_2d"].astype(np.float32)
        except KeyError:
            self.keypoints_2d = np.zeros((len(self.center), 21, 3), dtype=np.float32)

        try:
            self.keypoints_3d = data["hand_keypoints_3d"].astype(np.float32)
        except KeyError:
            self.keypoints_3d = np.zeros((len(self.center), 21, 4), dtype=np.float32)

        try:
            self.hand_pose    = data["hand_pose"].astype(np.float32)
            self.has_hand_pose = data["has_hand_pose"].astype(np.float32)
        except KeyError:
            self.hand_pose    = np.zeros((len(self.center), num_pose), dtype=np.float32)
            self.has_hand_pose = np.zeros(len(self.center), dtype=np.float32)

        try:
            self.betas     = data["betas"].astype(np.float32)
            self.has_betas = data["has_betas"].astype(np.float32)
        except KeyError:
            self.betas     = np.zeros((len(self.center), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.center), dtype=np.float32)

        self.personid   = np.zeros(len(self.center), dtype=np.int32)
        self.extra_info = data.get(
            "extra_info", np.array([{} for _ in range(len(self.center))], dtype=object)
        )

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        vrs_path     = str(self.vrs_paths[idx])
        seq_dir      = str(self.seq_dirs[idx])
        ts           = int(self.timestamps[idx])
        stream_label = str(self.stream_labels[idx])

        # Load image from VRS (on-the-fly, cached provider per worker)
        img = _load_frame(vrs_path, seq_dir, stream_label, ts)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        center = self.center[idx].copy()
        scale  = self.scale[idx].copy()
        right  = self.right[idx].copy()

        if self.rescale_factor == -1:
            BBOX_SHAPE = self.cfg.MODEL.get("BBOX_SHAPE", None)
            bbox_size  = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
            bbox_expand_factor = bbox_size / ((scale * 200).max())
        else:
            bbox_expand_factor = self.rescale_factor
            bbox_size = bbox_expand_factor * scale.max() * 200

        hand_pose     = self.hand_pose[idx].copy().astype(np.float32)
        betas         = self.betas[idx].copy().astype(np.float32)
        has_hand_pose = self.has_hand_pose[idx].copy()
        has_betas     = self.has_betas[idx].copy()

        mano_params = {
            "global_orient": hand_pose[:3],
            "hand_pose":     hand_pose[3:],
            "betas":         betas,
        }
        has_mano_params = {
            "global_orient": has_hand_pose,
            "hand_pose":     has_hand_pose,
            "betas":         has_betas,
        }
        mano_params_is_axis_angle = {
            "global_orient": True,
            "hand_pose":     True,
            "betas":         False,
        }

        augm_config  = self.cfg.DATASETS.CONFIG
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        # get_example accepts either a file path or an RGB numpy array directly
        img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size = (
            get_example(
                img,
                center[0], center[1],
                bbox_size, bbox_size,
                keypoints_2d, keypoints_3d,
                mano_params, has_mano_params,
                self.flip_keypoint_permutation,
                self.img_size, self.img_size,
                self.mean, self.std,
                self.train,
                right,
                augm_config,
                is_bgr=False,
            )
        )

        return {
            "img":                       img_patch,
            "keypoints_2d":              keypoints_2d,
            "keypoints_3d":              keypoints_3d,
            "orig_keypoints_2d":         self.keypoints_2d[idx].copy(),
            "box_center":                center,
            "box_size":                  bbox_size,
            "bbox_expand_factor":        bbox_expand_factor,
            "img_size":                  np.array(img_size),
            "mano_params":               mano_params,
            "has_mano_params":           has_mano_params,
            "mano_params_is_axis_angle": mano_params_is_axis_angle,
            "imgname":                   f"{seq_dir}/{ts}",
            "vrs_path":                  vrs_path,
            "stream_label":              stream_label,
            "timestamp_ns":              ts,
            "personid":                  int(self.personid[idx]),
            "extra_info":                self.extra_info[idx],
            "idx":                       idx,
            "_scale":                    scale,
            "right":                     right,
        }
