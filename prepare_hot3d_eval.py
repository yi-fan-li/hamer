#!/usr/bin/env python3
"""
Build HOT3D metadata NPZ for HaMeR evaluation.

No images are extracted — HOT3DDataset reads frames directly from VRS at eval time.
Only a small NPZ (~50–200 MB) is written containing bounding boxes, timestamps,
VRS paths, and projected GT keypoints.

Requirements:
  - conda activate hamer   (or hamer_dino_train)  — both have projectaria_tools
  - MANO model files for GT keypoints (default path already set):
      /home/yifanli/github/hamer/_DATA/data/mano/MANO_RIGHT.pkl
      /home/yifanli/github/hamer/_DATA/data/mano/MANO_LEFT.pkl

Usage:
    # Training split, every 10th frame, with GT keypoints:
    python prepare_hot3d_eval.py \\
        --dataset_root /home/yifanli/hot3d/hot3d/dataset \\
        --out_npz hamer_evaluation_data/hot3d_test.npz \\
        --mano_dir /tmp/mano_models --split train --stride 10

    # Quick test on one sequence:
    python prepare_hot3d_eval.py \\
        --dataset_root /home/yifanli/hot3d/hot3d/dataset \\
        --out_npz /tmp/hot3d_test.npz \\
        --sequences P0001_10a27bf7 --stride 30
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Put the bundled hot3d code on the path (third-party/hot3d in the hamer repo)
# ---------------------------------------------------------------------------
HAMER_ROOT = Path(__file__).parent
HOT3D_CODE_DIR = str(HAMER_ROOT / "third-party" / "hot3d")
if HOT3D_CODE_DIR not in sys.path:
    sys.path.insert(0, HOT3D_CODE_DIR)

# HaMeR's OpenPose joint ordering applied to raw smplx MANO output
# (matches hamer/models/mano_wrapper.py: mano_to_openpose)
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_split_sequences(dataset_root: Path, split: str):
    """Return sequence directories for the requested split."""
    split_dir = dataset_root / split
    if split_dir.exists():
        # Use the official symlink-based split directories
        return sorted(p.resolve() for p in split_dir.iterdir() if p.is_dir())
    # Fallback: return everything
    return sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name != "assets")


def load_qa_mask(seq_dir: Path, stream_id):
    """
    Load mask_qa_pass.csv for the sequence and return a set of valid timestamps
    for the given stream_id, or None if the mask file is unavailable.
    """
    mask_path = seq_dir / "masks" / "mask_qa_pass.csv"
    if not mask_path.exists():
        return None
    try:
        from data_loaders.loader_masks import load_mask_data
        mask_data = load_mask_data(str(mask_path))
        stream_mask = mask_data.stream_mask(stream_id)
        if stream_mask is None:
            return None
        return {ts for ts, ok in stream_mask.items() if ok}
    except Exception:
        return None


def project_to_linear(landmarks_world, T_world_camera, linear_cal, img_w, img_h):
    """
    Project world-frame landmarks onto the undistorted (linear/pinhole) image.

    Args:
        landmarks_world: (N, 3) float array in world frame (metres)
        T_world_camera:  SE3 transform  (T_world_camera = T_world_device @ T_device_camera)
        linear_cal:      LINEAR CameraCalibration
        img_w, img_h:    undistorted image dimensions

    Returns:
        kp2d: (N, 3) [u, v, conf]  conf=1 if inside image
    """
    T_camera_world = T_world_camera.inverse()
    kp2d = np.zeros((len(landmarks_world), 3), dtype=np.float32)
    for i, pt_world in enumerate(landmarks_world):
        pt_cam = np.asarray(T_camera_world @ pt_world).reshape(3)
        if pt_cam[2] <= 0:
            continue
        uv = linear_cal.project(pt_cam)
        if uv is None:
            continue
        u, v = float(uv[0]), float(uv[1])
        if 0 <= u < img_w and 0 <= v < img_h:
            kp2d[i] = [u, v, 1.0]
    return kp2d


def process_sequence(
    seq_dir: Path,
    mano_model,
    object_library,
    stride: int,
    min_visibility: float,
):
    """Process one HOT3D sequence, returning a list of annotation dicts."""
    from dataset_api import Hot3dDataProvider
    from data_loaders.loader_hand_poses import Handedness, RIGHT_HAND_INDEX
    from projectaria_tools.core.calibration import FISHEYE624, LINEAR
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    try:
        provider = Hot3dDataProvider(
            sequence_folder=str(seq_dir),
            object_library=object_library,
            mano_hand_model=mano_model,
            fail_on_missing_data=False,
        )
    except Exception as e:
        print(f"  WARN: Cannot load {seq_dir.name}: {e}")
        return []

    device_prov    = provider.device_data_provider
    hand_box_prov  = provider.hand_box2d_data_provider
    mano_prov      = provider.mano_hand_data_provider
    device_pose_prov = provider.device_pose_data_provider

    if hand_box_prov is None:
        return []

    # Pick the RGB camera stream
    stream_ids = device_prov.get_image_stream_ids()
    rgb_stream = next(
        (s for s in stream_ids
         if "rgb" in device_prov.get_image_stream_label(s).lower()),
        stream_ids[0],
    )
    stream_label = device_prov.get_image_stream_label(rgb_stream)

    # Camera calibration: fisheye for extrinsics, linear for GT projection
    try:
        [T_device_camera, fisheye_cal] = device_prov.get_camera_calibration(rgb_stream, FISHEYE624)
        [_,               linear_cal]  = device_prov.get_camera_calibration(rgb_stream, LINEAR)
        img_w, img_h = linear_cal.get_image_size()
    except Exception as e:
        print(f"  WARN: No calibration for {seq_dir.name}: {e}")
        return []

    # QA mask for this stream
    qa_timestamps = load_qa_mask(seq_dir, rgb_stream)

    bbox_ts_list = hand_box_prov.get_timestamp_ns_list(rgb_stream)
    if not bbox_ts_list:
        return []

    vrs_path = str(seq_dir / "recording.vrs")
    annotations = []

    for ts in bbox_ts_list[::stride]:
        # Skip frames that fail QA
        if qa_timestamps is not None and ts not in qa_timestamps:
            continue

        bbox_result = hand_box_prov.get_bbox_at_timestamp(
            rgb_stream, ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
        )
        if bbox_result is None or not bbox_result.box2d_collection.box2ds:
            continue

        # Device pose → T_world_camera for each hand
        T_world_camera = None
        if device_pose_prov is not None:
            dp = device_pose_prov.get_pose_at_timestamp(
                ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE,
                acceptable_time_delta=int(2e7),
            )
            if dp is not None:
                # Tutorial formula: T_world_camera = T_world_device @ T_device_camera
                T_world_camera = dp.pose3d.T_world_device @ T_device_camera

        # MANO poses
        mano_poses = None
        if mano_prov is not None:
            mh = mano_prov.get_pose_at_timestamp(
                ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE,
                acceptable_time_delta=int(2e7),
            )
            if mh is not None:
                mano_poses = mh.pose3d_collection.poses

        for hand_idx, hand_box in bbox_result.box2d_collection.box2ds.items():
            if hand_box.box2d is None:
                continue
            vis = hand_box.visibility_ratio or 0.0
            if vis < min_visibility:
                continue

            box = hand_box.box2d
            cx = (box.left + box.right)  / 2.0
            cy = (box.top  + box.bottom) / 2.0
            w  = box.right - box.left
            h  = box.bottom - box.top
            is_right = int(hand_idx == RIGHT_HAND_INDEX)

            kp2d = np.zeros((21, 3), dtype=np.float32)
            kp3d = np.zeros((21, 4), dtype=np.float32)

            if mano_poses is not None and T_world_camera is not None:
                handedness = Handedness.Right if is_right else Handedness.Left
                if handedness in mano_poses:
                    lm_tensor = mano_prov.get_hand_landmarks(mano_poses[handedness])
                    if lm_tensor is not None:
                        lm_world = lm_tensor.numpy()   # raw smplx joints (joint_mapper=None)
                        if len(lm_world) >= 21:
                            lm_world = lm_world[MANO_TO_OPENPOSE]  # → HaMeR OpenPose order

                        # World → camera (for 3D GT)
                        T_camera_world = T_world_camera.inverse()
                        lm_cam = np.array(
                            [np.asarray(T_camera_world @ pt).reshape(3) for pt in lm_world],
                            dtype=np.float32,
                        )
                        kp3d[:len(lm_cam), :3] = lm_cam
                        kp3d[:len(lm_cam),  3] = 1.0

                        # Project onto undistorted (linear) image for 2D GT
                        if len(lm_world) == 21:
                            kp2d = project_to_linear(
                                lm_world, T_world_camera, linear_cal, img_w, img_h
                            )

            annotations.append({
                "vrs_path":           vrs_path,
                "seq_dir":            str(seq_dir),
                "timestamp_ns":       np.int64(ts),
                "stream_label":       stream_label,
                "center":             np.array([cx, cy], dtype=np.float32),
                "scale":              np.array([w,  h],  dtype=np.float32),
                "right":              np.float32(is_right),
                "hand_keypoints_2d":  kp2d,
                "hand_keypoints_3d":  kp3d,
                "extra_info": {
                    "seq":              seq_dir.name,
                    "timestamp_ns":     int(ts),
                    "stream_label":     stream_label,
                    "hand_index":       hand_idx,
                    "visibility_ratio": float(vis),
                },
            })

    return annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build HOT3D metadata NPZ for HaMeR eval (no images extracted)"
    )
    parser.add_argument(
        "--dataset_root",
        default="/home/yifanli/hot3d/hot3d/dataset",
        help="Root of the HOT3D dataset (contains train/, test/, assets/, P000X_xxx/…)",
    )
    parser.add_argument(
        "--out_npz",
        default="hamer_evaluation_data/hot3d_test.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--mano_dir",
        default="/home/yifanli/github/hamer/_DATA/data/mano",
        help="Dir with MANO_LEFT.pkl + MANO_RIGHT.pkl (enables GT keypoints)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "all"],
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample every N-th frame",
    )
    parser.add_argument(
        "--min_visibility",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated sequence names to process instead of a full split",
    )
    args = parser.parse_args()

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    object_library_path = dataset_root / "assets"

    # Object library (needed by Hot3dDataProvider)
    from data_loaders.loader_object_library import load_object_library
    object_library = load_object_library(object_library_folderpath=str(object_library_path))

    # Optional MANO model (joint_mapper=None → raw smplx joints → apply OpenPose mapping)
    mano_model = None
    if args.mano_dir:
        from data_loaders.mano_layer import MANOHandModel
        try:
            mano_model = MANOHandModel(args.mano_dir, joint_mapper=None)
            print(f"Loaded MANO model from {args.mano_dir}")
        except Exception as e:
            print(f"WARN: Could not load MANO model ({e}) — GT keypoints disabled.")

    # Collect sequences
    if args.sequences:
        sequences = [dataset_root / s.strip() for s in args.sequences.split(",")]
    elif args.split == "all":
        sequences = sorted(p for p in dataset_root.iterdir()
                           if p.is_dir() and p.name not in ("assets", "train", "test"))
    else:
        sequences = get_split_sequences(dataset_root, args.split)

    print(f"Processing {len(sequences)} sequence(s)  [split={args.split}, stride={args.stride}]")

    all_anns = []
    for seq_dir in tqdm(sequences, desc="Sequences"):
        if not seq_dir.exists():
            print(f"  WARN: {seq_dir} not found")
            continue
        anns = process_sequence(seq_dir, mano_model, object_library, args.stride, args.min_visibility)
        all_anns.extend(anns)
        tqdm.write(f"  {seq_dir.name}: +{len(anns)}  total={len(all_anns)}")

    if not all_anns:
        print("No annotations collected.")
        return

    print(f"\nSaving {len(all_anns)} samples → {out_npz}")
    np.savez(
        str(out_npz),
        vrs_path          = np.array([a["vrs_path"]          for a in all_anns], dtype=object),
        seq_dir           = np.array([a["seq_dir"]            for a in all_anns], dtype=object),
        timestamp_ns      = np.array([a["timestamp_ns"]       for a in all_anns], dtype=np.int64),
        stream_label      = np.array([a["stream_label"]       for a in all_anns], dtype=object),
        center            = np.array([a["center"]             for a in all_anns], dtype=np.float32),
        scale             = np.array([a["scale"]              for a in all_anns], dtype=np.float32),
        right             = np.array([a["right"]              for a in all_anns], dtype=np.float32),
        hand_keypoints_2d = np.array([a["hand_keypoints_2d"]  for a in all_anns], dtype=np.float32),
        hand_keypoints_3d = np.array([a["hand_keypoints_3d"]  for a in all_anns], dtype=np.float32),
        extra_info        = np.array([a["extra_info"]         for a in all_anns], dtype=object),
    )

    has_kp2d = (np.array([a["hand_keypoints_2d"] for a in all_anns])[:, :, 2] > 0).any(axis=1).mean()
    print(f"  NPZ size:            {out_npz.stat().st_size / 1e6:.1f} MB  (no images on disk)")
    print(f"  Right-hand fraction: {np.array([a['right'] for a in all_anns]).mean():.2f}")
    print(f"  GT 2D keypoints:     {has_kp2d:.1%} of samples")
    print(
        f"\nNext: run eval in the HaMeR env:\n"
        f"  python eval.py --dataset HOT3D-VAL --checkpoint <ckpt>\n"
        f"  (datasets_eval.yaml HOT3D-VAL DATASET_FILE should point to {out_npz.resolve()})"
    )


if __name__ == "__main__":
    main()
