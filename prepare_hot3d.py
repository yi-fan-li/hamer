"""
Preprocesses HOT3D sequences into HaMeR-compatible npz + JPEG files.

Processes Aria sequences only (RGB camera 214-1).
Outputs:
  - <img_dir>/<seq>/<frame>_<hand>.jpg  — cropped region saved at full image resolution
  - <output_npz>                         — npz with imgname, center, scale, right,
                                           keypoints_2d (21, 3), keypoints_3d (21, 4)
                                           in HaMeR's OpenPose joint ordering.

Usage:
    python prepare_hot3d.py \\
        --hot3d_root ~/hot3d/hot3d/dataset \\
        --mano_dir   _DATA/data/mano \\
        --img_dir    hamer_evaluation_data/hot3d_imgs \\
        --output_npz hamer_evaluation_data/hot3d_test.npz \\
        --every_nth  10
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── HOT3D imports ──────────────────────────────────────────────────────────────
HOT3D_ROOT = os.path.expanduser("~/hot3d/hot3d")
sys.path.insert(0, HOT3D_ROOT)

from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.mano_layer import MANOHandModel
from data_loaders.loader_hand_poses import Handedness
from data_loaders.headsets import Headset
from dataset_api import Hot3dDataProvider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.calibration import FISHEYE624

# ── Joint ordering mapping ─────────────────────────────────────────────────────
#
# HOT3D get_hand_landmarks() returns 20 joints in this order:
#   0  THUMB_TIP          6  THUMB_INTERMEDIATE (MCP)   12 MIDDLE_INTERMEDIATE
#   1  INDEX_TIP          7  THUMB_DISTAL (IP)           13 MIDDLE_DISTAL
#   2  MIDDLE_TIP         8  INDEX_PROXIMAL  (MCP)       14 RING_PROXIMAL
#   3  RING_TIP           9  INDEX_INTERMEDIATE          15 RING_INTERMEDIATE
#   4  PINKY_TIP         10  INDEX_DISTAL                16 RING_DISTAL
#   5  WRIST             11  MIDDLE_PROXIMAL             17 PINKY_PROXIMAL
#                                                        18 PINKY_INTERMEDIATE
#                                                        19 PINKY_DISTAL
#
# HaMeR OpenPose ordering (21 joints, pelvis_ind=0 = wrist):
#   0  wrist             5  index MCP     9  middle MCP    13 ring MCP    17 pinky MCP
#   1  thumb CMC (*)     6  index PIP    10  middle PIP    14 ring PIP    18 pinky PIP
#   2  thumb MCP         7  index DIP    11  middle DIP    15 ring DIP    19 pinky DIP
#   3  thumb IP          8  index TIP    12  middle TIP    16 ring TIP    20 pinky TIP
#   4  thumb TIP
#
# (*) thumb CMC has no HOT3D equivalent → interpolated as midpoint(wrist, thumb_MCP)
#
# For each HaMeR joint index i: HOT3D_SRC[i] is the HOT3D index to use (-1 = interpolate).
HOT3D_SRC = [
    5,   # 0  wrist           ← HOT3D 5
    -1,  # 1  thumb CMC       ← interpolated
    6,   # 2  thumb MCP       ← HOT3D 6
    7,   # 3  thumb IP        ← HOT3D 7
    0,   # 4  thumb TIP       ← HOT3D 0
    8,   # 5  index MCP       ← HOT3D 8
    9,   # 6  index PIP       ← HOT3D 9
    10,  # 7  index DIP       ← HOT3D 10
    1,   # 8  index TIP       ← HOT3D 1
    11,  # 9  middle MCP      ← HOT3D 11
    12,  # 10 middle PIP      ← HOT3D 12
    13,  # 11 middle DIP      ← HOT3D 13
    2,   # 12 middle TIP      ← HOT3D 2
    14,  # 13 ring MCP        ← HOT3D 14
    15,  # 14 ring PIP        ← HOT3D 15
    16,  # 15 ring DIP        ← HOT3D 16
    3,   # 16 ring TIP        ← HOT3D 3
    17,  # 17 pinky MCP       ← HOT3D 17
    18,  # 18 pinky PIP       ← HOT3D 18
    19,  # 19 pinky DIP       ← HOT3D 19
    4,   # 20 pinky TIP       ← HOT3D 4
]
N_JOINTS = 21


def hot3d_to_hamer(landmarks_hot3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 20-joint HOT3D landmarks (N,3) to 21-joint HaMeR OpenPose (N,3) + confidence (N,).
    Thumb CMC is estimated as the midpoint of wrist and thumb MCP with conf=0.5.
    """
    kps = np.zeros((N_JOINTS, 3), dtype=np.float32)
    conf = np.ones(N_JOINTS, dtype=np.float32)

    for hamer_idx, hot3d_idx in enumerate(HOT3D_SRC):
        if hot3d_idx >= 0:
            kps[hamer_idx] = landmarks_hot3d[hot3d_idx]
        else:
            # thumb CMC: midpoint between wrist (HOT3D 5) and thumb MCP (HOT3D 6)
            kps[hamer_idx] = (landmarks_hot3d[5] + landmarks_hot3d[6]) / 2.0
            conf[hamer_idx] = 0.5

    return kps, conf


def transform_points(T_mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) array."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T_mat @ pts_h.T)[:3].T


def process_sequence(
    seq_dir: str,
    mano_model: MANOHandModel,
    img_dir: str,
    stream_id: StreamId,
    every_nth: int,
    max_time_delta_ns: int,
) -> list[dict]:
    """Process one HOT3D sequence; return list of annotation dicts."""

    # Load provider (no object library needed for hand eval)
    try:
        provider = Hot3dDataProvider(
            sequence_folder=seq_dir,
            object_library=ObjectLibrary({}),
            mano_hand_model=mano_model,
            fail_on_missing_data=False,
        )
    except Exception as e:
        print(f"  Skipping {seq_dir}: {e}")
        return []

    # Only process Aria (has RGB camera)
    if provider.get_device_type() != Headset.Aria:
        return []

    if provider.mano_hand_data_provider is None:
        return []

    device_data = provider.device_data_provider
    hand_data = provider.mano_hand_data_provider
    hand_box2d = provider.hand_box2d_data_provider
    device_pose = provider.device_pose_data_provider

    # Camera calibration (FISHEYE624 for accurate projection into raw image)
    try:
        T_device_camera, cam_calib = device_data.get_camera_calibration(
            stream_id, camera_model=FISHEYE624
        )
        T_device_camera_mat = T_device_camera.to_matrix()  # 4x4
    except Exception as e:
        print(f"  Camera calibration failed for {seq_dir}: {e}")
        return []

    # All timestamps for this camera stream
    all_ts = device_data.get_sequence_timestamps(stream_id, TimeDomain.TIME_CODE)
    selected_ts = all_ts[::every_nth]

    seq_name = os.path.basename(seq_dir)
    results = []

    for ts in selected_ts:
        # ── Device pose ────────────────────────────────────────────────────────
        pose_with_dt = device_pose.get_pose_at_timestamp(
            ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
        )
        if pose_with_dt is None or abs(pose_with_dt.time_delta_ns) > max_time_delta_ns:
            continue
        T_world_device = pose_with_dt.pose3d.T_world_device  # SE3

        # World → camera transform
        T_world_camera = T_world_device @ T_device_camera
        T_camera_world_mat = T_world_camera.inverse().to_matrix()  # 4x4

        # ── Hand pose ──────────────────────────────────────────────────────────
        hand_with_dt = hand_data.get_pose_at_timestamp(
            ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
        )
        if hand_with_dt is None or abs(hand_with_dt.time_delta_ns) > max_time_delta_ns:
            continue

        # ── 2D bounding boxes ──────────────────────────────────────────────────
        box_with_dt = hand_box2d.get_bbox_at_timestamp(
            stream_id, ts, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE
        )
        if box_with_dt is None:
            continue

        # ── Image ──────────────────────────────────────────────────────────────
        img_arr = device_data.get_image(ts, stream_id)
        if img_arr is None:
            continue
        img_h, img_w = img_arr.shape[:2]

        # ── Per-hand processing ────────────────────────────────────────────────
        hand_index_to_handedness = {
            0: Handedness.Left,
            1: Handedness.Right,
        }
        for hand_index, handedness in hand_index_to_handedness.items():
            # Bounding box
            box2d_dict = box_with_dt.box2d_collection.box2ds
            if hand_index not in box2d_dict:
                continue
            hand_box = box2d_dict[hand_index]
            if hand_box.box2d is None:
                continue
            box = hand_box.box2d
            x_min, y_min = max(box.left, 0), max(box.top, 0)
            x_max, y_max = min(box.right, img_w), min(box.bottom, img_h)
            if x_max <= x_min or y_max <= y_min:
                continue
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            bbox_size = float(max(x_max - x_min, y_max - y_min))
            if bbox_size < 20:
                continue

            # Hand pose
            pose_collection = hand_with_dt.pose3d_collection.poses
            if handedness not in pose_collection:
                continue
            hand_pose = pose_collection[handedness]

            # 3D landmarks (world space, shape 20×3)
            landmarks_world = hand_data.get_hand_landmarks(hand_pose)
            if landmarks_world is None:
                continue
            lm_world = landmarks_world.numpy().astype(np.float32)  # (20, 3)

            # Transform to camera space
            lm_cam = transform_points(T_camera_world_mat, lm_world)  # (20, 3)

            # Reorder to HaMeR OpenPose ordering
            kps3d_xyz, conf3d = hot3d_to_hamer(lm_cam)  # (21, 3), (21,)
            keypoints_3d = np.concatenate(
                [kps3d_xyz, conf3d[:, None]], axis=1
            ).astype(np.float32)  # (21, 4)

            # Project 3D → 2D (image-space pixels)
            kps2d = np.zeros((N_JOINTS, 3), dtype=np.float32)
            for j in range(N_JOINTS):
                px = cam_calib.project(kps3d_xyz[j])
                if px is not None:
                    kps2d[j] = [float(px[0]), float(px[1]), conf3d[j]]
                else:
                    kps2d[j] = [0.0, 0.0, 0.0]

            # Save image
            hand_side = "right" if handedness == Handedness.Right else "left"
            rel_path = os.path.join(seq_name, f"{ts}_{hand_side}.jpg")
            save_path = os.path.join(img_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Convert RGB → BGR for cv2
            img_bgr = img_arr[:, :, ::-1].copy() if img_arr.ndim == 3 else img_arr
            cv2.imwrite(save_path, img_bgr)

            results.append({
                "imgname": rel_path,
                "center": np.array([cx, cy], dtype=np.float32),
                "scale": np.array([bbox_size], dtype=np.float32),
                "right": np.float32(1.0 if handedness == Handedness.Right else 0.0),
                "keypoints_2d": kps2d,
                "keypoints_3d": keypoints_3d,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess HOT3D for HaMeR eval")
    parser.add_argument("--hot3d_root", type=str, default=os.path.expanduser("~/hot3d/hot3d/dataset"),
                        help="Path to HOT3D dataset directory (contains sequence folders)")
    parser.add_argument("--mano_dir", type=str, default="_DATA/data/mano",
                        help="Directory with MANO_RIGHT.pkl and MANO_LEFT.pkl")
    parser.add_argument("--img_dir", type=str, default="hamer_evaluation_data/hot3d_imgs",
                        help="Output directory for extracted JPEG images")
    parser.add_argument("--output_npz", type=str, default="hamer_evaluation_data/hot3d_test.npz",
                        help="Output npz path")
    parser.add_argument("--every_nth", type=int, default=10,
                        help="Sample every Nth frame (default 10 ≈ 3fps from 30fps)")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Limit number of sequences processed (for testing)")
    parser.add_argument("--max_time_delta_ms", type=float, default=50.0,
                        help="Max allowed timestamp delta in ms when querying annotations")
    args = parser.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)

    # Load MANO model
    print(f"Loading MANO from {args.mano_dir}")
    mano_model = MANOHandModel(args.mano_dir)

    stream_id = StreamId("214-1")  # Aria RGB
    max_time_delta_ns = int(args.max_time_delta_ms * 1e6)

    # Find all sequence directories
    seq_dirs = sorted([
        os.path.join(args.hot3d_root, d)
        for d in os.listdir(args.hot3d_root)
        if os.path.isdir(os.path.join(args.hot3d_root, d))
    ])
    if args.max_sequences is not None:
        seq_dirs = seq_dirs[:args.max_sequences]

    print(f"Processing {len(seq_dirs)} sequences...")

    all_records = []
    for seq_dir in tqdm(seq_dirs):
        records = process_sequence(
            seq_dir, mano_model, args.img_dir,
            stream_id, args.every_nth, max_time_delta_ns
        )
        all_records.extend(records)

    if not all_records:
        print("No records found. Check that sequences have Aria RGB data and MANO annotations.")
        return

    print(f"Collected {len(all_records)} hand samples. Saving npz...")

    np.savez(
        args.output_npz,
        imgname=np.array([r["imgname"] for r in all_records]),
        center=np.stack([r["center"] for r in all_records]),
        scale=np.stack([r["scale"] for r in all_records]),
        right=np.array([r["right"] for r in all_records]),
        keypoints_2d=np.stack([r["keypoints_2d"] for r in all_records]),
        keypoints_3d=np.stack([r["keypoints_3d"] for r in all_records]),
    )
    print(f"Saved {args.output_npz}")
    print(f"  right hands: {sum(r['right'] == 1.0 for r in all_records)}")
    print(f"  left hands:  {sum(r['right'] == 0.0 for r in all_records)}")


if __name__ == "__main__":
    main()
