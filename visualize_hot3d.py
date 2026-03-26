"""
Visualize HaMeR predictions on HOT3D data and compile into a video.

Loads frames directly from VRS files — no image extraction required.
Groups entries by (sequence, timestamp) so both hands in the same frame
are rendered together in a single scene.

Output: two-panel video — [original frame | mesh overlay on full frame]

Usage:
    python visualize_hot3d.py \
        --checkpoint _DATA/hamer_ckpts/dinov3/checkpoints/dinov3_base.ckpt \
        --npz hamer_evaluation_data/hot3d_test.npz \
        --out_video hot3d_vis.mp4 \
        --num_frames 200
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hamer.datasets.hot3d_dataset import HOT3DDataset, _load_frame
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE  = (0.65098039, 0.74117647, 0.85882353)
HAND_ORANGE = (0.96, 0.60, 0.22)   # left hand in a distinct colour


def run_model_on_item(model, model_cfg, item, device):
    """Run HaMeR on a single dataset item. Returns (verts, cam_t, is_right) or None."""
    batch = {k: torch.from_numpy(v[None]).to(device) if isinstance(v, np.ndarray)
             else torch.tensor([v]).to(device) if isinstance(v, (int, float, np.floating, np.integer))
             else v
             for k, v in item.items()
             if not isinstance(v, (str, dict))}
    # img tensor is already float from the dataset
    with torch.no_grad():
        out = model(batch)

    is_right   = float(item['right'])
    verts      = out['pred_vertices'][0].cpu().numpy().copy()
    pred_cam   = out['pred_cam']                            # (1, 3)

    # Flip x for left hand (model always predicts right-hand space)
    verts[:, 0] *= (2 * is_right - 1)
    pred_cam_full = pred_cam.clone()
    pred_cam_full[:, 1] *= (2 * is_right - 1)

    box_center = torch.tensor(item['box_center'][None], device=device)
    box_size   = torch.tensor([item['box_size']], device=device)
    img_size   = torch.tensor(item['img_size'][None], device=device)

    cam_t = cam_crop_to_full(
        pred_cam_full, box_center, box_size, img_size,
        focal_length=model_cfg.EXTRA.FOCAL_LENGTH,
    )[0].cpu().numpy()

    return verts, cam_t, is_right


def render_hands_on_frame(renderer, full_rgb, verts_list, cam_t_list, is_right_list):
    """
    Render all hands in verts_list onto full_rgb (H×W×3 float32 [0,1] RGB).
    Returns composited RGB float32 image.
    """
    H, W = full_rgb.shape[:2]
    colors = [LIGHT_BLUE if r else HAND_ORANGE for r in is_right_list]

    rgba = renderer.render_rgba_multiple(
        vertices   = verts_list,
        cam_t      = cam_t_list,
        render_res = [W, H],
        focal_length = renderer.focal_length,
        is_right   = [int(r) for r in is_right_list],
        mesh_base_color = LIGHT_BLUE,   # per-mesh colour not supported; use single colour
    )
    # Composite RGBA over the original frame
    mask = rgba[:, :, 3:4]
    composited = rgba[:, :, :3] * mask + full_rgb * (1 - mask)
    return composited.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='_DATA/hamer_ckpts/dinov3/checkpoints/dinov3_base.ckpt')
    parser.add_argument('--npz', type=str,
                        default='hamer_evaluation_data/hot3d_test.npz',
                        help='NPZ produced by prepare_hot3d_eval.py')
    parser.add_argument('--out_video', type=str, default='hot3d_vis.mp4')
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Max number of unique frames (timestamps) to render')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--panel_height', type=int, default=720,
                        help='Height of each panel in the output video')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # -----------------------------------------------------------------------
    # Load dataset and group indices by (vrs_path, timestamp_ns)
    # Sorting by (vrs_path, timestamp_ns) gives temporal order per sequence.
    # -----------------------------------------------------------------------
    dataset = HOT3DDataset(
        cfg=model_cfg,
        dataset_file=args.npz,
        train=False,
        rescale_factor=2.0,
    )

    # Build frame groups: key = (vrs_path, timestamp_ns) → list of dataset indices
    frame_groups = defaultdict(list)
    for idx in range(len(dataset)):
        key = (str(dataset.vrs_paths[idx]), int(dataset.timestamps[idx]))
        frame_groups[key].append(idx)

    # Sort groups by (vrs_path, timestamp) for temporal continuity
    sorted_keys = sorted(frame_groups.keys(), key=lambda k: k)

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------
    video_writer = None
    frames_done  = 0
    pbar = tqdm(total=min(args.num_frames, len(sorted_keys)), desc='Rendering', unit='frame')

    for vrs_path, timestamp_ns in sorted_keys:
        if frames_done >= args.num_frames:
            break

        indices = frame_groups[(vrs_path, timestamp_ns)]

        # Load the full frame once
        item0    = dataset[indices[0]]
        seq_dir  = str(Path(item0['imgname']).parent)
        stream_label = item0['stream_label']
        full_rgb = _load_frame(vrs_path, seq_dir, stream_label, timestamp_ns)
        if full_rgb is None:
            continue
        full_rgb_f = full_rgb.astype(np.float32) / 255.0   # RGB float [0,1]

        # Run model on each hand in this frame
        verts_list, cam_t_list, is_right_list = [], [], []
        for idx in indices:
            item = dataset[idx]
            try:
                verts, cam_t, is_right = run_model_on_item(model, model_cfg, item, device)
                verts_list.append(verts)
                cam_t_list.append(cam_t)
                is_right_list.append(is_right)
            except Exception as e:
                print(f'  WARN: inference failed for idx={idx}: {e}')

        if not verts_list:
            continue

        # Render all hands onto the full frame
        overlay_rgb = render_hands_on_frame(
            renderer, full_rgb_f, verts_list, cam_t_list, is_right_list
        )

        # Build two-panel frame
        H = args.panel_height
        def to_bgr_panel(rgb_f):
            bgr = (rgb_f[:, :, ::-1] * 255).astype(np.uint8)
            new_w = max(1, int(bgr.shape[1] * H / bgr.shape[0]))
            return cv2.resize(bgr, (new_w, H))

        panel_orig    = to_bgr_panel(full_rgb_f)
        panel_overlay = to_bgr_panel(overlay_rgb)
        frame_out = np.concatenate([panel_orig, panel_overlay], axis=1)

        # Init video writer on first frame
        if video_writer is None:
            fh, fw = frame_out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.out_video, fourcc, args.fps, (fw, fh))

        video_writer.write(frame_out)
        frames_done += 1
        pbar.update(1)

    pbar.close()
    if video_writer is not None:
        video_writer.release()
        print(f'Saved {frames_done} frames → {args.out_video}')
    else:
        print('No frames rendered.')


if __name__ == '__main__':
    main()
