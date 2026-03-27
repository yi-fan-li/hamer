"""
Visualize model predictions on eval dataset samples.
Bypasses ViTPose/mmpose by using pre-computed bboxes from the eval dataset.

Usage:
    python visualize_eval.py --checkpoint logs/train/runs/hamer/checkpoints/dinov3_base.ckpt \
                              --dataset FREIHAND-VAL --num_samples 50
"""
import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER, dataset_eval_config
from hamer.datasets import create_dataset
from hamer.models import download_models, load_hamer, load_hamer_gcn, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def main():
    parser = argparse.ArgumentParser(description='Visualize HaMeR predictions on eval data')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str, default='FREIHAND-VAL',
                        help='Eval dataset name (must be in datasets_eval.yaml)')
    parser.add_argument('--out_folder', type=str, default='vis_out',
                        help='Output folder for rendered images')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--side_view', action='store_true', default=False,
                        help='Also render a side view next to the main render')
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    # Load model — auto-detect GCN checkpoint by presence of GCN config section
    download_models(CACHE_DIR_HAMER)
    from pathlib import Path
    from hamer.configs import get_config
    _cfg_path = str(Path(args.checkpoint).parent.parent / 'model_config.yaml')
    _cfg = get_config(_cfg_path, update_cachedir=True)
    if hasattr(_cfg, 'GCN'):
        model, model_cfg = load_hamer_gcn(args.checkpoint)
    else:
        model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load eval dataset (provides pre-computed bboxes — no ViTPose needed)
    dataset_cfg = dataset_eval_config()[args.dataset]
    dataset = create_dataset(model_cfg, dataset_cfg, train=False, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    renderer = Renderer(model_cfg, faces=model.mano.faces)

    total_saved = 0
    pbar = tqdm(total=args.num_samples, desc='Visualizing', unit='img')
    for batch in dataloader:
        if total_saved >= args.num_samples:
            break

        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_vertices = out['pred_vertices']       # (B, 778, 3)
        pred_cam      = out['pred_cam']             # (B, 3)
        box_center    = batch['box_center']         # (B, 2)
        box_size      = batch['box_size']           # (B,)
        img_size      = batch['img_size']           # (B, 2)  [W, H]
        right         = batch['right']              # (B,)  1=right hand, 0=left hand
        imgnames      = batch['imgname']            # list of B strings

        # Flip x-translation for left hands (model predicts in flipped/right-hand space)
        multiplier = (2 * right - 1)
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        # Convert crop-space camera to full-image camera translation
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                 focal_length=model_cfg.EXTRA.FOCAL_LENGTH)

        batch_size = pred_vertices.shape[0]
        for i in range(batch_size):
            if total_saved >= args.num_samples:
                break

            imgname = imgnames[i] if isinstance(imgnames[i], str) else imgnames[i].decode()
            is_right = right[i].cpu().numpy()
            verts    = pred_vertices[i].cpu().numpy()
            # Flip vertex x for left hands to restore image-space orientation
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            t        = cam_t[i].cpu().numpy()

            img_full = cv2.imread(imgname)
            if img_full is None:
                print(f'Warning: could not read {imgname}, skipping.')
                continue

            # 1. Overlay: mesh on top of original image
            overlay = renderer(
                vertices=verts,
                camera_translation=t.copy(),
                image=batch['img'][i],
                full_frame=True,
                imgname=imgname,
                mesh_base_color=LIGHT_BLUE,
            )
            overlay_bgr = (overlay[:, :, ::-1] * 255).astype(np.uint8)

            # 2. Mesh only: render on black background at same resolution
            mesh_only = renderer(
                vertices=verts,
                camera_translation=t.copy(),
                image=batch['img'][i],
                full_frame=True,
                imgname=imgname,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                return_rgba=True,
            )
            mesh_only_bgr = (mesh_only[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)

            # Resize all panels to the same height before concatenating
            h = img_full.shape[0]
            def resize_h(img):
                return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

            panels = [img_full, resize_h(overlay_bgr), resize_h(mesh_only_bgr)]

            if args.side_view:
                side = renderer(
                    vertices=verts,
                    camera_translation=t.copy(),
                    image=batch['img'][i],
                    full_frame=True,
                    imgname=imgname,
                    side_view=True,
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    return_rgba=True,
                )
                side_bgr = (side[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
                panels.append(resize_h(side_bgr))

            out_path = os.path.join(args.out_folder, f'{total_saved:04d}.jpg')
            cv2.imwrite(out_path, np.concatenate(panels, axis=1))
            total_saved += 1
            pbar.update(1)

    pbar.close()
    print(f'Saved {total_saved} images to {args.out_folder}/')


if __name__ == '__main__':
    main()
