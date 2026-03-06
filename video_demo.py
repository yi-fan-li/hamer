from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import shutil

# --- NEW IMPORTS ---
import json
from scipy.spatial.transform import Rotation as R
# --- END NEW IMPORTS ---

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional


# --- NEW HELPER FUNCTION TO SAVE MOCAP DATA ---
def save_mocap_data(output_folder: str, frame_name: str, person_id: int, mano_params: Dict, keypoints_3d: torch.Tensor, cam_t: np.ndarray):
    """
    Saves the MANO parameters and 3D keypoints in EasyMocap-compatible JSON format.
    """
    # Define output file paths
    smpl_path = os.path.join(output_folder, f'{frame_name}_{person_id}_smpl.json')
    keypoints_path = os.path.join(output_folder, f'{frame_name}_{person_id}.json')

    # --- 1. Process and save MANO Parameters (smpl format) ---

    # Convert global orientation from 3x3 matrix to 1x3 axis-angle (Rh)
    rot_matrix_rh = mano_params['global_orient'].cpu().numpy()
    rot_vec_rh = R.from_matrix(rot_matrix_rh).as_rotvec()

    # Convert hand pose from 15x3x3 matrices to 1x45 flattened axis-angle (poses)
    rot_matrices_hand = mano_params['hand_pose'].cpu().numpy() # Shape (15, 3, 3)
    rot_vecs_hand = R.from_matrix(rot_matrices_hand).as_rotvec() # Shape (15, 3)
    poses_flat = rot_vecs_hand.flatten().tolist()

    # Get shape parameters (betas)
    shapes = mano_params['betas'].cpu().numpy().flatten().tolist()

    # Get camera translation (Th)
    translation = cam_t.flatten().tolist()

    # Create the dictionary structure
    smpl_data = [{
        'id': person_id,
        'Rh': [rot_vec_rh.tolist()],
        'Th': [translation],
        'poses': [poses_flat],
        'shapes': [shapes]
    }]

    # Write the smpl JSON file
    with open(smpl_path, 'w') as f:
        json.dump(smpl_data, f, indent=4)

    # --- 2. Process and save 3D Keypoints ---

    # Get keypoints and convert to numpy array
    keypoints = keypoints_3d.cpu().numpy() # Shape (21, 3)

    # Add a confidence value of 1.0 to each keypoint to match the (X, Y, Z, conf) format
    keypoints_with_conf = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))]).tolist()

    # Create the dictionary structure
    keypoints_data = [{
        'id': person_id,
        'keypoints3d': keypoints_with_conf
    }]

    # Write the keypoints JSON file
    with open(keypoints_path, 'w') as f:
        json.dump(keypoints_data, f, indent=4)

# --- END NEW HELPER FUNCTION ---

def main():
    parser = argparse.ArgumentParser(description='HaMeR video demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_video_path', type=str, default='output_video.mp4', help='Path to output video file')
    parser.add_argument('--temp_frame_folder', type=str, default='_temp_input_frames', help='Temporary folder to store extracted input frames')
    parser.add_argument('--out_folder', type=str, default='out_video_demo', help='Output folder for processed frames before video reassembly')
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate for the output video. Also used for extracting frames from input video.')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory for processed frames
    processed_frames_folder = os.path.join(args.out_folder, 'processed_frames')
    os.makedirs(processed_frames_folder, exist_ok=True)
    os.makedirs(args.temp_frame_folder, exist_ok=True)

    # --- NEW: MAKE OUTPUT DIRECTORY FOR MOCAP JSONS ---
    mocap_output_folder = os.path.join(args.out_folder, 'mocap_output')
    os.makedirs(mocap_output_folder, exist_ok=True)
    # --- END NEW ---

    print(f"Extracting frames from {args.video_path} to {args.temp_frame_folder}...")
    # Use ffmpeg to extract frames
    ffmpeg_extract_command = f'ffmpeg -i {args.video_path} -r {args.frame_rate} {args.temp_frame_folder}/frame_%06d.png'
    os.system(ffmpeg_extract_command)
    print("Frame extraction complete.")

    img_paths = sorted(list(Path(args.temp_frame_folder).glob('*.png'))) # Assuming png for extracted frames

    # Iterate over all images (frames) in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])

                # --- NEW: SAVE MOCAP JSON FILES ---
                # Gather data for the current person
                person_mano_params = {k: v[n] for k, v in out['pred_mano_params'].items()}
                person_keypoints_3d = out['pred_keypoints_3d'][n]
                person_cam_t = pred_cam_t_full[n]

                # Call the helper function to save the data
                save_mocap_data(
                    output_folder=mocap_output_folder,
                    frame_name=img_fn,
                    person_id=person_id,
                    mano_params=person_mano_params,
                    keypoints_3d=person_keypoints_3d,
                    cam_t=person_cam_t
                )
                # --- END NEW ---

                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(processed_frames_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(processed_frames_folder, f'{img_fn}_{person_id}.obj')) # Save meshes to processed_frames_folder as well

        # Render full frame if enabled and hands were detected
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(processed_frames_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    # Reassemble frames into a video
    print(f"Reassembling frames from {processed_frames_folder} into {args.output_video_path}...")
    # Use ffmpeg to reassemble frames into a video
    ffmpeg_reassemble_command = f'ffmpeg -r {args.frame_rate} -i {processed_frames_folder}/frame_%06d_all.jpg -c:v libx264 -crf 1 -pix_fmt yuv420p {args.output_video_path}'
    print(ffmpeg_reassemble_command)
    os.system(ffmpeg_reassemble_command)
    print("Video reassembly complete.")

    # Clean up temporary frame folders
    # print(f"Cleaning up temporary input frame folder: {args.temp_frame_folder}")
    # shutil.rmtree(args.temp_frame_folder)
    # print(f"Cleaning up temporary processed frames folder: {processed_frames_folder}")
    # shutil.rmtree(processed_frames_folder)


if __name__ == '__main__':
    main()