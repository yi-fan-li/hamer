import torch
import pytorch_lightning as pl
from typing import Any, Dict, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import perspective_projection
from ..utils.pylogger import get_pylogger
from .hamer import HAMER
from .losses import Keypoint3DLoss, Keypoint2DLoss
from .gcn_refinement import GCNRefinementHead, rot_mat_to_6d, six_d_to_rot_mat

log = get_pylogger(__name__)

# Maps MANO kinematic joint index (0-15) → OpenPose joint index in mano_output.joints.
# MANO wrapper reorders joints to OpenPose format, so we need this to recover the
# 16 kinematic joint 3D positions in the correct order for feature sampling.
#   MANO kin:  0=wrist, 1-3=index, 4-6=middle, 7-9=little, 10-12=ring, 13-15=thumb
#   OpenPose:  0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=little
_MANO_KIN_TO_OPENPOSE = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]


class HAMERWithGCN(pl.LightningModule):
    """
    Frozen HaMeR + trainable GCN pose refinement head.

    The GCN refines the 16 MANO joint rotations (6D representation) predicted
    by the frozen HaMeR head. Refined rotations are converted back to rotation
    matrices via Gram-Schmidt and passed through the MANO layer together with
    the original shape betas to recover a refined mesh and keypoints.

    Only GCNRefinementHead parameters are updated during training.
    The HAMER backbone + MANO head are fully frozen regardless of backbone type.
    """

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg

        # --- Load and freeze HaMeR ---
        hamer_ckpt = cfg.GCN.HAMER_CHECKPOINT
        log.info(f'Loading frozen HaMeR from {hamer_ckpt}')
        self.hamer = HAMER.load_from_checkpoint(
            hamer_ckpt, strict=False, init_renderer=False,
        )
        for param in self.hamer.parameters():
            param.requires_grad = False
        self.hamer.eval()

        # Convenience aliases for shared HAMER sub-objects
        self.mano = self.hamer.mano
        self.hamer_cfg = self.hamer.cfg

        # --- GCN refinement head ---
        self.gcn_head = GCNRefinementHead(
            feat_channels=cfg.GCN.FEAT_CHANNELS,
            hidden_dim=cfg.GCN.HIDDEN_DIM,
            num_layers=cfg.GCN.NUM_LAYERS,
        )

        # --- Losses (same types as HaMeR) ---
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')

        # --- Renderers ---
        if init_renderer:
            self.renderer = SkeletonRenderer(self.hamer_cfg)
            self.mesh_renderer = MeshRenderer(self.hamer_cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        x = batch['img']
        batch_size = x.shape[0]

        # --- Frozen HaMeR forward ---
        with torch.no_grad():
            self.hamer.eval()
            feat_map = self.hamer.backbone(x[:, :, :, 32:-32])
            pred_mano_params, pred_cam, _ = self.hamer.mano_head(feat_map)

            device = pred_mano_params['hand_pose'].device
            dtype  = pred_mano_params['hand_pose'].dtype

            focal_length = self.hamer_cfg.EXTRA.FOCAL_LENGTH * torch.ones(
                batch_size, 2, device=device, dtype=dtype,
            )
            pred_cam_t = torch.stack([
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (
                    self.hamer_cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9
                ),
            ], dim=-1)

            pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
            pred_mano_params['hand_pose']     = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
            pred_mano_params['betas']         = pred_mano_params['betas'].reshape(batch_size, -1)

            # Coarse MANO forward (needed for joint positions used in feature sampling)
            coarse_mano_output = self.mano(
                **{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False,
            )
            coarse_joints_3d = coarse_mano_output.joints    # [B, 21, 3]  OpenPose order
            coarse_vertices  = coarse_mano_output.vertices  # [B, 778, 3]

            # Extract 16 kinematic joint positions in MANO kinematic order
            # (needed for 2D projection → GCN feature sampling)
            kin_joints_3d = coarse_joints_3d[:, _MANO_KIN_TO_OPENPOSE, :]  # [B, 16, 3]

            # Project 16 kinematic joints to 2D for feature sampling grid
            kin_joints_2d = perspective_projection(
                kin_joints_3d,
                translation=pred_cam_t,
                focal_length=focal_length / self.hamer_cfg.MODEL.IMAGE_SIZE,
            )  # [B, 16, 2]

            # Stack global_orient + hand_pose → [B, 16, 3, 3], convert to 6D
            pose_mat = torch.cat(
                [pred_mano_params['global_orient'], pred_mano_params['hand_pose']], dim=1,
            )  # [B, 16, 3, 3]
            pose_6d = rot_mat_to_6d(pose_mat)  # [B, 16, 6]

        # --- GCN refinement (trainable) ---
        refined_pose_6d = self.gcn_head(pose_6d, feat_map, kin_joints_2d)  # [B, 16, 6]

        # Convert refined 6D back to rotation matrices via Gram-Schmidt
        refined_pose_mat = six_d_to_rot_mat(refined_pose_6d)  # [B, 16, 3, 3]

        refined_mano_params = {
            'global_orient': refined_pose_mat[:, :1],   # [B, 1, 3, 3]
            'hand_pose':     refined_pose_mat[:, 1:],   # [B, 15, 3, 3]
            'betas':         pred_mano_params['betas'],
        }

        # MANO forward with refined pose + original shape
        refined_mano_output = self.mano(
            **{k: v.float() for k, v in refined_mano_params.items()}, pose2rot=False,
        )
        refined_joints_3d = refined_mano_output.joints    # [B, 21, 3]
        refined_vertices  = refined_mano_output.vertices  # [B, 778, 3]

        # Reproject refined joints for 2D loss
        refined_keypoints_2d = perspective_projection(
            refined_joints_3d,
            translation=pred_cam_t,
            focal_length=focal_length / self.hamer_cfg.MODEL.IMAGE_SIZE,
        )  # [B, 21, 2]

        output = {
            'pred_cam':           pred_cam,
            'pred_mano_params':   {k: v.clone() for k, v in pred_mano_params.items()},
            'pred_cam_t':         pred_cam_t,
            'focal_length':       focal_length,
            # refined outputs used for loss + eval
            'pred_vertices':      refined_vertices.reshape(batch_size, -1, 3),
            'pred_keypoints_3d':  refined_joints_3d.reshape(batch_size, -1, 3),
            'pred_keypoints_2d':  refined_keypoints_2d.reshape(batch_size, -1, 2),
            # coarse outputs kept for debugging / logging
            'coarse_keypoints_3d': coarse_joints_3d.reshape(batch_size, -1, 3),
            'coarse_vertices':     coarse_vertices.reshape(batch_size, -1, 3),
        }
        return output

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']

        loss_keypoints_3d = self.keypoint_3d_loss(
            pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0,
        )
        loss_keypoints_2d = self.keypoint_2d_loss(
            pred_keypoints_2d, gt_keypoints_2d,
        )

        loss = (
            self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d
            + self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d
        )

        output['losses'] = dict(
            loss=loss.detach(),
            loss_keypoints_3d=loss_keypoints_3d.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach(),
        )
        return loss

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------

    def training_step(self, joint_batch: Dict, batch_idx: int) -> torch.Tensor:
        batch = joint_batch['img']
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output)

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self._tensorboard_log(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'],
                 on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output)
        output['loss'] = loss
        self._tensorboard_log(batch, output, self.global_step, train=False)
        return output

    # ------------------------------------------------------------------
    # Optimiser — GCN head only
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.gcn_head.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @pl.utilities.rank_zero.rank_zero_only
    def _tensorboard_log(
        self, batch: Dict, output: Dict, step_count: int, train: bool = True,
    ) -> None:
        mode = 'train' if train else 'val'
        losses = output['losses']

        if self.logger is not None:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(
                    f'{mode}/{loss_name}', val.detach().item(), step_count,
                )

        if self.mesh_renderer is not None:
            batch_size = batch['keypoints_2d'].shape[0]
            num_images = min(batch_size, self.hamer_cfg.EXTRA.NUM_LOG_IMAGES)

            images = batch['img']
            images = images * torch.tensor(
                [0.229, 0.224, 0.225], device=images.device,
            ).reshape(1, 3, 1, 1)
            images = images + torch.tensor(
                [0.485, 0.456, 0.406], device=images.device,
            ).reshape(1, 3, 1, 1)

            pred_vertices    = output['pred_vertices'].detach()
            pred_cam_t       = output['pred_cam_t'].detach()
            focal_length     = output['focal_length'].detach()
            pred_keypoints_2d = output['pred_keypoints_2d'].detach()
            gt_keypoints_2d  = batch['keypoints_2d']

            predictions = self.mesh_renderer.visualize_tensorboard(
                pred_vertices[:num_images].cpu().numpy(),
                pred_cam_t[:num_images].cpu().numpy(),
                images[:num_images].cpu().numpy(),
                pred_keypoints_2d[:num_images].cpu().numpy(),
                gt_keypoints_2d[:num_images].cpu().numpy(),
                focal_length=focal_length[:num_images].cpu().numpy(),
            )
            if self.logger is not None:
                self.logger.experiment.add_image(
                    f'{mode}/predictions', predictions, step_count,
                )
