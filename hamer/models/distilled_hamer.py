"""Teacher-student backbone distillation for HaMeR.

Teacher: ViT-H HaMeR (frozen).
Student: ViT-B backbone + MANOTransformerDecoderHead (trained from scratch).

The student is trained with two losses:
  L_total = L_task + DISTILL.LOSS_WEIGHT * L_distill

L_task  — standard HaMeR supervised loss (2D/3D keypoints + MANO params + adversarial).
L_distill — KL divergence between teacher and student post-softmax attention maps,
            averaged over 12 uniformly-spaced layer pairs and their 1-to-1 matched heads.

Head matching (1-to-1, computed once on the first training batch):
  For each layer pair, a bipartite Hungarian assignment maximises cosine similarity
  between student (12 heads) and teacher (16 heads) attention maps. Each student head
  is matched to exactly one teacher head; 4 teacher heads are unmatched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple

from scipy.optimize import linear_sum_assignment
from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from .hamer import HAMER
from .mano_wrapper import MANO

log = get_pylogger(__name__)

# Uniform layer pairing: student layer l → teacher layer round(l * 32/12),
# with the first and last layers forced to match exactly.
# Teacher has 32 layers (indices 0–31), student has 12 layers (indices 0–11).
LAYER_PAIRS: List[Tuple[int, int]] = [
    (0,  0),
    (1,  3),
    (2,  5),
    (3,  8),
    (4,  11),
    (5,  14),
    (6,  17),
    (7,  20),
    (8,  23),
    (9,  26),
    (10, 29),
    (11, 31),
]
_NUM_PAIRS = len(LAYER_PAIRS)  # 12
_STUDENT_LAYERS = [p[0] for p in LAYER_PAIRS]   # [0..11]
_TEACHER_LAYERS = [p[1] for p in LAYER_PAIRS]   # paired teacher layer indices


class DistilledHAMER(pl.LightningModule):
    """ViT-B student trained to mimic ViT-H teacher attention maps + HaMeR task loss."""

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # Teacher — frozen ViT-H HaMeR                                        #
        # ------------------------------------------------------------------ #
        from pathlib import Path
        from ..configs import get_config

        teacher_ckpt = cfg.DISTILL.TEACHER_CHECKPOINT
        log.info(f'Loading frozen teacher HaMeR from {teacher_ckpt}')
        teacher_cfg = get_config(
            str(Path(teacher_ckpt).parent.parent / 'model_config.yaml'),
            update_cachedir=True,
        )
        self.teacher = HAMER.load_from_checkpoint(
            teacher_ckpt, strict=False, init_renderer=False, cfg=teacher_cfg,
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        # Enable attention storage only on the 12 paired teacher layers.
        self.teacher.backbone.set_store_attn(_TEACHER_LAYERS)

        # ------------------------------------------------------------------ #
        # Student — ViT-B backbone + MANO head                               #
        # ------------------------------------------------------------------ #
        self.backbone = create_backbone(cfg)
        self.mano_head = build_mano_head(cfg)

        student_ckpt = cfg.DISTILL.get('STUDENT_CHECKPOINT', None)
        if student_ckpt:
            log.info(f'Initialising student from {student_ckpt}')
            student_init = HAMER.load_from_checkpoint(
                student_ckpt, strict=False, init_renderer=False, cfg=get_config(
                    str(Path(student_ckpt).parent.parent / 'model_config.yaml'),
                    update_cachedir=True,
                ),
            )
            self.backbone.load_state_dict(student_init.backbone.state_dict())
            self.mano_head.load_state_dict(student_init.mano_head.state_dict())
            del student_init
            log.info('Student backbone and MANO head loaded from checkpoint.')
        else:
            log.info('No STUDENT_CHECKPOINT provided; student MANO head is randomly initialised.')

        # Enable attention storage on all 12 student layers.
        self.backbone.set_store_attn(_STUDENT_LAYERS)

        # ------------------------------------------------------------------ #
        # Discriminator + losses (same as HAMER)                              #
        # ------------------------------------------------------------------ #
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()

        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # ------------------------------------------------------------------ #
        # Head-assignment buffers                                             #
        # head_assignments[i, s] = teacher head index for student head s      #
        # at layer pair i.  Initialised to -1; filled on the first batch.     #
        # ------------------------------------------------------------------ #
        self.register_buffer(
            'head_assignments',
            torch.full((_NUM_PAIRS, 12), -1, dtype=torch.long),
        )
        self.register_buffer('assignment_computed', torch.tensor(False))

        # Accumulators for similarity matrices used during head-assignment warmup.
        # Not checkpointed (Python lists); recomputed from scratch if training
        # resumes before assignment_computed becomes True.
        self._sim_accumulator: List[torch.Tensor] = []   # list of [S_H, T_H] per layer pair
        self._sim_accum_count: int = 0

        # ------------------------------------------------------------------ #
        # Renderers                                                           #
        # ------------------------------------------------------------------ #
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        self.automatic_optimization = False

    # ---------------------------------------------------------------------- #
    # Optimisers — student backbone + head only; teacher stays frozen         #
    # ---------------------------------------------------------------------- #

    def get_parameters(self):
        return list(self.backbone.parameters()) + list(self.mano_head.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.get_parameters()),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        optimizer_disc = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer, optimizer_disc

    # ---------------------------------------------------------------------- #
    # Head assignment                                                         #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _compute_sim_matrices(
        self,
        teacher_maps: List[torch.Tensor],
        student_maps: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Compute per-layer-pair cosine similarity matrix [S_H, T_H].

        Averages over the batch dimension so the result is cheap to accumulate.
        """
        sims = []
        for t_maps, s_maps in zip(teacher_maps, student_maps):
            B, T_H, N, _ = t_maps.shape
            _, S_H, _, _ = s_maps.shape
            t_flat = F.normalize(t_maps.reshape(B, T_H, N * N).float(), dim=-1)
            s_flat = F.normalize(s_maps.reshape(B, S_H, N * N).float(), dim=-1)
            sim = torch.bmm(s_flat, t_flat.transpose(1, 2)).mean(dim=0)  # [S_H, T_H]
            sims.append(sim)
        return sims

    def _assign_from_sim_matrices(self, sims: List[torch.Tensor]) -> None:
        """Run Hungarian on (averaged) similarity matrices and store the result."""
        log.info(
            f'Computing head assignment via Hungarian algorithm '
            f'(averaged over {self._sim_accum_count} batches = '
            f'{self._sim_accum_count * self.cfg.TRAIN.BATCH_SIZE} images)…'
        )
        for pair_i, sim in enumerate(sims):
            row_ind, col_ind = linear_sum_assignment(-sim.cpu().numpy())
            assignment = torch.zeros(sim.shape[0], dtype=torch.long)
            for s, t in zip(row_ind, col_ind):
                assignment[s] = t
            self.head_assignments[pair_i] = assignment.to(self.head_assignments.device)
        log.info('Head assignment complete.')

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """Run a forward step of the student network.

        When train=True, also runs the teacher to collect attention maps and
        adds 'loss_distill' and 'teacher_attn_maps' / 'student_attn_maps' to
        the output dict.
        """
        x = batch['img']
        batch_size = x.shape[0]
        x_cropped = x[:, :, :, 32:-32]

        # ---- Teacher (train only) ---------------------------------------- #
        if train:
            with torch.no_grad():
                self.teacher.eval()
                self.teacher.backbone(x_cropped)
            teacher_attn_maps = [
                m.detach()
                for m in self.teacher.backbone.get_attn_maps(_TEACHER_LAYERS)
            ]

        # ---- Student backbone -------------------------------------------- #
        conditioning_feats = self.backbone(x_cropped)

        if train:
            student_attn_maps = self.backbone.get_attn_maps(_STUDENT_LAYERS)

        # ---- Student MANO head ------------------------------------------- #
        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        output: Dict = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k, v in pred_mano_params.items()}

        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = (
            self.cfg.EXTRA.FOCAL_LENGTH
            * torch.ones(batch_size, 2, device=device, dtype=dtype)
        )
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0]
                / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(
            batch_size, -1, 3, 3
        )
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(
            batch_size, -1, 3, 3
        )
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(
            **{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False
        )
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
        )
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        if train:
            # Return raw maps so training_step can own the assignment warmup
            # and distillation loss computation.
            output['teacher_attn_maps'] = teacher_attn_maps
            output['student_attn_maps'] = student_attn_maps

        return output

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    # ---------------------------------------------------------------------- #
    # Distillation loss                                                       #
    # ---------------------------------------------------------------------- #

    def _compute_distill_loss(
        self,
        teacher_maps: List[torch.Tensor],
        student_maps: List[torch.Tensor],
    ) -> torch.Tensor:
        """KL(teacher ∥ student) averaged over layer pairs and matched head pairs.

        Args:
            teacher_maps: list of _NUM_PAIRS tensors [B, 16, N, N].
            student_maps: list of _NUM_PAIRS tensors [B, 12, N, N].

        Returns:
            Scalar loss tensor.
        """
        total_kl = teacher_maps[0].new_zeros(())
        for pair_i, (t_maps, s_maps) in enumerate(zip(teacher_maps, student_maps)):
            B, S_H, N, _ = s_maps.shape
            assignment = self.head_assignments[pair_i]  # [S_H]

            # Gather the matched teacher heads: [B, S_H, N, N]
            t_matched = t_maps[:, assignment, :, :]

            # Reshape to [B*S_H*N, N]: treat each query-token distribution
            # as an independent sample for batchmean reduction.
            t_flat = t_matched.reshape(B * S_H * N, N)
            s_flat = s_maps.reshape(B * S_H * N, N)

            # KL(teacher ∥ student): F.kl_div(log_Q, P) = P*(log P - log_Q)
            kl = F.kl_div(
                (s_flat + 1e-8).log(),
                t_flat,
                reduction='batchmean',
                log_target=False,
            )
            total_kl = total_kl + kl

        return total_kl / _NUM_PAIRS

    # ---------------------------------------------------------------------- #
    # Task loss (mirrors HAMER.compute_loss)                                  #
    # ---------------------------------------------------------------------- #

    def compute_task_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_mano_params['hand_pose'].shape[0]

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(
            pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0
        )

        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(
                pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt
            )

        loss = (
            self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d
            + self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d
            + sum(
                loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()]
                for k in loss_mano_params
            )
        )

        losses = dict(
            loss=loss.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach(),
            loss_keypoints_3d=loss_keypoints_3d.detach(),
        )
        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()
        output['losses'] = losses

        return loss

    # ---------------------------------------------------------------------- #
    # Discriminator step (mirrors HAMER.training_step_discriminator)          #
    # ---------------------------------------------------------------------- #

    def training_step_discriminator(
        self,
        batch: Dict,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        batch_size = hand_pose.shape[0]
        gt_rotmat = aa_to_rotmat(batch['hand_pose'].view(-1, 3)).view(
            batch_size, -1, 3, 3
        )
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, batch['betas'])
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    # ---------------------------------------------------------------------- #
    # Training step                                                           #
    # ---------------------------------------------------------------------- #

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        teacher_attn_maps = output.pop('teacher_attn_maps')
        student_attn_maps = output.pop('student_attn_maps')

        # ---- Head-assignment warmup ---------------------------------------- #
        # Accumulate cosine-similarity matrices over HEAD_ASSIGN_BATCHES batches
        # before running Hungarian. During warmup, the distillation loss is zero.
        n_warmup = int(self.cfg.DISTILL.get('HEAD_ASSIGN_BATCHES', 8))
        if not self.assignment_computed.item():
            batch_sims = self._compute_sim_matrices(teacher_attn_maps, student_attn_maps)
            if not self._sim_accumulator:
                self._sim_accumulator = batch_sims
            else:
                self._sim_accumulator = [
                    a + b for a, b in zip(self._sim_accumulator, batch_sims)
                ]
            self._sim_accum_count += 1

            if self._sim_accum_count >= n_warmup:
                avg_sims = [s / self._sim_accum_count for s in self._sim_accumulator]
                self._assign_from_sim_matrices(avg_sims)
                self.assignment_computed.fill_(True)
                self._sim_accumulator = []

            loss_distill = torch.zeros((), device=batch['img'].device)
        else:
            loss_distill = self._compute_distill_loss(teacher_attn_maps, student_attn_maps)

        # Task loss
        loss_task = self.compute_task_loss(batch, output)

        # Adversarial loss
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(
                pred_mano_params['hand_pose'].reshape(batch_size, -1),
                pred_mano_params['betas'].reshape(batch_size, -1),
            )
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss_task = loss_task + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        loss = loss_task + self.cfg.DISTILL.LOSS_WEIGHT * loss_distill

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                self.get_parameters(),
                self.cfg.TRAIN.GRAD_CLIP_VAL,
                error_if_nonfinite=True,
            )
            self.log('train/grad_norm', gn, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        optimizer.step()

        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(
                mocap_batch,
                pred_mano_params['hand_pose'].reshape(batch_size, -1),
                pred_mano_params['betas'].reshape(batch_size, -1),
                optimizer_disc,
            )
            output['losses']['loss_gen'] = loss_adv.detach()
            output['losses']['loss_disc'] = loss_disc

        # Log task and distillation losses separately (for λ tuning).
        output['losses']['loss_task'] = loss_task.detach()
        output['losses']['loss_distill'] = loss_distill.detach()

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self._tensorboard_log(batch, output, self.global_step, train=True)

        self.log(
            'train/loss',
            output['losses']['loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        return output

    # ---------------------------------------------------------------------- #
    # Validation step                                                         #
    # ---------------------------------------------------------------------- #

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        output = self.forward_step(batch, train=False)
        loss = self.compute_task_loss(batch, output)
        output['loss'] = loss
        self._tensorboard_log(batch, output, self.global_step, train=False)
        return output

    # ---------------------------------------------------------------------- #
    # Logging                                                                 #
    # ---------------------------------------------------------------------- #

    @pl.utilities.rank_zero.rank_zero_only
    def _tensorboard_log(
        self,
        batch: Dict,
        output: Dict,
        step_count: int,
        train: bool = True,
    ) -> None:
        mode = 'train' if train else 'val'
        losses = output.get('losses', {})

        if self.logger is not None:
            sw = self.logger.experiment
            for loss_name, val in losses.items():
                sw.add_scalar(f'{mode}/{loss_name}', val.detach().item(), step_count)

        if self.mesh_renderer is not None:
            batch_size = batch['keypoints_2d'].shape[0]
            num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
            images = batch['img']
            images = images * torch.tensor(
                [0.229, 0.224, 0.225], device=images.device
            ).reshape(1, 3, 1, 1)
            images = images + torch.tensor(
                [0.485, 0.456, 0.406], device=images.device
            ).reshape(1, 3, 1, 1)
            predictions = self.mesh_renderer.visualize_tensorboard(
                output['pred_vertices'][:num_images].detach().cpu().numpy(),
                output['pred_cam_t'][:num_images].detach().cpu().numpy(),
                images[:num_images].cpu().numpy(),
                output['pred_keypoints_2d'][:num_images].detach().cpu().numpy(),
                batch['keypoints_2d'][:num_images].cpu().numpy(),
                focal_length=output['focal_length'][:num_images].detach().cpu().numpy(),
            )
            if self.logger is not None:
                self.logger.experiment.add_image(
                    f'{mode}/predictions', predictions, step_count
                )
