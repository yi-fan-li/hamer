"""Training script for the GCN joint refinement head on top of a frozen HaMeR model.

Usage:
    python train_gcn.py \
        exp_name=gcn_dinov3b \
        experiment=hamer_gcn \
        GCN.HAMER_CHECKPOINT=/path/to/hamer_dinov3b.ckpt \
        data=mix_all \
        trainer=gpu \
        launcher=local
"""
from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

from yacs.config import CfgNode
from hamer.configs import dataset_config
from hamer.datasets import HAMERDataModule
from hamer.models.hamer_gcn import HAMERWithGCN
from hamer.utils.pylogger import get_pylogger
from hamer.utils.misc import task_wrapper, log_hyperparameters

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    dataset_cfg = dataset_config(data_root=cfg.paths.get('training_data_dir', ''))
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    datamodule = HAMERDataModule(cfg, dataset_cfg)
    model = HAMERWithGCN(cfg, init_renderer=False)

    logger = TensorBoardLogger(
        os.path.join(cfg.paths.output_dir, 'tensorboard'),
        name='', version='', default_hp_metric=False,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=[logger],
        plugins=(
            SLURMEnvironment(requeue_signal=signal.SIGUSR2)
            if cfg.get('launcher', None) is not None
            else None
        ),
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    log_hyperparameters(object_dict)

    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
    log.info("Fitting done")


@hydra.main(
    version_base="1.2",
    config_path=str(root / "hamer/configs_hydra"),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
