# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from s2rnf.model.plmodel import HRTFFieldModule
from s2rnf.utils.dataset import HRTFSim2RealDataModule
from s2rnf.utils.util import seed_everything, torch_reproducible


def train(dm, plmodel, config, path):
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=path.joinpath("pretrain"),
        filename="best_{epoch:02d}",
        save_top_k=1,
        enable_version_counter=False,
    )
    checkpoint_callback_measured = ModelCheckpoint(
        monitor="valid_measured_loss",
        mode="min",
        dirpath=path.joinpath("pretrain"),
        filename="best_measured",
        save_top_k=1,
        enable_version_counter=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",
        min_delta=0.0,
        patience=config.learning.patience,
    )
    callbacks = [
        checkpoint_callback,
        checkpoint_callback_measured,
        early_stop_callback,
        LearningRateMonitor(),
        RichProgressBar(),
    ]

    logger = TensorBoardLogger(
        path.joinpath("pretrain_tensorboard"),
        name="",
        version="",
        default_hp_metric=False,
    )
    trainer = Trainer(
        max_epochs=config.learning.num_epoch,
        accelerator="auto",
        devices=1,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        gradient_clip_val=config.learning.clip,
    )

    trainer.fit(plmodel, dm)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()
    seed_everything(0)
    torch_reproducible()

    path = args.config_path
    path.joinpath("pretrain").mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(path.joinpath("config.yaml"))

    dm = HRTFSim2RealDataModule(
        config.dataset,
        config.learning,
        nfft=config.loss.nfft,
        return_itd=False,
        fs=config.loss.sr,
    )

    if hasattr(config, "embmse_weight"):
        raise NotImplementedError("embmse_weight is not implemented yet.")
    else:
        plmodel = HRTFFieldModule(config)

    pretraining_checkpoint = getattr(config, "pretraining_checkpoint", None)
    if pretraining_checkpoint is not None:
        plmodel = HRTFFieldModule.load_from_checkpoint(pretraining_checkpoint, config=config)

    train(dm, plmodel, config, path)


if __name__ == "__main__":
    main()
