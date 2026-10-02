# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools

import lightning as L
import torch
import torch.optim as optim
from einops import rearrange, repeat

from s2rnf.model import neuralfield
from s2rnf.utils.loss_functions import rmse_nandetect


class HRTFFieldModule(L.LightningModule):
    DATASET_MEASURED = 0
    DATASET_SIMULATED = 1
    DATASET_NAMES = ("measured", "simulated")

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = getattr(neuralfield, config.model.name)(**config.model.config)
        self.loss_fn = self._define_loss()
        self.real_weight = getattr(self.hparams.loss, "real_weight", 1.0)
        self.simu_weight = getattr(self.hparams.loss, "simu_weight", 1.0)
        self.inner_ratio = getattr(self.hparams.loss, "inner_ratio", 0.5)
        self.outer_ratio = getattr(self.hparams.loss, "outer_ratio", 0.5)
        if self.inner_ratio + self.outer_ratio > 1.0:
            raise ValueError("inner_ratio and outer_ratio should sum up to 1.0 or less.")

    def _define_loss(self):
        floor = getattr(self.hparams.loss, "floor", None)
        freq_range = getattr(self.hparams.loss, "freq_range", None)

        return functools.partial(
            rmse_nandetect,
            floor=floor,
            freq_range=freq_range,
            apply_mean=False,
        )

    def _per_sample_loss(self, target_db, prediction):
        loss = self.loss_fn(target_db, prediction)
        loss = torch.mean(loss, -1)
        return loss

    def _log_split_losses(
        self,
        prefix,
        real_loss,
        sim_loss,
    ):
        self.log(
            f"{prefix}_loss",
            torch.mean(torch.cat([real_loss, sim_loss], dim=0)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(real_loss.numel() + sim_loss.numel()),
        )
        self.log(
            f"{prefix}_measured_loss",
            torch.mean(real_loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(real_loss.numel()),
        )
        self.log(
            f"{prefix}_simulated_loss",
            torch.mean(sim_loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(sim_loss.numel()),
        )

    def compute_subject_specific_params(self, sim_log_specs, locs, dataset_idx=1):
        batch_size = sim_log_specs.shape[0]
        subject_specific_params = torch.zeros(
            batch_size,
            self.model.latent_dim,
            device=locs.device,
            requires_grad=True,
        )

        with torch.enable_grad():
            prediction = self.forward(locs, subject_specific_params, dataset_idx=dataset_idx)
            inner_loss = self._per_sample_loss(sim_log_specs, prediction)
            inner_loss = torch.sum(torch.mean(inner_loss, dim=1))

            grad_z = torch.autograd.grad(inner_loss, [subject_specific_params], create_graph=True)[0]
            subject_specific_params = subject_specific_params - grad_z
        return subject_specific_params

    def forward(self, locs, subject_specific_params, dataset_idx):
        batch_size, num_directions = locs.shape[:2]

        locs = rearrange(locs, "b d c -> (b d) c")
        z_rep = repeat(subject_specific_params, "b h -> (b d) h", d=num_directions)
        dataset_idxs = torch.ones(batch_size * num_directions, device=locs.device, dtype=torch.long) * dataset_idx

        prediction = self.model(locs, z_rep, dataset_idxs)
        prediction = rearrange(prediction, "(b d) c f -> b d c f", b=batch_size, d=num_directions)
        return prediction

    def compute_loss(self, real_linear_specs, sim_linear_specs, locs, didxs_for_innerloop, didxs_for_outerloop):
        subject_specific_params = self.compute_subject_specific_params(
            20 * torch.log10(sim_linear_specs[:, didxs_for_innerloop, :, :]),
            locs[:, didxs_for_innerloop, :],
            dataset_idx=self.DATASET_SIMULATED,
        )
        real_prediction = self.forward(
            locs[:, didxs_for_outerloop, :],
            subject_specific_params,
            dataset_idx=self.DATASET_MEASURED,
        )
        sim_prediction = self.forward(
            locs[:, didxs_for_outerloop, :],
            subject_specific_params,
            dataset_idx=self.DATASET_SIMULATED,
        )

        real_target_db = 20 * torch.log10(real_linear_specs[:, didxs_for_outerloop, :])
        sim_target_db = 20 * torch.log10(sim_linear_specs[:, didxs_for_outerloop, :])

        real_loss = self._per_sample_loss(real_target_db, real_prediction)
        sim_loss = self._per_sample_loss(sim_target_db, sim_prediction)
        return real_loss, sim_loss

    def training_step(self, batch, batch_idx):
        _didxs = torch.randperm(batch[-1].shape[1], device=batch[-1].device)
        didxs_for_innerloop = _didxs[: int(len(_didxs) * self.inner_ratio)]
        didxs_for_outerloop = _didxs[
            int(len(_didxs) * self.inner_ratio) : int(len(_didxs) * (self.inner_ratio + self.outer_ratio))
        ]

        loss_real, loss_sim = self.compute_loss(*batch, didxs_for_innerloop, didxs_for_outerloop)
        loss_full = self.real_weight * torch.mean(loss_real) + self.simu_weight * torch.mean(loss_sim)

        self._log_split_losses("train", loss_real, loss_sim)
        return loss_full

    def validation_step(self, batch, batch_idx):
        didxs_full = torch.arange(batch[-1].shape[1], device=batch[-1].device)

        loss_real, loss_sim = self.compute_loss(*batch, didxs_full, didxs_full)
        loss_full = self.real_weight * torch.mean(loss_real) + self.simu_weight * torch.mean(loss_sim)

        self._log_split_losses("valid", loss_real, loss_sim)
        return loss_full

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.learning.optimizer.name)(
            self.parameters(), **self.hparams.learning.optimizer.config
        )

        if not hasattr(self.hparams.learning, "scheduler"):
            return {"optimizer": optimizer}

        scheduler = getattr(optim.lr_scheduler, self.hparams.learning.scheduler.name)(
            optimizer,
            **self.hparams.learning.scheduler.config,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": getattr(self.hparams.learning.scheduler, "interval", "epoch"),
            "frequency": 1,
            "monitor": "valid_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
