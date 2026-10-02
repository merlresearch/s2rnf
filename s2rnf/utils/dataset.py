# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from s2rnf.utils.util import sonicom_subjects


class HRTFSim2RealDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        config,
        mode="train",
        nfft=412,
        fs=44100,
        epsilon=1.0e-4,
        return_itd=False,
    ):
        if return_itd:
            raise NotImplementedError("ITD prediction is not supported in the sim2real dataset.")

        self.dataset_name = config.dataset_name
        self.npz_path = Path(config.npz_path) / config.dataset_name
        self.mode = mode

        training_subjects = sonicom_subjects
        training_subjects = np.array(sorted(list(training_subjects)))
        exclude_subjects = set(getattr(config, "valid_subjects", [])) | set(getattr(config, "test_subjects", []))

        exclude_subjects = np.array(sorted(list(exclude_subjects)))

        if self.mode == "train":
            self.subjects = np.setdiff1d(training_subjects, exclude_subjects)

        elif self.mode == "valid":
            if hasattr(config, "valid_subjects"):
                valid_subjects = config.valid_subjects
            else:
                raise ValueError("valid_subjects must be specified.")

            self.subjects = np.sort(np.array(valid_subjects))

        elif self.mode == "test":
            self.subjects = np.sort(np.array(config.test_subjects))

        else:
            raise ValueError(f"mode {mode} is not supported.")

        self.nfft = nfft
        self.fs = fs
        self.epsilon = epsilon

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]

        npz = np.load(self.npz_path / f"P{subject_id:04}.npz")
        if npz["fs"] != self.fs or npz["nfft"] != self.nfft:
            raise ValueError("`fs` and `nfft` must be consistent with the config.")

        _floor = np.max(npz["real_linear_specs"].astype(np.float32)) * self.epsilon
        real_linear_specs = npz["real_linear_specs"].astype(np.float32) + _floor
        sim_linear_specs = npz["sim_linear_specs"].astype(np.float32) + _floor
        locs = npz["locs"].astype(np.float32)

        return real_linear_specs, sim_linear_specs, locs


class HRTFSim2RealDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config,
        learning_config,
        nfft=512,
        fs=48000,
        return_itd=False,
    ):
        super().__init__()
        if return_itd:
            raise NotImplementedError("ITD prediction is not supported in the sim2real datamodule.")

        self.train_dataset = HRTFSim2RealDataset(
            dataset_config,
            mode="train",
            nfft=nfft,
            fs=fs,
            return_itd=return_itd,
        )

        self.dev_dataset = HRTFSim2RealDataset(
            dataset_config,
            mode="valid",
            nfft=nfft,
            fs=fs,
            return_itd=return_itd,
        )

        self.learning_config = learning_config

    def train_dataloader(self):
        data_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.learning_config.num_workers,
            batch_size=self.learning_config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(
            dataset=self.dev_dataset,
            num_workers=self.learning_config.num_workers,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        return data_loader
