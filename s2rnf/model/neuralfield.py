# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch
from torch import nn


def get_random_mat(output_feature, input_feature=4, scale=1.0):
    rng = np.random.default_rng(0)
    random_mat = scale * rng.normal(
        loc=0.0,
        scale=1.0,
        size=(output_feature, input_feature),
    )
    return random_mat


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.0,
        dropout_position="post",
        activation="Identity",
        bias=True,
    ):
        super().__init__()
        self.fc = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
        self.dropout_position = dropout_position

    def forward(self, x, u=0.0, v=0.0, b=0.0, peft="none"):
        if self.dropout_position == "pre":
            x = self.dropout(x)

        y = self.fc(x)

        if peft == "bitfit":
            y = y + b

        y = self.activation(y)
        if self.dropout_position == "post":
            return self.dropout(y)
        else:
            return y


class IGONSim2RealField(nn.Module):
    def __init__(
        self,
        hidden_features=128,
        hidden_layers=1,
        out_features=258,
        scale=1,
        dropout=0.1,
        dropout_position="post",
        num_datasets=2,
        activation="SiLU",
        peft="bitfit",
        subject_specific=[],
        dataset_specific=[],
        norm="Identity",
    ):
        super().__init__()
        _subject_specific = set(subject_specific)
        _dataset_specific = set(dataset_specific)
        if len(_subject_specific) != len(subject_specific):
            raise ValueError("Duplicate indices in subject-specific.")
        if len(_dataset_specific) != len(dataset_specific):
            raise ValueError("Duplicate indices in dataset-specific.")
        if not _subject_specific.isdisjoint(_dataset_specific):
            raise ValueError("Subject-specific and dataset-specific must be disjoint.")
        if (_subject_specific | _dataset_specific) - set(range(hidden_layers)):
            raise ValueError("Subject-specific and dataset-specific must be within the number of hidden layers.")

        if hidden_features % 2 != 0:
            raise ValueError("`hidden_features` must be even due to the design of random Fourier features.")

        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.latent_dim = len(subject_specific) * hidden_features
        self.peft = peft

        self.bmat = torch.nn.Parameter(
            torch.tensor(get_random_mat(hidden_features // 2, scale=scale), dtype=torch.float32),
            requires_grad=False,
        )

        self.in_linear = MLP(
            hidden_features,
            hidden_features,
            dropout=0.0,
            dropout_position=dropout_position,
            activation=activation,
        )
        self.hidden_mlps = nn.ModuleList(
            [
                MLP(
                    hidden_features,
                    hidden_features,
                    dropout=dropout,
                    dropout_position=dropout_position,
                    activation=activation,
                )
                for _ in range(hidden_layers)
            ]
        )
        self.norm = nn.ModuleList([getattr(nn, norm)(hidden_features) for _ in range(hidden_layers // 2)])
        self.out_linear = MLP(
            hidden_features,
            out_features,
            dropout=0.0,
            dropout_position=dropout_position,
            activation="Identity",
        )

        self.peft_indicators = [("none", None) for _ in range(hidden_layers)]

        for m, n in enumerate(subject_specific):
            self.peft_indicators[n] = ("subject", m)

        for m, n in enumerate(dataset_specific):
            self.peft_indicators[n] = ("dataset", m)

        if peft == "bitfit":
            self.dataset_bitfits = nn.ModuleList(
                [nn.Embedding(num_datasets, hidden_features) for n in range(len(dataset_specific))]
            )
            for emb in self.dataset_bitfits:
                nn.init.uniform_(emb.weight, -1.0 / np.sqrt(num_datasets), 1.0 / np.sqrt(num_datasets))

        else:
            raise NotImplementedError(f"PEFT method {peft} is not supported.")

    def _get_rff(self, locs):
        azimuth, elevation = locs[:, 0], locs[:, 1]
        emb = [azimuth.sin(), azimuth.cos(), elevation.sin(), elevation.cos()]
        emb = torch.stack(emb, -1) @ self.bmat.T
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

    def forward(self, locs, subject_latents, dataset_idxs):
        emb = self._get_rff(locs)
        x = self.in_linear(emb)
        residual = x
        subject_latents = torch.split(subject_latents, self.hidden_features, dim=-1)

        for n in range(self.hidden_layers):
            if n % 2 == 0:
                x = self.norm[n // 2](x)

            peft_type, peft_idx = self.peft_indicators[n]
            if peft_type == "dataset":
                peft = self.peft
                b = self.dataset_bitfits[peft_idx](dataset_idxs)
            elif peft_type == "subject":
                peft = self.peft
                b = subject_latents[peft_idx]
            else:
                peft = "none"
                b = 0.0

            x = self.hidden_mlps[n](x, b=b, peft=peft)

            if n % 2 == 1:
                x = x + residual
                residual = x

        x = self.out_linear(x).reshape(locs.shape[0], 2, -1)
        return x
