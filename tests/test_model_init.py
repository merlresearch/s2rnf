# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from s2rnf.model.neuralfield import IGONSim2RealField


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("nfft", [256, 442])
@pytest.mark.parametrize("hidden_features", [16, 64])
@pytest.mark.parametrize("hidden_layers", [4, 6])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("activation", ["GELU", "SiLU"])
@pytest.mark.parametrize("subject_specific", [[0], [0, 1]])
@pytest.mark.parametrize("norm", ["Identity", "LayerNorm"])
def test_peft_neural_field(
    nfft,
    batch_size,
    hidden_features,
    hidden_layers,
    scale,
    dropout,
    activation,
    subject_specific,
    norm,
):
    out_features = (nfft // 2 + 1) * 2
    dataset_specific = [x for x in range(hidden_layers) if x not in subject_specific]
    model = IGONSim2RealField(
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        scale=scale,
        dropout=dropout,
        activation=activation,
        peft="bitfit",
        subject_specific=subject_specific,
        dataset_specific=dataset_specific,
        norm=norm,
    )

    model.train()
    locs = torch.rand(batch_size, 2)
    subject_latents = torch.rand(batch_size, model.latent_dim)
    dataset_idxs = torch.randint(low=0, high=2, size=(batch_size,))

    pred = model(locs, subject_latents, dataset_idxs)
    assert list(pred.shape) == [batch_size, 2, nfft // 2 + 1]
