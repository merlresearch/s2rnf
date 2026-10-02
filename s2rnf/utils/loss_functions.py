# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def rmse_nandetect(target, prediction, dim=-1, apply_mean=True, floor=None, freq_range=None, eps=1.0e-6):
    if floor is not None:
        target = torch.clamp(target, min=floor)
        prediction = torch.clamp(prediction, min=floor)

    if freq_range is not None:
        target = target[..., freq_range[0] : freq_range[1]]
        prediction = prediction[..., freq_range[0] : freq_range[1]]

    error = torch.nan_to_num(target - prediction, nan=0.0)
    rmse = torch.sqrt(torch.mean(torch.square(error), dim) + eps)

    if apply_mean:
        return torch.mean(rmse)

    else:
        return rmse
