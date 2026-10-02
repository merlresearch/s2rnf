# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import sofa
import torch
from scipy.fft import fft
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import load_data as ld

FLOW = {"full": 20, "mid": 5000}
FHIGH = {"full": 20000, "mid": 12000}
MAE_FRANGE = {"HUTUBS": (1, 201), "SONICOM": (1, 107)}
LOG_EPS = 1.0e-4


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def torch_reproducible():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_hrtf(fname: Path, fs: int = 48000, nfft: int = None):
    hrtf = ld.HRTF(fname)
    if hrtf.fs != fs:
        hrir = librosa.resample(hrtf.hrir, orig_sr=hrtf.fs, target_sr=fs, axis=-1)
    else:
        hrir = hrtf.hrir

    if nfft is None:
        nfft = hrir.shape[-1]

    complex_specs = fft(hrir, n=nfft, axis=-1)[..., : nfft // 2 + 1]
    specs = np.abs(complex_specs)
    _, itds, _ = hf.itd_estimator_maxiacce(hrir, hrtf.fs)
    itds = np.array(itds)

    HRTF = sofa.Database.open(fname)
    locs = HRTF.Source.Position.get_values(system="spherical")
    HRTF.close()

    locs = np.deg2rad(locs)
    locs[:, 0] -= np.pi

    return hrtf.hrir, specs, itds, locs


def pad_or_truncate_hrir(hrir, hrir_len):
    if hrir.shape[-1] < hrir_len:
        pad_width = [(0, 0) for _ in range(hrir.ndim - 1)]
        pad_width.append((0, hrir_len - hrir.shape[-1]))
        hrir = np.pad(hrir, pad_width)
    else:
        hrir = hrir[..., :hrir_len]
    return hrir


def write_sofa(avg_hrir, template_path, output_path, fs, suffix=""):
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        avg_path = output_path / f"avg_hrir_measured_{suffix}.sofa"
    else:
        avg_path = output_path

    # NOTE: Reuse a measured template SOFA to preserve metadata/layout.
    shutil.copyfile(template_path, avg_path)
    HRTF = sofa.Database.open(avg_path, mode="r+")
    HRTF.Data.SamplingRate.set_values(fs)
    HRTF.Data.Delay.set_values(np.zeros_like(HRTF.Data.Delay.get_values()))
    target_n = HRTF.Dimensions.dataset.dimensions["N"].size
    if avg_hrir.shape[-1] < target_n:
        avg_hrir = np.pad(avg_hrir, ((0, 0), (0, 0), (0, target_n - avg_hrir.shape[-1])))
    else:
        avg_hrir = avg_hrir[..., :target_n]
    HRTF.Data.IR.set_values(avg_hrir)
    HRTF.close()
    return avg_path


def compute_mae(hrir_gt, hrir_pred, nfft, freq_range):
    mag_gt = np.abs(fft(hrir_gt[..., :nfft], n=nfft, axis=-1))[..., : nfft // 2 + 1]
    _floor = np.max(mag_gt[..., freq_range]) * LOG_EPS

    mag_gt = 20 * np.log10(mag_gt[..., freq_range] + _floor)

    mag_pred = np.abs(fft(hrir_pred[..., :nfft], n=nfft, axis=-1))[..., : nfft // 2 + 1]
    mag_pred = 20 * np.log10(mag_pred[..., freq_range] + _floor)

    mae = np.mean(np.abs(mag_gt - mag_pred))
    return mae


def to_cartesian(x):
    if x.ndim == 1:
        x = x[None, :]
        ndim = 1
    else:
        ndim = 2

    y = np.stack([np.cos(x[:, 0]) * np.cos(x[:, 1]), np.sin(x[:, 0]) * np.cos(x[:, 1]), np.sin(x[:, 1])], -1)
    y *= x[:, 2, None]

    if ndim == 1:
        return y[0, :]
    else:
        return y


sonicom_subjects = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        14,
        15,
        16,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        30,
        31,
        32,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        50,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        61,
        62,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        74,
        75,
        76,
        77,
        79,
        80,
        81,
        82,
        83,
        86,
        88,
        89,
        92,
        95,
        97,
        98,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        109,
        110,
        111,
        112,
        113,
        115,
        116,
        117,
        118,
        119,
        121,
        122,
        123,
        124,
        125,
        130,
        131,
        132,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        160,
        161,
        162,
        164,
        169,
        170,
        171,
        172,
        176,
        181,
        182,
        187,
        188,
        191,
        192,
        198,
        199,
        200,
        202,
        204,
        205,
        206,
        207,
        208,
        211,
        213,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        228,
        229,
        230,
        231,
        233,
        235,
        236,
        237,
        239,
        240,
        243,
        244,
        250,
        256,
        257,
        259,
        261,
        266,
        279,
        280,
        282,
        285,
        290,
        291,
        295,
        297,
        298,
        301,
        302,
        304,
        306,
        307,
        308,
        309,
        311,
        315,
        316,
        319,
        320,
        321,
        323,
        324,
        325,
        326,
        327,
    ]
)


if __name__ == "__main__":
    print(f"SONICOM SUBJECTS: {len(sonicom_subjects)}")
