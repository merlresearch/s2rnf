# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy.signal as sn
from scipy.fft import ifft
from spatialaudiometrics.hrtf_metrics import itd_estimator_maxiacce


def calculate_itd(hrir, fs=48000, upper_cut_freq=3000, filter_order=10):
    _, itd_samps, _ = itd_estimator_maxiacce(
        hrir[None, ...], np.array(fs), upper_cut_freq=upper_cut_freq, filter_order=filter_order
    )
    return np.array(itd_samps)[0, ...]


def hrtf2hrir_minph(mag, itd, nfft=None, fs=48000, itd_search_width=5):
    _, _, nfreqs = mag.shape

    if nfft is None:
        nfft = 2 * (nfreqs - 1)
    else:
        assert nfft == 2 * (nfreqs - 1)

    mag = mag.astype(np.float64)
    mag = np.concatenate([mag, np.flip(mag[..., 1:-1], -1)], -1)
    phase = -np.imag(sn.hilbert(np.log(mag), axis=-1))
    hrir = ifft(mag * np.exp(1j * phase), n=nfft, axis=-1).real

    for d, itd_pred in enumerate(itd[:, 0]):
        itd_candidates, itd_errors = [], []
        for shift in range(-itd_search_width, itd_search_width + 1):

            itd_candidate = int(np.round(itd_pred + shift))
            itd_candidates.append(itd_candidate)
            if itd_candidate < 0:
                hl = hrir[d, 0, :]
                hr = np.roll(hrir[d, 1, :], -itd_candidate)
            else:
                hl = np.roll(hrir[d, 0, :], itd_candidate)
                hr = hrir[d, 1, :]

            itd_errors.append(calculate_itd(np.stack([hl, hr], 0), fs=fs) - itd_pred)

        itd_optimal = itd_candidates[np.argmin(np.abs(np.array(itd_errors)))]

        if itd_optimal < 0:
            hrir[d, 1, :] = np.roll(hrir[d, 1, :], -itd_optimal)

        else:
            hrir[d, 0, :] = np.roll(hrir[d, 0, :], itd_optimal)

    return hrir
