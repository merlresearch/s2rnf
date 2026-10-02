# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from s2rnf.utils.util import load_hrtf, sonicom_subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, choices=("sonicom",))
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("--nfft", type=int, default=None)
    parser.add_argument("--fs", type=int, default=None)
    args = parser.parse_args()

    subjects = sonicom_subjects
    nfft = 256 if args.nfft is None else args.nfft
    fs = 48000 if args.fs is None else args.fs
    real_template = "P{subject:04}_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa"
    sim_template = "P{subject:04}_HRIR_SONICOM_Synthetic_Windowed_NoITD_Scaled.sofa"
    out_template = "P{subject:04}.npz"

    output_dir = args.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in tqdm(subjects):
        real_path = args.original_dir / real_template.format(subject=int(subject_id))
        sim_path = args.original_dir / sim_template.format(subject=int(subject_id))
        _, real_linear_specs, _, locs = load_hrtf(real_path, fs=fs, nfft=nfft)
        _, sim_linear_specs, _, sim_locs = load_hrtf(sim_path, fs=fs, nfft=nfft)

        if not np.allclose(locs, sim_locs):
            raise ValueError(f"Location mismatch between real and simulated HRTFs for subject {subject_id}.")

        np.savez(
            output_dir / out_template.format(subject=subject_id),
            real_linear_specs=real_linear_specs,
            sim_linear_specs=sim_linear_specs,
            locs=locs,
            fs=fs,
            nfft=nfft,
        )


if __name__ == "__main__":
    main()
