# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from spatialaudiometrics import hrtf_metrics as hf

from s2rnf.utils.reconstruction import hrtf2hrir_minph
from s2rnf.utils.util import (
    FHIGH,
    FLOW,
    MAE_FRANGE,
    compute_mae,
    load_hrtf,
    pad_or_truncate_hrir,
    sonicom_subjects,
    write_sofa,
)


def compute_average_mag(original_dir, subjects, nfft, fs):
    spec_sum = 0
    for subject in subjects:
        sofa_path = original_dir / f"P{int(subject):04}_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa"
        spec = load_hrtf(sofa_path, fs=fs, nfft=nfft)[1]
        spec_sum += 20 * np.log10(spec)

    spec_avg = spec_sum / len(subjects)
    return 10 ** (spec_avg / 20)


def compute_metrics(args):
    config = OmegaConf.load(args.split_config).dataset
    training_subjects = sonicom_subjects
    exclude_subjects = set(getattr(config, "valid_subjects", [])) | set(getattr(config, "test_subjects", []))
    exclude_subjects = np.array(sorted(list(exclude_subjects)))
    training_subjects = np.setdiff1d(training_subjects, exclude_subjects)
    test_subjects = np.array(config.test_subjects)

    spec_avg = compute_average_mag(args.original_dir, training_subjects, args.nfft, args.fs)

    metrics = {
        "avg_rmse_full": [],
        "avg_mae_full": [],
        "sim_rmse_full": [],
        "sim_mae_full": [],
    }
    for subject in test_subjects:
        hrir_gt, _, itd_gt, _ = load_hrtf(
            args.original_dir / f"P{int(subject):04}_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa",
            nfft=args.nfft,
            fs=args.fs,
        )
        hrir_sim = load_hrtf(
            args.original_dir / f"P{int(subject):04}_HRIR_SONICOM_Synthetic_Windowed_NoITD_Scaled.sofa",
            nfft=args.nfft,
            fs=args.fs,
        )[0]

        for pred in ["sim", "avg"]:
            if pred == "sim":
                hrir_pred = hrir_sim
            else:
                hrir_pred = hrtf2hrir_minph(
                    spec_avg,
                    itd=itd_gt[..., None],
                    nfft=args.nfft,
                    fs=args.fs,
                )
                if args.dump_avg_hrir:
                    save_path = args.save_dir / f"P{int(subject):04}_HRIR_SONICOM_AVG_MAG_GT_ITD.sofa"
                    write_sofa(
                        hrir_pred,
                        args.original_dir / f"P{int(subject):04}_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa",
                        save_path,
                        fs=args.fs,
                    )

            lsd = hf.calculate_lsd_across_locations(
                pad_or_truncate_hrir(hrir_gt, hrir_len=args.nfft),
                pad_or_truncate_hrir(hrir_pred, hrir_len=args.nfft),
                fs=args.fs,
                flow=FLOW["full"],
                fhigh=FHIGH["full"],
            )[0]
            metrics[f"{pred}_rmse_full"].append(lsd)

            mae_full = compute_mae(
                hrir_gt,
                hrir_pred,
                nfft=args.nfft,
                freq_range=np.arange(*MAE_FRANGE["SONICOM"]),
            )
            metrics[f"{pred}_mae_full"].append(mae_full)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("--split_config", type=Path)
    parser.add_argument("--nfft", type=int, default=256)
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--dump_avg_hrir", action="store_true")
    parser.add_argument("--save_dir", type=Path, default=None)
    args = parser.parse_args()

    metrics = compute_metrics(args)

    for key, value in metrics.items():
        print(f"{key}: {np.mean(value):.2f}")


if __name__ == "__main__":
    main()
