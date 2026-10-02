# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from spatialaudiometrics import hrtf_metrics as hf

from s2rnf.model.plmodel import HRTFFieldModule
from s2rnf.utils.dataset import HRTFSim2RealDataset
from s2rnf.utils.reconstruction import hrtf2hrir_minph
from s2rnf.utils.util import (
    FHIGH,
    FLOW,
    MAE_FRANGE,
    compute_mae,
    load_hrtf,
    pad_or_truncate_hrir,
    seed_everything,
    torch_reproducible,
    write_sofa,
)


def load_from_checkpoint(checkpoint_path: Path, config):
    if checkpoint_path.suffix == ".ckpt":
        plmodel = HRTFFieldModule.load_from_checkpoint(checkpoint_path)

    elif checkpoint_path.suffix == ".pt":
        state = torch.load(checkpoint_path, map_location="cpu")
        plmodel = HRTFFieldModule(config)
        plmodel.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")

    return plmodel


def compute_estimate(plmodel, test_dataset, itd, eps=1.0e-6, fs=48000):
    device = next(plmodel.parameters()).device
    real_linear_specs, sim_linear_specs, locs = (torch.as_tensor(x, device=device) for x in test_dataset[0])

    locs = locs.unsqueeze(0)
    sim_log_specs = 20 * torch.log10(sim_linear_specs).unsqueeze(0)
    subject_specific_params = plmodel.compute_subject_specific_params(
        sim_log_specs,
        locs,
        plmodel.DATASET_SIMULATED,
    )

    real_prediction_db = plmodel.forward(locs, subject_specific_params, plmodel.DATASET_MEASURED)
    real_prediction_db = torch.clamp(real_prediction_db, min=20 * np.log10(eps)) if eps > 0 else real_prediction_db

    pred = 10.0 ** (real_prediction_db[0].detach().to("cpu").numpy() / 20.0)
    nfft = (pred.shape[-1] - 1) * 2

    hrir_pred = hrtf2hrir_minph(pred, itd=itd[:, None], nfft=nfft, fs=fs)
    return hrir_pred


def evaluate_neural_field(args):
    config = OmegaConf.load(args.checkpoint_path.parent.parent.joinpath("config.yaml"))
    plmodel = load_from_checkpoint(args.checkpoint_path, config)
    plmodel.eval()
    plmodel.to("cuda" if torch.cuda.is_available() else "cpu")

    test_subjects = list(config.dataset.test_subjects)
    dataset_config = config.dataset

    lsds, maes = [], []
    for subject_id in test_subjects:
        gt_path = args.sonicom_path / f"P{subject_id:04}_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa"
        hrir_gt, _, itd_gt, _ = load_hrtf(gt_path, fs=args.fs, nfft=args.nfft)

        dataset_config.test_subjects = [subject_id]
        test_dataset = HRTFSim2RealDataset(
            dataset_config,
            mode="test",
            nfft=config.loss.nfft,
            fs=config.loss.sr,
        )

        hrir_pred = compute_estimate(
            plmodel,
            test_dataset,
            itd=itd_gt,
            eps=args.estimate_eps,
            fs=args.fs,
        )

        lsd = hf.calculate_lsd_across_locations(
            pad_or_truncate_hrir(hrir_gt, hrir_len=args.nfft),
            pad_or_truncate_hrir(hrir_pred, hrir_len=args.nfft),
            args.fs,
            args.flow,
            args.fhigh,
        )[0]
        lsds.append(lsd)

        mae_full = compute_mae(
            hrir_gt,
            hrir_pred,
            nfft=args.nfft,
            freq_range=np.arange(*MAE_FRANGE["SONICOM"]),
        )
        maes.append(mae_full)

        if args.dump_sofa:
            save_path = args.checkpoint_path.parent / f"P{subject_id:04}_HRIR_SONICOM_GT_ITD.sofa"
            write_sofa(
                hrir_pred,
                gt_path,
                save_path,
                fs=args.fs,
            )

    return float(np.mean(lsds)), float(np.mean(maes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sonicom_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--nfft", type=int, default=256)
    parser.add_argument("--flow", type=int, default=FLOW["full"])
    parser.add_argument("--fhigh", type=int, default=FHIGH["full"])
    parser.add_argument("--estimate_eps", type=float, default=0.0)
    parser.add_argument("--dump_sofa", action="store_true")
    args = parser.parse_args()
    seed_everything(0)
    torch_reproducible()

    lsd, mae = evaluate_neural_field(args)
    print(f"Mean LSD (sonicom test subjects): {lsd:.2f} dB")
    print(f"Mean MAE (sonicom test subjects): {mae:.2f} dB")

    log_name = args.checkpoint_path.parent.joinpath("eval.log")
    logging.basicConfig(filename=log_name, level=logging.INFO)
    logging.info(f"Checkpoint: {args.checkpoint_path.resolve()}")
    logging.info(f"LSD (dB): {lsd}")
    logging.info(f"MAE (dB): {mae}")


if __name__ == "__main__":
    main()
