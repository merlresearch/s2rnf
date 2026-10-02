<!--
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# HRTF Personalization via Sim-to-Real Neural Field

This repository includes source code for training and evaluating on the SONICOM dataset the sim-to-real neural field (S2RNF) proposed in the following Interspeech submission:

    @InProceedings{Masuyama2026interspeech_s2rnf,
      author    =  {Masuyama, Yoshiki and Wichern, Gordon and Boeddeker, Christoph and Richter, Julius and Edo, Takahiro and Bhosale, Swapnil and {Le Roux}, Jonathan},
      title     =  {HRTF Personalization via Sim-to-Real Neural Field},
      booktitle =  {Proc. ISCA Interspeech},
      year      =  2026,
      month     =  sep
    }

## Table of contents

1. [Environment setup](#environment-setup)
2. [Downloading HRTFs](#downloading-hrtfs)
3. [Pretrained models](#pretrained-models)
4. [Training and evaluating S2RNF](#training-and-evaluating-s2rnf)
5. [Evaluating baseline methods](#evaluating-baseline-methods)
6. [Contributing](#contributing)
7. [Copyright and license](#copyright-and-license)

## Environment setup
- The code has been tested using `python 3.11.13` and `cuda 12.1` on Linux (x86_64).
- Populate a virtual environment using [uv](https://docs.astral.sh/uv/).
```
uv sync --frozen
uv pip install "git+https://github.com/Katarina-Poole/Spatial-Audio-Metrics.git@fc63fa0" --no-deps
```

## Downloading HRTFs
- The code currently supports training and evaluation on the [SONICOM dataset](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/).
- `download_sonicom.sh` downloads 200 pairs of measured and simulated HRTFs (i.e., the pairs for `P0001` to `P0327` because the pairs are not provided for every subject), which requires [Cyberduck CLI](https://duck.sh), and stores all pairs under `/path/to/sonicom`.
- For example, the data for the first subject will be `/path/to/sonicom/P0001_HRIR_SONICOM_Measured_Windowed_NoITD_Scaled.sofa` and `/path/to/sonicom/P0001_HRIR_SONICOM_Synthetic_Windowed_NoITD_Scaled.sofa`.

## Pretrained models
We include pre-trained checkpoints to reproduce Table 1 in the paper via Git LFS:
- `./exp-sonicom-200-paper/s2rnf/pretrain/official_best_measured.ckpt`
- `./exp-sonicom-200-paper/s2rnf_wo_simuloss/pretrain/official_best_measured.ckpt`

## Training and evaluating S2RNF
To evaluate the provided checkpoint, execute `run.sh` with the checkpoint path as an argument, which skips the following Stages 1 and 2.
To train and evaluate S2RNF from scratch, execute `run.sh` without the argument, which consists of three stages.

- **Stage 1:**
   - Preprocess the pairs of measured and simulated HRTFs.
   - With the default parameters, the preprocessed data will be stored under `/path/to/preprocessed/preprocessed_sonicom_v2`.

- **Stage 2:**
   - This stage trains the neural field, where the default config is set to `./exp-sonicom-200-paper/s2rnf/config.yaml`.
   - Before starting the training, change `npz_path` in the config file to your `/path/to/preprocessed`.
   - Checkpoints will be stored under `./exp-sonicom-200-paper/s2rnf/pretrain`, while TensorBoard logs will be written in `./exp-sonicom-200-paper/s2rnf/pretrain_tensorboard`.

- **Stage 3:**
   - This stage runs inference and evaluates the results.
   - The root mean squared error (RMSE) between log-magnitude spectra, also known as log-spectral distortion (LSD), and the mean absolute error (MAE) will be written to `./exp-sonicom-200-paper/s2rnf/pretrain/eval.log`.
   - We are not able to support the polar RMSE (PolRMSE in the paper) at this time, as it requires core MATLAB and additional toolboxes.

## Evaluating learning-free baseline methods
The following code will compute scores for the simulated HRTFs and the average baselines.
```
uv run python -m s2rnf.evaluation_baselines_sonicom \
   /path/to/sonicom/ \
   --split_config=exp-sonicom-200-paper/s2rnf/config.yaml
```


## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.


## Copyright and license
Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```
