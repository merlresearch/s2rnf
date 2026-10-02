#!/bin/bash
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -e
set -u
set -o pipefail

PATH_TO_SONICOM=./scratch/raw_sonicom_v2
PATH_TO_PREPROCESSED=./scratch/preprocessed_sonicom_v2
EXP_DIR=./exp-sonicom-200-paper
EXP_CONFIG=s2rnf

# NOTE: If a checkpoint path is given, this script will skip training and evaluate the provided checkpoint.
CHECKPOINT=${1:-}

if [[ -z "$CHECKPOINT" ]]; then
    echo "Stage 1: Preprocess the SONICOM dataset"
    uv run python -m s2rnf.preprocess.dump_npz \
        sonicom \
        "$PATH_TO_SONICOM" \
        "$PATH_TO_PREPROCESSED"

    echo "Stage 2: Train S2RNF"
    uv run python -m s2rnf.pretrain \
        "${EXP_DIR}/${EXP_CONFIG}"

    CHECKPOINT="${EXP_DIR}/${EXP_CONFIG}/pretrain/best_measured.ckpt"
fi

echo "Stage 3: Evaluate S2RNF"
uv run python -m s2rnf.evaluation_original_itd_sonicom \
    "$PATH_TO_SONICOM" \
    "$CHECKPOINT"
