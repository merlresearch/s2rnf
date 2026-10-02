#!/bin/bash
# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -e
set -u
set -o pipefail

BASE_URL="davs://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET"
USERNAME="anonymous"
PATH_TO_SONICOM="./scratch/raw_sonicom_v2"

command -v duck >/dev/null 2>&1 || {
  echo "Error: duck is not installed." >&2
  exit 1
}

echo "Starting download loop for P0001 to P0360..."
mkdir -p "$PATH_TO_SONICOM"

for i in $(seq 1 360); do
    ID=$(printf "P%04d" $i)
    for mode in "Measured" "Synthetic"; do
        FILENAME="${ID}_HRIR_SONICOM_${mode}_Windowed_NoITD_Scaled.sofa"
        TARGET_URL="${BASE_URL}/${ID}/SYNTHETIC_HRTF/${FILENAME}"

        if duck --download "$TARGET_URL" "$PATH_TO_SONICOM" --username "$USERNAME" --quiet; then
            echo "Success: $FILENAME"
        else
            echo "Skipped: $FILENAME (File not found or connection error)"
        fi
    done
done

echo "Done."
