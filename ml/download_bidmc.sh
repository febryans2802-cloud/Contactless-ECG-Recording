#!/usr/bin/env bash
# Download the BIDMC PPG and Respiration Dataset (PhysioNet, ODC-By 1.0 licence).
# Usage: ml/download_bidmc.sh [target_dir]   (default: ml/data)
set -euo pipefail
dir="${1:-$(dirname "$0")/data}"
mkdir -p "$dir"
for i in $(seq -w 1 53); do
  f="$dir/bidmc_${i}_Signals.csv"
  [ -s "$f" ] || curl -sSf -o "$f" "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_${i}_Signals.csv"
done
echo "53 records in $dir"
