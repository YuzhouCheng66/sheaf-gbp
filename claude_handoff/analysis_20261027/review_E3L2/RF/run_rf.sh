#!/bin/bash
# R-F diagnostic launcher (bigboy, GPU 0 only). Usage: run_rf.sh OUTDIR [extra args]
set -euo pipefail
B=$HOME/claude-sheaf-gbp
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$B/data/e3r-pyroot:$B/data/e3r-pyroot/exploration_grid
export HOUSEEXPO_JSON_DIR=$B/data/houseexpo-v365/json
export HOUSEEXPO_INDEX=$B/data/houseexpo-v365/houseexpo_index_v365.json
export TMPDIR=$B/tmp
export PYTHONDONTWRITEBYTECODE=1
OUT=$1; shift
mkdir -p "$OUT"
exec $HOME/miniconda3/envs/mlpsp/bin/python -B $B/diag_e3l2/rf_diag.py --out "$OUT" "$@"
