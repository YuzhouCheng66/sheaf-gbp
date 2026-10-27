#!/bin/bash
# reviewer's CPU end-to-end smoke of g4_e3l.py (few updates; TRAIN smoke cache; local E3-R checkpoint / export copies; read only)
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill
P=D:/anaconda3/envs/jax-gpu/python.exe
RV=claude_handoff/analysis_20261027/g4/review_e3l/smoke
IN=claude_handoff/analysis_20261027/review_E3R/inputs
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
rm -rf $RV; mkdir -p $RV
date -u
$P -B scripts/claude_v342/g4_e3l.py train --cache claude_handoff/analysis_20261027/g4/e3l_smoke/cache --evidence "c:$IN/e1c_fit.json:A:cam_nf" --seed 342101 \
  --out $RV/s342101 --updates 6 --device cpu --threads 4 --batch 2 --warmup 2 --rate-every 2 --log-every 2 --val-every 6 --holdout-samples 2 \
  --ctrl-ckpt $IN/s342101/final.pt --ctrl-threshold $IN/s342101/threshold.json > $RV/train.log 2>&1; echo "train exit $?"
date -u
$P -B scripts/claude_v342/g4_e3l.py eval --smoke --only val_r1_i0_s1 --export $IN/export --out $RV/eval --ckpt $RV/s342101/checkpoints/final.pt --workers 1 --threads 4 > $RV/eval.log 2>&1; echo "eval exit $?"
date -u
$P -B scripts/claude_v342/g4_e3l.py aggregate --smoke --out $RV/eval --runs $RV/s342101 --control-summary claude_handoff/analysis_20261027/g4/e3r/summary.json > $RV/aggregate.log 2>&1; echo "aggregate exit $?"
date -u
