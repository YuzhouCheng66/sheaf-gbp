#!/bin/bash
# reviewer's end-to-end CPU smoke of g4_e3l2.py (TRAIN houses only; E3-L smoke cache; existing TRAIN export of the implementer's smoke)
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill || exit 1
export CUDA_VISIBLE_DEVICES=
P=D:/anaconda3/envs/jax-gpu/python.exe
S=scripts/claude_v342/g4_e3l2.py
G4=claude_handoff/analysis_20261027/g4
O=$G4/review_e3l2/smoke
CACHE=$G4/e3l_smoke/cache
E3R=claude_handoff/analysis_20261027/review_E3R/inputs
FIT=C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill/claude_handoff/analysis_20261027/review_E3R/inputs/e1c_fit.json
MAN=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
t(){ echo "$(date -u +%FT%TZ) $*"; }
t start
$P -B $S train --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/s342101 --updates 6 --device cpu --threads 2 --batch 2 --log-every 2 \
  --val-every 6 --holdout-samples 2 --warmup 2 --rate-every 2 --time-from 2 --snap 4 --manifest-sha $MAN --ctrl-ckpt $E3R/s342101/final.pt > $O/train.log 2>&1; t "train exit $?"
$P -B $S ctrlprep --cache $CACHE --e3r $E3R --out $O/ctrl_holdout.json --smoke --workers 2 --threads 1 > $O/ctrlprep.log 2>&1; t "ctrlprep exit $?"
$P -B $S kselect --run $O/runs/s342101 --ctrl $O/ctrl_holdout.json --smoke --workers 1 --threads 2 > $O/kselect.log 2>&1; t "kselect exit $?"
$P -B $S select --run $O/runs/s342101 --ctrl $O/ctrl_holdout.json --smoke --kset all --workers 1 --threads 2 > $O/select.log 2>&1; t "select exit $?"
$P -B $S kselect --run $O/runs/s342101 --ctrl $O/ctrl_holdout.json --name upd4 --smoke --workers 1 --threads 2 > $O/kselect_upd4.log 2>&1; t "kselect upd4 exit $?"
$P -B $S select --run $O/runs/s342101 --ctrl $O/ctrl_holdout.json --name upd4 --smoke --kset kstar --workers 1 --threads 2 > $O/select_upd4.log 2>&1; t "select upd4 exit $?"
$P -B $S eval --export $G4/e3l2_smoke/export_train --out $O/eval --ckpt $O/runs/s342101/checkpoints/final.pt --snap $O/runs/s342101/checkpoints/upd4.pt \
  --qfile $G4/e5_quant/criteria_quantizer.json --smoke --workers 1 --threads 2 > $O/eval.log 2>&1; t "eval exit $?"
$P -B $S aggregate --out $O/eval --runs $O/runs/s342101 --control-summary $G4/e3r/summary.json --e3l-summary $G4/e3l/summary.json --smoke > $O/aggregate.log 2>&1; t "aggregate exit $?"
t done
