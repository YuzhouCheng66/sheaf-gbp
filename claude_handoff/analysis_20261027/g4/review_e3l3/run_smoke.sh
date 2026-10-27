#!/bin/bash
# reviewer's end-to-end CPU smoke of g4_e3l3.py at the reviewed sha (TRAIN houses only: E3-L smoke cache, the E3-L2 smoke ctrl_holdout and TRAIN export)
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill || exit 1
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
P=D:/anaconda3/envs/jax-gpu/python.exe
S=scripts/claude_v342/g4_e3l3.py
G4=claude_handoff/analysis_20261027/g4
O=$G4/review_e3l3/smoke
CACHE=$G4/e3l_smoke/cache
E3R=claude_handoff/analysis_20261027/review_E3R/inputs
CTRL=$G4/e3l2_smoke/ctrl_holdout.json
FIT=C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill/claude_handoff/analysis_20261027/review_E3R/inputs/e1c_fit.json
MAN=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
t(){ echo "$(date -u +%FT%TZ) $*"; }
t "start; g4_e3l3.py sha $(sha256sum $S | cut -c1-16)"
mkdir -p $O
$P -B $S train --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/s342101 --updates 6 --device cpu --threads 4 --batch 2 --log-every 1 \
  --probe-every 3 --holdout-samples 2 --warmup 2 --time-from 2 --snap 4 --zstat-every 1 --manifest-sha $MAN --ctrl-ckpt $E3R/s342101/final.pt --ctrl-holdout $CTRL > $O/train.log 2>&1; t "train exit $?"
$P -B $S select --run $O/runs/s342101 --ctrl $CTRL --workers 2 --threads 2 > $O/select.log 2>&1; t "select exit $?"
$P -B $S select --run $O/runs/s342101 --ctrl $CTRL --workers 2 --threads 2 > $O/select_again.log 2>&1; t "select again (must refuse) exit $?"
$P -B $S eval --export $G4/e3l2_smoke/export_train --out $O/eval --ckpt $O/runs/s342101/checkpoints/final.pt --smoke --workers 1 --threads 4 > $O/eval.log 2>&1; t "eval exit $?"
$P -B $S aggregate --out $O/eval --runs $O/runs/s342101 --control-summary $G4/e3r/summary.json --e3l-summary $G4/e3l/summary.json --e5-summary $G4/e5/summary.json --smoke > $O/aggregate.log 2>&1; t "aggregate exit $?"
$P -B $G4/review_e3l3/neg_asserts.py > $O/neg_asserts.log 2>&1; t "neg asserts exit $?"
$P -B $S select --run $O/neg/w09 --ctrl $CTRL --workers 1 --threads 4 > $O/neg_select_w09.log 2>&1; t "select on w = 0.9 (must fail on the known assertion) exit $?"
grep -m1 "AssertionError" $O/neg_select_w09.log
t done
