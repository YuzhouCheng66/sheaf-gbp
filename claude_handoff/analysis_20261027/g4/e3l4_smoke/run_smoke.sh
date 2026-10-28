#!/bin/bash
# E3-L4 local CPU smoke (TRAIN houses only: the E3-L smoke cache, the E3-L2 smoke ctrl_holdout and TRAIN export). Never touches VAL / TEST.
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill || exit 1
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
P=D:/anaconda3/envs/jax-gpu/python.exe
S=scripts/claude_v342/g4_e3l4.py
G4=claude_handoff/analysis_20261027/g4
O=$G4/e3l4_smoke
CACHE=$G4/e3l_smoke/cache
E3R=claude_handoff/analysis_20261027/review_E3R/inputs
CTRL=$G4/e3l2_smoke/ctrl_holdout.json
FIT=C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill/claude_handoff/analysis_20261027/review_E3R/inputs/e1c_fit.json
MAN=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
t(){ echo "$(date -u +%FT%TZ) $*"; }
t "start; g4_e3l4.py sha $(sha256sum $S | cut -c1-16); g4_e3l3.py sha $(sha256sum scripts/claude_v342/g4_e3l3.py | cut -c1-16)"
mkdir -p $O/e3r_layout/s342101/checkpoints
cp $E3R/s342101/final.pt $O/e3r_layout/s342101/checkpoints/final.pt && cp $E3R/s342101/threshold.json $O/e3r_layout/s342101/threshold.json
$P -B $S train --w-star 4 --prereg-updates 9562 --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/refused --updates 6 --device cpu > $O/train_refused.log 2>&1; t "train without --second-note (must refuse) exit $? : $(tail -1 $O/train_refused.log)"
$P -B $S train --w-star 4 --prereg-updates 9562 --second-note "smoke: 6 updates instead of 9562" --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/s342101 \
  --updates 6 --device cpu --threads 4 --batch 2 --log-every 1 --probe-every 3 --holdout-samples 2 --warmup 2 --time-from 2 --snap 4 --zstat-every 1 --manifest-sha $MAN \
  --ctrl-ckpt $E3R/s342101/final.pt --ctrl-holdout $CTRL > $O/train.log 2>&1; t "train exit $?"
$P -B $S select --w-star 4 --run $O/runs/s342101 --ctrl $CTRL --workers 2 --threads 2 > $O/select.log 2>&1; t "select exit $?"
$P -B $S select --w-star 4 --run $O/runs/s342101 --ctrl $CTRL --workers 2 --threads 2 > $O/select_again.log 2>&1; t "select again (must refuse) exit $?"
$P -B $S select --w-star 6 --run $O/runs/s342101 --ctrl $CTRL --workers 2 --threads 2 > $O/select_w6.log 2>&1; t "select with w* 6 (must refuse) exit $?"
$P -B $S eval --w-star 4 --export $G4/e3l2_smoke/export_train --out $O/eval --ckpt $O/runs/s342101/checkpoints/final.pt --expect-seeds 203 --smoke --workers 1 --threads 4 > $O/eval.log 2>&1; t "eval exit $?"
$P -B scripts/claude_v342/g4_e3r.py eval --export $G4/e3l2_smoke/export_train --out $O/ctrl --ckpt $O/e3r_layout/s342101/checkpoints/final.pt --smoke --workers 1 --threads 4 > $O/ctrl_eval.log 2>&1; t "control (g4_e3r eval) exit $?"
$P -B $S ctrlagg --out $O/ctrl --expect-seeds 203 --smoke > $O/ctrl_agg.log 2>&1; t "ctrlagg exit $?"
$P -B $S aggregate --w-star 4 --out $O/eval --runs $O/runs/s342101 --control-summary $O/ctrl/summary.json --e5-summary $G4/e5/summary.json --smoke > $O/aggregate.log 2>&1; t "aggregate exit $?"
head -2 $O/aggregate.log
t done
