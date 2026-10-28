#!/bin/bash
# Reviewer's local CPU smoke of E3-L4 (TRAIN houses only: the E3-L smoke cache, the E3-L2 smoke ctrl_holdout and TRAIN export).
# Differs from the author's smoke: w* = 3 (the smallest grid value), BOTH training seeds, both checkpoints in one eval / aggregate.
# <= 4 processes at any time (2 trainers in parallel; then 1 parent + 2 workers). Never touches VAL / TEST, never the GPU.
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill || exit 1
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
P=D:/anaconda3/envs/jax-gpu/python.exe
S=scripts/claude_v342/g4_e3l4.py
G4=claude_handoff/analysis_20261027/g4
O=$G4/review_e3l4/smoke
CACHE=$G4/e3l_smoke/cache
E3R=claude_handoff/analysis_20261027/review_E3R/inputs
CTRL=$G4/e3l2_smoke/ctrl_holdout.json
FIT=C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill/claude_handoff/analysis_20261027/review_E3R/inputs/e1c_fit.json
MAN=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
W=3
t(){ echo "$(date -u +%FT%TZ) $*"; }
t "start; g4_e3l4.py sha $(sha256sum $S | cut -c1-16); g4_e3l3.py sha $(sha256sum scripts/claude_v342/g4_e3l3.py | cut -c1-16); w* $W"
for s in 342101 342202; do
  mkdir -p $O/e3r_layout/s$s/checkpoints
  cp $E3R/s$s/final.pt $O/e3r_layout/s$s/checkpoints/final.pt && cp $E3R/s$s/threshold.json $O/e3r_layout/s$s/threshold.json
done
$P -B $S train --w-star 5 --prereg-updates 9562 --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/refused_w5 --updates 9562 --device cpu > $O/train_refused_w5.log 2>&1; t "train w*=5 (must refuse) exit $? : $(tail -1 $O/train_refused_w5.log)"
$P -B $S train --w-star 3 --prereg-updates 9562 --cache $CACHE --evidence "c:$FIT:A:cam_nf" --seed 342101 --out $O/runs/refused_up --updates 9563 --device cpu > $O/train_refused_up.log 2>&1; t "train 9563 > 9562 (must refuse) exit $? : $(tail -1 $O/train_refused_up.log)"
EVA="c:$FIT:A:cam_nf"; EVB="c:$FIT:B:cam_nf"
for s in 342101 342202; do
  [ $s = 342101 ] && EV=$EVA || EV=$EVB
  $P -B $S train --w-star $W --prereg-updates 9562 --second-note "review smoke: 8 updates instead of 9562" --cache $CACHE --evidence "$EV" --seed $s --out $O/runs/s$s \
    --updates 8 --device cpu --threads 2 --batch 2 --log-every 1 --probe-every 4 --holdout-samples 2 --warmup 2 --time-from 2 --snap 6 --zstat-every 1 --manifest-sha $MAN \
    --ctrl-ckpt $E3R/s$s/final.pt --ctrl-holdout $CTRL > $O/train_s$s.log 2>&1 &
done
wait; t "train both seeds done: $(tail -1 $O/train_s342101.log | cut -c1-160) | $(tail -1 $O/train_s342202.log | cut -c1-160)"
for s in 342101 342202; do
  $P -B $S select --w-star $W --run $O/runs/s$s --ctrl $CTRL --workers 1 --threads 1 > $O/select_s$s.log 2>&1 &
done
wait; t "select done: $(tail -1 $O/select_s342101.log | cut -c1-200) | $(tail -1 $O/select_s342202.log | cut -c1-200)"
$P -B $S select --w-star 4 --run $O/runs/s342101 --ctrl $CTRL --workers 1 --threads 1 > $O/select_w4_refused.log 2>&1; t "select with w* 4 on a w*=3 run (must refuse) exit $?: $(tail -1 $O/select_w4_refused.log | cut -c1-200)"
$P -B $S eval --w-star $W --export $G4/e3l2_smoke/export_train --out $O/eval --ckpt $O/runs/s342101/checkpoints/final.pt,$O/runs/s342202/checkpoints/final.pt --expect-seeds 203 --smoke --workers 2 --threads 1 > $O/eval.log 2>&1; t "eval exit $?"
$P -B scripts/claude_v342/g4_e3r.py eval --export $G4/e3l2_smoke/export_train --out $O/ctrl --ckpt $O/e3r_layout/s342101/checkpoints/final.pt,$O/e3r_layout/s342202/checkpoints/final.pt --smoke --workers 2 --threads 1 > $O/ctrl_eval.log 2>&1; t "control (g4_e3r eval) exit $?"
$P -B $S ctrlagg --out $O/ctrl --expect-seeds 203 --smoke > $O/ctrl_agg.log 2>&1; t "ctrlagg exit $?"
$P -B $S aggregate --w-star $W --out $O/eval --runs $O/runs/s342101,$O/runs/s342202 --control-summary $O/ctrl/summary.json --e5-summary $G4/e5/summary.json --smoke > $O/aggregate.log 2>&1; t "aggregate exit $?"
head -3 $O/aggregate.log
t done
