#!/bin/bash
# E3-L5 local CPU smoke (TRAIN houses only; no VAL, no TEST; <= 5 processes): ctrlprep on the 3-house smoke cache -> 2 x train (4 updates,
# batch 2, CPU) -> select (learned w + w = 4) -> eval on one G4-E5 TRAIN hold-out export file -> h1 on the H1 TRAIN smoke records -> aggregate.
cd /c/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill || exit 1
export CUDA_VISIBLE_DEVICES=
P=D:/anaconda3/envs/jax-gpu/python.exe
S=scripts/claude_v342
AN=claude_handoff/analysis_20261027
CACHE=$AN/probes/e3l5_smoke/cache_e3l5
OUT=$AN/g4/e3l5_smoke
INP=$AN/review_E3R/inputs
FIT=$AN/g4/e1c/fit.json
CRIT=$AN/g4/e7/criteria_w.json
set -x
date -u
[ -f $OUT/ctrl_holdout_smoke.json ] || $P -B $S/g4_e3l2.py ctrlprep --cache $CACHE --e3r $INP --out $OUT/ctrl_holdout_smoke.json --workers 2 --threads 1 || exit 2
rm -rf $OUT/runs $OUT/eval $OUT/h1 $OUT/agg
for s in 342101 342202; do
  h=A; [ $s = 342202 ] && h=B
  $P -B $S/g4_e3l5.py train --smoke --prereg-updates 9562 --cache $CACHE --evidence "c:$FIT:$h:cam_nf" --seed $s --out $OUT/runs/s$s --updates 4 --device cpu \
     --batch 2 --log-every 2 --probe-every 2 --snap 0 --time-from 1 --threads 4 --ctrl-ckpt $INP/s$s/final.pt --ctrl-holdout $OUT/ctrl_holdout_smoke.json || exit 3
  $P -B $S/g4_e3l5.py select --smoke --run $OUT/runs/s$s --ctrl $OUT/ctrl_holdout_smoke.json --workers 1 --threads 4 || exit 4
done
date -u
$P -B $S/g4_e3l5.py eval --smoke --export $AN/g4/e5_export/train_holdout --out $OUT/eval --runs $OUT/runs/s342101,$OUT/runs/s342202 --e3l3 $INP \
   --e3l3-criteria $CRIT --e3r $INP --expect-seeds 200 --panels trainho --tiers cam_nf --only trainho_r1_i19_s200 --workers 1 --threads 4 || exit 5
date -u
$P -B $S/g4_e3l5.py h1 --smoke --h1-jobs $AN/g4/h1_smoke2/train/jobs --h1-export $AN/g4/h1_export --out $OUT/h1 --runs $OUT/runs/s342101,$OUT/runs/s342202 \
   --e3l3 $INP --e3l3-criteria $CRIT --e3r $INP --fit $FIT --variants ctrl,novel,floating --seeds 203 --workers 3 --threads 1 || exit 6
date -u
$P -B $S/g4_e3l5.py aggregate --smoke --out $OUT/agg --runs $OUT/runs/s342101,$OUT/runs/s342202 --eval-out $OUT/eval --h1-out $OUT/h1 || exit 7
date -u
