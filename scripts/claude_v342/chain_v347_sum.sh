#!/bin/bash
# The sum arms ran 5x slower than the max arms when co-scheduled (300 vs 56 s/epoch: sealed v295 FP64 matrix path vs the light
# max loop) and would have been deadline-cut at ~18 epochs, breaking the equal-update pairing. They were stopped at epoch 2 and
# are re-run here alone (identical seeds, 36 epochs) once the max arms finish.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
while true; do [ -f runs/v347-fusion-pair/max-s342101/final.json ] && [ -f runs/v347-fusion-pair/max-s342202/final.json ] && break; sleep 60; done
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
python3 - <<PY
import json; p="runs/v347-fusion-pair/lease.json"; d=json.load(open(p))
d["sum_arms_relaunch"]={"lease_start_epoch":$START,"train_deadline_epoch":$DE,"total_deadline_epoch":$DT,"reason":"co-scheduled sum arms were 5x slower and would have been deadline-cut at ~18 epochs; re-run alone with identical seeds/epochs for equal-update pairing"}
json.dump(d,open(p,"w"),indent=1)
PY
for seed in 342101 342202; do
  nohup $P scripts/train_v347_maxsel.py --source $S --out runs/v347-fusion-pair/sum-s$seed --arm owned --size 128 --device cuda --epochs 36 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion sum > runs/v347-fusion-pair/sum-s$seed.log 2>&1 &
  echo "sum-s$seed pid $!"
done
echo "$(date -u) sum arms relaunched"
