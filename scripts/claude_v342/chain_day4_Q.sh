#!/bin/bash
# Day-4 batch Q (Claude, written 2026-10-23 ~05:30 UTC, before the batch-P result and before any batch-Q run).
# Question: is the HouseExpo model still under-trained at 43,200 updates? (D2400 training loss was still falling at the end.)
# ONE change vs control (batch P d4800-s<seed>, 3 epochs = 43,200 updates): 4 epochs = 57,600 updates (cosine over the run's own total).
# Same square house-fitted chart (kept by batch O), same 14,400-house cache, same seeds, same trainer (train_v378.py).
# Time: measured 2-job rate ~0.094 s/update -> ~5,400 s + loads/validations, inside the 6,600 s train deadline (a completion gate makes
# any truncated run INCONCLUSIVE). Runs after batch K (rate replication) so the GPU is never shared.
# Pre-registered reading (analyze_v382.py): 4-epoch >= 3-epoch + 1 pp final VAL R8 on BOTH seeds => still under-trained; |gain| < 1 pp
# on both => converged at 43,200 updates; else ambiguous.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_Q started pid $$ sha256 $(sha256sum scripts/chain_day4_Q.sh | cut -c1-16)"
mkdir -p runs/v382-long
until grep -q "batch K admissibility\|batch K: REVIEW_OK does not name" $LOG && grep -q "batch P analysis" $LOG; do sleep 60; done
while pgrep -f "runs/v369-rate/l.*-s342101" >/dev/null || pgrep -f "runs/v379-dose/d4800" >/dev/null; do sleep 60; done
log "batch Q start (square chart, 14,400 houses, 4 epochs = 57,600 updates)"
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v382-long/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "world": "houseexpo_fit",
 "arms": {"d4800e4-s342101": "4800/stratum, 4 ep", "d4800e4-s342202": "4800/stratum, 4 ep"}, "updates_each": 57600, "control": "runs/v379-dose/d4800-s<seed> (3 ep, 43,200)",
 "trainer_sha256": "$(sha256sum scripts/train_v378.py | cut -d' ' -f1)", "cache_manifest_sha256": "$(sha256sum $C/houseexpo_fit-k4800/manifest.json | cut -d' ' -f1)",
 "expectation": "4ep >= 3ep + 1pp final VAL R8 on both seeds => under-trained; |gain| < 1pp => converged; incomplete/identity mismatch => INCONCLUSIVE"}
JSON
for seed in 342101 342202; do
  nohup $P scripts/train_v378.py --source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --out runs/v382-long/d4800e4-s$seed --arm owned --size 128 --device cuda --epochs 4 --val-every 1 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 > runs/v382-long/d4800e4-s$seed.log 2>&1 &
  log "long d4800e4 seed $seed pid $!"
done
sleep 60
while pgrep -f "runs/v382-long/d4800e4" >/dev/null; do sleep 60; done
log "batch Q done"
(cd scripts && $P analyze_v382.py --runs ../runs --out ../runs/v382-long/analysis.json) > runs/v382-long/analysis.txt 2>&1
log "batch Q analysis: $(grep PRIMARY runs/v382-long/analysis.txt)"
