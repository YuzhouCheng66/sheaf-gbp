#!/bin/bash
# Day-4 batch M (Claude, written 2026-10-22 23:50 UTC, before these runs and before the batch-L ceilings finished).
# Question: is HouseExpo accuracy limited by CODE CAPACITY DENSITY? On the fixed 20 m chart a house covers ~24 of the 64 public blocks
# and an agent observes ~11; with the house-fitted chart (v371: chart side L = max(8, ceil(bbox + 1 m)), same 8x8 grid, same metric
# sensor/radio/poses, same splits) the house covers ~45 blocks and an agent observes ~19.
#  M1 v372 in-domain training on HouseExpo v371, recipe of v367 UNCHANGED (train_v371.py == train_v365.py + the world switch;
#     72 epochs, lr 3e-4 cosine, max, posterior_only, sealed wire), seeds 342101 / 342202. ONE change vs v367: the chart.
#     Pre-registered expectation: if capacity density is the limit, R8 and K0 rise by >= 5 pp vs v367 (.639/.654, K0 .712/.736) on both
#     seeds; < 2 pp means density is not the bottleneck. NOTE: absolute numbers are on a different pixel scale (cell L/128 ~ 0.11 m vs
#     0.156 m), so the comparison is per-house map accuracy at the native raster, reported as such.
#  M2 v373 ceiling hx-fit-8x8x1 (owned_ceiling_v371.py, same protocol as batch L) for the representation ceiling on v371.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_M started pid $$ sha256 $(sha256sum scripts/chain_day4_M.sh | cut -c1-16)"
until grep -q "batch I done" $LOG && grep -q "batch L done" $LOG; do sleep 60; done
log "batch M start"
mkdir -p runs/v372-hxfit-train runs/v373-hxfit-ceiling
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v372-hxfit-train/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "epochs": 72, "seeds": [342101, 342202],
 "world": "houseexpo_fit v371 (data module sha256 $(sha256sum scripts/mapping_houseexpo_data_v371.py | cut -d' ' -f1); index sha256 $(sha256sum $HOUSEEXPO_INDEX | cut -d' ' -f1))",
 "recipe": "v367 unchanged; one change: house-fitted chart", "trainer_sha256": "$(sha256sum scripts/train_v371.py | cut -d' ' -f1)",
 "expectation": "capacity-density limit => R8 and K0 >= +5pp vs v367 on both seeds; < 2pp => not the bottleneck", "shared_lease_with": ["runs/v373-hxfit-ceiling"]}
JSON
for seed in 342101 342202; do
  nohup $P scripts/train_v371.py --source $S --world houseexpo_fit --out runs/v372-hxfit-train/hxfit-long-s$seed --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v372-hxfit-train/hxfit-long-s$seed.log 2>&1 &
  log "hxfit-long-s$seed pid $!"
done
nohup $P scripts/owned_ceiling_v371.py --source $S --out runs/v373-hxfit-ceiling/hxfit-8x8x1 --world houseexpo_fit --owned 8x8x1 --bits 8 --head dt --q binary --size 128 --device cuda --train-indices 0-63 --train-rooms 1,2,3 --train-agents 8,16 --val-indices 0-7 --val-rooms 1,2,3 --val-agents 8,16 --steps 4000 --time-cap 2000 > runs/v373-hxfit-ceiling/hxfit-8x8x1.log 2>&1 &
log "ceiling hxfit-8x8x1 pid $!"
sleep 60
while pgrep -f "runs/v372-hxfit-train/hxfit-long" >/dev/null || pgrep -f "runs/v373-hxfit-ceiling" >/dev/null; do sleep 60; done
log "batch M done"
