#!/bin/bash
# Day-3 batch L (Claude, 2026-10-22 ~22:50 UTC, written before these runs). Diagnostic, read-only w.r.t. the main line.
# Why: HouseExpo in-domain training learns far slower than v293 (epoch 24: K0 .54-.59 vs .91), and zero-shot transfer collapsed
# (K0 .39-.42). First-principles question before any redesign: is the 64-scalar owned block code itself unable to hold HouseExpo
# maps (representation limit), or is it the encoder/training? Ceiling protocol of day 1 (owned_ceiling.py; per-agent codes fitted to
# the agent's OWN observation only, decoder trained on TRAIN with free codes, unions composed exactly), run on HouseExpo v365:
#   hx-8x8x1 (deployed layout), hx-4x4x4 (64 dims, coarser blocks / more scalars per block),
#   hx-16x16x1 (256 dims: DIAGNOSTIC ONLY, violates the 64-dim constraint; shows whether 64 is binding),
#   v293-8x8x1 (same protocol on v293, all three room strata: the apples-to-apples reference).
# All: H128, 8-bit codes, DT head, binary q, 4000 steps, time cap 2000 s. No gate (measurement).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day3_L started pid $$ sha256 $(sha256sum scripts/chain_day3_L.sh | cut -c1-16)"
until grep -q "batch H trainers finished" $LOG; do sleep 60; done
mkdir -p runs/v370-ceiling
START=$(date +%s)
echo "{\"lease_start_epoch\": $START, \"total_deadline_epoch\": $((START+7000)), \"inclusive_seconds\": 7200, \"script_sha256\": \"$(sha256sum scripts/owned_ceiling_v370.py | cut -d' ' -f1)\", \"design\": \"see chain_day3_L.sh header\"}" > runs/v370-ceiling/lease.json
for arm in "hx-8x8x1 houseexpo 8x8x1 1,2,3" "hx-4x4x4 houseexpo 4x4x4 1,2,3" "hx-16x16x1 houseexpo 16x16x1 1,2,3" "v293-8x8x1 v293 8x8x1 6,12,20"; do
  set -- $arm
  nohup $P scripts/owned_ceiling_v370.py --source $S --out runs/v370-ceiling/$1 --world $2 --owned $3 --allow-non64 --bits 8 --head dt --q binary --size 128 --device cuda --train-indices 0-63 --train-rooms $4 --train-agents 8,16 --val-indices 0-7 --val-rooms $4 --val-agents 8,16 --steps 4000 --time-cap 2000 > runs/v370-ceiling/$1.log 2>&1 &
  log "ceiling $1 pid $!"
done
wait
log "batch L done"
