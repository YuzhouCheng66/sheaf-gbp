#!/bin/bash
# Day-2 Phase C (Claude, written 2026-10-22 ~14:55 UTC, before Phase A/B results were known). Runs after Phase B (v354) finishes.
#   v356  H256 endpoint with max fusion: warm start from the 128 endpoint (runs/v354-endpoint-long/long-s342202 if its final.json
#         exists, else runs/v347-fusion-pair/max-s342202), native 256 grid (3.9 cm cells), 60 epochs at lr 1e-4 (the day-1 256
#         continuations destabilised at 3e-4), same wire/readout as Phase B (read from runs/v354-endpoint-long/lease.json).
#         Reference: day-1 best 256 = v342e (sum fusion, K0 .909 / R8 .788 / acc .952 / FPR .034).
#         Purpose: carry the selected H128 design to H256 (README target resolutions); not a paired control — reported as a continuation.
#   v350  dynamic fine-tune, second seed (342101), only if the first (dynft-s342202) finished and REVIEW_OK exists; same recipe.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day2_C started pid $$ sha256 $(sha256sum scripts/chain_day2_C.sh | cut -c1-16)"
# wait for phase B to have been launched and finished (both long-s* final.json, or its total deadline + 180 s)
while [ ! -f runs/v354-endpoint-long/lease.json ]; do sleep 120; done
while true; do
  [ -f runs/v354-endpoint-long/long-s342101/final.json ] && [ -f runs/v354-endpoint-long/long-s342202/final.json ] && break
  DTB=$(python3 -c "import json;print(int(json.load(open('runs/v354-endpoint-long/lease.json'))['total_deadline_epoch']))")
  if [ $(date +%s) -gt $((DTB+180)) ]; then log "phase B total deadline passed; continuing"; break; fi
  sleep 60
done
log "phase B finished; phase C start"
WIRE=$(python3 -c "import json;print(json.load(open('runs/v354-endpoint-long/lease.json'))['wire'])")
READOUT=$(python3 -c "import json;print(json.load(open('runs/v354-endpoint-long/lease.json'))['readout'])")
if [ -f runs/v354-endpoint-long/long-s342202/final.json ]; then INIT=runs/v354-endpoint-long/long-s342202/checkpoints/final.pt; else INIT=runs/v347-fusion-pair/max-s342202/checkpoints/final.pt; WIRE=sealed; READOUT=posterior_only; fi
mkdir -p runs/v356-h256-max
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v356-h256-max/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "epochs": 60, "size": 256, "seed": 342202, "lr": 1e-4,
 "fusion": "max", "wire": "$WIRE", "readout": "$READOUT", "warm_start": "$INIT",
 "purpose": "H256 endpoint of the selected H128 design (continuation, not a paired control); reference day-1 v342e sum: K0 .909 / R8 .788 / acc .952 / FPR .034",
 "shared_lease_with": ["runs/v350-dynamic-ft/dynft-s342101 (if launched)"]}
JSON
if [ "$WIRE" = kbit ]; then TR=scripts/train_v348_kbit.py; EXTRA="--wire kbit --mu-bits 8 --q-bits 8"; else TR=scripts/train_v347_maxsel.py; EXTRA=""; fi
nohup $P $TR --source $S --out runs/v356-h256-max/h256-s342202 --arm owned --size 256 --device cuda --epochs 60 --val-every 6 --seed 342202 --lr 1e-4 --init-from $INIT --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout $READOUT --fusion max $EXTRA > runs/v356-h256-max/h256-s342202.log 2>&1 &
log "h256-s342202 pid $! (trainer $TR, init $INIT, wire $WIRE, readout $READOUT)"
if [ -f runs/v350-dynamic-ft/REVIEW_OK ] && [ -f runs/v350-dynamic-ft/dynft-s342202/final.json ]; then
  nohup $P scripts/train_v350_dyn_edge.py --source $S --out runs/v350-dynamic-ft/dynft-s342101 --arm owned --size 128 --device cuda --dynamic --fusion max --readout posterior_only --init-from runs/v347-fusion-pair/max-s342101/checkpoints/final.pt --epochs 6 --lr 1e-4 --val-every 2 --seed 342101 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v350-dynamic-ft/dynft-s342101.log 2>&1 &
  log "dynft-s342101 pid $!"
  sleep 120
  while pgrep -f "runs/v350-dynamic-ft/dynft-s342101" >/dev/null; do sleep 60; done
  if [ -f runs/v350-dynamic-ft/dynft-s342101/checkpoints/final.pt ]; then
    $P scripts/eval_dynamic_v349.py --source $S --checkpoint runs/v350-dynamic-ft/dynft-s342101/checkpoints/final.pt --trainer scripts/train_v350_dyn_edge.py --out runs/v349-dynamic/dynft-s342101.json --device cuda > runs/v349-dynamic/dynft-s342101.log 2>&1
    log "dynamic eval of dynft-s342101 done"
  fi
fi
log "phase C launched/finished its launches"
