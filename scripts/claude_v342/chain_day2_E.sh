#!/bin/bash
# Day-2 Phase E (Claude, 2026-10-22 17:35 UTC, written before Phase C/D results were known). Runs after Phase D (v359 k-bit seed 2).
#   v360  dynamic fine-tune on top of the 72-epoch static endpoint, both seeds: --dynamic, warm start runs/v354-endpoint-long/long-s*/final.pt,
#         6 dynamic epochs (21,312 updates), lr 1e-4, max, posterior_only, sealed wire (the recipe that passed its gate in Phase A).
#         Purpose: the final H128 model should have both the static endpoint's accuracy and the dynamic robustness (K0 under growing maps,
#         memory fusion). GATE (pre-registered): static VALIDATION K0/R8 not worse than the long run by more than 1 pp AND dynamic own K0 at
#         t=5 >= long run's zero-shot + 3 pp AND crdt t=5 >= zero-shot + 3 pp (eval_dynamic_v349, radio 8, idx 0-7; fresh idx 8-15 and radio 5 reported).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day2_E started pid $$ sha256 $(sha256sum scripts/chain_day2_E.sh | cut -c1-16)"
while [ ! -f runs/v359-kbit-seed2/lease.json ]; do sleep 120; done
while true; do
  ok=1; for a in b8 b6 m6q4 m6q5; do [ -f runs/v359-kbit-seed2/$a/final.json ] || ok=0; done
  [ $ok = 1 ] && break
  DTD=$(python3 -c "import json;print(int(json.load(open('runs/v359-kbit-seed2/lease.json'))['total_deadline_epoch']))")
  if [ $(date +%s) -gt $((DTD+180)) ]; then log "phase D total deadline passed; continuing"; break; fi
  sleep 60
done
log "phase D finished; phase E start"
mkdir -p runs/v360-long-dynft
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v360-long-dynft/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "dynamic_epochs": 6, "updates_planned": 21312, "size": 128, "seeds": [342101, 342202],
 "warm_start": "runs/v354-endpoint-long/long-s{seed}/checkpoints/final.pt", "lr": 1e-4, "fusion": "max", "readout": "posterior_only", "wire": "sealed",
 "gate": "static VALIDATION K0/R8 >= long - 1pp AND dynamic own K0 t=5 >= long zero-shot + 3pp AND crdt t=5 >= long zero-shot + 3pp (radio 8, idx 0-7); fresh idx 8-15 and radio 5 reported",
 "trainer_sha256": "$(sha256sum scripts/train_v350_dyn_edge.py | cut -d' ' -f1)"}
JSON
for seed in 342101 342202; do
  nohup $P scripts/train_v350_dyn_edge.py --source $S --out runs/v360-long-dynft/ldynft-s$seed --arm owned --size 128 --device cuda --dynamic --fusion max --readout posterior_only --init-from runs/v354-endpoint-long/long-s$seed/checkpoints/final.pt --epochs 6 --lr 1e-4 --val-every 2 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v360-long-dynft/ldynft-s$seed.log 2>&1 &
  log "ldynft-s$seed pid $!"
done
# zero-shot dynamic evaluation of the long runs (the reference for the gate), light, sequential
for seed in 342101 342202; do
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint runs/v354-endpoint-long/long-s$seed/checkpoints/final.pt --trainer scripts/train_v350_dyn_edge.py --out runs/v349-dynamic/zeroshot-long-s$seed.json --device cuda > runs/v349-dynamic/zeroshot-long-s$seed.log 2>&1
done
log "zero-shot dynamic evals of the long runs done"
sleep 60
while pgrep -f "runs/v360-long-dynft/ldynft-s" >/dev/null; do sleep 60; done
log "phase E trainers finished"
for seed in 342101 342202; do
  CK=runs/v360-long-dynft/ldynft-s$seed/checkpoints/final.pt
  if [ -f $CK ]; then
    $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v350_dyn_edge.py --out runs/v349-dynamic/ldynft-s$seed.json --device cuda > runs/v349-dynamic/ldynft-s$seed.log 2>&1
    $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v350_dyn_edge.py --radio 5 --out runs/v349-dynamic/ldynft-s$seed-radio5.json --device cuda > runs/v349-dynamic/ldynft-s$seed-radio5.log 2>&1
    $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v350_dyn_edge.py --indices 8-15 --out runs/v349-dynamic/ldynft-s$seed-fresh8-15.json --device cuda > runs/v349-dynamic/ldynft-s$seed-fresh8-15.log 2>&1
    $P scripts/eval_threshold_v347.py --source $S --checkpoint $CK --trainer scripts/train_v350_dyn_edge.py --out runs/v355-threshold/tau-ldynft-s$seed.json --device cuda --taus 0.5 > runs/v355-threshold/tau-ldynft-s$seed.log 2>&1
  fi
done
log "phase E evaluations done"
