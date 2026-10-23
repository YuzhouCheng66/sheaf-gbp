#!/bin/bash
# Day-2 Phase C, revised (Claude, 2026-10-22 16:30 UTC; replaces chain_day2_C.sh, whose waiter was stopped before it launched anything).
# Change vs C: the H256 warm start is the BEST H128 checkpoint by VALIDATION R8 wall F1 among the posterior-only / sealed-wire max
# models {v347 max-s342202, v350 dynft-s342202, v354 long-s342202} (selection on VALIDATION is allowed; the fresh panel is never used),
# instead of long-s342202 by default; the H256 continuation runs with train_v350_dyn_edge.py (strict warm start; --allow-partial-init
# because the 256 trunk has extra stages), static data, 60 epochs, lr 1e-4, max fusion, posterior_only, sealed wire.
# Second job in the same lease: dynamic fine-tune of the other seed (dynft-s342101 from v347 max-s342101), same recipe as dynft-s342202.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day2_C2 started pid $$ sha256 $(sha256sum scripts/chain_day2_C2.sh | cut -c1-16)"
while [ ! -f runs/v354-endpoint-long/lease.json ]; do sleep 120; done
while true; do
  [ -f runs/v354-endpoint-long/long-s342101/final.json ] && [ -f runs/v354-endpoint-long/long-s342202/final.json ] && break
  DTB=$(python3 -c "import json;print(int(json.load(open('runs/v354-endpoint-long/lease.json'))['total_deadline_epoch']))")
  if [ $(date +%s) -gt $((DTB+180)) ]; then log "phase B total deadline passed; continuing"; break; fi
  sleep 60
done
log "phase B finished; phase C start"
INIT=$(python3 - <<PY
import json, os
cands = ["runs/v347-fusion-pair/max-s342202", "runs/v350-dynamic-ft/dynft-s342202", "runs/v354-endpoint-long/long-s342202"]
best = None
for c in cands:
    f = c + "/final.json"
    if os.path.exists(f):
        r8 = json.load(open(f))["val"]["8"]["all"]["wall_f1"]
        if best is None or r8 > best[1]: best = (c, r8)
print(best[0] + "/checkpoints/final.pt")
PY
)
mkdir -p runs/v356-h256-max
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v356-h256-max/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "epochs": 60, "size": 256, "seed": 342202, "lr": 1e-4,
 "fusion": "max", "wire": "sealed", "readout": "posterior_only", "warm_start": "$INIT", "warm_start_rule": "best VALIDATION R8 wall F1 among max-s342202 / dynft-s342202 / long-s342202 (posterior_only, sealed wire)",
 "trainer": "scripts/train_v350_dyn_edge.py --allow-partial-init (256 trunk has extra stages; everything else loads strictly)",
 "purpose": "H256 endpoint of the selected H128 design (continuation, not a paired control); reference day-1 v342e sum: K0 .909 / R8 .788 / acc .952 / FPR .034",
 "shared_lease_with": ["runs/v350-dynamic-ft/dynft-s342101 (if launched)"]}
JSON
nohup $P scripts/train_v350_dyn_edge.py --source $S --out runs/v356-h256-max/h256-s342202 --arm owned --size 256 --device cuda --epochs 60 --val-every 6 --seed 342202 --lr 1e-4 --init-from $INIT --allow-partial-init --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v356-h256-max/h256-s342202.log 2>&1 &
log "h256-s342202 pid $! (init $INIT)"
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
