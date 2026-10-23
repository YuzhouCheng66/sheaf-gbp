#!/bin/bash
# Relaunch of the H256 continuation (Claude, 2026-10-22 17:25 UTC): the first launch (17:17 UTC, pid 403754) died at start with CUDA OOM
# because the H256 max rollout needs ~18.6 GB and the two co-scheduled H128 trainers held 12.7 GB (31.4 GB card). Not resumed: the
# OOM run never completed an update. Waits until the two H128 trainers (staticft-s342202, dynft-s342101) have finished, then starts
# the identical command under a NEW lease (same warm start, same recipe).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "relaunch_h256 waiter started pid $$"
mv runs/v356-h256-max/h256-s342202 runs/v356-h256-max/ABORTED-oom-h256-s342202 2>/dev/null
mv runs/v356-h256-max/h256-s342202.log runs/v356-h256-max/ABORTED-oom-h256-s342202.log 2>/dev/null
while pgrep -f "runs/v350-dynamic-ft/staticft-s342202" >/dev/null || pgrep -f "runs/v350-dynamic-ft/dynft-s342101" >/dev/null; do sleep 60; done
sleep 30
INIT=runs/v354-endpoint-long/long-s342202/checkpoints/final.pt
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
python3 - "$START" "$DE" "$DT" <<'PY'
import json, sys
p = "runs/v356-h256-max/lease.json"; d = json.load(open(p))
d["relaunch"] = {"lease_start_epoch": int(sys.argv[1]), "train_deadline_epoch": int(sys.argv[2]), "total_deadline_epoch": int(sys.argv[3]),
 "reason": "first launch died at start with CUDA OOM (18.6 GB needed, 12.7 GB held by two co-scheduled H128 trainers); relaunched alone with the identical command under a new lease; the OOM run is kept as ABORTED-oom-h256-s342202 (no update completed)"}
json.dump(d, open(p, "w"), indent=1)
PY
nohup $P scripts/train_v350_dyn_edge.py --source $S --out runs/v356-h256-max/h256-s342202 --arm owned --size 256 --device cuda --epochs 60 --val-every 6 --seed 342202 --lr 1e-4 --init-from $INIT --allow-partial-init --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v356-h256-max/h256-s342202.log 2>&1 &
log "h256-s342202 RELAUNCHED pid $! (alone; init $INIT)"
