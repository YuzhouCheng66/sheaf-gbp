#!/bin/bash
# Day-4 batch R (Claude, written 2026-10-23 08:30 UTC after batch Q, before any batch-R run).
# Batch Q: 4 epochs (57,600 updates) beat 3 epochs by +2.5 / +1.1 pp VAL R8 -> still under-trained. R continues training: warm start from
# the batch-Q final weights (runs/v382-long/d4800e4-s<seed>/checkpoints/final.pt), a SECOND cosine cycle of 4 epochs (57,600 updates) at
# peak lr 1e-4 (1/3 of the first cycle), same data / chart / trainer / seeds. Declared as a two-cycle schedule (not an equivalent of one
# longer cosine). Control = the batch-Q final weights (the warm start itself).
# Pre-registered reading: R final VAL R8 >= Q final + 1 pp on BOTH seeds => more training still helps; |gain| < 1 pp on both => converged
# for this model size; else ambiguous. Incomplete run (updates != 57,600) => INCONCLUSIVE.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_R started pid $$ sha256 $(sha256sum scripts/chain_day4_R.sh | cut -c1-16)"
mkdir -p runs/v383-cont
while pgrep -f "runs/v382-long/d4800e4" >/dev/null; do sleep 60; done
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v383-cont/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "warm_start": "runs/v382-long/d4800e4-s<seed>/checkpoints/final.pt", "lr": 1e-4, "trainer_sha256": "$(sha256sum scripts/train_v378.py | cut -d' ' -f1)",
 "expectation": "R >= Q + 1pp final VAL R8 on both seeds => more training helps; |gain| < 1pp => converged; incomplete => INCONCLUSIVE"}
JSON
log "batch R start (second cosine cycle, warm start from batch Q final)"
for seed in 342101 342202; do
  nohup $P scripts/train_v378.py --source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --out runs/v383-cont/cont-s$seed --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --seed $seed --init-from runs/v382-long/d4800e4-s$seed/checkpoints/final.pt --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 > runs/v383-cont/cont-s$seed.log 2>&1 &
  log "cont seed $seed pid $!"
done
sleep 60
while pgrep -f "runs/v383-cont/cont" >/dev/null; do sleep 60; done
log "batch R done"
RES=$(python3 - <<'PY'
import json
g = []
for s in ("342101", "342202"):
    try:
        a = json.load(open(f"runs/v383-cont/cont-s{s}/final.json")); c = json.load(open(f"runs/v382-long/d4800e4-s{s}/final.json"))
        if a.get("updates") != 57600: print("INCONCLUSIVE"); raise SystemExit
        g.append(100 * (a["val"]["8"]["all"]["wall_f1"] - c["val"]["8"]["all"]["wall_f1"]))
    except SystemExit: raise
    except Exception as e: print("INCONCLUSIVE", e); raise SystemExit
v = "more training helps" if min(g) >= 1 else ("converged" if max(abs(x) for x in g) < 1 else "ambiguous")
print(v, [round(x, 2) for x in g])
PY
)
log "batch R analysis: $RES"
