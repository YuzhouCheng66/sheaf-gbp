#!/bin/bash
# Day-4 batch K (Claude, written 2026-10-23 ~04:20 UTC, after batch J and before any batch-K run): the replication pre-registered in
# chain_day3_J2.sh / chain_day4_J.sh -- the largest admissible lambda of batch J (l01, lambda 0.01) and its lambda=0 control on the
# second seed 342101, warm start runs/v354-endpoint-long/long-s342101/checkpoints/final.pt, same recipe/trainer (train_v368.py, sha in
# runs/v369-rate/REVIEW_OK). Runs after batch P so the GPU is never shared.
# Pre-registered reading (same admissibility rule as batch J, now on seed 342101): bits <= 0.8 x l0 bits AND R8 >= l0 - 1 pp AND
# FPR <= l0 + 0.3 pp (VAL) => lambda 0.01 REPLICATES; otherwise it does not. Exact bytes re-measured with eval_bytes_v380.py.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_K started pid $$ sha256 $(sha256sum scripts/chain_day4_K.sh | cut -c1-16)"
until grep -q "batch P analysis" $LOG; do sleep 60; done
while pgrep -f "runs/v379-dose/d4800" >/dev/null || pgrep -f "runs/v378-rect/rect" >/dev/null; do sleep 60; done
TSHA=$(sha256sum scripts/train_v368.py | cut -d" " -f1)
grep -q "$TSHA" runs/v369-rate/REVIEW_OK || { log "batch K: REVIEW_OK does not name trainer sha $TSHA; NOT launched"; exit 0; }
log "batch K start (replication of l01 / lambda 0.01 on seed 342101)"
START=$(date +%s); DE=$((START+6300)); DT=$((START+7100))
cat > runs/v369-rate/lease_K.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "seed": 342101, "warm_start": "runs/v354-endpoint-long/long-s342101/checkpoints/final.pt",
 "arms": {"l0": 0.0, "l01": 0.01}, "trainer_sha256": "$TSHA", "purpose": "replication of the largest admissible lambda from batch J on the second seed (same recipe)"}
JSON
for arm in "l0 0" "l01 0.01"; do
  set -- $arm
  nohup $P scripts/train_v368.py --source $S --out runs/v369-rate/$1-s342101 --arm owned --size 128 --device cuda --fusion max --readout posterior_only --wire kbit --mu-bits 8 --q-bits 8 --rate-model --rate-lambda $2 --init-from runs/v354-endpoint-long/long-s342101/checkpoints/final.pt --epochs 36 --lr 1e-4 --val-every 6 --seed 342101 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v369-rate/$1-s342101.log 2>&1 &
  log "rate replication arm $1 (lambda $2) seed 342101 pid $!"
done
sleep 60
while pgrep -f "runs/v369-rate/l.*-s342101" >/dev/null; do sleep 60; done
log "batch K finished"
RES=$(python3 - <<'PY'
import json
def g(a):
    try:
        d = json.load(open(f"runs/v369-rate/{a}-s342101/final.json"))
        if d.get("updates") != 21312: return None
        v = d["val"]; return v["rate"]["model_bits_per_observed_coordinate"], v["8"]["all"]["wall_f1"], v["8"]["all"]["unknown_known_fpr"]
    except Exception:
        return None
b0, b1 = g("l0"), g("l01")
if not (b0 and b1): print("INCONCLUSIVE (incomplete run)")
else:
    ok = b1[0] <= .8 * b0[0] and b1[1] >= b0[1] - .01 and b1[2] <= b0[2] + .003
    print(("REPLICATES" if ok else "DOES NOT REPLICATE") + f": bits {b1[0]:.2f} vs {b0[0]:.2f}, R8 {b1[1]:.4f} vs {b0[1]:.4f}, FPR {b1[2]:.4f} vs {b0[2]:.4f}")
PY
)
log "batch K admissibility (seed 342101): $RES"
for a in l0 l01; do
  OMP_NUM_THREADS=2 nice -n 10 $P scripts/eval_bytes_v380.py --source $S --checkpoint runs/v369-rate/$a-s342101/checkpoints/final.pt --out runs/v380-bytes/bytes-$a-s342101.json --device cpu > runs/v380-bytes/bytes-$a-s342101.log 2>&1
done
log "batch K exact bytes written (runs/v380-bytes/bytes-*-s342101.json)"
