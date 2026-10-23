#!/bin/bash
# Day-3 batch J, revision 2 (Claude, 2026-10-22 ~22:20 UTC; replaces chain_day3_J.sh, whose waiter was stopped before launching anything).
# Changes after the independent review (code_review_v368_rate.md): trainer v368 now pre-fits the entropy model on the warm-start encoder's
# TRAIN codes and gives it a constant-lr optimizer (M1); admissibility parsing is robust and requires complete runs (m1); the review
# marker must name the trainer sha256 that is launched (m2). Starts after batch I.
#  J  v369 rate-distortion fine-tune (priority 4: the code made cheap to ENTROPY-code rather than fewer bits per dimension, since
#     "6 bits free" did not replicate). train_v368.py (= v365 + lattice join + own-q + entropy model/rate term), v293 static data,
#     warm start runs/v354-endpoint-long/long-s342202/checkpoints/final.pt, k-bit 8/8 wire, 36 epochs (21,312 updates), lr 1e-4,
#     max, posterior_only, seed 342202. Arms: lambda in {0, 0.003, 0.01, 0.03} (rate = bits per OBSERVED coordinate under the learned
#     factorised entropy model; unobserved blocks cost the 64-bit presence mask only). lambda=0 is the matched control (same trainer,
#     wire and entropy-model fitting; no rate gradient into the encoder).
#     Reported: RD points (model bits per observed coordinate, empirical entropies, R8/K0 wall F1, acc, FPR, own@R8; VAL and fresh).
#     ADMISSIBILITY (pre-registered): an arm is admissible if bits <= 0.8 x lambda0 bits AND R8 >= lambda0 - 1pp AND FPR <= lambda0 + 0.3pp
#     (VAL). Caveat stated in advance: single seed, run-to-run sd ~2 pp; the largest admissible lambda is replicated in batch K.
#  K  replicate the largest admissible lambda (and its lambda=0 control) on seed 342101 from runs/v354-endpoint-long/long-s342101.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day3_J2 started pid $$ sha256 $(sha256sum scripts/chain_day3_J2.sh | cut -c1-16)"
until grep -q "batch I done" $LOG; do sleep 120; done
TSHA=$(sha256sum scripts/train_v368.py | cut -d" " -f1)
grep -q "$TSHA" runs/v369-rate/REVIEW_OK 2>/dev/null || { log "REVIEW_OK missing or names a different trainer sha (have $TSHA): batch J NOT launched"; exit 0; }
log "batch J start"
START=$(date +%s); DE=$((START+6300)); DT=$((START+7100))
cat > runs/v369-rate/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "seed": 342202, "epochs": 36,
 "warm_start": "runs/v354-endpoint-long/long-s342202/checkpoints/final.pt", "lr": 1e-4, "wire": "kbit 8/8", "fusion": "max (v368 lattice join + own-q)", "readout": "posterior_only",
 "arms": {"l0": 0.0, "l003": 0.003, "l01": 0.01, "l03": 0.03}, "trainer_sha256": "$(sha256sum scripts/train_v368.py | cut -d' ' -f1)", "review": "$(cat runs/v369-rate/REVIEW_OK)",
 "admissibility": "bits <= 0.8 x l0 bits AND R8 >= l0 - 1pp AND FPR <= l0 + 0.3pp (VAL)"}
JSON
for arm in "l0 0" "l003 0.003" "l01 0.01" "l03 0.03"; do
  set -- $arm
  nohup $P scripts/train_v368.py --source $S --out runs/v369-rate/$1-s342202 --arm owned --size 128 --device cuda --fusion max --readout posterior_only --wire kbit --mu-bits 8 --q-bits 8 --rate-model --rate-lambda $2 --init-from runs/v354-endpoint-long/long-s342202/checkpoints/final.pt --epochs 36 --lr 1e-4 --val-every 6 --seed 342202 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v369-rate/$1-s342202.log 2>&1 &
  log "rate arm $1 (lambda $2) pid $!"
done
sleep 60
while pgrep -f "runs/v369-rate/l" >/dev/null; do sleep 60; done
log "batch J finished"
BEST=$(python3 - <<'PY'
import json, os
def g(a):
    try:
        d = json.load(open(f"runs/v369-rate/{a}-s342202/final.json"))
        if d.get("updates") != 21312: return None
        v = d["val"]; return v["rate"]["model_bits_per_observed_coordinate"], v["8"]["all"]["wall_f1"], v["8"]["all"]["unknown_known_fpr"]
    except Exception:
        return None
b0 = g("l0"); best = "none"; lam = {"l003": 0.003, "l01": 0.01, "l03": 0.03}
if b0:
    for a in ("l003", "l01", "l03"):
        r = g(a)
        if r and r[0] <= .8 * b0[0] and r[1] >= b0[1] - .01 and r[2] <= b0[2] + .003: best = a
print(best, lam.get(best, 0))
PY
)
ARM=${BEST% *}; LAM=${BEST#* }
log "batch J admissibility: largest admissible arm = $ARM (lambda $LAM)"
{ [ -z "$ARM" ] || [ "$ARM" = none ] || [ -z "$LAM" ]; } && { log "no admissible lambda (or incomplete runs); batch K not launched"; exit 0; }
START=$(date +%s); DE=$((START+6300)); DT=$((START+7100))
cat > runs/v369-rate/lease_K.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "seed": 342101, "warm_start": "runs/v354-endpoint-long/long-s342101/checkpoints/final.pt",
 "arms": {"l0": 0.0, "$ARM": $LAM}, "purpose": "replication of the largest admissible lambda from batch J on the second seed (same recipe)"}
JSON
for arm in "l0 0" "$ARM $LAM"; do
  set -- $arm
  nohup $P scripts/train_v368.py --source $S --out runs/v369-rate/$1-s342101 --arm owned --size 128 --device cuda --fusion max --readout posterior_only --wire kbit --mu-bits 8 --q-bits 8 --rate-model --rate-lambda $2 --init-from runs/v354-endpoint-long/long-s342101/checkpoints/final.pt --epochs 36 --lr 1e-4 --val-every 6 --seed 342101 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v369-rate/$1-s342101.log 2>&1 &
  log "rate replication arm $1 (lambda $2) seed 342101 pid $!"
done
sleep 60
while pgrep -f "runs/v369-rate/l.*-s342101" >/dev/null; do sleep 60; done
log "batch K finished"
