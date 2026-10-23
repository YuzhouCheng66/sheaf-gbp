#!/bin/bash
# Day-4 batch S (Claude, written 2026-10-23 13:10 UTC after batch R, before any batch-S run). HANDOFF_20261023 section 6 step 1.
# Batch R (second cosine cycle, lr 1e-4, warm start from Q) beat Q by +2.42 / +2.43 pp VAL R8 -> training still not saturated.
# S = a THIRD cosine cycle: warm start from the batch-R final weights (runs/v383-cont/cont-s<seed>/checkpoints/final.pt), 4 epochs
# (57,600 updates) at peak lr 1e-4 (IDENTICAL to R: same lr, same data, same chart, same trainer code path train_v378.py sha
# db7bcc35..., same seeds). The ONLY variable relative to batch R is the warm-start point (R final instead of Q final); Adam state and
# the 200-update warm-up + cosine schedule restart exactly as in R (--init-from loads weights only). Declared as a three-cycle schedule.
# Control = the batch-R final weights (the warm start itself; VAL R8 .8825 / .8878, K0 .9067 / .9137).
# Pre-registered reading (final VAL R8 wall F1, 6-stratum equal weight, final weights, both seeds):
#   S >= R + 1 pp on BOTH seeds       => more training still helps (queue a 4th cycle or move to the decoder-size question);
#   |S - R| < 1 pp on BOTH seeds      => converged for this model size at this lr -> next lever is the decoder (architecture change);
#   otherwise                         => ambiguous -> keep R (simpler), do not chain further cycles on this evidence.
#   Secondary (reported, not decisive): K0, fresh R8/K0, 8+-room stratum, FPR must not rise by > 0.3 pp (protection rule).
#   updates != 57,600 on either seed  => INCONCLUSIVE (no verdict).
# Lease: one inclusive 7,200 s lease (train deadline +6,600 s, total +7,150 s); R needed 5,331 / 5,378 s with two jobs in parallel.
# Writes only under /mnt/datasets (runs/v384-cycle3). No other job on the 5090 (checked 13:06 UTC: 119 MiB, 0 %).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_S started pid $$ sha256 $(sha256sum scripts/chain_day4_S.sh | cut -c1-16)"
# Guards (pre-launch): trainer unchanged since R, warm starts present, GPU free of our jobs, no earlier S output.
TSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
[ "$TSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py sha changed ($TSHA)"; exit 2; }
for seed in 342101 342202; do [ -f runs/v383-cont/cont-s$seed/checkpoints/final.pt ] || { log "ABORT: missing warm start for $seed"; exit 2; }; done
# review S1: the warm starts must be exactly the batch-R finals verified at 13:06 UTC (sha256 recorded in the lease below)
[ "$(sha256sum runs/v383-cont/cont-s342101/checkpoints/final.pt | cut -d' ' -f1)" = "27bc72e7dca753dd4351f82960fdf6ecd285324e77071f4e7703eb2c7d2e3071" ] || { log "ABORT: warm start sha mismatch (342101)"; exit 2; }
[ "$(sha256sum runs/v383-cont/cont-s342202/checkpoints/final.pt | cut -d' ' -f1)" = "da0f00f552b38d91470df000e10436fbd2af0984404094f4b1add728a5c78a6b" ] || { log "ABORT: warm start sha mismatch (342202)"; exit 2; }
[ -d runs/v384-cycle3 ] && { log "ABORT: runs/v384-cycle3 already exists (never overwrite history)"; exit 2; }
# review S2 (and the 13:12 UTC false abort): anchor to the real trainer argv so an ssh wrapper / editor / grep mentioning the name cannot match
pgrep -f "^$P scripts/train_v378.py " >/dev/null && { log "ABORT: a train_v378.py process is already running"; exit 2; }
mkdir -p runs/v384-cycle3
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v384-cycle3/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "warm_start": "runs/v383-cont/cont-s<seed>/checkpoints/final.pt", "warm_start_sha256": {"342101": "27bc72e7dca753dd4351f82960fdf6ecd285324e77071f4e7703eb2c7d2e3071", "342202": "da0f00f552b38d91470df000e10436fbd2af0984404094f4b1add728a5c78a6b"},
 "lr": 1e-4, "trainer_sha256": "$TSHA", "control": "runs/v383-cont/cont-s<seed>/final.json (batch R final)",
 "expectation": "S >= R + 1pp final VAL R8 on both seeds => more training helps; |gain| < 1pp on both => converged at this size/lr; else ambiguous (keep R); incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
log "batch S start (third cosine cycle, lr 1e-4, warm start from batch R final)"
for seed in 342101 342202; do
  nohup $P scripts/train_v378.py --source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --out runs/v384-cycle3/c3-s$seed --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --seed $seed --init-from runs/v383-cont/cont-s$seed/checkpoints/final.pt --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 > runs/v384-cycle3/c3-s$seed.log 2>&1 &
  log "cycle3 seed $seed pid $!"
done
sleep 60
while pgrep -f "^$P scripts/train_v378.py .*runs/v384-cycle3/c3" >/dev/null; do sleep 60; done
log "batch S done"
RES=$(python3 - <<'PY'
import json
g, fpr = [], []
for s in ("342101", "342202"):
    try:
        a = json.load(open(f"runs/v384-cycle3/c3-s{s}/final.json")); c = json.load(open(f"runs/v383-cont/cont-s{s}/final.json"))
        if a.get("updates") != 57600: print("INCONCLUSIVE"); raise SystemExit
        g.append(100 * (a["val"]["8"]["all"]["wall_f1"] - c["val"]["8"]["all"]["wall_f1"]))
        fpr.append(100 * (a["val"]["8"]["all"]["unknown_known_fpr"] - c["val"]["8"]["all"]["unknown_known_fpr"]))
    except SystemExit: raise
    except Exception as e: print("INCONCLUSIVE", e); raise SystemExit
v = "more training helps" if min(g) >= 1 else ("converged" if max(abs(x) for x in g) < 1 else "ambiguous")
if max(fpr) > 0.3: v += " BUT FPR protection violated"
print(v, "R8 gain pp", [round(x, 2) for x in g], "FPR delta pp", [round(x, 2) for x in fpr])
PY
)
log "batch S analysis: $RES"
