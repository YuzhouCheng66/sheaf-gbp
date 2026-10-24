#!/bin/bash
# Day-5 batch X (Claude, written 2026-10-24 ~05:00 UTC after batch V, before any batch-X run). Follow-up pre-registered in D5-10 / D5-11.
# Batch V: public-frame conditioning on the wide decoder beat batch T by +1.82 / +0.99 pp VAL R8 at the batch-Q budget (ambiguous by the
# letter, but the gain rises monotonically with chart size on both seeds). Question: does it carry to the 3-cycle frontier, i.e. beat the
# wide decoder's own cycles (batch W, v391: cycle 2 .899/.899, cycle 3 .905/.909)?
# Intervention: the SAME schedule -- warm start from the batch-V finals, cycle 2 (4 epochs, 57,600 updates, peak lr 1e-4), then cycle 3 --
# with train_v392.py --decoder-width 2.0 --frame-cond. Controls: cycle 2 -> v391 c2-s<seed>, cycle 3 -> v391 c3-s<seed> (same seed).
# Pre-registered reading (final VAL R8, 6-stratum equal weight, both seeds, after cycle 3):
#   FRAME3 >= v391 c3 + 1 pp on BOTH seeds  => conditioning transfers to the frontier: new HouseExpo best, adopted;
#   |FRAME3 - v391 c3| < 1 pp on BOTH seeds => faster learning only; keep the unconditioned wide decoder;
#   otherwise                               => ambiguous (keep). Protection: FPR <= +0.3 pp. Reported: K0, fresh, 8+-room stratum, cycle-2 comparison.
# Scheduling: wide decoder => solo runs; four runs strictly sequential, each under its own inclusive 7,200 s lease; an incomplete cycle stops
# that SEED (the other seed still runs). Waits for the 3D batch (chain_day5_3D.sh / train_3d_v400.py) and never co-runs with any trainer.
# Writes only under /mnt/datasets (runs/v393-frame-cycles).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day5_X started pid $$ sha256 $(sha256sum scripts/chain_day5_X.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v392.py | cut -d' ' -f1)
[ "$TSHA" = "ef4dc60082ef9f5f7076cf6bec4b7173b987ee730dd5f63f26a8321d40db00d2" ] || { log "ABORT: train_v392.py sha changed ($TSHA)"; exit 2; }
for seed in 342101 342202; do
  for f in runs/v392-frame/frame-s$seed/checkpoints/final.pt runs/v392-frame/frame-s$seed/final.json runs/v391-wide-cycles/c2-s$seed/final.json runs/v391-wide-cycles/c3-s$seed/final.json; do [ -f $f ] || { log "ABORT: missing $f"; exit 2; }; done
  $P -c "import json,sys; f=json.load(open('runs/v392-frame/frame-s$seed/final.json')); sys.exit(0 if f.get('updates')==57600 else 1)" || { log "ABORT: batch-V run for $seed incomplete"; exit 2; }
done
ls -d runs/v393-frame-cycles/c2-s* >/dev/null 2>&1 && { log "ABORT: runs/v393-frame-cycles/c2-s* already exists (never overwrite history)"; exit 2; }
# wait for the 3D batch (its chain and trainer); never co-run
while pgrep -f "chain_day5_3D.sh" >/dev/null || pgrep -f "^$P scripts/train_3d_v400.py " >/dev/null; do sleep 120; done
pgrep -f "^$P scripts/train_" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
log "batch X: the 3D batch is gone; starting the wide+frame cycles (sequential, one lease per run)"
mkdir -p runs/v393-frame-cycles
COMMON="--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 2.0 --frame-cond"
run_cycle(){ # $1 seed, $2 cycle (2|3), $3 warm-start checkpoint, $4 control final.json
  local seed=$1 cyc=$2 warm=$3 ctrl=$4 out=runs/v393-frame-cycles/c$2-s$1
  local LS=$(date +%s); local LDE=$((LS+6600)); local LDT=$((LS+7150))
  cat > runs/v393-frame-cycles/lease_c${cyc}_s${seed}.json <<JSON
{"lease_start_epoch": $LS, "train_deadline_epoch": $LDE, "total_deadline_epoch": $LDT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "seed": $seed, "cycle": $cyc, "warm_start": "$warm", "warm_start_sha256": "$(sha256sum $warm | cut -d' ' -f1)", "lr": 1e-4, "trainer": "scripts/train_v392.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0, "frame_cond": true,
 "control": "$ctrl (wide decoder without frame conditioning, same seed, same cycle)",
 "expectation": "after cycle 3: FRAME3 >= v391 c3 + 1pp VAL R8 on both seeds => conditioning transfers (new best); |diff| < 1pp on both => faster learning only (keep wide); else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
  log "batch X cycle $cyc seed $seed start (warm start $warm, lease lease_c${cyc}_s${seed}.json)"
  $P scripts/train_v392.py $COMMON --out $out --seed $seed --init-from $warm --deadline-epoch $LDE --deadline-total $LDT > $out.log 2>&1
  local upd=$($P -c "import json; print(json.load(open('$out/final.json')).get('updates', -1))" 2>/dev/null)
  log "batch X cycle $cyc seed $seed done (updates $upd)"
  [ "$upd" = "57600" ] || { log "batch X: cycle $cyc seed $seed incomplete -> INCONCLUSIVE for this seed; no further cycle"; return 1; }
  return 0
}
for seed in 342101 342202; do
  run_cycle $seed 2 runs/v392-frame/frame-s$seed/checkpoints/final.pt runs/v391-wide-cycles/c2-s$seed/final.json || continue
  run_cycle $seed 3 runs/v393-frame-cycles/c2-s$seed/checkpoints/final.pt runs/v391-wide-cycles/c3-s$seed/final.json || continue
done
log "batch X done"
RES=$($P - <<'PY'
import json
out = {}
for cyc, ctrl in ((2, "runs/v391-wide-cycles/c2-s{s}"), (3, "runs/v391-wide-cycles/c3-s{s}")):
    g = []; fpr = []; ok = True
    for s in ("342101", "342202"):
        try:
            a = json.load(open(f"runs/v393-frame-cycles/c{cyc}-s{s}/final.json")); c = json.load(open(ctrl.format(s=s) + "/final.json"))
            if a.get("updates") != 57600: ok = False; continue
            g.append(100 * (a["val"]["8"]["all"]["wall_f1"] - c["val"]["8"]["all"]["wall_f1"]))
            fpr.append(100 * (a["val"]["8"]["all"]["unknown_known_fpr"] - c["val"]["8"]["all"]["unknown_known_fpr"]))
        except Exception as e:
            ok = False
    if ok and len(g) == 2:
        v = "conditioning transfers (new best)" if min(g) >= 1 else ("faster learning only" if max(abs(x) for x in g) < 1 else "ambiguous")
        if max(fpr) > 0.3: v += " BUT FPR protection violated"
    else:
        v = "INCONCLUSIVE"
    out[f"cycle{cyc}"] = dict(verdict=v, gains_pp=[round(x, 3) for x in g], fpr_delta_pp=[round(x, 3) for x in fpr])
json.dump(out, open("runs/v393-frame-cycles/analysis.json", "w"), indent=1)
print(out["cycle3"], "| cycle2 (secondary):", out["cycle2"])
PY
)
log "batch X analysis: $RES"
