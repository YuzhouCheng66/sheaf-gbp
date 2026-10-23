#!/bin/bash
# Day-4 batch W (Claude, written 2026-10-23 18:30 UTC after batch T, before any batch-W run). Follow-up pre-registered in D5-2 / D5-5.
# Batch T: decoder width x2 beat the narrow decoder by +2.16 / +1.56 pp VAL R8 at the batch-Q budget (57,600 updates from scratch).
# Open question (CR386 reservation 7): does the capacity gain TRANSFER to the 3-cycle frontier (narrow: v383 cycle 2 = .883/.888,
# v384 cycle 3 = .893/.901), or is it only faster learning that longer training absorbs?
# Intervention: the SAME schedule the narrow decoder received -- warm start from the batch-T finals, cycle 2 (4 epochs, 57,600 updates,
# peak lr 1e-4), then cycle 3 (same, warm start from cycle 2) -- with train_v386.py --decoder-width 2.0. Everything else = batches R/S.
# Controls: cycle 2 -> v383 cont-s<seed> (narrow cycle 2, same seed); cycle 3 -> v384 c3-s<seed> (narrow cycle 3, same seed).
# Pre-registered reading (final VAL R8, 6-stratum equal weight, both seeds, after cycle 3):
#   WIDE3 >= v384 + 1 pp on BOTH seeds   => capacity transfers to the frontier: new HouseExpo best, wide decoder adopted;
#   |WIDE3 - v384| < 1 pp on BOTH seeds  => the batch-T gain was faster learning only; keep the narrow decoder (cheaper);
#   otherwise                            => ambiguous (keep narrow). Protection: FPR <= +0.3 pp. Reported: K0, fresh, 8+-room stratum,
#   and the cycle-2 comparison (WIDE2 vs v383) as a secondary, non-decisive reading.
# Scheduling: the wide decoder needs the GPU alone (batch T: 0.1435 s/update shared vs 0.0828 solo). Four runs, strictly sequential,
# each under its own inclusive 7,200 s lease (lease_c2_s<seed>.json, lease_c3_s<seed>.json); a run that misses its deadline makes the
# chain stop (no further cycle on an incomplete run) and the batch INCONCLUSIVE for that seed. This chain WAITS for batch U
# (chain_day4_U.sh / train_v388.py) to finish and never co-runs with it. Writes only under /mnt/datasets (runs/v391-wide-cycles).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_W started pid $$ sha256 $(sha256sum scripts/chain_day4_W.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v386.py | cut -d' ' -f1)
[ "$TSHA" = "77e8bc73895bd9765765a9afebc6fe701603e826b99a83c30da3b2b7dd44a068" ] || { log "ABORT: train_v386.py sha changed ($TSHA)"; exit 2; }
for seed in 342101 342202; do
  for f in runs/v386-decoder/wide-s$seed/checkpoints/final.pt runs/v386-decoder/wide-s$seed/final.json runs/v383-cont/cont-s$seed/final.json runs/v384-cycle3/c3-s$seed/final.json; do [ -f $f ] || { log "ABORT: missing $f"; exit 2; }; done
  $P -c "import json,sys; f=json.load(open('runs/v386-decoder/wide-s$seed/final.json')); sys.exit(0 if f.get('updates')==57600 else 1)" || { log "ABORT: batch-T run for $seed incomplete"; exit 2; }
done
ls -d runs/v391-wide-cycles/c2-s* >/dev/null 2>&1 && { log "ABORT: runs/v391-wide-cycles/c2-s* already exists (never overwrite history)"; exit 2; }
# wait for batch U (its chain and trainer); never co-run
while pgrep -f "chain_day4_U.sh" >/dev/null || pgrep -f "^$P scripts/train_v388.py " >/dev/null; do sleep 120; done
pgrep -f "^$P scripts/train_v3" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
log "batch W: batch U is gone; starting the wide-decoder cycles (sequential, one lease per run)"
mkdir -p runs/v391-wide-cycles
COMMON="--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 2.0"
run_cycle(){ # $1 seed, $2 cycle (2|3), $3 warm-start checkpoint, $4 control final.json
  local seed=$1 cyc=$2 warm=$3 ctrl=$4 out=runs/v391-wide-cycles/c$2-s$1
  local LS=$(date +%s); local LDE=$((LS+6600)); local LDT=$((LS+7150))
  cat > runs/v391-wide-cycles/lease_c${cyc}_s${seed}.json <<JSON
{"lease_start_epoch": $LS, "train_deadline_epoch": $LDE, "total_deadline_epoch": $LDT, "inclusive_seconds": 7200, "world": "houseexpo_fit", "updates_each": 57600,
 "seed": $seed, "cycle": $cyc, "warm_start": "$warm", "warm_start_sha256": "$(sha256sum $warm | cut -d' ' -f1)", "lr": 1e-4, "trainer": "scripts/train_v386.py", "trainer_sha256": "$TSHA", "decoder_width": 2.0,
 "control": "$ctrl (narrow decoder, same seed, same cycle)",
 "expectation": "after cycle 3: WIDE3 >= v384 + 1pp VAL R8 on both seeds => capacity transfers (new best); |diff| < 1pp on both => faster learning only (keep narrow); else ambiguous; incomplete => INCONCLUSIVE; FPR may not rise > 0.3pp"}
JSON
  log "batch W cycle $cyc seed $seed start (warm start $warm, lease lease_c${cyc}_s${seed}.json)"
  $P scripts/train_v386.py $COMMON --out $out --seed $seed --init-from $warm --deadline-epoch $LDE --deadline-total $LDT > $out.log 2>&1
  local upd=$($P -c "import json; print(json.load(open('$out/final.json')).get('updates', -1))" 2>/dev/null)
  log "batch W cycle $cyc seed $seed done (updates $upd)"
  [ "$upd" = "57600" ] || { log "batch W: cycle $cyc seed $seed incomplete -> INCONCLUSIVE for this seed; no further cycle"; return 1; }
  return 0
}
for seed in 342101 342202; do
  run_cycle $seed 2 runs/v386-decoder/wide-s$seed/checkpoints/final.pt runs/v383-cont/cont-s$seed/final.json || continue
  run_cycle $seed 3 runs/v391-wide-cycles/c2-s$seed/checkpoints/final.pt runs/v384-cycle3/c3-s$seed/final.json || continue
done
log "batch W done"
RES=$($P - <<'PY'
import json
out = {}
for cyc, ctrl in ((2, "runs/v383-cont/cont-s{s}"), (3, "runs/v384-cycle3/c3-s{s}")):
    g = []; fpr = []; ok = True
    for s in ("342101", "342202"):
        try:
            a = json.load(open(f"runs/v391-wide-cycles/c{cyc}-s{s}/final.json")); c = json.load(open(ctrl.format(s=s) + "/final.json"))
            if a.get("updates") != 57600: ok = False; continue
            g.append(round(100 * (a["val"]["8"]["all"]["wall_f1"] - c["val"]["8"]["all"]["wall_f1"]), 2))
            fpr.append(round(100 * (a["val"]["8"]["all"]["unknown_known_fpr"] - c["val"]["8"]["all"]["unknown_known_fpr"]), 2))
        except Exception as e:
            ok = False
    if ok and len(g) == 2:
        v = "capacity transfers (new best)" if min(g) >= 1 else ("faster learning only" if max(abs(x) for x in g) < 1 else "ambiguous")
        if max(fpr) > 0.3: v += " BUT FPR protection violated"
    else:
        v = "INCONCLUSIVE"
    out[f"cycle{cyc}"] = dict(verdict=v, gains_pp=g, fpr_delta_pp=fpr)
json.dump(out, open("runs/v391-wide-cycles/analysis.json", "w"), indent=1)
print(out["cycle3"], "| cycle2 (secondary):", out["cycle2"])
PY
)
log "batch W analysis: $RES"
