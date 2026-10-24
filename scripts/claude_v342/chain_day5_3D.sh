#!/bin/bash
# Day-5 batch 3D-A (Claude, written 2026-10-24 ~03:00 UTC before any 3D run): the first pre-registered 3D Sheaf-GBP batch.
# Question: does the owned-block Sheaf-GBP design (block-local code, max-precision join, bottleneck-only decoder) transfer from 2D
# grids to 3D occupancy volumes, at the user's requested code size of 256 means + 256 precisions, on synthetic 4-storey buildings
# (mapping_3d_data_v400.py) at 64^3 voxels (0.2 m)? This batch is a FEASIBILITY + BASELINE batch: one arm, two seeds, no control arm.
# Pre-registered reading (final VAL, 32 scenes, N = 8, metrics in target-known scope):
#   R8 surface F1 >= 0.80 AND K0 surface F1 >= 0.85 AND variable-structure F1 (skeleton of exterior walls + slabs removed from both
#   sides; review M2: painting only the skeleton scores surface F1 ~.70/.73) R8 >= 0.60 AND K0 >= 0.65 on both seeds
#                                                                     => the design transfers to 3D at 64^3 (2D synthetic walls: .94 / .95);
#   R8 surface F1 < 0.70 OR R8 variable F1 < 0.45 on either seed     => it does not transfer at this code density (1 scalar per 1,024
#                                                                      voxels); next lever is the block layout (8x8x8 = 512 coordinates
#                                                                      would need the user's agreement) or a different code semantics;
#   in between                                                        => partial; report and decide with the user.
#   Fusion check: R8 surface F1 >= K0 surface F1 - 3 pp on both seeds (max join loses little, as in 2D); unknown-known FPR <= 0.03.
#   Compression accounted (reported): raw code 512 B per agent per message (256 x 16 bit) vs gzip of the agent's own ternary 64^3 map.
# Budget (pre-registered rule): a 100-update CUDA preflight per seed (both in parallel) measures s/update; UPDATES is set to the largest
#   multiple of 500 such that 1.05 x (load + UPDATES x rate + 3 x val) <= 6,000 s, capped at 16,000 and floored at 4,000 (below the floor
#   the batch aborts INCONCLUSIVE); if two parallel jobs cannot reach the floor, seeds run SEQUENTIALLY, each under its own lease, with the
#   solo rate. Validation every UPDATES/4 updates. One inclusive 7,200 s lease per run.
# Scheduling: waits for batch V (chain_day4_V.sh / train_v392.py) to finish; never co-runs. Writes only under /mnt/datasets (runs/v400-3d).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v400/train64-k1600
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day5_3D started pid $$ sha256 $(sha256sum scripts/chain_day5_3D.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_3d_v400.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_data_v400.py | cut -d' ' -f1)
[ -f $CACHE/manifest.json ] || { log "ABORT: 3D cache manifest missing"; exit 2; }
$P -c "import json,sys; m=json.load(open('$CACHE/manifest.json')); sys.exit(0 if m['scenes']==1600 and m['size']==64 and m['data_module_sha256']=='$DSHA' else 1)" || { log "ABORT: cache does not match the data module / size / count"; exit 2; }
ls -d runs/v400-3d/a64-s* >/dev/null 2>&1 && { log "ABORT: runs/v400-3d/a64-s* already exists (never overwrite history)"; exit 2; }
while pgrep -f "chain_day4_V.sh" >/dev/null || pgrep -f "^$P scripts/train_v392.py " >/dev/null; do sleep 120; done
pgrep -f "^$P scripts/train_" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
log "batch 3D-A: batch V is gone; starting"
mkdir -p runs/v400-3d
rate(){ $P - "$1" <<'PY'
import json, re, sys
try:                                                                        # review M3: never raise; a crashed preflight yields "nan 0 0"
    log = open(sys.argv[1]).read()
    g = re.search(r' gen (\d+)s', log); t = [int(m) for m in re.findall(r'upd(?:50|100) .* (\d+)s\n', log)]
    h = json.load(open(sys.argv[1].replace('.log', '') + '/history.json'))
    val = max(30., float(h[0]['seconds']) - (t[-1] - int(g.group(1)))) if h else 60.   # seconds spent in the one validation
    print(f"{(t[1]-t[0])/50:.4f} {g.group(1)} {val:.0f}" if len(t) >= 2 and g else "nan 0 0")
except Exception:
    print("nan 0 0")
PY
}
TS=$(date -u +%H%M%S)
GPUUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${FORCE_SOLO:-0}" = "1" ] || [ "${GPUUSED:-99999}" -gt 4000 ]; then      # review M3: a parallel pair needs ~26 GB; another resident job => go solo; FORCE_SOLO=1 after the 05:00 UTC incident (parallel rate 1.2 s/upd vs 0.74 in the preflight)
  log "batch 3D-A: parallel skipped (FORCE_SOLO=${FORCE_SOLO:-0}, GPU holds ${GPUUSED} MiB) -> SOLO"; PLAN="SOLO forced=${FORCE_SOLO:-0} gpu_busy=${GPUUSED}MiB"
else
log "batch 3D-A preflight start (100 updates x 2 seeds in parallel, 64^3, D = 256)"
for seed in 342101 342202; do
  $P scripts/train_3d_v400.py --out runs/v400-3d/pf-par-$TS-s$seed --cache $CACHE --size 64 --device cuda --seed $seed --updates 16000 --max-updates 100 --val-every 100 --val-scenes 32 --deadline-total 0 > runs/v400-3d/pf-par-$TS-s$seed.log 2>&1 &
done
wait
read R1 L1 V1 <<<"$(rate runs/v400-3d/pf-par-$TS-s342101.log)"; read R2 L2 V2 <<<"$(rate runs/v400-3d/pf-par-$TS-s342202.log)"
log "batch 3D-A preflight (parallel): s/update $R1 / $R2, load $L1 / $L2 s, val $V1 / $V2 s"
PLAN=$($P - "$R1" "$R2" "$L1" "$L2" "$V1" "$V2" <<'PY'
import sys, math
r1, r2, l1, l2, v1, v2 = map(float, sys.argv[1:])
if not (math.isfinite(r1) and math.isfinite(r2) and r1 > 0 and r2 > 0): print("ABORT"); raise SystemExit
rate = max(r1, r2); load = max(l1, l2); val = max(v1, v2)
def fit(rate):
    u = int((6000 / 1.05 - load - 3 * val) / rate); return min(16000, (u // 500) * 500)
u = fit(rate)
print(f"PARALLEL {u} rate={rate:.4f} load={load:.0f} val={val:.0f}" if u >= 4000 else f"SOLO rate_par={rate:.4f} load={load:.0f} val={val:.0f}")
PY
)
fi
case "$PLAN" in ABORT*) log "batch 3D-A: parallel preflight invalid ($PLAN) -> falling back to SOLO (review M3)"; PLAN="SOLO parallel_invalid" ;; esac
log "batch 3D-A plan: $PLAN"
case "$PLAN" in
  PARALLEL*)
    UPD=$(echo "$PLAN" | cut -d' ' -f2); START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
    cat > runs/v400-3d/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v400 3D storeys", "size": 64, "D": 256, "updates_each": $UPD,
 "mode": "parallel", "preflight": "$PLAN", "trainer": "scripts/train_3d_v400.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "cache": "$CACHE",
 "expectation": "R8 surface F1 >= .80 and K0 >= .85 and variable-structure F1 R8 >= .60 and K0 >= .65 on both seeds => transfers to 3D; R8 surface < .70 or R8 variable < .45 on either => does not transfer at this density; fusion check R8 >= K0 - 3pp; FPR <= .03; incomplete => INCONCLUSIVE"}
JSON
    log "batch 3D-A start (parallel, $UPD updates each)"
    for seed in 342101 342202; do
      nohup $P scripts/train_3d_v400.py --out runs/v400-3d/a64-s$seed --cache $CACHE --size 64 --device cuda --seed $seed --updates $UPD --val-every $((UPD/4)) --val-scenes 32 --deadline-epoch $DE --deadline-total $DT > runs/v400-3d/a64-s$seed.log 2>&1 &
      log "a64 seed $seed pid $!"
    done
    sleep 60
    while pgrep -f "^$P scripts/train_3d_v400.py .*runs/v400-3d/a64" >/dev/null; do sleep 60; done ;;
  SOLO*)
    for seed in 342101 342202; do
      TS2=$(date -u +%H%M%S)
      $P scripts/train_3d_v400.py --out runs/v400-3d/pf-solo-$TS2-s$seed --cache $CACHE --size 64 --device cuda --seed $seed --updates 16000 --max-updates 100 --val-every 100 --val-scenes 32 --deadline-total 0 > runs/v400-3d/pf-solo-$TS2-s$seed.log 2>&1
      read RS LS VS <<<"$(rate runs/v400-3d/pf-solo-$TS2-s$seed.log)"
      UPD=$($P -c "import math; r=float('$RS'); u=int((6000/1.05-float('$LS')-3*float('$VS'))/r) if math.isfinite(r) and r>0 else 0; print(min(16000,(u//500)*500))")
      log "batch 3D-A solo preflight seed $seed: s/update $RS load $LS val $VS -> updates $UPD"
      [ "$UPD" -ge 4000 ] || { log "ABORT INCONCLUSIVE: seed $seed cannot reach 4,000 updates in a lease"; continue; }
      START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
      cat > runs/v400-3d/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v400 3D storeys", "size": 64, "D": 256, "updates_each": $UPD,
 "mode": "sequential", "seed": $seed, "preflight_parallel": "$PLAN", "preflight_solo": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v400.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "cache": "$CACHE",
 "expectation": "R8 surface F1 >= .80 and K0 >= .85 and variable-structure F1 R8 >= .60 and K0 >= .65 on both seeds => transfers to 3D; R8 surface < .70 or R8 variable < .45 on either => does not transfer at this density; fusion check R8 >= K0 - 3pp; FPR <= .03; incomplete => INCONCLUSIVE"}
JSON
      log "batch 3D-A start seed $seed (sequential, $UPD updates)"
      $P scripts/train_3d_v400.py --out runs/v400-3d/a64-s$seed --cache $CACHE --size 64 --device cuda --seed $seed --updates $UPD --val-every $((UPD/4)) --val-scenes 32 --deadline-epoch $DE --deadline-total $DT > runs/v400-3d/a64-s$seed.log 2>&1
      log "a64 seed $seed done"
    done ;;
  *) log "ABORT: unexpected plan ($PLAN)"; exit 2 ;;
esac
log "batch 3D-A done"
RES=$($P - <<'PY'
import json
rows = {}
for s in ("342101", "342202"):
    try:
        f = json.load(open(f"runs/v400-3d/a64-s{s}/final.json")); m = json.load(open(f"runs/v400-3d/a64-s{s}/run.json"))
        if f["updates"] != m["updates"]: rows[s] = "INCONCLUSIVE (incomplete)"; continue
        v = f["val"]; rows[s] = dict(K0_surface=round(v["0"]["surface_f1"], 4), R8_surface=round(v["8"]["surface_f1"], 4), K0_var=round(v["0"]["var_occ_f1"], 4), R8_var=round(v["8"]["var_occ_f1"], 4), K0_occ=round(v["0"]["occ_f1"], 4), R8_occ=round(v["8"]["occ_f1"], 4), R8_acc=round(v["8"]["accuracy"], 4), R8_fpr=round(v["8"]["unknown_known_fpr"], 4), updates=f["updates"])
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
ok = all(isinstance(r, dict) for r in rows.values())
if ok:
    k0 = [r["K0_surface"] for r in rows.values()]; r8 = [r["R8_surface"] for r in rows.values()]
    k0v = [r["K0_var"] for r in rows.values()]; r8v = [r["R8_var"] for r in rows.values()]
    # review M2: surface F1 alone is not discriminating (painting the world-independent skeleton scores ~.70/.73); the variable-structure
    # F1 (skeleton removed) must clear its own bar: R8 >= .60 and K0 >= .65
    verdict = "transfers to 3D at 64^3" if min(r8) >= .80 and min(k0) >= .85 and min(r8v) >= .60 and min(k0v) >= .65 else ("does not transfer at this density" if min(r8) < .70 or min(r8v) < .45 else "partial")
    fusion_ok = all(r["R8_surface"] >= r["K0_surface"] - .03 for r in rows.values()); fpr_ok = all(r["R8_fpr"] <= .03 for r in rows.values())
else:
    verdict = "INCONCLUSIVE"; fusion_ok = fpr_ok = None
json.dump(dict(rows=rows, verdict=verdict, fusion_ok=fusion_ok, fpr_ok=fpr_ok), open("runs/v400-3d/analysis.json", "w"), indent=1)
print(verdict, "| fusion_ok", fusion_ok, "| fpr_ok", fpr_ok, "|", rows)
PY
)
log "batch 3D-A analysis: $RES"
