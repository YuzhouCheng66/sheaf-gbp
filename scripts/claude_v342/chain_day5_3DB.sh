#!/bin/bash
# Day-5 batch 3D-B (Claude, written 2026-10-24 ~12:30 UTC after batch 3D-A, before any 128^3 run). Pre-registered in D6-2.
# Question: does the FIXED code (256 means + 256 precisions = 512 B per message, one scalar per 8x8x4 public block) still decode the
# building when the map resolution doubles to 128^3 (0.1 m voxels, 8,192 voxels per coordinate instead of 1,024)?
# One variable: the voxel size, with the sensor's angular density matched to the voxel (beams x4 = 65,536, so the ray spacing at 5 m stays
# below the voxel; the measured porosity of the private maps does not increase). Same worlds, same agent positions / radio forest / reached
# sets as batch 3D-A (v401 data module, placement on the 0.2 m reference raster), same code semantics, same join, same loss and metrics,
# the same architecture and schedule on the 64^3 compute grid (v401: every convolution runs at 64^3; the voxel grid is reached by
# parameter-free space-to-depth / depth-to-space reshapes; the first / last convolutions widen to 27 in / 16 out channels, +18,536
# parameters, 1,143,403 vs 1,124,867, so the initial weights differ from 3D-A for the same seed; at 64^3 the model is v400 bit-for-bit).
# No control arm is run: the reference is batch 3D-A (runs/v400-3d/a64-s<seed>, 16,000 updates each), same seeds.
# Covariates recorded in advance (review v401 M2): per-agent known fraction .042 -> .035 (pooled back to 64^3: .039), R8 union .265 -> .226,
# 6-surface share of occupied voxels .79 -> .51 -- a difference in native surface F1 from 3D-A is therefore not attributable to the code alone.
# Pre-registered reading (final VAL, 32 scenes, N = 8, target-known scope, BOTH seeds), NATIVE 128^3 metrics:
#   R8 surface F1 >= .85 AND K0 surface F1 >= .88 AND variable-structure F1 R8 >= .80 AND K0 >= .84  => the 512 B code SCALES to 128^3
#                                                                                                     (within ~6 pp of 3D-A at twice the resolution);
#   R8 surface F1 < .75 OR R8 variable F1 < .65 on either seed                                     => it does NOT scale under the fixed-compute
#                                                                                                     block readout; the cause (512 B code vs the
#                                                                                                     27-feature -> 16-logit linear readout per
#                                                                                                     0.2 m cell) is NOT identified by this batch
#                                                                                                     and needs a decoder-capacity control (review
#                                                                                                     S8) before the code is touched;
#   in between                                                                                      => partial; report and decide with the user.
#   Fusion check: R8 surface F1 >= K0 surface F1 - 3 pp on both seeds; unknown-known FPR <= .03.
#   Secondary readings (reported, NOT part of the verdict, no thresholds): (i) the TOLERANCE-1 surface F1 at 128^3 (review M1 definition:
#   predicted occupied voxels within one voxel of the target-known region, scored against target surface voxels with a one-voxel
#   6-neighbourhood tolerance both ways), read BOTH ways: tol-1 R8 close to 3D-A's native R8 (.916 / .927) or to 3D-A's own tol-1 R8
#   (.958 / .963, post-hoc, runs/v401-3d128/posthoc) while the native 128^3 number is lower => the residual is +-1-voxel jitter at the
#   finer scale; tol-1 clearly lower as well => structure is missing;
#   (ii) the same metrics after pooling prediction, target and skeleton to 64^3 by the ternary union over each 2x2x2 block ("pooled64").
#   Caveat recorded in advance: the any-pooled 0.1 m raster is THICKER than the native 0.2 m raster (occupied fraction .21 -> .30 on
#   TRAIN 7), so pooled64 is a different target from 3D-A's and is never compared one-to-one with it.
#   Budget: UPDATES comes from the preflight (rule below). 16,000 is NOT reachable under the rule (it needs <= 0.33 s/update; the review's
#   4080 probe predicts 0.36-0.40 s/update on the 5090 => 14,000-15,000), so the comparison is <= 15,000 vs 16,000 updates by design,
#   accepted because 3D-A's VAL curve was flat from 12,000 to 16,000; below 12,000 the result is reported as budget-confounded.
# Budget (pre-registered rule, as 3D-A): SOLO only (never co-runs with another trainer), seeds strictly sequential, each under its own
#   inclusive 7,200 s lease; a 100-update CUDA preflight per seed (under a 1,800 s timeout, review S1) measures s/update; UPDATES = largest
#   multiple of 500 such that 1.05 x (load + UPDATES x rate + 3 x val) <= 6,000 s, capped at 16,000, floored at 4,000 (below the floor that
#   seed is INCONCLUSIVE). The run evaluates 6 times (4 in-loop + final-val + final-fresh); the 3 x val not in the formula (< 2 min at 128^3)
#   is covered by the 1.05 margin and the 6,600 / 7,150 s deadlines, exactly as in 3D-A (review S2).
#   Validation every UPDATES/4 updates. Train deadline = start + 6,600 s; total deadline = start + 7,150 s.
# Scheduling: waits for batch X (chain_day5_X.sh / train_v392.py) to finish. Writes only under /mnt/datasets (runs/v401-3d128).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v401/train128-k1600
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day5_3DB started pid $$ sha256 $(sha256sum scripts/chain_day5_3DB.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_3d_v401.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_data_v401.py | cut -d' ' -f1)
[ -f $CACHE/manifest.json ] || { log "ABORT: 3D-B cache manifest missing"; exit 2; }
$P -c "import json,sys; m=json.load(open('$CACHE/manifest.json')); sys.exit(0 if m['scenes']==1600 and m['size']==128 and m['version']==401 and m['beams']==65536 and m['data_module_sha256']=='$DSHA' else 1)" || { log "ABORT: cache does not match the data module / size / count / beams"; exit 2; }
for s in 342101 342202; do $P -c "import json,sys; f=json.load(open('runs/v400-3d/a64-s$s/final.json')); sys.exit(0 if f.get('updates')==16000 and 'val' in f else 1)" || { log "ABORT: 3D-A reference a64-s$s incomplete"; exit 2; }; done
ls -d runs/v401-3d128/b128-s* >/dev/null 2>&1 && { log "ABORT: runs/v401-3d128/b128-s* already exists (never overwrite history)"; exit 2; }
while pgrep -f "chain_day5_X[.]sh" >/dev/null || pgrep -f "^$P scripts/train_v392.py " >/dev/null; do sleep 120; done
pgrep -f "^$P scripts/train_" >/dev/null && { log "ABORT: a trainer process is already running"; exit 2; }
log "batch 3D-B: batch X is gone; starting (SOLO, sequential, one lease per seed)"
mkdir -p runs/v401-3d128
rate(){ $P - "$1" <<'PY'
import json, re, sys
try:                                                                        # never raise; a crashed preflight yields "nan 0 0"
    log = open(sys.argv[1]).read()
    g = re.search(r' gen (\d+)s', log); t = [int(m) for m in re.findall(r'upd(?:50|100) .* (\d+)s\n', log)]
    h = json.load(open(sys.argv[1].replace('.log', '') + '/history.json'))
    val = max(30., float(h[0]['seconds']) - (t[-1] - int(g.group(1)))) if h else 60.   # seconds spent in the one validation
    print(f"{(t[1]-t[0])/50:.4f} {g.group(1)} {val:.0f}" if len(t) >= 2 and g else "nan 0 0")
except Exception:
    print("nan 0 0")
PY
}
for seed in 342101 342202; do
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v401.py --out runs/v401-3d128/pf-solo-$TS2-s$seed --cache $CACHE --size 128 --device cuda --seed $seed --updates 16000 --max-updates 100 --val-every 100 --val-scenes 32 --deadline-total 0 --gen-workers 8 > runs/v401-3d128/pf-solo-$TS2-s$seed.log 2>&1
  read RS LS VS <<<"$(rate runs/v401-3d128/pf-solo-$TS2-s$seed.log)"
  UPD=$($P -c "import math; r=float('$RS'); u=int((6000/1.05-float('$LS')-3*float('$VS'))/r) if math.isfinite(r) and r>0 else 0; print(min(16000,(u//500)*500))")
  log "batch 3D-B solo preflight seed $seed: s/update $RS load $LS val $VS -> updates $UPD"
  [ "$UPD" -ge 4000 ] || { log "ABORT INCONCLUSIVE: seed $seed cannot reach 4,000 updates in a lease"; continue; }
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > runs/v401-3d128/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v401 3D storeys (v400 worlds)", "size": 128, "D": 256, "beams": 65536, "updates_each": $UPD,
 "mode": "sequential", "seed": $seed, "preflight_solo": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v401.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "cache": "$CACHE",
 "reference": "runs/v400-3d/a64-s$seed (batch 3D-A, 64^3, 16000 updates, same worlds / positions / forest)",
 "expectation": "native 128^3 VAL: R8 surface F1 >= .85 and K0 >= .88 and variable-structure F1 R8 >= .80 and K0 >= .84 on both seeds => the 512 B code scales to 128^3; R8 surface < .75 or R8 variable < .65 on either => does not scale under the fixed-compute block readout (cause not identified: needs a decoder-capacity control); between => partial; fusion check R8 >= K0 - 3pp; FPR <= .03; reported only: tolerance-1 surface F1 and pooled64 metrics (no thresholds); UPDATES < 12000 => budget-confounded; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3D-B start seed $seed (sequential, $UPD updates)"
  $P scripts/train_3d_v401.py --out runs/v401-3d128/b128-s$seed --cache $CACHE --size 128 --device cuda --seed $seed --updates $UPD --val-every $((UPD/4)) --val-scenes 32 --deadline-epoch $DE --deadline-total $DT --gen-workers 8 > runs/v401-3d128/b128-s$seed.log 2>&1
  log "b128 seed $seed done"
done
log "batch 3D-B done"
RES=$($P - <<'PY'
import json
rows = {}
for s in ("342101", "342202"):
    try:
        f = json.load(open(f"runs/v401-3d128/b128-s{s}/final.json")); m = json.load(open(f"runs/v401-3d128/b128-s{s}/run.json"))
        ref = json.load(open(f"runs/v400-3d/a64-s{s}/final.json"))["val"]
        if f["updates"] != m["updates"]: rows[s] = "INCONCLUSIVE (incomplete)"; continue
        v = f["val"]; p = f["val_pooled64"]
        rows[s] = dict(K0_surface=round(v["0"]["surface_f1"], 4), R8_surface=round(v["8"]["surface_f1"], 4), K0_var=round(v["0"]["var_occ_f1"], 4), R8_var=round(v["8"]["var_occ_f1"], 4),
                       K0_occ=round(v["0"]["occ_f1"], 4), R8_occ=round(v["8"]["occ_f1"], 4), R8_acc=round(v["8"]["accuracy"], 4), R8_fpr=round(v["8"]["unknown_known_fpr"], 4),
                       R8_surface_tol1=round(v["8"]["surface_f1_tol1"], 4), K0_surface_tol1=round(v["0"]["surface_f1_tol1"], 4),
                       pooled_R8_surface=round(p["8"]["surface_f1"], 4), pooled_R8_var=round(p["8"]["var_occ_f1"], 4), pooled_K0_surface=round(p["0"]["surface_f1"], 4),
                       ref64_R8_surface=round(ref["8"]["surface_f1"], 4), ref64_R8_var=round(ref["8"]["var_occ_f1"], 4), ref64_K0_surface=round(ref["0"]["surface_f1"], 4),
                       native_minus_ref_R8_surface_pp=round(100 * (v["8"]["surface_f1"] - ref["8"]["surface_f1"]), 2), tol1_minus_ref_R8_surface_pp=round(100 * (v["8"]["surface_f1_tol1"] - ref["8"]["surface_f1"]), 2),
                       updates=f["updates"], budget_confounded=f["updates"] < 12000)
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
ok = all(isinstance(r, dict) for r in rows.values())
if ok:
    k0 = [r["K0_surface"] for r in rows.values()]; r8 = [r["R8_surface"] for r in rows.values()]
    k0v = [r["K0_var"] for r in rows.values()]; r8v = [r["R8_var"] for r in rows.values()]
    verdict = "the 512 B code scales to 128^3" if min(r8) >= .85 and min(k0) >= .88 and min(r8v) >= .80 and min(k0v) >= .84 else ("does not scale under the block readout (cause not identified)" if min(r8) < .75 or min(r8v) < .65 else "partial")
    fusion_ok = all(r["R8_surface"] >= r["K0_surface"] - .03 for r in rows.values()); fpr_ok = all(r["R8_fpr"] <= .03 for r in rows.values())
    confounded = any(r["budget_confounded"] for r in rows.values())
else:
    verdict = "INCONCLUSIVE"; fusion_ok = fpr_ok = confounded = None
json.dump(dict(rows=rows, verdict=verdict, fusion_ok=fusion_ok, fpr_ok=fpr_ok, budget_confounded=confounded), open("runs/v401-3d128/analysis.json", "w"), indent=1)
print(verdict, "| fusion_ok", fusion_ok, "| fpr_ok", fpr_ok, "| budget_confounded", confounded, "|", rows)
PY
)
log "batch 3D-B analysis: $RES"
