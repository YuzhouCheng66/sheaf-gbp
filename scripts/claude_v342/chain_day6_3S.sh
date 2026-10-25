#!/bin/bash
# Day-6 batch 3S-A (Claude, written 2026-10-25 ~01:40 UTC, revised ~03:00 UTC after code_review_v420_storey.md M1-M3, before any v420
# training). Pre-registered in D8-0 / D8-1; design claude_handoff/DESIGN_3D_STOREY_V420.md.
# Question: does Sheaf-GBP (256 Gaussian variables + 256 diagonal precisions per agent = 512 B per message, block-local encoder, max-
# precision join over 8 rounds on the radio forest, bottleneck-only decoder) map FURNISHED single storeys -- HouseExpo layouts extruded to
# 3D with slabs, door lintels and a parametric furniture library whose footprint share (median .30 after the M3 fixes) matches the
# Replica scans (.30) --
# on the slab grid R1 = 128 x 128 x 32 (house-fitted chart L x L x 3.2 m; cells L/128 horizontally, 0.1 m vertically)?
# This is a FEASIBILITY + BASELINE batch for the new world (one arm, two seeds), as 3D-A was for the cube world.
# Pre-registered reading (final VAL: 24 HouseExpo VALIDATION houses, 8 per stratum, N = 8, target-known scope, BOTH seeds):
#   STRUCTURE: R8 surface F1 >= .80 AND K0 >= .85 AND variable-structure F1 (floor and ceiling slabs removed) R8 >= .65 AND K0 >= .70;
#   FURNITURE (review v420 M1): furniture-region F1 (F1 on the voxels that are neither slab nor wall / lintel, the structure dilated by
#   one voxel; only furniture can be occupied there) R8 >= .50 AND K0 >= .55;
#   structure AND furniture on both seeds                                 => Sheaf-GBP maps furnished storeys at R1 (baseline for 3S-B/C/D);
#   structure on both seeds but not furniture                             => maps the storey structure, not its furniture (partial);
#   R8 surface F1 < .70 OR R8 variable-structure F1 < .50 on either seed => does not with this model -- code density (1 scalar per 2,048
#                                                                           voxels), the block readout / two-stage decoder and the new
#                                                                           world are not separated by this batch;
#   otherwise                                                             => partial; report and decide with the user.
#   Fusion check: R8 surface F1 >= K0 surface F1 - 3 pp on both seeds; unknown-known FPR <= .03.
#   Reported, not judged: tolerance-1 surface F1, furniture recall, occupied F1, accuracy, the fresh panel (another 24 VAL houses),
#   per-stratum numbers, and the compression ratio (512 B vs the 2-bit raw own map, 128 KB, and vs its gzip).
#   References on this very panel (ref_3d_v420.py, runs/v420-storey/reference_predictors.json; computed before any training):
#   slab-only K0 / R8 surface F1 .625 / .585, variable-structure .000, furniture-region .000; a perfect structure-only mapper with NO
#   furniture K0 / R8 surface .915 / .900, variable-structure .813 / .814, furniture-region .000 (it passes every structure threshold,
#   hence the furniture criterion); furniture is .180 of the R8 target surface (fresh .155).
# Cache: TRAIN houses whose 8 agents cannot be placed >= 0.5 m apart are skipped (about 4 %, review S1); >= 4,500 scenes are required.
# Budget rule (as 3D-A / 3D-B): SOLO, seeds strictly sequential, each under its own inclusive 7,200 s lease; a 100-update CUDA preflight
#   per seed (timeout 1,800 s) measures s/update; UPDATES = largest multiple of 500 with 1.05 x (load + UPDATES x rate + 3 x val) <= 6,000 s,
#   capped at 16,000, floored at 4,000 (below: INCONCLUSIVE); validation every UPDATES/4; every run under a hard timeout at the lease's
#   total deadline (incident D6-3). Waits for batch RS-A. Writes only under /mnt/datasets (runs/v420-storey).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v420-storey
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3S started pid $$ sha256 $(sha256sum scripts/chain_day6_3S.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_3d_v420.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1)
[ "$TSHA" = "c39b71019f61ae564991122b2aa215f350a8873fab30fbf4d1ecdf98564c4973" ] || { log "ABORT: train_3d_v420.py sha changed ($TSHA)"; exit 2; }
[ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_s*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-A history (never overwrite)"; exit 2; fi   # review M2
while pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v420[.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null; do sleep 120; done   # review S5: also waits for the cache
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
$P -c "import json,sys; m=json.load(open('$CACHE/manifest.json')); sys.exit(0 if m['dims']==[128,128,32] and m['version']==420 and m['scenes']>=4500 and m['data_module_sha256']=='$DSHA' else 1)" || { log "ABORT: cache does not match the data module / dims / count"; exit 2; }
MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
log "batch 3S-A: GPU free; starting (SOLO, sequential, one lease per seed)"
mkdir -p $OUT
rate(){ $P - "$1" <<'PY'
import json, re, sys
try:
    log = open(sys.argv[1]).read()
    g = re.search(r' gen (\d+)s', log); t = [int(m) for m in re.findall(r'upd(?:50|100) .* (\d+)s\n', log)]
    h = json.load(open(sys.argv[1].replace('.log', '') + '/history.json'))
    val = max(30., float(h[0]['seconds']) - (t[-1] - int(g.group(1)))) if h else 60.
    print(f"{(t[1]-t[0])/50:.4f} {g.group(1)} {val:.0f}" if len(t) >= 2 and g else "nan 0 0")
except Exception:
    print("nan 0 0")
PY
}
COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8"
for seed in 342101 342202; do
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v420.py $COMMON --out $OUT/pf-$TS2-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-s$seed.log 2>&1
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-s$seed.log)"
  UPD=$($P -c "import math; r=float('$RS'); u=int((6000/1.05-float('$LS')-3*float('$VS'))/r) if math.isfinite(r) and r>0 else 0; print(min(16000,(u//500)*500))")
  log "batch 3S-A preflight seed $seed: s/update $RS load $LS val $VS -> updates $UPD"
  [ "$UPD" -ge 4000 ] || { log "ABORT INCONCLUSIVE: seed $seed cannot reach 4,000 updates in a lease"; continue; }
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": 256, "updates": $UPD, "seed": $seed,
 "preflight": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v420.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA",
 "expectation": "structure (R8 surface >= .80, K0 >= .85, var R8 >= .65, K0 >= .70) and furniture (furniture-region F1 R8 >= .50, K0 >= .55) on both seeds => maps furnished storeys at R1; structure only => maps the storey structure, not its furniture (partial); R8 surface < .70 or R8 var < .50 on either => does not with this model; else partial; fusion R8 >= K0 - 3pp; FPR <= .03; updates < 12000 => budget-confounded; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-A start seed $seed ($UPD updates)"
  timeout -k 60 $((DT - $(date +%s))) $P scripts/train_3d_v420.py $COMMON --out $OUT/a-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/a-s$seed.log 2>&1
  log "3S-A seed $seed done"
done
log "batch 3S-A done"
RES=$($P - <<'PY'
import gzip, json
import numpy as np
OUT = "runs/v420-storey"; rows = {}
for s in ("342101", "342202"):
    try:
        f = json.load(open(f"{OUT}/a-s{s}/final.json")); m = json.load(open(f"{OUT}/a-s{s}/run.json"))
        if f["updates"] != m["updates"]: rows[s] = f"INCONCLUSIVE (incomplete: {f['updates']} of {m['updates']})"; continue
        v = f["val"]; fr = f.get("fresh", {})
        rows[s] = {**{f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr") if v[h2][k] is not None},
                   "fresh_R8_surface_f1": round(fr["8"]["surface_f1"], 4) if fr else None, "fresh_R8_var_occ_f1": round(fr["8"]["var_occ_f1"], 4) if fr else None,
                   "fresh_R8_furn_f1": round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, "updates": f["updates"], "budget_confounded": f["updates"] < 12000}
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
ok = all(isinstance(r, dict) for r in rows.values())
try:                                                                       # review S4: a missing key is INCONCLUSIVE, not a crash
    if not ok:
        raise KeyError("incomplete seeds")
    g = lambda key: [r[key] for r in rows.values()]
    structure = min(g("R8_surface_f1")) >= .80 and min(g("K0_surface_f1")) >= .85 and min(g("R8_var_occ_f1")) >= .65 and min(g("K0_var_occ_f1")) >= .70
    furniture = min(g("R8_furn_f1")) >= .50 and min(g("K0_furn_f1")) >= .55
    if structure and furniture: verdict = "maps furnished storeys at R1"
    elif structure: verdict = "maps the storey structure, not its furniture (partial)"
    elif min(g("R8_surface_f1")) < .70 or min(g("R8_var_occ_f1")) < .50: verdict = "does not with this model (code density, readout / decoder and world not separated)"
    else: verdict = "partial"
    fusion_ok = all(r["R8_surface_f1"] >= r["K0_surface_f1"] - .03 for r in rows.values()); fpr_ok = all(r["R8_unknown_known_fpr"] <= .03 for r in rows.values())
except Exception as e:
    verdict = f"INCONCLUSIVE ({e})"; fusion_ok = fpr_ok = None
# compression on 16 cached TRAIN scenes: 2-bit packed raw own map vs gzip -9 vs the 512 B message
man = json.load(open("/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600/manifest.json")); gz = []
for r in man["rows"][:16]:
    priv = np.load("/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600/" + r["file"])["private"]
    for a in range(priv.shape[0]):
        q = priv[a].reshape(-1).astype(np.int16) + 1
        gz.append(len(gzip.compress((q[0::4] | (q[1::4] << 2) | (q[2::4] << 4) | (q[3::4] << 6)).astype(np.uint8).tobytes(), 9)))
comp = dict(raw_2bit_bytes=128 * 128 * 32 // 4, gzip_median_bytes=int(np.median(gz)), code_bytes=512, ratio_vs_raw=round(128 * 128 * 32 / 4 / 512, 1), ratio_vs_gzip=round(float(np.median(gz)) / 512, 2))
refs = json.load(open(f"{OUT}/reference_predictors.json"))
json.dump(dict(rows=rows, verdict=verdict, fusion_ok=fusion_ok, fpr_ok=fpr_ok, compression=comp, reference_predictors=refs), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| fusion_ok", fusion_ok, "| fpr_ok", fpr_ok, "| compression", comp, "|", json.dumps(rows))
PY
)
log "batch 3S-A analysis: $RES"
