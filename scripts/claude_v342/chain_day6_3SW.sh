#!/bin/bash
# Day-6 batch 3S-W (Claude; first written 2026-10-25 03:45 UTC, revised 04:00 UTC after code_review_v422_decoder.md M1-M3 and
# S1-S8, before any v422 training). Pre-registered in OVERNIGHT_REPORT D8-4.
# Question: is batch 3S-A (furnished HouseExpo storeys, R1 = 128 x 128 x 32, D = 256, R8 surface F1 .636 / .631) limited by the
# decoder's WIDTH? Post-hoc diagnosis (D8-3, not a criterion): wall / lintel recall .49 at K0; interior walls (only the code can place
# them) .09-.12 at K0 against .51 for boundary walls. That pattern fits an encoder / code failure as well as a decoder failure: with joint
# training the batch tests ONE side of it -- the decoder's width -- and a null result means "not decoder width x 2", nothing more.
# ONE variable: train_3d_v422.py --decoder-width 2.0 (every decoder channel x 2; decoder params 468,376 -> 1,832,704, 3.91x).
# Everything else is 3S-A: data module v420 and the SAME cache, seeds 342101 / 342202, 16,000 updates, lr 3e-4 cosine, N = 8, trunk /
# head identical initial tensors (test_3d_v422 B), same VAL / fresh panels. Control = 3S-A (runs/v420-storey, already run).
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = v422 - 3S-A; review M2: the wall signal is in the variable-
# structure F1, the surface F1 is ~41 % slab):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => decoder width is a lever (confirmed);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds  => not limited by decoder width x 2;
#   otherwise                                                                               => ambiguous.
#   The absolute 3S-A criteria are re-applied for the record (maps furnished storeys / structure only / does not / partial).
#   Reported, not judged: Delta K0, furniture-region F1, tolerance-1, fresh, FPR, the fusion gap, the D8-3 breakdown and the interior /
#   boundary wall recall on the new checkpoints, the share of logged updates whose total gradient norm exceeded the clip (5) in both arms
#   (review S3), compression (unchanged: 512 B per message).
# Budget rule: SOLO (no other GPU compute process), seeds strictly sequential, each under its own inclusive 7,200 s lease; a 100-update
#   CUDA preflight per seed (timeout 1,800 s); the arm MUST run the control's 16,000 updates: if 1.05 x (load + 16,000 x rate + 3 x val)
#   > 6,000 s the seed is not run (INCONCLUSIVE, budget) -- fewer updates would confound the one variable. Probe 03:40-03:43 UTC:
#   0.20-0.22 s/update, load 118 s => ~3,800 s. Every run under a hard timeout that ends inside the lease (timeout -k 30 at +7,150 s).
#   Writes only under /mnt/datasets (runs/v422-decoder).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v422-decoder
CTRL=runs/v420-storey
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
mkdir runs/v422-decoder.lock 2>/dev/null || { echo "$(date -u +%FT%TZ) ABORT: chain_day6_3SW already launched (runs/v422-decoder.lock exists)" | tee -a $LOG; exit 2; }   # review S4
log "chain_day6_3SW started pid $$ sha256 $(sha256sum scripts/chain_day6_3SW.sh | cut -c1-16)"
check_shas(){                                                                    # run before each seed (review S5: includes the cache manifest)
  TSHA=$(sha256sum scripts/train_3d_v422.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_3sa_v420.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v420.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "b59747bf7493deff55b74fe6df2fef2bfc61cc441c35f1d4b202e5a98041803c" ] || { log "ABORT: train_3d_v422.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "2c69149b3877e685988ded2f1b2215e3e29d96be0a8abc53a7146dc5f490a0ef" ] || { log "ABORT: diag_3sa_v420.py sha changed ($GSHA)"; exit 2; }                  # review M1
  [ "$ISHA" = "fd6e84127e1df72de76ce2d3b2a334a6551490c043619994424564ce2462e034" ] || { log "ABORT: diag_interior_v420.py sha changed ($ISHA)"; exit 2; }             # review S1
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the control's ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_s*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-W history (never overwrite)"; exit 2; fi
for s in 342101 342202; do [ -f $CTRL/a-s$s/final.json ] && [ -f $CTRL/diag-a-s$s/diag.json ] || { log "ABORT: control $CTRL/a-s$s (final.json / diag) missing"; exit 2; }; done
busy(){ pgrep -f "chain_day6_3S[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v420[.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done                                          # review S4: a second look closes gaps between ad-hoc jobs
log "batch 3S-W: GPU free; starting (SOLO, sequential, one lease per seed)"
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
COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --decoder-width 2.0"
for seed in 342101 342202; do
  check_shas                                                                  # the second seed re-checks the pinned files
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v422.py $COMMON --out $OUT/pf-$TS2-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-s$seed.log 2>&1
  PFX=$?
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-s$seed.log)"
  [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): seed $seed"; continue; }   # review S8: not a budget failure
  FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
  log "batch 3S-W preflight seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
  [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
  UPD=16000
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": 256, "updates": $UPD, "seed": $seed, "decoder_width": 2.0,
 "preflight": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v422.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/a-s$seed",
 "expectation": "paired vs 3S-A: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => decoder width is a lever; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => not limited by decoder width x2; else ambiguous; absolute 3S-A criteria re-applied for the record; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-W start seed $seed ($UPD updates, decoder width 2.0)"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v422.py $COMMON --out $OUT/a-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/a-s$seed.log 2>&1
  log "3S-W seed $seed done (exit $?)"
done
log "batch 3S-W training done"
for seed in 342101 342202; do                                                   # report-only breakdowns (D8-3 protocol)
  if [ -f $OUT/a-s$seed/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_3sa_v420.py --source $S --run $OUT/a-s$seed --out $OUT/diag-a-s$seed --trainer train_3d_v422 --decoder-width 2.0 > $OUT/diag-a-s$seed.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v420.py --source $S --run $OUT/a-s$seed --trainer train_3d_v422 --decoder-width 2.0 2>$OUT/diag-interior-a-s$seed.err | tail -1 > $OUT/diag-interior-a-s$seed.json
  fi
done
RES=$($P - <<'PY'
import json, math, re
OUT, CTRL = "runs/v422-decoder", "runs/v420-storey"; rows = {}
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def load(d, s):
    f = json.load(open(f"{d}/a-s{s}/final.json")); m = json.load(open(f"{d}/a-s{s}/run.json"))
    if f["updates"] != m["updates"] or f["updates"] != 16000:
        raise ValueError(f"incomplete: {f['updates']} of {m['updates']}")
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], decoder_width=m.get("decoder_width", 1.0))
    specs = [json.dumps(x["spec"], sort_keys=True) for x in json.load(open(f"{d}/a-s{s}/validation/final-val.json"))["rows"]]
    logtxt = open(f"{d}/a-s{s}.log").read(); norms = []
    for gd in re.findall(r"grad (\{[^}]*\})", logtxt):
        g = json.loads(gd.replace("'", '"')); norms.append(math.sqrt(sum(x * x for x in g.values())))
    r["clip_fraction_logged"] = round(sum(n > 5 for n in norms) / max(1, len(norms)), 4); r["logged_updates"] = len(norms)
    return r, specs
for s in ("342101", "342202"):
    try:
        (a, sa), (c, sc) = load(OUT, s), load(CTRL, s)
        assert a["decoder_width"] == 2.0 and c["decoder_width"] == 1.0, "decoder widths"
        assert sa == sc, "the two arms were scored on different VAL scenes"                        # review S8
        rows[s] = dict(arm=a, control=c, delta={k: round(a[k] - c[k], 4) for k in a if k in c and isinstance(a[k], float) and isinstance(c[k], float)})
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
try:
    if not all(isinstance(r, dict) for r in rows.values()):
        raise KeyError("incomplete seeds")
    d = lambda k: [r["delta"][k] for r in rows.values()]
    if min(d("R8_var_occ_f1")) >= .03 and min(d("R8_surface_f1")) >= .01: verdict = "decoder width is a lever (confirmed)"
    elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02: verdict = "not limited by decoder width x2"
    else: verdict = "ambiguous"
    g = lambda k: [r["arm"][k] for r in rows.values()]
    structure = min(g("R8_surface_f1")) >= .80 and min(g("K0_surface_f1")) >= .85 and min(g("R8_var_occ_f1")) >= .65 and min(g("K0_var_occ_f1")) >= .70
    furniture = min(g("R8_furn_f1")) >= .50 and min(g("K0_furn_f1")) >= .55
    absolute = ("maps furnished storeys at R1" if structure and furniture else "maps the storey structure, not its furniture (partial)" if structure
                else "does not with this model" if min(g("R8_surface_f1")) < .70 or min(g("R8_var_occ_f1")) < .50 else "partial")
    fusion_ok = all(r["arm"]["R8_surface_f1"] >= r["arm"]["K0_surface_f1"] - .03 for r in rows.values()); fpr_ok = all(r["arm"]["R8_unknown_known_fpr"] <= .03 for r in rows.values())
except Exception as e:
    verdict = f"INCONCLUSIVE ({e})"; absolute = fusion_ok = fpr_ok = None
diag = {}
for s in ("342101", "342202"):
    try:
        x = json.load(open(f"{OUT}/diag-a-s{s}/diag.json")); y = json.load(open(f"{CTRL}/diag-a-s{s}/diag.json"))
        diag[s] = {h: {k: (round(x[h][k], 3), round(y[h][k], 3)) for k in ("recall_slab", "recall_struct", "recall_obj", "recall_struct_2.1-3.2m", "fp_near_structure", "fp_furniture_region", "precision_occ")} for h in ("K0", "K8")}
        xi = json.load(open(f"{OUT}/diag-interior-a-s{s}.json")); yi = json.load(open(f"{CTRL}/diag-interior-a-s{s}.json"))
        diag[s]["interior_boundary_recall_arm_vs_control"] = {k: (xi[k]["recall"], yi[k]["recall"]) for k in ("K0_interior", "K0_boundary", "K8_interior", "K8_boundary")}
    except Exception as e:
        diag[s] = f"n/a ({e})"
json.dump(dict(rows=rows, verdict=verdict, absolute_verdict=absolute, fusion_ok=fusion_ok, fpr_ok=fpr_ok, diag_arm_vs_control=diag, code_bytes=512), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| absolute", absolute, "| fusion_ok", fusion_ok, "| fpr_ok", fpr_ok, "|", json.dumps({s: (r["delta"] if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch 3S-W analysis: $RES"
