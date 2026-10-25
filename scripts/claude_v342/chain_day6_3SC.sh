#!/bin/bash
# Day-6 batch 3S-C (Claude; written 2026-10-25 04:07 UTC while 3S-W runs, revised 04:31 UTC after code_review_v423_layout.md M1-M2 and
# S1-S9, before any v423 training and without looking at any 3S-W validation). Pre-registered in OVERNIGHT_REPORT D8-5; design claude_handoff/DESIGN_3D_STOREY_V420.md section 4 ("16 x 16 x 1 as a single-variable control batch").
# Question: does the LAYOUT of the D = 256 code limit batch 3S-A? An indoor storey is nearly 2.5-D: walls are vertical extrusions of
# the floor plan, slabs are planes, furniture heights are standard. 8 x 8 x 4 spends 3/4 of the coordinates on the vertical, where the
# map is nearly predictable, and leaves the floor plan at 8 x 8 -- the 2D line's density. 16 x 16 x 1 gives every coordinate a whole
# L/16 x L/16 x 3.2 m column: twice the horizontal resolution of the code, same 256 coordinates, same 512 B per message. The post-hoc
# diagnosis (D8-3) found interior walls -- the content only the code can carry -- recalled at .09-.12 (K0). Prior evidence (D4-6, the 2D
# representation ceiling on HouseExpo walls): 16x16x1 .918, 8x8x1 .694, and 4x4x4 -- four scalars in a coarser block, 3S-A's situation --
# .623. Counter-evidence: the tuned 2D recipe reaches K0 ~.93 at 8x8 (wide decoder, 57,600 updates x 3 cycles). Disclosed side effects
# of the layout (review S3, S4), all part of this one variable: the coverage mask, hence the known / unknown outline the decoder sees and
# the join's granularity, is 2x finer horizontally; an agent has ~0.76x as many evidence-bearing coordinates; the decoder's horizontal
# receptive field over the code shrinks from ~1.0 L to ~0.66 L; one scalar per column must carry the vertical profile too.
# ONE variable: train_3d_v423.py --layout 16x16x1 (encoder patch 4 x 4 x 16 compute cells, decoder input grid 16 x 16 x 1). Parameter
# counts and initial tensors are identical to v420 (test_3d_v423 B); decoder width 1 (v420). Everything else is 3S-A: data module v420
# and the SAME cache, seeds 342101 / 342202, 16,000 updates, lr 3e-4 cosine, N = 8, same panels. Control = 3S-A (runs/v420-storey).
# Order (D8-5 revises D8-4's next-step plan, before any 3S-W result was looked at, because of the D4-6 evidence): 3S-C runs after 3S-W
# WHATEVER 3S-W shows -- both are single-variable tests against the same control; the training-budget batch moves after 3S-C.
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = v423 - 3S-A):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => the code layout is a lever (confirmed);
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => 16x16x1 is worse (review M1);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds  => 16x16x1 is not better than 8x8x4;
#   otherwise                                                                               => ambiguous.
#   The absolute 3S-A criteria are re-applied for the record.
#   Mechanism reading (review S3, reported beside the verdict): Delta K0 interior-wall recall >= +5 pp on both seeds => the gain (if any)
#   comes with the code carrying interior walls; otherwise it cannot be attributed to that mechanism (finer outline / join).
#   Reported, not judged: Delta K0, furniture-region F1, tolerance-1, fresh, FPR, the fusion gap, the D8-3 breakdown and the interior /
#   boundary wall recall (the most specific readout of this hypothesis), the mask fraction (evidence-bearing coordinates) in both arms,
#   the gradient-clip share, compression (512 B per message). Identity checks: same initial-weights hash, cache manifest and parameter
#   counts as the control.
# Budget rule: as 3S-W (SOLO, sequential, one inclusive 7,200 s lease per seed, 100-update preflight, must run the control's 16,000
#   updates or INCONCLUSIVE, hard timeout inside the lease). Writes only under /mnt/datasets (runs/v423-layout).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v423-layout
CTRL=runs/v420-storey
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
mkdir runs/v423-layout.lock 2>/dev/null || { echo "$(date -u +%FT%TZ) ABORT: chain_day6_3SC already launched (runs/v423-layout.lock exists)" | tee -a $LOG; exit 2; }
log "chain_day6_3SC started pid $$ sha256 $(sha256sum scripts/chain_day6_3SC.sh | cut -c1-16)"
check_shas(){                                                                    # run before each seed (review S5: includes the cache manifest)
  TSHA=$(sha256sum scripts/train_3d_v423.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v423.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v423.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "a5540187592f2a8996be323c2cbe778177a0d72ed8cc62b6cfb32b493794eba6" ] || { log "ABORT: train_3d_v423.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "a28dfffc0e658b9d16a42195c6931e2652027dbbd5bcf1b104d2cedbd419de22" ] || { log "ABORT: diag_storey_v423.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "d68ee8b06290128df67928438aa2a236a3eb61d0da965ac4e54e9abc6bb52c33" ] || { log "ABORT: diag_interior_v423.py sha changed ($ISHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the control's ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_s*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-C history (never overwrite)"; exit 2; fi
for s in 342101 342202; do [ -f $CTRL/a-s$s/final.json ] && [ -f $CTRL/diag-a-s$s/diag.json ] || { log "ABORT: control $CTRL/a-s$s (final.json / diag) missing"; exit 2; }; done
busy(){ pgrep -f "bash scripts/chain_day6_3SW[.]sh" >/dev/null || pgrep -f "chain_day6_3S[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v420[.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done                                          # a second look closes gaps between ad-hoc jobs
V3SW=$($P -c "import json; print(json.load(open('runs/v422-decoder/analysis.json'))['verdict'])" 2>/dev/null || echo "missing")
log "batch 3S-C: 3S-W verdict on record: $V3SW (3S-C runs regardless, D8-5)"
log "batch 3S-C: GPU free; starting (SOLO, sequential, one lease per seed)"
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
COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout 16x16x1"
for seed in 342101 342202; do
  check_shas                                                                  # the second seed re-checks the pinned files
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-C: GPU busy before seed $seed; waiting"; sleep 120; done   # review S6
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v423.py $COMMON --out $OUT/pf-$TS2-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-s$seed.log 2>&1
  PFX=$?
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-s$seed.log)"
  [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): seed $seed"; continue; }
  FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
  log "batch 3S-C preflight seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
  [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
  UPD=16000
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": 256, "updates": $UPD, "seed": $seed, "layout": [16, 16, 1],
 "preflight": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v423.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/a-s$seed",
 "expectation": "paired vs 3S-A: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => the code layout is a lever; Delta R8 surface <= -1pp on both or Delta R8 var <= -2pp on both => 16x16x1 is worse; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => 16x16x1 is not better; else ambiguous; mechanism: Delta K0 interior-wall recall >= +5pp on both; absolute 3S-A criteria re-applied for the record; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-C start seed $seed ($UPD updates, layout 16x16x1)"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v423.py $COMMON --out $OUT/a-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/a-s$seed.log 2>&1
  log "3S-C seed $seed done (exit $?)"
done
log "batch 3S-C training done"
check_shas                                                                      # review S2: the diag copies are pinned too
for seed in 342101 342202; do                                                   # report-only breakdowns (D8-3 protocol)
  if [ -f $OUT/a-s$seed/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v423.py --source $S --run $OUT/a-s$seed --out $OUT/diag-a-s$seed --trainer train_3d_v423 > $OUT/diag-a-s$seed.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v423.py --source $S --run $OUT/a-s$seed --trainer train_3d_v423 2>$OUT/diag-interior-a-s$seed.err | tail -1 > $OUT/diag-interior-a-s$seed.json
  fi
done
RES=$($P - <<'PY'
import json, math, re
OUT, CTRL = "runs/v423-layout", "runs/v420-storey"; rows = {}
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def val_rows(d, s):
    return json.load(open(f"{d}/a-s{s}/validation/final-val.json"))["rows"]
def load(d, s):
    f = json.load(open(f"{d}/a-s{s}/final.json")); m = json.load(open(f"{d}/a-s{s}/run.json"))
    if f["updates"] != m["updates"] or f["updates"] != 16000:
        raise ValueError(f"incomplete: {f['updates']} of {m['updates']}")
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], blocks=m["blocks"], decoder_width=m.get("decoder_width", 1.0),
             init_sha=m["initial_weights_sha256"], manifest_sha=m["cache"]["manifest_sha256"], params=m["parameters"],
             val_mask_fraction=round(sum(x["mask_fraction"] for x in val_rows(d, s)) / len(val_rows(d, s)), 4))
    specs = [json.dumps(x["spec"], sort_keys=True) for x in json.load(open(f"{d}/a-s{s}/validation/final-val.json"))["rows"]]
    logtxt = open(f"{d}/a-s{s}.log").read(); norms = []
    for gd in re.findall(r"grad (\{[^}]*\})", logtxt):
        g = json.loads(gd.replace("'", '"')); norms.append(math.sqrt(sum(x * x for x in g.values())))
    r["clip_fraction_logged"] = round(sum(n > 5 for n in norms) / max(1, len(norms)), 4); r["logged_updates"] = len(norms)
    return r, specs
for s in ("342101", "342202"):
    try:
        (a, sa), (c, sc) = load(OUT, s), load(CTRL, s)
        assert a["blocks"] == [16, 16, 1] and c["blocks"] == [8, 8, 4] and a["decoder_width"] == c["decoder_width"] == 1.0, "layouts / widths"
        assert a["init_sha"] == c["init_sha"] and a["manifest_sha"] == c["manifest_sha"] and a["params"] == c["params"], "identity (review S7)"
        assert sa == sc, "the two arms were scored on different VAL scenes"                        # review S8
        rows[s] = dict(arm=a, control=c, delta={k: round(a[k] - c[k], 4) for k in a if k in c and isinstance(a[k], float) and isinstance(c[k], float)})
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
try:
    if not all(isinstance(r, dict) for r in rows.values()):
        raise KeyError("incomplete seeds")
    d = lambda k: [r["delta"][k] for r in rows.values()]
    if min(d("R8_var_occ_f1")) >= .03 and min(d("R8_surface_f1")) >= .01: verdict = "the code layout is a lever (confirmed)"
    elif max(d("R8_surface_f1")) <= -.01 or max(d("R8_var_occ_f1")) <= -.02: verdict = "16x16x1 is worse than 8x8x4"
    elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02: verdict = "16x16x1 is not better than 8x8x4"
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
mech = None
try:
    dk = [diag[s]["interior_boundary_recall_arm_vs_control"]["K0_interior"] for s in ("342101", "342202")]
    mech = ("interior walls carried (Delta K0 interior recall >= +5 pp on both seeds)" if all(a - c >= .05 for a, c in dk) else "gain not attributable to interior walls") + f" {dk}"
except Exception as e:
    mech = f"n/a ({e})"
json.dump(dict(rows=rows, verdict=verdict, mechanism=mech, absolute_verdict=absolute, fusion_ok=fusion_ok, fpr_ok=fpr_ok, diag_arm_vs_control=diag, code_bytes=512), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| mechanism", mech, "| absolute", absolute, "| fusion_ok", fusion_ok, "| fpr_ok", fpr_ok, "|", json.dumps({s: (r["delta"] if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch 3S-C analysis: $RES"
