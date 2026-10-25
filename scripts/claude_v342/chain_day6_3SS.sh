#!/bin/bash
# Day-6 batch 3S-S (Claude; written 2026-10-25 14:24 UTC while 3S-Q ran and the full-TRAIN cache generated, revised 14:45 UTC after
# code_review_v430_data.md S1-S7, before any 3S-S training). Pre-registered in OVERNIGHT_REPORT D8-21; direction D8-19 (user: if data
# quantity is the next lever, raise it). Operational precondition (review M1): chain_day6_3SD.sh must stay alive under that name with
# OUT=runs/v429-ladder until it writes its analysis (this chain's wait and gate depend on it).
# ONE variable: the TRAIN set. Control = the baseline's 16,000-update run on the 4,708-house cache (cache3d-v420/train-r1-k1600, 1,600
# houses per stratum); arm = the SAME recipe (decoder, width, layout, wire bits, precision, known-weight, lr, 16,000 updates, seeds) on the
# FULL-TRAIN cache cache3d-v420/train-r1-k5383 (5,383 houses per stratum = the smallest stratum, i.e. 16,149 of the 20,366 TRAIN houses,
# stratum-balanced; ~15,800 scenes after the usual placement skips, review S4), loaded lazily (train_3d_v430 --cache-lazy: bit-for-bit an
# eager run, test_3d_v430 B). At equal updates the arm sees
# each house ~1 time instead of ~3.4: the batch measures DIVERSITY at equal compute; the interaction with training length is for a later
# cycles batch. The baseline is read at run time from 3S-D's analysis (runs/v429-ladder/analysis.json: next_baseline_dir / prefix).
# The data order differs by construction (a different manifest); trunk / head / decoder initial tensors are asserted equal to the control's.
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = arm - control):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => data quantity is a lever (the full cache
#   becomes the recipe's);
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => worse (fewer passes hurt more
#   than diversity helps at 16k: pair the full cache with cycles next);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds => not better at equal compute;
#   otherwise => ambiguous.
#   Guards: R8 FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp; K0 variable-structure F1 >= control - 2 pp; a lever whose guard
#   fails is not adopted. Mechanism reading: the TRAIN-vs-VAL gap (diag_trainval_v42x.py on the control's own 24 trained houses) -- a
#   data lever should NOT widen it; reported with the fresh panel (another 24 houses) as the generalisation readout.
#   The absolute 3S-A criteria and the user's final targets are reported. Reported, not judged: Delta K0, furniture, tol-1, fresh, the
#   fusion gap, the D8-3 breakdown, interior / boundary walls, the veto fraction, compression (unchanged).
# Validity of the arm's cache (asserted before launch): manifest version 420, dims R1, per_stratum 5383, scenes >= 15,000, data module sha
#   equal to the control cache's; the first and last two rows regenerate (the trainer's own check).
# Budget: SOLO, after chain_day6_3SD and after the cache generator; one lease per run; must run 16,000 updates (preflight; the lazy loader
#   adds a few ms per update); hard timeouts inside the lease. Writes only under /mnt/datasets (runs/v430-data).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
BIG=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k5383
OUT=runs/v430-data
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SS started pid $$ sha256 $(sha256sum scripts/chain_day6_3SS.sh | cut -c1-16)"
check_shas(){
  TSHA=$(sha256sum scripts/train_3d_v430.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v429.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v429.py | cut -d' ' -f1); VSHA=$(sha256sum scripts/diag_veto_v429.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "d862da289ca7289a8e215a765e78539477b3d7eb1fb08b4b0354038fd1291391" ] || { log "ABORT: train_3d_v430.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "545a1532da83f520b1bf3b76ccaf385f6b4eef2bea49a921534c7f1cc400890c" ] || { log "ABORT: diag_storey_v429.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "1649cb5ee58fa55f99d4714135d867e97fffed82519e5f4c149ae242f0504033" ] || { log "ABORT: diag_interior_v429.py sha changed ($ISHA)"; exit 2; }
  [ "$VSHA" = "2c4385f8b1a5adeb8072c0dd0cbba8758d7ff2358783495e0ad91802f6d26fc7" ] || { log "ABORT: diag_veto_v429.py sha changed ($VSHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the controls' ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day6_3SS already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-S history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SD[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3ST[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SQ[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SR[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SP[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v42[01][.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
# ---- baseline from the previous batch's pre-registered analysis: label, control directory, run prefix
read BASE CTRL CP <<<"$($P - <<'PY'
import json
try:
    a = json.load(open("runs/v429-ladder/analysis.json"))
    print(a["next_baseline_dir"].split("/")[-1], a["next_baseline_dir"], a["next_baseline_prefix"])
except Exception as e:
    print("missing", "", "")
PY
)"
[ -n "$CTRL" ] && [ -d "$CTRL" ] && [ -n "$CP" ] || { log "ABORT: baseline '$BASE' -> control '$CTRL' prefix '$CP' not usable"; exit 2; }
for s in 342101 342202; do [ -f $CTRL/$CP$s/final.json ] && [ -f $CTRL/$CP$s/checkpoints/initial.pt ] || { log "ABORT: control $CTRL/$CP$s (final / initial) missing"; exit 2; }; done
read DEC WID LAY PREC KW WB LR DD CB <<<"$($P -c "import json; m=json.load(open('$CTRL/${CP}342101/run.json')); print(m.get('decoder','d3'), m.get('decoder_width',1.0), m.get('layout', 'x'.join(str(v) for v in m['blocks'])), m.get('precision','learned'), m.get('known_weight','uniform'), ','.join(str(b) for b in m.get('wire_bits', [8, 8])), m['lr'], m['D'], m.get('raw_code_bytes', 512))")"
$P -c "import json,sys; m=json.load(open('$BIG/manifest.json')); c=json.load(open('$CACHE/manifest.json')); sys.exit(0 if m['version']==420 and m['dims']==[128,128,32] and m['per_stratum']==5383 and m['scenes']>=15000 and m['data_module_sha256']==c['data_module_sha256'] else 1)" || { log "ABORT: full-TRAIN cache $BIG missing or invalid"; exit 2; }
BSHA=$(sha256sum $BIG/manifest.json | cut -d' ' -f1)
$P -c "import json,sys; m=json.load(open('$CTRL/${CP}342101/run.json')); sys.exit(0 if m['cache']['dir'].rstrip('/').endswith('train-r1-k1600') and m['updates']==16000 else 1)" || { log "ABORT: the control must be a 16k run on the 4,708-house cache"; exit 2; }
log "batch 3S-S gate: baseline '$BASE' = $CTRL/$CP* (decoder $DEC W $WID layout $LAY precision $PREC known-weight $KW wire $WB); arm = same recipe on the full-TRAIN cache ($BIG, lazy)"
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
COMMON="--source $S --cache $BIG --cache-lazy --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAY --allow-non256 --wire-bits $WB --decoder $DEC --decoder-width $WID --precision $PREC --known-weight $KW --lr $LR"
for seed in 342101 342202; do
  check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-S: GPU busy before seed $seed; waiting"; sleep 120; done
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v430.py $COMMON --out $OUT/pf-$TS2-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-s$seed.log 2>&1
  PFX=$?
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-s$seed.log)"
  [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): seed $seed"; continue; }
  FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
  log "batch 3S-S preflight seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
  [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
  UPD=16000
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": $DD, "code_bytes": $CB, "lr": $LR, "updates": $UPD, "seed": $seed,
 "batch": "3S-S", "variable": "TRAIN cache 4,708 -> full (lazy)", "arm_cache": "$BIG", "arm_cache_manifest_sha256": "$BSHA", "wire_bits": "$WB", "baseline": "$BASE", "baseline_recipe": {"decoder": "$DEC", "decoder_width": $WID, "layout": "$LAY", "precision": "$PREC", "known_weight": "$KW"}, "preflight": "rate=$RS load=$LS val=$VS",
 "trainer": "scripts/train_3d_v430.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "diag_veto_sha256": "$VSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/$CP$seed",
 "expectation": "paired vs control: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => lever; Delta R8 surface <= -1pp on both or Delta R8 var <= -2pp on both => worse; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => not better; else ambiguous; guards FPR <= control + 0.3pp, slab recall >= control - 2pp, K0 var >= control - 2pp; mechanism: TRAIN-vs-VAL gap not widened; fresh panel reported; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-S start seed $seed ($UPD updates, full-TRAIN cache on $DEC W $WID $LAY $PREC $KW wire $WB)"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v430.py $COMMON --out $OUT/a-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/a-s$seed.log 2>&1
  log "3S-S seed $seed done (exit $?)"
done
log "batch 3S-S training done"
check_shas
for s in 342101 342202; do                                                      # report-only breakdowns for the arm and the control (CPU for the veto)
  if [ -f $OUT/a-s$s/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v429.py --source $S --run $OUT/a-s$s --out $OUT/diag-a-s$s --trainer train_3d_v430 > $OUT/diag-a-s$s.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v429.py --source $S --run $OUT/a-s$s --trainer train_3d_v430 2>$OUT/diag-interior-a-s$s.err | tail -1 > $OUT/diag-interior-a-s$s.json
    CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v429.py --source $S --run $OUT/a-s$s --trainer train_3d_v430 --out $OUT/veto-a-s$s.json > $OUT/veto-a-s$s.log 2>&1
  fi
  [ -f $OUT/diag-control-s$s/diag.json ] || timeout -k 30 1800 $P scripts/diag_storey_v429.py --source $S --run $CTRL/$CP$s --out $OUT/diag-control-s$s --trainer train_3d_v430 > $OUT/diag-control-s$s.log 2>&1
  [ -s $OUT/diag-interior-control-s$s.json ] || timeout -k 30 1800 $P scripts/diag_interior_v429.py --source $S --run $CTRL/$CP$s --trainer train_3d_v430 2>$OUT/diag-interior-control-s$s.err | tail -1 > $OUT/diag-interior-control-s$s.json
  [ -s $OUT/veto-control-s$s.json ] || CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v429.py --source $S --run $CTRL/$CP$s --trainer train_3d_v430 --out $OUT/veto-control-s$s.json > $OUT/veto-control-s$s.log 2>&1
done
RES=$($P - "$CTRL" "$CP" <<'PY'
import json, math, re, sys, torch
OUT, CTRL, CP = "runs/v430-data", sys.argv[1], sys.argv[2]
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def load(d):
    f = json.load(open(f"{d}/final.json")); m = json.load(open(f"{d}/run.json"))
    if f["updates"] != m["updates"] or f["updates"] != 16000: raise ValueError(f"incomplete: {f['updates']} of {m['updates']}")
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], blocks=m["blocks"], decoder=m.get("decoder", "d3"),
             decoder_width=m.get("decoder_width", 1.0), precision=m.get("precision", "learned"), known_weight=m.get("known_weight", "uniform"), manifest_sha=m["cache"]["manifest_sha256"], params=m["parameters"],
             wire_bits=m.get("wire_bits", [8, 8]), cache_dir=m["cache"]["dir"], train_scenes=m["train_scenes"], lr=m["lr"],
             specs=[json.dumps(x["spec"], sort_keys=True) for x in json.load(open(f"{d}/validation/final-val.json"))["rows"]], seconds=round(f["seconds"]))
    logtxt = open(f"{d}.log").read(); norms = []
    for gd in re.findall(r"grad (\{[^}]*\})", logtxt):
        g = json.loads(gd.replace("'", '"')); norms.append(math.sqrt(sum(x * x for x in g.values())))
    r["clip_fraction_logged"] = round(sum(n > 5 for n in norms) / max(1, len(norms)), 4)
    return r
def extras(r, diag_dir, interior_file, veto_file):
    try:
        xi = json.load(open(interior_file)); r["K0_interior"] = xi["K0_interior"]["recall"]; r["K8_interior"] = xi["K8_interior"]["recall"]; r["K0_boundary"] = xi["K0_boundary"]["recall"]
    except Exception: r["K0_interior"] = None
    try:
        xd = json.load(open(f"{diag_dir}/diag.json")); r["R8_slab_recall"] = xd["K8"]["recall_slab"]; r["R8_struct_recall"] = xd["K8"]["recall_struct"]; r["K0_precision_occ"] = xd["K0"]["precision_occ"]; r["R8_precision_occ"] = xd["K8"]["precision_occ"]
    except Exception: r["R8_slab_recall"] = None
    try:
        vf = json.load(open(veto_file)); r["veto_K0"] = vf["K0"]["vetoed_fraction"]; r["veto_R8"] = vf["K8"]["vetoed_fraction"]; r["veto_occ_recall_R8"] = vf["K8"]["occ_head_recall_on_vetoed"]
    except Exception: r["veto_K0"] = r["veto_R8"] = None
    return r
def identity(a_dir, c_dir):
    a = torch.load(f"{a_dir}/checkpoints/initial.pt", map_location="cpu", weights_only=False)["state_dict"]; c = torch.load(f"{c_dir}/checkpoints/initial.pt", map_location="cpu", weights_only=False)["state_dict"]
    return list(a) == list(c) and all(torch.equal(a[k], c[k]) for k in c) and len(c) > 10            # the knob changes no tensor shape: EVERY initial tensor equal
rows = {}
for s in ("342101", "342202"):
    try:
        a = extras(load(f"{OUT}/a-s{s}"), f"{OUT}/diag-a-s{s}", f"{OUT}/diag-interior-a-s{s}.json", f"{OUT}/veto-a-s{s}.json")
        c = extras(load(f"{CTRL}/{CP}{s}"), f"{OUT}/diag-control-s{s}", f"{OUT}/diag-interior-control-s{s}.json", f"{OUT}/veto-control-s{s}.json")
        assert a["precision"] == c["precision"] and a["known_weight"] == c["known_weight"] and a["wire_bits"] == c["wire_bits"] and a["lr"] == c["lr"], "recipe"
        assert a["cache_dir"].rstrip("/").endswith("train-r1-k5383") and c["cache_dir"].rstrip("/").endswith("train-r1-k1600") and a["train_scenes"] >= 15000, "caches"
        assert a["blocks"] == c["blocks"] and a["decoder"] == c["decoder"] and a["decoder_width"] == c["decoder_width"], "recipe"
        assert a["manifest_sha"] != c["manifest_sha"] and a["specs"] == c["specs"], "VAL panel (the caches differ by design)"
        assert a["params"] == c["params"] and identity(f"{OUT}/a-s{s}", f"{CTRL}/{CP}{s}"), "identity (params, initial tensors)"
        d = {k: round(a[k] - c[k], 4) for k in a if k in c and isinstance(a[k], float) and isinstance(c[k], float)}
        for r in (a, c): r.pop("specs", None)
        rows[s] = dict(arm=a, control=c, delta=d)
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
try:
    if not all(isinstance(r, dict) for r in rows.values()): raise KeyError("incomplete seeds")
    d = lambda k: [r["delta"][k] for r in rows.values()]
    if min(d("R8_var_occ_f1")) >= .03 and min(d("R8_surface_f1")) >= .01: verdict = "lever"
    elif max(d("R8_surface_f1")) <= -.01 or max(d("R8_var_occ_f1")) <= -.02: verdict = "worse"
    elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02: verdict = "not better"
    else: verdict = "ambiguous"
    guards = {}
    for name, fn in (("fpr_ok", lambda r: r["arm"]["R8_unknown_known_fpr"] <= r["control"]["R8_unknown_known_fpr"] + .003), ("slab_ok", lambda r: r["arm"]["R8_slab_recall"] >= r["control"]["R8_slab_recall"] - .02), ("k0var_ok", lambda r: r["delta"]["K0_var_occ_f1"] >= -.02)):
        try: guards[name] = all(fn(r) for r in rows.values())
        except Exception as e: guards[name] = f"n/a ({e})"
    try:
        mech = "fresh panel (arm, control): " + json.dumps({s: (r["arm"]["fresh_R8_surface_f1"], r["control"]["fresh_R8_surface_f1"]) for s, r in rows.items()}) + "; TRAIN-vs-VAL gap reported post hoc (diag_trainval_v42x.py)"
    except Exception as e:
        mech = f"n/a ({e})"
except Exception as e:
    verdict = f"INCONCLUSIVE ({e})"; guards = mech = None
try:                                                                             # report-only (review S1: never overwrites the verdict)
    g = lambda k: [r["arm"][k] for r in rows.values()]
    structure = min(g("R8_surface_f1")) >= .80 and min(g("K0_surface_f1")) >= .85 and min(g("R8_var_occ_f1")) >= .65 and min(g("K0_var_occ_f1")) >= .70
    furniture = min(g("R8_furn_f1")) >= .50 and min(g("K0_furn_f1")) >= .55
    absolute = ("maps furnished storeys at R1" if structure and furniture else "maps the storey structure, not its furniture (partial)" if structure
                else "does not with this model" if min(g("R8_surface_f1")) < .70 or min(g("R8_var_occ_f1")) < .50 else "partial")
except Exception as e:
    absolute = f"n/a ({e})"
try:
    fusion_gap = {s: (round(r["arm"]["R8_surface_f1"] - r["arm"]["K0_surface_f1"], 4), round(r["control"]["R8_surface_f1"] - r["control"]["K0_surface_f1"], 4)) for s, r in rows.items()}
except Exception as e:
    fusion_gap = f"n/a ({e})"
adopt = verdict == "lever" and isinstance(guards, dict) and guards.get("fpr_ok") is True and guards.get("k0var_ok") is True    # review S4: pre-registered
if verdict == "lever" and not adopt: verdict = "lever, but a guard failed (FPR or K0 var): baseline unchanged"
nb_dir, nb_prefix = (OUT, "a-s") if adopt else (CTRL, CP)
nb_cache = (rows["342101"]["arm"]["cache_dir"] if adopt else rows["342101"]["control"]["cache_dir"]) if isinstance(rows.get("342101"), dict) else None   # review S6
try:
    g = lambda k: [r["arm"][k] for r in rows.values()]
    final_targets_met = min(g("R8_surface_f1")) >= .90 and min(g("K0_surface_f1")) >= .95 and min(g("R8_var_occ_f1")) >= .75           # review S7
except Exception:
    final_targets_met = None
json.dump(dict(control=CTRL, control_prefix=CP, baseline_recipe={k: rows["342101"]["control"][k] for k in ("decoder", "decoder_width", "blocks", "precision", "known_weight")} if isinstance(rows.get("342101"), dict) else None,
               rows=rows, verdict=verdict, guards=guards, mechanism=mech, absolute_verdict=absolute, user_final_targets_met=final_targets_met, fusion_gap_arm_vs_control=fusion_gap, next_baseline_dir=nb_dir, next_baseline_prefix=nb_prefix, next_baseline_cache=nb_cache, next_baseline_cache_lazy=bool(adopt)), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| guards", guards, "| mechanism", mech, "| absolute", absolute, "| fusion gap (arm, control)", fusion_gap, "| next baseline", nb_dir, nb_prefix, "|", json.dumps({s: (r["delta"] if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch 3S-S analysis: $RES"
