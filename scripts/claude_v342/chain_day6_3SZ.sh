#!/bin/bash
# Day-6 batch 3S-Z (Claude; written 2026-10-26 12:58 UTC while 3S-Y runs, before any 3S-Z training; LAUNCHED ONLY AFTER THE USER APPROVES
# 8 KB messages for a diagnostic arm). Pre-registered in OVERNIGHT_REPORT D8-29.
# Question: does the coordinate ladder keep paying? D8-24: 16x16x1 -> 32x32x1 gave +4.5 / +3.7 pp R8 surface and furniture .30 -> .42 at
# 16k; more bits per coordinate gave nothing. The next rung: 64x64x1 @ 8+8 (D = 4,096; every coordinate owns an L/64 x L/64 column =
# 2 x 2 x 32 voxels at R1; 8,192 B per message). Trainer train_3d_v431 (= v430 + this layout; the v429 fold, space-to-depth 4 onto the
# 16 x 16 start grid: the decoder's first convolution reads 32 channels, everything else is the control's; the trunk sees 1 x 1 x 16-cell
# patches, i.e. a pure per-column z-profile encoder). ONE variable: the layout. Control = L1 (runs/v429-ladder/l1-s*: 32x32x1 @ 8+8, d3
# W2, learned precision, uniform known-BCE, 16,000 updates, k1600 cache). Everything else copied from the control's run.json.
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = arm - L1):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => the rung is a lever;
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => worse (the column code too
#   thin / the decoder cannot use 4,096 coordinates at 16k);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds => not better (the ladder has saturated);
#   otherwise => ambiguous. Guards: R8 FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp; K0 variable-structure F1 >= control
#   - 2 pp; a lever whose guard fails is not adopted. Mechanism readings: furniture-region F1 and furniture recall, interior-wall recall;
#   bytes per message 8,192 recorded. The absolute 3S-A criteria and the user's final targets are reported.
# Next baseline (pre-registered): adopted => the 64x64x1 recipe at 16k (to be stacked with the two-cycle protocol next); else unchanged.
# Budget: SOLO, after chain_day6_3SY; one lease per run (two runs; the 4,096-coordinate rollout and 32,768 trunk patches per scene may be
#   slower than L1's 0.20 s/update -- the preflight decides; 16,000 updates must fit, else INCONCLUSIVE); hard timeouts inside the lease.
#   Writes only under /mnt/datasets (runs/v431-rung).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v431-rung
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SZ started pid $$ sha256 $(sha256sum scripts/chain_day6_3SZ.sh | cut -c1-16)"
check_shas(){
  TSHA=$(sha256sum scripts/train_3d_v431.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v429.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v429.py | cut -d' ' -f1); VSHA=$(sha256sum scripts/diag_veto_v429.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "9e1c5081ffacd0c5a023404359c730ff2399f4f5360f97d3da34088001e8624c" ] || { log "ABORT: train_3d_v431.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "545a1532da83f520b1bf3b76ccaf385f6b4eef2bea49a921534c7f1cc400890c" ] || { log "ABORT: diag_storey_v429.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "1649cb5ee58fa55f99d4714135d867e97fffed82519e5f4c149ae242f0504033" ] || { log "ABORT: diag_interior_v429.py sha changed ($ISHA)"; exit 2; }
  [ "$VSHA" = "2c4385f8b1a5adeb8072c0dd0cbba8758d7ff2358783495e0ad91802f6d26fc7" ] || { log "ABORT: diag_veto_v429.py sha changed ($VSHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the controls' ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day6_3SZ already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if compgen -G "$OUT/z*-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-Z history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SY[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SX[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SS[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SD[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3ST[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SQ[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SR[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SP[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
# ---- baseline from the previous batch's pre-registered analysis: label, control directory, run prefix
BASE=L1; CTRL=runs/v429-ladder; CP=l1-s                                          # fixed control: the 32x32x1 @ 8+8 rung at 16k (D8-24)
for s in 342101 342202; do [ -f $CTRL/$CP$s/final.json ] && [ -f $CTRL/$CP$s/checkpoints/initial.pt ] || { log "ABORT: control $CTRL/$CP$s (final / initial) missing"; exit 2; }; done
read DEC WID LAY PREC KW <<<"$($P -c "import json; m=json.load(open('$CTRL/${CP}342101/run.json')); print(m.get('decoder','d3'), m.get('decoder_width',1.0), m.get('layout', 'x'.join(str(v) for v in m['blocks'])), m.get('precision','learned'), m.get('known_weight','uniform'))")"
[ "$DEC" = "d3" ] && [ "$WID" = "2.0" ] && [ "$LAY" = "32x32x1" ] || { log "ABORT: the rung needs the L1 (d3 W2 32x32x1) control (got $DEC W $WID $LAY)"; exit 2; }
log "batch 3S-Z gate: control $CTRL/$CP* (decoder $DEC W $WID layout $LAY precision $PREC known-weight $KW); arm Z1 64x64x1 @ 8,8 (8 KB)"
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
for arm in "z1 64x64x1 8,8"; do
  set -- $arm; ARM=$1; LAYOUT=$2; BITS=$3
  COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAYOUT --allow-non256 --wire-bits $BITS --decoder $DEC --decoder-width $WID --precision $PREC --known-weight $KW"
  for seed in 342101 342202; do
    check_shas
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-Z: GPU busy before $ARM seed $seed; waiting"; sleep 120; done
    TS2=$(date -u +%H%M%S)
    timeout -k 60 1800 $P scripts/train_3d_v431.py $COMMON --out $OUT/pf-$TS2-$ARM-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-$ARM-s$seed.log 2>&1
    PFX=$?
    read RS LS VS <<<"$(rate $OUT/pf-$TS2-$ARM-s$seed.log)"
    [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): $ARM seed $seed"; continue; }
    FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
    log "batch 3S-Z preflight $ARM seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
    [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): $ARM seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
    UPD=16000
    START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
    cat > $OUT/lease_$ARM-s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "updates": $UPD, "seed": $seed,
 "batch": "3S-Z", "arm": "$ARM", "layout": "$LAYOUT", "wire_bits": "$BITS", "bytes_per_message": 8192, "baseline": "$BASE", "baseline_recipe": {"decoder": "$DEC", "decoder_width": $WID, "layout": "$LAY", "precision": "$PREC", "known_weight": "$KW"}, "preflight": "rate=$RS load=$LS val=$VS",
 "trainer": "scripts/train_3d_v431.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "diag_veto_sha256": "$VSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/$CP$seed",
 "expectation": "64x64x1 vs L1, paired: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => the rung is a lever; <= -1pp / <= -2pp on both => worse; < +1pp and < +2pp on both => not better (ladder saturated); else ambiguous; guards FPR <= control + 0.3pp, slab recall >= control - 2pp, K0 var >= control - 2pp; adopted => next baseline; incomplete => INCONCLUSIVE"}
JSON
    log "batch 3S-Z start $ARM seed $seed ($UPD updates, $LAYOUT @ $BITS bits on d3 W $WID, $PREC, $KW)"
    timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v431.py $COMMON --out $OUT/$ARM-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/$ARM-s$seed.log 2>&1
    log "3S-Z $ARM seed $seed done (exit $?)"
  done
done
log "batch 3S-Z training done"
check_shas
for run in z1-s342101 z1-s342202; do                                              # report-only breakdowns (CPU for the veto)
  if [ -f $OUT/$run/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v429.py --source $S --run $OUT/$run --out $OUT/diag-$run --trainer train_3d_v431 > $OUT/diag-$run.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v429.py --source $S --run $OUT/$run --trainer train_3d_v431 2>$OUT/diag-interior-$run.err | tail -1 > $OUT/diag-interior-$run.json
    CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v429.py --source $S --run $OUT/$run --trainer train_3d_v431 --out $OUT/veto-$run.json > $OUT/veto-$run.log 2>&1
  fi
done
for s in 342101 342202; do
  [ -f $OUT/diag-control-s$s/diag.json ] || timeout -k 30 1800 $P scripts/diag_storey_v429.py --source $S --run $CTRL/$CP$s --out $OUT/diag-control-s$s --trainer train_3d_v431 > $OUT/diag-control-s$s.log 2>&1
  [ -s $OUT/diag-interior-control-s$s.json ] || timeout -k 30 1800 $P scripts/diag_interior_v429.py --source $S --run $CTRL/$CP$s --trainer train_3d_v431 2>$OUT/diag-interior-control-s$s.err | tail -1 > $OUT/diag-interior-control-s$s.json
  [ -s $OUT/veto-control-s$s.json ] || CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v429.py --source $S --run $CTRL/$CP$s --trainer train_3d_v431 --out $OUT/veto-control-s$s.json > $OUT/veto-control-s$s.log 2>&1
done
RES=$($P - "$CTRL" "$CP" <<'PY'
import json, math, re, sys, torch
OUT, CTRL, CP = "runs/v431-rung", sys.argv[1], sys.argv[2]
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def load(d):
    f = json.load(open(f"{d}/final.json")); m = json.load(open(f"{d}/run.json"))
    if f["updates"] != m["updates"] or f["updates"] != 16000: raise ValueError(f"incomplete: {f['updates']} of {m['updates']}")
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], blocks=m["blocks"], decoder=m.get("decoder", "d3"),
             decoder_width=m.get("decoder_width", 1.0), precision=m.get("precision", "learned"), known_weight=m.get("known_weight", "uniform"), manifest_sha=m["cache"]["manifest_sha256"], params=m["parameters"],
             wire_bits=m.get("wire_bits", [8, 8]), raw_code_bytes=m.get("raw_code_bytes", 512), D=m["D"],
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
    ks = [k for k in c if not k.startswith("decoder.")]                                              # trunk / head drawn first: identical for every layout / width
    return all(k in a and a[k].shape == c[k].shape and torch.equal(a[k], c[k]) for k in ks) and len(ks) > 10
ARMS = {"z1": ("64x64x1", [8, 8], 8192)}
rows = {}
for s in ("342101", "342202"):
    try:
        c = extras(load(f"{CTRL}/{CP}{s}"), f"{OUT}/diag-control-s{s}", f"{OUT}/diag-interior-control-s{s}.json", f"{OUT}/veto-control-s{s}.json"); c.pop("specs_", None)
        assert c["blocks"] == [32, 32, 1] and c["wire_bits"] == [8, 8] and c["decoder"] == "d3" and c["decoder_width"] == 2.0 and c["raw_code_bytes"] == 2048, "control recipe (L1)"
        rows[f"ctrl-{s}"] = c
    except Exception as e:
        rows[f"ctrl-{s}"] = f"INCONCLUSIVE ({e})"
    for arm, (layout, bits, nbytes) in ARMS.items():
        try:
            a = extras(load(f"{OUT}/{arm}-s{s}"), f"{OUT}/diag-{arm}-s{s}", f"{OUT}/diag-interior-{arm}-s{s}.json", f"{OUT}/veto-{arm}-s{s}.json"); c = rows[f"ctrl-{s}"]
            assert isinstance(c, dict), "control row missing"
            assert "x".join(str(v) for v in a["blocks"]) == layout and a["wire_bits"] == bits and a["raw_code_bytes"] == nbytes, "arm layout / bits / bytes"
            assert a["decoder"] == c["decoder"] and a["decoder_width"] == c["decoder_width"] and a["precision"] == c["precision"] and a["known_weight"] == c["known_weight"], "recipe"
            assert a["manifest_sha"] == c["manifest_sha"] and a["specs"] == c["specs"], "cache / VAL panel"
            assert a["params"]["trunk"] == c["params"]["trunk"] and a["params"]["head"] == c["params"]["head"] and identity(f"{OUT}/{arm}-s{s}", f"{CTRL}/{CP}{s}"), "trunk / head identity"
            rows[f"{arm}-{s}"] = a
        except Exception as e:
            rows[f"{arm}-{s}"] = f"INCONCLUSIVE ({e})"
for k, r in rows.items():
    if isinstance(r, dict): r.pop("specs", None)
def compare(arm):
    try:
        rr = [(rows[f"{arm}-{s}"], rows[f"ctrl-{s}"]) for s in ("342101", "342202")]
        if not all(isinstance(a, dict) and isinstance(b, dict) for a, b in rr): return dict(verdict="INCONCLUSIVE (missing rows)")
        deltas = [{k: round(a[k] - b[k], 4) for k in a if k in b and isinstance(a[k], float) and isinstance(b[k], float)} for a, b in rr]
        ds = [d["R8_surface_f1"] for d in deltas]; dv = [d["R8_var_occ_f1"] for d in deltas]
        if min(dv) >= .03 and min(ds) >= .01: v = "lever"
        elif max(ds) <= -.01 or max(dv) <= -.02: v = "worse"
        elif max(ds) < .01 and max(dv) < .02: v = "not better"
        else: v = "ambiguous"
        guards = {}
        for name, fn in (("fpr_ok", lambda a, b: a["R8_unknown_known_fpr"] <= b["R8_unknown_known_fpr"] + .003), ("slab_ok", lambda a, b: a["R8_slab_recall"] >= b["R8_slab_recall"] - .02), ("k0var_ok", lambda a, b: a["K0_var_occ_f1"] >= b["K0_var_occ_f1"] - .02)):
            try: guards[name] = all(fn(a, b) for a, b in rr)
            except Exception as e: guards[name] = f"n/a ({e})"
        mech = {}
        for key in ("R8_furn_f1", "K0_furn_f1", "R8_obj_recall", "K0_interior", "K8_interior"):
            try: mech[key] = [(round(a[key], 4), round(b[key], 4)) for a, b in rr]
            except Exception: mech[key] = None
        adopt = v == "lever" and all(guards.get(g) is True for g in ("fpr_ok", "slab_ok", "k0var_ok"))        # review M2: all three guards, n/a = failed
        return dict(verdict_raw=v, verdict=v if adopt or v != "lever" else "lever, but a guard failed (FPR, slab or K0 var): not adopted", adopt=adopt, deltas=dict(zip(("342101", "342202"), deltas)), guards=guards, mechanism=mech, bytes_per_message=rr[0][0]["raw_code_bytes"], D=rr[0][0]["D"],
                    mean_R8_var=sum(a["R8_var_occ_f1"] for a, _ in rr) / 2)
    except Exception as e:
        return dict(verdict=f"INCONCLUSIVE ({e})")
res = {arm: compare(arm) for arm in ARMS}
ladder = "rung 64x64x1 vs 32x32x1: " + str(res["z1"].get("verdict"))
cands = [a for a in ARMS if res[a].get("adopt") is True]
nb = max(cands, key=lambda a: res[a]["mean_R8_var"]) if cands else None
nb_dir, nb_prefix = (OUT, f"{nb}-s") if nb else (CTRL, CP)
def absolute(arm):
    try:
        g = lambda k: [rows[f"{arm}-{s}"][k] for s in ("342101", "342202")]
        structure = min(g("R8_surface_f1")) >= .80 and min(g("K0_surface_f1")) >= .85 and min(g("R8_var_occ_f1")) >= .65 and min(g("K0_var_occ_f1")) >= .70
        furniture = min(g("R8_furn_f1")) >= .50 and min(g("K0_furn_f1")) >= .55
        final = min(g("R8_surface_f1")) >= .90 and min(g("K0_surface_f1")) >= .95 and min(g("R8_var_occ_f1")) >= .75
        return dict(batch_criteria=("maps furnished storeys at R1" if structure and furniture else "maps the storey structure, not its furniture (partial)" if structure
                    else "does not with this model" if min(g("R8_surface_f1")) < .70 or min(g("R8_var_occ_f1")) < .50 else "partial"), user_final_targets_met=final)
    except Exception as e:
        return f"n/a ({e})"
json.dump(dict(control=CTRL, control_prefix=CP, rows=rows, arms=res, rung_reading=ladder, absolute={a: absolute(a) for a in ARMS}, next_baseline_dir=nb_dir, next_baseline_prefix=nb_prefix, next_baseline_arm=nb,
               next_baseline_updates=16000, bytes_per_message={a: ARMS[a][2] for a in ARMS}), open(f"{OUT}/analysis.json", "w"), indent=1)
print("RUNG:", ladder, "|", {a: (res[a].get("verdict"), res[a].get("guards"), res[a].get("deltas")) for a in ARMS}, "| next baseline", nb_dir, nb_prefix, "| absolute", {a: absolute(a) for a in ARMS})
PY
)
log "batch 3S-Z analysis: $RES"
