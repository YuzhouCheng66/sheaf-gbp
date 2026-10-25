#!/bin/bash
# Day-6 batch 3S-D (Claude; written 2026-10-25 14:12 UTC while 3S-Q runs, before any 3S-D training). Pre-registered in OVERNIGHT_REPORT
# D8-20. The user (14:1x UTC) approved diagnostic arms whose message exceeds 512 B; the deployment budget is chosen afterwards.
# Question (D8-19): what is the ESSENTIAL limit of the code -- the number of coordinates (spatial units), the bits per coordinate, or
# neither (then the decoder / training)? Every confirmed lever so far was 'more independent spatial units' (3S-C) or 'enough decoder
# channels' (3S-W / D2); the largest remaining gap is furniture (.30 -> ~.7 needed for the user's K0 >= .95), and one 8-bit scalar per
# column cannot hold a wall's in-block position AND several furniture profiles. A ladder lets the data answer:
#   L1  --layout 32x32x1 --wire-bits 8,8     D = 1,024, 2,048 B per message: 4x the coordinates, same bits each (L/32 columns)
#   L2  --layout 16x16x1 --wire-bits 16,16   D = 256,   1,024 B: same coordinates, 2x the bits each (32,767 mean levels, 65,535 precision levels)
#   L3  --layout 16x16x2 --wire-bits 4,4     D = 512,   512 B: 2x the coordinates (two vertical layers) at half the bits (7 mean levels)
#   control: the baseline named by 3S-Q's analysis (16x16x1 @ 8+8, 512 B, d3 W2, 16k updates; its precision / known-weight knobs copied).
# Decoder d3 W2 for every arm; its upsampling schedule starts at the code grid when that is finer than 16 x 16 (32x32x1: start 32 x 32 x 8,
# ONE upsampling stage, decoder 1.50 M params instead of 1.83 M -- disclosed; unchanged for the other arms). Trunk / head identical for
# 16x16x1 arms; the 32x32x1 trunk sees 2 x 2 x 16-cell patches and the 16x16x2 trunk 4 x 4 x 8 (same parameter counts). The source
# quantiser follows the mean bits (LEVELS = 2^(M-1) - 1). Everything else: v420 data module and cache, seeds 342101 / 342202, 16,000
# updates, lr 3e-4 cosine, N = 8, panels.
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = arm - control; the usual rule per arm):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => lever; <= -1 / <= -2 on both => worse;
#   < +1 AND < +2 on both => not better; otherwise ambiguous. Guards: R8 FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp;
#   K0 variable-structure F1 >= control - 2 pp. A lever whose guard fails does not become the baseline.
#   LADDER READING (pre-registered): L1 lever and L2 not => the essential limit is the number of coordinates (deploy by choosing bits to
#   fit the budget); L2 lever and L1 not => the quantisation of the code; both levers => both bind (compare sizes); neither => the code
#   is not the limit at 16k updates: decoder depth / training / data next. L3 vs control reads whether coordinates can be bought with
#   bits at a fixed 512 B. Mechanism readings: furniture-region F1 and furniture recall (the ladder targets furniture), interior-wall
#   recall, the probe; bytes per message recorded per arm. The absolute 3S-A criteria and the user's final targets (R8 >= .90, K0 >= .95,
#   var >= .75) are reported per arm.
# Next baseline (pre-registered): among the levers (guards passed) the arm with the higher mean R8 variable-structure F1; none => unchanged.
# Budget: SOLO, after chain_day6_3ST; one lease per run (six runs, ~1 h each at ~0.2 s/update); must run 16,000 updates (preflight; else
#   INCONCLUSIVE); hard timeouts inside the lease. Writes only under /mnt/datasets (runs/v429-ladder).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v429-ladder
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SD started pid $$ sha256 $(sha256sum scripts/chain_day6_3SD.sh | cut -c1-16)"
check_shas(){
  TSHA=$(sha256sum scripts/train_3d_v429.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v427.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v427.py | cut -d' ' -f1); VSHA=$(sha256sum scripts/diag_veto_v427.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "65b6bed974c314880460394fd9446b9c1a96a3b49cc89fce6eb592417e132621" ] || { log "ABORT: train_3d_v429.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "ca6cd3faead242cb0754daa7f65eb8a0752d963ab61d2bf730354e34eec3b520" ] || { log "ABORT: diag_storey_v427.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "27460a321ab3152040ab7041a2ede6d16ff3da7ae2f53d9281a0cf52b3ff9fb9" ] || { log "ABORT: diag_interior_v427.py sha changed ($ISHA)"; exit 2; }
  [ "$VSHA" = "03d087fc9d76a5e499895f7da15dc97a8377543b559d243a6710e1dd844bfaa3" ] || { log "ABORT: diag_veto_v427.py sha changed ($VSHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the controls' ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day6_3SD already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/l*-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-D history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3ST[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SQ[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SR[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SP[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v42[01][.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
# ---- baseline from the previous batch's pre-registered analysis: label, control directory, run prefix
read BASE CTRL CP <<<"$($P - <<'PY'
import json
try:
    a = json.load(open("runs/v427-covq/analysis.json"))
    print(a["next_baseline_dir"].split("/")[-1], a["next_baseline_dir"], a["next_baseline_prefix"])
except Exception as e:
    print("missing", "", "")
PY
)"
[ -n "$CTRL" ] && [ -d "$CTRL" ] && [ -n "$CP" ] || { log "ABORT: baseline '$BASE' -> control '$CTRL' prefix '$CP' not usable"; exit 2; }
for s in 342101 342202; do [ -f $CTRL/$CP$s/final.json ] && [ -f $CTRL/$CP$s/checkpoints/initial.pt ] || { log "ABORT: control $CTRL/$CP$s (final / initial) missing"; exit 2; }; done
read DEC WID LAY PREC KW <<<"$($P -c "import json; m=json.load(open('$CTRL/${CP}342101/run.json')); print(m.get('decoder','d3'), m.get('decoder_width',1.0), m.get('layout', 'x'.join(str(v) for v in m['blocks'])), m.get('precision','learned'), m.get('known_weight','uniform'))")"
[ "$DEC" = "d3" ] && [ "$WID" = "2.0" ] && [ "$LAY" = "16x16x1" ] || { log "ABORT: the ladder needs a d3 W2 16x16x1 baseline (got $DEC W $WID $LAY)"; exit 2; }
log "batch 3S-D gate: baseline '$BASE' = $CTRL/$CP* (decoder $DEC W $WID layout $LAY precision $PREC known-weight $KW); arms L1 32x32x1@8,8 L2 16x16x1@16,16 L3 16x16x2@4,4"
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
for arm in "l1 32x32x1 8,8" "l2 16x16x1 16,16" "l3 16x16x2 4,4"; do
  set -- $arm; ARM=$1; LAYOUT=$2; BITS=$3
  COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAYOUT --allow-non256 --wire-bits $BITS --decoder $DEC --decoder-width $WID --precision $PREC --known-weight $KW"
  for seed in 342101 342202; do
    check_shas
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-D: GPU busy before $ARM seed $seed; waiting"; sleep 120; done
    TS2=$(date -u +%H%M%S)
    timeout -k 60 1800 $P scripts/train_3d_v429.py $COMMON --out $OUT/pf-$TS2-$ARM-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-$ARM-s$seed.log 2>&1
    PFX=$?
    read RS LS VS <<<"$(rate $OUT/pf-$TS2-$ARM-s$seed.log)"
    [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): $ARM seed $seed"; continue; }
    FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
    log "batch 3S-D preflight $ARM seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
    [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): $ARM seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
    UPD=16000
    START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
    cat > $OUT/lease_$ARM-s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "updates": $UPD, "seed": $seed,
 "batch": "3S-D", "arm": "$ARM", "layout": "$LAYOUT", "wire_bits": "$BITS", "baseline": "$BASE", "baseline_recipe": {"decoder": "$DEC", "decoder_width": $WID, "layout": "$LAY", "precision": "$PREC", "known_weight": "$KW"}, "preflight": "rate=$RS load=$LS val=$VS",
 "trainer": "scripts/train_3d_v429.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "diag_veto_sha256": "$VSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/$CP$seed",
 "expectation": "per arm, paired vs control: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => lever; <= -1pp / <= -2pp on both => worse; < +1pp and < +2pp on both => not better; else ambiguous; guards FPR <= control + 0.3pp, slab recall >= control - 2pp, K0 var >= control - 2pp; ladder: L1 lever and not L2 => coordinates; L2 lever and not L1 => quantisation; neither => code not the limit; next baseline = lever with higher mean R8 var among guard-passing arms; incomplete => INCONCLUSIVE"}
JSON
    log "batch 3S-D start $ARM seed $seed ($UPD updates, $LAYOUT @ $BITS bits on d3 W $WID, $PREC, $KW)"
    timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v429.py $COMMON --out $OUT/$ARM-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/$ARM-s$seed.log 2>&1
    log "3S-D $ARM seed $seed done (exit $?)"
  done
done
log "batch 3S-D training done"
check_shas
for run in l1-s342101 l1-s342202 l2-s342101 l2-s342202 l3-s342101 l3-s342202; do      # report-only breakdowns (CPU for the veto)
  if [ -f $OUT/$run/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v427.py --source $S --run $OUT/$run --out $OUT/diag-$run --trainer train_3d_v429 > $OUT/diag-$run.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v427.py --source $S --run $OUT/$run --trainer train_3d_v429 2>$OUT/diag-interior-$run.err | tail -1 > $OUT/diag-interior-$run.json
    CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v427.py --source $S --run $OUT/$run --trainer train_3d_v429 --out $OUT/veto-$run.json > $OUT/veto-$run.log 2>&1
  fi
done
for s in 342101 342202; do
  [ -f $OUT/diag-control-s$s/diag.json ] || timeout -k 30 1800 $P scripts/diag_storey_v427.py --source $S --run $CTRL/$CP$s --out $OUT/diag-control-s$s --trainer train_3d_v429 > $OUT/diag-control-s$s.log 2>&1
  [ -s $OUT/diag-interior-control-s$s.json ] || timeout -k 30 1800 $P scripts/diag_interior_v427.py --source $S --run $CTRL/$CP$s --trainer train_3d_v429 2>$OUT/diag-interior-control-s$s.err | tail -1 > $OUT/diag-interior-control-s$s.json
  [ -s $OUT/veto-control-s$s.json ] || CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v427.py --source $S --run $CTRL/$CP$s --trainer train_3d_v429 --out $OUT/veto-control-s$s.json > $OUT/veto-control-s$s.log 2>&1
done
RES=$($P - "$CTRL" "$CP" <<'PY'
import json, math, re, sys, torch
OUT, CTRL, CP = "runs/v429-ladder", sys.argv[1], sys.argv[2]
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
ARMS = {"l1": ("32x32x1", [8, 8], 2048), "l2": ("16x16x1", [16, 16], 1024), "l3": ("16x16x2", [4, 4], 512)}
rows = {}
for s in ("342101", "342202"):
    try:
        c = extras(load(f"{CTRL}/{CP}{s}"), f"{OUT}/diag-control-s{s}", f"{OUT}/diag-interior-control-s{s}.json", f"{OUT}/veto-control-s{s}.json"); c.pop("specs_", None)
        assert c["blocks"] == [16, 16, 1] and c["wire_bits"] == [8, 8] and c["decoder"] == "d3" and c["decoder_width"] == 2.0, "control recipe"
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
        adopt = v == "lever" and guards.get("fpr_ok") is True and guards.get("k0var_ok") is True
        return dict(verdict=v if adopt or v != "lever" else "lever, but a guard failed (FPR or K0 var): not adopted", adopt=adopt, deltas=dict(zip(("342101", "342202"), deltas)), guards=guards, mechanism=mech, bytes_per_message=rr[0][0]["raw_code_bytes"], D=rr[0][0]["D"],
                    mean_R8_var=sum(a["R8_var_occ_f1"] for a, _ in rr) / 2)
    except Exception as e:
        return dict(verdict=f"INCONCLUSIVE ({e})")
res = {arm: compare(arm) for arm in ARMS}
def is_lever(arm): return res[arm].get("adopt") is True or res[arm].get("verdict") == "lever"
l1, l2 = is_lever("l1"), is_lever("l2")
ladder = ("the essential limit is the NUMBER OF COORDINATES (deploy by choosing bits to fit the budget)" if l1 and not l2 else
          "the essential limit is the QUANTISATION of the code (bits per coordinate)" if l2 and not l1 else
          "both coordinates and bits bind (compare the sizes)" if l1 and l2 else
          "the code is not the limit at 16k updates: decoder depth / training / data next")
if any(not isinstance(res[a].get("verdict"), str) or res[a]["verdict"].startswith("INCONCLUSIVE") for a in ("l1", "l2")): ladder = "INCONCLUSIVE (L1 or L2 incomplete)"
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
json.dump(dict(control=CTRL, control_prefix=CP, rows=rows, arms=res, ladder_reading=ladder, absolute={a: absolute(a) for a in ARMS}, next_baseline_dir=nb_dir, next_baseline_prefix=nb_prefix, next_baseline_arm=nb,
               next_baseline_updates=16000, bytes_per_message={a: ARMS[a][2] for a in ARMS}), open(f"{OUT}/analysis.json", "w"), indent=1)
print("LADDER:", ladder, "|", {a: (res[a].get("verdict"), res[a].get("guards"), res[a].get("deltas")) for a in ARMS}, "| next baseline", nb_dir, nb_prefix, "| absolute", {a: absolute(a) for a in ARMS})
PY
)
log "batch 3S-D analysis: $RES"
