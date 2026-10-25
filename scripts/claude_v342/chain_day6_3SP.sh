#!/bin/bash
# Day-6 batch 3S-P (Claude; written 2026-10-25 07:03 UTC while 3S-C seed 2 ran, revised 07:29 UTC after code_review_v427_plancol.md M1-M2 and
# S1-S7, before any plan-column training; 3S-L is running).
# Pre-registered in OVERNIGHT_REPORT D8-10; programme in D8-9 (three independent first-principles analysts, claude_handoff/analysis_20261025/).
# Question: is the DECODER FAMILY the lever -- the 2.5-D PLAN + COLUMN decoder (analyst C, R1) against the 3-D convolutional decoder?
# Walls are vertical extrusions of the plan and column profiles are low-dimensional; the 3-D decoder spends 95 % of its MACs redrawing
# the plan on every z slice with 3x3x3 kernels and reads the code with only 3 convolutions at code resolution (2-D: 6), without the block
# phase planes that the 2-D line needed (batch V). The plan + column decoder (train_3d_v427 --decoder plan_column) reads the posterior as a
# 2-D map with 2 BZ channels, draws the plan with the 2-D line's working decoder shape (--decoder-width W: pre 64W, lift 128W, stages
# 96W/64W/48W/32W, public planes x, y, L/25.6 and the block phase u, v at every stage) and emits every column's 2 NZ logits from one
# 1x1 convolution. Reads only the posterior + public coordinates. Mechanism evidence: the wall-relocation probe (probe_wall_v425.py):
# interior walls drawn at 0/16 (3S-A), 11/16 and 5/16 (3S-W) of the in-block positions; boundary walls 16/16; the code's mu is an even
# function of the in-block position in every trained model.
# Baseline / control (gate on 3S-C's pre-registered verdict, D8-5): if 3S-C found the layout a lever -> layout 16x16x1, control 3S-C
#   (runs/v423-layout, d3 W1); otherwise -> layout 8x8x4, control 3S-W (runs/v422-decoder, d3 W2).
# Arms (sequential, each seed under its own inclusive 7,200 s lease). The gate took the 16x16x1 branch (3S-C: "the code layout is a lever
# (confirmed)", 07:10 UTC), so the control is d3 W1 and a plan-column arm at the 2-D line's width W2 would change family AND width at once
# (review M1). Therefore:
#   P2  --decoder plan_column --decoder-width 2.0   (the 2-D line's recipe)
#   D2  --decoder d3          --decoder-width 2.0   (the 3-D decoder at W2: the STACK of the two confirmed levers, width (3S-W) + layout
#                                                     (3S-C); it is the single-variable control for P2's decoder FAMILY at equal width)
#   Three paired comparisons: FAMILY = P2 - D2 (one variable: decoder family at W2; PRIMARY); WIDTH@16x16x1 = D2 - 3S-C (one variable:
#   width; the stack check); RECIPE = P2 - 3S-C (family + width together, disclosed as two variables). Everything else is the control's:
#   v420 data module and cache, seeds, 16,000 updates, lr, N = 8, panels, trunk / head initial tensors (asserted equal to the control's
#   initial.pt), learned precision, uniform known-BCE. (Against a 3S-W control -- the other gate branch -- P2 alone is single-variable and
#   D2 is not run.)
# Pre-registered reading (final VAL, 24 houses, paired by seed; the same rule for each of the three comparisons, Delta = first - second):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +1 pp on BOTH seeds => lever;
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => worse;
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds => not better;
#   otherwise => ambiguous. The control's two seeds are 5.4 / 6.3 pp apart (R8 surface / var), so "ambiguous" is the likely label for a
#   moderate effect (review S4); the paired deltas are reported either way.
#   Guards (reported beside each verdict): R8 FPR <= second arm + 0.3 pp; R8 slab recall >= second arm - 2 pp (column-readout failure
#   signal; relative, review S1). Mechanism reading for P2: Delta K0 interior-wall recall >= +10 pp vs the second arm on both seeds =>
#   the gain comes with interior walls being placed; the wall-relocation probe (block 7 of 16, 8 positions, review S2) is reported.
#   The absolute 3S-A criteria are re-applied to every arm for the record.
#   Reported, not judged: Delta K0, furniture-region F1, tolerance-1, fresh, the fusion gap, the D8-3 breakdown, the gradient-clip share,
#   s/update and parameter counts, compression (unchanged: 512 B per message).
# Pre-registered next step: the baseline for 3S-R (known-BCE boundary weighting, analyst B), then 3S-Q (coverage precision, analyst A),
#   then 3S-T (training length), is the arm among {P2, D2} that is a LEVER against 3S-C on both seeds with the higher mean R8 variable-
#   structure F1; if neither is a lever against 3S-C, the baseline stays 3S-C.
# Budget: SOLO, after chain_day6_3SL; must run the control's 16,000 updates (preflight; else INCONCLUSIVE); hard timeouts inside the lease.
#   Writes only under /mnt/datasets (runs/v427-plancol).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v427-plancol
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SP started pid $$ sha256 $(sha256sum scripts/chain_day6_3SP.sh | cut -c1-16)"
check_shas(){
  TSHA=$(sha256sum scripts/train_3d_v427.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v427.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v427.py | cut -d' ' -f1); PSHA=$(sha256sum scripts/probe_wall_v425.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "7003380f931aa22d3f2c2cf93b6269cb250edb0c46f433be35c64ea5bbd63b5d" ] || { log "ABORT: train_3d_v427.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "ca6cd3faead242cb0754daa7f65eb8a0752d963ab61d2bf730354e34eec3b520" ] || { log "ABORT: diag_storey_v427.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "27460a321ab3152040ab7041a2ede6d16ff3da7ae2f53d9281a0cf52b3ff9fb9" ] || { log "ABORT: diag_interior_v427.py sha changed ($ISHA)"; exit 2; }
  [ "$PSHA" = "6e8805e1e67d579378a20f7f619536d8068386d4bb67bffe016a075625e17865" ] || { log "ABORT: probe_wall_v425.py sha changed ($PSHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the controls' ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
mkdir runs/v427-plancol.lock 2>/dev/null || { log "ABORT: chain_day6_3SP already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/p*-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-P history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SW[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v42[01][.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
V3SC=$($P -c "import json; print(json.load(open('runs/v423-layout/analysis.json'))['verdict'])" 2>/dev/null || echo "missing")
case "$V3SC" in "the code layout is a lever"*) LAYOUT=16x16x1; CTRL=runs/v423-layout; CTRLW=1.0;; *) LAYOUT=8x8x4; CTRL=runs/v422-decoder; CTRLW=2.0;; esac
log "batch 3S-P gate: 3S-C verdict '$V3SC' -> layout $LAYOUT, control $CTRL"
for s in 342101 342202; do [ -f $CTRL/a-s$s/final.json ] && [ -f $CTRL/a-s$s/checkpoints/initial.pt ] && [ -f $CTRL/diag-interior-a-s$s.json ] || { log "ABORT: control $CTRL/a-s$s (final / initial / diag-interior) missing"; exit 2; }; done
log "batch 3S-P: GPU free; starting (SOLO, arms P2 then D2, seeds sequential, one lease per run)"
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
ARMS="p2 plan_column 2.0"; [ "$CTRLW" = "1.0" ] && ARMS="$ARMS|d2 d3 2.0"
IFS='|'; for arm in $ARMS; do IFS=' '
  set -- $arm; ARM=$1; DEC=$2; WIDTH=$3
  COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAYOUT --decoder $DEC --decoder-width $WIDTH --precision learned --known-weight uniform"
  for seed in 342101 342202; do
    check_shas
    while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-P: GPU busy before $ARM seed $seed; waiting"; sleep 120; done
    TS2=$(date -u +%H%M%S)
    timeout -k 60 1800 $P scripts/train_3d_v427.py $COMMON --out $OUT/pf-$TS2-$ARM-s$seed --seed $seed --updates 16000 --max-updates 100 --val-every 100 --deadline-total 0 > $OUT/pf-$TS2-$ARM-s$seed.log 2>&1
    PFX=$?
    read RS LS VS <<<"$(rate $OUT/pf-$TS2-$ARM-s$seed.log)"
    [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): $ARM seed $seed"; continue; }
    FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+16000*r+3*float('$VS'))<=6000 else 0)")
    log "batch 3S-P preflight $ARM seed $seed (exit $PFX): s/update $RS load $LS val $VS -> fits 16,000 updates: $FIT"
    [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): $ARM seed $seed cannot run the control's 16,000 updates in a lease"; continue; }
    UPD=16000
    START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
    cat > $OUT/lease_$ARM-s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": 256, "updates": $UPD, "seed": $seed,
 "arm": "$ARM", "decoder": "$DEC", "decoder_width": $WIDTH, "layout": "$LAYOUT", "preflight": "rate=$RS load=$LS val=$VS", "trainer": "scripts/train_3d_v427.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA",
 "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "probe_sha256": "$PSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/a-s$seed", "control_width": $CTRLW, "gate_3SC_verdict": "$V3SC",
 "expectation": "three paired comparisons FAMILY = P2 - D2 (primary), WIDTH = D2 - 3S-C, RECIPE = P2 - 3S-C, same rule each: Delta R8 var >= +3pp and Delta R8 surface >= +1pp on both seeds => lever; Delta R8 surface <= -1pp on both or Delta R8 var <= -2pp on both => worse; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => not better; else ambiguous; guards FPR <= second + 0.3pp, slab recall >= second - 2pp; mechanism P2: Delta K0 interior recall >= +10pp on both seeds; next baseline = lever vs 3S-C with higher mean R8 var among P2 / D2, else 3S-C; incomplete => INCONCLUSIVE"}
JSON
    log "batch 3S-P start $ARM seed $seed ($UPD updates, $DEC W $WIDTH, layout $LAYOUT)"
    timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v427.py $COMMON --out $OUT/$ARM-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/$ARM-s$seed.log 2>&1
    log "3S-P $ARM seed $seed done (exit $?)"
  done
done
log "batch 3S-P training done"
check_shas
for run in p2-s342101 p2-s342202 d2-s342101 d2-s342202; do                        # report-only breakdowns (D8-3 protocol) + the probe
  if [ -f $OUT/$run/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v427.py --source $S --run $OUT/$run --out $OUT/diag-$run --trainer train_3d_v427 > $OUT/diag-$run.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v427.py --source $S --run $OUT/$run --trainer train_3d_v427 2>$OUT/diag-interior-$run.err | tail -1 > $OUT/diag-interior-$run.json
    CUDA_VISIBLE_DEVICES= timeout -k 30 900 $P scripts/probe_wall_v425.py --source $S --run $OUT/$run --trainer train_3d_v427 --block 7 --out $OUT/probe-$run.json > $OUT/probe-$run.log 2>&1
  fi
done
for s in 342101 342202; do                                                          # the control's probe (CPU, seconds)
  CT=$([ "$CTRL" = "runs/v423-layout" ] && echo train_3d_v423 || echo train_3d_v422)
  CUDA_VISIBLE_DEVICES= timeout -k 30 900 $P scripts/probe_wall_v425.py --source $S --run $CTRL/a-s$s --trainer $CT --block 7 --out $OUT/probe-control-s$s.json > $OUT/probe-control-s$s.log 2>&1
done
RES=$($P - "$CTRL" <<'PY'
import json, math, re, sys, torch
OUT, CTRL = "runs/v427-plancol", sys.argv[1]
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def val_rows(d): return json.load(open(f"{d}/validation/final-val.json"))["rows"]
def load(d):
    f = json.load(open(f"{d}/final.json")); m = json.load(open(f"{d}/run.json"))
    if f["updates"] != m["updates"] or f["updates"] != 16000: raise ValueError(f"incomplete: {f['updates']} of {m['updates']}")
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], blocks=m["blocks"], decoder=m.get("decoder", "d3"),
             decoder_width=m.get("decoder_width", 1.0), precision=m.get("precision", "learned"), known_weight=m.get("known_weight", "uniform"), manifest_sha=m["cache"]["manifest_sha256"], params=m["parameters"],
             specs=[json.dumps(x["spec"], sort_keys=True) for x in val_rows(d)], seconds=round(f["seconds"]))
    logtxt = open(f"{d}.log").read(); norms = []
    for gd in re.findall(r"grad (\{[^}]*\})", logtxt):
        g = json.loads(gd.replace("'", '"')); norms.append(math.sqrt(sum(x * x for x in g.values())))
    r["clip_fraction_logged"] = round(sum(n > 5 for n in norms) / max(1, len(norms)), 4)
    return r
def identity(arm_dir, ctrl_dir):
    a = torch.load(f"{arm_dir}/checkpoints/initial.pt", map_location="cpu", weights_only=False)["state_dict"]; c = torch.load(f"{ctrl_dir}/checkpoints/initial.pt", map_location="cpu", weights_only=False)["state_dict"]
    ks = [k for k in c if not k.startswith("decoder.")]
    return all(k in a and torch.equal(a[k], c[k]) for k in ks) and len(ks) > 10
runs = {}                                                                      # every arm and the control as a row per seed
def row(d, s, diag_prefix=None, probe_path=None):
    r = load(d); r.pop("specs_keep", None)
    try:
        xi = json.load(open(diag_prefix + f"interior-{s}.json" if diag_prefix else f"{CTRL}/diag-interior-a-s{s}.json"))
        r["K0_interior"] = xi["K0_interior"]["recall"]; r["K8_interior"] = xi["K8_interior"]["recall"]; r["K0_boundary"] = xi["K0_boundary"]["recall"]
    except Exception as e:
        r["K0_interior"] = None; r["interior_err"] = str(e)
    try:
        xd = json.load(open((diag_prefix + f"{s}/diag.json") if diag_prefix else f"{CTRL}/diag-a-s{s}/diag.json")); r["R8_slab_recall"] = xd["K8"]["recall_slab"]; r["R8_struct_recall"] = xd["K8"]["recall_struct"]
    except Exception as e:
        r["R8_slab_recall"] = None
    try:
        pa = json.load(open(probe_path)); r["probe_interior"] = f"{pa['interior']['drawn']}/{pa['interior']['total']}"; r["probe_boundary"] = f"{pa['boundary']['drawn']}/{pa['boundary']['total']}"
    except Exception as e:
        r["probe_interior"] = None
    return r
for s in ("342101", "342202"):
    try:
        runs[f"ctrl-{s}"] = row(f"{CTRL}/a-s{s}", s, None, f"{OUT}/probe-control-s{s}.json")
    except Exception as e:
        runs[f"ctrl-{s}"] = f"INCONCLUSIVE ({e})"
    for arm in ("p2", "d2"):
        try:
            r = row(f"{OUT}/{arm}-s{s}", s, f"{OUT}/diag-{arm}-s", f"{OUT}/probe-{arm}-s{s}.json"); c = runs[f"ctrl-{s}"]
            assert isinstance(c, dict), "control row missing"
            assert r["blocks"] == c["blocks"] and r["precision"] == c["precision"] == "learned" and r["known_weight"] == c["known_weight"] == "uniform", "recipe"
            assert r["manifest_sha"] == c["manifest_sha"] and r["specs"] == c["specs"], "cache / VAL panel"
            assert r["params"]["trunk"] == c["params"]["trunk"] and r["params"]["head"] == c["params"]["head"] and identity(f"{OUT}/{arm}-s{s}", f"{CTRL}/a-s{s}"), "trunk / head identity"
            runs[f"{arm}-{s}"] = r
        except Exception as e:
            runs[f"{arm}-{s}"] = f"INCONCLUSIVE ({e})"
for k, r in runs.items():
    if isinstance(r, dict): r.pop("specs", None)
def compare(first, second):
    """paired by seed: deltas, verdict, guards, mechanism; None-safe (review S3)"""
    try:
        rr = [(runs[f"{first}-{s}"], runs[f"{second}-{s}"]) for s in ("342101", "342202")]
        if not all(isinstance(a, dict) and isinstance(b, dict) for a, b in rr): return dict(verdict=f"INCONCLUSIVE (missing: {[k for k in (first, second) for s in ('342101','342202') if not isinstance(runs[f'{k}-{s}'], dict)]})")
        deltas = [{k: round(a[k] - b[k], 4) for k in a if k in b and isinstance(a[k], float) and isinstance(b[k], float) and not k.startswith("probe")} for a, b in rr]
        ds = [d["R8_surface_f1"] for d in deltas]; dv = [d["R8_var_occ_f1"] for d in deltas]
        if min(dv) >= .03 and min(ds) >= .01: v = "lever"
        elif max(ds) <= -.01 or max(dv) <= -.02: v = "worse"
        elif max(ds) < .01 and max(dv) < .02: v = "not better"
        else: v = "ambiguous"
        guards = {}
        try: guards["fpr_ok"] = all(a["R8_unknown_known_fpr"] <= b["R8_unknown_known_fpr"] + .003 for a, b in rr)
        except Exception as e: guards["fpr_ok"] = f"n/a ({e})"
        try: guards["slab_ok"] = all(a["R8_slab_recall"] >= b["R8_slab_recall"] - .02 for a, b in rr)
        except Exception as e: guards["slab_ok"] = f"n/a ({e})"
        try: mech = ("interior walls placed (Delta K0 interior >= +10 pp both seeds)" if all(a["K0_interior"] - b["K0_interior"] >= .10 for a, b in rr) else "gain not attributable to interior-wall placement") + f" {[(a['K0_interior'], b['K0_interior']) for a, b in rr]}"
        except Exception as e: mech = f"n/a ({e})"
        return dict(verdict=v, deltas=dict(zip(("342101", "342202"), deltas)), guards=guards, mechanism=mech, probes=[(a.get("probe_interior"), b.get("probe_interior")) for a, b in rr])
    except Exception as e:
        return dict(verdict=f"INCONCLUSIVE ({e})")
family = compare("p2", "d2"); width = compare("d2", "ctrl"); recipe = compare("p2", "ctrl")
def absolute(arm):
    try:
        g = lambda k: [runs[f"{arm}-{s}"][k] for s in ("342101", "342202")]
        structure = min(g("R8_surface_f1")) >= .80 and min(g("K0_surface_f1")) >= .85 and min(g("R8_var_occ_f1")) >= .65 and min(g("K0_var_occ_f1")) >= .70
        furniture = min(g("R8_furn_f1")) >= .50 and min(g("K0_furn_f1")) >= .55
        return ("maps furnished storeys at R1" if structure and furniture else "maps the storey structure, not its furniture (partial)" if structure
                else "does not with this model" if min(g("R8_surface_f1")) < .70 or min(g("R8_var_occ_f1")) < .50 else "partial")
    except Exception as e:
        return f"n/a ({e})"
def mean_var(arm):
    try: return sum(runs[f"{arm}-{s}"]["R8_var_occ_f1"] for s in ("342101", "342202")) / 2
    except Exception: return -1.
cands = [arm for arm, cmp in (("p2", recipe), ("d2", width)) if cmp.get("verdict") == "lever"]
next_baseline = max(cands, key=mean_var) if cands else "ctrl (3S-C)"
json.dump(dict(control=CTRL, runs=runs, family_P2_minus_D2=family, width_at_16x16x1_D2_minus_ctrl=width, recipe_P2_minus_ctrl=recipe,
               absolute=dict(p2=absolute("p2"), d2=absolute("d2"), ctrl=absolute("ctrl")), next_baseline_for_3SR=next_baseline, code_bytes=512), open(f"{OUT}/analysis.json", "w"), indent=1)
print("FAMILY P2-D2:", family.get("verdict"), family.get("guards"), family.get("mechanism"), "| WIDTH D2-ctrl:", width.get("verdict"), width.get("guards"), "| RECIPE P2-ctrl:", recipe.get("verdict"), recipe.get("guards"),
      "| absolute", dict(p2=absolute("p2"), d2=absolute("d2")), "| next baseline:", next_baseline, "|", json.dumps({k: v.get("deltas") for k, v in (("family", family), ("width", width), ("recipe", recipe))}))
PY
)
log "batch 3S-P analysis: $RES"
