#!/bin/bash
# Day-6 batch 3S-T (Claude; written 2026-10-25 08:36 UTC, revised 08:58 after code_review_v427_length.md, REDESIGNED 11:23 UTC after 3S-P
# made d3 W2 16x16x1 (D2) the baseline -- its lease holds only ~25k updates, below the single-lease design's 32k floor (review S4) -- before
# any 3S-T training). Pre-registered in OVERNIGHT_REPORT D8-15 (revised); programme D8-9 (analyst B, 1 / 2 / 5.2: LR-floor-limited, not
# plateaued; VAL still rising at the end of every 16k run; 3S-C seed 1 jumped +13 pp in its last quarter; the 2D line needed 57,600 x 3 cycles).
# ONE variable: the amount of training, delivered as CYCLES (the 2D line's R / S / W / X protocol): cycle 1 trains the baseline's exact recipe
# for UPD updates (one lease filled: largest multiple of 500 with 1.05 x (load + UPD x rate + 3 x val) <= 6,000 s, cap 96,000, floor 24,000 =
# 1.5x the control; UPD computed once from seed 342101's 1,100-update preflight and reused for seed 342202); cycle 2 WARM-STARTS from cycle
# 1's final checkpoint (train_3d_v428 --init-from: every tensor loaded, optimiser and schedule fresh) and trains UPD more updates with peak
# lr 1e-4 (the warm cycle's lr, as the 2D protocol) under a NEW lease. Leases are never resumed or extended; each run is its own lease.
# Control = the baseline's own 16,000-update run (same recipe, seeds, initial tensors; same data order for the first 16k updates).
# The BASELINE is read at run time from 3S-Q's analysis (runs/v427-covq/analysis.json: next_baseline_dir / next_baseline_prefix).
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = cycle-2 final - control):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +2 pp on BOTH seeds => training amount is a lever (cycles become
#   part of the recipe);
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => worse (over-fitting: ~20 passes
#   over 4,708 houses; the next lever is data, not training);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds => not better (saturated at this recipe);
#   otherwise => ambiguous.
#   EARLY STOP (pre-registered): after cycle 1, if Delta(cycle-1 final - control) R8 surface < +1 pp on BOTH seeds, cycle 2 is not run and the
#   verdict is read on cycle 1 alone (labelled 'cycle 1 only'); otherwise cycle 2 runs and the verdict is read on cycle 2.
#   Guards (beside the verdict): R8 FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp; K0 variable-structure F1 >= control - 2 pp.
#   A lever whose FPR or K0-var guard fails is recorded as such and does NOT become the next baseline.
#   Mechanism reading: the fusion gap R8 - K0 (surface) must not widen by more than 2 pp versus the control on EITHER seed; reported: the
#   VAL trajectory per quarter of each cycle, the gain in each cycle's LR-floor quarter.
#   The absolute 3S-A criteria are re-applied for the record. Reported, not judged: Delta K0, furniture-region F1, tolerance-1, fresh,
#   the D8-3 breakdown, interior / boundary wall recall, the known-head veto fraction, the gradient-clip share, compression (512 B).
# Identity: cycle-1 initial tensors == the control's initial.pt (all tensors); cycle-2 initial_weights_sha256 == cycle-1 final weights_sha256.
# Budget: SOLO, after chain_day6_3SQ; one lease per run (up to 4); hard timeouts inside each lease. Writes only under /mnt/datasets (runs/v427-length).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v427-length
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3ST started pid $$ sha256 $(sha256sum scripts/chain_day6_3ST.sh | cut -c1-16)"
check_shas(){
  TSHA=$(sha256sum scripts/train_3d_v428.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  GSHA=$(sha256sum scripts/diag_storey_v427.py | cut -d' ' -f1); ISHA=$(sha256sum scripts/diag_interior_v427.py | cut -d' ' -f1); VSHA=$(sha256sum scripts/diag_veto_v427.py | cut -d' ' -f1); MSHA=$(sha256sum $CACHE/manifest.json | cut -d' ' -f1)
  [ "$TSHA" = "ef1e616a746498a941ee99fbeee8f0e700ecea0003d880f788726fcb30ed24b0" ] || { log "ABORT: train_3d_v428.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
  [ "$GSHA" = "ca6cd3faead242cb0754daa7f65eb8a0752d963ab61d2bf730354e34eec3b520" ] || { log "ABORT: diag_storey_v427.py sha changed ($GSHA)"; exit 2; }
  [ "$ISHA" = "27460a321ab3152040ab7041a2ede6d16ff3da7ae2f53d9281a0cf52b3ff9fb9" ] || { log "ABORT: diag_interior_v427.py sha changed ($ISHA)"; exit 2; }
  [ "$VSHA" = "03d087fc9d76a5e499895f7da15dc97a8377543b559d243a6710e1dd844bfaa3" ] || { log "ABORT: diag_veto_v427.py sha changed ($VSHA)"; exit 2; }
  [ "$MSHA" = "864b1fe3c711a189daffd86b841a8e39c959c465f4310daa2d982b0155c5452b" ] || { log "ABORT: cache manifest differs from the controls' ($MSHA)"; exit 2; }
}
[ -f $CACHE/manifest.json ] || { log "ABORT: v420 cache manifest missing"; exit 2; }
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day6_3ST already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-T history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SQ[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SR[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SP[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v42[01][.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
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
log "batch 3S-T gate: baseline '$BASE' = $CTRL/$CP* (decoder $DEC W $WID layout $LAY precision $PREC known-weight $KW); arm = same recipe, one lease of updates"
mkdir -p $OUT
rate(){ $P - "$1" <<'PY'
import json, re, sys
try:
    log = open(sys.argv[1]).read()
    g = re.search(r' gen (\d+)s', log); t = [int(m) for m in re.findall(r'upd(?:100|1100) .* (\d+)s\n', log)]
    h = json.load(open(sys.argv[1].replace('.log', '') + '/history.json'))
    val = max(30., float(h[0]['seconds']) - (t[-1] - int(g.group(1)))) if h else 60.
    print(f"{(t[1]-t[0])/1000:.4f} {g.group(1)} {val:.0f}" if len(t) >= 2 and g else "nan 0 0")
except Exception:
    print("nan 0 0")
PY
}
COMMON="--source $S --cache $CACHE --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAY --decoder $DEC --decoder-width $WID --precision $PREC --known-weight $KW"
run_seed(){                                                                     # $1 = cycle (1 | 2), $2 = seed
  CYC=$1; seed=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-T: GPU busy before c$CYC seed $seed; waiting"; sleep 120; done
  if [ "$CYC" = "2" ]; then INIT="--init-from $OUT/c1-s$seed/checkpoints/final.pt --lr 1e-4"; else INIT=""; fi
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v428.py $COMMON $INIT --out $OUT/pf-$TS2-c$CYC-s$seed --seed $seed --updates 16000 --max-updates 1100 --val-every 1100 --deadline-total 0 > $OUT/pf-$TS2-c$CYC-s$seed.log 2>&1
  PFX=$?
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-c$CYC-s$seed.log)"
  [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): c$CYC seed $seed"; return 1; }
  if [ -z "$UPD" ]; then                                                       # one count for every run, from the first preflight (review M2)
    UPD=$($P -c "import math; r=float('$RS'); u=int((6000/1.05-float('$LS')-3*float('$VS'))/r) if math.isfinite(r) and r>0 else 0; print(min(96000,(u//500)*500))"); UPDFROM="c${CYC}s$seed"
  fi
  FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+$UPD*r+3*float('$VS'))<=6000 else 0)")
  log "batch 3S-T preflight c$CYC seed $seed (exit $PFX): s/update $RS load $LS val $VS -> updates $UPD (from $UPDFROM; cap 96,000; floor 24,000); fits: $FIT"
  [ "$UPD" -ge 24000 ] && [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): c$CYC seed $seed: $UPD updates (floor 24,000) fit=$FIT"; return 1; }
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_c$CYC-s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": 256, "updates": $UPD, "seed": $seed, "cycle": $CYC,
 "batch": "3S-T", "variable": "training amount: cycle $CYC of $UPD updates (control 16,000)", "warm_start": "$INIT", "updates_from": "$UPDFROM", "baseline": "$BASE", "baseline_recipe": {"decoder": "$DEC", "decoder_width": $WID, "layout": "$LAY", "precision": "$PREC", "known_weight": "$KW"}, "preflight": "rate=$RS load=$LS val=$VS",
 "trainer": "scripts/train_3d_v428.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "diag_veto_sha256": "$VSHA", "cache": "$CACHE", "cache_manifest_sha256": "$MSHA", "control": "$CTRL/$CP$seed",
 "expectation": "paired vs control (cycle-2 final, or cycle 1 alone on early stop): Delta R8 var >= +3pp and Delta R8 surface >= +2pp on both seeds => lever; Delta R8 surface <= -1pp on both or Delta R8 var <= -2pp on both => worse; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => not better; else ambiguous; early stop after cycle 1 if Delta R8 surface < +1pp on both seeds; guards FPR <= control + 0.3pp, slab recall >= control - 2pp, K0 var >= control - 2pp; mechanism: fusion gap must not widen by > 2pp on either seed; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-T start c$CYC seed $seed ($UPD updates, $DEC W $WID $LAY $PREC $KW${INIT:+, warm start lr 1e-4})"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v428.py $COMMON $INIT --out $OUT/c$CYC-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/c$CYC-s$seed.log 2>&1
  log "3S-T c$CYC seed $seed done (exit $?)"
}
for seed in 342101 342202; do run_seed 1 $seed; done
EARLY=$($P - "$CTRL" "$CP" <<'PY'
import json, sys
CTRL, CP = sys.argv[1], sys.argv[2]; ds = []
for s in ("342101", "342202"):
    try:
        a = json.load(open(f"runs/v427-length/c1-s{s}/final.json")); c = json.load(open(f"{CTRL}/{CP}{s}/final.json"))
        ds.append(a["val"]["8"]["surface_f1"] - c["val"]["8"]["surface_f1"])
    except Exception as e:
        ds.append(None)
print("stop" if all(d is not None and d < .01 for d in ds) else ("incomplete" if any(d is None for d in ds) else "continue"), [None if d is None else round(d, 4) for d in ds])
PY
)
log "batch 3S-T cycle-1 read (pre-registered early stop if both seeds < +1 pp R8 surface): $EARLY"
case "$EARLY" in stop*|incomplete*) log "batch 3S-T: cycle 2 not run ($EARLY)";; *) for seed in 342101 342202; do run_seed 2 $seed; done;; esac
log "batch 3S-T training done"
check_shas
for s in 342101 342202; do                                                      # report-only breakdowns for the FINAL run of each seed and the control
  if [ -f $OUT/c2-s$s/checkpoints/final.pt ]; then ln -sfn c2-s$s $OUT/a-s$s; ln -sf c2-s$s.log $OUT/a-s$s.log; elif [ -f $OUT/c1-s$s/checkpoints/final.pt ]; then ln -sfn c1-s$s $OUT/a-s$s; ln -sf c1-s$s.log $OUT/a-s$s.log; fi
  if [ -f $OUT/a-s$s/checkpoints/final.pt ]; then
    timeout -k 30 1800 $P scripts/diag_storey_v427.py --source $S --run $OUT/a-s$s --out $OUT/diag-a-s$s --trainer train_3d_v427 > $OUT/diag-a-s$s.log 2>&1
    timeout -k 30 1800 $P scripts/diag_interior_v427.py --source $S --run $OUT/a-s$s --trainer train_3d_v427 2>$OUT/diag-interior-a-s$s.err | tail -1 > $OUT/diag-interior-a-s$s.json
    CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v427.py --source $S --run $OUT/a-s$s --trainer train_3d_v427 --out $OUT/veto-a-s$s.json > $OUT/veto-a-s$s.log 2>&1
  fi
  [ -f $OUT/diag-control-s$s/diag.json ] || timeout -k 30 1800 $P scripts/diag_storey_v427.py --source $S --run $CTRL/$CP$s --out $OUT/diag-control-s$s --trainer train_3d_v427 > $OUT/diag-control-s$s.log 2>&1
  [ -s $OUT/diag-interior-control-s$s.json ] || timeout -k 30 1800 $P scripts/diag_interior_v427.py --source $S --run $CTRL/$CP$s --trainer train_3d_v427 2>$OUT/diag-interior-control-s$s.err | tail -1 > $OUT/diag-interior-control-s$s.json
  [ -s $OUT/veto-control-s$s.json ] || CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P scripts/diag_veto_v427.py --source $S --run $CTRL/$CP$s --trainer train_3d_v427 --out $OUT/veto-control-s$s.json > $OUT/veto-control-s$s.log 2>&1
done
RES=$($P - "$CTRL" "$CP" <<'PY'
import json, math, re, sys, torch
OUT, CTRL, CP = "runs/v427-length", sys.argv[1], sys.argv[2]
KEYS = ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "obj_recall", "occ_f1", "accuracy", "unknown_known_fpr")
def load(d, expect=None):
    f = json.load(open(f"{d}/final.json")); m = json.load(open(f"{d}/run.json"))
    if f["updates"] != m["updates"] or (expect is not None and f["updates"] != expect) or (expect is None and f["updates"] < 24000): raise ValueError(f"incomplete: {f['updates']} of {m['updates']} (expect {expect})")
    try:
        h = json.load(open(f"{d}/history.json")); traj = [(x["updates"], round(x["validation"]["0"]["var_occ_f1"], 4), round(x["validation"]["8"]["surface_f1"], 4), round(x["validation"]["8"]["var_occ_f1"], 4)) for x in h]
    except Exception:
        traj = None
    v = f["val"]; fr = f.get("fresh", {})
    r = {f"{h}_{k}": round(v[h2][k], 4) for h, h2 in (("K0", "0"), ("R8", "8")) for k in KEYS if v[h2][k] is not None}
    r.update(fresh_R8_surface_f1=round(fr["8"]["surface_f1"], 4) if fr else None, fresh_R8_var_occ_f1=round(fr["8"]["var_occ_f1"], 4) if fr else None,
             fresh_R8_furn_f1=round(fr["8"]["furn_f1"], 4) if fr and fr["8"]["furn_f1"] is not None else None, updates=f["updates"], blocks=m["blocks"], decoder=m.get("decoder", "d3"),
             decoder_width=m.get("decoder_width", 1.0), precision=m.get("precision", "learned"), known_weight=m.get("known_weight", "uniform"), manifest_sha=m["cache"]["manifest_sha256"], params=m["parameters"], lr=m["lr"],
             specs=[json.dumps(x["spec"], sort_keys=True) for x in json.load(open(f"{d}/validation/final-val.json"))["rows"]], seconds=round(f["seconds"]), trajectory=traj,
             warm_start=m.get("warm_start"), initial_sha=m["initial_weights_sha256"], final_sha=f["weights_sha256"])
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
        c = extras(load(f"{CTRL}/{CP}{s}", 16000), f"{OUT}/diag-control-s{s}", f"{OUT}/diag-interior-control-s{s}.json", f"{OUT}/veto-control-s{s}.json")
        assert a["precision"] == c["precision"] and a["known_weight"] == c["known_weight"] and a["updates"] >= 24000 and c["updates"] == 16000, "recipe / updates"
        c1 = load(f"{OUT}/c1-s{s}"); assert c1["lr"] == c["lr"] and c1["warm_start"] is None, "cycle 1 recipe"
        cycles = 1
        if a["warm_start"]:                                                       # cycle 2: warm-started from cycle 1's final, lr 1e-4
            assert a["warm_start"]["source_weights_sha256"] == c1["final_sha"] and a["initial_sha"] == c1["final_sha"] and a["lr"] == 1e-4 and a["updates"] == c1["updates"], "cycle 2 warm start"
            cycles = 2
        a["cycles"] = cycles; a["total_updates"] = cycles * c1["updates"]; a["cycle1"] = {k: c1[k] for k in ("R8_surface_f1", "R8_var_occ_f1", "K0_surface_f1", "K0_var_occ_f1", "trajectory")}
        assert a["blocks"] == c["blocks"] and a["decoder"] == c["decoder"] and a["decoder_width"] == c["decoder_width"], "recipe"
        assert a["manifest_sha"] == c["manifest_sha"] and a["specs"] == c["specs"], "cache / VAL panel"
        assert a["params"] == c["params"] and identity(f"{OUT}/c1-s{s}", f"{CTRL}/{CP}{s}"), "identity (params, cycle-1 initial tensors == control)"
        d = {k: round(a[k] - c[k], 4) for k in a if k in c and isinstance(a[k], float) and isinstance(c[k], float)}
        for r in (a, c): r.pop("specs", None)
        rows[s] = dict(arm=a, control=c, delta=d)
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
try:
    if not all(isinstance(r, dict) for r in rows.values()): raise KeyError("incomplete seeds")
    if len({(r["arm"]["updates"], r["arm"]["cycles"]) for r in rows.values()}) != 1: raise ValueError("the two seeds trained for different counts / cycles")
    label = f"cycles {rows['342101']['arm']['cycles']} x {rows['342101']['arm']['updates']}"; verdict_label = label
    d = lambda k: [r["delta"][k] for r in rows.values()]
    if min(d("R8_var_occ_f1")) >= .03 and min(d("R8_surface_f1")) >= .02: verdict = "lever"
    elif max(d("R8_surface_f1")) <= -.01 or max(d("R8_var_occ_f1")) <= -.02: verdict = "worse"
    elif max(d("R8_surface_f1")) < .01 and max(d("R8_var_occ_f1")) < .02: verdict = "not better"
    else: verdict = "ambiguous"
    guards = {}
    for name, fn in (("fpr_ok", lambda r: r["arm"]["R8_unknown_known_fpr"] <= r["control"]["R8_unknown_known_fpr"] + .003), ("slab_ok", lambda r: r["arm"]["R8_slab_recall"] >= r["control"]["R8_slab_recall"] - .02), ("k0var_ok", lambda r: r["delta"]["K0_var_occ_f1"] >= -.02)):
        try: guards[name] = all(fn(r) for r in rows.values())
        except Exception as e: guards[name] = f"n/a ({e})"
    try:
        mech = ("fusion gap did not widen by > 2 pp (both seeds)" if all((r["arm"]["R8_surface_f1"] - r["arm"]["K0_surface_f1"]) - (r["control"]["R8_surface_f1"] - r["control"]["K0_surface_f1"]) >= -.02 for r in rows.values()) else "fusion gap widened by > 2 pp: longer training fed K0 more than R8; address fusion next") + " " + json.dumps({s: dict(gap_arm=round(r["arm"]["R8_surface_f1"] - r["arm"]["K0_surface_f1"], 4), gap_control=round(r["control"]["R8_surface_f1"] - r["control"]["K0_surface_f1"], 4), updates=r["arm"]["updates"], trajectory=r["arm"]["trajectory"]) for s, r in rows.items()})
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
nb_updates = (rows["342101"]["arm"]["total_updates"] if adopt else 16000) if isinstance(rows.get("342101"), dict) else None
try: verdict = f"{verdict} ({verdict_label})"
except Exception: pass
json.dump(dict(control=CTRL, control_prefix=CP, next_baseline_updates=nb_updates, baseline_recipe={k: rows["342101"]["control"][k] for k in ("decoder", "decoder_width", "blocks", "precision", "known_weight")} if isinstance(rows.get("342101"), dict) else None,
               rows=rows, verdict=verdict, guards=guards, mechanism=mech, absolute_verdict=absolute, fusion_gap_arm_vs_control=fusion_gap, next_baseline_dir=nb_dir, next_baseline_prefix=nb_prefix, code_bytes=512), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| guards", guards, "| mechanism", mech, "| absolute", absolute, "| fusion gap (arm, control)", fusion_gap, "| next baseline", nb_dir, nb_prefix, "|", json.dumps({s: (r["delta"] if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch 3S-T analysis: $RES"
