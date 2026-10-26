#!/bin/bash
# Day-6 batch 3S-X, the STACK (Claude; written 2026-10-26 04:14 UTC while 3S-S ran, revised 04:32 UTC after code_review_v430_stack.md M1-M3
# and S1-S7, before any 3S-X training). Pre-registered in
# OVERNIGHT_REPORT D8-25. Purpose: the user's F1 goal (R8 >= .90, K0 >= .95, var >= .75). This is a COMBINATION batch, disclosed as such:
# it puts the levers confirmed one at a time (D8-16 width + layout -> D2; D8-23 two cycles; D8-24 the code-layout ladder: 32x32x1 @ 8+8 is
# a lever at 16k, +4.5 / +3.7 pp R8 surface, furniture .30 -> .42; and 3S-S's data verdict, read at run time) together on one recipe.
# Arm = the recipe named by 3S-D's analysis (next_baseline: 32x32x1 @ 8+8, d3 W2, learned precision, uniform known-BCE; 2 KB per message,
# approved for diagnostics) [+ the full-TRAIN cache, lazily loaded, if 3S-S adopted it], trained with EXACTLY 3S-T's two-cycle protocol:
# cycle 1 = the control's cycle-1 update count at lr 3e-4, cycle 2 = warm start from cycle 1's final at peak lr 1e-4 for the same count,
# each under its own lease; the pre-registered early stop after cycle 1 (both seeds < +1 pp R8 surface AND < +2 pp var vs the CONTROL's
# cycle 1 => no cycle 2). Control = 3S-T's two-cycle D2 (runs/v427-length/c2-s*, 2 x 25,500; 16x16x1 @ 8+8 on the 4,708-house cache).
# So the comparison holds the training amount, the cycle protocol, the decoder family / width, the precision and the known-BCE fixed and
# changes the code layout (and, if 3S-S said so, the data); trunk / head initial tensors are asserted equal to the control's cycle 1.
# Disclosed (review S7): the decoder's initial tensors all differ from the control's (its first conv is wider, which shifts the seeded
# draw), and with the full cache the data order differs too; these come with the two variables and are not separable here.
# Pre-registered reading (final VAL, 24 houses, paired by seed, Delta = arm cycle-2 final - control cycle-2 final):
#   Delta R8 variable-structure F1 >= +3 pp AND Delta R8 surface F1 >= +2 pp on BOTH seeds => the stack is a lever (new baseline);
#   Delta R8 surface F1 <= -1 pp on BOTH seeds OR Delta R8 variable-structure F1 <= -2 pp on BOTH seeds => worse (the layout lever does
#   not survive long training -- over-fitting of the larger code at ~11 passes);
#   Delta R8 surface F1 < +1 pp AND Delta R8 variable-structure F1 < +2 pp on BOTH seeds => not better (the levers are not additive);
#   otherwise => ambiguous.
#   Guards: R8 FPR <= control + 0.3 pp; R8 slab recall >= control - 2 pp; K0 variable-structure F1 >= control - 2 pp; a lever whose guard
#   fails is not adopted. Mechanism readings: furniture-region F1 and furniture recall (the layout lever's signature), interior-wall
#   recall, the fusion gap (must not widen by > 2 pp on either seed); the user's final targets are evaluated per seed and reported.
#   The absolute 3S-A criteria are re-applied for the record. Bytes per message recorded (2,048 at 32x32x1 @ 8+8).
# Budget: SOLO, after chain_day6_3SS; one lease per run (up to 4, ~1.5 h each at ~0.2 s/update); hard timeouts inside each lease.
#   Writes only under /mnt/datasets (runs/v429-stack).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v429-stack
CTRL1=runs/v427-length                                                        # 3S-T's two-cycle D2 (control)
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SX started pid $$ sha256 $(sha256sum scripts/chain_day6_3SX.sh | cut -c1-16)"
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
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day6_3SX already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/a-s*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-X history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SS[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SD[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3ST[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SQ[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SR[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SP[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SL[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
# ---- baseline from the previous batch's pre-registered analysis: label, control directory, run prefix
read BASE RDIR RP <<<"$($P - <<'PY'
import json
try:
    a = json.load(open("runs/v429-ladder/analysis.json")); print(a["next_baseline_arm"] or "ctrl", a["next_baseline_dir"], a["next_baseline_prefix"])
except Exception as e:
    print("missing", "", "")
PY
)"
[ -n "$RDIR" ] && [ -d "$RDIR" ] && [ -n "$RP" ] && [ -f $RDIR/${RP}342101/run.json ] || { log "ABORT: recipe source '$BASE' -> '$RDIR' prefix '$RP' not usable"; exit 2; }
read ACACHE LAZY <<<"$($P - <<'PY'
import json
try:
    a = json.load(open("runs/v430-data/analysis.json")); c = a.get("next_baseline_cache")
    print(c if c else "/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600", "--cache-lazy" if (c and a.get("next_baseline_cache_lazy")) else "")   # review S3: INCONCLUSIVE 3S-S -> k1600
except Exception as e:
    print("/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600", "")
PY
)"
[ -f $ACACHE/manifest.json ] || { log "ABORT: arm cache $ACACHE has no manifest"; exit 2; }
AMSHA=$(sha256sum $ACACHE/manifest.json | cut -d' ' -f1)
read DD CB <<<"$($P -c "import json; m=json.load(open('$RDIR/${RP}342101/run.json')); print(m['D'], m.get('raw_code_bytes', 512))")"
CTRL=$CTRL1; CP=c2-s
for s in 342101 342202; do [ -f $CTRL/c2-s$s/final.json ] && [ -f $CTRL/c1-s$s/checkpoints/initial.pt ] && [ -f $CTRL/c1-s$s/final.json ] || { log "ABORT: control $CTRL/c{1,2}-s$s (final / initial) missing"; exit 2; }; done
read DEC WID LAY PREC KW WB <<<"$($P -c "import json; m=json.load(open('$RDIR/${RP}342101/run.json')); print(m.get('decoder','d3'), m.get('decoder_width',1.0), m.get('layout', 'x'.join(str(v) for v in m['blocks'])), m.get('precision','learned'), m.get('known_weight','uniform'), ','.join(str(b) for b in m.get('wire_bits', [8, 8])))")"
read CDEC CWID CLAY CPREC CKW CUPD CLR <<<"$($P -c "import json; m=json.load(open('$CTRL/c2-s342101/run.json')); c1=json.load(open('$CTRL/c1-s342101/run.json')); assert m.get('warm_start') and m['lr']==1e-4 and m['updates']==c1['updates'], 'control is not a two-cycle run'; print(m.get('decoder','d3'), m.get('decoder_width',1.0), m.get('layout', 'x'.join(str(v) for v in m['blocks'])), m.get('precision','learned'), m.get('known_weight','uniform'), c1['updates'], c1['lr'])")"
[ "$DEC" = "$CDEC" ] && [ "$WID" = "$CWID" ] && [ "$PREC" = "$CPREC" ] && [ "$KW" = "$CKW" ] || { log "ABORT: the recipe ($DEC W $WID $PREC $KW) and the control ($CDEC W $CWID $CPREC $CKW) differ in more than the layout / data"; exit 2; }
[ "$LAY" != "$CLAY" ] || [ -n "$LAZY" ] || { log "ABORT: nothing stacked (same layout as the control, same cache)"; exit 2; }
log "batch 3S-X gate: recipe from $RDIR/$RP* (decoder $DEC W $WID layout $LAY wire $WB precision $PREC known-weight $KW), cache $ACACHE $LAZY; control $CTRL/c2-s* (two cycles of $CUPD, lr $CLR then 1e-4, layout $CLAY)"
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
COMMON="--source $S --cache $ACACHE $LAZY --dims 128,128,32 --device cuda --val-per-stratum 8 --gen-workers 8 --layout $LAY --allow-non256 --wire-bits $WB --decoder $DEC --decoder-width $WID --precision $PREC --known-weight $KW"
run_seed(){                                                                     # $1 = cycle (1 | 2), $2 = seed
  CYC=$1; seed=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-X: GPU busy before c$CYC seed $seed; waiting"; sleep 120; done
  if [ "$CYC" = "2" ]; then INIT="--init-from $OUT/c1-s$seed/checkpoints/final.pt --lr 1e-4"; else INIT="--lr $CLR"; fi            # review S4: cycle 1 at the control's lr
  TS2=$(date -u +%H%M%S)
  timeout -k 60 1800 $P scripts/train_3d_v430.py $COMMON $INIT --out $OUT/pf-$TS2-c$CYC-s$seed --seed $seed --updates 16000 --max-updates 1100 --val-every 1100 --deadline-total 0 > $OUT/pf-$TS2-c$CYC-s$seed.log 2>&1
  PFX=$?
  read RS LS VS <<<"$(rate $OUT/pf-$TS2-c$CYC-s$seed.log)"
  [ "$RS" = "nan" ] && { log "ABORT INCONCLUSIVE (preflight failed, exit $PFX): c$CYC seed $seed"; return 1; }
  UPD=$CUPD; UPDFROM="control"; BUD=6300                                      # 3S-X: the control's cycle count (same training amount); the preflight only confirms the fit (review M3: 6,300 s, the deadline is 6,600 s)
  FIT=$($P -c "import math; r=float('$RS'); print(1 if math.isfinite(r) and r>0 and 1.05*(float('$LS')+$UPD*r+3*float('$VS'))<=$BUD else 0)")
  log "batch 3S-X preflight c$CYC seed $seed (exit $PFX): s/update $RS load $LS val $VS -> updates $UPD (from $UPDFROM; cap 96,000; floor 24,000); fits: $FIT"
  [ "$UPD" -ge 24000 ] && [ "$FIT" = "1" ] || { log "ABORT INCONCLUSIVE (budget): c$CYC seed $seed: $UPD updates (floor 24,000) fit=$FIT"; return 1; }
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_c$CYC-s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v420 furnished HouseExpo storeys", "dims": [128, 128, 32], "D": $DD, "code_bytes": $CB, "updates": $UPD, "seed": $seed, "cycle": $CYC,
 "batch": "3S-X", "variable": "stack: layout $LAY (control $CLAY) + cache $ACACHE $LAZY, two cycles of $UPD as the control", "wire_bits": "$WB", "warm_start": "$INIT", "updates_from": "$UPDFROM", "baseline": "$BASE", "baseline_recipe": {"decoder": "$DEC", "decoder_width": $WID, "layout": "$LAY", "precision": "$PREC", "known_weight": "$KW"}, "preflight": "rate=$RS load=$LS val=$VS",
 "trainer": "scripts/train_3d_v430.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "diag_sha256": "$GSHA", "diag_interior_sha256": "$ISHA", "diag_veto_sha256": "$VSHA", "cache": "$ACACHE", "cache_manifest_sha256": "$AMSHA", "cache_lazy": "$LAZY", "control": "$CTRL/$CP$seed",
 "expectation": "stack vs two-cycle D2 (cycle-2 final, or cycle 1 alone on early stop): Delta R8 var >= +3pp and Delta R8 surface >= +2pp on both seeds => lever; Delta R8 surface <= -1pp on both or Delta R8 var <= -2pp on both => worse; Delta R8 surface < +1pp and Delta R8 var < +2pp on both => not better; else ambiguous; early stop after cycle 1 if Delta R8 surface < +1pp on both seeds; guards FPR <= control + 0.3pp, slab recall >= control - 2pp, K0 var >= control - 2pp; mechanism: fusion gap must not widen by > 2pp on either seed; incomplete => INCONCLUSIVE"}
JSON
  log "batch 3S-X start c$CYC seed $seed ($UPD updates, $DEC W $WID $LAY $PREC $KW${INIT:+, warm start lr 1e-4})"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/train_3d_v430.py $COMMON $INIT --out $OUT/c$CYC-s$seed --seed $seed --updates $UPD --val-every $((UPD/4)) --deadline-epoch $DE --deadline-total $DT > $OUT/c$CYC-s$seed.log 2>&1
  log "3S-X c$CYC seed $seed done (exit $?)"
}
for seed in 342101 342202; do run_seed 1 $seed; done
EARLY=$($P - "$CTRL" "$CP" <<'PY'
import json, sys
CTRL, CP = sys.argv[1], sys.argv[2]; ds = []
for s in ("342101", "342202"):
    try:
        a = json.load(open(f"runs/v429-stack/c1-s{s}/final.json")); m = json.load(open(f"runs/v429-stack/c1-s{s}/run.json")); c = json.load(open(f"{CTRL}/c1-s{s}/final.json"))
        if a["updates"] != m["updates"]: raise ValueError("truncated")                      # review S1: a deadline-cut cycle 1 is incomplete
        ds.append((a["val"]["8"]["surface_f1"] - c["val"]["8"]["surface_f1"], a["val"]["8"]["var_occ_f1"] - c["val"]["8"]["var_occ_f1"]))
    except Exception as e:
        ds.append(None)
if any(d is None for d in ds): print("incomplete", ds)
elif all(d[0] < .01 and d[1] < .02 for d in ds): print("stop", [tuple(round(x, 4) for x in d) for d in ds])   # review S6: the 'not better' rule
else: print("continue", [tuple(round(x, 4) for x in d) for d in ds])
PY
)
log "batch 3S-X cycle-1 read vs the control's cycle 1 (pre-registered early stop if both seeds < +1 pp R8 surface and < +2 pp R8 var): $EARLY"
case "$EARLY" in continue*) for seed in 342101 342202; do run_seed 2 $seed || break; done;; *) log "batch 3S-X: cycle 2 not run ($EARLY)";; esac
log "batch 3S-X training done"
check_shas
for s in 342101 342202; do                                                      # report-only breakdowns for the FINAL run of each seed and the control
  if [ -f $OUT/c2-s$s/checkpoints/final.pt ]; then ln -sfn c2-s$s $OUT/a-s$s; ln -sf c2-s$s.log $OUT/a-s$s.log; elif [ -f $OUT/c1-s$s/checkpoints/final.pt ]; then ln -sfn c1-s$s $OUT/a-s$s; ln -sf c1-s$s.log $OUT/a-s$s.log; fi
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
OUT, CTRL, CP = "runs/v429-stack", sys.argv[1], sys.argv[2]
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
             wire_bits=m.get("wire_bits", [8, 8]), raw_code_bytes=m.get("raw_code_bytes", 512), cache_dir=m["cache"]["dir"], train_scenes=m["train_scenes"],
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
    ks = [k for k in c if not k.startswith("decoder.")]                                              # the layout changes the decoder's first conv: trunk / head must match
    return all(k in a and a[k].shape == c[k].shape and torch.equal(a[k], c[k]) for k in ks) and len(ks) > 10
rows = {}
for s in ("342101", "342202"):
    try:
        a = extras(load(f"{OUT}/a-s{s}"), f"{OUT}/diag-a-s{s}", f"{OUT}/diag-interior-a-s{s}.json", f"{OUT}/veto-a-s{s}.json")
        c = extras(load(f"{CTRL}/c2-s{s}"), f"{OUT}/diag-control-s{s}", f"{OUT}/diag-interior-control-s{s}.json", f"{OUT}/veto-control-s{s}.json")
        cc1 = load(f"{CTRL}/c1-s{s}"); assert c["warm_start"] and c["lr"] == 1e-4 and c["updates"] == cc1["updates"], "control is not a two-cycle run"
        assert a["precision"] == c["precision"] and a["known_weight"] == c["known_weight"] and a["decoder"] == c["decoder"] and a["decoder_width"] == c["decoder_width"] and a["wire_bits"] == c["wire_bits"], "recipe"
        c1 = load(f"{OUT}/c1-s{s}"); assert c1["lr"] == cc1["lr"] and c1["warm_start"] is None and c1["updates"] == cc1["updates"], "cycle 1 recipe / count"
        cycles = 1
        if a["warm_start"]:                                                       # cycle 2: warm-started from cycle 1's final, lr 1e-4
            assert a["warm_start"]["source_weights_sha256"] == c1["final_sha"] and a["initial_sha"] == c1["final_sha"] and a["lr"] == 1e-4 and a["updates"] == c1["updates"], "cycle 2 warm start"
            cycles = 2
        a["cycles"] = cycles; a["total_updates"] = cycles * c1["updates"]; a["cycle1"] = {k: c1[k] for k in ("R8_surface_f1", "R8_var_occ_f1", "K0_surface_f1", "K0_var_occ_f1", "trajectory")}
        a["control_cycle1"] = {k: cc1[k] for k in ("R8_surface_f1", "R8_var_occ_f1", "K0_surface_f1", "K0_var_occ_f1")}
        assert a["blocks"] != c["blocks"] or a["cache_dir"] != c["cache_dir"], "nothing stacked"
        assert a["specs"] == c["specs"], "VAL panel"
        assert a["params"]["trunk"] == c["params"]["trunk"] and a["params"]["head"] == c["params"]["head"] and identity(f"{OUT}/c1-s{s}", f"{CTRL}/c1-s{s}"), "identity (trunk / head params and cycle-1 initial tensors == control's cycle 1)"
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
    if rows["342101"]["arm"]["cycles"] == 1:                                           # review M2: no cycle 2 -> never compare cycle 1 with the control's cycle 2
        es = all(r["arm"]["cycle1"]["R8_surface_f1"] - r["arm"]["control_cycle1"]["R8_surface_f1"] < .01 and r["arm"]["cycle1"]["R8_var_occ_f1"] - r["arm"]["control_cycle1"]["R8_var_occ_f1"] < .02 for r in rows.values())
        verdict = "not better (early stop after cycle 1 vs the control's cycle 1)" if es else "INCONCLUSIVE (cycle 2 missing)"
    guards = {}
    for name, fn in (("fpr_ok", lambda r: r["arm"]["R8_unknown_known_fpr"] <= r["control"]["R8_unknown_known_fpr"] + .003), ("slab_ok", lambda r: r["arm"]["R8_slab_recall"] >= r["control"]["R8_slab_recall"] - .02), ("k0var_ok", lambda r: r["delta"]["K0_var_occ_f1"] >= -.02)):
        try: guards[name] = all(fn(r) for r in rows.values())
        except Exception as e: guards[name] = f"n/a ({e})"
    try:
        mech = ("fusion gap did not widen by > 2 pp (both seeds)" if all((r["arm"]["R8_surface_f1"] - r["arm"]["K0_surface_f1"]) - (r["control"]["R8_surface_f1"] - r["control"]["K0_surface_f1"]) >= -.02 for r in rows.values()) else "fusion gap widened by > 2 pp") + " " + json.dumps({s: dict(furn_R8=(r["arm"]["R8_furn_f1"], r["control"]["R8_furn_f1"]), obj_recall_R8=(r["arm"].get("R8_obj_recall"), r["control"].get("R8_obj_recall")), interior_K0=(r["arm"].get("K0_interior"), r["control"].get("K0_interior")), gap_arm=round(r["arm"]["R8_surface_f1"] - r["arm"]["K0_surface_f1"], 4), gap_control=round(r["control"]["R8_surface_f1"] - r["control"]["K0_surface_f1"], 4), final_targets_met=bool(r["arm"]["R8_surface_f1"] >= .90 and r["arm"]["K0_surface_f1"] >= .95 and r["arm"]["R8_var_occ_f1"] >= .75), bytes=r["arm"]["raw_code_bytes"], cache=r["arm"]["cache_dir"].split("/")[-1]) for s, r in rows.items()})
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
adopt = verdict == "lever" and isinstance(guards, dict) and all(guards.get(g) is True for g in ("fpr_ok", "slab_ok", "k0var_ok"))   # review S5: all three guards
if verdict == "lever" and not adopt: verdict = "lever, but a guard failed (FPR, slab or K0 var): baseline unchanged"
nb_dir, nb_prefix = (OUT, "a-s") if adopt else (CTRL, CP)
nb_updates = (rows["342101"]["arm"]["total_updates"] if adopt else 2 * rows["342101"]["control"]["updates"]) if isinstance(rows.get("342101"), dict) else None     # review S1: the control is a two-cycle run
nb_cycles = (dict(cycles=rows["342101"]["arm"]["cycles"], updates_per_cycle=rows["342101"]["arm"]["updates"], final_runs={s: ("c2-s" if rows[s]["arm"]["cycles"] == 2 else "c1-s") + s for s in rows}) if adopt else dict(cycles=2, updates_per_cycle=rows["342101"]["control"]["updates"], final_runs={s: "c2-s" + s for s in rows})) if isinstance(rows.get("342101"), dict) else None
try: verdict = f"{verdict} ({verdict_label})"
except Exception: pass
nb_recipe = ({k: rows["342101"]["arm"][k] for k in ("decoder", "decoder_width", "blocks", "wire_bits", "precision", "known_weight", "cache_dir")} if adopt else {k: rows["342101"]["control"][k] for k in ("decoder", "decoder_width", "blocks", "wire_bits", "precision", "known_weight", "cache_dir")}) if isinstance(rows.get("342101"), dict) else None
json.dump(dict(control=CTRL, control_prefix=CP, next_baseline_updates=nb_updates, next_baseline_cycles=nb_cycles, next_baseline_recipe=nb_recipe, baseline_recipe={k: rows["342101"]["control"][k] for k in ("decoder", "decoder_width", "blocks", "precision", "known_weight")} if isinstance(rows.get("342101"), dict) else None,
               rows=rows, verdict=verdict, guards=guards, mechanism=mech, absolute_verdict=absolute, fusion_gap_arm_vs_control=fusion_gap, next_baseline_dir=nb_dir, next_baseline_prefix=nb_prefix, code_bytes=(rows["342101"]["arm"]["raw_code_bytes"] if adopt else rows["342101"]["control"]["raw_code_bytes"]) if isinstance(rows.get("342101"), dict) else None), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| guards", guards, "| mechanism", mech, "| absolute", absolute, "| fusion gap (arm, control)", fusion_gap, "| next baseline", nb_dir, nb_prefix, "|", json.dumps({s: (r["delta"] if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch 3S-X analysis: $RES"
