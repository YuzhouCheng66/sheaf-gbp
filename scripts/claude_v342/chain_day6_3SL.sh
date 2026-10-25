#!/bin/bash
# Day-6 batch 3S-L (Claude; written 2026-10-25 05:40 UTC while 3S-W ran, revised 06:20 UTC after code_review_v424_ceiling.md M1-M4 and
# S1-S7, before any ceiling run). Pre-registered in OVERNIGHT_REPORT D8-7.
# Question (problem localisation, D8-6): what can a code of the deployed layout CARRY at all, with an ideal encoder? The D4-6 protocol in 3D
# (owned_ceiling_3d_v424.py): no encoder; each agent's (mu, q) per public block is a free parameter; the decoder is the deployed Decoder3D;
# fusion is the deployed max-precision join with the deployed 8+8-bit wire. TRAIN: 192 TRAIN houses (indices 0-63 x 3 strata), decoder +
# codes trained jointly on K0 and R8 targets (union loss does not update codes). VAL: the 24-house criterion panel, decoder frozen, each
# agent's code fitted to ITS OWN observation only (300 Adam steps), unions composed by the deployed join and scored against the never-fitted
# targets. A MEASUREMENT (one seed per arm, as batch L), not a two-seed criterion batch. What it bounds (review M3): block-local encoders
# paired with THIS decoder family at this budget (decoder trained on 192 houses); the number is itself a LOWER bound on the true ceiling
# (free codes can under-fit), so a low ceiling never proves a representation limit on its own -- the validity gate below guards that.
# Arms (sequential, each under its own inclusive 7,200 s lease; expected 12-20 min each):
#   C1  8x8x4   W 1   -- the ceiling of 3S-A's code and decoder;
#   C2  8x8x4   W 2   -- the ceiling of 3S-W's (does decoder width bind even with ideal codes?);
#   C3  16x16x1 W 1   -- the ceiling of 3S-C's layout;
#   C4  16x16x4 W 1   -- D = 1,024, DIAGNOSTIC ONLY (violates the 256-dim rule; never deployed): is the dimension binding at all?
# Pre-registered reading (VAL, own-fitted codes, K0; trained references: 3S-A K0 surface .685 / .681, var .436 / .433; 3S-W seed 342101
# K0 surface .733, var .534):
#   C1 K0 surface >= .85 AND K0 var >= .70  => the 256-D 8x8x4 code CAN carry furnished storeys: the trained deficit is encoder / training;
#   C1 K0 surface <  .75 OR  K0 var <  .55  => NOT REACHED by free codes with this decoder (representation or decoder limit); then
#                                              C3 - C1 (layout) and C4 - C3 (dimension, review S1) say which lever lifts it (>= +10 pp var);
#   otherwise                               => partial: representation and encoder / training both bind.
#   C2 - C1 >= +5 pp var                    => the decoder binds even with ideal codes (width is part of the ceiling).
#   VALIDITY (review M2): an arm is INVALID (no reading) unless it completed its steps, scored 24 VAL and >= 180 TRAIN houses, and its
#   K0 var is not more than 2 pp BELOW the trained model of the same (layout, width) where one exists (3S-A .436 / .433 for C1, 3S-W
#   for C2, 3S-C for C3): a ceiling under the trained model is an under-fit artefact, as D4-6's 8x8x1 .694 vs trained .71-.74 showed.
#   Codes use a CONSTANT lr 5e-2 for 16,000 steps (one scene per step: ~83 visits per code; a cosine on the codes capped |mu| at .4).
#   The K0 ceiling reads a continuous precision scalar plus the exact block outline (review S6); the R8 ceiling goes through the wire.
#   Reported, not judged: R8 ceilings (fusion of own-fitted codes; bounds the fusion loss at the representation level), furniture-region
#   F1, tol-1, FPR, TRAIN-fitted numbers, fit loss, decoder parameters.
# Budget: SOLO, after chain_day6_3SC; --steps 16000 --time-cap 3000 (~20 min W1, ~35 min W2); every run under a hard timeout inside its lease. Writes only under
# /mnt/datasets (runs/v424-ceiling).
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
OUT=runs/v424-ceiling
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_3SL started pid $$ sha256 $(sha256sum scripts/chain_day6_3SL.sh | cut -c1-16)"
check_shas(){
  CSHA=$(sha256sum scripts/owned_ceiling_3d_v424.py | cut -d' ' -f1); TSHA=$(sha256sum scripts/train_3d_v424.py | cut -d' ' -f1)
  DSHA=$(sha256sum scripts/mapping_3d_storey_data_v420.py | cut -d' ' -f1); KSHA=$(sha256sum scripts/train_v378.py | cut -d' ' -f1)
  [ "$CSHA" = "e23550d014c47af69768a84dae66a1891ddb88840ef80ebfe06700177b7597c4" ] || { log "ABORT: owned_ceiling_3d_v424.py sha changed ($CSHA)"; exit 2; }
  [ "$TSHA" = "d228707971d9232e215d8e076d32f87de4a99b3ef1b44d0eee5d56e7115f84f8" ] || { log "ABORT: train_3d_v424.py sha changed ($TSHA)"; exit 2; }
  [ "$DSHA" = "4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed ($DSHA)"; exit 2; }
  [ "$KSHA" = "db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff" ] || { log "ABORT: train_v378.py (k-bit wire) sha changed ($KSHA)"; exit 2; }
}
check_shas
mkdir runs/v424-ceiling.lock 2>/dev/null || { log "ABORT: chain_day6_3SL already launched (lock exists)"; exit 2; }
if compgen -G "$OUT/C*" >/dev/null || compgen -G "$OUT/lease_*.json" >/dev/null || [ -e $OUT/analysis.json ]; then log "ABORT: $OUT already has 3S-L history (never overwrite)"; exit 2; fi
busy(){ pgrep -f "bash scripts/chain_day6_3SC[.]sh" >/dev/null || pgrep -f "bash scripts/chain_day6_3SW[.]sh" >/dev/null || pgrep -f "chain_day6_RS[.]sh" >/dev/null || pgrep -f "gen_3d_cache_v42[01][.]py" >/dev/null || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
log "batch 3S-L: GPU free; starting (SOLO, 4 arms sequential, one lease per arm)"
mkdir -p $OUT
for arm in "C1 8x8x4 1.0" "C2 8x8x4 2.0" "C3 16x16x1 1.0" "C4 16x16x4 1.0 --allow-non256"; do
  set -- $arm; NAME=$1; LAYOUT=$2; WIDTH=$3; EXTRA=$4
  check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "3S-L: GPU busy before $NAME; waiting"; sleep 120; done
  START=$(date +%s); DT=$((START+7150))
  cat > $OUT/lease_$NAME.json <<JSON
{"lease_start_epoch": $START, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "arm": "$NAME", "layout": "$LAYOUT", "decoder_width": $WIDTH, "diagnostic_only": $([ "$LAYOUT" = "16x16x4" ] && echo true || echo false),
 "script": "scripts/owned_ceiling_3d_v424.py", "script_sha256": "$CSHA", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "kbit_sha256": "$KSHA", "protocol": "D4-6 in 3D: free own codes, deployed decoder + join + wire; VAL codes fitted to own observation only",
 "expectation": "validity: complete steps, 24 VAL, >= 180 TRAIN, K0 var >= trained model of the same config - 2 pp, else INVALID; C1 K0 surface >= .85 and var >= .70 => code can carry (deficit = encoder / training); C1 K0 surface < .75 or var < .55 => not reached by free codes with this decoder (representation or decoder limit; C3 - C1 layout, C4 - C3 dimension, >= +10 pp var = lever); else partial; C2 - C1 >= +5 pp var => decoder binds even with ideal codes"}
JSON
  log "batch 3S-L start $NAME (layout $LAYOUT, width $WIDTH)"
  timeout -k 30 $((DT - $(date +%s))) $P scripts/owned_ceiling_3d_v424.py --source $S --out $OUT/$NAME --layout $LAYOUT --decoder-width $WIDTH $EXTRA --device cuda --steps 16000 --time-cap 3000 --code-lr 5e-2 --seed 342101 > $OUT/$NAME.log 2>&1
  log "3S-L $NAME done (exit $?)"
done
RES=$($P - <<'PY'
import json
OUT = "runs/v424-ceiling"; arms = {}
for name in ("C1", "C2", "C3", "C4"):
    try:
        r = json.load(open(f"{OUT}/{name}/report.json")); v = r["val_summary"]; t = r["train_summary"]
        arms[name] = dict(layout=r["layout"], D=r["D"], width=r["decoder_width"], steps=r["steps_done"], complete=r["steps_done"] == r["args"]["steps"], decoder_params=r["decoder_parameters"],
                          train_scenes=r["train_scenes"], val_scenes=r["val_scenes"], train_mu_abs_max=round(r["train_mu_abs_max"], 3), fit_curve_mean=[round(sum(x["fit_curve"][i][1] for x in r["val_rows"]) / len(r["val_rows"]), 4) for i in range(len(r["val_rows"][0]["fit_curve"]))],
                          **{f"val_{h}_{m}": round(v[h][m], 4) for h in ("K0", "K8") for m in ("surface_f1", "var_occ_f1", "furn_f1", "surface_f1_tol1", "unknown_known_fpr") if v[h][m] is not None},
                          **{f"train_{h}_{m}": round(t[h][m], 4) for h in ("K0", "K8") for m in ("surface_f1", "var_occ_f1")})
    except Exception as e:
        arms[name] = f"n/a ({e})"
def trained_var(name):                                                            # trained model of the same (layout, width), K0 var, min over seeds
    paths = {"C1": "runs/v420-storey", "C2": "runs/v422-decoder", "C3": "runs/v423-layout"}.get(name)
    if not paths: return None
    vals = []
    for sd in ("342101", "342202"):
        try: vals.append(json.load(open(f"{paths}/a-s{sd}/final.json"))["val"]["0"]["var_occ_f1"])
        except Exception: pass
    return min(vals) if vals else None
validity = {}
for name, a in arms.items():
    if not isinstance(a, dict): validity[name] = "missing"; continue
    tv = trained_var(name); reasons = []
    if not a["complete"]: reasons.append("incomplete steps")
    if a["val_scenes"] != 24: reasons.append(f"val scenes {a['val_scenes']}")
    if a["train_scenes"] < 180: reasons.append(f"train scenes {a['train_scenes']}")
    if tv is not None and a["val_K0_var_occ_f1"] < tv - .02: reasons.append(f"K0 var {a['val_K0_var_occ_f1']} below trained {round(tv, 4)} - 2pp: under-fit artefact")
    validity[name] = "valid" if not reasons else "INVALID: " + "; ".join(reasons)
try:
    if validity["C1"] != "valid": raise ValueError(f"C1 {validity['C1']}")
    c1 = arms["C1"]; s, v = c1["val_K0_surface_f1"], c1["val_K0_var_occ_f1"]
    if s >= .85 and v >= .70: verdict = "8x8x4 256-D code CAN carry furnished storeys: trained deficit = encoder / training"
    elif s < .75 or v < .55: verdict = "not reached by free codes with this decoder (representation or decoder limit)"
    else: verdict = "partial: representation and encoder / training both bind"
    levers = {}
    if validity.get("C3") == "valid":
        d = round(arms["C3"]["val_K0_var_occ_f1"] - v, 4); levers["layout 16x16x1 (C3 - C1)"] = dict(delta_K0_var=d, lever=d >= .10)
        if validity.get("C4") == "valid":
            d4 = round(arms["C4"]["val_K0_var_occ_f1"] - arms["C3"]["val_K0_var_occ_f1"], 4); levers["dimension 1024 (C4 - C3, diagnostic)"] = dict(delta_K0_var=d4, lever=d4 >= .10)
    width = None
    if validity.get("C2") == "valid":
        d2 = round(arms["C2"]["val_K0_var_occ_f1"] - v, 4); width = dict(delta_K0_var=d2, decoder_binds_with_ideal_codes=d2 >= .05)
except Exception as e:
    verdict = f"INCONCLUSIVE ({e})"; levers = width = None
json.dump(dict(arms=arms, validity=validity, verdict=verdict, levers=levers, decoder_width_effect=width), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "| validity", validity, "| levers", levers, "| width", width, "|", json.dumps({k: ({m: a[m] for m in a if m.startswith("val_K0") or m.startswith("val_K8_surface")} if isinstance(a, dict) else a) for k, a in arms.items()}))
PY
)
log "batch 3S-L analysis: $RES"
