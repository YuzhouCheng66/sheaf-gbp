#!/bin/bash
# Day-6 batch RS-A (Claude; first written 2026-10-24 ~22:30 UTC, revised 2026-10-25 ~00:10 UTC after code_review_v410_realscan2d.md
# M1-M4, before any real-scan training run). Pre-registered in D7-0 / D7-1.
# Question (hold the system fixed, change only the training data): the HouseExpo working baseline (wide decoder + public-frame
# conditioning, runs/v393-frame-cycles/c3-s<seed>, VAL R8 .915 / .913 on HouseExpo) meets REAL scans: floor slices of Replica meshes under
# the v371 fixture contract (mapping_realscan2d_data_v410: same chart rule, sensor, poses, radio, N in {8,16}; only the geometry differs;
# review M1: meshes gravity-aligned, a floor is an upward-facing surface; review M2: every chart is aligned with its floor's main wall
# direction, TRAIN adds quarter turns, so TRAIN / VAL / HouseExpo share one orientation distribution).
# Arms per seed, all from the same starting weights c3-s<seed>:
#   ZS     the baseline evaluated as is on Replica VAL;
#   FT_RS  14,400 updates on the Replica TRAIN cache (4,800 fixtures x 3 epochs), peak lr 1e-4, 200-step warm-up, cosine (the W / X peak
#          lr and warm-up; a quarter of a W / X cycle's 57,600 updates);
#   FT_HX  the SAME 14,400 updates on HouseExpo data (hx-cache-v375/houseexpo_fit-k4800, 14,400 houses x 1 epoch) -- review M3: one
#          extra HouseExpo cycle (c2 -> c3) alone raised Replica zero-shot R8 by +1.4 / +1.6 pp, so "more training" must be controlled.
# Pre-registered reading (final VAL, 4 scenes x N{8,16} x 8 placements, R8 wall F1 equal weight per (scene, N), BOTH seeds):
#   G_data = FT_RS - FT_HX >= +3 pp on both seeds   => real-scan data is the lever (HM3D / Gibson training is the next priority);
#   |G_data| < 1 pp on both seeds                   => HouseExpo training already transfers at this scale;
#   otherwise                                       => partial.
#   FPR protection is part of the verdict: if FT_RS raises the unknown-known FPR by > 0.3 pp over FT_HX on either seed, the verdict
#   gets "BUT FPR protection violated" and cannot be "lever".
#   Reported, not judged: G = FT_RS - ZS and FT_HX - ZS (the more-training effect), every number per scene (rooms1 = apartment_2,
#   rooms2 = hotel_0, rooms3 = office_3, rooms4 = office_4), K0, the fresh panel (placements 8-15 of the same VAL scenes), FT_RS on
#   HouseExpo VAL (forgetting) vs .915 / .913, and FT_RS / FT_HX VAL R8 per epoch (history, never used to select).
# Disclosures (review M4): (i) a CPU smoke run of ZS on 4 fixtures (VAL placements 0-1 of apartment_2, N8: R8 .495, K0 .568, per fixture
# .520 / .470; fresh placements 8-9: .496 / .580), and the reviewer's full-panel ZS before M1 / M2 (.402 / .410, office_3 .18 because its
# walls ran at 51.5 deg to the chart), a rotation experiment (an aligned VAL scene turned 40 deg loses 16-31 pp) and the c2 -> c3 effect
# above were all seen before this revision; the thresholds are those written before the smoke run and are not changed.
# Caveats recorded in advance: about 10 distinct TRAIN layouts (15 floors; the six FRL apartments are one flat restaged; office_1 has
# 4.2 m^2 of free space), so FT_RS measures how fast a little real data helps, not its ceiling; VALIDATION is new rooms more than new
# buildings; the fresh panel re-uses the VAL scenes; Replica has no TEST split; the 128^2 conservative raster erodes 3-11 % of the free
# area (review S3); any surface within 0.15-1.60 m blocks the 2D beam (review S8).
# Scheduling: solo, seeds strictly sequential, each seed (all five runs) under ONE inclusive 7,200 s lease; every run under a hard
# `timeout` at the lease's total deadline and skipped as INCONCLUSIVE when < 120 s remain (review S1). Writes only under /mnt/datasets.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export REALSCAN_BASE_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-base
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
C=/mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-train-k4800
CHX=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375/houseexpo_fit-k4800
LOG=runs/chain_day3.log
OUT=runs/v410-realscan
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day6_RS started pid $$ sha256 $(sha256sum scripts/chain_day6_RS.sh | cut -c1-16)"
TSHA=$(sha256sum scripts/train_v410.py | cut -d' ' -f1); DSHA=$(sha256sum scripts/mapping_realscan2d_data_v410.py | cut -d' ' -f1)
[ "$TSHA" = "27878023c90cc0d52da957d9f517a3eea5b576c933129a39e85cdf20434ed0a4" ] || { log "ABORT: train_v410.py sha changed ($TSHA)"; exit 2; }               # review S2: pinned
[ "$DSHA" = "f1a37cb8b197c2da0b0ae5d904007763d2a87ce58be888ee84360d46a792126c" ] || { log "ABORT: mapping_realscan2d_data_v410.py sha changed ($DSHA)"; exit 2; }
ISHA=$(sha256sum $REALSCAN_BASE_DIR/index.json | cut -d' ' -f1); MSHA=$(sha256sum $C/manifest.json | cut -d' ' -f1)
$P -c "import json,sys; m=json.load(open('$C/manifest.json')); sys.exit(0 if m['scenes']==4800 and m['world']=='realscan2d' and m['size']==128 and m['data_module_sha256']=='$DSHA' else 1)" || { log "ABORT: real-scan cache does not match the data module / count"; exit 2; }
$P -c "import json,sys; m=json.load(open('$CHX/manifest.json')); sys.exit(0 if m['scenes']==14400 and m['world']=='houseexpo_fit' else 1)" || { log "ABORT: HouseExpo cache unexpected"; exit 2; }
for s in 342101 342202; do
  [ -f runs/v393-frame-cycles/c3-s$s/checkpoints/final.pt ] || { log "ABORT: baseline c3-s$s missing"; exit 2; }
  $P -c "import json,sys; f=json.load(open('runs/v393-frame-cycles/c3-s$s/final.json')); sys.exit(0 if f.get('updates')==57600 else 1)" || { log "ABORT: baseline c3-s$s incomplete"; exit 2; }
done
ls -d $OUT/zs-s* $OUT/ftrs-s* $OUT/fthx-s* $OUT/analysis.json >/dev/null 2>&1 && { log "ABORT: earlier RS-A outputs exist in $OUT (never overwrite history)"; exit 2; }
while pgrep -f "^$P scripts/train_" >/dev/null; do sleep 120; done
log "batch RS-A: GPU free; starting (solo, one lease per seed)"
mkdir -p $OUT
MODEL="--source $S --arm owned --size 128 --device cuda --readout posterior_only --fusion max --decoder-width 2.0 --frame-cond --refine-steps 0"
run(){  # run NAME ARGS... : under a hard timeout at the lease's total deadline; skipped (INCONCLUSIVE) with < 120 s left
  local name=$1; shift; local R=$((DT - $(date +%s)))
  if [ $R -le 120 ]; then log "batch RS-A: $name skipped ($R s left in the lease) -> INCONCLUSIVE"; return 1; fi
  log "batch RS-A: $name ($R s left)"
  timeout -k 60 $R $P scripts/train_v410.py $MODEL "$@" --deadline-total $DT --out $OUT/$name > $OUT/$name.log 2>&1
}
for seed in 342101 342202; do
  INIT=runs/v393-frame-cycles/c3-s$seed/checkpoints/final.pt
  START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
  cat > $OUT/lease_s$seed.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "world": "v410 real-scan floor slices (Replica, gravity- and wall-aligned)", "seed": $seed,
 "runs": ["zs-s$seed (eval)", "ftrs-s$seed (Replica, 4800 x 3 = 14400 updates)", "ftrs-s$seed-hx (eval on HouseExpo VAL)", "fthx-s$seed (HouseExpo, 14400 x 1 = 14400 updates)", "fthx-s$seed-rs (eval on Replica VAL)"], "init": "$INIT",
 "trainer": "scripts/train_v410.py", "trainer_sha256": "$TSHA", "data_module_sha256": "$DSHA", "base_index_sha256": "$ISHA", "cache": "$C", "cache_manifest_sha256": "$MSHA", "houseexpo_cache": "$CHX",
 "expectation": "G_data = FT_RS - FT_HX R8 wall F1 (Replica VAL, equal weight per scene x N): >= +3pp on both seeds => real-scan data is the lever; |G_data| < 1pp on both => HouseExpo training transfers; else partial; FT_RS FPR > FT_HX FPR + 0.3pp on either seed => BUT FPR protection violated (no lever); incomplete => INCONCLUSIVE"}
JSON
  run zs-s$seed --world realscan2d --epochs 0 --seed $seed --init-from $INIT
  run ftrs-s$seed --world realscan2d --train-cache $C --epochs 3 --lr 1e-4 --val-every 1 --expect-updates 14400 --seed $seed --init-from $INIT --deadline-epoch $DE
  [ -f $OUT/ftrs-s$seed/checkpoints/final.pt ] && run ftrs-s$seed-hx --world houseexpo_fit --epochs 0 --seed $seed --init-from $OUT/ftrs-s$seed/checkpoints/final.pt
  run fthx-s$seed --world houseexpo_fit --train-cache $CHX --epochs 1 --lr 1e-4 --val-every 1 --expect-updates 14400 --seed $seed --init-from $INIT --deadline-epoch $DE
  [ -f $OUT/fthx-s$seed/checkpoints/final.pt ] && run fthx-s$seed-rs --world realscan2d --epochs 0 --seed $seed --init-from $OUT/fthx-s$seed/checkpoints/final.pt
  log "batch RS-A seed $seed done"
done
log "batch RS-A done"
RES=$($P - <<'PY'
import json
OUT = "runs/v410-realscan"
SCENES = {"rooms1": "apartment_2", "rooms2": "hotel_0", "rooms3": "office_3", "rooms4": "office_4"}
def eq(f, panel="val", hop="8", key="wall_f1"):
    st = f[panel][hop]["strata"]; return sum(v[key] for v in st.values()) / len(st)
def per_scene(f, panel="val"):
    st = f[panel]["8"]["strata"]; out = {}
    for k, v in st.items():
        g, n = k.split("/"); out.setdefault(SCENES[g], []).append(v["wall_f1"])
    return {s: round(sum(v) / len(v), 4) for s, v in out.items()}
def hist(name):
    try:
        return [round(h["validation"]["8"]["all"]["wall_f1"], 4) for h in json.load(open(f"{OUT}/{name}/history.json"))]
    except Exception:
        return None
rows = {}
for s in ("342101", "342202"):
    try:
        zs = json.load(open(f"{OUT}/zs-s{s}/final.json")); ftrs = json.load(open(f"{OUT}/ftrs-s{s}/final.json")); fthx_rs = json.load(open(f"{OUT}/fthx-s{s}-rs/final.json"))
        fthx = json.load(open(f"{OUT}/fthx-s{s}/final.json"))
        if ftrs["updates"] != 14400 or fthx["updates"] != 14400:
            rows[s] = f"INCONCLUSIVE (updates FT_RS {ftrs['updates']}, FT_HX {fthx['updates']})"; continue
        base = json.load(open(f"runs/v393-frame-cycles/c3-s{s}/final.json"))
        try:
            hx_r8 = round(eq(json.load(open(f"{OUT}/ftrs-s{s}-hx/final.json"))), 4)
        except Exception:
            hx_r8 = None
        rows[s] = dict(ZS_R8=round(eq(zs), 4), FT_RS_R8=round(eq(ftrs), 4), FT_HX_R8=round(eq(fthx_rs), 4),
                       G_data_pp=round(100 * (eq(ftrs) - eq(fthx_rs)), 2), G_vs_ZS_pp=round(100 * (eq(ftrs) - eq(zs)), 2), more_training_pp=round(100 * (eq(fthx_rs) - eq(zs)), 2),
                       fpr_delta_vs_FT_HX_pp=round(100 * (eq(ftrs, key="unknown_known_fpr") - eq(fthx_rs, key="unknown_known_fpr")), 3),
                       K0=dict(ZS=round(eq(zs, hop="0"), 4), FT_RS=round(eq(ftrs, hop="0"), 4), FT_HX=round(eq(fthx_rs, hop="0"), 4)),
                       fresh_R8=dict(ZS=round(eq(zs, "fresh"), 4), FT_RS=round(eq(ftrs, "fresh"), 4), FT_HX=round(eq(fthx_rs, "fresh"), 4)),
                       per_scene_R8=dict(ZS=per_scene(zs), FT_RS=per_scene(ftrs), FT_HX=per_scene(fthx_rs)),
                       per_scene_G_data_pp={k: round(100 * (per_scene(ftrs)[k] - per_scene(fthx_rs)[k]), 2) for k in per_scene(ftrs)},
                       houseexpo_R8_baseline=round(eq(base), 4), houseexpo_R8_after_FT_RS=hx_r8,
                       history_R8=dict(FT_RS=hist(f"ftrs-s{s}"), FT_HX_houseexpo_val=hist(f"fthx-s{s}")))
    except Exception as e:
        rows[s] = f"INCONCLUSIVE ({e})"
ok = all(isinstance(r, dict) for r in rows.values())
if ok:
    g = [r["G_data_pp"] for r in rows.values()]; fpr_bad = any(r["fpr_delta_vs_FT_HX_pp"] > 0.3 for r in rows.values())
    verdict = "real-scan data is the lever" if min(g) >= 3 else ("HouseExpo training transfers" if max(abs(x) for x in g) < 1 else "partial")
    if fpr_bad:
        verdict = ("partial" if verdict == "real-scan data is the lever" else verdict) + " BUT FPR protection violated"
else:
    verdict = "INCONCLUSIVE"; fpr_bad = None
json.dump(dict(rows=rows, verdict=verdict, fpr_protection_violated=fpr_bad, scenes=SCENES), open(f"{OUT}/analysis.json", "w"), indent=1)
print(verdict, "|", json.dumps({s: (dict(G_data_pp=r["G_data_pp"], per_scene=r["per_scene_G_data_pp"]) if isinstance(r, dict) else r) for s, r in rows.items()}))
PY
)
log "batch RS-A analysis: $RES"
