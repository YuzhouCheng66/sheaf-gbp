#!/bin/bash
# Day-4 batch N, revision 2 (Claude; rev1 written 2026-10-23 00:40 UTC, rev2 ~01:45 UTC after the independent review
# code_review_v375_datascale.md -- both BEFORE the batch-M final results and before any of these runs).
# rev2: M1 = 200-update CUDA preflight of all three arms under the real co-schedule, time projection (drop D4800 if the projection does
#   not fit), completion gate (updates == 43200 and epochs_completed == --epochs, else INCONCLUSIVE); M2 = trainer asserts the exact
#   TRAIN plan / index sha / regeneration of first+last rows / --expect-updates 43200, and the chain checks the cache's first 8 private
#   hashes against the control's run.json; M3 = near-duplicate-stratified secondary analysis pre-registered in analyze_v375.py.
# Question: is the HouseExpo decoder prior DATA-LIMITED? (v374 gap diagnostic: v367 models score R8 .81/.83 on the 600 TRAIN houses
# they were trained on but .66/.68 on never-trained TRAIN houses -- a ~15 pp gap on both seeds; HouseExpo has 20,366 TRAIN houses.)
# ONE change vs the matched 600-house control: the number of distinct TRAIN houses, at the SAME number of updates (43,200 = 72 x 600).
#  D2400: 2,400 houses per stratum (7,200), 6 epochs, seeds 342101 / 342202.   D4800: 4,800 per stratum (14,400), 3 epochs, seed 342101.
#  Everything else = v367/v372 recipe (lr 3e-4 cosine with 200-update warm-up over total updates, max fusion, posterior_only, sealed
#  wire, clip 5); train_v375.py = train_v374.py + --train-cache (targets/DT rebuilt on the GPU per update; equality tests pass).
# Chart (pre-registered rule): houseexpo_fit (v371) if BOTH batch-M seeds reach final VAL R8 >= v367 same-seed R8 + 2 pp
#  (v367: .639 / .654), else houseexpo (v365, 20 m chart). Control = the same-chart, same-seed 600-house run (v372 or v367).
# Pre-registered expectation: data-limited prior => D2400 final VAL R8 >= control + 5 pp on BOTH seeds; < 2 pp => not data-limited.
#  Dose-response (descriptive, one seed): D4800 vs D2400 same seed. K0 reported alongside; equal-weight 6 strata; fresh panel 8-15.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_N started pid $$ sha256 $(sha256sum scripts/chain_day4_N.sh | cut -c1-16)"
mkdir -p runs/v375-datascale
# waits for: batch M finished, all four caches written, and an independent review naming the exact trainer sha256 that will be launched
until grep -q "batch M done" $LOG && [ "$(grep -c 'exit 0' $C/done.txt 2>/dev/null)" = 4 ] && grep -q "$(sha256sum scripts/train_v375.py | cut -d' ' -f1)" runs/v375-datascale/REVIEW_OK 2>/dev/null; do sleep 60; done
TSHA=$(sha256sum scripts/train_v375.py | cut -d" " -f1)
WORLD=$(python3 - <<'PY'
import json
ref = {"342101": .639, "342202": .654}; ok = True
for s, r in ref.items():
    try:
        d = json.load(open(f"runs/v372-hxfit-train/hxfit-long-s{s}/final.json")); v = d["val"]["8"]["all"]["wall_f1"]
        ok = ok and d.get("epochs_completed") == 72 and v >= r + .02
    except Exception:
        ok = False
print("houseexpo_fit" if ok else "houseexpo")
PY
)
if [ "$WORLD" = houseexpo_fit ]; then CTRL=runs/v372-hxfit-train/hxfit-long; else CTRL=runs/v367-hx-train/hx-long; fi
log "batch N start: chart rule -> world $WORLD (control $CTRL-s<seed>)"
python3 - "$WORLD" "$CTRL" "$C" <<'PY' || { log "cache/control cross-check FAILED: batch N NOT launched"; exit 0; }
import json, sys
world, ctrl, C = sys.argv[1:]
for seed in ("342101", "342202"):
    ref = json.load(open(f"{ctrl}-s{seed}/run.json"))["train_private_sha256"][:8]
    for k in ("2400", "4800"):
        got = [r["private_sha256"] for r in json.load(open(f"{C}/{world}-k{k}/manifest.json"))["rows"][:8]]
        assert got == ref, (seed, k)
print("cache first-8 private hashes == control run.json")
PY
log "cache/control cross-check passed"
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
ARMS="d2400:342101:2400:6 d2400:342202:2400:6 d4800:342101:4800:3"
# ---- M1 preflight: 200 updates of each arm under the real 3-way co-schedule (exercises pinned H2D, CUDA int8 amax, cache asserts)
for a in $ARMS; do IFS=: read arm seed k ep <<< "$a"
  nohup $P scripts/train_v375.py --source $S --world $WORLD --train-cache $C/$WORLD-k$k --out runs/v375-datascale/preflight-$arm-s$seed --arm owned --size 128 --device cuda --epochs $ep --val-every 1000 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 43200 --max-updates 200 > runs/v375-datascale/preflight-$arm-s$seed.log 2>&1 &
done
sleep 30; while pgrep -f "runs/v375-datascale/preflight" >/dev/null; do sleep 15; done
PROJ=$(python3 - <<'PY'
import json, glob, math
sp = []
for f in glob.glob("runs/v375-datascale/preflight-*/history.json"):
    h = json.load(open(f))[-1]
    assert h["updates"] == 200 and h["train_loss"] is not None and math.isfinite(h["train_loss"]), f
    sp.append(h["epoch_seconds"] / 200)
assert len(sp) == 3
print(int(max(sp) * 43200 + 6 * 60 + 120))
PY
) || { log "preflight FAILED (see runs/v375-datascale/preflight-*.log): batch N NOT launched"; exit 0; }
LEFT=$(( DE - $(date +%s) ))
log "preflight ok: projected train seconds $PROJ (3-way co-schedule), seconds left to train deadline $LEFT"
if [ "$PROJ" -gt "$LEFT" ]; then ARMS="d2400:342101:2400:6 d2400:342202:2400:6"; log "projection does not fit: D4800 dropped (pre-registered fallback), 2 arms"; fi
cat > runs/v375-datascale/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "world": "$WORLD",
 "arms": "$ARMS", "updates_each": 43200, "preflight_projected_train_seconds": $PROJ,
 "control": "$CTRL-s<seed> (600 houses x 72 ep)", "trainer_sha256": "$TSHA", "review": "$(cat runs/v375-datascale/REVIEW_OK)",
 "caches": {"k2400": "$(sha256sum $C/$WORLD-k2400/manifest.json | cut -d' ' -f1)", "k4800": "$(sha256sum $C/$WORLD-k4800/manifest.json | cut -d' ' -f1)"},
 "expectation": "data-limited prior => D2400 final VAL R8 >= control + 5pp on both seeds; < 2pp => not data-limited; incomplete runs INCONCLUSIVE; near-copy-free secondary in analyze_v375.py"}
JSON
for a in $ARMS; do IFS=: read arm seed k ep <<< "$a"
  nohup $P scripts/train_v375.py --source $S --world $WORLD --train-cache $C/$WORLD-k$k --out runs/v375-datascale/$arm-s$seed --arm owned --size 128 --device cuda --epochs $ep --val-every 1 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 43200 > runs/v375-datascale/$arm-s$seed.log 2>&1 &
  log "datascale $arm seed $seed pid $!"
done
sleep 60
while pgrep -f "runs/v375-datascale/d" >/dev/null; do sleep 60; done
log "batch N done"
$P scripts/analyze_v375.py --world $WORLD --runs runs --neardup scripts/neardup2_v375.json --out runs/v375-datascale/analysis.json > runs/v375-datascale/analysis.txt 2>&1
log "batch N analysis: $(grep PRIMARY runs/v375-datascale/analysis.txt)"
