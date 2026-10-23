#!/bin/bash
# Day-4 batch O, revision 2 (Claude; rev1 written 2026-10-23 ~02:40 UTC, rev2 ~03:10 UTC after code_review_v378_rect.md; both before any
# v378 training run). rev2: paired poses (v371 poses translated, M1; cache regenerated as houseexpo_rect-k2400, the unpaired cache kept as
# houseexpo_rect-k2400-unpaired-v378a), pre-launch cross-check that every cache row has the same adjacency and forest edges as the
# square-chart cache row (pairing reached the cache), control-identity guard in analyze_v378.py (M2), chart-changed / placebo secondary.
# Question: does fitting the public chart per axis (RECTANGLE, aspect <= 2; v378) add accuracy over the square house-fitted chart (v371)?
# Rationale: capacity per house area is the binding limit (D4-8 .. D4-11); the rectangle raises the house-bbox share of the 64 blocks
# from 0.57 to 0.76 (median). ONE change vs control: the chart (same trainer code path, same TRAIN houses, same seeds, same recipe).
# Recipe (pre-registered, mechanical): the batch-N recipe with the higher mean final VAL R8 over the two seeds among COMPLETE runs:
#   D2400 (7,200 houses, 6 epochs, cache houseexpo_rect-k2400) vs H600 (600 houses, 72 epochs); both 43,200 updates.
#   Control = batch-N d2400-s<seed> (square) or v372 hxfit-long-s<seed> (square) accordingly.
# Pre-registered reading (analyze_v378.py): rect >= square + 2 pp final VAL R8 on BOTH seeds => adopt the rectangle; < 1 pp on both =>
# no gain (keep the simpler square chart); else ambiguous. Incomplete runs INCONCLUSIVE. Anisotropic cells => per-cell F1 is not
# equal-area; reported as per-house accuracy on each arm's native raster.
# Runs after batch J (rate-distortion, v293) so the two batches never share the GPU.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_O started pid $$ sha256 $(sha256sum scripts/chain_day4_O.sh | cut -c1-16)"
mkdir -p runs/v378-rect
until grep -q "batch N analysis" $LOG && grep -q "batch J admissibility" $LOG && grep -q "houseexpo_rect-paired k2400 exit 0" $C/done_v378.txt 2>/dev/null \
      && grep -q "$(sha256sum scripts/train_v378.py | cut -d' ' -f1)" runs/v378-rect/REVIEW_OK 2>/dev/null \
      && grep -q "$(sha256sum scripts/mapping_houseexpo_data_v378.py | cut -d' ' -f1)" runs/v378-rect/REVIEW_OK 2>/dev/null; do sleep 60; done
while pgrep -f "runs/v369-rate/l" >/dev/null; do sleep 60; done          # never share the GPU with batch J
RECIPE=$($P - <<'PY'
import json
try:
    a = json.load(open("runs/v375-datascale/analysis.json"))
    d = [r for r in a["comparisons"] if r["arm"].startswith("d2400") and r["verdict_input"] == "ok"]
    print("d2400" if len(d) == 2 and sum(r["val_R8"]["arm"] - r["val_R8"]["control"] for r in d) > 0 else "h600")
except Exception:
    print("h600")
PY
)
log "batch O start: recipe $RECIPE (rule: higher mean final VAL R8 in batch N among complete runs)"
if [ "$RECIPE" = d2400 ]; then
  $P - "$C" <<'PY' || { log "paired-cache cross-check FAILED: batch O NOT launched"; exit 0; }
import json, sys, numpy as np
C = sys.argv[1]
mr = json.load(open(f"{C}/houseexpo_rect-k2400/manifest.json")); ms = json.load(open(f"{C}/houseexpo_fit-k2400/manifest.json"))
assert [r["spec"] for r in mr["rows"]] == [r["spec"] for r in ms["rows"]]
for a, b in zip(mr["rows"], ms["rows"]):
    za = np.load(f"{C}/houseexpo_rect-k2400/{a['file']}"); zb = np.load(f"{C}/houseexpo_fit-k2400/{b['file']}")
    assert np.array_equal(za["adjacency"], zb["adjacency"]) and np.array_equal(za["edges"], zb["edges"]) and np.array_equal(za["reached"], zb["reached"]), a["file"]
print("paired cache: adjacency / edges / reached identical in", len(mr["rows"]), "rows")
PY
  log "paired-cache cross-check passed (adjacency, forest edges, reached identical to the square-chart cache in all rows)"
fi
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v378-rect/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "world": "houseexpo_rect",
 "recipe": "$RECIPE", "seeds": [342101, 342202], "updates_each": 43200, "trainer_sha256": "$(sha256sum scripts/train_v378.py | cut -d' ' -f1)",
 "data_module_sha256": "$(sha256sum scripts/mapping_houseexpo_data_v378.py | cut -d' ' -f1)", "review": "$(cat runs/v378-rect/REVIEW_OK)",
 "expectation": "rect >= square + 2pp final VAL R8 on both seeds => adopt; < 1pp => keep square; incomplete => INCONCLUSIVE"}
JSON
for seed in 342101 342202; do
  if [ "$RECIPE" = d2400 ]; then
    EXTRA="--train-cache $C/houseexpo_rect-k2400 --epochs 6 --val-every 1 --expect-updates 43200"
  else
    EXTRA="--epochs 72 --val-every 6"
  fi
  nohup $P scripts/train_v378.py --source $S --world houseexpo_rect $EXTRA --out runs/v378-rect/rect-$RECIPE-s$seed --arm owned --size 128 --device cuda --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v378-rect/rect-$RECIPE-s$seed.log 2>&1 &
  log "rect $RECIPE seed $seed pid $!"
done
sleep 60
while pgrep -f "runs/v378-rect/rect" >/dev/null; do sleep 60; done
log "batch O done"
(cd scripts && $P analyze_v378.py --recipe $RECIPE --runs ../runs --source $S --out ../runs/v378-rect/analysis.json) > runs/v378-rect/analysis.txt 2>&1
log "batch O analysis: $(grep PRIMARY runs/v378-rect/analysis.txt)"
