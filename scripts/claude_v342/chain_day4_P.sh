#!/bin/bash
# Day-4 batch P (Claude, written 2026-10-23 ~03:50 UTC, before the batch-O result and before any batch-P run).
# Question: is the decoder prior still data-limited at 7,200 TRAIN houses? (batch N: 600 -> 7,200 gave +9.6 / +5.3 pp R8.) This is the
# D4800 arm deferred from batch N (stopped there only to protect the D2400 arms' completion), now on two seeds and on the chart batch O
# selects. ONE change vs control: number of distinct TRAIN houses (14,400 vs 7,200) at the same 43,200 updates (3 vs 6 epochs).
# Chart (mechanical): houseexpo_rect if batch O's analysis verdict is "adopt rectangular chart", else houseexpo_fit.
# Control: the same-chart D2400 run (batch O rect-d2400-s<seed> or batch N d2400-s<seed>). Trainer train_v378.py (reviewed; the training
# path equals train_v375 for houseexpo_fit), caches: houseexpo_fit-k4800 (reviewed digests) / houseexpo_rect-k4800 (paired poses).
# Pre-registered reading (analyze_v379.py): D4800 >= D2400 + 1 pp final VAL R8 on BOTH seeds => more data still helps; |gain| < 1 pp on
# both => saturated at 7,200; else ambiguous; incomplete run / identity mismatch => INCONCLUSIVE.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
C=/mnt/datasets/yuzhou-sheaf-gbp-research/hx-cache-v375
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day4_P started pid $$ sha256 $(sha256sum scripts/chain_day4_P.sh | cut -c1-16)"
mkdir -p runs/v379-dose
until grep -q "batch O analysis" $LOG && grep -q "houseexpo_rect-paired k4800 exit 0" $C/done_v378.txt 2>/dev/null; do sleep 60; done
while pgrep -f "runs/v378-rect/rect" >/dev/null || pgrep -f "runs/v369-rate/l" >/dev/null; do sleep 60; done
if grep -q "adopt rectangular chart" runs/v378-rect/analysis.txt 2>/dev/null; then WORLD=houseexpo_rect; else WORLD=houseexpo_fit; fi
log "batch P start: world $WORLD (batch-O rule)"
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v379-dose/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "world": "$WORLD",
 "arms": {"d4800-s342101": "4800/stratum, 3 ep", "d4800-s342202": "4800/stratum, 3 ep"}, "updates_each": 43200,
 "trainer_sha256": "$(sha256sum scripts/train_v378.py | cut -d' ' -f1)", "cache_manifest_sha256": "$(sha256sum $C/$WORLD-k4800/manifest.json | cut -d' ' -f1)",
 "expectation": "D4800 >= D2400 + 1pp final VAL R8 on both seeds => more data still helps; |gain| < 1pp => saturated; incomplete/identity mismatch => INCONCLUSIVE"}
JSON
for seed in 342101 342202; do
  nohup $P scripts/train_v378.py --source $S --world $WORLD --train-cache $C/$WORLD-k4800 --out runs/v379-dose/d4800-s$seed --arm owned --size 128 --device cuda --epochs 3 --val-every 1 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 43200 > runs/v379-dose/d4800-s$seed.log 2>&1 &
  log "dose d4800 seed $seed pid $!"
done
sleep 60
while pgrep -f "runs/v379-dose/d4800" >/dev/null; do sleep 60; done
log "batch P done"
(cd scripts && $P analyze_v379.py --world $WORLD --runs ../runs --neardup neardup2_v375.json --out ../runs/v379-dose/analysis.json) > runs/v379-dose/analysis.txt 2>&1
log "batch P analysis: $(grep PRIMARY runs/v379-dose/analysis.txt)"
