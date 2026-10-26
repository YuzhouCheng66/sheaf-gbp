#!/bin/bash
# Batch 3S-DR (Claude, 2026-10-26; pre-registered in OVERNIGHT_REPORT D8-31, revised after code_review_v431_decision.md): re-score the
# 3S-X checkpoints (runs/v429-stack/c2-s*) under the v420, calibrated and occ_only decision rules on the VAL and fresh panels, plus the
# TRAIN-24 threshold sweep (mechanism). Evaluation only: CPU, nice 19, never the GPU (3S-Z trains there). Writes only runs/v431-decision.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v431-decision
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
ESHA=$(sha256sum scripts/eval_decision_v431.py | cut -d' ' -f1); TSHA=$(sha256sum scripts/train_3d_v431.py | cut -d' ' -f1); ASHA=$(sha256sum scripts/analyse_3SDR.py | cut -d' ' -f1)
[ "$ESHA" = "e39d4aa7e46f945f2ee8f463e00ac6f10eddb16969b453b904bbc9d35598c787" ] && [ "$TSHA" = "9097801fe7b32ec5bb6b6085f1b3c9c79a6249f2e9b82af6f359a92615b966c0" ] && [ "$ASHA" = "b7fa4343e5fe193a309b52f1b4471fb020c2dc618e1561f0215970ea844197b4" ] || { log "ABORT batch 3S-DR: script sha changed ($ESHA $TSHA $ASHA)"; exit 2; }
mkdir $OUT.lock 2>/dev/null || { log "ABORT batch 3S-DR: already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
[ -e $OUT/analysis.json ] && { log "ABORT batch 3S-DR: $OUT already has an analysis (never overwrite)"; exit 2; }
mkdir -p $OUT
log "batch 3S-DR start pid $$ (CPU only; eval $(echo $ESHA | cut -c1-16))"
for s in 342101 342202; do
  [ -f $OUT/c2-s$s.json ] || nice -n 19 timeout 7200 $P scripts/eval_decision_v431.py --source $S --run runs/v429-stack/c2-s$s --out $OUT/c2-s$s.json --device cpu --threads 4 --workers 4 --train-sweep --cache $CACHE > $OUT/c2-s$s.log 2>&1
  log "batch 3S-DR seed $s done (exit $?)"
done
$P scripts/analyse_3SDR.py $OUT/c2-s342101.json $OUT/c2-s342202.json > $OUT/analysis.json 2> $OUT/analysis.err
log "batch 3S-DR analysis: $($P -c "import json; a=json.load(open('$OUT/analysis.json')); f=a['fresh']['calibrated']; print(a['verdict'], '| guards', f['guards'], '| fresh deltas', f['deltas'], '| mechanism', a['mechanism'])" 2>&1 | tail -c 1500)"
