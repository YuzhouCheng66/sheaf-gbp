#!/bin/bash
# Report-only readouts for batch 3S-Z registered before its result (D8-30 house-size readout; D8-31: 3S-Z vs L1 under both decision rules,
# z1's TRAIN-optimal threshold). Re-scores the four finished checkpoints with eval_decision_v431.py on the CPU (nice 19; never the GPU),
# waiting for z1-s342202 to finish; then analyse_3SZ_rescore.py. Never changes 3S-Z's verdict (the chain's, v420 rule). Writes only
# runs/v431-rung-decision.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600
OUT=runs/v431-rung-decision
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
ESHA=$(sha256sum scripts/eval_decision_v431.py | cut -d' ' -f1); RSHA=$(sha256sum scripts/analyse_3SZ_rescore.py | cut -d' ' -f1)
[ "$ESHA" = "ef1275f158e10694b50e3deb9f8fa7d31c933cd62c341460c8a52171b39f5544" ] && [ "$RSHA" = "e5db3dbf26b75875df435bf694723b8486a13e9ec0436de4a95f72809c2409db" ] || { log "ABORT 3S-Z rescore: script sha changed ($ESHA $RSHA)"; exit 2; }
mkdir $OUT.lock 2>/dev/null || { log "ABORT 3S-Z rescore: already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
[ -e $OUT/rescore.json ] && { log "ABORT 3S-Z rescore: $OUT already has a result (never overwrite)"; exit 2; }
mkdir -p $OUT
log "3S-Z rescore start pid $$ (CPU only; report only)"
ev(){ [ -f $OUT/$2.json ] || nice -n 19 timeout 7200 $P scripts/eval_decision_v431.py --source $S --run $1 --out $OUT/$2.json --device cpu --threads 4 --workers 4 $3 > $OUT/$2.log 2>&1; log "3S-Z rescore $2 done (exit $?)"; }
ev runs/v429-ladder/l1-s342101 l1-s342101 ""
ev runs/v429-ladder/l1-s342202 l1-s342202 ""
ev runs/v431-rung/z1-s342101 z1-s342101 "--train-sweep --cache $CACHE"
for i in $(seq 1 180); do [ -f runs/v431-rung/z1-s342202/final.json ] && break; sleep 60; done
[ -f runs/v431-rung/z1-s342202/final.json ] || { log "ABORT 3S-Z rescore: z1-s342202 has no final.json after 3 h"; exit 2; }
ev runs/v431-rung/z1-s342202 z1-s342202 "--train-sweep --cache $CACHE"
for f in z1-s342101 z1-s342202 l1-s342101 l1-s342202; do [ -f $OUT/$f.json ] || { log "ABORT 3S-Z rescore: $f has no result"; exit 2; }; done
$P scripts/analyse_3SZ_rescore.py $OUT/z1-s342101.json $OUT/z1-s342202.json $OUT/l1-s342101.json $OUT/l1-s342202.json > $OUT/rescore.tmp 2> $OUT/rescore.err && mv $OUT/rescore.tmp $OUT/rescore.json
log "3S-Z rescore result: $($P -c "import json; a=json.load(open('$OUT/rescore.json')); print({k: (a[k]['reading'], a[k]['dR8_surface'], a[k]['dR8_var']) for k in ('val_v420', 'val_calibrated', 'fresh_calibrated')}, '| size', a['house_size_readout']['v420_K8'], '| z1 threshold', a['z1_train_optimal_occ_threshold'])" 2>&1 | tail -c 1500)"
