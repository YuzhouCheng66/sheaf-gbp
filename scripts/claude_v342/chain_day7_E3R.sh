#!/bin/bash
# Day-7 batch E3-R "读出端学习先验" (Claude). Pre-registration: OVERNIGHT_REPORT D8-44 (criteria there are authoritative); code review
# claude_handoff/analysis_20261027/g4/code_review_e3r.md (M1-M5, S1-S3 applied here and in g4_e3r.py).
# ONE 5090 lease, SOLO, two training seeds 342101 / 342202 run sequentially inside it (<= 3,000 s each, load + compile + threshold selection
# included); 7,200 s inclusive: lease start = before the preflight; training deadline +6,600 s, total deadline +7,150 s; timeout -k 30;
# no renewal, no resume. Before the lease (CPU): sha pins, import self-check (PYTHONPATH mirror e3r-pyroot), the D8-44 evidence rule from
# the FINAL E1c summary (asserted equal to the filled EV_*), complete E1c fit for the B6 controls. After the lease (CPU only,
# CUDA_VISIBLE_DEVICES= ): export of the VAL / fresh inputs (noise seeds 1, 2; tiers cam_nf / cam / clean; asserted to reproduce the
# G4-E1 job files), CPU inference + scoring, aggregation with the pre-registered criteria file.
# INCOMPLETE cases are decided by g4_e3r.py verdict (review M3); early exits also write $OUT/summary.json (review S6).
# Writes only under /mnt/datasets (runs/e3r, $EXPORT). The coordinator fills every FILL_* after review; the chain refuses to run otherwise.
# Launch (never from an interactive shell that may die):
#   ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/chain_day7_E3R.sh "D8-44" > runs/chain_day7_E3R.out 2>&1 < /dev/null &)'
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; [ "$PREREG" = "D8-44" ] || { echo "usage: chain_day7_E3R.sh D8-44"; exit 2; }
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid   # review M1
PYROOT=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot                 # read-only mirror: exploration_grid/ + scripts/*.py (the world's dependencies)
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
REF=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref                       # deployed: e1_jobs/ (96 G4-E1 job files), e1c_fit.json, e1c_summary.json, criteria.json
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1               # TRAIN cache (g4_e3r.py gen, before the chain), pinned by manifest sha
EXPORT=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1-export       # VAL / fresh inputs, written after the lease; must not exist yet (M5)
OUT=runs/e3r
LOG=runs/chain_day7.log
# ---------------------------------------------------------------- pinned by the coordinator (after review)
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970
MAN_SHA=14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57
PYR_SHA=de96be65169582d0e3d2c7567261907e97af468d0bfe17d67156a6cdc93246c7                                        # (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum)
FIT_SHA=8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2
E1C_SUM_SHA=fcc9fe3a10d98d4ce20081557d15c5c272268052d988857bf99a895d14955011
CRIT_SHA=13f35095259a575677a9d3dea7971e56554232c756a017cbf8c4958681700a41
EV_342101="c:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref/e1c_fit.json:A:cam_nf"                                          # D8-44 rule (checked below): E1c LEVER / AMBIGUOUS -> "c:$REF/e1c_fit.json:A:cam_nf", else L
EV_342202="c:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref/e1c_fit.json:B:cam_nf"                                          #                                                    -> "c:$REF/e1c_fit.json:B:cam_nf", else L
B6ARGS="--b6 c:$REF/e1c_fit.json:A:cam_nf --b6 c:$REF/e1c_fit.json:B:cam_nf"   # D8-44 control C = max(B4, B6_A, B6_B) (B4 is always computed)
# ---------------------------------------------------------------- frozen dependencies
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f       # g4_noise_panels.py (G4-E1, frozen)
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d       # mapping_3d_storey_data_v420.py
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
E1JOBS_SHA=181acc314d29cce7fe2c7b4ac7681e4df5f02b2e39f59a1f478b8a7fbb6ee559   # review S7: LC_ALL=C ls *.json | sort | xargs cat | sha256sum
BUDGET=3000                                                                # seconds per training seed, load + compile + selection included
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
incomplete(){ mkdir -p $OUT; echo "{\"verdict\": \"INCOMPLETE: $1\", \"stage\": \"chain\", \"time\": \"$(date -u +%FT%TZ)\"}" > $OUT/summary.json; log "E3-R INCOMPLETE: $1"; }
log "chain_day7_E3R started pid $$ sha256 $(sha256sum scripts/chain_day7_E3R.sh | cut -c1-16) prereg $PREREG"
if grep -q "FILL_" <<<"$E3R_SHA $MAN_SHA $PYR_SHA $FIT_SHA $E1C_SUM_SHA $CRIT_SHA $EV_342101 $EV_342202"; then log "ABORT: unfilled FILL_ placeholders (coordinator pins them after review)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
pyroot_sha(){ (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
e1jobs_sha(){ (cd $REF/e1_jobs && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
check_shas(){
  [ "$(sha scripts/g4_e3r.py)" = "$E3R_SHA" ] || { log "ABORT: g4_e3r.py sha changed ($(sha scripts/g4_e3r.py))"; exit 2; }
  [ "$(sha scripts/g4_noise_panels.py)" = "$G_SHA" ] || { log "ABORT: g4_noise_panels.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_3d_storey_data_v420.py)" = "$D_SHA" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_houseexpo_data_v365.py)" = "$V365_SHA" ] || { log "ABORT: mapping_houseexpo_data_v365.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_houseexpo_data_v371.py)" = "$V371_SHA" ] || { log "ABORT: mapping_houseexpo_data_v371.py sha changed"; exit 2; }
  [ "$(sha $HOUSEEXPO_INDEX)" = "$IDX_SHA" ] || { log "ABORT: HouseExpo index sha changed"; exit 2; }
  [ "$(pyroot_sha)" = "$PYR_SHA" ] || { log "ABORT: e3r-pyroot mirror content changed ($(pyroot_sha))"; exit 2; }
  [ "$(sha $CACHE/manifest.json)" = "$MAN_SHA" ] || { log "ABORT: TRAIN cache manifest sha changed ($(sha $CACHE/manifest.json))"; exit 2; }
  [ "$(sha $REF/e1c_fit.json)" = "$FIT_SHA" ] || { log "ABORT: E1c fit.json sha changed"; exit 2; }
  [ "$(sha $REF/e1c_summary.json)" = "$E1C_SUM_SHA" ] || { log "ABORT: E1c summary.json sha changed"; exit 2; }
  [ "$(sha $REF/criteria.json)" = "$CRIT_SHA" ] || { log "ABORT: criteria.json sha changed"; exit 2; }
  [ "$(e1jobs_sha)" = "$E1JOBS_SHA" ] || { log "ABORT: $REF/e1_jobs content differs from the 96 G4-E1 job files"; exit 2; }
}
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day7_E3R already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if [ -e $OUT ] || [ -e $EXPORT ]; then log "ABORT: $OUT or $EXPORT already exists (never overwrite; review M5)"; exit 2; fi
mkdir -p $OUT
# ---------------------------------------------------------------- before the lease: import self-check (M1) and the D8-44 evidence rule (M4)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 180 $P -B scripts/g4_e3r.py selfcheck --pyroot $PYROOT --fit $REF/e1c_fit.json 2>&1 | tail -1)   # same sys.path as every step
case "$SELF" in "import self-check ok"*) log "E3-R $SELF";; *) log "ABORT: import self-check failed (PYTHONPATH / scripts package / incomplete E1c fit): $SELF"; incomplete "pre-lease import self-check failed"; exit 2;; esac
RULE=$(CUDA_VISIBLE_DEVICES= timeout -k 30 180 $P -B scripts/g4_e3r.py e1c-rule --summary $REF/e1c_summary.json --fit $REF/e1c_fit.json --ref-fit-path $REF/e1c_fit.json --expect-342101 "$EV_342101" --expect-342202 "$EV_342202" 2>/dev/null | tail -1)
RX=$?
case "$RULE" in "{"*'"problems": []}') log "E3-R D8-44 evidence rule ok: $RULE";; *) log "ABORT: D8-44 evidence rule / E1c fit check failed (exit $RX): $RULE"; incomplete "evidence rule check failed"; exit 2;; esac
# 5090 SOLO: no other chain (day-6 / day-7 / older), no project trainer / diag, no compute process on the GPU
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day7_E3R" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > $OUT/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "batch": "E3-R", "prereg": "$PREREG",
 "seeds": [342101, 342202], "per_seed_budget_seconds": $BUDGET, "evidence": {"342101": "$EV_342101", "342202": "$EV_342202"}, "e1c_rule": $RULE,
 "trainer": "scripts/g4_e3r.py train", "g4_e3r_sha256": "$E3R_SHA", "noise_panels_sha256": "$G_SHA", "data_module_sha256": "$D_SHA",
 "pythonpath": "$PYTHONPATH", "pyroot_content_sha256": "$PYR_SHA", "cache": "$CACHE", "cache_manifest_sha256": "$MAN_SHA",
 "e1c_fit_sha256": "$FIT_SHA", "e1c_summary_sha256": "$E1C_SUM_SHA", "e1_jobs_content_sha256": "$E1JOBS_SHA", "criteria_sha256": "$CRIT_SHA",
 "chain_sha256": "$(sha scripts/chain_day7_E3R.sh)", "renewal": "none", "resume": "none"}
JSON
log "E3-R lease start $START (train deadline +6600, total +7150)"
PF=$OUT/pf-s342101
timeout -k 30 600 $P -B scripts/g4_e3r.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342101" --seed 342101 --out $PF --updates 100000 --max-updates 300 \
  --log-every 50 --val-every 100000 --holdout-samples 8 --select 0 --deadline-epoch $((START+590)) --reserve 0 > $PF.log 2>&1
PFX=$?
read RATE LOADS WARM <<<"$($P -c "import json; f=json.load(open('$PF/final.json')); print(f['rate_s_per_update'] or 'nan', f['load_seconds'], f['first_window_s'] or 0)" 2>/dev/null || echo "nan 0 0")"
UPD=$($P -c "
import math
r, L, w = float('$RATE'), float('$LOADS'), float('$WARM')
avail = $BUDGET - 60 - L - w - 150 - 60          # margin, data load, first window (cudnn autotune), selection reserve, holdout curve
print(int(max(0, min(200000, math.floor(0.90 * avail / r)))) if math.isfinite(r) and r > 0 else 0)" 2>/dev/null || echo 0)
log "E3-R preflight (exit $PFX): s/update $RATE load ${LOADS}s first window ${WARM}s -> $UPD updates per seed (same for both seeds; safety 0.90)"
echo "{\"preflight_exit\": $PFX, \"rate_s_per_update\": \"$RATE\", \"load_seconds\": \"$LOADS\", \"first_window_seconds\": \"$WARM\", \"updates\": $UPD}" > $OUT/updates.json
if [ "$UPD" -lt 2000 ]; then incomplete "budget: $UPD updates < 2,000 (preflight exit $PFX)"; exit 3; fi
run_seed(){                                                                     # $1 = seed, $2 = evidence spec
  seed=$1; EV=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $(date +%s) -lt $((DE-900)) ]; do log "E3-R: GPU busy before seed $seed; waiting"; sleep 30; done   # review S12
  NOW=$(date +%s); SDT=$((NOW+BUDGET)); [ $SDT -gt $DT ] && SDT=$DT; SDE=$((SDT-30)); [ $SDE -gt $DE ] && SDE=$DE
  if [ $((SDE-NOW)) -lt 900 ]; then log "E3-R seed $seed: only $((SDE-NOW)) s left in the lease -> not run (the verdict will be INCOMPLETE)"; return 1; fi
  log "E3-R seed $seed start ($UPD updates, evidence $EV, trainer deadline $SDE, kill at $SDT)"
  timeout -k 30 $((SDT-NOW)) $P -B scripts/g4_e3r.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV" --seed $seed --out $OUT/s$seed \
    --updates $UPD --val-every $((UPD/10)) --log-every 200 --deadline-epoch $SDE --reserve 150 > $OUT/s$seed.log 2>&1
  log "E3-R seed $seed done (exit $?): $(tail -1 $OUT/s$seed.log)"
}
run_seed 342101 "$EV_342101"
run_seed 342202 "$EV_342202"
END=$(date +%s)
echo "{\"lease_end_epoch\": $END, \"lease_seconds\": $((END-START))}" > $OUT/lease_end.json
log "E3-R lease over after $((END-START)) s; the GPU is not used again by this chain"
# ================================================================ after the lease: CPU only
check_shas
CKS=""
for s in 342101 342202; do [ -f $OUT/s$s/checkpoints/final.pt ] && [ -f $OUT/s$s/threshold.json ] && CKS="$CKS,$OUT/s$s/checkpoints/final.pt"; done
CKS=${CKS#,}
CUDA_VISIBLE_DEVICES= timeout -k 30 7200 $P -B scripts/g4_e3r.py export --cache $EXPORT --out $OUT/export --panels val,fresh --seeds 1,2 --workers 10 --e1-jobs $REF/e1_jobs > $OUT/export.log 2>&1
log "E3-R export done (exit $?): $(grep -c "'cam_nf': True" $OUT/export.log) jobs with cam_nf reproduced"
if [ -z "$CKS" ]; then incomplete "no trained checkpoint with a threshold"; exit 3; fi
CUDA_VISIBLE_DEVICES= timeout -k 30 10800 $P -B scripts/g4_e3r.py eval --export $EXPORT --out $OUT/eval --ckpt $CKS $B6ARGS --workers 10 --threads 1 > $OUT/eval.log 2>&1
log "E3-R eval done (exit $?)"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P -B scripts/g4_e3r.py aggregate --out $OUT/eval --criteria $REF/criteria.json --runs $OUT/s342101,$OUT/s342202 > $OUT/aggregate.log 2>&1
log "E3-R aggregate (exit $?): $(head -1 $OUT/aggregate.log)"
log "chain_day7_E3R finished"
