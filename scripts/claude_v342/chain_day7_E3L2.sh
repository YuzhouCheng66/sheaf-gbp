#!/bin/bash
# Day-7 batch E3-L2 "sheaf 隐码 GBP，硬容量码率" (Claude). Pre-registration: OVERNIGHT_REPORT D8-59 (criteria hard-coded in g4_e3l2.py D8_59;
# operationalisations P1-P12 in its header). Structure = chain_day7_E3L.sh (sha pins, lock, never overwrite, SOLO GPU wait, 7,200 s lease,
# same log line format). Changes (D8-59 "程序修正"): fixed-shape encoder blocks (in the trainer); preflight 400 updates timed on updates
# 200-400 only (window CV < 10 %, else the trainer reports a warning and the median window rate); updates = floor(0.90 x avail / rate) with no
# selection reserve; K and threshold selection AFTER the lease on CPU; a snapshot at update 6,510 (report only).
# ONE 5090 lease (the 3rd and last), SOLO, seeds 342101 / 342202 sequentially (<= 3,000 s each, load + compile included); lease start = before
# the preflight; training deadline +6,600 s, total deadline +7,150 s; timeout -k 30; no renewal, no resume.
# After the lease (CPU only, CUDA_VISIBLE_DEVICES= ): kselect (K* on the 19 held-out TRAIN houses x draws 200-202, the pinned control file),
# select (thresholds for every K of the grid; K0 at K*), the same two for the 6,510 snapshots (K* only), eval on the EXISTING E3-R export
# (read only), aggregation against the E3-R summary (control) and the E3-L summary (report only).
# Writes only under runs/e3l2. The coordinator fills every FILL_* after review; the chain refuses to run otherwise.
# Launch (never from an interactive shell that may die):
#   ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/chain_day7_E3L2.sh "D8-59" > runs/chain_day7_E3L2.out 2>&1 < /dev/null &)'
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; [ "$PREREG" = "D8-59" ] || { echo "usage: chain_day7_E3L2.sh D8-59"; exit 2; }
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid
PYROOT=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
REF=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref                       # e1c_fit.json = the evidence tables A / B (as E3-R / E3-L)
REF2=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l2-ref                     # ctrl_holdout.json (g4_e3l2.py ctrlprep, before launch), criteria_quantizer.json (E5)
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1               # TRAIN cache (E3-R), pinned by manifest sha
EXPORT=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1-export       # E3-R VAL / fresh export: READ ONLY, pinned by records content
E3R=runs/e3r                                                             # control: s342101/, s342202/ (checkpoints + thresholds), eval/summary.json
E3L=runs/e3l                                                             # E3-L (report only): eval/summary.json
OUT=runs/e3l2
LOG=runs/chain_day7.log
# ---------------------------------------------------------------- pinned by the coordinator (after review)
E3L2_SHA=FILL_g4_e3l2_sha256                                             # scripts/g4_e3l2.py as deployed
E3R_SUM_SHA=FILL_e3r_summary_sha256                                      # $E3R/eval/summary.json (local copy g4/e3r/summary.json: 8a0bd5ddf3e87a1c206c421e86fb1a6fd3447af0b8d0d4e3d379d06946e7f577)
TH1_SHA=FILL_e3r_threshold_342101_sha256                                 # $E3R/s342101/threshold.json (local copy: c2dc9883968c9ce29fcbc95f2ac2728dc9ed29dbf8252c7971ef8b8fec3ef9bf)
TH2_SHA=FILL_e3r_threshold_342202_sha256                                 # $E3R/s342202/threshold.json (local copy: 775a0d4874e3d01bf32113395e2b34df74016181b33c3eb4c6e630410e8c14e3)
EXPREC_SHA=FILL_export_records_content_sha256                            # (cd $EXPORT/records && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum) (local copy: 62527a542b88e4142235788b53962a4aa16117ace6dc9158578225919d712675)
CTRLH_SHA=FILL_ctrl_holdout_sha256                                       # $REF2/ctrl_holdout.json = g4_e3l2.py ctrlprep --cache $CACHE --e3r $E3R --out $REF2/ctrl_holdout.json (CPU, before launch)
E3L_SUM_SHA=FILL_e3l_summary_sha256                                      # $E3L/eval/summary.json (report only; local copy g4/e3l/summary.json)
QFILE_SHA=FILL_criteria_quantizer_sha256                                 # $REF2/criteria_quantizer.json (E5; local g4/e5_quant/criteria_quantizer.json: 8ad9d8242865f5b1ee332fee5f17952684959388a46282eb8a4d562bd6126266)
# ---------------------------------------------------------------- frozen (values of the E3-L / E3-R chains and summaries)
E3L_SHA=3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be    # g4_e3l.py (E3-L, frozen; imported read only; = the E3-L summary's script_sha256)
CK1_SHA=1b035fd1b6f0b01b136972505da834a384d61574e148f48e388e7fbbcdfb85cd    # $E3R/s342101/checkpoints/final.pt
CK2_SHA=a58b2b059d225ec95b438c56319a6945ac67ca779461db7d2cb059d99b5bb5c4    # $E3R/s342202/checkpoints/final.pt
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970    # g4_e3r.py (E3-R, frozen)
MAN_SHA=14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57
PYR_SHA=de96be65169582d0e3d2c7567261907e97af468d0bfe17d67156a6cdc93246c7
FIT_SHA=8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f      # g4_noise_panels.py (G4-E1, frozen)
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d      # mapping_3d_storey_data_v420.py
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
EV_342101="c:$REF/e1c_fit.json:A:cam_nf"                                  # D8-59 pairing (= E3-R / E3-L); asserted by the trainer against the E3-R checkpoint
EV_342202="c:$REF/e1c_fit.json:B:cam_nf"
BUDGET=3000                                                                # seconds per training seed, load + compile included (no selection in the lease)
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
incomplete(){ mkdir -p $OUT; echo "{\"verdict\": \"INCOMPLETE: $1\", \"stage\": \"chain\", \"time\": \"$(date -u +%FT%TZ)\"}" > $OUT/summary.json; log "E3-L2 INCOMPLETE: $1"; }
log "chain_day7_E3L2 started pid $$ sha256 $(sha256sum scripts/chain_day7_E3L2.sh | cut -c1-16) prereg $PREREG"
if grep -q "FILL_" <<<"$E3L2_SHA $E3R_SUM_SHA $TH1_SHA $TH2_SHA $EXPREC_SHA $CTRLH_SHA $E3L_SUM_SHA $QFILE_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator pins them after review)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
pyroot_sha(){ (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
exprec_sha(){ (cd $EXPORT/records && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
check_shas(){
  [ "$(sha scripts/g4_e3l2.py)" = "$E3L2_SHA" ] || { log "ABORT: g4_e3l2.py sha changed ($(sha scripts/g4_e3l2.py))"; exit 2; }
  [ "$(sha scripts/g4_e3l.py)" = "$E3L_SHA" ] || { log "ABORT: g4_e3l.py sha changed"; exit 2; }
  [ "$(sha scripts/g4_e3r.py)" = "$E3R_SHA" ] || { log "ABORT: g4_e3r.py sha changed"; exit 2; }
  [ "$(sha scripts/g4_noise_panels.py)" = "$G_SHA" ] || { log "ABORT: g4_noise_panels.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_3d_storey_data_v420.py)" = "$D_SHA" ] || { log "ABORT: mapping_3d_storey_data_v420.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_houseexpo_data_v365.py)" = "$V365_SHA" ] || { log "ABORT: mapping_houseexpo_data_v365.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_houseexpo_data_v371.py)" = "$V371_SHA" ] || { log "ABORT: mapping_houseexpo_data_v371.py sha changed"; exit 2; }
  [ "$(sha $HOUSEEXPO_INDEX)" = "$IDX_SHA" ] || { log "ABORT: HouseExpo index sha changed"; exit 2; }
  [ "$(pyroot_sha)" = "$PYR_SHA" ] || { log "ABORT: e3r-pyroot mirror content changed ($(pyroot_sha))"; exit 2; }
  [ "$(sha $CACHE/manifest.json)" = "$MAN_SHA" ] || { log "ABORT: TRAIN cache manifest sha changed ($(sha $CACHE/manifest.json))"; exit 2; }
  [ "$(sha $REF/e1c_fit.json)" = "$FIT_SHA" ] || { log "ABORT: E1c fit.json sha changed"; exit 2; }
  [ "$(sha $E3R/eval/summary.json)" = "$E3R_SUM_SHA" ] || { log "ABORT: E3-R summary.json (control) sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/checkpoints/final.pt)" = "$CK1_SHA" ] || { log "ABORT: E3-R checkpoint 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/checkpoints/final.pt)" = "$CK2_SHA" ] || { log "ABORT: E3-R checkpoint 342202 sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/threshold.json)" = "$TH1_SHA" ] || { log "ABORT: E3-R threshold 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/threshold.json)" = "$TH2_SHA" ] || { log "ABORT: E3-R threshold 342202 sha changed"; exit 2; }
  [ "$(exprec_sha)" = "$EXPREC_SHA" ] || { log "ABORT: $EXPORT/records content differs from the E3-R export"; exit 2; }
  [ "$(sha $REF2/ctrl_holdout.json)" = "$CTRLH_SHA" ] || { log "ABORT: $REF2/ctrl_holdout.json sha changed or missing"; exit 2; }
  [ "$(sha $E3L/eval/summary.json)" = "$E3L_SUM_SHA" ] || { log "ABORT: E3-L summary.json sha changed"; exit 2; }
  [ "$(sha $REF2/criteria_quantizer.json)" = "$QFILE_SHA" ] || { log "ABORT: E5 criteria_quantizer.json sha changed"; exit 2; }
}
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day7_E3L2 already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if [ -e $OUT ]; then log "ABORT: $OUT already exists (never overwrite)"; exit 2; fi
mkdir -p $OUT
# ---------------------------------------------------------------- before the lease: import self-check (same sys.path as every step)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 180 $P -B scripts/g4_e3l2.py selfcheck --pyroot $PYROOT 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "E3-L2 $SELF";; *) log "ABORT: import self-check failed (PYTHONPATH / scripts package): $SELF"; incomplete "pre-lease import self-check failed"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day7_E3L2" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > $OUT/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "batch": "E3-L2", "prereg": "$PREREG",
 "seeds": [342101, 342202], "per_seed_budget_seconds": $BUDGET, "evidence": {"342101": "$EV_342101", "342202": "$EV_342202"},
 "control": {"342101": "$E3R/s342101", "342202": "$E3R/s342202", "summary": "$E3R/eval/summary.json", "summary_sha256": "$E3R_SUM_SHA", "holdout": "$REF2/ctrl_holdout.json", "holdout_sha256": "$CTRLH_SHA"},
 "trainer": "scripts/g4_e3l2.py train", "g4_e3l2_sha256": "$E3L2_SHA", "g4_e3l_sha256": "$E3L_SHA", "g4_e3r_sha256": "$E3R_SHA", "noise_panels_sha256": "$G_SHA", "data_module_sha256": "$D_SHA",
 "pythonpath": "$PYTHONPATH", "pyroot_content_sha256": "$PYR_SHA", "cache": "$CACHE", "cache_manifest_sha256": "$MAN_SHA", "export": "$EXPORT",
 "export_records_content_sha256": "$EXPREC_SHA", "e1c_fit_sha256": "$FIT_SHA", "chain_sha256": "$(sha scripts/chain_day7_E3L2.sh)", "renewal": "none", "resume": "none"}
JSON
log "E3-L2 lease start $START (train deadline +6600, total +7150)"
PF=$OUT/pf-s342101
timeout -k 30 600 $P -B scripts/g4_e3l2.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342101" --seed 342101 --out $PF --updates 100000 --max-updates 400 \
  --log-every 50 --time-from 200 --val-every 100000 --holdout-samples 8 --snap 0 --deadline-epoch $((START+590)) --reserve 0 > $PF.log 2>&1
PFX=$?
read RATE LOADS WARM CV <<<"$($P -c "import json; f=json.load(open('$PF/final.json')); print(f['rate_s_per_update'] or 'nan', f['load_seconds'], f['first_window_s'] or 0, f['rate_cv'] if f['rate_cv'] is not None else 'nan')" 2>/dev/null || echo "nan 0 0 nan")"
PFW=$($P -c "import json; print(json.load(open('$PF/final.json')).get('rate_warning') or 'none')" 2>/dev/null || echo "unreadable")
UPD=$($P -c "
import math
r, L, w = float('$RATE'), float('$LOADS'), float('$WARM')
avail = $BUDGET - 60 - L - w - 60                  # margin, data load, first window (50 updates + cudnn autotune), holdout curve; no selection reserve
print(int(max(0, min(200000, math.floor(0.90 * avail / r)))) if math.isfinite(r) and r > 0 else 0)" 2>/dev/null || echo 0)
log "E3-L2 preflight (exit $PFX): s/update $RATE (updates 200-400, window cv $CV, warning: $PFW) load ${LOADS}s first window ${WARM}s -> $UPD updates per seed (same for both seeds; safety 0.90)"
echo "{\"preflight_exit\": $PFX, \"rate_s_per_update\": \"$RATE\", \"rate_cv\": \"$CV\", \"rate_warning\": \"$PFW\", \"load_seconds\": \"$LOADS\", \"first_window_seconds\": \"$WARM\", \"timed_updates\": \"200-400\", \"updates\": $UPD}" > $OUT/updates.json
if [ "$UPD" -lt 2000 ]; then incomplete "budget: $UPD updates < 2,000 (preflight exit $PFX)"; exit 3; fi
run_seed(){                                                                     # $1 = seed, $2 = evidence spec
  seed=$1; EV=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $(date +%s) -lt $((DE-900)) ]; do log "E3-L2: GPU busy before seed $seed; waiting"; sleep 30; done
  NOW=$(date +%s); SDT=$((NOW+BUDGET)); [ $SDT -gt $DT ] && SDT=$DT; SDE=$((SDT-30)); [ $SDE -gt $DE ] && SDE=$DE
  if [ $((SDE-NOW)) -lt 900 ]; then log "E3-L2 seed $seed: only $((SDE-NOW)) s left in the lease -> not run (the verdict will be INCOMPLETE)"; return 1; fi
  log "E3-L2 seed $seed start ($UPD updates, evidence $EV, pairing $E3R/s$seed, snapshot 6510, trainer deadline $SDE, kill at $SDT)"
  timeout -k 30 $((SDT-NOW)) $P -B scripts/g4_e3l2.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV" --seed $seed --out $OUT/s$seed \
    --updates $UPD --val-every $((UPD/10)) --log-every 200 --time-from 200 --snap 6510 --deadline-epoch $SDE --reserve 60 \
    --ctrl-ckpt $E3R/s$seed/checkpoints/final.pt > $OUT/s$seed.log 2>&1
  log "E3-L2 seed $seed done (exit $?): $(tail -1 $OUT/s$seed.log)"
  RS=$($P -c "import json; d=json.load(open('$OUT/s$seed/final.json')); r=d['rate_s_per_update']; print('%.4f (%+.0f %% vs preflight; train %.0f s of $BUDGET)' % (r, 100 * (r / float('$RATE') - 1), d['train_seconds']))" 2>/dev/null || echo "unreadable")
  log "E3-L2 seed $seed steady-state s/update $RS"             # recorded only; the update count of the next seed is not changed (D8-54)
}
run_seed 342101 "$EV_342101"
run_seed 342202 "$EV_342202"
END=$(date +%s)
echo "{\"lease_end_epoch\": $END, \"lease_seconds\": $((END-START))}" > $OUT/lease_end.json
log "E3-L2 lease over after $((END-START)) s; the GPU is not used again by this chain"
# ================================================================ after the lease: CPU only (K, thresholds, eval; the E3-R export is read, never written)
check_shas
CTRLH=$REF2/ctrl_holdout.json
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] || continue
  CUDA_VISIBLE_DEVICES= timeout -k 30 3600 $P -B scripts/g4_e3l2.py kselect --run $OUT/s$s --ctrl $CTRLH --workers 10 --threads 1 > $OUT/kselect_s$s.log 2>&1
  log "E3-L2 kselect seed $s (exit $?): $(tail -1 $OUT/kselect_s$s.log)"
  CUDA_VISIBLE_DEVICES= timeout -k 30 10800 $P -B scripts/g4_e3l2.py select --run $OUT/s$s --ctrl $CTRLH --kset all --workers 10 --threads 1 > $OUT/select_s$s.log 2>&1
  log "E3-L2 select seed $s (exit $?): $(tail -1 $OUT/select_s$s.log)"
  if [ -f $OUT/s$s/checkpoints/upd6510.pt ]; then                                 # the equal-step snapshot (report only)
    CUDA_VISIBLE_DEVICES= timeout -k 30 3600 $P -B scripts/g4_e3l2.py kselect --run $OUT/s$s --ctrl $CTRLH --name upd6510 --workers 10 --threads 1 > $OUT/kselect_upd6510_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES= timeout -k 30 3600 $P -B scripts/g4_e3l2.py select --run $OUT/s$s --ctrl $CTRLH --name upd6510 --kset kstar --workers 10 --threads 1 > $OUT/select_upd6510_s$s.log 2>&1
    log "E3-L2 snapshot 6510 seed $s (exit $?): $(tail -1 $OUT/select_upd6510_s$s.log)"
  fi
done
CKS=""; SNAPS=""
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] && [ -f $OUT/s$s/threshold.json ] && CKS="$CKS,$OUT/s$s/checkpoints/final.pt"
  [ -f $OUT/s$s/checkpoints/upd6510.pt ] && [ -f $OUT/s$s/threshold_upd6510.json ] && SNAPS="$SNAPS,$OUT/s$s/checkpoints/upd6510.pt"
done
CKS=${CKS#,}; SNAPS=${SNAPS#,}
if [ -z "$CKS" ]; then incomplete "no trained checkpoint with K* and thresholds"; exit 3; fi
CUDA_VISIBLE_DEVICES= timeout -k 30 14400 $P -B scripts/g4_e3l2.py eval --export $EXPORT --out $OUT/eval --ckpt $CKS --snap "$SNAPS" --qfile $REF2/criteria_quantizer.json \
  --workers 10 --threads 1 > $OUT/eval.log 2>&1
log "E3-L2 eval done (exit $?)"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P -B scripts/g4_e3l2.py aggregate --out $OUT/eval --runs $OUT/s342101,$OUT/s342202 --control-summary $E3R/eval/summary.json \
  --e3l-summary $E3L/eval/summary.json > $OUT/aggregate.log 2>&1
log "E3-L2 aggregate (exit $?): $(head -1 $OUT/aggregate.log)"
log "chain_day7_E3L2 finished"
