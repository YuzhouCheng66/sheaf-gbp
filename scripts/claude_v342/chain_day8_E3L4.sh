#!/bin/bash
# Day-8 batch E3-L4 "在固定的有限 w* 下训练学习隐码 sheaf GBP" (Claude). Pre-registration: OVERNIGHT_REPORT D8-68 (criteria hard-coded in
# g4_e3l4.py D8_68 = g4_e3l3.py D8_63 on noise seeds 3 / 4; operationalisations Q1-Q9 in its header). Single variable vs E3-L3: w = w*,
# omega = log w* frozen (training and evaluation). Structure and safety rules = chain_day7_E3L2.sh (uk-linux 5090: sha pins, FILL_ check,
# lock, never overwrite, import self-check, SOLO GPU wait, ONE 7,200 s lease, timeout -k on our own child PIDs only, no pkill / killall,
# CPU stages after the lease); trainer flags = chain_bigboy_E3L3.sh (preflight 400 updates timed on updates 200-400 with one probe at 400;
# probes every 1,000; snapshot at 6,000; --ctrl-ckpt / --ctrl-holdout pairing).
# Lease: seeds 342101 / 342202 SEQUENTIALLY, each <= 3,000 s wall clock (load + compile included); lease start = before the preflight;
# training deadline +6,600 s, total deadline +7,150 s; no renewal, no resume. Updates = 9,562 for both seeds (D8-68); if 9,562 does not fit
# 0.90 x the available time per seed (E3-L3 rule: avail = 3,000 - 60 - load - first window - 30 - probes), both seeds run the number that
# fits (the SECOND DIFFERENCE, written into the log, updates.json, <run>/e3l4.json and final.json); < 2,000 -> INCOMPLETE.
# After the lease (CPU only, CUDA_VISIBLE_DEVICES= , nice 10, 9 workers + 1 parent = 10 processes, every stage under timeout):
#   select (thresholds on the 19 held-out TRAIN houses; pinned ctrl_holdout.json) -> control NET re-evaluated by g4_e3r.py eval on the
#   G4-E5 export (noise seeds 3 / 4) + ctrlagg -> eval of E3-L4 on the G4-E5 export (judged VAL + fresh demotion) -> aggregate = the
#   D8-68 verdict -> report only: eval on the E3-R export (noise seeds 1 / 2) + report12 vs the E3-R summary and the E3-L3 summary.
# Writes only under /mnt/datasets (runs/e3l4, runs/chain_day8.log, TMPDIR tmp-claude). The exports are read, never written.
# The coordinator fills W_STAR (G4-E7 step 0, TRAIN only) and re-pins E3L4_SHA if the reviewed wrapper changes; the chain refuses FILL_.
# Launch (never from an interactive shell that may die; one chain per ssh command whose text names only this chain):
#   timeout 60 ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/chain_day8_E3L4.sh "D8-68" > runs/chain_day8_E3L4.out 2>&1 < /dev/null &)'
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; [ "$PREREG" = "D8-68" ] || { echo "usage: chain_day8_E3L4.sh D8-68"; exit 2; }
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
mkdir -p $TMPDIR
PYROOT=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
REF=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref                       # e1c_fit.json = the evidence tables A / B (as E3-R / E3-L / E3-L2 / E3-L3)
REF2=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l2-ref                     # ctrl_holdout.json (g4_e3l2.py ctrlprep; = the E3-L3 control file)
REF4=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l4-ref                     # G4-E5 export (e5_export/), e5_summary.json, e3l3_summary.json (report only)
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1               # TRAIN cache (E3-R), pinned by manifest sha
EXPORT34=$REF4/e5_export                                                 # G4-E5 export, VAL / fresh x noise seeds 3, 4 (JUDGED): READ ONLY
EXPORT12=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1-export     # E3-R export, VAL / fresh x noise seeds 1, 2 (REPORT ONLY): READ ONLY
E3R=runs/e3r                                                             # E3-R checkpoints + thresholds (control NET), eval/summary.json (seeds 1 / 2)
OUT=runs/e3l4
LOG=runs/chain_day8.log
# ---------------------------------------------------------------- pinned by the coordinator (after review)
W_STAR=4                                                                        # G4-E7 step 0 (TRAIN only), one of 3 / 4 / 6 / 8
E3L4_SHA=54428ba6b4ccddf0445c9e1f6fa63e35950dd3905046e89546c10f51aa9f857b    # scripts/g4_e3l4.py as deployed (re-pin if the review changes it)
# ---------------------------------------------------------------- frozen (verified on uk-linux 2026-10-28 03:2x UTC)
E3L3_SHA=c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c    # g4_e3l3.py (E3-L3 as run on bigboy; imported read only)
E3L2_SHA=bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314    # g4_e3l2.py
E3L_SHA=3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be     # g4_e3l.py
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970     # g4_e3r.py (also runs the control NET eval)
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f       # g4_noise_panels.py
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d       # mapping_3d_storey_data_v420.py
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
PYR_SHA=de96be65169582d0e3d2c7567261907e97af468d0bfe17d67156a6cdc93246c7     # e3r-pyroot content (scripts + exploration_grid *.py)
MAN_SHA=14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57     # $CACHE/manifest.json
FIT_SHA=8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2     # $REF/e1c_fit.json
CTRLH_SHA=f0b55ed9343c3b1f1a9da468bd84733fabef6e76481aa639b4b468182f8657e4   # $REF2/ctrl_holdout.json
E3R_SUM_SHA=8a0bd5ddf3e87a1c206c421e86fb1a6fd3447af0b8d0d4e3d379d06946e7f577 # $E3R/eval/summary.json (control of the seeds 1 / 2 report)
TH1_SHA=c2dc9883968c9ce29fcbc95f2ac2728dc9ed29dbf8252c7971ef8b8fec3ef9bf     # $E3R/s342101/threshold.json
TH2_SHA=775a0d4874e3d01bf32113395e2b34df74016181b33c3eb4c6e630410e8c14e3     # $E3R/s342202/threshold.json
CK1_SHA=1b035fd1b6f0b01b136972505da834a384d61574e148f48e388e7fbbcdfb85cd     # $E3R/s342101/checkpoints/final.pt
CK2_SHA=a58b2b059d225ec95b438c56319a6945ac67ca779461db7d2cb059d99b5bb5c4     # $E3R/s342202/checkpoints/final.pt
EXPREC_SHA=62527a542b88e4142235788b53962a4aa16117ace6dc9158578225919d712675  # records content of $EXPORT12 (E3-R export, seeds 1 / 2)
E5REC_SHA=7554d1fd4f0797dddd1965de976ee00675c80e99a5c4560f5b7c9cded6590986   # records content of $EXPORT34 (= local g4/e5_export/records, 96 files)
E5RUN_SHA=fbf67c87472d33c684d4fe94bf023cdaba249055642c571b98010995a7b75f88   # $EXPORT34/export_run_VALIDATION.json (G4-E5 export run, g4_quant_wire a63ce7d2)
E5_SUM_SHA=b33297e3c7e02c911cffe2b54c2d0e0352de5a7f48be2f0bc773c07fafccb46c  # $REF4/e5_summary.json (G4-E5 summary; Q-inf cross-check + QPOST4, report only)
E3L3_SUM_SHA=e1e964c558fc0f8fa1a2620bc8a39046fc5ac30255dc189b992892b96306ca19 # $REF4/e3l3_summary.json (E3-L3 result, seeds 1 / 2, report only)
EV_342101="c:$REF/e1c_fit.json:A:cam_nf"                                  # D8-68 pairing (= E3-R / E3-L3); asserted by the trainer against the E3-R checkpoint
EV_342202="c:$REF/e1c_fit.json:B:cam_nf"
BUDGET=3000                                                                # seconds per training seed, wall clock from process start (load + compile included)
UPD_PREREG=9562                                                            # D8-68: E3-L3's planned count, both seeds
WORKERS=9                                                                  # CPU stages: 9 workers + 1 idle parent = 10 processes
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
incomplete(){ local r=${1//\"/\'}; mkdir -p $OUT; echo "{\"verdict\": \"INCOMPLETE: $r\", \"stage\": \"chain\", \"time\": \"$(date -u +%FT%TZ)\"}" > $OUT/summary.json; log "E3-L4 INCOMPLETE: $r"; }
log "chain_day8_E3L4 started pid $$ host $(hostname) sha256 $(sha256sum scripts/chain_day8_E3L4.sh | cut -c1-16) prereg $PREREG"
if grep -q -e "FILL_" -e "@E3L4" <<<"$W_STAR $E3L4_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator pins W_STAR after G4-E7 step 0)"; exit 2; fi
case "$W_STAR" in 3|4|6|8) ;; *) log "ABORT: W_STAR=$W_STAR is not one of 3 / 4 / 6 / 8 (G4-E7 step-0 grid)"; exit 2;; esac
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
pyroot_sha(){ (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
rec_sha(){ (cd $1/records && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
check_shas(){
  [ "$(sha scripts/g4_e3l4.py)" = "$E3L4_SHA" ] || { log "ABORT: g4_e3l4.py sha changed ($(sha scripts/g4_e3l4.py))"; exit 2; }
  [ "$(sha scripts/g4_e3l3.py)" = "$E3L3_SHA" ] || { log "ABORT: g4_e3l3.py sha changed"; exit 2; }
  [ "$(sha scripts/g4_e3l2.py)" = "$E3L2_SHA" ] || { log "ABORT: g4_e3l2.py sha changed"; exit 2; }
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
  [ "$(sha $REF2/ctrl_holdout.json)" = "$CTRLH_SHA" ] || { log "ABORT: $REF2/ctrl_holdout.json sha changed or missing"; exit 2; }
  [ "$(sha $E3R/eval/summary.json)" = "$E3R_SUM_SHA" ] || { log "ABORT: E3-R summary.json sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/threshold.json)" = "$TH1_SHA" ] || { log "ABORT: E3-R threshold 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/threshold.json)" = "$TH2_SHA" ] || { log "ABORT: E3-R threshold 342202 sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/checkpoints/final.pt)" = "$CK1_SHA" ] || { log "ABORT: E3-R checkpoint 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/checkpoints/final.pt)" = "$CK2_SHA" ] || { log "ABORT: E3-R checkpoint 342202 sha changed"; exit 2; }
  [ "$(rec_sha $EXPORT12)" = "$EXPREC_SHA" ] || { log "ABORT: $EXPORT12/records content differs from the E3-R export"; exit 2; }
  [ "$(rec_sha $EXPORT34)" = "$E5REC_SHA" ] || { log "ABORT: $EXPORT34/records content differs from the G4-E5 export"; exit 2; }
  [ "$(sha $EXPORT34/export_run_VALIDATION.json)" = "$E5RUN_SHA" ] || { log "ABORT: $EXPORT34/export_run_VALIDATION.json sha changed"; exit 2; }
  [ "$(sha $REF4/e5_summary.json)" = "$E5_SUM_SHA" ] || { log "ABORT: $REF4/e5_summary.json sha changed or missing"; exit 2; }
  [ "$(sha $REF4/e3l3_summary.json)" = "$E3L3_SUM_SHA" ] || { log "ABORT: $REF4/e3l3_summary.json sha changed or missing"; exit 2; }
}
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day8_E3L4 already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if [ -e $OUT ]; then log "ABORT: $OUT already exists (never overwrite)"; exit 2; fi
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
mkdir -p $OUT
# ---------------------------------------------------------------- before the lease: import self-check + w* check (same sys.path as every step)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 180 $P -B scripts/g4_e3l4.py selfcheck --pyroot $PYROOT --w-star $W_STAR 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "E3-L4 $SELF (w* $W_STAR)";; *) log "ABORT: import self-check failed (PYTHONPATH / scripts package / w*): $SELF"; incomplete "pre-lease import self-check failed"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day8_E3L4" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > $OUT/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "batch": "E3-L4", "prereg": "$PREREG",
 "w_star": $W_STAR, "seeds": [342101, 342202], "sequential": true, "per_seed_budget_seconds": $BUDGET, "updates_prereg": $UPD_PREREG,
 "evidence": {"342101": "$EV_342101", "342202": "$EV_342202"},
 "control": {"342101": "$E3R/s342101", "342202": "$E3R/s342202", "holdout": "$REF2/ctrl_holdout.json", "holdout_sha256": "$CTRLH_SHA",
             "judged_export": "$EXPORT34", "judged_export_records_sha256": "$E5REC_SHA", "report_export": "$EXPORT12", "report_summary": "$E3R/eval/summary.json"},
 "trainer": "scripts/g4_e3l4.py train (wraps g4_e3l3.py train)", "g4_e3l4_sha256": "$E3L4_SHA", "g4_e3l3_sha256": "$E3L3_SHA", "g4_e3l2_sha256": "$E3L2_SHA",
 "g4_e3l_sha256": "$E3L_SHA", "g4_e3r_sha256": "$E3R_SHA", "noise_panels_sha256": "$G_SHA", "pythonpath": "$PYTHONPATH", "pyroot_content_sha256": "$PYR_SHA",
 "cache": "$CACHE", "cache_manifest_sha256": "$MAN_SHA", "e1c_fit_sha256": "$FIT_SHA", "chain_sha256": "$(sha scripts/chain_day8_E3L4.sh)", "renewal": "none", "resume": "none"}
JSON
log "E3-L4 lease start $START (train deadline +6600, total +7150); w* $W_STAR"
PF=$OUT/pf-s342101
timeout -k 30 600 $P -B scripts/g4_e3l4.py train --w-star $W_STAR --prereg-updates $UPD_PREREG --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342101" --seed 342101 --out $PF --updates 100000 --max-updates 400 --log-every 50 --time-from 200 --probe-every 400 --snap 0 --deadline-epoch $((START+590)) --reserve 0 --expect-gpu 'RTX 5090' --ctrl-holdout $REF2/ctrl_holdout.json > $PF.log 2>&1
PFX=$?
read RATE LOADS WARM PROBE CV NPF <<<"$($P -c "import json; f=json.load(open('$PF/final.json')); print(f['rate_s_per_update'] or 'nan', f['load_seconds'], f['first_window_s'] or 0, f['probe_seconds_max'] or 0, f['rate_cv'] if f['rate_cv'] is not None else 'nan', f['updates_done'])" 2>/dev/null || echo "nan 0 0 0 nan 0")"
PFW=$($P -c "import json; print(json.load(open('$PF/final.json')).get('rate_warning') or 'none')" 2>/dev/null || echo "unreadable")
read UPDFIT UPD <<<"$($P -c "
import math
r, L_, W_, Q_, n = float('$RATE'), float('$LOADS'), float('$WARM'), float('$PROBE'), int('$NPF')
ok = math.isfinite(r) and r > 0 and n == 400
base = $BUDGET - 60 - L_ - W_ - 30                          # margin, data load, first window (50 updates + cudnn autotune), save / exit
u0 = min(math.floor(0.90 * base / r), $UPD_PREREG) if ok else 0
avail = base - Q_ * (u0 // 1000 + 1)                        # the probes (every 1,000 updates and at the end; record only)
fit = int(max(0, min(200000, math.floor(0.90 * avail / r)))) if ok else 0
print(fit, min(fit, $UPD_PREREG))" 2>/dev/null || echo "0 0")"
SECOND_NOTE=""
if [ "$UPD" -lt "$UPD_PREREG" ] && [ "$UPD" -ge 2000 ]; then
  SECOND_NOTE="second difference (D8-68): $UPD_PREREG updates per seed do not fit 0.90 x the available time on the 5090 (preflight $RATE s/update on updates 200-400, load ${LOADS} s, first window ${WARM} s, probe ${PROBE} s, $BUDGET s per seed; fit $UPDFIT) -> both seeds run $UPD updates (E3-L3 rule)"
  log "E3-L4 SECOND DIFFERENCE: $SECOND_NOTE"
fi
log "E3-L4 preflight (exit $PFX): s/update $RATE (updates 200-400, window cv $CV, warning: $PFW) load ${LOADS}s first window ${WARM}s probe ${PROBE}s -> fit $UPDFIT, $UPD updates per seed (pre-registered $UPD_PREREG; same for both seeds; safety 0.90)"
$P -c "import json; json.dump(dict(preflight_exit=$PFX, rate_s_per_update='$RATE', rate_cv='$CV', rate_warning='$PFW', load_seconds='$LOADS', first_window_seconds='$WARM', probe_seconds='$PROBE', timed_updates='200-400', updates_fit=$UPDFIT, updates=$UPD, updates_prereg=$UPD_PREREG, second_difference=($UPD < $UPD_PREREG), second_difference_note='''$SECOND_NOTE''' or None, w_star=$W_STAR), open('$OUT/updates.json', 'w'), indent=1)" 2>/dev/null || log "E3-L4 WARNING: updates.json not written"
if [ "$UPD" -lt 2000 ]; then incomplete "budget: $UPD updates < 2,000 (preflight exit $PFX)"; exit 3; fi
run_seed(){                                                                     # $1 = seed, $2 = evidence spec
  seed=$1; EV=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $(date +%s) -lt $((DE-900)) ]; do log "E3-L4: GPU busy before seed $seed; waiting"; sleep 30; done
  NOW=$(date +%s); SDT=$((NOW+BUDGET)); [ $SDT -gt $DT ] && SDT=$DT; SDE=$((SDT-30)); [ $SDE -gt $DE ] && SDE=$DE
  if [ $((SDE-NOW)) -lt 900 ]; then log "E3-L4 seed $seed: only $((SDE-NOW)) s left in the lease -> not run (the verdict will be INCOMPLETE)"; return 1; fi
  log "E3-L4 seed $seed start ($UPD updates, w* $W_STAR frozen, evidence $EV, pairing $E3R/s$seed, snapshot 6000, trainer deadline $SDE, kill at $SDT)"
  timeout -k 30 $((SDT-NOW)) $P -B scripts/g4_e3l4.py train --w-star $W_STAR --prereg-updates $UPD_PREREG --second-note "$SECOND_NOTE" --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV" --seed $seed --out $OUT/s$seed --updates $UPD --log-every 200 --time-from 200 --probe-every 1000 --snap 6000 --deadline-epoch $SDE --reserve 60 --expect-gpu 'RTX 5090' --ctrl-ckpt $E3R/s$seed/checkpoints/final.pt --ctrl-holdout $REF2/ctrl_holdout.json > $OUT/s$seed.log 2>&1
  log "E3-L4 seed $seed done (exit $?): $(tail -1 $OUT/s$seed.log)"
  RS=$($P -c "import json; d=json.load(open('$OUT/s$seed/final.json')); r=d['rate_s_per_update']; print('%d updates (%s), %.4f s/update (%+.0f %% vs preflight), train %.0f s, total %.0f s of $BUDGET, w %s' % (d['updates_done'], 'complete' if d['complete'] else 'TRUNCATED' if d['truncated'] else 'partial', r, 100 * (r / float('$RATE') - 1), d['train_seconds'], d['seconds'], d['w']))" 2>/dev/null || echo "final.json unreadable")
  log "E3-L4 seed $seed: $RS"                                    # recorded only; the update count of the next seed is not changed
}
run_seed 342101 "$EV_342101"
run_seed 342202 "$EV_342202"
END=$(date +%s)
echo "{\"lease_end_epoch\": $END, \"lease_seconds\": $((END-START))}" > $OUT/lease_end.json
log "E3-L4 lease over after $((END-START)) s; the GPU is not used again by this chain"
# ================================================================ after the lease: CPU only (<= 10 processes; exports are read, never written)
check_shas
CTRLH=$REF2/ctrl_holdout.json
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] || { log "E3-L4 seed $s: no final.pt, no select"; continue; }
  log "E3-L4 select seed $s start ($WORKERS workers x 1 thread, nice 10, timeout 10,800 s)"
  CUDA_VISIBLE_DEVICES= timeout -k 30 10800 nice -n 10 $P -B scripts/g4_e3l4.py select --w-star $W_STAR --run $OUT/s$s --ctrl $CTRLH --workers $WORKERS --threads 1 > $OUT/select_s$s.log 2>&1
  SX=$?; [ $SX -eq 124 ] && log "E3-L4 select seed $s TIMED OUT after 10,800 s"
  log "E3-L4 select seed $s (exit $SX): $(tail -1 $OUT/select_s$s.log)"
done
log "E3-L4 control NET on the G4-E5 maps start (g4_e3r.py eval, E3-R checkpoints + thresholds, $WORKERS workers, timeout 14,400 s)"
CUDA_VISIBLE_DEVICES= timeout -k 30 14400 nice -n 10 $P -B scripts/g4_e3r.py eval --export $EXPORT34 --out $OUT/ctrl34 --ckpt $E3R/s342101/checkpoints/final.pt,$E3R/s342202/checkpoints/final.pt --workers $WORKERS --threads 1 > $OUT/ctrl34_eval.log 2>&1
CX=$?; [ $CX -eq 124 ] && log "E3-L4 control eval TIMED OUT after 14,400 s (missing jobs -> INCOMPLETE)"
log "E3-L4 control eval done (exit $CX): $(tail -1 $OUT/ctrl34_eval.log)"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/g4_e3l4.py ctrlagg --out $OUT/ctrl34 --expect-seeds 3,4 > $OUT/ctrl34_agg.log 2>&1
log "E3-L4 control summary (exit $?): $(tail -1 $OUT/ctrl34_agg.log)"
CKS=""
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] && [ -f $OUT/s$s/threshold.json ] && CKS="$CKS,$OUT/s$s/checkpoints/final.pt"
done
CKS=${CKS#,}
if [ -z "$CKS" ]; then incomplete "no trained checkpoint with thresholds"; exit 3; fi
log "E3-L4 eval (G4-E5 export, noise seeds 3 / 4) start ($WORKERS workers x 1 thread, nice 10, timeout 43,200 s): $CKS"
CUDA_VISIBLE_DEVICES= timeout -k 30 43200 nice -n 10 $P -B scripts/g4_e3l4.py eval --w-star $W_STAR --export $EXPORT34 --out $OUT/eval --ckpt $CKS --expect-seeds 3,4 --workers $WORKERS --threads 1 > $OUT/eval.log 2>&1
EX=$?; [ $EX -eq 124 ] && log "E3-L4 eval TIMED OUT after 43,200 s (missing jobs -> INCOMPLETE)"
log "E3-L4 eval done (exit $EX): $(tail -1 $OUT/eval.log)"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/g4_e3l4.py aggregate --w-star $W_STAR --out $OUT/eval --runs $OUT/s342101,$OUT/s342202 --control-summary $OUT/ctrl34/summary.json --e5-summary $REF4/e5_summary.json > $OUT/aggregate.log 2>&1
log "E3-L4 aggregate (exit $?): $(head -1 $OUT/aggregate.log)"
# ---------------------------------------------------------------- report only: noise seeds 1 / 2 (E3-R export) vs the E3-R summary and E3-L3
log "E3-L4 report-only eval (E3-R export, noise seeds 1 / 2) start ($WORKERS workers, nice 10, timeout 43,200 s)"
CUDA_VISIBLE_DEVICES= timeout -k 30 43200 nice -n 10 $P -B scripts/g4_e3l4.py eval --w-star $W_STAR --export $EXPORT12 --out $OUT/eval12 --ckpt $CKS --expect-seeds 1,2 --workers $WORKERS --threads 1 > $OUT/eval12.log 2>&1
log "E3-L4 report-only eval done (exit $?): $(tail -1 $OUT/eval12.log)"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 nice -n 10 $P -B scripts/g4_e3l4.py report12 --w-star $W_STAR --out $OUT/eval12 --control-summary $E3R/eval/summary.json --e3l3-summary $REF4/e3l3_summary.json > $OUT/report12.log 2>&1
log "E3-L4 report12 (exit $?): $(head -1 $OUT/report12.log)"
USE=$(du -sm $OUT | cut -f1)
log "E3-L4 disk: $OUT $USE MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day8_E3L4 finished"
