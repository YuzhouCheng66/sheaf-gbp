#!/bin/bash
# bigboy batch E3-L3 "全分辨率、单通道三级的学习隐码 sheaf GBP" (Claude). Pre-registration: OVERNIGHT_REPORT D8-63 (criteria hard-coded in
# g4_e3l3.py D8_63; operationalisations P1-P12 in its header). Structure = chain_day7_E3L2.sh (sha pins, FILL_ check, lock, never overwrite,
# import self-check, preflight timed on updates 200-400 only, updates = floor(0.90 x avail / rate), same for both seeds, < 2,000 -> INCOMPLETE).
# Differences (D8-63): bigboy, NOT uk-linux; all paths under ~/claude-sheaf-gbp/; two RTX 6000 Ada in PARALLEL: GPU 0 = seed 342101, GPU 1 =
# seed 342202 (GPUs 2 / 3 are never used; CUDA_DEVICE_ORDER=PCI_BUS_ID so the indices are nvidia-smi's, and the trainer asserts the device name);
# both preflights run in parallel (400 updates each), rate = the slower card; each seed <= 6,000 s wall clock from its process start (load +
# compile included); deadline (trainer, --deadline-epoch) and kill (timeout -k, which signals only its own child PID) per process, never by name.
# After training (CPU only, CUDA_VISIBLE_DEVICES= , <= 10 processes): select (thresholds on the 19 held-out TRAIN houses; the pinned
# ctrl_holdout.json of g4_e3l2 ctrlprep), eval on the EXISTING E3-R export (read only), aggregate against the E3-R summary (control) and the
# E3-L / E3-L2 (/ E5) summaries (report only). Writes only under ~/claude-sheaf-gbp/ (runs/e3l3, runs/chain_bigboy.log, tmp/).
# Disk: only checkpoints final.pt and upd6000.pt per seed (~5 MB each), eval job JSONs; total checked against 2 GB at the end.
# The coordinator fills every FILL_* after review; the chain refuses to run otherwise. Never launched by the implementer.
# Launch (never from an interactive shell that may die):
#   ssh bigboy 'cd ~/claude-sheaf-gbp/work && (setsid nohup bash scripts/chain_bigboy_E3L3.sh "D8-63" > runs/chain_bigboy_E3L3.out 2>&1 < /dev/null &)'
cd $HOME/claude-sheaf-gbp/work || exit 1
PREREG="$1"; [ "$PREREG" = "D8-63" ] || { echo "usage: chain_bigboy_E3L3.sh D8-63"; exit 2; }
B=$HOME/claude-sheaf-gbp
export HOUSEEXPO_JSON_DIR=$B/data/houseexpo-v365/json HOUSEEXPO_INDEX=$B/data/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=$HOME/claude-sheaf-gbp/data/e3r-pyroot:$HOME/claude-sheaf-gbp/data/e3r-pyroot/exploration_grid
export TMPDIR=$HOME/claude-sheaf-gbp/tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p $TMPDIR
PYROOT=$B/data/e3r-pyroot
P=$HOME/miniconda3/envs/mlpsp/bin/python
REF=$B/data/e3r-ref                                                      # e1c_fit.json = the evidence tables A / B (as E3-R / E3-L / E3-L2)
REF2=$B/data/e3l2-ref                                                    # ctrl_holdout.json (g4_e3l2.py ctrlprep; sha f0b55ed9), optional e5_summary.json
CACHE=$B/data/cache-e3r-v1                                               # TRAIN cache (E3-R), pinned by manifest sha
EXPORT=$B/data/cache-e3r-v1-export                                       # E3-R VAL / fresh export: READ ONLY, pinned by records content
E3R=runs/e3r                                                             # control: s342101/, s342202/ (checkpoints + thresholds), eval/summary.json
E3L=runs/e3l                                                             # E3-L (report only): eval/summary.json
E3L2=runs/e3l2                                                           # E3-L2 (report only): eval/summary.json
OUT=runs/e3l3
LOG=$HOME/claude-sheaf-gbp/work/runs/chain_bigboy.log
# ---------------------------------------------------------------- pinned by the coordinator (after review)
E3L3_SHA=FILL_E3L3_SHA                                                   # scripts/g4_e3l3.py as deployed
E5_SUM_SHA=FILL_E5_SUM_SHA_OR_none                                       # $REF2/e5_summary.json (local g4/e5/summary.json: b33297e3c7e02c911cffe2b54c2d0e0352de5a7f48be2f0bc773c07fafccb46c), or "none" = not reported
# ---------------------------------------------------------------- frozen (verified on bigboy 2026-10-27 22:1x UTC; = the E3-L2 chain's pins)
E3L2_SHA=bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314    # g4_e3l2.py (E3-L2, frozen; imported read only)
E3L_SHA=3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be     # g4_e3l.py (E3-L, frozen)
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970     # g4_e3r.py (E3-R, frozen)
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f       # g4_noise_panels.py (G4-E1, frozen)
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d       # mapping_3d_storey_data_v420.py
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
PYR_SHA=de96be65169582d0e3d2c7567261907e97af468d0bfe17d67156a6cdc93246c7     # e3r-pyroot content (scripts + exploration_grid *.py)
MAN_SHA=14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57     # $CACHE/manifest.json
FIT_SHA=8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2     # $REF/e1c_fit.json
E3R_SUM_SHA=8a0bd5ddf3e87a1c206c421e86fb1a6fd3447af0b8d0d4e3d379d06946e7f577 # $E3R/eval/summary.json (control)
TH1_SHA=c2dc9883968c9ce29fcbc95f2ac2728dc9ed29dbf8252c7971ef8b8fec3ef9bf     # $E3R/s342101/threshold.json
TH2_SHA=775a0d4874e3d01bf32113395e2b34df74016181b33c3eb4c6e630410e8c14e3     # $E3R/s342202/threshold.json
CK1_SHA=1b035fd1b6f0b01b136972505da834a384d61574e148f48e388e7fbbcdfb85cd     # $E3R/s342101/checkpoints/final.pt
CK2_SHA=a58b2b059d225ec95b438c56319a6945ac67ca779461db7d2cb059d99b5bb5c4     # $E3R/s342202/checkpoints/final.pt
EXPREC_SHA=62527a542b88e4142235788b53962a4aa16117ace6dc9158578225919d712675  # (cd $EXPORT/records && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum)
CTRLH_SHA=f0b55ed9343c3b1f1a9da468bd84733fabef6e76481aa639b4b468182f8657e4   # $REF2/ctrl_holdout.json (g4_e3l2.py ctrlprep; reproduces E3-R holdout .9806 / .9807)
E3L_SUM_SHA=7ea7cbbac682b6d326ae66426bc0fa84d54ea8d724bff3ac3690fd5abaff6a88 # $E3L/eval/summary.json (report only)
E3L2_SUM_SHA=b9bb7c0e20e1d5e5177cfcd91656eb58dd65ee50e0e86127f7a94a3d68953b1e # $E3L2/eval/summary.json (report only)
EV_342101="c:$REF/e1c_fit.json:A:cam_nf"                                  # D8-63 pairing (= E3-R / E3-L / E3-L2); asserted by the trainer against the E3-R checkpoint
EV_342202="c:$REF/e1c_fit.json:B:cam_nf"
BUDGET=6000                                                                # seconds per training seed, wall clock from process start (load + compile included)
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
incomplete(){ mkdir -p $OUT; echo "{\"verdict\": \"INCOMPLETE: $1\", \"stage\": \"chain\", \"time\": \"$(date -u +%FT%TZ)\"}" > $OUT/summary.json; log "E3-L3 INCOMPLETE: $1"; }
log "chain_bigboy_E3L3 started pid $$ host $(hostname) sha256 $(sha256sum scripts/chain_bigboy_E3L3.sh | cut -c1-16) prereg $PREREG"
if grep -q "FILL_" <<<"$E3L3_SHA $E5_SUM_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator pins them after review)"; exit 2; fi
[ "$(hostname)" = "bigboy" ] || { log "ABORT: not on bigboy"; exit 2; }
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
pyroot_sha(){ (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
exprec_sha(){ (cd $EXPORT/records && LC_ALL=C ls *.json | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
check_shas(){
  [ "$(sha scripts/g4_e3l3.py)" = "$E3L3_SHA" ] || { log "ABORT: g4_e3l3.py sha changed ($(sha scripts/g4_e3l3.py))"; exit 2; }
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
  [ "$(sha $E3R/eval/summary.json)" = "$E3R_SUM_SHA" ] || { log "ABORT: E3-R summary.json (control) sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/checkpoints/final.pt)" = "$CK1_SHA" ] || { log "ABORT: E3-R checkpoint 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/checkpoints/final.pt)" = "$CK2_SHA" ] || { log "ABORT: E3-R checkpoint 342202 sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/threshold.json)" = "$TH1_SHA" ] || { log "ABORT: E3-R threshold 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/threshold.json)" = "$TH2_SHA" ] || { log "ABORT: E3-R threshold 342202 sha changed"; exit 2; }
  [ "$(exprec_sha)" = "$EXPREC_SHA" ] || { log "ABORT: $EXPORT/records content differs from the E3-R export"; exit 2; }
  [ "$(sha $REF2/ctrl_holdout.json)" = "$CTRLH_SHA" ] || { log "ABORT: $REF2/ctrl_holdout.json sha changed or missing"; exit 2; }
  [ "$(sha $E3L/eval/summary.json)" = "$E3L_SUM_SHA" ] || { log "ABORT: E3-L summary.json sha changed"; exit 2; }
  [ "$(sha $E3L2/eval/summary.json)" = "$E3L2_SUM_SHA" ] || { log "ABORT: E3-L2 summary.json sha changed"; exit 2; }
  if [ "$E5_SUM_SHA" != "none" ]; then [ "$(sha $REF2/e5_summary.json)" = "$E5_SUM_SHA" ] || { log "ABORT: $REF2/e5_summary.json sha changed or missing"; exit 2; }; fi
}
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_bigboy_E3L3 already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if [ -e $OUT ]; then log "ABORT: $OUT already exists (never overwrite)"; exit 2; fi
FREE_MB=$(df -Pm $HOME | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /home (< 5,000)"; exit 2; fi
mkdir -p $OUT
# GPU 0 and 1 must be the two RTX 6000 Ada (PCI order = nvidia-smi index)
for g in 0 1; do
  nm=$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)
  case "$nm" in *"RTX 6000 Ada"*) ;; *) log "ABORT: GPU $g is '$nm', not an RTX 6000 Ada"; incomplete "GPU $g is not an RTX 6000 Ada"; exit 2;; esac
done
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader); U1=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
# ---------------------------------------------------------------- import self-check (same sys.path as every step)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 300 $P -B scripts/g4_e3l3.py selfcheck --pyroot $PYROOT 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "E3-L3 $SELF";; *) log "ABORT: import self-check failed (PYTHONPATH / scripts package): $SELF"; incomplete "import self-check failed"; exit 2;; esac
# GPU 0 / 1 idle: no compute process on either (other users' jobs on GPU 2 / 3 are ignored); wait and record otherwise
gpus_busy(){ nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q -e "$U0" -e "$U1"; }
while gpus_busy; do log "E3-L3: GPU 0 / 1 busy ($(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep -e "$U0" -e "$U1" | tr '\n' ';')); waiting 300 s"; sleep 300; done
sleep 30; while gpus_busy; do log "E3-L3: GPU 0 / 1 busy again; waiting 300 s"; sleep 300; done
check_shas
cat > $OUT/plan.json <<JSON
{"batch": "E3-L3", "prereg": "$PREREG", "host": "bigboy", "seeds": {"342101": {"gpu": 0, "evidence": "$EV_342101"}, "342202": {"gpu": 1, "evidence": "$EV_342202"}},
 "per_seed_budget_seconds": $BUDGET, "cuda_device_order": "PCI_BUS_ID", "gpu0_uuid": "$U0", "gpu1_uuid": "$U1",
 "control": {"342101": "$E3R/s342101", "342202": "$E3R/s342202", "summary": "$E3R/eval/summary.json", "summary_sha256": "$E3R_SUM_SHA", "holdout": "$REF2/ctrl_holdout.json", "holdout_sha256": "$CTRLH_SHA"},
 "trainer": "scripts/g4_e3l3.py train", "g4_e3l3_sha256": "$E3L3_SHA", "g4_e3l2_sha256": "$E3L2_SHA", "g4_e3l_sha256": "$E3L_SHA", "g4_e3r_sha256": "$E3R_SHA", "noise_panels_sha256": "$G_SHA",
 "pythonpath": "$PYTHONPATH", "python": "$P", "pyroot_content_sha256": "$PYR_SHA", "cache": "$CACHE", "cache_manifest_sha256": "$MAN_SHA", "export": "$EXPORT",
 "export_records_content_sha256": "$EXPREC_SHA", "e1c_fit_sha256": "$FIT_SHA", "chain_sha256": "$(sha scripts/chain_bigboy_E3L3.sh)", "resume": "none"}
JSON
# ================================================================ preflight: both cards in parallel, 400 updates, timed on updates 200-400
PFS=$(date +%s)
log "E3-L3 preflight start (GPU 0: 342101, GPU 1: 342202; 400 updates each, timed 200-400, one probe at 400)"
CUDA_VISIBLE_DEVICES=0 timeout -k 30 1200 $P -B scripts/g4_e3l3.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342101" --seed 342101 --out $OUT/pf-s342101 --updates 100000 --max-updates 400 --log-every 50 --time-from 200 --probe-every 400 --snap 0 --deadline-epoch $((PFS+1170)) --reserve 0 --expect-gpu 'RTX 6000 Ada' --ctrl-holdout $REF2/ctrl_holdout.json > $OUT/pf-s342101.log 2>&1 &
PF1=$!
CUDA_VISIBLE_DEVICES=1 timeout -k 30 1200 $P -B scripts/g4_e3l3.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342202" --seed 342202 --out $OUT/pf-s342202 --updates 100000 --max-updates 400 --log-every 50 --time-from 200 --probe-every 400 --snap 0 --deadline-epoch $((PFS+1170)) --reserve 0 --expect-gpu 'RTX 6000 Ada' --ctrl-holdout $REF2/ctrl_holdout.json > $OUT/pf-s342202.log 2>&1 &
PF2=$!
echo "{\"preflight_pids\": {\"342101\": $PF1, \"342202\": $PF2}}" > $OUT/pf_pids.json
wait $PF1; PFX1=$?; wait $PF2; PFX2=$?
pfread(){ $P -c "import json; f=json.load(open('$1/final.json')); print(f['rate_s_per_update'] or 'nan', f['load_seconds'], f['first_window_s'] or 0, f['probe_seconds_max'] or 0, f['rate_cv'] if f['rate_cv'] is not None else 'nan', f['updates_done'])" 2>/dev/null || echo "nan 0 0 0 nan 0"; }
read R1 L1 W1 Q1 CV1 N1 <<<"$(pfread $OUT/pf-s342101)"
read R2 L2 W2 Q2 CV2 N2 <<<"$(pfread $OUT/pf-s342202)"
PFW1=$($P -c "import json; print(json.load(open('$OUT/pf-s342101/final.json')).get('rate_warning') or 'none')" 2>/dev/null || echo "unreadable")
PFW2=$($P -c "import json; print(json.load(open('$OUT/pf-s342202/final.json')).get('rate_warning') or 'none')" 2>/dev/null || echo "unreadable")
UPD=$($P -c "
import math
r1, r2 = float('$R1'), float('$R2')
ok = all(map(math.isfinite, (r1, r2))) and min(r1, r2) > 0 and int('$N1') == 400 and int('$N2') == 400
rate = max(r1, r2)                                          # the slower card sets the common update count
L_, W_, Q_ = max(float('$L1'), float('$L2')), max(float('$W1'), float('$W2')), max(float('$Q1'), float('$Q2'))
base = $BUDGET - 60 - L_ - W_ - 30                          # margin, data load, first window (50 updates + cudnn autotune), save / exit
u0 = math.floor(0.90 * base / rate) if ok else 0
avail = base - Q_ * (u0 // 1000 + 1)                        # the probes (every 1,000 updates and at the end; record only)
print(int(max(0, min(200000, math.floor(0.90 * avail / rate)))) if ok else 0)" 2>/dev/null || echo 0)
log "E3-L3 preflight (exit $PFX1 / $PFX2): s/update GPU0 $R1 (cv $CV1, warning: $PFW1) GPU1 $R2 (cv $CV2, warning: $PFW2); load ${L1}/${L2}s first window ${W1}/${W2}s probe ${Q1}/${Q2}s -> $UPD updates per seed (same for both seeds; rate = max; safety 0.90)"
echo "{\"preflight_exit\": [$PFX1, $PFX2], \"rate_s_per_update\": [\"$R1\", \"$R2\"], \"rate_cv\": [\"$CV1\", \"$CV2\"], \"rate_warning\": [\"$PFW1\", \"$PFW2\"], \"load_seconds\": [\"$L1\", \"$L2\"], \"first_window_seconds\": [\"$W1\", \"$W2\"], \"probe_seconds\": [\"$Q1\", \"$Q2\"], \"timed_updates\": \"200-400\", \"updates\": $UPD}" > $OUT/updates.json
if [ "$UPD" -lt 2000 ]; then incomplete "budget: $UPD updates < 2,000 (preflight exit $PFX1 / $PFX2)"; exit 3; fi
# ================================================================ training: both seeds in parallel, <= 6,000 s each from process start
while gpus_busy; do log "E3-L3: GPU 0 / 1 busy before training; waiting 60 s"; sleep 60; done
check_shas
TS=$(date +%s)
log "E3-L3 training start: $UPD updates per seed; GPU 0 seed 342101, GPU 1 seed 342202; trainer deadline $((TS+BUDGET-30)) (reserve 60), kill at $((TS+BUDGET)) (+30 s grace)"
CUDA_VISIBLE_DEVICES=0 timeout -k 30 $BUDGET $P -B scripts/g4_e3l3.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342101" --seed 342101 --out $OUT/s342101 --updates $UPD --log-every 200 --time-from 200 --probe-every 1000 --snap 6000 --deadline-epoch $((TS+BUDGET-30)) --reserve 60 --expect-gpu 'RTX 6000 Ada' --ctrl-ckpt $E3R/s342101/checkpoints/final.pt --ctrl-holdout $REF2/ctrl_holdout.json > $OUT/s342101.log 2>&1 &
T1=$!
CUDA_VISIBLE_DEVICES=1 timeout -k 30 $BUDGET $P -B scripts/g4_e3l3.py train --cache $CACHE --manifest-sha $MAN_SHA --evidence "$EV_342202" --seed 342202 --out $OUT/s342202 --updates $UPD --log-every 200 --time-from 200 --probe-every 1000 --snap 6000 --deadline-epoch $((TS+BUDGET-30)) --reserve 60 --expect-gpu 'RTX 6000 Ada' --ctrl-ckpt $E3R/s342202/checkpoints/final.pt --ctrl-holdout $REF2/ctrl_holdout.json > $OUT/s342202.log 2>&1 &
T2=$!
echo "{\"train_start_epoch\": $TS, \"pids\": {\"342101\": $T1, \"342202\": $T2}, \"budget_seconds\": $BUDGET}" > $OUT/train_pids.json
wait $T1; X1=$?; wait $T2; X2=$?
TE=$(date +%s)
for s in 342101 342202; do
  RS=$($P -c "import json; d=json.load(open('$OUT/s$s/final.json')); r=d['rate_s_per_update']; print('%d updates (%s), %.4f s/update (%+.0f %% vs preflight max), train %.0f s, total %.0f s of $BUDGET, w %.1f' % (d['updates_done'], 'complete' if d['complete'] else 'TRUNCATED' if d['truncated'] else 'partial', r, 100 * (r / max(float('$R1'), float('$R2')) - 1), d['train_seconds'], d['seconds'], d['w']))" 2>/dev/null || echo "final.json unreadable")
  log "E3-L3 seed $s: $RS; $(tail -1 $OUT/s$s.log)"
done
log "E3-L3 training over after $((TE-TS)) s (exit $X1 / $X2); the GPUs are not used again by this chain"
# ================================================================ after training: CPU only (<= 10 processes; the E3-R export is read, never written)
check_shas
CTRLH=$REF2/ctrl_holdout.json
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] || { log "E3-L3 seed $s: no final.pt, no select"; continue; }
  log "E3-L3 select seed $s start (10 workers x 1 thread, timeout 10,800 s)"
  CUDA_VISIBLE_DEVICES= timeout -k 30 10800 $P -B scripts/g4_e3l3.py select --run $OUT/s$s --ctrl $CTRLH --workers 10 --threads 1 > $OUT/select_s$s.log 2>&1
  SX=$?; [ $SX -eq 124 ] && log "E3-L3 select seed $s TIMED OUT after 10,800 s"
  log "E3-L3 select seed $s (exit $SX): $(tail -1 $OUT/select_s$s.log)"
done
CKS=""
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] && [ -f $OUT/s$s/threshold.json ] && CKS="$CKS,$OUT/s$s/checkpoints/final.pt"
done
CKS=${CKS#,}
if [ -z "$CKS" ]; then incomplete "no trained checkpoint with thresholds"; exit 3; fi
log "E3-L3 eval start (10 workers x 1 thread, timeout 43,200 s): $CKS"
CUDA_VISIBLE_DEVICES= timeout -k 30 43200 $P -B scripts/g4_e3l3.py eval --export $EXPORT --out $OUT/eval --ckpt $CKS --workers 10 --threads 1 > $OUT/eval.log 2>&1
EX=$?; [ $EX -eq 124 ] && log "E3-L3 eval TIMED OUT after 43,200 s (missing jobs -> INCOMPLETE)"
log "E3-L3 eval done (exit $EX): $(tail -1 $OUT/eval.log)"
E5ARG=""; [ "$E5_SUM_SHA" != "none" ] && E5ARG="--e5-summary $REF2/e5_summary.json"
CUDA_VISIBLE_DEVICES= timeout -k 30 1800 $P -B scripts/g4_e3l3.py aggregate --out $OUT/eval --runs $OUT/s342101,$OUT/s342202 --control-summary $E3R/eval/summary.json \
  --e3l-summary $E3L/eval/summary.json --e3l2-summary $E3L2/eval/summary.json $E5ARG > $OUT/aggregate.log 2>&1
log "E3-L3 aggregate (exit $?): $(head -1 $OUT/aggregate.log)"
USE=$(du -sm $OUT | cut -f1)
[ "$USE" -gt 2048 ] && log "E3-L3 WARNING: $OUT uses $USE MB (> 2 GB)"
log "E3-L3 disk: $OUT $USE MB; /home free $(df -Pm $HOME | awk 'NR==2{print $4}') MB"
log "chain_bigboy_E3L3 finished"
