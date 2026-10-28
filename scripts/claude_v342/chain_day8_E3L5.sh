#!/bin/bash
# Day-8 batch E3-L5 "带形状多样性的训练分布，治本处理学习隐码的形状先验" (Claude). Pre-registration: OVERNIGHT_REPORT D8-76 (criteria
# hard-coded in g4_e3l5.py D8_76 / judge(); operationalisations F1-F5 there, V1-V8 in mapping_3d_storey_data_v420div.py, G1-G4 in
# g4_e3l5_gen.py). Single variable vs E3-L3: the TRAIN cache (cache-e3l5-div-v1 = v420 50 % + train_div 50 %; prepared on the uk-linux CPU
# by scripts/run_e3l5_gen.sh, NOT by this chain). Model, w learnable, optimiser, schedule, probes, threshold rule = g4_e3l3 unchanged.
# Structure and safety rules = chain_day8_E3L4.sh (uk-linux 5090: sha pins, FILL_ check, lock, never overwrite, import self-check, SOLO GPU wait,
# ONE 7,200 s lease, timeout -k on our own child PIDs only, no pkill / killall, CPU stages after the lease). Trainer flags = E3-L4's
# (preflight 400 updates timed on updates 200-400 with one probe at 400; probes every 1,000; snapshot at 6,000; --ctrl-ckpt / --ctrl-holdout).
# Lease: seeds 342101 / 342202 SEQUENTIALLY, each <= 3,000 s wall clock (load + compile included); lease start = before the preflight; training
# deadline +6,600 s, total deadline +7,150 s; no renewal, no resume. Updates = 9,562 for both seeds (E3-L3's planned count, D8-76 "步数口径不变");
# if 9,562 does not fit 0.90 x the available time per seed (E3-L3 rule: avail = 3,000 - 60 - load - first window - 30 - probes), both seeds run the
# number that fits (the SECOND DIFFERENCE, written into the log, updates.json, <run>/e3l5.json and final.json); < 2,000 -> INCOMPLETE.
# After the lease (CPU only, CUDA_VISIBLE_DEVICES= , nice 10, 9 workers + 1 parent = 10 processes, every stage under timeout):
#   select (learned w -> threshold.json; w = 4 -> threshold_w4.json; 19 held-out houses of the NEW cache; pinned ctrl_holdout_e3l5.json) ->
#   eval on the G4-E5 export (VAL noise seeds 3 / 4; tiers cam_nf + clean; arms E3L5W4 / E3L5 / E3L3W4 + NET; criteria 3 / 4) ->
#   h1 on the 192 G4-H1 VAL exports (ctrl / novel / scale / floating x seeds 1 / 2; Y-family zones; criteria 1 / 2) -> aggregate = the D8-76 verdict.
# Writes only under /mnt/datasets (runs/e3l5, runs/chain_day8.log, TMPDIR tmp-claude). Exports and refs are read, never written.
#
# The coordinator fills every FILL_ (after the review and after the user confirms the GPU; the chain refuses FILL_):
#   E3L5_SHA   scripts/g4_e3l5.py as deployed (re-pin if the review changes it)
#   MAN5_SHA   $CACHE5/manifest.json   (run_e3l5_gen.sh prints it; also in runs/e3l5_gen.log)
#   DIV5_SHA   $CACHE5/e3l5_div.json   (g4_e3l5_gen summary: Y totals 0, house set == cache-e3r-v1)
#   VER5_SHA   $CACHE5/verify_orig.json (the v420 half == cache-e3r-v1 array by array)
#   CTRLH5_SHA $REF5/ctrl_holdout_e3l5.json (g4_e3l2.py ctrlprep on the new cache, run by run_e3l5_gen.sh)
#   APPROVAL   the UTC time of the user's GPU confirmation (D8-76: 发射前须用户确认算力) -- passed as $2, written into lease.json
# Launch (never from an interactive shell that may die; one chain per ssh command whose text names only this chain):
#   timeout 60 ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/chain_day8_E3L5.sh "D8-76" "<APPROVAL UTC>" > runs/chain_day8_E3L5.out 2>&1 < /dev/null &)'
#
# IF THE COORDINATOR MOVES THIS BATCH TO BIGBOY (notes only; no second chain is written): start from chain_bigboy_E3L3.sh (two RTX 6000 Ada in
# PARALLEL, GPU 0 = 342101, GPU 1 = 342202, 6,000 s per seed, continuous 300 s idle check on GPU 0 / 1, record-only monitor of other processes,
# CUDA_DEVICE_ORDER=PCI_BUS_ID, --expect-gpu 'RTX 6000 Ada') and change:
#   cd / work dir      /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021      -> $HOME/claude-sheaf-gbp/work
#   python             /home/yuzhou/anaconda3/envs/jax-gpu/bin/python                    -> $HOME/miniconda3/envs/mlpsp/bin/python (as E3-L3; never modify envs)
#   HOUSEEXPO_*        /mnt/datasets/.../houseexpo-v365/{json,houseexpo_index_v365.json}  -> $HOME/claude-sheaf-gbp/data/houseexpo-v365/...
#   PYTHONPATH/PYROOT  /mnt/datasets/.../e3r-pyroot                                      -> $HOME/claude-sheaf-gbp/data/e3r-pyroot
#   TMPDIR             /mnt/datasets/.../tmp-claude                                      -> $HOME/claude-sheaf-gbp/tmp
#   REF  (e1c_fit)     /mnt/datasets/.../e3r-ref                                         -> $HOME/claude-sheaf-gbp/data/e3r-ref
#   CACHE5             /mnt/datasets/.../cache-e3l5-div-v1 (~130 MB)                     -> copy to $HOME/claude-sheaf-gbp/data/cache-e3l5-div-v1 (re-check MAN5_SHA)
#   REF5               /mnt/datasets/.../e3l5-ref (ctrl_holdout_e3l5.json, e3l3_s*/, criteria_w.json, e7_summary.json, h1_jobs/, h1_export/ ~55 MB)
#                                                                                         -> $HOME/claude-sheaf-gbp/data/e3l5-ref (same files, same shas)
#   EXPORT34           /mnt/datasets/.../e3l4-ref/e5_export (G4-E5, ~71 MB)              -> $HOME/claude-sheaf-gbp/data/e3l4-ref/e5_export
#   E3R                runs/e3r (E3-R checkpoints + thresholds)                            -> $HOME/claude-sheaf-gbp/work/runs/e3r (exists for E3-L3)
#   scripts            deploy g4_e3l5.py, mapping_3d_storey_data_v420ood.py, mapping_3d_storey_data_v420div.py, g4_e3l5_gen.py, g4_ood_panel.py,
#                      g4_h1_e3l3.py into ~/claude-sheaf-gbp/work/scripts (sha-checked as below)
#   budget             BUDGET=6000 per seed in parallel (as E3-L3: 9,562 updates fitted there at 0.52-0.55 s/update), seeds on GPU 0 / 1
#   LOG / OUT          $HOME/claude-sheaf-gbp/work/runs/chain_bigboy.log, runs/e3l5 ; disk: /home is ~96 % full -> keep the total under 2 GB
#   CPU stages         same commands; <= 10 processes; nice -n 10.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
PREREG="$1"; APPROVAL="$2"
[ "$PREREG" = "D8-76" ] && [ -n "$APPROVAL" ] || { echo "usage: chain_day8_E3L5.sh D8-76 <user GPU approval UTC>"; exit 2; }
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
mkdir -p $TMPDIR
PYROOT=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
REF=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref                       # e1c_fit.json = the evidence tables A / B (as E3-R .. E3-L4)
REF4=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l4-ref                     # G4-E5 export (e5_export/), as E3-L4
REF5=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l5-ref                     # ctrl_holdout_e3l5.json, e3l3_s*/ (frozen E3-L3), criteria_w.json + e7_summary.json (G4-E7), h1_jobs/ + h1_export/ (G4-H1 VAL)
CACHE5=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3l5-div-v1         # NEW TRAIN cache (the single variable), pinned by manifest sha
EXPORT34=$REF4/e5_export                                                 # G4-E5 export, VAL / fresh x noise seeds 3, 4: READ ONLY
E3R=runs/e3r                                                             # E3-R checkpoints + thresholds (control NET; ctrl-ckpt pairing)
OUT=runs/e3l5
LOG=runs/chain_day8.log
# ---------------------------------------------------------------- pinned by the coordinator (after review / after the cache is generated)
E3L5_SHA=FILL_E3L5_SHA                                                     # scripts/g4_e3l5.py as deployed
MAN5_SHA=FILL_MAN5_SHA                                                     # $CACHE5/manifest.json
DIV5_SHA=FILL_DIV5_SHA                                                     # $CACHE5/e3l5_div.json
VER5_SHA=FILL_VER5_SHA                                                     # $CACHE5/verify_orig.json
CTRLH5_SHA=FILL_CTRLH5_SHA                                                 # $REF5/ctrl_holdout_e3l5.json
# ---------------------------------------------------------------- frozen (verified on uk-linux 2026-10-28 11:2x-11:5x UTC)
GEN5_SHA=64168499650d0b2a54858cf9faa2aa2e17bbbf6edef233a0f2415e99cccb57a1     # g4_e3l5_gen.py (wrote the cache; imported by nothing here, pinned for provenance)
DIV_SHA=3286d879f8c3d3578742eda6f78c8a18066f8f23df766b15fe8bd7207e6ef2f5      # mapping_3d_storey_data_v420div.py (generator of the cache)
OOD_SHA=9e1c7c2aac5ddd3e53ed5035a3db707d87c1a1026987ece782796b39f4e1f92c      # mapping_3d_storey_data_v420ood.py (G4-H1 generator; imported by g4_ood_panel)
OP_SHA=3a6959da19eec5326f9a17e583d58e6ab1b12fd2a2c398172b61ace36a49ce23       # g4_ood_panel.py (H1 scoring; imported lazily)
H1L_SHA=dc0f69d7c8ea2b2a8513fa2c04095664ebd7eaf0f1c0637d497943b57537da57      # g4_h1_e3l3.py (H1L; e3l3_preds / compare_controls / z_zone_counts imported lazily)
E3L3_SHA=c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c    # g4_e3l3.py (E3-L3; the trainer)
E3L2_SHA=bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314    # g4_e3l2.py
E3L_SHA=3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be     # g4_e3l.py
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970     # g4_e3r.py
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f       # g4_noise_panels.py
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d       # mapping_3d_storey_data_v420.py
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
PYR_SHA=de96be65169582d0e3d2c7567261907e97af468d0bfe17d67156a6cdc93246c7     # e3r-pyroot content (scripts + exploration_grid *.py)
FIT_SHA=8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2     # $REF/e1c_fit.json
TH1_SHA=c2dc9883968c9ce29fcbc95f2ac2728dc9ed29dbf8252c7971ef8b8fec3ef9bf     # $E3R/s342101/threshold.json
TH2_SHA=775a0d4874e3d01bf32113395e2b34df74016181b33c3eb4c6e630410e8c14e3     # $E3R/s342202/threshold.json
CK1_SHA=1b035fd1b6f0b01b136972505da834a384d61574e148f48e388e7fbbcdfb85cd     # $E3R/s342101/checkpoints/final.pt
CK2_SHA=a58b2b059d225ec95b438c56319a6945ac67ca779461db7d2cb059d99b5bb5c4     # $E3R/s342202/checkpoints/final.pt
E5REC_SHA=7554d1fd4f0797dddd1965de976ee00675c80e99a5c4560f5b7c9cded6590986   # records content of $EXPORT34 (= local g4/e5_export/records, 96 files)
E5RUN_SHA=fbf67c87472d33c684d4fe94bf023cdaba249055642c571b98010995a7b75f88   # $EXPORT34/export_run_VALIDATION.json
L3CK1_SHA=1fa0cb396f8f112ff14108691de992c195130513cbd86c83c84481b3155ecf6f   # $REF5/e3l3_s342101/final.pt (E3-L3, comparator)
L3TH1_SHA=c583fa1ebad811db4f8e5ac2dc67fdbb7e2d6271a76b2fac0079b025cf50605e   # $REF5/e3l3_s342101/threshold.json
L3CK2_SHA=3d332d400813374869df58b7c19f6361899812e8d8b2fdc6d9305969c7d4111c   # $REF5/e3l3_s342202/final.pt
L3TH2_SHA=56f4bf7f9ecb33798dd5a14538f44632195a325ca8754f1399c53d122d2c8c73   # $REF5/e3l3_s342202/threshold.json
E7CRIT_SHA=320317aaefad21d85d613596a5156da5ad596b764d13a63bcaedaadc17b83060  # $REF5/criteria_w.json (G4-E7: w* = 4 and the E3-L3 w = 4 thresholds)
E7SUM_SHA=c45eaca7e456fdbc7cbe4bdb6a0ed6f832b0f724d3a07b438b79d7fec8565cd9   # $REF5/e7_summary.json (G4-E7 result; comparator gate)
H1REC_SHA=42899de8864221d9f2b2b8d73b92ceddd2fd28e4c391a0dd95527ad8150dc014   # content of $REF5/h1_jobs (96 G4-H1 VAL records)
H1NPZ_SHA=7cc07bd1b6aa8b94698d59f43dd2d8583d4af682c9c6bdcb149e8199c58e33c0   # content of $REF5/h1_export (192 G4-H1 VAL npz)
EV_342101="c:$REF/e1c_fit.json:A:cam_nf"                                  # pairing = E3-R / E3-L3 (asserted by the trainer against the E3-R checkpoint)
EV_342202="c:$REF/e1c_fit.json:B:cam_nf"
BUDGET=3000                                                                # seconds per training seed (5090, one lease, sequential), load + compile included
UPD_PREREG=9562                                                            # E3-L3's planned count (D8-76: 步数口径不变)
WORKERS=9                                                                  # CPU stages: 9 workers + 1 idle parent = 10 processes
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
incomplete(){ local r=${1//\"/\'}; mkdir -p $OUT; echo "{\"verdict\": \"INCOMPLETE: $r\", \"stage\": \"chain\", \"time\": \"$(date -u +%FT%TZ)\"}" > $OUT/summary.json; log "E3-L5 INCOMPLETE: $r"; }
log "chain_day8_E3L5 started pid $$ host $(hostname) sha256 $(sha256sum scripts/chain_day8_E3L5.sh | cut -c1-16) prereg $PREREG approval $APPROVAL"
if grep -q -e "FILL_" <<<"$E3L5_SHA $MAN5_SHA $DIV5_SHA $VER5_SHA $CTRLH5_SHA"; then log "ABORT: unfilled FILL_ placeholders (coordinator)"; exit 2; fi
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
pyroot_sha(){ (cd $PYROOT && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
rec_sha(){ (cd $1 && LC_ALL=C ls $2 | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64); }
check_shas(){
  [ "$(sha scripts/g4_e3l5.py)" = "$E3L5_SHA" ] || { log "ABORT: g4_e3l5.py sha changed ($(sha scripts/g4_e3l5.py))"; exit 2; }
  [ "$(sha scripts/g4_e3l5_gen.py)" = "$GEN5_SHA" ] || { log "ABORT: g4_e3l5_gen.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_3d_storey_data_v420div.py)" = "$DIV_SHA" ] || { log "ABORT: mapping_3d_storey_data_v420div.py sha changed"; exit 2; }
  [ "$(sha scripts/mapping_3d_storey_data_v420ood.py)" = "$OOD_SHA" ] || { log "ABORT: mapping_3d_storey_data_v420ood.py sha changed"; exit 2; }
  [ "$(sha scripts/g4_ood_panel.py)" = "$OP_SHA" ] || { log "ABORT: g4_ood_panel.py sha changed or missing"; exit 2; }
  [ "$(sha scripts/g4_h1_e3l3.py)" = "$H1L_SHA" ] || { log "ABORT: g4_h1_e3l3.py sha changed or missing ($(sha scripts/g4_h1_e3l3.py))"; exit 2; }
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
  [ "$(sha $CACHE5/manifest.json)" = "$MAN5_SHA" ] || { log "ABORT: E3-L5 cache manifest sha changed ($(sha $CACHE5/manifest.json))"; exit 2; }
  [ "$(sha $CACHE5/e3l5_div.json)" = "$DIV5_SHA" ] || { log "ABORT: E3-L5 cache e3l5_div.json sha changed"; exit 2; }
  [ "$(sha $CACHE5/verify_orig.json)" = "$VER5_SHA" ] || { log "ABORT: E3-L5 cache verify_orig.json sha changed"; exit 2; }
  [ "$(sha $REF5/ctrl_holdout_e3l5.json)" = "$CTRLH5_SHA" ] || { log "ABORT: $REF5/ctrl_holdout_e3l5.json sha changed or missing"; exit 2; }
  [ "$(sha $REF/e1c_fit.json)" = "$FIT_SHA" ] || { log "ABORT: E1c fit.json sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/threshold.json)" = "$TH1_SHA" ] || { log "ABORT: E3-R threshold 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/threshold.json)" = "$TH2_SHA" ] || { log "ABORT: E3-R threshold 342202 sha changed"; exit 2; }
  [ "$(sha $E3R/s342101/checkpoints/final.pt)" = "$CK1_SHA" ] || { log "ABORT: E3-R checkpoint 342101 sha changed"; exit 2; }
  [ "$(sha $E3R/s342202/checkpoints/final.pt)" = "$CK2_SHA" ] || { log "ABORT: E3-R checkpoint 342202 sha changed"; exit 2; }
  [ "$(rec_sha $EXPORT34/records '*.json')" = "$E5REC_SHA" ] || { log "ABORT: $EXPORT34/records content differs from the G4-E5 export"; exit 2; }
  [ "$(sha $EXPORT34/export_run_VALIDATION.json)" = "$E5RUN_SHA" ] || { log "ABORT: $EXPORT34/export_run_VALIDATION.json sha changed"; exit 2; }
  [ "$(sha $REF5/e3l3_s342101/final.pt)" = "$L3CK1_SHA" ] || { log "ABORT: E3-L3 checkpoint 342101 sha changed"; exit 2; }
  [ "$(sha $REF5/e3l3_s342101/threshold.json)" = "$L3TH1_SHA" ] || { log "ABORT: E3-L3 threshold 342101 sha changed"; exit 2; }
  [ "$(sha $REF5/e3l3_s342202/final.pt)" = "$L3CK2_SHA" ] || { log "ABORT: E3-L3 checkpoint 342202 sha changed"; exit 2; }
  [ "$(sha $REF5/e3l3_s342202/threshold.json)" = "$L3TH2_SHA" ] || { log "ABORT: E3-L3 threshold 342202 sha changed"; exit 2; }
  [ "$(sha $REF5/criteria_w.json)" = "$E7CRIT_SHA" ] || { log "ABORT: G4-E7 criteria_w.json sha changed"; exit 2; }
  [ "$(sha $REF5/e7_summary.json)" = "$E7SUM_SHA" ] || { log "ABORT: G4-E7 summary sha changed"; exit 2; }
  [ "$(rec_sha $REF5/h1_jobs '*.json')" = "$H1REC_SHA" ] || { log "ABORT: $REF5/h1_jobs content differs from the G4-H1 VAL records"; exit 2; }
  [ "$(rec_sha $REF5/h1_export '*.npz')" = "$H1NPZ_SHA" ] || { log "ABORT: $REF5/h1_export content differs from the G4-H1 VAL exports"; exit 2; }
}
check_shas
mkdir $OUT.lock 2>/dev/null || { log "ABORT: chain_day8_E3L5 already launched (lock exists)"; exit 2; }
trap 'rmdir $OUT.lock 2>/dev/null' EXIT
if [ -e $OUT ]; then log "ABORT: $OUT already exists (never overwrite)"; exit 2; fi
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}')
if [ "$FREE_MB" -lt 5000 ]; then log "ABORT: only ${FREE_MB} MB free on /mnt/datasets (< 5,000)"; exit 2; fi
mkdir -p $OUT
# ---------------------------------------------------------------- before the lease: import self-check (same sys.path as every step)
SELF=$(CUDA_VISIBLE_DEVICES= timeout -k 30 180 $P -B scripts/g4_e3l5.py selfcheck --pyroot $PYROOT 2>&1 | tail -1)
case "$SELF" in "import self-check ok"*) log "E3-L5 $SELF";; *) log "ABORT: import self-check failed (PYTHONPATH / scripts package): $SELF"; incomplete "pre-lease import self-check failed"; exit 2;; esac
# 5090 SOLO: no other chain, no project trainer / diag, no compute process on the GPU (wait, never preempt)
busy(){ pgrep -af "bash scripts/chain_" | grep -v "chain_day8_E3L5" | grep -q . || pgrep -f "^$P scripts/train_" >/dev/null || pgrep -f "^$P scripts/diag_" >/dev/null || pgrep -f "^$P scripts/owned_ceiling" >/dev/null || pgrep -f "run_e3l5_gen[.]sh" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while busy; do sleep 120; done
sleep 60; while busy; do sleep 120; done
check_shas
# ================================================================ the lease (7,200 s inclusive)
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > $OUT/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "batch": "E3-L5", "prereg": "$PREREG",
 "user_gpu_approval_utc": "$APPROVAL", "seeds": [342101, 342202], "sequential": true, "per_seed_budget_seconds": $BUDGET, "updates_prereg": $UPD_PREREG,
 "single_variable": "TRAIN cache = $CACHE5 (manifest $MAN5_SHA; v420 50 % + train_div 50 %)",
 "evidence": {"342101": "$EV_342101", "342202": "$EV_342202"},
 "control": {"342101": "$E3R/s342101", "342202": "$E3R/s342202", "holdout": "$REF5/ctrl_holdout_e3l5.json", "holdout_sha256": "$CTRLH5_SHA",
             "judged_export": "$EXPORT34", "judged_export_records_sha256": "$E5REC_SHA", "h1_records_sha256": "$H1REC_SHA", "e3l3": "$REF5/e3l3_s*",
             "e7_criteria_sha256": "$E7CRIT_SHA", "e7_summary_sha256": "$E7SUM_SHA"},
 "trainer": "scripts/g4_e3l5.py train (wraps g4_e3l3.py train)", "g4_e3l5_sha256": "$E3L5_SHA", "g4_e3l3_sha256": "$E3L3_SHA", "g4_e3l2_sha256": "$E3L2_SHA",
 "g4_e3l_sha256": "$E3L_SHA", "g4_e3r_sha256": "$E3R_SHA", "noise_panels_sha256": "$G_SHA", "pythonpath": "$PYTHONPATH", "pyroot_content_sha256": "$PYR_SHA",
 "cache": "$CACHE5", "cache_manifest_sha256": "$MAN5_SHA", "e3l5_div_sha256": "$DIV5_SHA", "e1c_fit_sha256": "$FIT_SHA", "chain_sha256": "$(sha scripts/chain_day8_E3L5.sh)",
 "renewal": "none", "resume": "none"}
JSON
log "E3-L5 lease start $START (train deadline +6600, total +7150); cache $CACHE5 ($MAN5_SHA)"
PF=$OUT/pf-s342101
timeout -k 30 600 $P -B scripts/g4_e3l5.py train --prereg-updates $UPD_PREREG --cache $CACHE5 --manifest-sha $MAN5_SHA --evidence "$EV_342101" --seed 342101 --out $PF --updates 100000 --max-updates 400 --log-every 50 --time-from 200 --probe-every 400 --snap 0 --deadline-epoch $((START+590)) --reserve 0 --expect-gpu 'RTX 5090' --ctrl-holdout $REF5/ctrl_holdout_e3l5.json > $PF.log 2>&1
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
  SECOND_NOTE="second difference (D8-76 / E3-L4 rule): $UPD_PREREG updates per seed do not fit 0.90 x the available time on the 5090 (preflight $RATE s/update on updates 200-400, load ${LOADS} s, first window ${WARM} s, probe ${PROBE} s, $BUDGET s per seed; fit $UPDFIT) -> both seeds run $UPD updates (E3-L3 rule)"
  log "E3-L5 SECOND DIFFERENCE: $SECOND_NOTE"
fi
log "E3-L5 preflight (exit $PFX): s/update $RATE (updates 200-400, window cv $CV, warning: $PFW) load ${LOADS}s first window ${WARM}s probe ${PROBE}s -> fit $UPDFIT, $UPD updates per seed (pre-registered $UPD_PREREG; same for both seeds; safety 0.90)"
$P -c "import json; json.dump(dict(preflight_exit=$PFX, rate_s_per_update='$RATE', rate_cv='$CV', rate_warning='$PFW', load_seconds='$LOADS', first_window_seconds='$WARM', probe_seconds='$PROBE', timed_updates='200-400', updates_fit=$UPDFIT, updates=$UPD, updates_prereg=$UPD_PREREG, second_difference=($UPD < $UPD_PREREG), second_difference_note='''$SECOND_NOTE''' or None), open('$OUT/updates.json', 'w'), indent=1)" 2>/dev/null || log "E3-L5 WARNING: updates.json not written"
if [ "$UPD" -lt 2000 ]; then incomplete "budget: $UPD updates < 2,000 (preflight exit $PFX)"; exit 3; fi
run_seed(){                                                                     # $1 = seed, $2 = evidence spec
  seed=$1; EV=$2; check_shas
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $(date +%s) -lt $((DE-900)) ]; do log "E3-L5: GPU busy before seed $seed; waiting"; sleep 30; done
  NOW=$(date +%s); SDT=$((NOW+BUDGET)); [ $SDT -gt $DT ] && SDT=$DT; SDE=$((SDT-30)); [ $SDE -gt $DE ] && SDE=$DE
  if [ $((SDE-NOW)) -lt 900 ]; then log "E3-L5 seed $seed: only $((SDE-NOW)) s left in the lease -> not run (the verdict will be INCOMPLETE)"; return 1; fi
  log "E3-L5 seed $seed start ($UPD updates, w learnable as E3-L3, evidence $EV, pairing $E3R/s$seed, snapshot 6000, trainer deadline $SDE, kill at $SDT)"
  timeout -k 30 $((SDT-NOW)) $P -B scripts/g4_e3l5.py train --prereg-updates $UPD_PREREG --second-note "$SECOND_NOTE" --cache $CACHE5 --manifest-sha $MAN5_SHA --evidence "$EV" --seed $seed --out $OUT/s$seed --updates $UPD --log-every 200 --time-from 200 --probe-every 1000 --snap 6000 --deadline-epoch $SDE --reserve 60 --expect-gpu 'RTX 5090' --ctrl-ckpt $E3R/s$seed/checkpoints/final.pt --ctrl-holdout $REF5/ctrl_holdout_e3l5.json > $OUT/s$seed.log 2>&1
  log "E3-L5 seed $seed done (exit $?): $(tail -1 $OUT/s$seed.log)"
  RS=$($P -c "import json; d=json.load(open('$OUT/s$seed/final.json')); r=d['rate_s_per_update']; print('%d updates (%s), %.4f s/update (%+.0f %% vs preflight), train %.0f s, total %.0f s of $BUDGET, w %s' % (d['updates_done'], 'complete' if d['complete'] else 'TRUNCATED' if d['truncated'] else 'partial', r, 100 * (r / float('$RATE') - 1), d['train_seconds'], d['seconds'], d['w']))" 2>/dev/null || echo "final.json unreadable")
  log "E3-L5 seed $seed: $RS"                                    # recorded only; the update count of the next seed is not changed
}
run_seed 342101 "$EV_342101"
run_seed 342202 "$EV_342202"
END=$(date +%s)
echo "{\"lease_end_epoch\": $END, \"lease_seconds\": $((END-START))}" > $OUT/lease_end.json
log "E3-L5 lease over after $((END-START)) s; the GPU is not used again by this chain"
# ================================================================ after the lease: CPU only (<= 10 processes; exports and refs are read, never written)
check_shas
CTRLH=$REF5/ctrl_holdout_e3l5.json
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] || { log "E3-L5 seed $s: no final.pt, no select"; continue; }
  log "E3-L5 select seed $s start (learned w + w = 4; $WORKERS workers x 1 thread, nice 10, timeout 14,400 s)"
  CUDA_VISIBLE_DEVICES= timeout -k 30 14400 nice -n 10 $P -B scripts/g4_e3l5.py select --run $OUT/s$s --ctrl $CTRLH --cache $CACHE5 --workers $WORKERS --threads 1 > $OUT/select_s$s.log 2>&1
  SX=$?; [ $SX -eq 124 ] && log "E3-L5 select seed $s TIMED OUT after 14,400 s"
  log "E3-L5 select seed $s (exit $SX): $(grep -a 'E3-L3 select\|E3-L5 select' $OUT/select_s$s.log | tail -2 | tr '\n' ' ')"
done
RUNS=""
for s in 342101 342202; do
  [ -f $OUT/s$s/checkpoints/final.pt ] && [ -f $OUT/s$s/threshold.json ] && [ -f $OUT/s$s/threshold_w4.json ] && RUNS="$RUNS,$OUT/s$s"
done
RUNS=${RUNS#,}
if [ -z "$RUNS" ]; then incomplete "no trained checkpoint with both threshold files"; exit 3; fi
log "E3-L5 eval (G4-E5 export, VAL noise seeds 3 / 4, tiers cam_nf + clean) start ($WORKERS workers x 1 thread, nice 10, timeout 43,200 s): $RUNS"
CUDA_VISIBLE_DEVICES= timeout -k 30 43200 nice -n 10 $P -B scripts/g4_e3l5.py eval --export $EXPORT34 --out $OUT/eval --runs $RUNS --e3l3 $REF5 --e3l3-criteria $REF5/criteria_w.json --e3r $E3R --expect-seeds 3,4 --panels val --tiers cam_nf,clean --workers $WORKERS --threads 1 > $OUT/eval.log 2>&1
EX=$?; [ $EX -eq 124 ] && log "E3-L5 eval TIMED OUT after 43,200 s (missing jobs -> INCOMPLETE)"
log "E3-L5 eval done (exit $EX): $(tail -1 $OUT/eval.log)"
log "E3-L5 h1 (G4-H1 VAL exports, 4 variants x 24 houses x seeds 1 / 2) start ($WORKERS workers x 1 thread, nice 10, timeout 43,200 s)"
CUDA_VISIBLE_DEVICES= timeout -k 30 43200 nice -n 10 $P -B scripts/g4_e3l5.py h1 --h1-jobs $REF5/h1_jobs --h1-export $REF5/h1_export --out $OUT/h1 --runs $RUNS --e3l3 $REF5 --e3l3-criteria $REF5/criteria_w.json --e3r $E3R --fit $REF/e1c_fit.json --workers $WORKERS --threads 1 > $OUT/h1.log 2>&1
HX=$?; [ $HX -eq 124 ] && log "E3-L5 h1 TIMED OUT after 43,200 s (missing jobs -> INCOMPLETE)"
log "E3-L5 h1 done (exit $HX): $(tail -1 $OUT/h1.log)"
CUDA_VISIBLE_DEVICES= timeout -k 30 3600 nice -n 10 $P -B scripts/g4_e3l5.py aggregate --out $OUT/verdict --runs $OUT/s342101,$OUT/s342202 --eval-out $OUT/eval --h1-out $OUT/h1 --e7-summary $REF5/e7_summary.json > $OUT/aggregate.log 2>&1
log "E3-L5 aggregate (exit $?): $(head -1 $OUT/aggregate.log)"
USE=$(du -sm $OUT | cut -f1)
log "E3-L5 disk: $OUT $USE MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "chain_day8_E3L5 finished"
