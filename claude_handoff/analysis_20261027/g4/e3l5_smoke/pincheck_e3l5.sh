#!/bin/bash
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
PYROOT=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
REF=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-ref                       # e1c_fit.json = the evidence tables A / B (as E3-R .. E3-L4)
REF4=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l4-ref                     # G4-E5 export (e5_export/), as E3-L4
REF5=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l5-ref                     # ctrl_holdout_e3l5.json, e3l3_s*/ (frozen E3-L3), criteria_w.json + e7_summary.json (G4-E7), h1_jobs/ + h1_export/ (G4-H1 VAL)
CACHE5=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3l5-div-v1         # NEW TRAIN cache (the single variable), pinned by manifest sha
EXPORT34=$REF4/e5_export                                                 # G4-E5 export, VAL / fresh x noise seeds 3, 4: READ ONLY
E3R=runs/e3r                                                             # E3-R checkpoints + thresholds (control NET; ctrl-ckpt pairing)
OUT=runs/e3l5
LOG=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude/e3l5_pincheck.log
E3L5_SHA=d055eca77b799ad210504fc51a79ae12a11faba9291bd44023bdc1502b543e2a
MAN5_SHA=80b8cdc0bb6397ca1d2054e14d1f52fd1addf92b68c12c2e92aef56b52796434
DIV5_SHA=7043a5aa2a95d8bc0dfc4ae6391e0959151d374476a9de35ba708e5a85d19da6
VER5_SHA=2bd3d8da2c81852ec7da1b35727de7fbf634eaaa3a6b21df53e0951de625ae2c
CTRLH5_SHA=2f7fbe1d3b4843427b13c8fa76fdb854d657bdec2a9f089ea724b37977b7f8ef
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
PF=$OUT/pf-s342101
PFX=$?
SECOND_NOTE=""
CTRLH=$REF5/ctrl_holdout_e3l5.json
RUNS=""
RUNS=${RUNS#,}
check_shas && echo "ALL PINS OK (read-only check; nothing launched)"
