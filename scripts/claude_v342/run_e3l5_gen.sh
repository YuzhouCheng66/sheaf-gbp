#!/bin/bash
# E3-L5 (OVERNIGHT_REPORT D8-76) CPU-only preparation on uk-linux (Claude, 2026-10-28): the new TRAIN cache cache-e3l5-div-v1
# (g4_e3l5_gen.py gen: v420 50 % + train_div 50 %, cam_nf noise seeds 200-202 + clean, the cache-e3r-v1 format and manifest), then
# verify-orig (the v420 half == cache-e3r-v1 array by array; the train_div half has the same slab, other furniture), then the control
# file of the new cache's held-out houses (g4_e3l2.py ctrlprep with the frozen E3-R checkpoints runs/e3r; = what E3-L3's select / probe
# need for THIS cache). No GPU (CUDA_VISIBLE_DEVICES= ), nice 10, 9 workers + 1 parent = 10 processes, writes only under /mnt/datasets:
#   $CACHE (new), $REF5/ctrl_holdout_e3l5.json (new), runs/e3l5_gen.log, $TMPDIR. Never overwrites: refuses if $CACHE/manifest.json or
#   the ctrl file exists. Touches no other process.
# Launch (one ssh command naming only this script):
#   timeout 60 ssh uk-linux 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/run_e3l5_gen.sh > runs/run_e3l5_gen.out 2>&1 < /dev/null &)'
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid
export TMPDIR=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude
export CUDA_VISIBLE_DEVICES=
mkdir -p $TMPDIR
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python
CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3l5-div-v1
REFCACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1
REF5=/mnt/datasets/yuzhou-sheaf-gbp-research/e3l5-ref
E3R=runs/e3r
LOG=runs/e3l5_gen.log
WORKERS=9
E3R_SHA=1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970     # g4_e3r.py
E3L2_SHA=bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314    # g4_e3l2.py (ctrlprep)
E3L_SHA=3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be     # g4_e3l.py
G_SHA=2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f       # g4_noise_panels.py
D_SHA=4793a3279ebd9564315500721e548ab9dd0e10d8f192df632c9b8ddf2964b61d       # mapping_3d_storey_data_v420.py
OOD_SHA=9e1c7c2aac5ddd3e53ed5035a3db707d87c1a1026987ece782796b39f4e1f92c     # mapping_3d_storey_data_v420ood.py (G4-H1)
V365_SHA=e1e6b5aab5105190b0b4e2f7cf40c42ac71504867ac84ecf789634787aa68b91
V371_SHA=6abdc221f01c323b968409b1281d0f1c6d2b03526901419088cd33aea915e009
IDX_SHA=7b4730cc52689c9044228a6f9bf8e1b4785cb5f6f40f54a422e603c6000101c1
MAN_E3R_SHA=14e90b409a04ff5bf64484b3d8035dd964a787494c16cdda80b849a9e44b7e57 # cache-e3r-v1/manifest.json (house set, verify-orig reference)
DIV_SHA=$1                                                                  # mapping_3d_storey_data_v420div.py as deployed (argument)
GEN_SHA=$2                                                                  # g4_e3l5_gen.py as deployed (argument)
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
sha(){ sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
log "run_e3l5_gen started pid $$ host $(hostname) sha256 $(sha scripts/run_e3l5_gen.sh | cut -c1-16)"
for pair in "g4_e3r.py $E3R_SHA" "g4_e3l2.py $E3L2_SHA" "g4_e3l.py $E3L_SHA" "g4_noise_panels.py $G_SHA" "mapping_3d_storey_data_v420.py $D_SHA" \
            "mapping_3d_storey_data_v420ood.py $OOD_SHA" "mapping_houseexpo_data_v365.py $V365_SHA" "mapping_houseexpo_data_v371.py $V371_SHA" \
            "mapping_3d_storey_data_v420div.py $DIV_SHA" "g4_e3l5_gen.py $GEN_SHA"; do
  set -- $pair
  [ "$(sha scripts/$1)" = "$2" ] || { log "ABORT: scripts/$1 sha $(sha scripts/$1) != $2"; exit 2; }
done
[ "$(sha $HOUSEEXPO_INDEX)" = "$IDX_SHA" ] || { log "ABORT: HouseExpo index sha changed"; exit 2; }
[ "$(sha $REFCACHE/manifest.json)" = "$MAN_E3R_SHA" ] || { log "ABORT: cache-e3r-v1 manifest sha changed"; exit 2; }
[ -e $CACHE/manifest.json ] && { log "ABORT: $CACHE/manifest.json exists (never overwrite)"; exit 2; }
[ -e $REF5/ctrl_holdout_e3l5.json ] && { log "ABORT: $REF5/ctrl_holdout_e3l5.json exists (never overwrite)"; exit 2; }
FREE_MB=$(df -Pm /mnt/datasets | awk 'NR==2{print $4}'); [ "$FREE_MB" -lt 5000 ] && { log "ABORT: only ${FREE_MB} MB free"; exit 2; }
mkdir -p $CACHE $REF5
log "E3-L5 gen start: $CACHE, $WORKERS workers, nice 10, CPU only"
timeout -k 30 21600 nice -n 10 $P -B scripts/g4_e3l5_gen.py gen --cache $CACHE --workers $WORKERS > runs/e3l5_gen_detail.log 2>&1
GX=$?
log "E3-L5 gen done (exit $GX): $(grep -a 'manifest ' runs/e3l5_gen_detail.log | tail -1)"
grep -a -A12 '^E3-L5 cache' runs/e3l5_gen_detail.log | tee -a $LOG
[ $GX -eq 0 ] || { log "E3-L5 gen FAILED its checks (exit $GX): the cache must not be used"; exit 3; }
MAN5=$(sha $CACHE/manifest.json)
log "E3-L5 manifest sha256 $MAN5; e3l5_div.json sha256 $(sha $CACHE/e3l5_div.json)"
timeout -k 30 7200 nice -n 10 $P -B scripts/g4_e3l5_gen.py verify-orig --cache $CACHE --ref $REFCACHE > runs/e3l5_verify_orig.log 2>&1
VX=$?
log "E3-L5 verify-orig (exit $VX): $(tail -1 runs/e3l5_verify_orig.log)"
[ $VX -eq 0 ] || { log "E3-L5 verify-orig FAILED: the cache must not be used"; exit 3; }
log "E3-L5 ctrlprep start (g4_e3l2.py ctrlprep, frozen E3-R checkpoints $E3R, $WORKERS workers)"
timeout -k 30 10800 nice -n 10 $P -B scripts/g4_e3l2.py ctrlprep --cache $CACHE --e3r $E3R --out $REF5/ctrl_holdout_e3l5.json --manifest-sha $MAN5 --workers $WORKERS --threads 1 > runs/e3l5_ctrlprep.log 2>&1
CX=$?
log "E3-L5 ctrlprep done (exit $CX): $(tail -1 runs/e3l5_ctrlprep.log | cut -c1-400)"
[ $CX -eq 0 ] || { log "E3-L5 ctrlprep FAILED"; exit 3; }
log "E3-L5 ctrl_holdout_e3l5.json sha256 $(sha $REF5/ctrl_holdout_e3l5.json); cache $(du -sm $CACHE | cut -f1) MB; /mnt/datasets free $(df -Pm /mnt/datasets | awk 'NR==2{print $4}') MB"
log "run_e3l5_gen finished"
