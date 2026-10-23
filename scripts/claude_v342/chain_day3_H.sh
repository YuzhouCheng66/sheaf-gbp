#!/bin/bash
# Day-3 batch H (Claude, written 2026-10-22 21:45 UTC, before any of these runs started). Starts when the H256 dynamic fine-tune
# (runs/v362-h256-dynft/dynft256-s342202) has finished. One 7200 s lease for three 128 trainers + light zero-shot evaluations.
#  H1 v363 seed 3 (342303), step 1: 72-epoch static run, identical to v354 (train_v347_maxsel.py). Replaces the co-scheduled attempt
#     that was stopped at epoch 17 (ABORTED-coschedule-long-s342303). Step 2 (dynamic fine-tune) follows in the next lease.
#  H2 v366 HouseExpo zero-shot transfer: the v293-trained final models (ldynft-s342101/202, long-s342202) evaluated on HouseExpo v365
#     VALIDATION (48 scenes) and fresh panel, via train_v365.py --world houseexpo --epochs 0 (same forward path as training).
#  H3 v367 HouseExpo in-domain training from scratch, two seeds (342101, 342202): train_v365.py --world houseexpo, the v354 static
#     recipe unchanged (72 epochs, lr 3e-4 cosine, max, posterior_only, sealed wire). ONE change vs v354: the world family.
#     No pass/fail gate (first measurement on this family). Stated expectation: in-domain R8 within 5 pp of the v293 in-domain .924/.938;
#     zero-shot transfer clearly below in-domain. Cross-domain (HouseExpo-trained on v293 VALIDATION) evaluated afterwards.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
export HOUSEEXPO_JSON_DIR=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json HOUSEEXPO_INDEX=/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day3_H started pid $$ sha256 $(sha256sum scripts/chain_day3_H.sh | cut -c1-16)"
sleep 30
while pgrep -f "runs/v362-h256-dynft/dynft256" >/dev/null; do sleep 60; done
log "H256 dynft finished; batch H start"
mkdir -p runs/v363-seed3 runs/v366-hx-zeroshot runs/v367-hx-train
START=$(date +%s); DE=$((START+6600)); DT=$((START+7150))
cat > runs/v367-hx-train/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "epochs": 72, "seeds": [342101, 342202],
 "world": "houseexpo v365 (index sha256 $(sha256sum /mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json | cut -d' ' -f1), data module sha256 $(sha256sum scripts/mapping_houseexpo_data_v365.py | cut -d' ' -f1))",
 "recipe": "v354 static recipe unchanged (max, posterior_only, sealed, lr 3e-4 cosine, 72 epochs); one change: world family", "trainer_sha256": "$(sha256sum scripts/train_v365.py | cut -d' ' -f1)",
 "train_scenes": "600 distinct TRAIN houses (200 per room stratum, N alternating 8/16)", "gate": "none (first measurement); expectation stated in chain_day3_H.sh",
 "shared_lease_with": ["runs/v363-seed3/long-s342303", "runs/v366-hx-zeroshot"]}
JSON
cat > runs/v363-seed3/lease_long.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "seed": 342303, "epochs": 72,
 "recipe": "identical to runs/v354-endpoint-long (train_v347_maxsel.py)", "trainer_sha256": "$(sha256sum scripts/train_v347_maxsel.py | cut -d' ' -f1)",
 "note": "relaunch of the co-scheduled attempt stopped at epoch 17 (ABORTED-coschedule-long-s342303); fresh lease, from scratch", "shared_lease_with": ["runs/v367-hx-train"]}
JSON
nohup $P scripts/train_v347_maxsel.py --source $S --out runs/v363-seed3/long-s342303 --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed 342303 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v363-seed3/long-s342303.log 2>&1 &
log "long-s342303 pid $!"
for seed in 342101 342202; do
  nohup $P scripts/train_v365.py --source $S --world houseexpo --out runs/v367-hx-train/hx-long-s$seed --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v367-hx-train/hx-long-s$seed.log 2>&1 &
  log "hx-long-s$seed pid $!"
done
# H2 zero-shot transfer (light, sequential)
for m in v360-long-dynft/ldynft-s342101 v360-long-dynft/ldynft-s342202 v354-endpoint-long/long-s342202; do
  name=$(echo $m | tr '/' '_')
  $P scripts/train_v365.py --source $S --world houseexpo --out runs/v366-hx-zeroshot/$name --arm owned --size 128 --device cuda --epochs 0 --seed 342202 --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --init-from runs/$m/checkpoints/final.pt > runs/v366-hx-zeroshot/$name.log 2>&1
  log "zero-shot HouseExpo eval of $m done"
done
sleep 60
while pgrep -f "runs/v367-hx-train/hx-long" >/dev/null || pgrep -f "runs/v363-seed3/long-s342303" >/dev/null; do sleep 60; done
log "batch H trainers finished"
# ---------------- batch I: seed-3 dynamic fine-tune + cross-domain evaluation of the HouseExpo models
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v363-seed3/lease_dynft.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "seed": 342303, "dynamic_epochs": 6,
 "warm_start": "runs/v363-seed3/long-s342303/checkpoints/final.pt", "lr": 1e-4, "trainer": "scripts/train_v361.py", "trainer_sha256": "$(sha256sum scripts/train_v361.py | cut -d' ' -f1)",
 "purpose": "third seed, step 2 (replication of v360)"}
JSON
if [ -f runs/v363-seed3/long-s342303/final.json ]; then
  nohup $P scripts/train_v361.py --source $S --out runs/v363-seed3/ldynft-s342303 --arm owned --size 128 --device cuda --dynamic --fusion max --readout posterior_only --init-from runs/v363-seed3/long-s342303/checkpoints/final.pt --epochs 6 --lr 1e-4 --val-every 2 --seed 342303 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v363-seed3/ldynft-s342303.log 2>&1 &
  log "ldynft-s342303 pid $!"
fi
for seed in 342101 342202; do
  CK=runs/v367-hx-train/hx-long-s$seed/checkpoints/final.pt
  [ -f $CK ] && $P scripts/train_v365.py --source $S --world v293 --out runs/v366-hx-zeroshot/cross_hx-long-s${seed}_on_v293 --arm owned --size 128 --device cuda --epochs 0 --seed $seed --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --init-from $CK > runs/v366-hx-zeroshot/cross_hx-long-s${seed}_on_v293.log 2>&1 && log "cross-domain eval of hx-long-s$seed on v293 done"
done
sleep 60
while pgrep -f "runs/v363-seed3/ldynft-s342303" >/dev/null; do sleep 60; done
CK=runs/v363-seed3/ldynft-s342303/checkpoints/final.pt
if [ -f $CK ]; then
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --out runs/v349-dynamic/ldynft-s342303.json --device cuda > runs/v349-dynamic/ldynft-s342303.log 2>&1
  $P scripts/eval_threshold_v347.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --out runs/v355-threshold/tau-ldynft-s342303.json --device cuda --taus 0.5 > runs/v355-threshold/tau-ldynft-s342303.log 2>&1
  $P scripts/train_v365.py --source $S --world houseexpo --out runs/v366-hx-zeroshot/v363-seed3_ldynft-s342303 --arm owned --size 128 --device cuda --epochs 0 --seed 342303 --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --init-from $CK > runs/v366-hx-zeroshot/v363-seed3_ldynft-s342303.log 2>&1
fi
log "batch I done"
