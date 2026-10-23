#!/bin/bash
# Day-3 queue F/G (Claude, written 2026-10-22 22:35 UTC before any of these runs started). User approved the priorities on 22:22.
# F1  v362 H256 dynamic fine-tune (seed 342202): train_v361.py (= v350 + host-side caching of dynamic slices, patch F; no maths change),
#     warm start runs/v356-h256-max/h256-s342202/checkpoints/final.pt, --dynamic --size 256, 6 dynamic epochs (21,312 updates), lr 1e-4,
#     max, posterior_only, sealed wire. Same recipe that passed at H128 (v360).
#     GATE (pre-registered): static VALIDATION K0 and R8 >= h256 - 1pp AND dynamic (size 256, radio 8, idx 0-7) own K0 t=5 and crdt t=5
#     >= h256 zero-shot + 3pp. Reported regardless: fresh panel, doors/topology.
# F2  v363 third seed (342303) of the final H128 recipe, step 1: 72-epoch static (train_v347_maxsel.py, identical to v354).
# G1  v363 step 2: dynamic fine-tune of the seed-3 static endpoint (train_v361.py at 128, identical recipe to v360 except the patch-F
#     data placement). No gate: replication; report mean and sd over the three seeds for static, fresh, dynamic and door metrics.
# G2  evaluations: zero-shot dynamic of h256, dynamic of h256-dynft (also radio 5 and fresh idx 8-15), door/topology for all new models.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day3_FG started pid $$ sha256 $(sha256sum scripts/chain_day3_FG.sh | cut -c1-16)"
mkdir -p runs/v362-h256-dynft runs/v363-seed3
START=$(date +%s); DE=$((START+6000)); DT=$((START+7100))
cat > runs/v362-h256-dynft/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 256, "seed": 342202, "dynamic_epochs": 6, "updates_planned": 21312,
 "warm_start": "runs/v356-h256-max/h256-s342202/checkpoints/final.pt", "lr": 1e-4, "fusion": "max", "readout": "posterior_only", "wire": "sealed",
 "trainer": "scripts/train_v361.py", "trainer_sha256": "$(sha256sum scripts/train_v361.py | cut -d' ' -f1)",
 "gate": "static VAL K0,R8 >= h256 - 1pp AND dynamic(size256, radio8, idx0-7) own K0 t5 and crdt t5 >= h256 zero-shot + 3pp",
 "shared_lease_with": ["runs/v363-seed3/long-s342303"]}
JSON
cat > runs/v363-seed3/lease_long.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "size": 128, "seed": 342303, "epochs": 72,
 "recipe": "identical to runs/v354-endpoint-long (train_v347_maxsel.py, max, posterior_only, sealed, lr 3e-4 default, cosine)", "trainer_sha256": "$(sha256sum scripts/train_v347_maxsel.py | cut -d' ' -f1)",
 "purpose": "third seed for the final recipe (replication; no gate)", "shared_lease_with": ["runs/v362-h256-dynft/dynft256-s342202"]}
JSON
nohup $P scripts/train_v361.py --source $S --out runs/v362-h256-dynft/dynft256-s342202 --arm owned --size 256 --device cuda --dynamic --fusion max --readout posterior_only --init-from runs/v356-h256-max/h256-s342202/checkpoints/final.pt --epochs 6 --lr 1e-4 --val-every 2 --seed 342202 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 > runs/v362-h256-dynft/dynft256-s342202.log 2>&1 &
log "dynft256-s342202 pid $!"
nohup $P scripts/train_v347_maxsel.py --source $S --out runs/v363-seed3/long-s342303 --arm owned --size 128 --device cuda --epochs 72 --val-every 6 --seed 342303 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max > runs/v363-seed3/long-s342303.log 2>&1 &
log "long-s342303 pid $!"
sleep 60
while pgrep -f "runs/v362-h256-dynft/dynft256" >/dev/null || pgrep -f "runs/v363-seed3/long-s342303" >/dev/null; do sleep 60; done
log "phase F trainers finished"
# ---------------- phase G
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
# zero-shot dynamic reference for the H256 gate (moved here: nothing else may share the card with the two phase-F trainers)
$P scripts/eval_dynamic_v349.py --source $S --checkpoint runs/v356-h256-max/h256-s342202/checkpoints/final.pt --trainer scripts/train_v361.py --size 256 --out runs/v349-dynamic/zeroshot-h256-s342202.json --device cuda > runs/v349-dynamic/zeroshot-h256-s342202.log 2>&1
log "zero-shot dynamic eval of h256 done"
CK=runs/v362-h256-dynft/dynft256-s342202/checkpoints/final.pt
if [ -f $CK ]; then
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --out runs/v349-dynamic/dynft256-s342202.json --device cuda > runs/v349-dynamic/dynft256-s342202.log 2>&1
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --radio 5 --out runs/v349-dynamic/dynft256-s342202-radio5.json --device cuda > runs/v349-dynamic/dynft256-s342202-radio5.log 2>&1
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --indices 8-15 --out runs/v349-dynamic/dynft256-s342202-fresh8-15.json --device cuda > runs/v349-dynamic/dynft256-s342202-fresh8-15.log 2>&1
  $P scripts/eval_threshold_v347.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --out runs/v355-threshold/tau-dynft256-s342202.json --device cuda --taus 0.5 > runs/v355-threshold/tau-dynft256-s342202.log 2>&1
  log "h256 dynft evaluations done"
fi
$P scripts/eval_threshold_v347.py --source $S --checkpoint runs/v356-h256-max/h256-s342202/checkpoints/final.pt --trainer scripts/train_v361.py --size 256 --out runs/v355-threshold/tau-h256-s342202.json --device cuda --taus 0.5 > runs/v355-threshold/tau-h256-s342202.log 2>&1
sleep 60
while pgrep -f "runs/v363-seed3/ldynft-s342303" >/dev/null; do sleep 60; done
CK=runs/v363-seed3/ldynft-s342303/checkpoints/final.pt
if [ -f $CK ]; then
  $P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --out runs/v349-dynamic/ldynft-s342303.json --device cuda > runs/v349-dynamic/ldynft-s342303.log 2>&1
  $P scripts/eval_threshold_v347.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --out runs/v355-threshold/tau-ldynft-s342303.json --device cuda --taus 0.5 > runs/v355-threshold/tau-ldynft-s342303.log 2>&1
fi
log "phase G done"
