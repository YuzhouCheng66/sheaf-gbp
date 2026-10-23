#!/bin/bash
# Day-2 Phase D, revision 2 (Claude, 2026-10-22 17:30 UTC; identical to chain_day2_D.sh except that the Phase-C deadline fallback also honours the
# H256 relaunch lease, so Phase D cannot start while the relaunched H256 is still training). Runs after Phase C (v356 H256 + dynft-s342101).
#   v359  k-bit second seed + decoupled mu/q bits (literature-round review D2/top-1): seed 342202, max fusion, posterior_only,
#         train_v348_kbit.py UNCHANGED (same tie-break as the seed-342101 batch, so the two seeds pair cleanly), 36 epochs, arms:
#           b8 (mu8,q8)  b6 (mu6,q6)  m6q4 (mu6,q4)  m6q5 (mu6,q5)
#         GATE (pre-registered): a configuration is admissible if on BOTH seeds (where available) K0 and R8 wall F1 >= b8 - 1pp and
#         FPR <= b8 + 0.3pp; the paired per-scene bootstrap (paired_stats_v342.py) is reported for b6 - b8 on both seeds.
#         Expectation stated in advance: q at 4-5 bits is free (H(q) = 4.57 bits measured on the 6-bit model); mu below 6 bits is not.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day2.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day2_D2 started pid $$ sha256 $(sha256sum scripts/chain_day2_D2.sh | cut -c1-16)"
while [ ! -f runs/v356-h256-max/lease.json ]; do sleep 120; done
while true; do
  done_c=1
  [ -f runs/v356-h256-max/h256-s342202/final.json ] || done_c=0
  if [ -f runs/v350-dynamic-ft/dynft-s342101.log ]; then [ -f runs/v349-dynamic/dynft-s342101.json ] || done_c=0; fi
  [ $done_c = 1 ] && break
  DTC=$(python3 -c "import json;d=json.load(open('runs/v356-h256-max/lease.json'));print(int(max(d['total_deadline_epoch'], d.get('relaunch',{}).get('total_deadline_epoch',0))))")
  if [ $(date +%s) -gt $((DTC+300)) ]; then log "phase C total deadline passed; continuing"; break; fi
  sleep 60
done
log "phase C finished; phase D start"
mkdir -p runs/v359-kbit-seed2
START=$(date +%s); DE=$((START+5400)); DT=$((START+7000))
cat > runs/v359-kbit-seed2/lease.json <<JSON
{"lease_start_epoch": $START, "train_deadline_epoch": $DE, "total_deadline_epoch": $DT, "inclusive_seconds": 7200, "epochs": 36, "size": 128, "seed": 342202,
 "fusion": "max", "readout": "posterior_only", "wire": "kbit (floor codebook), trainer unchanged from v348 for clean pairing with seed 342101",
 "arms": {"b8": [8, 8], "b6": [6, 6], "m6q4": [6, 4], "m6q5": [6, 5]},
 "gate": "admissible iff on both seeds K0 and R8 wall F1 >= b8 - 1pp and FPR <= b8 + 0.3pp; paired bootstrap b6 - b8 reported on both seeds",
 "prior_expectation": "q at 4-5 bits free (H(q)=4.57 bits on the 6-bit model); mu below 6 bits not free",
 "trainer_sha256": "$(sha256sum scripts/train_v348_kbit.py | cut -d' ' -f1)"}
JSON
for arm in "b8 8 8" "b6 6 6" "m6q4 6 4" "m6q5 6 5"; do
  set -- $arm
  nohup $P scripts/train_v348_kbit.py --source $S --out runs/v359-kbit-seed2/$1 --arm owned --size 128 --device cuda --epochs 36 --val-every 6 --seed 342202 --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --wire kbit --mu-bits $2 --q-bits $3 > runs/v359-kbit-seed2/$1.log 2>&1 &
  log "$1 pid $!"
done
log "phase D launched"
