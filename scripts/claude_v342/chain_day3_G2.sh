#!/bin/bash
# Day-3 phase G, revision 2 (Claude, 2026-10-22 ~22:55 UTC). Replaces phase G of chain_day3_FG.sh, whose waiter was stopped before it
# launched anything: its pgrep pattern also matched the RELAUNCHED seed-3 run of batch H, which would have made FG and batch I both
# launch ldynft-s342303 and write the same evaluation files. Seed 3 is now handled only by batch I (chain_day3_H.sh).
# This script runs the H256 evaluations of phase G (unchanged commands), light, sequential, now.
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python; S=/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
LOG=runs/chain_day3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
log "chain_day3_G2 started pid $$ sha256 $(sha256sum scripts/chain_day3_G2.sh | cut -c1-16)"
$P scripts/eval_dynamic_v349.py --source $S --checkpoint runs/v356-h256-max/h256-s342202/checkpoints/final.pt --trainer scripts/train_v361.py --size 256 --out runs/v349-dynamic/zeroshot-h256-s342202.json --device cuda > runs/v349-dynamic/zeroshot-h256-s342202.log 2>&1
log "zero-shot dynamic eval of h256 done"
CK=runs/v362-h256-dynft/dynft256-s342202/checkpoints/final.pt
$P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --out runs/v349-dynamic/dynft256-s342202.json --device cuda > runs/v349-dynamic/dynft256-s342202.log 2>&1
$P scripts/eval_dynamic_v349.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --radio 5 --out runs/v349-dynamic/dynft256-s342202-radio5.json --device cuda > runs/v349-dynamic/dynft256-s342202-radio5.log 2>&1
$P scripts/eval_threshold_v347.py --source $S --checkpoint $CK --trainer scripts/train_v361.py --size 256 --out runs/v355-threshold/tau-dynft256-s342202.json --device cuda --taus 0.5 > runs/v355-threshold/tau-dynft256-s342202.log 2>&1
$P scripts/eval_threshold_v347.py --source $S --checkpoint runs/v356-h256-max/h256-s342202/checkpoints/final.pt --trainer scripts/train_v361.py --size 256 --out runs/v355-threshold/tau-h256-s342202.json --device cuda --taus 0.5 > runs/v355-threshold/tau-h256-s342202.log 2>&1
log "phase G2 (H256 evaluations) done"
