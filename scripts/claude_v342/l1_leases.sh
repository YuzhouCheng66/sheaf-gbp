#!/bin/bash
# L1 lease queue (uk-linux 5090). Runs chain_day9_L1.sh one lease after another, each launch being its own inclusive
# 7,200 s lease (the chain never resumes or extends a lease; a job that cannot fit the remaining time is not started).
# The user approved on 2026-10-29 (~18:42 UTC, "5090+bigboy，可换模型") back-to-back leases until the morning, at most
# the number given here as the last lease. Stops early when stage 0 fails (STAGE0_FAIL), when a lease aborts (lock /
# sha problems) or when a lease started no job at all (nothing left to run).
#   usage: l1_leases.sh <first lease> <last lease> "<user GPU approval UTC>" [chain args, e.g. --est-latent 90]
#   launch: timeout 60 ssh uk-linux 'cd $W && (setsid nohup bash scripts/l1_leases.sh 1 6 "<approval>" > runs/l1/leases.out 2>&1 < /dev/null &)'
set -uo pipefail
W=/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021
cd $W || exit 2
FIRST=$1; LAST=$2; APPROVAL=$3; shift 3
[ "$FIRST" -ge 1 ] && [ "$LAST" -ge "$FIRST" ] || { echo "bad lease range $FIRST..$LAST"; exit 2; }
for N in $(seq "$FIRST" "$LAST"); do
  [ -f runs/l1/STAGE0_FAIL ] && { echo "$(date -u) queue: STAGE0_FAIL exists -> no further lease"; break; }
  OUT=runs/chain_day9_L1_lease$N.out
  [ -e "$OUT" ] && { echo "$(date -u) queue: $OUT exists -> lease $N was launched before; stopping (never overwrite)"; break; }
  echo "$(date -u) queue: launching lease $N"
  bash scripts/chain_day9_L1.sh D8-82 "$APPROVAL" --lease "$N" "$@" > "$OUT" 2>&1
  X=$?
  echo "$(date -u) queue: lease $N exited $X"
  grep -q "ABORT" "$OUT" && { echo "$(date -u) queue: lease $N aborted -> stopping"; break; }
  if ! grep -q " start (estimate" "$OUT"; then echo "$(date -u) queue: lease $N started no job -> nothing left; stopping"; break; fi
  sleep 90
done
echo "$(date -u) QUEUE_DONE"
