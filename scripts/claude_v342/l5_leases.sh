#!/bin/bash
# L5 lease queue (uk-linux 5090). Runs chain_day20_L5.sh one lease after another, each launch being its own inclusive 7,200 s lease (the chain
# never resumes or extends a lease; a job that cannot fit the remaining time is not started). Lease numbers 53-56 = the job table of D8-95
# section 4 (a lease whose condition is not met is skipped by the chain in seconds and the queue moves on). Stops early only when a lease
# ABORTS (lock / sha / placeholder problems, exit 2).
# GPU approval: the user's "批准，做" (2026-11-03, D8-95 leases 53-56), passed as $3 into every lease json.
#   usage: l5_leases.sh <first lease> <last lease> "<user GPU approval UTC>" [chain args]
#     cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/l5_leases.sh 53 56 "<approval>" > runs/l5/leases_53-56.out 2>&1 < /dev/null &)
cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1
FIRST="$1"; LAST="$2"; APPROVAL="$3"; shift 3
case "$FIRST$LAST" in ''|*[!0-9]*) echo "usage: l5_leases.sh <first> <last> \"<approval>\" [chain args]"; exit 2;; esac
[ -n "$APPROVAL" ] || { echo "the user's GPU approval text is required"; exit 2; }
mkdir -p runs/l5
for N in $(seq "$FIRST" "$LAST"); do
  OUT=runs/chain_day20_L5_lease$N.out
  [ -e "$OUT" ] && { echo "$(date -u) queue: $OUT exists -> lease $N was launched before; stopping (never relaunch a lease number)"; break; }
  echo "$(date -u) queue: launching lease $N"
  bash scripts/chain_day20_L5.sh D8-95 "$APPROVAL" --lease "$N" "$@" > "$OUT" 2>&1
  X=$?
  echo "$(date -u) queue: lease $N exited $X"
  [ $X -eq 2 ] && { echo "$(date -u) queue: lease $N ABORTED (exit 2) -> stopping; the coordinator decides"; break; }
  sleep 30
done
echo "$(date -u) queue: done" | tee -a runs/l5/QUEUE_DONE
