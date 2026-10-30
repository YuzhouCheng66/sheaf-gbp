#!/bin/bash
# L1b lease queue (uk-linux 5090). Runs chain_day10_L1b.sh one lease after another, each launch being its own inclusive 7,200 s lease (the
# chain never resumes or extends a lease; a job that cannot fit the remaining time is not started). Lease numbers continue L1's (11, 12, ...).
# The user approved on 2026-10-30 (~10:26 UTC, "同意，推进吧" to "B then C") at most ~4 leases for L1b after the L1 text arms; this queue is
# launched only after the L1 queue printed QUEUE_DONE (the chain also waits while any other chain runs). Stops early when the G0 gate fails
# (L1B_G0_FAIL), when a lease aborts (lock / sha problems) or when a lease started no job at all (nothing left to run).
#   usage: l1b_leases.sh <first lease> <last lease> "<user GPU approval UTC>" [chain args, e.g. --est-l1b 110]
#   launch (from the Bash tool, the script read from stdin so that no chain name is on the ssh command line):
#     ssh uk-linux bash -s <<'EOF'
#     cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/l1b_leases.sh 11 14 "<approval>" > runs/l1b/leases_11-14.out 2>&1 < /dev/null &)
#     EOF
set -uo pipefail
W=/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021
cd $W || exit 2
FIRST=$1; LAST=$2; APPROVAL=$3; shift 3
[ "$FIRST" -ge 1 ] && [ "$LAST" -ge "$FIRST" ] || { echo "bad lease range $FIRST..$LAST"; exit 2; }
mkdir -p runs/l1b
for N in $(seq "$FIRST" "$LAST"); do
  [ -f runs/l1b/L1B_G0_FAIL ] && { echo "$(date -u) queue: L1B_G0_FAIL exists -> no further lease"; break; }
  OUT=runs/chain_day10_L1b_lease$N.out
  [ -e "$OUT" ] && { echo "$(date -u) queue: $OUT exists -> lease $N was launched before; stopping (never overwrite)"; break; }
  echo "$(date -u) queue: launching lease $N"
  bash scripts/chain_day10_L1b.sh D8-85 "$APPROVAL" --lease "$N" "$@" > "$OUT" 2>&1
  X=$?
  echo "$(date -u) queue: lease $N exited $X"
  grep -q "ABORT" "$OUT" && { echo "$(date -u) queue: lease $N aborted -> stopping"; break; }
  if ! grep -q " start (estimate" "$OUT"; then echo "$(date -u) queue: lease $N started no job -> nothing left; stopping"; break; fi
  sleep 90
done
echo "$(date -u) QUEUE_DONE"
