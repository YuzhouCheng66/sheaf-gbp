#!/bin/bash
# L1f lease queue (uk-linux 5090). Runs chain_day14_L1f.sh one lease after another, each launch being its own inclusive 7,200 s lease (the
# chain never resumes or extends a lease; a job that cannot fit the remaining time is not started). Lease number = the last of the L1d approval (18).
# The user approved L1d on 2026-10-31 ("做 L1d", at most 3 leases, 16-18) and on 2026-11-01 "先A后B": A = L1f (D8-89) on lease 18. Stops early when
# the G1f gate fails (L1F_G1_FAIL), when a lease aborts (lock / sha problems) or when a lease started no job at all (nothing left to run).
#   usage: l1f_leases.sh <first lease> <last lease> "<user GPU approval UTC>" [chain args, e.g. --est-l1f 55]
#   launch (from the Bash tool, the script read from stdin so that no chain name is on the ssh command line):
#     ssh uk-linux bash -s <<'EOF'
#     cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 && (setsid nohup bash scripts/l1f_leases.sh 18 18 "<approval>" > runs/l1f/leases_18.out 2>&1 < /dev/null &)
#     EOF
set -uo pipefail
W=/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021
cd $W || exit 2
FIRST=$1; LAST=$2; APPROVAL=$3; shift 3
[ "$FIRST" -ge 1 ] && [ "$LAST" -ge "$FIRST" ] || { echo "bad lease range $FIRST..$LAST"; exit 2; }
mkdir -p runs/l1f
for N in $(seq "$FIRST" "$LAST"); do
  [ -f runs/l1f/L1F_G1_FAIL ] && { echo "$(date -u) queue: L1F_G1_FAIL exists -> no further lease"; break; }
  OUT=runs/chain_day14_L1f_lease$N.out
  [ -e "$OUT" ] && { echo "$(date -u) queue: $OUT exists -> lease $N was launched before; stopping (never overwrite)"; break; }
  echo "$(date -u) queue: launching lease $N"
  bash scripts/chain_day14_L1f.sh D8-89 "$APPROVAL" --lease "$N" "$@" > "$OUT" 2>&1
  X=$?
  echo "$(date -u) queue: lease $N exited $X"
  grep -q "ABORT" "$OUT" && { echo "$(date -u) queue: lease $N aborted -> stopping"; break; }
  if ! grep -q " start (estimate" "$OUT"; then echo "$(date -u) queue: lease $N started no job -> nothing left; stopping"; break; fi
  sleep 90
done
echo "$(date -u) QUEUE_DONE"
