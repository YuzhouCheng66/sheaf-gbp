"""Build a read-only pin check from chain_day8_E3L5.sh: the variable lines + sha helpers + check_shas, with the FILL_ values substituted and the
log redirected to tmp-claude (never runs.../chain_day8.log). It only reads files and prints; it never locks, launches or touches a GPU."""
import re
import sys

chain, out = sys.argv[1], sys.argv[2]
fill = dict(E3L5_SHA=sys.argv[3], MAN5_SHA=sys.argv[4], DIV5_SHA=sys.argv[5], VER5_SHA=sys.argv[6], CTRLH5_SHA=sys.argv[7])
lines = open(chain, encoding='utf-8').read().splitlines()
keep = ['#!/bin/bash', 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021 || exit 1']
i = 0
while i < len(lines):
    l = lines[i]
    # only pure assignments (no command, no command substitution): a CPU-stage line 'CUDA_VISIBLE_DEVICES= timeout ...' must never be copied
    if (l.startswith('export ') and '$(' not in l) or re.match(r'^[A-Z0-9_]+=("[^"]*"|[^\s(]*)\s*(#.*)?$', l):
        m = re.match(r'^([A-Z0-9_]+)=', l)
        if m and m.group(1) in fill:
            l = f'{m.group(1)}={fill[m.group(1)]}'
        if m and m.group(1) == 'LOG':
            l = 'LOG=/mnt/datasets/yuzhou-sheaf-gbp-research/tmp-claude/e3l5_pincheck.log'
        if not l.startswith('START='):
            keep.append(l)
    elif l.startswith(('log(){', 'sha(){', 'pyroot_sha(){', 'rec_sha(){')):
        keep.append(l)
    elif l.startswith('check_shas(){'):
        while True:
            keep.append(lines[i])
            if lines[i] == '}':
                break
            i += 1
    i += 1
keep += ['check_shas && echo "ALL PINS OK (read-only check; nothing launched)"']
open(out, 'w', newline='\n').write('\n'.join(keep) + '\n')
