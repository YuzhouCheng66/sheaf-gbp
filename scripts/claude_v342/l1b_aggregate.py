"""l1b_aggregate.py (Claude, 2026-10-30): the L1b verdict (pre-registration D8-85 section 5) from summary.json files.
Reads B0-hetero-s<seed> and T-hetero-s<seed> from the L1 runs dir (--l1; the L1 runs are reused, not retrained) and S1b-hetero-s<seed>,
Ub-hetero-s<seed> from the L1b runs dir (--l1b). VAL EM (percentage points, mean over agents) decides; fresh only for the degrade rule.
  G0  (channel gate, per seed)  S1b - B0 >= 3.0 pt        P1b  S1b - T >= 2.0 pt        P2b  S1b - Ub >= 1.0 pt       (both seeds must pass)
  degrade: (S1b - T) on fresh lower than on VAL by more than 1.0 pt on either seed caps the verdict at AMBIGUOUS
  verdict: S1b missing / C assertion false / unit problems -> INCOMPLETE; G0 fails on the first seed -> CHANNEL-NO (the second seed is not
  needed); G0 fails on the second seed -> CHANNEL-NO; then T and Ub both seeds needed (else INCOMPLETE); P2b fails -> NO; P1b fails ->
  AMBIGUOUS; all pass -> LEVER-pending-P3b (S1L1b not run in L1b).
Prints one line 'G0(s<seed>): PASS | FAIL | INCOMPLETE' per seed (--gate <seed>: only that seed; the chain's gate reads it) and
'VERDICT: ...'; writes <out>.json and <out>.md. Always exits 0.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time

SEEDS = (342101, 342202)
G0_PT, P1B_PT, P2B_PT, DEGRADE_PT = 3.0, 2.0, 1.0, 1.0
PAIR = 'hetero'
L1_ARMS, L1B_ARMS = ('B0', 'T'), ('S1b', 'Ub')
ARM_DIR = {'B0': 'l1', 'T': 'l1', 'S1b': 'l1b', 'Ub': 'l1b'}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def read_run(root, arm, seed):
    """-> dict(path, found, val_em, fresh_em, c_assert, problems, scripts_sha, arm_in_file)"""
    name = f'{arm}-{PAIR}-s{seed}'
    path = os.path.join(root, name, 'summary.json')
    out = dict(name=name, path=path, found=os.path.isfile(path), val_em=None, fresh_em=None, c_assert=None, problems=[], scripts_sha=None, arm_in_file=None)
    if not out['found']:
        return out
    try:
        with open(path, encoding='utf-8') as fh:
            s = json.load(fh)
    except Exception as e:                                                   # noqa: BLE001
        out['problems'].append(f'unreadable: {e}')
        return out
    if not isinstance(s, dict):
        out['problems'].append('summary is not a dict')
        return out
    out['arm_in_file'] = s.get('arm')
    if s.get('arm') != arm or s.get('pair') != PAIR or str(s.get('seed')) != str(seed):
        out['problems'].append(f'arm / pair / seed {s.get("arm")}, {s.get("pair")}, {s.get("seed")} disagree with the directory')
    for split in ('val', 'fresh'):
        blk = s.get(split)
        em = blk.get('em') if isinstance(blk, dict) else None
        if not _num(em) or not 0.0 <= float(em) <= 100.0:
            out['problems'].append(f'{split}.em is not a number in [0, 100]')
        else:
            out[f'{split}_em'] = float(em)
    if arm == 'S1b':
        if 'c_assert' not in s:
            out['problems'].append('c_assert missing')
        elif s['c_assert'] is not True:
            out['problems'].append('c_assert false')
        out['c_assert'] = s.get('c_assert')
    out['scripts_sha'] = s.get('scripts_sha')
    return out


def margin(a, b):
    return None if a is None or b is None else a - b


def load_all(l1, l1b):
    runs = {}
    for arm in L1_ARMS + L1B_ARMS:
        for seed in SEEDS:
            runs[(arm, seed)] = read_run(l1 if ARM_DIR[arm] == 'l1' else l1b, arm, seed)
    return runs


def _em(runs, arm, seed, split='val'):
    r = runs[(arm, seed)]
    return None if r['problems'] else r[f'{split}_em']


def criteria(runs):
    out = {}
    for key, a, b, thr in (('G0', 'S1b', 'B0', G0_PT), ('P1b', 'S1b', 'T', P1B_PT), ('P2b', 'S1b', 'Ub', P2B_PT)):
        m = {str(s): margin(_em(runs, a, s), _em(runs, b, s)) for s in SEEDS}
        ok = {k: (None if v is None else bool(v >= thr)) for k, v in m.items()}
        out[key] = dict(arms=[a, b], threshold=thr, margin=m, pass_seed=ok, **{'pass': (None if any(v is None for v in ok.values()) else all(ok.values()))})
    d_val = {str(s): margin(_em(runs, 'S1b', s), _em(runs, 'T', s)) for s in SEEDS}
    d_fresh = {str(s): margin(_em(runs, 'S1b', s, 'fresh'), _em(runs, 'T', s, 'fresh')) for s in SEEDS}
    drop = {k: (None if d_val[k] is None or d_fresh[k] is None else d_val[k] - d_fresh[k]) for k in d_val}
    trig = None if any(v is None for v in drop.values()) else any(v > DEGRADE_PT for v in drop.values())
    out['degrade'] = dict(val=d_val, fresh=d_fresh, drop=drop, max_drop_pt=DEGRADE_PT, triggered=trig)
    return out


def g0_line(crit, seed):
    ok = crit['G0']['pass_seed'][str(seed)]
    return f'G0(s{seed}): ' + ('INCOMPLETE' if ok is None else 'PASS' if ok else 'FAIL')


def decide(runs, crit):
    reasons = [f'{r["name"]}: {p}' for r in runs.values() for p in r['problems']]
    s1, s2 = SEEDS
    g0 = crit['G0']['pass_seed']
    if not runs[('S1b', s1)]['found']:
        return 'INCOMPLETE', reasons + [f'missing S1b-{PAIR}-s{s1}']
    if runs[('S1b', s1)]['problems'] or g0[str(s1)] is None:
        return 'INCOMPLETE', reasons + ([f'G0(s{s1}) undecidable (B0-{PAIR}-s{s1} missing or invalid)'] if g0[str(s1)] is None and not runs[('S1b', s1)]['problems'] else [])
    if g0[str(s1)] is False:
        return 'CHANNEL-NO', reasons + [f'G0 failed on seed {s1}: S1b - B0 = {crit["G0"]["margin"][str(s1)]:.2f} < {G0_PT}']
    if not runs[('S1b', s2)]['found'] or runs[('S1b', s2)]['problems'] or g0[str(s2)] is None:
        return 'INCOMPLETE', reasons + [f'S1b-{PAIR}-s{s2} missing / invalid or G0(s{s2}) undecidable']
    if g0[str(s2)] is False:
        return 'CHANNEL-NO', reasons + [f'G0 failed on seed {s2}: S1b - B0 = {crit["G0"]["margin"][str(s2)]:.2f} < {G0_PT}']
    for arm in ('Ub', 'T'):
        for s in SEEDS:
            if not runs[(arm, s)]['found'] or runs[(arm, s)]['problems']:
                return 'INCOMPLETE', reasons + [f'{arm}-{PAIR}-s{s} missing or invalid']
    if crit['P2b']['pass'] is False:
        return 'NO', reasons + ['P2b failed: the sheaf consensus does not beat the one-way projection']
    if crit['P1b']['pass'] is False:
        return 'AMBIGUOUS', reasons + ['P1b failed: the latent channel does not beat the 64-token text exchange']
    if crit['degrade']['triggered']:
        return 'AMBIGUOUS', reasons + ['degrade: S1b - T drops by more than 1.0 pt from VAL to fresh']
    return 'LEVER-pending-P3b', reasons


def aggregate(l1, l1b, gate=None):
    runs = load_all(l1, l1b)
    crit = criteria(runs)
    verdict, reasons = decide(runs, crit)
    lines = [g0_line(crit, s) for s in ((gate,) if gate else SEEDS)]
    out = dict(batch='L1b', prereg='D8-85', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), l1=os.path.abspath(l1), l1b=os.path.abspath(l1b),
               runs={f'{a}-{PAIR}-s{s}': r for (a, s), r in runs.items()}, criteria=crit, verdict=verdict if gate is None else None, reasons=reasons,
               g0_lines=lines, gate_seed=gate, script_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest())
    return out


def to_markdown(agg):
    c = agg['criteria']
    L = [f"# L1b aggregate (D8-85 section 5)", '', f"l1: `{agg['l1']}`  l1b: `{agg['l1b']}`  utc: {agg['utc']}", '',
         f"**VERDICT: {agg['verdict']}**" if agg['verdict'] is not None else f"**gate only (seed {agg['gate_seed']})**", '', *agg['g0_lines'], '',
         '| run | found | VAL EM | fresh EM | c_assert | problems |', '|---|---|---|---|---|---|']
    for name, r in agg['runs'].items():
        f = lambda x: '-' if x is None else f'{x:.2f}'  # noqa: E731
        L.append(f"| {name} | {r['found']} | {f(r['val_em'])} | {f(r['fresh_em'])} | {r['c_assert']} | {'; '.join(r['problems']) or '-'} |")
    L += ['', '| crit | arms | threshold | margin s1 | margin s2 | pass |', '|---|---|---|---|---|---|']
    for k in ('G0', 'P1b', 'P2b'):
        m = c[k]['margin']
        f = lambda x: '-' if x is None else f'{x:.2f}'  # noqa: E731
        pf = lambda x: '-' if x is None else 'PASS' if x else 'FAIL'  # noqa: E731
        L.append(f"| {k} | {c[k]['arms'][0]} - {c[k]['arms'][1]} | >= {c[k]['threshold']} | {f(m[str(SEEDS[0])])} | {f(m[str(SEEDS[1])])} | {pf(c[k]['pass'])} |")
    d = c['degrade']
    L += ['', f"degrade (S1b - T, VAL vs fresh; drop > {d['max_drop_pt']} pt on either seed caps LEVER at AMBIGUOUS): "
          + ', '.join(f"s{s}: VAL {d['val'][str(s)]} fresh {d['fresh'][str(s)]} drop {d['drop'][str(s)]}" for s in SEEDS) + f"; triggered {d['triggered']}",
          '', '## reasons', ''] + ([f'- {r}' for r in agg['reasons']] or ['- none']) + ['', f"l1b_aggregate.py sha256 {agg['script_sha256'][:12]}", '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--l1', required=True, help='L1 runs dir (B0-hetero-s*, T-hetero-s*)')
    ap.add_argument('--l1b', required=True, help='L1b runs dir (S1b-hetero-s*, Ub-hetero-s*)')
    ap.add_argument('--gate', type=int, default=None, help='print only G0 for this seed (no verdict)')
    ap.add_argument('--out', default=None, help='output prefix (default <l1b>/aggregate_l1b)')
    a = ap.parse_args(argv)
    agg = aggregate(a.l1, a.l1b, a.gate)
    prefix = a.out or os.path.join(a.l1b, 'aggregate_l1b' + (f'_gate{a.gate}' if a.gate else ''))
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + '.json.tmp', 'w', encoding='utf-8') as fh:
        json.dump(agg, fh, indent=1)
    os.replace(prefix + '.json.tmp', prefix + '.json')
    with open(prefix + '.md', 'w', encoding='utf-8') as fh:
        fh.write(to_markdown(agg))
    for line in agg['g0_lines']:
        print(line)
    if a.gate is None:
        print(f'VERDICT: {agg["verdict"]}')
    for r in agg['reasons']:
        print(f'- {r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
