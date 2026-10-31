"""l1e_aggregate.py (Claude, 2026-10-31): the L1e verdict (pre-registration D8-88 = D8-87 section 5) from summary.json files.
Reads Eu-hetero-s<seed> and Es-hetero-s<seed> from the L1e runs dir (--l1e); B0 / T from the L1 runs dir (--l1) are reference only.
VAL EM of the HARD rendering (summary.val; percentage points, mean over agents) decides; the baseline `none` is the one computed IN THE SAME RUN
by the frozen receiver on the plain B0 prompt (summary.baseline.none.val); fresh only for the degrade note.
  G1e (channel, per seed)   Eu - none >= 3.0 pt  AND  other-bucket Eu - none >= 5.0 pt        (both seeds must pass)
  P2e (sheaf vs one-way)    Es - Eu >= +1.0 -> ADDS,  <= -1.0 -> HURTS,  between -> NEUTRAL  (seeds in different classes -> AMBIGUOUS)
  degrade: (Eu - none) on fresh lower than on VAL by more than 1.0 pt on either seed -> ' (degrade: AMBIGUOUS)' appended
  verdict: Eu s1 missing / invalid -> INCOMPLETE; G1e(s1) fails -> CHANNEL-NO-4; Eu s2 missing -> INCOMPLETE; G1e(s2) fails -> CHANNEL-NO-4;
  else CHANNEL-YES / SHEAF-<ADDS|HURTS|NEUTRAL|AMBIGUOUS|pending|INCOMPLETE>. Margins rounded to 6 decimals.
Prints 'G1e(s<seed>): PASS | FAIL | INCOMPLETE' per seed (--gate <seed>: only that seed) and 'VERDICT: ...'; writes <out>.json / .md. Exits 0.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time

SEEDS = (342101, 342202)
G1_EM_PT, G1_OTHER_PT, P2_PT, DEGRADE_PT = 3.0, 5.0, 1.0, 1.0
PAIR = 'hetero'
L1_ARMS, L1D_ARMS = ('B0', 'T'), ('Eu', 'Es')
ARM_DIR = {'B0': 'l1', 'T': 'l1', 'Eu': 'l1e', 'Es': 'l1e'}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _em_other(blk):
    try:
        v = blk['em_by_owner']['other']['em']
        return float(v) if _num(v) else None
    except (KeyError, TypeError):
        return None


def read_run(root, arm, seed):
    name = f'{arm}-{PAIR}-s{seed}'
    path = os.path.join(root, name, 'summary.json')
    out = dict(name=name, path=path, found=os.path.isfile(path), val_em=None, fresh_em=None, val_other=None, none_val_em=None, none_val_other=None,
               none_fresh_em=None, gold_hint_val_em=None, gold_hint_val_other=None, soft_val_em=None, hint_em_other=None, c_assert=None, problems=[],
               scripts_sha=None, arm_in_file=None)
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
    if arm in L1D_ARMS:
        out['val_other'] = _em_other(s.get('val') or {})
        base = (s.get('baseline') or {})
        none_val = (base.get('none') or {}).get('val')
        if not isinstance(none_val, dict) or not _num(none_val.get('em')):
            out['problems'].append('baseline.none.val missing (the frozen receiver on the plain B0 prompt)')
        else:
            out['none_val_em'] = float(none_val['em'])
            out['none_val_other'] = _em_other(none_val)
        none_fresh = (base.get('none') or {}).get('fresh')
        out['none_fresh_em'] = float(none_fresh['em']) if isinstance(none_fresh, dict) and _num(none_fresh.get('em')) else None
        gh = (base.get('gold_hint') or {}).get('val')
        if isinstance(gh, dict) and _num(gh.get('em')):
            out['gold_hint_val_em'], out['gold_hint_val_other'] = float(gh['em']), _em_other(gh)
        sv = (s.get('soft') or {}).get('val')
        out['soft_val_em'] = float(sv['em']) if isinstance(sv, dict) and _num(sv.get('em')) else None
        try:
            hv = s['hint']['val']['hint_by_receiver_bucket']
            out['hint_em_other'] = {a: d.get('other', {}).get('em_gold') for a, d in hv.items()}
        except (KeyError, TypeError, AttributeError):
            out['hint_em_other'] = None
        if out['val_other'] is None:
            out['problems'].append('val.em_by_owner.other missing')
        if s.get('complete') is not True:
            out['problems'].append('complete is not true')
        ch = s.get('channel') or {}
        if ch.get('rendering') not in (None, 'hard'):
            out['problems'].append('criterion rendering is not hard')
        if ch.get('receiver_training') not in (None, 'none (frozen B0 checkpoint)'):
            out['problems'].append('receiver was trained')
        if arm == 'Es':
            if 'c_assert' not in s:
                out['problems'].append('c_assert missing')
            elif s['c_assert'] is not True:
                out['problems'].append('c_assert false')
            out['c_assert'] = s.get('c_assert')
    out['scripts_sha'] = s.get('scripts_sha') if isinstance(s.get('scripts_sha'), dict) else None
    return out


def sha_conflicts(runs):
    seen = {}
    for (arm, seed), r in runs.items():
        if arm not in L1D_ARMS or not r['scripts_sha']:
            continue
        for k, v in r['scripts_sha'].items():
            seen.setdefault(k, {}).setdefault(str(v), []).append(r['name'])
    return {k: vv for k, vv in seen.items() if len(vv) > 1}


def margin(a, b):
    return None if a is None or b is None else round(a - b, 6)


def load_all(l1, l1e):
    runs = {}
    for arm in L1_ARMS + L1D_ARMS:
        for seed in SEEDS:
            runs[(arm, seed)] = read_run(l1 if ARM_DIR[arm] == 'l1' else l1e, arm, seed)
    return runs


def _val(runs, arm, seed, key='val_em'):
    r = runs[(arm, seed)]
    return None if r['problems'] else r[key]


def criteria(runs):
    out = {}
    g_em = {str(s): margin(_val(runs, 'Eu', s), _val(runs, 'Eu', s, 'none_val_em')) for s in SEEDS}
    g_other = {str(s): margin(_val(runs, 'Eu', s, 'val_other'), _val(runs, 'Eu', s, 'none_val_other')) for s in SEEDS}
    ok = {}
    for k in g_em:
        ok[k] = None if (g_em[k] is None or g_other[k] is None) else bool(g_em[k] >= G1_EM_PT and g_other[k] >= G1_OTHER_PT)
    out['G1e'] = dict(arms=['Eu', 'none'], threshold_em=G1_EM_PT, threshold_other=G1_OTHER_PT, margin_em=g_em, margin_other=g_other, pass_seed=ok,
                      **{'pass': (None if any(v is None for v in ok.values()) else all(ok.values()))})
    p2 = {str(s): margin(_val(runs, 'Es', s), _val(runs, 'Eu', s)) for s in SEEDS}
    cls = {k: (None if v is None else 'ADDS' if v >= P2_PT else 'HURTS' if v <= -P2_PT else 'NEUTRAL') for k, v in p2.items()}
    if any(v is None for v in cls.values()):
        verdict = None
    elif len(set(cls.values())) == 1:
        verdict = next(iter(cls.values()))
    else:
        verdict = 'AMBIGUOUS'
    out['P2e'] = dict(arms=['Es', 'Eu'], threshold=P2_PT, margin=p2, class_seed=cls, verdict=verdict)
    d_val = {str(s): margin(_val(runs, 'Eu', s), _val(runs, 'Eu', s, 'none_val_em')) for s in SEEDS}
    d_fresh = {str(s): margin(_val(runs, 'Eu', s, 'fresh_em'), _val(runs, 'Eu', s, 'none_fresh_em')) for s in SEEDS}
    drop = {k: (None if d_val[k] is None or d_fresh[k] is None else round(d_val[k] - d_fresh[k], 6)) for k in d_val}
    trig = None if any(v is None for v in drop.values()) else any(v > DEGRADE_PT for v in drop.values())
    out['degrade'] = dict(val=d_val, fresh=d_fresh, drop=drop, max_drop_pt=DEGRADE_PT, triggered=trig)
    out['reference'] = {f'{a}_s{s}': _val(runs, a, s) for a in ('B0', 'T') for s in SEEDS}
    for s in SEEDS:
        r = runs[('Eu', s)]
        out['reference'].update({f'gold_hint_s{s}': r['gold_hint_val_em'], f'gold_hint_other_s{s}': r['gold_hint_val_other'], f'soft_Du_s{s}': r['soft_val_em'],
                                 f'hint_em_other_Du_s{s}': r['hint_em_other']})
    return out


def g1_line(crit, seed):
    ok = crit['G1e']['pass_seed'][str(seed)]
    return f'G1e(s{seed}): ' + ('INCOMPLETE' if ok is None else 'PASS' if ok else 'FAIL')


def decide(runs, crit):
    reasons = [f'{r["name"]}: {p}' for (a, s), r in runs.items() if a in L1D_ARMS for p in r['problems']]
    conflicts = sha_conflicts(runs)
    reasons += [f'scripts_sha of {k} differs between L1e runs: ' + '; '.join(f'{sha[:12]} <- {sorted(v)}' for sha, v in sorted(vv.items())) for k, vv in sorted(conflicts.items())]
    if conflicts:
        return 'INCOMPLETE', reasons
    g1 = crit['G1e']['pass_seed']
    for s in SEEDS:
        r = runs[('Eu', s)]
        if not r['found']:
            return 'INCOMPLETE', reasons + [f'missing Eu-{PAIR}-s{s}']
        if r['problems'] or g1[str(s)] is None:
            return 'INCOMPLETE', reasons + [f'G1e(s{s}) undecidable']
        if g1[str(s)] is False:
            return 'CHANNEL-NO-4', reasons + [f'G1e failed on seed {s}: Eu - none = {crit["G1e"]["margin_em"][str(s)]:.2f} pt (>= {G1_EM_PT}?), '
                                              f'other bucket {crit["G1e"]["margin_other"][str(s)]:.2f} pt (>= {G1_OTHER_PT}?)']
    deg = ''
    if crit['degrade']['triggered']:
        reasons.append('degrade: Eu - none drops by more than 1.0 pt from VAL to fresh -> marked AMBIGUOUS (D8-88 section 5); the channel verdict stands')
        deg = ' (degrade: AMBIGUOUS)'
    sub = crit['P2e']['verdict']
    if sub is None:
        missing = [f'Es-{PAIR}-s{s}' for s in SEEDS if not runs[('Es', s)]['found']]
        invalid = [f'Es-{PAIR}-s{s}' for s in SEEDS if runs[('Es', s)]['found'] and runs[('Es', s)]['problems']]
        if invalid:
            return 'CHANNEL-YES / SHEAF-INCOMPLETE' + deg, reasons + [f'Es invalid (C assertion / schema / complete): {invalid}'] + ([f'Es missing: {missing}'] if missing else [])
        return 'CHANNEL-YES / SHEAF-pending' + deg, reasons + [f'Es not run yet: {missing}']
    return f'CHANNEL-YES / SHEAF-{sub}' + deg, reasons + [f'P2e: Es - Eu = {crit["P2e"]["margin"]} -> {crit["P2e"]["class_seed"]}']


def aggregate(l1, l1e, gate=None):
    runs = load_all(l1, l1e)
    crit = criteria(runs)
    verdict, reasons = decide(runs, crit)
    lines = [g1_line(crit, s) for s in ((gate,) if gate else SEEDS)]
    return dict(batch='L1e', prereg='D8-88', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), l1=os.path.abspath(l1), l1e=os.path.abspath(l1e),
                runs={f'{a}-{PAIR}-s{s}': r for (a, s), r in runs.items()}, criteria=crit, verdict=verdict if gate is None else None, reasons=reasons,
                g1_lines=lines, gate_seed=gate, script_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest())


def to_markdown(agg):
    c = agg['criteria']
    f = lambda x: '-' if x is None else f'{x:.2f}'  # noqa: E731
    L = ['# L1e aggregate (D8-88 section 5)', '', f"l1: `{agg['l1']}`  l1e: `{agg['l1e']}`  utc: {agg['utc']}", '',
         f"**VERDICT: {agg['verdict']}**" if agg['verdict'] is not None else f"**gate only (seed {agg['gate_seed']})**", '', *agg['g1_lines'], '',
         '| run | found | VAL EM (hard) | VAL other | none EM | none other | gold_hint EM | gold_hint other | soft EM | fresh EM | c_assert | problems |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, r in agg['runs'].items():
        L.append(f"| {name} | {r['found']} | {f(r['val_em'])} | {f(r['val_other'])} | {f(r['none_val_em'])} | {f(r['none_val_other'])} | {f(r['gold_hint_val_em'])} | "
                 f"{f(r['gold_hint_val_other'])} | {f(r['soft_val_em'])} | {f(r['fresh_em'])} | {r['c_assert']} | {'; '.join(r['problems']) or '-'} |")
    g = c['G1e']
    pf = lambda x: '-' if x is None else 'PASS' if x else 'FAIL'  # noqa: E731
    L += ['', '| crit | seed | Eu - none EM (>= 3.0) | Eu - none other (>= 5.0) | pass |', '|---|---|---|---|---|']
    for s in SEEDS:
        L.append(f"| G1e | {s} | {f(g['margin_em'][str(s)])} | {f(g['margin_other'][str(s)])} | {pf(g['pass_seed'][str(s)])} |")
    p = c['P2e']
    L += ['', 'P2e (Es - Eu, VAL): ' + ', '.join(f"s{s}: {f(p['margin'][str(s)])} -> {p['class_seed'][str(s)]}" for s in SEEDS) + f"; sheaf verdict {p['verdict']}",
          '', 'degrade (Eu - none, VAL vs fresh): ' + ', '.join(f"s{s}: VAL {f(c['degrade']['val'][str(s)])} fresh {f(c['degrade']['fresh'][str(s)])} drop {f(c['degrade']['drop'][str(s)])}" for s in SEEDS)
          + f"; triggered {c['degrade']['triggered']}", '', 'reference: ' + json.dumps(c['reference']), '', '## reasons', '']
    L += [f'- {r}' for r in agg['reasons']] or ['- none']
    L += ['', f"l1e_aggregate.py sha256 {agg['script_sha256'][:12]}", '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--l1', required=True, help='L1 runs dir (B0 / T: reference only)')
    ap.add_argument('--l1e', required=True, help='L1e runs dir (Eu-hetero-s*, Es-hetero-s*)')
    ap.add_argument('--gate', type=int, default=None, help='print only G1e for this seed (no verdict)')
    ap.add_argument('--out', default=None, help='output prefix (default <l1e>/aggregate_l1e)')
    a = ap.parse_args(argv)
    agg = aggregate(a.l1, a.l1e, a.gate)
    prefix = a.out or os.path.join(a.l1e, 'aggregate_l1e' + (f'_gate{a.gate}' if a.gate else ''))
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + '.json.tmp', 'w', encoding='utf-8') as fh:
        json.dump(agg, fh, indent=1)
    os.replace(prefix + '.json.tmp', prefix + '.json')
    with open(prefix + '.md', 'w', encoding='utf-8') as fh:
        fh.write(to_markdown(agg))
    for line in agg['g1_lines']:
        print(line)
    if a.gate is None:
        print(f'VERDICT: {agg["verdict"]}')
    for r in agg['reasons']:
        print(f'- {r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
