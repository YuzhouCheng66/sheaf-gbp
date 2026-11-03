"""l2_gate.py (Claude, 2026-11-01): the per-job verdicts of the L2 ladder (pre-registration D8-92 section 4) from summary.json files.
  python -B scripts/claude_v342/l2_gate.py --job R1-same2-s342101 --runs runs/l2 --l1h runs/l1h --out runs/l2/gates
Prints 'GATE <job>: <verdict>' and writes <out>/<job>.json plus the marker file <runs>/GATE_<job>_<verdict> (the chain's conditions read it).
Jobs and criteria (VAL, hard rendering; margins in EM points, rounded to 6 decimals):
  R1-<pair>-s<seed>, V1-same2-s<seed>, Hu-hetero-s<seed>, Hu-same2-s<seed>:  EM - none >= 3.0 AND other - none_other >= 5.0 -> PASS / FAIL
  Hs-same2-s<seed>:          Hs - Hu(same seed, runs/l1h) >= +1.0 -> ADDS, <= -1.0 -> HURTS, else NEUTRAL (P2h)
  Hu-d32-same2-s<seed>:      |EM - Hu(same seed)| <= 1.0 -> NONINFERIOR / INFERIOR (CODE-LOWDIM)
  Hu-L9-same2-s<seed>:       slot-0 copy vs the sender's answer >= 0.95 AND slots 1-2 >= 0.80 -> PASS / FAIL (EM reported)
  V1s-same2-s<seed>:         V1s - V1(same seed, runs/l2) >= +1.0 -> SHEAF-ADDS, <= -1.0 -> SHEAF-HURTS, else SHEAF-NEUTRAL (appendix 3; C assertion must hold)
A missing / invalid summary -> INCOMPLETE. Exits 0.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_aggregate as A1  # noqa: E402
import l1_model as M  # noqa: E402

G1_EM_PT, G1_OTHER_PT, P2_PT, NONINF_PT = 3.0, 5.0, 1.0, 1.0
L9_SLOT0, L9_SLOT12 = 0.95, 0.80
JOB_RE = re.compile(r'^(R1|V1s|V2|V3|V1|Hs|Hu|Hu-d32|Hu-L9)-(same2|hetero)-s(\d+)$')


def _r(x):
    return None if x is None else round(float(x), 6)


def read_summary(path, arm, pair, seed):
    if not os.path.isfile(path):
        return None, ['summary.json missing']
    try:
        s = M.jload(path)
    except Exception as e:                                                     # noqa: BLE001
        return None, [f'unreadable summary: {e}']
    problems = A1.validate_summary(s, arm, pair, seed)
    if not s.get('complete', False):
        problems.append('run incomplete')
    return s, problems


def _val(s, key='em'):
    return s['val'][key]


def _other(blk):
    return ((blk.get('em_by_owner') or {}).get('other') or {}).get('em')


def judge(job, runs, l1h):
    m = JOB_RE.match(job)
    if not m:
        raise ValueError(f'unknown job name {job}')
    kind, pair, seed = m.group(1), m.group(2), int(m.group(3))
    arm = {'R1': 'R1', 'V1': 'V1', 'V1s': 'V1s', 'V2': 'V2', 'V3': 'V3', 'Hs': 'Hs', 'Hu': 'Hu', 'Hu-d32': 'Hu', 'Hu-L9': 'Hu'}[kind]
    where = l1h if kind in ('Hs', 'Hu') and pair == 'same2' else runs
    s, problems = read_summary(os.path.join(where, job, 'summary.json'), arm, pair, seed)
    out = dict(job=job, kind=kind, pair=pair, seed=seed, dir=os.path.join(where, job), problems=problems, criterion=None, verdict='INCOMPLETE', detail={})
    if s is None or problems:
        return out
    if kind == 'V1s':                                                       # appendix 3: P2-V1s vs the V1 run of the same seed
        v1, vp = read_summary(os.path.join(runs, f'V1-same2-s{seed}', 'summary.json'), 'V1', 'same2', seed)
        if v1 is None or vp:
            out['problems'].append(f'reference V1-same2-s{seed} unusable: {vp}')
            return out
        d = _r(_val(s) - _val(v1))
        v = 'SHEAF-ADDS' if d >= P2_PT else 'SHEAF-HURTS' if d <= -P2_PT else 'SHEAF-NEUTRAL'
        if s.get('c_assert') is False:
            v = 'INCOMPLETE'
            out['problems'].append('C assertion failed')
        none = (s.get('baseline') or {}).get('none', {}).get('val') or {}
        out.update(criterion=f'V1s - V1 >= +{P2_PT} ADDS / <= -{P2_PT} HURTS / else NEUTRAL', verdict=v,
                   detail=dict(v1s=_val(s), v1=_val(v1), margin=d, c_assert=s.get('c_assert'), w=s.get('w'), none=none.get('em'),
                               margin_em_vs_none=(_r(_val(s) - none['em']) if none else None), val_oracle=(s.get('val_oracle') or {}).get('em'), v1_oracle=(v1.get('val_oracle') or {}).get('em')))
        return out
    if kind in ('V2', 'V3'):                                                # D8-93 / D8-94: G1 thresholds then vs the V1 run of the same seed (>= +2 -> SELECT-YES)
        none = (s.get('baseline') or {}).get('none', {}).get('val')
        if not none:
            out['problems'].append('baseline none missing')
            return out
        em = _r(_val(s) - none['em'])
        other = _r(_other(s['val']) - _other(none)) if _other(s['val']) is not None and _other(none) is not None else None
        g_ok = em is not None and other is not None and em >= G1_EM_PT and other >= G1_OTHER_PT
        v1, vp = read_summary(os.path.join(runs, f'V1-same2-s{seed}', 'summary.json'), 'V1', 'same2', seed)
        d = _r(_val(s) - _val(v1)) if v1 is not None and not vp else None
        verdict = 'FAIL' if not g_ok else ('PASS' if d is None else ('SELECT-YES' if d >= 2.0 else 'SELECT-NO'))
        out.update(criterion='G-V2 (EM - none >= 3.0 and other >= 5.0) then V2 - V1 >= +2.0 -> SELECT-YES', verdict=verdict,
                   detail=dict(em=_val(s), none=none['em'], margin_em=em, margin_other=other, v1=(_val(v1) if v1 is not None and not vp else None), margin_vs_v1=d,
                               val_oracle=(s.get('val_oracle') or {}).get('em'), sel_overlap50=((s.get('sel_quality') or {}).get('overlap50')), fresh=s['fresh']['em']))
        return out
    if kind in ('R1', 'V1', 'Hu'):
        none = (s.get('baseline') or {}).get('none', {}).get('val')
        if not none:
            out['problems'].append('baseline none missing')
            return out
        em, other = _r(_val(s) - none['em']), (_r(_other(s['val']) - _other(none)) if _other(s['val']) is not None and _other(none) is not None else None)
        ok = em is not None and other is not None and em >= G1_EM_PT and other >= G1_OTHER_PT
        out.update(criterion=f'EM - none >= {G1_EM_PT} and other - none_other >= {G1_OTHER_PT}', verdict='PASS' if ok else 'FAIL',
                   detail=dict(em=_val(s), none=none['em'], margin_em=em, other=_other(s['val']), none_other=_other(none), margin_other=other,
                               fresh=s['fresh']['em'], none_fresh=((s.get('baseline') or {}).get('none', {}).get('fresh') or {}).get('em'),
                               val_oracle=(s.get('val_oracle') or {}).get('em'), none_recv=((s.get('baseline') or {}).get('none_recv', {}).get('val') or {}).get('em')))
        return out
    hu, hp = read_summary(os.path.join(l1h, f'Hu-same2-s{seed}', 'summary.json'), 'Hu', 'same2', seed)
    if hu is None or hp:
        out['problems'].append(f'reference Hu-same2-s{seed} unusable: {hp}')
        return out
    if kind == 'Hs':
        d = _r(_val(s) - _val(hu))
        v = 'ADDS' if d >= P2_PT else 'HURTS' if d <= -P2_PT else 'NEUTRAL'
        if s.get('c_assert') is False:
            v, problems = 'INCOMPLETE', out['problems'] + ['C assertion failed']
        out.update(criterion=f'Hs - Hu >= +{P2_PT} ADDS / <= -{P2_PT} HURTS / else NEUTRAL', verdict=v, detail=dict(hs=_val(s), hu=_val(hu), margin=d, c_assert=s.get('c_assert')))
        return out
    if kind == 'Hu-d32':
        d = _r(_val(s) - _val(hu))
        out.update(criterion=f'|EM - Hu| <= {NONINF_PT}', verdict='NONINFERIOR' if abs(d) <= NONINF_PT else 'INFERIOR', detail=dict(em=_val(s), hu=_val(hu), diff=d, d_e=(s.get('channel') or {}).get('d_e')))
        return out
    sa = (s.get('slot_acc') or {}).get('acc_sender_by_slot') or []
    s0 = sa[0] if len(sa) > 0 else None
    s12 = [x for x in sa[1:3] if x is not None]
    ok = s0 is not None and len(s12) == 2 and s0 >= L9_SLOT0 and min(s12) >= L9_SLOT12
    out.update(criterion=f'slot0 copy >= {L9_SLOT0} and slots 1-2 >= {L9_SLOT12}', verdict='PASS' if ok else 'FAIL',
               detail=dict(em=_val(s), hu=_val(hu), diff=_r(_val(s) - _val(hu)), acc_sender_by_slot=sa, read_layer=[a.get('read_layer') for a in (s.get('agents') or [])]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', required=True)
    ap.add_argument('--runs', required=True, help='the L2 runs dir (markers are written here)')
    ap.add_argument('--l1h', required=True, help='the L1h runs dir (Hu / Hs same2 runs)')
    ap.add_argument('--out', required=True, help='dir of the per-job json records')
    args = ap.parse_args(argv)
    r = judge(args.job, args.runs, args.l1h)
    os.makedirs(args.out, exist_ok=True)
    r['utc'] = M.utc()
    M.jdump(r, os.path.join(args.out, f'{args.job}.json'))
    for old in os.listdir(args.runs):
        if old.startswith(f'GATE_{args.job}_'):
            os.remove(os.path.join(args.runs, old))
    with open(os.path.join(args.runs, f'GATE_{args.job}_{r["verdict"]}'), 'w') as fh:
        fh.write(r['utc'] + '\n')
    print(f'GATE {args.job}: {r["verdict"]}')
    if r['detail']:
        print('  ' + ', '.join(f'{k} {v}' for k, v in r['detail'].items() if not isinstance(v, (list, dict))))
    if r['problems']:
        print('  problems: ' + '; '.join(r['problems']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
