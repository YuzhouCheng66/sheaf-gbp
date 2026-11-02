"""l3_gate.py (Claude, 2026-11-02): the per-job verdicts of ladder L3 (D8-93 section 4) from summary.json files.
  python -B scripts/claude_v342/l3_gate.py --job S2b-same2-s342101 --runs runs/l3 --l2 runs/l2 --out runs/l3/gates
Prints 'GATE <job>: <verdict>' and writes <out>/<job>.json plus the marker <runs>/GATE_<job>_<verdict>.
  S2a-same2-s<seed>:  DONE (readouts: round-1 EM, either-correct, consensus decoded EM, lam AUROC) or INCOMPLETE
  S2b-same2-s<seed>:  S2 - V2(same seed, <l2>/V2-same2-s<seed>) >= +3 -> CONSENSUS-ADDS, <= -3 -> CONSENSUS-HURTS, else CONSENSUS-NEUTRAL;
                      + S2 - T2(same seed, <runs>/T2-same2-s<seed>) >= +2 -> LATENT-BEATS-TEXT-R2, <= -2 -> TEXT-BEATS-LATENT-R2, else TIE
                      (pendingT2 when T2 is missing); verdict = '<consensus>+<text>'
  T2-same2-s<seed>:   DONE (readouts) or INCOMPLETE
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

CONS_PT, TEXT_PT = 3.0, 2.0
JOB_RE = re.compile(r'^(S2a|S2b|T2)-(same2)-s(\d+)$')


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


def judge(job, runs, l2):
    m = JOB_RE.match(job)
    if not m:
        raise ValueError(f'unknown job name {job}')
    kind, pair, seed = m.group(1), m.group(2), int(m.group(3))
    s, problems = read_summary(os.path.join(runs, job, 'summary.json'), kind, pair, seed)
    out = dict(job=job, kind=kind, pair=pair, seed=seed, dir=os.path.join(runs, job), problems=problems, criterion=None, verdict='INCOMPLETE', detail={})
    if s is None or problems:
        return out
    r1 = (s['val'].get('round1') or {})
    cons = s['val'].get('consensus') or {}
    base = dict(em=s['val']['em'], fresh=s['fresh']['em'], none=((s.get('baseline') or {}).get('none', {}).get('val') or {}).get('em'), v2=(s.get('v2') or {}).get('val_em'),
                round1_em_agent=r1.get('em_agent'), either=r1.get('either_correct'), both=r1.get('both_correct'), p_auroc=r1.get('p_auroc'),
                consensus_em_agent=cons.get('em_agent'), lam_auroc=cons.get('lam_auroc_vs_round1_correct'), w2=cons.get('w2'))
    if kind in ('S2a', 'T2'):
        out.update(criterion='readouts only', verdict='DONE', detail=base)
        return out
    v2, vp = read_summary(os.path.join(l2, f'V2-same2-s{seed}', 'summary.json'), 'V2', 'same2', seed)
    if v2 is None or vp:
        out['problems'].append(f'reference V2-same2-s{seed} unusable: {vp}')
        return out
    d_v2 = _r(s['val']['em'] - v2['val']['em'])
    c = 'CONSENSUS-ADDS' if d_v2 >= CONS_PT else 'CONSENSUS-HURTS' if d_v2 <= -CONS_PT else 'CONSENSUS-NEUTRAL'
    t2, tp = read_summary(os.path.join(runs, f'T2-same2-s{seed}', 'summary.json'), 'T2', 'same2', seed)
    if t2 is None or tp:
        t, d_t2 = 'pendingT2', None
    else:
        d_t2 = _r(s['val']['em'] - t2['val']['em'])
        t = 'LATENT-BEATS-TEXT-R2' if d_t2 >= TEXT_PT else 'TEXT-BEATS-LATENT-R2' if d_t2 <= -TEXT_PT else 'TIE'
    out.update(criterion=f'S2 - V2 >= +{CONS_PT} ADDS / <= -{CONS_PT} HURTS; S2 - T2 >= +{TEXT_PT} LATENT-BEATS-TEXT-R2 / <= -{TEXT_PT} TEXT-BEATS-LATENT-R2 / else TIE',
               verdict=f'{c}+{t}', detail=dict(base, v2_em=v2['val']['em'], margin_vs_v2=d_v2, t2_em=(t2['val']['em'] if t2 is not None and not tp else None), margin_vs_t2=d_t2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', required=True)
    ap.add_argument('--runs', required=True, help='the L3 runs dir (markers are written here)')
    ap.add_argument('--l2', required=True, help='the L2 runs dir (V2 runs)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    r = judge(args.job, args.runs, args.l2)
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
