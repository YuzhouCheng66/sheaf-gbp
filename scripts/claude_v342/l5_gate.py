"""l5_gate.py (Claude, 2026-11-03): the per-job verdicts of ladder L5 (D8-95 section 1) from summary.json files.
  python -B scripts/claude_v342/l5_gate.py --job B1-same2-s342101 --runs runs/l5 --l4 runs/l4 --out runs/l5/gates
Prints 'GATE <job>: <verdict>' and writes <out>/<job>.json plus the marker <runs>/GATE_<job>_<verdict>.
  B1-same2-s<seed>:   poe_k4 - select_p >= +2 -> BELIEF-ADDS, <= -2 -> BELIEF-HURTS, else BELIEF-NEUTRAL;
                      + poe_k4 - T2(same seed, <l4>/T2-same2-s<seed>) >= +2 -> BELIEF-BEATS-TEXT-R2, <= -2 -> TEXT-BEATS-BELIEF, else TIE (pendingT2 when T2 is missing)
  B1t-same2-s<seed>:  poe_k4 >= 60.2 (one-round text T 58.2 + 2) -> TWO-ROUND-BEATS-TEXT else TWO-ROUND-BELOW-TEXT;
                      + poe_k4 >= 61.7 (all documents, B4) -> BEATS-ALL-DOCS else BELOW-ALL-DOCS
Two seeds of the same kind are needed for a conclusion (D8-95 section 1)."""
from __future__ import annotations
import argparse
import os
import re
import sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402

SEL_PT, TEXT_PT = 2.0, 2.0
T_TEXT, B4_ALL = 58.2, 61.7                       # D8-95 section 1: the one-round text channel T and the all-documents single model B4 (L1)
JOB_RE = re.compile(r'^(B1|B1t)-(same2)-s(\d+)$')


def _r(x):
    return None if x is None else round(float(x), 6)


def read_summary(path, arm, seed, kinds=None):
    if not os.path.isfile(path):
        return None, ['summary.json missing']
    try:
        s = M.jload(path)
    except Exception as e:                                                     # noqa: BLE001
        return None, [f'unreadable summary: {e}']
    problems = []
    if s.get('arm') != arm:
        problems.append(f'arm {s.get("arm")} != {arm}')
    if int(s.get('seed', -1)) != int(seed):
        problems.append(f'seed {s.get("seed")} != {seed}')
    if not s.get('complete', False):
        problems.append('run incomplete')
    if not isinstance((s.get('val') or {}).get('em'), (int, float)):
        problems.append('val.em missing')
    return s, problems


def judge(job, runs, l4):
    m = JOB_RE.match(job)
    if not m:
        raise ValueError(f'unknown job name {job}')
    kind, pair, seed = m.group(1), m.group(2), int(m.group(3))
    s, problems = read_summary(os.path.join(runs, job, 'summary.json'), kind, seed)
    out = dict(job=job, kind=kind, pair=pair, seed=seed, dir=os.path.join(runs, job), problems=problems, criterion=None, verdict='INCOMPLETE', detail={})
    if s is None or problems:
        return out
    v = s['val']
    ev = v.get('em_variants') or {}
    r1 = v.get('round1') or {}
    un = v.get('union') or {}
    base = dict(em=v['em'], fresh=s['fresh']['em'], select_p=ev.get('select_p'), poe_k1=ev.get('poe_k1'), poe_k4_norm=ev.get('poe_k4_norm'), poe_k4_p=ev.get('poe_k4_p'),
                round1_em_agent=r1.get('em_agent'), either=r1.get('either_correct'), agree=r1.get('agree'), union_any=un.get('any_correct'), v_em=(s.get('v') or {}).get('val_em'),
                asym_know_wins=(v.get('asym') or {}).get('know_wins'), bytes=(v.get('bytes') or {}).get('total_bytes'), round2_s=(v.get('timing') or {}).get('round2_s_per_example'))
    if kind == 'B1':
        d_sel = _r(v['em'] - ev['select_p'])
        c = 'BELIEF-ADDS' if d_sel >= SEL_PT else 'BELIEF-HURTS' if d_sel <= -SEL_PT else 'BELIEF-NEUTRAL'
        t2, tp = read_summary(os.path.join(l4, f'T2-same2-s{seed}', 'summary.json'), 'T2', seed)
        if t2 is None or tp:
            t, d_t2, t2_em = 'pendingT2', None, None
        else:
            t2_em = t2['val']['em']
            d_t2 = _r(v['em'] - t2_em)
            t = 'BELIEF-BEATS-TEXT-R2' if d_t2 >= TEXT_PT else 'TEXT-BEATS-BELIEF' if d_t2 <= -TEXT_PT else 'TIE'
        out.update(criterion=f'poe_k4 - select_p >= +{SEL_PT} ADDS / <= -{SEL_PT} HURTS; poe_k4 - T2 >= +{TEXT_PT} BELIEF-BEATS-TEXT-R2 / <= -{TEXT_PT} TEXT-BEATS-BELIEF / else TIE',
                   verdict=f'{c}+{t}', detail=dict(base, margin_vs_select_p=d_sel, t2_em=t2_em, margin_vs_t2=d_t2))
    else:
        a = 'TWO-ROUND-BEATS-TEXT' if v['em'] >= T_TEXT + TEXT_PT else 'TWO-ROUND-BELOW-TEXT'
        b = 'BEATS-ALL-DOCS' if v['em'] >= B4_ALL else 'BELOW-ALL-DOCS'
        out.update(criterion=f'poe_k4 >= {T_TEXT + TEXT_PT} TWO-ROUND-BEATS-TEXT; >= {B4_ALL} BEATS-ALL-DOCS', verdict=f'{a}+{b}',
                   detail=dict(base, margin_vs_text=_r(v['em'] - T_TEXT), margin_vs_b4=_r(v['em'] - B4_ALL)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', required=True)
    ap.add_argument('--runs', required=True, help='the L5 runs dir (markers are written here)')
    ap.add_argument('--l4', required=True, help='the L4 runs dir (T2 runs)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    r = judge(args.job, args.runs, args.l4)
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
