"""l6_gate.py (Claude, 2026-11-03): the per-job verdicts of ladder L6 (D8-96 section 1) from summary.json files.
  python -B scripts/claude_v342/l6_gate.py --job VER-same2-s342101 --runs runs/l6 --l5 runs/l5 --l4 runs/l4 --out runs/l6/gates
Prints 'GATE <job>: <verdict>' and writes <out>/<job>.json plus the marker <runs>/GATE_<job>_<verdict>.
  GEN-same2-s<seed>:  summary.json complete -> GEN-DONE
  VER-same2-s<seed> / VERX-same2-s<seed> (doc-swapped negatives):  H_ver  ver_sum - select_p >= +2 -> VERIFIER-ADDS, <= -2 -> VERIFIER-HURTS, else VERIFIER-NEUTRAL
                      H_text ver_sum - T2(same seed, <l4>/T2-same2-s<seed>) >= +2 -> BELIEF-BEATS-TEXT-R2, <= -2 -> TEXT-BEATS-BELIEF, else TIE (pendingT2 when missing);
                             ver_sum >= 61.7 -> BEATS-ALL-DOCS, >= 60.2 (one-round text 58.2 + 2) -> BEATS-TEXT, else BELOW-TEXT
                      H_sum  ver_sum - max_i ver_self_i >= +2 -> SUM-ADDS else SUM-NEUTRAL
                      (select_p is cross-checked against the L5 B1t summary of the seed: the same candidates must give the same number)
Two seeds are needed for a conclusion (D8-96 section 1)."""
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

SEL_PT, TEXT_PT, SUM_PT = 2.0, 2.0, 2.0
T_TEXT, B4_ALL = 58.2, 61.7
JOB_RE = re.compile(r'^(VER|VERX|GEN)-(same2)-s(\d+)$')


def _r(x):
    return None if x is None else round(float(x), 6)


def read_summary(path, arm, seed):
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
    if arm == 'VER' and not isinstance((s.get('val') or {}).get('em'), (int, float)):
        problems.append('val.em missing')
    return s, problems


def judge(job, runs, l5, l4):
    m = JOB_RE.match(job)
    if not m:
        raise ValueError(f'unknown job name {job}')
    kind, pair, seed = m.group(1), m.group(2), int(m.group(3))
    s, problems = read_summary(os.path.join(runs, job, 'summary.json'), kind, seed)
    out = dict(job=job, kind=kind, pair=pair, seed=seed, dir=os.path.join(runs, job), problems=problems, criterion=None, verdict='INCOMPLETE', detail={})
    if s is None or problems:
        return out
    if kind == 'GEN':
        g = s.get('gen') or {}
        out.update(criterion='summary.json complete', verdict='GEN-DONE', detail=dict(n_rows=s.get('n_rows'), union_any_correct=g.get('union_any_correct'), either_top1=g.get('either_top1'),
                                                                                       size_mean=g.get('size_mean'), positive_rate=g.get('positive_rate')))
        return out
    v = s['val']
    ev = v.get('em_variants') or {}
    r1 = v.get('round1') or {}
    un = v.get('union') or {}
    ver = v.get('verifier') or {}
    l5s, l5p = read_summary(os.path.join(l5, f'B1t-same2-s{seed}', 'summary.json'), 'B1t', seed)
    l5_sel = ((l5s or {}).get('val') or {}).get('em_variants', {}).get('select_p') if l5s and not l5p else None
    if l5_sel is not None and ev.get('select_p') is not None and abs(float(l5_sel) - float(ev['select_p'])) > 0.05:
        out['problems'].append(f'select_p {ev["select_p"]} != L5 B1t {l5_sel} (different candidates?)')
        return out
    self_max = max(float(ev.get('ver_self0', 0.0)), float(ev.get('ver_self1', 0.0)))
    d_sel = _r(v['em'] - ev['select_p'])
    c1 = 'VERIFIER-ADDS' if d_sel >= SEL_PT else 'VERIFIER-HURTS' if d_sel <= -SEL_PT else 'VERIFIER-NEUTRAL'
    t2, tp = read_summary(os.path.join(l4, f'T2-same2-s{seed}', 'summary.json'), 'T2', seed)
    if t2 is None or tp:
        c2, d_t2, t2_em = 'pendingT2', None, None
    else:
        t2_em = t2['val']['em']
        d_t2 = _r(v['em'] - t2_em)
        c2 = 'BELIEF-BEATS-TEXT-R2' if d_t2 >= TEXT_PT else 'TEXT-BEATS-BELIEF' if d_t2 <= -TEXT_PT else 'TIE'
    c3 = 'BEATS-ALL-DOCS' if v['em'] >= B4_ALL else 'BEATS-TEXT' if v['em'] >= T_TEXT + TEXT_PT else 'BELOW-TEXT'
    d_sum = _r(v['em'] - self_max)
    c4 = 'SUM-ADDS' if d_sum >= SUM_PT else 'SUM-NEUTRAL'
    by_rel = ver.get('by_rel') or {}
    out.update(criterion=f'ver_sum - select_p >= +{SEL_PT} ADDS / <= -{SEL_PT} HURTS; ver_sum - T2 >= +{TEXT_PT} BEATS-TEXT-R2 / <= -{TEXT_PT} TEXT-BEATS; >= {B4_ALL} BEATS-ALL-DOCS / >= {T_TEXT + TEXT_PT} BEATS-TEXT; '
                         f'ver_sum - max ver_self >= +{SUM_PT} SUM-ADDS',
               verdict=f'{c1}+{c2}+{c3}+{c4}',
               detail=dict(em=v['em'], fresh=s['fresh']['em'], select_p=ev.get('select_p'), l5_select_p=l5_sel, poe_k4=ev.get('poe_k4'), ver_self=[ev.get('ver_self0'), ev.get('ver_self1')],
                           ver_select=ev.get('ver_select'), ver_sum_p=ev.get('ver_sum_p'), ver_owner=ev.get('ver_owner'), margin_vs_select_p=d_sel, t2_em=t2_em, margin_vs_t2=d_t2,
                           margin_vs_text=_r(v['em'] - T_TEXT), margin_vs_b4=_r(v['em'] - B4_ALL), margin_vs_self=d_sum, round1_em_agent=r1.get('em_agent'), either=r1.get('either_correct'),
                           union_any=un.get('any_correct'), know_wins=(v.get('asym') or {}).get('know_wins'), flat_when_ignorant=ver.get('flat_when_ignorant'),
                           auroc_own=(by_rel.get('own') or {}).get('auroc'), auroc_other=(by_rel.get('other') or {}).get('auroc'), auroc_all=(by_rel.get('all') or {}).get('auroc'),
                           train_complete=s.get('train_complete'), steps=s.get('steps'), steps_planned=s.get('steps_planned'), bytes=(v.get('bytes') or {}).get('total_bytes'),
                           round2_s=(v.get('timing') or {}).get('round2_s_per_example')))
    if s.get('train_complete') is False:
        out['notes'] = [f'training stopped early: {s.get("steps")}/{s.get("steps_planned")} steps (the verdict stands; recorded)']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', required=True)
    ap.add_argument('--runs', required=True, help='the L6 runs dir (markers are written here)')
    ap.add_argument('--l5', required=True, help='the L5 runs dir (B1t references)')
    ap.add_argument('--l4', required=True, help='the L4 runs dir (T2 references)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    r = judge(args.job, args.runs, args.l5, args.l4)
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
        print('  ' + ', '.join(f'{k} {v}' for k, v in r['detail'].items() if not isinstance(v, dict)))
    if r.get('notes'):
        print('  notes: ' + '; '.join(r['notes']))
    if r['problems']:
        print('  problems: ' + '; '.join(r['problems']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
