"""l7_gate.py (Claude, 2026-11-04): the per-job verdicts of ladder L7 (D8-97 section 1) from summary.json files.
  python -B scripts/claude_v342/l7_gate.py --job VER-N4-s342101 --runs runs/l7 --out runs/l7/gates
Prints 'GATE <job>: <verdict>' and writes <out>/<job>.json plus the marker <runs>/GATE_<job>_<verdict>.
  TSEND / TRECV / RECV / GEN -N<N>-s<seed>:  summary complete (training complete) -> <KIND>-DONE
  VER-N<N>-s<seed>  (needs TRECV-N<N>-s<seed>):  B = ver_sum VAL EM, T = the text arm's VAL EM (mean over the N agents)
      H_N:  B - T >= +2 -> BELIEF-BEATS-TEXT-N<N>, <= -2 -> TEXT-BEATS-BELIEF-N<N>, else TIE-N<N>
      H_time:  belief total seconds per example <= text total seconds per example -> FASTER-N<N> else SLOWER-N<N>
      H_trend (N != 2, when VER-N2 and TRECV-N2 of the seed exist):  (B_N - B_2) - (T_N - T_2) >= +2 BELIEF-SCALES-BETTER / <= -2 TEXT-SCALES-BETTER / SAME-TREND
      H_ign (N > 2):  know_wins >= 75 and flat_when_ignorant -> IGNORANT-FLAT else IGNORANT-NOT-FLAT
Two seeds are needed for a conclusion (D8-97 section 1); bytes are reported, never judged."""
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

PT = 2.0
KNOW_PT = 75.0
JOB_RE = re.compile(r'^(TSEND|TRECV|RECV|GEN|VER)-N(\d+)-s(\d+)$')


def _r(x):
    return None if x is None else round(float(x), 6)


def read_summary(path, kind, n_agents, seed):
    if not os.path.isfile(path):
        return None, ['summary.json missing']
    try:
        s = M.jload(path)
    except Exception as e:                                                     # noqa: BLE001
        return None, [f'unreadable summary: {e}']
    problems = []
    if s.get('job') != kind.lower():
        problems.append(f'job {s.get("job")} != {kind.lower()}')
    if int(s.get('n_agents', -1)) != int(n_agents):
        problems.append(f'n_agents {s.get("n_agents")} != {n_agents}')
    if int(s.get('seed', -1)) != int(seed):
        problems.append(f'seed {s.get("seed")} != {seed}')
    if not s.get('complete', False):
        problems.append('run incomplete')
    return s, problems


def labels(d, n):
    return f'BELIEF-BEATS-TEXT-N{n}' if d >= PT else f'TEXT-BEATS-BELIEF-N{n}' if d <= -PT else f'TIE-N{n}'


def judge(job, runs):
    m = JOB_RE.match(job)
    if not m:
        raise ValueError(f'unknown job name {job}')
    kind, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
    s, problems = read_summary(os.path.join(runs, job, 'summary.json'), kind, n, seed)
    out = dict(job=job, kind=kind, n_agents=n, seed=seed, dir=os.path.join(runs, job), problems=problems, criterion=None, verdict='INCOMPLETE', detail={})
    if s is None or problems:
        return out
    if kind != 'VER':
        tc = s.get('train_complete', True)
        d = {}
        if kind == 'TRECV':
            v = s.get('val') or {}
            d = dict(em=v.get('em'), em_agent=v.get('em_agent'), vote=v.get('vote'), either=v.get('either_correct'), fresh=(s.get('fresh') or {}).get('em'),
                     total_s=(v.get('timing') or {}).get('total_s_per_example'), bytes=(v.get('bytes') or {}).get('message_bytes'))
        elif kind == 'GEN':
            d = dict(n_rows=s.get('n_rows'), **(s.get('gen') or {}))
        elif kind == 'RECV':
            v = s.get('val') or {}
            d = dict(em_agent=v.get('em_agent'), either=v.get('either_correct'))
        elif kind == 'TSEND':
            d = dict(s.get('val') or {})
        out.update(criterion='summary complete and training complete', verdict=f'{kind}-DONE' if tc else f'{kind}-TRAIN-STOPPED', detail=d)
        return out
    v = s['val']
    ev = v.get('em_variants') or {}
    t, tp = read_summary(os.path.join(runs, f'TRECV-N{n}-s{seed}', 'summary.json'), 'TRECV', n, seed)
    if t is None or tp:
        out['problems'].append(f'TRECV-N{n}-s{seed}: {"; ".join(tp or ["missing"])}')
        out['verdict'] = 'pendingTEXT'
        out['detail'] = dict(em=v.get('em'), fresh=(s.get('fresh') or {}).get('em'))
        return out
    B, T = float(v['em']), float(t['val']['em'])
    d_bt = _r(B - T)
    c1 = labels(d_bt, n)
    tb, tt = (v.get('timing') or {}).get('total_s_per_example'), (t['val'].get('timing') or {}).get('total_s_per_example')
    c2 = ('FASTER' if (tb is not None and tt is not None and tb <= tt) else 'SLOWER') + f'-N{n}'
    parts = [c1, c2]
    trend = None
    if n != 2:
        v2, p2 = read_summary(os.path.join(runs, f'VER-N2-s{seed}', 'summary.json'), 'VER', 2, seed)
        t2, q2 = read_summary(os.path.join(runs, f'TRECV-N2-s{seed}', 'summary.json'), 'TRECV', 2, seed)
        if v2 is not None and not p2 and t2 is not None and not q2:
            trend = _r((B - float(v2['val']['em'])) - (T - float(t2['val']['em'])))
            parts.append('BELIEF-SCALES-BETTER' if trend >= PT else 'TEXT-SCALES-BETTER' if trend <= -PT else 'SAME-TREND')
        else:
            parts.append('pendingN2')
        kw = (v.get('know') or {}).get('know_wins')
        flat = (v.get('verifier') or {}).get('flat_when_ignorant')
        parts.append('IGNORANT-FLAT' if (kw is not None and kw >= KNOW_PT and flat) else 'IGNORANT-NOT-FLAT')
    by_rel = (v.get('verifier') or {}).get('by_rel') or {}
    out.update(criterion=f'B - T >= +{PT} BEATS / <= -{PT} TEXT-BEATS; belief s/example <= text s/example FASTER; trend +-{PT}; know_wins >= {KNOW_PT} and flat -> IGNORANT-FLAT',
               verdict='+'.join(parts),
               detail=dict(B=B, T=T, margin=d_bt, fresh_B=(s.get('fresh') or {}).get('em'), fresh_T=(t.get('fresh') or {}).get('em'), B_s=tb, T_s=tt, trend=trend,
                           vote=ev.get('vote'), select_p=ev.get('select_p'), ver_select=ev.get('ver_select'), ver_prior=ev.get('ver_prior'), ver_prior_half=ev.get('ver_prior_half'),
                           ver_informed=ev.get('ver_informed'), ver_owner=ev.get('ver_owner'), round1=v.get('round1'), union_any=(v.get('union') or {}).get('any_correct'),
                           own_arm=v.get('own'), know=v.get('know'), auroc_informed=(by_rel.get('informed') or {}).get('auroc'), auroc_ignorant=(by_rel.get('ignorant') or {}).get('auroc'),
                           auroc_all=(by_rel.get('all') or {}).get('auroc'), flat_when_ignorant=(v.get('verifier') or {}).get('flat_when_ignorant'),
                           text_vote=t['val'].get('vote'), text_em_agent=t['val'].get('em_agent'), bytes_B=(v.get('bytes') or {}).get('message_bytes'), bytes_T=(t['val'].get('bytes') or {}).get('message_bytes'),
                           train_complete=s.get('train_complete'), steps=s.get('steps'), steps_planned=s.get('steps_planned')))
    if s.get('train_complete') is False:
        out['notes'] = [f'verifier training stopped early: {s.get("steps")}/{s.get("steps_planned")} steps (recorded)']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', required=True)
    ap.add_argument('--runs', required=True, help='the L7 runs dir (markers are written here)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    r = judge(args.job, args.runs)
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
        print('  ' + ', '.join(f'{k} {v}' for k, v in r['detail'].items() if not isinstance(v, (dict, list))))
    if r.get('notes'):
        print('  notes: ' + '; '.join(r['notes']))
    if r['problems']:
        print('  problems: ' + '; '.join(r['problems']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
