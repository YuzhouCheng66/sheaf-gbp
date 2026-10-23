"""Pre-registered analysis of batch N (v375 data-scale), written 2026-10-23 ~01:40 UTC BEFORE any v375 result exists (Claude).
Primary (from chain_day4_N.sh): D2400 final VAL R8 wall F1 (equal weight over the 6 rooms x N strata, final weights, NOT best.pt)
  >= same-chart same-seed 600-house control + 5 pp on BOTH seeds => data-limited prior; < 2 pp => not data-limited; else ambiguous.
Completion gate (review M1): any arm or control with updates != 43200 or epochs_completed != its --epochs is INCONCLUSIVE.
Secondary (review M3, near-duplicate stratified): the same gain recomputed on VAL-panel houses with NO layout near-copy in EITHER the
  arm's or the control's training set (layout screen of code_review_v375_datascale.md: wall-boundary F1 with 1-cell tolerance, max
  over the 8 dihedral transforms, flag at >= 0.90; neardup2.json). If the primary passes but the near-copy-free gain is below 2 pp,
  the conclusion is "gain is (partly) near-duplicate memorisation", not "data-limited prior". Fresh panel (indices 8-15) reported the
  same way, descriptively. K0 reported alongside.
python analyze_v375.py --world houseexpo_fit|houseexpo --runs runs --neardup reports/neardup2_v375.json --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np

FLAG = .90


def eq_weight(rows, k, metric='wall_f1', drop=frozenset()):
    strata = {}
    for r in rows:
        if (r['spec']['rooms'], r['spec']['index']) in drop:
            continue
        v = r['hops'][str(k)][metric]
        if v is not None:
            strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(v)
    return float(np.mean([np.mean(v) for v in strata.values()])) if strata else None, {f'r{a}/N{b}': len(v) for (a, b), v in sorted(strata.items())}


def load(run):
    fin = json.load(open(run / 'final.json')); meta = json.load(open(run / 'run.json'))
    out = dict(run=str(run), updates=fin.get('updates'), epochs_completed=fin.get('epochs_completed'), epochs=meta['epochs'], initial_weights_sha256=meta['initial_weights_sha256'])
    for p in ('val', 'fresh'):
        f = run / 'validation' / f'final-{p}.json'
        out[p] = json.load(open(f))['rows'] if f.exists() else None
    out['complete'] = out['updates'] == 43200 and out['epochs_completed'] == out['epochs'] and out['val'] is not None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--world', required=True); ap.add_argument('--runs', default='runs'); ap.add_argument('--neardup', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); R = Path(args.runs)
    ctrl_dir = R / ('v372-hxfit-train/hxfit-long' if args.world == 'houseexpo_fit' else 'v367-hx-train/hx-long')
    nd = json.load(open(args.neardup))['val']
    flagged = {K: {(h['rooms'], h['index']) for h in nd if h[K]['f1'] >= FLAG} for K in ('k200', 'k2400', 'k4800')}
    arms = [('d2400', 342101, 'k2400'), ('d2400', 342202, 'k2400'), ('d4800', 342101, 'k4800')]
    res = dict(world=args.world, flag_threshold=FLAG, flagged={K: sorted(map(list, v)) for K, v in flagged.items()}, comparisons=[])
    for arm, seed, K in arms:
        a = load(R / 'v375-datascale' / f'{arm}-s{seed}'); c = load(Path(f'{ctrl_dir}-s{seed}'))
        row = dict(arm=f'{arm}-s{seed}', control=c['run'], arm_complete=a['complete'], control_complete=c['complete'],
                   same_initial_weights=a['initial_weights_sha256'] == c['initial_weights_sha256'])
        drop = flagged[K] | flagged['k200']
        for p in ('val', 'fresh'):
            if a[p] is None or c[p] is None:
                continue
            for k in (0, 8):
                va, _ = eq_weight(a[p], k); vc, _ = eq_weight(c[p], k)
                ra, na = eq_weight(a[p], k, drop=drop); rc, _ = eq_weight(c[p], k, drop=drop)
                row[f'{p}_R{k}'] = dict(arm=va, control=vc, gain_pp=100 * (va - vc), nearcopy_free=dict(arm=ra, control=rc, gain_pp=100 * (ra - rc), houses_dropped=len(drop), rows_per_stratum=na))
        row['verdict_input'] = 'INCONCLUSIVE (incomplete run)' if not (a['complete'] and c['complete']) else 'ok'
        res['comparisons'].append(row)
    d24 = [r for r in res['comparisons'] if r['arm'].startswith('d2400')]
    if all(r['verdict_input'] == 'ok' for r in d24) and len(d24) == 2:
        g = [r['val_R8']['gain_pp'] for r in d24]; gf = [r['val_R8']['nearcopy_free']['gain_pp'] for r in d24]
        if min(g) >= 5:
            v = 'data-limited prior (primary passes on both seeds)'
            if max(gf) < 2:
                v = 'gain is (partly) near-duplicate memorisation: primary passes but near-copy-free gain < 2 pp'
        elif max(g) < 2:
            v = 'not data-limited (gain < 2 pp)'
        else:
            v = 'ambiguous (between thresholds or seeds disagree)'
        res['primary_verdict'] = dict(verdict=v, gains_pp=g, nearcopy_free_gains_pp=gf)
    else:
        res['primary_verdict'] = dict(verdict='INCONCLUSIVE (an arm or control did not complete 43,200 updates)')
    json.dump(res, open(args.out, 'w'), indent=1)
    for r in res['comparisons']:
        print(r['arm'], r['verdict_input'], 'init-same' if r['same_initial_weights'] else 'INIT-DIFFERS',
              {k: (round(v['gain_pp'], 1), round(v['nearcopy_free']['gain_pp'], 1), round(v['arm'], 3), round(v['control'], 3)) for k, v in r.items() if k.endswith(('R0', 'R8'))})
    print('PRIMARY:', res['primary_verdict'])


if __name__ == '__main__':
    main()
