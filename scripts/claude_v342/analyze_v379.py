"""Pre-registered analysis of batch P (v379: data dose-response, 14,400 vs 7,200 TRAIN houses), written 2026-10-23 ~03:45 UTC before
any batch-P run and before the batch-O result (Claude).
Arms: D4800 (4,800 houses per stratum, 3 epochs) vs D2400 (2,400 per stratum, 6 epochs), both 43,200 updates, same chart (the chart
batch O selects: houseexpo_rect if O's verdict is "adopt rectangular chart", else houseexpo_fit), same seeds 342101 / 342202, same
trainer code path. Control = the D2400 run of that chart (batch O rect-d2400-s<seed> or batch N d2400-s<seed>).
Primary: final VAL R8 wall F1 (equal weight over 6 strata, final weights) D4800 >= D2400 + 1 pp on BOTH seeds => more data still helps
(not saturated at 7,200); |gain| < 1 pp on both => saturated; else ambiguous. Completion gate (43,200 updates, epochs as requested) and
identity guard (world, recipe args, initial weights, cache per_stratum 4800 vs 2400) else INCONCLUSIVE. Near-copy-free secondary as in
analyze_v375 (layout screen >= 0.90 against the k4800 set vs the k2400 set; neardup2_v375.json). K0 and fresh reported alongside.
python analyze_v379.py --world houseexpo_rect|houseexpo_fit --runs runs --neardup neardup2_v375.json --out <json>
"""
import argparse, json
from pathlib import Path
from analyze_v375 import eq_weight

RECIPE_ARGS = ('lr', 'fusion', 'readout', 'precision', 'wire', 'mu_bits', 'q_bits', 'dt_trunc', 'consistency', 'edge_select', 'size', 'arm', 'seed', 'world')


def load(run):
    fin = json.load(open(run / 'final.json')); meta = json.load(open(run / 'run.json'))
    out = dict(run=str(run), updates=fin.get('updates'), epochs_completed=fin.get('epochs_completed'), epochs=meta['epochs'], args=meta['args'],
               initial_weights_sha256=meta['initial_weights_sha256'],
               per_stratum=json.load(open(Path(meta['train_cache']['dir']) / 'manifest.json'))['per_stratum'] if meta.get('train_cache') else None)
    for p in ('val', 'fresh'):
        f = run / 'validation' / f'final-{p}.json'
        out[p] = json.load(open(f))['rows'] if f.exists() else None
    out['complete'] = out['updates'] == 43200 and out['epochs_completed'] == out['epochs'] and out['val'] is not None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--world', required=True, choices=['houseexpo_rect', 'houseexpo_fit']); ap.add_argument('--runs', default='runs')
    ap.add_argument('--neardup', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); R = Path(args.runs)
    nd = json.load(open(args.neardup))['val']
    flagged = {K: {(h['rooms'], h['index']) for h in nd if h[K]['f1'] >= .90} for K in ('k2400', 'k4800')}
    drop = flagged['k2400'] | flagged['k4800']
    res = dict(world=args.world, comparisons=[], nearcopy_dropped=sorted(map(list, drop)))
    for seed in (342101, 342202):
        a = load(R / 'v379-dose' / f'd4800-s{seed}')
        c = load(R / (('v378-rect/rect-d2400' if args.world == 'houseexpo_rect' else 'v375-datascale/d2400') + f'-s{seed}'))
        diffs = {k: (a['args'].get(k), c['args'].get(k)) for k in RECIPE_ARGS if a['args'].get(k) != c['args'].get(k)}
        ok = (a['complete'] and c['complete'] and a['args'].get('world') == args.world and not diffs and a['initial_weights_sha256'] == c['initial_weights_sha256']
              and a['per_stratum'] == 4800 and c['per_stratum'] == 2400 and a['epochs'] == 3 and c['epochs'] == 6)
        row = dict(seed=seed, arm=a['run'], control=c['run'], ok=ok, recipe_diffs=diffs)
        for p in ('val', 'fresh'):
            if a[p] is None or c[p] is None:
                continue
            for k in (0, 8):
                va, _ = eq_weight(a[p], k); vc, _ = eq_weight(c[p], k)
                fa, _ = eq_weight(a[p], k, drop=drop); fc, _ = eq_weight(c[p], k, drop=drop)
                row[f'{p}_R{k}'] = dict(d4800=va, d2400=vc, gain_pp=100 * (va - vc), nearcopy_free_gain_pp=100 * (fa - fc))
        res['comparisons'].append(row)
    if all(r['ok'] for r in res['comparisons']):
        g = [r['val_R8']['gain_pp'] for r in res['comparisons']]
        v = 'more data still helps' if min(g) >= 1 else ('saturated at 7,200 houses' if max(abs(x) for x in g) < 1 else 'ambiguous')
        res['primary_verdict'] = dict(verdict=v, gains_pp=g, nearcopy_free_gains_pp=[r['val_R8']['nearcopy_free_gain_pp'] for r in res['comparisons']])
    else:
        res['primary_verdict'] = dict(verdict='INCONCLUSIVE (incomplete run or identity mismatch)')
    json.dump(res, open(args.out, 'w'), indent=1)
    for r in res['comparisons']:
        print(r['seed'], 'ok' if r['ok'] else 'NOT-OK', r['recipe_diffs'], {k: (round(v['gain_pp'], 1), round(v['d4800'], 3), round(v['d2400'], 3)) for k, v in r.items() if k.endswith(('R0', 'R8'))})
    print('PRIMARY:', res['primary_verdict'])


if __name__ == '__main__':
    main()
