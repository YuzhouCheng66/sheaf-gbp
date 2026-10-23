"""Pre-registered analysis of batch Q (v382: training length), written 2026-10-23 ~05:30 UTC before any batch-Q run and before the
batch-P result (Claude).
Arms: D4800 for 4 epochs (57,600 updates) vs D4800 for 3 epochs (43,200 updates; batch P), square house-fitted chart (houseexpo_fit,
the chart kept by batch O), same 14,400 TRAIN houses, same seeds 342101 / 342202, same trainer code path (train_v378.py); the cosine
schedule spans each run's own total updates (that is the change: a longer schedule).
Primary: final VAL R8 wall F1 (equal weight over 6 strata, final weights) 4-epoch >= 3-epoch + 1 pp on BOTH seeds => still under-trained
(longer schedules help); |gain| < 1 pp on both => converged for this model size; else ambiguous. Completion gate (57,600 resp. 43,200
updates, epochs as requested) and identity guard (world houseexpo_fit, recipe args except epochs, initial weights, cache per_stratum
4800 for both) else INCONCLUSIVE. K0 and fresh reported alongside.
python analyze_v382.py --runs runs --out <json>
"""
import argparse, json
from pathlib import Path
from analyze_v375 import eq_weight

RECIPE_ARGS = ('lr', 'fusion', 'readout', 'precision', 'wire', 'mu_bits', 'q_bits', 'dt_trunc', 'consistency', 'edge_select', 'size', 'arm', 'seed', 'world')


def load(run, updates):
    fin = json.load(open(run / 'final.json')); meta = json.load(open(run / 'run.json'))
    out = dict(run=str(run), updates=fin.get('updates'), epochs_completed=fin.get('epochs_completed'), epochs=meta['epochs'], args=meta['args'],
               initial_weights_sha256=meta['initial_weights_sha256'],
               per_stratum=json.load(open(Path(meta['train_cache']['dir']) / 'manifest.json'))['per_stratum'] if meta.get('train_cache') else None)
    for p in ('val', 'fresh'):
        f = run / 'validation' / f'final-{p}.json'
        out[p] = json.load(open(f))['rows'] if f.exists() else None
    out['complete'] = out['updates'] == updates and out['epochs_completed'] == out['epochs'] and out['val'] is not None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='runs'); ap.add_argument('--out', required=True)
    args = ap.parse_args(); R = Path(args.runs)
    res = dict(comparisons=[])
    for seed in (342101, 342202):
        a = load(R / 'v382-long' / f'd4800e4-s{seed}', 57600); c = load(R / 'v379-dose' / f'd4800-s{seed}', 43200)
        diffs = {k: (a['args'].get(k), c['args'].get(k)) for k in RECIPE_ARGS if a['args'].get(k) != c['args'].get(k)}
        ok = (a['complete'] and c['complete'] and a['args'].get('world') == 'houseexpo_fit' and not diffs and a['initial_weights_sha256'] == c['initial_weights_sha256']
              and a['per_stratum'] == 4800 and c['per_stratum'] == 4800 and a['epochs'] == 4 and c['epochs'] == 3)
        row = dict(seed=seed, arm=a['run'], control=c['run'], ok=ok, recipe_diffs=diffs)
        for p in ('val', 'fresh'):
            if a[p] is None or c[p] is None:
                continue
            for k in (0, 8):
                va, _ = eq_weight(a[p], k); vc, _ = eq_weight(c[p], k)
                row[f'{p}_R{k}'] = dict(e4=va, e3=vc, gain_pp=100 * (va - vc))
        res['comparisons'].append(row)
    if all(r['ok'] for r in res['comparisons']):
        g = [r['val_R8']['gain_pp'] for r in res['comparisons']]
        v = 'still under-trained (longer helps)' if min(g) >= 1 else ('converged at 43,200 updates' if max(abs(x) for x in g) < 1 else 'ambiguous')
        res['primary_verdict'] = dict(verdict=v, gains_pp=g)
    else:
        res['primary_verdict'] = dict(verdict='INCONCLUSIVE (incomplete run or identity mismatch)')
    json.dump(res, open(args.out, 'w'), indent=1)
    for r in res['comparisons']:
        print(r['seed'], 'ok' if r['ok'] else 'NOT-OK', r['recipe_diffs'], {k: (round(v['gain_pp'], 1), round(v['e4'], 3), round(v['e3'], 3)) for k, v in r.items() if k.endswith(('R0', 'R8'))})
    print('PRIMARY:', res['primary_verdict'])


if __name__ == '__main__':
    main()
