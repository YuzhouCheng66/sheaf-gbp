"""Pre-registered analysis of batch O (v378 rectangular house-fitted chart), written 2026-10-23 ~02:35 UTC before any v378 run (Claude).
One change vs the control: the public chart (rectangle fitted per axis, aspect <= 2) instead of the square house-fitted chart; same
recipe, same seeds, same TRAIN houses. Control = batch-N D2400 runs if the recipe is D2400, else the v372 600-house runs.
Primary: final VAL R8 wall F1 (equal weight over the 6 rooms x N strata, final weights) rect >= square + 2 pp on BOTH seeds => adopt the
rectangular chart; < 1 pp on both => no gain (keep the simpler square chart); otherwise ambiguous. Completion gate: updates must equal
43,200 and epochs_completed the requested epochs for every arm and control, else INCONCLUSIVE. K0, fresh panel and per-stratum values
reported alongside. Caveat stated in advance: raster cells become anisotropic, so per-cell wall F1 is not an equal-area metric and the
pixel scale differs per axis between the arms; the comparison is per-house accuracy on each arm's native raster.
Review code_review_v378_rect.md: M1 paired poses (v371 poses translated); M2 control identity (control world houseexpo_fit, arm world
houseexpo_rect, identical initial weights and recipe args, per_stratum 2400 on the d2400 path) else INCONCLUSIVE; secondary: gain on
chart-changed houses only and the unchanged-chart placebo (fixtures identical in both arms; any difference there is training-distribution).
python analyze_v378.py --recipe d2400|h600 --runs runs --source <sealed> --out <json>   (env HOUSEEXPO_JSON_DIR / HOUSEEXPO_INDEX)
"""
import argparse, json
from pathlib import Path
import numpy as np
from analyze_v375 import eq_weight


RECIPE_ARGS = ('epochs', 'lr', 'fusion', 'readout', 'precision', 'wire', 'mu_bits', 'q_bits', 'dt_trunc', 'consistency', 'edge_select', 'size', 'arm', 'seed')


def load(run, epochs):
    fin = json.load(open(run / 'final.json')); meta = json.load(open(run / 'run.json'))
    out = dict(run=str(run), updates=fin.get('updates'), epochs_completed=fin.get('epochs_completed'), epochs=meta['epochs'], args=meta['args'],
               initial_weights_sha256=meta['initial_weights_sha256'], train_cache=meta.get('train_cache'))
    for p in ('val', 'fresh'):
        f = run / 'validation' / f'final-{p}.json'
        out[p] = json.load(open(f))['rows'] if f.exists() else None
    out['complete'] = out['updates'] == 43200 and out['epochs_completed'] == out['epochs'] == epochs and out['val'] is not None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--recipe', required=True, choices=['d2400', 'h600']); ap.add_argument('--runs', default='runs'); ap.add_argument('--out', required=True)
    ap.add_argument('--source', required=True, help='sealed source root (for the chart computation of the placebo split)')
    args = ap.parse_args()
    import sys; src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent)); R = Path(args.runs); ep = 6 if args.recipe == 'd2400' else 72
    res = dict(recipe=args.recipe, comparisons=[])
    import mapping_houseexpo_data_v371 as v371, mapping_houseexpo_data_v378 as v378
    unchanged = set()
    for split, offs in (('VALIDATION', 0),):
        for rooms in (1, 2, 3):
            for idx in range(16):
                L = v371.world_spec(split, idx, rooms)['chart_m']; lx, ly = v378.world_spec(split, idx, rooms)['chart_xy_m']
                if lx == ly == L:
                    unchanged.add((rooms, idx))
    res['unchanged_chart_houses'] = sorted(map(list, unchanged))
    for seed in (342101, 342202):
        a = load(R / 'v378-rect' / f'rect-{args.recipe}-s{seed}', ep)
        ctrl = ('v375-datascale/d2400' if args.recipe == 'd2400' else 'v372-hxfit-train/hxfit-long') + f'-s{seed}'
        c = load(R / ctrl, ep)
        # review M2: the control must be the SQUARE house-fitted chart with the same recipe and initial weights, else INCONCLUSIVE
        ident = dict(arm_world=a['args'].get('world'), control_world=c['args'].get('world'),
                     same_initial_weights=a['initial_weights_sha256'] == c['initial_weights_sha256'],
                     recipe_diffs={k: (a['args'].get(k), c['args'].get(k)) for k in RECIPE_ARGS if a['args'].get(k) != c['args'].get(k)})
        if args.recipe == 'd2400':
            ka = json.load(open(Path(a['train_cache']['dir']) / 'manifest.json'))['per_stratum'] if a['train_cache'] else None
            kc = json.load(open(Path(c['train_cache']['dir']) / 'manifest.json'))['per_stratum'] if c['train_cache'] else None
            ident['per_stratum'] = (ka, kc)
        ok_ident = (ident['arm_world'] == 'houseexpo_rect' and ident['control_world'] == 'houseexpo_fit' and ident['same_initial_weights']
                    and not ident['recipe_diffs'] and (args.recipe != 'd2400' or ident['per_stratum'] == (2400, 2400)))
        row = dict(seed=seed, arm=a['run'], control=c['run'], complete=a['complete'] and c['complete'] and ok_ident, identity=ident, identity_ok=ok_ident)
        for p in ('val', 'fresh'):
            if a[p] is None or c[p] is None:
                continue
            for k in (0, 8):
                va, _ = eq_weight(a[p], k); vc, _ = eq_weight(c[p], k)
                strata = {}
                for rooms in (1, 2, 3):
                    sa = [r for r in a[p] if r['spec']['rooms'] == rooms]; sc = [r for r in c[p] if r['spec']['rooms'] == rooms]
                    strata[f'rooms{rooms}'] = (round(eq_weight(sa, k)[0], 4), round(eq_weight(sc, k)[0], 4))
                # secondary (review m1): chart-changed houses only, and the unchanged-chart placebo (identical fixtures with paired poses)
                ch_a, _ = eq_weight(a[p], k, drop=unchanged); ch_c, _ = eq_weight(c[p], k, drop=unchanged)
                keep_only = {(r['spec']['rooms'], r['spec']['index']) for r in a[p]} - unchanged
                pl_a, _ = eq_weight(a[p], k, drop=keep_only); pl_c, _ = eq_weight(c[p], k, drop=keep_only)
                row[f'{p}_R{k}'] = dict(rect=va, square=vc, gain_pp=100 * (va - vc), per_rooms_rect_square=strata,
                                        chart_changed_only=dict(rect=ch_a, square=ch_c, gain_pp=None if ch_a is None else 100 * (ch_a - ch_c)),
                                        unchanged_placebo=dict(rect=pl_a, square=pl_c, gain_pp=None if pl_a is None else 100 * (pl_a - pl_c), houses=len(unchanged)))
        res['comparisons'].append(row)
    if all(r['complete'] for r in res['comparisons']):
        g = [r['val_R8']['gain_pp'] for r in res['comparisons']]
        v = 'adopt rectangular chart' if min(g) >= 2 else ('no gain (keep square)' if max(g) < 1 else 'ambiguous')
        res['primary_verdict'] = dict(verdict=v, gains_pp=g)
    else:
        res['primary_verdict'] = dict(verdict='INCONCLUSIVE (a run did not complete 43,200 updates)')
    json.dump(res, open(args.out, 'w'), indent=1)
    for r in res['comparisons']:
        print(r['seed'], 'complete' if r['complete'] else 'INCOMPLETE', {k: (round(v['gain_pp'], 1), round(v['rect'], 3), round(v['square'], 3)) for k, v in r.items() if k.endswith(('R0', 'R8'))})
    print('PRIMARY:', res['primary_verdict'])


if __name__ == '__main__':
    main()
