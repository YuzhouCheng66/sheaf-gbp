"""Pre-registered analysis of batch T (v386: decoder capacity as a single variable). Claude, 2026-10-23, written before any batch-T run.
Arms (per seed 342101 / 342202): WIDE = train_v386.py --decoder-width 2.0 (decoder 2,491,330 params) trained from scratch with the
batch-Q recipe (houseexpo_fit, 14,400-house cache, 4 epochs = 57,600 updates, lr 3e-4, max fusion, posterior_only, sealed wire);
CONTROL = batch Q itself (runs/v382-long/d4800e4-s<seed>, decoder 630,498 params, same recipe, same seed => same encoder/head init and
same data order; initial weights differ only in the decoder, which is the intervention).
Primary: final VAL R8 wall F1 (6-stratum equal weight, final weights). WIDE >= CONTROL + 1 pp on BOTH seeds => decoder capacity is a
lever at this budget; |gain| < 1 pp on both => not a lever at this budget; else ambiguous. Protection: VAL R8 unknown-known FPR may not
rise by more than 0.3 pp. Reported alongside: K0, fresh R8/K0, 8+-room stratum. Completion gate: 57,600 updates and 4 epochs on both
arms; identity guard: world, cache per_stratum 4800, recipe args equal except decoder_width (control: absent or 1.0; arm: 2.0),
encoder+head tensors of initial.pt identical (bit-exact) => else INCONCLUSIVE.
python analyze_v386.py --runs runs --out <json>
"""
import argparse, json
from pathlib import Path
import torch
from analyze_v375 import eq_weight

RECIPE_ARGS = ('lr', 'fusion', 'readout', 'precision', 'wire', 'mu_bits', 'q_bits', 'dt_trunc', 'consistency', 'edge_select', 'size', 'arm', 'seed', 'world', 'epochs', 'train_cache')


def load(run, updates):
    fin = json.load(open(run / 'final.json')); meta = json.load(open(run / 'run.json'))
    out = dict(run=str(run), updates=fin.get('updates'), epochs_completed=fin.get('epochs_completed'), epochs=meta['epochs'], args=meta['args'],
               decoder_width=meta.get('decoder_width', meta['args'].get('decoder_width', 1.0)), parameters=meta.get('parameters'),
               per_stratum=json.load(open(Path(meta['train_cache']['dir']) / 'manifest.json'))['per_stratum'] if meta.get('train_cache') else None,
               final=fin)
    for p in ('val', 'fresh'):
        f = run / 'validation' / f'final-{p}.json'
        out[p] = json.load(open(f))['rows'] if f.exists() else None
    out['complete'] = out['updates'] == updates and out['epochs_completed'] == out['epochs'] and out['val'] is not None
    return out


def encoder_init_identical(a, c):
    sa = torch.load(Path(a) / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=False)['state_dict']
    sc = torch.load(Path(c) / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=False)['state_dict']
    keys = [k for k in sc if k.startswith('trunk.') or k.startswith('head.')]
    return bool(keys) and all(k in sa and torch.equal(sa[k], sc[k]) for k in keys), len(keys)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='runs'); ap.add_argument('--out', required=True)
    ap.add_argument('--arm-dir', default='v386-decoder'); ap.add_argument('--arm-name', default='wide')
    args = ap.parse_args(); R = Path(args.runs)
    res = dict(comparisons=[])
    for seed in (342101, 342202):
        a = load(R / args.arm_dir / f'{args.arm_name}-s{seed}', 57600); c = load(R / 'v382-long' / f'd4800e4-s{seed}', 57600)
        diffs = {k: (a['args'].get(k), c['args'].get(k)) for k in RECIPE_ARGS if a['args'].get(k) != c['args'].get(k)}
        enc_same, n_enc = encoder_init_identical(a['run'], c['run'])
        ok = (a['complete'] and c['complete'] and a['args'].get('world') == 'houseexpo_fit' and not diffs and a['per_stratum'] == 4800 and c['per_stratum'] == 4800
              and a['epochs'] == 4 and c['epochs'] == 4 and float(a['decoder_width']) == 2.0 and float(c['decoder_width']) == 1.0 and enc_same)
        row = dict(seed=seed, arm=a['run'], control=c['run'], ok=ok, recipe_diffs=diffs, encoder_init_identical=enc_same, encoder_tensors=n_enc,
                   decoder_params=dict(arm=(a['parameters'] or {}).get('decoder'), control=(c['parameters'] or {}).get('decoder')))
        for p in ('val', 'fresh'):
            if a[p] is None or c[p] is None:
                continue
            for k in (0, 8):
                va, _ = eq_weight(a[p], k); vc, _ = eq_weight(c[p], k)
                row[f'{p}_R{k}'] = dict(wide=va, control=vc, gain_pp=100 * (va - vc))
        fa, fc = a['final'].get('val', {}).get('8', {}).get('all', {}), c['final'].get('val', {}).get('8', {}).get('all', {})
        if fa and fc:
            row['val_R8_fpr_delta_pp'] = 100 * (fa['unknown_known_fpr'] - fc['unknown_known_fpr'])
            row['val_R8_rooms3'] = {s: dict(wide=a['final']['val']['8']['strata'][s]['wall_f1'], control=c['final']['val']['8']['strata'][s]['wall_f1']) for s in ('rooms3/N8', 'rooms3/N16')}
        res['comparisons'].append(row)
    if all(r['ok'] for r in res['comparisons']):
        g = [r['val_R8']['gain_pp'] for r in res['comparisons']]
        v = 'decoder capacity is a lever at this budget' if min(g) >= 1 else ('not a lever at this budget' if max(abs(x) for x in g) < 1 else 'ambiguous')
        fpr = [r.get('val_R8_fpr_delta_pp', 0.) for r in res['comparisons']]
        if max(fpr) > 0.3:
            v += ' BUT FPR protection violated'
        res['primary_verdict'] = dict(verdict=v, gains_pp=g, fpr_delta_pp=fpr)
    else:
        res['primary_verdict'] = dict(verdict='INCONCLUSIVE (incomplete run or identity mismatch)', detail=[dict(seed=r['seed'], ok=r['ok'], diffs=r['recipe_diffs'], enc=r['encoder_init_identical']) for r in res['comparisons']])
    json.dump(res, open(args.out, 'w'), indent=1)
    for r in res['comparisons']:
        print(r['seed'], 'ok' if r['ok'] else 'NOT-OK', r['recipe_diffs'], 'enc_same', r['encoder_init_identical'],
              {k: (round(v['gain_pp'], 2), round(v['wide'], 4), round(v['control'], 4)) for k, v in r.items() if k.endswith(('R0', 'R8'))})
    print('PRIMARY:', res['primary_verdict'])


if __name__ == '__main__':
    main()
