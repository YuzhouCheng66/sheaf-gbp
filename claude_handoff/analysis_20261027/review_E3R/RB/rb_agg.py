"""aggregate R-B probe (a) / (c) outputs: per seed, the E3-R headline statistic (mean of component-expanded R8 rows) + pooled F1,
paired per-house differences vs the full arm, and bytes (mean / median per house, KB)"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def expand(rows):
    return [r for r, k in rows for _ in range(k)]


def stat(rows, key='furn_f1'):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def pooled(rows):
    tp = sum(r['furn_tp'] for r in rows); fp = sum(r['furn_fp'] for r in rows); fn = sum(r['furn_fn'] for r in rows)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def agg_a(panel='val'):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_a', panel + '_*.json')))]
    out = {}
    for seed in ('s1', 's2'):
        R = [r for r in recs if r['file'].endswith(f'_{seed}.npz')]
        if not R:
            continue
        arms = list(R[0]['arms'].keys()); o = {'n_houses': len(R)}
        for arm in arms:
            rows = [x for r in R for x in expand(r['arms'][arm]['R8_net'])]
            rowsl = [x for r in R for x in expand(r['arms'][arm]['R8_lin'])]
            full = arm.split(':')[0] + ':full'
            dh = [np.mean([x['furn_f1'] for x in expand(r['arms'][arm]['R8_net'])]) - np.mean([x['furn_f1'] for x in expand(r['arms'][full]['R8_net'])]) for r in R]
            d = dict(net_furn=stat(rows), net_furn_pooled=pooled(rows), net_free=stat(rows, 'free_recall'), net_surf=stat(rows, 'surface_f1'),
                     lin_furn=stat(rowsl), d_vs_full_pp_mean=100 * float(np.mean(dh)), d_vs_full_pp_worst=100 * float(np.min(dh)))
            if 'K0_net' in R[0]['arms'][arm]:
                d['K0_net_furn'] = stat([x for r in R for x in r['arms'][arm]['K0_net']])
            if 'bytes' in R[0]['arms'][arm]:
                for k in R[0]['arms'][arm]['bytes']:
                    v = [r['arms'][arm]['bytes'][k] / 1000 for r in R if k in r['arms'][arm]['bytes']]
                    d[f'KB_{k}_mean'] = float(np.mean(v)); d[f'KB_{k}_med'] = float(np.median(v))
            o[arm] = d
        out[seed] = o
    return out


def agg_c():
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out_c', '*.json')))]
    if not recs:
        return None
    o = dict(files=[r['file'] for r in recs], clean_recon_mismatch_agent0=[r['clean_recon_mismatch_agent0'] for r in recs])
    A = [a for r in recs for a in r['agents']]
    for k in ('boxes', 'boxes_clean', 'plan_cells', 'plan_cells_clean', 'tern_ctx', 'tern_ctx_clean', 'mask_ctx', 'mask_ctx_clean', 'occ_vox', 'occ_vox_clean'):
        o[k + '_med'] = float(np.median([a[k] for a in A]))
    for w in ('w3', 'hit'):
        for k in ('known_act', 'known_rec', 'miss', 'extra', 'occ_diff', 'diff', 'resid_bits'):
            o[f'{w}_{k}_med'] = float(np.median([a[w][k] for a in A]))
        o[f'{w}_resid_KB_med'] = o[f'{w}_resid_bits_med'] / 8000
    for k in ('k', 'n_occ', 'union_mask'):
        o[f'field_{k}_KB_mean'] = float(np.mean([sum(f[k] for f in r['field']) for r in recs])) / 1000
    o['own_tern_KB_house_mean'] = float(np.mean([sum(a['tern_ctx'] for a in r['agents']) for r in recs])) / 1000
    o['own_tern_clean_KB_house_mean'] = float(np.mean([sum(a['tern_ctx_clean'] for a in r['agents']) for r in recs])) / 1000
    for name in ('exact', 'w3_recon', 'hit_recon', 'clean_full', 'clean_q3', 'clean_sign'):
        for seed in ('s1', 's2'):
            rows = [x for r in recs if r['file'].endswith(f'_{seed}.npz') for x in expand(r['R8'][name])]
            if rows:
                o[f'R8_{name}_{seed}_furn'] = stat(rows); o[f'R8_{name}_{seed}_free'] = stat(rows, 'free_recall')
    return o


if __name__ == '__main__':
    res = dict(a=agg_a(), a_fresh=agg_a('fresh'), c=agg_c())
    with open(os.path.join(HERE, 'rb_summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    for seed, o in list((res['a'] or {}).items()) + [('fresh_' + k, v) for k, v in (res['a_fresh'] or {}).items()]:
        print('==', seed, 'houses', o['n_houses'])
        for arm, d in o.items():
            if arm == 'n_houses':
                continue
            b = '' if 'KB_sum_mean' not in d else ' | KB sum %.0f cnt %.0f tern %.0f mask %.0f up %.0f+%.0f down_ro %.0f lev %s k0own %s' % (
                d['KB_sum_mean'], d['KB_count_mean'], d['KB_tern_mean'], d['KB_mask_mean'], d['KB_up_sum_mean'], d['KB_up_count_mean'],
                d['KB_down_readout_mean'], ('%.0f' % d['KB_levels_mean']) if 'KB_levels_mean' in d else '-',
                ('%.1f' % d['KB_k0_own_mean']) if 'KB_k0_own_mean' in d else '-')
            print('%-14s net %.4f (pooled %.4f) d %.2f pp worst %.2f | lin %.4f free %.4f K0 %s%s' % (
                arm, d['net_furn'], d['net_furn_pooled'], d['d_vs_full_pp_mean'], d['d_vs_full_pp_worst'], d['lin_furn'], d['net_free'],
                ('%.4f' % d['K0_net_furn']) if 'K0_net_furn' in d else '-', b))
    if res['c']:
        print(json.dumps(res['c'], indent=1))
