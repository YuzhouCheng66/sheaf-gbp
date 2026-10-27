"""R-A (E1 review): read-only extraction from the frozen G4-E1 job files (no re-evaluation). Pooled furniture TP/FP/FN, precision /
recall, and an ADDITIVE loss split by stratum: 1 - F1_pooled = sum_s (FP_s + FN_s) / (2 TP + FP + FN).
usage: python -B ra_e1_extract.py > ra_e1_extract.json"""
import os, sys, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, '..', 'g4', 'e1', 'jobs')
M_LAB = ['m1', 'm2', 'm3', 'm4', 'm5+']; H_LAB = ['h<.3', 'h.3-.9', 'h>=.9']


def acc_rows(rows, out, key_strata=('h_furn', 'm_furn')):
    for r in rows:
        out['tp'] += r['furn_tp']; out['fp'] += r['furn_fp']; out['fn'] += r['furn_fn']
        out['f1s'].append(r['furn_f1']); out['tol1'].append(r['furn_f1_tol1']); out['free'].append(r['free_recall'])
        out['fpr'].append(r['unknown_known_fpr'])
        for k in key_strata:
            if r.get('strata') and k in r['strata']:
                a = np.asarray(r['strata'][k], np.int64)
                out[k] = a if out.get(k) is None else out[k] + a


def finish(o):
    tp, fp, fn = o['tp'], o['fp'], o['fn']; den = 2 * tp + fp + fn
    res = dict(f1_mean=float(np.mean([x for x in o['f1s'] if x is not None])), f1_pooled=2 * tp / den, prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1),
               tol1_mean=float(np.mean([x for x in o['tol1'] if x is not None])), free=float(np.mean([x for x in o['free'] if x is not None])),
               fpr=float(np.mean([x for x in o['fpr'] if x is not None])), loss_fp=fp / den, loss_fn=fn / den)
    for k, lab in (('m_furn', M_LAB), ('h_furn', H_LAB)):
        if o.get(k) is not None:
            a = o[k]
            res[k] = {lab[i]: dict(n_t=int(a[i, 0] + a[i, 2]), f1=2 * a[i, 0] / max(2 * a[i, 0] + a[i, 1] + a[i, 2], 1), prec=a[i, 0] / max(a[i, 0] + a[i, 1], 1),
                                   rec=a[i, 0] / max(a[i, 0] + a[i, 2], 1), loss_fp=a[i, 1] / den, loss_fn=a[i, 2] / den) for i in range(len(lab))}
    return res


def new():
    return dict(tp=0, fp=0, fn=0, f1s=[], tol1=[], free=[], fpr=[], h_furn=None, m_furn=None)


def main():
    files = sorted(glob.glob(os.path.join(JOBS, '*.json')))
    acc = collections.defaultdict(new); b5 = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        j = json.load(open(f))
        if j.get('skipped'):
            continue
        ps = (j['panel'], j['seed'])
        for cn, rec in j['conds'].items():
            for I in ('V3', 'V1'):
                r = rec['by_int'][I]
                acc_rows(r['K0'], acc[(ps, cn, I, 'K0')])
                for fu, rows in r['R8'].items():
                    if fu in ('sum', 'sum_truepose', 'maxabs'):
                        acc_rows([x for x, k in rows for _ in range(k)], acc[(ps, cn, I, 'R8_' + fu)])
                if 'B5' in r and cn == 'cam':
                    for key, v in r['B5'].items():
                        fam, tau, m, f = key.split('|')
                        if fam == 'cc' and m == '4' and f == '111':
                            rows = [x for x, k in v['pub']['sum']['R8'] for _ in range(k)]
                            b5[(ps, I)][tau].append((sum(x['furn_f1'] * 1 for x in rows) / len(rows), v['pub']['sum']['rnet']))
    out = {}
    for (ps, cn, I, what), o in sorted(acc.items()):
        out[f'{ps[0]}|s{ps[1]}|{cn}|{I}|{what}'] = finish(o)
    for (ps, I), d in sorted(b5.items()):
        out[f'{ps[0]}|s{ps[1]}|cam|{I}|B5_cc_m4_sum_by_tau'] = {t: dict(furn_mean=float(np.mean([x[0] for x in v])), rnet_med=float(np.median([x[1] for x in v])))
                                                                for t, v in d.items()}
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def b5_hulls():
    """from summary.json (already aggregated by the frozen code): B5 hull points on cam V3 (pub = cam, truepose = cam_nf maps)"""
    sys.path.insert(0, HERE)
    s = json.load(open(os.path.join(HERE, '..', 'g4', 'e1', 'summary.json')))
    def hull(points, x0):
        pts = [(x, y) for x, y in points if x is not None and y is not None]; best = None
        for x, y in pts:
            if x <= x0: best = y if best is None else max(best, y)
        if best is None: return None
        for (x1, y1) in pts:
            for (x2, y2) in pts:
                if x1 < x0 < x2: best = max(best, y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))
        return best
    out = {}
    for p in ('val', 'fresh'):
        for sd in ('1', '2'):
            d = s['panels'][p]['seeds'][sd]['conds']['cam']['V3']
            for ms in ('pub', 'truepose'):
                pts = d[f'B5_hull_{ms}']['points']
                B5 = d['B5']
                out[f'{p}|s{sd}|{ms}'] = dict(at_3S0=hull(pts, 3 * 6066.0), at_6S0=hull(pts, 6 * 6066.0), at_quarterB4=hull(pts, 0.25 * d['rnet_median']['B4_relay']),
                                              min_rate=min(x for x, _ in pts), best_any=max(y for _, y in pts),
                                              tau0_noDel_sum=B5['cc|0|4|111'][f'{ms}_sum']['R8']['furn_f1'], tau0_sum_rate=B5['cc|0|4|111'][f'{ms}_sum']['rnet_median'],
                                              cheapest=sorted([(round(x), round(y, 3)) for x, y in pts])[:6])
            cn = s['panels'][p]['seeds'][sd]['conds']['cam_nf']['V3']
            out[f'{p}|s{sd}|cam_nf_B4'] = dict(rnet=cn['rnet_median']['B4_relay'], furn=cn['R8']['sum']['furn_f1'], k0=cn['K0']['furn_f1'],
                                               free=cn['R8']['sum']['free_recall'], fpr=cn['R8']['sum']['unknown_known_fpr'])
    return out


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'hull':
    json.dump(b5_hulls(), sys.stdout, indent=1)
