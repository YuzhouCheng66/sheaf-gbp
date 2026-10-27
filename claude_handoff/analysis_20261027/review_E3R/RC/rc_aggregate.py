"""aggregate R-C probes: pooled R8 furniture F1 (component rows expanded by size, as G._pooled_f1(G._expand)), pooled free recall,
furniture F1 by noisy multiplicity k, bytes per house (median / mean)."""
import os, sys, json, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def load(d):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, d, '*.json')))]


def main():
    recs = load('out') + load('out2') + load('out3') + load('out4')
    cells = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(tp=0, fp=0, fn=0, fh=0.0, ft=0.0, kf=np.zeros((6, 3), np.int64), surf=[], ftol=[], frow=[])))
    byt = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        seed = r['file'].split('_s')[-1].split('.')[0]; key = (r['seed_tag'], seed, r.get('tier', 'cam_nf'))
        for arm, rows in r['arms'].items():
            c = cells[key][arm]
            for x in rows:
                m = x['m']; c['tp'] += x['tp'] * m; c['fp'] += x['fp'] * m; c['fn'] += x['fn'] * m
                if x['free'] is not None:
                    c['fh'] += x['free'] * x['nfree'] * m; c['ft'] += x['nfree'] * m
                c['surf'] += [x['surf']] * m
                if x['tp'] + x['fp'] + x['fn']:
                    c['frow'] += [f1(x['tp'], x['fp'], x['fn'])] * m
                if x['ftol'] is not None:
                    c['ftol'] += [x['ftol']] * m
            c['kf'] += np.asarray(r['kf'][arm], np.int64)
        for k, v in r.get('bytes', {}).items():
            if k != 'n_directed':
                byt[key]['wire_' + k].append(v)
        for k, v in r.get('own_bytes', {}).items():
            byt[key]['own_' + k].append(v)
    out = {}
    for key in sorted(cells):
        tag = '|'.join(key); out[tag] = {}
        nh = None
        for arm, c in cells[key].items():
            kf = c['kf']
            out[tag][arm] = dict(furn_f1_rowmean=round(float(np.mean(c['frow'])), 4), furn_f1_pooled=round(f1(c['tp'], c['fp'], c['fn']), 4), free_recall=round(c['fh'] / max(c['ft'], 1), 4),
                                 surface_f1_mean=round(float(np.mean(c['surf'])), 4), furn_tol1_mean=round(float(np.mean(c['ftol'])), 4),
                                 k1=round(f1(*kf[1]), 4), k2=round(f1(*kf[2]), 4), k3_4=round(f1(*(kf[3] + kf[4])), 4), k5p=round(f1(*kf[5]), 4),
                                 fp=int(c['fp']), fn=int(c['fn']))
        out[tag]['bytes'] = {k: dict(median=round(float(np.median(v))), mean=round(float(np.mean(v))), n=len(v)) for k, v in byt[key].items()}
    with open(os.path.join(HERE, 'rc_probe_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    for tag, d in out.items():
        print('==', tag)
        for arm, v in d.items():
            print(f'  {arm:8s} {v}')


if __name__ == '__main__':
    main()
