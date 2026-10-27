"""Reviewer's independent recompute for G4-H1 (code review, 2026-10-27). TRAIN smoke npz only (never VAL).
Own code for: component sums S / k (table lookup), B6 / NET thresholds, region / target, R8 furniture F1 (E3-R expanded mean), zone
pooled F1, free recall, flip precision (S <= 0), phantom blocks (+ variants of the truth set), and a decomposition of the wrong flips.
Uses from the frozen code only: E.load_ckpt + E.net_logits (the network forward pass). Compares to the job JSON of the smoke run."""
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
sys.dont_write_bytecode = True
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa
import g4_e3r as E  # noqa  (frozen; only load_ckpt / net_logits used)
torch.set_num_threads(2)

AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
EXP = os.path.join(AN, 'g4', 'h1_export')
JOBS = os.path.join(AN, 'g4', 'h1_smoke', 'train_final', 'jobs')
INP = os.path.join(AN, 'review_E3R', 'inputs')
FIT = json.load(open(os.path.join(AN, 'g4', 'e1c', 'fit.json')))
S6 = ndimage.generate_binary_structure(3, 1)
C26 = np.ones((3, 3, 3), bool)
LMIN = -40


def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()


def d6(m):
    return ndimage.binary_dilation(m, structure=S6)


def comps_of(edges, n):
    par = list(range(n))
    def f(a):
        while par[a] != a:
            a = par[a]
        return a
    for i, j, _ in np.asarray(edges).reshape(-1, 3):
        par[f(int(i))] = f(int(j))
    g = {}
    for a in range(n):
        g.setdefault(f(a), []).append(a)
    return sorted(g.values())


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else None


def run(name, seed):
    p = os.path.join(EXP, f'{name}_s{seed}.npz')
    job = json.load(open(os.path.join(JOBS, name + '.json')))
    js = job['seeds'][str(seed)]
    out = dict(npz=os.path.basename(p), sha_match=sha(p) == js['file_sha256'])
    z = np.load(p)
    occ = z['occ'] == 1; slab = z['slab'].astype(bool); obj = z['obj'].astype(bool); priv = z['private']; L = z['L_k0_cam_nf']
    n = priv.shape[0]; comps = comps_of(z['edges'], n)
    out['comps_match'] = comps == [list(c) for c in js['components']]
    struct = occ & ~obj & ~slab
    region = ~slab & ~d6(struct)
    zone = d6(z['ood_mask'].astype(bool))
    tabs = {h: (np.asarray(FIT['halves'][h]['tiers']['cam_nf']['cq'], np.int64), int(FIT['halves'][h]['tiers']['cam_nf']['theta'])) for h in 'AB'}
    nets = {}
    for ts, h in (('342101', 'A'), ('342202', 'B')):
        ck, model = E.load_ckpt(os.path.join(INP, f's{ts}', 'final.pt'), 'cpu')
        assert list(ck['recipe']['evidence']['table']) == tabs[h][0].tolist() and int(ck['recipe']['evidence']['theta']) == tabs[h][1]
        j = json.load(open(os.path.join(INP, f's{ts}', 'threshold.json')))['j']
        nets[ts] = (model, j, h)
    known_l = L != -128
    dist = ndimage.distance_transform_cdt(~obj, metric='chessboard')
    res = {}
    acc = {}
    for ci, cm in enumerate(comps):
        w = len(cm)
        T = priv[cm].max(0)
        Tk = T >= 0; To = T == 1; Tf = T == 0
        tr = To & region; scope = Tk & region
        k = known_l[cm].sum(0).astype(np.int32)
        S = {}
        for h, (tab, th) in tabs.items():
            e = np.where(known_l[cm], tab[np.clip(L[cm].astype(np.int64) - LMIN, 0, len(tab) - 1)], 0)
            S[h] = e.sum(0).astype(np.int32)
        preds = {}
        for h, (tab, th) in tabs.items():
            preds[f'B6_{h}'] = np.where(k > 0, (S[h] > th).astype(np.int8), -1)
        for ts, (model, j, h) in nets.items():
            lg = E.net_logits(model, [S[h]], [k], 'cpu', 1)[0]
            t = -40.0 + j * 0.02
            preds[f'NET_{ts}'] = np.where(k > 0, (lg.astype(np.float64) >= t - 1e-9).astype(np.int8), -1)
            # exact-bin version (the implementation's floor((lg - LO) / STEP) >= j)
            pb = np.where(k > 0, (np.clip(np.floor((lg.astype(np.float64) + 40.0) / 0.02), 0, 3999) >= j).astype(np.int8), -1)
            acc.setdefault(f'net_bin_vs_t_diff_{ts}', 0)
            acc[f'net_bin_vs_t_diff_{ts}'] += int((pb != preds[f'NET_{ts}']).sum())
            preds[f'NET_{ts}'] = pb
        near_all = ndimage.binary_dilation(occ, structure=C26)
        near_noslab = ndimage.binary_dilation(occ & ~slab, structure=C26)
        near_obj = ndimage.binary_dilation(obj, structure=C26)
        for arm, pr_ in preds.items():
            pr = (pr_ == 1) & scope
            tp, fp, fn = int((pr & tr).sum()), int((pr & ~tr).sum()), int((~pr & tr).sum())
            a = acc.setdefault(arm, dict(f1w=0., freew=0., zone=np.zeros(3), ph=np.zeros(3), ph_alt=np.zeros(2)))
            a['f1w'] += (f1(tp, fp, fn) or 0.) * w / n
            nfree = int(Tf.sum())
            a['freew'] += (((pr_ == 0) & Tf).sum() / nfree) * w / n
            a['zone'] += np.array([int((pr & zone & tr).sum()), int((pr & zone & ~tr).sum()), int((~pr & zone & tr).sum())]) * w
            m = (pr_ == 1) & region
            lab, nb = ndimage.label(m, structure=C26)
            if nb:
                ids = np.arange(1, nb + 1)
                hit_all = ndimage.maximum(near_all, lab, ids) > 0
                hit_ns = ndimage.maximum(near_noslab, lab, ids) > 0
                hit_ob = ndimage.maximum(near_obj, lab, ids) > 0
                sizes = ndimage.sum(np.ones_like(m), lab, ids)
                a['ph'] += np.array([int((~hit_all).sum()), float(sizes[~hit_all].sum()), nb]) * w / n
                a['ph_alt'] += np.array([int((~hit_ns).sum()), int((~hit_ob).sum())]) * w / n
        for ts, (model, j, h) in nets.items():
            pn = preds[f'NET_{ts}'] == 1
            for hb in 'AB':
                pb6 = preds[f'B6_{hb}'] == 0
                sel = pn & pb6 & scope & (S[h] <= 0)
                f = acc.setdefault(f'flip_{ts}_vs_{hb}', np.zeros(4, np.int64))
                f += np.array([int((sel & tr).sum()) * w, int((sel & ~tr).sum()) * w, int((sel & tr).sum()), int((sel & ~tr).sum())])
            # decomposition of the wrong flips (own pairing): distance to the nearest generator furniture voxel, in / out of zone, height, k
            sel = pn & (preds[f'B6_{h}'] == 0) & scope & (S[h] <= 0)
            wrong = sel & ~tr
            dd = acc.setdefault(f'wrongflip_{ts}', dict(cheb1=0, cheb2=0, far=0, in_zone=0, z1_floor_layer=0, k1=0, k2=0, k3p=0, n=0,
                                                         gen_occ=0, right_cheb1=0, right_n=0))
            dd['n'] += int(wrong.sum()); dd['cheb1'] += int((wrong & (dist == 1)).sum()); dd['cheb2'] += int((wrong & (dist == 2)).sum())
            dd['far'] += int((wrong & (dist > 2)).sum()); dd['in_zone'] += int((wrong & zone).sum())
            dd['z1_floor_layer'] += int(wrong[:, :, 1].sum()); dd['gen_occ'] += int((wrong & occ).sum())
            dd['k1'] += int((wrong & (k == 1)).sum()); dd['k2'] += int((wrong & (k == 2)).sum()); dd['k3p'] += int((wrong & (k >= 3)).sum())
            right = sel & tr
            dd['right_n'] += int(right.sum())
    for arm in ('B6_A', 'B6_B', 'NET_342101', 'NET_342202'):
        a = acc[arm]; zt = a['zone']
        rows = [r for r, w in js['R8'][arm]['rows'] for _ in range(w)]
        impl_f1 = float(np.mean([r['furn_f1'] for r in rows])); impl_free = float(np.mean([r['free_recall'] for r in rows]))
        c = np.asarray(js['R8'][arm]['zones']['zone']['c'], np.float64) if 'zone' in js['R8'][arm]['zones'] else None
        res[arm] = dict(furn_f1_mine=a['f1w'], furn_f1_impl=impl_f1, free_mine=a['freew'], free_impl=impl_free,
                        zone_counts_mine=zt.astype(int).tolist(), zone_counts_impl=(c[:3].astype(int).tolist() if c is not None else None),
                        zone_f1_mine=f1(*zt), phantoms_mine=a['ph'].tolist(), phantoms_impl=js['R8'][arm]['phantoms'],
                        phantoms_truth_noslab=a['ph_alt'][0], phantoms_truth_objonly=a['ph_alt'][1])
    for ts in ('342101', '342202'):
        for hb in 'AB':
            f = acc[f'flip_{ts}_vs_{hb}']
            impl = js['flips'][f'NET_{ts}'][f'B6_{hb}']['le_-0']
            res[f'flip_{ts}_vs_{hb}'] = dict(mine_weighted=f[:2].tolist(), prec_w=(f[0] / (f[0] + f[1])) if f[0] + f[1] else None,
                                             mine_unweighted=f[2:].tolist(), impl=impl[:2])
        res[f'wrongflip_{ts}'] = acc[f'wrongflip_{ts}']
        res[f'net_bin_vs_t_diff_{ts}'] = acc[f'net_bin_vs_t_diff_{ts}']
    out['res'] = res
    out['ood_voxels'] = int(z['ood_mask'].sum())
    # are the new objects seen and in scope?  (component union of clean views)
    seen_occ = (priv == 1).any(0); seen = (priv >= 0).any(0)
    om = z['ood_mask'].astype(bool)
    out['ood_seen_occ'] = int((om & seen_occ).sum()); out['ood_seen_occ_in_region'] = int((om & seen_occ & region).sum())
    out['ood_in_region'] = int((om & region).sum())
    out['ood_subset_obj'] = bool(not (om & ~obj).any())
    return out


if __name__ == '__main__':
    todo = sys.argv[1].split(',')
    seeds = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [203]
    allr = {}
    for nm in todo:
        for sd in seeds:
            r = run(nm, sd); allr[f'{nm}_s{sd}'] = r
            print(json.dumps(r, default=float, indent=1), flush=True)
    json.dump(allr, open(os.path.join(HERE, 'recompute_' + '_'.join(todo) + '.json'), 'w'), default=float, indent=1)
