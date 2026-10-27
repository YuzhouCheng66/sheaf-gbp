"""Reviewer's diagnostic (report only, TRAIN smoke npz): where do the S <= 0 flips sit (family of the nearest generator object, height
band, Chebyshev distance), and how the phantom count changes with the truth set (occ incl. slabs = implementation; occ minus floor slab,
blocks reaching >= 2 layers above the floor slab)."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa
import g4_e3r as E  # noqa
torch.set_num_threads(2)
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
EXP = os.path.join(AN, 'g4', 'h1_export'); INP = os.path.join(AN, 'review_E3R', 'inputs')
FIT = json.load(open(os.path.join(AN, 'g4', 'e1c', 'fit.json')))
S6 = ndimage.generate_binary_structure(3, 1); C26 = np.ones((3, 3, 3), bool)


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


def run(nm):
    z = np.load(os.path.join(EXP, nm + '.npz'))
    occ = z['occ'] == 1; slab = z['slab'].astype(bool); obj = z['obj'].astype(bool); priv = z['private']; L = z['L_k0_cam_nf']
    n = priv.shape[0]; comps = comps_of(z['edges'], n)
    region = ~slab & ~ndimage.binary_dilation(occ & ~obj & ~slab, structure=S6)
    shape = occ.shape; oid = np.full(int(np.prod(shape)), -1, np.int64)
    idx, off = z['objvox_idx'], z['objvox_off']
    for i in range(len(off) - 1):
        oid[idx[off[i]:off[i + 1]]] = i
    oid = oid.reshape(shape); fam = [str(x) for x in z['obj_family']]; kinds = [str(x) for x in z['obj_kinds']]
    dist, ind = ndimage.distance_transform_edt(oid < 0, return_indices=True)
    near_id = oid[ind[0], ind[1], ind[2]]
    cheb = ndimage.distance_transform_cdt(~obj, metric='chessboard')
    zc = np.arange(shape[2])
    floor_k = slab[0, 0, :]; ceil_k = slab[0, 0, :] & (zc > shape[2] // 2)
    near_all = ndimage.binary_dilation(occ, structure=C26)
    near_nofloor = ndimage.binary_dilation(occ & ~(slab & ~ceil_k[None, None, :]), structure=C26)
    out = {}
    for ts, h in (('342101', 'A'), ('342202', 'B')):
        tab = np.asarray(FIT['halves'][h]['tiers']['cam_nf']['cq'], np.int64); th = int(FIT['halves'][h]['tiers']['cam_nf']['theta'])
        ck, model = E.load_ckpt(os.path.join(INP, f's{ts}', 'final.pt'), 'cpu')
        j = json.load(open(os.path.join(INP, f's{ts}', 'threshold.json')))['j']
        byfam = {}; byh = {}; ph = dict(net_impl=0., b6_impl=0., net_nofloor_tall=0., b6_nofloor_tall=0., net_nofloor_all=0., b6_nofloor_all=0.)
        for cm in comps:
            w = len(cm) / n
            T = priv[cm].max(0); tr = (T == 1) & region; scope = (T >= 0) & region
            kn = L[cm] != -128; k = kn.sum(0)
            S = np.where(kn, tab[L[cm].astype(np.int64) + 40], 0).sum(0)
            lg = E.net_logits(model, [S.astype(np.int32)], [k.astype(np.int32)], 'cpu', 1)[0]
            pn = (k > 0) & (np.clip(np.floor((lg.astype(np.float64) + 40.0) / 0.02), 0, 3999) >= j)
            pb = (k > 0) & (S > th)
            sel = pn & ~pb & scope & (S <= 0)
            for name, m in (('right', sel & tr), ('wrong', sel & ~tr)):
                for v in np.argwhere(m):
                    i_ = near_id[tuple(v)]; key = f'{fam[i_]}/{kinds[i_]}' if i_ >= 0 else 'none'
                    d = byfam.setdefault(key, dict(right=0, wrong=0, wrong_cheb1=0)); d[name] += 1
                    if name == 'wrong' and cheb[tuple(v)] == 1:
                        d['wrong_cheb1'] += 1
                    hb = 'z_layer1' if v[2] == 1 else ('z<0.9m' if v[2] < 10 else 'z>=0.9m')
                    e = byh.setdefault(hb, dict(right=0, wrong=0)); e[name] += 1
            for arm, p in (('net', pn), ('b6', pb)):
                lab, nb = ndimage.label(p & region, structure=C26)
                if not nb:
                    continue
                ids = np.arange(1, nb + 1)
                h_all = ndimage.maximum(near_all, lab, ids) > 0
                h_nf = ndimage.maximum(near_nofloor, lab, ids) > 0
                zmax = ndimage.maximum(np.broadcast_to(zc[None, None, :], shape).astype(np.int64), lab, ids)
                ph[f'{arm}_impl'] += float((~h_all).sum()) * w
                ph[f'{arm}_nofloor_all'] += float((~h_nf).sum()) * w
                ph[f'{arm}_nofloor_tall'] += float((~h_nf & (zmax >= 2)).sum()) * w
        out[ts] = dict(flips_by_nearest=byfam, flips_by_height=byh, phantoms=ph)
    return out


if __name__ == '__main__':
    res = {}
    for nm in sys.argv[1].split(','):
        res[nm] = run(nm); print(nm, json.dumps(res[nm]), flush=True)
    json.dump(res, open(os.path.join(HERE, 'flip_family_' + sys.argv[2] + '.json'), 'w'), indent=1)
