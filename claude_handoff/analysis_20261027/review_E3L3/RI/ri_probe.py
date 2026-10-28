"""ri_probe.py (reviewer R-I, 2026-10-28): CPU probes on E3-L3 (D8-65). Probes only (scheduling), never a verdict.
Imports the frozen code read-only (g4_e3l3 / g4_e3l / g4_e3r / g4_noise_panels / g4_ood_panel / g4_island_compose / g4_registration),
modifies nothing. Writes only under review_E3L3/RI/.

modes (one job each):
  val  rooms index seed      : VAL export (review_E3R/inputs/export), tiers cam_nf (+K0) and cam (unregistered); arms B6_A/B, NET x2, E3L3 x2;
                               extended strata: k, height band, object kind, whole-object miss, FN surface/interior, FP adjacent/far,
                               paired voxel diff E3L3 vs NET (same seed pairing 342101<->A, 342202<->B)
  ood  variant rooms index seed : TRAIN house, v420ood variant (ctrl / novel / floating), cam_nf; export with the H1 writing into RI/ood_export,
                               OP.evaluate (NET / B6, H1 code) + the same layers for E3L3 (zone, kinds, flips, phantoms)
  e6   rooms index seed      : VAL cam; noreg vs G4-E6 island composition (E4 edge estimates, D8-51 rule), aligned row (E4 eval_arm), all arms
  tier rooms index seed      : TRAIN house; tiers cam_nf / lidar / harsh: R8 on public maps (unregistered) and on agent-frame maps (= *_nf), K0
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json, time, traceback, multiprocessing, argparse  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
SCR = os.path.join(ROOT, 'scripts', 'claude_v342')
if SCR not in sys.path:
    sys.path.insert(0, SCR)
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
EXPORT = os.path.join(INP, 'export')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
E4_JOBS = os.path.join(AN, 'g4', 'e4', 'jobs')
OUT = os.path.join(HERE, 'out')
OOD_EXPORT = os.path.join(HERE, 'ood_export')
SEEDS = ('342101', '342202')
PAIR = {'342101': 'A', '342202': 'B'}

import g4_e3l3 as E3  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

L = E3.L
R = E3.R          # g4_e3r
G = E3.G          # g4_noise_panels
UNK = G.UNK
_M = None


def load_all(threads):
    global _M
    if _M is not None:
        return _M
    torch.set_num_threads(max(1, int(threads)))
    b6 = {h: R.Evidence.from_spec(f'c:{FIT}:{h}:cam_nf') for h in 'AB'}
    nets, e3 = {}, {}
    for ts in SEEDS:
        h = PAIR[ts]
        ck, model = R.load_ckpt(os.path.join(INP, f's{ts}', 'final.pt'), 'cpu')
        thr = json.load(open(os.path.join(INP, f's{ts}', 'threshold.json')))
        ev = R.Evidence.from_recipe(ck['recipe']['evidence'])
        assert np.array_equal(ev.table, b6[h].table)
        nets[ts] = dict(model=model, j=int(thr['j']), j_sep=int(thr['k0']['j_sep']), half=h)
        ck3, m3 = E3.load_model(os.path.join(INP, f'e3l3_s{ts}', 'final.pt'), 'cpu')
        thr3 = json.load(open(os.path.join(INP, f'e3l3_s{ts}', 'threshold.json')))
        tb = np.asarray(ck3['recipe']['evidence']['table'])
        e3[ts] = dict(model=m3, w=float(m3.w.detach()), jo=int(thr3['R8']['j_occ']), jk=int(thr3['K0']['j_occ']),
                      tb=torch.as_tensor(tb, dtype=torch.int32), table_is_b6=bool(np.array_equal(tb, b6[h].table)), half=h)
    _M = dict(b6=b6, nets=nets, e3=e3)
    return _M


# ---------------------------------------------------------------- object ids (per-voxel object index and category)
def rasterize_ids(spec, dims, obj_mask):
    """v420 rasterize's object loop, one object at a time (last writer wins), masked by the generator's obj mask"""
    W = G.W
    NX, NY, NZ = dims
    Lc = spec['chart_m']; cx, cz = Lc / NX, W.HEIGHT_M / NZ
    zc = (np.arange(NZ) + .5) * cz; xc = (np.arange(NX) + .5) * cx; yc = (np.arange(NY) + .5) * cx
    oid = np.full((NX, NY, NZ), -1, np.int32)
    for i, o in enumerate(spec['objects']):
        m = np.zeros((NX, NY, NZ), bool)
        for p in o['prims']:
            if p[0] == 'box':
                _, x0, y0, z0, x1, y1, z1 = p
                ix = (xc >= x0) & (xc < x1); iy = (yc >= y0) & (yc < y1); iz = (zc >= W.FLOOR_T + z0) & (zc < W.FLOOR_T + z1)
                ix = ix if ix.any() else W._cell_of((x0 + x1) / 2, cx, NX)
                iy = iy if iy.any() else W._cell_of((y0 + y1) / 2, cx, NY)
                iz = iz if iz.any() else W._cell_of(W.FLOOR_T + (z0 + z1) / 2, cz, NZ)
                m[np.ix_(ix, iy, iz)] = True
            else:
                _, ccx, ccy, r, z0, z1 = p
                m2 = (xc[:, None] - ccx) ** 2 + (yc[None, :] - ccy) ** 2 <= r ** 2
                if not m2.any():
                    m2[W._cell_of(ccx, cx, NX), W._cell_of(ccy, cx, NY)] = True
                iz = (zc >= W.FLOOR_T + z0) & (zc < W.FLOOR_T + z1)
                iz = iz if iz.any() else W._cell_of(W.FLOOR_T + (z0 + z1) / 2, cz, NZ)
                m |= m2[:, :, None] & iz[None, None, :]
        oid[m] = i
    oid[~obj_mask] = -1
    cov = float(((oid >= 0) == obj_mask).mean())
    return oid, [str(o['kind']) for o in spec['objects']], cov


def ids_from_npz(z, label='obj_family'):
    shape = z['occ'].shape; V = int(np.prod(shape))
    oid = np.full(V, -1, np.int32); idx, off = z['objvox_idx'], z['objvox_off']
    for i in range(len(off) - 1):
        oid[idx[off[i]:off[i + 1]]] = i
    oid = oid.reshape(shape); oid[~np.asarray(z['obj'], bool)] = -1
    return oid, [str(k) for k in z[label]]


class ObjCtx:
    def __init__(self, oid, kinds):
        self.oid = oid; self.kinds = kinds; self.nobj = len(kinds)
        self.cats = sorted(set(kinds)); ci = {c: i for i, c in enumerate(self.cats)}
        self.cat_of_obj = np.array([ci[k] for k in kinds], np.int64) if kinds else np.zeros(0, np.int64)
        self.catvox = np.where(oid >= 0, self.cat_of_obj[np.maximum(oid, 0)] if kinds else -1, -1)
        if (oid >= 0).any():
            ind = ndimage.distance_transform_edt(oid < 0, return_distances=False, return_indices=True)
            self.nearcat = self.catvox[ind[0], ind[1], ind[2]]
        else:
            self.nearcat = np.full(oid.shape, -1, np.int64)


# ---------------------------------------------------------------- extended counts
def ext_counts(p, T, H, kc, O):
    scope = T.known & H.region
    pr = (p == 1) & scope
    tp, fp, fn = pr & T.tr, pr & ~T.tr, T.tr & ~pr
    frh = (p == 0) & T.free
    kk = np.clip(kc, 0, 5).astype(np.int64)
    d = {}
    d['base'] = np.array([tp.sum(), fp.sum(), fn.sum(), frh.sum(), T.free.sum()], np.int64)
    d['k'] = np.stack([np.bincount(kk[x], minlength=6) for x in (tp, fp, fn)], 1).astype(np.int64)
    d['kfree'] = np.stack([np.bincount(kk[frh], minlength=6), np.bincount(kk[T.free], minlength=6)], 1).astype(np.int64)
    d['h'] = np.stack([G._band_sum(x, H.band_z, 3) for x in (tp, fp, fn)], 1).astype(np.int64)
    d['fn_split'] = np.array([(fn & T.surf).sum(), (fn & ~T.surf).sum()], np.int64)             # surface / interior
    d['fp_split'] = np.array([(fp & T.dil_tr).sum(), (fp & ~T.dil_tr).sum()], np.int64)          # adjacent to true furniture / far
    cat = {}
    for ci, c in enumerate(O.cats):
        tm = O.catvox == ci; nm = O.nearcat == ci
        cat[c] = np.array([(tp & tm).sum(), (fp & nm).sum(), (fn & tm).sum()], np.int64)
    d['cat'] = cat
    if O.nobj:
        oi = O.oid
        ntrue = np.bincount(oi[T.tr & (oi >= 0)], minlength=O.nobj); ntp = np.bincount(oi[tp & (oi >= 0)], minlength=O.nobj)
        whole = (ntrue > 0) & (ntp == 0)
        sizes = ntrue[whole]
        d['whole'] = np.array([int(ntrue[whole].sum()), int(whole.sum()), int((ntrue > 0).sum()), int((T.tr & (oi < 0)).sum())], np.int64)
        d['whole_size'] = np.array([((sizes >= lo) & (sizes < hi)).sum() for lo, hi in ((1, 5), (5, 20), (20, 100), (100, 10 ** 9))], np.int64)
        d['whole_size_vox'] = np.array([sizes[(sizes >= lo) & (sizes < hi)].sum() for lo, hi in ((1, 5), (5, 20), (20, 100), (100, 10 ** 9))], np.int64)
        wc = {}
        for ci, c in enumerate(O.cats):
            sel = whole & (O.cat_of_obj == ci)
            wc[c] = np.array([int(ntrue[sel].sum()), int(sel.sum()), int(((ntrue > 0) & (O.cat_of_obj == ci)).sum())], np.int64)
        d['whole_cat'] = wc
    return d


def diff_counts(pa, pb, T, H, kc):
    """paired voxel diff: a wrong & b right (lose) and a right & b wrong (win), by truth (furniture / not), boundary class, k, height"""
    scope = T.known & H.region
    ya = ((pa == 1) & scope); yb = ((pb == 1) & scope)
    wa = scope & (ya != T.tr); wb = scope & (yb != T.tr)
    lose = wa & ~wb; win = ~wa & wb
    kk = np.clip(kc, 0, 5).astype(np.int64)
    d = {}
    for nm, m in (('lose', lose), ('win', win)):
        fnm = m & T.tr; fpm = m & ~T.tr
        d[nm] = dict(fn=np.array([(fnm & T.surf).sum(), (fnm & ~T.surf).sum()], np.int64),
                     fp=np.array([(fpm & T.dil_tr).sum(), (fpm & ~T.dil_tr).sum()], np.int64),
                     k=np.stack([np.bincount(kk[fnm], minlength=6), np.bincount(kk[fpm], minlength=6)], 1).astype(np.int64),
                     h=np.stack([G._band_sum(fnm, H.band_z, 3), G._band_sum(fpm, H.band_z, 3)], 1).astype(np.int64))
    return d


def addw(acc, key, d, w):
    if isinstance(d, dict):
        a = acc.setdefault(key, {})
        for k, v in d.items():
            addw(a, k, v, w)
    else:
        v = np.asarray(d, np.float64) * w
        acc[key] = acc[key] + v if key in acc else v


def lists(d):
    if isinstance(d, dict):
        return {k: lists(v) for k, v in d.items()}
    return np.asarray(d).tolist()


# ---------------------------------------------------------------- read-outs
@torch.no_grad()
def readouts(maps, F, M, k0_maps=None, want_lg=False):
    """R8 predictions per component (per member for E3L3) for B6_A/B, NET_*, E3L3_*; K0 per agent (if k0_maps)"""
    comps = F['comps']
    res = dict(R8={}, K0={}, kc=[], S={})
    sums = {h: [R.direct_sums(maps, cm, M['b6'][h].table) for cm in comps] for h in 'AB'}
    res['kc'] = [s[1] for s in sums['A']]
    res['S'] = {h: [s[0] for s in sums[h]] for h in 'AB'}
    for h in 'AB':
        res['R8'][f'B6_{h}'] = [[R.linear_readout(S, kc, M['b6'][h].theta)] * len(cm) for (S, kc), cm in zip(sums[h], comps)]
    for ts, nt in M['nets'].items():
        s = sums[nt['half']]
        lg = R.net_logits(nt['model'], [x[0] for x in s], [x[1] for x in s], 'cpu', 1)
        res['R8'][f'NET_{ts}'] = [[R.net_readout(lg[ci], s[ci][1], nt['j'])] * len(cm) for ci, cm in enumerate(comps)]
    for ts, m in M['e3'].items():
        zp, lp = L.encode_maps(m['model'], list(maps), m['tb'], 'cpu')
        hp, lq, zq = E3.source(zp, lp, 'round')
        out = []
        for ci, cm in enumerate(comps):
            Fl = L.local_forest(F, cm)
            hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, m['w'], 'round')
            lg = L.decode_np(m['model'], hb, lb, 2)
            kc = res['kc'][ci]; lbn = lb.numpy()
            assert all(np.array_equal(lbn[i] > 0, kc > 0) for i in range(len(cm)))
            out.append([R.net_readout(lg[i], kc, m['jo']) for i in range(len(cm))])
        res['R8'][f'E3L3_{ts}'] = out
        zl = zq[lp > 0]
        res.setdefault('zdist', {})[ts] = [int((zl < -0.5).sum()), int((zl.abs() < 0.5).sum()), int((zl > 0.5).sum())]
        if k0_maps is not None:
            if k0_maps is maps:
                hk, lk = hp, lq
            else:
                zk, lk0 = L.encode_maps(m['model'], list(k0_maps), m['tb'], 'cpu'); hk, lk, _ = E3.source(zk, lk0, 'round')
            lgk = L.decode_np(m['model'], hk, lk, 2)
            res['K0'][f'E3L3_{ts}'] = [R.net_readout(lgk[a], (k0_maps[a] != UNK).astype(np.int32), m['jk']) for a in range(len(k0_maps))]
    if k0_maps is not None:
        for ts, nt in M['nets'].items():
            own = [R.stalk(Lm, M['b6'][nt['half']].table) for Lm in k0_maps]
            lk = R.net_logits(nt['model'], [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 1)
            res['K0'][f'NET_{ts}'] = [R.net_readout(lk[a], own[a][1].astype(np.int32), nt['j_sep']) for a in range(len(k0_maps))]
        for h in 'AB':
            res['K0'][f'B6_{h}'] = []
            for Lm in k0_maps:
                e, k = R.stalk(Lm, M['b6'][h].table)
                res['K0'][f'B6_{h}'].append(R.linear_readout(e, k.astype(np.int32), M['b6'][h].theta))
    return res


def score_block(rd, F, H, R8T, K0T, O, k0_maps=None):
    """rows (E3-R scored, mean over expanded agents) + extended counts + paired diffs"""
    out = dict(R8={}, K0={}, diff={})
    for arm, per in rd['R8'].items():
        rows, acc = [], {}
        for ci, cm in enumerate(F['comps']):
            kc = rd['kc'][ci]
            for p in per[ci]:
                r = G.score(p, R8T[ci]); rows.append(dict(furn_f1=r['furn_f1'], furn_f1_tol1=r['furn_f1_tol1'], free_recall=r['free_recall'],
                                                          surface_f1=r['surface_f1'], unknown_known_fpr=r['unknown_known_fpr']))
                addw(acc, 'c', ext_counts(p, R8T[ci], H, kc, O), 1.0)
        out['R8'][arm] = dict(rows=rows, counts=lists(acc.get('c', {})))
    for ts in SEEDS:
        acc = {}
        for ci, cm in enumerate(F['comps']):
            pn = rd['R8'][f'NET_{ts}'][ci][0]
            for p in rd['R8'][f'E3L3_{ts}'][ci]:
                addw(acc, 'd', diff_counts(p, pn, R8T[ci], H, rd['kc'][ci]), 1.0)
        out['diff'][ts] = lists(acc.get('d', {}))
    if k0_maps is not None:
        for arm, per in rd['K0'].items():
            rows, acc = [], {}
            for a, p in enumerate(per):
                ki = (k0_maps[a] != UNK).astype(np.int32)
                r = G.score(p, K0T[a]); rows.append(dict(furn_f1=r['furn_f1'], free_recall=r['free_recall'], furn_f1_tol1=r['furn_f1_tol1']))
                addw(acc, 'c', ext_counts(p, K0T[a], H, ki, O), 1.0)
            out['K0'][arm] = dict(rows=rows, counts=lists(acc.get('c', {})))
        for ts in SEEDS:
            acc = {}
            for a in range(len(k0_maps)):
                addw(acc, 'd', diff_counts(rd['K0'][f'E3L3_{ts}'][a], rd['K0'][f'NET_{ts}'][a], K0T[a], H, (k0_maps[a] != UNK).astype(np.int32)), 1.0)
            out['diff'][f'K0_{ts}'] = lists(acc.get('d', {}))
    out['zdist'] = rd.get('zdist')
    return out


# ---------------------------------------------------------------- modes
def job_val(rooms, index, seed, M):
    z = np.load(os.path.join(EXPORT, f'val_r{rooms}_i{index}_s{seed}.npz'))
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    spec = G.W.world_spec('VALIDATION', index, rooms)
    oid, kinds, cov = rasterize_ids(spec, z['occ'].shape, np.asarray(z['obj'], bool)); O = ObjCtx(oid, kinds)
    out = dict(mode='val', rooms=rooms, index=index, seed=seed, n=n, comps=F['comps'], oid_cover=cov, tiers={})
    for tn in ('cam_nf', 'cam'):
        L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
        t0 = time.time()
        k0m = L_k0 if tn == 'cam_nf' else None
        rd = readouts(L_pub, F, M, k0m)
        out['tiers'][tn] = score_block(rd, F, H, R8T, K0T, O, k0m); out['tiers'][tn]['seconds'] = time.time() - t0
    return out


def job_tier(rooms, index, seed, M):
    R.W = G.W
    ctx = R.house_ctx('TRAIN', rooms, index)
    H = ctx['H']; F = ctx['F']; n = ctx['n']; private = ctx['private']
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(ctx['T8'][cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    spec = G.W.world_spec('TRAIN', index, rooms)
    oid, kinds, cov = rasterize_ids(spec, ctx['occ'].shape, np.asarray(ctx['fx']['obj'], bool)); O = ObjCtx(oid, kinds)
    out = dict(mode='tier', rooms=rooms, index=index, seed=seed, n=n, comps=F['comps'], oid_cover=cov, tiers={})
    k0cache = {}
    for tn in ('cam_nf', 'lidar', 'harsh'):
        t0 = time.time()
        L_k0, L_pub, has_inter = R.tier_maps(ctx, tn, seed, k0cache)
        rd = readouts(L_pub, F, M, L_k0)
        out['tiers'][tn] = score_block(rd, F, H, R8T, K0T, O, L_k0)
        if has_inter:
            rd2 = readouts(L_k0, F, M, None)
            out['tiers'][tn + '_nf'] = score_block(rd2, F, H, R8T, K0T, O, None)
        out['tiers'][tn]['seconds'] = time.time() - t0
    return out


def job_ood(variant, rooms, index, seed, M, threads):
    import g4_ood_panel as OP
    import mapping_3d_storey_data_v420ood as WO
    OPM = OP.load_models(threads)
    R.W = WO.module_for(variant)
    try:
        ctx = R.house_ctx('TRAIN', rooms, index)
    except ValueError as e:
        R.W = G.W
        return dict(mode='ood', variant=variant, rooms=rooms, index=index, seed=seed, skipped=repr(e)[:200])
    fx = ctx['fx']
    base = dict(occ=np.asarray(ctx['occ'], np.int8), slab=np.asarray(fx['slab'], bool), obj=np.asarray(fx['obj'], bool), cell=ctx['cell'],
                private=np.asarray(ctx['private'], np.int8), edges=R.edges_array(ctx['edges']),
                ood_mask=np.asarray(fx['ood_mask'], bool), objvox_idx=fx['objvox_idx'], objvox_off=fx['objvox_off'],
                obj_kinds=fx['obj_kinds'], obj_family=fx['obj_family'], obj_ood=fx['obj_ood'])
    os.makedirs(OOD_EXPORT, exist_ok=True)
    L_k0, _, _ = R.tier_maps(ctx, 'cam_nf', seed, {})
    R.W = G.W
    path = os.path.join(OOD_EXPORT, f'{variant}_train_r{rooms}_i{index}_s{seed}.npz'); tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, L_k0_cam_nf=np.stack(L_k0).astype(np.int8), **base); os.replace(tmp, path)
    t0 = time.time()
    op = OP.evaluate(path, OPM)                       # NET / B6 / B4 with the H1 code
    t_op = time.time() - t0
    # E3L3 on the same file, H1 layers
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    K0T = [G.Target(private[a], H) for a in range(n)]
    Lm = list(z['L_k0_cam_nf'])
    occ = z['occ'] == 1; slab = np.asarray(z['slab'], bool); NZ = occ.shape[2]
    near = ndimage.binary_dilation(occ, structure=OP.CUBE)
    floor_k = slab.all((0, 1)) & (np.arange(NZ) < NZ // 2); layer1 = int(np.nonzero(floor_k)[0].max()) + 1
    near_nf = ndimage.binary_dilation(occ & ~floor_k[None, None, :], structure=OP.CUBE)
    zone, kzones, fzones = OP.object_zones(z); zones = dict(zone=zone, rest=~zone)
    rd = readouts(Lm, F, M, None)
    e3 = {}
    for ts in SEEDS:
        h = PAIR[ts]
        acc, kacc, facc, ph = {}, {}, {}, np.zeros(4)
        rows = []
        fl = {}
        for ci, cm in enumerate(F['comps']):
            kc = rd['kc'][ci]; pb = rd['R8'][f'B6_{h}'][ci][0]
            for p in rd['R8'][f'E3L3_{ts}'][ci]:
                r = G.score(p, R8T[ci]); rows.append(dict(furn_f1=r['furn_f1'], free_recall=r['free_recall'], furn_f1_tol1=r['furn_f1_tol1']))
                for zn, d in OP.zone_counts(p, R8T[ci], H, kc, zones).items():
                    OP._addw(acc, zn, d, 1)
                for kd, v in OP.kind_counts(p, R8T[ci], H, fzones).items():
                    OP._addw(facc, kd, v, 1)
                ph += np.asarray(OP.phantom_blocks(p, kc, H, near, near_nf, layer1 + 1), np.float64) / n
                OP._addw(fl, 'f', OP.flip_counts(p, pb, rd['S'][h][ci], R8T[ci], H, zone), 1)
        # also the NET re-computed here (same pipeline as E3L3) for a like-for-like zone comparison
        nacc = {}; nfl = {}
        for ci, cm in enumerate(F['comps']):
            kc = rd['kc'][ci]; pn = rd['R8'][f'NET_{ts}'][ci][0]; pb = rd['R8'][f'B6_{h}'][ci][0]
            for zn, d in OP.zone_counts(pn, R8T[ci], H, kc, zones).items():
                OP._addw(nacc, zn, d, len(cm))
            OP._addw(nfl, 'f', OP.flip_counts(pn, pb, rd['S'][h][ci], R8T[ci], H, zone), len(cm))
        e3[f'E3L3_{ts}'] = dict(rows=rows, zones=OP._lists(acc), families=OP._lists(facc), phantoms=ph.tolist(), flips=OP._lists(fl['f']),
                                net_zones_here=OP._lists(nacc), net_flips_here=OP._lists(nfl['f']))
    # paired diff E3L3 vs NET on this OOD world (by family of nearest object)
    O = ObjCtx(*ids_from_npz(z, 'obj_family'))
    sb = score_block(rd, F, H, R8T, K0T, O, None)
    return dict(mode='ood', variant=variant, rooms=rooms, index=index, seed=seed, n=n, comps=F['comps'], op=op, e3=e3, ext=sb, t_op=t_op,
                file=os.path.basename(path))


def job_e6(rooms, index, seed, M):
    import g4_island_compose as OI
    REG = OI.R
    W = G.W
    name = f'val_r{rooms}_i{index}_s{seed}'
    e4 = json.load(open(os.path.join(E4_JOBS, name + '.json')))
    if e4.get('skipped'):
        return dict(mode='e6', rooms=rooms, index=index, seed=seed, skipped='e4 skipped')
    prm = e4['fingerprint']['match']
    fx = W.make_fixture('VALIDATION', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    K0T = [G.Target(private[a], H) for a in range(n)]
    spec = W.world_spec('VALIDATION', index, rooms)
    oid, kinds, cov = rasterize_ids(spec, occ.shape, np.asarray(fx['obj'], bool)); O = ObjCtx(oid, kinds)
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    c = G.CONDITIONS['cam']; e4c = e4['conds']['cam']
    scans_pub, L_pub = [], []
    for a in range(n):
        z_ = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(z_[2] * c['inter_yaw'])), np.array([z_[0] * c['inter_t'], z_[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
        pub = [G.to_public(s, frame) for s in scans]
        L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3']); scans_pub.append(pub)
    sha_ok = W.array_hash(np.stack(L_pub)) == e4c['pub_sha256']
    bpos = OI.believed_positions(scans_pub)
    Ke6, C6, isl = OI.e6_corrections(n, e4c['registration']['edges'], bpos)
    I3 = OI.I3
    out = dict(mode='e6', rooms=rooms, index=index, seed=seed, n=n, comps=F['comps'], pub_sha_ok=bool(sha_ok), islands=isl, arms={})
    for arm, K in (('noreg', [I3.copy() for _ in range(n)]), ('e6', Ke6)):
        t0 = time.time()
        maps0 = [L_pub[a] if np.array_equal(K[a], I3) else REG._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
        _, preds = G.fuse(maps0, F, 'sum')
        al = [None] * n; Gs = []
        for ci, cm in enumerate(F['comps']):
            ev_t = REG.evidence(REG.planes_tern(T8[cm[0]], cell, prm), cell, prm)
            ev_b = REG.evidence(REG.planes_tern(preds[cm[0]], cell, prm), cell, prm)
            res = REG.register_pair(ev_t, ev_b, cell, prm); Gc = res['M']; Gs.append(REG.params(Gc))
            for a in cm:
                al[a] = maps0[a] if np.array_equal(Gc, I3) else REG._integrate(shape, cell, ds, scans_pub[a], Gc @ K[a])
        rd = readouts(al, F, M, None)
        out['arms'][arm] = dict(aligned=score_block(rd, F, H, R8T, K0T, O, None), G=Gs, seconds=time.time() - t0)
        del maps0, al, rd
    return out


def run(job):
    mode = job[0]; threads = job[-1]
    t0 = time.time()
    try:
        M = load_all(threads)
        if mode == 'val':
            r = job_val(*job[1:4], M)
        elif mode == 'tier':
            r = job_tier(*job[1:4], M)
        elif mode == 'ood':
            r = job_ood(*job[1:5], M, threads)
        elif mode == 'e6':
            r = job_e6(*job[1:4], M)
        r['e3_table_is_b6'] = {ts: M['e3'][ts]['table_is_b6'] for ts in SEEDS}
        r['w'] = {ts: M['e3'][ts]['w'] for ts in SEEDS}
    except Exception:
        r = dict(mode=mode, job=list(job[:-1]), failed=traceback.format_exc()[-3000:])
    r['seconds'] = time.time() - t0
    tag = '_'.join(str(x) for x in job[:-1])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, tag + '.json.tmp'), 'w') as fh:
        json.dump(r, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(os.path.join(OUT, tag + '.json.tmp'), os.path.join(OUT, tag + '.json'))
    return tag, r['seconds'], bool(r.get('failed')), r.get('skipped')


def _init():
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    G.setup_env()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='semicolon list of mode,args...')
    ap.add_argument('--workers', type=int, default=7); ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    jobs = []
    for s in a.jobs.split(';'):
        p = s.split(',')
        args = [p[0]] + [x if not x.lstrip('-').isdigit() else int(x) for x in p[1:]]
        jobs.append(tuple(args) + (a.threads,))
    print(time.strftime('%H:%M:%S', time.gmtime()), 'jobs', len(jobs), flush=True)
    if a.workers <= 1:
        for j in jobs:
            print(run(j), flush=True)
        return
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(a.workers, initializer=_init, maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(run, jobs):
            print(time.strftime('%H:%M:%S', time.gmtime()), res, flush=True)


if __name__ == '__main__':
    main()
