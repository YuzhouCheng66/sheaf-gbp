"""A05 probe 3 (CPU, demo data): anatomy of the current model's furniture-region errors (3S-X s342101 + calibrated rule, K0 and R8).
Region = metrics3d's furniture region (not slab, not structure dilated by one voxel), target-known scope.
FN split: predicted unknown (known-head veto) vs predicted free. FP split: 6-adjacent to a WORLD-occupied voxel (thickening) vs not
(hallucination). Target-occupied furniture voxels by part shape in gt_obj: pole (one voxel thin in x and y: legs, lamp poles),
sheet (thin in z: table tops, shelves, seats of chairs), panel (thin in exactly one of x / y: bookshelf sides, chair backs, headboards),
bulk (no thin axis). Recall per class, share of the target, and oracle F1 if one error class were fixed."""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets, dilate6, DIMS  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))


def thin_axes(obj):
    o = np.pad(obj, 1)
    tx = obj & ~o[2:, 1:-1, 1:-1] & ~o[:-2, 1:-1, 1:-1]
    ty = obj & ~o[1:-1, 2:, 1:-1] & ~o[1:-1, :-2, 1:-1]
    tz = obj & ~o[1:-1, 1:-1, 2:] & ~o[1:-1, 1:-1, :-2]
    return tx, ty, tz


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    acc = {}
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        region = ~slab & ~dilate6(torch.from_numpy(struct.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)
        tx, ty, tz = thin_axes(obj)
        pole = tx & ty; sheet = tz & ~pole; panel = (tx ^ ty) & ~tz; bulk = obj & ~tx & ~ty & ~tz
        other = obj & ~(pole | sheet | panel | bulk)                         # thin in z and one of x/y etc.
        near_world = dilate6(torch.from_numpy(occ.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS) & ~occ
        zidx = np.broadcast_to(np.arange(32)[None, None, :], DIMS)
        T8 = targets(own, m['reached'][m['hops'].index(8)])
        for hop, T in ((0, own), (8, T8)):
            P = pred[hop]
            for a in range(n):
                t, p = T[a], P[a]; known = t >= 0
                to = (t == 1) & known & region; po = (p == 1) & known & region
                tp = to & po; fp = po & ~to; fn = to & ~po
                d = acc.setdefault(hop, dict(tp=0, fp=0, fn=0, fn_veto=0, fn_free=0, fp_thick=0, fp_far=0, cls={}, zb={}))
                d['tp'] += int(tp.sum()); d['fp'] += int(fp.sum()); d['fn'] += int(fn.sum())
                d['fn_veto'] += int((fn & (p == -1)).sum()); d['fn_free'] += int((fn & (p == 0)).sum())
                d['fp_thick'] += int((fp & near_world).sum()); d['fp_far'] += int((fp & ~near_world & ~occ).sum())
                d.setdefault('fp_on_world_occ', 0); d['fp_on_world_occ'] += int((fp & occ).sum())   # predicted occ where world is occ but target says free? (cannot) / unknown? excluded
                for name, cm in (('pole', pole), ('sheet', sheet), ('panel', panel), ('bulk', bulk), ('other', other), ('nonobj', ~obj)):
                    c = d['cls'].setdefault(name, dict(target=0, tp=0, fn=0, fn_veto=0))
                    c['target'] += int((to & cm).sum()); c['tp'] += int((tp & cm).sum()); c['fn'] += int((fn & cm).sum()); c['fn_veto'] += int((fn & cm & (p == -1)).sum())
                for bname, (z0, z1) in (('z0.1-0.4m', (1, 4)), ('z0.4-0.9m', (4, 9)), ('z0.9-1.5m', (9, 15)), ('z1.5m+', (15, 32))):
                    zm = (zidx >= z0) & (zidx < z1)
                    c = d['zb'].setdefault(bname, dict(target=0, tp=0, fp=0))
                    c['target'] += int((to & zm).sum()); c['tp'] += int((tp & zm).sum()); c['fp'] += int((fp & zm).sum())
        print(m['house'], 'done', flush=True)

    S = {}
    for hop, d in acc.items():
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        s = dict(pooled_f1=f1(tp, fp, fn), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                 fn_share_veto=d['fn_veto'] / max(fn, 1), fn_share_free=d['fn_free'] / max(fn, 1),
                 fp_share_thickening=d['fp_thick'] / max(fp, 1), fp_share_hallucination=d['fp_far'] / max(fp, 1), fp_on_world_occ=d['fp_on_world_occ'] / max(fp, 1))
        tot_t = sum(c['target'] for k, c in d['cls'].items() if k != 'nonobj')
        s['classes'] = {k: dict(share_of_target=c['target'] / max(tp + fn, 1), recall=c['tp'] / max(c['target'], 1), fn_veto_share=c['fn_veto'] / max(c['fn'], 1)) for k, c in d['cls'].items()}
        for k in ('pole', 'sheet', 'panel'):
            c = d['cls'][k]; s[f'oracle_f1_fix_{k}'] = f1(tp + c['fn'], fp, fn - c['fn'])
        thin_fn = sum(d['cls'][k]['fn'] for k in ('pole', 'sheet', 'panel', 'other'))
        s['oracle_f1_fix_all_thin'] = f1(tp + thin_fn, fp, fn - thin_fn)
        s['oracle_f1_fix_veto'] = f1(tp + d['fn_veto'], fp, fn - d['fn_veto'])
        s['oracle_f1_fix_thickening'] = f1(tp, fp - d['fp_thick'], fn)
        s['oracle_f1_fix_hallucination'] = f1(tp, fp - d['fp_far'], fn)
        s['zbands'] = {k: dict(recall=c['tp'] / max(c['target'], 1), precision=c['tp'] / max(c['tp'] + c['fp'], 1), share=c['target'] / max(tp + fn, 1)) for k, c in d['zb'].items()}
        S[f'K{hop}' if hop == 0 else f'R{hop}'] = s
    json.dump(S, open(os.path.join(HERE, 'p3_furniture_anatomy.json'), 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
