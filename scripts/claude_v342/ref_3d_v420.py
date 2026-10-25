"""Reference predictors on the 3S-A criterion panel (Claude, 2026-10-25; review v420 M1 iii). CPU only, no model.
For the VALIDATION panel (indices 0-7 per stratum = val, 8-15 = fresh; N = 8) and hops K0 / R8, scores with train_3d_v420.metrics3d:
  slab-only     : predict the floor + ceiling slabs occupied and everything else free, inside the target-known scope;
  no-furniture  : the target itself with every furniture voxel set to free (a perfect structure-only mapper);
and reports the furniture share of the target surface. Writes JSON to --out.
python ref_3d_v420.py --source <sealed> --out FILE
"""
import argparse, json, multiprocessing, os, sys
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=6)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v420 as T
import mapping_3d_storey_data_v420 as W

if __name__ == '__main__':
    dims = W.DIMS['R1']
    plan = W.scene_plan('VALIDATION', 16, dims)
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, plan, chunksize=1).get(timeout=3600)
    out = {}
    for panel, sel in (('val', lambda s: s['index'] < 8), ('fresh', lambda s: s['index'] >= 8)):
        acc = {('slab', k): [] for k in (0, 8)}; acc.update({('nofurn', k): [] for k in (0, 8)}); share = []
        for sp, fx in zip(plan, fxs):
            if fx is None or not sel(sp):
                continue
            own = torch.from_numpy(fx['private']); slab = torch.from_numpy(fx['slab'].reshape(-1)); obj = torch.from_numpy(fx['obj'].reshape(-1)); struct = torch.from_numpy(fx['struct'].reshape(-1))
            for k in (0, 8):
                tgt = T.stream_targets(own, torch.from_numpy(fx['reached'][T.HOPS.index(k)]))
                slab_pred = torch.where(tgt >= 0, slab.to(torch.int8)[None].expand_as(tgt), torch.full_like(tgt, -1))
                nofurn = torch.where((tgt == 1) & obj[None], torch.zeros_like(tgt), tgt)
                acc[('slab', k)].append(T.agg(T.metrics3d(slab_pred, tgt, dims, slab, obj, struct)))
                acc[('nofurn', k)].append(T.agg(T.metrics3d(nofurn, tgt, dims, slab, obj, struct)))
                if k == 8:
                    surf = T.surface_of(tgt, dims); share.append(float((surf & obj[None]).sum()) / max(1., float(surf.sum())))
        out[panel] = {f'{name}_K{k}': {m: (float(np.mean([r[m] for r in v if r[m] is not None])) if any(r[m] is not None for r in v) else None) for m in T.METRIC_KEYS}
                      for (name, k), v in acc.items()}
        out[panel]['furniture_share_of_R8_target_surface'] = float(np.mean(share)); out[panel]['scenes'] = len(share)
    json.dump(out, open(args.out, 'w'), indent=1)
    for panel in ('val', 'fresh'):
        o = out[panel]
        print(panel, 'scenes', o['scenes'], '| furniture share of target surface %.3f' % o['furniture_share_of_R8_target_surface'])
        for key in ('slab_K0', 'slab_K8', 'nofurn_K0', 'nofurn_K8'):
            r = o[key]; print('  %-10s surface %.3f var %.3f furn %.3f obj_recall %s' % (key, r['surface_f1'], r['var_occ_f1'] or 0., r['furn_f1'] or 0., 'n/a' if r['obj_recall'] is None else '%.3f' % r['obj_recall']))
