"""p4_render_known_ours.py (A03): zero-training test of a RENDER-KNOWN decoder on our current best (3S-X + calibrated rule, demo planes).
The learned known head is replaced by ray casting from the transmitted poses through the decoded occupancy:
  ours      = the exported prediction (known head + occupancy head, calibrated rule);
  rk_free   = world := predicted-occupied voxels (everything else free); known := what the rays of the poses reach; occupied := hits
              (leaks through holes in the decoded surfaces; hidden predicted voxels become unknown);
  rk_block  = world := predicted-occupied OR predicted-unknown; a hit on a predicted-unknown voxel is reported unknown
              (visibility-consistent version of the model's own known region; can only shrink it).
K0: own poses; R8: poses of every reached agent. Limitation: the demo stores only THRESHOLDED planes (no logits), so the occupancy
outside the predicted-known region is unavailable; a trained render-known model would also be trained to close its surfaces.
Usage: python -B p4_render_known_ours.py"""
import json, os, sys, time
import numpy as np
from a03np import observe_np as observe_t, cell_of, targets, load_fx, score_np as score, fpr, HERE
sys.path.insert(0, os.path.join(HERE, '..', '..'))

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))


def load(path):
    """= rate_baselines_lib.load without torch"""
    import base64, gzip
    m = json.load(open(path)); raw = gzip.decompress(base64.b64decode(m['bin_gz_b64'])); DIMS = (128, 128, 32)

    def plane(name):
        o, n = m['offsets'][name]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:int(np.prod(DIMS))].astype(bool).reshape(DIMS)
    n = m['agents']
    own = np.stack([np.where(plane(f'own_known_{a}'), plane(f'own_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)])
    pred = {k: np.stack([np.where(plane(f'pred_{k}_known_{a}'), plane(f'pred_{k}_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)]) for k in m['hops']}
    return m, own, pred, None


def house(name):
    t0 = time.time()
    m, own, pred, gt = load(os.path.join(ROOT, 'demo_web3d', 'data', f'house_{name}.json'))
    fx, meta = load_fx(name); assert (fx['private'] == own).all()
    cell = cell_of(meta['chart_m']); traj = fx['trajectory']; n = own.shape[0]
    slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = fx['occ'].astype(bool) & ~obj & ~slab
    rows = []
    for k in (0, 8):
        R = np.asarray(m['reached'][m['hops'].index(k)], bool); T = targets(own, R)
        for a in range(n):
            P = pred[k][a]; ps = [traj[:, b] for b in np.nonzero(R[a])[0]]
            row = dict(house=name, hop=k, agent=a)
            row['ours'] = dict(**score(P, T[a], slab, obj, struct), fpr=fpr(P, T[a]))
            U = None
            for pose in ps:
                o = observe_t(P == 1, cell, pose); U = o if U is None else np.maximum(U, o)
            row['rk_free'] = dict(**score(U, T[a], slab, obj, struct), fpr=fpr(U, T[a]))
            V = None
            for pose in ps:
                o = observe_t((P == 1) | (P < 0), cell, pose); o = np.where((o == 1) & (P < 0), np.int8(-1), o); V = o if V is None else np.maximum(V, o)
            row['rk_block'] = dict(**score(V, T[a], slab, obj, struct), fpr=fpr(V, T[a]))
            rows.append(row)
    print(name, f'{time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    import multiprocessing as mp
    hs = ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1']
    with mp.Pool(6) as pool:
        rows = [r for rr in pool.map(house, hs) for r in rr]
    S = {}
    for k in (0, 8):
        for v in ('ours', 'rk_free', 'rk_block'):
            S[f'K{k}|{v}'] = {key: float(np.mean([r[v][key] for r in rows if r['hop'] == k and r[v][key] is not None])) for key in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr')}
    for key, v in S.items():
        print(f"{key:14s} surf {v['surface_f1']:.4f} var {v['var_occ_f1']:.4f} furn {v['furn_f1']:.4f} fpr {v['fpr']:.4f}")
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p4_render_known_ours.json'), 'w'), indent=1)
