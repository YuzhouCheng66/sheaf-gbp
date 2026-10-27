"""A02 probe P6 (2026-10-27, CPU only): AMODAL COMPLETION AT THE ENCODER.  Unknown voxels are never sampled by any ray (that is what
'unknown' means), so ANY occupancy G with  hits <= G <= hits | unknown  renders (ray casting from the agent's poses) to exactly the own
map.  The sender may therefore fill the occluded / out-of-range space with whatever its prior finds cheapest ("generate the unseen part so
that the visibility boundary need not be coded"); the receiver renders, and every filled voxel stays invisible.
Variants per own map (6 demo VAL houses x 8 agents) and per round-8 union target:
  V1 hits                 (the fact set of P1 / P2)
  V2 column fill          hits | (unknown voxels in every (x, y) column that contains a hit)
  V3 column + slab fill   V2 | unknown voxels of the floor layer (z = 0) and of every layer at or above the lowest observed ceiling hit
  V4 all unknown filled   hits | unknown  (= complement of the observed free space)
  V5 oracle               the true world occupancy (what a perfect amodal prior would generate; not available to the sender)
Code length: adaptive KT binary ctx (9 causal neighbours) and a leave-one-house-out static model (nb13 + layer + range, P2).
Exactness is checked on synthetic maps (P1 sensor) for V2 / V3: render(V, poses) == own map.  Output: probes/A02/p6_amodal.json"""
import glob, json, math, os, sys, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets  # noqa: E402
from p1_render_hits import observe, fibonacci_sphere, ctx_bytes_binary  # noqa: E402
from p2_prior_ctx import Static, range_bucket, NB13  # noqa: E402
torch.set_num_threads(8)


def variants(T, occ_true):
    hits = T == 1; unk = T < 0
    colhit = hits.any(2, keepdims=True)
    v2 = hits | (unk & colhit)
    zc = np.arange(T.shape[2])
    ceil_hits = hits & (zc[None, None, :] >= T.shape[2] // 2)
    lo = int(zc[ceil_hits.any((0, 1))].min()) if ceil_hits.any() else T.shape[2]
    slabz = (zc == 0) | (zc >= lo)
    v3 = v2 | (unk & slabz[None, None, :])
    v4 = hits | unk
    return dict(V1=hits, V2=v2, V3=v3, V4=v4, V5=occ_true.astype(bool))


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_web3d', 'data')
    paths = sorted(glob.glob(os.path.join(data, 'house_*.json')))
    # exactness check on synthetic maps (2 agents x 2 houses)
    rays = fibonacci_sphere(65536); checks = []
    for path in paths[:2]:
        m, own, pred, gt = load(path); occ = gt['occ'].astype(np.uint8)
        for a in range(2):
            rng = np.random.default_rng(zlib.crc32(m['house'].encode()) * 16 + a)
            jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.; poses = np.asarray(m['positions_m'][a])[None] + jit
            syn = observe(occ, m['cell_m'], poses, rays); V = variants(syn, gt['occ'])
            for k in ('V2', 'V3', 'V4', 'V5'):
                checks.append(dict(house=m['house'], agent=a, variant=k, mismatch=int((observe(V[k].astype(np.uint8), m['cell_m'], poses, rays) != syn).sum())))
            print('check', checks[-4:], flush=True)
    houses = []
    for path in paths:
        m, own, pred, gt = load(path); n = own.shape[0]
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        pos = [np.asarray(p, float) for p in m['positions_m']]
        items = []
        for kind, maps in (('own', own), ('union8', T8)):
            for a in range(n):
                cs = [pos[a]] if kind == 'own' else [pos[j] for j in np.nonzero(R8[a])[0]]
                items.append(dict(kind=kind, agent=a, V=variants(maps[a], gt['occ']), rb=range_bucket(cs, m['cell_m'], own.shape[1:]).to(torch.int8)))
        houses.append(dict(name=m['house'], items=items)); print('loaded', m['house'], flush=True)
    rows = []
    for hold in (1, 2, 5):                                   # held-out houses r1_i1, r2_i0, r3_i1 (CPU is shared: 3 folds)
        for kind in ('own', 'union8'):
            models = {}
            for vk in ('V1', 'V2', 'V3'):
                tr = [(torch.from_numpy(it['V'][vk].astype(np.int64)), it['rb'].long()) for h_, H in enumerate(houses) if h_ != hold for it in H['items'] if it['kind'] == kind]
                models[vk] = Static(2, NB13).fit(tr); del tr
            for it in houses[hold]['items']:
                if it['kind'] != kind:
                    continue
                r = dict(house=houses[hold]['name'], kind=kind, agent=it['agent'])
                for vk, Vm in it['V'].items():
                    r[f'{vk}_adaptive'] = ctx_bytes_binary(Vm)
                    if vk in models:
                        r[f'{vk}_static'] = models[vk].bits(torch.from_numpy(Vm.astype(np.int64)), it['rb'].long())
                rows.append(r); print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    S = {kind: {k: dict(median=float(np.median([r[k] for r in rows if r['kind'] == kind])), mean=float(np.mean([r[k] for r in rows if r['kind'] == kind])))
                for k in rows[0] if k not in ('house', 'kind', 'agent')} for kind in ('own', 'union8')}
    S['exactness_checks'] = checks
    print(json.dumps(S, indent=1))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p6_amodal.json'), 'w'), indent=1)
