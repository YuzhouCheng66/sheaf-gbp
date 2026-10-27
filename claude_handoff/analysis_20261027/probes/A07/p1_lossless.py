"""p1_lossless.py (A07): is ctx really the strongest LOSSLESS baseline?
Claim tested: an own map is a deterministic function of (hit set H = occupied voxels, sensor poses, public beam pattern), because a ray of
the map stops exactly at the first voxel of H on its path (every voxel of H is truly occupied; every voxel a ray crossed is truly free).
So 'H + poses, decoder re-casts the rays' is a LOSSLESS codec of the own map and of any union (a union = cast every reached agent's
poses through the union's H). We measure its size (binary KT context code of H + 5.5 B per agent of poses) next to ctx / gzip / lzma
and an OctoMap-binary (.bt) size estimate, on the 6 VAL demo houses (own maps, round-8 unions, directed tree-edge sender unions),
and we VERIFY exactness by re-simulating own maps with our own jittered poses (exact float poses, and poses rounded to 1 cm)."""
import gzip as _gz, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a07_lib import (load, targets, ctx_ter, ctx_bin, gz, observe, RAYS, POSE_BYTES, demo_paths)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p1_lossless.json')


def bt_bytes(m):
    """OctoMap binary (.bt) size: 2 bytes per inner node (8 children x 2 bit), leaves pruned when 8 equal known children; z padded to a cube."""
    X, Y, Z = m.shape; n = max(X, Y, Z)
    st = np.zeros((n, n, n), np.int8); st[:X, :Y, :Z] = np.where(m < 0, 0, np.where(m == 1, 1, 2))   # 0 unknown 1 occ 2 free 3 inner
    inner = 0; stream = []
    while st.shape[0] > 1:
        k = st.shape[0] // 2
        ch = st.reshape(k, 2, k, 2, k, 2).transpose(0, 2, 4, 1, 3, 5).reshape(k, k, k, 8)
        allu = (ch == 0).all(-1); same = (ch == ch[..., :1]).all(-1) & (ch[..., 0] != 0) & (ch[..., 0] != 3)
        par = np.where(allu, 0, np.where(same, ch[..., 0], 3)).astype(np.int8)
        isin = par == 3; inner += int(isin.sum())
        code = {0: 0, 1: 1, 2: 2, 3: 3}
        sel = ch[isin]                                                     # child codes of inner nodes (2 bit each)
        packed = (sel[:, 0::4].astype(np.uint8) << 6 | sel[:, 1::4].astype(np.uint8) << 4 | sel[:, 2::4].astype(np.uint8) << 2 | sel[:, 3::4].astype(np.uint8))
        stream.append(packed.tobytes())
        st = par
    raw = b''.join(stream[::-1])
    return 2 * inner, len(_gz.compress(raw, 9))


def sizes(m):
    H = m == 1
    bt, btgz = bt_bytes(m)
    return dict(ctx=ctx_ter(m), gzip=gz(m), H_ctx=ctx_bin(H), known_ctx=ctx_bin(m >= 0), n_occ=int(H.sum()), known=float((m >= 0).mean()), bt=bt, bt_gz=btgz)


def house(path):
    t0 = time.time()
    meta, own, pred, gt = load(path); n = own.shape[0]
    rows = dict(house=meta['house'], own=[], union8=[], edge=[], exact={})
    for a in range(n):
        r = sizes(own[a]); r['H_plus_poses'] = r['H_ctx'] + POSE_BYTES; rows['own'].append(r)
    hi8 = meta['hops'].index(8); R8 = np.asarray(meta['reached'][hi8], bool); T8 = targets(own, R8)
    for a in range(n):
        r = sizes(T8[a]); r['reached'] = int(R8[a].sum()); r['H_plus_poses'] = r['H_ctx'] + POSE_BYTES * r['reached']; rows['union8'].append(r)
    adj = {i: set() for i in range(n)}
    for i, j in meta['edges']:
        adj[i].add(j); adj[j].add(i)
    for i, j in meta['edges']:
        for s, rcv in ((i, j), (j, i)):
            side, stack = {s}, [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v != rcv and v not in side:
                        side.add(v); stack.append(v)
            U = own[sorted(side)].max(0)
            rows['edge'].append(dict(sender=s, receiver=rcv, side=len(side), ctx=ctx_ter(U), gzip=gz(U), H_ctx=ctx_bin(U == 1), H_plus_poses=ctx_bin(U == 1) + POSE_BYTES * len(side)))
    # ---- exactness by re-simulation (our own jitter; the demo does not carry the private jitter)
    occ = gt['occ'].astype(np.uint8); cell = np.asarray(meta['cell_m'], np.float64)
    rng = np.random.default_rng(7000 + sum(map(ord, meta['house'])))
    pos = np.asarray(meta['positions_m'], np.float64)
    sim, poses_all = [], []
    for a in range(n):
        jit = rng.uniform(-.1, .1, (3, 3)); jit[:, 2] = 0.
        poses = pos[a][None] + jit; poses_all.append(poses)
        sim.append(observe(occ, cell, poses))
    sim = np.stack(sim)
    mism, mism_q, mism_u = [], [], []
    for a in range(n):
        H = (sim[a] == 1).astype(np.uint8)
        rec = observe(H, cell, poses_all[a]); mism.append(int((rec != sim[a]).sum()))
        recq = observe(H, cell, np.round(poses_all[a], 2)); mism_q.append(int((recq != sim[a]).sum()))
    U = sim[R8[0]].max(0); Hu = (U == 1).astype(np.uint8)
    recu = np.stack([observe(Hu, cell, poses_all[b]) for b in np.nonzero(R8[0])[0]]).max(0)
    rows['exact'] = dict(own_mismatch_voxels=mism, own_mismatch_voxels_pose_1cm=mism_q, union_agent0_mismatch=int((recu != U).sum()),
                         sim_known_mean=float((sim >= 0).mean()), sim_ctx=[ctx_ter(sim[a]) for a in range(n)], sim_H_ctx=[ctx_bin(sim[a] == 1) for a in range(n)])
    print(meta['house'], f'{time.time() - t0:.0f}s', 'mismatch', mism, mism_q, rows['exact']['union_agent0_mismatch'], flush=True)
    return rows


if __name__ == '__main__':
    with Pool(6) as pool:
        res = pool.map(house, demo_paths())
    def med(grp, key):
        return float(np.median([r[key] for h in res for r in h[grp]]))
    S = {}
    for grp in ('own', 'union8', 'edge'):
        keys = ['ctx', 'gzip', 'H_ctx', 'H_plus_poses'] + (['known_ctx', 'bt', 'bt_gz', 'n_occ'] if grp != 'edge' else [])
        S[grp] = {k: med(grp, k) for k in keys}
        S[grp]['ratio_ctx_over_Hposes_median'] = float(np.median([r['ctx'] / r['H_plus_poses'] for h in res for r in h[grp]]))
    S['exact'] = dict(own_mismatch_total=int(sum(sum(h['exact']['own_mismatch_voxels']) for h in res)),
                      own_mismatch_pose1cm_total=int(sum(sum(h['exact']['own_mismatch_voxels_pose_1cm']) for h in res)),
                      own_mismatch_pose1cm_max=int(max(max(h['exact']['own_mismatch_voxels_pose_1cm']) for h in res)),
                      union_mismatch_total=int(sum(h['exact']['union_agent0_mismatch'] for h in res)),
                      sim_ctx_median=float(np.median([v for h in res for v in h['exact']['sim_ctx']])),
                      sim_H_ctx_median=float(np.median([v for h in res for v in h['exact']['sim_H_ctx']])))
    json.dump(dict(summary=S, houses=res), open(OUT, 'w'), indent=1)
    print(json.dumps(S, indent=1))
