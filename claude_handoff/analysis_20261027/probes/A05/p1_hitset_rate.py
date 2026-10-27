"""A05 probe 1 (CPU, demo data): what does the map cost if the decoder re-derives visibility by ray casting from poses?
Theorem used (deterministic first-hit sensor): own map = RayCast(H, poses) where H = observed-occupied voxels, and the R8 union =
max over reached agents of RayCast(H_union, poses_a). So a pose-aware lossless codec only needs H (+ 44 bit of poses per agent).
Measured per map (own / union8 / directed-edge sender side): ctx ternary (as rate_baselines), ctx of the known mask, ctx of H alone,
ctx of H given the known mask; sparse index sets (blocks that contain any hit) at several block sizes and their ctx cost;
hit counts by class (slab / structure / furniture); innovation messages: H of the sender side coded with the receiver side's
ternary map as common side information (two-pass tree schedule)."""
import glob, json, math, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from rate_baselines_lib import load, targets, ctx_bytes, NB, LG  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))), 'demo_web3d', 'data')


def _ctx(sym, ctx_ids, nsym):
    """exact adaptive KT code length in bytes (+2 B) of symbols sym given integer contexts ctx_ids"""
    idx = (ctx_ids * nsym + sym).reshape(-1)
    cnt = torch.bincount(idx, minlength=int(ctx_ids.max()) * nsym + nsym).reshape(-1, nsym).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    a = 0.5 * nsym
    bits = ((LG(N + a) - LG(a)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


def causal_ctx(s, base):
    X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in NB:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return c


def ctx_bin(mask, side=None, side_base=1):
    """binary 9-neighbour causal context code; optional co-located side information (integer map with side_base values)"""
    s = torch.from_numpy(mask.astype(np.int64)); c = causal_ctx(s, 2)
    if side is not None:
        c = c * side_base + torch.from_numpy(side.astype(np.int64))
    return _ctx(s, c, 2)


def blocks_any(m, b):
    X, Y, Z = m.shape
    return m.reshape(X // b[0], b[0], Y // b[1], b[1], Z // b[2], b[2]).any((1, 3, 5))


BLOCKS = {'2x2x2': (2, 2, 2), '4x4x4': (4, 4, 4), '8x8x8': (8, 8, 8), 'col2x2': (2, 2, 32), 'col4x4': (4, 4, 32)}


def main():
  rows = []
  for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    m, own, pred, gt = load(path); n = own.shape[0]
    slab = gt['slab']; obj = gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    T8 = targets(own, m['reached'][m['hops'].index(8)])
    def describe(kind, M, extra):
        H = M == 1; K = M >= 0
        r = dict(kind=kind, house=m['house'], chart_m=m['chart_m'], **extra)
        r['ctx_full'] = ctx_bytes(M); r['ctx_known'] = ctx_bin(K); r['ctx_H'] = ctx_bin(H)
        r['ctx_H_given_K'] = ctx_bin(H, K, 2)
        r['hits'] = int(H.sum()); r['hits_slab'] = int((H & slab).sum()); r['hits_struct'] = int((H & struct).sum()); r['hits_obj'] = int((H & obj).sum())
        r['known'] = int(K.sum())
        for name, b in BLOCKS.items():
            a = blocks_any(H, b); r[f'act_{name}'] = int(a.sum()); r[f'tot_{name}'] = int(a.size)
            r[f'idx_{name}'] = ctx_bin(a) if min(a.shape) > 2 else None
            r[f'actK_{name}'] = int(blocks_any(K, b).sum())
            r[f'actobj_{name}'] = int(blocks_any(H & obj, b).sum())
        return r
    for a in range(n):
        rows.append(describe('own', own[a], dict(agent=a)))
        rows.append(describe('union8', T8[a], dict(agent=a)))
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    def side_of(s, r):
        side, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v != r and v not in side:
                    side.add(v); stack.append(v)
        return sorted(side)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            A = own[side_of(s, r)].max(0); B = own[side_of(r, s)].max(0); ownr = own[r]
            HA = A == 1
            rows.append(dict(kind='edge', house=m['house'], sender=s, receiver=r, side=len(side_of(s, r)),
                             ctx_full=ctx_bytes(A), ctx_H=ctx_bin(HA),
                             ctx_H_given_recvside=ctx_bin(HA, B + 1, 3),        # two-pass: receiver side union known to both ends
                             ctx_H_given_recvown=ctx_bin(HA, ownr + 1, 3),      # (only if the receiver's own map were common knowledge)
                             hits=int(HA.sum()), new_hits=int((HA & (B != 1)).sum())))
    print(m['house'], 'done', flush=True)

  S = {}
  for kind in ('own', 'union8', 'edge'):
    r = [x for x in rows if x['kind'] == kind]
    keys = [k for k in r[0] if isinstance(r[0][k], (int, float)) and k not in ('agent', 'sender', 'receiver')]
    S[kind] = {k: dict(median=float(np.median([x[k] for x in r if x.get(k) is not None])), mean=float(np.mean([x[k] for x in r if x.get(k) is not None])))
               for k in keys if any(x.get(k) is not None for x in r)}
    S[kind]['n'] = len(r)
  json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p1_hitset_rate.json'), 'w'), indent=1)
  for kind in S:
    print('==', kind, 'n', S[kind]['n'])
    for k, v in S[kind].items():
        if k != 'n':
            print(f"  {k:24s} median {v['median']:10.1f}  mean {v['mean']:10.1f}")


if __name__ == '__main__':
    main()
