"""A09 probe 1b (Claude, 2026-10-27, CPU only): code the WORLD, not the observation.

By construction of observe() (first-hit marching), observe(W, poses) == own map for EVERY set W with S <= W <= S | U, where S = observed
occupied, F = sampled free, U = never sampled (unknown): rays sample only F voxels before their first hit (which is in S), so any voxel of U
may be occupied or not without changing a single observed voxel. The encoder may therefore send the CHEAPEST world in the interval [S, S | U];
the receiver re-simulates the sensor (poses, ~24 B / agent as float32 jitters) and recovers the observed map exactly.
Candidates (all asserted to satisfy W & F == 0):
  S            the observed surfaces themselves (probe 1);
  grav         S + 'gravity / gap fill' through U only, no GT: in every column, U voxels below an S voxel down to the next non-U voxel, and U
               voxels vertically between two S voxels (furniture tops become solid blocks, holes in walls close) -- what a dumb encoder can do;
  gt_cols      GT occupancy restricted to the columns that contain any S voxel (oracle completion, shows what a learned prior could reach);
  gt_cols_d2   the same with the column footprint dilated by 2 cells (whole objects);
  gt_world     the whole-house GT occupancy (a single world valid for every agent).
Measured: ctx (binary, 9-voxel causal context) bytes, for own maps, R8 unions and sender-side edge unions of the 6 demo houses.
Output: probes/A09/p1b_world_interval.json"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a09lib import load, targets, ctx_bytes_binary   # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))


def grav_fill(m):
    """m ternary [X, Y, Z] (z up). Fill U voxels (-1) below an S voxel down to the next non-U voxel, and U voxels between two S voxels."""
    S = m == 1; U = m == -1
    W = S.copy()
    X, Y, Z = m.shape
    carry = np.zeros((X, Y), bool)
    for z in range(Z - 1, -1, -1):                                   # downward sweep: propagate 'occupied above' through U
        carry = S[:, :, z] | (carry & U[:, :, z])
        W[:, :, z] |= carry & U[:, :, z]
    below = np.zeros((X, Y), bool); up = np.zeros_like(W)
    for z in range(Z):                                               # upward sweep: U voxels that have an S voxel somewhere below in the run
        below = S[:, :, z] | (below & U[:, :, z]); up[:, :, z] = below & U[:, :, z]
    above = np.zeros((X, Y), bool); down = np.zeros_like(W)
    for z in range(Z - 1, -1, -1):
        above = S[:, :, z] | (above & U[:, :, z]); down[:, :, z] = above & U[:, :, z]
    W |= up & down                                                   # U runs bounded by S on both sides
    return W


def dil2d(c, r):
    o = c.copy()
    for _ in range(r):
        p = o.copy(); p[1:] |= o[:-1]; p[:-1] |= o[1:]; p[:, 1:] |= o[:, :-1]; p[:, :-1] |= o[:, 1:]; o = p
    return o


def variants(m, gt_occ):
    S = m == 1; F = m == 0
    cols = S.any(2)
    out = dict(S=S, grav=grav_fill(m), gt_cols=gt_occ & cols[:, :, None], gt_cols_d2=gt_occ & dil2d(cols, 2)[:, :, None], gt_world=gt_occ)
    for k, W in out.items():
        assert not (W & F).any(), k
        assert (W >= S).all(), k
    return out


def main():
    rows = dict(own=[], union8=[], edge=[]); t0 = time.time(); world_cache = {}
    for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
        m, own, pred, gt = load(path); n = own.shape[0]; G = gt['occ']
        gw = ctx_bytes_binary(G)
        T8 = targets(own, m['reached'][m['hops'].index(8)])
        for name, maps in (('own', own), ('union8', T8)):
            for a in range(n):
                v = variants(maps[a], G)
                rows[name].append(dict(house=m['house'], agent=a, **{k: (gw if k == 'gt_world' else ctx_bytes_binary(W)) for k, W in v.items()}))
        adj = {i: set() for i in range(n)}
        for i, j in m['edges']:
            adj[i].add(j); adj[j].add(i)
        for i, j in m['edges']:
            for s, r in ((i, j), (j, i)):
                side, stack = {s}, [s]
                while stack:
                    u = stack.pop()
                    for w in adj[u]:
                        if w != r and w not in side:
                            side.add(w); stack.append(w)
                U = own[sorted(side)].max(0)
                v = variants(U, G)
                rows['edge'].append(dict(house=m['house'], sender=s, side=len(side), **{k: (gw if k == 'gt_world' else ctx_bytes_binary(W)) for k, W in v.items()}))
        print(m['house'], f'{time.time() - t0:.0f}s', flush=True)

    summary = {}
    for name, r in rows.items():
        summary[name] = {k: float(np.median([x[k] for x in r])) for k in ('S', 'grav', 'gt_cols', 'gt_cols_d2', 'gt_world')}
        summary[name].update({f'ratio_S_over_{k}': float(np.median([x['S'] / x[k] for x in r])) for k in ('grav', 'gt_cols', 'gt_cols_d2')})
        if name == 'edge':
            summary[name]['side_median'] = float(np.median([x['side'] for x in r]))
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, 'p1b_world_interval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
