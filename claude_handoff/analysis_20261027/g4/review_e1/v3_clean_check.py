"""v3_clean_check.py: clean tier under the per-beam sequential-clamp integrator (V3) equals observe() voxel for voxel (every agent),
and the R8 sum / max_occ fusions equal the R8 targets; two VAL houses, CPU, single process."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1b_perbeam_probe as P  # noqa: E402
G = P.G; W = G.W
out = []
for rooms, index in ((1, 0), (2, 0)):
    fx = W.make_fixture('VALIDATION', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']; n = private.shape[0]
    ds = 0.5 * float(cell.min()); rays = W.fibonacci_sphere(W.BEAMS); F = G.forest(fx['edges'], n)
    T8 = G.stream_targets(private, np.asarray(fx['reached'][G.HOPS.index(8)], bool))
    maps = []
    for a in range(n):
        scans = []
        for j in range(traj.shape[0]):
            hs, ls = G.trace_pose(occ, cell, traj[j, a], rays)
            scans.append(G.measure(G.CONDITIONS['clean'], hs, ls, traj[j, a], G.pose_draws(1, rooms, index, a, j, len(rays)), ds, rays)[0])
        maps.append(P.agent_maps(occ.shape, cell, ds, scans)['V3_perbeam'])
    k0 = all(np.array_equal(G.tern(maps[a]), private[a]) for a in range(n))
    r8 = {kind: all(np.array_equal(p, T8[a]) for a, p in enumerate(G.fuse(maps, F, kind)[1])) for kind in G.FUSIONS}
    out.append(dict(house=f'r{rooms}_i{index}', K0_exact=k0, R8_exact=r8))
    print(out[-1], flush=True)
json.dump(out, open(os.path.join(HERE, 'v3_clean_check.json'), 'w'), indent=1)
