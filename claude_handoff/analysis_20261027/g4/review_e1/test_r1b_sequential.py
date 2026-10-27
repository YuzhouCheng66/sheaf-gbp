"""check of the reviewer's per-beam sequential clamp (r1b_perbeam_probe.apply_sequential) against a naive Python loop, and of
scan_events against g4_noise_panels.integrate_scan (counts) on a small synthetic world with a noisy scan."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1b_perbeam_probe as P  # noqa: E402
G = P.G

rng = np.random.default_rng(0)
for trial in range(20):
    V = 50; n = 400
    v = rng.integers(0, V, n); b = rng.permutation(n)                 # each (voxel, beam) at most once is not needed here
    d = np.where(rng.random(n) < .4, G.L_HIT, G.L_MISS).astype(np.int64)
    L = rng.integers(-40, 71, V).astype(np.int64); ref = L.copy()
    for i in np.lexsort((b, v)):
        ref[v[i]] = min(max(ref[v[i]] + d[i], G.L_MIN), G.L_MAX)
    P.apply_sequential(L, v, b, d)
    assert np.array_equal(L, ref), trial

occ = np.zeros((24, 20, 12), np.uint8); occ[0] = occ[-1] = 1; occ[:, 0] = occ[:, -1] = 1; occ[:, :, 0] = occ[:, :, -1] = 1; occ[6:10, 5:9, 1:5] = 1
cell = np.array([.1, .1, .1]); ds = 0.05; rays = G.W.fibonacci_sphere(4096); p = np.array([1.25, 1.05, .65])
hs, ls = G.trace_pose(occ, cell, p, rays)
scan, _ = G.measure(G.CONDITIONS['cam'], hs, ls, p, G.pose_draws(1, 1, 0, 0, 0, len(rays)), ds, rays)
nh, nm = G.integrate_scan(occ.shape, cell, ds, *scan)
v, b, d, nh2, nm2 = P.scan_events(occ.shape, cell, ds, *scan)
assert np.array_equal(nh, nh2) and np.array_equal(nm, nm2)
print('ok: sequential clamp = naive loop (20 trials); scan_events counts = integrate_scan')
