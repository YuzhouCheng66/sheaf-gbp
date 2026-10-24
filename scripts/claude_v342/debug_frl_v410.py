"""Debug helper (Claude, 2026-10-24): why do the Replica FRL apartments yield ~4 m^2 of free floor in prep_realscan_base_v410?"""
import sys
import numpy as np, open3d as o3d
sys.path.insert(0, 'scripts')
import prep_realscan_base_v410 as P

scene = sys.argv[1] if len(sys.argv) > 1 else 'frl_apartment_0'
V, F = P.read_replica_ply(f'/mnt/datasets/replica_v1/{scene}/mesh.ply'); V = V - V.min(0)
print('extent', V.max(0).round(2), 'faces', len(F))
floors = P.floors_of(V, F); print('floors', np.round(floors, 2))
zf = floors[0]
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
shape = (int(np.ceil(V[:, 1].max() / P.CELL)), int(np.ceil(V[:, 0].max() / P.CELL)))
ny, nx = shape
X, Y = np.meshgrid((np.arange(nx) + .5) * P.CELL, (np.arange(ny) + .5) * P.CELL)
for h in (0.8, 1.2, 1.6, 2.0):
    Q = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf + h)], 1).astype(np.float32)
    down = np.concatenate([Q, np.tile(np.float32([0, 0, -1]), (len(Q), 1))], 1); up = np.concatenate([Q, np.tile(np.float32([0, 0, 1]), (len(Q), 1))], 1)
    td = sc.cast_rays(o3d.core.Tensor(down))['t_hit'].numpy(); tu = sc.cast_rays(o3d.core.Tensor(up))['t_hit'].numpy()
    fin_d = np.isfinite(td); fin_u = np.isfinite(tu)
    print(f'origin floor+{h}: down hit {fin_d.mean():.3f} (median {np.median(td[fin_d]):.2f} m, <1.2: {(td < 1.2).mean():.3f}), up hit {fin_u.mean():.3f} (median {np.median(tu[fin_u]):.2f}, <3.5: {(tu < 3.5).mean():.3f}); interior {((td < h + 0.4) & (tu < 3.5)).mean():.3f}')
# area of horizontal triangles by height (does the floor exist everywhere?)
n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1) / 2; nz = n[:, 2] / np.maximum(2 * a, 1e-12); zc = V[F, 2].mean(1)
h, e = np.histogram(zc, bins=np.arange(0, V[:, 2].max() + 0.2, 0.2), weights=a * (np.abs(nz) > 0.9))
print('horizontal area by height (m^2):', {f'{e[i]:.1f}': round(float(h[i]), 1) for i in range(len(h)) if h[i] > 0.5})
rng = np.random.default_rng(0)
obs = P.band_obstacles(V, F, zf, shape, rng)
print('obstacle fraction of raster', round(float(obs.mean()), 3), 'area m^2', round(float(obs.sum() * P.CELL ** 2), 1), 'raster area', round(nx * ny * P.CELL ** 2, 1))
