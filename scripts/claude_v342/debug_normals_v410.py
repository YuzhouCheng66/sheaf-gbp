"""Debug helper (Claude, 2026-10-24): which way do Replica floor normals point (vertex normals vs triangle winding)?"""
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')
import prep_realscan_base_v410 as P
for scene in sys.argv[1:]:
    V, F, VN = P.read_replica_ply(f'/mnt/datasets/replica_v1/{scene}/mesh.ply', normals=True)
    g = json.load(open(f'/mnt/datasets/replica_v1/{scene}/habitat/info_semantic.json'))['gravity_dir']
    R = P.gravity_rotation(g); V = V @ R.T; VN = VN @ R.T; V = V - V.min(0)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1); nzg = n[:, 2] / np.maximum(a, 1e-12)
    nzv = VN[F].mean(1)[:, 2]; zc = V[F, 2].mean(1)
    floors = P.floors_of(V, F); zf = floors[0]
    near = (np.abs(zc - zf) < 0.05) & (np.abs(nzg) > 0.9)
    horiz = np.abs(nzg) > 0.9; low = horiz & (zc < np.percentile(zc[horiz], 30))
    print(scene, 'floors', np.round(floors, 2), '| floor-level tris', int(near.sum()), 'area-weighted winding nz', round(float(np.sum(a[near] * nzg[near]) / a[near].sum()), 3),
          'vertex nz', round(float(np.sum(a[near] * nzv[near]) / a[near].sum()), 3), '| lowest-30% horizontal: winding', round(float(np.sum(a[low] * nzg[low]) / a[low].sum()), 3),
          'vertex', round(float(np.sum(a[low] * nzv[low]) / a[low].sum()), 3), '| zmin..zf area below floor-0.1:', round(float(a[horiz & (zc < zf - 0.1)].sum() / 2), 2), 'm^2')
