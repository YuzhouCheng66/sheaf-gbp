"""Base floor rasters for the real-scan 2D world family v410 (Claude, 2026-10-24). Run once per dataset on uk-linux (CPU).

Review v410 M1 / M2 (applied 2026-10-24, before any training): every mesh is first rotated so that -gravity = +z, using Replica's own
habitat/info_semantic.json gravity_dir (all 18 scenes are ~ -z; room_2 is tilted 8.7 deg, apartment_0 1.3 deg, apartment_1 0.9 deg);
for HM3D (no such field) the area-weighted mean normal of the floor triangles will be used. A floor cell counts as interior only if
the downward ray hits an UPWARD-facing surface (triangle-winding normal z >= 0.7; Replica's stored vertex normals point INTO
the surfaces, z ~ -1 on floors, while the winding normals point out, z ~ +1, in every scene checked), so the back of the ceiling below (apartment_0 f1) no longer
counts. Each floor also gets its main wall direction (wall_angle_deg in [0, 90): area-weighted histogram of the wall-triangle normal
directions mod 90 deg), which the world module uses to align the chart with the building as v371 does for HouseExpo.
For every scene mesh and every detected floor, writes a 2 cm raster in the gravity-aligned scene frame (translated so the mesh bbox minimum is 0):
  obstacle[iy, ix]  some mesh surface lies in the height band [floor + 0.15 m, floor + 1.60 m] above that 2 cm column
                    (area-weighted surface sampling at 4 points / cm^2 of the triangles crossing the band, clipped to the band),
  interior[iy, ix]  the first surface hit by a ray straight down from floor + 0.8 m faces up and lies at the floor height +-0.10 m: only the scanned inside of the building has a floor; the void seen
                    through exterior windows, scan holes and stairwells / double-height voids do not. (No ceiling test: the Replica FRL
                    apartments were scanned without ceilings; upward rays hit something over only ~20 % of their floor.)
free = interior & ~obstacle is the floor's navigable / visible free space; everything else is "wall" in the v371 contract. Glass is
not part of the Replica meshes: interior glass is free in the slice; beyond an exterior window there is no floor, hence wall.
Projection convention (review S8): any surface within the 0.15-1.60 m band blocks the 2D beam (a table top hides what is under it).
Replica's mesh.ply files are quad meshes in a binary layout Open3D partly fails on, so they are parsed directly (quads split in two).
Writes <out>/<scene>_f<k>.npz and <out>/index.json (per-file sha256, scene list, floors, extents).
python prep_realscan_base_v410.py --dataset replica --root /mnt/datasets/replica_v1 --out /mnt/datasets/yuzhou-sheaf-gbp-research/realscan-v410/replica-base
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import open3d as o3d

CELL = 0.02
BAND = (0.15, 1.60)
POINTS_PER_CM2 = 4.0


def read_replica_ply(path, normals=False):
    raw = open(path, 'rb').read(); end = raw.index(b'end_header\n') + len(b'end_header\n'); header = raw[:end].decode('ascii').splitlines()
    iv = next(i for i, l in enumerate(header) if l.startswith('element vertex')); jf = next(i for i, l in enumerate(header) if l.startswith('element face'))
    nv = int(header[iv].split()[-1]); nf = int(header[jf].split()[-1])
    tmap = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'double': '<f8'}
    vdt = np.dtype([(l.split()[2], tmap[l.split()[1]]) for l in header[iv + 1:jf] if l.startswith('property')])
    vert = np.frombuffer(raw, vdt, nv, end); off = end + nv * vdt.itemsize; cnt = raw[off]
    faces = np.frombuffer(raw, np.dtype([('n', 'u1'), ('idx', '<i4', (cnt,))]), nf, off)
    assert (faces['n'] == cnt).all(), 'mixed polygon sizes'
    idx = faces['idx'].astype(np.int64)
    F = idx if cnt == 3 else np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
    V = np.stack([vert['x'], vert['y'], vert['z']], 1).astype(np.float64)
    if normals:
        return V, F, np.stack([vert['nx'], vert['ny'], vert['nz']], 1).astype(np.float64)
    return V, F


def gravity_rotation(g):
    """rotation matrix R with R @ (-g / |g|) = +z (Rodrigues)."""
    a = -np.asarray(g, np.float64); a /= np.linalg.norm(a); b = np.array([0., 0., 1.])
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def wall_angle_deg(V, F, zf):
    """main wall direction in [0, 90) degrees: area-weighted histogram (0.5 deg bins, circular smoothing) of the horizontal direction
    of the normals of near-vertical triangles (|n_z| < 0.3) that cross the floor's obstacle band, taken mod 90 deg (walls are mostly
    rectilinear). Exact for planar walls; computed from the mesh, not the raster (a raster gradient is biased by pixel staircases)."""
    lo, hi = zf + BAND[0], zf + BAND[1]
    zt = V[F, 2]; sel = (zt.max(1) >= lo) & (zt.min(1) <= hi)
    T = V[F[sel]]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); a = np.linalg.norm(n, axis=1) / 2
    nn = n / np.maximum(2 * a[:, None], 1e-12)
    vert = np.abs(nn[:, 2]) < 0.3
    ang = np.degrees(np.arctan2(nn[vert, 1], nn[vert, 0])) % 90.
    hist, edges = np.histogram(ang, bins=180, range=(0., 90.), weights=a[vert])
    k = np.array([1, 4, 6, 4, 1], np.float64); k /= k.sum()
    hs = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode='same')[2:-2]
    j = int(np.argmax(hs))
    return float((edges[j] + edges[j + 1]) / 2)


def floors_of(V, F):
    """floor heights: area-weighted histogram of upward-facing triangle heights, peaks >= 15 % of the largest and >= 2 m apart."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1) / 2
    nz = n[:, 2] / np.maximum(2 * a, 1e-12); z = V[F, 2].mean(1)
    hist, edges = np.histogram(z, bins=np.arange(0, V[:, 2].max() + 0.1, 0.05), weights=a * (nz > 0.9))
    floors = []
    for i in np.argsort(-hist):
        if hist[i] < 0.15 * hist.max():
            break
        zf = (edges[i] + edges[i + 1]) / 2
        if all(abs(zf - f) > 2.0 for f in floors):
            floors.append(float(zf))
    return sorted(floors)


def band_obstacles(V, F, zf, shape, rng):
    lo, hi = zf + BAND[0], zf + BAND[1]
    zt = V[F, 2]
    sel = (zt.max(1) >= lo) & (zt.min(1) <= hi)
    T = V[F[sel]]                                                                   # [m,3,3]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    k = np.maximum(1, np.ceil(area * 1e4 * POINTS_PER_CM2)).astype(np.int64)
    occ = np.zeros(shape, bool)
    for start in range(0, len(T), 200000):                                          # chunks keep memory bounded
        Tc, kc = T[start:start + 200000], k[start:start + 200000]
        rep = np.repeat(np.arange(len(Tc)), kc)
        u, v = rng.random(len(rep)), rng.random(len(rep)); flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        P = Tc[rep, 0] + u[:, None] * (Tc[rep, 1] - Tc[rep, 0]) + v[:, None] * (Tc[rep, 2] - Tc[rep, 0])
        P = P[(P[:, 2] >= lo) & (P[:, 2] <= hi)]
        ix = np.clip((P[:, 0] / CELL).astype(np.int64), 0, shape[1] - 1); iy = np.clip((P[:, 1] / CELL).astype(np.int64), 0, shape[0] - 1)
        occ[iy, ix] = True
    return occ


def down_hits(scene, zf, shape, F, VN):
    """per 2 cm column: distance of the first surface below floor + 0.8 m and the winding-normal z of the hit triangle (VN unused:
    kept in the signature so callers stay unchanged; Replica's vertex normals are inverted)."""
    ny, nx = shape
    X, Y = np.meshgrid((np.arange(nx) + .5) * CELL, (np.arange(ny) + .5) * CELL)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf + 0.8)], 1).astype(np.float32)
    down = np.concatenate([P, np.tile(np.float32([0, 0, -1]), (len(P), 1))], 1)
    r = scene.cast_rays(o3d.core.Tensor(down))
    td = r['t_hit'].numpy(); pid = r['primitive_ids'].numpy().astype(np.int64)
    hit = np.isfinite(td)
    nz = np.full(len(td), np.nan)
    Vt = scene_vertices(scene)
    T = Vt[F[pid[hit]]]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); nz[hit] = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    return td.reshape(ny, nx), nz.reshape(ny, nx)


_SCENE_V = {}
def scene_vertices(scene):
    return _SCENE_V[id(scene)]


def make_scene(V, F):
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    _SCENE_V[id(sc)] = V
    return sc


def interior_of(scene, zf, shape, F, VN):
    # the first surface below floor + 0.8 m must face up and lie at the floor height (+-0.10 m): stair treads below a landing, the
    # back of the ceiling below and voids are not floor (test G1 found 1.3 % stair-tread cells on apartment_0 f1 with a 0.4 m window)
    td, nz = down_hits(scene, zf, shape, F, VN)
    return (td >= 0.70) & (td <= 0.90) & (nz >= 0.7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True, choices=['replica']); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    scenes = sorted(p.name for p in Path(args.root).iterdir() if (p / 'mesh.ply').exists())
    rows = []; t0 = time.time()
    for scene in scenes:
        V, F, VN = read_replica_ply(str(Path(args.root) / scene / 'mesh.ply'), normals=True)
        g = json.load(open(Path(args.root) / scene / 'habitat' / 'info_semantic.json'))['gravity_dir']   # review M1: gravity alignment
        R = gravity_rotation(g)
        V = V @ R.T; VN = VN @ R.T
        V = V - V.min(0)
        tilt = float(np.degrees(np.arccos(np.clip(-np.asarray(g) @ [0, 0, 1] / np.linalg.norm(g), -1, 1))))
        # sanity (replaces the old sign heuristic): the lowest horizontal triangles are floor and their vertex normals point up
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); a = np.linalg.norm(n, axis=1); nzg = n[:, 2] / np.maximum(a, 1e-12)
        zc = V[F, 2].mean(1); horiz = np.abs(nzg) > 0.9; low = horiz & (zc < np.percentile(zc[horiz], 30))
        assert np.sum(a[low] * nzg[low]) > 0, f'{scene}: floor winding normals point down after gravity alignment'
        print(f'{scene}: gravity tilt {tilt:.2f} deg', flush=True)
        floors = floors_of(V, F)
        sc = make_scene(V, F)
        shape = (int(np.ceil(V[:, 1].max() / CELL)), int(np.ceil(V[:, 0].max() / CELL)))
        for k, zf in enumerate(floors):
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'v410:{scene}:{k}'.encode()).digest()[:8], 'little'))
            obstacle = band_obstacles(V, F, zf, shape, rng); interior = interior_of(sc, zf, shape, F, VN)
            free = interior & ~obstacle
            if free.sum() * CELL ** 2 < 4.0:                                            # a "floor" with < 4 m^2 free space is not a storey (e.g. a mezzanine artefact)
                print(f'{scene} floor {k} at {zf:.2f} m skipped: free {free.sum() * CELL ** 2:.1f} m^2', flush=True); continue
            fn = f'{scene}_f{k}.npz'
            wa = wall_angle_deg(V, F, zf)
            np.savez_compressed(out / fn, obstacle=obstacle, interior=interior, cell=CELL, floor_z=zf, band=np.array(BAND), extent_m=V.max(0), wall_angle_deg=wa)
            rows.append(dict(file=fn, scene=scene, floor=k, floor_z=zf, shape=list(shape), free_m2=float(free.sum() * CELL ** 2), wall_angle_deg=wa,
                             gravity_dir=[float(x) for x in g], gravity_tilt_deg=tilt, gravity_rotation=R.round(9).tolist(),
                             obstacle_m2=float((obstacle & interior).sum() * CELL ** 2), sha256=hashlib.sha256(open(out / fn, 'rb').read()).hexdigest()))
            print(f'{scene} floor {k} z {zf:.2f}: raster {shape}, free {rows[-1]["free_m2"]:.1f} m^2, obstacle {rows[-1]["obstacle_m2"]:.1f} m^2, wall angle {wa:.1f} deg, {time.time()-t0:.0f}s', flush=True)
    json.dump(dict(version=410, revision='review-M1M2 gravity + upward floor + wall angle', dataset=args.dataset, root=args.root, cell_m=CELL, band_m=list(BAND), points_per_cm2=POINTS_PER_CM2, rows=rows), open(out / 'index.json', 'w'), indent=1)
    print('done', len(rows), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
