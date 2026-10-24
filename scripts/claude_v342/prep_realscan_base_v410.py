"""Base floor rasters for the real-scan 2D world family v410 (Claude, 2026-10-24). Run once per dataset on uk-linux (CPU).

For every scene mesh and every detected floor, writes a 2 cm raster in the scene frame (z-up, translated so the mesh bbox minimum is 0):
  obstacle[iy, ix]  some mesh surface lies in the height band [floor + 0.15 m, floor + 1.60 m] above that 2 cm column
                    (area-weighted surface sampling at 4 points / cm^2 of the triangles crossing the band, clipped to the band),
  interior[iy, ix]  a ray straight down from floor + 0.8 m hits a surface within 1.2 m, i.e. a floor (or furniture) no more than 0.4 m
                    below this floor: only the scanned inside of the building has a floor; the void seen through windows, scan holes and
                    stairwells / double-height voids do not. (No ceiling test: the Replica FRL apartments were scanned without ceilings,
                    upward rays hit something over only ~20 % of their floor.)
free = interior & ~obstacle is the floor's navigable / visible free space; everything else is "wall" in the v371 contract.
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


def read_replica_ply(path):
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
    return np.stack([vert['x'], vert['y'], vert['z']], 1).astype(np.float64), F


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


def interior_of(scene, zf, shape):
    ny, nx = shape
    X, Y = np.meshgrid((np.arange(nx) + .5) * CELL, (np.arange(ny) + .5) * CELL)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf + 0.8)], 1).astype(np.float32)
    down = np.concatenate([P, np.tile(np.float32([0, 0, -1]), (len(P), 1))], 1)
    td = scene.cast_rays(o3d.core.Tensor(down))['t_hit'].numpy().reshape(ny, nx)
    return td < 1.2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True, choices=['replica']); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    scenes = sorted(p.name for p in Path(args.root).iterdir() if (p / 'mesh.ply').exists())
    rows = []; t0 = time.time()
    for scene in scenes:
        V, F = read_replica_ply(str(Path(args.root) / scene / 'mesh.ply'))            # Replica: vertical axis is z, but its sign varies
        V = V - V.min(0)
        # vertical sign (as in fig_replica_preview.load): the lowest horizontal triangles are floor, which the scan faces from above
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
        zc = V[F, 2].mean(1); horiz = np.abs(nz) > 0.9
        low = horiz & (zc < np.percentile(zc[horiz], 30))
        a = np.linalg.norm(n, axis=1)
        if np.sum(a[low] * np.sign(nz[low])) < 0:                                      # floors face down => the mesh is upside down
            V[:, 2] = V[:, 2].max() - V[:, 2]; V[:, 1] = V[:, 1].max() - V[:, 1]      # rotate 180 deg about x (keeps handedness)
            F = F.copy()
        print(f'{scene}: vertical sign {"flipped" if np.sum(a[low] * np.sign(nz[low])) < 0 else "kept"}', flush=True)
        floors = floors_of(V, F)
        sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        shape = (int(np.ceil(V[:, 1].max() / CELL)), int(np.ceil(V[:, 0].max() / CELL)))
        for k, zf in enumerate(floors):
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'v410:{scene}:{k}'.encode()).digest()[:8], 'little'))
            obstacle = band_obstacles(V, F, zf, shape, rng); interior = interior_of(sc, zf, shape)
            free = interior & ~obstacle
            if free.sum() * CELL ** 2 < 4.0:                                            # a "floor" with < 4 m^2 free space is not a storey (e.g. a mezzanine artefact)
                print(f'{scene} floor {k} at {zf:.2f} m skipped: free {free.sum() * CELL ** 2:.1f} m^2', flush=True); continue
            fn = f'{scene}_f{k}.npz'
            np.savez_compressed(out / fn, obstacle=obstacle, interior=interior, cell=CELL, floor_z=zf, band=np.array(BAND), extent_m=V.max(0))
            rows.append(dict(file=fn, scene=scene, floor=k, floor_z=zf, shape=list(shape), free_m2=float(free.sum() * CELL ** 2),
                             obstacle_m2=float((obstacle & interior).sum() * CELL ** 2), sha256=hashlib.sha256(open(out / fn, 'rb').read()).hexdigest()))
            print(f'{scene} floor {k} z {zf:.2f}: raster {shape}, free {rows[-1]["free_m2"]:.1f} m^2, obstacle {rows[-1]["obstacle_m2"]:.1f} m^2, {time.time()-t0:.0f}s', flush=True)
    json.dump(dict(version=410, dataset=args.dataset, root=args.root, cell_m=CELL, band_m=list(BAND), points_per_cm2=POINTS_PER_CM2, rows=rows), open(out / 'index.json', 'w'), indent=1)
    print('done', len(rows), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
