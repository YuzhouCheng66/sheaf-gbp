"""Preview of real-scan multi-agent mapping on Replica (Claude, 2026-10-24). Replica is already on uk-linux; HM3D would go through the
same code. No learning: this only shows (1) what the real scans look like, (2) their 2D floor-slice occupancy and 3D surface-voxel
occupancy, (3) what each agent's own map looks like when agents random-walk with ground-truth poses and a 2D lidar / 3D depth sensor
ray-cast against the mesh (the standard occupancy update with exact poses and noise-free ranges).
python fig_replica_preview.py --replica /mnt/datasets/replica_v1 --scenes apartment_0 frl_apartment_0 office_3 hotel_0 --out DIR
"""
import argparse, math, time
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ap = argparse.ArgumentParser()
ap.add_argument('--replica', required=True); ap.add_argument('--scenes', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--up', default='z', choices=['auto', 'x', 'y', 'z'], help='up axis of the meshes (Replica is z-up)'); ap.add_argument('--preview-scene', default='apartment_0'); ap.add_argument('--agents', type=int, default=4); ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'figure.dpi': 200, 'savefig.dpi': 200})
AGENT_COL = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']


def read_replica_ply(path):
    """Replica mesh.ply: binary little-endian, vertices (x,y,z,nx,ny,nz float32; r,g,b uint8), faces as uint8 count + int32 indices (quads).
    Open3D's reader fails on some of these quad meshes, so parse directly and split every quad into two triangles."""
    raw = open(path, 'rb').read(); end = raw.index(b'end_header\n') + len(b'end_header\n'); header = raw[:end].decode('ascii').splitlines()
    nv = int(next(l for l in header if l.startswith('element vertex')).split()[-1]); nf = int(next(l for l in header if l.startswith('element face')).split()[-1])
    props = [l.split()[1:] for l in header[header.index(next(l for l in header if l.startswith('element vertex'))) + 1:] if l.startswith('property') and 'list' not in l]
    props = props[:[i for i, l in enumerate(header) if l.startswith('element face')][0] - [i for i, l in enumerate(header) if l.startswith('element vertex')][0] - 1]
    tmap = {'float': '<f4', 'uchar': 'u1', 'int': '<i4', 'double': '<f8'}
    vdt = np.dtype([(name, tmap[t]) for t, name in props])
    vert = np.frombuffer(raw, vdt, nv, end)
    off = end + nv * vdt.itemsize
    cnt = raw[off]
    fdt = np.dtype([('n', 'u1'), ('idx', '<i4', (cnt,))])
    faces = np.frombuffer(raw, fdt, nf, off)
    assert (faces['n'] == cnt).all(), 'mixed polygon sizes'
    idx = faces['idx'].astype(np.int64)
    F = idx if cnt == 3 else np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
    V = np.stack([vert['x'], vert['y'], vert['z']], 1).astype(np.float64)
    C = np.stack([vert['red'], vert['green'], vert['blue']], 1).astype(np.float64) / 255. if 'red' in vdt.names else np.full((nv, 3), .7)
    return V, F, C


def load(scene):
    V, F, C = read_replica_ply(str(Path(args.replica) / scene / 'mesh.ply'))
    # up axis: the axis along which the largest area of triangles faces (floors + ceilings); walls split over the two other axes
    e1 = V[F[:, 1]] - V[F[:, 0]]; e2 = V[F[:, 2]] - V[F[:, 0]]; n = np.cross(e1, e2); area = np.linalg.norm(n, axis=1) / 2; nu = n / np.maximum(2 * area[:, None], 1e-12)
    share = [(area * (np.abs(nu[:, k]) > 0.9)).sum() / area.sum() for k in range(3)]
    up = int(np.argmax(share)) if args.up == 'auto' else 'xyz'.index(args.up)
    # sign: the floor faces up; floor triangles are the lower ones among horizontal ones
    horiz = np.abs(nu[:, up]) > 0.9; zc = V[F, up].mean(1)
    sgn = 1.0 if np.median(nu[horiz & (zc < np.percentile(zc[horiz], 30)), up]) > 0 else -1.0
    perm = [k for k in range(3) if k != up] + [up]
    V = V[:, perm]; V[:, 2] *= sgn
    if sgn < 0:
        V[:, 1] *= -1                                                    # keep a right-handed frame
    V -= V.min(0)
    # floor heights: area-weighted histogram of upward-facing triangle heights, peaks at least 2 m apart
    nz = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 2] / np.maximum(2 * area, 1e-12)
    z = V[F, 2].mean(1); w = area * (nz > 0.9)
    hist, edges = np.histogram(z, bins=np.arange(0, V[:, 2].max() + 0.1, 0.05), weights=w)
    floors = []
    for i in np.argsort(-hist):
        if hist[i] < 0.15 * hist.max():
            break
        zf = (edges[i] + edges[i + 1]) / 2
        if all(abs(zf - f) > 2.0 for f in floors):
            floors.append(zf)
    floors = sorted(floors)
    return dict(V=V, F=F, C=C, floors=floors, share=share)


def scene_of(V, F):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return s


def render(mesh, zcut, w=900, h=640):
    """dollhouse colour rendering: geometry above zcut removed, camera above at an oblique angle, vertex colours interpolated."""
    V, F, C = mesh['V'], mesh['F'], mesh['C']
    keep = V[F, 2].max(1) < zcut
    Fk = F[keep]; sc = scene_of(V, Fk)
    ctr = V.max(0) / 2; ext = V.max(0)[:2].max()
    eye = ctr + np.array([0.55 * ext, -0.95 * ext, 1.25 * ext]); eye[2] = zcut + 1.1 * ext
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=42, center=list(ctr - [0, 0, ctr[2] - 0.5]), eye=list(eye), up=[0, 0, 1], width_px=w, height_px=h)
    r = sc.cast_rays(rays)
    pid = r['primitive_ids'].numpy(); uv = r['primitive_uvs'].numpy(); hit = r['t_hit'].numpy() < 1e9; nrm = r['primitive_normals'].numpy()
    img = np.ones((h, w, 3))
    tri = Fk[pid[hit]]; u = uv[hit][:, 0:1]; v = uv[hit][:, 1:2]
    col = (1 - u - v) * C[tri[:, 0]] + u * C[tri[:, 1]] + v * C[tri[:, 2]]
    shade = 0.75 + 0.25 * np.abs(nrm[hit] @ (np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])))
    img[hit] = np.clip(col * shade[:, None], 0, 1)
    return img


def slice2d(mesh, zf, cell=0.05):
    """2D occupancy of one floor: occupied = mesh surface within 0.15-1.6 m above the floor; free = interior (floor below, ceiling above)."""
    V, F = mesh['V'], mesh['F']; sc = scene_of(V, F)
    nx, ny = int(math.ceil(V[:, 0].max() / cell)), int(math.ceil(V[:, 1].max() / cell))
    xs = (np.arange(nx) + .5) * cell; ys = (np.arange(ny) + .5) * cell
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    occ = np.zeros((nx, ny), bool)
    for dz in np.arange(0.15, 1.61, 0.05):
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf + dz)], 1).astype(np.float32)
        d = sc.compute_distance(o3d.core.Tensor(P)).numpy().reshape(nx, ny)
        occ |= d < cell * 0.75
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, zf + 0.8)], 1).astype(np.float32)
    down = np.concatenate([P, np.tile([0, 0, -1], (len(P), 1))], 1).astype(np.float32); upr = np.concatenate([P, np.tile([0, 0, 1], (len(P), 1))], 1).astype(np.float32)
    td = sc.cast_rays(o3d.core.Tensor(down))['t_hit'].numpy().reshape(nx, ny); tu = sc.cast_rays(o3d.core.Tensor(upr))['t_hit'].numpy().reshape(nx, ny)
    interior = (td < 1.2) & (tu < 3.5)
    grid = np.where(occ, 1, np.where(interior, 0, -1)).astype(np.int8)
    return grid, cell


def voxels3d(mesh, zcut, cell=0.1):
    """surface voxels (centre within ~half a cell of the mesh) below zcut."""
    V, F = mesh['V'], mesh['F']; sc = scene_of(V, F)
    n = np.ceil(np.array([V[:, 0].max(), V[:, 1].max(), min(V[:, 2].max(), zcut)]) / cell).astype(int)
    g = [(np.arange(k) + .5) * cell for k in n]
    X, Y, Z = np.meshgrid(*g, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    d = sc.compute_distance(o3d.core.Tensor(P)).numpy()
    return P[d < cell * 0.6], cell


def show_map(ax, grid, title=None):
    ax.imshow(grid.T, cmap=ListedColormap(['#e9e9e9', '#ffffff', '#1a1a1a']), vmin=-1, vmax=1, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('#999999')
    if title:
        ax.set_title(title, pad=3)


def scatter3d(ax, P, c, s=0.6, alpha=1.0):
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, s=s, alpha=alpha, linewidths=0, depthshade=False)


def frame3d(ax, ext, zmax):
    ax.set_xlim(0, ext[0]); ax.set_ylim(0, ext[1]); ax.set_zlim(0, zmax)
    ax.set_box_aspect((ext[0], ext[1], zmax)); ax.view_init(elev=38, azim=-60)
    ax.set_axis_off()


# ------------------------------------------------------------------ figure 1: four real scans
t0 = time.time(); meshes = {}
fig = plt.figure(figsize=(11, 2.9 * len(args.scenes)))
for r, scene in enumerate(args.scenes):
    m = load(scene); meshes[scene] = m
    zf = m['floors'][0]; zcut = zf + 1.9
    img = render(m, zcut)
    grid, cell = slice2d(m, zf)
    P, vcell = voxels3d(m, zcut)
    ext = m['V'].max(0)
    print(f'{scene}: extent {ext.round(2)} m, up-axis shares {np.round(m["share"], 2)}, floors {np.round(m["floors"], 2)}, faces {len(m["F"]):,}, '
          f'2D cells {grid.size:,} (occupied {int((grid == 1).sum()):,}), 3D surface voxels {len(P):,}, {time.time()-t0:.0f}s', flush=True)
    ax = fig.add_subplot(len(args.scenes), 3, 3 * r + 1); ax.imshow(img); ax.set_axis_off()
    ax.set_title(f'{scene.replace("_", " ")}: real scan (ceiling removed)', pad=3)
    ax = fig.add_subplot(len(args.scenes), 3, 3 * r + 2); show_map(ax, grid, f'2D occupancy, floor slice 0.15-1.6 m, {cell * 100:.0f} cm cells')
    ax.text(0.01, 0.01, f'{ext[0]:.1f} x {ext[1]:.1f} m', transform=ax.transAxes, fontsize=6.5, color='#555555')
    ax = fig.add_subplot(len(args.scenes), 3, 3 * r + 3, projection='3d')
    scatter3d(ax, P, plt.cm.viridis((P[:, 2] - zf) / max(zcut - zf, 1e-6)), s=0.5)
    frame3d(ax, ext, zcut); ax.set_title(f'3D occupancy, {vcell * 100:.0f} cm surface voxels ({len(P):,})', pad=0)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.04, hspace=0.12)
fig.savefig(out / 'replica_real_scans.png'); plt.close(fig)

# ------------------------------------------------------------------ figure 2: multi-agent preview on one scan
m = meshes.get(args.preview_scene) or load(args.preview_scene)
zf = m['floors'][0]; grid, cell = slice2d(m, zf); sc = scene_of(m['V'], m['F'])
rng = np.random.default_rng(args.seed)
free = grid == 0
try:
    from scipy.ndimage import distance_transform_edt
    clear = distance_transform_edt(grid != 1) * cell
except ImportError:
    clear = free.astype(float)
ok = free & (clear > 0.35)
cand = np.argwhere(ok)
walks = []
for a in range(args.agents):
    for _ in range(1000):                                                # start points spread over the free space
        c = cand[rng.integers(len(cand))]
        if all(np.linalg.norm(c - w[0] / cell) > 2.5 / cell for w in walks):
            break
    p = (c + .5) * cell; th = rng.uniform(0, 2 * np.pi); path = [p.copy()]
    for t in range(80):                                                  # local random walk, 0.2 m steps, smooth heading
        for _ in range(20):
            th2 = th + rng.normal(0, 0.6); q = p + 0.2 * np.array([np.cos(th2), np.sin(th2)])
            i, j = (q / cell).astype(int)
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and ok[i, j]:
                p, th = q, th2; break
            th = th + np.pi / 2
        path.append(p.copy())
    walks.append(np.array(path))
# 2D lidar on the floor slice: 360 beams, 5 m, ray marching on the 2D grid (same rule as observe())
ang = np.linspace(0, 2 * np.pi, 360, endpoint=False); dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
own2d = []
for wlk in walks:
    g = np.full(grid.shape, -1, np.int8)
    for p in wlk[::2]:
        alive = np.ones(len(dirs), bool)
        for s in range(1, int(5.0 / (cell / 2)) + 1):
            q = p[None] + dirs * s * cell / 2; idx = (q / cell).astype(int)
            inside = (idx[:, 0] >= 0) & (idx[:, 0] < grid.shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < grid.shape[1]) & alive
            ii = idx[inside]; v = grid[ii[:, 0], ii[:, 1]]
            hit = v == 1; outside = v == -1
            fr = ii[~hit & ~outside]; g[fr[:, 0], fr[:, 1]] = np.maximum(g[fr[:, 0], fr[:, 1]], 0)
            hh = ii[hit]; g[hh[:, 0], hh[:, 1]] = 1
            al = np.nonzero(inside)[0]; alive[al[hit | outside]] = False; alive &= inside | ~alive
    own2d.append(g)
union2d = np.max(np.stack(own2d), 0)
# 3D depth: spherical rays (4,096 Fibonacci directions, 5 m) from the sensor 1.2 m above the floor every 8th step, cast against the mesh
i = np.arange(4096) + .5; phi = np.arccos(1 - 2 * i / 4096); tht = np.pi * (1 + 5 ** .5) * i
sph = np.stack((np.cos(tht) * np.sin(phi), np.sin(tht) * np.sin(phi), np.cos(phi)), 1).astype(np.float32)
own3d = []
for wlk in walks:
    pts = []
    for p in wlk[::8]:
        o = np.array([p[0], p[1], zf + 1.2], np.float32)
        rays = np.concatenate([np.tile(o, (len(sph), 1)), sph], 1)
        t = sc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        h = t < 5.0; pts.append(o[None] + sph[h] * t[h][:, None])
    P = np.concatenate(pts); P = np.unique(np.floor(P / 0.1).astype(int), axis=0) * 0.1 + 0.05
    own3d.append(P[P[:, 2] < zf + 1.9])
ext = m['V'].max(0)
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, args.agents + 1, wspace=0.04, hspace=0.12, left=0.01, right=0.99, top=0.93, bottom=0.02)
ax = fig.add_subplot(gs[0, 0]); show_map(ax, grid, 'Ground truth (floor slice)')
for a, wlk in enumerate(walks):
    ax.plot(wlk[:, 0] / cell - .5, wlk[:, 1] / cell - .5, '-', color=AGENT_COL[a], lw=0.9); ax.plot(wlk[0, 0] / cell - .5, wlk[0, 1] / cell - .5, 'o', color=AGENT_COL[a], ms=2.5)
for a in range(args.agents):
    ax = fig.add_subplot(gs[0, a + 1]); show_map(ax, own2d[a], f'Agent {a + 1}: own 2D lidar map')
    ax.plot(walks[a][:, 0] / cell - .5, walks[a][:, 1] / cell - .5, '-', color=AGENT_COL[a], lw=0.7)
    ax.text(0.01, 0.01, f'known {100 * (own2d[a] >= 0).mean():.1f} %', transform=ax.transAxes, fontsize=6.5, color='#555555')
ax = fig.add_subplot(gs[1, 0]); show_map(ax, union2d, f'Union of the {args.agents} own maps (2D)')
ax.text(0.01, 0.01, f'known {100 * (union2d >= 0).mean():.1f} %', transform=ax.transAxes, fontsize=6.5, color='#555555')
ax = fig.add_subplot(gs[1, 1:3], projection='3d')
for a in np.argsort([-len(o) for o in own3d]):                     # largest first, so smaller maps stay visible on top
    scatter3d(ax, own3d[a], AGENT_COL[a], s=0.5, alpha=0.9)
frame3d(ax, ext, zf + 1.9); ax.set_title('3D: occupied 10 cm voxels seen by each agent (colour = agent)', pad=0)
Pg, _ = voxels3d(m, zf + 1.9)
ax = fig.add_subplot(gs[1, 3:], projection='3d')
scatter3d(ax, Pg, plt.cm.viridis((Pg[:, 2] - zf) / 1.9), s=0.5)
frame3d(ax, ext, zf + 1.9); ax.set_title(f'3D ground truth below 1.9 m: {len(Pg):,} surface voxels\n(the 4 agents together saw {len(np.unique(np.concatenate(own3d), axis=0)):,})', pad=0)
fig.suptitle(f'Multi-agent mapping input on a real scan ({args.preview_scene.replace("_", " ")}, Replica): {args.agents} agents, local random walks, ground-truth poses, 5 m range. No fusion or learning yet.', fontsize=8.5, y=0.985)
fig.savefig(out / f'replica_multiagent_preview_{args.preview_scene}.png'); plt.close(fig)
print('done', f'{time.time()-t0:.0f}s')
