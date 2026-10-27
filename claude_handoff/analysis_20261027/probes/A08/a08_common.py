"""A08 (moonshot) probe helpers, CPU only (Claude, 2026-10-27).
- observe(): verbatim copy of mapping_3d_storey_data_v420.observe / fibonacci_sphere (first-hit ray marching), so the probe can
  re-cast maps from decoded geometry exactly as the world generator does.
- ctx_bin(): exact adaptive KT code length of a binary 3D mask with the same 9-voxel causal context as rate_baselines_lib.ctx_bytes,
  optionally with side-information context bits from a mask both ends already hold (legal common knowledge, not Wyner-Ziv).
- plan_split(): oracle split of an observed-occupied set into slab / wall columns / lintel columns / furniture (GT labels; a learned
  encoder would have to do this itself -- walls span floor to ceiling, so it is easy, but it is an oracle here).
- box_cover(): learning-free 'consistency-set' primitive fitter: greedy maximal axis-aligned boxes that cover every observed-occupied
  furniture voxel and never enter observed-free space (O subset D subset not-F)."""
import math, os, sys
import numpy as np
import torch
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from rate_baselines_lib import load, targets, metrics, ctx_bytes, NB, LG, DIMS, dilate6  # noqa: E402,F401

SENSOR_RANGE = 5.0
BEAMS = 65536


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


RAYS = fibonacci_sphere(BEAMS)


def observe(occ, cell, poses, rays=RAYS):
    """verbatim v420 first-hit ray marching; occ uint8/bool [NX,NY,NZ]; poses [P,3] metres -> ternary int8."""
    occ = occ.astype(np.uint8)
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


def _ctx_index(s, nb, base):
    X, Y, Z = s.shape; c = torch.zeros_like(s)
    for dx, dy, dz in nb:
        sh = torch.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return c


def ctx_bin(mask, side=None, nb=NB):
    """exact adaptive KT code length in BYTES (+2 B) of a binary 3D mask, causal context of 9 earlier voxels; if `side` (bool, same
    shape, known to the decoder) is given, the context also gets side[v] and min(2, #6-neighbours of v in side)."""
    s = torch.from_numpy(mask.astype(np.int64))
    c = _ctx_index(s, nb, 2)
    if side is not None:
        sd = torch.from_numpy(side.astype(np.int64))
        n6 = torch.from_numpy(ndimage.convolve(side.astype(np.int64), _K6, mode='constant')).clamp(max=2)
        c = (c * 2 + sd) * 3 + n6
    K = int(c.max()) + 1
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 * K).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    bits = ((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2)
    return float(bits) / 8 + 2


_K6 = np.zeros((3, 3, 3), np.int64); _K6[1, 1, 0] = _K6[1, 1, 2] = _K6[1, 0, 1] = _K6[1, 2, 1] = _K6[0, 1, 1] = _K6[2, 1, 1] = 1


def ctx_tern2d(plan):
    """ternary 2-D image (-1 / 0 / 1) coded with ctx_bytes on a Z = 1 grid (z-neighbours read as 'unknown')."""
    return ctx_bytes(plan[:, :, None].astype(np.int8))


def gt_layers(gt):
    """GT label layers from the demo planes: slab, furniture, wall columns, lintel columns, lintel start index."""
    occ, slab, obj = gt['occ'], gt['slab'], gt['obj']
    struct = occ & ~obj & ~slab
    W = struct[:, :, 5].copy()                                         # a wall column is solid at every z (v420 extrusion)
    Lcol = struct.any(2) & ~W                                          # lintel columns: solid only from the door height up
    kz = np.where(struct[Lcol]) if Lcol.any() else None
    k_d = int(np.argmax(struct[Lcol].any(0))) if Lcol.any() else 32
    return dict(struct=struct, W=W, Lcol=Lcol, k_d=k_d)


def plan_split(O, gt, lay):
    """oracle split of observed-occupied O into classes; returns the observed wall plan (-1 none / 0 wall / 1 lintel) and O_furn."""
    O_struct = O & lay['struct']
    colobs = O_struct.any(2)
    plan = np.full(O.shape[:2], -1, np.int8)
    plan[colobs & lay['W']] = 0
    plan[colobs & lay['Lcol']] = 1
    O_furn = O & gt['obj']
    return plan, O_furn


def render_structure(plan, gt, lay):
    """decoded structure from the observed plan + public heights: slabs everywhere, wall columns full height, lintels from k_d up."""
    D = gt['slab'].copy()
    D |= (plan == 0)[:, :, None]
    lint = np.zeros(32, bool); lint[lay['k_d']:] = True
    D |= (plan == 1)[:, :, None] & lint[None, None, :]
    return D


def box_cover(O, A, conn=26):
    """greedy maximal boxes covering every voxel of O, each box inside A (allowed = not observed-free) and inside the bounding
    box of its 26-connected component of O.  Returns a list of (x0, x1, y0, y1, z0, z1), half-open."""
    st = np.ones((3, 3, 3), bool) if conn == 26 else None
    lab, n = ndimage.label(O, structure=st)
    boxes = []
    for ci, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        off = np.array([s.start for s in sl])
        todo = (lab[sl] == ci + 1).copy()
        Asub = A[sl]
        shp = np.array(todo.shape)
        while todo.any():
            seed = np.argwhere(todo)[0]
            lo = seed.copy(); hi = seed + 1
            grown = True
            while grown:
                grown = False
                for ax in range(3):
                    for side in (0, 1):
                        if side == 0 and lo[ax] == 0: continue
                        if side == 1 and hi[ax] == shp[ax]: continue
                        a, b = lo.copy(), hi.copy()
                        if side == 0: a[ax] -= 1; b[ax] = a[ax] + 1
                        else: a[ax] = hi[ax]; b[ax] = hi[ax] + 1
                        if Asub[a[0]:b[0], a[1]:b[1], a[2]:b[2]].all():
                            if side == 0: lo[ax] -= 1
                            else: hi[ax] += 1
                            grown = True
            todo[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
            g0 = lo + off; g1 = hi + off
            boxes.append((int(g0[0]), int(g1[0]), int(g0[1]), int(g1[1]), int(g0[2]), int(g1[2])))
    return boxes


def render_boxes(boxes, shape=DIMS):
    D = np.zeros(shape, bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        D[x0:x1, y0:y1, z0:z1] = True
    return D


def box_bits(boxes):
    """two prices per box list: raw fixed-length (7+7+7+7+5+5 = 38 bit) and an adaptive code (start corner 7+7+5 bit raw, sizes
    dx, dy, dz with adaptive KT models shared within the list) -- the adaptive price is what a simple entropy coder would pay."""
    n = len(boxes)
    if n == 0:
        return 0., 0.
    raw = 38. * n
    def kt_bits(vals, alphabet):
        cnt = np.zeros(alphabet); bits = 0.
        for v in vals:
            p = (cnt[v] + .5) / (cnt.sum() + .5 * alphabet); bits -= math.log2(p); cnt[v] += 1
        return bits
    dx = [b[1] - b[0] - 1 for b in boxes]; dy = [b[3] - b[2] - 1 for b in boxes]; dz = [b[5] - b[4] - 1 for b in boxes]
    ad = 19. * n + kt_bits(dx, 128) + kt_bits(dy, 128) + kt_bits(dz, 32)
    return raw, ad
