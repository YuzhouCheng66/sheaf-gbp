"""a09lib (Claude, 2026-10-27): numpy-only copies of rate_baselines_lib (load, targets, ctx coders, metrics) so probes run without importing
torch (the machine's commit charge is near its limit while other agents' probes run). Every metric is validated against the demo export
(our own K0 / R8 predictions must reproduce m['metrics'] to 1e-3), exactly as rate_baselines.py does."""
import base64, gzip, json, math
import numpy as np
from scipy.special import gammaln

DIMS = (128, 128, 32)
NB = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]


def load(path):
    m = json.load(open(path)); raw = gzip.decompress(base64.b64decode(m['bin_gz_b64']))

    def plane(name):
        o, n = m['offsets'][name]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:int(np.prod(DIMS))].astype(bool).reshape(DIMS)
    n = m['agents']
    own = np.stack([np.where(plane(f'own_known_{a}'), plane(f'own_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)])
    pred = {k: np.stack([np.where(plane(f'pred_{k}_known_{a}'), plane(f'pred_{k}_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)]) for k in m['hops']}
    gt = {g: plane(f'gt_{g}') for g in ('occ', 'slab', 'obj')}
    return m, own, pred, gt


def targets(own, reached):
    r = np.asarray(reached, bool)
    return np.where(r[:, :, None, None, None], own[None], np.int8(-1)).max(1)


def _ctx_index(s, base):
    X, Y, Z = s.shape; c = np.zeros_like(s)
    for dx, dy, dz in NB:                                              # out-of-grid neighbours read as symbol 0
        sh = np.zeros_like(s); sh[dx:, dy:, dz:] = s[:X - dx, :Y - dy, :Z - dz]; c = c * base + sh
    return (c * base + s).reshape(-1)


def _kt_bits(idx, base, alpha):
    cnt = np.bincount(idx, minlength=base ** (len(NB) + 1)).reshape(-1, base).astype(np.float64)
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    return float(((gammaln(N + base * alpha) - gammaln(base * alpha)) - (gammaln(cnt + alpha) - gammaln(alpha)).sum(1)).sum() / math.log(2))


def ctx_bytes(m):
    """ternary map (-1/0/1), identical to rate_baselines_lib.ctx_bytes"""
    s = m.astype(np.int64) + 1
    return _kt_bits(_ctx_index(s, 3), 3, 0.5) / 8 + 2


def ctx_bytes_binary(mask):
    """binary mask, identical to rate_split.ctx_bytes_binary"""
    s = mask.astype(np.int64)
    return _kt_bits(_ctx_index(s, 2), 2, 0.5) / 8 + 2


def dilate6(m):
    """m: bool [..., X, Y, Z] -> 6-neighbourhood dilation (incl. itself), as train_3d _pool6"""
    o = m.copy()
    o[..., 1:, :, :] |= m[..., :-1, :, :]; o[..., :-1, :, :] |= m[..., 1:, :, :]
    o[..., :, 1:, :] |= m[..., :, :-1, :]; o[..., :, :-1, :] |= m[..., :, 1:, :]
    o[..., :, :, 1:] |= m[..., :, :, :-1]; o[..., :, :, :-1] |= m[..., :, :, 1:]
    return o


def _f1(a, b):
    tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def metrics(pred, target, slab, obj, struct):
    """pred, target int8 [n, X, Y, Z]; slab, obj, struct bool [X, Y, Z]. Same logic as train_3d_v431.metrics3d (surface / var / furn F1),
    plus unknown_known_fpr and the free-space recall (target-known-free voxels predicted free; UNSCORED by the training metrics)."""
    surf = (target == 1) & dilate6(target == 0)
    region = ~slab & ~dilate6(struct)
    rows = []
    for i in range(pred.shape[0]):
        p, t, s = pred[i], target[i], surf[i]; known = t >= 0
        po = (p == 1) & known; to = (t == 1) & known; ps = po & ~((t == 1) & ~s)
        tf = t == 0
        rows.append(dict(surface_f1=_f1(ps, s), var_occ_f1=_f1(po & ~slab, to & ~slab), furn_f1=_f1(po & region, to & region),
                         fpr=float((p[~known] >= 0).mean()) if (~known).any() else None,
                         free_recall=float((p[tf] == 0).mean()) if tf.any() else None))
    return rows


def check_export(m, own, pred, gt):
    """assert that metrics() reproduces the exported per-agent metrics of our own K0 / R8 predictions"""
    slab, obj = gt['slab'], gt['obj']; struct = gt['occ'] & ~gt['obj'] & ~gt['slab']
    T = {0: own, 8: targets(own, m['reached'][m['hops'].index(8)])}
    worst = 0.
    for k in (0, 8):
        rows = metrics(pred[k], T[k], slab, obj, struct)
        for a, r in enumerate(rows):
            ref = m['metrics'][str(k)][a]
            for key in ('surface_f1', 'var_occ_f1', 'furn_f1'):
                if ref[key] is not None and r[key] is not None:
                    worst = max(worst, abs(ref[key] - r[key]))
                    assert abs(ref[key] - r[key]) < 1e-3, (m['house'], k, a, key, ref[key], r[key])
    return worst
