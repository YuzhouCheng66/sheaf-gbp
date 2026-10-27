"""p13_free_space.py (A03, debate round): is free space scored at all? (6 demo VAL houses x 8 agents, K0 and R8, numpy only, 1 process)
Three decoders scored with the training metric copy (surface / variable / furniture F1, unknown->known FPR) AND two metrics the
project does not report: free-space F1 (predicted free vs target free, inside the target-known region) and 3-way accuracy there.
  ours    = the exported 3S-X prediction (calibrated rule);
  S-only  = predict exactly the target's occupied voxels, everything else UNKNOWN (needs no poses, no visibility at all);
  exact   = any exact render-decoded codec (RAY-SHEAF / ELS / ProgSheaf ...) = the target itself.
Bytes of S-only: ctx of the binary occupied mask (rate_split's binary coder, numpy copy)."""
import base64, gzip, json, os
import numpy as np
from a03np import load_fx, score_np, fpr, targets, ctx_bytes_np, HERE

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
DIMS = (128, 128, 32)


def load_demo(name):
    m = json.load(open(os.path.join(ROOT, 'demo_web3d', 'data', f'house_{name}.json'))); raw = gzip.decompress(base64.b64decode(m['bin_gz_b64']))

    def plane(k):
        o, n = m['offsets'][k]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:int(np.prod(DIMS))].astype(bool).reshape(DIMS)
    n = m['agents']
    own = np.stack([np.where(plane(f'own_known_{a}'), plane(f'own_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)])
    pred = {k: np.stack([np.where(plane(f'pred_{k}_known_{a}'), plane(f'pred_{k}_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(n)]) for k in (0, 8)}
    return m, own, pred


def f1(a, b):
    tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def extra(p, t):
    k = t >= 0
    return dict(free_f1=f1((p == 0) & k, t == 0), acc3=float((p[k] == t[k]).mean()), fpr=fpr(p, t))


rows = []
for name in ('r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1'):
    m, own, pred = load_demo(name)
    fx, meta = load_fx(name); assert (fx['private'] == own).all()
    slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = fx['occ'].astype(bool) & ~obj & ~slab
    for k in (0, 8):
        T = targets(own, np.asarray(m['reached'][m['hops'].index(k)], bool))
        for a in range(own.shape[0]):
            t = T[a]; S = np.where(t == 1, np.int8(1), np.int8(-1))
            for tag, p in (('ours', pred[k][a]), ('S-only', S), ('exact', t)):
                rows.append(dict(house=name, hop=k, agent=a, decoder=tag, **score_np(p, t, slab, obj, struct), **extra(p, t)))
            if k == 0:
                rows.append(dict(house=name, hop=k, agent=a, decoder='S-only-bytes', s_bytes=ctx_bytes_np(t == 1, 2, 0)))
        print(name, 'done', flush=True)
S = {}
for k in (0, 8):
    for tag in ('ours', 'S-only', 'exact'):
        rr = [r for r in rows if r['hop'] == k and r['decoder'] == tag]
        S[f'K{k}|{tag}'] = {key: float(np.mean([r[key] for r in rr if r[key] is not None])) for key in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_f1', 'acc3')}
S['S-only bytes (K0, median)'] = float(np.median([r['s_bytes'] for r in rows if r['decoder'] == 'S-only-bytes']))
for key, v in S.items():
    print(key, v if not isinstance(v, dict) else ' '.join(f'{a} {b:.4f}' for a, b in v.items()))
json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p13_free_space.json'), 'w'), indent=1)
