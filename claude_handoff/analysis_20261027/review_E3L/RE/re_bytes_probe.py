"""re_bytes_probe.py (R-E design review, 2026-10-27): CPU-only byte geometry of the w -> inf special cases of the E3-L energy.
Only read-only imports of the frozen g4_e3l / g4_e3r / g4_noise_panels. No accuracy is computed (bytes only), on FRESH houses, noise seed 1.

For each house (cam_nf agent maps; agent frame = public frame) and evidence alphabet (Qinf / Q3 / VOTE of table A, E5 quantizer file):
  CTRL   : E3-R control wire: sum over directed edges of int_ctx_bytes(S_e, k_e > 0) + int_ctx_bytes(k_e, k_e > 0)   (= 'Sk')
  FR1    : full-resolution latent, D = 1, Lam = 1[known], z = e  ->  messages (h, Lam) = (S_e, k_e); O7 coder (mask once + values)
  S2D    : 4x4x2 cells, Lam_cell = 1[agent touched the block] (scalar per cell), z = space-to-depth of (e_v, 1[known_v]) (D = 64)
           -> messages (h = 64 channels, Lam = number of subtree agents touching the block); O7 coder on the 32x32x16 cell grid
  S2D_S  : as S2D but only the 32 S channels (no per-voxel count channels; the known head would be learned, as in E3-L)
Each with a breakdown (mask / Lam values / h values) and the per-agent SOURCE message (own potential) bytes.
Usage: CUDA_VISIBLE_DEVICES= python -B re_bytes_probe.py --files fresh_r1_i8_s1,... --out OUT.json
"""
import os, sys, json, time, ast, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_e3l as L  # noqa: E402  (frozen, read only)
R = L.R; G = L.G
EXPORT = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs', 'export')
QFILE = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e5_quant', 'criteria_quantizer.json')
B = (4, 4, 2)


def tables():
    q = json.load(open(QFILE))['quantizer']['A']
    return {k: np.asarray(ast.literal_eval(q[k])['table'] if isinstance(q[k], str) else q[k]['table'], np.int64) for k in ('Qinf', 'Q3', 'VOTE')}


def s2d(a):
    """[X, Y, Z] -> [32, X/4, Y/4, Z/2] (space-to-depth)"""
    X, Y, Z = a.shape
    a = a.reshape(X // 4, 4, Y // 4, 4, Z // 2, 2).transpose(1, 3, 5, 0, 2, 4)
    return a.reshape(32, X // 4, Y // 4, Z // 2)


def subtree_msgs(own, F):
    """w -> inf GaBP messages = sender-side subtree sums; own: list of tuples of arrays"""
    msgs = {}
    for s, r in F['order']:
        acc = [x.copy() for x in own[s]]
        for k in F['nb'][s]:
            if k != r:
                for t, x in zip(acc, msgs[(k, s)]):
                    t += x
        msgs[(s, r)] = acc
    return msgs


def o7_parts(hi, li):
    kn = li > 0
    m = L.mask_bytes(kn); lv = float(L.value_bytes_given_mask(li, kn)); hv = float(sum(L.value_bytes_given_mask(hi[d], kn) for d in range(hi.shape[0])))
    return m, lv, hv


def run(fn, tabs):
    z = np.load(os.path.join(EXPORT, fn + '.npz'))
    maps = z['L_k0_cam_nf']; n = maps.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    out = dict(file=fn, n=n, comps=[len(c) for c in F['comps']], directed=len(F['order']))
    for an, tb in tabs.items():
        t0 = time.time(); res = {}
        st = [R.stalk(maps[a], tb) for a in range(n)]                       # (e int32, known bool)
        # CTRL and FR1 (full res)
        own = [(e.astype(np.int64), k.astype(np.int64)) for e, k in st]
        M = subtree_msgs(own, F)
        ctrl = sum(G.int_ctx_bytes(S, k > 0) + G.int_ctx_bytes(k, k > 0) for S, k in M.values())
        fr = np.array([o7_parts(S[None], k) for S, k in M.values()]).sum(0)
        fr_src = np.array([o7_parts(e[None], k) for e, k in own]).sum(0) / n
        res['CTRL'] = float(ctrl); res['FR1'] = dict(total=float(fr.sum()), mask=fr[0], lam=fr[1], h=fr[2], src_per_agent=float(fr_src.sum()))
        if an in ('Qinf', 'Q3'):
            own2 = []
            for e, k in own:
                tc = (s2d(k).sum(0) > 0).astype(np.int64)
                own2.append((np.concatenate([s2d(e), s2d(k)], 0), tc))
            M2 = subtree_msgs(own2, F)
            p = np.array([o7_parts(h, l) for h, l in M2.values()]).sum(0)
            ps = np.array([o7_parts(h[:32], l) for h, l in M2.values()]).sum(0)
            src = np.array([o7_parts(h, l) for h, l in own2]).sum(0) / n
            res['S2D'] = dict(total=float(p.sum()), mask=p[0], lam=p[1], h=p[2], src_per_agent=float(src.sum()))
            res['S2D_S'] = dict(total=float(ps.sum()), mask=ps[0], lam=ps[1], h=ps[2])
        res['seconds'] = time.time() - t0
        out[an] = res
        print(fn, an, json.dumps({k: (round(v) if isinstance(v, float) else {kk: round(vv) for kk, vv in v.items()}) for k, v in res.items()}), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--files', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tabs = tables(); rows = []
    for fn in a.files.split(','):
        rows.append(run(fn, tabs))
        json.dump(rows, open(a.out, 'w'), indent=1, default=float)
