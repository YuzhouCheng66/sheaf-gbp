"""R-B probe after G4-E1 (CPU, TRAIN houses only). Imports the frozen g4_noise_panels.py (never modifies it).
Per house, noise seed 1, integrator V3 (governing), tiers cam (and its agent-frame maps = cam_nf):
  (1) own-map byte decomposition: known mask / occupancy-given-known / magnitudes (free vs occupied state), clean vs noisy,
      counterfactual "noisy known mask with clean occupancy";
  (2) sum-fusion precision sweep: per-agent k-level quantisation of L before the forest sum (vote = ternary + count, k = 3, 4, 8,
      full /8, full), R8 furniture / FPR / free recall vs clean truth, relay bytes (B4 coder), and a collect-dispatch variant whose
      downward messages carry only the ternary of the component sum (identical beliefs on a tree);
  plus the max_occ ternary join for reference.
Usage: python -B rb_e1_probe.py OUTDIR rooms:index [rooms:index ...]"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import json, time, collections  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_noise_panels as G  # noqa: E402

W = G.W
SEED = int(os.environ.get('RB_SEED', '1'))
I = 'V3'

QUANT = {   # name -> (right-closed upper edges in 0.05 log-odds units, reconstruction rule)
    'vote_k2': ([0], 'vote'),
    'k3_occ': ([0, 33], 'centroid'),
    'k3_free': ([-21, 0], 'centroid'),
    'k4': ([-21, 0, 33], 'centroid'),
    'k8': ([-39, -21, -9, 0, 17, 33, 69], 'centroid'),
    'full_div8': (None, 'div8'),
    'full': (None, 'identity'),
}


def quantise(maps, name):
    edges, rule = QUANT[name]
    ks = [~G.is_unk(L) for L in maps]
    if rule == 'identity':
        return [(np.where(k, L.astype(np.int32), 0), k) for L, k in zip(maps, ks)], None
    if rule == 'div8':
        out = []
        for L, k in zip(maps, ks):
            q = np.rint(L.astype(np.float64) / 8).astype(np.int32); q = np.where((L > 0) & (q <= 0), 1, q)
            out.append((np.where(k, q, 0), k))
        return out, None
    allv = np.concatenate([L[k].astype(np.int64) for L, k in zip(maps, ks)])
    nb = len(edges) + 1
    bi = np.digitize(allv, edges, right=True)
    if rule == 'vote':
        rec = np.array([-1, 1], np.int32)
    else:
        cen = np.array([allv[bi == i].mean() if (bi == i).any() else 0. for i in range(nb)])
        rec = np.rint(cen / 8).astype(np.int32)
        thr_pos = np.array([i >= edges.index(0) + 1 for i in range(nb)])        # bins above L = 0 are occupied
        rec = np.where(thr_pos & (rec <= 0), 1, rec); rec = np.where(~thr_pos & (rec >= 0), -1, rec)
    out = []
    for L, k in zip(maps, ks):
        b = np.digitize(L.astype(np.int64), edges, right=True)
        out.append((np.where(k, rec[b], 0).astype(np.int32), k))
    return out, rec.tolist()


def bits_split(v, known):
    """B4 coder parts in bytes: known-mask only, ternary, magnitudes by state (free / occupied)"""
    known = np.asarray(known, bool)
    t = np.where(known, (np.asarray(v) > 0).astype(np.int8), np.int8(-1))
    km = np.where(known, np.int8(0), np.int8(-1))
    e0, e1, p0, p1, s0, s1 = G.int_magnitude_events(v, known)
    nf = len(e0) // 2; NB = G.INT_NBC
    mag = {}
    for st, name in ((0, 'free'), (1, 'occ')):
        fb = G.kt_bin_bits(e0[st * nf:(st + 1) * nf], e1[st * nf:(st + 1) * nf])
        pb = G.kt_bin_bits(p0[st * NB:(st + 1) * NB], p1[st * NB:(st + 1) * NB])
        sb = G.kt_bin_bits(s0[st], s1[st])
        mag[name] = dict(flag=fb / 8, prefix=pb / 8, suffix=sb / 8)
    return dict(known=G.ctx_tern_bits(km) / 8, tern=G.ctx_tern_bits(t) / 8, mag=mag, total=G.int_ctx_bytes(v, known))


def center_root(F, cm):
    best = None
    for r in cm:
        d = {r: 0}; q = [r]
        while q:
            u = q.pop(0)
            for w in F['nb'][u]:
                if w not in d:
                    d[w] = d[u] + 1; q.append(w)
        ecc = max(d.values())
        if best is None or ecc < best[0]:
            best = (ecc, r, d)
    return best[1], best[2]


def comp_rows(preds, F, R8T):
    rows = []
    for ci, cm in enumerate(F['comps']):
        rows.append((G.score(preds[cm[0]], R8T[ci]), len(cm)))
    keys = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'var_occ_f1', 'accuracy', 'unknown_known_fpr', 'free_recall', 'furn_tp', 'furn_fp', 'furn_fn')
    out = {}
    for k in keys:
        vals = [(r[k], w) for r, w in rows if r.get(k) is not None]
        if k.startswith('furn_t') or k.startswith('furn_f') and k in ('furn_fp', 'furn_fn'):
            out[k] = float(sum(v * w for v, w in vals))
        else:
            out[k] = float(sum(v * w for v, w in vals) / max(sum(w for _, w in vals), 1)) if vals else None
    return out


def run_house(rooms, index):
    t0 = time.time()
    fx = W.make_fixture('TRAIN', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    c = G.CONDITIONS['cam']
    L_k0, L_pub = [], []
    for a in range(n):
        z = G.frame_draws(SEED, rooms, index, a)
        frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        scans = []
        for j in range(traj.shape[0]):
            hs, ls = traces[a][j]
            scan, _ = G.measure(c, hs, ls, traj[j, a], G.pose_draws(SEED, rooms, index, a, j, len(rays)), ds, rays)
            scans.append(scan)
        L_k0.append(G.integrate_agent(shape, cell, ds, scans, (I,))[I])
        L_pub.append(G.integrate_agent(shape, cell, ds, [G.to_public(s, frame) for s in scans], (I,))[I])
    t_maps = time.time() - t0
    res = dict(house=f'TRAIN_r{rooms}_i{index}', agents=n, comps=F['comps'], directed=len(F['directed']), shape=list(shape), t_maps=t_maps)
    # ---------------- (1) own-map decomposition (agent frame = cam_nf maps; clean = private)
    own = []
    for a in range(n):
        L = L_k0[a]; k = ~G.is_unk(L); tn = G.tern(L); tc = private[a]
        kc = tc >= 0
        mix = np.where(k, np.where(kc, tc, tn), np.int8(-1)).astype(np.int8)
        hist = np.bincount(np.digitize(L[k].astype(np.int64), [-40, -21, -1, 0, 16, 33, 69], right=True), minlength=8)
        own.append(dict(
            noisy=bits_split(L, k), noisy_pub_total=G.int_ctx_bytes(L_pub[a], ~G.is_unk(L_pub[a])),
            clean_known=G.ctx_tern_bits(np.where(kc, np.int8(0), np.int8(-1))) / 8, clean_tern=G.ctx_tern_bytes(tc) - 2,
            mix_tern=G.ctx_tern_bits(mix) / 8,
            n_known=int(k.sum()), n_known_clean=int(kc.sum()), n_known_xor=int((k ^ kc).sum()),
            n_occ=int((tn == 1).sum()), n_occ_clean=int((tc == 1).sum()),
            occ_fp_in_free=int(((tn == 1) & (tc == 0)).sum()), occ_in_clean_unknown=int(((tn == 1) & ~kc).sum()),
            occ_fn=int(((tc == 1) & (tn == 0)).sum()), occ_fn_unknown=int(((tc == 1) & ~k).sum()),
            hist_L=hist.tolist()))
    res['own'] = own
    # ---------------- (2) sum-fusion precision sweep
    res['sweep'] = {}
    for mapset, maps in (('pub', L_pub), ('truepose', L_k0)):
        rec = {}
        Tm = [G.tern(L) for L in maps]
        msgs, preds = G.relay(Tm, F, G.comb_maxocc)
        rec['tern_join'] = dict(metrics=comp_rows(preds, F, R8T), rnet=float(sum(G.ctx_tern_bytes(msgs[e]) for e in F['directed'])),
                                own=float(sum(G.ctx_tern_bytes(t) for t in Tm)))
        for q in QUANT:
            own_q, recv = quantise(maps, q)
            msgs, bel = G.relay(own_q, F, G.comb_sum)
            preds = [G.sum_tern(b) for b in bel]
            eb = {e: G.int_ctx_bytes(*msgs[e]) for e in F['directed']}
            rnet = float(sum(eb.values()))
            # collect-dispatch: up = integer sums toward a centre root, down = ternary of the component sum (= every belief)
            up, down, down_innov = 0., 0., 0.
            for cm in F['comps']:
                if len(cm) < 2:
                    continue
                root, depth = center_root(F, cm)
                tb = G.ctx_tern_bytes(preds[cm[0]])
                for (s, r) in F['directed']:
                    if s in depth and r in depth:
                        if depth[r] < depth[s]:
                            up += eb[(s, r)]
                        else:
                            down += tb
                            have = G.sum_tern(msgs[(r, s)])           # what the child r already holds: the sum of its own side
                            d = np.where(preds[cm[0]] != have, preds[cm[0]], np.int8(-1)).astype(np.int8)
                            down_innov += G.ctx_tern_bytes(d)
            tern_part = float(sum(G.ctx_tern_bits(G.sum_tern(msgs[e])) / 8 for e in F['directed'])) if q == 'full' else None
            thr = {}
            if q == 'vote_k2':
                for t in (-2, -1, 1, 2):
                    pt = [np.where(b[1], (b[0] > t).astype(np.int8), np.int8(-1)).astype(np.int8) for b in bel]
                    thr[str(t)] = comp_rows(pt, F, R8T)
            rec[q] = dict(metrics=comp_rows(preds, F, R8T), rnet=rnet, rnet_cd_up_int_down_tern=up + down, cd_up=up, cd_down=down,
                          cd_down_innov=down_innov, rnet_cd_innov=up + down_innov, vote_thresholds=thr,
                          own=float(sum(G.int_ctx_bytes(v, k) for v, k in own_q)), recon=recv, rnet_tern_part=tern_part)
        res['sweep'][mapset] = rec
    res['t_total'] = time.time() - t0
    return res


if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for spec in sys.argv[2:]:
        rooms, index = (int(x) for x in spec.split(':'))
        r = run_house(rooms, index)
        with open(os.path.join(out, f"{r['house']}.json"), 'w') as fh:
            json.dump(r, fh, indent=1)
        print(r['house'], f"{r['t_maps']:.0f}s maps, {r['t_total']:.0f}s total", flush=True)
