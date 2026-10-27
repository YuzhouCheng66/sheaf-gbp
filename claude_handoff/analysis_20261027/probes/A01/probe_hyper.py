"""probe_hyper.py (A01, 2026-10-27): does the CURRENT learned code help the exact ray-order lossless coder as side information?
Real demo own maps and the exact poses (poses_demo.json, bit-exact). Ray-order codec as in probe_visorder (one bit per first visit;
context = 6 face neighbours decided before, occupied edge-neighbour count, distance band; 6,561 contexts). Hyper variant: the context is
extended by the model's decoded state at the voxel (3S-X + calibrated rule: unknown / free / occupied), i.e. the receiver first decodes
the 2 KB (or entropy-coded) code and then uses its prediction as a prior for the exact residual ('hyperprior' use of the sheaf code).
K0: prediction of the own code; R8: prediction of the fused code; union events processed agent after agent (reached set order).
Probabilities: adaptive KT per map and LOO-pretrained (counts from the other 5 houses' own-map events of the same context family).
Total rate of the hyper scheme = latent bytes (probe_wire: entropy-coded current code) + residual bytes.
Usage: python probe_hyper.py DATA_DIR POSES_JSON OUT_JSON [processes]"""
import glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from probe_visorder import fibonacci_sphere, first_visits, nb_flat, event_contexts, kt_bits, FACE, EDGE, INF, NCTX  # noqa: E402

NCTX_H = NCTX * 3


def house_job(args):
    path, poses_json = args
    from rate_baselines_lib import load
    t0 = time.time()
    m, own, pred, gt = load(path); n = m['agents']; cell = tuple(m['cell_m']); occ = gt['occ'].astype(np.uint8); shape = occ.shape
    traj = np.asarray(json.load(open(poses_json))[m['house']]['trajectory_m'])
    rays = fibonacci_sphere(65536)
    NBF = [nb_flat(shape, d) for d in FACE]; NBE = [nb_flat(shape, d) for d in EDGE]
    X, Y, Z = shape
    xc = (np.arange(X) + .5) * cell[0]; yc = (np.arange(Y) + .5) * cell[1]; zc = (np.arange(Z) + .5) * cell[2]
    state = occ.reshape(-1).astype(np.int64)
    ts, bands = [], []
    for a in range(n):
        o, t = first_visits(occ, cell, traj[:, a], rays)
        assert np.array_equal(o, own[a]), ('own map not reproduced', m['house'], a)
        ts.append(t)
        centre = traj[:, a].mean(0)
        d = np.sqrt((xc[:, None, None] - centre[0]) ** 2 + (yc[None, :, None] - centre[1]) ** 2 + (zc[None, None, :] - centre[2]) ** 2).reshape(-1)
        bands.append(np.digitize(d, [1.0, 3.0]).astype(np.int64))
    R8 = np.asarray(m['reached'][m['hops'].index(8)], bool)
    BIG = 10 ** 9
    ev = {}
    for hop, key in ((0, 'own'), (8, 'union8')):
        for a in range(n):
            if hop == 0:
                tu, band = ts[a], bands[a]
            else:
                tu = np.full(occ.size, INF, np.int64); band = np.zeros(occ.size, np.int64)
                for rank, s_ in enumerate(np.nonzero(R8[a])[0]):
                    ts_ = np.where(ts[s_] < INF, rank * BIG + ts[s_], INF)
                    take = ts_ < tu; tu = np.where(take, ts_, tu); band = np.where(take, bands[s_], band)
            v, ctx, sym, order = event_contexts(tu, state, band, NBF, NBE)
            ps = (pred[hop][a].reshape(-1)[v] + 1).astype(np.int64)        # 0 unknown, 1 free, 2 occupied (model's decoded state)
            ev[(key, a)] = dict(ctx=ctx[order].astype(np.int32), ctxh=(ctx * 3 + ps)[order].astype(np.int32), sym=sym[order].astype(np.int8),
                                agree=float((ps[order] - 1 == sym[order]).mean()))
    cnt = np.zeros((NCTX, 2)); cnth = np.zeros((NCTX_H, 2))
    for (key, a), e in ev.items():
        if key == 'own':
            np.add.at(cnt, (e['ctx'].astype(np.int64), e['sym'].astype(np.int64)), 1.); np.add.at(cnth, (e['ctxh'].astype(np.int64), e['sym'].astype(np.int64)), 1.)
    print(m['house'], f'{time.time()-t0:.0f}s', flush=True)
    return dict(house=m['house'], ev={f'{k[0]}|{k[1]}': v for k, v in ev.items()}, cnt=cnt, cnth=cnth)


if __name__ == '__main__':
    import multiprocessing
    DATA, POSES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; procs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    with multiprocessing.get_context('spawn').Pool(procs) as pool:
        H = pool.map(house_job, [(p, POSES) for p in paths])
    rows = []
    for hi, h in enumerate(H):
        pri = sum(g['cnt'] for gi, g in enumerate(H) if gi != hi); prih = sum(g['cnth'] for gi, g in enumerate(H) if gi != hi)
        for key, e in h['ev'].items():
            kind, a = key.split('|')
            r = dict(house=h['house'], kind=kind, agent=int(a), events=int(len(e['sym'])), model_agrees=e['agree'],
                     base_adaptive=float(kt_bits(e['ctx'], e['sym']).sum() / 8 + 2), base_pretrained=float(kt_bits(e['ctx'], e['sym'], pri).sum() / 8 + 2),
                     hyper_adaptive=float(kt_bits(e['ctxh'], e['sym']).sum() / 8 + 2), hyper_pretrained=float(kt_bits(e['ctxh'], e['sym'], prih).sum() / 8 + 2))
            rows.append(r)
    S = {k: {c: float(np.median([r[c] for r in rows if r['kind'] == k])) for c in ('events', 'model_agrees', 'base_adaptive', 'base_pretrained', 'hyper_adaptive', 'hyper_pretrained')} for k in ('own', 'union8')}
    json.dump(dict(summary=S, rows=rows), open(OUT, 'w'), indent=1)
    print(json.dumps(S, indent=1))
