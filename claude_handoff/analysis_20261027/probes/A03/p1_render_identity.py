"""p1_render_identity.py (A03): is a map a deterministic function of its OCCUPIED set + the poses, and what does that save?
(1) observe_t(true world, agent poses) == the cached own map (torch re-implementation is exact);
(2) observe_t(own_a == 1 as the whole world, poses_a) == own_a  (visibility is free given the hits);
(3) R8 target T_a == join_b observe_t(T_a == 1, poses_b) over reached b (the fused map is hits + poses as well);
(4) bytes: ctx ternary (lossless map) vs ctx binary of the hit set + 44 bit per pose triple (lossless 'hits + poses' codec).
Usage: python -B p1_render_identity.py [all|demo]"""
import json, os, sys, time
import numpy as np
from a03lib import observe_t, cell_of, ctx_bytes, ctx_bytes_binary, targets, load_fx, names, HERE

def house(name):
    rows = []
    t0 = time.time()
    fx, meta = load_fx(name); own = fx['private']; n = own.shape[0]; cell = cell_of(meta['chart_m']); traj = fx['trajectory']
    occ = fx['occ'].astype(bool)
    ok_true = ok_hits = 0
    for a in range(n):
        o = observe_t(occ, cell, traj[:, a]) if a < 2 else None          # spot-check the re-implementation on 2 agents per house
        if o is not None:
            ok_true += int((o == own[a]).all())
        r = observe_t(own[a] == 1, cell, traj[:, a]); ok_hits += int((r == own[a]).all())
        rows.append(dict(house=name, agent=a, kind='own', ctx=ctx_bytes(own[a]), hits_ctx=ctx_bytes_binary(own[a] == 1), poses_bytes=44 / 8,
                         n_hits=int((own[a] == 1).sum()), known=float((own[a] >= 0).mean())))
    R = fx['reached'][4]; T8 = targets(own, R)
    seen = {}
    ok_union = 0
    for a in range(n):
        key = tuple(np.nonzero(R[a])[0])
        if key not in seen:
            U = None
            for b in key:
                o = observe_t(T8[a] == 1, cell, traj[:, b]); U = o if U is None else np.maximum(U, o)
            seen[key] = bool((U == T8[a]).all())
        ok_union += int(seen[key])
        rows.append(dict(house=name, agent=a, kind='union8', reached=len(key), ctx=ctx_bytes(T8[a]), hits_ctx=ctx_bytes_binary(T8[a] == 1),
                         poses_bytes=44 * len(key) / 8, n_hits=int((T8[a] == 1).sum()), known=float((T8[a] >= 0).mean())))
    print(f'{name}: observe_t==cache {ok_true}/2, own==render(hits) {ok_hits}/{n}, T8==join render(hits) {ok_union}/{n}, {time.time()-t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
  import multiprocessing as mp
  which = sys.argv[1] if len(sys.argv) > 1 else 'demo'
  with mp.Pool(24) as pool:
    rows = [r for rr in pool.map(house, names(demo_only=(which == 'demo'))) for r in rr]
  S = {}
  for k in ('own', 'union8'):
    r = [x for x in rows if x['kind'] == k]
    full = np.array([x['ctx'] for x in r]); hp = np.array([x['hits_ctx'] + x['poses_bytes'] for x in r])
    S[k] = dict(n=len(r), ctx_median=float(np.median(full)), hits_plus_poses_median=float(np.median(hp)), ratio_median=float(np.median(full / hp)),
                ratio_p10=float(np.percentile(full / hp, 10)), ratio_p90=float(np.percentile(full / hp, 90)), n_hits_median=float(np.median([x['n_hits'] for x in r])))
  print(json.dumps(S, indent=1))
  json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, f'p1_render_identity_{which}.json'), 'w'), indent=1)
