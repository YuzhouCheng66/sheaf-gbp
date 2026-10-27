"""a10_probe.py (A10, 2026-10-27): distributed-source-coding probes on the 6 VAL demo houses (8 agents, 7 tree edges, 14 directed).
Lossless, exact adaptive context codes (a10_np). Quantities:
  per agent   : ctx of own map; ctx of its visible surface V = (own == 1) (+ 5.5 B of poses = the DL's 44 bit; the own map is EXACTLY
                raycast(V, poses) for first-hit rays, so V + poses is a lossless code of the own map); gzip; the agent's INNOVATION over
                all others (what only it saw), ternary and V.
  per directed tree edge s->r (S = sender side, R = receiver side):
    full      ctx(U_S)                               -- what a lossless converged relay message costs (evidence pack 3.1: 1,752 B)
    wz_own    cond(U_S | own_r)                       -- Wyner-Ziv / Slepian-Wolf proxy: decoder side information = receiver's own map
    inn_own   innov(U_S v own_r  over own_r)          -- only what the receiver itself lacks (round-1 innovation)
    inn       innov(U_all over U_R)                   -- only what the receiver SIDE lacks (= the converged innovation; cut-set bound)
    Vfull / Vinn / Vinn_own : the same with the surface set V (+ poses of S), the known mask being re-derived by ray casting.
  network totals per house: our fixed protocol (8 rounds x 14 x 2,048 B), event-triggered, once-per-edge; flood of own maps (gzip,
  ctx, V); union relay; collect-distribute with conditional distribute messages (best root); cut-set innovation sums.
  scaling: ctx / V of the union of BFS-grown connected subsets of k = 1..8 agents.
Context designs fixed from a10_ctxsearch (innovation ternary: 6 causal x 3 forward side-known flags x sender-reachability flag;
conditional: 6 causal x side value; V innovation: 9 causal x reachability flag). CPU / numpy only."""
import glob, gzip as gz_, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a10_np import *  # noqa

DATA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'demo_web3d', 'data'))
POSE_B = 44 / 8                                                       # bytes per agent pose triple (world_dl.py: 44 bit)


def planes(m):
    return np.packbits(np.stack([m >= 0, m == 1]).reshape(-1)).tobytes()


def gzb(m):
    return len(gz_.compress(planes(m), 9))


def innov_t(Um, R, reach):
    t, s = T(Um), T(R); sel = (s == 0)
    c = (causal(t, 3, NB6) * 8 + fwd_flags(s > 0)) * 2 + reach.astype(np.int64)
    return kt_bits(t[sel], c[sel], 3)[0] / 8 + 2


def cond_t(S, side):
    t, s = T(S), T(side)
    return kt_bits(t, causal(t, 3, NB6) * 3 + s, 3)[0] / 8 + 2


def innov_v(Vm, Rknown, reach):
    v = Vm.astype(np.int64); sel = ~Rknown
    c = causal(v, 2) * 2 + reach.astype(np.int64)
    return kt_bits(v[sel], c[sel], 2)[0] / 8 + 2


def reach_of(agents, side_map, pos, cell):
    r = np.zeros(side_map.shape, bool)
    for a in agents:
        r |= reachable(side_map == 1, cell, pos[a])
    return r


def do_house(path):
    out = dict(agents=[], edges=[], houses=[], scaling=[])
    m, own, pred, gt = load(path); n = own.shape[0]; adj = adjacency(m['edges'], n)
    pos = np.asarray(m['positions_m']); cell = m['cell_m']; H = m['house']
    allA = list(range(n)); Uall = union(own, allA); Vall = Uall == 1
    # ---------------- per agent
    for a in range(n):
        others = [b for b in allA if b != a]; Ro = union(own, others)
        reach = reach_of([a], Ro, pos, cell)
        out['agents'].append(dict(house=H, agent=a, gzip=gzb(own[a]), ctx=tern_bytes(own[a]), V=bin_bytes(own[a] == 1) + POSE_B,
                                  known=float((own[a] >= 0).mean()), surf_vox=int((own[a] == 1).sum()),
                                  inn_vs_others=innov_t(Uall, Ro, reach), Vinn_vs_others=innov_v(Vall, Ro >= 0, reach) + POSE_B))
    # ---------------- per directed edge
    E = {}
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Sset = side_of(adj, s_, r_); Rset = side_of(adj, r_, s_)
            S = union(own, Sset); R = union(own, Rset); Ro = own[r_]
            reachR = reach_of(Sset, R, pos, cell); reachO = reach_of(Sset, Ro, pos, cell)
            US_o = np.maximum(S, Ro)
            e = dict(house=H, s=s_, r=r_, nS=len(Sset), nR=len(Rset), full=tern_bytes(S), gzip=gzb(S),
                     wz_own=cond_t(S, Ro), inn_own=innov_t(US_o, Ro, reachO), inn=innov_t(Uall, R, reachR),
                     Vfull=bin_bytes(S == 1) + POSE_B * len(Sset), Vinn_own=innov_v(US_o == 1, Ro >= 0, reachO) + POSE_B * len(Sset),
                     Vinn=innov_v(Vall, R >= 0, reachR) + POSE_B * len(Sset),
                     new_known_vox=int(((S >= 0) & (R < 0)).sum()), new_surf_vox=int(((S == 1) & (R < 0)).sum()))
            out['edges'].append(e); E[(s_, r_)] = e
    # ---------------- network totals
    nd = 2 * len(m['edges'])
    roots = {}
    for root in allA:
        d = hops_from(adj, root); tot_t = tot_v = tot_t_o = 0.
        for i, j in m['edges']:
            c, p = (i, j) if d[i] > d[j] else (j, i)                  # c = child (farther from root)
            tot_t += E[(c, p)]['full'] + E[(p, c)]['inn']             # collect: unconditional side union; distribute: innovation over child side
            tot_v += E[(c, p)]['Vfull'] + E[(p, c)]['Vinn']
        roots[root] = (tot_t, tot_v)
    # event-triggered fixed-length: message s->r changes whenever a new sender-side node enters its (t-1)-hop ball
    ev = 0
    for i, j in m['edges']:
        for s_, r_ in ((i, j), (j, i)):
            Sset = side_of(adj, s_, r_); d = hops_from({u: adj[u] & set(Sset) for u in Sset}, s_)
            ev += len(set(min(t - 1, max(d.values())) for t in range(1, 9)))
    ag = [x for x in out['agents'] if x['house'] == H]; ed = [x for x in out['edges'] if x['house'] == H]
    h = dict(house=H, n_dir=nd,
             ours_fixed_2KB=8 * nd * 2048, ours_event_2KB=ev * 2048, ours_once_2KB=nd * 2048, ours_once_8KB=nd * 8192, event_sends=ev,
             flood_gzip=sum(x['gzip'] for x in ag) * len(m['edges']), flood_ctx=sum(x['ctx'] for x in ag) * len(m['edges']),
             flood_V=sum(x['V'] for x in ag) * len(m['edges']),
             relay_union_gzip=sum(x['gzip'] for x in ed), relay_union_ctx=sum(x['full'] for x in ed), relay_union_V=sum(x['Vfull'] for x in ed),
             cd_ctx_best=min(v[0] for v in roots.values()), cd_V_best=min(v[1] for v in roots.values()),
             cut_ctx=sum(x['inn'] for x in ed), cut_V=sum(x['Vinn'] for x in ed),
             union_ctx_one=tern_bytes(Uall), union_V_one=bin_bytes(Vall) + POSE_B * n,
             sum_inn_vs_others_ctx=sum(x['inn_vs_others'] for x in ag), sum_inn_vs_others_V=sum(x['Vinn_vs_others'] for x in ag))
    out['houses'].append(h)
    # ---------------- scaling with the number of agents (BFS-grown connected subsets)
    for seed in allA:
        d = hops_from(adj, seed); order = sorted(allA, key=lambda u: (d[u], u))
        for k in range(1, n + 1):
            Uk = union(own, order[:k])
            out['scaling'].append(dict(house=H, seed=seed, k=k, ctx=tern_bytes(Uk), V=bin_bytes(Uk == 1) + POSE_B * k, known=float((Uk >= 0).mean())))
    print(H, json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in h.items()}), flush=True)
    return out


# the shared CPU box runs out of COMMIT memory (dozens of probe processes): cache per house and retry an allocation failure
import gc
out = dict(agents=[], edges=[], houses=[], scaling=[])
t0 = time.time()
for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
    cache = os.path.join(HERE, 'cache_probe_' + os.path.basename(path))
    for attempt in range(60):
        if os.path.exists(cache):
            break
        try:
            res = do_house(path); json.dump(res, open(cache, 'w'))
        except MemoryError as exc:
            print('MemoryError, retry', attempt, exc, flush=True); gc.collect(); time.sleep(20)
    res = json.load(open(cache))
    for k in out:
        out[k] += res[k]
    print(os.path.basename(path), f'{time.time() - t0:.0f}s', flush=True)
json.dump(out, open(os.path.join(HERE, 'a10_probe.json'), 'w'), indent=1)


def med(v):
    return float(np.median(v))


print('\nAGENTS (median over 48):', {k: round(med([x[k] for x in out['agents']]), 1) for k in ('gzip', 'ctx', 'V', 'inn_vs_others', 'Vinn_vs_others', 'known', 'surf_vox')})
print('EDGES (median over 84):', {k: round(med([x[k] for x in out['edges']]), 1) for k in ('gzip', 'full', 'wz_own', 'inn_own', 'inn', 'Vfull', 'Vinn_own', 'Vinn', 'new_known_vox', 'new_surf_vox')})
for nS in range(1, 8):
    r = [x for x in out['edges'] if x['nS'] == nS]
    if r:
        print(f'  |S|={nS} n={len(r):2d}', {k: round(med([x[k] for x in r])) for k in ('full', 'wz_own', 'inn_own', 'inn', 'Vfull', 'Vinn')})
print('HOUSES (mean over 6):', {k: round(float(np.mean([x[k] for x in out['houses']]))) for k in out['houses'][0] if k != 'house'})
for k in range(1, 9):
    r = [x for x in out['scaling'] if x['k'] == k]
    print(f'  scaling k={k}: ctx {med([x["ctx"] for x in r]):7.0f}  V {med([x["V"] for x in r]):7.0f}  known {med([x["known"] for x in r]):.3f}')
