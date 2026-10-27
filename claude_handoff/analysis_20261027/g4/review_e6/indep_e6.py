"""Reviewer's independent recompute of G4-E6 (D8-51) on ONE task, val_r2_i3_s1 (the smoke's multi-island house).
Written 2026-10-27 by the code reviewer. Does not import g4_island_compose. Uses only the frozen pieces that define the world and the
matcher (g4_noise_panels: fixture, measure, V3 integration, score; g4_registration: register_pair / evidence / planes_tern; g4_e3r:
direct_sums, net_logits, net_readout, load_ckpt). Everything that G4-E6 adds is re-implemented here:
  own SE(2) algebra, own BFS islands of accepted edges, own believed positions, own island gauge (mean yaw, mean pivot displacement),
  own scan transform, own B4 (direct sum of the L stalk > 0), own aligned loop (G_c from register_pair(truth, own B4 belief)),
  own NET / B6 read-outs on DIRECT sums (not relay_sums), own agent-expanded mean.
Then compares number for number with the implementation's smoke job file.
Usage: python -B indep_e6.py --tier cam|cam_nf|harsh [--k-only]
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, collections, json, math, time  # noqa: E401
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_e3r as E  # noqa: E402
import g4_registration as R  # noqa: E402
G = E.G; W = G.W; torch = E.torch
A27 = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(A27, 'review_E3R', 'inputs')
FIT = os.path.join(A27, 'g4', 'e1c', 'fit.json')
JOB = 'val_r2_i3_s1'; ROOMS, INDEX, SEED = 2, 3, 1


# ---------------- own SE(2)
def T(yaw_deg, tx, ty):
    th = math.radians(yaw_deg); c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def Tinv(M):
    Rt = M[:2, :2].T; out = np.eye(3); out[:2, :2] = Rt; out[:2, 2] = -Rt @ M[:2, 2]; return out


def yaw_of(M):
    return math.atan2(M[1, 0], M[0, 0])


def to_params(M):
    return [math.degrees(yaw_of(M)), float(M[0, 2]), float(M[1, 2])]


def island_gauge(Cs, piv):
    """H: rotation by the mean yaw about piv, then the mean displacement of piv"""
    th = float(np.mean([yaw_of(M) for M in Cs]))
    d = np.mean([M[:2, :2] @ piv + M[:2, 2] - piv for M in Cs], 0)
    c, s = math.cos(th), math.sin(th); Rm = np.array([[c, -s], [s, c]])
    H = np.eye(3); H[:2, :2] = Rm; H[:2, 2] = piv + d - Rm @ piv
    return H


def islands_bfs(n, acc):
    adj = collections.defaultdict(list)
    for (i, j) in acc:
        adj[i].append(j); adj[j].append(i)
    seen, out = set(), []
    for a in range(n):
        if a in seen:
            continue
        comp, q = [], collections.deque([a]); seen.add(a)
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        out.append(sorted(comp))
    return out


def my_K(n, edges, bpos, flip=False):
    """edge (i parent, j child), est = Delta mapping j's frame into i's: C_j = C_i Delta (flip=True: the WRONG direction, for the check)"""
    acc = {}
    for e in edges:
        if e['corrected']:
            D = T(*e['est']); acc[(int(e['i']), int(e['j']))] = Tinv(D) if flip else D
    isl = islands_bfs(n, acc)
    K = [None] * n; C = {}
    for g in isl:
        C[g[0]] = np.eye(3); todo = collections.deque([g[0]])
        while todo:
            u = todo.popleft()
            for (i, j), D in acc.items():
                if i == u and j not in C:
                    C[j] = C[i] @ D; todo.append(j)
                elif j == u and i not in C:
                    C[i] = C[j] @ Tinv(D); todo.append(i)
        piv = np.mean([bpos[a] for a in g], 0)
        Hi = Tinv(island_gauge([C[a] for a in g], piv))
        for a in g:
            K[a] = Hi @ C[a]
    return K, isl


def move_scan(scan, M):
    o, D, r, sm = scan
    R3 = np.eye(3); R3[:2, :2] = M[:2, :2]; t3 = np.array([M[0, 2], M[1, 2], 0.])
    return (R3 @ o + t3, D @ R3.T, r, sm)


def integ(shape, cell, ds, scans, M):
    if np.array_equal(M, np.eye(3)):
        return G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3']
    return G.integrate_agent(shape, cell, ds, [move_scan(s, M) for s in scans], ('V3',))['V3']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tier', required=True); ap.add_argument('--k-only', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(1); tier = a.tier; t0 = time.time()
    e4 = json.load(open(os.path.join(A27, 'g4', 'e4', 'jobs', JOB + '.json')))
    sm = json.load(open(os.path.join(A27, 'g4', 'e6_smoke', 'jobs', JOB + '.json')))
    e4c = e4['conds'][tier]; smc = sm['conds'][tier]
    fx = W.make_fixture('VALIDATION', INDEX, ROOMS)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    Hh = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n); comps = F['comps']
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool); T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(private[cm].max(0), Hh, mult=(private[cm] >= 0).sum(0)) for cm in comps]
    rays = W.fibonacci_sphere(W.BEAMS); c = G.CONDITIONS[tier]; has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0
    pubs, MAs = [], []
    for ag in range(n):
        frame = None
        if has_inter:
            z = G.frame_draws(SEED, ROOMS, INDEX, ag)
            frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[ag], np.float64))
        MAs.append(R.frame_matrix(frame))
        tr = [G.trace_pose(occ, cell, traj[jj, ag], rays) for jj in range(traj.shape[0])]
        scans = [G.measure(c, *tr[jj], traj[jj, ag], G.pose_draws(SEED, ROOMS, INDEX, ag, jj, len(rays)), ds, rays)[0] for jj in range(traj.shape[0])]
        pubs.append([G.to_public(s, frame) for s in scans])
    L0 = [integ(shape, cell, ds, pubs[ag], np.eye(3)) for ag in range(n)]
    out = dict(job=JOB, tier=tier, pub_sha_equal_e4=W.array_hash(np.stack(L0)) == e4c['pub_sha256'])
    # ---- K
    bpos = [np.mean([np.asarray(s[0], np.float64)[:2] for s in pubs[ag]], 0) for ag in range(n)]
    out['believed_minus_true_pos_cm'] = [float(np.linalg.norm(bpos[ag] - np.asarray(positions[ag])[:2]) * 100) for ag in range(n)]
    edges = [dict(i=e['i'], j=e['j'], corrected=e['corrected'], est=e['est']) for e in e4c['registration']['edges']]
    K, isl = my_K(n, edges, bpos)
    Kimpl = [T(*p) for p in smc['arms']['e6']['K']]
    out['islands'] = isl; out['islands_equal_impl'] = isl == smc['e6']['islands']
    out['K_params_mine'] = [to_params(x) for x in K]
    out['K_maxabs_diff_vs_impl'] = float(max(np.abs(x - y).max() for x, y in zip(K, Kimpl)))
    # direction check (truth used ONLY in this review diagnostic): within an island the corrected frames K_a A_a should agree
    def spread(KK):
        res = []
        for g in isl:
            if len(g) < 2:
                continue
            Es = [KK[x] @ MAs[x] for x in g]
            for x, Ex in zip(g, Es):
                for y, Ey in zip(g, Es):
                    if x < y:
                        D = Tinv(Ex) @ Ey; cen = np.asarray(positions[y])[:2]
                        res.append([abs(math.degrees(yaw_of(D))), float(np.linalg.norm(D[:2, :2] @ cen + D[:2, 2] - cen) * 100)])
        return np.asarray(res)
    Kflip, _ = my_K(n, edges, bpos, flip=True)
    if has_inter:
        s_ok, s_flip, s_no = spread(K), spread(Kflip), spread([np.eye(3)] * n)
        out['within_island_rel_err'] = dict(e6=dict(yaw_deg_max=float(s_ok[:, 0].max()), disp_cm_max=float(s_ok[:, 1].max()), disp_cm_mean=float(s_ok[:, 1].mean())),
                                            flipped_direction=dict(yaw_deg_max=float(s_flip[:, 0].max()), disp_cm_max=float(s_flip[:, 1].max()), disp_cm_mean=float(s_flip[:, 1].mean())),
                                            noreg=dict(yaw_deg_max=float(s_no[:, 0].max()), disp_cm_max=float(s_no[:, 1].max()), disp_cm_mean=float(s_no[:, 1].mean())))
    # true-position pivot variant (R-A probe), report
    Kt, _ = my_K(n, edges, [np.asarray(p, np.float64)[:2] for p in positions])
    out['K_truepivot_maxabs_diff'] = float(max(np.abs(x - y).max() for x, y in zip(K, Kt)))
    if a.k_only:
        out['seconds'] = time.time() - t0
        json.dump(out, open(os.path.join(HERE, f'indep_{tier}.json'), 'w'), indent=1); print(json.dumps(out, indent=1)); return
    # ---- read-outs
    ck = []
    for sd, dn, half in (('342101', 's342101', 'A'), ('342202', 's342202', 'B')):
        ckd, model = E.load_ckpt(os.path.join(INP, dn, 'final.pt'), 'cpu')
        ev = E.Evidence.from_recipe(ckd['recipe']['evidence'])
        evh = E.Evidence.from_spec(f'c:{FIT}:{half}:cam_nf')
        assert np.array_equal(ev.table, evh.table) and ev.theta == evh.theta
        j = int(json.load(open(os.path.join(INP, dn, 'threshold.json')))['j'])
        ck.append((f'NET_{sd}', f'B6_{half}', model, ev, j))
    Ltab = E.Evidence.from_spec('L').table

    def read(maps):
        res = {k: [] for k in ('B4', 'B6_A', 'B6_B', 'NET_342101', 'NET_342202')}; b4 = []
        for ci, cm in enumerate(comps):
            SL, kL = E.direct_sums(maps, cm, Ltab)
            p4 = np.where(kL > 0, (SL > 0).astype(np.int8), np.int8(-1)).astype(np.int8); b4.append(p4)
            res['B4'].append([G.score(p4, R8T[ci]), len(cm)])
            for name, b6, model, ev, j in ck:
                S, kc = E.direct_sums(maps, cm, ev.table); assert np.array_equal(kc, kL)
                res[b6].append([G.score(np.where(kc > 0, (S > ev.theta).astype(np.int8), np.int8(-1)).astype(np.int8), R8T[ci]), len(cm)])
                lg = E.net_logits(model, [S], [kc], 'cpu', 1)[0]
                res[name].append([G.score(E.net_readout(lg, kc, j), R8T[ci]), len(cm)])
        return res, b4

    def expmean(rows, key):
        xs = [r[key] for r, k in rows for _ in range(k) if r.get(key) is not None]
        return float(np.mean(xs)) if xs else None
    prm = dict(R.MATCH); arms = {}
    for arm, KK in (('noreg', [np.eye(3)] * n), ('e6', K)):
        maps = [L0[ag] if np.array_equal(KK[ag], np.eye(3)) else integ(shape, cell, ds, pubs[ag], KK[ag]) for ag in range(n)]
        asis, b4 = read(maps); Gs = []; al = [None] * n
        for ci, cm in enumerate(comps):
            ev_t = R.evidence(R.planes_tern(T8[cm[0]], cell, prm), cell, prm); ev_b = R.evidence(R.planes_tern(b4[ci], cell, prm), cell, prm)
            Gc = R.register_pair(ev_t, ev_b, cell, prm)['M']; Gs.append(to_params(Gc))
            for ag in cm:
                al[ag] = maps[ag] if np.array_equal(Gc, np.eye(3)) else integ(shape, cell, ds, pubs[ag], Gc @ KK[ag])
        aligned, _ = read(al)
        arms[arm] = dict(G=Gs, asis={ro: {k: expmean(v, k) for k in ('furn_f1', 'free_recall', 'unknown_known_fpr')} for ro, v in asis.items()},
                         aligned={ro: {k: expmean(v, k) for k in ('furn_f1', 'free_recall', 'unknown_known_fpr')} for ro, v in aligned.items()})
        # compare with the implementation's smoke job
        imp = smc['arms'][arm]
        arms[arm]['G_equal_impl'] = [list(x['G']) for x in imp['align']] == Gs
        cmpd = {}
        for row in ('asis', 'aligned'):
            for ro in arms[arm][row]:
                for k in ('furn_f1', 'free_recall', 'unknown_known_fpr'):
                    mine = arms[arm][row][ro][k]; theirs = expmean(imp[row][ro], k)
                    cmpd[f'{row}:{ro}:{k}'] = [mine, theirs, None if (mine is None or theirs is None) else mine - theirs]
        arms[arm]['vs_impl'] = cmpd
        del maps, al
    out['arms'] = arms
    out['delta_pp_aligned'] = {ro: 100 * (arms['e6']['aligned'][ro]['furn_f1'] - arms['noreg']['aligned'][ro]['furn_f1']) for ro in arms['e6']['aligned']}
    out['delta_pp_asis'] = {ro: 100 * (arms['e6']['asis'][ro]['furn_f1'] - arms['noreg']['asis'][ro]['furn_f1']) for ro in arms['e6']['asis']}
    out['max_abs_diff_vs_impl'] = max(abs(v[2]) for arm in arms.values() for v in arm['vs_impl'].values() if v[2] is not None)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, f'indep_{tier}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'arms'}, indent=1))
    for arm in arms:
        print(arm, 'G_equal_impl', arms[arm]['G_equal_impl'])
        for k, v in arms[arm]['vs_impl'].items():
            if 'furn_f1' in k:
                print('  ', arm, k, v)


if __name__ == '__main__':
    main()
