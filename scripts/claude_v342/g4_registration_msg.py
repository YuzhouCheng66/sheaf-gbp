"""g4_registration_msg.py (Claude, 2026-10-27): batch G4-E4b "沿 GBP 消息配准", pre-registered in claude_handoff/OVERNIGHT_REPORT_20261021.md
D8-40 and its addendum "D8-39 / D8-40 补充" (entry condition met in D8-45: VAL ceiling_islands - ceiling_pairwise = .338 / .333). CPU only.

D8-40 (verbatim, the parts that bind this code):
  底线：位姿 stalk 仍是向量值 sheaf；树边 e 把它所在的连通块切成根一侧 A_e 和子树一侧 B_e。树上 GBP 沿 e 的两条消息就是两侧证据的和。限制映射作用
    在整侧上：修正 Δ_e 把 B_e 的全部智能体刚性地移进 A_e 的坐标系。能量 E_msg = Σ_e E_edge(S(A_e), S(B_e))，其中 S(·) 是一侧消息的 surfel 集合 =
    该侧各智能体按 E4 的定义从自己 V3 墙柱平面提取的 surfel，按各自当前修正放到公共坐标系后取并；E_edge 用 E4 的对称截断点到线形式与同一组常数。
    只用消息，不用真值。
  单变量：边因子的输入。E4 = 两端智能体自己的 surfel；E4b = 边两侧消息的 surfel。优化改为按根出发的广度优先顺序逐边做 E4 的两步（相关式扫描候选
    + ICP，局部只接受让本边项下降且过门控的步），每条边做完后重算全局 E_msg，不下降就把这条边退回原值；固定扫两遍。全局 E_msg 轨迹逐步不增：断言，
    违反即任务 INCOMPLETE。其余全部与 E4 相同（同一批任务与噪声抽样、V3 重新积分、森林求和、"原样"与"整体对齐后"两行、gauge 'prior'、计分与分层）。
    匹配器和门控沿用 E4 冻结的数值。
  判据（cam 档、整体对齐后、R8 家具、两个噪声种子都要满足）：杠杆：VAL 和 fresh 都满足 D8-39 的杠杆条件（收回 >= 85 %；E_msg 单调断言全过；cam_nf
    上不劣于 -0.5 pp；空闲召回不降；FPR 不劣于 +0.3 pp）。不够：VAL 两种子收回都 < 50 %。其余 ambiguous。
  机制（只报告）：VAL cam 档上，门控拒绝、只能保持不修正的树边比例比 E4 至少减半；逐智能体去掉整体偏移后的位置误差中位数低于 E4。若 E4b 的收回比
    E4 高不到 +.15（两种子都如此），"瓶颈在消息而不在匹配器"这一假设判为不成立。只报告：harsh 档同样读数；E4b 减 E4 的收回差；位姿层每条消息的字节
    数（surfel 个数 x 每个 6 B），以及它占 S0 一次融合 6.1 KB 的比例。

Frozen dependencies, imported and never modified: g4_registration.py (G4-E4, sha256 e9574c7e..., pinned in every job) and through it
g4_noise_panels.py (G4-E1, 2313d75b...). Everything except the pose layer is E4's own code: the same jobs, noise draws, V3 maps, evidence
planes and surfels (R.planes_logodds / R.evidence), edge energy (R.edge_cost: symmetric truncated point-to-line, same constants), the
two-step pair registration (R.register_pair: correlative candidate + ICP, both accepting only steps that lower the edge term AND pass E4's
frozen gate), tree composition order (R.tree_order), gauge (R.gauge_fix 'prior'), re-integration, forest sum, the three rows (as-is,
aligned, aligned_best: R.eval_arm), the report-only truth arms oracle_all / ceiling_pairwise / ceiling_islands (R.compose / R.truth_deltas /
R.islands_K / R.observability_islands, identical to E4 -> their rows must equal E4's job files, checked at aggregation with --e4-out).

The pose layer (register_messages; inputs = the agents' believed-frame V3 maps, the public radio forest, public grid constants):
  state: one correction C_a per agent (believed frame -> common frame), start I. Tree rooted at the smallest agent of each component.
  edge e = (parent i, child j): B_e = j's subtree, A_e = the rest of the component.
  side message S(X) = union (concatenation) of the members' E4 surfels moved by their current C_a (centroids and normals; weights and
    orientation flags unchanged); side plane for the correlative search = the members' E4 w planes warped by their current C_a onto the grid
    (bilinear) and combined by the PIXEL-WISE MAXIMUM (chosen a priori: the same pooling as the islands' pooled evidence in E4; checked on
    TRAIN r2_i3 / r3_i2, seed 7; no constant was changed).
  E_msg(C) = sum over tree edges of R.edge_cost(S(B_e), S(A_e), I) (both sides already in the common frame).
  optimisation: 2 sweeps; per sweep the tree edges in BFS order of their child (root outwards); per edge R.register_pair(side A, side B)
    (target A, source B) gives Delta mapping the B side into the A side; the candidate state moves every agent of B_e rigidly,
    C_b <- Delta C_b; the global E_msg is recomputed and the candidate is kept only if it is STRICTLY lower, else the edge returns to its
    previous value. The E_msg trajectory (initial value, then every kept edge update) is asserted non-increasing (R.assert_monotone ->
    RuntimeError -> the job's failed record -> INCOMPLETE). The gate keeps E4's frozen numbers (its overlap fraction is the better-covered
    side's, so a large A side does not dilute it; checked on TRAIN, unchanged).
  final corrections: R.gauge_fix(F, C, 'prior', pivots) with E4's pivots (centroid of the component's believed-frame surfels).
  per tree edge (report): estimated relative pose inv(C_i) C_j vs the true one, errors before / after (R.pose_err), the number of kept
    updates, corrected (= any kept non-identity update), overlap_true / failed exactly as E4, the side surfel counts n_A / n_B and the bytes
    of the two pose-layer messages (6 B per surfel: two 16-bit coordinates, an 8-bit direction, an 8-bit weight).

Verdict (D8-40; R.verdict supplies the per-panel D8-39 readouts -- recovery, E_msg monotone, cam_nf guard, free recall, FPR, the G4-E1
  control check (rows must equal G4-E1's on every VAL AND fresh cam / cam_nf job), house sets, failed jobs):
  INCOMPLETE unless VAL and fresh are both complete (seeds {1, 2}, all houses) and, with --e4-out, the control and truth rows equal E4's;
  LEVER iff the D8-39 lever conditions hold on VAL AND on fresh (both seeds each); INSUFFICIENT iff VAL recovery < .50 in both seeds; else
  AMBIGUOUS. Mechanism (report, needs --e4-out): VAL cam fraction of tree edges left uncorrected, E4b vs E4 (halved?); median per-agent
  position error after removing the component mean, E4b vs E4 (lower?); recovery E4b - E4 per seed (< +.15 in both seeds -> the
  hypothesis "the bottleneck is the message, not the matcher" is rejected); harsh readouts; pose-layer message bytes vs S0 (6,066 B).
  Review e4b M1-M3 and the "D8-45 补充" (written before the formal run, after the VAL smoke diagnostics): report-only optimizer
  diagnostics per job and tier -- E_msg at E4's solution (E4's estimator R.register_forest on the same L_pub, no truth), E_msg at the true
  poses (truth only labels it), at every global rejection the E_msg of moving the child alone (logged, never used); the 'uncorrected'
  mechanism row counts GATE-REFUSED edges only (no local step in any sweep), globally rejected edges are reported separately; the
  mechanism reading 'hypothesis rejected' additionally needs, in each VAL seed, >= 90 % of the cam jobs with E4b final E_msg <= E_msg at
  E4's solution, otherwise it reads "limited by the D8-40 optimizer's move set: this batch cannot test the hypothesis"; the verdict
  text carries the oracle_all check flag (addendum 1) and the aligned_best "sensitive to alignment" flag (addendum 4); no --e4-out at
  aggregation -> INCOMPLETE.

Resources and usage (python -B; -O refused; spawned single-threaded workers, ~0.85 GB each; one job = one (house, noise seed), 3 tiers):
  python -B g4_registration_msg.py --out DIR [--panels val,fresh] [--seeds 1,2] [--workers 4] [--houses N] [--conds cam_nf,cam,harsh]
         --e1-jobs DIR --e4-out DIR [--smoke-asserts]
  python -B g4_registration_msg.py --out DIR --aggregate-only --e1-jobs DIR --e4-out DIR
Output: DIR/jobs/*.json, DIR/summary.json, DIR/run.json.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True

import argparse, collections, gc, glob, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_registration as R  # noqa: E402   (frozen G4-E4 code: imported, never modified)

G = R.G; W = R.W; I3 = R.I3; _check = R._check
E4_SHA256 = 'e9574c7eff5e1400b0bd901e0befc3abb65520d1612e8efbeade80307b5899fe'
JOB_FORMAT = 'E4b-1'
SWEEPS = 2
SIDE_PLANE = 'max'
BYTES_PER_SURFEL = 6
S0_RNET = 6066.0                                                         # D8-36 fact 2: S0 one fusion, bytes / house
CONDS_DEFAULT = R.CONDS_DEFAULT
COMPARE_ARMS = ('noreg', 'truepose') + R.REPORT_ARMS                     # rows that must equal E4's (same code, same draws)


# ================================================================ the message pose layer (no truth)
def warp_plane(w, M, cell):
    """an agent's w plane (its believed frame) moved by the SE(2) M into the common frame on the same grid (bilinear, outside = 0)"""
    if np.array_equal(M, I3):
        return w
    cx = float(cell[0]); RT = M[:2, :2].T; t = M[:2, 2]
    off = RT @ (0.5 * cx - t) / cx - 0.5                                 # input index = R^T output index + off
    return ndimage.affine_transform(w, RT, offset=off, order=1, mode='constant', cval=0.)


def side_message(S_list, w_list, C, members, cell):
    """S(X): the union of the members' surfels moved by their current corrections, plus the side plane (pixel-wise max of the moved w
    planes) for the correlative search"""
    plane = np.max([warp_plane(w_list[a], C[a], cell) for a in members], 0)
    return dict(w=plane, f=None, S=_side_surfels(S_list, C, members))


def tree_sides(F):
    """per tree edge (parent i, child j): (A_e, B_e) = (component minus j's subtree, j's subtree); edges in BFS order of the child"""
    parent, order = R.tree_order(F)
    children = collections.defaultdict(list)
    for a in order:
        if parent[a] is not None:
            children[parent[a]].append(a)

    def subtree(a):
        out, st = [], [a]
        while st:
            u = st.pop(); out.append(u); st.extend(children[u])
        return sorted(out)
    sides = []
    for a in order:
        if parent[a] is not None:
            B = subtree(a); comp = F['comps'][F['comp_of'][a]]
            sides.append(((parent[a], a), sorted(set(comp) - set(B)), B))
    return sides


def message_energy(S_list, C, sides, prm):
    """E_msg(C) = sum_e E_edge(S(A_e), S(B_e)) (fixed edge order)"""
    total, per = 0., []
    for _, A, B in sides:
        SA = _side_surfels(S_list, C, A)
        SB = _side_surfels(S_list, C, B)
        e = R.edge_cost(SB, SA, I3, prm)
        per.append(e); total += e
    return float(total), per


def _side_surfels(S_list, C, members):
    ms, ns, ws, os_ = [], [], [], []
    for a in members:
        S = S_list[a]
        if len(S['m']):
            ms.append(R.apply2(C[a], S['m'])); ns.append(S['n'] @ C[a][:2, :2].T); ws.append(S['w']); os_.append(S['ori'])
    S = dict(m=np.concatenate(ms) if ms else np.zeros((0, 2)), n=np.concatenate(ns) if ns else np.zeros((0, 2)),
             w=np.concatenate(ws) if ws else np.zeros(0), ori=np.concatenate(os_) if os_ else np.zeros(0, bool))
    S['tree'] = cKDTree(S['m']) if len(S['m']) else None
    return S


def register_messages(maps, F, cell, prm, sweeps=SWEEPS):
    """THE E4b ESTIMATOR. Inputs: the agents' own V3 maps in their believed public frames, the public radio forest, public grid constants
    and E4's frozen matcher settings -- no truth. Returns the corrections (prior gauge), the E_msg trajectory (asserted non-increasing)
    and per-edge bookkeeping."""
    n = len(maps)
    ev = [R.evidence(R.planes_logodds(L, cell, prm), cell, prm) for L in maps]
    S_list = [e['S'] for e in ev]; w_list = [e['w'] for e in ev]
    sides = tree_sides(F)
    C = {a: I3.copy() for a in range(n)}
    E, per = message_energy(S_list, C, sides, prm)
    traj = [E]; log = []
    stats = {k: dict(kept=0, tried=0, rejected_global=0, no_step=0, local=[]) for k, _, _ in sides}
    for sweep in range(sweeps):
        for k, A, B in sides:
            evA = side_message(S_list, w_list, C, A, cell); evB = side_message(S_list, w_list, C, B, cell)
            res = R.register_pair(evA, evB, cell, prm)                  # Delta: B side -> A side (both in the common frame)
            st = stats[k]; st['tried'] += 1
            st['local'].append(dict(sweep=sweep, steps=len(res['traj']) - 1, coarse_accepted=res['coarse_accepted'], icp_stop=res['icp_stop'],
                                    final_gate=res['final_gate'], delta=R.params(res['M'])))
            if np.array_equal(res['M'], I3):
                st['no_step'] += 1; log.append(dict(sweep=sweep, edge=list(k), kept=False, reason='no local step')); continue
            Cn = dict(C)
            for b in B:
                Cn[b] = res['M'] @ C[b]
            En, pern = message_energy(S_list, Cn, sides, prm)
            if En < E:
                C, E, per = Cn, En, pern; traj.append(E); st['kept'] += 1
                log.append(dict(sweep=sweep, edge=list(k), kept=True, E_msg=E))
            else:
                Cc = dict(C); Cc[k[1]] = res['M'] @ C[k[1]]                # review M1 (report only): move the child alone
                E_child = message_energy(S_list, Cc, sides, prm)[0]
                st['rejected_global'] += 1
                log.append(dict(sweep=sweep, edge=list(k), kept=False, reason='E_msg not lower', E_try=En, E_child_only=E_child,
                                child_only_would_lower=bool(E_child < E)))
    checks = R.assert_monotone(traj, 'E_msg')
    pivots = []
    for cm in F['comps']:                                               # = E4's register_forest pivots
        P = np.concatenate([ev[a]['S']['m'] for a in cm]) if any(len(ev[a]['S']['m']) for a in cm) else np.zeros((1, 2))
        pivots.append(P.mean(0))
    K = R.gauge_fix(F, C, 'prior', pivots)
    return dict(C=[C[a] for a in range(n)], K=[K[a] for a in range(n)], E_msg=traj, E_edges_final=per, monotone_checks=checks, S_list=S_list,
                sides=sides, stats=stats, log=log, pivots=pivots, surf_m=[S['m'] for S in S_list],
                n_surfels=[int(len(S['m'])) for S in S_list], keys=[k for k, _, _ in sides])


# ================================================================ one (house, noise seed) job
def fingerprint(opts):
    return json.loads(json.dumps({k: opts[k] for k in sorted(opts) if k not in ('out', 'check_e1_maps')}))


def run_job(job):
    panel, rooms, index, seed, opts = job
    _check(__debug__, 'python -O strips the pre-registered assertions of the frozen G4-E1 code: refused')
    _check(R.file_sha256(os.path.abspath(__file__)) == opts['script_sha256'], 'g4_registration_msg.py changed since the run started')
    _check(R.file_sha256(R.__file__) == opts['g4_registration_sha256'] == E4_SHA256, 'g4_registration.py is not the frozen E4 code')
    _check(R.file_sha256(G.__file__) == opts['g4_noise_panels_sha256'], 'g4_noise_panels.py changed since the run started')
    prm = opts['match']; name = f'r{rooms}_i{index}'; fp = fingerprint(opts)
    t_start = time.time(); timing = {}
    try:
        fx = W.make_fixture('VALIDATION', index, rooms)
    except ValueError as e:
        return dict(panel=panel, house=name, rooms=rooms, index=index, seed=seed, skipped=repr(e)[:200], fingerprint=fp)
    timing['fixture'] = time.time() - t_start
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    for a in range(n):
        _check(set(np.nonzero(reached8[a])[0].tolist()) == set(F['comps'][F['comp_of'][a]]), 'reached8 != component')
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    t0 = time.time()
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    timing['trace'] = time.time() - t0
    rstats = G.RelayStats()
    out = dict(panel=panel, house=name, rooms=rooms, index=index, seed=seed, agents=n, chart_m=float(fx['chart_m']), cell=cell.tolist(),
               components=F['comps'], tree_edges=[list(e) for e in F['und']], world_id=fx['world_id'], private_sha256=W.array_hash(private),
               fingerprint=fp, conds={})
    k0cache, armcache = {}, {}
    for cname in opts['conds']:
        c = G.CONDITIONS[cname]; tc = time.time(); has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0; ck = G._ck(c)
        scans_k0, scans_pub, L_k0, L_pub, MAs, frames = [], [], [], [], [], []
        for a in range(n):                                              # = E4's run_job (same draws, same calls)
            frame = None
            if has_inter:
                z = G.frame_draws(seed, rooms, index, a)
                frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
                frames.append([float(z[0] * c['inter_t']), float(z[1] * c['inter_t']), float(z[2] * c['inter_yaw'])])
            MAs.append(R.frame_matrix(frame))
            scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
            pub = [G.to_public(s, frame) for s in scans]
            if (ck, a) not in k0cache:
                k0cache[(ck, a)] = G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3']
            L_k0.append(k0cache[(ck, a)])
            L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3'] if has_inter else L_k0[a])
            scans_k0.append(scans); scans_pub.append(pub)
        timing[f'{cname}_integrate'] = time.time() - tc
        rec = dict(inter_frames_true=frames, k0_sha256=W.array_hash(np.stack(L_k0)), pub_sha256=W.array_hash(np.stack(L_pub)))
        # ---- the message pose layer (sees L_pub only)
        tr = time.time()
        reg = register_messages(L_pub, F, cell, prm)
        timing[f'{cname}_register'] = time.time() - tr
        planes_k0 = [R.planes_logodds(L, cell, prm) for L in L_k0]
        ev_k0 = [R.evidence(p, cell, prm) for p in planes_k0]
        obs = {(i, j): R.observable_pair(ev_k0[i], ev_k0[j], prm) for (i, j) in reg['keys']}
        edges = []
        for (k, A, B) in reg['sides']:
            i, j = k; st = reg['stats'][k]
            Dt = MAs[i] @ R.inv(MAs[j]); Dest = R.inv(reg['C'][i]) @ reg['C'][j]; P = reg['surf_m'][j]
            cpiv = P.mean(0) if len(P) else np.zeros(2)
            eb, ea = R.pose_err(I3, Dt, P, cpiv), R.pose_err(Dest, Dt, P, cpiv); corrected = st['kept'] > 0
            failed = (not corrected or (ea['disp_rms_cm'] or 0.) > (eb['disp_rms_cm'] or 0.)) if has_inter else None
            nA = int(sum(reg['n_surfels'][a] for a in A)); nB = int(sum(reg['n_surfels'][a] for a in B))
            edges.append(dict(i=int(i), j=int(j), side_A=[int(a) for a in A], side_B=[int(b) for b in B], true=R.params(Dt), est=R.params(Dest),
                              err_before=eb, err_after=ea, kept_updates=st['kept'], tried=st['tried'], rejected_global=st['rejected_global'],
                              no_local_step=st['no_step'], local=st['local'], coarse_accepted=any(x['coarse_accepted'] for x in st['local']),
                              corrected=corrected, overlap_true=obs[k][1], failed=failed, n_surfels_A=nA, n_surfels_B=nB,
                              msg_bytes_up=BYTES_PER_SURFEL * nB, msg_bytes_down=BYTES_PER_SURFEL * nA))
        e4reg = R.register_forest(L_pub, F, cell, prm, 'prior')          # review M1 (report only): E4's estimator, no truth
        E_e4 = message_energy(reg['S_list'], {a: e4reg['C'][a] for a in range(n)}, reg['sides'], prm)[0]
        E_true = message_energy(reg['S_list'], {a: R.inv(MAs[a]) for a in range(n)}, reg['sides'], prm)[0]
        rej = [x for x in reg['log'] if x.get('reason') == 'E_msg not lower']
        opt_diag = dict(E_msg_E4b_final=reg['E_msg'][-1], E_msg_at_E4_solution=E_e4, E_msg_at_true_poses=E_true, E_msg_initial=reg['E_msg'][0],
                        E4b_final_above_E4_solution=bool(reg['E_msg'][-1] > E_e4), true_poses_below_E4b_final=bool(E_true < reg['E_msg'][-1]),
                        global_rejections=len(rej), child_only_would_lower=int(sum(x['child_only_would_lower'] for x in rej)))
        rec['registration'] = dict(method='messages', gauge='prior', sweeps=SWEEPS, side_plane=SIDE_PLANE, edges=edges, optimizer_diag=opt_diag,
                                   E_pose=reg['E_msg'], E_msg=reg['E_msg'], E_pose_monotone=True, monotone_checks=reg['monotone_checks'],
                                   log=reg['log'], K=[R.params(K) for K in reg['K']], C_root=[R.params(C) for C in reg['C']],
                                   n_surfels=reg['n_surfels'], E_edges_final=reg['E_edges_final'],
                                   agent_err_before=R.agent_errors([I3] * n, MAs, F, reg['pivots']),
                                   agent_err_after=R.agent_errors(reg['K'], MAs, F, reg['pivots']),
                                   msg_bytes_house=int(sum(e['msg_bytes_up'] + e['msg_bytes_down'] for e in edges)))
        # ---- arms (= E4's code path; reg = the message corrections)
        ta = time.time()
        L_reg = [L_pub[a] if np.array_equal(reg['K'][a], I3) else R._integrate(shape, cell, ds, scans_pub[a], reg['K'][a]) for a in range(n)]
        arms = {}
        arms['noreg'] = (armcache[('k0', ck)] if (not has_inter and ('k0', ck) in armcache)
                         else R.eval_arm(L_pub, scans_pub, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats))
        arms['reg'] = R.eval_arm(L_reg, scans_pub, reg['K'], F, R8T, T8, shape, cell, ds, prm, rstats)
        if ('k0', ck) not in armcache:
            armcache[('k0', ck)] = arms['noreg'] if not has_inter else R.eval_arm(L_k0, scans_k0, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats)
        arms['truepose'] = armcache[('k0', ck)]
        timing[f'{cname}_arms'] = time.time() - ta
        if cname == R.JUDGED_COND:                                      # report-only truth arms: E4's functions, E4's pivots
            tq = time.time()
            islands, n_cov, n_pairs = R.observability_islands(F, planes_k0, cell, prm)
            rec['observability'] = dict(islands=[list(x) for x in islands], n_islands=len(islands), covisible_pairs=n_cov, pairs=n_pairs,
                                        tree_edges_observable=int(sum(o[0] for o in obs.values())), tree_edges=len(obs))
            keys = sorted(reg['keys'])
            Ks = dict(oracle_all=R.compose(F, R.truth_deltas(keys, MAs), 'prior', reg['pivots'])[1],
                      ceiling_pairwise=R.compose(F, R.truth_deltas(keys, MAs, {k: o[0] for k, o in obs.items()}), 'prior', reg['pivots'])[1],
                      ceiling_islands=R.islands_K(F, islands, MAs, 'prior', reg['pivots'], reg['surf_m']))
            done = [('noreg', [I3] * n)]                                # E4 also offered 'reg' for reuse: never equal to a truth arm in practice
            for nm in R.REPORT_ARMS:
                K = Ks[nm]
                prev = next((p for p, Kp in done if all(np.allclose(x, y, rtol=0., atol=1e-12) for x, y in zip(K, Kp))), None)
                if prev is not None:
                    res_arm = dict(arms[prev], reused_from=prev)
                else:
                    L_arm = [L_pub[a] if np.array_equal(K[a], I3) else R._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
                    res_arm = R.eval_arm(L_arm, scans_pub, K, F, R8T, T8, shape, cell, ds, prm, rstats)
                    del L_arm
                arms[nm] = dict(res_arm, K=[R.params(x) for x in K])
                done.append((nm, K))
            timing[f'{cname}_report_arms'] = time.time() - tq
        rec['arms'] = arms
        timing[f'{cname}_total'] = time.time() - tc
        out['conds'][cname] = rec
        del scans_k0, scans_pub, L_k0, L_pub, L_reg
        later = {G._ck(G.CONDITIONS[c2]) for c2 in opts['conds'][opts['conds'].index(cname) + 1:]}
        for key in [k for k in k0cache if k[0] not in later]:
            del k0cache[key]
        gc.collect()
    if 'cam' in out['conds'] and 'cam_nf' in out['conds']:
        h = (out['conds']['cam']['k0_sha256'], out['conds']['cam_nf']['k0_sha256'], out['conds']['cam_nf']['pub_sha256'])
        _check(h[0] == h[1] == h[2], f'cam truepose != cam_nf map (hashes {h})')
        out['cam_truepose_equals_cam_nf_map'] = True
    out['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    timing['total'] = time.time() - t_start
    out['timing'] = timing; out['peak_mb'] = G._peak_mb()
    return out


def _job_path(out_dir, panel, rooms, index, seed):
    return os.path.join(out_dir, 'jobs', f'{panel}_r{rooms}_i{index}_s{seed}.json')


def _worker_init(json_dir, index_file):
    sys.dont_write_bytecode = True
    G.setup_env(json_dir, index_file)


def _run_and_save(job):
    panel, rooms, index, seed, opts = job
    try:
        res = run_job(job)
    except Exception:                                                   # noqa: BLE001 (recorded; aggregation -> INCOMPLETE)
        res = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, failed=traceback.format_exc()[-2048:],
                   fingerprint=fingerprint(opts))
    path = _job_path(opts['out'], panel, rooms, index, seed); tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return path, res.get('timing', {}).get('total'), res.get('peak_mb'), res.get('skipped'), res.get('failed')


# ================================================================ aggregation and verdict (D8-40)
def e4_compare_job(job, e4_dir, _cache={}):
    """per tier: do this job's control and truth rows (as-is and aligned of noreg / truepose / oracle_all / ceiling_pairwise /
    ceiling_islands) equal E4's job file? equal / unequal / missing / no_e4_dir"""
    out = {}
    path = os.path.join(e4_dir, 'jobs', f"{job['panel']}_r{job['rooms']}_i{job['index']}_s{job['seed']}.json") if e4_dir else None
    if path and path not in _cache and os.path.exists(path):
        _cache[path] = json.load(open(path))
    j = _cache.get(path) if path else None
    norm = lambda x: json.loads(json.dumps(x))  # noqa: E731
    for c, rec in job['conds'].items():
        if not e4_dir:
            out[c] = 'no_e4_dir'; continue
        if j is None or c not in j.get('conds', {}):
            out[c] = 'missing'; continue
        A, B = rec['arms'], j['conds'][c]['arms']
        eq = all(norm(A[a][row]) == B[a][row] for a in COMPARE_ARMS if a in A or a in B for row in ('asis', 'aligned')
                 if a in A and a in B) and set(a for a in COMPARE_ARMS if a in A) == set(a for a in COMPARE_ARMS if a in B)
        out[c] = 'equal' if eq else 'unequal'
    return out


def _panel_seeds(jobs, panel, e1_dir):
    P = {}
    for seed in sorted({j['seed'] for j in jobs if j['panel'] == panel}):
        J_all = [j for j in jobs if j['panel'] == panel and j['seed'] == seed]
        J = [j for j in J_all if 'skipped' not in j and 'failed' not in j]
        D, _ = R.summarise_seed(J, {j['house']: R.e1_compare_job(j, e1_dir) for j in J})
        D.update(houses=sorted(j['house'] for j in J), skipped=sorted(j['house'] for j in J_all if 'skipped' in j),
                 failed=sorted(j['house'] for j in J_all if 'failed' in j), n_houses=len(J))
        D['harsh_readouts'] = {row: R.readouts(D['conds'], 'harsh', 'none', row) for row in R.ROWS} if 'harsh' in D['conds'] else None
        P[str(seed)] = D
    return P


def _edge_frac_uncorrected(jobs, panel, seed, cond='cam'):
    """E4 side: fraction of tree edges left uncorrected (all E4 non-corrections come from the gate or from no descent)"""
    E = [e for j in jobs if j['panel'] == panel and j['seed'] == seed and 'conds' in j and cond in j['conds']
         for e in j['conds'][cond]['registration']['edges']]
    return (sum(1 for e in E if not e['corrected']) / len(E)) if E else None, len(E)


def _edge_fracs_e4b(jobs, panel, seed, cond='cam'):
    """review M2: E4b edges split by why they stayed uncorrected -- gate-refused (no local step in any sweep), globally rejected (a local
    step existed but never lowered E_msg), any reason"""
    E = [e for j in jobs if j['panel'] == panel and j['seed'] == seed and 'conds' in j and cond in j['conds']
         for e in j['conds'][cond]['registration']['edges']]
    if not E:
        return dict(n_edges=0, gate_refused=None, globally_rejected=None, uncorrected_any=None)
    g = sum(1 for e in E if e['kept_updates'] == 0 and e['rejected_global'] == 0)
    r = sum(1 for e in E if e['kept_updates'] == 0 and e['rejected_global'] > 0)
    return dict(n_edges=len(E), gate_refused=g / len(E), globally_rejected=r / len(E), uncorrected_any=(g + r) / len(E))


def optimizer_summary(jobs, panel, seed, cond):
    """review M1 (report only): per (panel, seed, tier) -- jobs whose E4b final E_msg is above E_msg at E4's solution (and the sum of the
    excess), global rejections where moving the child alone would have lowered E_msg, jobs whose true poses have a lower E_msg"""
    X = [j['conds'][cond]['registration']['optimizer_diag'] for j in jobs if j['panel'] == panel and j['seed'] == seed and 'conds' in j
         and cond in j['conds']]
    if not X:
        return None
    above = [x for x in X if x['E4b_final_above_E4_solution']]
    return dict(jobs=len(X), E4b_final_above_E4_solution=len(above),
                frac_E4b_final_le_E4_solution=1. - len(above) / len(X),
                excess_sum=float(sum(x['E_msg_E4b_final'] - x['E_msg_at_E4_solution'] for x in above)),
                global_rejections=int(sum(x['global_rejections'] for x in X)),
                rejections_child_only_would_lower=int(sum(x['child_only_would_lower'] for x in X)),
                true_poses_below_E4b_final=int(sum(x['true_poses_below_E4b_final'] for x in X)))


def _agent_err_median(jobs, panel, seed, cond='cam'):
    v = [x['disp_cm'] for j in jobs if j['panel'] == panel and j['seed'] == seed and 'conds' in j and cond in j['conds']
         for x in j['conds'][cond]['registration']['agent_err_after']['per_agent']]
    return float(np.median(v)) if v else None


OPT_OK_FRAC = 0.90                                                       # D8-45 addendum: optimizer-not-the-bottleneck threshold


def mechanism(jobs, e4_jobs):
    """D8-40 mechanism readouts (report only), per VAL seed, E4b vs E4 on the SAME (house, seed) jobs: edges left uncorrected (E4b split
    into gate-refused / globally rejected, review M2; the pre-registered 'at least halved' uses the GATE-REFUSED fraction), median
    per-agent position error after removing the component mean, recovery (cam, aligned, pooled from house sums) and its difference.
    Reading (D8-45 addendum): 'hypothesis rejected' needs E4b - E4 < +.15 in both seeds AND, in each VAL seed, >= 90 % of the cam jobs with
    E4b final E_msg <= E_msg at E4's solution; otherwise 'limited by the D8-40 optimizer's move set: this batch cannot test it'."""
    out = {}
    for seed in (1, 2):
        s = str(seed)
        Jb = [j for j in jobs if j['panel'] == 'val' and j['seed'] == seed and 'conds' in j and 'cam' in j['conds']]
        houses = {j['house'] for j in Jb}
        Ja = [j for j in e4_jobs if j['panel'] == 'val' and j['seed'] == seed and 'conds' in j and 'cam' in j['conds'] and j['house'] in houses]
        fr = _edge_fracs_e4b(Jb, 'val', seed); fa, na = _edge_frac_uncorrected(Ja, 'val', seed)
        mb, ma = _agent_err_median(Jb, 'val', seed), _agent_err_median(Ja, 'val', seed)
        rb = R.recovery_distribution(Jb, 'cam', 'aligned')['pooled'] if Jb else None
        ra = R.recovery_distribution(Ja, 'cam', 'aligned')['pooled'] if len(Ja) == len(Jb) and Ja else None
        opt = optimizer_summary(Jb, 'val', seed, 'cam')
        out[s] = dict(houses=len(Jb), e4_houses_matched=len(Ja), n_edges=(fr['n_edges'], na),
                      frac_gate_refused_edges_E4b=fr['gate_refused'], frac_globally_rejected_edges_E4b=fr['globally_rejected'],
                      frac_uncorrected_any_E4b=fr['uncorrected_any'], frac_uncorrected_edges_E4=fa,
                      gate_refused_at_least_halved=(fr['gate_refused'] is not None and fa is not None and fr['gate_refused'] <= 0.5 * fa),
                      agent_pos_err_median_cm_E4b=mb, agent_pos_err_median_cm_E4=ma, agent_err_lower=(mb is not None and ma is not None and mb < ma),
                      recovery_E4b=rb, recovery_E4=ra, recovery_diff=(rb - ra) if (rb is not None and ra is not None) else None,
                      optimizer=opt, optimizer_ok=(opt is not None and opt['frac_E4b_final_le_E4_solution'] >= OPT_OK_FRAC))
    d = [out[s]['recovery_diff'] for s in out]
    if any(x is None for x in d):
        hyp = None
    elif not all(x < 0.15 for x in d):
        hyp = 'not rejected (E4b - E4 >= +.15 in a seed)'
    elif all(out[s]['optimizer_ok'] for s in out):
        hyp = 'rejected: E4b - E4 < +.15 in both seeds with the optimizer not the bottleneck -> the bottleneck is not the message'
    else:
        hyp = ("limited by the D8-40 optimizer's move set (rigid subtree moves + global rejection): this batch cannot test the hypothesis "
               "'the bottleneck is the message'")
    return dict(per_seed=out, message_bottleneck_hypothesis=hyp, optimizer_ok_threshold=OPT_OK_FRAC)


def message_bytes(jobs, panel, cond='cam'):
    per_msg = [b for j in jobs if j['panel'] == panel and 'conds' in j and cond in j['conds'] for e in j['conds'][cond]['registration']['edges']
               for b in (e['msg_bytes_up'], e['msg_bytes_down'])]
    per_house = [j['conds'][cond]['registration']['msg_bytes_house'] for j in jobs if j['panel'] == panel and 'conds' in j and cond in j['conds']]
    med_h = float(np.median(per_house)) if per_house else None
    return dict(per_message_median=float(np.median(per_msg)) if per_msg else None, per_message_mean=float(np.mean(per_msg)) if per_msg else None,
                per_house_median=med_h, per_house_over_S0=(med_h / S0_RNET) if med_h is not None else None, bytes_per_surfel=BYTES_PER_SURFEL)


def _combine(val_cat, fresh_cat):
    if val_cat == 'LEVER' and fresh_cat == 'LEVER':
        return 'LEVER'
    if val_cat == 'INSUFFICIENT':
        return 'INSUFFICIENT'
    return 'AMBIGUOUS'


def verdict_e4b(Vv, Vf, extra=()):
    """D8-40: LEVER iff VAL and fresh both meet the D8-39 lever conditions; INSUFFICIENT iff VAL recovery < .50 in both seeds; else
    AMBIGUOUS. Flags (D8-39 / D8-40 addendum 1 and 4, review M3): 'alignment pipeline check failed' if either panel's oracle_all check
    fails; 'sensitive to alignment' if the same rule read on the aligned_best rows gives another category."""
    reasons = list(extra)
    if not Vv or not Vv.get('complete'):
        reasons.append('VAL: ' + ('; '.join((Vv or {}).get('incomplete_reasons', ['missing'])) or 'incomplete'))
    if not Vf or not Vf.get('complete'):
        reasons.append('fresh: ' + ('; '.join((Vf or {}).get('incomplete_reasons', ['missing'])) or 'incomplete'))
    flags = []
    oracle = {p: (V or {}).get('oracle_check_ok') for p, V in (('val', Vv), ('fresh', Vf))}
    if any(v is False for v in oracle.values()):
        flags.append('[alignment pipeline check failed: oracle_all recovery deviates from 1 by more than .10 on ' +
                     ', '.join(p for p, v in oracle.items() if v is False) + ']')
    if reasons:
        return dict(category=None, verdict='INCOMPLETE: ' + ' | '.join(reasons) + ''.join(' ' + f for f in flags), reasons=reasons,
                    oracle_check_ok=oracle, category_aligned_best=None)
    cat = _combine(Vv['category'], Vf['category'])
    v = {'LEVER': 'LEVER (杠杆): message registration meets the D8-39 lever conditions on VAL and on fresh, both seeds',
         'INSUFFICIENT': 'INSUFFICIENT (不够): VAL recovery < 50 % in both seeds', 'AMBIGUOUS': 'AMBIGUOUS'}[cat]
    best = (_combine(Vv.get('category_aligned_best'), Vf.get('category_aligned_best'))
            if Vv.get('category_aligned_best') is not None and Vf.get('category_aligned_best') is not None else None)
    if best is not None and best != cat:
        flags.append(f'[sensitive to alignment: the aligned_best rows read {best}]')
    return dict(category=cat, verdict=v + ''.join(' ' + f for f in flags), reasons=[], val_category=Vv['category'], fresh_category=Vf['category'],
                category_aligned_best=best, oracle_check_ok=oracle)


def aggregate(out_dir, e1_jobs=None, e4_out=None):
    paths = sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))
    if not paths:
        raise SystemExit(f'no job files in {out_dir}/jobs')
    jobs = [json.load(open(p)) for p in paths]
    fps = {json.dumps(j.get('fingerprint'), sort_keys=True) for j in jobs}
    if len(fps) != 1:
        raise SystemExit(f'refusing to aggregate {out_dir}: {len(fps)} different job fingerprints')
    fp = jobs[0]['fingerprint']
    if fp is None or fp.get('job_format') != JOB_FORMAT:
        raise SystemExit(f'refusing to aggregate {out_dir}: job format != {JOB_FORMAT}')
    e1_dir = os.path.abspath(e1_jobs) if e1_jobs else None; e4_dir = os.path.abspath(e4_out) if e4_out else None
    last = lambda t: (t.strip().splitlines() or [''])[-1][:300]  # noqa: E731
    S = dict(fingerprint=fp, e1_jobs=e1_dir, e4_out=e4_dir, panels={},
             skipped=[dict(panel=j['panel'], house=j['house'], seed=j['seed'], reason=j['skipped']) for j in jobs if 'skipped' in j],
             failed=[dict(panel=j['panel'], house=j['house'], seed=j['seed'], error=last(j['failed'])) for j in jobs if 'failed' in j])
    e4_cmp_reasons = []
    for panel in sorted({j['panel'] for j in jobs}):
        expected = frozenset(f'r{r}_i{i}' for r, i in G.panel_specs(panel))
        seeds = _panel_seeds(jobs, panel, e1_dir)
        cmp = collections.Counter(); bad = []
        for j in jobs:
            if j['panel'] == panel and 'conds' in j:
                for c, st in e4_compare_job(j, e4_dir).items():
                    cmp[st] += 1
                    if st != 'equal':
                        bad.append(f"{j['house']} s{j['seed']} {c}: {st}")
        if e4_dir and bad:
            e4_cmp_reasons.append(f'{panel}: control / truth rows != E4 ({len(bad)}: {bad[:3]})')
        if not e4_dir:
            e4_cmp_reasons.append(f'{panel}: no comparison with G4-E4 (--e4-out)')
        extra = [f"job {panel} {j['house']} s{j['seed']} failed: {last(j['failed'])}" for j in jobs if j['panel'] == panel and 'failed' in j]
        V = R.verdict(seeds, expected, judged=True, extra_reasons=extra)  # D8-39 readouts incl. the G4-E1 control check, on BOTH panels
        S['panels'][panel] = dict(seeds=seeds, verdict=V, e4_compare=dict(cmp), message_bytes=message_bytes(jobs, panel),
                                  optimizer={s_: {c: optimizer_summary(jobs, panel, int(s_), c) for c in seeds[s_]['conds']} for s_ in seeds})
    Vv = S['panels'].get('val', {}).get('verdict'); Vf = S['panels'].get('fresh', {}).get('verdict')
    D = verdict_e4b(Vv, Vf, e4_cmp_reasons)
    e4_jobs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(e4_dir, 'jobs', '*.json')))] if e4_dir else []
    S['decision'] = dict(D, judged_cond=R.JUDGED_COND, judged_row=R.JUDGED_ROW, panels_judged=['val', 'fresh'],
                         mechanism=mechanism(jobs, e4_jobs) if e4_dir else None,
                         message_bytes=S['panels'].get('val', {}).get('message_bytes'))
    return S


# ================================================================ main
def main(argv=None):
    if not __debug__:
        raise SystemExit('refusing to run under python -O: the pre-registered checks of the frozen G4-E1 code are asserts')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--panels', default='val,fresh')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--houses', type=int, default=0)
    ap.add_argument('--conds', default=','.join(CONDS_DEFAULT))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--e1-jobs', default='', help='G4-E1 jobs dir (control check at aggregation; missing -> INCOMPLETE)')
    ap.add_argument('--e4-out', default='', help='G4-E4 output dir (rows must equal E4; mechanism readouts)')
    ap.add_argument('--houseexpo-json-dir', default=G.HOUSEEXPO_JSON_DIR)
    ap.add_argument('--houseexpo-index', default=G.HOUSEEXPO_INDEX)
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--smoke-asserts', action='store_true')
    args = ap.parse_args(argv)
    G.setup_env(args.houseexpo_json_dir, args.houseexpo_index)
    os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
    conds = [c for c in args.conds.split(',') if c]
    for c in conds:
        if c not in G.CONDITIONS:
            raise SystemExit(f'unknown tier {c}')
    conds = [c for c in G.COND_ORDER if c in conds]
    opts = dict(out=args.out, conds=conds, match=dict(R.MATCH), sweeps=SWEEPS, side_plane=SIDE_PLANE, gauge='prior', job_format=JOB_FORMAT,
                governing_integrator='V3', script_sha256=R.file_sha256(os.path.abspath(__file__)), g4_registration_sha256=R.file_sha256(R.__file__),
                g4_noise_panels_sha256=R.file_sha256(G.__file__), data_module_sha256=R.file_sha256(W.__file__),
                houseexpo_index_sha256=R.file_sha256(args.houseexpo_index))
    if opts['g4_registration_sha256'] != E4_SHA256:
        raise SystemExit('g4_registration.py is not the frozen E4 code (sha e9574c7e...)')
    fp = fingerprint(opts); t0 = time.time()
    if not args.aggregate_only:
        jobs = []
        for panel in args.panels.split(','):
            specs = G.panel_specs(panel)[:args.houses] if args.houses else G.panel_specs(panel)
            for seed in [int(s) for s in args.seeds.split(',')]:
                for rooms, index in specs:
                    path = _job_path(args.out, panel, rooms, index, seed)
                    if os.path.exists(path):
                        old = json.load(open(path))
                        if old.get('fingerprint') != fp:
                            raise SystemExit(f'{path} was written with another script version / options; use a fresh --out')
                        if 'failed' not in old:
                            continue
                    jobs.append((panel, rooms, index, seed, opts))
        print(f'{len(jobs)} jobs, {args.workers} workers', flush=True)
        if jobs:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.houseexpo_json_dir, args.houseexpo_index), maxtasksperchild=1) as pool:
                for i, (path, sec, mb, sk, fl) in enumerate(pool.imap_unordered(_run_and_save, jobs), 1):
                    print(f'[{i}/{len(jobs)}] {os.path.basename(path)} {sec if sec is None else round(sec)} s peak {mb if mb is None else round(mb)} MB'
                          + (f' SKIPPED {sk}' if sk else '') + (' FAILED: ' + fl.strip().splitlines()[-1][:200] if fl else ''), f'{time.time() - t0:.0f}s', flush=True)
    S = aggregate(args.out, args.e1_jobs or None, args.e4_out or None)
    S['args'] = vars(args)
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1)
    with open(os.path.join(args.out, 'aggregate.json' if args.aggregate_only else 'run.json'), 'w') as fh:
        json.dump(dict(script=os.path.basename(__file__), fingerprint=fp, args=vars(args), seconds=time.time() - t0), fh, indent=1)
    f2 = lambda v, k=3: 'NA' if v is None else f'{v:.{k}f}'  # noqa: E731
    print('E4b verdict:', S['decision']['verdict'], flush=True)
    for panel, P in S['panels'].items():
        print(panel, '(D8-39 readout)', P['verdict']['verdict'][:160], '| rows vs E4', P['e4_compare'], flush=True)
        for s, D in P['seeds'].items():
            for c, d in D['conds'].items():
                A = d['arms']; e = d['edges']
                line = ' | '.join(f"{arm} " + '/'.join(f"{A[arm][row]['furn_f1']:.4f}" for row in R.ROWS) for arm in R.ARMS + R.REPORT_ARMS if arm in A)
                print(f'  seed {s} {c:7s} asis/aligned: {line} | edge rms cm {f2(e["err_before"]["disp_rms_cm"]["p50"], 2)} -> '
                      f'{f2(e["err_after"]["disp_rms_cm"]["p50"], 2)} (median of {e["n_edges"]}; corrected {e["corrected"]}, no overlap at true pose '
                      f'{e["no_overlap_true"]}) | E_msg monotone {d["E_pose_monotone_all"]} ({d["monotone_checks"]} checks) | G4-E1 '
                      f'{(d.get("e1_compare") or {}).get("counts")}', flush=True)
            x = P['verdict']['per_seed'].get(s, {})
            if x and 'missing' not in x:
                r = x['readouts']['aligned']
                print(f"  seed {s} cam aligned: recovery {f2(r['recovery'])} (noreg {r['F_noreg']:.4f} reg {r['F_reg']:.4f} truepose {r['F_truepose']:.4f}) "
                      f"oracle {f2(r.get('recovery_oracle_all'))} pairwise {f2(r.get('recovery_ceiling_pairwise'))} islands {f2(r.get('recovery_ceiling_islands'))} "
                      f"| cam_nf reg-noreg {r['nf_delta']:+.4f} free {r['free_delta']:+.4f} fpr {r['fpr_delta']:+.5f}", flush=True)
        print('  message bytes', P['message_bytes'], flush=True)
    print('decision:', json.dumps({k: v for k, v in S['decision'].items() if k != 'verdict'}), flush=True)
    if args.smoke_asserts:
        for panel, P in S['panels'].items():
            if args.e4_out:
                _check(set(P['e4_compare']) == {'equal'}, f'{panel}: rows differ from E4 {P["e4_compare"]}')
            for s, D in P['seeds'].items():
                _check(not D['failed'], f'failed jobs {D["failed"]}')
                for c, d in D['conds'].items():
                    _check(d['E_pose_monotone_all'] and d['monotone_checks'] > 0, f'{c}: E_msg monotone assertion did not run')
                    if args.e1_jobs:
                        _check(d['e1_compare']['counts'] == {'equal': D['n_houses']}, f"{c}: rows differ from G4-E1 {d['e1_compare']}")
                ro = P['verdict']['per_seed'][s]['readouts']['aligned'].get('recovery_oracle_all')
                _check(ro is not None and abs(ro - 1.) <= R.ORACLE_TOL, f'oracle_all recovery {ro} is not ~1')
        print('smoke asserts ok')
    return S


if __name__ == '__main__':
    main()
