"""g4_registration.py (Claude, 2026-10-27): batch G4-E4 "经典位姿 stalk 配准", pre-registered in claude_handoff/OVERNIGHT_REPORT_20261021.md,
D8-39 (criteria copied verbatim below; nothing here may change them). CPU only (numpy / scipy), no GPU, 0 leases.

D8-39 (verbatim, the parts that bind this code):
  底线：位姿 stalk 是向量值 sheaf，每条树边的限制映射是相对位姿变换；能量 E_pose = Σ_边 匹配残差的截断二次和。生成森林上没有环，最小化 E_pose
    等于逐边估计相对位姿再从根沿树复合，这正是树上 GBP 的精确解；迭代时只接受让 E_pose 下降的步（断言）。体素证据在新位姿下重新落到公共
    坐标系，再按 G4-E1 的 V3 积分与森林求和融合。
  单变量：位姿层开 / 关。臂 = 经典配准：对每条树边，用双方噪声证据在重叠区的二维墙柱占据投影做相关式扫描匹配加 ICP 细化，估计相对 SE(2)
    修正，只用两智能体自己的证据，不用真值。对照 = 不配准（G4-E1 的 cam 档求和融合）。上限读数 = 真机间位姿（cam_nf）。
  数据与评分：与 G4-E1 完全相同（同一代码、同一噪声抽样、V3 积分、森林求和、按干净真值评分、VAL 判决、fresh 报告、两个噪声种子）。所有臂同时报
    "原样"和"整体对齐后"两行；判决用整体对齐后：对每个臂的融合结果，用同一个对真值的单一 SE(2) 最优对齐，再计分。
  判据（cam 档，两个噪声种子都要满足）：
    杠杆：R8 家具（按真值、整体对齐后）收回 cam_nf 与 cam 之差的 >= 85 %，且 E_pose 逐步不增（断言），且在 cam_nf 档上不劣于 -0.5 pp（没有误差时
      配准不能伤害结果），空闲召回不降、FPR 不劣于 +0.3 pp。
    不够：收回 < 50 % -> 经典方法不够，学习配准（G5-B，2 个租约）进入候选。其余 ambiguous。
  只报告：harsh 档的同样读数；逐边相对位姿误差（估计对真值，cm 与度）；按观测重数分层的家具 F1。

Frozen dependency: g4_noise_panels.py (G4-E1; its sha256 is pinned in every job of this batch, and that file is NEVER modified here) supplies
the scene rebuild (make_fixture, trace_pose), the noise injection with common random numbers (measure, pose_draws, frame_draws, to_public),
the V3 integrator (integrate_agent), the forest sum relay with provenance assertions (fuse 'sum') and the clean-truth scoring (House, Target,
score, stream_targets). This file only adds the pose layer and the re-integration.

Per (house, noise seed) job and per tier (default cam_nf, cam, harsh; G4-E1 parameters, the same draws):
  maps: K0 = V3 map of the agent's scans in its own frame (no inter-agent error); pub = V3 map of the same scans after the agent's
    inter-agent SE(2) error (G4-E1 to_public) = what the agent believes is the public frame. Both bit-identical to G4-E1's V3 maps
    (--check-e1-maps recomputes them with G4-E1's 4-integrator call; --e1-jobs compares the scored rows with G4-E1 job files).
  ESTIMATOR (register_forest; its inputs are only the agents' own believed-frame V3 maps, the public radio forest and public grid constants):
    evidence plane of an agent = wall-column occupancy projection of its V3 log-odds over the band [FLOOR_T + 0.3, FLOOR_T + 2.3) m
      (below every ceiling, 2.4-3.0 m; floor excluded): w(x, y) = mean_z clip(L, 0, 70) / 70 over known voxels (unknown = 0), and the free
      plane f(x, y) = mean_z [known and L <= 0];
    surfels: every column with w >= 0.2 -> weighted centroid / covariance of w over a (2 rad + 1)^2 box (rad = round(0.15 m / cell), >= 1):
      centroid m (sub-cell), normal n = minor eigenvector, kept if linearity 1 - lmin / lmax >= 0.3; oriented towards the observed free side
      when the free centroid lies >= 0.3 cell off along n (else unoriented); weight = w;
    edge factor (child j -> parent i of the tree rooted at the smallest agent of each component): Delta_ij maps j's believed frame into i's.
      E_edge(Delta) = sum_p w_p min(r_p^2, tau^2) over j's surfels moved by Delta against i's  +  the same over i's surfels moved by
      Delta^-1 against j's (symmetric: the residual of the restriction map does not depend on which agent is the parent);
      r_p = n_q . (Delta m_p - m_q), q = nearest orientation-compatible (45 deg) surfel within 0.30 m (none: r_p := tau); tau = 0.10 m.
      E_pose = sum over tree edges of E_edge (m^2).
    gate (trust region + overlap; a state that fails it is never accepted, so an edge without an acceptable state keeps the identity =
      no correction): |yaw| <= 10 deg and pivot displacement <= 0.6 m per axis (the search window); >= 50 inlier surfels (|r| < tau) in
      each direction; inlier weight >= 20 % of the better-covered side; the weakest Gauss-Newton direction (rotation with its lever arm
      in m) carries >= 0.02 of the inlier weight (a single wall, where the shift along it is unobservable, is refused).
    1) correlative scan matching: j's w plane rotated about the overlap pivot (yaw grid -10..10 deg, 0.5 deg step), correlated by FFT with i's
       w plane (both up-sampled 2x, i blurred with sigma 0.75 cell) over translations |t| <= 0.6 m (per axis, half-cell pixels); the
       argmax is a CANDIDATE step, accepted only if it lowers E_edge AND passes the gate;
    2) ICP refinement (from the accepted state if it, or the identity, passes the gate): point-to-line Gauss-Newton with Levenberg
       damping on the current inliers of both directions, correspondences re-found every iteration; a step is accepted only if it lowers
       E_edge (strict) and passes the gate, else the damping grows (4 tries), else stop; also stop when a step lowers E_edge by less than
       1e-6 relative, or after 60 iterations.
    (Provenance of these settings: checked on a synthetic room / planes and on TRAIN houses r1_i4, r1_i9, r2_i3, r3_i5 at noise seed 7,
    end to end: pooled aligned cam recovery .52, cam_nf reg - noreg -0.5 pp, harsh recovery .51; the trust region + gate were added
    after an ungated harsh edge drifted to -38 deg on TRAIN; a stricter overlap gate (0.5) gave .15 / +0.2 pp / .27 and was rejected.
    One exploratory VAL job (r1_i0, seed 1 = a smoke house) ran with the pre-gate matcher; nothing was tuned on VAL or fresh.)
    Every accepted step lowers its edge's term, so the E_pose trajectory (identity, after step 1, after every ICP iteration; edges padded
    with their last value, summed in a fixed order) is non-increasing: ASSERTED in every job (a violation crashes the job -> INCOMPLETE).
    Tree composition (exact tree solution): C_root = I, C_child = C_parent . Delta_parent,child. Gauge (does not change E_pose): 'prior'
    (default) = the unit prior "every believed frame is unbiased": H = mean over the component of C_a (yaw and the pivot displacement averaged
    about the centroid of the component's surfels), K_a = H^-1 . C_a; 'root' = K_a = C_a (reported on cam as row asis_rootgauge of the reg arm).
  arms (judged): noreg = pub maps (= G4-E1's cam sum); reg = every agent re-integrates ITS OWN beams after the correction K_a (V3), then
    forest sum; truepose = K0 maps (no inter-agent error; on cam this is exactly cam_nf's map: run_job raises unless the map hashes of
    cam K0, cam_nf K0 and cam_nf pub are equal (review M1c)).
  arms (cam only, REPORT ONLY, truth used only to build these labelled rows; D8-39 / D8-40 addendum 1, review M2):
    oracle_all       every tree edge gets its TRUE relative pose (pipeline fairness check: its recovery should be ~1; a deviation > .10
                     appends "alignment pipeline check failed" to the verdict text);
    ceiling_pairwise a tree edge whose two agents pass the matcher's gate AT THE TRUE RELATIVE POSE on their K0 evidence (overlap_true)
                     gets its true relative pose, every other tree edge the identity = the observability ceiling of the pairwise
                     tree-edge protocol;
    ceiling_islands  observability islands (review e4 diag_covis rule, restricted to a radio component): agents linked by pairs that pass
                     the gate at the true pose (K0 evidence), then groups whose POOLED evidence (max of their w / f planes) passes the gate
                     against another group are merged, repeatedly, until nothing changes; inside an island the members are put together
                     at their TRUE relative poses (C_a = A_anchor A_a^-1, anchor = smallest member), and each island is then placed by the
                     unit prior "every believed frame is unbiased" (coordinator, before the freeze): gauge 'prior' is applied PER ISLAND,
                     H_isl = mean_transform(C over the island, island pivot), K_a = H_isl^-1 C_a, island pivot = centroid of the members'
                     believed-frame surfels (as reg's pivots; the component pivot if the island has none); no second, component-level
                     gauge. So an island sits at the mean of its n members' believed frames (error ~ 1/sqrt(n)) = the ceiling of any
                     method that uses map evidence plus the unit prior only. Its as-is row is therefore reported in this island-level
                     gauge (the aligned row is unaffected). With --gauge root the anchors keep their believed frames (no island gauge).
    oracle_all / ceiling_pairwise: corrections composed along the tree in the run's gauge (component level). All three: same eval_arm
    (as-is + aligned rows). A report arm whose corrections equal an already evaluated one (|dK| <= 1e-12) reuses its rows (reused_from);
    a single-island component gives the same corrections as oracle_all.
  rows: asis = the arm's fused belief scored as G4-E1 does; aligned = per component ONE SE(2) G_c from the same matcher (coarse + ICP)
    between the arm's fused ternary belief (source) and the clean R8 truth (target), both through the same wall-column planes; every
    agent of the component re-integrates its beams under G_c . K_a; forest sum; scored. Identical procedure for every arm.
    aligned_best (report, review S2) = per component the better of asis / aligned by furniture F1 (no extra computation), with the count of
    components whose aligned row is below the as-is row; if the verdict category read on aligned_best differs, the verdict text says
    "sensitive to alignment".
  reported per edge: estimated and true relative pose, errors before (identity) / after (yaw deg, displacement of the pivot cm, RMS / max
    displacement over j's surfels cm), E_edge trajectory, overlap statistics, overlap_true (the gate at the true pose on K0 evidence) and
    failed (= no correction or the error grew; tiers with an inter-agent error only); per agent the gauge-free frame error (ATE cm / ARE
    deg after removing the component mean); per arm the alignment G_c; strata (height band, observation multiplicity) via G4-E1's score.
    NOTE (review S7): on cam_nf the "true" relative pose is taken as I, but each agent's 3 poses carry a per-frame error whose mean is
    ~1.2 cm / 0.29 deg, so part of cam_nf's err_after is a real correction; the cam_nf F1 guard, not err_after, is what counts.
Verdict (VAL, cam, aligned row, both noise seeds; INCOMPLETE unless seeds == {1, 2} with the same complete VAL house set, cam + cam_nf
  with all three judged arms in every job, no failed job (review S1), and -- review M1 -- every VAL cam and cam_nf job's as-is noreg and
  truepose rows equal to the G4-E1 job file's V3 sum / sum_truepose rows (compared at aggregation, --e1-jobs; missing = INCOMPLETE)):
  recovery = (F_reg - F_noreg) / (F_truepose - F_noreg) on R8 furniture F1 (G4-E1 aggregation: mean over agents and houses); LEVER iff
  recovery >= .85 AND E_pose monotone in every job AND cam_nf: F_reg - F_noreg >= -.005 AND cam: free recall reg >= noreg AND FPR reg <=
  noreg + .003, in both seeds; INSUFFICIENT iff recovery < .50 in both seeds; else AMBIGUOUS. Reading (addendum 2, review M3):
  INSUFFICIENT reads "classical method insufficient -> G5-B a candidate" only if recovery_ceiling_pairwise (aligned) >= .85 in both seeds,
  otherwise "relative poses not identifiable from two-agent evidence (panel observability); G5-B does not become a candidate"; every
  verdict text carries both ceilings. Recovery may exceed 1 (registration also removes part of the per-frame mean error). Reported per
  seed (review S3): the gap F_truepose - F_noreg, per-house recoveries (median, quartiles), a house-level bootstrap 95 % interval.
  E4b entry (addendum 3): decision.e4b_entry = VAL, both seeds: recovery_ceiling_islands - recovery_ceiling_pairwise >= .15 (aligned).
  The as-is row gets the same readouts (report only). fresh and harsh: the same readouts, not judged.

Resources and usage (review S9; python -B; python -O is refused because the pre-registered checks of G4-E1's frozen code are asserts):
  spawned workers, single-threaded, peak ~0.82 GB each (smoke); one job = one (house, noise seed) = 3 tiers, ~8-11 min per job while
  G4-E1's 10 processes share the CPU; 96 jobs (VAL + fresh x 2 seeds) / 4 workers ~ 4-4.5 h. ALWAYS a fresh --out for a changed script;
  a failed job writes a 'failed' record (traceback tail) and is re-run on resume. --check-e1-maps (voxel-for-voxel recomputation with
  G4-E1's own call) is a smoke option, not needed in the formal run.
  python -B g4_registration.py --out DIR [--panels val,fresh] [--seeds 1,2] [--workers 4] [--houses N] [--conds cam_nf,cam,harsh]
         [--gauge prior|root] [--e1-jobs DIR] [--check-e1-maps] [--smoke-asserts]
  python -B g4_registration.py --out DIR --aggregate-only --e1-jobs DIR      # re-aggregate (e.g. once G4-E1 has finished fresh)
Output: DIR/jobs/*.json, DIR/summary.json (per panel / seed / tier / arm / row, per house, edges, verdict, decision), DIR/run.json.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True

import argparse, collections, gc, glob, itertools, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_noise_panels as G  # noqa: E402   (frozen G4-E1 code: imported, never modified)

W = G.W
JOB_FORMAT = 2
CONDS_DEFAULT = ('cam_nf', 'cam', 'harsh')
ARMS = ('noreg', 'reg', 'truepose')                                    # judged arms
REPORT_ARMS = ('oracle_all', 'ceiling_pairwise', 'ceiling_islands')    # cam only, report only (truth-built labelled rows)
ROWS = ('asis', 'aligned')
REPORT_ROWS = ('aligned_best', 'asis_rootgauge')
JUDGED_COND, UPPER_COND, JUDGED_ROW = 'cam', 'cam_nf', 'aligned'
CEILING_READ, ORACLE_TOL, E4B_MARGIN = 0.85, 0.10, 0.15                # D8-39 / D8-40 addendum 1-3
BOOT_B, BOOT_SEED = 2000, 20261027
MATCH = dict(band_lo=0.3, band_hi=2.3, w_thr=0.2, surf_rad_m=0.15, lin_min=0.3, ori_min=0.3, up=2, blur_cells=0.75, yaw_max=10.0,
             yaw_step=0.5, t_max=0.6, tau=0.10, r_nn=0.30, k_nn=8, cos_nn=math.cos(math.radians(45.)), icp_iters=60,
             icp_rtol=1e-6, sym=True, gate_min_inl=50, gate_min_wfrac=0.20, gate_min_eig=0.02)
LEVER_RECOVERY, INSUFF_RECOVERY, NF_TOL, FPR_TOL = 0.85, 0.50, -0.005, 0.003
I3 = np.eye(3)


def _check(ok, msg):
    """pre-registered checks raise explicitly (review S4): they must not vanish under python -O"""
    if not ok:
        raise RuntimeError(msg)


# ================================================================ SE(2) (metres, radians; x' = R x + t as 3 x 3 homogeneous)
def se2(theta, tx=0., ty=0.):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def about(theta, c, t):
    """x -> R(theta) (x - c) + c + t"""
    c = np.asarray(c, np.float64); t = np.asarray(t, np.float64); M = se2(theta)
    M[:2, 2] = c + t - M[:2, :2] @ c
    return M


def inv(M):
    R = M[:2, :2]; Mi = np.eye(3); Mi[:2, :2] = R.T; Mi[:2, 2] = -R.T @ M[:2, 2]
    return Mi


def yaw(M):
    return math.atan2(M[1, 0], M[0, 0])


def apply2(M, P):
    return np.asarray(P, np.float64) @ M[:2, :2].T + M[:2, 2]


def params(M):
    """(yaw deg, tx m, ty m) of the origin-based form"""
    return [math.degrees(yaw(M)), float(M[0, 2]), float(M[1, 2])]


def frame_matrix(frame):
    """G4-E1's inter-agent error (Ra (3 x 3 rotz), ta, cen): x -> Ra (x - cen) + cen + ta, as an SE(2) matrix (z untouched)"""
    if frame is None:
        return np.eye(3)
    Ra, ta, cen = frame
    return about(math.atan2(Ra[1, 0], Ra[0, 0]), np.asarray(cen)[:2], np.asarray(ta)[:2])


def transform_scan(scan, M):
    """a scan's believed geometry moved by the SE(2) M through G4-E1's own to_public arithmetic (identity: the scan itself)"""
    if np.array_equal(M, I3):
        return scan
    R3 = np.eye(3); R3[:2, :2] = M[:2, :2]
    return G.to_public(scan, (R3, np.array([M[0, 2], M[1, 2], 0.]), np.zeros(3)))


def mean_transform(Ms, c):
    """the average of SE(2) transforms about the pivot c: mean yaw, mean displacement of c"""
    th = [yaw(M) for M in Ms]; d = [apply2(M, c) - c for M in Ms]
    return about(float(np.mean(th)), c, np.mean(d, 0))


def pose_err(M_est, M_true, P, c):
    """estimated vs true transform (both map the same source frame): yaw error (deg), displacement of the pivot c (cm), RMS / max
    displacement over the source points P (cm)"""
    dth = math.degrees(math.atan2(math.sin(yaw(M_est) - yaw(M_true)), math.cos(yaw(M_est) - yaw(M_true))))
    dc = float(np.linalg.norm(apply2(M_est, c) - apply2(M_true, c))) * 100
    if len(P):
        d = np.linalg.norm(apply2(M_est, P) - apply2(M_true, P), axis=1) * 100
        rms, mx = float(np.sqrt((d ** 2).mean())), float(d.max())
    else:
        rms = mx = None
    return dict(yaw_deg=dth, disp_pivot_cm=dc, disp_rms_cm=rms, disp_max_cm=mx)


# ================================================================ evidence planes and surfels (public grid constants only)
def band_mask(nz, cz, prm):
    zc = (np.arange(nz) + .5) * float(cz)
    return (zc >= W.FLOOR_T + prm['band_lo']) & (zc < W.FLOOR_T + prm['band_hi'])


def planes_logodds(L, cell, prm):
    """wall-column occupancy projection of an agent's V3 log-odds map (int8, unknown = -128): soft occupied plane w, free plane f"""
    b = band_mask(L.shape[2], cell[2], prm); Lb = L[:, :, b]; known = ~G.is_unk(Lb)
    soft = np.where(known, np.clip(Lb.astype(np.float64), 0., float(G.L_MAX)) / G.L_MAX, 0.)
    return soft.mean(2), (known & (Lb <= 0)).mean(2)


def planes_tern(T, cell, prm):
    """the same projection of a ternary map (-1 unknown / 0 free / 1 occupied): used for the global alignment (belief and truth)"""
    b = band_mask(T.shape[2], cell[2], prm); Tb = T[:, :, b]
    return (Tb == 1).mean(2).astype(np.float64), (Tb == 0).mean(2).astype(np.float64)


def surfels(w, f, cell, prm):
    cx = float(cell[0]); X, Y = w.shape
    xs = (np.arange(X) + .5) * cx; ys = (np.arange(Y) + .5) * cx
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    rad = max(1, int(round(prm['surf_rad_m'] / cx)))
    cand = w >= prm['w_thr']
    empty = dict(m=np.zeros((0, 2)), n=np.zeros((0, 2)), w=np.zeros(0), ori=np.zeros(0, bool))
    if not cand.any():
        return empty
    U = lambda a, r: ndimage.uniform_filter(a, size=2 * r + 1, mode='constant', cval=0.)[cand]  # noqa: E731
    S0 = U(w, rad); mx = U(w * Xg, rad) / S0; my = U(w * Yg, rad) / S0
    cxx = U(w * Xg * Xg, rad) / S0 - mx * mx; cxy = U(w * Xg * Yg, rad) / S0 - mx * my; cyy = U(w * Yg * Yg, rad) / S0 - my * my
    C = np.stack([np.stack([cxx, cxy], -1), np.stack([cxy, cyy], -1)], -2)
    lam, vec = np.linalg.eigh(C)
    n = vec[:, :, 0].copy(); lin = 1. - lam[:, 0] / np.maximum(lam[:, 1], 1e-18)
    keep = (lam[:, 1] > 1e-12) & (lin >= prm['lin_min'])
    Sf = U(f, rad + 1); gx = U(f * Xg, rad + 1) - mx * Sf; gy = U(f * Yg, rad + 1) - my * Sf
    s = (gx * n[:, 0] + gy * n[:, 1]) / (Sf * cx + 1e-12)                  # free centroid offset along n, in cells
    ori = np.abs(s) >= prm['ori_min']
    n[ori & (s < 0)] *= -1
    m = np.stack([mx, my], 1)
    return dict(m=m[keep], n=n[keep], w=w[cand][keep].astype(np.float64), ori=ori[keep])


def evidence(planes, cell, prm):
    w, f = planes; S = surfels(w, f, cell, prm)
    S['tree'] = cKDTree(S['m']) if len(S['m']) else None
    return dict(w=w, f=f, S=S)


# ================================================================ edge energy, correlative search, ICP
def _match(Ss, St, M, prm):
    """source surfels Ss moved by M against target surfels St: truncated quadratic sum_p w_p min(r_p^2, tau^2) of the point-to-line
    residuals r_p = n_q . (M m_p - m_q), q = the nearest orientation-compatible target surfel within r_nn (none: r_p := tau)"""
    tau = prm['tau']; n = len(Ss['m'])
    y = apply2(M, Ss['m']) if n else np.zeros((0, 2))
    if n == 0 or St['tree'] is None:
        return float(Ss['w'].sum() * tau * tau), dict(y=y, nq=np.zeros((n, 2)), r=np.full(n, tau), inl=np.zeros(n, bool))
    ns = Ss['n'] @ M[:2, :2].T
    K = min(prm['k_nn'], len(St['m']))
    _, idx = St['tree'].query(y, k=K, distance_upper_bound=prm['r_nn'])
    if K == 1:
        idx = idx[:, None]
    valid = idx < len(St['m']); idc = np.where(valid, idx, 0)
    dot = np.einsum('nkd,nd->nk', St['n'][idc], ns)
    both = St['ori'][idc] & Ss['ori'][:, None]
    compat = valid & np.where(both, dot >= prm['cos_nn'], np.abs(dot) >= prm['cos_nn'])
    has = compat.any(1); q = idc[np.arange(n), compat.argmax(1)]; nq = St['n'][q]
    r = np.where(has, np.einsum('nd,nd->n', y - St['m'][q], nq), tau)
    E = float(np.dot(Ss['w'], np.minimum(r * r, tau * tau)))
    return E, dict(y=y, nq=nq, r=r, inl=has & (np.abs(r) < tau))


def edge_cost(Sj, Si, M, prm, full=False):
    """E_edge(M) for the edge whose relative pose M maps j's frame into i's: symmetric (sym = True) sum of the truncated quadratics of
    j's surfels moved by M against i's and of i's surfels moved by M^-1 against j's (the restriction-map residual does not depend on
    which agent is the parent); sym = False keeps the forward term only"""
    Ef, f = _match(Sj, Si, M, prm)
    if prm.get('sym', True):
        Er, rv = _match(Si, Sj, inv(M), prm)
    else:
        Er, rv = 0., None
    E = Ef + Er
    return (E, dict(f=f, r=rv)) if full else E


def gn_system(Sj, Si, M, info):
    """Gauss-Newton rows of the current inliers for a left perturbation delta (theta, tx, ty) about the inlier centroid c, in i's frame:
    M' = delta . M. Forward rows: d r / d xi = [n_q . J (y - c), n_q]; reverse rows (i's points x, j-normals n moved into i's frame by
    R_M): d r / d xi = -[(R_M n) . J (x - c), R_M n]."""
    f = info['f']; rv = info['r']
    P, N, r, w, sg = [f['y'][f['inl']]], [f['nq'][f['inl']]], [f['r'][f['inl']]], [Sj['w'][f['inl']]], [np.ones(int(f['inl'].sum()))]
    if rv is not None:
        P.append(Si['m'][rv['inl']]); N.append(rv['nq'][rv['inl']] @ M[:2, :2].T); r.append(rv['r'][rv['inl']])
        w.append(Si['w'][rv['inl']]); sg.append(-np.ones(int(rv['inl'].sum())))
    P, N, r, w, sg = (np.concatenate(x) for x in (P, N, r, w, sg))
    if len(r) < 3:
        return None
    c = (w[:, None] * P).sum(0) / w.sum(); d = P - c
    A = sg[:, None] * np.stack([N[:, 1] * d[:, 0] - N[:, 0] * d[:, 1], N[:, 0], N[:, 1]], 1)
    return A, r, w, c


def gate(Sj, Si, M, info, pivot, prm):
    """a state may be accepted only inside the search window (trust region: |yaw| <= yaw_max, pivot displacement <= t_max per axis)
    and with a non-degenerate overlap: >= gate_min_inl inliers in each direction, inlier weight >= gate_min_wfrac of the smaller side,
    and the weakest Gauss-Newton direction carrying >= gate_min_eig of the inlier weight (rotation measured with its lever arm, m)"""
    dp = apply2(M, pivot) - pivot
    tr = abs(math.degrees(yaw(M))) <= prm['yaw_max'] + 1e-9 and abs(dp[0]) <= prm['t_max'] + 1e-9 and abs(dp[1]) <= prm['t_max'] + 1e-9
    f = info['f']; rv = info['r']
    nf = int(f['inl'].sum()); wf = float(Sj['w'][f['inl']].sum() / max(Sj['w'].sum(), 1e-12))
    nr, wr = (int(rv['inl'].sum()), float(Si['w'][rv['inl']].sum() / max(Si['w'].sum(), 1e-12))) if rv is not None else (nf, wf)
    sysm = gn_system(Sj, Si, M, info)
    if sysm is None:
        eig = 0.
    else:
        A, _, w, _ = sysm
        eig = float(np.linalg.eigvalsh(A.T @ (w[:, None] * A))[0] / w.sum())
    ok = tr and min(nf, nr) >= prm['gate_min_inl'] and max(wf, wr) >= prm['gate_min_wfrac'] and eig >= prm['gate_min_eig']
    return bool(ok), dict(trust=bool(tr), n_inl=min(nf, nr), n_inl_fwd=nf, n_inl_rev=nr, inl_wfrac=max(wf, wr), inl_wfrac_fwd=wf,
                          inl_wfrac_rev=wr, min_eig=eig)


def overlap_pivot(Sj, Si, prm):
    if len(Sj['m']) == 0:
        return np.zeros(2)
    if Si['tree'] is not None:
        d, _ = Si['tree'].query(Sj['m'], k=1, distance_upper_bound=prm['r_nn'] + prm['t_max'])
        ov = np.isfinite(d)
        if ov.sum() >= 3:
            return (Sj['w'][ov, None] * Sj['m'][ov]).sum(0) / Sj['w'][ov].sum()
    return (Sj['w'][:, None] * Sj['m']).sum(0) / Sj['w'].sum()


def coarse_search(w_i, w_j, cell, pivot, prm):
    """correlative scan matching: argmax over yaw (grid) and translation (FFT, half-cell pixels) of sum_x F_i(x + t) w_j,yaw(x), where F_i
    is i's plane blurred, w_j,yaw is j's plane rotated by yaw about the pivot. Returns the candidate Delta (j -> i) and the score."""
    up = int(prm['up']); cx = float(cell[0]); p = cx / up
    A = ndimage.gaussian_filter(np.repeat(np.repeat(w_i, up, 0), up, 1), prm['blur_cells'] * up, mode='constant')
    B = np.repeat(np.repeat(w_j, up, 0), up, 1)
    T = int(math.ceil(prm['t_max'] / p)); shape = (A.shape[0] + 2 * T, A.shape[1] + 2 * T)
    FA = np.fft.rfft2(A, s=shape)
    ix = np.arange(-T, T + 1) % shape[0]; iy = np.arange(-T, T + 1) % shape[1]
    c = np.asarray(pivot, np.float64); best = (-np.inf, 0., 0, 0)
    n_yaw = int(round(prm['yaw_max'] / prm['yaw_step']))
    for k in range(-n_yaw, n_yaw + 1):
        th = math.radians(k * prm['yaw_step'])
        Rinv = se2(-th)[:2, :2]
        off = Rinv @ (0.5 * p - c) / p + c / p - 0.5                       # input index = Rinv @ output index + off
        Bt = ndimage.affine_transform(B, Rinv, offset=off, order=1, mode='constant', cval=0.) if k else B
        C = np.fft.irfft2(FA * np.conj(np.fft.rfft2(Bt, s=shape)), s=shape)[np.ix_(ix, iy)]
        a = int(np.argmax(C)); v = float(C.flat[a])
        if v > best[0]:
            best = (v, th, a // C.shape[1] - T, a % C.shape[1] - T)
    v, th, kx, ky = best
    return about(th, c, (kx * p, ky * p)), dict(score=v, yaw_deg=math.degrees(th), tx=kx * p, ty=ky * p)


def icp_refine(Sj, Si, M, E, pivot, prm, traj):
    """point-to-line ICP (Gauss-Newton on the current inliers, correspondences re-found every iteration, Levenberg damping); a step is
    taken only if it strictly lowers E_edge AND the new state passes the gate; appends every accepted energy to traj"""
    it, stop = 0, 'max_iter'
    for it in range(1, prm['icp_iters'] + 1):
        _, info = edge_cost(Sj, Si, M, prm, full=True)
        sysm = gn_system(Sj, Si, M, info)
        if sysm is None:
            stop = 'no_inliers'; it -= 1; break
        A, r, w, c = sysm
        H = A.T @ (w[:, None] * A); g = A.T @ (w * r)
        took = False
        for lam in (1e-4, 1e-2, 1., 1e2):
            Hd = H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(3)
            try:
                xi = -np.linalg.solve(Hd, g)
            except np.linalg.LinAlgError:
                xi = -np.linalg.lstsq(Hd, g, rcond=None)[0]
            Mn = about(float(xi[0]), c, xi[1:]) @ M
            En, infn = edge_cost(Sj, Si, Mn, prm, full=True)
            if En < E and gate(Sj, Si, Mn, infn, pivot, prm)[0]:
                took = True; break
        if not took:
            stop = 'no_descent'; it -= 1; break
        dE = E - En; M, E = Mn, En; traj.append(E)
        if (abs(xi[0]) < 1e-7 and math.hypot(xi[1], xi[2]) < 1e-6) or dE <= prm['icp_rtol'] * (E + dE):
            stop = 'converged'; break
    return M, E, it, stop


def register_pair(ev_i, ev_j, cell, prm):
    """relative SE(2) Delta mapping j's frame into i's from the two evidence planes alone: E_edge at identity; the correlative candidate
    (accepted iff it lowers E_edge and passes the gate); ICP (every step lowers E_edge and passes the gate). The returned trajectory is
    strictly decreasing. A pair without an acceptable state keeps the identity (no correction)."""
    Si, Sj = ev_i['S'], ev_j['S']
    rec = dict(n_src=int(len(Sj['m'])), n_tgt=int(len(Si['m'])), w_src=float(Sj['w'].sum()))
    E, info = edge_cost(Sj, Si, I3, prm, full=True); traj = [E]; M = I3
    if len(Sj['m']) < 3 or len(Si['m']) < 3:
        rec.update(M=M, E=E, traj=traj, pivot=[0., 0.], coarse=None, coarse_accepted=False, icp_iters=0, icp_stop='no_surfels',
                   final_gate=gate(Sj, Si, M, info, np.zeros(2), prm)[1], identity_gate=None)
        return rec
    pivot = overlap_pivot(Sj, Si, prm)
    g0 = gate(Sj, Si, M, info, pivot, prm)
    Mc, cinfo = coarse_search(ev_i['w'], ev_j['w'], cell, pivot, prm)
    Ec, infc = edge_cost(Sj, Si, Mc, prm, full=True)
    gok, gst = gate(Sj, Si, Mc, infc, pivot, prm)
    acc = bool(Ec < E and gok)
    if acc:
        M, E = Mc, Ec
        traj.append(E)
    cinfo.update(E=Ec, gate=gok, **gst)
    if acc or g0[0]:                                                    # refine only from a state that passes the gate
        M, E, iters, stop = icp_refine(Sj, Si, M, E, pivot, prm, traj)
    else:
        iters, stop = 0, 'gate'
    Ef, inff = edge_cost(Sj, Si, M, prm, full=True)
    _check(Ef == E, 'E_edge bookkeeping: the tracked energy differs from a fresh evaluation at the final pose')
    rec.update(M=M, E=E, traj=traj, pivot=[float(pivot[0]), float(pivot[1])], coarse=cinfo, coarse_accepted=acc, icp_iters=int(iters),
               icp_stop=stop, final_gate=gate(Sj, Si, M, inff, pivot, prm)[1], identity_gate=g0[1])
    return rec


def assert_monotone(traj, what='E_pose'):
    """the pre-registered assertion: an energy trajectory never increases (explicit raise, review S4). Returns the number of checks."""
    for k in range(1, len(traj)):
        _check(traj[k] <= traj[k - 1], f'{what} increased at step {k}: {traj[k - 1]!r} -> {traj[k]!r}')
    return len(traj) - 1


def tree_order(F):
    """per component: root = its smallest agent; BFS order and parent of every other agent"""
    parent, order = {}, []
    for cm in F['comps']:
        root = cm[0]; parent[root] = None; q = collections.deque([root]); seen = {root}
        while q:
            u = q.popleft(); order.append(u)
            for v in F['nb'][u]:
                if v not in seen:
                    seen.add(v); parent[v] = u; q.append(v)
    return parent, order


def compose(F, deltas, gauge, pivots):
    """exact tree solution: C_root = I, C_child = C_parent . Delta_parent,child; gauge 'prior' -> K_a = H^-1 C_a with H the mean of the
    component's C_a about its pivot; 'root' -> K_a = C_a"""
    parent, order = tree_order(F)
    C = {}
    for a in order:
        C[a] = I3.copy() if parent[a] is None else C[parent[a]] @ deltas[(parent[a], a)]
    K = gauge_fix(F, C, gauge, pivots)
    return [C[a] for a in range(len(order))], [K[a] for a in range(len(order))]


def gauge_fix(F, C, gauge, pivots):
    """C: {agent: correction}; 'prior' -> K_a = H^-1 C_a, H = mean of the component's C_a about its pivot; 'root' -> K_a = C_a"""
    K = dict(C)
    if gauge == 'prior':
        for ci, cm in enumerate(F['comps']):
            Hinv = inv(mean_transform([C[a] for a in cm], pivots[ci]))
            for a in cm:
                K[a] = Hinv @ C[a]
    elif gauge != 'root':
        raise ValueError(gauge)
    return K


def register_forest(maps, F, cell, prm, gauge='prior'):
    """THE ESTIMATOR. Inputs: the agents' own V3 maps in their believed public frames, the public radio forest, the public grid (cell size,
    FLOOR_T) and the matcher settings -- no truth, no true pose, no inter-agent error. Returns the per-edge relative poses, the E_pose
    trajectory (asserted non-increasing) and the corrections K_a (believed frame of a -> common frame)."""
    n = len(maps); parent, order = tree_order(F)
    ev = [evidence(planes_logodds(L, cell, prm), cell, prm) for L in maps]
    edges = {}
    for a in order:
        if parent[a] is not None:
            edges[(parent[a], a)] = register_pair(ev[parent[a]], ev[a], cell, prm)
    keys = sorted(edges)
    checks = 0
    for k in keys:
        checks += assert_monotone(edges[k]['traj'], f'E_edge{k}')
    T = max([len(edges[k]['traj']) for k in keys], default=1)
    traj = [float(sum(edges[k]['traj'][min(t, len(edges[k]['traj']) - 1)] for k in keys)) for t in range(T)]
    checks += assert_monotone(traj, 'E_pose')
    pivots = []
    for cm in F['comps']:
        P = np.concatenate([ev[a]['S']['m'] for a in cm]) if any(len(ev[a]['S']['m']) for a in cm) else np.zeros((1, 2))
        pivots.append(P.mean(0))
    C, K = compose(F, {k: edges[k]['M'] for k in keys}, gauge, pivots)
    return dict(edges=edges, keys=keys, E_pose=traj, monotone_checks=checks, C=C, K=K, pivots=pivots,
                surf_m=[ev[a]['S']['m'] for a in range(n)], n_surfels=[int(len(ev[a]['S']['m'])) for a in range(n)])


# ================================================================ report-only observability rows (truth used only to label them)
def observable_pair(ev_i, ev_j, prm):
    """the matcher's gate at the identity on two evidences that are already in a common frame (K0 maps = true relative pose): is the
    pair's relative pose observable from its two evidences? (j = source, i = target, as for a tree edge parent i <- child j)"""
    Si, Sj = ev_i['S'], ev_j['S']
    if len(Si['m']) < 3 or len(Sj['m']) < 3:
        return False, dict(ok=False, n_inl=0, inl_wfrac=0., min_eig=0.)
    _, info = edge_cost(Sj, Si, I3, prm, full=True)
    ok, st = gate(Sj, Si, I3, info, overlap_pivot(Sj, Si, prm), prm)
    return ok, dict(ok=ok, n_inl=st['n_inl'], inl_wfrac=st['inl_wfrac'], min_eig=st['min_eig'])


def covis_groups(members, pair_ok):
    """connected components of the 'pair passes the gate' graph over members (union-find)"""
    par = {a: a for a in members}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for x, y in itertools.combinations(sorted(members), 2):
        if pair_ok(x, y):
            par[find(x)] = find(y)
    return sorted(sorted(a for a in members if find(a) == r) for r in {find(a) for a in members})


def merge_islands(groups, group_ok):
    """the reviewer's covis rule: repeatedly merge the first pair of groups (in order) whose pooled evidence passes group_ok, until no
    pair does; a merged group is appended at the end"""
    Q = [sorted(q) for q in groups]
    while True:
        merged = False
        for x, y in itertools.combinations(range(len(Q)), 2):
            if group_ok(Q[x], Q[y]):
                Q = [q for k, q in enumerate(Q) if k not in (x, y)] + [sorted(Q[x] + Q[y])]; merged = True
                break
        if not merged:
            return sorted(tuple(q) for q in Q)


def observability_islands(F, planes, cell, prm):
    """per radio component: co-visibility groups (pairs passing the gate on K0 evidence), then pooled-evidence merging (max of the
    groups' w / f planes) -> islands. Returns (islands as tuples, number of co-visible pairs, number of pairs)."""
    ev = [evidence(p, cell, prm) for p in planes]; cache = {}

    def pooled(g):
        g = tuple(sorted(g))
        if g not in cache:
            cache[g] = ev[g[0]] if len(g) == 1 else evidence((np.max([planes[b][0] for b in g], 0), np.max([planes[b][1] for b in g], 0)), cell, prm)
        return cache[g]
    pair_cache = {}

    def pair_ok(x, y):
        pair_cache[(x, y)] = observable_pair(ev[x], ev[y], prm)[0]
        return pair_cache[(x, y)]

    def group_ok(gx, gy):
        return observable_pair(pooled(gx), pooled(gy), prm)[0]
    islands = []
    for cm in F['comps']:
        islands += merge_islands(covis_groups(cm, pair_ok), group_ok)
    return sorted(islands), int(sum(pair_cache.values())), len(pair_cache)


def truth_deltas(keys, MAs, use=None):
    """true relative pose of every tree edge (use[k] False -> identity)"""
    return {(i, j): (MAs[i] @ inv(MAs[j]) if (use is None or use[(i, j)]) else I3) for (i, j) in keys}


def islands_K(F, islands, MAs, gauge, pivots, surf_m=None):
    """ceiling_islands: inside an island the members get their TRUE relative poses (C_a = A_anchor A_a^-1, anchor = smallest member);
    gauge 'prior' is then applied PER ISLAND (unit prior: every believed frame unbiased -> the island sits at the mean of its members'
    believed frames): K_a = H_isl^-1 C_a, H_isl = mean_transform(C over the island, island pivot), island pivot = centroid of the members'
    surfels (surf_m, computed as reg's pivots), else the component pivot. gauge 'root': K_a = C_a (anchors keep their believed frames)."""
    C = {}
    for isl in islands:
        a0 = min(isl)
        for a in isl:
            C[a] = I3.copy() if a == a0 else MAs[a0] @ inv(MAs[a])
    if gauge == 'root':
        K = C
    elif gauge == 'prior':
        K = {}
        for isl in islands:
            members = sorted(isl)
            if surf_m is not None and any(len(surf_m[a]) for a in members):
                piv = np.concatenate([surf_m[a] for a in members]).mean(0)
            else:
                piv = pivots[F['comp_of'][members[0]]]
            Hinv = inv(mean_transform([C[a] for a in members], piv))
            for a in members:
                K[a] = Hinv @ C[a]
    else:
        raise ValueError(gauge)
    return [K[a] for a in range(len(MAs))]


# ================================================================ one (house, noise seed) job
def file_sha256(path):
    return G.file_sha256(path)


def fingerprint(opts):
    return json.loads(json.dumps({k: opts[k] for k in sorted(opts) if k not in ('out', 'e1_jobs', 'check_e1_maps')}))


def _integrate(shape, cell, ds, scans, M):
    return G.integrate_agent(shape, cell, ds, [transform_scan(s, M) for s in scans], ('V3',))['V3']


def _rows(preds, R8T, comps):
    return [[G.score(preds[cm[0]], R8T[ci], strata=True), len(cm)] for ci, cm in enumerate(comps)]


def eval_arm(maps0, base_scans, Ks, F, R8T, T8, shape, cell, ds, prm, rstats):
    """rows 'asis' (the arm's maps fused) and 'aligned' (per component one SE(2) from the matcher between the fused ternary belief and the
    clean R8 truth; every agent re-integrates its beams under G_c . K_a)"""
    n = len(maps0); comps = F['comps']
    _, preds = G.fuse(maps0, F, 'sum', rstats)
    out = dict(asis=_rows(preds, R8T, comps), align=[])
    al = [None] * n
    for ci, cm in enumerate(comps):
        ev_t = evidence(planes_tern(T8[cm[0]], cell, prm), cell, prm); ev_b = evidence(planes_tern(preds[cm[0]], cell, prm), cell, prm)
        res = register_pair(ev_t, ev_b, cell, prm)
        Gc = res['M']
        out['align'].append(dict(G=params(Gc), E0=res['traj'][0], E=res['E'], steps=len(res['traj']) - 1, coarse_accepted=res['coarse_accepted'],
                                 icp_stop=res['icp_stop'], final_gate=res['final_gate']))
        for a in cm:
            M = Gc @ Ks[a]
            al[a] = maps0[a] if np.array_equal(Gc, I3) else _integrate(shape, cell, ds, base_scans[a], M)
    _, preds_al = G.fuse(al, F, 'sum', rstats)
    out['aligned'] = _rows(preds_al, R8T, comps)
    out['aligned_best'], out['aligned_below_asis'] = aligned_best(out['asis'], out['aligned'])
    return out


def aligned_best(asis, aligned):
    """review S2 (report only): per component the better of the as-is / aligned rows by furniture F1 (ties and undefined -> aligned),
    and the number of components whose aligned row is below the as-is row"""
    best, below = [], 0
    for (ra, ka), (rl, kl) in zip(asis, aligned):
        fa, fl = ra.get('furn_f1'), rl.get('furn_f1')
        if fa is not None and fl is not None and fl < fa:
            below += 1
        best.append([ra, ka] if (fa is not None and (fl is None or fa > fl)) else [rl, kl])
    return best, below


def agent_errors(Ks, MAs, F, pivots):
    """gauge-free per-agent frame error: R_a = K_a . A_a (true -> corrected) minus the component mean (about the component pivot)"""
    out = []
    for ci, cm in enumerate(F['comps']):
        c = pivots[ci]; R = [Ks[a] @ MAs[a] for a in cm]; Mbar = mean_transform(R, c)
        for a, Ra in zip(cm, R):
            E = inv(Mbar) @ Ra
            out.append(dict(agent=int(a), yaw_deg=math.degrees(yaw(E)), disp_cm=float(np.linalg.norm(apply2(E, c) - c) * 100)))
    yaws = [e['yaw_deg'] for e in out]; ds_ = [e['disp_cm'] for e in out]
    return dict(per_agent=out, ARE_deg=float(np.sqrt(np.mean(np.square(yaws)))) if out else None,
                ATE_cm=float(np.sqrt(np.mean(np.square(ds_)))) if out else None)


def e1_compare_job(job, e1_dir, _cache={}):
    """review M1 (aggregation time): per tier, do this job's as-is noreg / truepose rows equal the G4-E1 job file's V3 'sum' /
    'sum_truepose' rows (cam_nf has no inter-agent error: 'sum' for both)? Status per tier: equal / unequal / missing (no E1 file) /
    other_e1_version (E1 job written by another g4_noise_panels than the one pinned in this job) / tier_missing / no_e1_dir."""
    out = {}
    path = os.path.join(e1_dir, f"{job['panel']}_r{job['rooms']}_i{job['index']}_s{job['seed']}.json") if e1_dir else None
    if path and path not in _cache and os.path.exists(path):
        _cache[path] = json.load(open(path))
    j = _cache.get(path) if path else None
    norm = lambda x: json.loads(json.dumps(x))  # noqa: E731
    for c, rec in job['conds'].items():
        if not e1_dir:
            out[c] = 'no_e1_dir'
        elif j is None:
            out[c] = 'missing'
        elif (j.get('fingerprint') or {}).get('script_sha256') != job['fingerprint'].get('g4_noise_panels_sha256'):
            out[c] = 'other_e1_version'
        elif c not in (j.get('conds') or {}):
            out[c] = 'tier_missing'
        else:
            R8 = j['conds'][c]['by_int']['V3']['R8']
            eq = norm(rec['arms']['noreg']['asis']) == R8['sum'] and norm(rec['arms']['truepose']['asis']) == R8.get('sum_truepose', R8['sum'])
            out[c] = 'equal' if eq else 'unequal'
    return out


def run_job(job):
    panel, rooms, index, seed, opts = job
    _check(__debug__, 'python -O strips the pre-registered assertions of the frozen G4-E1 code: refused')
    _check(file_sha256(os.path.abspath(__file__)) == opts['script_sha256'], 'g4_registration.py changed since the run started')
    _check(file_sha256(G.__file__) == opts['g4_noise_panels_sha256'], 'g4_noise_panels.py changed since the run started')
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
        for a in range(n):
            frame = None
            if has_inter:
                z = G.frame_draws(seed, rooms, index, a)
                frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
                frames.append([float(z[0] * c['inter_t']), float(z[1] * c['inter_t']), float(z[2] * c['inter_yaw'])])
            MAs.append(frame_matrix(frame))
            scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
            pub = [G.to_public(s, frame) for s in scans]
            if (ck, a) not in k0cache:
                k0cache[(ck, a)] = G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3']
            L_k0.append(k0cache[(ck, a)])
            L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3'] if has_inter else L_k0[a])
            scans_k0.append(scans); scans_pub.append(pub)
        timing[f'{cname}_integrate'] = time.time() - tc
        rec = dict(inter_frames_true=frames, k0_sha256=W.array_hash(np.stack(L_k0)), pub_sha256=W.array_hash(np.stack(L_pub)))
        if opts.get('check_e1_maps'):                                   # G4-E1's own call (all four integrators) -> its V3 maps
            eq_k0 = eq_pub = 0
            for a in range(n):
                eq_k0 += int(np.array_equal(G.integrate_agent(shape, cell, ds, scans_k0[a], list(G.INTEGRATORS))['V3'], L_k0[a]))
                if has_inter:
                    eq_pub += int(np.array_equal(G.integrate_agent(shape, cell, ds, scans_pub[a], list(G.INTEGRATORS))['V3'], L_pub[a]))
            rec['e1_map_check'] = dict(k0_equal=eq_k0, pub_equal=eq_pub if has_inter else eq_k0, agents=n)
        # ---- the pose layer (estimator sees L_pub only)
        tr = time.time()
        reg = register_forest(L_pub, F, cell, prm, opts['gauge'])
        timing[f'{cname}_register'] = time.time() - tr
        # ---- report only: overlap of every tree edge at the TRUE relative pose (K0 evidence, the same gate)
        planes_k0 = [planes_logodds(L, cell, prm) for L in L_k0]
        ev_k0 = [evidence(p, cell, prm) for p in planes_k0]
        obs = {(i, j): observable_pair(ev_k0[i], ev_k0[j], prm) for (i, j) in reg['keys']}
        edges = []
        for (i, j) in reg['keys']:
            e = reg['edges'][(i, j)]; Dt = MAs[i] @ inv(MAs[j]); P = reg['surf_m'][j]; cpiv = np.asarray(e['pivot'])
            eb, ea = pose_err(I3, Dt, P, cpiv), pose_err(e['M'], Dt, P, cpiv); corrected = not np.array_equal(e['M'], I3)
            failed = (not corrected or (ea['disp_rms_cm'] or 0.) > (eb['disp_rms_cm'] or 0.)) if has_inter else None
            edges.append(dict(i=int(i), j=int(j), true=params(Dt), est=params(e['M']), pivot=e['pivot'],
                              err_before=eb, err_after=ea,
                              E_traj=[float(x) for x in e['traj']], coarse=e['coarse'], coarse_accepted=e['coarse_accepted'],
                              icp_iters=e['icp_iters'], icp_stop=e['icp_stop'], n_src=e['n_src'], n_tgt=e['n_tgt'],
                              identity_gate=e['identity_gate'], final_gate=e['final_gate'], corrected=corrected,
                              overlap_true=obs[(i, j)][1], failed=failed))
        rec['registration'] = dict(gauge=opts['gauge'], edges=edges, E_pose=reg['E_pose'], E_pose_monotone=True, monotone_checks=reg['monotone_checks'],
                                   K=[params(K) for K in reg['K']], C_root=[params(C) for C in reg['C']], n_surfels=reg['n_surfels'],
                                   agent_err_before=agent_errors([I3] * n, MAs, F, reg['pivots']),
                                   agent_err_after=agent_errors(reg['K'], MAs, F, reg['pivots']))
        # ---- arms
        ta = time.time()
        L_reg = [L_pub[a] if np.array_equal(reg['K'][a], I3) else _integrate(shape, cell, ds, scans_pub[a], reg['K'][a]) for a in range(n)]
        arms = {}
        arms['noreg'] = (armcache[('k0', ck)] if (not has_inter and ('k0', ck) in armcache)
                         else eval_arm(L_pub, scans_pub, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats))
        arms['reg'] = eval_arm(L_reg, scans_pub, reg['K'], F, R8T, T8, shape, cell, ds, prm, rstats)
        if ('k0', ck) not in armcache:
            armcache[('k0', ck)] = arms['noreg'] if not has_inter else eval_arm(L_k0, scans_k0, [I3] * n, F, R8T, T8, shape, cell, ds, prm, rstats)
        arms['truepose'] = armcache[('k0', ck)]
        if opts['gauge'] == 'prior' and cname == JUDGED_COND:          # report row (cam only): the same corrections in the root gauge
            Lr = [L_pub[a] if np.array_equal(reg['C'][a], I3) else _integrate(shape, cell, ds, scans_pub[a], reg['C'][a]) for a in range(n)]
            _, pr = G.fuse(Lr, F, 'sum', rstats)
            arms['reg']['asis_rootgauge'] = _rows(pr, R8T, F['comps'])
            del Lr, pr
        timing[f'{cname}_arms'] = time.time() - ta
        if cname == JUDGED_COND:                                        # ---- report-only arms (truth used only to label them)
            tq = time.time()
            islands, n_cov, n_pairs = observability_islands(F, planes_k0, cell, prm)
            rec['observability'] = dict(islands=[list(x) for x in islands], n_islands=len(islands), covisible_pairs=n_cov, pairs=n_pairs,
                                        tree_edges_observable=int(sum(o[0] for o in obs.values())), tree_edges=len(obs))
            Ks = dict(oracle_all=compose(F, truth_deltas(reg['keys'], MAs), opts['gauge'], reg['pivots'])[1],
                      ceiling_pairwise=compose(F, truth_deltas(reg['keys'], MAs, {k: o[0] for k, o in obs.items()}), opts['gauge'], reg['pivots'])[1],
                      ceiling_islands=islands_K(F, islands, MAs, opts['gauge'], reg['pivots'], reg['surf_m']))
            done = [('noreg', [I3] * n), ('reg', reg['K'])]
            for name in REPORT_ARMS:
                K = Ks[name]
                prev = next((nm for nm, Kp in done if all(np.allclose(x, y, rtol=0., atol=1e-12) for x, y in zip(K, Kp))), None)
                if prev is not None:
                    res_arm = dict(arms[prev], reused_from=prev)
                else:
                    L_arm = [L_pub[a] if np.array_equal(K[a], I3) else _integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
                    res_arm = eval_arm(L_arm, scans_pub, K, F, R8T, T8, shape, cell, ds, prm, rstats)
                    del L_arm
                res_arm.pop('asis_rootgauge', None)
                arms[name] = dict(res_arm, K=[params(x) for x in K])
                done.append((name, K))
            timing[f'{cname}_report_arms'] = time.time() - tq
        rec['arms'] = arms
        timing[f'{cname}_total'] = time.time() - tc
        out['conds'][cname] = rec
        del scans_k0, scans_pub, L_k0, L_pub, L_reg
        later = {G._ck(G.CONDITIONS[c2]) for c2 in opts['conds'][opts['conds'].index(cname) + 1:]}
        for key in [k for k in k0cache if k[0] not in later]:
            del k0cache[key]
        gc.collect()
    if 'cam' in out['conds'] and 'cam_nf' in out['conds']:            # review M1c: cam truepose IS cam_nf's map
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
    """review S1: a failing job writes a 'failed' record (traceback tail, 2 KB) and the pool goes on; aggregation then reports it and
    the verdict is INCOMPLETE; a resumed run re-queues failed jobs"""
    panel, rooms, index, seed, opts = job
    try:
        res = run_job(job)
    except Exception:                                                   # noqa: BLE001 (recorded, never swallowed silently)
        res = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, failed=traceback.format_exc()[-2048:],
                   fingerprint=fingerprint(opts))
    path = _job_path(opts['out'], panel, rooms, index, seed); tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return path, res.get('timing', {}).get('total'), res.get('peak_mb'), res.get('skipped'), res.get('failed')


# ================================================================ aggregation and verdict
def _q(v, qs=(50, 90)):
    v = [x for x in v if x is not None]
    return {f'p{q}': (float(np.percentile(v, q)) if v else None) for q in qs} | {'mean': float(np.mean(v)) if v else None, 'n': len(v)}


def _abs(v):
    return [abs(x) for x in v if x is not None]


def edge_summary(edge_lists):
    E = [e for el in edge_lists for e in el]
    out = dict(n_edges=len(E), coarse_accepted=sum(e['coarse_accepted'] for e in E))
    for when in ('err_before', 'err_after'):
        out[when] = dict(yaw_abs_deg=_q(_abs([e[when]['yaw_deg'] for e in E])), disp_pivot_cm=_q([e[when]['disp_pivot_cm'] for e in E]),
                         disp_rms_cm=_q([e[when]['disp_rms_cm'] for e in E]), disp_max_cm=_q([e[when]['disp_max_cm'] for e in E]))
    out['est_abs_yaw_deg'] = _q(_abs([e['est'][0] for e in E]))
    out['improved_rms'] = sum(1 for e in E if e['err_before']['disp_rms_cm'] is not None and e['err_after']['disp_rms_cm'] < e['err_before']['disp_rms_cm'])
    out['worsened_rms'] = sum(1 for e in E if e['err_before']['disp_rms_cm'] is not None and e['err_after']['disp_rms_cm'] > e['err_before']['disp_rms_cm'])
    out['corrected'] = sum(1 for e in E if e['corrected'])
    no_ov = [e for e in E if e.get('overlap_true') is not None and not e['overlap_true']['ok']]
    fl = [e for e in E if e.get('failed')]
    out.update(no_overlap_true=len(no_ov), frac_no_overlap_true=(len(no_ov) / len(E)) if E else None, failed=len(fl),
               failed_with_no_overlap_true=sum(1 for e in fl if not e['overlap_true']['ok']),
               frac_failed_with_no_overlap_true=(sum(1 for e in fl if not e['overlap_true']['ok']) / len(fl)) if fl else None)
    return out


def _row_mean(rows):
    return G.summarise_rows(G._expand(rows))


def check_none_positions(job):
    """review S6: the arms of one tier must have furniture F1 undefined on the SAME component rows, else their means would be taken over
    different row sets"""
    for c, rec in job['conds'].items():
        for row in ROWS + ('aligned_best',):
            pats = {arm: tuple(k for k, (r, _) in enumerate(v[row]) if r.get('furn_f1') is None) for arm, v in rec['arms'].items() if row in v}
            _check(len(set(pats.values())) <= 1, f"furn_f1 undefined on different rows across arms: {job['panel']} {job['house']} s{job['seed']} {c} {row} {pats}")


def _house_sum(rows):
    xs = [r['furn_f1'] for r in G._expand(rows) if r.get('furn_f1') is not None]
    return float(sum(xs)), len(xs)


def _pooled_recovery(parts, arm='reg'):
    """parts: [(house, {arm: (sum, count)})] -> the pooled recovery of `arm` (means over agents and houses, as the verdict)"""
    m = {}
    for a in ('noreg', arm, 'truepose'):
        sm = sum(p[1][a][0] for p in parts); cn = sum(p[1][a][1] for p in parts)
        if not cn:
            return None
        m[a] = sm / cn
    gap = m['truepose'] - m['noreg']
    return (m[arm] - m['noreg']) / gap if gap > 0 else None


def recovery_distribution(J, c, row):
    """review S3: per-house recoveries (median, quartiles), the pooled recovery recomputed from house sums, the gap, and a house-level
    bootstrap 95 % interval (BOOT_B resamples of houses with replacement, fixed seed)"""
    parts = [(j['house'], {a: _house_sum(j['conds'][c]['arms'][a][row]) for a in ('noreg', 'reg', 'truepose')}) for j in J]
    per_house = {h: _pooled_recovery([(h, x)]) for h, x in parts}
    vals = [v for v in per_house.values() if v is not None]
    rng = np.random.default_rng(BOOT_SEED); boots = []
    for _ in range(BOOT_B if parts else 0):
        v = _pooled_recovery([parts[k] for k in rng.integers(0, len(parts), len(parts))])
        if v is not None:
            boots.append(v)
    sums = {a: (sum(x[a][0] for _, x in parts), sum(x[a][1] for _, x in parts)) for a in ('noreg', 'truepose')}
    gap = (sums['truepose'][0] / sums['truepose'][1] - sums['noreg'][0] / sums['noreg'][1]) if parts and sums['noreg'][1] and sums['truepose'][1] else None
    return dict(pooled=_pooled_recovery(parts), gap=gap, per_house=per_house,
                median=float(np.median(vals)) if vals else None, q25=float(np.percentile(vals, 25)) if vals else None,
                q75=float(np.percentile(vals, 75)) if vals else None,
                boot_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                boot_valid_frac=(len(boots) / BOOT_B) if parts else None,
                note='recovery may exceed 1: registration also removes part of each agent\'s per-frame mean pose error')


def summarise_seed(J, e1_status=None):
    """J: the completed jobs of one (panel, seed); e1_status: {house: {tier: status}} from e1_compare_job (None = not compared)"""
    for j in J:
        check_none_positions(j)
    conds = [c for c in G.COND_ORDER if J and all(c in j['conds'] for j in J)]
    D = dict(conds={}, per_house={})
    for c in conds:
        R = [j['conds'][c] for j in J]; has_inter = G.CONDITIONS[c]['inter_t'] > 0 or G.CONDITIONS[c]['inter_yaw'] > 0
        d = dict(arms={}, edges=edge_summary([r['registration']['edges'] for r in R]),
                 E_pose_monotone_all=all(r['registration']['E_pose_monotone'] for r in R),
                 monotone_checks=int(sum(r['registration']['monotone_checks'] for r in R)),
                 ATE_cm_before=_q([r['registration']['agent_err_before']['ATE_cm'] for r in R]),
                 ATE_cm_after=_q([r['registration']['agent_err_after']['ATE_cm'] for r in R]),
                 ARE_deg_before=_q([r['registration']['agent_err_before']['ARE_deg'] for r in R]),
                 ARE_deg_after=_q([r['registration']['agent_err_after']['ARE_deg'] for r in R]))
        if not has_inter:
            d['edges']['note'] = ('true relative pose taken as I; each agent\'s per-frame mean pose error (~1.2 cm / 0.29 deg on cam_nf) makes '
                                  'part of err_after a real correction -- the F1 guard, not err_after, is what counts (review S7)')
        for arm in ARMS + REPORT_ARMS:
            if not all(arm in r['arms'] for r in R):
                continue
            d['arms'][arm] = {row: G.summarise_rows([x for r in R for x in G._expand(r['arms'][arm][row])])
                              for row in ROWS + REPORT_ROWS if all(row in r['arms'][arm] for r in R)}
            d['arms'][arm]['align_G'] = dict(yaw_abs_deg=_q(_abs([g['G'][0] for r in R for g in r['arms'][arm]['align']])),
                                             shift_cm=_q([math.hypot(g['G'][1], g['G'][2]) * 100 for r in R for g in r['arms'][arm]['align']]))
            d['arms'][arm]['aligned_below_asis_components'] = int(sum(r['arms'][arm].get('aligned_below_asis', 0) for r in R))
            ru = collections.Counter(r['arms'][arm].get('reused_from') for r in R if r['arms'][arm].get('reused_from'))
            if ru:
                d['arms'][arm]['reused_from'] = dict(ru)
        if all('observability' in r for r in R):
            d['observability'] = dict(n_islands=_q([r['observability']['n_islands'] for r in R]),
                                      tree_edges_observable=int(sum(r['observability']['tree_edges_observable'] for r in R)),
                                      tree_edges=int(sum(r['observability']['tree_edges'] for r in R)),
                                      covisible_pairs=int(sum(r['observability']['covisible_pairs'] for r in R)),
                                      pairs=int(sum(r['observability']['pairs'] for r in R)))
        if e1_status is not None:
            st = {j['house']: e1_status[j['house']].get(c, 'tier_missing') for j in J}
            d['e1_compare'] = dict(counts=dict(collections.Counter(st.values())), bad=sorted(h for h, v in st.items() if v != 'equal'))
        mc = [r.get('e1_map_check') for r in R if r.get('e1_map_check')]
        if mc:
            d['e1_map_check'] = dict(agents=sum(x['agents'] for x in mc), k0_equal=sum(x['k0_equal'] for x in mc), pub_equal=sum(x['pub_equal'] for x in mc))
        if has_inter:
            d['recovery_dist'] = {row: recovery_distribution(J, c, row) for row in ROWS}
        D['conds'][c] = d
    for j in J:
        H = D['per_house'][j['house']] = {}
        for c in conds:
            rec = j['conds'][c]; A = rec['arms']; E = rec['registration']['edges']
            H[c] = {arm: {row: {k: _row_mean(A[arm][row]).get(k) for k in ('furn_f1', 'free_recall', 'unknown_known_fpr')}
                          for row in ROWS + ('aligned_best',) if row in A[arm]} for arm in ARMS + REPORT_ARMS if arm in A}
            H[c]['edges_rms_cm'] = [[e['err_before']['disp_rms_cm'], e['err_after']['disp_rms_cm']] for e in E]
            H[c]['edges_yaw_deg'] = [[e['err_before']['yaw_deg'], e['err_after']['yaw_deg']] for e in E]
            H[c]['edges_corrected'] = [bool(e['corrected']) for e in E]
            H[c]['edges_overlap_true'] = [bool(e['overlap_true']['ok']) for e in E]
            fl = [e for e in E if e.get('failed')]
            H[c]['frac_edges_no_overlap_true'] = sum(1 for e in E if not e['overlap_true']['ok']) / len(E) if E else None
            H[c]['frac_failed_edges_with_no_overlap_true'] = (sum(1 for e in fl if not e['overlap_true']['ok']) / len(fl)) if fl else None
            H[c]['E_pose'] = rec['registration']['E_pose']
            if 'observability' in rec:
                H[c]['n_islands'] = rec['observability']['n_islands']; H[c]['islands'] = rec['observability']['islands']
    return D, conds


def readouts(C, cond, upper_from, row):
    """recovery and guards on one tier and row; upper = that tier's truepose arm (on cam: = cam_nf noreg); report arms' recoveries
    against the same noreg / truepose"""
    T = C[cond]['arms']; up = T['truepose'][row]['furn_f1']; Fn = T['noreg'][row]['furn_f1']; Fr = T['reg'][row]['furn_f1']
    gap = up - Fn
    r = dict(F_noreg=Fn, F_reg=Fr, F_truepose=up, gap=gap, recovery=((Fr - Fn) / gap) if gap > 0 else None,
             free_delta=T['reg'][row]['free_recall'] - T['noreg'][row]['free_recall'],
             fpr_delta=T['reg'][row]['unknown_known_fpr'] - T['noreg'][row]['unknown_known_fpr'])
    for arm in REPORT_ARMS:
        if arm in T and row in T[arm]:
            r[f'F_{arm}'] = T[arm][row]['furn_f1']
            r[f'recovery_{arm}'] = ((T[arm][row]['furn_f1'] - Fn) / gap) if gap > 0 else None
    if upper_from in C:
        U = C[upper_from]['arms']
        r['nf_delta'] = U['reg'][row]['furn_f1'] - U['noreg'][row]['furn_f1']
        r['F_cam_nf_noreg'] = U['noreg'][row]['furn_f1']
    return r


def _fmt(v, k=3):
    return 'NA' if v is None else f'{v:.{k}f}'


def _crit(a, mono):
    return dict(recovery_ge_85=a['recovery'] is not None and a['recovery'] >= LEVER_RECOVERY, E_pose_monotone=mono,
                cam_nf_not_worse_05pp=a['nf_delta'] >= NF_TOL, free_recall_not_lower=a['free_delta'] >= 0, fpr_not_worse_03pp=a['fpr_delta'] <= FPR_TOL)


def _category(per, lever_key, insuff_key):
    if all(per[s][lever_key] for s in per):
        return 'LEVER'
    if all(per[s][insuff_key] for s in per):
        return 'INSUFFICIENT'
    return 'AMBIGUOUS'


def verdict(seeds, expected, judged=True, extra_reasons=()):
    reasons = list(extra_reasons)
    if sorted(seeds) != ['1', '2']:
        reasons.append(f'noise seeds {sorted(seeds)} != [1, 2]')
    hs = {s: frozenset(D['houses']) for s, D in seeds.items()}; sk = {s: frozenset(D['skipped']) for s, D in seeds.items()}
    if len(set(hs.values())) > 1 or len(set(sk.values())) > 1:
        reasons.append('house sets differ between seeds')
    for s in seeds:
        if hs[s] | sk[s] != expected:
            reasons.append(f'seed {s}: {len(expected - hs[s] - sk[s])} of {len(expected)} houses missing')
    per = {}
    for s, D in seeds.items():
        C = D['conds']
        miss = [c for c in (JUDGED_COND, UPPER_COND) if c not in C or any(a not in C[c]['arms'] or JUDGED_ROW not in C[c]['arms'][a] for a in ARMS)]
        if miss:
            reasons.append(f'seed {s}: tiers / arms {miss} missing'); per[s] = dict(missing=miss); continue
        if judged:                                                      # review M1: the control arm must BE G4-E1's cam sum
            for c in (JUDGED_COND, UPPER_COND):
                e1 = C[c].get('e1_compare')
                if e1 is None:
                    reasons.append(f'seed {s}: {c}: no comparison with G4-E1 (aggregate with --e1-jobs)')
                elif e1['bad']:
                    reasons.append(f"seed {s}: control arm != G4-E1 on {c} ({len(e1['bad'])} jobs, e.g. {e1['bad'][:3]}; {e1['counts']})")
        rows = [row for row in ROWS + ('aligned_best',) if all(row in C[t]['arms'][a] for t in (JUDGED_COND, UPPER_COND) for a in ARMS)]
        x = {row: readouts(C, JUDGED_COND, UPPER_COND, row) for row in rows}
        a = x[JUDGED_ROW]; mono = bool(C[JUDGED_COND]['E_pose_monotone_all'] and C[UPPER_COND]['E_pose_monotone_all'])
        crit = _crit(a, mono)
        per[s] = dict(readouts=x, criteria=crit, lever=all(crit.values()),
                      insufficient=a['recovery'] is not None and a['recovery'] < INSUFF_RECOVERY,
                      note=None if a['recovery'] is not None else 'no gap between truepose and noreg after alignment: recovery undefined',
                      cam_nf_upper_equals_cam_truepose=abs(a['F_cam_nf_noreg'] - a['F_truepose']) < 1e-12,
                      ceilings={k: a.get(f'recovery_{arm}') for k, arm in (('pairwise', 'ceiling_pairwise'), ('islands', 'ceiling_islands'),
                                                                             ('oracle_all', 'oracle_all'))},
                      ceilings_available=all(f'recovery_{arm}' in a for arm in REPORT_ARMS),
                      recovery_dist=C[JUDGED_COND].get('recovery_dist'))
        if 'aligned_best' in x:
            b = x['aligned_best']; cb = _crit(b, mono)
            per[s].update(lever_aligned_best=all(cb.values()), insufficient_aligned_best=b['recovery'] is not None and b['recovery'] < INSUFF_RECOVERY)
    ok = [s for s in per if 'missing' not in per[s]]
    okd = {s: per[s] for s in ok}
    category = _category(okd, 'lever', 'insufficient') if ok else None
    best = (_category(okd, 'lever_aligned_best', 'insufficient_aligned_best')
            if ok and all('lever_aligned_best' in per[s] for s in ok) else None)
    reading = None
    if category == 'INSUFFICIENT':
        if not all(per[s]['ceilings_available'] for s in ok):
            reading = 'ceiling rows unavailable: the reading of INSUFFICIENT is undetermined'
        elif all(per[s]['ceilings']['pairwise'] is not None and per[s]['ceilings']['pairwise'] >= CEILING_READ for s in ok):
            reading = ('classical method insufficient: the pairwise observability ceiling is >= .85 in both seeds -> learned registration '
                       '(G5-B, 2 leases) becomes a candidate')
        else:
            reading = ("relative poses not identifiable from the two agents' evidence (panel observability): learned pairwise registration "
                       'is bounded by the same ceiling; G5-B does NOT become a candidate from this batch (next: a separate pre-registration -- '
                       'explicit Manhattan yaw prior, more overlap, or an observable-only denominator)')
    if reasons:
        v = 'INCOMPLETE: ' + '; '.join(reasons)
    elif category == 'LEVER':
        v = 'LEVER (杠杆): classical pose-stalk registration recovers >= 85 % of the cam_nf - cam gap'
    elif category == 'INSUFFICIENT':
        v = 'INSUFFICIENT (不够): recovery < 50 % in both seeds; reading (D8-39 / D8-40 addendum 2): ' + reading
    else:
        v = 'AMBIGUOUS'
    oracle_ok = None
    if ok and all(per[s]['ceilings_available'] for s in ok):
        v += ('; ceilings (cam, aligned recovery): pairwise ' + ', '.join(f"s{s} {_fmt(per[s]['ceilings']['pairwise'])}" for s in ok)
              + ' | islands ' + ', '.join(f"s{s} {_fmt(per[s]['ceilings']['islands'])}" for s in ok)
              + ' | oracle_all ' + ', '.join(f"s{s} {_fmt(per[s]['ceilings']['oracle_all'])}" for s in ok))
        oracle_ok = all(per[s]['ceilings']['oracle_all'] is not None and abs(per[s]['ceilings']['oracle_all'] - 1.) <= ORACLE_TOL for s in ok)
        if not oracle_ok:
            v += ' [alignment pipeline check failed: oracle_all recovery deviates from 1 by more than .10]'
    if not reasons and best is not None and best != category:
        v += f' [sensitive to alignment: the aligned_best row reads {best}]'
    if not judged:
        v = 'NOT JUDGED (report panel); readout: ' + v
    return dict(judged=judged, complete=not reasons, incomplete_reasons=reasons, judged_row=JUDGED_ROW, judged_cond=JUDGED_COND, per_seed=per,
                category=None if reasons else category, category_aligned_best=best, reading=reading, oracle_check_ok=oracle_ok, verdict=v)


def e4b_entry(val_verdict):
    """D8-39 / D8-40 addendum 3: E4b is implemented and run only if, on VAL, BOTH seeds have recovery_ceiling_islands -
    recovery_ceiling_pairwise >= .15 (cam, aligned)"""
    per = val_verdict.get('per_seed', {}); diffs = {}
    for s, x in per.items():
        r = (x.get('readouts') or {}).get(JUDGED_ROW, {})
        ci, cp = r.get('recovery_ceiling_islands'), r.get('recovery_ceiling_pairwise')
        diffs[s] = (ci - cp) if (ci is not None and cp is not None) else None
    have = sorted(diffs) == ['1', '2'] and all(d is not None for d in diffs.values())
    value = bool(have and all(d >= E4B_MARGIN for d in diffs.values()))
    complete = bool(val_verdict.get('complete')) and have
    return dict(rule='VAL, both seeds: recovery_ceiling_islands - recovery_ceiling_pairwise >= .15 (cam, aligned)', per_seed=diffs,
                value=value, complete=complete, note=None if complete else 'provisional: VAL incomplete or a ceiling undefined')


def aggregate(out_dir, e1_jobs=None):
    paths = sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))
    if not paths:
        raise SystemExit(f'no job files in {out_dir}/jobs')
    jobs = [json.load(open(p)) for p in paths]
    fps = {json.dumps(j.get('fingerprint'), sort_keys=True) for j in jobs}
    if len(fps) != 1:
        raise SystemExit(f'refusing to aggregate {out_dir}: {len(fps)} different job fingerprints (mixed script versions / options)')
    fp = jobs[0]['fingerprint']
    if fp is None or fp.get('job_format') != JOB_FORMAT:
        raise SystemExit(f'refusing to aggregate {out_dir}: job format {None if fp is None else fp.get("job_format")} != {JOB_FORMAT}')
    e1_dir = os.path.abspath(e1_jobs) if e1_jobs else None
    last = lambda t: (t.strip().splitlines() or [''])[-1][:300]  # noqa: E731
    S = dict(fingerprint=fp, e1_jobs=e1_dir, panels={},
             skipped=[dict(panel=j['panel'], house=j['house'], seed=j['seed'], reason=j['skipped']) for j in jobs if 'skipped' in j],
             failed=[dict(panel=j['panel'], house=j['house'], seed=j['seed'], error=last(j['failed'])) for j in jobs if 'failed' in j])
    for panel in sorted({j['panel'] for j in jobs}):
        expected = frozenset(f'r{r}_i{i}' for r, i in G.panel_specs(panel))
        P = dict(seeds={})
        for seed in sorted({j['seed'] for j in jobs if j['panel'] == panel}):
            J_all = [j for j in jobs if j['panel'] == panel and j['seed'] == seed]
            J = [j for j in J_all if 'skipped' not in j and 'failed' not in j]
            e1_status = {j['house']: e1_compare_job(j, e1_dir) for j in J}
            D, conds = summarise_seed(J, e1_status)
            D.update(houses=sorted(j['house'] for j in J), skipped=sorted(j['house'] for j in J_all if 'skipped' in j),
                     failed=sorted(j['house'] for j in J_all if 'failed' in j), n_houses=len(J))
            D['harsh_readouts'] = {row: readouts(D['conds'], 'harsh', 'none', row) for row in ROWS} if 'harsh' in D['conds'] else None
            P['seeds'][str(seed)] = D
        extra = [f"job {panel} {j['house']} s{j['seed']} failed: {last(j['failed'])}" for j in jobs if j['panel'] == panel and 'failed' in j]
        P['verdict'] = verdict(P['seeds'], expected, judged=(panel == 'val'), extra_reasons=extra)
        S['panels'][panel] = P
    val = S['panels'].get('val', {}).get('verdict', {})
    S['decision'] = dict(judged_panel='val', judged_cond=JUDGED_COND, judged_row=JUDGED_ROW, verdict=val.get('verdict'), reading=val.get('reading'),
                         category=val.get('category'), category_aligned_best=val.get('category_aligned_best'), oracle_check_ok=val.get('oracle_check_ok'),
                         e4b_entry=e4b_entry(val) if val else None, fresh=S['panels'].get('fresh', {}).get('verdict', {}).get('verdict'))
    return S


# ================================================================ main
def main(argv=None):
    if not __debug__:                                                   # review S4
        raise SystemExit('refusing to run under python -O: the pre-registered checks of the frozen G4-E1 code are asserts')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--panels', default='val,fresh')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--houses', type=int, default=0, help='first N houses of each panel (0 = all 24)')
    ap.add_argument('--conds', default=','.join(CONDS_DEFAULT))
    ap.add_argument('--gauge', default='prior', choices=('prior', 'root'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--e1-jobs', default='', help='G4-E1 jobs dir: at aggregation, compare every job\'s as-is noreg / truepose rows with '
                                                  'its V3 sum rows (VAL cam / cam_nf: any mismatch or missing file -> INCOMPLETE)')
    ap.add_argument('--check-e1-maps', action='store_true', help='recompute every map with G4-E1 4-integrator call and compare (smoke)')
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
    conds = [c for c in G.COND_ORDER if c in conds]                    # cam_nf before cam (shared agent-frame maps)
    opts = dict(out=args.out, conds=conds, gauge=args.gauge, match=dict(MATCH), job_format=JOB_FORMAT, governing_integrator='V3',
                script_sha256=file_sha256(os.path.abspath(__file__)), g4_noise_panels_sha256=file_sha256(G.__file__),
                data_module_sha256=file_sha256(W.__file__), houseexpo_index_sha256=file_sha256(args.houseexpo_index),
                check_e1_maps=bool(args.check_e1_maps))
    fp = fingerprint(opts)
    t0 = time.time()
    if not args.aggregate_only:
        jobs = []
        for panel in args.panels.split(','):
            specs = G.panel_specs(panel)
            if args.houses:
                specs = specs[:args.houses]
            for seed in [int(s) for s in args.seeds.split(',')]:
                for rooms, index in specs:
                    path = _job_path(args.out, panel, rooms, index, seed)
                    if os.path.exists(path):
                        old = json.load(open(path))
                        if old.get('fingerprint') != fp:
                            raise SystemExit(f'{path} was written with another script version / options; use a fresh --out')
                        if 'failed' not in old:                         # completed or deterministically skipped: keep
                            continue
                    jobs.append((panel, rooms, index, seed, opts))
        print(f'{len(jobs)} jobs, {args.workers} workers', flush=True)
        if jobs:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.houseexpo_json_dir, args.houseexpo_index), maxtasksperchild=1) as pool:
                for i, (path, sec, mb, sk, fl) in enumerate(pool.imap_unordered(_run_and_save, jobs), 1):
                    print(f'[{i}/{len(jobs)}] {os.path.basename(path)} {sec if sec is None else round(sec)} s peak {mb if mb is None else round(mb)} MB'
                          + (f' SKIPPED {sk}' if sk else '') + (' FAILED: ' + fl.strip().splitlines()[-1][:200] if fl else ''),
                          f'{time.time() - t0:.0f}s', flush=True)
    S = aggregate(args.out, args.e1_jobs or None)
    if S['fingerprint'] != fp:
        print('WARNING: the job files were written by another script version / options than this invocation; summary reflects the job files', flush=True)
    S['args'] = vars(args)
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1)
    meta = dict(script=os.path.basename(__file__), fingerprint=fp, args=vars(args), conditions={c: G.CONDITIONS[c] for c in conds}, seconds=time.time() - t0)
    with open(os.path.join(args.out, 'aggregate.json' if args.aggregate_only else 'run.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    f2 = lambda v, k=2: 'NA' if v is None else f'{v:.{k}f}'  # noqa: E731
    for panel, P in S['panels'].items():
        print(panel, P['verdict']['verdict'], flush=True)
        for s, D in P['seeds'].items():
            for c, d in D['conds'].items():
                A = d['arms']; e = d['edges']
                line = ' | '.join(f"{arm} " + '/'.join(f"{A[arm][row]['furn_f1']:.4f}" for row in ROWS) for arm in ARMS + REPORT_ARMS if arm in A)
                print(f'  seed {s} {c:7s} R8 furn asis/aligned: {line} | edge rms cm {f2(e["err_before"]["disp_rms_cm"]["p50"])} -> '
                      f'{f2(e["err_after"]["disp_rms_cm"]["p50"])}, |yaw| deg {f2(e["err_before"]["yaw_abs_deg"]["p50"], 3)} -> '
                      f'{f2(e["err_after"]["yaw_abs_deg"]["p50"], 3)} (median of {e["n_edges"]}; corrected {e["corrected"]}, worse {e["worsened_rms"]}, '
                      f'no overlap at the true pose {e["no_overlap_true"]}, failed {e["failed"]} of which no overlap {e["failed_with_no_overlap_true"]}) | '
                      f'E_pose monotone {d["E_pose_monotone_all"]} ({d["monotone_checks"]} checks) | G4-E1 rows {(d.get("e1_compare") or {}).get("counts")}'
                      + (f' | islands {d["observability"]["n_islands"]["mean"]:.2f}/house' if 'observability' in d else ''), flush=True)
            x = P['verdict']['per_seed'].get(s, {})
            if 'missing' not in x and x:
                for row in ROWS + ('aligned_best',):
                    if row not in x['readouts']:
                        continue
                    r = x['readouts'][row]
                    print(f"  seed {s} cam {row:12s}: recovery {f2(r['recovery'], 3)} (noreg {r['F_noreg']:.4f} reg {r['F_reg']:.4f} truepose {r['F_truepose']:.4f}; "
                          f"gap {r['gap']:+.4f}) oracle_all {f2(r.get('recovery_oracle_all'), 3)} ceiling_pairwise {f2(r.get('recovery_ceiling_pairwise'), 3)} "
                          f"ceiling_islands {f2(r.get('recovery_ceiling_islands'), 3)} | cam_nf reg-noreg {r['nf_delta']:+.4f} free {r['free_delta']:+.4f} "
                          f"fpr {r['fpr_delta']:+.5f}", flush=True)
                rd = (x.get('recovery_dist') or {}).get(JUDGED_ROW)
                if rd:
                    print(f"  seed {s} cam aligned per house: median {f2(rd['median'], 3)} [q25 {f2(rd['q25'], 3)}, q75 {f2(rd['q75'], 3)}], "
                          f"bootstrap 95 % {rd['boot_ci95']}", flush=True)
    print('decision:', json.dumps({k: v for k, v in S['decision'].items() if k != 'verdict'}), flush=True)
    if args.smoke_asserts:
        for panel, P in S['panels'].items():
            for s, D in P['seeds'].items():
                C = D['conds']
                _check(not D['failed'], f'failed jobs {D["failed"]}')
                for c, d in C.items():
                    _check(d['E_pose_monotone_all'] and d['monotone_checks'] > 0, f'{c}: E_pose monotone assertion did not run')
                    if 'e1_map_check' in d:
                        mc = d['e1_map_check']
                        _check(mc['k0_equal'] == mc['agents'] and mc['pub_equal'] == mc['agents'], f'{c}: maps differ from G4-E1 {mc}')
                    if args.e1_jobs:
                        _check(d['e1_compare']['counts'] == {'equal': D['n_houses']}, f"{c}: rows differ from G4-E1 {d['e1_compare']}")
                if 'cam' in C and 'cam_nf' in C:
                    _check(abs(C['cam']['arms']['truepose']['asis']['furn_f1'] - C['cam_nf']['arms']['noreg']['asis']['furn_f1']) < 1e-12, 'cam truepose != cam_nf noreg')
                    ea, eb = C['cam']['edges']['err_after'], C['cam']['edges']['err_before']
                    _check(ea['disp_rms_cm']['p50'] < eb['disp_rms_cm']['p50'], 'cam: per-edge pose error did not fall')
                    ro = P['verdict']['per_seed'][s]['readouts'][JUDGED_ROW].get('recovery_oracle_all')
                    _check(ro is not None and abs(ro - 1.) <= ORACLE_TOL, f'oracle_all recovery {ro} is not ~1: the alignment pipeline is not fair')
        print('smoke asserts ok')
    return S


if __name__ == '__main__':
    main()
