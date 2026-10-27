"""g4_calib_stalk.py (Claude, R-A, 2026-10-27): batch G4-E1c "计数标定的证据 stalk", pre-registered in
claude_handoff/OVERNIGHT_REPORT_20261021.md, D8-42 (criteria there are authoritative; nothing here may change them). CPU only, numpy /
scipy, no torch work, no GPU. Imports the FROZEN g4_noise_panels.py (G4-E1, sha 2313d75...) and never modifies it.

Single variable: the evidence function. B4 (control) = G4-E1's V3 forest sum, occupied iff sum_a L_a > 0. B6 (arm) = occupied iff
l0 + sum_a c(L_a) > t, where c(L) = log P(L | occupied) / P(L | free) is ONE public 111-entry table over the clamped V3 alphabet
L in [-40, 70] (+0.5 smoothing), l0 is a global prior logit added ONCE at the read-out (never inside a stalk), and t is a threshold.
Same V3 maps (asserted to reproduce the G4-E1 job files' K0 rows AND R8 sum rows value for value; any mismatch -> INCOMPLETE), same
radio forest, same relay with E1's provenance assertions (G.relay / G.comb_sum), same scoring (G.score / G.Target / G.House).

Integer wire: the table is published quantised to the E1 log-odds unit (0.05): cq(L) = round(c(L) / 0.05) (int), l0q = round(l0 / 0.05),
and the arm decides  l0q + sum_a cq(L_a) > tq  <=>  sum_a cq(L_a) > theta,  theta = tq - l0q (all integers, exact relay sums).
K0 (one agent): l0q + cq(L_a) > tq, i.e. the prior is counted once for one agent as for eight.

Fit (TRAIN only; TEST is never generated, VAL / fresh never touched): two disjoint TRAIN halves from scene_plan('TRAIN', 70):
half A = even index, half B = odd index, the first --train-per-stratum of each stratum (rooms 1 / 2 / 3); a half needs >= 12 fitted
houses (else INCOMPLETE). Noise seeds 101, 102 (disjoint from 1, 2, 7). Tables are fitted on AGENT-FRAME maps (= the sensor's own
likelihood; the inter-agent error is the registration layer's business): cam_nf (governing; also used for clean and cam), lidar and
harsh (report rows; their own agent-frame maps). Per agent, every voxel the agent knows inside its component's clean-known scope
(R8 target known) adds one count to (L, truth) -> c(L). l0 = logit of the occupied fraction among the voxels of the scope known to >= 1
agent of the component. t: on the half, pooled (component-expanded, as E1's furn_f1_pooled) furniture F1 maximal subject to pooled free
recall >= that of the B4 rule sum L > 0 on the same half (threshold sweep over EVERY integer theta in [-4097, 4095] units, i.e. the
full +-4096-unit histogram range, through histograms; unit-tested equal to G.score; values outside the range and a chosen theta on
its boundary are asserted against). Report-only row W: occupied iff l0 + w(k) * sum c > t_w with k = number of the component's agents knowing the voxel
(k = 1..4, >= 5; relayed as a second sum, sum 1), w(k) = 1-D logistic fit y ~ sigmoid(l0 + w_k * sum c) on the half, t_w by the same rule.
Fit artefacts (tables, l0, t, w, t_w, per half and tier) go to <fit-out>/fit.json with its sha256; eval jobs record that sha.

Eval: VAL (scene_plan('VALIDATION', 16) index < 8, judged) and fresh (index 8-15, report) x noise seeds 1, 2; tiers clean, cam_nf, cam,
lidar, harsh (table per tier: clean / cam_nf / cam -> cam_nf table; lidar -> lidar; harsh -> harsh); both halves' tables per tier.

Verdict (D8-42; VAL, cam_nf, R8 furniture F1 = E1's mean over component-expanded rows; cells = 2 halves x 2 noise seeds):
  LEVER: in all 4 cells B6 - B4 >= +3 pp AND FPR(B6) <= FPR(B4) + 0.3 pp AND free recall(B6) >= free recall(B4) AND R8 surface F1(B6)
         >= B4 - 0.5 pp; AND clean tier (both halves' cam_nf tables) furniture >= .99 and free recall >= .99 (R8 B6, every seed).
  NOT_ENOUGH: any cell < +1 pp.  Otherwise AMBIGUOUS.
  fresh: report; if the sign of the 4-cell mean B6 - B4 on fresh is opposite to VAL's, a LEVER / NOT_ENOUGH verdict is downgraded to
  AMBIGUOUS (coordinator's reading of "符号与 VAL 相反时降为 ambiguous": the mean over the same 4 cells).
  INCOMPLETE (review M1 / M2 / S6): a missing VAL or fresh cell (both panels need both noise seeds, EXACTLY the 24-house plan each
  -- G4-E1 skipped none, so any skip counts --, and both halves' B6); a failed job; an E1 reproduction mismatch; fit provenance:
  a smoke fit.json, a half with < 12 houses, fit seeds not disjoint from {1, 2, 7}, a failed fit job, halves' world ids missing or
  overlapping, fit sha != the sha recorded in the eval jobs, fit script sha != the eval jobs' script sha (aggregate without --fit
  takes the fit recorded in <out>/eval_run.json). Non-smoke eval refuses such a fit.json before running any job.
Report only: K0 furniture; cam / lidar / harsh; the W row; the B4t row (sum L > theta4, theta4 chosen on the same TRAIN half by the
same rule: separates the table's gain from the threshold's, review S1); calibration slopes by k (logistic fit with intercept of truth on the
read-out logit, VAL scope = known to >= 1 agent and clean-known, binned); the two halves' table difference; bytes (own maps are the
same L as B4 -> same bytes; relay = B4 integer coder on the relayed sum cq messages, cam_nf; B4's relay bytes from the E1 job file).

Failures: a job that raises writes {"failed": true, "error": ...} and the run continues (as E4's S1); aggregation counts it missing.
Data hygiene: the TRAIN map cache (--cache) holds derived voxel maps; keep it OUTSIDE the public repo (default: %TEMP%/g4_e1c_cache).

Usage (always python -B, CUDA_VISIBLE_DEVICES= ; <= 4 workers on the shared box):
  python -B g4_calib_stalk.py fit  --out DIR [--train-per-stratum 5] [--fit-seeds 101,102] [--workers 4] [--cache C]
  python -B g4_calib_stalk.py eval --out DIR --fit DIR/fit.json [--panels val,fresh] [--seeds 1,2] [--houses N] [--workers 4]
  python -B g4_calib_stalk.py aggregate --out DIR
Smoke: fit --train-houses A:1:60,B:2:61 --fit-seeds 7 --smoke ; eval --panels val --houses 2 --seeds 1 --smoke
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, collections, glob, hashlib, json, math, multiprocessing, tempfile, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_noise_panels as G  # noqa: E402   (frozen; read only)

W = G.W
ROOT = G.ROOT
E1_JOBS = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e1', 'jobs')
UNIT = G.LO_UNIT
LMIN, LMAX = G.L_MIN, G.L_MAX
NL = LMAX - LMIN + 1                                    # 111
SMOOTH = 0.5
FIT_TIERS = ('cam_nf', 'lidar', 'harsh')
EVAL_TIERS = ('clean', 'cam_nf', 'cam', 'lidar', 'harsh')
TABLE_OF = {'clean': 'cam_nf', 'cam_nf': 'cam_nf', 'cam': 'cam_nf', 'lidar': 'lidar', 'harsh': 'harsh'}
HALVES = ('A', 'B')
KMAX = 5
VOFF = 4096; VB = 2 * VOFF                              # integer value histograms, clipped to [-4096, 4095]
MIN_FIT_HOUSES = 12
JUDGED = 'cam_nf'
JOB_FORMAT = 1


def file_sha256(path):
    return G.file_sha256(path)


# ================================================================ table, read-out, histograms
def hist_L(L, scope, y):
    """counts of the clamped V3 values of ONE agent over the voxels it knows inside scope: (n_occ[111], n_free[111]); UNK ignored"""
    L = np.asarray(L); k = (~G.is_unk(L)) & scope
    v = L[k].astype(np.int64) - LMIN
    assert v.size == 0 or (v.min() >= 0 and v.max() < NL), 'value outside the clamped alphabet'
    yy = y[k]
    return np.bincount(v[yy], minlength=NL).astype(np.int64), np.bincount(v[~yy], minlength=NL).astype(np.int64)


def fit_table(n_occ, n_free):
    """c(L) = log P(L | occ) / P(L | free) with +0.5 smoothing per cell (finite everywhere)"""
    n_occ = np.asarray(n_occ, np.float64); n_free = np.asarray(n_free, np.float64)
    po = (n_occ + SMOOTH) / (n_occ.sum() + SMOOTH * NL); pf = (n_free + SMOOTH) / (n_free.sum() + SMOOTH * NL)
    return np.log(po) - np.log(pf)


def quantise(c):
    return np.rint(np.asarray(c, np.float64) / UNIT).astype(np.int64)


def evidence(L, cq):
    """one agent's stalk: (cq(L) on known voxels else 0 as int32, known mask). The prior is NOT in the stalk."""
    L = np.asarray(L); k = ~G.is_unk(L)
    e = np.zeros(L.shape, np.int32)
    e[k] = np.asarray(cq, np.int64)[L[k].astype(np.int64) - LMIN]
    return e, k


def readout(S, K, theta):
    """ternary belief of the arm: known iff K, occupied iff sum cq > theta (= l0q + sum cq > tq)"""
    return np.where(K, (S > theta).astype(np.int8), np.int8(-1)).astype(np.int8)


def readout_w(S, K, kc, wk, l0q, tq):
    """report row W: occupied iff l0q + w(k) * sum cq > tq, k = min(#agents knowing, 5)"""
    kk = np.clip(kc, 1, KMAX) - 1
    w = np.asarray(wk, np.float64)[kk]
    return np.where(K, ((l0q + w * S) > tq).astype(np.int8), np.int8(-1)).astype(np.int8)


def _vbin(x):
    return np.clip(np.asarray(x, np.int64), -VOFF, VOFF - 1) + VOFF


def comp_sums(maps, cm, cq=None):
    """direct component sums (fit side): sum L (int), sum cq (int, if cq), known union, knowing-agent count"""
    shape = maps[cm[0]].shape
    S4 = np.zeros(shape, np.int32); S6 = np.zeros(shape, np.int32); K = np.zeros(shape, bool); kc = np.zeros(shape, np.int32)
    for a in cm:
        L = maps[a]; k = ~G.is_unk(L)
        S4[k] += L[k].astype(np.int32); K |= k; kc += k
        if cq is not None:
            S6 += evidence(L, cq)[0]
    return S4, S6, K, kc


def thr_hists(S, K, kc, T, H, weight):
    """per k (1..5+) value histograms of the three categories that determine pooled furniture tp / fp and free hits for ANY threshold:
    A = predicted-known, clean-known, furniture-region target voxels; B = predicted-known, clean-known, region, non-target; C = clean-free
    and predicted-known. Pooled with the component weight (= E1's component expansion). Also the totals |tr| and |free|."""
    out = np.zeros((3, KMAX, VB), np.int64)
    assert not K.any() or (S[K].min() >= -VOFF and S[K].max() < VOFF), 'value outside the histogram range'   # review S3
    kk = np.clip(kc, 1, KMAX) - 1
    A = K & T.known & H.region & T.tr
    B = K & T.known & H.region & ~T.tr
    C = K & T.free
    for i, m in enumerate((A, B, C)):
        np.add.at(out[i], (kk[m], _vbin(S[m])), weight)
    return out, int(T.tr.sum()) * weight, int(T.free.sum()) * weight


def cal_hists(S, K, kc, T):
    """per k value histograms of (truth occupied, truth free) over the scope K & clean-known (weight 1): calibration and w(k)"""
    out = np.zeros((2, KMAX, VB), np.int64)
    assert not K.any() or (S[K].min() >= -VOFF and S[K].max() < VOFF), 'value outside the histogram range'   # review S3
    kk = np.clip(kc, 1, KMAX) - 1; m = K & T.known; y = T.t == 1
    np.add.at(out[0], (kk[m & y], _vbin(S[m & y])), 1)
    np.add.at(out[1], (kk[m & ~y], _vbin(S[m & ~y])), 1)
    return out


def counts_at(th, tr_tot, free_tot, theta):
    """pooled (tp, fp, fn, free_hit, free_tot) of 'occupied iff value > theta' from thr_hists (summed over k)"""
    A, B, C = (th[i].sum(0) for i in range(3))
    j = int(np.clip(theta + VOFF + 1, 0, VB))                  # bins with value > theta
    tp = int(A[j:].sum()); fp = int(B[j:].sum()); fhit = int(C[:j].sum())
    return tp, fp, tr_tot - tp, fhit, free_tot


def counts_at_w(th, tr_tot, free_tot, wk, l0q, tq):
    """same for the W row: per k, value v occupied iff l0q + w_k v > tq  <=>  v > (tq - l0q) / w_k  (w_k > 0)"""
    tp = fp = fhit = 0
    vals = np.arange(VB) - VOFF
    for k in range(KMAX):
        occ = (l0q + wk[k] * vals) > tq
        tp += int(th[0, k][occ].sum()); fp += int(th[1, k][occ].sum()); fhit += int(th[2, k][~occ].sum())
    return tp, fp, tr_tot - tp, fhit, free_tot


def f1_of(c):
    tp, fp, fn = c[:3]
    return 2 * tp / max(2 * tp + fp + fn, 1)


def sweep(th, tr_tot, free_tot):
    """all thresholds at once: theta scanned over every integer in [-4097, 4095] (= -VOFF-1 .. VOFF-1, i.e. the full +-4096-unit
    histogram range, +-204.8 natural log-odds); pooled furniture F1 and free recall of 'occupied iff value > theta'"""
    A, B, C = (th[i].sum(0) for i in range(3))
    sa = np.r_[np.cumsum(A[::-1])[::-1], 0]; sb = np.r_[np.cumsum(B[::-1])[::-1], 0]; pc = np.r_[0, np.cumsum(C)]
    thetas = np.arange(-VOFF - 1, VOFF); j = thetas + VOFF + 1                    # bins with value > theta start at j
    tp = sa[j]; fp = sb[j]; fn = tr_tot - tp; fhit = pc[j]
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return thetas, f1, fhit / max(free_tot, 1)


def select_theta(th6, th4, tr_tot, free_tot):
    """theta maximising pooled furniture F1 of the arm subject to pooled free recall >= that of the B4 rule (sum L > 0);
    ties -> the smallest theta"""
    ref = counts_at(th4, tr_tot, free_tot, 0)
    fr_ref = ref[3] / max(ref[4], 1)
    thetas, f1, fr = sweep(th6, tr_tot, free_tot)
    ok = fr + 1e-15 >= fr_ref
    assert ok.any(), 'no threshold satisfies the free-recall constraint'
    i = int(np.flatnonzero(ok)[np.argmax(f1[ok])])
    return dict(theta=int(thetas[i]), f1=float(f1[i]), free=float(fr[i]), b4_f1=f1_of(ref), b4_free=fr_ref)


def logistic(x, y1, y0, offset=0., intercept=True):
    """weighted 1-D logistic fit on binned data: P(y=1) = sigmoid(a + b x + offset); returns (a, b) (a = 0 without intercept)"""
    from scipy.optimize import minimize
    x = np.asarray(x, np.float64); n1 = np.asarray(y1, np.float64); n0 = np.asarray(y0, np.float64)
    u = (n1 + n0) > 0; x, n1, n0 = x[u], n1[u], n0[u]; N = (n1 + n0).sum()
    if N == 0 or n1.sum() == 0 or n0.sum() == 0:
        return None, None
    def f(p):
        a = p[0] if intercept else 0.; b = p[-1]
        z = a + b * x + offset
        ll = (n1 * np.logaddexp(0, -z) + n0 * np.logaddexp(0, z)).sum() / N
        q = 1 / (1 + np.exp(-z)); g = (q * (n1 + n0) - n1) / N
        return ll, (np.array([g.sum(), (g * x).sum()]) if intercept else np.array([(g * x).sum()]))
    r = minimize(f, np.array([0., 1.]) if intercept else np.array([1.]), jac=True, method='L-BFGS-B')
    return (float(r.x[0]) if intercept else 0.), float(r.x[-1])


def slopes_by_k(cal, unit_offset_q=0):
    """calibration slope / intercept of truth on the read-out logit (l0 offset in units + value) * UNIT, per k and pooled"""
    vals = (np.arange(VB) - VOFF + unit_offset_q) * UNIT
    out = {}
    for k in range(KMAX):
        a, b = logistic(vals, cal[0, k], cal[1, k])
        out[f'k{k + 1}' if k < KMAX - 1 else f'k{KMAX}+'] = dict(n=int(cal[:, k].sum()), occ=int(cal[0, k].sum()), intercept=a, slope=b)
    a, b = logistic(vals, cal[0].sum(0), cal[1].sum(0))
    out['all'] = dict(n=int(cal.sum()), intercept=a, slope=b)
    return out


# ================================================================ house context and maps (= G4-E1 run_job, V3 only)
def house_ctx(split, rooms, index):
    fx = W.make_fixture(split, index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    n = private.shape[0]
    H = G.House(occ, fx['slab'], fx['obj'], cell)
    F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    for a in range(n):
        assert set(np.nonzero(reached8[a])[0].tolist()) == set(F['comps'][F['comp_of'][a]]), 'reached8 != component'
    T8 = G.stream_targets(private, reached8)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    ds = 0.5 * float(cell.min())
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    return dict(fx=fx, occ=occ, cell=cell, private=private, traj=traj, positions=fx['positions_m'], n=n, shape=occ.shape, ds=ds, H=H, F=F,
                T8=T8, K0T=K0T, R8T=R8T, rays=rays, traces=traces, rooms=int(rooms), index=int(index), world_id=fx['world_id'])


def tier_maps(ctx, cname, seed, k0cache, pub=True):
    """V3 agent-frame (K0) and public-frame maps of one tier, exactly as G4-E1 run_job (common random numbers, same calls);
    pub=False (fit stage, review S8) skips the public-frame integration and returns L_pub = None"""
    c = G.CONDITIONS[cname]; has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0; ck = G._ck(c)
    L_k0, L_pub = [], []
    for a in range(ctx['n']):
        frame = None
        if has_inter:
            z = G.frame_draws(seed, ctx['rooms'], ctx['index'], a)
            frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(ctx['positions'][a], np.float64))
        scans = []
        for j in range(ctx['traj'].shape[0]):
            hs, ls = ctx['traces'][a][j]
            scan, _ = G.measure(c, hs, ls, ctx['traj'][j, a], G.pose_draws(seed, ctx['rooms'], ctx['index'], a, j, len(ctx['rays'])), ctx['ds'], ctx['rays'])
            scans.append(scan)
        if (ck, a) not in k0cache:
            k0cache[(ck, a)] = G.integrate_agent(ctx['shape'], ctx['cell'], ctx['ds'], scans, ('V3',))['V3']
        mk = k0cache[(ck, a)]
        if pub:
            mp = G.integrate_agent(ctx['shape'], ctx['cell'], ctx['ds'], [G.to_public(s, frame) for s in scans], ('V3',))['V3'] if has_inter else mk
            L_pub.append(mp)
        L_k0.append(mk)
    return L_k0, (L_pub if pub else None), has_inter


# ================================================================ fit stage (TRAIN)
def default_cache():
    return os.path.join(tempfile.gettempdir(), 'g4_e1c_cache')


def train_specs(per_stratum, explicit=None):
    """[(half, rooms, index)]: half A = even, B = odd TRAIN index of scene_plan('TRAIN', 70), first per_stratum per stratum"""
    if explicit:
        out = []
        for tok in explicit.split(','):
            h, r, i = tok.split(':'); assert h in HALVES and (int(i) % 2 == (0 if h == 'A' else 1)), tok
            out.append((h, int(r), int(i)))
        return out
    plan = W.scene_plan('TRAIN', 70)
    assert all(s['split'] == 'TRAIN' for s in plan)
    out = []
    for h, par in (('A', 0), ('B', 1)):
        for rooms in (1, 2, 3):
            idx = sorted(s['index'] for s in plan if s['rooms'] == rooms and s['index'] % 2 == par)
            out += [(h, rooms, i) for i in idx[:per_stratum]]
    return out


def run_fit_job(job):
    half, rooms, index, seeds, opts = job
    t0 = time.time()
    rec = dict(stage='fit', half=half, rooms=rooms, index=index, seeds=list(seeds), script_sha256=opts['script_sha256'])
    try:
        ctx = house_ctx('TRAIN', rooms, index)
    except ValueError as e:
        rec['skipped'] = repr(e)[:200]; return rec
    rec['world_id'] = ctx['world_id']; rec['tiers'] = {}
    cache = {}
    for seed in seeds:
        k0cache = {}
        for tn in FIT_TIERS:
            L_k0, _, _ = tier_maps(ctx, tn, seed, k0cache, pub=False)
            no = np.zeros(NL, np.int64); nf = np.zeros(NL, np.int64); p_occ = p_free = 0
            for ci, cm in enumerate(ctx['F']['comps']):
                T = ctx['R8T'][ci]; y = T.t == 1
                for a in cm:
                    o, f = hist_L(L_k0[a], T.known, y); no += o; nf += f
                _, _, K, _ = comp_sums(L_k0, cm)
                p_occ += int((K & T.known & y).sum()); p_free += int((K & T.known & ~y).sum())
            rec['tiers'][f'{tn}|{seed}'] = dict(n_occ=no.tolist(), n_free=nf.tolist(), prior_occ=p_occ, prior_free=p_free)
            cache[f'{tn}_{seed}'] = np.stack(L_k0).astype(np.int8)
    os.makedirs(opts['cache'], exist_ok=True)
    np.savez_compressed(os.path.join(opts['cache'], f'TRAIN_r{rooms}_i{index}.npz'), private=ctx['private'], occ=ctx['occ'], slab=ctx['fx']['slab'],
                        obj=ctx['fx']['obj'], cell=ctx['cell'], comps=np.array([ctx['F']['comp_of'][a] for a in range(ctx['n'])]),
                        script_sha256=np.array(opts['script_sha256']), seeds=np.array(list(seeds), np.int64), **cache)   # review S4
    rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    return rec


def _load_cache(path, script_sha=None, seeds=None):
    z = np.load(path)
    if script_sha is not None:                                          # review S4: the cache must come from this script and these seeds
        assert str(z['script_sha256']) == script_sha and sorted(z['seeds'].tolist()) == sorted(seeds), f'stale TRAIN cache {path}'
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell'])
    comp_of = z['comps']; priv = z['private']
    comps = [list(np.nonzero(comp_of == c)[0]) for c in sorted(set(comp_of.tolist()))]
    R8T = [G.Target(priv[cm].max(0), H, mult=(priv[cm] >= 0).sum(0)) for cm in comps]
    return z, H, comps, R8T


def fit_reasons(fit, smoke=False, fit_sha=None, job_fit_shas=None, job_script_shas=None):
    """review M1: provenance problems of a fit.json (empty list = usable). Non-smoke: not a smoke fit, both halves >= 12 houses,
    noise seeds non-empty and disjoint from {1, 2, 7}, no failed fit job, halves' world ids present and disjoint, every fit tier
    present. Always: fit sha == the (single) fit sha recorded in the eval jobs, fit script sha == the (single) eval script sha."""
    if fit is None:
        return ['no fit.json']
    r = []
    halves = fit.get('halves') or {}
    if not smoke:
        if fit.get('smoke', True):
            r.append('fit.json is a smoke fit (or does not record the smoke flag)')
        for h in HALVES:
            n = (halves.get(h) or {}).get('n_houses', 0)
            if n < MIN_FIT_HOUSES:
                r.append(f'fit half {h}: {n} houses < {MIN_FIT_HOUSES}')
        seeds = set(fit.get('seeds') or [])
        if not seeds or seeds & {1, 2, 7}:
            r.append(f'fit noise seeds {sorted(seeds)} empty or not disjoint from 1, 2, 7')
        if fit.get('failed'):
            r.append(f"failed fit jobs: {fit['failed']}")
        wa = set((halves.get('A') or {}).get('world_ids') or []); wb = set((halves.get('B') or {}).get('world_ids') or [])
        if not wa or not wb or (wa & wb):
            r.append('fit halves: world ids missing or not disjoint')
        if any(tn not in ((halves.get(h) or {}).get('tiers') or {}) for h in HALVES for tn in FIT_TIERS):
            r.append('fit tiers missing')
    if job_fit_shas is not None and set(job_fit_shas) != {fit_sha}:
        r.append(f'fit.json sha {str(fit_sha)[:12]} != the fit sha(s) in the eval jobs {[str(x)[:12] for x in sorted(job_fit_shas)]}')
    if job_script_shas is not None and set(job_script_shas) != {fit.get('script_sha256')}:
        r.append("fit.json script sha != the eval jobs' script sha")
    return r


def fit_aggregate(out_dir, opts):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, 'fit_jobs', '*.json')))]
    for r in recs:                                                      # review M1: every fit job from this script and these seeds
        if r.get('script_sha256') != opts['script_sha256'] or sorted(r.get('seeds', [])) != sorted(opts['fit_seeds']):
            raise SystemExit(f"fit job {r.get('half')}:r{r.get('rooms')}_i{r.get('index')} was written by another script / seed set; use a fresh --out")
    res = dict(script_sha256=opts['script_sha256'], noise_panels_sha256=file_sha256(G.__file__), seeds=opts['fit_seeds'], smoke=bool(opts['smoke']),
               train_per_stratum=opts.get('train_per_stratum'), train_houses_explicit=opts.get('train_houses') or None, fit_frame='agent',
               halves={}, failed=[], skipped=[])
    for r in recs:
        if r.get('failed'):
            res['failed'].append(f"{r['half']}:r{r['rooms']}_i{r['index']}")
        elif r.get('skipped'):
            res['skipped'].append(f"{r['half']}:r{r['rooms']}_i{r['index']}")
    for h in HALVES:
        hr = [r for r in recs if r['half'] == h and not r.get('failed') and not r.get('skipped')]
        if not hr:
            res['halves'][h] = dict(houses=[], world_ids=[], n_houses=0, tiers={}); continue
        H_out = dict(houses=[f"r{r['rooms']}_i{r['index']}" for r in hr], world_ids=sorted(r['world_id'] for r in hr), n_houses=len(hr), tiers={})
        for tn in FIT_TIERS:
            no = sum(np.asarray(r['tiers'][f'{tn}|{s}']['n_occ'], np.int64) for r in hr for s in opts['fit_seeds'])
            nf = sum(np.asarray(r['tiers'][f'{tn}|{s}']['n_free'], np.int64) for r in hr for s in opts['fit_seeds'])
            po = sum(r['tiers'][f'{tn}|{s}']['prior_occ'] for r in hr for s in opts['fit_seeds'])
            pf = sum(r['tiers'][f'{tn}|{s}']['prior_free'] for r in hr for s in opts['fit_seeds'])
            c = fit_table(no, nf); cq = quantise(c); l0 = float(np.log(po / pf)); l0q = int(np.rint(l0 / UNIT))
            th6 = np.zeros((3, KMAX, VB), np.int64); th4 = np.zeros_like(th6); cal6 = np.zeros((2, KMAX, VB), np.int64); trt = frt = 0
            for r in hr:
                z, Ho, comps, R8T = _load_cache(os.path.join(opts['cache'], f"TRAIN_r{r['rooms']}_i{r['index']}.npz"), opts['script_sha256'], opts['fit_seeds'])
                for s in opts['fit_seeds']:
                    maps = z[f'{tn}_{s}']
                    for ci, cm in enumerate(comps):
                        S4, S6, K, kc = comp_sums(maps, cm, cq)
                        a6, tr6, fr6 = thr_hists(S6, K, kc, R8T[ci], Ho, len(cm)); a4, _, _ = thr_hists(S4, K, kc, R8T[ci], Ho, len(cm))
                        th6 += a6; th4 += a4; trt += tr6; frt += fr6; cal6 += cal_hists(S6, K, kc, R8T[ci])
            sel = select_theta(th6, th4, trt, frt)
            sel4 = select_theta(th4, th4, trt, frt)                     # review S1 (report only): B4 with a TRAIN threshold, same rule
            for x in (sel, sel4):                                       # review S3: the chosen threshold is interior to the scanned range
                assert -VOFF - 1 < x['theta'] < VOFF - 1, ('threshold at the scan boundary', tn, x)
            vals = (np.arange(VB) - VOFF) * UNIT
            wk = []
            for k in range(KMAX):
                _, b = logistic(vals, cal6[0, k], cal6[1, k], offset=l0, intercept=False)
                wk.append(b if (b is not None and b > 0) else 1.0)
            best_w = None
            for tq in range(-400, 401):
                cw = counts_at_w(th6, trt, frt, wk, l0q, tq)
                if cw[3] / max(cw[4], 1) + 1e-15 < sel['b4_free']:
                    continue
                if best_w is None or f1_of(cw) > best_w[1] + 1e-15:
                    best_w = (tq, f1_of(cw))
            assert best_w is None or -400 < best_w[0] < 400, ('W threshold at the scan boundary', tn, best_w)     # review S3
            H_out['tiers'][tn] = dict(c=c.tolist(), cq=cq.tolist(), n_occ=no.tolist(), n_free=nf.tolist(), l0=l0, l0q=l0q, theta=sel['theta'],
                                      b4_theta=sel4['theta'], train_b4t_f1=sel4['f1'], train_b4t_free=sel4['free'],
                                      tq=sel['theta'] + l0q, train_f1=sel['f1'], train_free=sel['free'], train_b4_f1=sel['b4_f1'], train_b4_free=sel['b4_free'],
                                      w=wk, w_tq=best_w[0] if best_w else None, w_train_f1=best_w[1] if best_w else None,
                                      train_slopes=slopes_by_k(cal6, l0q))
        res['halves'][h] = H_out
    diff = {}
    if all(res['halves'].get(h, {}).get('tiers') for h in HALVES):
        for tn in FIT_TIERS:
            A, B = (res['halves'][h]['tiers'][tn] for h in HALVES)
            ca, cb = np.asarray(A['c']), np.asarray(B['c']); n = np.minimum(np.asarray(A['n_occ']) + np.asarray(A['n_free']), np.asarray(B['n_occ']) + np.asarray(B['n_free']))
            sup = n >= 1000
            diff[tn] = dict(max_abs_supported=float(np.abs(ca - cb)[sup].max()) if sup.any() else None,
                            wmean_abs=float((np.abs(ca - cb) * n).sum() / max(n.sum(), 1)), theta=[A['theta'], B['theta']], l0=[A['l0'], B['l0']])
    res['half_difference'] = diff
    res['complete'] = not fit_reasons(res, smoke=False)                # informational; eval / aggregate re-check (review M1)
    path = os.path.join(out_dir, 'fit.json')
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1)
    res['sha256'] = file_sha256(path)
    return res, path


# ================================================================ eval stage (VAL / fresh)
def _rt(x):
    return json.loads(json.dumps(x))


def _mean_rows(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def run_eval_job(job):
    panel, rooms, index, seed, opts = job
    t0 = time.time()
    fit = json.load(open(opts['fit']))
    assert file_sha256(opts['fit']) == opts['fit_sha256'], 'fit.json changed since the run started'
    rec = dict(stage='eval', panel=panel, rooms=rooms, index=index, seed=seed, script_sha256=opts['script_sha256'], fit_sha256=opts['fit_sha256'])
    try:
        ctx = house_ctx('VALIDATION', rooms, index)
    except ValueError as e:
        rec['skipped'] = repr(e)[:200]; return rec
    e1p = os.path.join(E1_JOBS, f'{panel}_r{rooms}_i{index}_s{seed}.json')
    e1 = json.load(open(e1p)) if os.path.exists(e1p) else None
    rec['e1_job'] = os.path.basename(e1p) if e1 else None
    F = ctx['F']; n = ctx['n']; rstats = G.RelayStats(); k0cache = {}
    rec['components'] = F['comps']; rec['tiers'] = {}; match = {}
    for tn in EVAL_TIERS:
        L_k0, L_pub, has_inter = tier_maps(ctx, tn, seed, k0cache)
        T = {}
        # ---- B4 (= E1), with the reproduction assertion
        k0_b4 = [G.score(G.tern(L_k0[a]), ctx['K0T'][a], strata=True) for a in range(n)]
        _, preds = G.fuse(L_pub, F, 'sum', rstats)
        r8_b4 = [[G.score(preds[cm[0]], ctx['R8T'][ci], strata=True), len(cm)] for ci, cm in enumerate(F['comps'])]
        if e1 is not None and tn in e1['conds']:
            ev = e1['conds'][tn]['by_int']['V3']
            match[tn] = bool(_rt(k0_b4) == ev['K0'] and _rt(r8_b4) == ev['R8']['sum'])
        else:
            match[tn] = None
        T['B4'] = dict(K0=k0_b4, R8=r8_b4)
        own4 = [G.sum_payload(L) for L in L_pub]                        # review S1: the B4 sums, relayed, for the B4 + TRAIN threshold row
        _, bel4 = G.relay(own4, F, G.comb_sum, rstats)
        cal4 = np.zeros((2, KMAX, VB), np.int64)
        for ci, cm in enumerate(F['comps']):
            assert np.array_equal(G.sum_tern(bel4[cm[0]]), preds[cm[0]]), 'B4 relay mismatch'
            S4, _, K4, kc4 = comp_sums(L_pub, cm)
            assert np.array_equal(S4, bel4[cm[0]][0]) and np.array_equal(K4, bel4[cm[0]][1])
            cal4 += cal_hists(S4, K4, kc4, ctx['R8T'][ci])
        T['cal_B4'] = _sparse(cal4)
        if e1 is not None and tn == JUDGED:
            T['B4_rnet_relay_e1'] = e1['conds'][tn]['by_int']['V3']['rnet']['B4_int_relay']
        # ---- B6 per half (relayed integer sums with E1's provenance assertions) + W row
        for h in HALVES:
            tb = fit['halves'][h]['tiers'][TABLE_OF[tn]]
            cq = np.asarray(tb['cq'], np.int64); theta = int(tb['theta']); l0q = int(tb['l0q']); tq = int(tb['tq'])
            own = [evidence(L, cq) for L in L_pub]
            msgs, bel = G.relay(own, F, G.comb_sum, rstats)
            ownn = [(k.astype(np.int32), k) for _, k in own]
            _, beln = G.relay(ownn, F, G.comb_sum, rstats)
            r8, r8w, cal = [], [], np.zeros((2, KMAX, VB), np.int64)
            for ci, cm in enumerate(F['comps']):
                direct = G.comb_sum([own[a] for a in cm])
                for a in cm:
                    assert np.array_equal(bel[a][0], direct[0]) and np.array_equal(bel[a][1], direct[1]), 'relayed sum != component sum'
                    assert np.array_equal(beln[a][0], sum(ownn[b][0] for b in cm)), 'relayed count != component count'
                    rstats.belief_checks += 1
                S, K = bel[cm[0]]; kc = beln[cm[0]][0]
                r8.append([G.score(readout(S, K, theta), ctx['R8T'][ci], strata=True), len(cm)])
                if tb.get('w_tq') is not None:
                    r8w.append([G.score(readout_w(S, K, kc, tb['w'], l0q, tb['w_tq']), ctx['R8T'][ci]), len(cm)])
                cal += cal_hists(S, K, kc, ctx['R8T'][ci])
            k0 = []
            for a in range(n):
                e, k = evidence(L_k0[a], cq)
                k0.append(G.score(readout(e, k, theta), ctx['K0T'][a], strata=True))
            out = dict(K0=k0, R8=r8, R8_w=r8w, cal=_sparse(cal), theta=theta, l0q=l0q, tq=tq)
            if tn == JUDGED and opts.get('bytes', True):
                out['rnet_relay'] = float(sum(G.int_ctx_bytes(*msgs[e]) for e in F['directed']))
            T[f'B6_{h}'] = out
            th4 = int(tb['b4_theta'])                                   # review S1 (report only): sum L > theta4 (TRAIN threshold)
            T[f'B4t_{h}'] = dict(theta=th4, K0=[G.score(readout(*G.sum_payload(L_k0[a]), th4), ctx['K0T'][a]) for a in range(n)],
                                 R8=[[G.score(readout(*bel4[cm[0]], th4), ctx['R8T'][ci]), len(cm)] for ci, cm in enumerate(F['comps'])])
        rec['tiers'][tn] = T
        del L_k0, L_pub
    rec['e1_match'] = match
    rec['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    return rec


def _sparse(h):
    nz = np.nonzero(h)
    return dict(shape=list(h.shape), idx=[x.tolist() for x in nz], val=h[nz].tolist())


def _dense(s):
    h = np.zeros(s['shape'], np.int64)
    if s['val']:
        h[tuple(np.asarray(i) for i in s['idx'])] = s['val']
    return h


# ================================================================ aggregation and verdict
def _expand(comp_rows):
    return [row for row, k in comp_rows for _ in range(k)]


def _summ(rows):
    d = {k: _mean_rows(rows, k) for k in ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'var_occ_f1', 'free_recall', 'unknown_known_fpr', 'accuracy')}
    tp = sum(r.get('furn_tp', 0) or 0 for r in rows); fp = sum(r.get('furn_fp', 0) or 0 for r in rows); fn = sum(r.get('furn_fn', 0) or 0 for r in rows)
    d['furn_f1_pooled'] = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    return d


def expected_houses(panel):
    return [f'r{r}_i{i}' for r, i in G.panel_specs(panel)]


NOTES = dict(
    fit_frame='agent: every table, prior and threshold is fitted on AGENT-FRAME TRAIN maps (no inter-agent error); the lidar / harsh '
              'tables are refitted per tier on agent-frame maps and are report-only (review S5)',
    w_row_bytes='the W row needs a second relayed sum (count of knowing agents); its bytes are NOT counted (review S7)',
    b4t_row='B4t = sum L > theta4 with theta4 chosen on the same TRAIN half by the same rule (report only; review S1): separates the gain '
            'of the table from that of the threshold',
    fresh_rule='a LEVER / NOT_ENOUGH verdict is downgraded to AMBIGUOUS when the sign of the 4-cell mean (B6 - B4) on fresh is opposite '
               "to VAL's; per-cell and per-seed fresh signs are reported (review S2)",
    threshold_scan='theta is scanned over every integer in [-4097, 4095] units of 0.05 (the +-4096 histogram range); interior asserted')


def aggregate(out_dir, fit_path=None, smoke=False):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    fit_sha = {r.get('fit_sha256') for r in recs if r.get('fit_sha256')}
    scr_sha = {r.get('script_sha256') for r in recs if r.get('script_sha256')}
    S = dict(fit_sha256=sorted(fit_sha), script_sha256=sorted(scr_sha), panels={}, failed=[], skipped=[], e1_mismatch=[], notes=NOTES)
    reasons = []
    if len(fit_sha) > 1 or len(scr_sha) > 1:
        reasons.append('mixed fit / script versions in the job files')
    if not fit_path:                                                    # review M1: without --fit, take the fit recorded by the eval run
        rp = os.path.join(out_dir, 'eval_run.json')
        fit_path = json.load(open(rp)).get('fit') if os.path.exists(rp) else None
    fit = json.load(open(fit_path)) if fit_path and os.path.exists(fit_path) else None
    S['fit_path'] = fit_path
    if fit is not None:
        S['fit_sha256_file'] = file_sha256(fit_path); S['fit_houses'] = {h: fit['halves'][h]['n_houses'] for h in fit.get('halves', {})}
        S['fit_smoke'] = fit.get('smoke'); S['fit_seeds'] = fit.get('seeds'); S['half_difference'] = fit.get('half_difference')
        S['fit_frame'] = fit.get('fit_frame')
    reasons += fit_reasons(fit, smoke=smoke, fit_sha=S.get('fit_sha256_file'), job_fit_shas=fit_sha, job_script_shas=scr_sha)
    for r in recs:
        tag = f"{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}_s{r.get('seed')}"
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append(tag)
        elif any(v is False or v is None for v in r['e1_match'].values()):
            S['e1_mismatch'].append(tag)
    for panel in sorted({r['panel'] for r in recs if r.get('panel')}):
        P = dict(seeds={})
        for seed in sorted({r['seed'] for r in recs if r.get('panel') == panel}):
            X = [r for r in recs if r.get('panel') == panel and r.get('seed') == seed and not r.get('failed') and not r.get('skipped')]
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={})
            for tn in EVAL_TIERS:
                if not all(tn in r['tiers'] for r in X):
                    continue
                d = dict(B4=dict(K0=_summ([a for r in X for a in r['tiers'][tn]['B4']['K0']]), R8=_summ([a for r in X for a in _expand(r['tiers'][tn]['B4']['R8'])])))
                cal4 = sum(_dense(r['tiers'][tn]['cal_B4']) for r in X)
                d['B4']['slopes'] = slopes_by_k(cal4)
                if tn == JUDGED and all('B4_rnet_relay_e1' in r['tiers'][tn] for r in X):
                    d['B4']['rnet_relay_median'] = float(np.median([r['tiers'][tn]['B4_rnet_relay_e1'] for r in X]))
                for h in HALVES:
                    k = f'B6_{h}'
                    if not all(k in r['tiers'][tn] for r in X):
                        continue
                    e = dict(K0=_summ([a for r in X for a in r['tiers'][tn][k]['K0']]), R8=_summ([a for r in X for a in _expand(r['tiers'][tn][k]['R8'])]))
                    if all(r['tiers'][tn][k]['R8_w'] for r in X):
                        e['R8_w'] = _summ([a for r in X for a in _expand(r['tiers'][tn][k]['R8_w'])])
                    cal = sum(_dense(r['tiers'][tn][k]['cal']) for r in X)
                    e['slopes'] = slopes_by_k(cal, X[0]['tiers'][tn][k]['l0q'])
                    if all('rnet_relay' in r['tiers'][tn][k] for r in X):
                        e['rnet_relay_median'] = float(np.median([r['tiers'][tn][k]['rnet_relay'] for r in X]))
                    d[k] = e
                for h in HALVES:
                    k = f'B4t_{h}'
                    if all(k in r['tiers'][tn] for r in X):
                        d[k] = dict(theta=X[0]['tiers'][tn][k]['theta'], K0=_summ([a for r in X for a in r['tiers'][tn][k]['K0']]),
                                    R8=_summ([a for r in X for a in _expand(r['tiers'][tn][k]['R8'])]))
                D['tiers'][tn] = d
            P['seeds'][str(seed)] = D
        S['panels'][panel] = P
    S['agg_reasons'] = reasons
    return S


def _delta(S, panel, seed, h, tn=JUDGED):
    d = S['panels'][panel]['seeds'][str(seed)]['tiers'][tn]
    return d[f'B6_{h}']['R8'], d['B4']['R8']


def panel_reasons(S, panel, smoke=False):
    """review M2 / S6: a panel is complete iff both noise seeds are present, each with EXACTLY the 24-house plan (G4-E1 skipped no
    house, so any skip is incomplete), and the needed tiers carry both halves' B6 (VAL: cam_nf and clean; fresh: cam_nf)"""
    if panel not in S['panels']:
        return [f'no {panel} panel']
    r = []; exp = sorted(expected_houses(panel))
    for seed in (1, 2):
        D = S['panels'][panel]['seeds'].get(str(seed))
        if D is None:
            r.append(f'{panel} seed {seed} missing'); continue
        if D['houses'] != exp and not smoke:
            r.append(f'{panel} seed {seed}: {len(D["houses"])} of {len(exp)} houses')
        for tn in ((JUDGED, 'clean') if panel == 'val' else (JUDGED,)):
            if tn not in D['tiers'] or any(f'B6_{h}' not in D['tiers'][tn] for h in HALVES):
                r.append(f'{panel} seed {seed}: tier {tn} incomplete')
    return r


def verdict(S, smoke=False):
    reasons = list(S.get('agg_reasons', []))
    V = dict(cells={}, fresh_cells={})
    if S['failed']:
        reasons.append(f"failed jobs: {S['failed']}")
    if S['e1_mismatch']:
        reasons.append(f"E1 reproduction mismatch: {S['e1_mismatch']}")
    if S['skipped']:
        reasons.append(f"skipped houses (G4-E1 skipped none): {S['skipped']}")
    reasons += panel_reasons(S, 'val', smoke) + panel_reasons(S, 'fresh', smoke)
    lever = []; notenough = []
    if 'val' in S['panels']:
        for seed in (1, 2):
            D = S['panels']['val']['seeds'].get(str(seed))
            if D is None or JUDGED not in D['tiers'] or 'clean' not in D['tiers']:
                continue
            for h in HALVES:
                if f'B6_{h}' not in D['tiers'][JUDGED] or f'B6_{h}' not in D['tiers']['clean']:
                    continue
                a, b = _delta(S, 'val', seed, h)
                cl = D['tiers']['clean'][f'B6_{h}']['R8']
                c = dict(B4_furn=b['furn_f1'], B6_furn=a['furn_f1'], delta_pp=100 * (a['furn_f1'] - b['furn_f1']),
                         fpr_ok=a['unknown_known_fpr'] <= b['unknown_known_fpr'] + 0.003, free_ok=a['free_recall'] >= b['free_recall'],
                         surface_ok=a['surface_f1'] >= b['surface_f1'] - 0.005, clean_ok=(cl['furn_f1'] >= 0.99 and cl['free_recall'] >= 0.99),
                         B6_free=a['free_recall'], B4_free=b['free_recall'], B6_fpr=a['unknown_known_fpr'], B4_fpr=b['unknown_known_fpr'],
                         clean_furn=cl['furn_f1'], clean_free=cl['free_recall'], B6_K0=D['tiers'][JUDGED][f'B6_{h}']['K0']['furn_f1'],
                         B4_K0=D['tiers'][JUDGED]['B4']['K0']['furn_f1'],
                         B4t_furn=(D['tiers'][JUDGED].get(f'B4t_{h}') or {}).get('R8', {}).get('furn_f1'))
                c['lever'] = c['delta_pp'] >= 3 and c['fpr_ok'] and c['free_ok'] and c['surface_ok'] and c['clean_ok']
                V['cells'][f'{h}|s{seed}'] = c; lever.append(c['lever']); notenough.append(c['delta_pp'] < 1)
    if len(V['cells']) != 4:
        reasons.append(f"{len(V['cells'])} of 4 VAL cells")
    if 'fresh' in S['panels']:
        for seed in (1, 2):
            D = S['panels']['fresh']['seeds'].get(str(seed))
            if D is None or JUDGED not in D['tiers']:
                continue
            for h in HALVES:
                if f'B6_{h}' in D['tiers'][JUDGED]:
                    a, b = _delta(S, 'fresh', seed, h)
                    V['fresh_cells'][f'{h}|s{seed}'] = dict(delta_pp=100 * (a['furn_f1'] - b['furn_f1']), B6_furn=a['furn_f1'], B4_furn=b['furn_f1'])
    if len(V['fresh_cells']) != 4:                                      # review M2
        reasons.append(f"{len(V['fresh_cells'])} of 4 fresh cells")
    V['fresh_cell_signs'] = {k: int(np.sign(c['delta_pp'])) for k, c in V['fresh_cells'].items()}          # review S2
    V['fresh_seed_mean_pp'] = {s: float(np.mean([c['delta_pp'] for k, c in V['fresh_cells'].items() if k.endswith(f'|s{s}')]))
                               for s in (1, 2) if any(k.endswith(f'|s{s}') for k in V['fresh_cells'])}
    if reasons:
        V['verdict'] = 'INCOMPLETE'; V['reasons'] = reasons
        if not smoke:
            return V
    if not V['cells']:
        V['verdict'] = 'INCOMPLETE'; return V
    v = 'LEVER' if all(lever) else ('NOT_ENOUGH' if any(notenough) else 'AMBIGUOUS')
    mv = float(np.mean([c['delta_pp'] for c in V['cells'].values()]))
    V['val_mean_delta_pp'] = mv
    if V['fresh_cells']:
        mf = float(np.mean([c['delta_pp'] for c in V['fresh_cells'].values()])); V['fresh_mean_delta_pp'] = mf
        if v in ('LEVER', 'NOT_ENOUGH') and np.sign(mf) != 0 and np.sign(mv) != 0 and np.sign(mf) != np.sign(mv):
            v = 'AMBIGUOUS'; V['fresh_downgrade'] = True
    V['verdict'] = v if not reasons else f'INCOMPLETE (smoke readout: {v})'
    return V


# ================================================================ drivers
def _save(path, rec):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)


def _run_fit(job):
    half, rooms, index, seeds, opts = job
    path = os.path.join(opts['out'], 'fit_jobs', f'{half}_r{rooms}_i{index}.json')
    try:
        rec = run_fit_job(job)
    except Exception:                                                   # noqa: BLE001  (E4 S1: record and continue)
        rec = dict(stage='fit', half=half, rooms=rooms, index=index, seeds=list(seeds), failed=True, error=traceback.format_exc()[-4000:],
                   script_sha256=opts['script_sha256'])
    _save(path, rec)
    return path, rec.get('seconds'), rec.get('peak_mb'), rec.get('skipped') or ('FAILED' if rec.get('failed') else None)


def _run_eval(job):
    panel, rooms, index, seed, opts = job
    path = os.path.join(opts['out'], 'jobs', f'{panel}_r{rooms}_i{index}_s{seed}.json')
    try:
        rec = run_eval_job(job)
    except Exception:                                                   # noqa: BLE001
        rec = dict(stage='eval', panel=panel, rooms=rooms, index=index, seed=seed, failed=True, error=traceback.format_exc()[-4000:],
                   script_sha256=opts['script_sha256'], fit_sha256=opts['fit_sha256'])
    _save(path, rec)
    return path, rec.get('seconds'), rec.get('peak_mb'), rec.get('skipped') or ('FAILED' if rec.get('failed') else None)


def _pool(fn, jobs, workers):
    t0 = time.time()
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, (path, sec, mb, flag) in enumerate(pool.imap_unordered(fn, jobs), 1):
            print(f'[{i}/{len(jobs)}] {os.path.basename(path)} {None if sec is None else round(sec)} s peak {None if mb is None else round(mb)} MB'
                  + (f' {flag}' if flag else ''), f'{time.time() - t0:.0f}s', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=('fit', 'eval', 'aggregate'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--fit', default='', help='fit.json for eval / aggregate')
    ap.add_argument('--train-per-stratum', type=int, default=5)
    ap.add_argument('--train-houses', default='', help='explicit fit houses, e.g. A:1:60,B:2:61 (smoke)')
    ap.add_argument('--fit-seeds', default='101,102')
    ap.add_argument('--cache', default=default_cache())
    ap.add_argument('--panels', default='val,fresh')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--houses', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-bytes', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='allow < 12 fit houses / partial panels; verdict marked INCOMPLETE')
    a = ap.parse_args(argv)
    fit_seeds = [int(s) for s in a.fit_seeds.split(',') if s]
    assert not ({1, 2} & set(fit_seeds)) and (7 not in fit_seeds or a.smoke), 'fit noise seeds must be disjoint from 1, 2, 7'
    sha = file_sha256(os.path.abspath(__file__))
    t0 = time.time()
    if a.stage == 'fit':
        os.makedirs(os.path.join(a.out, 'fit_jobs'), exist_ok=True)
        cache = os.path.abspath(a.cache)
        assert not cache.startswith(os.path.abspath(ROOT)) or a.smoke, 'keep the TRAIN map cache outside the public repo'
        opts = dict(out=a.out, cache=cache, script_sha256=sha, fit_seeds=fit_seeds, smoke=bool(a.smoke), train_per_stratum=a.train_per_stratum,
                    train_houses=a.train_houses)
        for p in glob.glob(os.path.join(a.out, 'fit_jobs', '*.json')):   # review M1: never silently reuse fit jobs of another script / seed set
            r = json.load(open(p))
            if r.get('script_sha256') != sha or sorted(r.get('seeds', [])) != sorted(fit_seeds):
                raise SystemExit(f'{p} was written by another script / seed set; use a fresh --out')
        jobs = [(h, r, i, tuple(fit_seeds), opts) for h, r, i in train_specs(a.train_per_stratum, a.train_houses)
                if not os.path.exists(os.path.join(a.out, 'fit_jobs', f'{h}_r{r}_i{i}.json'))]
        print(f'fit: {len(jobs)} TRAIN jobs, seeds {fit_seeds}, cache {cache}', flush=True)
        if jobs:
            _pool(_run_fit, jobs, a.workers)
        res, path = fit_aggregate(a.out, opts)
        print('fit.json', path, 'sha256', res['sha256'], 'complete', res['complete'], flush=True)
        for h in HALVES:
            if h in res['halves']:
                for tn, tb in res['halves'][h]['tiers'].items():
                    print(f"  half {h} ({res['halves'][h]['n_houses']} houses) {tn}: l0 {tb['l0']:.3f} theta {tb['theta']} tq {tb['tq']} "
                          f"TRAIN F1 {tb['train_f1']:.4f} (B4 {tb['train_b4_f1']:.4f}) free {tb['train_free']:.4f} (B4 {tb['train_b4_free']:.4f}) "
                          f"B4t theta {tb['b4_theta']} F1 {tb['train_b4t_f1']:.4f} | w {[round(x, 3) for x in tb['w']]} wF1 {tb['w_train_f1']}", flush=True)
        print('half difference', json.dumps(res['half_difference']), flush=True)
        return res
    if a.stage == 'eval':
        assert a.fit and os.path.exists(a.fit), '--fit fit.json required'
        fit = json.load(open(a.fit))
        bad = fit_reasons(fit, smoke=a.smoke)                           # review M1: refuse a smoke / thin / wrong-seed fit before any job
        if fit.get('script_sha256') != sha:
            bad.append('fit.json was written by another script version')
        if bad:
            raise SystemExit('refusing fit.json ' + a.fit + ': ' + '; '.join(bad))
        os.makedirs(os.path.join(a.out, 'jobs'), exist_ok=True)
        opts = dict(out=a.out, fit=os.path.abspath(a.fit), fit_sha256=file_sha256(a.fit), script_sha256=sha, bytes=not a.no_bytes)
        rp = os.path.join(a.out, 'eval_run.json')
        if os.path.exists(rp) and json.load(open(rp)).get('fit_sha256') != opts['fit_sha256']:
            raise SystemExit(f'{rp} records another fit.json; use a fresh --out')
        with open(rp, 'w') as fh:
            json.dump(dict(fit=opts['fit'], fit_sha256=opts['fit_sha256'], script_sha256=sha, smoke=bool(a.smoke), args=vars(a)), fh, indent=1)
        jobs = []
        for panel in a.panels.split(','):
            specs = G.panel_specs(panel)
            if a.houses:
                specs = specs[:a.houses]
            for seed in [int(s) for s in a.seeds.split(',')]:
                for rooms, index in specs:
                    p = os.path.join(a.out, 'jobs', f'{panel}_r{rooms}_i{index}_s{seed}.json')
                    if os.path.exists(p):
                        r = json.load(open(p))
                        if r.get('fit_sha256') != opts['fit_sha256'] or r.get('script_sha256') != sha:
                            raise SystemExit(f'{p} was written with another script / fit; use a fresh --out')
                        continue
                    jobs.append((panel, rooms, index, seed, opts))
        print(f'eval: {len(jobs)} jobs, fit sha {opts["fit_sha256"][:12]}, script sha {sha[:12]}', flush=True)
        if jobs:
            _pool(_run_eval, jobs, a.workers)
    S = aggregate(a.out, a.fit or None, smoke=a.smoke)
    S['verdict'] = verdict(S, smoke=a.smoke)
    S['args'] = vars(a); S['seconds'] = time.time() - t0
    with open(os.path.join(a.out, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1)
    V = S['verdict']
    print('verdict', V['verdict'], V.get('reasons', ''), flush=True)
    for k, c in V['cells'].items():
        print(f"  VAL {k}: B4 {c['B4_furn']:.4f} B6 {c['B6_furn']:.4f} d {c['delta_pp']:+.2f} pp | free {c['B4_free']:.4f}->{c['B6_free']:.4f} "
              f"fpr {c['B4_fpr']:.4f}->{c['B6_fpr']:.4f} | clean {c['clean_furn']:.4f}/{c['clean_free']:.4f} | K0 {c['B4_K0']:.4f}->{c['B6_K0']:.4f} "
              f"| B4t {c['B4t_furn']} | lever {c['lever']}", flush=True)
    for k, c in V['fresh_cells'].items():
        print(f"  fresh {k}: B4 {c['B4_furn']:.4f} B6 {c['B6_furn']:.4f} d {c['delta_pp']:+.2f} pp", flush=True)
    return S


if __name__ == '__main__':
    main()
