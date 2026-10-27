"""g4_noise_panels.py (Claude, 2026-10-27): batch G4-E1 "加固面板的地板", pre-registered in claude_handoff/OVERNIGHT_REPORT_20261021.md,
D8-36 (parameters and criteria copied verbatim below; nothing here may change them). CPU only, numpy / scipy, no torch, no GPU.

Question (falsifies the round-3 value claim): in a noisy world, scored against the CLEAN truth, is there room for a learned codec?

Panels (v420 storey world rebuilt locally; TEST is never generated): scene_plan('VALIDATION', 16), index < 8 = VAL (judged),
index 8-15 = fresh (reported). HOUSEEXPO_JSON_DIR = claude_handoff/analysis_20261027/data/houseexpo_json, HOUSEEXPO_INDEX =
claude_handoff/claude_reviews_20261021/houseexpo_index_v365.json. A house whose make_fixture raises ValueError is skipped and listed.

Sensor tiers (single variable; C1 table, D8-36):
  clean                                                      (exact v420 observe)
  lidar  range sigma 2 cm,      drop 5 %,  -          , per-frame 1 cm / 0.2 deg, inter-agent 3 cm / 0.5 deg
  cam    range sigma 1.5 % r,   drop 10 %, early 0.5 %, per-frame 2 cm / 0.5 deg, inter-agent 5 cm / 1 deg   (main tier)
  harsh  range sigma 3 % r,     drop 20 %, early 1 %,   per-frame 3 cm / 1 deg,   inter-agent 10 cm / 2 deg
  range_only (cam range sigma + drop + early returns), pose_only (cam per-frame error), inter_only (cam inter-agent error), cam_nf (cam
  without the inter-agent error).
Noise model (A07 p3_noise, C1 c1_noise_seed2): every beam is traced in the TRUE world from the TRUE pose with the generator's own
marching (65,536 Fibonacci beams, samples at s * ds, ds = half the smallest cell, strict 5 m, first hit); then, per kept beam,
r' = r + N(0, 1) (sigma0 + k r) on hits, a dropped beam marks nothing, an early-return outlier reports r' = U(0.2, 1) r (hits only),
r' >= 0.05 m on noisy tiers; no-hit beams keep their free extent. The agent integrates the measurement at its BELIEVED pose: per frame
(= per pose) p_hat = p + (N(0, s_t), N(0, s_t), N(0, s_t / 2)) and yaw error N(0, s_yaw) (A07 convention: per-axis sigmas, z at half).
Inter-agent error = one SE(2) offset per agent of its whole map w.r.t. the public frame: rotation N(0, s_yaw) about the agent's centre
(positions_m) and translation (N(0, s_t), N(0, s_t), 0), applied to the believed geometry (public frame); K0 is integrated without it
(agent frame). Common random numbers: the standard variates are drawn per (noise seed, house, agent, pose) independently of the tier
and scaled by the tier's sigmas, so tiers differ only in their parameters (cam vs cam_nf is a paired comparison; "cam at the true
inter-agent poses" IS cam_nf's map). Two noise seeds (default 1, 2); clean is seed-free.
Integrator (review M1, coordinator decision 2026-10-27): OctoMap-style log-odds, +0.85 per hit / -0.40 per miss, clamped to the OctoMap
default bounds (clamping_thres_min 0.1192 / max 0.971 -> -1.999 / +3.512 log-odds). Every beam is one measurement (OctoMap insertRay):
one miss per voxel it crosses (consecutive samples in one voxel merged; never its endpoint voxel) and one hit at the endpoint voxel
(A07 convention). Three integrators are computed from the SAME beam events of every scan (scans in pose order):
  V3 (GOVERNING, the verdict): per beam, beams in Fibonacci index order, the clamp applied after EVERY single update (updateNodeLogOdds);
  V1 (required sensitivity row, not judged): OctoMap insertPointCloud: per scan a voxel gets ONE update, a hit if any endpoint lies in it,
     else a miss if any beam crosses it, clamp after every update; B5 uses tau ">=" here (review S4: with ">" tau = 2.55 is empty);
  V0 (report row, the first implementation): per scan L <- clamp(L + 17 hits - 8 misses), one clamp per scan;
  V2 (report row, coordinator addendum): V3 WITHOUT the clamp (= the A07 / A10 / C1 probe integrator, the review's V2).
Each row gets the full K0 / R8 / bytes pipeline (B5 on cam) and its own (a)-(d) readouts; only V3 enters the verdict
(governing_integrator = "V3"). INTEGER LOG-ODDS: 1 unit = 0.05, so hit = +17, miss = -8, clamp [-40, +70] are all exact integers (no
quantisation error at all); clamped maps are int8 with unknown = -128 (never touched), V2 maps int32 with unknown = int32 min. Ternary read-out: occupied iff L > 0, free iff touched and L <= 0 (A07). Clean tier: hits only land
in occupied cells and misses only in free cells, so under every integrator the ternary of the clean integer map IS observe(): asserted
voxel-for-voxel for every agent of every house (crit. (a)).

Truth and scoring: the clean sensor at the true poses (the fixture's private own maps; R8 truth = stream_targets(private, reached8)),
scored only in the clean-known region (metrics3d target-known scope); K0 in the agent frame, R8 in the public frame. Metrics = numpy
port of train_3d_v431.metrics3d (furniture-region F1, surface F1, surface F1 tol 1, variable F1, occ F1, 3-way accuracy, unknown->known
FPR, object recall) + furniture-region F1 within 1 voxel (A07 p7 definition) + free recall; strata: height above the floor top
[0, .3) / [.3, .9) / >= .9 m and observation multiplicity m = 1, 2, 3, 4, >= 5 (agents whose clean map knows the voxel; R8).

Fusion on the radio forest, relayed once per directed edge, messages = the sender side's fusion with the receiver's branch excluded:
  sum    : integer log-odds SUM (Bayes for independent beams, exact on a tree). Every agent's evidence enters exactly once: provenance
           bitmasks travel with every message and are asserted disjoint at every combination, every belief's provenance must equal its
           component, and every belief must equal the direct sum over the component (integer equality).
  max_occ: the literal "max": per-voxel max L, whose ternary read-out is exactly the occupied-wins max (= stream_targets, rollout_max in
           the clean world, the probes' "max"); also the fidelity of the ternary pipelines B2 / B3.
  maxabs : per voxel the value with the largest |L| (ties: occupied) = the noisy counterpart of the current system's max-precision
           selection; a max under a total order (idempotent join).
  Criterion (c) is CONJUNCTIVE (review M2, coordinator): sum must beat BOTH max_occ and maxabs.
Baselines and bytes (R_net = one R8 fusion, all links; relay = each directed edge once; flood = every own map over every edge of its
component, x (|component| - 1)):
  B1 gzip-9 of the int8 log-odds own maps, flooded (also: gzip of the int16 sum messages, relayed);
  B2 OctoMap binary tree (.bt: header + 2 B per inner node, children x | y << 1 | z << 2, 00 unknown / 01 occ / 10 free / 11 inner,
     max-likelihood + pruned, depth-16 key space with the map at key 32768) then gzip-9, of the ternary maps (relay of max_occ messages;
     also flood). It carries the ML ternary only, so its fidelity is the max_occ pipeline's;
  B3 ternary ctx (rate_baselines_lib.ctx_bytes: 9 causal neighbours, KT exact length + 2 B) of the max_occ messages, relayed;
  B4 lossless transmission of the noisy integer log-odds map = ternary ctx (sign) + magnitudes: an "equal to the first same-state
     causal neighbour" flag, else exp-Golomb(0) bins, all adaptive KT binary contexts (int_magnitude_events); exact adaptive code
     length + 4 B header (no real round trip; brute-force sequential coder agrees in the tests; beats gzip-9 and lzma-extreme on every
     smoke message). B4 = min(integer ctx coder, structure code) (review M4): the structure code is S0 (observed hits + float32 poses,
     receiver ray casting), which rebuilds the clean integer map exactly under every integrator, so on the CLEAN tier B4_relay = S0 =
     --s0-rnet bytes per house (B4_int_relay is kept as information); on noisy tiers S0 cannot express the noisy magnitudes (and the
     ternary part is only 10-15 % of B4), so the structure branch is +inf and B4_relay = B4_int_relay = relay of the sum messages.
     B4_maxabs = relay of the maxabs messages, B4_flood = flood of the own maps, B4_best_info = min(relay, flood) (information only);
  B5 classical denoise-then-encode, per agent on the encoder side: occupancy threshold tau on L (tau in {0, 17, 34, 51} units =
     0 / .85 / 1.7 / 2.55 log-odds; occupied iff L > tau, under V1 L >= tau for tau > 0), then ONE of two deletion families
       cc : occupied 26-connected components with < m voxels set free;
       box: the pre-registered "delete boxes" (review M3): A03 greedy solid box cover of the occupied voxels that never enters an observed
            free voxel (box_cover, copied verbatim from probes/A03/a03np.py), keep the boxes holding >= m occupied voxels, decode
            known ? (inside a kept box) : unknown;
     with m in {4, 8, 16, 32}, x optional 2x2x1 block ternary down-sampling (known if any known, occupied if >= 50 % of the known voxels;
     rate_baselines down_up); fused by the sum of +1 / -1 votes (relayed vote sums, B4 coder) or by the occupied-wins join (relayed
     ternary, ctx). The coder is ternary ctx, NOT the A03 structure code with ray-cast decoding: at the 1/4 B4 budget this changes bytes,
     not the hull reading. Hull (time sharing) over the UNION of both families and both fusions (128 points), read at R_net = 1/4 B4;
     per-family hulls are reported. Upper readout (governing integrator): the same at the TRUE inter-agent poses.
  S0 (clean exact structure code, G4-E0's subject) is not re-implemented: its measured clean R_net 6,066 B / house (D8-36 fact 2,
  24-house median, "6.1 KB") is the reference constant (--s0-rnet).
Verdict (cam tier, VAL panel, governing integrator V3; D8-36 verbatim; returns INCOMPLETE unless the noise seeds are exactly {1, 2},
both seeds hold the same complete VAL house set (24 minus the deterministic make_fixture skips), every seed has clean / cam / cam_nf
and cam has B5 (review M5)):
  (a) clean: B4 identical to S0 voxel-for-voxel, free recall = 1 (self-check)  -> here: every clean K0 / R8 (all fusions) map equals the
      clean truth, free recall 1, FPR 0, 3-way accuracy 1, AND the clean B4 bytes equal S0 on every house; failing = bug -> INCONCLUSIVE;
  (b) B4 R8 furniture (vs truth) <= .85 AND B4 R_net >= 1.5 x clean S0 AND B5 at <= 1/4 B4 bytes has furniture <= B4 + 2 pp
      -> LEARNING_ROOM (go to E2-E5, needs user-approved leases); B5 with no point at <= 1/4 B4 passes this arm and is flagged;
  stop-loss: B4 furniture >= .92 OR B4 R_net < 1.3 x clean -> STOP (no learning room even with noise); otherwise AMBIGUOUS;
  (c) R8 furniture sum - max_occ >= +3 pp AND sum - maxabs >= +3 pp, FPR(sum) <= FPR(each) + 0.3 pp, free recall not below either
      -> noisy tiers use the sum;
  (d) report only: cam_nf - cam furniture loss >= 10 pp -> the registration panel must be reported.
Job files carry a fingerprint (script / data module / HouseExpo index sha256 + every option); a run refuses job files with another
fingerprint and the aggregation refuses mixed fingerprints (review M6).

Usage (python -B; spawned workers, peak ~1.0 GB each, ~0.5 GB of it torch imported transitively by the world module; one job =
one (house, noise seed); 96 jobs for VAL + fresh x 2 seeds; ALWAYS a fresh --out for a changed script):
  python -B g4_noise_panels.py --out DIR [--panels val,fresh] [--seeds 1,2] [--workers 6] [--houses N] [--conds ...] [--b5-conds ...]
  python -B g4_noise_panels.py --out DIR --aggregate-only            # re-aggregate existing job files (writes aggregate.json)
Smoke: --panels val --houses 2 --seeds 1 --workers 2 --smoke-asserts.  Output: DIR/jobs/*.json, DIR/summary.json, DIR/run.json.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, collections, gc, gzip, hashlib, itertools, json, math, multiprocessing, time  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.special import gammaln  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HOUSEEXPO_JSON_DIR = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'data', 'houseexpo_json')
HOUSEEXPO_INDEX = os.path.join(ROOT, 'claude_handoff', 'claude_reviews_20261021', 'houseexpo_index_v365.json')


def setup_env(json_dir=HOUSEEXPO_JSON_DIR, index=HOUSEEXPO_INDEX):
    os.environ['HOUSEEXPO_JSON_DIR'] = json_dir
    os.environ['HOUSEEXPO_INDEX'] = index
    for p in (HERE, os.path.join(ROOT, 'exploration_grid'), ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


setup_env()
import mapping_3d_storey_data_v420 as W  # noqa: E402

SENSOR_RANGE = W.SENSOR_RANGE
FLOOR_T = W.FLOOR_T
HOPS = tuple(W.HOPS)

# ---------------------------------------------------------------- integer log-odds (1 unit = 0.05 log-odds)
LO_UNIT = 0.05
L_HIT, L_MISS = 17, -8                      # +0.85 / -0.40
L_MIN, L_MAX = -40, 70                      # OctoMap defaults 0.1192 / 0.971 -> logit -1.999 / +3.512
UNK = -128
assert abs(L_HIT * LO_UNIT - 0.85) < 1e-12 and abs(L_MISS * LO_UNIT + 0.40) < 1e-12
assert abs(L_MIN * LO_UNIT - math.log(0.1192 / 0.8808)) < 0.01 and abs(L_MAX * LO_UNIT - math.log(0.971 / 0.029)) < 0.02

SALT = 20261027
S0_RNET_CLEAN = 6066.0                      # D8-36 fact 2 (A03 P9b, 24 VAL houses, median bytes per house, "6.1 KB")

_Z = dict(sigma0=0., k=0., p_drop=0., p_out=0., pose_t=0., pose_yaw=0., inter_t=0., inter_yaw=0.)
_CAM_RANGE = dict(k=0.015, p_drop=0.10, p_out=0.005)
_CAM_POSE = dict(pose_t=0.02, pose_yaw=0.5)
_CAM_INTER = dict(inter_t=0.05, inter_yaw=1.0)
CONDITIONS = {                              # D8-36 / C1 table; angles in degrees, lengths in metres
    'clean': dict(_Z),
    'inter_only': dict(_Z, **_CAM_INTER),
    'range_only': dict(_Z, **_CAM_RANGE),
    'pose_only': dict(_Z, **_CAM_POSE),
    'cam_nf': dict(_Z, **_CAM_RANGE, **_CAM_POSE),
    'cam': dict(_Z, **_CAM_RANGE, **_CAM_POSE, **_CAM_INTER),
    'lidar': dict(_Z, sigma0=0.02, p_drop=0.05, pose_t=0.01, pose_yaw=0.2, inter_t=0.03, inter_yaw=0.5),
    'harsh': dict(_Z, k=0.03, p_drop=0.20, p_out=0.01, pose_t=0.03, pose_yaw=1.0, inter_t=0.10, inter_yaw=2.0),
}
COND_ORDER = ('clean', 'inter_only', 'range_only', 'pose_only', 'cam_nf', 'cam', 'lidar', 'harsh')
B5_TAUS = (0, 17, 34, 51)
B5_MS = (4, 8, 16, 32)
B5_DS = ((1, 1, 1), (2, 2, 1))
B5_FAMILIES = ('cc', 'box')
FUSIONS = ('sum', 'maxabs', 'max_occ')
INTEGRATORS = ('V3', 'V1', 'V0', 'V2')
INTEGRATOR_DEFS = {'V3': 'per beam (OctoMap insertRay), beams in index order, clamp [-40, 70] after every update  [GOVERNING]',
                   'V1': 'per scan one update per voxel (OctoMap insertPointCloud: hit if any endpoint else miss), clamp after every update',
                   'V0': 'per scan L <- clamp(L + 17 hits - 8 misses) (first implementation)',
                   'V2': 'per beam without clamp (= V3 unclamped = the A07 / A10 / C1 probe integrator)'}
UNCLAMPED = ('V2',)
GOVERNING = 'V3'
JOB_FORMAT = 2
M_STRATA = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 99))
H_BANDS = ((0.0, 0.3), (0.3, 0.9), (0.9, 99.0))
NB = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]   # rate_baselines_lib.NB
LN2 = math.log(2.0)
ROW_KEYS = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'surface_f1_tol1', 'var_occ_f1', 'occ_f1', 'obj_recall', 'accuracy',
            'unknown_known_fpr', 'free_recall', 'furn_tp', 'furn_fp', 'furn_fn')
B5_ROW_KEYS = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'var_occ_f1', 'accuracy', 'unknown_known_fpr', 'free_recall')


# ================================================================ panels
def panel_specs(panel):
    plan = W.scene_plan('VALIDATION', 16)                              # raises on TEST; never called with anything else
    assert all(s['split'] == 'VALIDATION' for s in plan)
    lo, hi = {'val': (0, 8), 'fresh': (8, 16)}[panel]
    return [(int(s['rooms']), int(s['index'])) for s in plan if lo <= s['index'] < hi]


# ================================================================ sensing
def rotz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def trace_pose(occ, cell, p, rays):
    """the generator's observe() marching for one pose: per beam the first-hit sample step (0 = none) and the last free sample step
    (hit beams: s_hit - 1; others: the last in-grid step, at most nsteps). Same arithmetic as observe, element for element."""
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    R = len(rays); hit = np.zeros(R, np.int32); last = np.full(R, nsteps, np.int32)
    alive = np.arange(R)
    for s in range(1, nsteps + 1):
        if not len(alive):
            break
        idx = np.floor((p[None] + rays[alive] * (s * ds)) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), 1)
        last[alive[~inside]] = s - 1
        alive = alive[inside]; idx = idx[inside]
        h = occ[idx[:, 0], idx[:, 1], idx[:, 2]] == 1
        hit[alive[h]] = s; last[alive[h]] = s - 1
        alive = alive[~h]
    return hit, last


def pose_draws(seed, rooms, index, a, j, R):
    """common random numbers of one (noise seed, house, agent, pose), independent of the tier"""
    rng = np.random.default_rng([SALT, int(seed), int(rooms), int(index), int(a), int(j)])
    return dict(u_drop=rng.random(R), z_r=rng.standard_normal(R), u_out=rng.random(R), u_frac=rng.random(R), z_pose=rng.standard_normal(4))


def frame_draws(seed, rooms, index, a):
    rng = np.random.default_rng([SALT, int(seed), int(rooms), int(index), int(a), 1000])
    return rng.standard_normal(3)


def measure(c, hit_step, last, p, dr, ds, rays):
    """one scan as the agent integrates it in ITS frame: (origin, directions, ranges (nan = no hit), last free step) of the kept beams,
    plus noise statistics. A zero sigma leaves the corresponding quantity bit-identical (clean = observe)."""
    R = len(rays); hit = hit_step > 0
    keep = dr['u_drop'] >= c['p_drop'] if c['p_drop'] > 0 else np.ones(R, bool)
    r_true = hit_step * ds                                               # float64, s * ds as in observe
    r = np.where(hit, r_true, np.nan)
    if c['sigma0'] > 0 or c['k'] > 0:
        r[hit] = r_true[hit] + dr['z_r'][hit] * (c['sigma0'] + c['k'] * r_true[hit])
    out = np.zeros(R, bool)
    if c['p_out'] > 0:
        out = hit & (dr['u_out'] < c['p_out'])
        r[out] = (0.2 + 0.8 * dr['u_frac'][out]) * r_true[out]
    regular = hit & keep & ~out
    err = r[regular] - r_true[regular]; rel = err / r_true[regular]
    clipped = 0
    if c['sigma0'] > 0 or c['k'] > 0 or c['p_out'] > 0:
        clipped = int((hit & keep & (r < 0.05)).sum())
        r[hit] = np.maximum(r[hit], 0.05)
    o = p; D = rays; z = dr['z_pose']; pose_err = [0., 0., 0., 0.]
    if c['pose_t'] > 0:
        dp = np.array([z[0] * c['pose_t'], z[1] * c['pose_t'], z[2] * c['pose_t'] / 2]); o = p + dp; pose_err[:3] = dp.tolist()
    if c['pose_yaw'] > 0:
        yaw = float(z[3] * c['pose_yaw']); D = rays @ rotz(yaw).T; pose_err[3] = yaw
    stats = dict(beams=R, dropped=int((~keep).sum()), hits=int(hit.sum()), hits_kept=int((hit & keep).sum()), outliers=int((out & keep).sum()),
                 n_err=int(err.size), s_err=float(err.sum()), s_err2=float((err ** 2).sum()), s_rel=float(rel.sum()), s_rel2=float((rel ** 2).sum()),
                 clipped=clipped, pose=pose_err)
    return (o, D[keep], r[keep], last[keep]), stats


def to_public(scan, frame):
    """apply an agent's inter-agent SE(2) error (rotation about its centre + translation) to a scan's believed geometry"""
    if frame is None:
        return scan
    Ra, ta, cen = frame; o, D, r, sm = scan
    return (Ra @ (o - cen) + cen + ta, D @ Ra.T, r, sm)


def scan_events(shape, cell, ds, o, D, r, smax):
    """the beam events of one scan: (flat voxel, beam index, delta) with one miss (-8) per voxel a beam crosses = samples s * ds strictly
    before the measured range (hit beams) or up to the last free step (no-hit beams), consecutive samples in one voxel merged, never the
    beam's own endpoint voxel; and one hit (+17) at the endpoint voxel. Also the per-voxel hit / miss counts (flat int64 [V]).
    (= the review's r1b_perbeam_probe.scan_events; a beam has at most one event per voxel.)"""
    V = int(np.prod(shape)); shape_a = np.array(shape); cell = np.asarray(cell, np.float64)
    hit = ~np.isnan(r); end_f = np.full(len(D), -1, np.int64)
    ev_v, ev_b, ev_d = [], [], []
    if hit.any():
        hi = np.nonzero(hit)[0]
        e = np.floor((o[None] + D[hi] * r[hi, None]) / cell).astype(np.int64)
        ok = np.all((e >= 0) & (e < shape_a), 1)
        ef = (e[ok, 0] * shape[1] + e[ok, 1]) * shape[2] + e[ok, 2]
        end_f[hi[ok]] = ef
        ev_v.append(ef); ev_b.append(hi[ok]); ev_d.append(np.full(len(ef), L_HIT, np.int8))
    rr = np.where(hit, r - 1e-9, np.inf); sm = np.where(hit, np.iinfo(np.int32).max, smax)
    act = np.arange(len(D)); last_f = np.full(len(D), -1, np.int64); s = 0
    while len(act):
        s += 1; t = s * ds
        act = act[(t < rr[act]) & (s <= sm[act])]
        if not len(act):
            break
        idx = np.floor((o[None] + D[act] * t) / cell).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape_a), 1)
        a2 = act[inside]; idx = idx[inside]
        f = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
        new = (f != last_f[a2]) & (f != end_f[a2])
        ev_v.append(f[new]); ev_b.append(a2[new]); ev_d.append(np.full(int(new.sum()), L_MISS, np.int8)); last_f[a2] = f
    v = np.concatenate(ev_v) if ev_v else np.zeros(0, np.int64)
    b = np.concatenate(ev_b).astype(np.int32) if ev_b else np.zeros(0, np.int32)
    d = np.concatenate(ev_d) if ev_d else np.zeros(0, np.int8)
    nh = np.bincount(v[d > 0], minlength=V); nm = np.bincount(v[d < 0], minlength=V)
    return v, b, d, nh, nm


def integrate_scan(shape, cell, ds, o, D, r, smax):
    """per-voxel hit / miss beam counts of one scan (flat int64 [V])"""
    return scan_events(shape, cell, ds, o, D, r, smax)[3:]


def apply_sequential(L, v, b, d):
    """V3: per voxel, apply its events in beam order, clamping to [L_MIN, L_MAX] after EVERY update (the review's
    r1b_perbeam_probe.apply_sequential: groups sorted by size, one vectorised step per event rank)"""
    if not len(v):
        return
    order = np.lexsort((b, v)); v = v[order]; d = d[order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]]); sizes = np.diff(np.r_[starts, len(v)])
    gs = np.argsort(-sizes, kind='stable'); starts = starts[gs]; sizes = sizes[gs]
    vox = v[starts]; cur = L[vox].astype(np.int64); neg = -sizes
    for k in range(int(sizes.max())):
        nl = int(np.searchsorted(neg, -k, side='left'))                  # groups with more than k events
        cur[:nl] = np.clip(cur[:nl] + d[starts[:nl] + k], L_MIN, L_MAX)
    L[vox] = cur


def integrate_agent(shape, cell, ds, scans, integrators=INTEGRATORS):
    """an agent's integer log-odds maps under the requested integrators, all from the same beam events (scans in pose order):
    V3 per-beam sequential clamp; V1 one update per scan per voxel (hit if any endpoint, else miss), clamp per update;
    V0 per scan L <- clamp(L + 17 hits - 8 misses); V2 per beam without any clamp (= sum of all events, order-free).
    Returns {integrator: map}: int8 with unknown = -128 for the clamped integrators, int32 with unknown = int32 min for V2."""
    V = int(np.prod(shape)); Ls = {k: np.zeros(V, np.int64) for k in integrators}; known = np.zeros(V, bool)
    for o, D, r, sm in scans:
        v, b, d, nh, nm = scan_events(shape, cell, ds, o, D, r, sm)
        t = (nh + nm) > 0
        if 'V0' in Ls:
            L0 = Ls['V0']; L0[t] = np.clip(L0[t] + L_HIT * nh[t] + L_MISS * nm[t], L_MIN, L_MAX)
        if 'V1' in Ls:
            L1 = Ls['V1']; h = nh > 0; m = (nm > 0) & ~h
            L1[h] = np.minimum(L1[h] + L_HIT, L_MAX); L1[m] = np.maximum(L1[m] + L_MISS, L_MIN)
        if 'V2' in Ls:
            Ls['V2'] += L_HIT * nh + L_MISS * nm
        if 'V3' in Ls:
            apply_sequential(Ls['V3'], v, b, d)
        known |= t
        del v, b, d
    out = {}
    for k, L in Ls.items():
        dt = np.int32 if k in UNCLAMPED else np.int8
        lo = np.iinfo(dt).min
        assert not known.any() or (L[known].min() > lo and L[known].max() <= np.iinfo(dt).max), (k, 'log-odds outside the storage type')
        out[k] = np.where(known, L, lo).astype(dt).reshape(shape)
    return out


def is_unk(L):
    """unknown sentinel by storage type: int8 maps (clamped integrators) use -128 (= UNK), wider maps (V2) their type's minimum"""
    L = np.asarray(L)
    return L == (np.iinfo(L.dtype).min if L.dtype != np.int8 else UNK)


def pack_sum(S, K):
    """sum message as little-endian int16 (or int32 when it does not fit) with the type's minimum as the unknown sentinel (B1 relay)"""
    big = K.any() and (np.abs(S[K]).max() >= 32767)
    dt = '<i4' if big else '<i2'
    return np.where(K, S, np.iinfo(dt).min).astype(dt)


def tern(L):
    return np.where(is_unk(L), np.int8(-1), (L > 0).astype(np.int8)).astype(np.int8)


# ================================================================ metrics (numpy port of train_3d_v431.metrics3d + extras)
def dil6(m):
    o = m.copy()
    o[1:] |= m[:-1]; o[:-1] |= m[1:]
    o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    o[:, :, 1:] |= m[:, :, :-1]; o[:, :, :-1] |= m[:, :, 1:]
    return o


def _f1(a, b):
    tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
    return (2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None), tp, fp, fn


class House:
    def __init__(self, occ, slab, obj, cell):
        self.slab = np.asarray(slab, bool); self.obj = np.asarray(obj, bool)
        struct = np.asarray(occ, bool) & ~self.obj & ~self.slab
        self.region = ~self.slab & ~dil6(struct)
        Z = occ.shape[2]; h = (np.arange(Z) + .5) * float(cell[2]) - FLOOR_T
        self.band_z = np.full(Z, -1, np.int64)
        for b, (lo, hi) in enumerate(H_BANDS):
            self.band_z[(h >= lo) & (h < hi)] = b


class Target:
    def __init__(self, t, H, mult=None):
        self.t = t; self.H = H; self.known = t >= 0; self.occ = t == 1; self.free = t == 0
        self.surf = self.occ & dil6(self.free); self.interior = self.occ & ~self.surf
        self.dil_known = dil6(self.known); self.dil_surf = dil6(self.surf)
        self.tvar = self.occ & ~H.slab; self.tr = self.occ & H.region; self.dil_tr = dil6(self.tr); self.tobj = self.occ & H.obj
        self.n_known = int(self.known.sum()); self.mult = mult


def _band_sum(mask3, band_z, nb):
    per_z = mask3.sum((0, 1)); out = np.zeros(nb, np.int64)
    ok = band_z >= 0; np.add.at(out, band_z[ok], per_z[ok]); return out


def score(p, T, strata=False):
    """one metrics3d row (target-known scope) + furn_f1_tol1 (A07 p7) + free recall + pooled furniture counts (+ strata counts)"""
    if T.n_known == 0:
        return {k: None for k in ROW_KEYS}
    H = T.H; pocc = p == 1
    po = pocc & T.known
    ps = po & ~T.interior
    surface_f1 = _f1(ps, T.surf)[0]
    po_any = pocc & T.dil_known; po1 = po_any & ~T.interior
    if po_any.any() and T.surf.any():
        prec1 = float((po1 & T.dil_surf).sum()) / float(po1.sum()) if po1.any() else 1.
        rec1 = float((T.surf & dil6(po_any)).sum()) / float(T.surf.sum())
        tol1 = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-12)
    else:
        tol1 = 0. if (po_any.any() or T.surf.any()) else None
    pr = po & H.region
    furn, tp, fp, fn = _f1(pr, T.tr)
    npr, ntr = int(pr.sum()), int(T.tr.sum())
    if npr + ntr:
        pp = float((pr & T.dil_tr).sum()) / npr if npr else 1.; rr = float((T.tr & dil6(pr)).sum()) / ntr if ntr else 1.
        ftol = 2 * pp * rr / max(pp + rr, 1e-12)
    else:
        ftol = None
    nfree = int(T.free.sum()); unk = ~T.known
    row = dict(furn_f1=furn, furn_f1_tol1=ftol, surface_f1=surface_f1, surface_f1_tol1=tol1, var_occ_f1=_f1(po & ~H.slab, T.tvar)[0],
               occ_f1=_f1(po, T.occ)[0], obj_recall=(float((po & T.tobj).sum()) / float(T.tobj.sum())) if T.tobj.any() else None,
               accuracy=float((p[T.known] == T.t[T.known]).mean()), unknown_known_fpr=float((p[unk] >= 0).mean()) if unk.any() else None,
               free_recall=(float(((p == 0) & T.free).sum()) / nfree) if nfree else None, furn_tp=tp, furn_fp=fp, furn_fn=fn)
    if strata:
        nb = len(H_BANDS); tpm, fpm, fnm = pr & T.tr, pr & ~T.tr, ~pr & T.tr; frh = (p == 0) & T.free
        st = dict(h_furn=np.stack([_band_sum(x, H.band_z, nb) for x in (tpm, fpm, fnm)], 1).tolist(),
                  h_free=np.stack([_band_sum(frh, H.band_z, nb), _band_sum(T.free, H.band_z, nb)], 1).tolist())
        if T.mult is not None:
            cls = np.full(T.mult.shape, -1, np.int64)
            for i, (lo, hi) in enumerate(M_STRATA):
                cls[(T.mult >= lo) & (T.mult <= hi)] = i
            def by_m(x):
                v = cls[x & (cls >= 0)]; return np.bincount(v, minlength=len(M_STRATA)).astype(np.int64)
            st['m_furn'] = np.stack([by_m(x) for x in (tpm, fpm, fnm)], 1).tolist()
            st['m_free'] = np.stack([by_m(frh), by_m(T.free)], 1).tolist()
        row['strata'] = st
    return row


# ================================================================ coders
def ctx_tern_bits(t):
    """= rate_baselines_lib.ctx_bytes (bits, no header): KT (alpha 1/2, 3 symbols), 9 causal neighbours, out-of-grid = unknown.
    In-place uint16 context arithmetic (3^10 = 59,049 < 2^16)."""
    t = np.asarray(t); X, Y, Z = t.shape
    s = (t.astype(np.int16) + 1).astype(np.uint16)
    c = np.zeros(t.shape, np.uint16)
    for dx, dy, dz in NB:
        c *= 3; c[dx:, dy:, dz:] += s[:X - dx, :Y - dy, :Z - dz]
    c *= 3; c += s
    cnt = np.bincount(c.ravel(), minlength=3 ** (len(NB) + 1)).reshape(-1, 3).astype(np.float64)
    N = cnt.sum(1); u = N > 0; cnt, N = cnt[u], N[u]
    return float(((gammaln(N + 1.5) - gammaln(1.5)) - (gammaln(cnt + .5) - gammaln(.5)).sum(1)).sum() / LN2)


def ctx_tern_bytes(t):
    return ctx_tern_bits(t) / 8 + 2


def kt_bin_bits(n0, n1):
    n0 = np.asarray(n0, np.float64).ravel(); n1 = np.asarray(n1, np.float64).ravel(); n = n0 + n1; u = n > 0
    return float((gammaln(n[u] + 1) - (gammaln(n0[u] + .5) - gammaln(.5)) - (gammaln(n1[u] + .5) - gammaln(.5))).sum() / LN2)


INT_JC, INT_KM, INT_NBC = 12, 32, 11
INT_CAUSAL = ((0, 0, 1), (0, 1, 0), (1, 0, 0))


def _klog(u):
    return (np.frexp((u + 1).astype(np.float64))[1] - 1).astype(np.int64)    # floor(log2(u + 1)), exact below 2^53


def int_magnitude_events(v, known):
    """binary event counts of the B4 magnitude stream. For every known voxel (C raster order, after the ternary pass, so its state
    s = [v > 0] is known to the decoder) with magnitude u = v - 1 (s = 1) or -v (s = 0):
      predictor = u of the first causal neighbour among (x, y, z-1), (x, y-1, z), (x-1, y, z) that is known and has the same state;
      if a predictor exists, a flag [u == predictor] is coded with context (s, min(k_pred, 9), number of the other same-state causal
      neighbours that also equal the predictor, 0-2);
      if there is no predictor or the flag is 0, u is exp-Golomb(0) binarised: prefix k = floor(log2(u + 1)) ones + a zero with context
      (s, class, min(bin, 11)), class = 1 + min(k_pred, 9) or 0 without predictor; suffix k bits MSB first with context (s, k, bit).
    Returns (flag n0, flag n1, prefix n0, prefix n1, suffix n0, suffix n1)."""
    v = np.asarray(v).astype(np.int32); known = np.asarray(known, bool)
    stb = v > 0; u = np.where(known, np.where(stb, v - 1, -v), 0).astype(np.int32)
    assert (u >= 0).all()
    has = np.zeros(v.shape, bool); up = np.zeros(v.shape, np.int32); nsame = np.zeros(v.shape, np.int8)
    for dx, dy, dz in INT_CAUSAL:                                       # neighbour (x-dx, y-dy, z-dz), read through views
        X, Y, Z = v.shape; dst = np.s_[dx:, dy:, dz:]; src = np.s_[:X - dx, :Y - dy, :Z - dz]
        sd = np.zeros(v.shape, bool); sd[dst] = known[src] & (stb[src] == stb[dst])
        ud = np.zeros(v.shape, np.int32); ud[dst] = u[src]
        sel = sd & ~has; up[sel] = ud[sel]
        nsame += (sd & has & (ud == up)).astype(np.int8)                 # other same-state neighbours equal to the predictor
        has |= sd
    sel = known & has
    eq = sel & (u == up)
    st = stb.astype(np.int64)[known]; kn_has = has[known]; kn_eq = eq[known]
    kp = np.minimum(_klog(up[known]), INT_NBC - 2)
    f_ctx = (st * (INT_NBC - 1) + kp) * 3 + np.minimum(nsame[known].astype(np.int64), 2)
    nf = 2 * (INT_NBC - 1) * 3
    f1 = np.bincount(f_ctx[kn_has & kn_eq], minlength=nf); fa = np.bincount(f_ctx[kn_has], minlength=nf)
    lit = ~kn_eq
    uu = u[known][lit].astype(np.int64); kk = _klog(uu); sk = st[lit]; ck = np.where(kn_has, kp + 1, 0)[lit]
    assert kk.max(initial=0) < INT_KM
    Hh = np.bincount((sk * INT_NBC + ck) * INT_KM + kk, minlength=2 * INT_NBC * INT_KM).reshape(2 * INT_NBC, INT_KM).astype(np.int64)
    tot = Hh.sum(1, keepdims=True); G = tot - np.cumsum(Hh, 1)               # G[:, j] = #(k > j)
    p0 = np.zeros((2 * INT_NBC, INT_JC), np.int64); p1 = np.zeros_like(p0)
    p0[:, :INT_JC - 1] = Hh[:, :INT_JC - 1]; p0[:, INT_JC - 1] = Hh[:, INT_JC - 1:].sum(1)
    p1[:, :INT_JC - 1] = G[:, :INT_JC - 1]; p1[:, INT_JC - 1] = G[:, INT_JC - 1:].sum(1)
    s0 = np.zeros((2, INT_KM, INT_KM), np.int64); s1 = np.zeros_like(s0)
    for kv in np.unique(kk[kk > 0]):
        sel = kk == kv; rem = uu[sel] + 1 - (1 << int(kv)); ss = sk[sel]
        for b in range(int(kv)):
            bit = (rem >> (int(kv) - 1 - b)) & 1
            ones = np.bincount(ss, weights=bit, minlength=2).astype(np.int64); alln = np.bincount(ss, minlength=2)
            s1[:, kv, b] += ones; s0[:, kv, b] += alln - ones
    return fa - f1, f1, p0, p1, s0, s1


def int_ctx_bytes(v, known):
    """B4 coder: exact adaptive code length of an integer map with a known mask = ternary ctx (sign) + predicted / binarised
    magnitudes (int_magnitude_events), + 4 B header"""
    known = np.asarray(known, bool)
    t = np.where(known, (np.asarray(v) > 0).astype(np.int8), np.int8(-1))
    e0, e1, p0, p1, s0, s1 = int_magnitude_events(v, known)
    return (ctx_tern_bits(t) + kt_bin_bits(e0, e1) + kt_bin_bits(p0, p1) + kt_bin_bits(s0, s1)) / 8 + 4


def gzip_len(arr):
    return len(gzip.compress(np.ascontiguousarray(arr).tobytes(), 9))


def _morton(x, y, z, bits):
    code = np.zeros(np.shape(x), np.int64)
    for b in range(bits):
        code |= (((x >> b) & 1) << (3 * b)) | (((y >> b) & 1) << (3 * b + 1)) | (((z >> b) & 1) << (3 * b + 2))
    return code


def octomap_bt(t, res=0.1, tree_depth=16):
    """OctoMap writeBinary (.bt) of a ternary grid: toMaxLikelihood + prune, depth-16 key space, voxel i at key 32768 + i on each axis
    (the map lies at non-negative coordinates). Child index x | y << 1 | z << 2; per inner node 2 bytes (children 0-3, 4-7), child i
    at bits 2i, 2i+1: 01 free, 10 occupied (bit 2i+1), 11 has children, 00 unknown; pre-order depth first."""
    t = np.asarray(t); X, Y, Z = t.shape
    S = 1 << int(math.ceil(math.log2(max(X, Y, Z)))); dc = int(round(math.log2(S)))
    assert dc < tree_depth
    st = np.zeros((S, S, S), np.int8); st[:X, :Y, :Z] = np.where(t < 0, 0, np.where(t == 1, 2, 1))
    levels = [st]
    for _ in range(dc):
        c = levels[-1]; n = c.shape[0] // 2; ch = c.reshape(n, 2, n, 2, n, 2)
        af = (ch == 1).all((1, 3, 5)); ao = (ch == 2).all((1, 3, 5)); aa = (ch == 0).all((1, 3, 5))
        levels.append(np.where(af, 1, np.where(ao, 2, np.where(aa, 0, 3))).astype(np.int8))
    root = int(levels[dc][0, 0, 0])
    size, recs, keys, levs = 0, [], [], []
    if root != 0:
        size = tree_depth - dc + 1                                          # chain of single-child ancestors + the cube root
        for l in range(dc, 0, -1):
            P = levels[l]; n = P.shape[0]
            child = levels[l - 1].reshape(n, 2, n, 2, n, 2).transpose(0, 2, 4, 5, 3, 1).reshape(n, n, n, 8).astype(np.int64)
            inner = P == 3
            size += int((child[inner] != 0).sum())                          # existing children of inner nodes
            if not inner.any():
                continue
            cc = child[inner]
            b0 = (cc[:, 0] | (cc[:, 1] << 2) | (cc[:, 2] << 4) | (cc[:, 3] << 6)).astype(np.uint8)
            b1 = (cc[:, 4] | (cc[:, 5] << 2) | (cc[:, 6] << 4) | (cc[:, 7] << 6)).astype(np.uint8)
            xi, yi, zi = np.nonzero(inner)
            recs.append(np.stack([b0, b1], 1)); keys.append(_morton(xi, yi, zi, dc - l) << (3 * l)); levs.append(np.full(len(xi), l))
    if recs:
        R_ = np.concatenate(recs); K_ = np.concatenate(keys); L_ = np.concatenate(levs)
        body = R_[np.lexsort((-L_, K_))].tobytes()
    else:
        body = b''
    chain = b''
    if root != 0:
        for Lv in range(tree_depth, dc, -1):                                # node covering 2^Lv keys; child from key bit Lv - 1
            cstate = 3 if Lv - 1 > dc else root
            ci = 7 if Lv == tree_depth else 0
            b = [0, 0]; b[ci // 4] = cstate << (2 * (ci % 4)); chain += bytes(b)
    header = ('# Octomap OcTree binary file\n# (feel free to add / change comments, but leave the first line as it is!)\n#\n'
              f'id OcTree\nsize {size}\nres {res}\ndata\n')
    return header.encode() + chain + body


def bt_gzip_bytes(t):
    return len(gzip.compress(octomap_bt(t), 9))


# ================================================================ fusion on the forest (relay, provenance-checked)
def forest(edges, n):
    und = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j, *_ in edges})
    nb = collections.defaultdict(list)
    for i, j in und:
        nb[i].append(j); nb[j].append(i)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in und:
        parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for a in range(n):
        groups[find(a)].append(a)
    comps = sorted(groups.values())
    comp_of = {a: ci for ci, cm in enumerate(comps) for a in cm}
    directed = [(i, j) for i, j in und] + [(j, i) for i, j in und]
    nbs = {a: sorted(nb[a]) for a in range(n)}

    def side(s, r):                                                     # agents reached from s without crossing to r
        seen, st = {s}, [s]
        while st:
            u = st.pop()
            for w in nbs[u]:
                if w != r and w not in seen:
                    seen.add(w); st.append(w)
        return seen
    sides = {e: sorted(side(*e)) for e in directed}
    order = sorted(directed, key=lambda e: (len(sides[e]), e))           # a message only needs strictly smaller sender sides
    return dict(und=und, directed=directed, nb=nbs, comps=comps, comp_of=comp_of, sides=sides, order=order)


class RelayStats:
    def __init__(self):
        self.combines = 0; self.belief_checks = 0


def _merge(parts, combine, stats):
    prov = 0
    for _, p in parts:
        assert prov & p == 0, 'evidence counted twice'
        prov |= p
    if stats is not None:
        stats.combines += 1
    return combine([x for x, _ in parts]), prov


def relay(own, F, combine, stats=None):
    """sum-product style relay on the forest: msg(s -> r) = combine(own[s], msg(k -> s) for k in nb(s), k != r) (receiver branch
    excluded), belief(a) = combine(own[a], msg(k -> a) for all k). Provenance bitmasks are carried and asserted DISJOINT at every
    combination (no agent's evidence counted twice); every message must carry exactly its sender side and every belief exactly its
    component. (No closures: a recursive closure would keep every message array alive until the cyclic GC runs.)"""
    nb = F['nb']; memo = {}
    for s, r in F['order']:
        memo[(s, r)] = _merge([(own[s], 1 << s)] + [memo[(k, s)] for k in nb[s] if k != r], combine, stats)
        assert memo[(s, r)][1] == sum(1 << b for b in F['sides'][(s, r)]), 'message provenance != sender side'
    beliefs = []
    for a in range(len(own)):
        val, prov = _merge([(own[a], 1 << a)] + [memo[(k, a)] for k in nb[a]], combine, stats)
        assert prov == sum(1 << b for b in F['comps'][F['comp_of'][a]]), 'belief provenance != component'
        beliefs.append(val)
    msgs = {e: memo[e][0] for e in F['directed']}
    memo.clear()
    return msgs, beliefs


def comb_sum(parts):
    S = parts[0][0].copy(); K = parts[0][1].copy()
    for s, k in parts[1:]:
        S += s; K |= k
    return S, K


def _key(L):
    u = is_unk(L); L = L.astype(np.int64); return np.where(u, -1, 2 * np.abs(L) + (L > 0))


def comb_maxabs(parts):
    out = parts[0].copy(); ko = _key(out)
    for p in parts[1:]:
        kp = _key(p); m = kp > ko; out[m] = p[m]; ko[m] = kp[m]
    return out


def comb_maxocc(parts):
    return np.maximum.reduce(parts)


def sum_payload(L):
    k = ~is_unk(L); return np.where(k, L.astype(np.int32), 0), k


def sum_tern(payload):
    S, K = payload; return np.where(K, (S > 0).astype(np.int8), np.int8(-1)).astype(np.int8)


def fuse(maps, F, kind, stats=None, check=True):
    """relay maps (int8 log-odds for sum / maxabs, ternary for max_occ) -> (messages, ternary belief per agent)"""
    if kind == 'sum':
        own = [sum_payload(L) for L in maps]; msgs, bel = relay(own, F, comb_sum, stats)
        if check:
            for ci, cm in enumerate(F['comps']):
                direct = comb_sum([own[a] for a in cm])
                for a in cm:
                    assert np.array_equal(bel[a][0], direct[0]) and np.array_equal(bel[a][1], direct[1]), 'relayed sum != component sum'
                    if stats is not None:
                        stats.belief_checks += 1
        return msgs, [sum_tern(b) for b in bel]
    if kind == 'maxabs':
        msgs, bel = relay(list(maps), F, comb_maxabs, stats)
        if check:
            for cm in F['comps']:
                direct = comb_maxabs([maps[a] for a in cm])
                assert all(np.array_equal(bel[a], direct) for a in cm)
        return msgs, [tern(b) for b in bel]
    if kind == 'max_occ':
        T = [tern(L) for L in maps]
        msgs, bel = relay(T, F, comb_maxocc, stats)
        if check:
            for cm in F['comps']:
                direct = comb_maxocc([T[a] for a in cm])
                assert all(np.array_equal(bel[a], direct) for a in cm)
        return msgs, bel
    raise ValueError(kind)


def stream_targets(own, reached):
    r = np.asarray(reached, bool)
    return np.where(r[:, :, None, None, None], own[None], np.int8(-1)).max(1)


# ================================================================ B5 classical denoise-then-encode
S26 = np.ones((3, 3, 3), bool)


def down_tern(t, f):
    X, Y, Z = t.shape; bx, by, bz = f
    k = (t >= 0).reshape(X // bx, bx, Y // by, by, Z // bz, bz).sum((1, 3, 5)); o = (t == 1).reshape(X // bx, bx, Y // by, by, Z // bz, bz).sum((1, 3, 5))
    return np.where(k > 0, (o >= np.maximum(0.5 * k, 1)).astype(np.int8), np.int8(-1)).astype(np.int8)


def up_tern(c, f):
    return c.repeat(f[0], 0).repeat(f[1], 1).repeat(f[2], 2)


def _psum(m):                                                           # A03 probes/A03/a03np.py, verbatim
    p = np.zeros(tuple(d + 1 for d in m.shape), np.int64)
    p[1:, 1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return p


def _bsum(p, x0, x1, y0, y1, z0, z1):                                   # A03 probes/A03/a03np.py, verbatim
    return (p[x1, y1, z1] - p[x0, y1, z1] - p[x1, y0, z1] - p[x1, y1, z0] + p[x0, y0, z1] + p[x0, y1, z0] + p[x1, y0, z0] - p[x0, y0, z0])


def box_cover(must, forbid):
    """A03 probes/A03/a03np.py box_cover, copied verbatim (= G4-E0 g4_exact_codec.box_cover): greedy maximal boxes seeded at the first
    uncovered must-voxel in (z, x, y) order, grown one layer at a time in 6 directions while the layer holds no forbidden voxel and at
    least one must-voxel. Half-open (x0, x1, y0, y1, z0, z1)."""
    X, Y, Z = must.shape
    pm = _psum(must); pf = _psum(forbid)
    cov = np.zeros_like(must)
    boxes = []
    for z, x, y in np.argwhere(must.transpose(2, 0, 1)):
        if cov[x, y, z]:
            continue
        b = [x, x + 1, y, y + 1, z, z + 1]
        grown = True
        while grown:
            grown = False
            for ax, sgn in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
                lo, hi = 2 * ax, 2 * ax + 1; lim = (X, Y, Z)[ax]
                nb = list(b)
                if sgn > 0:
                    if b[hi] >= lim:
                        continue
                    nb[lo], nb[hi] = b[hi], b[hi] + 1
                else:
                    if b[lo] <= 0:
                        continue
                    nb[lo], nb[hi] = b[lo] - 1, b[lo]
                if _bsum(pf, *nb) == 0 and _bsum(pm, *nb) > 0:
                    if sgn > 0:
                        b[hi] += 1
                    else:
                        b[lo] -= 1
                    grown = True
        cov[b[0]:b[1], b[2]:b[3], b[4]:b[5]] |= must[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
        boxes.append(tuple(int(v) for v in b))
    return boxes


def b5_occ(L, tau, tau_ge=False):
    """B5 occupancy threshold: L > tau (V3 / V0); under V1 L >= tau for tau > 0 (review S4)"""
    known = ~is_unk(L)
    return known, known & ((L >= tau) if (tau_ge and tau > 0) else (L > tau))


def b5_variants(L, tau_ge=False, families=B5_FAMILIES):
    """{(family, tau, m, f): ternary map at the transmitted resolution} for one agent's integer log-odds map; the known set is never
    changed (deleted occupied voxels become free)"""
    out = {}
    for tau in B5_TAUS:
        known, occ = b5_occ(L, tau, tau_ge)
        for fam in families:
            keep = {}
            if fam == 'cc':                                             # 26-connected components with < m voxels deleted
                lab, _ = ndimage.label(occ, S26); sizes = np.bincount(lab.ravel())
                for m in B5_MS:
                    small = sizes < m; small[0] = False; keep[m] = occ & ~small[lab]
            elif fam == 'box':                                          # pre-registered "delete boxes": keep A03 boxes with support >= m
                boxes = box_cover(occ, known & ~occ)
                sup = [int(occ[x0:x1, y0:y1, z0:z1].sum()) for x0, x1, y0, y1, z0, z1 in boxes]
                for m in B5_MS:
                    P = np.zeros(L.shape, bool)
                    for (x0, x1, y0, y1, z0, z1), s in zip(boxes, sup):
                        if s >= m:
                            P[x0:x1, y0:y1, z0:z1] = True
                    keep[m] = P & known
            else:
                raise ValueError(fam)
            for m in B5_MS:
                D = np.where(known, keep[m].astype(np.int8), np.int8(-1)).astype(np.int8)
                for f in B5_DS:
                    out[(fam, tau, m, f)] = D if f == (1, 1, 1) else down_tern(D, f)
    return out


def b5_key(fam, tau, m, f):
    return f'{fam}|{tau}|{m}|{f[0]}{f[1]}{f[2]}'


def _reduce(row):
    return {k: row.get(k) for k in B5_ROW_KEYS}


def run_b5(L_pub, L_k0, has_inter, F, K0T, R8T, rstats, timing, tau_ge=False, truepose=True):
    """all B5 operating points of one tier: R8 rows stored once per component ([row, members]), relay bytes per fusion; the true
    inter-agent pose readout only when truepose (governing integrator); K0 rows need agent-frame maps (None otherwise)"""
    t0 = time.time(); n = len(L_pub)
    sets = [('pub', L_pub)] + ([('truepose', L_k0)] if (has_inter and truepose) else [])
    var = {name: [b5_variants(L, tau_ge) for L in maps] for name, maps in sets}
    k0var = var['pub'] if not has_inter else var.get('truepose')
    fm = [len(F['comps'][F['comp_of'][a]]) - 1 for a in range(n)]
    res = {}
    for key in itertools.product(B5_FAMILIES, B5_TAUS, B5_MS, B5_DS):
        f = key[3]; rec = {}
        for name, _ in sets:
            D = [var[name][a][key] for a in range(n)]
            out = {}
            for fusion in ('sum', 'max'):
                if fusion == 'sum':
                    own = [(np.where(d < 0, 0, np.where(d == 1, 1, -1)).astype(np.int32), d >= 0) for d in D]
                    msgs, bel = relay(own, F, comb_sum, rstats)
                    for cm in F['comps']:
                        direct = comb_sum([own[a] for a in cm])
                        for a in cm:
                            assert np.array_equal(bel[a][0], direct[0]) and np.array_equal(bel[a][1], direct[1])
                            rstats.belief_checks += 1
                    preds = [sum_tern(b) for b in bel]
                    rnet = sum(int_ctx_bytes(S, K) for S, K in msgs.values())
                else:
                    msgs, preds = relay(D, F, comb_maxocc, rstats)
                    rnet = sum(ctx_tern_bytes(T) for T in msgs.values())
                out[fusion] = dict(rnet=float(rnet), R8=[[_reduce(score(up_tern(preds[cm[0]], f), R8T[ci])), len(cm)] for ci, cm in enumerate(F['comps'])])
            rec[name] = out
        own_b = [ctx_tern_bytes(var['pub'][a][key]) for a in range(n)]
        rec['own_bytes'] = own_b
        rec['flood_bytes'] = float(sum(b * k for b, k in zip(own_b, fm)))
        rec['K0_rows'] = [_reduce(score(up_tern(k0var[a][key], f), K0T[a])) for a in range(n)] if k0var is not None else None
        res[b5_key(*key)] = rec
    timing['b5'] = timing.get('b5', 0.) + time.time() - t0
    return res


# ================================================================ one (house, noise seed) job
def _peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, 'peak_wset', getattr(mi, 'rss', 0))) / 2 ** 20
    except Exception:
        return None


def file_sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fingerprint(opts):
    """everything that determines a job's content (review M6), JSON-normalised"""
    return json.loads(json.dumps({k: opts[k] for k in sorted(opts) if k != 'out'}))


def _pipelines(I, cname, L_k0, L_pub, has_inter, F, K0T, R8T, T8, private, fm, rstats, opts, timing):
    """K0, R8 (three fusions, + the true inter-agent pose readout), bytes (B1-B4) and B5 of one integrator on one tier"""
    n = len(L_pub); r = {}; tp = time.time()
    Tk0 = [tern(L) for L in L_k0]
    r['K0'] = [score(Tk0[a], K0T[a], strata=True) for a in range(n)]
    checks = dict(K0_exact=[bool(np.array_equal(Tk0[a], private[a])) for a in range(n)])
    r['R8'] = {}; msgs_pub = {}
    for mapset, maps in (('pub', L_pub), ('truepose', L_k0)) if has_inter else (('pub', L_pub),):
        for kind in FUSIONS:
            msgs, preds = fuse(maps, F, kind, rstats)
            r['R8'][kind if mapset == 'pub' else f'{kind}_truepose'] = [[score(preds[cm[0]], R8T[ci], strata=True), len(cm)] for ci, cm in enumerate(F['comps'])]
            if mapset == 'pub':
                msgs_pub[kind] = msgs
                checks[f'R8_{kind}_exact'] = [bool(np.array_equal(preds[a], T8[a])) for a in range(n)]
    tb = time.time()
    own = dict(B4=[int_ctx_bytes(L, ~is_unk(L)) for L in L_pub], B1_gzip=[gzip_len(L) for L in L_pub],
               B2_bt_gzip=[bt_gzip_bytes(tern(L)) for L in L_pub], B3_ctx=[ctx_tern_bytes(tern(L)) for L in L_pub])
    edges = dict(B4_sum=[int_ctx_bytes(*msgs_pub['sum'][e]) for e in F['directed']],
                 B1_gzip_sum=[gzip_len(pack_sum(*msgs_pub['sum'][e])) for e in F['directed']],
                 B4_maxabs=[int_ctx_bytes(msgs_pub['maxabs'][e], ~is_unk(msgs_pub['maxabs'][e])) for e in F['directed']],
                 B3_ctx_maxocc=[ctx_tern_bytes(msgs_pub['max_occ'][e]) for e in F['directed']],
                 B2_bt_gzip_maxocc=[bt_gzip_bytes(msgs_pub['max_occ'][e]) for e in F['directed']])
    fl = lambda xs: float(sum(x * k for x, k in zip(xs, fm)))  # noqa: E731
    rnet = dict(B1_gzip_flood=fl(own['B1_gzip']), B1_gzip_relay=float(sum(edges['B1_gzip_sum'])),
                B2_bt_gzip_flood=fl(own['B2_bt_gzip']), B2_bt_gzip_relay=float(sum(edges['B2_bt_gzip_maxocc'])),
                B3_ctx_flood=fl(own['B3_ctx']), B3_ctx_relay=float(sum(edges['B3_ctx_maxocc'])),
                B4_int_relay=float(sum(edges['B4_sum'])), B4_flood=fl(own['B4']), B4_maxabs_relay=float(sum(edges['B4_maxabs'])))
    # review M4: B4 = min(integer ctx coder, structure code); the structure code (S0) applies only to the exact clean maps
    clean_exact = cname == 'clean' and all(all(v) for v in checks.values())
    rnet['B4_struct_relay'] = float(opts['s0_rnet']) if clean_exact else None
    rnet['B4_relay'] = min(rnet['B4_int_relay'], rnet['B4_struct_relay']) if clean_exact else rnet['B4_int_relay']
    rnet['B4_best_info'] = min(rnet['B4_relay'], rnet['B4_flood'])                         # information only
    r['bytes_own'] = own; r['bytes_edges'] = edges; r['rnet'] = rnet; r['checks'] = checks
    timing[f'{cname}_{I}_maps'] = tb - tp; timing[f'{cname}_{I}_bytes'] = time.time() - tb
    if cname in (opts['b5_conds'] if I == opts['governing'] else opts['b5_conds_sens']):
        r['B5'] = run_b5(L_pub, L_k0, has_inter, F, K0T, R8T, rstats, timing, tau_ge=(I == 'V1'), truepose=(I == opts['governing']))
        r['B5_tau_rule'] = '>= (tau > 0)' if I == 'V1' else '>'
    return r


def _ck(c):
    """the parameters that determine an agent-frame (K0) map: tiers that differ only in the inter-agent error share them"""
    return tuple(c[x] for x in ('sigma0', 'k', 'p_drop', 'p_out', 'pose_t', 'pose_yaw'))


def run_job(job):
    panel, rooms, index, seed, opts = job
    assert file_sha256(os.path.abspath(__file__)) == opts['script_sha256'], 'g4_noise_panels.py changed since the run started'
    name = f'r{rooms}_i{index}'; fp = fingerprint(opts)
    t_start = time.time(); timing = {}
    try:
        fx = W.make_fixture('VALIDATION', index, rooms)
    except ValueError as e:                                             # only agent placement may skip a house (review v420 S7)
        return dict(panel=panel, house=name, rooms=rooms, index=index, seed=seed, skipped=repr(e)[:200], fingerprint=fp)
    timing['fixture'] = time.time() - t_start
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape
    ds = 0.5 * float(cell.min())
    H = House(occ, fx['slab'], fx['obj'], cell)
    F = forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][HOPS.index(8)], bool)
    for a in range(n):                                                  # R8 = the whole tree component (hop 8 >= any diameter)
        assert set(np.nonzero(reached8[a])[0].tolist()) == set(F['comps'][F['comp_of'][a]]), 'reached8 != component'
    T8 = stream_targets(private, reached8)
    K0T = [Target(private[a], H) for a in range(n)]
    R8T = [Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    t0 = time.time()
    traces = [[trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    timing['trace'] = time.time() - t0
    fm = [len(F['comps'][F['comp_of'][a]]) - 1 for a in range(n)]
    rstats = RelayStats(); ints = list(opts['integrators'])
    out = dict(panel=panel, house=name, rooms=rooms, index=index, seed=seed, agents=n, chart_m=float(fx['chart_m']), cell=cell.tolist(),
               n_objects=int(fx['n_objects']), components=F['comps'], directed_edges=len(F['directed']), world_id=fx['world_id'],
               private_sha256=W.array_hash(private), fingerprint=fp, integrator_defs={I: INTEGRATOR_DEFS[I] for I in ints}, conds={})
    k0cache = {}
    for cname in opts['conds']:
        c = CONDITIONS[cname]; tc = time.time()
        has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0
        ck = _ck(c)
        L_k0 = {I: [] for I in ints}; L_pub = {I: [] for I in ints}; stats = []; frames = []
        for a in range(n):
            frame = None
            if has_inter:
                z = frame_draws(seed, rooms, index, a)
                frame = (rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
                frames.append([float(z[0] * c['inter_t']), float(z[1] * c['inter_t']), float(z[2] * c['inter_yaw'])])
            scans = []
            for j in range(traj.shape[0]):
                hs, ls = traces[a][j]
                scan, st = measure(c, hs, ls, traj[j, a], pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)
                scans.append(scan); stats.append(st)
            if (ck, a) not in k0cache:
                k0cache[(ck, a)] = integrate_agent(shape, cell, ds, scans, ints)
            mk = k0cache[(ck, a)]
            mp = integrate_agent(shape, cell, ds, [to_public(s, frame) for s in scans], ints) if has_inter else mk
            for I in ints:
                L_k0[I].append(mk[I]); L_pub[I].append(mp[I])
            del scans
        timing[f'{cname}_integrate'] = time.time() - tc
        rec = dict(noise=dict(beams=sum(s['beams'] for s in stats), dropped=sum(s['dropped'] for s in stats), hits=sum(s['hits'] for s in stats),
                              hits_kept=sum(s['hits_kept'] for s in stats), outliers=sum(s['outliers'] for s in stats), n_err=sum(s['n_err'] for s in stats),
                              s_err=sum(s['s_err'] for s in stats), s_err2=sum(s['s_err2'] for s in stats), s_rel=sum(s['s_rel'] for s in stats),
                              s_rel2=sum(s['s_rel2'] for s in stats), clipped=sum(s['clipped'] for s in stats),
                              pose=[s['pose'] for s in stats], frame=frames))
        rec['by_int'] = {I: _pipelines(I, cname, L_k0[I], L_pub[I], has_inter, F, K0T, R8T, T8, private, fm, rstats, opts, timing) for I in ints}
        timing[f'{cname}_total'] = time.time() - tc
        out['conds'][cname] = rec
        del L_pub, L_k0
        later = {_ck(CONDITIONS[c2]) for c2 in opts['conds'][opts['conds'].index(cname) + 1:]}
        for key in [k for k in k0cache if k[0] not in later]:           # keep agent-frame maps only while a later tier reuses them
            del k0cache[key]
        gc.collect()
    out['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    timing['total'] = time.time() - t_start
    out['timing'] = timing; out['peak_mb'] = _peak_mb()
    return out


def _job_path(out_dir, panel, rooms, index, seed):
    return os.path.join(out_dir, 'jobs', f'{panel}_r{rooms}_i{index}_s{seed}.json')


def _worker_init(json_dir, index_file):
    setup_env(json_dir, index_file)


def _run_and_save(job):
    panel, rooms, index, seed, opts = job
    res = run_job(job)
    path = _job_path(opts['out'], panel, rooms, index, seed); tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return path, res.get('timing', {}).get('total'), res.get('peak_mb'), res.get('skipped')


# ================================================================ aggregation and verdict
def _mean(v):
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None


def _med(v):
    v = [x for x in v if x is not None]; return float(np.median(v)) if v else None


def _expand(comp_rows):
    return [row for row, k in comp_rows for _ in range(k)]


def _pooled_f1(rows):
    tp = sum(r['furn_tp'] for r in rows if r.get('furn_tp') is not None); fp = sum(r['furn_fp'] for r in rows if r.get('furn_fp') is not None)
    fn = sum(r['furn_fn'] for r in rows if r.get('furn_fn') is not None)
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


def _strata(rows):
    acc = {}
    for r in rows:
        for k, v in (r.get('strata') or {}).items():
            a = np.asarray(v, np.int64); acc[k] = a if k not in acc else acc[k] + a
    out = {}
    for k, a in acc.items():
        if k.endswith('furn'):
            out[k] = [dict(f1=(2 * x[0] / max(2 * x[0] + x[1] + x[2], 1)) if x.sum() else None, prec=(x[0] / (x[0] + x[1])) if x[0] + x[1] else None,
                           n_target=int(x[0] + x[2])) for x in a]
        else:
            out[k] = [dict(recall=(x[0] / x[1]) if x[1] else None, n_target=int(x[1])) for x in a]
    return out


def summarise_rows(rows, keys=ROW_KEYS):
    s = {k: _mean([r.get(k) for r in rows]) for k in keys if k not in ('furn_tp', 'furn_fp', 'furn_fn')}
    if rows and 'furn_tp' in rows[0]:
        s['furn_f1_pooled'] = _pooled_f1(rows)
    if rows and 'strata' in rows[0]:
        s['strata'] = _strata(rows)
    return s


def hull_value(points, x0):
    """upper convex hull (time sharing between two operating points) of (rate, F1) evaluated at rate x0: the best F1 any mixture with
    rate <= x0 reaches; None if no point has rate <= x0"""
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    best = None
    for x, y in pts:
        if x <= x0:
            best = y if best is None else max(best, y)
    if best is None:
        return None
    for (x1, y1), (x2, y2) in itertools.product(pts, pts):
        if x1 < x0 < x2:
            y = y1 + (y2 - y1) * (x0 - x1) / (x2 - x1); best = max(best, y)
    return best


def noise_summary(recs, c):
    """pooled injected-noise statistics of one tier vs its settings"""
    N = collections.Counter()
    poses, frames = [], []
    for r in recs:
        for k in ('beams', 'dropped', 'hits', 'hits_kept', 'outliers', 'n_err', 'clipped'):
            N[k] += r[k]
        for k in ('s_err', 's_err2', 's_rel', 's_rel2'):
            N[k] += r[k]
        poses += r['pose']; frames += r['frame']
    n = max(N['n_err'], 1)
    P = np.asarray(poses, np.float64).reshape(-1, 4); Fr = np.asarray(frames, np.float64).reshape(-1, 3)
    return dict(drop_frac=N['dropped'] / max(N['beams'], 1), drop_set=c['p_drop'], outlier_frac=N['outliers'] / max(N['hits_kept'], 1), outlier_set=c['p_out'],
                err_mean=N['s_err'] / n, err_std=math.sqrt(max(N['s_err2'] / n - (N['s_err'] / n) ** 2, 0.)), err_set_sigma0=c['sigma0'],
                rel_std=math.sqrt(max(N['s_rel2'] / n - (N['s_rel'] / n) ** 2, 0.)), rel_set_k=c['k'], clipped=N['clipped'], n_range=N['n_err'],
                pose_xy_std=float(np.sqrt((P[:, :2] ** 2).mean())) if len(P) else None, pose_z_std=float(np.sqrt((P[:, 2] ** 2).mean())) if len(P) else None,
                pose_yaw_std=float(np.sqrt((P[:, 3] ** 2).mean())) if len(P) else None, pose_set=(c['pose_t'], c['pose_yaw']), n_pose=len(P),
                frame_xy_std=float(np.sqrt((Fr[:, :2] ** 2).mean())) if len(Fr) else None, frame_yaw_std=float(np.sqrt((Fr[:, 2] ** 2).mean())) if len(Fr) else None,
                frame_set=(c['inter_t'], c['inter_yaw']), n_frame=len(Fr))


def noise_ok(ns):
    """loose consistency of injected vs set noise (sampling tolerances); used by --smoke-asserts"""
    ok = abs(ns['drop_frac'] - ns['drop_set']) < 0.003 and abs(ns['outlier_frac'] - ns['outlier_set']) < max(0.0015, 0.2 * ns['outlier_set'])
    if ns['err_set_sigma0'] > 0:
        ok &= abs(ns['err_std'] / ns['err_set_sigma0'] - 1) < 0.03
    if ns['rel_set_k'] > 0:
        ok &= abs(ns['rel_std'] / ns['rel_set_k'] - 1) < 0.03
    if ns['err_set_sigma0'] == 0 and ns['rel_set_k'] == 0:
        ok &= ns['err_std'] == 0
    for got, want, cnt in ((ns['pose_xy_std'], ns['pose_set'][0], 2 * ns['n_pose']), (ns['pose_yaw_std'], ns['pose_set'][1], ns['n_pose']),
                           (ns['frame_xy_std'], ns['frame_set'][0], 2 * ns['n_frame']), (ns['frame_yaw_std'], ns['frame_set'][1], ns['n_frame'])):
        if got is None or cnt == 0:
            continue
        ok &= (got == 0) if want == 0 else abs(got / want - 1) < 4.0 / math.sqrt(2 * cnt)   # ~4 sigma of a sample RMS
    return bool(ok)


def summarise_int(X, s0):
    """one integrator on one tier over the houses of one (panel, seed)"""
    d = dict(K0=summarise_rows([r for x in X for r in x['K0']]), R8={})
    for key in X[0]['R8']:
        d['R8'][key] = summarise_rows([r for x in X for r in _expand(x['R8'][key])])
    d['rnet_median'] = {k: _med([x['rnet'][k] for x in X]) for k in X[0]['rnet']}
    d['rnet_mean'] = {k: _mean([x['rnet'][k] for x in X]) for k in X[0]['rnet']}
    d['own_bytes_median'] = {k: _med([b for x in X for b in x['bytes_own'][k]]) for k in X[0]['bytes_own']}
    d['exact'] = {k: all(all(v) for v in (x['checks'][k] for x in X)) for k in X[0]['checks']}
    d['b4_relay_equals_s0_all_houses'] = all(x['rnet']['B4_relay'] == s0 for x in X)
    if all('B5' in x for x in X):
        B5 = {}
        for key in X[0]['B5']:
            e = dict(own_bytes_median=_med([b for x in X for b in x['B5'][key]['own_bytes']]),
                     K0=(summarise_rows([r for x in X for r in x['B5'][key]['K0_rows']], B5_ROW_KEYS)
                         if all(x['B5'][key]['K0_rows'] is not None for x in X) else None))
            for ms in ('pub', 'truepose'):
                if ms in X[0]['B5'][key]:
                    for fu in ('sum', 'max'):
                        e[f'{ms}_{fu}'] = dict(rnet_median=_med([x['B5'][key][ms][fu]['rnet'] for x in X]),
                                               R8=summarise_rows([r for x in X for r in _expand(x['B5'][key][ms][fu]['R8'])], B5_ROW_KEYS))
            B5[key] = e
        d['B5'] = B5; d['B5_tau_rule'] = X[0].get('B5_tau_rule')
        q = 0.25 * d['rnet_median']['B4_relay']
        for ms in ('pub', 'truepose'):
            for fams, tag in ((B5_FAMILIES, ''), (('cc',), '_cc'), (('box',), '_box')):
                pts = [(e[f'{ms}_{fu}']['rnet_median'], e[f'{ms}_{fu}']['R8']['furn_f1']) for key, e in B5.items()
                       if key.split('|')[0] in fams for fu in ('sum', 'max') if f'{ms}_{fu}' in e]
                if pts:
                    h = dict(at_quarter_B4=hull_value(pts, q), quarter_B4_bytes=q, n_points=len(pts),
                             best_point_le_quarter=max([y for x, y in pts if x <= q], default=None), best_any=max(y for _, y in pts))
                    if not tag:
                        h['points'] = [[x, y] for x, y in pts]
                    d[f'B5_hull_{ms}{tag}'] = h
    return d


def verdict(seeds, s0, I, expected, judged):
    """D8-36 criteria for integrator I on the cam tier. seeds: {seed: per-seed summary}. Readouts are always computed for the seeds
    present; the verdict is INCOMPLETE unless the seeds are exactly {1, 2} with the same complete house set and every input (review M5)."""
    reasons = []
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
        miss = [c for c in ('clean', 'cam', 'cam_nf') if c not in C or I not in C[c]]
        if miss:
            reasons.append(f'seed {s}: tiers {miss} missing'); per[s] = dict(missing=miss); continue
        cl, cam, nf = C['clean'][I], C['cam'][I], C['cam_nf'][I]
        hull = cam.get('B5_hull_pub')
        if hull is None:
            reasons.append(f'seed {s}: cam has no B5 for {I}')
        a_map = all(cl['exact'].values()) and all(cl['K0'].get(k) == v and cl['R8']['sum'].get(k) == v
                                                  for k, v in (('free_recall', 1.), ('unknown_known_fpr', 0.), ('accuracy', 1.)))
        a_bytes = bool(cl['b4_relay_equals_s0_all_houses'])
        sm = cam['R8']['sum']; b4f = sm['furn_f1']; b4r = cam['rnet_median']['B4_relay']
        b5q = hull['at_quarter_B4'] if hull else None
        b = dict(furn_le_085=b4f <= .85, rnet_ge_15x_s0=b4r >= 1.5 * s0, b5_le_b4_plus2=(hull is not None) and (b5q is None or b5q <= b4f + .02))
        stop = dict(furn_ge_092=b4f >= .92, rnet_lt_13x_s0=b4r < 1.3 * s0)
        c = {}
        for vs in ('max_occ', 'maxabs'):
            mx = cam['R8'][vs]
            c[vs] = dict(furn_gain=sm['furn_f1'] - mx['furn_f1'], fpr_delta=sm['unknown_known_fpr'] - mx['unknown_known_fpr'],
                         free_delta=sm['free_recall'] - mx['free_recall'])
            c[vs]['pass'] = c[vs]['furn_gain'] >= .03 and c[vs]['fpr_delta'] <= .003 and c[vs]['free_delta'] >= 0
        dl = nf['R8']['sum']['furn_f1'] - b4f
        per[s] = dict(a_map_exact=a_map, a_b4_bytes_equal_s0=a_bytes, a_clean_exact=a_map and a_bytes,
                      B4_R8_furn=b4f, B4_K0_furn=cam['K0']['furn_f1'], B4_rnet_median=b4r, B4_rnet_over_s0=b4r / s0,
                      info_B4_int_cam_over_int_clean=cam['rnet_median']['B4_int_relay'] / cl['rnet_median']['B4_int_relay'],
                      info_B4_best_rnet_median=cam['rnet_median'].get('B4_best_info'),
                      B5_hull_at_quarter_B4=b5q, B5_note=('B5 cannot run at <= 1/4 B4 (arm passes)' if hull and b5q is None else None),
                      B5_hull_cc=(cam.get('B5_hull_pub_cc') or {}).get('at_quarter_B4'), B5_hull_box=(cam.get('B5_hull_pub_box') or {}).get('at_quarter_B4'),
                      B5_truepose_hull=(cam.get('B5_hull_truepose') or {}).get('at_quarter_B4'), B5_best_any=(hull or {}).get('best_any'),
                      b=b, b_pass=all(b.values()), stop=stop, stop_pass=any(stop.values()), c=c, c_pass=all(v['pass'] for v in c.values()),
                      maxabs_R8_furn=cam['R8']['maxabs']['furn_f1'], max_occ_R8_furn=cam['R8']['max_occ']['furn_f1'],
                      d_cam_nf_minus_cam=dl, d_registration_required=dl >= .10)
    ok = [s for s in per if 'missing' not in per[s]]
    if reasons:
        v = 'INCOMPLETE: ' + '; '.join(reasons)
    elif not all(per[s]['a_clean_exact'] for s in ok):
        v = 'INCONCLUSIVE (clean self-check (a) failed: bug)'
    elif all(per[s]['b_pass'] for s in ok):
        v = 'LEARNING_ROOM (b): go to E2-E5 (needs user-approved leases)'
    elif all(per[s]['stop_pass'] for s in ok):
        v = 'STOP: no learning room under noise; S0 + exact lattice layer are the system result'
    else:
        v = 'AMBIGUOUS'
    if not judged:
        v = f'NOT JUDGED (sensitivity row {I}); readout: ' + v
    c_all = bool(ok) and all(per[s]['c_pass'] for s in ok); d_all = bool(ok) and all(per[s]['d_registration_required'] for s in ok)
    return dict(integrator=I, judged=judged, complete=not reasons, incomplete_reasons=reasons, per_seed=per, verdict=v,
                fusion_operator=('sum' if c_all else 'undecided (c not met)') + ('' if not reasons else ' [provisional: incomplete]'),
                registration_panel=('must be reported' if d_all else 'not required by (d)') + ('' if not reasons else ' [provisional: incomplete]'))


def aggregate(out_dir):
    import glob
    paths = sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))
    if not paths:
        raise SystemExit(f'no job files in {out_dir}/jobs')
    jobs = [json.load(open(p)) for p in paths]
    fps = {json.dumps(j.get('fingerprint'), sort_keys=True) for j in jobs}
    if len(fps) != 1:                                                   # review M6: never mix script versions or options
        raise SystemExit(f'refusing to aggregate {out_dir}: {len(fps)} different job fingerprints (mixed script versions / options)')
    fp = jobs[0]['fingerprint']
    if fp is None or fp.get('job_format') != JOB_FORMAT:
        raise SystemExit(f'refusing to aggregate {out_dir}: job format {None if fp is None else fp.get("job_format")} != {JOB_FORMAT}')
    s0 = float(fp['s0_rnet']); ints = fp['integrators']; gov = fp['governing']
    S = dict(fingerprint=fp, governing_integrator=gov, integrators=ints, integrator_defs={I: INTEGRATOR_DEFS[I] for I in ints}, s0_rnet_clean=s0, panels={},
             skipped=[dict(panel=j['panel'], house=j['house'], seed=j['seed'], reason=j['skipped']) for j in jobs if 'skipped' in j])
    for panel in sorted({j['panel'] for j in jobs}):
        expected = frozenset(f'r{r}_i{i}' for r, i in panel_specs(panel))
        P = dict(seeds={})
        for seed in sorted({j['seed'] for j in jobs if j['panel'] == panel}):
            J_all = [j for j in jobs if j['panel'] == panel and j['seed'] == seed]
            J = [j for j in J_all if 'skipped' not in j]
            conds = [c for c in COND_ORDER if J and all(c in j['conds'] for j in J)]
            D = dict(houses=sorted(j['house'] for j in J), skipped=sorted(j['house'] for j in J_all if 'skipped' in j), n_houses=len(J), conds={})
            for c in conds:
                R = [j['conds'][c] for j in J]
                d = dict(noise=noise_summary([x['noise'] for x in R], CONDITIONS[c]))
                d['noise_ok'] = noise_ok(d['noise'])
                for I in ints:
                    d[I] = summarise_int([x['by_int'][I] for x in R], s0)
                D['conds'][c] = d
            P['seeds'][str(seed)] = D
        P['verdict'] = dict(governing_integrator=gov, **{I: verdict(P['seeds'], s0, I, expected, judged=(I == gov)) for I in ints})
        S['panels'][panel] = P
    val = S['panels'].get('val', {}).get('verdict', {}); fresh = S['panels'].get('fresh', {}).get('verdict', {})
    S['decision'] = dict(governing_integrator=gov, judged_panel='val', verdict=(val.get(gov) or {}).get('verdict'),
                         fusion_operator=(val.get(gov) or {}).get('fusion_operator'), registration_panel=(val.get(gov) or {}).get('registration_panel'),
                         sensitivity={I: (val.get(I) or {}).get('verdict') for I in ints if I != gov},
                         report_panel='fresh', fresh={I: (fresh.get(I) or {}).get('verdict') for I in ints})
    return S


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--panels', default='val,fresh')
    ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--houses', type=int, default=0, help='first N houses of each panel (0 = all 24)')
    ap.add_argument('--conds', default=','.join(COND_ORDER))
    ap.add_argument('--b5-conds', default='lidar,cam,harsh', help='tiers with B5 (+ the true-pose readout) for the governing integrator')
    ap.add_argument('--b5-conds-sens', default='cam', help='tiers with B5 for the sensitivity integrators (V1, V0)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--s0-rnet', type=float, default=S0_RNET_CLEAN, help='clean S0 R_net reference, bytes / house (G4-E0 median if available)')
    ap.add_argument('--houseexpo-json-dir', default=HOUSEEXPO_JSON_DIR)
    ap.add_argument('--houseexpo-index', default=HOUSEEXPO_INDEX)
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--smoke-asserts', action='store_true', help='assert clean exactness (maps and bytes), noise statistics after aggregation')
    args = ap.parse_args(argv)
    setup_env(args.houseexpo_json_dir, args.houseexpo_index)
    os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
    conds = [c for c in args.conds.split(',') if c]
    for c in conds:
        assert c in CONDITIONS, c
    if 'clean' not in conds:
        conds = ['clean'] + conds
    conds = [c for c in COND_ORDER if c in conds]                      # cam_nf before cam (shared agent-frame maps)
    opts = dict(out=args.out, conds=conds, b5_conds=[c for c in args.b5_conds.split(',') if c], b5_conds_sens=[c for c in args.b5_conds_sens.split(',') if c],
                integrators=list(INTEGRATORS), governing=GOVERNING, s0_rnet=float(args.s0_rnet), job_format=JOB_FORMAT,
                script_sha256=file_sha256(os.path.abspath(__file__)), data_module_sha256=file_sha256(W.__file__),
                houseexpo_index_sha256=file_sha256(args.houseexpo_index))
    fp = fingerprint(opts)
    t0 = time.time()
    if not args.aggregate_only:
        jobs = []
        for panel in args.panels.split(','):
            specs = panel_specs(panel)
            if args.houses:
                specs = specs[:args.houses]
            for seed in [int(s) for s in args.seeds.split(',')]:
                for rooms, index in specs:
                    path = _job_path(args.out, panel, rooms, index, seed)
                    if os.path.exists(path):                            # resume only with the identical fingerprint (review M6)
                        if json.load(open(path)).get('fingerprint') != fp:
                            raise SystemExit(f'{path} was written with another script version / options; use a fresh --out')
                        continue
                    jobs.append((panel, rooms, index, seed, opts))
        print(f'{len(jobs)} jobs, {args.workers} workers', flush=True)
        if jobs:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.houseexpo_json_dir, args.houseexpo_index), maxtasksperchild=1) as pool:
                for i, (path, sec, mb, sk) in enumerate(pool.imap_unordered(_run_and_save, jobs), 1):
                    print(f'[{i}/{len(jobs)}] {os.path.basename(path)} {sec if sec is None else round(sec)} s peak {mb if mb is None else round(mb)} MB'
                          + (f' SKIPPED {sk}' if sk else ''), f'{time.time() - t0:.0f}s', flush=True)
    S = aggregate(args.out)
    if S['fingerprint'] != fp:
        print('WARNING: the job files were written by another script version / options than this invocation; summary reflects the job files', flush=True)
    S['args'] = vars(args)
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1)
    meta = dict(script=os.path.basename(__file__), fingerprint=fp, args=vars(args), conditions={c: CONDITIONS[c] for c in conds}, seconds=time.time() - t0)
    with open(os.path.join(args.out, 'aggregate.json' if args.aggregate_only else 'run.json'), 'w') as fh:   # review S6
        json.dump(meta, fh, indent=1)
    for panel, P in S['panels'].items():
        for I in S['integrators']:
            V = P['verdict'][I]
            print(panel, I, V['verdict'], '|', V['fusion_operator'], '|', V['registration_panel'], flush=True)
            for s, x in V['per_seed'].items():
                if 'missing' not in x:
                    print(f"   seed {s}: a {x['a_clean_exact']} | B4 R8 furn {x['B4_R8_furn']:.4f} K0 {x['B4_K0_furn']:.4f} R_net {x['B4_rnet_median']:.0f} "
                          f"({x['B4_rnet_over_s0']:.1f} x S0) | B5@1/4 {x['B5_hull_at_quarter_B4']} (cc {x['B5_hull_cc']}, box {x['B5_hull_box']}) | "
                          f"c: -max_occ {x['c']['max_occ']['furn_gain']:+.4f} -maxabs {x['c']['maxabs']['furn_gain']:+.4f} | d {x['d_cam_nf_minus_cam']:+.4f}", flush=True)
    if args.smoke_asserts:
        for panel, P in S['panels'].items():
            for seed, D in P['seeds'].items():
                for I in S['integrators']:
                    cl = D['conds']['clean'][I]
                    assert all(cl['exact'].values()), ('clean not exact', I, cl['exact'])
                    assert cl['b4_relay_equals_s0_all_houses'] and cl['rnet_median']['B4_relay'] == S['s0_rnet_clean'], ('clean B4 != S0', I)
                    assert cl['K0']['free_recall'] == 1. and cl['R8']['sum']['free_recall'] == 1. and cl['K0']['unknown_known_fpr'] == 0.
                    if 'inter_only' in D['conds']:
                        assert D['conds']['inter_only'][I]['exact']['K0_exact'], ('inter_only K0 must be the clean map', I)
                for c, d in D['conds'].items():
                    assert d['noise_ok'], (c, d['noise'])
        print('smoke asserts ok')
    return S


if __name__ == '__main__':
    main()
