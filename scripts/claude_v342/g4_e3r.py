"""g4_e3r.py (Claude, 2026-10-27): batch E3-R "读出端学习先验" (direction D8-43, pre-registration D8-44,
one 5090 lease; code review claude_handoff/analysis_20261027/g4/code_review_e3r.md, changes M1-M5 / S1-S3 applied). The verdict reads
its margins from the pre-registered criteria JSON (D8-44); without margins the aggregation returns the readouts and 'PENDING_CRITERIA'. Imports the FROZEN g4_noise_panels.py (G4-E1, sha 2313d75...) read-only; never imports g4_calib_stalk.py (E1c),
only reads its fit.json (a public table).

Floor (D8-43, RC 2.1). Every agent's stalk e_a is unchanged and non-zero only on the voxels the agent knows; fusion is E1's forest
relay (G.relay / G.comb_sum, provenance bitmasks asserted disjoint, every belief asserted equal to the direct component sum). A message
carries TWO integer sums: S = sum_a e_a and k = sum_a 1[a knows v] (relayed as two payloads, both asserted). The learned prior is a
network D_phi that reads ONLY the fused (S, k) and the normalised height, once, at the read-out:
    logit = D_phi(S x 0.05 / 4, log(1 + k), z)      (no l0 in the input, no x / y coordinates)
Read-out: known iff k > 0 (asserted bit-for-bit equal to B4's known mask, so unknown->known FPR is the baseline's); occupied iff
logit >= t. K0 = the same network on one agent's (e_a, 1[a knows]); R8 = on the component sum. The network never fills unknown voxels.

Evidence function (pluggable; one per trained network; recorded with the table itself in the checkpoint):
  'L'                        e = V3 integer log-odds L (x 0.05 nats); linear read-out B4: sum L > 0 (K0: L > 0);
  'c:<fit.json>:<half>:<tier>'  e = E1c's quantised table cq(L) (x 0.05); linear read-out B6: sum cq > theta (E1c's theta = tq - l0q
                             of that half / tier; K0: cq(L_a) > theta).

Network: 3-level 3D U-Net (channels w, 2w, 4w; w = 16), two 3x3x3 conv + ReLU per level, max-pool 2 / transposed conv 2, 1x1 head,
no normalisation layers (the output at a voxel depends only on its receptive field, ~44 voxels; asserted >= 9 in the tests),
~0.34 M parameters (<= 1 M asserted). Grid 128 x 128 x 32 (R1).

Data (TRAIN only; TEST is never generated, VAL / fresh never touched by gen / train): scene_plan('TRAIN', 70) (210 houses, all three
strata); held-out houses = TRAIN index % 10 == 9 (21 houses; threshold t and the holdout curve), the rest train. Per house the cache
holds the clean V3 maps (seed-free) and the cam_nf V3 maps of noise seeds 200, 201, 202 (agent frame = public frame on cam_nf), the
private clean maps (= the scoring scope) and occ / slab / obj; int8, np.savez_compressed, one npz per house, a manifest with every
file's sha256. Calls = E1 run_job (house_ctx / tier_maps below are its V3-only copy, E1c's too). Training sample = (house, tier
clean 10 % / cam_nf 90 % (seed uniform), component, subset s of the component's agents with |s| uniform in 1..n, dihedral xy
augmentation); inputs summed on the device from the cached int8 maps; label = clean occupancy; loss = BCE on the scope = the subset's
clean-known region (union of its private maps, E1's Target known) INTERSECTED with k > 0 (--loss-mask scope_known, default; the
literal 'scope' is the alternative, see the report). Optimiser AdamW, warm-up + cosine; the trainer stops at --deadline-epoch.

Threshold t (after training, on the held-out TRAIN houses, cam_nf draws, R8 component sums, E1's component expansion): pooled
furniture F1 maximal subject to pooled free recall >= that of the SAME evidence function's linear read-out (B4: sum L > 0; B6: sum cq >
theta); logits are binned at 0.02 in [-40, 40) and t is a bin edge (the read-out uses the bin index, so selection and read-out agree
exactly on one device; the selection runs with TF32 off, GPU vs CPU fp32 logits differ by ~1e-5); ties -> the smallest t. K0 (D8-44
guard): the same rule on single agents gives the network's t_K0 and the linear read-out's integer theta_K0 (NET / LIN K0_sep rows).

Eval (CPU, after the lease): export = VAL / fresh x noise seeds 1, 2 x tiers cam_nf / cam / clean, the E1 calls (common random numbers);
the V3 maps must reproduce the G4-E1 job files' K0 rows AND R8 sum rows value for value (else the job is an E1 mismatch -> INCOMPLETE).
eval = for every checkpoint: NET (network read-out) and LIN (its evidence's linear read-out) on K0 / R8, plus the controls B4 (= E1)
and every given B6 table; rows = G.score (E1 metrics: furniture F1, tol 1, surface, variable, free recall, FPR, strata by height and
clean multiplicity) + strata by k (noisy multiplicity) + calibration histograms per k (network logit; linear (l0q + S) x 0.05).
RC 2.2: the slope ratio kappa and the pair correlation rho_hat test conditional independence of the STALKS; E3-R leaves the stalks
unchanged, so rho_hat is the evidence function's (E1c / RC) and is not recomputed; kappa is reported for the linear sum (= the stalks)
and the per-k slopes / ECE of the network logit are calibration readouts of the read-out, not an independence test.

Usage (always python -B; CPU stages with CUDA_VISIBLE_DEVICES= ):
  gen       --cache C [--seeds 200,201,202] [--houses r:i,...] [--workers 6]       (TRAIN cache; ends with C/manifest.json)
  train     --cache C --evidence E --seed S --out R --updates N [--device cuda] [--batch 8] [--max-updates M] [--deadline-epoch T]
  select    --run R [--device cpu]                                                  (re-run the threshold selection of a trained run)
  export    --cache X --out O [--panels val,fresh] [--seeds 1,2] [--houses N] [--workers 6]
  eval      --export X --out O --ckpt R1/checkpoints/final.pt,R2/... [--b6 c:fit.json:A:cam_nf ...] [--workers 6] [--threads 2]
  aggregate --out O [--criteria crit.json] [--runs R1,R2]
  e1c-rule  --summary e1c_summary.json --fit e1c_fit.json --ref-fit-path P [--expect-342101 E --expect-342202 E]   (D8-44 evidence rule)
  fingerprint --npz X.npz [--write J] [--expect J] [--arrays a,b]                (review S1: cross-platform export check)
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
_CVD = os.environ.get('CUDA_VISIBLE_DEVICES')                  # g4_noise_panels blanks it at import; restored below for the trainer
_HX = (os.environ.get('HOUSEEXPO_JSON_DIR'), os.environ.get('HOUSEEXPO_INDEX'))   # g4_noise_panels overwrites both at import
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, collections, contextlib, glob, hashlib, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_noise_panels as G  # noqa: E402   (frozen; read only)

if _CVD is None:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = _CVD
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as Fnn  # noqa: E402

W = G.W
ROOT = G.ROOT
E1_JOBS = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e1', 'jobs')
UNIT = G.LO_UNIT
LMIN, LMAX = G.L_MIN, G.L_MAX
NL = LMAX - LMIN + 1                                         # 111
UNK = G.UNK
TRAIN_PER_STRATUM = 70
GEN_SEEDS = (200, 201, 202)
RESERVED_SEEDS = frozenset({1, 2, 7, 101, 102})              # VAL / fresh (1, 2), E1 review (7), E1c fit (101, 102)
EVAL_SEEDS = (1, 2)
EVAL_TIERS = ('cam_nf', 'cam', 'clean')
JUDGED_TIER = 'cam_nf'
TRAIN_SEEDS = (342101, 342202)
HOLDOUT_MOD, HOLDOUT_REM = 10, 9
S_SCALE = 4.0
KMAX = 5
P_CLEAN = 0.10
LOGIT_LO, LOGIT_STEP, NBIN = -40.0, 0.02, 4000              # bins [LO + j STEP, LO + (j + 1) STEP), clipped at both ends
VOFF = 4096; VB = 2 * VOFF                                   # integer evidence-value histograms (linear K0 threshold), clipped
MIN_UPDATES = 2000                                           # D8-44: fewer updates -> INCOMPLETE
CACHE_FORMAT = 1
RUN_FORMAT = 1


def file_sha256(path):
    return G.file_sha256(path)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _rt(x):
    return json.loads(json.dumps(x))


# ================================================================ evidence function, stalks, sums, features, read-outs
class Evidence:
    """the pluggable evidence function: an integer table over the clamped V3 alphabet [-40, 70] (units of 0.05 nats) and the theta of
    its linear read-out (occupied iff sum e > theta)"""

    def __init__(self, table, theta, l0q=0, desc=None):
        self.table = np.asarray(table, np.int64)
        assert self.table.shape == (NL,), self.table.shape
        self.theta = int(theta); self.l0q = int(l0q); self.desc = dict(desc or {})

    @classmethod
    def from_spec(cls, spec, allow_incomplete=False):
        if spec == 'L':
            return cls(np.arange(LMIN, LMAX + 1, dtype=np.int64), 0, 0, dict(kind='L', spec='L', linear='B4: sum L > 0'))
        if spec.startswith('c:'):
            path, half, tier = spec[2:].rsplit(':', 2)                 # (Windows paths contain ':')
            fit = _jload(path)
            assert allow_incomplete or fit.get('complete') is True, f'{path}: E1c fit is not complete (refused; --smoke only)'
            tb = fit['halves'][half]['tiers'][tier]
            assert int(tb['tq']) - int(tb['l0q']) == int(tb['theta']), 'E1c table: theta != tq - l0q'
            return cls(tb['cq'], tb['theta'], tb['l0q'], dict(kind='c', spec=spec, path=os.path.abspath(path), sha256=file_sha256(path), half=half,
                                                               tier=tier, fit_complete=fit.get('complete'), l0=tb.get('l0'),
                                                               linear=f'B6_{half}: sum cq > {int(tb["theta"])}'))
        raise ValueError(f'evidence spec {spec!r}: expected L or c:<fit.json>:<half>:<tier>')

    def recipe(self):
        return dict(table=self.table.tolist(), theta=self.theta, l0q=self.l0q, desc=self.desc)

    @classmethod
    def from_recipe(cls, r):
        return cls(r['table'], r['theta'], r.get('l0q', 0), r.get('desc'))

    def tag(self):
        return 'B4' if self.desc.get('kind') == 'L' else f"B6_{self.desc.get('half')}"


def e1c_rule(summary, fit_sha256, fit_ref_path):
    # D8-44: the evidence of each training seed from the FINAL G4-E1c summary. E1c LEVER or AMBIGUOUS -> B6 tables (half A for 342101,
    # half B for 342202); NOT_ENOUGH or INCOMPLETE -> the raw L (B4). Also checks that the fit file is the one the E1c verdict used.
    # Returns (dict seed -> evidence spec, E1c verdict string, list of problems).
    v = (summary.get('verdict') or {}).get('verdict') if isinstance(summary.get('verdict'), dict) else summary.get('verdict')
    v = str(v)
    probs = []
    if summary.get('fit_sha256') != [fit_sha256]:
        probs.append(f"E1c job files used fit {summary.get('fit_sha256')} != {fit_sha256}")
    if summary.get('fit_sha256_file') != fit_sha256:
        probs.append(f"E1c summary fit file sha {summary.get('fit_sha256_file')} != {fit_sha256}")
    if v.startswith(('LEVER', 'AMBIGUOUS')):
        ev = {'342101': f'c:{fit_ref_path}:A:cam_nf', '342202': f'c:{fit_ref_path}:B:cam_nf'}
    else:
        ev = {'342101': 'L', '342202': 'L'}
    return ev, v, probs


def e1_rows_match(k0_rows, r8_rows, e1_v3):
    """the export's E1 reproduction test: K0 rows and R8 sum rows equal the G4-E1 job file's V3 rows value for value (JSON round trip)"""
    return bool(_rt(k0_rows) == e1_v3['K0'] and _rt(r8_rows) == e1_v3['R8']['sum'])


def stalk(L, table):
    """one agent's stalk: e = table[L] on the voxels it knows, 0 elsewhere (int32), and its known mask. No prior inside."""
    L = np.asarray(L); k = ~G.is_unk(L)
    e = np.zeros(L.shape, np.int32)
    e[k] = np.asarray(table, np.int64)[L[k].astype(np.int64) - LMIN]
    return e, k


def direct_sums(maps, members, table):
    """S = sum_a e_a and k = sum_a 1[a knows] over the members (direct, no relay)"""
    shape = np.asarray(maps[members[0]]).shape
    S = np.zeros(shape, np.int32); kc = np.zeros(shape, np.int32)
    for a in members:
        e, k = stalk(maps[a], table); S += e; kc += k
    return S, kc


def relay_sums(maps, F, table, rstats=None):
    """the two relayed sums of every component (E1 relay, provenance asserted), each belief asserted equal to the direct sums.
    Returns ([(S, k) per component], sum messages, count messages)"""
    own = [stalk(L, table) for L in maps]
    msgs, bel = G.relay(own, F, G.comb_sum, rstats)
    ownn = [(k.astype(np.int32), k) for _, k in own]
    msgn, beln = G.relay(ownn, F, G.comb_sum, rstats)
    out = []
    for cm in F['comps']:
        dS, dk = direct_sums(maps, cm, table)
        for a in cm:
            assert np.array_equal(bel[a][0], dS) and np.array_equal(bel[a][1], dk > 0), 'relayed sum != component sum'
            assert np.array_equal(beln[a][0], dk), 'relayed count != component count'
            if rstats is not None:
                rstats.belief_checks += 1
        out.append((dS, dk))
    return out, msgs, msgn


def zplane(Z):
    return ((np.arange(Z) + 0.5) / Z * 2.0 - 1.0).astype(np.float32)


def features_np(S, kc):
    """[3, X, Y, Z] float32: S x 0.05 / 4 (nats / 4), log(1 + k), z in [-1, 1]. No l0, no x / y."""
    S = np.asarray(S); f = np.empty((3,) + S.shape, np.float32)
    f[0] = S.astype(np.float32) * np.float32(UNIT / S_SCALE)
    f[1] = np.log1p(np.asarray(kc, np.float32))
    f[2] = zplane(S.shape[2])[None, None, :]
    return f


def features_t(S, kc):
    """torch twin of features_np (S, kc: [..., X, Y, Z] integer tensors) -> [..., 3, X, Y, Z] float32; used by train AND eval"""
    Z = S.shape[-1]
    z = torch.as_tensor(zplane(Z), device=S.device).expand(S.shape)
    return torch.stack((S.float() * (UNIT / S_SCALE), torch.log1p(kc.float()), z), dim=-4)


def logit_bin(logit):
    return np.clip(np.floor((np.asarray(logit, np.float64) - LOGIT_LO) / LOGIT_STEP), 0, NBIN - 1).astype(np.int64)


def bin_edge(j):
    return LOGIT_LO + j * LOGIT_STEP


def net_readout(logit, kc, j):
    """known iff k > 0; occupied iff logit bin >= j (i.e. logit >= t = LO + j STEP, exactly as selected)"""
    return np.where(np.asarray(kc) > 0, (logit_bin(logit) >= j).astype(np.int8), np.int8(-1)).astype(np.int8)


def linear_readout(S, kc, theta):
    return np.where(np.asarray(kc) > 0, (np.asarray(S) > theta).astype(np.int8), np.int8(-1)).astype(np.int8)


# ================================================================ network
def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.Conv3d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet3(nn.Module):
    """3-level 3D U-Net, channels (w, 2w, 4w), no normalisation; input [B, 3, X, Y, Z] (X, Y, Z divisible by 4) -> logit [B, X, Y, Z].
    skip=True adds a learnable linear term alpha x S (nats) + beta, initialised at alpha = 1, beta = bias0 = -theta x 0.05 (the
    evidence's linear read-out), and the U-Net head starts near 0: training starts AT the linear read-out B4 / B6 instead of at zero
    (the input is still only S, k, z; the prior is still the network's own, used once)."""

    def __init__(self, cin=3, width=16, skip=False, bias0=0.0):
        super().__init__()
        self.skip = bool(skip)
        if self.skip:
            self.alpha = nn.Parameter(torch.tensor(1.0)); self.beta = nn.Parameter(torch.tensor(float(bias0)))
        c1, c2, c3 = width, 2 * width, 4 * width
        self.e1 = _block(cin, c1); self.e2 = _block(c1, c2); self.b = _block(c2, c3)
        self.u2 = nn.ConvTranspose3d(c3, c2, 2, stride=2); self.d2 = _block(2 * c2, c2)
        self.u1 = nn.ConvTranspose3d(c2, c1, 2, stride=2); self.d1 = _block(2 * c1, c1)
        self.head = nn.Conv3d(c1, 1, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu'); nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=1e-3); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(Fnn.max_pool3d(x1, 2))
        xb = self.b(Fnn.max_pool3d(x2, 2))
        y2 = self.d2(torch.cat((self.u2(xb), x2), 1))
        y1 = self.d1(torch.cat((self.u1(y2), x1), 1))
        out = self.head(y1)[:, 0]
        if self.skip:
            out = out + self.alpha * (x[:, 0] * S_SCALE) + self.beta
        return out


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def weights_sha(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def net_logits(model, S_list, k_list, device, chunk=2):
    """logits (float32 numpy) of the network on a list of (S, k) integer maps"""
    out = []
    for i in range(0, len(S_list), chunk):
        S = torch.from_numpy(np.stack(S_list[i:i + chunk]).astype(np.int32)).to(device)
        k = torch.from_numpy(np.stack(k_list[i:i + chunk]).astype(np.int32)).to(device)
        out += list(model(features_t(S, k)).float().cpu().numpy())
    return out


# ================================================================ house context and V3 maps (= G4-E1 run_job, V3 only; = E1c)
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
    rays = W.fibonacci_sphere(W.BEAMS)
    ds = 0.5 * float(cell.min())
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    return dict(fx=fx, occ=occ, cell=cell, private=private, traj=traj, positions=fx['positions_m'], n=n, shape=occ.shape, ds=ds, H=H, F=F,
                T8=T8, rays=rays, traces=traces, rooms=int(rooms), index=int(index), world_id=fx['world_id'], edges=fx['edges'])


def tier_maps(ctx, cname, seed, k0cache):
    """V3 agent-frame (K0) and public-frame maps of one tier, exactly as G4-E1 run_job (common random numbers, same calls).
    k0cache is keyed by (agent-frame parameters, agent): pass a fresh dict per noise seed."""
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
        mp = G.integrate_agent(ctx['shape'], ctx['cell'], ctx['ds'], [G.to_public(s, frame) for s in scans], ('V3',))['V3'] if has_inter else mk
        L_k0.append(mk); L_pub.append(mp)
    return L_k0, L_pub, has_inter


def edges_array(edges):
    return np.asarray([[int(i), int(j), float(d)] for i, j, d in edges], np.float64).reshape(-1, 3)


def edges_list(arr):
    return [(int(r[0]), int(r[1]), float(r[2])) for r in np.asarray(arr).reshape(-1, 3)]


# ================================================================ stage gen: the TRAIN cache
def train_plan():
    plan = W.scene_plan('TRAIN', TRAIN_PER_STRATUM)
    assert all(s['split'] == 'TRAIN' for s in plan)
    return [(int(s['rooms']), int(s['index'])) for s in plan]


def is_holdout(index):
    return int(index) % HOLDOUT_MOD == HOLDOUT_REM


def house_name(rooms, index):
    return f'TRAIN_r{rooms}_i{index}'


def gen_job(job):
    rooms, index, seeds, opts = job
    t0 = time.time(); name = house_name(rooms, index)
    rec = dict(stage='gen', house=name, rooms=rooms, index=index, seeds=list(seeds), holdout=is_holdout(index), script_sha256=opts['script_sha256'])
    try:
        try:
            ctx = house_ctx('TRAIN', rooms, index)
        except ValueError as e:                                       # only agent placement may skip a house (as E1)
            rec['skipped'] = repr(e)[:200]; return rec
        priv = ctx['private']; occ1 = np.asarray(ctx['occ']) == 1
        for a in range(ctx['n']):
            kn = priv[a] >= 0
            assert np.array_equal(priv[a][kn] == 1, occ1[kn]), 'private map disagrees with occ'
        arrays = dict(occ=np.asarray(ctx['occ'], np.int8), slab=np.asarray(ctx['fx']['slab'], bool), obj=np.asarray(ctx['fx']['obj'], bool),
                      cell=ctx['cell'], private=np.asarray(priv, np.int8), comp_of=np.array([ctx['F']['comp_of'][a] for a in range(ctx['n'])], np.int64),
                      edges=edges_array(ctx['edges']))
        Lc, _, _ = tier_maps(ctx, 'clean', 0, {})
        for a in range(ctx['n']):                                     # E1 criterion (a): the clean V3 ternary IS the private map
            assert np.array_equal(G.tern(Lc[a]), priv[a]), 'clean V3 map != private map'
        arrays['L_clean'] = np.stack(Lc).astype(np.int8)
        known = {'clean': float(np.mean([np.mean(L != UNK) for L in Lc]))}
        for s in seeds:
            Lk, Lp, has_inter = tier_maps(ctx, 'cam_nf', s, {})
            assert not has_inter and all(x is y for x, y in zip(Lk, Lp))
            arrays[f'L_cam_nf_s{s}'] = np.stack(Lk).astype(np.int8)
            known[f'cam_nf_s{s}'] = float(np.mean([np.mean(L != UNK) for L in Lk]))
        os.makedirs(opts['cache'], exist_ok=True)
        path = os.path.join(opts['cache'], name + '.npz'); tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
        rec.update(world_id=ctx['world_id'], agents=ctx['n'], components=ctx['F']['comps'], private_sha256=W.array_hash(priv),
                   file=os.path.basename(path), file_sha256=file_sha256(path), file_bytes=os.path.getsize(path), known_frac=known,
                   seconds=time.time() - t0, peak_mb=G._peak_mb())
    except Exception as e:                                            # a failed job is recorded and counted missing
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _gen_and_save(job):
    rec = gen_job(job)
    opts = job[-1]
    d = os.path.join(opts['cache'], 'records'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, rec['house'] + '.json')
    with open(p + '.tmp', 'w') as fh:
        json.dump(rec, fh)
    os.replace(p + '.tmp', p)
    return rec


def write_manifest(cache, opts):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(cache, 'records', '*.json')))]
    ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
    man = dict(format=CACHE_FORMAT, script_sha256=opts['script_sha256'], noise_panels_sha256=file_sha256(G.__file__),
               data_module_sha256=file_sha256(W.__file__), houseexpo_index_sha256=file_sha256(os.environ['HOUSEEXPO_INDEX']),
               seeds=opts['seeds'], holdout_rule=f'TRAIN index % {HOLDOUT_MOD} == {HOLDOUT_REM}', p_clean_at_train=P_CLEAN,
               houses=[dict(house=r['house'], rooms=r['rooms'], index=r['index'], holdout=r['holdout'], agents=r['agents'], components=r['components'],
                            file=r['file'], file_sha256=r['file_sha256'], file_bytes=r['file_bytes'], world_id=r['world_id'], seeds=r['seeds'])
                       for r in sorted(ok, key=lambda r: (r['rooms'], r['index']))],
               skipped=[r['house'] for r in recs if r.get('skipped')], failed=[r['house'] for r in recs if r.get('failed')])
    man['n_train'] = sum(1 for h in man['houses'] if not h['holdout']); man['n_holdout'] = sum(1 for h in man['houses'] if h['holdout'])
    man['total_bytes'] = int(sum(h['file_bytes'] for h in man['houses']))
    assert len({tuple(r['seeds']) for r in ok}) <= 1 and len({r['script_sha256'] for r in ok}) <= 1, 'mixed records (seeds / script) in the cache'
    path = os.path.join(cache, 'manifest.json')
    with open(path, 'w') as fh:
        json.dump(man, fh, indent=1)
    return man, path


# ================================================================ stage train
class HouseData:
    """one cached TRAIN house: maps [D, n, X, Y, Z] int8 (draw 0 = clean, 1.. = cam_nf seeds), pk = bitmask of the private known maps,
    occ (clean truth), components; on the training device as tensors (t_*) and as numpy (for the threshold selection)"""

    def __init__(self, path, entry, seeds):
        with np.load(path) as z:
            self.name = entry['house']; self.private = z['private']; self.n = self.private.shape[0]
            self.draws = ['clean'] + [f'cam_nf_s{s}' for s in seeds]
            self.maps = np.stack([z['L_' + d] for d in self.draws])
            self.occ = z['occ'] == 1; self.slab = z['slab']; self.obj = z['obj']; self.cell = z['cell']
            comp_of = z['comp_of']
        self.comps = [list(map(int, np.nonzero(comp_of == c)[0])) for c in sorted(set(comp_of.tolist()))]
        assert self.n <= 8
        pk = np.zeros(self.occ.shape, np.uint8)
        for a in range(self.n):
            pk |= ((self.private[a] >= 0).astype(np.uint8) << a)
        self.pk = pk

    def to(self, device):
        self.t_maps = torch.from_numpy(self.maps).to(device); self.t_pk = torch.from_numpy(self.pk).to(device)
        self.t_occ = torch.from_numpy(self.occ).to(device)
        return self


def load_cache(cache, manifest, which, device=None, verify=True):
    out = []
    seeds = manifest['seeds']
    for e in manifest['houses']:
        if (which == 'train' and e['holdout']) or (which == 'holdout' and not e['holdout']):
            continue
        p = os.path.join(cache, e['file'])
        if verify:
            assert file_sha256(p) == e['file_sha256'], f'{p}: sha256 differs from the manifest'
        h = HouseData(p, e, seeds)
        out.append(h.to(device) if device is not None else h)
    return out


def sample_spec(rng, houses, p_clean=P_CLEAN):
    """(house, draw, component, subset, aug): tier clean w.p. p_clean else a cam_nf seed uniformly; component uniform; |s| uniform 1..n"""
    h = int(rng.integers(len(houses))); H = houses[h]
    d = 0 if rng.random() < p_clean else 1 + int(rng.integers(len(H.draws) - 1))
    cm = H.comps[int(rng.integers(len(H.comps)))]
    m = 1 + int(rng.integers(len(cm)))
    sub = tuple(sorted(int(a) for a in rng.choice(cm, m, replace=False)))
    return h, d, sub, int(rng.integers(8))


def sample_tensors(H, d, sub, table_t, loss_mask='scope_known'):
    """on H's device: inputs (S, k) of the subset's sum, label (clean occ), loss mask (scope [& k > 0])"""
    L = H.t_maps[d][list(sub)]
    kn = L != UNK
    e = torch.where(kn, table_t[(L.long() - LMIN).clamp(0, NL - 1)], torch.zeros((), dtype=table_t.dtype, device=L.device))
    S = e.sum(0, dtype=torch.int32); kc = kn.sum(0, dtype=torch.int32)
    bits = sum(1 << a for a in sub)
    scope = (H.t_pk.int() & bits) != 0
    mask = (scope & (kc > 0)) if loss_mask == 'scope_known' else scope
    return S, kc, H.t_occ, mask


def augment(x, y, m, aug):
    """dihedral xy symmetry: bit 0 transpose x <-> y, bit 1 flip x, bit 2 flip y (x: [C, X, Y, Z]; y, m: [X, Y, Z])"""
    if aug & 1:
        x = x.transpose(1, 2); y = y.transpose(0, 1); m = m.transpose(0, 1)
    if aug & 2:
        x = x.flip(1); y = y.flip(0); m = m.flip(0)
    if aug & 4:
        x = x.flip(2); y = y.flip(1); m = m.flip(1)
    return x, y, m


def make_batch(specs, houses, table_t, loss_mask):
    xs, ys, ms = [], [], []
    for h, d, sub, aug in specs:
        S, kc, y, m = sample_tensors(houses[h], d, sub, table_t, loss_mask)
        x, y, m = augment(features_t(S, kc), y, m, aug)
        xs.append(x); ys.append(y); ms.append(m)
    return torch.stack(xs).contiguous(), torch.stack(ys).float(), torch.stack(ms).float()


def masked_bce(logits, y, m):
    l = Fnn.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return (l * m).sum() / m.sum().clamp(min=1.0)


def lr_at(u, updates, lr, warm):
    return lr * min(1.0, (u + 1) / max(warm, 1)) * 0.5 * (1.0 + math.cos(math.pi * min(u, updates) / max(updates, 1)))


@torch.no_grad()
def holdout_bce(model, specs, houses, table_t, loss_mask, bs):
    model.eval(); tot = 0.; cnt = 0.
    for i in range(0, len(specs), bs):
        x, y, m = make_batch([(h, d, s, 0) for h, d, s, _ in specs[i:i + bs]], houses, table_t, loss_mask)
        l = Fnn.binary_cross_entropy_with_logits(model(x), y, reduction='none')
        tot += float((l * m).sum()); cnt += float(m.sum())
    model.train()
    return tot / max(cnt, 1.)


def train(args):
    t_start = time.time()
    assert args.seed in TRAIN_SEEDS or args.smoke, f'training seed {args.seed} not in {TRAIN_SEEDS}'
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(max(1, args.threads))
    man_path = os.path.join(args.cache, 'manifest.json'); man_sha = file_sha256(man_path)
    if args.manifest_sha:
        assert man_sha == args.manifest_sha, f'cache manifest sha {man_sha} != pinned {args.manifest_sha}'
    man = _jload(man_path)
    assert man['format'] == CACHE_FORMAT and not (set(man['seeds']) & RESERVED_SEEDS), 'cache format / reserved noise seeds'
    ev = Evidence.from_spec(args.evidence, allow_incomplete=args.smoke)
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    tr = load_cache(args.cache, man, 'train', dev); ho = load_cache(args.cache, man, 'holdout', dev)
    assert tr and ho, 'empty train or holdout set'
    load_s = time.time() - t_start
    table_t = torch.as_tensor(ev.table, dtype=torch.int32, device=dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 32))
    bias0 = -ev.theta * UNIT if args.skip else 0.0
    model = UNet3(3, args.width, skip=bool(args.skip), bias0=bias0).to(dev)
    assert n_params(model) <= 1_000_000
    init_sha = weights_sha(model)
    decay = [q for q in model.parameters() if q.ndim > 1]; nodecay = [q for q in model.parameters() if q.ndim <= 1]   # no decay on biases / alpha / beta
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=nodecay, weight_decay=0.0)], lr=args.lr)
    rng = np.random.default_rng([args.seed, 3])                                   # data order (per training seed)
    hrng = np.random.default_rng([20261027, 9])                                   # holdout curve samples (same for both seeds)
    hspecs = [sample_spec(hrng, ho) for _ in range(args.holdout_samples)]
    recipe = dict(width=args.width, cin=3, skip=bool(args.skip), bias0=bias0, s_scale=S_SCALE, unit=UNIT, inputs='S x 0.05 / 4, log(1 + k), z', loss_mask=args.loss_mask,
                  evidence=ev.recipe(), p_clean=P_CLEAN, batch=args.batch, lr=args.lr, wd=args.wd, warmup=args.warmup, aug='dihedral xy')
    run = dict(format=RUN_FORMAT, stage='train', seed=args.seed, updates=args.updates, max_updates=args.max_updates, device=str(dev), recipe=recipe,
               parameters=n_params(model), initial_weights_sha256=init_sha, script_sha256=file_sha256(os.path.abspath(__file__)),
               noise_panels_sha256=file_sha256(G.__file__), cache=os.path.abspath(args.cache), manifest_sha256=man_sha,
               train_houses=len(tr), holdout_houses=len(ho), deadline_epoch=args.deadline_epoch, load_seconds=load_s, smoke=bool(args.smoke),
               torch=torch.__version__, cudnn=(torch.backends.cudnn.version() if dev.type == 'cuda' else None),
               gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None), cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark), pythonpath=os.environ.get('PYTHONPATH'))
    with open(os.path.join(args.out, 'run.json'), 'w') as fh:
        json.dump(run, fh, indent=1)
    print(f'E3-R train seed {args.seed}: {len(tr)} train / {len(ho)} holdout houses, {n_params(model)} parameters, evidence {ev.desc.get("spec")}, '
          f'load {load_s:.0f}s', flush=True)
    stop_at = min(args.updates, args.max_updates) if args.max_updates else args.updates
    history = []; run_loss = 0.; run_n = 0; marks = {}; truncated = False; u = 0
    t_train = time.time()
    model.train()
    for u in range(stop_at):
        if args.deadline_epoch and time.time() > args.deadline_epoch - args.reserve:
            truncated = True; break
        for g in opt.param_groups:
            g['lr'] = lr_at(u, args.updates, args.lr, args.warmup)
        x, y, m = make_batch([sample_spec(rng, tr) for _ in range(args.batch)], tr, table_t, args.loss_mask)
        loss = masked_bce(model(x), y, m)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        run_loss += loss.detach().item(); run_n += 1
        done = u + 1
        log_now = done % args.log_every == 0 or done == stop_at
        if log_now:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            marks[done] = time.time() - t_train
            print(f'upd{done} loss {run_loss / run_n:.5f} grad {gn:.3f} lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t_start:.0f}s', flush=True)
        if done % args.val_every == 0 or done == stop_at:
            hb = holdout_bce(model, hspecs, ho, table_t, args.loss_mask, args.batch)
            history.append(dict(updates=done, train_bce=run_loss / max(run_n, 1), holdout_bce=hb, seconds=time.time() - t_start))
            print(f'  holdout bce {hb:.5f}', flush=True)
            with open(os.path.join(args.out, 'history.json'), 'w') as fh:
                json.dump(history, fh, indent=1)
        if log_now:
            run_loss = 0.; run_n = 0
    else:
        u = stop_at
    done = u if truncated else stop_at
    ks = sorted(marks)
    rate = None
    if len(ks) >= 2:
        k0 = ks[0] if len(ks) == 2 else ks[1]                      # skip the first window (cudnn autotune / warm-up)
        rate = (marks[ks[-1]] - marks[k0]) / max(ks[-1] - k0, 1)
    ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, recipe=recipe, seed=args.seed, updates_done=done,
              updates_planned=args.updates, truncated=truncated, run=run)
    ck_path = os.path.join(args.out, 'checkpoints', 'final.pt')
    torch.save(ck, ck_path)
    fin = dict(updates_done=done, updates_planned=args.updates, max_updates=args.max_updates, truncated=truncated,
               complete=(done == args.updates and not truncated), rate_s_per_update=rate, first_window_s=(marks[ks[0]] if ks else None),
               first_window_updates=(ks[0] if ks else None), load_seconds=load_s, train_seconds=time.time() - t_train, weights_sha256=weights_sha(model),
               checkpoint=ck_path, checkpoint_sha256=file_sha256(ck_path), history_last=(history[-1] if history else None))
    if args.select:
        ts = time.time()
        sel = select_threshold(model, ev, ho, dev, chunk=args.batch)
        sel['seconds'] = time.time() - ts
        with open(os.path.join(args.out, 'threshold.json'), 'w') as fh:
            json.dump(sel, fh, indent=1)
        fin['threshold'] = dict(t=sel['t'], j=sel['j'], f1=sel['f1'], free=sel['free'], linear_f1=sel['linear']['f1'], linear_free=sel['linear']['free'],
                                k0_f1=sel['k0']['f1'], k0_linear_f1=sel['k0']['linear_f1'], k0_t_sep=sel['k0']['t_sep'], k0_f1_sep=sel['k0']['f1_sep'],
                                k0_linear_theta_sep=sel['k0']['linear_theta_sep'], k0_linear_f1_sep=sel['k0']['linear_f1_sep'], holdout_houses=sel['n_houses'],
                                device=sel['device'], tf32=sel['tf32'], seconds=sel['seconds'])
        print(f"threshold t {sel['t']:.2f} (bin {sel['j']}): holdout R8 pooled furn F1 {sel['f1']:.4f} free {sel['free']:.4f} | linear "
              f"{ev.tag()} {sel['linear']['f1']:.4f} free {sel['linear']['free']:.4f} | K0 {sel['k0']['f1']:.4f} (own t {sel['k0']['t_sep']:.2f}: "
              f"{sel['k0']['f1_sep']:.4f}; linear {sel['k0']['linear_f1']:.4f}, own theta {sel['k0']['linear_theta_sep']}: {sel['k0']['linear_f1_sep']:.4f}) | "
              f"{sel['n_houses']} holdout houses", flush=True)
    fin['seconds'] = time.time() - t_start
    with open(os.path.join(args.out, 'final.json'), 'w') as fh:
        json.dump(fin, fh, indent=1)
    print(f'E3-R train seed {args.seed} done: {done} updates ({"truncated" if truncated else "complete" if fin["complete"] else "partial"}), '
          f'rate {rate} s/update, {fin["seconds"]:.0f}s', flush=True)
    return fin


# ================================================================ threshold selection (held-out TRAIN houses)
def _cat_masks(kc, T, H):
    kn = kc > 0
    return kn & T.known & H.region & T.tr, kn & T.known & H.region & ~T.tr, kn & T.free


@contextlib.contextmanager
def no_tf32():
    """review S2: the threshold selection runs its convolutions in full fp32 (TF32 off) and restores the flags afterwards"""
    a, b = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = a; torch.backends.cuda.matmul.allow_tf32 = b


def _vbin(v):
    return np.clip(np.asarray(v, np.int64), -VOFF, VOFF - 1) + VOFF


def select_threshold(model, ev, houses, device, chunk=2):
    """D8-44 thresholds, all on the held-out TRAIN houses' cam_nf draws, rule = pooled furniture F1 maximal s.t. pooled free recall >=
    that of the evidence's linear read-out at its default theta, ties -> the smallest threshold:
      t      network, R8 component sums (component-expanded): 'logit >= t'                       (the R8 read-out)
      t_K0   network, single agents (K0): 'logit >= t_K0'                                         (NET K0_sep)
      th_K0  linear read-out, single agents (K0): 'e > th_K0' (integer)                           (LIN K0_sep, review M2)
    Convolutions in fp32 with TF32 off (review S2); the device is recorded (GPU and CPU logits agree to ~1e-5, not bitwise)."""
    model.eval()
    A = np.zeros(NBIN, np.int64); B = np.zeros(NBIN, np.int64); C = np.zeros(NBIN, np.int64); tr_tot = fr_tot = 0
    lin = np.zeros(3, np.int64)                                        # linear read-out: tp, fp, free hits (pooled)
    k0h = np.zeros((3, NBIN), np.int64); k0tr = k0fr = 0; k0lin = np.zeros(3, np.int64)
    k0v = np.zeros((3, VB), np.int64)                                  # linear K0: value histograms of e (furniture target, other region, free)
    with no_tf32():
        for Hd in houses:
            Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
            R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]
            K0T = [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
            for d in range(1, len(Hd.draws)):                          # cam_nf draws only
                maps = Hd.maps[d]
                sums = [direct_sums(maps, cm, ev.table) for cm in Hd.comps]
                lg = net_logits(model, [s for s, _ in sums], [k for _, k in sums], device, chunk)
                for ci, cm in enumerate(Hd.comps):
                    S, kc = sums[ci]; T = R8T[ci]; w = len(cm); b = logit_bin(lg[ci])
                    a_, b_, c_ = _cat_masks(kc, T, Hh)
                    A += np.bincount(b[a_], minlength=NBIN) * w; B += np.bincount(b[b_], minlength=NBIN) * w; C += np.bincount(b[c_], minlength=NBIN) * w
                    tr_tot += int(T.tr.sum()) * w; fr_tot += int(T.free.sum()) * w
                    occ_lin = S > ev.theta
                    lin += np.array([int((a_ & occ_lin).sum()), int((b_ & occ_lin).sum()), int((c_ & ~occ_lin).sum())]) * w
                own = [stalk(maps[a], ev.table) for a in range(Hd.n)]
                lk = net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], device, chunk)
                for a in range(Hd.n):
                    e, k = own[a]; T = K0T[a]; b = logit_bin(lk[a]); a_, b_, c_ = _cat_masks(k.astype(np.int32), T, Hh)
                    k0h[0] += np.bincount(b[a_], minlength=NBIN); k0h[1] += np.bincount(b[b_], minlength=NBIN); k0h[2] += np.bincount(b[c_], minlength=NBIN)
                    vb = _vbin(e)
                    k0v[0] += np.bincount(vb[a_], minlength=VB); k0v[1] += np.bincount(vb[b_], minlength=VB); k0v[2] += np.bincount(vb[c_], minlength=VB)
                    k0tr += int(T.tr.sum()); k0fr += int(T.free.sum())
                    occ_lin = e > ev.theta
                    k0lin += np.array([int((a_ & occ_lin).sum()), int((b_ & occ_lin).sum()), int((c_ & ~occ_lin).sum())])
    j, f1, fr, tp, fp, fn, lin_f1, lin_fr = _select(A, B, C, tr_tot, fr_tot, lin)
    jk, f1k, frk, _, _, _, lin_f1k, lin_frk = _select(k0h[0], k0h[1], k0h[2], k0tr, k0fr, k0lin)
    jl, f1l, frl, _, _, _, _, _ = _select(k0v[0], k0v[1], k0v[2], k0tr, k0fr, k0lin)   # 'value bin >= jl' <=> e > jl - VOFF - 1
    k0tp = int(k0h[0][j:].sum()); k0fp = int(k0h[1][j:].sum()); k0fh = int(k0h[2][:j].sum())
    model.train()
    dev = torch.device(device)
    return dict(t=bin_edge(j), j=j, f1=float(f1[j]), free=float(fr[j]), tp=int(tp[j]), fp=int(fp[j]), fn=int(fn[j]), tr_tot=int(tr_tot), free_tot=int(fr_tot),
                linear=dict(rule=ev.desc.get('linear'), theta=ev.theta, f1=float(lin_f1), free=float(lin_fr)),
                k0=dict(f1=2 * k0tp / max(2 * k0tp + k0fp + (k0tr - k0tp), 1), free=k0fh / max(k0fr, 1), linear_f1=float(lin_f1k), linear_free=float(lin_frk),
                        j_sep=jk, t_sep=bin_edge(jk), f1_sep=float(f1k[jk]), free_sep=float(frk[jk]),
                        linear_theta_sep=int(jl - VOFF - 1), linear_f1_sep=float(f1l[jl]), linear_free_sep=float(frl[jl]),
                        rule_sep='K0 thresholds (network t_sep, linear theta_sep): the same rule on single-agent inputs (D8-44 K0 guard)'),
                houses=[h.name for h in houses], n_houses=len(houses), draws=houses[0].draws[1:] if houses else [],
                scope='R8 component sums, cam_nf draws, component-expanded; K0 single agents',
                device=str(dev), tf32=False, gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None),
                curve=dict(j=list(range(0, NBIN, 25)), f1=f1[::25].tolist(), free=fr[::25].tolist()))


def _select(A, B, C, tr_tot, fr_tot, lin):
    """the bin j maximising pooled furniture F1 of 'bin >= j' s.t. pooled free recall >= the linear read-out's (ties -> smallest j)"""
    sa = np.cumsum(A[::-1])[::-1]; sb = np.cumsum(B[::-1])[::-1]; pc = np.r_[0, np.cumsum(C)[:-1]]   # predicate: bin >= j
    tp = sa; fp = sb; fn = tr_tot - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1); fr = pc / max(fr_tot, 1)
    lin_f1 = 2 * lin[0] / max(2 * lin[0] + lin[1] + (tr_tot - lin[0]), 1); lin_fr = lin[2] / max(fr_tot, 1)
    ok = fr + 1e-15 >= lin_fr
    assert ok.any(), 'no threshold satisfies the free-recall constraint'
    j = int(np.flatnonzero(ok)[np.argmax(f1[ok])])
    return j, f1, fr, tp, fp, fn, lin_f1, lin_fr


def load_ckpt(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    r = ck['recipe']; model = UNet3(r['cin'], r['width'], skip=r.get('skip', False), bias0=r.get('bias0', 0.0)).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return ck, model


def select_cmd(args):
    dev = torch.device(args.device)
    if dev.type == 'cpu':
        torch.set_num_threads(max(1, args.threads))
    run = _jload(os.path.join(args.run, 'run.json'))
    ck, model = load_ckpt(os.path.join(args.run, 'checkpoints', 'final.pt'), dev)
    ev = Evidence.from_recipe(ck['recipe']['evidence'])
    man = _jload(os.path.join(run['cache'] if not args.cache else args.cache, 'manifest.json'))
    ho = load_cache(run['cache'] if not args.cache else args.cache, man, 'holdout')
    sel = select_threshold(model, ev, ho, dev)
    if os.path.exists(os.path.join(args.run, 'threshold.json')) and not args.overwrite:
        raise SystemExit(f'{args.run}/threshold.json exists (use --overwrite to replace it)')
    with open(os.path.join(args.run, 'threshold.json'), 'w') as fh:
        json.dump(sel, fh, indent=1)
    print(json.dumps({k: sel[k] for k in ('t', 'j', 'f1', 'free', 'linear', 'k0')}), flush=True)
    return sel


# ================================================================ stage export (VAL / fresh inputs, E1 reproduction)
def export_job(job):
    panel, rooms, index, seed, opts = job
    t0 = time.time(); split = opts['split']
    rec = dict(stage='export', panel=panel, rooms=rooms, index=index, seed=seed, split=split, script_sha256=opts['script_sha256'])
    try:
        try:
            ctx = house_ctx(split, rooms, index)
        except ValueError as e:
            rec['skipped'] = repr(e)[:200]; return rec
        e1p = os.path.join(opts['e1_jobs'], f'{panel}_r{rooms}_i{index}_s{seed}.json')
        e1 = _jload(e1p) if (split == 'VALIDATION' and os.path.exists(e1p)) else None
        rec['e1_job'] = os.path.basename(e1p) if e1 else None
        H = ctx['H']; F = ctx['F']; n = ctx['n']
        K0T = [G.Target(ctx['private'][a], H) for a in range(n)]
        R8T = [G.Target(ctx['T8'][cm[0]], H, mult=(ctx['private'][cm] >= 0).sum(0)) for cm in F['comps']]
        arrays = dict(occ=np.asarray(ctx['occ'], np.int8), slab=np.asarray(ctx['fx']['slab'], bool), obj=np.asarray(ctx['fx']['obj'], bool),
                      cell=ctx['cell'], private=np.asarray(ctx['private'], np.int8), edges=edges_array(ctx['edges']))
        match = {}; k0cache = {}
        for tn in opts['tiers']:
            L_k0, L_pub, has_inter = tier_maps(ctx, tn, seed, k0cache)
            if e1 is not None and tn in e1['conds']:
                ev3 = e1['conds'][tn]['by_int']['V3']
                k0 = [G.score(G.tern(L_k0[a]), K0T[a], strata=True) for a in range(n)]
                _, preds = G.fuse(L_pub, F, 'sum')
                r8 = [[G.score(preds[cm[0]], R8T[ci], strata=True), len(cm)] for ci, cm in enumerate(F['comps'])]
                match[tn] = e1_rows_match(k0, r8, ev3)
                if tn == JUDGED_TIER:
                    rec['B4_rnet_relay_e1'] = ev3['rnet']['B4_int_relay']
            else:
                match[tn] = None
            arrays[f'L_k0_{tn}'] = np.stack(L_k0).astype(np.int8)
            if has_inter:
                arrays[f'L_pub_{tn}'] = np.stack(L_pub).astype(np.int8)
        os.makedirs(opts['cache'], exist_ok=True)
        path = os.path.join(opts['cache'], f'{panel}_r{rooms}_i{index}_s{seed}.npz'); tmp = path + '.tmp.npz'
        np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
        rec.update(e1_match=match, world_id=ctx['world_id'], agents=n, components=F['comps'], file=os.path.basename(path), file_sha256=file_sha256(path),
                   private_sha256=W.array_hash(ctx['private']), seconds=time.time() - t0, peak_mb=G._peak_mb())
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _export_and_save(job):
    rec = export_job(job); opts = job[-1]
    d = os.path.join(opts['cache'], 'records'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json")
    with open(p + '.tmp', 'w') as fh:
        json.dump(rec, fh)
    os.replace(p + '.tmp', p)
    return rec


# ================================================================ stage eval (CPU): network / linear / controls on the exported maps
def k_strata(pred, T, H, kc):
    """furniture tp / fp / fn and free hits / totals by noisy multiplicity k = 0 (unknown: only FN / missed free), 1, 2, 3, 4, >= 5,
    inside the target-known scope (E1's categories; G._strata turns '*furn' into F1 and the rest into recall)"""
    cls = np.clip(kc, 0, KMAX).astype(np.int64)
    pr = (pred == 1) & T.known & H.region
    tpm, fpm, fnm = pr & T.tr, pr & ~T.tr, ~pr & T.tr
    frh = (pred == 0) & T.free

    def by(x):
        return np.bincount(cls[x], minlength=KMAX + 1).astype(np.int64)
    return dict(k_furn=np.stack([by(x) for x in (tpm, fpm, fnm)], 1).tolist(), k_free=np.stack([by(frh), by(T.free)], 1).tolist())


def scored(pred, T, H, kc):
    row = G.score(pred, T, strata=True)
    if row.get('strata') is not None:
        row['strata'].update(k_strata(pred, T, H, kc))
    return row


def cal_hist(logit, kc, T):
    """per k (1..4, >= 5) histograms over logit bins of (truth occupied, truth free) on the scope k > 0 & clean-known"""
    out = np.zeros((2, KMAX, NBIN), np.int64)
    m = (kc > 0) & T.known; y = T.t == 1
    kk = np.clip(kc, 1, KMAX) - 1; b = logit_bin(logit)
    np.add.at(out[0], (kk[m & y], b[m & y]), 1)
    np.add.at(out[1], (kk[m & ~y], b[m & ~y]), 1)
    return out


def _sparse(h):
    nz = np.nonzero(h)
    return dict(shape=list(h.shape), idx=[x.tolist() for x in nz], val=h[nz].tolist())


def _dense(s):
    h = np.zeros(s['shape'], np.int64)
    if s['val']:
        h[tuple(np.asarray(i) for i in s['idx'])] = s['val']
    return h


def eval_job(job):
    rec_e, opts = job
    t0 = time.time()
    rec = dict(stage='eval', panel=rec_e['panel'], rooms=rec_e['rooms'], index=rec_e['index'], seed=rec_e['seed'], split=rec_e.get('split'),
               e1_match=rec_e.get('e1_match'), script_sha256=opts['script_sha256'], ckpts=opts['ckpt_shas'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        path = os.path.join(opts['export'], rec_e['file'])
        assert file_sha256(path) == rec_e['file_sha256'], 'export file changed'
        z = np.load(path)
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in rec_e['components']]
        K0T = [G.Target(private[a], H) for a in range(n)]
        R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
        models = []
        for p in opts['ckpts']:
            ck, model = load_ckpt(p, 'cpu'); thr = _jload(os.path.join(os.path.dirname(os.path.dirname(p)), 'threshold.json'))
            models.append((str(ck['seed']), Evidence.from_recipe(ck['recipe']['evidence']), model, int(thr['j']), thr['k0'].get('j_sep'),
                           thr['k0'].get('linear_theta_sep')))
        controls = [Evidence.from_spec(s, allow_incomplete=opts.get('smoke', False)) for s in opts['b6']]
        rstats = G.RelayStats(); rec['tiers'] = {}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            T = {}
            # B4 = E1 (G.fuse sum) and the known mask every arm must share
            _, preds = G.fuse(L_pub, F, 'sum', rstats)
            sumsL, _, _ = relay_sums(L_pub, F, Evidence.from_spec('L').table, rstats)
            r8_known = [sumsL[ci][1] > 0 for ci in range(len(F['comps']))]
            for ci, cm in enumerate(F['comps']):
                assert np.array_equal(preds[cm[0]] >= 0, r8_known[ci]) and np.array_equal(preds[cm[0]], linear_readout(*sumsL[ci], 0)), 'B4 != linear L read-out'
            T['B4'] = dict(K0=[scored(G.tern(L_k0[a]), K0T[a], H, (L_k0[a] != UNK).astype(np.int32)) for a in range(n)],
                           R8=[[scored(preds[cm[0]], R8T[ci], H, sumsL[ci][1]), len(cm)] for ci, cm in enumerate(F['comps'])])
            cal4 = np.zeros((2, KMAX, NBIN), np.int64)
            for ci in range(len(F['comps'])):
                cal4 += cal_hist(sumsL[ci][0] * UNIT, sumsL[ci][1], R8T[ci])
            T['B4']['cal'] = _sparse(cal4)
            cache_sums = {}

            def sums_for(ev):
                key = hashlib.sha256(ev.table.tobytes()).hexdigest()
                if key not in cache_sums:
                    s, msgs, msgn = relay_sums(L_pub, F, ev.table, rstats)
                    by = None
                    if tn == JUDGED_TIER and opts['bytes']:
                        by = dict(sum=float(sum(G.int_ctx_bytes(*msgs[e]) for e in F['directed'])), count=float(sum(G.int_ctx_bytes(*msgn[e]) for e in F['directed'])))
                    cache_sums[key] = (s, by)
                return cache_sums[key]
            for ev in controls:                                         # B6 controls (linear read-out of each given table)
                s, by = sums_for(ev); calc = np.zeros((2, KMAX, NBIN), np.int64)
                r8 = []
                for ci, cm in enumerate(F['comps']):
                    S, kc = s[ci]; p = linear_readout(S, kc, ev.theta)
                    assert np.array_equal(p >= 0, r8_known[ci]), 'B6 known mask != B4'
                    r8.append([scored(p, R8T[ci], H, kc), len(cm)]); calc += cal_hist((ev.l0q + S) * UNIT, kc, R8T[ci])
                k0 = []
                for a in range(n):
                    e, k = stalk(L_k0[a], ev.table); k0.append(scored(linear_readout(e, k.astype(np.int32), ev.theta), K0T[a], H, k.astype(np.int32)))
                T[ev.tag()] = dict(K0=k0, R8=r8, cal=_sparse(calc), theta=ev.theta, l0q=ev.l0q, spec=ev.desc.get('spec'), bytes=by)
            for sd, ev, model, j, jk, thk in models:                             # the networks (and their own evidence's linear read-out)
                s, by = sums_for(ev)
                lg = net_logits(model, [x[0] for x in s], [x[1] for x in s], 'cpu', opts['chunk'])
                r8, r8l = [], []; calN = np.zeros((2, KMAX, NBIN), np.int64); calL = np.zeros((2, KMAX, NBIN), np.int64)
                for ci, cm in enumerate(F['comps']):
                    S, kc = s[ci]; p = net_readout(lg[ci], kc, j)
                    assert np.array_equal(p >= 0, r8_known[ci]), 'network known mask != B4 (the network must not fill unknown voxels)'
                    r8.append([scored(p, R8T[ci], H, kc), len(cm)]); calN += cal_hist(lg[ci], kc, R8T[ci])
                    r8l.append([scored(linear_readout(S, kc, ev.theta), R8T[ci], H, kc), len(cm)]); calL += cal_hist((ev.l0q + S) * UNIT, kc, R8T[ci])
                own = [stalk(L_k0[a], ev.table) for a in range(n)]
                lk = net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', opts['chunk'])
                k0, k0l, k0s, k0ls = [], [], [], []
                for a in range(n):
                    e, k = own[a]; ki = k.astype(np.int32); p = net_readout(lk[a], ki, j)
                    assert np.array_equal(p >= 0, L_k0[a] != UNK), 'network K0 known mask != own map'
                    k0.append(scored(p, K0T[a], H, ki)); k0l.append(scored(linear_readout(e, ki, ev.theta), K0T[a], H, ki))
                    if jk is not None:
                        k0s.append(scored(net_readout(lk[a], ki, jk), K0T[a], H, ki))
                    if thk is not None:
                        k0ls.append(scored(linear_readout(e, ki, thk), K0T[a], H, ki))
                T[f'NET_{sd}'] = dict(K0=k0, R8=r8, cal=_sparse(calN), j=j, t=bin_edge(j), evidence=ev.desc.get('spec'), bytes=by)
                if jk is not None:
                    T[f'NET_{sd}'].update(K0_sep=k0s, j_k0=jk, t_k0=bin_edge(jk))
                T[f'LIN_{sd}'] = dict(K0=k0l, R8=r8l, cal=_sparse(calL), theta=ev.theta, l0q=ev.l0q, evidence=ev.desc.get('spec'))
                if thk is not None:
                    T[f'LIN_{sd}'].update(K0_sep=k0ls, theta_k0=int(thk))
            if tn == JUDGED_TIER and rec_e.get('B4_rnet_relay_e1') is not None:
                T['B4_rnet_relay_e1'] = rec_e['B4_rnet_relay_e1']
            rec['tiers'][tn] = T
        rec['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _eval_and_save(job):
    rec = eval_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json")
    with open(p + '.tmp', 'w') as fh:
        json.dump(rec, fh)
    os.replace(p + '.tmp', p)
    return rec


# ================================================================ aggregation, calibration, verdict interface
def logistic(x, y1, y0):
    """weighted 1-D logistic fit (binned): P(y = 1) = sigmoid(a + b x) -> (a, b)"""
    from scipy.optimize import minimize
    x = np.asarray(x, np.float64); n1 = np.asarray(y1, np.float64); n0 = np.asarray(y0, np.float64)
    u = (n1 + n0) > 0; x, n1, n0 = x[u], n1[u], n0[u]; N = (n1 + n0).sum()
    if N == 0 or n1.sum() == 0 or n0.sum() == 0:
        return None, None

    def f(p):
        z = p[0] + p[1] * x
        ll = (n1 * np.logaddexp(0, -z) + n0 * np.logaddexp(0, z)).sum() / N
        q = 1 / (1 + np.exp(-z)); g = (q * (n1 + n0) - n1) / N
        return ll, np.array([g.sum(), (g * x).sum()])
    r = minimize(f, np.array([0., 1.]), jac=True, method='L-BFGS-B')
    return float(r.x[0]), float(r.x[1])


def calibration(cal):
    """per k: slope / intercept of truth on the logit (bin centres), ECE (20 equal-width probability bins), n; kappa = slope(>=5) / slope(1)"""
    x = LOGIT_LO + (np.arange(NBIN) + 0.5) * LOGIT_STEP; p = 1 / (1 + np.exp(-x))
    out = {}
    for k in range(KMAX):
        n1, n0 = cal[0, k], cal[1, k]; a, b = logistic(x, n1, n0)
        N = int((n1 + n0).sum()); pb = np.minimum((p * 20).astype(np.int64), 19)
        ece = None
        if N:
            e = 0.
            for q in range(20):
                m = pb == q; nn_ = float((n1 + n0)[m].sum())
                if nn_:
                    e += abs(float(n1[m].sum()) - float(((n1 + n0) * p)[m].sum()))
            ece = e / N
        out[f'k{k + 1}' if k < KMAX - 1 else f'k{KMAX}+'] = dict(n=N, occ=int(n1.sum()), intercept=a, slope=b, ece=ece)
    s1, s5 = out['k1']['slope'], out[f'k{KMAX}+']['slope']
    out['kappa'] = (s5 / s1) if (s1 and s5 is not None) else None
    return out


def _summ(rows):
    return G.summarise_rows(rows)


def aggregate(out_dir, smoke=False):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             ckpts=sorted({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}))
    reasons = []
    if len(S['script_sha256']) > 1 or len(S['ckpts']) > 1:
        reasons.append('mixed script / checkpoint versions in the job files')
    for r in recs:
        tag = f"{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}_s{r.get('seed')}"
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append(tag)
        elif r.get('split') == 'VALIDATION' and (not r.get('e1_match') or any(v is not True for v in r['e1_match'].values())):
            S['e1_mismatch'].append(tag)
    for panel in sorted({r['panel'] for r in recs}):
        P = dict(seeds={})
        for seed in sorted({r['seed'] for r in recs if r['panel'] == panel}):
            X = [r for r in recs if r['panel'] == panel and r['seed'] == seed and not r.get('failed') and not r.get('skipped')]
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={})
            for tn in EVAL_TIERS:
                if not X or not all(tn in r['tiers'] for r in X):
                    continue
                arms = sorted(k for k in X[0]['tiers'][tn] if isinstance(X[0]['tiers'][tn][k], dict) and 'K0' in X[0]['tiers'][tn][k])
                d = {}
                for arm in arms:
                    if not all(arm in r['tiers'][tn] for r in X):
                        continue
                    e = dict(K0=_summ([a for r in X for a in r['tiers'][tn][arm]['K0']]), R8=_summ([a for r in X for a in G._expand(r['tiers'][tn][arm]['R8'])]))
                    if all('K0_sep' in r['tiers'][tn][arm] for r in X):
                        e['K0_sep'] = _summ([a for r in X for a in r['tiers'][tn][arm]['K0_sep']])
                    cal = sum(_dense(r['tiers'][tn][arm]['cal']) for r in X)
                    e['calibration_R8'] = calibration(cal)
                    if all((r['tiers'][tn][arm].get('bytes') or None) for r in X):
                        e['rnet_relay_median'] = dict(sum=float(np.median([r['tiers'][tn][arm]['bytes']['sum'] for r in X])),
                                                      count=float(np.median([r['tiers'][tn][arm]['bytes']['count'] for r in X])))
                    d[arm] = e
                if tn == JUDGED_TIER and all('B4_rnet_relay_e1' in r['tiers'][tn] for r in X):
                    d['B4_rnet_relay_e1_median'] = float(np.median([r['tiers'][tn]['B4_rnet_relay_e1'] for r in X]))
                D['tiers'][tn] = d
            P['seeds'][str(seed)] = D
        S['panels'][panel] = P
    S['agg_reasons'] = reasons
    S['notes'] = dict(rho_hat='not recomputed: E3-R leaves the stalks unchanged, so the pairwise residual correlation is that of the evidence function '
                              '(E1c / RC 2.2); kappa of the LIN / B4 / B6 arms is the stalk-sum slope ratio (RC T1)',
                      kappa_network='the per-k slopes / ECE of NET_* are calibration readouts of the learned read-out (one network applied once to the '
                                    'fused sum), not an independence test')
    return S


DEFAULT_CRITERIA = dict(judged_panel='val', report_panel='fresh', judged_tier=JUDGED_TIER, train_seeds=[str(s) for s in TRAIN_SEEDS],
                        noise_seeds=[str(s) for s in EVAL_SEEDS],
                        control_arms=['B4', 'B6_A', 'B6_B'],   # D8-44: C = max(B4, B6_A, B6_B) per cell; a missing listed arm -> INCOMPLETE
                        lever_r8_margin=None,                   # D8-44: 0.03 (NET - C >= +3 pp in all 4 cells)
                        not_enough_margin=None,                 # D8-44: 0.01 (any cell below -> NOT_ENOUGH)
                        k0_margin=None,                         # D8-44: -0.01 (NET K0_sep - LIN K0_sep, both with their own holdout K0 thresholds)
                        guard_free=0.005, guard_surface=0.005,  # D8-44: free recall / R8 surface F1 >= C - 0.5 pp
                        guard_fpr=0.0,                          # review S5: FPR must EQUAL the control's (known masks asserted identical)
                        clean_min=0.99,                         # D8-44: clean NET R8 furniture and free recall >= .99
                        calib_slope_range=None, calib_ece_max=None,
                        min_updates=MIN_UPDATES,                # D8-44: fewer updates -> INCOMPLETE
                        truncated_blocks_lever=True,            # D8-44: a truncated run cannot be a lever (NOT_ENOUGH / AMBIGUOUS still judged)
                        fresh_required=True,                    # review M3.2: fresh (report panel) must be complete, else INCOMPLETE
                        fresh_sign_check=True)                  # D8-44: 4-cell mean sign opposite to VAL -> AMBIGUOUS


def _run_info(r, C, reasons, truncated):
    """one training run directory: run.json / final.json / threshold.json (review M3.1)"""
    try:
        run = _jload(os.path.join(r, 'run.json')); sd = str(run['seed'])
    except Exception as e:
        reasons.append(f'run {r}: run.json unreadable ({e!r})'); return None, None
    try:
        fin = _jload(os.path.join(r, 'final.json'))
    except Exception:
        reasons.append(f'training seed {sd}: no final.json'); return sd, None
    tp = os.path.join(r, 'threshold.json'); th = _jload(tp) if os.path.exists(tp) else None
    info = dict(updates_done=fin.get('updates_done'), updates_planned=fin.get('updates_planned'), complete=fin.get('complete'), truncated=fin.get('truncated'),
                evidence=run['recipe']['evidence']['desc'].get('spec'), train_houses=run.get('train_houses'),
                holdout_houses=(th.get('n_houses', len(th.get('houses', []))) if th else None), threshold=fin.get('threshold'),
                threshold_device=(th.get('device') if th else None))
    if th is None:
        reasons.append(f'training seed {sd}: no threshold.json')
    if (info['updates_done'] or 0) < C['min_updates']:
        reasons.append(f"training seed {sd}: {info['updates_done']} updates < {C['min_updates']}")
    if info['truncated'] or not info['complete']:
        truncated.append(sd)
    return sd, info


def _panel_houses(S, P, name, C, reasons, smoke):
    """every noise seed of a panel holds the full house set (minus that panel's and seed's deterministic skips; review S8)"""
    if smoke:
        return
    exp = {f'r{r}_i{i}' for r, i in G.panel_specs(name)}
    for ns in C['noise_seeds']:
        D = P['seeds'].get(ns)
        if D is None:
            reasons.append(f'{name}: noise seed {ns} missing'); continue
        skipped = {t[len(name) + 1:].rsplit('_s', 1)[0] for t in S.get('skipped', []) if t.startswith(name + '_') and t.endswith(f'_s{ns}')}
        miss = exp - set(D['houses']) - skipped
        if miss:
            reasons.append(f'{name}: noise seed {ns}: {len(miss)} of {len(exp)} houses missing')


def _cell(P, ts, ns, C, reasons, where, judged):
    D = P['seeds'].get(ns)
    if D is None or C['judged_tier'] not in D['tiers']:
        reasons.append(f'{where}: tier {C["judged_tier"]} missing'); return None
    tier = D['tiers'][C['judged_tier']]
    miss = [a for a in C['control_arms'] if a not in tier]
    if miss:
        reasons.append(f'{where}: control arm(s) {miss} missing'); return None             # review M3.3
    if f'NET_{ts}' not in tier:
        reasons.append(f'{where}: NET_{ts} missing'); return None
    ctrl = max(C['control_arms'], key=lambda a: tier[a]['R8']['furn_f1'])
    net = tier[f'NET_{ts}']; c = tier[ctrl]
    cell = dict(control=ctrl, net_R8_furn=net['R8']['furn_f1'], control_R8_furn=c['R8']['furn_f1'], d_R8_furn=net['R8']['furn_f1'] - c['R8']['furn_f1'],
                control_arms={a: tier[a]['R8']['furn_f1'] for a in C['control_arms']})
    if not judged:
        return cell
    lin = tier.get(f'LIN_{ts}')
    if 'K0_sep' not in net or lin is None or 'K0_sep' not in lin:
        reasons.append(f'{where}: NET / LIN K0_sep rows missing (D8-44 K0 guard)'); return None   # review M2
    cell.update(d_R8_furn_tol1=net['R8']['furn_f1_tol1'] - c['R8']['furn_f1_tol1'],
                d_K0_sep_net_minus_lin=net['K0_sep']['furn_f1'] - lin['K0_sep']['furn_f1'], net_K0_sep=net['K0_sep']['furn_f1'], lin_K0_sep=lin['K0_sep']['furn_f1'],
                report_d_K0_shared_net_minus_lin=net['K0']['furn_f1'] - lin['K0']['furn_f1'], report_d_K0_vs_control=net['K0']['furn_f1'] - c['K0']['furn_f1'],
                fpr_delta=net['R8']['unknown_known_fpr'] - c['R8']['unknown_known_fpr'], free_delta=net['R8']['free_recall'] - c['R8']['free_recall'],
                surface_delta=net['R8']['surface_f1'] - c['R8']['surface_f1'], var_delta=net['R8']['var_occ_f1'] - c['R8']['var_occ_f1'],
                net_calibration=net.get('calibration_R8'))
    if abs(cell['fpr_delta']) > C['guard_fpr'] + 1e-12:                 # review S5: identical known masks => identical FPR (else a bug)
        reasons.append(f"{where}: FPR differs from the control by {cell['fpr_delta']:+.6f} (known-mask invariant broken)")
    cl = D['tiers'].get('clean', {}).get(f'NET_{ts}')
    if cl is None:
        reasons.append(f'{where}: clean-tier NET_{ts} row missing'); return None             # review M3.4
    cell['clean_R8_furn'] = cl['R8']['furn_f1']; cell['clean_R8_free'] = cl['R8']['free_recall']
    g = dict(free=cell['free_delta'] >= -C['guard_free'] - 1e-12, surface=cell['surface_delta'] >= -C['guard_surface'] - 1e-12,
             clean=cl['R8']['furn_f1'] >= C['clean_min'] and cl['R8']['free_recall'] >= C['clean_min'])
    if C['k0_margin'] is not None:
        g['k0'] = cell['d_K0_sep_net_minus_lin'] >= C['k0_margin'] - 1e-12
    cal = net.get('calibration_R8') or {}
    if C['calib_slope_range']:
        lo, hi = C['calib_slope_range']
        g['calib_slope'] = all(v['slope'] is not None and lo <= v['slope'] <= hi for k, v in cal.items() if k.startswith('k') and v['n'])
    if C['calib_ece_max'] is not None:
        g['calib_ece'] = all(v['ece'] is not None and v['ece'] <= C['calib_ece_max'] for k, v in cal.items() if k.startswith('k') and v['n'])
    cell['guards'] = g; cell['guards_pass'] = all(g.values())
    return cell


def verdict(S, criteria=None, runs=None, smoke=False):
    """D8-44 (margins from the criteria JSON). INCOMPLETE: failed / skipped-beyond-plan jobs, E1 reproduction mismatch, a missing training
    seed, final.json or threshold.json, < min_updates, different update counts, a missing VAL or fresh cell / house / control arm / clean
    NET row / K0_sep row, FPR != control. A truncated run only blocks LEVER (review M3)."""
    C = dict(DEFAULT_CRITERIA); C.update(criteria or {})
    reasons = list(S.get('agg_reasons', []))
    if S.get('failed'):
        reasons.append(f"failed jobs: {S['failed']}")
    if S.get('e1_mismatch'):
        reasons.append(f"E1 reproduction mismatch: {S['e1_mismatch']}")
    run_info = {}; truncated = []
    if runs is None:
        reasons.append('training runs (run.json / final.json / threshold.json) not given')
    else:
        for r in runs:
            sd, info = _run_info(r, C, reasons, truncated)
            if sd is not None and info is not None:
                run_info[sd] = info
        for sd in C['train_seeds']:
            if sd not in run_info:
                reasons.append(f'training seed {sd} missing')
        if len({i['updates_planned'] for i in run_info.values()}) > 1:
            reasons.append(f"training seeds have different update counts {sorted({i['updates_planned'] for i in run_info.values()})}")
    cells, fresh = {}, {}
    P = S['panels'].get(C['judged_panel'])
    if P is None:
        reasons.append(f"panel {C['judged_panel']} missing")
    else:
        _panel_houses(S, P, C['judged_panel'], C, reasons, smoke)
        for ts in C['train_seeds']:
            for ns in C['noise_seeds']:
                c = _cell(P, ts, ns, C, reasons, f"{C['judged_panel']} cell (train {ts}, noise {ns})", judged=True)
                if c is not None:
                    cells[f'{ts}|{ns}'] = c
    Pf = S['panels'].get(C['report_panel'])
    if Pf is None:
        if C['fresh_required']:
            reasons.append(f"report panel {C['report_panel']} missing")
    else:
        if C['fresh_required']:
            _panel_houses(S, Pf, C['report_panel'], C, reasons, smoke)
        rs = reasons if C['fresh_required'] else []
        for ts in C['train_seeds']:
            for ns in C['noise_seeds']:
                c = _cell(Pf, ts, ns, C, rs, f"{C['report_panel']} cell (train {ts}, noise {ns})", judged=False)
                if c is not None:
                    fresh[f'{ts}|{ns}'] = c
    need = [C['lever_r8_margin'], C['not_enough_margin']]
    notes = []
    if reasons:
        v = 'INCOMPLETE: ' + '; '.join(reasons)
    elif any(x is None for x in need):
        v = 'PENDING_CRITERIA (readouts computed; the pre-registered margins are not filled in)'
    else:
        ds = [c['d_R8_furn'] for c in cells.values()]
        lever_cells = all(c['d_R8_furn'] >= C['lever_r8_margin'] - 1e-12 and c['guards_pass'] for c in cells.values())
        lever = lever_cells and not (C['truncated_blocks_lever'] and truncated)
        if lever_cells and not lever:
            notes.append(f'lever blocked: truncated training seed(s) {truncated}')
        notenough = any(d < C['not_enough_margin'] for d in ds)
        v = 'LEVER' if lever else ('NOT_ENOUGH' if notenough else 'AMBIGUOUS')
        if C['fresh_sign_check'] and fresh and v in ('LEVER', 'NOT_ENOUGH'):
            mv = float(np.mean(ds)); mf = float(np.mean([c['d_R8_furn'] for c in fresh.values()]))
            if mv * mf < 0:
                v = 'AMBIGUOUS (fresh sign opposite to VAL)'
    return dict(verdict=v, criteria=C, cells=cells, fresh_cells=fresh, runs=run_info, truncated_seeds=truncated, notes=notes, incomplete_reasons=reasons,
                holdout_houses={sd: i.get('holdout_houses') for sd, i in run_info.items()}, train_houses={sd: i.get('train_houses') for sd, i in run_info.items()})


# ================================================================ main
def _pool(workers, fn, jobs, json_dir, index_file):
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=G._worker_init, initargs=(json_dir, index_file), maxtasksperchild=1) as pool:
        for i, rec in enumerate(pool.imap_unordered(fn, jobs), 1):
            yield i, rec


def _check_seeds(seeds, allow):
    bad = set(seeds) & RESERVED_SEEDS
    assert allow or not bad, f'noise seeds {sorted(bad)} are reserved (VAL / fresh / E1 review / E1c fit)'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('gen', 'train', 'select', 'export', 'eval', 'aggregate', 'e1c-rule', 'fingerprint', 'selfcheck'):
        p = sub.add_parser(name)
        p.add_argument('--houseexpo-json-dir', default=None, help='default: $HOUSEEXPO_JSON_DIR at start, else the local data dir')
        p.add_argument('--houseexpo-index', default=None, help='default: $HOUSEEXPO_INDEX at start, else the local index')
        p.add_argument('--smoke', action='store_true')
        p.add_argument('--workers', type=int, default=6)
        p.add_argument('--threads', type=int, default=1)
        if name == 'gen':
            p.add_argument('--cache', required=True); p.add_argument('--seeds', default=','.join(map(str, GEN_SEEDS)))
            p.add_argument('--houses', default='', help='explicit r:i list (smoke); default = scene_plan(TRAIN, 70)')
            p.add_argument('--holdout-only', action='store_true')
        if name == 'train':
            p.add_argument('--cache', required=True); p.add_argument('--evidence', required=True); p.add_argument('--seed', type=int, required=True)
            p.add_argument('--out', required=True); p.add_argument('--updates', type=int, required=True); p.add_argument('--max-updates', type=int, default=0)
            p.add_argument('--device', default='cuda'); p.add_argument('--batch', type=int, default=8); p.add_argument('--width', type=int, default=16)
            p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--wd', type=float, default=1e-4); p.add_argument('--warmup', type=int, default=300)
            p.add_argument('--skip', type=int, default=1, help='linear skip alpha x S + beta initialised at the linear read-out (1) or none (0)')
            p.add_argument('--clip', type=float, default=1.0); p.add_argument('--loss-mask', default='scope_known', choices=('scope_known', 'scope'))
            p.add_argument('--log-every', type=int, default=100); p.add_argument('--val-every', type=int, default=1000)
            p.add_argument('--holdout-samples', type=int, default=64); p.add_argument('--deadline-epoch', type=float, default=0)
            p.add_argument('--reserve', type=float, default=300., help='seconds kept before the deadline for the threshold selection and saving')
            p.add_argument('--select', type=int, default=1); p.add_argument('--manifest-sha', default='')
        if name == 'select':
            p.add_argument('--run', required=True); p.add_argument('--device', default='cpu'); p.add_argument('--cache', default='')
            p.add_argument('--overwrite', action='store_true')
        if name == 'export':
            p.add_argument('--cache', required=True); p.add_argument('--out', required=True); p.add_argument('--panels', default='val,fresh')
            p.add_argument('--seeds', default='1,2'); p.add_argument('--houses', type=int, default=0); p.add_argument('--tiers', default=','.join(EVAL_TIERS))
            p.add_argument('--split', default='VALIDATION', choices=('VALIDATION', 'TRAIN')); p.add_argument('--train-houses', default='')
            p.add_argument('--e1-jobs', default=E1_JOBS)
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--ckpt', required=True)
            p.add_argument('--b6', action='append', default=[]); p.add_argument('--bytes', type=int, default=1); p.add_argument('--chunk', type=int, default=2)
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--criteria', default=''); p.add_argument('--runs', default='')
        if name == 'e1c-rule':
            p.add_argument('--summary', required=True); p.add_argument('--fit', required=True); p.add_argument('--ref-fit-path', required=True)
            p.add_argument('--expect-342101', default=None); p.add_argument('--expect-342202', default=None)
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True); p.add_argument('--fit', default='')
        if name == 'fingerprint':
            p.add_argument('--npz', required=True); p.add_argument('--write', default=''); p.add_argument('--expect', default='')
            p.add_argument('--arrays', default='L_k0_cam_nf,L_k0_cam,L_pub_cam', help='arrays that must be present on both sides')
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or _HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or _HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    sha = file_sha256(os.path.abspath(__file__))
    t0 = time.time()
    if args.cmd == 'gen':
        seeds = [int(s) for s in args.seeds.split(',') if s]
        _check_seeds(seeds, False)
        plan = train_plan()
        if args.houses:
            want = {tuple(int(x) for x in tok.split(':')) for tok in args.houses.split(',')}
            assert want <= set(plan), 'explicit houses must come from scene_plan(TRAIN, 70)'
            plan = [h for h in plan if h in want]
        if args.holdout_only:
            plan = [h for h in plan if is_holdout(h[1])]
        opts = dict(cache=args.cache, seeds=seeds, script_sha256=sha)
        jobs = []
        for rooms, index in plan:
            rp = os.path.join(args.cache, 'records', house_name(rooms, index) + '.json')
            if os.path.exists(rp):
                r = _jload(rp)
                if r.get('script_sha256') == sha and r.get('seeds') == seeds and not r.get('failed'):
                    continue                                          # resume only an identical record
            jobs.append((rooms, index, seeds, opts))
        print(f'gen: {len(jobs)} TRAIN houses to do ({len(plan)} planned), seeds {seeds}, {args.workers} workers', flush=True)
        for i, rec in _pool(args.workers, _gen_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['house']} {'SKIPPED ' + rec['skipped'] if rec.get('skipped') else 'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
                  f" {rec.get('seconds', 0):.0f}s {rec.get('file_bytes', 0) / 2 ** 20:.2f} MB peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
        man, path = write_manifest(args.cache, opts)
        print(f"manifest {path} sha256 {file_sha256(path)}: {len(man['houses'])} houses ({man['n_train']} train / {man['n_holdout']} holdout), "
              f"{man['total_bytes'] / 2 ** 20:.1f} MB, skipped {len(man['skipped'])}, failed {len(man['failed'])}", flush=True)
        return man
    if args.cmd == 'train':
        return train(args)
    if args.cmd == 'select':
        return select_cmd(args)
    if args.cmd == 'export':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        seeds = [int(s) for s in args.seeds.split(',') if s]
        if args.split == 'TRAIN':
            assert args.smoke, 'TRAIN-house export is for the smoke only'
            _check_seeds(seeds, False)
        tiers = [t for t in args.tiers.split(',') if t]
        assert all(t in G.CONDITIONS for t in tiers)
        tiers = [t for t in G.COND_ORDER if t in tiers]                # cam_nf before cam (shared agent-frame maps, as E1)
        opts = dict(cache=args.cache, out=args.out, split=args.split, tiers=tiers, e1_jobs=args.e1_jobs, script_sha256=sha)
        jobs = []
        if args.split == 'TRAIN':
            specs = [tuple(int(x) for x in tok.split(':')) for tok in args.train_houses.split(',') if tok]
            panels = [('train', specs)]
        else:
            panels = [(p, G.panel_specs(p)[:args.houses] if args.houses else G.panel_specs(p)) for p in args.panels.split(',')]
        for panel, specs in panels:
            for seed in seeds:
                for rooms, index in specs:
                    jobs.append((panel, rooms, index, seed, opts))
        print(f'export: {len(jobs)} jobs ({args.split}), tiers {tiers}, {args.workers} workers', flush=True)
        for i, rec in _pool(args.workers, _export_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} e1_match {rec.get('e1_match')} "
                  f"{'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}{rec.get('seconds', 0):.0f}s  {time.time() - t0:.0f}s", flush=True)
        return None
    if args.cmd == 'eval':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        ckpts = [p for p in args.ckpt.split(',') if p]
        ck_shas = {p: file_sha256(p) for p in ckpts}
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', '*.json')))]
        assert recs, f'no export records in {args.export}/records'
        recs_ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
        opts = dict(export=args.export, out=args.out, ckpts=ckpts, ckpt_shas=ck_shas, b6=args.b6, tiers=[t for t in EVAL_TIERS], bytes=bool(args.bytes),
                    threads=args.threads, chunk=args.chunk, script_sha256=sha, smoke=bool(args.smoke))
        for spec in args.b6:                                            # refuse an incomplete E1c fit before any work (M4)
            Evidence.from_spec(spec, allow_incomplete=args.smoke)
        os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
        for r in recs:
            if r.get('failed') or r.get('skipped'):
                with open(os.path.join(args.out, 'jobs', f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.json"), 'w') as fh:
                    json.dump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                                   skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))), fh)
        jobs = [(r, opts) for r in recs_ok]
        print(f'eval: {len(jobs)} jobs, checkpoints {ckpts}, controls {args.b6}, {args.workers} workers x {args.threads} threads', flush=True)
        for i, rec in _pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
                  f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
        return None
    if args.cmd == 'e1c-rule':
        ev, v, probs = e1c_rule(_jload(args.summary), file_sha256(args.fit), args.ref_fit_path)
        for sd, want in (('342101', args.expect_342101), ('342202', args.expect_342202)):
            if want is not None and want != ev[sd]:
                probs.append(f'seed {sd}: evidence {want!r} != D8-44 rule {ev[sd]!r} (E1c verdict {v!r})')
        if ev['342101'] != 'L':
            try:
                Evidence.from_spec(f'c:{args.fit}:A:cam_nf'); Evidence.from_spec(f'c:{args.fit}:B:cam_nf')
            except AssertionError as e:
                probs.append(str(e))
        print(json.dumps(dict(e1c_verdict=v, evidence=ev, fit_sha256=file_sha256(args.fit), summary_sha256=file_sha256(args.summary), problems=probs)), flush=True)
        if probs:
            raise SystemExit(2)
        return ev
    if args.cmd == 'selfcheck':                                         # review M1: run with the SAME sys.path as every chain step
        import importlib
        R = os.path.abspath(args.pyroot) + os.sep
        m = sys.modules.get('scripts.mapping_multiroom_data_v293') or importlib.import_module('scripts.mapping_multiroom_data_v293')
        oc = importlib.import_module('occupancy_codec')
        po = getattr(oc, '__file__', None) or list(oc.__path__)[0]
        bad = [x for x in (m.__file__, po) if not os.path.abspath(x).startswith(R)]
        local = [x.__name__ for x in (G, W, sys.modules.get('mapping_houseexpo_data_v365'), sys.modules.get('mapping_houseexpo_data_v371'))
                 if x is None or os.path.dirname(os.path.abspath(x.__file__)) != HERE]
        if args.fit:
            Evidence.from_spec(f'c:{args.fit}:A:cam_nf'); Evidence.from_spec(f'c:{args.fit}:B:cam_nf')
        if bad or local:
            raise SystemExit(f'import self-check FAILED: outside {R}: {bad}; not from {HERE}: {local}')
        print(f'import self-check ok {m.__file__} {po}', flush=True)
        return True
    if args.cmd == 'fingerprint':
        with np.load(args.npz) as z:
            fp = {k: W.array_hash(z[k]) for k in sorted(z.files)}
        if args.write:
            with open(args.write, 'w') as fh:
                json.dump(dict(npz=os.path.basename(args.npz), arrays=fp), fh, indent=1)
        if args.expect:
            want = _jload(args.expect)['arrays']; need = [a for a in args.arrays.split(',') if a]
            missing = [a for a in need if a not in fp or a not in want]
            common = sorted(set(fp) & set(want)); bad = [a for a in common if fp[a] != want[a]]
            print(json.dumps(dict(compared=common, mismatched=bad, missing_required=missing)), flush=True)
            if bad or missing:
                raise SystemExit(f'fingerprint check FAILED: mismatched {bad}, missing {missing}')
            print('fingerprint check OK', flush=True)
        else:
            print(json.dumps(fp, indent=1), flush=True)
        return fp
    if args.cmd == 'aggregate':
        S = aggregate(args.out, smoke=args.smoke)
        crit = _jload(args.criteria) if args.criteria else None
        runs = [r for r in args.runs.split(',') if r] or None
        S['criteria_file'] = dict(path=args.criteria, sha256=file_sha256(args.criteria)) if args.criteria else None
        S['decision'] = verdict(S, crit, runs, smoke=args.smoke)
        with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
            json.dump(S, fh, indent=1)
        print('E3-R verdict:', S['decision']['verdict'], flush=True)
        for k, c in S['decision']['cells'].items():
            print(f"  cell {k}: NET R8 furn {c['net_R8_furn']:.4f} vs {c['control']} {c['control_R8_furn']:.4f} (d {c['d_R8_furn']:+.4f}; K0 NET-LIN (own "
                  f"thresholds) {c['d_K0_sep_net_minus_lin']}) guards {c['guards']}", flush=True)
        print('  runs:', json.dumps(S['decision']['runs']), flush=True)
        return S


if __name__ == '__main__':
    main()
