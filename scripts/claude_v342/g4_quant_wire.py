"""g4_quant_wire.py (Claude, 2026-10-27): batch G4-E5 "噪声档诚实线路：源端量化 stalk，冻结 NET 读出", pre-registered in
claude_handoff/OVERNIGHT_REPORT_20261021.md D8-52 (authoritative; nothing here may change it). Background D8-46 / D8-48 / D8-50, R-B
review_E3R/RB_compression_occam.md sections 2 / 4 (probe RB/rb_probe_a.py = reference for the quantiser and the bytes), R-C
RC_sheaf_plan.md 2.4-2.5 (probe RC/rc_qpost_probe.py = reference for the report-only QPOST4 arm).
CPU only (CUDA_VISIBLE_DEVICES blanked before torch is imported), 0 leases, no retraining. Imports the FROZEN g4_e3r.py (E3-R),
g4_noise_panels.py (G4-E1) and g4_calib_stalk.py (E1c) read-only and never modifies them.

Single variable: each agent's stalk alphabet.
  Qinf (control) = E1c's 111-value table cq (half A for checkpoint 342101, half B for 342202) = E3-R as run.
  Q3 (main arm) / Q4 (fallback) = mutual-information-optimal contiguous partition of the 111 V3 symbols sorted by E1c's c(L)
     (TRAIN counts n_occ / n_free of fit.json, tier cam_nf, stable sort, ties by L) into 3 / 4 bins by dynamic programming
     (minimise sum_b N_b h(p_b) = N H(Y | bin), unsmoothed counts, every bin non-empty); level value = E1c's formula applied to the
     pooled counts of the bin, i.e. g4_calib_stalk.quantise(g4_calib_stalk.fit_table(pooled n_occ, pooled n_free)) with E1c's
     smoothing normaliser (N + 0.5 x 111), so a singleton bin reproduces E1c's cq exactly (unit-tested). Levels are numbered in
     increasing value (asserted strictly increasing).
  VOTE (report) = one-vote bins {L <= 0}, {L > 0}, values by the same formula (a ternary map on the wire).
  Cuts / values are computed BEFORE any evaluation by `quantize` from the TRAIN counts only and written with the D8-52 margins to
  e5_quant/criteria_quantizer.json (sha printed; every eval job records it).
Floor (C1-C5): the quantiser runs once per agent at the source on its own L (E.stalk with the arm's table, nothing else); the known
mask is the agent's own (asserted equal to B4's k > 0 for every arm, R8 and QPOST4); fusion = E.relay_sums (G.relay / G.comb_sum,
provenance and belief == direct component sum asserted) of the integer level values and of the count k; the level-count messages
used for the bytes are relayed the same way and asserted to reproduce the (S, k) messages exactly; the frozen NET (threshold j of
threshold.json, never re-selected) reads (S, k, z) once at the read-out. QPOST4 is the only arm with the prior at the source.

Readouts (G.score through E.scored, E3-R scopes): R8 NET furniture = mean over component-expanded rows (E3-R headline); per house
the same mean over that house's expanded rows; LIN = E1c theta of the half (unchanged); K0 NET (report) = the network on single agents
with the checkpoint's K0_sep threshold (cam_nf, arms Qinf / Q3 / Q4). Tiers: cam_nf (judged), cam (report), clean (guard .99).

Bytes (cam_nf, per house = sum over ALL directed edges of the room's forest; collect-distribute, downlink also carries sums):
  Sk(e)   = G.int_ctx_bytes(S message) + G.int_ctx_bytes(k message)          (E3-R's coder, unchanged; = Qinf's B)
  lev(e)  = lev_bytes(level-count message) of the level-count coder below (k plus every level's count except the most frequent one)
  B       = min(sum_e Sk(e), sum_e lev(e)) + 1 bit per room (the room's flag; operationalisation of "1 bit 标记": ONE format per
            room, so every node relays level counts in 'lev' rooms and sums in 'Sk' rooms -- decodable for any alphabet; the per-edge
            min is reported as B_edge_min, it is decodable only where (S, k) determines the level counts).  B(Qinf) = sum_e Sk(e).
  criterion (3): median over the cell's houses of B(arm) <= 0.5 x median of B(Qinf) (ratio of medians; median of per-house ratios
            reported alongside).
  B_RO (report, VIOLATES C5): per component the root minimising the uplink bytes (edges pointing to the root, same room format as
            B), plus (|component| - 1) x G.ctx_tern_bytes(the component's NET read-out) for the downlink, + 1 bit.
  own (report): sum over agents of the agent's own map: Sk = G.int_ctx_bytes(e_a, known_a) (k_a is the mask, already coded by the
            ternary pass), lev = lev_bytes(one-hot level map); min of the two sums reported too.
Level-count coder (written here, decodable): IDEAL ADAPTIVE CODE LENGTH of binary decisions, each with a sequential KT estimator
(alpha 1/2) per context, summed in closed form with log-Gamma (G.kt_bin_bits) -- the same class as G.int_ctx_bytes (KT contexts on a
ternary pass + binarised magnitudes) and as E0's range coder (measured overhead 0.1 %). Decisions in decoding order (C raster order
per pass; every context uses only data of earlier passes or causal neighbours already decoded):
  pass 1  known bit of every voxel; context = known bits of the 9 causal neighbours G.NB (out of grid = unknown), 512 contexts;
  pass 2  k - 1 of every known voxel, unary (bits [m > i], i = 0 .. kmax - 2, the last 1 implicit; kmax in the header); context =
          (min(i, 3), min(k(x, y, z-1), 4), min(max(k(x, y-1, z), k(x-1, y, z)), 4)), unknown / outside = 0;
  pass 3  for every known voxel, for each level except the message's most frequent one (ascending level id): its count n given the
          remaining r = k - counts already coded, unary (bits [n > i], i < r, n = r implicit); context = (slot, min(r, 3), min(i, 2),
          class of the causal neighbours (x, y, z-1), (x, y-1, z), (x-1, y, z)); class = the neighbour's dominant level (argmax of
          its counts, ties -> lower level) or 'unknown'. The most frequent level's count = the remainder.
  + 4 B header (q, dropped level, kmax), as G.int_ctx_bytes' 4 B. The test file encodes and decodes small fields with a real binary
  arithmetic coder (48-bit integer coder, KT counts 2n + 1) using the same contexts (decoded field == original, bits <= ideal + 16).
  The vectorised counter and the scalar decoder share the context definition (unit test: identical per-context counts).

QPOST4 (report only; R-C's definition, RC/rc_qpost_probe.py and RC_sheaf_plan.md 2.5 'QPOST4 (tau = 2)'): each agent runs the frozen
NET on its OWN Qinf stalk (cam_nf: L_k0 = L_pub; cam: L_pub, the map as the agent would send it), d = logit - t_K0 (t_K0 = the
checkpoint's K0 threshold, selected on TRAIN hold-out by E3-R), level q = -3 / -1 / +1 / +3 for d < -2 / [-2, 0) / [0, 2) / >= 2 on
its known voxels; relay sum of q (G.relay, asserted == direct sum); read-out sum q >= theta_post. theta_post is selected by
`qpost-select` ONLY on the TRAIN hold-out houses (scene_plan('TRAIN', 70), index % 10 == 9, cam_nf, noise seeds 200-202 = E3-R's
hold-out draws, exported here with E3-R's export_job), rule = E3-R's (E._select): pooled component-expanded furniture F1 maximal
subject to pooled free recall >= that of the same half's B6 linear read-out on the same houses, ties -> smallest theta. R-C's own
read-out (sum q >= 0) is reported as QPOST4_rc0. Bytes: same B definition with the 4-level counts.

Export: E3-R's export_job verbatim (house_ctx / tier_maps, common random numbers) through E._export_and_save, new noise seeds 3, 4
for VAL / fresh (asserted disjoint from E3-R's reserved {1, 2, 7, 101, 102} and training seeds {200, 201, 202}), tiers clean / cam_nf
/ cam; seeds 1, 2 are read from review_E3R/inputs/export (continuity report only). TEST is never generated.

Verdict (D8-52; cells = VAL cam_nf x noise seeds 3, 4 x checkpoints 342101 / 342202; arm vs Qinf of the same checkpoint and seed):
  (1) accuracy: d R8 NET furniture >= -1.0 pp in all 4 cells AND every cell's worst-house d >= -5.0 pp;
  (2) guards: d free recall >= -0.3 pp, d R8 surface F1 >= -0.5 pp, unknown->known FPR identical (asserted per component; any
      difference -> INCOMPLETE), clean-tier R8 NET furniture of the arm >= .99, in all 4 cells;
  (3) bytes: ratio of medians <= 0.5 in all 4 cells.
  arm verdict: (1)(2)(3) -> LEVER; (1)(2) pass and every cell's ratio <= 0.75 -> AMBIGUOUS; otherwise NO [operationalisation: D8-52
  names "(1) fails -> NO" only; a failed guard or a ratio > 0.75 is also mapped to NO here, reasons listed]; a cell whose furniture
  delta, worst house or byte ratio is undefined -> INCOMPLETE.
  Final (review S3): Q3 if LEVER; else Q4 if LEVER (written as the fallback); else the batch verdict is Q3's, Q4's listed beside it.
  fresh (review M1, D8-52 补充): the same 4 cells of the report panel; a LEVER is downgraded to AMBIGUOUS when the chosen arm's fresh
  4-cell mean d furniture < -1.0 pp (the accuracy margin of (1)); NO and AMBIGUOUS are never changed by fresh. The literal D8-52 sign
  comparison (fresh mean vs VAL mean) is reported as fresh_sign_opposite, with fresh's per-cell deltas and worst houses.
  INCOMPLETE: failed job, a missing VAL or fresh cell / house (24 per panel and seed), mixed script / criteria shas, job criteria sha !=
  the criteria file given to aggregate (no fall-back to built-in margins; review M2a), job script sha != the criteria file's script
  sha, smoke job files, a missing clean row, an FPR difference. Non-smoke aggregate judges ONLY on --seeds 3,4 (seeds 1, 2 only as
  --continuity; review M2b); --smoke refuses non-smoke job files and prefixes the verdict with SMOKE (review S5).
  Report only (review S2): Qinf's S+k codes the known mask twice (ternary pass of S and of k); mask_dup = sum_e G.ctx_tern_bits(the k
  message's ternary) / 8 is subtracted in the 'mask once' ratio (arm B with its Sk mode corrected the same way) -- criterion (3) stays
  on the D8-52 bytes.
  Input guards (review S1): non-smoke eval refuses a smoke / incomplete QPOST4 threshold file (houses must equal E3-R's hold-out list
  in threshold.json x seeds 200-202) or one bound to another criteria file, a criteria file written by another script version, and
  checkpoint / threshold.json shas different from the criteria file's.

Usage (python -B; CUDA_VISIBLE_DEVICES= ; <= 12 processes):
  quantize     [--out e5_quant/criteria_quantizer.json]
  export       --out DIR [--panels val,fresh] [--seeds 3,4] [--houses N] [--only r:i,...] [--workers 8]
  export       --out DIR --train-holdout [--seeds 200,201,202] [--only r:i,...] [--workers 8]          (QPOST4 threshold inputs)
  check-export --npz X.npz [--record R.json]                       (generated & readable; prints no score)
  qpost-select --export DIR --criteria C.json --out Q.json [--workers 8] [--smoke]
  eval         --export D1[,D2] --out OUT --criteria C.json [--qpost Q.json] [--panels val,fresh] [--seeds 3,4] [--files a.npz,...]
               [--workers 12] [--threads 2] [--smoke]
  aggregate    --out OUT [--criteria e5_quant/criteria_quantizer.json] [--seeds 3,4] [--continuity 1,2] [--smoke]
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3r as E  # noqa: E402   (frozen; read only)
import g4_calib_stalk as C  # noqa: E402   (frozen; read only: fit_table / quantise)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch  # noqa: E402

G = E.G
W = G.W
ROOT = E.ROOT
LMIN, LMAX, NL, UNK = E.LMIN, E.LMAX, E.NL, E.UNK
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
EXPORT_OLD = os.path.join(INP, 'export')
EXPORT_NEW = os.path.join(AN, 'g4', 'e5_export')
QUANT_DIR = os.path.join(AN, 'g4', 'e5_quant')
SMOKE_DIR = os.path.join(AN, 'g4', 'e5_smoke')
CKPTS = {'342101': dict(half='A', dir=os.path.join(INP, 's342101'), sha_prefix='1b035fd1'),
         '342202': dict(half='B', dir=os.path.join(INP, 's342202'), sha_prefix='a58b2b05')}
FIT_SHA_PREFIX = '8dd8b195'
TIER_FIT = 'cam_nf'
NEW_SEEDS = (3, 4)
CONT_SEEDS = (1, 2)
E1C_FIT_SEEDS = (101, 102)
FORBIDDEN_NEW = frozenset(E.RESERVED_SEEDS) | frozenset(E.GEN_SEEDS) | frozenset(E1C_FIT_SEEDS)
ARMS = ('Qinf', 'Q3', 'Q4', 'VOTE')
LEV_ARMS = ('Q3', 'Q4', 'VOTE')
K0_ARMS = ('Qinf', 'Q3', 'Q4')
TIERS = ('cam_nf', 'cam', 'clean')
JUDGED = 'cam_nf'
QPOST_TAU = 2.0
QPOST_VALS = np.array([-3, -1, 1, 3], np.int32)
QPOST_TIERS = ('cam_nf', 'cam')
FLAG_BYTES = 1.0 / 8.0
LEV_HEADER_B = 4.0
K_IC, K_NC = 4, 5            # level coder pass 2: unary index classes, neighbour k classes
R_C, I_C = 3, 3              # level coder pass 3: remaining-count classes, unary index classes
CRITERIA = dict(prereg='D8-52', judged_panel='val', report_panel='fresh', judged_tier=JUDGED, noise_seeds=list(NEW_SEEDS),
                continuity_seeds=list(CONT_SEEDS), checkpoints={k: v['half'] for k, v in CKPTS.items()}, main_arm='Q3', fallback_arm='Q4',
                control_arm='Qinf', acc_mean_min=-0.010, acc_worst_house_min=-0.050, guard_free_min=-0.003, guard_surface_min=-0.005,
                guard_fpr_identical=True, clean_furn_min=0.99, bytes_lever_max=0.5, bytes_ambiguous_max=0.75,
                fresh_rule='noninf_mean_downgrade_only', fresh_mean_min=-0.010,
                arm_selection='Q3 LEVER -> Q3; else Q4 LEVER -> Q4 (fallback); else the batch verdict is Q3s (Q4 listed)',
                unspecified_mappings='(2) guard failure -> NO; (3) any cell > 0.75 -> NO (both keep Qinf, as "(1) fails -> NO")',
                flag_bits=1, houses_per_panel=24)


def file_sha256(path):
    return G.file_sha256(path)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def utc():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def check_new_seeds(seeds):
    """the judged noise seeds must be fresh: not E3-R's reserved / training seeds, not E1c's fit seeds"""
    bad = set(int(s) for s in seeds) & FORBIDDEN_NEW
    assert not bad, f'noise seeds {sorted(bad)} are reserved (E3-R {sorted(E.RESERVED_SEEDS)}, training {list(E.GEN_SEEDS)}, E1c fit)'
    return True


# ================================================================ quantiser (TRAIN counts only)
def fit_counts(fit, half, tier=TIER_FIT):
    t = fit['halves'][half]['tiers'][tier]
    return np.asarray(t['n_occ'], np.float64), np.asarray(t['n_free'], np.float64), t


def _xlogx_cost(o, f):
    """N h(p) in nats for a bin with o occupied and f free counts (0 for an empty bin)"""
    N = o + f
    if N <= 0:
        return 0.
    r = 0.
    for x in (o, f):
        if x > 0:
            r -= x * math.log(x / N)
    return r


def opt_partition(no, nf, c, q):
    """MI-optimal contiguous partition of the symbols sorted by c (stable) into exactly q non-empty bins: minimise sum_b N_b h(p_b).
    Returns (labels per symbol, bins 0..q-1 in increasing c; cost in nats; order)"""
    no = np.asarray(no, np.float64); nf = np.asarray(nf, np.float64)
    order = np.argsort(np.asarray(c, np.float64), kind='stable'); a = no[order]; b = nf[order]; n = len(a)
    assert 1 <= q <= n
    A = np.concatenate([[0.], np.cumsum(a)]); B = np.concatenate([[0.], np.cumsum(b)])
    INF = float('inf'); D = np.full((q + 1, n + 1), INF); P = np.zeros((q + 1, n + 1), np.int64); D[0, 0] = 0.
    for k in range(1, q + 1):
        for j in range(k, n + 1):
            best, bi = INF, -1
            for i in range(k - 1, j):
                if D[k - 1, i] == INF:
                    continue
                v = D[k - 1, i] + _xlogx_cost(A[j] - A[i], B[j] - B[i])
                if v < best - 1e-12:
                    best, bi = v, i
            D[k, j] = best; P[k, j] = bi
    cuts = []; j = n
    for k in range(q, 0, -1):
        i = int(P[k, j]); cuts.append((i, j)); j = i
    cuts = cuts[::-1]
    lab = np.zeros(n, np.int64)
    for bidx, (i, j) in enumerate(cuts):
        lab[order[i:j]] = bidx
    return lab, float(D[q, n]), order


def partition_cost(no, nf, lab):
    return float(sum(_xlogx_cost(float(no[lab == l].sum()), float(nf[lab == l].sum())) for l in np.unique(lab)))


def level_values(no, nf, lab):
    """E1c's per-symbol formula applied to each level's pooled counts (normaliser N + 0.5 x 111: a singleton bin == E1c's cq)"""
    q = int(lab.max()) + 1
    po = np.array([no[lab == l].sum() for l in range(q)], np.float64); pf = np.array([nf[lab == l].sum() for l in range(q)], np.float64)
    assert po.sum() == no.sum() and pf.sum() == nf.sum()
    return C.quantise(C.fit_table(po, pf)).astype(np.int64)


def mutual_info_bits(no, nf, lab):
    N = no.sum() + nf.sum()
    hy = _xlogx_cost(no.sum(), nf.sum()) / N
    return float((hy - partition_cost(no, nf, lab) / N) / math.log(2))


def vote_labels():
    return (np.arange(LMIN, LMAX + 1) > 0).astype(np.int64)


def build_quantizer(fit):
    out = {}
    for half in ('A', 'B'):
        no, nf, t = fit_counts(fit, half)
        c = C.fit_table(no, nf); cq = C.quantise(c)
        assert np.allclose(c, np.asarray(t['c'], np.float64), atol=1e-9), 'fit_table(n_occ, n_free) != fit.json c'
        assert np.array_equal(cq, np.asarray(t['cq'], np.int64)), 'quantise(fit_table) != fit.json cq'
        H = dict(theta=int(t['theta']), l0q=int(t['l0q']), tq=int(t['tq']), n_occ_total=float(no.sum()), n_free_total=float(nf.sum()))
        H['Qinf'] = dict(table=cq.tolist(), labels=None, values=None, q=NL, mi_bits=mutual_info_bits(no, nf, np.arange(NL)))
        for arm, q in (('Q3', 3), ('Q4', 4)):
            lab, cost, order = opt_partition(no, nf, c, q)
            vals = level_values(no, nf, lab)
            assert np.all(np.diff(vals) > 0), f'{half} {arm}: level values not strictly increasing {vals}'
            tab = vals[lab]
            lv = []
            for l in range(q):
                syms = np.flatnonzero(lab == l)
                lv.append(dict(level=l, value=int(vals[l]), n_symbols=int(len(syms)), L=[int(s + LMIN) for s in syms],
                               c_min=float(c[syms].min()), c_max=float(c[syms].max()), n_occ=float(no[syms].sum()), n_free=float(nf[syms].sum())))
            H[arm] = dict(table=tab.tolist(), labels=lab.tolist(), values=vals.tolist(), q=q, cost_nats=cost, mi_bits=mutual_info_bits(no, nf, lab),
                          levels=lv, c_order_cut_sizes=[d['n_symbols'] for d in lv])
        lab = vote_labels(); vals = level_values(no, nf, lab)
        assert np.all(np.diff(vals) > 0)
        H['VOTE'] = dict(table=vals[lab].tolist(), labels=lab.tolist(), values=vals.tolist(), q=2, mi_bits=mutual_info_bits(no, nf, lab),
                         levels=[dict(level=0, value=int(vals[0]), L='<= 0', n_symbols=int((lab == 0).sum())),
                                 dict(level=1, value=int(vals[1]), L='> 0', n_symbols=int((lab == 1).sum()))])
        out[half] = H
    return out


def quantize_cmd(args):
    fit_sha = file_sha256(FIT)
    assert fit_sha.startswith(FIT_SHA_PREFIX), f'E1c fit sha {fit_sha[:8]} != {FIT_SHA_PREFIX}'
    fit = _jload(FIT)
    assert fit.get('complete') is True and not fit.get('smoke'), 'E1c fit must be the complete (non-smoke) fit'
    ck = {}
    for sd, d in CKPTS.items():
        s = file_sha256(os.path.join(d['dir'], 'final.pt'))
        assert s.startswith(d['sha_prefix']), f'{sd}: checkpoint sha {s[:8]} != {d["sha_prefix"]}'
        thr = _jload(os.path.join(d['dir'], 'threshold.json'))
        ck[sd] = dict(half=d['half'], final_pt_sha256=s, threshold_sha256=file_sha256(os.path.join(d['dir'], 'threshold.json')), j=int(thr['j']),
                      t=float(thr['t']), j_k0=int(thr['k0']['j_sep']), t_k0=float(thr['k0']['t_sep']))
    Q = build_quantizer(fit)
    for sd in CKPTS:                                                    # the checkpoint's own evidence table == Qinf of its half
        ckp, _ = E.load_ckpt(os.path.join(CKPTS[sd]['dir'], 'final.pt'), 'cpu')
        ev = E.Evidence.from_recipe(ckp['recipe']['evidence'])
        assert ev.table.tolist() == Q[CKPTS[sd]['half']]['Qinf']['table'] and ev.theta == Q[CKPTS[sd]['half']]['theta'], f'{sd}: evidence != E1c half'
    rb = dict(Q3_A_values=[-97, 5, 95], Q3_A_cut_sizes=[31, 56, 24])   # R-B probe (RB_compression_occam.md section 4), for comparison only
    doc = dict(batch='G4-E5', prereg='claude_handoff/OVERNIGHT_REPORT_20261021.md D8-52', written_utc=utc(), script_sha256=file_sha256(os.path.abspath(__file__)),
               fit=dict(path=os.path.relpath(FIT, ROOT), sha256=fit_sha, tier=TIER_FIT, seeds=fit.get('seeds'), houses={h: fit['halves'][h]['n_houses'] for h in ('A', 'B')}),
               checkpoints=ck, criteria=CRITERIA, quantizer=Q,
               formula=dict(order='symbols sorted by E1c c(L) (fit_table of the TRAIN counts), stable', objective='min sum_b N_b h(p_b) (unsmoothed), exactly q non-empty contiguous bins',
                            value='g4_calib_stalk.quantise(g4_calib_stalk.fit_table(pooled n_occ, pooled n_free)) = round((log((no_b+.5)/(N_occ+.5*111)) - log((nf_b+.5)/(N_free+.5*111)))/0.05)',
                            vote='bins L <= 0 / L > 0, same value formula'),
               rb_probe_reference=rb, rb_probe_match=dict(Q3_A_values=Q['A']['Q3']['values'] == rb['Q3_A_values'],
                                                          Q3_A_cut_sizes=Q['A']['Q3']['c_order_cut_sizes'] == rb['Q3_A_cut_sizes']))
    _jdump(doc, args.out)
    sha = file_sha256(args.out)
    print(f'criteria/quantizer {args.out}\nsha256 {sha}', flush=True)
    for h in ('A', 'B'):
        for arm in ('Q3', 'Q4', 'VOTE'):
            d = Q[h][arm]
            print(f"  half {h} {arm}: values {d['values']} symbols per level {[x['n_symbols'] for x in d['levels']]} MI {d['mi_bits']:.4f} bits "
                  f"(Qinf {Q[h]['Qinf']['mi_bits']:.4f})", flush=True)
    print('  R-B probe match (half A Q3):', json.dumps(doc['rb_probe_match']), flush=True)
    return doc


def load_criteria(path):
    doc = _jload(path)
    Q = doc['quantizer']
    for h in ('A', 'B'):
        for arm in ARMS:
            d = Q[h][arm]; d['table_np'] = np.asarray(d['table'], np.int64)
            d['labels_np'] = None if d['labels'] is None else np.asarray(d['labels'], np.int64)
            d['values_np'] = None if d['values'] is None else np.asarray(d['values'], np.int64)
            assert d['table_np'].shape == (NL,)
            if d['labels_np'] is not None:
                assert np.array_equal(d['values_np'][d['labels_np']], d['table_np'])
    return doc


# ================================================================ level-count coder (ideal adaptive code length; decodable)
def _shift(a, d, fill):
    """b[x, y, z] = a[x - dx, y - dy, z - dz]; fill outside the grid"""
    dx, dy, dz = d
    X, Y, Z = a.shape
    b = np.full(a.shape, fill, a.dtype)
    b[dx:, dy:, dz:] = a[:X - dx, :Y - dy, :Z - dz]
    return b


def lev_ctx_sizes(q):
    return 1 << len(G.NB), K_IC * K_NC * K_NC, (q - 1) * R_C * I_C * (q + 1) ** 3


def lev_header(N):
    N = np.asarray(N); q = N.shape[0]; k = N.sum(0)
    tot = N.reshape(q, -1).sum(1); drop = int(np.argmax(tot))
    kmax = int(k.max()) if k.size else 0
    assert 2 <= q <= 8 and kmax <= 255
    return dict(q=q, drop=drop, kmax=kmax, coded=[l for l in range(q) if l != drop])


def lev_counts(N):
    """vectorised per-context (n0, n1) of every binary decision of the level-count coder (header excluded)"""
    N = np.asarray(N, np.int64); assert N.ndim == 4 and (N >= 0).all()
    hd = lev_header(N); q = hd['q']; k = N.sum(0); K = k > 0
    s1, s2, s3 = lev_ctx_sizes(q); o2, o3 = s1, s1 + s2
    n0 = np.zeros(s1 + s2 + s3, np.int64); n1 = np.zeros_like(n0)

    def add(ctx, bit):
        n1[:] += np.bincount(ctx[bit], minlength=len(n1)); n0[:] += np.bincount(ctx[~bit], minlength=len(n0))
    Ki = K.astype(np.int64); c1 = np.zeros(K.shape, np.int64)
    for i, d in enumerate(G.NB):                                        # pass 1: known mask
        c1 |= _shift(Ki, d, 0) << i
    add(c1.ravel(), K.ravel())
    kz = np.minimum(_shift(k, (0, 0, 1), 0), K_NC - 1)                 # pass 2: k - 1, unary
    kyx = np.minimum(np.maximum(_shift(k, (0, 1, 0), 0), _shift(k, (1, 0, 0), 0)), K_NC - 1)
    m = k - 1
    for i in range(max(hd['kmax'] - 1, 0)):
        sel = K & (m >= i)
        if not sel.any():
            break
        add(o2 + (min(i, K_IC - 1) * K_NC + kz[sel]) * K_NC + kyx[sel], m[sel] > i)
    cls = np.argmax(N, 0).astype(np.int64); cls[~K] = q                 # pass 3: level counts given the remainder
    Q1 = q + 1
    nbc = (_shift(cls, (0, 0, 1), q) * Q1 + _shift(cls, (0, 1, 0), q)) * Q1 + _shift(cls, (1, 0, 0), q)
    r = k.copy()
    for j, l in enumerate(hd['coded']):
        nl = N[l]; rc = np.minimum(r, R_C) - 1
        for i in range(int(r.max()) if K.any() else 0):
            sel = K & (nl >= i) & (r > i)
            if not sel.any():
                break
            add(o3 + ((j * R_C + rc[sel]) * I_C + min(i, I_C - 1)) * Q1 ** 3 + nbc[sel], nl[sel] > i)
        r = r - nl
    assert np.array_equal(r, N[hd['drop']]), 'level coder: remainder != dropped level'
    return n0, n1, hd


def lev_bits(N):
    n0, n1, _ = lev_counts(N)
    return G.kt_bin_bits(n0, n1)


def lev_bytes(N):
    return lev_bits(N) / 8 + LEV_HEADER_B


def lev_symbols(N):
    """scalar reference walker: the (context, bit) decisions in DECODING order (voxel-outer, C raster order per pass)"""
    N = np.asarray(N, np.int64); hd = lev_header(N); q = hd['q']; k = N.sum(0); K = k > 0
    X, Y, Z = k.shape; s1, s2, s3 = lev_ctx_sizes(q); o2, o3 = s1, s1 + s2; Q1 = q + 1
    cls = np.argmax(N, 0).astype(np.int64); cls[~K] = q

    def g(a, x, y, z, fill):
        return int(a[x, y, z]) if (x >= 0 and y >= 0 and z >= 0) else fill
    seq = []
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                c = 0
                for i, (dx, dy, dz) in enumerate(G.NB):
                    c |= g(K, x - dx, y - dy, z - dz, 0) << i
                seq.append((c, bool(K[x, y, z])))
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not K[x, y, z]:
                    continue
                m = int(k[x, y, z]) - 1
                kz = min(g(k, x, y, z - 1, 0), K_NC - 1); kyx = min(max(g(k, x, y - 1, z, 0), g(k, x - 1, y, z, 0)), K_NC - 1)
                for i in range(hd['kmax'] - 1):
                    bit = m > i; seq.append((o2 + (min(i, K_IC - 1) * K_NC + kz) * K_NC + kyx, bit))
                    if not bit:
                        break
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not K[x, y, z]:
                    continue
                nb = (g(cls, x, y, z - 1, q) * Q1 + g(cls, x, y - 1, z, q)) * Q1 + g(cls, x - 1, y, z, q)
                r = int(k[x, y, z])
                for j, l in enumerate(hd['coded']):
                    if r == 0:
                        break
                    n = int(N[l, x, y, z]); rc = min(r, R_C) - 1
                    for i in range(r):
                        bit = n > i; seq.append((o3 + ((j * R_C + rc) * I_C + min(i, I_C - 1)) * Q1 ** 3 + nb, bit))
                        if not bit:
                            break
                    r -= n
    return seq, hd


class _ArithEnc:
    """binary arithmetic coder (Witten-Neal-Cleary, pending bits), P-bit integer state; probabilities from integer counts"""
    P = 48

    def __init__(self):
        self.lo, self.hi, self.pend, self.out = 0, (1 << self.P) - 1, 0, []

    def _emit(self, b):
        self.out.append(b); self.out.extend([1 - b] * self.pend); self.pend = 0

    def encode(self, bit, c0, c1):
        H, Qt = 1 << (self.P - 1), 1 << (self.P - 2)
        rng = self.hi - self.lo + 1; split = self.lo + (rng * c0) // (c0 + c1) - 1
        if bit:
            self.lo = split + 1
        else:
            self.hi = split
        while True:
            if self.hi < H:
                self._emit(0)
            elif self.lo >= H:
                self._emit(1); self.lo -= H; self.hi -= H
            elif self.lo >= Qt and self.hi < 3 * Qt:
                self.pend += 1; self.lo -= Qt; self.hi -= Qt
            else:
                break
            self.lo = 2 * self.lo; self.hi = 2 * self.hi + 1

    def finish(self):
        self.pend += 1
        self._emit(0 if self.lo < (1 << (self.P - 2)) else 1)
        return self.out


class _ArithDec:
    P = 48

    def __init__(self, bits):
        self.bits, self.pos = bits, 0
        self.lo, self.hi, self.val = 0, (1 << self.P) - 1, 0
        for _ in range(self.P):
            self.val = 2 * self.val + self._next()

    def _next(self):
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def decode(self, c0, c1):
        H, Qt = 1 << (self.P - 1), 1 << (self.P - 2)
        rng = self.hi - self.lo + 1; split = self.lo + (rng * c0) // (c0 + c1) - 1
        if self.val <= split:
            bit = 0; self.hi = split
        else:
            bit = 1; self.lo = split + 1
        while True:
            if self.hi < H:
                pass
            elif self.lo >= H:
                self.lo -= H; self.hi -= H; self.val -= H
            elif self.lo >= Qt and self.hi < 3 * Qt:
                self.lo -= Qt; self.hi -= Qt; self.val -= Qt
            else:
                break
            self.lo = 2 * self.lo; self.hi = 2 * self.hi + 1; self.val = 2 * self.val + self._next()
        return bit


def lev_encode(N):
    """real arithmetic code of a level-count field with the coder's adaptive KT contexts (counts 2n + 1): (bits, header)"""
    seq, hd = lev_symbols(N); enc = _ArithEnc(); cnt = {}
    for ctx, bit in seq:
        c = cnt.setdefault(ctx, [1, 1]); enc.encode(int(bit), c[0], c[1]); c[int(bit)] += 2
    return enc.finish(), hd


def lev_decode(bits, shape, hd):
    """the decoder: rebuilds the field from the bits and the 4 B header, computing every context from decoded data only"""
    q, kmax, coded, drop = hd['q'], hd['kmax'], hd['coded'], hd['drop']
    X, Y, Z = shape; s1, s2, s3 = lev_ctx_sizes(q); o2, o3 = s1, s1 + s2; Q1 = q + 1
    dec = _ArithDec(bits); cnt = {}

    def d(ctx):
        c = cnt.setdefault(ctx, [1, 1]); b = dec.decode(c[0], c[1]); c[b] += 2; return b
    K = np.zeros(shape, bool); k = np.zeros(shape, np.int64); N = np.zeros((q,) + tuple(shape), np.int64); cls = np.full(shape, q, np.int64)

    def g(a, x, y, z, fill):
        return int(a[x, y, z]) if (x >= 0 and y >= 0 and z >= 0) else fill
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                c = 0
                for i, (dx, dy, dz) in enumerate(G.NB):
                    c |= g(K, x - dx, y - dy, z - dz, 0) << i
                K[x, y, z] = bool(d(c))
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not K[x, y, z]:
                    continue
                kz = min(g(k, x, y, z - 1, 0), K_NC - 1); kyx = min(max(g(k, x, y - 1, z, 0), g(k, x - 1, y, z, 0)), K_NC - 1)
                m = kmax - 1
                for i in range(kmax - 1):
                    if not d(o2 + (min(i, K_IC - 1) * K_NC + kz) * K_NC + kyx):
                        m = i; break
                k[x, y, z] = m + 1
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not K[x, y, z]:
                    continue
                nb = (g(cls, x, y, z - 1, q) * Q1 + g(cls, x, y - 1, z, q)) * Q1 + g(cls, x - 1, y, z, q)
                r = int(k[x, y, z])
                for j, l in enumerate(coded):
                    if r == 0:
                        break
                    rc = min(r, R_C) - 1; n = r
                    for i in range(r):
                        if not d(o3 + ((j * R_C + rc) * I_C + min(i, I_C - 1)) * Q1 ** 3 + nb):
                            n = i; break
                    N[l, x, y, z] = n; r -= n
                N[drop, x, y, z] = r
                cls[x, y, z] = int(np.argmax(N[:, x, y, z]))
    return N


def onehot_levels(L, labels, q):
    """one agent's level-count map: 1 at (level of L, voxel) on its known voxels (int32), and its known mask"""
    L = np.asarray(L); K = ~G.is_unk(L)
    li = np.asarray(labels, np.int64)[np.clip(L.astype(np.int64) - LMIN, 0, NL - 1)]
    N = np.zeros((q,) + L.shape, np.int32)
    for l in range(q):
        N[l] = (K & (li == l))
    return N, K


# ================================================================ bytes helpers
def toward_root(F, cm, root):
    """directed edges of the component cm that point toward root (root not on the sender's side)"""
    return [e for e in F['directed'] if e[0] in cm and root not in F['sides'][e]]


def b_ro(F, edge_bytes_by_mode, tern_by_comp, flag=True):
    """report-only B_RO (violates C5): one format per room; per component the root with the least uplink bytes, + (|cm| - 1) x the
    component's ternary read-out on the downlink, + the room flag"""
    best = None
    for mode, eb in edge_bytes_by_mode.items():
        up = 0.
        for cm in F['comps']:
            if len(cm) > 1:
                up += min(sum(eb[e] for e in toward_root(F, cm, r)) for r in cm)
        best = up if best is None else min(best, up)
    dn = sum((len(cm) - 1) * tern_by_comp[ci] for ci, cm in enumerate(F['comps']))
    return best + dn + (FLAG_BYTES if flag else 0.)


def ekey(e):
    return f'{e[0]}>{e[1]}'


def qpost_levels(logit, t_k0, tau=QPOST_TAU):
    """R-C's QPOST4 levels 0..3 (values -3 / -1 / +1 / +3) of d = logit - t_K0"""
    d = np.asarray(logit, np.float64) - t_k0
    return np.where(d < -tau, 0, np.where(d < 0, 1, np.where(d < tau, 2, 3))).astype(np.int64)


# ================================================================ export (E3-R's export_job, new seeds)
def export_cmd(args, json_dir, index_file):
    sha = file_sha256(os.path.abspath(__file__))
    if args.train_holdout:
        seeds = [int(s) for s in (args.seeds or ','.join(map(str, E.GEN_SEEDS))).split(',') if s]
        assert set(seeds) <= set(E.GEN_SEEDS), 'TRAIN hold-out export: E3-R hold-out draws only (seeds 200-202)'
        tiers = [JUDGED]; split = 'TRAIN'
        specs = [h for h in E.train_plan() if E.is_holdout(h[1])]
        if args.only:
            want = {tuple(int(x) for x in t.split(':')) for t in args.only.split(',') if t}
            assert want <= set(specs), '--only must name TRAIN hold-out houses'
            specs = [h for h in specs if h in want]
        elif args.houses:
            specs = specs[:args.houses]
        panels = [('trainho', specs)]
    else:
        seeds = [int(s) for s in (args.seeds or ','.join(map(str, NEW_SEEDS))).split(',') if s]
        check_new_seeds(seeds)
        tiers = [t for t in G.COND_ORDER if t in TIERS]; split = 'VALIDATION'
        panels = []
        for p in args.panels.split(','):
            specs = G.panel_specs(p)
            if args.only:
                want = {tuple(int(x) for x in t.split(':')) for t in args.only.split(',') if t}
                specs = [h for h in specs if h in want]
            elif args.houses:
                specs = specs[:args.houses]
            panels.append((p, specs))
    opts = dict(cache=args.out, out=args.out, split=split, tiers=tiers, e1_jobs=E.E1_JOBS, script_sha256=sha)
    jobs = []
    for panel, specs in panels:
        for seed in seeds:
            for rooms, index in specs:
                rp = os.path.join(args.out, 'records', f'{panel}_r{rooms}_i{index}_s{seed}.json')
                if os.path.exists(rp):
                    r = _jload(rp)
                    if not r.get('failed') and r.get('script_sha256') == sha and (r.get('skipped') or os.path.exists(os.path.join(args.out, r.get('file', '')))):
                        continue
                jobs.append((panel, rooms, index, seed, opts))
    os.makedirs(args.out, exist_ok=True)
    _jdump(dict(stage='export', utc=utc(), script_sha256=sha, g4_e3r_sha256=file_sha256(E.__file__), noise_panels_sha256=file_sha256(G.__file__),
                split=split, seeds=seeds, tiers=tiers, panels={p: [list(h) for h in s] for p, s in panels}), os.path.join(args.out, f'export_run_{split}.json'))
    print(f'export: {len(jobs)} jobs ({split}), seeds {seeds}, tiers {tiers}, {args.workers} workers', flush=True)
    t0 = time.time()
    for i, rec in E._pool(args.workers, E._export_and_save, jobs, json_dir, index_file):
        st = 'SKIPPED ' + rec['skipped'] if rec.get('skipped') else ('FAILED ' + rec.get('error', '') if rec.get('failed') else 'ok')
        print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {st} {rec.get('seconds', 0):.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)


def check_export_cmd(args):
    """'能生成、能读' check of one exported file: sha against its record, array names / shapes / dtypes, alphabet. No score."""
    rec = _jload(args.record) if args.record else None
    if rec is not None:
        assert file_sha256(args.npz) == rec['file_sha256'], 'export file sha != record'
    with np.load(args.npz) as z:
        info = {k: [list(z[k].shape), str(z[k].dtype)] for k in sorted(z.files)}
        n = z['private'].shape[0]
        for k in z.files:
            if k.startswith('L_'):
                a = z[k]; assert a.dtype == np.int8 and a.shape[0] == n
                v = a[a != UNK]; assert v.size == 0 or (v.min() >= LMIN and v.max() <= LMAX), f'{k}: value outside the V3 alphabet'
        need = ['L_k0_cam_nf'] + (['L_k0_cam', 'L_pub_cam', 'L_k0_clean'] if not os.path.basename(args.npz).startswith('trainho') else [])
        miss = [k for k in need if k not in z.files]
        assert not miss, f'missing arrays {miss}'
    print(json.dumps(dict(npz=os.path.basename(args.npz), record_sha_ok=rec is not None, agents=n, arrays=info,
                          components=(rec or {}).get('components'), seconds=(rec or {}).get('seconds'))), flush=True)


# ================================================================ QPOST4 threshold on TRAIN hold-out
def e3r_holdout_houses(sd):
    """the hold-out houses E3-R selected its thresholds on (threshold.json of the checkpoint), e.g. 'TRAIN_r1_i9'"""
    return list(_jload(os.path.join(CKPTS[sd]['dir'], 'threshold.json'))['houses'])


def check_qpost_file(qp, criteria_sha, smoke):
    """review S1: the QPOST4 threshold file a (non-smoke) eval may use"""
    if smoke:
        return
    assert not qp.get('smoke'), 'non-smoke eval refuses a smoke QPOST4 threshold file'
    assert qp.get('criteria_sha256') == criteria_sha, 'QPOST4 threshold file was selected with another criteria file'
    for sd in CKPTS:
        hs = set(e3r_holdout_houses(sd))
        assert set(qp.get('houses', [])) == hs, f'QPOST4 threshold houses != E3-R hold-out houses ({len(hs)})'
    assert sorted(qp.get('seeds', [])) == sorted(E.GEN_SEEDS) and len(qp['files']) == len(qp['houses']) * len(E.GEN_SEEDS), 'QPOST4 threshold file incomplete'


def _qpost_hist_job(job):
    path, opts = job
    t0 = time.time(); rec = dict(file=os.path.basename(path))
    try:
        torch.set_num_threads(max(1, opts['threads']))
        crit = load_criteria(opts['criteria'])
        z = np.load(path)
        Hh = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(E.edges_list(z['edges']), n)
        R8T = [G.Target(private[cm].max(0), Hh) for cm in F['comps']]          # = E.select_threshold's scope
        maps = list(z['L_k0_cam_nf'])
        out = {}
        for sd, d in CKPTS.items():
            ck, model = E.load_ckpt(os.path.join(d['dir'], 'final.pt'), 'cpu')
            thr = _jload(os.path.join(d['dir'], 'threshold.json')); tk = E.bin_edge(int(thr['k0']['j_sep']))
            Qd = crit['quantizer'][d['half']]; tab = Qd['Qinf']['table_np']; theta = int(Qd['theta'])
            own = [E.stalk(L, tab) for L in maps]
            lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', opts['chunk'])
            qs = [np.where(own[a][1], QPOST_VALS[qpost_levels(lk[a], tk)], 0).astype(np.int32) for a in range(n)]
            A = np.zeros(E.VB, np.int64); B = np.zeros(E.VB, np.int64); Cc = np.zeros(E.VB, np.int64); lin = np.zeros(3, np.int64); tr = fr = 0
            for ci, cm in enumerate(F['comps']):
                Qs = sum(qs[a] for a in cm); S6, kc = E.direct_sums(maps, cm, tab); T = R8T[ci]; w = len(cm)
                a_, b_, c_ = E._cat_masks(kc, T, Hh); vb = E._vbin(Qs)
                A += np.bincount(vb[a_], minlength=E.VB) * w; B += np.bincount(vb[b_], minlength=E.VB) * w; Cc += np.bincount(vb[c_], minlength=E.VB) * w
                tr += int(T.tr.sum()) * w; fr += int(T.free.sum()) * w
                ol = S6 > theta
                lin += np.array([int((a_ & ol).sum()), int((b_ & ol).sum()), int((c_ & ~ol).sum())]) * w
            out[sd] = dict(A=A.tolist(), B=B.tolist(), C=Cc.tolist(), lin=lin.tolist(), tr=tr, fr=fr, t_k0=tk)
        rec['hist'] = out; rec['seconds'] = time.time() - t0
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def qpost_select_cmd(args, json_dir, index_file):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', 'trainho_*.json')))]
    assert recs, 'no TRAIN hold-out export records'
    ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
    bad = [r for r in recs if r.get('failed')]
    holdout = [h for h in E.train_plan() if E.is_holdout(h[1])]
    if not args.smoke:
        assert not bad, f'failed TRAIN hold-out exports: {[r["file"] for r in bad]}'
        have = {(r['rooms'], r['index'], r['seed']) for r in recs if not r.get('failed')}
        miss = [(h, s) for h in holdout for s in E.GEN_SEEDS if (h[0], h[1], s) not in have]
        assert not miss, f'missing TRAIN hold-out exports {miss[:5]}... ({len(miss)})'
        got = {f"TRAIN_r{r['rooms']}_i{r['index']}" for r in ok}
        for sd in CKPTS:                                               # the same 19 hold-out houses as E3-R's threshold selection
            assert got == set(e3r_holdout_houses(sd)), f'{sd}: hold-out houses {sorted(got)} != threshold.json houses'
        assert len(ok) == len(got) * len(E.GEN_SEEDS) and {int(r['seed']) for r in ok} == set(E.GEN_SEEDS)
    for r in ok:
        assert file_sha256(os.path.join(args.export, r['file'])) == r['file_sha256'], f"{r['file']} changed"
    opts = dict(criteria=args.criteria, threads=args.threads, chunk=2)
    jobs = [(os.path.join(args.export, r['file']), opts) for r in ok]
    print(f'qpost-select: {len(jobs)} TRAIN hold-out files, {args.workers} workers', flush=True)
    t0 = time.time(); res = []
    for i, rec in E._pool(args.workers, _qpost_hist_job, jobs, json_dir, index_file):
        assert not rec.get('failed'), rec.get('traceback')
        res.append(rec); print(f"[{i}/{len(jobs)}] {rec['file']} {rec['seconds']:.0f}s  {time.time() - t0:.0f}s", flush=True)
    out = dict(stage='qpost-select', utc=utc(), script_sha256=file_sha256(os.path.abspath(__file__)), criteria_sha256=file_sha256(args.criteria),
               smoke=bool(args.smoke), tau=QPOST_TAU, values=QPOST_VALS.tolist(), files=sorted(r['file'] for r in res),
               n_houses=len({r['file'].rsplit('_s', 1)[0] for r in res}),
               houses=sorted({'TRAIN_' + r['file'][len('trainho_'):].rsplit('_s', 1)[0] for r in res}),
               seeds=sorted({int(r['file'].rsplit('_s', 1)[1].split('.')[0]) for r in res}),
               rule='E._select on TRAIN hold-out R8 component sums of q (component-expanded): pooled furniture F1 max s.t. pooled free recall >= '
                    "the same half's B6 linear read-out (sum cq > theta) on the same houses; ties -> smallest theta; read-out sum q >= theta",
               definition="R-C rc_qpost_probe.py: d = NET logit on the agent's own Qinf stalk - t_K0 (threshold.json k0.j_sep); "
                          'q = -3 / -1 / +1 / +3 for d < -tau / [-tau, 0) / [0, tau) / >= tau, tau = 2', checkpoints={})
    for sd in CKPTS:
        A = sum(np.asarray(r['hist'][sd]['A'], np.int64) for r in res); B = sum(np.asarray(r['hist'][sd]['B'], np.int64) for r in res)
        Cc = sum(np.asarray(r['hist'][sd]['C'], np.int64) for r in res); lin = sum(np.asarray(r['hist'][sd]['lin'], np.int64) for r in res)
        tr = sum(r['hist'][sd]['tr'] for r in res); fr = sum(r['hist'][sd]['fr'] for r in res)
        j, f1, frc, tp, fp, fn, lin_f1, lin_fr = E._select(A, B, Cc, tr, fr, lin)
        theta = int(j - E.VOFF)
        assert -E.VOFF < theta < E.VOFF - 1, 'theta on the histogram boundary'
        j0 = E.VOFF                                                    # R-C's read-out: sum q >= 0
        out['checkpoints'][sd] = dict(theta=theta, f1=float(f1[j]), free=float(frc[j]), lin_f1=float(lin_f1), lin_free=float(lin_fr),
                                      rc0_f1=float(f1[j0]), rc0_free=float(frc[j0]), t_k0=res[0]['hist'][sd]['t_k0'])
        print(f"  {sd}: theta_post {theta} (sum q >= theta)  TRAIN-holdout F1 {f1[j]:.4f} free {frc[j]:.4f} | lin F1 {lin_f1:.4f} free {lin_fr:.4f} | "
              f"R-C sum q >= 0: F1 {f1[j0]:.4f} free {frc[j0]:.4f}", flush=True)
    _jdump(out, args.out)
    print(f'qpost threshold {args.out} sha256 {file_sha256(args.out)}', flush=True)
    return out


# ================================================================ eval
def _known_assert(p, known, what):
    assert np.array_equal(p >= 0, known), f'{what}: known mask != B4 (k > 0)'


def eval_job(job):
    spec, opts = job
    t0 = time.time()
    rec = dict(stage='eval', panel=spec['panel'], rooms=spec['rooms'], index=spec['index'], seed=spec['seed'], file=spec['file'],
               export_dir=spec['export_dir'], e1_match=spec.get('e1_match'), script_sha256=opts['script_sha256'], criteria_sha256=opts['criteria_sha256'],
               qpost_sha256=opts.get('qpost_sha256'), smoke=bool(opts.get('smoke')), timing={})
    try:
        torch.set_num_threads(max(1, opts['threads']))
        crit = load_criteria(opts['criteria']); Q = crit['quantizer']
        qpost = _jload(opts['qpost']) if opts.get('qpost') else None
        path = os.path.join(spec['export_dir'], spec['file'])
        assert file_sha256(path) == spec['file_sha256'], 'export file changed'
        z = np.load(path)
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(E.edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in spec['components']]
        K0T = [G.Target(private[a], H) for a in range(n)]
        R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
        models = {}
        for sd, d in CKPTS.items():
            ck, model = E.load_ckpt(os.path.join(d['dir'], 'final.pt'), 'cpu')
            thr = _jload(os.path.join(d['dir'], 'threshold.json'))
            models[sd] = (model, int(thr['j']), int(thr['k0']['j_sep']), d['half'])
        rstats = G.RelayStats(); rec['tiers'] = {}; rec['comps'] = F['comps']; rec['n'] = n
        Ltab = E.Evidence.from_spec('L').table
        for tn in opts['tiers']:
            ta = time.time()
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            sumsL, _, msgnL = E.relay_sums(L_pub, F, Ltab, rstats)            # B4: the known mask every arm must share
            r8_known = [s[1] > 0 for s in sumsL]; kcs = [s[1] for s in sumsL]
            kbytes = {e: G.int_ctx_bytes(*msgnL[e]) for e in F['directed']} if tn == JUDGED else None
            if tn == JUDGED:                                                   # review S2: the k message's ternary pass = the mask again
                rec['mask_dup_bytes'] = float(sum(G.ctx_tern_bits(np.where(msgnL[e][1], np.int8(1), np.int8(-1))) for e in F['directed']) / 8)
            T = {}
            for sd, (model, j, jk, half) in models.items():
                Qh = Q[half]; theta = int(Qh['theta']); D = {}
                for arm in ARMS:
                    qa = Qh[arm]; tab = qa['table_np']
                    sums, msgs, msgn = E.relay_sums(L_pub, F, tab, rstats)
                    for e in F['directed']:                                    # the count channel does not depend on the alphabet
                        assert np.array_equal(msgn[e][0], msgnL[e][0])
                    lg = E.net_logits(model, [s[0] for s in sums], [s[1] for s in sums], 'cpu', opts['chunk'])
                    rn, rl, preds = [], [], []
                    for ci, cm in enumerate(F['comps']):
                        S, kc = sums[ci]; assert np.array_equal(kc, kcs[ci])
                        p = E.net_readout(lg[ci], kc, j); _known_assert(p, r8_known[ci], f'{tn} {sd} {arm} NET')
                        pl = E.linear_readout(S, kc, theta); _known_assert(pl, r8_known[ci], f'{tn} {sd} {arm} LIN')
                        rn.append([E.scored(p, R8T[ci], H, kc) if tn == JUDGED else G.score(p, R8T[ci]), len(cm)])
                        rl.append([G.score(pl, R8T[ci]), len(cm)]); preds.append(p)
                    a = dict(R8_net=rn, R8_lin=rl)
                    if tn == JUDGED:
                        tb = time.time()
                        a['bytes'] = arm_bytes(F, L_pub, qa, msgs, msgn, kbytes, preds, arm != 'Qinf')
                        rec['timing'][f'bytes_{sd}_{arm}'] = time.time() - tb
                    D[arm] = a
                if tn == JUDGED:                                               # K0 (report), single agents, K0_sep threshold
                    for arm in K0_ARMS:
                        tab = Qh[arm]['table_np']
                        own = [E.stalk(L, tab) for L in L_k0]
                        lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', opts['chunk'])
                        rows = []
                        for a_ in range(n):
                            ki = own[a_][1].astype(np.int32); p = E.net_readout(lk[a_], ki, jk)
                            assert np.array_equal(p >= 0, L_k0[a_] != UNK)
                            rows.append(G.score(p, K0T[a_]))
                        D[arm]['K0_net'] = rows
                        if arm == 'Qinf' and L_pub is L_k0:
                            D['_own_logits_qinf'] = lk
                if tn in QPOST_TIERS and qpost is not None:
                    D['QPOST4'] = qpost_arm(F, L_pub, L_k0, Qh, model, D.pop('_own_logits_qinf', None), qpost['checkpoints'][sd], jk, R8T, H,
                                            r8_known, kcs, kbytes if tn == JUDGED else None, rstats, opts['chunk'], tn == JUDGED)
                D.pop('_own_logits_qinf', None)
                T[sd] = D
            rec['tiers'][tn] = T; rec['timing'][tn] = time.time() - ta
        rec['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-3000:]
    return rec


def level_relay(L_pub, F, labels, values, msgs, msgn):
    """relay the per-level count maps (the level-count messages) and assert they reproduce the (S, k) messages exactly"""
    q = len(values)
    own = [onehot_levels(L, labels, q) for L in L_pub]
    lm, _ = G.relay(own, F, G.comb_sum)
    v = np.asarray(values, np.int64)
    for e in F['directed']:
        Nm, Km = lm[e]
        assert np.array_equal(np.tensordot(v, Nm.astype(np.int64), axes=1), msgs[e][0].astype(np.int64)), 'level counts do not reproduce S'
        assert np.array_equal(Nm.sum(0), msgn[e][0]) and np.array_equal(Km, msgs[e][1]), 'level counts do not reproduce k'
    return lm, own


def arm_bytes(F, L_pub, qa, msgs, msgn, kbytes, preds, quantised):
    bS = {e: G.int_ctx_bytes(*msgs[e]) for e in F['directed']}
    sk = {e: bS[e] + kbytes[e] for e in F['directed']}
    out = dict(Sk=float(sum(sk.values())), S_only=float(sum(bS.values())), k_only=float(sum(kbytes.values())), edges={ekey(e): [bS[e], kbytes[e]] for e in F['directed']})
    modes = dict(Sk=sk)
    if quantised:
        lm, own = level_relay(L_pub, F, qa['labels_np'], qa['values_np'], msgs, msgn)
        lv = {e: lev_bytes(lm[e][0]) for e in F['directed']}
        for e in F['directed']:
            out['edges'][ekey(e)].append(lv[e])
        out['lev'] = float(sum(lv.values())); modes['lev'] = lv
        out['B'] = min(out['Sk'], out['lev']) + FLAG_BYTES; out['mode'] = 'lev' if out['lev'] < out['Sk'] else 'Sk'
        out['B_edge_min'] = float(sum(min(sk[e], lv[e]) for e in F['directed'])) + FLAG_BYTES * len(F['directed'])
        own_sk = [G.int_ctx_bytes(*E.stalk(L, qa['table_np'])) for L in L_pub]
        own_lv = [lev_bytes(N) for N, _ in own]
        out['own'] = dict(Sk=float(sum(own_sk)), lev=float(sum(own_lv)), min_of_sums=float(min(sum(own_sk), sum(own_lv))) + FLAG_BYTES)
    else:
        out['B'] = out['Sk']; out['mode'] = 'Sk'; out['B_edge_min'] = out['Sk']
        out['own'] = dict(Sk=float(sum(G.int_ctx_bytes(*E.stalk(L, qa['table_np'])) for L in L_pub)))
    tern = [G.ctx_tern_bytes(p) for p in preds]
    out['tern_readout_by_comp'] = tern
    out['B_RO'] = b_ro(F, modes, tern, flag=quantised)
    return out


def qpost_arm(F, L_pub, L_k0, Qh, model, lk, thr, jk, R8T, H, r8_known, kcs, kbytes, rstats, chunk, judged):
    """report-only QPOST4: source-denoised (own Qinf stalk -> NET logit), 4 levels, exact sum, TRAIN hold-out theta"""
    tab = Qh['Qinf']['table_np']; n = len(L_pub); tk = E.bin_edge(jk)
    assert abs(tk - thr['t_k0']) < 1e-9
    own = [E.stalk(L, tab) for L in L_pub]
    if lk is None:
        lk = E.net_logits(model, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', chunk)
    levs = [np.where(own[a][1], qpost_levels(lk[a], tk), -1) for a in range(n)]
    qs = [(np.where(own[a][1], QPOST_VALS[np.clip(levs[a], 0, 3)], 0).astype(np.int32), own[a][1]) for a in range(n)]
    qm, qb = G.relay(qs, F, G.comb_sum, rstats)
    rows, rows0 = [], []
    for ci, cm in enumerate(F['comps']):
        Qs = sum(qs[a][0] for a in cm)
        for a in cm:
            assert np.array_equal(qb[a][0], Qs), 'QPOST4 relayed sum != component sum'
        kc = kcs[ci]
        p = np.where(kc > 0, (Qs >= int(thr['theta'])).astype(np.int8), np.int8(-1)).astype(np.int8); _known_assert(p, r8_known[ci], 'QPOST4')
        p0 = np.where(kc > 0, (Qs >= 0).astype(np.int8), np.int8(-1)).astype(np.int8)
        rows.append([G.score(p, R8T[ci]), len(cm)]); rows0.append([G.score(p0, R8T[ci]), len(cm)])
    out = dict(R8=rows, R8_rc0=rows0, theta=int(thr['theta']))
    if judged:
        msgn = {e: (sum(qs[a][1].astype(np.int32) for a in F['sides'][e]), qm[e][1]) for e in F['directed']}
        bS = {e: G.int_ctx_bytes(*qm[e]) for e in F['directed']}
        sk = {e: bS[e] + kbytes[e] for e in F['directed']}
        lm, _ = G.relay([(np.stack([(levs[a] == l) for l in range(4)]).astype(np.int32), own[a][1]) for a in range(n)], F, G.comb_sum)
        for e in F['directed']:
            assert np.array_equal(np.tensordot(QPOST_VALS.astype(np.int64), lm[e][0].astype(np.int64), axes=1), qm[e][0].astype(np.int64))
            assert np.array_equal(lm[e][0].sum(0), msgn[e][0])
        lv = {e: lev_bytes(lm[e][0]) for e in F['directed']}
        out['bytes'] = dict(Sk=float(sum(sk.values())), lev=float(sum(lv.values())), B=min(float(sum(sk.values())), float(sum(lv.values()))) + FLAG_BYTES,
                            own_Sk=float(sum(G.int_ctx_bytes(q, k) for q, k in qs)))
    return out


def _eval_and_save(job):
    rec = eval_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json")
    with open(p + '.tmp', 'w') as fh:
        json.dump(rec, fh)
    os.replace(p + '.tmp', p)
    return rec


def eval_cmd(args, json_dir, index_file):
    sha = file_sha256(os.path.abspath(__file__))
    seeds = [int(s) for s in args.seeds.split(',') if s]
    panels = [p for p in args.panels.split(',') if p]
    if args.smoke:
        assert not (set(seeds) & set(NEW_SEEDS)) or set(panels) <= {'trainho'}, 'smoke: no read-out on the new VAL / fresh seeds 3, 4'
    files = {f for f in args.files.split(',') if f} if args.files else None
    specs = []
    for d in [x for x in args.export.split(',') if x]:
        for p in sorted(glob.glob(os.path.join(d, 'records', '*.json'))):
            r = _jload(p)
            if r.get('panel') not in panels or int(r.get('seed')) not in seeds:
                continue
            if files is not None and r.get('file', f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.npz") not in files:
                continue
            specs.append(dict(r, export_dir=d))
    assert specs, 'no export records match'
    csha = file_sha256(args.criteria); cdoc = _jload(args.criteria)
    if not args.smoke:
        assert cdoc.get('script_sha256') == sha, 'criteria file was written by another script version: re-run quantize'
    for sd, d in CKPTS.items():                                         # review S1: checkpoints / thresholds = the criteria file's
        assert file_sha256(os.path.join(d['dir'], 'final.pt')) == cdoc['checkpoints'][sd]['final_pt_sha256'], f'{sd}: final.pt sha != criteria'
        assert file_sha256(os.path.join(d['dir'], 'threshold.json')) == cdoc['checkpoints'][sd]['threshold_sha256'], f'{sd}: threshold.json sha != criteria'
    if args.qpost:
        check_qpost_file(_jload(args.qpost), csha, args.smoke)
    elif not args.smoke:
        print('WARNING: no QPOST4 threshold file: the report-only QPOST4 arm is not run', flush=True)
    opts = dict(out=args.out, criteria=args.criteria, criteria_sha256=csha, qpost=args.qpost or None,
                qpost_sha256=file_sha256(args.qpost) if args.qpost else None, threads=args.threads, chunk=args.chunk, script_sha256=sha,
                tiers=[t for t in args.tiers.split(',') if t], smoke=bool(args.smoke))
    os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
    jobs = []
    for r in specs:
        tag = f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}"
        jp = os.path.join(args.out, 'jobs', tag + '.json')
        if r.get('failed') or r.get('skipped'):
            _jdump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                        skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))), jp)
            continue
        if os.path.exists(jp):
            o = _jload(jp)
            if not o.get('failed') and o.get('script_sha256') == sha and o.get('criteria_sha256') == opts['criteria_sha256'] and o.get('qpost_sha256') == opts['qpost_sha256']:
                continue
        jobs.append((dict(panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=int(r['seed']), file=r['file'], file_sha256=r['file_sha256'],
                          components=r['components'], e1_match=r.get('e1_match'), export_dir=r['export_dir']), opts))
    _jdump(dict(stage='eval', utc=utc(), script_sha256=sha, g4_e3r_sha256=file_sha256(E.__file__), noise_panels_sha256=file_sha256(G.__file__),
                calib_sha256=file_sha256(C.__file__), opts={k: v for k, v in opts.items()}, seeds=seeds, panels=panels, n_jobs=len(jobs), smoke=bool(args.smoke)),
           os.path.join(args.out, 'eval_run.json'))
    print(f'eval: {len(jobs)} jobs, seeds {seeds}, panels {panels}, {args.workers} workers x {args.threads} threads, qpost {bool(args.qpost)}', flush=True)
    t0 = time.time()
    if args.workers <= 1:
        it = ((i, _eval_and_save(j)) for i, j in enumerate(jobs, 1))
    else:
        it = E._pool(args.workers, _eval_and_save, jobs, json_dir, index_file)
    for i, rec in it:
        print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else 'ok'} "
              f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)


# ================================================================ aggregate and verdict
def _exp(rows):
    return G._expand(rows)


def _m(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def _cell_stats(R, sd, tier, arm, key='R8_net'):
    """R: job records of one panel / seed. Returns the E3-R headline means over expanded rows, per-house means and bytes"""
    rows, per_house = [], {}
    for r in R:
        a = r['tiers'][tier][sd][arm]
        x = _exp(a[key]) if key in a else []
        rows += x; per_house[f"r{r['rooms']}_i{r['index']}"] = _m(x, 'furn_f1')
    s = dict(furn=_m(rows, 'furn_f1'), free=_m(rows, 'free_recall'), surface=_m(rows, 'surface_f1'), fpr=_m(rows, 'unknown_known_fpr'),
             furn_tol1=_m(rows, 'furn_f1_tol1'), furn_pooled=G._pooled_f1(rows), per_house=per_house)
    return s


def _cell(R, sd, tier, arm, ctrl='Qinf', judged=True):
    a = _cell_stats(R, sd, tier, arm); c = _cell_stats(R, sd, tier, ctrl)
    dh = {h: a['per_house'][h] - c['per_house'][h] for h in a['per_house'] if a['per_house'][h] is not None and c['per_house'][h] is not None}
    wh = min(dh, key=dh.get) if dh else None
    out = dict(arm_furn=a['furn'], ctrl_furn=c['furn'], d_furn=a['furn'] - c['furn'], worst_house=wh, d_worst_house=dh.get(wh),
               d_free=a['free'] - c['free'], d_surface=a['surface'] - c['surface'], d_fpr=a['fpr'] - c['fpr'], d_furn_tol1=a['furn_tol1'] - c['furn_tol1'],
               d_furn_pooled=a['furn_pooled'] - c['furn_pooled'], arm_lin_furn=_cell_stats(R, sd, tier, arm, 'R8_lin')['furn'],
               ctrl_lin_furn=_cell_stats(R, sd, tier, ctrl, 'R8_lin')['furn'])
    if tier == JUDGED and all('bytes' in r['tiers'][tier][sd][arm] for r in R):
        Ba = [r['tiers'][tier][sd][arm]['bytes']['B'] for r in R]; Bc = [r['tiers'][tier][sd][ctrl]['bytes']['B'] for r in R]
        out.update(B_med=float(np.median(Ba)), B_ctrl_med=float(np.median(Bc)), ratio=float(np.median(Ba) / np.median(Bc)),
                   ratio_med_per_house=float(np.median(np.asarray(Ba) / np.asarray(Bc))),
                   modes={m: sum(r['tiers'][tier][sd][arm]['bytes']['mode'] == m for r in R) for m in ('Sk', 'lev')},
                   Sk_med=float(np.median([r['tiers'][tier][sd][arm]['bytes']['Sk'] for r in R])),
                   lev_med=float(np.median([r['tiers'][tier][sd][arm]['bytes']['lev'] for r in R])) if arm != ctrl else None,
                   B_edge_min_ratio=float(np.median([r['tiers'][tier][sd][arm]['bytes']['B_edge_min'] for r in R]) / np.median(Bc)),
                   B_RO_med=float(np.median([r['tiers'][tier][sd][arm]['bytes']['B_RO'] for r in R])),
                   B_RO_ratio=float(np.median([r['tiers'][tier][sd][arm]['bytes']['B_RO'] for r in R]) / np.median(Bc)),
                   own_med={k: float(np.median([r['tiers'][tier][sd][arm]['bytes']['own'][k] for r in R])) for k in R[0]['tiers'][tier][sd][arm]['bytes']['own']})
        if all('mask_dup_bytes' in r for r in R):                       # review S2, report only: the known mask coded once
            dup = [r['mask_dup_bytes'] for r in R]
            Bc1 = [b - d for b, d in zip(Bc, dup)]
            Ba1 = Bc1 if arm == ctrl else [min(r['tiers'][tier][sd][arm]['bytes']['Sk'] - d, r['tiers'][tier][sd][arm]['bytes']['lev']) + FLAG_BYTES
                                           for r, d in zip(R, dup)]
            out.update(mask_dup_med=float(np.median(dup)), B_ctrl_mask_once_med=float(np.median(Bc1)), ratio_mask_once=float(np.median(Ba1) / np.median(Bc1)))
    if tier == JUDGED and all('K0_net' in r['tiers'][tier][sd][arm] for r in R) and all('K0_net' in r['tiers'][tier][sd][ctrl] for r in R):
        ka = _m([x for r in R for x in r['tiers'][tier][sd][arm]['K0_net']], 'furn_f1'); kc = _m([x for r in R for x in r['tiers'][tier][sd][ctrl]['K0_net']], 'furn_f1')
        out.update(K0_arm=ka, K0_ctrl=kc, d_K0=ka - kc)
    return out


def _qpost_cell(R, sd, tier):
    if not all('QPOST4' in r['tiers'][tier][sd] for r in R):
        return None
    rows = [x for r in R for x in _exp(r['tiers'][tier][sd]['QPOST4']['R8'])]; rows0 = [x for r in R for x in _exp(r['tiers'][tier][sd]['QPOST4']['R8_rc0'])]
    c = _cell_stats(R, sd, tier, 'Qinf')
    out = dict(furn=_m(rows, 'furn_f1'), d_furn=_m(rows, 'furn_f1') - c['furn'], free=_m(rows, 'free_recall'), d_free=_m(rows, 'free_recall') - c['free'],
               rc0_furn=_m(rows0, 'furn_f1'), d_rc0_furn=_m(rows0, 'furn_f1') - c['furn'], theta=R[0]['tiers'][tier][sd]['QPOST4']['theta'])
    if tier == JUDGED and all('bytes' in r['tiers'][tier][sd]['QPOST4'] for r in R):
        Bc = [r['tiers'][tier][sd]['Qinf']['bytes']['B'] for r in R]
        out.update(B_med=float(np.median([r['tiers'][tier][sd]['QPOST4']['bytes']['B'] for r in R])),
                   ratio=float(np.median([r['tiers'][tier][sd]['QPOST4']['bytes']['B'] for r in R]) / np.median(Bc)))
        if all('mask_dup_bytes' in r for r in R):
            Ba1 = [min(r['tiers'][tier][sd]['QPOST4']['bytes']['Sk'] - r['mask_dup_bytes'], r['tiers'][tier][sd]['QPOST4']['bytes']['lev']) + FLAG_BYTES for r in R]
            out['ratio_mask_once'] = float(np.median(Ba1) / np.median([b - r['mask_dup_bytes'] for b, r in zip(Bc, R)]))
    return out


def _arm_verdict(cells, C_):
    if not cells:
        return dict(verdict='INCOMPLETE', reasons=['no cells'])
    undef = sorted(k for k, c in cells.items() if any(c.get(x) is None for x in ('d_furn', 'd_worst_house', 'd_free', 'd_surface', 'ratio')))
    if undef:                                                           # review S7: never a TypeError, never a silent pass
        return dict(verdict='INCOMPLETE', reasons=[f'cells {undef}: furniture delta / worst house / guards / byte ratio undefined'])
    acc = all(c['d_furn'] >= C_['acc_mean_min'] - 1e-12 and c['d_worst_house'] >= C_['acc_worst_house_min'] - 1e-12 for c in cells.values())
    guard = all(c['d_free'] >= C_['guard_free_min'] - 1e-12 and c['d_surface'] >= C_['guard_surface_min'] - 1e-12 and c['clean_furn'] is not None
                and c['clean_furn'] >= C_['clean_furn_min'] for c in cells.values())
    byt = all(c['ratio'] <= C_['bytes_lever_max'] + 1e-12 for c in cells.values())
    amb = all(c['ratio'] <= C_['bytes_ambiguous_max'] + 1e-12 for c in cells.values())
    reasons = []
    if not acc:
        reasons.append('(1) accuracy')
    if not guard:
        reasons.append('(2) guards')
    if not byt:
        reasons.append('(3) bytes > 0.5' + ('' if amb else ' and > 0.75'))
    v = 'LEVER' if (acc and guard and byt) else ('AMBIGUOUS' if (acc and guard and amb) else 'NO')
    return dict(verdict=v, acc=acc, guards=guard, bytes=byt, bytes_le_075=amb, reasons=reasons)


def final_decision(av, cells, fresh, C_):
    """batch verdict from the two arms' VAL verdicts (review S3) and the fresh rule (review M1, D8-52 补充): only a LEVER can be
    downgraded (to AMBIGUOUS, fresh 4-cell mean d furniture < fresh_mean_min); NO / AMBIGUOUS never change; the literal sign comparison
    is reported only"""
    assert C_.get('fresh_rule') == 'noninf_mean_downgrade_only', f"unknown fresh rule {C_.get('fresh_rule')!r}"
    if av['Q3']['verdict'] == 'LEVER':
        arm, note = 'Q3', 'main arm'
    elif av['Q4']['verdict'] == 'LEVER':
        arm, note = 'Q4', 'FALLBACK (Q3 did not pass)'
    else:
        arm, note = 'Q3', 'main arm (neither arm is LEVER: the batch verdict is Q3s; Q4 listed)'
    out = dict(verdict=av[arm]['verdict'], arm=arm, note=note, q3_verdict=av['Q3']['verdict'], q4_verdict=av['Q4']['verdict'])
    vc, fc = cells.get(arm) or {}, fresh.get(arm) or {}
    if vc and fc and all(c.get('d_furn') is not None for c in list(vc.values()) + list(fc.values())):
        mv = float(np.mean([c['d_furn'] for c in vc.values()])); mf = float(np.mean([c['d_furn'] for c in fc.values()]))
        out.update(val_mean_d_furn=mv, fresh_mean_d_furn=mf, fresh_mean_ok=bool(mf >= C_['fresh_mean_min'] - 1e-12),
                   fresh_sign_opposite=bool(mv * mf < 0), fresh_cells_d_furn={k: c['d_furn'] for k, c in fc.items()},
                   fresh_worst_house={k: [c.get('worst_house'), c.get('d_worst_house')] for k, c in fc.items()})
        if out['verdict'] == 'LEVER' and not out['fresh_mean_ok']:
            out['verdict'] = 'AMBIGUOUS'
            out['note'] += f"; fresh 4-cell mean d furniture {100 * mf:+.2f} pp < {100 * C_['fresh_mean_min']:+.1f} pp: LEVER downgraded (D8-52 补充)"
    elif out['verdict'] == 'LEVER':
        out['verdict'] = 'INCOMPLETE'; out['note'] += '; fresh cells missing'
    return out


def aggregate(out_dir, seeds, cont_seeds, smoke=False, criteria_path=None):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(utc=utc(), smoke=bool(smoke), failed=[], skipped=[], script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             criteria_sha256=sorted({r.get('criteria_sha256') for r in recs if r.get('criteria_sha256')}),
             qpost_sha256=sorted({str(r.get('qpost_sha256')) for r in recs if not r.get('failed')}), panels={}, timing={}, warnings=[])
    reasons = []
    ran = [r for r in recs if r.get('script_sha256')]
    if smoke and any(not r.get('smoke') for r in ran):
        raise SystemExit('aggregate --smoke refused: the job files are not from a smoke eval (review S5)')
    if not smoke and any(r.get('smoke') for r in ran):
        reasons.append('smoke job files present')
    if len(S['script_sha256']) > 1 or len(S['criteria_sha256']) > 1:
        reasons.append('mixed script / criteria shas in the job files')
    if len(S['qpost_sha256']) > 1:
        S['warnings'].append(f"mixed QPOST4 threshold shas in the job files {S['qpost_sha256']}")
    cp = criteria_path or os.path.join(QUANT_DIR, 'criteria_quantizer.json')
    C_ = None
    if not os.path.exists(cp):
        reasons.append(f'criteria file {cp} missing')
    else:                                                               # review M2a: no fall-back to built-in margins
        cdoc = _jload(cp); csha = file_sha256(cp); C_ = dict(cdoc['criteria'])
        S['criteria_file'] = dict(path=os.path.relpath(cp, ROOT), sha256=csha, script_sha256=cdoc.get('script_sha256'))
        if S['criteria_sha256'] != [csha]:
            reasons.append(f"job criteria sha {[s[:8] for s in S['criteria_sha256']]} != criteria file {csha[:8]}")
        if S['script_sha256'] and S['script_sha256'] != [cdoc.get('script_sha256')]:
            reasons.append(f"job script sha {[s[:8] for s in S['script_sha256']]} != the criteria file's script sha {str(cdoc.get('script_sha256'))[:8]}")
    good = []
    for r in recs:
        tag = f"{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}_s{r.get('seed')}"
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append(tag)
        else:
            good.append(r)
    secs = [r['seconds'] for r in good if r.get('seconds')]
    S['timing'] = dict(n=len(secs), mean_s=float(np.mean(secs)) if secs else None, max_s=float(np.max(secs)) if secs else None)
    if S['failed']:
        reasons.append(f"failed jobs {S['failed']}")
    for panel in ('val', 'fresh', 'trainho'):
        P = {}
        for seed in sorted({int(r['seed']) for r in good if r['panel'] == panel}):
            R = sorted([r for r in good if r['panel'] == panel and int(r['seed']) == seed], key=lambda r: (r['rooms'], r['index']))
            D = dict(houses=[f"r{r['rooms']}_i{r['index']}" for r in R], cells={})
            for sd in CKPTS:
                cs = {}
                for tier in TIERS:
                    if not all(tier in r['tiers'] for r in R):
                        continue
                    cs[tier] = {arm: _cell(R, sd, tier, arm) for arm in ARMS if arm != 'Qinf'}
                    cs[tier]['Qinf_abs'] = _cell_stats(R, sd, tier, 'Qinf'); cs[tier]['Qinf_abs'].pop('per_house')
                    if tier in QPOST_TIERS:
                        cs[tier]['QPOST4'] = _qpost_cell(R, sd, tier)
                D['cells'][sd] = cs
            P[str(seed)] = D
        if P:
            S['panels'][panel] = P
    for pn, P in S['panels'].items():                                   # review S6: QPOST4 is report-only, but D8-52 lists it
        for seed, D in P.items():
            for sd, cs in D['cells'].items():
                for tier in QPOST_TIERS:
                    if tier in cs and not cs[tier].get('QPOST4'):
                        S['warnings'].append(f'QPOST4 missing: {pn} seed {seed} {sd} {tier}')
    for w in S['warnings']:
        print('WARNING:', w, flush=True)
    notes = dict(B_RO='report only: uplink sums + downlink ternary read-out at the best root; NOT within the floor (violates C5)',
                 QPOST4='report only: prior at the source (R-C definition), theta_post from TRAIN hold-out only',
                 bytes_ratio='criterion (3) = median_houses B(arm) / median_houses B(Qinf); ratio_med_per_house reported',
                 ratio_mask_once='report only (review S2): the known mask coded once (Qinf S+k and the arm Sk mode minus the k ternary pass)')
    if C_ is None:
        S['decision'] = dict(final=dict(verdict='INCOMPLETE', reasons=reasons, smoke=smoke), notes=notes)
        return S
    # verdict
    cells = {arm: {} for arm in ('Q3', 'Q4')}; fresh = {arm: {} for arm in ('Q3', 'Q4')}
    for pn, target in ((C_['judged_panel'], cells), (C_['report_panel'], fresh)):
        P = S['panels'].get(pn, {})
        for seed in seeds:
            D = P.get(str(seed))
            if D is None:
                reasons.append(f'{pn}: noise seed {seed} missing'); continue
            if not smoke:
                exp = {f'r{r}_i{i}' for r, i in G.panel_specs(pn)}
                miss = exp - set(D['houses'])
                if miss:
                    reasons.append(f'{pn} seed {seed}: {len(miss)} of {len(exp)} houses missing')
            for sd in CKPTS:
                cs = D['cells'].get(sd, {})
                if JUDGED not in cs or 'clean' not in cs:
                    reasons.append(f'{pn} seed {seed} {sd}: tier missing'); continue
                for arm in ('Q3', 'Q4'):
                    c = dict(cs[JUDGED][arm]); c['clean_furn'] = cs['clean'][arm]['arm_furn']
                    if abs(c['d_fpr']) > 1e-12:
                        reasons.append(f'{pn} seed {seed} {sd} {arm}: FPR differs ({c["d_fpr"]:+.2e})')
                    target[arm][f'{sd}|{seed}'] = c
    av = {arm: _arm_verdict(cells[arm], C_) for arm in ('Q3', 'Q4')}
    final = final_decision(av, cells, fresh, C_)
    final.update(reasons=reasons, smoke=smoke)
    if smoke:
        final['verdict'] = 'SMOKE ' + final['verdict'] + (' (would be INCOMPLETE: see reasons)' if reasons else '')
    elif reasons:
        final['verdict_without_incomplete'] = final['verdict']; final['verdict'] = 'INCOMPLETE'
    S['decision'] = dict(final=final, arms=av, cells=cells, fresh_cells=fresh, criteria=C_, seeds=seeds, continuity_seeds=cont_seeds, notes=notes)
    return S


def _fmt_cell(c):
    s = (f"d furn {100 * c['d_furn']:+.2f} pp (arm {c['arm_furn']:.4f} / Qinf {c['ctrl_furn']:.4f}), worst house {c['worst_house']} {100 * c['d_worst_house']:+.2f} pp, "
         f"d free {100 * c['d_free']:+.2f}, d surf {100 * c['d_surface']:+.2f}, d FPR {c['d_fpr']:+.1e}, LIN {c['arm_lin_furn']:.4f}/{c['ctrl_lin_furn']:.4f}")
    if 'ratio' in c:
        s += (f" | B {c['B_med'] / 1e3:.0f} KB / Qinf {c['B_ctrl_med'] / 1e3:.0f} KB = {c['ratio']:.3f} (per-house med {c['ratio_med_per_house']:.3f}; Sk {c['Sk_med'] / 1e3:.0f}, "
              f"lev {(c['lev_med'] or 0) / 1e3:.0f}; modes {c['modes']})"
              + (f" [mask coded once, report only: {c['ratio_mask_once']:.3f}]" if 'ratio_mask_once' in c else '')
              + f" | B_RO {c['B_RO_ratio']:.3f} [violates C5, report only] | own {json.dumps({k: round(v / 1e3, 1) for k, v in c['own_med'].items()})} KB")
    if 'd_K0' in c:
        s += f" | K0 {c['K0_arm']:.4f} (d {100 * c['d_K0']:+.2f} pp)"
    if 'clean_furn' in c:
        s += f" | clean {c['clean_furn']:.4f}"
    return s


def aggregate_cmd(args):
    seeds = [int(s) for s in args.seeds.split(',') if s]; cont = [int(s) for s in args.continuity.split(',') if s]
    if not args.smoke:                                                  # review M2b: verdicts only on the new seeds
        if sorted(seeds) != sorted(NEW_SEEDS) or not set(cont) <= set(CONT_SEEDS):
            raise SystemExit(f'aggregate refused: the verdict is on --seeds {",".join(map(str, NEW_SEEDS))} only; seeds '
                             f'{",".join(map(str, CONT_SEEDS))} only as --continuity (got --seeds {seeds} --continuity {cont})')
    S = aggregate(args.out, seeds, cont, smoke=args.smoke, criteria_path=args.criteria or None)
    _jdump(S, os.path.join(args.out, 'summary.json'))
    dec = S['decision']; fin = dec['final']
    print(f"G4-E5 verdict: {fin['verdict']}  arm {fin.get('arm')} ({fin.get('note')})  Q3 {fin.get('q3_verdict')} / Q4 {fin.get('q4_verdict')}  "
          f"reasons {fin.get('reasons')}", flush=True)
    if 'fresh_mean_d_furn' in fin:
        print(f"  fresh rule (D8-52 补充): fresh 4-cell mean d {100 * fin['fresh_mean_d_furn']:+.2f} pp (>= {100 * dec['criteria']['fresh_mean_min']:+.1f}: "
              f"{fin['fresh_mean_ok']}); VAL mean {100 * fin['val_mean_d_furn']:+.2f} pp; literal sign opposite (report only): {fin['fresh_sign_opposite']}", flush=True)
    if 'arms' not in dec:
        return S
    for arm in ('Q3', 'Q4'):
        print(f"  {arm}: {json.dumps(dec['arms'][arm])}", flush=True)
        for k, c in dec['cells'][arm].items():
            print(f'    VAL {k}: {_fmt_cell(c)}', flush=True)
        for k, c in dec['fresh_cells'][arm].items():
            print(f'    fresh {k}: {_fmt_cell(c)}', flush=True)
    for pn, P in S['panels'].items():
        for seed, D in P.items():
            tag = 'judged seed, all tiers / report rows' if int(seed) in seeds else ('continuity' if int(seed) in cont else 'report')
            print(f'  [{tag}] {pn} seed {seed} ({len(D["houses"])} houses)', flush=True)
            for sd, cs in D['cells'].items():
                for tier in TIERS:
                    if tier not in cs:
                        continue
                    print(f"    {sd} {tier}: Qinf NET {cs[tier]['Qinf_abs']['furn']:.4f}", flush=True)
                    for arm in ('Q3', 'Q4', 'VOTE'):
                        print(f'      {arm}: {_fmt_cell(cs[tier][arm])}', flush=True)
                    if cs[tier].get('QPOST4'):
                        print(f"      QPOST4 [prior at the source, report only]: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in cs[tier]['QPOST4'].items()})}", flush=True)
    print(f"  timing: {json.dumps(S['timing'])}", flush=True)
    return S


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('quantize', 'export', 'check-export', 'qpost-select', 'eval', 'aggregate'):
        p = sub.add_parser(name)
        p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=2); p.add_argument('--smoke', action='store_true')
        if name == 'quantize':
            p.add_argument('--out', default=os.path.join(QUANT_DIR, 'criteria_quantizer.json'))
        if name == 'export':
            p.add_argument('--out', required=True); p.add_argument('--panels', default='val,fresh'); p.add_argument('--seeds', default='')
            p.add_argument('--houses', type=int, default=0); p.add_argument('--only', default=''); p.add_argument('--train-holdout', action='store_true')
        if name == 'check-export':
            p.add_argument('--npz', required=True); p.add_argument('--record', default='')
        if name == 'qpost-select':
            p.add_argument('--export', required=True); p.add_argument('--criteria', required=True); p.add_argument('--out', required=True)
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--criteria', required=True)
            p.add_argument('--qpost', default=''); p.add_argument('--panels', default='val,fresh'); p.add_argument('--seeds', default='3,4')
            p.add_argument('--files', default=''); p.add_argument('--tiers', default=','.join(TIERS)); p.add_argument('--chunk', type=int, default=2)
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--seeds', default='3,4'); p.add_argument('--continuity', default='1,2')
            p.add_argument('--criteria', default=os.path.join(QUANT_DIR, 'criteria_quantizer.json'))
    args = ap.parse_args(argv)
    json_dir = E._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = E._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    assert args.workers <= 12, 'at most 12 processes'
    if args.cmd == 'quantize':
        return quantize_cmd(args)
    if args.cmd == 'export':
        return export_cmd(args, json_dir, index_file)
    if args.cmd == 'check-export':
        return check_export_cmd(args)
    if args.cmd == 'qpost-select':
        return qpost_select_cmd(args, json_dir, index_file)
    if args.cmd == 'eval':
        return eval_cmd(args, json_dir, index_file)
    if args.cmd == 'aggregate':
        return aggregate_cmd(args)


if __name__ == '__main__':
    main()
