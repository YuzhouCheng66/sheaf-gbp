"""g4_exact_codec.py (Claude, 2026-10-27): batch G4-E0 "exact-code final" of report D8-36 (pre-registered 2026-10-27 02:20 UTC,
claude_handoff/OVERNIGHT_REPORT_20261021.md). CPU only, no learning, no GPU (CUDA_VISIBLE_DEVICES is blanked at import).

Single variable: the WIRE LANGUAGE (three arms), each in two tiers:
  W1  hit set + float32 poses (= A07 'ELS'): binary code of the observed-occupied voxel indicator, 9-voxel causal context
      (rate_baselines NB), coded in anti-diagonal (x+y+z) wavefront order so the context is causal; only voxels within the sensor
      reach of the message's poses are coded (public after the poses are decoded).
  W2  ray-order events + float32 poses (= A01 code I): the receiver marches the public rays (step-major, the 3 poses of an agent
      interleaved, agents in id order); every time a ray enters an UNDECIDED voxel one binary event is decoded (occupied -> the ray
      stops). Context = A01's 6 face states x occupied-edge count x distance band (6,561) in strict event order (step, pose, ray):
      the part decided before the batch (one step of one pose) is computed vectorised, the earlier events of the same batch enter
      through a dependency list (decoder: one tight sequential loop per batch).
  W3  S0 = plan + consistent solid boxes + residual + float32 poses (= A03 p8/p9 label-free S0): global floor/ceiling layers
      (kf, kc) and a lintel start lz; a 128 x 128 ternary plan (0 / wall column / lintel column, 8-neighbour 2-D context, coded in
      2x+y wavefronts); greedy solid boxes (A03 box_cover) covering every remaining observed hit without entering an observed free
      voxel (A03 'joint' token order: dpx, dpy | yabs, z0, dz | z0, dx, dy | bucket(dx)). Consistency is guaranteed by construction
      (painted voxels never meet an observed free voxel), so the boxes explain every hit; the residual (see below) carries the rest.
Tiers:
  u   unconditional: the message is the sender's full content (exclude-back message / own map), decodable on its own.
  c   conditioned on what the receiver already holds IN COMMON with the sender (the edge's shared knowledge S = everything exchanged
      on that edge so far = cavity / innovation): agents already in S are not re-sent, voxels known in S are not coded (W1), are
      pre-decided (W2) or are joined into the painted world (W3), and S features enter the contexts.
Every message, all wires: [varint payload length] + range-coded payload = 8-bit agent mask + 6 float32 (x, y of the 3 poses) per new
agent (z = W.FLOOR_T + W.SENSOR_HEIGHT is a protocol constant) + wire body + RESIDUAL list (voxel index + ternary value, terminated by
a stop flag). The encoder runs the decoder's own reconstruction and lists every voxel that still differs from the target, so the
decoded map is exact by construction; the residual is non-empty only where the float32-pose march differs from the generator's
float64 march (A10 saw 1 voxel in 96 maps). The residual is ordered by voxel index, not by ray order (it is (almost) always empty).
Decoding: every wire ends with W.observe (public rays fibonacci_sphere(BEAMS), poses rebuilt from the decoded float32 values) of the
decoded occupancy world from each new agent's poses, joined (max) on the voxel grid with S. W.observe calls are memoised exactly:
a cached render of pose P on world W'' is reused for world W' iff W'[K] == W''[K] on its visited set K (each ray then meets the same
values along its path and stops at the same sample), checked on every call; --no-render-cache disables it.
Entropy coding: a real range coder (48-bit window, carry propagation, Python ints), integer frequency tables (total 2^20, every
symbol >= 1). Priors = per-context count tables fitted ONLY on a TRAIN half (A = even, B = odd index of scene_plan('TRAIN', 70)) by
running the same encoders on those houses; they are the shared OFFLINE model of both ends and are NOT counted in message bytes.
Header-like fields (mask, poses, kf/kc/lz, box count, residual) use fixed uniform / fixed models.
Protocols (one fusion; 8 agents on the Kruskal radio forest):
  K0    every agent's own map, unconditional (what K0 decodes from the agent's own code).
  CD    collect-distribute on the forest, each directed edge once (root = smallest id per component); tiers u and c (c: a distribute
        message is conditioned on the collect message that came the other way).
  SYNC  8 synchronous rounds. u: every round every directed edge re-sends its full exclude-back message (rollout_max schedule);
        c: every round every directed edge sends only the innovation of the sender's belief over the edge's shared knowledge.
  CAV   rate readout of the converged message m_(u->v) (sender-side union) conditioned on the cavity m_(v->u) (receiver side), every
        directed edge (the "conditioned converged message" of criterion (ii)).
Assertions (criterion (i)): every message decodes voxel-exactly to its target (S v content), decoded poses equal the float32 poses,
SYNC beliefs of every agent after rounds 0/1/2/4/8 equal the stream targets (max over the reached own maps, forest_reached), CD final
beliefs equal the R8 targets; free recall = 1 and unknown->known FPR = 0 are computed on every check. Any inequality, exception or
march/render disagreement makes the batch INCONCLUSIVE (bug) -- it never passes silently.
Panels: VAL = scene_plan('VALIDATION', 16) with index < 8, fresh = index 8-15; scenes whose 8 agents cannot be placed (make_fixture
ValueError) are skipped deterministically. TEST is never generated. The 6 demo houses (demo_web3d/data) are rebuilt and compared voxel
for voxel before anything else.
Operational definitions used by the automatic verdict (D8-36 wording -> this code):
  own map = median K0 message bytes (tier u, decodable from the own code alone); conditioned converged message = median CAV row
  (m(u->v) conditioned on the cavity m(v->u), every directed edge; a rate readout, not a protocol); one fusion per house = total CD
  tier-c bytes (every directed edge once); 6.5 KB = 6,500 B; best arm = wire with the lowest sum over the panel x prior cells of the
  median per-house CD-c bytes; (iii) = |A - B| / min(A, B) <= 5 % for each of the three medians in each panel, best arm.
  The judged cells are FIXED to {VAL, fresh} x {A, B} (review M1). Reported next to the literal reading (review M2): the A03-definition
  reading (unconditioned converged message CD-u vs 420 B, CD-u per house vs 6,500 B -- the measurements the thresholds were set on),
  the realisable conditioned distribute-phase messages, and the result if the arm were fixed to W3.
  INCONCLUSIVE if any message / belief check is not voxel-exact, any decode raises, W2 march and W.observe disagree, any worker
  crashes (TRAIN or eval), the demo houses do not rebuild, the workers ran a different file sha than the main process, or a HouseExpo
  house appears in two of {TRAIN A, TRAIN B, VAL + fresh}. INCOMPLETE (not a verdict) if a panel has no evaluated house, a cell is
  missing, the demo check was skipped, a --*-limit / --wires subset / --fault was used. Skipped houses per panel are in the verdict.
Known limitations (for the reviewer):
  1. The W3 "ray-order residual" is a voxel-index-ordered list (index 19 bit + ternary value); with the consistent covers it is empty
     unless the float32-pose march differs from the float64 one, so its order has no measurable byte effect.
  2. W.observe is memoised (exact by the visited-set argument, validated on every call) and W2 marches replay a recorded ray
     traversal (bit-identical, tested also when rays run past their recording); --no-render-cache disables both for audits.
  3. Priors are static count tables (KT for binary / ternary, total pseudo-count 8 for large alphabets, 2^20 integer totals), pooled
     over all message types of a TRAIN half, fitted by running the encoders under the exactness that criterion (i) then verifies;
     no adaptation inside a message. W1 / W2 / W3 contexts follow A07 / A01 / A03; W2's distance band is the sample distance s * ds
     (A01: voxel-centre distance to the agent centre); W1 codes only voxels within the poses' sensor reach.
  4. W3's label-free split uses generator-aware constants inherited from A03 p8 (structure band z 23-24, lintel free top <= 21).
  5. Poses cost 6 raw float32 = 24 B per agent per message (not entropy-coded); the sensor height is a protocol constant.
  6. SYNC tier u re-sends the full exclude-back message every round (rollout_max schedule); an empty tier-c slot costs 1 B (length).
  7. Cost: W2 dominates (sequential event decoding in Python); smoke timings are in run.log, the formal run is not tuned further.
Usage (from the project root, CPU):
  smoke : D:/anaconda3/envs/jax-gpu/python.exe -B scripts/claude_v342/g4_exact_codec.py --out claude_handoff/analysis_20261027/g4/e0_smoke
          --val-limit 2 --fresh-limit 0 --train-limit 10 --procs 6
  full  : D:/anaconda3/envs/jax-gpu/python.exe -B scripts/claude_v342/g4_exact_codec.py --out claude_handoff/analysis_20261027/g4/e0 --procs 6
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_JSON = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'data', 'houseexpo_json')
INDEX_JSON = os.path.join(ROOT, 'claude_handoff', 'claude_reviews_20261021', 'houseexpo_index_v365.json')
DEMO_DIR = os.path.join(ROOT, 'demo_web3d', 'data')
os.environ['HOUSEEXPO_JSON_DIR'] = DATA_JSON
os.environ['HOUSEEXPO_INDEX'] = INDEX_JSON
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
sys.dont_write_bytecode = True
for _p in (HERE, os.path.join(ROOT, 'exploration_grid'), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, base64, bisect, gzip, hashlib, json, math, time, traceback  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
import mapping_3d_storey_data_v420 as W  # noqa: E402

VERSION = 'g4-e0-1'
CODE_SHA = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()   # stamped into every worker result
DIMS = tuple(W.DIMS['R1']); NX, NY, NZ = DIMS; V = NX * NY * NZ; YZ = NY * NZ
RAYS = W.fibonacci_sphere(W.BEAMS)
Z_SENSOR = W.FLOOR_T + W.SENSOR_HEIGHT
HOPS = tuple(int(h) for h in W.HOPS)
WIRES = ('W1', 'W2', 'W3'); TIERS = ('u', 'c')
SYNC_ROUNDS = 8
TOT_BITS = 20; TOT = 1 << TOT_BITS
COMPS = ('length', 'header', 'pose', 'plan', 'box', 'hit', 'event', 'residual', 'coder')
THRESH = dict(own=250.0, cav=420.0, house=6500.0, halves_rel=0.05)
CONTROL_3SX = dict(source='OVERNIGHT_REPORT_20261021.md D8-33 (quoted, not re-run)', bytes_per_message=2048, rounds=8,
                   r8_surface_f1_VAL_D8_33=[0.9075, 0.9100], r8_furniture_f1_fresh_D8_33=[0.5807, 0.5878], seeds=[342101, 342202],
                   note='3S-X checkpoints runs/v429-stack/c2-s* + calibrated rule; 8 rounds x every directed edge x 2,048 B; '
                        'furniture .58/.59 is the D8-33 table row (fresh panel), as quoted by the coordinator')


# ============================================================================ range coder (48-bit window, carry propagation)
_BOT = 1 << 40
_TOP = 1 << 48
_M40 = (1 << 40) - 1


class RangeEncoder:
    """Range coder over Python ints: low is a 48-bit window (a carry ripples into the bytes already written), range in [2^40, 2^48].
    A symbol with cumulative frequency c, frequency f out of t narrows [low, low + range) to [low + r c, low + r c + r f), r = range // t."""

    def __init__(self):
        self.low = 0; self.rng = _TOP; self.out = bytearray()

    def encode(self, cum, freq, tot):
        low, rng, out = self.low, self.rng, self.out
        for c, f in zip(cum, freq):
            r = rng // tot
            low += r * c
            rng = r * f
            if low >= _TOP:
                low -= _TOP
                i = len(out) - 1
                while out[i] == 255:
                    out[i] = 0; i -= 1
                out[i] += 1
            while rng < _BOT:
                out.append(low >> 40)
                low = (low & _M40) << 8
                rng <<= 8
        self.low, self.rng = low, rng

    def finish(self):
        """shortest byte string whose zero-padded value lies in [low, low + range); trailing zero bytes are dropped (the decoder pads)."""
        low, rng, out = self.low, self.rng, self.out
        for m in range(7):
            step = 1 << (8 * (6 - m))
            v = -(-low // step) * step
            if v < low + rng:
                break
        if v >= _TOP:
            v -= _TOP
            i = len(out) - 1
            while out[i] == 255:
                out[i] = 0; i -= 1
            out[i] += 1
        if m:
            out += (v >> (8 * (6 - m))).to_bytes(m, 'big')
        while out and out[-1] == 0:
            out.pop()
        return bytes(out)


class RangeDecoder:
    def __init__(self, data):
        self.data = bytes(data); self.n = len(self.data); self.pos = 6
        self.code = int.from_bytes(self.data[:6].ljust(6, b'\0'), 'big'); self.rng = _TOP

    def binary(self, f0, tot):
        """decode len(f0) binary symbols; symbol 0 owns [0, f0[i]) of tot."""
        code, rng, data, pos, n = self.code, self.rng, self.data, self.pos, self.n
        out = bytearray(len(f0))
        for i, f in enumerate(f0):
            r = rng // tot
            t = r * f
            if code < t:
                rng = t
            else:
                code -= t; rng = r * (tot - f); out[i] = 1
            while rng < _BOT:
                code = (code << 8) | (data[pos] if pos < n else 0); pos += 1; rng <<= 8
        self.code, self.rng, self.pos = code, rng, pos
        return np.frombuffer(bytes(out), np.uint8).astype(np.int64)

    def categorical(self, cum_rows, freq_rows, tot):
        code, rng, data, pos, n = self.code, self.rng, self.data, self.pos, self.n
        out = []
        for cr, fr in zip(cum_rows, freq_rows):
            r = rng // tot
            f = code // r
            if f >= tot:
                f = tot - 1
            s = bisect.bisect_right(cr, f) - 1
            code -= r * cr[s]; rng = r * fr[s]
            while rng < _BOT:
                code = (code << 8) | (data[pos] if pos < n else 0); pos += 1; rng <<= 8
            out.append(s)
        self.code, self.rng, self.pos = code, rng, pos
        return np.asarray(out, np.int64)

    def binary_w2(self, f0tab, Fb, Ob, band, starts, dj, dk, tot):
        """W2 events in strict event order: context = 9 F + 3 min(occ_edges, 2) + band, where the base digits Fb / Ob come from the
        state before the batch and every earlier event of the same batch that is a face (k < 6) or edge (k >= 6) neighbour adds its
        decoded state (dependency list per event i: dj / dk[starts[i]:starts[i + 1]])."""
        code, rng, data, pos, n = self.code, self.rng, self.data, self.pos, self.n
        m = len(Fb); out = [0] * m
        for i in range(m):
            F = Fb[i]; oc = Ob[i]
            for q in range(starts[i], starts[i + 1]):
                s = out[dj[q]]; k = dk[q]
                if k < 6:
                    F += W2_FACE_W[k] * (1 + s)
                else:
                    oc += s
            f = f0tab[9 * F + 3 * (oc if oc < 2 else 2) + band]
            r = rng // tot
            t = r * f
            if code < t:
                rng = t
            else:
                code -= t; rng = r * (tot - f); out[i] = 1
            while rng < _BOT:
                code = (code << 8) | (data[pos] if pos < n else 0); pos += 1; rng <<= 8
        self.code, self.rng, self.pos = code, rng, pos
        return np.asarray(out, np.int64)

    def uniform(self, tot, count):
        code, rng, data, pos, n = self.code, self.rng, self.data, self.pos, self.n
        out = []
        for _ in range(count):
            r = rng // tot
            f = code // r
            if f >= tot:
                f = tot - 1
            code -= r * f; rng = r
            while rng < _BOT:
                code = (code << 8) | (data[pos] if pos < n else 0); pos += 1; rng <<= 8
            out.append(f)
        self.code, self.rng, self.pos = code, rng, pos
        return np.asarray(out, np.int64)


def varint(n):
    out = bytearray()
    while True:
        b = n & 127; n >>= 7
        out.append(b | (128 if n else 0))
        if not n:
            return bytes(out)


def read_varint(data):
    n = 0; shift = 0
    for i, b in enumerate(data):
        n |= (b & 127) << shift; shift += 7
        if not b & 128:
            return n, i + 1
        if i > 8:
            break
    raise ValueError('bad varint')


# ============================================================================ static models (priors)
def model_specs():
    """model id -> (number of contexts, alphabet size)"""
    s = {'w1u': (512, 2), 'w1c': (512 * 6, 2), 'w2u': (6561, 2), 'w2c': (6561, 2), 'w3pu': (6561, 3), 'w3pc': (6561 * 2, 3)}
    for t in TIERS:
        s.update({f'w3{t}.dpx': (1, 128), f'w3{t}.dpy': (1, 128), f'w3{t}.yabs': (1, 128), f'w3{t}.z0': (1, 32), f'w3{t}.dz': (32, 32),
                  f'w3{t}.dx': (1, 128), f'w3{t}.dy': (6, 128)})
    return s


SPECS = model_specs()


def alpha_of(K):
    return 0.5 if K <= 3 else 8.0 / K                                   # KT for binary / ternary; total pseudo-count 8 for large alphabets


class Table:
    """integer frequency table [nctx, K] with sum TOT per row and every entry >= 1, from counts with add-alpha smoothing."""

    def __init__(self, counts, alpha):
        c = np.asarray(counts, np.float64); K = c.shape[1]
        p = (c + alpha) / (c.sum(1, keepdims=True) + alpha * K)
        f = np.floor(p * (TOT - K)).astype(np.int64) + 1
        f[np.arange(len(f)), np.argmax(p, 1)] += TOT - f.sum(1)
        assert (f >= 1).all() and (f.sum(1) == TOT).all()
        self.freq = f; self.cum = np.cumsum(f, 1) - f
        self._lists = None

    def lists(self):
        if self._lists is None:
            self._lists = (self.cum.tolist(), self.freq.tolist())
        return self._lists


FLAG = Table(np.array([[TOT - 17, 15]]), 0.5)                          # fixed 'one more residual entry' flag: p(more) ~ 2^-16


class Prior:
    def __init__(self, name, counts, houses=None):
        self.name = name; self.houses = list(houses or [])
        self.counts = {m: np.asarray(counts[m], np.int64) for m in SPECS}
        self.tables = {m: Table(self.counts[m], alpha_of(SPECS[m][1])) for m in SPECS}

    def save(self, path):
        np.savez_compressed(path, **{'c__' + m.replace('.', '__'): v for m, v in self.counts.items()}, houses=np.array(json.dumps(self.houses)))

    @staticmethod
    def load(name, path):
        z = np.load(path)
        counts = {k[3:].replace('__', '.'): z[k] for k in z.files if k.startswith('c__')}
        return Prior(name, counts, json.loads(str(z['houses'])))


def zero_counts():
    return {m: np.zeros(SPECS[m], np.int64) for m in SPECS}


# ============================================================================ coding IO: one program, two directions
class EncIO:
    """records (kind, model / total, contexts, symbols, component) chunks; the symbols are supplied by the program."""
    enc = True

    def __init__(self):
        self.chunks = []

    def m(self, model, ctx, sym, comp):
        ctx = np.asarray(ctx, np.int64).ravel(); sym = np.asarray(sym, np.int64).ravel()
        assert ctx.shape == sym.shape
        nctx, K = SPECS[model]
        assert sym.size == 0 or (0 <= sym.min() and sym.max() < K and 0 <= ctx.min() and ctx.max() < nctx), (model, sym.max() if sym.size else None)
        if sym.size:
            self.chunks.append(('m', model, ctx, sym, comp))
        return sym

    def uni(self, tot, vals, comp, count=None):
        vals = np.asarray(vals, np.int64).ravel()
        assert vals.size == 0 or (0 <= vals.min() and vals.max() < tot)
        if vals.size:
            self.chunks.append(('u', tot, None, vals, comp))
        return vals

    def flag(self, val, comp):
        self.chunks.append(('f', 'flag', None, np.asarray([val], np.int64), comp))
        return int(val)

    def w2(self, model, parts, sym, comp):
        """W2 batch in strict event order (see RangeDecoder.binary_w2): the full contexts follow from the known symbols."""
        sym = np.asarray(sym, np.int64)
        return self.m(model, w2_full_ctx(parts, sym), sym, comp)


class DecIO:
    enc = False

    def __init__(self, payload, prior):
        self.rd = RangeDecoder(payload); self.prior = prior

    def m(self, model, ctx, sym=None, comp=None):
        ctx = np.asarray(ctx, np.int64).ravel()
        if ctx.size == 0:
            return np.zeros(0, np.int64)
        tab = self.prior.tables[model]
        if SPECS[model][1] == 2:
            return self.rd.binary(tab.freq[ctx, 0].tolist(), TOT)
        cl, fl = tab.lists()
        return self.rd.categorical([cl[c] for c in ctx.tolist()], [fl[c] for c in ctx.tolist()], TOT)

    def uni(self, tot, vals=None, comp=None, count=1):
        return self.rd.uniform(tot, count)

    def flag(self, val=None, comp=None):
        return int(self.rd.binary([int(FLAG.freq[0, 0])], TOT)[0])

    def w2(self, model, parts, sym=None, comp=None):
        Fb, Ob, band, di, dj, dk = parts
        if len(Fb) == 0:
            return np.zeros(0, np.int64)
        tab = self.prior.tables[model]
        if not hasattr(tab, '_f0'):
            tab._f0 = tab.freq[:, 0].tolist()
        starts = np.searchsorted(di, np.arange(len(Fb) + 1)).tolist()   # di is sorted (built that way in w2_parts)
        return self.rd.binary_w2(tab._f0, Fb.tolist(), Ob.tolist(), band, starts, dj.tolist(), dk.tolist(), TOT)


def encode_chunks(chunks, prior):
    """range-code recorded chunks with `prior`; returns payload bytes and the ideal code length (bits) per component."""
    enc = RangeEncoder(); bits = dict.fromkeys(COMPS, 0.0)
    for kind, model, ctx, sym, comp in chunks:
        if kind == 'm':
            tab = prior.tables[model]
            f = tab.freq[ctx, sym]; c = tab.cum[ctx, sym]
            enc.encode(c.tolist(), f.tolist(), TOT)
            bits[comp] += float(np.sum(TOT_BITS - np.log2(f)))
        elif kind == 'u':
            enc.encode(sym.tolist(), [1] * len(sym), model)
            bits[comp] += len(sym) * math.log2(model)
        else:
            f = FLAG.freq[0, sym]; c = FLAG.cum[0, sym]
            enc.encode(c.tolist(), f.tolist(), TOT)
            bits[comp] += float(np.sum(TOT_BITS - np.log2(f)))
    return enc.finish(), bits


def accumulate(chunks, counts):
    for kind, model, ctx, sym, comp in chunks:
        if kind == 'm':
            nctx, K = SPECS[model]
            counts[model] += np.bincount(ctx * K + sym, minlength=nctx * K).reshape(nctx, K)


# ============================================================================ geometry helpers
_lin = np.arange(V, dtype=np.int64)
_xs, _ys, _zs = _lin // YZ, (_lin // NZ) % NY, _lin % NZ
_wf = _xs + _ys + _zs
WF_IDX = _lin[np.lexsort((_lin, _wf))]
WF_PTR = np.searchsorted(_wf[WF_IDX], np.arange(_wf.max() + 2))
NWF = int(_wf.max()) + 1
del _lin, _xs, _ys, _zs, _wf
_cl = np.arange(NX * NY, dtype=np.int64)
_w2 = 2 * (_cl // NY) + _cl % NY
WF2_IDX = _cl[np.lexsort((_cl, _w2))]
WF2_PTR = np.searchsorted(_w2[WF2_IDX], np.arange(_w2.max() + 2))
NWF2 = int(_w2.max()) + 1
del _cl, _w2
NB3 = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0)]   # rate_baselines NB (causal)
NB2 = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 1), (1, 2)]                                    # A03 plan context (causal in 2x+y)
FACE = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
EDGE = [(a, b, 0) for a in (-1, 1) for b in (-1, 1)] + [(a, 0, b) for a in (-1, 1) for b in (-1, 1)] + [(0, a, b) for a in (-1, 1) for b in (-1, 1)]


def _nbr(u, d):
    x = u // YZ; y = (u // NZ) % NY; z = u % NZ
    nx, ny, nz = x + d[0], y + d[1], z + d[2]
    ok = (nx >= 0) & (nx < NX) & (ny >= 0) & (ny < NY) & (nz >= 0) & (nz < NZ)
    return np.where(ok, (nx * NY + ny) * NZ + nz, 0), ok


def ctx_w1(Hc, idx):
    x = idx // YZ; y = (idx // NZ) % NY; z = idx % NZ
    c = np.zeros(len(idx), np.int64)
    for dx, dy, dz in NB3:
        ok = (x >= dx) & (y >= dy) & (z >= dz)
        n = np.where(ok, idx - (dx * YZ + dy * NZ + dz), 0)
        c = c * 2 + (ok & Hc[n])
    return c


def sfeat(S, idx):
    """receiver-shared features of the 6 face neighbours: min(#known free, 2) + 3 * any known occupied (0..5)"""
    nf = np.zeros(len(idx), np.int64); no = np.zeros(len(idx), bool)
    for d in FACE:
        n, ok = _nbr(idx, d)
        s = S[n]
        nf += ok & (s == 0); no |= ok & (s == 1)
    return np.minimum(nf, 2) + 3 * no


W2_FACE_W = [243, 81, 27, 9, 3, 1]                                     # base-3 weight of face neighbour k in F
_POS = np.full(V, -1, np.int64)                                         # scratch: position of a voxel in the current batch


def w2_parts(grid, u, band):
    """A01 context in strict event order (events ordered by step, pose, ray): ctx = 9 F + 3 min(#occupied edge neighbours, 2) + band,
    F = base-3 number of the 6 face neighbours' states (0 undecided, 1 free, 2 occupied). Returned as the part known before the batch
    (Fb, Ob: neighbours decided in earlier batches) plus the dependency list (event i, earlier event j of the same batch, neighbour slot
    k) through which the decoded states of earlier events of this batch enter the context of event i."""
    n = len(u); x = u // YZ; y = (u // NZ) % NY; z = u % NZ
    Fb = np.zeros(n, np.int64); Ob = np.zeros(n, np.int64)
    ar = np.arange(n); _POS[u] = ar
    di, dj, dk = [], [], []
    for k, (dx, dy, dz) in enumerate(FACE + EDGE):
        nx, ny, nz = x + dx, y + dy, z + dz
        ok = (nx >= 0) & (nx < NX) & (ny >= 0) & (ny < NY) & (nz >= 0) & (nz < NZ)
        nb = np.where(ok, u + (dx * YZ + dy * NZ + dz), 0)
        g = grid[nb]                                                    # pre-batch state (this batch's voxels are still undecided)
        if k < 6:
            Fb = Fb * 3 + np.where(ok, g + 1, 0)
        else:
            Ob += ok & (g == 1)
        p = np.where(ok, _POS[nb], -1)
        e = (p >= 0) & (p < ar)
        if e.any():
            di.append(ar[e]); dj.append(p[e]); dk.append(np.full(int(e.sum()), k, np.int64))
    _POS[u] = -1
    if di:
        di = np.concatenate(di); dj = np.concatenate(dj); dk = np.concatenate(dk)
        o = np.lexsort((dk, di)); di, dj, dk = di[o], dj[o], dk[o]
    else:
        di = dj = dk = np.zeros(0, np.int64)
    return Fb, Ob, band, di, dj, dk


def w2_full_ctx(parts, sym):
    Fb, Ob, band, di, dj, dk = parts
    F = Fb.copy(); O = Ob.copy()
    face = dk < 6
    np.add.at(F, di[face], np.asarray(W2_FACE_W, np.int64)[dk[face]] * (1 + sym[dj[face]]))
    np.add.at(O, di[~face], sym[dj[~face]])
    return 9 * F + 3 * np.minimum(O, 2) + band


def ctx_plan(Pl, cols):
    x = cols // NY; y = cols % NY
    c = np.zeros(len(cols), np.int64)
    for dx, dy in NB2:
        nx, ny = x - dx, y - dy
        ok = (nx >= 0) & (ny >= 0) & (ny < NY)
        c = c * 3 + np.where(ok, Pl[np.where(ok, nx * NY + ny, 0)], 0)
    return c


def march_params(cell):
    cellv = np.asarray(cell, np.float64)
    ds = 0.5 * float(cellv.min())
    return cellv, ds, int(math.ceil(W.SENSOR_RANGE / ds))


def reach_mask(P64s, cell):
    """voxels a public ray of these poses can enter (box within nsteps * ds of a pose, + 1e-6 m margin); identical on both ends."""
    cellv, ds, nsteps = march_params(cell)
    R2 = (nsteps * ds + 1e-6) ** 2
    lo = [np.arange(n) * c for n, c in zip(DIMS, cellv)]; hi = [(np.arange(n) + 1) * c for n, c in zip(DIMS, cellv)]
    m = np.zeros(DIMS, bool)
    for P in P64s:
        for p in P:
            d = [np.maximum(np.maximum(lo[k] - p[k], p[k] - hi[k]), 0.) ** 2 for k in range(3)]
            m |= (d[0][:, None, None] + d[1][None, :, None] + d[2][None, None, :]) <= R2
    return m.ravel()


def pose64(P32):
    """decoded float32 (x, y) of the 3 poses -> float64 [3, 3] with the protocol-constant sensor height."""
    P = np.empty((P32.shape[0], 3), np.float64)
    P[:, :2] = P32.astype(np.float64); P[:, 2] = Z_SENSOR
    return P


# ============================================================================ memoised W.observe
class RenderCache:
    """render(W', P) = W.observe(W', cell, P, RAYS) as (visited voxels K, W'[K]); a cached entry is reused only if W' equals the cached
    world on K (then every ray meets the same values on its path and stops at the same sample -> identical output)."""

    def __init__(self, cell, enabled=True):
        self.cell = cell; self.enabled = enabled; self.entries = {}; self.hits = 0; self.misses = 0; self.seconds = 0.

    def render(self, Wp, P):
        key = P.tobytes()
        if self.enabled:
            for K, wv in self.entries.get(key, ()):
                if np.array_equal(Wp[K], wv):
                    self.hits += 1
                    return K, wv
        t0 = time.time()
        out = W.observe(Wp.reshape(DIMS).view(np.uint8), self.cell, P, RAYS).ravel()
        K = np.flatnonzero(out >= 0); wv = Wp[K].copy()
        assert np.array_equal(out[K] == 1, wv), 'observe: visited values differ from the world'
        self.seconds += time.time() - t0; self.misses += 1
        if self.enabled:
            self.entries.setdefault(key, []).append((K, wv))
        return K, wv


# ============================================================================ W3 structure (A03 S0, label-free, consistent by construction)
def _psum(m):
    p = np.zeros(tuple(d + 1 for d in m.shape), np.int64)
    p[1:, 1:, 1:] = m.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    return p


def _bsum(p, x0, x1, y0, y1, z0, z1):
    return (p[x1, y1, z1] - p[x0, y1, z1] - p[x1, y0, z1] - p[x1, y1, z0] + p[x0, y0, z1] + p[x0, y1, z0] + p[x1, y0, z0] - p[x0, y0, z0])


def box_cover(must, forbid):
    """A03 a03np.box_cover (copied): greedy maximal boxes seeded at the first uncovered must-voxel in (z, x, y) order, grown one layer at
    a time in 6 directions while the layer holds no forbidden voxel and at least one must-voxel. Half-open (x0, x1, y0, y1, z0, z1)."""
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


BAND = (23, 25)            # z 2.3-2.5 m: above every furniture top of the generator, below the lowest ceiling (A03 p8)
LINTEL_MAX_FREE = 21       # a lintel column's observed free space ends at z index <= 21 (door tops 2.1-2.25 m)


def paint(kf, kc, lz, plan, boxes):
    g = np.zeros(DIMS, bool)
    g[:, :, :kf] = True; g[:, :, kc:] = True
    g[plan == 1, :] = True
    g[plan == 2, lz:] = True
    for x0, x1, y0, y1, z0, z1 in boxes:
        g[x0:x1, y0:y1, z0:z1] = True
    return g.ravel()


def describe_w3(T, Socc):
    """S0 description of target T (cond: Socc = receiver-shared hits, which the decoder joins into the painted world for free)."""
    Tg = T.reshape(DIMS); F = Tg == 0
    Hn = (Tg == 1) if Socc is None else ((Tg == 1) & ~Socc.reshape(DIMS))
    zf = np.flatnonzero(F.any((0, 1)))
    kf = int(zf[0]) if zf.size else 0
    kc = int(zf[-1]) + 1 if zf.size else NZ
    anyf = F.any(2)
    colfree_max = np.where(anyf, NZ - 1 - np.argmax(F[:, :, ::-1], 2), -1)
    upper = (Tg[:, :, BAND[0]:BAND[1]] == 1).any(2)
    if Socc is not None:
        upper &= Hn.any(2)
    wall = upper & (colfree_max < 0)
    lcand = upper & (colfree_max >= 0) & (colfree_max <= LINTEL_MAX_FREE)
    lz = int(colfree_max[lcand].max()) + 1 if lcand.any() else NZ
    plan = np.where(wall, 1, np.where(lcand, 2, 0)).astype(np.int64)
    P = paint(kf, kc, lz, plan, []).reshape(DIMS)
    assert not (P & F).any(), 'W3 paint meets an observed free voxel'
    boxes = box_cover(Hn & ~P, F)
    return kf, kc, lz, plan, boxes


def dy_bucket(dx):
    return 0 if dx <= 1 else 1 if dx == 2 else 2 if dx <= 4 else 3 if dx <= 8 else 4 if dx <= 16 else 5


def code_plan(io, tier, S, plan=None):
    model = 'w3p' + tier
    sflag = (S.reshape(DIMS)[:, :, BAND[0]:BAND[1]] == 1).any(2).ravel().astype(np.int64) if tier == 'c' else None
    flat = plan.ravel() if io.enc else None
    Pl = np.zeros(NX * NY, np.int64)
    for w in range(NWF2):
        cols = WF2_IDX[WF2_PTR[w]:WF2_PTR[w + 1]]
        ctx = ctx_plan(Pl, cols)
        if sflag is not None:
            ctx = ctx + 6561 * sflag[cols]
        Pl[cols] = io.m(model, ctx, flat[cols] if io.enc else None, 'plan')
    return Pl.reshape(NX, NY)


def code_boxes(io, tier, boxes=None):
    t = tier
    n = int(io.uni(4096, [len(boxes)] if io.enc else None, 'box')[0])
    B = sorted(boxes, key=lambda b: (b[0], b[2], b[4])) if io.enc else None
    out = []; px = py = 0
    for i in range(n):
        if io.enc:
            x0, x1, y0, y1, z0, z1 = B[i]
        dpx = int(io.m(f'w3{t}.dpx', [0], [x0 - px] if io.enc else None, 'box')[0]); X0 = px + dpx
        if dpx == 0:
            Y0 = py + int(io.m(f'w3{t}.dpy', [0], [y0 - py] if io.enc else None, 'box')[0])
        else:
            Y0 = int(io.m(f'w3{t}.yabs', [0], [y0] if io.enc else None, 'box')[0])
        Z0 = int(io.m(f'w3{t}.z0', [0], [z0] if io.enc else None, 'box')[0])
        DZ = 1 + int(io.m(f'w3{t}.dz', [Z0], [z1 - z0 - 1] if io.enc else None, 'box')[0])
        DX = 1 + int(io.m(f'w3{t}.dx', [0], [x1 - x0 - 1] if io.enc else None, 'box')[0])
        DY = 1 + int(io.m(f'w3{t}.dy', [dy_bucket(DX)], [y1 - y0 - 1] if io.enc else None, 'box')[0])
        out.append((X0, X0 + DX, Y0, Y0 + DY, Z0, Z0 + DZ))
        px, py = X0, Y0
    return out


# ============================================================================ wire bodies
def body_w1(io, tier, S, P64s, cell, T, fault=None):
    cand = reach_mask(P64s, cell)
    if tier == 'c':
        cand &= S < 0
    Hc = (S == 1) if tier == 'c' else np.zeros(V, bool)
    model = 'w1' + tier
    for w in range(NWF):
        idx = WF_IDX[WF_PTR[w]:WF_PTR[w + 1]]
        idx = idx[cand[idx]]
        if idx.size == 0:
            continue
        ctx = ctx_w1(Hc, idx)
        if tier == 'c':
            ctx = ctx + 512 * sfeat(S, idx)
        Hc[idx] = io.m(model, ctx, (T[idx] == 1) if io.enc else None, 'hit').astype(bool)
    if fault == 'decoder_drop_hit' and not io.enc:                      # test hook: a decoder bug that loses one decoded hit
        on = np.flatnonzero(Hc & cand)
        if on.size:
            Hc[on[on.size // 2]] = False
    return Hc


class Traversal:
    """the voxel every public ray of ONE pose samples at each step -- a function of the pose and the public rays only (the same float64
    ops as W.observe) -- recorded on the first march of that pose up to where each ray stopped. hitend[r]: the recording stopped at a
    hit, so a later march in a different world may continue that ray (it is then computed on the fly); otherwise the ray left the grid
    (or the range) right after its last recorded step."""

    def __init__(self, ids, lins, hit_ids, R):
        ids = np.concatenate(ids) if ids else np.zeros(0, np.int64); lins = np.concatenate(lins) if lins else np.zeros(0, np.int32)
        order = np.argsort(ids, kind='stable')                         # recorded step-major -> stable sort gives each ray's steps in order
        self.seq = lins[order].astype(np.int32)
        self.L = np.bincount(ids, minlength=R).astype(np.int64)
        self.off = np.cumsum(self.L) - self.L
        self.hitend = np.zeros(R, bool)
        if hit_ids:
            self.hitend[np.concatenate(hit_ids)] = True


def _sample(P, ids, sd, cellv, shape):
    idx = np.floor((P[None] + RAYS[ids] * sd) / cellv).astype(np.int64)      # == W.observe's p[None] + rays * (s * ds), row by row
    inside = np.all((idx >= 0) & (idx < shape), 1)
    return (idx[:, 0] * NY + idx[:, 1]) * NZ + idx[:, 2], inside


def event_march(io, model, grid, P, cell, T, travs=None):
    """W2: march the public rays of one agent's poses (step-major, poses interleaved) exactly as W.observe does (same float64 ops),
    deciding every undecided voxel a ray enters by one coded event; a ray stops in a voxel decided occupied.
    travs: per-house dict pose-bytes -> Traversal (recorded here on the first march of a pose, replayed afterwards)."""
    cellv, ds, nsteps = march_params(cell)
    shape = np.array(DIMS); R = len(RAYS)
    alive = [np.arange(R) for _ in range(len(P))]; done = [False] * len(P)
    tr = [travs.get(P[k].tobytes()) if travs is not None else None for k in range(len(P))]
    rec = [([], [], []) if (travs is not None and tr[k] is None) else None for k in range(len(P))]
    for s in range(1, nsteps + 1):
        if all(done):
            break
        sd = s * ds
        band = 0 if sd < 1.0 else (1 if sd < 3.0 else 2)
        for k in range(len(P)):
            if done[k]:
                continue
            ids = alive[k]
            if tr[k] is not None:                                       # replay; rays past their recorded end are computed on the fly
                t = tr[k]
                cov = t.L[ids] >= s
                lin = np.zeros(len(ids), np.int64); inside = cov.copy()
                lin[cov] = t.seq[t.off[ids[cov]] + (s - 1)]
                if not cov.all():
                    pos = np.flatnonzero(~cov); pos = pos[t.hitend[ids[pos]]]
                    if pos.size:
                        lin[pos], inside[pos] = _sample(P[k], ids[pos], sd, cellv, shape)
            else:
                lin, inside = _sample(P[k], ids, sd, cellv, shape)
            if not inside.any():
                done[k] = True; alive[k] = ids[:0]
                continue
            if not inside.all():
                ids = ids[inside]; lin = lin[inside]
            if rec[k] is not None:
                rec[k][0].append(ids); rec[k][1].append(lin.astype(np.int32))
            st = grid[lin]
            new = st < 0
            if new.any():
                ln = lin[new]
                u, first = np.unique(ln, return_index=True)
                u = u[np.argsort(first, kind='stable')]
                grid[u] = io.w2(model, w2_parts(grid, u, band), (T[u] != 0) if io.enc else None, 'event')
                st = grid[lin]
            hit = st == 1
            if rec[k] is not None and hit.any():
                rec[k][2].append(ids[hit])
            alive[k] = ids[~hit]
    for k in range(len(P)):
        if rec[k] is not None:
            travs[P[k].tobytes()] = Traversal(rec[k][0], rec[k][1], rec[k][2], R)
    return grid


def body_w2(io, tier, S, P64s, cell, T, travs=None):
    grid = S.copy() if tier == 'c' else np.full(V, -1, np.int8)
    for P in P64s:
        event_march(io, 'w2' + tier, grid, P, cell, T, travs)
    return grid == 1, grid


def body_w3(io, tier, S, T):
    Socc = (S == 1) if tier == 'c' else None
    if io.enc:
        kf, kc, lz, plan, boxes = describe_w3(T, Socc)
    else:
        kf = kc = lz = plan = boxes = None
    kf = int(io.uni(NZ, [kf] if io.enc else None, 'plan')[0])
    kc = int(io.uni(NZ + 1, [kc] if io.enc else None, 'plan')[0])
    lz = int(io.uni(NZ + 1, [lz] if io.enc else None, 'plan')[0])
    plan = code_plan(io, tier, S, plan)
    boxes = code_boxes(io, tier, boxes)
    Wp = paint(kf, kc, lz, plan, boxes)
    if Socc is not None:
        Wp |= Socc
    return Wp, len(boxes)


# ============================================================================ the message program (encoder and decoder run the same code)
class PublicCtx:
    """everything a decoder may use besides the bit stream, the edge's shared map S and the prior: the session grid (cell), its own
    render memo and traversal store (both functions of public data only). __slots__: any other attribute access raises."""
    __slots__ = ('cell', 'rcache', 'travs')

    def __init__(self, cell, rcache, travs):
        self.cell = cell; self.rcache = rcache; self.travs = travs


def run_message(io, wire, tier, S, pub, tgt=None, fault=None):
    """io = EncIO with tgt = dict(T = target map after decoding, mask = agents to send, P32 = their float32 poses) or DecIO with tgt = None.
    pub = PublicCtx. S = the edge's shared map (tier c) or None. Returns the decoded map, decoded agent list, their float32 poses, info."""
    enc = io.enc
    mask = int(io.uni(256, [tgt['mask']] if enc else None, 'header')[0])
    agents = [a for a in range(8) if (mask >> a) & 1]
    P32 = {}
    for a in agents:
        if enc:
            u32 = tgt['P32'][a].view(np.uint32).ravel()
            halves = np.stack([u32 >> 16, u32 & 0xFFFF], 1).ravel().astype(np.int64)
        h = io.uni(1 << 16, halves if enc else None, 'pose', count=12)
        u32d = ((h[0::2].astype(np.uint32) << np.uint32(16)) | h[1::2].astype(np.uint32)).astype(np.uint32)
        P32[a] = u32d.view(np.float32).reshape(3, 2)
    P64s = [pose64(P32[a]) for a in agents]
    if fault == 'decoder_pose_round' and not enc:                       # test hook: a decoder that rebuilds poses at 1 cm
        for P in P64s:
            P[:, :2] = np.round(P[:, :2], 2)
    D = S.copy() if tier == 'c' else np.full(V, -1, np.int8)
    T = tgt['T'] if enc else None
    info = {}
    if agents:
        if wire == 'W1':
            Wp = body_w1(io, tier, S, P64s, pub.cell, T, fault)
        elif wire == 'W2':
            Wp, grid = body_w2(io, tier, S, P64s, pub.cell, T, pub.travs)
        else:
            Wp, info['boxes'] = body_w3(io, tier, S, T)
        for P in P64s:
            K, wv = pub.rcache.render(Wp, P)
            D[K] = np.maximum(D[K], wv.astype(np.int8))
        if wire == 'W2':
            info['march_render_equal'] = bool(np.array_equal(grid, D))
            if enc and not info['march_render_equal']:
                raise AssertionError('W2 event march and W.observe render disagree')
    # residual: every voxel where the reconstruction still differs from the target (encoder side), index + ternary value
    if enc:
        ridx = np.flatnonzero(D != T); rval = T[ridx].astype(np.int64)
    k = 0; got_i, got_v = [], []
    while True:
        more = io.flag(int(k < len(ridx)) if enc else None, 'residual')
        if not more:
            break
        i = int(io.uni(V, [ridx[k]] if enc else None, 'residual')[0]); v = int(io.uni(3, [rval[k] + 1] if enc else None, 'residual')[0]) - 1
        got_i.append(i); got_v.append(v); k += 1
        if k > 100000:
            raise ValueError('runaway residual list')
    if got_i:
        D[np.asarray(got_i)] = np.asarray(got_v, np.int8)
    info['residual'] = len(got_i)
    return D, agents, P32, info


# ============================================================================ houses and protocols
def scene_key(sc):
    return f"{sc['split'][:3]}:r{sc['rooms']}:i{sc['index']}"


def digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else repr(a).encode())
    return h.hexdigest()


def join(ks):
    m = ks[0][0].copy(); a = set(ks[0][1])
    for mm, aa in ks[1:]:
        np.maximum(m, mm, out=m); a |= aa
    return m, frozenset(a)


def tree_side(nbrs, u, v):
    side = {u}; stack = [u]
    while stack:
        x = stack.pop()
        for w in nbrs[x]:
            if w not in side and not (x == u and w == v):
                side.add(w); stack.append(w)
    return sorted(side)


def score_f1(p, t, slab, obj, struct):
    """rate_baselines_lib.metrics on one map (target-known scope), numpy/scipy (as A03 a03np.score_np)."""
    cross = ndimage.generate_binary_structure(3, 1)
    d6 = lambda m: ndimage.binary_dilation(m, cross)
    def f1(a, b):
        tp = int((a & b).sum()); fp = int((a & ~b).sum()); fn = int((~a & b).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    p = p.reshape(DIMS); t = t.reshape(DIMS)
    s = (t == 1) & d6(t == 0); region = ~slab & ~d6(struct); known = t >= 0
    po = (p == 1) & known; to = (t == 1) & known; ps = po & ~((t == 1) & ~s)
    return dict(surface_f1=f1(ps, s), var_occ_f1=f1(po & ~slab, to & ~slab), furn_f1=f1(po & region, to & region))


MSG_COLS = ['panel', 'house', 'wire', 'prior', 'proto', 'tier', 'round', 'u', 'v', 'bytes', 'new_agents', 'exact', 'mismatch', 'residual',
            'error'] + ['bits_' + c for c in COMPS]
CHK_COLS = ['panel', 'house', 'wire', 'prior', 'proto', 'tier', 'hop', 'agent', 'exact', 'mismatch', 'free_recall', 'fpr']


class House:
    def __init__(self, scene, panel, priors=(), counts=None, fault=None, render_cache=True):
        t0 = time.time()
        fx = W.make_fixture(scene['split'], scene['index'], scene['rooms'])
        self.t_fixture = time.time() - t0
        self.scene = scene; self.key = scene_key(scene); self.panel = panel
        self.n = int(fx['private'].shape[0]); assert self.n == 8
        self.cell = fx['cell']
        self.own = fx['private'].reshape(self.n, V).astype(np.int8)
        traj = fx['trajectory_m']
        assert np.all(traj[:, :, 2] == Z_SENSOR), 'sensor height is not the protocol constant'
        self.P32 = [np.ascontiguousarray(traj[:, a, :2].astype(np.float32)) for a in range(self.n)]
        self.edges = [(int(i), int(j)) for i, j, _ in fx['edges']]
        self.nbrs = {a: [] for a in range(self.n)}
        for i, j in self.edges:
            self.nbrs[i].append(j); self.nbrs[j].append(i)
        for a in self.nbrs:
            self.nbrs[a].sort()
        self.directed = sorted([(i, j) for i, j in self.edges] + [(j, i) for i, j in self.edges])
        R = np.asarray(fx['reached'], bool)
        self.targets = {}
        for k, h in enumerate(HOPS):
            self.targets[h] = np.stack([self.own[R[k][a]].max(0) for a in range(self.n)])
        self.slab = fx['slab'].astype(bool); self.obj = fx['obj'].astype(bool)
        self.struct = fx['occ'].astype(bool) & ~self.obj & ~self.slab
        self.chart_m = float(fx['chart_m'])
        self.priors = list(priors); self.counts = counts; self.fault = fault
        self.house_id = str(fx['world_id']).split(':')[-1]              # HouseExpo id (prior / panel disjointness evidence)
        self.rcache = RenderCache(self.cell, render_cache)
        self.travs = {} if render_cache else None                       # W2 ray traversals (world-independent), see Traversal
        self.public = PublicCtx(self.cell, self.rcache, self.travs)     # the only house object a decoder sees
        self.rows = []; self.checks = []; self.errors = []; self.dec_count = 0

    # ---------------------------------------------------------------- one message
    def send(self, wire, tier, prior, content, Sst, proto, rnd, u, v):
        U, A = content
        if tier == 'c':
            S, SA = Sst
            T = np.maximum(S, U); new = sorted(A - SA)
        else:
            S, SA = None, frozenset(); T = U; new = sorted(A)
        mask = sum(1 << a for a in new)
        key = (wire, tier, digest(U, tuple(sorted(A))), digest(S, tuple(sorted(SA))) if S is not None else None)
        rec = self.enc_cache.get(key)
        if rec is None:
            rec = self._encode(wire, tier, S, T, mask); self.enc_cache[key] = rec
        if prior is None:                                               # counting mode (TRAIN): the codec is exact by construction
            return (T, SA | frozenset(new)), None
        data = rec['bytes'][prior.name]
        D, dec_agents, poses_ok, info, err = self._decode(wire, tier, S, data, prior)
        if D is None:
            D = S.copy() if S is not None else np.full(V, -1, np.int8)
        mism = int((D != T).sum())
        exact = bool(mism == 0 and sorted(dec_agents) == new and poses_ok and not err and info.get('march_render_equal', True))
        bits = rec['bits'][prior.name]
        row = [self.panel, self.key, wire, prior.name, proto, tier, rnd, u, v, len(data), len(new), exact, mism, info.get('residual', -1),
               err or ('' if info.get('march_render_equal', True) else 'march/render disagree')] + [round(bits[c], 3) for c in COMPS]
        self.rows.append(row)
        if not exact:
            self.errors.append(dict(wire=wire, prior=prior.name, proto=proto, tier=tier, round=rnd, u=u, v=v, mismatch=mism, error=err,
                                    poses_ok=poses_ok, agents=[sorted(dec_agents), new]))
        return (D, SA | frozenset(dec_agents)), row

    def _encode(self, wire, tier, S, T, mask):
        io = EncIO()
        D, agents, P32, info = run_message(io, wire, tier, S, self.public, dict(T=T, mask=mask, P32=self.P32))
        assert np.array_equal(D, T), 'encoder-side reconstruction is not exact'
        rec = dict(residual=info['residual'], bytes={}, bits={})
        if self.counts is not None:
            accumulate(io.chunks, self.counts)
        for p in self.priors:
            payload, bits = encode_chunks(io.chunks, p)
            data = varint(len(payload)) + payload
            bits['length'] = 8.0 * (len(data) - len(payload))
            bits['coder'] = 8.0 * len(payload) - sum(bits[c] for c in COMPS if c not in ('length', 'coder'))
            rec['bytes'][p.name] = data; rec['bits'][p.name] = bits
        return rec

    def _decode(self, wire, tier, S, data, prior):
        dkey = (wire, tier, prior.name, data, digest(S) if S is not None else None)
        hit = self.dec_cache.get(dkey)
        if hit is not None:
            return hit
        self.dec_count += 1
        try:
            if self.fault and self.fault.get('kind') == 'payload_bitflip' and self.fault.get('wire') == wire and self.dec_count == self.fault.get('nth', 3):
                b = bytearray(data); b[len(b) // 2] ^= 0x10; data = bytes(b)
            n, off = read_varint(data); payload = data[off:off + n]
            if len(payload) != n or off + n != len(data):
                raise ValueError('length field does not match the payload')
            io = DecIO(payload, prior)
            fk = self.fault.get('kind') if (self.fault and self.fault.get('wire') == wire) else None
            D, agents, P32, info = run_message(io, wire, tier, S, self.public, None, fault=fk)
            poses_ok = all(np.array_equal(P32[a].view(np.uint32), self.P32[a].view(np.uint32)) for a in agents)
            out = (D, agents, poses_ok, info, None)
        except Exception as e:                                          # garbage streams must be caught and flagged, not crash the batch
            out = (None, [], False, {}, f'{type(e).__name__}: {e}'[:200])
        self.dec_cache[dkey] = out
        return out

    def check(self, wire, prior, proto, tier, hop, agent, D, T):
        mism = int((D != T).sum())
        tf = T == 0; tu = T < 0
        fr = float((D[tf] == 0).sum() / max(int(tf.sum()), 1)); fpr = float((D[tu] >= 0).sum() / max(int(tu.sum()), 1))
        ok = mism == 0
        self.checks.append([self.panel, self.key, wire, prior, proto, tier, hop, agent, ok, mism, round(fr, 6), round(fpr, 8)])
        if not ok:
            self.errors.append(dict(wire=wire, prior=prior, proto=proto, tier=tier, hop=hop, agent=agent, mismatch=mism, kind='belief'))

    # ---------------------------------------------------------------- protocols
    def own_k(self, a):
        return self.own[a], frozenset([a])

    def sim_k0(self, wire, prior):
        pn = prior.name if prior else None
        for a in range(self.n):
            (D, _), row = self.send(wire, 'u', prior, self.own_k(a), None, 'K0', 0, a, -1)
            if prior is not None:
                self.check(wire, pn, 'K0', 'u', 0, a, D, self.own[a])

    def sim_cd(self, wire, tier, prior):
        pn = prior.name if prior else None
        parent = {}; order = []
        for r in range(self.n):
            if r in parent:
                continue
            parent[r] = None; q = [r]
            while q:
                x = q.pop(0); order.append(x)
                for w in self.nbrs[x]:
                    if w not in parent:
                        parent[w] = x; q.append(w)
        children = {a: [w for w in self.nbrs[a] if parent.get(w) == a] for a in range(self.n)}
        D = {}; empty = (np.full(V, -1, np.int8), frozenset())
        for x in reversed(order):                                       # collect: children before parents
            p = parent[x]
            if p is None:
                continue
            cont = join([self.own_k(x)] + [D[(c, x)] for c in children[x]])
            D[(x, p)], _ = self.send(wire, tier, prior, cont, empty if tier == 'c' else None, 'CD', 1, x, p)
        for x in order:                                                 # distribute: parents before children
            for c in children[x]:
                cont = join([self.own_k(x)] + [D[(w, x)] for w in self.nbrs[x] if w != c])
                D[(x, c)], _ = self.send(wire, tier, prior, cont, D[(c, x)] if tier == 'c' else None, 'CD', 2, x, c)
        if prior is not None:
            for v in range(self.n):
                b = join([self.own_k(v)] + [D[(u, v)] for u in self.nbrs[v]])
                self.check(wire, pn, 'CD', tier, 8, v, b[0], self.targets[8][v])

    def sim_sync(self, wire, tier, prior):
        pn = prior.name if prior else None
        belief = [self.own_k(a) for a in range(self.n)]
        recv = {}
        S = {tuple(sorted(e)): (np.full(V, -1, np.int8), frozenset()) for e in self.edges}
        final = None
        for t in range(1, SYNC_ROUNDS + 1):
            newD = {}
            for u, v in self.directed:
                if tier == 'u':
                    cont = join([self.own_k(u)] + [recv[(w, u)] for w in self.nbrs[u] if w != v and (w, u) in recv])
                    newD[(u, v)], _ = self.send(wire, 'u', prior, cont, None, 'SYNC', t, u, v)
                else:
                    newD[(u, v)], _ = self.send(wire, 'c', prior, belief[u], S[tuple(sorted((u, v)))], 'SYNC', t, u, v)
            if tier == 'u':
                recv = newD
                belief = [join([self.own_k(v)] + [newD[(u, v)] for u in self.nbrs[v]]) for v in range(self.n)]
            else:
                belief = [join([belief[v]] + [newD[(u, v)] for u in self.nbrs[v]]) for v in range(self.n)]
                for i, j in self.edges:
                    S[tuple(sorted((i, j)))] = join([newD[(i, j)], newD[(j, i)]])
            if prior is not None and t in HOPS:
                for v in range(self.n):
                    self.check(wire, pn, 'SYNC', tier, t, v, belief[v][0], self.targets[t][v])
            final = belief
        if prior is not None:
            for v in range(self.n):                                     # round 0 = own map (the sender's starting belief)
                self.check(wire, pn, 'SYNC', tier, 0, v, self.own[v], self.targets[0][v])
        return final

    def sim_cav(self, wire, prior):
        pn = prior.name if prior else None
        for u, v in self.directed:
            su = tree_side(self.nbrs, u, v); sv = tree_side(self.nbrs, v, u)
            cont = (self.own[su].max(0), frozenset(su)); Sst = (self.own[sv].max(0), frozenset(sv))
            (D, _), _ = self.send(wire, 'c', prior, cont, Sst, 'CAV', 0, u, v)
            if prior is not None:
                self.check(wire, pn, 'CAV', 'c', 8, u * 8 + v, D, self.targets[8][v])

    def run(self, wires=WIRES):
        timing = {}
        f1 = {}
        for wire in wires:
            t0 = time.time()
            self.enc_cache = {}
            for prior in (self.priors or [None]):
                self.dec_cache = {}
                self.sim_k0(wire, prior)
                self.sim_cd(wire, 'u', prior); self.sim_cd(wire, 'c', prior)
                self.sim_sync(wire, 'u', prior)
                final = self.sim_sync(wire, 'c', prior)
                self.sim_cav(wire, prior)
                if prior is not None:
                    f1[f'{wire}|{prior.name}'] = [score_f1(final[v][0], self.targets[8][v], self.slab, self.obj, self.struct) for v in range(self.n)]
            timing[wire] = round(time.time() - t0, 1)
            self.enc_cache = {}; self.dec_cache = {}
        return timing, f1


# ============================================================================ scenes, demo check, workers
def panels(val_limit=None, fresh_limit=None, train_limit=None):
    def rr(scenes, k):                                                  # round-robin over the room strata (index-major)
        s = sorted(scenes, key=lambda d: (d['index'], d['rooms']))
        return s if k is None else s[:k]
    val = [s for s in W.scene_plan('VALIDATION', 16) if s['index'] < 8]
    fresh = [s for s in W.scene_plan('VALIDATION', 16) if 8 <= s['index'] < 16]
    train = W.scene_plan('TRAIN', 70)
    A = [s for s in train if s['index'] % 2 == 0]; B = [s for s in train if s['index'] % 2 == 1]
    h = None if train_limit is None else train_limit // 2
    return dict(VAL=rr(val, val_limit), fresh=rr(fresh, fresh_limit), trainA=rr(A, h), trainB=rr(B, h if train_limit is None else train_limit - h))


def load_demo(path):
    """numpy copy of claude_handoff/analysis_20261027/rate_baselines_lib.load (own maps and ground truth planes of a demo house)."""
    m = json.load(open(path)); raw = gzip.decompress(base64.b64decode(m['bin_gz_b64']))
    def plane(name):
        o, n = m['offsets'][name]
        return np.unpackbits(np.frombuffer(raw[o:o + n], dtype=np.uint8))[:V].astype(bool).reshape(DIMS)
    own = np.stack([np.where(plane(f'own_known_{a}'), plane(f'own_occ_{a}').astype(np.int8), np.int8(-1)) for a in range(m['agents'])])
    gt = {g: plane(f'gt_{g}') for g in ('occ', 'slab', 'obj')}
    return m, own, gt


def demo_task(name):
    path = os.path.join(DEMO_DIR, f'house_{name}.json')
    rooms = int(name.split('_')[0][1:]); index = int(name.split('_')[1][1:])
    try:
        m, own, gt = load_demo(path)
        fx = W.make_fixture('VALIDATION', index, rooms)
        res = dict(house=name, own=bool(np.array_equal(own, fx['private'])), occ=bool(np.array_equal(gt['occ'], fx['occ'].astype(bool))),
                   slab=bool(np.array_equal(gt['slab'], fx['slab'].astype(bool))), obj=bool(np.array_equal(gt['obj'], fx['obj'].astype(bool))),
                   edges=[list(e) for e in m['edges']] == [[int(i), int(j)] for i, j, _ in fx['edges']],
                   reached=bool(np.array_equal(np.asarray(m['reached'], bool), np.asarray(fx['reached'], bool))),
                   chart=abs(m['chart_m'] - fx['chart_m']) < 1e-9,
                   positions=bool(np.abs(np.asarray(m['positions_m']) - fx['positions_m']).max() <= 5.01e-4))
        res['ok'] = all(v for k, v in res.items() if k != 'house')
    except Exception as e:
        res = dict(house=name, ok=False, error=repr(e))
    return res


def _mem_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, 'peak_wset', mi.rss) / 2 ** 20, 1)
    except Exception:
        return None


def count_task(task):
    """TRAIN half: run every encoder on every message of every protocol (no bytes, no decoding) and return the context counts.
    An exception other than make_fixture's ValueError (e.g. the encoder-side exactness assertion) is returned as a crash."""
    scene, half, opts = task
    t0 = time.time()
    cache = None
    if opts.get('counts_dir'):                                          # resumable: per-house counts, reused only for the same code sha / wires
        cache = os.path.join(opts['counts_dir'], scene_key(scene).replace(':', '_') + '.npz')
        if os.path.exists(cache):
            try:
                z = np.load(cache)
                meta = json.loads(str(z['meta']))
                if meta['code_sha'] == CODE_SHA and meta['wires'] == list(opts.get('wires', WIRES)):
                    return dict(scene=scene, half=half, counts={m: z['c__' + m.replace('.', '__')] for m in SPECS}, seconds=0.0, timing='cached',
                                mem_mb=None, code_sha=CODE_SHA, house_id=meta['house_id'], render=None)
            except Exception:
                pass
    try:
        counts = zero_counts()
        h = House(scene, 'train' + half, priors=(), counts=counts, render_cache=not opts.get('no_render_cache'))
    except ValueError as e:
        return dict(scene=scene, half=half, skipped=str(e))
    except Exception:
        return dict(scene=scene, half=half, crash=traceback.format_exc()[-2000:])
    try:
        timing, _ = h.run(opts.get('wires', WIRES))
    except Exception:
        return dict(scene=scene, half=half, crash=traceback.format_exc()[-2000:])
    if cache:
        np.savez_compressed(cache, **{'c__' + m.replace('.', '__'): v for m, v in counts.items()},
                            meta=np.array(json.dumps(dict(code_sha=CODE_SHA, wires=list(opts.get('wires', WIRES)), house_id=h.house_id))))
    return dict(scene=scene, half=half, counts=counts, seconds=round(time.time() - t0, 1), timing=timing, mem_mb=_mem_mb(), code_sha=CODE_SHA,
                house_id=h.house_id, render=dict(hits=h.rcache.hits, misses=h.rcache.misses, seconds=round(h.rcache.seconds, 1)))


def eval_task(task):
    scene, panel, prior_paths, opts = task
    t0 = time.time()
    try:
        priors = [Prior.load(n, p) for n, p in prior_paths]
        h = House(scene, panel, priors=priors, fault=opts.get('fault'), render_cache=not opts.get('no_render_cache'))
    except ValueError as e:
        return dict(scene=scene, panel=panel, skipped=str(e))
    except Exception:
        return dict(scene=scene, panel=panel, init_crash=traceback.format_exc()[-2000:])
    try:
        timing, f1 = h.run(opts.get('wires', WIRES))
        err = None
    except Exception:
        timing, f1, err = {}, {}, traceback.format_exc()[-2000:]
    return dict(scene=scene, panel=panel, key=h.key, rows=h.rows, checks=h.checks, errors=h.errors[:50], n_errors=len(h.errors), crash=err,
                n_edges=len(h.edges), chart_m=h.chart_m, seconds=round(time.time() - t0, 1), fixture_s=round(h.t_fixture, 1), timing=timing,
                f1_sync_c_r8=f1, mem_mb=_mem_mb(), code_sha=CODE_SHA, house_id=h.house_id,
                render=dict(hits=h.rcache.hits, misses=h.rcache.misses, seconds=round(h.rcache.seconds, 1)))


def run_pool(fn, tasks, procs, log):
    out = []
    if procs <= 1:
        for t in tasks:
            r = fn(t); out.append(r); log(r)
        return out
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(fn, tasks):
            out.append(r); log(r)
    return out


# ============================================================================ summaries, factors, verdict
def _med(x):
    return float(np.median(x)) if len(x) else None


def summarise(rows, checks, houses, panel_names, prior_names):
    ci = {c: i for i, c in enumerate(MSG_COLS)}
    S = {}; F = {}; B = {}
    for panel in panel_names:
        for pn in prior_names:
            for wire in WIRES:
                R = [r for r in rows if r[ci['panel']] == panel and r[ci['prior']] == pn and r[ci['wire']] == wire]
                if not R:
                    continue
                sel = lambda proto, tier=None: [r for r in R if r[ci['proto']] == proto and (tier is None or r[ci['tier']] == tier)]
                all_h = sorted({r[ci['house']] for r in R})                 # a house without forest edges counts as 0 bytes per fusion
                per_house = lambda rr: [sum(r[ci['bytes']] for r in rr if r[ci['house']] == h) for h in all_h]
                k0, cav = sel('K0'), sel('CAV')
                cdu, cdc, syu, syc = sel('CD', 'u'), sel('CD', 'c'), sel('SYNC', 'u'), sel('SYNC', 'c')
                key = f'{panel}|{pn}|{wire}'
                S[key] = dict(
                    houses=len({r[ci['house']] for r in R}),
                    own_median=_med([r[ci['bytes']] for r in k0]), own_p90=float(np.percentile([r[ci['bytes']] for r in k0], 90)) if k0 else None,
                    cav_median=_med([r[ci['bytes']] for r in cav]), cav_p90=float(np.percentile([r[ci['bytes']] for r in cav], 90)) if cav else None,
                    converged_uncond_msg_median=_med([r[ci['bytes']] for r in cdu]),
                    cd_cond_msg_median=_med([r[ci['bytes']] for r in cdc]),
                    cd_distribute_cond_median=_med([r[ci['bytes']] for r in cdc if r[ci['round']] == 2]),
                    house_cd_cond_median=_med(per_house(cdc)), house_cd_uncond_median=_med(per_house(cdu)),
                    house_sync_cond_median=_med(per_house(syc)), house_sync_uncond_median=_med(per_house(syu)),
                    sync_cond_empty_messages=sum(1 for r in syc if r[ci['new_agents']] == 0), sync_cond_messages=len(syc),
                    residual_entries=int(sum(max(r[ci['residual']], 0) for r in R)),
                    messages=len(R), messages_exact=sum(1 for r in R if r[ci['exact']]))
                # per-round bytes (sum over directed edges, median over houses)
                S[key]['sync_round_bytes'] = {tier: [_med([sum(r[ci['bytes']] for r in rr if r[ci['house']] == h and r[ci['round']] == t)
                                                          for h in sorted({r[ci['house']] for r in rr})]) for t in range(1, SYNC_ROUNDS + 1)]
                                              for tier, rr in (('u', syu), ('c', syc))}
                # factor decomposition per house (protocol x coding x rounds)
                hs = sorted({r[ci['house']] for r in R})
                tot = lambda rr, h: sum(r[ci['bytes']] for r in rr if r[ci['house']] == h)
                fac = dict(rounds_resend_over_once=_med([tot(syu, h) / tot(cdu, h) for h in hs if tot(cdu, h)]),
                           cond_cd_uncond_over_cond=_med([tot(cdu, h) / tot(cdc, h) for h in hs if tot(cdc, h)]),
                           sync_cond_over_cd_cond=_med([tot(syc, h) / tot(cdc, h) for h in hs if tot(cdc, h)]),
                           sync_uncond_over_sync_cond=_med([tot(syu, h) / tot(syc, h) for h in hs if tot(syc, h)]))
                F[key] = fac
                # where the bits go
                def share(rr):
                    tb = np.array([[r[ci['bits_' + c]] for c in COMPS] for r in rr], np.float64)
                    if not len(tb):
                        return None
                    s = tb.sum(0); return {c: round(float(v / max(s.sum(), 1e-9)), 4) for c, v in zip(COMPS, s)} | {'total_bits': round(float(s.sum()), 1)}
                B[key] = dict(own=share(k0), cav=share(cav), cd_cond=share(cdc), sync_cond=share(syc))
            # wire ratios at the per-house CD-cond total
            for wire in ('W1', 'W2'):
                a, b = S.get(f'{panel}|{pn}|{wire}'), S.get(f'{panel}|{pn}|W3')
                if a and b and b['house_cd_cond_median']:
                    F[f'{panel}|{pn}|{wire}']['wire_over_W3_house_cd_cond'] = a['house_cd_cond_median'] / b['house_cd_cond_median']
    return S, F, B


PANELS = ('VAL', 'fresh'); HALVES = ('A', 'B')                       # the judged cells are FIXED (review M1)


def _rel(a, b):
    return abs(a - b) / min(a, b) if a and b else None


def verdict(S, all_exact, all_checks_ok, demo_ok, run=None):
    """D8-36 criteria on the fixed cells PANELS x HALVES. run = dict(limits, wires_all, skip_demo, fault, evaluated={panel: n},
    skipped={panel: n}); any missing cell / empty panel / limit / skipped demo check / fault -> INCOMPLETE (criterion (i) failures stay
    INCONCLUSIVE, which takes precedence). The literal reading (CAV, CD-c) decides; the A03-definition reading (uncond converged
    message, CD-u per house -- the measurements the 420 B / 6.5 KB thresholds were set on) is reported next to it (review M2)."""
    run = dict(run or {})
    cells = [(p, q) for p in PANELS for q in HALVES]
    per_arm = {}
    for wire in WIRES:
        ok = ok_a03 = True; vals = {}
        for p, q in cells:
            s = S.get(f'{p}|{q}|{wire}')
            if not s:
                ok = ok_a03 = False; continue
            c = dict(own=s['own_median'], cav=s['cav_median'], house=s['house_cd_cond_median'],
                     uncond_msg_median=s.get('converged_uncond_msg_median'), house_cd_uncond_median=s.get('house_cd_uncond_median'),
                     cd_distribute_cond_median=s.get('cd_distribute_cond_median'))
            vals[f'{p}|{q}'] = c
            ok &= all(c[k] is not None and c[k] <= THRESH[k] for k in ('own', 'cav', 'house'))
            ok_a03 &= (c['own'] is not None and c['own'] <= THRESH['own'] and c['uncond_msg_median'] is not None and c['uncond_msg_median'] <= THRESH['cav']
                       and c['house_cd_uncond_median'] is not None and c['house_cd_uncond_median'] <= THRESH['house'])
        halves, halves_a03 = {}, {}
        for p in PANELS:
            a, b = S.get(f'{p}|A|{wire}'), S.get(f'{p}|B|{wire}')
            if a and b:
                halves[p] = {k: _rel(a[k], b[k]) for k in ('own_median', 'cav_median', 'house_cd_cond_median')}
                halves_a03[p] = {k: _rel(a.get(k), b.get(k)) for k in ('converged_uncond_msg_median', 'house_cd_uncond_median')}
        h_ok = len(halves) == len(PANELS) and all(v is not None and v <= THRESH['halves_rel'] for d in halves.values() for v in d.values())
        rank = sum(v['house'] for v in vals.values()) if len(vals) == len(cells) and all(v['house'] is not None for v in vals.values()) else float('inf')
        per_arm[wire] = dict(ii=bool(ok), iii=bool(h_ok), ii_a03_definition=bool(ok_a03), cells=vals, halves_rel=halves,
                             halves_rel_a03_report_only=halves_a03, rank_sum_house=rank)
    best = min(WIRES, key=lambda w: per_arm[w]['rank_sum_house'])
    evaluated = run.get('evaluated', {}); skipped = run.get('skipped', {})
    reasons = []
    if run.get('limits'):
        reasons.append('--*-limit run')
    if not run.get('wires_all', True):
        reasons.append('not all wires')
    if run.get('skip_demo') or demo_ok is None:
        reasons.append('demo check skipped (empty)')
    if run.get('fault'):
        reasons.append('--fault run')
    for p in PANELS:
        if evaluated.get(p, 0) == 0:
            reasons.append(f'panel {p} has no evaluated house ({skipped.get(p, 0)} skipped)')
    if any(not math.isfinite(per_arm[w]['rank_sum_house']) for w in WIRES):
        reasons.append('missing panel x prior cells')
    def literal(w):
        a = per_arm[w]
        return 'PASS' if a['ii'] and a['iii'] else ('FAIL_ii' if not a['ii'] else 'FAIL_iii')
    def a03(w):
        a = per_arm[w]
        return 'PASS' if a['ii_a03_definition'] and a['iii'] else 'FAIL'
    i_ok = bool(all_exact and all_checks_ok and demo_ok is not False)
    provisional = literal(best)
    if not i_ok:
        v = 'INCONCLUSIVE'
    elif reasons:
        v = 'INCOMPLETE'
    else:
        v = provisional
    line = (f"{v}" + (f" (provisional reading {provisional})" if v == 'INCOMPLETE' else '') +
            f" | literal D8-36 reading (CAV, CD-c): {provisional}, best arm {best} | A03-definition reading (uncond converged message, CD-u per house): "
            f"{a03(best)} | if the arm were fixed to W3: literal {literal('W3')}, A03-definition {a03('W3')}")
    return dict(verdict=v, verdict_line=line, criterion_i=i_ok, incomplete_reasons=reasons, provisional_literal=provisional,
                a03_definition_reading=a03(best), fixed_W3=dict(literal=literal('W3'), a03_definition=a03('W3')),
                houses_evaluated=dict(evaluated), houses_skipped=dict(skipped),
                best_arm=best, best_arm_rule='lowest sum over the 4 panel x prior cells of the median per-house collect-distribute (tier c) bytes',
                arms=per_arm, thresholds=THRESH, cells=[f'{p}|{q}' for p, q in cells],
                definitions=dict(own='median K0 own-map message bytes (tier u: decoded from the own code alone)',
                                 cav='LITERAL: median over directed edges of m(u->v) conditioned on the cavity m(v->u) (CAV rows; a rate readout)',
                                 house='LITERAL: median over houses of the total bytes of one collect-distribute fusion, tier c (every directed edge once)',
                                 uncond_msg_median='A03 definition (report): median CD-u message = unconditioned sender-side union (A03 p9 edge), vs 420 B',
                                 house_cd_uncond_median='A03 definition (report): median per-house CD-u total (A03 6.1 KB measurement), vs 6,500 B',
                                 cd_distribute_cond_median='report: median of the realisable conditioned messages (CD tier c, distribute phase)',
                                 halves='|A - B| / min(A, B) of each of the three literal medians, per panel, for the best arm'),
                reading='PASS -> the S0-class exact code is locked as the lossless reference line OF THIS SYNTHETIC WORLD (W3\'s split constants and '
                        'box language are isomorphic to the generator; generalisation to other worlds is not covered by this criterion); FAIL_ii -> '
                        '~200 B was a small-panel artefact, raise the byte targets before any GPU; INCONCLUSIVE -> a bug (some decode was not '
                        'voxel-exact), fix before reading any number; INCOMPLETE -> not a verdict (smoke / partial / fault-injected run)')


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True); ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--val-limit', type=int); ap.add_argument('--fresh-limit', type=int); ap.add_argument('--train-limit', type=int)
    ap.add_argument('--wires', default=','.join(WIRES)); ap.add_argument('--skip-demo-check', action='store_true')
    ap.add_argument('--no-render-cache', action='store_true'); ap.add_argument('--fault', default=None, help='test hook, e.g. W1:decoder_drop_hit')
    args = ap.parse_args()
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    wires = tuple(w for w in args.wires.split(',') if w)
    opts = dict(wires=wires, no_render_cache=args.no_render_cache, counts_dir=os.path.join(out, 'train_counts'))
    os.makedirs(opts['counts_dir'], exist_ok=True)
    if args.fault:
        w_, k_ = args.fault.split(':'); opts['fault'] = dict(wire=w_, kind=k_)
    logf = open(os.path.join(out, 'run.log'), 'a', encoding='utf-8')
    T0 = time.time()
    def log(msg):
        line = f'[{time.time() - T0:7.0f}s] {msg}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'G4-E0 {VERSION} code sha256 {CODE_SHA} args {vars(args)}')
    P = panels(args.val_limit, args.fresh_limit, args.train_limit)
    log('scenes ' + ', '.join(f'{k} {len(v)}' for k, v in P.items()))
    demo = []
    if not args.skip_demo_check:
        demo = run_pool(demo_task, ['r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1'], args.procs, lambda r: log(f'demo {r}'))
    demo_ok = all(d['ok'] for d in demo) if demo else None
    # ---- priors from the TRAIN halves (counting mode: the same encoders, no bytes)
    priors = {}; train_crashes = []
    res_all = run_pool(count_task, [(s, half, opts) for half in ('A', 'B') for s in P['train' + half]], args.procs,
                       lambda r: log(f"train{r['half']} {scene_key(r['scene'])} " + (f"skipped ({r['skipped']})" if 'skipped' in r else
                                     f"CRASH {r['crash']}" if 'crash' in r else f"{r['seconds']}s {r['timing']} mem {r['mem_mb']} MB render {r['render']}")))
    for half in ('A', 'B'):
        res = sorted([r for r in res_all if r['half'] == half], key=lambda r: scene_key(r['scene']))
        counts = zero_counts(); used = []; ids = []
        for r in res:
            if 'counts' in r:
                for m in SPECS:
                    counts[m] += r['counts'][m]
                used.append(scene_key(r['scene'])); ids.append(r['house_id'])
            if 'crash' in r:
                train_crashes.append(dict(half=half, house=scene_key(r['scene']), crash=r['crash']))
        pr = Prior(half, counts, [dict(scene=s, house_id=i) for s, i in zip(used, ids)]); pth = os.path.join(out, f'prior_{half}.npz'); pr.save(pth)
        priors[half] = dict(path=pth, houses=used, house_ids=ids, skipped=[scene_key(r['scene']) for r in res if 'skipped' in r],
                            crashed=[scene_key(r['scene']) for r in res if 'crash' in r], symbols={m: int(counts[m].sum()) for m in SPECS},
                            npz_bytes=os.path.getsize(pth), table_entries=int(sum(np.prod(SPECS[m]) for m in SPECS)))
        log(f'prior {half}: {len(used)} houses, symbols ' + json.dumps({m: v for m, v in priors[half]['symbols'].items() if not m.startswith('w3u.') and not m.startswith('w3c.')}))
    prior_paths = [(n, priors[n]['path']) for n in ('A', 'B')]
    # ---- evaluation panels
    tasks = [(s, panel, prior_paths, opts) for panel in ('VAL', 'fresh') for s in P[panel]]
    res = run_pool(eval_task, tasks, args.procs, lambda r: log(
        f"{r['panel']} {scene_key(r['scene'])} " + (f"skipped ({r['skipped']})" if 'skipped' in r else f"INIT CRASH {r['init_crash']}" if 'init_crash' in r else
        f"{r['seconds']}s (fixture {r['fixture_s']}s) {r['timing']} errors {r['n_errors']} crash {bool(r['crash'])} mem {r['mem_mb']} MB render {r['render']}")))
    res = sorted(res, key=lambda r: (r['panel'], scene_key(r['scene'])))
    done = [r for r in res if 'rows' in r]
    init_crashes = [dict(panel=r['panel'], house=scene_key(r['scene']), crash=r['init_crash']) for r in res if 'init_crash' in r]
    rows = [row for r in done for row in r['rows']]; checks = [c for r in done for c in r['checks']]
    all_exact = (all(r[MSG_COLS.index('exact')] for r in rows) and not any(r['crash'] for r in done) and all(r['n_errors'] == 0 for r in done)
                 and not init_crashes and not train_crashes)
    all_checks_ok = all(c[CHK_COLS.index('exact')] and c[CHK_COLS.index('free_recall')] == 1.0 and c[CHK_COLS.index('fpr')] == 0.0 for c in checks)
    panel_names = [p for p in PANELS if any(r['panel'] == p for r in done)]
    S, F, B = summarise(rows, checks, done, panel_names, list(HALVES))
    evaluated = {p: sum(1 for r in done if r['panel'] == p) for p in PANELS}
    skipped_n = {p: sum(1 for r in res if r['panel'] == p and 'skipped' in r) for p in PANELS}
    run_flags = dict(limits=any(x is not None for x in (args.val_limit, args.fresh_limit, args.train_limit)), wires_all=set(wires) == set(WIRES),
                     skip_demo=bool(args.skip_demo_check), fault=bool(args.fault), evaluated=evaluated, skipped=skipped_n)
    V_ = verdict(S, all_exact, all_checks_ok, demo_ok, run_flags)
    if demo_ok is False:
        V_['verdict'] = 'INCONCLUSIVE'; V_['criterion_i'] = False; V_['demo_failure'] = True
    shas = sorted({r['code_sha'] for r in list(res_all) + list(res) if 'code_sha' in r} | {CODE_SHA})
    if len(shas) != 1:                                                  # the file changed while workers were being spawned
        V_['verdict'] = 'INCONCLUSIVE'; V_['criterion_i'] = False; V_['code_changed_during_run'] = shas
    # review S2: the two TRAIN halves and the evaluation panels must be disjoint HouseExpo houses (true by construction; kept as evidence)
    idA, idB = set(priors['A']['house_ids']), set(priors['B']['house_ids'])
    ideval = {p: sorted(r['house_id'] for r in done if r['panel'] == p) for p in PANELS}
    overlap = dict(A_B=sorted(idA & idB), train_eval=sorted((idA | idB) & set(ideval['VAL'] + ideval['fresh'])), VAL_fresh=sorted(set(ideval['VAL']) & set(ideval['fresh'])))
    V_['house_disjointness'] = dict(ok=not any(overlap.values()), overlaps=overlap)
    if any(overlap.values()):
        V_['verdict'] = 'INCONCLUSIVE'; V_['criterion_i'] = False
    V_['verdict_line'] = V_['verdict'] + V_['verdict_line'][V_['verdict_line'].index(' '):] if V_['verdict'] == 'INCONCLUSIVE' else V_['verdict_line']
    control = dict(CONTROL_3SX)
    for p in panel_names:
        ne = [r['n_edges'] for r in done if r['panel'] == p]
        control[f'{p}_bytes_per_house_median'] = _med([SYNC_ROUNDS * 2 * e * CONTROL_3SX['bytes_per_message'] for e in ne])
        for key in [k for k in F if k.startswith(p + '|')]:
            if S[key]['house_cd_cond_median']:
                F[key]['control_3SX_house_over_house_cd_cond'] = control[f'{p}_bytes_per_house_median'] / S[key]['house_cd_cond_median']
                F[key]['control_3SX_msg_over_own'] = CONTROL_3SX['bytes_per_message'] / S[key]['own_median']
    result = dict(batch='G4-E0', version=VERSION, preregistration='claude_handoff/OVERNIGHT_REPORT_20261021.md D8-36 (written 2026-10-27 02:20 UTC)',
                  written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), args=vars(args),
                  code_sha256=CODE_SHA, worker_code_sha256=shas,
                  notes=['priors are offline shared models fitted on the TRAIN half only; they are not counted in message bytes',
                         'every message byte count = varint length field + range-coded payload (agent mask, float32 poses, body, residual)',
                         'sensor height z = W.FLOOR_T + W.SENSOR_HEIGHT is a protocol constant (asserted equal to every generator pose)',
                         'decoding = W.observe (memoised exactly, see module docstring) from the decoded float32 poses, joined with the shared map',
                         '6.5 KB = 6,500 B; bits_* columns are ideal code lengths per component, bits_coder = 8 x payload bytes - their sum',
                         'SYNC tier u re-sends the full exclude-back message every round (rollout_max schedule); an empty tier-c message costs its 1-byte length field',
                         'W3 uses GENERATOR KNOWLEDGE: its plan split uses BAND = z %d-%d and LINTEL_MAX_FREE = %d (inherited from A03 p8: above every '
                         'furniture top of the generator, below the lowest ceiling; door tops 2.1-2.25 m), and its box language is isomorphic to the '
                         "generator's axis-aligned boxes. This affects bytes only, never exactness (paint never meets an observed free voxel). Reviewer "
                         'sensitivity (claude_handoff/analysis_20261027/g4/review_e0, 2 VAL houses, 3+3 TRAIN prior; reference only, NOT arms): '
                         'as run own 195/194.5 B, CAV 288.5/288 B, CD-c 4,537/4,452 B per house; noplan (no plan, boxes only) own 188.5/182.5, '
                         'CAV 303.5/306, CD-c 4,764/4,667; generic (BAND = full height, no lintel limit) own 221/222.5, CAV 334/331.5, CD-c 5,364/5,182 '
                         '(priors A/B)' % (BAND[0], BAND[1] - 1, LINTEL_MAX_FREE),
                         'the session grid cell = (L/128, L/128, 0.1) m (L = house-fitted chart) is a public session constant and is not counted in '
                         'message bytes (at most ~1 B per house)',
                         'prior size (offline, public, not counted): see priors[half].npz_bytes / table_entries'],
                  w3_constants=dict(BAND=list(BAND), LINTEL_MAX_FREE=LINTEL_MAX_FREE, source='A03 p8 (generator-aware)'),
                  demo_check=demo, demo_ok=demo_ok, scenes={k: [scene_key(s) for s in v] for k, v in P.items()},
                  house_ids=dict(trainA=priors['A']['house_ids'], trainB=priors['B']['house_ids'], **ideval),
                  skipped=[dict(panel=r['panel'], house=scene_key(r['scene']), reason=r['skipped']) for r in res if 'skipped' in r],
                  crashes=dict(train=train_crashes, eval_init=init_crashes),
                  priors=priors, all_messages_exact=all_exact, all_checks_ok=all_checks_ok,
                  houses=[{k: r[k] for k in ('panel', 'key', 'n_edges', 'chart_m', 'seconds', 'fixture_s', 'timing', 'mem_mb', 'render', 'n_errors',
                                             'errors', 'crash', 'f1_sync_c_r8')} for r in done],
                  summary=S, factors=F, bits=B, verdict=V_, control_3SX=control, message_columns=MSG_COLS, messages=rows,
                  check_columns=CHK_COLS, checks=checks, seconds=round(time.time() - T0, 1))
    with open(os.path.join(out, 'g4_e0.json'), 'w') as f:
        json.dump(result, f, separators=(',', ':'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log(f"verdict {V_['verdict_line']}")
    log(f"criterion i {V_['criterion_i']}; incomplete reasons {V_['incomplete_reasons']}; houses evaluated {evaluated} skipped {skipped_n}; "
        f"disjoint {V_['house_disjointness']['ok']}")
    for w in WIRES:
        for cell, c in V_['arms'][w]['cells'].items():
            log(f"report {w} {cell}: own {c['own']} | literal: CAV {c['cav']} house CD-c {c['house']} | A03 definition: uncond msg {c['uncond_msg_median']} "
                f"house CD-u {c['house_cd_uncond_median']} | realisable distribute msg {c['cd_distribute_cond_median']}")
    for k, s in S.items():
        log(f"{k}: own {s['own_median']} cav {s['cav_median']} house CD-c {s['house_cd_cond_median']} CD-u {s['house_cd_uncond_median']} "
            f"SYNC-c {s['house_sync_cond_median']} SYNC-u {s['house_sync_uncond_median']} exact {s['messages_exact']}/{s['messages']} residual {s['residual_entries']}")
    logf.close()
    return 0 if V_['criterion_i'] else 2


if __name__ == '__main__':
    sys.exit(main())
