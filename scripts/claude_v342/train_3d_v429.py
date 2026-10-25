"""train_3d_v429 (Claude, 2026-10-25, batch 3S-D: the code-layout / dimension ladder, user-approved to exceed 512 B per message for the
DIAGNOSTIC arms) = train_3d_v428 + (i) layouts 32x32x1 (D = 1,024: every coordinate owns an L/32 x L/32 column, 2 KB per message at
8+8 bits) and 16x16x2 (D = 512: two vertical layers per column, floor zone 0-1.6 m and upper zone); both need --allow-non256;
(ii) --wire-bits M,Q: the k-bit wire carries M-bit means (2^(M-1)-1 levels per side, also the source quantiser LEVELS) and Q-bit log
precisions (2^Q-1 levels over [1e-8, 128]); default 8,8 = v428; 16,16 doubles the bits per coordinate (1 KB at D = 256), 4,4 halves
them (512 B at D = 512); (iii) a code grid finer than the decoder's 16 x 16 start grid is FOLDED into it by space-to-depth (32x32x1
-> 16 x 16 x 1 with 2 x 2 x 2 = 8 channels), so the decoder's schedule, depth, channel plan and receptive field are exactly the
control's and only the first convolution's input width changes (review v429 M3); unchanged for BX, BY <= 16, so every earlier recipe
is bit-for-bit (test_3d_v429 A). The original docstring follows.
"""
"""train_3d_v428 (Claude, 2026-10-25, batch 3S-T) = train_3d_v427 + ONE knob, --init-from CKPT: a WARM START for training cycles under
separate leases (the 2D line's batches R / S / W / X protocol). The checkpoint's state_dict is loaded strictly (every tensor, same
shapes) after the model is built with the same seed; the optimiser, the schedule (warmup + cosine over --updates) and the data order
start fresh; the checkpoint must carry the same recipe (layout, decoder, width, precision, known-weight, D), asserted; run.json records
warm_start (path, sha256, source updates, source weights hash) and initial_weights_sha256 is the LOADED weights' hash. Without
--init-from the trainer is train_3d_v427 bit-for-bit (test_3d_v428). The original docstring follows.
"""
"""train_3d_v427 (Claude, 2026-10-25) = train_3d_v426 + ONE knob, --known-weight {uniform, boundary} (analyst memo
claude_handoff/analysis_20261025/analyst_B_training.md, 3.3 / 5.1). The known head vetoes the occupancy head (predict: known_logit < 0
-> unknown) and its BCE is a plain mean over all voxels, 87 % of which are unknown and most known ones interior free: the known /
unknown border, one voxel thick, is unweighted, so it jitters by +-1 voxel and 13-17 % of the target-known OCCUPIED voxels are vetoed
(the surface IS the border of the known region). boundary = the known-BCE gets weight 1 + 4 x [boundary voxel] with boundary = target
surface voxels (known side) | unknown voxels 6-adjacent to a known voxel (unknown side), as a weighted mean -- mirroring the occupancy
term's surface weight, two-sided so the border sharpens without moving. uniform = v426 bit-for-bit (test_3d_v427). The original
docstring follows.
"""
"""train_3d_v426 (Claude, 2026-10-25) = train_3d_v425 + ONE knob, --precision {learned, coverage} (analyst memo
claude_handoff/analysis_20261025/analyst_A_representation.md, 6.1). learned = v420: q = 1e-4 + (4 - 1e-4) sigmoid(head), a learned
per-block precision that is at once the join's source-selection key and the decoder's second channel; measured on 3S-W / 3S-C it does
not rank sources by completeness (the max-precision winner is the best-coverage source in only 58 % / 37 % of contested blocks), and
re-keying the join by coverage at evaluation time recovered 76 % / 90 % of the R8 - K0 gap. coverage = DETERMINISTIC precision
q = PRECISION_MAX x (fraction of the block's voxels the agent knows), so the lattice join picks the source that saw the block most
completely and the decoder reads the exact block coverage from log(1 / p); the wire, join, decoder, loss and data are unchanged, and
the head keeps its 2-output shape (the q output is simply unused: identical initial tensors; 129 parameters get no gradient). The 2D
line has the same mode (train_v378 PRECISION_MODE = coverage). With --precision learned the model is train_3d_v425 bit-for-bit
(test_3d_v426). The original docstring follows.
"""
"""train_3d_v425 (Claude, 2026-10-25, batch 3S-P) = train_3d_v424 + ONE knob, --decoder {d3, plan_column}.
d3 = the 3D Decoder3D of v420 / v422 (bit-for-bit, test_3d_v425 A). plan_column = the 2.5-D PLAN + COLUMN decoder (analyst memo
claude_handoff/analysis_20261025/analyst_C_architecture.md, R1): the fused posterior on the BX x BY x BZ block grid is read as a 2-D
map with 2 BZ channels (every vertical layer's mu and log var / 8); a 2-D convolutional trunk shaped like the 2-D line's working
decoder (train_v392 --decoder-width W --frame-cond: pre 64W, lift 128W, stages 96W / 128W... public planes x, y, L / 25.6 and the
block phase u, v at every stage) draws the floor plan at full horizontal resolution NX x NY; the last 1x1 convolution emits 2 NZ
channels = the known / occupancy logits of the whole vertical COLUMN at (x, y). Reads only the posterior + public coordinates, like
d3. Walls are vertical extrusions of the plan and column profiles are low-dimensional, so the 2-D trunk spends its capacity on the
plan (where the information is) instead of redrawing it on every z slice with 3x3x3 kernels; ~0.1x the MACs of d3 at W = 2.
The original v424 / v420 docstring follows.
"""
"""train_3d_v424 (Claude, 2026-10-25) = train_3d_v420 with the two knobs of v422 and v423 together: --decoder-width W (every decoder
channel x W; v422) and --layout 8x8x4 | 16x16x1 (public block grid of the D = 256 code; v423), plus the DIAGNOSTIC layout 16x16x4
(D = 1,024; violates the 256-dim rule; needs --allow-non256; representation-ceiling arm only). With W = 1 and 8x8x4 the model is
train_3d_v420 bit-for-bit; with W = 2 it is train_3d_v422; with 16x16x1 it is train_3d_v423 (test_3d_v424). D follows the layout.
The original v420 docstring follows.
"""
"""3D Sheaf-GBP storey trainer v420 (Claude, 2026-10-25). train_3d_v401 moved from a cube to a SLAB: one furnished storey of the
v420 world (mapping_3d_storey_data_v420) on an NX x NY x NZ voxel grid (R1 = 128 x 128 x 32; cells L/128 horizontally, 0.1 m
vertically, L = the house-fitted chart side). Design: claude_handoff/DESIGN_3D_STOREY_V420.md.
Unchanged from v400 / v401 (Sheaf-GBP): each agent's map -> D = 256 Gaussian variables + 256 diagonal precisions, one per public block
of an 8 x 8 x 4 grid (L/8 x L/8 x 0.8 m); block-local encoder; mu tanh hard-8, q in [1e-4, 4] x block coverage; 8+8-bit k-bit wire;
per-coordinate max-precision selection (lattice join) over 8 synchronous rounds on the radio forest; a decoder that reads ONLY the
fused posterior; stream targets per hop; loss = known BCE + occupancy BCE on target-known voxels with weight 1 + 4 x [surface].
Slab changes (the only ones): (i) every convolution runs on the COMPUTE grid NX/s x NY/s x NZ/s (s = 2: 64 x 64 x 16) and the voxel
grid is reached by parameter-free space-to-depth / depth-to-space (the v401 block readout, s = 2 as in batch 3D-B); (ii) the public
coordinate planes carry a fourth channel L / 25.6 (the chart scale: horizontal cells differ per house -- the 3D counterpart of the 2D
public-frame conditioning); (iii) the decoder schedule ends at the compute grid's anisotropic shape; (iv) metrics take the scene's own
slab mask (floor + ceiling) for the variable-structure F1, the furniture-region F1 (review v420 M1) and the furniture recall.
"""
import argparse, hashlib, json, math, multiprocessing, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util
_spec = importlib.util.spec_from_file_location('train_v378_kbit', HERE / 'train_v378.py'); _v378 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v378)
kbit_wire_quantize = _v378.kbit_wire_quantize

HOPS = (0, 1, 2, 4, 8)
WEIGHTS = {0: .5, 1: .125, 2: .125, 4: .125, 8: .125}
BX, BY, BZ = 8, 8, 4                                                  # v424: set from --layout in main()
D = BX * BY * BZ                                                      # v424: follows the layout
LAYOUTS = {'8x8x4': (8, 8, 4), '16x16x1': (16, 16, 1), '16x16x4': (16, 16, 4), '32x32x1': (32, 32, 1), '16x16x2': (16, 16, 2)}
DIAGNOSTIC_LAYOUTS = ('16x16x4', '32x32x1', '16x16x2')
PRECISION_MIN, PRECISION_MAX = 1e-4, 4.
LATENT_PRIOR = 1e-4
LEVELS = 127                                                          # v429: 2^(MU_BITS-1) - 1, set from --wire-bits in main()
WIRE_BITS = (8, 8)                                                    # v429: (mean bits, precision bits) on the wire; set from --wire-bits
SUB = 2
DECODER_WIDTH = 1.0                                                   # v424: set from --decoder-width in main()
DECODER = 'd3'                                                        # v425: set from --decoder in main()
PRECISION = 'learned'                                                 # v426: set from --precision in main()
KNOWN_WEIGHT = 'uniform'                                              # v427: set from --known-weight in main()
CHART_REF = 25.6
CHART_M = 12.8                                                        # set per scene by the loops (public chart side)
STOP = False


def _sigterm(*_):
    global STOP
    STOP = True


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def latent_quantize(z):
    zc = z.clamp(-1, 1)
    return (torch.round(zc * LEVELS) / LEVELS).detach() + (zc - zc.detach())


def space_to_depth(x, s):
    """[N, C, X, Y, Z] -> [N, C s^3, X/s, Y/s, Z/s]; channel c s^3 + dx s^2 + dy s + dz at (i, j, k) holds voxel (s i + dx, s j + dy, s k + dz)."""
    if s == 1:
        return x
    n, c, X, Y, Z = x.shape
    return x.reshape(n, c, X // s, s, Y // s, s, Z // s, s).permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(n, c * s ** 3, X // s, Y // s, Z // s)


def depth_to_space(x, s):
    if s == 1:
        return x
    n, cs, a, b, c3 = x.shape; c = cs // s ** 3
    return x.reshape(n, c, s, s, s, a, b, c3).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(n, c, a * s, b * s, c3 * s)


def public3(count, sx, sy, sz, device):
    """cell-centre coordinates in [-1, 1] and the chart scale plane L / 25.6 (CHART_M of the current scene)."""
    x = (torch.arange(sx, device=device) + .5) / sx * 2 - 1
    y = (torch.arange(sy, device=device) + .5) / sy * 2 - 1
    z = (torch.arange(sz, device=device) + .5) / sz * 2 - 1
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    scale = torch.full_like(xx, CHART_M / CHART_REF)
    return torch.stack((xx, yy, zz, scale))[None].expand(count, -1, -1, -1, -1)


class PatchTrunk3D(nn.Module):
    """strictly block-local on the compute grid: each 8x8x4 public block (16 x 16 x 8 voxels at R1) is encoded independently."""
    def __init__(self, dims, width=128):
        super().__init__()
        self.dims = tuple(dims); self.comp = tuple(d // SUB for d in dims); self.bs = (self.comp[0] // BX, self.comp[1] // BY, self.comp[2] // BZ)
        assert all(d % (SUB * b) == 0 for d, b in zip(dims, (BX, BY, BZ))), dims
        self.net = nn.Sequential(nn.Conv3d(3 * SUB ** 3 + 4, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv3d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv3d(96, width, 3, stride=2, padding=1), nn.SiLU())
        self.proj = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())
        self.width = width

    def forward(self, private):                                       # [N, NX, NY, NZ] int8
        n = private.shape[0]
        assert tuple(private.shape[1:]) == self.dims, (private.shape, self.dims)
        cls = torch.cat(((private >= 0).float()[:, None], (private == 0).float()[:, None], (private == 1).float()[:, None]), 1)
        pub = public3(1, *self.comp, private.device)[0]
        inp = torch.cat((space_to_depth(cls, SUB), pub[None].expand(n, -1, -1, -1, -1)), 1)
        c = inp.shape[1]; bx, by, bz = self.bs
        p = inp.unfold(2, bx, bx).unfold(3, by, by).unfold(4, bz, bz)
        p = p.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(n * D, c, bx, by, bz)
        f = self.net(p)
        f = torch.cat((f.mean((2, 3, 4)), f.amax((2, 3, 4))), 1)
        return self.proj(f).reshape(n, D, self.width)


class OwnedHead(nn.Module):
    def __init__(self, width, dims):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.k = (dims[0] // BX, dims[1] // BY, dims[2] // BZ)

    def forward(self, feats, private):
        out = self.mlp(feats)
        mu = out[..., 0].tanh()
        q = PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * out[..., 1].sigmoid()
        coverage = F.avg_pool3d((private >= 0).float()[:, None], self.k).flatten(1)
        if PRECISION == 'coverage':                                        # v426: deterministic completeness key (analyst A 6.1)
            q = PRECISION_MAX * coverage.clamp_min(PRECISION_MIN / PRECISION_MAX)
        return mu, q, (coverage > 0).float()


class Decoder3D(nn.Module):
    """posterior on the 8x8x4 block grid -> known / occupancy logits on the voxel grid (convolutions end at the compute grid)."""
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims); comp = tuple(d // SUB for d in dims); self.start = (min(16, comp[0]), min(16, comp[1]), min(8, comp[2]))
        c0 = int(round(64 * DECODER_WIDTH))                              # v422: decoder channel multiplier (64 at W = 1)
        self.fold = max(1, BX // 16); assert BX == BY and BX % 16 == 0 or BX < 16, (BX, BY)   # v429: 32x32 -> fold 2
        self.pre = nn.Sequential(nn.Conv3d(2 * self.fold ** 2, c0, 3, padding=1), nn.SiLU(), nn.Conv3d(c0, c0, 3, padding=1), nn.SiLU())
        shapes, chans = [], []
        sx, sy, sz = self.start; c = 64
        while (sx, sy, sz) != comp:
            sx = min(sx * 2, comp[0]); sy = min(sy * 2, comp[1]); sz = min(sz * 2, comp[2])
            shapes.append((sx, sy, sz)); chans.append(max(16, c * 3 // 4)); c = chans[-1]
        chans = [int(round(ch * DECODER_WIDTH)) for ch in chans]          # v422: 48 / 36 at W = 1
        self.shapes = shapes
        blocks, incoming = [], c0
        for ch in chans:
            blocks.append(nn.Sequential(nn.Conv3d(incoming + 4, ch, 3, padding=1), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1), nn.SiLU()))
            incoming = ch
        self.blocks = nn.ModuleList(blocks)
        self.lift = nn.Sequential(nn.Conv3d(c0 + 4, c0, 3, padding=1), nn.SiLU())
        self.output = nn.Conv3d(incoming, 2 * SUB ** 3, 1)

    def forward(self, post):
        b = post.shape[0]
        grid = torch.cat((post[:, :D].reshape(b, 1, BX, BY, BZ), (post[:, D:] / 8.).reshape(b, 1, BX, BY, BZ)), 1)
        if self.fold > 1:                                                 # v429: space-to-depth in x, y onto the 16 x 16 start grid
            f = self.fold; grid = grid.reshape(b, 2, BX // f, f, BY // f, f, BZ).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, 2 * f * f, BX // f, BY // f, BZ)
        h = F.interpolate(self.pre(grid), self.start, mode='trilinear', align_corners=False)
        h = self.lift(torch.cat((h, public3(b, *self.start, h.device)), 1))
        for shape, block in zip(self.shapes, self.blocks):
            h = F.interpolate(h, shape, mode='trilinear', align_corners=False)
            h = block(torch.cat((h, public3(b, *shape, h.device)), 1))
        out = depth_to_space(self.output(h), SUB)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


def public2(count, sx, sy, device):
    """[count, 5, sx, sy]: cell-centre x, y in [-1, 1], the chart scale L / 25.6, and the block phase u, v in [-0.5, 0.5) on the BX x BY public grid."""
    x = (torch.arange(sx, device=device) + .5) / sx
    y = (torch.arange(sy, device=device) + .5) / sy
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    scale = torch.full_like(xx, CHART_M / CHART_REF)
    u = torch.frac(xx * BX) - .5; v = torch.frac(yy * BY) - .5
    return torch.stack((xx * 2 - 1, yy * 2 - 1, scale, u, v))[None].expand(count, -1, -1, -1)


class PlanColumnDecoder(nn.Module):
    """v425 plan + column decoder: posterior on the BX x BY x BZ grid -> 2-D map [b, 2 BZ, BX, BY] -> 2-D trunk at 16 -> ... -> NX x NY
    (public planes x, y, L / 25.6, u, v concatenated at every stage) -> 1x1 conv to 2 NZ channels = the column's known / occupancy logits."""
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims); nx, ny, nz = self.dims; assert nx == ny and nx >= 16 and nx % 16 == 0, dims
        W = DECODER_WIDTH; c0 = int(round(64 * W)); lift = int(round(128 * W))
        widths = [int(round(w * W)) for w in (96, 64, 48, 32)] + [int(round(32 * W))] * int(math.log2(nx // 128))
        self.pre = nn.Sequential(nn.Conv2d(2 * BZ, c0, 3, padding=1), nn.SiLU(), nn.Conv2d(c0, c0, 3, padding=1), nn.SiLU())
        self.lift = nn.Sequential(nn.Conv2d(c0 + 5, lift, 3, padding=1), nn.SiLU(), nn.Conv2d(lift, lift, 3, padding=1), nn.SiLU())
        blocks, incoming = [], lift
        for w in widths:
            blocks.append(nn.Sequential(nn.Conv2d(incoming + 5, w, 3, padding=1), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1), nn.SiLU())); incoming = w
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(incoming, 2 * nz, 1)

    def forward(self, post):
        b = post.shape[0]; nx, ny, nz = self.dims
        mu = post[:, :D].reshape(b, BX, BY, BZ); lv = (post[:, D:] / 8.).reshape(b, BX, BY, BZ)
        grid = torch.cat((mu.permute(0, 3, 1, 2), lv.permute(0, 3, 1, 2)), 1)                       # [b, 2 BZ, BX, BY]
        h = F.interpolate(self.pre(grid), (16, 16), mode='bilinear', align_corners=False)
        h = self.lift(torch.cat((h, public2(b, 16, 16, h.device)), 1))
        extent = 16
        for block in self.blocks:
            h = F.interpolate(h, (extent, extent), mode='bilinear', align_corners=False)
            h = block(torch.cat((h, public2(b, extent, extent, h.device)), 1))
            extent = min(extent * 2, nx)
        out = self.output(h).reshape(b, 2, nz, nx, ny).permute(0, 1, 3, 4, 2)                        # channel t nz + z -> (t, x, y, z)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


class SheafModel3D(nn.Module):
    def __init__(self, dims, seed):
        super().__init__()
        self.dims = tuple(dims)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.trunk = PatchTrunk3D(dims); self.head = OwnedHead(self.trunk.width, dims)
            self.decoder = PlanColumnDecoder(dims) if DECODER == 'plan_column' else Decoder3D(dims)   # v425 (trunk / head drawn first: identical for every decoder)

    def source(self, private):
        mu, q, mask = self.head(self.trunk(private), private)
        return latent_quantize(mu), q, mask

    def parameter_groups(self):
        return dict(trunk=list(self.trunk.parameters()), head=list(self.head.parameters()), decoder=list(self.decoder.parameters()))

    def weights_hash(self):
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


# ----------------------------------------------------------------------------- max-precision join on the forest (unchanged from v400)
def select(p_list, e_list):
    P = torch.stack(p_list, 0); E = torch.stack(e_list, 0)
    tie = torch.arange(P.shape[0], device=P.device, dtype=P.dtype).view(-1, 1) * 1e-15
    idx = (P - tie).argmax(0, keepdim=True)
    return P.gather(0, idx)[0], E.gather(0, idx)[0]


def rollout_max(mu, q_learned, mask, edges, rounds=8, stages=HOPS):
    n = mu.shape[0]
    prior = LATENT_PRIOR / (n * D)
    evidence_q = mask.double() * q_learned.double()
    p_own = evidence_q + prior; eta_own = evidence_q * mu.double()
    graph = [(min(i, j), max(i, j), w) for i, j, w in edges]
    directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
    incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
    msg_p = [torch.zeros(D, dtype=torch.float64, device=mu.device) for _ in directed]
    msg_e = [torch.zeros(D, dtype=torch.float64, device=mu.device) for _ in directed]
    p_wire, e_wire, _ = kbit_wire_quantize(p_own, eta_own)
    def belief():
        ps, es = [], []
        for j in range(n):
            pj, ej = select([p_own[j]] + [msg_p[a] for a in incoming[j]], [eta_own[j]] + [msg_e[a] for a in incoming[j]])
            ps.append(pj); es.append(ej)
        P = torch.stack(ps); E = torch.stack(es)
        return torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1)
    feats = {}
    if 0 in stages:
        feats[0] = belief()
    for r in range(1, rounds + 1):
        new_p, new_e = [], []
        for a, (i, j) in enumerate(directed):
            pc, ec = select([p_wire[i]] + [msg_p[b] for b in incoming[i] if directed[b][0] != j], [e_wire[i]] + [msg_e[b] for b in incoming[i] if directed[b][0] != j])
            new_p.append(pc); new_e.append(ec)
        msg_p, msg_e = new_p, new_e
        if r in stages:
            feats[r] = belief()
    return feats


# ----------------------------------------------------------------------------- targets, loss, metrics
def stream_targets(own, reached):
    n = own.shape[0]; flat = own.reshape(n, -1)
    minus = torch.full((), -1, dtype=flat.dtype, device=flat.device)
    return torch.where(reached[:, :, None], flat[None].expand(n, -1, -1), minus).amax(1)


def _pool6(m, dims):
    m = m.reshape(-1, 1, *dims).float()
    return ((F.max_pool3d(m, (3, 1, 1), stride=1, padding=(1, 0, 0)) + F.max_pool3d(m, (1, 3, 1), stride=1, padding=(0, 1, 0)) + F.max_pool3d(m, (1, 1, 3), stride=1, padding=(0, 0, 1))) > 0)


def dilate6(mask_flat, dims):
    return _pool6(mask_flat, dims).reshape(mask_flat.shape[0], -1)


def surface_of(target_flat, dims):
    occ = (target_flat == 1)
    return occ & dilate6(target_flat == 0, dims)


def stage_loss(decoder, feats, target, dims, w_surface=4.):
    known_logit, occ_logit = decoder(feats)
    known = (target >= 0).float()
    if KNOWN_WEIGHT == 'boundary':                                     # v427: sharpen the known / unknown border (analyst B 3.3)
        border = surface_of(target, dims) | (dilate6(target >= 0, dims) & (target < 0))
        wk = 1. + w_surface * border.float()
        known_bce = (wk * F.binary_cross_entropy_with_logits(known_logit, known, reduction='none')).sum(-1) / wk.sum(-1).clamp_min(1)
    else:
        known_bce = F.binary_cross_entropy_with_logits(known_logit, known, reduction='none').mean(-1)
    occ = (target == 1).float()
    weight = known * (1. + w_surface * surface_of(target, dims).float())
    occ_bce = (weight * F.binary_cross_entropy_with_logits(occ_logit, occ, reduction='none')).sum(-1) / weight.sum(-1).clamp_min(1)
    return known_bce.mean() + occ_bce.mean(), float(known_bce.mean().detach()), float(occ_bce.mean().detach())


@torch.no_grad()
def predict(decoder, feats):
    known_logit, occ_logit = decoder(feats)
    return torch.where(known_logit >= 0, (occ_logit >= 0).to(torch.int8), torch.full_like(occ_logit, -1, dtype=torch.int8))


METRIC_KEYS = ('occ_f1', 'surface_f1', 'surface_f1_tol1', 'var_occ_f1', 'furn_f1', 'obj_recall', 'accuracy', 'unknown_known_fpr')


def _f1(a, b):
    tp = (a & b).sum().item(); fp = (a & ~b).sum().item(); fn = (~a & b).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None


@torch.no_grad()
def metrics3d(pred, target, dims, slab, obj, struct):
    """per node, target-known scope: occupied F1; surface F1 (6-nbhd surface of the target; interior predictions ignored); tolerance-1
    surface F1 (v401 review-M1 definition); variable-structure F1 (the scene's floor + ceiling slab removed from both sides); furniture
    recall (target-occupied furniture voxels predicted occupied); FURNITURE-REGION F1 (review v420 M1: F1 restricted to the voxels that are
    neither slab nor wall / lintel -- the structure dilated by one voxel, so +-1-voxel wall jitter is not counted as furniture -- where only
    furniture can be occupied: missing and invented furniture both count); 3-way accuracy; unknown-known FPR. slab, obj, struct: flat bool [V]."""
    rows = []
    surf = surface_of(target, dims)
    region = ~slab & ~dilate6(struct[None], dims)[0]
    for i in range(pred.shape[0]):
        p, t, s = pred[i], target[i], surf[i]
        known = t >= 0
        if known.sum() == 0:
            rows.append({k: None for k in METRIC_KEYS}); continue
        po = (p == 1) & known; to = (t == 1) & known
        ps = po & ~((t == 1) & ~s)
        po_any = (p == 1) & dilate6(known[None], dims)[0]; po1 = po_any & ~((t == 1) & ~s)
        if po_any.any() and s.any():
            prec1 = float((po1 & dilate6(s[None], dims)[0]).sum()) / float(po1.sum()) if po1.any() else 1.
            rec1 = float((s & dilate6(po_any[None], dims)[0]).sum()) / float(s.sum())
            tol1 = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-12)
        else:
            tol1 = 0. if (po_any.any() or s.any()) else None
        to_obj = to & obj
        rows.append(dict(occ_f1=_f1(po, to), surface_f1=_f1(ps, s), surface_f1_tol1=tol1, var_occ_f1=_f1(po & ~slab, to & ~slab), furn_f1=_f1(po & region, to & region),
                         obj_recall=(float((po & to_obj).sum()) / float(to_obj.sum())) if to_obj.any() else None,
                         accuracy=float((p[known] == t[known]).float().mean()), unknown_known_fpr=float((p[~known] >= 0).float().mean()) if (~known).any() else None))
    return rows


def agg(rows):
    out = {}
    for k in METRIC_KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


# ----------------------------------------------------------------------------- main
def _fixture(spec):
    sys.path.insert(0, os.environ.get('SEALED_SOURCE', '')); sys.path.insert(0, os.path.join(os.environ.get('SEALED_SOURCE', ''), 'exploration_grid'))
    import mapping_3d_storey_data_v420 as W
    try:
        fx = W.make_fixture(spec['split'], spec['index'], spec['rooms'], tuple(spec['dims']), spec['agents'])
    except ValueError:                                                      # agents cannot be placed: the scene is left out (deterministic)
        return None
    return dict(spec=fx['spec'], private=fx['private'], reached=fx['reached'], edges=fx['edges'], slab=fx['slab'], obj=fx['obj'],
                struct=fx['occ'].astype(bool) & ~fx['obj'] & ~fx['slab'], chart_m=fx['chart_m'])


def main():
    global CHART_M, DECODER_WIDTH, BX, BY, BZ, D, DECODER, PRECISION, KNOWN_WEIGHT, LEVELS, WIRE_BITS
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True); ap.add_argument('--source', required=True, help='sealed source (HouseExpo v365 / v371 helpers)')
    ap.add_argument('--dims', default='128,128,32'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=342101); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--updates', type=int, default=16000); ap.add_argument('--val-every', type=int, default=4000); ap.add_argument('--val-per-stratum', type=int, default=8)
    ap.add_argument('--deadline-epoch', type=float, default=1e18); ap.add_argument('--deadline-total', type=float, default=1e18)
    ap.add_argument('--max-updates', type=int, default=0); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--gen-workers', type=int, default=8); ap.add_argument('--gen-timeout', type=float, default=1800.)
    ap.add_argument('--decoder-width', type=float, default=1.0, help='v422: channel multiplier of the decoder only (1.0 = v420)')
    ap.add_argument('--layout', choices=sorted(LAYOUTS), default='8x8x4', help='v423: public block grid of the code')
    ap.add_argument('--allow-non256', action='store_true', help='DIAGNOSTIC ONLY: allow a layout with D != 256 (violates the 256-dim rule)')
    ap.add_argument('--decoder', choices=('d3', 'plan_column'), default='d3', help='v425: 3-D decoder (v420) or the 2.5-D plan + column decoder')
    ap.add_argument('--precision', choices=('learned', 'coverage'), default='learned', help='v426: learned per-block precision (v420) or deterministic coverage precision')
    ap.add_argument('--known-weight', choices=('uniform', 'boundary'), default='uniform', help='v427: known-BCE weighting (boundary: 1 + 4 x [known / unknown border])')
    ap.add_argument('--init-from', default='', help='v428: warm start -- load this checkpoint (same recipe) and train a fresh schedule from it')
    ap.add_argument('--wire-bits', default='8,8', help='v429: mean,precision bits per coordinate on the wire (8,8 = v420; 16,16; 4,4)')
    args = ap.parse_args()
    DECODER_WIDTH = args.decoder_width; assert DECODER_WIDTH > 0; DECODER = args.decoder; PRECISION = args.precision; KNOWN_WEIGHT = args.known_weight
    WIRE_BITS = tuple(int(v) for v in args.wire_bits.split(',')); assert len(WIRE_BITS) == 2 and all(2 <= b <= 16 for b in WIRE_BITS), args.wire_bits
    _v378.MU_BITS, _v378.Q_BITS = WIRE_BITS; LEVELS = 2 ** (WIRE_BITS[0] - 1) - 1                      # v429: the wire and the source quantiser follow --wire-bits
    BX, BY, BZ = LAYOUTS[args.layout]; D = BX * BY * BZ
    assert D == 256 or (args.allow_non256 and args.layout in DIAGNOSTIC_LAYOUTS), 'D != 256 needs --allow-non256 (diagnostic only)'
    signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGINT, _sigterm)
    torch.set_num_threads(args.threads)
    os.environ['SEALED_SOURCE'] = args.source
    sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid'))
    import mapping_3d_storey_data_v420 as W
    dims = tuple(int(x) for x in args.dims.split(',')); V = int(np.prod(dims))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir(); (out / 'validation').mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    cache = Path(args.cache); man = json.load(open(cache / 'manifest.json'))
    assert tuple(man['dims']) == dims and man['version'] == W.VERSION and man['hops'] == list(HOPS) and all(r['spec']['split'] == 'TRAIN' for r in man['rows'])
    assert man['data_module_sha256'] == sha_file(W.__file__), 'cache written by a different data module'
    vplan = W.scene_plan('VALIDATION', 2 * args.val_per_stratum, dims)
    val_specs = [s for s in vplan if s['index'] < args.val_per_stratum]; fresh_specs = [s for s in vplan if s['index'] >= args.val_per_stratum]
    check = man['rows'][:2] + man['rows'][-2:]
    specs = [r['spec'] for r in check] + val_specs + fresh_specs
    with multiprocessing.get_context('spawn').Pool(min(args.gen_workers, len(specs))) as pool:   # spawn + timeout (incident D6-3)
        gen = pool.map_async(_fixture, specs, chunksize=1).get(timeout=args.gen_timeout)
    for r, fx in zip(check, gen[:len(check)]):
        assert fx is not None and W.array_hash(fx['private']) == r['private_sha256'], f"cache row {r['file']} does not regenerate"
    def panel(fxs):
        fxs = [fx for fx in fxs if fx is not None]
        return [dict(spec=fx['spec'], own_t=torch.from_numpy(fx['private']).to(device), reached_t=torch.from_numpy(fx['reached']).to(device), edges=fx['edges'],
                     slab=torch.from_numpy(fx['slab'].reshape(-1)).to(device), obj=torch.from_numpy(fx['obj'].reshape(-1)).to(device),
                     struct=torch.from_numpy(fx['struct'].reshape(-1)).to(device), chart_m=float(fx['chart_m'])) for fx in fxs]
    skipped_panels = [sp for sp, fx in zip(val_specs + fresh_specs, gen[len(check):]) if fx is None]          # review v420 S7
    val = panel(gen[len(check):len(check) + len(val_specs)]); fresh = panel(gen[len(check) + len(val_specs):]); del gen
    train = []
    for r in man['rows']:
        z = np.load(cache / r['file'])
        train.append(dict(spec=r['spec'], own_t=torch.from_numpy(z['private']), reached_t=torch.from_numpy(z['reached']), edges=[(int(i), int(j), float(w)) for i, j, w in z['edges']], chart_m=float(z['chart_m'])))
    print(f'[3s] scenes train {len(train)} val {len(val)} fresh {len(fresh)} gen {time.time()-t0:.0f}s', flush=True)

    model = SheafModel3D(dims, args.seed).to(device)
    warm_start = None
    if args.init_from:                                                    # v428
        payload = torch.load(args.init_from, map_location='cpu', weights_only=False); pm = payload.get('meta', {})
        for key, want in (('blocks', [BX, BY, BZ]), ('D', D), ('decoder', DECODER), ('decoder_width', DECODER_WIDTH), ('precision', PRECISION), ('known_weight', KNOWN_WEIGHT), ('dims', list(dims)), ('version', W.VERSION), ('wire_bits', list(WIRE_BITS))):
            assert pm.get(key, want) == want, f'warm start {key} mismatch: checkpoint {pm.get(key)} vs run {want}'
        model.load_state_dict(payload['state_dict'], strict=True)
        warm_start = dict(init_from=str(args.init_from), init_file_sha256=sha_file(args.init_from), source_updates=payload.get('updates'), source_weights_sha256=payload.get('weights_sha256'), source_seed=pm.get('seed'))
        print(f"[3s] warm start from {args.init_from} (updates {payload.get('updates')})", flush=True)
    groups = model.parameter_groups()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    total = args.updates
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1., (u + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(u, total) / max(1, total)))))
    meta = dict(version=W.VERSION, dims=list(dims), sub=SUB, agents=W.AGENTS, seed=args.seed, D=D, blocks=[BX, BY, BZ], beams=W.BEAMS, decoder_width=DECODER_WIDTH, layout=args.layout, decoder=DECODER, precision=PRECISION, known_weight=KNOWN_WEIGHT, kbit_source_sha256=sha_file(HERE / 'train_v378.py'),
                parameters={k: sum(p.numel() for p in v) for k, v in groups.items()}, total_parameters=sum(p.numel() for p in model.parameters()), initial_weights_sha256=model.weights_hash(),
                train_scenes=len(train), updates=total, lr=args.lr, cache=dict(dir=str(cache), manifest_sha256=sha_file(cache / 'manifest.json')), args=vars(args),
                wire=f'kbit {WIRE_BITS[0]}+{WIRE_BITS[1]}', wire_bits=list(WIRE_BITS), fusion='max', readout='posterior_only', bits_per_coordinate_on_wire=sum(WIRE_BITS), raw_code_bytes=D * sum(WIRE_BITS) // 8,
                val_scenes=len(val), fresh_scenes=len(fresh), skipped_panel_specs=skipped_panels, warm_start=warm_start)
    json.dump(meta, open(out / 'run.json', 'w'), indent=1)
    torch.save(dict(state_dict=model.state_dict(), meta=meta, updates=0), out / 'checkpoints' / 'initial.pt')

    def save(name, updates):
        torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), updates=updates, meta=meta, weights_sha256=model.weights_hash()), out / 'checkpoints' / name)

    @torch.no_grad()
    def evaluate(scenes, label):
        global CHART_M
        rows = []
        for si, sc in enumerate(scenes):
            CHART_M = sc['chart_m']
            mu, q, mask = model.source(sc['own_t'])
            feats = rollout_max(mu, q, mask, sc['edges'])
            hops = {}; preds = {}
            for k in HOPS:
                tgt = stream_targets(sc['own_t'], sc['reached_t'][HOPS.index(k)])
                pred = predict(model.decoder, feats[k])
                hops[str(k)] = agg(metrics3d(pred, tgt, dims, sc['slab'], sc['obj'], sc['struct'])); preds[k] = pred
            rows.append(dict(spec=sc['spec'], hops=hops, mask_fraction=float(mask.mean()), q_mean=float((q * mask).sum() / mask.sum().clamp_min(1))))
            if si < 2:
                np.savez_compressed(out / 'validation' / f'{label}-panel-{si}.npz', own=sc['own_t'].cpu().numpy(), mu=mu.cpu().numpy(), q=(q * mask).cpu().numpy(),
                                    **{f'pred_k{k}': preds[k].cpu().numpy().reshape(-1, *dims) for k in HOPS})
        summary = {str(k): {m: (float(np.mean([r['hops'][str(k)][m] for r in rows if r['hops'][str(k)][m] is not None])) if any(r['hops'][str(k)][m] is not None for r in rows) else None)
                            for m in METRIC_KEYS} for k in HOPS}
        return dict(rows=rows, summary=summary)

    def fmt(s):
        f3 = lambda v: 'n/a' if v is None else f'{v:.3f}'
        return ' | '.join(f"R{k}: surf {f3(s[str(k)]['surface_f1'])} tol1 {f3(s[str(k)]['surface_f1_tol1'])} var {f3(s[str(k)]['var_occ_f1'])} furn {f3(s[str(k)]['furn_f1'])} obj {f3(s[str(k)]['obj_recall'])} occ {f3(s[str(k)]['occ_f1'])} acc {f3(s[str(k)]['accuracy'])} fpr {f3(s[str(k)]['unknown_known_fpr'])}" for k in HOPS)

    history = []; updates = 0; best = (-1., None); losses = []
    order_rng = np.random.default_rng(args.seed); perm = order_rng.permutation(len(train)); pos = 0
    model.train(); tick = time.time()
    while updates < total:
        if STOP or time.time() > args.deadline_epoch or (args.max_updates and updates >= args.max_updates):
            break
        if pos >= len(perm):
            perm = order_rng.permutation(len(train)); pos = 0
        sc = train[perm[pos]]; pos += 1
        CHART_M = sc['chart_m']
        own_d = sc['own_t'].to(device, non_blocking=True); reached_d = sc['reached_t'].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        mu, q, mask = model.source(own_d)
        feats = rollout_max(mu, q, mask, sc['edges'])
        total_loss = 0.; parts = {}
        for k in HOPS:
            tgt = stream_targets(own_d, reached_d[HOPS.index(k)])
            l, kn, oc = stage_loss(model.decoder, feats[k], tgt, dims)
            total_loss = total_loss + WEIGHTS[k] * l; parts[k] = (kn, oc)
        total_loss.backward()
        logging = (updates + 1) % 50 == 0
        gnorm = {name: round(math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)), 6) for name, ps in groups.items()} if logging else None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not math.isfinite(float(norm)):
            raise FloatingPointError('nonfinite gradient')
        opt.step(); sched.step(); updates += 1; losses.append(float(total_loss.detach()))
        if logging:
            print(f"[3s] upd{updates} loss {np.mean(losses[-50:]):.4f} k0(kn,occ)={parts[0][0]:.3f},{parts[0][1]:.3f} k8={parts[8][0]:.3f},{parts[8][1]:.3f} grad {gnorm} lr {sched.get_last_lr()[0]:.2e} mask {float(mask.mean()):.2f} {time.time()-t0:.0f}s", flush=True)
        if updates % args.val_every == 0 or updates == total:
            model.eval(); ev = evaluate(val, f'val-u{updates:06d}'); model.train()
            rec = dict(updates=updates, train_loss=float(np.mean(losses[-args.val_every:])), seconds=time.time() - tick, weights_sha256=model.weights_hash(), validation=ev['summary'])
            json.dump(dict(updates=updates, **ev), open(out / 'validation' / f'upd-{updates:06d}.json', 'w'), indent=1)
            print(f"[3s] VAL u{updates}: {fmt(ev['summary'])}", flush=True)
            score = ev['summary']['8']['surface_f1']
            if score is not None and score > best[0]:
                best = (score, updates); save('best.pt', updates)
            save('latest.pt', updates); history.append(rec); json.dump(history, open(out / 'history.json', 'w'), indent=1)
    model.eval()
    final = dict(updates=updates, weights_sha256=model.weights_hash(), seconds=time.time() - t0)
    save('final.pt', updates)
    if time.time() < args.deadline_total:
        ev = evaluate(val, 'final-val'); final['val'] = ev['summary']; json.dump(ev, open(out / 'validation' / 'final-val.json', 'w'), indent=1); print(f"[3s] FINAL val: {fmt(ev['summary'])}", flush=True)
    if time.time() < args.deadline_total:
        ev = evaluate(fresh, 'final-fresh'); final['fresh'] = ev['summary']; json.dump(ev, open(out / 'validation' / 'final-fresh.json', 'w'), indent=1); print(f"[3s] FINAL fresh: {fmt(ev['summary'])}", flush=True)
    json.dump(final, open(out / 'final.json', 'w'), indent=1)
    print(f'[3s] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
