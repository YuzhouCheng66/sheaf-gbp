"""v342 matched pair: OWNED-block 64-variable Sheaf code vs DENSE global64 code (Claude, 2026-10-21).

One intervention: the semantics of the 64 Gaussian variables.
  owned : variable k is owned by public block k of an 8x8 grid (16 px at H128).
          mu_k, q_k are computed ONLY from the agent's own pixels inside block k
          (strict per-block encoder, no cross-block receptive field); q_k is
          multiplied by the hard indicator "block k contains >=1 own-known cell"
          (disclosed use of the agent's OWN observation inside its own encoder;
          never a decoder input). Hence unobserved blocks contribute exactly the
          public prior, and the unchanged v295 forest GBP average is the union.
  dense : same per-block trunk, but a global head (flatten -> MLP -> 64 mu, 64 q)
          with the v277 'active' semantics (q>0 on every coordinate whenever the
          agent saw anything) - the anchor family's semantics.
Everything else identical in both arms: data (v306 TRAIN plan at H128, via the
sealed deterministic generator), stage targets (actual arrived unions R0/1/2/4/8),
stage weights .5/.125x4, loss (known BCE + truncated-DT L1 wall head; wall:=DT<0.5),
decoder trunk family (v121-shaped CNN), hard-8 source means, v295 profile-4
quantised forest wire, optimiser, seed, update count, wall-clock lease.
Readout input is only [hard8 posterior mean 64, log posterior variance 64].
Local iterative encoding (each agent refining its own code against its own
observation before messaging) is an EVALUATION variant reported separately.
"""
import argparse, hashlib, json, math, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HOPS = (0, 1, 2, 4, 8)
WEIGHTS = {0: .5, 1: .125, 2: .125, 4: .125, 8: .125}
STOP = False


def _sigterm(*_):
    global STOP
    STOP = True


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- model
def public(count, sy, sx, size, device):
    y = (torch.arange(sy, device=device) + .5) / sy * 2 - 1
    x = (torch.arange(sx, device=device) + .5) / sx * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    scale = torch.full_like(yy, 1. / size)
    return torch.stack((xx, yy, scale, scale))[None].expand(count, -1, -1, -1)


class CnnTrunk(nn.Module):
    def __init__(self, in_planes, widths=(96, 64, 48, 32), extra_stages=0, out_channels=2):
        super().__init__()
        self.lift = nn.Sequential(nn.Conv2d(in_planes + 4, 128, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        blocks, incoming = [], 128
        widths = list(widths) + [32] * extra_stages
        for outgoing in widths:
            blocks.append(nn.Sequential(nn.Conv2d(incoming + 4, outgoing, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.SiLU()))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv2d(incoming, out_channels, 1)

    def forward(self, grid, size):
        count = grid.shape[0]
        hidden = self.lift(torch.cat((grid, public(count, 16, 16, size, grid.device)), 1))
        extent = 16
        for block in self.blocks:
            hidden = F.interpolate(hidden, (extent, extent), mode='bilinear', align_corners=False)
            hidden = block(torch.cat((hidden, public(count, extent, extent, size, grid.device)), 1))
            extent = min(extent * 2, size)
        return self.output(hidden)


class PatchTrunk(nn.Module):
    """Strictly block-local features: each 8x8 block of the map is encoded independently."""
    def __init__(self, size, width=128):
        super().__init__()
        self.size, self.block = size, size // 8
        self.net = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(96, width, 3, stride=2, padding=1), nn.SiLU())
        self.proj = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())
        self.width = width

    def forward(self, private):
        n, h, _ = private.shape; p = self.block
        xy = public(1, h, h, h, private.device)[0, :2]
        inp = torch.cat(((private >= 0).float()[:, None], (private == 0).float()[:, None], (private == 1).float()[:, None],
                         xy[None].expand(n, -1, -1, -1)), 1)                      # [N,5,H,H]
        patches = inp.unfold(2, p, p).unfold(3, p, p)                              # [N,5,8,8,p,p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n * 64, 5, p, p)
        f = self.net(patches)                                                      # [N*64,W,p/8,p/8]
        f = torch.cat((f.mean((2, 3)), f.amax((2, 3))), 1)
        return self.proj(f).reshape(n, 64, self.width)                            # block-major (row, col)


PRECISION_MIN, PRECISION_MAX = 1e-4, 4.
LATENT_PRIOR = 1e-4
MU_BITS, Q_BITS, WIRE = 8, 8, 'sealed'   # --mu-bits/--q-bits/--wire kbit: k-bit code study (item 1)
EDGE_SELECT_M = 0.   # --edge-select R: restriction map of edge (i,j) = selection of the blocks whose centre lies within R metres of the edge midpoint (0 = identity)


def edge_selection(positions_m, i, j, device):
    """[64] 0/1 selection (row-major 8x8 blocks of the 20 m world) for the stalk of edge (i,j); identity when EDGE_SELECT_M <= 0."""
    if EDGE_SELECT_M <= 0:
        return torch.ones(64, dtype=torch.float64, device=device)
    c = (torch.arange(8, dtype=torch.float64, device=device) + .5) * 2.5
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    mid = (torch.as_tensor(positions_m[i], dtype=torch.float64, device=device) + torch.as_tensor(positions_m[j], dtype=torch.float64, device=device)) / 2
    dist = ((cx - mid[0]) ** 2 + (cy - mid[1]) ** 2).sqrt()
    return (dist <= EDGE_SELECT_M).double().flatten()


def kbit_wire_quantize(p, eta):
    """k-bit wire: precision -> log codebook with 2^Q_BITS-1 levels over [1e-8,128] (exact 0 kept), mean -> uniform 2^(MU_BITS-1)-1 per side in [-1,1].
    Returns (p_hat, p_hat*mu_hat) in FP64 with identity STE. Messages therefore cost MU_BITS+Q_BITS bits per coordinate."""
    low, high = 1e-8, 128.
    nq = 2 ** Q_BITS - 1
    active = p.detach() > 0
    safe = torch.where(active, p.detach(), torch.ones_like(p))
    # floor (conservative, p_hat <= p): a prior-only factor can then never out-rank the receiver's own prior under max-selection (pre-launch review M1, 2026-10-22 14:32 UTC)
    code = torch.floor((safe.clamp(low, high).log() - math.log(low)) / (math.log(high) - math.log(low)) * (nq - 1))
    p_hat = torch.where(active, torch.exp(code / (nq - 1) * (math.log(high) - math.log(low)) + math.log(low)), torch.zeros_like(p))
    levels = 2 ** (MU_BITS - 1) - 1
    mu = torch.where(active, eta.detach() / safe, torch.zeros_like(eta)).clamp(-1, 1)
    mu_hat = torch.round(mu * levels) / levels
    return p_hat + (p - p.detach()), p_hat * mu_hat + (eta - eta.detach()), None


PRECISION_MODE = 'learned'   # --precision learned | coverage  (coverage: q_k = PRECISION_MAX * observed fraction of block k; no precision head)


class OwnedHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, feats, private):
        n, h, _ = private.shape; p = h // 8
        out = self.mlp(feats)                                                      # [N,64,2]
        mu = out[..., 0].tanh()
        coverage = F.avg_pool2d((private >= 0).float()[:, None], p).flatten(1)      # [N,64] observed fraction of block
        observed = coverage > 0
        if PRECISION_MODE == 'coverage':
            q = (PRECISION_MAX * coverage).clamp_min(PRECISION_MIN)                 # deterministic observation-count precision (TSDF/occupancy-grid semantics)
        else:
            q = PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * out[..., 1].sigmoid()
        return mu, q, observed.float()


class DenseHead(nn.Module):
    """Anchor-like global head: shared per-block projection to 32, flatten (2048, the v277 global_head fan-in) -> 64 mu + 64 q."""
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, 32), nn.SiLU())
        self.head = nn.Linear(64 * 32, 128)

    def forward(self, feats, private):
        n = feats.shape[0]
        out = self.head(self.block(feats).reshape(n, -1))
        mu = out[:, :64].tanh()
        q = PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * out[:, 64:].sigmoid()
        active = (private >= 0).flatten(1).any(-1).float()[:, None].expand(-1, 64)  # v277 semantics
        return mu, q, active


READOUT_MODE = 'anchored'   # --readout anchored | posterior_only


class OwnedDecoder(nn.Module):
    """anchored: input = [own hard8 mean, own log var, posterior hard8 mean, posterior log var] (4 planes); posterior_only: 2 planes."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        planes = 2 if READOUT_MODE == 'posterior_only' else 4
        self.pre = nn.Sequential(nn.Conv2d(planes, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.trunk = CnnTrunk(64, extra_stages=int(math.log2(size // 128)))

    def forward(self, features):
        b = features.shape[0]
        own, post = features[:, :128], features[:, 128:]
        if READOUT_MODE == 'posterior_only':
            grid = torch.cat((post[:, :64].reshape(b, 1, 8, 8), (post[:, 64:] / 8.).reshape(b, 1, 8, 8)), 1)
        else:
            grid = torch.cat((own[:, :64].reshape(b, 1, 8, 8), (own[:, 64:] / 8.).reshape(b, 1, 8, 8),
                              post[:, :64].reshape(b, 1, 8, 8), (post[:, 64:] / 8.).reshape(b, 1, 8, 8)), 1)
        grid = F.interpolate(self.pre(grid), (16, 16), mode='bilinear', align_corners=False)
        out = self.trunk(grid, self.size)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


class DenseDecoder(nn.Module):
    """v299-shaped, source-anchored: linear [own 128, posterior 128] -> 2048 -> 4 mean planes (tanh) + 4 log1p(q) planes at 16x16 -> trunk."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.lift = nn.Linear(256, 2048)
        self.trunk = CnnTrunk(8, extra_stages=int(math.log2(size // 128)))

    def forward(self, features):
        own, post = features[:, :128], features[:, 128:]
        x = torch.cat((own[:, :64], own[:, 64:] / 8., post[:, :64], post[:, 64:] / 8.), 1)
        mean_raw, q_raw = self.lift(x).chunk(2, -1)
        q = 1e-4 + (4. - 1e-4) * q_raw.sigmoid()
        grid = torch.cat((mean_raw.tanh().reshape(-1, 4, 16, 16), q.log1p().reshape(-1, 4, 16, 16)), 1)
        out = self.trunk(grid, self.size)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


class SheafModel(nn.Module):
    def __init__(self, arm, size, seed):
        super().__init__()
        self.arm, self.size = arm, size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.trunk = PatchTrunk(size)
            if arm == 'owned':
                self.head, self.decoder = OwnedHead(self.trunk.width), OwnedDecoder(size)
            else:
                self.head, self.decoder = DenseHead(self.trunk.width), DenseDecoder(size)

    def source(self, private):
        """private [N,H,H] int8 -> mu [N,64] (hard8 via STE), q_learned [N,64], mask [N,64]."""
        mu, q, mask = self.head(self.trunk(private), private)
        return self.latent_quantize(mu), q, mask

    @staticmethod
    def latent_quantize(z):
        levels = 2 ** (MU_BITS - 1) - 1          # 8 bits -> 127 (the sealed hard8 alphabet); k bits -> 2^(k-1)-1 per side
        zc = z.clamp(-1, 1)
        return (torch.round(zc * levels) / levels).detach() + (zc - zc.detach())

    def parameter_groups(self):
        return dict(trunk=list(self.trunk.parameters()), head=list(self.head.parameters()), decoder=list(self.decoder.parameters()))

    def weights_hash(self):
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


# ----------------------------------------------------------------------------- GBP (sealed v295 primitives)
def make_rollout(forest, latent_quantize, fusion='sum'):
    wire_q = (lambda p, e: kbit_wire_quantize(p, e)) if WIRE == 'kbit' else forest.quantize
    """fusion='sum': sealed v295 Gaussian sum-product forest GBP (precision-weighted average per coordinate).
    fusion='max': per-coordinate MAX-PRECISION SELECTION on the same factor graph, same packets (p, eta), same profile-4 wire
    quantiser, same synchronous rounds. Each message carries, per coordinate, the most precise source factor known on the
    cavity side; a node's belief is the most precise factor among its own and all incoming. Join-semilattice: idempotent,
    commutative, associative => exact on any graph (cycles, duplicates, stale re-sends) and equal to Gaussian BP whenever at most
    one source has non-prior precision on a coordinate (the owned ideal). It is the beta->inf limit of q^beta-tempered averaging."""
    def rollout_sum_kbit(mu, q_learned, mask, edges, rounds=8, stages=HOPS, positions_m=None):
        """Same synchronous cavity-sum forest GBP as v295.advance, with the k-bit wire quantiser (for the k-bit study only)."""
        n = mu.shape[0]
        prior = LATENT_PRIOR / (n * 64)
        evidence_q = mask.double() * q_learned.double()
        p_own = evidence_q + prior; eta_own = evidence_q * mu.double()
        graph = forest.validate_forest(edges, n) if edges else []
        directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
        incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
        msg_p = [torch.zeros(64, dtype=torch.float64, device=mu.device) for _ in directed]
        msg_e = [torch.zeros(64, dtype=torch.float64, device=mu.device) for _ in directed]
        def belief():
            P = torch.stack([p_own[j] + sum([msg_p[a] for a in incoming[j]], torch.zeros_like(p_own[j])) for j in range(n)])
            E = torch.stack([eta_own[j] + sum([msg_e[a] for a in incoming[j]], torch.zeros_like(eta_own[j])) for j in range(n)])
            return torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1)
        feats = {}
        if 0 in stages:
            feats[0] = belief()
        for r in range(1, rounds + 1):
            new_p, new_e = [], []
            for a, (i, j) in enumerate(directed):
                cp = p_own[i] + sum([msg_p[b] for b in incoming[i] if directed[b][0] != j], torch.zeros_like(p_own[i]))
                ce = eta_own[i] + sum([msg_e[b] for b in incoming[i] if directed[b][0] != j], torch.zeros_like(eta_own[i]))
                pq, eq, _ = wire_q(cp[None], ce[None]); sel = edge_selection(positions_m, i, j, mu.device) if positions_m is not None else 1.
                new_p.append(pq[0] * sel); new_e.append(eq[0] * sel)
            msg_p, msg_e = new_p, new_e
            if r in stages:
                feats[r] = belief()
        return feats, None

    def rollout_sum(mu, q_learned, mask, edges, rounds=8, stages=HOPS, positions_m=None):
        if WIRE == 'kbit' or EDGE_SELECT_M > 0:
            return rollout_sum_kbit(mu, q_learned, mask, edges, rounds, stages, positions_m)
        n = mu.shape[0]
        prior = LATENT_PRIOR / (n * 64)
        evidence_q = mask.double() * q_learned.double()
        precision = evidence_q + prior
        q = torch.diag_embed(precision)[None]
        h = (evidence_q * mu.double())[None]
        state = forest.initialize(q, h, edges)
        feats = {}
        def features(state):
            mean, var = forest.scalar_belief(state)
            return torch.cat((latent_quantize(mean[0].float()), var[0].log().float()), -1)
        if 0 in stages:
            feats[0] = features(state)
        for r in range(1, rounds + 1):
            state = forest.advance(state, 1, quantized=True, trace=False)['state']
            if r in stages:
                feats[r] = features(state)
        return feats, state

    def select(p_list, eta_list):
        """per-coordinate argmax precision over a list of [N,64] tensors (first entry preferred on ties)."""
        P = torch.stack(p_list, 0); E = torch.stack(eta_list, 0)
        tie = torch.arange(P.shape[0], device=P.device, dtype=P.dtype).view(-1, 1, 1) * 1e-15
        idx = (P - tie).argmax(0, keepdim=True)
        return P.gather(0, idx)[0], E.gather(0, idx)[0]

    def rollout_max(mu, q_learned, mask, edges, rounds=8, stages=HOPS, allow_cycles=False, positions_m=None):
        n = mu.shape[0]
        prior = LATENT_PRIOR / (n * 64)
        evidence_q = mask.double() * q_learned.double()
        p_own = evidence_q + prior                                   # [N,64] FP64
        eta_own = evidence_q * mu.double()
        graph = ([(min(i, j), max(i, j), w) for i, j, w in edges] if allow_cycles else forest.validate_forest(edges, n)) if edges else []
        directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
        incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
        msg_p = [torch.zeros(64, dtype=torch.float64, device=mu.device) for _ in directed]
        msg_e = [torch.zeros(64, dtype=torch.float64, device=mu.device) for _ in directed]
        # Quantise each source factor ONCE at its origin (profile-4 wire codebook, STE); relays forward codebook values bit-exactly,
        # so the join-semilattice is exact on the wire (no re-rounding drift along multi-hop paths).
        p_wire, e_wire, _ = wire_q(p_own, eta_own)
        def belief():
            ps, es = [], []
            for j in range(n):
                cand_p = [p_own[j]] + [msg_p[a] for a in incoming[j]]
                cand_e = [eta_own[j]] + [msg_e[a] for a in incoming[j]]
                pj, ej = select([c[None] for c in cand_p], [c[None] for c in cand_e])
                ps.append(pj[0]); es.append(ej[0])
            P = torch.stack(ps); E = torch.stack(es)
            mean = E / P
            return torch.cat((latent_quantize(mean.float()), (1. / P).log().float()), -1)
        feats = {}
        if 0 in stages:
            feats[0] = belief()
        for r in range(1, rounds + 1):
            new_p, new_e = [], []
            for a, (i, j) in enumerate(directed):
                cand_p = [p_wire[i]] + [msg_p[b] for b in incoming[i] if directed[b][0] != j]
                cand_e = [e_wire[i]] + [msg_e[b] for b in incoming[i] if directed[b][0] != j]
                pc, ec = select([c[None] for c in cand_p], [c[None] for c in cand_e])
                sel = edge_selection(positions_m, i, j, mu.device) if positions_m is not None else 1.
                new_p.append(pc[0] * sel); new_e.append(ec[0] * sel)
            msg_p, msg_e = new_p, new_e
            if r in stages:
                feats[r] = belief()
        return feats, None

    return rollout_max if fusion == 'max' else rollout_sum


# ----------------------------------------------------------------------------- loss / predict
def truncated_dt(wall, T):
    b, h, w = wall.shape
    best = torch.full((b, h, w), float(T), device=wall.device)
    pad = F.pad(wall.float(), (T, T, T, T), value=0.)
    for dy in range(-T, T + 1):
        for dx in range(-T, T + 1):
            d = math.hypot(dy, dx)
            if d >= T:
                continue
            shifted = pad[:, T + dy:T + dy + h, T + dx:T + dx + w] > 0
            best = torch.where(shifted, torch.minimum(best, torch.full_like(best, d)), best)
    return best


def stage_loss(decoder, feats, target, dt_target, w_near=4.):
    known_logit, wall_ch = decoder(feats)
    known = (target >= 0).float()
    known_bce = F.binary_cross_entropy_with_logits(known_logit, known, reduction='none').mean(-1)
    dt_pred = F.softplus(wall_ch)
    weight = known * (1. + w_near * (dt_target <= 1.5).float())
    wall_term = (weight * (dt_pred - dt_target).abs()).sum(-1) / weight.sum(-1).clamp_min(1)
    return known_bce.mean() + wall_term.mean(), float(known_bce.mean()), float(wall_term.mean())


@torch.no_grad()
def predict(decoder, feats):
    known_logit, wall_ch = decoder(feats)
    wall = F.softplus(wall_ch) < .5
    return torch.where(known_logit >= 0, wall.to(torch.int8), torch.full_like(wall, -1, dtype=torch.int8))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', required=True, choices=['owned', 'dense'])
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=342101)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--deadline-epoch', type=float, default=1e18, help='monotonic deadline (time.time()) for the TRAIN phase')
    ap.add_argument('--deadline-total', type=float, default=1e18, help='absolute time.time() deadline for everything')
    ap.add_argument('--val-every', type=int, default=2)
    ap.add_argument('--dt-trunc', type=int, default=4)
    ap.add_argument('--refine-steps', type=int, default=50)
    ap.add_argument('--refine-lr', type=float, default=.03)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--precision', default='learned', choices=['learned', 'coverage'])
    ap.add_argument('--consistency', type=float, default=0., help='weight of the fusion-consistency loss: ||GBP posterior mean - enc(union observation)||^2 on union-observed blocks, stages 1/2/4/8')
    ap.add_argument('--readout', default='posterior_only', choices=['anchored', 'posterior_only'])
    ap.add_argument('--fusion', default='sum', choices=['sum', 'max'])
    ap.add_argument('--mu-bits', type=int, default=8)
    ap.add_argument('--q-bits', type=int, default=8)
    ap.add_argument('--wire', default='sealed', choices=['sealed', 'kbit'])
    ap.add_argument('--edge-select', type=float, default=0., help='metres; restriction map = block selection around the edge midpoint (0 = identity)')
    ap.add_argument('--world', default='v293', choices=['v293', 'houseexpo'], help='world family: sealed v293 corridor-spine or HouseExpo v365 (env HOUSEEXPO_JSON_DIR, HOUSEEXPO_INDEX)')
    ap.add_argument('--dynamic', action='store_true', help='train on time slices of dyn_data_v349 exploration episodes (T=6) instead of the static 3-pose maps')
    ap.add_argument('--max-updates', type=int, default=0, help='preflight: stop training after this many updates')
    ap.add_argument('--init-from', default='', help='warm start: load all same-name same-shape tensors from this checkpoint (e.g. H128 final.pt for an H256 continuation); mismatched/new tensors keep fresh init')
    ap.add_argument('--allow-partial-init', action='store_true', help='permit a warm start that leaves some tensors at fresh init (H128->H256 continuation); otherwise any fresh/skipped tensor is fatal (review M1)')
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGINT, _sigterm)
    global PRECISION_MODE, READOUT_MODE, MU_BITS, Q_BITS, WIRE, EDGE_SELECT_M; PRECISION_MODE = args.precision; READOUT_MODE = args.readout; MU_BITS = args.mu_bits; Q_BITS = args.q_bits; WIRE = args.wire; EDGE_SELECT_M = args.edge_select
    assert not (args.wire == 'sealed' and args.mu_bits != 8), 'the sealed profile-4 wire assumes hard-8 means; use --wire kbit for k-bit studies'

    src = Path(args.source).resolve()
    sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
    from scripts import mapping_multiroom_data_v293 as data
    if args.world == 'houseexpo':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mapping_houseexpo_data_v365 as data                       # same fixture contract; only the world geometry differs
        assert not args.dynamic, 'dynamic exploration episodes are defined for v293 only'
    from scripts import mapping_native_bank_v306 as bank
    from scripts import mapping_multiroom_forest_v295 as forest
    from scripts import train_mapping_support_native_v306 as base
    from scripts.mapping_recognition_metrics_v277 import node_metrics
    from scripts.mapping_implicit_model_v192 import latent_quantize

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'checkpoints').mkdir()
    (out / 'validation').mkdir()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    size, T = args.size, args.dt_trunc
    t0 = time.time()
    rollout = make_rollout(forest, latent_quantize, fusion=args.fusion)

    # ---- data: exact v306 plan specs, regenerated deterministically
    scene_plan = data.scene_plan if args.world == 'houseexpo' else base.scene_plan
    train_specs = [s for s in scene_plan('TRAIN') if s['size'] == size] if args.epochs > 0 else []   # --epochs 0: evaluation only, no training cache (review M8)
    val_specs = [s for s in scene_plan('VALIDATION') if s['size'] == size]
    fresh_specs = [dict(s, index=s['index'] + 8) for s in val_specs]          # never-selected-on panel (indices 8-15)
    if args.smoke:
        train_specs, val_specs, fresh_specs = train_specs[:6], val_specs[:2], fresh_specs[:2]

    def load(spec):
        fx = data.make_fixture(**spec); inp = fx['inputs']
        targets, reached = {}, {}
        for k in HOPS:
            targets[k], reached[k] = bank.forest_targets(inp, k)
        fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
        return dict(spec=spec, private=inp['private'], targets=targets, reached=reached, edges=fsel['edges'], positions_m=inp['positions_m'],
                    door_mask=fx['evaluation']['door_mask'], private_sha256=data.array_hash(inp['private']))

    def load_dynamic(spec):
        """one training item per time slice t of a T=6 exploration episode (stateless per-step supervision on the current forest)."""
        import dyn_data_v349 as dyn
        fx = dyn.make_dynamic_fixture(spec['split'], spec['index'], size=spec['size'], rooms=spec['rooms'], agents=spec['agents'])
        items = []
        for t in range(fx['maps'].shape[0]):
            inp = dict(private=fx['maps'][t], positions_m=fx['paths_m'][t], trajectory_m=np.stack([fx['paths_m'][max(t - 2, 0)], fx['paths_m'][max(t - 1, 0)], fx['paths_m'][t]]),
                       adjacency=fx['adjacency'][t], public_frame=bank.expected_frame(spec['size']))
            targets, reached = {}, {}
            for k in HOPS:
                targets[k], reached[k] = bank.forest_targets(inp, k)
            fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], 8.)
            items.append(dict(spec=dict(spec, t=t), private=inp['private'], targets=targets, reached=reached, edges=fsel['edges'], positions_m=inp['positions_m'],
                              door_mask=fx['door_mask'], private_sha256=data.array_hash(inp['private'])))
        return items

    if args.dynamic:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        train = [it for s in train_specs for it in load_dynamic(s)]
    else:
        train = [load(s) for s in train_specs]
    val = [load(s) for s in val_specs]
    fresh = [load(s) for s in fresh_specs]
    print(f'[{args.arm}] scenes train {len(train)} val {len(val)} fresh {len(fresh)} gen {time.time()-t0:.0f}s', flush=True)
    # cache targets/dt on device
    for sc in train + val + fresh:
        n = sc['private'].shape[0]
        host = args.dynamic and 't' in sc['spec']                       # dynamic training slice: keep on (pinned) host memory (patch F)
        place = (lambda x: x.pin_memory()) if (host and device.type == 'cuda') else ((lambda x: x) if host else (lambda x: x.to(device)))
        sc['own_t'] = place(torch.from_numpy(sc['private']))
        sc['tgt'] = {k: place(torch.from_numpy(sc['targets'][k].reshape(n, -1))) for k in HOPS}
        sc['host'] = host
        if not (args.dynamic and 't' in sc['spec']):                      # dynamic training slices (3552 items, 13 GiB of float32 DT): computed per update instead (review M7)
            sc['dt'] = {k: truncated_dt(sc['tgt'][k].reshape(n, size, size) == 1, T).reshape(n, -1) for k in HOPS}

    model = SheafModel(args.arm, size, args.seed).to(device)
    init_report = None
    if args.init_from:
        payload = torch.load(args.init_from, map_location='cpu', weights_only=False)
        src_state = payload['state_dict']; own_state = model.state_dict()
        loaded = {k: v for k, v in src_state.items() if k in own_state and own_state[k].shape == v.shape}
        missing = [k for k in own_state if k not in loaded]; skipped = [k for k in src_state if k not in loaded]
        model.load_state_dict({**own_state, **loaded}, strict=True)
        init_report = dict(init_from=str(args.init_from), init_file_sha256=sha_file(args.init_from), loaded=len(loaded), fresh=missing, skipped_shape_mismatch=skipped,
                           source_arm=payload.get('meta', {}).get('arm'), source_updates=payload.get('updates'))
        assert payload.get('meta', {}).get('arm') == args.arm, 'warm start must come from the same arm'
        for mkey, mval in (('readout_mode', args.readout), ('precision_mode', args.precision), ('mu_bits', args.mu_bits), ('q_bits', args.q_bits), ('wire', args.wire)):
            assert payload.get('meta', {}).get(mkey, mval) == mval, f'warm start {mkey} mismatch: checkpoint {payload.get("meta", {}).get(mkey)} vs CLI {mval}'
        if not args.allow_partial_init:
            assert not missing and not skipped, f'partial warm start (fresh {missing}, skipped {skipped}); pass --allow-partial-init if intended'
        print(f'[{args.arm}] warm start: loaded {len(loaded)} tensors, fresh {missing}, skipped {skipped}', flush=True)
    groups = model.parameter_groups()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    total_updates = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1., (u + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(u, total_updates) / max(1, total_updates)))))
    meta = dict(arm=args.arm, size=size, seed=args.seed, parameters={k: sum(p.numel() for p in v) for k, v in groups.items()},
                total_parameters=sum(p.numel() for p in model.parameters()), initial_weights_sha256=model.weights_hash(),
                train_scenes=len(train), val_scenes=len(val), fresh_scenes=len(fresh), epochs=args.epochs, lr=args.lr,
                train_private_sha256=[sc['private_sha256'] for sc in train][:8], args=vars(args), warm_start=init_report, precision_mode=args.precision, readout_mode=args.readout, consistency=args.consistency, fusion=args.fusion, mu_bits=args.mu_bits, q_bits=args.q_bits, wire=args.wire, edge_select_m=args.edge_select, dynamic=args.dynamic, bits_per_coordinate_on_wire=(args.mu_bits + args.q_bits if args.wire == 'kbit' else 16))
    with open(out / 'run.json', 'w') as f:
        json.dump(meta, f, indent=1)
    torch.save(dict(state_dict=model.state_dict(), meta=meta, updates=0), out / 'checkpoints' / 'initial.pt')

    def save(name, updates, epoch):
        torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), updates=updates, epoch=epoch,
                        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                                 numpy=np.random.get_state(), python=random.getstate()), meta=meta, weights_sha256=model.weights_hash()),
                   out / 'checkpoints' / name)

    @torch.no_grad()
    def evaluate(scenes, label, refine_steps=0):
        rows = []
        for sc in scenes:
            n = sc['private'].shape[0]
            mu, q, mask = model.source(sc['own_t'])
            if refine_steps:
                with torch.enable_grad():
                    z = mu.detach().clone().requires_grad_(True)
                    o = torch.optim.Adam([z], lr=args.refine_lr)
                    prior = LATENT_PRIOR / (n * 64)
                    for _ in range(refine_steps):
                        o.zero_grad(set_to_none=True)
                        zq = model.latent_quantize(z) * mask          # unobserved blocks stay exactly 0 (as in rollout)
                        var = 1. / (mask * q + prior)
                        own_f = torch.cat((zq, var.log()), -1)
                        loss, _, _ = stage_loss(model.decoder, torch.cat((own_f, own_f), -1), sc['tgt'][0], sc['dt'][0])
                        loss.backward(); o.step()
                        with torch.no_grad():
                            z.clamp_(-1, 1)
                    mu = model.latent_quantize(z.detach()) * mask
            feats, _ = rollout(mu, q, mask, sc['edges'], positions_m=sc['positions_m'])
            hops = {}
            preds = {}
            for k in HOPS:
                pred = predict(model.decoder, torch.cat((feats[0], feats[k]), -1)).cpu().numpy().reshape(n, size, size)
                m = node_metrics(pred, sc['private'], sc['targets'][k], {'door_mask': sc['door_mask']})
                agg = {}
                for key in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr', 'own_wall_f1', 'own_typed_accuracy', 'peer_wall_f1', 'known_recall'):
                    vals = [r[key] for r in m if r.get(key) is not None]
                    agg[key] = float(np.mean(vals)) if vals else None
                hops[str(k)] = agg; preds[k] = pred
            rows.append(dict(spec=sc['spec'], hops=hops, mask_fraction=float(mask.mean()), q_mean=float((q * mask).sum() / mask.sum().clamp_min(1))))
            if sc['spec']['index'] in (0, 8):
                np.savez_compressed(out / 'validation' / f"{label}-panel-r{sc['spec']['rooms']}-n{n}-i{sc['spec']['index']}.npz",
                                    own=sc['private'], mu=mu.cpu().numpy(), q=(q * mask).cpu().numpy(),
                                    **{f'target_k{k}': sc['targets'][k] for k in HOPS}, **{f'pred_k{k}': preds[k] for k in HOPS})
        # README-style aggregate: equal weight per (rooms, N) stratum
        summary = {}
        for k in HOPS:
            strata = {}
            for r in rows:
                strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['hops'][str(k)])
            per = {f'rooms{a}/N{b}': {m: (float(np.mean([x[m] for x in v if x[m] is not None])) if any(x[m] is not None for x in v) else None) for m in v[0]} for (a, b), v in strata.items()}
            summary[str(k)] = dict(all={m: (float(np.mean([p[m] for p in per.values() if p[m] is not None])) if any(p[m] is not None for p in per.values()) else None) for m in next(iter(per.values()))}, strata=per)
        return dict(rows=rows, summary=summary)

    def fmt(summary):
        return ' | '.join(f"R{k}: F1 {summary[str(k)]['all']['wall_f1']:.3f} acc {summary[str(k)]['all']['union_accuracy']:.3f} fpr {summary[str(k)]['all']['unknown_known_fpr']:.3f} own {summary[str(k)]['all']['own_wall_f1']:.3f}" for k in HOPS)

    # ---- training
    history = []; updates = 0; best = (-1., None)
    order_rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        perm = order_rng.permutation(len(train))
        model.train(); tick = time.time(); losses = []
        for i, si in enumerate(perm):
            if STOP or time.time() > args.deadline_epoch or (args.max_updates and updates >= args.max_updates):
                break
            sc = train[si]
            if sc.get('host'):
                sc = dict(sc, own_t=sc['own_t'].to(device, non_blocking=True), tgt={k: v.to(device, non_blocking=True) for k, v in sc['tgt'].items()})
            opt.zero_grad(set_to_none=True)
            mu, q, mask = model.source(sc['own_t'])
            feats, _ = rollout(mu, q, mask, sc['edges'], positions_m=sc['positions_m'])
            total = 0.; parts = {}; cons_val = 0.
            for k in HOPS:
                dt_k = sc['dt'][k] if 'dt' in sc else truncated_dt(sc['tgt'][k].reshape(-1, size, size) == 1, T).reshape(-1, size * size)
                l, kn, wl = stage_loss(model.decoder, torch.cat((feats[0], feats[k]), -1), sc['tgt'][k], dt_k)
                total = total + WEIGHTS[k] * l; parts[k] = (kn, wl)
                if args.consistency > 0 and k > 0:
                    # fusion-consistency (homomorphism) term: the fused block code must equal the code the encoder assigns to the
                    # union observation of the actually reached sources. Uses only the agents' own observations (their union is the stage target).
                    mu_u, _, mask_u = model.source(sc['tgt'][k].reshape(-1, size, size))
                    diff = (feats[k][:, :64] - mu_u) ** 2 * mask_u
                    cons = diff.sum() / mask_u.sum().clamp_min(1)
                    total = total + args.consistency * WEIGHTS[k] * cons; cons_val += float(cons)
            total.backward()
            logging = (updates + 1) % 50 == 0
            gnorm = {name: round(math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)), 6) for name, ps in groups.items()} if logging else None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not math.isfinite(float(norm)):
                raise FloatingPointError('nonfinite gradient')
            opt.step(); sched.step(); updates += 1
            losses.append(float(total))
            if logging:
                sat = dict(mu_sat=float((mu.detach().abs() > .99).float().mean()), q_sat=float((q.detach() > 3.99).float().mean()), q_floor=float((q.detach() < .01).float().mean()))
                print(f"[{args.arm}] ep{epoch} upd{updates} loss {np.mean(losses[-50:]):.4f} k0(kn,wl)={parts[0][0]:.3f},{parts[0][1]:.3f} k8={parts[8][0]:.3f},{parts[8][1]:.3f} cons {cons_val:.4f} grad {gnorm} sat {sat} lr {sched.get_last_lr()[0]:.2e} mask {float(mask.mean()):.2f} {time.time()-t0:.0f}s", flush=True)
        model.eval()
        rec = dict(epoch=epoch, updates=updates, train_loss=float(np.mean(losses)) if losses else None, epoch_seconds=time.time() - tick, weights_sha256=model.weights_hash())
        stopped = STOP or time.time() > args.deadline_epoch or (args.max_updates and updates >= args.max_updates)
        if epoch % args.val_every == 0 or epoch == args.epochs or stopped:
            ev = evaluate(val, f'val-e{epoch:02d}')
            rec['validation'] = ev['summary']
            with open(out / 'validation' / f'epoch-{epoch:04d}.json', 'w') as f:
                json.dump(dict(epoch=epoch, updates=updates, weights_sha256=rec['weights_sha256'], **ev), f, indent=1)
            score = ev['summary']['8']['all']['wall_f1']
            print(f"[{args.arm}] VAL e{epoch}: {fmt(ev['summary'])}", flush=True)
            if score > best[0]:
                best = (score, epoch); save('best.pt', updates, epoch)
        save('latest.pt', updates, epoch)
        history.append(rec)
        with open(out / 'history.json', 'w') as f:
            json.dump(history, f, indent=1)
        if stopped:
            print(f'[{args.arm}] stop requested / epoch deadline reached after epoch {epoch}', flush=True); break

    # ---- final: fresh panel + refinement variants on the final weights (not selected on)
    final = dict(arm=args.arm, updates=updates, epochs_completed=len(history), weights_sha256=model.weights_hash(), seconds=time.time() - t0)
    save('final.pt', updates, len(history))
    if time.time() < args.deadline_total:
        ev_val = evaluate(val, 'final-val')
        final['val'] = ev_val['summary']; print(f"[{args.arm}] FINAL val: {fmt(ev_val['summary'])}", flush=True)
        with open(out / 'validation' / 'final-val.json', 'w') as f:
            json.dump(ev_val, f, indent=1)
    if time.time() < args.deadline_total:
        ev_fresh = evaluate(fresh, 'final-fresh')
        final['fresh'] = ev_fresh['summary']; print(f"[{args.arm}] FINAL fresh(8-15): {fmt(ev_fresh['summary'])}", flush=True)
        with open(out / 'validation' / 'final-fresh.json', 'w') as f:
            json.dump(ev_fresh, f, indent=1)
    if time.time() < args.deadline_total and args.refine_steps > 0:
        ev_ref = evaluate(val, 'final-val-refined', refine_steps=args.refine_steps)
        final['val_refined'] = ev_ref['summary']; print(f"[{args.arm}] FINAL val + local refinement({args.refine_steps}): {fmt(ev_ref['summary'])}", flush=True)
        with open(out / 'validation' / 'final-val-refined.json', 'w') as f:
            json.dump(ev_ref, f, indent=1)
    with open(out / 'final.json', 'w') as f:
        json.dump(final, f, indent=1)
    print(f'[{args.arm}] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
