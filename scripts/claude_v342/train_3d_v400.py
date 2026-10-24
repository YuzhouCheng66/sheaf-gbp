"""3D Sheaf-GBP map fusion trainer v400 (Claude, 2026-10-24). The 2D owned-block design (train_v378) moved to 3D with ONLY the
dimension changed: each agent's 3D occupancy map (S^3 voxels, S in {64,128,256}) is described by D = 256 Gaussian variables + 256
diagonal precisions, one per public block of an 8 x 8 x 4 grid (1.6 m x 1.6 m x one storey). Block-local 3D encoder (mu_k, q_k from
the agent's own voxels inside block k only; q_k times the indicator "block k has >= 1 own-known voxel"); 8+8-bit k-bit wire (exact,
per coordinate); per-coordinate MAX-PRECISION selection (lattice join) over 8 synchronous rounds on the radio forest; a 3D CNN decoder
that reads ONLY the fused posterior (hard-8 mean, log variance) and renders known / occupancy logits at S^3.
Targets per hop k: ternary union of the own maps of the sources actually reached within k forest hops (stream targets).
Loss: known BCE + occupancy BCE on target-known voxels with weight 1 + 4 x [target surface voxel]. Metrics in target-known scope:
occupied F1, SURFACE F1 (occupied voxels with a free 6-neighbour in the target: the informative part of thick solids), three-way
accuracy, unknown-known FPR. Scenes come from a cache written by gen_3d_cache_v400.py (TRAIN only); VALIDATION / fresh panels are
generated on the fly (indices 1,000,000+ and 1,000,032+). TEST is never generated.
"""
import argparse, hashlib, json, math, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapping_3d_data_v400 as d3
import importlib.util
_spec = importlib.util.spec_from_file_location('train_v378_kbit', HERE / 'train_v378.py'); _v378 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v378)
kbit_wire_quantize = _v378.kbit_wire_quantize          # 8+8 bit per coordinate, exact for selection (MU_BITS = Q_BITS = 8 in that module)

HOPS = (0, 1, 2, 4, 8)
WEIGHTS = {0: .5, 1: .125, 2: .125, 4: .125, 8: .125}
BX, BY, BZ = d3.BLOCKS
D = BX * BY * BZ                                                     # 256 coordinates
PRECISION_MIN, PRECISION_MAX = 1e-4, 4.
LATENT_PRIOR = 1e-4
LEVELS = 127
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


# ----------------------------------------------------------------------------- model
def public3(count, sx, sy, sz, size, device):
    x = (torch.arange(sx, device=device) + .5) / sx * 2 - 1
    y = (torch.arange(sy, device=device) + .5) / sy * 2 - 1
    z = (torch.arange(sz, device=device) + .5) / sz * 2 - 1
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    scale = torch.full_like(xx, 1. / size)
    return torch.stack((xx, yy, zz, scale))[None].expand(count, -1, -1, -1, -1)


class PatchTrunk3D(nn.Module):
    """strictly block-local: each of the 8x8x4 blocks (S/8 x S/8 x S/4 voxels) is encoded independently."""
    def __init__(self, size, width=128):
        super().__init__()
        self.size = size; self.bs = (size // BX, size // BY, size // BZ)
        self.net = nn.Sequential(nn.Conv3d(6, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv3d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv3d(96, width, 3, stride=2, padding=1), nn.SiLU())
        self.proj = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())
        self.width = width

    def forward(self, private):                                       # [N,S,S,S] int8
        n, s = private.shape[0], private.shape[1]
        xyz = public3(1, s, s, s, s, private.device)[0, :3]
        inp = torch.cat(((private >= 0).float()[:, None], (private == 0).float()[:, None], (private == 1).float()[:, None],
                         xyz[None].expand(n, -1, -1, -1, -1)), 1)   # [N,6,S,S,S]
        bx, by, bz = self.bs
        p = inp.unfold(2, bx, bx).unfold(3, by, by).unfold(4, bz, bz)          # [N,6,BX,BY,BZ,bx,by,bz]
        p = p.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(n * D, 6, bx, by, bz)
        f = self.net(p)
        f = torch.cat((f.mean((2, 3, 4)), f.amax((2, 3, 4))), 1)
        return self.proj(f).reshape(n, D, self.width)                 # block-major (x, y, z)


class OwnedHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, feats, private):
        n, s = private.shape[0], private.shape[1]
        out = self.mlp(feats)
        mu = out[..., 0].tanh()
        q = PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * out[..., 1].sigmoid()
        coverage = F.avg_pool3d((private >= 0).float()[:, None], (s // BX, s // BY, s // BZ)).flatten(1)   # [N,256]
        return mu, q, (coverage > 0).float()


class Decoder3D(nn.Module):
    """posterior (hard-8 mean, log var) on the 8x8x4 block grid -> known logit + occupancy logit at S^3."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.pre = nn.Sequential(nn.Conv3d(2, 64, 3, padding=1), nn.SiLU(), nn.Conv3d(64, 64, 3, padding=1), nn.SiLU())
        # resolution schedule from (16,16,8) to (S,S,S): double x,y,z until x,y reach S, then z alone
        shapes, chans = [], []
        sx, sy, sz = 16, 16, 8; c = 64
        while (sx, sy, sz) != (size, size, size):
            sx = min(sx * 2, size); sy = min(sy * 2, size); sz = min(sz * 2, size)
            shapes.append((sx, sy, sz)); chans.append(max(16, c * 3 // 4)); c = chans[-1]
        self.shapes = shapes
        blocks, incoming = [], 64
        for ch in chans:
            blocks.append(nn.Sequential(nn.Conv3d(incoming + 4, ch, 3, padding=1), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1), nn.SiLU()))
            incoming = ch
        self.blocks = nn.ModuleList(blocks)
        self.lift = nn.Sequential(nn.Conv3d(64 + 4, 64, 3, padding=1), nn.SiLU())
        self.output = nn.Conv3d(incoming, 2, 1)

    def forward(self, post):                                           # post: [b, 2D] = (hard-8 mean [D], log var [D])
        b = post.shape[0]
        grid = torch.cat((post[:, :D].reshape(b, 1, BX, BY, BZ), (post[:, D:] / 8.).reshape(b, 1, BX, BY, BZ)), 1)
        h = F.interpolate(self.pre(grid), (16, 16, 8), mode='trilinear', align_corners=False)
        h = self.lift(torch.cat((h, public3(b, 16, 16, 8, self.size, h.device)), 1))
        for shape, block in zip(self.shapes, self.blocks):
            h = F.interpolate(h, shape, mode='trilinear', align_corners=False)
            h = block(torch.cat((h, public3(b, *shape, self.size, h.device)), 1))
        out = self.output(h)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


class SheafModel3D(nn.Module):
    def __init__(self, size, seed):
        super().__init__()
        self.size = size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.trunk = PatchTrunk3D(size); self.head = OwnedHead(self.trunk.width); self.decoder = Decoder3D(size)

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


# ----------------------------------------------------------------------------- max-precision join on the forest (D coordinates)
def select(p_list, e_list):
    P = torch.stack(p_list, 0); E = torch.stack(e_list, 0)
    tie = torch.arange(P.shape[0], device=P.device, dtype=P.dtype).view(-1, 1) * 1e-15
    idx = (P - tie).argmax(0, keepdim=True)
    return P.gather(0, idx)[0], E.gather(0, idx)[0]


def rollout_max(mu, q_learned, mask, edges, rounds=8, stages=HOPS):
    """identical to train_v378.rollout_max with 64 -> D coordinates and the k-bit wire (source factor quantised once at the origin)."""
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


def surface_of(target_flat, size):
    """target surface voxels: occupied with a free 6-neighbour (both in the target). [n, S^3] bool."""
    t = target_flat.reshape(-1, 1, size, size, size)
    occ = (t == 1).float(); free = (t == 0).float()
    free_nb = F.max_pool3d(free, 3, stride=1, padding=1)
    return ((occ > 0) & (free_nb > 0)).reshape(target_flat.shape[0], -1)


def stage_loss(decoder, feats, target, size, w_surface=4.):
    known_logit, occ_logit = decoder(feats)
    known = (target >= 0).float()
    known_bce = F.binary_cross_entropy_with_logits(known_logit, known, reduction='none').mean(-1)
    occ = (target == 1).float()
    weight = known * (1. + w_surface * surface_of(target, size).float())
    occ_bce = (weight * F.binary_cross_entropy_with_logits(occ_logit, occ, reduction='none')).sum(-1) / weight.sum(-1).clamp_min(1)
    return known_bce.mean() + occ_bce.mean(), float(known_bce.mean()), float(occ_bce.mean())


@torch.no_grad()
def predict(decoder, feats):
    known_logit, occ_logit = decoder(feats)
    return torch.where(known_logit >= 0, (occ_logit >= 0).to(torch.int8), torch.full_like(occ_logit, -1, dtype=torch.int8))


@torch.no_grad()
def metrics3d(pred, target, size):
    """per node: occupied F1, surface F1 (target surface voxels vs predicted occupied), accuracy (3-way over target-known), unknown FPR."""
    rows = []
    surf = surface_of(target, size)
    for i in range(pred.shape[0]):
        p, t, s = pred[i], target[i], surf[i]
        known = t >= 0
        if known.sum() == 0:
            rows.append(dict(occ_f1=None, surface_f1=None, accuracy=None, unknown_known_fpr=float((p[~known] >= 0).float().mean()) if (~known).any() else None)); continue
        po = (p == 1) & known; to = (t == 1) & known
        tp = (po & to).sum().item(); fp = (po & ~to).sum().item(); fn = (~po & to).sum().item()
        occ_f1 = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
        # surface F1: target surface voxels as positives; predicted positives = predicted occupied voxels that are target-surface OR not target-occupied-interior
        ps = po & ~((t == 1) & ~s)                      # ignore predicted occupied inside solid interiors (correct but uninformative)
        tp_s = (ps & s).sum().item(); fp_s = (ps & ~s).sum().item(); fn_s = (~ps & s).sum().item()
        surface_f1 = 2 * tp_s / max(2 * tp_s + fp_s + fn_s, 1) if (tp_s + fp_s + fn_s) else None
        acc = float((p[known] == t[known]).float().mean())
        fpr = float((p[~known] >= 0).float().mean()) if (~known).any() else None
        rows.append(dict(occ_f1=occ_f1, surface_f1=surface_f1, accuracy=acc, unknown_known_fpr=fpr))
    return rows


def agg(rows):
    out = {}
    for k in ('occ_f1', 'surface_f1', 'accuracy', 'unknown_known_fpr'):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True, help='directory from gen_3d_cache_v400.py (manifest.json + npz per TRAIN scene)')
    ap.add_argument('--size', type=int, default=64); ap.add_argument('--agents', type=int, default=8); ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=342101); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--updates', type=int, default=12000, help='total updates (cosine schedule spans them); epochs = updates / scenes')
    ap.add_argument('--val-every', type=int, default=2000); ap.add_argument('--val-scenes', type=int, default=32)
    ap.add_argument('--deadline-epoch', type=float, default=1e18); ap.add_argument('--deadline-total', type=float, default=1e18)
    ap.add_argument('--max-updates', type=int, default=0, help='preflight: stop after this many updates')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sigterm); signal.signal(signal.SIGINT, _sigterm)
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir(); (out / 'validation').mkdir()
    device = torch.device(args.device); size = args.size; n_ag = args.agents
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    t0 = time.time()

    # ---- data
    cache = Path(args.cache); man = json.load(open(cache / 'manifest.json'))
    assert man['size'] == size and man['agents'] == n_ag and man['version'] == d3.VERSION and all(r['spec']['split'] == 'TRAIN' for r in man['rows'])
    assert man['data_module_sha256'] == sha_file(d3.__file__), 'cache written by a different data module'
    for r in man['rows'][:2] + man['rows'][-2:]:
        assert d3.array_hash(d3.make_fixture(**r['spec'])['private']) == r['private_sha256'], f"cache row {r['file']} does not regenerate"
    pin = (lambda x: x.pin_memory()) if device.type == 'cuda' else (lambda x: x)
    train = []
    for r in man['rows']:
        z = np.load(cache / r['file'])
        train.append(dict(spec=r['spec'], own_t=pin(torch.from_numpy(z['private'])), reached_t=pin(torch.from_numpy(z['reached'])),
                          edges=[(int(i), int(j), float(w)) for i, j, w in z['edges']]))
    def panel(base):
        scenes = []
        for k in range(args.val_scenes):
            fx = d3.make_fixture('VALIDATION', base + k, size=size, agents=n_ag)
            scenes.append(dict(spec=fx['spec'], own_t=torch.from_numpy(fx['private']).to(device), reached_t=torch.from_numpy(fx['reached']).to(device), edges=fx['edges'], occ=fx['occ']))
        return scenes
    val = panel(1_000_000); fresh = panel(1_000_000 + args.val_scenes)
    print(f'[3d] scenes train {len(train)} val {len(val)} fresh {len(fresh)} gen {time.time()-t0:.0f}s', flush=True)

    model = SheafModel3D(size, args.seed).to(device)
    groups = model.parameter_groups()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    total = args.updates
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1., (u + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(u, total) / max(1, total)))))
    meta = dict(version=d3.VERSION, size=size, agents=n_ag, seed=args.seed, D=D, blocks=list(d3.BLOCKS), parameters={k: sum(p.numel() for p in v) for k, v in groups.items()},
                total_parameters=sum(p.numel() for p in model.parameters()), initial_weights_sha256=model.weights_hash(), train_scenes=len(train), updates=total, lr=args.lr,
                cache=dict(dir=str(cache), manifest_sha256=sha_file(cache / 'manifest.json')), args=vars(args), wire='kbit 8+8', fusion='max', readout='posterior_only',
                bits_per_coordinate_on_wire=16, raw_code_bytes=D * 2)
    json.dump(meta, open(out / 'run.json', 'w'), indent=1)
    torch.save(dict(state_dict=model.state_dict(), meta=meta, updates=0), out / 'checkpoints' / 'initial.pt')

    def save(name, updates):
        torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), updates=updates, meta=meta, weights_sha256=model.weights_hash()), out / 'checkpoints' / name)

    @torch.no_grad()
    def evaluate(scenes, label):
        rows = []
        for si, sc in enumerate(scenes):
            mu, q, mask = model.source(sc['own_t'])
            feats = rollout_max(mu, q, mask, sc['edges'])
            hops = {}; preds = {}
            for k in HOPS:
                tgt = stream_targets(sc['own_t'], sc['reached_t'][HOPS.index(k)])
                pred = predict(model.decoder, feats[k])
                hops[str(k)] = agg(metrics3d(pred, tgt, size)); preds[k] = pred
            rows.append(dict(spec=sc['spec'], hops=hops, mask_fraction=float(mask.mean()), q_mean=float((q * mask).sum() / mask.sum().clamp_min(1))))
            if si < 2:
                np.savez_compressed(out / 'validation' / f"{label}-panel-{si}.npz", own=sc['own_t'].cpu().numpy(), occ=sc['occ'], mu=mu.cpu().numpy(), q=(q * mask).cpu().numpy(),
                                    **{f'pred_k{k}': preds[k].cpu().numpy().reshape(-1, size, size, size) for k in HOPS})
        summary = {str(k): {m: (float(np.mean([r['hops'][str(k)][m] for r in rows if r['hops'][str(k)][m] is not None])) if any(r['hops'][str(k)][m] is not None for r in rows) else None)
                            for m in ('occ_f1', 'surface_f1', 'accuracy', 'unknown_known_fpr')} for k in HOPS}
        return dict(rows=rows, summary=summary)

    def fmt(s):
        return ' | '.join(f"R{k}: surf {s[str(k)]['surface_f1']:.3f} occ {s[str(k)]['occ_f1']:.3f} acc {s[str(k)]['accuracy']:.3f} fpr {s[str(k)]['unknown_known_fpr']:.3f}" for k in HOPS)

    # ---- training
    history = []; updates = 0; best = (-1., None); losses = []
    order_rng = np.random.default_rng(args.seed); perm = order_rng.permutation(len(train)); pos = 0
    model.train(); tick = time.time()
    while updates < total:
        if STOP or time.time() > args.deadline_epoch or (args.max_updates and updates >= args.max_updates):
            break
        if pos >= len(perm):
            perm = order_rng.permutation(len(train)); pos = 0
        sc = train[perm[pos]]; pos += 1
        own_d = sc['own_t'].to(device, non_blocking=True); reached_d = sc['reached_t'].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        mu, q, mask = model.source(own_d)
        feats = rollout_max(mu, q, mask, sc['edges'])
        total_loss = 0.; parts = {}
        for k in HOPS:
            tgt = stream_targets(own_d, reached_d[HOPS.index(k)])
            l, kn, oc = stage_loss(model.decoder, feats[k], tgt, size)
            total_loss = total_loss + WEIGHTS[k] * l; parts[k] = (kn, oc)
        total_loss.backward()
        logging = (updates + 1) % 50 == 0
        gnorm = {name: round(math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in ps if p.grad is not None)), 6) for name, ps in groups.items()} if logging else None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not math.isfinite(float(norm)):
            raise FloatingPointError('nonfinite gradient')
        opt.step(); sched.step(); updates += 1; losses.append(float(total_loss))
        if logging:
            print(f"[3d] upd{updates} loss {np.mean(losses[-50:]):.4f} k0(kn,occ)={parts[0][0]:.3f},{parts[0][1]:.3f} k8={parts[8][0]:.3f},{parts[8][1]:.3f} grad {gnorm} lr {sched.get_last_lr()[0]:.2e} mask {float(mask.mean()):.2f} {time.time()-t0:.0f}s", flush=True)
        if updates % args.val_every == 0 or updates == total:
            model.eval(); ev = evaluate(val, f'val-u{updates:06d}'); model.train()
            rec = dict(updates=updates, train_loss=float(np.mean(losses[-args.val_every:])), seconds=time.time() - tick, weights_sha256=model.weights_hash(), validation=ev['summary'])
            json.dump(dict(updates=updates, **ev), open(out / 'validation' / f'upd-{updates:06d}.json', 'w'), indent=1)
            print(f"[3d] VAL u{updates}: {fmt(ev['summary'])}", flush=True)
            score = ev['summary']['8']['surface_f1']
            if score is not None and score > best[0]:
                best = (score, updates); save('best.pt', updates)
            save('latest.pt', updates); history.append(rec); json.dump(history, open(out / 'history.json', 'w'), indent=1)
    model.eval()
    final = dict(updates=updates, weights_sha256=model.weights_hash(), seconds=time.time() - t0)
    save('final.pt', updates)
    if time.time() < args.deadline_total:
        ev = evaluate(val, 'final-val'); final['val'] = ev['summary']; json.dump(ev, open(out / 'validation' / 'final-val.json', 'w'), indent=1); print(f"[3d] FINAL val: {fmt(ev['summary'])}", flush=True)
    if time.time() < args.deadline_total:
        ev = evaluate(fresh, 'final-fresh'); final['fresh'] = ev['summary']; json.dump(ev, open(out / 'validation' / 'final-fresh.json', 'w'), indent=1); print(f"[3d] FINAL fresh: {fmt(ev['summary'])}", flush=True)
    json.dump(final, open(out / 'final.json', 'w'), indent=1)
    print(f'[3d] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
