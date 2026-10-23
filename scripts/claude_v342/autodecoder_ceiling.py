"""Decoder-family CEILING diagnostic (auto-decoder, DeepSDF style). Claude 2026-10-21.

Question: for a given decoder family D and a 64-dim hard-8-bit code, what is the
best wall F1 reachable on K0 own targets and on R8 actually-arrived union
targets when the code is chosen freely (oracle) per recipient?

Procedure
  1. TRAIN: jointly optimise D's weights and one free 64-code per (scene, agent,
     stage) on TRAIN worlds (v293 generator, split='TRAIN'). Codes pass through
     hard-8 STE so every code is representable on the 64-byte wire.
  2. VALIDATION: freeze D, fit a fresh 64-code per recipient for VALIDATION
     worlds (300 Adam steps from zero), report metrics with the sealed
     mapping_recognition_metrics_v277.node_metrics (pixel-exact, raw ternary).
This is an attainability bound for the decoder family, NOT a deployable model:
no encoder, no GBP, VALIDATION targets are used as the fitting objective.
It also yields TRAIN oracle codes that later encoder/fusion work can be
compared against.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HOPS = (0, 1, 2, 4, 8)


# ----------------------------------------------------------------------------- data
def build_scenes(data, bank, split, indices, rooms_list, agents_list, size, stages):
    scenes = []
    for rooms in rooms_list:
        for agents in agents_list:
            for index in indices:
                spec = dict(split=split, index=index, rooms=rooms, agents=agents, size=size)
                fx = data.make_fixture(**spec)
                inp = fx['inputs']
                targets = {}
                for k in stages:
                    targets[k], _ = bank.forest_targets(inp, k)
                scenes.append(dict(spec=spec, private=inp['private'], targets=targets,
                                   door_mask=fx['evaluation']['door_mask']))
    return scenes


def truncated_dt(wall, T):
    """Euclidean distance (cells) to nearest wall cell, truncated at T. wall: [B,H,W] bool."""
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


# ----------------------------------------------------------------------------- decoder families
def public(count, sy, sx, size, device):
    y = (torch.arange(sy, device=device) + .5) / sy * 2 - 1
    x = (torch.arange(sx, device=device) + .5) / sx * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    scale = torch.full_like(yy, 1. / size)
    return torch.stack((xx, yy, scale, scale))[None].expand(count, -1, -1, -1)


class CnnTrunk(nn.Module):
    """v121 ContextDecoder-shaped trunk: lift grid 16x16 -> 16,32,64,128(,256,512)."""
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


class Decoder(nn.Module):
    def __init__(self, family, size):
        super().__init__()
        self.family, self.size = family, size
        extra = int(math.log2(size // 128))
        if family.startswith('cnn'):
            # v299/v310-shaped: linear 64->2048 -> mean(tanh)/q(sigmoid) planes 4x16x16 (8 input planes incl. log1p q)
            self.lift = nn.Linear(64, 2048)
            self.trunk = CnnTrunk(8, extra_stages=extra)
        elif family.startswith('mlp'):
            self.lift = nn.Sequential(nn.Linear(64, 1024), nn.SiLU(), nn.Linear(1024, 16 * 16 * 16))
            self.trunk = CnnTrunk(16, extra_stages=extra)
        else:
            raise ValueError(family)

    def forward(self, code):
        if self.family.startswith('cnn'):
            mean_raw, q_raw = self.lift(code).chunk(2, -1)
            q = 1e-4 + (4. - 1e-4) * q_raw.sigmoid()
            grid = torch.cat((mean_raw.tanh().reshape(-1, 4, 16, 16), q.log1p().reshape(-1, 4, 16, 16)), 1)
        else:
            grid = self.lift(code).reshape(-1, 16, 16, 16)
        out = self.trunk(grid, self.size)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)  # known logit, wall channel

    def head_is_dt(self):
        return self.family.endswith('_dt')


def hard8_ste(z):
    zc = z.clamp(-1, 1)
    return (torch.round(zc * 127.) / 127.).detach() + (zc - zc.detach())


def loss_terms(decoder, codes, target, dt_target, T, w_near=4.):
    """target [B,H*W] int8 ternary; dt_target [B,H*W] float."""
    known_logit, wall_ch = decoder(codes)
    known = (target >= 0).float()
    known_bce = F.binary_cross_entropy_with_logits(known_logit, known, reduction='none').mean(-1)
    if decoder.head_is_dt():
        dt_pred = F.softplus(wall_ch)
        weight = known * (1. + w_near * (dt_target <= 1.5).float())
        wall_term = (weight * (dt_pred - dt_target).abs()).sum(-1) / weight.sum(-1).clamp_min(1)
    else:
        wall_bce = F.binary_cross_entropy_with_logits(wall_ch, (target == 1).float(), reduction='none')
        wall_term = (wall_bce * known).sum(-1) / known.sum(-1).clamp_min(1)
    return known_bce.mean() + wall_term.mean(), known_bce.mean(), wall_term.mean()


@torch.no_grad()
def predict(decoder, codes):
    known_logit, wall_ch = decoder(codes)
    if decoder.head_is_dt():
        wall = F.softplus(wall_ch) < .5
    else:
        wall = wall_ch >= 0
    return torch.where(known_logit >= 0, wall.to(torch.int8), torch.full_like(wall, -1, dtype=torch.int8))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='sealed v310 source root (for generator + metrics)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--family', required=True, choices=['cnn_bce', 'cnn_dt', 'mlp_bce', 'mlp_dt'])
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--train-indices', default='0-63')
    ap.add_argument('--train-rooms', default='6,12,20')
    ap.add_argument('--train-agents', default='8,16')
    ap.add_argument('--val-indices', default='0-7')
    ap.add_argument('--val-rooms', default='12,20')
    ap.add_argument('--val-agents', default='8,16')
    ap.add_argument('--stages', default='0,8')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--code-lr', type=float, default=2e-2)
    ap.add_argument('--fit-steps', type=int, default=300)
    ap.add_argument('--fit-lr', type=float, default=3e-2)
    ap.add_argument('--dt-trunc', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--time-cap', type=float, default=3000., help='seconds for TRAIN phase')
    args = ap.parse_args()

    def rng_list(text):
        out = []
        for part in text.split(','):
            if '-' in part:
                a, b = part.split('-'); out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    src = Path(args.source).resolve()
    sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
    from scripts import mapping_multiroom_data_v293 as data
    from scripts import mapping_native_bank_v306 as bank
    from scripts.mapping_recognition_metrics_v277 import node_metrics

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    stages = [int(v) for v in args.stages.split(',')]
    size = args.size; T = args.dt_trunc
    t0 = time.time()

    train = build_scenes(data, bank, 'TRAIN', rng_list(args.train_indices), [int(v) for v in args.train_rooms.split(',')],
                         [int(v) for v in args.train_agents.split(',')], size, stages)
    val = build_scenes(data, bank, 'VALIDATION', rng_list(args.val_indices), [int(v) for v in args.val_rooms.split(',')],
                       [int(v) for v in args.val_agents.split(',')], size, stages)
    print(f'scenes train {len(train)} val {len(val)} gen {time.time()-t0:.0f}s', flush=True)

    # flatten TRAIN targets to [M,H*W]
    rows = []
    for si, sc in enumerate(train):
        for k in stages:
            for a in range(sc['private'].shape[0]):
                rows.append((si, k, a))
    M = len(rows)
    tgt = torch.empty((M, size * size), dtype=torch.int8)
    for m, (si, k, a) in enumerate(rows):
        tgt[m] = torch.from_numpy(train[si]['targets'][k][a].reshape(-1))
    tgt = tgt.to(device)
    dts = torch.empty((M, size * size), dtype=torch.float32, device=device)
    for start in range(0, M, 256):
        chunk = tgt[start:start + 256].reshape(-1, size, size) == 1
        dts[start:start + 256] = truncated_dt(chunk, T).reshape(-1, size * size)
    print(f'train codes {M}; dt done {time.time()-t0:.0f}s', flush=True)

    decoder = Decoder(args.family, size).to(device)
    codes = torch.zeros((M, 64), device=device, requires_grad=True)
    opt = torch.optim.Adam([dict(params=decoder.parameters(), lr=args.lr), dict(params=[codes], lr=args.code_lr)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * .05)
    log = []
    t_train = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, M, (args.batch,), device=device)
        opt.zero_grad(set_to_none=True)
        total, kn, wl = loss_terms(decoder, hard8_ste(codes[idx]), tgt[idx], dts[idx], T)
        total.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
        opt.step(); sched.step()
        with torch.no_grad():
            codes.clamp_(-1, 1)
        if step % 200 == 0 or step == args.steps - 1:
            log.append(dict(step=step, total=float(total), known=float(kn), wall=float(wl), seconds=time.time() - t_train))
            print(f'step {step} total {float(total):.4f} known {float(kn):.4f} wall {float(wl):.4f} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t_train > args.time_cap:
            print('time cap reached at step', step, flush=True); break
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    # TRAIN-fit quality (oracle codes as trained)
    def evaluate(scene_list, code_tensor_fn, label):
        results = []
        for si, sc in enumerate(scene_list):
            n = sc['private'].shape[0]
            for k in stages:
                z = code_tensor_fn(si, k, n)
                pred = predict(decoder, hard8_ste(z)).cpu().numpy().reshape(n, size, size)
                rows_m = node_metrics(pred, sc['private'], sc['targets'][k], {'door_mask': sc['door_mask']})
                agg = {}
                for key in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'known_recall'):
                    vals = [r[key] for r in rows_m if r.get(key) is not None]
                    agg[key] = float(np.mean(vals)) if vals else None
                results.append(dict(spec=sc['spec'], stage=k, **agg))
                if si < 2:
                    np.savez_compressed(out / f'{label}-panel-r{sc["spec"]["rooms"]}-n{n}-i{sc["spec"]["index"]}-k{k}.npz',
                                        own=sc['private'], target=sc['targets'][k], pred=pred, code=hard8_ste(z).detach().cpu().numpy())
        return results

    index_of = {(si, k, a): m for m, (si, k, a) in enumerate(rows)}
    train_eval = evaluate(train[:24], lambda si, k, n: codes[[index_of[(si, k, a)] for a in range(n)]].detach(), 'train')

    # VALIDATION: fit fresh codes against frozen decoder
    val_eval = []
    fit_log = []
    for si, sc in enumerate(val):
        n = sc['private'].shape[0]
        for k in stages:
            target = torch.from_numpy(sc['targets'][k].reshape(n, -1)).to(device)
            dt_t = truncated_dt(target.reshape(n, size, size) == 1, T).reshape(n, -1)
            z = torch.zeros((n, 64), device=device, requires_grad=True)
            o = torch.optim.Adam([z], lr=args.fit_lr)
            for step in range(args.fit_steps):
                o.zero_grad(set_to_none=True)
                total, _, _ = loss_terms(decoder, hard8_ste(z), target, dt_t, T)
                total.backward(); o.step()
                with torch.no_grad():
                    z.clamp_(-1, 1)
            pred = predict(decoder, hard8_ste(z)).cpu().numpy().reshape(n, size, size)
            rows_m = node_metrics(pred, sc['private'], sc['targets'][k], {'door_mask': sc['door_mask']})
            agg = {}
            for key in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'known_recall'):
                vals = [r[key] for r in rows_m if r.get(key) is not None]
                agg[key] = float(np.mean(vals)) if vals else None
            val_eval.append(dict(spec=sc['spec'], stage=k, final_loss=float(total), **agg))
            if sc['spec']['index'] == 0:
                np.savez_compressed(out / f'val-panel-r{sc["spec"]["rooms"]}-n{n}-i0-k{k}.npz',
                                    own=sc['private'], target=sc['targets'][k], pred=pred, code=hard8_ste(z).detach().cpu().numpy())
            print(f"VAL r{sc['spec']['rooms']} n{n} i{sc['spec']['index']} k{k} wallF1 {agg['wall_f1']:.3f} acc {agg['union_accuracy']:.3f} fpr {agg['unknown_known_fpr']:.3f} {time.time()-t0:.0f}s", flush=True)

    def summarize(ev):
        s = {}
        for k in stages:
            for rooms in sorted({e['spec']['rooms'] for e in ev}):
                for agents in sorted({e['spec']['agents'] for e in ev}):
                    sel = [e for e in ev if e['stage'] == k and e['spec']['rooms'] == rooms and e['spec']['agents'] == agents]
                    if sel:
                        s[f'k{k}/rooms{rooms}/N{agents}'] = {m: float(np.mean([e[m] for e in sel if e[m] is not None])) for m in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr')}
            sel = [e for e in ev if e['stage'] == k]
            s[f'k{k}/all'] = {m: float(np.mean([e[m] for e in sel if e[m] is not None])) for m in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr')}
        return s

    report = dict(args=vars(args), family=args.family, size=size, train_codes=M, train_scenes=len(train), val_scenes=len(val),
                  decoder_parameters=sum(p.numel() for p in decoder.parameters()), train_log=log,
                  train_summary=summarize(train_eval), val_summary=summarize(val_eval), val_rows=val_eval, train_rows=train_eval,
                  seconds=time.time() - t0)
    torch.save(dict(state_dict=decoder.state_dict(), family=args.family, size=size, codes=codes.detach().cpu(), rows=rows), out / 'decoder_and_codes.pt')
    with open(out / 'report.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(dict(train=report['train_summary'], val=report['val_summary']), indent=1))


if __name__ == '__main__':
    main()
