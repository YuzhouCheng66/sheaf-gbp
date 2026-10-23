"""[v370 variant: --world houseexpo and --allow-non64 (diagnostic reference only), Claude 2026-10-22]
Ownership-structured 64-code ceiling with EXACT diagonal-Gaussian fusion (Claude 2026-10-21).

Hypothesis H2: if each of the 64 variables is owned by a fixed public spatial
block (q_d = 1 iff the agent observed block d, else 0 and mean 0), then the
diagonal precision-weighted average performed by Sheaf-GBP/forest fusion IS the
union composition, exactly. Question: how much wall fidelity can such a code
carry at 128 (and 256), when each agent encodes ONLY its own observation?

Protocol (no encoder, no GBP code needed - fusion is closed form):
  TRAIN: free owned K0 code per (scene, agent), masked to observed blocks, hard
         k-bit STE; decoder D trained on K0 targets (own code) and R8 targets
         (union code = per-dim weighted average over actually reached sources,
         exactly what the forest GBP computes with q in {0,1}).
  VAL:   fit each agent's owned code against ITS OWN observation only (the
         deployable local-encoding limit), compose unions by the same rule and
         score R1/R2/R4/R8 against the actual arrived-union targets with the
         sealed pixel-exact metrics. Union targets are NEVER fitted at VAL.
Compare families: owned 8x8x1, 4x4x4, 2x2x16 (less local), bits 8/16,
q binary vs fraction, against the dense free-code oracle from autodecoder_ceiling.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from autodecoder_ceiling import CnnTrunk, truncated_dt


def public(count, sy, sx, size, device):
    y = (torch.arange(sy, device=device) + .5) / sy * 2 - 1
    x = (torch.arange(sx, device=device) + .5) / sx * 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    scale = torch.full_like(yy, 1. / size)
    return torch.stack((xx, yy, scale, scale))[None].expand(count, -1, -1, -1)


class OwnedDecoder(nn.Module):
    """code [B,64] laid out as (planes, side, side); input grid = [code planes, log1p(count) planes] -> 16x16 -> trunk."""
    def __init__(self, side, planes, size, head):
        super().__init__()
        self.side, self.planes, self.size, self.head = side, planes, size, head
        extra = int(math.log2(size // 128))
        self.pre = nn.Sequential(nn.Conv2d(2 * planes, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.trunk = CnnTrunk(64, extra_stages=extra)

    def forward(self, code, logcount):
        b = code.shape[0]
        grid = torch.cat((code.reshape(b, self.planes, self.side, self.side), logcount.reshape(b, self.planes, self.side, self.side)), 1)
        grid = F.interpolate(self.pre(grid), (16, 16), mode='bilinear', align_corners=False) if self.side != 16 else self.pre(grid)
        out = self.trunk(grid, self.size)
        return out[:, 0].flatten(1), out[:, 1].flatten(1)

    def head_is_dt(self):
        return self.head == 'dt'


def quantize_ste(z, bits):
    levels = 2 ** (bits - 1) - 1
    zc = z.clamp(-1, 1)
    return (torch.round(zc * levels) / levels).detach() + (zc - zc.detach())


def block_observed(private, side, planes_split, min_cells):
    """private [N,H,H] int8 -> observed indicator per (plane, block) [N, planes*side*side] in (planes, side, side) layout.
    All planes of a block share the same ownership mask (a block owns `planes` scalars)."""
    n, h, _ = private.shape
    known = (private >= 0).float()[:, None]
    cnt = F.avg_pool2d(known, h // side) * (h // side) ** 2  # observed cells per block
    obs = (cnt >= min_cells).float()                        # [N,1,side,side]
    frac = (cnt / (h // side) ** 2)
    return obs.expand(-1, planes_split, -1, -1).reshape(n, -1), frac.expand(-1, planes_split, -1, -1).reshape(n, -1)


def loss_terms(decoder, code, logcount, target, dt_target, w_near=4.):
    known_logit, wall_ch = decoder(code, logcount)
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
def predict(decoder, code, logcount):
    known_logit, wall_ch = decoder(code, logcount)
    wall = (F.softplus(wall_ch) < .5) if decoder.head_is_dt() else (wall_ch >= 0)
    return torch.where(known_logit >= 0, wall.to(torch.int8), torch.full_like(wall, -1, dtype=torch.int8))


def fuse(codes, q, reached):
    """codes [N,64], q [N,64] (0/1 or fraction), reached [R,N] bool -> union codes [R,64], logcount [R,64]."""
    w = reached.float() @ q                      # [R,64] total precision per dim
    num = reached.float() @ (q * codes)
    code_u = torch.where(w > 0, num / w.clamp_min(1e-9), torch.zeros_like(num))
    return code_u, torch.log1p(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--owned', default='8x8x1', help='side x side x planes, product must be 64')
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--head', default='bce', choices=['bce', 'dt'])
    ap.add_argument('--q', default='binary', choices=['binary', 'fraction'])
    ap.add_argument('--min-cells', type=int, default=1)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--train-indices', default='0-63')
    ap.add_argument('--train-rooms', default='6,12,20')
    ap.add_argument('--train-agents', default='8,16')
    ap.add_argument('--val-indices', default='0-7')
    ap.add_argument('--val-rooms', default='12,20')
    ap.add_argument('--val-agents', default='8,16')
    ap.add_argument('--train-stages', default='0,8')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--code-lr', type=float, default=2e-2)
    ap.add_argument('--union-code-grad', action='store_true', help='let union losses update source codes (co-adaptation)')
    ap.add_argument('--fit-steps', type=int, default=300)
    ap.add_argument('--fit-lr', type=float, default=3e-2)
    ap.add_argument('--dt-trunc', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--time-cap', type=float, default=2400.)
    ap.add_argument('--world', default='v293', choices=['v293', 'houseexpo'])
    ap.add_argument('--allow-non64', action='store_true', help='DIAGNOSTIC ONLY: allow layouts with != 64 variables (violates the 64-dim constraint)')
    args = ap.parse_args()

    def rng_list(text):
        out = []
        for part in text.split(','):
            if '-' in part:
                a, b = part.split('-'); out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    side, side2, planes = (int(v) for v in args.owned.split('x'))
    D = side * side * planes
    assert side == side2 and (D == 64 or args.allow_non64), 'owned layout must have 64 variables'
    src = Path(args.source).resolve()
    sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
    from scripts import mapping_multiroom_data_v293 as data
    if args.world == 'houseexpo':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mapping_houseexpo_data_v365 as data
    from scripts import mapping_native_bank_v306 as bank
    from scripts.mapping_recognition_metrics_v277 import node_metrics

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); size = args.size; T = args.dt_trunc
    HOPS = (0, 1, 2, 4, 8)
    t0 = time.time()

    def build(split, indices, rooms_list, agents_list):
        scenes = []
        for rooms in rooms_list:
            for agents in agents_list:
                for index in indices:
                    spec = dict(split=split, index=index, rooms=rooms, agents=agents, size=size)
                    fx = data.make_fixture(**spec); inp = fx['inputs']
                    targets, reached = {}, {}
                    for k in HOPS:
                        targets[k], reached[k] = bank.forest_targets(inp, k)
                    priv = torch.from_numpy(inp['private']).to(device)
                    obs, frac = block_observed(priv, side, planes, args.min_cells)
                    scenes.append(dict(spec=spec, private=inp['private'], targets=targets, reached=reached,
                                       door_mask=fx['evaluation']['door_mask'], obs=obs, frac=frac,
                                       q=(obs if args.q == 'binary' else frac * obs)))
        return scenes

    train = build('TRAIN', rng_list(args.train_indices), [int(v) for v in args.train_rooms.split(',')], [int(v) for v in args.train_agents.split(',')])
    val = build('VALIDATION', rng_list(args.val_indices), [int(v) for v in args.val_rooms.split(',')], [int(v) for v in args.val_agents.split(',')])
    train_stages = [int(v) for v in args.train_stages.split(',')]
    print(f'scenes train {len(train)} val {len(val)} gen {time.time()-t0:.0f}s', flush=True)

    # per-scene code tensors
    codes = [torch.zeros((sc['private'].shape[0], D), device=device, requires_grad=True) for sc in train]
    items = [(si, k, a) for si, sc in enumerate(train) for k in train_stages for a in range(sc['private'].shape[0])]
    tgt_cache = {}
    for si, sc in enumerate(train):
        for k in train_stages:
            t = torch.from_numpy(sc['targets'][k].reshape(sc['private'].shape[0], -1)).to(device)
            tgt_cache[(si, k)] = (t, truncated_dt((t.reshape(-1, size, size) == 1), T).reshape(t.shape[0], -1))
    decoder = OwnedDecoder(side, planes, size, args.head).to(device)
    opt = torch.optim.Adam([dict(params=decoder.parameters(), lr=args.lr), dict(params=codes, lr=args.code_lr)])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * .05)
    log = []; t_train = time.time()
    for step in range(args.steps):
        pick = [items[i] for i in np.random.randint(0, len(items), args.batch)]
        opt.zero_grad(set_to_none=True)
        total_loss = 0.
        # group by (scene, stage) to vectorise
        groups = {}
        for si, k, a in pick:
            groups.setdefault((si, k), []).append(a)
        n_items = 0
        for (si, k), agents in groups.items():
            sc = train[si]; c = quantize_ste(codes[si], args.bits) * sc['obs']
            t, dt_t = tgt_cache[(si, k)]
            idx = torch.tensor(agents, device=device)
            if k == 0:
                code_in, logcount = c[idx], torch.log1p(sc['q'][idx])
            else:
                reached = torch.from_numpy(sc['reached'][k]).to(device)[idx]
                src_codes = c if args.union_code_grad else c.detach()
                code_in, logcount = fuse(src_codes, sc['q'], reached)
            total, _, _ = loss_terms(decoder, code_in, logcount, t[idx], dt_t[idx])
            total_loss = total_loss + total * len(agents); n_items += len(agents)
        total_loss = total_loss / n_items
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
        opt.step(); sched.step()
        with torch.no_grad():
            for c in codes:
                c.clamp_(-1, 1)
        if step % 200 == 0 or step == args.steps - 1:
            log.append(dict(step=step, total=float(total_loss), seconds=time.time() - t_train))
            print(f'step {step} total {float(total_loss):.4f} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t_train > args.time_cap:
            print('time cap at step', step, flush=True); break
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    def evaluate_scene(sc, c):
        """c: owned K0 codes [N,64] (already masked/quantised). Returns metrics per stage."""
        n = sc['private'].shape[0]; res = {}; preds = {}
        for k in HOPS:
            if k == 0:
                code_in, logcount = c, torch.log1p(sc['q'])
            else:
                reached = torch.from_numpy(sc['reached'][k]).to(device)
                code_in, logcount = fuse(c, sc['q'], reached)
            pred = predict(decoder, code_in, logcount).cpu().numpy().reshape(n, size, size)
            rows = node_metrics(pred, sc['private'], sc['targets'][k], {'door_mask': sc['door_mask']})
            agg = {}
            for key in ('union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr', 'own_wall_f1', 'peer_wall_f1', 'known_recall'):
                vals = [r[key] for r in rows if r.get(key) is not None]
                agg[key] = float(np.mean(vals)) if vals else None
            res[str(k)] = agg; preds[k] = pred
        return res, preds

    # TRAIN sanity on a few scenes with trained codes
    train_eval = []
    for si in range(0, len(train), max(1, len(train) // 12)):
        sc = train[si]
        with torch.no_grad():
            c = quantize_ste(codes[si], args.bits) * sc['obs']
        res, _ = evaluate_scene(sc, c); train_eval.append(dict(spec=sc['spec'], **{f'R{k}': v for k, v in res.items()}))

    # VAL: fit codes on OWN observation only, then compose unions
    val_eval = []
    for sc in val:
        n = sc['private'].shape[0]
        own_t = torch.from_numpy(sc['private'].reshape(n, -1)).to(device)
        own_dt = truncated_dt((own_t.reshape(n, size, size) == 1), T).reshape(n, -1)
        z = torch.zeros((n, D), device=device, requires_grad=True)
        o = torch.optim.Adam([z], lr=args.fit_lr)
        for step in range(args.fit_steps):
            o.zero_grad(set_to_none=True)
            c = quantize_ste(z, args.bits) * sc['obs']
            total, _, _ = loss_terms(decoder, c, torch.log1p(sc['q']), own_t, own_dt)
            total.backward(); o.step()
            with torch.no_grad():
                z.clamp_(-1, 1)
        with torch.no_grad():
            c = quantize_ste(z, args.bits) * sc['obs']
        res, preds = evaluate_scene(sc, c)
        val_eval.append(dict(spec=sc['spec'], **{f'R{k}': v for k, v in res.items()}))
        print(f"VAL r{sc['spec']['rooms']} n{n} i{sc['spec']['index']} | K0 wallF1 {res['0']['wall_f1']:.3f} acc {res['0']['union_accuracy']:.3f} | R1 {res['1']['wall_f1']:.3f} | R8 wallF1 {res['8']['wall_f1']:.3f} acc {res['8']['union_accuracy']:.3f} fpr {res['8']['unknown_known_fpr']:.3f} own {res['8']['own_wall_f1']:.3f} peer {res['8']['peer_wall_f1'] if res['8']['peer_wall_f1'] is None else round(res['8']['peer_wall_f1'],3)} | {time.time()-t0:.0f}s", flush=True)
        if sc['spec']['index'] == 0:
            np.savez_compressed(out / f"val-panel-r{sc['spec']['rooms']}-n{n}.npz", own=sc['private'], code=c.cpu().numpy(),
                                **{f'target_k{k}': sc['targets'][k] for k in HOPS}, **{f'pred_k{k}': preds[k] for k in HOPS})

    def summarize(ev):
        s = {}
        for k in HOPS:
            strata = {}
            for e in ev:
                strata.setdefault((e['spec']['rooms'], e['spec']['agents']), []).append(e[f'R{k}'])
            per = {}
            for key, rows in strata.items():
                per[f'rooms{key[0]}/N{key[1]}'] = {m: float(np.mean([r[m] for r in rows if r[m] is not None])) if any(r[m] is not None for r in rows) else None for m in rows[0]}
            allm = {m: float(np.mean([v[m] for v in per.values() if v[m] is not None])) if any(v[m] is not None for v in per.values()) else None for m in next(iter(per.values()))}
            s[f'R{k}'] = dict(all=allm, strata=per)
        return s

    report = dict(args=vars(args), owned=args.owned, bits=args.bits, head=args.head, q=args.q, size=size,
                  decoder_parameters=sum(p.numel() for p in decoder.parameters()), train_log=log,
                  train_summary=summarize(train_eval), val_summary=summarize(val_eval), val_rows=val_eval, seconds=time.time() - t0)
    torch.save(dict(state_dict=decoder.state_dict(), args=vars(args)), out / 'decoder.pt')
    with open(out / 'report.json', 'w') as f:
        json.dump(report, f, indent=1)
    for k in HOPS:
        m = report['val_summary'][f'R{k}']['all']
        print(f"VAL R{k}: wallF1 {m['wall_f1']:.3f} R {m['wall_recall']:.3f} P {m['wall_precision']:.3f} acc {m['union_accuracy']:.3f} fpr {m['unknown_known_fpr']:.3f} own {m['own_wall_f1']:.3f}")


if __name__ == '__main__':
    main()
