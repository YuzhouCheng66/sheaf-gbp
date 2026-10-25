"""Representation ceiling of the 3D storey code (batch 3S-L; Claude, 2026-10-25). The D4-6 protocol in 3D: NO encoder.
Each agent's code (mu and q per public block, D = BX x BY x BZ) is a FREE parameter; the decoder is the deployed Decoder3D of
train_3d_v424 (--decoder-width W, --layout); fusion is the deployed max-precision join with the deployed 8+8-bit wire (rollout_max).
  TRAIN (TRAIN houses --train-indices per stratum): decoder and per-(scene, agent) codes trained jointly on the K0 target (own map) and the
        R8 target (union over reached agents). The union loss does NOT update the codes (no co-adaptation; --union-code-grad allows it), as
        in the 2D protocol, because at VAL codes are fitted to the agent's own observation only.
  VAL   (the 24-house criterion panel): decoder frozen; each agent's code fitted to ITS OWN observation only (K0 loss, --fit-steps Adam
        steps); unions composed with the deployed join; scored at K0 and R8 with train_3d_v424.metrics3d against the actual targets, which
        are never fitted.
Reads: what a code of this layout CAN carry through THIS decoder family when the encoder is ideal (review v424 M3): it bounds block-local
encoders paired with this decoder at this budget (decoder trained on --train-indices houses), and is itself a LOWER bound on the true
representation ceiling (free codes may under-fit: the TRAIN-fitted vs VAL-own-fitted gap and the validity gate in chain_day6_3SL.sh read
that). Not reachable in deployment (the fit sees the decoder). Codes keep a CONSTANT learning rate (review M1: one scene per step means a
code is visited only steps / scenes times; a cosine on the codes capped |mu| at ~.4); the decoder follows the cosine. Layout 16x16x4
(D = 1,024) is diagnostic only (--allow-non256).
python owned_ceiling_3d_v424.py --source <sealed> --out DIR --layout 8x8x4 --decoder-width 1.0 --device cuda
"""
import argparse, json, multiprocessing, os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--layout', default='8x8x4'); ap.add_argument('--decoder-width', type=float, default=1.0); ap.add_argument('--allow-non256', action='store_true')
ap.add_argument('--train-indices', default='0-63'); ap.add_argument('--val-indices', default='0-7'); ap.add_argument('--rooms', default='1,2,3')
ap.add_argument('--steps', type=int, default=16000); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--code-lr', type=float, default=5e-2)
ap.add_argument('--fit-steps', type=int, default=300); ap.add_argument('--fit-lr', type=float, default=3e-2); ap.add_argument('--union-code-grad', action='store_true')
ap.add_argument('--time-cap', type=float, default=2400.); ap.add_argument('--seed', type=int, default=342101); ap.add_argument('--device', default='cuda')
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--gen-timeout', type=float, default=3600.); ap.add_argument('--train-eval', type=int, default=12)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v424 as T
import mapping_3d_storey_data_v420 as W
T.DECODER_WIDTH = args.decoder_width; T.BX, T.BY, T.BZ = T.LAYOUTS[args.layout]; T.D = T.BX * T.BY * T.BZ
assert T.D == 256 or (args.allow_non256 and args.layout in T.DIAGNOSTIC_LAYOUTS), 'D != 256 needs --allow-non256 (diagnostic only)'
VAL_KEYS = ('surface_f1', 'var_occ_f1', 'furn_f1', 'surface_f1_tol1', 'occ_f1', 'accuracy', 'unknown_known_fpr')


def rng_list(spec):
    a, b = (int(v) for v in spec.split('-')); return list(range(a, b + 1))


def coverage(private, dims):
    k = (dims[0] // T.BX, dims[1] // T.BY, dims[2] // T.BZ)
    return (F.avg_pool3d((private >= 0).float()[:, None], k).flatten(1) > 0).float()


def source(z_mu, z_q, mask):
    mu = T.latent_quantize(z_mu.tanh())
    q = T.PRECISION_MIN + (T.PRECISION_MAX - T.PRECISION_MIN) * z_q.sigmoid()
    return mu, q, mask


def to_dev(fx, dev, dims):
    own = torch.from_numpy(fx['private']).to(dev)
    return dict(spec=fx['spec'], own=own, reached=torch.from_numpy(fx['reached']).to(dev), edges=fx['edges'], chart_m=float(fx['chart_m']), mask=coverage(own, dims),
                slab=torch.from_numpy(fx['slab'].reshape(-1)).to(dev), obj=torch.from_numpy(fx['obj'].reshape(-1)).to(dev), struct=torch.from_numpy(fx['struct'].reshape(-1)).to(dev))


@torch.no_grad()
def score(decoder, sc, mu, q, dims):
    T.CHART_M = sc['chart_m']
    feats = T.rollout_max(mu, q, sc['mask'], sc['edges'], stages=(0, 8)); res = {}; preds = {}
    for k in (0, 8):
        tgt = T.stream_targets(sc['own'], sc['reached'][T.HOPS.index(k)]); pred = T.predict(decoder, feats[k])
        res[f'K{k}'] = T.agg(T.metrics3d(pred, tgt, dims, sc['slab'], sc['obj'], sc['struct'])); preds[k] = pred
    return res, preds


if __name__ == '__main__':
    t0 = time.time(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    dims = W.DIMS['R1']; dev = torch.device(args.device); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    rooms = [int(r) for r in args.rooms.split(',')]
    tr_specs = [dict(split='TRAIN', index=i, rooms=s, dims=list(dims), agents=W.AGENTS) for s in rooms for i in rng_list(args.train_indices)]
    va_specs = [dict(split='VALIDATION', index=i, rooms=s, dims=list(dims), agents=W.AGENTS) for s in rooms for i in rng_list(args.val_indices)]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, tr_specs + va_specs, chunksize=1).get(timeout=args.gen_timeout)
    train = [to_dev(fx, dev, dims) for fx in fxs[:len(tr_specs)] if fx is not None]; val = [to_dev(fx, dev, dims) for fx in fxs[len(tr_specs):] if fx is not None]; del fxs
    print(f'[ceil] layout {args.layout} D {T.D} width {args.decoder_width} | scenes train {len(train)} val {len(val)} gen {time.time()-t0:.0f}s', flush=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed); decoder = T.Decoder3D(dims).to(dev)
    n_dec = sum(p.numel() for p in decoder.parameters())
    codes = [(torch.zeros((sc['own'].shape[0], T.D), device=dev, requires_grad=True), torch.zeros((sc['own'].shape[0], T.D), device=dev, requires_grad=True)) for sc in train]
    opt = torch.optim.Adam([dict(params=decoder.parameters(), lr=args.lr), dict(params=[z for pair in codes for z in pair], lr=args.code_lr)])
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda u: .05 + .95 * .5 * (1 + math.cos(math.pi * min(u, args.steps) / args.steps)), lambda u: 1.])   # decoder cosine, codes constant (review M1)
    log = []; t_train = time.time(); order = np.random.permutation(len(train)); pos = 0; losses = []
    for step in range(args.steps):
        if pos >= len(order):
            order = np.random.permutation(len(train)); pos = 0
        si = int(order[pos]); pos += 1; sc = train[si]; T.CHART_M = sc['chart_m']
        opt.zero_grad(set_to_none=True)
        mu, q, mask = source(codes[si][0], codes[si][1], sc['mask'])
        f0 = T.rollout_max(mu, q, mask, sc['edges'], stages=(0,))[0]
        l0, _, _ = T.stage_loss(decoder, f0, T.stream_targets(sc['own'], sc['reached'][0]), dims)
        if args.union_code_grad:
            f8 = T.rollout_max(mu, q, mask, sc['edges'], stages=(8,))[8]
        else:
            f8 = T.rollout_max(mu.detach(), q.detach(), mask, sc['edges'], stages=(8,))[8]
        l8, _, _ = T.stage_loss(decoder, f8, T.stream_targets(sc['own'], sc['reached'][T.HOPS.index(8)]), dims)
        loss = .5 * l0 + .5 * l8; loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.); opt.step(); sched.step(); losses.append(float(loss.detach()))
        if step % 500 == 0 or step == args.steps - 1:
            log.append(dict(step=step, loss=float(np.mean(losses[-500:])), k0=float(l0), k8=float(l8), seconds=time.time() - t_train))
            print(f'[ceil] step {step} loss {np.mean(losses[-500:]):.4f} k0 {float(l0):.3f} k8 {float(l8):.3f} {time.time()-t0:.0f}s', flush=True)
        if time.time() - t_train > args.time_cap:
            print('[ceil] time cap at step', step, flush=True); log.append(dict(time_cap_step=step)); break
    steps_done = step + 1
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    # TRAIN sanity: trained codes on a few train scenes
    train_rows = []
    for si in range(0, len(train), max(1, len(train) // args.train_eval)):
        with torch.no_grad():
            mu, q, mask = source(codes[si][0], codes[si][1], train[si]['mask'])
        res, _ = score(decoder, train[si], mu, q, dims); train_rows.append(dict(spec=train[si]['spec'], **res))
    # VAL: fit each agent's code to its OWN observation only, decoder frozen; unions composed by the deployed join, never fitted
    val_rows = []; t_fit = time.time()
    for vi, sc in enumerate(val):
        T.CHART_M = sc['chart_m']; n = sc['own'].shape[0]
        z_mu = torch.zeros((n, T.D), device=dev, requires_grad=True); z_q = torch.zeros((n, T.D), device=dev, requires_grad=True)
        o = torch.optim.Adam([z_mu, z_q], lr=args.fit_lr); tgt0 = T.stream_targets(sc['own'], sc['reached'][0])
        fit_curve = []
        for it in range(args.fit_steps):
            o.zero_grad(set_to_none=True)
            mu, q, mask = source(z_mu, z_q, sc['mask'])
            l, _, _ = T.stage_loss(decoder, T.rollout_max(mu, q, mask, sc['edges'], stages=(0,))[0], tgt0, dims); l.backward(); o.step()
            if (it + 1) % 100 == 0 or it + 1 == args.fit_steps:
                fit_curve.append((it + 1, round(float(l.detach()), 4)))                                   # review S3: convergence of the fit
        with torch.no_grad():
            mu, q, mask = source(z_mu, z_q, sc['mask'])
        res, preds = score(decoder, sc, mu, q, dims)
        val_rows.append(dict(spec=sc['spec'], fit_loss=float(l.detach()), fit_curve=fit_curve, mu_abs_max=float(mu.abs().max()), mask_fraction=float(mask.mean()), q_mean=float((q * mask).sum() / mask.sum().clamp_min(1)), **res))
        f3 = lambda v: 'n/a' if v is None else f'{v:.3f}'                                                   # review S4: None-safe
        print(f"[ceil] VAL r{sc['spec']['rooms']} i{sc['spec']['index']} | K0 surf {f3(res['K0']['surface_f1'])} var {f3(res['K0']['var_occ_f1'])} furn {f3(res['K0']['furn_f1'])} | "
              f"R8 surf {f3(res['K8']['surface_f1'])} var {f3(res['K8']['var_occ_f1'])} furn {f3(res['K8']['furn_f1'])} fpr {f3(res['K8']['unknown_known_fpr'])} | fit {fit_curve} {time.time()-t0:.0f}s", flush=True)
        if vi < 2:
            np.savez_compressed(out / f'val-panel-{vi}.npz', own=sc['own'].cpu().numpy(), mu=mu.cpu().numpy(), q=(q * mask).cpu().numpy(),
                                **{f'pred_k{k}': preds[k].cpu().numpy().reshape(-1, *dims) for k in (0, 8)})

    def summarize(rows):
        return {h: {m: (float(np.mean([r[h][m] for r in rows if r[h][m] is not None])) if any(r[h][m] is not None for r in rows) else None) for m in VAL_KEYS} for h in ('K0', 'K8')}
    with torch.no_grad():
        mu_max_train = float(max(source(z[0], z[1], sc['mask'])[0].abs().max() for z, sc in zip(codes, train)))
    report = dict(args=vars(args), layout=args.layout, D=T.D, decoder_width=args.decoder_width, decoder_parameters=n_dec, steps_done=steps_done, train_scenes=len(train), val_scenes=len(val),
                  train_mu_abs_max=mu_max_train, complete=(steps_done == args.steps),
                  train_log=log, train_summary=summarize(train_rows), val_summary=summarize(val_rows), val_rows=val_rows, fit_seconds=round(time.time() - t_fit), seconds=round(time.time() - t0))
    torch.save(dict(state_dict=decoder.state_dict(), args=vars(args)), out / 'decoder.pt')
    json.dump(report, open(out / 'report.json', 'w'), indent=1)
    for name, s in (('TRAIN(fitted codes)', report['train_summary']), ('VAL(own-fitted)', report['val_summary'])):
        for h in ('K0', 'K8'):
            m = s[h]; f3 = lambda v: 'n/a' if v is None else f'{v:.3f}'
            print(f"[ceil] {name} {h}: surf {f3(m['surface_f1'])} var {f3(m['var_occ_f1'])} furn {f3(m['furn_f1'])} tol1 {f3(m['surface_f1_tol1'])} fpr {f3(m['unknown_known_fpr'])}", flush=True)
    print(f'[ceil] done {time.time()-t0:.0f}s', flush=True)
