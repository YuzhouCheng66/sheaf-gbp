"""Read-only decodability diagnostic for the v310 E12 anchor (Claude, 2026-10-21).

Question: given the FROZEN v310 decoder/lift, what is the best wall F1 that ANY
64-dim hard-8-bit code (with the actual GBP log-variance features held fixed)
can reach on (a) the K0 own target and (b) the R8 actually-arrived union target?

This separates "encoder/fusion loses the information" from "the decoder+code
family cannot express the target". Nothing is trained; model weights are
verified by SHA before load and hashed again after every fit. Outputs are
written only under --out (a fresh directory on /mnt/datasets).

Privileged optimisation (TRAIN truth is NOT used; VALIDATION targets are the
fit objective here, so this is an upper bound / attainability number, never a
deployable score).
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--checkpoint-sha', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sizes', default='128,256')
    ap.add_argument('--rooms', default='12,20')
    ap.add_argument('--agents', default='8,16')
    ap.add_argument('--indices', default='0,1,2,3')
    ap.add_argument('--stages', default='0,8')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--fit-extra', action='store_true', help='also optimise the 64 log-variance features')
    ap.add_argument('--max-scenes', type=int, default=0)
    ap.add_argument('--objective', default='bce', choices=['bce', 'dice', 'focal'])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    src = Path(args.source).resolve()
    sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
    from scripts.mapping_multiscale_v299_v310 import MultiscaleV299, native_context_decode
    from scripts import mapping_multiroom_forest_v295 as forest
    from scripts import mapping_multiroom_data_v293 as data
    from scripts import mapping_native_bank_v306 as bank
    from scripts import mapping_support_training_v306 as losses
    from scripts.mapping_recognition_metrics_v277 import node_metrics
    from scripts.mapping_local_capacity_v269 import Q_MIN, Q_MAX

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    actual_sha = sha(args.checkpoint)
    assert actual_sha == args.checkpoint_sha, 'checkpoint SHA mismatch; refusing to load'
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = MultiscaleV299.from_payload(payload['model'])
    model_sha = model.weights_hash()
    device = torch.device(args.device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print('loaded', actual_sha[:12], 'model', model_sha[:12], 'device', device, flush=True)

    r = model.readout

    def decode_free(code, extra, size):
        """Same arithmetic as CommonMap64.decode / _decode_native_once, without the hard-8 check."""
        extra_n = (extra - r.normalization_mean[64:]) / r.normalization_std[64:]
        lifted = (F.linear(code, r.lift.weight[:, :64].contiguous(), r.lift.bias)
                  + F.linear(extra_n, r.lift.weight[:, 64:].contiguous(), None))
        mean_raw, q_raw = lifted.chunk(2, -1)
        q = Q_MIN + (Q_MAX - Q_MIN) * q_raw.sigmoid()
        h = q * mean_raw.tanh()
        if size == 128:
            logp = r.decoder(q[:, None], h[:, None], 128, 128)[:, 0]
        else:
            logp = native_context_decode(r.decoder, q[:, None], h[:, None], size)[:, 0]
        known = (torch.logsumexp(logp[..., 1:], -1) - logp[..., 0]).flatten(1)
        wall = (logp[..., 2] - logp[..., 1]).flatten(1)
        return dict(known_logits=known[None], wall_logits=wall[None])

    def hard8(z):
        return (torch.round(z.clamp(-1, 1) * 127.) / 127.).float()

    def metrics_of(heads, scene, k, shape):
        known = heads['known_logits'][0].detach().cpu().numpy().reshape(shape)
        wall = heads['wall_logits'][0].detach().cpu().numpy().reshape(shape)
        pred = np.where(known >= 0, (wall >= 0).astype(np.int8), -1).astype(np.int8)
        rows = node_metrics(pred, scene['private'], scene['targets'][k], {'door_mask': scene['door_mask']})
        keys = ['union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr',
                'own_wall_f1', 'own_typed_accuracy', 'peer_wall_f1', 'known_recall']
        agg = {}
        for key in keys:
            vals = [row[key] for row in rows if row.get(key) is not None]
            agg[key] = float(np.mean(vals)) if vals else None
        return agg, pred

    sizes = [int(v) for v in args.sizes.split(',')]
    rooms_list = [int(v) for v in args.rooms.split(',')]
    agents_list = [int(v) for v in args.agents.split(',')]
    indices = [int(v) for v in args.indices.split(',')]
    stages = [int(v) for v in args.stages.split(',')]
    HOPS = (0, 1, 2, 4, 8)
    results = []
    t0 = time.time()
    count = 0
    for size in sizes:
        for rooms in rooms_list:
            for agents in agents_list:
                for index in indices:
                    if args.max_scenes and count >= args.max_scenes:
                        break
                    count += 1
                    spec = dict(split='VALIDATION', index=index, rooms=rooms, agents=agents, size=size)
                    fx = data.make_fixture(**spec)
                    inp = fx['inputs']
                    targets, reached = {}, {}
                    for k in HOPS:
                        targets[k], reached[k] = bank.forest_targets(inp, k)
                    fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
                    scene = dict(private=inp['private'], targets=targets, reached=reached, door_mask=fx['evaluation']['door_mask'])
                    own = torch.as_tensor(inp['private'][None], device=device)
                    with torch.no_grad():
                        roll = forest.rollout(model, own, fsel['edges'], decode_rounds=(), serialize=False, trace=False)
                    shape = inp['private'].shape
                    row = dict(spec=spec, private_sha256=data.array_hash(inp['private']), stages={})
                    for k in stages:
                        feats = roll['features'][k].detach()  # [1,N,128]
                        with torch.no_grad():
                            heads = model.decode(feats, resolution=size)
                        actual, pred_actual = metrics_of(heads, scene, k, shape)
                        target = torch.as_tensor(targets[k][None], device=device)
                        code0 = feats[0, :, :64].clone()
                        extra0 = feats[0, :, 64:].clone()
                        z = code0.clone().requires_grad_(True)
                        params = [z]
                        if args.fit_extra:
                            e = extra0.clone().requires_grad_(True); params.append(e)
                        else:
                            e = extra0
                        opt = torch.optim.Adam(params, lr=args.lr)
                        best = None
                        curve = []
                        for step in range(args.steps):
                            opt.zero_grad(set_to_none=True)
                            # STE: forward with hard-8 code so the optimum is a representable code
                            zq = hard8(z).detach() + (z.clamp(-1, 1) - z.clamp(-1, 1).detach())
                            heads_f = decode_free(zq, e, size)
                            terms = losses.proper_terms(heads_f, target)
                            if args.objective == 'bce':
                                objective = terms['total']
                            else:
                                flat = target.reshape(1, target.shape[1], -1)
                                knownm = (flat >= 0).float(); wallt = (flat == 1).float()
                                pw = torch.sigmoid(heads_f['wall_logits']) * knownm
                                if args.objective == 'dice':
                                    inter = (pw * wallt).sum(-1); dice = (2 * inter + 1.) / (pw.sum(-1) + wallt.sum(-1) + 1.)
                                    objective = terms['known'] + (1. - dice).mean()
                                else:
                                    bce = F.binary_cross_entropy_with_logits(heads_f['wall_logits'], wallt, reduction='none')
                                    pt = torch.exp(-bce); focal = ((1 - pt) ** 2 * bce * knownm).sum(-1) / knownm.sum(-1).clamp_min(1)
                                    objective = terms['known'] + 4. * focal.mean()
                            objective.backward()
                            opt.step()
                            with torch.no_grad():
                                z.clamp_(-1, 1)
                            if step % 50 == 0 or step == args.steps - 1:
                                curve.append((step, float(terms['total'])))
                        with torch.no_grad():
                            zq = hard8(z)
                            fit_feats = torch.cat((zq, e), -1)[None].float()
                            heads_fit = model.decode(fit_feats, resolution=size)  # official path, hard-8 checked
                            fitted, pred_fit = metrics_of(heads_fit, scene, k, shape)
                            # continuous (unquantised) code bound as well
                            heads_c = decode_free(z.clamp(-1, 1), e, size)
                            cont, _ = metrics_of(heads_c, scene, k, shape)
                        row['stages'][str(k)] = dict(actual=actual, fitted_hard8=fitted, fitted_continuous=cont,
                                                     loss_curve=curve, code_l2_move=float((zq - code0).norm()),
                                                     code_mean_abs_move=float((zq - code0).abs().mean()))
                        if index == 0:
                            np.savez_compressed(out / f'panel-H{size}-r{rooms}-n{agents}-i{index}-k{k}.npz',
                                                own=inp['private'], target=targets[k], pred_actual=pred_actual, pred_fit=pred_fit,
                                                code_actual=code0.cpu().numpy(), code_fit=zq.cpu().numpy(), extra=extra0.cpu().numpy())
                        print(f'H{size} r{rooms} n{agents} i{index} k{k} | actual wallF1 {actual["wall_f1"]:.3f} acc {actual["union_accuracy"]:.3f}'
                              f' | fit hard8 wallF1 {fitted["wall_f1"]:.3f} acc {fitted["union_accuracy"]:.3f} fpr {fitted["unknown_known_fpr"]:.3f}'
                              f' | cont wallF1 {cont["wall_f1"]:.3f} | {time.time()-t0:.0f}s', flush=True)
                    results.append(row)
                    with open(out / 'results.json', 'w') as f:
                        json.dump(dict(checkpoint_sha256=actual_sha, model_sha256=model_sha, model_sha256_after=model.weights_hash(),
                                       args=vars(args), rows=results, seconds=time.time() - t0), f, indent=1)
    assert model.weights_hash() == model_sha, 'model changed'
    # aggregate
    agg = {}
    for row in results:
        for k, v in row['stages'].items():
            key = f"H{row['spec']['size']}/k{k}"
            agg.setdefault(key, {'actual': [], 'fitted_hard8': [], 'fitted_continuous': []})
            for name in agg[key]:
                agg[key][name].append(v[name])
    summary = {}
    for key, groups in agg.items():
        summary[key] = {}
        for name, rows in groups.items():
            summary[key][name] = {m: float(np.mean([r[m] for r in rows if r[m] is not None])) for m in rows[0]}
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
