"""Inference-only diagnostic: LOCAL iterative code refinement at K0 on the frozen v310 E12 anchor.

Each agent refines ITS OWN 64-dim hard-8 code by gradient steps on the shared
public decoder against ITS OWN raw observation (exactly the K0 target; no peer
data, no GT, no private skip). The refined code replaces mu in the source
factor (q unchanged, or jointly refined in --refine-q mode); then the UNCHANGED
v295 forest GBP runs and all stages R0/1/2/4/8 are decoded by the unchanged
decoder. Compared against the anchor's amortised codes on the same scenes.

Read-only: weights verified by SHA, hashed before/after. Output only under --out.
Cost disclosure: each agent spends --steps decoder forward/backward passes
locally before its first message; no extra bytes on the wire.
"""
import argparse, hashlib, json, sys, time
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
    ap.add_argument('--rooms', default='6,12,20')
    ap.add_argument('--agents', default='8,16')
    ap.add_argument('--indices', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--refine-q', action='store_true')
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
    from scripts.mapping_recognition_hops_v277 import PRECISION_MIN, PRECISION_MAX, LATENT_PRIOR
    from scripts.mapping_implicit_model_v192 import latent_quantize

    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    actual_sha = sha(args.checkpoint)
    assert actual_sha == args.checkpoint_sha, 'checkpoint SHA mismatch'
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = MultiscaleV299.from_payload(payload['model'])
    model_sha = model.weights_hash()
    device = torch.device(args.device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    r = model.readout
    HOPS = (0, 1, 2, 4, 8)

    def decode_free(code, extra, size):
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

    def metrics_of(pred, scene, k):
        rows = node_metrics(pred, scene['private'], scene['targets'][k], {'door_mask': scene['door_mask']})
        keys = ['union_accuracy', 'wall_f1', 'wall_recall', 'wall_precision', 'unknown_known_fpr',
                'own_wall_f1', 'own_typed_accuracy', 'peer_wall_f1', 'known_recall', 'door_free_recall']
        agg = {}
        for key in keys:
            vals = [row[key] for row in rows if row.get(key) is not None]
            agg[key] = float(np.mean(vals)) if vals else None
        return agg

    def decode_pred(feats, size, shape):
        heads = model.decode(feats, resolution=size)
        known = heads['known_logits'][0].cpu().numpy().reshape(shape)
        wall = heads['wall_logits'][0].cpu().numpy().reshape(shape)
        return np.where(known >= 0, (wall >= 0).astype(np.int8), -1).astype(np.int8)

    sizes = [int(v) for v in args.sizes.split(',')]
    results = []; t0 = time.time()
    for size in sizes:
        for rooms in [int(v) for v in args.rooms.split(',')]:
            for agents in [int(v) for v in args.agents.split(',')]:
                for index in [int(v) for v in args.indices.split(',')]:
                    spec = dict(split='VALIDATION', index=index, rooms=rooms, agents=agents, size=size)
                    fx = data.make_fixture(**spec); inp = fx['inputs']
                    targets = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
                    fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
                    scene = dict(private=inp['private'], targets=targets, door_mask=fx['evaluation']['door_mask'])
                    own = torch.as_tensor(inp['private'][None], device=device)
                    shape = inp['private'].shape; n = shape[0]
                    # --- baseline (amortised encoder)
                    with torch.no_grad():
                        base_factors = model.source.factors(own)
                        roll0 = forest.rollout(model, own, fsel['edges'], decode_rounds=(), serialize=False, trace=False)
                        base = {str(k): metrics_of(decode_pred(roll0['features'][k], size, shape), scene, k) for k in HOPS}
                    # --- local refinement: each agent against its OWN observation only
                    mu0 = base_factors['mu'][0].detach().float()                      # [N,64] hard8
                    lq0 = base_factors['learned_q'][0].detach().float()               # [N,64]
                    prior = LATENT_PRIOR / (n * 64)
                    k0_target = own[0].reshape(1, n, -1)                             # own observation == K0 target
                    z = mu0.clone().requires_grad_(True)
                    params = [z]
                    if args.refine_q:
                        # parametrise learned_q through its sigmoid pre-image
                        s = ((lq0 - PRECISION_MIN) / (PRECISION_MAX - PRECISION_MIN)).clamp(1e-6, 1 - 1e-6)
                        qlogit = torch.log(s / (1 - s)).clone().requires_grad_(True); params.append(qlogit)
                    opt = torch.optim.Adam(params, lr=args.lr)
                    tstart = time.time()
                    for step in range(args.steps):
                        opt.zero_grad(set_to_none=True)
                        zq = hard8(z).detach() + (z.clamp(-1, 1) - z.clamp(-1, 1).detach())
                        lq = (PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * qlogit.sigmoid()) if args.refine_q else lq0
                        extra = torch.log(1. / (lq + prior))                           # K0 log marginal variance
                        heads = decode_free(zq, extra, size)
                        terms = losses.proper_terms(heads, own)
                        terms['total'].backward(); opt.step()
                        with torch.no_grad():
                            z.clamp_(-1, 1)
                    refine_seconds = time.time() - tstart
                    with torch.no_grad():
                        mu_ref = hard8(z)
                        lq_ref = (PRECISION_MIN + (PRECISION_MAX - PRECISION_MIN) * qlogit.sigmoid()).detach() if args.refine_q else lq0
                        active = (own[0] >= 0).flatten(1).any(-1)
                        evidence_q = active[:, None].double() * lq_ref.double()
                        precision = evidence_q + prior
                        refined = dict(q=torch.diag_embed(precision)[None], h=(evidence_q * mu_ref.double())[None], mu=mu_ref[None],
                                       learned_q=lq_ref[None], active=active[None], prior_precision=prior)
                        original_factors = model.source.factors
                        model.source.factors = lambda private, _f=refined: _f
                        try:
                            roll1 = forest.rollout(model, own, fsel['edges'], decode_rounds=(), serialize=False, trace=False)
                        finally:
                            model.source.factors = original_factors
                        ref = {str(k): metrics_of(decode_pred(roll1['features'][k], size, shape), scene, k) for k in HOPS}
                    row = dict(spec=spec, baseline=base, refined=ref, refine_seconds=refine_seconds,
                               code_mean_abs_move=float((mu_ref - mu0).abs().mean()), code_changed_fraction=float((mu_ref != mu0).float().mean()))
                    results.append(row)
                    print(f"H{size} r{rooms} n{agents} i{index} | K0 wallF1 {base['0']['wall_f1']:.3f}->{ref['0']['wall_f1']:.3f} | R1 {base['1']['wall_f1']:.3f}->{ref['1']['wall_f1']:.3f}"
                          f" | R8 {base['8']['wall_f1']:.3f}->{ref['8']['wall_f1']:.3f} acc {base['8']['union_accuracy']:.3f}->{ref['8']['union_accuracy']:.3f} fpr {base['8']['unknown_known_fpr']:.3f}->{ref['8']['unknown_known_fpr']:.3f}"
                          f" | own@R8 {base['8']['own_wall_f1']:.3f}->{ref['8']['own_wall_f1']:.3f} | {refine_seconds:.1f}s refine | {time.time()-t0:.0f}s", flush=True)
                    if index == 0:
                        np.savez_compressed(out / f'panel-H{size}-r{rooms}-n{agents}.npz', own=inp['private'],
                                            **{f'target_k{k}': targets[k] for k in HOPS},
                                            **{f'pred_base_k{k}': decode_pred(roll0['features'][k], size, shape) for k in HOPS},
                                            **{f'pred_ref_k{k}': decode_pred(roll1['features'][k], size, shape) for k in HOPS})
                    with open(out / 'results.json', 'w') as f:
                        json.dump(dict(checkpoint_sha256=actual_sha, model_sha256=model_sha, args=vars(args), rows=results, seconds=time.time() - t0), f, indent=1)
    assert model.weights_hash() == model_sha
    # README-style aggregate: per size, equal weight over rooms x N strata
    summary = {}
    for size in sizes:
        summary[str(size)] = {}
        for k in HOPS:
            for name in ('baseline', 'refined'):
                strata = {}
                for row in results:
                    if row['spec']['size'] != size: continue
                    strata.setdefault((row['spec']['rooms'], row['spec']['agents']), []).append(row[name][str(k)])
                means = []
                for rows in strata.values():
                    means.append({m: np.mean([x[m] for x in rows if x[m] is not None]) if any(x[m] is not None for x in rows) else None for m in rows[0]})
                summary[str(size)][f'R{k}/{name}'] = {m: (float(np.mean([mm[m] for mm in means if mm[m] is not None])) if any(mm[m] is not None for mm in means) else None) for m in means[0]}
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    for size in sizes:
        for k in HOPS:
            b = summary[str(size)][f'R{k}/baseline']; rr = summary[str(size)][f'R{k}/refined']
            print(f"H{size} R{k}: wallF1 {b['wall_f1']:.3f}->{rr['wall_f1']:.3f} | acc {b['union_accuracy']:.3f}->{rr['union_accuracy']:.3f} | fpr {b['unknown_known_fpr']:.3f}->{rr['unknown_known_fpr']:.3f} | own wallF1 {b['own_wall_f1']:.3f}->{rr['own_wall_f1']:.3f}")


if __name__ == '__main__':
    main()
