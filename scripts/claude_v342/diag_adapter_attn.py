#!/usr/bin/env python
"""diag_adapter_attn.py -- zero-training diagnostic after D8-87 (CHANNEL-NO-3): does the sender adapter's cross-attention read the input at all?
Hypothesis: the layer-L hidden states of these models have attention-sink / massive-activation positions (the first token) whose raw, un-normalised
states dominate k_proj(h), so every learned query attends to the same position and z is (nearly) example-independent -> content-free messages
in L1 / L1b / L1c / L1d alike.
Measures, on N VAL examples with a trained L1d checkpoint (and optionally at a fresh init):
  (1) per-position L2 norm of the layer-L hidden states (max position, norm ratio first token / median);
  (2) the cross-attention mass per block / head on position 0 and on the last `tail` positions (the sender's own answer tokens in L1d);
  (3) the spread of z across examples: mean over slots and dims of std_examples(z) divided by the mean |z| ("relative spread").
Reads only; writes <out>/attn.json. GPU or CPU (--device).
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402
import l1c_model as MC  # noqa: E402
import l1d_model as MD  # noqa: E402


class Capture:
    """replaces l1c_model's Fn.scaled_dot_product_attention inside the adapter call only"""

    def __init__(self):
        self.weights = []
        self.orig = MC.Fn.scaled_dot_product_attention

    def __call__(self, q, k, v, attn_mask=None, **kw):
        scores = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        w = torch.softmax(scores, -1)
        self.weights.append(w.detach().float().cpu())
        return self.orig(q, k, v, attn_mask=attn_mask, **kw)

    @contextlib.contextmanager
    def active(self):
        MC.Fn.scaled_dot_product_attention = self
        try:
            yield
        finally:
            MC.Fn.scaled_dot_product_attention = self.orig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--receiver-from', required=True)
    ap.add_argument('--ckpt', default='', help='L1d trainable.pt (adapters + decoders); empty = fresh init')
    ap.add_argument('--arm', default='Du')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--k', type=int, default=MD.K_HINT)
    ap.add_argument('--tail', type=int, default=6)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    val = M.DATA.read_jsonl(args.val)[:args.n]
    state, info = MD.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
    pair = MD.build_pair(args.arm, 'hetero', dev, hf_cache=args.hf_cache, receiver_state=state, max_ctx=900, k=args.k, lora_dropout=0.0)
    variant = 'init'
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        M.load_trainable_state(pair, ck['state'], strict=True)
        variant = 'checkpoint'
    pair.eval()
    out = dict(variant=variant, n=len(val), agents={})
    ac = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else contextlib.nullcontext()
    with torch.no_grad(), ac:
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            pb0, answers = pair.sender_pieces(val, i)
            pcs = [(p, a_ids, []) for (p, _, _), (_, a_ids) in zip(pb0, answers)]
            A = ag.assemble([(p, s, []) for p, s, _ in pcs], None, 'right', embeds=False)
            h = ag.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False).hidden_states[ag.layer].float()
            mask = A['mask']
            norms = h.norm(dim=-1)                                           # (B, T)
            lens = [l for l in A['lens']]
            first = norms[:, 0]
            med = torch.stack([norms[b, :lens[b]].median() for b in range(len(val))])
            argmax_pos = [int(norms[b, :lens[b]].argmax()) for b in range(len(val))]
            top_dims = h[0, 0].abs().topk(5)
            cap = Capture()
            with cap.active():
                z, lam = ag.adapter(h, mask)
            blocks = []
            for bi, w in enumerate(cap.weights):                              # (B, H, K, T)
                mass0 = w[:, :, :, 0].mean().item()
                tail = []
                for b in range(len(val)):
                    L = lens[b]
                    tail.append(w[b, :, :, max(0, L - args.tail):L].sum(-1).mean().item())
                ent = (-(w * (w + 1e-12).log()).sum(-1)).mean().item()
                top_pos = []
                for b in range(len(val)):
                    L = lens[b]
                    top_pos.append(int(w[b, :, :, :L].mean((0, 1)).argmax()))
                blocks.append(dict(block=bi, mass_on_pos0=mass0, mass_on_tail=float(np.mean(tail)), entropy_nats=ent, uniform_entropy_nats=float(np.log(np.mean(lens))),
                                   top_position_per_example=top_pos, lens=lens))
            zf = z.float()
            spread = float((zf.std(0).mean() / zf.abs().mean()))
            zc = zf.reshape(len(val), -1)
            zc = zc - zc.mean(0, keepdim=True)
            cos = (zc @ zc.t()) / (zc.norm(dim=1)[:, None] * zc.norm(dim=1)[None, :] + 1e-9)
            lg = ag.reader.logits(z)
            pred = lg.argmax(-1)
            decoded = [ag.tok.decode([t for t in pred[b].tolist() if t != ag.filler], skip_special_tokens=True) for b in range(min(6, len(val)))]
            out['agents'][ag.name] = dict(hidden_norm_first=first.mean().item(), hidden_norm_median=med.mean().item(), ratio_first_over_median=float((first / med).mean()),
                                          argmax_norm_position=argmax_pos, top_abs_dims_of_pos0=[int(x) for x in top_dims.indices.tolist()],
                                          top_abs_values_of_pos0=[round(float(x), 1) for x in top_dims.values.tolist()], cross_attention=blocks,
                                          z_relative_spread=spread, z_mean_abs=float(zf.abs().mean()), z_pairwise_cos_centered_mean=float((cos.sum() - cos.diag().sum()) / (len(val) * (len(val) - 1))),
                                          decoded_hints=decoded, sender_answers=[a[0] for a in answers[:6]], gold=[ex['answer'] for ex in val[:6]])
            print(ag.name, json.dumps({k: v for k, v in out['agents'][ag.name].items() if k not in ('cross_attention',)}, ensure_ascii=False)[:900])
            for blk in blocks:
                print('   block', blk['block'], 'mass pos0 %.3f tail %.3f entropy %.2f (uniform %.2f) top positions %s' % (blk['mass_on_pos0'], blk['mass_on_tail'], blk['entropy_nats'], blk['uniform_entropy_nats'], blk['top_position_per_example'][:8]))
    json.dump(out, open(os.path.join(args.out, f'attn_{variant}.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
