#!/usr/bin/env python
"""diag_z_probe.py -- zero-training diagnostic after D8-88 (CHANNEL-NO-4): does the sender's z carry the identity of its own answer?
With a trained L1e (or L1d) checkpoint, for N VAL examples: pass 1 of each sender j -> z_j (K x d_e, flattened); the sender's own first answer
token (its own tokenizer); a linear probe (multinomial logistic regression, Adam, held-out split) from z_j to that token restricted to the
`--classes` most frequent first tokens (others -> one class); the majority baseline; the centred-cosine similarity of z between examples with the
same first token vs different tokens; and, on the real channel path, the receiver's decoder argmax per slot against the gold answer tokens
(position-wise accuracy, slot 0 separately). Reads only; writes <out>/probe.json.
"""
import argparse
import contextlib
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402


def probe(X, y, n_classes, seed=0, steps=400, lr=1e-2, train_frac=0.67):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_tr = int(train_frac * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X[tr] @ W + b, y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = float(((X[tr] @ W + b).argmax(1) == y[tr]).float().mean())
        acc_te = float(((X[te] @ W + b).argmax(1) == y[te]).float().mean())
        maj = int(torch.bincount(y[tr], minlength=n_classes).argmax())
        base_te = float((y[te] == maj).float().mean())
    return dict(train_acc=acc_tr, test_acc=acc_te, majority_test_acc=base_te, n_train=int(n_tr), n_test=int(n - n_tr))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', default='l1e_model', choices=('l1d_model', 'l1e_model'))
    ap.add_argument('--arm', default='Eu')
    ap.add_argument('--val', required=True)
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--receiver-from', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--classes', type=int, default=50)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    MX = __import__(args.module)
    val = M.DATA.read_jsonl(args.val)[:args.n]
    state, _ = MX.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
    pair = MX.build_pair(args.arm, 'hetero', dev, hf_cache=args.hf_cache, receiver_state=state, max_ctx=900, k=args.k, lora_dropout=0.0)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    M.load_trainable_state(pair, ck['state'], strict=True)
    pair.eval()
    ac = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else contextlib.nullcontext()
    Z = [[] for _ in range(M.N_AGENTS)]
    first_tok = [[] for _ in range(M.N_AGENTS)]
    sender_correct = [[] for _ in range(M.N_AGENTS)]
    slot_hits = [[] for _ in range(M.N_AGENTS)]                 # receiver i: decoder argmax == gold target per slot
    slot0_hits = [[] for _ in range(M.N_AGENTS)]
    with torch.no_grad(), ac:
        for batch in M.chunks(val, args.eval_batch):
            z, lam = pair.encode_both(batch)
            for j in range(M.N_AGENTS):
                _, answers = pair.sender_pieces(batch, j)
                Z[j].append(z[j].float().reshape(len(batch), -1).cpu())
                for b, ex in enumerate(batch):
                    ids = answers[b][1]
                    first_tok[j].append(int(ids[0]) if ids else -1)
                    sender_correct[j].append(float(M.DATA.exact_match(answers[b][0], ex['answer'])))
            xs, _, _ = pair.messages_eval(z, lam)
            for i in range(M.N_AGENTS):
                ag = pair.agent(i)
                pred = ag.reader.logits(xs[i]).argmax(-1).cpu()
                tgt, lens = pair.targets(batch, i)
                tgt = tgt.cpu()
                for b in range(len(batch)):
                    n_msg = max(1, lens[b])
                    slot_hits[i].append(float((pred[b, :n_msg] == tgt[b, :n_msg]).float().mean()))
                    slot0_hits[i].append(float(pred[b, 0] == tgt[b, 0]))
    out = dict(n=len(val), agents={})
    for j in range(M.N_AGENTS):
        X = torch.cat(Z[j])
        toks = first_tok[j]
        cnt = Counter(t for t in toks if t >= 0)
        top = [t for t, _ in cnt.most_common(args.classes)]
        idx = {t: k for k, t in enumerate(top)}
        y = torch.tensor([idx.get(t, len(top)) for t in toks])
        pr = probe(X, y, len(top) + 1)
        Xc = X - X.mean(0)
        Xn = Xc / (Xc.norm(dim=1, keepdim=True) + 1e-9)
        cos = Xn @ Xn.t()
        same = torch.tensor([[1 if (a == b2 and a >= 0) else 0 for b2 in toks] for a in toks], dtype=torch.bool)
        eye = torch.eye(len(toks), dtype=torch.bool)
        same_mean = float(cos[same & ~eye].mean()) if bool((same & ~eye).any()) else None
        diff_mean = float(cos[~same & ~eye].mean())
        name = pair.agent(j).name
        out['agents'][name] = dict(role='sender', first_token_probe=pr, top_classes=len(top), coverage_of_top=float(sum(cnt[t] for t in top) / max(1, len(toks))),
                                   most_common_first_tokens=[(pair.agent(j).tok.decode([t]), c) for t, c in cnt.most_common(8)],
                                   cos_same_first_token=same_mean, cos_different_first_token=diff_mean, sender_answer_em=float(np.mean(sender_correct[j])),
                                   as_receiver=dict(decoder_slot_acc_vs_gold=float(np.mean(slot_hits[j])), decoder_slot0_acc_vs_gold=float(np.mean(slot0_hits[j]))))
        print(name, json.dumps(out['agents'][name], ensure_ascii=False))
    json.dump(out, open(os.path.join(args.out, 'probe.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
