#!/usr/bin/env python
"""diag_l1b_channel.py -- zero-GPU diagnostics after D8-85 (verdict CHANNEL-NO). Not a pre-registered batch: readouts only, no training.

  ceiling  frozen receivers (no LoRA, no adapters) read the partner's gold supporting sentences placed AT THE SOFT-TOKEN SLOT of the
           latent prompt (prefix | slot | question) as (gold) exact token embeddings of the receiver's tokenizer, (gold_p0.7) blurred
           0.7 E[tok] + 0.3 mean(E), (gold_noise) E[tok] + N(0, (0.5 rms)^2), (rand) random vocabulary tokens, compared with (none) no slot
           and (text) the T-style prompt "Message from the other agent: <gold>". Question answered: can an UNTOUCHED LLM use information
           that arrives at the slot in (approximately) its own embedding coordinates? -- the premise of a decoder anchored to the
           embedding space (D8-86 candidate). Records keep the owner bucket (own / other / none).
  grad     at the saved S1b checkpoint (D8-85, runs/l1b/S1b-hetero-s342101/trainable.pt) and at a fresh init: the gradient norm of the
           aux loss and of the answer loss AT THE SOFT TOKENS versus at the LoRA / adapter / reader / w parameters, and the aux-loss
           sensitivity to the soft-token content (as produced / zeros / the batch rolled by one / own z only).

CPU only (uk-linux: nice 10, CUDA_VISIBLE_DEVICES=''); float32. Writes <out>/ceiling.json, <out>/grad.json, <out>/log.txt.
--tiny runs the random tiny models on the test fixture (local check of the code paths only).
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch  # noqa: E402
import l1_model as M  # noqa: E402
import l1b_model as MB  # noqa: E402

CONDITIONS = ('none', 'gold', 'gold_p0.7', 'gold_noise', 'rand', 'text')


def ac(dev):
    """bf16 autocast on CUDA (as l1_model.train_steps); no-op on CPU"""
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else contextlib.nullcontext()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Log:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')

    def __call__(self, msg):
        line = f'{utc()} {msg}'
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()


def read_examples(args):
    if args.tiny:
        import test_l1_model as TL
        return list(TL.EXAMPLES), list(TL.EXAMPLES)
    val = M.DATA.read_jsonl(args.val)
    train = M.DATA.read_jsonl(args.train) if args.train else []
    return train, val


def select(val, n):
    """the first n VAL examples whose answer lies in one shard (own / other buckets exist), in VAL order"""
    owned = [ex for ex in val if ex.get('answer_owner') is not None]
    return owned[:n]


# ================================================================ ceiling
@torch.no_grad()
def ceiling(args, log, tiny):
    dev = torch.device(args.device)
    pair = M.build_pair('B0', args.pair, dev, hf_cache=args.hf_cache, tiny=tiny, lora=bool(args.receiver_ckpt), grad_ckpt=False, max_ctx=args.max_ctx,
                        lora_dropout=0.0)
    if args.receiver_ckpt:                              # a B0 checkpoint: base + the B0 LoRA (answer format learned, never saw soft tokens), all frozen
        saved = torch.load(args.receiver_ckpt, map_location='cpu', weights_only=False)
        state = saved['state'] if isinstance(saved, dict) and 'state' in saved else saved
        missing = M.load_trainable_state(pair, state, strict=True)
        log(f'ceiling: receiver = B0 checkpoint {args.receiver_ckpt} ({len(state)} tensors; not loaded: {len(missing)})')
    for p_ in pair.parameters():
        p_.requires_grad_(False)
    pair.eval()
    _, val = read_examples(args)
    exs = select(val, args.n)
    conds = [c for c in args.conditions.split(',') if c]
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f'unknown condition {c}')
    g = torch.Generator().manual_seed(args.seed)
    stats = {}
    for i in range(M.N_AGENTS):
        ag = pair.agent(i)
        W = ag.embed.weight.detach().float()
        stats[i] = dict(mean=W.mean(0), rms=float(W.pow(2).mean().sqrt()), vocab=int(W.shape[0]))
    recs = {c: [] for c in conds}
    t_start = time.time()
    log(f'ceiling: {len(exs)} examples x {M.N_AGENTS} agents x {conds}; k <= {args.k}; max_ctx {args.max_ctx} (text {args.max_ctx_text}); '
        f'models {[pair.agent(i).name for i in range(M.N_AGENTS)]}')
    done = 0
    for n, ex in enumerate(exs):
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            log(f'ceiling: time budget {args.max_minutes} min reached after {n} examples; summarising what is done')
            break
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            facts = M.DATA.text_sender_target(ex, 1 - i)
            toks = ag.ids(facts)[:args.k]
            kk = max(1, len(toks))
            E = ag.embed.weight.detach().float()
            for c in conds:
                t0 = time.time()
                soft = None
                if c == 'text':
                    (pre, suf, _), _tr = M.make_pieces(ag, ex, i, 'receiver', args.max_ctx_text, target=None, message=facts)
                else:
                    (pre, suf, _), _tr = M.make_pieces(ag, ex, i, 'latent', args.max_ctx, target=None)
                    if c == 'gold':
                        soft = E[toks]
                    elif c == 'gold_p0.7':
                        soft = 0.7 * E[toks] + 0.3 * stats[i]['mean']
                    elif c == 'gold_noise':
                        soft = E[toks] + 0.5 * stats[i]['rms'] * torch.randn(len(toks), E.shape[1], generator=g).to(E.device)
                    elif c == 'rand':
                        ids = torch.randint(0, stats[i]['vocab'], (kk,), generator=g).to(E.device)
                        soft = E[ids]
                    if soft is not None:
                        soft = soft[None].to(ag.dtype)
                with ac(dev):
                    texts, gi = ag.generate([(pre, suf, [])], soft, args.max_new_tokens, want_logits=False)
                r = M._record(ex, i, ag.name, texts[0], gi['steps'][0], None)
                r['cond'] = c
                r['k'] = 0 if soft is None else int(soft.shape[1])
                r['seconds'] = round(time.time() - t0, 2)
                recs[c].append(r)
        done = n + 1
        if done % 4 == 0 or done == len(exs):
            el = time.time() - t_start
            ems = ' '.join(f"{c}:{100 * sum(r['em'] for r in recs[c]) / max(1, len(recs[c])):.1f}" for c in conds)
            log(f'ceiling: {done}/{len(exs)} examples, {el / 60:.1f} min, {el / max(1, done):.1f} s/example; EM {ems}')
    out = dict(part='ceiling', utc=utc(), receiver_ckpt=args.receiver_ckpt, n_examples=done, n_requested=len(exs), conditions=conds, k=args.k, max_ctx=args.max_ctx,
               max_ctx_text=args.max_ctx_text, max_new_tokens=args.max_new_tokens, models=[pair.agent(i).name for i in range(M.N_AGENTS)],
               embedding_rms={pair.agent(i).name: stats[i]['rms'] for i in range(M.N_AGENTS)}, seconds=time.time() - t_start, summary={}, paired={})
    for c in conds:
        out['summary'][c] = M.summarise(recs[c]) if recs[c] else None
    # paired view: gain of each condition over 'none' per (example, agent), by owner bucket
    if 'none' in conds and recs['none']:
        base = {(r['id'], r['agent']): r for r in recs['none']}
        for c in conds:
            if c == 'none':
                continue
            pairs = [(base[(r['id'], r['agent'])], r) for r in recs[c] if (r['id'], r['agent']) in base]
            for rel in ('other', 'own', 'none'):
                sel = [(a, b) for a, b in pairs if a['rel'] == rel]
                if sel:
                    out['paired'].setdefault(c, {})[rel] = dict(n=len(sel), em_none=100 * sum(a['em'] for a, _ in sel) / len(sel),
                                                              em_cond=100 * sum(b['em'] for _, b in sel) / len(sel),
                                                              gained=sum(b['em'] > a['em'] for a, b in sel), lost=sum(b['em'] < a['em'] for a, b in sel))
    if args.keep_records:
        out['records'] = recs
    json.dump(out, open(os.path.join(args.out, 'ceiling.json'), 'w'), indent=1, ensure_ascii=False)
    parts = []
    for c in conds:
        s = out['summary'][c]
        if s:
            parts.append(f"{c} EM {s['em']:.2f} other {s['em_by_owner']['other']['em']}")
    log('ceiling: ' + ' | '.join(parts))
    return out


# ================================================================ grad
def group_norms(module):
    sq = dict(lora=0.0, adapter=0.0, reader=0.0, w=0.0, other=0.0)
    cnt = dict(lora=0, adapter=0, reader=0, w=0, other=0)
    for n, p in module.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if 'lora_' in n:
            gkey = 'lora'
        elif '.adapter.' in n:
            gkey = 'adapter'
        elif '.reader.' in n:
            gkey = 'reader'
        elif n.endswith('w_raw'):
            gkey = 'w'
        else:
            gkey = 'other'
        sq[gkey] += float(p.grad.detach().float().pow(2).sum())
        cnt[gkey] += p.grad.numel()
    return {k: dict(norm=math.sqrt(v), n_params=cnt[k], rms=(math.sqrt(v / cnt[k]) if cnt[k] else None)) for k, v in sq.items()}


def losses_with_softs(pair, batch, softs):
    la, lx = [], []
    for i in range(M.N_AGENTS):
        a_, x_, _ = pair.agent(i).loss_aux(pair.pieces(batch, i, 'latent'), softs[i], pair.aux_ids(batch, i))
        la.append(a_)
        lx.append(x_)
    return torch.stack(la).mean(), torch.stack(lx).mean()


def grad_one(pair, batch, which):
    """which = 'aux' | 'ans': one backward of that loss alone -> gradient norms at the soft tokens and per parameter group"""
    pair.zero_grad(set_to_none=True)
    with ac(pair.device):
        z, lam = pair.encode_both(batch)
        softs, _ = pair.soft_from_train(z, lam, pair.arm, False, None)
        for s in softs:
            s.retain_grad()
        l_ans, l_aux = losses_with_softs(pair, batch, softs)
    loss = l_aux if which == 'aux' else l_ans
    loss.backward()
    res = dict(which=which, loss=float(loss.detach()), loss_ans=float(l_ans.detach()), loss_aux=float(l_aux.detach()), soft=[], params=group_norms(pair))
    for i, s in enumerate(softs):
        g = s.grad.detach().float()
        sn = float(s.detach().float().norm())
        res['soft'].append(dict(agent=i, grad_norm=float(g.norm()), grad_rms=float(g.pow(2).mean().sqrt()), soft_norm=sn,
                                soft_rms=float(s.detach().float().pow(2).mean().sqrt()), shape=list(s.shape),
                                first_order_rel_change=float(g.norm()) * sn))
    pair.zero_grad(set_to_none=True)
    return res


@torch.no_grad()
def sensitivity(pair, batch):
    with ac(pair.device):
        z, lam = pair.encode_both(batch)
        softs, _ = pair.soft_from_train(z, lam, pair.arm, False, None)
        out = {}
        la, lx = losses_with_softs(pair, batch, softs)
        out['as_produced'] = dict(ans=float(la), aux=float(lx))
        la, lx = losses_with_softs(pair, batch, [torch.zeros_like(s) for s in softs])
        out['zeros'] = dict(ans=float(la), aux=float(lx))
        if len(batch) > 1:
            la, lx = losses_with_softs(pair, batch, [torch.roll(s, 1, 0) for s in softs])
            out['rolled_batch'] = dict(ans=float(la), aux=float(lx))
        own = [pair.agent(i).reader(z[i], pair.agent(i).dtype) for i in range(M.N_AGENTS)]
        la, lx = losses_with_softs(pair, batch, own)
        out['own_z_only'] = dict(ans=float(la), aux=float(lx))
    return out


def grad(args, log, tiny):
    dev = torch.device(args.device)
    train, val = read_examples(args)
    src = train if train else val
    batches = [src[j:j + args.grad_batch] for j in range(0, args.grad_batch * args.grad_batches, args.grad_batch)]
    batches = [b for b in batches if b]
    variants = []
    if args.ckpt:
        variants.append(('checkpoint', args.ckpt))
    if not args.no_init:
        variants.append(('init', None))
    results = dict(part='grad', utc=utc(), batches=[[ex.get('id') for ex in b] for b in batches], variants={}, max_ctx=args.max_ctx)
    for vname, ck in variants:
        M.seed_all(args.seed)
        pair = MB.build_pair('S1b', args.pair, dev, hf_cache=args.hf_cache, tiny=tiny, lora=True, grad_ckpt=True, max_ctx=args.max_ctx, lora_dropout=0.0)
        if ck:
            saved = torch.load(ck, map_location='cpu', weights_only=False)
            state = saved['state'] if isinstance(saved, dict) and 'state' in saved else saved
            missing = M.load_trainable_state(pair, state, strict=True)
            log(f'grad[{vname}]: loaded {len(state)} tensors from {ck}; trainable params not in the file: {len(missing)}')
        pair.train()                                     # gradient checkpointing is active only in train mode; LoRA dropout is 0 here
        vres = dict(w=pair.w_value, reader_scale={pair.agent(i).name: float(pair.agent(i).reader.scale) for i in range(M.N_AGENTS)}, batches=[])
        for bi, batch in enumerate(batches):
            t0 = time.time()
            r = dict(ids=[ex.get('id') for ex in batch])
            r['aux'] = grad_one(pair, batch, 'aux')
            a = r['aux']
            log(f"grad[{vname}] batch {bi}: aux loss {a['loss']:.4f}; |g_soft| {[round(s['grad_norm'], 6) for s in a['soft']]}; "
                f"|g_lora| {a['params']['lora']['norm']:.4g} |g_adapter| {a['params']['adapter']['norm']:.4g} |g_reader| {a['params']['reader']['norm']:.4g} "
                f"({time.time() - t0:.0f} s)")
            r['ans'] = grad_one(pair, batch, 'ans')
            a = r['ans']
            log(f"grad[{vname}] batch {bi}: ans loss {a['loss']:.4f}; |g_soft| {[round(s['grad_norm'], 6) for s in a['soft']]}; "
                f"|g_lora| {a['params']['lora']['norm']:.4g} |g_adapter| {a['params']['adapter']['norm']:.4g}")
            pair.eval()
            r['sensitivity'] = sensitivity(pair, batch)
            pair.train()
            sens = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r['sensitivity'].items()}
            log(f"grad[{vname}] batch {bi}: sensitivity {json.dumps(sens)} ({time.time() - t0:.0f} s total)")
            vres['batches'].append(r)
            partial = dict(results)
            partial['variants'] = dict(results['variants'])
            partial['variants'][vname] = vres
            json.dump(partial, open(os.path.join(args.out, 'grad.json'), 'w'), indent=1)
        results['variants'][vname] = vres
        del pair
    json.dump(results, open(os.path.join(args.out, 'grad.json'), 'w'), indent=1)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('ceiling', 'grad', 'both'), default='both')
    ap.add_argument('--pair', default='hetero')
    ap.add_argument('--val')
    ap.add_argument('--train')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='', help='S1b trainable.pt (grad); empty = init only')
    ap.add_argument('--no-init', action='store_true')
    ap.add_argument('--n', type=int, default=64, help='ceiling: VAL examples (with an answer owner)')
    ap.add_argument('--k', type=int, default=M.K_SLOTS)
    ap.add_argument('--conditions', default=','.join(CONDITIONS))
    ap.add_argument('--max-ctx', type=int, default=900)
    ap.add_argument('--max-ctx-text', type=int, default=972)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--max-minutes', type=float, default=0.0)
    ap.add_argument('--grad-batch', type=int, default=2)
    ap.add_argument('--grad-batches', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--keep-records', action='store_true')
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    ap.add_argument('--receiver-ckpt', default='', help='ceiling: a B0 trainable.pt; the receiver is then base + B0 LoRA, frozen')
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = Log(os.path.join(args.out, 'log.txt'))
    if args.threads:
        torch.set_num_threads(args.threads)
    log(f'diag_l1b_channel {args.part} threads {torch.get_num_threads()} torch {torch.__version__} '
        f'cuda_visible {os.environ.get("CUDA_VISIBLE_DEVICES")!r} args {vars(args)}')
    tiny = None
    if args.tiny:
        train, val = read_examples(args)
        tiny = M.tiny_models(train + val, args.seed)
    if args.part in ('grad', 'both'):
        grad(args, log, tiny)
    if args.part in ('ceiling', 'both'):
        ceiling(args, log, tiny)
    log('diag_l1b_channel done')


if __name__ == '__main__':
    main()
