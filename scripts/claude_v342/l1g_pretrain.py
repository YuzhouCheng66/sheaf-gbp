"""l1g_pretrain.py (Claude, 2026-11-01): stage P of batch L1g (pre-registration D8-90 section 3) -- self-supervised pretraining of the
cross-vocabulary "dictionary" = SenderAdapter3 + Decoder of both directions, on the paragraphs of the TRAIN split (no labels used).
One sample: a random paragraph of a random example; the sender j reads "[BOS?] <paragraph text, cut to --prefix-max-tokens> \\nAnswer: <span>"
(teacher-forced, the B0 LoRA enabled, no gradient; the span is the paragraph's title with probability 1/2, otherwise a random window of 1..6
words of one of its sentences), its adapter gives z_j; the receiver i's decoder maps x_i = z_j to logits; the target is the span in the
RECEIVER's tokenizer (' ' + span, cut to K, filler-padded); CE over the K positions; both directions every step; Adam on the adapters and
decoders only. Held-out spans (from the VAL split's paragraphs, fixed seed) give token accuracy (slot 0, message slots) and decoded EM.
Writes <out>/trainable.pt (adapters + decoders; the stage-F init), <out>/summary.json, <out>/log.txt. INCOMPLETE.json (exit 3) when the
deadline falls into training before --min-steps; a deadline after that ends training early and still writes the checkpoint (complete=false).
  python -B scripts/claude_v342/l1g_pretrain.py --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --train T.jsonl --val V.jsonl --out DIR
"""
from __future__ import annotations
import argparse
import math
import os
import platform
import random
import sys
import time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import l1_model as M  # noqa: E402
import l1_train as T1  # noqa: E402
import l1g_model as MG  # noqa: E402

STEPS = 15000
MAX_SPAN_WORDS = 6


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (trainable.pt = the frozen LoRA of both models); required unless --tiny')
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--out')
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--min-steps', type=int, default=1000, help='a deadline before this many steps -> INCOMPLETE; after -> the checkpoint is kept (complete=false)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--k', type=int, default=MG.K_HINT); ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--n-blocks', type=int, default=2)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--prefix-max-tokens', type=int, default=256)
    ap.add_argument('--max-span-words', type=int, default=MAX_SPAN_WORDS)
    ap.add_argument('--title-prob', type=float, default=0.5)
    ap.add_argument('--holdout', type=int, default=2000)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--eval-every', type=int, default=0, help='also evaluate 256 held-out spans every N steps (0 = only at the end)')
    ap.add_argument('--limit-train', type=int, default=0)
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--allow-fallback-data', action='store_true')
    return ap


# ---------------------------------------------------------------- spans
def sample_span(rng, ex, title_prob, max_words):
    """-> (paragraph text, span text, kind) from one example's paragraphs"""
    paras = ex['all_paragraphs']
    title, sents = paras[rng.randrange(len(paras))]
    text = ' '.join(sents)
    if rng.random() < title_prob or not any(s.split() for s in sents):
        return text, str(title), 'title'
    sent = sents[rng.randrange(len(sents))]
    words = sent.split()
    if not words:
        return text, str(title), 'title'
    n = rng.randint(1, min(max_words, len(words)))
    a = rng.randrange(0, len(words) - n + 1)
    return text, ' '.join(words[a:a + n]), 'window'


def make_spans(examples, n, seed, title_prob, max_words, limit_examples=0):
    rng = random.Random(seed)
    exs = examples[:limit_examples] if limit_examples else examples
    out = []
    for _ in range(n):
        ex = exs[rng.randrange(len(exs))]
        out.append(sample_span(rng, ex, title_prob, max_words))
    return out


def sender_pieces(ag, prefix_text, span_text, prefix_max):
    """[BOS?] prefix (cut) + '\\nAnswer:' as pre, ' ' + span as the teacher-forced tail"""
    pre = ag.ids('Documents:\n' + prefix_text)[:prefix_max] + ag.ids(MG.SPAN_PREFIX_TAIL)
    nb = [ag.bos] if ag.bos is not None else []
    return (nb + pre, [int(t) for t in ag.ids(' ' + span_text)], [])


def targets(ag, spans, k):
    ids, lens = [], []
    for _, span, _ in spans:
        t = [int(x) for x in ag.ids(' ' + span)[:k]]
        lens.append(len(t))
        ids.append(t + [ag.filler] * (k - len(t)))
    return torch.tensor(ids, dtype=torch.long, device=ag.device), lens


def step_loss(pair, spans, args, train=True):
    """both directions: sender j encodes its own reading of the span, receiver i decodes -> (loss, info)"""
    losses, accs, acc0 = [], [], []
    for j in range(M.N_AGENTS):
        i = 1 - j
        sj, ri = pair.agent(j), pair.agent(i)
        pcs = [sender_pieces(sj, text, span, args.prefix_max_tokens) for text, span, _ in spans]
        z, _ = sj.encode(pcs)
        lg = ri.reader.logits(z)
        tgt, lens = targets(ri, spans, args.k)
        ce = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
        pred = lg.detach().argmax(-1)
        hit = (pred == tgt)
        msk = torch.zeros_like(hit)
        for b, n in enumerate(lens):
            msk[b, :n] = True
        losses.append(ce)
        accs.append(float(hit[msk].float().mean()) if bool(msk.any()) else float('nan'))
        acc0.append(float(hit[:, 0].float().mean()))
    loss = torch.stack(losses).mean()
    return loss, dict(loss_agent=[float(l.detach()) for l in losses], msg_acc=float(np.nanmean(accs)), msg_acc_dir=accs, acc0_dir=acc0)


@torch.no_grad()
def evaluate(pair, spans, args, deadline=None, n_examples=8):
    """held-out spans -> per direction: slot-0 accuracy, message-slot accuracy, decoded EM against the span; a few decoded examples"""
    pair.eval()
    res = {}
    for j in range(M.N_AGENTS):
        i = 1 - j
        sj, ri = pair.agent(j), pair.agent(i)
        hit0, hitm, em, exs, kinds = [], [], [], [], {}
        for batch in M.chunks(spans, args.eval_batch):
            M.check_deadline(deadline, 'pretrain eval')
            pcs = [sender_pieces(sj, text, span, args.prefix_max_tokens) for text, span, _ in batch]
            z, _ = sj.encode(pcs)
            pred = ri.reader.logits(z).argmax(-1)
            tgt, lens = targets(ri, batch, args.k)
            for b, (_, span, kind) in enumerate(batch):
                n = max(1, lens[b])
                hit0.append(float(pred[b, 0] == tgt[b, 0]))
                hitm.append(float((pred[b, :n] == tgt[b, :n]).float().mean()))
                pid = pred[b].tolist()
                cut = len(pid)
                for q in range(len(pid) - 1, -1, -1):
                    if pid[q] != ri.filler:
                        break
                    cut = q
                text = ri.tok.decode(pid[:cut], skip_special_tokens=True).strip()
                e = float(M.DATA.exact_match(text, span))
                em.append(e)
                kinds.setdefault(kind, []).append(e)
                if len(exs) < n_examples:
                    exs.append(dict(span=span, decoded=text, kind=kind))
        res[f'{sj.name} -> {ri.name}'] = dict(n=len(spans), slot0_acc=float(np.mean(hit0)), msg_slot_acc=float(np.mean(hitm)), decoded_em=float(np.mean(em)),
                                               decoded_em_by_kind={k: float(np.mean(v)) for k, v in kinds.items()}, examples=exs)
    pair.train()
    return res


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('seed', 'train', 'val', 'out'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from is required outside --tiny')
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l1g_pretrain start pid {os.getpid()} host {platform.node()} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    train = M.DATA.read_jsonl(args.train)
    val = M.DATA.read_jsonl(args.val)
    if args.limit_train:
        train = train[:args.limit_train]
    log(f'device {device} torch {torch.__version__}; train examples {len(train)} ({sum(len(ex["all_paragraphs"]) for ex in train)} paragraphs); '
        f'steps {args.steps} x batch {args.batch} = {args.steps * args.batch} spans; prefix cap {args.prefix_max_tokens}; k {args.k}; title prob {args.title_prob}')
    tiny = M.tiny_models(train + val, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = MG.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        if receiver.get('arm') not in (None, 'B0') or receiver.get('seed') not in (None, args.seed):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        log(f'receiver / sender LoRA: {receiver}')
    pair = MG.build_pair(MG.PRETRAIN_ARM, 'hetero', device, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=False, max_ctx=None, receiver_state=receiver_state,
                         tau=args.tau, n_blocks=args.n_blocks, **agent_kw)
    counts = M.count_params(pair)
    info = T1.agents_info(pair)
    log(f'model: {info}; trainable {counts["trainable"]:,} ({100 * counts["trainable_frac"]:.3f} %)')
    holdout = make_spans(val, args.holdout, args.seed + 7, args.title_prob, args.max_span_words)
    log(f'held-out spans {len(holdout)} (title {sum(k == "title" for _, _, k in holdout)}), e.g. {holdout[:3]}')
    rng = random.Random(args.seed)
    opt, sched, _ = M.build_optimizer(pair, 0.0, args.lr_adapter, args.steps, args.warmup, args.weight_decay)
    params = [p for g in opt.param_groups for p in g['params']]
    pair.train()
    use_ac = device.type == 'cuda'
    times, losses, accs, hist = [], [], [], []
    steps_done, where = 0, None
    t_train = time.time()
    try:
        for s in range(args.steps):
            M.check_deadline(deadline, f'pretrain step {s}')
            spans = make_spans(train, args.batch, rng.randrange(1 << 30), args.title_prob, args.max_span_words)
            M.sync(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
                loss, inf = step_loss(pair, spans, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {s}')
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            M.sync(device)
            times.append(time.perf_counter() - t0); losses.append(float(loss.detach())); accs.append(inf['msg_acc'])
            steps_done = s + 1
            if steps_done % args.log_every == 0 or steps_done == args.steps:
                post = times[M.TIMING_WARMUP_STEPS:]
                rate = f'{np.mean(post):.3f} s/step' if post else 'warm-up'
                mem = f', peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GB' if use_ac else ''
                log(f'pretrain step {steps_done}/{args.steps} loss {np.mean(losses[-args.log_every:]):.4f} dirs {["%.3f" % x for x in inf["loss_agent"]]} '
                    f'msg acc {np.nanmean(accs[-args.log_every:]):.3f} slot0 {["%.2f" % x for x in inf["acc0_dir"]]} lr {sched.get_last_lr()[0]:.2e} gnorm {float(gn):.2f} {rate}{mem}')
                hist.append(dict(step=steps_done, loss=float(np.mean(losses[-args.log_every:])), msg_acc=float(np.nanmean(accs[-args.log_every:]))))
            if args.eval_every and steps_done % args.eval_every == 0 and steps_done < args.steps:
                r = evaluate(pair, holdout[:256], args, deadline)
                log(f'held-out (256) at step {steps_done}: ' + ' | '.join(f'{k}: slot0 {v["slot0_acc"]:.3f} msg {v["msg_slot_acc"]:.3f} em {v["decoded_em"]:.3f}' for k, v in r.items()))
    except M.Deadline as e:
        where = str(e)
        log(f'deadline during training at step {steps_done}: {e}')
        if steps_done < args.min_steps:
            T1.incomplete(args.out, where, log)
            return 3
    train_s = time.time() - t_train
    post = times[M.TIMING_WARMUP_STEPS:]
    state = MG.channel_state(pair)
    ckp = os.path.join(args.out, 'trainable.pt')
    pre_summary = dict(steps_done=steps_done, steps_planned=args.steps, loss_first100=float(np.mean(losses[:100])) if losses else None,
                       loss_last100=float(np.mean(losses[-100:])) if losses else None, msg_acc_last100=float(np.nanmean(accs[-100:])) if accs else None)
    torch.save(dict(batch=MG.BATCH, prereg=MG.PREREG, stage='P', seed=args.seed, models=[a['name'] for a in info], state=state, agents=info, receiver=receiver,
                    summary=pre_summary, config=vars(args)), ckp)
    log(f'checkpoint saved ({len(state)} tensors) before the held-out evaluation')
    t_eval = time.time()
    try:
        held = evaluate(pair, holdout, args, deadline)
        eval_where = None
    except M.Deadline as e:
        held, eval_where = {}, str(e)
        log(f'held-out evaluation cut by the deadline: {e}')
    for k, v in held.items():
        log(f'held-out {k}: slot0 {v["slot0_acc"]:.3f} msg slots {v["msg_slot_acc"]:.3f} decoded EM {v["decoded_em"]:.3f} by kind {v["decoded_em_by_kind"]}; e.g. {[(e["span"], e["decoded"]) for e in v["examples"][:3]]}')
    summary = dict(batch=MG.BATCH, prereg=MG.PREREG, stage='P', seed=args.seed, pair='hetero', models=list(dict.fromkeys(a['name'] for a in info)), agents=info,
                   complete=bool(steps_done == args.steps), steps=steps_done, steps_planned=args.steps, deadline_in_training=where, eval_incomplete=eval_where,
                   train=dict(**pre_summary, seconds=train_s, step_time_post_warmup_s=float(np.mean(post)) if post else None, steps_timed=len(post), history=hist),
                   heldout=held, n_heldout=len(holdout), receiver=receiver, params=counts,
                   timings=dict(train_s=train_s, eval_s=time.time() - t_eval, total_s=time.time() - t_start),
                   config=dict(vars(args), device=str(device), torch=torch.__version__), scripts_sha=M.scripts_sha(MG.SCRIPTS), trainable_sha256=M.file_sha256(ckp),
                   utc_end=M.utc(), pid=os.getpid())
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    M.jdump(summary, sp)
    log(f'l1g_pretrain done: {steps_done}/{args.steps} steps; held-out ' + ' | '.join(f'{k}: slot0 {v["slot0_acc"]:.3f} em {v["decoded_em"]:.3f}' for k, v in held.items()) +
        f'; total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
