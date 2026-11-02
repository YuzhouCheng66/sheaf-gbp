"""l2_train.py (Claude, 2026-11-01): trainer of the L2 ladder (pre-registration D8-92) on top of l2_model.py / l1h_model.py.
Arms:  R1 = stage 2 only: the receiver LoRA 'recv' (initialised from B0) learns to read the partner's HARD-rendered hint of a FROZEN sender /
            decoder loaded with --init-from (an L1h Hu run, or an L2 Hu run on the hetero pair);
       V1 = evidence content (same2 only): stage 1 trains the aligned sender / decoder to COPY the tokens at the sender's oracle span (the gold
            supporting sentences of its own gold paragraph, K = 32), stage 2 = R1 on those hints; the criterion is evaluated with the
            attention-selected spans (--sel-layer, chosen in E0c), the oracle spans are the ceiling readout;
       Hu = stage 1 only (the L1h sender on the pair given; used for the hetero pair), no receiver training.
The B0 checkpoint of the seed (--receiver-from) is the frozen 'default' LoRA of every agent (sender role + the `none` reference baseline).
Criterion = the pinned eval_latent with the HARD rendering on VAL (recv adapter when trained) vs the baseline none (B0 adapter).
Readouts: fresh, none_recv (the trained receiver on the plain prompt), gold_hint, hint metrics (answer mode), slot accuracy, soft rendering,
absence p = 1, fresh none; V1 also the oracle-span criterion block (val_oracle). Writes <out>/summary.json (l1_aggregate schema + L2 blocks),
trainable.pt (recv LoRA + sender + decoder) before the evaluation, INCOMPLETE.json (exit 3) on the deadline.
  python -B scripts/claude_v342/l2_train.py --arm R1 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from runs/l1h/Hu-same2-s342101 ...
"""
from __future__ import annotations
import argparse
import os
import platform
import sys
import time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402
import l1_train as T1  # noqa: E402
import l1c_model as MC  # noqa: E402
import l2_model as L2  # noqa: E402
from l1d_model import lam_by_bucket as MD_lam_by_bucket  # noqa: E402

SEEDS = T1.SEEDS
STAGE1_STEPS = {'R1': 0, 'V1': 1000, 'Hu': 2000, 'V1s': 1000, 'V2': 1000}        # D8-92 section 3 (+ appendix 3: V1s; D8-93: V2)
STAGE2_STEPS = {'R1': 1500, 'V1': 1000, 'Hu': 0, 'V1s': 1000, 'V2': 1000}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=L2.ARMS)
    ap.add_argument('--pair', choices=L2.PAIRS)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (trainable.pt = the frozen B0 LoRA); required unless --tiny')
    ap.add_argument('--init-from', default=None, help='R1: the run dir whose trainable.pt holds the frozen sender / decoder (L1h Hu or L2 Hu)')
    ap.add_argument('--stage1-steps', type=int, default=None)
    ap.add_argument('--stage2-steps', type=int, default=None)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--k', type=int, default=None, help='slots: default 8 (answer arms) / 32 (V1)')
    ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--n-blocks', type=int, default=MC.N_BLOCKS)
    ap.add_argument('--tau', type=float, default=MC.TAU)
    ap.add_argument('--sel-layer', type=int, default=None, help='V1: the attention layer of the span selection (E0c); required unless --tiny')
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='random tiny LlamaConfig models + WordLevel tokenizers (tests only)')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--no-grad-ckpt', action='store_true', help='stage 2 runs with gradient checkpointing unless given')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--debug-deadline-at', default=None, choices=('readouts',), help='tests only: deadline in the past right before the readouts')
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def _owner(blk):
    return [(k, round(x['em'], 2) if x['em'] is not None else None) for k, x in blk['em_by_owner'].items()]


@torch.no_grad()
def slot_accuracy(pair, examples, eval_batch=16, deadline=None):
    """per slot: the receiver's decoded token == the target token (gold answer / the sender's span tokens); answer mode also vs the sender's
    own answer tokens (same tokenizer only)"""
    pair.eval()
    K = pair.agent(0).k
    hit_g = [[0, 0] for _ in range(K)]
    hit_s = [[0, 0] for _ in range(K)]
    with_sender = pair.mode == 'answer' and pair.pair == 'same2'
    saved = dict(pair.trunc)
    for batch in M.chunks(examples, eval_batch):
        M.check_deadline(deadline, 'slot_accuracy')
        z, lam = pair.encode_both(batch)
        xs, _, _ = pair.messages_eval(z, lam)
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            pred = ag.reader.logits(xs[i]).argmax(-1).cpu()
            tgt, lens = pair.targets(batch, i)
            tgt = tgt.cpu()
            answers = pair.sender_pieces(batch, 1 - i)[1] if with_sender else None
            for b in range(len(batch)):
                for k in range(K):
                    if k < lens[b]:
                        hit_g[k][1] += 1
                        hit_g[k][0] += int(pred[b, k] == tgt[b, k])
                if answers is not None:
                    s_ids = [int(t) for t in answers[b][1][:K]]
                    for k in range(len(s_ids)):
                        hit_s[k][1] += 1
                        hit_s[k][0] += int(int(pred[b, k]) == s_ids[k])
    pair.trunc.clear()
    pair.trunc.update(saved)
    acc_g = [(h / n if n else None) for h, n in hit_g]
    acc_s = [(h / n if n else None) for h, n in hit_s] if with_sender else None
    return dict(n=len(examples), mode=pair.mode, acc_gold_by_slot=acc_g, acc_sender_by_slot=acc_s, n_by_slot=[n for _, n in hit_g],
                acc_gold_mean=float(np.mean([a for a in acc_g if a is not None])) if any(a is not None for a in acc_g) else None)


def readouts(pair, val, fresh, args, log, deadline):
    """soft-deadline readouts -> (dict, where it stopped or None)"""
    out, where = dict(hint={}, baseline={'none': {}, 'none_recv': {}, 'gold_hint': {}}, soft={}, absence={}, slot_acc=None, val_oracle=None, sel_quality=None), None
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, keep_preds=False, log=log)
    recv = pair.agent(0).recv
    try:
        if pair.mode in ('span_attn', 'span_sel'):
            crit = pair.mode
            where = 'oracle-span criterion block (ceiling)'
            pair.set_mode('span_oracle')
            r = L2.eval_render(pair, val, hard=True, drop_p=0.0, c_check=False, seed=args.seed, deadline=deadline, **ekw)
            pair.set_mode(crit)
            out['val_oracle'] = r
            log(f'VAL oracle spans (hard, ceiling): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        if pair.mode == 'span_sel':
            where = 'selector quality val'
            r = L2.sel_quality(pair, val, eval_batch=args.eval_batch, deadline=deadline)
            out['sel_quality'] = r
            log(f'VAL selector quality: overlap50 {r["overlap50"]:.3f} over {r["n"]} pairs')
        where = 'slot accuracy val'
        r = slot_accuracy(pair, val, eval_batch=args.eval_batch, deadline=deadline)
        out['slot_acc'] = r
        log(f'VAL slot accuracy ({pair.mode}) vs target {[None if a is None else round(a, 3) for a in r["acc_gold_by_slot"]]} '
            f'vs sender answer {None if r["acc_sender_by_slot"] is None else [None if a is None else round(a, 3) for a in r["acc_sender_by_slot"]]} n {r["n_by_slot"]}')
        if pair.mode == 'answer':
            for split, exs in (('val', val), ('fresh', fresh)):
                where = f'hint {split}'
                r = L2.hint_metrics(pair, exs, eval_batch=args.eval_batch, deadline=deadline, log=log)
                out['hint'][split] = r
                log(f'{split} hints by receiver bucket: {r["hint_by_receiver_bucket"]}; e.g. {[(e["rel"], e["hint"], e["sender_answer"], e["gold"]) for e in r["examples"][:4]]}')
        if recv:
            where = 'baseline none_recv val'
            r = L2.eval_baseline(pair, val, 'none_recv', deadline=deadline, **ekw)
            out['baseline']['none_recv']['val'] = r
            log(f'VAL baseline none_recv (trained receiver, plain prompt): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = 'baseline gold_hint val'
        r = L2.eval_baseline(pair, val, 'gold_hint', deadline=deadline, **ekw)
        out['baseline']['gold_hint']['val'] = r
        log(f'VAL baseline gold_hint ({"recv" if recv else "default"}): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = 'soft val'
        r = L2.eval_render(pair, val, hard=False, drop_p=0.0, c_check=False, seed=args.seed, deadline=deadline, **ekw)
        out['soft']['val'] = r
        log(f'VAL soft rendering: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = 'absence p=1'
        r = L2.eval_render(pair, val, hard=True, drop_p=1.0, c_check=False, seed=args.seed, deadline=deadline, **ekw)
        out['absence']['1.0'] = r
        log(f'VAL p=1.0 (hard): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)} dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        where = 'baseline none fresh'
        r = L2.eval_baseline(pair, fresh, 'none', deadline=deadline, **ekw)
        out['baseline']['none']['fresh'] = r
        log(f'fresh baseline none: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = None
    except M.Deadline as e:
        log(f'readouts stopped at {where}: {e}')
        where = f'{where}: {e}'
    return out, where


def _state(pair):
    """the checkpoint state: the recv LoRA (when enabled) + the sender adapter + decoder of every agent"""
    st = {}
    for n, p in pair.named_parameters():
        if ('.adapter.' in n or '.reader.' in n or '.selector.' in n) or ('lora_' in n and '.recv.' in n) or n == 'w_raw':
            st[n] = p.detach().cpu()
    return st


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('arm', 'pair', 'seed', 'train', 'val', 'fresh', 'out'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from (the B0 run dir of this seed) is required outside --tiny')
    if args.arm == 'R1' and not args.init_from:
        raise SystemExit('--init-from (the frozen sender / decoder run dir) is required for R1')
    if args.arm in L2.SPAN_ARMS and args.pair != 'same2':
        raise SystemExit('V1 / V1s (evidence spans) run on pair same2 only')
    if args.arm in L2.SPAN_ARMS and args.arm not in L2.SELECTOR_ARMS and args.sel_layer is None:
        if not args.tiny:
            raise SystemExit('--sel-layer (E0c) is required for V1 / V1s')
        args.sel_layer = 1
    s1 = STAGE1_STEPS[args.arm] if args.stage1_steps is None else args.stage1_steps
    s2 = STAGE2_STEPS[args.arm] if args.stage2_steps is None else args.stage2_steps
    if args.arm == 'R1' and s1:
        raise SystemExit('R1 has no stage 1 (the sender is frozen)')
    if args.arm == 'Hu' and s2:
        raise SystemExit('Hu has no stage 2')
    mode = 'span_oracle' if args.arm in L2.SPAN_ARMS else 'answer'
    recv = args.arm in L2.RECV_ARMS
    if args.k is None:
        args.k = L2.K_SPAN if args.arm in L2.SPAN_ARMS else L2.K_HINT
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l2_train start pid {os.getpid()} host {platform.node()} arm {args.arm} (base {L2.BASE_ARM[args.arm]}) pair {args.pair} seed {args.seed} out {args.out}; '
        f'mode {mode} k {args.k} recv {recv} stage1 {s1} stage2 {s2} sel_layer {args.sel_layer}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it (--tiny / --allow-fallback-data for smoke only)')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    log(f'device {device} torch {torch.__version__} data api {"FALLBACK (l1_data absent)" if M.DATA_IS_FALLBACK else "l1_data"}'
        f'{"" if args.seed in SEEDS else " (NOTE: seed not pre-registered)"}; tau {args.tau} blocks {args.n_blocks} d_e {args.d_e} layer_frac {args.layer_frac}')
    train, val, fresh = T1.read_data(args, log)
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = L2.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        b0 = os.path.join(args.receiver_from, 'summary.json')
        if os.path.isfile(b0):
            s0 = M.jload(b0)
            receiver.update(b0_arm=s0.get('arm'), b0_seed=s0.get('seed'), b0_val_em=(s0.get('val') or {}).get('em'), b0_fresh_em=(s0.get('fresh') or {}).get('em'))
        if receiver.get('arm') not in (None, 'B0') or (receiver.get('seed') not in (None, args.seed)):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        for k in ('lora_r', 'lora_alpha'):
            if receiver.get(k) is not None and receiver[k] != getattr(args, k):
                raise SystemExit(f'--receiver-from {k} = {receiver[k]} differs from --{k.replace("_", "-")} {getattr(args, k)} (a silent LoRA scale mismatch)')
        log(f'receiver (= the sender model, B0): {receiver}')
    init_state, init_meta = None, None
    if args.init_from:
        init_state, init_meta = L2.load_init(os.path.join(args.init_from, 'trainable.pt'))
        if init_meta.get('seed') not in (None, args.seed):
            raise SystemExit(f'--init-from seed {init_meta.get("seed")} != {args.seed}')
        if init_meta.get('pair') not in (None, args.pair):
            raise SystemExit(f'--init-from pair {init_meta.get("pair")} != {args.pair}')
        log(f'init (frozen sender / decoder): {init_meta}')
    ckp = os.path.join(args.out, 'trainable.pt')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair = L2.build_pair(args.arm, args.pair, device, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=(recv and not args.no_grad_ckpt), max_ctx=args.max_ctx_tokens,
                             receiver_state=receiver_state, tau=args.tau, n_blocks=args.n_blocks, mode=mode, span_k=args.k, init_state=init_state, recv=recv,
                             sel_layer=args.sel_layer, **agent_kw)
        load_s = time.time() - t0
        info = T1.agents_info(pair)
        if receiver is not None and receiver.get('models') is not None:
            names = [a['name'] for a in info]
            ok = (list(receiver['models'])[0] == names[0] and names[0] == names[1]) if args.pair == 'same2' else (list(receiver['models']) == names)
            if not ok:
                raise SystemExit(f'--receiver-from models {receiver["models"]} do not provide the LoRA of this pair\'s models {names}')
        if init_meta is not None and init_meta.get('models') is not None and list(init_meta['models']) != [a['name'] for a in info]:
            raise SystemExit(f'--init-from models {init_meta["models"]} != this pair {[a["name"] for a in info]}')
        fillers = {a.name: dict(id=a.filler, is_pad=a.filler_is_pad, vocab=a.reader.vocab, hint_pre=len(a.hint_pre), hint_post=len(a.hint_post)) for a in pair.agents}
        template = (L2.SPAN_PRE, L2.SPAN_POST) if mode != 'answer' else (L2.HINT_PRE, L2.HINT_POST)
        log(f'model: {info}; template {template!r} + {args.k} slots; fillers {fillers}')
        stages = {}
        # ---------------------------------------------------------------- stage 1: the sender / decoder (V1: copy CE; Hu: answer CE)
        if s1:
            pair.use('default')
            L2.freeze_sender(pair, frozen=False)
            counts = M.count_params(pair)
            log(f'stage 1: trainable {counts["trainable"]:,} (sender adapter + decoder' + (' + lam head + w' if pair.w is not None else '') + ')')
            tr = M.train_steps(pair, lambda b: pair.step(b, train=True), M.batch_stream(train, args.batch, args.seed), s1, 0.0, args.lr_adapter,
                               args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/{args.pair}/s{args.seed} stage1',
                               weight_decay=args.weight_decay, extra_log=lambda i: f'msg acc {i["msg_acc"]:.3f}' + (f' w {pair.w_value:.4f}' if pair.w is not None else '') + (f' sel {i["sel_loss"]:.3f}' if 'sel_loss' in i else ''))
            tr['message'] = pair.msg_summary()
            tr['span'] = dict(seen=pair.span_seen, missing=pair.span_missing, skipped=pair.n_skipped)
            tr['answers_generated'] = {a.name: a.n_generated for a in pair.agents}
            tr['params'] = counts
            tr['selector_loss_first100'] = float(np.mean(pair.sel_hist[:100])) if pair.sel_hist else None
            tr['selector_loss_last100'] = float(np.mean(pair.sel_hist[-100:])) if pair.sel_hist else None
            stages['stage1'] = tr
            log(f'stage 1 done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
                f'msg acc first/last {tr["message"].get("msg_acc_first")}/{tr["message"].get("msg_acc_last")}; spans seen {pair.span_seen} missing {pair.span_missing}')
            pair.msg_hist = []
        # ---------------------------------------------------------------- stage 2: the receiver LoRA on the hard-rendered hints
        if s2:
            L2.freeze_sender(pair, frozen=True)
            pair.use('recv')
            counts = M.count_params(pair)
            log(f'stage 2: trainable {counts["trainable"]:,} (recv LoRA of both agents; sender / decoder frozen)')
            tr = M.train_steps(pair, lambda b: pair.step_recv(b, train=True), M.batch_stream(train, args.batch, args.seed + 1), s2, args.lr_lora, 0.0,
                               args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/{args.pair}/s{args.seed} stage2',
                               weight_decay=args.weight_decay)
            tr['span'] = dict(seen=pair.span_seen, missing=pair.span_missing)
            tr['answers_generated'] = {a.name: a.n_generated for a in pair.agents}
            tr['params'] = counts
            stages['stage2'] = tr
            log(f'stage 2 done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
                f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed)')
        L2.freeze_sender(pair, frozen=True)
        if recv:
            pair.use('recv')
        state = _state(pair)
        torch.save(dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info, receiver=receiver,
                        init_from=init_meta, recv=recv, mode=mode, config=vars(args)), ckp)
        log(f'trainable.pt saved before the evaluation ({len(state)} tensors)')
        # ---------------------------------------------------------------- the criterion and the readouts
        t_eval = time.time()
        ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=True, log=log)
        crit_mode = ('span_sel' if args.arm in L2.SELECTOR_ARMS else 'span_attn') if args.arm in L2.SPAN_ARMS else 'answer'
        pair.set_mode(crit_mode)
        fusion = pair.arm in M.FUSION_ARMS
        v = L2.eval_render(pair, val, hard=True, drop_p=0.0, c_check=fusion, seed=args.seed, **ekw)
        log(f'VAL p=0.0 (hard, criterion, {crit_mode}, {"recv" if recv else "default"}): EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {_owner(v)}')
        c_assert = None
        if fusion:
            c_assert = bool(v['c_assert'])
            log(f'C assertion: {c_assert} {v["c_assert_detail"]}')
        lam_bucket = MD_lam_by_bucket(v['records']) if fusion else None
        if lam_bucket:
            log(f'lam by bucket (VAL): {lam_bucket}')
        f = L2.eval_render(pair, fresh, hard=True, drop_p=0.0, c_check=False, seed=args.seed, **ekw)
        log(f'fresh (hard): EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        n0 = L2.eval_baseline(pair, val, 'none', **ekw)
        log(f'VAL baseline none (the plain B0 prompt, B0 adapter): EM {n0["em"]:.2f} F1 {n0["f1"]:.2f} owner {_owner(n0)}')
        if args.keep_preds:
            M.jdump(v.pop('records'), os.path.join(args.out, 'preds_val.json'))
            M.jdump(f.pop('records'), os.path.join(args.out, 'preds_fresh.json'))
            M.jdump(n0.pop('records'), os.path.join(args.out, 'preds_val_none.json'))
        else:
            for blk in (v, f, n0):
                blk.pop('records', None)
        if args.debug_deadline_at == 'readouts':
            deadline = time.time() - 1.0
            log('debug: deadline moved into the past before the readouts')
        ro, where = readouts(pair, val, fresh, args, log, deadline)
        ro['baseline']['none']['val'] = n0
        absence = {str(0.0): v, **ro['absence']}
        eval_s = time.time() - t_eval
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for a in info))
    done = all(stages[k]['steps_done'] == n for k, n in (('stage1', s1), ('stage2', s2)) if n)
    summary = dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=L2.BATCH, prereg=L2.PREREG, val=v, fresh=f,
                   val_oracle=ro['val_oracle'], sel_quality=ro['sel_quality'], absence=absence, soft=ro['soft'], baseline=ro['baseline'], hint=ro['hint'], slot_acc=ro['slot_acc'], complete=bool(done),
                   steps=sum(st['steps_done'] for st in stages.values()), steps_planned=s1 + s2, stages=stages, readouts_incomplete=where, agents=info,
                   params=M.count_params(pair), receiver=receiver, init_from=init_meta, lam_by_bucket=lam_bucket,
                   span=dict(seen=pair.span_seen, missing=pair.span_missing, attn_passes=[a.n_attn_passes for a in pair.agents], sel_layer=args.sel_layer),
                   channel=dict(tau=args.tau, n_blocks=args.n_blocks, k=args.k, d_e=args.d_e, filler=fillers, msg_tokens=args.k, rendering='hard', mode=mode, criterion_mode=crit_mode,
                                hint_pre=template[0], hint_post=template[1], recv=recv, stage1_steps=s1, stage2_steps=s2,
                                loss=('position CE vs the gold answer' if mode == 'answer' else 'position CE vs the tokens at the oracle span') + (' + receiver answer CE on hard hints' if s2 else ''),
                                sender_pass1=('B0 prompt + own greedy answer, LoRA enabled' if mode == 'answer' else 'B0 prompt (documents + question), LoRA enabled'),
                                adapter='AlignedSender (slot k = LN(W h[position k]), no attention)',
                                receiver_training=('recv LoRA initialised from B0, trained on the hard-rendered hints' if recv else 'none (frozen B0 checkpoint)'),
                                fusion=('sheaf-GBP consensus (S1: lam head + w learned in stage 1)' if fusion else 'one-way projection (U)'),
                                pair_note=('same2: two independent Qwen3-1.7B instances, the hetero B0 Qwen LoRA on both' if args.pair == 'same2' else 'hetero: Qwen3-1.7B + Llama-3.2-1B-Instruct, the hetero B0 LoRA')),
                   timings=dict(load_s=load_s, train_s=sum(st['seconds'] for st in stages.values()), eval_s=eval_s, total_s=time.time() - t_start,
                                timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS), u_drop_p=M.U_DROP_P, edges=M.EDGES,
                               wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__, data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data'),
                   scripts_sha=M.scripts_sha(L2.SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if fusion:
        summary['c_assert'] = c_assert
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l2_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} (none {n0["em"]:.2f}) fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'readouts {"complete" if not where else "INCOMPLETE (" + where + ")"} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
