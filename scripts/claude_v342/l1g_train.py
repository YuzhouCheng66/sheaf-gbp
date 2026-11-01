"""l1g_train.py (Claude, 2026-11-01): batch L1g stage-F trainer for the arms Gu / Gs (pre-registration D8-90 section 4 = L1e initialised from the dictionary pretraining).
Mirrors l1e_train.py with l1g_model.build_pair and --init-from (the stage-P checkpoint: adapters + decoders): the receiver AND the sender's model are the frozen B0 checkpoint of the same seed
(--receiver-from runs/l1/B0-hetero-s<seed>); the adapter + decoder are trained with the message CE against the gold answer (K = 8 slots).
Criterion = the pinned eval_latent with the HARD rendering (argmax tokens inside the hint template) on VAL (C assertion for Gs) and fresh, plus
the VAL baseline `none` (the plain B0 prompt) computed before the readouts. Readouts under the soft deadline: hint metrics (VAL, fresh), the
baseline gold_hint (VAL), the soft rendering (VAL), absence p = 1 (hard, VAL), the baseline none (fresh), lam by bucket (Gs, from the VAL records).
Writes <out>/summary.json in the l1_aggregate schema plus the L1d / L1e / L1g blocks; trainable.pt before the evaluation; INCOMPLETE.json (exit 3) when the
deadline falls into training or the criterion evaluation.
  python -B scripts/claude_v342/l1g_train.py --arm Gu --pair hetero --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from runs/l1g/pretrain-s342101 --train T.jsonl --val V.jsonl --fresh F.jsonl --out DIR
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
import torch  # noqa: E402
import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402
import l1_train as T1  # noqa: E402
import l1c_model as MC  # noqa: E402
import l1g_model as MD  # noqa: E402

STEPS = 2000                 # D8-90 section 4 (= D8-88)
SEEDS = T1.SEEDS


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=MD.ARMS)
    ap.add_argument('--pair', choices=tuple(M.MODELS))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (trainable.pt = the frozen receiver / sender LoRA); required unless --tiny')
    ap.add_argument('--init-from', default=None, help='the stage-P run dir (trainable.pt = pretrained adapters + decoders); required unless --tiny')
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--k', type=int, default=MD.K_HINT); ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--n-blocks', type=int, default=MC.N_BLOCKS)
    ap.add_argument('--tau', type=float, default=MC.TAU)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--absence-ps', default=','.join(str(p) for p in MD.ABSENCE_PS))
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='random tiny LlamaConfig models + WordLevel tokenizers (tests only)')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--debug-deadline-at', default=None, choices=('readouts',), help='tests only: deadline in the past right before the readouts')
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def _owner(blk):
    return [(k, round(x['em'], 2) if x['em'] is not None else None) for k, x in blk['em_by_owner'].items()]


def readouts(pair, val, fresh, args, log, deadline):
    """soft-deadline readouts -> (dict(hint, baseline, soft, absence), where it stopped or None)"""
    out, where = dict(hint={}, baseline={'none': {}, 'gold_hint': {}}, soft={}, absence={}), None
    ps = [float(p) for p in args.absence_ps.split(',') if p.strip() and float(p) != 0.0]
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, keep_preds=False, log=log)
    try:
        for split, exs in (('val', val), ('fresh', fresh)):
            where = f'hint {split}'
            r = MD.hint_metrics(pair, exs, eval_batch=args.eval_batch, deadline=deadline, log=log)
            out['hint'][split] = r
            log(f'{split} hints by receiver bucket: {r["hint_by_receiver_bucket"]}; sender own answers by bucket: {r["sender_answer_by_own_bucket"]}; '
                f'e.g. {[(e["rel"], e["hint"], e["sender_answer"], e["gold"]) for e in r["examples"][:4]]}')
        where = 'baseline gold_hint val'
        r = MD.eval_baseline(pair, val, 'gold_hint', deadline=deadline, **ekw)
        out['baseline']['gold_hint']['val'] = r
        log(f'VAL baseline gold_hint: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = 'soft val'
        r = MD.eval_render(pair, val, hard=False, drop_p=0.0, c_check=False, seed=args.seed, deadline=deadline, **ekw)
        out['soft']['val'] = r
        log(f'VAL soft rendering: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        for p in ps:
            where = f'absence p={p}'
            r = MD.eval_render(pair, val, hard=True, drop_p=p, c_check=False, seed=args.seed, deadline=deadline, **ekw)
            out['absence'][str(p)] = r
            log(f'VAL p={p} (hard): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)} dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        where = 'baseline none fresh'
        r = MD.eval_baseline(pair, fresh, 'none', deadline=deadline, **ekw)
        out['baseline']['none']['fresh'] = r
        log(f'fresh baseline none: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        where = None
    except M.Deadline as e:
        log(f'readouts stopped at {where}: {e}')
        where = f'{where}: {e}'
    return out, where


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('arm', 'pair', 'seed', 'train', 'val', 'fresh', 'out'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from (the B0 run dir of this seed) is required outside --tiny')
    if not args.tiny and not args.init_from:
        raise SystemExit('--init-from (the stage-P run dir) is required outside --tiny')
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l1g_train start pid {os.getpid()} host {platform.node()} arm {args.arm} (base {MD.BASE_ARM[args.arm]}) pair {args.pair} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it (--tiny / --allow-fallback-data for smoke only)')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    log(f'device {device} torch {torch.__version__} data api {"FALLBACK (l1_data absent)" if M.DATA_IS_FALLBACK else "l1_data"}'
        f'{"" if args.seed in SEEDS else " (NOTE: seed not pre-registered)"}; k {args.k} tau {args.tau} blocks {args.n_blocks} absence ps {args.absence_ps}; '
        f'hint template {MD.HINT_PRE!r} + {args.k} slots + {MD.HINT_POST!r}')
    train, val, fresh = T1.read_data(args, log)
    log(f'steps {args.steps} x batch {args.batch} = {args.steps * args.batch} examples ({args.steps * args.batch / max(1, len(train)):.2f} epochs)')
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = MD.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        b0 = os.path.join(args.receiver_from, 'summary.json')
        if os.path.isfile(b0):
            s0 = M.jload(b0)
            receiver.update(b0_arm=s0.get('arm'), b0_seed=s0.get('seed'), b0_val_em=(s0.get('val') or {}).get('em'), b0_fresh_em=(s0.get('fresh') or {}).get('em'))
        if receiver.get('arm') not in (None, 'B0') or (receiver.get('seed') not in (None, args.seed)):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        for k in ('lora_r', 'lora_alpha'):
            if receiver.get(k) is not None and receiver[k] != getattr(args, k):
                raise SystemExit(f'--receiver-from {k} = {receiver[k]} differs from --{k.replace("_", "-")} {getattr(args, k)} (a silent LoRA scale mismatch)')
        log(f'receiver (= the sender model): {receiver}')
    ckp = os.path.join(args.out, 'trainable.pt')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair = MD.build_pair(args.arm, args.pair, device, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=False, max_ctx=args.max_ctx_tokens,
                             receiver_state=receiver_state, tau=args.tau, n_blocks=args.n_blocks, **agent_kw)
        load_s = time.time() - t0
        counts = M.count_params(pair)
        info = T1.agents_info(pair)
        if receiver is not None and receiver.get('models') is not None and list(receiver['models']) != [a['name'] for a in info]:
            raise SystemExit(f'--receiver-from models {receiver["models"]} differ from this pair {[a["name"] for a in info]}')
        init = None
        if args.init_from:
            init = MD.load_init(pair, os.path.join(args.init_from, 'trainable.pt'))
            log(f'init from {init["path"]} ({init["n_loaded"]} tensors, sha {init["sha256"][:12]}); fresh: {init["not_in_file"]}; pretrain summary {init["pretrain_summary"]}')
        fillers = {a.name: dict(id=a.filler, is_pad=a.filler_is_pad, vocab=a.reader.vocab, hint_pre=len(a.hint_pre), hint_post=len(a.hint_post)) for a in pair.agents}
        log(f'model: {info}; trainable {counts["trainable"]:,} / {counts["total"]:,} ({100 * counts["trainable_frac"]:.3f} %); w {pair.w_value}; fillers / template {fillers}')
        tr = M.train_steps(pair, lambda b: pair.step(b, train=True), M.batch_stream(train, args.batch, args.seed), args.steps, 0.0, args.lr_adapter,
                           args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/{args.pair}/s{args.seed}', weight_decay=args.weight_decay,
                           extra_log=lambda i: f'msg acc {i["msg_acc"]:.3f}' + (f' w {pair.w_value:.4f}' if pair.w is not None else ''))
        tr['truncation_train'] = dict(pair.trunc)
        tr['message'] = pair.msg_summary()
        tr['answers_generated'] = {a.name: a.n_generated for a in pair.agents}
        log(f'train done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
            f'message token acc first/last {tr["message"].get("msg_acc_first")}/{tr["message"].get("msg_acc_last")}; '
            f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed); sender answers generated {tr["answers_generated"]}')
        state = M.trainable_state(pair)
        torch.save(dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info,
                        receiver=receiver, config=vars(args)), ckp)
        log('trainable.pt saved before the evaluation')
        t_eval = time.time()
        ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=True, log=log)
        v = MD.eval_render(pair, val, hard=True, drop_p=0.0, c_check=True, seed=args.seed, **ekw)
        log(f'VAL p=0.0 (hard, criterion): EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {_owner(v)}')
        c_assert = None
        if pair.arm in M.FUSION_ARMS:
            c_assert = bool(v['c_assert'])
            log(f'C assertion: {c_assert} {v["c_assert_detail"]}')
        f = MD.eval_render(pair, fresh, hard=True, drop_p=0.0, c_check=False, seed=args.seed, **ekw)
        log(f'fresh (hard): EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        n0 = MD.eval_baseline(pair, val, 'none', **ekw)
        log(f'VAL baseline none (the plain B0 prompt): EM {n0["em"]:.2f} F1 {n0["f1"]:.2f} owner {_owner(n0)}')
        lam_bucket = MD.lam_by_bucket(v['records']) if pair.arm in M.FUSION_ARMS else None
        if lam_bucket:
            log(f'lam by bucket (VAL): {lam_bucket}')
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
    summary = dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=MD.BATCH, prereg=MD.PREREG, val=v, fresh=f,
                   absence=absence, soft=ro['soft'], baseline=ro['baseline'], hint=ro['hint'], lam_by_bucket=lam_bucket, complete=bool(tr['steps_done'] == args.steps),
                   steps=tr['steps_done'], steps_planned=args.steps, readouts_incomplete=where, agents=info, params=counts, train=tr, receiver=receiver,
                   init=(None if init is None else {k: v for k, v in init.items() if k != 'loaded'}),
                   channel=dict(tau=args.tau, n_blocks=args.n_blocks, k=args.k, d_e=args.d_e, filler=fillers, msg_tokens=args.k, rendering='hard',
                                hint_pre=MD.HINT_PRE, hint_post=MD.HINT_POST, loss='position CE vs the gold answer', sender_pass1='B0 prompt + own greedy answer, LoRA enabled', adapter='SenderAdapter3 (LayerNorm on hidden states before k/v, position 0 masked)', init='stage-P dictionary pretraining (l1g_pretrain.py)' if args.init_from else 'fresh',
                                receiver_training='none (frozen B0 checkpoint)'),
                   timings=dict(load_s=load_s, train_s=tr['seconds'], eval_s=eval_s, total_s=time.time() - t_start,
                                step_time_post_warmup_s=tr['step_time_post_warmup_s'], timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS), u_drop_p=M.U_DROP_P, edges=M.EDGES,
                               wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__, data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data',
                               r2_rounds=M.R2_ROUNDS),
                   scripts_sha=M.scripts_sha(MD.SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if pair.arm in M.FUSION_ARMS:
        summary['c_assert'] = c_assert
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l1g_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} (none {n0["em"]:.2f}) fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'readouts {"complete" if not where else "INCOMPLETE (" + where + ")"} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
