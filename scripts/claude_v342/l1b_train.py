"""l1b_train.py (Claude, 2026-10-30): batch L1b trainer for the arms S1b / Ub / S1L1b (pre-registration D8-85).
Mirrors l1_train.py (PINNED and untouched; imported for read_data / agents_info / incomplete / AGENT_KEYS) with l1b_model.build_pair (AgentB:
embedding-scale reader; PairB: answer CE + beta * partner-facts CE), the absence curve at p in {0, 1} (D8-85 3c) and the decodability readouts.
Writes <out>/summary.json in the l1_aggregate schema (arm = the L1b arm name, base_arm = its L1 arm) plus the L1b blocks (aux, decodability);
trainable.pt before the evaluation; INCOMPLETE.json (exit 3) when the deadline falls into training or the criterion evaluation (VAL p = 0,
fresh); the readouts (absence p = 1, decodability) are under the soft deadline (readouts_incomplete).
  python -B scripts/claude_v342/l1b_train.py --arm S1b --pair hetero --seed 342101 --train T.jsonl --val V.jsonl --fresh F.jsonl --out DIR
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
import l1b_model as MB  # noqa: E402

EPOCH_STEPS = T1.EPOCH_STEPS
SEEDS = T1.SEEDS


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=MB.ARMS)
    ap.add_argument('--pair', choices=tuple(M.MODELS))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--steps', type=int, default=EPOCH_STEPS)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr-lora', type=float, default=2e-4)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--k', type=int, default=M.K_SLOTS); ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=M.LORA_DROPOUT)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--absence-ps', default=','.join(str(p) for p in MB.ABSENCE_PS), help='D8-85 3c: p = 0 (criterion) and 1 (no message) by default')
    ap.add_argument('--beta', type=float, default=MB.AUX_BETA, help='weight of the partner-facts CE (D8-85 3a)')
    ap.add_argument('--aux-max-tokens', type=int, default=MB.AUX_MAX_TOKENS, help='the partner facts are cut to this many receiver tokens')
    ap.add_argument('--decode-n', type=int, default=MB.DECODE_N, help='decodability readout on the first N VAL examples (0 = skip)')
    ap.add_argument('--decode-max-new', type=int, default=MB.DECODE_MAX_NEW)
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='random tiny LlamaConfig models + WordLevel tokenizers (tests only)')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--no-grad-ckpt', action='store_true')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--debug-deadline-at', default=None, choices=('readouts',), help='tests only: deadline in the past right before the readouts')
    ap.add_argument('--keep-preds', action='store_true', help='also write the per-example predictions (preds_val.json / preds_fresh.json)')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def readouts(pair, val, args, log, deadline):
    """the soft-deadline readouts: the absence curve at p in --absence-ps \\ {0} on VAL, then the decodability at p = 0 and p = 1 on
    VAL[:decode_n]. -> (dict(absence={p: block}, decodability={p: block}), where it stopped or None)"""
    out, where = dict(absence={}, decodability={}), None
    ps = [float(p) for p in args.absence_ps.split(',') if p.strip() and float(p) != 0.0]
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, keep_preds=args.keep_preds, log=log)
    try:
        for p in ps:
            where = f'absence p={p}'
            r = M.eval_latent(pair, val, drop_p=p, seed=args.seed, c_check=False, deadline=deadline, **ekw)
            out['absence'][str(p)] = r
            log(f'VAL p={p}: EM {r["em"]:.2f} F1 {r["f1"]:.2f} agents {r["em_agent"]} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]} '
                f'dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        if args.decode_n > 0:
            sub = val[:args.decode_n]
            for p in (0.0, 1.0):
                where = f'decodability p={p}'
                r = MB.eval_decodability(pair, sub, drop_p=p, eval_batch=args.eval_batch, seed=args.seed, max_new_tokens=args.decode_max_new, deadline=deadline, log=log)
                out['decodability'][str(p)] = r
                log(f'decodability p={p}: message F1 vs partner gold {r["f1"]:.3f} agents {r["f1_agent"]} tokens {r["n_tokens_mean"]} ({r["n"]} examples)')
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
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l1b_train start pid {os.getpid()} host {platform.node()} arm {args.arm} (base {MB.BASE_ARM[args.arm]}) pair {args.pair} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it (--tiny / --allow-fallback-data for smoke only)')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    log(f'device {device} torch {torch.__version__} data api {"FALLBACK (l1_data absent)" if M.DATA_IS_FALLBACK else "l1_data"}'
        f'{"" if args.seed in SEEDS else " (NOTE: seed not pre-registered)"}; beta {args.beta} aux max tokens {args.aux_max_tokens} absence ps {args.absence_ps}')
    train, val, fresh = T1.read_data(args, log)
    log(f'steps {args.steps} x batch {args.batch} = {args.steps * args.batch} examples ({args.steps * args.batch / max(1, len(train)):.2f} epochs)')
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS}
    ckp = os.path.join(args.out, 'trainable.pt')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair = MB.build_pair(args.arm, args.pair, device, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=not args.no_grad_ckpt, max_ctx=args.max_ctx_tokens,
                             beta=args.beta, aux_max=args.aux_max_tokens, **agent_kw)
        load_s = time.time() - t0
        counts = M.count_params(pair)
        info = T1.agents_info(pair)
        scales = {a.name: a.reader_scale for a in pair.agents}
        log(f'model: {info}; trainable {counts["trainable"]:,} / {counts["total"]:,} ({100 * counts["trainable_frac"]:.2f} %); w {pair.w_value}; reader scales {scales}')
        tr = M.train_steps(pair, lambda b: pair.step(b, train=True), M.batch_stream(train, args.batch, args.seed), args.steps, args.lr_lora, args.lr_adapter,
                           args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/{args.pair}/s{args.seed}', weight_decay=args.weight_decay,
                           extra_log=lambda i: f'ans {i["loss_ans"]:.3f} aux {i["loss_aux"]:.3f}' + (f' w {pair.w_value:.4f}' if pair.w is not None else ''))
        tr['truncation_train'] = dict(pair.trunc)
        tr['aux'] = pair.aux_summary()
        log(f'train done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
            f'ans first/last {tr["aux"].get("loss_ans_first")}/{tr["aux"].get("loss_ans_last")} aux first/last {tr["aux"].get("loss_aux_first")}/{tr["aux"].get("loss_aux_last")}; '
            f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed); truncations {tr["truncation_train"]}')
        state = M.trainable_state(pair)
        torch.save(dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info,
                        reader_scale=scales, config=vars(args)), ckp)
        log('trainable.pt saved before the evaluation')
        t_eval = time.time()
        ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=args.keep_preds, log=log)
        v = M.eval_latent(pair, val, drop_p=0.0, seed=args.seed, c_check=True, **ekw)
        log(f'VAL p=0.0: EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {[(k, x["em"]) for k, x in v["em_by_owner"].items()]}')
        c_assert = None
        if pair.arm in M.FUSION_ARMS:
            c_assert = bool(v['c_assert'])                                   # true or false, never skipped
            log(f'C assertion: {c_assert} {v["c_assert_detail"]}')
        f = M.eval_latent(pair, fresh, drop_p=0.0, seed=args.seed, c_check=False, **ekw)
        log(f'fresh: EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        if args.debug_deadline_at == 'readouts':
            deadline = time.time() - 1.0
            log('debug: deadline moved into the past before the readouts')
        ro, where = readouts(pair, val, args, log, deadline)
        absence = {str(0.0): v, **ro['absence']}
        if args.keep_preds:
            M.jdump(v.pop('records'), os.path.join(args.out, 'preds_val.json'))
            M.jdump(f.pop('records'), os.path.join(args.out, 'preds_fresh.json'))
            for p, r in absence.items():
                if 'records' in r:
                    M.jdump(r.pop('records'), os.path.join(args.out, f'preds_val_p{p}.json'))
        eval_s = time.time() - t_eval
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for a in info))
    summary = dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=MB.BATCH, prereg=MB.PREREG, val=v, fresh=f,
                   absence=absence, decodability=ro['decodability'], complete=bool(tr['steps_done'] == args.steps), steps=tr['steps_done'], steps_planned=args.steps,
                   epoch_steps=EPOCH_STEPS, readouts_incomplete=where, agents=info, params=counts, train=tr,
                   aux=dict(beta=args.beta, max_tokens=args.aux_max_tokens, prompt=MB.AUX_PROMPT, reader_scale=scales, train=tr['aux']),
                   timings=dict(load_s=load_s, train_s=tr['seconds'], eval_s=eval_s, total_s=time.time() - t_start,
                                step_time_post_warmup_s=tr['step_time_post_warmup_s'], timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), k=args.k, d_e=args.d_e, log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS),
                               u_drop_p=M.U_DROP_P, edges=M.EDGES, wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__,
                               data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data', r2_rounds=M.R2_ROUNDS),
                   scripts_sha=M.scripts_sha(MB.SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if pair.arm in M.FUSION_ARMS:
        summary['c_assert'] = c_assert
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l1b_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'readouts {"complete" if not where else "INCOMPLETE (" + where + ")"} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
