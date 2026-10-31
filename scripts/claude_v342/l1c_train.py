"""l1c_train.py (Claude, 2026-10-31): batch L1c trainer for the arms Cu / Cs (pre-registration D8-86).
Mirrors l1b_train.py (imports the PINNED l1_train.py for read_data / agents_info / incomplete / AGENT_KEYS) with l1c_model.build_pair: the receiver
is the frozen B0 checkpoint of the same seed (--receiver-from runs/l1/B0-hetero-s<seed>), the sender adapter + decoder are trained with the message
CE only (no answer loss, no trainable LoRA). Evaluation: the pinned eval_latent (soft message, p = 0: the criterion; C assertion for Cs), fresh,
then under the soft deadline the readouts: absence p = 1, the hard (argmax) message on VAL / fresh, the baselines none (= the B0 prompt) and
gold32 (the target tokens' real embeddings at the slot) on VAL / fresh, and the message metrics. The VAL baseline `none` is part of the
criterion (G1 compares against it) and is computed before the readouts.
Writes <out>/summary.json in the l1_aggregate schema plus the L1c blocks (hard, baseline, message, receiver, channel); trainable.pt before the
evaluation; INCOMPLETE.json (exit 3) when the deadline falls into training or the criterion evaluation.
  python -B scripts/claude_v342/l1c_train.py --arm Cu --pair hetero --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --train T.jsonl --val V.jsonl --fresh F.jsonl --out DIR
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

STEPS = 5000                 # D8-86 section 8: 2 epochs of the 20,000 TRAIN examples at batch 8
SEEDS = T1.SEEDS


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=MC.ARMS)
    ap.add_argument('--pair', choices=tuple(M.MODELS))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (trainable.pt = the frozen receiver LoRA); required unless --tiny')
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--k', type=int, default=M.K_SLOTS); ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--n-blocks', type=int, default=MC.N_BLOCKS)
    ap.add_argument('--tau', type=float, default=MC.TAU)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=16)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--absence-ps', default=','.join(str(p) for p in MC.ABSENCE_PS))
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='random tiny LlamaConfig models + WordLevel tokenizers (tests only)')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--debug-deadline-at', default=None, choices=('readouts',), help='tests only: deadline in the past right before the readouts')
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def readouts(pair, val, fresh, args, log, deadline):
    """soft-deadline readouts -> (dict(absence, hard, baseline, message), where it stopped or None)"""
    out, where = dict(absence={}, hard={}, baseline={'none': {}, 'gold32': {}}, message={}), None
    ps = [float(p) for p in args.absence_ps.split(',') if p.strip() and float(p) != 0.0]
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, keep_preds=False, log=log)
    try:
        # order by diagnostic value per second: the decoded messages (no generation) and the comparable ceiling first (review MINOR 4)
        for split, exs in (('val', val), ('fresh', fresh)):
            where = f'message {split}'
            r = MC.message_metrics(pair, exs, eval_batch=args.eval_batch, deadline=deadline, log=log)
            out['message'][split] = r
            log(f'{split} messages: token acc {r["token_acc"]:.3f} (message tokens {r["msg_token_acc"]}) F1 vs gold full {r["f1_full"]:.3f} / cut {r["f1_trunc"]:.3f} '
                f'tokens {r["n_tokens_mean"]}; e.g. {r["examples"][0]["pred"][:120]!r}' if r['examples'] else f'{split} messages: no examples')
        where = 'baseline gold32 val'
        r = MC.eval_baseline(pair, val, 'gold32', deadline=deadline, **ekw)
        out['baseline']['gold32']['val'] = r
        log(f'VAL baseline gold32: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]}')
        for p in ps:
            where = f'absence p={p}'
            r = M.eval_latent(pair, val, drop_p=p, seed=args.seed, c_check=False, deadline=deadline, **ekw)
            out['absence'][str(p)] = r
            log(f'VAL p={p}: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]} dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        for split, exs in (('val', val), ('fresh', fresh)):
            where = f'hard {split}'
            r = MC.eval_hard(pair, exs, seed=args.seed, deadline=deadline, **ekw)
            out['hard'][split] = r
            log(f'{split} hard message: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]}')
        for kind in ('none', 'gold32'):
            where = f'baseline {kind} fresh'
            r = MC.eval_baseline(pair, fresh, kind, deadline=deadline, **ekw)
            out['baseline'][kind]['fresh'] = r
            log(f'fresh baseline {kind}: EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]}')
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
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l1c_train start pid {os.getpid()} host {platform.node()} arm {args.arm} (base {MC.BASE_ARM[args.arm]}) pair {args.pair} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it (--tiny / --allow-fallback-data for smoke only)')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    log(f'device {device} torch {torch.__version__} data api {"FALLBACK (l1_data absent)" if M.DATA_IS_FALLBACK else "l1_data"}'
        f'{"" if args.seed in SEEDS else " (NOTE: seed not pre-registered)"}; tau {args.tau} blocks {args.n_blocks} absence ps {args.absence_ps}')
    train, val, fresh = T1.read_data(args, log)
    log(f'steps {args.steps} x batch {args.batch} = {args.steps * args.batch} examples ({args.steps * args.batch / max(1, len(train)):.2f} epochs)')
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = MC.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        b0 = os.path.join(args.receiver_from, 'summary.json')
        if os.path.isfile(b0):
            s0 = M.jload(b0)
            receiver.update(b0_arm=s0.get('arm'), b0_seed=s0.get('seed'), b0_val_em=(s0.get('val') or {}).get('em'), b0_fresh_em=(s0.get('fresh') or {}).get('em'))
        if receiver.get('arm') not in (None, 'B0') or (receiver.get('seed') not in (None, args.seed)):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        for k in ('lora_r', 'lora_alpha'):
            if receiver.get(k) is not None and receiver[k] != getattr(args, k):
                raise SystemExit(f'--receiver-from {k} = {receiver[k]} differs from --{k.replace("_", "-")} {getattr(args, k)} (a silent LoRA scale mismatch)')
        log(f'receiver: {receiver}')
    ckp = os.path.join(args.out, 'trainable.pt')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair = MC.build_pair(args.arm, args.pair, device, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=False, max_ctx=args.max_ctx_tokens,
                             receiver_state=receiver_state, tau=args.tau, n_blocks=args.n_blocks, **agent_kw)
        load_s = time.time() - t0
        counts = M.count_params(pair)
        info = T1.agents_info(pair)
        if receiver is not None and receiver.get('models') is not None and list(receiver['models']) != [a['name'] for a in info]:
            raise SystemExit(f'--receiver-from models {receiver["models"]} differ from this pair {[a["name"] for a in info]}')
        fillers = {a.name: dict(id=a.filler, is_pad=a.filler_is_pad, vocab=a.reader.vocab) for a in pair.agents}
        log(f'model: {info}; trainable {counts["trainable"]:,} / {counts["total"]:,} ({100 * counts["trainable_frac"]:.3f} %); w {pair.w_value}; fillers {fillers}')
        tr = M.train_steps(pair, lambda b: pair.step(b, train=True), M.batch_stream(train, args.batch, args.seed), args.steps, 0.0, args.lr_adapter,
                           args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/{args.pair}/s{args.seed}', weight_decay=args.weight_decay,
                           extra_log=lambda i: f'msg acc {i["msg_acc"]:.3f}' + (f' w {pair.w_value:.4f}' if pair.w is not None else ''))
        tr['truncation_train'] = dict(pair.trunc)
        tr['message'] = pair.msg_summary()
        log(f'train done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
            f'message token acc first/last {tr["message"].get("msg_acc_first")}/{tr["message"].get("msg_acc_last")}; '
            f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed); truncations {tr["truncation_train"]}')
        state = M.trainable_state(pair)
        torch.save(dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info,
                        receiver=receiver, config=vars(args)), ckp)
        log('trainable.pt saved before the evaluation')
        t_eval = time.time()
        ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=args.keep_preds, log=log)
        v = M.eval_latent(pair, val, drop_p=0.0, seed=args.seed, c_check=True, **ekw)
        log(f'VAL p=0.0 (soft): EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {[(k, x["em"]) for k, x in v["em_by_owner"].items()]}')
        c_assert = None
        if pair.arm in M.FUSION_ARMS:
            c_assert = bool(v['c_assert'])
            log(f'C assertion: {c_assert} {v["c_assert_detail"]}')
        f = M.eval_latent(pair, fresh, drop_p=0.0, seed=args.seed, c_check=False, **ekw)
        log(f'fresh (soft): EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        n0 = MC.eval_baseline(pair, val, 'none', **ekw)
        log(f'VAL baseline none (the frozen receiver, no slot): EM {n0["em"]:.2f} F1 {n0["f1"]:.2f} owner {[(k, x["em"]) for k, x in n0["em_by_owner"].items()]}')
        if args.debug_deadline_at == 'readouts':
            deadline = time.time() - 1.0
            log('debug: deadline moved into the past before the readouts')
        ro, where = readouts(pair, val, fresh, args, log, deadline)
        ro['baseline']['none']['val'] = n0
        absence = {str(0.0): v, **ro['absence']}
        if args.keep_preds:
            M.jdump(v.pop('records'), os.path.join(args.out, 'preds_val.json'))
            M.jdump(f.pop('records'), os.path.join(args.out, 'preds_fresh.json'))
            M.jdump(n0.pop('records'), os.path.join(args.out, 'preds_val_none.json'))
        eval_s = time.time() - t_eval
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for a in info))
    summary = dict(arm=args.arm, base_arm=pair.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=MC.BATCH, prereg=MC.PREREG, val=v, fresh=f,
                   absence=absence, hard=ro['hard'], baseline=ro['baseline'], message=ro['message'], complete=bool(tr['steps_done'] == args.steps),
                   steps=tr['steps_done'], steps_planned=args.steps, readouts_incomplete=where, agents=info, params=counts, train=tr, receiver=receiver,
                   channel=dict(tau=args.tau, n_blocks=args.n_blocks, k=args.k, d_e=args.d_e, filler=fillers, msg_tokens=args.k, loss='position CE vs partner gold facts',
                                receiver_training='none (frozen B0 checkpoint)', pass1_lora='disabled'),
                   timings=dict(load_s=load_s, train_s=tr['seconds'], eval_s=eval_s, total_s=time.time() - t_start,
                                step_time_post_warmup_s=tr['step_time_post_warmup_s'], timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS), u_drop_p=M.U_DROP_P, edges=M.EDGES,
                               wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__, data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data',
                               r2_rounds=M.R2_ROUNDS),
                   scripts_sha=M.scripts_sha(MC.SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if pair.arm in M.FUSION_ARMS:
        summary['c_assert'] = c_assert
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l1c_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} (none {n0["em"]:.2f}) fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'readouts {"complete" if not where else "INCOMPLETE (" + where + ")"} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
