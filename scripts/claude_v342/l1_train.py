"""l1_train.py (Claude, 2026-10-29): batch L1 trainer for the arms B0 / B4 / U / S1 / S1L1 (pre-registration D8-82; the model side is
l1_model.py, the fusion l1_fusion.py, the prompts / metrics l1_data.py, the verdict l1_aggregate.py). One run = one arm x one pair x one seed:
  train one epoch (TRAIN 20,000 at batch 8 = 2,500 steps; --steps stops earlier / wraps around with a reshuffle) with AdamW
  (LoRA 2e-4, adapters / readers / w 1e-3), 100 warm-up steps + cosine, bf16 autocast on CUDA, grad clipping 1.0, everything seeded;
  then evaluate on VAL (latent arms: the absence curve p in {0, .3, .6, 1}; the p = 0 pass is the judged one and carries the C assertion)
  and on fresh (p = 0 only); write <out>/summary.json (l1_aggregate.summary_schema keys + readouts + config + timings + scripts_sha),
  <out>/trainable.pt (LoRA + adapters + readers + w only) and <out>/log.txt. Never overwrites an existing summary.json.
B4 = the full-context upper bound of EVERY model of the pair: one B4 run trains the hetero models one after the other (Qwen3 then Llama;
  same pair: Qwen3 only) and writes one summary whose val.em_model / em_agent hold both (l1_aggregate reads P0 per model from the single
  B4-<pair>-s<seed> run); --model restricts a B4 run to one model (diagnostics; such a summary cannot feed P0 for the other model).
c_assert is written for S1 / S1L1 only (true or false, never skipped); the key is absent for arms without fusion (B0, B4, U).
Usage:
  python -B scripts/claude_v342/l1_train.py --arm S1 --pair hetero --seed 342101 --train T.jsonl --val V.jsonl --fresh F.jsonl --out DIR
  --tiny builds random tiny models (unit tests); --limit-train / --limit-eval cut the data; --deadline-epoch E stops with INCOMPLETE.json
  (exit 3) instead of overrunning a lease.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import argparse  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402

EPOCH_STEPS = 2500          # 20,000 TRAIN examples / batch 8
SEEDS = (342101, 342202)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', required=True, choices=M.ARMS)
    ap.add_argument('--pair', required=True, choices=tuple(M.MODELS))
    ap.add_argument('--model', default=None, help='B4 only: restrict the run to one model (default: every model of the pair; --tiny: tiny-a / tiny-b)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--train', required=True); ap.add_argument('--val', required=True); ap.add_argument('--fresh', required=True)
    ap.add_argument('--out', required=True)
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
    ap.add_argument('--absence-ps', default=','.join(str(p) for p in M.ABSENCE_PS))
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='random tiny LlamaConfig models + WordLevel tokenizers (tests only)')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--no-grad-ckpt', action='store_true')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--keep-preds', action='store_true', help='also write the per-example predictions (preds_val.json / preds_fresh.json)')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def incomplete(out, reason, log):
    M.jdump(dict(incomplete=True, reason=reason, utc=M.utc()), os.path.join(out, 'INCOMPLETE.json'))
    log(f'L1 INCOMPLETE: {reason}')


def run_pair(pair, args, device, log, deadline, label, tag=''):
    """train + evaluate one Pair -> dict(tr, val, fresh, absence, c_assert (fusion arms), agents, params, state)"""
    counts = M.count_params(pair)
    agents_info = [dict(name=a.name, n_layers=a.n_layers, read_layer=a.layer, d_model=a.d_model, bos=a.bos, eos=a.eos, pad=a.pad, dtype=str(a.dtype),
                        fixed_lam=(a.adapter.fixed_lam if a.adapter is not None else None), comm=a.comm, lora=a.lora) for a in pair.agents]
    log(f'model{tag}: {agents_info}; trainable {counts["trainable"]:,} / {counts["total"]:,} ({100 * counts["trainable_frac"]:.2f} %); w {pair.w_value}')
    stream = M.batch_stream(pair.train_examples, args.batch, args.seed)
    tr = M.train_steps(pair, lambda b: pair.step(b, args.arm, train=True), stream, args.steps, args.lr_lora, args.lr_adapter, args.warmup, args.clip, device, log,
                       args.log_every, deadline, label=label, weight_decay=args.weight_decay,
                       extra_log=(lambda info: f'w {pair.w_value:.4f}') if pair.w is not None else None)
    tr['truncation_train'] = dict(pair.trunc)
    log(f'train{tag} done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
        f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed); truncations {tr["truncation_train"]}')
    state = M.trainable_state(pair)
    t_eval = time.time()
    ps = [float(p) for p in args.absence_ps.split(',') if p.strip()]
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=args.keep_preds, log=log)
    out = dict(agents=agents_info, params=counts, state=state, tr=tr)
    if args.arm in M.LATENT_ARMS:
        absence = M.absence_curve(pair, pair.val_examples, ps, seed=args.seed, **ekw)
        v = absence[str(0.0)]
        f = M.eval_latent(pair, pair.fresh_examples, drop_p=0.0, seed=args.seed, c_check=False, **ekw)
        for p, r in absence.items():
            log(f'VAL{tag} p={p}: EM {r["em"]:.2f} F1 {r["f1"]:.2f} agents {r["em_agent"]} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]} '
                f'dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        if args.arm in M.FUSION_ARMS:
            out['c_assert'] = bool(v['c_assert'])                     # true or false, never skipped
            log(f'C assertion{tag}: {out["c_assert"]} {v["c_assert_detail"]}')
    else:
        absence = None
        v = M.eval_plain(pair, pair.val_examples, **ekw)
        f = M.eval_plain(pair, pair.fresh_examples, **ekw)
        log(f'VAL{tag}: EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} models {v["em_model"]} owner {[(k, x["em"]) for k, x in v["em_by_owner"].items()]}')
    log(f'fresh{tag}: EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
    if args.keep_preds:                                                   # the per-example predictions leave the summary (preds_*.json)
        sfx = ('_' + tag.strip(' []').replace('/', '_')) if tag else ''
        M.jdump(v.pop('records'), os.path.join(args.out, f'preds_val{sfx}.json'))
        M.jdump(f.pop('records'), os.path.join(args.out, f'preds_fresh{sfx}.json'))
        for p, r in (absence or {}).items():
            if 'records' in r:
                M.jdump(r.pop('records'), os.path.join(args.out, f'preds_val_p{p}{sfx}.json'))
    out.update(val=v, fresh=f, absence=absence, eval_s=time.time() - t_eval)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def merge_models(names, blocks):
    """B4: the per-model VAL / fresh blocks -> one block (em_model per model, em_agent in model order, em = mean over the models;
    every model answered the same examples, so bucket means are means over the models); the full per-model readouts under per_model"""
    out = dict(n=blocks[0]['n'], n_records=sum(b['n_records'] for b in blocks), em=_mean([b['em'] for b in blocks]), f1=_mean([b['f1'] for b in blocks]),
               em_agent=[b['em'] for b in blocks], f1_agent=[b['f1'] for b in blocks], em_model={n: b['em'] for n, b in zip(names, blocks)},
               f1_model={n: b['f1'] for n, b in zip(names, blocks)},
               em_by_owner={rel: dict(em=_mean([b['em_by_owner'][rel]['em'] for b in blocks]), n=sum(b['em_by_owner'][rel]['n'] for b in blocks)) for rel in ('own', 'other', 'none', 'all')},
               em_by_type={t: dict(em=_mean([b['em_by_type'].get(t, {}).get('em') for b in blocks]), n=sum(b['em_by_type'].get(t, {}).get('n', 0) for b in blocks))
                           for t in sorted({t for b in blocks for t in b['em_by_type']})},
               answer_gen_s_per_example=_mean([b['answer_gen_s_per_example'] for b in blocks]), answer_decode_steps_mean=_mean([b['answer_decode_steps_mean'] for b in blocks]),
               sender_latency_s_per_example=0.0, wire_bytes_per_example=0, truncation={k: sum(b['truncation'].get(k, 0) for b in blocks) for k in sorted({k for b in blocks for k in b['truncation']})},
               per_model={n: b for n, b in zip(names, blocks)})
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l1_train start pid {os.getpid()} host {platform.node()} arm {args.arm} pair {args.pair} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it (--tiny / --allow-fallback-data for smoke only)')
    if args.arm != 'B4' and args.model:
        raise SystemExit('--model is only for B4')
    if args.arm == 'B4' and args.model and not args.tiny and args.model not in M.MODELS[args.pair]:
        raise SystemExit(f'--model {args.model} is not a model of pair {args.pair}')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    log(f'device {device} torch {torch.__version__} data api {"FALLBACK (l1_data absent)" if M.DATA_IS_FALLBACK else "l1_data"}'
        f'{"" if args.seed in SEEDS else " (NOTE: seed not pre-registered)"}')
    # ---------------------------------------------------------------- data
    train = M.DATA.read_jsonl(args.train); val = M.DATA.read_jsonl(args.val); fresh = M.DATA.read_jsonl(args.fresh)
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_eval:
        val, fresh = val[:args.limit_eval], fresh[:args.limit_eval]
    log(f'data: train {len(train)} val {len(val)} fresh {len(fresh)}; steps {args.steps} x batch {args.batch} = {args.steps * args.batch} examples '
        f'({args.steps * args.batch / max(1, len(train)):.2f} epochs)')
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = dict(k=args.k, d_e=args.d_e, n_heads=args.n_heads, layer_frac=args.layer_frac, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout)
    if args.arm == 'B4':
        if args.tiny:
            names = [args.model] if args.model else (list(M.TINY_NAMES) if args.pair == 'hetero' else [M.TINY_NAMES[0]])
        else:
            names = M.model_names(args.pair, 'B4', args.model)
        models = names
    else:
        names = [None]
        models = M.model_names(args.pair, args.arm)
    # ---------------------------------------------------------------- train + evaluate (B4: one Pair per model, sequentially)
    results, load_s = [], 0.0
    try:
        for name in names:
            M.seed_all(args.seed)
            t0 = time.time()
            pair = M.build_pair(args.arm, args.pair, device, model=name, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=not args.no_grad_ckpt,
                                max_ctx=args.max_ctx_tokens, **agent_kw)
            pair.train_examples, pair.val_examples, pair.fresh_examples = train, val, fresh
            load_s += time.time() - t0
            tag = f' [{name}]' if name else ''
            results.append(run_pair(pair, args, device, log, deadline, label=f'{args.arm}/{args.pair}/s{args.seed}{tag}', tag=tag))
            if len(names) > 1:
                del pair
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    except M.Deadline as e:
        incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for r in results for a in r['agents']))     # the agents actually built (--tiny: tiny-a / tiny-b)
    # ---------------------------------------------------------------- summary
    if args.arm == 'B4':
        v, f = merge_models(models, [r['val'] for r in results]), merge_models(models, [r['fresh'] for r in results])
        trb = {n: r['tr'] for n, r in zip(models, results)}
        states = {n: r['state'] for n, r in zip(models, results)}
        steps_done = min(r['tr']['steps_done'] for r in results)
        step_times = {n: r['tr']['step_time_post_warmup_s'] for n, r in zip(models, results)}
        w = None
    else:
        v, f = results[0]['val'], results[0]['fresh']
        trb, states = results[0]['tr'], results[0]['state']
        steps_done, step_times = results[0]['tr']['steps_done'], results[0]['tr']['step_time_post_warmup_s']
    torch.save(dict(arm=args.arm, pair=args.pair, seed=args.seed, models=models, state=states, agents=[a for r in results for a in r['agents']], config=vars(args)),
               os.path.join(args.out, 'trainable.pt'))
    summary = dict(arm=args.arm, pair=args.pair, seed=args.seed, model=args.model, models=models, batch_name=M.BATCH, prereg=M.PREREG, val=v, fresh=f,
                   absence=results[0]['absence'] if args.arm != 'B4' else None, complete=bool(steps_done == args.steps), steps=steps_done, steps_planned=args.steps,
                   epoch_steps=EPOCH_STEPS, agents=[a for r in results for a in r['agents']], params=[r['params'] for r in results] if args.arm == 'B4' else results[0]['params'],
                   train=trb, timings=dict(load_s=load_s, train_s=sum(r['tr']['seconds'] for r in results), eval_s=sum(r['eval_s'] for r in results),
                                           total_s=time.time() - t_start, step_time_post_warmup_s=step_times, timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), k=args.k, d_e=args.d_e, log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS),
                               u_drop_p=M.U_DROP_P, edges=M.EDGES, wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__,
                               data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data', r2_rounds=M.R2_ROUNDS),
                   scripts_sha=M.scripts_sha(('l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')), utc_end=M.utc(), pid=os.getpid())
    if args.arm in M.FUSION_ARMS:
        summary['c_assert'] = results[0]['c_assert']
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l1_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
