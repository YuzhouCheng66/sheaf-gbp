"""l1_train.py (Claude, 2026-10-29): batch L1 trainer for the arms B0 / B4 / U / S1 / S1L1 (pre-registration D8-82; the model side is
l1_model.py, the fusion l1_fusion.py, the prompts / metrics l1_data.py, the verdict l1_aggregate.py). One run = one arm x one pair x one seed:
  train one epoch (TRAIN 20,000 at batch 8 = 2,500 steps; --steps stops earlier / wraps around with a reshuffle) with AdamW
  (LoRA 2e-4, adapters / readers / w 1e-3), 100 warm-up steps + cosine, bf16 autocast on CUDA, grad clipping 1.0, everything seeded;
  save <out>/trainable.pt (LoRA + adapters + readers + w only); then evaluate IN THIS ORDER: VAL at p = 0 (the criterion metric; carries the
  C assertion), fresh at p = 0, then the readouts (the absence curve p in {.3, .6, 1} on VAL); write <out>/summary.json
  (l1_aggregate.summary_schema keys + readouts + config + timings + scripts_sha) and <out>/log.txt. Never overwrites a summary.json.
Deadline (--deadline-epoch, the chain passes lease end - 60 s): during training or the VAL / fresh evaluation -> INCOMPLETE.json, exit 3, no
  summary.json; during the readouts -> summary.json with every complete key plus "readouts_incomplete": "<where it stopped>", exit 0 (the
  aggregator needs only val / fresh em / f1 / em_model and c_assert, all from the VAL / fresh p = 0 passes).
--eval-only DIR: load DIR/trainable.pt of a finished run and recompute only the readouts into DIR/readouts.json (summary.json untouched),
  so a run with readouts_incomplete can be completed later on the GPU; --val / --limit-eval / --eval-batch may override the run's config.
B4 = the full-context upper bound of EVERY model of the pair. l1_aggregate reads P0 per model from the single B4-<pair>-s<seed>/summary.json
  (val.em_model with both hetero models). Without --model one B4 run trains the pair's models one after the other (same pair: Qwen3 only;
  hetero: both, ~111-139 min on the 5090 = too long for a lease); with --model NAME it trains that one model and writes a valid per-model
  summary (em_model with that single key); `--merge-b4 DIR_A DIR_B --out DIR` (CPU only, no models, no data) merges two such per-model
  runs of the same pair / seed into the B4-<pair>-s<seed> summary (merge_models over the two per-model blocks; parts + shas recorded;
  DIR/merged_from.json). The chain runs hetero B4 as B4l (Llama) + B4q (Qwen3) + merge.
c_assert is written for S1 / S1L1 only (true or false, never skipped); the key is absent for arms without fusion (B0, B4, U).
Usage:
  python -B scripts/claude_v342/l1_train.py --arm S1 --pair hetero --seed 342101 --train T.jsonl --val V.jsonl --fresh F.jsonl --out DIR
  python -B scripts/claude_v342/l1_train.py --arm B4 --pair hetero --model meta-llama/Llama-3.2-1B-Instruct --seed 342101 ... --out DIR_L
  python -B scripts/claude_v342/l1_train.py --merge-b4 DIR_Q DIR_L --out runs/l1/B4-hetero-s342101
  python -B scripts/claude_v342/l1_train.py --eval-only DIR [--val V.jsonl]
  --tiny builds random tiny models (unit tests); --limit-train / --limit-eval cut the data; --debug-deadline-at readouts (tests only) puts the
  deadline in the past right before the readouts.
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
SCRIPTS = ('l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
AGENT_KEYS = ('k', 'd_e', 'n_heads', 'layer_frac', 'lora_r', 'lora_alpha', 'lora_dropout')


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=M.ARMS)
    ap.add_argument('--pair', choices=tuple(M.MODELS))
    ap.add_argument('--model', default=None, help='B4 only: restrict the run to one model (default: every model of the pair; --tiny: tiny-a / tiny-b)')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--eval-only', default=None, metavar='DIR', help='recompute only the readouts of a finished run into DIR/readouts.json')
    ap.add_argument('--merge-b4', nargs=2, default=None, metavar=('DIR_A', 'DIR_B'),
                    help='CPU only: merge two per-model B4 runs (--model) of the same pair / seed into --out DIR (the B4-<pair>-s<seed> summary the aggregator reads)')
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
    ap.add_argument('--debug-deadline-at', default=None, choices=('readouts',), help='tests only: deadline in the past right before the readouts')
    ap.add_argument('--keep-preds', action='store_true', help='also write the per-example predictions (preds_val.json / preds_fresh.json)')
    ap.add_argument('--allow-fallback-data', action='store_true', help='use l1_model._FallbackData outside --tiny (smoke only)')
    return ap


def incomplete(out, reason, log):
    M.jdump(dict(incomplete=True, reason=reason, utc=M.utc()), os.path.join(out, 'INCOMPLETE.json'))
    log(f'L1 INCOMPLETE: {reason}')


def agents_info(pair):
    return [dict(name=a.name, n_layers=a.n_layers, read_layer=a.layer, d_model=a.d_model, bos=a.bos, eos=a.eos, pad=a.pad, dtype=str(a.dtype),
                 fixed_lam=(a.adapter.fixed_lam if a.adapter is not None else None), comm=a.comm, lora=a.lora) for a in pair.agents]


def train_pair(pair, train, args, device, log, deadline, label, tag=''):
    """train one Pair -> dict(tr, agents, params, state); a Deadline propagates (INCOMPLETE)"""
    counts = M.count_params(pair)
    info = agents_info(pair)
    log(f'model{tag}: {info}; trainable {counts["trainable"]:,} / {counts["total"]:,} ({100 * counts["trainable_frac"]:.2f} %); w {pair.w_value}')
    tr = M.train_steps(pair, lambda b: pair.step(b, args.arm, train=True), M.batch_stream(train, args.batch, args.seed), args.steps, args.lr_lora, args.lr_adapter,
                       args.warmup, args.clip, device, log, args.log_every, deadline, label=label, weight_decay=args.weight_decay,
                       extra_log=(lambda i: f'w {pair.w_value:.4f}') if pair.w is not None else None)
    tr['truncation_train'] = dict(pair.trunc)
    log(f'train{tag} done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; '
        f'step time post warm-up {tr["step_time_post_warmup_s"]} s ({tr["steps_timed"]} timed); truncations {tr["truncation_train"]}')
    return dict(tr=tr, agents=info, params=counts, state=M.trainable_state(pair))


def readouts_pair(pair, val, args, log, deadline, tag=''):
    """the readouts that may be cut by the soft deadline: the absence curve at p in {.3, .6, 1} on VAL (latent arms).
    -> (dict p -> block (complete ones only), where it stopped or None)"""
    out, where = {}, None
    ps = [float(p) for p in args.absence_ps.split(',') if p.strip() and float(p) != 0.0]
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, keep_preds=args.keep_preds, log=log)
    try:
        for p in ps:
            where = f'absence p={p}'
            r = M.eval_latent(pair, val, drop_p=p, seed=args.seed, c_check=False, deadline=deadline() if callable(deadline) else deadline, **ekw)
            out[str(p)] = r
            log(f'VAL{tag} p={p}: EM {r["em"]:.2f} F1 {r["f1"]:.2f} agents {r["em_agent"]} owner {[(k, x["em"]) for k, x in r["em_by_owner"].items()]} '
                f'dropped {r["n_messages_dropped"]}/{r["n_messages"]}')
        where = None
    except M.Deadline as e:
        log(f'readouts{tag} stopped at {where}: {e}')
        where = f'{where}: {e}'
    return out, where


def eval_pair(pair, val, fresh, args, device, log, deadline, tag=''):
    """VAL p = 0 (criterion + C assertion), fresh p = 0 (a Deadline here propagates: INCOMPLETE), then the readouts (soft deadline).
    -> dict(val, fresh, absence, c_assert (fusion arms), readouts_incomplete, eval_s)"""
    t_eval = time.time()
    ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=args.keep_preds, log=log)
    out = {}
    if args.arm in M.LATENT_ARMS:
        v = M.eval_latent(pair, val, drop_p=0.0, seed=args.seed, c_check=True, **ekw)
        log(f'VAL{tag} p=0.0: EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {[(k, x["em"]) for k, x in v["em_by_owner"].items()]}')
        if args.arm in M.FUSION_ARMS:
            out['c_assert'] = bool(v['c_assert'])                     # true or false, never skipped
            log(f'C assertion{tag}: {out["c_assert"]} {v["c_assert_detail"]}')
        f = M.eval_latent(pair, fresh, drop_p=0.0, seed=args.seed, c_check=False, **ekw)
        log(f'fresh{tag}: EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        if args.debug_deadline_at == 'readouts':
            deadline = time.time() - 1.0
            log('debug: deadline moved into the past before the readouts')
        absence, where = readouts_pair(pair, val, args, log, deadline, tag)
        absence = {str(0.0): v, **absence}
    else:
        v = M.eval_plain(pair, val, **ekw)
        log(f'VAL{tag}: EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} models {v["em_model"]} owner {[(k, x["em"]) for k, x in v["em_by_owner"].items()]}')
        f = M.eval_plain(pair, fresh, **ekw)
        log(f'fresh{tag}: EM {f["em"]:.2f} F1 {f["f1"]:.2f} agents {f["em_agent"]}')
        absence, where = None, None
    if args.keep_preds:                                                   # the per-example predictions leave the summary (preds_*.json)
        sfx = ('_' + tag.strip(' []').replace('/', '_')) if tag else ''
        M.jdump(v.pop('records'), os.path.join(args.out, f'preds_val{sfx}.json'))
        M.jdump(f.pop('records'), os.path.join(args.out, f'preds_fresh{sfx}.json'))
        for p, r in (absence or {}).items():
            if 'records' in r:
                M.jdump(r.pop('records'), os.path.join(args.out, f'preds_val_p{p}{sfx}.json'))
    out.update(val=v, fresh=f, absence=absence, readouts_incomplete=where, eval_s=time.time() - t_eval)
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


def read_data(args, log, need_train=True):
    train = M.DATA.read_jsonl(args.train) if need_train else []
    val = M.DATA.read_jsonl(args.val); fresh = M.DATA.read_jsonl(args.fresh)
    if args.limit_train and need_train:
        train = train[:args.limit_train]
    if args.limit_eval:
        val, fresh = val[:args.limit_eval], fresh[:args.limit_eval]
    log(f'data: train {len(train)} val {len(val)} fresh {len(fresh)}')
    return train, val, fresh


# ================================================================ --eval-only: the readouts of a finished run
def eval_only(args):
    src = args.eval_only
    ckp = os.path.join(src, 'trainable.pt')
    if not os.path.isfile(ckp):
        raise SystemExit(f'{ckp}: missing')
    ck = torch.load(ckp, map_location='cpu', weights_only=False)
    cfg = ck['config']
    for k in ('train', 'val', 'fresh', 'hf_cache'):
        if getattr(args, k) is None:
            setattr(args, k, cfg.get(k))
    for k in ('arm', 'pair', 'seed'):
        setattr(args, k, ck[k])
    for k in AGENT_KEYS + ('max_ctx_tokens', 'absence_ps', 'max_new_tokens', 'model'):
        if k in cfg:
            setattr(args, k, cfg[k])
    args.tiny = bool(cfg.get('tiny'))
    args.out = src
    log = M.Log(os.path.join(src, 'log_readouts.txt'))
    log(f'l1_train eval-only start pid {os.getpid()} host {platform.node()} source {src} arm {args.arm} pair {args.pair} seed {args.seed} '
        f'trainable {M.file_sha256(ckp)[:12]}')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    t0 = time.time()
    out = dict(arm=args.arm, pair=args.pair, seed=args.seed, source=os.path.abspath(src), trainable_sha256=M.file_sha256(ckp), eval_only=True,
               config=dict(val=args.val, limit_eval=args.limit_eval, eval_batch=args.eval_batch, absence_ps=args.absence_ps, device=str(device)),
               scripts_sha=M.scripts_sha(SCRIPTS), utc_start=M.utc())
    if args.arm not in M.LATENT_ARMS:
        out.update(absence=None, readouts_incomplete=None, note='no readouts beyond VAL / fresh for this arm', utc_end=M.utc())
        M.jdump(out, os.path.join(src, 'readouts.json'))
        log('eval-only: nothing to recompute for this arm; readouts.json written')
        return 0
    train, val, fresh = read_data(args, log, need_train=args.tiny)
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in AGENT_KEYS}
    pair = M.build_pair(args.arm, args.pair, device, model=args.model, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=not cfg.get('no_grad_ckpt', False),
                        max_ctx=args.max_ctx_tokens, **agent_kw)
    missing = M.load_trainable_state(pair, ck['state'])
    if missing:
        raise SystemExit(f'{ckp}: {len(missing)} trainable parameters missing, e.g. {missing[:3]}')
    log(f'loaded {len(ck["state"])} tensors into {args.arm}/{args.pair} (w {pair.w_value}); load {time.time() - t0:.0f} s')
    absence, where = readouts_pair(pair, val, args, log, deadline)
    out.update(absence=absence, readouts_incomplete=where, w=pair.w_value, seconds=time.time() - t0, utc_end=M.utc())
    M.jdump(out, os.path.join(src, 'readouts.json'))
    log(f'eval-only done: readouts.json ({sorted(absence)}; incomplete: {where})')
    return 0


# ================================================================ --merge-b4: two per-model B4 runs -> one B4-<pair>-s<seed> summary (CPU, no models, no data)
def merge_b4(args):
    if not args.out:
        raise SystemExit('--merge-b4 needs --out DIR')
    parts = []
    for d in args.merge_b4:
        p = os.path.join(d, 'summary.json')
        if not os.path.isfile(p):
            raise SystemExit(f'{p}: missing')
        s = M.jload(p)
        if s.get('arm') != 'B4':
            raise SystemExit(f'{p}: arm {s.get("arm")} != B4')
        if not s.get('model') or list(s.get('models') or []) != [s['model']] or set(s['val'].get('em_model', {})) != {s['model']}:
            raise SystemExit(f'{p}: not a single-model B4 run (--model NAME): model {s.get("model")}, models {s.get("models")}, em_model {list(s["val"].get("em_model", {}))}')
        parts.append(dict(dir=os.path.abspath(d), summary_sha256=M.file_sha256(p), summary=s, model=s['model']))
    a, b = parts
    if a['model'] == b['model']:
        raise SystemExit(f'both parts are the model {a["model"]}: two different single models are needed')
    for k in ('pair', 'seed'):
        if a['summary'].get(k) != b['summary'].get(k):
            raise SystemExit(f'{k} differs between the parts: {a["summary"].get(k)} vs {b["summary"].get(k)}')
    if a['summary'].get('scripts_sha') != b['summary'].get('scripts_sha'):
        raise SystemExit(f'scripts_sha differ between the parts: {a["summary"].get("scripts_sha")} vs {b["summary"].get("scripts_sha")}')
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    names = [a['model'], b['model']]

    def blk(part, split):
        v = part['summary'][split]
        return (v.get('per_model') or {}).get(part['model']) or v
    v, f = merge_models(names, [blk(a, 'val'), blk(b, 'val')]), merge_models(names, [blk(a, 'fresh'), blk(b, 'fresh')])
    sa, sb = a['summary'], b['summary']
    summary = dict(arm='B4', pair=sa['pair'], seed=sa['seed'], model=None, models=names, merged=True, batch_name=M.BATCH, prereg=M.PREREG, val=v, fresh=f, absence=None,
                   complete=bool(sa.get('complete') and sb.get('complete')), steps=min(int(sa.get('steps', 0)), int(sb.get('steps', 0))),
                   steps_planned=[sa.get('steps_planned'), sb.get('steps_planned')], epoch_steps=EPOCH_STEPS, readouts_incomplete=None,
                   agents=list(sa.get('agents') or []) + list(sb.get('agents') or []), params={n: s.get('params') for n, s in zip(names, (sa, sb))},
                   train={n: s.get('train') for n, s in zip(names, (sa, sb))}, timings={n: s.get('timings') for n, s in zip(names, (sa, sb))},
                   config={'parts': {n: s.get('config') for n, s in zip(names, (sa, sb))}},
                   parts={p['model']: dict(dir=p['dir'], summary_sha256=p['summary_sha256']) for p in parts},
                   scripts_sha=sa['scripts_sha'], merge_script_sha256=M.file_sha256(os.path.abspath(__file__)), utc_end=M.utc(), pid=os.getpid())
    M.jdump(summary, sp)                                                  # no c_assert key (nothing is fused in B4)
    M.jdump(dict(arm='B4', pair=sa['pair'], seed=sa['seed'], models=names, parts=summary['parts'], merged_utc=summary['utc_end'],
                 merge_script_sha256=summary['merge_script_sha256']), os.path.join(args.out, 'merged_from.json'))
    print(f'B4 merge: {names} pair {sa["pair"]} seed {sa["seed"]} -> {sp}; VAL em_model {v["em_model"]}', flush=True)
    return 0


# ================================================================ a full run
def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.merge_b4:
        return merge_b4(args)
    if args.eval_only:
        return eval_only(args)
    for k in ('arm', 'pair', 'seed', 'train', 'val', 'fresh', 'out'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k} is required (unless --eval-only)')
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
    train, val, fresh = read_data(args, log)
    log(f'steps {args.steps} x batch {args.batch} = {args.steps * args.batch} examples ({args.steps * args.batch / max(1, len(train)):.2f} epochs)')
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in AGENT_KEYS}
    if args.arm == 'B4':
        if args.tiny:
            names = [args.model] if args.model else (list(M.TINY_NAMES) if args.pair == 'hetero' else [M.TINY_NAMES[0]])
        else:
            names = M.model_names(args.pair, 'B4', args.model)
    else:
        names = [None]
    ckp = os.path.join(args.out, 'trainable.pt')
    # ---------------------------------------------------------------- train -> save -> evaluate (B4: one Pair per model, sequentially)
    results, states, load_s = [], {}, 0.0
    try:
        for name in names:
            M.seed_all(args.seed)
            t0 = time.time()
            pair = M.build_pair(args.arm, args.pair, device, model=name, hf_cache=args.hf_cache, tiny=tiny, grad_ckpt=not args.no_grad_ckpt,
                                max_ctx=args.max_ctx_tokens, **agent_kw)
            load_s += time.time() - t0
            tag = f' [{name}]' if name else ''
            r = train_pair(pair, train, args, device, log, deadline, label=f'{args.arm}/{args.pair}/s{args.seed}{tag}', tag=tag)
            states[name or 'pair'] = r['state']
            torch.save(dict(arm=args.arm, pair=args.pair, seed=args.seed, models=[a['name'] for rr in results + [r] for a in rr['agents']],
                            state=states if args.arm == 'B4' else r['state'], agents=[a for rr in results + [r] for a in rr['agents']], config=vars(args)), ckp)
            log(f'trainable.pt saved before the evaluation{tag}')
            r.update(eval_pair(pair, val, fresh, args, device, log, deadline, tag=tag))
            results.append(r)
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
        steps_done = min(r['tr']['steps_done'] for r in results)
        step_times = {n: r['tr']['step_time_post_warmup_s'] for n, r in zip(models, results)}
    else:
        v, f = results[0]['val'], results[0]['fresh']
        trb = results[0]['tr']
        steps_done, step_times = results[0]['tr']['steps_done'], results[0]['tr']['step_time_post_warmup_s']
    where = [r['readouts_incomplete'] for r in results if r.get('readouts_incomplete')]
    summary = dict(arm=args.arm, pair=args.pair, seed=args.seed, model=args.model, models=models, batch_name=M.BATCH, prereg=M.PREREG, val=v, fresh=f,
                   absence=results[0]['absence'] if args.arm != 'B4' else None, complete=bool(steps_done == args.steps), steps=steps_done, steps_planned=args.steps,
                   epoch_steps=EPOCH_STEPS, readouts_incomplete=('; '.join(where) if where else None),
                   agents=[a for r in results for a in r['agents']], params=[r['params'] for r in results] if args.arm == 'B4' else results[0]['params'],
                   train=trb, timings=dict(load_s=load_s, train_s=sum(r['tr']['seconds'] for r in results), eval_s=sum(r['eval_s'] for r in results),
                                           total_s=time.time() - t_start, step_time_post_warmup_s=step_times, timing_warmup_steps=M.TIMING_WARMUP_STEPS),
                   config=dict(vars(args), k=args.k, d_e=args.d_e, log_lam_bounds=[M.LOG_LAM_MIN, M.LOG_LAM_MAX], lora_targets=list(M.LORA_TARGETS),
                               u_drop_p=M.U_DROP_P, edges=M.EDGES, wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__,
                               data_api='fallback' if M.DATA_IS_FALLBACK else 'l1_data', r2_rounds=M.R2_ROUNDS),
                   scripts_sha=M.scripts_sha(SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if args.arm in M.FUSION_ARMS:
        summary['c_assert'] = results[0]['c_assert']
        summary['w'] = v.get('w')
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l1_train done: {args.arm} {args.pair} s{args.seed} VAL EM {v["em"]:.2f} fresh EM {f["em"]:.2f} c_assert {summary.get("c_assert", "n/a")} '
        f'readouts {"complete" if not where else "INCOMPLETE (" + summary["readouts_incomplete"] + ")"} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
