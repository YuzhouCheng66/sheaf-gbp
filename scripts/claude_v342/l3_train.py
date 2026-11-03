"""l3_train.py (Claude, 2026-11-02): trainer of ladder L3 (pre-registration D8-93) on top of l3_model.py / l2_model.py.
Arms:  S2a  stage 3: the answer senders + lam heads + w2 + Decoder2 trained with the consensus CE vs the gold answer on top of the FROZEN V2
            solution (--v2-from = the V2 run dir); the round-1 cache (evidence tokens, round-1 answers, their mean token probability) is built
            for the training examples met and for VAL / fresh, and saved as round1_cache.json; readouts = eval_round2 with the V2 receiver.
       S2b  stage 4: the recv LoRA trained on the round-2 prompts (consensus slots); --init-from = the S2a run dir (its trainable.pt and cache);
            evaluation = greedy answers on the round-2 prompt (the criterion), baseline none (B0) for reference.
       T2   stage 4 with the text round-2 message (the partner's round-1 answer + its confidence); --init-from = the S2a run dir (cache only).
  python -B scripts/claude_v342/l3_train.py --arm S2a --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v2-from runs/l2/V2-same2-s342101 ...
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
import l2_model as L2  # noqa: E402
import l3_model as L3  # noqa: E402

SEEDS = T1.SEEDS
STAGE3_STEPS = 1500
STAGE4_STEPS = 1000


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=L3.ARMS)
    ap.add_argument('--pair', choices=('same2',))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (the frozen default LoRA); required unless --tiny')
    ap.add_argument('--v2-from', default=None, help='the V2 run dir (trainable.pt = the frozen round-1 solution incl. the recv adapters); required')
    ap.add_argument('--init-from', default=None, help='S2b / T2: the S2a run dir (trainable.pt + round1_cache.json)')
    ap.add_argument('--round1-cache', default=None, help='override the round-1 cache path')
    ap.add_argument('--stage3-steps', type=int, default=STAGE3_STEPS)
    ap.add_argument('--stage4-steps', type=int, default=STAGE4_STEPS)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr-adapter', type=float, default=1e-3)
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true')
    return ap


def _owner(blk):
    return [(k, round(x['em'], 2) if x['em'] is not None else None) for k, x in blk['em_by_owner'].items()]


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('arm', 'pair', 'seed', 'train', 'val', 'fresh', 'out', 'v2_from'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from is required outside --tiny')
    if args.arm in ('S2b', 'T2', 'S3b') and not args.init_from:
        raise SystemExit('--init-from (the S2a / S3a run dir) is required for S2b / S3b / T2')
    consensus = L3.ARM_CONSENSUS[args.arm]
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l3_train start pid {os.getpid()} host {platform.node()} arm {args.arm} pair {args.pair} seed {args.seed} out {args.out}; v2 {args.v2_from} init {args.init_from}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    train, val, fresh = T1.read_data(args, log)
    tiny = M.tiny_models(train + val + fresh, args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS if k != 'k'}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = L2.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        if receiver.get('arm') not in (None, 'B0') or (receiver.get('seed') not in (None, args.seed)):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        log(f'receiver (B0): {receiver}')
    v2_ckpt = os.path.join(args.v2_from, 'trainable.pt')
    v2_summary = M.jload(os.path.join(args.v2_from, 'summary.json')) if os.path.isfile(os.path.join(args.v2_from, 'summary.json')) else {}
    if v2_summary and int(v2_summary.get('seed', args.seed)) != args.seed:
        raise SystemExit(f'--v2-from seed {v2_summary.get("seed")} != {args.seed}')
    l3_sd, init_meta = None, None
    if args.init_from:
        ck3 = torch.load(os.path.join(args.init_from, 'trainable.pt'), map_location='cpu', weights_only=False)
        if ck3.get('arm') not in ('S2a', 'S3a') or int(ck3.get('seed', -1)) != args.seed:
            raise SystemExit(f'--init-from is not the S2a / S3a checkpoint of seed {args.seed} (arm {ck3.get("arm")}, seed {ck3.get("seed")})')
        if args.arm == 'S3b' and ck3.get('arm') != 'S3a':
            raise SystemExit('S3b needs an S3a checkpoint')
        l3_sd = ck3['state']
        init_meta = dict(arm=ck3.get('arm'), seed=ck3.get('seed'), sha256=M.file_sha256(os.path.join(args.init_from, 'trainable.pt')), v2=ck3.get('v2'))
        log(f'init (S2a): {init_meta}')
    cache_path = args.round1_cache or (os.path.join(args.init_from, 'round1_cache.json') if args.init_from else None)
    ckp = os.path.join(args.out, 'trainable.pt')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pl3, v2_meta = L3.build_l3(v2_ckpt, device, hf_cache=args.hf_cache, tiny=tiny, max_ctx=args.max_ctx_tokens, receiver_state=receiver_state, l3_state_dict=l3_sd, consensus=consensus, **agent_kw)
        load_s = time.time() - t0
        info = T1.agents_info(pl3.pair)
        log(f'model: {info}; v2 {v2_meta}; V2 VAL EM {v2_summary.get("val", {}).get("em")} (none {((v2_summary.get("baseline") or {}).get("none", {}).get("val") or {}).get("em")})')
        n_loaded = 0
        if cache_path and os.path.isfile(cache_path):
            n_loaded = pl3.load_cache(cache_path)
            log(f'round-1 cache loaded: {n_loaded} examples from {cache_path}')
        elif args.arm not in ('S2a', 'S3a'):
            raise SystemExit(f'round-1 cache missing: {cache_path}')
        stages = {}
        if args.arm in ('S2a', 'S3a'):
            L3.trainable_sets(pl3, 3)
            counts = M.count_params(pl3)
            log(f'stage 3 ({consensus}): trainable {counts["trainable"]:,} (answer senders + lam heads + w2 + Decoder2 + the calibration head)')
            tr = M.train_steps(pl3, lambda b: pl3.step3(b, train=True), M.batch_stream(train, args.batch, args.seed), args.stage3_steps, 0.0, args.lr_adapter,
                               args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/s{args.seed} stage3', weight_decay=args.weight_decay,
                               extra_log=lambda i: f'cons acc {i["msg_acc"]:.3f} ' + ('lam ' if consensus == 'logprob' else 'w2 ') + f'{i["w2"]:.3f}')
            tr['params'] = counts
            tr['round1_generated'] = pl3.n_round1
            stages['stage3'] = tr
            log(f'stage 3 done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; round-1 answers for {pl3.n_round1} examples')
            pl3.eval()
            for name, exs in (('val', val), ('fresh', fresh)):
                t1 = time.time()
                for batch in M.chunks(exs, args.eval_batch):
                    M.check_deadline(deadline, f'round-1 cache {name}')
                    pl3.round1(batch, args.max_new_tokens)
                log(f'round-1 cache for {name}: {len(exs)} examples in {time.time() - t1:.0f} s (total cached {len(pl3.cache)})')
            pl3.save_cache(os.path.join(args.out, 'round1_cache.json'))
        else:
            L3.trainable_sets(pl3, 4)
            counts = M.count_params(pl3)
            log(f'stage 4 ({args.arm}): trainable {counts["trainable"]:,} (recv LoRA)')
            tr = M.train_steps(pl3, lambda b: pl3.step4(b, train=True, arm=args.arm), M.batch_stream(train, args.batch, args.seed), args.stage4_steps, args.lr_lora, 0.0,   # the SAME stream as stage 3: every training example is in the S2a cache
                               args.warmup, args.clip, device, log, args.log_every, deadline, label=f'{args.arm}/s{args.seed} stage4', weight_decay=args.weight_decay)
            tr['params'] = counts
            tr['round1_generated'] = pl3.n_round1
            stages['stage4'] = tr
            log(f'stage 4 done: {tr["steps_done"]} steps in {tr["seconds"]:.0f} s; loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; round-1 cache misses {pl3.n_round1}')
            pl3.pair.use('recv')
        state = L3.l3_state(pl3)
        torch.save(dict(arm=args.arm, pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info, receiver=receiver, v2=v2_meta,
                        init_from=init_meta, config=vars(args)), ckp)
        log(f'trainable.pt saved before the evaluation ({len(state)} tensors)')
        # ---------------------------------------------------------------- evaluation
        t_eval = time.time()
        ekw = dict(eval_batch=args.eval_batch, max_new_tokens=args.max_new_tokens, deadline=deadline, keep_preds=True, log=log)
        arm_eval = 'T2' if args.arm == 'T2' else 'S2b'
        v = pl3.eval_round2(val, arm_eval, **ekw)
        log(f'VAL round-2 ({arm_eval}, recv{"" if args.arm not in ("S2a", "S3a") else " of V2 / V3, untrained on round 2"}): EM {v["em"]:.2f} F1 {v["f1"]:.2f} agents {v["em_agent"]} owner {_owner(v)}; '
            f'round-1 EM {v["round1"]["em_agent"]} either {v["round1"]["either_correct"]:.2f} both {v["round1"]["both_correct"]:.2f} p-AUROC {v["round1"]["p_auroc"]}'
            + (f'; consensus decoded EM {v["consensus"]["em_agent"]} lam-AUROC {v["consensus"]["lam_auroc_vs_round1_correct"]} w2 {v["consensus"]["w2"]:.3f} e.g. {v["consensus"]["examples"][:4]}' if 'consensus' in v else ''))
        f = pl3.eval_round2(fresh, arm_eval, **ekw)
        log(f'fresh round-2: EM {f["em"]:.2f} F1 {f["f1"]:.2f}; round-1 EM {f["round1"]["em_agent"]} either {f["round1"]["either_correct"]:.2f}')
        n0 = L2.eval_baseline(pl3.pair, val, 'none', **ekw)
        log(f'VAL baseline none (B0): EM {n0["em"]:.2f} owner {_owner(n0)}')
        if args.keep_preds:
            M.jdump(v.pop('records'), os.path.join(args.out, 'preds_val.json'))
            M.jdump(f.pop('records'), os.path.join(args.out, 'preds_fresh.json'))
            M.jdump(n0.pop('records'), os.path.join(args.out, 'preds_val_none.json'))
        else:
            for blk in (v, f, n0):
                blk.pop('records', None)
        eval_s = time.time() - t_eval
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for a in info))
    planned = args.stage3_steps if args.arm in ('S2a', 'S3a') else args.stage4_steps
    summary = dict(arm=args.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=L3.BATCH, prereg=L3.PREREG, val=v, fresh=f,
                   baseline=dict(none=dict(val=n0)), complete=bool(all(st['steps_done'] == planned for st in stages.values())), steps=sum(st['steps_done'] for st in stages.values()),
                   steps_planned=planned, stages=stages, agents=info, params=M.count_params(pl3), receiver=receiver, v2=dict(v2_meta, run=args.v2_from, val_em=(v2_summary.get('val') or {}).get('em'),
                   none_em=((v2_summary.get('baseline') or {}).get('none', {}).get('val') or {}).get('em')), init_from=init_meta, round1_cache=dict(path=cache_path, loaded=n_loaded, total=len(pl3.cache), generated=pl3.n_round1),
                   channel=dict(consensus=consensus, k_ans=pl3.k_ans, d_e=pl3.pair.agent(0).d_e, k_ev=pl3.pair.span_k, hint2_pre=L3.HINT2_PRE, hint2_post=L3.HINT2_POST, t2_pre=L3.T2_PRE, rendering='hard',
                                round1='frozen V2: span_sel evidence both ways, greedy answers with the V2 recv adapters',
                                round2=('precision-weighted consensus of the slot log-probabilities (calibrated lam head), argmax -> 8 hard tokens' if consensus == 'logprob' and args.arm != 'T2' else 'consensus of the aligned-read answer beliefs (lam heads, w2) decoded into 8 hard tokens' if args.arm != 'T2' else "the partner's round-1 answer as text with its mean token probability"),
                                w2=pl3.w2_value),
                   timings=dict(load_s=load_s, train_s=sum(st['seconds'] for st in stages.values()), eval_s=eval_s, total_s=time.time() - t_start),
                   config=dict(vars(args), lora_targets=list(M.LORA_TARGETS), edges=M.EDGES, wire_dtype=str(FU.WIRE_DTYPE), device=str(device), torch=torch.__version__),
                   scripts_sha=M.scripts_sha(L3.SCRIPTS), trainable_sha256=M.file_sha256(ckp), utc_end=M.utc(), pid=os.getpid())
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l3_train done: {args.arm} s{args.seed} VAL EM {v["em"]:.2f} (none {n0["em"]:.2f}; V2 {summary["v2"]["val_em"]}) fresh EM {f["em"]:.2f} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
