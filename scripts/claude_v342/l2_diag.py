"""l2_diag.py (Claude, 2026-11-01): E0 of the L2 ladder (pre-registration D8-92 section 3), no training, on the L1h Hu checkpoint:
  (a) own_hint        the partner's own B0 answer tokens in the hint template (the 100 %-copy ceiling of the frozen channel);
  (b) own_hint_oracle the same hint only where the partner's answer is right, else the plain prompt (the confidence-gating ceiling);
  (c) select_quality  the attention-selected 32-token window vs the oracle supporting-sentence span, per layer on TRAIN[:n_train] (the layer
                      with the best overlap is chosen), then reported on VAL[:n_val] at that layer.
Writes <out>/E0.json and <out>/log.txt; prints 'E0C_BEST_LAYER=<n>'. Reads only.
  python -B scripts/claude_v342/l2_diag.py --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --init-from runs/l1h/Hu-same2-s342101 \
      --train D/train.jsonl --val D/val.jsonl --hf-cache H --out runs/l2_diag --layers 6,9,12,15,18,21 --n-train 300 --n-val 500
"""
from __future__ import annotations
import argparse
import os
import sys
import time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch  # noqa: E402
import l1_model as M  # noqa: E402
import l2_model as L2  # noqa: E402


def _owner(blk):
    return [(k, round(x['em'], 2) if x['em'] is not None else None) for k, x in blk['em_by_owner'].items()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--receiver-from', required=True)
    ap.add_argument('--init-from', required=True)
    ap.add_argument('--train', required=True); ap.add_argument('--val', required=True)
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='6,9,12,15,18,21')
    ap.add_argument('--n-train', type=int, default=300); ap.add_argument('--n-val', type=int, default=500)
    ap.add_argument('--limit-eval', type=int, default=0, help='VAL examples for (a) / (b); 0 = all')
    ap.add_argument('--k', type=int, default=L2.K_HINT); ap.add_argument('--span-k', type=int, default=L2.K_SPAN)
    ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--eval-batch', type=int, default=32)
    ap.add_argument('--skip', default='', help='comma list of a,b,c to skip')
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = M.Log(os.path.join(args.out, 'log.txt'))
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    t0 = time.time()
    M.seed_all(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = M.DATA.read_jsonl(args.train)[:args.n_train]
    val_all = M.DATA.read_jsonl(args.val)
    val = val_all[:args.limit_eval] if args.limit_eval else val_all
    state, receiver = L2.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
    init, meta = L2.load_init(os.path.join(args.init_from, 'trainable.pt'))
    log(f'l2_diag start: device {dev} seed {args.seed} receiver {receiver.get("sha256", "")[:16]} init {meta} train {len(train)} val {len(val)} layers {args.layers}')
    pair = L2.build_pair('Hu', 'same2', dev, hf_cache=args.hf_cache, grad_ckpt=False, max_ctx=args.max_ctx_tokens, receiver_state=state, mode='answer',
                         span_k=args.span_k, init_state=init, recv=False, k=args.k, d_e=args.d_e, n_heads=args.n_heads, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.0, layer_frac=args.layer_frac)
    pair.eval()
    out = dict(seed=args.seed, receiver=receiver, init=meta, n_val=len(val), n_train=len(train), layers=[int(x) for x in args.layers.split(',')])
    ekw = dict(eval_batch=args.eval_batch, log=log)
    if 'a' not in skip:
        r = L2.eval_baseline(pair, val, 'own_hint', **ekw)
        out['own_hint'] = r
        log(f'E0a own_hint (100 % copy, frozen receiver): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
    if 'b' not in skip:
        r = L2.eval_baseline(pair, val, 'own_hint_oracle', **ekw)
        out['own_hint_oracle'] = r
        log(f'E0b own_hint_oracle (hint only when the partner is right): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
        r = L2.eval_baseline(pair, val, 'none', **ekw)
        out['none'] = r
        log(f'none (B0): EM {r["em"]:.2f} F1 {r["f1"]:.2f} owner {_owner(r)}')
    best = None
    if 'c' not in skip:
        q = L2.select_quality(pair, train, out['layers'], eval_batch=4, log=log)
        out['select_train'] = q
        best = q['best_layer']
        log(f'E0c TRAIN[:{len(train)}] overlap50 by layer {q["by_layer"]} oracle found {q["oracle_found"]:.3f} -> best layer {best}')
        if best is not None:
            qv = L2.select_quality(pair, val_all[:args.n_val], [best], eval_batch=4, log=log)
            out['select_val'] = qv
            log(f'E0c VAL[:{args.n_val}] at layer {best}: overlap50 {qv["by_layer"][str(best)]["overlap50"]:.3f} (n {qv["by_layer"][str(best)]["n"]}) oracle found {qv["oracle_found"]:.3f}')
    out['best_layer'] = best
    out['seconds'] = time.time() - t0
    out['utc'] = M.utc()
    M.jdump(out, os.path.join(args.out, 'E0.json'))
    log(f'l2_diag done in {out["seconds"]:.0f} s')
    print(f'E0C_BEST_LAYER={best}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
