"""l6_verifier.py (Claude, 2026-11-03): ladder L6 (pre-registration D8-96): TRAINED VERIFIER factors over the candidate union.
The one variable against D8-95 B1t: the per-agent factor on a whole-answer candidate c is no longer the generative sequence likelihood
l_i(c) but a trained verification log-odds s_i(c) = logit(' yes') - logit(' no') read at a verdict cue appended to the agent's OWN round-1
prompt + the candidate. Consensus = the sheaf-GBP sum on the discrete belief: S(c) = sum_i s_i(c), argmax ('ver_sum', the criterion).
  --mode gen    the B1t round-1 pipeline of l5_belief (frozen V1 pair, the partner's attention-selected span as clean tokens, beam-4
                candidates per agent) on train[:N] -> <out>/cands_train.json (candidates, union, evidence tokens, gold, owner) + summary.json
  --mode train  the LoRA 'verify' (one per agent, initialised from that agent's V1 recv adapter) trained with BCE on
                (round-1 prompt_i, c, EM(c, gold)) over the union of every training example (one epoch, soft time stop), saved to trainable.pt;
                then the evaluation on val / fresh with the candidates and unions of the L5 B1t records of the same seed (--l5-from:
                preds_val.json / preds_fresh.json of D8-95: the same union, the same ceiling), the evidence tokens recomputed by the same
                attention selection.  --swap-frac f (arm VERX): per agent and step, a fraction f of extra DOC-SWAPPED groups -- the documents and
                the received evidence of another training row with this row's question and candidates, every label 0 -- so that 'yes' needs
                document support, not recall (the V1 pair memorised its training examples: D8-96 appendix 1).
  python -B scripts/claude_v342/l6_verifier.py --mode gen --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 \
      --v-from runs/l2/V1-same2-s342101 --train hotpot/splits_l1/train.jsonl --limit-train 4000 --val ... --fresh ... --out runs/l6/GEN-same2-s342101
  python -B scripts/claude_v342/l6_verifier.py --mode train --pair same2 --seed 342101 --receiver-from ... --v-from ... --cands-from runs/l6/GEN-same2-s342101/cands_train.json \
      --l5-from runs/l5/B1t-same2-s342101 --train ... --val ... --fresh ... --out runs/l6/VER-same2-s342101 --train-stop-epoch <epoch>
"""
from __future__ import annotations
import argparse
import math
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
import torch.nn.functional as Fn  # noqa: E402
import l1_model as M  # noqa: E402
import l1_train as T1  # noqa: E402
import l2_model as L2  # noqa: E402
import l5_belief as L5  # noqa: E402

BATCH = 'L6'
PREREG = 'D8-96'
MODES = ('gen', 'train')
VERIFY_CUE = '\nIs the answer above correct? Reply yes or no.\nVerdict:'
YES, NO = ' yes', ' no'
YN_EXAMPLE = dict(question='yes no ?', answer='yes')                      # --tiny only: the verdict words must exist in the tiny vocabulary
VARIANTS = ('ver_sum', 'ver_self0', 'ver_self1', 'ver_select', 'ver_sum_p', 'ver_owner', 'select_p', 'poe_k4', 'agent0', 'agent1')
SCRIPTS = ('l6_verifier.py', 'l6_gate.py', 'l5_belief.py', 'l2_model.py', 'l1h_model.py', 'l1f_model.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py',
           'l1_train.py', 'l1_fusion.py', 'l1_data.py')
norm, em, union, owner_label, msg_bytes = L5.norm, L5.em, L5.union, L5.owner_label, L5.msg_bytes


# ================================================================ pure functions (tested on CPU)
def auroc(scores, labels):
    """rank AUROC (Mann-Whitney, ties averaged); None when one class is absent"""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    sv = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(x)))))


def argmax_first(vals):
    best, bi = None, None
    for i, v in enumerate(vals):
        if best is None or float(v) > best:
            best, bi = float(v), i
    return bi


def exclude_seen(examples, spec):
    """spec 'SEED:N,SEED:N' -> the examples NOT among the first N of M.epoch_order(len(examples), SEED) for any pair (the examples a training
    of N example-passes with that data seed never saw), in file order; -> (kept examples, meta)"""
    if not spec:
        return list(examples), dict(spec=None, n_total=len(examples), n_seen=0, n_kept=len(examples))
    seen = set()
    pairs = []
    for part in spec.split(','):
        sd, n = part.split(':')
        sd, n = int(sd), int(n)
        pairs.append((sd, n))
        seen.update(M.epoch_order(len(examples), sd)[:n])
    kept = [ex for j, ex in enumerate(examples) if j not in seen]
    return kept, dict(spec=spec, pairs=pairs, n_total=len(examples), n_seen=len(seen), n_kept=len(kept))


def rel_of(owner, i):
    """answer_owner label ('agent0' / 'agent1' / 'none') -> 'own' / 'other' / 'none' for agent i"""
    if owner in ('agent0', 'agent1'):
        return 'own' if int(owner[-1]) == i else 'other'
    return 'none'


def judge(ex, cands, C, S, G, p):
    """one example. cands: per agent the candidate strings (top-1 first); C: the union; S[i][ci] = the verifier log-odds s_i(c);
    G[i][ci] = the generative sequence log-probability of D8-95 (the control); p[i] = the mean token probability of agent i's top-1."""
    gold = ex['answer']
    n_ag = len(cands)
    top = [cands[i][0] if cands[i] else '' for i in range(n_ag)]
    idx = {norm(c): ci for ci, c in enumerate(C)}
    top_idx = [idx[norm(t)] for t in top]
    n = len(C)
    ssum = [sum(float(S[i][ci]) for i in range(n_ag)) for ci in range(n)]
    picks = {}
    picks['ver_sum'] = argmax_first(ssum)
    for i in range(n_ag):
        picks[f'ver_self{i}'] = argmax_first([float(S[i][ci]) for ci in range(n)])
    picks['ver_select'] = argmax_first([max(float(S[i][ci]) for i in range(n_ag)) for ci in range(n)])
    picks['ver_sum_p'] = argmax_first([sum(float(p[i]) * float(S[i][ci]) for i in range(n_ag)) for ci in range(n)])
    owner = owner_label(ex)
    picks['ver_owner'] = picks[f'ver_self{int(owner[-1])}'] if owner in ('agent0', 'agent1') else picks['ver_sum']
    picks['select_p'] = top_idx[0 if p[0] >= p[1] else 1]
    picks['poe_k4'] = argmax_first([sum(float(G[i][ci]) for i in range(n_ag)) for ci in range(n)])
    picks['agent0'], picks['agent1'] = top_idx[0], top_idx[1]
    ok = {k: em(C[ci], gold) for k, ci in picks.items()}
    e0, e1 = ok['agent0'], ok['agent1']
    agree = norm(top[0]) == norm(top[1])
    labels = [em(c, gold) for c in C]
    any_union = max(labels) if labels else 0.0
    any_agent = [max((em(c, gold) for c in cands[i]), default=0.0) for i in range(n_ag)]
    asym = None
    if not agree and (e0 + e1) == 1.0:
        a = 0 if e0 == 1.0 else 1
        w = 1 - a
        X, Y = top_idx[a], top_idx[w]
        asym = dict(knower=a, m_know=float(S[a][X] - S[a][Y]), m_ign=float(S[w][Y] - S[w][X]), ver_sum_correct=ok['ver_sum'], select_correct=ok['select_p'])
    cand_recs = [dict(agent=i, rel=rel_of(owner, i), s=float(S[i][ci]), label=labels[ci]) for i in range(n_ag) for ci in range(n)]
    return dict(picks=picks, ok=ok, either=max(e0, e1), both=min(e0, e1), agree=float(agree), agree_ok=(e0 if agree else None), any_union=any_union, any_agent=any_agent,
                fused=ssum, asym=asym, top=top, top_idx=top_idx, owner=owner, cand_recs=cand_recs)


def aggregate(rows):
    """rows: judge() outputs (+ 'bytes') -> the split readouts (percent)"""
    n = len(rows)
    pct = lambda xs: (100.0 * float(np.mean(xs))) if len(xs) else None       # noqa: E731
    out = dict(n=n)
    out['em_variants'] = {k: pct([r['ok'][k] for r in rows]) for k in VARIANTS}
    out['em'] = out['em_variants']['ver_sum']
    agree_rows = [r for r in rows if r['agree'] == 1.0]
    dis = [r for r in rows if r['agree'] != 1.0]
    out['round1'] = dict(em_agent=[out['em_variants']['agent0'], out['em_variants']['agent1']], either_correct=pct([r['either'] for r in rows]), both_correct=pct([r['both'] for r in rows]),
                         agree=pct([r['agree'] for r in rows]), agree_acc=pct([r['agree_ok'] for r in agree_rows]), select_p=out['em_variants']['select_p'],
                         disagree=dict(n=len(dis), select_p_correct=pct([r['ok']['select_p'] for r in dis]), ver_sum_correct=pct([r['ok']['ver_sum'] for r in dis]),
                                       poe_k4_correct=pct([r['ok']['poe_k4'] for r in dis]), either_correct=pct([r['either'] for r in dis])))
    out['union'] = dict(any_correct=pct([r['any_union'] for r in rows]), any_agent=[pct([r['any_agent'][i] for r in rows]) for i in range(2)],
                        size_mean=float(np.mean([len(r['fused']) for r in rows])) if n else None)
    asym = [r['asym'] for r in rows if r['asym'] is not None]
    if asym:
        out['asym'] = dict(n=len(asym), m_know_mean=float(np.mean([a['m_know'] for a in asym])), m_know_median=float(np.median([a['m_know'] for a in asym])),
                           m_ign_mean=float(np.mean([a['m_ign'] for a in asym])), m_ign_median=float(np.median([a['m_ign'] for a in asym])),
                           know_wins=pct([float(a['m_know'] > a['m_ign']) for a in asym]), ver_sum_correct=pct([a['ver_sum_correct'] for a in asym]),
                           select_correct=pct([a['select_correct'] for a in asym]))
    else:
        out['asym'] = dict(n=0)
    owners = sorted({r['owner'] for r in rows})
    out['em_by_owner'] = {o: dict(n=len([r for r in rows if r['owner'] == o]), **{k: pct([r['ok'][k] for r in rows if r['owner'] == o]) for k in ('ver_sum', 'select_p', 'poe_k4', 'ver_owner')})
                          for o in owners}
    cr = [c for r in rows for c in r['cand_recs']]
    ver = {}
    for rel in ('own', 'other', 'none', 'all'):
        sub = [c for c in cr if rel == 'all' or c['rel'] == rel]
        ver[rel] = dict(n=len(sub), auroc=auroc([c['s'] for c in sub], [c['label'] for c in sub]) if sub else None,
                        abs_s_median=float(np.median([abs(c['s']) for c in sub])) if sub else None,
                        p_yes_pos=(float(np.mean([sigmoid(c['s']) for c in sub if c['label'] == 1.0])) if any(c['label'] == 1.0 for c in sub) else None),
                        p_yes_neg=(float(np.mean([sigmoid(c['s']) for c in sub if c['label'] == 0.0])) if any(c['label'] == 0.0 for c in sub) else None))
    flat = None
    if ver['own']['n'] and ver['other']['n']:
        flat = bool(ver['other']['abs_s_median'] < ver['own']['abs_s_median'])
    out['verifier'] = dict(by_rel=ver, positive_rate=pct([c['label'] for c in cr]), flat_when_ignorant=flat)
    if n and 'bytes' in rows[0]:
        out['bytes'] = {k: float(np.mean([r['bytes'][k] for r in rows])) for k in rows[0]['bytes']}
    return out


# ================================================================ the models
def build_pair_v1(v_ckpt, device, hf_cache=None, tiny=None, max_ctx=None, receiver_state=None, sel_layer=None, grad_ckpt=False, **agent_kw):
    """the frozen V1 pair (B0 default adapters + the V1 recv adapters) in span_attn mode = l5_belief.build_round1_pair('B1t'), with the
    gradient-checkpointing switch the verifier training needs"""
    ck = torch.load(v_ckpt, map_location='cpu', weights_only=False)
    if ck.get('arm') != 'V1' or not ck.get('recv'):
        raise ValueError(f'{v_ckpt}: a V1 checkpoint with recv adapters is needed (arm {ck.get("arm")}, recv {ck.get("recv")})')
    init = {k: v for k, v in ck['state'].items() if '.adapter.' in k or '.reader.' in k or '.selector.' in k}
    kw = dict(agent_kw)
    kw.setdefault('k', int(ck['config'].get('k', L2.K_SPAN)))
    pair = L2.build_pair('V1', 'same2', device, hf_cache=hf_cache, tiny=tiny, grad_ckpt=grad_ckpt, max_ctx=max_ctx, receiver_state=receiver_state, mode='span_oracle',
                         span_k=kw['k'], init_state=init, recv=True, sel_layer=sel_layer, **kw)
    recv = {k: v for k, v in ck['state'].items() if 'lora_' in k and '.recv.' in k}
    own = [n for n, _ in pair.named_parameters() if 'lora_' in n and '.recv.' in n]
    missing = sorted(set(own) - set(recv))
    if not recv or missing:
        raise ValueError(f'{v_ckpt}: recv tensors {len(recv)} cover {len(own) - len(missing)} of {len(own)}; e.g. missing {missing[:3]}')
    M.load_trainable_state(pair, {k: recv[k] for k in own}, strict=True)
    pair.set_mode('span_attn')
    if any(a.sel_layer is None for a in pair.agents):
        raise ValueError('span_attn needs sel_layer (the attention layer of the V1 span selection)')
    for p_ in pair.parameters():
        p_.requires_grad_(False)
    pair.eval()
    pair.use('recv')
    return pair, dict(v_arm=ck.get('arm'), v_seed=ck.get('seed'), v_sha256=M.file_sha256(v_ckpt), v_models=ck.get('models'), k=kw['k'])


def enable_verify(pair):
    """add the LoRA 'verify' to every agent, initialised from that agent's recv adapter -> tensors copied per agent"""
    counts = []
    for ag in pair.agents:
        if getattr(ag, '_lora_verify', None):
            counts.append(len(ag._lora_verify))
            continue
        ag.model.add_adapter('verify', ag.model.peft_config['default'])
        params = dict(ag.model.named_parameters())
        n_copied = 0
        for n, p_ in params.items():
            if 'lora_' in n and '.recv.' in n:
                q = params[n.replace('.recv.', '.verify.')]
                if q.shape != p_.shape:
                    raise RuntimeError(f'{ag.name}: verify tensor {n} shape {tuple(q.shape)} != {tuple(p_.shape)}')
                with torch.no_grad():
                    q.copy_(p_)
                n_copied += 1
        ag._lora_verify = [p_ for n, p_ in params.items() if 'lora_' in n and '.verify.' in n]
        if not ag._lora_verify or n_copied != len(ag._lora_verify):
            raise RuntimeError(f'{ag.name}: verify adapter has {len(ag._lora_verify)} tensors, {n_copied} copied')
        counts.append(n_copied)
    return counts


def use_verify(pair, train=False):
    """activate 'verify' on both agents; only the verify tensors may require gradients (and only when train)"""
    for ag in pair.agents:
        ag.use('verify')
        for p_ in ag.parameters():
            p_.requires_grad_(False)
        for p_ in ag._lora_verify:
            p_.requires_grad_(bool(train))


def verify_state(pair):
    return {n: p_.detach().cpu() for n, p_ in pair.named_parameters() if 'lora_' in n and '.verify.' in n}


def load_verify_state(pair, state):
    own = [n for n, _ in pair.named_parameters() if 'lora_' in n and '.verify.' in n]
    missing = sorted(set(own) - set(state))
    if missing:
        raise ValueError(f'verify state covers {len(own) - len(missing)} of {len(own)} tensors; e.g. missing {missing[:3]}')
    M.load_trainable_state(pair, {k: state[k] for k in own}, strict=True)


def yes_no_ids(ag):
    y, n_ = ag.ids(YES), ag.ids(NO)
    if not y or not n_:
        raise ValueError(f'{ag.name}: the verdict words have no tokens')
    return int(y[0]), int(n_[0])


def round1_prompts(pair, batch, i, evs, docs_from=None):
    """-> the round-1 prompt id lists of agent i: [BOS] documents | SPAN template with the partner's evidence tokens | question.
    docs_from (examples, same length): the documents (and evs) come from these examples instead, the question stays (doc-swapped negatives)"""
    ag = pair.agent(i)
    pcs, _dl = pair.doc_pieces(batch, i)
    if docs_from is not None:
        other, _ = pair.doc_pieces(docs_from, i)
        pcs = [(o[0], p_[1], []) for o, p_ in zip(other, pcs)]
    return [list(p_) + ag.hint_pre + list(evs[b]) + ag.hint_post + list(s) for b, (p_, s, _) in enumerate(pcs)]


def evidence_for(pair, examples, batch=8, deadline=None, log=None):
    """the partner's attention-selected span tokens for every example and agent (the B0 adapter), as {id: [ev_0, ev_1]}; the adapters are
    left on 'default' (the caller re-activates what it needs)"""
    out = {}
    for bi, b in enumerate(M.chunks(examples, batch)):
        M.check_deadline(deadline, f'evidence batch {bi}')
        evs = [L5.evidence_ids(pair, b, i) for i in range(M.N_AGENTS)]
        for j, ex in enumerate(b):
            out[ex['id']] = [evs[i][j] for i in range(M.N_AGENTS)]
    return out


def verify_pieces(ag, prompt, cand, cue):
    """(prompt, [], ' ' + candidate + verdict cue): the verdict logits are read at the last position"""
    return (list(prompt), [], ([int(t) for t in ag.ids(' ' + cand)] if cand else []) + list(cue))


def verify_logits(ag, pieces, yn, train=False):
    """-> s = logit(yes) - logit(no) at the last real position of every row (B,) float32"""
    A = ag.assemble(pieces, None, 'right')
    emb = A['embeds']
    if train and not emb.requires_grad:
        emb.requires_grad_(True)
    h = ag.decoder(inputs_embeds=emb, attention_mask=A['mask'], use_cache=False).last_hidden_state
    pos = torch.tensor([A['ans_start'][b] + len(a) - 1 for b, (_p, _s, a) in enumerate(pieces)], dtype=torch.long, device=h.device)
    hs = h[torch.arange(len(pieces), device=h.device), pos]
    logits = ag.lm_head(hs).float()
    return logits[:, yn[0]] - logits[:, yn[1]]


@torch.no_grad()
def score_verify(ag, prompts, cands, cue, yn, batch_size=32):
    ag.eval()
    out = []
    for lo in range(0, len(prompts), batch_size):
        pcs = [verify_pieces(ag, p_, c, cue) for p_, c in zip(prompts[lo:lo + batch_size], cands[lo:lo + batch_size])]
        out.extend(float(x) for x in verify_logits(ag, pcs, yn, train=False).tolist())
    return out


# ================================================================ --mode gen
@torch.no_grad()
def gen_rows(pair, examples, k, max_new, gen_batch, deadline=None, log=None):
    rows = []
    t_gen = t_ev = 0.0
    for bi, batch in enumerate(M.chunks(examples, gen_batch)):
        M.check_deadline(deadline, f'gen batch {bi}')
        t0 = time.time()
        evs = [L5.evidence_ids(pair, batch, i) for i in range(M.N_AGENTS)]
        r1 = [[(pr, []) for pr in round1_prompts(pair, batch, i, evs[i])] for i in range(M.N_AGENTS)]
        t_ev += time.time() - t0
        t0 = time.time()
        cands = [L5.beam_candidates(pair.agent(i), r1[i], k, max_new) for i in range(M.N_AGENTS)]
        t_gen += time.time() - t0
        for b, ex in enumerate(batch):
            cb = [cands[i][b] for i in range(M.N_AGENTS)]
            rows.append(dict(id=ex['id'], gold=ex['answer'], owner=owner_label(ex), cands=cb, union=union(cb), ev=[[int(t) for t in evs[i][b]] for i in range(M.N_AGENTS)]))
        if log and (bi + 1) % 25 == 0:
            log(f'gen: {len(rows)} examples; s/example evidence {t_ev / len(rows):.3f} beams {t_gen / len(rows):.3f}')
    n = max(1, len(rows))
    stats = dict(n=len(rows), union_any_correct=100.0 * float(np.mean([max(em(c, r['gold']) for c in r['union']) for r in rows])) if rows else None,
                 either_top1=100.0 * float(np.mean([max(em(r['cands'][i][0], r['gold']) for i in range(M.N_AGENTS)) for r in rows])) if rows else None,
                 size_mean=float(np.mean([len(r['union']) for r in rows])) if rows else None,
                 positive_rate=100.0 * float(np.mean([em(c, r['gold']) for r in rows for c in r['union']])) if rows else None,
                 timing=dict(evidence_s_per_example=t_ev / n, gen_s_per_example=t_gen / n))
    return rows, stats


# ================================================================ --mode train
def train_verifier(pair, rows, by_id, cue, yns, args, device, log, deadline=None, soft_stop=None):
    """one epoch (or --steps) of BCE on the verdict log-odds; -> the training block (steps_done, planned, losses, timing)"""
    use_verify(pair, train=True)
    pair.train()
    n = len(rows)
    planned = int(args.steps) if args.steps else int(math.ceil(n / args.batch))
    order = M.epoch_order(n, args.seed)
    opt, sched, counts = M.build_optimizer(pair, args.lr_lora, args.lr_adapter, planned, args.warmup, args.weight_decay)
    params = [p_ for g in opt.param_groups for p_ in g['params']]
    if counts['adapter'] != 0:
        raise RuntimeError(f'only the verify LoRA may train (adapter params {counts["adapter"]})')
    log(f'train: {planned} steps x {args.batch} examples (n {n}), lr {args.lr_lora} ({counts["lora"]:,} verify params), warm-up {args.warmup}, clip {args.clip}, soft stop {soft_stop}')
    use_ac = device.type == 'cuda'
    losses, hist, times = [], [], []
    acc_pos, acc_neg, n_seq, n_swap = [], [], 0, 0
    swap_frac = float(getattr(args, 'swap_frac', 0.0) or 0.0)
    rng = np.random.RandomState(args.seed + 7)
    t_start = time.time()
    done = 0
    stopped = None
    for s in range(planned):
        M.check_deadline(deadline, f'train step {s}')
        if soft_stop and time.time() >= soft_stop:
            stopped = f'soft time stop at step {s}/{planned}'
            break
        lo = (s * args.batch) % n
        idx = [order[(lo + j) % n] for j in range(args.batch)]
        batch_rows = [rows[j] for j in idx]
        batch = [by_id[r['id']] for r in batch_rows]
        M.sync(device)
        t0 = time.perf_counter()
        loss_agents = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
            for i in range(M.N_AGENTS):
                ag = pair.agent(i)
                prompts = round1_prompts(pair, batch, i, [r['ev'][i] for r in batch_rows])
                pcs, lab = [], []
                for b, r in enumerate(batch_rows):
                    for c in r['union']:
                        pcs.append(verify_pieces(ag, prompts[b], c, cue))
                        lab.append(em(c, r['gold']))
                sc = verify_logits(ag, pcs, yns[i], train=True)
                y = torch.tensor(lab, dtype=torch.float32, device=sc.device)
                li = Fn.binary_cross_entropy_with_logits(sc, y, reduction='sum')
                n_i = len(lab)
                with torch.no_grad():
                    pr = torch.sigmoid(sc.detach())
                    acc_pos.extend(pr[y == 1].tolist()); acc_neg.extend(pr[y == 0].tolist()); n_seq += len(lab)
                swap = [b for b in range(len(batch_rows)) if swap_frac > 0 and rng.rand() < swap_frac]
                if swap:                                                       # doc-swapped negatives: another row's documents + evidence, this question, label 0
                    src = [rows[int(rng.randint(n))] for _ in swap]
                    src = [r2 if r2['id'] != batch_rows[b]['id'] else rows[(order[0] + 1) % n] for r2, b in zip(src, swap)]
                    sp = round1_prompts(pair, [batch[b] for b in swap], i, [r2['ev'][i] for r2 in src], docs_from=[by_id[r2['id']] for r2 in src])
                    pcs2 = [verify_pieces(ag, sp[q], c, cue) for q, b in enumerate(swap) for c in batch_rows[b]['union']]
                    sc2 = verify_logits(ag, pcs2, yns[i], train=True)
                    li = li + Fn.binary_cross_entropy_with_logits(sc2, torch.zeros_like(sc2), reduction='sum')
                    n_i += len(pcs2); n_seq += len(pcs2); n_swap += len(pcs2)
                    with torch.no_grad():
                        acc_neg.extend(torch.sigmoid(sc2.detach()).tolist())
                loss_agents.append(li / max(1, n_i))
            loss = sum(loss_agents) / len(loss_agents)
        if not torch.isfinite(loss):
            raise RuntimeError(f'train: non-finite loss at step {s}')
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        M.sync(device)
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
        done = s + 1
        if done % args.log_every == 0 or done == planned:
            mem = f', peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GB' if use_ac else ''
            log(f'train step {done}/{planned} loss {np.mean(losses[-args.log_every:]):.4f} agents {["%.3f" % float(x.detach()) for x in loss_agents]} lr {sched.get_last_lr()[0]:.2e} gnorm {float(gn):.2f} '
                f'p_yes pos {np.mean(acc_pos[-500:]) if acc_pos else float("nan"):.3f} neg {np.mean(acc_neg[-2000:]) if acc_neg else float("nan"):.3f} seq {n_seq} {np.mean(times[-args.log_every:]):.3f} s/step{mem}')
            hist.append(dict(step=done, loss=float(np.mean(losses[-args.log_every:])), lr=float(sched.get_last_lr()[0])))
    if stopped:
        log(f'train: {stopped}')
    pair.eval()
    use_verify(pair, train=False)
    return dict(steps_done=done, steps_planned=planned, examples=n, batch=args.batch, seconds=time.time() - t_start, step_time_s=float(np.mean(times)) if times else None,
                sequences=n_seq, swapped_sequences=n_swap, swap_frac=swap_frac, loss_first100=float(np.mean(losses[:100])) if losses else None, loss_last100=float(np.mean(losses[-100:])) if losses else None, history=hist,
                stopped=stopped, param_counts=counts)


@torch.no_grad()
def eval_split(pair, examples, recs, evs, cue, yns, score_batch, deadline=None, log=None, label=''):
    """examples with the L5 records (candidates, union, generative scores, p) and the evidence tokens -> (readouts, records)"""
    pair.eval()
    use_verify(pair, train=False)
    rows, records = [], []
    t_sc = 0.0
    for bi, batch in enumerate(M.chunks(examples, 8)):
        M.check_deadline(deadline, f'l6 {label} batch {bi}')
        t0 = time.time()
        S = [[None] * len(batch) for _ in range(M.N_AGENTS)]
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            prompts = round1_prompts(pair, batch, i, [evs[ex['id']][i] for ex in batch])
            P, Cc, where = [], [], []
            for b, ex in enumerate(batch):
                for ci, c in enumerate(recs[ex['id']]['union']):
                    P.append(prompts[b]); Cc.append(c); where.append((b, ci))
            got = score_verify(ag, P, Cc, cue, yns[i], score_batch)
            for (b, ci), sc in zip(where, got):
                if S[i][b] is None:
                    S[i][b] = [None] * len(recs[batch[b]['id']]['union'])
                S[i][b][ci] = sc
        t_sc += time.time() - t0
        for b, ex in enumerate(batch):
            r = recs[ex['id']]
            j = judge(ex, r['cands'], r['union'], [S[i][b] for i in range(M.N_AGENTS)], r['scores'], r['p'])
            j['bytes'] = msg_bytes(r['cands'], r['union'])
            rows.append(j)
            records.append(dict(id=ex['id'], gold=ex['answer'], owner=j['owner'], cands=r['cands'], union=r['union'], p=r['p'], gen_scores=r['scores'],
                                ver_scores=[[round(x, 4) for x in S[i][b]] for i in range(M.N_AGENTS)], picks={kk: r['union'][ci] for kk, ci in j['picks'].items()}, ok=j['ok']))
        if log and (bi + 1) % 25 == 0:
            log(f'l6 {label}: {len(rows)} examples; ver_sum {100.0 * np.mean([r["ok"]["ver_sum"] for r in rows]):.2f} select_p {100.0 * np.mean([r["ok"]["select_p"] for r in rows]):.2f}; '
                f'verify s/example {t_sc / len(rows):.3f}')
    out = aggregate(rows)
    out['timing'] = dict(verify_s_per_example=t_sc / max(1, len(rows)), round2_s_per_example=t_sc / max(1, len(rows)))
    out['examples'] = [dict(gold=r['gold'], top=[c[0] for c in r['cands']], ver_sum=r['picks']['ver_sum'], s=r['ver_scores']) for r in records[:8]]
    return out, records


def load_l5_records(path, examples):
    recs = {r['id']: r for r in M.jload(path)}
    miss = [ex['id'] for ex in examples if ex['id'] not in recs]
    if miss:
        raise SystemExit(f'{path}: {len(miss)} examples missing, e.g. {miss[:3]}')
    for ex in examples:
        r = recs[ex['id']]
        for k in ('cands', 'union', 'scores', 'p'):
            if k not in r:
                raise SystemExit(f'{path}: record {ex["id"]} has no {k}')
        if norm(r['gold']) != norm(ex['answer']):
            raise SystemExit(f'{path}: record {ex["id"]} gold {r["gold"]!r} != {ex["answer"]!r}')
    return recs


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--pair', choices=('same2',))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (the frozen default LoRA); required unless --tiny')
    ap.add_argument('--v-from', default=None, help='the V1 run dir of this seed (trainable.pt with the recv adapters); required')
    ap.add_argument('--cands-from', default=None, help='train: the cands_train.json of the GEN job of this seed')
    ap.add_argument('--l5-from', default=None, help='train: the L5 B1t run dir of this seed (preds_val.json / preds_fresh.json / summary.json)')
    ap.add_argument('--sel-layer', type=int, default=None, help='the attention layer of the span selection (default: span.sel_layer of the V1 summary)')
    ap.add_argument('--k-beams', type=int, default=L5.K_BEAMS)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--gen-batch', type=int, default=8)
    ap.add_argument('--score-batch', type=int, default=32)
    ap.add_argument('--batch', type=int, default=2, help='train: examples per step (every union candidate of every agent)')
    ap.add_argument('--steps', type=int, default=0, help='train: 0 = one epoch over the training rows')
    ap.add_argument('--lr-lora', type=float, default=1e-4)
    ap.add_argument('--lr-adapter', type=float, default=1e-3, help='unused (no adapter params train)')
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--train-stop-epoch', type=float, default=0.0, help='train: soft stop of the training (epoch seconds); the evaluation follows')
    ap.add_argument('--swap-frac', type=float, default=0.0, help='train: fraction of extra doc-swapped negative groups per agent and example (arm VERX when > 0)')
    ap.add_argument('--no-grad-ckpt', action='store_true')
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--limit-eval', type=int, default=0); ap.add_argument('--limit-train', type=int, default=0)
    ap.add_argument('--exclude-seen', default=None, help="gen: 'SEED:N,...' drops the train examples among the first N of the epoch order of SEED (seen by a training with that data seed)")
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('mode', 'pair', 'seed', 'train', 'val', 'fresh', 'out', 'v_from'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from is required outside --tiny')
    if args.mode == 'train' and not (args.cands_from and args.l5_from):
        raise SystemExit('--cands-from and --l5-from are required for --mode train')
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l6_verifier start pid {os.getpid()} host {platform.node()} mode {args.mode} pair {args.pair} seed {args.seed} out {args.out}; v {args.v_from} cands {args.cands_from} l5 {args.l5_from}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    seen_meta = None
    if args.mode == 'gen':
        full = argparse.Namespace(**dict(vars(args), limit_train=0))
        train, val, fresh = T1.read_data(full, log, need_train=True)
        train, seen_meta = exclude_seen(train, args.exclude_seen)
        if args.limit_train:
            train = train[:args.limit_train]
        log(f'gen rows: {len(train)} train examples after exclude-seen {seen_meta} and limit {args.limit_train}')
    else:
        train, val, fresh = T1.read_data(args, log, need_train=False)
    tiny = M.tiny_models(train + val + fresh + [YN_EXAMPLE], args.seed) if args.tiny else None
    agent_kw = {k: getattr(args, k) for k in T1.AGENT_KEYS if k != 'k'}
    receiver_state, receiver = None, None
    if args.receiver_from:
        receiver_state, receiver = L2.load_receiver(os.path.join(args.receiver_from, 'trainable.pt'))
        if receiver.get('arm') not in (None, 'B0') or (receiver.get('seed') not in (None, args.seed)):
            raise SystemExit(f'--receiver-from is not the B0 checkpoint of seed {args.seed}: {receiver}')
        log(f'receiver (B0): {receiver}')
    v_ckpt = os.path.join(args.v_from, 'trainable.pt')
    v_summary = M.jload(os.path.join(args.v_from, 'summary.json')) if os.path.isfile(os.path.join(args.v_from, 'summary.json')) else {}
    if v_summary and int(v_summary.get('seed', args.seed)) != args.seed:
        raise SystemExit(f'--v-from seed {v_summary.get("seed")} != {args.seed}')
    sel_layer = args.sel_layer
    if sel_layer is None:
        sel_layer = (v_summary.get('span') or {}).get('sel_layer')
        if sel_layer is None:
            raise SystemExit('--sel-layer is required (the V1 summary has no span.sel_layer)')
        sel_layer = int(sel_layer)
    cands_meta, rows, recs_val, recs_fresh, l5_summary = None, None, None, None, None
    if args.mode == 'train':
        d = M.jload(args.cands_from)
        if int(d.get('seed', -1)) != args.seed:
            raise SystemExit(f'--cands-from seed {d.get("seed")} != {args.seed}')
        rows = d['rows'][:args.limit_train] if args.limit_train else d['rows']
        cands_meta = dict(path=args.cands_from, n_rows=len(rows), n_total=len(d['rows']), sha256=M.file_sha256(args.cands_from), v=d.get('v'))
        recs_val = load_l5_records(os.path.join(args.l5_from, 'preds_val.json'), val)
        recs_fresh = load_l5_records(os.path.join(args.l5_from, 'preds_fresh.json'), fresh)
        l5_summary = M.jload(os.path.join(args.l5_from, 'summary.json')) if os.path.isfile(os.path.join(args.l5_from, 'summary.json')) else {}
        if l5_summary and (l5_summary.get('arm') != 'B1t' or int(l5_summary.get('seed', -1)) != args.seed):
            raise SystemExit(f'--l5-from is not the B1t run of seed {args.seed}: arm {l5_summary.get("arm")} seed {l5_summary.get("seed")}')
        log(f'training rows {len(rows)} (of {len(d["rows"])}); L5 records val {len(recs_val)} fresh {len(recs_fresh)}; L5 select_p {((l5_summary.get("val") or {}).get("em_variants") or {}).get("select_p")}')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair, v_meta = build_pair_v1(v_ckpt, device, hf_cache=args.hf_cache, tiny=tiny, max_ctx=args.max_ctx_tokens, receiver_state=receiver_state, sel_layer=sel_layer,
                                     grad_ckpt=(args.mode == 'train' and not args.no_grad_ckpt), **agent_kw)
        load_s = time.time() - t0
        info = T1.agents_info(pair)
        log(f'model: {info}; v {v_meta}; V1 VAL EM {v_summary.get("val", {}).get("em")}; mode {pair.mode} k_ev {pair.span_k} sel_layer {sel_layer}')
        if args.mode == 'gen':
            t1 = time.time()
            rows, stats = gen_rows(pair, train, args.k_beams, args.max_new_tokens, args.gen_batch, deadline=deadline, log=log)
            gen_s = time.time() - t1
            M.jdump(dict(seed=args.seed, pair=args.pair, n=len(rows), k_beams=args.k_beams, max_new=args.max_new_tokens, sel_layer=sel_layer, k_ev=pair.span_k, v=v_meta, split='train',
                         limit_train=args.limit_train, exclude_seen=seen_meta, rows=rows), os.path.join(args.out, 'cands_train.json'))
            log(f'gen: {stats}')
            summary = dict(mode='gen', arm='GEN', pair=args.pair, seed=args.seed, batch_name=BATCH, prereg=PREREG, complete=True, n_rows=len(rows), gen=stats, exclude_seen=seen_meta, agents=info, receiver=receiver,
                           v=dict(v_meta, run=args.v_from, val_em=(v_summary.get('val') or {}).get('em'), sel_layer=sel_layer), timings=dict(load_s=load_s, gen_s=gen_s, total_s=time.time() - t_start),
                           config=dict(vars(args), device=str(device), torch=torch.__version__), scripts_sha=M.scripts_sha(SCRIPTS), utc_end=M.utc(), pid=os.getpid())
        else:
            counts = enable_verify(pair)
            yns = [yes_no_ids(ag) for ag in pair.agents]
            cues = [[int(t) for t in ag.ids(VERIFY_CUE)] for ag in pair.agents]
            for i, ag in enumerate(pair.agents):
                if yns[i][0] == yns[i][1]:
                    log(f'WARNING {ag.name}: yes / no share the token {yns[i][0]} (tiny vocabulary?) -> the verdict log-odds is identically 0')
            log(f'verify adapters: {counts} tensors copied from recv; yes/no ids {yns}; cue tokens {[len(c) for c in cues]}')
            t1 = time.time()
            evs = evidence_for(pair, val + fresh, batch=args.gen_batch, deadline=deadline, log=log)
            ev_s = time.time() - t1
            log(f'evidence (span_attn, layer {sel_layer}) for {len(evs)} VAL / fresh examples in {ev_s:.0f} s')
            by_id = {ex['id']: ex for ex in train} if train else {}
            if not by_id:
                tr_all = M.DATA.read_jsonl(args.train)
                by_id = {ex['id']: ex for ex in tr_all}
            miss = [r['id'] for r in rows if r['id'] not in by_id]
            if miss:
                raise SystemExit(f'{len(miss)} training rows are not in --train, e.g. {miss[:3]}')
            tr = train_verifier(pair, rows, by_id, cues[0], yns, args, device, log, deadline=deadline, soft_stop=(args.train_stop_epoch or None))
            log(f'train done: {tr["steps_done"]}/{tr["steps_planned"]} steps in {tr["seconds"]:.0f} s ({tr["sequences"]} sequences); loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; {tr["stopped"]}')
            state = verify_state(pair)
            torch.save(dict(arm=('VERX' if args.swap_frac > 0 else 'VER'), pair=args.pair, seed=args.seed, models=[a['name'] for a in info], state=state, agents=info, receiver=receiver, v=v_meta, cands=cands_meta,
                            train=dict(tr, history=None), config=vars(args)), os.path.join(args.out, 'trainable.pt'))
            log(f'trainable.pt saved before the evaluation ({len(state)} tensors)')
            t_eval = time.time()
            v, v_rec = eval_split(pair, val, recs_val, evs, cues[0], yns, args.score_batch, deadline=deadline, log=log, label='val')
            log(f'VAL: ver_sum {v["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in v["em_variants"].items())} | round1 {v["round1"]["em_agent"]} either {v["round1"]["either_correct"]:.2f} '
                f'agree {v["round1"]["agree"]:.2f} | union any {v["union"]["any_correct"]:.2f} | asym {v["asym"]} | verifier {v["verifier"]} | bytes {v.get("bytes")} | {v["timing"]} | {time.time() - t_eval:.0f} s')
            t1 = time.time()
            f, f_rec = eval_split(pair, fresh, recs_fresh, evs, cues[0], yns, args.score_batch, deadline=deadline, log=log, label='fresh')
            log(f'fresh: ver_sum {f["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in f["em_variants"].items())} | either {f["round1"]["either_correct"]:.2f} union any {f["union"]["any_correct"]:.2f} | {time.time() - t1:.0f} s')
            eval_s = time.time() - t_eval
            if args.keep_preds:
                M.jdump(v_rec, os.path.join(args.out, 'preds_val.json'))
                M.jdump(f_rec, os.path.join(args.out, 'preds_fresh.json'))
            l5v = (l5_summary.get('val') or {}) if l5_summary else {}
            summary = dict(mode='train', arm=('VERX' if args.swap_frac > 0 else 'VER'), pair=args.pair, seed=args.seed, batch_name=BATCH, prereg=PREREG, val=v, fresh=f, complete=True,
                           train_complete=bool(tr['steps_done'] == tr['steps_planned']), steps=tr['steps_done'], steps_planned=tr['steps_planned'], train=tr, agents=info, receiver=receiver,
                           params=dict(verify_tensors=len(state), verify_params=int(sum(t.numel() for t in state.values()))),
                           v=dict(v_meta, run=args.v_from, val_em=(v_summary.get('val') or {}).get('em'), sel_layer=sel_layer), cands=cands_meta,
                           l5=dict(run=args.l5_from, val_em=l5v.get('em'), val_select_p=(l5v.get('em_variants') or {}).get('select_p'), val_union_any=(l5v.get('union') or {}).get('any_correct'),
                                   preds_val_sha256=M.file_sha256(os.path.join(args.l5_from, 'preds_val.json')), preds_fresh_sha256=M.file_sha256(os.path.join(args.l5_from, 'preds_fresh.json'))),
                           channel=dict(cue=VERIFY_CUE, yes=YES, no=NO, yes_no_ids=yns, adapter='verify', init='recv', mode=pair.mode, k_ev=pair.span_k, span_pre=L2.SPAN_PRE, span_post=L2.SPAN_POST,
                                        round1='frozen V1 recv adapters: the partner\'s attention-selected span as clean tokens; the beam-4 candidates of the L5 B1t records',
                                        round2='candidate union; per agent the trained verification log-odds s_i(c) under its own round-1 prompt; consensus = sum_i s_i(c), argmax',
                                        training=f'BCE on EM(c, gold) over the union candidates of train rows (one verify LoRA per agent); doc-swapped negatives frac {args.swap_frac}'),
                           timings=dict(load_s=load_s, evidence_s=ev_s, train_s=tr['seconds'], eval_s=eval_s, total_s=time.time() - t_start),
                           config=dict(vars(args), lora_targets=list(M.LORA_TARGETS), device=str(device), torch=torch.__version__), scripts_sha=M.scripts_sha(SCRIPTS), utc_end=M.utc(), pid=os.getpid())
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    if args.mode == 'gen':
        log(f'l6_verifier done: GEN s{args.seed} {len(rows)} rows; union any {summary["gen"]["union_any_correct"]} either top-1 {summary["gen"]["either_top1"]} total {time.time() - t_start:.0f} s')
    else:
        log(f'l6_verifier done: {summary["arm"]} s{args.seed} VAL ver_sum {v["em"]:.2f} (select_p {v["em_variants"]["select_p"]:.2f}; poe_k4 {v["em_variants"]["poe_k4"]:.2f}; self {v["em_variants"]["ver_self0"]:.2f}/{v["em_variants"]["ver_self1"]:.2f}; '
            f'union {v["union"]["any_correct"]:.2f}) fresh ver_sum {f["em"]:.2f}; steps {tr["steps_done"]}/{tr["steps_planned"]}; total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
