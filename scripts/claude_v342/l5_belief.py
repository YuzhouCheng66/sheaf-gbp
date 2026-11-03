"""l5_belief.py (Claude, 2026-11-03): ladder L5 (pre-registration D8-95): BELIEF-LEVEL consensus, zero training.
Round 1  each agent answers with beam search (k candidates + the sequence log-probabilities) on 'documents | partner evidence | question':
  B1   the frozen V3 pair of D8-94 (recv adapters): the partner's span_sel evidence tokens exactly as in the S3a round-1 cache (--round1-cache)
  B1t  the frozen V1 pair (recv adapters): the partner's attention-selected span (span_attn, K = 32) as CLEAN tokens in the SPAN template
Round 2  the candidate union C (normalised strings); every agent scores every c in C teacher-forced under ITS OWN round-1 prompt:
  l_i(c) = sum_t log p_i(c_t | prompt_i, c_<t) incl. EOS.  Consensus s(c) = sum_i l_i(c) (product of experts = the sheaf-GBP consensus on the
  natural parameters of the discrete belief over whole-answer hypotheses), argmax -> the answer (the criterion, 'poe_k4').
Readouts (per split): the agents' top-1 EM, either / both / agree, select-by-p (the partner-answer + confidence protocol T2 reduced to its
  selection), poe_k1 (the two top-1s only), poe_k4, the length-normalised and lam ~ p variants, the union ceiling (any candidate correct), the
  knower-vs-ignorant asymmetry, message bytes, seconds per example (beam generation, evidence, scoring).
  python -B scripts/claude_v342/l5_belief.py --arm B1 --pair same2 --seed 342101 --receiver-from runs/l1/B0-hetero-s342101 --v-from runs/l2/V3-same2-s342101 \
      --round1-cache runs/l4/S3a-same2-s342101/round1_cache.json --val ... --fresh ... --out runs/l5/B1-same2-s342101
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
import l1_model as M  # noqa: E402
import l1_train as T1  # noqa: E402
import l2_model as L2  # noqa: E402

BATCH = 'L5'
PREREG = 'D8-95'
ARMS = ('B1', 'B1t')
ARM_V = {'B1': ('V3', 'V2'), 'B1t': ('V1',)}          # the frozen round-1 solution each arm is built on
ARM_MODE = {'B1': 'span_sel', 'B1t': 'span_attn'}
K_BEAMS = 4
VARIANTS = ('poe_k4', 'poe_k1', 'poe_k4_norm', 'poe_k4_p', 'select_p', 'agent0', 'agent1')
SCRIPTS = ('l5_belief.py', 'l5_gate.py', 'l2_model.py', 'l1h_model.py', 'l1f_model.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py',
           'l1_fusion.py', 'l1_data.py')


def norm(s):
    f = getattr(M.DATA, 'normalize_answer', None)
    return f(s) if f is not None else M._normalize_answer(s)


def em(pred, gold):
    return float(M.DATA.exact_match(pred, gold))


# ================================================================ pure functions (tested on CPU)
def union(cands):
    """cands: per agent the ordered candidate strings -> the union in first-seen order, deduplicated by the normalised string"""
    out, seen = [], set()
    for lst in cands:
        for c in lst:
            k = norm(c)
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out


def poe_pick(C, scores, weights=None, lens=None, restrict=None):
    """C: candidates; scores[i][c_idx] = l_i(c); weights[i] (default 1); lens[c_idx] (None = no length normalisation); restrict = set of
    candidate indices (None = all) -> (index of the argmax of sum_i w_i l_i(c) [/ len(c)], the fused scores); ties -> the first (agent 0's rank)"""
    n_ag = len(scores)
    w = [1.0] * n_ag if weights is None else [float(x) for x in weights]
    best, best_s, fused = None, None, []
    for ci in range(len(C)):
        s = sum(w[i] * float(scores[i][ci]) for i in range(n_ag))
        if lens is not None:
            s = s / max(1, int(lens[ci]))
        fused.append(s)
        if restrict is not None and ci not in restrict:
            continue
        if best is None or s > best_s:
            best, best_s = ci, s
    return best, fused


def judge(ex, cands, C, scores, lens, p):
    """one example -> the picks of every variant and the readout flags. cands: per agent the candidate strings (rank order, top-1 first),
    C: the union, scores[i][ci], lens[ci] = tokens incl. EOS (agent 0's tokenizer), p[i] = mean token probability of agent i's top-1."""
    gold = ex['answer']
    top = [cands[i][0] if cands[i] else '' for i in range(len(cands))]
    idx = {norm(c): ci for ci, c in enumerate(C)}
    top_idx = [idx[norm(t)] for t in top]
    picks = {}
    picks['poe_k4'], fused = poe_pick(C, scores)
    picks['poe_k1'], _ = poe_pick(C, scores, restrict=set(top_idx))
    picks['poe_k4_norm'], _ = poe_pick(C, scores, lens=lens)
    picks['poe_k4_p'], _ = poe_pick(C, scores, weights=p)
    i_sel = 0 if p[0] >= p[1] else 1
    picks['select_p'] = top_idx[i_sel]
    picks['agent0'] = top_idx[0]
    picks['agent1'] = top_idx[1]
    ok = {k: em(C[ci], gold) for k, ci in picks.items()}
    e0, e1 = ok['agent0'], ok['agent1']
    agree = norm(top[0]) == norm(top[1])
    any_union = max((em(c, gold) for c in C), default=0.0)
    any_agent = [max((em(c, gold) for c in cands[i]), default=0.0) for i in range(len(cands))]
    asym = None
    if not agree and (e0 + e1) == 1.0:                                         # exactly one top-1 is right: the knower a and the ignorant w
        a = 0 if e0 == 1.0 else 1
        wv = 1 - a
        X, Y = top_idx[a], top_idx[wv]
        m_know = float(scores[a][X] - scores[a][Y])                            # the knower's rejection of the wrong answer (> 0 expected)
        m_ign = float(scores[wv][Y] - scores[wv][X])                           # the ignorant's preference for its own answer
        asym = dict(knower=a, m_know=m_know, m_ign=m_ign, m_know_norm=float(scores[a][X] / max(1, lens[X]) - scores[a][Y] / max(1, lens[Y])),
                    m_ign_norm=float(scores[wv][Y] / max(1, lens[Y]) - scores[wv][X] / max(1, lens[X])), poe_k1_correct=ok['poe_k1'], select_correct=ok['select_p'])
    return dict(picks=picks, ok=ok, either=max(e0, e1), both=min(e0, e1), agree=float(agree), agree_ok=(e0 if agree else None), any_union=any_union, any_agent=any_agent,
                fused=fused, asym=asym, top=top, top_idx=top_idx)


def msg_bytes(cands, C):
    """message bytes per example: every agent sends its candidate strings + one float per candidate; round 2 returns one float per union member"""
    cand_b = sum(len(c.encode('utf-8')) for lst in cands for c in lst)
    floats = sum(len(lst) for lst in cands) + len(cands) * len(C)
    return dict(cand_bytes=cand_b, float_bytes=4 * floats, total_bytes=cand_b + 4 * floats)


def owner_label(ex):
    o = ex.get('answer_owner')
    try:
        if o is None or o == 'none' or int(o) < 0:
            return 'none'
        return f'agent{int(o)}'
    except (TypeError, ValueError):
        return 'none'


def aggregate(rows):
    """rows: judge() outputs (+ 'bytes', 'owner') -> the split readouts (percent)"""
    n = len(rows)
    pct = lambda xs: (100.0 * float(np.mean(xs))) if len(xs) else None       # noqa: E731
    out = dict(n=n)
    out['em_variants'] = {k: pct([r['ok'][k] for r in rows]) for k in VARIANTS}
    out['em'] = out['em_variants']['poe_k4']
    agree_rows = [r for r in rows if r['agree'] == 1.0]
    dis = [r for r in rows if r['agree'] != 1.0]
    out['round1'] = dict(em_agent=[out['em_variants']['agent0'], out['em_variants']['agent1']], either_correct=pct([r['either'] for r in rows]), both_correct=pct([r['both'] for r in rows]),
                         agree=pct([r['agree'] for r in rows]), agree_acc=pct([r['agree_ok'] for r in agree_rows]), select_p=out['em_variants']['select_p'],
                         disagree=dict(n=len(dis), select_p_correct=pct([r['ok']['select_p'] for r in dis]), poe_k1_correct=pct([r['ok']['poe_k1'] for r in dis]),
                                       poe_k4_correct=pct([r['ok']['poe_k4'] for r in dis]), either_correct=pct([r['either'] for r in dis])))
    out['union'] = dict(any_correct=pct([r['any_union'] for r in rows]), any_agent=[pct([r['any_agent'][i] for r in rows]) for i in range(2)],
                        size_mean=float(np.mean([len(r['fused']) for r in rows])) if n else None)
    asym = [r['asym'] for r in rows if r['asym'] is not None]
    if asym:
        out['asym'] = dict(n=len(asym), m_know_mean=float(np.mean([a['m_know'] for a in asym])), m_know_median=float(np.median([a['m_know'] for a in asym])),
                           m_ign_mean=float(np.mean([a['m_ign'] for a in asym])), m_ign_median=float(np.median([a['m_ign'] for a in asym])),
                           know_wins=pct([float(a['m_know'] > a['m_ign']) for a in asym]), know_wins_norm=pct([float(a['m_know_norm'] > a['m_ign_norm']) for a in asym]),
                           poe_k1_correct=pct([a['poe_k1_correct'] for a in asym]), select_correct=pct([a['select_correct'] for a in asym]))
    else:
        out['asym'] = dict(n=0)
    owners = sorted({r['owner'] for r in rows if 'owner' in r})
    out['em_by_owner'] = {o: dict(n=len([r for r in rows if r.get('owner') == o]), poe_k4=pct([r['ok']['poe_k4'] for r in rows if r.get('owner') == o]),
                                  select_p=pct([r['ok']['select_p'] for r in rows if r.get('owner') == o])) for o in owners}
    if n and 'bytes' in rows[0]:
        out['bytes'] = {k: float(np.mean([r['bytes'][k] for r in rows])) for k in rows[0]['bytes']}
    return out


# ================================================================ the models
def build_round1_pair(arm, v_ckpt, device, hf_cache=None, tiny=None, max_ctx=None, receiver_state=None, sel_layer=None, **agent_kw):
    """the frozen V pair (V3 / V2 for B1, V1 for B1t) with its recv adapters, in the arm's position mode; nothing trainable"""
    if arm not in ARMS:
        raise ValueError(arm)
    ck = torch.load(v_ckpt, map_location='cpu', weights_only=False)
    if ck.get('arm') not in ARM_V[arm] or not ck.get('recv'):
        raise ValueError(f'{v_ckpt}: {arm} needs a {ARM_V[arm]} checkpoint with recv adapters (arm {ck.get("arm")}, recv {ck.get("recv")})')
    init = {k: v for k, v in ck['state'].items() if '.adapter.' in k or '.reader.' in k or '.selector.' in k}
    kw = dict(agent_kw)
    kw.setdefault('k', int(ck['config'].get('k', L2.K_SPAN)))
    pair = L2.build_pair(ck['arm'], 'same2', device, hf_cache=hf_cache, tiny=tiny, grad_ckpt=False, max_ctx=max_ctx, receiver_state=receiver_state, mode='span_oracle',
                         span_k=kw['k'], init_state=init, recv=True, sel_layer=sel_layer, **kw)
    recv = {k: v for k, v in ck['state'].items() if 'lora_' in k and '.recv.' in k}
    own = [n for n, _ in pair.named_parameters() if 'lora_' in n and '.recv.' in n]
    missing = sorted(set(own) - set(recv))
    if not recv or missing:
        raise ValueError(f'{v_ckpt}: recv tensors {len(recv)} cover {len(own) - len(missing)} of {len(own)}; e.g. missing {missing[:3]}')
    M.load_trainable_state(pair, {k: recv[k] for k in own}, strict=True)
    pair.set_mode(ARM_MODE[arm])
    if arm == 'B1t' and any(a.sel_layer is None for a in pair.agents):
        raise ValueError('B1t needs --sel-layer (the attention layer of the V1 span selection)')
    for p in pair.parameters():
        p.requires_grad_(False)
    pair.eval()
    pair.use('recv')
    return pair, dict(v_arm=ck.get('arm'), v_seed=ck.get('seed'), v_sha256=M.file_sha256(v_ckpt), v_models=ck.get('models'), k=kw['k'])


def evidence_ids(pair, batch, i, cache=None):
    """the partner's evidence tokens placed in receiver i's SPAN template: B1 = the cached hard tokens of the S3a round 1 ('ev' of agent i);
    B1t = the partner's own span tokens (clean text, pair.targets in span_attn mode)"""
    if cache is not None:
        out = []
        for ex in batch:
            r = cache.get(ex.get('id'))
            if r is None or str(i) not in r:
                raise KeyError(f'round-1 cache has no entry for example {ex.get("id")} agent {i}')
            out.append([int(t) for t in r[str(i)]['ev']])
        return out
    ids, lens = pair.targets(batch, i)
    return [[int(t) for t in ids[b, :int(lens[b])].tolist()] for b in range(len(batch))]


def round1_pieces(pair, batch, i, cache=None):
    """-> [(pre = [BOS] documents | SPAN template with the evidence, suf = question)] of agent i"""
    ag = pair.agent(i)
    pcs, _dl = pair.doc_pieces(batch, i)
    evs = evidence_ids(pair, batch, i, cache)
    return [(list(p) + ag.hint_pre + evs[b] + ag.hint_post, list(s)) for b, (p, s, _) in enumerate(pcs)]


@torch.no_grad()
def beam_candidates(ag, pre_suf, k, max_new, adapter='recv'):
    """beam search (k beams, k returned) on the assembled prompts (LEFT padding) -> per example the ordered, deduplicated candidate strings
    (first line, stripped; empty strings dropped unless nothing else remains)"""
    ag.use(adapter)
    ag.eval()
    A = ag.assemble([(p, s, []) for p, s in pre_suf], None, 'left')
    out = ag.model.generate(inputs_embeds=A['embeds'], attention_mask=A['mask'], max_new_tokens=int(max_new), do_sample=False, num_beams=int(k), num_return_sequences=int(k),
                            early_stopping=True, length_penalty=1.0, pad_token_id=ag.pad, eos_token_id=ag.eos, use_cache=True, return_dict_in_generate=True)
    seqs = out.sequences
    if seqs.shape[1] > max_new:
        seqs = seqs[:, -int(max_new):]
    rows = seqs.tolist()
    res = []
    for b in range(len(pre_suf)):
        cands, seen = [], set()
        for r in rows[b * k:(b + 1) * k]:
            cut = r.index(ag.eos) if ag.eos in r else len(r)
            t = ag.tok.decode(r[:cut], skip_special_tokens=True).split('\n')[0].strip()
            key = norm(t)
            if t and key not in seen:
                seen.add(key)
                cands.append(t)
        res.append(cands if cands else [''])
    return res


@torch.no_grad()
def score_candidates(ag, prompts, cands, adapter='recv', batch_size=32):
    """teacher-forced log-probabilities: prompts[j] (id list), cands[j] (id list WITHOUT EOS) -> (sum log p incl. EOS, n tokens incl. EOS,
    mean token probability excl. EOS (over the EOS alone for an empty candidate)) per pair"""
    ag.use(adapter)
    ag.eval()
    out = []
    for lo in range(0, len(prompts), batch_size):
        P, Cc = prompts[lo:lo + batch_size], cands[lo:lo + batch_size]
        pcs = [(list(p), [], [int(t) for t in c] + [ag.eos]) for p, c in zip(P, Cc)]
        A = ag.assemble(pcs, None, 'right', embeds=False)
        hs = ag.decoder(input_ids=A['ids'], attention_mask=A['mask'], use_cache=False).last_hidden_state
        for b, (_p, _s, a) in enumerate(pcs):
            a0 = A['ans_start'][b]
            pos = torch.tensor([a0 + j - 1 for j in range(len(a))], dtype=torch.long, device=hs.device)
            lp = torch.log_softmax(ag.lm_head(hs[b, pos]).float(), -1)
            tok = torch.tensor(a, dtype=torch.long, device=hs.device)
            lpt = lp[torch.arange(len(a), device=hs.device), tok]
            body = lpt[:-1] if len(a) > 1 else lpt
            out.append((float(lpt.sum()), len(a), float(body.exp().mean())))
        del hs
    return out


@torch.no_grad()
def run_split(pair, examples, cache=None, k=K_BEAMS, max_new=M.ANSWER_MAX_NEW, gen_batch=8, score_batch=32, deadline=None, log=None, label=''):
    """the whole protocol on `examples` -> (readouts, records)"""
    pair.eval()
    rows, records = [], []
    t_gen = t_ev = t_sc = 0.0
    for bi, batch in enumerate(M.chunks(examples, gen_batch)):
        M.check_deadline(deadline, f'l5 {label}')
        t0 = time.time()
        r1 = [round1_pieces(pair, batch, i, cache) for i in range(M.N_AGENTS)]
        t_ev += time.time() - t0
        t0 = time.time()
        cands = [beam_candidates(pair.agent(i), r1[i], k, max_new) for i in range(M.N_AGENTS)]    # cands[i][b] = the strings of agent i
        t_gen += time.time() - t0
        t0 = time.time()
        Cs = [union([cands[i][b] for i in range(M.N_AGENTS)]) for b in range(len(batch))]
        scores = [[None] * len(batch) for _ in range(M.N_AGENTS)]
        lens = [None] * len(batch)
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            prompts, cids, where = [], [], []
            for b in range(len(batch)):
                pr = r1[i][b][0] + r1[i][b][1]
                for ci, c in enumerate(Cs[b]):
                    prompts.append(pr)
                    cids.append([int(t) for t in ag.ids(' ' + c)] if c else [])
                    where.append((b, ci))
            got = score_candidates(ag, prompts, cids, 'recv', score_batch)
            for (b, ci), (s, n, mp) in zip(where, got):
                if scores[i][b] is None:
                    scores[i][b] = [None] * len(Cs[b])
                scores[i][b][ci] = (s, n, mp)
                if i == 0:
                    if lens[b] is None:
                        lens[b] = [None] * len(Cs[b])
                    lens[b][ci] = n
        t_sc += time.time() - t0
        for b, ex in enumerate(batch):
            C = Cs[b]
            sc = [[scores[i][b][ci][0] for ci in range(len(C))] for i in range(M.N_AGENTS)]
            idx = {norm(c): ci for ci, c in enumerate(C)}
            p = [scores[i][b][idx[norm(cands[i][b][0])]][2] for i in range(M.N_AGENTS)]
            j = judge(ex, [cands[i][b] for i in range(M.N_AGENTS)], C, sc, lens[b], p)
            j['bytes'] = msg_bytes([cands[i][b] for i in range(M.N_AGENTS)], C)
            j['owner'] = owner_label(ex)
            rows.append(j)
            rec = dict(id=ex.get('id'), gold=ex['answer'], owner=j['owner'], cands=[cands[i][b] for i in range(M.N_AGENTS)], union=C, p=p,
                       scores=[[round(x, 4) for x in sc[i]] for i in range(M.N_AGENTS)], lens=lens[b], picks={kk: C[ci] for kk, ci in j['picks'].items()}, ok=j['ok'])
            if cache is not None:
                cr = cache.get(ex.get('id')) or {}
                rec['cached_greedy'] = [dict(ans=cr[str(i)]['ans'], p=cr[str(i)]['p'], em=em(cr[str(i)]['ans'], ex['answer'])) for i in range(M.N_AGENTS) if str(i) in cr]
            records.append(rec)
        if log and (bi + 1) % 25 == 0:
            done = len(rows)
            log(f'l5 {label}: {done} examples; poe_k4 {100.0 * np.mean([r["ok"]["poe_k4"] for r in rows]):.2f} select_p {100.0 * np.mean([r["ok"]["select_p"] for r in rows]):.2f} '
                f'either {100.0 * np.mean([r["either"] for r in rows]):.2f}; s/example gen {t_gen / done:.3f} ev {t_ev / done:.3f} score {t_sc / done:.3f}')
    out = aggregate(rows)
    n = max(1, len(rows))
    out['timing'] = dict(gen_s_per_example=t_gen / n, evidence_s_per_example=t_ev / n, score_s_per_example=t_sc / n, round2_s_per_example=t_sc / n)
    if cache is not None and records and records[0].get('cached_greedy'):
        cg = [r['cached_greedy'] for r in records if r.get('cached_greedy')]
        out['cached_greedy'] = dict(em_agent=[100.0 * float(np.mean([c[i]['em'] for c in cg])) for i in range(M.N_AGENTS)],
                                    select_p=100.0 * float(np.mean([c[0]['em'] if c[0]['p'] >= c[1]['p'] else c[1]['em'] for c in cg])))
    out['examples'] = [dict(gold=r['gold'], top=[c[0] for c in r['cands']], poe=r['picks']['poe_k4']) for r in records[:12]]
    return out, records


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--pair', choices=('same2',))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train', default=None, help='unused (no training); accepted for the record')
    ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--receiver-from', default=None, help='the B0 run dir of this seed (the frozen default LoRA); required unless --tiny')
    ap.add_argument('--v-from', default=None, help='B1: the V3 run dir; B1t: the V1 run dir (trainable.pt with the recv adapters); required')
    ap.add_argument('--round1-cache', default=None, help='B1: the S3a round1_cache.json of this seed (the evidence tokens of D8-94); required for B1')
    ap.add_argument('--sel-layer', type=int, default=None, help='B1t: the attention layer of the span selection (default: span.sel_layer of the V1 summary)')
    ap.add_argument('--k-beams', type=int, default=K_BEAMS)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--gen-batch', type=int, default=8)
    ap.add_argument('--score-batch', type=int, default=32)
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--d-e', type=int, default=M.D_E); ap.add_argument('--n-heads', type=int, default=M.N_HEADS)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA)
    ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--limit-eval', type=int, default=0); ap.add_argument('--limit-train', type=int, default=0, help='unused (no training)')
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('arm', 'pair', 'seed', 'val', 'fresh', 'out', 'v_from'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} is required')
    if not args.tiny and not args.receiver_from:
        raise SystemExit('--receiver-from is required outside --tiny')
    if args.arm == 'B1' and not args.round1_cache:
        raise SystemExit('--round1-cache (the S3a round1_cache.json) is required for B1')
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l5_belief start pid {os.getpid()} host {platform.node()} arm {args.arm} pair {args.pair} seed {args.seed} out {args.out}; v {args.v_from} cache {args.round1_cache}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deadline = args.deadline_epoch or None
    _train, val, fresh = T1.read_data(args, log, need_train=False)
    tiny = M.tiny_models(val + fresh, args.seed) if args.tiny else None
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
    if args.arm == 'B1t' and sel_layer is None:
        sel_layer = (v_summary.get('span') or {}).get('sel_layer')
        if sel_layer is None:
            raise SystemExit('--sel-layer is required for B1t (the V1 summary has no span.sel_layer)')
        sel_layer = int(sel_layer)
    cache = None
    if args.round1_cache:
        d = M.jload(args.round1_cache)
        cache = d['cache']
        miss = [ex.get('id') for ex in val + fresh if ex.get('id') not in cache]
        if miss:
            raise SystemExit(f'round-1 cache misses {len(miss)} VAL / fresh examples, e.g. {miss[:3]}')
        log(f'round-1 cache: {len(cache)} examples (k_ev {d.get("k_ev")}); VAL + fresh fully covered')
    try:
        M.seed_all(args.seed)
        t0 = time.time()
        pair, v_meta = build_round1_pair(args.arm, v_ckpt, device, hf_cache=args.hf_cache, tiny=tiny, max_ctx=args.max_ctx_tokens, receiver_state=receiver_state, sel_layer=sel_layer, **agent_kw)
        load_s = time.time() - t0
        info = T1.agents_info(pair)
        log(f'model: {info}; v {v_meta}; V VAL EM {v_summary.get("val", {}).get("em")}; mode {pair.mode} k_ev {pair.span_k} sel_layer {sel_layer}; beams {args.k_beams}')
        kw = dict(k=args.k_beams, max_new=args.max_new_tokens, gen_batch=args.gen_batch, score_batch=args.score_batch, deadline=deadline, log=log)
        t1 = time.time()
        v, v_rec = run_split(pair, val, cache, label='val', **kw)
        log(f'VAL: poe_k4 {v["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in v["em_variants"].items())} | round1 {v["round1"]["em_agent"]} either {v["round1"]["either_correct"]:.2f} '
            f'agree {v["round1"]["agree"]:.2f} | union any {v["union"]["any_correct"]:.2f} size {v["union"]["size_mean"]:.2f} | asym {v["asym"]} | bytes {v.get("bytes")} | {v["timing"]} | {time.time() - t1:.0f} s')
        t1 = time.time()
        f, f_rec = run_split(pair, fresh, cache, label='fresh', **kw)
        log(f'fresh: poe_k4 {f["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in f["em_variants"].items())} | either {f["round1"]["either_correct"]:.2f} union any {f["union"]["any_correct"]:.2f} | {time.time() - t1:.0f} s')
        if args.keep_preds:
            M.jdump(v_rec, os.path.join(args.out, 'preds_val.json'))
            M.jdump(f_rec, os.path.join(args.out, 'preds_fresh.json'))
    except M.Deadline as e:
        T1.incomplete(args.out, str(e), log)
        return 3
    models = list(dict.fromkeys(a['name'] for a in info))
    summary = dict(arm=args.arm, pair=args.pair, seed=args.seed, model=None, models=models, batch_name=BATCH, prereg=PREREG, val=v, fresh=f, complete=True, steps=0, steps_planned=0,
                   agents=info, receiver=receiver, v=dict(v_meta, run=args.v_from, val_em=(v_summary.get('val') or {}).get('em'), none_em=((v_summary.get('baseline') or {}).get('none', {}).get('val') or {}).get('em'),
                                                           answer_gen_s_per_example=(v_summary.get('val') or {}).get('answer_gen_s_per_example'), sel_layer=sel_layer),
                   round1_cache=dict(path=args.round1_cache, n=(len(cache) if cache is not None else 0)),
                   channel=dict(k_beams=args.k_beams, max_new=args.max_new_tokens, adapter='recv', mode=pair.mode, k_ev=pair.span_k, span_pre=L2.SPAN_PRE, span_post=L2.SPAN_POST,
                                round1=('frozen V3 context of D8-94: cached span_sel evidence tokens, beam search with the V3 recv adapters' if args.arm == 'B1'
                                        else "frozen V1 recv adapters: the partner's attention-selected span as clean tokens, beam search"),
                                round2='candidate union; teacher-forced sequence log-probabilities under each agent\'s own prompt; consensus = sum_i l_i(c), argmax', training='none'),
                   timings=dict(load_s=load_s, total_s=time.time() - t_start),
                   config=dict(vars(args), lora_targets=list(M.LORA_TARGETS), device=str(device), torch=torch.__version__), scripts_sha=M.scripts_sha(SCRIPTS), utc_end=M.utc(), pid=os.getpid())
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        summary['config']['gpu'] = torch.cuda.get_device_name(device)
    M.jdump(summary, sp)
    log(f'l5_belief done: {args.arm} s{args.seed} VAL poe_k4 {v["em"]:.2f} (select_p {v["em_variants"]["select_p"]:.2f}; agents {v["round1"]["em_agent"]}; either {v["round1"]["either_correct"]:.2f}; '
        f'union {v["union"]["any_correct"]:.2f}) fresh poe_k4 {f["em"]:.2f} total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
