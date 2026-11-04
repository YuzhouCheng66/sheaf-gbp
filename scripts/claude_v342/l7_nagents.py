"""l7_nagents.py (Claude, 2026-11-04): ladder L7 (pre-registration D8-97): the agent count N as the single variable.
One base model (Qwen3-1.7B, bf16) carries every role as a LoRA shared by the N agents; an agent is a shard index (ex['shards'][i]).
Adapters: 'default' = the frozen B0 of the seed (runs/l1/B0-hetero-s<seed>, agent 0's LoRA), 'tsend' / 'trecv' = the text arm (L1's T recipe,
fresh LoRAs), 'recv' = the belief arm's round-1 reader (initialised from B0), 'verify' = the verifier (initialised from recv).
  --job tsend   train the text sender (target: the supporting sentences of the gold paragraph in the agent's own shard, ' (none)' without one),
                then generate the <= --budget token messages of every agent for the stage-2 subset, VAL and fresh -> messages.json, trainable.pt
  --job trecv   --tsend-from: the text receiver, stage 1 one pass of gold messages, stage 2 on the generated messages of the subset; every agent
                reads the N-1 messages of the others ("Agent j says: ..."); evaluation = the mean EM over the N agents' answers (T_N)
  --job recv    the belief arm's reader: every agent reads the N-1 attention-selected 32-token windows of the others' documents (token ids,
                "From agent j's documents: ...") and answers; CE on the answer (the recipe of V1 stage 2, the same agent-example budget as trecv)
  --job gen     --recv-from: for every training row the windows of every agent and every agent's greedy answer -> cands_train.json
  --job ver     --recv-from --cands-from: the verifier LoRA (BCE on EM(c, gold); 2 (agent, candidate) sequences per row) then the evaluation:
                candidates = the N greedy answers' union, s_i(c) under agent i's own round-1 prompt, consensus argmax sum_i s_i(c) (B_N);
                readouts: the no-evidence belief arm (B'_N: own-document answers and scores), the prior-subtracted sums, the per-agent and
                informed / ignorant verifier statistics, select-by-p, majority vote, bytes and seconds per example.
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

BATCH = 'L7'
PREREG = 'D8-97'
JOBS = ('tsend', 'trecv', 'recv', 'gen', 'ver')
ROLE_OF = {'tsend': 'tsend', 'trecv': 'trecv', 'recv': 'recv', 'gen': 'recv', 'ver': 'verify'}
BUDGET = 64                                      # text message tokens (L1 T)
K_EV = 32                                        # window tokens per sender (V1)
SINK_SKIP = 2                                    # l2_model.SINK_SKIP: the first document positions are the attention sink, never selected
ATTN_SUB_BATCH = 4                               # l2_model.ATTN_SUB_BATCH
VERIFY_CUE = '\nIs the answer above correct? Reply yes or no.\nVerdict:'
YES, NO = ' yes', ' no'
SPAN_PRE = "From agent {j}'s documents:"
SPAN_POST = '\n'
MSG_LINE = 'Agent {j} says:{m}\n'
PRIOR_PRE = 'Question: {q}\nAnswer:'
STAGE2_EXAMPLES = 5000
SCRIPTS = ('l7_nagents.py', 'l7_gate.py', 'l7_data.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
# --tiny only: the words of the templates must exist in the tiny vocabulary
TINY_WORDS = dict(question="from agent 0 ' s documents says is the answer above correct ? reply yes or no . verdict : facts my partner needs : ( none ) ( nothing ) context", answer='yes no')


def norm(s):
    f = getattr(M.DATA, 'normalize_answer', None)
    return f(s) if f is not None else M._normalize_answer(s)


def em(pred, gold):
    return float(M.DATA.exact_match(pred, gold))


def union(cands):
    out, seen = [], set()
    for lst in cands:
        for c in lst:
            k = norm(c)
            if k not in seen:
                seen.add(k)
                out.append(c)
    return out


# ================================================================ pure functions (tested on CPU)
def auroc(scores, labels):
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


def majority(tops):
    """the most frequent normalised top-1 (ties: the first agent's order) -> index into tops"""
    counts = {}
    for i, t in enumerate(tops):
        k = norm(t)
        counts.setdefault(k, [0, i])
        counts[k][0] += 1
    best = max(counts.values(), key=lambda v: (v[0], -v[1]))
    return best[1]


def informed_of(ex):
    gs = ex.get('gold_shards')
    if gs is None:                                                           # splits_l1 rows: both agents hold a gold paragraph
        return list(range(len(ex['shards'])))
    return sorted({int(g) for g in gs})


def owner_label(ex):
    o = ex.get('answer_owner')
    try:
        if o is None or o == 'none' or int(o) < 0:
            return 'none'
        return f'agent{int(o)}'
    except (TypeError, ValueError):
        return 'none'


def pick_pairs(row, n_agents, rng):
    """the 2 training sequences of a row: (agent, candidate, label); one positive and one negative when both exist"""
    C = row['union']
    labels = [em(c, row['gold']) for c in C]
    pos = [k for k, l in enumerate(labels) if l == 1.0]
    neg = [k for k, l in enumerate(labels) if l == 0.0]
    a1, a2 = int(rng.randint(n_agents)), int(rng.randint(n_agents))
    if pos and neg:
        return [(a1, C[pos[int(rng.randint(len(pos)))]], 1.0), (a2, C[neg[int(rng.randint(len(neg)))]], 0.0)]
    pool = pos if pos else neg
    k1, k2 = pool[int(rng.randint(len(pool)))], pool[int(rng.randint(len(pool)))]
    return [(a1, C[k1], labels[k1]), (a2, C[k2], labels[k2])]


def judge(ex, cands, C, S, S0, p, own_cands, C_own, S_own):
    """one example of the belief arm. cands[i]: agent i's candidate list (top-1 first); C: the union; S[i][ci]: s_i(c); S0[ci]: the
    no-document prior s_0(c); p[i]: mean token probability of agent i's top-1; own_cands / C_own / S_own: the no-evidence arm (B')."""
    gold = ex['answer']
    N = len(cands)
    inf = informed_of(ex)
    top = [cands[i][0] if cands[i] else '' for i in range(N)]
    idx = {norm(c): ci for ci, c in enumerate(C)}
    top_idx = [idx[norm(t)] for t in top]
    n = len(C)
    ssum = [sum(float(S[i][ci]) for i in range(N)) for ci in range(n)]
    picks = dict(ver_sum=argmax_first(ssum))
    for i in range(N):
        picks[f'ver_self{i}'] = argmax_first([float(S[i][ci]) for ci in range(n)])
        picks[f'agent{i}'] = top_idx[i]
    picks['ver_select'] = argmax_first([max(float(S[i][ci]) for i in range(N)) for ci in range(n)])
    picks['ver_prior'] = argmax_first([ssum[ci] - (N - 1) * float(S0[ci]) for ci in range(n)])
    picks['ver_prior_half'] = argmax_first([ssum[ci] - 0.5 * (N - 1) * float(S0[ci]) for ci in range(n)])
    picks['ver_informed'] = argmax_first([sum(float(S[i][ci]) for i in inf) for ci in range(n)])
    owner = owner_label(ex)
    picks['ver_owner'] = picks[f'ver_self{int(owner[-1])}'] if owner != 'none' else picks['ver_sum']
    picks['select_p'] = top_idx[argmax_first(p)]
    picks['vote'] = top_idx[majority(top)]
    ok = {k: em(C[ci], gold) for k, ci in picks.items()}
    e = [ok[f'agent{i}'] for i in range(N)]
    labels = [em(c, gold) for c in C]
    agree = float(len({norm(t) for t in top}) == 1)
    out = dict(picks=picks, ok=ok, either=max(e), informed_either=max(e[i] for i in inf), all_correct=min(e), agree=agree,
               any_union=(max(labels) if labels else 0.0), size=n, owner=owner, fused=ssum, top=top, top_idx=top_idx)
    out['know_case'] = bool(any(e[i] == 1.0 for i in inf) and any(e[i] == 0.0 for i in range(N)))
    out['cand_recs'] = [dict(agent=i, rel=('informed' if i in inf else 'ignorant'), s=float(S[i][ci]), label=labels[ci]) for i in range(N) for ci in range(n)]
    out['prior_recs'] = [dict(s=float(S0[ci]), label=labels[ci]) for ci in range(n)]
    # the no-evidence arm: own-document answers, scores under the own-document prompt
    own_top = [own_cands[i][0] if own_cands[i] else '' for i in range(N)]
    oidx = {norm(c): ci for ci, c in enumerate(C_own)}
    osum = [sum(float(S_own[i][ci]) for i in range(N)) for ci in range(len(C_own))]
    o_pick = argmax_first(osum)
    out['own'] = dict(em_agent=[em(t, gold) for t in own_top], ver_sum=em(C_own[o_pick], gold), vote=em(C_own[oidx[norm(own_top[majority(own_top)])]], gold),
                      any_union=(max((em(c, gold) for c in C_own), default=0.0)), size=len(C_own))
    return out


def aggregate(rows, N):
    n = len(rows)
    pct = lambda xs: (100.0 * float(np.mean(xs))) if len(xs) else None       # noqa: E731
    out = dict(n=n, n_agents=N)
    keys = list(rows[0]['ok'].keys()) if rows else []
    out['em_variants'] = {k: pct([r['ok'][k] for r in rows]) for k in keys}
    out['em'] = out['em_variants'].get('ver_sum')
    out['round1'] = dict(em_agent=[out['em_variants'].get(f'agent{i}') for i in range(N)], either_correct=pct([r['either'] for r in rows]),
                         informed_either=pct([r['informed_either'] for r in rows]), all_correct=pct([r['all_correct'] for r in rows]), agree_all=pct([r['agree'] for r in rows]),
                         vote=out['em_variants'].get('vote'), select_p=out['em_variants'].get('select_p'))
    out['union'] = dict(any_correct=pct([r['any_union'] for r in rows]), size_mean=float(np.mean([r['size'] for r in rows])) if n else None)
    kc = [r for r in rows if r['know_case']]
    out['know'] = dict(n=len(kc), know_wins=pct([r['ok']['ver_sum'] for r in kc]), select_p_correct=pct([r['ok']['select_p'] for r in kc]), vote_correct=pct([r['ok']['vote'] for r in kc]),
                       informed_either=pct([r['informed_either'] for r in kc]))
    owners = sorted({r['owner'] for r in rows})
    out['em_by_owner'] = {o: dict(n=len([r for r in rows if r['owner'] == o]), **{k: pct([r['ok'][k] for r in rows if r['owner'] == o]) for k in ('ver_sum', 'select_p', 'vote', 'ver_owner')})
                          for o in owners}
    cr = [c for r in rows for c in r['cand_recs']]
    ver = {}
    for rel in ('informed', 'ignorant', 'all'):
        sub = [c for c in cr if rel == 'all' or c['rel'] == rel]
        ver[rel] = dict(n=len(sub), auroc=auroc([c['s'] for c in sub], [c['label'] for c in sub]) if sub else None,
                        abs_s_median=float(np.median([abs(c['s']) for c in sub])) if sub else None,
                        s_median=float(np.median([c['s'] for c in sub])) if sub else None,
                        p_yes_pos=(float(np.mean([sigmoid(c['s']) for c in sub if c['label'] == 1.0])) if any(c['label'] == 1.0 for c in sub) else None),
                        p_yes_neg=(float(np.mean([sigmoid(c['s']) for c in sub if c['label'] == 0.0])) if any(c['label'] == 0.0 for c in sub) else None))
    flat = None
    if ver['informed']['n'] and ver['ignorant']['n']:
        flat = bool(ver['ignorant']['abs_s_median'] < ver['informed']['abs_s_median'])
    pr = [c for r in rows for c in r['prior_recs']]
    out['verifier'] = dict(by_rel=ver, positive_rate=pct([c['label'] for c in cr]), flat_when_ignorant=flat,
                           prior=dict(n=len(pr), auroc=auroc([c['s'] for c in pr], [c['label'] for c in pr]) if pr else None, s_median=float(np.median([c['s'] for c in pr])) if pr else None))
    out['own'] = dict(em_agent=[pct([r['own']['em_agent'][i] for r in rows]) for i in range(N)], ver_sum=pct([r['own']['ver_sum'] for r in rows]), vote=pct([r['own']['vote'] for r in rows]),
                      any_union=pct([r['own']['any_union'] for r in rows]), size_mean=float(np.mean([r['own']['size'] for r in rows])) if n else None)
    if n and 'bytes' in rows[0]:
        out['bytes'] = {k: float(np.mean([r['bytes'][k] for r in rows])) for k in rows[0]['bytes']}
    return out


def belief_bytes(N, k_ev, cands, C):
    """per example: every agent broadcasts its window (k_ev token ids x 2 B), its candidate string and |C| int8 scores; wire = x (N-1) receivers"""
    span_b = N * k_ev * 2
    cand_b = sum(len(c.encode('utf-8')) for lst in cands for c in lst[:1])
    score_b = N * len(C)
    msg = span_b + cand_b + score_b
    return dict(span_bytes=span_b, cand_bytes=cand_b, score_bytes_int8=score_b, score_bytes_f32=4 * score_b, message_bytes=msg, wire_bytes=msg * (N - 1))


# ================================================================ the agent (one base model, several LoRA roles)
class AgentN(M.Agent):
    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, comm=False, **kw)
        if not self.lora:
            raise ValueError('AgentN needs LoRA')
        self.roles = {'default': [p for n, p in self.model.named_parameters() if 'lora_' in n and '.default.' in n]}
        self.active = 'default'
        self.use('default')

    def add_role(self, name, init_from=None):
        """a new LoRA `name` (PEFT init: lora_B = 0 -> the base model) or a copy of the adapter `init_from`; -> tensors copied"""
        if name in self.roles:
            return 0
        self.model.add_adapter(name, self.model.peft_config['default'])
        params = dict(self.model.named_parameters())
        mine = [(n, p) for n, p in params.items() if 'lora_' in n and f'.{name}.' in n]
        if not mine:
            raise RuntimeError(f'{self.name}: adapter {name} has no tensors')
        n_copied = 0
        if init_from is not None:
            for n, p in mine:
                q = params[n.replace(f'.{name}.', f'.{init_from}.')]
                if q.shape != p.shape:
                    raise RuntimeError(f'{self.name}: {name} tensor {n} shape {tuple(p.shape)} != {tuple(q.shape)}')
                with torch.no_grad():
                    p.copy_(q)
                n_copied += 1
        self.roles[name] = [p for _, p in mine]
        self.use(self.active)
        return n_copied

    def use(self, name, train=False):
        if name not in self.roles:
            raise ValueError(f'{self.name}: no adapter {name} (have {sorted(self.roles)})')
        if len(self.roles) > 1:
            self.model.set_adapter(name)                                       # PEFT re-sets requires_grad: fixed below
        for r, ps in self.roles.items():
            for p in ps:
                p.requires_grad_(bool(train) and r == name)
        self.active = name

    def role_state(self, name):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if 'lora_' in n and f'.{name}.' in n}

    def load_role_state(self, name, state):
        own = [n for n, _ in self.named_parameters() if 'lora_' in n and f'.{name}.' in n]
        missing = sorted(set(own) - set(state))
        if not own or missing:
            raise ValueError(f'{name}: state covers {len(own) - len(missing)} of {len(own)} tensors; e.g. missing {missing[:3]}')
        M.load_trainable_state(self, {k: state[k] for k in own}, strict=True)
        return len(own)

    @torch.no_grad()
    def attn_positions(self, pieces, doc_lens, layer, k):
        """l2_model.AgentL2.attn_positions: the k-window of the documents region with the largest summed attention of the question tokens at `layer`
        (heads and question positions averaged; the first SINK_SKIP document positions zeroed) with the B0 adapter -> (B, k) positions (-1 = none)"""
        self.use('default')
        nb = 1 if self.bos is not None else 0
        B = len(pieces)
        pos = torch.full((B, k), -1, dtype=torch.long)
        for lo_b in range(0, B, ATTN_SUB_BATCH):
            sub = pieces[lo_b:lo_b + ATTN_SUB_BATCH]
            A = self.assemble([(p, s, []) for p, s, _ in sub], None, 'right', embeds=False)
            out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_attentions=True, use_cache=False)
            att = out.attentions[layer]
            for bb in range(len(sub)):
                b = lo_b + bb
                last = A['start'][bb] + A['lens'][bb] - 1
                qlo = A['start'][bb] + len(sub[bb][0])
                if qlo > last:
                    qlo = last
                lo, hi = A['start'][bb] + nb, A['start'][bb] + nb + int(doc_lens[b])
                if hi <= lo:
                    continue
                row = att[bb, :, qlo:last + 1, lo:hi].float().mean(0).mean(0).clone()
                row[:min(SINK_SKIP, row.numel())] = 0.0
                n_ = int(hi - lo)
                if n_ <= k:
                    start = int(lo)
                else:
                    c = torch.cumsum(torch.cat([row.new_zeros(1), row]), 0)
                    start = int(lo) + int((c[k:] - c[:-k]).argmax())
                m = min(k, hi - start)
                pos[b, :m] = start + torch.arange(m)
            del out, att
        return pos


def load_b0(ag, path):
    """runs/l1/B0-hetero-s<seed>/trainable.pt: agent 0's LoRA (Qwen) -> this agent's 'default' adapter"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck
    a0 = {k[len('agents.0.'):]: v for k, v in state.items() if k.startswith('agents.0.') and 'lora_' in k}
    if not a0:
        raise ValueError(f'{path}: no agents.0.* LoRA tensors')
    n = ag.load_role_state('default', a0)
    return dict(path=os.path.abspath(path), sha256=M.file_sha256(path), arm=ck.get('arm') if isinstance(ck, dict) else None, seed=ck.get('seed') if isinstance(ck, dict) else None,
                models=ck.get('models') if isinstance(ck, dict) else None, n_tensors=n)


def build_agent(args, tiny, device):
    kw = dict(lora=True, grad_ckpt=not args.no_grad_ckpt, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, layer_frac=args.layer_frac)
    if tiny is not None:
        m, t = tiny[M.TINY_NAMES[0]]
        ag = AgentN.from_pretrained_objects(m, t, M.TINY_NAMES[0], **kw)
    else:
        ag = AgentN.from_pretrained(M.MODELS['same'][0], device, args.hf_cache, **kw)
    b0 = None
    if args.b0_from:
        b0 = load_b0(ag, os.path.join(args.b0_from, 'trainable.pt'))
        if b0.get('arm') not in (None, 'B0') or b0.get('seed') not in (None, args.seed):
            raise SystemExit(f'--b0-from is not the B0 checkpoint of seed {args.seed}: {b0}')
    ag.use('default')
    return ag, b0


def load_role_ckpt(ag, path, role, job, seed, n_agents):
    ck = torch.load(os.path.join(path, 'trainable.pt'), map_location='cpu', weights_only=False)
    if ck.get('job') != job or ck.get('role') != role or int(ck.get('seed', -1)) != seed or int(ck.get('n_agents', -1)) != n_agents:
        raise SystemExit(f'{path}: not the {job}/{role} checkpoint of seed {seed} N {n_agents}: job {ck.get("job")} role {ck.get("role")} seed {ck.get("seed")} N {ck.get("n_agents")}')
    ag.add_role(role)
    n = ag.load_role_state(role, ck['state'])
    return dict(path=os.path.abspath(path), sha256=M.file_sha256(os.path.join(path, 'trainable.pt')), job=job, role=role, seed=seed, n_tensors=n)


# ================================================================ prompts and pieces
def fit_pieces(ag, ex, i, build, max_ctx, target=None):
    """build(ex) -> (prefix text, suffix text); the shard of agent i is shrunk (M.shrink_paragraphs) until [BOS] prefix suffix fits max_ctx;
    -> ((pre with BOS, suf, ans = target tokens + EOS or []), truncation code 0 / 1 / 2)"""
    nb = 1 if ag.bos is not None else 0
    pre_t, suf_t = build(ex)
    pre, suf = ag.ids(pre_t), ag.ids(suf_t)
    trunc = 0
    if max_ctx and nb + len(pre) + len(suf) > max_ctx:
        over = nb + len(pre) + len(suf) - max_ctx
        for attempt in range(4):
            ex = M.shrink_paragraphs(ag, ex, i, 'latent', over)
            pre_t, suf_t = build(ex)
            pre, suf = ag.ids(pre_t), ag.ids(suf_t)
            trunc = 1
            over = nb + len(pre) + len(suf) - max_ctx
            if over <= 0:
                break
            over += 4 * (attempt + 1)
        if nb + len(pre) + len(suf) > max_ctx:
            pre = pre[:max(1, max_ctx - nb - len(suf))]
            trunc = 2
    ans = (ag.ids(target) + [ag.eos]) if target is not None else []
    return (([ag.bos] if nb else []) + pre, suf, ans), trunc


def doc_parts(ex, i):
    return M.prompt_parts(ex, i, 'latent')                                   # ("Context:\n<shard>\n", "\nQuestion: q\nAnswer:")


def sender_parts(ex, i):
    return M.prompt_parts(ex, i, 'sender')                                   # ("Context:\n<shard>\n\nQuestion: q\nFacts my partner needs:", '')


def own_parts(ex, i):
    return M.prompt_parts(ex, i, 'B0')                                       # ("Context:\n<shard>\n\nQuestion: q\nAnswer:", '')


def receiver_parts(ex, i, msgs):
    """msgs: {j: message text} of the other agents"""
    pre, _ = doc_parts(ex, i)
    lines = ''.join(MSG_LINE.format(j=j, m=((' ' + m.strip()) if m.strip() else ' (nothing)')) for j, m in sorted(msgs.items()))
    return pre + '\n' + lines, f"Question: {ex['question']}\nAnswer:"


class Prompter:
    """pieces of every kind for one AgentN; window ids cached per (split tag, agent, example id)"""

    def __init__(self, ag, n_agents, max_ctx, k_ev, sel_layer, trunc=None):
        self.ag, self.N, self.max_ctx, self.k_ev, self.sel_layer = ag, int(n_agents), int(max_ctx) if max_ctx else None, int(k_ev), sel_layer
        self.doc_ctx = (self.max_ctx - (self.N - 1) * (self.k_ev + 12)) if self.max_ctx else None   # room for the N-1 windows and their hints
        self.trunc = trunc if trunc is not None else {}
        self.span_cache = {}
        self.n_attn = 0

    def _count(self, tr):
        self.trunc[tr] = self.trunc.get(tr, 0) + 1

    def pieces(self, batch, i, kind, target=None, msgs=None):
        """kind: 'own' | 'sender' | 'treceiver' (msgs[b] = {j: text}) ; target: 'answer' | 'sender' | None"""
        out = []
        for b, ex in enumerate(batch):
            if kind == 'own':
                build = lambda e, i=i: own_parts(e, i)                                        # noqa: E731
            elif kind == 'sender':
                build = lambda e, i=i: sender_parts(e, i)                                     # noqa: E731
            elif kind == 'treceiver':
                build = lambda e, i=i, m=msgs[b]: receiver_parts(e, i, m)                     # noqa: E731
            else:
                raise ValueError(kind)
            tgt = M.DATA.answer_target(ex) if target == 'answer' else M.DATA.text_sender_target(ex, i) if target == 'sender' else None
            pc, tr = fit_pieces(self.ag, ex, i, build, self.max_ctx, tgt)
            self._count(tr)
            out.append(pc)
        return out

    @torch.no_grad()
    def windows(self, batch, i, tag=''):
        """agent i's attention-selected window (token ids of its own documents) for every example (B0 adapter); cached by example id"""
        out = [None] * len(batch)
        need = []
        for b, ex in enumerate(batch):
            key = (tag, i, ex.get('id'))
            if ex.get('id') is not None and key in self.span_cache:
                out[b] = self.span_cache[key]
            else:
                need.append(b)
        if need:
            pcs, dl = [], []
            nb = 1 if self.ag.bos is not None else 0
            for b in need:
                (pre, suf, _), tr = fit_pieces(self.ag, batch[b], i, lambda e, i=i: doc_parts(e, i), self.doc_ctx)
                pcs.append((pre, suf, []))
                dl.append(len(pre) - nb)
            pos = self.ag.attn_positions(pcs, dl, self.sel_layer, self.k_ev)
            self.n_attn += len(need)
            for q, b in enumerate(need):
                seq = list(pcs[q][0]) + list(pcs[q][1])
                ids = [int(seq[int(p)]) for p in pos[q].tolist() if p >= 0]
                out[b] = ids
                if batch[b].get('id') is not None:
                    self.span_cache[(tag, i, batch[b]['id'])] = ids
        return out

    def all_windows(self, batch, tag=''):
        return [self.windows(batch, i, tag) for i in range(self.N)]       # [agent][b]

    def span_prompt(self, batch, i, wins, target=None):
        """round-1 prompt of agent i: [BOS] own documents | for every other agent j: hint_j + window_j + '\n' | question. wins[j][b] = agent j's
        window ids; -> pieces (pre, suf, ans)"""
        out = []
        for b, ex in enumerate(batch):
            tgt = M.DATA.answer_target(ex) if target == 'answer' else target
            (pre, suf, ans), tr = fit_pieces(self.ag, ex, i, lambda e, i=i: doc_parts(e, i), self.doc_ctx, tgt)
            self._count(tr)
            mid = []
            for j in range(self.N):
                if j == i:
                    continue
                mid += self.ag.ids(SPAN_PRE.format(j=j)) + list(wins[j][b]) + self.ag.ids(SPAN_POST)
            out.append((list(pre) + mid, list(suf), ans))
        return out

    def prior_prompt(self, ex):
        nb = [self.ag.bos] if self.ag.bos is not None else []
        return nb + self.ag.ids(PRIOR_PRE.format(q=ex['question']))


# ================================================================ model calls
def verify_pieces(ag, prompt, cand, cue):
    return (list(prompt), [], ([int(t) for t in ag.ids(' ' + cand)] if cand else []) + list(cue))


def verify_logits(ag, pieces, yn, train=False):
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


@torch.no_grad()
def mean_token_prob(ag, prompts, cand_ids, batch_size=16):
    """teacher-forced mean token probability of every candidate (excl. EOS; over the EOS alone for an empty candidate)"""
    ag.eval()
    out = []
    for lo in range(0, len(prompts), batch_size):
        pcs = [(list(p_), [], list(c) + [ag.eos]) for p_, c in zip(prompts[lo:lo + batch_size], cand_ids[lo:lo + batch_size])]
        A = ag.assemble(pcs, None, 'right')
        h = ag.decoder(inputs_embeds=A['embeds'], attention_mask=A['mask'], use_cache=False).last_hidden_state
        for b, (_p, _s, a) in enumerate(pcs):
            a0 = A['ans_start'][b]
            pos = torch.arange(a0 - 1, a0 - 1 + len(a), device=h.device)
            lp = torch.log_softmax(ag.lm_head(h[b, pos]).float(), -1)
            tl = lp[torch.arange(len(a), device=h.device), torch.tensor(a, device=h.device)]
            body = tl[:-1] if len(a) > 1 else tl
            out.append(float(body.mean().exp()))
    return out


@torch.no_grad()
def greedy(ag, pieces, role, max_new, want_logits=False):
    ag.use(role)
    texts, g = ag.generate(pieces, None, max_new, want_logits=want_logits, first_line=True)
    return texts, g


@torch.no_grad()
def gen_messages(ag, P, examples, budget, batch, deadline=None, log=None, label=''):
    """the text sender's messages of every agent -> dict(texts [N][n], token_ids [N][n], seconds_per_example (all N agents), n)"""
    ag.use('tsend')
    ag.eval()
    N = P.N
    ids = [[None] * len(examples) for _ in range(N)]
    t_all = 0.0
    for bi, idx in enumerate(M.chunks(list(range(len(examples))), batch)):
        M.check_deadline(deadline, f'messages {label} batch {bi}')
        b_ex = [examples[j] for j in idx]
        for i in range(N):
            pcs = P.pieces(b_ex, i, 'sender')
            M.sync(ag.device); t0 = time.perf_counter()
            _, g = ag.generate(pcs, None, budget, want_logits=False, first_line=False)
            M.sync(ag.device); t_all += time.perf_counter() - t0
            for b, j in enumerate(idx):
                ids[i][j] = [int(t) for t in g['ids'][b]]
        if log and (bi + 1) % 20 == 0:
            log(f'messages {label}: {(bi + 1) * batch} examples')
    texts = [[ag.tok.decode(r[:budget], skip_special_tokens=True).strip() for r in ids[i]] for i in range(N)]
    return dict(texts=texts, token_ids=ids, n_tokens=[[len(r) for r in ids[i]] for i in range(N)], seconds_per_example=t_all / max(1, len(examples)), n=len(examples), budget=budget)


def gold_message(ag, ex, j, budget):
    return ag.tok.decode(ag.ids(M.DATA.text_sender_target(ex, j))[:budget], skip_special_tokens=True).strip()


# ================================================================ training loop (soft stop + hard deadline)
def train_loop(ag, role, step_fn, stream, steps, args, device, log, label, deadline=None, soft_stop=None, lr=None):
    ag.use(role, train=True)
    ag.train()
    lr = args.lr_lora if lr is None else lr
    opt, sched, counts = M.build_optimizer(ag, lr, lr, steps, args.warmup, args.weight_decay)
    params = [p for g in opt.param_groups for p in g['params']]
    if counts['lora'] == 0 or counts['adapter'] != 0:
        raise RuntimeError(f'{label}: trainable params lora {counts["lora"]} adapter {counts["adapter"]}')
    log(f'{label}: {steps} steps, lr {lr} ({counts["lora"]:,} params of {role}), warm-up {args.warmup}, clip {args.clip}, soft stop {soft_stop}')
    use_ac = device.type == 'cuda'
    times, losses, hist = [], [], []
    t_start = time.time()
    done, stopped = 0, None
    for s in range(steps):
        M.check_deadline(deadline, f'{label} step {s}')
        if soft_stop and time.time() >= soft_stop:
            stopped = f'soft time stop at step {s}/{steps}'
            break
        batch = next(stream)
        M.sync(device); t0 = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
            loss, info = step_fn(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'{label}: non-finite loss at step {s}')
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        M.sync(device)
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
        done = s + 1
        if done % args.log_every == 0 or done == steps:
            mem = f', peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GB' if use_ac else ''
            extra = ' '.join(f'{k} {v}' for k, v in info.items() if k != 'loss_agent')
            log(f'{label} step {done}/{steps} loss {np.mean(losses[-args.log_every:]):.4f} agents {["%.3f" % x for x in info.get("loss_agent", [])]} lr {sched.get_last_lr()[0]:.2e} '
                f'gnorm {float(gn):.2f} {extra} {np.mean(times[-args.log_every:]):.3f} s/step{mem}')
            hist.append(dict(step=done, loss=float(np.mean(losses[-args.log_every:])), lr=float(sched.get_last_lr()[0])))
    if stopped:
        log(f'{label}: {stopped}')
    ag.eval()
    ag.use(role, train=False)
    return dict(steps_done=done, steps_planned=steps, seconds=time.time() - t_start, step_time_s=float(np.mean(times)) if times else None,
                loss_first100=float(np.mean(losses[:100])) if losses else None, loss_last100=float(np.mean(losses[-100:])) if losses else None, history=hist, stopped=stopped,
                param_counts=counts, complete=bool(done == steps))


# ================================================================ jobs
def subset_of(train, seed, n):
    idx = M.epoch_order(len(train), seed + 7)[:n]
    return idx, [train[j] for j in idx]


def job_tsend(C):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    ag.add_role('tsend')

    def step(batch):
        losses = [ag.loss(P.pieces(batch, i, 'sender', target='sender'))[0] for i in range(N)]
        return torch.stack(losses).mean(), dict(loss_agent=[float(l.detach()) for l in losses])
    tr = train_loop(ag, 'tsend', step, M.batch_stream(C['train'], a.batch, a.seed), a.steps, a, C['device'], log, f'tsend/N{N}/s{a.seed}', C['deadline'], C['soft_stop'])
    t1 = time.time()
    sub_idx, subset = subset_of(C['train'], a.seed, a.stage2_examples)
    sub_idx = [int(j) for j in sub_idx]
    gens = {s: gen_messages(ag, P, C[s], a.budget, a.eval_batch, C['deadline'], log, s) for s in ('val', 'fresh')}
    gens['subset'] = gen_messages(ag, P, subset, a.budget, a.eval_batch, C['deadline'], log, 'subset')
    msg_s = time.time() - t1
    torch.save(dict(job='tsend', role='tsend', n_agents=N, seed=a.seed, state=ag.role_state('tsend'), config=vars(a), train=dict(tr, history=None)), os.path.join(a.out, 'trainable.pt'))
    stats = {}
    for s, exs in (('val', C['val']), ('fresh', C['fresh']), ('subset', subset)):
        g = gens[s]
        f1 = [float(M.DATA.f1_score(g['texts'][i][j], M.DATA.text_sender_target(ex, i))) for i in range(N) for j, ex in enumerate(exs)]
        stats[s] = dict(n=g['n'], seconds_per_example=g['seconds_per_example'], n_tokens_mean=float(np.mean([x for r in g['n_tokens'] for x in r])) if exs else None,
                        chars_mean_per_message=float(np.mean([len(t) for r in g['texts'] for t in r])) if exs else None, message_f1_vs_gold=float(np.mean(f1)) if f1 else None)
    M.jdump(dict(job='tsend', n_agents=N, seed=a.seed, budget=a.budget, subset=dict(idx=sub_idx, example_ids=[ex.get('id') for ex in subset]), messages={s: gens[s] for s in ('val', 'fresh', 'subset')},
                 stats=stats), os.path.join(a.out, 'messages.json'))
    log(f'messages: VAL {stats["val"]}')
    return dict(train=tr, train_complete=tr['complete'], messages=stats, val=dict(sender_s_per_example=stats['val']['seconds_per_example'], message_f1_vs_gold=stats['val']['message_f1_vs_gold'],
                                                                                    chars_mean_per_message=stats['val']['chars_mean_per_message']),
                timings=dict(train_s=tr['seconds'], messages_s=msg_s))


def eval_text(C, msgs, examples, label):
    """every agent answers with the generated messages of the others -> readouts"""
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    ag.use('trecv')
    ag.eval()
    recs, t_gen = [], 0.0
    for bi, idx in enumerate(M.chunks(list(range(len(examples))), a.eval_batch)):
        M.check_deadline(C['deadline'], f'eval_text {label} batch {bi}')
        b_ex = [examples[j] for j in idx]
        tops = [[None] * len(b_ex) for _ in range(N)]
        for i in range(N):
            m = [{j: msgs['texts'][j][q] for j in range(N) if j != i} for q in idx]
            pcs = P.pieces(b_ex, i, 'treceiver', msgs=m)
            M.sync(ag.device); t0 = time.perf_counter()
            texts, g = ag.generate(pcs, None, a.max_new_tokens, want_logits=False, first_line=True)
            M.sync(ag.device); t_gen += time.perf_counter() - t0
            for b in range(len(b_ex)):
                tops[i][b] = texts[b]
        for b, ex in enumerate(b_ex):
            inf = informed_of(ex)
            e = [em(tops[i][b], ex['answer']) for i in range(N)]
            v = majority([tops[i][b] for i in range(N)])
            recs.append(dict(id=ex.get('id'), gold=ex['answer'], owner=owner_label(ex), tops=[tops[i][b] for i in range(N)], em_agent=e, vote=em(tops[v][b], ex['answer']),
                             either=max(e), informed_either=max(e[i] for i in inf), msg_chars=sum(len(msgs['texts'][j][idx[b]]) for j in range(N))))
        if log and (bi + 1) % 20 == 0:
            log(f'eval_text {label}: {len(recs)} examples')
    pct = lambda xs: (100.0 * float(np.mean(xs))) if len(xs) else None       # noqa: E731
    n = len(recs)
    out = dict(n=n, n_agents=N, em=pct([x for r in recs for x in r['em_agent']]), em_agent=[pct([r['em_agent'][i] for r in recs]) for i in range(N)], vote=pct([r['vote'] for r in recs]),
               either_correct=pct([r['either'] for r in recs]), informed_either=pct([r['informed_either'] for r in recs]),
               em_by_owner={o: dict(n=len([r for r in recs if r['owner'] == o]), em=pct([x for r in recs if r['owner'] == o for x in r['em_agent']])) for o in sorted({r['owner'] for r in recs})},
               timing=dict(answer_s_per_example=t_gen / max(1, n), sender_s_per_example=msgs['seconds_per_example'], total_s_per_example=t_gen / max(1, n) + msgs['seconds_per_example']),
               bytes=dict(message_bytes=float(np.mean([r['msg_chars'] for r in recs])) if n else None, wire_bytes=float(np.mean([r['msg_chars'] for r in recs])) * (N - 1) if n else None),
               truncation=dict(P.trunc))
    return out, recs


def job_trecv(C):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    if not a.tsend_from:
        raise SystemExit('--job trecv needs --tsend-from')
    info = load_role_ckpt(ag, a.tsend_from, 'tsend', 'tsend', a.seed, N)
    mj = M.jload(os.path.join(a.tsend_from, 'messages.json'))
    if int(mj.get('n_agents', -1)) != N or int(mj.get('seed', -1)) != a.seed or int(mj.get('budget', -1)) != a.budget:
        raise SystemExit(f'{a.tsend_from}/messages.json: N / seed / budget mismatch')
    sub = mj['subset']
    if any(j >= len(C['train']) for j in sub['idx']) or [C['train'][j].get('id') for j in sub['idx']] != sub['example_ids']:
        raise SystemExit(f'{a.tsend_from}/messages.json: the stage-2 subset does not match --train')
    subset = [C['train'][j] for j in sub['idx']]
    pos = {ex.get('id'): q for q, ex in enumerate(subset)}
    info['messages_sha256'] = M.file_sha256(os.path.join(a.tsend_from, 'messages.json'))
    ag.add_role('trecv')
    gold_cache = {}

    def gold_for(ex, j):
        key = (ex.get('id'), j)
        if key not in gold_cache:
            gold_cache[key] = gold_message(ag, ex, j, a.budget)
        return gold_cache[key]

    def stage1(batch):
        losses = []
        for i in range(N):
            m = [{j: gold_for(ex, j) for j in range(N) if j != i} for ex in batch]
            losses.append(ag.loss(P.pieces(batch, i, 'treceiver', target='answer', msgs=m))[0])
        return torch.stack(losses).mean(), dict(loss_agent=[float(l.detach()) for l in losses])

    def stage2(batch):
        losses = []
        for i in range(N):
            m = [{j: mj['messages']['subset']['texts'][j][pos[ex.get('id')]] for j in range(N) if j != i} for ex in batch]
            losses.append(ag.loss(P.pieces(batch, i, 'treceiver', target='answer', msgs=m))[0])
        return torch.stack(losses).mean(), dict(loss_agent=[float(l.detach()) for l in losses])
    tr1 = train_loop(ag, 'trecv', stage1, M.batch_stream(C['train'], a.batch, a.seed + 1), a.steps, a, C['device'], log, f'trecv-stage1/N{N}/s{a.seed}', C['deadline'], C['soft_stop'])
    tr2 = train_loop(ag, 'trecv', stage2, M.batch_stream(subset, a.batch, a.seed + 2), a.stage2_steps, a, C['device'], log, f'trecv-stage2/N{N}/s{a.seed}', C['deadline'], C['soft_stop'])
    torch.save(dict(job='trecv', role='trecv', n_agents=N, seed=a.seed, state=ag.role_state('trecv'), config=vars(a), tsend_from=info), os.path.join(a.out, 'trainable.pt'))
    t_eval = time.time()
    v, v_rec = eval_text(C, mj['messages']['val'], C['val'], 'val')
    log(f'VAL text: em {v["em"]:.2f} agents {[round(x, 2) for x in v["em_agent"]]} vote {v["vote"]:.2f} either {v["either_correct"]:.2f} | {v["timing"]} | {v["bytes"]}')
    f, f_rec = eval_text(C, mj['messages']['fresh'], C['fresh'], 'fresh')
    log(f'fresh text: em {f["em"]:.2f} agents {[round(x, 2) for x in f["em_agent"]]} vote {f["vote"]:.2f}')
    if a.keep_preds:
        M.jdump(v_rec, os.path.join(a.out, 'preds_val.json')); M.jdump(f_rec, os.path.join(a.out, 'preds_fresh.json'))
    return dict(train=dict(stage1=dict(tr1, history=None), stage2=dict(tr2, history=None)), train_complete=bool(tr1['complete'] and tr2['complete']), tsend_from=info, val=v, fresh=f,
                channel=dict(medium='text', budget=a.budget, sender='LoRA tsend: supporting sentences of the own gold paragraph (L1 T)', receiver=f'LoRA trecv: reads the {N - 1} messages "Agent j says:"'),
                timings=dict(train_s=tr1['seconds'] + tr2['seconds'], eval_s=time.time() - t_eval))


@torch.no_grad()
def round1_answers(C, examples, tag, deadline=None, log=None, label=''):
    """windows of every agent + every agent's greedy answer with the recv adapter -> (wins {id: [N lists]}, cands {id: [N strings]}, cand_ids, timing)"""
    a, ag, P, N = C['args'], C['ag'], C['P'], C['N']
    wins, cands, cids = {}, {}, {}
    t_w = t_g = 0.0
    for bi, batch in enumerate(M.chunks(examples, a.gen_batch)):
        M.check_deadline(deadline, f'round1 {label} batch {bi}')
        M.sync(ag.device); t0 = time.perf_counter()
        W = P.all_windows(batch, tag)
        M.sync(ag.device); t_w += time.perf_counter() - t0
        t0 = time.perf_counter()
        tops = []
        for i in range(N):
            pcs = P.span_prompt(batch, i, W)
            texts, _ = greedy(ag, pcs, 'recv', a.max_new_tokens)
            tops.append(texts)
        M.sync(ag.device); t_g += time.perf_counter() - t0
        for b, ex in enumerate(batch):
            wins[ex['id']] = [[int(t) for t in W[i][b]] for i in range(N)]
            cands[ex['id']] = [[tops[i][b]] for i in range(N)]
            cids[ex['id']] = [[int(t) for t in ag.ids(' ' + tops[i][b])] if tops[i][b] else [] for i in range(N)]
        if log and (bi + 1) % 25 == 0:
            log(f'round1 {label}: {len(cands)} examples; s/example windows {t_w / len(cands):.3f} answers {t_g / len(cands):.3f}')
    n = max(1, len(cands))
    return wins, cands, cids, dict(windows_s_per_example=t_w / n, gen_s_per_example=t_g / n)


def job_recv(C):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    n_copied = ag.add_role('recv', init_from='default')
    log(f'recv adapter: {n_copied} tensors copied from B0')

    def step(batch):
        W = P.all_windows(batch, 'train')
        losses = [ag.loss(P.span_prompt(batch, i, W, target='answer'))[0] for i in range(N)]
        return torch.stack(losses).mean(), dict(loss_agent=[float(l.detach()) for l in losses])
    tr = train_loop(ag, 'recv', step, M.batch_stream(C['train'], a.batch, a.seed + 3), a.steps, a, C['device'], log, f'recv/N{N}/s{a.seed}', C['deadline'], C['soft_stop'], lr=a.lr_recv)
    P.span_cache.clear()
    torch.save(dict(job='recv', role='recv', n_agents=N, seed=a.seed, state=ag.role_state('recv'), config=vars(a), train=dict(tr, history=None)), os.path.join(a.out, 'trainable.pt'))
    t_eval = time.time()
    _w, cands, _c, tm = round1_answers(C, C['val'], 'val', C['deadline'], log, 'val')
    pct = lambda xs: (100.0 * float(np.mean(xs))) if len(xs) else None       # noqa: E731
    e = {ex['id']: [em(cands[ex['id']][i][0], ex['answer']) for i in range(N)] for ex in C['val']}
    v = dict(n=len(e), em_agent=[pct([e[k][i] for k in e]) for i in range(N)], either_correct=pct([max(e[k]) for k in e]), em=pct([x for k in e for x in e[k]]), timing=tm)
    log(f'VAL recv round 1: agents {[round(x, 2) for x in v["em_agent"]]} either {v["either_correct"]:.2f} | {tm}')
    return dict(train=tr, train_complete=tr['complete'], val=v, timings=dict(train_s=tr['seconds'], eval_s=time.time() - t_eval))


def job_gen(C):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    if not a.recv_from:
        raise SystemExit('--job gen needs --recv-from')
    info = load_role_ckpt(ag, a.recv_from, 'recv', 'recv', a.seed, N)
    rows = []
    t0 = time.time()
    complete = True
    try:
        for bi, batch in enumerate(M.chunks(C['train'], a.gen_batch)):
            M.check_deadline(C['deadline'], f'gen batch {bi}')
            W = P.all_windows(batch, 'train')
            tops = []
            for i in range(N):
                texts, _ = greedy(ag, P.span_prompt(batch, i, W), 'recv', a.max_new_tokens)
                tops.append(texts)
            for b, ex in enumerate(batch):
                cb = [[tops[i][b]] for i in range(N)]
                rows.append(dict(id=ex['id'], gold=ex['answer'], owner=owner_label(ex), gold_shards=informed_of(ex), cands=cb, union=union(cb), wins=[[int(t) for t in W[i][b]] for i in range(N)]))
            P.span_cache.clear()
            if log and (bi + 1) % 50 == 0:
                log(f'gen: {len(rows)} rows; {(time.time() - t0) / len(rows):.3f} s/example')
    except M.Deadline as e:
        complete = False
        log(f'gen: deadline after {len(rows)} rows ({e}); the rows so far are written, complete=false')
    gen_s = time.time() - t0
    stats = dict(n=len(rows), union_any_correct=100.0 * float(np.mean([max(em(c, r['gold']) for c in r['union']) for r in rows])) if rows else None,
                 either_top1=100.0 * float(np.mean([max(em(r['cands'][i][0], r['gold']) for i in range(N)) for r in rows])) if rows else None,
                 size_mean=float(np.mean([len(r['union']) for r in rows])) if rows else None,
                 positive_rate=100.0 * float(np.mean([em(c, r['gold']) for r in rows for c in r['union']])) if rows else None, s_per_example=gen_s / max(1, len(rows)))
    M.jdump(dict(job='gen', n_agents=N, seed=a.seed, n=len(rows), complete=complete, k_ev=a.k_ev, sel_layer=a.sel_layer, recv_from=info, rows=rows), os.path.join(a.out, 'cands_train.json'))
    log(f'gen: {stats}')
    return dict(n_rows=len(rows), gen=stats, recv_from=info, timings=dict(gen_s=gen_s)), complete


def train_verifier(C, rows, by_id, cue, yn):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    n = len(rows)
    steps = int(a.steps) if a.steps else int(math.ceil(n / a.batch))
    order = M.epoch_order(n, a.seed)
    rng = np.random.RandomState(a.seed + 11)
    stats = dict(pos=[], neg=[], n_seq=0)
    state = dict(s=0)

    def stream():
        while True:
            lo = (state['s'] * a.batch) % n
            yield [rows[order[(lo + j) % n]] for j in range(a.batch)]

    def step(batch_rows):
        state['s'] += 1
        seqs = [(ai, c, lab, r) for r in batch_rows for (ai, c, lab) in pick_pairs(r, N, rng)]
        total, cnt = None, 0
        loss_agent = []
        for i in range(N):
            mine = [(c, lab, r) for (ai, c, lab, r) in seqs if ai == i]
            if not mine:
                loss_agent.append(0.0)
                continue
            b_ex = [by_id[r['id']] for (_c, _l, r) in mine]
            W = [[r['wins'][j] for (_c, _l, r) in mine] for j in range(N)]
            prompts = [list(p) + list(s) for (p, s, _a) in P.span_prompt(b_ex, i, W)]
            pcs = [verify_pieces(ag, pr, c, cue) for pr, (c, _l, _r) in zip(prompts, mine)]
            sc = verify_logits(ag, pcs, yn, train=True)
            y = torch.tensor([lab for (_c, lab, _r) in mine], dtype=torch.float32, device=sc.device)
            li = Fn.binary_cross_entropy_with_logits(sc, y, reduction='sum')
            total = li if total is None else total + li
            cnt += len(mine)
            loss_agent.append(float(li.detach()) / len(mine))
            with torch.no_grad():
                pr_ = torch.sigmoid(sc.detach())
                stats['pos'].extend(pr_[y == 1].tolist()); stats['neg'].extend(pr_[y == 0].tolist()); stats['n_seq'] += len(mine)
        loss = total / max(1, cnt)
        return loss, dict(loss_agent=loss_agent, p_pos=round(float(np.mean(stats['pos'][-500:])), 3) if stats['pos'] else None, p_neg=round(float(np.mean(stats['neg'][-2000:])), 3) if stats['neg'] else None, seq=stats['n_seq'])
    tr = train_loop(ag, 'verify', step, stream(), steps, a, C['device'], log, f'verify/N{N}/s{a.seed}', C['deadline'], C['soft_stop'], lr=a.lr_verify)
    tr['sequences'] = stats['n_seq']
    tr['rows'] = n
    tr['batch_rows'] = a.batch
    return tr


@torch.no_grad()
def eval_belief(C, examples, cue, yn, label):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    wins, cands, cids, tm = round1_answers(C, examples, label, C['deadline'], log, label)
    rows, records = [], []
    t_v = t_pr = t_p = t_own = 0.0
    for bi, batch in enumerate(M.chunks(examples, a.gen_batch)):
        M.check_deadline(C['deadline'], f'eval_belief {label} batch {bi}')
        Cs = [union(cands[ex['id']]) for ex in batch]
        W = [[wins[ex['id']][j] for ex in batch] for j in range(N)]
        # the verifier under every agent's round-1 prompt
        M.sync(ag.device); t0 = time.perf_counter()
        ag.use('verify')
        S = [[None] * len(batch) for _ in range(N)]
        for i in range(N):
            prompts = [list(p) + list(s) for (p, s, _a) in P.span_prompt(batch, i, W)]
            Pq, Cq, where = [], [], []
            for b in range(len(batch)):
                for ci, c in enumerate(Cs[b]):
                    Pq.append(prompts[b]); Cq.append(c); where.append((b, ci))
            got = score_verify(ag, Pq, Cq, cue, yn, a.score_batch)
            for (b, ci), sc in zip(where, got):
                if S[i][b] is None:
                    S[i][b] = [None] * len(Cs[b])
                S[i][b][ci] = sc
        M.sync(ag.device); t_v += time.perf_counter() - t0
        # the no-document prior s_0(c) (readout)
        t0 = time.perf_counter()
        Pq, Cq, where = [], [], []
        for b, ex in enumerate(batch):
            pp = P.prior_prompt(ex)
            for ci, c in enumerate(Cs[b]):
                Pq.append(pp); Cq.append(c); where.append((b, ci))
        got = score_verify(ag, Pq, Cq, cue, yn, a.score_batch)
        S0 = [[None] * len(Cs[b]) for b in range(len(batch))]
        for (b, ci), sc in zip(where, got):
            S0[b][ci] = sc
        M.sync(ag.device); t_pr += time.perf_counter() - t0
        # select-by-p: the mean token probability of every agent's own top-1 under its round-1 prompt (recv adapter)
        t0 = time.perf_counter()
        ag.use('recv')
        p = [[None] * len(batch) for _ in range(N)]
        for i in range(N):
            prompts = [list(pr) + list(s) for (pr, s, _a) in P.span_prompt(batch, i, W)]
            got = mean_token_prob(ag, prompts, [cids[ex['id']][i] for ex in batch], a.eval_batch)
            for b in range(len(batch)):
                p[i][b] = got[b]
        M.sync(ag.device); t_p += time.perf_counter() - t0
        # the no-evidence arm B': own-document answers (B0) and scores under the own-document prompt (verify)
        t0 = time.perf_counter()
        own = [[None] * len(batch) for _ in range(N)]
        own_pcs = {}
        for i in range(N):
            pcs = P.pieces(batch, i, 'own')
            own_pcs[i] = pcs
            texts, _ = greedy(ag, pcs, 'default', a.max_new_tokens)
            for b in range(len(batch)):
                own[i][b] = [texts[b]]
        C_own = [union([own[i][b] for i in range(N)]) for b in range(len(batch))]
        ag.use('verify')
        S_own = [[None] * len(batch) for _ in range(N)]
        for i in range(N):
            prompts = [list(pr) + list(s) for (pr, s, _a) in own_pcs[i]]
            Pq, Cq, where = [], [], []
            for b in range(len(batch)):
                for ci, c in enumerate(C_own[b]):
                    Pq.append(prompts[b]); Cq.append(c); where.append((b, ci))
            got = score_verify(ag, Pq, Cq, cue, yn, a.score_batch)
            for (b, ci), sc in zip(where, got):
                if S_own[i][b] is None:
                    S_own[i][b] = [None] * len(C_own[b])
                S_own[i][b][ci] = sc
        M.sync(ag.device); t_own += time.perf_counter() - t0
        for b, ex in enumerate(batch):
            cb = cands[ex['id']]
            j = judge(ex, cb, Cs[b], [S[i][b] for i in range(N)], S0[b], [p[i][b] for i in range(N)], [own[i][b] for i in range(N)], C_own[b], [S_own[i][b] for i in range(N)])
            j['bytes'] = belief_bytes(N, a.k_ev, cb, Cs[b])
            rows.append(j)
            records.append(dict(id=ex['id'], gold=ex['answer'], owner=j['owner'], informed=informed_of(ex), cands=cb, union=Cs[b], p=[p[i][b] for i in range(N)],
                                ver_scores=[[round(x, 4) for x in S[i][b]] for i in range(N)], prior_scores=[round(x, 4) for x in S0[b]],
                                own_cands=[own[i][b] for i in range(N)], own_union=C_own[b], own_scores=[[round(x, 4) for x in S_own[i][b]] for i in range(N)],
                                picks={kk: Cs[b][ci] for kk, ci in j['picks'].items()}, ok=j['ok']))
        if log and (bi + 1) % 25 == 0:
            log(f'eval_belief {label}: {len(rows)} examples; ver_sum {100.0 * np.mean([r["ok"]["ver_sum"] for r in rows]):.2f} select_p {100.0 * np.mean([r["ok"]["select_p"] for r in rows]):.2f} '
                f'vote {100.0 * np.mean([r["ok"]["vote"] for r in rows]):.2f}; verify s/example {t_v / len(rows):.3f}')
    n = max(1, len(rows))
    out = aggregate(rows, N)
    out['timing'] = dict(windows_s_per_example=tm['windows_s_per_example'], gen_s_per_example=tm['gen_s_per_example'], verify_s_per_example=t_v / n,
                         total_s_per_example=tm['windows_s_per_example'] + tm['gen_s_per_example'] + t_v / n, prior_s_per_example=t_pr / n, p_s_per_example=t_p / n, own_arm_s_per_example=t_own / n)
    out['truncation'] = dict(P.trunc)
    out['examples'] = [dict(gold=r['gold'], top=[c[0] for c in r['cands']], ver_sum=r['picks']['ver_sum'], s=r['ver_scores']) for r in records[:6]]
    return out, records


def job_ver(C):
    a, ag, P, log, N = C['args'], C['ag'], C['P'], C['log'], C['N']
    if not (a.recv_from and a.cands_from):
        raise SystemExit('--job ver needs --recv-from and --cands-from')
    info = load_role_ckpt(ag, a.recv_from, 'recv', 'recv', a.seed, N)
    d = M.jload(a.cands_from)
    if int(d.get('seed', -1)) != a.seed or int(d.get('n_agents', -1)) != N:
        raise SystemExit(f'--cands-from seed / N mismatch: {d.get("seed")} / {d.get("n_agents")}')
    rows = d['rows'][:a.limit_train] if a.limit_train else d['rows']
    cands_meta = dict(path=a.cands_from, n_rows=len(rows), n_total=len(d['rows']), complete=d.get('complete'), sha256=M.file_sha256(a.cands_from))
    by_id = {ex['id']: ex for ex in C['train']}
    miss = [r['id'] for r in rows if r['id'] not in by_id]
    if miss:
        raise SystemExit(f'{len(miss)} training rows are not in --train, e.g. {miss[:3]}')
    n_copied = ag.add_role('verify', init_from='recv')
    yn = (int(ag.ids(YES)[0]), int(ag.ids(NO)[0]))
    cue = [int(t) for t in ag.ids(VERIFY_CUE)]
    if yn[0] == yn[1]:
        log(f'WARNING: yes / no share the token {yn[0]} (tiny vocabulary?)')
    log(f'verify adapter: {n_copied} tensors copied from recv; yes/no ids {yn}; cue tokens {len(cue)}; training rows {len(rows)} (of {len(d["rows"])}, complete {d.get("complete")})')
    tr = train_verifier(C, rows, by_id, cue, yn)
    log(f'train done: {tr["steps_done"]}/{tr["steps_planned"]} steps in {tr["seconds"]:.0f} s ({tr["sequences"]} sequences); loss first100 {tr["loss_first100"]} last100 {tr["loss_last100"]}; {tr["stopped"]}')
    torch.save(dict(job='ver', role='verify', n_agents=N, seed=a.seed, state=ag.role_state('verify'), config=vars(a), train=dict(tr, history=None), recv_from=info, cands=cands_meta),
               os.path.join(a.out, 'trainable.pt'))
    t_eval = time.time()
    v, v_rec = eval_belief(C, C['val'], cue, yn, 'val')
    log(f'VAL: ver_sum {v["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in v["em_variants"].items())} | round1 {v["round1"]} | union {v["union"]} | know {v["know"]} | '
        f'verifier {v["verifier"]} | own {v["own"]} | bytes {v.get("bytes")} | {v["timing"]}')
    f, f_rec = eval_belief(C, C['fresh'], cue, yn, 'fresh')
    log(f'fresh: ver_sum {f["em"]:.2f} | variants {dict((k, round(x, 2)) for k, x in f["em_variants"].items())} | either {f["round1"]["either_correct"]:.2f} union {f["union"]["any_correct"]:.2f}')
    if a.keep_preds:
        M.jdump(v_rec, os.path.join(a.out, 'preds_val.json')); M.jdump(f_rec, os.path.join(a.out, 'preds_fresh.json'))
    return dict(train=dict(tr, history=None), train_complete=tr['complete'], steps=tr['steps_done'], steps_planned=tr['steps_planned'], recv_from=info, cands=cands_meta, val=v, fresh=f,
                channel=dict(medium='belief', k_ev=a.k_ev, sel_layer=a.sel_layer, span_pre=SPAN_PRE, cue=VERIFY_CUE, yes_no_ids=list(yn),
                             round1=f'every agent reads the {N - 1} attention-selected {a.k_ev}-token windows (token ids) of the others with the recv LoRA and answers greedily',
                             round2='candidate union of the N greedy answers; s_i(c) = logit(yes) - logit(no) under agent i\'s round-1 prompt (verify LoRA); consensus argmax sum_i s_i(c)',
                             training=f'BCE on EM(c, gold), 2 (agent, candidate) sequences per row, {len(rows)} rows'),
                timings=dict(train_s=tr['seconds'], eval_s=time.time() - t_eval))


# ================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    ap.add_argument('--job', choices=JOBS)
    ap.add_argument('--n-agents', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--train'); ap.add_argument('--val'); ap.add_argument('--fresh')
    ap.add_argument('--out')
    ap.add_argument('--b0-from', default=None, help='the B0 run dir of this seed (runs/l1/B0-hetero-s<seed>): the frozen default LoRA; required unless --tiny')
    ap.add_argument('--tsend-from', default=None); ap.add_argument('--recv-from', default=None); ap.add_argument('--cands-from', default=None)
    ap.add_argument('--sel-layer', type=int, default=None, help='the attention layer of the window selection (the V1 value)')
    ap.add_argument('--k-ev', type=int, default=K_EV)
    ap.add_argument('--budget', type=int, default=BUDGET)
    ap.add_argument('--steps', type=int, default=2500, help='tsend: sender steps; trecv: stage-1 steps; recv: steps; ver: 0 = one pass over the rows')
    ap.add_argument('--stage2-steps', type=int, default=625); ap.add_argument('--stage2-examples', type=int, default=STAGE2_EXAMPLES)
    ap.add_argument('--batch', type=int, default=8, help='examples per step (ver: rows per step, 2 sequences each)')
    ap.add_argument('--lr-lora', type=float, default=2e-4, help='tsend / trecv (L1 T)'); ap.add_argument('--lr-recv', type=float, default=1e-4, help='recv (V1 stage 2)')
    ap.add_argument('--lr-verify', type=float, default=1e-4, help='verify (D8-96)')
    ap.add_argument('--warmup', type=int, default=100); ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--eval-batch', type=int, default=16); ap.add_argument('--gen-batch', type=int, default=8); ap.add_argument('--score-batch', type=int, default=32)
    ap.add_argument('--max-new-tokens', type=int, default=M.ANSWER_MAX_NEW)
    ap.add_argument('--max-ctx-tokens', type=int, default=900)
    ap.add_argument('--limit-train', type=int, default=0); ap.add_argument('--limit-eval', type=int, default=0)
    ap.add_argument('--train-stop-epoch', type=float, default=0.0, help='soft stop of the training loops (epoch seconds); the evaluation follows')
    ap.add_argument('--no-grad-ckpt', action='store_true')
    ap.add_argument('--layer-frac', type=float, default=M.LAYER_FRAC)
    ap.add_argument('--lora-r', type=int, default=M.LORA_R); ap.add_argument('--lora-alpha', type=int, default=M.LORA_ALPHA); ap.add_argument('--lora-dropout', type=float, default=0.0)
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--hf-cache', default=None)
    ap.add_argument('--deadline-epoch', type=float, default=0.0)
    ap.add_argument('--keep-preds', action='store_true')
    ap.add_argument('--allow-fallback-data', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    for k in ('job', 'n_agents', 'seed', 'train', 'val', 'fresh', 'out'):
        if getattr(args, k) is None:
            raise SystemExit(f'--{k.replace("_", "-")} is required')
    if not args.tiny and not args.b0_from:
        raise SystemExit('--b0-from is required outside --tiny')
    if args.job in ('recv', 'gen', 'ver') and args.sel_layer is None:
        raise SystemExit('--sel-layer is required for recv / gen / ver')
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    sp = os.path.join(args.out, 'summary.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} exists (never overwrite)')
    log = M.Log(os.path.join(args.out, 'log.txt'))
    log(f'l7_nagents start pid {os.getpid()} host {platform.node()} job {args.job} N {args.n_agents} seed {args.seed} out {args.out}')
    if M.DATA_IS_FALLBACK and not (args.tiny or args.allow_fallback_data):
        raise SystemExit('l1_data.py not importable: a formal run needs it')
    M.seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = int(args.n_agents)
    need_train = args.job in ('tsend', 'trecv', 'recv', 'gen', 'ver')
    train = M.DATA.read_jsonl(args.train) if need_train else []
    val, fresh = M.DATA.read_jsonl(args.val), M.DATA.read_jsonl(args.fresh)
    if args.limit_train and args.job != 'ver':
        train = train[:args.limit_train]
    if args.limit_eval:
        val, fresh = val[:args.limit_eval], fresh[:args.limit_eval]
    for name, exs in (('train', train), ('val', val), ('fresh', fresh)):
        bad = [ex.get('id') for ex in exs if len(ex.get('shards', [])) != N]
        if bad:
            raise SystemExit(f'{name}: {len(bad)} examples do not have {N} shards, e.g. {bad[:3]}')
    log(f'data: train {len(train)} val {len(val)} fresh {len(fresh)}; N {N}')
    tiny = M.tiny_models(train + val + fresh + [TINY_WORDS], args.seed) if args.tiny else None
    deadline = args.deadline_epoch or None
    soft_stop = args.train_stop_epoch or None
    try:
        t0 = time.time()
        ag, b0 = build_agent(args, tiny, device)
        load_s = time.time() - t0
        P = Prompter(ag, N, args.max_ctx_tokens, args.k_ev, args.sel_layer)
        log(f'model: {ag.name} layers {ag.n_layers} d {ag.d_model} bos {ag.bos} eos {ag.eos}; B0 {b0}; doc ctx {P.doc_ctx} (max {args.max_ctx_tokens})')
        C = dict(args=args, ag=ag, P=P, log=log, N=N, train=train, val=val, fresh=fresh, device=device, deadline=deadline, soft_stop=soft_stop)
        complete = True
        if args.job == 'tsend':
            res = job_tsend(C)
        elif args.job == 'trecv':
            res = job_trecv(C)
        elif args.job == 'recv':
            res = job_recv(C)
        elif args.job == 'gen':
            res, complete = job_gen(C)
        else:
            res = job_ver(C)
    except M.Deadline as e:
        M.jdump(dict(incomplete=True, reason=str(e), utc=M.utc()), os.path.join(args.out, 'INCOMPLETE.json'))
        log(f'L7 INCOMPLETE: {e}')
        return 3
    summary = dict(job=args.job, arm=args.job.upper(), n_agents=N, seed=args.seed, batch_name=BATCH, prereg=PREREG, complete=complete, b0=b0, model=ag.name, **res)
    summary['timings'] = dict(res.get('timings', {}), load_s=load_s, total_s=time.time() - t_start)
    if device.type == 'cuda':
        summary['timings']['peak_gpu_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    summary.update(config=dict(vars(args), lora_targets=list(M.LORA_TARGETS), device=str(device), torch=torch.__version__, gpu=(torch.cuda.get_device_name(device) if device.type == 'cuda' else None)),
                   scripts_sha=M.scripts_sha(SCRIPTS), utc_end=M.utc(), pid=os.getpid())
    M.jdump(summary, sp)
    head = {'tsend': lambda r: f'sender s/example {r["val"]["sender_s_per_example"]:.3f} msg F1 {r["val"]["message_f1_vs_gold"]}',
            'trecv': lambda r: f'VAL text em {r["val"]["em"]:.2f} vote {r["val"]["vote"]:.2f} fresh {r["fresh"]["em"]:.2f}',
            'recv': lambda r: f'VAL round-1 agents {[round(x, 2) for x in r["val"]["em_agent"]]} either {r["val"]["either_correct"]:.2f}',
            'gen': lambda r: f'{r["n_rows"]} rows; union any {r["gen"]["union_any_correct"]} either {r["gen"]["either_top1"]} complete {complete}',
            'ver': lambda r: f'VAL ver_sum {r["val"]["em"]:.2f} (vote {r["val"]["em_variants"]["vote"]:.2f} select_p {r["val"]["em_variants"]["select_p"]:.2f} union {r["val"]["union"]["any_correct"]:.2f}) fresh {r["fresh"]["em"]:.2f}'}
    log(f'l7_nagents done: {args.job.upper()}-N{N}-s{args.seed} {head[args.job](res)}; total {time.time() - t_start:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
