"""l1d_model.py (Claude, 2026-10-31): batch L1d model side (pre-registration D8-87) on top of l1c_model.py (SenderAdapter2, Decoder, the PairC
evaluation path; itself on top of the PINNED L1 files, which are imported and never edited). What changes against L1c (D8-87 section 3):
  (a) the sender's pass 1 reads the hidden layer of its OWN frozen B0 model (LoRA ENABLED) over "B0 prompt + the answer that model just generated
      greedily" (<= 16 tokens, first line, cached per example id); the adapter (K = 8 slot queries) turns those states into z_j, lam_j;
  (b) the message target is the GOLD ANSWER in the receiver's tokenizer (leading space), cut to K and filler-padded; the same target for Du / Ds;
  (c) the receiver's prompt is documents_i | HINT_PRE | K slots | HINT_POST | question (the template tokens are real embeddings);
  (d) the criterion rendering is HARD (argmax tokens = the latent decoded into the receiver's vocabulary and read as text); soft = readout;
  (e) arms Du (one-way projection, 30 % training dropout, lam = 1) and Ds (GBP consensus, learned lam / w).
Readouts: the decoded hint against the gold answer and against the sender's own answer by owner bucket, the sender's own B0 accuracy by bucket,
lam by bucket (Ds), the baselines none (the plain B0 prompt) and gold_hint (the gold answer tokens inside the template).
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402
import l1c_model as MC  # noqa: E402

BATCH = 'L1d'
PREREG = 'D8-87'
ARMS = ('Du', 'Ds')
BASE_ARM = {'Du': 'U', 'Ds': 'S1'}
K_HINT = 8                                      # slots = answer tokens (answers average 2-3 tokens)
HINT_PRE = 'Hint from the other agent: the answer is'
HINT_POST = '.\n'
ABSENCE_PS = (0.0, 1.0)
BASELINES = ('none', 'gold_hint')
SCRIPTS = ('l1d_model.py', 'l1d_train.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
HINT_EXAMPLES = 12


# ================================================================ the decoder with a default rendering
class DecoderD(MC.Decoder):
    """l1c_model.Decoder whose rendering, when the caller does not say, is `default_hard` (set by eval_render): the pinned absence-p1 detail
    calls reader(z, dtype) without the flag and must compare like with like."""

    def __init__(self, d_e, embed_weight, tau=MC.TAU):
        super().__init__(d_e, embed_weight, tau)
        self.default_hard = False

    def forward(self, x, dtype=None, hard=None):
        return super().forward(x, dtype, self.default_hard if hard is None else hard)


# ================================================================ (a) (b) the agent
class AgentD(MC.AgentC):
    """l1c_model.AgentC whose pass 1 is the B0 model (LoRA enabled, frozen) over its own prompt + its own greedy answer, and whose message
    target is the gold answer. `answer_cache`: example id -> (answer text, this agent's own tokenizer ids of ' ' + text, fed back into its pass 1)."""

    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, **kw)
        self.reader = DecoderD(self.d_e, self.embed.weight, kw.get('tau', MC.TAU)).to(self.device)
        self.hint_pre = [int(t) for t in self.ids(HINT_PRE)]
        self.hint_post = [int(t) for t in self.ids(HINT_POST)]
        self.answer_cache = {}
        self.gen_max_new = M.ANSWER_MAX_NEW
        self.n_generated = 0

    @torch.no_grad()
    def own_answers(self, batch, pieces_b0):
        """the B0 model's greedy answers (first line, cut at EOS) for the batch, cached by example id -> list of (text, ids)"""
        out = [None] * len(batch)
        need = []
        for b, ex in enumerate(batch):
            key = ex.get('id')
            if key is not None and key in self.answer_cache:
                out[b] = self.answer_cache[key]
            else:
                need.append(b)
        if need:
            texts, _ = self.generate([pieces_b0[b] for b in need], None, self.gen_max_new, want_logits=False)
            self.n_generated += len(need)
            for b, t in zip(need, texts):
                ids = [int(x) for x in self.ids(' ' + t)] if t else []
                out[b] = (t, ids)
                if batch[b].get('id') is not None:
                    self.answer_cache[batch[b]['id']] = out[b]
        return out

    def encode(self, pieces):
        """pass 1 on [BOS?] B0 prompt + own answer ids with the LoRA ENABLED (the agent's B0 model), no gradient -> z, lam (grad into the adapter)"""
        A = self.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
        with torch.no_grad():
            out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.layer].detach()
        self.freeze_lora()
        z, log_lam = self.adapter(h, A['mask'])
        return z, log_lam.exp()

    def target_ids(self, ex, partner=None):
        """-> (K receiver-token ids of ' ' + the gold answer, cut to K and right-padded with the filler; the length before padding)"""
        ids = [int(t) for t in self.ids(' ' + ex['answer'])[:self.k]]
        return ids + [self.filler] * (self.k - len(ids)), len(ids)


# ================================================================ (c) (e) the pair
class PairD(MC.PairC):
    """PairC of AgentD's: the latent prompt wraps the K slots into the hint template; pass 1 runs on the B0 prompt + own answer."""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair != 'hetero':
            raise ValueError('PairD supports pair hetero only (one answer cache per agent object; D8-87 runs hetero)')
        M.Pair.__init__(self, agents, pair, BASE_ARM[arm], max_ctx=max_ctx)
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []

    def pieces(self, batch, i, mode, target='answer', messages=None):
        if mode != 'latent':
            return M.Pair.pieces(self, batch, i, mode, target, messages)
        ag = self.agent(i)
        out = []
        for ex in batch:                                                       # the documents cap is max_ctx; the template (~12 tokens) sits on top (D8-87 section 2)
            (pre, suf, ans), tr = M.make_pieces(ag, ex, i, 'latent', self.max_ctx, target, None)
            self.trunc[tr] += 1
            out.append((list(pre) + ag.hint_pre, ag.hint_post + list(suf), ans))
        return out

    def sender_pieces(self, batch, i):
        """-> (B0 pieces (pre = the whole B0 prompt), answers [(text, ids)]) of agent i"""
        pb0 = M.Pair.pieces(self, batch, i, 'B0', target=None)
        answers = self.agent(i).own_answers(batch, pb0)
        return pb0, answers

    def encode_both(self, batch):
        """pass 1 of both agents on their B0 prompt + own answer -> z (2, B, K, D_e), lam (2, B, K, D_e) float32"""
        zs, ls = [], []
        saved = dict(self.trunc)                                               # the sender's B0 pieces are not part of the receiver's truncation readout
        for i in range(M.N_AGENTS):
            pb0, answers = self.sender_pieces(batch, i)
            pcs = [(p, a_ids, []) for (p, _, _), (_, a_ids) in zip(pb0, answers)]
            z, lam = self.agent(i).encode(pcs)
            zs.append(z)
            ls.append(lam)
        self.trunc.clear()
        self.trunc.update(saved)
        return torch.stack(zs), torch.stack(ls)


# ================================================================ construction
def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, **agent_kw):
    """as l1c_model.build_pair with AgentD / PairD (the receiver state = the B0 checkpoint, loaded into both agents and frozen; it is also the
    sender's model). k defaults to K_HINT."""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    agent_kw = dict(agent_kw)
    agent_kw.setdefault('lora_dropout', 0.0)
    agent_kw.setdefault('k', K_HINT)
    if tiny is not None:
        tn = list(M.TINY_NAMES) if pair == 'hetero' else [M.TINY_NAMES[0]]
        if any(n not in tiny for n in tn):
            raise ValueError(f'tiny models {tn} not in {list(tiny)}')
        agents = [AgentD.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau,
                                                 n_blocks=n_blocks, **agent_kw) for n in tn]
    else:
        agents = [AgentD.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in M.model_names(pair, base, model)]
    pd = PairD(agents, pair, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        lora_keys = [k for k in receiver_state if 'lora_' in k]
        if not lora_keys:
            raise ValueError('receiver_state has no LoRA tensors (expected the B0 checkpoint state)')
        own = [n for n, _ in pd.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(lora_keys))
        if missing:
            raise ValueError(f'receiver_state covers {len(lora_keys)} of the {len(own)} LoRA tensors of this pair; e.g. missing {missing[:3]}')
        M.load_trainable_state(pd, {k: receiver_state[k] for k in lora_keys}, strict=True)
    for a in pd.agents:
        a.freeze_lora()
    for n, p in pd.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pd


load_receiver = MC.load_receiver


# ================================================================ readouts
@torch.no_grad()
def eval_render(pair, examples, hard, drop_p=0.0, c_check=False, **kw):
    """the pinned eval_latent with the hard (argmax) or soft rendering of the message"""
    pair.hard = bool(hard)
    for ag in pair.agents:
        ag.reader.default_hard = bool(hard)
    try:
        return M.eval_latent(pair, examples, drop_p=drop_p, c_check=c_check, **kw)
    finally:
        pair.hard = False
        for ag in pair.agents:
            ag.reader.default_hard = False


@torch.no_grad()
def eval_baseline(pair, examples, kind, eval_batch=16, max_new_tokens=M.ANSWER_MAX_NEW, deadline=None, keep_preds=False, log=None):
    """none: the plain B0 prompt (no hint at all; = the receiver's own B0 numbers); gold_hint: the gold answer tokens (same cut + filler as the
    targets) inside the hint template at the slot (the ceiling of this channel)."""
    if kind not in BASELINES:
        raise ValueError(kind)
    pair.eval()
    dev = pair.device
    recs, t_gen = [], 0.0
    pair.trunc.clear()
    for bi, batch in enumerate(M.chunks(examples, eval_batch)):
        M.check_deadline(deadline, f'eval_baseline {kind}')
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            if kind == 'none':
                pcs = pair.pieces(batch, i, 'B0', target=None)
                soft = None
            else:
                pcs = pair.pieces(batch, i, 'latent', target=None)
                tgt, _ = pair.targets(batch, i)
                soft = ag.reader.tokens(tgt, ag.dtype)
            M.sync(dev)
            t0 = time.perf_counter()
            texts, g = ag.generate(pcs, soft, max_new_tokens, want_logits=True)
            M.sync(dev)
            t_gen += time.perf_counter() - t0
            for b, ex in enumerate(batch):
                ent = float(g['first_token_entropy'][b]) if g['first_token_entropy'] is not None else None
                recs.append(M._record(ex, i, ag.name, texts[b], g['steps'][b], ent))
        if log and (bi + 1) % 20 == 0:
            log(f'eval baseline {kind}: {len(recs)} records')
    out = M.summarise(recs)
    out.update(kind=kind, answer_gen_s_per_example=t_gen / max(1, len(recs)), truncation=dict(pair.trunc))
    if keep_preds:
        out['records'] = recs
    return out


def _bucket_means(rows):
    """rows: list of (agent, rel, value dict) -> {agent: {rel: {key: percentage mean, 'n': count}}}"""
    out = {}
    for agent, rel, vals in rows:
        d = out.setdefault(str(agent), {}).setdefault(rel, {})
        for k, v in vals.items():
            d.setdefault(k, []).append(float(v))
    res = {}
    for a, rels in out.items():
        res[a] = {}
        for rel, d in rels.items():
            m = {k: 100.0 * float(np.mean(v)) for k, v in d.items()}
            m['n'] = len(next(iter(d.values()))) if d else 0
            res[a][rel] = m
    return res


@torch.no_grad()
def hint_metrics(pair, examples, eval_batch=16, deadline=None, log=None, n_examples=HINT_EXAMPLES):
    """the decoded (argmax) hint at p = 0 by the RECEIVER's owner bucket: EM / F1 against the gold answer, agreement with the sender's own
    answer; the sender's own B0 answer against the gold by the SENDER's bucket. Percentages."""
    pair.eval()
    hint_rows, sender_rows, exs = [], [], []
    for batch in M.chunks(examples, eval_batch):
        M.check_deadline(deadline, 'hint_metrics')
        z, lam = pair.encode_both(batch)
        xs, _, _ = pair.messages_eval(z, lam)
        answers = [pair.sender_pieces(batch, j)[1] for j in range(M.N_AGENTS)]               # served from the cache filled by encode_both
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            j = 1 - i
            lg = ag.reader.logits(xs[i])
            pred = lg.argmax(-1)
            for b, ex in enumerate(batch):
                pid = pred[b].tolist()
                cut = len(pid)
                for q in range(len(pid) - 1, -1, -1):
                    if pid[q] != ag.filler:
                        break
                    cut = q
                text = ag.tok.decode(pid[:cut], skip_special_tokens=True).strip()
                gold = ex['answer']
                sender_ans = answers[j][b][0]
                rel = M._rel(ex.get('answer_owner'), i)
                hint_rows.append((i, rel, dict(em_gold=float(M.DATA.exact_match(text, gold)), f1_gold=float(M.DATA.f1_score(text, gold)),
                                               agree_sender=float(M.DATA.exact_match(text, sender_ans)))))
                if len(exs) < n_examples:
                    exs.append(dict(id=ex.get('id'), receiver=i, rel=rel, hint=text, sender_answer=sender_ans, gold=gold))
        for j in range(M.N_AGENTS):
            for b, ex in enumerate(batch):
                sender_rows.append((j, M._rel(ex.get('answer_owner'), j), dict(em_gold=float(M.DATA.exact_match(answers[j][b][0], ex['answer'])))))
    return dict(n=len(examples), hint_by_receiver_bucket=_bucket_means(hint_rows), sender_answer_by_own_bucket=_bucket_means(sender_rows), examples=exs)


def lam_by_bucket(records):
    """eval_latent records (with mean_lam, rel, agent) -> {agent: {rel: dict(mean_lam, mean_log_lam, n)}}"""
    out = {}
    for r in records:
        if r.get('mean_lam') is None:
            continue
        d = out.setdefault(str(r['agent']), {}).setdefault(r['rel'], dict(lam=[], log_lam=[]))
        d['lam'].append(float(r['mean_lam']))
        d['log_lam'].append(float(r['mean_log_lam']))
    return {a: {rel: dict(mean_lam=float(np.mean(d['lam'])), mean_log_lam=float(np.mean(d['log_lam'])), n=len(d['lam'])) for rel, d in rels.items()} for a, rels in out.items()}
