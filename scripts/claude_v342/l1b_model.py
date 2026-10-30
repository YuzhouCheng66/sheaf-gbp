"""l1b_model.py (Claude, 2026-10-30): batch L1b "给隐态信道一个稠密学习信号" -- the model side, pre-registration OVERNIGHT_REPORT D8-85.
L1b = the latent arms of L1 (l1_model.py, PINNED and untouched; this module only subclasses it) with two changes to the shared skeleton
(every latent arm alike; the arms still differ in the fusion rule only):
  (a) an auxiliary reconstruction loss: pass 2 runs on
          [BOS?] prefix (own shard) | K soft tokens | suffix (question) | answer EOS | AUX_PROMPT | partner facts EOS
      where "partner facts" = l1_data.text_sender_target(ex, 1 - i) (the OTHER agent's gold supporting sentences = what T's sender is
      trained to write), tokenised by the receiver, cut to AUX_MAX_TOKENS. loss = CE(answer) + AUX_BETA * CE(partner facts), each a mean over
      its own tokens, then the mean over the two agents. The aux segment comes AFTER the answer: causally the answer logits (and the
      evaluation, which decodes after "Answer:" to EOS) are exactly those of l1_model.Agent.loss (test: the answer CE is identical).
  (b) soft tokens at the embedding scale: Reader = Linear + LayerNorm (as L1) times a fixed per-model scalar s_i = mean L2 norm of the
      vocabulary embedding rows / sqrt(d_i) (Qwen3-1.7B ~ 0.034, Llama-3.2-1B ~ 0.021; computed from the frozen embeddings at init, a buffer).
  (c) readouts: the decodability of the message (gold answer teacher-forced, then the aux segment is generated greedily and scored by token
      F1 against the partner's gold sentences) at drop_p = 0 (real messages) and 1 (every message dropped: S1 -> own z, U -> zeros);
      the training loss recorded per segment. The absence curve defaults to p in {0, 1} (D8-85 3c).
Arms: S1b (fusion, learned lam and w), Ub (one-way projection, lam = 1, 30 % training dropout), S1L1b (fusion, lam = 1); each maps to its L1
base arm (S1 / U / S1L1) for everything the base class does (fusion, GBP evaluation, the C assertion, the absence curve).
"""
from __future__ import annotations
import math
import os
import sys
import time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402

BATCH = 'L1b'
PREREG = 'D8-85'
AUX_PROMPT = '\nFacts from the other agent:'    # follows the answer's EOS; the partner's gold supporting sentences follow it (leading space inside the target)
AUX_BETA = 1.0                                  # loss = CE(answer) + AUX_BETA * CE(partner facts)
AUX_MAX_TOKENS = 64                             # the partner facts are cut to this many receiver tokens (= T's message budget)
ABSENCE_PS = (0.0, 1.0)                         # D8-85 3c: the absence curve at p = 0 and 1 only
DECODE_N = 256                                  # decodability readout: the first DECODE_N VAL examples
DECODE_MAX_NEW = 64                             # ... generating at most this many tokens after AUX_PROMPT
ARMS = ('S1b', 'Ub', 'S1L1b')
BASE_ARM = {'S1b': 'S1', 'Ub': 'U', 'S1L1b': 'S1L1'}
SCRIPTS = ('l1b_model.py', 'l1b_train.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')     # l1b_train imports l1_train (read_data etc.)


def embedding_scale(embed_weight):
    """s = mean over the vocabulary of ||e_v||_2 / sqrt(d): the RMS of a typical input embedding (the LayerNorm output has RMS 1)"""
    w = embed_weight.detach().float()
    return float(w.norm(dim=-1).mean() / math.sqrt(w.shape[1]))


class ReaderB(M.Reader):
    """R_i = (Linear + LayerNorm as in L1) * s_i, s_i a fixed buffer (D8-85 3b)"""

    def __init__(self, d_e, d_model, scale):
        super().__init__(d_e, d_model)
        self.register_buffer('scale', torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x, dtype=None):
        y = super().forward(x, None) * self.scale
        return y.to(dtype) if dtype is not None else y


class AgentB(M.Agent):
    """l1_model.Agent whose Reader is a ReaderB and which has the two-segment loss"""

    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, **kw)
        self.reader_scale = None
        if self.reader is not None:
            self.reader_scale = embedding_scale(self.embed.weight)
            self.reader = ReaderB(self.d_e, self.d_model, self.reader_scale).to(self.device)

    def loss_aux(self, pieces, soft, aux):
        """pieces: (pre, suf, ans) with ans = answer tokens + EOS; aux: per example the token ids of AUX_PROMPT + partner facts + EOS, appended
        after ans. -> (loss_ans, loss_aux, dict(per_example (B,), first_token_entropy (B,), n_tokens, n_aux_tokens)). loss_ans is the mean CE
        over the answer tokens (the definition of l1_model.Agent.loss); loss_aux the mean CE over the aux tokens of the batch."""
        full = [(p, s, list(a) + list(x)) for (p, s, a), x in zip(pieces, aux)]
        A = self.assemble(full, soft, 'right')
        emb = A['embeds']
        if self.training and not emb.requires_grad:
            emb.requires_grad_(True)
        h = self.decoder(inputs_embeds=emb, attention_mask=A['mask'], use_cache=False).last_hidden_state
        b_idx, t_idx, tgt, seg, first, counts = [], [], [], [], [], []
        for b, ((_, _, a), x) in enumerate(zip(pieces, aux)):
            if not a:
                raise ValueError('loss_aux needs answer tokens')
            a0 = A['ans_start'][b]
            first.append(len(tgt))
            for j, t in enumerate(list(a) + list(x)):
                b_idx.append(b); t_idx.append(a0 + j - 1); tgt.append(int(t)); seg.append(0 if j < len(a) else 1)
            counts.append(len(a))
        dev = h.device
        hs = h[torch.tensor(b_idx, device=dev), torch.tensor(t_idx, device=dev)]
        logits = self.lm_head(hs).float()
        tg = torch.tensor(tgt, dtype=torch.long, device=dev)
        sg = torch.tensor(seg, dtype=torch.long, device=dev)
        ce = Fn.cross_entropy(logits, tg, reduction='none')
        l_ans = ce[sg == 0].mean()
        l_aux = ce[sg == 1].mean() if bool((sg == 1).any()) else ce.sum() * 0.0
        per = torch.stack([ce[f:f + c].mean() for f, c in zip(first, counts)]).detach()
        lp = torch.log_softmax(logits[torch.tensor(first, device=dev)].detach(), -1)
        ent = -(lp.exp() * lp).sum(-1)
        return l_ans, l_aux, dict(per_example=per, first_token_entropy=ent, n_tokens=int((sg == 0).sum()), n_aux_tokens=int((sg == 1).sum()))


class PairB(M.Pair):
    """l1_model.Pair of AgentB's; self.arm is the L1 base arm (what the fusion / evaluation code reads), self.arm_b the L1b arm name."""

    def __init__(self, agents, pair, arm, max_ctx=None, beta=AUX_BETA, aux_max=AUX_MAX_TOKENS):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        super().__init__(agents, pair, BASE_ARM[arm], max_ctx=max_ctx)
        self.arm_b, self.beta, self.aux_max = arm, float(beta), int(aux_max)
        self._aux_cache = {}
        self.aux_hist = []                       # (loss_ans, loss_aux) of every training step

    def aux_ids(self, batch, i):
        """per example of the batch: receiver-i token ids of AUX_PROMPT + partner (1 - i) gold facts (cut to aux_max) + EOS"""
        ag = self.agent(i)
        out = []
        for b, ex in enumerate(batch):
            key = (ag.name, ex.get('id'), i)
            ids = self._aux_cache.get(key) if ex.get('id') is not None else None
            if ids is None:
                ids = ag.ids(AUX_PROMPT) + ag.ids(M.DATA.text_sender_target(ex, 1 - i))[:self.aux_max] + [ag.eos]
                if ex.get('id') is not None:
                    self._aux_cache[key] = ids
            out.append(ids)
        return out

    def step(self, batch, arm=None, train=True, drop=None):
        """-> (loss, info); loss = mean_i CE_ans_i + beta * mean_i CE_aux_i; info as l1_model.Pair.step plus loss_aux_agent / loss_ans / loss_aux"""
        base = self.arm
        z, lam = self.encode_both(batch)
        softs, used = self.soft_from_train(z, lam, base, train, drop)
        la, lx, exs = [], [], []
        for i in range(M.N_AGENTS):
            a_, x_, e_ = self.agent(i).loss_aux(self.pieces(batch, i, 'latent'), softs[i], self.aux_ids(batch, i))
            la.append(a_); lx.append(x_); exs.append(e_)
        loss_ans, loss_aux = torch.stack(la).mean(), torch.stack(lx).mean()
        loss = loss_ans + self.beta * loss_aux
        info = dict(z=z, lam=lam, soft=softs, drop=used, loss_agent=[float(l.detach()) for l in la], loss_aux_agent=[float(l.detach()) for l in lx],
                    loss_ans=float(loss_ans.detach()), loss_aux=float(loss_aux.detach()), first_token_entropy=[e['first_token_entropy'] for e in exs],
                    n_aux_tokens=[e['n_aux_tokens'] for e in exs])
        if train:
            self.aux_hist.append((info['loss_ans'], info['loss_aux']))
        return loss, info

    def aux_summary(self, window=100):
        h = self.aux_hist
        if not h:
            return dict(steps=0)
        a, x = np.array([t[0] for t in h]), np.array([t[1] for t in h])
        return dict(steps=len(h), loss_ans_first=float(a[:window].mean()), loss_ans_last=float(a[-window:].mean()),
                    loss_aux_first=float(x[:window].mean()), loss_aux_last=float(x[-window:].mean()),
                    history=[dict(step=s + 1, loss_ans=float(a[max(0, s + 1 - window):s + 1].mean()), loss_aux=float(x[max(0, s + 1 - window):s + 1].mean()))
                             for s in range(window - 1, len(h), window)] + ([] if len(h) % window == 0 else [dict(step=len(h), loss_ans=float(a[-window:].mean()), loss_aux=float(x[-window:].mean()))]))


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=True, max_ctx=None, beta=AUX_BETA, aux_max=AUX_MAX_TOKENS, **agent_kw):
    """as l1_model.build_pair for the L1b arms (always comm=True): tiny = dict name -> (model, tokenizer) from l1_model.tiny_models()"""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    if tiny is not None:
        tn = list(M.TINY_NAMES) if pair == 'hetero' else [M.TINY_NAMES[0]]
        if any(n not in tiny for n in tn):
            raise ValueError(f'tiny models {tn} not in {list(tiny)}')
        agents = [AgentB.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, **agent_kw) for n in tn]
    else:
        agents = [AgentB.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, **agent_kw)
                  for n in M.model_names(pair, base, model)]
    return PairB(agents, pair, arm, max_ctx=max_ctx, beta=beta, aux_max=aux_max)


@torch.no_grad()
def eval_decodability(pair, examples, drop_p=0.0, eval_batch=16, seed=0, max_new_tokens=DECODE_MAX_NEW, deadline=None, log=None):
    """D8-85 3c: pass 1 -> messages as in evaluation (gbp_sync / raw z; every directed message dropped with probability drop_p, seeded) ->
    with the GOLD answer teacher-forced after "Answer:", the aux segment is generated greedily after AUX_PROMPT (<= max_new_tokens) and scored
    by token F1 against the partner's gold supporting sentences. -> dict(drop_p, n, n_records, f1, f1_agent, f1_model, n_tokens_mean, seconds)"""
    arm = pair.arm
    pair.eval()
    dev = pair.device
    g = torch.Generator().manual_seed(int(seed) * 7919 + int(round(drop_p * 100)))
    diam = FU.tree_diameter(M.N_AGENTS, M.EDGES)
    t0 = time.time()
    recs = []
    for bi, batch in enumerate(M.chunks(examples, eval_batch)):
        M.check_deadline(deadline, f'decodability p={drop_p}')
        B = len(batch)
        drop = (torch.rand(2 * len(M.EDGES), B, generator=g) < drop_p)
        z, lam = pair.encode_both(batch)
        softs, _, _ = pair.soft_from_eval(z, lam, arm, drop.to(dev), rounds=diam, wire_dtype=FU.WIRE_DTYPE)
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            prompt = ag.ids(AUX_PROMPT)
            pcs = pair.pieces(batch, i, 'latent')                                       # target 'answer': ans = gold answer + EOS
            gen = [(p, list(s) + list(a) + prompt, []) for p, s, a in pcs]
            texts, gi = ag.generate(gen, softs[i], max_new_tokens, want_logits=False, first_line=False)
            for b, ex in enumerate(batch):
                gold = M.DATA.text_sender_target(ex, 1 - i).strip()
                recs.append(dict(id=ex.get('id'), agent=i, model=ag.name, f1=float(M.DATA.f1_score(texts[b], gold)), n_tokens=len(gi['ids'][b]), pred=texts[b]))
        if log and (bi + 1) % 20 == 0:
            log(f'decodability p={drop_p}: {len(recs)} records')
    f1 = [r['f1'] for r in recs]
    out = dict(drop_p=float(drop_p), n=len(examples), n_records=len(recs), f1=float(np.mean(f1)) if f1 else None,
               f1_agent=[float(np.mean([r['f1'] for r in recs if r['agent'] == i])) if recs else None for i in range(M.N_AGENTS)],
               f1_model={m: float(np.mean([r['f1'] for r in recs if r['model'] == m])) for m in dict.fromkeys(r['model'] for r in recs)},
               n_tokens_mean=float(np.mean([r['n_tokens'] for r in recs])) if recs else None, max_new_tokens=int(max_new_tokens), seconds=time.time() - t0,
               examples=[dict(id=r['id'], agent=r['agent'], f1=r['f1'], pred=r['pred'][:300]) for r in recs[:8]])
    return out
