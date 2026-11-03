"""l3_model.py (Claude, 2026-11-02): ladder L3 (pre-registration D8-93): a two-round protocol on top of the FROZEN V2 solution (l2_model).
Round 1 (frozen V2): the agents exchange span_sel evidence (32 hard tokens in the SPAN template); each agent answers greedily with its recv
adapter on 'documents | evidence | question' (the round-1 answer: text, ids, mean token probability), cached per example id (saved to disk
and reused by the later stages).
Round 2:
  S2  each agent reads the layer-L hidden states (recv adapter) of its OWN round-1 answer tokens inside 'documents | evidence | question |
      answer' (AlignedSender, K = 8, with the precision head) -> (z2, lam2); the two agents' beliefs are fused by the sheaf-GBP consensus
      (fuse_direct, w2 learned) -> x2 per agent -> Decoder2 (the receiver's vocabulary) -> 8 hard tokens in the template
      'Consensus answer of both agents: ___.' placed after the evidence and before the question.
      Stage 3 trains the answer senders + lam heads + w2 + Decoder2 with the position CE of the decoded consensus vs the gold answer.
      Stage 4 trains the recv LoRA on the round-2 prompts (answer CE). Evaluation: greedy answers on the round-2 prompt.
  T2  the same round 1; round 2 = the PARTNER's round-1 answer as text with its token probability: "Other agent's answer: X (confidence 0.73)."
      Stage 4 only (recv LoRA on those prompts). Evaluation: greedy answers on that prompt.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402
import l1d_model as MD  # noqa: E402
import l1h_model as MH  # noqa: E402
import l2_model as L2  # noqa: E402

BATCH = 'L3'
PREREG = 'D8-93'
ARMS = ('S2a', 'S2b', 'T2', 'S3a', 'S3b')
CONSENSUS = ('code', 'logprob')             # code: D8-93 (fuse_direct on the 256-d codes); logprob: D8-94 (precision-weighted sum of the slot log-probabilities)
ARM_CONSENSUS = {'S2a': 'code', 'S2b': 'code', 'T2': 'code', 'S3a': 'logprob', 'S3b': 'logprob'}
K_ANS = 8
LAM_FEATS = 3                                 # [mean token probability of the round-1 answer, mean max slot probability, mean slot entropy / log V]
HINT2_PRE = 'Consensus answer of both agents:'
HINT2_POST = '.\n'
T2_PRE = "Other agent's answer:"
SCRIPTS = ('l3_model.py', 'l3_train.py', 'l2_model.py', 'l2_train.py', 'l1h_model.py', 'l1f_model.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py',
           'l1_fusion.py', 'l1_data.py')


def softplus_inv(x):
    return float(np.log(np.expm1(x)))


def auroc(scores, labels):
    """area under the ROC of `scores` (higher = predicted positive) for binary `labels`; None if one class is missing"""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


@torch.no_grad()
def hidden_with(ag, pieces, adapter):
    """pass 1 of `ag` under `adapter` (no gradient) -> (h at layer L, the assembled batch); right padding"""
    ag.use(adapter)
    A = ag.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
    out = ag.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
    return out.hidden_states[ag.layer].detach(), A


@torch.no_grad()
def answer_logprob(ag, prompts, answers, adapter):
    """mean token probability of each answer (list of id lists) given its prompt (list of id lists) under `adapter`; empty answer -> 0.0"""
    ag.use(adapter)
    out = []
    pcs = [(p, a, []) for p, a in zip(prompts, answers)]
    A = ag.assemble(pcs, None, 'right', embeds=False)
    hs = ag.decoder(input_ids=A['ids'], attention_mask=A['mask'], use_cache=False).last_hidden_state   # (B, T, d): the vocabulary projection only at the answer positions
    for b, (p, a) in enumerate(zip(prompts, answers)):
        if not a:
            out.append(0.0)
            continue
        start = A['start'][b] + len(p)
        pos = torch.tensor([start + k - 1 for k in range(len(a))], dtype=torch.long, device=hs.device)
        lg = ag.lm_head(hs[b, pos]).float()                                    # (len(a), V)
        lp = torch.log_softmax(lg, -1)
        tok = torch.tensor([int(t) for t in a], dtype=torch.long, device=hs.device)
        out.append(float(lp[torch.arange(len(a), device=hs.device), tok].exp().mean()))
    return out


class PairL3(nn.Module):
    """the two-round protocol around a frozen PairL2 (V2) with its recv adapters"""

    def __init__(self, pair, k_ans=K_ANS, consensus='code'):
        super().__init__()
        if consensus not in CONSENSUS:
            raise ValueError(consensus)
        self.consensus_mode = consensus
        self.pair = pair
        if pair.mode != 'span_sel':
            raise ValueError('PairL3 needs the V2 pair in span_sel mode')
        if not all(a.recv for a in pair.agents):
            raise ValueError('PairL3 needs the recv adapters (the V2 receivers)')
        self.k_ans = int(k_ans)
        d_e = pair.agent(0).d_e
        self.ans_senders = nn.ModuleList([MH.AlignedSender(a.d_model, self.k_ans, d_e, fixed_lam=False).to(a.device) for a in pair.agents])
        self.dec2 = nn.ModuleList([MD.DecoderD(d_e, a.embed.weight).to(a.device) for a in pair.agents])
        self.w2_raw = nn.Parameter(torch.tensor(softplus_inv(1.0), dtype=torch.float32, device=pair.device))
        self.lam_head = nn.Sequential(nn.Linear(LAM_FEATS, 16), nn.Tanh(), nn.Linear(16, 1)).to(pair.device)   # logprob: the calibrated precision (shared by the agents)
        nn.init.zeros_(self.lam_head[2].weight)
        nn.init.zeros_(self.lam_head[2].bias)
        self.cache = {}                                                        # example id -> {str(i): dict(ev, ans, ans_ids, p)}
        self.n_round1 = 0
        self.hint2_pre = [[int(t) for t in a.ids(HINT2_PRE)] for a in pair.agents]
        self.hint2_post = [[int(t) for t in a.ids(HINT2_POST)] for a in pair.agents]
        self.hist = []

    # ---------------------------------------------------------------- helpers
    @property
    def device(self):
        return self.pair.device

    @property
    def w2(self):
        return Fn.softplus(self.w2_raw)

    @property
    def w2_value(self):
        return float(self.w2.detach())

    def agent(self, i):
        return self.pair.agent(i)

    def gold_ids(self, ex, i):
        ag = self.agent(i)
        ids = [int(t) for t in ag.ids(' ' + ex['answer'])[:self.k_ans]]
        return ids + [ag.filler] * (self.k_ans - len(ids)), len(ids)

    # ---------------------------------------------------------------- round 1 (frozen V2)
    @torch.no_grad()
    def round1(self, batch, max_new=M.ANSWER_MAX_NEW):
        """fill the cache for the batch: per agent i the partner's evidence tokens (32 ids), the agent's round-1 answer (text, ids) and its mean
        token probability; -> list of per-example dicts"""
        need = [b for b, ex in enumerate(batch) if ex.get('id') is None or ex['id'] not in self.cache]
        if need:
            sub = [batch[b] for b in need]
            pair = self.pair
            was_training = self.training
            self.eval()                                                        # generation and the teacher-forced pass never run in train mode
            z, lam = pair.encode_both(sub)
            xs, _, _ = pair.messages_eval(z, lam)
            evs = [pair.agent(i).reader.logits(xs[i]).argmax(-1).cpu().tolist() for i in range(M.N_AGENTS)]
            recs = [{} for _ in sub]
            for i in range(M.N_AGENTS):
                ag = pair.agent(i)
                pcs, _dl = pair.doc_pieces(sub, i)
                prompts = [list(p) + ag.hint_pre + [int(t) for t in evs[i][b]] + ag.hint_post for b, (p, s, _) in enumerate(pcs)]
                sufs = [list(s) for (p, s, _) in pcs]
                texts, _g = ag.generate([(pr, sf, []) for pr, sf in zip(prompts, sufs)], None, max_new, want_logits=False, adapter='recv')
                ans_ids = [[int(t) for t in ag.ids(' ' + t)] if t else [] for t in texts]
                probs = answer_logprob(ag, [pr + sf for pr, sf in zip(prompts, sufs)], ans_ids, 'recv')
                for b in range(len(sub)):
                    recs[b][str(i)] = dict(ev=[int(t) for t in evs[i][b]], ans=texts[b], ans_ids=ans_ids[b], p=float(probs[b]))
            self.n_round1 += len(sub)
            if was_training:
                self.train()
            for b, ex in zip(need, sub):
                if ex.get('id') is not None:
                    self.cache[ex['id']] = recs[need.index(b)]
            # examples without an id are not cached: return them directly
            out = [None] * len(batch)
            for q, b in enumerate(need):
                out[b] = recs[q]
            for b, ex in enumerate(batch):
                if out[b] is None:
                    out[b] = self.cache[ex['id']]
            return out
        return [self.cache[ex['id']] for ex in batch]

    def save_cache(self, path):
        M.jdump(dict(n=len(self.cache), k_ev=self.pair.span_k, cache=self.cache), path)

    def load_cache(self, path):
        d = M.jload(path)
        self.cache.update(d['cache'])
        return len(d['cache'])

    def pieces_r1(self, batch, i, recs=None):
        """-> (pre1 = [BOS] documents | SPAN template with the partner's evidence, suf1 = question) per example"""
        recs = recs if recs is not None else self.round1(batch)
        ag = self.agent(i)
        pcs, _dl = self.pair.doc_pieces(batch, i)
        out = []
        for b, (p, s, _) in enumerate(pcs):
            r = recs[b][str(i)]
            out.append((list(p) + ag.hint_pre + [int(t) for t in r['ev']] + ag.hint_post, list(s)))
        return out

    # ---------------------------------------------------------------- round 2: beliefs, consensus, decoding
    def beliefs(self, batch, recs=None):
        """-> z2 (2, B, K, d_e), lam2 (2, B, K, d_e): the aligned read of each agent's own round-1 answer tokens (recv adapter)"""
        recs = recs if recs is not None else self.round1(batch)
        zs, ls = [], []
        for i in range(M.N_AGENTS):
            ag = self.agent(i)
            r1 = self.pieces_r1(batch, i, recs)
            pcs = [(pre + suf, [int(t) for t in recs[b][str(i)]['ans_ids']], []) for b, (pre, suf) in enumerate(r1)]
            h, A = hidden_with(ag, pcs, 'recv')
            B = len(batch)
            pos = torch.full((B, self.k_ans), -1, dtype=torch.long)
            for b, (p, s, _) in enumerate(pcs):
                n = min(len(s), self.k_ans)
                if n:
                    pos[b, :n] = A['start'][b] + len(p) + torch.arange(n)
            z, log_lam = self.ans_senders[i](h, A['mask'], pos)
            zs.append(z)
            ls.append(log_lam.exp())
        return torch.stack(zs), torch.stack(ls)

    def consensus(self, z2, lam2):
        with torch.autocast(device_type=z2.device.type, enabled=False):
            return FU.fuse_direct(z2.float(), lam2.float(), M.EDGES, self.w2)

    def consensus_ids(self, x2):
        """hard tokens of the consensus per agent -> list of (B, K) long"""
        return [self.dec2[i].logits(x2[i]).argmax(-1) for i in range(M.N_AGENTS)]

    # ---------------------------------------------------------------- the log-probability consensus (D8-94)
    def lam_feats(self, batch, recs, i, logp):
        """(B, LAM_FEATS): the round-1 answer's mean token probability, the mean max probability and the mean normalised entropy of the slot
        distributions (over the slots that carry the answer)"""
        B = len(batch)
        p = torch.tensor([float(recs[b][str(i)]['p']) for b in range(B)], device=logp.device)
        pr = logp.exp()
        mx = pr.max(-1).values                                                 # (B, K)
        ent = -(pr * logp).sum(-1) / float(np.log(logp.shape[-1]))             # (B, K) in [0, 1]
        lens = torch.tensor([max(1, min(self.k_ans, len(recs[b][str(i)]['ans_ids']))) for b in range(B)], device=logp.device)
        msk = (torch.arange(self.k_ans, device=logp.device)[None] < lens[:, None]).float()
        return torch.stack([p, (mx * msk).sum(1) / lens, (ent * msk).sum(1) / lens], 1)

    def fused(self, batch, recs=None):
        """-> (fused log-probabilities (B, K, V) shared by both receivers (one vocabulary), lam (2, B), u (2, B) the pre-softplus precision
        logits, per-agent log-probabilities list): lam_i = softplus(u_i); fused = (lam_i l_i + lam_j l_j) / (lam_i + lam_j) = the two-agent
        GBP consensus on the natural parameters"""
        recs = recs if recs is not None else self.round1(batch)
        z2, _lam2 = self.beliefs(batch, recs)
        logps, us = [], []
        for i in range(M.N_AGENTS):
            with torch.autocast(device_type=z2.device.type, enabled=False):
                lp = torch.log_softmax(self.dec2[i].logits(z2[i].float()), -1)
            logps.append(lp)
            us.append(self.lam_head(self.lam_feats(batch, recs, i, lp.detach())).squeeze(-1))
        u = torch.stack(us)                                                    # (2, B)
        lam = Fn.softplus(u) + 1e-3
        w = lam / lam.sum(0, keepdim=True)
        fused = sum(w[i][:, None, None] * logps[i] for i in range(M.N_AGENTS))
        return fused, lam, u, logps

    def step3_logprob(self, batch, train=True):
        recs = self.round1(batch)
        fused, lam, u, logps = self.fused(batch, recs)
        tg = [self.gold_ids(ex, 0) for ex in batch]
        tgt = torch.tensor([t for t, _ in tg], dtype=torch.long, device=fused.device)
        ce = Fn.cross_entropy(fused.reshape(-1, fused.shape[-1]), tgt.reshape(-1))
        correct = torch.tensor([[float(M.DATA.exact_match(recs[b][str(i)]['ans'], ex['answer'])) for b, ex in enumerate(batch)] for i in range(M.N_AGENTS)], device=fused.device)
        cal = Fn.binary_cross_entropy_with_logits(u, correct)
        loss = ce + cal
        pred = fused.detach().argmax(-1)
        msk = torch.zeros_like(tgt, dtype=torch.bool)
        for b, (_, n) in enumerate(tg):
            msk[b, :n] = True
        acc = float((pred == tgt)[msk].float().mean()) if bool(msk.any()) else float('nan')
        info = dict(loss_agent=[float(ce.detach()), float(cal.detach())], msg_acc=acc, w2=float(lam.mean()))
        if train:
            self.hist.append((float(loss.detach()), acc))
        return loss, info

    # ---------------------------------------------------------------- stage 3: the consensus CE
    def step3(self, batch, train=True):
        if self.consensus_mode == 'logprob':
            return self.step3_logprob(batch, train)
        recs = self.round1(batch)
        z2, lam2 = self.beliefs(batch, recs)
        x2 = self.consensus(z2, lam2)
        losses, accs = [], []
        for i in range(M.N_AGENTS):
            lg = self.dec2[i].logits(x2[i])
            tg = [self.gold_ids(ex, i) for ex in batch]
            tgt = torch.tensor([t for t, _ in tg], dtype=torch.long, device=lg.device)
            ce = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            losses.append(ce)
            pred = lg.detach().argmax(-1)
            msk = torch.zeros_like(tgt, dtype=torch.bool)
            for b, (_, n) in enumerate(tg):
                msk[b, :n] = True
            accs.append(float((pred == tgt)[msk].float().mean()) if bool(msk.any()) else float('nan'))
        loss = torch.stack(losses).mean()
        info = dict(loss_agent=[float(l.detach()) for l in losses], msg_acc=float(np.nanmean(accs)), w2=self.w2_value)
        if train:
            self.hist.append((float(loss.detach()), info['msg_acc']))
        return loss, info

    # ---------------------------------------------------------------- stage 4 / eval prompts
    @torch.no_grad()
    def soft2(self, batch, recs=None):
        if self.consensus_mode == 'logprob':
            fused, lam, _u, _lp = self.fused(batch, recs)
            ids = fused.argmax(-1)
            return [self.dec2[i].tokens(ids, self.agent(i).dtype) for i in range(M.N_AGENTS)], [ids, ids], lam
        z2, lam2 = self.beliefs(batch, recs)
        x2 = self.consensus(z2, lam2)
        ids = self.consensus_ids(x2)
        return [self.dec2[i].tokens(ids[i], self.agent(i).dtype) for i in range(M.N_AGENTS)], ids, lam2

    def pieces_r2(self, batch, i, target='answer', recs=None):
        """(pre1 + HINT2_PRE, HINT2_POST + suf1, ans): the K_ANS slots go between (assemble inserts the soft tokens after `pre`)"""
        ag = self.agent(i)
        out = []
        for b, (pre, suf) in enumerate(self.pieces_r1(batch, i, recs)):
            ans = (ag.ids(' ' + batch[b]['answer']) + [ag.eos]) if target == 'answer' else []
            out.append((pre + self.hint2_pre[i], self.hint2_post[i] + suf, ans))
        return out

    def pieces_t2(self, batch, i, target='answer', recs=None):
        """T2: the partner's round-1 answer as text with its confidence, no slots"""
        recs = recs if recs is not None else self.round1(batch)
        ag = self.agent(i)
        j = 1 - i
        out = []
        for b, (pre, suf) in enumerate(self.pieces_r1(batch, i, recs)):
            r = recs[b][str(j)]
            txt = f"{T2_PRE} {r['ans'] if r['ans'] else '(none)'} (confidence {r['p']:.2f}).\n"
            ans = (ag.ids(' ' + batch[b]['answer']) + [ag.eos]) if target == 'answer' else []
            out.append((pre + [int(t) for t in ag.ids(txt)], suf, ans))
        return out

    def step4(self, batch, train=True, arm='S2b'):
        """the receivers' answer CE on the round-2 prompts (S2b: the consensus slots; T2: the text answer + confidence)"""
        recs = self.round1(batch)
        losses = []
        if arm == 'T2':
            for i in range(M.N_AGENTS):
                loss, _ = self.agent(i).loss(self.pieces_t2(batch, i, 'answer', recs), None)
                losses.append(loss)
        else:
            softs, _ids, _lam = self.soft2(batch, recs)
            for i in range(M.N_AGENTS):
                loss, _ = self.agent(i).loss(self.pieces_r2(batch, i, 'answer', recs), softs[i])
                losses.append(loss)
        loss = torch.stack(losses).mean()
        info = dict(loss_agent=[float(l.detach()) for l in losses], msg_acc=float('nan'))
        if train:
            self.hist.append((float(loss.detach()), float('nan')))
        return loss, info

    # ---------------------------------------------------------------- evaluation
    @torch.no_grad()
    def eval_round2(self, examples, arm='S2b', eval_batch=16, max_new_tokens=M.ANSWER_MAX_NEW, deadline=None, keep_preds=False, log=None):
        """greedy answers on the round-2 prompts (S2b: consensus slots; T2: text) -> summarise(records) + the round-2 readouts"""
        self.eval()
        self.pair.eval()
        recs_all, r1_rows, cons_rows, lam_rows = [], [], [], []
        for bi, batch in enumerate(M.chunks(examples, eval_batch)):
            M.check_deadline(deadline, f'eval_round2 {arm}')
            recs = self.round1(batch)
            if arm == 'T2':
                softs, ids, lam2 = None, None, None
            else:
                softs, ids, lam2 = self.soft2(batch, recs)
            for i in range(M.N_AGENTS):
                ag = self.agent(i)
                pcs = self.pieces_t2(batch, i, None, recs) if arm == 'T2' else self.pieces_r2(batch, i, None, recs)
                texts, g = ag.generate(pcs, None if softs is None else softs[i], max_new_tokens, want_logits=True, adapter='recv')
                for b, ex in enumerate(batch):
                    ent = float(g['first_token_entropy'][b]) if g['first_token_entropy'] is not None else None
                    recs_all.append(M._record(ex, i, ag.name, texts[b], g['steps'][b], ent))
                    r = recs[b][str(i)]
                    rj = recs[b][str(1 - i)]
                    em_i = M.DATA.exact_match(r['ans'], ex['answer'])
                    em_j = M.DATA.exact_match(rj['ans'], ex['answer'])
                    r1_rows.append(dict(agent=i, em=em_i, em_partner=em_j, either=max(em_i, em_j), p=r['p'], rel=M._rel(ex.get('answer_owner'), i)))
                    if ids is not None:
                        pid = ids[i][b].tolist()
                        cut = len(pid)
                        for q in range(len(pid) - 1, -1, -1):
                            if pid[q] != ag.filler:
                                break
                            cut = q
                        cons = ag.tok.decode(pid[:cut], skip_special_tokens=True).strip()
                        cons_rows.append(dict(agent=i, em=M.DATA.exact_match(cons, ex['answer']), text=cons))
                        lam_rows.append(dict(agent=i, lam=float(lam2[i, b].mean()), em=em_i, p=r['p'], picked=int(cons.strip() != '' and M.DATA.exact_match(cons, r['ans']))))
            if log and (bi + 1) % 20 == 0:
                log(f'eval_round2 {arm}: {len(recs_all)} records')
        out = M.summarise(recs_all)
        r1 = dict(em_agent=[100.0 * float(np.mean([r['em'] for r in r1_rows if r['agent'] == i])) for i in range(M.N_AGENTS)],
                  either_correct=100.0 * float(np.mean([r['either'] for r in r1_rows])),
                  both_correct=100.0 * float(np.mean([min(r['em'], r['em_partner']) for r in r1_rows])),
                  p_auroc=auroc([r['p'] for r in r1_rows], [r['em'] for r in r1_rows]),
                  em_by_owner={rel: 100.0 * float(np.mean([r['em'] for r in r1_rows if r['rel'] == rel])) for rel in ('own', 'other', 'none') if any(r['rel'] == rel for r in r1_rows)})
        out.update(arm=arm, round1=r1)
        if cons_rows:
            out['consensus'] = dict(em_agent=[100.0 * float(np.mean([r['em'] for r in cons_rows if r['agent'] == i])) for i in range(M.N_AGENTS)],
                                    examples=[r['text'] for r in cons_rows[:12]],
                                    lam_auroc_vs_round1_correct=auroc([r['lam'] for r in lam_rows], [r['em'] for r in lam_rows]),
                                    lam_mean=float(np.mean([r['lam'] for r in lam_rows])), w2=self.w2_value, mode=self.consensus_mode,
                                    picked_own_answer=100.0 * float(np.mean([r['picked'] for r in lam_rows])),
                                    pick_correct=100.0 * float(np.mean([r['em'] for r in cons_rows])))
        if keep_preds:
            out['records'] = recs_all
        return out


def trainable_sets(pl3, stage):
    """stage 3: answer senders + dec2 + w2 trainable, everything else frozen; stage 4: the recv LoRA only"""
    for p in pl3.parameters():
        p.requires_grad_(False)
    L2.freeze_sender(pl3.pair, frozen=True)
    if stage == 3:
        for m in list(pl3.ans_senders) + list(pl3.dec2) + [pl3.lam_head]:
            for p in m.parameters():
                p.requires_grad_(True)
        pl3.w2_raw.requires_grad_(True)
        pl3.pair.use('default')
    elif stage == 4:
        pl3.pair.use('recv')
    else:
        raise ValueError(stage)


def l3_state(pl3):
    """the checkpoint: the V2 state (sender adapter, selector, reader, recv LoRA) + the stage-3 modules + w2"""
    st = {}
    for n, p in pl3.named_parameters():
        if n.startswith('ans_senders.') or n.startswith('dec2.') or n.startswith('lam_head.') or n == 'w2_raw' or '.adapter.' in n or '.reader.' in n or '.selector.' in n or ('lora_' in n and '.recv.' in n):
            st[n] = p.detach().cpu()
    return st


def build_l3(v2_ckpt, device, hf_cache=None, tiny=None, max_ctx=None, receiver_state=None, l3_state_dict=None, consensus='code', **agent_kw):
    """the frozen V2 pair (span_sel, recv adapters loaded) + the L3 modules (optionally loaded from an S2a checkpoint)"""
    ck = torch.load(v2_ckpt, map_location='cpu', weights_only=False)
    if ck.get('arm') not in ('V2', 'V3') or not ck.get('recv'):
        raise ValueError(f'{v2_ckpt}: not a V2 / V3 checkpoint with recv adapters (arm {ck.get("arm")}, recv {ck.get("recv")})')
    init = {k: v for k, v in ck['state'].items() if '.adapter.' in k or '.reader.' in k or '.selector.' in k}
    kw = dict(agent_kw)
    kw.setdefault('k', int(ck['config'].get('k', L2.K_SPAN)))
    pair = L2.build_pair(ck['arm'], 'same2', device, hf_cache=hf_cache, tiny=tiny, grad_ckpt=True, max_ctx=max_ctx, receiver_state=receiver_state, mode='span_oracle',
                         span_k=kw['k'], init_state=init, recv=True, **kw)
    recv = {k: v for k, v in ck['state'].items() if 'lora_' in k and '.recv.' in k}
    if not recv:
        raise ValueError(f'{v2_ckpt}: no recv LoRA tensors')
    own = [n for n, _ in pair.named_parameters() if 'lora_' in n and '.recv.' in n]
    missing = sorted(set(own) - set(recv))
    if missing:
        raise ValueError(f'the V2 checkpoint covers {len(recv)} of the {len(own)} recv tensors; e.g. missing {missing[:3]}')
    M.load_trainable_state(pair, {k: recv[k] for k in own}, strict=True)
    pair.set_mode('span_sel')
    pl3 = PairL3(pair, consensus=consensus)
    if l3_state_dict is not None:
        sub = {k: v for k, v in l3_state_dict.items() if k.startswith('ans_senders.') or k.startswith('dec2.') or k.startswith('lam_head.') or k == 'w2_raw'}
        own3 = [n for n, _ in pl3.named_parameters() if n.startswith('ans_senders.') or n.startswith('dec2.') or n.startswith('lam_head.') or n == 'w2_raw']
        missing = sorted(set(own3) - set(sub))
        if missing:
            raise ValueError(f'the L3 state misses {len(missing)} tensors, e.g. {missing[:3]}')
        M.load_trainable_state(pl3, {k: sub[k] for k in own3}, strict=True)
    for p in pl3.parameters():
        p.requires_grad_(False)
    return pl3, dict(v2_arm=ck.get('arm'), v2_seed=ck.get('seed'), v2_sha256=M.file_sha256(v2_ckpt), v2_models=ck.get('models'))
