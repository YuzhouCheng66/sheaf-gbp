"""l2_model.py (Claude, 2026-11-01): experiment ladder L2 (pre-registration D8-92) on top of l1h_model.py (the aligned-read sender).
  (a) AgentL2 = AgentH with TWO LoRA adapters: 'default' = the frozen B0 (the sender role: own answers, pass 1, the `none` baseline) and 'recv'
      = a trainable copy initialised from B0 (the receiver role: pass 2 loss / answers), added by enable_recv(); use(name) switches them.
  (b) position modes of the aligned read: 'answer' (L1h: the sender's own answer tokens), 'span_oracle' (the gold supporting sentences of the
      sender's own gold paragraph located in its prompt ids, first K tokens), 'span_attn' (the K-token window of the documents region with the
      largest summed attention from the QUESTION tokens (heads and question positions averaged) at layer `sel_layer`; the first SINK_SKIP
      positions of the region -- the attention sink -- are excluded; D8-92 addendum of 2026-11-01).
  (c) PairL2 (same2 or hetero): stage 1 = PairC.step (position CE; targets = the gold answer (answer mode) or the tokens at the sender's read
      positions (span modes, one tokenizer)); stage 2 = step_recv (the receiver's answer CE on the partner's HARD-rendered hint, message
      dropout as U, the sender frozen); baselines none / none_recv / gold_hint / own_hint / own_hint_oracle; select_quality (E0c).
"""
from __future__ import annotations
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402
import l1c_model as MC  # noqa: E402
import l1d_model as MD  # noqa: E402
import l1f_model as MF  # noqa: E402
import l1h_model as MH  # noqa: E402

BATCH = 'L2'
PREREG = 'D8-92'
ARMS = ('R1', 'V1', 'Hu', 'V1s', 'V2')
BASE_ARM = {'R1': 'U', 'V1': 'U', 'Hu': 'U', 'V1s': 'S1', 'V2': 'U'}   # V1s (appendix 3): V1 with the sheaf-GBP consensus; V2 (D8-93): V1 + a trained span selector
SPAN_ARMS = ('V1', 'V1s', 'V2')
RECV_ARMS = ('R1', 'V1', 'V1s', 'V2')
SELECTOR_ARMS = ('V2',)
SEL_LOSS_WEIGHT = 1.0                           # D8-93 section 3: BCE of the selector added to the stage-1 copy CE
PAIRS = ('same2', 'hetero')
MODES = ('answer', 'span_oracle', 'span_attn', 'span_sel')
K_SPAN = 32
SPAN_PRE = "From the other agent's documents:"
SPAN_POST = '\n'
SINK_SKIP = 2                                   # the first positions of the documents region are the attention sink (D8-87 appendix 2): never selected
ATTN_SUB_BATCH = 4                              # output_attentions keeps every layer: small sub-batches
BASELINES = ('none', 'none_recv', 'gold_hint', 'own_hint', 'own_hint_oracle')
SCRIPTS = ('l2_model.py', 'l2_train.py', 'l1h_model.py', 'l1f_model.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
K_HINT, HINT_PRE, HINT_POST, HINT_EXAMPLES = MD.K_HINT, MD.HINT_PRE, MD.HINT_POST, MD.HINT_EXAMPLES
MODELS_SAME2 = MF.MODELS_SAME2


def _find(seq, sub):
    """first index of the list `sub` inside the list `seq`, or None"""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return None
    first = sub[0]
    for i in range(n - m + 1):
        if seq[i] == first and seq[i:i + m] == sub:
            return i
    return None


# ================================================================ the agent: two adapters, three position modes
class AgentL2(MH.AgentH):
    """AgentH whose LoRA 'default' is the frozen B0 (sender role) and whose optional LoRA 'recv' is the trainable receiver"""

    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, **kw)
        self.recv = False
        self.gen_adapter = 'default'
        self.active = 'default'
        self._lora_default = list(self._lora_params)
        self._lora_recv = []
        self.sel_layer = None
        self.n_attn_passes = 0
        self.selector = None                                                   # V2: Linear(d_model, 1) on the layer-L hidden states (enable_selector)

    # ---------------------------------------------------------------- adapters
    def enable_recv(self):
        """add the LoRA 'recv' initialised from 'default' (B0); -> number of tensors copied"""
        if not self.lora:
            raise RuntimeError(f'{self.name}: enable_recv needs LoRA')
        if self.recv:
            return len(self._lora_recv)
        self.model.add_adapter('recv', self.model.peft_config['default'])
        params = dict(self.model.named_parameters())
        n_copied = 0
        for n, p in params.items():
            if 'lora_' in n and '.default.' in n:
                q = params[n.replace('.default.', '.recv.')]
                if q.shape != p.shape:
                    raise RuntimeError(f'{self.name}: recv tensor {n} shape {tuple(q.shape)} != {tuple(p.shape)}')
                with torch.no_grad():
                    q.copy_(p)
                n_copied += 1
        self._lora_recv = [p for n, p in params.items() if 'lora_' in n and '.recv.' in n]
        if not self._lora_recv or n_copied != len(self._lora_recv):
            raise RuntimeError(f'{self.name}: recv adapter has {len(self._lora_recv)} tensors, {n_copied} copied')
        self.recv = True
        self.gen_adapter = 'recv'
        self.use('default')
        return n_copied

    def enable_selector(self):
        if self.selector is None:
            self.selector = nn.Linear(self.d_model, 1).to(self.device)
            nn.init.zeros_(self.selector.weight)
            nn.init.zeros_(self.selector.bias)
        return self.selector

    @torch.no_grad()
    def sel_positions(self, pieces, doc_lens, k):
        """V2: the k-window of the documents region with the largest summed selector probability (the layer-L hidden states of pass 1)"""
        if self.selector is None:
            raise ValueError(f'{self.name}: selector not enabled')
        h, A = self.hidden(pieces)
        sc = torch.sigmoid(self.selector(h.float()).squeeze(-1))              # (B, T)
        nb = 1 if self.bos is not None else 0
        B = len(pieces)
        pos = torch.full((B, k), -1, dtype=torch.long)
        for b in range(B):
            lo, hi = A['start'][b] + nb, A['start'][b] + nb + int(doc_lens[b])
            if hi <= lo:
                continue
            start = self.best_window(sc[b, lo:hi], lo, hi, k)
            n = min(k, hi - start)
            pos[b, :n] = start + torch.arange(n)
        return pos

    def use(self, name):
        """activate the adapter `name`; 'default' (B0) is never trainable, 'recv' is trainable only while active"""
        if name == 'recv' and not self.recv:
            raise ValueError(f'{self.name}: recv adapter not enabled')
        if self.lora and self.recv:
            self.model.set_adapter(name)                                       # PEFT re-sets requires_grad of every adapter: fixed below
        for p in self._lora_default:
            p.requires_grad_(False)
        for p in self._lora_recv:
            p.requires_grad_(name == 'recv')
        self.active = name

    def freeze_lora(self):
        for p in self._lora_default:
            p.requires_grad_(False)

    def generate(self, pieces, soft=None, max_new_tokens=M.ANSWER_MAX_NEW, want_logits=False, first_line=True, adapter=None):
        self.use(adapter or self.gen_adapter)
        return super().generate(pieces, soft, max_new_tokens, want_logits, first_line)

    def loss(self, pieces, soft=None):
        """pass 2 (the receiver role): the recv adapter when enabled, else the frozen B0"""
        self.use('recv' if self.recv else 'default')
        return super().loss(pieces, soft)

    @torch.no_grad()
    def own_answers(self, batch, pieces_b0):
        """as AgentD.own_answers, always with the B0 adapter"""
        out = [None] * len(batch)
        need = []
        for b, ex in enumerate(batch):
            key = ex.get('id')
            if key is not None and key in self.answer_cache:
                out[b] = self.answer_cache[key]
            else:
                need.append(b)
        if need:
            texts, _ = self.generate([pieces_b0[b] for b in need], None, self.gen_max_new, want_logits=False, adapter='default')
            self.n_generated += len(need)
            for b, t in zip(need, texts):
                ids = [int(x) for x in self.ids(' ' + t)] if t else []
                out[b] = (t, ids)
                if batch[b].get('id') is not None:
                    self.answer_cache[batch[b]['id']] = out[b]
        return out

    # ---------------------------------------------------------------- pass 1
    def hidden(self, pieces):
        """pass 1 with the B0 adapter, no gradient -> (h (B, T, d_model) at layer L, the assembled batch)"""
        self.use('default')
        A = self.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
        with torch.no_grad():
            out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.layer].detach()
        return h, A

    def encode(self, pieces, positions=None):
        """aligned read at `positions` (B, K; -1 = empty), default = the own-answer positions (L1h) -> z, lam"""
        h, A = self.hidden(pieces)
        if positions is None:
            positions = self.answer_positions(pieces, A)
        z, log_lam = self.adapter(h, A['mask'], positions)
        return z, log_lam.exp()

    @staticmethod
    def best_window(row, lo, hi, k):
        """row: attention mass over the positions [lo, hi); -> the absolute start of the k-window with the largest summed mass (lo if the
        region is shorter than k)"""
        n = int(hi - lo)
        if n <= k:
            return int(lo)
        c = torch.cumsum(torch.cat([row.new_zeros(1), row.float()]), 0)
        sums = c[k:] - c[:-k]                                                  # sums[t] = mass of [lo + t, lo + t + k)
        return int(lo) + int(sums.argmax())

    @torch.no_grad()
    def attn_positions(self, pieces, doc_lens, layer, k):
        """pieces (pre = [BOS] documents, suf = question, []) right-padded; doc_lens[b] = document tokens after the BOS. The attention of the
        question tokens (heads and positions averaged; the last token alone when there is no suffix) at `layer` over the documents region, the
        first SINK_SKIP positions zeroed -> the k-window with the largest summed attention (-1 beyond the region when it is shorter than k)."""
        self.use('default')
        nb = 1 if self.bos is not None else 0
        B = len(pieces)
        pos = torch.full((B, k), -1, dtype=torch.long)
        for lo_b in range(0, B, ATTN_SUB_BATCH):
            sub = pieces[lo_b:lo_b + ATTN_SUB_BATCH]
            A = self.assemble([(p, s, []) for p, s, _ in sub], None, 'right', embeds=False)
            out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_attentions=True, use_cache=False)
            att = out.attentions[layer]                                        # (b, H, T, T)
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
                start = self.best_window(row, lo, hi, k)
                n = min(k, hi - start)
                pos[b, :n] = start + torch.arange(n)
            self.n_attn_passes += len(sub)
            del out, att
        return pos


# ================================================================ the pair
class PairL2(MH.PairH):
    """PairH logic (same2) or the hetero pair, with the position modes and the stage-2 receiver step"""

    def __init__(self, agents, pair, arm, max_ctx=None, mode='answer', span_k=K_SPAN):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair not in PAIRS:
            raise ValueError(f'pair {pair} not in {PAIRS}')
        if mode not in MODES:
            raise ValueError(f'mode {mode} not in {MODES}')
        if mode != 'answer' and pair != 'same2':
            raise ValueError('the span modes need one tokenizer on both sides (pair same2)')
        agents = list(agents)
        if len(agents) != 2 or agents[0] is agents[1]:
            raise ValueError('two INDEPENDENT agent instances are needed')
        M.Pair.__init__(self, agents, 'hetero', BASE_ARM[arm], max_ctx=max_ctx)
        self.pair = pair
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []
        self.mode = mode
        self.span_k = int(span_k)
        self._span_cache = {}
        self.span_seen = 0
        self.span_missing = 0
        self.n_skipped = 0
        self._last = {}                                                        # agent idx -> (h, A, pieces, doc_lens) of the last span-mode pass 1 (the selector loss)
        self.sel_hist = []

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        if mode != 'answer' and self.pair != 'same2':
            raise ValueError('span modes need pair same2')
        self.mode = mode

    def use(self, name):
        for a in self.agents:
            a.use(name)

    # ---------------------------------------------------------------- span modes: pieces, positions, targets
    def doc_pieces(self, batch, i):
        """-> (pieces [(pre = [BOS] documents, suf = question, [])], doc_lens) of agent i: the latent prompt without the slots"""
        ag = self.agent(i)
        nb = 1 if ag.bos is not None else 0
        out, lens = [], []
        for ex in batch:
            (pre, suf, _), _tr = M.make_pieces(ag, ex, i, 'latent', self.max_ctx, None, None)
            out.append((list(pre), list(suf), []))
            lens.append(len(pre) - nb)
        return out, lens

    @staticmethod
    def locate(ag, sentences, pre, k):
        """positions (indices into `pre`) of the first k tokens of the supporting `sentences` in order; a sentence cut by the truncation is
        matched by its first 8 tokens; -> list or None when nothing is found"""
        out = []
        for s in sentences:
            if len(out) >= k:
                break
            full = ag.ids(' ' + s)
            bare = ag.ids(s)
            hit, n = None, 0
            for cand in (full, bare, full[:-1] if len(full) > 3 else [], bare[:-1] if len(bare) > 3 else []):   # a final '.' may merge with the newline
                hit = _find(pre, cand) if cand else None
                if hit is not None:
                    n = len(cand)
                    break
            if hit is None and len(full) >= 8:
                hit = _find(pre, full[:8])
                if hit is not None:
                    n = min(len(full), len(pre) - hit)
            if hit is None:
                continue
            out.extend(range(hit, hit + n))
        out = out[:k]
        return out if out else None

    def oracle_positions(self, batch, i, pieces):
        ag = self.agent(i)
        K = self.span_k
        pos = torch.full((len(batch), K), -1, dtype=torch.long)
        found = []
        for b, ex in enumerate(batch):
            key = (i, 'oracle', ex.get('id'))
            if ex.get('id') is not None and key in self._span_cache:
                p = self._span_cache[key]
            else:
                sup = ex['sup'][i] if ex.get('sup') else []
                p = self.locate(ag, list(sup), pieces[b][0], K)
                self.span_seen += 1
                self.span_missing += int(p is None)
                if ex.get('id') is not None:
                    self._span_cache[key] = p
            found.append(p is not None)
            if p:
                pos[b, :len(p)] = torch.tensor(p, dtype=torch.long)
        return pos, found

    def attn_positions(self, batch, i, pieces, doc_lens):
        ag = self.agent(i)
        if ag.sel_layer is None:
            raise ValueError(f'{ag.name}: sel_layer not set (span_attn)')
        K = self.span_k
        pos = torch.full((len(batch), K), -1, dtype=torch.long)
        need = []
        for b, ex in enumerate(batch):
            key = (i, 'attn', int(ag.sel_layer), ex.get('id'))
            if ex.get('id') is not None and key in self._span_cache:
                pos[b] = self._span_cache[key]
            else:
                need.append(b)
        if need:
            got = ag.attn_positions([pieces[b] for b in need], [doc_lens[b] for b in need], int(ag.sel_layer), K)
            for q, b in enumerate(need):
                pos[b] = got[q]
                if batch[b].get('id') is not None:
                    self._span_cache[(i, 'attn', int(ag.sel_layer), batch[b]['id'])] = got[q].clone()
        return pos, [True] * len(batch)

    def sel_positions(self, batch, i, pieces, doc_lens):
        ag = self.agent(i)
        K = self.span_k
        pos = torch.full((len(batch), K), -1, dtype=torch.long)
        need = []
        for b, ex in enumerate(batch):
            key = (i, 'sel', ex.get('id'))
            if ex.get('id') is not None and key in self._span_cache:
                pos[b] = self._span_cache[key]
            else:
                need.append(b)
        if need:
            got = ag.sel_positions([pieces[b] for b in need], [doc_lens[b] for b in need], K)
            for q, b in enumerate(need):
                pos[b] = got[q]
                if batch[b].get('id') is not None:
                    self._span_cache[(i, 'sel', batch[b]['id'])] = got[q].clone()
        return pos, [True] * len(batch)

    def clear_sel_cache(self):
        for k in [k for k in self._span_cache if k[1] == 'sel']:
            del self._span_cache[k]

    def span_positions(self, batch, i, pieces, doc_lens, mode=None):
        mode = mode or self.mode
        if mode == 'span_oracle':
            return self.oracle_positions(batch, i, pieces)
        if mode == 'span_attn':
            return self.attn_positions(batch, i, pieces, doc_lens)
        if mode == 'span_sel':
            return self.sel_positions(batch, i, pieces, doc_lens)
        raise ValueError(mode)

    def encode_both(self, batch):
        if self.mode == 'answer':
            return MD.PairD.encode_both(self, batch)
        zs, ls = [], []
        for i in range(M.N_AGENTS):
            pcs, dl = self.doc_pieces(batch, i)
            pos, _ = self.span_positions(batch, i, pcs, dl)
            ag = self.agent(i)
            h, A = ag.hidden(pcs)
            if ag.selector is not None:
                self._last[i] = (h, A, pcs, dl)
            z, log_lam = ag.adapter(h, A['mask'], pos)
            zs.append(z)
            ls.append(log_lam.exp())
        return torch.stack(zs), torch.stack(ls)

    def targets(self, batch, i):
        """answer mode: the gold answer in the receiver's tokenizer (PairC / AgentD); span modes: the tokens the SENDER reads at its
        positions (one tokenizer), cut to K and filler-padded"""
        if self.mode == 'answer':
            return MC.PairC.targets(self, batch, i)
        j = 1 - i
        ag = self.agent(i)
        pcs, dl = self.doc_pieces(batch, j)
        pos, _ = self.span_positions(batch, j, pcs, dl)
        ids, lens = [], []
        for b in range(len(batch)):
            seq = list(pcs[b][0]) + list(pcs[b][1])
            toks = [int(seq[int(p)]) for p in pos[b].tolist() if p >= 0][:ag.k]
            ids.append(toks + [ag.filler] * (ag.k - len(toks)))
            lens.append(len(toks))
        return torch.tensor(ids, dtype=torch.long, device=self.device), lens

    # ---------------------------------------------------------------- stage 1 (span_oracle: the examples without a located span are skipped, D8-92 section 3)
    def step(self, batch, arm=None, train=True, drop=None):
        if self.mode == 'span_oracle':
            keep = []
            for ex in batch:
                ok = True
                for i in range(M.N_AGENTS):
                    pcs, _dl = self.doc_pieces([ex], i)
                    _pos, found = self.oracle_positions([ex], i, pcs)
                    ok = ok and found[0]
                keep.append(ok)
            if not all(keep) and any(keep):                                    # nothing located in a whole batch: trained as it is (empty slots / filler targets)
                idx = [b for b, ok in enumerate(keep) if ok]
                self.n_skipped += len(batch) - len(idx)
                batch = [batch[b] for b in idx]
                if drop is not None:
                    drop = drop[:, idx]
        loss, info = MC.PairC.step(self, batch, arm, train, drop)
        if self.mode == 'span_oracle' and any(a.selector is not None for a in self.agents):
            loss = loss + SEL_LOSS_WEIGHT * self.selector_loss(batch, info)
        return loss, info

    def selector_loss(self, batch, info=None):
        """V2 stage 1: BCE of each agent's selector on the documents region of the pass-1 hidden states just computed (encode_both), labels =
        the oracle span tokens (positives re-weighted to the negatives' count); mean over the agents"""
        losses = []
        for i in range(M.N_AGENTS):
            ag = self.agent(i)
            if ag.selector is None or i not in self._last:
                continue
            h, A, pcs, dl = self._last[i]
            opos, found = self.oracle_positions(batch, i, pcs)
            nb = 1 if ag.bos is not None else 0
            logits, labels = [], []
            for b in range(len(batch)):
                if not found[b]:
                    continue
                lo, hi = A['start'][b] + nb, A['start'][b] + nb + int(dl[b])
                lab = torch.zeros(hi - lo, device=h.device)
                lab[[int(p) - lo for p in opos[b].tolist() if p >= 0]] = 1.0
                logits.append(ag.selector(h[b, lo:hi].float()).squeeze(-1))
                labels.append(lab)
            if not logits:
                continue
            lg, lb = torch.cat(logits), torch.cat(labels)
            n_pos = float(lb.sum())
            pw = torch.tensor(max(1.0, (lb.numel() - n_pos) / max(1.0, n_pos)), device=h.device)
            losses.append(nn.functional.binary_cross_entropy_with_logits(lg, lb, pos_weight=pw))
        if not losses:
            return torch.zeros((), device=self.device)
        loss = torch.stack(losses).mean()
        self.sel_hist.append(float(loss.detach()))
        if info is not None:
            info['sel_loss'] = float(loss.detach())
        return loss

    # ---------------------------------------------------------------- stage 2
    def step_recv(self, batch, train=True, drop=None):
        """the receiver's answer CE (pass 2, recv adapter) on the partner's HARD-rendered message; the sender side runs without gradient"""
        with torch.no_grad():
            z, lam = self.encode_both(batch)
            xs, used = self.messages_train(z, lam, train, drop)
            softs = [self.agent(i).reader(xs[i], self.agent(i).dtype, hard=True) for i in range(M.N_AGENTS)]
        losses, n_tok = [], []
        for i in range(M.N_AGENTS):
            pcs = self.pieces(batch, i, 'latent', target='answer')
            loss, inf = self.agent(i).loss(pcs, softs[i])
            losses.append(loss)
            n_tok.append(inf.get('n_tokens'))
        loss = torch.stack(losses).mean()
        info = dict(drop=used, loss_agent=[float(l.detach()) for l in losses], n_tokens=n_tok, msg_acc=float('nan'))
        if train:
            self.msg_hist.append((float(loss.detach()), float('nan')))
        return loss, info


# ================================================================ construction
def load_init(path):
    """the sender / decoder tensors of an L1h or L2 checkpoint (trainable.pt): {name: tensor} of the adapter.* / reader.* entries"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = {k: v for k, v in ck['state'].items() if '.adapter.' in k or '.reader.' in k or '.selector.' in k}
    if not st:
        raise ValueError(f'{path}: no adapter / reader tensors in the checkpoint state')
    meta = {k: ck.get(k) for k in ('arm', 'base_arm', 'pair', 'seed', 'models', 'receiver')}
    meta['sha256'] = M.file_sha256(path)
    meta['n_tensors'] = len(st)
    return st, meta


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, mode='answer', span_k=K_SPAN, init_state=None, recv=False, sel_layer=None, use_selector=None, **agent_kw):
    """same2: two independent instances of MODELS_SAME2[0] with the hetero B0's agent-0 LoRA remapped to both; hetero: the two models with
    the hetero B0 LoRA as saved. init_state: the frozen sender / decoder tensors (load_init); recv: add the trainable receiver adapter."""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    if pair not in PAIRS:
        raise ValueError(f'pair {pair} not in {PAIRS}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    agent_kw = dict(agent_kw)
    agent_kw.setdefault('lora_dropout', 0.0)
    if mode == 'answer':
        agent_kw.setdefault('k', K_HINT)
    else:
        agent_kw['k'] = int(span_k)                                            # the span modes read span_k positions: the slots must match
    kw = dict(lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
    if tiny is not None:
        if pair == 'same2':
            n0 = M.TINY_NAMES[0]
            if n0 not in tiny:
                raise ValueError(f'tiny model {n0} not in {list(tiny)}')
            m0, t0 = tiny[n0]
            agents = [AgentL2.from_pretrained_objects(m, t0, n0, **kw) for m in (m0, copy.deepcopy(m0))]
        else:
            tn = list(M.TINY_NAMES)
            if any(n not in tiny for n in tn):
                raise ValueError(f'tiny models {tn} not in {list(tiny)}')
            agents = [AgentL2.from_pretrained_objects(tiny[n][0], tiny[n][1], n, **kw) for n in tn]
    else:
        names = list(MODELS_SAME2) if pair == 'same2' else M.model_names('hetero', base, model)
        agents = [AgentL2.from_pretrained(n, device, hf_cache, **kw) for n in names]
    pl = PairL2(agents, pair, arm, max_ctx=max_ctx, mode=mode, span_k=span_k)
    if receiver_state is not None:
        state2 = MF.remap_receiver_state(receiver_state) if pair == 'same2' else {k: v for k, v in receiver_state.items() if 'lora_' in k}
        if not state2:
            raise ValueError('receiver_state has no LoRA tensors')
        own = [n for n, _ in pl.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(state2))
        if missing:
            raise ValueError(f'the receiver state covers {len(state2)} of the {len(own)} LoRA tensors; e.g. missing {missing[:3]}')
        M.load_trainable_state(pl, {k: state2[k] for k in own}, strict=True)
    for a in pl.agents:
        a.freeze_lora()
    if init_state is not None:
        own = dict(pl.named_parameters())
        init_state = {k: v for k, v in init_state.items() if '.selector.' not in k} if not use_selector and use_selector is not None else init_state
        core = {k: v for k, v in init_state.items() if '.selector.' not in k}
        missing = [k for k in core if k not in own]
        if missing:
            raise ValueError(f'init_state has {len(missing)} tensors not in this pair, e.g. {missing[:3]}')
        bad = [k for k, v in core.items() if tuple(v.shape) != tuple(own[k].shape)]
        if bad:
            raise ValueError(f'init_state shape mismatch, e.g. {bad[:3]}')
        need = [k for k in own if ('.adapter.' in k or '.reader.' in k) and k not in core]
        if need:
            raise ValueError(f'init_state misses {len(need)} sender / decoder tensors, e.g. {need[:3]}')
        M.load_trainable_state(pl, core, strict=True)
    if mode != 'answer':
        for a in pl.agents:
            a.hint_pre = [int(t) for t in a.ids(SPAN_PRE)]
            a.hint_post = [int(t) for t in a.ids(SPAN_POST)]
    if use_selector is None:
        use_selector = arm in SELECTOR_ARMS
    if use_selector:
        for a in pl.agents:
            a.enable_selector()
        if init_state is not None:
            own = dict(pl.named_parameters())
            sel = {k: v for k, v in init_state.items() if '.selector.' in k}
            if sel:
                M.load_trainable_state(pl, sel, strict=True)
    for a in pl.agents:
        a.sel_layer = sel_layer
    if recv:
        for a in pl.agents:
            a.enable_recv()
    pl.use('default')
    return pl


def freeze_sender(pair, frozen=True):
    """the sender adapter + decoder of every agent (and the fusion weight w of a consensus arm): requires_grad = not frozen (stage 2 trains
    the receiver only)"""
    for a in pair.agents:
        mods = [a.adapter, a.reader] + ([a.selector] if a.selector is not None else [])
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(not frozen)
    for n, p in pair.named_parameters():
        if n == 'w_raw':
            p.requires_grad_(not frozen)


# ================================================================ readouts
@torch.no_grad()
def eval_baseline(pair, examples, kind, eval_batch=16, max_new_tokens=M.ANSWER_MAX_NEW, deadline=None, keep_preds=False, log=None):
    """none: the plain B0 prompt with the B0 adapter (the pre-registered reference); none_recv: the same prompt with the trained receiver;
    gold_hint: the gold answer tokens in the template (the receiver adapter in use); own_hint: the PARTNER's own B0 answer tokens in the
    template with the B0 adapter (the 100 %-copy ceiling of the frozen channel; answer mode); own_hint_oracle: own_hint only where the
    partner's answer is right, else the plain B0 prompt (the confidence-gating ceiling)."""
    if kind not in BASELINES:
        raise ValueError(kind)
    if kind in ('own_hint', 'own_hint_oracle') and pair.mode != 'answer':
        raise ValueError(f'{kind} needs the answer mode')
    pair.eval()
    dev = pair.device
    recs = []
    pair.trunc.clear()
    for bi, batch in enumerate(M.chunks(examples, eval_batch)):
        M.check_deadline(deadline, f'eval_baseline {kind}')
        answers = [pair.sender_pieces(batch, j)[1] for j in range(M.N_AGENTS)] if kind.startswith('own_hint') else None
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            j = 1 - i
            if kind in ('none', 'none_recv'):
                groups = [(list(range(len(batch))), pair.pieces(batch, i, 'B0', target=None), None, 'default' if kind == 'none' else 'recv')]
            elif kind == 'gold_hint':
                tgt, _ = pair.targets(batch, i)
                groups = [(list(range(len(batch))), pair.pieces(batch, i, 'latent', target=None), ag.reader.tokens(tgt, ag.dtype), ag.gen_adapter)]
            else:
                lat = pair.pieces(batch, i, 'latent', target=None)
                b0 = pair.pieces(batch, i, 'B0', target=None)
                ids = []
                for b, ex in enumerate(batch):
                    a_ids = [int(t) for t in answers[j][b][1][:ag.k]]
                    ids.append(a_ids + [ag.filler] * (ag.k - len(a_ids)))
                with_hint = [b for b, ex in enumerate(batch) if kind == 'own_hint' or M.DATA.exact_match(answers[j][b][0], ex['answer'])]
                without = [b for b in range(len(batch)) if b not in with_hint]
                groups = []
                if with_hint:
                    soft = ag.reader.tokens(torch.tensor([ids[b] for b in with_hint], dtype=torch.long, device=dev), ag.dtype)
                    groups.append((with_hint, [lat[b] for b in with_hint], soft, 'default'))
                if without:
                    groups.append((without, [b0[b] for b in without], None, 'default'))
            texts = [None] * len(batch)
            ents = [None] * len(batch)
            steps = [None] * len(batch)
            for idx, pcs, soft, adapter in groups:
                tx, g = ag.generate(pcs, soft, max_new_tokens, want_logits=True, adapter=adapter)
                for q, b in enumerate(idx):
                    texts[b] = tx[q]
                    steps[b] = g['steps'][q]
                    ents[b] = float(g['first_token_entropy'][q]) if g['first_token_entropy'] is not None else None
            for b, ex in enumerate(batch):
                recs.append(M._record(ex, i, ag.name, texts[b], steps[b], ents[b]))
        if log and (bi + 1) % 20 == 0:
            log(f'eval baseline {kind}: {len(recs)} records')
    out = M.summarise(recs)
    out.update(kind=kind, truncation=dict(pair.trunc))
    if keep_preds:
        out['records'] = recs
    return out


@torch.no_grad()
def sel_quality(pair, examples, eval_batch=16, deadline=None):
    """V2 readout: the fraction of (example, agent) pairs whose selector window covers >= 50 % of the oracle span tokens"""
    pair.eval()
    hit = n = 0
    for batch in M.chunks(examples, eval_batch):
        M.check_deadline(deadline, 'sel_quality')
        for i in range(M.N_AGENTS):
            pcs, dl = pair.doc_pieces(batch, i)
            opos, found = pair.oracle_positions(batch, i, pcs)
            spos, _ = pair.sel_positions(batch, i, pcs, dl)
            for b in range(len(batch)):
                if not found[b]:
                    continue
                o = {int(p) for p in opos[b].tolist() if p >= 0}
                a = {int(p) for p in spos[b].tolist() if p >= 0}
                n += 1
                hit += int(len(o & a) >= 0.5 * len(o))
    return dict(overlap50=hit / max(1, n), n=n)


@torch.no_grad()
def select_quality(pair, examples, layers, eval_batch=4, deadline=None, log=None):
    """E0c: per layer, the fraction of (example, agent) pairs whose attention window covers >= 50 % of the oracle span tokens (among the
    pairs with an oracle span); plus the oracle-found rate"""
    pair.eval()
    out = {}
    n_found = n_all = 0
    for layer in layers:
        hit = n = 0
        for batch in M.chunks(examples, eval_batch):
            M.check_deadline(deadline, f'select_quality layer {layer}')
            for i in range(M.N_AGENTS):
                pcs, dl = pair.doc_pieces(batch, i)
                opos, found = pair.oracle_positions(batch, i, pcs)
                apos = pair.agent(i).attn_positions(pcs, dl, int(layer), pair.span_k)
                for b in range(len(batch)):
                    if layer == layers[0]:
                        n_all += 1
                        n_found += int(found[b])
                    if not found[b]:
                        continue
                    o = {int(p) for p in opos[b].tolist() if p >= 0}
                    a = {int(p) for p in apos[b].tolist() if p >= 0}
                    n += 1
                    hit += int(len(o & a) >= 0.5 * len(o))
        out[str(int(layer))] = dict(overlap50=hit / max(1, n), n=n)
        if log:
            log(f'select_quality layer {layer}: overlap50 {out[str(int(layer))]["overlap50"]:.3f} over {n} pairs')
    best = max(out, key=lambda k: out[k]['overlap50']) if out else None
    return dict(by_layer=out, best_layer=(int(best) if best is not None else None), oracle_found=(n_found / max(1, n_all)), n_pairs=n_all)


load_receiver = MC.load_receiver
remap_receiver_state = MF.remap_receiver_state
eval_render = MD.eval_render
hint_metrics = MD.hint_metrics
slot_accuracy = MH.slot_accuracy
