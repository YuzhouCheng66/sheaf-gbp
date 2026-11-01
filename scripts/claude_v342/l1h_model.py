"""l1h_model.py (Claude, 2026-11-01): batch L1h model side (pre-registration D8-91) = l1f_model.py (same-model pair same2, the answer-belief
channel, the frozen B0 receiver / sender, the hard rendering) with the sender adapter replaced by an ALIGNED READ: slot k is a learned linear
projection (+ LayerNorm) of the layer-L hidden state at the position of the sender's k-th answer token; slots beyond the answer length carry a
learned "empty" vector; output LayerNorm and the precision head as before. No queries, no cross-attention, no slot self-attention: the sender
knows where its own answer is. Readout `slot_accuracy`: the receiver's decoded token per slot against the gold answer and against the sender's
own answer tokens.
"""
from __future__ import annotations
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
import l1e_model as ME  # noqa: E402
import l1f_model as MF  # noqa: E402

BATCH = 'L1h'
PREREG = 'D8-91'
ARMS = ('Hu', 'Hs')
BASE_ARM = {'Hu': 'U', 'Hs': 'S1'}
PAIR = MF.PAIR                                   # 'same2'
MODELS_SAME2 = MF.MODELS_SAME2
SCRIPTS = ('l1h_model.py', 'l1h_train.py', 'l1f_model.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
K_HINT, HINT_PRE, HINT_POST, ABSENCE_PS, BASELINES, HINT_EXAMPLES = MD.K_HINT, MD.HINT_PRE, MD.HINT_POST, MD.ABSENCE_PS, MD.BASELINES, MD.HINT_EXAMPLES


class AlignedSender(nn.Module):
    """P_i (D8-91 section 3): z_k = LN_out(LN_in(W h[pos_k])) for a real answer token at slot k, else LN_out(empty_k); precision head as L1
    (lam = 1 at init; fixed_lam -> lam = 1, no head). forward(h, mask, positions): h (B, T, d_model), positions (B, K) long with -1 = empty."""

    def __init__(self, d_model, k=M.K_SLOTS, d_e=M.D_E, fixed_lam=False):
        super().__init__()
        self.k, self.d_e, self.fixed_lam, self.n_heads = int(k), int(d_e), bool(fixed_lam), 0
        self.proj = nn.Linear(d_model, self.d_e)
        self.ln_in = nn.LayerNorm(self.d_e)
        self.empty = nn.Parameter(torch.randn(self.k, self.d_e) * 0.02)
        self.ln_out = nn.LayerNorm(self.d_e)
        if self.fixed_lam:
            self.lam_head = None
        else:
            self.lam_head = nn.Linear(self.d_e, self.d_e)
            nn.init.zeros_(self.lam_head.weight)
            nn.init.zeros_(self.lam_head.bias)

    def forward(self, h, mask, positions):
        B = h.shape[0]
        if positions.shape != (B, self.k):
            raise ValueError(f'positions {tuple(positions.shape)} != ({B}, {self.k})')
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = h.float()
            idx = positions.clamp_min(0).to(h.device)
            g = hf[torch.arange(B, device=h.device)[:, None], idx]                   # (B, K, d_model)
            x = self.ln_in(self.proj(g))
            empty = (positions.to(h.device) < 0)[..., None]
            x = torch.where(empty, self.empty[None].expand(B, -1, -1), x)
            z = self.ln_out(x)
            if self.lam_head is None:
                log_lam = torch.zeros_like(z)
            else:
                log_lam = self.lam_head(x).clamp(M.LOG_LAM_MIN, M.LOG_LAM_MAX)
        return z, log_lam


class AgentH(ME.AgentE):
    """AgentE whose adapter is an AlignedSender and whose pass 1 passes the answer-token positions"""

    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, **kw)
        fixed = self.adapter.fixed_lam
        self.adapter = AlignedSender(self.d_model, self.k, self.d_e, fixed).to(self.device)

    def answer_positions(self, pieces, A):
        """pieces (pre, ans_ids, []) right-padded by assemble -> (B, K) positions of the first K answer tokens, -1 beyond the answer"""
        B = len(pieces)
        pos = torch.full((B, self.k), -1, dtype=torch.long)
        for b, (p, s, _) in enumerate(pieces):
            n = min(len(s), self.k)
            if n:
                pos[b, :n] = A['start'][b] + len(p) + torch.arange(n)
        return pos

    def encode(self, pieces):
        A = self.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
        with torch.no_grad():
            out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.layer].detach()
        self.freeze_lora()
        z, log_lam = self.adapter(h, A['mask'], self.answer_positions(pieces, A))
        return z, log_lam.exp()


class PairH(MF.PairF):
    """PairF under the L1h arm names"""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair != PAIR:
            raise ValueError(f'PairH supports pair {PAIR} only')
        agents = list(agents)
        if len(agents) != 2 or agents[0] is agents[1]:
            raise ValueError('pair same2 needs two INDEPENDENT agent instances')
        M.Pair.__init__(self, agents, 'hetero', BASE_ARM[arm], max_ctx=max_ctx)
        self.pair = PAIR
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, **agent_kw):
    """as l1f_model.build_pair with AgentH / PairH"""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    if pair != PAIR:
        raise ValueError(f'build_pair: pair {pair} is not {PAIR}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    agent_kw = dict(agent_kw)
    agent_kw.setdefault('lora_dropout', 0.0)
    agent_kw.setdefault('k', K_HINT)
    if tiny is not None:
        import copy
        n0 = M.TINY_NAMES[0]
        if n0 not in tiny:
            raise ValueError(f'tiny model {n0} not in {list(tiny)}')
        m0, t0 = tiny[n0]
        m1 = copy.deepcopy(m0)
        agents = [AgentH.from_pretrained_objects(m, t0, n0, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for m in (m0, m1)]
    else:
        agents = [AgentH.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in MODELS_SAME2]
    ph = PairH(agents, PAIR, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        state2 = MF.remap_receiver_state(receiver_state)
        own = [n for n, _ in ph.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(state2))
        if missing:
            raise ValueError(f'the remapped receiver state covers {len(state2)} of the {len(own)} LoRA tensors; e.g. missing {missing[:3]}')
        M.load_trainable_state(ph, {k: state2[k] for k in own}, strict=True)
    for a in ph.agents:
        a.freeze_lora()
    for n, p in ph.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return ph


@torch.no_grad()
def slot_accuracy(pair, examples, eval_batch=16, deadline=None):
    """per slot k: the receiver's decoded token == the gold target token / == the sender's own answer token (same tokenizer), over the slots that
    carry a real token; also the mean over slots and the fraction of examples whose first slot is right"""
    pair.eval()
    K = pair.agent(0).k
    hit_g = [[0, 0] for _ in range(K)]
    hit_s = [[0, 0] for _ in range(K)]
    saved = dict(pair.trunc)                                                   # the senders' B0 pieces are not part of the receiver's truncation readout
    for batch in M.chunks(examples, eval_batch):
        M.check_deadline(deadline, 'slot_accuracy')
        z, lam = pair.encode_both(batch)
        xs, _, _ = pair.messages_eval(z, lam)
        for i in range(M.N_AGENTS):
            j = 1 - i
            ag = pair.agent(i)
            pred = ag.reader.logits(xs[i]).argmax(-1).cpu()
            tgt, lens = pair.targets(batch, i)
            tgt = tgt.cpu()
            _, answers = pair.sender_pieces(batch, j)                          # cached by encode_both (same example ids)
            for b in range(len(batch)):
                s_ids = [int(t) for t in answers[b][1][:K]]
                for k in range(K):
                    if k < lens[b]:
                        hit_g[k][1] += 1
                        hit_g[k][0] += int(pred[b, k] == tgt[b, k])
                    if k < len(s_ids):
                        hit_s[k][1] += 1
                        hit_s[k][0] += int(int(pred[b, k]) == s_ids[k])
    pair.trunc.clear()
    pair.trunc.update(saved)
    acc_g = [(h / n if n else None) for h, n in hit_g]
    acc_s = [(h / n if n else None) for h, n in hit_s]
    return dict(n=len(examples), acc_gold_by_slot=acc_g, acc_sender_by_slot=acc_s, n_by_slot=[n for _, n in hit_g],
                acc_gold_mean=float(np.mean([a for a in acc_g if a is not None])), acc_sender_mean=float(np.mean([a for a in acc_s if a is not None])))


load_receiver = MC.load_receiver
remap_receiver_state = MF.remap_receiver_state
eval_render = MD.eval_render
eval_baseline = MD.eval_baseline
hint_metrics = MD.hint_metrics
lam_by_bucket = MD.lam_by_bucket
