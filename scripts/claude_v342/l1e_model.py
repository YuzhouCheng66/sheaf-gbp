"""l1e_model.py (Claude, 2026-10-31): batch L1e model side (pre-registration D8-88) = l1d_model.py with the sender adapter's attention-sink fix
(D8-87 appendix 2: the raw layer-L hidden states of these models carry an attention-sink first token whose norm is 40-100x the median, so the
cross-attention of the L1 / L1c / L1d adapters collapsed onto it and the Qwen side sent a constant message in every batch so far).
  (a) SenderAdapter3 = SenderAdapter2 with a per-position LayerNorm on the hidden states before the key / value projections;
  (b) position 0 is masked out of the cross-attention.
Everything else (the answer-belief message, K = 8, the hint template, the gold-answer targets, the frozen B0 receiver / sender, the hard rendering,
the arms one-way / consensus, the criteria) is l1d_model.py's, imported and unchanged; the arms are named Eu / Es for distinct run directories.
"""
from __future__ import annotations
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402
import l1c_model as MC  # noqa: E402
import l1d_model as MD  # noqa: E402

BATCH = 'L1e'
PREREG = 'D8-88'
ARMS = ('Eu', 'Es')
BASE_ARM = {'Eu': 'U', 'Es': 'S1'}
SCRIPTS = ('l1e_model.py', 'l1e_train.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
K_HINT, HINT_PRE, HINT_POST, ABSENCE_PS, BASELINES, HINT_EXAMPLES = MD.K_HINT, MD.HINT_PRE, MD.HINT_POST, MD.ABSENCE_PS, MD.BASELINES, MD.HINT_EXAMPLES


class SenderAdapter3(MC.SenderAdapter2):
    """SenderAdapter2 + (a) LayerNorm(d_model) on every hidden-state position before k_proj / v_proj, (b) position 0 masked (D8-88 section 3)"""

    def __init__(self, d_model, k=M.K_SLOTS, d_e=M.D_E, n_heads=M.N_HEADS, fixed_lam=False, n_blocks=MC.N_BLOCKS):
        super().__init__(d_model, k, d_e, n_heads, fixed_lam, n_blocks)
        self.ln_h = nn.LayerNorm(d_model)

    def forward(self, h, mask):
        B = h.shape[0]
        with torch.autocast(device_type=h.device.type, enabled=False):
            hn = self.ln_h(h.float())                                           # (a) the sink position no longer dominates the keys / values
            m = mask.clone()
            m[:, 0] = 0                                                         # (b) the first token (attention sink / BOS) is not attended
            x = self.queries[None].expand(B, -1, -1)
            for blk in self.blocks:
                x = blk(x, hn, m)
            z = self.ln_out(x)
            if self.lam_head is None:
                log_lam = torch.zeros_like(z)
            else:
                log_lam = self.lam_head(x).clamp(M.LOG_LAM_MIN, M.LOG_LAM_MAX)
        return z, log_lam


class AgentE(MD.AgentD):
    """AgentD with SenderAdapter3"""

    def __init__(self, model, tokenizer, name, **kw):
        super().__init__(model, tokenizer, name, **kw)
        fixed = self.adapter.fixed_lam
        self.adapter = SenderAdapter3(self.d_model, self.k, self.d_e, kw.get('n_heads', M.N_HEADS), fixed, kw.get('n_blocks', MC.N_BLOCKS)).to(self.device)


class PairE(MD.PairD):
    """PairD of AgentE's under the L1e arm names"""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair != 'hetero':
            raise ValueError('PairE supports pair hetero only')
        M.Pair.__init__(self, agents, pair, BASE_ARM[arm], max_ctx=max_ctx)
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, **agent_kw):
    """as l1d_model.build_pair with AgentE / PairE"""
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
        agents = [AgentE.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau,
                                                 n_blocks=n_blocks, **agent_kw) for n in tn]
    else:
        agents = [AgentE.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in M.model_names(pair, base, model)]
    pe = PairE(agents, pair, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        lora_keys = [k for k in receiver_state if 'lora_' in k]
        if not lora_keys:
            raise ValueError('receiver_state has no LoRA tensors (expected the B0 checkpoint state)')
        own = [n for n, _ in pe.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(lora_keys))
        if missing:
            raise ValueError(f'receiver_state covers {len(lora_keys)} of the {len(own)} LoRA tensors of this pair; e.g. missing {missing[:3]}')
        M.load_trainable_state(pe, {k: receiver_state[k] for k in lora_keys}, strict=True)
    for a in pe.agents:
        a.freeze_lora()
    for n, p in pe.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pe


load_receiver = MC.load_receiver
eval_render = MD.eval_render
eval_baseline = MD.eval_baseline
hint_metrics = MD.hint_metrics
lam_by_bucket = MD.lam_by_bucket
