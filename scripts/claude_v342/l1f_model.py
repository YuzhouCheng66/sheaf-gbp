"""l1f_model.py (Claude, 2026-11-01): batch L1f model side (pre-registration D8-89) = l1e_model.py on the SAME-MODEL pair 'same2' = two independent
Qwen3-1.7B instances (agent 0 reads shard 0, agent 1 reads shard 1; each with its own answer cache, adapter and decoder), both carrying the Qwen
LoRA of the hetero B0 checkpoint of the same seed (the `agents.0.*` tensors, re-mapped to `agents.1.*` for the second instance), all frozen.
Purpose (D8-89): with one tokenizer on both sides the cross-vocabulary dictionary disappears; if Fu passes G1 while L1e's Eu did not, the
dictionary was the remaining bottleneck of the heterogeneous channel. Everything else is l1e_model.py's.
"""
from __future__ import annotations
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_model as M  # noqa: E402
import l1c_model as MC  # noqa: E402
import l1d_model as MD  # noqa: E402
import l1e_model as ME  # noqa: E402

BATCH = 'L1f'
PREREG = 'D8-89'
ARMS = ('Fu', 'Fs')
BASE_ARM = {'Fu': 'U', 'Fs': 'S1'}
PAIR = 'same2'
MODELS_SAME2 = ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B')
SCRIPTS = ('l1f_model.py', 'l1f_train.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
K_HINT, HINT_PRE, HINT_POST, ABSENCE_PS, BASELINES, HINT_EXAMPLES = MD.K_HINT, MD.HINT_PRE, MD.HINT_POST, MD.ABSENCE_PS, MD.BASELINES, MD.HINT_EXAMPLES


class PairF(ME.PairE):
    """PairE of two AgentE instances of the same model; the pinned Pair logic runs as for 'hetero' (two agents, idx [0, 1]); self.pair = 'same2'"""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair != PAIR:
            raise ValueError(f'PairF supports pair {PAIR} only')
        agents = list(agents)
        if len(agents) != 2 or agents[0] is agents[1]:
            raise ValueError('pair same2 needs two INDEPENDENT agent instances')
        M.Pair.__init__(self, agents, 'hetero', BASE_ARM[arm], max_ctx=max_ctx)
        self.pair = PAIR
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []


def remap_receiver_state(receiver_state):
    """the hetero B0 state -> the same2 state: agent 0's LoRA tensors for both instances (agents.0.* and a copy renamed agents.1.*)"""
    a0 = {k: v for k, v in receiver_state.items() if k.startswith('agents.0.') and 'lora_' in k}
    if not a0:
        raise ValueError('receiver_state has no agents.0.* LoRA tensors')
    out = dict(a0)
    for k, v in a0.items():
        out['agents.1.' + k[len('agents.0.'):]] = v
    return out


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, **agent_kw):
    """two independent instances of MODELS_SAME2[0] (tiny: two deep copies of tiny-a, wrapped separately)"""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    if pair != PAIR:
        raise ValueError(f'build_pair: pair {pair} is not {PAIR}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    agent_kw = dict(agent_kw)
    agent_kw.setdefault('lora_dropout', 0.0)
    agent_kw.setdefault('k', K_HINT)
    if tiny is not None:                                   # two INDEPENDENT instances of the same tiny model (deep-copied before the LoRA wrapping)
        n0 = M.TINY_NAMES[0]
        if n0 not in tiny:
            raise ValueError(f'tiny model {n0} not in {list(tiny)}')
        m0, t0 = tiny[n0]
        m1 = copy.deepcopy(m0)
        agents = [ME.AgentE.from_pretrained_objects(m, t0, n0, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for m in (m0, m1)]
    else:
        agents = [ME.AgentE.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in MODELS_SAME2]
    pf = PairF(agents, PAIR, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        state2 = remap_receiver_state(receiver_state)
        own = [n for n, _ in pf.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(state2))
        if missing:
            raise ValueError(f'the remapped receiver state covers {len(state2)} of the {len(own)} LoRA tensors; e.g. missing {missing[:3]}')
        M.load_trainable_state(pf, {k: state2[k] for k in own}, strict=True)
    for a in pf.agents:
        a.freeze_lora()
    for n, p in pf.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pf


load_receiver = MC.load_receiver
eval_render = MD.eval_render
eval_baseline = MD.eval_baseline
hint_metrics = MD.hint_metrics
lam_by_bucket = MD.lam_by_bucket
