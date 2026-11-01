"""l1g_model.py (Claude, 2026-11-01): batch L1g model side (pre-registration D8-90) = l1e_model.py (hetero pair, SenderAdapter3, the answer-belief
channel) with arm names Gu / Gs and `load_init`: the adapters + decoders are initialised from the dictionary-pretraining checkpoint written by
l1g_pretrain.py (runs/l1g/pretrain-s<seed>/trainable.pt). Everything else is l1e_model.py's.
"""
from __future__ import annotations
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

BATCH = 'L1g'
PREREG = 'D8-90'
ARMS = ('Gu', 'Gs')
BASE_ARM = {'Gu': 'U', 'Gs': 'S1'}
PRETRAIN_ARM = 'Gu'                              # the dictionary is pretrained one-way (x_i = z_j), lam = 1
SCRIPTS = ('l1g_model.py', 'l1g_train.py', 'l1g_pretrain.py', 'l1e_model.py', 'l1d_model.py', 'l1c_model.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
K_HINT, HINT_PRE, HINT_POST, ABSENCE_PS, BASELINES, HINT_EXAMPLES = MD.K_HINT, MD.HINT_PRE, MD.HINT_POST, MD.ABSENCE_PS, MD.BASELINES, MD.HINT_EXAMPLES
SPAN_PREFIX_TAIL = '\nAnswer:'                   # the pretraining prefix ends like the task prompt, so the learned attention position transfers


class PairG(ME.PairE):
    """PairE under the L1g arm names"""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair != 'hetero':
            raise ValueError('PairG supports pair hetero only')
        M.Pair.__init__(self, agents, pair, BASE_ARM[arm], max_ctx=max_ctx)
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=MC.TAU,
               n_blocks=MC.N_BLOCKS, **agent_kw):
    """as l1e_model.build_pair with PairG"""
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
        agents = [ME.AgentE.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau,
                                                    n_blocks=n_blocks, **agent_kw) for n in tn]
    else:
        agents = [ME.AgentE.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in M.model_names(pair, base, model)]
    pg = PairG(agents, pair, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        lora_keys = [k for k in receiver_state if 'lora_' in k]
        if not lora_keys:
            raise ValueError('receiver_state has no LoRA tensors (expected the B0 checkpoint state)')
        own = [n for n, _ in pg.named_parameters() if 'lora_' in n]
        missing = sorted(set(own) - set(lora_keys))
        if missing:
            raise ValueError(f'receiver_state covers {len(lora_keys)} of the {len(own)} LoRA tensors of this pair; e.g. missing {missing[:3]}')
        M.load_trainable_state(pg, {k: receiver_state[k] for k in lora_keys}, strict=True)
    for a in pg.agents:
        a.freeze_lora()
    for n, p in pg.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pg


def channel_state(pair):
    """the adapters' and decoders' parameters (what the dictionary pretraining learns and the fine-tune inherits); lam heads / w excluded"""
    return {n: p.detach().cpu() for n, p in pair.named_parameters() if p.requires_grad and ('.adapter.' in n or '.reader.' in n) and 'lam_head' not in n}


def load_init(pair, path):
    """load the pretrained channel (adapters + decoders) into `pair` -> dict(loaded, not_in_file, ignored, sha256, path). The lam heads (Gs) and w
    start fresh; a tensor of the file that the pair does not have (or has with another shape) is an error."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck
    own = dict(pair.named_parameters())
    take, ignored = {}, []
    for k, v in state.items():
        if k not in own:
            raise KeyError(f'init tensor {k} is not a parameter of this pair')
        if tuple(own[k].shape) != tuple(v.shape):
            raise ValueError(f'init tensor {k} shape {tuple(v.shape)} != {tuple(own[k].shape)}')
        if 'lora_' in k:
            ignored.append(k)
            continue
        take[k] = v
    M.load_trainable_state(pair, take, strict=True)
    not_in_file = [n for n, p in own.items() if p.requires_grad and n not in take]
    return dict(path=os.path.abspath(path), sha256=M.file_sha256(path), loaded=sorted(take), n_loaded=len(take), not_in_file=not_in_file, ignored=ignored,
                pretrain_summary=(ck.get('summary') if isinstance(ck, dict) else None))


load_receiver = MC.load_receiver
eval_render = MD.eval_render
eval_baseline = MD.eval_baseline
hint_metrics = MD.hint_metrics
lam_by_bucket = MD.lam_by_bucket
