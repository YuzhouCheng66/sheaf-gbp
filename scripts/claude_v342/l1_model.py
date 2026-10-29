"""l1_model.py (Claude, 2026-10-29): batch L1 "LLM 隐态协作：sheaf 共识 vs 文字交换" -- the model side.
Pre-registration OVERNIGHT_REPORT D8-82; design claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md (sections 3, 4, 5, 8).

Pieces (design section 3):
  SenderAdapter  P_i: K learned queries cross-attend (multi-head, attention mask) to the layer-L hidden states of the agent's OWN pass 1
                 (own shard + question), small MLP, LayerNorm -> z in R^{K x D_e} (the trivial-solution guard), precision head ->
                 log lam in [ln 1e-3, ln 1e3] -> lam = exp(log lam); fixed_lam=True -> lam = 1, no precision parameters (S1-Lambda=1, U).
  Reader         R_i: R^{D_e} -> R^{d_i}, Linear + LayerNorm -> K soft tokens in the base model's dtype.
  Agent          one frozen causal LM + LoRA (r 16, alpha 32, dropout .05, q/k/v/o/gate/up/down) + SenderAdapter + Reader; read layer
                 L = floor(layer_frac * n_layers) (hidden_states[L] of output_hidden_states=True; [0] = embeddings).
                 encode = pass 1 -> (z, lam); loss = pass 2 with inputs_embeds = [BOS?] prefix | K soft tokens | suffix | answer EOS
                 (right padding, cross-entropy on the answer tokens + EOS only); generate = greedy, left padding, inputs_embeds.
  Pair           two Agents (pair hetero: Qwen3-1.7B + Llama-3.2-1B-Instruct) or ONE shared Agent used twice (pair same: Qwen3-1.7B;
                 shared LoRA, adapter, reader); w = softplus(w_raw), w = 1 at init; step(batch, arm) implements the arms:
                   B0   each agent answers its own shard without soft tokens (no_comm_prompt);
                   B4   one agent answers from all 10 paragraphs (full_prompt), no soft tokens;
                   U    soft_i = Reader_i(z_j): one-way projection, no fusion, no precision; in training each directed message is
                        replaced by zeros with probability 0.3 (per example);
                   S1   x = fuse_direct(z, lam, [(0, 1)], w); soft_i = Reader_i(x_i); no message dropout;
                   S1L1 S1 with lam = 1 (fixed_lam).
                 Training loss = mean of the two agents' answer losses (B4: the single agent); pass 1 is NOT under no_grad, so the
                 answer loss reaches both encoders / LoRAs through the fusion.
  eval_latent    messages really go through l1_fusion.gbp_sync (rounds = tree_diameter, fp16 wire); absence curve p in {0, .3, .6, 1}
                 (each directed message dropped independently per example, seeded; U -> zero vector, S1 -> own z = what gbp_sync does);
                 the C assertion on the first batch (fuse_direct vs gbp_sync without drop / wire, float32, max |diff| <= 1e-4 (1 + max |x|));
                 readouts of design section 5 (EM / F1 overall, per agent, per model, by answer_owner own / other / none, pre-fusion
                 disagreement ||z_0 - z_1|| and its correlation with correctness, lam statistics and the correlation of mean lam with
                 the first answer token's entropy, wire bytes, sender latency, decode steps).
Fusion math = l1_fusion.py (coordinator; read only). Prompts / metrics / jsonl = l1_data.py (another agent); when it is absent a minimal
fallback (_FallbackData) is used so that the unit tests and --tiny runs work; l1_train.py refuses the fallback outside --tiny, and the chain
pins l1_data.py's sha. R = 2 is a readout only (design section 4) and is NOT implemented: R2_ROUNDS must stay None (set -> error).
The pre-registered numbers below are defaults; the CLIs (l1_train.py / l1_text.py) expose them as flags.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import l1_fusion as FU  # noqa: E402

try:
    import l1_data as _L1DATA  # noqa: E402
except Exception:  # pragma: no cover - the file is being written concurrently; tests use the fallback
    _L1DATA = None

BATCH = 'L1'
PREREG = 'D8-82'
K_SLOTS = 32                                   # K soft tokens / slots per message
D_E = 256                                      # edge-space dimension D_e
N_HEADS = 4                                    # adapter cross-attention heads
LOG_LAM_MIN, LOG_LAM_MAX = math.log(1e-3), math.log(1e3)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
LAYER_FRAC = 0.6667                            # L = floor(frac * n_layers): 18 of 28 (Qwen3-1.7B), 10 of 16 (Llama-3.2-1B) = floor(2/3 n)
U_DROP_P = 0.3                                 # U: training message dropout
ABSENCE_PS = (0.0, 0.3, 0.6, 1.0)              # absence curve at evaluation
ANSWER_MAX_NEW = 16                            # greedy answer tokens
EDGES = [(0, 1)]                               # the two-agent graph
N_AGENTS = 2
MODELS = {'hetero': ('Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.2-1B-Instruct'), 'same': ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B')}
ARMS = ('B0', 'B4', 'U', 'S1', 'S1L1')
LATENT_ARMS = ('U', 'S1', 'S1L1')
FUSION_ARMS = ('S1', 'S1L1')
C_ASSERT_TOL = 1e-4
TIMING_WARMUP_STEPS = 50                       # step times are reported only after this many steps
R2_ROUNDS = None                               # TODO (deferred, design 4 / 5: R = 2 readout). Anything but None raises in Pair.__init__.
TINY_NAMES = ('tiny-a', 'tiny-b')              # --tiny models (random LlamaConfig; tiny-b has no BOS token)
_CORR_MIN = 3


# ================================================================ small utilities shared by l1_train.py / l1_text.py
def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scripts_sha(names=('l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')):
    out = {}
    for n in names:
        p = os.path.join(HERE, n)
        out[n] = file_sha256(p) if os.path.exists(p) else None
    return out


def jload(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def jdump(obj, path):
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(path + '.tmp', path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def epoch_order(n, seed, epoch=0):
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def batch_stream(examples, batch, seed):
    """seeded data order; one epoch = ceil(n / batch) batches; reshuffled per epoch when more steps are asked for"""
    epoch = 0
    while True:
        order = epoch_order(len(examples), seed, epoch)
        for i in range(0, len(order), batch):
            yield [examples[j] for j in order[i:i + batch]]
        epoch += 1


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < _CORR_MIN or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def softplus_inv(y):
    return math.log(math.expm1(y))


class Log:
    """stdout + <out>/log.txt, UTC stamped (the chain reads the last line)"""

    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, msg):
        line = f'{utc()} {msg}'
        print(line, flush=True)
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as fh:
                fh.write(line + '\n')


class Deadline(RuntimeError):
    pass


def check_deadline(deadline, where):
    if deadline and time.time() > deadline:
        raise Deadline(f'deadline {deadline} passed during {where}')


# ================================================================ data API (l1_data.py) and the minimal fallback
def _normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


class _FallbackData:
    """Stand-in for l1_data.py with the same API (used ONLY when l1_data is absent: unit tests, --tiny). Never for a formal run."""
    name = 'fallback'

    @staticmethod
    def read_jsonl(path):
        with open(path, encoding='utf-8') as fh:
            return [json.loads(l) for l in fh if l.strip()]

    @staticmethod
    def _paras(paras):
        return '\n'.join(f'{t}: {" ".join(s)}' for t, s in paras)

    @classmethod
    def latent_prompt(cls, ex, agent):
        return f"Context:\n{cls._paras(ex['shards'][agent])}\n", f"\nQuestion: {ex['question']}\nAnswer:"

    @classmethod
    def no_comm_prompt(cls, ex, agent):
        p, s = cls.latent_prompt(ex, agent)
        return p + s

    @classmethod
    def full_prompt(cls, ex):
        return f"Context:\n{cls._paras(ex['all_paragraphs'])}\n\nQuestion: {ex['question']}\nAnswer:"

    @classmethod
    def text_sender_prompt(cls, ex, agent):
        return f"Context:\n{cls._paras(ex['shards'][agent])}\n\nQuestion: {ex['question']}\nFacts my partner needs:"

    @staticmethod
    def text_sender_target(ex, agent):
        sup = ex['sup'][agent]                                   # l1_data schema: sup[agent] = supporting sentences of the gold paragraph in shard `agent`
        return ' ' + ' '.join(sup) if sup else ' (none)'

    @classmethod
    def text_receiver_prompt(cls, ex, agent, message):
        return f"Context:\n{cls._paras(ex['shards'][agent])}\n\nPartner says:{(' ' + message.strip()) if message.strip() else ' (nothing)'}\nQuestion: {ex['question']}\nAnswer:"

    @staticmethod
    def answer_target(ex):
        return ' ' + ex['answer']

    normalize_answer = staticmethod(_normalize_answer)

    @staticmethod
    def exact_match(pred, gold):
        return int(_normalize_answer(pred) == _normalize_answer(gold))

    @staticmethod
    def f1_score(pred, gold):
        p, g = _normalize_answer(pred).split(), _normalize_answer(gold).split()
        if not p or not g:
            return float(p == g)
        common = collections.Counter(p) & collections.Counter(g)
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        pr, rc = ns / len(p), ns / len(g)
        return 2 * pr * rc / (pr + rc)


DATA = _L1DATA if _L1DATA is not None else _FallbackData
DATA_IS_FALLBACK = _L1DATA is None


def prompt_parts(ex, agent, mode, message=None):
    """(prefix, suffix) text of one example for one agent; the K soft tokens go between the two (latent) -- suffix '' otherwise"""
    if mode == 'latent':
        p, s = DATA.latent_prompt(ex, agent)
        return p, s
    if mode == 'B0':
        return DATA.no_comm_prompt(ex, agent), ''
    if mode == 'B4':
        return DATA.full_prompt(ex), ''
    if mode == 'sender':
        return DATA.text_sender_prompt(ex, agent), ''
    if mode == 'receiver':
        return DATA.text_receiver_prompt(ex, agent, message if message is not None else ''), ''
    raise ValueError(mode)


def _paragraph_tokens(agent, paras):
    return [len(agent.ids(' '.join(s))) for _, s in paras]


def shrink_paragraphs(agent, ex, agent_idx, mode, over):
    """Truncate every paragraph of the shard this prompt reads (B4: all 10) proportionally so that about `over` tokens disappear."""
    paras = ex['all_paragraphs'] if mode == 'B4' else ex['shards'][agent_idx]
    nt = _paragraph_tokens(agent, paras)
    total = sum(nt)
    if total <= 0:
        return ex
    frac = max(0.0, (total - over - 2 * len(paras)) / total)
    new = []
    for (title, sents), n in zip(paras, nt):
        keep = max(1, int(math.floor(frac * n)))
        ids = agent.ids(' '.join(sents))[:keep]
        new.append([title, [agent.tok.decode(ids).strip()]])
    ex2 = dict(ex)
    if mode == 'B4':
        ex2['all_paragraphs'] = new
    else:
        shards = list(ex['shards'])
        shards[agent_idx] = new
        ex2['shards'] = shards
    return ex2


def make_pieces(agent, ex, agent_idx, mode, max_ctx=None, target='answer', message=None):
    """-> ((pre_ids, suf_ids, ans_ids), truncation code 0 / 1 (paragraphs shrunk) / 2 (hard tail cut of the prefix after shrinking)).
    pre starts with BOS when the tokenizer has one; ans = target tokens + EOS (target 'answer' = l1_data.answer_target, 'sender' =
    text_sender_target, None = no answer, or an explicit string)."""
    nb = 1 if agent.bos is not None else 0
    pre_t, suf_t = prompt_parts(ex, agent_idx, mode, message)
    pre, suf = agent.ids(pre_t), agent.ids(suf_t)
    trunc = 0
    if max_ctx and nb + len(pre) + len(suf) > max_ctx:
        ex = shrink_paragraphs(agent, ex, agent_idx, mode, nb + len(pre) + len(suf) - max_ctx)
        pre_t, suf_t = prompt_parts(ex, agent_idx, mode, message)
        pre, suf = agent.ids(pre_t), agent.ids(suf_t)
        trunc = 1
        if nb + len(pre) + len(suf) > max_ctx:
            pre = pre[:max(1, max_ctx - nb - len(suf))]
            trunc = 2
    if target == 'answer':
        ans_t = DATA.answer_target(ex)
    elif target == 'sender':
        ans_t = DATA.text_sender_target(ex, agent_idx)
    else:
        ans_t = target
    pre = ([agent.bos] if nb else []) + pre
    ans = (agent.ids(ans_t) + [agent.eos]) if ans_t is not None else []
    return (pre, suf, ans), trunc


# ================================================================ the modules
class SenderAdapter(nn.Module):
    """P_i: K learned queries in R^{d_e} cross-attend to the layer-L hidden states (keys / values projected d_model -> d_e), MLP,
    LayerNorm -> z (B, K, d_e); precision head -> log lam clamped to [ln 1e-3, ln 1e3]. fixed_lam: lam = 1, no precision parameters.
    Runs in float32 (autocast disabled): z and lam feed the float32 fusion."""

    def __init__(self, d_model, k=K_SLOTS, d_e=D_E, n_heads=N_HEADS, fixed_lam=False):
        super().__init__()
        if d_e % n_heads:
            raise ValueError(f'd_e {d_e} not divisible by n_heads {n_heads}')
        self.k, self.d_e, self.n_heads, self.fixed_lam = k, d_e, n_heads, bool(fixed_lam)
        self.queries = nn.Parameter(torch.randn(k, d_e) * 0.02)
        self.q_proj = nn.Linear(d_e, d_e)
        self.k_proj = nn.Linear(d_model, d_e)
        self.v_proj = nn.Linear(d_model, d_e)
        self.o_proj = nn.Linear(d_e, d_e)
        self.ln_attn = nn.LayerNorm(d_e)
        self.mlp = nn.Sequential(nn.Linear(d_e, 4 * d_e), nn.GELU(), nn.Linear(4 * d_e, d_e))
        self.ln_mlp = nn.LayerNorm(d_e)
        self.ln_out = nn.LayerNorm(d_e)                      # the trivial-solution guard (z cannot collapse to 0)
        if self.fixed_lam:
            self.lam_head = None
        else:
            self.lam_head = nn.Linear(d_e, d_e)
            nn.init.zeros_(self.lam_head.weight)             # lam = 1 at init (log lam = 0)
            nn.init.zeros_(self.lam_head.bias)

    def forward(self, h, mask):
        """h (B, T, d_model) any dtype, mask (B, T) 1 = valid -> z (B, K, d_e) float32, log_lam (B, K, d_e) float32"""
        B, T, _ = h.shape
        H, dh = self.n_heads, self.d_e // self.n_heads
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            q = self.q_proj(self.queries).view(1, self.k, H, dh).transpose(1, 2).expand(B, -1, -1, -1)
            kk = self.k_proj(h).view(B, T, H, dh).transpose(1, 2)
            vv = self.v_proj(h).view(B, T, H, dh).transpose(1, 2)
            am = mask.bool()[:, None, None, :]
            att = Fn.scaled_dot_product_attention(q, kk, vv, attn_mask=am)         # (B, H, K, dh); queries see valid positions only
            a = att.transpose(1, 2).reshape(B, self.k, self.d_e)
            x = self.ln_attn(self.o_proj(a) + self.queries)
            x = self.ln_mlp(x + self.mlp(x))
            z = self.ln_out(x)
            if self.lam_head is None:
                log_lam = torch.zeros_like(z)
            else:
                log_lam = self.lam_head(x).clamp(LOG_LAM_MIN, LOG_LAM_MAX)
        return z, log_lam


class Reader(nn.Module):
    """R_i: R^{d_e} -> R^{d_model}, Linear + LayerNorm -> K soft tokens in the base model's dtype"""

    def __init__(self, d_e, d_model):
        super().__init__()
        self.proj = nn.Linear(d_e, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, dtype=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.ln(self.proj(x.float()))
        return y.to(dtype) if dtype is not None else y


class Agent(nn.Module):
    """One causal LM (frozen base) + optional LoRA + optional SenderAdapter / Reader (comm=False for B0 / B4 / text arms)."""

    def __init__(self, model, tokenizer, name, *, lora=True, comm=True, k=K_SLOTS, d_e=D_E, n_heads=N_HEADS, fixed_lam=False,
                 layer_frac=LAYER_FRAC, grad_ckpt=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                 lora_targets=LORA_TARGETS):
        super().__init__()
        self.name, self.tok = name, tokenizer
        for p in model.parameters():
            p.requires_grad_(False)
        if grad_ckpt and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=list(lora_targets), bias='none',
                             task_type=TaskType.CAUSAL_LM)
            self.model = get_peft_model(model, cfg)
            self.lm = self.model.base_model.model                     # the CausalLM with the LoRA layers injected
        else:
            self.model = model
            self.lm = model
        self.lora = bool(lora)
        self.decoder = self.lm.get_decoder() if hasattr(self.lm, 'get_decoder') else self.lm.model
        self.lm_head = self.lm.get_output_embeddings()
        self.embed = self.lm.get_input_embeddings()
        self.n_layers = int(self.lm.config.num_hidden_layers)
        self.d_model = int(self.lm.config.hidden_size)
        self.layer_frac = float(layer_frac)
        self.layer = int(math.floor(self.layer_frac * self.n_layers))
        self.k, self.d_e = k, d_e
        self.comm = bool(comm)
        self.adapter = SenderAdapter(self.d_model, k, d_e, n_heads, fixed_lam) if comm else None
        self.reader = Reader(d_e, self.d_model) if comm else None
        if self.adapter is not None:
            self.adapter.to(self.device)
            self.reader.to(self.device)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        if self.eos is None:
            raise ValueError(f'{name}: tokenizer has no EOS token')

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained_objects(cls, model, tokenizer, name, **kw):
        return cls(model, tokenizer, name, **kw)

    @classmethod
    def from_pretrained(cls, name, device, hf_cache=None, **kw):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, cache_dir=hf_cache)
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, cache_dir=hf_cache, attn_implementation='sdpa')
        model.to(device)
        return cls(model, tok, name, **kw)

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def ids(self, text):
        if not text:
            return []
        return list(self.tok(text, add_special_tokens=False)['input_ids'])

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    # ---------------------------------------------------------------- sequence assembly
    def assemble(self, pieces, soft=None, side='right', embeds=True):
        """pieces: list of (pre, suf, ans) id lists; soft (B, K, d_model) or None (B0 / B4 / text: nothing inserted).
        Returns dict(ids, mask, embeds, start (first real token), ans_start, k). The soft tokens occupy [start + len(pre), + K)."""
        B = len(pieces)
        k = 0 if soft is None else int(soft.shape[1])
        lens = [len(p) + k + len(s) + len(a) for p, s, a in pieces]
        T = max(lens)
        dev = self.device
        ids = torch.full((B, T), self.pad, dtype=torch.long, device=dev)
        mask = torch.zeros(B, T, dtype=torch.long, device=dev)
        start, ans_start = [], []
        for b, (p, s, a) in enumerate(pieces):
            seq = list(p) + [self.pad] * k + list(s) + list(a)
            off = T - len(seq) if side == 'left' else 0
            ids[b, off:off + len(seq)] = torch.tensor(seq, dtype=torch.long, device=dev)
            mask[b, off:off + len(seq)] = 1
            start.append(off)
            ans_start.append(off + len(p) + k + len(s))
        out = dict(ids=ids, mask=mask, start=start, ans_start=ans_start, k=k, lens=lens)
        if embeds:
            emb = self.embed(ids)
            if soft is not None:
                if soft.shape[0] != B or soft.shape[2] != emb.shape[2]:
                    raise ValueError(f'soft tokens {tuple(soft.shape)} do not match batch {B} x d_model {emb.shape[2]}')
                b_idx = torch.arange(B, device=dev).repeat_interleave(k)
                t_idx = torch.tensor([start[b] + len(pieces[b][0]) + j for b in range(B) for j in range(k)], dtype=torch.long, device=dev)
                full = torch.zeros_like(emb).index_put((b_idx, t_idx), soft.reshape(B * k, -1).to(emb.dtype))
                sel = torch.zeros(B, T, 1, dtype=torch.bool, device=dev).index_put((b_idx, t_idx), torch.ones(B * k, 1, dtype=torch.bool, device=dev))
                emb = torch.where(sel, full, emb)
            out['embeds'] = emb
        return out

    # ---------------------------------------------------------------- pass 1
    def encode(self, pieces):
        """pass 1 on [BOS?] prefix suffix (own shard + question, no soft tokens, no answer) -> z (B, K, d_e), lam (B, K, d_e) float32"""
        if self.adapter is None:
            raise RuntimeError(f'{self.name}: no SenderAdapter (comm=False)')
        A = self.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
        out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer]
        z, log_lam = self.adapter(h, A['mask'])
        return z, log_lam.exp()

    # ---------------------------------------------------------------- pass 2: loss
    def loss(self, pieces, soft=None):
        """answer-token cross-entropy (mean over the answer tokens + EOS of the batch); logits only at the answer positions.
        Returns (loss, dict(per_example (B,), first_token_entropy (B,), n_tokens))"""
        A = self.assemble(pieces, soft, 'right')
        emb = A['embeds']
        if self.training and not emb.requires_grad:
            emb.requires_grad_(True)
        h = self.decoder(inputs_embeds=emb, attention_mask=A['mask'], use_cache=False).last_hidden_state
        b_idx, t_idx, tgt, first, counts = [], [], [], [], []
        for b, (_, _, a) in enumerate(pieces):
            if not a:
                raise ValueError('loss needs answer tokens')
            a0 = A['ans_start'][b]
            first.append(len(tgt))
            for j, t in enumerate(a):
                b_idx.append(b); t_idx.append(a0 + j - 1); tgt.append(t)
            counts.append(len(a))
        hs = h[torch.tensor(b_idx, device=h.device), torch.tensor(t_idx, device=h.device)]
        logits = self.lm_head(hs).float()
        tg = torch.tensor(tgt, dtype=torch.long, device=h.device)
        ce = Fn.cross_entropy(logits, tg, reduction='none')
        loss = ce.mean()
        per = torch.stack([ce[f:f + c].mean() for f, c in zip(first, counts)]).detach()
        lp = torch.log_softmax(logits[first].detach(), -1)
        ent = -(lp.exp() * lp).sum(-1)
        return loss, dict(per_example=per, first_token_entropy=ent, n_tokens=len(tgt))

    # ---------------------------------------------------------------- pass 2: generation
    @torch.no_grad()
    def generate(self, pieces, soft=None, max_new_tokens=ANSWER_MAX_NEW, want_logits=False, first_line=True):
        """greedy; LEFT padding; model.generate(inputs_embeds, attention_mask) returns only the new tokens.
        Returns (texts, dict(ids (per example, cut at EOS), steps (decode steps incl. EOS), first_token_entropy (B,) or None))"""
        was_training = self.training
        self.eval()
        A = self.assemble([(p, s, []) for p, s, _ in pieces], soft, 'left')
        out = self.model.generate(inputs_embeds=A['embeds'], attention_mask=A['mask'], max_new_tokens=int(max_new_tokens), do_sample=False,
                                  num_beams=1, pad_token_id=self.pad, eos_token_id=self.eos, use_cache=True, return_dict_in_generate=True,
                                  output_logits=bool(want_logits))
        seqs = out.sequences
        if seqs.shape[1] > max_new_tokens:                                          # defensive: a transformers that echoes the prompt
            seqs = seqs[:, -int(max_new_tokens):]
        texts, ids, steps = [], [], []
        for row in seqs.tolist():
            cut = row.index(self.eos) if self.eos in row else len(row)
            ids.append(row[:cut])
            steps.append(min(cut + 1, len(row)))
            t = self.tok.decode(row[:cut], skip_special_tokens=True)
            texts.append((t.split('\n')[0] if first_line else t).strip())
        ent = None
        if want_logits and out.logits:
            lp = torch.log_softmax(out.logits[0].float(), -1)
            ent = -(lp.exp() * lp).sum(-1)
        if was_training:
            self.train()
        return texts, dict(ids=ids, steps=steps, first_token_entropy=ent, new_tokens=int(seqs.shape[1]))


class Pair(nn.Module):
    """Two agents (or one shared) + w; the arms."""

    def __init__(self, agents, pair, arm, max_ctx=None):
        super().__init__()
        if R2_ROUNDS is not None:
            raise NotImplementedError('R = 2 is a deferred readout (design section 4): R2_ROUNDS must stay None')
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        if pair not in MODELS:
            raise ValueError(f'pair {pair} not in {tuple(MODELS)}')
        agents = list(agents)
        if arm == 'B4':
            if len(agents) != 1:
                raise ValueError('B4 uses exactly one agent (--model)')
            self.idx = [0, 0]
        elif pair == 'hetero':
            if len(agents) != 2:
                raise ValueError('pair hetero needs two agents')
            self.idx = [0, 1]
        else:
            if len(agents) != 1:
                raise ValueError('pair same uses one shared agent')
            self.idx = [0, 0]
        self.agents = nn.ModuleList(agents)
        self.pair, self.arm, self.max_ctx = pair, arm, max_ctx
        if arm in LATENT_ARMS:
            for a in agents:
                if a.adapter is None:
                    raise ValueError(f'arm {arm}: agent {a.name} has no adapter / reader (comm=False)')
                if (arm in ('U', 'S1L1')) != a.adapter.fixed_lam:
                    raise ValueError(f'arm {arm}: agent {a.name} adapter fixed_lam={a.adapter.fixed_lam}')
        self.w_raw = nn.Parameter(torch.tensor(softplus_inv(1.0), dtype=torch.float32, device=agents[0].device)) if arm in FUSION_ARMS else None
        self.trunc = collections.Counter()

    def agent(self, i):
        return self.agents[self.idx[i]]

    @property
    def device(self):
        return self.agents[0].device

    @property
    def w(self):
        return Fn.softplus(self.w_raw) if self.w_raw is not None else None

    @property
    def w_value(self):
        return float(self.w.detach()) if self.w_raw is not None else None

    @property
    def n_agents(self):
        return 1 if self.arm == 'B4' else N_AGENTS

    def model_of(self, i):
        return self.agent(i).name

    def pieces(self, batch, i, mode, target='answer', messages=None):
        ag = self.agent(i) if i is not None else self.agents[0]
        out = []
        for b, ex in enumerate(batch):
            pc, tr = make_pieces(ag, ex, i, mode, self.max_ctx, target, None if messages is None else messages[b])
            self.trunc[tr] += 1
            out.append(pc)
        return out

    # ---------------------------------------------------------------- messages
    def encode_both(self, batch):
        """pass 1 of both agents on their own shard -> z (2, B, K, D_e), lam (2, B, K, D_e) float32"""
        zs, ls = [], []
        for i in range(N_AGENTS):
            z, lam = self.agent(i).encode(self.pieces(batch, i, 'latent', target=None))
            zs.append(z); ls.append(lam)
        return torch.stack(zs), torch.stack(ls)

    def soft_from_train(self, z, lam, arm=None, train=True, drop=None):
        """training-time messages -> list of 2 soft-token tensors (B, K, d_i). U: drop (2, B) bool per directed message
        (random with p = U_DROP_P when train and drop is None; directed edge e: src[e] -> dst[e]); S1 / S1L1: fuse_direct."""
        arm = arm or self.arm
        B = z.shape[1]
        if arm in FUSION_ARMS:
            x = FU.fuse_direct(z, lam, EDGES, self.w)
            return [self.agent(i).reader(x[i], self.agent(i).dtype) for i in range(N_AGENTS)], None
        if arm == 'U':
            src, dst = FU.directed_edges(EDGES, z.device)
            if drop is None:
                drop = (torch.rand(src.shape[0], B, device=z.device) < U_DROP_P) if train else torch.zeros(src.shape[0], B, dtype=torch.bool, device=z.device)
            drop = drop.to(z.device).bool()
            softs = []
            for i in range(N_AGENTS):
                e = int((dst == i).nonzero()[0])
                msg = z[int(src[e])] * (~drop[e]).float()[:, None, None]
                softs.append(self.agent(i).reader(msg, self.agent(i).dtype))
            return softs, drop
        raise ValueError(arm)

    def soft_from_eval(self, z, lam, arm=None, drop=None, rounds=None, wire_dtype=FU.WIRE_DTYPE):
        """evaluation-time messages: S1 / S1L1 through gbp_sync (really passed messages, fp16 wire, optional drop (2E, B) bool);
        U: the other agent's z, zeroed where dropped. Returns (softs, fused x or None, belief precision or None)."""
        arm = arm or self.arm
        B, K, D = z.shape[1], z.shape[2], z.shape[3]
        src, dst = FU.directed_edges(EDGES, z.device)
        e2 = src.shape[0]
        if drop is None:
            drop = torch.zeros(e2, B, dtype=torch.bool, device=z.device)
        drop = drop.to(z.device).bool()
        if arm in FUSION_ARMS:
            rounds = FU.tree_diameter(N_AGENTS, EDGES) if rounds is None else rounds
            full = drop[:, :, None, None].expand(e2, B, K, D)
            x, prec = FU.gbp_sync(z, lam, EDGES, self.w.detach(), rounds=rounds, drop=full, wire_dtype=wire_dtype)
            return [self.agent(i).reader(x[i], self.agent(i).dtype) for i in range(N_AGENTS)], x, prec
        if arm == 'U':
            softs = []
            for i in range(N_AGENTS):
                e = int((dst == i).nonzero()[0])
                msg = z[int(src[e])] * (~drop[e]).float()[:, None, None]
                softs.append(self.agent(i).reader(msg, self.agent(i).dtype))
            return softs, None, None
        raise ValueError(arm)

    # ---------------------------------------------------------------- one training step
    def step(self, batch, arm=None, train=True, drop=None):
        """-> (loss, info). Loss = mean of the agents' answer losses; info has loss_agent, z, lam, soft, drop (U), entropies."""
        arm = arm or self.arm
        info = {}
        if arm == 'B0':
            losses, ex = zip(*[self.agent(i).loss(self.pieces(batch, i, 'B0')) for i in range(N_AGENTS)])
        elif arm == 'B4':
            l0, e0 = self.agents[0].loss(self.pieces(batch, None, 'B4'))
            losses, ex = (l0,), (e0,)
        elif arm in LATENT_ARMS:
            z, lam = self.encode_both(batch)
            softs, used = self.soft_from_train(z, lam, arm, train, drop)
            losses, ex = zip(*[self.agent(i).loss(self.pieces(batch, i, 'latent'), softs[i]) for i in range(N_AGENTS)])
            info.update(z=z, lam=lam, soft=softs, drop=used)
        else:
            raise ValueError(arm)
        loss = torch.stack(list(losses)).mean()
        info['loss_agent'] = [float(l.detach()) for l in losses]
        info['first_token_entropy'] = [e['first_token_entropy'] for e in ex]
        return loss, info


# ================================================================ building pairs (HF or tiny)
def model_names(pair, arm, model=None):
    """the models of a run: B4 = every model of the pair (one B4 run holds both hetero models, l1_aggregate reads em_model per model)
    unless --model restricts it; the other arms = the unique models of the pair (hetero 2, same 1)"""
    if arm == 'B4' and model:
        return [model]
    return list(dict.fromkeys(MODELS[pair]))


def tiny_vocab(examples, cap=200):
    words = collections.Counter()
    for ex in examples:
        for t in (ex.get('question', ''), ex.get('answer', '')):
            words.update(t.lower().split())
        for paras in list(ex.get('shards', [])) + [ex.get('all_paragraphs', [])]:
            for title, sents in paras:
                words.update(str(title).lower().split())
                for s in sents:
                    words.update(str(s).lower().split())
    for w in ('context', 'question', 'answer', 'partner', 'says', 'facts', 'my', 'needs', ':', '?', '.', ',', 'nothing', '(', ')'):
        words[w] += 1
    return [w for w, _ in words.most_common(cap)]


def tiny_model_and_tokenizer(words, hidden, seed, bos=True, layers=2, heads=2, inter=64):
    """a random LlamaConfig model + WordLevel PreTrainedTokenizerFast ([PAD] [BOS] [EOS] [UNK]; bos=False -> no BOS token registered)"""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    kw = dict(pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    if bos:
        kw['bos_token'] = '[BOS]'
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, **kw)
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=heads, max_position_embeddings=1024, pad_token_id=0, bos_token_id=1 if bos else None, eos_token_id=2,
                      tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg)
    return model, tok


def tiny_models(examples, seed):
    words = tiny_vocab(examples)
    out = {}
    for name, hidden, bos, s in ((TINY_NAMES[0], 32, True, seed), (TINY_NAMES[1], 48, False, seed + 1)):
        out[name] = tiny_model_and_tokenizer(words, hidden, s, bos=bos)
    return out


def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=True, max_ctx=None, **agent_kw):
    """tiny: dict name -> (model, tokenizer) from tiny_models() (pair hetero = tiny-a + tiny-b, same = tiny-a)."""
    comm = arm in LATENT_ARMS
    fixed_lam = arm in ('U', 'S1L1')
    if arm == 'B4' and not model:
        raise ValueError('build_pair B4 needs the one model of this Pair (l1_train loops over model_names)')
    if tiny is not None:
        tn = [model] if arm == 'B4' else (list(TINY_NAMES) if pair == 'hetero' else [TINY_NAMES[0]])
        if any(n not in tiny for n in tn):
            raise ValueError(f'tiny models {tn} not in {list(tiny)}')
        agents = [Agent.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=comm, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, **agent_kw)
                  for n in tn]
    else:
        agents = [Agent.from_pretrained(n, device, hf_cache, lora=lora, comm=comm, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, **agent_kw)
                  for n in model_names(pair, arm, model)]
    return Pair(agents, pair, arm, max_ctx=max_ctx)


# ================================================================ optimiser / schedule / checkpoint
def param_groups(module, lr_lora, lr_adapter, weight_decay=0.0):
    lora, other = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (lora if 'lora_' in n else other).append(p)
    groups = []
    if lora:
        groups.append(dict(params=lora, lr=lr_lora, name='lora'))
    if other:
        groups.append(dict(params=other, lr=lr_adapter, name='adapter'))
    return groups, dict(lora=sum(p.numel() for p in lora), adapter=sum(p.numel() for p in other))


def build_optimizer(module, lr_lora, lr_adapter, steps, warmup, weight_decay=0.0):
    groups, counts = param_groups(module, lr_lora, lr_adapter, weight_decay)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if steps <= warmup:
            return 1.0
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    return opt, sched, counts


def count_params(module):
    tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in module.parameters())
    return dict(trainable=int(tr), total=int(tot), trainable_frac=float(tr / max(1, tot)))


def trainable_state(module):
    return {n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad}


def load_trainable_state(module, state, strict=True):
    own = dict(module.named_parameters())
    missing = [n for n in state if n not in own]
    if missing and strict:
        raise KeyError(f'{len(missing)} saved parameters not in the module, e.g. {missing[:3]}')
    with torch.no_grad():
        for n, t in state.items():
            if n in own:
                own[n].copy_(t.to(own[n].device, own[n].dtype))
    return [n for n, p in own.items() if p.requires_grad and n not in state]


def train_steps(module, step_fn, stream, steps, lr_lora, lr_adapter, warmup, clip, device, log, log_every=50, deadline=None,
                label='train', weight_decay=0.0, extra_log=None):
    """generic loop: AdamW (two lrs), warm-up + cosine, bf16 autocast on CUDA, grad clipping; step times after TIMING_WARMUP_STEPS only."""
    opt, sched, counts = build_optimizer(module, lr_lora, lr_adapter, steps, warmup, weight_decay)
    params = [p for g in opt.param_groups for p in g['params']]
    log(f'{label}: {steps} steps, lr lora {lr_lora} ({counts["lora"]:,} params) / adapter {lr_adapter} ({counts["adapter"]:,} params), warm-up {warmup}, clip {clip}')
    module.train()
    use_ac = device.type == 'cuda'
    times, losses, hist = [], [], []
    t_start = time.time()
    for s in range(steps):
        check_deadline(deadline, f'{label} step {s}')
        batch = next(stream)
        sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_ac):
            loss, info = step_fn(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'{label}: non-finite loss at step {s}')
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(float(loss.detach()))
        if (s + 1) % log_every == 0 or s + 1 == steps:
            post = times[TIMING_WARMUP_STEPS:]
            rate = f'{np.mean(post):.3f} s/step (post warm-up, {len(post)} steps)' if post else 'rate: warm-up'
            mem = f', peak {torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GB' if use_ac else ''
            ex = f' {extra_log(info)}' if extra_log else ''
            log(f'{label} step {s + 1}/{steps} loss {np.mean(losses[-log_every:]):.4f} agents {["%.3f" % x for x in info.get("loss_agent", [])]} '
                f'lr {sched.get_last_lr()[0]:.2e} gnorm {float(gn):.2f}{ex} {rate}{mem}')
            hist.append(dict(step=s + 1, loss=float(np.mean(losses[-log_every:])), lr=float(sched.get_last_lr()[0])))
    post = times[TIMING_WARMUP_STEPS:]
    return dict(steps_done=steps, seconds=time.time() - t_start, step_time_post_warmup_s=float(np.mean(post)) if post else None,
                steps_timed=len(post), timing_warmup_steps=TIMING_WARMUP_STEPS, loss_last100=float(np.mean(losses[-100:])) if losses else None,
                loss_first100=float(np.mean(losses[:100])) if losses else None, history=hist, param_counts=counts)


# ================================================================ evaluation
def _rel(owner, i):
    if owner is None:
        return 'none'
    return 'own' if int(owner) == i else 'other'


def summarise(recs):
    """records: dict(id, agent, model, em, f1, rel, owner, type [, disagree, mean_log_lam, mean_lam, frac_lo, frac_hi, entropy, steps])"""
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def pct(xs):
        m = mean(xs)
        return None if m is None else 100.0 * m

    out = dict(n=len({r['id'] for r in recs}), n_records=len(recs), em=pct([r['em'] for r in recs]), f1=pct([r['f1'] for r in recs]))
    n_ag = (max(r['agent'] for r in recs) + 1) if recs else 0
    out['em_agent'] = [pct([r['em'] for r in recs if r['agent'] == i]) for i in range(n_ag)]
    out['f1_agent'] = [pct([r['f1'] for r in recs if r['agent'] == i]) for i in range(n_ag)]
    models = list(dict.fromkeys(r['model'] for r in recs))
    out['em_model'] = {m: pct([r['em'] for r in recs if r['model'] == m]) for m in models}
    out['f1_model'] = {m: pct([r['f1'] for r in recs if r['model'] == m]) for m in models}
    out['em_by_owner'] = {rel: dict(em=pct([r['em'] for r in recs if r['rel'] == rel]), n=sum(r['rel'] == rel for r in recs)) for rel in ('own', 'other', 'none', 'all')}
    out['em_by_owner_agent'] = [{rel: pct([r['em'] for r in recs if r['rel'] == rel and r['agent'] == i]) for rel in ('own', 'other', 'none', 'all')} for i in range(n_ag)]
    out['em_by_type'] = {t: dict(em=pct([r['em'] for r in recs if r['type'] == t]), n=sum(r['type'] == t for r in recs)) for t in sorted({str(r['type']) for r in recs})}
    out['answer_decode_steps_mean'] = mean([r.get('steps') for r in recs])
    if any('disagree' in r for r in recs):
        per = {}
        for r in recs:
            per.setdefault(r['id'], dict(d=r['disagree'], em=[]))['em'].append(r['em'])
        ds = [v['d'] for v in per.values()]
        out['disagreement'] = dict(mean=mean(ds), std=float(np.std(ds)) if ds else None, corr_with_em=pearson(ds, [np.mean(v['em']) for v in per.values()]),
                                   n=len(ds))
    if any('mean_log_lam' in r for r in recs):
        out['lam'] = []
        for i in range(n_ag):
            ri = [r for r in recs if r['agent'] == i]
            out['lam'].append(dict(mean_log_lam=mean([r['mean_log_lam'] for r in ri]), frac_at_lower=mean([r['frac_lo'] for r in ri]),
                                   frac_at_upper=mean([r['frac_hi'] for r in ri]), mean_lam=mean([r['mean_lam'] for r in ri]),
                                   corr_mean_lam_entropy=pearson([r['mean_lam'] for r in ri], [r['entropy'] for r in ri]),
                                   corr_mean_log_lam_entropy=pearson([r['mean_log_lam'] for r in ri], [r['entropy'] for r in ri])))
    if any(r.get('entropy') is not None for r in recs):
        out['first_token_entropy_mean'] = mean([r.get('entropy') for r in recs])
    return out


def _record(ex, i, model, pred, steps=None, ent=None, rel=None):
    gold = ex['answer']
    return dict(id=ex.get('id'), agent=i, model=model, pred=pred, gold=gold, em=int(DATA.exact_match(pred, gold)), f1=float(DATA.f1_score(pred, gold)),
                rel=rel if rel is not None else _rel(ex.get('answer_owner'), i), owner=ex.get('answer_owner'), type=ex.get('type'), steps=steps, entropy=ent)


@torch.no_grad()
def eval_plain(pair, examples, eval_batch=16, max_new_tokens=ANSWER_MAX_NEW, deadline=None, keep_preds=False, log=None):
    """B0 (both agents, own shard) / B4 (one agent, all paragraphs); no soft tokens"""
    pair.eval()
    dev = pair.device
    recs, t_gen = [], 0.0
    pair.trunc.clear()
    mode = pair.arm
    agents = [None] if mode == 'B4' else list(range(N_AGENTS))
    for bi, batch in enumerate(chunks(examples, eval_batch)):
        check_deadline(deadline, 'eval_plain')
        for i in agents:
            ag = pair.agents[0] if i is None else pair.agent(i)
            pcs = pair.pieces(batch, i, mode, target=None)
            sync(dev); t0 = time.perf_counter()
            texts, g = ag.generate(pcs, None, max_new_tokens, want_logits=True)
            sync(dev); t_gen += time.perf_counter() - t0
            for b, ex in enumerate(batch):
                ent = float(g['first_token_entropy'][b]) if g['first_token_entropy'] is not None else None
                recs.append(_record(ex, 0 if i is None else i, ag.name, texts[b], g['steps'][b], ent, rel='all' if i is None else None))
        if log and (bi + 1) % 20 == 0:
            log(f'eval {mode}: {len(recs)} records')
    out = summarise(recs)
    out.update(answer_gen_s_per_example=t_gen / max(1, len(recs)), truncation=dict(pair.trunc), wire_bytes_per_example=0, sender_latency_s_per_example=0.0)
    if keep_preds:
        out['records'] = recs
    return out


@torch.no_grad()
def eval_latent(pair, examples, drop_p=0.0, eval_batch=16, seed=0, max_new_tokens=ANSWER_MAX_NEW, c_check=True, deadline=None, keep_preds=False, log=None):
    """U / S1 / S1L1: pass 1 -> messages through gbp_sync (S1 / S1L1) or the raw z (U) with each directed message dropped
    independently per example with probability drop_p (seeded) -> soft tokens -> greedy answers. Readouts of design section 5."""
    arm = pair.arm
    if arm not in LATENT_ARMS:
        raise ValueError(arm)
    pair.eval()
    dev = pair.device
    g = torch.Generator().manual_seed(int(seed) * 1009 + int(round(drop_p * 100)))
    diam = FU.tree_diameter(N_AGENTS, EDGES)
    k, d_e = pair.agent(0).k, pair.agent(0).d_e
    recs, t_sender, t_gen = [], 0.0, 0.0
    c_assert, c_detail, p1_detail = None, None, None
    pair.trunc.clear()
    n_dropped = 0
    for bi, batch in enumerate(chunks(examples, eval_batch)):
        check_deadline(deadline, 'eval_latent')
        B = len(batch)
        drop = (torch.rand(2 * len(EDGES), B, generator=g) < drop_p)
        n_dropped += int(drop.sum())
        sync(dev); t0 = time.perf_counter()
        z, lam = pair.encode_both(batch)
        softs, x, prec = pair.soft_from_eval(z, lam, arm, drop.to(dev), rounds=diam, wire_dtype=FU.WIRE_DTYPE)
        sync(dev); t_sender += time.perf_counter() - t0
        if c_check and c_assert is None and arm in FUSION_ARMS:
            zf, lf, wf = z.float(), lam.float(), pair.w.detach().float()
            xd = FU.fuse_direct(zf, lf, EDGES, wf)
            xg, _ = FU.gbp_sync(zf, lf, EDGES, wf, rounds=diam, drop=None, wire_dtype=None)
            diff, scale = float((xd - xg).abs().max()), float(xd.abs().max())
            c_assert = bool(diff <= C_ASSERT_TOL * (1.0 + scale))
            c_detail = dict(max_abs_diff=diff, max_abs_x=scale, tol=C_ASSERT_TOL * (1.0 + scale), rounds=diam, w=float(wf), batch_examples=B)
        if drop_p >= 1.0 and p1_detail is None:
            own = [pair.agent(i).reader(z[i], pair.agent(i).dtype) for i in range(N_AGENTS)]
            if arm in FUSION_ARMS:
                md = max(float((softs[i].float() - own[i].float()).abs().max()) for i in range(N_AGENTS))
                p1_detail = dict(max_abs_diff_soft_vs_reader_own_z=md, equal=bool(md <= 1e-5 * (1.0 + max(float(o.float().abs().max()) for o in own))))
            else:
                zero = [pair.agent(i).reader(torch.zeros_like(z[i]), pair.agent(i).dtype) for i in range(N_AGENTS)]
                md = max(float((softs[i].float() - zero[i].float()).abs().max()) for i in range(N_AGENTS))
                p1_detail = dict(max_abs_diff_soft_vs_reader_zero=md, equal=bool(md == 0.0))
        res = FU.edge_residual(z, EDGES)[0]                                        # (B,) ||z_0 - z_1||
        log_lam = lam.clamp_min(1e-30).log()
        for i in range(N_AGENTS):
            ag = pair.agent(i)
            pcs = pair.pieces(batch, i, 'latent', target=None)
            sync(dev); t0 = time.perf_counter()
            texts, gi = ag.generate(pcs, softs[i], max_new_tokens, want_logits=True)
            sync(dev); t_gen += time.perf_counter() - t0
            ll = log_lam[i].reshape(B, -1)
            for b, ex in enumerate(batch):
                ent = float(gi['first_token_entropy'][b]) if gi['first_token_entropy'] is not None else None
                r = _record(ex, i, ag.name, texts[b], gi['steps'][b], ent)
                r.update(disagree=float(res[b]), mean_log_lam=float(ll[b].mean()), mean_lam=float(lam[i, b].mean()),
                         frac_lo=float((ll[b] <= LOG_LAM_MIN + 1e-6).float().mean()), frac_hi=float((ll[b] >= LOG_LAM_MAX - 1e-6).float().mean()))
                recs.append(r)
        if log and (bi + 1) % 20 == 0:
            log(f'eval {arm} p={drop_p}: {len(recs)} records')
    out = summarise(recs)
    n = max(1, len(examples))
    per_msg = FU.wire_bytes(k, d_e) if arm in FUSION_ARMS else k * d_e * torch.empty(0, dtype=FU.WIRE_DTYPE).element_size()
    out.update(drop_p=drop_p, n_messages_dropped=n_dropped, n_messages=2 * len(EDGES) * len(examples), sender_latency_s_per_example=t_sender / n,
               answer_gen_s_per_example=t_gen / max(1, len(recs)), wire_bytes_per_example=2 * len(EDGES) * per_msg, wire_bytes_per_message=per_msg,
               wire_dtype=str(FU.WIRE_DTYPE), gbp_rounds=diam if arm in FUSION_ARMS else None, w=float(pair.w.detach()) if pair.w is not None else None,
               truncation=dict(pair.trunc), c_assert=c_assert, c_assert_detail=c_detail, absence_p1_detail=p1_detail)
    if keep_preds:
        out['records'] = recs
    return out


def absence_curve(pair, examples, ps=ABSENCE_PS, **kw):
    """{str(p): eval_latent(...)}; the p = 0 entry carries the C assertion"""
    out = {}
    for p in ps:
        out[str(float(p))] = eval_latent(pair, examples, drop_p=float(p), c_check=(p == 0.0), **kw)
    return out
