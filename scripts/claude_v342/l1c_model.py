"""l1c_model.py (Claude, 2026-10-31): batch L1c model side (pre-registration D8-86) on top of the PINNED L1 files l1_model.py / l1_fusion.py /
l1_data.py (imported, never edited). What changes against L1 / L1b (D8-86 section 3):
  (a) the receiver is the FROZEN B0 checkpoint of the same seed (base + B0 LoRA, loaded by build_pair from runs/l1/B0-hetero-s<seed>/trainable.pt
      and frozen); pass 1 (the sender reading its own hidden layer) runs with the LoRA DISABLED and without gradient;
  (b) SenderAdapter2: K slot queries -> N_BLOCKS x [slot self-attention, cross-attention to the layer-L hidden states, MLP] -> LN -> z, precision
      head as L1 (the message is now 32 ORDERED tokens, one block of fixed queries cannot locate and copy);
  (c) Decoder D_i replaces the Reader: logits = LN(W x) . E_i^T over the receiver's vocabulary (E_i = its frozen input embedding), soft tokens =
      softmax(logits / tau) @ E_i (a convex combination of real token embeddings); hard = E_i[argmax];
  (d) the training loss is the position-wise CE of the decoder logits against the partner's gold supporting sentences in the receiver's
      tokenizer, cut to K tokens and right-padded with the filler token (a space). No answer loss, nothing trainable in the LLMs.
  (e) arms Cu (one-way projection x_i = z_j, 30 % training dropout as U) and Cs (GBP consensus with learned lam / w as S1); the pinned fusion,
      GBP evaluation path, C assertion and absence curve are reused unchanged through PairC.soft_from_eval.
Everything else (data, prompts, K = 32, d_e = 256, read layer, EM / F1, owner buckets) is L1's.
"""
from __future__ import annotations
import contextlib
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402

BATCH = 'L1c'
PREREG = 'D8-86'
ARMS = ('Cu', 'Cs')
BASE_ARM = {'Cu': 'U', 'Cs': 'S1'}
FILLER_TEXT = ' '                               # short messages are right-padded with this token (D8-86 3d)
TAU = 1.0                                       # softmax temperature of the decoder (fixed)
N_BLOCKS = 2                                    # SenderAdapter2 blocks (D8-86 3b)
ABSENCE_PS = (0.0, 1.0)
BASELINES = ('none', 'gold32')
SCRIPTS = ('l1c_model.py', 'l1c_train.py', 'l1_model.py', 'l1_train.py', 'l1_fusion.py', 'l1_data.py')
MSG_EXAMPLES = 8                                # decoded messages kept in the summary


# ================================================================ (b) the sender
class _Block(nn.Module):
    """[LN -> slot self-attention] + [LN -> cross-attention to h] + [LN -> MLP], all residual, float32"""

    def __init__(self, d_e, d_model, n_heads):
        super().__init__()
        if d_e % n_heads:
            raise ValueError(f'd_e {d_e} not divisible by n_heads {n_heads}')
        self.n_heads = n_heads
        self.ln_self = nn.LayerNorm(d_e)
        self.self_attn = nn.MultiheadAttention(d_e, n_heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_e)
        self.q_proj = nn.Linear(d_e, d_e)
        self.k_proj = nn.Linear(d_model, d_e)
        self.v_proj = nn.Linear(d_model, d_e)
        self.o_proj = nn.Linear(d_e, d_e)
        self.ln_mlp = nn.LayerNorm(d_e)
        self.mlp = nn.Sequential(nn.Linear(d_e, 4 * d_e), nn.GELU(), nn.Linear(4 * d_e, d_e))

    def forward(self, x, h, mask):
        B, K, D = x.shape
        H, dh = self.n_heads, D // self.n_heads
        a = self.ln_self(x)
        x = x + self.self_attn(a, a, a, need_weights=False)[0]
        a = self.ln_cross(x)
        q = self.q_proj(a).view(B, K, H, dh).transpose(1, 2)
        k = self.k_proj(h).view(B, -1, H, dh).transpose(1, 2)
        v = self.v_proj(h).view(B, -1, H, dh).transpose(1, 2)
        att = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool()[:, None, None, :])
        x = x + self.o_proj(att.transpose(1, 2).reshape(B, K, D))
        return x + self.mlp(self.ln_mlp(x))


class SenderAdapter2(nn.Module):
    """P_i (D8-86 3b). Interface of l1_model.SenderAdapter: forward(h, mask) -> (z (B, K, d_e) float32, log_lam (B, K, d_e) float32);
    attributes k, d_e, n_heads, fixed_lam, lam_head. lam = 1 at init (zero-initialised head), clamped as L1; fixed_lam -> lam = 1, no head."""

    def __init__(self, d_model, k=M.K_SLOTS, d_e=M.D_E, n_heads=M.N_HEADS, fixed_lam=False, n_blocks=N_BLOCKS):
        super().__init__()
        self.k, self.d_e, self.n_heads, self.fixed_lam, self.n_blocks = int(k), int(d_e), int(n_heads), bool(fixed_lam), int(n_blocks)
        self.queries = nn.Parameter(torch.randn(self.k, self.d_e) * 0.02)
        self.blocks = nn.ModuleList([_Block(self.d_e, d_model, self.n_heads) for _ in range(self.n_blocks)])
        self.ln_out = nn.LayerNorm(self.d_e)                   # the trivial-solution guard of L1 (z cannot collapse to 0)
        if self.fixed_lam:
            self.lam_head = None
        else:
            self.lam_head = nn.Linear(self.d_e, self.d_e)
            nn.init.zeros_(self.lam_head.weight)
            nn.init.zeros_(self.lam_head.bias)

    def forward(self, h, mask):
        B = h.shape[0]
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            x = self.queries[None].expand(B, -1, -1)
            for blk in self.blocks:
                x = blk(x, h, mask)
            z = self.ln_out(x)
            if self.lam_head is None:
                log_lam = torch.zeros_like(z)
            else:
                log_lam = self.lam_head(x).clamp(M.LOG_LAM_MIN, M.LOG_LAM_MAX)
        return z, log_lam


# ================================================================ (c) the decoder
class Decoder(nn.Module):
    """D_i (D8-86 3c): x (B, K, d_e) -> logits (B, K, V) through the receiver's FROZEN input embedding matrix E (a float32, non-persistent copy:
    never trained, never saved); soft tokens = softmax(logits / tau) @ E (convex combination of real token embeddings), hard = E[argmax].
    forward(x, dtype, hard) has the Reader's signature so that the pinned evaluation code runs unchanged."""

    def __init__(self, d_e, embed_weight, tau=TAU):
        super().__init__()
        d_model = int(embed_weight.shape[1])
        self.proj = nn.Linear(d_e, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.tau = float(tau)
        self.register_buffer('E32', embed_weight.detach().float().clone(), persistent=False)

    @property
    def vocab(self):
        return int(self.E32.shape[0])

    def logits(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            hdn = self.ln(self.proj(x.float()))
            return hdn @ self.E32.t()

    def soft_from_logits(self, lg, hard=False):
        if hard:
            return self.E32[lg.argmax(-1)]
        return torch.softmax(lg / self.tau, -1) @ self.E32

    def forward(self, x, dtype=None, hard=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = self.soft_from_logits(self.logits(x), hard)
        return s.to(dtype) if dtype is not None else s

    def tokens(self, ids, dtype=None):
        """token ids (B, K) -> their real embeddings (B, K, d_model) (the gold-32 baseline)"""
        s = self.E32[ids]
        return s.to(dtype) if dtype is not None else s


# ================================================================ (a) the agent
class AgentC(M.Agent):
    """l1_model.Agent with SenderAdapter2 as `adapter` and a Decoder as `reader` (the attribute name is kept: eval_latent's absence detail calls
    agent.reader). encode() = pass 1 with the LoRA disabled and no gradient (the pretrained model's layer L); the gradient reaches the adapter only.
    The LoRA (the B0 receiver) is loaded and frozen by build_pair."""

    def __init__(self, model, tokenizer, name, *, tau=TAU, n_blocks=N_BLOCKS, **kw):
        super().__init__(model, tokenizer, name, **kw)
        if self.adapter is None:
            raise ValueError(f'{name}: AgentC needs comm=True')
        fixed = self.adapter.fixed_lam
        self.adapter = SenderAdapter2(self.d_model, self.k, self.d_e, kw.get('n_heads', M.N_HEADS), fixed, n_blocks).to(self.device)
        self.reader = Decoder(self.d_e, self.embed.weight, tau).to(self.device)
        ids = self.ids(FILLER_TEXT)
        self.filler = int(ids[0]) if ids else int(self.pad)
        self.filler_is_pad = not bool(ids)
        self._lora_params = [p for n, p in self.named_parameters() if 'lora_' in n]

    def freeze_lora(self):
        """the receiver's LoRA is never trained. PEFT's disable_adapter() context re-enables the adapter layers on exit through set_adapter(),
        which also sets requires_grad = True on them: re-freeze after every pass 1 (and after loading the receiver state)."""
        for p in self._lora_params:
            p.requires_grad_(False)

    def encode(self, pieces):
        """pass 1 on [BOS?] prefix suffix (own shard + question) with the LoRA DISABLED, no gradient -> z, lam (float32, grad into the adapter)"""
        A = self.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
        ctx = self.model.disable_adapter() if self.lora else contextlib.nullcontext()
        try:
            with torch.no_grad(), ctx:
                out = self.decoder(input_ids=A['ids'], attention_mask=A['mask'], output_hidden_states=True, use_cache=False)
                h = out.hidden_states[self.layer].detach()
        finally:
            self.freeze_lora()
        z, log_lam = self.adapter(h, A['mask'])
        return z, log_lam.exp()

    def target_ids(self, ex, partner):
        """-> (K receiver-token ids of the partner's gold facts, cut to K and right-padded with the filler; the message length before padding)"""
        ids = [int(t) for t in self.ids(M.DATA.text_sender_target(ex, partner))[:self.k]]
        return ids + [self.filler] * (self.k - len(ids)), len(ids)


# ================================================================ (e) the pair
class PairC(M.Pair):
    """l1_model.Pair of AgentC's; self.arm is the L1 base arm (what the pinned fusion / evaluation code reads), self.arm_c the L1c arm.
    self.hard switches the evaluation-time soft tokens to the hard (argmax) message."""

    def __init__(self, agents, pair, arm, max_ctx=None):
        if arm not in ARMS:
            raise ValueError(f'arm {arm} not in {ARMS}')
        super().__init__(agents, pair, BASE_ARM[arm], max_ctx=max_ctx)
        self.arm_c = arm
        self.hard = False
        self._tgt_cache = {}
        self.msg_hist = []                        # (loss, token accuracy) of every training step

    # ---------------------------------------------------------------- targets
    def targets(self, batch, i):
        """-> (ids (B, K) long on the device, lengths (B,) list): receiver-i tokens of the partner's (1 - i) gold facts"""
        ag = self.agent(i)
        ids, lens = [], []
        for ex in batch:
            key = (ag.name, ex.get('id'), i)
            t = self._tgt_cache.get(key) if ex.get('id') is not None else None
            if t is None:
                t = ag.target_ids(ex, 1 - i)
                if ex.get('id') is not None:
                    self._tgt_cache[key] = t
            ids.append(t[0])
            lens.append(t[1])
        return torch.tensor(ids, dtype=torch.long, device=self.device), lens

    # ---------------------------------------------------------------- messages (x per receiver), training and evaluation paths
    def messages_train(self, z, lam, train=True, drop=None):
        """-> (xs: list of (B, K, d_e) per receiver, drop used (U) or None). Cs: fuse_direct (exact, differentiable); Cu: the partner's z, zeroed
        where dropped (p = U_DROP_P in training when drop is None)."""
        B = z.shape[1]
        if self.arm in M.FUSION_ARMS:
            with torch.autocast(device_type=z.device.type, enabled=False):
                x = FU.fuse_direct(z.float(), lam.float(), M.EDGES, self.w)
            return [x[i] for i in range(M.N_AGENTS)], None
        src, dst = FU.directed_edges(M.EDGES, z.device)
        if drop is None:
            drop = (torch.rand(src.shape[0], B, device=z.device) < M.U_DROP_P) if train else torch.zeros(src.shape[0], B, dtype=torch.bool, device=z.device)
        drop = drop.to(z.device).bool()
        xs = []
        for i in range(M.N_AGENTS):
            e = int((dst == i).nonzero()[0])
            xs.append(z[int(src[e])] * (~drop[e]).float()[:, None, None])
        return xs, drop

    def messages_eval(self, z, lam, drop=None, rounds=None, wire_dtype=FU.WIRE_DTYPE):
        """evaluation-time messages: Cs through gbp_sync (really passed, fp16 wire), Cu the partner's z zeroed where dropped -> (xs, x, prec)"""
        B, K, D = z.shape[1], z.shape[2], z.shape[3]
        src, dst = FU.directed_edges(M.EDGES, z.device)
        e2 = src.shape[0]
        if drop is None:
            drop = torch.zeros(e2, B, dtype=torch.bool, device=z.device)
        drop = drop.to(z.device).bool()
        if self.arm in M.FUSION_ARMS:
            rounds = FU.tree_diameter(M.N_AGENTS, M.EDGES) if rounds is None else rounds
            full = drop[:, :, None, None].expand(e2, B, K, D)
            x, prec = FU.gbp_sync(z, lam, M.EDGES, self.w.detach(), rounds=rounds, drop=full, wire_dtype=wire_dtype)
            return [x[i] for i in range(M.N_AGENTS)], x, prec
        xs = []
        for i in range(M.N_AGENTS):
            e = int((dst == i).nonzero()[0])
            xs.append(z[int(src[e])] * (~drop[e]).float()[:, None, None])
        return xs, None, None

    def soft_from_train(self, z, lam, arm=None, train=True, drop=None):
        xs, used = self.messages_train(z, lam, train, drop)
        return [self.agent(i).reader(xs[i], self.agent(i).dtype, hard=False) for i in range(M.N_AGENTS)], used

    def soft_from_eval(self, z, lam, arm=None, drop=None, rounds=None, wire_dtype=FU.WIRE_DTYPE):
        xs, x, prec = self.messages_eval(z, lam, drop, rounds, wire_dtype)
        return [self.agent(i).reader(xs[i], self.agent(i).dtype, hard=self.hard) for i in range(M.N_AGENTS)], x, prec

    # ---------------------------------------------------------------- (d) one training step: message CE only
    def step(self, batch, arm=None, train=True, drop=None):
        """-> (loss, info): loss = mean over the two receivers of the position-wise CE of the decoder logits against the K target tokens"""
        z, lam = self.encode_both(batch)
        xs, used = self.messages_train(z, lam, train, drop)
        losses, accs, msg_accs = [], [], []
        for i in range(M.N_AGENTS):
            lg = self.agent(i).reader.logits(xs[i])                               # (B, K, V) float32
            tgt, lens = self.targets(batch, i)
            ce = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            pred = lg.detach().argmax(-1)
            hit = (pred == tgt)
            msk = torch.zeros_like(hit)
            for b, n in enumerate(lens):
                msk[b, :n] = True
            losses.append(ce)
            accs.append(float(hit.float().mean()))
            msg_accs.append(float(hit[msk].float().mean()) if bool(msk.any()) else float('nan'))
        loss = torch.stack(losses).mean()
        info = dict(z=z, lam=lam, drop=used, loss_agent=[float(l.detach()) for l in losses], acc_agent=accs, msg_acc_agent=msg_accs,
                    msg_acc=float(np.nanmean(msg_accs)), acc=float(np.mean(accs)))
        if train:
            self.msg_hist.append((float(loss.detach()), info['msg_acc']))
        return loss, info

    def msg_summary(self, window=100):
        h = self.msg_hist
        if not h:
            return dict(steps=0)
        lo, ac = np.array([t[0] for t in h]), np.array([t[1] for t in h])
        return dict(steps=len(h), loss_first=float(lo[:window].mean()), loss_last=float(lo[-window:].mean()),
                    msg_acc_first=float(np.nanmean(ac[:window])), msg_acc_last=float(np.nanmean(ac[-window:])),
                    history=[dict(step=s + 1, loss=float(lo[max(0, s + 1 - window):s + 1].mean()), msg_acc=float(np.nanmean(ac[max(0, s + 1 - window):s + 1])))
                             for s in range(window - 1, len(h), window)])


# ================================================================ construction
def build_pair(arm, pair, device, model=None, hf_cache=None, tiny=None, lora=True, grad_ckpt=False, max_ctx=None, receiver_state=None, tau=TAU,
               n_blocks=N_BLOCKS, **agent_kw):
    """as l1b_model.build_pair for the L1c arms. receiver_state: the B0 checkpoint's trainable state (its LoRA; loaded into both agents, then
    frozen). Without it (tests) the LoRA stays at its zero init (= the base model). Trainable afterwards: adapters, decoders, w (Cs)."""
    if arm not in ARMS:
        raise ValueError(f'arm {arm} not in {ARMS}')
    base = BASE_ARM[arm]
    fixed_lam = base in ('U', 'S1L1')
    agent_kw = dict(agent_kw)
    agent_kw.setdefault('lora_dropout', 0.0)
    if tiny is not None:
        tn = list(M.TINY_NAMES) if pair == 'hetero' else [M.TINY_NAMES[0]]
        if any(n not in tiny for n in tn):
            raise ValueError(f'tiny models {tn} not in {list(tiny)}')
        agents = [AgentC.from_pretrained_objects(tiny[n][0], tiny[n][1], n, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau,
                                                 n_blocks=n_blocks, **agent_kw) for n in tn]
    else:
        agents = [AgentC.from_pretrained(n, device, hf_cache, lora=lora, comm=True, fixed_lam=fixed_lam, grad_ckpt=grad_ckpt, tau=tau, n_blocks=n_blocks, **agent_kw)
                  for n in M.model_names(pair, base, model)]
    pc = PairC(agents, pair, arm, max_ctx=max_ctx)
    if receiver_state is not None:
        lora_keys = [k for k in receiver_state if 'lora_' in k]
        if not lora_keys:
            raise ValueError('receiver_state has no LoRA tensors (expected the B0 checkpoint state)')
        M.load_trainable_state(pc, {k: receiver_state[k] for k in lora_keys}, strict=True)
    for a in pc.agents:
        a.freeze_lora()
    for n, p in pc.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pc


def load_receiver(path):
    """runs/l1/B0-hetero-s<seed>/trainable.pt -> (state dict, info dict(path, sha256, arm, seed, models))"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck
    info = dict(path=os.path.abspath(path), sha256=M.file_sha256(path), arm=ck.get('arm') if isinstance(ck, dict) else None,
                seed=ck.get('seed') if isinstance(ck, dict) else None, models=ck.get('models') if isinstance(ck, dict) else None, n_tensors=len(state))
    return state, info


# ================================================================ readouts beyond the pinned eval_latent
@torch.no_grad()
def eval_hard(pair, examples, **kw):
    """the pinned eval_latent with the hard (argmax) message instead of the convex combination"""
    pair.hard = True
    try:
        return M.eval_latent(pair, examples, drop_p=0.0, c_check=False, **kw)
    finally:
        pair.hard = False


@torch.no_grad()
def eval_baseline(pair, examples, kind, eval_batch=16, max_new_tokens=M.ANSWER_MAX_NEW, deadline=None, keep_preds=False, log=None):
    """the frozen receiver with (none) nothing at the slot = the B0 prompt, or (gold32) the K target tokens' real embeddings at the slot (the ceiling
    of this channel). Same prompt / assembly / generation as the latent arms."""
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
            pcs = pair.pieces(batch, i, 'latent', target=None)
            soft = None
            if kind == 'gold32':
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


@torch.no_grad()
def message_metrics(pair, examples, eval_batch=16, deadline=None, log=None, n_examples=MSG_EXAMPLES):
    """the decoded (argmax) messages at p = 0 against the targets: token accuracy over the K slots and over the message tokens, token F1 of the
    decoded text against the partner's gold facts (full) and against the K-token target text -> dict per agent + a few examples"""
    pair.eval()
    acc_all, acc_msg, f1_full, f1_trunc, n_tok, exs = [[] for _ in range(M.N_AGENTS)], [[] for _ in range(M.N_AGENTS)], [[] for _ in range(M.N_AGENTS)], \
        [[] for _ in range(M.N_AGENTS)], [[] for _ in range(M.N_AGENTS)], []
    for batch in M.chunks(examples, eval_batch):
        M.check_deadline(deadline, 'message_metrics')
        z, lam = pair.encode_both(batch)
        xs, _, _ = pair.messages_eval(z, lam)
        for i in range(M.N_AGENTS):
            ag = pair.agent(i)
            lg = ag.reader.logits(xs[i])
            pred = lg.argmax(-1)
            tgt, lens = pair.targets(batch, i)
            hit = (pred == tgt)
            for b, ex in enumerate(batch):
                n = lens[b]
                acc_all[i].append(float(hit[b].float().mean()))
                if n > 0:
                    acc_msg[i].append(float(hit[b, :n].float().mean()))
                pid = pred[b].tolist()
                cut = len(pid)
                for j in range(len(pid) - 1, -1, -1):                              # strip trailing fillers
                    if pid[j] != ag.filler:
                        break
                    cut = j
                text = ag.tok.decode(pid[:cut], skip_special_tokens=True).strip()
                gold_full = M.DATA.text_sender_target(ex, 1 - i).strip()
                gold_trunc = ag.tok.decode(tgt[b, :n].tolist(), skip_special_tokens=True).strip()
                f1_full[i].append(float(M.DATA.f1_score(text, gold_full)) if gold_full else 0.0)
                f1_trunc[i].append(float(M.DATA.f1_score(text, gold_trunc)) if gold_trunc else 0.0)
                n_tok[i].append(cut)
                if len(exs) < n_examples:
                    exs.append(dict(id=ex.get('id'), agent=i, pred=text, gold_trunc=gold_trunc, f1_trunc=f1_trunc[i][-1], f1_full=f1_full[i][-1]))
    mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    return dict(n=len(examples), token_acc_agent=[mean(a) for a in acc_all], msg_token_acc_agent=[mean(a) for a in acc_msg],
                token_acc=mean([x for a in acc_all for x in a]), msg_token_acc=mean([x for a in acc_msg for x in a]),
                f1_full_agent=[mean(a) for a in f1_full], f1_trunc_agent=[mean(a) for a in f1_trunc],
                f1_full=mean([x for a in f1_full for x in a]), f1_trunc=mean([x for a in f1_trunc for x in a]),
                n_tokens_mean=mean([x for a in n_tok for x in a]), examples=exs)
