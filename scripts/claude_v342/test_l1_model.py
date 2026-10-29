"""Unit tests for batch L1's model side (D8-82): l1_model.py, l1_train.py, l1_text.py, chain_day9_L1.sh. CPU only, no network, no data:
two tiny random LlamaConfig models (hidden 32 / 48, 2 layers, 2 heads) with WordLevel tokenizers built here (tiny-b has NO BOS token so
both code paths run) and three synthetic examples in the l1_data schema. The prompt builders of l1_data.py are used when the file
exists (the fallback in l1_model otherwise).
Run: CUDA_VISIBLE_DEVICES= python -B -m pytest -q scripts/claude_v342/test_l1_model.py scripts/claude_v342/test_l1_fusion.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pytest  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

import l1_fusion as FU  # noqa: E402
import l1_model as M  # noqa: E402

CHAIN = os.path.join(HERE, 'chain_day9_L1.sh')
TRAIN = os.path.join(HERE, 'l1_train.py')
TEXT = os.path.join(HERE, 'l1_text.py')
K, DE = 8, 16                     # small K / D_e for speed (the pre-registered 32 / 256 are the CLI defaults, checked below)
torch.manual_seed(0)


def _p(title, *sents):
    return [title, list(sents)]


# l1_data schema: shards = 2 lists of [title, sentences] with exactly one gold paragraph each; gold_title = [title in shard 0, title in
# shard 1]; sup[agent] = the supporting sentences of the gold paragraph in shard `agent`; answer_owner = the shard whose gold paragraph
# contains the answer (None when both / neither do)
EXAMPLES = [
    dict(id='ex1', question='who wrote the red book ?', answer='alice', type='bridge', level='hard',
         all_paragraphs=[_p('alpha', 'alice wrote the red book .', 'it was long .'), _p('beta', 'bob likes tea .'), _p('gamma', 'the red book is famous .'), _p('delta', 'carol sings .')],
         shards=[[_p('alpha', 'alice wrote the red book .', 'it was long .'), _p('beta', 'bob likes tea .')], [_p('gamma', 'the red book is famous .'), _p('delta', 'carol sings .')]],
         gold_title=['alpha', 'gamma'], answer_owner=0, sup=[['alice wrote the red book .'], ['the red book is famous .']]),
    dict(id='ex2', question='which city is older , rome or paris ?', answer='rome', type='comparison', level='medium',
         all_paragraphs=[_p('rome', 'rome is very old .'), _p('paris', 'paris is old .'), _p('tea', 'tea is a drink .'), _p('sea', 'the sea is blue and wide .', 'fish swim .')],
         shards=[[_p('rome', 'rome is very old .'), _p('tea', 'tea is a drink .')], [_p('paris', 'paris is old .'), _p('sea', 'the sea is blue and wide .', 'fish swim .')]],
         gold_title=['rome', 'paris'], answer_owner=None, sup=[['rome is very old .'], ['paris is old .']]),
    dict(id='ex3', question='what does bob like ?', answer='tea', type='bridge', level='easy',
         all_paragraphs=[_p('beta', 'bob likes tea .'), _p('alpha', 'alice wrote the red book .'), _p('delta', 'carol sings .'), _p('sea', 'the sea is blue and wide .')],
         shards=[[_p('delta', 'carol sings .'), _p('alpha', 'alice wrote the red book .')], [_p('beta', 'bob likes tea .'), _p('sea', 'the sea is blue and wide .')]],
         gold_title=['delta', 'beta'], answer_owner=1, sup=[['carol sings .'], ['bob likes tea .']]),
]


def _words():
    words = set()
    for ex in EXAMPLES:
        for t in (ex['question'], ex['answer']):
            words |= set(t.lower().split())
        for title, sents in ex['all_paragraphs']:
            words.add(title.lower())
            for s in sents:
                words |= set(s.lower().split())
    words |= {'context', 'question', 'answer', 'partner', 'says', 'facts', 'my', 'needs', ':', '?', '.', ',', 'nothing', '(', ')', 'useful', 'for'}
    return sorted(words)


def _tiny(hidden, seed, bos):
    """a random LlamaConfig model + a WordLevel PreTrainedTokenizerFast ([PAD] [BOS] [EOS] [UNK]; bos=False: no bos token registered)"""
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
    for w in _words():
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    kw = dict(pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    if bos:
        kw['bos_token'] = '[BOS]'
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, **kw)
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=0, bos_token_id=1 if bos else None, eos_token_id=2, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg), tok


def _agents(arm, pair='hetero'):
    comm = arm in M.LATENT_ARMS
    fixed = arm in ('U', 'S1L1')
    a = M.Agent.from_pretrained_objects(*_tiny(32, 1, True), 'tiny-a', comm=comm, fixed_lam=fixed, k=K, d_e=DE, n_heads=2)
    if pair == 'same' or arm == 'B4':
        return [a]
    b = M.Agent.from_pretrained_objects(*_tiny(48, 2, False), 'tiny-b', comm=comm, fixed_lam=fixed, k=K, d_e=DE, n_heads=2)
    return [a, b]


def _pair(arm, pair='hetero', max_ctx=None):
    return M.Pair(_agents(arm, pair), pair, arm, max_ctx=max_ctx)


# ---------------------------------------------------------------- 0. constants = the pre-registration
def test_preregistered_numbers_are_the_defaults():
    import l1_train as T
    import l1_text as X
    assert (M.K_SLOTS, M.D_E) == (32, 256) and M.LORA_R == 16 and M.LORA_ALPHA == 32 and M.LORA_DROPOUT == 0.05
    assert abs(M.LOG_LAM_MIN - torch.log(torch.tensor(1e-3)).item()) < 1e-6 and abs(M.LOG_LAM_MAX - torch.log(torch.tensor(1e3)).item()) < 1e-6
    assert set(M.LORA_TARGETS) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    assert M.U_DROP_P == 0.3 and tuple(M.ABSENCE_PS) == (0.0, 0.3, 0.6, 1.0) and M.ANSWER_MAX_NEW == 16 and M.EDGES == [(0, 1)]
    assert int(M.LAYER_FRAC * 28) == 18 and int(M.LAYER_FRAC * 16) == 10          # floor(2/3 n) for Qwen3-1.7B / Llama-3.2-1B
    assert M.MODELS['hetero'] == ('Qwen/Qwen3-1.7B', 'meta-llama/Llama-3.2-1B-Instruct') and M.MODELS['same'][0] == 'Qwen/Qwen3-1.7B'
    d = vars(T.build_parser().parse_args(['--arm', 'S1', '--pair', 'hetero', '--seed', '1', '--train', 'a', '--val', 'b', '--fresh', 'c', '--out', 'd']))
    assert (d['steps'], d['batch'], d['lr_lora'], d['lr_adapter'], d['warmup'], d['clip'], d['max_ctx_tokens'], d['eval_batch']) == (2500, 8, 2e-4, 1e-3, 100, 1.0, 900, 16)
    assert d['k'] == 32 and d['d_e'] == 256 and d['layer_frac'] == 0.6667 and d['absence_ps'] == '0.0,0.3,0.6,1.0'
    x = vars(X.build_parser().parse_args(['--arm', 'T', '--pair', 'hetero', '--seed', '1', '--train', 'a', '--val', 'b', '--fresh', 'c', '--out', 'd']))
    assert (x['budget'], x['budgets'], x['steps'], x['stage2_steps'], x['stage2_examples'], x['batch'], x['lr_lora']) == (64, '16,64,256', 2500, 625, 5000, 8, 2e-4)
    assert M.R2_ROUNDS is None and M.TIMING_WARMUP_STEPS == 50


def test_r2_flag_errors_when_set(monkeypatch):
    monkeypatch.setattr(M, 'R2_ROUNDS', 2)
    with pytest.raises(NotImplementedError):
        _pair('S1')


# ---------------------------------------------------------------- 1. soft tokens sit exactly between prefix and suffix
@pytest.mark.parametrize('which', [0, 1])
def test_soft_tokens_inserted_between_prefix_and_suffix(which):
    agents = _agents('S1')
    ag = agents[which]
    ex = EXAMPLES[0]
    pre_t, suf_t = M.DATA.latent_prompt(ex, which)
    assert pre_t and suf_t and ex['question'] in (pre_t + suf_t)
    (p, s, a), tr = M.make_pieces(ag, ex, which, 'latent')
    assert tr == 0
    exp_pre = ([ag.bos] if ag.bos is not None else []) + ag.ids(pre_t)
    assert p == exp_pre and s == ag.ids(suf_t) and a == ag.ids(M.DATA.answer_target(ex)) + [ag.eos]
    assert (ag.bos is not None) == (which == 0), 'tiny-a has a BOS token, tiny-b has none'
    soft = torch.randn(1, K, ag.d_model)
    A = ag.assemble([(p, s, a)], soft, 'right')
    emb, E = A['embeds'][0], ag.embed.weight
    n = len(p)
    assert torch.allclose(emb[:n], E[p]) and torch.allclose(emb[n:n + K], soft[0]) and torch.allclose(emb[n + K:n + K + len(s)], E[s])
    assert torch.allclose(emb[n + K + len(s):n + K + len(s) + len(a)], E[a]) and A['ans_start'][0] == n + K + len(s)
    assert A['mask'][0].sum() == n + K + len(s) + len(a) and A['embeds'].shape[1] == n + K + len(s) + len(a)
    # left padding (generation): the block moves right by the pad length, nothing else changes
    (p2, s2, _), _ = M.make_pieces(ag, EXAMPLES[1], which, 'latent', target=None)
    soft2 = torch.randn(2, K, ag.d_model)
    L = ag.assemble([(p, s, []), (p2, s2, [])], soft2, 'left')
    T = L['embeds'].shape[1]
    for b, (pp, ss) in enumerate(((p, s), (p2, s2))):
        off = T - (len(pp) + K + len(ss))
        assert L['start'][b] == off and (L['mask'][b, :off] == 0).all() and (L['mask'][b, off:] == 1).all()
        assert torch.allclose(L['embeds'][b, off + len(pp):off + len(pp) + K], soft2[b])


def test_b0_b4_insert_nothing_and_truncation_counts():
    pair = _pair('B0')
    for i in (0, 1):
        (p, s, a), _ = M.make_pieces(pair.agent(i), EXAMPLES[0], i, 'B0')
        assert s == [] and p[len(p) - len(pair.agent(i).ids(M.DATA.no_comm_prompt(EXAMPLES[0], i))):] == pair.agent(i).ids(M.DATA.no_comm_prompt(EXAMPLES[0], i))
        A = pair.agent(i).assemble([(p, s, a)], None, 'right')
        assert A['k'] == 0 and A['embeds'].shape[1] == len(p) + len(a)
    b4 = _pair('B4')
    (p, s, a), tr = M.make_pieces(b4.agents[0], EXAMPLES[0], None, 'B4')
    assert s == [] and tr == 0 and len(p) > len(M.make_pieces(b4.agents[0], EXAMPLES[0], 0, 'B0')[0][0]), 'B4 reads all paragraphs'
    # --max-ctx-tokens: paragraphs are shrunk proportionally and the truncation is counted
    ag = b4.agents[0]
    full = len(p)
    (p3, s3, _), tr3 = M.make_pieces(ag, EXAMPLES[0], None, 'B4', max_ctx=full - 6)
    assert tr3 == 1 and len(p3) + len(s3) <= full - 6 and p3[-3:] == p[-3:], 'paragraphs are shrunk; the question and "Answer:" at the tail survive'
    (p5, _, _), tr5 = M.make_pieces(ag, EXAMPLES[0], None, 'B4', max_ctx=full - 14)
    assert tr5 == 1 and len(p5) <= full - 14 and p5[-3:] == p[-3:]
    for i in (0, 1):                                                       # latent mode: the suffix (question) is never touched by any cut
        (pl, sl, _), trl = M.make_pieces(pair.agent(i), EXAMPLES[1], i, 'latent', max_ctx=len(M.make_pieces(pair.agent(i), EXAMPLES[1], i, 'latent')[0][0]) - 2)
        assert trl >= 1 and sl == pair.agent(i).ids(M.DATA.latent_prompt(EXAMPLES[1], i)[1])
    (p4, _, _), tr4 = M.make_pieces(ag, EXAMPLES[0], 0, 'latent', max_ctx=10 ** 6)
    assert tr4 == 0
    loss, info = b4.step(EXAMPLES[:2], 'B4')
    assert torch.isfinite(loss) and len(info['loss_agent']) == 1


# ---------------------------------------------------------------- 2. the arms
def test_u_uses_other_agents_z_through_own_reader_and_zeroes_dropped_messages():
    pair = _pair('U')
    pair.eval()
    assert pair.w is None and all(a.adapter.lam_head is None for a in pair.agents)
    z, lam = pair.encode_both(EXAMPLES[:3])
    assert z.shape == (2, 3, K, DE) and torch.all(lam == 1)
    drop = torch.tensor([[False, True, False], [True, False, False]])          # directed 0: 0->1, 1: 1->0
    softs, used = pair.soft_from_train(z, lam, 'U', train=True, drop=drop)
    assert torch.equal(used, drop)
    keep1 = (~drop[0]).float()[:, None, None]                                  # what agent 1 receives (from agent 0)
    keep0 = (~drop[1]).float()[:, None, None]
    assert torch.allclose(softs[1], pair.agent(1).reader(z[0] * keep1, pair.agent(1).dtype))
    assert torch.allclose(softs[0], pair.agent(0).reader(z[1] * keep0, pair.agent(0).dtype))
    assert torch.allclose(softs[1][1], pair.agent(1).reader(torch.zeros(K, DE)[None], pair.agent(1).dtype)[0])
    assert not torch.allclose(softs[0][1], pair.agent(0).reader(torch.zeros(K, DE)[None], pair.agent(0).dtype)[0])
    # training-time dropout: with p = 0.3 over many draws some messages are zeroed, at eval none
    torch.manual_seed(3)
    n_drop = sum(int(pair.soft_from_train(z, lam, 'U', train=True)[1].sum()) for _ in range(40))
    assert 0 < n_drop < 40 * 6
    assert int(pair.soft_from_train(z, lam, 'U', train=False)[1].sum()) == 0
    loss, info = pair.step(EXAMPLES[:2], 'U', train=True)
    assert torch.isfinite(loss) and info['drop'].shape == (2, 2)


def test_s1l1_has_lam_one_and_no_precision_parameters():
    pair = _pair('S1L1')
    names = [n for n, _ in pair.named_parameters()]
    assert not any('lam_head' in n for n in names) and pair.w is not None and abs(pair.w_value - 1.0) < 1e-6 and pair.w.requires_grad
    z, lam = pair.encode_both(EXAMPLES[:2])
    assert torch.all(lam == 1.0) and lam.shape == z.shape
    s1 = _pair('S1')
    assert any('lam_head' in n for n in [n for n, _ in s1.named_parameters()])
    n_s1 = sum(p.numel() for p in s1.parameters() if p.requires_grad)
    n_l1 = sum(p.numel() for p in pair.parameters() if p.requires_grad)
    assert n_s1 - n_l1 == 2 * (DE * DE + DE), 'S1 - S1L1 = the two precision heads'
    z2, lam2 = s1.encode_both(EXAMPLES[:2])
    assert torch.allclose(lam2, torch.ones_like(lam2)), 'precision head starts at lam = 1'
    assert (lam2.log() >= M.LOG_LAM_MIN).all() and (lam2.log() <= M.LOG_LAM_MAX).all()
    # the z LayerNorm guard: every slot has unit variance
    assert torch.allclose(z.var(-1, unbiased=False), torch.ones(z.shape[:-1]), atol=1e-3)


def test_s1_loss_finite_and_gradients_reach_everything_on_both_agents():
    pair = _pair('S1')
    pair.train()
    loss, info = pair.step(EXAMPLES[:2], 'S1', train=True)
    assert torch.isfinite(loss) and len(info['loss_agent']) == 2 and info['drop'] is None
    loss.backward()
    for i, ag in enumerate(pair.agents):
        lora = [p for n, p in ag.named_parameters() if 'lora_' in n and p.requires_grad]
        assert lora and sum(float(p.grad.abs().sum()) for p in lora if p.grad is not None) > 0, f'agent {i}: no LoRA gradient'
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in ag.adapter.parameters()), f'agent {i}: adapter gradient missing'
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in ag.reader.parameters()), f'agent {i}: reader gradient missing'
        assert all(p.grad is None for n, p in ag.lm.named_parameters() if 'lora_' not in n), 'base weights must stay frozen'
    assert pair.w_raw.grad is not None and float(pair.w_raw.grad.abs()) > 0
    # gradient checkpointing is on and base weights frozen; B0 still trains its LoRA
    b0 = _pair('B0')
    b0.train()
    l0, _ = b0.step(EXAMPLES[:2], 'B0')
    l0.backward()
    assert all(sum(float(p.grad.abs().sum()) for n, p in a.named_parameters() if 'lora_' in n and p.grad is not None) > 0 for a in b0.agents)
    assert b0.w is None


def test_train_steps_updates_and_reports_post_warmup_only(monkeypatch):
    pair = _pair('S1')
    monkeypatch.setattr(M, 'TIMING_WARMUP_STEPS', 2)
    before = {n: p.detach().clone() for n, p in pair.named_parameters() if p.requires_grad}
    msgs = []
    tr = M.train_steps(pair, lambda b: pair.step(b, 'S1', True), M.batch_stream(EXAMPLES, 2, 0), 4, 2e-4, 1e-3, 1, 1.0, torch.device('cpu'), msgs.append, log_every=2)
    assert tr['steps_done'] == 4 and tr['steps_timed'] == 2 and tr['step_time_post_warmup_s'] > 0 and tr['timing_warmup_steps'] == 2
    changed = [n for n, p in pair.named_parameters() if p.requires_grad and not torch.equal(p.detach(), before[n])]
    assert any('lora_' in n for n in changed) and any('adapter' in n for n in changed) and any('reader' in n for n in changed) and 'w_raw' in changed
    assert tr['param_counts']['lora'] > 0 and tr['param_counts']['adapter'] > 0


# ---------------------------------------------------------------- 3. evaluation: gbp_sync == fuse_direct, absence, C
def test_eval_messages_gbp_equals_direct_and_p1_falls_back_to_own_z():
    pair = _pair('S1')
    pair.eval()
    with torch.no_grad():
        pair.w_raw.fill_(0.3)
        for a in pair.agents:                                                  # non-trivial precisions
            a.adapter.lam_head.weight.normal_(0, 0.5); a.adapter.lam_head.bias.normal_(0, 1.0)
        z, lam = pair.encode_both(EXAMPLES[:3])
        assert not torch.allclose(lam, torch.ones_like(lam))
        softs_gbp, x, prec = pair.soft_from_eval(z, lam, 'S1', drop=None, wire_dtype=None)
        softs_dir, _ = pair.soft_from_train(z, lam, 'S1', train=False)
        for i in range(2):
            assert torch.allclose(softs_gbp[i], softs_dir[i], atol=1e-5)
        assert torch.allclose(x, FU.fuse_direct(z, lam, M.EDGES, pair.w), atol=1e-5) and (prec >= lam).all()
        softs_wire, _, _ = pair.soft_from_eval(z, lam, 'S1', drop=None)        # the fp16 wire of the design: close, not equal
        assert all(torch.allclose(softs_wire[i], softs_dir[i], atol=5e-2) for i in range(2))
        full = torch.ones(2, 3, dtype=torch.bool)
        softs_p1, x1, _ = pair.soft_from_eval(z, lam, 'S1', drop=full, wire_dtype=None)
        assert torch.allclose(x1, z, atol=1e-6)
        for i in range(2):
            assert torch.allclose(softs_p1[i], pair.agent(i).reader(z[i], pair.agent(i).dtype), atol=1e-6)
        one = torch.tensor([[False, False, False], [True, True, True]])      # 1->0 lost: agent 0 keeps its own z, agent 1 still fuses
        s_one, x_one, _ = pair.soft_from_eval(z, lam, 'S1', drop=one, wire_dtype=None)
        assert torch.allclose(x_one[0], z[0], atol=1e-6) and torch.allclose(x_one[1], x[1], atol=1e-5)
    r0 = M.eval_latent(pair, EXAMPLES[:3], drop_p=0.0, eval_batch=2, seed=1, max_new_tokens=3)
    assert r0['c_assert'] is True and r0['c_assert_detail']['max_abs_diff'] <= r0['c_assert_detail']['tol'] and r0['n_messages_dropped'] == 0
    assert r0['c_assert_detail']['dtype'] == 'float64' and r0['c_assert_detail']['max_abs_diff_wire_vs_direct'] is not None  # review M1
    assert r0['gbp_rounds'] == 1 and r0['wire_bytes_per_example'] == 2 * FU.wire_bytes(K, DE) and r0['sender_latency_s_per_example'] > 0
    assert r0['n'] == 3 and r0['n_records'] == 6 and len(r0['em_agent']) == 2 and set(r0['em_model']) == {'tiny-a', 'tiny-b'}
    assert set(r0['em_by_owner']) == {'own', 'other', 'none', 'all'} and r0['em_by_owner']['own']['n'] == 2 and r0['em_by_owner']['other']['n'] == 2 and r0['em_by_owner']['none']['n'] == 2
    assert 'disagreement' in r0 and r0['disagreement']['n'] == 3 and r0['disagreement']['mean'] > 0
    assert len(r0['lam']) == 2 and all(0 <= l['frac_at_lower'] <= 1 and 0 <= l['frac_at_upper'] <= 1 for l in r0['lam'])
    r1 = M.eval_latent(pair, EXAMPLES[:3], drop_p=1.0, eval_batch=2, seed=1, max_new_tokens=3, c_check=False)
    assert r1['c_assert'] is None and r1['n_messages_dropped'] == 6 and r1['absence_p1_detail']['equal'] is True
    ru = M.eval_latent(_pair('U'), EXAMPLES[:3], drop_p=1.0, eval_batch=2, seed=1, max_new_tokens=3)
    assert ru['absence_p1_detail']['equal'] is True and ru['c_assert'] is None and ru['wire_bytes_per_example'] == 2 * K * DE * 2


def test_c_assert_false_is_written_not_skipped(monkeypatch):
    pair = _pair('S1')
    real = FU.gbp_sync

    def broken(z, lam, edges, w, rounds, drop=None, wire_dtype=None):
        m, p = real(z, lam, edges, w, rounds, drop, wire_dtype)
        return m + 1.0, p

    monkeypatch.setattr(FU, 'gbp_sync', broken)
    r = M.eval_latent(pair, EXAMPLES[:2], drop_p=0.0, eval_batch=2, seed=1, max_new_tokens=2)
    assert r['c_assert'] is False and r['c_assert_detail']['max_abs_diff'] > r['c_assert_detail']['tol']


def test_absence_curve_has_the_four_p_values():
    pair = _pair('S1L1')
    curve = M.absence_curve(pair, EXAMPLES[:2], eval_batch=2, seed=5, max_new_tokens=2)
    assert list(curve) == ['0.0', '0.3', '0.6', '1.0']
    assert curve['0.0']['c_assert'] is True and curve['1.0']['n_messages_dropped'] == 4 and all('em' in v for v in curve.values())
    assert curve['0.3']['c_assert'] is None


# ---------------------------------------------------------------- 4. generation
@pytest.mark.parametrize('which', [0, 1])
def test_generate_left_padded_batch_matches_single_and_returns_new_tokens_only(which):
    ag = _agents('S1')[which]
    ag.eval()
    torch.manual_seed(11)
    with torch.no_grad():                                                     # make the tiny LM less flat so greedy decoding is decisive
        ag.lm.lm_head.weight.mul_(4.0) if not ag.lm.config.tie_word_embeddings else ag.embed.weight.mul_(4.0)
    pcs = [M.make_pieces(ag, ex, which, 'latent', target=None)[0] for ex in EXAMPLES[:2]]
    assert len(pcs[0][0]) != len(pcs[1][0]), 'two different prompt lengths'
    soft = torch.randn(2, K, ag.d_model)
    texts, g = ag.generate(pcs, soft, max_new_tokens=5, want_logits=True)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts) and g['new_tokens'] <= 5 and len(g['ids']) == 2 and len(g['steps']) == 2
    assert g['first_token_entropy'].shape == (2,)
    for b in range(2):
        t1, g1 = ag.generate([pcs[b]], soft[b:b + 1], max_new_tokens=5)
        assert t1[0] == texts[b] and g1['ids'][0] == g['ids'][b]
    assert all(len(r) <= 5 for r in g['ids']) and all(1 <= s <= 5 for s in g['steps'])
    # transformers returns only the new tokens when inputs_embeds is given (no prompt echo)
    A = ag.assemble([(p, s, []) for p, s, _ in pcs], soft, 'left')
    out = ag.model.generate(inputs_embeds=A['embeds'], attention_mask=A['mask'], max_new_tokens=5, do_sample=False, pad_token_id=ag.pad, eos_token_id=ag.eos)
    assert out.shape[1] <= 5
    # no soft tokens (B0 path) and the plain text split
    t0, _ = ag.generate([(pcs[0][0], pcs[0][1], [])], None, max_new_tokens=3)
    assert len(t0) == 1


# ---------------------------------------------------------------- 5. the CLIs end to end (--tiny, subprocess) and the summary schema
def _write_jsonl(path, exs):
    with open(path, 'w', encoding='utf-8') as fh:
        for e in exs:
            fh.write(json.dumps(e) + '\n')


def _validate(s, arm, pair, seed):
    """every key of l1_aggregate.summary_schema() (c_assert: S1 / S1L1 only) and l1_aggregate.validate_summary == []; a fallback key
    list when l1_aggregate.py is not there yet"""
    try:
        import l1_aggregate as AGG
    except Exception:
        AGG = None
    if AGG is not None:
        problems = AGG.validate_summary(s, arm, pair, seed)
        assert problems == [], problems
        for k in AGG.summary_schema():
            assert k in s or (k == 'c_assert' and arm not in AGG.GBP_ARMS), f'summary.json misses key {k}'
        for split in ('val', 'fresh'):
            for k in AGG.summary_schema()['val']:
                assert k in s[split], f'{split}.{k} missing'
        return
    for k in ('arm', 'pair', 'seed', 'val', 'fresh', 'scripts_sha') + (('c_assert',) if arm in ('S1', 'S1L1') else ()):
        assert k in s, f'summary.json misses key {k}'
    for split in ('val', 'fresh'):
        for k in ('em', 'f1', 'em_agent', 'em_model'):
            assert k in s[split], f'{split}.{k} missing'
    assert 'c_assert' not in s or s['c_assert'] is True


def _run(args, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-B'] + args, capture_output=True, text=True, env=env, timeout=timeout)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return r


@pytest.fixture(scope='module')
def data_dir():
    d = tempfile.mkdtemp(prefix='l1_test_')
    _write_jsonl(os.path.join(d, 'train.jsonl'), EXAMPLES * 2)
    _write_jsonl(os.path.join(d, 'val.jsonl'), EXAMPLES)
    _write_jsonl(os.path.join(d, 'fresh.jsonl'), EXAMPLES[:2])
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_l1_train_tiny_writes_summary_with_schema_keys(data_dir):
    out = os.path.join(data_dir, 'S1-hetero-s7')
    _run([TRAIN, '--arm', 'S1', '--pair', 'hetero', '--seed', '7', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'),
          '--fresh', os.path.join(data_dir, 'fresh.jsonl'), '--out', out, '--steps', '3', '--batch', '2', '--eval-batch', '2', '--tiny', '--k', '4', '--d-e', '8',
          '--n-heads', '2', '--max-new-tokens', '3', '--log-every', '1', '--warmup', '1', '--keep-preds'])
    s = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
    _validate(s, 'S1', 'hetero', 7)
    assert s['arm'] == 'S1' and s['pair'] == 'hetero' and s['seed'] == 7 and s['c_assert'] is True and s['complete'] is True and s['steps'] == 3
    assert list(s['absence']) == ['0.0', '0.3', '0.6', '1.0'] and s['val']['n'] == 3 and s['fresh']['n'] == 2 and len(s['val']['em_agent']) == 2
    assert set(s['val']['em_model']) == {'tiny-a', 'tiny-b'} and 0 <= s['val']['em'] <= 100 and 0 <= s['val']['f1'] <= 100
    assert s['scripts_sha']['l1_model.py'] == M.file_sha256(os.path.join(HERE, 'l1_model.py')) and s['scripts_sha']['l1_fusion.py'] == M.file_sha256(FU.__file__)
    assert s['timings']['step_time_post_warmup_s'] is None, 'no rate before the 50-step warm-up'
    assert s['val']['wire_bytes_per_example'] == 2 * FU.wire_bytes(4, 8) and s['w'] is not None and s['params']['trainable'] > 0
    ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
    assert all(('lora_' in n) or ('adapter' in n) or ('reader' in n) or n == 'w_raw' for n in ck['state']) and 'w_raw' in ck['state']
    assert os.path.exists(os.path.join(out, 'log.txt')) and os.path.exists(os.path.join(out, 'preds_val.json'))
    # never overwrite
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-B', TRAIN, '--arm', 'S1', '--pair', 'hetero', '--seed', '7', '--train', os.path.join(data_dir, 'train.jsonl'), '--val',
                        os.path.join(data_dir, 'val.jsonl'), '--fresh', os.path.join(data_dir, 'fresh.jsonl'), '--out', out, '--tiny'], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and 'never overwrite' in (r.stderr + r.stdout)


def test_l1_train_tiny_b0_same_and_b4_hetero_both_models(data_dir):
    outs = {}
    for arm, pair in (('B0', 'same'), ('B4', 'hetero')):
        out = os.path.join(data_dir, f'{arm}-{pair}-s3')
        _run([TRAIN, '--arm', arm, '--pair', pair, '--seed', '3', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'),
              '--fresh', os.path.join(data_dir, 'fresh.jsonl'), '--out', out, '--steps', '2', '--batch', '2', '--eval-batch', '3', '--tiny', '--max-new-tokens', '2',
              '--warmup', '1'])
        outs[arm] = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
        _validate(outs[arm], arm, pair, 3)
    b0, b4 = outs['B0'], outs['B4']
    assert 'c_assert' not in b0 and b0['absence'] is None and len(b0['val']['em_agent']) == 2 and list(b0['val']['em_model']) == ['tiny-a'] and b0['models'] == ['tiny-a']
    assert b0['val']['em_by_owner']['own']['n'] == 2 and b0['val']['em_by_owner']['other']['n'] == 2 and b0['val']['em_by_owner']['none']['n'] == 2
    # one B4 run = both models of the hetero pair, trained one after the other, em_model per model (what l1_aggregate's P0 reads)
    assert b4['models'] == ['tiny-a', 'tiny-b'] and b4['model'] is None and 'c_assert' not in b4 and b4['absence'] is None
    assert list(b4['val']['em_model']) == ['tiny-a', 'tiny-b'] and len(b4['val']['em_agent']) == 2 and b4['val']['em_by_owner']['all']['n'] == 6 and b4['val']['n'] == 3
    assert set(b4['val']['per_model']) == {'tiny-a', 'tiny-b'} and set(b4['train']) == {'tiny-a', 'tiny-b'} and len(b4['agents']) == 2
    assert abs(b4['val']['em'] - (b4['val']['em_model']['tiny-a'] + b4['val']['em_model']['tiny-b']) / 2) < 1e-9
    ck = torch.load(os.path.join(data_dir, 'B4-hetero-s3', 'trainable.pt'), map_location='cpu', weights_only=False)
    assert set(ck['state']) == {'tiny-a', 'tiny-b'} and all('lora_' in n for st in ck['state'].values() for n in st)


def test_l1_text_tiny_runs_t_and_t0_end_to_end(data_dir):
    out = os.path.join(data_dir, 'T-hetero-s5')
    common = ['--pair', 'hetero', '--seed', '5', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'), '--fresh',
              os.path.join(data_dir, 'fresh.jsonl'), '--tiny', '--batch', '2', '--eval-batch', '2', '--max-new-tokens', '2', '--budget', '4', '--budgets', '2,4,6',
              '--warmup', '1', '--log-every', '1']
    _run([TEXT, '--arm', 'T', '--out', out, '--steps', '2', '--stage2-steps', '1', '--stage2-examples', '4'] + common)
    s = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
    _validate(s, 'T', 'hetero', 5)
    assert s['arm'] == 'T' and s['phase'] == 'all' and 'c_assert' not in s and s['budget'] == 4 and sorted(s['budget_sweep']['val']) == ['2', '4', '6']
    assert list(s['absence']) == ['0.0', '0.3', '0.6', '1.0'] and s['sweep256'] == 'included' and s['budgets_done'] == [2, 4, 6] and s['readouts_incomplete'] is None
    assert s['sender_train']['steps_done'] == 2 and s['receiver_train']['stage1']['steps_done'] == 2 and s['receiver_train']['stage2']['steps_done'] == 1
    assert s['val']['sender_latency_s_per_example'] > 0 and s['val']['sender_decode_steps_mean'] >= 1 and s['val']['message_f1_vs_gold'] is not None
    assert s['budget_sweep']['val']['6']['sender']['seconds_per_example'] > 0 and 'derived_from_budget' in s['budget_sweep']['val']['2']['sender']
    ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
    assert ck['phase'] == 'all' and set(ck['sender']) == {'tiny-a', 'tiny-b'} == set(ck['receiver']) and all('lora_' in n for st in ck['receiver'].values() for n in st)
    msgs = json.load(open(os.path.join(out, 'messages.json'), encoding='utf-8'))
    assert len(msgs['val']['4']['texts']) == 2 and len(msgs['val']['4']['texts'][0]) == 3 and len(msgs['val']['6']['token_ids'][1]) == 3 and len(msgs['subset']['idx']) == 4
    out0 = os.path.join(data_dir, 'T0-hetero-s5')
    _run([TEXT, '--arm', 'T0', '--out', out0, '--receiver-from', out] + common)
    s0 = json.load(open(os.path.join(out0, 'summary.json'), encoding='utf-8'))
    _validate(s0, 'T0', 'hetero', 5)
    assert s0['arm'] == 'T0' and 'c_assert' not in s0 and s0['sender_train'] is None and s0['receiver_train']['loaded_from']['path'] == os.path.abspath(out) and s0['val']['n'] == 3
    assert s0['budgets'] == [2, 4, 6] and not os.path.exists(os.path.join(out0, 'trainable.pt'))


def _sha(p):
    return M.file_sha256(p)


def test_l1_text_phases_sender_receiver_sweep_and_t0(data_dir):
    """the lease-sized T jobs: --phase sender -> receiver (--sender-from) -> sweep256 (--sender-from --receiver-from), then T0 with --budgets"""
    base = ['--pair', 'hetero', '--seed', '6', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'), '--fresh',
            os.path.join(data_dir, 'fresh.jsonl'), '--tiny', '--batch', '2', '--eval-batch', '2', '--max-new-tokens', '2', '--budget', '4', '--sweep-budget', '6',
            '--warmup', '1', '--log-every', '1']
    ds, dr, dw, d0 = (os.path.join(data_dir, n) for n in ('Tsend-hetero-s6', 'T-hetero-s6', 'Tsweep-hetero-s6', 'T0-hetero-s6'))
    # sender phase
    _run([TEXT, '--arm', 'T', '--phase', 'sender', '--out', ds, '--steps', '2', '--stage2-examples', '4', '--budgets', '2,4,6'] + base)
    assert os.path.exists(os.path.join(ds, 'summary_sender.json')) and os.path.exists(os.path.join(ds, 'messages.json')) and os.path.exists(os.path.join(ds, 'trainable.pt'))
    assert not os.path.exists(os.path.join(ds, 'summary.json'))
    ss = json.load(open(os.path.join(ds, 'summary_sender.json'), encoding='utf-8'))
    assert ss['arm'] == 'T' and ss['phase'] == 'sender' and ss['sender_train']['steps_done'] == 2 and ss['val']['sender_latency_s_per_example'] > 0
    assert set(ss['messages']) == {'val', 'fresh', 'subset'} and ss['messages']['val']['decode_steps_mean'] >= 1 and ss['messages']['val']['message_f1_vs_gold'] is not None
    assert 'scripts_sha' in ss and ss['scripts_sha']['l1_text.py'] == _sha(TEXT)
    ck = torch.load(os.path.join(ds, 'trainable.pt'), map_location='cpu', weights_only=False)
    assert ck['phase'] == 'sender' and set(ck['sender']) == {'tiny-a', 'tiny-b'} and 'receiver' not in ck
    mj = json.load(open(os.path.join(ds, 'messages.json'), encoding='utf-8'))
    assert mj['budget'] == 4 and len(mj['subset']['idx']) == 4 and len(mj['subset']['example_ids']) == 4 and set(mj['val']) == {'4'}
    assert len(mj['val']['4']['texts'][0]) == 3 and len(mj['fresh']['4']['token_ids'][1]) == 2 and len(mj['val']['4']['n_tokens'][0]) == 3
    # receiver phase without --sender-from fails clearly
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-B', TEXT, '--arm', 'T', '--phase', 'receiver', '--out', os.path.join(data_dir, 'T-bad')] + base, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and '--sender-from' in (r.stderr + r.stdout)
    # receiver phase
    _run([TEXT, '--arm', 'T', '--phase', 'receiver', '--out', dr, '--sender-from', ds, '--steps', '2', '--stage2-steps', '1', '--budgets', '2,4,6'] + base)
    s = json.load(open(os.path.join(dr, 'summary.json'), encoding='utf-8'))
    _validate(s, 'T', 'hetero', 6)
    assert s['phase'] == 'receiver' and s['budgets'] == [2, 4] and s['budgets_done'] == [2, 4] and s['sweep256'] == 'pending' and s['readouts_incomplete'] is None
    assert sorted(s['budget_sweep']['val']) == ['2', '4'] and list(s['absence']) == ['0.0', '0.3', '0.6', '1.0'] and 'c_assert' not in s
    assert s['sender_from']['trainable_sha256'] == _sha(os.path.join(ds, 'trainable.pt')) and s['sender_from']['messages_sha256'] == _sha(os.path.join(ds, 'messages.json'))
    assert s['sender_from']['path'] == os.path.abspath(ds) and s['sender_from']['summary_sender']['messages']['val']['decode_steps_mean'] >= 1
    assert s['receiver_train']['stage1']['steps_done'] == 2 and s['receiver_train']['stage2']['steps_done'] == 1 and s['val']['sender_latency_s_per_example'] > 0
    ckr = torch.load(os.path.join(dr, 'trainable.pt'), map_location='cpu', weights_only=False)
    assert ckr['phase'] == 'receiver' and set(ckr['receiver']) == {'tiny-a', 'tiny-b'} and 'sender' not in ckr
    # sweep256 phase
    _run([TEXT, '--arm', 'T', '--phase', 'sweep256', '--out', dw, '--sender-from', ds, '--receiver-from', dr, '--budgets', '2,4,6'] + base)
    assert os.path.exists(os.path.join(dw, 'summary_sweep256.json')) and not os.path.exists(os.path.join(dw, 'summary.json')) and not os.path.exists(os.path.join(dw, 'trainable.pt'))
    sw = json.load(open(os.path.join(dw, 'summary_sweep256.json'), encoding='utf-8'))
    assert sw['phase'] == 'sweep256' and sw['sweep_budget'] == 6 and 0 <= sw['val']['em'] <= 100 and sw['val']['sender_latency_s_per_example'] > 0 and sw['fresh']['n'] == 2
    assert sw['sender_from']['trainable_sha256'] == _sha(os.path.join(ds, 'trainable.pt')) and sw['receiver_from']['trainable_sha256'] == _sha(os.path.join(dr, 'trainable.pt'))
    assert os.path.exists(os.path.join(dw, 'messages_6.json'))
    # T0 from the receiver-phase dir with the chain's --budgets 16,64 (here 2,4)
    _run([TEXT, '--arm', 'T0', '--out', d0, '--receiver-from', dr, '--budgets', '2,4'] + base)
    s0 = json.load(open(os.path.join(d0, 'summary.json'), encoding='utf-8'))
    _validate(s0, 'T0', 'hetero', 6)
    assert s0['budgets'] == [2, 4] and s0['sweep256'] == 'pending' and s0['receiver_train']['loaded_from']['phase'] == 'receiver' and s0['sender_train'] is None
    # a phase never overwrites its final file; sweep256 refuses a missing --receiver-from
    r = subprocess.run([sys.executable, '-B', TEXT, '--arm', 'T', '--phase', 'sender', '--out', ds] + base, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and 'never overwrite' in (r.stderr + r.stdout)
    r = subprocess.run([sys.executable, '-B', TEXT, '--arm', 'T', '--phase', 'sweep256', '--out', os.path.join(data_dir, 'Tsweep-bad'), '--sender-from', ds] + base,
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and '--receiver-from' in (r.stderr + r.stdout)


def test_l1_train_b4_per_model_runs_and_merge(data_dir):
    """the hetero B4 of the chain: two --model runs (valid per-model summaries) + --merge-b4 -> the B4-<pair>-s<seed> summary the aggregator reads"""
    common = ['--arm', 'B4', '--pair', 'hetero', '--seed', '4', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'), '--fresh',
              os.path.join(data_dir, 'fresh.jsonl'), '--steps', '2', '--batch', '2', '--eval-batch', '3', '--tiny', '--max-new-tokens', '2', '--warmup', '1']
    dq, dl, dm = (os.path.join(data_dir, n) for n in ('B4q-hetero-s4', 'B4l-hetero-s4', 'B4-hetero-s4'))
    parts = {}
    for d, model in ((dq, 'tiny-a'), (dl, 'tiny-b')):
        _run([TRAIN, '--out', d, '--model', model] + common)
        parts[model] = json.load(open(os.path.join(d, 'summary.json'), encoding='utf-8'))
        _validate(parts[model], 'B4', 'hetero', 4)
        assert parts[model]['model'] == model and parts[model]['models'] == [model] and list(parts[model]['val']['em_model']) == [model] and len(parts[model]['val']['em_agent']) == 1
    _run([TRAIN, '--merge-b4', dq, dl, '--out', dm])
    m = json.load(open(os.path.join(dm, 'summary.json'), encoding='utf-8'))
    _validate(m, 'B4', 'hetero', 4)
    assert m['merged'] is True and m['models'] == ['tiny-a', 'tiny-b'] and m['model'] is None and 'c_assert' not in m and m['absence'] is None
    assert list(m['val']['em_model']) == ['tiny-a', 'tiny-b'] and list(m['fresh']['em_model']) == ['tiny-a', 'tiny-b'] and len(m['val']['em_agent']) == 2
    for model in ('tiny-a', 'tiny-b'):
        assert m['val']['em_model'][model] == parts[model]['val']['em'] and m['fresh']['f1_model'][model] == parts[model]['fresh']['f1']
        assert m['parts'][model]['summary_sha256'] == _sha(os.path.join(dq if model == 'tiny-a' else dl, 'summary.json'))
        assert m['config']['parts'][model]['model'] == model and model in m['train'] and model in m['timings']
    assert abs(m['val']['em'] - (parts['tiny-a']['val']['em'] + parts['tiny-b']['val']['em']) / 2) < 1e-9 and m['val']['em_by_owner']['all']['n'] == 6
    assert m['scripts_sha'] == parts['tiny-a']['scripts_sha'] and set(m['val']['per_model']) == {'tiny-a', 'tiny-b'} and m['steps'] == 2
    mf = json.load(open(os.path.join(dm, 'merged_from.json'), encoding='utf-8'))
    assert set(mf['parts']) == {'tiny-a', 'tiny-b'} and mf['seed'] == 4
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    # never overwrite the merged summary
    r = subprocess.run([sys.executable, '-B', TRAIN, '--merge-b4', dq, dl, '--out', dm], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and 'never overwrite' in (r.stderr + r.stdout)
    # mismatched seed, same model twice, a multi-model part: non-zero exit, nothing written
    bad = os.path.join(data_dir, 'B4l-hetero-s5-bad')
    os.makedirs(bad, exist_ok=True)
    s5 = dict(parts['tiny-b'], seed=5)
    with open(os.path.join(bad, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(s5, fh)
    for args_, msg in (([dq, bad], 'seed differs'), ([dq, dq], 'two different single models'), ([os.path.join(data_dir, 'B4-hetero-s3'), dl], 'not a single-model B4 run')):
        out = os.path.join(data_dir, 'B4-merge-bad')
        r = subprocess.run([sys.executable, '-B', TRAIN, '--merge-b4'] + args_ + ['--out', out], capture_output=True, text=True, env=env)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (msg, r.stderr[-500:])
        assert not os.path.exists(os.path.join(out, 'summary.json'))


def test_l1_train_soft_deadline_in_readouts_then_eval_only(data_dir):
    """a Deadline during the readouts keeps summary.json (readouts_incomplete), exit 0; --eval-only recomputes them into readouts.json;
    a Deadline during training means INCOMPLETE.json and no summary.json"""
    out = os.path.join(data_dir, 'S1-hetero-s9')
    common = ['--pair', 'hetero', '--seed', '9', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'), '--fresh',
              os.path.join(data_dir, 'fresh.jsonl'), '--batch', '2', '--eval-batch', '2', '--tiny', '--k', '4', '--d-e', '8', '--n-heads', '2', '--max-new-tokens', '2',
              '--warmup', '1']
    _run([TRAIN, '--arm', 'S1', '--out', out, '--steps', '2', '--debug-deadline-at', 'readouts'] + common)
    s = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
    _validate(s, 'S1', 'hetero', 9)
    assert s['readouts_incomplete'] and s['readouts_incomplete'].startswith('absence p=0.3') and list(s['absence']) == ['0.0'] and s['c_assert'] is True
    assert s['complete'] is True and s['trainable_sha256'] == _sha(os.path.join(out, 'trainable.pt')) and not os.path.exists(os.path.join(out, 'INCOMPLETE.json'))
    before = _sha(os.path.join(out, 'summary.json'))
    _run([TRAIN, '--eval-only', out])
    ro = json.load(open(os.path.join(out, 'readouts.json'), encoding='utf-8'))
    assert ro['eval_only'] is True and sorted(ro['absence']) == ['0.3', '0.6', '1.0'] and ro['readouts_incomplete'] is None and ro['arm'] == 'S1' and ro['seed'] == 9
    assert ro['trainable_sha256'] == _sha(os.path.join(out, 'trainable.pt')) and all('em' in v and v['n_messages_dropped'] >= 0 for v in ro['absence'].values())
    assert ro['absence']['1.0']['n_messages_dropped'] == 6 and ro['absence']['1.0']['absence_p1_detail']['equal'] is True
    assert _sha(os.path.join(out, 'summary.json')) == before, 'eval-only must never touch summary.json'
    assert os.path.exists(os.path.join(out, 'log_readouts.txt'))
    # a deadline in the past during training -> INCOMPLETE.json, exit 3, no summary.json
    out2 = os.path.join(data_dir, 'S1-hetero-s9-late')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-B', TRAIN, '--arm', 'S1', '--out', out2, '--steps', '2', '--deadline-epoch', '1'] + common, capture_output=True, text=True, env=env)
    assert r.returncode == 3 and os.path.exists(os.path.join(out2, 'INCOMPLETE.json')) and not os.path.exists(os.path.join(out2, 'summary.json'))
    # B0 has no readouts: eval-only only writes the note
    outb = os.path.join(data_dir, 'B0-same-s3')
    _run([TRAIN, '--eval-only', outb])
    rb = json.load(open(os.path.join(outb, 'readouts.json'), encoding='utf-8'))
    assert rb['absence'] is None and 'note' in rb


# ---------------------------------------------------------------- 6. the chain (static)
def test_chain_static():
    bash = shutil.which('bash')
    if bash:
        r = subprocess.run([bash, '-n', CHAIN], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    s = open(CHAIN, encoding='utf-8').read()
    code = '\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('#'))
    assert 'pkill' not in code and 'killall' not in code and 'kill -9' not in code
    for v in ('MODEL_SHA', 'TRAIN_SHA', 'TEXT_SHA', 'FUSION_SHA', 'DATA_SHA', 'AGG_SHA', 'TRAINJ_SHA', 'VALJ_SHA', 'FRESHJ_SHA'):
        assert re.search(rf'^{v}=FILL_{v}\b', s, re.M), f'{v} must be a FILL_ placeholder for the coordinator'
    assert 'if grep -q -e "FILL_"' in s and '[ "$PREREG" = "D8-82" ] && [ -n "$APPROVAL" ]' in s
    used = set(re.findall(r'scripts/(l1_[a-z_]+\.py)', code))
    checked = set(re.findall(r'\(sha scripts/(l1_[a-z_]+\.py)\)', code))
    assert used <= checked and used >= {'l1_train.py', 'l1_text.py', 'l1_aggregate.py'}, f'scripts run without a sha check: {used - checked}'
    assert checked >= {'l1_model.py', 'l1_train.py', 'l1_text.py', 'l1_fusion.py', 'l1_data.py', 'l1_aggregate.py'}
    assert 'START+7150' in s and '"inclusive_seconds": 7200' in s and 'mkdir $RUNS.lock' in code and 'trap' in code and 'timeout -k 30' in code
    assert 'HF_HUB_CACHE=/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache/hub' in code and 'HF_HUB_OFFLINE=1' in code
    assert 'P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python' in code and 'cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021' in code
    assert 'DATA=$WORK/hotpot/splits_l1' in code and 'RUNS=$WORK/runs/l1' in code and 'STAGE0_FAIL' in code and 'STAGE0_PASS' in code
    assert '--stage0' in code and 'l1_aggregate.py --runs $RUNS' in code and 'nvidia-smi --query-compute-apps=pid' in code
    assert 'date -u' in code and '--deadline-epoch $DL' in code
    # the fixed order of stage 1 and the stage-0 gate before it
    i0 = code.index('stage0_jobs()'); i1 = code.index('stage1_jobs()')
    assert i0 < i1
    body = code[i1:code.index('if [ ! -f $RUNS/STAGE0_PASS ] && [ ! -f $RUNS/STAGE0_FAIL ]; then')]
    order = ['S1-hetero-s$s $EST_LATENT', 'tsend hetero $s; trecv hetero $s', 'U-hetero-s$s $EST_LATENT', 'S1L1-hetero-s$s $EST_LATENT', 't0 hetero $s',
             'B0-same-s$s $EST_BASE', 'B4-same-s$s $EST_B4', 'tsend same $s; trecv same $s', 'S1-same-s$s $EST_LATENT', 'tsweep hetero $s', 'tsweep same $s']
    pos = [body.index(o) for o in order]
    assert pos == sorted(pos), 'stage-1 job order: S1 first, Tsend / T per seed, ..., Tsweep at the end'
    assert body.index('tsweep hetero $s') > body.index('S1-same-s$s $EST_LATENT'), 'Tsweep after the whole job list'
    stage0 = code[code.index('stage0_jobs(){'):code.index('stage0_complete(){')]
    assert 'job B0-hetero-s$s $EST_BASE summary.json scripts/l1_train.py --arm B0 --pair hetero --seed $s $COMMON $SHARD' in stage0
    assert 'job B4l-hetero-s$s $EST_B4L summary.json scripts/l1_train.py --arm B4 --pair hetero --model $LLAMA --seed $s $COMMON $FULL' in stage0
    assert 'job B4q-hetero-s$s $EST_B4Q summary.json scripts/l1_train.py --arm B4 --pair hetero --model $QWEN --seed $s $COMMON $FULL' in stage0
    assert stage0.index('B0-hetero-s$s') < stage0.index('B4l-hetero-s$s') < stage0.index('B4q-hetero-s$s') < stage0.index('merge_b4'), 'B0 s, B4l s per seed, then B4q, then the merge'
    assert 'B4-hetero-s$s $EST_B4' not in stage0, 'the hetero B4 is never one job'
    merge = code[code.index('merge_b4(){'):code.index('stage0_jobs(){')]
    assert '[ -f $RUNS/B4q-hetero-s$s/summary.json ] && [ -f $RUNS/B4l-hetero-s$s/summary.json ] && [ ! -f $RUNS/B4-hetero-s$s/summary.json ]' in merge
    ml = [l for l in merge.splitlines() if '--merge-b4' in l]
    assert len(ml) == 1 and 'CUDA_VISIBLE_DEVICES= timeout -k 30 600 nice -n 10 $P -B scripts/l1_train.py --merge-b4 $RUNS/B4q-hetero-s$s $RUNS/B4l-hetero-s$s --out $RUNS/B4-hetero-s$s' in ml[0]
    assert 'for r in B0-hetero B4-hetero; do [ -f $RUNS/$r-s$s/summary.json ]' in code, 'stage 0 is complete only with the merged B4'
    assert 'job B4-same-s$s $EST_B4 summary.json scripts/l1_train.py --arm B4 --pair same --seed $s $COMMON $FULL' in code
    # the T phases: Tsend (sender), T (receiver, --sender-from, the aggregator's dir), Tsweep (sweep256, both froms), T0 with --budgets 16,64
    assert re.search(r'^tsend\(\)\{ job Tsend-\$1-s\$2 \$EST_TSEND summary_sender\.json scripts/l1_text\.py --arm T --phase sender --pair \$1 --seed \$2 \$COMMON \$TSEND; \}', s, re.M)
    assert 'job T-$1-s$2 $EST_TEXT summary.json scripts/l1_text.py --arm T --phase receiver --pair $1 --seed $2 --sender-from $RUNS/Tsend-$1-s$2 $COMMON $TRECV' in code
    assert '[ -f $RUNS/Tsend-$1-s$2/summary_sender.json ]; then job T-$1-s$2' in code
    assert 'job Tsweep-$1-s$2 $EST_SWEEP summary_sweep256.json scripts/l1_text.py --arm T --phase sweep256 --pair $1 --seed $2 --sender-from $RUNS/Tsend-$1-s$2 --receiver-from $RUNS/T-$1-s$2 $COMMON $TSWEEP' in code
    assert '[ -f $RUNS/Tsend-$1-s$2/summary_sender.json ] && [ -f $RUNS/T-$1-s$2/summary.json ]; then job Tsweep-$1-s$2' in code
    assert 'job T0-$1-s$2 $EST_T0 summary.json scripts/l1_text.py --arm T0 --pair $1 --seed $2 --receiver-from $RUNS/T-$1-s$2 --budgets 16,64 $COMMON $TRECV' in code
    # the done marker is an argument of job(); context caps and checkpointing are chain variables
    assert 'local name=$1 est=$2 marker=$3; shift 3' in code and 'if [ -f $dir/$marker ]; then' in code
    assert re.search(r'^CTX_SHARD=900\b', s, re.M) and re.search(r'^CTX_FULL=2048\b', s, re.M) and re.search(r'^NOCKPT_SHARD=""', s, re.M)
    assert 'SHARD="--max-ctx-tokens $CTX_SHARD $NOCKPT_SHARD"' in code and 'FULL="--max-ctx-tokens $CTX_FULL"' in code
    # review M2: the text prompts' instruction line / message are not charged to the documents' CTX_SHARD budget
    assert 'TSEND="--max-ctx-tokens $((CTX_SHARD+16)) $NOCKPT_SHARD"' in code
    assert 'TRECV="--max-ctx-tokens $((CTX_SHARD+72)) $NOCKPT_SHARD"' in code
    assert 'TSWEEP="--max-ctx-tokens $((CTX_SHARD+264)) $NOCKPT_SHARD"' in code
    assert re.search(r'^COMMON="--train \$DATA/train\.jsonl --val \$DATA/val\.jsonl --fresh \$DATA/fresh\.jsonl --hf-cache \$HF_HUB_CACHE --keep-preds"', code, re.M)
    for l in code.splitlines():
        if 'scripts/l1_train.py --arm B4' in l:
            assert '$FULL' in l and '$SHARD' not in l and 'NOCKPT' not in l, l
        elif 'scripts/l1_train.py --arm' in l:
            assert '$SHARD' in l and '$FULL' not in l, l
        elif 'scripts/l1_text.py --arm T --phase sender' in l:
            assert '$TSEND' in l and '$SHARD' not in l and '$FULL' not in l, l
        elif 'scripts/l1_text.py --arm T --phase sweep256' in l:
            assert '$TSWEEP' in l and '$SHARD' not in l and '$FULL' not in l, l
        elif 'scripts/l1_text.py --arm' in l:
            assert '$TRECV' in l and '$SHARD' not in l and '$FULL' not in l, l
    # review S4: the fit check reads the clock after the GPU-idle wait
    assert code.index('GPU busy before start; waiting') < code.index('rem=$((DL-now))')
    assert 'LEASE=""; EST_BASE=62; EST_B4L=55; EST_B4Q=100; EST_B4=100; EST_LATENT=115; EST_TSEND=90; EST_TEXT=85; EST_T0=15; EST_SWEEP=35' in code
    assert all(f'--est-{k}) EST_{k.upper()}="$2"; shift 2;;' in code for k in ('base', 'b4l', 'b4q', 'b4', 'latent', 'tsend', 'text', 't0', 'sweep'))
    agg_lines = [l for l in code.splitlines() if '$P -B scripts/l1_aggregate.py' in l]
    assert agg_lines and all('--runs $RUNS' in l and '--out' not in l and 'CUDA_VISIBLE_DEVICES= ' in l and 'nice -n 10' in l and 'timeout -k 30' in l for l in agg_lines), agg_lines
    assert '"P0: PASS")' in code and '"P0: FAIL")' in code, 'the chain reads the P0: line printed by l1_aggregate.py --stage0'
    assert 'elif [ "$LEASE" -eq 1 ]; then' in code and '$marker exists -> skipped (done)' in code and 'estimate $est min > remaining' in code
    for d in re.findall(r'^(?:RUNS|DATA|WORK|TMPDIR|LOG)=(\S+)', s, re.M):
        assert d.startswith('/mnt/datasets/') or d.startswith('$WORK') or d.startswith('runs/'), d
    assert 'bigboy' in s.lower()
