"""Unit tests for l1b_model.py / l1b_train.py / l1b_aggregate.py (batch L1b, D8-85): CPU, tiny random models from test_l1_model.py.
What is tested: (b) the reader scale equals the embedding RMS and the soft tokens come out at that scale; (a) the answer CE of the two-segment
loss is IDENTICAL to l1_model.Agent.loss (the aux segment sits after the answer: causality), the aux CE alone reaches both senders' adapters
and readers through the fusion, Pair.step combines the two with beta; the arms map to their L1 base arms (fixed lam, U dropout); the L1
evaluation path runs unchanged on a PairB (GBP messages, C assertion, absence p = 1 falls back to own z) and the decodability readout works
at p = 0 and 1; the CLI (--tiny) writes summary.json with the L1 schema keys plus the L1b blocks, and the soft deadline leaves the summary;
the aggregator's verdict table (D8-85 section 5) on synthetic summaries."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import pytest  # noqa: E402
import torch  # noqa: E402
import l1_model as M  # noqa: E402
import l1b_aggregate as AG  # noqa: E402
import l1b_model as MB  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = TL.K, TL.DE
TRAIN_B = os.path.join(HERE, 'l1b_train.py')
AGG_B = os.path.join(HERE, 'l1b_aggregate.py')


def _pair(arm, max_ctx=None, beta=1.0, aux_max=64):
    base = MB.BASE_ARM[arm]
    fixed = base in ('U', 'S1L1')
    a = MB.AgentB.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2)
    b = MB.AgentB.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2)
    return MB.PairB([a, b], 'hetero', arm, max_ctx=max_ctx, beta=beta, aux_max=aux_max)


# ---------------------------------------------------------------- (b) the reader scale
def test_reader_scale_is_the_embedding_rms_and_soft_tokens_land_at_it():
    pair = _pair('S1b')
    pair.eval()
    with torch.no_grad():
        z, lam = pair.encode_both(TL.EXAMPLES)
        softs, _ = pair.soft_from_train(z, lam, train=False)
    for i, ag in enumerate(pair.agents):
        w = ag.embed.weight.float()
        expected = float(w.norm(dim=-1).mean() / w.shape[1] ** 0.5)
        assert isinstance(ag.reader, MB.ReaderB) and abs(float(ag.reader.scale) - expected) < 1e-6 and abs(ag.reader_scale - expected) < 1e-6
        rms = float(softs[i].float().pow(2).mean().sqrt())
        assert abs(rms / expected - 1.0) < 0.05, (rms, expected)                 # LayerNorm output has RMS ~1 (gamma 1, beta 0 at init) -> times the scale
        assert softs[i].shape == (len(TL.EXAMPLES), K, ag.d_model)
    # the pre-registered defaults
    assert MB.AUX_BETA == 1.0 and MB.AUX_MAX_TOKENS == 64 and MB.ABSENCE_PS == (0.0, 1.0) and MB.DECODE_N == 256 and MB.DECODE_MAX_NEW == 64
    assert MB.BASE_ARM == {'S1b': 'S1', 'Ub': 'U', 'S1L1b': 'S1L1'} and MB.BATCH == 'L1b' and MB.PREREG == 'D8-85'


# ---------------------------------------------------------------- (a) the two-segment loss
def test_answer_ce_is_identical_to_l1_loss_and_aux_segment_follows_the_answer():
    pair = _pair('S1b')
    pair.eval()                                                                  # no LoRA dropout: the two forward passes must agree exactly
    with torch.no_grad():
        z, lam = pair.encode_both(TL.EXAMPLES)
        softs, _ = pair.soft_from_train(z, lam, train=False)
    for i in range(2):
        ag = pair.agent(i)
        pcs = pair.pieces(TL.EXAMPLES, i, 'latent')
        aux = pair.aux_ids(TL.EXAMPLES, i)
        with torch.no_grad():
            l_ans, l_aux, info = ag.loss_aux(pcs, softs[i], aux)
            l_plain, info_plain = M.Agent.loss(ag, pcs, softs[i])
        assert torch.allclose(l_ans, l_plain, atol=1e-5), (float(l_ans), float(l_plain))
        assert torch.allclose(info['per_example'], info_plain['per_example'], atol=1e-5)
        assert torch.allclose(info['first_token_entropy'], info_plain['first_token_entropy'], atol=1e-5)
        assert info['n_tokens'] == info_plain['n_tokens'] and info['n_aux_tokens'] == sum(len(x) for x in aux) and float(l_aux) > 0
        for ex, x in zip(TL.EXAMPLES, aux):                                      # prompt + partner (1 - i) facts + EOS, decoded back
            assert x[-1] == ag.eos and x[:len(ag.ids(MB.AUX_PROMPT))] == ag.ids(MB.AUX_PROMPT)
            facts = ag.tok.decode(x[len(ag.ids(MB.AUX_PROMPT)):-1]).strip()
            assert facts == M.DATA.text_sender_target(ex, 1 - i).strip().lower()


def test_aux_ids_are_cut_to_aux_max_and_cached():
    pair = _pair('S1b', aux_max=2)
    for i in range(2):
        ag = pair.agent(i)
        for ex, x in zip(TL.EXAMPLES, pair.aux_ids(TL.EXAMPLES, i)):
            assert len(x) == len(ag.ids(MB.AUX_PROMPT)) + 2 + 1 and x[-1] == ag.eos
        assert pair.aux_ids(TL.EXAMPLES, i) == pair.aux_ids(TL.EXAMPLES, i)       # the cache returns the same ids
    assert len(pair._aux_cache) == 2 * len(TL.EXAMPLES)


def test_aux_loss_alone_reaches_both_senders_through_the_fusion():
    pair = _pair('S1b')
    pair.train()
    z, lam = pair.encode_both(TL.EXAMPLES)
    softs, _ = pair.soft_from_train(z, lam, train=True)
    total = 0
    for i in range(2):
        _, l_aux, _ = pair.agent(i).loss_aux(pair.pieces(TL.EXAMPLES, i, 'latent'), softs[i], pair.aux_ids(TL.EXAMPLES, i))
        total = total + l_aux
    total.backward()
    for ag in pair.agents:
        for mod, what in ((ag.adapter, 'adapter'), (ag.reader, 'reader')):
            g = [p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
            assert g and any(float(x.abs().sum()) > 0 for x in g), f'{ag.name} {what} gets no gradient from the aux loss'
        lora = [p.grad for n, p in ag.named_parameters() if 'lora_' in n and p.grad is not None]
        assert lora and any(float(x.abs().sum()) > 0 for x in lora)
    assert pair.w_raw.grad is not None and float(pair.w_raw.grad.abs()) > 0


@pytest.mark.parametrize('arm', ['S1b', 'Ub', 'S1L1b'])
def test_pair_step_combines_the_two_losses_with_beta_and_maps_to_the_base_arm(arm):
    pair = _pair(arm, beta=0.5)
    pair.train()
    loss, info = pair.step(TL.EXAMPLES, train=True)
    assert torch.isfinite(loss) and abs(float(loss) - (info['loss_ans'] + 0.5 * info['loss_aux'])) < 1e-5
    assert len(info['loss_agent']) == 2 == len(info['loss_aux_agent']) and abs(sum(info['loss_agent']) / 2 - info['loss_ans']) < 1e-5
    assert pair.arm == MB.BASE_ARM[arm] and pair.arm_b == arm and pair.aux_hist == [(info['loss_ans'], info['loss_aux'])]
    assert all(ag.adapter.fixed_lam == (arm in ('Ub', 'S1L1b')) for ag in pair.agents)
    assert (pair.w is not None) == (arm != 'Ub')
    if arm == 'Ub':
        assert info['drop'] is not None and info['drop'].shape == (2, len(TL.EXAMPLES))
    else:
        assert info['drop'] is None
    s = pair.aux_summary(window=1)
    assert s['steps'] == 1 and s['loss_aux_last'] == info['loss_aux'] and s['history'][-1]['step'] == 1


def test_build_pair_tiny_and_rejects_l1_arm_names():
    tiny = M.tiny_models(TL.EXAMPLES, 5)
    pair = MB.build_pair('Ub', 'hetero', torch.device('cpu'), tiny=tiny, k=K, d_e=DE, n_heads=2)
    assert isinstance(pair, MB.PairB) and [a.name for a in pair.agents] == list(M.TINY_NAMES) and pair.arm == 'U'
    with pytest.raises(ValueError):
        MB.build_pair('S1', 'hetero', torch.device('cpu'), tiny=tiny, k=K, d_e=DE, n_heads=2)
    with pytest.raises(ValueError):
        MB.PairB(pair.agents, 'hetero', 'S1')


# ---------------------------------------------------------------- the L1 evaluation path on a PairB + the decodability readout
@pytest.mark.parametrize('arm', ['S1b', 'Ub'])
def test_eval_latent_runs_unchanged_and_decodability_at_p0_and_p1(arm):
    pair = _pair(arm)
    r0 = M.eval_latent(pair, TL.EXAMPLES, drop_p=0.0, eval_batch=2, seed=3, max_new_tokens=2, c_check=True)
    assert 0 <= r0['em'] <= 100 and r0['n_records'] == 2 * len(TL.EXAMPLES) and set(r0['em_model']) == {'tiny-a', 'tiny-b'}
    if arm == 'S1b':
        assert r0['c_assert'] is True and r0['gbp_rounds'] == 1
    r1 = M.eval_latent(pair, TL.EXAMPLES, drop_p=1.0, eval_batch=2, seed=3, max_new_tokens=2, c_check=False)
    assert r1['n_messages_dropped'] == r1['n_messages'] == 2 * len(TL.EXAMPLES) and r1['absence_p1_detail']['equal'] is True
    for p in (0.0, 1.0):
        d = MB.eval_decodability(pair, TL.EXAMPLES, drop_p=p, eval_batch=2, seed=3, max_new_tokens=3)
        assert d['drop_p'] == p and d['n'] == len(TL.EXAMPLES) and d['n_records'] == 2 * len(TL.EXAMPLES)
        assert 0.0 <= d['f1'] <= 1.0 and len(d['f1_agent']) == 2 and set(d['f1_model']) == {'tiny-a', 'tiny-b'} and 0 <= d['n_tokens_mean'] <= 3
        assert len(d['examples']) == min(8, 2 * len(TL.EXAMPLES)) and all(0.0 <= e['f1'] <= 1.0 for e in d['examples'])


# ---------------------------------------------------------------- the CLI end to end (--tiny)
@pytest.fixture(scope='module')
def data_dir():
    d = tempfile.mkdtemp(prefix='l1b_test_')
    TL._write_jsonl(os.path.join(d, 'train.jsonl'), TL.EXAMPLES * 2)
    TL._write_jsonl(os.path.join(d, 'val.jsonl'), TL.EXAMPLES)
    TL._write_jsonl(os.path.join(d, 'fresh.jsonl'), TL.EXAMPLES[:2])
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _cli(data_dir, arm, out, extra=()):
    return TL._run([TRAIN_B, '--arm', arm, '--pair', 'hetero', '--seed', '7', '--train', os.path.join(data_dir, 'train.jsonl'), '--val', os.path.join(data_dir, 'val.jsonl'),
                    '--fresh', os.path.join(data_dir, 'fresh.jsonl'), '--out', out, '--steps', '3', '--batch', '2', '--eval-batch', '2', '--tiny', '--k', '4', '--d-e', '8',
                    '--n-heads', '2', '--max-new-tokens', '2', '--decode-n', '2', '--decode-max-new', '3', '--warmup', '1', '--log-every', '1', '--keep-preds'] + list(extra))


def test_l1b_train_tiny_writes_summary_with_l1_schema_and_l1b_blocks(data_dir):
    out = os.path.join(data_dir, 'S1b-hetero-s7')
    _cli(data_dir, 'S1b', out)
    s = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
    TL._validate(s, 'S1b', 'hetero', 7)
    assert s['arm'] == 'S1b' and s['base_arm'] == 'S1' and s['batch_name'] == 'L1b' and s['prereg'] == 'D8-85' and s['c_assert'] is True and s['complete'] is True
    assert set(s['absence']) == {'0.0', '1.0'} and set(s['decodability']) == {'0.0', '1.0'} and s['readouts_incomplete'] is None
    assert s['aux']['beta'] == 1.0 and s['aux']['max_tokens'] == 64 and s['aux']['prompt'] == MB.AUX_PROMPT and set(s['aux']['reader_scale']) == {'tiny-a', 'tiny-b'}
    assert s['aux']['train']['steps'] == 3 and s['train']['aux']['loss_aux_last'] > 0 and s['train']['steps_done'] == 3
    assert set(s['scripts_sha']) == set(MB.SCRIPTS) and all(v for v in s['scripts_sha'].values())
    assert s['decodability']['0.0']['n'] == 2 and 0.0 <= s['decodability']['1.0']['f1'] <= 1.0 and s['w'] is not None
    assert s['config']['absence_ps'] == '0.0,1.0' and s['config']['k'] == 4
    ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
    assert ck['arm'] == 'S1b' and ck['base_arm'] == 'S1' and set(ck['reader_scale']) == {'tiny-a', 'tiny-b'}
    assert any('reader.proj' in n for n in ck['state']) and any('adapter.' in n for n in ck['state']) and any('lora_' in n for n in ck['state']) and 'w_raw' in ck['state']
    assert not any('reader.scale' in n for n in ck['state'])                      # the scale is a buffer, never trained
    for f in ('preds_val.json', 'preds_fresh.json', 'preds_val_p1.0.json', 'log.txt'):
        assert os.path.exists(os.path.join(out, f)), f
    r = subprocess.run([sys.executable, '-B', TRAIN_B, '--arm', 'S1b', '--pair', 'hetero', '--seed', '7', '--train', os.path.join(data_dir, 'train.jsonl'), '--val',
                        os.path.join(data_dir, 'val.jsonl'), '--fresh', os.path.join(data_dir, 'fresh.jsonl'), '--out', out, '--tiny'], capture_output=True, text=True)
    assert r.returncode != 0 and 'never overwrite' in r.stdout + r.stderr


def test_l1b_train_tiny_ub_and_soft_deadline_keeps_the_summary(data_dir):
    out = os.path.join(data_dir, 'Ub-hetero-s7')
    _cli(data_dir, 'Ub', out, extra=('--debug-deadline-at', 'readouts'))
    s = json.load(open(os.path.join(out, 'summary.json'), encoding='utf-8'))
    assert s['arm'] == 'Ub' and s['base_arm'] == 'U' and 'c_assert' not in s and s['w'] is None if 'w' in s else True
    assert s['readouts_incomplete'] and s['readouts_incomplete'].startswith('absence p=1.0') and list(s['absence']) == ['0.0'] and s['decodability'] == {}
    assert s['complete'] is True and 0 <= s['val']['em'] <= 100 and s['val']['n_messages_dropped'] == 0


# ---------------------------------------------------------------- the aggregator (D8-85 section 5)
def _summary(d, arm, seed, val, fresh, c_assert=True):
    p = os.path.join(d, f'{arm}-hetero-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair='hetero', seed=seed, val=dict(em=val, f1=val + 10, em_agent=[val, val], em_model={}), fresh=dict(em=fresh, f1=fresh + 10, em_agent=[fresh, fresh], em_model={}),
             scripts_sha={'x.py': 'abc'})
    if arm == 'S1b' and c_assert is not None:
        s['c_assert'] = c_assert
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w', encoding='utf-8'))


def test_aggregate_verdict_table(tmp_path):
    l1, l1b = str(tmp_path / 'l1'), str(tmp_path / 'l1b')
    os.makedirs(l1); os.makedirs(l1b)
    s1, s2 = AG.SEEDS
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'INCOMPLETE' and agg['g0_lines'] == [f'G0(s{s1}): INCOMPLETE', f'G0(s{s2}): INCOMPLETE']
    _summary(l1, 'B0', s1, 42.0, 41.0); _summary(l1, 'B0', s2, 42.5, 41.5)
    _summary(l1b, 'S1b', s1, 44.0, 43.0)                                        # +2.0 < 3.0 -> the channel gate fails on seed 1
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'CHANNEL-NO' and agg['g0_lines'][0] == f'G0(s{s1}): FAIL' and agg['criteria']['G0']['margin'][str(s1)] == 2.0
    assert AG.aggregate(l1, l1b, gate=s1)['g0_lines'] == [f'G0(s{s1}): FAIL'] and AG.aggregate(l1, l1b, gate=s1)['verdict'] is None
    shutil.rmtree(os.path.join(l1b, f'S1b-hetero-s{s1}'))
    _summary(l1b, 'S1b', s1, 50.0, 49.0)                                        # +8.0 -> G0 passes on seed 1; seed 2 missing -> INCOMPLETE
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'INCOMPLETE' and agg['g0_lines'][0] == f'G0(s{s1}): PASS'
    _summary(l1b, 'S1b', s2, 43.0, 42.0)                                        # +0.5 -> fails on seed 2
    assert AG.aggregate(l1, l1b)['verdict'] == 'CHANNEL-NO'
    shutil.rmtree(os.path.join(l1b, f'S1b-hetero-s{s2}'))
    _summary(l1b, 'S1b', s2, 51.0, 50.0)                                        # both seeds pass G0; T and Ub missing -> INCOMPLETE
    assert AG.aggregate(l1, l1b)['verdict'] == 'INCOMPLETE'
    _summary(l1, 'T', s1, 47.0, 46.0); _summary(l1, 'T', s2, 48.0, 47.0)
    _summary(l1b, 'Ub', s1, 50.0, 49.0); _summary(l1b, 'Ub', s2, 50.5, 49.5)      # P2b: +0.0 / +0.5 < 1.0 -> NO
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'NO' and agg['criteria']['P2b']['pass'] is False and agg['criteria']['P1b']['pass'] is True
    for s, v, f in ((s1, 48.0, 47.0), (s2, 49.0, 48.0)):                           # P2b passes (+2.0 / +2.0)
        shutil.rmtree(os.path.join(l1b, f'Ub-hetero-s{s}')); _summary(l1b, 'Ub', s, v, f)
    shutil.rmtree(os.path.join(l1, f'T-hetero-s{s1}')); _summary(l1, 'T', s1, 49.0, 48.0)   # P1b: +1.0 on seed 1 -> AMBIGUOUS
    assert AG.aggregate(l1, l1b)['verdict'] == 'AMBIGUOUS'
    shutil.rmtree(os.path.join(l1, f'T-hetero-s{s1}')); _summary(l1, 'T', s1, 47.0, 48.5)   # VAL +3.0, fresh +0.5 -> drop 2.5 -> degrade
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'AMBIGUOUS' and agg['criteria']['degrade']['triggered'] is True
    shutil.rmtree(os.path.join(l1, f'T-hetero-s{s1}')); _summary(l1, 'T', s1, 47.0, 46.5)
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'LEVER-pending-P3b' and agg['reasons'] == [] and all(c['pass'] for k, c in agg['criteria'].items() if k != 'degrade')
    shutil.rmtree(os.path.join(l1b, f'S1b-hetero-s{s2}')); _summary(l1b, 'S1b', s2, 51.0, 50.0, c_assert=False)
    agg = AG.aggregate(l1, l1b)
    assert agg['verdict'] == 'INCOMPLETE' and any('c_assert false' in r for r in agg['reasons'])
    md = AG.to_markdown(agg)
    assert 'VERDICT: INCOMPLETE' in md and 'G0 | S1b - B0' in md
    r = subprocess.run([sys.executable, '-B', AGG_B, '--l1', l1, '--l1b', l1b, '--gate', str(s1)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.splitlines()[0] == f'G0(s{s1}): PASS' and 'VERDICT' not in r.stdout
    assert os.path.exists(os.path.join(l1b, f'aggregate_l1b_gate{s1}.json')) and os.path.exists(os.path.join(l1b, f'aggregate_l1b_gate{s1}.md'))
    r = subprocess.run([sys.executable, '-B', AGG_B, '--l1', l1, '--l1b', l1b], capture_output=True, text=True)
    assert r.returncode == 0 and 'VERDICT: INCOMPLETE' in r.stdout and os.path.exists(os.path.join(l1b, 'aggregate_l1b.md'))
