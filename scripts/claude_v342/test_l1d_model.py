"""Unit tests for l1d_model.py / l1d_train.py / l1d_aggregate.py (batch L1d, D8-87): CPU, tiny random models from test_l1_model.py.
What is tested: the receiver's latent prompt wraps the K slots into the hint template while the B0 pieces are untouched; the targets are the
gold answer cut to K and filler-padded; pass 1 runs on the B0 prompt + the sender's own greedy answer WITH the LoRA (perturbing the LoRA changes z)
and the answers are cached per example id; the message loss reaches the adapter, the decoder and w (Ds) but never the LoRA; the pinned
evaluation path runs on a PairD with the hard and the soft rendering (C assertion, absence p = 1 fallback); the baselines none (= the plain B0
prompt) and gold_hint, the hint metrics and lam-by-bucket run; build_pair loads and freezes a receiver state; the CLI (--tiny) writes
summary.json with the L1 schema keys plus the L1d blocks, never overwrites, keeps the summary under the soft deadline; the aggregator's verdict
table (D8-87 section 5) on synthetic summaries, with rounded margins."""
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
import l1d_aggregate as AG  # noqa: E402
import l1d_model as MD  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_D = os.path.join(HERE, 'l1d_train.py')
AGG_D = os.path.join(HERE, 'l1d_aggregate.py')


def _pair(arm, max_ctx=None, n_blocks=2):
    fixed = MD.BASE_ARM[arm] == 'U'
    a = MD.AgentD.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=n_blocks)
    b = MD.AgentD.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=n_blocks)
    pd = MD.PairD([a, b], 'hetero', arm, max_ctx=max_ctx)
    for ag in pd.agents:
        ag.freeze_lora()
    for n, p in pd.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pd


# ---------------------------------------------------------------- (c) the prompt layout and (b) the targets
def test_latent_pieces_wrap_the_slots_into_the_hint_template_and_b0_pieces_are_plain():
    pair = _pair('Du')
    for i in range(2):
        ag = pair.agent(i)
        (pre, suf, ans), _ = M.make_pieces(ag, TL.EXAMPLES[0], i, 'latent', None, 'answer', None)
        (pre2, suf2, ans2) = pair.pieces(TL.EXAMPLES[:1], i, 'latent')[0]
        assert pre2 == list(pre) + ag.hint_pre and suf2 == ag.hint_post + list(suf) and ans2 == ans
        assert ag.hint_pre and ag.hint_post
        (pb, sb, ab) = pair.pieces(TL.EXAMPLES[:1], i, 'B0', target=None)[0]
        (pb0, sb0, ab0), _ = M.make_pieces(ag, TL.EXAMPLES[0], i, 'B0', None, None, None)
        assert pb == pb0 and sb == [] and ab == []
    # the context cap leaves room for the template
    pc = _pair('Du', max_ctx=40)
    ag = pc.agent(0)
    (pre, suf, _) = pc.pieces(TL.EXAMPLES[:1], 0, 'latent', target=None)[0]
    assert len(pre) + len(suf) <= 40


def test_targets_are_the_gold_answer_cut_and_filler_padded():
    pair = _pair('Du')
    ex = TL.EXAMPLES[0]
    for i in range(2):
        ag = pair.agent(i)
        ids, n = ag.target_ids(ex, 1 - i)
        full = ag.ids(' ' + ex['answer'])
        assert len(ids) == K and n == min(K, len(full)) and ids[:n] == [int(t) for t in full[:K]] and all(t == ag.filler for t in ids[n:])
        ids_b, _ = ag.target_ids(ex, i)                                       # the partner index is irrelevant
        assert ids_b == ids
    tgt, lens = pair.targets(TL.EXAMPLES[:2], 1)
    assert tgt.shape == (2, K) and len(lens) == 2


# ---------------------------------------------------------------- (a) pass 1: the sender's own model (LoRA on) over prompt + own answer
def test_pass1_uses_the_lora_and_the_own_answer_and_caches_it():
    pair = _pair('Ds')
    batch = TL.EXAMPLES[:2]
    pair.eval()
    z1, _ = pair.encode_both(batch)
    assert all(pair.agent(i).n_generated == 2 for i in range(2))
    for i in range(2):
        for ex in batch:
            assert ex['id'] in pair.agent(i).answer_cache
    z1b, _ = pair.encode_both(batch)                                           # served from the cache: no new generation, same z
    assert all(pair.agent(i).n_generated == 2 for i in range(2)) and torch.allclose(z1, z1b)
    with torch.no_grad():
        for n, p in pair.named_parameters():
            if 'lora_B' in n:
                p.add_(0.5)
    for ag in pair.agents:
        ag.answer_cache.clear()
    z2, _ = pair.encode_both(batch)
    assert not torch.allclose(z1, z2, atol=1e-4), 'pass 1 must see the LoRA (the B0 model)'
    assert all(not p.requires_grad for n, p in pair.named_parameters() if 'lora_' in n)


def test_message_loss_reaches_only_the_channel():
    for arm in MD.ARMS:
        pair = _pair(arm)
        pair.train()
        drop = torch.zeros(2, 2, dtype=torch.bool) if arm == 'Du' else None     # no message dropped: every adapter gets a gradient
        loss, info = pair.step(TL.EXAMPLES[:2], train=True, drop=drop)
        assert torch.isfinite(loss)
        loss.backward()
        names = dict(pair.named_parameters())
        assert all(names[n].grad is None for n in names if 'lora_' in n)
        for i in range(2):
            ad = [p for n, p in names.items() if f'agents.{i}.adapter.' in n and p.grad is not None]
            rd = [p for n, p in names.items() if f'agents.{i}.reader.' in n and p.grad is not None]
            assert ad and any(float(p.grad.abs().sum()) > 0 for p in ad) and rd and any(float(p.grad.abs().sum()) > 0 for p in rd)
        if arm == 'Ds':
            assert names['w_raw'].grad is not None and info['drop'] is None
        else:
            assert info['drop'].shape == (2, 2) and not bool(info['drop'].any())


# ---------------------------------------------------------------- the pinned evaluation path, the renderings and the readouts
def test_eval_render_baselines_hints_and_lam():
    pair = _pair('Ds')
    v = MD.eval_render(pair, TL.EXAMPLES, hard=True, drop_p=0.0, c_check=True, eval_batch=2, seed=1, keep_preds=True)
    assert v['c_assert'] is True and pair.hard is False and v['n_records'] == 2 * len(TL.EXAMPLES)
    s = MD.eval_render(pair, TL.EXAMPLES, hard=False, drop_p=0.0, c_check=False, eval_batch=2, seed=1)
    assert 'em' in s and all(ag.reader.default_hard is False for ag in pair.agents)
    p1 = MD.eval_render(pair, TL.EXAMPLES, hard=True, drop_p=1.0, c_check=False, eval_batch=2, seed=1)
    assert p1['absence_p1_detail']['equal'] is True and p1['n_messages_dropped'] == p1['n_messages']
    lb = MD.lam_by_bucket(v['records'])
    assert set(lb) == {'0', '1'} and all('mean_lam' in d for rels in lb.values() for d in rels.values())
    n0 = MD.eval_baseline(pair, TL.EXAMPLES, 'none', eval_batch=2)
    g = MD.eval_baseline(pair, TL.EXAMPLES, 'gold_hint', eval_batch=2)
    assert n0['kind'] == 'none' and g['kind'] == 'gold_hint' and n0['n_records'] == g['n_records'] == 2 * len(TL.EXAMPLES)
    with pytest.raises(ValueError):
        MD.eval_baseline(pair, TL.EXAMPLES, 'gold32')
    h = MD.hint_metrics(pair, TL.EXAMPLES, eval_batch=2)
    assert set(h) == {'n', 'hint_by_receiver_bucket', 'sender_answer_by_own_bucket', 'examples'} and h['n'] == len(TL.EXAMPLES)
    for a in ('0', '1'):
        for rel, d in h['hint_by_receiver_bucket'][a].items():
            assert set(d) == {'em_gold', 'f1_gold', 'agree_sender', 'n'} and 0.0 <= d['em_gold'] <= 100.0
        for rel, d in h['sender_answer_by_own_bucket'][a].items():
            assert set(d) == {'em_gold', 'n'}
    assert h['examples'] and set(h['examples'][0]) == {'id', 'receiver', 'rel', 'hint', 'sender_answer', 'gold'}
    # the Du arm: absence p = 1 falls back to D(0)
    pu = _pair('Du')
    r = MD.eval_render(pu, TL.EXAMPLES, hard=True, drop_p=1.0, c_check=False, eval_batch=2, seed=1)
    assert r['absence_p1_detail']['equal'] is True


def test_build_pair_loads_and_freezes_the_receiver_state():
    p0 = MD.build_pair('Du', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    assert p0.agent(0).k == K
    state = {n: torch.randn_like(p) for n, p in p0.named_parameters() if 'lora_' in n}
    p1 = MD.build_pair('Ds', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=state, k=K, d_e=DE, n_heads=2)
    names = dict(p1.named_parameters())
    for n, t in state.items():
        assert torch.equal(names[n].detach(), t) and not names[n].requires_grad
    p1.encode_both(TL.EXAMPLES[:2])
    assert all(not p.requires_grad for n, p in p1.named_parameters() if 'lora_' in n)
    assert all('lora_' not in n for n in M.trainable_state(p1)) and 'w_raw' in M.trainable_state(p1)
    pk = MD.build_pair('Du', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), d_e=DE, n_heads=2)
    assert pk.agent(0).k == MD.K_HINT
    with pytest.raises(ValueError):
        MD.build_pair('Du', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=dict(list(state.items())[:2]), k=K, d_e=DE, n_heads=2)
    with pytest.raises(ValueError):
        MD.build_pair('Cu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


# ---------------------------------------------------------------- the CLI on the tiny fixture
def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def _run(d, arm, out, extra=()):
    cmd = [sys.executable, '-B', TRAIN_D, '--tiny', '--arm', arm, '--pair', 'hetero', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
           '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
           '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2', *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_cli_tiny_run_writes_the_summary_and_never_overwrites():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Ds')
        r = _run(d, 'Ds', out, ['--keep-preds'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        for k in ('arm', 'base_arm', 'pair', 'seed', 'batch_name', 'prereg', 'val', 'fresh', 'absence', 'soft', 'baseline', 'hint', 'lam_by_bucket', 'complete', 'train',
                  'receiver', 'channel', 'timings', 'scripts_sha', 'c_assert', 'w'):
            assert k in s, k
        assert s['arm'] == 'Ds' and s['base_arm'] == 'S1' and s['batch_name'] == 'L1d' and s['prereg'] == 'D8-87' and s['complete'] is True and s['c_assert'] is True
        assert set(s['absence']) == {'0.0', '1.0'} and set(s['soft']) == {'val'} and set(s['hint']) == {'val', 'fresh'}
        assert set(s['baseline']['none']) == {'val', 'fresh'} and set(s['baseline']['gold_hint']) == {'val'}
        assert s['baseline']['none']['val']['kind'] == 'none' and 'em_by_owner' in s['baseline']['none']['val']
        assert s['channel']['rendering'] == 'hard' and s['channel']['receiver_training'] == 'none (frozen B0 checkpoint)' and s['channel']['k'] == K
        assert set(s['scripts_sha']) == set(MD.SCRIPTS) and s['readouts_incomplete'] is None and s['lam_by_bucket']
        assert all(os.path.isfile(os.path.join(out, f)) for f in ('trainable.pt', 'preds_val.json', 'preds_fresh.json', 'preds_val_none.json', 'log.txt'))
        import l1_aggregate as A1
        problems = A1.validate_summary(s, 'Ds', 'hetero', 5)
        assert not problems, problems
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert all('lora_' not in n for n in ck['state']) and ck['receiver'] is None
        r2 = _run(d, 'Ds', out)
        assert r2.returncode != 0 and 'never overwrite' in (r2.stdout + r2.stderr)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cli_soft_deadline_keeps_the_summary_and_marks_the_readouts():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Du')
        r = _run(d, 'Du', out, ['--debug-deadline-at', 'readouts'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['readouts_incomplete'] and s['absence'].get('1.0') is None and s['baseline']['none']['val']['em'] is not None and 'c_assert' not in s
        assert s['lam_by_bucket'] is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- the aggregator (D8-87 section 5)
def _summary(d, arm, seed, val, fresh, other=None, none=None, none_other=None, none_fresh=None, c_assert=True, complete=True, sha='abc', rendering='hard'):
    p = os.path.join(d, f'{arm}-hetero-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair='hetero', seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=complete, scripts_sha={'l1d_model.py': sha})
    if arm in ('Du', 'Ds'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=none_fresh if none_fresh is not None else 40.0)),
                             gold_hint=dict(val=dict(em=52.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['soft'] = dict(val=dict(em=val - 1.0))
        s['hint'] = dict(val=dict(hint_by_receiver_bucket={'0': {'other': dict(em_gold=50.0, n=5)}, '1': {'other': dict(em_gold=48.0, n=5)}}))
        s['channel'] = dict(rendering=rendering, receiver_training='none (frozen B0 checkpoint)')
    if arm == 'Ds':
        s['c_assert'] = c_assert
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def test_aggregator_verdicts():
    d = tempfile.mkdtemp()
    try:
        l1, l1d = os.path.join(d, 'l1'), os.path.join(d, 'l1d')
        os.makedirs(l1); os.makedirs(l1d)
        for s in AG.SEEDS:
            _summary(l1, 'B0', s, 42.0, 41.0)
            _summary(l1, 'T', s, 58.0, 56.0)
        a = AG.aggregate(l1, l1d)
        assert a['verdict'] == 'INCOMPLETE' and a['g1_lines'][0].endswith('INCOMPLETE')
        _summary(l1d, 'Du', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1d)
        assert a['verdict'] == 'CHANNEL-NO-3' and a['g1_lines'][0] == f'G1d(s{AG.SEEDS[0]}): FAIL'
        shutil.rmtree(os.path.join(l1d, f'Du-hetero-s{AG.SEEDS[0]}'))
        _summary(l1d, 'Du', AG.SEEDS[0], 43.3, 42.0, other=23.3, none=40.3, none_other=18.3)        # exactly 3.0 / 5.0 -> PASS
        a = AG.aggregate(l1, l1d)
        assert a['g1_lines'][0] == f'G1d(s{AG.SEEDS[0]}): PASS' and a['verdict'] == 'INCOMPLETE'
        g = AG.aggregate(l1, l1d, gate=AG.SEEDS[0])
        assert g['g1_lines'] == [f'G1d(s{AG.SEEDS[0]}): PASS'] and g['verdict'] is None
        _summary(l1d, 'Du', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        assert AG.aggregate(l1, l1d)['verdict'] == 'CHANNEL-YES / SHEAF-pending'
        _summary(l1d, 'Ds', AG.SEEDS[0], 44.3, 43.0)
        _summary(l1d, 'Ds', AG.SEEDS[1], 47.0, 46.0)
        assert AG.aggregate(l1, l1d)['verdict'] == 'CHANNEL-YES / SHEAF-ADDS'
        for s in AG.SEEDS:
            shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{s}'))
        _summary(l1d, 'Ds', AG.SEEDS[0], 42.3, 41.0)
        _summary(l1d, 'Ds', AG.SEEDS[1], 45.0, 44.0)
        assert AG.aggregate(l1, l1d)['verdict'] == 'CHANNEL-YES / SHEAF-HURTS'
        shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{AG.SEEDS[1]}'))
        _summary(l1d, 'Ds', AG.SEEDS[1], 46.5, 44.0)
        assert AG.aggregate(l1, l1d)['verdict'] == 'CHANNEL-YES / SHEAF-AMBIGUOUS'
        shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{AG.SEEDS[0]}'))
        _summary(l1d, 'Ds', AG.SEEDS[0], 43.8, 43.0)
        assert AG.aggregate(l1, l1d)['verdict'] == 'CHANNEL-YES / SHEAF-NEUTRAL'
        shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{AG.SEEDS[0]}'))
        _summary(l1d, 'Ds', AG.SEEDS[0], 43.8, 43.0, c_assert=False)
        a = AG.aggregate(l1, l1d)
        assert a['verdict'] == 'CHANNEL-YES / SHEAF-INCOMPLETE' and any('c_assert false' in r for r in a['reasons'])
        shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{AG.SEEDS[0]}'))
        _summary(l1d, 'Ds', AG.SEEDS[0], 43.8, 43.0, sha='zzz')
        assert AG.aggregate(l1, l1d)['verdict'] == 'INCOMPLETE'
        shutil.rmtree(os.path.join(l1d, f'Ds-hetero-s{AG.SEEDS[0]}'))
        _summary(l1d, 'Ds', AG.SEEDS[0], 43.8, 43.0)
        # a soft-rendered criterion run, an incomplete run and a degrade are caught / marked
        shutil.rmtree(os.path.join(l1d, f'Du-hetero-s{AG.SEEDS[1]}'))
        _summary(l1d, 'Du', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0, rendering='soft')
        assert AG.aggregate(l1, l1d)['verdict'] == 'INCOMPLETE'
        shutil.rmtree(os.path.join(l1d, f'Du-hetero-s{AG.SEEDS[1]}'))
        _summary(l1d, 'Du', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0, complete=False)
        assert AG.aggregate(l1, l1d)['verdict'] == 'INCOMPLETE'
        shutil.rmtree(os.path.join(l1d, f'Du-hetero-s{AG.SEEDS[1]}'))
        _summary(l1d, 'Du', AG.SEEDS[1], 46.0, 41.5, other=25.0, none=40.0, none_other=18.0, none_fresh=40.0)   # VAL +6, fresh +1.5 -> drop 4.5
        a = AG.aggregate(l1, l1d)
        assert a['verdict'] == 'CHANNEL-YES / SHEAF-NEUTRAL (degrade: AMBIGUOUS)'
        shutil.rmtree(os.path.join(l1d, f'Du-hetero-s{AG.SEEDS[1]}'))
        _summary(l1d, 'Du', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        r = subprocess.run([sys.executable, '-B', AGG_D, '--l1', l1, '--l1d', l1d, '--out', os.path.join(d, 'agg')], capture_output=True, text=True)
        assert r.returncode == 0 and 'VERDICT: CHANNEL-YES / SHEAF-NEUTRAL' in r.stdout and os.path.isfile(os.path.join(d, 'agg.md'))
        r = subprocess.run([sys.executable, '-B', AGG_D, '--l1', l1, '--l1d', l1d, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1d(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
