"""Unit tests for l1c_model.py / l1c_train.py / l1c_aggregate.py (batch L1c, D8-86): CPU, tiny random models from test_l1_model.py.
What is tested: the decoder emits convex combinations of the receiver's real token embeddings (and E[argmax] in hard mode; E is not saved);
the targets are the partner's gold facts cut to K and filler-padded; pass 1 ignores the LoRA (the receiver) and the message loss reaches the
adapter, the decoder and w (Cs) but never the LoRA / base; Cu applies the training dropout; the pinned evaluation path runs on a PairC
(GBP messages, C assertion, absence p = 1 falls back as in L1); the hard evaluation, the baselines and the message metrics run; build_pair
loads and freezes a receiver state; the CLI (--tiny) writes summary.json with the L1 schema keys plus the L1c blocks, never overwrites, and
keeps the summary under the soft deadline; the aggregator's verdict table (D8-86 section 5) on synthetic summaries, with rounded margins."""
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
import l1c_aggregate as AG  # noqa: E402
import l1c_model as MC  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = TL.K, TL.DE
TRAIN_C = os.path.join(HERE, 'l1c_train.py')
AGG_C = os.path.join(HERE, 'l1c_aggregate.py')


def _pair(arm, max_ctx=None, n_blocks=2):
    fixed = MC.BASE_ARM[arm] == 'U'
    a = MC.AgentC.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=n_blocks)
    b = MC.AgentC.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=n_blocks)
    pc = MC.PairC([a, b], 'hetero', arm, max_ctx=max_ctx)
    for n, p in pc.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pc


# ---------------------------------------------------------------- (c) the decoder
def test_decoder_is_a_convex_combination_of_real_embeddings_and_hard_is_argmax():
    pair = _pair('Cu')
    ag = pair.agent(0)
    x = torch.randn(2, K, DE)
    lg = ag.reader.logits(x)
    assert lg.shape == (2, K, ag.reader.vocab) and lg.dtype == torch.float32
    p = torch.softmax(lg, -1)
    assert torch.allclose(p.sum(-1), torch.ones(2, K), atol=1e-5)
    soft = ag.reader(x)
    assert torch.allclose(soft, p @ ag.reader.E32, atol=1e-5)
    hard = ag.reader(x, hard=True)
    assert torch.equal(hard, ag.reader.E32[lg.argmax(-1)])
    assert 'E32' not in ag.reader.state_dict() and all('E32' not in n for n in M.trainable_state(pair))
    assert torch.equal(ag.reader.E32, ag.embed.weight.detach().float())


# ---------------------------------------------------------------- (d) the targets
def test_targets_are_the_partner_facts_cut_to_k_and_filler_padded():
    pair = _pair('Cu')
    ex = TL.EXAMPLES[0]
    for i in range(2):
        ag = pair.agent(i)
        ids, n = ag.target_ids(ex, 1 - i)
        full = ag.ids(M.DATA.text_sender_target(ex, 1 - i))
        assert len(ids) == K and n == min(K, len(full))
        assert ids[:n] == [int(t) for t in full[:K]]
        assert all(t == ag.filler for t in ids[n:])
    tgt, lens = pair.targets(TL.EXAMPLES[:2], 1)
    assert tgt.shape == (2, K) and tgt.dtype == torch.long and len(lens) == 2
    tgt2, _ = pair.targets(TL.EXAMPLES[:2], 1)                                 # cached
    assert torch.equal(tgt, tgt2)


# ---------------------------------------------------------------- (a) pass 1 ignores the LoRA; the gradient never reaches it
def test_encode_ignores_the_lora_and_the_loss_reaches_only_the_channel():
    for arm in MC.ARMS:
        pair = _pair(arm)
        batch = TL.EXAMPLES[:2]
        pair.eval()
        z1, _ = pair.encode_both(batch)
        with torch.no_grad():
            for n, p in pair.named_parameters():
                if 'lora_B' in n:
                    p.add_(1.0)
        z2, _ = pair.encode_both(batch)
        assert torch.allclose(z1, z2, atol=1e-6), f'{arm}: pass 1 must not see the LoRA'
        pair.train()
        loss, info = pair.step(batch, train=True)
        assert torch.isfinite(loss)
        loss.backward()
        names = dict(pair.named_parameters())
        assert all(names[n].grad is None for n in names if 'lora_' in n)
        assert all(not p.requires_grad for n, p in names.items() if 'lora_' not in n and '.adapter.' not in n and '.reader.' not in n and n != 'w_raw')
        for i in range(2):
            ad = [p for n, p in names.items() if f'agents.{i}.adapter.' in n and p.grad is not None]
            assert ad and any(float(p.grad.abs().sum()) > 0 for p in ad), f'{arm}: adapter {i} got no gradient'
            rd = [p for n, p in names.items() if f'agents.{i}.reader.' in n and p.grad is not None]
            assert rd and any(float(p.grad.abs().sum()) > 0 for p in rd), f'{arm}: decoder {i} got no gradient'
        if arm == 'Cs':
            assert names['w_raw'].grad is not None
            assert info['drop'] is None
        else:
            assert info['drop'].shape == (2, 2) and info['drop'].dtype == torch.bool
        assert set(info) >= {'loss_agent', 'acc_agent', 'msg_acc_agent', 'msg_acc', 'acc'}


def test_cu_training_dropout_zeroes_the_message_and_eval_does_not():
    pair = _pair('Cu')
    z = torch.randn(2, 3, K, DE)
    lam = torch.ones_like(z)
    torch.manual_seed(0)
    drop = torch.tensor([[True, False, True], [False, False, True]])
    xs, used = pair.messages_train(z, lam, train=True, drop=drop)
    assert torch.equal(used, drop)
    import l1_fusion as FU
    src_all, dst_all = FU.directed_edges(M.EDGES, torch.device('cpu'))
    for i in range(2):
        e = int((dst_all == i).nonzero()[0])                                  # the directed edge into receiver i
        src = int(src_all[e])
        for b in range(3):
            if drop[e, b]:
                assert torch.equal(xs[i][b], torch.zeros(K, DE))
            else:
                assert torch.equal(xs[i][b], z[src, b])
    xs2, used2 = pair.messages_train(z, lam, train=False)
    assert used2 is not None and not bool(used2.any())
    xe, x, prec = pair.messages_eval(z, lam)
    assert x is None and prec is None and all(torch.equal(xe[i], z[1 - i]) for i in range(2))


# ---------------------------------------------------------------- the pinned evaluation path on a PairC
def test_pinned_eval_latent_runs_with_c_assertion_and_absence_fallback():
    pair = _pair('Cs')
    r = M.eval_latent(pair, TL.EXAMPLES, drop_p=0.0, eval_batch=2, c_check=True, seed=1)
    assert r['c_assert'] is True and r['n'] == len(TL.EXAMPLES) and 'em_by_owner' in r
    r1 = M.eval_latent(pair, TL.EXAMPLES, drop_p=1.0, eval_batch=2, c_check=False, seed=1)
    assert r1['absence_p1_detail']['equal'] is True and r1['n_messages_dropped'] == r1['n_messages']
    pu = _pair('Cu')
    ru = M.eval_latent(pu, TL.EXAMPLES, drop_p=1.0, eval_batch=2, c_check=False, seed=1)
    assert ru['absence_p1_detail']['equal'] is True


def test_hard_eval_baselines_and_message_metrics_run():
    pair = _pair('Cu')
    h = MC.eval_hard(pair, TL.EXAMPLES, eval_batch=2, seed=1)
    assert pair.hard is False and 'em' in h and h['n_records'] == 2 * len(TL.EXAMPLES)
    n0 = MC.eval_baseline(pair, TL.EXAMPLES, 'none', eval_batch=2)
    g = MC.eval_baseline(pair, TL.EXAMPLES, 'gold32', eval_batch=2)
    assert n0['kind'] == 'none' and g['kind'] == 'gold32' and n0['n_records'] == g['n_records'] == 2 * len(TL.EXAMPLES)
    with pytest.raises(ValueError):
        MC.eval_baseline(pair, TL.EXAMPLES, 'bogus')
    mm = MC.message_metrics(pair, TL.EXAMPLES, eval_batch=2)
    assert set(mm) >= {'token_acc', 'msg_token_acc', 'f1_full', 'f1_trunc', 'n_tokens_mean', 'examples'} and mm['n'] == len(TL.EXAMPLES)
    assert 0.0 <= mm['token_acc'] <= 1.0 and len(mm['examples']) <= MC.MSG_EXAMPLES


def test_build_pair_loads_and_freezes_the_receiver_state():
    p0 = MC.build_pair('Cu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    state = {n: torch.randn_like(p) for n, p in p0.named_parameters() if 'lora_' in n}
    assert state
    p1 = MC.build_pair('Cs', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=state, k=K, d_e=DE, n_heads=2)
    names = dict(p1.named_parameters())
    for n, t in state.items():
        assert torch.equal(names[n].detach(), t) and not names[n].requires_grad
    tr = M.trainable_state(p1)
    assert tr and all('lora_' not in n for n in tr) and 'w_raw' in tr
    # pass 1 must leave the receiver frozen (PEFT's disable_adapter() exit would re-enable requires_grad)
    p1.encode_both(TL.EXAMPLES[:2])
    assert all(not p.requires_grad for n, p in p1.named_parameters() if 'lora_' in n)
    with pytest.raises(ValueError):
        MC.build_pair('Cu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state={'w_raw': torch.zeros(())}, k=K, d_e=DE, n_heads=2)
    with pytest.raises(ValueError):
        MC.build_pair('S1', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


# ---------------------------------------------------------------- the CLI on the tiny fixture
def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def _run(d, arm, out, extra=()):
    cmd = [sys.executable, '-B', TRAIN_C, '--tiny', '--arm', arm, '--pair', 'hetero', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
           '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
           '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2', *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_cli_tiny_run_writes_the_summary_and_never_overwrites():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Cs')
        r = _run(d, 'Cs', out, ['--keep-preds'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        for k in ('arm', 'base_arm', 'pair', 'seed', 'batch_name', 'prereg', 'val', 'fresh', 'absence', 'hard', 'baseline', 'message', 'complete', 'train', 'receiver',
                  'channel', 'timings', 'scripts_sha', 'c_assert', 'w'):
            assert k in s, k
        assert s['arm'] == 'Cs' and s['base_arm'] == 'S1' and s['batch_name'] == 'L1c' and s['prereg'] == 'D8-86' and s['complete'] is True and s['c_assert'] is True
        assert set(s['absence']) == {'0.0', '1.0'} and set(s['hard']) == {'val', 'fresh'}
        assert set(s['baseline']['none']) == {'val', 'fresh'} and set(s['baseline']['gold32']) == {'val', 'fresh'}
        assert s['baseline']['none']['val']['kind'] == 'none' and 'em_by_owner' in s['baseline']['none']['val']
        assert set(s['message']) == {'val', 'fresh'} and s['channel']['receiver_training'] == 'none (frozen B0 checkpoint)'
        assert set(s['scripts_sha']) == set(MC.SCRIPTS) and s['readouts_incomplete'] is None
        assert all(os.path.isfile(os.path.join(out, f)) for f in ('trainable.pt', 'preds_val.json', 'preds_fresh.json', 'preds_val_none.json', 'log.txt'))
        import l1_aggregate as A1
        problems = A1.validate_summary(s, 'Cs', 'hetero', 5)
        assert not problems, problems
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert all('lora_' not in n for n in ck['state']) and ck['receiver'] is None
        r2 = _run(d, 'Cs', out)
        assert r2.returncode != 0 and 'never overwrite' in (r2.stdout + r2.stderr)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cli_soft_deadline_keeps_the_summary_and_marks_the_readouts():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Cu')
        r = _run(d, 'Cu', out, ['--debug-deadline-at', 'readouts'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['readouts_incomplete'] and s['absence'].get('1.0') is None and 'val' in s['baseline']['none'] and s['baseline']['none']['val']['em'] is not None
        assert 'c_assert' not in s
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- the aggregator (D8-86 section 5)
def _summary(d, arm, seed, val, fresh, other=None, none=None, none_other=None, none_fresh=None, c_assert=True, complete=True, sha='abc', trained=False):
    p = os.path.join(d, f'{arm}-hetero-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair='hetero', seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=complete, scripts_sha={'l1c_model.py': sha})
    if arm in ('Cu', 'Cs'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=none_fresh if none_fresh is not None else 40.0)),
                             gold32=dict(val=dict(em=50.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['hard'] = dict(val=dict(em=val))
        s['message'] = dict(val=dict(f1_trunc=0.5, token_acc=0.6))
        s['channel'] = dict(receiver_training='trained' if trained else 'none (frozen B0 checkpoint)')
    if arm == 'Cs':
        s['c_assert'] = c_assert
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def _agg(l1, l1c, gate=None):
    return AG.aggregate(l1, l1c, gate)


def test_aggregator_verdicts():
    d = tempfile.mkdtemp()
    try:
        l1, l1c = os.path.join(d, 'l1'), os.path.join(d, 'l1c')
        os.makedirs(l1); os.makedirs(l1c)
        for s in AG.SEEDS:
            _summary(l1, 'B0', s, 42.0, 41.0)
            _summary(l1, 'T', s, 58.0, 56.0)
        a = _agg(l1, l1c)
        assert a['verdict'] == 'INCOMPLETE' and a['g1_lines'][0].endswith('INCOMPLETE')
        # G1 fails on seed 1 (EM gain 2.9 < 3) -> CHANNEL-NO-2
        _summary(l1c, 'Cu', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = _agg(l1c and l1, l1c)
        assert a['verdict'] == 'CHANNEL-NO-2' and a['g1_lines'][0] == f'G1(s{AG.SEEDS[0]}): FAIL'
        # G1 passes on EM but not on the other bucket -> FAIL
        shutil.rmtree(os.path.join(l1c, f'Cu-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cu', AG.SEEDS[0], 45.0, 44.0, other=22.9, none=40.0, none_other=18.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-NO-2'
        # exactly the thresholds (3.0 / 5.0 in floating point) -> PASS; seed 2 missing -> INCOMPLETE; gate mode prints one line only
        shutil.rmtree(os.path.join(l1c, f'Cu-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cu', AG.SEEDS[0], 43.3, 42.0, other=23.3, none=40.3, none_other=18.3)
        a = _agg(l1, l1c)
        assert a['g1_lines'][0] == f'G1(s{AG.SEEDS[0]}): PASS' and a['verdict'] == 'INCOMPLETE'
        g = _agg(l1, l1c, gate=AG.SEEDS[0])
        assert g['g1_lines'] == [f'G1(s{AG.SEEDS[0]}): PASS'] and g['verdict'] is None
        # both Cu pass, Cs missing -> CHANNEL-YES / SHEAF-pending
        _summary(l1c, 'Cu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-YES / SHEAF-pending'
        # Cs - Cu = +1.0 on both seeds (44.3 - 43.3, 47.0 - 46.0) -> ADDS
        _summary(l1c, 'Cs', AG.SEEDS[0], 44.3, 43.0)
        _summary(l1c, 'Cs', AG.SEEDS[1], 47.0, 46.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-YES / SHEAF-ADDS'
        # Cs below Cu by 1.0 on both -> HURTS; mixed -> AMBIGUOUS; within -> NEUTRAL
        for s in AG.SEEDS:
            shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{s}'))
        _summary(l1c, 'Cs', AG.SEEDS[0], 42.3, 41.0)
        _summary(l1c, 'Cs', AG.SEEDS[1], 45.0, 44.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-YES / SHEAF-HURTS'
        shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{AG.SEEDS[1]}'))
        _summary(l1c, 'Cs', AG.SEEDS[1], 46.5, 44.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-YES / SHEAF-AMBIGUOUS'
        shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cs', AG.SEEDS[0], 43.8, 43.0)
        assert _agg(l1, l1c)['verdict'] == 'CHANNEL-YES / SHEAF-NEUTRAL'
        # C assertion false on a Cs run -> INCOMPLETE; a trained receiver -> INCOMPLETE; sha conflict -> INCOMPLETE; complete false -> INCOMPLETE
        shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cs', AG.SEEDS[0], 43.8, 43.0, c_assert=False)
        a = _agg(l1, l1c)
        assert a['verdict'] == 'CHANNEL-YES / SHEAF-pending' and any('c_assert false' in r for r in a['reasons'])
        shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cs', AG.SEEDS[0], 43.8, 43.0, sha='zzz')
        assert _agg(l1, l1c)['verdict'] == 'INCOMPLETE'
        shutil.rmtree(os.path.join(l1c, f'Cs-hetero-s{AG.SEEDS[0]}'))
        _summary(l1c, 'Cs', AG.SEEDS[0], 43.8, 43.0)
        shutil.rmtree(os.path.join(l1c, f'Cu-hetero-s{AG.SEEDS[1]}'))
        _summary(l1c, 'Cu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0, trained=True)
        assert _agg(l1, l1c)['verdict'] == 'INCOMPLETE'
        shutil.rmtree(os.path.join(l1c, f'Cu-hetero-s{AG.SEEDS[1]}'))
        _summary(l1c, 'Cu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0, complete=False)
        assert _agg(l1, l1c)['verdict'] == 'INCOMPLETE'
        # the CLI writes json + md and prints the lines
        shutil.rmtree(os.path.join(l1c, f'Cu-hetero-s{AG.SEEDS[1]}'))
        _summary(l1c, 'Cu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        r = subprocess.run([sys.executable, '-B', AGG_C, '--l1', l1, '--l1c', l1c, '--out', os.path.join(d, 'agg')], capture_output=True, text=True)
        assert r.returncode == 0 and 'VERDICT: CHANNEL-YES / SHEAF-NEUTRAL' in r.stdout and os.path.isfile(os.path.join(d, 'agg.md')) and os.path.isfile(os.path.join(d, 'agg.json'))
        r = subprocess.run([sys.executable, '-B', AGG_C, '--l1', l1, '--l1c', l1c, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
