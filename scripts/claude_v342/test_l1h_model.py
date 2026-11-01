"""Unit tests for l1h_model.py / l1h_train.py / l1h_aggregate.py (batch L1h, D8-91): CPU, tiny random models from test_l1_model.py.
What is tested: AlignedSender reads slot k from the hidden state at position pos_k ONLY (the other positions do not matter), fills the slots
beyond the answer with the learned empty vectors, keeps the L1 precision head (lam = 1 at init; none for fixed_lam) and rejects a wrong
positions shape; AgentH.answer_positions gives len(pre) + k for the first K answer tokens and -1 beyond, and encode() reads them; PairH is the
L1f pair (same2, two independent instances, the remapped receiver state frozen) and trains the aligned sender only; slot_accuracy counts per
slot; the CLI (--tiny, --pair same2) writes an L1h summary with the slot readout; the aggregator uses the L1h names (G1h / CHANNEL-NO-6-same-aligned /
CHANNEL-YES-same-aligned)."""
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
import l1e_model as ME  # noqa: E402
import l1f_model as MF  # noqa: E402
import l1h_aggregate as AG  # noqa: E402
import l1h_model as MH  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_H = os.path.join(HERE, 'l1h_train.py')
AGG_H = os.path.join(HERE, 'l1h_aggregate.py')


def test_aligned_sender_reads_the_answer_positions_only():
    torch.manual_seed(0)
    ad = MH.AlignedSender(32, k=K, d_e=DE, fixed_lam=False)
    assert ad.lam_head is not None and ad.n_heads == 0 and not ad.fixed_lam
    h = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.long)
    pos = torch.tensor([[3, 4, 5, -1], [6, -1, -1, -1]])
    z, ll = ad(h, mask, pos)
    assert z.shape == (2, K, DE) and ll.shape == (2, K, DE) and torch.allclose(ll, torch.zeros_like(ll))   # lam = 1 at init
    assert float(z.mean(-1).abs().max()) < 1e-4                                 # LayerNorm output
    h2 = h.clone()                                                              # the positions that are not read leave z unchanged
    h2[0, [0, 1, 2, 6, 7, 8]] += 100.0 * torch.randn(6, 32)
    h2[1, [0, 1, 2, 3, 4, 5, 7, 8]] += 100.0 * torch.randn(8, 32)
    z2, _ = ad(h2, mask, pos)
    assert torch.allclose(z, z2, atol=1e-5)
    h3 = h.clone()                                                              # a read position changes its own slot only
    h3[0, 4] += 3.0 * torch.randn(32)
    z3, _ = ad(h3, mask, pos)
    assert not torch.allclose(z[0, 1], z3[0, 1], atol=1e-4)
    assert torch.allclose(z[0, 0], z3[0, 0], atol=1e-6) and torch.allclose(z[0, 2:], z3[0, 2:], atol=1e-6) and torch.allclose(z[1], z3[1], atol=1e-6)
    e = ad.ln_out(ad.empty)                                                     # empty slots = the learned empty vectors, whatever the example
    assert torch.allclose(z[0, 3], e[3], atol=1e-5) and torch.allclose(z[1, 1:], e[1:], atol=1e-5)
    with pytest.raises(ValueError):
        ad(h, mask, pos[:, :2])
    fixed = MH.AlignedSender(32, k=K, d_e=DE, fixed_lam=True)
    assert fixed.lam_head is None and fixed.fixed_lam
    z4, ll4 = fixed(h, mask, pos)
    assert z4.shape == (2, K, DE) and torch.allclose(ll4, torch.zeros_like(ll4))


def _agents(arm):
    fixed = MH.BASE_ARM[arm] == 'U'
    a = MH.AgentH.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    b = MH.AgentH.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    return a, b


def test_agent_h_answer_positions_and_encode():
    a, _ = _agents('Hu')
    assert isinstance(a, ME.AgentE) and isinstance(a.adapter, MH.AlignedSender) and a.adapter.fixed_lam
    pieces = [([5, 6, 7], [8, 9], []), ([5, 6], [8, 9, 10, 11, 12], []), ([4], [], [])]
    A = a.assemble([(p, s, []) for p, s, _ in pieces], None, 'right', embeds=False)
    pos = a.answer_positions(pieces, A)
    assert pos.tolist() == [[3, 4, -1, -1], [2, 3, 4, 5], [-1, -1, -1, -1]]
    for b, (p, s, _) in enumerate(pieces):
        for k in range(min(len(s), K)):
            assert int(A['ids'][b, int(pos[b, k])]) == s[k]                      # the slot positions carry the answer tokens
    z, lam = a.encode(pieces)
    assert z.shape == (3, K, DE) and lam.shape == (3, K, DE) and torch.allclose(lam, torch.ones_like(lam))
    e = a.adapter.ln_out(a.adapter.empty).detach()
    assert torch.allclose(z[2].detach(), e, atol=1e-5) and torch.allclose(z[0, 2:].detach(), e[2:], atol=1e-5)
    assert not torch.allclose(z[0, 0].detach(), e[0], atol=1e-3)
    z2, _ = a.encode([([5, 6, 7], [8, 9], [])])                                # the same prompt + answer alone -> the same slots (right padding, causal)
    assert torch.allclose(z[0].detach(), z2[0].detach(), atol=1e-4)
    assert all(not p.requires_grad for n, p in a.named_parameters() if 'lora_' in n)


def test_pair_h_trains_the_aligned_sender_only():
    for arm in MH.ARMS:
        a, b = _agents(arm)
        ph = MH.PairH([a, b], 'same2', arm)
        assert isinstance(ph, MF.PairF) and ph.pair == 'same2' and ph.arm == MH.BASE_ARM[arm] and ph.arm_c == arm and ph.idx == [0, 1] and ph.n_agents == 2
        for ag in ph.agents:
            ag.freeze_lora()
        for n, p in ph.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(False)
        ph.train()
        drop = torch.zeros(2, 2, dtype=torch.bool) if arm == 'Hu' else None
        loss, info = ph.step(TL.EXAMPLES[:2], train=True, drop=drop)
        assert torch.isfinite(loss) and info['z'].shape == (2, 2, K, DE)
        loss.backward()
        names = dict(ph.named_parameters())
        assert all(names[n].grad is None for n in names if 'lora_' in n)
        n_real = sum(len(ag.answer_cache[ex['id']][1]) for ag in ph.agents for ex in TL.EXAMPLES[:2])
        got = {n for n, p in names.items() if '.adapter.' in n and p.grad is not None and float(p.grad.abs().sum()) > 0}
        assert got, 'the adapter must receive a gradient'
        if n_real:
            assert any('.adapter.proj.' in n for n in got)
        if n_real < 2 * 2 * K:
            assert any('.adapter.empty' in n for n in got)
        if arm == 'Hs':
            assert names['w_raw'].grad is not None
    with pytest.raises(ValueError):
        MH.PairH([a, a], 'same2', 'Hu')                                        # the same instance twice
    with pytest.raises(ValueError):
        MH.PairH([a, b], 'hetero', 'Hu')
    with pytest.raises(ValueError):
        MH.PairH([a, b], 'same2', 'Fu')


def test_build_pair_remap_eval_paths_and_slot_accuracy():
    p0 = MH.build_pair('Hu', 'same2', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    assert p0.pair == 'same2' and all(isinstance(ag.adapter, MH.AlignedSender) and ag.k == K for ag in p0.agents) and p0.agent(0) is not p0.agent(1)
    state = {n: torch.randn_like(p) for n, p in p0.named_parameters() if 'lora_' in n and n.startswith('agents.0.')}
    p1 = MH.build_pair('Hs', 'same2', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=state, k=K, d_e=DE, n_heads=2)
    names = dict(p1.named_parameters())
    for k, t in state.items():
        assert torch.equal(names[k].detach(), t) and torch.equal(names['agents.1.' + k[len('agents.0.'):]].detach(), t)
    assert all(not p.requires_grad for n, p in names.items() if 'lora_' in n) and 'w_raw' in M.trainable_state(p1)
    r = MH.slot_accuracy(p1, TL.EXAMPLES, eval_batch=2)
    assert r['n'] == len(TL.EXAMPLES) and len(r['acc_gold_by_slot']) == K and len(r['acc_sender_by_slot']) == K and len(r['n_by_slot']) == K
    assert r['n_by_slot'] == sorted(r['n_by_slot'], reverse=True) and r['n_by_slot'][0] == 2 * len(TL.EXAMPLES)   # every gold answer has a first token; two receivers
    assert all(a is None or 0.0 <= a <= 1.0 for a in r['acc_gold_by_slot'] + r['acc_sender_by_slot']) and 0.0 <= r['acc_gold_mean'] <= 1.0
    assert all(k in r for k in ('acc_sender_mean',))
    v = MH.eval_render(p1, TL.EXAMPLES, hard=True, drop_p=0.0, c_check=True, eval_batch=2, seed=1)
    assert v['c_assert'] is True
    assert MH.eval_baseline(p1, TL.EXAMPLES, 'gold_hint', eval_batch=2)['kind'] == 'gold_hint'
    assert MH.hint_metrics(p1, TL.EXAMPLES, eval_batch=2)['n'] == len(TL.EXAMPLES)
    with pytest.raises(ValueError):
        MH.build_pair('Hu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    with pytest.raises(ValueError):
        MH.build_pair('Fu', 'same2', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def test_cli_tiny_run_writes_an_l1h_summary():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Hu')
        cmd = [sys.executable, '-B', TRAIN_H, '--tiny', '--arm', 'Hu', '--pair', 'same2', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
               '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
               '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2']
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['batch_name'] == 'L1h' and s['prereg'] == 'D8-91' and s['arm'] == 'Hu' and s['pair'] == 'same2' and s['base_arm'] == 'U'
        assert set(s['scripts_sha']) == set(MH.SCRIPTS) and 'AlignedSender' in s['channel']['adapter'] and s['channel']['rendering'] == 'hard'
        assert s['slot_acc'] is not None and len(s['slot_acc']['acc_gold_by_slot']) == K and s['slot_acc']['n'] == len(TL.EXAMPLES)
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'Hu', 'same2', 5)
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert any('.adapter.proj.' in n for n in ck['state']) and any('.adapter.empty' in n for n in ck['state']) and all('lora_' not in n for n in ck['state'])
        assert not any('.adapter.queries' in n or '.adapter.blocks.' in n for n in ck['state'])
        r2 = subprocess.run(cmd[:3] + ['--tiny', '--arm', 'Hu', '--pair', 'hetero'] + cmd[7:], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r2.returncode != 0                                               # the L1h trainer accepts pair same2 only
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _summary(d, arm, pair, seed, val, fresh, other=None, none=None, none_other=None):
    p = os.path.join(d, f'{arm}-{pair}-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair=pair, seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=True, scripts_sha={'l1h_model.py': 'abc'})
    if arm in ('Hu', 'Hs'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=40.0)), gold_hint=dict(val=dict(em=52.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['soft'] = dict(val=dict(em=val - 1.0))
        s['hint'] = dict(val=dict(hint_by_receiver_bucket={'0': {'other': dict(em_gold=50.0, n=5)}}))
        s['channel'] = dict(rendering='hard', receiver_training='none (frozen B0 checkpoint)')
    if arm == 'Hs':
        s['c_assert'] = True
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def test_aggregator_reads_hetero_l1_and_same2_l1h_runs():
    d = tempfile.mkdtemp()
    try:
        l1, l1h = os.path.join(d, 'l1'), os.path.join(d, 'l1h')
        os.makedirs(l1); os.makedirs(l1h)
        for s in AG.SEEDS:
            _summary(l1, 'B0', 'hetero', s, 42.0, 41.0)
            _summary(l1, 'T', 'hetero', s, 58.0, 56.0)
        assert AG.PAIR == 'same2' and AG.L1D_ARMS == ('Hu', 'Hs') and AG.ARM_DIR['Hu'] == 'l1h'
        _summary(l1h, 'Hu', 'same2', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1h)
        assert a['verdict'] == 'CHANNEL-NO-6-same-aligned' and a['g1_lines'][0] == f'G1h(s{AG.SEEDS[0]}): FAIL' and a['batch'] == 'L1h' and a['prereg'] == 'D8-91'
        assert a['runs'][f'B0-hetero-s{AG.SEEDS[0]}']['found'] and not a['runs'][f'B0-hetero-s{AG.SEEDS[0]}']['problems']
        shutil.rmtree(os.path.join(l1h, f'Hu-same2-s{AG.SEEDS[0]}'))
        _summary(l1h, 'Hu', 'same2', AG.SEEDS[0], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1h)
        assert a['verdict'] == 'CHANNEL-YES-same-aligned-pending-s2' and a['g1_lines'][0].endswith('PASS')
        _summary(l1h, 'Hu', 'same2', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1h)
        assert a['verdict'] == 'CHANNEL-YES-same-aligned / SHEAF-pending' and a['criteria']['reference'][f'B0_s{AG.SEEDS[0]}'] == 42.0 and 'G1h' in a['criteria'] and 'P2h' in a['criteria']
        r = subprocess.run([sys.executable, '-B', AGG_H, '--l1', l1, '--l1h', l1h, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1h(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
