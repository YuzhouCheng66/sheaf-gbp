"""Unit tests for l1f_model.py / l1f_train.py / l1f_aggregate.py (batch L1f, D8-89): CPU, tiny random models from test_l1_model.py.
What is tested: PairF needs two independent instances and carries the pair label 'same2' while the pinned Pair logic runs as for two agents;
remap_receiver_state copies agent 0's LoRA tensors to agent 1 (and nothing else); build_pair loads the remapped state into both instances
and freezes it; the CLI (--tiny, --pair same2) writes an L1f summary (pair 'same2'); the aggregator reads L1 arms from the hetero directories
and the L1f arms from the same2 directories and uses the L1f names (G1f / CHANNEL-NO-5-same / CHANNEL-YES-same)."""
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
import l1f_aggregate as AG  # noqa: E402
import l1f_model as MF  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_F = os.path.join(HERE, 'l1f_train.py')
AGG_F = os.path.join(HERE, 'l1f_aggregate.py')


def _agents(arm):
    fixed = MF.BASE_ARM[arm] == 'U'
    a = ME.AgentE.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    b = ME.AgentE.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    return a, b


def test_pair_f_label_and_guards():
    a, b = _agents('Fu')
    pf = MF.PairF([a, b], 'same2', 'Fu')
    assert pf.pair == 'same2' and pf.arm == 'U' and pf.arm_c == 'Fu' and pf.idx == [0, 1] and pf.n_agents == 2
    with pytest.raises(ValueError):
        MF.PairF([a, a], 'same2', 'Fu')                                        # the same instance twice
    with pytest.raises(ValueError):
        MF.PairF([a, b], 'hetero', 'Fu')
    with pytest.raises(ValueError):
        MF.PairF([a, b], 'same2', 'Eu')
    pf.train()
    loss, info = pf.step(TL.EXAMPLES[:2], train=True, drop=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_remap_and_build_pair_with_the_receiver_state():
    p0 = MF.build_pair('Fu', 'same2', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    assert p0.pair == 'same2'
    state = {n: torch.randn_like(p) for n, p in p0.named_parameters() if 'lora_' in n and n.startswith('agents.0.')}
    state['agents.1.model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'] = torch.zeros(1)   # a foreign agent-1 tensor is ignored
    state['w_raw'] = torch.zeros(())
    r = MF.remap_receiver_state(state)
    assert set(r) == {k for k in state if k.startswith('agents.0.')} | {'agents.1.' + k[len('agents.0.'):] for k in state if k.startswith('agents.0.')}
    for k in state:
        if k.startswith('agents.0.'):
            assert torch.equal(r['agents.1.' + k[len('agents.0.'):]], state[k])
    p1 = MF.build_pair('Fs', 'same2', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=state, k=K, d_e=DE, n_heads=2)
    names = dict(p1.named_parameters())
    for k, t in state.items():
        if k.startswith('agents.0.'):
            assert torch.equal(names[k].detach(), t) and torch.equal(names['agents.1.' + k[len('agents.0.'):]].detach(), t)
    assert all(not p.requires_grad for n, p in names.items() if 'lora_' in n) and 'w_raw' in M.trainable_state(p1)
    with pytest.raises(ValueError):
        MF.remap_receiver_state({'agents.1.x.lora_A': torch.zeros(1)})
    with pytest.raises(ValueError):
        MF.build_pair('Fu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def test_cli_tiny_run_writes_an_l1f_summary():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Fu')
        cmd = [sys.executable, '-B', TRAIN_F, '--tiny', '--arm', 'Fu', '--pair', 'same2', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
               '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
               '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2']
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['batch_name'] == 'L1f' and s['prereg'] == 'D8-89' and s['arm'] == 'Fu' and s['pair'] == 'same2' and s['base_arm'] == 'U'
        assert set(s['scripts_sha']) == set(MF.SCRIPTS) and 'pair_note' in s['channel'] and s['channel']['rendering'] == 'hard'
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'Fu', 'same2', 5)
        r2 = subprocess.run(cmd[:3] + ['--tiny', '--arm', 'Fu', '--pair', 'hetero'] + cmd[7:], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r2.returncode != 0                                               # the L1f trainer accepts pair same2 only
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _summary(d, arm, pair, seed, val, fresh, other=None, none=None, none_other=None):
    p = os.path.join(d, f'{arm}-{pair}-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair=pair, seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=True, scripts_sha={'l1f_model.py': 'abc'})
    if arm in ('Fu', 'Fs'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=40.0)), gold_hint=dict(val=dict(em=52.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['soft'] = dict(val=dict(em=val - 1.0))
        s['hint'] = dict(val=dict(hint_by_receiver_bucket={'0': {'other': dict(em_gold=50.0, n=5)}}))
        s['channel'] = dict(rendering='hard', receiver_training='none (frozen B0 checkpoint)')
    if arm == 'Fs':
        s['c_assert'] = True
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def test_aggregator_reads_hetero_l1_and_same2_l1f_runs():
    d = tempfile.mkdtemp()
    try:
        l1, l1f = os.path.join(d, 'l1'), os.path.join(d, 'l1f')
        os.makedirs(l1); os.makedirs(l1f)
        for s in AG.SEEDS:
            _summary(l1, 'B0', 'hetero', s, 42.0, 41.0)
            _summary(l1, 'T', 'hetero', s, 58.0, 56.0)
        assert AG.PAIR == 'same2' and AG.L1D_ARMS == ('Fu', 'Fs')
        _summary(l1f, 'Fu', 'same2', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1f)
        assert a['verdict'] == 'CHANNEL-NO-5-same' and a['g1_lines'][0] == f'G1f(s{AG.SEEDS[0]}): FAIL'
        assert a['runs'][f'B0-hetero-s{AG.SEEDS[0]}']['found'] and not a['runs'][f'B0-hetero-s{AG.SEEDS[0]}']['problems']
        shutil.rmtree(os.path.join(l1f, f'Fu-same2-s{AG.SEEDS[0]}'))
        for s in AG.SEEDS:
            _summary(l1f, 'Fu', 'same2', s, 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1f)
        assert a['verdict'] == 'CHANNEL-YES-same / SHEAF-pending' and a['criteria']['reference'][f'B0_s{AG.SEEDS[0]}'] == 42.0
        r = subprocess.run([sys.executable, '-B', AGG_F, '--l1', l1, '--l1f', l1f, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1f(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
