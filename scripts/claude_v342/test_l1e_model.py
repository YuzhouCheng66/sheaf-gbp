"""Unit tests for l1e_model.py / l1e_train.py / l1e_aggregate.py (batch L1e, D8-88): CPU, tiny random models from test_l1_model.py.
What is tested: SenderAdapter3 normalises the hidden states per position before the key / value projections and never attends to position 0
(a huge vector at position 0 leaves z unchanged, a change at position 1 does not); its output interface and the precision head are those of
SenderAdapter2; the L1e pair is a PairD with the new adapter and the Eu / Es arm names (same targets, template, caching, frozen LoRA); build_pair;
the CLI (--tiny) writes an L1e summary; the aggregator uses the L1e names (G1e / P2e / CHANNEL-NO-4) on synthetic summaries."""
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
import l1c_model as MC  # noqa: E402
import l1d_model as MD  # noqa: E402
import l1e_aggregate as AG  # noqa: E402
import l1e_model as ME  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_E = os.path.join(HERE, 'l1e_train.py')
AGG_E = os.path.join(HERE, 'l1e_aggregate.py')


def _pair(arm, max_ctx=None):
    fixed = ME.BASE_ARM[arm] == 'U'
    a = ME.AgentE.from_pretrained_objects(*TL._tiny(32, 1, True), 'tiny-a', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    b = ME.AgentE.from_pretrained_objects(*TL._tiny(48, 2, False), 'tiny-b', comm=True, fixed_lam=fixed, k=K, d_e=DE, n_heads=2, lora_dropout=0.0, n_blocks=2)
    pe = ME.PairE([a, b], 'hetero', arm, max_ctx=max_ctx)
    for ag in pe.agents:
        ag.freeze_lora()
    for n, p in pe.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(False)
    return pe


def test_adapter3_normalises_and_masks_position_0():
    torch.manual_seed(0)
    ad = ME.SenderAdapter3(32, k=K, d_e=DE, n_heads=2, fixed_lam=False, n_blocks=2)
    assert isinstance(ad, MC.SenderAdapter2) and isinstance(ad.ln_h, torch.nn.LayerNorm) and ad.lam_head is not None
    h = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.long)
    mask[1, 5:] = 0
    z, ll = ad(h, mask)
    assert z.shape == (2, K, DE) and ll.shape == (2, K, DE) and torch.allclose(ll, torch.zeros_like(ll))   # lam = 1 at init
    h2 = h.clone()
    h2[:, 0] += 1000.0 * torch.randn(32)                                      # a massive-activation first token (not a uniform shift: LayerNorm would remove that)
    z2, _ = ad(h2, mask)
    assert torch.allclose(z, z2, atol=1e-5), 'position 0 must not be attended'
    h3 = h.clone()
    h3[:, 1] += 3.0 * torch.randn(32)
    z3, _ = ad(h3, mask)
    assert not torch.allclose(z, z3, atol=1e-4), 'other positions must be read'
    # the keys / values see normalised states: scaling a position's state does not change z
    h4 = h.clone()
    h4[:, 2] *= 50.0
    z4, _ = ad(h4, mask)
    assert torch.allclose(z, z4, atol=1e-4)
    fixed = ME.SenderAdapter3(32, k=K, d_e=DE, n_heads=2, fixed_lam=True, n_blocks=2)
    assert fixed.lam_head is None and fixed.fixed_lam


def test_pair_e_is_a_pair_d_with_the_new_adapter_and_names():
    for arm in ME.ARMS:
        pair = _pair(arm)
        assert isinstance(pair, MD.PairD) and pair.arm == ME.BASE_ARM[arm] and pair.arm_c == arm
        assert all(isinstance(ag.adapter, ME.SenderAdapter3) and isinstance(ag, MD.AgentD) for ag in pair.agents)
        pair.train()
        drop = torch.zeros(2, 2, dtype=torch.bool) if arm == 'Eu' else None
        loss, info = pair.step(TL.EXAMPLES[:2], train=True, drop=drop)
        assert torch.isfinite(loss)
        loss.backward()
        names = dict(pair.named_parameters())
        assert all(names[n].grad is None for n in names if 'lora_' in n)
        assert any(float(p.grad.abs().sum()) > 0 for n, p in names.items() if '.adapter.ln_h.' in n and p.grad is not None)
        if arm == 'Es':
            assert names['w_raw'].grad is not None
    with pytest.raises(ValueError):
        ME.PairE([_pair('Eu').agent(0)], 'same', 'Eu')
    with pytest.raises(ValueError):
        ME.PairE(list(_pair('Eu').agents), 'hetero', 'Du')


def test_build_pair_and_eval_paths():
    p0 = ME.build_pair('Eu', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)
    assert p0.agent(0).k == K and isinstance(p0.agent(0).adapter, ME.SenderAdapter3)
    state = {n: torch.randn_like(p) for n, p in p0.named_parameters() if 'lora_' in n}
    p1 = ME.build_pair('Es', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), receiver_state=state, k=K, d_e=DE, n_heads=2)
    assert all(not p.requires_grad for n, p in p1.named_parameters() if 'lora_' in n) and 'w_raw' in M.trainable_state(p1)
    v = ME.eval_render(p1, TL.EXAMPLES, hard=True, drop_p=0.0, c_check=True, eval_batch=2, seed=1)
    assert v['c_assert'] is True
    p = ME.eval_render(p1, TL.EXAMPLES, hard=True, drop_p=1.0, c_check=False, eval_batch=2, seed=1)
    assert p['absence_p1_detail']['equal'] is True
    assert ME.eval_baseline(p1, TL.EXAMPLES, 'gold_hint', eval_batch=2)['kind'] == 'gold_hint'
    assert ME.hint_metrics(p1, TL.EXAMPLES, eval_batch=2)['n'] == len(TL.EXAMPLES)
    with pytest.raises(ValueError):
        ME.build_pair('Du', 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def test_cli_tiny_run_writes_an_l1e_summary():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'Es')
        cmd = [sys.executable, '-B', TRAIN_E, '--tiny', '--arm', 'Es', '--pair', 'hetero', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
               '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
               '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2']
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['batch_name'] == 'L1e' and s['prereg'] == 'D8-88' and s['arm'] == 'Es' and s['base_arm'] == 'S1' and s['c_assert'] is True
        assert set(s['scripts_sha']) == set(ME.SCRIPTS) and s['channel']['rendering'] == 'hard' and 'SenderAdapter3' in s['channel']['adapter']
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'Es', 'hetero', 5)
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert any('.adapter.ln_h.' in n for n in ck['state']) and all('lora_' not in n for n in ck['state'])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _summary(d, arm, seed, val, fresh, other=None, none=None, none_other=None, c_assert=True):
    p = os.path.join(d, f'{arm}-hetero-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair='hetero', seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=True, scripts_sha={'l1e_model.py': 'abc'})
    if arm in ('Eu', 'Es'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=40.0)), gold_hint=dict(val=dict(em=52.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['soft'] = dict(val=dict(em=val - 1.0))
        s['hint'] = dict(val=dict(hint_by_receiver_bucket={'0': {'other': dict(em_gold=50.0, n=5)}}))
        s['channel'] = dict(rendering='hard', receiver_training='none (frozen B0 checkpoint)')
    if arm == 'Es':
        s['c_assert'] = c_assert
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def test_aggregator_names_and_verdicts():
    d = tempfile.mkdtemp()
    try:
        l1, l1e = os.path.join(d, 'l1'), os.path.join(d, 'l1e')
        os.makedirs(l1); os.makedirs(l1e)
        for s in AG.SEEDS:
            _summary(l1, 'B0', s, 42.0, 41.0)
            _summary(l1, 'T', s, 58.0, 56.0)
        assert AG.L1D_ARMS == ('Eu', 'Es') and AG.ARM_DIR['Eu'] == 'l1e'
        _summary(l1e, 'Eu', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1e)
        assert a['verdict'] == 'CHANNEL-NO-4' and a['g1_lines'][0] == f'G1e(s{AG.SEEDS[0]}): FAIL' and a['batch'] == 'L1e' and a['prereg'] == 'D8-88'
        shutil.rmtree(os.path.join(l1e, f'Eu-hetero-s{AG.SEEDS[0]}'))
        _summary(l1e, 'Eu', AG.SEEDS[0], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        _summary(l1e, 'Eu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        _summary(l1e, 'Es', AG.SEEDS[0], 47.5, 46.0)
        _summary(l1e, 'Es', AG.SEEDS[1], 47.0, 46.0)
        a = AG.aggregate(l1, l1e)
        assert a['verdict'] == 'CHANNEL-YES / SHEAF-ADDS' and 'P2e' in a['criteria'] and 'G1e' in a['criteria']
        r = subprocess.run([sys.executable, '-B', AGG_E, '--l1', l1, '--l1e', l1e, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1e(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
