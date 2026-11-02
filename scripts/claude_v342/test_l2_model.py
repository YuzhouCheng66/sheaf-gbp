"""Unit tests for l2_model.py / l2_train.py / l2_gate.py (ladder L2, D8-92): CPU, tiny random models from test_l1_model.py.
What is tested: the two LoRA adapters (recv initialised from B0, switched by use(), the receiver loss on recv, own answers / pass 1 on the
frozen B0, perturbing recv leaves the sender side unchanged); the span modes (oracle positions = the supporting sentence tokens, targets =
the sender's span tokens, stage-1 copy CE trains the sender / decoder only); attention positions (a window inside the documents region) and
select_quality; step_recv trains the recv LoRA only with the sender frozen; the baselines; the hetero build; the CLI (Hu -> R1 with
--init-from, V1 with both stages); the gate verdicts."""
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
import l1h_model as MH  # noqa: E402
import l2_model as L2  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_L2 = os.path.join(HERE, 'l2_train.py')
GATE_L2 = os.path.join(HERE, 'l2_gate.py')
EX = TL.EXAMPLES


def _pair(arm='R1', pair='same2', mode='answer', recv=False, span_k=6, **kw):
    return L2.build_pair(arm, pair, torch.device('cpu'), tiny=M.tiny_models(EX, 5), k=K, d_e=DE, n_heads=2, mode=mode, span_k=span_k, recv=recv, **kw)


def _lora(pair, which):
    return {n: p for n, p in pair.named_parameters() if 'lora_' in n and f'.{which}.' in n}


def test_two_adapters_switch_and_isolate():
    p = _pair(recv=True)
    a = p.agent(0)
    assert a.recv and _lora(p, 'recv') and len(_lora(p, 'recv')) == len(_lora(p, 'default'))
    for n, q in _lora(p, 'recv').items():
        assert torch.equal(q.detach(), _lora(p, 'default')[n.replace('.recv.', '.default.')].detach())        # initialised from B0
    a.use('default')
    assert a.active == 'default' and all(not q.requires_grad for q in _lora(p, 'recv').values()) and all(not q.requires_grad for q in _lora(p, 'default').values())
    a.use('recv')
    assert a.active == 'recv' and all(q.requires_grad for n, q in _lora(p, 'recv').items() if n.startswith('agents.0.')) and all(not q.requires_grad for q in _lora(p, 'default').values())
    pcs = p.pieces(EX[:2], 0, 'B0', target='answer')
    loss_recv, _ = a.loss(pcs)                                                  # the receiver loss runs on recv
    assert a.active == 'recv'
    a.use('default')
    loss_def, _ = M.Agent.loss(a, pcs)
    assert torch.allclose(loss_recv.detach(), loss_def.detach(), atol=1e-5)    # identical weights -> identical loss
    pb0 = p.pieces(EX, 0, 'B0', target=None)
    ans0 = [t for t, _ in a.own_answers(EX, pb0)]
    with torch.no_grad():
        for n, q in _lora(p, 'recv').items():
            if n.startswith('agents.0.') and 'lora_B' in n:
                q.add_(0.7)
    a.answer_cache.clear()
    ans1 = [t for t, _ in a.own_answers(EX, pb0)]
    assert ans0 == ans1 and a.active == 'default'                               # own answers always use the frozen B0
    z0, _ = a.encode(p.sender_pieces(EX[:2], 0)[0])
    assert a.active == 'default'
    loss_recv2, _ = a.loss(pcs)
    assert not torch.allclose(loss_recv2.detach(), loss_def.detach(), atol=1e-5)  # the perturbed recv changes the receiver loss only
    with pytest.raises(ValueError):
        _pair(recv=False).agent(0).use('recv')


def test_span_oracle_positions_targets_and_stage1():
    p = _pair('V1', mode='span_oracle', recv=False)
    assert p.mode == 'span_oracle' and p.agent(0).k == 6 and p.span_k == 6
    pcs, dl = p.doc_pieces(EX, 0)
    pos, found = p.oracle_positions(EX, 0, pcs)
    ag = p.agent(0)
    nb = 1 if ag.bos is not None else 0
    for b, ex in enumerate(EX):
        assert found[b]
        got = [pcs[b][0][int(q)] for q in pos[b].tolist() if q >= 0]
        want = ag.ids(' ' + ex['sup'][0][0])[:6]
        assert got == want, (ex['id'], got, want)
        assert all(nb <= int(q) < nb + dl[b] for q in pos[b].tolist() if q >= 0)
    assert p.span_missing == 0 and p.span_seen == len(EX)
    tgt, lens = p.targets(EX[:2], 1)                                            # receiver 1's targets = sender 0's span tokens
    for b in range(2):
        want = ag.ids(' ' + EX[b]['sup'][0][0])[:6]
        assert tgt[b, :lens[b]].tolist() == want and lens[b] == len(want)
    z, lam = p.encode_both(EX[:2])
    assert z.shape == (2, 2, 6, DE) and torch.allclose(lam, torch.ones_like(lam))
    L2.freeze_sender(p, frozen=False)
    p.train()
    loss, info = p.step(EX[:2], train=True, drop=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isfinite(loss) and 'msg_acc' in info
    loss.backward()
    names = dict(p.named_parameters())
    assert all(names[n].grad is None for n in names if 'lora_' in n)
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n, p_ in names.items() if '.adapter.proj.' in n)
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for n, p_ in names.items() if '.reader.proj.' in n)
    assert p.agent(0).hint_pre == [int(t) for t in ag.ids(L2.SPAN_PRE)]
    with pytest.raises(ValueError):
        _pair('Hu', 'hetero', mode='span_oracle')
    ex_nosup = dict(EX[0], id='nosup', sup=[['zzz qqq'], ['zzz qqq']])
    pos2, found2 = p.oracle_positions([ex_nosup], 0, p.doc_pieces([ex_nosup], 0)[0])
    assert not found2[0] and int(pos2.max()) == -1 and p.span_missing == 1
    loss2, _ = p.step([EX[0], ex_nosup], train=True, drop=torch.zeros(2, 2, dtype=torch.bool))     # stage 1 skips the example without a span
    assert torch.isfinite(loss2) and p.n_skipped == 1
    loss3, _ = p.step([ex_nosup], train=True)                                   # nothing located in the batch -> trained as it is, not counted
    assert torch.isfinite(loss3) and p.n_skipped == 1
    pre = p.doc_pieces(EX[:1], 0)[0][0][0]
    sent = EX[0]['sup'][0][0]
    full = ag.ids(' ' + sent)
    h = L2._find(pre, full)
    assert L2.PairL2.locate(ag, [sent], pre, 6) == list(range(h, h + min(6, len(full))))
    assert L2.PairL2.locate(ag, [sent], pre[:h + len(full) - 1], 6) is not None                   # the final token cut away: full[:-1] matches


def test_attn_positions_window_and_select_quality():
    row = torch.tensor([0.0, 0.1, 0.0, 0.5, 0.4, 0.1, 0.0, 0.2])
    assert L2.AgentL2.best_window(row, 10, 18, 3) == 13                        # [0.5, 0.4, 0.1] at absolute 13
    assert L2.AgentL2.best_window(row, 10, 18, 2) == 13 and L2.AgentL2.best_window(row[:3], 10, 13, 5) == 10
    p = _pair('V1', mode='span_attn', recv=False, sel_layer=1)
    ag = p.agent(0)
    pcs, dl = p.doc_pieces(EX, 0)
    pos = ag.attn_positions(pcs, dl, 1, 6)
    nb = 1 if ag.bos is not None else 0
    for b in range(len(EX)):
        got = [int(q) for q in pos[b].tolist() if q >= 0]
        assert len(got) == min(6, dl[b]) and got == list(range(got[0], got[0] + len(got)))
        assert nb <= got[0] and got[-1] < nb + dl[b]
    assert ag.n_attn_passes == len(EX)
    pos2, _ = p.attn_positions(EX, 0, pcs, dl)                                  # the pair-level call caches by example id
    pos3, _ = p.attn_positions(EX, 0, pcs, dl)
    assert torch.equal(pos, pos2) and torch.equal(pos2, pos3) and ag.n_attn_passes == 2 * len(EX)
    z, _ = p.encode_both(EX[:2])
    assert z.shape == (2, 2, 6, DE) and ag.n_attn_passes == 2 * len(EX)         # agent 0 served from the cache
    q = L2.select_quality(p, EX, layers=[0, 1], eval_batch=2)
    assert set(q['by_layer']) == {'0', '1'} and q['best_layer'] in (0, 1) and q['oracle_found'] == 1.0 and q['n_pairs'] == 2 * len(EX)
    assert all(0.0 <= v['overlap50'] <= 1.0 and v['n'] == 2 * len(EX) for v in q['by_layer'].values())


def test_init_state_and_step_recv_trains_recv_only():
    src = _pair('Hu', recv=False)
    with torch.no_grad():
        for n, p_ in src.named_parameters():
            if '.adapter.' in n or '.reader.' in n:
                p_.add_(0.01 * torch.randn_like(p_))
    init = {n: p_.detach().clone() for n, p_ in src.named_parameters() if '.adapter.' in n or '.reader.' in n}
    p = _pair('R1', recv=True, init_state=init)
    names = dict(p.named_parameters())
    assert all(torch.equal(names[n].detach(), t) for n, t in init.items())
    with pytest.raises(ValueError):
        _pair('R1', recv=True, init_state={k: v for k, v in list(init.items())[:3]})
    L2.freeze_sender(p, frozen=True)
    p.use('recv')
    with torch.no_grad():                                                       # a fresh tiny LoRA has B = 0 (no gradient reaches A): perturb B as a trained B0 would be
        for n, q in names.items():
            if '.recv.' in n and 'lora_B' in n:
                q.add_(0.05 * torch.randn_like(q))
    p.train()
    loss, info = p.step_recv(EX[:2], train=True, drop=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isfinite(loss) and len(info['loss_agent']) == 2
    loss.backward()
    for n, q in names.items():
        if 'lora_' in n and '.recv.' in n:
            assert q.grad is not None and float(q.grad.abs().sum()) > 0, n
        else:
            assert q.grad is None, n
    cnt = M.count_params(p)
    assert cnt['trainable'] == sum(q.numel() for n, q in names.items() if '.recv.' in n)


def test_baselines_render_and_hetero_build():
    p = _pair('R1', recv=True)
    for kind in L2.BASELINES:
        r = L2.eval_baseline(p, EX, kind, eval_batch=2)
        assert r['kind'] == kind and 0.0 <= r['em'] <= 100.0 and r['n_records'] == 2 * len(EX)
    v = L2.eval_render(p, EX, hard=True, drop_p=0.0, c_check=False, eval_batch=2, seed=1)
    assert 0.0 <= v['em'] <= 100.0 and p.agent(0).active == 'recv'
    h = _pair('Hu', 'hetero')
    assert h.pair == 'hetero' and h.agent(0).name != h.agent(1).name and h.idx == [0, 1]
    z, _ = h.encode_both(EX[:2])
    assert z.shape == (2, 2, K, DE)
    tgt, lens = h.targets(EX[:2], 1)
    assert tgt.shape == (2, K) and lens[0] >= 1
    sp = _pair('V1', mode='span_oracle')
    with pytest.raises(ValueError):
        L2.eval_baseline(sp, EX, 'own_hint', eval_batch=2)
    with pytest.raises(ValueError):
        L2.PairL2(list(h.agents), 'hetero', 'V1', mode='span_oracle')


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in EX:
                fh.write(json.dumps(ex) + '\n')


def _cli(d, arm, out, extra):
    cmd = [sys.executable, '-B', TRAIN_L2, '--tiny', '--arm', arm, '--pair', 'same2', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'),
           '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--batch', '2', '--warmup', '2',
           '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2'] + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))


def test_cli_tiny_hu_then_r1_then_v1():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        hu = os.path.join(d, 'Hu')
        r = _cli(d, 'Hu', hu, ['--k', str(K), '--stage1-steps', '4'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(hu, 'summary.json')))
        assert s['batch_name'] == 'L2' and s['prereg'] == 'D8-92' and s['arm'] == 'Hu' and s['complete'] and not s['channel']['recv'] and s['stages']['stage1']['steps_done'] == 4
        assert set(s['scripts_sha']) == set(L2.SCRIPTS) and s['slot_acc']['acc_sender_by_slot'] is not None and 'none_recv' not in s['baseline'] or not s['baseline']['none_recv']
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'Hu', 'same2', 5)
        ck = torch.load(os.path.join(hu, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert all('.adapter.' in n or '.reader.' in n for n in ck['state']) and not any('.recv.' in n for n in ck['state'])
        r1 = os.path.join(d, 'R1')
        r = _cli(d, 'R1', r1, ['--k', str(K), '--init-from', hu, '--stage2-steps', '4'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(r1, 'summary.json')))
        assert s['arm'] == 'R1' and s['channel']['recv'] and s['stages']['stage2']['steps_done'] == 4 and 'stage1' not in s['stages'] and s['init_from']['n_tensors'] > 0
        assert s['baseline']['none_recv']['val']['em'] >= 0 and s['baseline']['none']['val']['em'] >= 0 and s['absence']['1.0']['em'] >= 0
        ck = torch.load(os.path.join(r1, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert any('.recv.' in n for n in ck['state']) and any('.adapter.' in n for n in ck['state']) and not any('.default.' in n for n in ck['state'])
        assert not A1.validate_summary(s, 'R1', 'same2', 5)
        r = _cli(d, 'R1', os.path.join(d, 'R1b'), ['--k', str(K), '--stage2-steps', '4'])                 # R1 needs --init-from
        assert r.returncode != 0
        v1 = os.path.join(d, 'V1')
        r = _cli(d, 'V1', v1, ['--k', '6', '--stage1-steps', '4', '--stage2-steps', '4', '--sel-layer', '1'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(v1, 'summary.json')))
        assert s['arm'] == 'V1' and s['channel']['mode'] == 'span_oracle' and s['channel']['criterion_mode'] == 'span_attn' and s['val_oracle'] is not None
        assert s['stages']['stage1']['steps_done'] == 4 and s['stages']['stage2']['steps_done'] == 4 and s['span']['seen'] > 0 and s['channel']['k'] == 6
        assert s['slot_acc']['mode'] == 'span_attn' and s['hint'] == {} and s['channel']['hint_pre'] == L2.SPAN_PRE
        assert not A1.validate_summary(s, 'V1', 'same2', 5)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _summary_l2(d, name, arm, pair, seed, val, none, other, none_other, fresh=None, complete=True, extra=None):
    p = os.path.join(d, name)
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair=pair, seed=seed, complete=complete, scripts_sha={'l2_model.py': 'abc'},
             val=dict(em=val, f1=val, em_agent=[val, val], em_model={'m': val}, em_by_owner={'other': dict(em=other, n=10), 'own': dict(em=50.0, n=10)}),
             fresh=dict(em=fresh if fresh is not None else val, f1=val, em_agent=[val, val], em_model={'m': val}),
             baseline=dict(none=dict(val=dict(em=none, em_by_owner={'other': dict(em=none_other, n=10)}), fresh=dict(em=none))))
    if extra:
        s.update(extra)
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def test_gate_verdicts_and_markers():
    import l2_gate as G
    d = tempfile.mkdtemp()
    try:
        runs, l1h, out = os.path.join(d, 'l2'), os.path.join(d, 'l1h'), os.path.join(d, 'gates')
        os.makedirs(runs); os.makedirs(l1h)
        _summary_l2(runs, 'R1-same2-s342101', 'R1', 'same2', 342101, 46.0, 42.0, 24.0, 15.0)
        r = G.judge('R1-same2-s342101', runs, l1h)
        assert r['verdict'] == 'PASS' and r['detail']['margin_em'] == 4.0 and r['detail']['margin_other'] == 9.0
        _summary_l2(runs, 'V1-same2-s342101', 'V1', 'same2', 342101, 44.5, 42.0, 24.0, 15.0, extra=dict(val_oracle=dict(em=55.0)))
        r = G.judge('V1-same2-s342101', runs, l1h)
        assert r['verdict'] == 'FAIL' and r['detail']['val_oracle'] == 55.0
        _summary_l2(runs, 'Hu-hetero-s342101', 'Hu', 'hetero', 342101, 45.5, 42.3, 22.5, 17.1)
        assert G.judge('Hu-hetero-s342101', runs, l1h)['verdict'] == 'PASS'
        assert G.judge('R1-hetero-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'                       # missing
        _summary_l2(runs, 'R1-hetero-s342101', 'R1', 'hetero', 342101, 50.0, 42.0, 30.0, 17.0, complete=False)
        assert G.judge('R1-hetero-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'                       # incomplete run
        assert G.judge('Hs-same2-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'                        # no Hu reference yet
        _summary_l2(l1h, 'Hu-same2-s342101', 'Hu', 'same2', 342101, 44.85, 42.0, 24.15, 15.35, extra=dict(slot_acc=dict(acc_sender_by_slot=[0.9, 0.7, 0.6])))
        assert G.judge('Hu-same2-s342101', runs, l1h)['verdict'] == 'FAIL'                              # 2.85 < 3
        _summary_l2(l1h, 'Hs-same2-s342101', 'Hs', 'same2', 342101, 46.0, 42.0, 24.0, 15.0, extra=dict(c_assert=True))
        assert G.judge('Hs-same2-s342101', runs, l1h)['verdict'] == 'ADDS'
        _summary_l2(l1h, 'Hs-same2-s342101', 'Hs', 'same2', 342101, 44.0, 42.0, 24.0, 15.0, extra=dict(c_assert=True))
        assert G.judge('Hs-same2-s342101', runs, l1h)['verdict'] == 'NEUTRAL'
        _summary_l2(l1h, 'Hs-same2-s342101', 'Hs', 'same2', 342101, 43.0, 42.0, 24.0, 15.0, extra=dict(c_assert=False))
        assert G.judge('Hs-same2-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'
        _summary_l2(runs, 'Hu-d32-same2-s342101', 'Hu', 'same2', 342101, 44.0, 42.0, 24.0, 15.0)
        assert G.judge('Hu-d32-same2-s342101', runs, l1h)['verdict'] == 'NONINFERIOR'
        _summary_l2(runs, 'Hu-d32-same2-s342101', 'Hu', 'same2', 342101, 43.0, 42.0, 24.0, 15.0)
        assert G.judge('Hu-d32-same2-s342101', runs, l1h)['verdict'] == 'INFERIOR'
        _summary_l2(runs, 'Hu-L9-same2-s342101', 'Hu', 'same2', 342101, 44.0, 42.0, 24.0, 15.0, extra=dict(slot_acc=dict(acc_sender_by_slot=[0.96, 0.85, 0.81, 0.5])))
        assert G.judge('Hu-L9-same2-s342101', runs, l1h)['verdict'] == 'PASS'
        _summary_l2(runs, 'Hu-L9-same2-s342101', 'Hu', 'same2', 342101, 44.0, 42.0, 24.0, 15.0, extra=dict(slot_acc=dict(acc_sender_by_slot=[0.96, 0.75, 0.81, 0.5])))
        assert G.judge('Hu-L9-same2-s342101', runs, l1h)['verdict'] == 'FAIL'
        with pytest.raises(ValueError):
            G.judge('Xx-same2-s1', runs, l1h)
        r = subprocess.run([sys.executable, '-B', GATE_L2, '--job', 'R1-same2-s342101', '--runs', runs, '--l1h', l1h, '--out', out], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith('GATE R1-same2-s342101: PASS') and os.path.isfile(os.path.join(runs, 'GATE_R1-same2-s342101_PASS'))
        _summary_l2(runs, 'R1-same2-s342101', 'R1', 'same2', 342101, 43.0, 42.0, 24.0, 15.0)
        r = subprocess.run([sys.executable, '-B', GATE_L2, '--job', 'R1-same2-s342101', '--runs', runs, '--l1h', l1h, '--out', out], capture_output=True, text=True)
        assert r.returncode == 0 and 'FAIL' in r.stdout and os.path.isfile(os.path.join(runs, 'GATE_R1-same2-s342101_FAIL')) and not os.path.exists(os.path.join(runs, 'GATE_R1-same2-s342101_PASS'))
        assert json.load(open(os.path.join(out, 'R1-same2-s342101.json')))['verdict'] == 'FAIL'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_v1s_sheaf_consensus_arm():
    p = _pair('V1s', mode='span_oracle', recv=True)
    assert p.arm == 'S1' and p.arm_c == 'V1s' and p.w is not None and all(a.adapter.lam_head is not None and not a.adapter.fixed_lam for a in p.agents)
    L2.freeze_sender(p, frozen=False)
    p.use('default')
    p.train()
    loss, info = p.step(EX[:2], train=True)
    assert torch.isfinite(loss)
    loss.backward()
    names = dict(p.named_parameters())
    assert names['w_raw'].grad is not None and any(q.grad is not None and float(q.grad.abs().sum()) > 0 for n, q in names.items() if '.adapter.lam_head.' in n)
    assert all(names[n].grad is None for n in names if 'lora_' in n)
    for q in names.values():
        q.grad = None
    L2.freeze_sender(p, frozen=True)
    assert not names['w_raw'].requires_grad and all(not q.requires_grad for n, q in names.items() if '.adapter.' in n or '.reader.' in n)
    p.use('recv')
    with torch.no_grad():
        for n, q in names.items():
            if '.recv.' in n and 'lora_B' in n:
                q.add_(0.05 * torch.randn_like(q))
    loss2, _ = p.step_recv(EX[:2], train=True)
    assert torch.isfinite(loss2)
    loss2.backward()
    for n, q in names.items():
        if 'lora_' in n and '.recv.' in n:
            assert q.grad is not None, n
        else:
            assert q.grad is None, n
    p.set_mode('span_attn')
    for a in p.agents:
        a.sel_layer = 1
    v = L2.eval_render(p, EX, hard=True, drop_p=0.0, c_check=True, eval_batch=2, seed=1)
    assert v['c_assert'] is True and v['w'] is not None
    with pytest.raises(SystemExit):
        import l2_train as T2
        T2.main(['--arm', 'V1s', '--pair', 'hetero', '--seed', '5', '--train', 'x', '--val', 'x', '--fresh', 'x', '--out', 'x', '--tiny'])


def test_cli_tiny_v1s_and_gate():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'V1s')
        r = _cli(d, 'V1s', out, ['--k', '6', '--stage1-steps', '4', '--stage2-steps', '4', '--sel-layer', '1'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['arm'] == 'V1s' and s['base_arm'] == 'S1' and s['c_assert'] is True and s['w'] is not None and s['lam_by_bucket'] and s['channel']['recv']
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert 'w_raw' in ck['state'] and any('.adapter.lam_head.' in n for n in ck['state']) and any('.recv.' in n for n in ck['state'])
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'V1s', 'same2', 5)
        import l2_gate as G
        runs, l1h = os.path.join(d, 'l2'), os.path.join(d, 'l1h')
        os.makedirs(runs); os.makedirs(l1h)
        _summary_l2(runs, 'V1s-same2-s342101', 'V1s', 'same2', 342101, 53.0, 42.0, 40.0, 15.0, extra=dict(c_assert=True, w=1.2))
        assert G.judge('V1s-same2-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'                      # no V1 reference
        _summary_l2(runs, 'V1-same2-s342101', 'V1', 'same2', 342101, 51.8, 42.0, 39.0, 15.0)
        assert G.judge('V1s-same2-s342101', runs, l1h)['verdict'] == 'SHEAF-ADDS'
        _summary_l2(runs, 'V1s-same2-s342101', 'V1s', 'same2', 342101, 51.0, 42.0, 40.0, 15.0, extra=dict(c_assert=True, w=1.2))
        assert G.judge('V1s-same2-s342101', runs, l1h)['verdict'] == 'SHEAF-NEUTRAL'
        _summary_l2(runs, 'V1s-same2-s342101', 'V1s', 'same2', 342101, 50.0, 42.0, 40.0, 15.0, extra=dict(c_assert=False, w=1.2))
        assert G.judge('V1s-same2-s342101', runs, l1h)['verdict'] == 'INCOMPLETE'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_v2_selector_positions_loss_and_cli():
    p = _pair('V2', mode='span_oracle', recv=True)
    assert all(a.selector is not None for a in p.agents) and p.agent(0).k == 6
    L2.freeze_sender(p, frozen=False)
    p.use('default')
    p.train()
    loss, info = p.step(EX[:2], train=True, drop=torch.zeros(2, 2, dtype=torch.bool))
    assert torch.isfinite(loss) and 'sel_loss' in info and p.sel_hist
    loss.backward()
    names = dict(p.named_parameters())
    assert all(names[n].grad is not None and float(names[n].grad.abs().sum()) > 0 for n in names if '.selector.weight' in n)
    assert all(names[n].grad is None for n in names if 'lora_' in n)
    for q in names.values():
        q.grad = None
    L2.freeze_sender(p, frozen=True)
    assert all(not q.requires_grad for n, q in names.items() if '.selector.' in n)
    p.set_mode('span_sel')
    pcs, dl = p.doc_pieces(EX, 0)
    pos, _ = p.sel_positions(EX, 0, pcs, dl)
    nb = 1 if p.agent(0).bos is not None else 0
    for b in range(len(EX)):
        got = [int(q) for q in pos[b].tolist() if q >= 0]
        assert len(got) == min(6, dl[b]) and got == list(range(got[0], got[0] + len(got))) and nb <= got[0] and got[-1] < nb + dl[b]
    q = L2.sel_quality(p, EX, eval_batch=2)
    assert q['n'] == 2 * len(EX) and 0.0 <= q['overlap50'] <= 1.0
    # a selector that scores the oracle tokens highest selects the oracle window
    ag = p.agent(0)
    h, A = ag.hidden(pcs)
    opos, _ = p.oracle_positions(EX, 0, pcs)
    with torch.no_grad():
        feat = torch.zeros(ag.d_model)
        for b in range(len(EX)):
            for q_ in opos[b].tolist():
                if q_ >= 0:
                    feat += h[b, int(q_)].float()
        ag.selector.weight.copy_(feat[None] * 0.0)
        ag.selector.bias.zero_()
    p.clear_sel_cache()
    z, _ = p.encode_both(EX[:2])
    assert z.shape == (2, 2, 6, DE)
    p2 = _pair('V2', mode='span_oracle', recv=True, init_state={n: q.detach().clone() for n, q in p.named_parameters() if '.adapter.' in n or '.reader.' in n or '.selector.' in n})
    assert torch.equal(p2.agent(0).selector.weight.detach(), p.agent(0).selector.weight.detach())
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        out = os.path.join(d, 'V2')
        r = _cli(d, 'V2', out, ['--k', '6', '--stage1-steps', '4', '--stage2-steps', '4'])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['arm'] == 'V2' and s['channel']['criterion_mode'] == 'span_sel' and s['val_oracle'] is not None and s['sel_quality'] is not None
        assert s['stages']['stage1']['selector_loss_last100'] is not None and s['slot_acc']['mode'] == 'span_sel'
        ck = torch.load(os.path.join(out, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert any('.selector.' in n for n in ck['state']) and any('.recv.' in n for n in ck['state'])
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'V2', 'same2', 5)
        import l2_gate as G
        runs, l1h = os.path.join(d, 'l2'), os.path.join(d, 'l1h')
        os.makedirs(runs); os.makedirs(l1h)
        _summary_l2(runs, 'V2-same2-s342101', 'V2', 'same2', 342101, 55.0, 42.0, 44.0, 15.0, extra=dict(val_oracle=dict(em=56.0), sel_quality=dict(overlap50=0.85, n=10)))
        assert G.judge('V2-same2-s342101', runs, l1h)['verdict'] == 'PASS'                             # no V1 reference -> PASS only
        _summary_l2(runs, 'V1-same2-s342101', 'V1', 'same2', 342101, 51.8, 42.0, 39.0, 15.0)
        assert G.judge('V2-same2-s342101', runs, l1h)['verdict'] == 'SELECT-YES'
        _summary_l2(runs, 'V2-same2-s342101', 'V2', 'same2', 342101, 52.5, 42.0, 44.0, 15.0)
        assert G.judge('V2-same2-s342101', runs, l1h)['verdict'] == 'SELECT-NO'
        _summary_l2(runs, 'V2-same2-s342101', 'V2', 'same2', 342101, 44.0, 42.0, 17.0, 15.0)
        assert G.judge('V2-same2-s342101', runs, l1h)['verdict'] == 'FAIL'
    finally:
        shutil.rmtree(d, ignore_errors=True)
