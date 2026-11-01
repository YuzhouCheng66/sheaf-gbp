"""Unit tests for l1g_model.py / l1g_pretrain.py / l1g_train.py / l1g_aggregate.py (batch L1g, D8-90): CPU, tiny random models.
What is tested: channel_state holds exactly the adapters' and decoders' parameters (no LoRA, no lam head, no w); load_init loads them strictly
(unknown or mis-shaped tensors refused, LoRA tensors ignored, lam heads / w reported fresh); the span sampler and the sender pieces (prefix cut,
'\\nAnswer:' tail, span as the teacher-forced tail); the pretraining CLI (--tiny) trains both directions, writes trainable.pt + summary with the
held-out blocks, never overwrites, and the stage-F CLI with --init-from loads it (summary.init) and writes an L1g summary; the aggregator's
DICT-NO / TRANSFER-NO / CHANNEL-YES-hetero-pending-s2 verdicts and the --pretrain rule on synthetic summaries."""
import json
import os
import random
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
import l1g_aggregate as AG  # noqa: E402
import l1g_model as MG  # noqa: E402
import l1g_pretrain as PT  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 4, TL.DE
TRAIN_G = os.path.join(HERE, 'l1g_train.py')
PRE_G = os.path.join(HERE, 'l1g_pretrain.py')
AGG_G = os.path.join(HERE, 'l1g_aggregate.py')


def _pair(arm):
    return MG.build_pair(arm, 'hetero', torch.device('cpu'), tiny=M.tiny_models(TL.EXAMPLES, 5), k=K, d_e=DE, n_heads=2)


def test_channel_state_and_load_init():
    pu = _pair('Gu')
    st = MG.channel_state(pu)
    assert st and all(('.adapter.' in n or '.reader.' in n) and 'lora_' not in n and 'lam_head' not in n for n in st)
    assert set(st) == {n for n, p in pu.named_parameters() if p.requires_grad}                   # Gu: no lam head, no w
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, 'trainable.pt')
        state = {n: torch.randn_like(t) for n, t in st.items()}
        torch.save(dict(state=state, summary=dict(steps_done=3)), path)
        ps = _pair('Gs')
        info = MG.load_init(ps, path)
        names = dict(ps.named_parameters())
        assert info['n_loaded'] == len(state) and all(torch.equal(names[n].detach(), t) for n, t in state.items())
        assert any('lam_head' in n for n in info['not_in_file']) and 'w_raw' in info['not_in_file'] and info['pretrain_summary'] == dict(steps_done=3)
        torch.save(dict(state=dict(state, **{'agents.0.adapter.queries': torch.zeros(1, 1)})), path)
        with pytest.raises(ValueError):
            MG.load_init(_pair('Gu'), path)
        torch.save(dict(state=dict(state, bogus=torch.zeros(1))), path)
        with pytest.raises(KeyError):
            MG.load_init(_pair('Gu'), path)
        lora_name = next(n for n, _ in pu.named_parameters() if 'lora_A' in n)
        torch.save(dict(state=dict(state, **{lora_name: names[lora_name].detach().clone()})), path)
        info = MG.load_init(_pair('Gu'), path)
        assert info['ignored'] == [lora_name]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_spans_and_sender_pieces():
    rng = random.Random(0)
    kinds = set()
    for _ in range(40):
        text, span, kind = PT.sample_span(rng, TL.EXAMPLES[0], 0.5, 3)
        kinds.add(kind)
        assert span and (kind == 'title' or 1 <= len(span.split()) <= 3) and text
    assert kinds == {'title', 'window'}
    spans = PT.make_spans(TL.EXAMPLES, 10, 1, 0.5, 6)
    assert len(spans) == 10 and PT.make_spans(TL.EXAMPLES, 10, 1, 0.5, 6) == spans
    pu = _pair('Gu')
    ag = pu.agent(0)
    pre, tail, ans = PT.sender_pieces(ag, 'alice wrote the red book .', 'red book', 5)
    assert ans == [] and tail == [int(t) for t in ag.ids(' red book')] and pre[0] == ag.bos
    assert pre[1:] == ag.ids('Documents:\nalice wrote the red book .')[:5] + ag.ids(MG.SPAN_PREFIX_TAIL)
    tgt, lens = PT.targets(ag, [('x', 'red book', 'window'), ('y', 'alice', 'title')], K)
    assert tgt.shape == (2, K) and lens == [2, 1] and tgt[1, 1] == ag.filler
    pu.train()
    loss, info = PT.step_loss(pu, spans[:2], type('A', (), dict(prefix_max_tokens=32, k=K))())
    assert torch.isfinite(loss) and len(info['loss_agent']) == 2
    loss.backward()
    assert all(p.grad is None for n, p in pu.named_parameters() if 'lora_' in n)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in pu.named_parameters() if '.adapter.' in n)


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in TL.EXAMPLES:
                fh.write(json.dumps(ex) + '\n')


def test_cli_pretrain_then_train_with_init():
    d = tempfile.mkdtemp()
    try:
        _fixture(d)
        pre = os.path.join(d, 'pretrain')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        cmd = [sys.executable, '-B', PRE_G, '--tiny', '--seed', '5', '--train', os.path.join(d, 'train.jsonl'), '--val', os.path.join(d, 'val.jsonl'), '--out', pre,
               '--steps', '6', '--min-steps', '2', '--batch', '2', '--warmup', '2', '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--holdout', '6', '--eval-batch', '3',
               '--eval-every', '3', '--log-every', '2', '--prefix-max-tokens', '32']
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.load(open(os.path.join(pre, 'summary.json')))
        assert s['batch_name' if 'batch_name' in s else 'batch'] == 'L1g' and s['stage'] == 'P' and s['complete'] is True and s['steps'] == 6
        assert len(s['heldout']) == 2 and all({'slot0_acc', 'msg_slot_acc', 'decoded_em', 'examples'} <= set(v) for v in s['heldout'].values())
        assert set(s['scripts_sha']) == set(MG.SCRIPTS)
        ck = torch.load(os.path.join(pre, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert ck['stage'] == 'P' and all('lora_' not in n and 'lam_head' not in n for n in ck['state'])
        r2 = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r2.returncode != 0 and 'never overwrite' in (r2.stdout + r2.stderr)
        out = os.path.join(d, 'Gs')
        cmd2 = [sys.executable, '-B', TRAIN_G, '--tiny', '--arm', 'Gs', '--pair', 'hetero', '--seed', '5', '--init-from', pre, '--train', os.path.join(d, 'train.jsonl'),
                '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--out', out, '--steps', '4', '--batch', '2', '--warmup', '2',
                '--k', str(K), '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2']
        r3 = subprocess.run(cmd2, capture_output=True, text=True, env=env)
        assert r3.returncode == 0, r3.stdout[-2000:] + r3.stderr[-2000:]
        s3 = json.load(open(os.path.join(out, 'summary.json')))
        assert s3['batch_name'] == 'L1g' and s3['prereg'] == 'D8-90' and s3['arm'] == 'Gs' and s3['init']['n_loaded'] == len(ck['state']) and s3['c_assert'] is True
        assert s3['channel']['init'].startswith('stage-P') and set(s3['scripts_sha']) == set(MG.SCRIPTS)
        import l1_aggregate as A1
        assert not A1.validate_summary(s3, 'Gs', 'hetero', 5)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _summary(d, arm, seed, val, fresh, other=None, none=None, none_other=None):
    p = os.path.join(d, f'{arm}-hetero-s{seed}')
    os.makedirs(p, exist_ok=True)
    s = dict(arm=arm, pair='hetero', seed=seed, val=dict(em=val, f1=val, em_by_owner={'other': dict(em=other if other is not None else val, n=10)}),
             fresh=dict(em=fresh, f1=fresh), complete=True, scripts_sha={'l1g_model.py': 'abc'})
    if arm in ('Gu', 'Gs'):
        s['baseline'] = dict(none=dict(val=dict(em=none if none is not None else 40.0, em_by_owner={'other': dict(em=none_other if none_other is not None else 18.0, n=10)}),
                                       fresh=dict(em=40.0)), gold_hint=dict(val=dict(em=52.0, em_by_owner={'other': dict(em=41.0, n=10)})))
        s['soft'] = dict(val=dict(em=val - 1.0))
        s['hint'] = dict(val=dict(hint_by_receiver_bucket={'0': {'other': dict(em_gold=50.0, n=5)}}))
        s['channel'] = dict(rendering='hard', receiver_training='none (frozen B0 checkpoint)')
    if arm == 'Gs':
        s['c_assert'] = True
    json.dump(s, open(os.path.join(p, 'summary.json'), 'w'))


def _pretrain(d, slot0, complete=True):
    p = os.path.join(d, 'pretrain-s342101')
    os.makedirs(p, exist_ok=True)
    s = dict(batch='L1g', stage='P', complete=complete, heldout={'a -> b': dict(slot0_acc=slot0, msg_slot_acc=slot0, decoded_em=slot0 / 2), 'b -> a': dict(slot0_acc=slot0 + 0.1, msg_slot_acc=slot0, decoded_em=slot0 / 2)})
    path = os.path.join(p, 'summary.json')
    json.dump(s, open(path, 'w'))
    return path


def test_aggregator_verdicts():
    d = tempfile.mkdtemp()
    try:
        l1, l1g = os.path.join(d, 'l1'), os.path.join(d, 'l1g')
        os.makedirs(l1); os.makedirs(l1g)
        for s in AG.SEEDS:
            _summary(l1, 'B0', s, 42.0, 41.0)
            _summary(l1, 'T', s, 58.0, 56.0)
        _summary(l1g, 'Gu', AG.SEEDS[0], 42.9, 42.0, other=30.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1g)
        assert a['verdict'] == 'TRANSFER-NO' and a['g1_lines'][0] == f'G1g(s{AG.SEEDS[0]}): FAIL'
        pre = _pretrain(l1g, 0.2)
        a = AG.aggregate(l1, l1g, pretrain=pre)
        assert a['verdict'] == 'DICT-NO' and a['pretrain']['slot0_min'] == 0.2
        pre = _pretrain(l1g, 0.5)
        a = AG.aggregate(l1, l1g, pretrain=pre)
        assert a['verdict'] == 'TRANSFER-NO'
        shutil.rmtree(os.path.join(l1g, f'Gu-hetero-s{AG.SEEDS[0]}'))
        _summary(l1g, 'Gu', AG.SEEDS[0], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1g, pretrain=pre)
        assert a['verdict'] == 'CHANNEL-YES-hetero-pending-s2'
        _summary(l1g, 'Gu', AG.SEEDS[1], 46.0, 45.0, other=25.0, none=40.0, none_other=18.0)
        a = AG.aggregate(l1, l1g, pretrain=pre)
        assert a['verdict'] == 'CHANNEL-YES-hetero / SHEAF-pending'
        r = subprocess.run([sys.executable, '-B', AGG_G, '--l1', l1, '--l1g', l1g, '--pretrain', pre, '--gate', str(AG.SEEDS[0]), '--out', os.path.join(d, 'gate')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith(f'G1g(s{AG.SEEDS[0]}): PASS') and 'VERDICT' not in r.stdout
        a = AG.aggregate(l1, l1g, pretrain=os.path.join(d, 'nope.json'))
        assert a['verdict'].startswith('CHANNEL-YES-hetero') and any('not usable' in x for x in a['reasons'])
    finally:
        shutil.rmtree(d, ignore_errors=True)
