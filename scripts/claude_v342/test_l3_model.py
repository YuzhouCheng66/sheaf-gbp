"""Unit tests for l3_model.py / l3_train.py / l3_gate.py (ladder L3, D8-93): CPU, tiny random models.
What is tested: the round-1 cache (evidence ids, answers, probabilities; save / load), the aligned read of the round-1 answers (recv adapter),
the consensus + Decoder2, stage 3 trains the L3 modules only, stage 4 trains the recv LoRA only (S2b and T2 prompts), eval_round2 readouts,
build_l3 from a V2 checkpoint, the CLI chain V2 -> S2a -> S2b / T2, and the gate verdicts."""
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
import l2_model as L2  # noqa: E402
import l3_model as L3  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 6, TL.DE
EX = TL.EXAMPLES
TRAIN_L2 = os.path.join(HERE, 'l2_train.py')
TRAIN_L3 = os.path.join(HERE, 'l3_train.py')
GATE_L3 = os.path.join(HERE, 'l3_gate.py')


def _v2():
    p = L2.build_pair('V2', 'same2', torch.device('cpu'), tiny=M.tiny_models(EX, 5), k=K, d_e=DE, n_heads=2, mode='span_oracle', span_k=K, recv=True)
    p.set_mode('span_sel')
    return p


def test_round1_cache_beliefs_consensus_and_stages():
    pl3 = L3.PairL3(_v2(), k_ans=4)
    recs = pl3.round1(EX)
    assert len(recs) == len(EX) and all(set(r) == {'0', '1'} for r in recs) and pl3.n_round1 == len(EX) and len(pl3.cache) == len(EX)
    r = recs[0]['0']
    assert len(r['ev']) == K and isinstance(r['ans'], str) and isinstance(r['ans_ids'], list) and 0.0 <= r['p'] <= 1.0
    pl3.round1(EX)
    assert pl3.n_round1 == len(EX)                                              # cached
    pre, suf = pl3.pieces_r1(EX[:1], 0)[0]
    ag = pl3.agent(0)
    hp = len(ag.hint_post)                                                     # the tiny tokenizer drops the newline: hint_post may be empty
    assert (hp == 0 or pre[-hp:] == ag.hint_post) and pre[len(pre) - hp - K:len(pre) - hp] == r['ev']
    z2, lam2 = pl3.beliefs(EX[:2])
    assert z2.shape == (2, 2, 4, DE) and lam2.shape == (2, 2, 4, DE) and bool((lam2 > 0).all())
    x2 = pl3.consensus(z2, lam2)
    assert x2.shape == z2.shape and torch.isfinite(x2).all()
    ids = pl3.consensus_ids(x2)
    assert ids[0].shape == (2, 4)
    L3.trainable_sets(pl3, 3)
    names = dict(pl3.named_parameters())
    trainable = {n for n, p in names.items() if p.requires_grad}
    assert trainable and all(n.startswith('ans_senders.') or n.startswith('dec2.') or n == 'w2_raw' for n in trainable)
    pl3.train()
    loss, info = pl3.step3(EX[:2])
    assert torch.isfinite(loss) and 'msg_acc' in info and 'w2' in info
    loss.backward()
    assert names['w2_raw'].grad is not None and any(names[n].grad is not None and float(names[n].grad.abs().sum()) > 0 for n in trainable if 'ans_senders' in n)
    assert all(names[n].grad is None for n in names if n not in trainable)
    for p_ in names.values():
        p_.grad = None
    L3.trainable_sets(pl3, 4)
    trainable4 = {n for n, p in names.items() if p.requires_grad}
    assert trainable4 and all('lora_' in n and '.recv.' in n for n in trainable4)
    with torch.no_grad():
        for n in trainable4:
            if 'lora_B' in n:
                names[n].add_(0.05 * torch.randn_like(names[n]))
    for arm in ('S2b', 'T2'):
        loss4, _ = pl3.step4(EX[:2], train=True, arm=arm)
        assert torch.isfinite(loss4)
        loss4.backward()
        assert all(names[n].grad is not None for n in trainable4) and all(names[n].grad is None for n in names if n not in trainable4)
        for p_ in names.values():
            p_.grad = None
    pcs = pl3.pieces_r2(EX[:1], 0, 'answer')
    assert pcs[0][0][-len(pl3.hint2_pre[0]):] == pl3.hint2_pre[0] and pcs[0][1][:len(pl3.hint2_post[0])] == pl3.hint2_post[0] and pcs[0][2][-1] == ag.eos
    t2 = pl3.pieces_t2(EX[:1], 0, None)
    assert t2[0][2] == [] and len(t2[0][0]) > len(pre)
    for arm in ('S2b', 'T2'):
        v = pl3.eval_round2(EX, arm, eval_batch=2)
        assert 0.0 <= v['em'] <= 100.0 and v['n_records'] == 2 * len(EX) and len(v['round1']['em_agent']) == 2 and 0.0 <= v['round1']['either_correct'] <= 100.0
        assert ('consensus' in v) == (arm == 'S2b')
    d = tempfile.mkdtemp()
    try:
        pl3.save_cache(os.path.join(d, 'c.json'))
        pl4 = L3.PairL3(_v2(), k_ans=4)
        assert pl4.load_cache(os.path.join(d, 'c.json')) == len(EX) and pl4.round1(EX)[1]['1']['ans'] == recs[1]['1']['ans'] and pl4.n_round1 == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert L3.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0 and L3.auroc([0.5, 0.5], [1, 0]) == 0.5 and L3.auroc([0.1, 0.9], [1, 1]) is None


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in EX:
                fh.write(json.dumps(ex) + '\n')


def _data(d):
    return ['--train', os.path.join(d, 'train.jsonl'), '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'), '--batch', '2', '--warmup', '2',
            '--d-e', str(DE), '--n-heads', '2', '--eval-batch', '2', '--log-every', '2', '--tiny', '--pair', 'same2', '--seed', '5']


def test_cli_v2_then_s2a_s2b_t2_and_gate():
    d = tempfile.mkdtemp()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        _fixture(d)
        v2 = os.path.join(d, 'V2-same2-s5')
        r = subprocess.run([sys.executable, '-B', TRAIN_L2, '--arm', 'V2', '--k', str(K), '--stage1-steps', '4', '--stage2-steps', '4', '--out', v2] + _data(d), capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        s2a = os.path.join(d, 'S2a-same2-s5')
        r = subprocess.run([sys.executable, '-B', TRAIN_L3, '--arm', 'S2a', '--v2-from', v2, '--stage3-steps', '4', '--out', s2a] + _data(d), capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(s2a, 'summary.json')))
        assert s['arm'] == 'S2a' and s['batch_name'] == 'L3' and s['stages']['stage3']['steps_done'] == 4 and s['complete'] and os.path.isfile(os.path.join(s2a, 'round1_cache.json'))
        assert 'consensus' in s['val'] and s['val']['round1']['either_correct'] >= 0 and s['round1_cache']['total'] == len(EX)
        ck = torch.load(os.path.join(s2a, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert any(n.startswith('ans_senders.') for n in ck['state']) and 'w2_raw' in ck['state'] and any('.recv.' in n for n in ck['state'])
        import l1_aggregate as A1
        assert not A1.validate_summary(s, 'S2a', 'same2', 5)
        for arm in ('S2b', 'T2'):
            out = os.path.join(d, f'{arm}-same2-s5')
            r = subprocess.run([sys.executable, '-B', TRAIN_L3, '--arm', arm, '--v2-from', v2, '--init-from', s2a, '--stage4-steps', '4', '--out', out] + _data(d), capture_output=True, text=True, env=env)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
            s = json.load(open(os.path.join(out, 'summary.json')))
            assert s['arm'] == arm and s['stages']['stage4']['steps_done'] == 4 and s['round1_cache']['loaded'] == len(EX) and s['round1_cache']['generated'] == 0
            assert ('consensus' in s['val']) == (arm == 'S2b') and not A1.validate_summary(s, arm, 'same2', 5)
        r = subprocess.run([sys.executable, '-B', TRAIN_L3, '--arm', 'S2b', '--v2-from', v2, '--stage4-steps', '4', '--out', os.path.join(d, 'x')] + _data(d), capture_output=True, text=True, env=env)
        assert r.returncode != 0                                                 # S2b needs --init-from
        # gate
        import l3_gate as G
        runs, l2 = os.path.join(d, 'l3'), os.path.join(d, 'l2')
        os.makedirs(runs); os.makedirs(l2)
        shutil.copytree(s2a, os.path.join(runs, 'S2a-same2-s5'))
        assert G.judge('S2a-same2-s5', runs, l2)['verdict'] == 'DONE'
        shutil.copytree(os.path.join(d, 'S2b-same2-s5'), os.path.join(runs, 'S2b-same2-s5'))
        assert G.judge('S2b-same2-s5', runs, l2)['verdict'] == 'INCOMPLETE'      # no V2 reference
        shutil.copytree(v2, os.path.join(l2, 'V2-same2-s5'))
        v = G.judge('S2b-same2-s5', runs, l2)['verdict']
        assert v.startswith('CONSENSUS-') and v.endswith('+pendingT2')
        shutil.copytree(os.path.join(d, 'T2-same2-s5'), os.path.join(runs, 'T2-same2-s5'))
        v = G.judge('S2b-same2-s5', runs, l2)['verdict']
        assert v.startswith('CONSENSUS-') and v.split('+')[1] in ('LATENT-BEATS-TEXT-R2', 'TEXT-BEATS-LATENT-R2', 'TIE')
        assert G.judge('T2-same2-s5', runs, l2)['verdict'] == 'DONE'
        r = subprocess.run([sys.executable, '-B', GATE_L3, '--job', 'S2b-same2-s5', '--runs', runs, '--l2', l2, '--out', os.path.join(d, 'gates')], capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.startswith('GATE S2b-same2-s5: CONSENSUS-') and any(f.startswith('GATE_S2b-same2-s5_') for f in os.listdir(runs))
        with pytest.raises(ValueError):
            G.judge('V2-same2-s5', runs, l2)
    finally:
        shutil.rmtree(d, ignore_errors=True)
