"""test_l7_nagents.py (Claude, 2026-11-04): CPU tests of ladder L7 (D8-97): the N-sharding, the pure readouts, and the five CLI jobs + the gate
end to end on tiny models with N = 3 agents.  python -B -m pytest scripts/claude_v342/test_l7_nagents.py -q -p no:cacheprovider"""
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import l1_data as D  # noqa: E402
import l7_data as L7D  # noqa: E402
import l7_gate as G  # noqa: E402
import l7_nagents as L7  # noqa: E402

RUNNER = os.path.join(HERE, 'l7_nagents.py')
GATE = os.path.join(HERE, 'l7_gate.py')
WORDS = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon'.split()


def raw_example(k, rng):
    """a fake HotpotQA example: 10 paragraphs (title = the word, two sentences), supporting facts on paragraphs 0 and 1 of the context"""
    words = list(WORDS)
    rng.shuffle(words)
    ctx = [[f'{w} title', [f'{w} is a {words[(i + 1) % 20]} fact .', f'the answer here is {w} .']] for i, w in enumerate(words[:10])]
    sup = [[ctx[0][0], 1], [ctx[1][0], 0]]
    return dict(_id=f'ex{k}', question=f'what is the {words[0]} ?', answer=words[0], type='bridge', level='easy', supporting_facts=sup, context=ctx)


def rows_for(n, n_agents, seed=5):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        row = D.shard_example(raw_example(k, rng), rng)
        assert row is not None
        out.append(L7D.reshard_row(row, n_agents, rng))
    return out


def test_data():
    rng = random.Random(1)
    raws = [raw_example(k, rng) for k in range(6)]
    for n_agents in (2, 3, 4, 5):
        for ex in raws:
            row = D.shard_example(ex, random.Random(2))
            r = L7D.reshard_row(row, n_agents, random.Random(3))
            assert len(r['shards']) == n_agents and sum(len(s) for s in r['shards']) == 10 and r['n_agents'] == n_agents
            assert sorted(r['shard_sizes']) == sorted([10 // n_agents + (1 if j < 10 % n_agents else 0) for j in range(n_agents)])
            g0, g1 = r['gold_shards']
            assert g0 != g1 and {p[0] for p in r['shards'][g0]} >= {r['gold_title'][0]} and r['gold_title'][1] in {p[0] for p in r['shards'][g1]}
            titles = [p[0] for sh in r['shards'] for p in sh]
            assert sorted(titles) == sorted(p[0] for p in r['all_paragraphs'])
            for i in range(n_agents):
                assert (r['sup'][i] != []) == (i in (g0, g1))
                assert D.text_sender_target(r, i) == (' ' + ' '.join(r['sup'][i]) if r['sup'][i] else ' (none)')
            assert r['answer_owner'] in (g0, g1, None)
            # the answer is in both gold paragraphs' text? then owner None; the fake answer word sits in gold 0's second sentence only
            assert L7.informed_of(r) == sorted({g0, g1})
    # deterministic
    row = D.shard_example(raws[0], random.Random(2))
    assert L7D.reshard_row(row, 4, random.Random(9)) == L7D.reshard_row(row, 4, random.Random(9))
    with tempfile.TemporaryDirectory() as td:
        tj = os.path.join(td, 'train.json')
        with open(tj, 'w', encoding='utf-8') as fh:
            json.dump(raws, fh)
        ex_dir = os.path.join(td, 'splits_l1')
        os.makedirs(ex_dir)
        base = [D.shard_example(ex, random.Random(4)) for ex in raws]
        D.write_jsonl(os.path.join(ex_dir, 'train.jsonl'), base[:2])                 # ids ex0, ex1 are excluded
        D.write_jsonl(os.path.join(ex_dir, 'val.jsonl'), base[4:5])
        D.write_jsonl(os.path.join(ex_dir, 'fresh.jsonl'), base[5:6])
        out = os.path.join(td, 'splits_l7')
        meta = L7D.make_splits7(tj, ex_dir, os.path.join(ex_dir, 'val.jsonl'), os.path.join(ex_dir, 'fresh.jsonl'), out, seed=7, n_train=2, n_list=(2, 4))
        assert meta['counts']['excluded_ids'] == 4 and meta['counts']['unseen_pool'] == 2 and meta['counts']['written_train'] == 2
        tr2 = D.read_jsonl(os.path.join(out, 'train7_N2.jsonl'))
        tr4 = D.read_jsonl(os.path.join(out, 'train7_N4.jsonl'))
        assert len(tr2) == 2 and len(tr4) == 2 and [r['id'] for r in tr2] == [r['id'] for r in tr4]
        assert not {r['id'] for r in tr2} & {'ex0', 'ex1', 'ex4', 'ex5'}
        assert all(len(r['shards']) == 2 for r in tr2) and all(len(r['shards']) == 4 for r in tr4)
        v4 = D.read_jsonl(os.path.join(out, 'val_N4.jsonl'))
        assert len(v4) == 1 and v4[0]['id'] == 'ex4' and len(v4[0]['shards']) == 4
        assert not os.path.exists(os.path.join(out, 'val_N2.jsonl'))
        assert set(meta['files']) == {'train7_N2.jsonl', 'train7_N4.jsonl', 'val_N4.jsonl', 'fresh_N4.jsonl'}


def test_pure():
    assert L7.auroc([0.1, 0.9, 0.5, 0.7], [0, 1, 0, 1]) == 1.0 and L7.auroc([1, 1], [1, 1]) is None and abs(L7.auroc([0.2, 0.2, 0.8], [1, 0, 1]) - 0.75) < 1e-9
    assert L7.majority(['a', 'b', 'a']) == 0 and L7.majority(['a', 'b', 'c']) == 0 and L7.majority(['x', 'y', 'y']) == 1
    assert L7.argmax_first([1.0, 3.0, 3.0]) == 1 and L7.sigmoid(0.0) == 0.5
    rng = np.random.RandomState(0)
    row = dict(union=['paris', 'rome', 'oslo'], gold='Paris')
    for _ in range(20):
        pairs = L7.pick_pairs(row, 3, rng)
        assert len(pairs) == 2 and {p[2] for p in pairs} == {0.0, 1.0} and all(0 <= p[0] < 3 for p in pairs)
    pairs = L7.pick_pairs(dict(union=['rome', 'oslo'], gold='Paris'), 2, rng)
    assert [p[2] for p in pairs] == [0.0, 0.0]
    ex = dict(answer='Paris', shards=[[], [], []], gold_shards=[0, 2], answer_owner=0, id='e')
    cands = [['Paris'], ['Rome'], ['Rome']]
    C = L7.union(cands)
    assert C == ['Paris', 'Rome']
    S = [[3.0, -1.0], [-0.5, 0.2], [0.1, 0.1]]                                   # sum: Paris 2.6, Rome -0.7
    S0 = [2.0, -2.0]                                                              # prior strongly likes Paris: prior-subtracted (N-1=2): Paris -1.4, Rome 3.3
    j = L7.judge(ex, cands, C, S, S0, [0.9, 0.5, 0.4], [['Paris'], ['Oslo'], ['Rome']], ['Paris', 'Oslo', 'Rome'], [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.0, 0.3]])
    assert j['ok']['ver_sum'] == 1.0 and j['ok']['vote'] == 0.0 and j['ok']['select_p'] == 1.0 and j['ok']['ver_prior'] == 0.0 and j['ok']['ver_informed'] == 1.0
    assert j['ok']['agent0'] == 1.0 and j['ok']['agent1'] == 0.0 and j['either'] == 1.0 and j['informed_either'] == 1.0 and j['agree'] == 0.0 and j['know_case']
    assert j['ok']['ver_owner'] == 1.0 and j['own']['ver_sum'] == 1.0 and j['own']['vote'] == 1.0 and j['own']['any_union'] == 1.0
    rels = {(c['agent'], c['rel']) for c in j['cand_recs']}
    assert rels == {(0, 'informed'), (1, 'ignorant'), (2, 'informed')}
    j['bytes'] = L7.belief_bytes(3, 32, cands, C)
    assert j['bytes']['span_bytes'] == 192 and j['bytes']['score_bytes_int8'] == 6 and j['bytes']['wire_bytes'] == 2 * j['bytes']['message_bytes']
    agg = L7.aggregate([j, j], 3)
    assert agg['em'] == 100.0 and agg['em_variants']['vote'] == 0.0 and agg['round1']['em_agent'] == [100.0, 0.0, 0.0] and agg['know']['n'] == 2 and agg['know']['know_wins'] == 100.0
    assert agg['verifier']['by_rel']['ignorant']['n'] == 4 and agg['verifier']['flat_when_ignorant'] is True and agg['own']['ver_sum'] == 100.0 and agg['bytes']['span_bytes'] == 192.0
    assert G.labels(2.0, 4) == 'BELIEF-BEATS-TEXT-N4' and G.labels(-2.0, 4) == 'TEXT-BEATS-BELIEF-N4' and G.labels(1.9, 2) == 'TIE-N2'


def run(args, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-B'] + args, capture_output=True, text=True, env=env, cwd=cwd)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-4000:]
    return r


def test_cli_tiny():
    N = 3
    with tempfile.TemporaryDirectory() as td:
        rows = rows_for(6, N)
        tr, va, fr = rows[:4], rows[4:5], rows[5:6]
        for r in va:
            r['id'] += 'v'
        for r in fr:
            r['id'] += 'f'
        paths = {}
        for name, rs in (('train', tr), ('val', va), ('fresh', fr)):
            paths[name] = os.path.join(td, f'{name}.jsonl')
            D.write_jsonl(paths[name], rs)
        common = ['--n-agents', str(N), '--seed', '5', '--train', paths['train'], '--val', paths['val'], '--fresh', paths['fresh'], '--tiny', '--keep-preds', '--allow-fallback-data',
                  '--eval-batch', '2', '--gen-batch', '2', '--score-batch', '4', '--warmup', '1', '--log-every', '1', '--max-ctx-tokens', '400', '--k-ev', '4', '--budget', '8',
                  '--max-new-tokens', '3', '--sel-layer', '1']
        runs = os.path.join(td, 'runs')
        jobs = {}
        # 1 the text sender
        jobs['TSEND'] = os.path.join(runs, f'TSEND-N{N}-s5')
        run([RUNNER, '--job', 'tsend', '--out', jobs['TSEND'], '--steps', '2', '--batch', '2', '--stage2-examples', '2'] + common, td)
        s = json.load(open(os.path.join(jobs['TSEND'], 'summary.json')))
        mj = json.load(open(os.path.join(jobs['TSEND'], 'messages.json')))
        assert s['job'] == 'tsend' and s['complete'] and s['train_complete'] and s['n_agents'] == N and s['prereg'] == 'D8-97'
        assert len(mj['messages']['val']['texts']) == N and len(mj['messages']['val']['texts'][0]) == 1 and len(mj['subset']['idx']) == 2
        # 2 the text receiver
        jobs['TRECV'] = os.path.join(runs, f'TRECV-N{N}-s5')
        run([RUNNER, '--job', 'trecv', '--out', jobs['TRECV'], '--tsend-from', jobs['TSEND'], '--steps', '2', '--stage2-steps', '1', '--batch', '2', '--stage2-examples', '2'] + common, td)
        s = json.load(open(os.path.join(jobs['TRECV'], 'summary.json')))
        assert s['train_complete'] and len(s['val']['em_agent']) == N and 'total_s_per_example' in s['val']['timing'] and s['val']['bytes']['wire_bytes'] == s['val']['bytes']['message_bytes'] * (N - 1)
        assert os.path.isfile(os.path.join(jobs['TRECV'], 'preds_val.json'))
        # 3 the belief reader
        jobs['RECV'] = os.path.join(runs, f'RECV-N{N}-s5')
        run([RUNNER, '--job', 'recv', '--out', jobs['RECV'], '--steps', '2', '--batch', '2'] + common, td)
        s = json.load(open(os.path.join(jobs['RECV'], 'summary.json')))
        assert s['train_complete'] and len(s['val']['em_agent']) == N
        ck = __import__('torch').load(os.path.join(jobs['RECV'], 'trainable.pt'), map_location='cpu', weights_only=False)
        assert ck['role'] == 'recv' and ck['n_agents'] == N and all('.recv.' in k for k in ck['state'])
        # 4 the candidates of the training rows
        jobs['GEN'] = os.path.join(runs, f'GEN-N{N}-s5')
        run([RUNNER, '--job', 'gen', '--out', jobs['GEN'], '--recv-from', jobs['RECV']] + common, td)
        cj = json.load(open(os.path.join(jobs['GEN'], 'cands_train.json')))
        assert cj['complete'] and cj['n'] == 4 and all(len(r['cands']) == N and len(r['wins']) == N and all(len(w) <= 4 for w in r['wins']) for r in cj['rows'])
        # 5 the verifier + the evaluation
        jobs['VER'] = os.path.join(runs, f'VER-N{N}-s5')
        run([RUNNER, '--job', 'ver', '--out', jobs['VER'], '--recv-from', jobs['RECV'], '--cands-from', os.path.join(jobs['GEN'], 'cands_train.json'), '--steps', '2', '--batch', '2'] + common, td)
        s = json.load(open(os.path.join(jobs['VER'], 'summary.json')))
        v = s['val']
        assert s['train_complete'] and s['train']['sequences'] == 8 and v['n'] == 1 and v['n_agents'] == N
        for k in ('ver_sum', 'ver_select', 'ver_prior', 'ver_prior_half', 'ver_informed', 'ver_owner', 'select_p', 'vote', 'agent0', 'agent2', 'ver_self1'):
            assert k in v['em_variants']
        assert set(v['verifier']['by_rel']) == {'informed', 'ignorant', 'all'} and 'total_s_per_example' in v['timing'] and v['bytes']['span_bytes'] == N * 4 * 2
        assert 'own' in v and len(v['own']['em_agent']) == N
        pv = json.load(open(os.path.join(jobs['VER'], 'preds_val.json')))
        assert len(pv) == 1 and len(pv[0]['ver_scores']) == N and len(pv[0]['prior_scores']) == len(pv[0]['union'])
        # 6 the gates
        for kind, want in (('TSEND', 'TSEND-DONE'), ('TRECV', 'TRECV-DONE'), ('RECV', 'RECV-DONE'), ('GEN', 'GEN-DONE')):
            r = run([GATE, '--job', f'{kind}-N{N}-s5', '--runs', runs, '--out', os.path.join(runs, 'gates')], td)
            assert f'GATE {kind}-N{N}-s5: {want}' in r.stdout, r.stdout
        r = run([GATE, '--job', f'VER-N{N}-s5', '--runs', runs, '--out', os.path.join(runs, 'gates')], td)
        line = [l for l in r.stdout.splitlines() if l.startswith('GATE')][0]
        assert f'-N{N}+' in line and ('FASTER' in line or 'SLOWER' in line) and 'pendingN2' in line and ('IGNORANT-FLAT' in line or 'IGNORANT-NOT-FLAT' in line), line
        gj = json.load(open(os.path.join(runs, 'gates', f'VER-N{N}-s5.json')))
        assert gj['detail']['B'] == v['em'] and gj['detail']['T'] is not None and any(f.startswith(f'GATE_VER-N{N}-s5_') for f in os.listdir(runs))
        # 7 synthetic trend: fake N2 summaries of the same seed
        for kind, em_, extra in (('VER', 50.0, dict(train_complete=True)), ('TRECV', 48.0, {})):
            d = os.path.join(runs, f'{kind}-N2-s5')
            os.makedirs(d)
            json.dump(dict(job=kind.lower(), n_agents=2, seed=5, complete=True, val=dict(em=em_, timing=dict(total_s_per_example=1.0)), fresh=dict(em=em_), **extra), open(os.path.join(d, 'summary.json'), 'w'))
        r = run([GATE, '--job', f'VER-N{N}-s5', '--runs', runs, '--out', os.path.join(runs, 'gates')], td)
        line = [l for l in r.stdout.splitlines() if l.startswith('GATE')][0]
        assert 'pendingN2' not in line and ('SCALES' in line or 'SAME-TREND' in line), line
        # 8 never overwrite
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        r = subprocess.run([sys.executable, '-B', RUNNER, '--job', 'gen', '--out', jobs['GEN'], '--recv-from', jobs['RECV']] + common, capture_output=True, text=True, env=env, cwd=td)
        assert r.returncode != 0 and 'never overwrite' in (r.stdout + r.stderr)
