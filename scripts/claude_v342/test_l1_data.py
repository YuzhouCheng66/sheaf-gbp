"""Unit tests for l1_data.py (batch L1, D8-82; design sections 2, 4, 5) and run_l1_data.sh.
CPU only, no network, no raw HotpotQA: a synthetic HotpotQA-format fixture is written to tmp_path (never to the repo).
Run: python -m pytest -q scripts/claude_v342/test_l1_data.py"""
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l1_data as D  # noqa: E402

RUN_SH = os.path.join(HERE, 'run_l1_data.sh')
SMALL = dict(n_train=20, n_val=4, n_fresh=3)


# ---------------------------------------------------------------- synthetic HotpotQA fixture
def _example(split, i, kind='bridge', n_para=10, sup_mode='two', answer=None, answer_in=(True, False)):
    """One HotpotQA-format example; gold paragraphs at context positions g0, g1; the answer string is put into the gold
    paragraphs according to answer_in. sup_mode: 'two' (valid), 'one' (single supporting title), 'absent' (title not in
    the context). The supporting index 9 is out of range on purpose (must be skipped)."""
    titles = [f'{split}{i} doc {j}' for j in range(n_para)]
    g0 = i % n_para
    g1 = (i * 7 + 3) % n_para
    if g1 == g0:
        g1 = (g0 + 1) % n_para
    answer = f'Zq{i}x' if answer is None else answer
    paras = []
    for j, t in enumerate(titles):
        sents = [f'{t} opens here.', f' {t} continues with detail {j}.', f' {t} ends.']
        if j == g0 and answer_in[0]:
            sents[1] = f' {t} mentions {answer} in detail.'
        if j == g1 and answer_in[1]:
            sents[2] = f' {t} also mentions {answer}.'
        paras.append([t, sents])
    if sup_mode == 'two':
        sf = [[titles[g0], 1], [titles[g1], 2], [titles[g1], 0], [titles[g1], 9]]
    elif sup_mode == 'one':
        sf = [[titles[g0], 0], [titles[g0], 1]]
    else:
        sf = [[titles[g0], 0], ['Nowhere', 0]]
    return dict(_id=f'{split}-{i}', question=f'What is asked in {split} {i}?', answer=answer, type=kind, level='medium',
                supporting_facts=sf, context=paras)


def _kind(i):
    return 'comparison' if i % 3 == 0 else 'bridge'


@pytest.fixture
def hotpot(tmp_path):
    train = []
    for i in range(34):
        kw = {}
        if i == 3:
            kw = dict(answer='yes', answer_in=(False, False))      # yes / no answer: owner None
        if i == 6:
            kw = dict(answer_in=(True, True))                       # answer in both gold paragraphs: owner None
        if i == 9:
            kw = dict(answer_in=(False, True))                      # answer in the second gold paragraph (context order)
        if i == 30:
            kw = dict(n_para=9)                                     # filtered: 9 paragraphs
        if i == 31:
            kw = dict(sup_mode='one')                               # filtered: a single supporting title
        if i == 32:
            kw = dict(sup_mode='absent')                            # filtered: supporting title not in the context
        train.append(_example('train', i, _kind(i), **kw))
    dev = []
    for i in range(14):
        kw = {}
        if i == 3:
            kw = dict(answer='no', answer_in=(False, False))
        if i == 13:
            kw = dict(n_para=9)
        dev.append(_example('dev', i, _kind(i), **kw))
    tp, dp = tmp_path / 'hotpot_train.json', tmp_path / 'hotpot_dev.json'
    tp.write_text(json.dumps(train), encoding='utf-8')
    dp.write_text(json.dumps(dev), encoding='utf-8')
    return str(tp), str(dp), train, dev


def _check_shard(ex, sh):
    titles = [p[0] for p in ex['context']]
    by_title = {p[0]: p[1] for p in ex['context']}
    sup_titles = {t for t, _ in ex['supporting_facts']}
    assert (sh['id'], sh['question'], sh['answer'], sh['type'], sh['level']) == (ex['_id'], ex['question'], ex['answer'], ex['type'], ex['level'])
    assert sh['all_paragraphs'] == ex['context']
    assert len(sh['shards']) == 2 and all(len(s) == 5 for s in sh['shards'])
    got = [p[0] for s in sh['shards'] for p in s]
    assert sorted(got) == sorted(titles) and len(set(got)) == 10
    assert set(sh['gold_title']) == sup_titles
    for k in range(2):
        golds = [p for p in sh['shards'][k] if p[0] in sup_titles]
        assert len(golds) == 1 and golds[0][0] == sh['gold_title'][k]
        para = by_title[sh['gold_title'][k]]
        idx = sorted({i for t, i in ex['supporting_facts'] if t == sh['gold_title'][k] and 0 <= i < len(para)})
        assert sh['sup'][k] == [para[i] for i in idx]
        for p in sh['shards'][k]:
            assert p == [p[0], by_title[p[0]]]
    hit = [ex['answer'].lower() in ' '.join(by_title[t]).lower() for t in sh['gold_title']]
    assert sh['answer_owner'] == (0 if hit == [True, False] else 1 if hit == [False, True] else None)


# ---------------------------------------------------------------- sharding
def test_shard_invariants_on_every_fixture_example(hotpot):
    _, _, train, dev = hotpot
    rng = random.Random(0)
    n_ok = 0
    for ex in train + dev:
        sh = D.shard_example(ex, rng)
        if sh is not None:
            _check_shard(ex, sh)
            n_ok += 1
    assert n_ok == 31 + 13


def test_shard_specific_examples(hotpot):
    _, _, train, _ = hotpot
    sh = D.shard_example(train[1], random.Random(5))             # g0 = 1, g1 = 0
    assert set(sh['gold_title']) == {'train1 doc 1', 'train1 doc 0'}
    k = sh['gold_title'].index('train1 doc 1')
    assert sh['sup'][k] == [' train1 doc 1 mentions Zq1x in detail.']
    assert sh['sup'][1 - k] == ['train1 doc 0 opens here.', ' train1 doc 0 ends.']      # index 9 skipped, order kept
    assert sh['answer_owner'] == k and sh['type'] == 'bridge'
    assert D.shard_example(train[3], random.Random(5))['answer_owner'] is None           # yes
    assert D.shard_example(train[6], random.Random(5))['answer_owner'] is None           # in both
    sh9 = D.shard_example(train[9], random.Random(5))                                   # g0 = 9, g1 = 6
    assert sh9['answer_owner'] == sh9['gold_title'].index('train9 doc 6') and sh9['type'] == 'comparison'


def test_filtering(hotpot):
    _, _, train, dev = hotpot
    rng = random.Random(1)
    assert D.shard_example(train[30], rng) is None and D.filter_reason(train[30]) == 'n_paragraphs'
    assert D.shard_example(train[31], rng) is None and D.filter_reason(train[31]) == 'n_sup_titles'
    assert D.shard_example(train[32], rng) is None and D.filter_reason(train[32]) == 'sup_title_missing'
    assert D.shard_example(dev[13], rng) is None
    three = copy.deepcopy(train[0])
    three['supporting_facts'].append([three['context'][5][0], 0])
    assert D.filter_reason(three) == 'n_sup_titles'
    dup = copy.deepcopy(train[0])
    dup['context'][5][0] = dup['context'][4][0]
    assert D.filter_reason(dup) == 'duplicate_titles'
    assert D.filter_reason(train[0]) is None


def test_shard_deterministic_and_input_untouched(hotpot):
    _, _, train, _ = hotpot
    before = copy.deepcopy(train)
    a = [D.shard_example(ex, random.Random(7)) for ex in train]
    b = [D.shard_example(ex, random.Random(7)) for ex in train]
    assert a == b and train == before
    rng1, rng2 = random.Random(11), random.Random(11)
    assert [D.shard_example(ex, rng1) for ex in train] == [D.shard_example(ex, rng2) for ex in train]
    c = [D.shard_example(ex, random.Random(8)) for ex in train]
    assert any(x is not None and x['shards'] != y['shards'] for x, y in zip(a, c))
    assert any(x is not None and x['gold_title'] != y['gold_title'] for x, y in zip(a, c))


# ---------------------------------------------------------------- splits
def test_make_splits_deterministic_bytes(hotpot, tmp_path):
    tp, dp, _, _ = hotpot
    m1 = D.make_splits(tp, dp, str(tmp_path / 'o1'), seed=342101, **SMALL)
    m2 = D.make_splits(tp, dp, str(tmp_path / 'o2'), seed=342101, **SMALL)
    assert m1 == m2
    for name in D.SPLITS:
        b1 = (tmp_path / 'o1' / f'{name}.jsonl').read_bytes()
        assert b1 == (tmp_path / 'o2' / f'{name}.jsonl').read_bytes() and b'\r\n' not in b1
    D.make_splits(tp, dp, str(tmp_path / 'o3'), seed=1, **SMALL)
    assert (tmp_path / 'o1' / 'train.jsonl').read_bytes() != (tmp_path / 'o3' / 'train.jsonl').read_bytes()
    a = {r['id']: r for r in D.read_jsonl(str(tmp_path / 'o1' / 'train.jsonl'))}
    c = {r['id']: r for r in D.read_jsonl(str(tmp_path / 'o3' / 'train.jsonl'))}
    shared = set(a) & set(c)
    assert len(shared) >= 9                                        # 20 + 20 draws from 31
    assert any(a[i]['shards'] != c[i]['shards'] for i in shared)


def test_splits_disjoint_sized_sealed_and_meta(hotpot, tmp_path):
    tp, dp, train, dev = hotpot
    out = tmp_path / 'splits'
    meta = D.make_splits(tp, dp, str(out), **SMALL)
    assert sorted(os.listdir(out)) == ['fresh.jsonl', 'splits_meta.json', 'train.jsonl', 'val.jsonl']
    rows = {n: D.read_jsonl(str(out / f'{n}.jsonl')) for n in D.SPLITS}
    assert [len(rows[n]) for n in D.SPLITS] == [20, 4, 3]
    ids = {n: {r['id'] for r in rows[n]} for n in D.SPLITS}
    assert all(len(ids[n]) == len(rows[n]) for n in D.SPLITS)
    assert ids['train'] <= {e['_id'] for e in train}
    assert ids['val'] | ids['fresh'] <= {e['_id'] for e in dev} and not ids['val'] & ids['fresh']
    src = {e['_id']: e for e in train + dev}
    for n in D.SPLITS:
        for r in rows[n]:
            _check_shard(src[r['id']], r)
    assert meta['seed'] == 342101 and meta['prereg'] == 'D8-82'
    assert meta['counts'] == dict(
        loaded={'train': 34, 'dev': 14},
        filtered_out={'train': dict(n_paragraphs=1, duplicate_titles=0, n_sup_titles=1, sup_title_missing=1),
                      'dev': dict(n_paragraphs=1, duplicate_titles=0, n_sup_titles=0, sup_title_missing=0)},
        usable={'train': 31, 'dev': 13}, written={'train': 20, 'val': 4, 'fresh': 3}, sealed_test_count=6)
    for n in D.SPLITS:
        st = meta['per_split'][n]
        assert st['n'] == len(rows[n])
        assert st['type'] == {k: sum(r['type'] == k for r in rows[n]) for k in sorted({r['type'] for r in rows[n]})}
        assert sum(st['type'].values()) == len(rows[n]) and set(st['type']) <= {'bridge', 'comparison'}
        exp_owner = {'0': 0, '1': 0, 'none': 0}
        for r in rows[n]:
            exp_owner['none' if r['answer_owner'] is None else str(r['answer_owner'])] += 1
        assert st['answer_owner'] == exp_owner
    assert json.load(open(out / 'splits_meta.json', encoding='utf-8')) == meta
    with pytest.raises(ValueError):
        D.make_splits(tp, dp, str(tmp_path / 'big'), n_train=40, n_val=4, n_fresh=3)
    with pytest.raises(ValueError):
        D.make_splits(tp, dp, str(tmp_path / 'big2'), n_train=20, n_val=10, n_fresh=4)


def test_meta_sha256_matches_files(hotpot, tmp_path):
    tp, dp, _, _ = hotpot
    out = tmp_path / 'splits'
    meta = D.make_splits(tp, dp, str(out), **SMALL)
    for n in D.SPLITS:
        assert meta['files'][n]['path'] == f'{n}.jsonl' and meta['files'][n]['n'] == meta['counts']['written'][n]
        assert meta['files'][n]['sha256'] == hashlib.sha256((out / f'{n}.jsonl').read_bytes()).hexdigest()
    assert meta['sources']['train']['sha256'] == hashlib.sha256(open(tp, 'rb').read()).hexdigest()
    assert meta['sources']['dev']['sha256'] == hashlib.sha256(open(dp, 'rb').read()).hexdigest()
    assert meta['script_sha256'] == hashlib.sha256(open(D.__file__, 'rb').read()).hexdigest()


def test_preregistered_numbers():
    assert (D.SEED, D.N_TRAIN, D.N_VAL, D.N_FRESH) == (342101, 20000, 1000, 1000)
    assert (D.N_PARAGRAPHS, D.N_GOLD, D.SHARD_SIZE) == (10, 2, 5)


# ---------------------------------------------------------------- prompts (literal strings)
EX = dict(question='Who wrote it?', answer='Bob',
          shards=[[['A', ['Sa1.', ' Sa2.']], ['B', ['Sb1.']]], [['C', ['Sc1.']]]],
          all_paragraphs=[['A', ['Sa1.', ' Sa2.']], ['B', ['Sb1.']], ['C', ['Sc1.']]],
          sup=[['Sa1.', ' Sa2.'], []])


def test_prompt_builders_exact_strings():
    assert D.paragraphs_text(EX['shards'][0]) == 'Title: A\nSa1.  Sa2.\nTitle: B\nSb1.\n'
    assert D.paragraphs_text([]) == ''
    assert D.latent_prompt(EX, 0) == ('Documents:\nTitle: A\nSa1.  Sa2.\nTitle: B\nSb1.\n', 'Question: Who wrote it?\nAnswer:')
    assert D.latent_prompt(EX, 1) == ('Documents:\nTitle: C\nSc1.\n', 'Question: Who wrote it?\nAnswer:')
    assert D.full_prompt(EX) == 'Documents:\nTitle: A\nSa1.  Sa2.\nTitle: B\nSb1.\nTitle: C\nSc1.\nQuestion: Who wrote it?\nAnswer:'
    assert D.no_comm_prompt(EX, 1) == 'Documents:\nTitle: C\nSc1.\nQuestion: Who wrote it?\nAnswer:'
    assert D.no_comm_prompt(EX, 0) == ''.join(D.latent_prompt(EX, 0))
    assert D.text_sender_prompt(EX, 1) == ('Documents:\nTitle: C\nSc1.\nQuestion: Who wrote it?\n'
                                           'Write the sentences from the documents that help answer the question:')
    assert D.text_sender_target(EX, 0) == ' Sa1.  Sa2.'
    assert D.text_sender_target(EX, 1) == ' (none)'
    assert D.text_receiver_prompt(EX, 1, 'hello there') == ('Documents:\nTitle: C\nSc1.\nMessage from the other agent: hello there\n'
                                                            'Question: Who wrote it?\nAnswer:')
    assert D.answer_target(EX) == ' Bob'


# ---------------------------------------------------------------- metrics (hotpot_evaluate_v1 rules)
def test_metrics_known_cases():
    assert D.normalize_answer('The  Quick, brown fox!') == 'quick brown fox'
    assert D.normalize_answer('An apple a day') == 'apple day'
    assert D.exact_match('An apple.', 'apple') == 1.0
    assert D.exact_match('the cat', 'Cat') == 1.0
    assert D.exact_match('x y', 'y x') == 0.0
    assert D.exact_match('a b', 'b a') == 1.0                           # 'a' is an article: both normalise to 'b'
    assert D.exact_match('Yes', 'yes') == 1.0 and D.exact_match('yes', 'no') == 0.0
    assert abs(D.f1_score('New York City', 'New York') - 0.8) < 1e-12
    assert abs(D.f1_score('the New York', 'New York, the') - 1.0) < 1e-12
    assert D.f1_score('Paris', 'London') == 0.0
    assert D.f1_score('', 'London') == 0.0
    assert abs(D.f1_score('a a b', 'a b b') - 2 / 3) < 1e-12           # Counter intersection: a x1 + b x1 of 3
    assert D.f1_score('yes', 'no') == 0.0 and D.f1_score('yes', 'yes') == 1.0 and D.f1_score('no', 'No') == 1.0
    assert D.f1_score('yes it is', 'yes') == 0.0                        # gold special, pred not identical -> 0
    assert D.f1_score('noanswer', 'yes') == 0.0 and D.f1_score('noanswer', 'noanswer') == 1.0
    assert D.f1_score('yes', 'yes please') == 0.0                       # pred special, gold not identical -> 0


# ---------------------------------------------------------------- CLI and launcher
def test_cli_prints_meta_and_writes_files(hotpot, tmp_path):
    tp, dp, _, _ = hotpot
    out = tmp_path / 'cli'
    r = subprocess.run([sys.executable, '-B', D.__file__, '--train-json', tp, '--dev-json', dp, '--out', str(out),
                        '--n-train', '20', '--n-val', '4', '--n-fresh', '3'], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    printed = json.loads(r.stdout)
    assert printed == json.load(open(out / 'splits_meta.json', encoding='utf-8'))
    assert printed['seed'] == 342101 and printed['counts']['written'] == {'train': 20, 'val': 4, 'fresh': 3}


def _posix(p):
    return str(p).replace('\\', '/')


def _bash():
    b = shutil.which('bash')
    if not b:
        pytest.skip('bash not available')
    return b


def test_run_sh_static():
    bash = _bash()
    r = subprocess.run([bash, '-n', _posix(RUN_SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    raw = open(RUN_SH, 'rb').read()
    assert b'\r' not in raw, 'CRLF would break bash on the Linux box'
    txt = raw.decode('utf-8')
    assert txt.startswith('#!/bin/bash')
    assert 'set -euo pipefail' in txt
    for s in ('hotpot_train_v1.1.json', 'hotpot_dev_distractor_v1.json', 'http://curtis.ml.cmu.edu/datasets/hotpot',
              'l1_hotpot_from_hf.py', 'HF_HUB_CACHE=', 'HF_DATASETS_CACHE=', 'hf-cache', 'splits_l1', 'l1_data.py', 'sha256sum',
              '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/hotpot'):
        assert s in txt, s
    code = '\n'.join(l for l in txt.splitlines() if not l.lstrip().startswith('#'))
    for p in re.findall(r'/mnt/\S+', code):                             # the only absolute paths: the data dir and the HF caches
        assert p.startswith('/mnt/datasets/yuzhou-sheaf-gbp-research/'), p
    assert not re.search(r'--n-train|--n-val|--n-fresh|--seed', txt)      # pre-registered sizes = l1_data.py defaults


def _run_sh(env_extra, tmp_path, timeout=300):
    env = dict(os.environ, PYTHON=_posix(sys.executable), **env_extra)
    return subprocess.run([_bash(), _posix(RUN_SH)], capture_output=True, text=True, env=env, timeout=timeout, cwd=str(tmp_path))


def test_run_sh_existing_files_then_refuses_to_overwrite(hotpot, tmp_path):
    tp, dp, _, _ = hotpot
    data = tmp_path / 'data'
    data.mkdir()
    shutil.copy(tp, data / 'hotpot_train_v1.1.json')
    shutil.copy(dp, data / 'hotpot_dev_distractor_v1.json')
    env = dict(DATA_DIR=_posix(data), CURL='false', L1_DATA_EXTRA_ARGS='--n-train 20 --n-val 4 --n-fresh 3')
    r = _run_sh(env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'source: existing' in r.stdout and 'hotpot_train_v1.1.json sha256 ' in r.stdout
    meta = json.load(open(data / 'splits_l1' / 'splits_meta.json', encoding='utf-8'))
    assert meta['counts']['written'] == {'train': 20, 'val': 4, 'fresh': 3} and meta['seed'] == 342101
    assert sorted(os.listdir(data)) == ['hotpot_dev_distractor_v1.json', 'hotpot_train_v1.1.json', 'splits_l1']
    before = (data / 'splits_l1' / 'train.jsonl').read_bytes()
    r2 = _run_sh(env, tmp_path)
    assert r2.returncode == 0 and 'not overwritten' in r2.stdout
    assert (data / 'splits_l1' / 'train.jsonl').read_bytes() == before


def test_run_sh_falls_back_to_hf_mirror_when_curl_fails(hotpot, tmp_path):
    tp, dp, _, _ = hotpot
    data, cache = tmp_path / 'data2', tmp_path / 'hfcache'
    stub = tmp_path / 'stub_hf.py'                                       # stands in for l1_hotpot_from_hf.py
    stub.write_text('import argparse, json, os, shutil\n'
                    'ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()\n'
                    'os.makedirs(a.out, exist_ok=True)\n'
                    f'shutil.copy({tp!r}, os.path.join(a.out, "hotpot_train_v1.1.json"))\n'
                    f'shutil.copy({dp!r}, os.path.join(a.out, "hotpot_dev_distractor_v1.json"))\n'
                    'json.dump(dict(hub=os.environ.get("HF_HUB_CACHE"), ds=os.environ.get("HF_DATASETS_CACHE")), '
                    'open(os.path.join(a.out, "hf_source.json"), "w"))\n', encoding='utf-8')
    env = dict(DATA_DIR=_posix(data), CURL='false', HF_CONVERTER=_posix(stub), HF_CACHE_ROOT=_posix(cache),
               L1_DATA_EXTRA_ARGS='--n-train 20 --n-val 4 --n-fresh 3')
    r = _run_sh(env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'source: hf-mirror' in r.stdout
    src = json.load(open(data / 'hf_source.json', encoding='utf-8'))
    assert src == dict(hub=_posix(cache) + '/hub', ds=_posix(cache) + '/datasets')
    assert (cache / 'hub').is_dir() and (cache / 'datasets').is_dir()
    assert not list(data.glob('*.part'))
    assert json.load(open(data / 'splits_l1' / 'splits_meta.json', encoding='utf-8'))['counts']['written'] == {'train': 20, 'val': 4, 'fresh': 3}
