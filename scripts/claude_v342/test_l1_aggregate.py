"""Unit tests for l1_aggregate.py (batch L1 verdict logic; D8-82, design section 6).
CPU only, no network: synthetic runs/<arm>-<pair>-s<seed>/summary.json trees are written to tmp_path.
Run: python -m pytest -q scripts/claude_v342/test_l1_aggregate.py"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l1_aggregate as A  # noqa: E402

S1, S2 = A.SEEDS
Q, LL = A.MODELS
SHAS = {'l1_train.py': 'a' * 64, 'l1_model.py': 'b' * 64, 'l1_fusion.py': 'c' * 64}
# default tree = LEVER: P1 5.0, P2 2.0, P3 1.5; P0 gains Qwen 28, Llama 22 with |dB0| = 0; same pair S1 - T = 3.0
EM = {'hetero': dict(B0=30.0, B4=58.0, T=40.0, T0=35.0, U=43.0, S1=45.0, S1L1=43.5),
      'same': dict(B0=31.0, B4=59.0, T=41.0, S1=44.0)}
EM_MODEL = {'B0': {Q: 30.0, LL: 28.0}, 'B4': {Q: 58.0, LL: 50.0}}


def _summary(arm, pair, seed, em, fresh_em=None, em_model=None, c_assert=True, shas=SHAS):
    def blk(e):
        return dict(em=e, f1=min(100.0, e + 10.0), em_agent=[e] if arm == 'B4' else [e - 1.0, e + 1.0],
                    em_model=dict(em_model if em_model is not None else EM_MODEL.get(arm, {Q: e, LL: e})))
    s = dict(arm=arm, pair=pair, seed=seed, val=blk(em), fresh=blk(em if fresh_em is None else fresh_em), scripts_sha=dict(shas))
    if arm in A.LATENT_ARMS:
        s['c_assert'] = c_assert
    return s


def build_tree(root, em=None, fresh=None, em_model=None, c_assert=None, shas=None, drop=(), raw=None):
    """Write every default run under root; the dict arguments are keyed by (arm, pair, seed) and override one run each
    (raw: keys merged into the summary after it is built; a value of None deletes the key)."""
    em, fresh, em_model, c_assert, shas, raw = (x or {} for x in (em, fresh, em_model, c_assert, shas, raw))
    for pair, arms in EM.items():
        for arm, e in arms.items():
            for seed in A.SEEDS:
                key = (arm, pair, seed)
                if key in drop:
                    continue
                s = _summary(arm, pair, seed, em.get(key, e), fresh.get(key), em_model.get(key), c_assert.get(key, True), shas.get(key, SHAS))
                for k, v in raw.get(key, {}).items():
                    if v is None:
                        s.pop(k, None)
                    else:
                        s[k] = v
                d = os.path.join(root, A.run_dir_name(arm, pair, seed))
                os.makedirs(d)
                with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as fh:
                    json.dump(s, fh)
    return str(root)


def _both(arm, pair, value):
    return {(arm, pair, s): value for s in A.SEEDS}


# ---------------------------------------------------------------- pre-registered numbers and schema
def test_thresholds_are_d8_82():
    assert A.SEEDS == (342101, 342202)
    assert (A.P0_MIN_GAIN, A.P0_SEED_FACTOR, A.DEGRADE_PT) == (15.0, 5.0, 1.0)
    assert {k: v['threshold'] for k, v in A.CRITERIA.items()} == {'P1': 2.0, 'P2': 1.0, 'P3': 1.0}
    assert {k: v['arms'] for k, v in A.CRITERIA.items()} == {'P1': ('S1', 'T'), 'P2': ('S1', 'U'), 'P3': ('S1', 'S1L1')}
    assert A.REQUIRED_ARMS == ('B0', 'B4', 'T', 'U', 'S1', 'S1L1') and A.STAGE0_ARMS == ('B0', 'B4')
    assert A.run_dir_name('S1', 'hetero', 342101) == 'S1-hetero-s342101'


def test_summary_schema_and_validate():
    sch = A.summary_schema()
    assert set(sch) == {'arm', 'pair', 'seed', 'val', 'fresh', 'c_assert', 'scripts_sha'}
    assert set(sch['val']) == set(sch['fresh']) == {'em', 'f1', 'em_agent', 'em_model'}
    assert A.validate_summary(_summary('S1', 'hetero', S1, 45.0), 'S1', 'hetero', S1) == []
    assert A.validate_summary(_summary('B0', 'same', S2, 30.0), 'B0', 'same', S2) == []      # no c_assert needed
    assert A.validate_summary(_summary('S1', 'hetero', S1, 45.0), 'S1', 'hetero', S2) != []   # directory disagrees
    s = _summary('S1', 'hetero', S1, 45.0)
    del s['c_assert']
    assert any('c_assert missing' in p for p in A.validate_summary(s, 'S1', 'hetero', S1))
    u = _summary('U', 'hetero', S1, 43.0)
    del u['c_assert']
    assert A.validate_summary(u, 'U', 'hetero', S1) == []                                      # optional for U
    bad = _summary('T', 'hetero', S1, 40.0)
    bad['val']['em'] = 0.4                                                                      # fraction, not points? allowed (0.4 pt)
    assert A.validate_summary(bad, 'T', 'hetero', S1) == []
    bad['val']['em'] = 140.0
    assert any('percentage points' in p for p in A.validate_summary(bad, 'T', 'hetero', S1))
    del bad['fresh']
    assert any('missing key fresh' in p for p in A.validate_summary(bad, 'T', 'hetero', S1))
    assert A.validate_summary([], 'T', 'hetero', S1) == ['summary is not a dict']


# ---------------------------------------------------------------- verdicts
def test_lever(tmp_path):
    agg = A.aggregate(build_tree(tmp_path))
    assert agg['verdict'] == 'LEVER' and agg['incomplete_reasons'] == [] and agg['stage'] == 1
    assert agg['p0']['pass'] is True and agg['p0_line'] == 'P0: PASS'
    c = agg['criteria']
    assert c['P1']['margin'] == {str(S1): 5.0, str(S2): 5.0} and c['P1']['pass'] is True
    assert c['P2']['margin'][str(S1)] == 2.0 and c['P3']['margin'][str(S2)] == 1.5
    assert all(c[k]['pass'] for k in ('P1', 'P2', 'P3'))
    assert agg['degrade']['triggered'] is False and agg['degrade']['drop'] == {str(S1): 0.0, str(S2): 0.0}
    assert agg['same_pair']['margin'] == {str(S1): 3.0, str(S2): 3.0} and agg['same_pair']['pass'] is True
    assert agg['scripts_sha'] == SHAS and agg['sha_conflicts'] == {}
    assert len(agg['runs']) == 22
    assert agg['absent'] == [A.run_dir_name(a, 'same', s) for a in ('T0', 'U', 'S1L1') for s in A.SEEDS]
    assert len(agg['script_sha256']) == 64
    md = A.to_markdown(agg)
    assert 'VERDICT: LEVER' in md and '| S1-hetero-s342101 | S1 | hetero | 342101 | 45.0 | 55.0 | 45.0 | 55.0 | True | - |' in md


def test_no_when_p1_fails(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, em=_both('S1', 'hetero', 41.5)))     # S1 - T = 1.5 < 2.0
    assert agg['verdict'] == 'NO' and agg['criteria']['P1']['pass'] is False and agg['incomplete_reasons'] == []


def test_p1_exactly_at_threshold_passes_even_with_float_noise(tmp_path):
    em = {**_both('T', 'hetero', 40.3), **_both('S1', 'hetero', 42.3)}       # 42.3 - 40.3 = 1.9999999999999964 in binary
    agg = A.aggregate(build_tree(tmp_path, em=em))
    assert agg['criteria']['P1']['margin'] == {str(S1): 2.0, str(S2): 2.0} and agg['criteria']['P1']['pass'] is True
    assert agg['verdict'] == 'AMBIGUOUS'                                          # P2: 42.3 - 43 < 1 -> AMBIGUOUS, not NO


def test_ambiguous_when_p2_fails(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, em=_both('U', 'hetero', 44.5)))      # S1 - U = 0.5 < 1.0
    assert agg['verdict'] == 'AMBIGUOUS' and agg['criteria']['P2']['pass'] is False and agg['criteria']['P1']['pass'] is True


def test_ambiguous_when_p3_fails(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, em=_both('S1L1', 'hetero', 44.5)))   # S1 - S1L1 = 0.5 < 1.0
    assert agg['verdict'] == 'AMBIGUOUS' and agg['criteria']['P3']['pass'] is False


def test_one_seed_failing_fails_the_criterion(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', em={('U', 'hetero', S2): 44.5}))
    assert agg['criteria']['P2']['pass_seed'] == {str(S1): True, str(S2): False}
    assert agg['criteria']['P2']['pass'] is False and agg['verdict'] == 'AMBIGUOUS'
    agg = A.aggregate(build_tree(tmp_path / 'b', em={('S1', 'hetero', S1): 41.0}))
    assert agg['criteria']['P1']['pass_seed'] == {str(S1): False, str(S2): True} and agg['verdict'] == 'NO'
    agg = A.aggregate(build_tree(tmp_path / 'c', em_model={('B4', 'hetero', S2): {Q: 58.0, LL: 42.0}}), stage0=True)
    assert agg['p0']['per_model'][LL]['pass_seed'] == {str(S1): True, str(S2): False} and agg['p0_line'] == 'P0: FAIL'


def test_incomplete_missing_required_arm(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', drop={('U', 'hetero', S2)}))
    assert agg['verdict'] == 'INCOMPLETE' and agg['incomplete_reasons'] == ['missing required run U-hetero-s342202']
    assert agg['criteria']['P2']['pass'] is None and agg['criteria']['P1']['pass'] is True
    # T0 and the same pair are report-only: dropping them does not block the verdict
    agg = A.aggregate(build_tree(tmp_path / 'b', drop={('T0', 'hetero', s) for s in A.SEEDS} | {(a, 'same', s) for a in EM['same'] for s in A.SEEDS}))
    assert agg['verdict'] == 'LEVER' and agg['same_pair']['pass'] is None and agg['same_pair']['margin'] == {str(S1): None, str(S2): None}


def test_incomplete_c_assert_false_or_missing(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', c_assert={('S1', 'hetero', S1): False}))
    assert agg['verdict'] == 'INCOMPLETE' and agg['incomplete_reasons'] == ['S1-hetero-s342101: c_assert false']
    agg = A.aggregate(build_tree(tmp_path / 'b', c_assert={('S1', 'same', S2): False}))           # "any c_assert" includes the same pair
    assert agg['verdict'] == 'INCOMPLETE'
    agg = A.aggregate(build_tree(tmp_path / 'c', c_assert={('U', 'hetero', S2): False}))
    assert agg['verdict'] == 'INCOMPLETE'
    agg = A.aggregate(build_tree(tmp_path / 'd', raw={('S1L1', 'hetero', S1): {'c_assert': None}}))  # key deleted
    assert agg['verdict'] == 'INCOMPLETE' and agg['incomplete_reasons'] == ['S1L1-hetero-s342101: c_assert missing']
    agg = A.aggregate(build_tree(tmp_path / 'e', raw={('U', 'hetero', S1): {'c_assert': None}}))     # optional for U
    assert agg['verdict'] == 'LEVER'


def test_incomplete_sha_mismatch(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, shas={('T', 'hetero', S2): {**SHAS, 'l1_train.py': 'd' * 64}}))
    assert agg['verdict'] == 'INCOMPLETE'
    assert len(agg['incomplete_reasons']) == 1 and agg['incomplete_reasons'][0].startswith('scripts_sha of l1_train.py differs between runs')
    assert set(agg['sha_conflicts']) == {'l1_train.py'} and agg['sha_conflicts']['l1_train.py']['d' * 64] == ['T-hetero-s342202']
    assert agg['scripts_sha']['l1_model.py'] == 'b' * 64


def test_incomplete_when_summary_disagrees_with_directory(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', raw={('T', 'hetero', S1): {'seed': S2}}))
    assert agg['verdict'] == 'INCOMPLETE' and 'disagree with the directory' in agg['incomplete_reasons'][0]
    agg = A.aggregate(build_tree(tmp_path / 'b', raw={('S1', 'hetero', S1): {'pair': 'same'}}))
    assert agg['verdict'] == 'INCOMPLETE'


def test_fresh_degrade_caps_lever_at_ambiguous(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', fresh={('S1', 'hetero', S2): 43.4}))   # fresh margin 3.4 vs VAL 5.0: drop 1.6
    assert agg['degrade']['drop'] == {str(S1): 0.0, str(S2): 1.6} and agg['degrade']['triggered'] is True
    assert all(agg['criteria'][k]['pass'] for k in ('P1', 'P2', 'P3')) and agg['verdict'] == 'AMBIGUOUS'
    agg = A.aggregate(build_tree(tmp_path / 'b', fresh={('S1', 'hetero', S2): 44.0}))   # drop exactly 1.0: not "more than"
    assert agg['degrade']['triggered'] is False and agg['verdict'] == 'LEVER'
    agg = A.aggregate(build_tree(tmp_path / 'c', fresh={('T', 'hetero', S1): 42.0}))    # drop via T on fresh (fresh margin 3.0)
    assert agg['degrade']['triggered'] is True and agg['verdict'] == 'AMBIGUOUS'
    agg = A.aggregate(build_tree(tmp_path / 'd', em=_both('S1', 'hetero', 41.5), fresh={('S1', 'hetero', S1): 30.0}))
    assert agg['degrade']['triggered'] is True and agg['verdict'] == 'NO'               # the cap never lifts NO
    agg = A.aggregate(build_tree(tmp_path / 'e', fresh={('S1', 'hetero', S1): 49.0}))   # fresh better than VAL: no cap
    assert agg['degrade']['drop'][str(S1)] == -4.0 and agg['verdict'] == 'LEVER'


def test_same_pair_is_diagnostic_only(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'a', em=_both('S1', 'same', 38.0)))          # same S1 - T = -3
    assert agg['same_pair']['pass'] is False and agg['same_pair']['margin'] == {str(S1): -3.0, str(S2): -3.0}
    assert agg['verdict'] == 'LEVER'
    agg = A.aggregate(build_tree(tmp_path / 'b', em={**_both('S1', 'hetero', 41.5), **_both('S1', 'same', 50.0)}))
    assert agg['same_pair']['pass'] is True and agg['verdict'] == 'NO'                   # hetero decides; same pair only reported
    assert 'diagnostic only' in agg['same_pair']['note']


def test_p0_stage0_pass_fail_incomplete(tmp_path):
    agg = A.aggregate(build_tree(tmp_path / 'pass'), stage0=True)
    assert agg['p0_line'] == 'P0: PASS' and agg['verdict'] is None and agg['stage'] == 0 and 'criteria' not in agg
    assert sorted(agg['runs']) == sorted(A.run_dir_name(a, 'hetero', s) for a in ('B0', 'B4') for s in A.SEEDS)
    pm = agg['p0']['per_model']
    assert pm[Q]['gain'] == {str(S1): 28.0, str(S2): 28.0} and pm[LL]['gain'] == {str(S1): 22.0, str(S2): 22.0}
    assert pm[Q]['delta_b0_seeds'] == 0.0 and pm[Q]['required_gain'] == 15.0 and pm[Q]['pass'] is True and pm[LL]['pass'] is True
    assert 'P0: PASS' in A.to_markdown(agg)
    # gain below 15 pt for one model
    agg = A.aggregate(build_tree(tmp_path / 'gain', em_model=_both('B4', 'hetero', {Q: 58.0, LL: 40.0})), stage0=True)
    assert agg['p0_line'] == 'P0: FAIL' and agg['p0']['per_model'][LL]['pass'] is False and agg['p0']['per_model'][Q]['pass'] is True
    # gain above 15 pt but below 5 x |dB0|: Qwen B0 30 / 36 -> dB0 = 6 -> required 30; seed-2 gain 22
    agg = A.aggregate(build_tree(tmp_path / 'spread', em_model={('B0', 'hetero', S2): {Q: 36.0, LL: 28.0}}), stage0=True)
    assert agg['p0_line'] == 'P0: FAIL'
    pm = agg['p0']['per_model'][Q]
    assert pm['delta_b0_seeds'] == 6.0 and pm['required_gain'] == 30.0 and pm['gain'][str(S2)] == 22.0 and pm['pass_seed'][str(S2)] is False
    # missing B4 seed 2
    agg = A.aggregate(build_tree(tmp_path / 'missing', drop={('B4', 'hetero', S2)}), stage0=True)
    assert agg['p0_line'] == 'P0: INCOMPLETE' and agg['incomplete_reasons'] == ['missing required run B4-hetero-s342202']
    assert agg['p0']['pass'] is None and agg['p0']['missing'] == [Q, LL]
    # missing em_model key for one model
    agg = A.aggregate(build_tree(tmp_path / 'nokey', em_model={('B0', 'hetero', S1): {Q: 30.0}}), stage0=True)
    assert agg['p0_line'] == 'P0: INCOMPLETE' and agg['p0']['missing'] == [LL]


def test_p0_fail_makes_the_stage1_verdict_incomplete(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, em_model=_both('B4', 'hetero', {Q: 58.0, LL: 40.0})))
    assert agg['p0_line'] == 'P0: FAIL' and agg['verdict'] == 'INCOMPLETE'
    assert agg['incomplete_reasons'] == ['P0 gate failed (stage 1 was not supposed to run)']


def test_cli_writes_files_and_prints_lines(tmp_path):
    root = build_tree(tmp_path)
    r = subprocess.run([sys.executable, '-B', A.__file__, '--runs', root], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-2:] == ['P0: PASS', 'VERDICT: LEVER']
    agg = json.load(open(os.path.join(root, 'aggregate.json'), encoding='utf-8'))
    assert agg['verdict'] == 'LEVER' and agg['stage'] == 1 and agg['runs_dir'] == os.path.abspath(root)
    md = open(os.path.join(root, 'aggregate.md'), encoding='utf-8').read()
    assert '**VERDICT: LEVER**' in md and '| P1 | ' in md and 'Same-model pair' in md and '\r' not in md
    r = subprocess.run([sys.executable, '-B', A.__file__, '--runs', root, '--stage0'], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == 'P0: PASS' and 'VERDICT' not in r.stdout
    assert json.load(open(os.path.join(root, 'aggregate.json'), encoding='utf-8'))['stage'] == 0
    bad = build_tree(tmp_path / 'bad', drop={('S1', 'hetero', S1)})
    r = subprocess.run([sys.executable, '-B', A.__file__, '--runs', bad], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == 'VERDICT: INCOMPLETE'
    assert 'INCOMPLETE: missing required run S1-hetero-s342101' in r.stdout


def test_fractions_instead_of_points_are_incomplete(tmp_path):
    em = {(a, p, s): EM[p][a] / 100.0 for p in EM for a in EM[p] for s in A.SEEDS}
    agg = A.aggregate(build_tree(tmp_path, em=em))
    assert agg['verdict'] == 'INCOMPLETE' and any('percentage points' in r for r in agg['incomplete_reasons'])


def test_markdown_handles_missing_values(tmp_path):
    agg = A.aggregate(build_tree(tmp_path, drop={('S1', 'hetero', S1), ('B0', 'hetero', S2)}))
    md = A.to_markdown(agg)
    assert 'VERDICT: INCOMPLETE' in md and '| - |' in md and 'missing required run S1-hetero-s342101' in md
