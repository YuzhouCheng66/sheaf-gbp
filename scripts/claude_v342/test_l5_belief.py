"""Unit tests for l5_belief.py / l5_gate.py (ladder L5, D8-95): CPU, tiny random models.
What is tested: the pure consensus functions (union, product-of-experts picks, the knower-vs-ignorant asymmetry, aggregation), the tiny
end-to-end protocol for both arms (B1t: span_attn evidence + beams + cross scoring; B1: the cached evidence), the CLI run, and the gate verdicts."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch  # noqa: E402
import l1_model as M  # noqa: E402
import l2_model as L2  # noqa: E402
import l5_belief as L5  # noqa: E402
import l5_gate as G  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 6, TL.DE
EX = TL.EXAMPLES
BELIEF = os.path.join(HERE, 'l5_belief.py')
GATE = os.path.join(HERE, 'l5_gate.py')


def test_pure_consensus():
    assert L5.union([['Paris', 'paris ', 'Rome'], ['Rome', 'Lyon']]) == ['Paris', 'Rome', 'Lyon']
    ex = dict(answer='Paris', answer_owner=0)
    cands = [['Paris', 'Rome'], ['Lyon', 'Paris']]
    C = L5.union(cands)                                                        # Paris, Rome, Lyon
    # the knower (agent 0) rejects Lyon strongly (-9), the ignorant (agent 1) only mildly prefers Lyon over Paris (-1 vs -3)
    scores = [[-1.0, -4.0, -9.0], [-3.0, -6.0, -1.0]]
    lens = [2, 2, 3]
    j = L5.judge(ex, cands, C, scores, lens, p=[0.6, 0.9])
    assert j['picks']['poe_k4'] == 0 and j['ok']['poe_k4'] == 1.0            # sum: Paris -4, Rome -10, Lyon -10
    assert j['picks']['poe_k1'] == 0 and j['picks']['select_p'] == 2 and j['ok']['select_p'] == 0.0   # select-by-p trusts the confident ignorant
    assert j['picks']['agent0'] == 0 and j['picks']['agent1'] == 2 and j['either'] == 1.0 and j['both'] == 0.0 and j['agree'] == 0.0
    assert j['any_union'] == 1.0 and j['any_agent'] == [1.0, 1.0]
    assert j['asym']['knower'] == 0 and j['asym']['m_know'] == 8.0 and j['asym']['m_ign'] == 2.0 and j['asym']['poe_k1_correct'] == 1.0 and j['asym']['select_correct'] == 0.0
    # length normalisation and lam ~ p change the pick when the margins are small
    j2 = L5.judge(ex, cands, C, [[-1.0, -4.0, -3.0], [-3.0, -6.0, -1.0]], [1, 2, 4], p=[0.5, 0.9])
    assert j2['picks']['poe_k4'] == 0 and j2['picks']['poe_k4_norm'] == 2 and j2['picks']['poe_k4_p'] == 2   # sums: Paris -4, Lyon -4 -> tie -> first (Paris); norm: Lyon -1 beats Paris -4; p-weighted: Lyon -2.4 vs Paris -3.2
    i, fused = L5.poe_pick(['a', 'b'], [[-1.0, -1.0], [-2.0, -2.0]])
    assert i == 0 and fused == [-3.0, -3.0]
    rows = [dict(j, bytes=L5.msg_bytes(cands, C), owner='agent0'), dict(j2, bytes=L5.msg_bytes(cands, C), owner='none')]
    agg = L5.aggregate(rows)
    assert agg['n'] == 2 and agg['em'] == 100.0 and agg['em_variants']['select_p'] == 0.0 and agg['round1']['either_correct'] == 100.0 and agg['round1']['disagree']['n'] == 2
    assert agg['union']['any_correct'] == 100.0 and agg['asym']['n'] == 2 and agg['asym']['know_wins'] == 50.0 and set(agg['em_by_owner']) == {'agent0', 'none'}
    assert agg['bytes']['cand_bytes'] == float(len('ParisRomeLyonParis')) and agg['bytes']['float_bytes'] == 4.0 * (4 + 2 * 3)


def _pair(arm, mode):
    varm = 'V3' if arm == 'B1' else 'V1'
    p = L2.build_pair(varm, 'same2', torch.device('cpu'), tiny=M.tiny_models(EX, 5), k=K, d_e=DE, n_heads=2, mode='span_oracle', span_k=K, recv=True, sel_layer=1)
    p.set_mode(mode)
    p.eval()
    return p


def test_tiny_protocol_both_arms():
    pair = _pair('B1t', 'span_attn')
    r1 = L5.round1_pieces(pair, EX[:2], 0)
    ag = pair.agent(0)
    hp = len(ag.hint_post)
    assert len(r1) == 2 and (hp == 0 or r1[0][0][-hp:] == ag.hint_post) and len(r1[0][1]) > 0
    cands = L5.beam_candidates(ag, r1, 3, 5)
    assert len(cands) == 2 and all(1 <= len(c) <= 3 for c in cands) and all(isinstance(t, str) for c in cands for t in c)
    got = L5.score_candidates(ag, [r1[0][0] + r1[0][1], r1[1][0] + r1[1][1]], [ag.ids(' x y'), []], 'recv', 32)
    assert len(got) == 2 and got[0][1] == len(ag.ids(' x y')) + 1 and got[1][1] == 1 and all(s <= 0.0 and 0.0 <= mp <= 1.0 for s, _n, mp in got)
    out, recs = L5.run_split(pair, EX, None, k=3, max_new=5, gen_batch=2, score_batch=8)
    assert out['n'] == len(EX) == len(recs) and 0.0 <= out['em'] <= 100.0 and out['union']['any_correct'] >= out['round1']['either_correct'] >= max(out['round1']['em_agent'])
    assert set(out['em_variants']) == set(L5.VARIANTS) and 'timing' in out and out['bytes']['total_bytes'] > 0
    r = recs[0]
    assert len(r['cands']) == 2 and set(r['union']) >= set(r['cands'][0]) and len(r['scores'][0]) == len(r['union']) == len(r['lens']) and all(x <= 0 for s in r['scores'] for x in s)
    # B1: the cached evidence tokens replace the span selection
    pair_b1 = _pair('B1', 'span_sel')
    cache = {ex['id']: {str(i): dict(ev=[int(t) for t in pair_b1.agent(i).ids(' '.join(ex['shards'][1 - i][0][1]))[:K]], ans='x', ans_ids=[], p=0.5) for i in range(2)} for ex in EX}
    r1b = L5.round1_pieces(pair_b1, EX[:1], 1, cache)
    ev = cache[EX[0]['id']]['1']['ev']
    hp = len(pair_b1.agent(1).hint_post)
    assert r1b[0][0][len(r1b[0][0]) - hp - len(ev):len(r1b[0][0]) - hp] == ev
    out_b1, recs_b1 = L5.run_split(pair_b1, EX[:2], cache, k=2, max_new=4, gen_batch=2, score_batch=8)
    assert out_b1['n'] == 2 and 'cached_greedy' in out_b1 and recs_b1[0]['cached_greedy'][0]['ans'] == 'x'


def _fixture(d):
    for name in ('val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in EX:
                fh.write(json.dumps(ex) + '\n')


def test_cli_b1t_and_gates():
    d = tempfile.mkdtemp()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        _fixture(d)
        v1 = os.path.join(d, 'V1-same2-s5')
        os.makedirs(v1)
        pair = _pair('B1t', 'span_attn')
        st = {k: v.detach().cpu() for k, v in pair.named_parameters() if '.adapter.' in k or '.reader.' in k or ('lora_' in k and '.recv.' in k)}
        torch.save(dict(arm='V1', pair='same2', seed=5, models=['tiny-a', 'tiny-a'], state=st, recv=True, config=dict(k=K)), os.path.join(v1, 'trainable.pt'))
        M.jdump(dict(arm='V1', seed=5, val=dict(em=50.0), span=dict(sel_layer=1)), os.path.join(v1, 'summary.json'))
        out = os.path.join(d, 'B1t-same2-s5')
        r = subprocess.run([sys.executable, '-B', BELIEF, '--arm', 'B1t', '--pair', 'same2', '--seed', '5', '--v-from', v1, '--val', os.path.join(d, 'val.jsonl'), '--fresh', os.path.join(d, 'fresh.jsonl'),
                            '--out', out, '--tiny', '--d-e', str(DE), '--n-heads', '2', '--k-beams', '2', '--max-new-tokens', '4', '--gen-batch', '2', '--score-batch', '8', '--keep-preds'],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(out, 'summary.json')))
        assert s['arm'] == 'B1t' and s['complete'] and s['prereg'] == 'D8-95' and s['v']['sel_layer'] == 1 and os.path.isfile(os.path.join(out, 'preds_val.json'))
        # gates: B1t from the real summary; B1 from synthetic summaries (+ a T2 reference)
        gates = os.path.join(d, 'gates')
        r = subprocess.run([sys.executable, '-B', GATE, '--job', 'B1t-same2-s5', '--runs', d, '--l4', d, '--out', gates], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and ('TWO-ROUND-BEATS-TEXT' in r.stdout or 'TWO-ROUND-BELOW-TEXT' in r.stdout), r.stdout + r.stderr
        assert any(f.startswith('GATE_B1t-same2-s5_') for f in os.listdir(d))
        b1 = os.path.join(d, 'B1-same2-s5')
        os.makedirs(b1)
        M.jdump(dict(arm='B1', seed=5, complete=True, val=dict(em=58.0, em_variants=dict(select_p=54.0, poe_k1=56.0), round1={}, union={}), fresh=dict(em=57.0)), os.path.join(b1, 'summary.json'))
        assert G.judge('B1-same2-s5', d, d)['verdict'] == 'BELIEF-ADDS+pendingT2'
        t2 = os.path.join(d, 'T2-same2-s5')
        os.makedirs(t2)
        M.jdump(dict(arm='T2', seed=5, complete=True, val=dict(em=55.2), fresh=dict(em=55.0)), os.path.join(t2, 'summary.json'))
        g = G.judge('B1-same2-s5', d, d)
        assert g['verdict'] == 'BELIEF-ADDS+BELIEF-BEATS-TEXT-R2' and g['detail']['margin_vs_t2'] == 2.8
        M.jdump(dict(arm='B1', seed=5, complete=True, val=dict(em=52.0, em_variants=dict(select_p=54.0), round1={}, union={}), fresh=dict(em=50.0)), os.path.join(b1, 'summary.json'))
        assert G.judge('B1-same2-s5', d, d)['verdict'] == 'BELIEF-HURTS+TEXT-BEATS-BELIEF'
        assert G.judge('B1t-same2-s5', d, d)['verdict'] in ('TWO-ROUND-BEATS-TEXT+BEATS-ALL-DOCS', 'TWO-ROUND-BEATS-TEXT+BELOW-ALL-DOCS', 'TWO-ROUND-BELOW-TEXT+BELOW-ALL-DOCS')
    finally:
        shutil.rmtree(d, ignore_errors=True)
