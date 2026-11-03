"""Unit tests for l6_verifier.py / l6_gate.py (ladder L6, D8-96): CPU, tiny random models.
What is tested: the pure functions (AUROC, the verifier picks and the asymmetry, aggregation), the tiny end-to-end pipeline (verify adapter
from recv, candidate generation, two training steps that move only the verify tensors, the evaluation with L5-style records, the state round
trip), the CLI in both modes and the gate verdicts."""
import argparse
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
import l6_verifier as L6  # noqa: E402
import l6_gate as G  # noqa: E402
import test_l1_model as TL  # noqa: E402

K, DE = 6, TL.DE
EX = TL.EXAMPLES
VERIFIER = os.path.join(HERE, 'l6_verifier.py')
GATE = os.path.join(HERE, 'l6_gate.py')


def test_pure():
    assert L6.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75 and L6.auroc([1, 1, 1], [1, 1, 1]) is None and L6.auroc([0.5, 0.5], [0, 1]) == 0.5
    assert L6.argmax_first([1.0, 3.0, 3.0]) == 1 and L6.argmax_first([]) is None
    assert L6.rel_of('agent0', 0) == 'own' and L6.rel_of('agent0', 1) == 'other' and L6.rel_of('none', 0) == 'none'
    assert L6.sigmoid(0.0) == 0.5 and L6.sigmoid(-1000.0) > 0.0
    kept, meta = L6.exclude_seen(EX, '5:1')
    assert len(kept) == 2 and meta['n_seen'] == 1 and meta['n_kept'] == 2 and kept == [ex for ex in EX if ex is not EX[M.epoch_order(3, 5)[0]]]
    kept2, meta2 = L6.exclude_seen(EX, '5:1,6:1')
    assert meta2['n_seen'] in (1, 2) and len(kept2) == 3 - meta2['n_seen'] and L6.exclude_seen(EX, None)[0] == EX
    ex = dict(answer='Paris', answer_owner=0)
    cands = [['Paris', 'Rome'], ['Lyon', 'Paris']]
    C = L5.union(cands)                                                        # Paris, Rome, Lyon
    S = [[3.0, -2.0, -4.0], [-0.5, -1.0, 0.5]]                                 # the knower endorses Paris; the ignorant mildly prefers Lyon
    Gs = [[-1.0, -4.0, -9.0], [-3.0, -6.0, -1.0]]
    j = L6.judge(ex, cands, C, S, Gs, p=[0.6, 0.9])
    assert j['picks']['ver_sum'] == 0 and j['ok']['ver_sum'] == 1.0 and j['fused'] == [2.5, -3.0, -3.5]
    assert j['picks']['ver_self0'] == 0 and j['picks']['ver_self1'] == 2 and j['picks']['ver_select'] == 0 and j['picks']['ver_owner'] == 0 and j['picks']['ver_sum_p'] == 0
    assert j['picks']['select_p'] == 2 and j['ok']['select_p'] == 0.0 and j['picks']['poe_k4'] == 0 and j['picks']['agent0'] == 0 and j['picks']['agent1'] == 2
    assert j['asym']['knower'] == 0 and j['asym']['m_know'] == 7.0 and j['asym']['m_ign'] == 1.0 and j['either'] == 1.0 and j['both'] == 0.0 and j['agree'] == 0.0
    assert len(j['cand_recs']) == 6 and j['cand_recs'][0] == dict(agent=0, rel='own', s=3.0, label=1.0) and j['cand_recs'][3]['rel'] == 'other' and j['owner'] == 'agent0'
    ex2 = dict(answer='Lyon', answer_owner=None)
    j2 = L6.judge(ex2, cands, C, S, Gs, p=[0.6, 0.9])
    assert j2['picks']['ver_owner'] == j2['picks']['ver_sum'] == 0 and j2['ok']['ver_sum'] == 0.0 and j2['ok']['select_p'] == 1.0 and j2['owner'] == 'none'
    assert j2['asym']['knower'] == 1 and j2['asym']['m_know'] == 1.0 and j2['asym']['m_ign'] == 7.0
    agg = L6.aggregate([dict(j, bytes=L5.msg_bytes(cands, C)), dict(j2, bytes=L5.msg_bytes(cands, C))])
    assert agg['n'] == 2 and agg['em'] == 50.0 and agg['em_variants']['select_p'] == 50.0 and agg['round1']['either_correct'] == 100.0 and agg['round1']['disagree']['n'] == 2
    assert agg['union']['any_correct'] == 100.0 and agg['asym']['n'] == 2 and agg['asym']['know_wins'] == 50.0 and set(agg['em_by_owner']) == {'agent0', 'none'}
    vr = agg['verifier']
    assert vr['by_rel']['all']['n'] == 12 and abs(vr['positive_rate'] - 100.0 * 4 / 12) < 1e-9 and vr['flat_when_ignorant'] is True
    assert vr['by_rel']['own']['n'] == 3 and vr['by_rel']['other']['n'] == 3 and vr['by_rel']['none']['n'] == 6 and vr['by_rel']['own']['auroc'] == 1.0
    assert agg['bytes']['cand_bytes'] == float(len('ParisRomeLyonParis'))


def _tiny():
    return M.tiny_models(EX + [L6.YN_EXAMPLE], 5)


def _pair(grad_ckpt=False):
    p = L2.build_pair('V1', 'same2', torch.device('cpu'), tiny=_tiny(), k=K, d_e=DE, n_heads=2, mode='span_oracle', span_k=K, recv=True, sel_layer=1, grad_ckpt=grad_ckpt)
    p.set_mode('span_attn')
    p.eval()
    return p


def test_tiny_train_eval():
    pair = _pair()
    counts = L6.enable_verify(pair)
    assert len(counts) == 2 and counts[0] > 0 and L6.enable_verify(pair) == counts
    yns = [L6.yes_no_ids(a) for a in pair.agents]
    assert all(y != n for y, n in yns)
    cues = [a.ids(L6.VERIFY_CUE) for a in pair.agents]
    assert cues[0]
    rows, stats = L6.gen_rows(pair, EX, 2, 4, 2)
    assert len(rows) == len(EX) == stats['n'] and all(len(r['cands']) == 2 and len(r['ev']) == 2 and set(r['union']) >= set(r['cands'][0]) and r['owner'] in ('agent0', 'agent1', 'none') for r in rows)
    assert 0.0 <= stats['union_any_correct'] <= 100.0 and stats['size_mean'] >= 1.0
    by_id = {ex['id']: ex for ex in EX}
    params = dict(pair.named_parameters())
    recv_before = {k: v.clone() for k, v in params.items() if 'lora_' in k and '.recv.' in k}
    verify_before = {k: v.clone() for k, v in L6.verify_state(pair).items()}
    assert all(torch.equal(recv_before[k], verify_before[k.replace('.recv.', '.verify.')]) for k in recv_before)   # init = recv
    args = argparse.Namespace(steps=2, batch=2, seed=5, lr_lora=1e-2, lr_adapter=1e-3, warmup=1, weight_decay=0.0, clip=1.0, log_every=1, swap_frac=0.0)
    msgs = []
    tr = L6.train_verifier(pair, rows, by_id, cues[0], yns, args, torch.device('cpu'), msgs.append)
    assert tr['steps_done'] == 2 and tr['steps_planned'] == 2 and tr['sequences'] > 0 and tr['param_counts']['adapter'] == 0 and tr['stopped'] is None and len(tr['history']) == 2
    assert tr['swapped_sequences'] == 0
    trx = L6.train_verifier(pair, rows, by_id, cues[0], yns, argparse.Namespace(**dict(vars(args), swap_frac=1.0)), torch.device('cpu'), msgs.append)
    assert trx['swapped_sequences'] > 0 and trx['sequences'] == tr['sequences'] + trx['swapped_sequences'] and trx['swap_frac'] == 1.0
    p0 = L6.round1_prompts(pair, EX[:1], 0, [rows[0]['ev'][0]])[0]
    p1 = L6.round1_prompts(pair, EX[:1], 0, [rows[1]['ev'][0]], docs_from=EX[1:2])[0]
    q = pair.doc_pieces(EX[:1], 0)[0][0][1]
    assert p0 != p1 and p0[-len(q):] == q and p1[-len(q):] == q
    after = L6.verify_state(pair)
    assert any(not torch.equal(verify_before[k], after[k]) for k in after)
    assert all(torch.equal(recv_before[k], params[k]) for k in recv_before)   # recv untouched
    assert not any(p.requires_grad for p in pair.parameters())
    tr2 = L6.train_verifier(pair, rows, by_id, cues[0], yns, argparse.Namespace(**dict(vars(args), steps=5)), torch.device('cpu'), msgs.append, soft_stop=1.0)
    assert tr2['steps_done'] == 0 and tr2['stopped'] is not None
    recs = {r['id']: dict(id=r['id'], gold=r['gold'], cands=r['cands'], union=r['union'], p=[0.5, 0.6], scores=[[-1.0] * len(r['union'])] * 2) for r in rows}
    evs = L6.evidence_for(pair, EX, batch=2)
    assert set(evs) == {ex['id'] for ex in EX} and all(evs[ex['id']][i] == rows[b]['ev'][i] for b, ex in enumerate(EX) for i in range(2))
    out, records = L6.eval_split(pair, EX, recs, evs, cues[0], yns, 8)
    assert out['n'] == len(EX) == len(records) and set(out['em_variants']) == set(L6.VARIANTS) and 'verifier' in out and out['timing']['verify_s_per_example'] >= 0.0
    assert all(len(r['ver_scores'][0]) == len(r['union']) == len(r['ver_scores'][1]) for r in records) and out['union']['any_correct'] >= out['round1']['either_correct']
    pair2 = _pair()
    L6.enable_verify(pair2)
    L6.load_verify_state(pair2, after)
    own = dict(pair2.named_parameters())
    assert all(torch.equal(own[k], after[k]) for k in after)
    out2, _ = L6.eval_split(pair2, EX, recs, evs, cues[0], yns, 8)
    assert out2['em_variants'] == out['em_variants']


def _fixture(d):
    for name in ('train', 'val', 'fresh'):
        with open(os.path.join(d, name + '.jsonl'), 'w', encoding='utf-8') as fh:
            for ex in EX:
                fh.write(json.dumps(ex) + '\n')


def test_cli_gen_train_gate():
    d = tempfile.mkdtemp()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    try:
        _fixture(d)
        v1 = os.path.join(d, 'V1-same2-s5')
        os.makedirs(v1)
        pair = _pair()
        st = {k: v.detach().cpu() for k, v in pair.named_parameters() if '.adapter.' in k or '.reader.' in k or ('lora_' in k and '.recv.' in k)}
        torch.save(dict(arm='V1', pair='same2', seed=5, models=['tiny-a', 'tiny-a'], state=st, recv=True, config=dict(k=K)), os.path.join(v1, 'trainable.pt'))
        M.jdump(dict(arm='V1', seed=5, val=dict(em=50.0), span=dict(sel_layer=1)), os.path.join(v1, 'summary.json'))
        data = [a for n in ('train', 'val', 'fresh') for a in ('--' + n, os.path.join(d, n + '.jsonl'))]
        common = data + ['--pair', 'same2', '--seed', '5', '--v-from', v1, '--tiny', '--d-e', str(DE), '--n-heads', '2', '--k-beams', '2', '--max-new-tokens', '4', '--gen-batch', '2', '--score-batch', '8']
        gen = os.path.join(d, 'GEN-same2-s5')
        r = subprocess.run([sys.executable, '-B', VERIFIER, '--mode', 'gen', '--out', gen, '--limit-train', '2', '--exclude-seen', '5:1'] + common, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        sg = json.load(open(os.path.join(gen, 'summary.json')))
        assert sg['arm'] == 'GEN' and sg['complete'] and sg['n_rows'] == 2 and sg['prereg'] == 'D8-96' and sg['exclude_seen']['n_seen'] == 1 and sg['exclude_seen']['n_kept'] == 2
        cj = json.load(open(os.path.join(gen, 'cands_train.json')))
        assert cj['seed'] == 5 and len(cj['rows']) == 2 and all(len(r_['ev']) == 2 for r_ in cj['rows']) and cj['exclude_seen']['spec'] == '5:1'
        assert {r_['id'] for r_ in cj['rows']} == {ex['id'] for ex in L6.exclude_seen(EX, '5:1')[0]}
        # L5-style records (the candidates of the GEN rows stand in for the D8-95 B1t records)
        l5 = os.path.join(d, 'B1t-same2-s5')
        os.makedirs(l5)
        rows_all, _ = L6.gen_rows(pair, EX, 2, 4, 2)
        recs = [dict(id=r_['id'], gold=r_['gold'], owner=r_['owner'], cands=r_['cands'], union=r_['union'], p=[0.5, 0.6], scores=[[-1.0] * len(r_['union'])] * 2, lens=[1] * len(r_['union'])) for r_ in rows_all]
        M.jdump(recs, os.path.join(l5, 'preds_val.json'))
        M.jdump(recs, os.path.join(l5, 'preds_fresh.json'))
        M.jdump(dict(arm='B1t', seed=5, complete=True, val=dict(em=40.0, em_variants=dict(select_p=None), union=dict(any_correct=60.0)), fresh=dict(em=40.0)), os.path.join(l5, 'summary.json'))
        ver = os.path.join(d, 'VER-same2-s5')
        r = subprocess.run([sys.executable, '-B', VERIFIER, '--mode', 'train', '--out', ver, '--cands-from', os.path.join(gen, 'cands_train.json'), '--l5-from', l5, '--steps', '2', '--batch', '2',
                            '--warmup', '1', '--log-every', '1', '--keep-preds'] + common, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
        s = json.load(open(os.path.join(ver, 'summary.json')))
        assert s['arm'] == 'VER' and s['complete'] and s['train_complete'] and s['steps'] == 2 and s['prereg'] == 'D8-96' and s['v']['sel_layer'] == 1 and s['cands']['n_rows'] == 2
        assert set(s['val']['em_variants']) == set(L6.VARIANTS) and os.path.isfile(os.path.join(ver, 'trainable.pt')) and os.path.isfile(os.path.join(ver, 'preds_val.json'))
        ck = torch.load(os.path.join(ver, 'trainable.pt'), map_location='cpu', weights_only=False)
        assert ck['arm'] == 'VER' and all('.verify.' in k for k in ck['state']) and len(ck['state']) == s['params']['verify_tensors']
        # gates: the real VER summary (+ the L5 cross-check), GEN, and synthetic VER summaries
        M.jdump(dict(arm='B1t', seed=5, complete=True, val=dict(em=40.0, em_variants=dict(select_p=s['val']['em_variants']['select_p']), union={}), fresh=dict(em=40.0)), os.path.join(l5, 'summary.json'))
        gates = os.path.join(d, 'gates')
        r = subprocess.run([sys.executable, '-B', GATE, '--job', 'VER-same2-s5', '--runs', d, '--l5', d, '--l4', d, '--out', gates], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and 'GATE VER-same2-s5: VERIFIER-' in r.stdout and 'pendingT2' in r.stdout, r.stdout + r.stderr
        assert any(f.startswith('GATE_VER-same2-s5_') for f in os.listdir(d))
        assert G.judge('GEN-same2-s5', d, d, d)['verdict'] == 'GEN-DONE'
        M.jdump(dict(arm='B1t', seed=5, complete=True, val=dict(em=40.0, em_variants=dict(select_p=s['val']['em_variants']['select_p'] + 1.0), union={}), fresh=dict(em=40.0)), os.path.join(l5, 'summary.json'))
        bad = G.judge('VER-same2-s5', d, d, d)
        assert bad['verdict'] == 'INCOMPLETE' and any('L5' in p_ for p_ in bad['problems'])
        syn = os.path.join(d, 'VER-same2-s6')
        os.makedirs(syn)
        def summ(em, sel=55.0, s0=56.0, s1=57.0, tc=True):
            return dict(arm='VER', seed=6, complete=True, train_complete=tc, steps=10, steps_planned=10 if tc else 20,
                        val=dict(em=em, em_variants=dict(select_p=sel, ver_self0=s0, ver_self1=s1, poe_k4=52.0), round1={}, union={}, verifier=dict(by_rel={}, flat_when_ignorant=True), asym={}), fresh=dict(em=em))
        M.jdump(summ(60.0), os.path.join(syn, 'summary.json'))
        g = G.judge('VER-same2-s6', d, d, d)
        assert g['verdict'] == 'VERIFIER-ADDS+pendingT2+BELOW-TEXT+SUM-ADDS' and g['detail']['margin_vs_self'] == 3.0
        t2 = os.path.join(d, 'T2-same2-s6')
        os.makedirs(t2)
        M.jdump(dict(arm='T2', seed=6, complete=True, val=dict(em=55.2), fresh=dict(em=55.0)), os.path.join(t2, 'summary.json'))
        assert G.judge('VER-same2-s6', d, d, d)['verdict'] == 'VERIFIER-ADDS+BELIEF-BEATS-TEXT-R2+BELOW-TEXT+SUM-ADDS'
        M.jdump(summ(62.0, tc=False), os.path.join(syn, 'summary.json'))
        g = G.judge('VER-same2-s6', d, d, d)
        assert g['verdict'] == 'VERIFIER-ADDS+BELIEF-BEATS-TEXT-R2+BEATS-ALL-DOCS+SUM-ADDS' and g.get('notes')
        M.jdump(summ(52.5, s0=53.0, s1=52.0), os.path.join(syn, 'summary.json'))
        assert G.judge('VER-same2-s6', d, d, d)['verdict'] == 'VERIFIER-HURTS+TEXT-BEATS-BELIEF+BELOW-TEXT+SUM-NEUTRAL'
        M.jdump(summ(54.0, s0=53.0, s1=52.0), os.path.join(syn, 'summary.json'))
        assert G.judge('VER-same2-s6', d, d, d)['verdict'] == 'VERIFIER-NEUTRAL+TIE+BELOW-TEXT+SUM-NEUTRAL'
        M.jdump(summ(60.5, s0=59.0), os.path.join(syn, 'summary.json'))
        assert G.judge('VER-same2-s6', d, d, d)['verdict'] == 'VERIFIER-ADDS+BELIEF-BEATS-TEXT-R2+BEATS-TEXT+SUM-NEUTRAL'
        synx = os.path.join(d, 'VERX-same2-s6')
        os.makedirs(synx)
        M.jdump(dict(summ(60.5, s0=59.0), arm='VERX'), os.path.join(synx, 'summary.json'))
        assert G.judge('VERX-same2-s6', d, d, d)['verdict'] == 'VERIFIER-ADDS+BELIEF-BEATS-TEXT-R2+BEATS-TEXT+SUM-NEUTRAL'
        assert G.judge('VER-same2-s5', d, d, d)['verdict'] == 'INCOMPLETE'   # the L5 cross-check mismatch written above still stands
    finally:
        shutil.rmtree(d, ignore_errors=True)
