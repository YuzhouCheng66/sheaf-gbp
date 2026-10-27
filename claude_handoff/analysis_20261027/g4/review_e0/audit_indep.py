"""Reviewer audit (G4-E0): is the decoder independent of encoder-side state?
Phase 'enc': run K0, CD (u, c) and CAV for W1/W2/W3 on one VAL house exactly as the batch does (render cache + traversal replay ON),
record every (wire, tier, proto, u, v, S, payload, decoded map) and pickle it.
Phase 'dec' (a NEW process): rebuild the fixture independently, decode every recorded payload with a stub that exposes ONLY the public
grid cell (no own maps, no poses, no caches: RenderCache disabled, traversal replay off) and compare (a) with the phase-1 decode,
(b) K0 with the generator's own maps, (c) CAV with the R8 target, (d) CD final beliefs with the R8 target.
Usage: python -B audit_indep.py enc|dec OUTDIR"""
import os, sys, pickle, time
sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import numpy as np
import g4_exact_codec as G
import mapping_3d_storey_data_v420 as W

phase, out = sys.argv[1], sys.argv[2]
SCENE = dict(split='VALIDATION', index=0, rooms=1)
PRIOR = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e0_smoke2', 'prior_A.npz')
T0 = time.time()
say = lambda m: print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)

if phase == 'enc':
    pA = G.Prior.load('A', PRIOR)
    h = G.House(SCENE, 'VAL', priors=[pA], render_cache=True)
    recs = []; ctx = {}
    o_send, o_dec = h.send, h._decode
    def send(wire, tier, prior, content, Sst, proto, rnd, u, v):
        ctx.update(proto=proto, u=u, v=v)
        return o_send(wire, tier, prior, content, Sst, proto, rnd, u, v)
    def dec(wire, tier, S, data, prior):
        r = o_dec(wire, tier, S, data, prior)
        recs.append(dict(wire=wire, tier=tier, proto=ctx['proto'], u=ctx['u'], v=ctx['v'], S=None if S is None else S.copy(), data=data,
                         D=None if r[0] is None else r[0].copy(), exact_phase1=None))
        return r
    h.send, h._decode = send, dec
    for wire in G.WIRES:
        t0 = time.time(); h.enc_cache = {}; h.dec_cache = {}
        h.sim_k0(wire, pA); h.sim_cd(wire, 'u', pA); h.sim_cd(wire, 'c', pA); h.sim_cav(wire, pA)
        say(f'{wire} encoded+decoded {sum(1 for r in recs if r["wire"] == wire)} messages in {time.time() - t0:.0f}s, errors so far {len(h.errors)}')
    ci = {c: i for i, c in enumerate(G.MSG_COLS)}
    say(f'phase-1 rows {len(h.rows)} exact {sum(r[ci["exact"]] for r in h.rows)} checks {len(h.checks)} ok {sum(c[8] for c in h.checks)} '
        f'render hits {h.rcache.hits} misses {h.rcache.misses}')
    pickle.dump(dict(cell=h.cell, recs=recs, sha=G.CODE_SHA), open(os.path.join(out, 'audit_indep_records.pkl'), 'wb'))
else:
    d = pickle.load(open(os.path.join(out, 'audit_indep_records.pkl'), 'rb'))
    assert d['sha'] == G.CODE_SHA
    pA = G.Prior.load('A', PRIOR)
    fx = W.make_fixture(SCENE['split'], SCENE['index'], SCENE['rooms'])
    own = fx['private'].reshape(8, G.V).astype(np.int8); R8 = np.asarray(fx['reached'], bool)[list(G.HOPS).index(8)]
    T8 = np.stack([own[R8[a]].max(0) for a in range(8)])
    assert tuple(fx['cell']) == tuple(d['cell'])
    class Public:                                   # only the public grid; any other attribute access raises AttributeError
        __slots__ = ('cell', 'rcache', 'travs')
    pub = Public(); pub.cell = fx['cell']; pub.rcache = G.RenderCache(fx['cell'], enabled=False); pub.travs = None
    nbrs = {a: [] for a in range(8)}
    for i, j, _ in fx['edges']:
        nbrs[int(i)].append(int(j)); nbrs[int(j)].append(int(i))
    bad = []; n = 0; cd = {}
    for r in d['recs']:
        k = (r['wire'], r['tier'], r['proto'], r['u'], r['v'])
        if k in cd or (r['proto'] == 'CD' and (r['wire'], r['tier'], r['u'], r['v']) in cd):
            continue
        n += 1
        nn, off = G.read_varint(r['data']); payload = r['data'][off:off + nn]
        try:
            D, agents, P32, info = G.run_message(G.DecIO(payload, pA), r['wire'], r['tier'], r['S'], pub, None)
        except Exception as e:
            bad.append((k, 'EXC ' + repr(e)[:200])); continue
        if r['D'] is None or not np.array_equal(D, r['D']):
            bad.append((k, 'differs from phase-1 decode'))
        if r['wire'] == 'W2' and not info.get('march_render_equal', False):
            bad.append((k, 'W2 march != render'))
        if r['proto'] == 'K0' and not np.array_equal(D, own[r['u']]):
            bad.append((k, 'K0 != generator own map'))
        if r['proto'] == 'CAV' and not np.array_equal(D, T8[r['v']]):
            bad.append((k, 'CAV != R8 target'))
        if r['proto'] == 'CD':
            cd[(r['wire'], r['tier'], r['u'], r['v'])] = D
    for wire in G.WIRES:
        for tier in ('u', 'c'):
            for v in range(8):
                b = own[v].copy()
                for u in nbrs[v]:
                    b = np.maximum(b, cd[(wire, tier, u, v)])
                if not np.array_equal(b, T8[v]):
                    bad.append(((wire, tier, 'CD-final', v), 'belief != R8 target'))
    say(f'decoded {n} distinct messages in a fresh process with a public-only stub (no cache, no replay): {len(bad)} problems; '
        f'renders {pub.rcache.misses} ({pub.rcache.seconds:.0f}s)')
    for b in bad[:20]:
        say(f'  {b}')
    say('AUDIT OK' if not bad else 'AUDIT FAILED')
