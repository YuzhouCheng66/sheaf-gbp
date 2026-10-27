"""Tests for g4_exact_codec.py (batch G4-E0, report D8-36). CPU only. Run from the project root:
  D:/anaconda3/envs/jax-gpu/python.exe -B scripts/claude_v342/test_g4_exact_codec.py [--quick]
A. Range coder: random binary / categorical / uniform streams round-trip bit for bit; code length <= ideal + 16 bits; empty stream = 0 B;
   frequency tables sum to 2^20 with every entry >= 1.
B. Memoised W.observe: identical to a direct W.observe call for a consistent world (hit set), a painted W3 world and a damaged world
   (one hit removed) -- the cache must not hide the damage; the W2 event march decides exactly the voxels W.observe visits.
C. One VAL house, all three wires, both tiers, all protocols, two (flat) priors: every message and every belief check exact,
   free recall 1, FPR 0, decoded poses equal the float32 poses.
D. Deliberate breakage must be caught (K0 messages): a decoder that loses one decoded hit (W1), a flipped payload bit (W1, W2, W3), and
   decoder-side 1 cm pose rounding (W2) each produce inexact messages / beliefs, and the batch verdict becomes INCONCLUSIVE; 1 cm poses
   on BOTH ends are repaired by the residual (exact by construction).
E. The 6 demo houses rebuild voxel for voxel (own maps, occ / slab / obj, edges, reached, positions).
F. Verdict arithmetic on synthetic summaries: PASS / FAIL_ii / FAIL_iii / INCONCLUSIVE; literal vs A03-definition reading; fixed cells
   {VAL, fresh} x {A, B}: a missing fresh panel, a skipped demo check (empty), a --fault run and a --*-limit run are INCOMPLETE, never PASS;
   skipped houses per panel are reported (review M1 / M2).
G. Decoder independence (review S1): payloads encoded here are decoded in a FRESH PROCESS that gets only the bytes, the shared map S,
   the public grid cell and the (flat) prior -- a PublicCtx with the render memo and traversal replay switched OFF -- and must equal this
   process's decodes, the generator's own maps (K0) and the R8 targets (CAV), for all three wires.
--quick skips C for W2 (the slow wire) and E."""
import os, pickle, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import g4_exact_codec as G
import mapping_3d_storey_data_v420 as W

QUICK = '--quick' in sys.argv
T0 = time.time()
rng = np.random.default_rng(20261027)


def say(msg):
    print(f'[{time.time() - T0:6.0f}s] {msg}', flush=True)


# ---------------------------------------------------------------- A. range coder
for trial in range(40):
    n = int(rng.integers(0, 4000))
    f0 = rng.integers(1, G.TOT, n); sym = (rng.random(n) < 1 - f0 / G.TOT).astype(np.int64)
    if trial % 5 == 0:
        f0 = np.where(rng.random(n) < .5, 1, G.TOT - 1); sym = rng.integers(0, 2, n)          # extreme, mostly surprising symbols
    cum = np.where(sym == 1, f0, 0); freq = np.where(sym == 1, G.TOT - f0, f0)
    e = G.RangeEncoder(); e.encode(cum.tolist(), freq.tolist(), G.TOT); data = e.finish()
    ideal = float(np.sum(G.TOT_BITS - np.log2(freq)))
    assert np.array_equal(G.RangeDecoder(data).binary(f0.tolist(), G.TOT), sym), trial
    assert len(data) * 8 <= ideal + 16, (trial, len(data) * 8, ideal)
e = G.RangeEncoder(); assert e.finish() == b''
counts = rng.integers(0, 50, (7, 128)); counts[0] = 0; counts[1, :] = 0; counts[1, 5] = 10 ** 9
tab = G.Table(counts, G.alpha_of(128))
assert (tab.freq >= 1).all() and (tab.freq.sum(1) == G.TOT).all()
prior = G.Prior('A', G.zero_counts()); prior.tables['w3u.dy'] = G.Table(np.vstack([counts[:6]]), G.alpha_of(128))
for trial in range(20):                                                 # mixed chunk streams through EncIO / DecIO
    io = G.EncIO(); n = int(rng.integers(1, 500))
    ctx = rng.integers(0, 6, n); s = rng.integers(0, 128, n); u = rng.integers(0, 3, n); b = rng.integers(0, 2, n); bc = rng.integers(0, 512, n)
    io.m('w3u.dy', ctx, s, 'box'); io.uni(3, u, 'residual'); io.m('w1u', bc, b, 'hit'); io.flag(1, 'residual'); io.flag(0, 'residual'); io.uni(1 << 16, [65535], 'pose')
    payload, bits = G.encode_chunks(io.chunks, prior)
    d = G.DecIO(payload, prior)
    assert np.array_equal(d.m('w3u.dy', ctx), s) and np.array_equal(d.uni(3, count=n), u) and np.array_equal(d.m('w1u', bc), b)
    assert d.flag() == 1 and d.flag() == 0 and d.uni(1 << 16)[0] == 65535
    assert len(payload) * 8 <= sum(bits.values()) + 16
for n in (0, 1, 127, 128, 300, 70000):
    v = G.varint(n); assert G.read_varint(v + b'xyz') == (n, len(v))
say('A range coder ok')

# ---------------------------------------------------------------- B. memoised observe and the W2 march
h = G.House(dict(split='VALIDATION', index=0, rooms=1), 'VAL', priors=[])
rays = W.fibonacci_sphere(W.BEAMS)
for a in (0, 3):
    P = G.pose64(h.P32[a]); Hs = h.own[a] == 1
    direct = W.observe(Hs.reshape(G.DIMS).astype(np.uint8), h.cell, P, rays).ravel()
    K, wv = h.rcache.render(Hs, P)
    got = np.full(G.V, -1, np.int8); got[K] = wv
    assert np.array_equal(got, direct) and np.array_equal(direct, h.own[a]), 'memoised render != W.observe or float32 recast inexact'
    kf, kc, lz, plan, boxes = G.describe_w3(h.own[a], None)
    Wp = G.paint(kf, kc, lz, plan, boxes)
    m0 = h.rcache.misses; K2, wv2 = h.rcache.render(Wp, P); assert h.rcache.misses == m0, 'consistent painted world must hit the cache'
    got2 = np.full(G.V, -1, np.int8); got2[K2] = wv2; assert np.array_equal(got2, h.own[a])
    dam = Hs.copy(); dam[np.flatnonzero(Hs)[len(np.flatnonzero(Hs)) // 3]] = False
    K3, wv3 = h.rcache.render(dam, P)
    got3 = np.full(G.V, -1, np.int8); got3[K3] = wv3
    direct3 = W.observe(dam.reshape(G.DIMS).astype(np.uint8), h.cell, P, rays).ravel()
    assert np.array_equal(got3, direct3) and not np.array_equal(got3, h.own[a]), 'damaged world must re-render and differ'
    io = G.EncIO(); grid = np.full(G.V, -1, np.int8)
    G.event_march(io, 'w2u', grid, P, h.cell, h.own[a])
    assert np.array_equal(grid, h.own[a]), 'W2 march must visit exactly the voxels W.observe visits'
    assert sum(len(c[3]) for c in io.chunks) == int((h.own[a] >= 0).sum()), 'one event per visited voxel'
    # recorded traversal replay == direct march, also in a damaged world where rays run past their recorded end
    same = lambda c1, c2: len(c1) == len(c2) and all(np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3]) for x, y in zip(c1, c2))
    travs = {}
    for T_ in (h.own[a], h.own[a], np.where(np.isin(np.arange(G.V), np.flatnonzero(h.own[a] == 1)[::50]), 0, h.own[a]).astype(np.int8)):
        io1 = G.EncIO(); g1 = np.full(G.V, -1, np.int8); G.event_march(io1, 'w2u', g1, P, h.cell, T_)
        io2 = G.EncIO(); g2 = np.full(G.V, -1, np.int8); G.event_march(io2, 'w2u', g2, P, h.cell, T_, travs)
        assert np.array_equal(g1, g2) and same(io1.chunks, io2.chunks), 'traversal replay must equal the direct march'
    assert len(travs) == 3
say('B memoised observe / W2 march / traversal replay ok')

# ---------------------------------------------------------------- C. one house, every wire, tier, protocol, two priors
pa, pb = G.Prior('A', G.zero_counts()), G.Prior('B', G.zero_counts())
for k in pb.tables:                                                     # make B differ from A so the two byte streams differ
    c = np.zeros(G.SPECS[k], np.int64); c[:, 0] = 3
    pb.tables[k] = G.Table(c, G.alpha_of(G.SPECS[k][1]))
wires = ('W1', 'W3') if QUICK else G.WIRES
hc = G.House(dict(split='VALIDATION', index=0, rooms=1), 'VAL', priors=[pa, pb])
timing, f1 = hc.run(wires)
ci = {c: i for i, c in enumerate(G.MSG_COLS)}; ki = {c: i for i, c in enumerate(G.CHK_COLS)}
assert hc.rows and all(r[ci['exact']] for r in hc.rows), hc.errors[:3]
assert hc.checks and all(c[ki['exact']] and c[ki['free_recall']] == 1.0 and c[ki['fpr']] == 0.0 for c in hc.checks)
assert not hc.errors
protos = {(r[ci['proto']], r[ci['tier']]) for r in hc.rows}
assert protos == {('K0', 'u'), ('CD', 'u'), ('CD', 'c'), ('SYNC', 'u'), ('SYNC', 'c'), ('CAV', 'c')}, protos
hops = {c[ki['hop']] for c in hc.checks if c[ki['proto']] == 'SYNC'}; assert hops == {0, 1, 2, 4, 8}, hops
for w in wires:                                                         # the two priors give different bytes but the same maps
    ba = sum(r[ci['bytes']] for r in hc.rows if r[ci['wire']] == w and r[ci['prior']] == 'A')
    bb = sum(r[ci['bytes']] for r in hc.rows if r[ci['wire']] == w and r[ci['prior']] == 'B')
    assert ba != bb, w
    assert all(v['surface_f1'] == 1.0 and v['furn_f1'] in (1.0, None) for v in f1[f'{w}|A'])
n_sync = sum(1 for r in hc.rows if r[ci['proto']] == 'SYNC' and r[ci['wire']] == wires[0] and r[ci['prior']] == 'A' and r[ci['tier']] == 'c')
assert n_sync == 8 * 2 * len(hc.edges)
say(f'C one house exact: {len(hc.rows)} messages, {len(hc.checks)} checks, timing {timing}')

# ---------------------------------------------------------------- D. deliberate breakage must be caught
def broken(fault, wire='W1'):
    hb = G.House(dict(split='VALIDATION', index=0, rooms=1), 'VAL', priors=[pa], fault=fault)
    hb.enc_cache = {}; hb.dec_cache = {}
    hb.sim_k0(wire, pa)
    return hb


hb = broken(dict(wire='W1', kind='decoder_drop_hit'))
assert hb.errors and not all(r[ci['exact']] for r in hb.rows), 'a decoder that drops a hit must be caught'
assert any(not c[ki['exact']] for c in hb.checks)
flips = {}
for w in G.WIRES:                                                       # review S3: a flipped payload bit, every wire
    hf = broken(dict(wire=w, kind='payload_bitflip', nth=3), wire=w)
    assert hf.errors and not all(r[ci['exact']] for r in hf.rows), f'a flipped payload bit must be caught ({w})'
    flips[w] = len(hf.errors)
hd = broken(dict(wire='W2', kind='decoder_pose_round'), wire='W2')
assert hd.errors and not all(r[ci['exact']] for r in hd.rows), 'a decoder that rebuilds poses at 1 cm must be caught'
orig = G.pose64                                                         # 1 cm poses on BOTH ends: the residual must repair it (exact by construction)
G.pose64 = lambda P32: (lambda P: (P.__setitem__((slice(None), slice(0, 2)), np.round(P[:, :2], 2)), P)[1])(orig(P32))
try:
    hp = broken(None, wire='W3')
    assert not hp.errors and all(r[ci['exact']] for r in hp.rows), 'both ends at 1 cm: the residual must keep every map exact'
    assert sum(max(r[ci['residual']], 0) for r in hp.rows) > 0, 'both ends at 1 cm: the residual must be non-empty'
finally:
    G.pose64 = orig
rows_b = hb.rows; S, F_, B_ = G.summarise(rows_b, hb.checks, None, ['VAL'], ['A'])
v = G.verdict(S, all(r[ci['exact']] for r in rows_b), all(c[ki['exact']] for c in hb.checks), True, {})
assert v['verdict'] == 'INCONCLUSIVE' and not v['criterion_i'] and v['verdict_line'].startswith('INCONCLUSIVE'), v['verdict']
say(f"D breakage caught: drop_hit {len(hb.errors)} errors, bitflip {flips} errors, decoder 1 cm poses {len(hd.errors)} errors; "
    f"1 cm poses on both ends repaired by {sum(max(r[ci['residual']], 0) for r in hp.rows)} residual entries; verdict {v['verdict']}")

# ---------------------------------------------------------------- E. demo houses rebuild voxel for voxel
if not QUICK:
    for name in ('r1_i0', 'r1_i1', 'r2_i0', 'r2_i1', 'r3_i0', 'r3_i1'):
        r = G.demo_task(name)
        assert r['ok'], r
    say('E demo houses rebuild exactly')


# ---------------------------------------------------------------- F. verdict arithmetic (fixed cells, INCOMPLETE, two readings)
def fake(own, cav, house, umsg=None, uhouse=None):
    return dict(own_median=own, cav_median=cav, house_cd_cond_median=house, converged_uncond_msg_median=umsg if umsg is not None else cav + 50,
                house_cd_uncond_median=uhouse if uhouse is not None else house + 400, cd_distribute_cond_median=cav - 20)


FULL = dict(limits=False, wires_all=True, skip_demo=False, fault=False, evaluated={'VAL': 24, 'fresh': 23}, skipped={'VAL': 0, 'fresh': 1})
S = {}
for p in ('VAL', 'fresh'):
    for q in ('A', 'B'):
        S[f'{p}|{q}|W1'] = fake(500, 900, 15000); S[f'{p}|{q}|W2'] = fake(300, 500, 9000)
        S[f'{p}|{q}|W3'] = fake(220 if q == 'A' else 225, 340, 6000)
v = G.verdict(S, True, True, True, FULL)
assert v['verdict'] == 'PASS' and v['best_arm'] == 'W3' and v['a03_definition_reading'] == 'PASS' and v['fixed_W3']['literal'] == 'PASS', v['verdict_line']
assert v['houses_skipped'] == {'VAL': 0, 'fresh': 1} and v['houses_evaluated']['fresh'] == 23 and not v['incomplete_reasons']
assert 'A03-definition reading' in v['verdict_line'] and 'literal D8-36 reading' in v['verdict_line']
S2 = dict(S); S2['VAL|A|W3'] = fake(220, 340, 6000, umsg=450)          # literal passes, A03 definition fails (uncond message 450 > 420)
v = G.verdict(S2, True, True, True, FULL)
assert v['verdict'] == 'PASS' and v['a03_definition_reading'] == 'FAIL', v['verdict_line']
S2['VAL|A|W3'] = fake(220, 340, 6000, uhouse=6600)
v = G.verdict(S2, True, True, True, FULL); assert v['a03_definition_reading'] == 'FAIL'
S['fresh|B|W3'] = fake(260, 340, 6000); v = G.verdict(S, True, True, True, FULL); assert v['verdict'] == 'FAIL_ii', v
S['fresh|B|W3'] = fake(240, 340, 6000); v = G.verdict(S, True, True, True, FULL); assert v['verdict'] == 'FAIL_iii', v
S['fresh|B|W3'] = fake(225, 340, 6000)
v = G.verdict(S, False, True, True, FULL); assert v['verdict'] == 'INCONCLUSIVE'
v = G.verdict(S, True, True, False, FULL); assert v['verdict'] == 'INCONCLUSIVE'                      # demo houses did not rebuild
Sv = {k: x for k, x in S.items() if not k.startswith('fresh|')}                                   # review M1: fresh panel all skipped
v = G.verdict(Sv, True, True, True, dict(FULL, evaluated={'VAL': 24, 'fresh': 0}, skipped={'VAL': 0, 'fresh': 24}))
assert v['verdict'] == 'INCOMPLETE' and v['provisional_literal'] != 'PASS' and any('fresh' in r for r in v['incomplete_reasons']), v
v = G.verdict(S, True, True, None, dict(FULL, skip_demo=True)); assert v['verdict'] == 'INCOMPLETE', v     # demo check skipped (empty)
v = G.verdict(S, True, True, True, dict(FULL, fault=True)); assert v['verdict'] == 'INCOMPLETE', v
v = G.verdict(S, True, True, True, dict(FULL, limits=True)); assert v['verdict'] == 'INCOMPLETE' and v['provisional_literal'] == 'PASS', v
v = G.verdict(S, True, True, True, dict(FULL, wires_all=False)); assert v['verdict'] == 'INCOMPLETE', v
say('F verdict arithmetic ok (fixed cells, INCOMPLETE gates, literal vs A03-definition readings)')

# ---------------------------------------------------------------- G. decoder independence in a fresh process (review S1)
CHILD = '''import os, sys, pickle
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import g4_exact_codec as G
d = pickle.load(open(os.path.join(sys.argv[2], 'in.pkl'), 'rb'))
prior = G.Prior('A', G.zero_counts())
pub = G.PublicCtx(d['cell'], G.RenderCache(d['cell'], enabled=False), None)
out = []
for wire, tier, S, data in d['recs']:
    n, off = G.read_varint(data)
    try:
        D, agents, P32, info = G.run_message(G.DecIO(data[off:off + n], prior), wire, tier, S, pub, None)
        out.append((D, info.get('march_render_equal', True), None))
    except Exception as e:
        out.append((None, False, repr(e)))
pickle.dump(dict(out=out, misses=pub.rcache.misses), open(os.path.join(sys.argv[2], 'out.pkl'), 'wb'))
'''
recs = []; want = []
own8 = hc.own; R8 = hc.targets[8]
small = sorted(hc.directed, key=lambda e: (len(G.tree_side(hc.nbrs, e[0], e[1])), e))[:2]
for w in G.WIRES:
    for a in (0, 1):
        rec = hc._encode(w, 'u', None, own8[a], 1 << a); data = rec['bytes']['A']
        D = hc._decode(w, 'u', None, data, pa)[0]
        recs.append((w, 'u', None, data)); want.append((D, own8[a], f'{w} K0 {a}'))
    for u, v_ in small:
        su = G.tree_side(hc.nbrs, u, v_); sv = G.tree_side(hc.nbrs, v_, u)
        Sm = own8[sv].max(0); T = np.maximum(Sm, own8[su].max(0))
        rec = hc._encode(w, 'c', Sm, T, sum(1 << b for b in su)); data = rec['bytes']['A']
        D = hc._decode(w, 'c', Sm, data, pa)[0]
        recs.append((w, 'c', Sm, data)); want.append((D, R8[v_], f'{w} CAV {u}->{v_}'))
tmp = tempfile.mkdtemp(prefix='g4e0_testG_')
try:
    pickle.dump(dict(cell=hc.cell, recs=recs), open(os.path.join(tmp, 'in.pkl'), 'wb'))
    subprocess.run([sys.executable, '-B', '-c', CHILD, HERE, tmp], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    res = pickle.load(open(os.path.join(tmp, 'out.pkl'), 'rb'))
finally:
    for f in ('in.pkl', 'out.pkl'):
        if os.path.exists(os.path.join(tmp, f)):
            os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
assert len(res['out']) == len(want)
for (D2, mre, err), (D1, target, name) in zip(res['out'], want):
    assert err is None and mre, (name, err)
    assert np.array_equal(D2, D1) and np.array_equal(D2, target), f'{name}: fresh-process public-only decode differs'
assert res['misses'] > 0                                                # the child really rendered (no memo, no replay)
say(f'G decoder independence ok: {len(want)} messages decoded in a fresh process with PublicCtx only (no memo, no replay), '
    f'{res["misses"]} direct W.observe renders')
say('ALL TESTS PASSED')
