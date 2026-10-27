"""rb_wire_cd.py (reviewer R-B, 2026-10-27): honest-wire bytes of the CURRENT learned code family (3S-X 32x32x1, 3S-Z2 64x64x1,
both 8+8 = 16 bit per stalk on the fixed wire), per message and per house for ONE R8 fusion, on the 6 VAL demo houses.
CPU only, one process, no GPU (CUDA_VISIBLE_DEVICES blanked), reads only demo_web3d/data and A01's probe_wire.json.

Same simulation and proxy as A01 probe_wire.py (copied logic): the stalk key is the block coverage (fraction of the column's voxels the
agent knows; 0 iff unobserved, as the OwnedHead's zero precision); select() = max key, tie -> lower source. COUNTS of evidenced stalks
are exact for any learned key that is zero iff unobserved; winner identities (and so the dominance pruning) are a proxy.

Protocols (bytes = stalks * b / 8 + addressing / 8 + 2 B header per message; addressing = min(log2 C(D, k), 2-D context code)):
  fixed8   : 8 synchronous rounds, every directed edge sends D stalks x 16 bit (deployed rollout_max wire)          [exact bytes]
  once     : every directed edge sends the fixed 16-bit message once                                                 [exact bytes]
  sync8_dd : A01 'delta + dominance' over 8 synchronous rounds (re-computed per house from A01's rows)
  cd       : NEW here. Collect-distribute on the radio forest (root = smallest id of each component), each directed edge once:
             collect (child -> parent): every evidenced stalk of the exclude-back message (the sender knows nothing of the parent);
             distribute (parent -> child): only stalks where m(p->c) beats the child's own collect message m(c->p) in select() order
             (the child already holds m(c->p), so its belief select(m(c->p), m(p->c)) is unchanged). Beliefs of every agent are
             asserted equal to the converged synchronous beliefs.
b (payload bit per sent stalk): 13.63 = marginal entropy of 3S-X's 8+8 symbols (analyst B, as trained, F1 unchanged);
  10 = A10's assumption; 7 = 4+4 matched re-quantisation (analyst B on 3S-X: R8 exact -0.4 pp, furniture -1.4 pp; NOT measured on
  3S-Z2). Per-stalk entropy of 3S-Z2's own codes was not measured (no checkpoint on this machine) -> all 3S-Z2 bytes are ESTIMATES.
Usage: python -B rb_wire_cd.py"""
import glob, json, math, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(HERE, '..', '..'))                       # claude_handoff/analysis_20261027
ROOT = os.path.abspath(os.path.join(AN, '..', '..'))
sys.path.insert(0, AN)
from rate_baselines_lib import load, LG  # noqa: E402

DATA = os.path.join(ROOT, 'demo_web3d', 'data')
BITS = (13.63, 10.0, 7.0)
ROUNDS = 8


def ctx2d_bits(mask):
    s = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)); X, Y = s.shape
    def sh(dx, dy):
        o = torch.zeros_like(s)
        xs = slice(max(dx, 0), X + min(dx, 0)); xd = slice(max(-dx, 0), X - max(dx, 0))
        ys = slice(max(dy, 0), Y + min(dy, 0)); yd = slice(max(-dy, 0), Y - max(dy, 0))
        o[xs, ys] = s[xd, yd]; return o
    c = torch.zeros_like(s)
    for d in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        c = c * 2 + sh(*d)
    idx = (c * 2 + s).reshape(-1)
    cnt = torch.bincount(idx, minlength=2 ** 7).reshape(-1, 2).double()
    N = cnt.sum(1); used = N > 0; cnt, N = cnt[used], N[used]
    return float(((LG(N + 1.) - LG(1.)) - (LG(cnt + 0.5) - LG(0.5)).sum(1)).sum() / math.log(2))


def log2_binom(n, k):
    return float((LG(n + 1.) - LG(k + 1.) - LG(n - k + 1.)) / math.log(2))


def addr_bits(sel_mask, G):
    k = int(sel_mask.sum())
    if k == 0:
        return 1.
    return min(log2_binom(G * G, k), ctx2d_bits(sel_mask.reshape(G, G)))


def better(ka, sa, kb, sb):
    return (ka > kb) | ((ka == kb) & (sa < sb))


def run_house(path, G):
    m, own, pred, gt = load(path); n = own.shape[0]; s = 128 // G; D = G * G
    cov = (own >= 0).reshape(n, G, s, G, s, 32).mean((2, 4, 5)).reshape(n, D)
    own_k = np.where(cov > 0, cov, 0.); own_s = np.where(cov > 0, np.arange(n)[:, None], 10 ** 6).astype(np.int64)
    edges = [tuple(e) for e in m['edges']]
    nb = {i: [] for i in range(n)}
    for i, j in edges:
        nb[i].append(j); nb[j].append(i)
    directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
    NONE = (np.zeros(D), np.full(D, 10 ** 6, dtype=np.int64))

    def sel(cands):
        K = np.stack([c[0] for c in cands]); S = np.stack([c[1] for c in cands])
        order = np.lexsort((S, -K), axis=0)[0]
        return K[order, np.arange(D)], S[order, np.arange(D)]

    # converged exclude-back messages by synchronous iteration (fixed point asserted)
    msg = {e: NONE for e in directed}
    for r in range(ROUNDS + 2):
        new = {(i, j): sel([(own_k[i], own_s[i])] + [msg[(k, i)] for k in nb[i] if k != j]) for (i, j) in directed}
        if r >= ROUNDS:
            for e in directed:
                assert np.array_equal(new[e][0], msg[e][0]) and np.array_equal(new[e][1], msg[e][1]), ('not converged', path, G, e)
        msg = new
    belief = [sel([(own_k[j], own_s[j])] + [msg[(k, j)] for k in nb[j]]) for j in range(n)]

    # collect-distribute: BFS parents from the smallest id of each component
    parent = {}; seen = set()
    for r0 in range(n):
        if r0 in seen:
            continue
        seen.add(r0); parent[r0] = None; q = [r0]
        while q:
            u = q.pop(0)
            for v in sorted(nb[u]):
                if v not in seen:
                    seen.add(v); parent[v] = u; q.append(v)
    rows = []
    held = {j: [] for j in range(n)}
    for c in range(n):
        p = parent[c]
        if p is None:
            continue
        up = msg[(c, p)]; ev_up = up[0] > 0
        rows.append(dict(kind='collect', k=int(ev_up.sum()), addr=addr_bits(ev_up, G)))
        held[p].append(up)
        down = msg[(p, c)]
        dom = better(down[0], down[1], up[0], up[1]) & (down[0] > 0)
        rows.append(dict(kind='distribute', k=int(dom.sum()), addr=addr_bits(dom, G)))
        held[c].append((np.where(dom, down[0], 0.), np.where(dom, down[1], 10 ** 6)))
    for j in range(n):
        bj = sel([(own_k[j], own_s[j])] + held[j])
        assert np.array_equal(bj[0], belief[j][0]) and np.array_equal(bj[1], belief[j][1]), ('CD belief differs', path, G, j)
    conv = [int((msg[e][0] > 0).sum()) for e in directed]
    k0 = [int((cov[a] > 0).sum()) for a in range(n)]
    k0_addr = [addr_bits(cov[a] > 0, G) for a in range(n)]
    conv_addr = [addr_bits(msg[e][0] > 0, G) for e in directed]
    return dict(house=m['house'], G=G, D=D, n=n, E=len(directed), cd=rows, k0=k0, k0_addr=k0_addr, conv=conv, conv_addr=conv_addr)


def main():
    a01 = json.load(open(os.path.join(AN, 'probes', 'A01', 'probe_wire.json')))['rows']
    out = dict(rows=[], summary={})
    for G in (32, 64):
        H = []
        for path in sorted(glob.glob(os.path.join(DATA, 'house_*.json'))):
            r = run_house(path, G); out['rows'].append(r)
            a = [x for x in a01 if x['house'] == r['house'] and x['G'] == G][0]
            per = {}
            for b in BITS:
                cd = sum(x['k'] * b / 8 + x['addr'] / 8 + 2 for x in r['cd'])
                sdd = sum(np.asarray(a['delta_dom_session_per_edge']) * b / 8 + np.asarray(a['delta_dom_addr_bits_session_per_edge']) / 8 + 2 * ROUNDS)
                per[str(b)] = dict(cd=float(cd), sync8_dd=float(sdd),
                                   k0_msg_median=float(np.median(np.asarray(r['k0']) * b / 8 + np.asarray(r['k0_addr']) / 8 + 2)),
                                   conv_msg_median=float(np.median(np.asarray(r['conv']) * b / 8 + np.asarray(r['conv_addr']) / 8 + 2)))
            H.append(dict(house=r['house'], E=r['E'], fixed8=ROUNDS * r['E'] * r['D'] * 2, once=r['E'] * r['D'] * 2,
                          cd_stalks=sum(x['k'] for x in r['cd']), cd_collect_stalks=sum(x['k'] for x in r['cd'] if x['kind'] == 'collect'),
                          sync8_dd_stalks=int(np.sum(a['delta_dom_session_per_edge'])), per=per))
            print(G, r['house'], 'ok', H[-1]['cd_stalks'], H[-1]['sync8_dd_stalks'], flush=True)
        S = dict(houses=[h['house'] for h in H],
                 fixed8_median=float(np.median([h['fixed8'] for h in H])), once_median=float(np.median([h['once'] for h in H])),
                 cd_stalks_median=float(np.median([h['cd_stalks'] for h in H])),
                 sync8_dd_stalks_median=float(np.median([h['sync8_dd_stalks'] for h in H])))
        for b in BITS:
            for key in ('cd', 'sync8_dd', 'k0_msg_median', 'conv_msg_median'):
                v = [h['per'][str(b)][key] for h in H]
                S[f'b{b}_{key}_median'] = float(np.median(v)); S[f'b{b}_{key}_mean'] = float(np.mean(v))
        out['summary'][f'G{G}'] = S; out[f'houses_G{G}'] = H
    json.dump(out, open(os.path.join(HERE, 'rb_wire_cd.json'), 'w'), indent=1)
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
