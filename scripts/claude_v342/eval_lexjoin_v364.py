"""F4 (2026-10-22): make max-precision selection an exact lattice join, measured inference-only on trained checkpoints.
Two changes vs train_v347/v361 rollout_max (literature-round reviews D7 / fusion-D2):
  (a) LEX join: ties in the quantised precision are broken by the quantised mean value (both codebook values, identical at every
      node), not by candidate order -> commutative, associative, idempotent on every coordinate => Ghrist-Riess / CRDT theorems apply.
  (b) OWN-Q: a node's belief uses its own factor exactly as the wire encodes it (quantised once at the origin), so its state is an
      element of the same lattice as the messages (no raw-vs-quantised own echo on cycles).
Modes evaluated on the VALIDATION panel (and fresh): base (trainer's rollout), lex, lex+ownq. Also: math tests (order invariance,
loopy == centralised join on ALL coordinates incl. own echoes) and the tie rate among observed sources.
python eval_lexjoin_v364.py --source <sealed> --checkpoint <final.pt> --trainer <train_v347_maxsel.py> --out <json> [--device cpu]"""
import argparse, json, sys, time, itertools
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', default=''); ap.add_argument('--trainer', required=True)
ap.add_argument('--out', default=''); ap.add_argument('--device', default='cpu'); ap.add_argument('--size', type=int, default=128)
ap.add_argument('--panels', default='val,fresh'); ap.add_argument('--tests-only', action='store_true')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(args.trainer).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location('trainer', args.trainer); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_implicit_model_v192 import latent_quantize
HOPS = (0, 1, 2, 4, 8)

def select_lex(P, E):
    """P,E: [C,N,64]. Join = max precision, ties -> max mean (both codebook values). Order-independent."""
    pmax = P.max(0, keepdim=True).values
    mean = E / P
    key = torch.where(P == pmax, mean, torch.full_like(mean, -float('inf')))
    idx = key.argmax(0, keepdim=True)
    return P.gather(0, idx)[0], E.gather(0, idx)[0]

def rollout_lex(mu, q_learned, mask, edges, rounds=8, stages=HOPS, own_q=True, allow_cycles=False, perm_seed=None):
    n = mu.shape[0]; prior = T.LATENT_PRIOR / (n * 64)
    evid = mask.double() * q_learned.double(); p_own = evid + prior; e_own = evid * mu.double()
    pw, ew, _ = forest.quantize(p_own, e_own)
    p_bel, e_bel = (pw, ew) if own_q else (p_own, e_own)
    graph = ([(min(i, j), max(i, j), w) for i, j, w in edges] if allow_cycles else forest.validate_forest(edges, n)) if edges else []
    directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
    rng = np.random.default_rng(perm_seed) if perm_seed is not None else None
    incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
    if rng is not None:
        incoming = {j: list(rng.permutation(v)) for j, v in incoming.items()}
    mp = [torch.zeros(64, dtype=torch.float64, device=mu.device) for _ in directed]; me = [torch.zeros_like(x) for x in mp]
    def belief():
        ps, es = [], []
        for j in range(n):
            P = torch.stack([p_bel[j]] + [mp[a] for a in incoming[j]])[:, None]; E = torch.stack([e_bel[j]] + [me[a] for a in incoming[j]])[:, None]
            if rng is not None:
                o = rng.permutation(P.shape[0]); P, E = P[o], E[o]
            pj, ej = select_lex(P, E); ps.append(pj[0]); es.append(ej[0])
        P = torch.stack(ps); E = torch.stack(es)
        return torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1), P, E
    feats = {}; raw = {}
    if 0 in stages:
        feats[0], *raw[0] = belief()
    for r in range(1, rounds + 1):
        nm, ne = [], []
        for a, (i, j) in enumerate(directed):
            P = torch.stack([pw[i]] + [mp[b] for b in incoming[i] if directed[b][0] != j])[:, None]
            E = torch.stack([ew[i]] + [me[b] for b in incoming[i] if directed[b][0] != j])[:, None]
            pc, ec = select_lex(P, E); nm.append(pc[0]); ne.append(ec[0])
        mp, me = nm, ne
        if r in stages:
            feats[r], *raw[r] = belief()
    return feats, raw

# ---------------------------------------------------------------- math tests
torch.manual_seed(0); ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS' if cond else 'FAIL'), name, detail, flush=True)
n = 6; prior = T.LATENT_PRIOR / (n * 64)
mu = latent_quantize(torch.rand(n, 64) * 2 - 1)
q = torch.full((n, 64), 2.0); q[:, ::2] = torch.rand(n, 32) * 3.9 + .1          # odd coordinates: exact precision TIES across sources
mask = (torch.rand(n, 64) < .6).float()
cyc = [(0, 1, 1.), (1, 2, 1.), (2, 0, 1.), (2, 3, 1.), (3, 4, 1.), (4, 5, 1.), (5, 2, 1.)]
f0, r0 = rollout_lex(mu, q, mask, cyc, rounds=12, stages=(12,), allow_cycles=True, perm_seed=None)
f1, r1 = rollout_lex(mu, q, mask, cyc, rounds=12, stages=(12,), allow_cycles=True, perm_seed=1)
check('lex join: result independent of candidate order (ties present)', torch.equal(f0[12], f1[12]))
evid = mask.double() * q.double() + prior; pw, ew, _ = forest.quantize(evid, mask.double() * q.double() * mu.double())
Pc, Ec = select_lex(pw[:, None], ew[:, None])                                   # centralised join over all sources (graph connected)
check('loopy lex flooding == centralised join on ALL coordinates and nodes (incl. own echoes)', torch.equal(r0[12][0], Pc.expand(n, 64)) and torch.equal(r0[12][1], Ec.expand(n, 64)))
f8, r8 = rollout_lex(mu, q, mask, cyc, rounds=24, stages=(24,), allow_cycles=True)
check('idempotent: 24 rounds == 12 rounds', torch.equal(f8[24], f0[12]))
tree = [(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 4, 1.), (4, 5, 1.)]
ft, rt = rollout_lex(mu, q, mask, tree, rounds=8, stages=(8,))
check('tree: every node == centralised join after diameter rounds', torch.equal(rt[8][0], Pc.expand(n, 64)))
if args.tests_only or not args.checkpoint:
    print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)

# ---------------------------------------------------------------- inference on a checkpoint
from scripts import mapping_multiroom_data_v293 as data
from scripts import mapping_native_bank_v306 as bank
from scripts import train_mapping_support_native_v306 as base
from scripts.mapping_recognition_metrics_v277 import node_metrics
device = torch.device(args.device); size = args.size
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
for name, key, default in (('READOUT_MODE', 'readout_mode', 'anchored'), ('PRECISION_MODE', 'precision_mode', 'learned')):
    if hasattr(T, name): setattr(T, name, meta.get(key, default))
assert meta.get('fusion') == 'max' and meta.get('readout_mode') == 'posterior_only' and meta.get('wire', 'sealed') == 'sealed'
model = T.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
base_roll = T.make_rollout(forest, latent_quantize, 'max')
KEYS = ('wall_f1', 'union_accuracy', 'unknown_known_fpr', 'own_wall_f1', 'door_free_recall', 'pair_topology_accuracy')
specs = [s for s in base.scene_plan('VALIDATION') if s['size'] == size]
panels = dict(val=specs, fresh=[dict(s, index=s['index'] + 8) for s in specs])
res = dict(checkpoint=args.checkpoint, checkpoint_sha256=T.sha_file(args.checkpoint), tests_ok=bool(ok), panels={})
tie_obs = []; t0 = time.time()
with torch.no_grad():
    for label in args.panels.split(','):
        rows = []
        for sp in panels[label]:
            fx = data.make_fixture(**sp); inp = fx['inputs']
            tg = {k: bank.forest_targets(inp, k)[0] for k in HOPS}
            edges = bank.select_forest(inp['positions_m'], inp['adjacency'], 8.)['edges']
            own = torch.from_numpy(inp['private']).to(device); n = own.shape[0]
            mu, q, mask = model.source(own)
            prior = T.LATENT_PRIOR / (n * 64); ev = mask.double() * q.double()
            pw, ew, _ = forest.quantize(ev + prior, ev * mu.double())
            obs = mask.bool()
            # tie rate: blocks observed by >=2 sources where the max quantised precision is shared by sources with different means
            m_ = ew / pw; ties = 0; multi = 0
            for b in range(64):
                srcs = torch.nonzero(obs[:, b]).flatten()
                if len(srcs) >= 2:
                    multi += 1; pm = pw[srcs, b].max(); top = srcs[pw[srcs, b] == pm]
                    if len(top) >= 2 and (m_[top, b].max() - m_[top, b].min()) > 1e-9: ties += 1
            tie_obs.append((ties, multi))
            fb, _ = base_roll(mu, q, mask, edges, stages=HOPS)
            fl, _ = rollout_lex(mu, q, mask, edges, stages=HOPS, own_q=False)
            fq, _ = rollout_lex(mu, q, mask, edges, stages=HOPS, own_q=True)
            row = dict(spec=sp, modes={})
            for mode, f in (('base', fb), ('lex', fl), ('lex_ownq', fq)):
                hops = {}
                for k in (0, 8):
                    pred = T.predict(model.decoder, torch.cat((f[0], f[k]), -1)).cpu().numpy().reshape(n, size, size)
                    m = node_metrics(pred, inp['private'], tg[k], {'door_mask': fx['evaluation']['door_mask']})
                    hops[str(k)] = {kk: (float(np.mean([r[kk] for r in m if r.get(kk) is not None])) if any(r.get(kk) is not None for r in m) else None) for kk in KEYS}
                row['modes'][mode] = hops
                row['feat_maxdiff_vs_base_R8'] = row.get('feat_maxdiff_vs_base_R8', {}); row['feat_maxdiff_vs_base_R8'][mode] = float((f[8] - fb[8]).abs().max())
            rows.append(row)
            print(label, sp['rooms'], sp['agents'], sp['index'], {m: round(row['modes'][m]['8']['wall_f1'], 4) for m in row['modes']}, f'{time.time()-t0:.0f}s', flush=True)
        summ = {}
        for mode in ('base', 'lex', 'lex_ownq'):
            summ[mode] = {}
            for k in ('0', '8'):
                strata = {}
                for r in rows:
                    strata.setdefault((r['spec']['rooms'], r['spec']['agents']), []).append(r['modes'][mode][k])
                per = [{kk: np.mean([x[kk] for x in v if x[kk] is not None]) for kk in KEYS} for v in strata.values()]
                summ[mode][k] = {kk: float(np.mean([p[kk] for p in per])) for kk in KEYS}
        res['panels'][label] = dict(rows=rows, summary=summ)
        for mode in summ:
            print(label, mode, 'K0', round(summ[mode]['0']['wall_f1'], 4), 'R8', {kk: round(v, 4) for kk, v in summ[mode]['8'].items()}, flush=True)
t = np.array(tie_obs); res['tie_rate_among_multi_source_blocks'] = float(t[:, 0].sum() / max(t[:, 1].sum(), 1))
print('tie rate among multi-source blocks:', res['tie_rate_among_multi_source_blocks'])
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
