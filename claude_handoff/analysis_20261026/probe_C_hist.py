"""probe_C_hist (analyst C, 2026-10-26; report-only diagnosis of the 3S-X checkpoints, CPU only).
For every fixture written by probe_C_gen.py (TRAIN subset and the 24-house VAL panel) and hops K0 / R8:
 (1) per agent, a 2-D histogram of (known_logit, occ_logit) in 64 x 64 bins of width 0.25 over [-8, 8) (clamped), split into 12 voxel
     categories = target class {unknown, free, surface-occupied, interior-occupied} x location {slab, structure zone, furniture region}
     (region = ~slab & ~dilate6(struct), the metrics3d definition). Every threshold pair on the bin edges, and the default rule (0, 0),
     can then be scored offline exactly for surface / var / furniture-region / occupied F1 and the unknown->known FPR (probe_C_analyze.py).
 (2) the fusion decomposition at R8: the max-precision join is re-run with the winning SOURCE AGENT carried along (identical selection,
     checked bit-for-bit against train_3d_v431.rollout_max); every voxel of agent i is classed S (its coordinate's winner is i and i
     observed the voxel), O (winner is another agent that observed it), M (the winner did not observe it: the information is not in the
     fused code); counts of target class x predicted state per class, plus the counterfactual 'copy the winner's own K0 decoding'.
 (3) R8 with the join re-keyed by exact block coverage (payload = the learned wire precision / mean of the selected source): an
     evaluation-time oracle for source ranking.
 (4) trainer metrics3d at K0 / R8 for a cross-check against the run's final.json.
python -B probe_C_hist.py --source <sealed> --scripts <dir> --run <run dir> --fx <fixture dir> --out <dir> [--splits train,val]
"""
import argparse, glob, importlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--scripts', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--fx', required=True); ap.add_argument('--out', required=True); ap.add_argument('--splits', default='train,val'); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--no-coverage', action='store_true')
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, args.scripts)
os.environ['SEALED_SOURCE'] = args.source
torch.set_num_threads(args.threads)
T = importlib.import_module('train_3d_v431')

NB, LO, BW = 64, -8., .25


def setup(meta):
    T.DECODER = meta.get('decoder', 'd3'); T.PRECISION = meta.get('precision', 'learned'); T.KNOWN_WEIGHT = meta.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = meta['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(meta.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = meta.get('decoder_width', 1.0)


def select3(keys, ps, es, ss):
    K = torch.stack(keys, 0); P = torch.stack(ps, 0); E = torch.stack(es, 0); S = torch.stack(ss, 0)
    tie = torch.arange(K.shape[0], dtype=K.dtype).view(-1, 1) * 1e-15
    idx = (K - tie).argmax(0, keepdim=True)
    return K.gather(0, idx)[0], P.gather(0, idx)[0], E.gather(0, idx)[0], S.gather(0, idx)[0]


def rollout_track(mu, q, mask, edges, key_own=None, rounds=8, stages=(0, 8)):
    """T.rollout_max with the winning source index carried along; key_own (n x D) replaces the selection key (default: the precision itself)."""
    n, D = mu.shape
    prior = max(T.LATENT_PRIOR / (n * D), 1e-8)
    evidence_q = mask.double() * q.double()
    p_own = evidence_q + prior; eta_own = evidence_q * mu.double()
    p_wire, e_wire, _ = T.kbit_wire_quantize(p_own, eta_own)
    k_own = p_own if key_own is None else key_own.double()
    k_wire = p_wire if key_own is None else key_own.double()
    src_own = torch.arange(n).view(n, 1).expand(n, D).double()
    graph = [(min(i, j), max(i, j), w) for i, j, w in edges]
    directed = [(i, j) for i, j, _ in graph] + [(j, i) for i, j, _ in graph]
    incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
    z = lambda: torch.zeros(D, dtype=torch.float64)
    mk = [z() for _ in directed]; mp = [z() for _ in directed]; me = [z() for _ in directed]; ms = [torch.full((D,), -1., dtype=torch.float64) for _ in directed]

    def belief():
        ps, es, ss = [], [], []
        for j in range(n):
            _, pj, ej, sj = select3([k_own[j]] + [mk[a] for a in incoming[j]], [p_own[j]] + [mp[a] for a in incoming[j]], [eta_own[j]] + [me[a] for a in incoming[j]], [src_own[j]] + [ms[a] for a in incoming[j]])
            ps.append(pj); es.append(ej); ss.append(sj)
        P = torch.stack(ps); E = torch.stack(es)
        return torch.cat((T.latent_quantize((E / P).float()), (1. / P).log().float()), -1), torch.stack(ss).long()

    feats = {}
    if 0 in stages:
        feats[0] = belief()
    for r in range(1, rounds + 1):
        nk, np_, ne, ns = [], [], [], []
        for a, (i, j) in enumerate(directed):
            sel = [b for b in incoming[i] if directed[b][0] != j]
            kc, pc, ec, sc = select3([k_wire[i]] + [mk[b] for b in sel], [p_wire[i]] + [mp[b] for b in sel], [e_wire[i]] + [me[b] for b in sel], [src_own[i]] + [ms[b] for b in sel])
            nk.append(kc); np_.append(pc); ne.append(ec); ns.append(sc)
        mk, mp, me, ms = nk, np_, ne, ns
        if r in stages:
            feats[r] = belief()
    return feats


def bins(x):
    return ((x - LO) / BW).floor().clamp(0, NB - 1).long()


def main():
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); meta = ck['meta']; setup(meta)
    dims = tuple(meta['dims']); V = int(np.prod(dims)); NX, NY, NZ = dims
    model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.eval()
    kx, ky, kz = NX // T.BX, NY // T.BY, NZ // T.BZ
    cx = torch.arange(NX) // kx; cy = torch.arange(NY) // ky; cz = torch.arange(NZ) // kz
    coord = (cx.view(-1, 1, 1) * (T.BY * T.BZ) + cy.view(1, -1, 1) * T.BZ + cz.view(1, 1, -1)).reshape(-1)       # voxel -> code coordinate (decoder reshape order)
    files = []
    for sp in args.splits.split(','):
        files += sorted(glob.glob(os.path.join(args.fx, f'{sp}-*.npz')))[:args.limit or None]
    checked = False
    for f in files:
        t0 = time.time(); name = Path(f).stem
        if (out / f'{name}.npz').exists():
            continue
        z = np.load(f)
        T.CHART_M = float(z['chart_m'])
        own = torch.from_numpy(z['private']); reached = torch.from_numpy(z['reached']); edges = [(int(i), int(j), float(w)) for i, j, w in z['edges']]
        slab = torch.from_numpy(z['slab'].reshape(-1)); obj = torch.from_numpy(z['obj'].reshape(-1)); struct = torch.from_numpy(z['struct'].reshape(-1))
        n = own.shape[0]
        region = ~slab & ~T.dilate6(struct[None], dims)[0]
        loc = torch.where(slab, 0, torch.where(region, 2, 1))
        known_all = own.reshape(n, -1) >= 0
        res = dict(spec=str(z['spec']), chart_m=float(z['chart_m']))
        with torch.no_grad():
            mu, q, mask = model.source(own)
            feats = rollout_track(mu, q, mask, edges)
            if not checked:                                                                # selection identical to the trainer's join
                ref = T.rollout_max(mu, q, mask, edges, stages=(0, 8))
                assert all(torch.equal(ref[k], feats[k][0]) for k in (0, 8)), 'rollout_track differs from rollout_max'
                checked = True
            H = np.zeros((2, n, 12, NB, NB), np.int32); pred0 = None
            for hi, k in enumerate((0, 8)):
                tgt = T.stream_targets(own, reached[T.HOPS.index(k)])
                kl, ol = model.decoder(feats[k][0])
                pred = torch.where(kl >= 0, (ol >= 0).to(torch.int8), torch.full_like(ol, -1, dtype=torch.int8))
                surf = T.surface_of(tgt, dims)
                tcls = torch.where(tgt < 0, 0, torch.where(tgt == 0, 1, torch.where(surf, 2, 3)))
                cat = tcls * 3 + loc[None]
                idx = ((torch.arange(n).view(-1, 1) * 12 + cat) * NB + bins(kl)) * NB + bins(ol)
                H[hi] = torch.bincount(idx.reshape(-1), minlength=n * 12 * NB * NB).reshape(n, 12, NB, NB).numpy()
                res[f'metrics_K{k}'] = T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct))
                w = feats[k][1][:, coord]                                                  # [n, V] winning source per voxel
                obs = known_all.gather(0, w)
                fcls = torch.where(~obs, 2, torch.where(w == torch.arange(n).view(-1, 1), 0, 1))
                pidx = (pred.long() + 1)                                                   # 0 unknown, 1 free, 2 occ
                res[f'fusion_K{k}'] = torch.bincount((((torch.arange(n).view(-1, 1) * 3 + fcls) * 4 + tcls) * 3 + pidx).reshape(-1), minlength=n * 36).reshape(n, 3, 4, 3).numpy().reshape(-1).tolist()
                if k == 0:
                    pred0 = pred
                else:                                                                      # counterfactual: copy the winner's own K0 decoding
                    cf = pred0.gather(0, w)
                    res['copywinner_K8'] = torch.bincount((((torch.arange(n).view(-1, 1) * 3 + fcls) * 4 + tcls) * 3 + (cf.long() + 1)).reshape(-1), minlength=n * 36).reshape(n, 3, 4, 3).numpy().reshape(-1).tolist()
                    res['metrics_copywinner_K8'] = T.agg(T.metrics3d(cf, tgt, dims, slab, obj, struct))
                    if not args.no_coverage:                                               # coverage-keyed join (oracle source ranking)
                        cov = torch.nn.functional.avg_pool3d((own >= 0).float()[:, None], (kx, ky, kz)).flatten(1)
                        fc = rollout_track(mu, q, mask, edges, key_own=cov, stages=(8,))
                        klc, olc = model.decoder(fc[8][0])
                        predc = torch.where(klc >= 0, (olc >= 0).to(torch.int8), torch.full_like(olc, -1, dtype=torch.int8))
                        res['metrics_covkey_K8'] = T.agg(T.metrics3d(predc, tgt, dims, slab, obj, struct))
                        wc = fc[8][1][:, coord]; obsc = known_all.gather(0, wc)
                        fclsc = torch.where(~obsc, 2, torch.where(wc == torch.arange(n).view(-1, 1), 0, 1))
                        res['fusion_covkey_K8'] = torch.bincount((((torch.arange(n).view(-1, 1) * 3 + fclsc) * 4 + tcls) * 3 + (predc.long() + 1)).reshape(-1), minlength=n * 36).reshape(n, 3, 4, 3).numpy().reshape(-1).tolist()
        np.savez_compressed(out / f'{name}.npz', H=H, res=json.dumps(res))
        print(f"{name} {time.time()-t0:.0f}s K0 {res['metrics_K0']['surface_f1']:.4f} K8 {res['metrics_K8']['surface_f1']:.4f} cw {res['metrics_copywinner_K8']['surface_f1']:.4f}"
              + ('' if args.no_coverage else f" cov {res['metrics_covkey_K8']['surface_f1']:.4f}"), flush=True)


if __name__ == '__main__':
    main()
