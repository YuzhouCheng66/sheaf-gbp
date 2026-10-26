"""eval_decision_v431.py (Claude, 2026-10-26, batch 3S-DR, evaluation only): re-score a finished checkpoint under two decision rules.

  v420 (default; = train_3d_v431.predict, asserted per scene): known if k >= 0; occupied if known and o >= 0.
  calibrated (constants frozen from 24 TRAIN houses by analyst C before this script existed; the offset matches the loss: the
    occupancy BCE weights target-surface voxels 1 + w = 5 times, so sigma(o) = 0.5 means a calibrated probability of 1/6 and the
    matching threshold is log 5 = 1.61): occupied if o >= OCC_T and k >= KNOWN_OCC_T; free if k >= 0 and not occupied; else unknown.
  occ_only (the fallback pre-registered in D8-31; review M1): known if k >= 0 (the unknown set, hence the FPR, equals v420's by
    construction); occupied if known and o >= OCC_T.

Panels: val (index < 8 per room stratum) and fresh (index 8..15) from W.scene_plan('VALIDATION', 16, R1), exactly the trainer's panels.
Sanity: the v420-rule summary is compared with the run's final.json (val / fresh); the max absolute difference is recorded and must be
<= --sanity-tol (CPU vs GPU convolutions may flip a few voxels at the threshold).
Pooled slab recall at R8 (the batch guard) is computed as in diag_storey_v429 (sum over scenes and nodes).
--train-sweep: on the first 8 TRAIN cache rows per room stratum (private maps hash-checked against the manifest), surface F1 at K0 / R8
for occupancy thresholds -1 .. 3 step 0.25 with the known rule k >= 0 -- the mechanism readout "TRAIN-optimal threshold".
Weights, data, loss and metric definitions are untouched. Writes one JSON to --out (never overwrites).
python eval_decision_v431.py --source <sealed> --run runs/v429-stack/c2-s342101 --out runs/v431-decision/c2-s342101.json [--train-sweep --cache <k1600>]
"""
import argparse, json, multiprocessing, os, sys
from pathlib import Path
import numpy as np, torch

OCC_T, KNOWN_OCC_T = 1.5, -0.5                       # frozen (analyst C, TRAIN houses; D8-31)
SWEEP = [round(float(t), 2) for t in np.arange(-1.0, 3.001, 0.25)]


def decide(k, o, rule):
    if rule == 'v420':
        return torch.where(k >= 0, (o >= 0).to(torch.int8), torch.full_like(o, -1, dtype=torch.int8))
    if rule == 'calibrated':
        occ = (o >= OCC_T) & (k >= KNOWN_OCC_T); known = (k >= 0) | occ
        return torch.where(occ, torch.ones_like(o, dtype=torch.int8), torch.where(known, torch.zeros_like(o, dtype=torch.int8), torch.full_like(o, -1, dtype=torch.int8)))
    if rule == 'occ_only':
        return torch.where(k >= 0, (o >= OCC_T).to(torch.int8), torch.full_like(o, -1, dtype=torch.int8))
    raise ValueError(rule)


def surface_f1_mean(pred, target, surf):
    """metrics3d's surface_f1 averaged over nodes (nodes without known voxels or without any surface / prediction skipped, as agg does),
    without the other metrics (review S7: the TRAIN sweep)."""
    vals = []
    for i in range(pred.shape[0]):
        p, t, sf = pred[i], target[i], surf[i]; known = t >= 0
        if known.sum() == 0:
            continue
        ps = ((p == 1) & known) & ~((t == 1) & ~sf)
        tp = (ps & sf).sum().item(); fp = (ps & ~sf).sum().item(); fn = (~ps & sf).sum().item()
        if tp + fp + fn:
            vals.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(vals)) if vals else None


def configure(T, meta):
    T.DECODER = meta.get('decoder', 'd3'); T.PRECISION = meta.get('precision', 'learned'); T.KNOWN_WEIGHT = meta.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = meta['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(meta.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = meta.get('decoder_width', 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='final.pt'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--trainer', default='train_3d_v431'); ap.add_argument('--sanity-tol', type=float, default=5e-4)   # review S2 (C measured ~3e-5 CPU vs GPU)
    ap.add_argument('--train-sweep', action='store_true'); ap.add_argument('--cache', default='')
    args = ap.parse_args()
    out = Path(args.out); assert not out.exists(), f'{out} exists (never overwrite)'; out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ['SEALED_SOURCE'] = args.source
    import importlib
    T = importlib.import_module(args.trainer)
    import mapping_3d_storey_data_v420 as W
    dims = W.DIMS['R1']; dev = torch.device(args.device)
    run = Path(args.run); ck = torch.load(run / 'checkpoints' / args.ckpt, map_location='cpu', weights_only=False)
    configure(T, ck['meta'])
    model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.to(dev).eval()
    final = json.load(open(run / 'final.json'))                                     # review S1: required
    assert final.get('weights_sha256') == ck.get('weights_sha256'), 'checkpoint weights differ from final.json'
    assert ck['meta'].get('D', T.D) == T.D and list(ck['meta'].get('dims', dims)) == list(dims), 'D / dims (review S5)'
    final_rows = {p: json.load(open(run / 'validation' / f'final-{p}.json'))['rows'] for p in ('val', 'fresh')}   # review S2: per-scene check

    plan = W.scene_plan('VALIDATION', 16, dims)
    panels = {'val': [s for s in plan if s['index'] < 8], 'fresh': [s for s in plan if s['index'] >= 8]}
    if args.train_sweep:
        man = json.load(open(Path(args.cache) / 'manifest.json')); rows = []
        for r in sorted({row['spec']['rooms'] for row in man['rows']}):
            rows += [row for row in man['rows'] if row['spec']['rooms'] == r][:8]
        panels['train24'] = [row['spec'] for row in rows]
    specs = [s for p in panels.values() for s in p]
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, specs, chunksize=1).get(timeout=5400)
    it = iter(fxs); gen = {name: [next(it) for _ in p] for name, p in panels.items()}
    if args.train_sweep:
        for row, fx in zip(rows, gen['train24']):
            assert fx is not None and W.array_hash(fx['private']) == row['private_sha256'], f"cache row {row['file']} does not regenerate"

    res = dict(run=str(run), ckpt=args.ckpt, weights_sha256=ck.get('weights_sha256'), meta={k: ck['meta'].get(k) for k in ('blocks', 'decoder', 'decoder_width', 'wire_bits', 'precision', 'known_weight', 'seed')},
               constants=dict(OCC_T=OCC_T, KNOWN_OCC_T=KNOWN_OCC_T), device=str(dev), panels={}, per_scene={})
    with torch.no_grad():
        for name, fl in gen.items():
            rules = ('v420', 'calibrated', 'occ_only') if name != 'train24' else ('v420',)
            rows_by = {r: {str(k): [] for k in (0, 8)} for r in rules}; slab_acc = {r: [0., 0.] for r in rules}
            sweep = {str(k): {str(t): [] for t in SWEEP} for k in (0, 8)}; scenes = []
            for sp, fx in zip(panels[name], fl):
                if fx is None:
                    continue
                T.CHART_M = fx['chart_m']
                own = torch.from_numpy(fx['private']).to(dev); reached = torch.from_numpy(fx['reached']).to(dev)
                slab = torch.from_numpy(fx['slab'].reshape(-1)).to(dev); obj = torch.from_numpy(fx['obj'].reshape(-1)).to(dev); struct = torch.from_numpy(fx['struct'].reshape(-1)).to(dev)
                mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
                row = dict(spec=sp, chart_m=float(fx['chart_m']))
                for k in (0, 8):
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); kl, ol = model.decoder(feats[k])
                    base = decide(kl, ol, 'v420'); assert torch.equal(base, T.predict(model.decoder, feats[k])), 'v420 rule != trainer predict'
                    for r in rules:
                        pred = base if r == 'v420' else decide(kl, ol, r)
                        a = T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct)); rows_by[r][str(k)].append(a); row[f'{r}_K{k}'] = a
                        if k == 8:
                            to = (tgt == 1) & (tgt >= 0) & slab; slab_acc[r][0] += float(((pred == 1) & to).sum()); slab_acc[r][1] += float(to.sum())
                    if name == 'train24':
                        surf = T.surface_of(tgt, dims)
                        assert abs((surface_f1_mean(base, tgt, surf) or 0.) - (row[f'v420_K{k}']['surface_f1'] or 0.)) < 1e-9, 'fast surface F1 != metrics3d'
                        for t in SWEEP:
                            p = torch.where(kl >= 0, (ol >= t).to(torch.int8), torch.full_like(ol, -1, dtype=torch.int8))
                            sweep[str(k)][str(t)].append(surface_f1_mean(p, tgt, surf))
                scenes.append(row)
            mean = lambda rs, m: (float(np.mean([x[m] for x in rs if x[m] is not None])) if any(x[m] is not None for x in rs) else None)
            res['panels'][name] = {r: dict(summary={k: {m: mean(rows_by[r][k], m) for m in T.METRIC_KEYS} for k in ('0', '8')},
                                           R8_slab_recall=slab_acc[r][0] / max(slab_acc[r][1], 1.)) for r in rules}
            res['panels'][name]['scenes'] = len(scenes)
            res['per_scene'][name] = scenes
            if name == 'train24':
                curve = {k: {t: float(np.mean([v for v in vs if v is not None])) for t, vs in sweep[k].items()} for k in sweep}
                res['panels'][name]['sweep_surface_f1'] = curve
                res['panels'][name]['train_optimal_occ_threshold'] = {k: max(curve[k], key=lambda t: curve[k][t]) for k in curve}
            if name in ('val', 'fresh'):                                            # review S1 / S2: both panels must reconcile
                d = max(abs(res['panels'][name]['v420']['summary'][k][m] - final[name][k][m]) for k in ('0', '8') for m in T.METRIC_KEYS
                        if final[name][k].get(m) is not None and res['panels'][name]['v420']['summary'][k][m] is not None)
                fr = final_rows[name]
                assert len(fr) == len(scenes) and all(json.dumps(a['spec'], sort_keys=True) == json.dumps(b['spec'], sort_keys=True) for a, b in zip(fr, scenes)), f'{name}: scene order'
                ds = max(abs(a['hops'][k]['surface_f1'] - b[f'v420_K{k}']['surface_f1']) for a, b in zip(fr, scenes) for k in ('0', '8') if a['hops'][k]['surface_f1'] is not None)
                res['panels'][name]['sanity_max_abs_diff_vs_final'] = d; res['panels'][name]['sanity_max_abs_diff_per_scene_surface'] = ds
                if d > args.sanity_tol or ds > 4 * args.sanity_tol:
                    res['sanity_failed'] = f'{name}: summary {d:.2e} per-scene {ds:.2e}'
            print(name, json.dumps({r: {k: {m: round(v, 4) for m, v in res['panels'][name][r]['summary'][k].items() if m in ('surface_f1', 'var_occ_f1', 'furn_f1', 'unknown_known_fpr') and v is not None} for k in ('0', '8')} for r in rules}), flush=True)
    json.dump(res, open(out, 'w'), indent=1)                                       # review S4: written before any failure exit
    print('wrote', out, flush=True)
    assert 'sanity_failed' not in res, res.get('sanity_failed')


if __name__ == '__main__':
    main()
