"""Post-hoc evaluation pre-registered in D6-2 (Claude, 2026-10-24): re-evaluate the batch 3D-A finals (train_3d_v400 checkpoints, 64^3)
with train_3d_v401's metrics, which add the tolerance-1 surface F1. Valid because SheafModel3D(64) in v401 is v400 bit-for-bit
(test_3d_v401.py C) and v401 fixtures at 64^3 are v400's (test A1). Reads only; writes one json next to the checkpoint's run directory
under runs/v401-3d128/posthoc/. CPU is enough (no lease, no GPU).
python eval_3d_v401_posthoc.py --run runs/v400-3d/a64-s342101 --out runs/v401-3d128/posthoc/a64-s342101-tol1.json --threads 6 --gen-workers 6
"""
import argparse, json, multiprocessing, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapping_3d_data_v401 as d3
import train_3d_v401 as T


def _fixture(spec):
    return d3.make_fixture(**spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--val-scenes', type=int, default=32); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--gen-workers', type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = Path(args.run); meta = json.load(open(run / 'run.json')); final = json.load(open(run / 'final.json'))
    size, n_ag = meta['size'], meta['agents']
    assert size == 64 and meta['version'] == 400 and final['updates'] == meta['updates'], 'post-hoc evaluation of a complete v400 64^3 run only'
    specs = [dict(split='VALIDATION', index=1_000_000 + k, size=size, agents=n_ag) for k in range(2 * args.val_scenes)]
    t0 = time.time()
    with multiprocessing.get_context('fork').Pool(min(args.gen_workers, len(specs))) as pool:
        fxs = pool.map(_fixture, specs, chunksize=1)
    device = torch.device(args.device)
    ck = torch.load(run / 'checkpoints' / 'final.pt', map_location='cpu')
    model = T.SheafModel3D(size, meta['seed']); model.load_state_dict(ck['state_dict']); model = model.to(device).eval()
    assert model.weights_hash() == final['weights_sha256'], 'loaded weights differ from the recorded final weights'
    out = dict(run=str(run), weights_sha256=final['weights_sha256'], updates=final['updates'], metrics_module='train_3d_v401', panels={})
    with torch.no_grad():
        for label, panel in (('val', fxs[:args.val_scenes]), ('fresh', fxs[args.val_scenes:])):
            rows = []
            for fx in panel:
                own = torch.from_numpy(fx['private']).to(device); reached = torch.from_numpy(fx['reached']).to(device)
                mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
                hops = {}
                for k in T.HOPS:
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); pred = T.predict(model.decoder, feats[k])
                    hops[str(k)] = T.agg(T.metrics3d(pred, tgt, size))
                rows.append(hops)
            summary = {str(k): {m: (float(np.mean([r[str(k)][m] for r in rows if r[str(k)][m] is not None])) if any(r[str(k)][m] is not None for r in rows) else None) for m in T.METRIC_KEYS} for k in T.HOPS}
            # consistency with the run's own final evaluation (same metrics except the added tol-1): surface F1 must agree to 1e-3
            ref = final[label]
            for k in T.HOPS:
                for m in ('surface_f1', 'var_occ_f1', 'occ_f1', 'accuracy', 'unknown_known_fpr'):
                    assert abs(summary[str(k)][m] - ref[str(k)][m]) < 1e-3, (label, k, m, summary[str(k)][m], ref[str(k)][m])
            out['panels'][label] = summary
            print(f"[posthoc] {label}: " + ' | '.join(f"R{k}: surf {summary[str(k)]['surface_f1']:.3f} tol1 {summary[str(k)]['surface_f1_tol1']:.3f} var {summary[str(k)]['var_occ_f1']:.3f}" for k in T.HOPS), flush=True)
    out['seconds'] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out, 'w'), indent=1)
    print('[posthoc] done', f'{out["seconds"]:.0f}s')


if __name__ == '__main__':
    main()
