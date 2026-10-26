"""probe_A_veto.py (analyst A, 2026-10-26; CPU-only, post-hoc, report-only).
71 % of the K0 exact-surface false negatives are surface voxels the KNOWN head vetoes (predicted unknown). How much of the
exact-vs-tol-1 gap is the known head's border rather than the occupancy head's placement? Re-score the frozen checkpoints with the
veto threshold relaxed: pred = occ-head decision where known_logit >= -t, else unknown (t = 0 is the trainer's predict()).
Reports exact / tol-1 surface F1, variable-structure F1 and the unknown->known FPR guard, K0 and R8, on the VAL fixtures of probe_A_errors.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPTS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
RUNS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--fx', required=True)
ap.add_argument('--runs', default=f'{RUNS}/v429-stack/c2-s342101,{RUNS}/v429-stack/c2-s342202')
ap.add_argument('--ts', default='0,0.5,1,2,4'); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, SCRIPTS)
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v431 as T
import mapping_3d_storey_data_v420 as W


def configure(ck):
    m = ck['meta']
    T.DECODER = m.get('decoder', 'd3'); T.PRECISION = m.get('precision', 'learned'); T.KNOWN_WEIGHT = m.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = m['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(m.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = m.get('decoder_width', 1.0)


def main():
    torch.set_num_threads(args.threads); dims = W.DIMS['R1']; ts = [float(v) for v in args.ts.split(',')]; t0 = time.time()
    fxs = []
    for f in sorted(Path(args.fx).glob('r*-i*.npz')):
        z = np.load(f)
        if 'skipped' not in z.files:
            fxs.append(dict(private=z['private'], reached=z['reached'], edges=[(int(i), int(j), float(w)) for i, j, w in z['edges']],
                            slab=torch.from_numpy(z['slab'].reshape(-1)), obj=torch.from_numpy(z['obj'].reshape(-1)), struct=torch.from_numpy(z['struct'].reshape(-1)), chart_m=float(z['chart_m'])))
    for run in args.runs.split(','):
        ck = torch.load(Path(run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); configure(ck)
        model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval()
        rows = {(k, t): [] for k in (0, 8) for t in ts}; vetoed = [0., 0., 0.]
        with torch.no_grad():
            for fx in fxs:
                T.CHART_M = fx['chart_m']; own = torch.from_numpy(fx['private']); reached = torch.from_numpy(fx['reached'])
                mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'], stages=(0, 8))
                for k in (0, 8):
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); kl, ol = model.decoder(feats[k])
                    if k == 0:
                        surf = T.surface_of(tgt, dims); fnv = surf & (kl < 0)
                        vetoed[0] += float(fnv.sum()); vetoed[1] += float((fnv & (ol >= 0)).sum()); vetoed[2] += float((surf & ((kl < 0) | (ol < 0))).sum())
                    for t in ts:
                        pred = torch.where(kl >= -t, (ol >= 0).to(torch.int8), torch.full_like(ol, -1, dtype=torch.int8))
                        rows[(k, t)].extend(T.metrics3d(pred, tgt, dims, fx['slab'], fx['obj'], fx['struct']))
        print(f'== {Path(run).name}: K0 surface voxels vetoed {vetoed[0]:.0f} of FN {vetoed[2]:.0f}; vetoed with occ_logit >= 0: {vetoed[1] / max(vetoed[0], 1):.3f}  ({time.time()-t0:.0f}s)')
        for k in (0, 8):
            for t in ts:
                a = T.agg(rows[(k, t)])
                print(f"  K{k} t={t:<4} exact {a['surface_f1']:.4f} tol1 {a['surface_f1_tol1']:.4f} var {a['var_occ_f1']:.4f} furn {a['furn_f1']:.4f} fpr {a['unknown_known_fpr']:.4f}", flush=True)


if __name__ == '__main__':
    main()
