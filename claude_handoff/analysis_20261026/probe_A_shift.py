"""probe_A_shift.py (analyst A, 2026-10-26; CPU-only, post-hoc, report-only).

Geometry-controlled version of probe_A_errors' grid-phase split. The same physical (surface, free) boundary pairs are scored with the
whole world translated horizontally by s voxels (s in {0, 1, 2, 4}, applied to x and y together; every agent's private map is rolled
and the wrapped band set to unknown; the radio forest and reach sets are unchanged); the prediction is rolled back and scored against
the ORIGINAL target. A pair's grid class depends on the plane e + s, its geometry does not. So for the same pairs:
  s = 1 flips the parity (inside a compute cell <-> on a compute-cell plane)   -> block-readout effect (a)
  s = 2 moves compute-cell planes inside a code column <-> code-column planes   -> code-column / 32-grid effect (d or the 32x32 stage)
  s = 4 moves code-column planes inside a start cell <-> start-cell planes     -> 16x16 start-grid effect (b)
Pairs within 10 voxels of the chart border are excluded. K0 only.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPTS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
RUNS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fx', required=True)
ap.add_argument('--runs', default=f'{RUNS}/v429-stack/c2-s342101,{RUNS}/v429-stack/c2-s342202')
ap.add_argument('--shifts', default='0,1,2,4'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--margin', type=int, default=10)
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


def shift(m, axis, sign):
    out = torch.zeros_like(m); dst = [slice(None)] * 3; src = [slice(None)] * 3
    if sign > 0:
        dst[axis] = slice(0, -1); src[axis] = slice(1, None)
    else:
        dst[axis] = slice(1, None); src[axis] = slice(0, -1)
    out[tuple(dst)] = m[tuple(src)]
    return out


def roll_world(own, s):
    """own [N, X, Y, Z] int8 -> translated by +s in x and y, wrapped band = unknown."""
    if s == 0:
        return own
    r = torch.roll(own, shifts=(s, s), dims=(1, 2)).clone(); r[:, :s] = -1; r[:, :, :s] = -1
    return r


def cls_of(e):
    return np.where(e % 8 == 0, 0, np.where(e % 8 == 4, 1, np.where(e % 2 == 0, 2, 3)))


NAMES = ('start8', 'code4', 'compute2', 'inside')


def main():
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); dims = W.DIMS['R1']; t0 = time.time(); shifts = [int(v) for v in args.shifts.split(',')]
    fxs = []
    for f in sorted(Path(args.fx).glob('r*-i*.npz')):
        z = np.load(f)
        if 'skipped' not in z.files:
            fxs.append(dict(name=f.stem, private=z['private'], reached=z['reached'], edges=[(int(i), int(j), float(w)) for i, j, w in z['edges']],
                            obj=z['obj'].reshape(dims), struct=z['struct'].reshape(dims), chart_m=float(z['chart_m'])))
    res = {}
    for run in args.runs.split(','):
        ck = torch.load(Path(run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); configure(ck)
        model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval()
        tag = Path(run).name
        # table[cat][s][orig_class][shift_class] = [pairs, wrong]
        table = {cat: {s: np.zeros((4, 4, 2)) for s in shifts} for cat in ('all', 'struct', 'obj')}
        with torch.no_grad():
            for fx in fxs:
                T.CHART_M = fx['chart_m']; own0 = torch.from_numpy(fx['private']); reached = torch.from_numpy(fx['reached'])
                tgt = T.stream_targets(own0, reached[0]).reshape(-1, *dims)
                preds = {}
                for s in shifts:
                    own = roll_world(own0, s)
                    mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'], stages=(0,))
                    p = T.predict(model.decoder, feats[0]).reshape(-1, *dims)
                    preds[s] = torch.roll(p, shifts=(-s, -s), dims=(1, 2)) if s else p
                inner = torch.zeros(dims, dtype=torch.bool); M = args.margin; inner[M:-M, M:-M, :] = True
                for i in range(tgt.shape[0]):
                    t = tgt[i]; known = t >= 0; occ = t == 1; free = t == 0
                    s_surf = occ & T._pool6(free.reshape(1, -1), dims).reshape(dims) & inner
                    for axis in (0, 1):
                        for sign in (1, -1):
                            pairs = s_surf & shift(free, axis, sign)
                            idx = pairs.nonzero(as_tuple=True)
                            if idx[0].numel() == 0:
                                continue
                            e = idx[axis].numpy() + (1 if sign > 0 else 0)
                            c0 = cls_of(e)
                            catm = {'all': np.ones(len(e), bool), 'struct': fx['struct'][idx[0].numpy(), idx[1].numpy(), idx[2].numpy()], 'obj': fx['obj'][idx[0].numpy(), idx[1].numpy(), idx[2].numpy()]}
                            for s in shifts:
                                p = preds[s][i]
                                wrong = (~(p == 1))[idx].numpy() | shift(p == 1, axis, sign)[idx].numpy()
                                cs = cls_of(e + s)
                                for cat, cm in catm.items():
                                    np.add.at(table[cat][s], (c0[cm], cs[cm], 0), 1); np.add.at(table[cat][s], (c0[cm], cs[cm], 1), wrong[cm].astype(float))
                print(f'[A] {tag} {fx["name"]} {time.time()-t0:.0f}s', flush=True)
        res[tag] = {cat: {str(s): table[cat][s].tolist() for s in shifts} for cat in table}
        json.dump(dict(names=NAMES, shifts=shifts, margin=args.margin, tables=res), open(out / 'probe_A_shift.json', 'w'), indent=1)
        for cat in ('all', 'struct', 'obj'):
            print(f'\n== {tag} {cat}: wrong rate of the SAME pairs (rows: class at s = 0) under each shift (column header: its class at that shift)')
            for c0 in range(4):
                line = f'  {NAMES[c0]:9s}'
                for s in shifts:
                    tb = table[cat][s][c0]; cs = int(np.argmax(tb[:, 0]))
                    line += f' | s={s}: ->{NAMES[cs]:8s} n {tb[:, 0].sum():7.0f} wrong {tb[:, 1].sum() / max(tb[:, 0].sum(), 1):.4f}'
                print(line)
    print(f'[A] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
