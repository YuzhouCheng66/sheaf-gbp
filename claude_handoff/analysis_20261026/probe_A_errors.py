"""probe_A_errors.py (analyst A, 2026-10-26; CPU-only, post-hoc, report-only; NOT a pre-registered criterion).

Question: the K0 exact surface F1 is ~.904 while tol-1 is ~.955. Where does the one-voxel misplacement come from?
  (a) block readout (no conv at voxel resolution; 1x1 conv + depth_to_space(2) from the 64x64x16 compute grid),
  (b) the 16x16 start grid (32x32 code folded into channels, then trilinear upsampling),
  (c) decoder depth, (d) the code itself.
Probe: for every (target surface voxel v, known-free 6-neighbour n) pair -- the exact place where a surface boundary sits -- the
pair is "wrong" if v is not predicted occupied or n is predicted occupied (exactly the voxels that the exact surface F1 counts as FN
or FP and tol-1 forgives). The boundary between v and n lies at a grid plane e = max(v_a, n_a) along axis a. We split the pair error
rate by which decoder grid that plane is a cell boundary of:
  horizontal axes (x, y): e % 8 == 0  -> boundary of the 16x16 START cell (8 voxels) and of the code column and of the compute cell
                          e % 8 == 4  -> code-column boundary (4 voxels, 32x32 code) and compute-cell boundary, inside a start cell
                          e % 4 == 2  -> compute-cell boundary only (2 voxels), inside a code column
                          e odd       -> INSIDE a compute cell: the two voxels come out of the SAME 72-d feature via the 1x1 readout
  vertical axis z:        e % 4 == 0  -> start-grid z boundary (16x16x8 -> 4 voxels); e % 4 == 2 -> compute-cell boundary; e odd -> inside
Signatures: (a) block readout -> pairs inside a compute cell (e odd) err more than pairs across (e even); (b) start grid -> e%8==0 errs
more than e%8==4; (d) code -> code-column boundaries (e%4==0) err more than e%4==2; (c) depth / generic smoothness -> no grid phase
effect. Also: the direction of the misplacement (into free space / thickening / into solid / hole) and the error budget by class.
Usage (uk-linux, CPU): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python probe_A_errors.py --source <sealed> --out <dir>
"""
import argparse, json, multiprocessing, os, sys, time
from pathlib import Path
import numpy as np, torch

SCRIPTS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
RUNS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--runs', default=f'{RUNS}/v429-stack/c2-s342101,{RUNS}/v429-stack/c2-s342202')
ap.add_argument('--indices', default='0,1,2', help='VAL indices per room stratum (VAL = 0..7)')
ap.add_argument('--hops', default='0,8'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--threads', type=int, default=4)
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


def load_fixtures(plan, fxdir, workers):
    fxdir.mkdir(parents=True, exist_ok=True)
    todo = [sp for sp in plan if not (fxdir / f"r{sp['rooms']}-i{sp['index']}.npz").exists()]
    if todo:
        with multiprocessing.get_context('spawn').Pool(min(workers, len(todo))) as pool:
            got = pool.map_async(T._fixture, todo, chunksize=1).get(timeout=3600)
        for sp, fx in zip(todo, got):
            if fx is None:
                np.savez_compressed(fxdir / f"r{sp['rooms']}-i{sp['index']}.npz", skipped=np.array(1)); continue
            np.savez_compressed(fxdir / f"r{sp['rooms']}-i{sp['index']}.npz", private=fx['private'], reached=fx['reached'], edges=np.array(fx['edges'], dtype=np.float64),
                                slab=fx['slab'], obj=fx['obj'], struct=fx['struct'], chart_m=np.array(fx['chart_m']), spec=np.array(json.dumps(sp)))
    out = []
    for sp in plan:
        z = np.load(fxdir / f"r{sp['rooms']}-i{sp['index']}.npz")
        if 'skipped' in z.files:
            continue
        out.append(dict(spec=sp, private=z['private'], reached=z['reached'], edges=[(int(i), int(j), float(w)) for i, j, w in z['edges']],
                        slab=z['slab'], obj=z['obj'], struct=z['struct'], chart_m=float(z['chart_m'])))
    return out


def shift(m, axis, sign):
    """out[v] = m[v + sign e_axis] (False beyond the border); m: [X, Y, Z]."""
    out = torch.zeros_like(m); dst = [slice(None)] * 3; src = [slice(None)] * 3
    if sign > 0:
        dst[axis] = slice(0, -1); src[axis] = slice(1, None)
    else:
        dst[axis] = slice(1, None); src[axis] = slice(0, -1)
    out[tuple(dst)] = m[tuple(src)]
    return out


def edge_class(dims, axis, sign):
    """per voxel v: class of the grid plane between v and v + sign e_axis."""
    idx = torch.arange(dims[axis]); e = idx + (1 if sign > 0 else 0)
    if axis < 2:
        cls = torch.where(e % 8 == 0, 0, torch.where(e % 8 == 4, 1, torch.where(e % 2 == 0, 2, 3)))
    else:
        cls = torch.where(e % 4 == 0, 0, torch.where(e % 2 == 0, 2, 3))
    shape = [1, 1, 1]; shape[axis] = dims[axis]
    return cls.reshape(shape).expand(*dims)


XY_CLASSES = ('start8', 'code4', 'compute2', 'inside')
Z_CLASSES = ('start4', None, 'compute2', 'inside')


def analyse_node(t, p, dims, slab, obj, struct, acc):
    """t, p: int8 [X, Y, Z] target / prediction of one node; slab / obj / struct bool [X, Y, Z]."""
    known = t >= 0; occ = t == 1; free = t == 0
    s = occ & T._pool6(free.reshape(1, -1), dims).reshape(dims)
    po = (p == 1) & known; pocc = p == 1
    tp = s & po; fn = s & ~po; fp = po & free
    near_any = T._pool6((pocc & T._pool6(known.reshape(1, -1), dims).reshape(dims)).reshape(1, -1), dims).reshape(dims)
    near_s = T._pool6(s.reshape(1, -1), dims).reshape(dims)
    def add(key, val):
        acc[key] = acc.get(key, 0.) + float(val)
    add('surface', s.sum()); add('tp', tp.sum()); add('fn', fn.sum()); add('fp', fp.sum())
    add('fn_near', (fn & near_any).sum()); add('fp_near', (fp & near_s).sum())
    add('fn_pred_unknown', (fn & (p < 0)).sum()); add('fn_pred_free', (fn & (p == 0)).sum())
    for name, cat in (('struct', struct), ('slab', slab), ('obj', obj)):
        add(f'surface_{name}', (s & cat).sum()); add(f'fn_{name}', (fn & cat).sum()); add(f'fn_near_{name}', (fn & near_any & cat).sum())
    fpn = fp & near_s
    # which class of surface each near FP touches (first match), and its axis
    for name, cat in (('struct', struct), ('slab', slab), ('obj', obj)):
        add(f'fp_near_{name}', (fpn & T._pool6((s & cat).reshape(1, -1), dims).reshape(dims)).sum())
    # facing axis of surface voxels
    free_xy = shift(free, 0, 1) | shift(free, 0, -1) | shift(free, 1, 1) | shift(free, 1, -1)
    free_z = shift(free, 2, 1) | shift(free, 2, -1)
    for name, m in (('xy_only', s & free_xy & ~free_z), ('z_only', s & free_z & ~free_xy), ('both', s & free_xy & free_z)):
        add(f'surface_face_{name}', m.sum()); add(f'fn_face_{name}', (m & fn).sum()); add(f'fn_near_face_{name}', (m & fn & near_any).sum())
    # boundary pairs (v surface, n = v + sign e_a known free)
    for axis in (0, 1, 2):
        for sign in (1, -1):
            n_free = shift(free, axis, sign); pairs = s & n_free
            n_pocc = shift(pocc, axis, sign); solid = shift(occ, axis, -sign); solid_pocc = shift(pocc, axis, -sign) & solid
            v_ok = po
            wrong = pairs & (~v_ok | n_pocc)
            into_free = pairs & ~v_ok & n_pocc                       # surface displaced one voxel into free space
            thick = pairs & v_ok & n_pocc                            # drawn one voxel too thick (into free)
            into_solid = pairs & ~v_ok & ~n_pocc & solid_pocc        # surface eroded / displaced into the solid
            hole = pairs & ~v_ok & ~n_pocc & ~solid_pocc             # missing, no one-voxel neighbour along this axis
            cls = edge_class(dims, axis, sign)
            names = XY_CLASSES if axis < 2 else Z_CLASSES
            grp = 'xy' if axis < 2 else 'z'
            for ci, cname in enumerate(names):
                if cname is None:
                    continue
                c = cls == ci
                for key, m in (('pairs', pairs), ('wrong', wrong), ('into_free', into_free), ('thick', thick), ('into_solid', into_solid), ('hole', hole)):
                    add(f'pair_{grp}_{cname}_{key}', (m & c).sum())
                for name, cat in (('struct', struct), ('obj', obj), ('slab', slab)):
                    add(f'pair_{grp}_{cname}_{name}_pairs', (pairs & c & cat).sum()); add(f'pair_{grp}_{cname}_{name}_wrong', (wrong & c & cat).sum())
    return acc


def main():
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dims = W.DIMS['R1']; idxs = [int(v) for v in args.indices.split(',')]; hops = [int(v) for v in args.hops.split(',')]
    plan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] in idxs]
    t0 = time.time()
    fxs = load_fixtures(plan, out / 'fx', args.workers)
    print(f'[A] fixtures {len(fxs)} in {time.time()-t0:.0f}s', flush=True)
    results = {}
    for run in args.runs.split(','):
        ck = torch.load(Path(run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); configure(ck)
        model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval()
        tag = Path(run).name; acc = {k: {} for k in hops}; per_scene = []
        with torch.no_grad():
            for fx in fxs:
                ts = time.time(); T.CHART_M = fx['chart_m']
                own = torch.from_numpy(fx['private']); reached = torch.from_numpy(fx['reached'])
                slab = torch.from_numpy(fx['slab'].reshape(-1)); obj = torch.from_numpy(fx['obj'].reshape(-1)); struct = torch.from_numpy(fx['struct'].reshape(-1))
                mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
                row = dict(rooms=fx['spec']['rooms'], index=fx['spec']['index'], chart_m=fx['chart_m'])
                for k in hops:
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); pred = T.predict(model.decoder, feats[k])
                    row[f'K{k}'] = T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct))
                    sa = {}
                    for i in range(pred.shape[0]):
                        analyse_node(tgt[i].reshape(dims), pred[i].reshape(dims), dims, slab.reshape(dims), obj.reshape(dims), struct.reshape(dims), sa)
                    for key, v in sa.items():
                        acc[k][key] = acc[k].get(key, 0.) + v
                    row[f'K{k}_counts'] = sa
                per_scene.append(row)
                print(f"[A] {tag} r{row['rooms']} i{row['index']} L {row['chart_m']:.1f} " + ' '.join(f"K{k} exact {row[f'K{k}']['surface_f1']:.3f} tol1 {row[f'K{k}']['surface_f1_tol1']:.3f}" for k in hops) + f' {time.time()-ts:.0f}s', flush=True)
        results[tag] = dict(totals={str(k): acc[k] for k in hops}, per_scene=per_scene,
                            mean={str(k): {m: float(np.mean([r[f'K{k}'][m] for r in per_scene if r[f'K{k}'][m] is not None])) for m in ('surface_f1', 'surface_f1_tol1', 'var_occ_f1')} for k in hops})
        json.dump(results, open(out / 'probe_A_errors.json', 'w'), indent=1)
    # compact report
    for tag, r in results.items():
        for k in hops:
            a = r['totals'][str(k)]
            print(f"\n== {tag} K{k}: mean exact {r['mean'][str(k)]['surface_f1']:.4f} tol1 {r['mean'][str(k)]['surface_f1_tol1']:.4f}")
            print(f"  surface {a['surface']:.0f} tp {a['tp']:.0f} fn {a['fn']:.0f} (near {a['fn_near']/a['fn']:.3f}, pred-unknown {a['fn_pred_unknown']/a['fn']:.3f}) fp {a['fp']:.0f} (near {a['fp_near']/a['fp']:.3f})")
            for grp, names in (('xy', XY_CLASSES), ('z', Z_CLASSES)):
                for c in names:
                    if c is None:
                        continue
                    P = a[f'pair_{grp}_{c}_pairs']
                    print(f"  {grp:2s} {c:9s} pairs {P:9.0f} wrong {a[f'pair_{grp}_{c}_wrong']/P:.4f} into_free {a[f'pair_{grp}_{c}_into_free']/P:.4f} thick {a[f'pair_{grp}_{c}_thick']/P:.4f} into_solid {a[f'pair_{grp}_{c}_into_solid']/P:.4f} hole {a[f'pair_{grp}_{c}_hole']/P:.4f}")
    print(f'[A] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
