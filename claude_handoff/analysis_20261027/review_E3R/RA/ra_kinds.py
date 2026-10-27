"""R-A probe (report only): NET vs B6 furniture F1 per furniture KIND (VAL cam_nf, R8), and evidence-contradicting flips.
Per object: its voxels from the frozen generator's rasterize (spec with that object only); scored inside the object's 1-voxel
dilation & the target-known scope & the furniture region (tp / fp / fn pooled per kind). Flip table: NET-occupied voxels whose fused
evidence says strongly free (S x 0.05 <= -thr nats) and their precision, by thr.  Usage: python -B ra_kinds.py --jobs val_r1_i0_s1,...
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.join(HERE, '..', 'inputs')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402
G = E.G; W = G.W
CLASS = dict(bed='box', wardrobe='box', fridge='box', cabinet_tall='box', nightstand='box', cabinet='box', tv_cabinet='box', sink='box', box='box',
             shoe_cabinet='box', bathtub='box', sofa='composite', toilet='composite', counter='overhang', bookshelf='thin', coffee_table='legs',
             dining_table='legs', desk='legs', lamp='cyl', plant='cyl', round_table='cyl', car='car')


def run(job, model, ev, j):
    rest = job.split('_r', 1)[1]; rooms = int(rest.split('_i')[0]); index = int(rest.split('_i')[1].split('_s')[0])
    z = np.load(os.path.join(INP, 'export', job + '.npz'))
    spec = W.world_spec('VALIDATION', index, rooms)
    dims = tuple(z['occ'].shape)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    L = list(z['L_k0_cam_nf'])
    acc = {}; flips = {}
    omasks = []
    for o in spec['objects']:
        m = W.rasterize(dict(spec, objects=[o]), dims)['obj']
        omasks.append((o['kind'], m))
    for cm in F['comps']:
        S, kc = E.direct_sums(L, cm, ev.table)
        T = G.Target(private[cm].max(0), H)
        lg = E.net_logits(model, [S], [kc], 'cpu', 1)[0]
        pn = (E.net_readout(lg, kc, j) == 1); pb = (E.linear_readout(S, kc, ev.theta) == 1)
        scope = T.known & H.region
        for kind, m in omasks:
            zone = G.dil6(m) & scope; tr = T.tr & zone
            for tag, p in (('NET', pn), ('B6', pb)):
                pr = p & zone
                a = acc.setdefault(CLASS.get(kind, kind), {}).setdefault(tag, [0, 0, 0])
                a[0] += int((pr & tr).sum()) * len(cm); a[1] += int((pr & ~tr).sum()) * len(cm); a[2] += int((~pr & tr).sum()) * len(cm)
        Sn = S * 0.05
        for thr in (0.0, 1.0, 2.0, 4.0):
            sel = pn & ~pb & scope & (Sn <= -thr)                       # NET says occupied against free evidence
            f = flips.setdefault(f'occ_vs_free_le_-{thr}', [0, 0])
            f[0] += int((sel & T.tr).sum()); f[1] += int((sel & ~T.tr).sum())
        sel = ~pn & pb & scope & (kc > 0)
        f = flips.setdefault('free_vs_occ', [0, 0]); f[0] += int((sel & ~T.tr).sum()); f[1] += int((sel & T.tr).sum())
        f = flips.setdefault('net_occ_total_in_scope', [0, 0]); f[0] += int((pn & scope & T.tr).sum()); f[1] += int((pn & scope & ~T.tr).sum())
    return dict(job=job, kinds=acc, flips=flips, n_objects=len(omasks))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ck, model = E.load_ckpt(os.path.join(INP, 's342101', 'final.pt'), 'cpu')
    ev = E.Evidence.from_recipe(ck['recipe']['evidence']); j = int(json.load(open(os.path.join(INP, 's342101', 'threshold.json')))['j'])
    od = os.path.join(HERE, 'out_kinds'); os.makedirs(od, exist_ok=True)
    for job in a.jobs.split(','):
        p = os.path.join(od, job + '.json')
        if os.path.exists(p):
            continue
        t0 = time.time()
        try:
            r = run(job, model, ev, j)
        except Exception:
            r = dict(job=job, failed=True, tb=traceback.format_exc()[-2000:])
        r['seconds'] = time.time() - t0
        json.dump(r, open(p, 'w')); print(job, r['seconds'], r.get('failed'), flush=True)


if __name__ == '__main__':
    main()
