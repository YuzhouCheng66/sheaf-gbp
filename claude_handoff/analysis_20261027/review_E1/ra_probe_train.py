"""R-A (E1 review) CPU probe, TRAIN houses only. Imports the FROZEN g4_noise_panels (read only; run with python -B so no .pyc).
Stage 1: per TRAIN house, V3 integrator, single-mechanism / leave-one-out tiers of the cam range model (sigma, drop, early return)
and the per-frame pose error, R8 = direct component sum (== relayed sum, asserted in E1), scored vs the clean truth exactly as E1.
Saves the cam_nf V3 maps + truth for the calibration stage (stage 2) to an npz OUTSIDE the repo (--npz-dir).
usage: python -B ra_probe_train.py --houses 1:10,2:10 --seed 7 --out DIR --npz-dir SCRATCH
"""
import os, sys, json, time, argparse
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import numpy as np
import g4_noise_panels as G
W = G.W

Z = dict(G._Z)
TIERS = {
    'clean': dict(Z),
    'sig_only': dict(Z, k=0.015),
    'drop_only': dict(Z, p_drop=0.10),
    'out_only': dict(Z, p_out=0.005),
    'range_only': dict(Z, **G._CAM_RANGE),
    'pose_only': dict(Z, **G._CAM_POSE),
    'cam_nf': dict(Z, **G._CAM_RANGE, **G._CAM_POSE),
    'nf_minus_sig': dict(Z, p_drop=0.10, p_out=0.005, **G._CAM_POSE),
    'nf_minus_drop': dict(Z, k=0.015, p_out=0.005, **G._CAM_POSE),
    'nf_minus_out': dict(Z, k=0.015, p_drop=0.10, **G._CAM_POSE),
}
for t in TIERS.values():
    assert set(t) == set(G._Z)
assert TIERS['cam_nf'] == G.CONDITIONS['cam_nf'] and TIERS['range_only'] == G.CONDITIONS['range_only']


def run_house(rooms, index, seed, npz_dir):
    fx = W.make_fixture('TRAIN', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    H = G.House(occ, fx['slab'], fx['obj'], cell)
    F = G.forest(fx['edges'], n)
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    T8 = G.stream_targets(private, reached8)
    K0T = [G.Target(private[a], H) for a in range(n)]
    mults = [(private[cm] >= 0).sum(0) for cm in F['comps']]
    R8T = [G.Target(T8[cm[0]], H, mult=mults[ci]) for ci, cm in enumerate(F['comps'])]
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    res = dict(house=f'TRAIN_r{rooms}_i{index}', seed=seed, comps=F['comps'], n_poses=int(traj.shape[0]), tiers={})
    for tn, c in TIERS.items():
        t0 = time.time(); L = []
        Lv1 = []
        for a in range(n):
            scans = []
            for j in range(traj.shape[0]):
                hs, ls = traces[a][j]
                scan, _ = G.measure(c, hs, ls, traj[j, a], G.pose_draws(seed, rooms, 10_000 + index, a, j, len(rays)), ds, rays)
                scans.append(scan)
            ints = ('V3', 'V1') if tn in ('cam_nf', 'clean') else ('V3',)
            m = G.integrate_agent(shape, cell, ds, scans, ints)
            L.append(m['V3'])
            if 'V1' in m:
                Lv1.append(m['V1'])
        rec = {}
        for I, maps in (('V3', L), ('V1', Lv1)):
            if not maps:
                continue
            k0 = [G.score(G.tern(maps[a]), K0T[a], strata=True) for a in range(n)]
            r8 = []
            for ci, cm in enumerate(F['comps']):
                S = np.zeros(shape, np.int32); K = np.zeros(shape, bool)
                for a in cm:
                    s, k = G.sum_payload(maps[a]); S += s; K |= k
                r8.append([G.score(G.sum_tern((S, K)), R8T[ci], strata=True), len(cm)])
            rec[I] = dict(K0=k0, R8=r8)
        res['tiers'][tn] = rec
        res['tiers'][tn]['sec'] = time.time() - t0
        if tn in ('cam_nf', 'clean') and npz_dir:
            np.savez_compressed(os.path.join(npz_dir, f'{res["house"]}_s{seed}_{tn}.npz'), L=np.stack(L), Lv1=np.stack(Lv1),
                                private=private, occ=occ, slab=fx['slab'], obj=fx['obj'], cell=cell,
                                comps=np.array([ci for a in range(n) for ci, cm in enumerate(F['comps']) if a in cm]))
        print(res['house'], tn, round(time.time() - t0, 1), 's',
              'R8 furn', round(G._pooled_f1(G._expand(rec['V3']['R8'])), 4), 'K0 furn', round(G._pooled_f1(rec['V3']['K0']), 4), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--houses', required=True); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', required=True); ap.add_argument('--npz-dir', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for h in a.houses.split(','):
        rooms, index = map(int, h.split(':'))
        r = run_house(rooms, index, a.seed, a.npz_dir)
        with open(os.path.join(a.out, f'{r["house"]}_s{a.seed}.json'), 'w') as fh:
            json.dump(r, fh)


if __name__ == '__main__':
    main()
