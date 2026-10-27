"""c1_verify.py (C1 synthesis, 2026-10-27): independent CPU re-checks for the round-3 synthesis.
(1) free-space loophole: an 'S-only' prediction (target-occupied -> occupied, everything else unknown) under the training metric
    (surface / var / furniture F1 + unknown->known FPR) vs free recall and 3-way accuracy; same for the 3S-X calibrated predictions.
(2) union render identity (not covered by the coordinator's verify_recast, which checked own maps only): for every connected set S at
    hop 8, max_{a in S} observe(OR_{b in S} own_b == 1, poses_a) must equal the R8 stream target; also the own-map identity.
observe() / fibonacci_sphere() are copied verbatim from scripts/claude_v342/mapping_3d_storey_data_v420.py (SENSOR_RANGE 5.0).
Poses: probes/A01/poses_demo.json (A01 recomputed the generator's jittered trajectories from the seeds).
CPU only, one process, CUDA hidden."""
import os, sys, json, math, time, glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
ROOT = r'C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill'
sys.path.insert(0, ROOT + '/claude_handoff/analysis_20261027')
import rate_baselines_lib as L
import torch

SENSOR_RANGE = 5.0


def fibonacci_sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); theta = math.pi * (1 + 5 ** .5) * i
    return np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)), 1)


def observe(occ, cell, poses, rays):
    shape = np.array(occ.shape); cell = np.asarray(cell, np.float64)
    out = np.full(occ.shape, -1, np.int8)
    ds = 0.5 * float(cell.min()); nsteps = int(math.ceil(SENSOR_RANGE / ds))
    for p in poses:
        alive = np.ones(len(rays), bool)
        for s in range(1, nsteps + 1):
            idx_all = np.floor((p[None] + rays * (s * ds)) / cell).astype(np.int64)
            inside = np.all((idx_all >= 0) & (idx_all < shape), 1) & alive
            if not inside.any():
                break
            ii = idx_all[inside]
            hit = occ[ii[:, 0], ii[:, 1], ii[:, 2]] == 1
            fr = ii[~hit]; out[fr[:, 0], fr[:, 1], fr[:, 2]] = np.maximum(out[fr[:, 0], fr[:, 1], fr[:, 2]], 0)
            hh = ii[hit]; out[hh[:, 0], hh[:, 1], hh[:, 2]] = 1
            ai = np.nonzero(inside)[0]; alive[ai[hit]] = False
            alive &= inside | ~alive
    return out


def extra(pred, target):
    """free recall (target free -> predicted free), 3-way accuracy on target-known, unknown->known FPR; per row"""
    rows = []
    for p, t in zip(pred, target):
        known = t >= 0; free = t == 0; unk = ~known
        rows.append(dict(free_recall=float((p[free] == 0).mean()) if free.any() else None,
                         acc3=float((p[known] == t[known]).mean()),
                         fpr=float((p[unk] >= 0).mean()) if unk.any() else None))
    return rows


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def main():
    poses = json.load(open(ROOT + '/claude_handoff/analysis_20261027/probes/A01/poses_demo.json'))
    rays = fibonacci_sphere(65536)
    out = dict(houses={}, pooled={})
    pool = {k: [] for k in ('S_only_K0', 'S_only_R8', 'model_K0', 'model_R8')}
    t0 = time.time()
    for path in sorted(glob.glob(ROOT + '/demo_web3d/data/house_r*_i*.json')):
        name = os.path.basename(path)[6:-5]
        m, own, pred, gt = L.load(path)
        n = own.shape[0]
        reached = np.asarray(m['reached'], bool)                     # [hops, n, n]
        hop_index = {h: i for i, h in enumerate(m['hops'])}
        T0 = L.targets(own, reached[hop_index[0]]); T8 = L.targets(own, reached[hop_index[8]])
        slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
        struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
        rec = {}
        for tag, T, P in (('K0', T0, pred[0]), ('R8', T8, pred[8])):
            S_only = np.where(T == 1, np.int8(1), np.int8(-1))
            for arm, pr in (('S_only', S_only), ('model', P)):
                tf = torch.from_numpy(T.reshape(n, -1)); pf = torch.from_numpy(pr.reshape(n, -1))
                rows = L.metrics(pf, tf, slab, obj, struct)
                ex = extra(pr.reshape(n, -1), T.reshape(n, -1))
                rows = [dict(a, **b) for a, b in zip(rows, ex)]
                pool[f'{arm}_{tag}'] += rows
                rec[f'{arm}_{tag}'] = {k: mean(rows, k) for k in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_recall', 'acc3')}
        # (2) render identities with the recomputed true poses
        traj = np.asarray(poses[name]['trajectory_m'], np.float64)    # [3 poses, n agents, 3]
        cell = np.asarray(m['cell_m'], np.float64)
        own_mis = []
        for a in range(n):
            r = observe((own[a] == 1).astype(np.uint8), cell, traj[:, a], rays)
            own_mis.append(int((r != own[a]).sum()))
        R8 = reached[hop_index[8]]
        union_mis = []
        done = {}
        for i in range(n):
            key = tuple(np.nonzero(R8[i])[0].tolist())
            if key not in done:
                H = np.zeros(own.shape[1:], np.uint8)
                for b in key:
                    H |= (own[b] == 1).astype(np.uint8)
                That = np.full(own.shape[1:], -1, np.int8)
                for a in key:
                    That = np.maximum(That, observe(H, cell, traj[:, a], rays))
                done[key] = That
            union_mis.append(int((done[key] != T8[i]).sum()))
        rec['own_identity_mismatch'] = own_mis; rec['union_identity_mismatch'] = union_mis
        rec['components_at_hop8'] = len(done)
        out['houses'][name] = rec
        print(name, f'{time.time() - t0:.0f}s', 'own mis', own_mis, 'union mis', union_mis, flush=True)
        print('  ', {k: {kk: (round(vv, 4) if vv is not None else None) for kk, vv in v.items()} for k, v in rec.items() if isinstance(v, dict)}, flush=True)
    for k, rows in pool.items():
        out['pooled'][k] = {kk: mean(rows, kk) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_recall', 'acc3')}
    out['pooled']['own_identity_maps_exact'] = sum(sum(1 for x in h['own_identity_mismatch'] if x == 0) for h in out['houses'].values())
    out['pooled']['union_identity_rows_exact'] = sum(sum(1 for x in h['union_identity_mismatch'] if x == 0) for h in out['houses'].values())
    print(json.dumps(out['pooled'], indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c1_verify.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
