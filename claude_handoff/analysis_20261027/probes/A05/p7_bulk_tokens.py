"""A05 probe 7 (CPU, demo data, exact poses): a LOSSY box code that sends only BULK boxes (no extent of 1 voxel) and drops thin ones
(legs, shelves, table tops, side panels). Two encoders: oracle (visible boxes of the greedy decomposition of gt_obj, p4) and realistic
(solid boxes grown from the agent's own hits, p6). Decode = exact structure + kept boxes + ray cast (union over reached agents' poses
at R8, tokens reduced by containment). Reports tokens kept / total, F1, FPR and the crude leave-one-house-out token code of p6."""
import glob, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from rate_baselines_lib import load, targets, metrics, dilate6, DIMS  # noqa: E402
from p2b_raycast_decode import ray_table, cast, struct_parts, fpr  # noqa: E402
from p4_box_tokens import greedy_boxes, raster  # noqa: E402
from p6_encoder_boxes import solid_boxes, reduce_contained, token_bits  # noqa: E402


def bulk(b):
    b = b.reshape(-1, 6)
    return b[((b[:, 1] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 2]) > 1) & ((b[:, 5] - b[:, 4]) > 1)]


if __name__ == '__main__':
    poses = json.load(open(os.path.join(HERE, 'poses.json')))
    rows = []; toks = {}
    for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
        t0 = time.time()
        m, own, pred, gt = load(path); n = own.shape[0]; key = m['house']
        traj = np.asarray(poses[key]['trajectory_m']); cell = tuple(poses[key]['cell']); lk = poses[key]['lintel_k']
        tables = {a: [ray_table(traj[k, a], cell) for k in range(traj.shape[0])] for a in range(n)}
        slab = gt['slab']; obj = gt['obj']; occ = gt['occ']; struct = occ & ~obj & ~slab
        slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
        R8 = np.asarray(m['reached'][m['hops'].index(8)], bool); T8 = targets(own, R8)
        world = greedy_boxes(obj).reshape(-1, 6)
        enc = {}
        for a in range(n):
            t = own[a]; Hf = (t == 1) & obj
            Gs, wall, _ = struct_parts(t == 1, gt, lk)
            near = Hf.copy()
            for _ in range(2):
                near = near | dilate6(torch.from_numpy(near.reshape(1, -1)), DIMS)[0].numpy().reshape(DIMS)
            allowed = Hf | (near & (t < 0) & ~wall[:, :, None] & ~slab)
            vis = np.array([Hf[x0:x1, y0:y1, z0:z1].any() for x0, x1, y0, y1, z0, z1 in world], bool)
            enc[('oracle', a)] = world[vis]; enc[('solid', a)] = solid_boxes(Hf, allowed).reshape(-1, 6)
        for hop, T in ((0, own), (8, T8)):
            seen = set()
            for a in range(n):
                agents = (a,) if hop == 0 else tuple(np.nonzero(R8[a])[0])
                if agents in seen:
                    continue
                seen.add(agents)
                t = T[a]; Gs, _, _ = struct_parts(t == 1, gt, lk); tabs = [tb for b_ in agents for tb in tables[b_]]
                for kind in ('oracle', 'solid'):
                    allb = reduce_contained(np.concatenate([enc[(kind, b_)] for b_ in agents]))
                    for keep in ('all', 'bulk'):
                        B = allb if keep == 'all' else bulk(allb)
                        Pr = cast(Gs | raster(B, DIMS), tabs)
                        mt = metrics(torch.from_numpy(Pr.reshape(1, -1)), torch.from_numpy(t.reshape(1, -1)), slab_t, obj_t, struct_t)[0]
                        rows.append(dict(house=key, hop=hop, agent=a, kind=kind, keep=keep, tokens=len(B), tokens_all=len(allb), fpr=fpr(Pr, t), **mt))
                        toks.setdefault((kind, keep, hop), []).append((key, B))
        print(key, f'{time.time() - t0:.0f}s', flush=True)
    S = {}
    for (kind, keep, hop), tb in toks.items():
        r = [x for x in rows if x['kind'] == kind and x['keep'] == keep and x['hop'] == hop]
        code = token_bits(tb)
        S[f'{kind} {keep} hop{hop}'] = dict(tokens=float(np.median([x['tokens'] for x in r])), tokens_all=float(np.median([x['tokens_all'] for x in r])),
                                            surface_f1=float(np.mean([x['surface_f1'] for x in r])), var_f1=float(np.mean([x['var_occ_f1'] for x in r])),
                                            furn_f1=float(np.mean([x['furn_f1'] for x in r if x['furn_f1'] is not None])), fpr=float(np.mean([x['fpr'] for x in r])),
                                            token_bytes=float(np.median([x['bytes'] for x in code])), bits_per_token=float(np.median([x['bytes'] * 8 / max(x['tokens'], 1) for x in code])), n=len(r))
    json.dump(dict(summary=S, rows=rows), open(os.path.join(HERE, 'p7_bulk_tokens.json'), 'w'), indent=1)
    for k, v in S.items():
        print(f'{k:22s} ' + ' '.join(f'{kk} {vv:.4f}' if isinstance(vv, float) else f'{kk} {vv}' for kk, vv in v.items()))
