"""A08 probe 2 (CPU): what does a 'send geometry, ray-cast visibility' codec cost on the REAL demo own maps (6 VAL houses x 8 agents)?
Per agent (K0 content), per round-8 union (R8 content) and per directed tree edge (one converged message), in bytes:
  ctx3      = rate_baselines_lib.ctx_bytes of the ternary map (the 'strongest lossless' reference, recomputed);
  O_ctx     = ctx_bin of the observed-occupied set O alone (lossless: the map = RayCast(O, poses), probe 1);
  plan      = 2-D observed wall plan (none / wall / lintel) coded with ctx; heights 2 B (public-free: wall and door height);
  furnvox   = ctx_bin of the observed furniture voxels (lossless residual of the plan);
  boxes     = greedy consistency-set box cover of the observed furniture voxels (O_furn subset D subset not-F), adaptive box code;
  poses     = 5 B per agent (stored centre on the R1 raster 14 bit + 3 jitters at 1 cm, 26 bit; probe 1 prices 1 cm).
Edges: innovation = what the sender side has that the receiver side does not, coded conditionally on what both ends hold
(O of the receiver side: legal common knowledge after the exchange, not Wyner-Ziv); 'full' = the sender side union coded alone.
Also counts conflicts |D_a cap F_b| between agents' box covers and other agents' observed free space (can covers be unioned?)."""
import glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from a08_common import load, targets, ctx_bytes, ctx_bin, ctx_tern2d, gt_layers, plan_split, box_cover, render_boxes, box_bits, render_structure  # noqa: E402

DATA = sys.argv[1]; OUT = sys.argv[2]
POSE_B = 5.0; HEIGHT_B = 2.0


def describe(M, gt, lay, with_boxes=True):
    O = M == 1; F = M == 0
    plan, O_furn = plan_split(O, gt, lay)
    d = dict(known=int((M >= 0).sum()), n_O=int(O.sum()), n_Ofurn=int(O_furn.sum()), ctx3=ctx_bytes(M), O_ctx=ctx_bin(O),
             plan=ctx_tern2d(plan), plan_px=int((plan >= 0).sum()), furnvox=ctx_bin(O_furn))
    # residual check: everything in O that the structure render does not explain must be furniture
    S = render_structure(plan, gt, lay)
    d['unexplained_nonfurn'] = int((O & ~S & ~gt['obj']).sum()); d['struct_in_free'] = int((S & F).sum())
    if with_boxes:
        bx = box_cover(O_furn, ~F)
        D = render_boxes(bx)
        d['n_boxes'] = len(bx); raw, ad = box_bits(bx); d['boxes_raw'] = raw / 8; d['boxes'] = ad / 8
        d['cover_ok'] = bool((O_furn <= D).all() and not (D & F).any())
        d['_boxes'] = bx
    return d


def house(path):
    t0 = time.time()
    m, own, pred, gt = load(path); n = own.shape[0]; lay = gt_layers(gt)
    rows = dict(house=m['house'], chart_m=m['chart_m'], n_objects=m['n_objects'], own=[], union8=[], edge=[], conflict=[])
    covers = []
    for a in range(n):
        d = describe(own[a], gt, lay); covers.append(render_boxes(d.pop('_boxes'))); d['agent'] = a; rows['own'].append(d)
    for a in range(n):                                                  # conflicts: agent a's cover inside agent b's observed free space
        for b in range(n):
            if a != b:
                rows['conflict'].append(int((covers[a] & (own[b] == 0)).sum()))
    hi8 = m['hops'].index(8); R8 = np.asarray(m['reached'][hi8], bool); T8 = targets(own, R8)
    for a in range(n):
        d = describe(T8[a], gt, lay); d.pop('_boxes'); d['agent'] = a; d['reached'] = int(R8[a].sum()); rows['union8'].append(d)
    adj = {i: set() for i in range(n)}
    for i, j in m['edges']:
        adj[i].add(j); adj[j].add(i)
    def side(s, r):
        out, st = {s}, [s]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v != r and v not in out:
                    out.add(v); st.append(v)
        return sorted(out)
    for i, j in m['edges']:
        for s, r in ((i, j), (j, i)):
            Ss, Sr = side(s, r), side(r, s)
            Ms = own[Ss].max(0); Mr = own[Sr].max(0)
            Os, Or = Ms == 1, Mr == 1; Fs, Fr = Ms == 0, Mr == 0
            plan_s, Of_s = plan_split(Os, gt, lay); plan_r, Of_r = plan_split(Or, gt, lay)
            innov_plan = np.where((plan_s >= 0) & (plan_r < 0), plan_s, np.int8(-1)).astype(np.int8)
            Xf = Of_s & ~Of_r
            bx_full = box_cover(Of_s, ~Fs); bx_inn = box_cover(Xf, ~(Fs | Fr))
            e = dict(sender=s, receiver=r, side=len(Ss), ctx3_full=ctx_bytes(Ms), O_full=ctx_bin(Os), O_innov=ctx_bin(Os & ~Or, side=Or),
                     plan_full=ctx_tern2d(plan_s), plan_innov=ctx_tern2d(innov_plan), boxes_full=box_bits(bx_full)[1] / 8, boxes_innov=box_bits(bx_inn)[1] / 8,
                     n_boxes_full=len(bx_full), n_boxes_innov=len(bx_inn), furnvox_full=ctx_bin(Of_s), furnvox_innov=ctx_bin(Xf, side=Of_r))
            rows['edge'].append(e)
    rows['seconds'] = time.time() - t0
    print(m['house'], f"{rows['seconds']:.0f}s", flush=True)
    return rows


if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(DATA, 'house_*.json')))
    with Pool(len(paths)) as pool:
        H = pool.map(house, paths)
    json.dump(H, open(OUT, 'w'), indent=1)
    med = lambda v: float(np.median(v))
    for grp in ('own', 'union8'):
        R = [r for h in H for r in h[grp]]
        rc = [r['O_ctx'] + POSE_B * (1 if grp == 'own' else r['reached']) for r in R]
        pv = [r['plan'] + r['furnvox'] + HEIGHT_B + POSE_B * (1 if grp == 'own' else r['reached']) for r in R]
        pb = [r['plan'] + r['boxes'] + HEIGHT_B + POSE_B * (1 if grp == 'own' else r['reached']) for r in R]
        print(f"[{grp}] n {len(R)} | ctx3 {med([r['ctx3'] for r in R]):.0f} | O_ctx {med([r['O_ctx'] for r in R]):.0f} -> RC lossless {med(rc):.0f} "
              f"| plan {med([r['plan'] for r in R]):.0f} furnvox {med([r['furnvox'] for r in R]):.0f} -> plan+vox {med(pv):.0f} "
              f"| boxes n {med([r['n_boxes'] for r in R]):.0f} {med([r['boxes'] for r in R]):.0f} B (raw {med([r['boxes_raw'] for r in R]):.0f}) -> plan+boxes {med(pb):.0f} "
              f"| cover_ok {all(r['cover_ok'] for r in R)} unexplained {sum(r['unexplained_nonfurn'] for r in R)} struct_in_free {sum(r['struct_in_free'] for r in R)} "
              f"| n_O {med([r['n_O'] for r in R]):.0f} furn share {np.mean([r['n_Ofurn'] / max(r['n_O'], 1) for r in R]):.3f}")
    E = [e for h in H for e in h['edge']]
    print(f"[edge] n {len(E)} | ctx3_full {med([e['ctx3_full'] for e in E]):.0f} | O_full {med([e['O_full'] for e in E]):.0f} O_innov {med([e['O_innov'] for e in E]):.0f} "
          f"| plan full/innov {med([e['plan_full'] for e in E]):.0f}/{med([e['plan_innov'] for e in E]):.0f} | boxes full/innov {med([e['boxes_full'] for e in E]):.0f}/{med([e['boxes_innov'] for e in E]):.0f} "
          f"(n {med([e['n_boxes_full'] for e in E]):.0f}/{med([e['n_boxes_innov'] for e in E]):.0f}) | furnvox full/innov {med([e['furnvox_full'] for e in E]):.0f}/{med([e['furnvox_innov'] for e in E]):.0f} | side {med([e['side'] for e in E]):.0f}")
    print('per-edge totals (median): RC innov', med([e['O_innov'] + POSE_B * e['side'] for e in E]), ' RC full', med([e['O_full'] + POSE_B * e['side'] for e in E]),
          ' prim innov', med([e['plan_innov'] + e['boxes_innov'] + HEIGHT_B + POSE_B * e['side'] for e in E]), ' prim full', med([e['plan_full'] + e['boxes_full'] + HEIGHT_B + POSE_B * e['side'] for e in E]))
    C = [c for h in H for c in h['conflict']]
    print('box-cover conflicts |D_a cap F_b| over ordered pairs: median', med(C), 'mean', float(np.mean(C)), 'share of pairs > 0', float(np.mean(np.asarray(C) > 0)))
