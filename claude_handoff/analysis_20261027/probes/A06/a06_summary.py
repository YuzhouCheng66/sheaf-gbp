"""A06 summary (Claude A06, 2026-10-27): every number the memo quotes, from the probe JSONs. Exact-pose accounting: the ray-cast decode
is exact only with exact poses; posequant.json measures, per agent, the voxels that differ when the poses are rounded, and the ctx bytes of
that correction set. Accounting used in the memo: poses at 0.1 mm (centre 14 bit + 3 x 2 x 11 bit = 80 bit = 10 B) + the median 0.1 mm
correction set, per agent ('exact-pose B'); the probes' own 5.5 B / agent (1 cm, world_dl's assumption) is replaced by it."""
import json, os
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
med = lambda v: float(np.median(v))

pq = L('posequant.json')
if pq:
    for q in (0.01, 0.001, 0.0001):
        r = [x for x in pq if x['q'] == q]
        print(f"pose rounding {q * 1000:g} mm: differing voxels median {med([x['diff'] for x in r]):.0f} (range {min(x['diff'] for x in r)}-{max(x['diff'] for x in r)}), "
              f"share of known {med([x['diff'] / x['known'] for x in r]):.5f}; correction ctx B median {med([x['corr_B'] for x in r]):.1f} (max {max(x['corr_B'] for x in r):.1f})")
    corr = med([x['corr_B'] for x in pq if x['q'] == 0.0001])
else:
    corr = 17.
POSE_EXACT = 10. + corr
print(f'exact-pose bytes per agent used: {POSE_EXACT:.1f}')

E = L('boxes_edges.json')
if E:
    for kind in ('own', 'union', 'edge'):
        rr = [r for h in E for r in h['rows'] if r['kind'] == kind]
        prog = [r['total_B'] - r['poses_B'] + r['agents'] * POSE_EXACT for r in rr]
        ray = [r['ctx_O'] + r['agents'] * POSE_EXACT for r in rr]
        ct, gz = [r['ctx_ternary'] for r in rr], [r['gzip'] for r in rr]
        print(f"[{kind}] n {len(rr)} agents med {med([r['agents'] for r in rr]):.0f} | ctx ternary {med(ct):.0f} gzip {med(gz):.0f} | "
              f"PROGRAM exact-pose med {med(prog):.0f} (boxes {med([r['box_B'] for r in rr]):.0f}, resid {med([r['resid_B'] for r in rr]):.0f}, n boxes {med([r['n_boxes'] for r in rr]):.0f}, p90 {np.percentile(prog, 90):.0f}) "
              f"ratios ctx {med(ct) / med(prog):.1f} gzip {med(gz) / med(prog):.1f} raw {131072 / med(prog):.0f} ours2048 {2048 / med(prog):.1f} | "
              f"RAYSHEAF ctx(O)+exact poses med {med(ray):.0f} (= {med(ray) / med(ct):.2f} x ctx; ratios gzip {med(gz) / med(ray):.1f} raw {131072 / med(ray):.0f})")

R = L('raycast.json')
if R:
    own = [r for h in R for r in h['own']]; uni = [r for h in R for r in h['union']]; edg = [r for h in R for r in h['edges']]
    print(f"raycast probe houses {len(R)}: agreement with demo own maps (all voxels) {np.mean([x for h in R for x in h['agree_all']]):.4f} (the stored known-IoU field of this run is invalid, see a06_raycast.py)")
    print(f"  exact re-cast through O: own {sum(r['exact'] for r in own)}/{len(own)}, union {sum(r['exact'] for r in uni)}/{len(uni)}")
    print(f"  synthetic maps: own ctx ternary {med([r['ctx_ternary'] for r in own]):.0f} ctx O {med([r['ctx_O'] for r in own]):.0f}; union {med([r['ctx_ternary'] for r in uni]):.0f} / {med([r['ctx_O'] for r in uni]):.0f}")
    print(f"  edges: ctx ternary {med([r['ctx_ternary'] for r in edg]):.0f}, ctx O {med([r['ctx_O'] for r in edg]):.0f}, ctx O | receiver side {med([r['ctx_O_given_recv'] for r in edg]):.0f} "
          f"(ratio {med([r['ctx_O_given_recv'] / r['ctx_O'] for r in edg]):.2f}); new occ share {np.sum([r['new_occ'] for r in edg]) / np.sum([r['n_occ'] for r in edg]):.3f}")
    for name in R[0]['lossy']:
        g = lambda hop, k: np.mean([x for h in R for x in h['lossy'][name][hop][k] if x is not None])
        b0 = med([x for h in R for x in h['lossy'][name]['K0_bytes']]); b8 = med([x for h in R for x in h['lossy'][name]['R8_bytes']])
        f0 = np.mean([x for h in R for x in h['lossy'][name]['K0_fpr']]); f8 = np.mean([x for h in R for x in h['lossy'][name]['R8_fpr']])
        print(f"  lossy {name:14s} K0 {b0:5.0f} B surf {g('K0', 'surface_f1'):.3f} var {g('K0', 'var_occ_f1'):.3f} furn {g('K0', 'furn_f1'):.3f} fpr {f0:.4f} | "
              f"R8 {b8:5.0f} B surf {g('R8', 'surface_f1'):.3f} var {g('R8', 'var_occ_f1'):.3f} furn {g('R8', 'furn_f1'):.3f} fpr {f8:.4f}")

B = L('boxes_m8.json')
if B:
    for kind in ('own', 'union'):
        rr = [r for h in B for r in h[kind]]
        f = lambda k: np.mean([r[k] for r in rr if r.get(k) is not None])
        print(f"program decode on re-simulated maps [{kind}] houses {len(B)}: exact {sum(r['exact'] for r in rr)}/{len(rr)}; total med {med([r['total_B'] for r in rr]):.0f} B; "
              f"boxes-only lossy (residual dropped): surf {f('lossy_surface_f1'):.3f} var {f('lossy_var_occ_f1'):.3f} furn {f('lossy_furn_f1'):.3f} at {med([r['box_B'] + r['poses_B'] for r in rr]):.0f} B")

F = L('furn_bits.json')
if F:
    for k, d in F['summary'].items():
        print(f"furn bits {k}: furniture boxes {d['n_furn_boxes']:.0f} = {d['furn_box_B']:.0f} B, structure boxes {d['n_struct_boxes']:.0f} = {d['struct_box_B']:.0f} B, residual {d['resid_B']:.0f} B; "
              f"drop furniture residual -> furn F1 {d['lossy_furn_f1']:.3f} surface {d['lossy_surface_f1']:.3f}")
