"""A06: how many bits does the FURNITURE part of a world program cost, and what furniture F1 does it buy? (Claude A06, 2026-10-27)
Demo own maps (K0) and R8 unions, greedy cover at m = 8 / 16 / 32 (a06_boxes.cover). A box is a furniture box when >= 50 % of the O voxels
it newly covers lie in the furniture region. Bits per box: leave-one-house-out static histograms from boxes_edges.json (all programs of
the other houses). Lossy point: keep every structure box and residual, keep furniture boxes, DROP the furniture residual voxels (thin parts
and anything the boxes did not explain) -> furniture-region F1 of that map against the target (no ray-cast: dropped voxels become free,
which is what the metric sees; the ray-cast would only add known voxels in target-unknown space)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, metrics_rows, mean  # noqa: E402
from a06_boxes import cover, render, fields, box_bits, FIELDS  # noqa: E402
from a06_raycast import ctx_bin  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
E = json.load(open(os.path.join(OUT, 'boxes_edges.json')))


def hist_without(house):
    hist = {name: np.zeros(nn) for name, nn in FIELDS}
    for g in E:
        if g['house'] != house:
            for r in g['rows']:
                for row in r['fields']:
                    for (name, nn), v in zip(FIELDS, row):
                        hist[name][v] += 1
    return hist


res = {}
for path in houses():
    S = scene(path); m = S['m']; n = m['agents']; h = m['house']; hist = hist_without(h)
    fr = S['region'] & S['obj']
    for kind, T in (('K0', S['own']), ('R8', S['tgt'][8])):
        for M in (8, 16, 32):
            preds, fb, sb, rb, nf, ns = [], [], [], [], [], []
            for a in range(n):
                t = T[a]; O = t == 1
                boxes, left = cover(t, M)
                cov = np.zeros_like(O); fbox, sbox = [], []
                for b in boxes:
                    sl = np.s_[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
                    new = O[sl] & ~cov[sl]
                    (fbox if (new & fr[sl]).sum() >= 0.5 * max(new.sum(), 1) else sbox).append(b)
                    cov[sl] = True
                wb = render(boxes, t.shape)
                fb.append(box_bits(fields(fbox), hist) / 8 - 2); sb.append(box_bits(fields(sbox), hist) / 8)
                rb.append(ctx_bin(left, side=wb)); nf.append(len(fbox)); ns.append(len(sbox))
                p = t.copy(); p[left & fr] = 0; preds.append(p)
            rows = metrics_rows(np.stack(preds), T, S)
            d = res.setdefault(f'{kind}_m{M}', dict(furn_box_B=[], struct_box_B=[], resid_B=[], n_furn_boxes=[], n_struct_boxes=[], lossy_furn_f1=[], lossy_surface_f1=[]))
            d['furn_box_B'] += fb; d['struct_box_B'] += sb; d['resid_B'] += rb; d['n_furn_boxes'] += nf; d['n_struct_boxes'] += ns
            d['lossy_furn_f1'] += [r['furn_f1'] for r in rows]; d['lossy_surface_f1'] += [r['surface_f1'] for r in rows]
    print(h, 'done', flush=True)
S2 = {k: {kk: (float(np.median(v)) if 'f1' not in kk else mean(v)) for kk, v in d.items()} for k, d in res.items()}
json.dump(dict(summary=S2, raw=res), open(os.path.join(OUT, 'furn_bits.json'), 'w'), indent=1)
for k, d in S2.items():
    print(f"{k:7s} furniture boxes {d['n_furn_boxes']:.0f} = {d['furn_box_B']:.0f} B; structure boxes {d['n_struct_boxes']:.0f} = {d['struct_box_B']:.0f} B; residual {d['resid_B']:.0f} B | "
          f"drop furniture residual -> furn F1 {d['lossy_furn_f1']:.3f}, surface {d['lossy_surface_f1']:.3f}")
