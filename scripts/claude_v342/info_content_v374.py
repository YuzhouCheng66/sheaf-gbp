"""How many bits does the map that fusion must convey actually contain?  (Claude, 2026-10-23; CPU only, no model, VALIDATION only.)

First-principles check for the HouseExpo gap. After max-precision fusion the WHOLE component's map lives in ONE 64-coordinate vector
(64 x 8-bit means = 512 bits of mean information, + 64 x 8-bit precisions that mostly say "observed or not"). Whatever the encoder and
decoder do, the decoded R8 map can carry at most this many bits about the target; the rest must come from the decoder's prior.
This script measures the information content of the targets with a real lossless code length:
  * adaptive context-model (JBIG-like 10-pixel causal template) binary coding with the Krichevsky-Trofimov estimator, whose total
    code length is order-free and has the closed form  L(a,b) = -log2[ G(a+1/2) G(b+1/2) / (pi G(a+b+1)) ]  per context
    (arithmetic coding achieves it to within 2 bits for the whole map). Ternary target = plane K (known) + plane W (wall | known),
    W coded only where K = 1 with W-template context + K at the pixel. This is a valid lossless code => an upper bound on the bits
    a perfect fused map needs; the fixed-chart 512-bit capacity is compared with it.
  * HouseExpo only: vector (polygon) description length of the whole house at the native raster (7 bits per coordinate; axis-aligned
    edges need 1 coordinate per vertex + 1 flag bit, general edges 2 coordinates + 1 flag) -- the "polyline lossless" baseline of
    README section 8, for the whole house (an upper bound on what any agent team could need to send about the walls).
Rows: v293 VAL, HouseExpo v365 (20 m chart) VAL, HouseExpo v371 (house-fitted chart) VAL; strata x N{8,16} x idx 0-7; targets K0 (own)
and R8 (component union); one row per distinct target.
python info_content_v374.py --source <sealed> --out <json>   (env HOUSEEXPO_JSON_DIR / HOUSEEXPO_INDEX for the HouseExpo rows)
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.special import gammaln

LN2 = math.log(2.)
# JBIG-like 10-pixel causal template (dy, dx)
TEMPLATE = [(-2, -1), (-2, 0), (-2, 1), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -2), (0, -1)]
TEMPLATES = {'t4': [(-1, -1), (-1, 0), (-1, 1), (0, -1)], 't6': [(-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (-2, 0)], 't10': TEMPLATE}
# the coder signals which template it used (2 bits) and codes with it -> min over templates + 2 bits is still a valid lossless code


def kt_bits(a, b):
    """total KT code length (bits) of a binary sequence with a zeros and b ones (any order)"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return -(gammaln(a + .5) + gammaln(b + .5) - math.log(math.pi) - gammaln(a + b + 1.)) / LN2


def context_ids(plane, template=TEMPLATE):
    """causal-template context index per pixel (outside the map = 0)"""
    h, w = plane.shape
    pad = np.zeros((h + 2, w + 4), np.int64); pad[2:, 2:w + 2] = plane
    ctx = np.zeros((h, w), np.int64)
    for bit, (dy, dx) in enumerate(template):
        ctx |= pad[2 + dy:2 + dy + h, 2 + dx:2 + dx + w] << bit
    return ctx


def plane_bits(plane, where=None, extra=None):
    return 2. + min(_plane_bits(plane, where, extra, t) for t in TEMPLATES.values())


def _plane_bits(plane, where, extra, template):
    ctx = context_ids(plane.astype(np.int64), template)
    if extra is not None:
        ctx = ctx * 2 + extra.astype(np.int64)
    x = plane.astype(np.int64)
    if where is not None:
        ctx, x = ctx[where], x[where]
    ctx = ctx.ravel(); x = x.ravel()
    if ctx.size == 0:
        return 0.
    n1 = np.bincount(ctx, weights=x, minlength=1); n = np.bincount(ctx, minlength=1)
    used = n > 0
    return float(kt_bits((n - n1)[used], n1[used]).sum())


def ternary_bits(t):
    """lossless bits of a ternary (-1 unknown / 0 free / 1 wall) map: K plane + W plane given K"""
    known = (t >= 0).astype(np.int8); wall = (t == 1).astype(np.int8)
    bk = plane_bits(known)
    bw = plane_bits(wall, where=known.astype(bool), extra=known)
    return bk, bw


def polygon_bits(verts_m, L, size):
    v = np.floor(np.asarray(verts_m) / (L / size)).astype(np.int64)
    keep = [0] + [i for i in range(1, len(v)) if (v[i] != v[i - 1]).any()]
    v = v[keep]
    if len(v) > 1 and (v[0] == v[-1]).all():
        v = v[:-1]
    bits = 16 + 14                                    # vertex count + first vertex
    ax = 0
    for i in range(1, len(v)):
        d = v[i] - v[i - 1]
        if d[0] == 0 or d[1] == 0:
            bits += 1 + 7 + 1; ax += 1                 # flag + one coordinate + which axis
        else:
            bits += 1 + 14
    return bits, len(v), ax / max(len(v) - 1, 1)


def rng_list(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--worlds', default='v293,v365,v371'); ap.add_argument('--indices', default='0-7'); ap.add_argument('--size', type=int, default=128)
    args = ap.parse_args()
    src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scripts import mapping_native_bank_v306 as bank
    rows = []; t0 = time.time()
    for world in args.worlds.split(','):
        if world == 'v293':
            from scripts import mapping_multiroom_data_v293 as data
            strata = (6, 12, 20)
        elif world == 'v365':
            import mapping_houseexpo_data_v365 as data
            strata = (1, 2, 3)
        else:
            import mapping_houseexpo_data_v371 as data
            strata = (1, 2, 3)
        for rooms in strata:
            for agents in (8, 16):
                for index in rng_list(args.indices):
                    fx = data.make_fixture(split='VALIDATION', index=index, size=args.size, rooms=rooms, agents=agents)
                    inp = fx['inputs']
                    gt = fx['evaluation']['walls'] if 'walls' in fx['evaluation'] else None
                    house = {}
                    if gt is not None:
                        g = (np.asarray(gt) != 0).astype(np.int8)
                        house['house_wall_raster_bits'] = plane_bits(g)
                    if world != 'v293':
                        spec = data.world_spec('VALIDATION', index, rooms)
                        L = spec.get('chart_m', 20.)
                        pb, nv, axf = polygon_bits(spec['verts_m'], L, args.size)
                        house.update(house_polygon_bits=pb, house_vertices=nv, axis_aligned_fraction=axf, chart_m=L)
                    for k in (0, 8):
                        targets, reached = bank.forest_targets(inp, k)
                        seen = set()
                        for i in range(len(targets)):
                            key = tuple(np.flatnonzero(reached[i]))
                            if key in seen:
                                continue
                            seen.add(key)
                            bk, bw = ternary_bits(targets[i])
                            rows.append(dict(world=world, rooms=rooms, agents=agents, index=index, hops=k, sources=len(key),
                                             known_cells=int((targets[i] >= 0).sum()), wall_cells=int((targets[i] == 1).sum()), bits_known=bk, bits_wall_given_known=bw,
                                             bits_total=bk + bw, **house))
                    print(f"{world} r{rooms} n{agents} i{index} | {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for world in args.worlds.split(','):
        for k in (0, 8):
            for rooms in sorted({r['rooms'] for r in rows if r['world'] == world}):
                sel = [r for r in rows if r['world'] == world and r['hops'] == k and r['rooms'] == rooms]
                if not sel:
                    continue
                f = lambda key: float(np.median([r[key] for r in sel]))
                s = dict(n=len(sel), median_bits_total=f('bits_total'), median_bits_wall_given_known=f('bits_wall_given_known'), median_bits_known=f('bits_known'),
                         median_known_cells=f('known_cells'), median_sources=f('sources'),
                         frac_targets_over_512_bits=float(np.mean([r['bits_total'] > 512 for r in sel])),
                         frac_wall_bits_over_512=float(np.mean([r['bits_wall_given_known'] > 512 for r in sel])))
                if 'house_polygon_bits' in sel[0]:
                    s.update(median_house_polygon_bits=f('house_polygon_bits'), median_house_vertices=f('house_vertices'), median_axis_aligned_fraction=f('axis_aligned_fraction'))
                if 'house_wall_raster_bits' in sel[0]:
                    s.update(median_house_wall_raster_bits=f('house_wall_raster_bits'))
                summary[f'{world}/R{k}/s{rooms}'] = s
                print(world, f'R{k}', rooms, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in s.items()})
    json.dump(dict(script='info_content_v374.py', template=TEMPLATE, estimator='KT adaptive context coding (closed form)', capacity_bits=dict(mean_only=512, mean_and_precision=1024),
                   rows=rows, summary=summary), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
