"""A06 furniture-region error anatomy (Claude A06, 2026-10-27). CPU only, 6 VAL demo houses x 8 agents, 3S-X + calibrated rule.
Target-side classes (every target-occupied furniture-region voxel = TP or FN): height band, part thickness (thin_xy / thin_z / bulk by
morphological opening of gt_obj), code-column phase of its free faces (32x32 grid: 4-voxel columns; also 64x64 2-voxel columns),
distance to the nearest wall column, against-wall component, column complexity (z-levels with furniture in the 4x4 column, wall in column).
FN split by predicted state (unknown = known-head side; free = occupancy-head side). FP split by distance to the nearest target furniture
voxel, adjacency-face phase, whether its 4x4 column holds any target furniture, height, wall distance, part class of the nearest target voxel.
Output: anatomy.json (pooled counts and F1 'what if this class were perfect') + printed tables."""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a06_common import houses, scene, furn_f1_map, f1, mean, metrics_rows  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
HB = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 9.9]
HN = ['0-.3', '.3-.6', '.6-.9', '.9-1.2', '1.2-1.5', '1.5-2.0', '2.0+']


def hband(z):
    h = (z + 0.5) * 0.1 - 0.1                      # height above the floor top (floor slab = z 0)
    return np.digitize(h, HB[1:-1])


def face_phase(t_free, C):
    """per voxel: min distance (in voxels) of its horizontal faces towards target-free voxels to the nearest code-column edge
    (C = column width in voxels); 9 = no horizontal free face."""
    X, Y, Z = t_free.shape
    d = np.full((X, Y, Z), 9, np.int8)
    ix = np.arange(X)
    dp = np.minimum((ix + 1) % C, C - (ix + 1) % C); dm = np.minimum(ix % C, C - ix % C)
    nb = np.zeros_like(t_free); nb[:-1] = t_free[1:]; d = np.where(nb, np.minimum(d, dp[:, None, None]), d)
    nb = np.zeros_like(t_free); nb[1:] = t_free[:-1]; d = np.where(nb, np.minimum(d, dm[:, None, None]), d)
    nb = np.zeros_like(t_free); nb[:, :-1] = t_free[:, 1:]; d = np.where(nb, np.minimum(d, dp[None, :, None]), d)
    nb = np.zeros_like(t_free); nb[:, 1:] = t_free[:, :-1]; d = np.where(nb, np.minimum(d, dm[None, :, None]), d)
    return d


def vert_free(t_free):
    nb = np.zeros_like(t_free); nb[:, :, :-1] |= t_free[:, :, 1:]; nb[:, :, 1:] |= t_free[:, :, :-1]
    return nb


def adj_phase(fp, to, C):
    """per FP voxel: min column-edge distance of a horizontal face shared with a target furniture voxel; 8 = only vertical adjacency; 9 = none."""
    X = fp.shape[0]; ix = np.arange(X)
    dp = np.minimum((ix + 1) % C, C - (ix + 1) % C); dm = np.minimum(ix % C, C - ix % C)
    d = np.full(fp.shape, 9, np.int8)
    nb = np.zeros_like(to); nb[:-1] = to[1:]; d = np.where(nb, np.minimum(d, dp[:, None, None]), d)
    nb = np.zeros_like(to); nb[1:] = to[:-1]; d = np.where(nb, np.minimum(d, dm[:, None, None]), d)
    nb = np.zeros_like(to); nb[:, :-1] = to[:, 1:]; d = np.where(nb, np.minimum(d, dp[None, :, None]), d)
    nb = np.zeros_like(to); nb[:, 1:] = to[:, :-1]; d = np.where(nb, np.minimum(d, dm[None, :, None]), d)
    vz = np.zeros_like(to); vz[:, :, :-1] |= to[:, :, 1:]; vz[:, :, 1:] |= to[:, :, :-1]
    d = np.where((d == 9) & vz, 8, d)
    return d


def add(dct, key, val, n):
    dct.setdefault(key, {}); dct[key][val] = dct[key].get(val, 0) + int(n)


R = {}
for path in houses():
    S = scene(path); m = S['m']; n = S['own'].shape[0]; region = S['region']; obj = S['obj']
    assert not ((S['own'] == 1) & region[None] & ~obj[None]).any(), 'target furniture-region voxels outside gt_obj'
    walls2d = S['struct'][:, :, 10]
    wd = ndimage.distance_transform_edt(~walls2d)                                   # voxels to nearest wall column
    wbin = np.digitize(wd, [2.01, 4.01, 8.01, 16.01])[:, :, None] * np.ones((1, 1, 32), np.int8)
    lab, nl = ndimage.label(obj, structure=np.ones((3, 3, 3)))
    against = np.zeros(nl + 1, bool)
    if nl:
        mins = ndimage.minimum(np.broadcast_to(wd[:, :, None], obj.shape), lab, index=np.arange(1, nl + 1))
        against[1:] = np.asarray(mins) <= 2.0
    against_m = against[lab] & obj
    part = np.where(S['thin_xy'], 0, np.where(S['thin_z'], 1, np.where(S['thin'], 2, 3)))  # 0 thin_xy, 1 thin_z, 2 thin other, 3 bulk
    zz = np.broadcast_to(np.arange(32)[None, None, :], obj.shape)
    hb = hband(zz)
    col_wall = S['struct'].reshape(32, 4, 32, 4, 32).any((1, 3, 4))                 # 4x4 code column holds wall/lintel
    for hop in (0, 8):
        T = S['tgt'][hop]; P = S['pred'][hop]
        rows = metrics_rows(P, T, S)
        D = R.setdefault(str(hop), dict(tgt={}, fp={}, fn_state={}, f1_official=[], f1_rows=[], tot=dict(tp=0, fp=0, fn=0)))
        for a in range(n):
            t, p = T[a], P[a]; known = t >= 0
            to = (t == 1) & region; po = (p == 1) & known & region
            tp_m = to & po; fn_m = to & ~po; fp_m = po & ~to
            f, tp, fp, fn = furn_f1_map(p, t, region)
            assert f is None or abs(f - rows[a]['furn_f1']) < 1e-9
            D['f1_official'].append(rows[a]['furn_f1']); D['tot']['tp'] += tp; D['tot']['fp'] += fp; D['tot']['fn'] += fn
            D['f1_rows'].append(dict(house=m['house'], agent=a, f1=f, tp=tp, fp=fp, fn=fn, surface=rows[a]['surface_f1']))
            tfree = t == 0
            ph4 = face_phase(tfree, 4); ph2 = face_phase(tfree, 2); vf = vert_free(tfree)
            cnt_lev = to.reshape(32, 4, 32, 4, 32).any((1, 3)).sum(-1)                  # z-levels with target furniture per 4x4 column
            col_lev = np.repeat(np.repeat(cnt_lev, 4, 0), 4, 1)[:, :, None] * np.ones((1, 1, 32), int)
            col_w = np.repeat(np.repeat(col_wall, 4, 0), 4, 1)[:, :, None] * np.ones((1, 1, 32), bool)
            cls = dict(height=hb, part=part, phase32=np.where(ph4 < 9, ph4, np.where(vf, 7, 8)), phase64=np.where(ph2 < 9, ph2, np.where(vf, 7, 8)),
                       wall=wbin, against=against_m.astype(np.int8), col_levels=np.digitize(col_lev, [2.5, 5.5, 10.5]), col_wall=col_w.astype(np.int8))
            for name, c in cls.items():
                for v in np.unique(c[to]):
                    sel = c == v
                    add(D['tgt'], name, f'{v}_tp', (tp_m & sel).sum()); add(D['tgt'], name, f'{v}_fn', (fn_m & sel).sum())
            add(D['fn_state'], 'all', 'unknown', (fn_m & (p == -1)).sum()); add(D['fn_state'], 'all', 'free', (fn_m & (p == 0)).sum())
            for v in range(4):
                add(D['fn_state'], f'part{v}', 'unknown', (fn_m & (p == -1) & (part == v)).sum()); add(D['fn_state'], f'part{v}', 'free', (fn_m & (p == 0) & (part == v)).sum())
            for v in range(len(HN)):
                add(D['fn_state'], f'h{v}', 'unknown', (fn_m & (p == -1) & (hb == v)).sum()); add(D['fn_state'], f'h{v}', 'free', (fn_m & (p == 0) & (hb == v)).sum())
            # FP side
            if fp_m.any():
                if to.any():
                    dist, ind = ndimage.distance_transform_edt(~to, return_indices=True)
                    near_part = part[ind[0], ind[1], ind[2]]
                else:
                    dist = np.full(to.shape, 99.); near_part = np.full(to.shape, 9)
                dob = ndimage.distance_transform_edt(~obj)                               # to any world furniture voxel (seen or not)
                colany = np.repeat(np.repeat(to.reshape(32, 4, 32, 4, 32).any((1, 3, 4)), 4, 0), 4, 1)[:, :, None] * np.ones((1, 1, 32), bool)
                fcls = dict(dist=np.digitize(dist, [1.01, 1.75, 2.01, 4.01]), dist_world=np.digitize(dob, [1.01, 2.01, 4.01]), near_part=near_part,
                            adj32=adj_phase(fp_m, to, 4), adj64=adj_phase(fp_m, to, 2), col_has_furn=colany.astype(np.int8), height=hb, wall=wbin)
                for name, c in fcls.items():
                    for v in np.unique(c[fp_m]):
                        add(D['fp'], name, str(v), (fp_m & (c == v)).sum())
    print(m['house'], 'done', flush=True)

# ---- summaries
def summarize(D):
    tp, fp, fn = D['tot']['tp'], D['tot']['fp'], D['tot']['fn']
    out = dict(f1_official_mean=mean(D['f1_official']), f1_pooled=f1(tp, fp, fn), tp=tp, fp=fp, fn=fn, recall=tp / (tp + fn), precision=tp / (tp + fp),
               fn_over_fp=fn / fp)
    tg = {}
    for name, c in D['tgt'].items():
        vals = sorted({k.rsplit('_', 1)[0] for k in c})
        tg[name] = {}
        for v in vals:
            a, b = c.get(f'{v}_tp', 0), c.get(f'{v}_fn', 0)
            tg[name][v] = dict(share=(a + b) / (tp + fn), recall=a / max(a + b, 1), fn_share=b / max(fn, 1), f1_if_fn_fixed=f1(tp + b, fp, fn - b))
    out['target_classes'] = tg
    fpo = {}
    for name, c in D['fp'].items():
        fpo[name] = {v: dict(fp_share=k / fp, f1_if_removed=f1(tp, fp - k, fn)) for v, k in sorted(c.items())}
    out['fp_classes'] = fpo
    out['fn_state'] = {k: dict(unknown=v.get('unknown', 0) / max(sum(v.values()), 1), n=sum(v.values())) for k, v in D['fn_state'].items()}
    return out


SUM = {hop: summarize(D) for hop, D in R.items()}
json.dump(dict(summary=SUM, rows={h: D['f1_rows'] for h, D in R.items()}, labels=dict(height=HN, part=['thin_xy', 'thin_z', 'thin_other', 'bulk'],
          phase='0/1/2 = face distance to code-column edge (voxels); 7 = only vertical free face; 8 = no 6-free face',
          wall='bins of 2D distance to wall column: <=2, 2-4, 4-8, 8-16, >16 voxels', col_levels='z-levels with furniture in the 4x4 column: <=2, 3-5, 6-10, >10',
          dist='FP distance to nearest target furniture voxel: <=1, (1,1.75], (1.75,2], (2,4], >4', dist_world='to any gt_obj voxel: <=1, (1,2], (2,4], >4',
          adj='FP shares a horizontal face with target furniture at column-edge distance 0/1/2; 8 = only vertical; 9 = none')),
          open(os.path.join(OUT, 'anatomy.json'), 'w'), indent=1)
for hop, s in SUM.items():
    print(f"\n=== hop {hop}: official furn F1 {s['f1_official_mean']:.3f} pooled {s['f1_pooled']:.3f} R {s['recall']:.3f} P {s['precision']:.3f} FN/FP {s['fn_over_fp']:.2f} (tp {s['tp']} fp {s['fp']} fn {s['fn']})")
    for name, c in s['target_classes'].items():
        print(f"  [tgt] {name}: " + '; '.join(f"{v}: sh {d['share']:.3f} rec {d['recall']:.3f} fnsh {d['fn_share']:.3f} F1fix {d['f1_if_fn_fixed']:.3f}" for v, d in c.items()))
    for name, c in s['fp_classes'].items():
        print(f"  [fp] {name}: " + '; '.join(f"{v}: {d['fp_share']:.3f} (F1rm {d['f1_if_removed']:.3f})" for v, d in c.items()))
    print('  [fn state, share predicted unknown]', {k: round(v['unknown'], 3) for k, v in s['fn_state'].items()})
