"""R-A: TRAIN (E3-R cache plan) vs VAL/fresh house overlap: exact id / contour hash, and a near-duplicate proxy (2D free-mask IoU
over the 8 dihedral maps for pairs with matching extents). Review only; writes RA/split_check.json."""
import os, sys, json, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_noise_panels as G  # noqa: E402  (sets the HouseExpo env)
import mapping_houseexpo_data_v365 as v365  # noqa: E402
import mapping_houseexpo_data_v371 as v371  # noqa: E402
W = G.W

tr = [(r, i) for r, i in [(int(s['rooms']), int(s['index'])) for s in W.scene_plan('TRAIN', 70)]]
va = [(int(s['rooms']), int(s['index'])) for s in W.scene_plan('VALIDATION', 16)]
def info(split, r, i):
    s = v365.world_spec(split, i, r)
    return dict(split=split, rooms=r, index=i, id=s['house_id'], h=s['contour_sha256'])
TR = [info('TRAIN', r, i) for r, i in tr]; VA = [info('VALIDATION', r, i) for r, i in va]
ids = set(x['id'] for x in TR) & set(x['id'] for x in VA); hs = set(x['h'] for x in TR) & set(x['h'] for x in VA)

def mask(split, r, i):
    b = v371.world_spec(split, i, r); g = v371.rasterize(b, 128)
    free = g['walls'] == 0
    rr, cc = np.nonzero(free)
    return free[rr.min():rr.max() + 1, cc.min():cc.max() + 1], b['chart_m']

def dihedral(m):
    out = []
    for k in range(4):
        x = np.rot90(m, k); out += [x, x[::-1]]
    return out
M_tr = {(x['rooms'], x['index']): mask('TRAIN', x['rooms'], x['index']) for x in TR}
M_va = {(x['rooms'], x['index']): mask('VALIDATION', x['rooms'], x['index']) for x in VA}
best = []
for kv, (mv, Lv) in M_va.items():
    b = (0.0, None)
    for kt, (mt, Lt) in M_tr.items():
        if abs(Lv - Lt) > 1e-9:
            continue                                                   # different chart size -> different cell size; not a duplicate at this scale
        for d in dihedral(mt):
            if d.shape != mv.shape:
                continue
            iou = float((d & mv).sum()) / max(float((d | mv).sum()), 1.)
            if iou > b[0]:
                b = (iou, kt)
    best.append(dict(val=list(kv), best_iou=b[0], train=list(b[1]) if b[1] else None))
best.sort(key=lambda x: -x['best_iou'])
out = dict(n_train=len(TR), n_val=len(VA), id_overlap=sorted(ids), hash_overlap=sorted(hs), near_dup_top=best[:8],
           n_val_iou_ge_0p9=sum(1 for x in best if x['best_iou'] >= 0.9), n_val_iou_ge_0p8=sum(1 for x in best if x['best_iou'] >= 0.8))
json.dump(out, open(os.path.join(HERE, 'split_check.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'near_dup_top'}), json.dumps(out['near_dup_top'][:5]))
