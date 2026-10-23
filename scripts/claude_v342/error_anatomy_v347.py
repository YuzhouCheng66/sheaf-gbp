"""Error anatomy of the v347 max-fusion endpoint on the saved VALIDATION/fresh panels (idx 0 and 8 of every stratum).
Splits the pixel-exact wall errors into (i) 1-px localisation jitter, (ii) hallucinated walls far from any wall (>1 px),
(iii) missed walls, and attributes each to door cells / near-door band / far from doors, and to blocks observed by 1 vs >=2 sources.
python error_anatomy_v347.py --source <sealed> --panels <dir> [--out json]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--panels', required=True); ap.add_argument('--out', default='')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
from scripts import mapping_multiroom_data_v293 as data

def f1(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1)), int(tp), int(fp), int(fn)

rows = []
agg = {}
def add(key, name, val):
    agg.setdefault(key, {}).setdefault(name, []).append(val)
for f in sorted(glob.glob(str(Path(args.panels) / '*.npz'))):
    name = Path(f).name                       # final-val-panel-r6-n8-i0.npz
    parts = name[:-4].split('-')
    rooms = int(parts[3][1:]); agents = int(parts[4][1:]); index = int(parts[5][1:])
    split = 'VALIDATION'
    fx = data.make_fixture(split=split, index=index, size=128, rooms=rooms, agents=agents)
    door = fx['evaluation']['door_mask'].astype(bool); walls = fx['inputs']['private']  # per-agent private maps
    gt_walls_world = (fx['evaluation']['walls'] if 'walls' in fx['evaluation'] else None)
    d = np.load(f)
    own = d['own']; n = own.shape[0]
    # per-block source count (how many agents observed each 16x16 block)
    obs = (own >= 0).reshape(n, 8, 16, 8, 16).any((2, 4))          # [N,8,8]
    nsrc = obs.sum(0)                                                # [8,8]
    nsrc_px = np.kron(nsrc, np.ones((16, 16), int))                  # [128,128]
    door_band = ndimage.binary_dilation(door, iterations=3)
    for k in (0, 8):
        tgt = d[f'target_k{k}']; pred = d[f'pred_k{k}']
        for a in range(n):
            known = tgt[a] >= 0
            gt_w = tgt[a] == 1; pr_w = (pred[a] == 1) & known      # evaluate inside target-known (as node_metrics does for wall F1 over known)
            if gt_w.sum() == 0:
                continue
            F, tp, fp, fn = f1(pr_w, gt_w)
            gt_d = ndimage.binary_dilation(gt_w, iterations=1); pr_d = ndimage.binary_dilation(pr_w, iterations=1)
            tp_tol = (pr_w & gt_d).sum(); fp_far = (pr_w & ~gt_d).sum(); fn_far = (gt_w & ~pr_d).sum()
            F_tol = float(2 * tp_tol / max(2 * tp_tol + fp_far + fn_far, 1))
            # attribution of far errors
            fp_far_m = pr_w & ~gt_d; fn_far_m = gt_w & ~pr_d
            key = (k,)
            add(key, 'F1_exact', F); add(key, 'F1_tol1', F_tol)
            add(key, 'fp_far_frac_of_pred', fp_far / max(pr_w.sum(), 1)); add(key, 'fn_far_frac_of_gt', fn_far / max(gt_w.sum(), 1))
            add(key, 'fp_far_in_door', (fp_far_m & door).sum() / max(fp_far, 1)); add(key, 'fp_far_in_doorband', (fp_far_m & door_band).sum() / max(fp_far, 1))
            add(key, 'fn_far_in_doorband', (fn_far_m & door_band).sum() / max(fn_far, 1))
            add(key, 'fp_far_multi_src', (fp_far_m & (nsrc_px >= 2)).sum() / max(fp_far, 1)); add(key, 'fn_far_multi_src', (fn_far_m & (nsrc_px >= 2)).sum() / max(fn_far, 1))
            add(key, 'gt_wall_multi_src_frac', (gt_w & (nsrc_px >= 2)).sum() / max(gt_w.sum(), 1))
            # door cells: fraction predicted free / wall / unknown among known door cells
            dk = door & known
            if dk.sum():
                add(key, 'door_pred_free', ((pred[a] == 0) & dk).sum() / dk.sum()); add(key, 'door_pred_wall', ((pred[a] == 1) & dk).sum() / dk.sum()); add(key, 'door_pred_unknown', ((pred[a] < 0) & dk).sum() / dk.sum())
                add(key, 'door_gt_free', ((tgt[a] == 0) & dk).sum() / dk.sum())
            # jitter: exact-miss pixels that are within 1 px of a predicted wall
            add(key, 'fn_jitter_frac_of_gt', ((gt_w & ~pr_w & pr_d).sum()) / max(gt_w.sum(), 1))
            add(key, 'fp_jitter_frac_of_pred', ((pr_w & ~gt_w & gt_d).sum()) / max(pr_w.sum(), 1))
out = {str(k[0]): {m: float(np.mean(v)) for m, v in d.items()} for k, d in agg.items()}
for k, d in out.items():
    print(f'== stage R{k}')
    for m, v in d.items():
        print(f'   {m:28s} {v:.4f}')
if args.out:
    json.dump(out, open(args.out, 'w'), indent=1)
