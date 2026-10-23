"""Paired per-world statistics for the v342 matched pair (owned vs dense), H128.
Usage: python3 paired_stats_v342.py <owned_run_dir> <dense_run_dir> [final-val|final-fresh] [B=10000]
Reads <run>/validation/<label>.json written by train_v342_owned.py (rows[i].spec, rows[i].hops[str(k)][metric]).
Unit of resampling = WORLD (rooms, index): both N=8 and N=16 scenes of a drawn world are kept together
(stratified cluster bootstrap: 8 worlds drawn with replacement inside each rooms stratum).
Point estimate = equal-weight mean over the 6 (rooms,N) strata of the per-scene paired difference,
which equals the difference of the two reported 'all' numbers.
"""
import json, sys, itertools
import numpy as np

def load(run, label):
    d = json.load(open(f'{run}/validation/{label}.json'))
    return {(r['spec']['rooms'], r['spec']['agents'], r['spec']['index']): r for r in d['rows']}

def macro(delta_by_scene, scenes):
    strata = {}
    for key in scenes:
        strata.setdefault((key[0], key[1]), []).append(delta_by_scene[key])
    return float(np.mean([np.mean(v) for v in strata.values()]))

def paired(owned, dense, stage, metric, B, seed=342):
    keys = sorted(set(owned) & set(dense))
    assert len(keys) == len(owned) == len(dense), 'scene lists differ'
    d = {}
    for k in keys:
        a, b = owned[k]['hops'][str(stage)][metric], dense[k]['hops'][str(stage)][metric]
        d[k] = (a if a is not None else np.nan) - (b if b is not None else np.nan)
    point = macro(d, keys)
    worlds = sorted({(k[0], k[2]) for k in keys})                       # (rooms, index): 24 clusters
    by_rooms = {}
    for w in worlds:
        by_rooms.setdefault(w[0], []).append(w)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(B):
        drawn = []
        for rooms, ws in by_rooms.items():
            pick = rng.integers(len(ws), size=len(ws))
            drawn += [ws[i] for i in pick]
        scenes = [k for w in drawn for k in keys if (k[0], k[2]) == w]  # keeps both N of each drawn world (duplicates allowed)
        boots.append(macro(d, scenes))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    world_means = {w: np.nanmean([d[k] for k in keys if (k[0], k[2]) == w]) for w in worlds}
    pos = sum(v > 0 for v in world_means.values())
    strata = {}
    for k in keys:
        strata.setdefault(f'rooms{k[0]}/N{k[1]}', []).append(d[k])
    return dict(metric=metric, stage=stage, point=point, ci95=(float(lo), float(hi)),
                worlds_positive=f'{pos}/{len(worlds)}', per_stratum={s: float(np.nanmean(v)) for s, v in strata.items()},
                scene_sd=float(np.nanstd(list(d.values()))))

def within(run_rows, stage_a, stage_b, metric, B):
    """Paired difference of one arm between two stages (G1: own_wall_f1 R8 - R0)."""
    fake = {k: dict(hops={str(stage_a): {metric: r['hops'][str(stage_b)][metric]}}) for k, r in run_rows.items()}
    return paired(run_rows, fake, stage_a, metric, B)

if __name__ == '__main__':
    owned_dir, dense_dir = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else 'final-val'
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    owned, dense = load(owned_dir, label), load(dense_dir, label)
    out = {}
    out['G2_R8_wall_f1'] = paired(owned, dense, 8, 'wall_f1', B)
    out['G3_R8_fpr'] = paired(owned, dense, 8, 'unknown_known_fpr', B)
    out['K0_wall_f1'] = paired(owned, dense, 0, 'wall_f1', B)
    out['R1_wall_f1'] = paired(owned, dense, 1, 'wall_f1', B)
    out['R8_own_wall_f1'] = paired(owned, dense, 8, 'own_wall_f1', B)
    out['R8_union_accuracy'] = paired(owned, dense, 8, 'union_accuracy', B)
    out['G1_owned_own_R8_minus_K0'] = within(owned, 8, 0, 'own_wall_f1', B)
    out['G1_dense_own_R8_minus_K0'] = within(dense, 8, 0, 'own_wall_f1', B)
    g2 = out['G2_R8_wall_f1']; g3 = out['G3_R8_fpr']
    out['verdict'] = dict(
        G2_pass=bool(g2['ci95'][0] >= 0.05 and int(g2['worlds_positive'].split('/')[0]) >= 21 and all(v > 0 for v in g2['per_stratum'].values())),
        G3_pass=bool(g3['ci95'][1] <= 0.005),
        G1_owned_fail_confirmed=bool(out['G1_owned_own_R8_minus_K0']['ci95'][1] < -0.02))
    print(json.dumps(out, indent=1))
