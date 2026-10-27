"""c2_checks.py (moderator C2, 2026-10-27): independent re-computation of the few numbers the synthesis leans on hardest.
CPU only, one process, < 2 GB.  Reads only existing files; writes probes/C2/c2_checks.json.

Part A (6 demo VAL houses x 8 agents, demo_web3d/data, same loader / metric copy as rate_baselines_lib):
  A1  the free-space loophole: predict occupied exactly on the target's occupied voxels, everything else 'unknown'
      -> surface / var / furniture-region F1, unknown->known FPR, free recall, 3-way accuracy (K0 and R8);
      the same readings for the exported 3S-X + calibrated predictions.
  A2  observation multiplicity of the R8 union: mu = sum_i |known_i| / |U known_i| over the reached set, share seen by >= 2 agents;
      this is the factor by which flooding own maps costs more than relaying unions, independent of the codec.
Part B (24 VAL houses, A03's p9_message_bytes.json rows + world_dl.json rows): one panel, one protocol at a time.
  per-message medians (own / directed-edge sender union / R8 union) for raw, gzip, ctx, A03 structured description re-priced with
  float32 poses (206 bit per pose triple instead of 44), parametric DL re-priced the same way;
  per-house network bytes: relay (every directed edge carries its converged sender-side union once) and flood (every own map crosses
  every edge of its component once), for gzip / ctx / structured; current protocol 2,048 B x 8 rounds x directed edges.
  -> coding factor (same protocol) and protocol factor (same codec) are reported separately."""
import glob, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = os.path.abspath(os.path.join(AN, '..', '..'))
sys.path.insert(0, AN)
import rate_baselines_lib as L

DIMS = L.DIMS
out = dict(A1=dict(), A2=dict(), B=dict())

# ---------------- Part A
rowsA = {k: [] for k in ('S_K0', 'S_R8', 'ours_K0', 'ours_R8')}
mult = []
for path in sorted(glob.glob(os.path.join(ROOT, 'demo_web3d', 'data', 'house_*.json'))):
    m, own, pred, gt = L.load(path)
    n = own.shape[0]
    slab = torch.from_numpy(gt['slab'].reshape(-1)); obj = torch.from_numpy(gt['obj'].reshape(-1))
    struct = torch.from_numpy((gt['occ'] & ~gt['obj'] & ~gt['slab']).reshape(-1))
    hi8 = m['hops'].index(8); R = np.asarray(m['reached'][hi8], bool)
    T8 = L.targets(own, R)
    for name, T, P in (('K0', own, pred[0]), ('R8', T8, pred[8])):
        S = np.where(T == 1, np.int8(1), np.int8(-1))
        for tag, Q in (('S', S), ('ours', P)):
            met = L.metrics(torch.from_numpy(Q.reshape(n, -1)), torch.from_numpy(T.reshape(n, -1)), slab, obj, struct)
            for a in range(n):
                t = T[a]; q = Q[a]; known = t >= 0
                free = t == 0
                r = dict(met[a])
                r['fpr'] = float((q[~known] >= 0).mean()) if (~known).any() else None
                r['free_recall'] = float((q[free] == 0).mean()) if free.any() else None
                r['acc3'] = float((q[known] == t[known]).mean())
                rowsA[f'{tag}_{name}'].append(r)
    for a in range(n):                                     # multiplicity of the R8 union seen by agent a
        K = (own[R[a]] >= 0)
        cnt = K.sum(0)
        U = cnt > 0
        mult.append(dict(house=m['house'], agent=a, reached=int(R[a].sum()), mu=float(cnt[U].sum() / U.sum()), share_ge2=float((cnt[U] >= 2).mean())))
    print(m['house'], 'done', flush=True)

for k, v in rowsA.items():
    out['A1'][k] = {kk: float(np.mean([r[kk] for r in v if r[kk] is not None])) for kk in ('surface_f1', 'var_occ_f1', 'furn_f1', 'fpr', 'free_recall', 'acc3')}
full = [r for r in mult if r['reached'] == 8]
out['A2'] = dict(rows=len(mult), rows_reached8=len(full), mu_mean=float(np.mean([r['mu'] for r in mult])), mu_median=float(np.median([r['mu'] for r in mult])),
                 share_ge2_mean=float(np.mean([r['share_ge2'] for r in mult])))

# ---------------- Part B
p9 = json.load(open(os.path.join(AN, 'probes', 'A03', 'p9_message_bytes.json')))['rows']
wd = json.load(open(os.path.join(AN, 'world_dl.json')))['rows']
POSE_EXTRA = (206 - 44) / 8.                               # float32 pose triple instead of 44 bit (A03 p11: 48/48 exact)
for r in p9:
    r['desc32'] = r['desc'] + r['agents'] * POSE_EXTRA
houses = sorted({r['house'] for r in p9})
med = {}
for kind in ('own', 'edge', 'union'):
    rr = [r for r in p9 if r['kind'] == kind]
    med[kind] = dict(n=len(rr), raw=131072, gzip=float(np.median([r['gzip'] for r in rr])), ctx=float(np.median([r['ctx'] for r in rr])),
                     desc32=float(np.median([r['desc32'] for r in rr])), desc44=float(np.median([r['desc'] for r in rr])),
                     ratio_ctx=float(np.median([r['ctx'] / r['desc32'] for r in rr])), ratio_gzip=float(np.median([r['gzip'] / r['desc32'] for r in rr])),
                     ratio_raw=float(np.median([131072 / r['desc32'] for r in rr])), ratio_2048=float(np.median([2048 / r['desc32'] for r in rr])))
for kind, wk in (('own', 'own'), ('union', 'union8')):
    rr = [r for r in wd if r['kind'] == wk]
    if rr:
        med[kind]['dl44'] = float(np.median([r['dl'] for r in rr]))
        nag = 1 if kind == 'own' else None
        if kind == 'own':
            med[kind]['dl32'] = float(np.median([r['dl'] + POSE_EXTRA for r in rr]))
out['B']['per_message_median'] = med
out['B']['world_dl_kinds'] = sorted({r['kind'] for r in wd})

net = {k: [] for k in ('current_fixed', 'current_once', 'gzip_flood', 'gzip_relay', 'ctx_flood', 'ctx_relay', 'desc32_flood', 'desc32_relay')}
for h in houses:
    E = [r for r in p9 if r['house'] == h and r['kind'] == 'edge']
    O = [r for r in p9 if r['house'] == h and r['kind'] == 'own']
    U = [r for r in p9 if r['house'] == h and r['kind'] == 'union']
    assert len(O) == len(U)
    fl = [u['agents'] - 1 for u in U]                       # own map a crosses every edge of its component once
    nd = len(E)
    net['current_fixed'].append(2048 * 8 * nd); net['current_once'].append(2048 * nd)
    for c in ('gzip', 'ctx', 'desc32'):
        net[f'{c}_relay'].append(sum(r[c] for r in E))
        net[f'{c}_flood'].append(sum(o[c] * f for o, f in zip(O, fl)))
net = {k: np.asarray(v, float) for k, v in net.items()}
out['B']['network_median_per_house'] = {k: float(np.median(v)) for k, v in net.items()}
out['B']['network_mean_per_house'] = {k: float(np.mean(v)) for k, v in net.items()}
R = lambda a, b: float(np.median(net[a] / net[b]))
out['B']['factors_median_over_houses'] = dict(
    protocol_gzip_flood_over_relay=R('gzip_flood', 'gzip_relay'), protocol_ctx_flood_over_relay=R('ctx_flood', 'ctx_relay'),
    protocol_desc_flood_over_relay=R('desc32_flood', 'desc32_relay'),
    coding_relay_gzip_over_desc=R('gzip_relay', 'desc32_relay'), coding_relay_ctx_over_desc=R('ctx_relay', 'desc32_relay'),
    coding_flood_gzip_over_desc=R('gzip_flood', 'desc32_flood'),
    total_gzipflood_over_descrelay=R('gzip_flood', 'desc32_relay'), total_ctxflood_over_descrelay=R('ctx_flood', 'desc32_relay'),
    current_fixed_over_descrelay=R('current_fixed', 'desc32_relay'), current_once_over_descrelay=R('current_once', 'desc32_relay'),
    current_fixed_over_gzipflood=R('current_fixed', 'gzip_flood'), current_fixed_over_ctxrelay=R('current_fixed', 'ctx_relay'),
    raw_flood_over_descrelay=float(np.median(np.array([131072. * sum(u['agents'] - 1 for u in [r for r in p9 if r['house'] == h and r['kind'] == 'union']) for h in houses]) / net['desc32_relay'])))
json.dump(out, open(os.path.join(HERE, 'c2_checks.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
