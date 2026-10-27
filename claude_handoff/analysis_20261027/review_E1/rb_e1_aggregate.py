"""aggregate rb_e1_probe outputs: python -B rb_e1_aggregate.py DIR [DIR ...]  -> prints tables, writes DIR0/../rb_e1_aggregate.json"""
import glob, json, os, sys  # noqa: E401
import numpy as np

dirs = sys.argv[1:]
files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, 'TRAIN_*.json')))
R = [json.load(open(f)) for f in files]
print('houses', len(R), [os.path.basename(os.path.dirname(f)) + '/' + os.path.basename(f)[6:-5] for f in files])
med = lambda x: float(np.median(x))  # noqa: E731
mean = lambda x: float(np.mean(x))  # noqa: E731
out = {'n_houses': len(R), 'files': files}
# own decomposition: per agent, pooled over houses, median
A = [o for r in R for o in r['own']]
def g(f):
    return med([f(o) for o in A])
dec = dict(
    b4_total=g(lambda o: o['noisy']['total']), b4_pub_total=g(lambda o: o['noisy_pub_total']),
    known=g(lambda o: o['noisy']['known']), tern=g(lambda o: o['noisy']['tern']),
    occ_given_known=g(lambda o: o['noisy']['tern'] - o['noisy']['known']),
    mag_free=g(lambda o: sum(o['noisy']['mag']['free'].values())), mag_occ=g(lambda o: sum(o['noisy']['mag']['occ'].values())),
    mag_total=g(lambda o: sum(o['noisy']['mag']['free'].values()) + sum(o['noisy']['mag']['occ'].values())),
    clean_known=g(lambda o: o['clean_known']), clean_tern=g(lambda o: o['clean_tern']), mix_tern=g(lambda o: o['mix_tern']),
    known_noise=g(lambda o: o['mix_tern'] - o['clean_tern']), occ_noise=g(lambda o: o['noisy']['tern'] - o['mix_tern']),
    mag_share=g(lambda o: (o['noisy']['total'] - o['noisy']['tern']) / o['noisy']['total']),
    n_known=g(lambda o: o['n_known']), n_known_xor=g(lambda o: o['n_known_xor']), n_occ=g(lambda o: o['n_occ']), n_occ_clean=g(lambda o: o['n_occ_clean']),
    occ_fp_in_free=g(lambda o: o['occ_fp_in_free']), occ_in_clean_unknown=g(lambda o: o['occ_in_clean_unknown']), occ_fn=g(lambda o: o['occ_fn']))
h = np.sum([o['hist_L'] for o in A], 0); dec['hist_L_frac'] = (h / h.sum()).round(4).tolist()
out['own_decomposition_per_agent_median'] = dec
print('\nown map per agent (median, bytes):')
for k, v in dec.items():
    print(f'  {k:22s} {v}')
out['sweep'] = {}
for ms in ('pub', 'truepose'):
    out['sweep'][ms] = {}
    print(f'\n[{ms}] mean over houses of R8 metrics; median bytes per house')
    print('  scheme       furn   tol1   surf   var    fpr     free   | relay   CD(up int+down tern)  CD-innov   own(8 agents)')
    for q in R[0]['sweep'][ms]:
        V = [r['sweep'][ms][q] for r in R]
        m = {k: mean([v['metrics'][k] for v in V if v['metrics'][k] is not None]) for k in ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'var_occ_f1', 'unknown_known_fpr', 'free_recall')}
        b = dict(rnet=med([v['rnet'] for v in V]), own=med([v['own'] for v in V]))
        if 'rnet_cd_up_int_down_tern' in V[0]:
            b['cd'] = med([v['rnet_cd_up_int_down_tern'] for v in V]); b['cd_up'] = med([v['cd_up'] for v in V]); b['cd_down'] = med([v['cd_down'] for v in V])
        if 'rnet_cd_innov' in V[0]:
            b['cd_innov'] = med([v['rnet_cd_innov'] for v in V]); b['cd_down_innov'] = med([v['cd_down_innov'] for v in V])
        if V[0].get('rnet_tern_part') is not None:
            b['tern_part'] = med([v['rnet_tern_part'] for v in V])
        thr = {}
        if V[0].get('vote_thresholds'):
            for t in V[0]['vote_thresholds']:
                thr[t] = {k: mean([v['vote_thresholds'][t][k] for v in V if v['vote_thresholds'][t][k] is not None]) for k in ('furn_f1', 'unknown_known_fpr', 'free_recall')}
        out['sweep'][ms][q] = dict(metrics=m, bytes=b, thresholds=thr)
        print(f"  {q:11s} {m['furn_f1']:.3f}  {m['furn_f1_tol1']:.3f}  {m['surface_f1']:.3f}  {m['var_occ_f1']:.3f}  {m['unknown_known_fpr']:.4f}  {m['free_recall']:.3f}  | {b['rnet']:8.0f} {b.get('cd', float('nan')):8.0f} ({b.get('cd_up', float('nan')):.0f}+{b.get('cd_down', float('nan')):.0f})  {b.get('cd_innov', float('nan')):8.0f}  {b['own']:8.0f}  {b.get('tern_part', '')}")
        for t, mm in thr.items():
            print(f'      vote threshold S > {t}: furn {mm["furn_f1"]:.3f} fpr {mm["unknown_known_fpr"]:.4f} free {mm["free_recall"]:.3f}')
# paired vote - full per house
for ms in ('pub', 'truepose'):
    d = [r['sweep'][ms]['vote_k2']['metrics']['furn_f1'] - r['sweep'][ms]['full']['metrics']['furn_f1'] for r in R]
    dfr = [r['sweep'][ms]['vote_k2']['metrics']['free_recall'] - r['sweep'][ms]['full']['metrics']['free_recall'] for r in R]
    out[f'paired_vote_minus_full_{ms}'] = dict(furn=d, free=dfr)
    print(f'\npaired vote - full [{ms}] furn pp: mean {100*np.mean(d):+.2f}, min {100*np.min(d):+.2f}, max {100*np.max(d):+.2f}; free recall pp mean {100*np.mean(dfr):+.2f}')
dst = os.path.join(os.path.dirname(os.path.abspath(dirs[0].rstrip('/\\'))), 'rb_e1_aggregate_' + '_'.join(os.path.basename(d.rstrip('/\\')) for d in dirs) + '.json')
json.dump(out, open(dst, 'w'), indent=1)
print('wrote', dst)
