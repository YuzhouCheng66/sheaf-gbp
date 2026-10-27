"""verdict() against the REAL E3-R summary (local copy): E3-L rows := the control's NET rows (+ bytes), so every cell has d = 0"""
import os, sys, json, copy, tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('../../../../scripts/claude_v342'))
import g4_e3l as L
ctrl = os.path.abspath('../e3r/summary.json')
Sc = json.load(open(ctrl))
def build(ratio=0.10, dK=0.0, cam=0.0):
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[], gbp={f'E3L_{t}': dict(cf_max=1e-12, violations=0, components=5, pass_=True) for t in ('342101', '342202')})
    for g in S['gbp'].values():
        g['pass'] = g.pop('pass_')
    for p, P in Sc['panels'].items():
        S['panels'][p] = dict(seeds={})
        for ns, D in P['seeds'].items():
            T = {}
            for tn, d in D['tiers'].items():
                T[tn] = {}
                for ts in ('342101', '342202'):
                    a = copy.deepcopy(d[f'NET_{ts}'])
                    if tn == 'cam_nf':
                        a['bytes_median'] = ratio * (a['rnet_relay_median']['sum'] + a['rnet_relay_median']['count']); a['K0_sep']['furn_f1'] += dK
                    if tn == 'cam':
                        a['R8']['furn_f1'] += cam
                    T[tn][f'E3L_{ts}'] = a
            S['panels'][p]['seeds'][ns] = dict(houses=D['houses'], tiers=T)
    return S
d = tempfile.mkdtemp()
runs = []
for sd in ('342101', '342202'):
    r = os.path.join(d, sd); os.makedirs(r)
    json.dump(dict(seed=int(sd), train_houses=186, recipe=dict(evidence=dict(desc=dict(spec='c:x:A:cam_nf')))), open(os.path.join(r, 'run.json'), 'w'))
    json.dump(dict(complete=True, truncated=False, updates_done=5000, updates_planned=5000), open(os.path.join(r, 'final.json'), 'w'))
    json.dump(dict(n_houses=19), open(os.path.join(r, 'threshold.json'), 'w'))
    runs.append(r)
out = {}
for name, kw in (('equal_ratio0.10', {}), ('ratio0.16', dict(ratio=0.16)), ('ratio0.51', dict(ratio=0.51)), ('K0_-0.031', dict(dK=-0.031)), ('cam_-0.031', dict(cam=-0.031))):
    v = L.verdict(build(**kw), runs, ctrl)
    out[name] = v['verdict']; print(name, '->', v['verdict'])
v = L.verdict(build(), runs, ctrl)
c = v['cells']['342101|1']
print('cell 342101|1 guards', c['guards'], 'clean', round(c['clean_R8_furn'], 4), 'control bytes', round(c['control_bytes']), 'fresh', v['fresh'])
json.dump(out, open('verdict_realctrl.json', 'w'), indent=1)
