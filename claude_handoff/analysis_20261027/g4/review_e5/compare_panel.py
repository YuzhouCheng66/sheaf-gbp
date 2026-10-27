import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
mine = json.load(open(os.path.join(HERE, 'out', 'panel.json')))
S = json.load(open(os.path.join(HERE, '..', 'e3r', 'summary.json')))
ref = S['panels']['val']['seeds']['1']['tiers']['cam_nf']
print(f'houses: {len(mine)}')
for sd in ('342101', '342202'):
    rows = [x for r in mine for x in r['ck'][sd]['Qinf']['rows'] for _ in range(x['size'])]
    furn = float(np.mean([x['furn'] for x in rows if x['furn'] is not None])); free = float(np.mean([x['free'] for x in rows if x['free'] is not None]))
    sb = float(np.median([r['ck'][sd]['Qinf']['S_sum'] for r in mine])); kb = float(np.median([r['ck'][sd]['Qinf']['k_sum'] for r in mine]))
    skm = float(np.median([r['ck'][sd]['Qinf']['Sk'] for r in mine]))
    R = ref[f'NET_{sd}']
    print(f"{sd}: R8 NET furn own {furn!r} E3-R {R['R8']['furn_f1']!r} diff {furn - R['R8']['furn_f1']:.1e}; free own {free!r} E3-R {R['R8']['free_recall']!r}; "
          f"median S bytes own {sb!r} E3-R {R['rnet_relay_median']['sum']!r}; median k bytes own {kb!r} E3-R {R['rnet_relay_median']['count']!r}; median Sk own {skm:.1f}")
