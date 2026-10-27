import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
mine = json.load(open(os.path.join(HERE, 'out', 'house.json')))
lines = []
worst = dict(furn=0., edge=0., B=0., Sk=0., lev=0.)
for r in mine:
    tag = r['file'].replace('.npz', '')
    impl = json.load(open(os.path.join(HERE, '..', 'e5_smoke', 'eval', 'jobs', tag + '.json')))
    assert impl['comps'] == r['comps']
    for sd in ('342101', '342202'):
        for arm in ('Qinf', 'Q3'):
            me = r['ck'][sd][arm]; im = impl['tiers']['cam_nf'][sd][arm]
            fm = [x['furn'] for x in me['rows']]; fi = [x[0]['furn_f1'] for x in im['R8_net']]
            df = max(abs(a - b) for a, b in zip(fm, fi))
            worst['furn'] = max(worst['furn'], df)
            de = 0.
            for e, v in me['edges'].items():
                w = im['bytes']['edges'][e]
                de = max(de, max(abs(a - b) for a, b in zip(v, w)))
            worst['edge'] = max(worst['edge'], de)
            s = f"{tag} {sd} {arm}: furn own {fm} impl {fi} |d| {df:.1e}; edges max |d| {de:.2e} B; Sk own {me['Sk']:.3f} impl {im['bytes']['Sk']:.3f}"
            if arm == 'Q3':
                s += f"; lev own {me['lev']:.3f} impl {im['bytes']['lev']:.3f}; B own {me['B']:.3f} impl {im['bytes']['B']:.3f} mode {im['bytes']['mode']}"
                worst['B'] = max(worst['B'], abs(me['B'] - im['bytes']['B'])); worst['lev'] = max(worst['lev'], abs(me['lev'] - im['bytes']['lev']))
                s += f"; ratio B(Q3)/B(Qinf) own {me['B'] / r['ck'][sd]['Qinf']['Sk']:.4f} impl {im['bytes']['B'] / impl['tiers']['cam_nf'][sd]['Qinf']['bytes']['B']:.4f}"
            worst['Sk'] = max(worst['Sk'], abs(me['Sk'] - im['bytes']['Sk']))
            lines.append(s)
        # known-mask double coding: k-message bytes vs total for Qinf
        me = r['ck'][sd]['Qinf']
        lines.append(f"{tag} {sd} Qinf: S bytes {me['S_sum']:.0f}, k bytes {me['k_sum']:.0f} ({100 * me['k_sum'] / me['Sk']:.1f} % of Sk)")
lines.append('max abs differences: ' + json.dumps({k: float(f'{v:.3e}') for k, v in worst.items()}))
print('\n'.join(lines))
