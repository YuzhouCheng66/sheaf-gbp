"""rb_pooled.py (reviewer R-B, 2026-10-27): pooled per-message medians from rb_wire_cd.json (48 own maps, 84 directed edges per grid),
so they are directly comparable with A01 probe_wire (which pools the same way): K0 message, converged message, CD collect / distribute.
Usage: python -B rb_pooled.py  (after rb_wire_cd.py)"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'rb_wire_cd.json')))
res = {}
for G in (32, 64):
    R = [r for r in d['rows'] if r['G'] == G]
    k0 = np.concatenate([r['k0'] for r in R]); k0a = np.concatenate([r['k0_addr'] for r in R])
    cv = np.concatenate([r['conv'] for r in R]); cva = np.concatenate([r['conv_addr'] for r in R])
    cd = [x for r in R for x in r['cd']]
    up = [x for x in cd if x['kind'] == 'collect']; dn = [x for x in cd if x['kind'] == 'distribute']
    for b in (13.63, 10.0, 7.0):
        res[f'G{G}_b{b}'] = dict(k0_pooled_median=float(np.median(k0 * b / 8 + k0a / 8 + 2)),
                                 conv_pooled_median=float(np.median(cv * b / 8 + cva / 8 + 2)),
                                 cd_collect_msg_median=float(np.median([x['k'] * b / 8 + x['addr'] / 8 + 2 for x in up])),
                                 cd_distribute_msg_median=float(np.median([x['k'] * b / 8 + x['addr'] / 8 + 2 for x in dn])),
                                 k0_evid_median=float(np.median(k0)), conv_evid_median=float(np.median(cv)),
                                 cd_up_stalks_median=float(np.median([x['k'] for x in up])), cd_down_stalks_median=float(np.median([x['k'] for x in dn])))
    H = d[f'houses_G{G}']
    res[f'G{G}_cd_share_collect'] = float(np.sum([h['cd_collect_stalks'] for h in H]) / np.sum([h['cd_stalks'] for h in H]))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(HERE, 'rb_pooled.json'), 'w'), indent=1)
