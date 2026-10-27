"""c2_dlbound.py (moderator C2, 2026-10-27): the full-fidelity ceiling with EXACT poses, paired per map.
world_dl.json prices each pose triple at 44 bit, which A03 p11 / A10 / the coordinator's verify_recast show is NOT exact (1 cm rounding
leaves a median of 326 wrong voxels per map).  Re-price it at float32 (206 bit per triple: +20.25 B per agent in the map) and pair every
own map / R8 union with A03's p9 row of the same house and agent (asserting the ctx byte counts agree) to get per-map ratios
gzip / DL, ctx / DL and S0 / DL (S0 = A03's label-free structured exact description, also re-priced at float32 poses).
Output: probes/C2/c2_dlbound.json"""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.abspath(os.path.join(HERE, '..', '..'))
p9 = json.load(open(os.path.join(AN, 'probes', 'A03', 'p9_message_bytes.json')))['rows']
wd = json.load(open(os.path.join(AN, 'world_dl.json')))['rows']
X = (206 - 44) / 8.
res = {}
for kind, wk in (('own', 'own'), ('union', 'union8')):
    ph, wh = defaultdict(list), defaultdict(list)
    for r in p9:
        if r['kind'] == kind: ph[r['house']].append(r)
    for r in wd:
        if r['kind'] == wk: wh[r['house']].append(r)
    g, c, s, d = [], [], [], []
    for h in ph:
        assert len(ph[h]) == len(wh[h]), h
        for a, b in zip(ph[h], wh[h]):
            assert abs(a['ctx'] - b['ctx']) < 1e-3, (h, a['ctx'], b['ctx'])
            dl32 = b['dl'] + X * b.get('reached', 1)
            g.append(a['gzip'] / dl32); c.append(a['ctx'] / dl32); s.append((a['desc'] + X * a['agents']) / dl32); d.append(dl32)
    res[kind] = dict(n=len(g), dl32_median=float(np.median(d)), gzip_over_dl32=float(np.median(g)), ctx_over_dl32=float(np.median(c)),
                     S0_over_dl32=float(np.median(s)))
json.dump(res, open(os.path.join(HERE, 'c2_dlbound.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
