"""p9b_exact_pose_bytes.py (A03, debate round): p9's per-message bytes re-priced with exact-pose costs (p11): every pose triple costs
B bit instead of 44 (B = 110 for 16-bit jitters, 206 for float32). Prints medians and multiples for own / edge / union and the
protocol total per house. Usage: python -B p9b_exact_pose_bytes.py 110"""
import json, os, sys
import numpy as np
from a03np import HERE

B = int(sys.argv[1]) if len(sys.argv) > 1 else 110
d = json.load(open(os.path.join(HERE, 'p9_message_bytes.json')))
rows = d['rows']
out = {}
for kind in ('own', 'edge', 'union'):
    rr = [r for r in rows if r['kind'] == kind]
    desc = np.array([r['desc'] + r['agents'] * (B - 44) / 8 for r in rr])
    ctx = np.array([r['ctx'] for r in rr]); gz = np.array([r['gzip'] for r in rr])
    out[kind] = dict(n=len(rr), desc_median=float(np.median(desc)), desc_p90=float(np.percentile(desc, 90)), agents_median=float(np.median([r['agents'] for r in rr])),
                     ctx_ratio=float(np.median(ctx / desc)), gzip_ratio=float(np.median(gz / desc)), raw_ratio=float(np.median(131072 / desc)), ours_ratio=float(np.median(2048 / desc)))
houses = sorted({r['house'] for r in rows})
tot = [sum(r['desc'] + r['agents'] * (B - 44) / 8 for r in rows if r['house'] == h and r['kind'] == 'edge') for h in houses]
out['protocol_total_median'] = float(np.median(tot))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, f'p9b_exact_pose_bytes_{B}.json'), 'w'), indent=1)
