"""print the p2 summary compactly (A09)"""
import json, os
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p2_raycast.json')))['summary']
for k in ('centre_S', 'centre_GT', 'centre_pred', 'learned_pred'):
    print(k, {kk: round(v, 4) for kk, v in d[k].items()})
for j in d['jitter']:
    print('jitter', j)
