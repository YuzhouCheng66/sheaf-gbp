"""reviewer: sha256 of the 192 VAL npz in g4/h1_export vs the seeds[s].file_sha256 of g4/h1/jobs (no scoring, no array read)"""
import os, json, glob, hashlib
D = os.path.dirname(os.path.abspath(__file__)); G4 = os.path.dirname(D)
bad, n = [], 0
for p in sorted(glob.glob(os.path.join(G4, 'h1', 'jobs', '*.json'))):
    r = json.load(open(p))
    for s, x in r['seeds'].items():
        f = os.path.join(G4, 'h1_export', x['file']); h = hashlib.sha256(open(f, 'rb').read()).hexdigest(); n += 1
        if h != x['file_sha256']:
            bad.append(x['file'])
print('files', n, 'mismatch', len(bad), bad[:5])
