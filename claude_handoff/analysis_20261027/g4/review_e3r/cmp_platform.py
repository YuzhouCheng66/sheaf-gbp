"""reviewer probe: are the remote (Linux) TRAIN cache maps bit-identical to the local (Windows) smoke cache maps for the same
TRAIN house / noise seed? (cross-platform reproducibility of the V3 pipeline, TRAIN only)"""
import sys, json, numpy as np
loc, rem = sys.argv[1], sys.argv[2]
for h in ('TRAIN_r1_i0', 'TRAIN_r1_i9'):
    a = np.load(f'{loc}/{h}.npz'); b = np.load(f'{rem}/{h}.npz')
    for k in a.files:
        if k in b.files:
            x, y = a[k], b[k]
            same = x.shape == y.shape and np.array_equal(x, y)
            print(h, k, x.shape, x.dtype, 'IDENTICAL' if same else f'DIFF ({int((x != y).sum()) if x.shape == y.shape else "shape"} entries)')
        else:
            print(h, k, 'missing remotely')
    print(h, 'remote-only arrays', sorted(set(b.files) - set(a.files)))
    rl = json.load(open(f'{loc}/records/{h}.json')); rr = json.load(open(f'{rem}/{h}.json'))
    print(h, 'private_sha equal', rl['private_sha256'] == rr['private_sha256'], 'world', rl['world_id'] == rr['world_id'], 'script', rr['script_sha256'][:8])
