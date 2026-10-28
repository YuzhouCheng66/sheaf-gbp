"""ri_k0agents.py (R-I probe): per-agent K0 furniture counts on VAL seed 1 cam_nf for E3L3 and NET (both checkpoints), to split the
K0 row-mean gap into 'small agents' and 'pooled voxels'. Writes RI/out_k0/<house>.json."""
import os, sys, json, multiprocessing  # noqa: E401
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ri_probe as P  # noqa: E402
import numpy as np  # noqa: E402

OUTK = os.path.join(HERE, 'out_k0')


def job(h):
    rooms, index = h
    M = P.load_all(3)
    z = np.load(os.path.join(P.EXPORT, f'val_r{rooms}_i{index}_s1.npz'))
    H = P.G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    K0T = [P.G.Target(private[a], H) for a in range(n)]
    Lk = list(z['L_k0_cam_nf'])
    F = P.G.forest(P.R.edges_list(z['edges']), n)
    rd = P.readouts(Lk, F, M, Lk)
    out = []
    for a in range(n):
        T = K0T[a]; row = dict(agent=a, n_true=int(T.tr.sum()))
        for arm in ('NET_342101', 'NET_342202', 'E3L3_342101', 'E3L3_342202'):
            p = rd['K0'][arm][a]; pr = (p == 1) & T.known & H.region
            row[arm] = [int((pr & T.tr).sum()), int((pr & ~T.tr).sum()), int((T.tr & ~pr).sum())]
        out.append(row)
    os.makedirs(OUTK, exist_ok=True)
    json.dump(out, open(os.path.join(OUTK, f'r{rooms}_i{index}.json'), 'w'))
    return h


if __name__ == '__main__':
    hs = [(r, i) for r in (1, 2, 3) for i in range(8)]
    with multiprocessing.get_context('spawn').Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4, initializer=P._init) as pool:
        for h in pool.imap_unordered(job, hs):
            print(h, flush=True)
