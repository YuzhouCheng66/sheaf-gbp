"""Q2 add-on: where do missed furniture voxels go in the formal E3-L read-out -- predicted UNKNOWN (known-head veto) or predicted FREE
(occupancy head / threshold)? Also F1 with the known head replaced by the target-known mask (oracle, diagnosis only), and with the
occupancy threshold lowered to t = 0 (logit 0, i.e. p = .5; no VAL tuning). VAL cam_nf seed 1, formal path only. -> rd_out/q2_fnsplit.json"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rd_probe as P  # noqa: E402
import torch  # noqa: E402
L = P.L; R = P.R; G = P.G


@torch.no_grad()
def main(jobs):
    acc = {}
    for job in jobs:
        t0 = time.time()
        z = np.load(os.path.join(P.INP, 'export', job + '.npz'))
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        R8T = [G.Target(private[cm].max(0), H) for cm in F['comps']]
        Lm = list(z['L_k0_cam_nf'])
        for sd, d in P.CKS.items():
            ck, model = L.load_model(os.path.join(d, 'final.pt'), 'cpu')
            thr = json.load(open(os.path.join(d, 'threshold.json')))
            jk, jo = thr['R8']['j_known'], thr['R8']['j_occ']
            table_t = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table'], np.int64), dtype=torch.int32)
            zz, lam = L.encode_maps(model, Lm, table_t, 'cpu')
            hs, ls, _, _ = model.source(zz, lam, 'round')
            a = acc.setdefault(sd, np.zeros(16, np.int64))
            for ci, cm in enumerate(F['comps']):
                hb, lb, _ = L.gabp(hs[cm], ls[cm], L.local_forest(F, cm), float(model.w), 'round')
                lg = L.decode_np(model, hb, lb, 2)
                T = R8T[ci]; reg = H.region; tr = T.tr
                for i in range(len(cm)):
                    p = L.readout(lg[i][0], lg[i][1], jk, jo)
                    po = R.logit_bin(lg[i][1]) >= jo; p0 = lg[i][1] >= 0.0
                    pr = (p == 1) & T.known & reg
                    pr_or = po & T.known & reg                       # known head replaced by the target-known mask
                    pr_or0 = p0 & T.known & reg                      # ... and occupancy threshold at logit 0
                    pr_t0 = (L.readout(lg[i][0], lg[i][1], jk, R.logit_bin(np.array(0.0))[()]) == 1) & T.known & reg
                    a += np.array([tr.sum(), (tr & (p == -1)).sum(), (tr & (p == 0)).sum(), (tr & pr).sum(), (pr & ~tr).sum(),
                                   (tr & pr_or).sum(), (pr_or & ~tr).sum(), (tr & pr_or0).sum(), (pr_or0 & ~tr).sum(),
                                   (tr & pr_t0).sum(), (pr_t0 & ~tr).sum(), (T.free & (p == 0)).sum(), T.free.sum(),
                                   (T.free & (L.readout(lg[i][0], lg[i][1], jk, R.logit_bin(np.array(0.0))[()]) == 0)).sum(), 0, 0], np.int64)
        print(job, f'{time.time() - t0:.0f}s', flush=True)
    out = {}
    for sd, a in acc.items():
        f1 = lambda tp, fp: 2 * tp / max(2 * tp + fp + (a[0] - tp), 1)
        fn = a[0] - a[3]
        out[sd] = dict(target_furn=int(a[0]), fn=int(fn), fn_pred_unknown_frac=float(a[1] / fn), fn_pred_free_frac=float(a[2] / fn),
                       pooled_f1_formal=f1(a[3], a[4]), pooled_f1_known_oracle=f1(a[5], a[6]), pooled_f1_known_oracle_t0=f1(a[7], a[8]),
                       pooled_f1_t0=f1(a[9], a[10]), free_recall_formal=float(a[11] / a[12]), free_recall_t0=float(a[13] / a[12]),
                       recall_formal=float(a[3] / a[0]), recall_t0=float(a[9] / a[0]))
    out['jobs'] = jobs
    with open(os.path.join(HERE, 'rd_out', 'q2_fnsplit.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main([f'val_r{r}_i{i}_s1' for r in (1, 2, 3) for i in (0, 2, 4, 6)])
