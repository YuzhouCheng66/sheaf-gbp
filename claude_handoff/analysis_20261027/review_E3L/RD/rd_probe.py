"""R-D probe (diagnosis only, not a verdict batch). VAL cam_nf noise seed 1, frozen E3-L checkpoints (local copies), frozen E3-R NET as reference.
Imports g4_e3l / g4_e3r read-only. Arms on the same encoder outputs (z, a):
  formal      : = the judged path (source potentials and every message rounded, step 1)
  float_qsup  : no rounding of values or messages; support = the formal support (Lam_q > 0)  -> isolates value quantisation
  float_full  : no rounding at all; every Lam > 0 cell kept (also the 'lost' cells with Lam < 0.5)
  restore_q   : Lam' = max(softplus(a_raw), 1) on every touched block, then the formal rounded path (support forced to all touched blocks)
  restore_f   : the same Lam', float path
  k0_formal   : single agent, own rounded potential, no messages (K0 with the R8 thresholds = summary 'report_e3l_K0_shared')
  NET (E3-R)  : R8 direct sums + NET with its frozen threshold; K0 with its own K0 threshold j_sep
Writes rd_out/q1/<job>.json (rows) and <job>.npz (occupancy-logit histograms for pooled threshold curves).
Usage: python -B rd_probe.py --jobs val_r1_i0_s1,...
"""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.abspath(os.path.join(HERE, '..', '..', 'review_E3R', 'inputs'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
from torch.nn import functional as Fnn  # noqa: E402
import g4_e3l as L  # noqa: E402
R = L.R; G = L.G
UNK = L.UNK
CKS = {'342101': os.path.join(INP, 'e3l_s342101'), '342202': os.path.join(INP, 'e3l_s342202')}
NETS = {'342101': os.path.join(INP, 's342101'), '342202': os.path.join(INP, 's342202')}
OUT = os.path.join(HERE, 'rd_out', 'q1')


def upblk(t):
    return np.repeat(np.repeat(np.repeat(t, 4, 0), 4, 1), 2, 2)


def err_decomp(p, T, H, lb_vox=None):
    """furniture error decomposition of one prediction: FN near (inside dil6 of predicted furniture) vs far; object-level recall;
    FN in latent blocks the agent's belief has no information on (Lam_b = 0)"""
    pr = (p == 1) & T.known & H.region; tr = T.tr
    fn = tr & ~pr; fp = pr & ~tr
    near = fn & G.dil6(pr)
    far = fn & ~near
    lab, nobj = ndimage.label(tr)
    out = dict(tp=int((pr & tr).sum()), fn=int(fn.sum()), fp=int(fp.sum()), fn_near=int(near.sum()), fn_far=int(far.sum()),
               fp_near=int((fp & G.dil6(tr)).sum()), n_obj=int(nobj))
    if nobj:
        tot = np.bincount(lab[tr], minlength=nobj + 1)[1:]
        hit = np.bincount(lab[tr & pr], minlength=nobj + 1)[1:]
        rec = hit / np.maximum(tot, 1)
        out.update(obj_missed=int((rec < 0.5).sum()), obj_missed_vox=int(tot[rec < 0.5].sum()), obj_vox=int(tot.sum()),
                   fn_in_missed_obj=int((fn & np.isin(lab, 1 + np.flatnonzero(rec < 0.5))).sum()))
    if far.any():
        fl, nf = ndimage.label(far)
        sz = np.bincount(fl[far])[1:]
        out.update(far_cc_n=int(nf), far_cc_vox_ge32=int(sz[sz >= 32].sum()), far_cc_median=float(np.median(sz)))
    if lb_vox is not None:
        z = lb_vox <= 0
        out.update(tr_nobelief=int((tr & z).sum()), fn_nobelief=int((fn & z).sum()), fn_far_nobelief=int((far & z).sum()),
                   fp_nobelief=int((fp & z).sum()))
    return out


class Hist:
    def __init__(self):
        self.A = np.zeros(R.NBIN, np.int64); self.B = np.zeros(R.NBIN, np.int64); self.C = np.zeros(R.NBIN, np.int64)
        self.tr = 0; self.fr = 0

    def add(self, lg, jk, T, H):
        pk = R.logit_bin(lg[0]) >= jk; ob = R.logit_bin(lg[1])
        self.A += np.bincount(ob[pk & T.known & H.region & T.tr], minlength=R.NBIN)
        self.B += np.bincount(ob[pk & T.known & H.region & ~T.tr], minlength=R.NBIN)
        self.C += np.bincount(ob[pk & T.free], minlength=R.NBIN)
        self.tr += int(T.tr.sum()); self.fr += int(T.free.sum())


@torch.no_grad()
def encode_raw(model, maps, table_t):
    zs, araw, sups = [], [], []
    for i in range(0, len(maps), 4):
        Lm = torch.from_numpy(np.ascontiguousarray(np.stack(maps[i:i + 4])))
        e, kn = L.evidence_t(Lm, table_t)
        f = R.features_t(e, kn.to(torch.int32))
        z, a = model.enc(f)
        sup = Fnn.max_pool3d(kn.to(f.dtype).unsqueeze(1), L.BLOCK, L.BLOCK)[:, 0] > 0
        zs.append(z); araw.append(a); sups.append(sup)
    return torch.cat(zs), torch.cat(araw), torch.cat(sups)


@torch.no_grad()
def run_job(job):
    t0 = time.time()
    z = np.load(os.path.join(INP, 'export', job + '.npz'))
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    Lm = list(z['L_k0_cam_nf'])
    rec = dict(job=job, n=n, comps=F['comps'], arms={})
    hists = {}
    for sd, d in CKS.items():
        ck, model = L.load_model(os.path.join(d, 'final.pt'), 'cpu')
        thr = json.load(open(os.path.join(d, 'threshold.json')))
        jk, jo = thr['R8']['j_known'], thr['R8']['j_occ']
        table = np.asarray(ck['recipe']['evidence']['table'], np.int64); table_t = torch.as_tensor(table, dtype=torch.int32)
        w = float(model.w)
        zz, araw, sup = encode_raw(model, Lm, table_t)
        z2, lam = L.encode_maps(model, Lm, table_t, 'cpu')
        assert torch.equal(z2, zz)
        lam_chk = Fnn.softplus(araw.clamp(min=L.A_FLOOR)) * sup.float()
        assert torch.allclose(lam_chk, lam), 'manual encode != model.encode'
        st = dict(touched=int(sup.sum()), formal_support=int((torch.round(lam) > 0).sum()),
                  a_raw_touched_q=np.quantile(araw[sup].numpy(), [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                  frac_touched_below_floor=float((araw[sup] < L.A_FLOOR).float().mean()),
                  frac_touched_lam_lt_half=float((lam[sup] < 0.5).float().mean()),
                  lam_supported_q=np.quantile(lam[torch.round(lam) > 0].numpy(), [0.1, 0.5, 0.9]).tolist() if (torch.round(lam) > 0).any() else None,
                  z_absmax_q=np.quantile(zz.abs().amax(1)[sup].numpy(), [0.1, 0.5, 0.9]).tolist())
        lam_r = torch.maximum(Fnn.softplus(araw), torch.ones_like(araw)) * sup.float()
        q_sup = (torch.round(lam) > 0).float()
        arms = {
            'formal': ('round', lam),
            'float_qsup': (None, lam * q_sup),
            'float_full': (None, lam),
            'restore_q': ('round', lam_r),
            'restore_f': (None, lam_r),
        }
        for an, (qm, lm) in arms.items():
            if qm == 'round':
                hs, ls, _, _ = model.source(zz, lm, 'round')
            else:
                hs, ls = lm.unsqueeze(1) * zz, lm
            rows = []; hist = Hist(); by = 0.0
            for ci, cm in enumerate(F['comps']):
                Fl = L.local_forest(F, cm)
                hb, lb, msgs = L.gabp(hs[cm], ls[cm], Fl, w, qm)
                if qm == 'round':
                    for e in Fl['directed']:
                        by += L.wire_bytes(*L.msg_ints(*msgs[e]))
                lg = L.decode_np(model, hb, lb, 2)
                kc = sum((Lm[a] != UNK).astype(np.int32) for a in cm)
                for i, a in enumerate(cm):
                    p = L.readout(lg[i][0], lg[i][1], jk, jo)
                    row = R.scored(p, R8T[ci], H, kc)
                    if an in ('formal', 'restore_q', 'float_full'):
                        row['err'] = err_decomp(p, R8T[ci], H, upblk(lb[i].numpy()))
                    row['known_frac'] = float((p >= 0).mean())
                    rows.append(row); hist.add(lg[i], jk, R8T[ci], H)
            rec['arms'][f'{an}_{sd}'] = dict(rows=rows, bytes=(by if qm == 'round' else None))
            hists[f'{an}_{sd}'] = hist
        # K0 formal (own rounded potential, no messages), R8 thresholds (= summary report_e3l_K0_shared) and own K0 thresholds
        hs, ls, _, _ = model.source(zz, lam, 'round')
        lgk = L.decode_np(model, hs, ls, 2)
        rows = []; rows_sep = []; hist = Hist()
        for a in range(n):
            ki = (Lm[a] != UNK).astype(np.int32)
            p = L.readout(lgk[a][0], lgk[a][1], jk, jo)
            row = R.scored(p, K0T[a], H, ki); row['err'] = err_decomp(p, K0T[a], H, upblk(ls[a].numpy())); rows.append(row)
            rows_sep.append(R.scored(L.readout(lgk[a][0], lgk[a][1], thr['K0']['j_known'], thr['K0']['j_occ']), K0T[a], H, ki))
            hist.add(lgk[a], jk, K0T[a], H)
        rec['arms'][f'k0_formal_{sd}'] = dict(rows=rows); rec['arms'][f'k0_sep_{sd}'] = dict(rows=rows_sep); hists[f'k0_formal_{sd}'] = hist
        rec['arms'][f'formal_{sd}']['stats'] = st
        # NET reference (E3-R frozen)
        nck, net = R.load_ckpt(os.path.join(NETS[sd], 'final.pt'), 'cpu')
        nthr = json.load(open(os.path.join(NETS[sd], 'threshold.json')))
        ev = R.Evidence.from_recipe(nck['recipe']['evidence']); j = int(nthr['j']); jsep = int(nthr['k0']['j_sep'])
        assert np.array_equal(ev.table, table), 'E3-L / E3-R evidence tables differ'
        rows = []
        for ci, cm in enumerate(F['comps']):
            S, kc = R.direct_sums(Lm, cm, ev.table)
            lg = R.net_logits(net, [S], [kc], 'cpu', 1)[0]
            p = R.net_readout(lg, kc, j)
            row = R.scored(p, R8T[ci], H, kc); row['err'] = err_decomp(p, R8T[ci], H)
            rows += [row] * len(cm)                                   # component-expanded (= one row per member)
        own = [R.stalk(Lm[a], ev.table) for a in range(n)]
        lk = R.net_logits(net, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', 2)
        rk = []
        for a in range(n):
            p = R.net_readout(lk[a], own[a][1].astype(np.int32), jsep)
            row = R.scored(p, K0T[a], H, own[a][1].astype(np.int32)); row['err'] = err_decomp(p, K0T[a], H); rk.append(row)
        rec['arms'][f'NET_{sd}'] = dict(rows=rows); rec['arms'][f'NET_K0_{sd}'] = dict(rows=rk)
    rec['seconds'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, job + '.npz'), **{f'{k}_{x}': getattr(h, x) for k, h in hists.items() for x in ('A', 'B', 'C')},
                        **{f'{k}_tot': np.array([h.tr, h.fr]) for k, h in hists.items()})
    with open(os.path.join(OUT, job + '.json'), 'w') as fh:
        json.dump(rec, fh)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    for job in a.jobs.split(','):
        try:
            r = run_job(job)
            fa = {k: np.mean([x['furn_f1'] for x in v['rows'] if x.get('furn_f1') is not None]) for k, v in r['arms'].items()}
            print(job, f"{r['seconds']:.0f}s", json.dumps({k: round(float(v), 4) for k, v in fa.items()}), flush=True)
        except Exception:
            print(job, 'FAILED', traceback.format_exc(), flush=True)


if __name__ == '__main__':
    main()
