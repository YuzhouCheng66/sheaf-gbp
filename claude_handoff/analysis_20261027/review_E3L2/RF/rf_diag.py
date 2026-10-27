"""rf_diag.py (reviewer R-F, 2026-10-27): diagnostic readout of the E3-L2 latent (and E3-L for reference). NOT a verdict.
Imports the frozen g4_e3l2 / g4_e3l / g4_e3r / g4_noise_panels read-only from ~/claude-sheaf-gbp/work/scripts; modifies nothing.
Phase A: TRAIN holdout (19 houses x cam_nf 200-202), R8 (every agent decodes its own GaBP belief vs the component target, pooled).
  Per (checkpoint, K, known-variant): 2-D histograms (coarse known-threshold index x occupancy-logit bin) of free / furniture / other-known-region
  voxels + FPR histograms -> threshold pairs chosen ONLY here.
Phase B: VAL (24 houses x noise 1, 2, cam_nf): the same histograms (report) + per-agent rows with the TRAIN thresholds (row means as E3-R summary).
Known variants (inference only, no retraining): head = known-logit >= t; fp_lit = head OR (footprint & Lam_b == 0);
  fp_and = footprint AND head; fp_and_lit = footprint AND (head OR Lam_b == 0); kmask = voxel k > 0 of the component (= the E3-R control known mask);
  oracle = target known (diagnostic ceiling of the occupancy head only).
Footprint = union over the component of every agent's touched 4x4x2 blocks (= Lam > 0 support before top-K).
Bytes: O7 wire bytes of the GaBP messages (L.wire_bytes); footprint = L.mask_bytes of the sender-side union per directed edge; kmask = the same at voxel level.
"""
import os
import sys
sys.dont_write_bytecode = True
import json, math, time, glob, argparse  # noqa
import multiprocessing as mp

WORK = os.path.expanduser('~/claude-sheaf-gbp/work')
DATA = os.path.expanduser('~/claude-sheaf-gbp/data')
sys.path.insert(0, os.path.join(WORK, 'scripts'))
import numpy as np  # noqa
import g4_e3l2 as L2  # noqa
L = L2.L; R = L2.R; G = L2.G
import torch  # noqa

NBIN = R.NBIN; UNK = R.UNK
J = None                                                          # known-threshold grid in logit bins (-10 .. +14, step 0.08) + official bins; set in main
VARS2 = ('head', 'fp_lit', 'fp_and', 'fp_and_lit')
VARS1 = ('kmask', 'oracle', 'oracle_sent')            # oracle_sent: universe restricted to Lam_b > 0 cells (decode quality where something was sent)
FREE_LEVELS = (0.90, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997)


def jdump(o, p):
    with open(p + '.tmp', 'w') as fh:
        json.dump(o, fh, default=float)
    os.replace(p + '.tmp', p)


# ---------------------------------------------------------------- CPU byte workers (numpy only)
def _bytes_msgs(job):
    tag, msgs = job
    return tag, float(sum(L.wire_bytes(h.astype(np.int64), l.astype(np.int64)) for h, l in msgs))


def _bytes_masks(job):
    tag, masks, shape = job
    return tag, float(sum(L.mask_bytes(np.unpackbits(m)[:int(np.prod(shape))].reshape(shape).astype(bool)) for m in masks))


# ---------------------------------------------------------------- configurations
def configs():
    out = []
    for sd in (342101, 342202):
        for nm in ('final', 'upd6510'):
            run = f'{WORK}/runs/e3l2/s{sd}'
            thr = json.load(open(f'{run}/threshold{"" if nm == "final" else "_" + nm}.json'))
            ks = sorted({256, 512, 1024, 2048, int(thr['k_star'])}) if nm == 'final' else sorted({1024, int(thr['k_star'])})
            for K in ks:
                out.append(dict(name=f'E3L2_{nm}_{sd}_K{K}', arm='E3L2', ckpt=f'{run}/checkpoints/{nm}.pt', seed=sd, K=K, k_star=int(thr['k_star']),
                                is_kstar=(K == int(thr['k_star'])), official=(thr['R8'].get(str(K)))))
        thr = json.load(open(f'{WORK}/runs/e3l/s{sd}/threshold.json'))
        out.append(dict(name=f'E3L_final_{sd}', arm='E3L', ckpt=f'{WORK}/runs/e3l/s{sd}/checkpoints/final.pt', seed=sd, K=None, k_star=None, is_kstar=True,
                        official=dict(j_known=thr['R8']['j_known'], j_occ=thr['R8']['j_occ'], t_known=thr['R8']['t_known'], t_occ=thr['R8']['t_occ'])))
    return out


def load(cfg, dev):
    if cfg['arm'] == 'E3L2':
        ck, m = L2.load_model(cfg['ckpt'], dev)
    else:
        ck, m = L.load_model(cfg['ckpt'], dev)
    tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32, device=dev)
    return m, tb, float(m.w.detach())


# ---------------------------------------------------------------- accumulators
class Acc:
    def __init__(self, dev):
        self.dev = dev; self.GJ = len(J); self.d = {}

    def get(self, key, kind):
        if key not in self.d:
            if kind == 2:
                self.d[key] = dict(h=torch.zeros(3, (self.GJ + 1) * NBIN, dtype=torch.int64, device=self.dev),
                                   u=torch.zeros(self.GJ + 1, dtype=torch.int64, device=self.dev),
                                   fc=torch.zeros(self.GJ + 1, dtype=torch.int64, device=self.dev), tot=torch.zeros(3, dtype=torch.int64, device=self.dev))
            elif kind == 1:
                self.d[key] = dict(h=torch.zeros(3, NBIN, dtype=torch.int64, device=self.dev), tot=torch.zeros(5, dtype=torch.int64, device=self.dev))
            else:
                self.d[key] = dict(kb=torch.zeros(6, NBIN, dtype=torch.int64, device=self.dev), ob=torch.zeros(4, NBIN, dtype=torch.int64, device=self.dev))
        return self.d[key]


def bincount(x, n):
    return torch.bincount(x, minlength=n)[:n]


def add_agent(acc, cname, kb, ob, kc, M, lb0, fp, kmask):
    """kb, ob: logit bins (flat int64); kc: coarse known index; M: target masks (flat bool); lb0 / fp / kmask: flat bool"""
    GJ = len(J); n2 = (GJ + 1) * NBIN
    for v in VARS2:
        if v == 'head':
            kv = kc
        elif v == 'fp_lit':
            kv = torch.where(fp & lb0, torch.full_like(kc, GJ), kc)
        elif v == 'fp_and':
            kv = torch.where(fp, kc, torch.zeros_like(kc))
        else:
            kv = torch.where(fp, torch.where(lb0, torch.full_like(kc, GJ), kc), torch.zeros_like(kc))
        a = acc.get((cname, v), 2)
        idx = kv * NBIN + ob
        a['h'][0] += bincount(idx[M['free']], n2); a['h'][1] += bincount(idx[M['tr']], n2); a['h'][2] += bincount(idx[M['ntr']], n2)
        a['u'] += bincount(kv[M['unk']], GJ + 1); a['fc'] += bincount(kv[M['free']], GJ + 1)
        a['tot'] += torch.stack([M['free'].sum(), M['tr'].sum(), M['unk'].sum()])
    for v in VARS1:
        U = ~lb0 if v == 'oracle_sent' else torch.ones_like(lb0)
        kn = kmask if v == 'kmask' else (M['known'] & U)
        fr, tr, nt, un = M['free'] & U, M['tr'] & U, M['ntr'] & U, M['unk'] & U
        a = acc.get((cname, v), 1)
        a['h'][0] += bincount(ob[kn & fr], NBIN); a['h'][1] += bincount(ob[kn & tr], NBIN); a['h'][2] += bincount(ob[kn & nt], NBIN)
        a['tot'] += torch.stack([fr.sum(), tr.sum(), un.sum(), (kn & un).sum(), (kn & fr).sum()])
    # task 3: Lam_b == 0 cells
    a = acc.get((cname, 'lb0'), 3)
    kn_ = M['known']; unk = M['unk']; lp = ~lb0
    groups = [lb0 & fp & kn_, lb0 & fp & unk, lb0 & ~fp & kn_, lb0 & ~fp & unk, lp & kn_, lp & unk]
    for i, g in enumerate(groups):
        a['kb'][i] += bincount(kb[g], NBIN)
    og = [lb0 & fp & M['free'], lb0 & fp & M['tr'], lp & M['free'], lp & M['tr']]
    for i, g in enumerate(og):
        a['ob'][i] += bincount(ob[g], NBIN)


# ---------------------------------------------------------------- selection on the histograms
def rev_cum(x, dim):
    return x.flip(dim).cumsum(dim).flip(dim)


def surfaces2(a):
    """-> F1 [GJ, NBIN], FREE [GJ, NBIN], FPR [GJ], COV [GJ] (known at g iff coarse index >= g + 1)"""
    GJ = len(J)
    H = a['h'].view(3, GJ + 1, NBIN).double()
    K = rev_cum(H, 1)[:, 1:]                                       # [3, GJ, NBIN]
    ft, tt, ut = [float(x) for x in a['tot'].cpu()]
    TP = rev_cum(K[1], 1); FP = rev_cum(K[2], 1); FN = tt - TP
    F1 = 2 * TP / torch.clamp(2 * TP + FP + FN, min=1)
    FREE = (K[0].cumsum(1) - K[0]) / max(ft, 1)
    FPR = rev_cum(a['u'].double(), 0)[1:] / max(ut, 1)
    COV = rev_cum(a['fc'].double(), 0)[1:] / max(ft, 1)
    return F1, FREE, FPR, COV, dict(free_tot=ft, tr_tot=tt, unk_tot=ut)


def surfaces1(a):
    H = a['h'].double(); tot = [float(x) for x in a['tot'].cpu()]
    TP = rev_cum(H[1], 0); FP = rev_cum(H[2], 0); FN = tot[1] - TP
    F1 = 2 * TP / torch.clamp(2 * TP + FP + FN, min=1)
    FREE = (H[0].cumsum(0) - H[0]) / max(tot[0], 1)
    return F1, FREE, tot[3] / max(tot[2], 1), tot[4] / max(tot[0], 1), dict(free_tot=tot[0], tr_tot=tot[1], unk_tot=tot[2])


def rules(ctrl):
    return dict(A_fpr30_maxF1=(ctrl['fpr'] + 0.003, None), B_fpr30_free50=(ctrl['fpr'] + 0.003, ctrl['free'] - 0.005),
                C_fpr15_free25=(ctrl['fpr'] + 0.0015, ctrl['free'] - 0.0025))


def pick2(F1, FREE, FPR, fpr_max, free_min):
    ok = (FPR <= fpr_max + 1e-15)[:, None].expand_as(F1)
    if free_min is not None:
        ok = ok & (FREE + 1e-15 >= free_min)
    sc = torch.where(ok, F1, torch.full_like(F1, -1.0))
    i = int(torch.argmax(sc)); g, jo = divmod(i, NBIN)
    return (g, jo, float(sc.view(-1)[i]) >= 0)


def summarise2(a, ctrl, official_jk=None):
    F1, FREE, FPR, COV, tot = surfaces2(a)
    out = dict(totals=tot, rules={})
    for rn, (fm, frm) in rules(ctrl).items():
        g, jo, feas = pick2(F1, FREE, FPR, fm, frm)
        out['rules'][rn] = dict(feasible=feas, g=g, j_known=int(J[g]), t_known=R.bin_edge(int(J[g])), j_occ=jo, t_occ=R.bin_edge(jo), f1=float(F1[g, jo]),
                                free=float(FREE[g, jo]), fpr=float(FPR[g]), coverage=float(COV[g]), fpr_max=fm, free_min=frm)
    # frontier: max F1 with FPR <= ctrl + 0.3 pp and free >= level
    fm = ctrl['fpr'] + 0.003
    out['frontier_fpr30'] = {str(lv): (lambda r: dict(f1=float(F1[r[0], r[1]]) if r[2] else None, t_known=R.bin_edge(int(J[r[0]])), t_occ=R.bin_edge(r[1])))(pick2(F1, FREE, FPR, fm, lv))
                             for lv in FREE_LEVELS}
    out['frontier_nofpr'] = {str(lv): (lambda r: dict(f1=float(F1[r[0], r[1]]) if r[2] else None, fpr=float(FPR[r[0]]), t_known=R.bin_edge(int(J[r[0]])),
                                                       t_occ=R.bin_edge(r[1])))(pick2(F1, FREE, FPR, 1.0, lv)) for lv in FREE_LEVELS}
    gm = len(J) - 1
    out['strictest_known'] = dict(t_known=R.bin_edge(int(J[gm])), fpr=float(FPR[gm]), coverage=float(COV[gm]))
    out['frontier_fpr15'] = {str(lv): (lambda r: dict(f1=float(F1[r[0], r[1]]) if r[2] else None))(pick2(F1, FREE, FPR, ctrl['fpr'] + 0.0015, lv)) for lv in FREE_LEVELS}
    # known-head curve: FPR vs free coverage
    sel = list(range(0, len(J), 5))
    out['known_curve'] = dict(t_known=[R.bin_edge(int(J[g])) for g in sel], fpr=[float(FPR[g]) for g in sel], free_coverage=[float(COV[g]) for g in sel])
    gl = [g for g in range(len(J)) if float(FPR[g]) <= fm + 1e-15]
    out['max_coverage_fpr30'] = float(COV[gl[0]]) if gl else None
    gl15 = [g for g in range(len(J)) if float(FPR[g]) <= ctrl['fpr'] + 0.0015 + 1e-15]
    out['max_coverage_fpr15'] = float(COV[gl15[0]]) if gl15 else None
    # F1-free curve at rule A's known threshold
    gA = out['rules']['A_fpr30_maxF1']['g']
    jos = [j for j in range(0, NBIN, 10) if float(F1[gA, j]) > 1e-3 or j % 200 == 0]
    jos = jos[::max(1, len(jos) // 80)]
    out['occ_curve_at_ruleA'] = dict(t_occ=[R.bin_edge(j) for j in jos], f1=[float(F1[gA, j]) for j in jos], free=[float(FREE[gA, j]) for j in jos])
    if official_jk is not None:
        go = int(np.searchsorted(J, official_jk, side='left'))
        if go < len(J) and J[go] == official_jk:
            jo = int(torch.argmax(F1[go]))
            out['at_official_t_known'] = dict(t_known=R.bin_edge(official_jk), fpr=float(FPR[go]), coverage=float(COV[go]), best_f1_any_free=float(F1[go, jo]),
                                              free_at_best=float(FREE[go, jo]))
    return out


def summarise1(a, ctrl):
    F1, FREE, fpr, cov, tot = surfaces1(a)
    out = dict(totals=tot, fpr=fpr, coverage=cov, rules={})
    for rn, (fm, frm) in rules(ctrl).items():
        ok = torch.ones_like(F1, dtype=torch.bool) if fpr <= fm + 1e-15 else torch.zeros_like(F1, dtype=torch.bool)
        if frm is not None:
            ok = ok & (FREE + 1e-15 >= frm)
        sc = torch.where(ok, F1, torch.full_like(F1, -1.0)); jo = int(torch.argmax(sc))
        out['rules'][rn] = dict(feasible=bool(float(sc[jo]) >= 0), j_occ=jo, t_occ=R.bin_edge(jo), f1=float(F1[jo]), free=float(FREE[jo]), fpr=fpr, fpr_max=fm, free_min=frm)
    jbest = int(torch.argmax(F1))
    out['best_any'] = dict(f1=float(F1[jbest]), free=float(FREE[jbest]), t_occ=R.bin_edge(jbest))
    out['frontier'] = {str(lv): (float(torch.where(FREE + 1e-15 >= lv, F1, torch.full_like(F1, -1.0)).max())) for lv in FREE_LEVELS}
    jos = [j for j in range(0, NBIN, 10) if float(F1[j]) > 1e-3]
    jos = jos[::max(1, len(jos) // 80)]
    out['occ_curve'] = dict(t_occ=[R.bin_edge(j) for j in jos], f1=[float(F1[j]) for j in jos], free=[float(FREE[j]) for j in jos])
    return out


def auc(pos, neg):
    pos = pos.double(); neg = neg.double(); P = pos.sum(); N = neg.sum()
    if P == 0 or N == 0:
        return None
    cn = neg.cumsum(0) - neg                                       # negatives strictly below each bin
    return float(((pos * (cn + 0.5 * neg)).sum()) / (P * N))


def summarise_lb0(a, jk_rules, jo_rule, official_jk):
    kb = a['kb']; ob = a['ob']; n = kb.sum(1).double()
    names = ['lb0_fp_known', 'lb0_fp_unk', 'lb0_nofp_known', 'lb0_nofp_unk', 'lbpos_known', 'lbpos_unk']
    out = dict(counts={nm: int(n[i]) for i, nm in enumerate(names)})
    kn_tot = float(n[0] + n[2] + n[4])
    out['share_target_known_in_lb0'] = float((n[0] + n[2]) / kn_tot) if kn_tot else None
    out['share_target_known_in_lb0_fp'] = float(n[0] / kn_tot) if kn_tot else None
    out['share_target_known_in_lb0_nofp'] = float(n[2] / kn_tot) if kn_tot else None
    ut = float(n[1] + n[3] + n[5])
    out['share_target_unknown_in_fp_lb0'] = float(n[1] / ut) if ut else None
    out['auc_known_vs_unk_in_lb0_fp'] = auc(kb[0], kb[1])
    out['auc_known_vs_unk_in_lb0'] = auc(kb[0] + kb[2], kb[1] + kb[3])
    out['auc_lb0fpknown_vs_lb0nofpunk'] = auc(kb[0], kb[3])
    out['auc_known_vs_unk_in_lbpos'] = auc(kb[4], kb[5])
    thr = dict(jk_rules); thr['official'] = official_jk
    out['pred_known_rate'] = {}
    for tn, jk in thr.items():
        if jk is None:
            continue
        out['pred_known_rate'][tn] = {nm: (float(kb[i, jk:].sum() / n[i]) if n[i] > 0 else None) for i, nm in enumerate(names)}
    on = ob.sum(1).double()
    out['occ_counts'] = dict(lb0_fp_free=int(on[0]), lb0_fp_tr=int(on[1]), lbpos_free=int(on[2]), lbpos_tr=int(on[3]))
    if jo_rule is not None:
        out['occ_at_ruleA'] = dict(lb0_fp_free_pred_free=(float(ob[0, :jo_rule].sum() / on[0]) if on[0] > 0 else None),
                                   lb0_fp_tr_pred_occ=(float(ob[1, jo_rule:].sum() / on[1]) if on[1] > 0 else None),
                                   lbpos_free_pred_free=(float(ob[2, :jo_rule].sum() / on[2]) if on[2] > 0 else None),
                                   lbpos_tr_pred_occ=(float(ob[3, jo_rule:].sum() / on[3]) if on[3] > 0 else None))
    out['kb_hist_coarse'] = {nm: kb[i].view(-1, 20).sum(1).tolist() for i, nm in enumerate(names)}
    return out


# ---------------------------------------------------------------- one house / draw
def gpu_masks(T, H, dev):
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev).reshape(-1)  # noqa
    known = t(T.known); tr = t(T.tr); region = t(H.region)
    return dict(known=known, unk=~known, free=t(T.free), tr=tr, ntr=known & region & ~tr, surf=t(T.surf), notint=~t(T.interior), region=region)


@torch.no_grad()
def process_item(item, models, cfgs, acc, pool, pend, dev, row_specs=None, rows_out=None):
    """item: dict(tag, maps [n, X, Y, Z] int8, comps, Fls (local forests), targets (per comp: G.Target), H)"""
    maps = item['maps']; n = maps.shape[0]
    Ms = [gpu_masks(T, item['H'], dev) for T in item['targets']]
    kms = [torch.from_numpy((maps[cm] != UNK).any(0)).to(dev).reshape(-1) for cm in item['comps']]
    Jt = torch.as_tensor(J, device=dev)
    # footprint / kmask bytes (independent of the model)
    enc_cache = {}
    for ci, cfg in enumerate(cfgs):
        m, tb, w = models[cfg['ckpt']]
        if cfg['ckpt'] not in enc_cache:
            enc_cache = {cfg['ckpt']: L.encode_maps(m, list(maps), tb, dev)}
        z, lam = enc_cache[cfg['ckpt']]
        if cfg['arm'] == 'E3L2':
            hq, lq, _ = L2.source_topk(z, lam, cfg['K'], 'round')
        else:
            hq, lq = m.source(z, lam, 'round')[:2]
        msgs_all = []
        for k, cm in enumerate(item['comps']):
            Fl = item['Fls'][k]
            hb, lb, msgs = L.gabp(hq[cm], lq[cm], Fl, w, 'round')
            for e in Fl['directed']:
                hi, li = L.msg_ints(*msgs[e]); msgs_all.append((hi.astype(np.int16), li.astype(np.int16)))
            fpc = (lam[cm] > 0).any(0)
            fpv = L.upsample(fpc.to(torch.uint8)).reshape(-1).bool()
            lg = []
            for i0 in range(0, len(cm), 4):
                lg.append(m.decode(hb[i0:i0 + 4], lb[i0:i0 + 4]).float())
            lg = torch.cat(lg)
            M = Ms[k]
            for i in range(len(cm)):
                kl = lg[i, 0].reshape(-1).double(); ol = lg[i, 1].reshape(-1).double()
                kb = torch.clamp(torch.floor((kl - R.LOGIT_LO) / R.LOGIT_STEP), 0, NBIN - 1).long()
                ob = torch.clamp(torch.floor((ol - R.LOGIT_LO) / R.LOGIT_STEP), 0, NBIN - 1).long()
                kc = torch.searchsorted(Jt, kb, right=True)
                lb0 = L.upsample((lb[i] == 0).to(torch.uint8)).reshape(-1).bool()
                add_agent(acc, cfg['name'], kb, ob, kc, M, lb0, fpv, kms[k])
                if row_specs is not None and int(M['known'].sum()) > 0:
                    rs = row_specs.get(cfg['name'], [])
                    vals = []
                    for (var, rule, jk, jo) in rs:
                        if var == 'oracle_sent':
                            continue
                        if var == 'kmask':
                            pk = kms[k]
                        elif var == 'oracle':
                            pk = M['known']
                        else:
                            hk = kb >= jk
                            if var == 'head':
                                pk = hk
                            elif var == 'fp_lit':
                                pk = hk | (fpv & lb0)
                            elif var == 'fp_and':
                                pk = fpv & hk
                            else:
                                pk = fpv & (hk | lb0)
                        oo = ob >= jo
                        pocc = pk & oo
                        pr = pocc & M['known'] & M['region']
                        ps = pocc & M['known'] & M['notint']
                        vals.append(torch.stack([(pr & M['tr']).sum(), (pr & ~M['tr']).sum(), M['tr'].sum(), (pk & ~oo & M['free']).sum(), M['free'].sum(),
                                                 (pk & M['unk']).sum(), M['unk'].sum(), (ps & M['surf']).sum(), (ps & ~M['surf']).sum(), M['surf'].sum()]))
                    if vals:
                        V = torch.stack(vals).cpu().numpy()
                        for (var, rule, jk, jo), v in zip([x for x in rs if x[0] != 'oracle_sent'], V):
                            rows_out.setdefault((cfg['name'], var, rule), []).append(dict(tag=item['tag'], v=v.tolist()))
        pend.append(pool.apply_async(_bytes_msgs, ((('wire', cfg['name'], item['tag']), msgs_all),)))
    # footprint / kmask bytes per house: sender-side unions on every directed edge
    fpm, kmm = [], []
    for k, cm in enumerate(item['comps']):
        Fl = item['Fls'][k]
        for e in Fl['directed']:
            side = [cm[a] for a in Fl['sides'][e]]
            fpm.append(np.packbits(np.stack([item['fp_agent'][a] for a in side]).any(0)))
            kmm.append(np.packbits((maps[side] != UNK).any(0)))
    pend.append(pool.apply_async(_bytes_masks, ((('fp', None, item['tag']), fpm, (32, 32, 16)),)))
    pend.append(pool.apply_async(_bytes_masks, ((('kmask', None, item['tag']), kmm, tuple(maps.shape[1:])),)))


def fp_agent_cells(maps):
    kn = torch.from_numpy(maps != UNK).float()
    return (torch.nn.functional.max_pool3d(kn.unsqueeze(1), L.BLOCK, L.BLOCK)[:, 0] > 0).numpy()


def train_items(seeds_only=None):
    man = json.load(open(f'{DATA}/cache-e3r-v1/manifest.json'))
    for e in man['houses']:
        if not e['holdout']:
            continue
        Hd = L.LHouse(os.path.join(f'{DATA}/cache-e3r-v1', e['file']), e, man['seeds'])
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        Ts = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]
        Fls = [Hd.local(cm) for cm in Hd.comps]
        for d in range(1, len(Hd.draws)):
            maps = Hd.maps[d]
            yield dict(tag=f"{e['house']}|{Hd.draws[d]}", house=e['house'], draw=Hd.draws[d], maps=maps, comps=Hd.comps, Fls=Fls, targets=Ts, H=Hh,
                       fp_cells={}, fp_agent=fp_agent_cells(maps))


def val_items():
    recs = [json.load(open(p)) for p in sorted(glob.glob(f'{DATA}/cache-e3r-v1-export/records/val_*.json'))]
    for r in recs:
        if r.get('failed') or r.get('skipped'):
            continue
        z = np.load(f"{DATA}/cache-e3r-v1-export/{r['file']}")
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in r['components']]
        maps = z['L_k0_cam_nf']
        Ts = [G.Target(private[cm].max(0), H) for cm in F['comps']]
        Fls = [L.local_forest(F, cm) for cm in F['comps']]
        yield dict(tag=f"r{r['rooms']}_i{r['index']}_s{r['seed']}", seed=r['seed'], maps=maps, comps=[list(c) for c in F['comps']], Fls=Fls, targets=Ts, H=H,
                   fp_cells={}, fp_agent=fp_agent_cells(maps))


def row_summary(rows):
    def f1(tp, fp, fn):
        return 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else None
    R_ = [r['v'] for r in rows]
    furn = [f1(v[0], v[1], v[2] - v[0]) for v in R_]
    free = [v[3] / v[4] if v[4] else None for v in R_]
    fpr = [v[5] / v[6] if v[6] else None for v in R_]
    surf = [f1(v[7], v[8], v[9] - v[7]) for v in R_]
    S = np.sum(R_, 0)
    m = lambda x: float(np.mean([y for y in x if y is not None])) if any(y is not None for y in x) else None  # noqa
    return dict(n_rows=len(R_), furn_f1=m(furn), free_recall=m(free), unknown_known_fpr=m(fpr), surface_f1=m(surf),
                furn_f1_pooled=f1(S[0], S[1], S[2] - S[0]), free_pooled=S[3] / max(S[4], 1), fpr_pooled=S[5] / max(S[6], 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pool = mp.get_context('fork').Pool(args.workers)               # before any CUDA call
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    dev = torch.device('cuda'); gpu = torch.cuda.get_device_name(dev); assert 'RTX 6000 Ada' in gpu, gpu
    assert torch.cuda.device_count() == 1
    cfgs = configs()
    global J
    offs = {int(c['official']['j_known']) for c in cfgs if c.get('official')}
    J = np.array(sorted(set(range(1500, 2701, 4)) | offs), np.int64)
    if args.only:
        cfgs = [c for c in cfgs if any(s in c['name'] for s in args.only.split(','))]
    models = {}
    for c in cfgs:
        if c['ckpt'] not in models:
            models[c['ckpt']] = load(c, dev)
    ctrlh = json.load(open(f'{DATA}/e3l2-ref/ctrl_holdout.json'))
    meta = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpu=gpu, torch=torch.__version__, J=[int(J[0]), int(J[-1]), 4], configs=cfgs,
                script_sha256=G.file_sha256(os.path.abspath(__file__)), e3l2_sha256=G.file_sha256(L2.__file__), e3l_sha256=G.file_sha256(L.__file__),
                w={c['name']: models[c['ckpt']][2] for c in cfgs}, ctrl_holdout_sha256=G.file_sha256(f'{DATA}/e3l2-ref/ctrl_holdout.json'))
    t0 = time.time()
    # ---------------- phase A: TRAIN holdout
    acc = Acc(dev); pend = []; nA = 0
    for it in train_items():
        process_item(it, models, cfgs, acc, pool, pend, dev); nA += 1
        print(f'A {nA} {it["tag"]} {time.time() - t0:.0f}s', flush=True)
        if args.limit and nA >= args.limit:
            break
    A = dict(meta=meta, items=nA, configs={})
    row_specs = {}
    for c in cfgs:
        ctrl = ctrlh['readouts'][str(c['seed'])]['R8']
        off_jk = int(c['official']['j_known']) if c.get('official') else None
        S = dict(control=dict(free=ctrl['free'], fpr=ctrl['fpr'], furn_f1=ctrl['furn_f1']), variants={})
        for v in VARS2:
            S['variants'][v] = summarise2(acc.d[(c['name'], v)], ctrl, off_jk if v == 'head' else None)
        for v in VARS1:
            S['variants'][v] = summarise1(acc.d[(c['name'], v)], ctrl)
        rA = S['variants']['head']['rules']['A_fpr30_maxF1']
        S['lb0'] = summarise_lb0(acc.d[(c['name'], 'lb0')], {rn: r['j_known'] for rn, r in S['variants']['head']['rules'].items()}, rA['j_occ'], off_jk)
        A['configs'][c['name']] = S
        rs = []
        for v in VARS2:
            for rn, r in S['variants'][v]['rules'].items():
                rs.append((v, rn, r['j_known'], r['j_occ']))
        for v in VARS1:
            for rn, r in S['variants'][v]['rules'].items():
                rs.append((v, rn, None, r['j_occ']))
        if c.get('official'):
            rs.append(('head', 'official', int(c['official']['j_known']), int(c['official']['j_occ'])))
        row_specs[c['name']] = rs
    del acc; torch.cuda.empty_cache()
    resA = [p.get() for p in pend]; pend = []
    A['bytes'] = {}
    for (kind, cname, tag), b in resA:
        A['bytes'].setdefault(kind if cname is None else cname, {})[tag] = b
    # byte ratios vs the paired control (per house, draw), median
    A['byte_ratio_median'] = {}
    for c in cfgs:
        cb = ctrlh['bytes'][str(c['seed'])]
        def ratio(d):
            return float(np.median([d[t] / cb[t.split('|')[0]][t.split('|')[1]] for t in d]))
        A['byte_ratio_median'][c['name']] = dict(model=ratio(A['bytes'][c['name']]), fp=ratio(A['bytes']['fp']), kmask=ratio(A['bytes']['kmask']),
                                                 model_plus_fp=ratio({t: A['bytes'][c['name']][t] + A['bytes']['fp'][t] for t in A['bytes'][c['name']]}),
                                                 model_plus_kmask=ratio({t: A['bytes'][c['name']][t] + A['bytes']['kmask'][t] for t in A['bytes'][c['name']]}))
    A['seconds'] = time.time() - t0
    jdump(A, os.path.join(args.out, 'phaseA_train.json'))
    print('phase A done', time.time() - t0, flush=True)
    # ---------------- phase B: VAL
    acc = Acc(dev); pend = []; rows = {}; nB = 0
    for it in val_items():
        process_item(it, models, cfgs, acc, pool, pend, dev, row_specs, rows); nB += 1
        print(f'B {nB} {it["tag"]} {time.time() - t0:.0f}s', flush=True)
        if args.limit and nB >= args.limit:
            break
    Sc = json.load(open(f'{WORK}/runs/e3r/eval/summary.json'))
    B = dict(meta=meta, items=nB, configs={}, rows={}, control={})
    for ns in ('1', '2'):
        for sd in ('342101', '342202'):
            c = Sc['panels']['val']['seeds'][ns]['tiers']['cam_nf'][f'NET_{sd}']
            B['control'][f'{sd}|{ns}'] = dict(furn_f1=c['R8']['furn_f1'], free_recall=c['R8']['free_recall'], unknown_known_fpr=c['R8']['unknown_known_fpr'],
                                              surface_f1=c['R8']['surface_f1'], furn_f1_pooled=c['R8'].get('furn_f1_pooled'),
                                              bytes=c['rnet_relay_median']['sum'] + c['rnet_relay_median']['count'])
    for c in cfgs:
        ctrl = ctrlh['readouts'][str(c['seed'])]['R8']
        S = dict(variants={})
        for v in VARS2:
            S['variants'][v] = summarise2(acc.d[(c['name'], v)], ctrl, None)
        for v in VARS1:
            S['variants'][v] = summarise1(acc.d[(c['name'], v)], ctrl)
        jA = A['configs'][c['name']]['variants']['head']['rules']
        S['lb0'] = summarise_lb0(acc.d[(c['name'], 'lb0')], {rn: r['j_known'] for rn, r in jA.items()}, jA['A_fpr30_maxF1']['j_occ'],
                                 int(c['official']['j_known']) if c.get('official') else None)
        B['configs'][c['name']] = S
    for (cname, var, rule), rr in rows.items():
        for ns in ('1', '2'):
            sub = [r for r in rr if r['tag'].endswith(f'_s{ns}')]
            if sub:
                B['rows'][f'{cname}|{var}|{rule}|{ns}'] = row_summary(sub)
    resB = [p.get() for p in pend]
    B['bytes'] = {}
    for (kind, cname, tag), b in resB:
        B['bytes'].setdefault(kind if cname is None else cname, {})[tag] = b
    B['bytes_median'] = {}
    for k, d in B['bytes'].items():
        for ns in ('1', '2'):
            vals = [b for t, b in d.items() if t.endswith(f'_s{ns}')]
            if vals:
                B['bytes_median'][f'{k}|{ns}'] = float(np.median(vals))
    for c in cfgs:
        for ns in ('1', '2'):
            tags = [t for t in B['bytes'][c['name']] if t.endswith(f'_s{ns}')]
            B['bytes_median'][f"{c['name']}+fp|{ns}"] = float(np.median([B['bytes'][c['name']][t] + B['bytes']['fp'][t] for t in tags]))
            B['bytes_median'][f"{c['name']}+kmask|{ns}"] = float(np.median([B['bytes'][c['name']][t] + B['bytes']['kmask'][t] for t in tags]))
    B['seconds'] = time.time() - t0; B['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    jdump(B, os.path.join(args.out, 'phaseB_val.json'))
    pool.close(); pool.join()
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
