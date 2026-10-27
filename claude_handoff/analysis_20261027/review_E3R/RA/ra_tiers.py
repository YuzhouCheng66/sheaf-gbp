"""R-A probe (report only): lidar / harsh tiers, which the E3-R export does not contain. Maps regenerated with g4_e3r.tier_maps (= E1
calls, common random numbers). Read-outs: B4; B6 with the tier's own E1c table (half A); NET_342101 fed with the cam_nf table it was
trained on (as deployed) and with the tier's own table; R8 on public maps (with inter-agent error) and on agent-frame maps (= the
'_nf' upper bound); K0 with the K0 thresholds.  Usage: python -B ra_tiers.py --jobs val_r1_i0_s1,... [--threads 2]
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.join(HERE, '..', 'inputs')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402
G = E.G


def ev_of(fit, tier):
    tb = fit['halves']['A']['tiers'][tier]
    return E.Evidence(tb['cq'], tb['theta'], tb['l0q'])


def run(job, model, j, jk, fit):
    t0 = time.time()
    rest = job.split('_r', 1)[1]; rooms = int(rest.split('_i')[0]); index = int(rest.split('_i')[1].split('_s')[0]); seed = int(rest.split('_s')[1])
    ctx = E.house_ctx('VALIDATION', rooms, index)
    H = ctx['H']; F = ctx['F']; private = ctx['private']; n = ctx['n']; comps = F['comps']
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in comps]
    K0T = [G.Target(private[a], H) for a in range(n)]
    camnf = ev_of(fit, 'cam_nf'); Lt = E.Evidence.from_spec('L')
    out = dict(job=job, tiers={})
    for tn in ('lidar', 'harsh'):
        L_k0, L_pub, _ = E.tier_maps(ctx, tn, seed, {})
        own = ev_of(fit, tn); T = {}
        for frame, maps in (('pub', L_pub), ('nf', L_k0)):
            rows = {k: [] for k in ('B4', 'B6own', 'NETcamnf', 'NETown')}
            for ci, cm in enumerate(comps):
                SL, kL = E.direct_sums(maps, cm, Lt.table); So, ko = E.direct_sums(maps, cm, own.table); Sc, kc = E.direct_sums(maps, cm, camnf.table)
                lgc = E.net_logits(model, [Sc], [kc], 'cpu', 1)[0]; lgo = E.net_logits(model, [So], [ko], 'cpu', 1)[0]
                P = dict(B4=E.linear_readout(SL, kL, 0), B6own=E.linear_readout(So, ko, own.theta), NETcamnf=E.net_readout(lgc, kc, j), NETown=E.net_readout(lgo, ko, j))
                for k, p in P.items():
                    r = G.score(p, R8T[ci]); rows[k].append([r['furn_f1'], len(cm), r['free_recall']])
            T['R8_' + frame] = rows
        k0 = {k: [] for k in ('B4', 'B6own', 'NETcamnf')}
        for a in range(n):
            e, kk = E.stalk(L_k0[a], camnf.table); ki = kk.astype(np.int32)
            eo, _ = E.stalk(L_k0[a], own.table)
            lg = E.net_logits(model, [e], [ki], 'cpu', 1)[0]
            k0['B4'].append(G.score(G.tern(L_k0[a]), K0T[a])['furn_f1'])
            k0['B6own'].append(G.score(E.linear_readout(eo, ki, own.theta), K0T[a])['furn_f1'])
            k0['NETcamnf'].append(G.score(E.net_readout(lg, ki, jk), K0T[a])['furn_f1'])
        T['K0'] = k0
        out['tiers'][tn] = T
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ck, model = E.load_ckpt(os.path.join(INP, 's342101', 'final.pt'), 'cpu')
    thr = json.load(open(os.path.join(INP, 's342101', 'threshold.json'))); j = int(thr['j']); jk = int(thr['k0']['j_sep'])
    fit = json.load(open(os.path.join(INP, 'e1c_fit.json')))
    assert list(fit['halves']['A']['tiers']['cam_nf']['cq']) == list(ck['recipe']['evidence']['table'])
    od = os.path.join(HERE, 'out_tiers'); os.makedirs(od, exist_ok=True)
    for job in a.jobs.split(','):
        p = os.path.join(od, job + '.json')
        if os.path.exists(p):
            continue
        try:
            r = run(job, model, j, jk, fit)
        except Exception:
            r = dict(job=job, failed=True, tb=traceback.format_exc()[-2000:])
        json.dump(r, open(p, 'w'))
        print(job, r.get('seconds'), r.get('failed'), flush=True)


if __name__ == '__main__':
    main()
