"""R-A probe isl (report only): E4's OWN edge estimates (job file), composed per ISLAND of accepted (corrected) edges, each island
placed at the mean of its members' believed frames (unit prior, no truth); rejected edges carry nothing. Then the oracle-gauge row
as ra_regnet2 (comparable with its noreg / reg / ceiling_islands rows).
Old: R-A probe v2 (report only): NET read-out on top of registration arms, cam tier, ORACLE-GAUGE row: per component the mean residual
SE(2) (mean_transform of K_a A_a about the agents' mean position) is removed exactly (truth used for the unobservable global gauge only).
Arm half: every agent keeps half of its true error (yaw/2 about its centre, translation/2).
Arms (each agent's believed public scans re-integrated under K_a; K_a read from the frozen G4-E4 job file, params -> SE(2)):
  noreg (= export L_pub_cam), reg (E4 classical), ceiling_pairwise, ceiling_islands, oracle_all (E4 truth-built rows),
  oracle_trans (probe_yaw's: true translation removed, yaw kept), truepose (= export L_k0_cam = cam_nf).
Read-outs per arm: B4 (sum L > 0), B6_A (sum cq > theta), NET_342101 (its R8 threshold). Score = G.score on the R8 target.
Imports frozen modules only.  Usage: python -B ra_regnet.py --jobs val_r1_i0_s1,... [--threads 2]
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import argparse, json, math, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
INP = os.path.join(HERE, '..', 'inputs')
E4J = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e4', 'jobs')
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import torch  # noqa: E402
import g4_e3r as E  # noqa: E402
import g4_registration as R  # noqa: E402
G = E.G; W = G.W


def K_of(p):
    return R.se2(math.radians(p[0]), p[1], p[2])


def readouts(maps, comps, R8T, ev, model, j):
    out = {}
    for tag in ('B4', 'B6', 'NET'):
        out[tag] = []
    for ci, cm in enumerate(comps):
        SL, kL = E.direct_sums(maps, cm, E.Evidence.from_spec('L').table)
        S, kc = E.direct_sums(maps, cm, ev.table)
        assert np.array_equal(kL, kc)
        lg = E.net_logits(model, [S], [kc], 'cpu', 1)[0]
        preds = dict(B4=E.linear_readout(SL, kL, 0), B6=E.linear_readout(S, kc, ev.theta), NET=E.net_readout(lg, kc, j))
        for tag, p in preds.items():
            r = G.score(p, R8T[ci])
            out[tag].append([r['furn_f1'], len(cm), r['free_recall'], r['furn_f1_tol1']])
    return out


def run(job, ck):
    t0 = time.time()
    panel, rest = job.split('_r', 1); rooms = int(rest.split('_i')[0]); index = int(rest.split('_i')[1].split('_s')[0]); seed = int(rest.split('_s')[1])
    z = np.load(os.path.join(INP, 'export', job + '.npz'))
    e4 = json.load(open(os.path.join(E4J, job + '.json')))['conds']['cam']
    fx = W.make_fixture('VALIDATION', index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    assert np.array_equal(private, z['private'])
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n); comps = F['comps']
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in comps]
    rays = W.fibonacci_sphere(W.BEAMS)
    c = G.CONDITIONS['cam']
    pubs, MAs = [], []
    for a in range(n):
        tr = [G.trace_pose(occ, cell, traj[jj, a], rays) for jj in range(traj.shape[0])]
        zf = G.frame_draws(seed, rooms, index, a)
        frame = (G.rotz(float(zf[2] * c['inter_yaw'])), np.array([zf[0] * c['inter_t'], zf[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        MAs.append(R.frame_matrix(frame))
        scans = [G.measure(c, *tr[jj], traj[jj, a], G.pose_draws(seed, rooms, index, a, jj, len(rays)), ds, rays)[0] for jj in range(traj.shape[0])]
        pubs.append([G.to_public(s, frame) for s in scans])
    chk = G.integrate_agent(shape, cell, ds, pubs[0], ('V3',))['V3']
    assert np.array_equal(chk, z['L_pub_cam'][0]), 'regenerated public scans do not reproduce the export map'
    model, ev, j = ck
    acc = {}
    for ed in e4['registration']['edges']:
        if ed.get('corrected'):
            M = K_of(ed['est']); acc[(ed['i'], ed['j'])] = M
    nb = {a: [] for a in range(n)}
    for (i, jj), M in acc.items():
        nb[i].append((jj, M)); nb[jj].append((i, R.inv(M)))   # C_child = C_parent . M  (as R.compose); reverse uses M^-1
    C = {}; isl = []
    for a0 in range(n):
        if a0 in C:
            continue
        C[a0] = R.I3.copy(); q = [a0]; mem = [a0]
        while q:
            u = q.pop()
            for v, M in nb[u]:
                if v not in C:
                    C[v] = C[u] @ M; q.append(v); mem.append(v)
        isl.append(sorted(mem))
    K = [None] * n
    for mem in isl:
        piv = np.mean([np.asarray(positions[a], np.float64)[:2] for a in mem], 0)
        Hinv = R.inv(R.mean_transform([C[a] for a in mem], piv))
        for a in mem:
            K[a] = Hinv @ C[a]
    res = dict(job=job, islands=e4['observability']['islands'], n_islands=e4['observability']['n_islands'], acc_islands=isl, arms={}, resid={})
    for arm, KK in (('reg_island', K),):
        Kg = [None] * n; rs = []
        for cm in comps:
            pv = np.mean([np.asarray(positions[a], np.float64)[:2] for a in cm], 0)
            Gc = R.inv(R.mean_transform([KK[a] @ MAs[a] for a in cm], pv))
            for a in cm:
                Kg[a] = Gc @ KK[a]; Ea = Kg[a] @ MAs[a]; cen = np.asarray(positions[a], np.float64)[:2]
                rs.append([math.degrees(R.yaw(Ea)), float(np.linalg.norm(R.apply2(Ea, cen) - cen) * 100)])
        maps = [R._integrate(shape, cell, ds, pubs[a], Kg[a]) for a in range(n)]
        res['arms'][arm] = readouts(maps, comps, R8T, ev, model, j)
        rs = np.asarray(rs); res['resid'][arm] = dict(yaw_rms_deg=float(np.sqrt((rs[:, 0] ** 2).mean())), disp_rms_cm=float(np.sqrt((rs[:, 1] ** 2).mean())))
        del maps
    res['seconds'] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    ck, model = E.load_ckpt(os.path.join(INP, 's342101', 'final.pt'), 'cpu')
    ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
    j = int(json.load(open(os.path.join(INP, 's342101', 'threshold.json')))['j'])
    od = os.path.join(HERE, 'out_regisl'); os.makedirs(od, exist_ok=True)
    for job in a.jobs.split(','):
        p = os.path.join(od, job + '.json')
        if os.path.exists(p):
            continue
        try:
            r = run(job, (model, ev, j))
        except Exception:
            r = dict(job=job, failed=True, tb=traceback.format_exc()[-2000:])
        json.dump(r, open(p, 'w'))
        print(job, r.get('seconds'), r.get('failed'), flush=True)


if __name__ == '__main__':
    main()
