"""Reviewer diagnostic for G4-E4b (NOT part of the batch; imports g4_registration_msg / g4_registration / g4_noise_panels, modifies none).
One house, one noise seed, one tier; pose layer only (no arms, no F1 scoring -> no new evaluation on VAL).
  1. re-runs M.register_messages on the believed-frame V3 maps and checks the E_msg trajectory against a job file (if given);
  2. replays the same loop instrumented: for every candidate the change of every edge term; for every GLOBALLY rejected candidate the
     energy of the alternative "move only the child agent j" (= Delta on edge e and Delta^-1 on j's child edges);
  3. E_msg at identity, at the TRUE corrections C_a = A_a^-1, at E4's corrections (R.register_forest) and at E4b's final corrections;
  4. per tree edge the gate at the TRUE pose (K0 evidence) for: the pair (E4 overlap_true), the two side unions (E4b's message form),
     the two sides' pooled max planes (E4's islands rule);
  5. per-agent position error (component mean removed) for E4 and E4b.
Usage: python -B diag_e4b.py SPLIT ROOMS INDEX SEED COND OUT_JSON [JOB_JSON]
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'scripts', 'claude_v342'))
sys.path.insert(0, SCRIPTS)

import json, time  # noqa: E401,E402
import numpy as np  # noqa: E402
import g4_registration_msg as M  # noqa: E402

R = M.R; G = R.G; W = G.W; I3 = R.I3


def main():
    split, rooms, index, seed, cond, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    job_json = sys.argv[7] if len(sys.argv) > 7 else None
    t0 = time.time(); prm = dict(R.MATCH)
    fx = W.make_fixture(split, index, rooms)
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min())
    F = G.forest(fx['edges'], n)
    rays = W.fibonacci_sphere(W.BEAMS)
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    c = G.CONDITIONS[cond]; has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0
    L_k0, L_pub, MAs = [], [], []
    for a in range(n):
        frame = None
        if has_inter:
            z = G.frame_draws(seed, rooms, index, a)
            frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
        MAs.append(R.frame_matrix(frame))
        scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
        L_k0.append(G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3'])
        L_pub.append(G.integrate_agent(shape, cell, ds, [G.to_public(s, frame) for s in scans], ('V3',))['V3'] if has_inter else L_k0[a])
    t_maps = time.time() - t0

    # 1. the estimator as shipped
    reg = M.register_messages(L_pub, F, cell, prm)
    out = dict(house=f'{split}_r{rooms}_i{index}', seed=seed, cond=cond, agents=n, components=F['comps'], E_msg=reg['E_msg'])
    if job_json:
        jb = json.load(open(job_json))['conds'][cond]['registration']
        out['traj_equals_job_file'] = bool(np.allclose(jb['E_msg'], reg['E_msg'], rtol=0, atol=0)) if len(jb['E_msg']) == len(reg['E_msg']) else False

    # 2. instrumented replay
    ev = [R.evidence(R.planes_logodds(L, cell, prm), cell, prm) for L in L_pub]
    S_list = [e['S'] for e in ev]; w_list = [e['w'] for e in ev]
    sides = M.tree_sides(F); keys = [k for k, _, _ in sides]
    parent, _ = R.tree_order(F)
    children = {a: [b for b in range(n) if parent.get(b) == a] for a in range(n)}
    C = {a: I3.copy() for a in range(n)}
    E, per = M.message_energy(S_list, C, sides, prm); tr = [E]; steps = []
    for sweep in range(M.SWEEPS):
        for k, A, B in sides:
            evA = M.side_message(S_list, w_list, C, A, cell); evB = M.side_message(S_list, w_list, C, B, cell)
            res = R.register_pair(evA, evB, cell, prm)
            if np.array_equal(res['M'], I3):
                steps.append(dict(sweep=sweep, edge=list(k), kept=False, reason='no local step', final_gate=res['final_gate'],
                                  identity_gate=res['identity_gate'], coarse=None if res['coarse'] is None else
                                  {kk: res['coarse'][kk] for kk in ('E', 'gate', 'n_inl', 'inl_wfrac', 'inl_wfrac_fwd', 'inl_wfrac_rev', 'min_eig', 'trust') if kk in res['coarse']}))
                continue
            Cn = dict(C)
            for b in B:
                Cn[b] = res['M'] @ C[b]
            En, pern = M.message_energy(S_list, Cn, sides, prm)
            d_terms = {str(kk): round(pern[q] - per[q], 4) for q, kk in enumerate(keys) if abs(pern[q] - per[q]) > 1e-9}
            rec = dict(sweep=sweep, edge=list(k), B=B, delta=R.params(res['M']), E_before=E, E_try=En, kept=bool(En < E), d_terms=d_terms,
                       local_dE=res['traj'][-1] - res['traj'][0])
            if not En < E:                                              # alternative: move only the child agent j
                j = k[1]; Ca = dict(C); Ca[j] = res['M'] @ C[j]
                Ea, pera = M.message_energy(S_list, Ca, sides, prm)
                rec.update(E_child_only=Ea, child_only_lower=bool(Ea < E),
                           d_terms_child_only={str(kk): round(pera[q] - per[q], 4) for q, kk in enumerate(keys) if abs(pera[q] - per[q]) > 1e-9})
            steps.append(rec)
            if En < E:
                C, E, per = Cn, En, pern; tr.append(E)
    out['replay_traj_equals'] = bool(tr == reg['E_msg'])
    out['steps'] = steps

    # 3. E_msg at reference states
    Ctrue = {a: R.inv(MAs[a]) for a in range(n)}
    e4 = R.register_forest(L_pub, F, cell, prm, 'prior')
    out['E_msg_at'] = dict(identity=M.message_energy(S_list, {a: I3 for a in range(n)}, sides, prm)[0],
                           truth=M.message_energy(S_list, Ctrue, sides, prm)[0],
                           e4=M.message_energy(S_list, {a: e4['C'][a] for a in range(n)}, sides, prm)[0],
                           e4b_final=M.message_energy(S_list, {a: reg['C'][a] for a in range(n)}, sides, prm)[0])
    out['E_msg_terms_at'] = dict(truth=M.message_energy(S_list, Ctrue, sides, prm)[1], e4b_final=reg['E_edges_final'],
                                 e4=M.message_energy(S_list, {a: e4['C'][a] for a in range(n)}, sides, prm)[1])
    # E4's own per-edge energy at E4's solution vs at the truth (pairwise objective)
    out['E_pose_pairwise_at'] = dict(e4=float(sum(R.edge_cost(S_list[j], S_list[i], e4['edges'][(i, j)]['M'], prm) for (i, j) in e4['keys'])),
                                     truth=float(sum(R.edge_cost(S_list[j], S_list[i], MAs[i] @ R.inv(MAs[j]), prm) for (i, j) in e4['keys'])))

    # 4. gates at the true pose (K0 evidence: every agent already in the true common frame)
    planes_k0 = [R.planes_logodds(L, cell, prm) for L in L_k0]
    evk = [R.evidence(p, cell, prm) for p in planes_k0]
    Sk = [e['S'] for e in evk]; Iall = {a: I3 for a in range(n)}
    gates = []
    for k, A, B in sides:
        i, j = k
        pair = R.observable_pair(evk[i], evk[j], prm)
        SA = M._side_surfels(Sk, Iall, A); SB = M._side_surfels(Sk, Iall, B)
        if len(SA['m']) >= 3 and len(SB['m']) >= 3:
            _, info = R.edge_cost(SB, SA, I3, prm, full=True)
            ok_u, st_u = R.gate(SB, SA, I3, info, R.overlap_pivot(SB, SA, prm), prm)
        else:
            ok_u, st_u = False, {}
        pool = lambda g: R.evidence((np.max([planes_k0[b][0] for b in g], 0), np.max([planes_k0[b][1] for b in g], 0)), cell, prm)  # noqa: E731
        ok_p, st_p = R.observable_pair(pool(A), pool(B), prm)
        gates.append(dict(edge=list(k), nA=len(A), nB=len(B), pair=dict(ok=pair[0], **{kk: pair[1][kk] for kk in ('n_inl', 'inl_wfrac', 'min_eig')}),
                          union=dict(ok=ok_u, **{kk: st_u.get(kk) for kk in ('n_inl', 'inl_wfrac', 'inl_wfrac_fwd', 'inl_wfrac_rev', 'min_eig')}),
                          pooled=dict(ok=ok_p, **{kk: st_p[kk] for kk in ('n_inl', 'inl_wfrac', 'min_eig')})))
    out['gates_at_truth'] = gates
    out['islands'] = [list(x) for x in R.observability_islands(F, planes_k0, cell, prm)[0]]

    # 5. per-agent errors
    out['agent_err'] = dict(before=[round(x['disp_cm'], 2) for x in R.agent_errors([I3] * n, MAs, F, reg['pivots'])['per_agent']],
                            e4=[round(x['disp_cm'], 2) for x in R.agent_errors(e4['K'], MAs, F, reg['pivots'])['per_agent']],
                            e4b=[round(x['disp_cm'], 2) for x in R.agent_errors(reg['K'], MAs, F, reg['pivots'])['per_agent']])
    out['agent_order'] = [x['agent'] for x in R.agent_errors([I3] * n, MAs, F, reg['pivots'])['per_agent']]
    out['seconds'] = dict(maps=t_maps, total=time.time() - t0); out['peak_mb'] = G._peak_mb()
    json.dump(out, open(out_path, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(json.dumps(dict(E_msg_at=out['E_msg_at'], E_pose_pairwise_at=out['E_pose_pairwise_at'], agent_err=out['agent_err'],
                          traj_equals_job_file=out.get('traj_equals_job_file'), replay=out['replay_traj_equals'], seconds=out['seconds']), indent=1))
    for s in steps:
        if not s.get('kept'):
            print('not kept:', s['sweep'], s['edge'], s.get('reason', ''), 'E_try-E', round(s.get('E_try', 0) - s.get('E_before', 0), 4) if 'E_try' in s else '',
                  'child_only_lower', s.get('child_only_lower'), 'E_child-E', round(s['E_child_only'] - s['E_before'], 4) if 'E_child_only' in s else '',
                  'd_terms', s.get('d_terms'))
    for g in gates:
        print('gate@truth', g['edge'], 'nA', g['nA'], 'nB', g['nB'], 'pair', g['pair']['ok'], 'union', g['union']['ok'], g['union'].get('n_inl'),
              None if g['union'].get('inl_wfrac') is None else round(g['union']['inl_wfrac'], 3), 'pooled', g['pooled']['ok'], g['pooled']['n_inl'], round(g['pooled']['inl_wfrac'], 3))
    print('islands', out['islands'])


if __name__ == '__main__':
    main()
