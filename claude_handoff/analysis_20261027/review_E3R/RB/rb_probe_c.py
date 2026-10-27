"""R-B probe (b)/(c), 2026-10-27: can a noisy cam_nf agent map be rebuilt by ray casting from "observed occupied voxels + believed
poses" (the W3 decoder), as in the clean world? Per agent: W3 description (g4_exact_codec.describe_w3, read-only import) of the
agent's own ternary map, the receiver's reconstruction = W.observe of the painted world from the agent's BELIEVED poses (true pose +
the cam_nf per-frame draw, which the agent knows), and the residual = voxels where reconstruction != the agent's map. Also the
hits-only world (no plan / boxes). NET read-out (frozen 342101, table A 'sign' values) on the reconstructed maps (lossy W3: no
residual) vs the exact maps. CPU only; inputs never copied out.
"""
import os, sys, json, time, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v, '1')
sys.dont_write_bytecode = True
ROOT = r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill'
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import g4_e3r as E
G = E.G
import g4_exact_codec as X
W = G.W
from scipy.special import gammaln
import torch
import rb_probe_a as PA

INP = PA.INP
OUT = os.path.join(HERE, 'out_c')


def log2C(N, d):
    return float((gammaln(N + 1) - gammaln(d + 1) - gammaln(N - d + 1)) / math.log(2))


def trajectory(split, index, rooms):
    spec = W.world_spec(split, index, rooms)
    r = W.rasterize(spec, tuple(W.DIMS['R1']))
    rng = np.random.default_rng(W._seed(spec['world_id'] + f':poses:{W.AGENTS}'))
    centres = W.agent_positions(spec, r['occ'].astype(bool), r['cell'], W.AGENTS, rng)
    jitter = rng.uniform(-.1, .1, (W.POSES, W.AGENTS, 3)); jitter[:, :, 2] = 0.
    return r['occ'], centres[None] + jitter


def recon(world, cell, poses):
    out = np.full(world.shape, -1, np.int8)
    for o, D in poses:
        out = np.maximum(out, W.observe(world, cell, o[None], D))
    return out


def diff_stats(act, rec):
    ak = act >= 0; rk = rec >= 0
    d = dict(known_act=int(ak.sum()), known_rec=int(rk.sum()), miss=int((ak & ~rk).sum()), extra=int((~ak & rk).sum()),
             occ_diff=int((ak & rk & (act != rec)).sum()))
    d['diff'] = d['miss'] + d['extra'] + d['occ_diff']
    d['resid_bits'] = log2C(act.size, d['diff']) + d['diff']
    return d


def job(fname):
    t0 = time.time(); torch.set_num_threads(2)
    panel, r, i, s = fname[:-4].split('_'); rooms = int(r[1:]); index = int(i[1:]); seed = int(s[1:])
    z = np.load(os.path.join(INP, 'export', fname))
    occ_chk, traj = trajectory('VALIDATION', index, rooms)
    assert np.array_equal(np.asarray(occ_chk, np.int8), z['occ']), 'trajectory rebuild: occ mismatch'
    cell = z['cell']; n = z['private'].shape[0]
    H = G.House(z['occ'], z['slab'], z['obj'], cell); F = G.forest(E.edges_list(z['edges']), n)
    R8T = [G.Target(z['private'][cm].max(0), H, mult=(z['private'][cm] >= 0).sum(0)) for cm in F['comps']]
    c = G.CONDITIONS['cam_nf']; rays = W.fibonacci_sphere(W.BEAMS)
    # sanity: the clean map of agent 0 is reproduced by observe from the true poses
    agents = []; rec_w3 = []; rec_hit = []
    for a in range(n):
        L = z['L_k0_cam_nf'][a]; act = G.tern(L); Lc = z['L_k0_clean'][a]; clean = G.tern(Lc)
        poses = []
        for j in range(traj.shape[0]):
            zp = G.pose_draws(seed, rooms, index, a, j, len(rays))['z_pose']
            o = traj[j, a] + np.array([zp[0] * c['pose_t'], zp[1] * c['pose_t'], zp[2] * c['pose_t'] / 2])
            D = rays @ G.rotz(float(zp[3] * c['pose_yaw'])).T
            poses.append((o, D))
        kf, kc, lz, plan, boxes = X.describe_w3(act.ravel(), None)
        painted = X.paint(kf, kc, lz, plan, boxes).reshape(act.shape)
        ckf, ckc, clz, cplan, cboxes = X.describe_w3(clean.ravel(), None)
        rw = recon(painted.astype(np.int8), cell, poses)
        rh = recon((act == 1).astype(np.int8), cell, poses)
        if a == 0:
            rc = recon(X.paint(ckf, ckc, clz, cplan, cboxes).reshape(act.shape).astype(np.int8), cell, [(traj[j, a], rays) for j in range(traj.shape[0])])
            clean_exact = int((rc != clean).sum())
        rec_w3.append(rw); rec_hit.append(rh)
        agents.append(dict(boxes=len(boxes), boxes_clean=len(cboxes), plan_cells=int((plan > 0).sum()), plan_cells_clean=int((cplan > 0).sum()),
                           tern_ctx=G.ctx_tern_bytes(act), tern_ctx_clean=G.ctx_tern_bytes(clean),
                           mask_ctx=G.ctx_tern_bytes(np.where(act >= 0, np.int8(0), np.int8(-1))),
                           mask_ctx_clean=G.ctx_tern_bytes(np.where(clean >= 0, np.int8(0), np.int8(-1))),
                           occ_vox=int((act == 1).sum()), occ_vox_clean=int((clean == 1).sum()),
                           w3=diff_stats(act, rw), hit=diff_stats(act, rh)))
    # NET read-out on exact ternary maps vs W3 reconstructions vs hits-only reconstructions (table A 'sign' values)
    tab, _ = PA.quant_table('A', 'sign')
    ck, model = E.load_ckpt(os.path.join(INP, 's342101', 'final.pt'), 'cpu')
    j = int(json.load(open(os.path.join(INP, 's342101', 'threshold.json')))['j'])

    def pseudoL(t):
        return np.where(t == 1, np.int8(70), np.where(t == 0, np.int8(-40), np.int8(E.UNK))).astype(np.int8)
    res = {}
    for name, maps in (('exact', [pseudoL(G.tern(z['L_k0_cam_nf'][a])) for a in range(n)]), ('w3_recon', [pseudoL(t) for t in rec_w3]),
                       ('hit_recon', [pseudoL(t) for t in rec_hit])):
        rows = []
        sums = [E.direct_sums(maps, cm, tab) for cm in F['comps']]
        lg = E.net_logits(model, [x[0] for x in sums], [x[1] for x in sums], 'cpu', 1)
        for ci, cm in enumerate(F['comps']):
            S, kc = sums[ci]; p = E.net_readout(lg[ci], kc, j)
            rows.append([G.score(p, R8T[ci]), len(cm)])
        res[name] = rows
    # protection: the clean tier under the quantised tables (frozen 342101, table A)
    for kind in ('full', 'q3', 'sign'):
        tabk, _ = PA.quant_table('A', kind)
        cmaps = list(z['L_k0_clean'])
        sums = [E.direct_sums(cmaps, cm, tabk) for cm in F['comps']]
        lg = E.net_logits(model, [x[0] for x in sums], [x[1] for x in sums], 'cpu', 1)
        res[f'clean_{kind}'] = [[G.score(E.net_readout(lg[ci], sums[ci][1], j), R8T[ci]), len(cm)] for ci, cm in enumerate(F['comps'])]
    # the root's input once: the component count field (k, n_occ) of the exact ternary maps, ctx-coded as one map each
    field = []
    for cm in F['comps']:
        T = [G.tern(z['L_k0_cam_nf'][a]) for a in cm]
        kc = sum((t >= 0).astype(np.int32) for t in T); no = sum((t == 1).astype(np.int32) for t in T); K = kc > 0
        field.append(dict(agents=len(cm), k=G.int_ctx_bytes(kc, K), n_occ=G.int_ctx_bytes(no, K),
                          union_mask=G.ctx_tern_bytes(np.where(K, np.int8(0), np.int8(-1)))))
    rec = dict(file=fname, agents=agents, R8=res, field=field, clean_recon_mismatch_agent0=clean_exact, seconds=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, fname.replace('.npz', '.json')), 'w') as fh:
        json.dump(rec, fh)
    return fname, rec['seconds']


if __name__ == '__main__':
    import multiprocessing as mp
    files = sys.argv[1].split(',')
    files = [f for f in files if not os.path.exists(os.path.join(OUT, f.replace('.npz', '.json')))]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if workers <= 1:
        for f in files:
            print(job(f), flush=True)
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(job, files):
                print(r, time.strftime('%H:%M:%S', time.gmtime()), flush=True)
