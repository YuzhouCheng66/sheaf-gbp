"""probe_B_code (analyst B, 2026-10-26, CPU only, report-only). Loads a finished checkpoint (default 3S-X c2-s342101: 32x32x1 @ 8+8) and
the 24-house VAL panel pickled by probe_B_genval.py, then
  --mode stats   : code utilisation on evidenced coordinates (mu level entropy, saturation, q-code entropy, mu / q vs column content)
  --mode bits    : post-hoc bit-depth sweep. mu re-quantised to 2^(M-1)-1 levels per side AT THE SOURCE and precision floored onto the
                   2^Q-1 log codebook AT THE SOURCE (own belief too), wire at the same M,Q; K0 and R8 surface / var / furniture F1.
                   (the model was trained at 8+8, so this is a LOWER bound on what a model trained at M+Q would reach)
  --mode routing : can the decoder put a coordinate's information in that coordinate's own column? Finite-difference footprint of
                   a +-delta change of one fused mean on the decoder output (|d known logit| + |d occ logit| summed over z), share of
                   the footprint in the coordinate's own column vs its fold siblings (same 16x16 start cell) vs non-sibling neighbours.
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python probe_B_code.py --mode bits --ckpt <run>/checkpoints/final.pt --val val24.pkl --out x.json
"""
import argparse, json, math, os, pickle, sys, time
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--mode', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--val', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--scripts', default='/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts')
ap.add_argument('--houses', default='all'); ap.add_argument('--configs', default='8,8;6,6;5,5;4,4;3,3;8,4;4,8'); ap.add_argument('--stages', default='0,8')
ap.add_argument('--delta', type=float, default=0.25); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--qrange', default='1e-8,128', help='bits mode: log codebook range of the SOURCE precision quantiser (1e-4,4 = matched to the head range)')
ap.add_argument('--qround', default='floor', choices=('floor', 'nearest')); ap.add_argument('--qtransparent', action='store_true', help='bits mode: wire precision at 16 bits (the source quantiser is the only one)')
args = ap.parse_args()
torch.set_num_threads(args.threads)
sys.path.insert(0, args.scripts)
import train_3d_v431 as T
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False); meta = ck['meta']
T.DECODER = meta.get('decoder', 'd3'); T.PRECISION = meta.get('precision', 'learned'); T.KNOWN_WEIGHT = meta.get('known_weight', 'uniform')
T.BX, T.BY, T.BZ = meta['blocks']; T.D = T.BX * T.BY * T.BZ; T.DECODER_WIDTH = meta.get('decoder_width', 1.0)
WB = tuple(meta.get('wire_bits', [8, 8]))
dims = (128, 128, 32)
model = T.SheafModel3D(dims, meta['seed']); model.load_state_dict(ck['state_dict']); model.eval()
V = pickle.load(open(args.val, 'rb')); fxs = [f for f in V['fixtures'] if f is not None]
if args.houses != 'all':
    fxs = [fxs[int(i)] for i in args.houses.split(',')]
print('ckpt', args.ckpt, 'blocks', meta['blocks'], 'wire', WB, 'width', T.DECODER_WIDTH, 'houses', len(fxs), flush=True)


def set_bits(M, Q):
    T.WIRE_BITS = (M, Q); T._v378.MU_BITS, T._v378.Q_BITS = M, Q; T.LEVELS = 2 ** (M - 1) - 1


def q_floor(q, Q, low=1e-8, high=128., rnd='floor'):
    nq = 2 ** Q - 1
    x = (q.double().clamp(low, high).log() - math.log(low)) / (math.log(high) - math.log(low)) * (nq - 1)
    code = torch.floor(x) if rnd == 'floor' else torch.round(x)
    return torch.exp(code / (nq - 1) * (math.log(high) - math.log(low)) + math.log(low)).float(), code


def scene_tensors(fx):
    T.CHART_M = fx['chart_m']
    own = torch.from_numpy(fx['private']); reached = torch.from_numpy(fx['reached'])
    slab = torch.from_numpy(fx['slab'].reshape(-1)); obj = torch.from_numpy(fx['obj'].reshape(-1)); struct = torch.from_numpy(fx['struct'].reshape(-1))
    return own, reached, slab, obj, struct


res = dict(ckpt=args.ckpt, blocks=meta['blocks'], wire_bits=list(WB), mode=args.mode)
t0 = time.time()
with torch.no_grad():
    if args.mode == 'bits':
        stages = tuple(int(s) for s in args.stages.split(','))
        configs = [tuple(int(v) for v in c.split(',')) for c in args.configs.split(';')]
        rows = {c: [] for c in configs}
        for hi, fx in enumerate(fxs):
            own, reached, slab, obj, struct = scene_tensors(fx)
            set_bits(*WB); trunk = model.trunk(own)
            for (M, Q) in configs:
                set_bits(M, 16 if args.qtransparent else Q)
                mu, q, mask = model.head(trunk, own); mu = T.latent_quantize(mu)
                lo_, hi_ = (float(v) for v in args.qrange.split(','))
                qb, _ = q_floor(q, Q, lo_, hi_, args.qround)
                feats = T.rollout_max(mu, qb, mask, fx['edges'], stages=stages)
                row = dict(house=hi, chart_m=float(fx['chart_m']), rooms=fx['spec']['rooms'])
                for k in stages:
                    tgt = T.stream_targets(own, reached[T.HOPS.index(k)]); pred = T.predict(model.decoder, feats[k])
                    row[f'K{k}'] = T.agg(T.metrics3d(pred, tgt, dims, slab, obj, struct))
                rows[(M, Q)].append(row)
            print('house', hi, 'L', fx['chart_m'], ' '.join('%d+%d K0 %.3f' % (c[0], c[1], rows[c][-1]['K0']['surface_f1']) + (' R8 %.3f' % rows[c][-1]['K8']['surface_f1'] if 8 in stages else '') for c in configs), '%.0fs' % (time.time() - t0), flush=True)
        set_bits(*WB)
        summ = {}
        for c in configs:
            s = {}
            for k in stages:
                for m in ('surface_f1', 'surface_f1_tol1', 'var_occ_f1', 'furn_f1', 'obj_recall', 'unknown_known_fpr'):
                    v = [r[f'K{k}'][m] for r in rows[c] if r[f'K{k}'][m] is not None]; s[f'K{k}_{m}'] = float(np.mean(v))
            s['bytes_per_message'] = T.D * (c[0] + c[1]) / 8
            summ['%d+%d' % c] = s
        res['summary'] = summ; res['rows'] = {'%d+%d' % c: rows[c] for c in configs}
        for c, s in summ.items():
            print(c, {k: round(v, 4) for k, v in s.items()})
    elif args.mode == 'wire':
        # how many coordinates of each wire message carry evidence (p above the prior), per round; lossless sparse size of a message
        set_bits(*WB); out = {r: [] for r in range(0, 9)}; mus, qs = [], []
        for hi, fx in enumerate(fxs):
            own, reached, slab, obj, struct = scene_tensors(fx)
            mu, q, mask = model.source(own); n = mu.shape[0]
            prior = max(T.LATENT_PRIOR / (n * T.D), 1e-8)
            ev = (mask.double() * q.double()); p_own = ev + prior; e_own = ev * mu.double()
            graph = [(min(i, j), max(i, j)) for i, j, _ in fx['edges']]; directed = [(i, j) for i, j in graph] + [(j, i) for i, j in graph]
            incoming = {j: [a for a, (i2, j2) in enumerate(directed) if j2 == j] for j in range(n)}
            pw, ew, _ = T.kbit_wire_quantize(p_own, e_own)
            msg_p = [torch.zeros(T.D, dtype=torch.float64) for _ in directed]; msg_e = [torch.zeros(T.D, dtype=torch.float64) for _ in directed]
            out[0].extend([float((p_own[i] > 2 * prior).float().mean()) for i in range(n)])
            for r in range(1, 9):
                new_p, new_e = [], []
                for a, (i, j) in enumerate(directed):
                    pc, ec = T.select([pw[i]] + [msg_p[b] for b in incoming[i] if directed[b][0] != j], [ew[i]] + [msg_e[b] for b in incoming[i] if directed[b][0] != j])
                    new_p.append(pc); new_e.append(ec)
                msg_p, msg_e = new_p, new_e
                out[r].extend([float((mp > 2 * prior).float().mean()) for mp in msg_p])
            m = mask.bool(); mus.append(torch.round(mu[m] * T.LEVELS).numpy()); qs.append(q_floor(q[m], WB[1])[1].numpy())
        mus = np.concatenate(mus); qs = np.concatenate(qs)
        def H(x):
            _, c = np.unique(x, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
        hmu, hq = H(mus), H(qs); hmu4 = H(np.round(mus / T.LEVELS * 7))
        res.update(D=T.D, active_fraction_by_round={r: float(np.mean(v)) for r, v in out.items()}, H_mu8_bits=hmu, H_q8_bits=hq, H_mu4_bits=hmu4,
                   active_fraction_p95_by_round={r: float(np.percentile(v, 95)) for r, v in out.items()}, active_fraction_max_by_round={r: float(np.max(v)) for r, v in out.items()})
        fixed = T.D * 2
        res['bytes_fixed'] = fixed
        res['bytes_sparse_lossless_by_round'] = {r: float(T.D / 8 + np.mean(v) * T.D * (hmu + hq) / 8) for r, v in out.items()}
        print(json.dumps(res, indent=1))
    elif args.mode == 'stats':
        set_bits(*WB); lev = T.LEVELS
        allmu, allq, feats_rows = [], [], []
        k = (dims[0] // T.BX, dims[1] // T.BY, dims[2] // T.BZ)
        for hi, fx in enumerate(fxs):
            own, reached, slab, obj, struct = scene_tensors(fx)
            mu, q, mask = model.source(own)
            m = mask.bool()
            allmu.append(mu[m].numpy()); allq.append(q[m].numpy())

            def pool(x):
                return torch.nn.functional.avg_pool3d(x.float()[:, None], k).flatten(1)
            kn = pool(own >= 0); fr = pool(own == 0); oc = pool(own == 1)
            objo = pool((own == 1) & torch.from_numpy(fx['obj'])[None]); lowo = pool((own == 1) & (torch.arange(32) < 3).view(1, 1, 1, 32))
            feats_rows.append(np.stack([kn[m].numpy(), fr[m].numpy(), oc[m].numpy(), objo[m].numpy(), lowo[m].numpy(), mu[m].numpy(), np.log(q[m].numpy())], 1))
        mu = np.concatenate(allmu); q = np.concatenate(allq); F = np.concatenate(feats_rows)
        lvl = np.round(mu * lev).astype(int); _, cnt = np.unique(lvl, return_counts=True); p = cnt / cnt.sum()
        _, qcode = q_floor(torch.from_numpy(q), WB[1]); _, qc = np.unique(qcode.numpy(), return_counts=True); pq = qc / qc.sum()

        def mi(a, b, bins=16):
            h, _, _ = np.histogram2d(a, b, bins=bins); pj = h / h.sum(); pa = pj.sum(1, keepdims=True); pb = pj.sum(0, keepdims=True)
            nz = pj > 0; return float((pj[nz] * np.log2(pj[nz] / (pa @ pb)[nz])).sum())

        def rk(x):
            return np.argsort(np.argsort(x))
        names = ['known_frac', 'free_frac', 'occ_frac', 'obj_occ_frac', 'low_occ_frac']
        res.update(n_evidenced=int(len(mu)), evidenced_per_agent=float(len(mu) / (8 * len(fxs))), D=T.D,
                   mu_levels_used=int(len(cnt)), mu_entropy_bits=float(-(p * np.log2(p)).sum()), mu_saturated_frac=float((np.abs(mu) > .98).mean()),
                   mu_near0_frac=float((np.abs(mu) < .05).mean()), q_codes_used=int(len(qc)), q_entropy_bits=float(-(pq * np.log2(pq)).sum()),
                   q_at_max_frac=float((q > 3.9).mean()), q_at_min_frac=float((q < 2e-4).mean()),
                   spearman_mu={n: float(np.corrcoef(rk(F[:, i]), rk(F[:, 5]))[0, 1]) for i, n in enumerate(names)},
                   spearman_logq={n: float(np.corrcoef(rk(F[:, i]), rk(F[:, 6]))[0, 1]) for i, n in enumerate(names)},
                   mi16_mu_bits={n: mi(F[:, i], F[:, 5]) for i, n in enumerate(names)}, mi16_logq_bits={n: mi(F[:, i], F[:, 6]) for i, n in enumerate(names)},
                   mi16_mu_logq_bits=mi(F[:, 5], F[:, 6]))
        X = np.c_[np.ones(len(F)), F[:, 5], F[:, 6], F[:, 5] ** 2, F[:, 6] ** 2, F[:, 5] * F[:, 6]]
        r2 = {}
        for i, n in enumerate(names):
            b, *_ = np.linalg.lstsq(X, F[:, i], rcond=None); r2[n] = float(1 - ((F[:, i] - X @ b) ** 2).mean() / F[:, i].var())
        res['quadratic_R2_from_mu_logq'] = r2
        print(json.dumps(res, indent=1))
    elif args.mode == 'routing':
        set_bits(*WB); fold = max(1, T.BX // 16); cw = dims[0] // T.BX          # coordinate width in voxels
        stats = []
        for hi, fx in enumerate(fxs):
            own, reached, slab, obj, struct = scene_tensors(fx)
            mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'], stages=(0,))
            post = feats[0][:1].clone()                                     # agent 0's K0 belief [1, 2D]
            ev = mask[0].reshape(T.BX, T.BY).bool()
            kl0, ol0 = model.decoder(post)
            base = torch.stack((kl0, ol0)).reshape(2, *dims)
            spacing = 16 if T.BX >= 32 else 8
            step = max(1, spacing // 4)
            for ox in range(0, spacing, step):
                for oy in range(0, spacing, step):
                    pts = [(i, j) for i in range(ox, T.BX, spacing) for j in range(oy, T.BY, spacing) if ev[i, j]]
                    if not pts:
                        continue
                    pp = post.clone()
                    for (i, j) in pts:
                        c = i * T.BY + j; v = float(pp[0, c]); pp[0, c] = v - args.delta if v > 0 else v + args.delta
                    kl, ol = model.decoder(pp)
                    resp = (torch.stack((kl, ol)).reshape(2, *dims) - base).abs().sum((0, 3))   # [128, 128]
                    for (i, j) in pts:
                        x0, y0 = i * cw, j * cw; win = 3 * cw * fold
                        sx, sy = max(0, x0 - win), max(0, y0 - win)
                        W_ = resp[sx:x0 + cw + win, sy:y0 + cw + win]; tot = float(W_.sum())
                        if tot <= 0:
                            continue
                        own_s = float(resp[x0:x0 + cw, y0:y0 + cw].sum()) / tot
                        pi, pj = i // fold, j // fold
                        sib, non = [], []
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                if di == 0 and dj == 0:
                                    continue
                                a, b = i + di, j + dj
                                if not (0 <= a < T.BX and 0 <= b < T.BY):
                                    continue
                                s_ = float(resp[a * cw:(a + 1) * cw, b * cw:(b + 1) * cw].sum()) / tot
                                (sib if (a // fold == pi and b // fold == pj) else non).append(s_)
                        xs = torch.arange(W_.shape[0]).float()[:, None] + sx; ys = torch.arange(W_.shape[1]).float()[None] + sy
                        cx = float((W_ * xs).sum()) / tot; cy = float((W_ * ys).sum()) / tot
                        ccx, ccy = x0 + (cw - 1) / 2, y0 + (cw - 1) / 2
                        pcx, pcy = pi * cw * fold + (cw * fold - 1) / 2, pj * cw * fold + (cw * fold - 1) / 2
                        dx, dy = ccx - pcx, ccy - pcy; nrm = dx * dx + dy * dy
                        proj = ((cx - pcx) * dx + (cy - pcy) * dy) / nrm if nrm > 0 else None   # 1 = centred on own column, 0 = on the fold parent
                        stats.append(dict(house=hi, i=i, j=j, own=own_s, sib=float(np.mean(sib)) if sib else None, non=float(np.mean(non)) if non else None,
                                          centroid_err_vox=float(math.hypot(cx - ccx, cy - ccy)), proj_own=proj))
            print('house', hi, 'n', len(stats), '%.0fs' % (time.time() - t0), flush=True)
        own_ = np.array([s['own'] for s in stats]); sib_ = np.array([s['sib'] for s in stats if s['sib'] is not None]); non_ = np.array([s['non'] for s in stats if s['non'] is not None])
        pr = np.array([s['proj_own'] for s in stats if s['proj_own'] is not None])
        res.update(n=len(stats), coord_width_vox=cw, fold=fold, own_share_mean=float(own_.mean()), sibling_share_mean=float(sib_.mean()) if len(sib_) else None,
                   nonsibling_share_mean=float(non_.mean()) if len(non_) else None, own_over_sibling=float(own_.mean() / sib_.mean()) if len(sib_) else None,
                   own_over_nonsibling=float(own_.mean() / non_.mean()) if len(non_) else None, centroid_err_vox_median=float(np.median([s['centroid_err_vox'] for s in stats])),
                   proj_own_median=float(np.median(pr)) if len(pr) else None, proj_own_mean=float(pr.mean()) if len(pr) else None)
        res['stats'] = stats
        print({k: v for k, v in res.items() if k != 'stats'})
res['seconds'] = time.time() - t0
json.dump(res, open(args.out, 'w'), indent=1, default=float)
