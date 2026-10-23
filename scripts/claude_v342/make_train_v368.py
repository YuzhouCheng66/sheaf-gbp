"""Build train_v368.py from train_v365.py (Claude, 2026-10-22 23:10 UTC).
v368 = v365 + two documented changes:
 (1) LATTICE JOIN adopted in training (D4-1): rollout_max selects by max quantised precision with ties broken by the quantised mean
     (order-independent; computed on detached tensors, so no where-NaN gradients), and a node's belief uses its own factor exactly as
     the wire encodes it (own-q). Inference-only effect on trained weights measured at <= 0.04 pp R8, -0.1 pp K0.
 (2) RATE TERM (item 1, entropy-coded codes; --rate-model): a learned factorised entropy model (shared logistic mixture, K=8) over the
     two code alphabets of an OBSERVED block: mean index on the 2^(k-1)-1-per-side grid and precision index on the k-bit log codebook
     (kbit wire). Unobserved blocks are not sent (64-bit presence mask). Training: hard codes (STE) feed the GBP/decoder as before;
     the rate is the Balle-style noisy likelihood -log2 P(x + U) of the continuous values; the entropy model is always fitted to the
     detached codes (so the rate is measured even at lambda = 0); the encoder receives lambda * rate (bits per observed coordinate).
     Evaluation reports the exact code length of the hard codes under the learned model and the empirical entropies.
The SheafModel state_dict is unchanged (warm starts load strictly); the entropy model is saved as checkpoint['entropy_model'].
python make_train_v368.py train_v365.py train_v368.py"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)

# ---------------------------------------------------------------- (1) lattice join + own-q
rep('''    def select(p_list, eta_list):
        """per-coordinate argmax precision over a list of [N,64] tensors (first entry preferred on ties)."""
        P = torch.stack(p_list, 0); E = torch.stack(eta_list, 0)
        tie = torch.arange(P.shape[0], device=P.device, dtype=P.dtype).view(-1, 1, 1) * 1e-15
        idx = (P - tie).argmax(0, keepdim=True)
        return P.gather(0, idx)[0], E.gather(0, idx)[0]
''', '''    def select(p_list, eta_list):
        """v368 lattice join: per-coordinate max quantised precision, ties broken by the larger quantised mean (codebook values,
        identical at every node) -> commutative/associative/idempotent on every coordinate. Index computed on detached tensors."""
        P = torch.stack(p_list, 0); E = torch.stack(eta_list, 0)
        with torch.no_grad():
            Pd = P.detach(); pmax = Pd.max(0, keepdim=True).values
            mean = E.detach() / Pd.clamp_min(1e-300)
            key = torch.where(Pd == pmax, mean, torch.full_like(mean, -float('inf')))
            idx = key.argmax(0, keepdim=True)
        return P.gather(0, idx)[0], E.gather(0, idx)[0]
''')
rep('''                cand_p = [p_own[j]] + [msg_p[a] for a in incoming[j]]
                cand_e = [eta_own[j]] + [msg_e[a] for a in incoming[j]]
                pj, ej = select([c[None] for c in cand_p], [c[None] for c in cand_e])''',
'''                cand_p = [p_wire[j]] + [msg_p[a] for a in incoming[j]]       # v368 own-q: the node's own factor as the wire encodes it
                cand_e = [e_wire[j]] + [msg_e[a] for a in incoming[j]]
                pj, ej = select([c[None] for c in cand_p], [c[None] for c in cand_e])''')

# ---------------------------------------------------------------- (2) entropy model
rep('''PRECISION_MODE = 'learned'   # --precision learned''', '''class LogisticMixture(nn.Module):
    """Shared factorised density for one scalar code alphabet (K logistic components). bits(x, step, lo, hi): -log2 P(bin of x),
    with the two end bins absorbing the tails (a proper pmf over the finite alphabet)."""
    def __init__(self, K=8, init_scale=.3, centre=0.):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(K)); self.means = nn.Parameter(torch.linspace(-1, 1, K) * 1. + centre)
        self.log_scales = nn.Parameter(torch.full((K,), math.log(init_scale)))

    def bits(self, x, step, lo, hi, detach_params=False):
        lg, m, ls = self.logits, self.means, self.log_scales
        if detach_params:
            lg, m, ls = lg.detach(), m.detach(), ls.detach()
        x = x.double()[..., None]; w = torch.softmax(lg.double(), 0); m = m.double(); s = ls.double().exp()
        up = torch.sigmoid((x + step / 2 - m) / s); dn = torch.sigmoid((x - step / 2 - m) / s)
        tol = 1e-6 * step
        up = torch.where(x + step / 2 >= hi - tol, torch.ones_like(up), up); dn = torch.where(x - step / 2 <= lo + tol, torch.zeros_like(dn), dn)
        p = ((up - dn) * w).sum(-1).clamp_min(1e-12)
        return -p.log2()


class EntropyModel(nn.Module):
    """Rate of an observed block's code: mean index (uniform grid, step 1/levels on [-1,1]) + precision index (log codebook)."""
    def __init__(self):
        super().__init__()
        self.mu = LogisticMixture(K=8, init_scale=.3, centre=0.)
        self.lp = LogisticMixture(K=8, init_scale=1., centre=-2.)

    def geometry(self):
        levels = 2 ** (MU_BITS - 1) - 1; nq = 2 ** Q_BITS - 1
        lo, hi = math.log(1e-8), math.log(128.)
        return levels, nq, lo, hi, (hi - lo) / (nq - 1)

    def rate(self, mu_c, logp_c, mask, noisy=True, detach_inputs=False, detach_params=False):
        """mean bits per observed coordinate. mu_c in [-1,1] (continuous), logp_c = log(q + prior) (continuous)."""
        levels, nq, lo, hi, s = self.geometry()
        m = mu_c.clamp(-1, 1); y = logp_c.clamp(lo, hi)
        if detach_inputs:
            m, y = m.detach(), y.detach()
        if noisy:
            m = m + (torch.rand_like(m) - .5) / levels
            y = y + (torch.rand_like(y) - .5) * s
        b = self.mu.bits(m, 1. / levels, -1., 1., detach_params) + self.lp.bits(y, s, lo, hi, detach_params)
        w = mask.double()
        return (b * w).sum() / w.sum().clamp_min(1)

    @torch.no_grad()
    def hard_codes(self, mu_c, logp_c):
        levels, nq, lo, hi, s = self.geometry()
        mi = torch.round(mu_c.clamp(-1, 1) * levels)                                   # -levels..levels
        ci = torch.floor((logp_c.clamp(lo, hi) - lo) / s).clamp(0, nq - 2)             # floor codebook (kbit wire); code nq-1 is the point p=128, unreachable (q<=4)
        return mi, ci

    @torch.no_grad()
    def code_bits(self, mu_c, logp_c, mask):
        """exact code length (bits) of the HARD codes under the model, summed over observed coordinates; plus count."""
        levels, nq, lo, hi, s = self.geometry()
        mi, ci = self.hard_codes(mu_c, logp_c)
        xm = mi / levels; xy = lo + (ci + .5) * s
        b = self.mu.bits(xm, 1. / levels, -1., 1.) + self.lp.bits(xy, s, lo, hi)
        w = mask.double()
        return float((b * w).sum()), float(w.sum())


PRECISION_MODE = 'learned'   # --precision learned''')

rep("    ap.add_argument('--world', default='v293',",
    "    ap.add_argument('--rate-model', action='store_true', help='fit the entropy model and report code lengths (requires --wire kbit)')\n"
    "    ap.add_argument('--rate-lambda', type=float, default=0., help='weight of the rate term (bits per observed coordinate) on the encoder')\n"
    "    ap.add_argument('--rate-prefit-steps', type=int, default=3000, help='full-batch fit of the entropy model on the warm-start encoder TRAIN codes before training (review v368 M1)')\n"
    "    ap.add_argument('--world', default='v293',")
rep("    model = SheafModel(args.arm, size, args.seed).to(device)\n",
    "    model = SheafModel(args.arm, size, args.seed).to(device)\n"
    "    assert not (args.rate_model and args.wire != 'kbit'), 'the rate model is defined on the k-bit wire codebooks'\n"
    "    assert not (args.rate_lambda > 0 and not args.rate_model), '--rate-lambda needs --rate-model'\n"
    "    em = EntropyModel().to(device) if args.rate_model else None\n")
rep("    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)\n",
    "    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)\n"
    "    if em is not None:\n"
    "        em_opt = torch.optim.Adam(em.parameters(), lr=1e-3)                  # own optimizer, CONSTANT lr (review v368 M1)\n")
open(dst, 'w').write(s)
print('wrote', dst)

# ---------------------------------------------------------------- (2b) hooks: training loop, evaluation, checkpoint, warm start
s = open(dst).read()
rep('''            mu, q, mask = model.source(sc['own_t'])
            feats, _ = rollout(mu, q, mask, sc['edges'], positions_m=sc['positions_m'])
            total = 0.; parts = {}; cons_val = 0.
''', '''            mu_c, q, mask = model.head(model.trunk(sc['own_t']), sc['own_t']); mu = model.latent_quantize(mu_c)   # == model.source, keeps mu_c
            feats, _ = rollout(mu, q, mask, sc['edges'], positions_m=sc['positions_m'])
            total = 0.; parts = {}; cons_val = 0.; rate_val = 0.; r_fit = None
            if em is not None:
                logp = (q.double() + LATENT_PRIOR / (mu_c.shape[0] * 64)).log()
                r_fit = em.rate(mu_c, logp, mask, detach_inputs=True)          # fits the density only (no encoder gradient)
                rate_val = float(r_fit)
                if args.rate_lambda > 0:
                    total = total + args.rate_lambda * em.rate(mu_c, logp, mask, detach_params=True)
''')
rep('''            total.backward()
''', '''            total.backward()
            if r_fit is not None:
                r_fit.backward()
''')
rep("cons {cons_val:.4f} grad", "cons {cons_val:.4f} rate {rate_val:.3f} grad")
rep('''        rows = []
        for sc in scenes:
            n = sc['private'].shape[0]
            mu, q, mask = model.source(sc['own_t'])
            if refine_steps:''', '''        rows = []; rate_rows = []
        for sc in scenes:
            n = sc['private'].shape[0]
            mu_c, q, mask = model.head(model.trunk(sc['own_t']), sc['own_t']); mu = model.latent_quantize(mu_c)
            if em is not None:
                logp = (q.double() + LATENT_PRIOR / (n * 64)).log()
                b_, c_ = em.code_bits(mu_c, logp, mask); mi_, ci_ = em.hard_codes(mu_c, logp); ob = mask.bool()
                rate_rows.append((b_, c_, mi_[ob].cpu().numpy(), ci_[ob].cpu().numpy(), float(ob.sum(1).double().mean())))
            if refine_steps:''')
rep('''        return dict(rows=rows, summary=summary)
''', '''        if rate_rows:
            def H(x):
                _, c = np.unique(x, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
            mi_all = np.concatenate([r[2] for r in rate_rows]); ci_all = np.concatenate([r[3] for r in rate_rows])
            summary['rate'] = dict(model_bits_per_observed_coordinate=sum(r[0] for r in rate_rows) / max(sum(r[1] for r in rate_rows), 1),
                                   H_mu=H(mi_all), H_q=H(ci_all), H_joint=H(mi_all.astype(np.int64) * 100000 + ci_all.astype(np.int64)),
                                   observed_coordinates_per_agent=float(np.mean([r[4] for r in rate_rows])),
                                   fixed_bits_per_observed_coordinate=MU_BITS + Q_BITS)
        return dict(rows=rows, summary=summary)
''')
rep("meta=meta, weights_sha256=model.weights_hash()),", "meta=meta, weights_sha256=model.weights_hash(), entropy_model=(em.state_dict() if em is not None else None)),")
rep("        model.load_state_dict({**own_state, **loaded}, strict=True)\n",
    "        model.load_state_dict({**own_state, **loaded}, strict=True)\n"
    "        if em is not None and payload.get('entropy_model'):\n"
    "            em.load_state_dict(payload['entropy_model'])\n")
open(dst, 'w').write(s)
print('hooks applied')

# ---------------------------------------------------------------- (2c) review v368 M1: pre-fit the entropy model; constant-lr em optimizer
s = open(dst).read()
rep("    groups = model.parameter_groups()\n",
    "    prefit_report = None\n"
    "    if em is not None and train and not (args.init_from and payload.get('entropy_model')):\n"
    "        # fit the entropy model to the detached codes of the (warm-start) encoder on ALL TRAIN scenes before the rate gradient is on\n"
    "        with torch.no_grad():\n"
    "            M_, L_, K_ = [], [], []\n"
    "            for sc_ in train:\n"
    "                own_ = sc_['own_t'].to(device)\n"
    "                mc_, q_, k_ = model.head(model.trunk(own_), own_)\n"
    "                M_.append(mc_); L_.append((q_.double() + LATENT_PRIOR / (mc_.shape[0] * 64)).log()); K_.append(k_)\n"
    "            M_ = torch.cat(M_); L_ = torch.cat(L_); K_ = torch.cat(K_)\n"
    "        b0_, c0_ = em.code_bits(M_, L_, K_)\n"
    "        pre_opt = torch.optim.Adam(em.parameters(), lr=1e-2)\n"
    "        for _ in range(args.rate_prefit_steps):\n"
    "            pre_opt.zero_grad(set_to_none=True); em.rate(M_, L_, K_, detach_inputs=True).backward(); pre_opt.step()\n"
    "        b1_, c1_ = em.code_bits(M_, L_, K_)\n"
    "        mi_, ci_ = em.hard_codes(M_, L_); ob_ = K_.bool()\n"
    "        def _H(x):\n"
    "            _, c = np.unique(x, return_counts=True); pr = c / c.sum(); return float(-(pr * np.log2(pr)).sum())\n"
    "        prefit_report = dict(steps=args.rate_prefit_steps, observed_coordinates=c1_, bits_per_coord_before=b0_ / max(c0_, 1), bits_per_coord_after=b1_ / max(c1_, 1),\n"
    "                             empirical_H_mu=_H(mi_[ob_].cpu().numpy()), empirical_H_q=_H(ci_[ob_].cpu().numpy()))\n"
    "        print(f'[{args.arm}] entropy-model prefit: {prefit_report}', flush=True)\n"
    "        del M_, L_, K_\n"
    "    groups = model.parameter_groups()\n")
rep("    meta = dict(arm=args.arm,", "    meta = dict(rate_prefit=prefit_report, arm=args.arm,")
rep('''            if r_fit is not None:
                r_fit.backward()
''', '''            if r_fit is not None:
                r_fit.backward(); em_opt.step(); em_opt.zero_grad(set_to_none=True)
''')
open(dst, 'w').write(s)
print('prefit applied')
