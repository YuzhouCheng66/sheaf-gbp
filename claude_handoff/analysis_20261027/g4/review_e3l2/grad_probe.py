"""review E3-L2 (reviewer, CPU only): instrumented replica of g4_e3l2.train on the TRAIN smoke cache.
Reads g4_e3l2 / g4_e3l read only; variants patch functions in this process only (the reviewed files are not changed).
  base   : g4_e3l2 as is
  upper1s: the minimal fix candidate -- Lam above LAMQ_MAX and |h| above HQ_LIM: one-sided straight-through (only the pull-back gradient passes)
  noclamp: K = 10**6 and clamp=False (= E3-L's forward without the entropy model), to separate top-K / clamp from the rest
usage: python -B grad_probe.py VARIANT UPDATES OUT.json [WARMUP] [BATCH]"""
import os, sys, json, math, time
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import numpy as np
import torch
import g4_e3l2 as M
L = M.L; R = M.R

variant, updates, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 4
batch = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(2)
cache = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4', 'e3l_smoke', 'cache')
spec = 'c:' + os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs', 'e1c_fit.json').replace('\\', '/') + ':A:cam_nf'
seed = 342101


class _OneSidedClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x); ctx.lo = lo; ctx.hi = hi
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = torch.where(x > ctx.hi, g.clamp(min=0.0), g)          # above: only the gradient that lowers x passes
        if ctx.lo is not None:
            g = torch.where(x < ctx.lo, g.clamp(max=0.0), g)      # below: only the gradient that raises x passes
        return g, None, None


orig_source = M.source_topk


def source_upper1s(z, lam, K, qmode, clamp=True):
    if qmode != 'ste' or not clamp:
        return orig_source(z, lam, K, qmode, clamp)
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    lam_c = torch.where(sel, _OneSidedClamp.apply(lam, None, float(M.LAMQ_MAX) + 0.5), lam)
    h = lam_c.unsqueeze(1) * z
    h_c = _OneSidedClamp.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5)
    lr = torch.round(lam).clamp(M.LAMQ_MIN, M.LAMQ_MAX); hr = torch.round(h).clamp(-M.HQ_LIM, M.HQ_LIM)
    lq = lr * s; hq = hr * s.unsqueeze(1)
    hg = h_c * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


def source_both1s(z, lam, K, qmode, clamp=True):
    """the proposed minimal fix (M1): forward identical to g4_e3l2.source_topk; backward one-sided straight-through at every clamp bound:
    selected Lam below 0.5 (forward 1) only the raising gradient passes, above 15.5 (forward 15) only the lowering one; |h| above 7.5
    (forward +-7) only the gradient that shrinks |h| passes. Inside the bounds the plain identity of the reviewed code."""
    if qmode != 'ste' or not clamp:
        return orig_source(z, lam, K, qmode, clamp)
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    h = lam.unsqueeze(1) * z
    lr = torch.round(lam).clamp(M.LAMQ_MIN, M.LAMQ_MAX); hr = torch.round(h).clamp(-M.HQ_LIM, M.HQ_LIM)
    lq = lr * s; hq = hr * s.unsqueeze(1)
    lam_c = torch.where(sel, _OneSidedClamp.apply(lam, M.LAMQ_MIN - 0.5, M.LAMQ_MAX + 0.5), lam)
    h_c = _OneSidedClamp.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5)
    hg = h_c * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


def source_cons1s(z, lam, K, qmode, clamp=True):
    """alternative fix (A'): on the selected cells the continuous Lam is clamped to [1, 15] FIRST (one-sided straight-through), and
    h = Lam_c z is built from the clamped Lam (so h_q / Lam_q ~ z whatever the raw Lam); h_q clamp +-7 one-sided as in both1s.
    Forward differs from g4_e3l2 only on selected cells with raw Lam < 1 or > 15."""
    if not clamp:
        return orig_source(z, lam, K, qmode, clamp)
    sel = M.topk_select(lam, K); s = sel.to(lam.dtype)
    lam_c = torch.where(sel, _OneSidedClamp.apply(lam, float(M.LAMQ_MIN), float(M.LAMQ_MAX)), lam)
    h = lam_c.unsqueeze(1) * z
    if qmode is None:
        return h * s.unsqueeze(1), lam_c * s, sel
    lq = torch.round(lam_c).clamp(M.LAMQ_MIN, M.LAMQ_MAX) * s; hq = torch.round(h).clamp(-M.HQ_LIM, M.HQ_LIM) * s.unsqueeze(1)
    if qmode == 'round':
        return hq, lq, sel
    hg = _OneSidedClamp.apply(h, -M.HQ_LIM - 0.5, M.HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


STORE = {}
orig_enc = M.encode_fixed


def enc_spy(model, f, kn, chunk=M.ENC_CHUNK):
    z, lam, P = orig_enc(model, f, kn, chunk)
    if z.requires_grad:
        z.retain_grad(); lam.retain_grad()
    STORE['z'] = z; STORE['lam'] = lam
    return z, lam, P


M.encode_fixed = enc_spy
if variant == 'upper1s':
    M.source_topk = source_upper1s
if variant == 'both1s':
    M.source_topk = source_both1s
if variant == 'cons1s':
    M.source_topk = source_cons1s
sample_spec = M.sample_spec
if variant == 'noclamp':
    def sample_spec(rng, houses, p_clean=R.P_CLEAN):
        s = M.sample_spec(rng, houses, p_clean); return s[:5] + (10 ** 6,)

man = json.load(open(os.path.join(cache, 'manifest.json')))
ev = R.Evidence.from_spec(spec, allow_incomplete=True)
dev = torch.device('cpu')
tr = L.load_cache(cache, man, 'train', dev); ho = L.load_cache(cache, man, 'holdout', dev)
table_t = torch.as_tensor(ev.table, dtype=torch.int32)
torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
model = M.E3L2(L.D_LAT, 16, L.W0)
L.probe_check(model, tr, table_t, np.random.default_rng([seed, 5]))
convs = [q for _, q in model.named_parameters() if q.ndim > 1]; rest = [q for _, q in model.named_parameters() if q.ndim <= 1]
opt = torch.optim.AdamW([dict(params=convs, weight_decay=1e-4), dict(params=rest, weight_decay=0.0)], lr=1e-3)
rng = np.random.default_rng([seed, 3])
rows = []
t0 = time.time()
for u in range(updates):
    for g in opt.param_groups:
        g['lr'] = R.lr_at(u, updates, 1e-3, warmup)
    specs = [sample_spec(rng, tr) for _ in range(batch)]
    lg, yk, yo, src = M.forward_batch(model, specs, tr, table_t, 'ste', clamp=(variant != 'noclamp'))
    lk, lo = L.bce_terms(lg, yk, yo)
    loss = lk + lo
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = {k: float(torch.sqrt(sum((p.grad.double() ** 2).sum() for n, p in model.named_parameters() if n.startswith(k) and p.grad is not None)))
          for k in ('enc.net', 'enc.head', 'dec', 'log_w')}
    tot = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    z, lam = STORE['z'], STORE['lam']; sel = src['sel']
    with torch.no_grad():
        ls = lam[sel]; h = (lam.unsqueeze(1) * z).permute(0, 2, 3, 4, 1)[sel]
        lam_t = lam[lam > 0]
        r = dict(u=u + 1, loss=float(loss), known=float(lk), occ=float(lo), grad_total=tot, **{f'g_{k}': v for k, v in gn.items()},
                 g_z=float(z.grad.norm()), g_lam=float(lam.grad.norm()), g_lam_sel=float(lam.grad[sel].norm()), g_lam_unsel=float(lam.grad[(lam > 0) & ~sel].norm()),
                 lam_sel_med=float(ls.median()) if ls.numel() else None, lam_sel_max=float(ls.max()) if ls.numel() else None,
                 lam_sel_frac_hi=float((ls > M.LAMQ_MAX + 0.5).float().mean()) if ls.numel() else None,
                 lam_sel_frac_lo=float((ls < 0.5).float().mean()) if ls.numel() else None,
                 lam_touched_max=float(lam_t.max()), lam_touched_med=float(lam_t.median()),
                 h_sel_frac_clip=float((h.abs() > M.HQ_LIM + 0.5).float().mean()) if h.numel() else None, h_sel_absmax=float(h.abs().max()) if h.numel() else None,
                 z_absmax=float(z.abs().max()), sel_cells=float(sel.flatten(1).sum(1).float().mean()), touched=float((lam > 0).flatten(1).sum(1).float().mean()),
                 K=src['K'], lr=opt.param_groups[0]['lr'], s=time.time() - t0)
    opt.step()
    rows.append(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'K'}), flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
hk, hoc = M.holdout_bce(model, [M.sample_spec(np.random.default_rng([20261027, 9]), ho) for _ in range(4)], ho, table_t, batch)
json.dump(dict(rows=rows, holdout_known=hk, holdout_occ=hoc, variant=variant), open(out, 'w'), indent=1)
print('holdout', hk, hoc, flush=True)
