"""Reviewer's independent check of E3-L GaBP (D8-54). Own numpy implementation (recursive messages, no schedule borrowed from
g4_e3l / G.forest); compared with (a) the dense closed form (diag Lam + w L)^-1 diag Lam z, (b) the implementation's gabp() outputs,
(c) the quantised integer wire, (d) w -> inf weighted mean and the E3-R special case, (e) STE gradient flow.
Run: CUDA_VISIBLE_DEVICES= python -B indep_gabp_check.py"""
import os, sys, json, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np

rng = np.random.default_rng(777)


# ------------------------------------------------------------ own implementation (no import from the reviewed script)
def rand_tree(m):
    """random labelled tree via a random Pruefer-like attachment with shuffled labels"""
    perm = rng.permutation(m)
    ed = []
    for k in range(1, m):
        ed.append((int(perm[k]), int(perm[rng.integers(k)])))
    return ed


def nbrs(ed, m):
    nb = {a: [] for a in range(m)}
    for i, j in ed:
        nb[i].append(j); nb[j].append(i)
    return nb


def my_msgs(h, lam, ed, w, quant=False):
    """messages by plain recursion (memoised): msg(j->i) = f(j's potential + all msgs into j except from i)"""
    m = lam.shape[0]; nb = nbrs(ed, m); memo = {}

    def q(hh, ll):
        if not quant:
            return hh, ll
        lq = np.round(ll); hq = np.round(hh) * (lq > 0)[None]
        return hq, lq

    def msg(j, i):
        if (j, i) in memo:
            return memo[(j, i)]
        lt = lam[j].copy(); ht = h[j].copy()
        for k in nb[j]:
            if k != i:
                hk, lk = msg(k, j); lt = lt + lk; ht = ht + hk
        f = w / (w + lt)
        memo[(j, i)] = q(ht * f[None], lt * f)
        return memo[(j, i)]
    hb = np.zeros_like(h); lb = np.zeros_like(lam)
    for a in range(m):
        lb[a] = lam[a]; hb[a] = h[a]
        for k in nb[a]:
            hk, lk = msg(k, a); lb[a] = lb[a] + lk; hb[a] = hb[a] + hk
    return hb, lb, memo


def my_closed(z, lam, ed, w):
    """dense per-cell solve (diag Lam + w L) x = Lam z with explicit Laplacian; z [m, D, C], lam [m, C]"""
    m, D, C = z.shape
    Lp = np.zeros((m, m))
    for i, j in ed:
        Lp[i, i] += 1; Lp[j, j] += 1; Lp[i, j] -= 1; Lp[j, i] -= 1
    x = np.full_like(z, np.nan)
    for c in range(C):
        if lam[:, c].sum() <= 0:
            continue
        A = np.diag(lam[:, c]) + w * Lp
        x[:, :, c] = np.linalg.solve(A, lam[:, c, None] * z[:, :, c])
    return x


def my_energy(x, z, lam, ed, w):
    e = 0.5 * np.sum(lam[:, None] * (x - z) ** 2)
    for i, j in ed:
        e += 0.5 * w * np.sum((x[i] - x[j]) ** 2)
    return e


out = {}
# ------------------------------------------------------------ (a) own GaBP == closed form on random trees
errs = []; emins = []
for t in range(300):
    m = int(rng.integers(1, 10)); D = int(rng.integers(1, 9)); C = 40
    ed = rand_tree(m)
    lam = rng.gamma(1.2, 2.0, (m, C)) * (rng.random((m, C)) > 0.4)
    z = rng.normal(0, 3, (m, D, C))
    w = float(rng.choice([1e-2, 0.3, 1.0, 7.0, 1e3, 1e6]))
    hb, lb, _ = my_msgs(lam[:, None] * z, lam, ed, w)
    pos = lb > 0
    x = np.where(pos[:, None], hb / np.where(pos, lb, 1)[:, None], 0.0)
    xc = my_closed(z, lam, ed, w)
    inf = lam.sum(0) > 0
    if inf.any():
        errs.append(float(np.abs(x[:, :, inf] - xc[:, :, inf]).max() / max(1.0, np.abs(xc[:, :, inf]).max())))
    # the GaBP point is the energy minimum: random perturbations never go lower (cells with info)
    zz = z[:, :, inf]; ll = lam[:, inf]; xx = x[:, :, inf]
    e0 = my_energy(xx, zz, ll, ed, w)
    for _ in range(3):
        emins.append(my_energy(xx + rng.normal(0, 1e-2, xx.shape), zz, ll, ed, w) - e0)
out['own_vs_closed_rel_max'] = max(errs)
out['perturbation_energy_min_delta'] = min(emins)

# ------------------------------------------------------------ (b) the implementation's gabp() == own (float and integer wire)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'scripts', 'claude_v342')))
import torch
import g4_e3l as L          # only to obtain the IMPLEMENTATION's outputs for comparison
G = L.G
d_float = []; d_int = 0; n_int = 0; cf_impl = []; viol = 0; steps = 0; belief_recompute_ok = True
for t in range(200):
    m = int(rng.integers(1, 9)); D = 8; g = (4, 4, 2); C = 32
    ed = rand_tree(m)
    Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
    lam = rng.gamma(1.2, 2.0, (m,) + g) * (rng.random((m,) + g) > 0.4)
    z = rng.normal(0, 3, (m, D) + g)
    w = float(rng.choice([0.0537, 0.517, 2.03, 31.7, 1003.3]))   # non-dyadic w: no exact .5 ties (see note)
    hT, lT = torch.from_numpy(lam[:, None] * z), torch.from_numpy(lam)
    hb_i, lb_i, msgs_i = L.gabp(hT, lT, Fl, w, None)
    hb_o, lb_o, msgs_o = my_msgs((lam[:, None] * z).reshape(m, D, C), lam.reshape(m, C), ed, w)
    d_float.append(max(float(np.abs(hb_i.numpy().reshape(m, D, C) - hb_o).max()), float(np.abs(lb_i.numpy().reshape(m, C) - lb_o).max())))
    assert set(msgs_i) == set(msgs_o) and len(msgs_i) == 2 * (m - 1)
    # integer wire: source potentials rounded, every hop rounded, Lam_q = 0 => h_q = 0
    lq0 = np.round(lam); hq0 = np.round(lam[:, None] * z) * (lq0 > 0)[:, None]
    hb_qi, lb_qi, mq_i = L.gabp(torch.from_numpy(hq0), torch.from_numpy(lq0), Fl, w, 'round')
    hb_qo, lb_qo, mq_o = my_msgs(hq0.reshape(m, D, C), lq0.reshape(m, C), ed, w, quant=True)
    d_int += int((hb_qi.numpy().reshape(m, D, C) != hb_qo).sum() + (lb_qi.numpy().reshape(m, C) != lb_qo).sum()); n_int += hb_qo.size + lb_qo.size
    # receiver recomputation: belief = own quantised potential + received integer messages, nothing else
    for a in range(m):
        hh = hq0[a].reshape(D, C).copy(); ll = lq0[a].reshape(C).copy()
        for (s, r), (hm, lm) in mq_i.items():
            if r == a:
                hi = hm.numpy().reshape(D, C); li = lm.numpy().reshape(C)
                assert np.array_equal(hi, np.round(hi)) and np.array_equal(li, np.round(li)) and not hi[:, li == 0].any()
                hh += hi; ll += li
        belief_recompute_ok &= bool(np.array_equal(hh, hb_qi[a].numpy().reshape(D, C)) and np.array_equal(ll, lb_qi[a].numpy().reshape(C)))
    c = L.gbp_checks(torch.from_numpy(z), torch.from_numpy(lam), Fl, w)
    cf_impl.append(c['cf_max']); viol += c['violations']; steps += c['steps']
out['impl_vs_own_float_max'] = max(d_float)
out['impl_vs_own_int_wire_mismatches'] = f'{d_int} / {n_int}'
out['receiver_recompute_ok'] = belief_recompute_ok
out['impl_gbp_checks_cf_max'] = max(cf_impl); out['impl_gbp_checks_violations'] = viol; out['impl_gbp_checks_steps'] = steps

# ------------------------------------------------------------ (c) monotone sequence: does the implementation's check detect a broken GaBP?
m = 5; ed = [(0, 1), (1, 2), (1, 3), (3, 4)]; Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
lam = rng.gamma(1.2, 2.0, (m, 32)); z = rng.normal(0, 3, (m, 8, 32)); w = 2.0
hb, lb, _ = my_msgs(lam[:, None] * z, lam, ed, w)
x = hb / lb[:, None]
x_bad = x.copy(); x_bad[2] += 0.5                                         # a wrong belief for agent 2
en_ok, _ = L.descent_energies(z, lam, Fl, w, x, lb)
en_bad, _ = L.descent_energies(z, lam, Fl, w, x_bad, lb)
tol = 1e-9 * max(1, en_ok[0])
out['descent_ok_violations'] = int(sum(b > a + tol for a, b in zip(en_ok[:-1], en_ok[1:])))
out['descent_detects_wrong_belief'] = int(sum(b > a + tol for a, b in zip(en_bad[:-1], en_bad[1:])))
out['descent_final_vs_closed'] = float(abs(en_ok[-1] - my_energy(my_closed(z, lam, ed, w), z, lam, ed, w)))

# ------------------------------------------------------------ (d) w -> inf: information-weighted mean; E3-R special case on the integer wire
m = 6; ed = rand_tree(m); Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
cnt = (rng.random((m, 8, 8, 4)) < 0.5).astype(np.float64)                     # Lam = count = 1[touched]
e = rng.integers(-120, 150, (m, 1, 8, 8, 4)).astype(np.float64) * cnt[:, None]  # z = calibrated evidence (D = 1)
hb_q, lb_q, mq = L.gabp(torch.from_numpy(cnt[:, None] * e), torch.from_numpy(cnt), Fl, 1e15, 'round')
S = e.sum(0)[0]; K = cnt.sum(0)
ok = all(np.array_equal(hb_q[a, 0].numpy(), S) and np.array_equal(lb_q[a].numpy(), K) for a in range(m))
nb = nbrs(ed, m)
def side(s, r):
    seen = {s}; st = [s]
    while st:
        u = st.pop()
        for v in nb[u]:
            if v != r and v not in seen:
                seen.add(v); st.append(v)
    return seen
ok_msgs = all(np.array_equal(lm.numpy(), sum(cnt[b] for b in side(s, r))) and np.array_equal(hm[0].numpy(), sum(e[b, 0] for b in side(s, r)))
              for (s, r), (hm, lm) in mq.items())
out['e3r_special_case_beliefs_exact'] = bool(ok); out['e3r_special_case_messages_exact'] = bool(ok_msgs)
# the actual init w0 = 1000 with Lam ~ 2: how far from the E3-R limit is a 2-hop message? (relative shrink of Lam per hop)
out['init_shrink_per_hop_Lam2'] = 1 - 1000 / (1000 + 2.0)
# the smallest w at which the 1-rounded wire still reproduces S, k exactly on this case
for wv in (1e3, 1e4, 1e5, 1e6):
    hq_, lq_, _ = L.gabp(torch.from_numpy(cnt[:, None] * e), torch.from_numpy(cnt), Fl, wv, 'round')
    out[f'e3r_exact_at_w_{wv:g}'] = bool(all(np.array_equal(hq_[a, 0].numpy(), S) and np.array_equal(lq_[a].numpy(), K) for a in range(m)))

# ------------------------------------------------------------ (e) gradients: through the STE wire and GaBP into OTHER agents' potentials and w
m = 4; ed = [(0, 1), (1, 2), (2, 3)]; Fl = G.forest([(i, j, 1.0) for i, j in ed], m)
lam = torch.tensor(rng.gamma(2.0, 2.0, (m, 4, 4, 2)) + 1.0, requires_grad=True)
zt = torch.tensor(rng.normal(0, 5, (m, 8, 4, 4, 2)), requires_grad=True)
logw = torch.tensor(math.log(3.0), requires_grad=True)
h = lam.unsqueeze(1) * zt
hq, lq = L.quantize(h, lam, 'ste')
hb, lb, _ = L.gabp(hq, lq, Fl, torch.exp(logw), 'ste')
x0 = L.belief_mean(hb, lb)[0]                                               # agent 0 decodes
(x0.sum() + torch.log1p(lb[0]).sum()).backward()
out['grad_into_agent3_z_nonzero'] = bool(zt.grad[3].abs().sum() > 0)
out['grad_into_agent3_lam_nonzero'] = bool(lam.grad[3].abs().sum() > 0)
out['grad_into_w_nonzero'] = bool(logw.grad is not None and abs(float(logw.grad)) > 0)
# straight-through floor: a = -50 -> forward softplus(-8), backward sigmoid(-8)
a = torch.tensor([-50.0], requires_grad=True)
af = a + (a.clamp(min=L.A_FLOOR) - a).detach()
y = torch.nn.functional.softplus(af); y.backward()
out['floor_forward'] = float(y); out['floor_grad'] = float(a.grad)

print(json.dumps(out, indent=1))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indep_gabp_check.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
