"""Reviewer probe (E3-L4 review, check 2): does per-hop rounding at finite w make distant known information vanish?
Uses the frozen L.gabp / L.quantize / L.gbp_checks (g4_e3l.py) exactly as E3-L3 / E3-L4 call them. CPU, synthetic only.
Cases: long chains observed only at one end (m up to 400), stars observed at one leaf (up to 200 leaves), random deep trees with sparse
footprints; w in {3, 4, 6, 8} plus the boundary values {1.0, 1.0001, 1.5, 2.0}; 'round' (float64 and float32) and 'ste' (float32)."""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342')))
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l as L  # noqa: E402

G = L.G
torch.set_num_threads(1)
rng = np.random.default_rng(20261028)


def chain(m):
    return G.forest([(i, i + 1, 1.0) for i in range(m - 1)], m)


def star(m):
    return G.forest([(0, i, 1.0) for i in range(1, m)], m)


def rand_tree(m):
    return G.forest([(int(rng.integers(i)), i, 1.0) for i in range(1, m)], m)


def run(Fl, lam, z, w, mode, dtype):
    lam_t = torch.as_tensor(lam, dtype=dtype); z_t = torch.as_tensor(z, dtype=dtype)
    h = (lam_t * z_t).unsqueeze(1)
    hb, lb, msgs = L.gabp(h, lam_t, Fl, torch.tensor(w, dtype=dtype) if dtype == torch.float32 else w, mode)
    k = lam_t.sum(0)
    mism = int(sum(int(((lb[a] > 0) != (k > 0)).sum()) for a in range(lam.shape[0])))
    lmax = max(float(m[1].max()) for m in msgs.values()) if msgs else 0.0
    hmax = max(float(m[0].abs().max()) for m in msgs.values()) if msgs else 0.0
    lmin_pos = min((float(m[1][m[1] > 0].min()) for m in msgs.values() if (m[1] > 0).any()), default=None)
    # messages that carry Lam >= 1 but h rounded to 0 although the float h was != 0 (the mean is lost, the known mask is not)
    return dict(mism=mism, lam_msg_max=lmax, h_msg_absmax=hmax, lam_msg_min_positive=lmin_pos), (hb, lb)


out = dict(cases=[])
W_ALL = (1.0, 1.0001, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
for w in W_ALL:
    agg = dict(w=w, mism_round64=0, mism_round32=0, mism_ste32=0, cases=0, lam_msg_max=0.0, h_msg_absmax=0.0, cf_max=0.0, violations=0,
               far_end_lb_chain400=None, far_end_x_chain400_q=None, far_end_x_chain400_float=None, quant_mean_absdiff_max=0.0)
    cases = []
    for m in (2, 3, 10, 50, 200, 400):                                          # chains: only node 0 observes, z = +1
        lam = np.zeros((m, 2, 2, 1)); lam[0] = 1.0; z = np.ones_like(lam); z[0, 1] = -1; z[0, 0, 1] = 0
        cases.append((chain(m), lam, z, f'chain{m}'))
    for m in (3, 20, 200):                                                       # stars: only the last leaf observes
        lam = np.zeros((m, 2, 2, 1)); lam[-1] = 1.0; z = np.ones_like(lam)
        cases.append((star(m), lam, z, f'star{m}'))
        lam = np.zeros((m, 2, 2, 1)); lam[1:] = 1.0; z = rng.integers(-1, 2, lam.shape).astype(float) * lam   # every leaf observes, centre not
        cases.append((star(m), lam, z, f'star{m}_allleaves'))
    for t in range(40):                                                          # random deep trees, sparse footprints
        m = int(rng.integers(2, 120))
        lam = (rng.random((m, 3, 3, 2)) < rng.choice([0.01, 0.05, 0.3])).astype(float)
        z = rng.integers(-1, 2, lam.shape).astype(float) * lam
        cases.append((rand_tree(m), lam, z, f'rand{t}_m{m}'))
    for Fl, lam, z, name in cases:
        r64, (hb, lb) = run(Fl, lam, z, w, 'round', torch.float64)
        r32, _ = run(Fl, lam, z, w, 'round', torch.float32)
        rs, _ = run(Fl, lam, z, w, 'ste', torch.float32)
        agg['mism_round64'] += r64['mism']; agg['mism_round32'] += r32['mism']; agg['mism_ste32'] += rs['mism']; agg['cases'] += 1
        agg['lam_msg_max'] = max(agg['lam_msg_max'], r64['lam_msg_max']); agg['h_msg_absmax'] = max(agg['h_msg_absmax'], r64['h_msg_absmax'])
        if lam.shape[0] <= 200:
            c = L.gbp_checks(torch.as_tensor(z).unsqueeze(1), torch.as_tensor(lam), Fl, w)
            agg['cf_max'] = max(agg['cf_max'], c['cf_max']); agg['violations'] += c['violations']
        hf, lf, _ = L.gabp((torch.as_tensor(lam) * torch.as_tensor(z)).unsqueeze(1), torch.as_tensor(lam), Fl, w, None)
        xq = L.belief_mean(hb, lb); xf = L.belief_mean(hf, lf); s = lb > 0
        if s.any():
            agg['quant_mean_absdiff_max'] = max(agg['quant_mean_absdiff_max'], float((xq - xf).abs().amax(1)[s].max()))
        if name == 'chain400':
            agg['far_end_lb_chain400'] = float(lb[-1, 0, 0, 0]); agg['far_end_x_chain400_q'] = float(xq[-1, 0, 0, 0, 0])
            agg['far_end_x_chain400_float'] = float(xf[-1, 0, 0, 0, 0]); agg['far_end_lb_chain400_float'] = float(lf[-1, 0, 0, 0])
    out['cases'].append(agg)
    print(json.dumps(agg), flush=True)
# the message map Lam_msg = round(w L / (w + L)) for integer L = 0..20 at each w (the induction step)
out['lam_map'] = {str(w): [float(torch.round(torch.tensor(w * l / (w + l)))) for l in range(0, 21)] for w in W_ALL}
for w in W_ALL:
    print(w, out['lam_map'][str(w)], flush=True)
json.dump(out, open(os.path.join(HERE, 'gabp_finite_w_probe.json'), 'w'), indent=1)
