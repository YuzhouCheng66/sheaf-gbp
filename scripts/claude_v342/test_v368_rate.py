"""Tests for train_v368 entropy model / rate term. python test_v368_rate.py"""
import math, torch, importlib.util
spec = importlib.util.spec_from_file_location('T', 'train_v368.py'); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
T.MU_BITS = T.Q_BITS = 8; ok = True
def check(name, c, d=''):
    global ok; ok &= bool(c); print('PASS' if c else 'FAIL', name, d)
torch.manual_seed(0)
em = T.EntropyModel()
with torch.no_grad():
    for p in em.parameters(): p.add_(torch.randn_like(p) * .3)
levels, nq, lo, hi, s = em.geometry()
xs = torch.arange(-levels, levels + 1, dtype=torch.float64) / levels
pm = (2 ** -em.mu.bits(xs, 1. / levels, -1., 1.)).sum()
ys = lo + (torch.arange(nq - 1, dtype=torch.float64) + .5) * s
pq = (2 ** -em.lp.bits(ys, s, lo, hi)).sum()
check('mean-code pmf sums to 1 over the 255 symbols', abs(float(pm) - 1) < 1e-9, f'{float(pm):.12f}')
check('precision-code pmf sums to 1 over the 254 reachable symbols', abs(float(pq) - 1) < 1e-9, f'{float(pq):.12f}')
# hard-code roundtrip: code_bits evaluates exactly the bins of the hard codes
mu_c = torch.rand(4, 64) * 2 - 1; q = torch.rand(4, 64) * 3.9 + .1; mask = (torch.rand(4, 64) < .5).float()
logp = (q.double() + 1e-4 / 256).log()
b, c = em.code_bits(mu_c, logp, mask)
check('code_bits finite and positive, counts observed coordinates', math.isfinite(b) and b > 0 and c == float(mask.sum()))
# gradient routing
mu_c.requires_grad_(True); q.requires_grad_(True)
r_fit = em.rate(mu_c, (q.double() + 1e-4 / 256).log(), mask, detach_inputs=True); r_fit.backward()
check('fit term: no gradient into encoder outputs', mu_c.grad is None and q.grad is None)
check('fit term: gradient into entropy-model parameters', all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in em.parameters()))
for p in em.parameters(): p.grad = None
r_enc = em.rate(mu_c, (q.double() + 1e-4 / 256).log(), mask, detach_params=True); r_enc.backward()
check('encoder term: gradient into encoder outputs (observed blocks only)', mu_c.grad is not None and float(mu_c.grad[mask == 0].abs().sum()) == 0 and float(mu_c.grad[mask == 1].abs().sum()) > 0)
check('encoder term: no gradient into entropy-model parameters', all(p.grad is None for p in em.parameters()))
# noisy rate approximates the hard code length on average
with torch.no_grad():
    rn = torch.stack([em.rate(mu_c.detach(), logp, mask) for _ in range(200)]).mean()
check('noisy-rate proxy within 0.5 bit of the hard code length per coordinate', abs(float(rn) - b / c) < .5, f'noisy {float(rn):.3f} vs hard {b / c:.3f}')
print('ALL PASS' if ok else 'SOME FAILED')
