"""Math checks for max-precision selection fusion (v347). python test_v347_max.py --source <sealed>"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_implicit_model_v192 import latent_quantize
from train_v347_maxsel import make_rollout, LATENT_PRIOR
torch.manual_seed(0); ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS' if cond else 'FAIL'), name, detail)
rsum = make_rollout(forest, latent_quantize, 'sum'); rmax = make_rollout(forest, latent_quantize, 'max')
n = 5; prior = LATENT_PRIOR / (n * 64)
mu = latent_quantize(torch.rand(n, 64) * 2 - 1); q = torch.rand(n, 64) * 3.9 + .1; mask = torch.ones(n, 64)
# (1) K0 identical between sum and max
fs, _ = rsum(mu, q, mask, [], rounds=0, stages=(0,)); fm, _ = rmax(mu, q, mask, [], rounds=0, stages=(0,))
check('K0: max == sum (own factor only)', torch.equal(fs[0], fm[0]))
# (2) complementary owned sources on a path: exact union, exactly (no averaging), through the quantised wire
mask = torch.zeros(n, 64); mask[0, :32] = 1; mask[1, 32:] = 1
edges = [(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 4, 1.)]
fm, _ = rmax(mu, q, mask, edges, rounds=8, stages=(1, 2, 4, 8))
union = torch.cat((mu[0, :32], mu[1, 32:]))
err = (fm[8][:, :64] - union[None]).abs().max()
check('complementary owned: every node R8 mean == union up to the profile-4 wire quantiser (identical error for sum and max: 1.57e-2 here)', err <= 3 / 127 + 1e-9, f'max err {float(err):.2e}')
check('max: R4 == R8 (idempotent beyond diameter)', torch.equal(fm[4], fm[8]))
# (3) overlapping block: the more precise source wins entirely (selection), sum would blend
mask = torch.zeros(n, 64); mask[0] = 1; mask[1] = 1
q2 = q.clone(); q2[0] = 3.0; q2[1] = 1.0
fm, _ = rmax(mu, q2, mask, [(0, 1, 1.)], rounds=1, stages=(1,)); fs, _ = rsum(mu, q2, mask, [(0, 1, 1.)], rounds=1, stages=(1,))
check('overlap: max picks source 0 everywhere at node 1', (fm[1][1, :64] - mu[0]).abs().max() <= 2 / 127 + 1e-9, f'{float((fm[1][1,:64]-mu[0]).abs().max()):.2e}')
check('overlap: sum blends (differs from both sources)', ((fs[1][1, :64] - mu[0]).abs().max() > .05) and ((fs[1][1, :64] - mu[1]).abs().max() > .05))
check('overlap: max belief variance == 1/(q_winner+prior) within one precision-codebook step (9.6%)', torch.allclose(fm[1][1, 64:], (-(torch.tensor(3.0, dtype=torch.float64) + prior).log()).float().expand(64), atol=0.1))
# (4) CYCLE exactness: loopy max on a triangle + chord graph == centralized max over the reached set; sum would overcount
mask = torch.zeros(n, 64); mask[0, :20] = 1; mask[1, 20:40] = 1; mask[2, 40:] = 1; mask[3, :10] = 1; mask[4, 50:] = 1
qc = torch.rand(n, 64) * 3.9 + .1
cyc = [(0, 1, 1.), (1, 2, 1.), (2, 0, 1.), (2, 3, 1.), (3, 4, 1.), (4, 2, 1.)]
fm, _ = rmax(mu, qc, mask, cyc, rounds=8, stages=(8,), allow_cycles=True)
evid = mask.double() * qc.double() + prior
pw, ew, _ = forest.quantize(evid, mask.double() * qc.double() * mu.double())      # what the wire carries from each source
P = pw.max(0).values; idx = pw.argmax(0)
central_mean = ew.gather(0, idx[None])[0] / P
# nodes read their OWN factor unquantised, so compare only coordinates whose winner is not the node itself
own_wins = torch.stack([(idx == j) for j in range(n)])
dev = (fm[8][:, :64].double() - latent_quantize(central_mean.float())[None].double()).abs()
check('cycles: loopy max belief == centralized max over all (quantised) sources, relayed coordinates exact', float(dev[~own_wins].max()) <= 1e-9, f'max {float(dev[~own_wins].max()):.2e}')
check('cycles: loopy max belief variance == 1/max precision (no overcounting) within one codebook step', torch.allclose(fm[8][:, 64:], (-(P).log()).float()[None].expand(n, -1), atol=0.1))
# (5) duplicate / stale resend: running more rounds never changes anything (join-semilattice)
fm2, _ = rmax(mu, qc, mask, cyc, rounds=16, stages=(16,), allow_cycles=True)
check('cycles: 16 rounds == 8 rounds (idempotent under repeated flooding)', torch.equal(fm2[16], fm[8]))
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
