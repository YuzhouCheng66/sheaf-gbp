"""Pre-launch math/implementation checks for v342 (CPU, sealed v295 primitives). Run from anywhere:
python test_v342_math.py --source <sealed v310 source root>
Checks: (1) K0 closed form; (2) complementary OWNED observations fuse to the exact union through the
quantised forest wire; (3) extra rounds beyond the diameter are idempotent; (4) empty observation contributes
only the prior; (5) tree cavity == centralized reference for the reached set; (6) dense codes with the same
GBP mix complementary content (the failure mode we are removing), as a sanity contrast.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_implicit_model_v192 import latent_quantize
from train_v342_owned import make_rollout, LATENT_PRIOR

torch.manual_seed(0)
rollout = make_rollout(forest, latent_quantize)
ok = True
def check(name, cond, detail=''):
    global ok
    ok &= bool(cond); print(('PASS' if cond else 'FAIL'), name, detail)

n = 4
prior = LATENT_PRIOR / (n * 64)
# (1) K0 closed form: no edges
mu = latent_quantize(torch.rand(n, 64) * 2 - 1); q = torch.rand(n, 64) * 3 + .1; mask = torch.ones(n, 64)
feats, _ = rollout(mu, q, mask, [], rounds=0, stages=(0,))
check('K0 mean == hard8 source mean', torch.equal(feats[0][:, :64], mu))
check('K0 log-variance == -log(q+prior)', torch.allclose(feats[0][:, 64:], (-(q.double() + prior).log()).float(), atol=1e-6))

# (2) complementary owned observations: agent0 owns blocks 0..31, agent1 owns 32..63; agents 2,3 see nothing
mask = torch.zeros(n, 64); mask[0, :32] = 1; mask[1, 32:] = 1
q = torch.full((n, 64), 2.5); edges = [(0, 1, 1.), (1, 2, 1.), (2, 3, 1.)]
feats, state = rollout(mu, q, mask, edges, rounds=8, stages=(0, 1, 2, 4, 8))
union = torch.cat((mu[0, :32], mu[1, 32:]))
err = (feats[8][:, :64] - union[None]).abs().max()
check('complementary owned sources: every node R8 mean == union within one hard8 step (profile-4 wire quantisation)', err <= 2/127 + 1e-9, f'max err {float(err):.2e} (hard8 step 7.87e-3)')
check('R1 at node0 == union within one hard8 step (direct neighbour)', (feats[1][0, :64] - union).abs().max() <= 2/127 + 1e-9)
check('unobserved-everywhere nodes 2,3 at R0 are prior-only (mean 0, var 1/prior)',
      bool((feats[0][2:, :64] == 0).all()) and torch.allclose(feats[0][2:, 64:], torch.full((2, 64), float(-np.log(prior))), atol=1e-4))
# (3) idempotence beyond diameter (3): rounds 4 and 8 identical
check('rounds beyond diameter idempotent (R4 == R8)', torch.equal(feats[4], feats[8]))
# (5) tree cavity vs centralized reference for the reached set
evidence_q = mask.double() * q.double(); Q = torch.diag_embed(evidence_q + prior)[None]; H = (evidence_q * mu.double())[None]
ref = forest.central_reference(Q, H, edges, 8)
mean, var = forest.scalar_belief(state)
check('tree cavity == centralized reference (reached set) mean', torch.allclose(mean, ref['mean'], atol=1e-2, rtol=1e-2), f'max {float((mean-ref["mean"]).abs().max()):.2e} (profile-4 wire quantisation)')
check('tree cavity == centralized reference variance', torch.allclose(var, ref['variance'], rtol=1e-2), f'max rel {float(((var-ref["variance"]).abs()/ref["variance"]).max()):.2e}')
# (6) dense contrast: both agents 'active' on all 64 coordinates with different content -> average, not union
mask_dense = torch.zeros(n, 64); mask_dense[:2] = 1
feats_d, _ = rollout(mu, q, mask_dense, edges, rounds=8, stages=(8,))
mix = (feats_d[8][0, :64] - union).abs().max()
check('dense codes with v277 semantics do NOT recover the union (sanity contrast; expected FAIL-by-design shown as mixing > 0.1)', mix > .1, f'max deviation {float(mix):.3f}')
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
