"""Equivalence / sanity tests for train_v392.py (public-frame conditioning). Claude, 2026-10-24, written before batch V.
1. --frame-cond off: bit-identical to v386 (state_dict, decoder outputs) at W = 1 and W = 2.
2. --frame-cond on (W = 2, the batch-V arm): every v386 tensor is bit-identical (the widened convs keep the original columns and bias),
   the new columns are zero, the decoder FUNCTION at init equals v386 exactly for CHART_M in {8, 14, 20}; parameter delta = 3*9*(lift + sum widths).
3. public_extra: u, v in [-0.5, 0.5) with exactly 8 phase cycles across the raster, L plane = CHART_M / 20, identical for all agents.
4. Gradients reach the new columns once CHART_M planes are non-zero (training can use them).
Runs on CPU; needs only torch/numpy.   python test_v392_frame.py
"""
import importlib.util
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


v386, v392 = load('train_v386'), load('train_v392')
for m in (v386, v392):
    m.READOUT_MODE = 'posterior_only'; m.PRECISION_MODE = 'learned'
SIZE, SEED = 128, 342101
private = torch.from_numpy(np.random.default_rng(0).integers(-1, 2, size=(4, SIZE, SIZE)).astype(np.int8))
for W in (1.0, 2.0):
    v386.DECODER_WIDTH = W; v392.DECODER_WIDTH = W; v392.FRAME_COND = False
    a = v386.SheafModel('owned', SIZE, SEED).eval(); b = v392.SheafModel('owned', SIZE, SEED).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), f'off-flag differs from v386 at W={W}'
    with torch.no_grad():
        mu, q, mask = a.source(private); feats = torch.cat((v386.SheafModel.latent_quantize(mu), (1. / (mask * q + 1e-6)).log()), -1)
        oa = a.decoder(torch.cat((feats, feats), -1)); ob = b.decoder(torch.cat((feats, feats), -1))
    assert torch.equal(oa[0], ob[0]) and torch.equal(oa[1], ob[1])
print('1. --frame-cond off: bit-identical to v386 at W=1 and W=2')

v386.DECODER_WIDTH = 2.0; v392.DECODER_WIDTH = 2.0; v392.FRAME_COND = True
a = v386.SheafModel('owned', SIZE, SEED).eval(); c = v392.SheafModel('owned', SIZE, SEED).eval()
sa, sc = a.state_dict(), c.state_dict()
widened = [k for k in sa if k.endswith('.weight') and sa[k].shape != sc[k].shape]
for k in sa:
    if k in widened:
        n = sa[k].shape[1]
        assert torch.equal(sc[k][:, :n], sa[k]) and float(sc[k][:, n:].abs().sum()) == 0., f'{k}: original columns changed or new columns non-zero'
        assert sc[k].shape[1] == n + 3
    else:
        assert torch.equal(sa[k], sc[k]), f'{k} differs'
assert len(widened) == 1 + len(c.decoder.trunk.blocks), widened
lift = c.decoder.trunk.lift[0].out_channels; widths = [blk[0].out_channels for blk in c.decoder.trunk.blocks]
n386 = sum(p.numel() for p in a.parameters()); n392 = sum(p.numel() for p in c.parameters())
assert n392 - n386 == 3 * 9 * (lift + sum(widths)), (n392 - n386, lift, widths)
with torch.no_grad():
    mu, q, mask = a.source(private); feats = torch.cat((v386.SheafModel.latent_quantize(mu), (1. / (mask * q + 1e-6)).log()), -1)
    oa = a.decoder(torch.cat((feats, feats), -1))
    for L in (8., 14., 20.):
        v392.CHART_M = L
        oc = c.decoder(torch.cat((feats, feats), -1))
        assert torch.equal(oa[0], oc[0]) and torch.equal(oa[1], oc[1]), f'decoder function differs at init (CHART_M={L})'
print(f'2. --frame-cond on (W=2): {len(widened)} convs widened by 3 zero columns, all other tensors identical; function equal at init; params {n386:,} -> {n392:,} (+{n392 - n386:,})')

v392.CHART_M = 14.
p = v392.public_extra(2, 128, 128, torch.device('cpu'))
u, v, Lp = p[0, 0], p[0, 1], p[0, 2]
assert p.shape == (2, 3, 128, 128) and torch.equal(p[0], p[1])
assert float(u.min()) >= -.5 and float(u.max()) < .5 and float(v.min()) >= -.5 and float(v.max()) < .5
assert int((torch.diff(u[0]) < 0).sum()) == 7, 'u must wrap exactly 7 times across 8 blocks'
assert int((torch.diff(v[:, 0]) < 0).sum()) == 7
assert torch.allclose(Lp, torch.full_like(Lp, 14. / 20.))
assert torch.equal(u[0], u[5]) and torch.equal(v[:, 0], v[:, 9]), 'u depends on x only, v on y only'
print('3. public_extra: u/v block phase in [-0.5,0.5) with 8 cycles, L plane = CHART_M/20, agent-independent')

feats_g = feats.clone().requires_grad_(True)
out = c.decoder(torch.cat((feats_g, feats_g), -1)); (out[0].sum() + out[1].sum()).backward()
g = c.decoder.trunk.lift[0].weight.grad
n = a.decoder.trunk.lift[0].in_channels
assert g is not None and float(g[:, n:].abs().sum()) > 0, 'no gradient on the new public columns'
print('4. gradients reach the new public columns')
print('ALL TESTS PASSED')
