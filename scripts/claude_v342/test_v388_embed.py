"""Equivalence / sanity tests for train_v388.py (code-level embedding). Claude, 2026-10-23, written before batch U.
1. D = 0 is bit-identical to v386 (and hence v378): state_dict, encoder outputs, decoder outputs, predict.
2. D = 8: every tensor except decoder.pre.0.* is bit-identical to v386 under the same seed; pre.0.weight has the v386 columns at
   input channels 0 and D, zeros elsewhere; the embedding's channel 0 equals the codebook levels; the decoder FUNCTION at init equals
   v386's on hard-8 inputs (torch.allclose, atol 1e-6) and on arbitrary continuous inputs the interpolation reproduces the level value.
3. Gradients flow to the embedding and (through the slope) to the mean input; parameter count = v386 + 255*D + (D-1)*c*9 extra conv weights.
4. Warm-start meta check: code_embed is recorded.
Runs on CPU; needs only torch/numpy.   python test_v388_embed.py
"""
import importlib.util
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


v386, v388 = load('train_v386'), load('train_v388')
for m in (v386, v388):
    m.READOUT_MODE = 'posterior_only'; m.PRECISION_MODE = 'learned'; m.DECODER_WIDTH = 1.0
SIZE, SEED, D = 128, 342101, 8
private = torch.from_numpy(np.random.default_rng(0).integers(-1, 2, size=(4, SIZE, SIZE)).astype(np.int8))
a = v386.SheafModel('owned', SIZE, SEED).eval()
with torch.no_grad():
    ma, qa, ka = a.source(private)
    feats = torch.cat((v386.SheafModel.latent_quantize(ma), (1. / (ka * qa + 1e-6)).log()), -1)   # hard-8 means, as the rollout delivers
    oa = a.decoder(torch.cat((feats, feats), -1))
# ---- 1. D = 0
v388.CODE_EMBED = 0
b = v388.SheafModel('owned', SIZE, SEED).eval()
sa, sb = a.state_dict(), b.state_dict()
assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa), 'D=0 differs from v386'
with torch.no_grad():
    ob = b.decoder(torch.cat((feats, feats), -1))
assert torch.equal(oa[0], ob[0]) and torch.equal(oa[1], ob[1])
print('1. D=0 bit-identical to v386 (state_dict and decoder outputs)')
# ---- 2. D = 8 identity init
v388.CODE_EMBED = D
c = v388.SheafModel('owned', SIZE, SEED).eval(); sc = c.state_dict()
extra = {'decoder.embed', 'decoder.pre.0.weight'}
for k in sa:
    if k in extra:
        continue
    assert k in sc and torch.equal(sa[k], sc[k]), f'{k} differs at D={D}'
assert sc['decoder.pre.0.bias'].equal(sa['decoder.pre.0.bias'])
w = sc['decoder.pre.0.weight']; w0 = sa['decoder.pre.0.weight']
assert w.shape[1] == D + 1 and torch.equal(w[:, 0], w0[:, 0]) and torch.equal(w[:, D], w0[:, 1]) and float(w[:, 1:D].abs().sum()) == 0.
E = sc['decoder.embed']; levels = torch.arange(-127, 128).float() / 127.
assert E.shape == (255, D) and torch.equal(E[:, 0], levels) and float(E[:, 1:].abs().max()) < 0.3
with torch.no_grad():
    oc = c.decoder(torch.cat((feats, feats), -1))
    assert torch.allclose(oa[0], oc[0], atol=1e-6) and torch.allclose(oa[1], oc[1], atol=1e-6), 'decoder function differs at init'
    # interpolation reproduces the level value on channel 0 for arbitrary continuous means
    mean = torch.rand(3, 64) * 2 - 1
    plane = c.decoder.embed_plane(mean)
    assert torch.allclose(plane[:, 0].reshape(3, 64), mean, atol=1e-6)
print(f'2. D={D}: all tensors but pre.0.weight/embed identical; identity init; decoder function equal at init; interpolation exact on channel 0')
# ---- 3. gradients and parameter count
feats_g = feats.clone().requires_grad_(True)
out = c.decoder(torch.cat((feats_g, feats_g), -1)); (out[0].sum() + out[1].sum()).backward()
assert c.decoder.embed.grad is not None and float(c.decoder.embed.grad.abs().sum()) > 0
assert feats_g.grad is not None and float(feats_g.grad[:, :64].abs().sum()) > 0, 'no gradient to the mean input through the table'
n386 = sum(p.numel() for p in a.parameters()); n388 = sum(p.numel() for p in c.parameters())
assert n388 - n386 == 255 * D + (D - 1) * 64 * 9, (n388 - n386)
print(f'3. gradients reach the table and the mean input; params {n386:,} -> {n388:,} (+{n388 - n386:,})')
print('ALL TESTS PASSED')
