"""Equivalence / sanity tests for train_v386.py (decoder width knob). Claude, 2026-10-23, written before batch T.
1. W = 1.0 is bit-identical to v378: same state_dict keys/shapes/values under the same seed, same encoder outputs, same decoder
   outputs on random inputs, and a v386 state_dict loads strictly into a v378 model.
2. For every W, the encoder (PatchTrunk) and head initial weights are identical to v378 (constructed before the decoder).
3. W = 2.0: decoder parameter count ~3.95x (2,491,330 vs 630,498 at H128), forward shapes unchanged.
Runs on CPU, needs only torch/numpy (no sealed source).   python test_v386_width.py
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


v378, v386 = load('train_v378'), load('train_v386')
for m in (v378, v386):
    m.READOUT_MODE = 'posterior_only'; m.PRECISION_MODE = 'learned'
SIZE, SEED = 128, 342101
torch.manual_seed(0)
private = torch.from_numpy(np.random.default_rng(0).integers(-1, 2, size=(4, SIZE, SIZE)).astype(np.int8))


def decoder_params(model):
    return sum(p.numel() for p in model.decoder.parameters())


# ---- 1. W = 1.0 equivalence
v386.DECODER_WIDTH = 1.0
a = v378.SheafModel('owned', SIZE, SEED).eval(); b = v386.SheafModel('owned', SIZE, SEED).eval()
sa, sb = a.state_dict(), b.state_dict()
assert list(sa.keys()) == list(sb.keys()), 'state_dict keys differ at W=1'
for k in sa:
    assert sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]), f'tensor {k} differs at W=1'
assert a.weights_hash() == b.weights_hash()
with torch.no_grad():
    ma, qa, ka = a.source(private); mb, qb, kb = b.source(private)
    assert torch.equal(ma, mb) and torch.equal(qa, qb) and torch.equal(ka, kb), 'encoder outputs differ at W=1'
    feats = torch.cat((ma, (1. / (ka * qa + 1e-6)).log()), -1)
    oa = a.decoder(torch.cat((feats, feats), -1)); ob = b.decoder(torch.cat((feats, feats), -1))
    assert torch.equal(oa[0], ob[0]) and torch.equal(oa[1], ob[1]), 'decoder outputs differ at W=1'
    pa = v378.predict(a.decoder, torch.cat((feats, feats), -1)); pb = v386.predict(b.decoder, torch.cat((feats, feats), -1))
    assert torch.equal(pa, pb)
a2 = v378.SheafModel('owned', SIZE, 1); a2.load_state_dict(sb, strict=True)     # v386 W=1 checkpoint loads strictly into v378
assert decoder_params(a) == decoder_params(b) == 630498, decoder_params(a)
print('1. W=1.0: state_dict, encoder, decoder, predict all bit-identical to v378; decoder params', decoder_params(a))

# ---- 2. encoder/head init identical for every W
for W in (1.5, 2.0):
    v386.DECODER_WIDTH = W
    c = v386.SheafModel('owned', SIZE, SEED)
    sc = c.state_dict()
    for k in sa:
        if k.startswith('trunk.') or k.startswith('head.'):
            assert torch.equal(sa[k], sc[k]), f'encoder/head tensor {k} differs at W={W}'
    with torch.no_grad():
        mc, qc, kc = c.source(private)
        assert torch.equal(mc, ma) and torch.equal(qc, qa) and torch.equal(kc, ka)
        oc = c.decoder(torch.cat((feats, feats), -1))
        assert oc[0].shape == oa[0].shape and oc[1].shape == oa[1].shape and torch.isfinite(oc[0]).all() and torch.isfinite(oc[1]).all()
    ratio = decoder_params(c) / decoder_params(a)
    print(f'2/3. W={W}: encoder+head init identical to v378; decoder params {decoder_params(c):,} ({ratio:.2f}x); total {sum(p.numel() for p in c.parameters()):,}')
    if W == 2.0:
        assert decoder_params(c) == 2491330, decoder_params(c)
        assert 3.9 < ratio < 4.0
# ---- 4. a different seed changes the encoder (sanity: the identity above is seed-driven, not constant)
d = v386.SheafModel('owned', SIZE, SEED + 1)
assert not torch.equal(d.state_dict()['trunk.net.0.weight'], sa['trunk.net.0.weight'])
print('4. different seed -> different encoder init (identity in 2 is seed-driven)')
print('ALL TESTS PASSED')
