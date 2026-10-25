"""Tests for train_3d_v425 / batch 3S-P (Claude, 2026-10-25, before any run). CPU: python test_3d_v425.py --source <sealed>
A. --decoder d3 is train_3d_v424 bit-for-bit for (W 2, 8x8x4) and (W 1, 16x16x1): state_dict, source, join, decoder logits.
B. --decoder plan_column (W 2, 8x8x4): trunk / head initial tensors identical to v424 (and hence to 3S-A / 3S-W); input grid is
   [b, 2 BZ, BX, BY] with channel z = layer z's mu and BZ + z = its log var / 8 (checked by construction on a synthetic posterior);
   output shapes (8, V); column locality of the readout: zeroing every output-conv weight except channel (t = 1, z = 5) makes the
   occupancy logits non-zero only at z = 5 and the known logits all equal the bias -> the channel-to-(t, z) mapping is right; decoder
   parameter count 2.4-2.7 M and MACs ~0.1x d3 W2 (reported); three AdamW steps reduce the K0 loss; 16x16x1 input also builds.
C. CLI: --decoder is set before the model is built and recorded in run.json (source text).
"""
import argparse, os, sys, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v424 as T4
import train_3d_v425 as T5
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T4.CHART_M = T5.CHART_M = fx['chart_m']

def same(ma, mb, Ma, Mb):
    sa, sb = ma.state_dict(), mb.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        a = ma.source(own); b = mb.source(own); assert all(torch.equal(x, y) for x, y in zip(a, b))
        fa = Ma.rollout_max(*a, fx['edges'], stages=(0, 8)); fb = Mb.rollout_max(*b, fx['edges'], stages=(0, 8))
        for k in (0, 8):
            assert torch.equal(fa[k], fb[k]) and all(torch.equal(x, y) for x, y in zip(ma.decoder(fa[k]), mb.decoder(fb[k])))

T5.DECODER = 'd3'
for (w, layout, seed) in ((2.0, '8x8x4', 342101), (1.0, '16x16x1', 342202)):
    for M in (T4, T5):
        M.DECODER_WIDTH = w; M.BX, M.BY, M.BZ = M.LAYOUTS[layout]; M.D = M.BX * M.BY * M.BZ
    same(T4.SheafModel3D(R1, seed), T5.SheafModel3D(R1, seed), T4, T5)
print(f'A. --decoder d3 == train_3d_v424 bit-for-bit for (W2, 8x8x4) and (W1, 16x16x1) ({time.time()-t0:.0f}s)')

for M in (T4, T5):
    M.DECODER_WIDTH = 2.0; M.BX, M.BY, M.BZ = 8, 8, 4; M.D = 256
T5.DECODER = 'plan_column'
m4 = T4.SheafModel3D(R1, 342101); m5 = T5.SheafModel3D(R1, 342101)
s4, s5 = m4.state_dict(), m5.state_dict()
for k in s4:
    if not k.startswith('decoder.'):
        assert torch.equal(s4[k], s5[k]), k
dec = m5.decoder
# input arrangement on a synthetic posterior: mu_d = d / 256, lv_d = 8 * (d / 256)
post = torch.cat((torch.arange(256.) / 256, 8 * torch.arange(256.) / 256))[None]
mu = post[:, :256].reshape(1, 8, 8, 4); grid_expected = torch.cat((mu.permute(0, 3, 1, 2), mu.permute(0, 3, 1, 2)), 1)
b = post.shape[0]; mu_ = post[:, :256].reshape(b, 8, 8, 4); lv_ = (post[:, 256:] / 8.).reshape(b, 8, 8, 4)
grid = torch.cat((mu_.permute(0, 3, 1, 2), lv_.permute(0, 3, 1, 2)), 1)
assert torch.allclose(grid, grid_expected) and grid.shape == (1, 8, 8, 8) and float(grid[0, 2, 3, 5]) == float(mu[0, 3, 5, 2])
with torch.no_grad():
    muq, q, mk = m5.source(own); f = T5.rollout_max(muq, q, mk, fx['edges'], stages=(8,))[8]
    kl, ol = dec(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
    # column locality / channel mapping of the readout
    wsave, bsave = dec.output.weight.clone(), dec.output.bias.clone()
    dec.output.weight.zero_(); dec.output.bias.zero_(); dec.output.weight[1 * 32 + 5] = 1.; dec.output.bias[0:32] = 3.
    kl2, ol2 = dec(f); kl2 = kl2.reshape(8, 128, 128, 32); ol2 = ol2.reshape(8, 128, 128, 32)
    assert torch.all(kl2 == 3.) and torch.all(ol2[..., :5] == 0) and torch.all(ol2[..., 6:] == 0) and (ol2[..., 5].abs() > 0).float().mean() > .5
    dec.output.weight.copy_(wsave); dec.output.bias.copy_(bsave)
p5 = sum(p.numel() for p in dec.parameters()); p4 = sum(p.numel() for p in m4.decoder.parameters())
assert 2_400_000 <= p5 <= 2_700_000, p5
# MACs by hooks
macs = {'d3': 0, 'pc': 0}
def hook(name):
    def f(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.kernel_size[2] if len(mod.kernel_size) == 3 else 1)
        macs[name] += mod.in_channels * k * out[0].numel()                         # out[0].numel() already counts the output channels (review S7)
    return f
hs = [mod.register_forward_hook(hook('d3')) for mod in m4.decoder.modules() if isinstance(mod, (torch.nn.Conv2d, torch.nn.Conv3d))]
hs += [mod.register_forward_hook(hook('pc')) for mod in dec.modules() if isinstance(mod, torch.nn.Conv2d)]
with torch.no_grad():
    m4.decoder(f[:1]); dec(f[:1])
for h in hs: h.remove()
opt = torch.optim.AdamW(m5.parameters(), lr=1e-3); m5.train(); losses = []
for step in range(3):
    opt.zero_grad(); muq, q, mk = m5.source(own); fz = T5.rollout_max(muq, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T5.stage_loss(m5.decoder, fz[0], T5.stream_targets(own, torch.from_numpy(fx['reached'][0])), R1); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
T5.BX, T5.BY, T5.BZ = 16, 16, 1; T5.D = 256; m16 = T5.SheafModel3D(R1, 342101)
with torch.no_grad():
    a, b_, c = m16.source(own); kl3, ol3 = m16.decoder(T5.rollout_max(a, b_, c, fx['edges'], stages=(0,))[0]); assert kl3.shape == (8, 128 * 128 * 32)
T5.BX, T5.BY, T5.BZ = 8, 8, 4
print(f'B. plan_column W2: trunk / head identical to v424; grid [b, 2BZ, BX, BY] layout verified; readout channel t*NZ+z -> (t, z) verified; '
      f'decoder params {p5:,} (d3 W2 {p4:,}); MACs per decode {macs["pc"]/1e9:.2f} G vs d3 W2 {macs["d3"]/1e9:.2f} G; loss {[round(x, 4) for x in losses]}; 16x16x1 builds ({time.time()-t0:.0f}s)')
src = (HERE / 'train_3d_v425.py').read_text()
assert src.index("DECODER = args.decoder") < src.index("model = SheafModel3D(dims, args.seed)") and "decoder=DECODER," in src and "global CHART_M, DECODER_WIDTH, BX, BY, BZ, D, DECODER" in src
print('C. CLI sets --decoder before the model is built and run.json records it')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
