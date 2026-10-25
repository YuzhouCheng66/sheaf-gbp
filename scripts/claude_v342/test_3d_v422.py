"""Tests for train_3d_v422 / batch 3S-W (Claude, 2026-10-25, before any v422 run). Run on uk-linux, CPU (CUDA_VISIBLE_DEVICES=):
python test_3d_v422.py --source <sealed>
A. --decoder-width 1.0 builds train_3d_v420 bit-for-bit: same state_dict keys and tensors, same source (mu, q, mask), same join, same
   decoder logits on a real v420 fixture.
B. --decoder-width 2.0: trunk and head initial tensors are identical to v420 (only the decoder differs); decoder channels pre 128, lift
   128, stages 96 / 72, output 72 -> 16; decoder parameter count; logits shape 2 x 128 x 128 x 32; three AdamW steps reduce the K0 loss.
C. The trainer's CLI sets the module global and records decoder_width in run.json (parsed from the source text, no cache needed).
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v420 as T0
import train_3d_v422 as T2
t0 = time.time()
R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1)
own = torch.from_numpy(fx['private']); T0.CHART_M = T2.CHART_M = fx['chart_m']

T2.DECODER_WIDTH = 1.0
for seed in (342101, 342202):
    m0 = T0.SheafModel3D(R1, seed); m1 = T2.SheafModel3D(R1, seed)
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0), seed
with torch.no_grad():
    a0 = m0.source(own); a1 = m1.source(own)
    assert all(torch.equal(x, y) for x, y in zip(a0, a1))
    f0 = T0.rollout_max(*a0, fx['edges'], stages=(0, 8)); f1 = T2.rollout_max(*a1, fx['edges'], stages=(0, 8))
    for k in (0, 8):
        assert torch.equal(f0[k], f1[k])
        assert all(torch.equal(x, y) for x, y in zip(m0.decoder(f0[k]), m1.decoder(f1[k])))
print(f'A. W = 1.0 == train_3d_v420 bit-for-bit (state_dict for both seeds, source, join, decoder logits) ({time.time()-t0:.0f}s)')

T2.DECODER_WIDTH = 2.0
m2 = T2.SheafModel3D(R1, 342202); s2 = m2.state_dict()
for k in s0:
    if not k.startswith('decoder.'):
        assert torch.equal(s0[k], s2[k]), k
dec = m2.decoder
assert dec.pre[0].out_channels == 128 and dec.pre[2].out_channels == 128 and dec.lift[0].out_channels == 128
assert [b[0].out_channels for b in dec.blocks] == [96, 72] and dec.output.in_channels == 72 and dec.output.out_channels == 16
p1 = sum(p.numel() for p in m1.decoder.parameters()); p2 = sum(p.numel() for p in dec.parameters())
assert p1 == 468376 and p2 == 1832704, (p1, p2)                          # review S7: asserted, not only printed
with torch.no_grad():
    mu, q, mk = m2.source(own); f = T2.rollout_max(mu, q, mk, fx['edges'], stages=(8,))[8]
    kl, ol = dec(f); assert kl.shape == ol.shape == (8, 128 * 128 * 32)
opt = torch.optim.AdamW(m2.parameters(), lr=1e-3); m2.train(); losses = []
for step in range(3):
    opt.zero_grad(); mu, q, mk = m2.source(own); fz = T2.rollout_max(mu, q, mk, fx['edges'], stages=(0,))
    l, _, _ = T2.stage_loss(m2.decoder, fz[0], T2.stream_targets(own, torch.from_numpy(fx['reached'][0])), R1); l.backward(); opt.step(); losses.append(float(l.detach()))
assert losses[-1] < losses[0], losses
print(f'B. W = 2.0: trunk / head initial tensors identical to v420; decoder 128/128/96/72 -> 16; decoder params {p1:,} -> {p2:,} ({p2/p1:.2f}x); '
      f'loss {[round(x, 4) for x in losses]} ({time.time()-t0:.0f}s)')

src = (HERE / 'train_3d_v422.py').read_text()
assert "DECODER_WIDTH = args.decoder_width" in src and "decoder_width=DECODER_WIDTH" in src and "global CHART_M, DECODER_WIDTH" in src
assert src.count('DECODER_WIDTH') == 7, src.count('DECODER_WIDTH')
print('C. CLI sets DECODER_WIDTH before any model is built and run.json records decoder_width')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
