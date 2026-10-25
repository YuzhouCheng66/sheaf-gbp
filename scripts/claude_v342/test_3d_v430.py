"""Tests for train_3d_v430 / batch 3S-S (Claude, 2026-10-25, before any run). CPU:
python test_3d_v430.py --source <sealed> --cache <v420 R1 cache dir>
A. Module level: v430 == v429 bit-for-bit (model, source, join, logits, loss, gradients) for the D2 recipe.
B. CLI: an eager run and a --cache-lazy run of 3 updates (cpu, 1 VAL house per stratum, same seed) end with the SAME final weights hash
   and the same training losses; run.json records cache_lazy; the lazy run holds no decoded scene before the loop (source text).
"""
import argparse, json, os, subprocess, sys, tempfile, time, re
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--cache', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v429 as T9
import train_3d_v430 as T0
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T9.CHART_M = T0.CHART_M = fx['chart_m']
for M in (T9, T0):
    M.DECODER = 'd3'; M.DECODER_WIDTH = 2.0; M.BX, M.BY, M.BZ = 16, 16, 1; M.D = 256; M.PRECISION = 'learned'; M.KNOWN_WEIGHT = 'uniform'
m9 = T9.SheafModel3D(R1, 342101); m0 = T0.SheafModel3D(R1, 342101)
s9, s0 = m9.state_dict(), m0.state_dict(); assert list(s9) == list(s0) and all(torch.equal(s9[k], s0[k]) for k in s9)
a9 = m9.source(own); a0 = m0.source(own); assert all(torch.equal(x, y) for x, y in zip(a9, a0))
f9 = T9.rollout_max(*a9, fx['edges'], stages=(0,)); f0 = T0.rollout_max(*a0, fx['edges'], stages=(0,)); tgt = T9.stream_targets(own, torch.from_numpy(fx['reached'][0]))
l9, _, _ = T9.stage_loss(m9.decoder, f9[0], tgt, R1); l0, _, _ = T0.stage_loss(m0.decoder, f0[0], tgt, R1); assert torch.equal(l9, l0)
g9 = torch.autograd.grad(l9, list(m9.parameters()), allow_unused=True); g0 = torch.autograd.grad(l0, list(m0.parameters()), allow_unused=True)
assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(g9, g0))
print(f'A. v430 == v429 bit-for-bit (model, source, join, loss, gradients) ({time.time()-t0:.0f}s)')

env = dict(os.environ, CUDA_VISIBLE_DEVICES='', SEALED_SOURCE=args.source)
base = ['python', str(HERE / 'train_3d_v430.py'), '--source', args.source, '--cache', args.cache, '--dims', '128,128,32', '--device', 'cpu', '--val-per-stratum', '1', '--gen-workers', '2',
        '--layout', '16x16x1', '--decoder', 'd3', '--decoder-width', '2.0', '--precision', 'learned', '--known-weight', 'uniform', '--seed', '342101', '--updates', '16000', '--max-updates', '3', '--val-every', '3', '--deadline-total', '0', '--threads', '4']
tmp = Path(tempfile.mkdtemp(prefix='v430test_'))
r1 = subprocess.run(base + ['--out', str(tmp / 'eager')], env=env, capture_output=True, text=True); assert r1.returncode == 0, r1.stderr[-2000:]
r2 = subprocess.run(base + ['--out', str(tmp / 'lazy'), '--cache-lazy'], env=env, capture_output=True, text=True); assert r2.returncode == 0, r2.stderr[-2000:]
fe = json.load(open(tmp / 'eager' / 'final.json')); fl = json.load(open(tmp / 'lazy' / 'final.json'))
assert fe['updates'] == fl['updates'] == 3 and fe['weights_sha256'] == fl['weights_sha256'], (fe['weights_sha256'], fl['weights_sha256'])
he = json.load(open(tmp / 'eager' / 'history.json')); hl = json.load(open(tmp / 'lazy' / 'history.json'))
assert he[0]['train_loss'] == hl[0]['train_loss'] and he[0]['weights_sha256'] == hl[0]['weights_sha256']
me = json.load(open(tmp / 'eager' / 'run.json')); ml = json.load(open(tmp / 'lazy' / 'run.json'))
assert me['cache_lazy'] is False and ml['cache_lazy'] is True and me['train_scenes'] == ml['train_scenes'] and me['initial_weights_sha256'] == ml['initial_weights_sha256']
src = (HERE / 'train_3d_v430.py').read_text()
assert "sc = scene_of(train[perm[pos]])" in src and "if args.cache_lazy else scene_from_npz" in src
print(f"B. eager and lazy runs of 3 updates: same final weights hash {fe['weights_sha256'][:12]}, same loss {he[0]['train_loss']:.6f}, run.json records cache_lazy ({time.time()-t0:.0f}s)")
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
