"""Tests for train_3d_v428 (Claude, 2026-10-25, before any run). CPU: python test_3d_v428.py --source <sealed> --ckpt <a v427/v423 final.pt>
A. Module level: v428 == v427 bit-for-bit (same model, source, join, logits, loss and gradients) for the D2 recipe (d3 W2 16x16x1).
B. CLI warm start, exercised through main() on a tiny run (--max-updates 2, --updates 16000, cpu, 2 val houses): (i) without --init-from,
   initial.pt equals a fresh seeded model; (ii) with --init-from CKPT the run's initial.pt equals CKPT's state_dict tensor-for-tensor, run.json
   records warm_start with the checkpoint's sha256 and source updates, and initial_weights_sha256 equals the checkpoint's weights hash;
   (iii) a recipe mismatch (--decoder-width 1.0 against a W2 checkpoint) is refused before training.
C. Source text: --init-from is parsed, the load happens after the model is built and before initial.pt is written; the optimiser is created
   after the load (fresh state).
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import torch
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--cache', required=True); args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mapping_3d_storey_data_v420 as W
import train_3d_v427 as T7
import train_3d_v428 as T8
t0 = time.time(); R1 = W.DIMS['R1']
fx = W.make_fixture('TRAIN', 7, 2, R1); own = torch.from_numpy(fx['private']); T7.CHART_M = T8.CHART_M = fx['chart_m']
for M in (T7, T8):
    M.DECODER = 'd3'; M.DECODER_WIDTH = 2.0; M.BX, M.BY, M.BZ = 16, 16, 1; M.D = 256; M.PRECISION = 'learned'; M.KNOWN_WEIGHT = 'uniform'
m7 = T7.SheafModel3D(R1, 342101); m8 = T8.SheafModel3D(R1, 342101)
s7, s8 = m7.state_dict(), m8.state_dict(); assert list(s7) == list(s8) and all(torch.equal(s7[k], s8[k]) for k in s7)
a7 = m7.source(own); a8 = m8.source(own); assert all(torch.equal(x, y) for x, y in zip(a7, a8))
f7 = T7.rollout_max(*a7, fx['edges'], stages=(0,)); f8 = T8.rollout_max(*a8, fx['edges'], stages=(0,)); tgt = T7.stream_targets(own, torch.from_numpy(fx['reached'][0]))
l7, _, _ = T7.stage_loss(m7.decoder, f7[0], tgt, R1); l8, _, _ = T8.stage_loss(m8.decoder, f8[0], tgt, R1); assert torch.equal(l7, l8)
g7 = torch.autograd.grad(l7, list(m7.parameters()), allow_unused=True); g8 = torch.autograd.grad(l8, list(m8.parameters()), allow_unused=True)
assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(g7, g8))
print(f'A. v428 == v427 bit-for-bit (model, source, join, logits, loss, gradients) ({time.time()-t0:.0f}s)')

ck = torch.load(args.ckpt, map_location='cpu', weights_only=False); meta = ck['meta']
assert meta['blocks'] == [16, 16, 1] and meta.get('decoder_width') == 2.0 and meta.get('decoder', 'd3') == 'd3', 'test needs a D2-recipe checkpoint'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', SEALED_SOURCE=args.source)
base = ['python', str(HERE / 'train_3d_v428.py'), '--source', args.source, '--cache', args.cache, '--dims', '128,128,32', '--device', 'cpu', '--val-per-stratum', '1', '--gen-workers', '2',
        '--layout', '16x16x1', '--decoder', 'd3', '--decoder-width', '2.0', '--precision', 'learned', '--known-weight', 'uniform', '--seed', '342101', '--updates', '16000', '--max-updates', '2', '--val-every', '2', '--deadline-total', '0', '--threads', '4']
tmp = Path(tempfile.mkdtemp(prefix='v428test_'))
r = subprocess.run(base + ['--out', str(tmp / 'cold')], env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
cold = torch.load(tmp / 'cold' / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=False)['state_dict']
assert all(torch.equal(cold[k], s8[k]) for k in s8) and json.load(open(tmp / 'cold' / 'run.json'))['warm_start'] is None
r = subprocess.run(base + ['--out', str(tmp / 'warm'), '--init-from', args.ckpt], env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
warm = torch.load(tmp / 'warm' / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=False); rj = json.load(open(tmp / 'warm' / 'run.json'))
assert list(warm['state_dict']) == list(ck['state_dict']) and all(torch.equal(warm['state_dict'][k], ck['state_dict'][k]) for k in ck['state_dict'])
assert rj['warm_start']['init_from'] == args.ckpt and rj['warm_start']['source_updates'] == ck['updates'] and rj['warm_start']['init_file_sha256'] == T8.sha_file(args.ckpt)
assert rj['initial_weights_sha256'] == ck['weights_sha256'] and rj['warm_start']['source_weights_sha256'] == ck['weights_sha256']
r = subprocess.run([x if x != '2.0' else '1.0' for x in base] + ['--out', str(tmp / 'bad'), '--init-from', args.ckpt], env=env, capture_output=True, text=True)
assert r.returncode != 0 and 'warm start decoder_width mismatch' in (r.stderr + r.stdout) and not (tmp / 'bad' / 'checkpoints' / 'initial.pt').exists()
print(f'B. CLI: cold initial.pt == seeded model; warm initial.pt == checkpoint tensor-for-tensor, run.json records warm_start + hashes; recipe mismatch refused before initial.pt ({time.time()-t0:.0f}s)')
src = (HERE / 'train_3d_v428.py').read_text()
i_model = src.index("    model = SheafModel3D(dims, args.seed).to(device)"); i_load = src.index("model.load_state_dict(payload['state_dict'], strict=True)"); i_opt = src.index("opt = torch.optim.AdamW(model.parameters()"); i_init = src.index("out / 'checkpoints' / 'initial.pt'")
assert i_model < i_load < i_opt and i_load < i_init and "warm_start=warm_start" in src
print('C. load happens after the model is built, before the optimiser and before initial.pt')
print('ALL TESTS PASSED', f'{time.time()-t0:.0f}s')
