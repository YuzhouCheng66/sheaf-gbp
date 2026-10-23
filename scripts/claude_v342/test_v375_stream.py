"""Tests for the v375 data-scale path (Claude, 2026-10-23). CPU. TRAIN split only.
1. cache round trip: gen_hx_cache_v375._one writes a scene whose private / positions / adjacency / edges / reached / door_mask equal what
   the trainer's load() derives from data.make_fixture + bank (exact equality, edge tuples with int/int/float types).
2. stream_targets(own, reached[k]) == bank.forest_targets(inputs, k)[0] for every k in HOPS, every node (exact int8 equality), and the
   truncated DT built from either is identical.
3. trainer smoke through --train-cache (2 houses per stratum, 1 epoch) runs and records the cache manifest in run.json; the plan / index /
   regeneration asserts pass on a correct cache and --expect-updates rejects a wrong-K cache (review M2).
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SEALED_SOURCE', HERE.parent / 'handoff_src/claude_handoff/reference_source_v310')).resolve()
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / 'exploration_grid')); sys.path.insert(0, str(HERE))
from scripts import mapping_native_bank_v306 as bank
import gen_hx_cache_v375 as gen
import importlib.util
spec = importlib.util.spec_from_file_location('t375', HERE / 'train_v375.py'); t375 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t375)

ok = 0
for world, modname in (('houseexpo', 'mapping_houseexpo_data_v365'), ('houseexpo_fit', 'mapping_houseexpo_data_v371')):
    data = __import__(modname)
    gen.G.clear(); gen._init(str(SRC), world)
    with tempfile.TemporaryDirectory() as tmp:
        for rooms, index, agents in ((1, 0, 8), (2, 3, 16), (3, 5, 16)):
            sp = dict(split='TRAIN', index=index, rooms=rooms, agents=agents, size=128)
            r = gen._one((sp, tmp)); z = np.load(Path(tmp) / r['file'])
            fx = data.make_fixture(**sp); inp = fx['inputs']
            fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
            assert np.array_equal(z['private'], inp['private']) and z['private'].dtype == np.int8
            assert np.array_equal(z['positions_m'], inp['positions_m']) and np.array_equal(z['adjacency'], inp['adjacency'])
            assert np.array_equal(z['door_mask'], fx['evaluation']['door_mask'])
            edges = [(int(i), int(j), float(w)) for i, j, w in z['edges']]
            assert edges == list(fsel['edges']) and all(type(e[0]) is int and type(e[1]) is int and type(e[2]) is float for e in edges)
            own = torch.from_numpy(z['private'])
            for j, k in enumerate(t375.HOPS):
                ref, reached = bank.forest_targets(inp, k)
                assert np.array_equal(z['reached'][j], reached)
                got = t375.stream_targets(own, torch.from_numpy(z['reached'][j]))
                n = own.shape[0]
                assert got.dtype == torch.int8 and torch.equal(got, torch.from_numpy(ref.reshape(n, -1))), (world, sp, k)
                d_ref = t375.truncated_dt(torch.from_numpy(ref.reshape(n, 128, 128)) == 1, 6)
                d_got = t375.truncated_dt(got.reshape(n, 128, 128) == 1, 6)
                assert torch.equal(d_ref, d_got)
                ok += 1
print('round-trip + target equality checks passed:', ok)

with tempfile.TemporaryDirectory() as tmp:
    cache = Path(tmp) / 'cache'
    subprocess.run([sys.executable, str(HERE / 'gen_hx_cache_v375.py'), '--source', str(SRC), '--world', 'houseexpo_fit', '--per-stratum', '2', '--workers', '2', '--out', str(cache)], check=True)
    out = Path(tmp) / 'run'
    subprocess.run([sys.executable, str(HERE / 'train_v375.py'), '--source', str(SRC), '--world', 'houseexpo_fit', '--train-cache', str(cache), '--out', str(out), '--arm', 'owned',
                    '--size', '128', '--device', 'cpu', '--epochs', '1', '--smoke', '--val-every', '1', '--deadline-epoch', '9999999999', '--deadline-total', '9999999999',
                    '--refine-steps', '0', '--readout', 'posterior_only', '--fusion', 'max', '--expect-updates', '6'], check=True)
    bad = subprocess.run([sys.executable, str(HERE / 'train_v375.py'), '--source', str(SRC), '--world', 'houseexpo_fit', '--train-cache', str(cache), '--out', str(out) + '-bad', '--arm', 'owned', '--size', '128', '--device', 'cpu', '--epochs', '1', '--deadline-epoch', '9999999999', '--deadline-total', '9999999999', '--refine-steps', '0', '--readout', 'posterior_only', '--fusion', 'max', '--expect-updates', '43200'], capture_output=True, text=True)
    assert bad.returncode != 0 and 'expected 43200' in bad.stderr, 'wrong-K cache must fail fast (review M2)'
    run = json.load(open(out / 'run.json')); hist = json.load(open(out / 'history.json'))
    assert run['train_scenes'] == 6 and run['train_cache']['manifest_sha256'] and hist[0]['updates'] == 6 and np.isfinite(hist[0]['train_loss'])
print('trainer smoke through --train-cache passed; ALL OK')
