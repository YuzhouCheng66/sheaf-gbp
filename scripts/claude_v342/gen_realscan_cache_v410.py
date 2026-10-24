"""v410 (Claude, 2026-10-24): gen_hx_cache_v378 with the world 'realscan2d' added (index sha from the real-scan base index).
Pre-generate a large HouseExpo TRAIN fixture cache for the data-scale study (Claude, 2026-10-23; CPU, TRAIN split only).

Why: the v374 gap diagnostic shows the v367 models score R8 .81-.83 on the 600 TRAIN houses they were trained on but .66-.68 on
never-trained TRAIN houses (gap ~15 pp, two seeds) -> the decoder prior is data-limited; HouseExpo has 20,366 TRAIN houses, the plan
used 600. This script writes, for scene_plan('TRAIN', train_per_stratum=K) of the chosen world module, one compressed npz per scene
with exactly what the trainer's load() derives from data.make_fixture (no evaluation-only field except the door mask, which the
trainer also keeps): private (int8 N x H x H), positions_m, adjacency, the selected forest edges, and reached[k] (bool N x N) for
HOPS = (0, 1, 2, 4, 8) from bank.forest_targets -- the trainer rebuilds targets[k][i] = max_{s in reached[k][i]} private[s] on the GPU,
which equals forest_targets' own[s].max(0) exactly (unit test in test_v375_stream.py).
Never touches VALIDATION or TEST.  python gen_hx_cache_v375.py --source <sealed> --world houseexpo|houseexpo_fit|houseexpo_rect --per-stratum K --out DIR
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HOPS = (0, 1, 2, 4, 8)
G = {}


def _init(source, world):
    src = Path(source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scripts import mapping_native_bank_v306 as bank
    if world == 'houseexpo':
        import mapping_houseexpo_data_v365 as data
    elif world == 'houseexpo_rect':
        import mapping_houseexpo_data_v378 as data
    elif world == 'realscan2d':
        import mapping_realscan2d_data_v410 as data
    else:
        import mapping_houseexpo_data_v371 as data
    G.update(bank=bank, data=data)


def _one(args):
    spec, out = args
    bank, data = G['bank'], G['data']
    path = Path(out) / f"r{spec['rooms']}-i{spec['index']:05d}.npz"
    fx = data.make_fixture(**spec); inp = fx['inputs']
    reached = np.stack([bank.forest_targets(inp, k)[1] for k in HOPS])
    fsel = bank.select_forest(inp['positions_m'], inp['adjacency'], inp['public_frame']['communication_radius_m'])
    edges = np.asarray([(i, j, w) for i, j, w in fsel['edges']], np.float64).reshape(-1, 3)
    np.savez_compressed(path, private=inp['private'], positions_m=inp['positions_m'], adjacency=inp['adjacency'], edges=edges, reached=reached,
                        door_mask=fx['evaluation']['door_mask'])
    return dict(file=path.name, spec=spec, private_sha256=data.array_hash(inp['private']), agents=int(inp['private'].shape[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--world', required=True, choices=['houseexpo', 'houseexpo_fit', 'houseexpo_rect', 'realscan2d'])
    ap.add_argument('--per-stratum', type=int, required=True); ap.add_argument('--size', type=int, default=128); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    _init(args.source, args.world)
    data = G['data']
    specs = [s for s in data.scene_plan('TRAIN', size=args.size, train_per_stratum=args.per_stratum)]
    assert all(s['split'] == 'TRAIN' for s in specs)
    t0 = time.time(); rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.source, args.world)) as pool:
        for i, r in enumerate(pool.imap(_one, [(s, str(out)) for s in specs], chunksize=8)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(specs)} {time.time() - t0:.0f}s', flush=True)
    mod = sys.modules[data.__name__].__file__
    manifest = dict(script='gen_hx_cache_v375.py', world=args.world, per_stratum=args.per_stratum, size=args.size, hops=list(HOPS), scenes=len(rows),
                    data_module=os.path.basename(mod), data_module_sha256=hashlib.sha256(open(mod, 'rb').read()).hexdigest(),
                    index_sha256=hashlib.sha256(open(data.INDEX_FILE if args.world == 'realscan2d' else os.environ['HOUSEEXPO_INDEX'], 'rb').read()).hexdigest(), seconds=time.time() - t0, rows=rows)
    json.dump(manifest, open(out / 'manifest.json', 'w'), indent=1)
    print(f'done {len(rows)} scenes {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
