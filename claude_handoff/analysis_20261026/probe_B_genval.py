"""probe_B_genval (analyst B, 2026-10-26): generate the 24-house VAL panel (index < 8 per room stratum, exactly the diag_storey_v429 plan)
on CPU and pickle the fixtures for the other B probes. VALIDATION split only (TEST is never touched). Run on uk-linux:
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python probe_B_genval.py --source <sealed> --scripts <claude-diag scripts dir> --out B/val24.pkl
"""
import argparse, multiprocessing, os, pickle, sys, time
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--scripts', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, args.scripts)
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v431 as T
import mapping_3d_storey_data_v420 as W
if __name__ == '__main__':
    t0 = time.time(); dims = W.DIMS['R1']
    plan = [s for s in W.scene_plan('VALIDATION', 16, dims) if s['index'] < 8]
    assert all(s['split'] == 'VALIDATION' for s in plan)
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        fxs = pool.map_async(T._fixture, plan, chunksize=1).get(timeout=3600)
    pickle.dump(dict(plan=plan, fixtures=fxs), open(args.out, 'wb'), protocol=4)
    print('generated', sum(f is not None for f in fxs), 'of', len(plan), 'in %.0f s' % (time.time() - t0))
