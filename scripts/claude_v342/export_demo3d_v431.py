"""export_demo3d_v431.py (Claude, 2026-10-26): data for the interactive 3D demo (demo_web3d/). Runs the current best model -- batch 3S-X
(runs/v429-stack/c2-s342101: 32x32x1 code, D = 1,024, 8+8-bit wire = 2,048 B per message, d3 W2, two cycles) with the CALIBRATED decision
rule adopted in D8-33 -- on fixed VAL houses (the first two houses of each room stratum that place 8 agents; not chosen by result) and writes
one JSON per house: meta, per-agent / per-round metrics (calibrated and v420), and gzip+base64 bit planes (np.packbits, C order over
(x, y, z) = 128 x 128 x 32): ground truth occ / slab / obj, each agent's own map (known, occ) and each agent's decoded map at rounds 0, 1, 2,
4, 8 (known, occ). Targets are rebuilt in the browser from the own maps and the reached sets (stream_targets = amax over reached agents).
CPU only. python export_demo3d_v431.py --source <sealed> --run runs/v429-stack/c2-s342101 --out demo3d --check runs/v431-decision/c2-s342101.json"""
import argparse, base64, gzip, json, os, sys, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--check', default=''); ap.add_argument('--per-stratum', type=int, default=2); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v431 as T
import eval_decision_v431 as E
import mapping_3d_storey_data_v420 as W

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
dims = W.DIMS['R1']; run = Path(args.run)
ck = torch.load(run / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); E.configure(T, ck['meta'])
model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval()
final = json.load(open(run / 'final.json')); assert final['weights_sha256'] == ck['weights_sha256']
check = {}
if args.check:
    cj = json.load(open(args.check))
    check = {json.dumps(r['spec'], sort_keys=True): r for r in cj['per_scene']['val']}
KEYS = ('surface_f1', 'var_occ_f1', 'furn_f1', 'surface_f1_tol1', 'unknown_known_fpr', 'obj_recall')
index_rows = []
with torch.no_grad():
    for rooms in (1, 2, 3):
        taken = 0
        for index in range(8):
            if taken >= args.per_stratum:
                break
            t0 = time.time()
            try:
                fx = W.make_fixture('VALIDATION', index, rooms, dims, W.AGENTS)
            except ValueError:
                continue
            taken += 1
            occ = fx['occ'].astype(bool); slab = fx['slab'].astype(bool); obj = fx['obj'].astype(bool); struct = occ & ~obj & ~slab
            own = torch.from_numpy(fx['private']); T.CHART_M = fx['chart_m']
            slab_t = torch.from_numpy(slab.reshape(-1)); obj_t = torch.from_numpy(obj.reshape(-1)); struct_t = torch.from_numpy(struct.reshape(-1))
            mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, fx['edges'])
            buf = bytearray(); offsets = {}
            def add(name, a):
                b = np.packbits(np.ascontiguousarray(a, dtype=bool).reshape(-1)).tobytes(); offsets[name] = [len(buf), len(b)]; buf.extend(b)
            add('gt_occ', occ); add('gt_slab', slab); add('gt_obj', obj)
            own_np = fx['private']
            for a in range(own_np.shape[0]):
                add(f'own_known_{a}', own_np[a] >= 0); add(f'own_occ_{a}', own_np[a] == 1)
            metrics = {}; means = {}
            for hi, k in enumerate(T.HOPS):
                tgt = T.stream_targets(own, torch.from_numpy(fx['reached'][hi]))
                kl, ol = model.decoder(feats[k])
                for rule in ('calibrated', 'v420'):
                    pred = E.decide(kl, ol, rule)
                    rows = T.metrics3d(pred, tgt, dims, slab_t, obj_t, struct_t)
                    if rule == 'calibrated':
                        metrics[str(k)] = [{m: (round(r[m], 4) if r[m] is not None else None) for m in KEYS} for r in rows]
                        pn = pred.numpy().reshape(-1, *dims)
                        for a in range(pn.shape[0]):
                            add(f'pred_{k}_known_{a}', pn[a] >= 0); add(f'pred_{k}_occ_{a}', pn[a] == 1)
                    means[f'{rule}_{k}'] = {m: (round(v, 4) if v is not None else None) for m, v in T.agg(rows).items() if m in KEYS}
            spec = fx['spec']; sk = json.dumps(spec, sort_keys=True)
            if sk in check:                                                   # the same numbers as the 3S-DR evaluation of this house
                c = check[sk]
                for rule in ('calibrated', 'v420'):
                    for k in ('0', '8'):
                        assert abs(c[f'{rule}_K{k}']['surface_f1'] - means[f'{rule}_{k}']['surface_f1']) < 2e-3, (sk, rule, k)
            own_packed = np.packbits(np.stack([own_np[0] >= 0, own_np[0] == 1]).reshape(-1)).tobytes()
            gz = gzip.compress(bytes(buf), 9)
            meta = dict(house=f'r{rooms}_i{index}', rooms=rooms, index=index, chart_m=float(fx['chart_m']), cell_m=[float(v) for v in np.asarray(fx['cell']).reshape(-1)], dims=list(dims),
                        wall_h=float(fx['wall_h']), n_objects=int(fx['n_objects']), agents=int(own_np.shape[0]), positions_m=np.asarray(fx['positions_m']).round(3).tolist(),
                        edges=[[int(i), int(j)] for i, j, _ in fx['edges']], hops=list(T.HOPS), reached=fx['reached'].astype(int).tolist(),
                        metrics=metrics, means=means, D=int(T.D), layout=list(ck['meta']['blocks']), wire_bits=list(T.WIRE_BITS), message_bytes=int(T.D * sum(T.WIRE_BITS) // 8),
                        own_map_2bit_bytes=len(own_packed), own_map_gzip_bytes=len(gzip.compress(own_packed, 9)),
                        known_fraction=[round(float((own_np[a] >= 0).mean()), 4) for a in range(own_np.shape[0])],
                        model=dict(run=str(run), weights_sha256=ck['weights_sha256'], decision='calibrated: occupied if occ_logit >= 1.5 and known_logit >= -0.5; free if known_logit >= 0; else unknown'),
                        offsets=offsets, bin_bytes=len(buf), checked_against_3SDR=sk in check, bin_gz_b64=base64.b64encode(gz).decode())
            (out / f"house_{meta['house']}.json").write_text(json.dumps(meta))
            index_rows.append({k: meta[k] for k in ('house', 'rooms', 'index', 'chart_m', 'n_objects')} | dict(R8=means['calibrated_8']['surface_f1'], K0=means['calibrated_0']['surface_f1'], gz_kb=round(len(gz) / 1024)))
            print(meta['house'], f"L {meta['chart_m']} m, R8 {means['calibrated_8']['surface_f1']:.3f} (v420 {means['v420_8']['surface_f1']:.3f}), K0 {means['calibrated_0']['surface_f1']:.3f}, gz {len(gz)/1024:.0f} KB, checked {sk in check}, {time.time()-t0:.0f}s", flush=True)
(out / 'index.json').write_text(json.dumps(index_rows, indent=1)); print('wrote', len(index_rows), 'houses')
