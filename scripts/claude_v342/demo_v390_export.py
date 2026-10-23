"""Export v390 (Claude, 2026-10-23): the v385 demo simulation (random walk + low-frequency sheaf-vector gossip + CRDT fusion +
decoding) exported as compact data for the interactive browser demo (no rendering here). Per run: <out>.json (paths, links, status,
per-agent F1 / sources, RLE offsets) and <out>.bin (RLE-coded ternary maps: own observation and fused decode per step per agent,
plus the ground-truth walls) and <out>.bin.gz. Same world/sensor/model/protocol as demo_v385_walk.py.

Original demo header: agents random-walk inside their own local region of a sealed v293 world, observe with the v293
sensor at every step, and at LOW FREQUENCY (every --comm-every steps) exchange their 64-D sheaf vectors with radio neighbours.
Fusion is the CRDT memory of eval_dynamic_v349 (per-coordinate max-precision join, newest version wins ties); one gossip round per
communication event on the full radius graph (join-semilattice => exact on any graph). Packets are billed as diffs (8 B header +
8 B change mask + 2 B per changed coordinate + N version bytes; silent when nothing changed).
Rendering (Nature/Science style, English): a = ground truth with trajectories, current poses and the radio links active at the last
communication event; b = each agent's own cumulative observation; c = the map each agent decodes from its fused Sheaf-GBP posterior.
Wall F1 in c is measured against the union of the observations the agent has actually received (its memory target).
Evaluation only: no training, no TEST split, CPU by default. Writes <out>.mp4, <out>.gif, <out>_t*.png and <out>.json.
"""
import argparse, json, math, sys, time, importlib.util
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--out', required=True, help='output path prefix (no extension)')
ap.add_argument('--size', type=int, default=128); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--split', default='VALIDATION'); ap.add_argument('--index', type=int, default=0); ap.add_argument('--rooms', type=int, default=12)
ap.add_argument('--agents', type=int, default=8); ap.add_argument('--T', type=int, default=90); ap.add_argument('--comm-every', type=int, default=6)
ap.add_argument('--radio', type=float, default=8.); ap.add_argument('--step', type=float, default=.35); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--fps', type=int, default=5)
args = ap.parse_args()
torch.set_num_threads(args.threads)
src = Path(args.source).resolve()
for p in (str(Path(__file__).resolve().parent), str(src / 'exploration_grid'), str(src)):
    sys.path.insert(0, p)
spec_ = importlib.util.spec_from_file_location('trainer', args.trainer); trainer = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(trainer)
from scripts import mapping_multiroom_data_v293 as d293
from scripts.prepare_houseexpo_pilot_v121 import make_rays, raycast_observe, radius_adjacency
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_recognition_metrics_v277 import node_metrics
from scripts.mapping_implicit_model_v192 import latent_quantize

size, n, T, K = args.size, args.agents, args.T, args.comm_every
device = torch.device(args.device)
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
trainer.READOUT_MODE = meta.get('readout_mode', 'anchored'); trainer.PRECISION_MODE = meta.get('precision_mode', 'learned')
model = trainer.SheafModel(meta['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict']); model.eval()
assert trainer.READOUT_MODE == 'posterior_only'
prior = trainer.LATENT_PRIOR / (n * 64)


def feats_of(P, E):
    return torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1)


def decode(feats):
    with torch.no_grad():
        return trainer.predict(model.decoder, torch.cat((feats, feats), -1)).cpu().numpy().reshape(-1, size, size)


def join(p_list, e_list, v_list):
    P = torch.stack(p_list); E = torch.stack(e_list); V = torch.tensor(v_list, dtype=torch.float64, device=P.device)[:, None]
    idx = (P + V * 1e-9).argmax(0, keepdim=True)
    return P.gather(0, idx)[0], E.gather(0, idx)[0]


# ----------------------------------------------------------------------------- world and random walk
spec = d293.world_spec(args.split, args.index, args.rooms)
geo = d293.rasterize(spec, size); walls = geo['walls'].astype(np.uint8)
cell = d293.EXTENT / size
start, assignment = d293.metric_trajectory(spec, n)
pos = start[1].copy()
rng = np.random.default_rng(args.seed)
regions = []
for i in range(n):
    if assignment[i] >= 0:
        x0, y0, x1, y1 = spec['rooms'][assignment[i]]['rectangle_m']
        regions.append((x0 + .45, y0 + .40, x1 - .45, y1 - .40))
    else:
        cx0, cy0, cx1, cy1 = spec['corridor_rectangle_m']
        regions.append((cx0 + .3, max(cy0 + .3, pos[i, 1] - 4.), cx1 - .3, min(cy1 - .3, pos[i, 1] + 4.)))


def inside(r, p):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def free(p):
    r, c = int(p[1] // cell), int(p[0] // cell)
    return 0 <= r < size and 0 <= c < size and walls[r, c] == 0


heading = rng.uniform(0, 2 * math.pi, n)
paths = [pos.copy()]
for t in range(1, T):
    for i in range(n):
        for attempt in range(30):
            th = heading[i] + rng.normal(0, .45) + (math.pi + rng.normal(0, .8) if attempt else 0.)
            L = rng.uniform(.7, 1.3) * args.step
            p = pos[i] + L * np.array([math.cos(th), math.sin(th)])
            if inside(regions[i], p) and free(p):
                heading[i] = th % (2 * math.pi); pos[i] = p; break
    paths.append(pos.copy())
paths = np.stack(paths)                                                            # [T,N,2] metres

# ----------------------------------------------------------------------------- observations (v293 sensor, cumulative)
rays = make_rays(d293.SENSOR_RANGE / cell, 256)
axis = (np.arange(size, dtype=np.float64) + .5) * cell
maps = np.full((T, n, size, size), -1, np.int8); current = np.full((n, size, size), -1, np.int8)
for t in range(T):
    rc = np.floor(paths[t][:, ::-1] / cell).astype(np.int64)
    observed = np.full_like(current, -1); raycast_observe(walls, rc, observed, rays)
    for node, (x, y) in enumerate(paths[t]):
        observed[node, (axis[None] - x) ** 2 + (axis[:, None] - y) ** 2 > d293.SENSOR_RANGE ** 2] = -1
    current = np.maximum(current, observed); maps[t] = current

# ----------------------------------------------------------------------------- simulation: codes, gossip, memory, decode
mem_p = [torch.full((64,), prior, dtype=torch.float64, device=device) for _ in range(n)]
mem_e = [torch.zeros(64, dtype=torch.float64, device=device) for _ in range(n)]
V = -np.ones((n, n), dtype=int); last_sent = {}
bytes_total = 0; packets = 0; events = 0
own_pred = np.zeros((T, n, size, size), np.int8); fused_pred = np.zeros_like(own_pred)
f1_own = np.full((T, n), np.nan); f1_fused = np.full((T, n), np.nan); links = {}; status = []; known_src = np.zeros((T, n), int)
t0 = time.time()
with torch.no_grad():
    for t in range(T):
        own = torch.from_numpy(maps[t]).to(device)
        mu, q, mask = model.source(own)
        evid = mask.double() * q.double(); p_own = evid + prior; e_own = evid * mu.double()
        pw, ew, _ = forest.quantize(p_own, e_own)
        for i in range(n):
            V[i, i] = t
            mem_p[i], mem_e[i] = join([mem_p[i], pw[i]], [mem_e[i], ew[i]], [0, t])
        if t > 0 and t % K == 0:
            adj = radius_adjacency(paths[t], args.radio)
            directed = [(i, j) for i in range(n) for j in range(n) if adj[i, j]]
            out = {}
            for (i, j) in directed:
                codes = (mem_p[i].cpu().numpy().round(12), mem_e[i].cpu().numpy().round(12), V[i].copy())
                changed = int(((codes[0] != last_sent[(i, j)][0]) | (codes[1] != last_sent[(i, j)][1])).sum()) if (i, j) in last_sent else 64
                vchanged = (i, j) not in last_sent or bool((codes[2] != last_sent[(i, j)][2]).any())
                packets += 1
                bytes_total += (8 + 8 + 2 * changed + n) if changed else ((8 + n) if vchanged else 0)
                last_sent[(i, j)] = codes
                out[(i, j)] = (mem_p[i].clone(), mem_e[i].clone(), V[i].copy())
            for j in range(n):
                inc = [k for k in directed if k[1] == j]
                if inc:
                    mem_p[j], mem_e[j] = join([mem_p[j]] + [out[k][0] for k in inc], [mem_e[j]] + [out[k][1] for k in inc], [0] * (1 + len(inc)))
                    for k in inc:
                        V[j] = np.maximum(V[j], out[k][2])
            events += 1
            links[t] = [(i, j) for (i, j) in directed if i < j]
        known_src[t] = (V >= 0).sum(1)
        own_pred[t] = decode(feats_of(p_own, e_own))
        fused_pred[t] = decode(feats_of(torch.stack(mem_p), torch.stack(mem_e)))
        target = np.full_like(maps[t], -1)
        for i in range(n):
            for j in range(n):
                if V[i, j] >= 0:
                    target[i] = np.maximum(target[i], maps[V[i, j]][j])
        for i, r in enumerate(node_metrics(fused_pred[t], maps[t], target, {'door_mask': geo['door_mask']})):
            f1_fused[t, i] = np.nan if r['wall_f1'] is None else r['wall_f1']
        for i, r in enumerate(node_metrics(own_pred[t], maps[t], maps[t], {'door_mask': geo['door_mask']})):
            f1_own[t, i] = np.nan if r['wall_f1'] is None else r['wall_f1']
        status.append(dict(t=t, events=events, packets=packets, bytes=bytes_total, sources_known=float((V >= 0).sum(1).mean()),
                           f1_own=float(np.nanmean(f1_own[t])), f1_fused=float(np.nanmean(f1_fused[t])), target_known=float((target >= 0).mean())))
        if t % 10 == 0 or t == T - 1:
            print(f"t={t:3d} events {events} packets {packets} bytes {bytes_total} | own F1 {status[-1]['f1_own']:.3f} fused F1 {status[-1]['f1_fused']:.3f} | {time.time()-t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- export (RLE ternary maps + json)
import gzip


def rle(arr):
    """int8 map in {-1,0,1} -> bytes of (value+1, run) pairs, run <= 255."""
    flat = np.asarray(arr, np.int8).reshape(-1) + 1
    out = bytearray()
    i = 0; L = len(flat)
    while i < L:
        v = flat[i]; j = i
        while j < L and flat[j] == v and j - i < 255:
            j += 1
        out += bytes((int(v), j - i)); i = j
    return bytes(out)


blob = bytearray(); offsets = {}
def put(key, arr):
    b = rle(arr); offsets[key] = [len(blob), len(b)]; blob.extend(b)
put('walls', walls.astype(np.int8))                                   # 0 free, 1 wall (stored as values 1/2 after +1)
for t in range(T):
    for i in range(n):
        put(f'own/{t}/{i}', maps[t, i]); put(f'fused/{t}/{i}', fused_pred[t, i]); put(f'owndec/{t}/{i}', own_pred[t, i])
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(str(out) + '.bin', 'wb') as f:
    f.write(bytes(blob))
with gzip.open(str(out) + '.bin.gz', 'wb', compresslevel=9) as f:
    f.write(bytes(blob))
meta_out = dict(version=390, world_id=spec['world_id'], rooms=args.rooms, index=args.index, split=args.split, agents=n, T=T, comm_every=K, radio=args.radio,
                step_m=args.step, seed=args.seed, size=size, extent_m=d293.EXTENT, sensor_range_m=d293.SENSOR_RANGE, checkpoint=args.checkpoint,
                checkpoint_sha256=trainer.sha_file(args.checkpoint), assignment=assignment.tolist(), paths_m=np.round(paths, 3).tolist(),
                links={str(k): v for k, v in links.items()}, status=status, f1_own=np.round(np.nan_to_num(f1_own, nan=-1), 4).tolist(),
                f1_fused=np.round(np.nan_to_num(f1_fused, nan=-1), 4).tolist(), known_src=known_src.tolist(), offsets=offsets, bin_bytes=len(blob))
json.dump(meta_out, open(str(out) + '.json', 'w'), separators=(',', ':'))
print('exported', str(out), 'bin', len(blob), 'gz', Path(str(out) + '.bin.gz').stat().st_size, f'{time.time()-t0:.0f}s')
