"""README section 8 baselines for the DYNAMIC protocol (CPU only, no model): lossless raw map exchange on the same episodes, same
per-step forests, same R rounds of synchronous flooding and the SAME diff/silence rule as the learned CRDT protocol. Claude 2026-10-22.
Protocols (each node keeps a raw ternary union map of everything it has heard; a packet from i to j carries i's union map):
  raw_full_2bit   : full 128x128 ternary map at 2 bit/cell every round (4096 B + 8 B header) -- the 'fair 2-bit raw' of README 8
  raw_full_zlib   : zlib(level 9) of the full int8 map every round (+8 B)                  -- strong lossless full
  raw_delta_zlib  : zlib of the DELTA image vs the last packet on that directed edge (0 = unchanged, 1/2/3 = new value+1), silent if
                    nothing changed (+8 B header)                                            -- strong lossless incremental (the hard one)
Quality of all three is lossless (F1 = 1 on the memory target) -- unmatched with the learned code; ratios are therefore bytes-at-
unmatched-quality and must be reported as such. Output: total bytes per protocol over the episodes, to be divided by crdt_diff /
max_diff of eval_dynamic_v349 on the same episodes/radio.
python raw_delta_baseline_v349.py --source <sealed> --radio 8 --out <json>"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--radio', type=float, default=8.); ap.add_argument('--indices', default='0-7'); ap.add_argument('--rooms', default='6,12,20'); ap.add_argument('--agents', default='8,16')
ap.add_argument('--rounds', type=int, default=8); ap.add_argument('--T', type=int, default=6); ap.add_argument('--size', type=int, default=128)
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dyn_data_v349 as dyn
from scripts import mapping_native_bank_v306 as bank
from scripts.prepare_houseexpo_pilot_v121 import radius_adjacency
def rng_list(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
HDR = 8; R = args.rounds; T = args.T; size = args.size
tot = dict(raw_full_2bit=0, raw_full_zlib=0, raw_delta_zlib=0, packets=0, silent_delta_packets=0)
per = []
for rooms in [int(v) for v in args.rooms.split(',')]:
    for agents in [int(v) for v in args.agents.split(',')]:
        for index in rng_list(args.indices):
            fx = dyn.make_dynamic_fixture('VALIDATION', index, size=size, rooms=rooms, agents=agents, T=T)
            maps, paths = fx['maps'], fx['paths_m']; n = agents
            union = [np.full((size, size), -1, np.int8) for _ in range(n)]
            last = {}
            b = dict(raw_full_2bit=0, raw_full_zlib=0, raw_delta_zlib=0, packets=0, silent_delta_packets=0)
            for t in range(T):
                adj = radius_adjacency(paths[t], args.radio)
                edges = bank.select_forest(paths[t], adj, float(args.radio))['edges']
                directed = [(i, j) for i, j, _ in edges] + [(j, i) for i, j, _ in edges]
                for i in range(n):
                    union[i] = np.maximum(union[i], maps[t][i])
                for r in range(R):
                    outgoing = {}
                    for (i, j) in directed:
                        payload = union[i]
                        b['packets'] += 1
                        b['raw_full_2bit'] += HDR + size * size // 4
                        b['raw_full_zlib'] += HDR + len(zlib.compress(payload.tobytes(), 9))
                        prev = last.get((i, j))
                        if prev is None:
                            delta = (payload + 2).astype(np.uint8)            # first packet: everything is new (values 1..3)
                        else:
                            delta = np.where(payload != prev, payload + 2, 0).astype(np.uint8)
                        if delta.any():
                            b['raw_delta_zlib'] += HDR + len(zlib.compress(delta.tobytes(), 9))
                        else:
                            b['silent_delta_packets'] += 1
                        last[(i, j)] = payload.copy(); outgoing[(i, j)] = payload.copy()
                    for (i, j), payload in outgoing.items():
                        union[j] = np.maximum(union[j], payload)
            per.append(dict(rooms=rooms, agents=agents, index=index, bytes=b))
            for k in tot:
                tot[k] += b[k]
            print(rooms, agents, index, {k: v for k, v in b.items()}, flush=True)
res = dict(args=vars(args), header_bytes=HDR, totals=tot, per_episode=per,
           note='lossless raw union-map exchange on the same forests/rounds/silence rule as the learned CRDT; quality unmatched (F1=1)')
json.dump(res, open(args.out, 'w'), indent=1)
print('TOTALS', tot)
