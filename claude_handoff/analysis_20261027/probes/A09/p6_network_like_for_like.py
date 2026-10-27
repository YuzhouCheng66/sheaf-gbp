"""A09 probe 6 (Claude, 2026-10-27, CPU, reads existing JSON only): network bytes per house, LIKE FOR LIKE.
Debate point: several round-1 memos quote "tens of x vs gzip" by comparing an in-network union relay (each directed tree edge carries the
converged sender-side union once) against gzip FLOODING (every own map crosses every tree edge). That mixes a protocol gain (the fusion's
observation-multiplicity saving) with a codec gain. Here every codec is put on the SAME protocol:
  relay   : each directed edge sends its converged sender-side union once (bytes summed over the house's directed edges);
  flood   : every own map is sent over every tree edge of its component, once (sum_a bytes(own_a) x (n_comp - 1)).
Codecs: gzip-9 and ctx ternary (rate_baselines.json), S = observed-occupied set + 24 B float32 poses per agent on the sender side
(p1_surface_rate.json), gravity-filled world W + poses and GT-column-completed world W + poses (p1b_world_interval.json).
Current protocol: 2,048 B x 8 rounds x directed edges (fixed), and 2,048 B x directed edges (send once).
Output: probes/A09/p6_network_like_for_like.json"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
rb = json.load(open(os.path.join(ROOT, 'rate_baselines.json')))['raw']
p1 = json.load(open(os.path.join(HERE, 'p1_surface_rate.json')))['rows']
p1b = json.load(open(os.path.join(HERE, 'p1b_world_interval.json')))['rows']
POSE_B = 24
houses = rb['houses']
out = {}
for h in houses:
    E = [r for r in rb['edge'] if r['house'] == h]
    E1 = [r for r in p1['edge'] if r['house'] == h]
    E2 = [r for r in p1b['edge'] if r['house'] == h]
    assert len(E) == len(E1) == len(E2), h
    for a, b, c in zip(E, E1, E2):                                       # same loop order in all three scripts
        assert a['sender'] == b['sender'] == c['sender'] and a['side'] == b['side'] == c['side'], h
        assert abs(a['ctx'] - b['ctx_ternary']) < 1e-6 and abs(b['ctx_S'] - c['S']) < 1e-6, h
    own = [r for r in rb['own'] if r['house'] == h]
    n = len(own); nd = len(E)
    n_edges = nd // 2
    comp = n_edges + 1 if n_edges == n - 1 else None                      # demo forests: check connectivity
    flood_mult = n - 1 if comp == n else None
    poses_on = lambda r: POSE_B * r['side']
    rec = dict(agents=n, directed_edges=nd,
               current_fixed_8rounds=2048 * 8 * nd, current_send_once=2048 * nd,
               gzip_flood=sum(r['gzip'] for r in own) * flood_mult if flood_mult else None,
               ctx_flood=sum(r['ctx'] for r in own) * flood_mult if flood_mult else None,
               gzip_relay=sum(r['gzip'] for r in E), ctx_relay=sum(r['ctx'] for r in E),
               S_pose_relay=sum(b['ctx_S'] + poses_on(b) for b in E1),
               gravW_pose_relay=sum(c['grav'] + poses_on(c) for c in E2),
               oracleW_pose_relay=sum(c['gt_cols'] + poses_on(c) for c in E2))
    out[h] = rec
keys = [k for k in out[houses[0]] if k not in ('agents', 'directed_edges')]
mean = {k: float(np.mean([out[h][k] for h in houses if out[h][k] is not None])) for k in keys}
ratios = {}
for ex in ('S_pose_relay', 'gravW_pose_relay', 'oracleW_pose_relay'):
    ratios[ex] = {base: float(np.median([out[h][base] / out[h][ex] for h in houses if out[h][base] is not None]))
                  for base in ('current_fixed_8rounds', 'current_send_once', 'gzip_flood', 'gzip_relay', 'ctx_flood', 'ctx_relay')}
ratios['protocol_gain_gzip_flood_over_relay'] = float(np.median([out[h]['gzip_flood'] / out[h]['gzip_relay'] for h in houses]))
ratios['protocol_gain_ctx_flood_over_relay'] = float(np.median([out[h]['ctx_flood'] / out[h]['ctx_relay'] for h in houses]))
res = dict(per_house=out, mean_bytes_per_house=mean, median_ratios=ratios)
print(json.dumps(dict(mean_bytes_per_house=mean, median_ratios=ratios), indent=1))
json.dump(res, open(os.path.join(HERE, 'p6_network_like_for_like.json'), 'w'), indent=1)
