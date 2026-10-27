"""A04 probe P3c (Claude, 2026-10-27): protocol-level bytes per house of the CURRENT codes under three wire protocols (no model needed:
which stalks are non-bottom is decided by coverage, as in the trainer's owned semantics; payload 16 bit / stalk at 8+8, or 11.1 bit / stalk
= the measured joint entropy of (mu, q) codes of the 2D 8-bit model (report D3-6), plus a coded presence / beat mask).
  fixed : every round, every directed edge sends D x 16 bit (what rollout_max does for 8 rounds)                  = 8 x 2|E| x 2,048 B
  sparse: each directed edge sends its CONVERGED message once, non-bottom stalks only (idempotent join: re-sending is redundant)
  delta : as sparse, but only stalks whose coverage key beats the receiver-side key (the sender knows it: the receiver's own message)
for layouts 32x32x1 and 64x64x1; the presence / beat mask is costed at its binary entropy (ideal) per message.
Reference: lossless ctx once per directed edge (P3 'one_full_per_directed_edge')."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a04_common import houses, load, tree_sides, HERE  # noqa: E402


def h2(p):
    return 0. if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


p3 = json.load(open(os.path.join(HERE, 'p3_delta_edges.json')))['summary']['sweeps']
lossless = {s['house']: s['one_full_per_directed_edge'] for s in p3}
out = []
for path in houses():
    m, own, pred, gt = load(path); n = own.shape[0]
    edges = [tuple(e) for e in m['edges']]; sides = tree_sides(n, edges)
    row = dict(house=m['house'], directed_edges=len(sides), lossless_ctx_once=lossless[m['house']])
    for name, g in (('32', 32), ('64', 64)):
        s = 128 // g; D = g * g
        cov = (own >= 0).reshape(n, g, s, g, s, 32).sum((2, 4, 5))
        fixed = 8 * len(sides) * D * 2
        sp16 = sp11 = de16 = de11 = 0.
        nsp = nde = 0
        for (snd, rcv), side_s in sides.items():
            ks = cov[side_s].max(0); kr = cov[sides[(rcv, snd)]].max(0)
            obs = int((ks > 0).sum()); beat = int((ks > kr).sum())
            mask_bits = D * h2(obs / D); beat_bits = D * h2(beat / D)
            sp16 += (obs * 16 + mask_bits) / 8 + 2; sp11 += (obs * 11.1 + mask_bits) / 8 + 2
            de16 += (beat * 16 + beat_bits) / 8 + 2; de11 += (beat * 11.1 + beat_bits) / 8 + 2
            nsp += obs; nde += beat
        row[name] = dict(fixed=fixed, sparse16=sp16, sparse11=sp11, delta16=de16, delta11=de11,
                         mean_msg_sparse16=sp16 / len(sides), mean_msg_delta11=de11 / len(sides), stalks_sparse=nsp, stalks_delta=nde)
    out.append(row)
S = {}
for name in ('32', '64'):
    S[name] = {k: float(np.mean([r[name][k] for r in out])) for k in ('fixed', 'sparse16', 'sparse11', 'delta16', 'delta11', 'mean_msg_sparse16', 'mean_msg_delta11')}
    S[name]['fixed_over_lossless'] = float(np.mean([r[name]['fixed'] / r['lossless_ctx_once'] for r in out]))
    S[name]['lossless_over_delta11'] = float(np.mean([r['lossless_ctx_once'] / r[name]['delta11'] for r in out]))
    S[name]['fixed_over_delta11'] = float(np.mean([r[name]['fixed'] / r[name]['delta11'] for r in out]))
S['lossless_ctx_once_mean'] = float(np.mean([r['lossless_ctx_once'] for r in out]))
print(json.dumps(S, indent=1))
json.dump(dict(summary=S, rows=out), open(os.path.join(HERE, 'p3c_protocol_bytes.json'), 'w'), indent=1)
