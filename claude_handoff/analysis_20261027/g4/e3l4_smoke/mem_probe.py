"""E3-L4 GPU-memory estimate for the 5090 (32 GB), CPU only: bytes of the tensors autograd saves for backward, per encoded agent and per
decoded sample, on one full-resolution TRAIN house of the local smoke cache (128 x 128 x 32), width 16 (= E3-L3 / E3-L4). Report only."""
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342')
sys.path.insert(0, os.path.abspath(SC))
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g4_e3l4 as M  # noqa: E402

L3, L, R = M.L3, M.L, M.R
torch.set_num_threads(2)
cache = os.path.join(HERE, '..', 'e3l_smoke', 'cache')
man = json.load(open(os.path.join(cache, 'manifest.json')))
tr = L.load_cache(cache, man, 'train', torch.device('cpu'))
H = tr[0]
fit = os.path.join(HERE, '..', '..', 'review_E3R', 'inputs', 'e1c_fit.json')
ev = R.Evidence.from_spec(f'c:{fit}:A:cam_nf')
tb = torch.as_tensor(ev.table, dtype=torch.int32)
M.configure(4.0)
torch.manual_seed(342101)
model = L3.E3L3(16, L3.W0)
tot = {'n': 0}
seen = set()


def pack(t):
    """count each saved storage once (a tensor saved by two ops, e.g. a ReLU output and the next conv's input, is one buffer)"""
    try:
        key = t.untyped_storage().data_ptr()
    except Exception:
        key = id(t)
    if key not in seen:
        seen.add(key); tot['n'] += t.untyped_storage().nbytes()
    return t


def run(m_agents):
    tot['n'] = 0
    sub = tuple(range(min(m_agents, H.n)))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        e, kn = L.evidence_t(H.t_maps[1][list(sub)], tb)
        f = R.features_t(e, kn.to(torch.int32))
        zhat, lam = model.encode(f, kn)
        enc = tot['n']
        h, lq, z = L3.source(zhat, lam, 'ste')
        tot['n'] = 0
        lg = model.decode(h[:1], lq[:1])                      # decoder cost does not depend on GaBP (one belief per decoded sample)
        dec = tot['n']
    return enc / len(sub), dec


print('grid', tuple(H.t_maps.shape[-3:]), 'agents in house', H.n, flush=True)
per_agent, per_dec = run(1)
print(f'saved for backward: encoder {per_agent / 2 ** 30:.3f} GiB per agent, decoder {per_dec / 2 ** 30:.3f} GiB per decoded sample', flush=True)
for n_agents in (24, 40, 64):
    est = n_agents * per_agent + 8 * per_dec
    print(f'  batch 8, {n_agents} encoded agents (padded to a multiple of 8): ~{est / 2 ** 30:.1f} GiB of saved activations (+ cudnn workspace, params, optimizer)', flush=True)
