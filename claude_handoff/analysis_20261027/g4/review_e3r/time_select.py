"""reviewer timing probe (read-only): cost of select_threshold's CPU-side work per held-out house/draw, network replaced by a
trivial stand-in; plus CPU vs CPU-float64 logits of the real U-Net to gauge the bin-edge sensitivity. Smoke cache only (TRAIN)."""
import os, sys, time, json
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill\scripts\claude_v342'
sys.path.insert(0, HERE)
import numpy as np, torch
import g4_e3r as E
torch.set_num_threads(1)
G = E.G
cache = sys.argv[1]
man = E._jload(os.path.join(cache, 'manifest.json'))
hs = E.load_cache(cache, man, 'holdout') + E.load_cache(cache, man, 'train')
for h in hs:
    print(h.name, 'agents', h.n, 'comps', h.comps, 'draws', h.draws, flush=True)

class SumNet(torch.nn.Module):
    def forward(self, x):
        return 2.0 * x[:, 0] - 0.3 * x[:, 1]

ev = E.Evidence.from_spec('L')
t = time.time(); sel = E.select_threshold(SumNet(), ev, hs, 'cpu', chunk=8); dt = time.time() - t
nd = sum(len(h.draws) - 1 for h in hs)
print(f'select (stand-in net): {dt:.1f}s for {len(hs)} houses, {nd} cam_nf draws -> {dt / nd:.2f}s per house-draw', flush=True)
# U-Net forward cost on CPU, 1 thread, one 128x128x32 map; and float32 vs float64 logit differences near the bin edges
torch.manual_seed(0)
m = E.UNet3(3, 16, skip=True, bias0=0.0).eval()
for p in m.parameters():                                   # perturb the head so the U-Net path is not ~0
    pass
torch.nn.init.normal_(m.head.weight, std=0.3)
H = hs[0]; S, kc = E.direct_sums(list(H.maps[1]), H.comps[0], ev.table)
x = E.features_t(torch.from_numpy(S), torch.from_numpy(kc))[None]
with torch.no_grad():
    t = time.time(); y32 = m(x); t1 = time.time() - t
    m64 = E.UNet3(3, 16, skip=True, bias0=0.0).double().eval(); m64.load_state_dict(m.state_dict())
    y64 = m64(x.double())
d = (y32.double() - y64).abs()
print(f'U-Net CPU forward 1 thread: {t1:.2f}s; |fp32 - fp64| logit: max {float(d.max()):.2e}, p99.9 {float(torch.quantile(d.flatten()[::7], .999)):.2e}', flush=True)
