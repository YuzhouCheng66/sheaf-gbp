"""reviewer probe: bytes saved for backward by UNet3 (skip) at 128x128x32, batch 1 (CPU), to size batch 8 on the 5090"""
import os, sys, time
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\Users\27118\Desktop\sheaf-gbp\experiments\sheaf-gbp-p6-local-skill\scripts\claude_v342')
import torch
import g4_e3r as E
torch.set_num_threads(2); torch.manual_seed(0)
m = E.UNet3(3, 16, skip=True, bias0=-2.5)
x = torch.randn(1, 3, 128, 128, 32); y = (torch.rand(1, 128, 128, 32) < .1).float(); mk = torch.ones_like(y)
seen = {}
def pack(t):
    seen[(t.data_ptr(), t.numel())] = t.numel() * t.element_size(); return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    t0 = time.time(); loss = E.masked_bce(m(x), y, mk)
loss.backward(); dt = time.time() - t0
saved = sum(seen.values())
print(f'saved-for-backward (unique storages) per sample: {saved / 2**20:.0f} MiB; batch 8 ~ {8 * saved / 2**30:.2f} GiB (+ grads/workspace); CPU fwd+bwd {dt:.1f}s')
