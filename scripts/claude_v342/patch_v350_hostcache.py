"""Patch F (2026-10-22 22:30 UTC): keep DYNAMIC training slices (own map + 5-hop targets) in pinned host memory and move each slice to
the device per update. Needed for H256 dynamic fine-tuning (3,552 slices x ~4.7 MB at 256 would add ~17 GB of device cache on top of
the ~18.6 GB H256 working set). Pure data placement: same tensors, same values, same order -> identical loss/gradients."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else 'train_v350_dyn_edge.py'
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
rep("        sc['own_t'] = torch.from_numpy(sc['private']).to(device)\n        sc['tgt'] = {k: torch.from_numpy(sc['targets'][k].reshape(n, -1)).to(device) for k in HOPS}\n",
    "        host = args.dynamic and 't' in sc['spec']                       # dynamic training slice: keep on (pinned) host memory (patch F)\n"
    "        place = (lambda x: x.pin_memory()) if (host and device.type == 'cuda') else ((lambda x: x) if host else (lambda x: x.to(device)))\n"
    "        sc['own_t'] = place(torch.from_numpy(sc['private']))\n"
    "        sc['tgt'] = {k: place(torch.from_numpy(sc['targets'][k].reshape(n, -1))) for k in HOPS}\n"
    "        sc['host'] = host\n")
rep("            sc = train[si]\n            opt.zero_grad(set_to_none=True)\n",
    "            sc = train[si]\n"
    "            if sc.get('host'):\n"
    "                sc = dict(sc, own_t=sc['own_t'].to(device, non_blocking=True), tgt={k: v.to(device, non_blocking=True) for k, v in sc['tgt'].items()})\n"
    "            opt.zero_grad(set_to_none=True)\n")
open(p, 'w').write(s); print('patched', p)
