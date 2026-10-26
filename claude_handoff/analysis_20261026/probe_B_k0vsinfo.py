"""probe_B_k0vsinfo (analyst B): per-house K0 / R8 surface F1 of the 3S-X recipe (and L1 16k) vs information per evidenced 32x32 coordinate (info.json)."""
import json, numpy as np
R = "/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs"
I = {(h["rooms"], h["index"]): h for h in json.load(open("info.json"))["summary"]["per_house"]}
for name in ("v430-stackdata/diag-control", "v429-ladder/diag-l1", "v429-ladder/diag-control"):
    for s in ("s342101", "s342202"):
        P = {(r["rooms"], r["index"]): r for r in json.load(open(f"{R}/{name}-{s}/diag.json"))["per_scene"]}
        k = sorted(set(P) & set(I)); b = np.array([I[x]["bits_per_ev32"] for x in k]); o = np.array([I[x]["obj_share"] for x in k])
        k0 = np.array([P[x]["K0_surface"] for x in k]); k8 = np.array([P[x]["K8_surface"] for x in k])
        q = np.quantile(b, [.25, .75]); lo, hi = b <= q[0], b >= q[1]; sl = np.polyfit(b, k0, 1)
        print(name, s, "K0 lo-quartile bits (%.0f) %.3f | hi-quartile (%.0f) %.3f | slope %.4f per bit | R8 lo %.3f hi %.3f | furn<.1 K0 %.3f n %d" % (b[lo].mean(), k0[lo].mean(), b[hi].mean(), k0[hi].mean(), sl[0], k8[lo].mean(), k8[hi].mean(), k0[o < .1].mean(), (o < .1).sum()))
