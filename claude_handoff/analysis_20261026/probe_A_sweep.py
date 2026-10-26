"""probe_A_sweep.py (analyst A, 2026-10-26; CPU-only, synthetic, report-only).
(c) vs (d): does the CODE carry the within-column position of a boundary, and does the DECODER place it? One agent, L = 16 m,
a fully known disc (r = 40 voxels, floor z = 0, ceiling z = 26, free between). A boundary plane at x = xa sweeps the 4 voxel
positions of one 32x32 code column (xa % 4 = 0..3), for two objects:
  wall : occupied plane x = xa (|y - 64| < 40, z 1..25), everything at x > xa unknown (a boundary wall)
  box  : furniture box [xa, xa+8) x [60, 68) x [1, 7) (0.6 m high); its 6-surface occupied, its interior unknown
Reports per position: the code (mu in hard-8 levels, q) of the code columns left of / containing the boundary, and the decoder's
K0 placement: fraction of the boundary pairs (x = xa-1 free, x = xa occupied) decoded exactly, and the mean signed offset of the
first predicted-occupied voxel along +x from xa (rows y 61..66, z 2..5 for the box; y 50..77, z 3..23 for the wall).
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch

SCRIPTS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
RUNS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True)
ap.add_argument('--runs', default=f'{RUNS}/v429-stack/c2-s342101,{RUNS}/v429-stack/c2-s342202'); ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, SCRIPTS)
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v431 as T


def configure(ck):
    m = ck['meta']
    T.DECODER = m.get('decoder', 'd3'); T.PRECISION = m.get('precision', 'learned'); T.KNOWN_WEIGHT = m.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = m['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(m.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = m.get('decoder_width', 1.0)


def scene(kind, xa):
    X, Y, Z = 128, 128, 32
    g = np.full((X, Y, Z), -1, np.int8)
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij'); disc = (xx - 64) ** 2 + (yy - 64) ** 2 < 40 ** 2
    g[disc, 0] = 1; g[disc, 26] = 1
    for z in range(1, 26):
        g[disc, z] = 0
    if kind == 'wall':
        band = np.abs(np.arange(Y) - 64) < 40
        g[xa, band, 1:26] = 1; g[xa + 1:, :, :] = -1
        rows = (slice(50, 78), slice(3, 24))
    else:
        B = np.zeros((X, Y, Z), bool); B[xa:xa + 8, 60:68, 1:7] = True
        free = ~B & (g == 0)
        nb = np.zeros_like(B)
        for ax in range(3):
            for s in (1, -1):
                nb |= np.roll(free, s, axis=ax)
        g[B & nb] = 1; g[B & ~nb] = -1
        rows = (slice(61, 67), slice(2, 6))
    return g, rows


def main():
    torch.set_num_threads(args.threads); dims = (128, 128, 32); T.CHART_M = 16.0
    for run in args.runs.split(','):
        ck = torch.load(Path(run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); configure(ck)
        model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval()
        print(f'== {Path(run).name}')
        with torch.no_grad():
            for kind, base in (('wall', 96), ('box', 80)):
                for j in range(4):
                    xa = base + j; g, (ys, zs) = scene(kind, xa)
                    own = torch.from_numpy(g)[None]
                    mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, [], stages=(0,))
                    pred = T.predict(model.decoder, feats[0]).reshape(dims).numpy()
                    yc = (ys.start + ys.stop) // 2 // 4
                    m = (mu.reshape(32, 32) * T.LEVELS).round().int().numpy(); qq = (q * mask).reshape(32, 32).numpy()
                    cx = xa // 4
                    codes = ' '.join(f'c{c}:{m[c, yc]:+4d}/{qq[c, yc]:.2f}' for c in (cx - 1, cx, cx + 1))
                    sub = pred[:, ys, zs]                                         # [X, ny, nz]
                    exact = float(((sub[xa] == 1) & (sub[xa - 1] != 1)).mean())
                    first = []
                    for a in range(sub.shape[1]):
                        for b in range(sub.shape[2]):
                            hit = np.nonzero(sub[xa - 6:xa + 5, a, b] == 1)[0]
                            first.append(hit[0] - 6 if len(hit) else np.nan)
                    first = np.array(first, float)
                    vet = float((sub[xa] == -1).mean())
                    print(f'  {kind:4s} xa {xa} (xa%4={xa % 4}) exact {exact:.2f} offset {np.nanmean(first):+.2f} (sd {np.nanstd(first):.2f}, none {np.isnan(first).mean():.2f}) vetoed {vet:.2f} | code {codes}', flush=True)


if __name__ == '__main__':
    main()
