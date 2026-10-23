"""Demo v385 (Claude, 2026-10-23): agents random-walk inside their own local region of a sealed v293 world, observe with the v293
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
ap.add_argument('--fps', type=int, default=5); ap.add_argument('--stills', default='0,12,36,89')
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
f1_own = np.full((T, n), np.nan); f1_fused = np.full((T, n), np.nan); links = {}; status = []
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

# ----------------------------------------------------------------------------- rendering
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9,
                     'axes.linewidth': .5, 'text.color': '#222222'})
CMAP_T = ListedColormap(['#e6e6e6', '#ffffff', '#262626'])       # unknown, free, wall
CMAP_W = ListedColormap(['#ffffff', '#d2d2d2'])          # free, occupied (ground truth; light so trajectories stand out)
COLORS = ['#3B6FB6', '#D1495B', '#2A9D8F', '#E9A23B', '#8E6BB8', '#5CB5E1', '#C76A2F', '#6AA84F', '#A0522D', '#4682B4', '#9ACD32', '#DA70D6', '#708090', '#FF7F50', '#20B2AA', '#B8860B']
WF, HF = 20., 8.4


def frame_axes(ax):
    for s in ax.spines.values():
        s.set_edgecolor('#c8c8c8'); s.set_linewidth(.5)
    ax.set_xticks([]); ax.set_yticks([])


def render(t):
    fig = plt.figure(figsize=(WF, HF), dpi=100)
    fig.patch.set_facecolor('white')
    fig.text(.03, .952, 'Sheaf-GBP multi-agent mapping', fontsize=14, fontweight='bold', color='#111111')
    fig.text(.03, .922, f'{n} agents, each limited to 64 Gaussian variables and 64 precisions. Neighbours within {args.radio:.0f} m exchange their '
                        f'sheaf vectors every {K} steps; the decoder reads only the fused posterior.', fontsize=9, color='#666666')
    # ---- a: ground truth, trajectories, links
    axa = fig.add_axes([.03, .10, .345, .78])
    axa.imshow(walls, cmap=CMAP_W, vmin=0, vmax=1, origin='lower', extent=(0, 20, 0, 20), interpolation='nearest')
    active = [s for s in links if t - 1 <= s <= t]
    for s in active:
        for (i, j) in links[s]:
            axa.plot([paths[s, i, 0], paths[s, j, 0]], [paths[s, i, 1], paths[s, j, 1]], color='#9a9a9a', lw=.9, alpha=.9, zorder=3)
    for i in range(n):
        c = COLORS[i]
        axa.add_patch(Circle(paths[t, i], d293.SENSOR_RANGE, fc='none', ec=c, lw=.5, alpha=.22, zorder=1))
        axa.plot(paths[:t + 1, i, 0], paths[:t + 1, i, 1], color=c, lw=1.0, alpha=.85, solid_capstyle='round', zorder=2)
        axa.scatter([paths[t, i, 0]], [paths[t, i, 1]], s=48, c=c, edgecolors='white', linewidths=.8, zorder=5)
        axa.text(paths[t, i, 0] + .35, paths[t, i, 1] + .35, f'A{i + 1}', color=c, fontsize=8, fontweight='bold', zorder=6)
    axa.plot([14.2, 19.2], [.3, .3], color='#333333', lw=1.2, solid_capstyle='butt'); axa.text(13.8, .3, '5 m', ha='right', va='center', fontsize=8, color='#333333')
    axa.set_xlim(0, 20); axa.set_ylim(0, 20); frame_axes(axa)
    fig.text(.03, .893, 'a', fontsize=12, fontweight='bold'); fig.text(.05, .893, 'Ground truth (grey = occupied), trajectories and radio links', fontsize=10)
    # ---- b / c panels
    x0, pitch, w, h, yb, yc = .405, .0745, .07, .1667, .655, .385
    fig.text(x0 - .005, yb + h + .022, 'b', fontsize=12, fontweight='bold'); fig.text(x0 + .015, yb + h + .022, 'Local observation (own sensor only)', fontsize=10)
    fig.text(x0 - .005, yc + h + .022, 'c', fontsize=12, fontweight='bold'); fig.text(x0 + .015, yc + h + .022, 'Map decoded from the fused Sheaf-GBP posterior', fontsize=10)
    for i in range(n):
        c = COLORS[i]
        axb = fig.add_axes([x0 + i * pitch, yb, w, h])
        axb.imshow(maps[t, i] + 1, cmap=CMAP_T, vmin=0, vmax=2, origin='lower', interpolation='nearest')
        axb.scatter([paths[t, i, 0] / cell - .5], [paths[t, i, 1] / cell - .5], s=14, c=c, edgecolors='white', linewidths=.5, zorder=5)
        axb.text(.0, 1.04, f'A{i + 1}', transform=axb.transAxes, color=c, fontsize=8, fontweight='bold', va='bottom')
        frame_axes(axb)
        axc = fig.add_axes([x0 + i * pitch, yc, w, h])
        axc.imshow(fused_pred[t, i] + 1, cmap=CMAP_T, vmin=0, vmax=2, origin='lower', interpolation='nearest')
        axc.text(.0, -.07, f'wall F1 {f1_fused[t, i]:.2f}' if np.isfinite(f1_fused[t, i]) else 'wall F1 n/a', transform=axc.transAxes, fontsize=8, color='#444444', va='top')
        frame_axes(axc)
    st = status[t]
    fig.text(x0, .285, f't = {t:2d}    communication events {st["events"]}    packets {st["packets"]}    payload {st["bytes"] / 1024:.1f} kB    '
                      f'mean wall F1  own {st["f1_own"]:.2f}  ·  fused {st["f1_fused"]:.2f}', fontsize=9, color='#333333')
    for k, (lab, col) in enumerate((('unknown', '#e6e6e6'), ('free', '#ffffff'), ('wall', '#262626'))):
        fig.patches.append(Rectangle((x0 + k * .055, .232), .010, .022, transform=fig.transFigure, fc=col, ec='#c8c8c8', lw=.5))
        fig.text(x0 + k * .055 + .014, .243, lab, fontsize=8, color='#555555', va='center')
    fig.text(x0, .200, 'F1 in c is measured against the union of observations each agent has received so far.', fontsize=8, color='#888888')
    # ---- d: time courses
    fig.text(x0 - .005, .150, 'd', fontsize=12, fontweight='bold'); fig.text(x0 + .015, .150, 'Mean wall F1 and cumulative payload', fontsize=10)
    ts = np.arange(T); fo = np.array([s_['f1_own'] for s_ in status]); ff = np.array([s_['f1_fused'] for s_ in status]); kb = np.array([s_['bytes'] for s_ in status]) / 1024.
    for k, (ax, y, col, lab, lo, hi) in enumerate(((fig.add_axes([x0 + .012, .055, .22, .085]), (fo, ff), ('#9a9a9a', '#1b1b1b'), ('own', 'fused'), .80, 1.0),
                                                    (fig.add_axes([x0 + .30, .055, .22, .085]), (kb,), ('#1b1b1b',), ('payload (kB)',), 0., max(kb.max() * 1.1, 1.)))):
        for yy, cc, ll in zip(y, col, lab):
            ax.plot(ts[:t + 1], yy[:t + 1], color=cc, lw=1.1, label=ll)
            ax.plot(ts[t + 1:], yy[t + 1:], color=cc, lw=1.1, alpha=.18)
        ax.axvline(t, color='#c8c8c8', lw=.6, zorder=0)
        for ev in links:
            ax.plot([ev], [lo + (hi - lo) * .04], marker='|', color='#b0b0b0', ms=4, mew=.6)
        ax.set_xlim(0, T - 1); ax.set_ylim(lo, hi); ax.set_xlabel('step', fontsize=8, labelpad=1, color='#444444'); ax.tick_params(labelsize=7, length=2, width=.5, colors='#444444')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_edgecolor('#999999'); ax.spines[sp].set_linewidth(.5)
        ax.set_ylabel('wall F1' if k == 0 else 'kB', fontsize=8, labelpad=2, color='#444444')
        if k == 0:
            ax.legend(frameon=False, fontsize=7, loc='lower right', ncol=2, handlelength=1.4, borderpad=.2, labelcolor='#444444')
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


import imageio.v2 as imageio
from PIL import Image
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
stills = {int(v) for v in args.stills.split(',') if v}
writer = imageio.get_writer(str(out) + '.mp4', fps=args.fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1, quality=8)
gif_frames = []
for t in range(T):
    img = render(t); writer.append_data(img)
    gif_frames.append(Image.fromarray(img).resize((1200, int(1200 * img.shape[0] / img.shape[1])), Image.LANCZOS))
    if t in stills:
        Image.fromarray(img).save(f'{out}_t{t:02d}.png')
    if t % 10 == 0:
        print(f'rendered {t}/{T} ({time.time()-t0:.0f}s)', flush=True)
writer.close()
gif_frames[0].save(str(out) + '.gif', save_all=True, append_images=gif_frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
json.dump(dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), args=vars(args), world_id=spec['world_id'],
               assignment=assignment.tolist(), status=status, links={str(k): v for k, v in links.items()}), open(str(out) + '.json', 'w'), indent=1)
print('done', str(out) + '.mp4', f'{time.time()-t0:.0f}s')
