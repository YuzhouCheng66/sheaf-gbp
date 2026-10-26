"""probe_A_readout.py (analyst A, 2026-10-26; CPU-only, post-hoc, report-only; NOT a pre-registered criterion).

Is the information for exact (one-voxel) surface placement already present in the decoder's last compute-grid features h
(64 x 64 x 16, 72 channels at W2), and is it lost only by the block readout (1x1 conv + depth_to_space(2))? Freeze the whole trained
model (encoder, fusion, decoder up to h), and train only small READOUT heads on top of h with the trainer's K0 loss, on TRAIN cache
scenes; score K0 exact / tol-1 surface F1 on VAL houses (fixtures from probe_A_errors.py). Every head is RESIDUAL on the original
1x1 readout and zero-initialised, so step 0 reproduces the checkpoint exactly:
  ft1x1 : fine-tune the original 1x1 readout only (control for "a few hundred extra steps on a few scenes")
  c3    : + a 3x3x3 conv 72 -> 16 on the compute grid (sees the 26 neighbouring cells), then depth_to_space
  c3x2  : + conv3 72 -> 72, SiLU, conv3 72 -> 16 on the compute grid (two more layers of depth at the end, no voxel-res conv)
  vox   : + 1x1 72 -> 64 -> depth_to_space -> 8 ch at VOXEL resolution -> SiLU -> conv3 8 -> 16 -> SiLU -> 1x1 16 -> 2
If c3 / c3x2 / vox gain exact F1 well beyond ft1x1, the readout is binding (a); if none gains, the misplacement is already in h
(upstream: start grid / depth / code).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F

SCRIPTS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
RUNS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/runs'
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fx', required=True, help='VAL fixture dir from probe_A_errors.py')
ap.add_argument('--run', default=f'{RUNS}/v429-stack/c2-s342101')
ap.add_argument('--cache', default='/mnt/datasets/yuzhou-sheaf-gbp-research/cache3d-v420/train-r1-k1600')
ap.add_argument('--train-scenes', type=int, default=8); ap.add_argument('--steps', type=int, default=240); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--heads', default='ft1x1,c3,vox,c3x2'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()
sys.path.insert(0, args.source); sys.path.insert(0, os.path.join(args.source, 'exploration_grid')); sys.path.insert(0, SCRIPTS)
os.environ['SEALED_SOURCE'] = args.source
import train_3d_v431 as T
import mapping_3d_storey_data_v420 as W


def configure(ck):                                                     # same as probe_A_errors.configure
    m = ck['meta']
    T.DECODER = m.get('decoder', 'd3'); T.PRECISION = m.get('precision', 'learned'); T.KNOWN_WEIGHT = m.get('known_weight', 'uniform')
    T.BX, T.BY, T.BZ = m['blocks']; T.D = T.BX * T.BY * T.BZ
    T.WIRE_BITS = tuple(m.get('wire_bits', [8, 8])); T._v378.MU_BITS, T._v378.Q_BITS = T.WIRE_BITS; T.LEVELS = 2 ** (T.WIRE_BITS[0] - 1) - 1
    T.DECODER_WIDTH = m.get('decoder_width', 1.0)


class Head(nn.Module):
    def __init__(self, kind, out_conv):
        super().__init__()
        self.kind = kind; c = out_conv.in_channels
        self.base = nn.Conv3d(c, out_conv.out_channels, 1); self.base.load_state_dict(out_conv.state_dict())
        self.base.requires_grad_(kind == 'ft1x1')
        if kind == 'c3':
            self.delta = nn.Conv3d(c, 16, 3, padding=1); nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        elif kind == 'c3x2':
            self.delta = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.SiLU(), nn.Conv3d(c, 16, 3, padding=1)); nn.init.zeros_(self.delta[2].weight); nn.init.zeros_(self.delta[2].bias)
        elif kind == 'vox':
            self.up = nn.Conv3d(c, 64, 1); self.ref = nn.Sequential(nn.SiLU(), nn.Conv3d(8, 16, 3, padding=1), nn.SiLU(), nn.Conv3d(16, 2, 1))
            nn.init.zeros_(self.ref[3].weight); nn.init.zeros_(self.ref[3].bias)

    def forward(self, h):
        lo = self.base(h)
        if self.kind in ('c3', 'c3x2'):
            lo = lo + self.delta(h)
        out = T.depth_to_space(lo, 2)
        if self.kind == 'vox':
            out = out + self.ref(T.depth_to_space(self.up(h), 2))
        return out[:, 0].flatten(1), out[:, 1].flatten(1)


def features(model, own, reached, edges, hop=0):
    """h (input of decoder.output) and the K0 target for every node of one scene."""
    store = {}
    hk = model.decoder.output.register_forward_hook(lambda m, i, o: store.__setitem__('h', i[0].detach()))
    mu, q, mask = model.source(own); feats = T.rollout_max(mu, q, mask, edges, stages=(hop,))
    model.decoder(feats[hop]); hk.remove()
    return store['h'], T.stream_targets(own, reached[T.HOPS.index(hop)])


def loss_fn(kl, ol, tgt, dims):
    known = (tgt >= 0).float()
    kb = F.binary_cross_entropy_with_logits(kl, known, reduction='none').mean(-1)
    occ = (tgt == 1).float(); w = known * (1. + 4. * T.surface_of(tgt, dims).float())
    ob = (w * F.binary_cross_entropy_with_logits(ol, occ, reduction='none')).sum(-1) / w.sum(-1).clamp_min(1)
    return kb.mean() + ob.mean()


def main():
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); dims = W.DIMS['R1']; t0 = time.time()
    ck = torch.load(Path(args.run) / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=False); configure(ck)
    model = T.SheafModel3D(dims, ck['meta']['seed']); model.load_state_dict(ck['state_dict']); model.eval(); model.requires_grad_(False)
    man = json.load(open(Path(args.cache) / 'manifest.json'))
    rng = np.random.default_rng(args.seed); rows = [man['rows'][i] for i in rng.choice(len(man['rows']), args.train_scenes, replace=False)]
    train = []
    with torch.no_grad():
        for r in rows:
            z = np.load(Path(args.cache) / r['file']); T.CHART_M = float(z['chart_m'])
            h, tgt = features(model, torch.from_numpy(z['private']), torch.from_numpy(z['reached']), [(int(i), int(j), float(w)) for i, j, w in z['edges']])
            train.append((h.half(), tgt))
        print(f'[A] train feats {len(train)} scenes {time.time()-t0:.0f}s', flush=True)
        val = []
        for f in sorted(Path(args.fx).glob('r*-i*.npz')):
            z = np.load(f)
            if 'skipped' in z.files:
                continue
            T.CHART_M = float(z['chart_m'])
            h, tgt = features(model, torch.from_numpy(z['private']), torch.from_numpy(z['reached']), [(int(i), int(j), float(w)) for i, j, w in z['edges']])
            val.append(dict(name=f.stem, h=h.half(), tgt=tgt, slab=torch.from_numpy(z['slab'].reshape(-1)), obj=torch.from_numpy(z['obj'].reshape(-1)), struct=torch.from_numpy(z['struct'].reshape(-1))))
        print(f'[A] val feats {len(val)} scenes {time.time()-t0:.0f}s', flush=True)

    def evaluate(head):
        res = []
        with torch.no_grad():
            for v in val:
                kl, ol = head(v['h'].float()); pred = torch.where(kl >= 0, (ol >= 0).to(torch.int8), torch.full_like(ol, -1, dtype=torch.int8))
                a = T.agg(T.metrics3d(pred, v['tgt'], dims, v['slab'], v['obj'], v['struct'])); res.append(a)
        return {m: float(np.mean([r[m] for r in res if r[m] is not None])) for m in ('surface_f1', 'surface_f1_tol1', 'var_occ_f1', 'furn_f1')}

    results = {}
    base = Head('ft1x1', model.decoder.output); results['orig'] = evaluate(base)
    print(f"[A] orig {results['orig']} {time.time()-t0:.0f}s", flush=True)
    for kind in args.heads.split(','):
        torch.manual_seed(args.seed); head = Head(kind, model.decoder.output)
        params = [p for p in head.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=args.lr if kind != 'ft1x1' else args.lr * .3)
        order = np.random.default_rng(args.seed + 1); ts = time.time(); losses = []
        for step in range(args.steps):
            h, tgt = train[order.integers(len(train))]
            kl, ol = head(h.float()); loss = loss_fn(kl, ol, tgt, dims)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
            if step == 0 or (step + 1) % 60 == 0:
                print(f'[A] {kind} step {step+1} loss {np.mean(losses[-60:]):.4f} {time.time()-ts:.0f}s', flush=True)
        results[kind] = dict(val=evaluate(head), params=sum(p.numel() for p in params), train_loss_first=losses[0], train_loss_last60=float(np.mean(losses[-60:])), seconds=time.time() - ts)
        print(f"[A] {kind} {results[kind]}", flush=True)
        json.dump(dict(run=args.run, train_rows=[r['file'] for r in rows], steps=args.steps, results=results), open(out / 'probe_A_readout.json', 'w'), indent=1)
    print(f'[A] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
