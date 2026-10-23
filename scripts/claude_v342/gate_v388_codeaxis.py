"""Zero-cost pre-launch gate for v388 (code-level embedding at the decoder input). Claude, 2026-10-23, evaluation only, CPU.
Three measurements on the batch-S best weights (v384), written BEFORE the measurements (reading rules at the bottom):
(a) LEVEL SWEEP: for sampled (VAL scene, agent, own-observed block) triples, replace that block's hard-8 level by each of the 255
    codebook values (all other coordinates and the log-variance plane unchanged, K0 context), decode, and count how many DISTINCT
    16x16 wall patterns the block produces (exact, and with a Hamming tolerance of 2 px). A smooth 1-D reading yields few distinct
    patterns; a symbol-like reading yields many.
(b) PERTURBATION: shift ALL own-observed levels by +-1 / +-2 / +-4 / +-8 codebook steps (clamped) and measure the VAL K0 wall-F1
    drop. If +-1 step already costs >= 1 pp, the decoder resolves single levels and a finer code axis has no headroom.
(c) RESIDUAL ANATOMY on the saved v384 VAL/fresh panels: wall F1 of pred_k8 vs target_k8 exact and with 1-px tolerance (dilated
    match), per stratum. If most of the residual is 1-px jitter, a code-axis change cannot help.
Reading rules (pre-registered): launch v388 if mean distinct patterns (Hamming > 2) < 128 AND the +-1-step K0 cost < 1 pp AND the
exact-vs-tolerant F1 gap is < half of the residual (1 - F1_exact); abort if distinct >= 200 or +-1 cost >= 1 pp; otherwise launch with
the expectation downgraded to +1 pp.
"""
import argparse, json, sys, time, importlib.util
from pathlib import Path
import numpy as np, torch
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--run', required=True, help='run dir with validation/*-panel-*.npz'); ap.add_argument('--out', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--scenes-per-stratum', type=int, default=2); ap.add_argument('--agents-per-scene', type=int, default=2); ap.add_argument('--blocks-per-agent', type=int, default=6)
args = ap.parse_args()
torch.set_num_threads(args.threads)
src = Path(args.source).resolve()
for p in (str(Path(__file__).resolve().parent), str(src / 'exploration_grid'), str(src)):
    sys.path.insert(0, p)
spec_ = importlib.util.spec_from_file_location('trainer', args.trainer); trainer = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(trainer)
import mapping_houseexpo_data_v371 as data
from scripts.mapping_recognition_metrics_v277 import node_metrics
from scripts.mapping_implicit_model_v192 import latent_quantize
size = args.size
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
trainer.READOUT_MODE = meta['readout_mode']; trainer.PRECISION_MODE = meta['precision_mode']
model = trainer.SheafModel(meta['arm'], size, 0); model.load_state_dict(payload['state_dict']); model.eval()
LEVELS = 127; prior_of = lambda n: trainer.LATENT_PRIOR / (n * 64)


def k0_feats(mu, q, mask, n):
    evid = mask.double() * q.double(); P = evid + prior_of(n); E = evid * mu.double()
    return torch.cat((latent_quantize((E / P).float()), (1. / P).log().float()), -1)


def decode(feats):
    with torch.no_grad():
        return trainer.predict(model.decoder, torch.cat((feats, feats), -1)).numpy().reshape(-1, size, size)


rng = np.random.default_rng(0)
val_specs = [s for s in data.scene_plan('VALIDATION') if s['size'] == size]
by_stratum = {}
for s in val_specs:
    by_stratum.setdefault(s['rooms'], []).append(s)
specs = [s for r in sorted(by_stratum) for s in by_stratum[r][:args.scenes_per_stratum]]
out = dict(checkpoint=args.checkpoint, checkpoint_sha256=trainer.sha_file(args.checkpoint), specs=specs, sweep=[], perturb={}, anatomy={})
t0 = time.time()
# ---------------------------------------------------------------- (a) level sweep and (b) perturbation
shifts = [0, 1, -1, 2, -2, 4, -4, 8, -8]
pert_rows = {s: [] for s in shifts}
p = size // 8
with torch.no_grad():
    for spec in specs:
        fx = data.make_fixture(**spec); own = torch.from_numpy(fx['inputs']['private']); n = own.shape[0]
        mu, q, mask = model.source(own)
        base = k0_feats(mu, q, mask, n); pred0 = decode(base)
        for s in shifts:
            mu_s = torch.where(mask > 0, (mu * LEVELS + s).clamp(-LEVELS, LEVELS) / LEVELS, mu)
            pred = decode(k0_feats(mu_s, q, mask, n))
            m = node_metrics(pred, fx['inputs']['private'], fx['inputs']['private'], {'door_mask': fx['evaluation']['door_mask']})
            pert_rows[s].append(np.mean([r['wall_f1'] for r in m if r.get('wall_f1') is not None]))
        for a in rng.choice(n, size=min(args.agents_per_scene, n), replace=False):
            obs = np.nonzero(mask[a].numpy() > 0)[0]
            for k in rng.choice(obs, size=min(args.blocks_per_agent, len(obs)), replace=False):
                feats = base[a:a + 1].repeat(255, 1)
                feats[:, k] = torch.arange(-LEVELS, LEVELS + 1).float() / LEVELS
                preds = decode(feats)                                               # [255,H,H]
                by, bx = divmod(int(k), 8)
                patch = (preds[:, by * p:(by + 1) * p, bx * p:(bx + 1) * p] == 1).reshape(255, -1)
                exact = len({row.tobytes() for row in patch})
                # Hamming-tolerant distinct count: greedy clustering with radius 2
                reps = []
                for row in patch:
                    if not any((row != r).sum() <= 2 for r in reps):
                        reps.append(row)
                known_frac = float((preds[:, by * p:(by + 1) * p, bx * p:(bx + 1) * p] >= 0).mean())
                out['sweep'].append(dict(spec=spec, agent=int(a), block=int(k), own_level=int(round(float(mu[a, k]) * LEVELS)), distinct_exact=exact, distinct_h2=len(reps), known_frac=known_frac,
                                         wall_px_true=int((fx['inputs']['private'][a, by * p:(by + 1) * p, bx * p:(bx + 1) * p] == 1).sum())))
        print(f"scene r{spec['rooms']} n{spec['agents']} i{spec['index']} done {time.time() - t0:.0f}s", flush=True)
out['perturb'] = {str(s): dict(k0_f1=float(np.mean(v)), delta_pp=float(100 * (np.mean(v) - np.mean(pert_rows[0])))) for s, v in pert_rows.items()}
sw = out['sweep']
out['sweep_summary'] = dict(blocks=len(sw), mean_distinct_exact=float(np.mean([r['distinct_exact'] for r in sw])), mean_distinct_h2=float(np.mean([r['distinct_h2'] for r in sw])),
                            median_distinct_h2=float(np.median([r['distinct_h2'] for r in sw])),
                            distinct_h2_by_stratum={str(r): float(np.mean([x['distinct_h2'] for x in sw if x['spec']['rooms'] == r])) for r in sorted(by_stratum)})
# ---------------------------------------------------------------- (c) residual anatomy on the saved panels
anat = {}
for f in sorted(Path(args.run).glob('validation/final-*-panel-*.npz')):
    z = np.load(f); pred = z['pred_k8']; tgt = z['target_k8']
    for i in range(pred.shape[0]):
        pw = pred[i] == 1; tw = tgt[i] == 1; scope = tgt[i] >= 0
        pw_s, tw_s = pw & scope, tw & scope
        tp = (pw_s & tw_s).sum(); fp = (pw_s & ~tw_s).sum(); fn = (~pw_s & tw_s).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        dt = ndimage.binary_dilation(tw_s, iterations=1); dp = ndimage.binary_dilation(pw_s, iterations=1)
        tp_t = (pw_s & dt).sum(); fp_t = (pw_s & ~dt).sum(); fn_t = (tw_s & ~dp).sum()
        f1_t = 2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1)
        key = f.name.split('-panel-')[0] + ':' + f.name.split('-panel-')[1].split('-n')[0]
        anat.setdefault(key, []).append(dict(f1_exact=float(f1), f1_tol1=float(f1_t), missed_px=int(fn), missed_px_tol1=int(fn_t), false_px=int(fp), false_px_tol1=int(fp_t)))
out['anatomy'] = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in anat.items()}
ex = np.mean([r['f1_exact'] for v in anat.values() for r in v]); tol = np.mean([r['f1_tol1'] for v in anat.values() for r in v])
out['anatomy_summary'] = dict(f1_exact=float(ex), f1_tol1=float(tol), residual_exact_pp=float(100 * (1 - ex)), jitter_share=float((tol - ex) / max(1 - ex, 1e-9)))
# ---------------------------------------------------------------- verdict (pre-registered rules in the docstring)
d2 = out['sweep_summary']['mean_distinct_h2']; c1 = -min(out['perturb']['1']['delta_pp'], out['perturb']['-1']['delta_pp']); js = out['anatomy_summary']['jitter_share']
if d2 >= 200 or c1 >= 1.0:
    verdict = 'ABORT (no headroom on the code axis)'
elif d2 < 128 and c1 < 1.0 and js < 0.5:
    verdict = 'LAUNCH'
else:
    verdict = 'LAUNCH with expectation downgraded to +1 pp'
out['verdict'] = dict(verdict=verdict, mean_distinct_h2=d2, pm1_step_cost_pp=c1, jitter_share=js)
json.dump(out, open(args.out, 'w'), indent=1)
print('SWEEP', out['sweep_summary']); print('PERTURB', out['perturb']); print('ANATOMY', out['anatomy_summary']); print('VERDICT', out['verdict'], f'{time.time() - t0:.0f}s')
