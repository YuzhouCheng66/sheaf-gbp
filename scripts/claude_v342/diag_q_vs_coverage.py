"""Does the trained owned encoder's learned precision q_ik track block coverage (fraction of block cells the agent observed)?
Inference-only on VALIDATION H128. Usage: python diag_q_vs_coverage.py --source <sealed> --checkpoint <final.pt> --trainer <train_v342_owned_v2.py>"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import importlib.util
ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--size', type=int, default=128); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', default='')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid'))
spec = importlib.util.spec_from_file_location('trainer', args.trainer); trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
from scripts import mapping_multiroom_data_v293 as data
from scripts import train_mapping_support_native_v306 as base
import torch.nn.functional as F
device = torch.device(args.device); size = args.size
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model = trainer.SheafModel(payload['meta']['arm'], size, 0).to(device); model.load_state_dict(payload['state_dict']); model.eval()
cov, qs, wallfrac = [], [], []
with torch.no_grad():
    for s in [x for x in base.scene_plan('VALIDATION') if x['size'] == size]:
        own = torch.from_numpy(data.make_fixture(**s)['inputs']['private']).to(device)
        mu, q, mask = model.source(own)
        p = size // 8
        c = F.avg_pool2d((own >= 0).float()[:, None], p).flatten(1)          # coverage fraction per block
        w = F.avg_pool2d((own == 1).float()[:, None], p).flatten(1)           # wall fraction per block
        sel = mask > 0
        cov.append(c[sel].cpu().numpy()); qs.append(q[sel].cpu().numpy()); wallfrac.append(w[sel].cpu().numpy())
cov, qs, wallfrac = map(np.concatenate, (cov, qs, wallfrac))
from scipy.stats import spearmanr
rho_c = spearmanr(cov, qs).correlation; rho_w = spearmanr(wallfrac, qs).correlation
bins = [0, .05, .1, .25, .5, .75, 1.0001]
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (cov >= lo) & (cov < hi)
    if m.sum():
        rows.append(dict(coverage=f'[{lo},{hi})', n=int(m.sum()), q_median=float(np.median(qs[m])), q_p10=float(np.percentile(qs[m], 10)), q_p90=float(np.percentile(qs[m], 90))))
res = dict(observed_blocks=int(len(qs)), spearman_q_vs_coverage=float(rho_c), spearman_q_vs_wallfrac=float(rho_w), q_overall=dict(median=float(np.median(qs)), p10=float(np.percentile(qs, 10)), p90=float(np.percentile(qs, 90)), frac_above_3=float((qs > 3).mean()), frac_below_0_1=float((qs < .1).mean())), by_coverage=rows)
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
