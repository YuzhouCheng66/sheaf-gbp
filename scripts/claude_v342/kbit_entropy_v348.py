"""Empirical code entropies of a trained k-bit model (item 1, next-step estimate without retraining). Claude 2026-10-22.
For the VALIDATION scenes: the source codes (mu index on the 2^(k-1)-1 per-side grid, q index on the 2^k-1 log codebook with floor)
of OBSERVED blocks; reports marginal entropies H(mu), H(q), the joint H(mu,q) and the ideal entropy-coded bits per directed message
under the masked wire (64-bit mask + per non-prior coordinate H(mu,q)) using the message statistics of kbit_bytes_v348.py.
Also the fraction of q at the floor level and the number of q levels actually used.
python kbit_entropy_v348.py --source <sealed> --checkpoint runs/v348-kbit/b6/checkpoints/final.pt --trainer train_v348_kbit.py --bytes kbit_bytes_val128.json --out <json>"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--trainer', required=True)
ap.add_argument('--bytes', required=True); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
args = ap.parse_args()
src = Path(args.source).resolve(); sys.path.insert(0, str(src)); sys.path.insert(0, str(src / 'exploration_grid')); sys.path.insert(0, str(Path(args.trainer).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location('trainer', args.trainer); T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from scripts import mapping_multiroom_data_v293 as data
from scripts import train_mapping_support_native_v306 as base
payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False); meta = payload['meta']
T.READOUT_MODE = meta.get('readout_mode', 'posterior_only'); T.PRECISION_MODE = meta.get('precision_mode', 'learned'); T.MU_BITS = meta.get('mu_bits', 8); T.Q_BITS = meta.get('q_bits', 8); T.WIRE = meta.get('wire', 'sealed')
k_mu, k_q = T.MU_BITS, T.Q_BITS
device = torch.device(args.device)
model = T.SheafModel(meta['arm'], 128, 0).to(device); model.load_state_dict(payload['state_dict'], strict=True); model.eval()
levels = 2 ** (k_mu - 1) - 1; nq = 2 ** k_q - 1; low, high = 1e-8, 128.
mu_idx, q_idx = [], []
with torch.no_grad():
    for s in [s for s in base.scene_plan('VALIDATION') if s['size'] == 128]:
        fx = data.make_fixture(**s); own = torch.from_numpy(fx['inputs']['private']).to(device)
        mu, q, mask = model.source(own)
        obs = mask > 0
        mi = torch.round(mu.clamp(-1, 1) * levels).long() + levels                                     # 0 .. 2*levels
        qi = torch.floor((q.double().clamp(low, high).log() - math.log(low)) / (math.log(high) - math.log(low)) * (nq - 1)).long()
        mu_idx.append(mi[obs].cpu().numpy()); q_idx.append(qi[obs].cpu().numpy())
mu_idx = np.concatenate(mu_idx); q_idx = np.concatenate(q_idx)
def H(x):
    _, c = np.unique(x, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
joint = mu_idx.astype(np.int64) * (nq + 1) + q_idx
res = dict(checkpoint=args.checkpoint, mu_bits=k_mu, q_bits=k_q, observed_blocks=int(len(mu_idx)), H_mu=H(mu_idx), H_q=H(q_idx), H_joint=H(joint),
           mu_levels_used=int(len(np.unique(mu_idx))), q_levels_used=int(len(np.unique(q_idx))), q_floor_fraction=float((q_idx == q_idx.min()).mean()))
b = json.load(open(args.bytes)); m = b['mean_nonprior_coordinates_per_message']
res['masked_fixed_bits_per_message'] = 64 + (k_mu + k_q) * m
res['masked_entropy_coded_bits_per_message'] = 64 + res['H_joint'] * m
res['entropy_coded_bytes_per_message'] = res['masked_entropy_coded_bits_per_message'] / 8
res['note'] = 'relayed coordinates carry other sources codes; their marginal distribution is assumed equal to the source-code distribution (same encoder)'
print(json.dumps(res, indent=1)); json.dump(res, open(args.out, 'w'), indent=1)
