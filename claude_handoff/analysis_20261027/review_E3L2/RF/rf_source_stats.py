"""rf_source_stats.py (R-F): source-side statistics of the E3-L2 checkpoints at K* on VAL cam_nf (selected cells): raw Lam, clamp / saturation shares,
h_q symbol entropy. Diagnostic only; GPU 0 on bigboy; frozen modules imported read-only."""
import os, sys, json, glob, math
sys.dont_write_bytecode = True
WORK = os.path.expanduser('~/claude-sheaf-gbp/work'); DATA = os.path.expanduser('~/claude-sheaf-gbp/data')
sys.path.insert(0, os.path.join(WORK, 'scripts'))
import numpy as np
import g4_e3l2 as L2
L = L2.L; R = L2.R; G = L2.G
import torch
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cuda')
recs = [json.load(open(p)) for p in sorted(glob.glob(f'{DATA}/cache-e3r-v1-export/records/val_*.json'))]
out = {}
for sd in (342101, 342202):
    for nm in ('final', 'upd6510'):
        run = f'{WORK}/runs/e3l2/s{sd}'; thr = json.load(open(f'{run}/threshold{"" if nm == "final" else "_" + nm}.json')); K = int(thr['k_star'])
        ck, m = L2.load_model(f'{run}/checkpoints/{nm}.pt', dev)
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32, device=dev)
        lam_s, zabs, hsat, hq_hist, lam_unsel, n_sel, n_touch = [], [], [], np.zeros(15, np.int64), [], 0, 0
        for r in recs:
            z = np.load(f"{DATA}/cache-e3r-v1-export/{r['file']}")
            with torch.no_grad():
                zz, lam = L.encode_maps(m, list(z['L_k0_cam_nf']), tb, dev)
                hq, lq, sel = L2.source_topk(zz, lam, K, 'round')
            ls = lam[sel]; lam_s.append(ls.cpu().numpy())
            zs = zz.movedim(1, -1)[sel]; zabs.append(zs.abs().cpu().numpy().ravel()[::7])
            h = (ls.clamp(1, 15).unsqueeze(1) * zs)
            hsat.append(float((h.abs() > 7.5).float().mean())); 
            hq_hist += np.bincount((hq.movedim(1, -1)[sel].round().long().clamp(-7, 7) + 7).cpu().numpy().ravel(), minlength=15)
            unsel = (lam > 0) & ~sel; lam_unsel.append(lam[unsel].cpu().numpy()[::5])
            n_sel += int(sel.sum()); n_touch += int((lam > 0).sum())
        ls = np.concatenate(lam_s); za = np.concatenate(zabs); lu = np.concatenate(lam_unsel)
        p = hq_hist / hq_hist.sum(); ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        out[f'{nm}_{sd}_K{K}'] = dict(K=K, selected_share_of_touched=n_sel / max(n_touch, 1), lam_raw_selected_quantiles={q: float(np.quantile(ls, q)) for q in (0.01, 0.1, 0.5, 0.9)},
                                     share_lam_raw_gt15=float((ls > 15).mean()), share_lam_raw_lt1=float((ls < 1).mean()), lam_q_values=sorted(set(np.round(np.clip(ls, 1, 15)).astype(int).tolist()))[:20],
                                     lam_raw_unselected_quantiles={q: float(np.quantile(lu, q)) for q in (0.1, 0.5, 0.9)} if lu.size else None,
                                     z_abs_quantiles={q: float(np.quantile(za, q)) for q in (0.5, 0.9, 0.99)}, share_h_saturated=float(np.mean(hsat)),
                                     hq_symbol_hist=hq_hist.tolist(), hq_symbol_entropy_bits=ent, share_hq_at_pm7=float((hq_hist[0] + hq_hist[14]) / hq_hist.sum()),
                                     w=float(m.w.detach()))
        print(f'{nm}_{sd}_K{K}', json.dumps({k: v for k, v in out[f"{nm}_{sd}_K{K}"].items() if k != 'hq_symbol_hist'}), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
