"""Q3: training dynamics from rate_curve_*.json and history.json (read only) -> rd_out/q3_summary.json"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(HERE, '..', '..'))
out = {}
for sd in ('342101', '342202'):
    rc = json.load(open(os.path.join(A, 'g4', 'e3l', f'rate_curve_{sd}.json')))
    hi = json.load(open(os.path.join(A, 'review_E3R', 'inputs', f'e3l_s{sd}', 'history.json')))
    u = np.array([r['updates'] for r in rc]); sup = np.array([r['support_cells'] for r in rc]); tch = np.array([r['touched_cells'] for r in rc])
    lam = np.array([r['lam_dual'] for r in rc]); rm = np.array([r['r_meas'] for r in rc]); meas = np.array([r['measured_bits'] for r in rc])
    frac = sup / np.maximum(tch, 1)
    post = u >= 300
    # collapse events: support fraction drops below 5 % (from above 20 %)
    ev = []; state = 'up'
    for i in range(len(u)):
        if state == 'up' and frac[i] < 0.05:
            ev.append(int(u[i])); state = 'down'
        elif state == 'down' and frac[i] > 0.20:
            state = 'up'
    # oscillation: sign changes of (r_meas - R*) after warm-up
    s = np.sign(rm[post] - 20000.0); cross = np.flatnonzero(np.diff(s) != 0)
    period = float(2 * np.mean(np.diff(u[post][cross]))) if len(cross) > 2 else None
    late = u >= 5500
    out[sd] = dict(n=len(u), first_zero_support=int(u[np.flatnonzero(sup == 0)[0]]) if (sup == 0).any() else None,
                   zero_support_updates=[int(x) for x in u[sup == 0]],
                   collapse_events_lt5pct=ev, n_collapse=len(ev),
                   frac_measurements_support_lt10pct_after_warmup=float((frac[post] < 0.10).mean()),
                   frac_measurements_support_gt50pct_after_warmup=float((frac[post] > 0.50).mean()),
                   support_frac_median_after_warmup=float(np.median(frac[post])),
                   lam_max=float(lam.max()), frac_lam_zero_after_warmup=float((lam[post] == 0).mean()),
                   r_meas_cross_period_updates=period, n_cross=int(len(cross)),
                   late_support_frac_mean=float(frac[late].mean()), late_support_frac_cv=float(frac[late].std() / max(frac[late].mean(), 1e-9)),
                   late_r_meas_mean=float(rm[late].mean()), late_meas_cv=float(meas[late].std() / meas[late].mean()),
                   holdout=[dict(u=h['updates'], ho_known=h['holdout_bce_known'], ho_occ=h['holdout_bce_occ'], sup=h['support_cells'], tch=h['touched_cells'],
                                 lam=h['lam_dual']) for h in hi])
    # holdout bce vs support fraction at the 10 snapshots (correlation)
    hs = np.array([[h['support_cells'] / max(h['touched_cells'], 1), h['holdout_bce_occ'], h['holdout_bce_known']] for h in hi[1:]])
    out[sd]['corr_supportfrac_vs_ho_occ'] = float(np.corrcoef(hs[:, 0], hs[:, 1])[0, 1])
    out[sd]['corr_supportfrac_vs_ho_known'] = float(np.corrcoef(hs[:, 0], hs[:, 2])[0, 1])
    # snapshot pairs: high support (> 0.6) vs end
    hiS = [h for h in hi if h['support_cells'] / max(h['touched_cells'], 1) > 0.6]
    out[sd]['ho_occ_at_high_support'] = [(h['updates'], round(h['support_cells'] / h['touched_cells'], 3), h['holdout_bce_occ'], h['holdout_bce_known']) for h in hiS]
    print(sd, json.dumps({k: v for k, v in out[sd].items() if k not in ('holdout', 'zero_support_updates')}))
with open(os.path.join(HERE, 'rd_out', 'q3_summary.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
