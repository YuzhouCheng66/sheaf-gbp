"""Reviewer: re-derive the W1/W2 table from wselect_jobs with own code (no import of g4_e3l3_w / frozen pickers); test w* under
alternative readings of the D8-67 fallback ('字节最小' over two checkpoints). numpy only."""
import json, glob, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); J = os.path.join(D, '..', 'e7', 'wselect_jobs')
crit = json.load(open(os.path.join(D, '..', 'e7', 'criteria_w.json')))
labs = ['learned', 'w3', 'w4', 'w6', 'w8']; out = {}
for sd in ('342101', '342202'):
    rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(J, f'{sd}_*.json')))]
    assert len(rs) == 19 and not any(r.get('failed') for r in rs)
    c = np.sum([r['ctrl_counts']['R8'] for r in rs], 0); free_ctrl = c[0] / c[1]; fmin = free_ctrl - 0.0025
    o = {}
    for lab in labs:
        A = np.sum([r['acc'][lab] for r in rs], 0); T = np.sum([r['tot'][lab] for r in rs], 0)
        tp = np.cumsum(A[0][::-1])[::-1]; fp = np.cumsum(A[1][::-1])[::-1]; fn = T[0] - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1); fr = np.r_[0, np.cumsum(A[2])[:-1]] / T[1]
        ok = fr + 1e-15 >= fmin; j = int(np.flatnonzero(ok)[np.argmax(f1[ok])])
        rat = [r['bytes'][lab][d] / r['ctrl_bytes'][d] for r in rs for d in r['draws']]
        rl = [r['bytes'][lab][d] / r['bytes']['learned'][d] for r in rs for d in r['draws']]
        o[lab] = dict(j=j, f1=float(f1[j]), free=float(fr[j]), crit_j=crit['thresholds'][sd][lab]['R8']['j_occ'],
                      crit_f1=crit['per_seed'][sd]['w'][lab]['furn_f1'], med=float(np.median(rat)), mean=float(np.mean(rat)),
                      med_vs_learned=float(np.median(rl)), ratios=rat, mism=sum(r['known_mismatch'][lab] for r in rs))
    out[sd] = o
print('own threshold re-selection == criteria:', all(out[s][l]['j'] == out[s][l]['crit_j'] and abs(out[s][l]['f1'] - out[s][l]['crit_f1']) < 1e-15 for s in out for l in labs))
for l in labs:
    print(l, ' | '.join(f"{s}: j {out[s][l]['j']} F1 {out[s][l]['f1']:.5f} d {100*(out[s][l]['f1']-out[s]['learned']['f1']):+.3f}pp free {out[s][l]['free']:.5f} "
                       f"med {out[s][l]['med']:.4f} mean {out[s][l]['mean']:.4f} xL {out[s][l]['med_vs_learned']:.3f} mism {out[s][l]['mism']}" for s in out))
ws = ['w3', 'w4', 'w6', 'w8']
furn_ok = {l: all(out[s][l]['f1'] >= out[s]['learned']['f1'] - 0.0015 - 1e-12 for s in out) for l in ws}
print('furn ok', furn_ok)
alts = dict(max_of_medians=lambda l: max(out[s][l]['med'] for s in out), mean_of_medians=lambda l: np.mean([out[s][l]['med'] for s in out]),
            pooled_median=lambda l: float(np.median(out['342101'][l]['ratios'] + out['342202'][l]['ratios'])),
            min_of_medians=lambda l: min(out[s][l]['med'] for s in out), max_of_means=lambda l: max(out[s][l]['mean'] for s in out),
            max_med_vs_learned=lambda l: max(out[s][l]['med_vs_learned'] for s in out))
for nm, f in alts.items():
    v = {l: round(float(f(l)), 5) for l in ws}
    both = [l for l in ws if furn_ok[l] and v[l] <= 0.13 + 1e-12] if nm != 'max_med_vs_learned' else []
    pick = (max(both, key=lambda l: float(l[1:])) if both else min([l for l in ws if furn_ok[l]], key=lambda l: (v[l], -float(l[1:]))))
    print(f'{nm:20s} {v}  main-branch set {both} -> w* = {pick}')
for s in out:
    print(s, 'per-checkpoint argmin of median:', min(ws, key=lambda l: out[s][l]['med']), ' per-ckpt both-rule:', [l for l in ws if out[s][l]['med'] <= 0.13 and out[s][l]['f1'] >= out[s]['learned']['f1'] - 0.0015])
r4 = np.array(out['342101']['w4']['ratios']); r3 = np.array(out['342101']['w3']['ratios'])
print('342101 per-(house,draw) w4 < w3 bytes:', int((r4 < r3).sum()), '/', len(r4), '; share of w4 ratios <= .13:', float((r4 <= .13).mean()))
