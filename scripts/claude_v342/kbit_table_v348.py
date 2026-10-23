"""Rate-accuracy table for the v348 k-bit batch (+ pre-registered admissibility gate) with the exact byte accounting of kbit_bytes_v348.py.
python kbit_table_v348.py <runs/v348-kbit> <kbit_bytes_val128.json> [<v347 max-s342101 final.json for the sealed-wire reference>]"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); bytes_ = json.load(open(sys.argv[2])); ref = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
rows = {}
for b in (8, 6, 5, 4):
    p = root / f'b{b}' / 'final.json'
    if not p.exists():
        continue
    d = json.load(open(p)); v = d['val']; f = d.get('fresh')
    rows[b] = dict(k0=v['0']['all']['wall_f1'], r8=v['8']['all']['wall_f1'], acc=v['8']['all']['union_accuracy'], fpr=v['8']['all']['unknown_known_fpr'], own=v['8']['all']['own_wall_f1'],
                   fresh_k0=f['0']['all']['wall_f1'] if f else None, fresh_r8=f['8']['all']['wall_f1'] if f else None, epochs=d['epochs_completed'], updates=d['updates'])
b8 = rows.get(8)
print('k | bits/coord | fixed B/msg | masked B/msg (vs 172 B sealed) | K0 F1 | R8 F1 | acc | FPR | own@R8 | fresh K0/R8 | gate(K0>=b8-1pp & R8>=b8-1pp & FPR<=b8+0.3pp)')
for b, r in rows.items():
    s = bytes_['schemes'][f'k{b}']
    gate = (r['k0'] >= b8['k0'] - 0.01 and r['r8'] >= b8['r8'] - 0.01 and r['fpr'] <= b8['fpr'] + 0.003) if b8 else None
    print(f"{b} | {2*b} | {s['fixed_bytes_per_message']:.0f} | {s['masked_mean_bytes_per_message']:.1f} ({s['masked_over_sealed']:.2f}) | {r['k0']:.3f} | {r['r8']:.3f} | {r['acc']:.3f} | {r['fpr']:.3f} | {r['own']:.3f} | {r['fresh_k0'] if r['fresh_k0'] is None else round(r['fresh_k0'],3)}/{r['fresh_r8'] if r['fresh_r8'] is None else round(r['fresh_r8'],3)} | {'PASS' if gate else 'fail'} (ep {r['epochs']})")
if ref:
    v = ref['val']; print(f"reference sealed profile-4 wire (v347 max-s342101, same seed): K0 {v['0']['all']['wall_f1']:.3f} R8 {v['8']['all']['wall_f1']:.3f} acc {v['8']['all']['union_accuracy']:.3f} FPR {v['8']['all']['unknown_known_fpr']:.3f} own {v['8']['all']['own_wall_f1']:.3f}; sealed packet 172 B/msg")
