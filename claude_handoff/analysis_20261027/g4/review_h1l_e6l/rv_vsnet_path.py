"""reviewer: exercise g4_h1_e3l3._cell's report-only E3L3 - NET path (never hit by the TRAIN smoke) for KeyError-swallowing.
Uses the TRAIN smoke summary as V and the published H1 summary as NET; prints only which keys are filled (no VAL value printed)."""
import os, sys, json
sys.dont_write_bytecode = True; os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'scripts', 'claude_v342')))
import g4_h1_e3l3 as HL
S = json.load(open(os.path.join(HERE, '..', 'h1l_smoke', 'train', 'summary.json')))
NETS = json.load(open(HL.H1_SUMMARY))
for v in ('novel', 'floating'):
    V = S['variants'][v]['203']; NET = NETS['variants'][v]['1']
    for ts in HL.SEEDS:
        c = HL._cell(V, ts, 'own', NET)
        vn = c.get('vs_net')
        print(v, ts, 'vs_net filled:', vn is not None, sorted(k for k, x in (vn or {}).items() if x is not None))
