"""Reviewer probe (G4-E0, item 5): byte effect of W3's generator-aware constants (BAND z 23-24, LINTEL_MAX_FREE 21).
Runs the unmodified g4_exact_codec.main() IN-PROCESS (--procs 1, so the patched module globals are what the codec uses) on exactly the
smoke2 scenes (TRAIN 3 + 3, VAL r1_i0 / r2_i0), W3 only, with:
  noplan  : BAND = (NZ, NZ) -> no wall / lintel plan at all; every hit is covered by boxes (+ data-derived kf / kc layers)
  generic : BAND = (0, NZ), LINTEL_MAX_FREE = NZ - 1 -> 'wall' = any column with a hit and no observed free voxel (no height constant)
Exactness is by construction in every variant (paint never meets an observed free voxel); only bytes change.
Usage: python -B variant_w3.py noplan|generic OUTDIR"""
import os, sys
sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_exact_codec as G
v, out = sys.argv[1], sys.argv[2]
if v == 'noplan':
    G.BAND = (G.NZ, G.NZ)
elif v == 'generic':
    G.BAND = (0, G.NZ); G.LINTEL_MAX_FREE = G.NZ - 1
else:
    raise SystemExit('variant?')
print('variant', v, 'BAND', G.BAND, 'LINTEL_MAX_FREE', G.LINTEL_MAX_FREE, flush=True)
os.chdir(ROOT)
sys.argv = ['g4_exact_codec.py', '--out', out, '--procs', '1', '--wires', 'W3', '--val-limit', '2', '--fresh-limit', '0', '--train-limit', '6',
            '--skip-demo-check']
sys.exit(G.main())
