"""review E1c: synthetic probes of g4_calib_stalk.verdict (no data)."""
import os, sys, copy, json
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_calib_stalk as C

def row(f, fr=.98, fpr=.02, sf=.92):
    return dict(furn_f1=f, free_recall=fr, unknown_known_fpr=fpr, surface_f1=sf)

def S_of(val_d, fresh_d=None, b4=.85, clean=(1., 1.)):
    exp = C.expected_houses('val')
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], agg_reasons=[])
    for panel, dd in (('val', val_d), ('fresh', fresh_d)):
        if dd is None:
            continue
        P = dict(seeds={})
        for s in ('1', '2'):
            t = {'cam_nf': {'B4': dict(R8=row(b4), K0=row(.7))}, 'clean': {'B4': dict(R8=row(1., 1., 0., 1.), K0=row(1.))}}
            for h in 'AB':
                t['cam_nf'][f'B6_{h}'] = dict(R8=row(b4 + dd[(h, s)] / 100, .99), K0=row(.75))
                t['clean'][f'B6_{h}'] = dict(R8=row(clean[0], clean[1], 0., 1.), K0=row(1.))
            P['seeds'][s] = dict(houses=sorted(exp) if panel == 'val' else sorted(C.expected_houses('fresh')), tiers=t)
        S['panels'][panel] = P
    return S

allc = lambda x: {(h, s): x for h in 'AB' for s in ('1', '2')}
cases = {
  'lever_no_fresh': S_of(allc(4.)),
  'lever_fresh_neg': S_of(allc(4.), allc(-0.5)),
  'lever_fresh_one_cell_neg_mean_pos': S_of(allc(4.), {**allc(1.), ('A', '1'): -2.}),
  'notenough': S_of({**allc(4.), ('B', '2'): 0.5}, allc(2.)),
  'ambig': S_of({**allc(4.), ('B', '2'): 2.}, allc(2.)),
  'clean_fail': S_of(allc(4.), allc(2.), clean=(.985, 1.)),
}
out = {}
for k, S in cases.items():
    V = C.verdict(copy.deepcopy(S))
    out[k] = dict(verdict=V['verdict'], reasons=V.get('reasons'), val_mean=V.get('val_mean_delta_pp'), fresh_mean=V.get('fresh_mean_delta_pp'))
print(json.dumps(out, indent=1))
