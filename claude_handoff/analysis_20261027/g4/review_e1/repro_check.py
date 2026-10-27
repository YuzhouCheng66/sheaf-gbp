"""repro_check.py (code review of G4-E1): re-run one smoke job (val r1_i0, noise seed 1) with the CURRENT g4_noise_panels.py in this
process (no pool) and compare it, field by field, with the smoke job file (written 03:26 by an earlier version of the script; the
smoke run.json carries the sha of the version that only re-aggregated at 03:31). Timing / peak memory fields are ignored."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_noise_panels as G  # noqa: E402

SMOKE = os.path.join(HERE, '..', 'e1_smoke', 'jobs', 'val_r1_i0_s1.json')
IGNORE = {'timing', 'peak_mb'}


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((path + '/' + k, 'missing in ' + ('new' if k not in a else 'smoke'))); continue
            diff(a[k], b[k], path + '/' + k, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f'len {len(a)} != {len(b)}')); return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f'{path}[{i}]', out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            out.append((path, f'{a!r} != {b!r}'))
    elif a != b:
        out.append((path, f'{a!r} != {b!r}'))


if __name__ == '__main__':
    opts = dict(out=HERE, conds=list(G.COND_ORDER), b5_conds=['lidar', 'cam', 'harsh'])
    new = json.loads(json.dumps(G.run_job(('val', 1, 0, 1, opts))))
    old = json.load(open(SMOKE))
    out = []
    diff(new, old, '', out)
    rep = dict(n_diff=len(out), first=out[:50], seconds=new['timing']['total'], peak_mb=new.get('peak_mb'))
    json.dump(rep, open(os.path.join(HERE, 'repro_check.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1)[:4000])
