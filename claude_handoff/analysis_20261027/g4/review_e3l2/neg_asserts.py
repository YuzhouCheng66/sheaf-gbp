"""review E3-L2 (reviewer): the guards fire. TRAIN smoke cache only, CPU, seconds. Writes only under review_e3l2/smoke_neg/."""
import os, sys, json, shutil
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'claude_v342'))
import g4_e3l2 as M
G4 = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'g4')
CACHE = os.path.join(G4, 'e3l_smoke', 'cache'); E3R = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027', 'review_E3R', 'inputs')
FIT = os.path.join(E3R, 'e1c_fit.json').replace(os.sep, '/')
O = os.path.join(G4, 'review_e3l2', 'smoke_neg'); os.makedirs(O, exist_ok=True)
RUN = os.path.join(G4, 'review_e3l2', 'smoke', 'runs', 's342101'); CTRL = os.path.join(G4, 'review_e3l2', 'smoke', 'ctrl_holdout.json')
res = []


def expect(name, argv, needle):
    try:
        M.main(argv); res.append((name, 'NO ERROR (guard did not fire)'))
    except (AssertionError, SystemExit, FileNotFoundError) as e:
        ok = needle in repr(e)
        res.append((name, ('fired: ' if ok else 'fired with another message: ') + repr(e)[:160]))


base = ['train', '--cache', CACHE, '--seed', '342101', '--updates', '2', '--device', 'cpu', '--batch', '1']
expect('manifest sha pin', base + ['--evidence', f'c:{FIT}:A:cam_nf', '--out', os.path.join(O, 'a'), '--manifest-sha', '0' * 64], 'cache manifest sha')
expect('pairing (table B vs E3-R 342101)', base + ['--evidence', f'c:{FIT}:B:cam_nf', '--out', os.path.join(O, 'b'), '--ctrl-ckpt', os.path.join(E3R, 's342101', 'final.pt')], 'evidence != paired')
expect('seed not a training seed', ['train', '--cache', CACHE, '--seed', '7', '--updates', '2', '--device', 'cpu', '--evidence', f'c:{FIT}:A:cam_nf', '--out', os.path.join(O, 'c')], 'not in')
c = json.load(open(CTRL)); c['houses'] = c['houses'] + ['TRAIN_r9_i9']; bad = os.path.join(O, 'ctrl_bad_houses.json'); json.dump(c, open(bad, 'w'))
expect('kselect: holdout houses != control file', ['kselect', '--run', RUN, '--ctrl', bad, '--workers', '1', '--smoke'], 'holdout houses')
c = json.load(open(CTRL)); c['e3r']['342101']['table_sha256'] = '0' * 64; bad2 = os.path.join(O, 'ctrl_bad_table.json'); json.dump(c, open(bad2, 'w'))
expect('select: run table != paired control table', ['select', '--run', RUN, '--ctrl', bad2, '--workers', '1', '--smoke'], 'evidence table')
expect('--bytes-max outside the smoke', ['kselect', '--run', RUN, '--ctrl', CTRL, '--bytes-max', '1'], 'smoke only')
for n, r in res:
    print(f'{n}: {r}')
