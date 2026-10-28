"""Reviewer's read-only check of cache-e3l5-div-v1 on uk-linux (E3-L5 code review, 2026-10-28). CPU only, one process.
Reads the cache, cache-e3r-v1's manifest and div_records; re-derives 3 houses' worlds with v420div.world_spec and counts kinds / families /
primitives with its own code; compares the re-rasterised object grid with the cached npz. Writes nothing but stdout."""
import os, sys, json, hashlib, collections
os.environ['CUDA_VISIBLE_DEVICES'] = ''
DIRS = '/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts'
sys.path.insert(0, DIRS)
import numpy as np
C5 = '/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3l5-div-v1'
CR = '/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


m5 = json.load(open(f'{C5}/manifest.json')); mr = json.load(open(f'{CR}/manifest.json'))
print('manifest sha', sha(f'{C5}/manifest.json'))
print('scripts sha: v420div', sha(f'{DIRS}/mapping_3d_storey_data_v420div.py')[:8], 'gen', sha(f'{DIRS}/g4_e3l5_gen.py')[:8],
      'v420ood', sha(f'{DIRS}/mapping_3d_storey_data_v420ood.py')[:8], 'g4_e3l5', sha(f'{DIRS}/g4_e3l5.py')[:8] if os.path.exists(f'{DIRS}/g4_e3l5.py') else None)
print('manifest script / data module sha', m5['script_sha256'][:8], m5['data_module_sha256'][:8], 'seeds', m5['seeds'], 'p_clean', m5.get('p_clean_at_train'))
h5 = {h['house']: h for h in m5['houses']}; hr = {h['house']: h for h in mr['houses']}
print('houses', len(h5), 'train/holdout', m5['n_train'], m5['n_holdout'], 'skipped', m5['skipped'], 'failed', m5['failed'])
print('house set == cache-e3r-v1:', sorted(h5) == sorted(hr), '| holdout flags equal:', all(h5[k]['holdout'] == hr[k]['holdout'] for k in h5 if k in hr),
      '| holdout rule index%10==9:', all(h['holdout'] == (h['index'] % 10 == 9) for h in m5['houses']))
var = collections.Counter(); varho = collections.Counter()
for h in m5['houses']:
    v = 'div' if h['world_id'].startswith('v420div:train_div:TRAIN:') else ('v420' if h['world_id'].startswith('v420:TRAIN:') else 'other')
    var[v] += 1; varho[(v, h['holdout'])] += 1
    if v == 'v420':
        assert h['world_id'] == hr[h['house']]['world_id']
print('variants by world_id', dict(var), dict(varho))
# all div_records: Y totals, families
fam = collections.defaultdict(lambda: [0, 0]); kinds = collections.defaultdict(lambda: [0, 0]); y = [0, 0, 0]; att = collections.Counter()
for h in m5['houses']:
    r = json.load(open(f"{C5}/div_records/{h['house']}.json"))
    assert r['world_id'] == h['world_id'] and r['gen_status'] == 'ok' and not r.get('stats_failed')
    for f, (no, nv) in r['families'].items():
        fam[(r['variant'], f)][0] += no; fam[(r['variant'], f)][1] += nv
    for k, (no, nv) in r['kinds'].items():
        if r['variant'] == 'train_div':
            kinds[k][0] += no; kinds[k][1] += nv
    y[0] += r['y_objects']; y[1] += r['y_voxels']; y[2] += r['y_prims']
    if r.get('attempt') is not None:
        att[r['attempt']] += 1
print('div_records Y objects / voxels / prims', y, '| train_div attempts', dict(att))
for k, v in sorted(fam.items()):
    print('  family', k, v)
print('  train_div kinds', dict(sorted(kinds.items())))
vo = json.load(open(f'{C5}/verify_orig.json'))
print('verify_orig ok', vo['ok'], 'n', vo['n'], 'n_orig', vo['n_orig'], 'n_div', vo['n_div'], 'problems', vo['problems'][:3])
d5 = json.load(open(f'{C5}/e3l5_div.json'))
print('e3l5_div ok', d5['ok'], 'formal', d5['formal'], 'manifest', d5['manifest_sha256'][:16], 'y_totals', d5['y_totals'], 'house_set_equals_e3r', d5['house_set_equals_e3r'])

# re-derive 3 houses with my own counting
os.environ.setdefault('HOUSEEXPO_JSON_DIR', '/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json')
os.environ.setdefault('HOUSEEXPO_INDEX', '/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json')
import g4_noise_panels as GN
_J, _I = '/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/json', '/mnt/datasets/yuzhou-sheaf-gbp-research/houseexpo-v365/houseexpo_index_v365.json'
GN.setup_env(_J, _I)
os.environ['HOUSEEXPO_JSON_DIR'] = _J; os.environ['HOUSEEXPO_INDEX'] = _I
import mapping_3d_storey_data_v420 as W
import mapping_3d_storey_data_v420div as D
YK = {'ball', 'ellipsoid', 'hanging', 'clutter', 'coat_rack'}
picks = [h for h in m5['houses'] if h['world_id'].startswith('v420div:')]
picks = [picks[0], picks[len(picks) // 2], [h for h in picks if h['holdout']][0]]
for h in picks:
    a = int(h['world_id'].split(':')[4][1:]) if len(h['world_id'].split(':')) == 5 else 0
    spec = D.world_spec('TRAIN', h['index'], h['rooms'], a)
    assert spec['world_id'] == h['world_id']
    c = collections.Counter(); pk = collections.Counter(); bad = []
    for o in spec['objects']:
        if o['kind'] in YK:
            bad.append(o['kind'])
        mode = o.get('mode'); kind = o['kind']
        if kind.startswith('rot_'):
            f = 'rotated'
        elif kind == 'wall_shelf':
            f = 'wall_shelf'
        elif kind in W.SIZES_KIND and o.get('scale') in (0.6, 1.5):
            f = f"scaled_{o['scale']:g}"
        else:
            f = 'OTHER:' + kind
        c[f] += 1
        for p in o['prims']:
            pk[p[0]] += 1
        if o.get('placement') != 'floor':
            bad.append('placement:' + str(o.get('placement')))
    # shelves really off the floor
    zmin_shelf = min([min(p[3] if p[0] in ('box',) else p[-2] for p in o['prims']) for o in spec['objects'] if o['kind'] == 'wall_shelf'] or [None]) \
        if any(o['kind'] == 'wall_shelf' for o in spec['objects']) else None
    r = D.rasterize(spec, tuple(W.DIMS['R1']))
    with np.load(f"{C5}/{h['file']}") as z, np.load(f"{CR}/{hr[h['house']]['file']}") as zr:
        same_obj = np.array_equal(np.asarray(z['obj'], bool), np.asarray(r['obj'], bool))
        same_slab_ref = np.array_equal(z['slab'], zr['slab'])
        n_obj_vox = int(np.asarray(z['obj'], bool).sum())
    print(h['house'], 'holdout', h['holdout'], 'attempt', a, 'objects', len(spec['objects']), 'families', dict(c), 'prims', dict(pk), 'Y/bad', bad,
          'npz obj == re-raster', same_obj, 'slab == e3r', same_slab_ref, 'obj voxels', n_obj_vox, 'min shelf prim z', zmin_shelf)
