"""g4_e3l5_gen.py (Claude, 2026-10-28): batch E3-L5 的新 TRAIN 缓存 cache-e3l5-div-v1（预注册 OVERNIGHT_REPORT D8-76；纯 CPU）。
薄包装：缓存的写法、数组、文件格式、records、manifest 全部由冻结的 g4_e3r.py（E3-R，sha 1d65d9a8...）的 gen_job / _gen_and_save /
write_manifest / _pool 原样完成；本文件只做一处替换 E.W = mapping_3d_storey_data_v420div.module_mixed()（E.house_ctx 通过模块全局 W
调 make_fixture；同 g4_ood_panel 的做法），于是每房按 v420div 的 V1 分配用 v420 原世界或 train_div 世界。不改任何已有文件。

与 cache-e3r-v1 相同：房子集合（205 = 186 训练 / 19 留出，index % 10 == 9）、噪声种子 200-202 的 cam_nf + clean、manifest 格式
（format 1，键相同）、npz 的数组（occ / slab / obj / cell / private / comp_of / edges / L_clean / L_cam_nf_s{seed}）。
不同（D8-76 的单变量）：一半房子的家具来自 train_div（X 家族），world_id 带变体前缀（'v420div:train_div:TRAIN:...'）；manifest 的
script_sha256 = 本文件、data_module_sha256 = v420div 的 sha（E.write_manifest 读 E.W.__file__）。

操作化（回复里列出）
  G1 records/ 与 manifest.json 由 E 原样写；另写 div_records/{house}.json（每房的分配、world_id、train_div 尝试序号、按家族 / 类别的物体数与
     体素数，v420div.family_stats，含 Y = 0 的断言）与汇总 e3l5_div.json（manifest sha、全部依赖 sha、分配规则、家族总计、Y 总计 = 0、
     与 cache-e3r-v1 房子集合的一致性）。
  G2 正式生成（无 --houses）结束时检查：205 房、186 / 19、没有 skipped / failed、Y 物体与体素总计 0、v420 那一半的 world_id 是 v420 原值、
     train_div 那一半是 'v420div:train_div:' 前缀；任一不符 -> 退出码 3（缓存不得用于训练）。
  G3 verify-orig：v420 那一半与 cache-e3r-v1 同名房逐数组相同（W.array_hash，全部数组）；train_div 那一半的 slab 与 cache-e3r-v1 相同、
     obj 不同（平面不变、家具变了）。结果写 verify_orig.json；任一不符 -> 退出码 3。
  G4 只生成 TRAIN；噪声种子不得用保留种子（E._check_seeds）；不碰 VALIDATION / TEST；workers <= 9（uk-linux 上 9 + 父进程 = 10 个进程）。

用法（总是 python -B；CUDA_VISIBLE_DEVICES= ）：
  gen          --cache C [--seeds 200,201,202] [--houses r:i,...] [--workers 9]
  verify-orig  --cache C --ref /mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1 [--houses r:i,...]
  stats        --cache C                       （只重写 e3l5_div.json，不生成）
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, json, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3r as E  # noqa: E402   (frozen; read only)

G = E.G
import mapping_3d_storey_data_v420 as W  # noqa: E402
import mapping_3d_storey_data_v420ood as WO  # noqa: E402
import mapping_3d_storey_data_v420div as D  # noqa: E402

assert W is G.W, 'v420 must be the module g4_noise_panels imported'
MAX_WORKERS = 9
FORMAT = 1


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def sha(p):
    return G.file_sha256(p)


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def dep_shas():
    return {'g4_e3l5_gen.py': sha(os.path.abspath(__file__)), 'g4_e3r.py': sha(E.__file__), 'g4_noise_panels.py': sha(G.__file__),
            'mapping_3d_storey_data_v420.py': sha(W.__file__), 'mapping_3d_storey_data_v420ood.py': sha(WO.__file__),
            'mapping_3d_storey_data_v420div.py': sha(D.__file__)}


def attempt_of(world_id):
    """v420div V6: 'v420div:train_div:TRAIN:{house_id}' (attempt 0) or '...:{house_id}:a{k}' (attempt k); v420 ids -> 0"""
    parts = world_id.split(':')
    if parts[0] == 'v420div' and len(parts) == 5:
        assert parts[4].startswith('a'), world_id
        return int(parts[4][1:])
    assert len(parts) == 3 or (parts[0] == 'v420div' and len(parts) == 4), world_id
    return 0


def expected_world_prefix(rooms, index):
    return 'v420:TRAIN:' if D.variant_of('TRAIN', index, rooms) == D.ORIG else f'v420div:{D.VARIANT}:TRAIN:'


# ================================================================ G1: one house (worker; spawn)
def _gen_div_and_save(job):
    """E.W -> the mixed module (this process only), then E3-R's own gen job; then the per-house family statistics"""
    E.W = D.module_mixed()
    rooms, index, seeds, opts = job
    rec = E._gen_and_save(job)
    name = E.house_name(rooms, index)
    st = dict(house=name, rooms=int(rooms), index=int(index), holdout=E.is_holdout(index), variant=D.variant_of('TRAIN', index, rooms),
              script_sha256=opts['script_sha256'])
    try:
        if rec.get('failed') or rec.get('skipped'):
            st['gen_status'] = 'failed' if rec.get('failed') else 'skipped'
        else:
            wid = rec['world_id']
            assert wid.startswith(expected_world_prefix(rooms, index)), f'{name}: world_id {wid} has the wrong prefix'
            att = attempt_of(wid)
            st.update(gen_status='ok', world_id=wid, **{k: v for k, v in D.family_stats('TRAIN', index, rooms, attempt=att).items()
                                                         if k not in ('house', 'variant', 'world_id')})
            assert st['y_objects'] == 0 and st['y_voxels'] == 0 and st['y_prims'] == 0
    except Exception as e:                                            # recorded; the formal check (G2) fails on it
        st['stats_failed'] = True; st['error'] = repr(e)[:500]; st['traceback'] = traceback.format_exc()[-2000:]
    d = os.path.join(opts['cache'], 'div_records'); os.makedirs(d, exist_ok=True)
    _jdump(st, os.path.join(d, name + '.json'))
    rec['div'] = dict(variant=st['variant'], families=st.get('families'), y_objects=st.get('y_objects'), stats_failed=st.get('stats_failed', False))
    return rec


# ================================================================ G1 / G2: the summary sidecar
def div_summary(cache, formal):
    man_p = os.path.join(cache, 'manifest.json')
    man = _jload(man_p)
    recs = {r['house']: r for r in (_jload(p) for p in sorted(glob.glob(os.path.join(cache, 'div_records', '*.json'))))}
    tot = {}; ytot = dict(objects=0, voxels=0, prims=0); per_var = {}; problems = []
    for h in man['houses']:
        r = recs.get(h['house'])
        if r is None or r.get('stats_failed') or r.get('gen_status') != 'ok':
            problems.append(f"{h['house']}: div record missing / failed ({(r or {}).get('error')})"); continue
        if r['world_id'] != h['world_id']:
            problems.append(f"{h['house']}: div record world_id != manifest")
        if not h['world_id'].startswith(expected_world_prefix(h['rooms'], h['index'])):
            problems.append(f"{h['house']}: world_id prefix")
        v = r['variant']; pv = per_var.setdefault(v, dict(houses=0, train=0, holdout=0, objects=0, obj_voxels=0, attempts={}))
        pv['houses'] += 1; pv['holdout' if h['holdout'] else 'train'] += 1; pv['objects'] += r['n_objects']; pv['obj_voxels'] += r['obj_voxels']
        if r.get('attempt') is not None:
            pv['attempts'][str(r['attempt'])] = pv['attempts'].get(str(r['attempt']), 0) + 1
        for f, (no, nv) in r['families'].items():
            a = tot.setdefault(v, {}).setdefault(f, [0, 0]); a[0] += no; a[1] += nv
        ytot['objects'] += r['y_objects']; ytot['voxels'] += r['y_voxels']; ytot['prims'] += r['y_prims']
    names = sorted(h['house'] for h in man['houses'])
    want = sorted(D.house_name(r, i) for r, i in D.train_plan())
    if formal:
        if names != want:
            problems.append(f'house set != cache-e3r-v1 ({len(names)} vs {len(want)})')
        if (man['n_train'], man['n_holdout']) != (186, 19):
            problems.append(f"n_train / n_holdout {man['n_train']} / {man['n_holdout']} != 186 / 19")
        if man['skipped'] or man['failed']:
            problems.append(f"skipped {man['skipped']} failed {man['failed']}")
    if ytot['objects'] or ytot['voxels'] or ytot['prims']:
        problems.append(f'held-out (Y) totals {ytot} != 0')
    ys = {v: {f: x for f, x in fam.items() if f in D.Y_FAMILIES} for v, fam in tot.items()}
    if any(ys.values()):
        problems.append(f'Y families present: {ys}')
    out = dict(format=FORMAT, batch='E3-L5', prereg='D8-76', formal=bool(formal), utc=_utc(), cache=os.path.abspath(cache), manifest_sha256=sha(man_p),
               n_houses=len(man['houses']), n_train=man['n_train'], n_holdout=man['n_holdout'], seeds=man['seeds'], skipped=man['skipped'], failed=man['failed'],
               assignment_rule='v420div V1: per (rooms stratum, holdout) group, sorted by sha256(v420div:assign:TRAIN:r{rooms}:i{index}), first floor(n/2) -> train_div',
               per_variant=per_var, families=tot, y_families=list(D.Y_FAMILIES), y_kinds=list(D.Y_KINDS), x_families=list(D.X_FAMILIES), y_totals=ytot,
               house_set_equals_e3r=bool(names == want), e3r_manifest_sha256=D.E3R_MANIFEST_SHA256, e3r_skipped=list(D.E3R_SKIPPED),
               houses={h['house']: dict(variant=recs.get(h['house'], {}).get('variant'), world_id=h['world_id'], holdout=h['holdout'],
                                        attempt=recs.get(h['house'], {}).get('attempt')) for h in man['houses']},
               deps_sha256=dep_shas(), problems=problems, ok=not problems)
    _jdump(out, os.path.join(cache, 'e3l5_div.json'))
    return out


def print_summary(S):
    print(f"E3-L5 cache {S['cache']}: manifest sha256 {S['manifest_sha256']}; {S['n_houses']} houses ({S['n_train']} train / {S['n_holdout']} holdout), "
          f"seeds {S['seeds']}, skipped {S['skipped']}, failed {S['failed']}", flush=True)
    for v, pv in sorted(S['per_variant'].items()):
        print(f"  {v}: {pv['houses']} houses ({pv['train']} train / {pv['holdout']} holdout), {pv['objects']} objects, {pv['obj_voxels']} object voxels, "
              f"attempts {pv['attempts']}", flush=True)
        for f, (no, nv) in sorted(S['families'].get(v, {}).items()):
            print(f'     family {f:12s} {no:6d} objects {nv:9d} voxels', flush=True)
    print(f"  Y totals {S['y_totals']} (families {S['y_families']}); house set == cache-e3r-v1: {S['house_set_equals_e3r']}; "
          f"problems: {S['problems'] or 'none'}", flush=True)


# ================================================================ G3: the v420 half vs cache-e3r-v1, array by array
def verify_orig(cache, ref, only=None):
    man = _jload(os.path.join(cache, 'manifest.json')); rman = _jload(os.path.join(ref, 'manifest.json'))
    rsha = sha(os.path.join(ref, 'manifest.json'))
    rby = {h['house']: h for h in rman['houses']}
    rows = {}; problems = []
    if rsha != D.E3R_MANIFEST_SHA256:
        problems.append(f'reference manifest sha {rsha} != cache-e3r-v1 {D.E3R_MANIFEST_SHA256}')
    for h in man['houses']:
        if only and (h['rooms'], h['index']) not in only:
            continue
        rh = rby.get(h['house'])
        if rh is None:
            problems.append(f"{h['house']}: not in the reference"); continue
        var = D.variant_of('TRAIN', h['index'], h['rooms'])
        with np.load(os.path.join(cache, h['file'])) as a, np.load(os.path.join(ref, rh['file'])) as b:
            ka, kb = sorted(a.files), sorted(b.files)
            ha = {k: W.array_hash(a[k]) for k in ka}; hb = {k: W.array_hash(b[k]) for k in kb}
        if ka != kb:
            problems.append(f"{h['house']}: array names differ {ka} vs {kb}")
        if var == D.ORIG:
            same = ka == kb and ha == hb and h['world_id'] == rh['world_id'] and h['components'] == rh['components']
            if not same:
                problems.append(f"{h['house']} (v420): arrays / world_id differ from cache-e3r-v1: {[k for k in ka if ha.get(k) != hb.get(k)]}")
        else:
            same = ha.get('slab') == hb.get('slab') and ha.get('obj') != hb.get('obj') and h['world_id'] != rh['world_id']
            if not same:
                problems.append(f"{h['house']} (train_div): slab must equal and obj / world_id differ")
        rows[h['house']] = dict(variant=var, ok=bool(same), differing=[k for k in ka if ha.get(k) != hb.get(k)])
    out = dict(stage='verify-orig', cache=os.path.abspath(cache), manifest_sha256=sha(os.path.join(cache, 'manifest.json')), ref=os.path.abspath(ref),
               ref_manifest_sha256=rsha, n=len(rows), n_orig=sum(1 for r in rows.values() if r['variant'] == D.ORIG),
               n_div=sum(1 for r in rows.values() if r['variant'] == D.VARIANT), rows=rows, problems=problems, ok=not problems, utc=_utc(),
               script_sha256=sha(os.path.abspath(__file__)))
    _jdump(out, os.path.join(cache, 'verify_orig.json'))
    print(f"verify-orig: {out['n']} houses ({out['n_orig']} v420 equal to cache-e3r-v1 array by array: "
          f"{sum(1 for r in rows.values() if r['variant'] == D.ORIG and r['ok'])}; {out['n_div']} train_div with the same slab and other furniture: "
          f"{sum(1 for r in rows.values() if r['variant'] == D.VARIANT and r['ok'])}); problems: {problems or 'none'}", flush=True)
    return out


# ================================================================ main
def _parse_houses(s):
    return {tuple(int(x) for x in tok.split(':')) for tok in s.split(',') if tok}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('gen', 'verify-orig', 'stats'):
        p = sub.add_parser(name)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--cache', required=True)
        if name == 'gen':
            p.add_argument('--seeds', default=','.join(map(str, E.GEN_SEEDS))); p.add_argument('--houses', default='')
            p.add_argument('--workers', type=int, default=MAX_WORKERS)
        if name == 'verify-orig':
            p.add_argument('--ref', required=True); p.add_argument('--houses', default='')
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or E._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or E._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.cmd == 'verify-orig':
        out = verify_orig(args.cache, args.ref, _parse_houses(args.houses) or None)
        raise SystemExit(0 if out['ok'] else 3)
    if args.cmd == 'stats':
        E.W = D.module_mixed()
        man = _jload(os.path.join(args.cache, 'manifest.json'))
        S = div_summary(args.cache, formal=len(man['houses']) == 205); print_summary(S)
        raise SystemExit(0 if S['ok'] else 3)
    # gen
    assert 1 <= args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} workers (+ the parent = {MAX_WORKERS + 1} processes)'
    seeds = [int(s) for s in args.seeds.split(',') if s]
    E._check_seeds(seeds, False)
    plan = D.train_plan()
    want = _parse_houses(args.houses)
    if want:
        assert want <= set(plan), 'explicit houses must be among the 205 cache-e3r-v1 TRAIN houses'
        plan = [h for h in plan if h in want]
    me = sha(os.path.abspath(__file__))
    opts = dict(cache=args.cache, seeds=seeds, script_sha256=me)
    jobs = []
    for rooms, index in plan:
        rp = os.path.join(args.cache, 'records', E.house_name(rooms, index) + '.json')
        dp = os.path.join(args.cache, 'div_records', E.house_name(rooms, index) + '.json')
        if os.path.exists(rp) and os.path.exists(dp):
            r = _jload(rp); q = _jload(dp)
            if r.get('script_sha256') == me and r.get('seeds') == seeds and not r.get('failed') and q.get('script_sha256') == me and not q.get('stats_failed'):
                continue                                              # resume only an identical record (as E3-R)
        jobs.append((rooms, index, seeds, opts))
    t0 = time.time()
    os.makedirs(args.cache, exist_ok=True)
    _jdump(dict(stage='gen', utc_start=_utc(), argv=sys.argv, seeds=seeds, planned=len(plan), todo=len(jobs), workers=args.workers, deps_sha256=dep_shas(),
                houseexpo_json_dir=json_dir, houseexpo_index=index_file, pid=os.getpid()), os.path.join(args.cache, f'gen_run_{int(t0)}.json'))
    print(f'E3-L5 gen: {len(jobs)} TRAIN houses to do ({len(plan)} planned; '
          f'{sum(1 for h in plan if D.variant_of("TRAIN", h[1], h[0]) == D.VARIANT)} train_div), seeds {seeds}, {args.workers} workers, {_utc()}', flush=True)
    for i, rec in E._pool(args.workers, _gen_div_and_save, jobs, json_dir, index_file):
        dv = rec.get('div') or {}
        print(f"[{i}/{len(jobs)}] {rec['house']} {dv.get('variant')} "
              f"{'SKIPPED ' + rec['skipped'] if rec.get('skipped') else 'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
              f"{'STATS FAILED ' if dv.get('stats_failed') else ''}{rec.get('seconds', 0):.0f}s {rec.get('file_bytes', 0) / 2 ** 20:.2f} MB "
              f"families {dv.get('families')}  {time.time() - t0:.0f}s", flush=True)
    E.W = D.module_mixed()                                            # data_module_sha256 = v420div (E.write_manifest reads E.W.__file__)
    man, path = E.write_manifest(args.cache, opts)
    print(f"manifest {path} sha256 {sha(path)}: {len(man['houses'])} houses ({man['n_train']} train / {man['n_holdout']} holdout), "
          f"{man['total_bytes'] / 2 ** 20:.1f} MB, skipped {len(man['skipped'])}, failed {len(man['failed'])}", flush=True)
    S = div_summary(args.cache, formal=not want)
    print_summary(S)
    print(f"e3l5_div.json sha256 {sha(os.path.join(args.cache, 'e3l5_div.json'))}  done {_utc()}", flush=True)
    raise SystemExit(0 if S['ok'] else 3)


if __name__ == '__main__':
    main()
