"""g4_e3l5.py (Claude, 2026-10-28): batch E3-L5 "带形状多样性的训练分布，治本处理学习隐码的形状先验"
（预注册 OVERNIGHT_REPORT D8-76；GPU 一个租约，发射前须用户确认算力）。
薄包装：只 import 冻结的 g4_e3l3.py（E3-L3，sha c7523cc8...；经它 g4_e3l2 / g4_e3l / g4_e3r / g4_noise_panels），CPU 评估阶段另外懒加载
冻结的 g4_ood_panel.py（H1，3a6959da...）与 g4_h1_e3l3.py（H1L，dc0f69d7...）的计分函数；不改任何已有文件。新的 TRAIN 缓存由
g4_e3l5_gen.py 生成（v420div：v420 50 % + train_div 50 %）。

单变量（D8-76）：训练分布 = 训练缓存。train 把参数原样交给 g4_e3l3 train（编码器、解码器、三级隐码、Lam = 足迹、二次能量 GaBP、
w 可学、优化器、日程、探针、阈值规则全部是 E3-L3 的代码与默认值），只是 --cache 指向 cache-e3l5-div-v1（--ctrl-holdout 用这个缓存的
留出房对照文件，g4_e3l2 ctrlprep 生成；这是"新缓存"的配套，不是第二个变量）。

操作化（D8-76 没写死、由本实现写死的部分；回复里逐条列出）
  F1 train：L3.main(['train', ...]) 原样执行；前置检查 = 缓存旁的 e3l5_div.json（g4_e3l5_gen 的汇总）ok、Y 总计为 0、它记录的 manifest
     sha 等于缓存的 manifest 且等于 --manifest-sha（正式运行必须给）。<run>/e3l5.json（不覆盖）记预注册、缓存、单变量说明；final.json 加
     'e3l5' 块。更新数口径 = E3-L3 的：预注册 9,562（E3-L3 计划数）；放不进租约时由链按 E3-L3 规则算能放下的数并写明"第二个差异"
     （--second-note；同 E3-L4 Q3）。< 2,000 -> INCOMPLETE。
  F2 select：(i) 学到的 w：L3.main(['select', ...]) 原样 -> threshold.json；(ii) w = 4：同一个 L3._sel_task（工作进程里把 L3.load_model
     换成"载入后 log_w <- log 4"的包装，断言 float32 下 exp(log_w) == 4.0），汇总逐行照 L3.select（已知不一致 = 0、FPR 计数与对照逐个相等、
     L3.pick_r8_k0：空闲召回 >= 对照 - 0.25 pp、家具 F1 最大、同分取最小）-> threshold_w4.json。都只在新缓存的 19 个留出房 x 抽样
     200-202 上选（D8-76：阈值按 E3-L3 规则在新缓存的 TRAIN 留出房上重选）。
  F3 eval（精度保护，判据 3、4）：G4-E5 导出（VAL x 噪声种子 3、4，只读，records 与 sha 核对）；每个任务用冻结的 L3.eval_arm 算
       E3L5W4_<ts>（w = 4，threshold_w4.json）、E3L5_<ts>（学到的 w，threshold.json；只报告），档 cam_nf 与 clean；
       E3L3W4_<ts>（冻结 E3-L3 检查点、w = 4、G4-E7 判据文件 criteria_w.json 为 w = 4 选的阈值；判据 3 的比较对象），档 cam_nf；
       NET_<ts>（配对 E3-R 检查点与阈值，照 g4_e3l3_w.net_block：R8 分量行、K0_sep、cam_nf 的 (S, k) 字节；保护项与字节的对照）。
     cam 档与 fresh 面板不在 D8-76 判据里，默认不算（--tiers / --panels 可加，只报告）。一致性闸门：E3L3W4 与 NET 的 VAL 格读数与 G4-E7
     summary.json（sha 钉住）同格之差 <= 1e-3（R8 家具），否则 INCOMPLETE（说明比较对象不是 G4-E7 的那一个）；差值逐格报告。
  F4 h1（判据 1、2）：G4-H1 的 192 个 VAL 导出（npz sha 与 H1 任务记录核对），ctrl / novel / scale / floating x 24 房 x 噪声种子 1、2。
       臂：E3L5W4_<ts>（判决）、E3L3W4_<ts>（判据 2 的比较对象）、E3L5_<ts>（学到的 w，只报告）；每台机器解码自己的 GaBP 信念
       （= g4_h1_e3l3.e3l3_preds，冻结），读出 R.net_readout(logit, k, j_occ)。
       对照 B4 / B6_A / B6_B 照 OP.evaluate 逐行计算，并与 H1 记录的 rows / zones / kinds / families / phantoms 逐值相等（H1.compare_controls），
       否则 INCOMPLETE。
       Y 区 = 家族 in {ellipsoid, hanging, clutter, pole} 的物体体素并集膨胀 1 格（= OP.object_zones 的家族区之并）；X 区同理
       （rotated、scaled_0.6、scaled_1.5、wall_shelf）。Y 只出现在 novel（ellipsoid / ball、clutter 含桌面杂物、coat_rack）与 floating
       （hanging）里，所以判据 1、2 的格 = (训练种子, 噪声种子) 上把 novel 与 floating 两个变体的 48 张图合并：
         Y 区家具 F1 = Y 区内 tp / fp / fn 合计（臂每台机器权重 1，对照每分量权重 |分量|，= 分量展开）；
         Y 区翻转精确率 = OP.flip_counts(p, B6 自己配对, S 自己配对的表, ..., zone = Y 区) 的 'le_-0' 的区内计数合计（分母 0 记通过，
           同 D8-53 P3）；
         每房幻影块 = OP.phantom_blocks（D8-53 原定义，整房；幻影块按定义不在任何物体区里，所以"只在 Y 区计分"对它不适用，取这 48 张图
           的每房均值）。
       判据 2 的比较：同一训练种子编号、同一噪声种子的 E3L3W4 的 Y 区翻转精确率；臂的分母为 0 记通过，比较对象分母为 0 而臂不为 0 记不过。
  F5 判决（D8-76，写死在 D8_76 / verdict()）：4 格（2 训练种子 x 2 噪声种子）都要满足，判据 1、2 用 H1 噪声种子 1、2，判据 3、4 用 VAL
     噪声种子 3、4（两组 4 格按训练种子与噪声种子的序号配对，只用于把四条判据并在同一张表上，判决是"每条在它自己的 4 格上全过"）：
       1. Y 区翻转精确率 >= .50，Y 区家具 F1 − B6 >= −1 pp，每房幻影块 <= B6 + 2；
       2. Y 区翻转精确率 >= E3L3W4 + .05；
       3. VAL cam_nf 家具 >= E3L3W4 − 0.5 pp；保护项（同 D8-63，对照 = NET）：空闲召回 >= NET − 0.5 pp，表面 F1 >= NET − 0.5 pp，
          未知 -> 已知 FPR 与 NET 逐位相同（|差| <= 1e-12），clean 档 E3L5W4 家具 >= .99，K0_sep >= NET − 3 pp；
       4. 字节：E3L5W4 的每房 O7 字节中位 <= 0.15 x NET 的 (S, k) 字节中位（sum + count，G4-E7 口径）。
     1–4 全过且训练未截断 -> LEVER；判据 1 任一格不过 -> NOT_ENOUGH；其余 AMBIGUOUS（截断只挡 LEVER，同 D8-63）。
     INCOMPLETE：缺任务 / 失败任务 / 缺格 / 缺房、H1 对照闸门不等、已知不一致 > 0、GaBP 核对失败、G4-E7 闸门、缺 run / final / 阈值文件、
     < 2,000 次、两种子更新数不同、混用脚本或检查点版本。
     只报告：X 家族（训练里见过）的读数、按家族与高度拆分、z 分布、训练曲线（每 1,000 次探针）、学到的 w 读法（同一判据算一遍，标
     would_be）、ctrl 自检（E3L5W4 − max(B4, B6) 整体，D8-69 的区间 [+4.3, +8.6] pp）、G4-E7 闸门差值。

用法（总是 python -B；CPU 阶段 CUDA_VISIBLE_DEVICES= ）：
  train     --prereg-updates 9562 [--second-note T] <g4_e3l3 train 的全部参数，--cache 必须是 E3-L5 缓存>
  select    --run R --ctrl CTRL.json [--workers 9]             （学到的 w 与 w = 4 两份阈值）
  eval      --export X --out O --runs R1,R2 --e3l3 DIR --e3l3-criteria C.json --e3r DIR --expect-seeds 3,4 [--workers 9]
  h1        --h1-jobs J --h1-export X --out O --runs R1,R2 --e3l3 DIR --e3l3-criteria C.json --fit FIT.json [--workers 9]
  aggregate --out O --runs R1,R2 --eval-out E --h1-out H --e7-summary S.json
  selfcheck --pyroot P
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, functools, glob, hashlib, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3l3 as L3  # noqa: E402   (frozen; read only)

L2 = L3.L2
L = L3.L
R = L3.R
G = L3.G
import torch  # noqa: E402

BATCH = 'E3-L5'
PREREG = 'D8-76'
ARM5 = 'E3L5'                          # learned w (report only)
ARM5W = 'E3L5W4'                       # w = 4 (judged)
ARM3W = 'E3L3W4'                       # frozen E3-L3 at w = 4 (the comparator of criteria 2 and 3)
NETA = 'NET'
W_JUDGED = 4.0
UPDATES_PREREG = 9562                  # E3-L3's planned count (D8-63 / D8-68); the update rule is E3-L3's
SEEDS = ('342101', '342202')
PAIR = {'342101': 'A', '342202': 'B'}
VAL_SEEDS = ('3', '4')                 # G4-E5 export (criteria 3 / 4)
H1_SEEDS = ('1', '2')                  # G4-H1 export (criteria 1 / 2)
H1_VARIANTS = ('ctrl', 'novel', 'scale', 'floating')
Y_VARIANTS = ('novel', 'floating')     # the only H1 variants holding Y families (F4)
Y_FAMILIES = ('ellipsoid', 'hanging', 'clutter', 'pole')
X_FAMILIES = ('rotated', 'scaled_0.6', 'scaled_1.5', 'wall_shelf')
CONTROLS = ('B4', 'B6_A', 'B6_B')
E7_GATE_TOL = 1e-3
RUN_FORMAT = 1
FROZEN = dict(g4_e3l3='c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c',
              g4_e3l2='bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314',
              g4_e3l='3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be',
              g4_e3r='1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970',
              g4_noise_panels='2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f',
              g4_ood_panel='3a6959da19eec5326f9a17e583d58e6ab1b12fd2a2c398172b61ace36a49ce23',
              g4_h1_e3l3='dc0f69d7c8ea2b2a8513fa2c04095664ebd7eaf0f1c0637d497943b57537da57',
              mapping_3d_storey_data_v420ood='9e1c7c2aac5ddd3e53ed5035a3db707d87c1a1026987ece782796b39f4e1f92c')
E3L3_CKPT_SHA = {'342101': ('1fa0cb396f8f112ff14108691de992c195130513cbd86c83c84481b3155ecf6f', 'c583fa1ebad811db4f8e5ac2dc67fdbb7e2d6271a76b2fac0079b025cf50605e'),
                 '342202': ('3d332d400813374869df58b7c19f6361899812e8d8b2fdc6d9305969c7d4111c', '56f4bf7f9ecb33798dd5a14538f44632195a325ca8754f1399c53d122d2c8c73')}
E7_CRITERIA_PREFIX = '320317aa'        # g4/e7/criteria_w.json (w* = 4, D8-67 supplement)
FIT_SHA256 = '8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2'
H1_SCRIPT_SHA256 = FROZEN['g4_ood_panel']

D8_76 = dict(prereg=PREREG, batch=BATCH, train_seeds=list(SEEDS), judged_w=W_JUDGED,
             h1_seeds=list(H1_SEEDS), h1_variants=list(Y_VARIANTS), y_families=list(Y_FAMILIES), x_families=list(X_FAMILIES), h1_houses=24,
             c1_flip_min=0.50,                 # 1. Y-zone flip precision >= .50
             c1_zone_margin=-0.010,            # 1. Y-zone furniture F1 - B6 (own pairing) >= -1 pp
             c1_phantom_margin=2.0,            # 1. phantom blocks per house <= B6 + 2
             c2_flip_gain=0.05,                # 2. Y-zone flip precision >= E3L3W4 + .05
             val_panel='val', val_seeds=list(VAL_SEEDS), val_houses=24, judged_tier='cam_nf', clean_tier='clean',
             c3_furn_margin=-0.005,            # 3. VAL cam_nf furniture >= E3L3W4 - 0.5 pp
             guard_free=-0.005, guard_surface=-0.005,   # 3. (D8-63 guards vs NET) free recall / surface F1 >= NET - 0.5 pp
             guard_fpr_abs=1e-12,              # 3. unknown -> known FPR identical to NET
             clean_min=0.99,                   # 3. clean-tier furniture >= .99
             k0_margin=-0.03,                  # 3. K0_sep >= NET K0_sep - 3 pp
             c4_bytes_ratio=0.15,              # 4. median per-house O7 bytes (w = 4) <= 0.15 x NET (S, k) bytes
             min_updates=R.MIN_UPDATES, truncated_blocks_lever=True, e7_gate_tol=E7_GATE_TOL, ctrl_band=[0.043, 0.086],
             rule='LEVER iff criteria 1-4 hold in all four cells and no seed is truncated; NOT_ENOUGH iff criterion 1 fails in any cell; else AMBIGUOUS')


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(path + '.tmp', path)


def file_sha256(p):
    return G.file_sha256(p)


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sub_args(rest, *names):
    """read a few forwarded options without consuming them (as g4_e3l4 Q4)"""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for n in names:
        if n == '--smoke':
            p.add_argument(n, action='store_true')
        else:
            p.add_argument(n, default=None)
    a, _ = p.parse_known_args(rest)
    return a


def check_frozen(names=('g4_e3l3', 'g4_e3l2', 'g4_e3l', 'g4_e3r', 'g4_noise_panels')):
    mods = dict(g4_e3l3=L3, g4_e3l2=L2, g4_e3l=L, g4_e3r=R, g4_noise_panels=G)
    out = {}
    for nm in names:
        m = mods.get(nm) or sys.modules.get(nm)
        p = os.path.abspath(m.__file__)
        assert os.path.dirname(p) == HERE, f'{nm} imported from {p}, not {HERE}'
        s = file_sha256(p)
        assert s.startswith(FROZEN[nm]), f'{nm}.py sha {s[:12]} != frozen {FROZEN[nm][:12]}'
        out[nm] = s
    return out


# ================================================================ F1: the cache check and train
def check_cache(cache, manifest_sha=None, smoke=False):
    """the E3-L5 cache: e3l5_div.json written by g4_e3l5_gen for THIS manifest, Y totals 0, formal and ok (smoke: only consistency)"""
    mp = os.path.join(cache, 'manifest.json'); dp = os.path.join(cache, 'e3l5_div.json')
    assert os.path.exists(dp), f'{cache}: no e3l5_div.json (not an E3-L5 cache from g4_e3l5_gen)'
    man_sha = file_sha256(mp); div = _jload(dp)
    assert div.get('batch') == BATCH and div.get('manifest_sha256') == man_sha, 'e3l5_div.json does not belong to this manifest'
    y = div.get('y_totals') or {}
    assert all(int(y.get(k, 1)) == 0 for k in ('objects', 'voxels', 'prims')), f'held-out (Y) totals {y} != 0'
    if not smoke:
        assert div.get('ok') and div.get('formal') and div.get('house_set_equals_e3r'), f"E3-L5 cache not ok / not formal: {div.get('problems')}"
        assert manifest_sha, 'the formal run needs --manifest-sha (the coordinator pins it)'
        assert manifest_sha == man_sha, f'cache manifest sha {man_sha} != pinned {manifest_sha}'
    return man_sha, div


def train(args, rest):
    a = _sub_args(rest, '--out', '--cache', '--manifest-sha', '--updates', '--max-updates', '--seed', '--smoke')
    smoke = bool(args.smoke or a.smoke)
    assert a.out and a.cache and a.updates is not None and a.seed is not None, 'train needs --out, --cache, --updates, --seed'
    man_sha, div = check_cache(a.cache, a.manifest_sha, smoke)
    upd = int(a.updates); mx = int(a.max_updates or 0); preflight = mx > 0
    assert args.prereg_updates == UPDATES_PREREG or smoke, f'--prereg-updates {args.prereg_updates} != {UPDATES_PREREG} (E3-L3 planned count)'
    second = (not preflight) and upd != args.prereg_updates
    if not preflight and not smoke:
        assert upd <= args.prereg_updates, f'--updates {upd} > the pre-registered {args.prereg_updates}'
    if second and not args.second_note.strip() and not smoke:
        raise SystemExit(f'--updates {upd} != {args.prereg_updates}: the second difference must be written down (--second-note)')
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, 'e3l5.json')
    if os.path.exists(mp):
        raise SystemExit(f'{mp} exists (never overwrite)')
    meta = dict(format=RUN_FORMAT, batch=BATCH, prereg=PREREG, seed=int(a.seed), single_variable='training distribution = TRAIN cache (cache-e3l5-div-v1: '
                'v420 50 % + train_div 50 %, X families rotated / scaled / wall_shelf; Y families never present)', cache=os.path.abspath(a.cache),
                cache_manifest_sha256=man_sha, e3l5_div_sha256=file_sha256(os.path.join(a.cache, 'e3l5_div.json')), cache_per_variant=div.get('per_variant'),
                updates_prereg=int(args.prereg_updates), updates=upd, max_updates=mx, preflight=preflight, second_difference=bool(second),
                second_difference_note=(args.second_note.strip() or None), wrapper_sha256=file_sha256(os.path.abspath(__file__)),
                frozen_sha256=check_frozen(), smoke=smoke, utc_start=_utc(), pid=os.getpid(),
                trainer='g4_e3l3.main(["train", ...]) with the forwarded arguments unchanged (model class, w learnable, defaults = E3-L3)')
    _jdump(meta, mp)
    if second:
        print(f'E3-L5 SECOND DIFFERENCE: {upd} updates instead of {args.prereg_updates}: {args.second_note.strip()}', flush=True)
    print(f'E3-L5 train seed {a.seed}: cache {a.cache} (manifest {man_sha[:12]}), {upd} updates{" (preflight, max " + str(mx) + ")" if preflight else ""}', flush=True)
    fin = L3.main(['train'] + list(rest) + (['--smoke'] if args.smoke and not a.smoke else []))
    fp = os.path.join(a.out, 'final.json')
    fj = _jload(fp)
    fj['e3l5'] = dict(meta, utc_end=_utc(), w_final=fj.get('w'))
    _jdump(fj, fp)
    print(f'E3-L5 train seed {a.seed} done: {fj["updates_done"]} / {fj["updates_planned"]} updates ({"complete" if fj["complete"] else "NOT complete"}), '
          f'learned w {fj.get("w")}, second difference: {("yes, " + args.second_note.strip()) if second else "no"}', flush=True)
    return fin


# ================================================================ F2: thresholds at the learned w and at w = 4 (new cache, held-out houses)
_ORIG_LOAD = L3.load_model


def _load_model_fixed_w(path, device='cpu', w=None):
    ck, model = _ORIG_LOAD(path, device)
    with torch.no_grad():
        model.log_w.fill_(math.log(float(w)))
    assert float(model.w.detach()) == float(w), f'exp(float32(log {w})) = {float(model.w.detach())!r} != {w}'
    return ck, model


def _sel_task_w(job):
    """worker (spawn): the frozen L3._sel_task with the model's w fixed at opts['w_fixed']"""
    entry, opts = job
    L3.load_model = functools.partial(_load_model_fixed_w, w=float(opts['w_fixed']))
    try:
        rec = L3._sel_task(job)
    finally:
        L3.load_model = _ORIG_LOAD
    rec['w_fixed'] = float(opts['w_fixed'])
    return rec


def _check_run(run, smoke=False):
    mp = os.path.join(run, 'e3l5.json')
    assert os.path.exists(mp), f'{run}: no e3l5.json (not an E3-L5 run)'
    m = _jload(mp)
    assert m.get('prereg') == PREREG and m.get('batch') == BATCH, f'{run}: e3l5.json prereg {m.get("prereg")}'
    assert smoke or not m.get('smoke'), f'{run}: a smoke run cannot be judged'
    return m


def select_w(run, ctrl_path, cache, w, workers, threads, json_dir, index_file):
    """F2 (ii): = L3.select with the model's w fixed at w; writes <run>/threshold_w4.json (never overwrites)"""
    ctrl = _jload(ctrl_path); runj = _jload(os.path.join(run, 'run.json')); sd = str(runj['seed'])
    ckp = os.path.join(run, 'checkpoints', 'final.pt'); ck = torch.load(ckp, map_location='cpu', weights_only=False)
    assert L2._table_sha(ck['recipe']['evidence']['table']) == ctrl['e3r'][sd]['table_sha256'], 'run evidence table != paired control table'
    cache = cache or runj['cache']; man = _jload(os.path.join(cache, 'manifest.json'))
    L2.check_ctrl(ctrl, cache, runj, sd)
    tp = os.path.join(run, f'threshold_w{w:g}.json')
    if os.path.exists(tp):
        raise SystemExit(f'{tp} exists (never overwrite)')
    ents = [e for e in man['houses'] if e['holdout']]
    assert sorted(e['house'] for e in ents) == ctrl['houses'], 'holdout houses != the control file'
    t0 = time.time()
    opts = dict(ckpt=ckp, cache=cache, seeds=man['seeds'], threads=threads, w_fixed=float(w))
    recs = [r for _, r in R._pool(workers, _sel_task_w, [(e, opts) for e in ents], json_dir, index_file)]
    bad = [r for r in recs if r.get('failed')]
    assert not bad, f"select (w = {w:g}) failed: {[(r['house'], r.get('error')) for r in bad]}"
    A = {wh: np.sum([r['acc'][wh] for r in recs], 0) for wh in ('R8', 'K0')}; tot = {wh: np.sum([r['tot'][wh] for r in recs], 0) for wh in ('R8', 'K0')}
    mism = int(sum(r['known_mismatch'] for r in recs))
    cr = ctrl['readouts'][sd]
    assert mism == 0, f'w = {w:g}: known (Lam_b > 0) differs from k > 0 on {mism} voxels'
    for wh in ('R8', 'K0'):
        assert [int(tot[wh][2]), int(tot[wh][3])] == [int(cr[wh]['counts'][2]), int(cr[wh]['counts'][3])], f'{wh} FPR counts differ from the control'
    sel = L3.pick_r8_k0(A, tot, cr)
    cb = ctrl['bytes'][sd]
    ratios = [r['bytes'][dn] / cb[r['house']][dn] for r in recs for dn in r['bytes']]
    out = dict(stage='select', batch=BATCH, prereg=PREREG, w=float(w), w_label=f'w{w:g}', seed=int(sd), ckpt=ckp, ckpt_sha256=file_sha256(ckp),
               w_learned=float(torch.exp(ck['state_dict']['log_w'].float())), ctrl=os.path.abspath(ctrl_path), ctrl_sha256=file_sha256(ctrl_path),
               houses=ctrl['houses'], n_houses=len(ents), draws=ctrl['draws'], device='cpu', R8=sel['R8'], K0=sel['K0'], known_mismatch=mism,
               holdout_bytes=dict(median_ratio=float(np.median(ratios)), max_ratio=float(np.max(ratios)), n=len(ratios), report_only=True,
                                  median_bytes=float(np.median([b for r in recs for b in r['bytes'].values()]))),
               z=L3.z_summary(np.sum([r['z'] for r in recs], 0)),
               rule=f'= g4_e3l3 select (P10) with w fixed at {w:g} in the GaBP: occupancy pooled free recall >= control NET - 0.25 pp, max furniture '
                    'F1, ties smallest; infeasible -> smallest bin of max free (flag); K0 the same on single agents (w-free), infeasible -> R8 threshold',
               script_sha256=file_sha256(os.path.abspath(__file__)), g4_e3l3_sha256=file_sha256(L3.__file__), seconds=time.time() - t0)
    _jdump(out, tp)
    print(f"E3-L5 select seed {sd} at w = {w:g}: R8 t {out['R8']['t_occ']} F1 {out['R8']['f1']:.4f} free {out['R8']['free']:.4f} (min {out['R8']['free_min']:.4f}, "
          f"feasible {out['R8']['feasible_occ']}) | K0 t {out['K0']['t_occ']} fallback {out['K0']['k0_fallback_r8']} | holdout bytes ratio "
          f"{out['holdout_bytes']['median_ratio']:.4f} | {out['seconds']:.0f}s", flush=True)
    return out


def select(args, rest, json_dir, index_file):
    a = _sub_args(rest, '--run', '--ctrl', '--cache', '--workers', '--threads', '--smoke')
    smoke = bool(args.smoke or a.smoke)
    assert a.run and a.ctrl, 'select needs --run and --ctrl'
    m = _check_run(a.run, smoke)
    check_cache(a.cache or m['cache'], m['cache_manifest_sha256'], smoke)
    if not os.path.exists(os.path.join(a.run, 'threshold.json')):
        L3.main(['select'] + list(rest) + (['--smoke'] if args.smoke and not a.smoke else []))
    else:
        print(f'E3-L5 select: {a.run}/threshold.json exists (learned w; kept)', flush=True)
    return select_w(a.run, a.ctrl, a.cache, W_JUDGED, int(a.workers or 6), int(a.threads or 1), json_dir, index_file)


# ================================================================ F3: eval on the G4-E5 export (VAL, noise seeds 3 / 4)
_MODELS = {}


def _load_arm(spec):
    """(model, w, thr, table_t) of one arm spec; cached per worker process"""
    key = json.dumps(spec, sort_keys=True)
    if key not in _MODELS:
        ck, model = L3.load_model(spec['ckpt'], 'cpu')
        w = float(model.w.detach()) if spec['w'] == 'learned' else float(spec['w'])
        thr = _jload(spec['thr']) if spec.get('thr') else spec['thr_obj']
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        _MODELS[key] = (model, w, thr, tb, str(ck['seed']))
    return _MODELS[key]


def _load_net(spec):
    key = 'NET' + json.dumps(spec, sort_keys=True)
    if key not in _MODELS:
        ckr, net = R.load_ckpt(spec['ckpt'], 'cpu'); thr_r = _jload(spec['thr'])
        _MODELS[key] = (net, R.Evidence.from_recipe(ckr['recipe']['evidence']).table, int(thr_r['j']), int(thr_r['k0']['j_sep']), str(ckr['seed']))
    return _MODELS[key]


def _strip(rows):
    out = []
    for r in rows:
        if isinstance(r, list):
            out.append([{k: v for k, v in r[0].items() if k != 'strata'}, r[1]])
        else:
            out.append({k: v for k, v in r.items() if k != 'strata'})
    return out


def net_block(net, table, j, jk, L_pub, L_k0, F, H, K0T, R8T, judged, chunk):
    """= g4_e3l3_w.net_block (report-only NET there; here the D8-63 guard and byte control): R8 component rows, K0_sep rows, (S, k) bytes"""
    s, msgs, msgn = R.relay_sums(L_pub, F, table)
    by = None
    if judged:
        by = dict(sum=float(sum(G.int_ctx_bytes(*msgs[e]) for e in F['directed'])), count=float(sum(G.int_ctx_bytes(*msgn[e]) for e in F['directed'])))
    lg = R.net_logits(net, [x[0] for x in s], [x[1] for x in s], 'cpu', chunk)
    r8 = [[R.scored(R.net_readout(lg[ci], s[ci][1], j), R8T[ci], H, s[ci][1]), len(cm)] for ci, cm in enumerate(F['comps'])]
    own = [R.stalk(L_k0[a], table) for a in range(len(L_k0))]
    lk = R.net_logits(net, [e for e, _ in own], [k.astype(np.int32) for _, k in own], 'cpu', chunk)
    k0s = [R.scored(R.net_readout(lk[a], own[a][1].astype(np.int32), jk), K0T[a], H, own[a][1].astype(np.int32)) for a in range(len(L_k0))]
    return dict(R8=r8, K0_sep=k0s, bytes=by, j=int(j), j_k0=int(jk))


def eval_task(job):
    rec_e, opts = job
    t0 = time.time()
    rec = dict(stage='eval', batch=BATCH, panel=rec_e['panel'], rooms=rec_e['rooms'], index=rec_e['index'], seed=rec_e['seed'], split=rec_e.get('split'),
               e1_match=rec_e.get('e1_match'), script_sha256=opts['script_sha256'], ckpts=opts['ckpt_shas'], smoke=opts['smoke'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        path = os.path.join(opts['export'], rec_e['file'])
        assert file_sha256(path) == rec_e['file_sha256'], 'export file changed'
        z = np.load(path)
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in rec_e['components']]
        K0T = [G.Target(private[a], H) for a in range(n)]
        R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
        rec['tiers'] = {}; rec['checks'] = {}
        for tn in opts['tiers']:
            if f'L_k0_{tn}' not in z.files:
                continue
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z.files else L_k0
            judged = tn == opts['judged_tier']
            T = {}; C = {}
            for arm, spec in opts['arms'].items():
                if tn not in spec['tiers']:
                    continue
                model, w, thr, tb, sd = _load_arm(spec)
                a = L3.eval_arm(model, w, thr, tb, L_pub, L_k0, F, H, K0T, R8T, opts['chunk'])
                a['thr'] = dict(R8=thr['R8']['j_occ'], K0=thr['K0']['j_occ']); a['w_used'] = w
                a['K0'] = _strip(a['K0']); a['K0_sep'] = _strip(a['K0_sep'])
                if not judged:
                    a['R8'] = _strip(a['R8'])
                T[arm] = a
            if judged:
                for ts, spec in opts['nets'].items():
                    net, table, j, jk, sd = _load_net(spec)
                    nb = net_block(net, table, j, jk, L_pub, L_k0, F, H, K0T, R8T, True, opts['chunk'])
                    nb['K0_sep'] = _strip(nb['K0_sep'])
                    T[f'{NETA}_{ts}'] = nb
            for ts in SEEDS:                                                    # identities (INCOMPLETE if broken)
                a5w, a5, n5 = T.get(f'{ARM5W}_{ts}'), T.get(f'{ARM5}_{ts}'), T.get(f'{NETA}_{ts}')
                if a5w is None:
                    continue
                fa = [r['unknown_known_fpr'] for r, _ in a5w['R8']]
                c = dict(known_mismatch=int(a5w['known_mismatch']) + int((a5 or {}).get('known_mismatch', 0)))
                if a5 is not None:
                    c['fpr_identical_learned'] = bool(fa == [r['unknown_known_fpr'] for r, _ in a5['R8']])
                    c['k0_identical_learned'] = bool(a5w['thr']['K0'] != a5['thr']['K0'] or json.dumps(a5w['K0_sep'], sort_keys=True) == json.dumps(a5['K0_sep'], sort_keys=True))
                if n5 is not None:
                    c['fpr_identical_net'] = bool(fa == [r['unknown_known_fpr'] for r in G._expand(n5['R8'])])
                C[ts] = c
            rec['tiers'][tn] = T; rec['checks'][tn] = C
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def job_tag(r):
    return f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}"


def _eval_and_save(job):
    rec = eval_task(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, job_tag(rec) + '.json')
    assert not os.path.exists(p), f'{p} exists (never overwrite)'
    _jdump(rec, p)
    return rec


def _runs(runs_arg):
    runs = [r for r in runs_arg.split(',') if r]
    out = {}
    for r in runs:
        sd = str(_jload(os.path.join(r, 'run.json'))['seed'])
        out[sd] = r
    return out


def arm_specs(runs, e3l3_dir, e3l3_criteria, e3r_dir, tiers, judged_tier, smoke=False):
    """F3 / F4 arms: E3L5W4 / E3L5 per training seed (run checkpoints + threshold_w4 / threshold), E3L3W4 per seed (frozen E3-L3 + G4-E7
    criteria_w thresholds for w = 4), NET per seed (E3-R checkpoint + threshold, eval only)"""
    crit = _jload(e3l3_criteria)
    csha = file_sha256(e3l3_criteria)
    assert smoke or csha.startswith(E7_CRITERIA_PREFIX), f'G4-E7 criteria sha {csha[:8]} != {E7_CRITERIA_PREFIX}'
    assert float(crit['w_star']) == W_JUDGED, f"G4-E7 w* {crit['w_star']} != {W_JUDGED}"
    arms, nets, shas = {}, {}, {}
    for ts in SEEDS:
        if ts not in runs:
            continue
        r = runs[ts]
        ck = os.path.join(r, 'checkpoints', 'final.pt')
        t5w, t5 = os.path.join(r, 'threshold_w4.json'), os.path.join(r, 'threshold.json')
        assert os.path.exists(t5w) and os.path.exists(t5), f'{r}: threshold.json / threshold_w4.json missing'
        assert float(_jload(t5w)['w']) == W_JUDGED
        arms[f'{ARM5W}_{ts}'] = dict(ckpt=ck, w=W_JUDGED, thr=t5w, tiers=list(tiers))
        arms[f'{ARM5}_{ts}'] = dict(ckpt=ck, w='learned', thr=t5, tiers=list(tiers))
        e3 = os.path.join(e3l3_dir, f'e3l3_s{ts}')
        e3ck, e3th = os.path.join(e3, 'final.pt'), os.path.join(e3, 'threshold.json')
        s1, s2 = file_sha256(e3ck), file_sha256(e3th)
        assert smoke or (s1, s2) == E3L3_CKPT_SHA[ts], f'E3-L3 {ts}: checkpoint / threshold sha {s1[:12]} / {s2[:12]} != pinned'
        arms[f'{ARM3W}_{ts}'] = dict(ckpt=e3ck, w=W_JUDGED, thr_obj=crit['thresholds'][ts]['w4'], tiers=[judged_tier])
        nets[ts] = dict(ckpt=os.path.join(e3r_dir, f's{ts}', 'checkpoints', 'final.pt'), thr=os.path.join(e3r_dir, f's{ts}', 'threshold.json'))
        if not os.path.exists(nets[ts]['ckpt']):
            nets[ts]['ckpt'] = os.path.join(e3r_dir, f's{ts}', 'final.pt')
        shas[ts] = dict(e3l5_ckpt=file_sha256(ck), e3l5_thr=file_sha256(t5), e3l5_thr_w4=file_sha256(t5w), e3l3_ckpt=s1, e3l3_thr=s2,
                        e3r_ckpt=file_sha256(nets[ts]['ckpt']), e3r_thr=file_sha256(nets[ts]['thr']))
        # every arm of a training seed reads the same evidence table as the paired E3-R / B6 (asserted)
        t_a = np.asarray(torch.load(ck, map_location='cpu', weights_only=False)['recipe']['evidence']['table'], np.int64)
        t_b = np.asarray(torch.load(e3ck, map_location='cpu', weights_only=False)['recipe']['evidence']['table'], np.int64)
        t_c = np.asarray(R.Evidence.from_recipe(torch.load(nets[ts]['ckpt'], map_location='cpu', weights_only=False)['recipe']['evidence']).table, np.int64)
        assert np.array_equal(t_a, t_b) and np.array_equal(t_a, t_c), f'{ts}: evidence tables of E3-L5 / E3-L3 / E3-R differ'
    return arms, nets, shas, dict(path=os.path.abspath(e3l3_criteria), sha256=csha)


def export_records(export, seeds, panels, smoke):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(export, 'records', '*.json')))]
    assert recs, f'no export records in {export}/records'
    recs = [r for r in recs if str(r['seed']) in seeds and r['panel'] in panels]
    for r in recs:
        if smoke:
            assert r['split'] == 'TRAIN', f'{job_tag(r)}: the smoke reads TRAIN exports only'
        else:
            assert r['split'] == 'VALIDATION', f'{job_tag(r)}: split {r["split"]} (TEST is never read)'
    return recs


def eval_cmd(args, json_dir, index_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    frozen = check_frozen()
    seeds = [s for s in args.expect_seeds.split(',') if s]
    panels = [p for p in args.panels.split(',') if p]
    if not args.smoke:
        assert sorted(seeds) == sorted(VAL_SEEDS) and set(panels) <= {'val', 'fresh'}, 'the formal eval reads VAL / fresh noise seeds 3 / 4'
    tiers = [t for t in args.tiers.split(',') if t]
    runs = _runs(args.runs)
    for r in runs.values():
        _check_run(r, args.smoke)
    arms, nets, shas, crit = arm_specs(runs, args.e3l3, args.e3l3_criteria, args.e3r, tiers, D8_76['judged_tier'], args.smoke)
    recs = export_records(args.export, seeds, panels, args.smoke)
    if args.only:
        assert args.smoke, '--only is for the smoke'
        want = set(args.only.split(',')); recs = [r for r in recs if job_tag(r) in want]
    jd = os.path.join(args.out, 'jobs')
    if os.path.isdir(jd) and os.listdir(jd):
        raise SystemExit(f'{jd} is not empty (never overwrite)')
    os.makedirs(jd, exist_ok=True)
    me = file_sha256(os.path.abspath(__file__))
    opts = dict(export=args.export, out=args.out, arms=arms, nets=nets, tiers=tiers, judged_tier=D8_76['judged_tier'], threads=args.threads, chunk=args.chunk,
                script_sha256=me, ckpt_shas=shas, smoke=bool(args.smoke))
    _jdump(dict(stage='eval', batch=BATCH, prereg=PREREG, export=os.path.abspath(args.export), seeds=seeds, panels=panels, tiers=tiers, arms=arms, nets=nets,
                shas=shas, e3l3_criteria=crit, frozen_sha256=frozen, script_sha256=me, utc=_utc(), smoke=bool(args.smoke)), os.path.join(args.out, 'eval_run.json'))
    for r in recs:
        if r.get('failed') or r.get('skipped'):
            _jdump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')), skipped=r.get('skipped'),
                        error='export ' + str(r.get('error') or r.get('skipped')), script_sha256=me), os.path.join(jd, job_tag(r) + '.json'))
    jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
    t0 = time.time()
    print(f'E3-L5 eval: {len(jobs)} jobs (seeds {seeds}, panels {panels}, tiers {tiers}), arms {sorted(arms)} + NET, {args.workers} workers, {_utc()}', flush=True)
    for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
        print(f"[{i}/{len(jobs)}] {job_tag(rec)} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}{rec.get('seconds', 0):.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)


def _net_summ(A):
    e = dict(R8=R._summ([x for a in A for x in G._expand(a['R8'])]), K0_sep=R._summ([x for a in A for x in a['K0_sep']]))
    if all(a.get('bytes') for a in A):
        e['bytes_median'] = dict(sum=float(np.median([a['bytes']['sum'] for a in A])), count=float(np.median([a['bytes']['count'] for a in A])))
    return e


def eval_aggregate(out_dir):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(panels={}, failed=[], skipped=[], reasons=[], n_jobs=len(recs), gbp={}, assertions={},
             script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             ckpts=sorted({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}))
    if len(S['script_sha256']) > 1 or len(S['ckpts']) > 1:
        S['reasons'].append('mixed script / checkpoint versions in the eval job files')
    ok = []
    for r in recs:
        (S['failed'] if r.get('failed') else S['skipped'] if r.get('skipped') else ok).append(job_tag(r) if (r.get('failed') or r.get('skipped')) else r)
    for r in ok:
        for tn, C in r['checks'].items():
            for ts, c in C.items():
                a = S['assertions'].setdefault(f'{tn}|{ts}', dict(known_mismatch=0, fpr_identical_learned=True, fpr_identical_net=True, k0_identical_learned=True, jobs=0))
                a['known_mismatch'] += c.get('known_mismatch', 0); a['jobs'] += 1
                for k in ('fpr_identical_learned', 'fpr_identical_net', 'k0_identical_learned'):
                    if k in c:
                        a[k] &= bool(c[k])
        for tn, T in r['tiers'].items():
            for arm, a in T.items():
                if arm.startswith(NETA):
                    continue
                g = S['gbp'].setdefault(arm, dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, jobs=0))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['cf_rel_max'] = max(g['cf_rel_max'], a['gbp']['cf_rel_max'])
                g['violations'] += a['gbp']['violations']; g['components'] += a['gbp']['components']; g['jobs'] += 1
    for g in S['gbp'].values():
        g['pass'] = bool(g['components'] > 0 and g['cf_max'] <= L.CF_TOL and g['cf_rel_max'] <= L.CF_TOL and g['violations'] == 0)
    for panel in sorted({r['panel'] for r in ok}):
        P = dict(seeds={})
        for seed in sorted({str(r['seed']) for r in ok if r['panel'] == panel}):
            X = sorted([r for r in ok if r['panel'] == panel and str(r['seed']) == seed], key=job_tag)
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={})
            for tn in sorted({t for r in X for t in r['tiers']}):
                if not all(tn in r['tiers'] for r in X):
                    S['reasons'].append(f'{panel} s{seed}: tier {tn} missing in some jobs'); continue
                arms = sorted(a for a in X[0]['tiers'][tn] if all(a in r['tiers'][tn] for r in X))
                D['tiers'][tn] = {a: (_net_summ if a.startswith(NETA) else L3._summ_arm)([r['tiers'][tn][a] for r in X]) for a in arms}
            P['seeds'][seed] = D
        S['panels'][panel] = P
    return S


# ================================================================ F4: the H1 panel, Y families only
def _h1mods():
    """lazy: g4_ood_panel / g4_h1_e3l3 blank CUDA_VISIBLE_DEVICES at import (CPU stages only)"""
    import g4_ood_panel as OP
    import g4_h1_e3l3 as H1
    assert OP.E is R and H1.E3 is L3, 'g4_ood_panel / g4_h1_e3l3 must share the frozen g4_e3r / g4_e3l3'
    return OP, H1


def flip_zone_counts(p, pb, S, T, H, zones):
    """report only: the S <= 0 flips (arm occupied, B6 free, scope) inside each zone -> {zone: [true furniture, not]} (= OP.flip_counts 'le_-0')"""
    sel = (p == 1) & (pb == 0) & T.known & H.region & (S * R.UNIT <= -0.0)
    return {zn: [int((sel & T.tr & Z).sum()), int((sel & ~T.tr & Z).sum())] for zn, Z in zones.items()}


def family_union(fzones, names, shape):
    m = np.zeros(shape, bool)
    for f in names:
        if f in fzones:
            m |= fzones[f]
    return m


_HM = None


def h1_models(opts):
    global _HM
    if _HM is None:
        torch.set_num_threads(max(1, int(opts['threads'])))
        assert file_sha256(opts['fit']) == FIT_SHA256 or opts['smoke'], 'E1c fit.json sha mismatch'
        b6 = {h: R.Evidence.from_spec(f"c:{opts['fit']}:{h}:cam_nf") for h in 'AB'}
        arms = {}
        for arm, spec in opts['arms'].items():
            model, w, thr, tb, sd = _load_arm(spec)
            ts = arm.rsplit('_', 1)[1]
            assert sd == ts, f'{arm}: checkpoint seed {sd}'
            assert np.array_equal(tb.numpy().astype(np.int64), b6[PAIR[ts]].table), f'{arm} evidence table != B6_{PAIR[ts]}'
            arms[arm] = dict(model=model, tb=tb, w=w, jo=int(thr['R8']['j_occ']), w_label=('learned' if spec['w'] == 'learned' else f'w{w:g}'), half=PAIR[ts])
        _HM = dict(b6=b6, L=R.Evidence.from_spec('L'), arms=arms)
    return _HM


def h1_file(path, M, rec_seed, gbp_check=True):
    """F4 one exported (variant, house, seed) npz: controls (+ the H1 gate) and every arm on the Y / X / family zones"""
    from scipy import ndimage
    OP, H1 = _h1mods()
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(R.edges_list(z['edges']), n)
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    Lm = list(z['L_k0_cam_nf'])
    occ = z['occ'] == 1; slab = np.asarray(z['slab'], bool); NZ = occ.shape[2]
    near = ndimage.binary_dilation(occ, structure=OP.CUBE)
    floor_k = slab.all((0, 1)) & (np.arange(NZ) < NZ // 2)
    layer1 = int(np.nonzero(floor_k)[0].max()) + 1
    near_nf = ndimage.binary_dilation(occ & ~floor_k[None, None, :], structure=OP.CUBE)
    zone, kzones, fzones = OP.object_zones(z)
    fams = sorted(fzones)
    Yz = family_union(fzones, Y_FAMILIES, occ.shape); Xz = family_union(fzones, X_FAMILIES, occ.shape)
    zones_g = dict(zone=zone, rest=~zone)
    zones_y = dict(Y=Yz, X=Xz, **{f'f:{f}': fzones[f] for f in fams})
    rstats = G.RelayStats()
    sL, _, _ = R.relay_sums(Lm, F, M['L'].table, rstats)
    sums = {h: R.relay_sums(Lm, F, M['b6'][h].table, rstats)[0] for h in 'AB'}
    nc = len(F['comps'])
    kcs = [sL[ci][1] for ci in range(nc)]; known = [kcs[ci] > 0 for ci in range(nc)]
    for h in 'AB':
        for ci in range(nc):
            assert np.array_equal(sums[h][ci][1], kcs[ci]), 'count sums differ between tables'
    preds = {'B4': [R.linear_readout(*sL[ci], 0) for ci in range(nc)]}
    for h in 'AB':
        preds[f'B6_{h}'] = [R.linear_readout(*sums[h][ci], M['b6'][h].theta) for ci in range(nc)]
    res = dict(controls={}, arms={}, families=fams, n_agents=n, components=F['comps'], layer1=layer1, y_zone_voxels=int(Yz.sum()), x_zone_voxels=int(Xz.sum()),
               family_voxels={f: int(np.diff(np.asarray(z['objvox_off']))[np.asarray([str(x) == f for x in z['obj_family']], bool)].sum()) for f in fams})
    gate = {}
    for arm in CONTROLS:                                                 # = OP.evaluate's R8 block (gate) + the Y / X / family zones
        ps = preds[arm]
        assert all(np.array_equal(ps[ci] >= 0, known[ci]) for ci in range(nc)), f'{arm}: known mask != B4'
        rows, acc, kacc, facc, accy, ph = [], {}, {}, {}, {}, np.zeros(4, np.float64)
        for ci, cm in enumerate(F['comps']):
            kc = kcs[ci]; w = len(cm); p = ps[ci]
            rows.append([R.scored(p, R8T[ci], H, kc), w])
            for zn, d in OP.zone_counts(p, R8T[ci], H, kc, zones_g).items():
                OP._addw(acc, zn, d, w)
            for kd, v in OP.kind_counts(p, R8T[ci], H, kzones).items():
                OP._addw(kacc, kd, v, w)
            for fk, v in OP.kind_counts(p, R8T[ci], H, fzones).items():
                OP._addw(facc, fk, v, w)
            for zn, d in OP.zone_counts(p, R8T[ci], H, kc, zones_y).items():
                OP._addw(accy, zn, d, w)
            ph += np.asarray(OP.phantom_blocks(p, kc, H, near, near_nf, layer1 + 1), np.float64) * w / n
        gate[arm] = dict(rows=rows, zones=OP._lists(acc), kinds=OP._lists(kacc), families=OP._lists(facc), phantoms=ph.tolist())
        res['controls'][arm] = dict(rows=rows, zones_y=OP._lists(accy), phantoms=ph.tolist())
    ok, bad = H1.compare_controls(dict(R8=gate, components=F['comps']), rec_seed)
    res['h1_controls_equal'] = bool(ok); res['h1_controls_unequal'] = bad
    for arm, m in M['arms'].items():
        t0 = time.time()
        ep, zq, fpm, info = H1.e3l3_preds(m, Lm, F, kcs, gbp_check)
        own = m['half']; b6 = f'B6_{own}'
        rows, accy, fl, flz, ph, reuse = [], {}, {}, {}, np.zeros(4, np.float64), 0
        for ci, cm in enumerate(F['comps']):
            kc = kcs[ci]; cache = {}
            for p in ep[ci]:
                assert np.array_equal(p >= 0, known[ci]), f'{arm}: prediction known mask != k > 0'
                key = hashlib.sha1(p.tobytes()).hexdigest()
                if key in cache:
                    lay = cache[key]; reuse += 1
                else:
                    S_own = sums[own][ci][0]
                    lay = dict(row=R.scored(p, R8T[ci], H, kc), zc=OP.zone_counts(p, R8T[ci], H, kc, zones_y),
                               ph=np.asarray(OP.phantom_blocks(p, kc, H, near, near_nf, layer1 + 1), np.float64),
                               fl=OP.flip_counts(p, preds[b6][ci], S_own, R8T[ci], H, Yz),
                               fz=flip_zone_counts(p, preds[b6][ci], S_own, R8T[ci], H, zones_y))
                    cache[key] = lay
                rows.append([lay['row'], 1])
                for zn, d in lay['zc'].items():
                    OP._addw(accy, zn, d, 1)
                OP._addw(fl, 'f', lay['fl'], 1); OP._addw(flz, 'f', lay['fz'], 1)
                ph += lay['ph'] / n
        res['arms'][arm] = dict(rows=rows, zones_y=OP._lists(accy), phantoms=ph.tolist(), flips=OP._lists(fl['f']), flips_zone=OP._lists(flz['f']), b6=b6,
                                z=H1.z_zone_counts(zq, fpm, dict(Y=Yz, X=Xz), occ), known_mismatch=info['known_mismatch'], gbp=info['gbp'], w=info['w'],
                                w_label=info['w_label'], j_occ=info['j_occ'], reused_agent_rows=reuse, seconds=time.time() - t0)
    return res


def h1_job(job):
    hpath, opts = job
    t0 = time.time()
    h1 = _jload(hpath)
    rec = dict(stage='h1', batch=BATCH, variant=h1.get('variant'), split=h1.get('split'), panel=h1.get('panel'), rooms=h1.get('rooms'), index=h1.get('index'),
               seeds={}, script_sha256=opts['script_sha256'], ckpts=opts['ckpt_shas'], h1_record_sha256=file_sha256(hpath), smoke=opts['smoke'])
    try:
        if file_sha256(os.path.abspath(__file__)) != opts['script_sha256']:
            raise RuntimeError('g4_e3l5.py changed since the run started')
        OP, H1 = _h1mods()
        if file_sha256(OP.__file__) != FROZEN['g4_ood_panel'] or not file_sha256(H1.__file__).startswith(FROZEN['g4_h1_e3l3']):
            raise RuntimeError('frozen g4_ood_panel / g4_h1_e3l3 sha mismatch')
        if h1.get('script_sha256') != H1_SCRIPT_SHA256 or h1.get('failed') or h1.get('skipped'):
            raise RuntimeError('H1 record not a completed record of the H1 script 3a6959da')
        M = h1_models(opts)
        for seed in opts['seeds']:
            hs = h1['seeds'].get(str(seed))
            if hs is None:
                raise RuntimeError(f'H1 record has no seed {seed}')
            path = os.path.join(opts['export'], hs['file'])
            fs = file_sha256(path)
            if fs != hs['file_sha256']:
                raise RuntimeError(f"{hs['file']}: sha {fs[:12]} != H1 record {hs['file_sha256'][:12]}")
            ts0 = time.time()
            r = h1_file(path, M, hs, opts['gbp_check'])
            r.update(file=hs['file'], file_sha256=fs, seconds=time.time() - ts0)
            rec['seeds'][str(seed)] = r
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:  # noqa: BLE001
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-3000:]
    return rec


def h1_tag(r):
    return f"{r.get('variant')}_{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}"


def _h1_and_save(job):
    rec = h1_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, h1_tag(rec) + '.json')
    assert not os.path.exists(p), f'{p} exists (never overwrite)'
    _jdump(rec, p)
    return rec


def h1_cmd(args, json_dir, index_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    frozen = check_frozen()
    OP, H1 = _h1mods()
    frozen.update(g4_ood_panel=file_sha256(OP.__file__), g4_h1_e3l3=file_sha256(H1.__file__))
    assert frozen['g4_ood_panel'] == FROZEN['g4_ood_panel'] and frozen['g4_h1_e3l3'].startswith(FROZEN['g4_h1_e3l3']), 'frozen H1 scripts changed'
    runs = _runs(args.runs)
    for r in runs.values():
        _check_run(r, args.smoke)
    arms, _, shas, crit = arm_specs(runs, args.e3l3, args.e3l3_criteria, args.e3r, ['cam_nf'], 'cam_nf', args.smoke)
    variants = [v for v in args.variants.split(',') if v]
    assert set(variants) <= set(H1_VARIANTS)
    seeds = [s for s in args.seeds.split(',') if s]
    recs = sorted(glob.glob(os.path.join(args.h1_jobs, '*.json')))
    jobs_p = []
    for p in recs:
        h = _jload(p)
        if h.get('variant') not in variants:
            continue
        if args.smoke:
            assert h.get('split') == 'TRAIN', f'{p}: the smoke reads TRAIN H1 records only'
        else:
            assert h.get('split') == 'VALIDATION' and h.get('panel') == 'val', f'{p}: the formal run reads VAL records only'
        jobs_p.append(p)
    if not args.smoke:
        assert sorted(seeds) == sorted(H1_SEEDS), 'the formal H1 run reads noise seeds 1, 2'
        assert len(jobs_p) == 24 * len(variants), f'{len(jobs_p)} H1 records for {variants} (expected {24 * len(variants)})'
    jd = os.path.join(args.out, 'jobs')
    if os.path.isdir(jd) and os.listdir(jd):
        raise SystemExit(f'{jd} is not empty (never overwrite)')
    os.makedirs(jd, exist_ok=True)
    me = file_sha256(os.path.abspath(__file__))
    opts = dict(out=args.out, export=args.h1_export, arms=arms, seeds=seeds, fit=args.fit, threads=args.threads, gbp_check=not args.no_gbp_check,
                script_sha256=me, ckpt_shas=shas, smoke=bool(args.smoke))
    _jdump(dict(stage='h1', batch=BATCH, prereg=PREREG, variants=variants, seeds=seeds, arms=arms, shas=shas, e3l3_criteria=crit, frozen_sha256=frozen,
                h1_jobs=os.path.abspath(args.h1_jobs), h1_export=os.path.abspath(args.h1_export), fit=dict(path=os.path.abspath(args.fit), sha256=file_sha256(args.fit)),
                script_sha256=me, utc=_utc(), smoke=bool(args.smoke)), os.path.join(args.out, 'h1_run.json'))
    t0 = time.time()
    print(f'E3-L5 h1: {len(jobs_p)} (variant, house) jobs x seeds {seeds}, arms {sorted(arms)}, {args.workers} workers, {_utc()}', flush=True)
    for i, rec in R._pool(args.workers, _h1_and_save, [(p, opts) for p in jobs_p], json_dir, index_file):
        print(f"[{i}/{len(jobs_p)}] {h1_tag(rec)} {rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB "
              f"{'FAILED ' + rec.get('error', '') if rec.get('failed') else 'ok'}  {time.time() - t0:.0f}s", flush=True)


def _sum_c(items):
    return np.sum([np.asarray(x, np.float64) for x in items], 0) if items else None


def _f1c(c):
    if c is None:
        return None
    tp, fp, fn = c[0], c[1], c[2]
    return (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else None


def _prec(v):
    return (v[0] / (v[0] + v[1])) if (v[0] + v[1]) else None


def h1_aggregate(out_dir, smoke=False):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(n_jobs=len(recs), failed=[h1_tag(r) for r in recs if r.get('failed')], h1_mismatch=[], reasons=[], known_mismatch={}, gbp={},
             script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             ckpts=sorted({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}), files={})
    if len(S['script_sha256']) > 1 or len(S['ckpts']) > 1:
        S['reasons'].append('mixed script / checkpoint versions in the H1 job files')
    ok = [r for r in recs if not r.get('failed')]
    for r in ok:
        if not smoke and (r.get('split') != 'VALIDATION' or r.get('panel') != 'val'):
            S['reasons'].append(f'{h1_tag(r)}: not a VAL job')
        for sd, x in r['seeds'].items():
            if not x.get('h1_controls_equal'):
                S['h1_mismatch'].append(f"{h1_tag(r)}_s{sd}:{x.get('h1_controls_unequal')}")
            for arm, a in x['arms'].items():
                S['known_mismatch'][arm] = S['known_mismatch'].get(arm, 0) + int(a['known_mismatch'])
                g = S['gbp'].setdefault(arm, dict(cf_max=0.0, violations=0, checked=True, pass_=True))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['violations'] += a['gbp']['violations']
                g['checked'] &= bool(a['gbp']['checked']); g['pass_'] &= bool(a['gbp']['pass'])
            S['files'].setdefault(r['variant'], {}).setdefault(sd, []).append(dict(house=f"r{r['rooms']}_i{r['index']}", x=x))
    return S


def h1_cells(S, C=D8_76):
    """F4 / F5: per (training seed, noise seed) the pooled Y readouts over the novel + floating maps, plus report-only breakdowns"""
    cells, reasons = {}, []
    for ts in C['train_seeds']:
        arm, cmp_, arm_l = f'{ARM5W}_{ts}', f'{ARM3W}_{ts}', f'{ARM5}_{ts}'
        b6 = f'B6_{PAIR[ts]}'
        for ns in C['h1_seeds']:
            X = [f['x'] for v in C['h1_variants'] for f in S['files'].get(v, {}).get(ns, [])]
            exp_n = C['h1_houses'] * len(C['h1_variants'])
            if len(X) != exp_n:
                reasons.append(f'H1 cell {ts}|{ns}: {len(X)} maps != {exp_n}')
            if not X or any(arm not in x['arms'] or cmp_ not in x['arms'] for x in X):
                reasons.append(f'H1 cell {ts}|{ns}: arm rows missing'); continue

            def ycell(a_name):
                c_arm = _sum_c([x['arms'][a_name]['zones_y']['Y']['c'] for x in X if 'Y' in x['arms'][a_name]['zones_y']])
                fl = _sum_c([x['arms'][a_name]['flips']['le_-0'] for x in X])
                ph = float(np.mean([x['arms'][a_name]['phantoms'][0] for x in X]))
                return dict(f1_Y=_f1c(c_arm), flip_prec_Y=_prec([fl[2], fl[3]]) if fl is not None else None, flip_n_Y=int(fl[2] + fl[3]) if fl is not None else 0,
                            flip_prec_all=_prec([fl[0], fl[1]]) if fl is not None else None, phantoms_per_house=ph)
            cB = _sum_c([x['controls'][b6]['zones_y']['Y']['c'] for x in X if 'Y' in x['controls'][b6]['zones_y']])
            phB = float(np.mean([x['controls'][b6]['phantoms'][0] for x in X]))
            a, c3, al = ycell(arm), ycell(cmp_), ycell(arm_l)
            cell = dict(b6=b6, n_maps=len(X), arm=a, e3l3w4=c3, learned=al, b6_f1_Y=_f1c(cB), b6_phantoms_per_house=phB,
                        d_zone_Y=(a['f1_Y'] - _f1c(cB)) if (a['f1_Y'] is not None and _f1c(cB) is not None) else None,
                        d_flip_vs_e3l3=(a['flip_prec_Y'] - c3['flip_prec_Y']) if (a['flip_prec_Y'] is not None and c3['flip_prec_Y'] is not None) else None)
            cell['families'] = h1_family_rows(X, ts)
            cells[f'{ts}|{ns}'] = cell
    return cells, reasons


def h1_family_rows(X, ts):
    """report only: per family zone F1 (arm / E3L3W4 / learned / B6) with by-height F1, flips inside the family zone"""
    b6 = f'B6_{PAIR[ts]}'
    fams = sorted({k[2:] for x in X for k in x['controls'][b6]['zones_y'] if k.startswith('f:')})
    out = {}
    for f in fams:
        key = f'f:{f}'; row = dict(group=('Y' if f in Y_FAMILIES else 'X' if f in X_FAMILIES else 'other'))
        for nm, get in (('arm', lambda x: x['arms'][f'{ARM5W}_{ts}']), ('e3l3w4', lambda x: x['arms'][f'{ARM3W}_{ts}']),
                        ('learned', lambda x: x['arms'][f'{ARM5}_{ts}']), ('b6', lambda x: x['controls'][b6])):
            zs = [get(x)['zones_y'][key] for x in X if key in get(x)['zones_y']]
            c = _sum_c([z['c'] for z in zs]); h = _sum_c([z['h'] for z in zs])
            row[nm] = dict(f1=_f1c(c), n_true=int(c[5]) if c is not None else 0, by_h=[_f1c(r) for r in h] if h is not None else None)
            if nm != 'b6':
                fz = _sum_c([get(x)['flips_zone'][key] for x in X if key in get(x)['flips_zone']])
                row[nm]['flip_prec'] = _prec(fz) if fz is not None else None
                row[nm]['flip_n'] = int(fz.sum()) if fz is not None else 0
        out[f] = row
    return out


def h1_variant_rows(S, C=D8_76):
    """report only: per variant and seed: overall furniture (row mean, E3-R style), ctrl self-check vs max(B4, B6)"""
    out = {}
    for v, per in S['files'].items():
        for ns, F_ in per.items():
            X = [f['x'] for f in F_]
            row = {}
            for nm in list(CONTROLS) + sorted({a for x in X for a in x['arms']}):
                rows = [r for x in X for r in G._expand((x['controls'] if nm in CONTROLS else x['arms'])[nm]['rows'])] if all(
                    nm in (x['controls'] if nm in CONTROLS else x['arms']) for x in X) else None
                row[nm] = G.summarise_rows(rows)['furn_f1'] if rows else None
            for ts in C['train_seeds']:
                a = row.get(f'{ARM5W}_{ts}')
                best = max((row[c] for c in CONTROLS if row.get(c) is not None), default=None)
                row[f'gain_{ARM5W}_{ts}_vs_max_B'] = (a - best) if (a is not None and best is not None) else None
            out[f'{v}|{ns}'] = dict(maps=len(X), furn=row)
    return out


# ================================================================ F5: the D8-76 verdict
def smoke_criteria(SE, SH):
    """the smoke reads TRAIN maps only (G4-E5 train hold-out export, H1 TRAIN smoke records): same numbers, its own panel / seeds / counts"""
    vp = sorted(SE['panels'])[0] if SE['panels'] else 'none'
    vs = sorted(SE['panels'].get(vp, {}).get('seeds', {}))
    hs = sorted({sd for v in Y_VARIANTS for sd in SH['files'].get(v, {})})
    nh = max([len(SH['files'].get(v, {}).get(s, [])) for v in Y_VARIANTS for s in hs] + [0])
    k = min(len(vs), len(hs))
    return dict(D8_76, val_panel=vp, val_seeds=vs[:k], h1_seeds=hs[:k], h1_houses=nh, min_updates=1, smoke=True)


def val_cells(SE, C=D8_76, check_houses=True):
    cells, reasons = {}, []
    P = SE['panels'].get(C['val_panel'], {}).get('seeds', {})
    exp = {f'r{r}_i{i}' for r, i in G.panel_specs(C['val_panel'])} if check_houses else None
    for ns in C['val_seeds']:
        D = P.get(ns)
        if D is None:
            reasons.append(f'VAL noise seed {ns} missing'); continue
        if check_houses and set(D['houses']) != exp:
            reasons.append(f'VAL noise seed {ns}: {len(set(D["houses"]) & exp)} of {len(exp)} houses')
        jt, kt = D['tiers'].get(C['judged_tier'], {}), D['tiers'].get(C['clean_tier'], {})
        for ts in C['train_seeds']:
            a, c3, net, al, cl = jt.get(f'{ARM5W}_{ts}'), jt.get(f'{ARM3W}_{ts}'), jt.get(f'{NETA}_{ts}'), jt.get(f'{ARM5}_{ts}'), kt.get(f'{ARM5W}_{ts}')
            if a is None or c3 is None or net is None or (cl is None and not C.get('smoke')):      # (the smoke's TRAIN export has no clean tier)
                reasons.append(f'VAL cell {ts}|{ns}: arm / comparator / NET / clean row missing'); continue
            nb = net.get('bytes_median') or {}
            nbt = (nb.get('sum', 0) + nb.get('count', 0)) or None
            if not nbt:
                reasons.append(f'VAL cell {ts}|{ns}: NET bytes missing'); continue
            cell = dict(arm_R8_furn=a['R8']['furn_f1'], e3l3w4_R8_furn=c3['R8']['furn_f1'], d_furn=a['R8']['furn_f1'] - c3['R8']['furn_f1'],
                        net_R8_furn=net['R8']['furn_f1'], arm_minus_net=a['R8']['furn_f1'] - net['R8']['furn_f1'],
                        free_delta=a['R8']['free_recall'] - net['R8']['free_recall'], surface_delta=a['R8']['surface_f1'] - net['R8']['surface_f1'],
                        fpr_delta=a['R8']['unknown_known_fpr'] - net['R8']['unknown_known_fpr'], clean_furn=(cl['R8']['furn_f1'] if cl else None),
                        K0_sep_arm=a['K0_sep']['furn_f1'], K0_sep_net=net['K0_sep']['furn_f1'], d_K0=a['K0_sep']['furn_f1'] - net['K0_sep']['furn_f1'],
                        bytes_median=a['bytes_median'], net_bytes=nbt, bytes_ratio=a['bytes_median'] / nbt, d_furn_tol1=a['R8']['furn_f1_tol1'] - c3['R8']['furn_f1_tol1'],
                        z=a.get('z'), w_used=a.get('w'), self_map=a.get('self_map'), learned=dict(
                            R8_furn=al['R8']['furn_f1'] if al else None, d_furn_vs_e3l3w4=(al['R8']['furn_f1'] - c3['R8']['furn_f1']) if al else None,
                            bytes_ratio=(al['bytes_median'] / nbt) if al else None, w=al.get('w') if al else None))
            cells[f'{ts}|{ns}'] = cell
    return cells, reasons


def e7_gate(SE, e7_path, C=D8_76):
    """F3: the recomputed comparator (E3L3W4) and NET vs G4-E7's summary (same maps, same checkpoints and thresholds)"""
    if not e7_path:
        return dict(ok=False, reason='no G4-E7 summary given', rows={})
    E7 = _jload(e7_path); rows = {}; worst = 0.0
    for ns in C['val_seeds']:
        for ts in C['train_seeds']:
            try:
                mine = SE['panels'][C['val_panel']]['seeds'][ns]['tiers'][C['judged_tier']]
                theirs = E7['panels'][C['val_panel']]['seeds'][ns]['tiers'][C['judged_tier']]
                d1 = abs(mine[f'{ARM3W}_{ts}']['R8']['furn_f1'] - theirs[f'E3L3W_{ts}']['R8']['furn_f1'])
                d2 = abs(mine[f'{NETA}_{ts}']['R8']['furn_f1'] - theirs[f'NET_{ts}']['R8']['furn_f1'])
                d3 = abs(mine[f'{ARM3W}_{ts}']['bytes_median'] - theirs[f'E3L3W_{ts}']['bytes_median'])
                rows[f'{ts}|{ns}'] = dict(d_e3l3w4_furn=d1, d_net_furn=d2, d_e3l3w4_bytes=d3, exact=bool(d1 == 0 and d2 == 0 and d3 == 0))
                worst = max(worst, d1, d2)
            except (KeyError, TypeError) as e:
                rows[f'{ts}|{ns}'] = dict(error=f'missing {e}'); worst = float('inf')
    return dict(ok=bool(worst <= C['e7_gate_tol']), max_abs_diff_furn=worst, rows=rows, path=os.path.abspath(e7_path), sha256=file_sha256(e7_path))


def judge(vcells, hcells, reasons, truncated, C=D8_76, arm_key='arm'):
    """D8-76 hard-coded (F5). Cells are paired by (training seed, k-th noise seed): H1 seeds 1 / 2 <-> VAL seeds 3 / 4."""
    eps = 1e-12
    keys = [(ts, hs, vs) for ts in C['train_seeds'] for hs, vs in zip(C['h1_seeds'], C['val_seeds'])]
    reasons = list(reasons)
    for ts, hs, vs in keys:
        if f'{ts}|{hs}' not in hcells:
            reasons.append(f'H1 cell {ts}|{hs} missing')
        if f'{ts}|{vs}' not in vcells:
            reasons.append(f'VAL cell {ts}|{vs} missing')
    out = dict(criteria=C, rows={}, notes=[])
    if reasons:
        out.update(verdict='INCOMPLETE: ' + '; '.join(reasons), reasons=reasons)
        return out
    allc = {1: True, 2: True, 3: True, 4: True}
    for ts, hs, vs in keys:
        h = hcells[f'{ts}|{hs}']; v = vcells[f'{ts}|{vs}']
        a = h[arm_key]
        fp = a['flip_prec_Y']
        c1 = dict(flip=bool(fp is None or fp >= C['c1_flip_min'] - eps),
                  zone=bool(h['d_zone_Y'] is not None and h['d_zone_Y'] >= C['c1_zone_margin'] - eps) if arm_key == 'arm' else
                  bool(a['f1_Y'] is not None and h['b6_f1_Y'] is not None and a['f1_Y'] - h['b6_f1_Y'] >= C['c1_zone_margin'] - eps),
                  phantom=bool(a['phantoms_per_house'] <= h['b6_phantoms_per_house'] + C['c1_phantom_margin'] + eps))
        cp = h['e3l3w4']['flip_prec_Y']
        c2 = bool(fp is None or (cp is not None and fp >= cp + C['c2_flip_gain'] - eps))
        if arm_key == 'arm':
            g = dict(free=v['free_delta'] >= C['guard_free'] - eps, surface=v['surface_delta'] >= C['guard_surface'] - eps,
                     fpr=abs(v['fpr_delta']) <= C['guard_fpr_abs'], clean=v['clean_furn'] is not None and v['clean_furn'] >= C['clean_min'] - eps,
                     k0=v['d_K0'] >= C['k0_margin'] - eps)
            c3 = dict(furn=bool(v['d_furn'] >= C['c3_furn_margin'] - eps), guards={k: bool(x) for k, x in g.items()})
            c3_ok = c3['furn'] and all(g.values())
            c4 = bool(v['bytes_ratio'] <= C['c4_bytes_ratio'] + eps)
        else:                                                           # learned-w reading (report only): criteria 3 / 4 on its furniture / bytes
            c3 = dict(furn=bool(v['learned']['d_furn_vs_e3l3w4'] is not None and v['learned']['d_furn_vs_e3l3w4'] >= C['c3_furn_margin'] - eps))
            c3_ok = c3['furn']
            c4 = bool(v['learned']['bytes_ratio'] is not None and v['learned']['bytes_ratio'] <= C['c4_bytes_ratio'] + eps)
        row = dict(h1_cell=f'{ts}|{hs}', val_cell=f'{ts}|{vs}', c1=c1, c1_ok=all(c1.values()), c2_ok=c2, c3=c3, c3_ok=c3_ok, c4_ok=c4,
                   flip_prec_Y=fp, e3l3w4_flip_prec_Y=cp, d_zone_Y=(a['f1_Y'] - h['b6_f1_Y']) if (a['f1_Y'] is not None and h['b6_f1_Y'] is not None) else None,
                   phantoms=a['phantoms_per_house'], b6_phantoms=h['b6_phantoms_per_house'], d_furn_val=v['d_furn'] if arm_key == 'arm' else v['learned']['d_furn_vs_e3l3w4'],
                   bytes_ratio=v['bytes_ratio'] if arm_key == 'arm' else v['learned']['bytes_ratio'])
        out['rows'][f'{ts}|{hs}|{vs}'] = row
        allc[1] &= row['c1_ok']; allc[2] &= row['c2_ok']; allc[3] &= row['c3_ok']; allc[4] &= row['c4_ok']
    out['criteria_pass'] = {str(k): bool(x) for k, x in allc.items()}
    if not allc[1]:
        vv = 'NOT_ENOUGH'
    elif all(allc.values()):
        vv = 'LEVER' if not (C['truncated_blocks_lever'] and truncated) else 'AMBIGUOUS (lever blocked: truncated training seed(s) ' + ','.join(truncated) + ')'
    else:
        vv = 'AMBIGUOUS'
    out['verdict'] = vv; out['reasons'] = []
    return out


def run_reasons(runs, C=D8_76, smoke=False):
    reasons, truncated, info = [], [], {}
    for r in runs:
        sd, inf = R._run_info(r, C, reasons, truncated)
        if sd is None:
            continue
        info[sd] = inf
        try:
            m = _check_run(r, smoke)
            info[sd]['e3l5'] = dict(cache_manifest_sha256=m.get('cache_manifest_sha256'), second_difference=m.get('second_difference'),
                                    second_difference_note=m.get('second_difference_note'), updates_prereg=m.get('updates_prereg'))
            fj = _jload(os.path.join(r, 'final.json'))
            if 'e3l5' not in fj:
                reasons.append(f'training seed {sd}: final.json has no e3l5 block')
            if not smoke and int(fj.get('updates_planned', -1)) != UPDATES_PREREG and not (m.get('second_difference') and m.get('second_difference_note')):
                reasons.append(f'training seed {sd}: {fj.get("updates_planned")} planned updates != {UPDATES_PREREG} and no second difference written down')
            if not os.path.exists(os.path.join(r, 'threshold_w4.json')):
                reasons.append(f'training seed {sd}: no threshold_w4.json')
        except (AssertionError, OSError, ValueError) as e:
            reasons.append(f'training seed {sd}: {e}')
    for sd in C['train_seeds']:
        if sd not in info:
            reasons.append(f'training seed {sd} missing')
    if len({i['updates_planned'] for i in info.values()}) > 1:
        reasons.append(f"training seeds have different update counts {sorted({i['updates_planned'] for i in info.values()})}")
    if len({(i.get('e3l5') or {}).get('cache_manifest_sha256') for i in info.values()}) > 1:
        reasons.append('training seeds used different caches')
    return reasons, truncated, info


def _probe_digest(run):
    try:
        Hh = _jload(os.path.join(run, 'history.json'))
    except Exception:
        return None
    out = []
    for h in Hh:
        p = h.get('probe')
        if p:
            out.append(dict(updates=h['updates'], furn_f1_at0=p.get('furn_f1_at0'), furn_f1_oracle=p.get('furn_f1_oracle'), bytes_ratio_median=p.get('bytes_ratio_median'),
                            z_minus=p.get('z_minus'), z_zero=p.get('z_zero'), z_plus=p.get('z_plus'), z_entropy_bits=p.get('z_entropy_bits'),
                            tanh_saturation=p.get('tanh_saturation'), w=p.get('w'), known_mismatch=p.get('known_mismatch'), holdout_bce=h.get('holdout_bce')))
    return out


def aggregate_cmd(args):
    p = os.path.join(args.out, 'summary.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} exists (never overwrite)')
    os.makedirs(args.out, exist_ok=True)
    runs = [r for r in args.runs.split(',') if r]
    SE = eval_aggregate(args.eval_out); SH = h1_aggregate(args.h1_out, smoke=args.smoke)
    C = smoke_criteria(SE, SH) if args.smoke else D8_76
    rr, truncated, rinfo = run_reasons(runs, C, smoke=args.smoke)
    reasons = list(rr) + SE['reasons'] + SH['reasons']
    if SE['failed']:
        reasons.append(f"failed eval jobs: {SE['failed'][:6]}")
    if SE['skipped']:
        reasons.append(f"skipped eval jobs: {SE['skipped'][:6]}")
    if SH['failed']:
        reasons.append(f"failed H1 jobs: {SH['failed'][:6]}")
    if SH['h1_mismatch']:
        reasons.append(f"{len(SH['h1_mismatch'])} H1 maps: B4 / B6 != the H1 record: {SH['h1_mismatch'][:4]}")
    for k, a in SE['assertions'].items():
        if a['known_mismatch']:
            reasons.append(f"eval {k}: known (Lam_b > 0) != k > 0 on {a['known_mismatch']} voxels")
        if not a['fpr_identical_learned']:
            reasons.append(f'eval {k}: R8 FPR differs between the w = 4 and learned-w arms')
        if not a['k0_identical_learned']:
            reasons.append(f'eval {k}: K0_sep rows differ between the w = 4 and learned-w arms at the same K0 threshold')
    for arm, n_ in SH['known_mismatch'].items():
        if n_:
            reasons.append(f'H1 {arm}: known (Lam_b > 0) != k > 0 on {n_} voxels')
    for nm, gb in (('eval', SE['gbp']), ('H1', SH['gbp'])):
        for arm, g in gb.items():
            if not g.get('pass', g.get('pass_')):
                reasons.append(f'{nm} {arm}: GaBP closed form / energy check failed')
    for ts in D8_76['train_seeds']:
        for arm in (f'{ARM5W}_{ts}', f'{ARM3W}_{ts}'):
            if arm not in SE['gbp']:
                reasons.append(f'eval: no GaBP checks for {arm}')
            if arm not in SH['gbp']:
                reasons.append(f'H1: no GaBP checks for {arm}')
    gate = e7_gate(SE, args.e7_summary, C) if not args.smoke else dict(ok=None, note='smoke: no G4-E7 gate')
    if not args.smoke and not gate['ok']:
        reasons.append(f"G4-E7 gate: the recomputed E3L3W4 / NET differ from G4-E7 by {gate.get('max_abs_diff_furn')} (> {E7_GATE_TOL}) or are missing")
    vcells, vr = val_cells(SE, C, check_houses=not args.smoke); hcells, hr = h1_cells(SH, C)
    dec = judge(vcells, hcells, reasons + vr + hr, truncated, C)
    would = judge(vcells, hcells, vr + hr, truncated, C, arm_key='learned') if not (vr or hr) else dict(verdict='n/a (cells missing)')
    if args.smoke:
        dec['verdict'] = 'SMOKE (not a verdict): ' + dec['verdict']
    S = dict(batch=BATCH, prereg=PREREG, decision=dec, learned_w_would_be=dict(verdict=would.get('verdict'), criteria_pass=would.get('criteria_pass'),
                                                                              rows=would.get('rows'), note='report only (D8-76: the verdict reads w = 4)'),
             val_cells=vcells, h1_cells=hcells, h1_variants=h1_variant_rows(SH), e7_gate=gate, runs=rinfo, truncated_seeds=truncated,
             training_probes={r: _probe_digest(r) for r in runs}, eval_summary=dict(gbp=SE['gbp'], assertions=SE['assertions'], failed=SE['failed'],
                                                                                     n_jobs=SE['n_jobs'], panels=SE['panels']),
             h1_summary=dict(gbp=SH['gbp'], known_mismatch=SH['known_mismatch'], failed=SH['failed'], h1_mismatch=SH['h1_mismatch'], n_jobs=SH['n_jobs']),
             script_sha256=file_sha256(os.path.abspath(__file__)), utc=_utc(), smoke=bool(args.smoke))
    _jdump(S, p)
    print(f"E3-L5 verdict: {dec['verdict']}", flush=True)
    for k, r in (dec.get('rows') or {}).items():
        print(f"  cell {k}: c1 {r['c1']} flip_Y {r['flip_prec_Y']} (E3-L3 w4 {r['e3l3w4_flip_prec_Y']}) dzone_Y {r['d_zone_Y']} phantoms {r['phantoms']:.2f} vs B6 "
              f"{r['b6_phantoms']:.2f} | c2 {r['c2_ok']} | c3 {r['c3']} d_furn {r['d_furn_val']} | c4 {r['c4_ok']} bytes x NET {r['bytes_ratio']}", flush=True)
    print(f"  learned-w would be: {would.get('verdict')}; G4-E7 gate ok {gate['ok']} (max diff {gate.get('max_abs_diff_furn')})", flush=True)
    return S


# ================================================================ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'select', 'eval', 'h1', 'aggregate', 'selfcheck'):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--smoke', action='store_true')
        if name == 'train':
            p.add_argument('--prereg-updates', type=int, required=True); p.add_argument('--second-note', default='')
        if name in ('eval', 'h1'):
            p.add_argument('--out', required=True); p.add_argument('--runs', required=True); p.add_argument('--e3l3', required=True)
            p.add_argument('--e3l3-criteria', required=True); p.add_argument('--e3r', required=True)
            p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=1)
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--expect-seeds', required=True); p.add_argument('--panels', default='val')
            p.add_argument('--tiers', default='cam_nf,clean'); p.add_argument('--chunk', type=int, default=2); p.add_argument('--only', default='')
        if name == 'h1':
            p.add_argument('--h1-jobs', required=True); p.add_argument('--h1-export', required=True); p.add_argument('--fit', required=True)
            p.add_argument('--variants', default=','.join(H1_VARIANTS)); p.add_argument('--seeds', default=','.join(H1_SEEDS))
            p.add_argument('--no-gbp-check', action='store_true')
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--runs', required=True); p.add_argument('--eval-out', required=True)
            p.add_argument('--h1-out', required=True); p.add_argument('--e7-summary', default='')
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd not in ('train', 'select'):
        ap.error(f'unrecognised arguments: {rest}')
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    for opt, val in (('--houseexpo-json-dir', args.houseexpo_json_dir), ('--houseexpo-index', args.houseexpo_index)):
        if val and args.cmd in ('train', 'select'):
            rest = list(rest) + [opt, val]
    G.setup_env(json_dir, index_file)
    if args.cmd == 'train':
        return train(args, rest)
    if args.cmd == 'select':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return select(args, rest, json_dir, index_file)
    if args.cmd == 'eval':
        return eval_cmd(args, json_dir, index_file)
    if args.cmd == 'h1':
        return h1_cmd(args, json_dir, index_file)
    if args.cmd == 'aggregate':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return aggregate_cmd(args)
    if args.cmd == 'selfcheck':
        fz = check_frozen()
        print(f'E3-L5 frozen ok: {json.dumps({k: v[:12] for k, v in fz.items()})}', flush=True)
        return R.main(['selfcheck', '--pyroot', args.pyroot, '--houseexpo-json-dir', json_dir, '--houseexpo-index', index_file])


if __name__ == '__main__':
    main()
