"""g4_e3l3_w.py (Claude, 2026-10-28): 批次 G4-E7 "冻结 E3-L3，评估时取有限 w*"（预注册 OVERNIGHT_REPORT D8-67；纯 CPU，0 个租约）。
冻结依赖只读 import：g4_e3l3.py（sha c7523cc8...：E3-L3 模型、source、eval_arm、pick_r8_k0、z_stats / z_summary、_summ_arm），经它
g4_e3l2.py（bab73d64...）、g4_e3l.py（3dc970d8...：gabp / gbp_checks / wire_bytes / msg_ints / control_holdout / encode_maps / decode_np）、
g4_e3r.py（1d65d9a8...：relay_sums、net_logits、net_readout、scored、_pool）、g4_noise_panels.py（2313d75b...）。G4-E5 的导出由
g4_quant_wire.py（a63ce7d2...）写出，这里只读它的 npz 与 records（核对 sha），不 import。本文件不改任何已有文件。

底线（D8-63 / D8-66 / D8-67）：学习隐码 x_i；二次 sheaf 能量
    E = sum_i sum_v 1/2 Lam_i(v) (x_i(v) - z_i(v))^2 + sum_(i,j) in F sum_v 1/2 w (x_i(v) - x_j(v))^2
上的两遍 GaBP（L.gabp，与 E3-L3 逐字相同：逐跳步长 1 取整；O7 字节；已知 = Lam_b > 0），然后每台机器只解码自己的信念。
单变量：评估时的 w（学到的约 1,240 / 1,358 -> 有限 w*）。C3'（D8-66）：融合是二次 sheaf 能量上固定、与任务无关的逐跳整数 GaBP；
整数求和是 w -> inf 的特例。有限 w 时消息精度 Lam_{j->i} = w Lam~ / (w + Lam~) < w，w 就是码率旋钮。

操作化（D8-67 没写死、由本实现写死的部分）
  W1 第 0 步 wselect（CPU；在看任何种子 3、4 读数之前）：TRAIN 留出 19 房 x cam_nf 抽样 200-202，取 G4-E5 的导出
     g4/e5_export/train_holdout（每个 (房, 抽样) 一个 npz；逐个按 records 核对 sha；房表与抽样必须等于 E3-L3 官方 threshold.json 的
     houses / draws）。读法与 E3-L3 select（E._sel_task）逐行相同：encode_maps chunk 4、decode chunk 2、单线程、R8 = 分量展开（每台机器解码
     自己的信念，对分量 R8 目标，目标不带 mult）、K0 = 各机自己的源端势。为每个 w in {学到的 w, 3, 4, 6, 8} 累计直方图，用冻结的
     E.pick_r8_k0 选阈值（空闲召回 >= 对照 - 0.25 pp、家具 F1 最大、同分取最小；K0 不可行时用 R8 阈值）。
     对照读数（空闲召回 / FPR / 家具）= 冻结的 L.control_holdout（配对 E3-R NET 与其阈值）在同一批导出上重算；字节对照 = NET 的 (S, k)
     字节（D8-54：R.relay_sums 的 sum 与 count 消息，G.int_ctx_bytes，全部有向边；与 E3-L2 ctrlprep 逐字相同）。
     断言：每个 w 的已知不一致体素 = 0，R8 / K0 的 FPR 计数与对照逐个相等；有限 w 的 float64 GaBP 与闭式解差 <= 1e-5、能量单调（记录）。
     对账：学到的 w 的阈值 j_occ（R8、K0）必须等于官方 threshold.json，否则不选 w*（INCOMPLETE）；F1、空闲召回、FPR、对照读数、留出字节比
     与 z 统计逐项记录与官方值的差。
  W2 w* 规则（D8-67）：取同时满足 (a) 两个检查点的留出合并家具 F1 都 >= 学到的 w 下的值 - 0.15 pp、(b) 两个检查点的每 (房, 抽样) 字节比
     中位都 <= 0.13 的最大 w（两检查点同一个 w*）。没有 w 满足 (b)：在满足 (a) 的 w 中取字节最小者（两检查点中位比的较大者最小，同分取
     较大 w），写明"预计够不到 0.15 倍"。没有 w 满足 (a)：w* = None，本批不跑。写 criteria_w.json（不覆盖），打印其 sha256。
  W3 eval：VAL / fresh x 噪声种子 3、4（判决）与 1、2（连续性）x cam_nf / cam / clean x 检查点 342101 / 342202。每个 (档, 检查点) 用冻结的
     E.eval_arm 算两次，只换 w 与阈值：臂 E3L3W = w*（criteria_w.json 为 w* 选的阈值），对照 E3L3 = 学到的 w（官方 threshold.json）。
     eval_arm 内含：在该 w 下 float64 GaBP 与闭式解差、能量单调（L.gbp_checks）；已知 Lam_b > 0 与 k > 0 不一致体素计数；O7 字节；量化与
     浮点信念差；机间信念差。另在任务内核对：两臂 K0_sep 行逐位相同（K0 不走消息）；R8 各行未知->已知 FPR 逐位相同。
     NET（只报告的绝对线）：配对 E3-R 检查点与其阈值（j、k0.j_sep），在同一组地图上照 R.eval_job 的 NET 段计 R8 / K0_sep 与 cam_nf 的
     (S, k) 字节。判决种子 3、4 只在 --confirm-criteria-sha 等于 criteria_w.json 的 sha（协调者确认 w* 之后）时运行；脚本 sha 必须等于
     criteria_w.json 记录的 sha；检查点 / 阈值 / 导出的 sha 逐个核对。job 文件不覆盖。为控制体积，K0 / K0_sep 行与非 cam_nf 档的 R8 行
     去掉 strata（只用于分层报告，不进判据）。
  W4 aggregate：D8-67 判据写死在 D8_67 与 judge() 里（见 judge 的说明）。臂与对照的汇总用冻结的 E._summ_arm（R8 = 分量展开行均值）；
     每房家具 = 该房分量展开行的均值。只报告：相对 NET 的绝对线（家具 >= NET - 1 pp？字节 <= 0.15 x NET 的 (S, k) 字节？）、配对池化家具
     F1（两臂同一组计分体素上合计 TP / FP / FN）与配对逐行均值（只取两臂都有定义的行）及其跨线是否与判决读数一致、机间信念差、量化与浮点
     信念差、按 k 与高度分层、容差 1、种子 1、2 的连续性（并与 E3-L3 官方 summary.json 的对照臂逐项对账）、NET 与 G4-E5 Qinf 行对账。

用法（总是 python -B；CUDA_VISIBLE_DEVICES= ；<= 10 个进程）：
  wselect   [--out g4/e7/criteria_w.json] [--workers 10] [--smoke --only TRAIN_r1_i19,...]
  eval      --criteria C.json --out O --seeds 3,4 [--panels val,fresh] --confirm-criteria-sha SHA [--workers 10] [--smoke --only tag,...]
  aggregate --out O --criteria C.json [--smoke]
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, hashlib, json, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3l3 as E  # noqa: E402   (frozen; read only; imports g4_e3l2, g4_e3l, g4_e3r, g4_noise_panels)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
L = E.L
L2 = E.L2
R = E.R
G = E.G
import torch  # noqa: E402

UNK = R.UNK
NBIN = R.NBIN
ROOT = G.ROOT
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
TRAIN_HO = os.path.join(AN, 'g4', 'e5_export', 'train_holdout')
EXPORT_NEW = os.path.join(AN, 'g4', 'e5_export')            # VAL / fresh noise seeds 3, 4 (G4-E5)
EXPORT_OLD = os.path.join(INP, 'export')                     # VAL / fresh noise seeds 1, 2 (E3-R)
EXPORTS = {3: EXPORT_NEW, 4: EXPORT_NEW, 1: EXPORT_OLD, 2: EXPORT_OLD}
EXPORT_SCRIPT = {3: 'a63ce7d2', 4: 'a63ce7d2', 1: '1d65d9a8', 2: '1d65d9a8', 200: 'a63ce7d2', 201: 'a63ce7d2', 202: 'a63ce7d2'}
E3L3_SUMMARY = os.path.join(AN, 'g4', 'e3l3', 'summary.json')
E5_JOBS = os.path.join(AN, 'g4', 'e5', 'jobs')
OUT_DIR = os.path.join(AN, 'g4', 'e7')
CRITERIA_DEFAULT = os.path.join(OUT_DIR, 'criteria_w.json')
SMOKE_DIR = os.path.join(OUT_DIR, 'smoke')
CKPTS = {'342101': dict(e3l3=os.path.join(INP, 'e3l3_s342101'), e3l3_prefix='1fa0cb39', e3r=os.path.join(INP, 's342101'), e3r_prefix='1b035fd1'),
         '342202': dict(e3l3=os.path.join(INP, 'e3l3_s342202'), e3l3_prefix='3d332d40', e3r=os.path.join(INP, 's342202'), e3r_prefix='a58b2b05')}
FROZEN = {'g4_e3l3.py': ('c7523cc8', E), 'g4_e3l2.py': ('bab73d64', L2), 'g4_e3l.py': ('3dc970d8', L), 'g4_e3r.py': ('1d65d9a8', R),
          'g4_noise_panels.py': ('2313d75b', G)}
W_GRID = (3.0, 4.0, 6.0, 8.0)
W_FURN_MARGIN = 0.0015                 # W2 (a): pooled hold-out furniture F1 >= learned-w value - 0.15 pp, both checkpoints
W_BYTES_MAX = 0.13                     # W2 (b): median per-(house, draw) byte ratio vs the NET (S, k) control <= 0.13, both checkpoints
HO_SEEDS = (200, 201, 202)
JUDGE_SEEDS = (3, 4)
CONT_SEEDS = (1, 2)
ARM = 'E3L3W'                          # finite w*
CTRL = 'E3L3'                          # learned w (E3-L3 as run)
NETA = 'NET'                           # paired E3-R (report only)
MAX_WORKERS = 10
FORMAT = 1

D8_67 = dict(prereg='D8-67', batch='G4-E7', judged_panel='val', report_panel='fresh', judged_tier='cam_nf', cam_tier='cam', clean_tier='clean',
             train_seeds=['342101', '342202'], noise_seeds=['3', '4'], continuity_seeds=['1', '2'], houses_per_cell=24,
             lever_furn_margin=-0.003,          # R8 furniture d >= -0.3 pp (4 cells)
             lever_worst_house=-0.020,          # every cell's worst single-house d >= -2.0 pp
             lever_bytes_ratio=0.60,            # median per-house O7 bytes, arm / control <= 0.60
             guard_free=-0.003, guard_surface=-0.003,   # free recall / surface F1 d >= -0.3 pp
             clean_min=0.99,                    # clean-tier arm R8 furniture >= .99
             cf_tol=L.CF_TOL,                   # GaBP == closed form (absolute and relative <= 1e-5), energy monotone
             cam_demote_mean=-0.010,            # cam (unregistered) 4-cell mean d < -1.0 pp -> LEVER demoted to AMBIGUOUS
             fresh_demote_mean=-0.003,          # fresh 4-cell mean d < -0.3 pp -> LEVER demoted to AMBIGUOUS
             no_furn_margin=-0.010, no_bytes_ratio=0.80,
             net_furn_line=-0.010, net_bytes_line=0.15,   # report only (absolute lines vs NET)
             w_grid=list(W_GRID), w_furn_margin=W_FURN_MARGIN, w_bytes_max=W_BYTES_MAX,
             assertions_incomplete='known mask (Lam_b > 0 == k > 0) and identical between arms, K0_sep rows identical, R8 FPR identical, '
                                   'GaBP closed form / energy at the arm and control w: a failure -> INCOMPLETE (D8-67 "断言失败 -> INCOMPLETE")',
             note='margin guards (free / surface / clean) failing -> NO; demotions only demote LEVER (只降不升)')


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def file_sha256(p):
    return G.file_sha256(p)


def utc():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def wlab(w):
    return f'w{float(w):g}'


def check_frozen():
    """the frozen modules are the reviewed versions (sha prefixes) and are imported from this directory"""
    out = {}
    for name, (pfx, mod) in FROZEN.items():
        p = os.path.abspath(mod.__file__)
        assert os.path.dirname(p) == HERE, f'{name} imported from {p}, not {HERE}'
        s = file_sha256(p)
        assert s.startswith(pfx), f'{name} sha {s[:8]} != frozen {pfx}'
        out[name] = s
    return out


def ckpt_paths(sd):
    c = CKPTS[sd]
    return dict(e3l3_ckpt=os.path.join(c['e3l3'], 'final.pt'), e3l3_thr=os.path.join(c['e3l3'], 'threshold.json'),
                e3r_ckpt=os.path.join(c['e3r'], 'final.pt'), e3r_thr=os.path.join(c['e3r'], 'threshold.json'))


def ckpt_shas(sd):
    p = ckpt_paths(sd); c = CKPTS[sd]
    s = {k: file_sha256(v) for k, v in p.items()}
    assert s['e3l3_ckpt'].startswith(c['e3l3_prefix']), f'E3-L3 {sd} checkpoint sha {s["e3l3_ckpt"][:8]} != {c["e3l3_prefix"]}'
    assert s['e3r_ckpt'].startswith(c['e3r_prefix']), f'E3-R {sd} checkpoint sha {s["e3r_ckpt"][:8]} != {c["e3r_prefix"]}'
    return s


# ================================================================ W1: the TRAIN hold-out houses (G4-E5 export, one npz per draw)
class HOHouse:
    """one TRAIN hold-out house from the G4-E5 export in the layout E3-L3's select read from the E3-R cache (R.HouseData / L.LHouse):
    draws = ['clean', 'cam_nf_s200', 'cam_nf_s201', 'cam_nf_s202'] (draw 0 is never read), maps[d] = [n, X, Y, Z] int8, occ = (occ == 1)"""

    def __init__(self, export_dir, recs):
        recs = sorted(recs, key=lambda r: int(r['seed']))
        assert len({(int(r['rooms']), int(r['index'])) for r in recs}) == 1, 'records of different houses'
        self.name = R.house_name(int(recs[0]['rooms']), int(recs[0]['index']))
        self.draws = ['clean'] + [f'cam_nf_s{int(r["seed"])}' for r in recs]
        self.maps = [None]; ref = None
        for r in recs:
            p = os.path.join(export_dir, r['file'])
            assert file_sha256(p) == r['file_sha256'], f'{p}: sha256 differs from its export record'
            with np.load(p) as z:
                assert 'L_k0_cam_nf' in z.files and 'L_pub_cam_nf' not in z.files, f'{p}: cam_nf must have one map per agent'
                cur = {k: np.asarray(z[k]) for k in ('occ', 'slab', 'obj', 'cell', 'private', 'edges')}
                mp = np.ascontiguousarray(z['L_k0_cam_nf'])
            if ref is None:
                ref = cur
            else:
                for k in ref:
                    assert np.array_equal(ref[k], cur[k]), f'{self.name}: {k} differs between draws'
            assert mp.dtype == np.int8 and mp.shape == (ref['private'].shape[0],) + ref['occ'].shape
            self.maps.append(mp)
        self.private = ref['private']; self.n = int(self.private.shape[0])
        self.occ = ref['occ'] == 1; self.slab = ref['slab']; self.obj = ref['obj']; self.cell = ref['cell']
        self.edges = R.edges_list(ref['edges']); self.F = G.forest(self.edges, self.n)
        self.comps = [list(map(int, c)) for c in self.F['comps']]
        for r in recs:
            assert self.comps == [list(map(int, c)) for c in r['components']], 'forest components != export record'
        self._fl = {}

    def local(self, members):
        key = tuple(members)
        if key not in self._fl:
            self._fl[key] = L.local_forest(self.F, key)
        return self._fl[key]


def ho_records(export_dir=TRAIN_HO, only=None):
    """{house name: [3 records (seeds 200-202)]} of the non-skipped TRAIN hold-out export records"""
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(export_dir, 'records', '*.json')))]
    assert recs, f'no export records in {export_dir}/records'
    out = {}
    for r in recs:
        if r.get('skipped'):
            continue
        assert not r.get('failed'), f"export record failed: {r.get('file')}"
        assert r['split'] == 'TRAIN' and int(r['seed']) in HO_SEEDS and r['script_sha256'].startswith(EXPORT_SCRIPT[200]), r.get('file')
        out.setdefault(R.house_name(int(r['rooms']), int(r['index'])), []).append(r)
    for h, rs in out.items():
        assert sorted(int(r['seed']) for r in rs) == list(HO_SEEDS), f'{h}: draws {sorted(r["seed"] for r in rs)}'
    if only:
        miss = sorted(set(only) - set(out))
        assert not miss, f'--only houses not in the export: {miss}'
        out = {h: out[h] for h in only}
    return out


def _add(acc, tot, lg, pk, T, Hh):
    """= E._sel_task's add(): occupancy-logit histograms (target furniture / other region / free) on predicted-known voxels + totals"""
    ob = R.logit_bin(lg)
    acc[0] += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NBIN)
    acc[1] += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NBIN)
    acc[2] += np.bincount(ob[pk & T.free], minlength=NBIN)
    unk = ~T.known
    tot += [int(T.tr.sum()), int(T.free.sum()), int((pk & unk).sum()), int(unk.sum())]


def _gbp_acc(g, c):
    g['cf_max'] = max(g['cf_max'], c['cf_max']); g['cf_rel_max'] = max(g['cf_rel_max'], c['cf_rel'])
    g['violations'] += c['violations']; g['components'] += 1; g['steps'] += c['steps']


def _wsel_task(job):
    """W1, one (house, checkpoint): per w the R8 histograms, O7 bytes per draw, known mismatches, GaBP checks; K0 histograms (w-free);
    the paired NET control: (S, k) bytes per draw and the frozen L.control_holdout counts"""
    hrecs, opts = job
    sd = opts['seed']; rec = dict(stage='wselect', seed=sd, house=None)
    t0 = time.time()
    try:
        torch.set_num_threads(max(1, opts['threads']))
        Hd = HOHouse(opts['export'], hrecs); rec['house'] = Hd.name; rec['draws'] = Hd.draws[1:]
        ck, model = E.load_model(opts['e3l3_ckpt'], 'cpu'); wl = float(model.w.detach())
        table = np.asarray(ck['recipe']['evidence']['table']); tb = torch.as_tensor(table, dtype=torch.int32)
        ws = [('learned', wl)] + [(wlab(w), float(w)) for w in opts['w_grid']]
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]; K0T = [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
        acc = {lab: np.zeros((3, NBIN), np.int64) for lab, _ in ws}; tot = {lab: np.zeros(4, np.int64) for lab, _ in ws}
        acck = np.zeros((3, NBIN), np.int64); totk = np.zeros(4, np.int64)
        mism = {lab: 0 for lab, _ in ws}; byts = {lab: {} for lab, _ in ws}; zc = np.zeros(5, np.int64)
        gbp = {lab: dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, steps=0) for lab, _ in ws}
        lam_max = {lab: 0 for lab, _ in ws}
        with torch.no_grad():
            for d in range(1, len(Hd.draws)):
                maps = Hd.maps[d]; dn = Hd.draws[d]
                zhat, lam = L.encode_maps(model, list(maps), tb, 'cpu')
                h, lq, z = E.source(zhat, lam, 'round'); zc += E.z_stats(z, zhat, lam)
                for lab, w in ws:
                    by = 0.0
                    for ci, cm in enumerate(Hd.comps):
                        Fl = Hd.local(cm)
                        hb, lb, msgs = L.gabp(h[cm], lq[cm], Fl, w, 'round')
                        mi = [L.msg_ints(*msgs[e]) for e in Fl['directed']]
                        by += sum(L.wire_bytes(*x) for x in mi)
                        lam_max[lab] = max([lam_max[lab]] + [int(x[1].max()) for x in mi if x[1].size])
                        if opts['checks']:
                            _gbp_acc(gbp[lab], L.gbp_checks(z[cm].unsqueeze(1), lq[cm], Fl, w))
                        kc = (maps[cm] != UNK).sum(0); lbn = lb.numpy()
                        for i, lg in enumerate(L.decode_np(model, hb, lb)):
                            pk = lbn[i] > 0; mism[lab] += int((pk != (kc > 0)).sum()); _add(acc[lab], tot[lab], lg, pk, R8T[ci], Hh)
                    byts[lab][dn] = float(by)
                for a, lg in enumerate(L.decode_np(model, h, lq)):
                    _add(acck, totk, lg, maps[a] != UNK, K0T[a], Hh)
        ckr = torch.load(opts['e3r_ckpt'], map_location='cpu', weights_only=False)
        tab_r = R.Evidence.from_recipe(ckr['recipe']['evidence']).table; del ckr
        assert np.array_equal(np.asarray(tab_r, np.int64), table.astype(np.int64)), 'E3-L3 evidence table != the paired E3-R table'
        cby = {}
        for d in range(1, len(Hd.draws)):                               # = E3-L2 _ctrl_task (D8-54 control bytes)
            _, msgs, msgn = R.relay_sums(list(Hd.maps[d]), Hd.F, tab_r)
            cby[Hd.draws[d]] = float(sum(G.int_ctx_bytes(*msgs[e]) for e in Hd.F['directed']) + sum(G.int_ctx_bytes(*msgn[e]) for e in Hd.F['directed']))
        c = L.control_holdout(opts['e3r_ckpt'], opts['e3r_thr'], [Hd], 'cpu')
        rec.update(w_learned=wl, labels=[lab for lab, _ in ws], w={lab: w for lab, w in ws}, acc={k: v.tolist() for k, v in acc.items()},
                   tot={k: v.tolist() for k, v in tot.items()}, k0_acc=acck.tolist(), k0_tot=totk.tolist(), known_mismatch=mism, bytes=byts,
                   ctrl_bytes=cby, ctrl_counts=dict(R8=c['R8']['counts'], K0=c['K0']['counts']), gbp=gbp, lam_msg_max=lam_max, z=zc.tolist(),
                   seconds=time.time() - t0, peak_mb=G._peak_mb())
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _readout(c):
    c = np.asarray(c, np.int64)
    return dict(free=c[0] / max(c[1], 1), fpr=c[2] / max(c[3], 1), furn_f1=2 * c[4] / max(2 * c[4] + c[5] + c[6], 1), counts=c.tolist())


def summarise_seed(recs, w_grid=W_GRID):
    """W1 for one checkpoint: control readouts, per-w thresholds (E.pick_r8_k0), byte ratios; asserts known masks and FPR counts"""
    labs = ['learned'] + [wlab(w) for w in w_grid]
    for r in recs:
        assert r['labels'] == labs, 'w labels differ between houses'
    cr = dict(R8=_readout(np.sum([r['ctrl_counts']['R8'] for r in recs], 0)), K0=_readout(np.sum([r['ctrl_counts']['K0'] for r in recs], 0)))
    Ak = np.sum([r['k0_acc'] for r in recs], 0); Tk = np.sum([r['k0_tot'] for r in recs], 0)
    out = dict(control=cr, w={}, w_learned=recs[0]['w_learned'])
    assert len({r['w_learned'] for r in recs}) == 1
    for lab in labs:
        A = np.sum([r['acc'][lab] for r in recs], 0); T = np.sum([r['tot'][lab] for r in recs], 0)
        mism = int(sum(r['known_mismatch'][lab] for r in recs))
        assert mism == 0, f'{lab}: known (Lam_b > 0) differs from k > 0 on {mism} voxels'
        for wh, tt in (('R8', T), ('K0', Tk)):
            assert [int(tt[2]), int(tt[3])] == [int(cr[wh]['counts'][2]), int(cr[wh]['counts'][3])], f'{lab} {wh}: FPR counts differ from the control'
        sel = E.pick_r8_k0({'R8': A, 'K0': Ak}, {'R8': T, 'K0': Tk}, cr)
        ratios = [r['bytes'][lab][dn] / r['ctrl_bytes'][dn] for r in recs for dn in r['draws']]
        vs_l = [r['bytes'][lab][dn] / r['bytes']['learned'][dn] for r in recs for dn in r['draws']]
        g = dict(cf_max=max(r['gbp'][lab]['cf_max'] for r in recs), cf_rel_max=max(r['gbp'][lab]['cf_rel_max'] for r in recs),
                 violations=int(sum(r['gbp'][lab]['violations'] for r in recs)), components=int(sum(r['gbp'][lab]['components'] for r in recs)))
        g['pass'] = bool(g['components'] > 0 and g['cf_max'] <= L.CF_TOL and g['cf_rel_max'] <= L.CF_TOL and g['violations'] == 0)
        out['w'][lab] = dict(w=recs[0]['w'][lab], R8=sel['R8'], K0=sel['K0'], furn_f1=sel['R8']['f1'], bytes_ratio_median=float(np.median(ratios)),
                             bytes_ratio_max=float(np.max(ratios)), bytes_ratio_min=float(np.min(ratios)), n=len(ratios),
                             bytes_median=float(np.median([r['bytes'][lab][dn] for r in recs for dn in r['draws']])),
                             bytes_vs_learned_median=float(np.median(vs_l)), known_mismatch=mism, gbp=g,
                             lam_msg_max=int(max(r['lam_msg_max'][lab] for r in recs)))
    out['z'] = E.z_summary(np.sum([r['z'] for r in recs], 0))
    out['ctrl_bytes_median'] = float(np.median([r['ctrl_bytes'][dn] for r in recs for dn in r['draws']]))
    return out


def official_check(ss, off):
    """W1: the learned-w re-selection on the export vs E3-L3's official threshold.json (computed on the E3-R cache on bigboy)"""
    a = ss['w']['learned']; cmp = {}
    for wh in ('R8', 'K0'):
        for k in ('j_occ', 'f1', 'free', 'fpr', 'free_min', 'feasible_occ'):
            cmp[f'{wh}.{k}'] = (a[wh][k], off[wh][k])
        for k in ('free', 'fpr', 'furn_f1'):
            cmp[f'{wh}.control.{k}'] = (ss['control'][wh][k], off[wh]['control'][k])
    cmp['K0.k0_fallback_r8'] = (a['K0'].get('k0_fallback_r8'), off['K0'].get('k0_fallback_r8'))
    hb = off.get('holdout_bytes', {})
    cmp['holdout_bytes.median_ratio'] = (a['bytes_ratio_median'], hb.get('median_ratio'))
    cmp['holdout_bytes.max_ratio'] = (a['bytes_ratio_max'], hb.get('max_ratio'))
    cmp['holdout_bytes.median_bytes'] = (a['bytes_median'], hb.get('median_bytes'))
    cmp['holdout_bytes.n'] = (a['n'], hb.get('n'))
    for k in ('z_minus', 'z_zero', 'z_plus', 'z_entropy_bits', 'tanh_saturation', 'footprint'):
        cmp[f'z.{k}'] = (ss['z'][k], off.get('z', {}).get(k))
    rows = {}
    for k, (mine, theirs) in cmp.items():
        if isinstance(mine, (bool, np.bool_)) or isinstance(theirs, bool) or theirs is None:
            rows[k] = dict(mine=mine, official=theirs, equal=bool(mine == theirs))
        else:
            rows[k] = dict(mine=mine, official=theirs, equal=bool(mine == theirs), abs_diff=abs(float(mine) - float(theirs)))
    j_ok = rows['R8.j_occ']['equal'] and rows['K0.j_occ']['equal']
    return dict(rows=rows, j_occ_equal=bool(j_ok), all_equal=bool(all(r['equal'] for r in rows.values())),
                max_abs_diff=max([r.get('abs_diff', 0.0) for r in rows.values()] + [0.0]))


def choose_w(per_seed, w_grid=W_GRID, furn_margin=W_FURN_MARGIN, bytes_max=W_BYTES_MAX):
    """W2 (D8-67): the largest w meeting (a) furniture and (b) bytes on BOTH checkpoints; else, among (a), the smallest bytes (max over the
    checkpoints of the median ratio; ties -> larger w) with the note 'expected to miss 0.15 x'; no w meets (a) -> None (batch not run)"""
    eps = 1e-12; rows = {}
    for w in w_grid:
        lab = wlab(w)
        fa = {sd: bool(s['w'][lab]['furn_f1'] >= s['w']['learned']['furn_f1'] - furn_margin - eps) for sd, s in per_seed.items()}
        fb = {sd: bool(s['w'][lab]['bytes_ratio_median'] <= bytes_max + eps) for sd, s in per_seed.items()}
        rows[lab] = dict(w=float(w), furn_ok=fa, bytes_ok=fb, furn_ok_all=all(fa.values()), bytes_ok_all=all(fb.values()),
                         d_furn={sd: s['w'][lab]['furn_f1'] - s['w']['learned']['furn_f1'] for sd, s in per_seed.items()},
                         bytes_ratio_median={sd: s['w'][lab]['bytes_ratio_median'] for sd, s in per_seed.items()},
                         worst_bytes_ratio=max(s['w'][lab]['bytes_ratio_median'] for s in per_seed.values()))
    both = [r['w'] for r in rows.values() if r['furn_ok_all'] and r['bytes_ok_all']]
    if both:
        return max(both), 'both', None, rows
    fa = [r for r in rows.values() if r['furn_ok_all']]
    if fa:
        best = min(fa, key=lambda r: (r['worst_bytes_ratio'], -r['w']))
        return best['w'], 'furniture_only', '预计够不到 0.15 倍（没有 w 满足字节条件 <= 0.13；取满足家具条件、字节最小的 w）', rows
    return None, 'none', '没有 w 满足家具条件：本批不跑', rows


def wselect(args, json_dir, index_file):
    frozen = check_frozen()
    qsha = file_sha256(os.path.join(HERE, 'g4_quant_wire.py'))
    assert qsha.startswith(EXPORT_SCRIPT[200]), f'g4_quant_wire.py sha {qsha[:8]} != the G4-E5 export script {EXPORT_SCRIPT[200]}'
    frozen['g4_quant_wire.py (export writer, not imported)'] = qsha
    out = args.out or (os.path.join(SMOKE_DIR, 'criteria_w.json') if args.smoke else CRITERIA_DEFAULT)
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (never overwrite)')
    assert args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} processes'
    assert args.smoke or not args.only, '--only is for the smoke'
    t0 = time.time(); started = utc()
    only = [h for h in args.only.split(',') if h] if args.only else None
    houses = ho_records(TRAIN_HO, only)
    seeds = list(CKPTS)
    off = {sd: _jload(ckpt_paths(sd)['e3l3_thr']) for sd in seeds}
    names = sorted(houses)
    if not args.smoke:
        for sd in seeds:
            assert names == sorted(off[sd]['houses']), f'hold-out houses != the official E3-L3 threshold.json ({sd})'
            assert off[sd]['draws'] == [f'cam_nf_s{s}' for s in HO_SEEDS]
    shas = {sd: ckpt_shas(sd) for sd in seeds}
    rec_list = sorted((r['file'], r['file_sha256']) for rs in houses.values() for r in rs)
    rec_sha = hashlib.sha256(json.dumps(rec_list).encode()).hexdigest()
    jobs = []
    for sd in seeds:
        o = dict(seed=sd, export=TRAIN_HO, threads=args.threads, w_grid=list(W_GRID), checks=True, **ckpt_paths(sd))
        jobs += [(houses[h], o) for h in names]
    jd = os.path.join(os.path.dirname(out), 'wselect_jobs')
    if os.path.isdir(jd) and os.listdir(jd):
        raise SystemExit(f'{jd} is not empty (never overwrite)')
    os.makedirs(jd, exist_ok=True)
    print(f'G4-E7 wselect: {len(names)} hold-out houses x {len(seeds)} checkpoints, w {list(W_GRID)} + learned, {args.workers} workers, {started}', flush=True)
    recs = []
    for i, r in R._pool(args.workers, _wsel_task, jobs, json_dir, index_file):
        recs.append(r)
        _jdump(r, os.path.join(jd, f"{r['seed']}_{r.get('house') or 'failed_' + str(i)}.json"))
        print(f"[{i}/{len(jobs)}] {r['seed']} {r.get('house')} {'FAILED ' + r.get('error', '') if r.get('failed') else ''}{r.get('seconds', 0):.0f}s  "
              f"{time.time() - t0:.0f}s", flush=True)
    bad = [(r['seed'], r.get('house'), r.get('error')) for r in recs if r.get('failed')]
    assert not bad, f'wselect failed: {bad}'
    per_seed = {}; checks = {}
    for sd in seeds:
        rs = sorted([r for r in recs if r['seed'] == sd], key=lambda r: r['house'])
        assert [r['house'] for r in rs] == names
        per_seed[sd] = summarise_seed(rs)
        checks[sd] = official_check(per_seed[sd], off[sd]) if not args.smoke else None
    reasons = []
    if not args.smoke:
        for sd in seeds:
            if not checks[sd]['j_occ_equal']:
                reasons.append(f'{sd}: learned-w thresholds re-selected on the export != the official threshold.json')
    for sd in seeds:
        for lab, a in per_seed[sd]['w'].items():
            if not a['gbp']['pass']:
                reasons.append(f'{sd} {lab}: GaBP closed form / energy check failed {a["gbp"]}')
    w_star, branch, note, rows = choose_w(per_seed)
    if reasons:
        w_star, branch, note = None, 'incomplete', 'INCOMPLETE: ' + '; '.join(reasons)
    thresholds = {sd: {lab: dict(R8=a['R8'], K0=a['K0'], w=a['w']) for lab, a in per_seed[sd]['w'].items()} for sd in seeds}
    crit = dict(format=FORMAT, stage='wselect', batch='G4-E7', prereg='D8-67', smoke=bool(args.smoke), utc_start=started, utc_written=utc(),
                w_star=w_star, w_star_label=(wlab(w_star) if w_star is not None else None), branch=branch, note=note, reasons=reasons,
                rule=('W2 (D8-67): the largest w in {3, 4, 6, 8} with (a) pooled hold-out furniture F1 >= learned-w value - 0.15 pp and (b) median '
                      'per-(house, draw) O7 byte ratio vs the NET (S, k) bytes <= 0.13, both checkpoints; none meets (b) -> among (a) the smallest '
                      'bytes (max over checkpoints of the median ratio, ties -> larger w), noted; none meets (a) -> not run'),
                threshold_rule='per w: E.pick_r8_k0 (pooled free recall >= control NET - 0.25 pp, furniture F1 max, ties smallest; K0 the same on '
                               'single agents, infeasible -> R8 threshold); control = L.control_holdout of the paired E3-R on the same export',
                table=rows, per_seed=per_seed, official_check=checks, thresholds=thresholds, houses=names, draws=[f'cam_nf_s{s}' for s in HO_SEEDS],
                n_houses=len(names), export=dict(dir=os.path.relpath(TRAIN_HO, ROOT), records_sha256=rec_sha, n_files=len(rec_list)),
                checkpoints={sd: dict({k: os.path.relpath(v, ROOT) for k, v in ckpt_paths(sd).items()}, sha256=shas[sd],
                                      w_learned=per_seed[sd]['w_learned']) for sd in seeds},
                eval_criteria=D8_67, script_sha256=file_sha256(os.path.abspath(__file__)), frozen_sha256=frozen,
                torch=torch.__version__, threads=args.threads, workers=args.workers, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (never overwrite)')
    _jdump(crit, out)
    sha = file_sha256(out)
    print(f'G4-E7 wselect: w* = {w_star} ({branch}){" -- " + note if note else ""}', flush=True)
    for lab in ['learned'] + [wlab(w) for w in W_GRID]:
        print('  ' + lab.ljust(8) + ' | '.join(f"{sd}: F1 {per_seed[sd]['w'][lab]['furn_f1']:.4f} (d {100 * (per_seed[sd]['w'][lab]['furn_f1'] - per_seed[sd]['w']['learned']['furn_f1']):+.3f} pp) "
                                              f"free {per_seed[sd]['w'][lab]['R8']['free']:.4f} t {per_seed[sd]['w'][lab]['R8']['t_occ']:+.2f} "
                                              f"bytes x NET {per_seed[sd]['w'][lab]['bytes_ratio_median']:.4f} (x learned {per_seed[sd]['w'][lab]['bytes_vs_learned_median']:.3f})"
                                              for sd in seeds), flush=True)
    if not args.smoke:
        for sd in seeds:
            print(f"  official check {sd}: j_occ equal {checks[sd]['j_occ_equal']}, all equal {checks[sd]['all_equal']}, max abs diff {checks[sd]['max_abs_diff']:.3e}", flush=True)
    print(f'criteria_w.json sha256 {sha}  ({out})', flush=True)
    return crit, sha


# ================================================================ W3: eval (CPU): both arms through the frozen E.eval_arm, + NET (report)
def _strip(rows):
    out = []
    for r in rows:
        if isinstance(r, list):
            out.append([{k: v for k, v in r[0].items() if k != 'strata'}, r[1]])
        else:
            out.append({k: v for k, v in r.items() if k != 'strata'})
    return out


def net_block(net, table, j, jk, L_pub, L_k0, F, H, K0T, R8T, tn, chunk):
    """report only: the paired E3-R NET exactly as R.eval_job's NET part (R8 component rows, K0_sep rows, cam_nf (S, k) bytes)"""
    s, msgs, msgn = R.relay_sums(L_pub, F, table)
    by = None
    if tn == R.JUDGED_TIER:
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
    rec = dict(stage='eval', batch='G4-E7', panel=rec_e['panel'], rooms=rec_e['rooms'], index=rec_e['index'], seed=rec_e['seed'], split=rec_e.get('split'),
               e1_match=rec_e.get('e1_match'), script_sha256=opts['script_sha256'], criteria_sha256=opts['criteria_sha256'], ckpts=opts['ckpt_shas'],
               w_star=opts['w_star'], smoke=opts['smoke'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        path = os.path.join(opts['exports'][str(rec_e['seed'])], rec_e['file'])
        assert file_sha256(path) == rec_e['file_sha256'], 'export file changed'
        z = np.load(path)
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in rec_e['components']]
        K0T = [G.Target(private[a], H) for a in range(n)]
        R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
        models = []
        for sd in opts['seeds_ckpt']:
            p = opts['paths'][sd]
            ck, model = E.load_model(p['e3l3_ckpt'], 'cpu'); wl = float(model.w.detach())
            tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
            ckr, net = R.load_ckpt(p['e3r_ckpt'], 'cpu'); thr_r = _jload(p['e3r_thr'])
            tab_r = R.Evidence.from_recipe(ckr['recipe']['evidence']).table
            assert np.array_equal(np.asarray(tab_r, np.int64), tb.numpy().astype(np.int64)), 'E3-L3 evidence table != paired E3-R table'
            models.append((sd, model, wl, _jload(p['e3l3_thr']), opts['thr_arm'][sd], tb, net, tab_r, int(thr_r['j']), int(thr_r['k0']['j_sep'])))
        rec['tiers'] = {}; rec['checks'] = {}; rec['w_learned'] = {}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            T = {}; C = {}
            for sd, model, wl, thr_c, thr_a, tb, net, tab_r, jn, jkn in models:
                rec['w_learned'][sd] = wl
                c = E.eval_arm(model, wl, thr_c, tb, L_pub, L_k0, F, H, K0T, R8T, opts['chunk'])
                a = E.eval_arm(model, float(opts['w_star']), thr_a, tb, L_pub, L_k0, F, H, K0T, R8T, opts['chunk'])
                nb = net_block(net, tab_r, jn, jkn, L_pub, L_k0, F, H, K0T, R8T, tn, opts['chunk'])
                fa = [r['unknown_known_fpr'] for r, _ in a['R8']]; fc = [r['unknown_known_fpr'] for r, _ in c['R8']]
                fn_ = [r['unknown_known_fpr'] for r in G._expand(nb['R8'])]
                C[sd] = dict(k0_identical=bool(json.dumps(a['K0_sep'], sort_keys=True) == json.dumps(c['K0_sep'], sort_keys=True)),
                             fpr_identical=bool(fa == fc), net_fpr_identical=bool(fa == fn_), known_mismatch_arm=int(a['known_mismatch']),
                             known_mismatch_ctrl=int(c['known_mismatch']), lost_cells_arm=int(a['qdiff']['lost_cells']), lost_cells_ctrl=int(c['qdiff']['lost_cells']))
                keep = tn == R.JUDGED_TIER
                for x in (a, c):
                    x['K0'] = _strip(x['K0']); x['K0_sep'] = _strip(x['K0_sep'])
                    if not keep:
                        x['R8'] = _strip(x['R8'])
                nb['K0_sep'] = _strip(nb['K0_sep'])
                if not keep:
                    nb['R8'] = _strip(nb['R8'])
                a['thr'] = dict(R8=thr_a['R8']['j_occ'], K0=thr_a['K0']['j_occ']); c['thr'] = dict(R8=thr_c['R8']['j_occ'], K0=thr_c['K0']['j_occ'])
                T[f'{ARM}_{sd}'] = a; T[f'{CTRL}_{sd}'] = c; T[f'{NETA}_{sd}'] = nb
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


def export_records(seeds, panels):
    """the VAL / fresh export records of the given noise seeds (seeds 3, 4: G4-E5; 1, 2: E3-R); never TEST"""
    out = []
    for s in seeds:
        rs = [_jload(p) for p in sorted(glob.glob(os.path.join(EXPORTS[s], 'records', '*.json')))]
        rs = [r for r in rs if int(r['seed']) == s and r['panel'] in panels]
        for r in rs:
            assert r['split'] in ('VALIDATION',), f"{job_tag(r)}: split {r['split']}"
            assert r.get('skipped') or r.get('failed') or r['script_sha256'].startswith(EXPORT_SCRIPT[s]), f'{job_tag(r)}: export script sha'
        out += rs
    return out


def load_criteria(path, smoke):
    crit = _jload(path); sha = file_sha256(path)
    assert crit.get('stage') == 'wselect' and crit.get('batch') == 'G4-E7', 'not a G4-E7 criteria file'
    assert smoke or not crit.get('smoke'), 'a smoke criteria file cannot drive the real eval'
    assert crit.get('w_star') is not None, f"criteria w* is None ({crit.get('note')}): the batch is not run"
    return crit, sha


def eval_cmd(args, json_dir, index_file):
    check_frozen()
    assert args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} processes'
    crit, csha = load_criteria(args.criteria, args.smoke)
    me = file_sha256(os.path.abspath(__file__))
    if not args.smoke:
        assert crit['script_sha256'] == me, f"script sha {me[:8]} != the sha that wrote the criteria {crit['script_sha256'][:8]} (sha mismatch -> refuse)"
    seeds = [int(s) for s in args.seeds.split(',') if s]
    panels = [p for p in args.panels.split(',') if p]
    assert set(panels) <= {'val', 'fresh'} and set(seeds) <= set(JUDGE_SEEDS) | set(CONT_SEEDS), 'panels val / fresh, noise seeds 1-4 only (no TEST)'
    if set(seeds) & set(JUDGE_SEEDS):
        assert not args.smoke, 'the smoke never reads the judged noise seeds 3, 4'
        assert args.confirm_criteria_sha == csha, ('judged noise seeds 3, 4 need --confirm-criteria-sha equal to the criteria file sha '
                                                   f'(the coordinator confirms w* first); file sha {csha}')
    paths = {sd: ckpt_paths(sd) for sd in CKPTS}
    shas = {sd: ckpt_shas(sd) for sd in CKPTS}
    for sd in CKPTS:
        assert shas[sd] == crit['checkpoints'][sd]['sha256'], f'{sd}: checkpoint / threshold sha differs from the criteria file'
    lab = crit['w_star_label']
    thr_arm = {sd: crit['thresholds'][sd][lab] for sd in CKPTS}
    recs = export_records(seeds, panels)
    if args.only:
        assert args.smoke, '--only is for the smoke'
        want = set(args.only.split(',')); recs = [r for r in recs if job_tag(r) in want]
        assert {job_tag(r) for r in recs} == want, f'--only tags not found: {sorted(want - {job_tag(r) for r in recs})}'
    jd = os.path.join(args.out, 'jobs'); os.makedirs(jd, exist_ok=True)
    clash = [job_tag(r) for r in recs if os.path.exists(os.path.join(jd, job_tag(r) + '.json'))]
    if clash:
        raise SystemExit(f'{len(clash)} job files exist in {jd} (never overwrite), e.g. {clash[:3]}')
    opts = dict(exports={str(s): EXPORTS[s] for s in seeds}, out=args.out, seeds_ckpt=list(CKPTS), paths=paths, thr_arm=thr_arm, w_star=float(crit['w_star']),
                tiers=list(R.EVAL_TIERS), threads=args.threads, chunk=args.chunk, script_sha256=me, criteria_sha256=csha,
                ckpt_shas={sd: shas[sd]['e3l3_ckpt'] for sd in CKPTS}, smoke=bool(args.smoke))
    for r in recs:
        if r.get('failed') or r.get('skipped'):
            _jdump(dict(stage='eval', batch='G4-E7', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                        skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped')), script_sha256=me, criteria_sha256=csha),
                   os.path.join(jd, job_tag(r) + '.json'))
    jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
    t0 = time.time()
    print(f'G4-E7 eval: {len(jobs)} jobs (seeds {seeds}, panels {panels}), w* = {crit["w_star"]}, criteria {csha[:12]}, {args.workers} workers x '
          f'{args.threads} thread(s), {utc()}', flush=True)
    for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
        print(f"[{i}/{len(jobs)}] {job_tag(rec)} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}{rec.get('seconds', 0):.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
    print(f'G4-E7 eval done {utc()}', flush=True)


# ================================================================ W4: aggregation and the D8-67 verdict
def _house_furn(rows):
    return G._mean([r.get('furn_f1') for r in G._expand(rows)])


def _net_summ(A):
    e = dict(R8=R._summ([x for a in A for x in G._expand(a['R8'])]), K0_sep=R._summ([x for a in A for x in a['K0_sep']]))
    if all(a.get('bytes') for a in A):
        e['bytes_median'] = dict(sum=float(np.median([a['bytes']['sum'] for a in A])), count=float(np.median([a['bytes']['count'] for a in A])))
    return e


def _paired(ra, rc):
    """report only (R-I 草案 D): pooled furniture F1 of the two arms on the same scored voxels and the paired row mean (rows where both defined)"""
    xa = G._expand(ra); xc = G._expand(rc)
    assert len(xa) == len(xc)
    both = [(a['furn_f1'], c['furn_f1']) for a, c in zip(xa, xc) if a.get('furn_f1') is not None and c.get('furn_f1') is not None]
    return dict(pooled_arm=G._pooled_f1(xa), pooled_ctrl=G._pooled_f1(xc), rows=len(both),
                rowmean_arm=float(np.mean([a for a, _ in both])) if both else None, rowmean_ctrl=float(np.mean([c for _, c in both])) if both else None)


def aggregate(out_dir, crit, csha, smoke=False):
    me = file_sha256(os.path.abspath(__file__))
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(panels={}, failed=[], skipped=[], reasons=[], script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             criteria_sha256=sorted({r.get('criteria_sha256') for r in recs if r.get('criteria_sha256')}), assertions={}, n_jobs=len(recs))
    if S['script_sha256'] != [me] and recs:
        S['reasons'].append(f"job script sha {S['script_sha256']} != this script {me}")
    if S['criteria_sha256'] != [csha] and recs:
        S['reasons'].append(f"job criteria sha {S['criteria_sha256']} != the criteria file {csha}")
    if not smoke and crit['script_sha256'] != me:
        S['reasons'].append('the criteria file was written by another script version')
    if len({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}) > 1:
        S['reasons'].append('mixed checkpoint versions in the job files')
    if any(bool(r.get('smoke')) != bool(smoke) for r in recs if 'smoke' in r):
        S['reasons'].append('smoke and non-smoke job files mixed')
    ok = []
    for r in recs:
        if r.get('failed'):
            S['failed'].append(job_tag(r))
        elif r.get('skipped'):
            S['skipped'].append(job_tag(r))
        else:
            ok.append(r)
    asr = S['assertions']
    for r in ok:
        for tn, C in r['checks'].items():
            for sd, c in C.items():
                k = f'{tn}|{sd}'; a = asr.setdefault(k, dict(k0_identical=True, fpr_identical=True, known_mismatch=0, lost_cells=0, net_fpr_identical=True,
                                                               jobs=0))
                a['k0_identical'] &= c['k0_identical']; a['fpr_identical'] &= c['fpr_identical']; a['net_fpr_identical'] &= c['net_fpr_identical']
                a['known_mismatch'] += c['known_mismatch_arm'] + c['known_mismatch_ctrl']; a['lost_cells'] += c['lost_cells_arm'] + c['lost_cells_ctrl']
                a['jobs'] += 1
    gb = {}
    for r in ok:
        for tn, T in r['tiers'].items():
            for arm, a in T.items():
                if arm.startswith(NETA):
                    continue
                g = gb.setdefault(arm, dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, jobs=0))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['cf_rel_max'] = max(g['cf_rel_max'], a['gbp']['cf_rel_max'])
                g['violations'] += a['gbp']['violations']; g['components'] += a['gbp']['components']; g['jobs'] += 1
    for g in gb.values():
        g['pass'] = bool(g['components'] > 0 and g['cf_max'] <= D8_67['cf_tol'] and g['cf_rel_max'] <= D8_67['cf_tol'] and g['violations'] == 0)
    S['gbp'] = gb
    for panel in sorted({r['panel'] for r in ok}):
        P = dict(seeds={})
        for seed in sorted({r['seed'] for r in ok if r['panel'] == panel}):
            X = sorted([r for r in ok if r['panel'] == panel and r['seed'] == seed], key=job_tag)
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={}, per_house={})
            for tn in R.EVAL_TIERS:
                if not X or not all(tn in r['tiers'] for r in X):
                    continue
                arms = sorted(a for a in X[0]['tiers'][tn] if all(a in r['tiers'][tn] for r in X))
                D['tiers'][tn] = {a: (_net_summ if a.startswith(NETA) else E._summ_arm)([r['tiers'][tn][a] for r in X]) for a in arms}
                D['per_house'][tn] = {f"r{r['rooms']}_i{r['index']}": {a: dict(furn=_house_furn(r['tiers'][tn][a]['R8']),
                                                                               bytes=(r['tiers'][tn][a].get('bytes') if not a.startswith(NETA) else
                                                                                      ((r['tiers'][tn][a]['bytes'] or {}).get('sum', 0) + (r['tiers'][tn][a]['bytes'] or {}).get('count', 0)) or None))
                                                                       for a in arms} for r in X}
                D.setdefault('paired', {})[tn] = {}
                for ts in D8_67['train_seeds']:
                    if f'{ARM}_{ts}' in arms and f'{CTRL}_{ts}' in arms:
                        ra = [x for r in X for x in r['tiers'][tn][f'{ARM}_{ts}']['R8']]; rc = [x for r in X for x in r['tiers'][tn][f'{CTRL}_{ts}']['R8']]
                        D['paired'][tn][ts] = _paired(ra, rc)
            P['seeds'][str(seed)] = D
        S['panels'][panel] = P
    return S


def make_cell(D, ts, C=D8_67):
    """one (panel, noise seed, train seed) cell from the aggregated D; None-safe for missing tiers (the caller records the reason)"""
    jt, ct, kt = C['judged_tier'], C['cam_tier'], C['clean_tier']
    a = D['tiers'][jt][f'{ARM}_{ts}']; c = D['tiers'][jt][f'{CTRL}_{ts}']
    ph = D['per_house'][jt]
    hd = {h: v[f'{ARM}_{ts}']['furn'] - v[f'{CTRL}_{ts}']['furn'] for h, v in ph.items()
          if v[f'{ARM}_{ts}']['furn'] is not None and v[f'{CTRL}_{ts}']['furn'] is not None}
    hr = [v[f'{ARM}_{ts}']['bytes'] / v[f'{CTRL}_{ts}']['bytes'] for v in ph.values()]
    worst = min(hd, key=hd.get) if hd else None
    cell = dict(arm_R8_furn=a['R8']['furn_f1'], ctrl_R8_furn=c['R8']['furn_f1'], d_furn=a['R8']['furn_f1'] - c['R8']['furn_f1'],
                worst_house=worst, worst_house_d=(hd[worst] if worst else None), n_houses=len(ph),
                arm_bytes_median=a['bytes_median'], ctrl_bytes_median=c['bytes_median'], bytes_ratio=a['bytes_median'] / c['bytes_median'],
                report_bytes_ratio_per_house_median=float(np.median(hr)), report_bytes_ratio_per_house_max=float(np.max(hr)),
                free_delta=a['R8']['free_recall'] - c['R8']['free_recall'], surface_delta=a['R8']['surface_f1'] - c['R8']['surface_f1'],
                fpr_delta=a['R8']['unknown_known_fpr'] - c['R8']['unknown_known_fpr'],
                d_furn_tol1=a['R8']['furn_f1_tol1'] - c['R8']['furn_f1_tol1'],
                K0_sep_arm=a['K0_sep']['furn_f1'], K0_sep_ctrl=c['K0_sep']['furn_f1'], w_arm=a['w'], w_ctrl=c['w'],
                belief_disagreement_rms=dict(arm=a['belief_disagreement_rms'], ctrl=c['belief_disagreement_rms']),
                quant_belief_diff=dict(arm=a['quant_belief_diff'], ctrl=c['quant_belief_diff']), self_map=a['self_map'], z=a['z'],
                strata=dict(arm=a['R8'].get('strata'), ctrl=c['R8'].get('strata')))
    cl = D['tiers'].get(kt, {}).get(f'{ARM}_{ts}'); cam_a = D['tiers'].get(ct, {}).get(f'{ARM}_{ts}'); cam_c = D['tiers'].get(ct, {}).get(f'{CTRL}_{ts}')
    cell['clean_arm_furn'] = cl['R8']['furn_f1'] if cl else None
    cell['cam_d_furn'] = (cam_a['R8']['furn_f1'] - cam_c['R8']['furn_f1']) if (cam_a and cam_c) else None
    cell['cam_arm_furn'] = cam_a['R8']['furn_f1'] if cam_a else None
    pr = D.get('paired', {}).get(jt, {}).get(ts)
    if pr:
        cell['paired'] = dict(pr, d_pooled=pr['pooled_arm'] - pr['pooled_ctrl'],
                              d_rowmean=(pr['rowmean_arm'] - pr['rowmean_ctrl']) if pr['rows'] else None)
    net = D['tiers'][jt].get(f'{NETA}_{ts}')
    if net:
        nb = net.get('bytes_median') or {}
        nbt = (nb.get('sum', 0) + nb.get('count', 0)) or None
        cell['net'] = dict(R8_furn=net['R8']['furn_f1'], arm_minus_net=a['R8']['furn_f1'] - net['R8']['furn_f1'],
                           ctrl_minus_net=c['R8']['furn_f1'] - net['R8']['furn_f1'], bytes=nbt,
                           arm_bytes_x_net=(a['bytes_median'] / nbt) if nbt else None, ctrl_bytes_x_net=(c['bytes_median'] / nbt) if nbt else None,
                           arm_furn_line_ok=bool(a['R8']['furn_f1'] >= net['R8']['furn_f1'] + C['net_furn_line'] - 1e-12),
                           arm_bytes_line_ok=(bool(a['bytes_median'] / nbt <= C['net_bytes_line'] + 1e-12) if nbt else None),
                           K0_sep=net['K0_sep']['furn_f1'], free=net['R8']['free_recall'])
        camn = D['tiers'].get(ct, {}).get(f'{NETA}_{ts}')
        if camn and cam_a:
            cell['net']['cam_arm_minus_net'] = cam_a['R8']['furn_f1'] - camn['R8']['furn_f1']
    return cell


def judge(cells, fresh, reasons, C=D8_67):
    """D8-67, hard-coded. cells / fresh: {'ts|ns': cell} for VAL / fresh cam_nf x train seeds 342101 / 342202 x noise seeds 3 / 4.
    LEVER iff in all 4 VAL cells: d furn >= -0.3 pp, worst single-house d >= -2.0 pp, bytes ratio (median per-house O7, arm / control) <= 0.60,
    free recall d >= -0.3 pp, surface F1 d >= -0.3 pp, clean-tier arm furniture >= .99 (the identity assertions -- known mask, K0, FPR, GaBP --
    are INCOMPLETE reasons, checked by the caller). Demotion only (只降不升): cam 4-cell mean d < -1.0 pp or fresh 4-cell mean d furn < -0.3 pp
    -> LEVER becomes AMBIGUOUS. NO iff any VAL cell d furn < -1.0 pp, or bytes ratio > 0.80, or a margin guard fails. Else AMBIGUOUS.
    INCOMPLETE: any reason (missing task / cell / house, failed job, assertion, sha mismatch)."""
    reasons = list(reasons); eps = 1e-12
    keys = [f'{ts}|{ns}' for ts in C['train_seeds'] for ns in C['noise_seeds']]
    for k in keys:
        for nm, st in (('val', cells), ('fresh', fresh)):
            if k not in st:
                reasons.append(f'{nm} cell {k} missing')
            elif st[k].get('n_houses') != C['houses_per_cell']:
                reasons.append(f"{nm} cell {k}: {st[k].get('n_houses')} houses != {C['houses_per_cell']}")
        if k in cells and (cells[k].get('clean_arm_furn') is None or cells[k].get('cam_d_furn') is None or cells[k].get('worst_house_d') is None):
            reasons.append(f'val cell {k}: clean / cam row or per-house rows missing')
    out = dict(criteria=C, cells=cells, fresh_cells=fresh, notes=[])
    if reasons:
        out.update(verdict='INCOMPLETE: ' + '; '.join(reasons), reasons=reasons)
        return out
    for k in keys:
        c = cells[k]
        g = dict(free=c['free_delta'] >= C['guard_free'] - eps, surface=c['surface_delta'] >= C['guard_surface'] - eps,
                 clean=c['clean_arm_furn'] >= C['clean_min'] - eps)
        c['guards'] = g; c['guards_pass'] = all(g.values())
        c['lever_ok'] = bool(c['d_furn'] >= C['lever_furn_margin'] - eps and c['worst_house_d'] >= C['lever_worst_house'] - eps
                             and c['bytes_ratio'] <= C['lever_bytes_ratio'] + eps and c['guards_pass'])
        c['no_hit'] = bool(c['d_furn'] < C['no_furn_margin'] - eps or c['bytes_ratio'] > C['no_bytes_ratio'] + eps or not c['guards_pass'])
    cam_mean = float(np.mean([cells[k]['cam_d_furn'] for k in keys])); fresh_mean = float(np.mean([fresh[k]['d_furn'] for k in keys]))
    val_mean = float(np.mean([cells[k]['d_furn'] for k in keys]))
    out.update(cam_mean_d=cam_mean, fresh_mean_d=fresh_mean, val_mean_d=val_mean, reasons=[])
    if any(cells[k]['no_hit'] for k in keys):
        v = 'NO'
    elif all(cells[k]['lever_ok'] for k in keys):
        dem = []
        if cam_mean < C['cam_demote_mean'] - eps:
            dem.append(f'cam 4-cell mean d {cam_mean * 100:+.3f} pp < {C["cam_demote_mean"] * 100:+.1f} pp')
        if fresh_mean < C['fresh_demote_mean'] - eps:
            dem.append(f'fresh 4-cell mean d {fresh_mean * 100:+.3f} pp < {C["fresh_demote_mean"] * 100:+.1f} pp')
        v = ('AMBIGUOUS (LEVER demoted: ' + '; '.join(dem) + ')') if dem else 'LEVER'
    else:
        v = 'AMBIGUOUS'
    out['verdict'] = v
    out['notes'].append(f'cam 4-cell mean d {cam_mean * 100:+.3f} pp; fresh 4-cell mean d {fresh_mean * 100:+.3f} pp; VAL mean d {val_mean * 100:+.3f} pp')
    rep = {}
    for k in keys:                                              # report only: does the paired pooled F1 cross the lines like the verdict reading?
        c = cells[k]; p = c.get('paired')
        if p:
            rep[k] = dict(d_pooled=p['d_pooled'], d_rowmean=p['d_rowmean'], d_verdict=c['d_furn'],
                          same_side_lever=bool((p['d_pooled'] >= C['lever_furn_margin'] - eps) == (c['d_furn'] >= C['lever_furn_margin'] - eps)),
                          same_side_no=bool((p['d_pooled'] < C['no_furn_margin'] - eps) == (c['d_furn'] < C['no_furn_margin'] - eps)),
                          same_sign=bool(np.sign(p['d_pooled']) == np.sign(c['d_furn'])))
    out['paired_report'] = rep
    return out


def integrity_reasons(S, crit, C=D8_67, seeds=None, smoke=False):
    """INCOMPLETE reasons from the aggregate: sha mismatches, failed jobs, assertions (known mask, K0, FPR, GaBP), missing tasks"""
    reasons = list(S['reasons'])
    seeds = seeds if seeds is not None else [int(s) for s in C['noise_seeds']]
    if S['failed']:
        reasons.append(f"failed jobs: {S['failed']}")
    for k, a in S['assertions'].items():
        if not a['k0_identical']:
            reasons.append(f'{k}: K0_sep rows differ between the arms (K0 carries no messages)')
        if not a['fpr_identical']:
            reasons.append(f'{k}: R8 unknown->known FPR differs between the arms')
        if a['known_mismatch']:
            reasons.append(f"{k}: known (Lam_b > 0) != k > 0 on {a['known_mismatch']} voxels")
    for arm, g in S['gbp'].items():
        if not g['pass']:
            reasons.append(f"GaBP check failed for {arm} (cf_max {g['cf_max']:.3e}, cf_rel {g['cf_rel_max']:.3e}, violations {g['violations']})")
    if not smoke:
        want = {job_tag(r) for r in export_records(seeds, [C['judged_panel'], C['report_panel']])}
        have = {job_tag(r) for r in _all_jobs(S)}
        miss = sorted(want - have)
        if miss:
            reasons.append(f'{len(miss)} tasks missing, e.g. {miss[:4]}')
        for ts in C['train_seeds']:
            for arm in (f'{ARM}_{ts}', f'{CTRL}_{ts}'):
                if arm not in S['gbp']:
                    reasons.append(f'no GaBP checks for {arm}')
    return reasons


def _all_jobs(S):
    out = []
    for panel, P in S['panels'].items():
        for seed, D in P['seeds'].items():
            for h in D['houses']:
                r, i = h[1:].split('_i')
                out.append(dict(panel=panel, rooms=int(r), index=int(i), seed=int(seed)))
    return out


def cells_for(S, seeds, C=D8_67):
    cells, fresh, reasons = {}, {}, []
    for panel, store in ((C['judged_panel'], cells), (C['report_panel'], fresh)):
        P = S['panels'].get(panel, {}).get('seeds', {})
        for ns in seeds:
            D = P.get(str(ns))
            if D is None:
                continue
            for ts in C['train_seeds']:
                try:
                    store[f'{ts}|{ns}'] = make_cell(D, ts, C)
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    reasons.append(f'{panel} cell {ts}|{ns}: {type(e).__name__} {e}')
    return cells, fresh, reasons


def official_continuity(S, path=E3L3_SUMMARY, C=D8_67):
    """report only: our learned-w control arm on noise seeds 1, 2 vs E3-L3's official summary.json (same code, same exports)"""
    if not os.path.exists(path):
        return None
    O = _jload(path); rows = {}
    for panel in (C['judged_panel'], C['report_panel']):
        for ns in C['continuity_seeds']:
            for tn in R.EVAL_TIERS:
                for ts in C['train_seeds']:
                    try:
                        mine = S['panels'][panel]['seeds'][ns]['tiers'][tn][f'{CTRL}_{ts}']; off = O['panels'][panel]['seeds'][ns]['tiers'][tn][f'E3L3_{ts}']
                    except (KeyError, TypeError):
                        continue
                    d = {}
                    for k in ('furn_f1', 'furn_f1_pooled', 'free_recall', 'surface_f1', 'unknown_known_fpr', 'furn_f1_tol1'):
                        d[f'R8.{k}'] = (mine['R8'][k], off['R8'][k])
                    d['K0_sep.furn_f1'] = (mine['K0_sep']['furn_f1'], off['K0_sep']['furn_f1']); d['bytes_median'] = (mine['bytes_median'], off['bytes_median'])
                    rows[f'{panel}|{ns}|{tn}|{ts}'] = dict(equal=all(x == y for x, y in d.values()),
                                                         max_abs_diff=max(abs(float(x) - float(y)) for x, y in d.values() if x is not None and y is not None), values=d)
    return dict(path=os.path.relpath(path, ROOT), sha256=file_sha256(path), rows=rows, n=len(rows), all_equal=bool(rows) and all(r['equal'] for r in rows.values()),
                max_abs_diff=max([r['max_abs_diff'] for r in rows.values()] + [0.0]))


def e5_net_check(out_dir, jobs_dir=E5_JOBS, seeds=None):
    """report only: our NET rows vs G4-E5's Qinf rows (same exports, same E3-R checkpoints): per-house R8 furniture and cam_nf (S, k) bytes"""
    diffs = dict(furn=0.0, bytes=0.0, n=0, missing=0)
    for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json'))):
        r = _jload(p)
        if r.get('failed') or r.get('skipped') or (seeds and int(r['seed']) not in seeds):
            continue
        q = os.path.join(jobs_dir, os.path.basename(p))
        if not os.path.exists(q):
            diffs['missing'] += 1; continue
        Q = _jload(q)
        for tn, T in r['tiers'].items():
            for sd in CKPTS:
                try:
                    qi = Q['tiers'][tn][sd]['Qinf']
                except (KeyError, TypeError):
                    diffs['missing'] += 1; continue
                nb = T[f'{NETA}_{sd}']
                diffs['furn'] = max(diffs['furn'], abs(_house_furn(nb['R8']) - _house_furn(qi['R8_net'])))
                if nb.get('bytes') and qi.get('bytes'):
                    diffs['bytes'] = max(diffs['bytes'], abs(nb['bytes']['sum'] + nb['bytes']['count'] - qi['bytes']['Sk']))
                diffs['n'] += 1
    return diffs


def aggregate_cmd(args):
    crit, csha = load_criteria(args.criteria, args.smoke)
    S = aggregate(args.out, crit, csha, smoke=args.smoke)
    S['criteria'] = dict(path=os.path.abspath(args.criteria), sha256=csha, w_star=crit['w_star'], branch=crit['branch'], note=crit.get('note'))
    jseeds = [int(s) for s in D8_67['noise_seeds']]
    have_j = any(str(s) in P['seeds'] for P in S['panels'].values() for s in jseeds)
    if have_j or not args.smoke:
        cells, fresh, r2 = cells_for(S, D8_67['noise_seeds'])
        S['decision'] = judge(cells, fresh, integrity_reasons(S, crit, smoke=args.smoke) + r2)
    else:
        S['decision'] = dict(verdict='INCOMPLETE: smoke (no judged noise seeds)')
    if args.smoke:
        S['decision']['verdict'] = 'SMOKE ' + S['decision']['verdict']
    cc, cf, r3 = cells_for(S, D8_67['continuity_seeds'])
    S['continuity'] = dict(cells=cc, fresh_cells=cf, reasons=r3, official_e3l3=official_continuity(S))
    S['e5_net_check'] = e5_net_check(args.out)
    S['script_sha256_now'] = file_sha256(os.path.abspath(__file__)); S['utc'] = utc()
    _jdump(S, os.path.join(args.out, 'summary.json'))
    dec = S['decision']
    print(f"G4-E7 verdict: {dec['verdict']}  (w* = {crit['w_star']}, criteria {csha[:12]})", flush=True)
    for nm, st in (('VAL', dec.get('cells', {})), ('fresh', dec.get('fresh_cells', {})), ('cont VAL', cc), ('cont fresh', cf)):
        for k, c in sorted(st.items()):
            print(f"  {nm} {k}: R8 furn {c['arm_R8_furn']:.4f} vs learned {c['ctrl_R8_furn']:.4f} (d {c['d_furn'] * 100:+.3f} pp, worst {c['worst_house']} "
                  f"{(c['worst_house_d'] or 0) * 100:+.2f} pp); bytes x learned {c['bytes_ratio']:.3f}; free {c['free_delta'] * 100:+.3f} surf "
                  f"{c['surface_delta'] * 100:+.3f} pp; clean {c['clean_arm_furn']}; cam d {(c['cam_d_furn'] or 0) * 100:+.3f} pp"
                  + (f"; NET {c['net']['R8_furn']:.4f} (arm-NET {c['net']['arm_minus_net'] * 100:+.3f} pp, arm bytes x NET {c['net']['arm_bytes_x_net']})" if c.get('net') else '')
                  + (f"; paired pooled d {c['paired']['d_pooled'] * 100:+.3f} pp" if c.get('paired') else ''), flush=True)
    print('  notes:', dec.get('notes'), flush=True)
    print('  gbp:', json.dumps(S['gbp']), flush=True)
    print('  official continuity:', json.dumps({k: S['continuity']['official_e3l3'][k] for k in ('n', 'all_equal', 'max_abs_diff')}) if S['continuity']['official_e3l3'] else None, flush=True)
    print('  e5 NET check:', json.dumps(S['e5_net_check']), flush=True)
    return S


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('wselect', 'eval', 'aggregate'):
        p = sub.add_parser(name)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=MAX_WORKERS); p.add_argument('--threads', type=int, default=1)
        p.add_argument('--only', default='')
        if name == 'wselect':
            p.add_argument('--out', default='')
        if name == 'eval':
            p.add_argument('--criteria', required=True); p.add_argument('--out', required=True); p.add_argument('--seeds', required=True)
            p.add_argument('--panels', default='val,fresh'); p.add_argument('--confirm-criteria-sha', default=''); p.add_argument('--chunk', type=int, default=2)
        if name == 'aggregate':
            p.add_argument('--criteria', required=True); p.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.cmd == 'wselect':
        return wselect(args, json_dir, index_file)
    if args.cmd == 'eval':
        return eval_cmd(args, json_dir, index_file)
    if args.cmd == 'aggregate':
        return aggregate_cmd(args)


if __name__ == '__main__':
    main()
