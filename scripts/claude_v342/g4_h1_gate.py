"""g4_h1_gate.py (Claude, 2026-10-28): 批次 G4-H1G "只用线上可得量 (Lam_b, x) 的读出门控，处理学习隐码 E3-L3 的幻觉"（纯 CPU，0 个租约）。
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-75（逐条照做）；背景 D8-53 / D8-55（H1 判据）、D8-69 / D8-71（G4-H1L：E3-L3 floating 有害）、
D8-67 / D8-74（G4-E7：w* = 4）、D8-63 / D8-65（E3-L3）。
只 import 冻结代码，从不修改：g4_h1_e3l3.py（H1L，dc0f69d7：面板、闸门、计分层、HL.aggregate / verdict = D8-69 的 CRIT）、g4_e3l3_w.py（G4-E7，
010785f9：HOHouse / ho_records / export_records / _strip / _house_furn / job_tag）、经它们 g4_e3l3（c7523cc8：load_model / source / eval_arm /
_summ_arm）、g4_e3l2、g4_e3l（encode_maps / gabp / local_forest / decode_np / gbp_checks）、g4_e3r、g4_noise_panels、g4_ood_panel、
mapping_3d_storey_data_v420ood（sha 都由 HL.check_frozen 与本文件 SELF_FROZEN 核对）。不改任何已有文件，不生成任何世界，不读 TEST。

门控（D8-75，写死）：每个 (检查点, w) 一张表；体素按 (Lam_b, x) 落格（Lam_b 线上整数，x 按 0.05 分箱、[-1, 1]）；在 TRAIN 19 个留出房 x 抽样
200-202（原生成器、cam_nf，与选阈值同一批）上统计每格"真值空闲"比例 q；q >= 0.995 且格内体素 >= 200 的格为强空闲格；读出时解码器判占据、
但落在强空闲格的体素改判为空闲；已知区不变；0.995 与 200 不调。
臂：E3-L3 两个检查点 x {学到的 w, w = 4} x {不门控, 门控}。面板：H1（D8-53 / D8-69，复用 g4/h1_export 的 192 个 VAL npz，sha 核对）；
精度保护（VAL / fresh cam_nf，噪声种子 3、4，G4-E5 导出，与 G4-E7 同一批地图）。

操作化（D8-75 没写死、在这里写定；写于看任何 H1 VAL 或种子 3、4 读数之前）
  G1 落格：在逐跳步长 1 取整的线路上，接收方信念的 h_b 与 Lam_b 是精确整数（断言），|h_b| <= Lam_b（断言）；x = h_b / Lam_b；
     xbin = floor((x + 1) / 0.05)，用整数精确计算 floor(20 (h_b + Lam_b) / Lam_b)，x = 1 归入最后一格，共 NX = 40 格；Lam_b 取 1..LMAX = 64
     （更大断言失败）。Lam_b = 0（未知）不落格、不门控。
  G2 拟合口径（TRAIN）：g4/e5_export/train_holdout（G4-E7 第 0 步同一批 19 房 x 抽样 200-202，records 逐个核对 sha；房表与抽样必须等于
     E3-L3 官方 threshold.json 的 houses / draws）。读法与 wselect / H1L / eval_arm 逐语句相同：L.encode_maps（默认分块）-> E3.source(.., 'round')
     -> 每分量 L.local_forest + L.gabp(w, 'round') -> L.decode_np(chunk 2)。每台机器自己的信念一行（权重 1，同 R8 计分与阈值选取）。
     格内体素数 n = 该机 Lam_b > 0 且分量 R8 目标已知（t >= 0）的体素数；q = 其中目标空闲（t = 0）的比例。目标未知的体素（n_notruth）不进 q，
     只报告。强空闲格：n >= 200 且 1000 * n_free >= 995 * n（整数比较）。
  G3 阈值：学到的 w 用 E3-L3 的 threshold.json（完整 sha 钉住，= HL.E3L3_CKPTS）；w = 4 用 g4/e7/criteria_w.json（完整 sha 钉住）的
     thresholds[ts]['w4']，断言其 w 字段 = 4、w_star = 4、检查点 sha 与本批钉的相同（审查 S1）。表的拟合不依赖阈值；阈值只用于只报告的
     "TRAIN 上会被改判的体素"与评估时的读出。
  G4 门控只作用于 R8 读出（每台机器解码自己的 GaBP 信念）：p_g = p，p_g[(p == 1) & 强空闲格(Lam_b, xbin)] = 0。已知区（p >= 0）、线路与
     字节、K0 / K0_sep（K0 不在 D8-75 判据里，保持不门控）都不变。
  G5 H1：计分循环是 HL.evaluate_file 的逐行复写（对照 B4 / B6 段逐字相同；E3-L3 段在同一次编码、GaBP、解码之后，对不门控与门控两套预测
     分别算同样的层），记录格式与 H1L 相同（每个 (w, 门控) 一个子目录），汇总用冻结的 HL.aggregate / HL.verdict（D8-69 的 CRIT、ctrl
     自检 [+4.3, +8.6] pp）。闸门：H1 记录逐值相等（HL.compare_controls）；正式运行前串行核对 192 个 npz 的 sha（审查 S3）。
     只报告的门控统计：每格改判数、改判中目标空闲 / 占据 / 未知的数，按最近物体家族（同 H1 的 near_fam）、高度带（G.H_BANDS + 楼板上第 1 层）、
     新物体区拆分，改判里的家具 TP 损失与 FP 减少。
  G6 guard：每个 (档, 检查点, w) 调冻结的 E3.eval_arm（与 G4-E7 同参数：chunk 2；学到的 w = 官方 threshold.json；w = 4 = criteria_w 阈值）。
     eval_arm 运行期间临时把 L.decode_np 包一层（只记录参数与输出、不改任何值，退出时恢复；只在本进程），取到每个分量的 (h_b, Lam_b, logit)，
     按 G4 门控后用 R.scored 计分（行权重 1，同 eval_arm）；整件漏报对门控后的预测重算。判决档 cam_nf 逐台核对"用记录的 logit 重算的不门控行
     == eval_arm 的行"（JSON 往返）。与 G4-E7 相同：K0 / K0_sep 去掉 strata，非 cam_nf 档的 R8 也去掉 strata。档默认 cam_nf / cam / clean
     （cam、clean 只报告，D8-75"门控对 cam 档与 clean 档的影响"）。
  G7 闸门（逐值相等，JSON 往返后比较）：
     H1：学到的 w 不门控那一臂的 HL.aggregate 汇总的 variants 与 verdict == g4/h1l/summary.json（完整 sha 钉住）；
     guard：w = 4 不门控 == g4/e7/summary.json 的 E3L3W_{ts}，学到的 w 不门控 == 其 E3L3_{ts}（每个 (面板, 种子, 档, 检查点) 的
     E3._summ_arm 全部字段；e7 summary 完整 sha 钉住）。代码路径对两个 w 相同，所以任一闸门不等 -> 两个 w 的结论都判 INCOMPLETE。
  G8 判据（D8-75，写死在 D8_75 与 decide()）：对 w = 4 与学到的 w 各判一次。
     "门控修好了幻觉"：(1) floating 门控后的 H1 判决不是 HARMFUL（即 4 格翻转精确率 >= .50、新物体区 Δ >= -1 pp、幻影块 <= B6 + 2）；
     (2) novel、scale 门控后判决的等级（SAFE > AMBIGUOUS > HARMFUL）不低于不门控；(3) 门控臂的 ctrl 自检通过。
     精度保护（VAL cam_nf 种子 3、4 x 2 检查点 = 4 格）：门控家具 - 不门控 >= -0.3 pp；空闲召回门控 >= 不门控（容差 1e-12）；FPR 逐位相同
     （每行与汇总）；字节逐位相同（每房与汇总中位）。fresh 4 格均值 Δ 家具 >= -0.3 pp，不满足只把 LEVER 降为 AMBIGUOUS。
     结论：INCOMPLETE（缺任务、失败、闸门不等、sha 不符、版本混用、断言）> NO（精度保护不过）> LEVER（1-3、精度保护、fresh 全过）> AMBIGUOUS。
  G9 输出：h1g/fit（gate_tables.json + jobs）；h1g/h1/{w_learned,w4}/{ungated,gated}；h1g/guard/{w_learned,w4}/{ungated,gated}；
     h1g/aggregate。冒烟在 h1g/smoke/ 下。写入路径断言在 h1g 之下（绝不写 g4/h1l、g4/e7 等）。任务记录不覆盖：已存在且完整、同脚本 sha、
     同表 sha → 跳过；否则拒绝运行。gate_tables.json 不覆盖。各子目录的 summary.json / report.txt 是本批的派生文件，重跑 aggregate 时重写。
  G10 门槛：h1 / guard 的正式运行必须给 --tables-sha 等于 gate_tables.json 的 sha，且表不是冒烟表；h1 正式 = H1 的 96 个 VAL 任务 x 种子 1、2；
     guard 正式 = VAL / fresh x 种子 3、4 的全部导出任务。冒烟：fit 用 --only 的 TRAIN 房；h1 只用 H1 冒烟的 TRAIN 记录（g4/h1_smoke2/train），
     闸门对 g4/h1l_smoke/train/summary.json；guard 只用 val_r1_i0_s1（R-I 看过的 VAL 种子 1），学到的 w 不门控逐值对 G4-E7 冒烟记录。

用法（总是 python -B；CUDA_VISIBLE_DEVICES= ；本机 <= 6 个进程）：
  fit       [--workers 6] [--smoke --only TRAIN_r1_i19,..]                                  # TRAIN only -> h1g/fit/gate_tables.json
  h1        --tables T --tables-sha SHA --formal [--ws learned,w4] [--workers N]             # H1 VAL 面板（正式）
  h1        --tables T --smoke [--workers N]                                                 # TRAIN 冒烟
  guard     --tables T --tables-sha SHA --formal [--ws learned,w4] [--tiers cam_nf,cam,clean] [--workers N]
  guard     --tables T --smoke
  aggregate --tables T [--smoke]
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, copy, glob, hashlib, json, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_h1_e3l3 as HL  # noqa: E402   (frozen H1L; imports g4_ood_panel, g4_e3l3 -> g4_e3l2 -> g4_e3l -> g4_e3r -> g4_noise_panels)
import g4_e3l3_w as W  # noqa: E402     (frozen G4-E7)

E3 = HL.E3
L = HL.L
R = HL.R
G = HL.G
OP = HL.OP
L2 = E3.L2
assert W.E is E3 and W.L is L and W.R is R and W.G is G, 'g4_e3l3_w and g4_h1_e3l3 must share the frozen modules'
torch = HL.torch
UNK = R.UNK

ROOT = G.ROOT
AN = HL.AN
G4 = os.path.join(AN, 'g4')
OUT_ROOT = os.path.join(G4, 'h1g')
SMOKE_ROOT = os.path.join(OUT_ROOT, 'smoke')
SELF_FROZEN = dict(g4_h1_e3l3='dc0f69d7c8ea2b2a8513fa2c04095664ebd7eaf0f1c0637d497943b57537da57',
                   g4_e3l3_w='010785f91fb4c414727f7466a6973ff97aaff4ffa964faee4e4f525c670af0f5')
CRITERIA_W = os.path.join(G4, 'e7', 'criteria_w.json')
CRITERIA_W_SHA256 = '320317aaefad21d85d613596a5156da5ad596b764d13a63bcaedaadc17b83060'
E7_SUMMARY = os.path.join(G4, 'e7', 'summary.json')
E7_SUMMARY_SHA256 = 'c45eaca7e456fdbc7cbe4bdb6a0ed6f832b0f724d3a07b438b79d7fec8565cd9'
H1L_SUMMARY = os.path.join(G4, 'h1l', 'summary.json')
H1L_SUMMARY_SHA256 = '2a6ae99a38d7ff8225c6ca2ea231770dc5b1cfb33dff731dc03b8c6e0a992c86'
H1L_SMOKE_SUMMARY = os.path.join(G4, 'h1l_smoke', 'train', 'summary.json')
H1L_SMOKE_SUMMARY_SHA256 = 'dad4a22b0dd82906763e943b8db87e4064d241b12880440752fc30ab66c8c9f0'
H1_SMOKE_JOBS = os.path.join(G4, 'h1_smoke2', 'train', 'jobs')
E7_SMOKE_JOB = os.path.join(G4, 'e7', 'smoke', 'jobs', 'val_r1_i0_s1.json')
E7_SMOKE_JOB_SHA256 = 'a398cd796347e18b27029686f3d40536e8c424c3274f1e3b4caf4ae0e443d2ee'
SMOKE_GUARD_TAGS = ('val_r1_i0_s1',)          # the VAL noise seed 1 task R-I already looked at
SEEDS = HL.SEEDS
PAIR = HL.PAIR
WLABS = ('learned', 'w4')
WDIR = {'learned': 'w_learned', 'w4': 'w4'}
WREC = {'learned': 'learned', 'w4': '4'}      # the H1L record convention for 'w'
W_STAR = 4.0
GATES = ('ungated', 'gated')
NX = 40                                       # x bins of width 0.05 on [-1, 1]
XW = 0.05
LMAX = 64
Q_NUM, Q_DEN = 995, 1000                      # q >= 0.995 (integer comparison)
N_MIN = 200
COLS = ('n_truth', 'n_free', 'n_notruth', 'n_pocc', 'n_pocc_free', 'n_pocc_region_tr', 'n_pocc_region_ntr', 'n_pocc_notruth')
NCOL = len(COLS)
NB = OP.NB
H_NAMES = OP.H_NAMES
JUDGED_TIER = R.JUDGED_TIER
GUARD_SEEDS = (3, 4)
GUARD_PANELS = ('val', 'fresh')
H1_SEEDS = HL.NOISE_SEEDS
MAX_WORKERS = 6
FORMAT = 1
RANK = {'HARMFUL': 0, 'AMBIGUOUS': 1, 'SAFE': 2}
D8_75 = dict(prereg='D8-75', batch='G4-H1G', x_bin_width=XW, n_x_bins=NX, lam_max=LMAX, q_min=Q_NUM / Q_DEN, n_min=N_MIN,
             guard_furn_margin=-0.003,            # gated R8 furniture - ungated >= -0.3 pp (4 VAL cam_nf cells, seeds 3, 4)
             guard_free='gated free recall >= ungated (1e-12)', guard_fpr='bit-identical (every row and the summary)',
             guard_bytes='bit-identical (every house and the summary median)',
             fresh_demote_mean=-0.003,            # fresh 4-cell mean d furniture < -0.3 pp -> LEVER demoted to AMBIGUOUS
             fix=['floating gated H1 verdict != HARMFUL', 'novel / scale gated verdict rank >= ungated (SAFE > AMBIGUOUS > HARMFUL)',
                  'gated ctrl self-check in [+4.3, +8.6] pp (HL.CRIT)'],
             order='INCOMPLETE > NO (precision guard fails) > LEVER (fix 1-3 + guard + fresh) > AMBIGUOUS',
             h1_criteria=HL.CRIT)


# ================================================================ small helpers
def sha(path):
    return G.file_sha256(path)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


_jdump = HL._jdump
_jr = HL._jr


def utc():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def safe_out(path):
    """G9: every write of this batch lives under g4/h1g"""
    p = _norm(path); root = _norm(OUT_ROOT)
    if not (p == root or p.startswith(root + os.sep)):
        raise SystemExit(f'{path}: outputs of G4-H1G must be under {OUT_ROOT} (never g4/h1l, g4/e7, ...)')
    return path


def check_self_frozen():
    for nm, mod in (('g4_h1_e3l3', HL), ('g4_e3l3_w', W)):
        s = sha(mod.__file__)
        if s != SELF_FROZEN[nm]:
            raise RuntimeError(f'{nm}.py sha {s[:12]} != frozen {SELF_FROZEN[nm][:12]}')
    HL.check_frozen()
    W.check_frozen()


def my_sha():
    return sha(os.path.abspath(__file__))


# ================================================================ the gate (G1 / G4)
def cell_index(hb, lb):
    """one agent's belief (h_b [X, Y, Z], Lam_b [X, Y, Z]; exact integers on the per-hop rounded wire) -> flat cell index Lam_b * NX + xbin
    (int32; -1 where Lam_b == 0). xbin = floor((x + 1) / 0.05) with x = h_b / Lam_b, computed exactly as floor(20 (h_b + Lam_b) / Lam_b),
    x = 1 -> the last bin"""
    lam = np.asarray(lb); h = np.asarray(hb)
    li = np.rint(lam).astype(np.int64); hi = np.rint(h).astype(np.int64)
    if not (np.array_equal(li, lam) and np.array_equal(hi, h)):
        raise RuntimeError('belief (h_b, Lam_b) is not integer (G1 needs the per-hop rounded wire)')
    if li.min() < 0 or li.max() > LMAX:
        raise RuntimeError(f'Lam_b outside [0, {LMAX}]: {li.min()}..{li.max()}')
    pos = li > 0
    if np.any(np.abs(hi[pos]) > li[pos]) or np.any(hi[~pos] != 0):
        raise RuntimeError('|h_b| > Lam_b, or h_b != 0 where Lam_b == 0')
    xb = np.zeros(li.shape, np.int64)
    lp = li[pos]
    xb[pos] = np.minimum(((NX // 2) * (hi[pos] + lp)) // lp, NX - 1)
    return np.where(pos, li * NX + xb, -1).astype(np.int32)


def fit_strong(counts, q_num=Q_NUM, q_den=Q_DEN, n_min=N_MIN):
    """counts [LMAX + 1, NX, >= 2] (col 0 = n with truth, col 1 = truly free) -> bool [LMAX + 1, NX]: n >= n_min and free / n >= q (exact)"""
    c = np.asarray(counts, np.int64)
    n, f = c[..., 0], c[..., 1]
    s = (n >= n_min) & (q_den * f >= q_num * n)
    s[0] = False
    return s


def apply_gate(p, idx, lut):
    """p int8 {-1, 0, 1} -> (gated copy, changed mask): only p == 1 voxels whose (Lam_b, x) cell is strong become 0"""
    m = (p == 1) & (idx >= 0)
    ch = np.zeros(p.shape, bool)
    ch[m] = lut[idx[m]]
    q = p.copy(); q[ch] = 0
    return q, ch


def lut_of(strong):
    return np.asarray(strong, bool).reshape(-1)


def load_tables(path, expect_sha=None, allow_smoke=True):
    """gate_tables.json -> (tables dict, sha, {ts: {wl: flat bool LUT}})"""
    s = sha(path)
    if expect_sha is not None and s != expect_sha:
        raise SystemExit(f'{path}: sha {s} != --tables-sha {expect_sha}')
    T = _jload(path)
    assert T.get('batch') == 'G4-H1G' and T.get('stage') == 'fit', 'not a G4-H1G gate table file'
    if not allow_smoke and T.get('smoke'):
        raise SystemExit('a smoke gate table cannot drive a formal run')
    assert T['rule']['n_x_bins'] == NX and T['rule']['lam_max'] == LMAX and T['rule']['q_min'] == Q_NUM / Q_DEN and T['rule']['n_min'] == N_MIN
    luts = {}
    for ts in SEEDS:
        luts[ts] = {}
        for wl in WLABS:
            st = np.zeros((LMAX + 1, NX), bool)
            for lam, xb in T['tables'][ts][wl]['strong']:
                st[int(lam), int(xb)] = True
            luts[ts][wl] = lut_of(st)
    return T, s, luts


# ================================================================ models and thresholds (G3)
def load_thresholds():
    """{ts: {'learned': official threshold.json, 'w4': criteria_w thresholds[ts]['w4']}} (sha pinned; checkpoints cross-checked)"""
    if sha(CRITERIA_W) != CRITERIA_W_SHA256:
        raise RuntimeError('g4/e7/criteria_w.json sha != pinned')
    crit = _jload(CRITERIA_W)
    if crit.get('w_star') != W_STAR or crit.get('w_star_label') != 'w4' or crit.get('smoke'):
        raise RuntimeError('criteria_w.json: w* != 4 or smoke')
    out = {}
    for ts in SEEDS:
        dn, ck_sha, thr_sha = HL.E3L3_CKPTS[ts]
        tp = os.path.join(HL.INP, dn, 'threshold.json')
        if sha(tp) != thr_sha:
            raise RuntimeError(f'E3-L3 {ts} threshold.json sha != pinned')
        cs = crit['checkpoints'][ts]['sha256']
        if cs['e3l3_ckpt'] != ck_sha or cs['e3l3_thr'] != thr_sha:
            raise RuntimeError(f'criteria_w.json checkpoints {ts} != the pinned E3-L3 checkpoint / threshold')
        t4 = crit['thresholds'][ts]['w4']
        if float(t4['w']) != W_STAR:
            raise RuntimeError('criteria_w.json thresholds w4 carry another w')
        out[ts] = dict(learned=_jload(tp), w4=t4)
    return out


def arm_cfg(M, thr, ts, wl):
    m = M['e3'][ts]
    if wl == 'learned':
        return dict(model=m['model'], tb=m['tb'], w=m['w'], jo=int(thr[ts]['learned']['R8']['j_occ']), w_label='learned', thr=thr[ts]['learned'])
    return dict(model=m['model'], tb=m['tb'], w=W_STAR, jo=int(thr[ts]['w4']['R8']['j_occ']), w_label=f'finite:{W_STAR:g}', thr=thr[ts]['w4'])


_THR = None


def thresholds():
    global _THR
    if _THR is None:
        _THR = load_thresholds()
    return _THR


def _init_worker(threads):
    HL._worker_init(threads)


def _pool_iter(workers, threads, fn, jobs, maxtasks=2):
    if workers <= 1:
        _init_worker(threads)
        for k, j in enumerate(jobs, 1):
            yield k, fn(j)
        return
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker, initargs=(threads,), maxtasksperchild=maxtasks) as pool:
        for k, rec in enumerate(pool.imap_unordered(fn, jobs), 1):
            yield k, rec


# ================================================================ fit (TRAIN hold-out; G2)
def _fit_task(job):
    hrecs, opts = job
    ts = opts['ts']; rec = dict(stage='fit', batch='G4-H1G', seed=ts, house=None, script_sha256=opts['script_sha256'])
    t0 = time.time()
    try:
        if my_sha() != opts['script_sha256']:
            raise RuntimeError('g4_h1_gate.py changed since the run started')
        check_self_frozen()
        M = HL.load_models(opts['threads']); thr = thresholds()
        Hd = W.HOHouse(opts['export'], hrecs); rec['house'] = Hd.name; rec['draws'] = Hd.draws[1:]
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]
        m0 = M['e3'][ts]; cfg = {wl: arm_cfg(M, thr, ts, wl) for wl in WLABS}
        nflat = (LMAX + 1) * NX
        cnt = {wl: np.zeros((nflat, NCOL), np.int64) for wl in WLABS}
        tot = {wl: np.zeros(5, np.int64) for wl in WLABS}                  # pooled furniture tp / fp / fn, free hit, free n
        mism = {wl: 0 for wl in WLABS}; lmx = {wl: 0 for wl in WLABS}; agents = 0
        with torch.no_grad():
            for d in range(1, len(Hd.draws)):
                maps = Hd.maps[d]
                zhat, lam = L.encode_maps(m0['model'], list(maps), m0['tb'], 'cpu')
                hp, lq, _ = E3.source(zhat, lam, 'round')
                for wl in WLABS:
                    c = cfg[wl]
                    for ci, cm in enumerate(Hd.comps):
                        Fl = Hd.local(cm)
                        hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, c['w'], 'round')
                        kc = (maps[cm] != UNK).sum(0); lbn = lb.numpy(); hbn = hb.numpy(); T = R8T[ci]
                        for i, lg in enumerate(L.decode_np(m0['model'], hb, lb, HL.DEC_CHUNK)):
                            pk = lbn[i] > 0; mism[wl] += int((pk != (kc > 0)).sum())
                            idx = cell_index(hbn[i, 0], lbn[i]); lmx[wl] = max(lmx[wl], int(lbn[i].max()))
                            p = R.net_readout(lg, kc, c['jo']); po = (p == 1) & pk
                            masks = (pk & T.known, pk & T.free, pk & ~T.known, po & T.known, po & T.free,
                                     po & T.known & Hh.region & T.tr, po & T.known & Hh.region & ~T.tr, po & ~T.known)
                            for j, mk in enumerate(masks):
                                cnt[wl][:, j] += np.bincount(idx[mk], minlength=nflat)
                            pr = po & T.known & Hh.region
                            tot[wl] += [int((pr & T.tr).sum()), int((pr & ~T.tr).sum()), int((T.tr & ~pr).sum()), int(((p == 0) & T.free).sum()),
                                        int(T.free.sum())]
                            if wl == WLABS[0]:
                                agents += 1
        rec.update(counts={wl: {str(l): cnt[wl].reshape(LMAX + 1, NX, NCOL)[l].tolist() for l in range(LMAX + 1)
                                if cnt[wl].reshape(LMAX + 1, NX, NCOL)[l].any()} for wl in WLABS},
                   totals={wl: tot[wl].tolist() for wl in WLABS}, known_mismatch=mism, lam_b_max=lmx, agent_rows=agents,
                   w={wl: cfg[wl]['w'] for wl in WLABS}, j_occ={wl: cfg[wl]['jo'] for wl in WLABS}, seconds=time.time() - t0, peak_mb=G._peak_mb())
    except Exception as e:  # noqa: BLE001
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-3000:]
    return rec


def _dense(cdict):
    a = np.zeros((LMAX + 1, NX, NCOL), np.int64)
    for l, v in cdict.items():
        a[int(l)] = np.asarray(v, np.int64)
    return a


def _f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def table_stats(C, tot, strong):
    """report only, TRAIN: coverage and what the gate would change in-sample"""
    n, nf, nn, po, pof, ptr, pntr, pon = (C[..., j] for j in range(NCOL))
    s = strong
    g_known = int(po[s].sum()); g_all = int(po[s].sum() + pon[s].sum())
    tp, fp, fn, fh, fnn = (int(x) for x in tot)
    tp2, fp2, fn2 = tp - int(ptr[s].sum()), fp - int(pntr[s].sum()), fn + int(ptr[s].sum())
    by_lam = {}
    for lam in range(1, LMAX + 1):
        xs = np.nonzero(s[lam])[0]
        if len(xs):
            by_lam[str(lam)] = dict(xbins=xs.tolist(), x_lo=float(-1 + XW * xs.min()), x_hi=float(min(1.0, -1 + XW * (xs.max() + 1))),
                                    voxels=int(n[lam][xs].sum()))
    return dict(n_strong=int(s.sum()), n_cells_nonempty=int((n > 0).sum()), n_cells_ge_nmin=int((n >= N_MIN).sum()),
                known_truth_voxels=int(n.sum()), coverage_known_truth=float(n[s].sum() / max(n.sum(), 1)),
                coverage_known_all=float((n[s].sum() + nn[s].sum()) / max(n.sum() + nn.sum(), 1)),
                strong_true_free_share=float(nf[s].sum() / max(n[s].sum(), 1)),
                pred_occ_voxels=int(po.sum() + pon.sum()), gated_voxels=g_all, gated_share_of_pred_occ=float(g_all / max(int(po.sum() + pon.sum()), 1)),
                gated_true_free_share=(float(pof[s].sum() / g_known) if g_known else None),
                gated_region_true_furniture=int(ptr[s].sum()), gated_region_not_furniture=int(pntr[s].sum()),
                train_pooled_furn_f1=dict(ungated=_f1(tp, fp, fn), gated=_f1(tp2, fp2, fn2)),
                train_pooled_free_recall=dict(ungated=fh / max(fnn, 1), gated=(fh + int(pof[s].sum())) / max(fnn, 1)),
                strong_by_lam=by_lam)


def fit_cmd(args):
    check_self_frozen()
    me = my_sha(); thr = thresholds()
    assert args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} processes'
    assert args.smoke or not args.only, '--only is for the smoke'
    root = SMOKE_ROOT if args.smoke else OUT_ROOT
    out_dir = safe_out(os.path.join(root, 'fit')); out = os.path.join(out_dir, 'gate_tables.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (never overwrite)')
    jd = os.path.join(out_dir, 'jobs')
    if os.path.isdir(jd) and os.listdir(jd):
        raise SystemExit(f'{jd} is not empty (never overwrite)')
    only = [h for h in args.only.split(',') if h] if args.only else None
    houses = W.ho_records(W.TRAIN_HO, only)
    names = sorted(houses)
    off = {ts: thr[ts]['learned'] for ts in SEEDS}
    if not args.smoke:
        for ts in SEEDS:
            assert names == sorted(off[ts]['houses']), f'hold-out houses != the official E3-L3 threshold.json ({ts})'
            assert off[ts]['draws'] == [f'cam_nf_s{s}' for s in W.HO_SEEDS]
    rec_list = sorted((r['file'], r['file_sha256']) for rs in houses.values() for r in rs)
    rec_sha = hashlib.sha256(json.dumps(rec_list).encode()).hexdigest()
    os.makedirs(jd, exist_ok=True)
    t0 = time.time(); started = utc()
    jobs = [(houses[h], dict(ts=ts, export=W.TRAIN_HO, threads=args.threads, script_sha256=me)) for ts in SEEDS for h in names]
    print(f'G4-H1G fit: {len(names)} TRAIN hold-out houses x {len(SEEDS)} checkpoints x w {list(WLABS)}, {args.workers} workers, {started}', flush=True)
    recs = []
    for k, r in _pool_iter(args.workers, args.threads, _fit_task, jobs):
        recs.append(r)
        _jdump(r, os.path.join(jd, f"{r['seed']}_{r.get('house') or 'failed_' + str(k)}.json"))
        print(f"[{k}/{len(jobs)}] {r['seed']} {r.get('house')} {'FAILED ' + r.get('error', '') if r.get('failed') else 'ok'} {r.get('seconds', 0):.0f}s "
              f"peak {r.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
    bad = [(r['seed'], r.get('house'), r.get('error')) for r in recs if r.get('failed')]
    assert not bad, f'fit failed: {bad}'
    tables = {}
    for ts in SEEDS:
        rs = sorted([r for r in recs if r['seed'] == ts], key=lambda r: r['house'])
        assert [r['house'] for r in rs] == names
        tables[ts] = {}
        for wl in WLABS:
            mism = int(sum(r['known_mismatch'][wl] for r in rs))
            assert mism == 0, f'{ts} {wl}: known (Lam_b > 0) != k > 0 on {mism} voxels'
            C = np.sum([_dense(r['counts'][wl]) for r in rs], 0)
            tot = np.sum([r['totals'][wl] for r in rs], 0)
            st = fit_strong(C)
            tables[ts][wl] = dict(w=rs[0]['w'][wl], j_occ=rs[0]['j_occ'][wl],
                                  threshold_source=('E3-L3 threshold.json R8 j_occ' if wl == 'learned' else 'g4/e7/criteria_w.json thresholds[ts][w4] R8 j_occ'),
                                  strong=[[int(a), int(b)] for a, b in zip(*np.nonzero(st))], cols=list(COLS),
                                  counts={str(l): C[l].tolist() for l in range(LMAX + 1) if C[l].any()},
                                  lam_b_max=int(max(r['lam_b_max'][wl] for r in rs)), agent_rows=int(sum(r['agent_rows'] for r in rs)),
                                  train=table_stats(C, tot, st))
    T = dict(format=FORMAT, batch='G4-H1G', prereg='D8-75', stage='fit', smoke=bool(args.smoke), utc_start=started, utc_written=utc(),
             rule=dict(n_x_bins=NX, x_bin_width=XW, x_range=[-1.0, 1.0], lam_max=LMAX, q_min=Q_NUM / Q_DEN, n_min=N_MIN,
                       binning='xbin = floor(20 (h_b + Lam_b) / Lam_b) clipped to 39 (= floor((x + 1) / 0.05), x = 1 in the last bin)',
                       population='per agent row (weight 1): voxels with Lam_b > 0 of the agent and R8 target known (t >= 0); q = share with t == 0',
                       strong='n >= 200 and 1000 n_free >= 995 n', readout='decoder occupied and in a strong cell -> free; known area unchanged'),
             tables=tables, houses=names, draws=[f'cam_nf_s{s}' for s in W.HO_SEEDS], n_houses=len(names),
             export=dict(dir=os.path.relpath(W.TRAIN_HO, ROOT), records_sha256=rec_sha, n_files=len(rec_list)),
             checkpoints={ts: dict(ckpt_sha256=HL.E3L3_CKPTS[ts][1], thr_sha256=HL.E3L3_CKPTS[ts][2]) for ts in SEEDS},
             criteria_w=dict(path=os.path.relpath(CRITERIA_W, ROOT), sha256=CRITERIA_W_SHA256),
             script_sha256=me, frozen_sha256=dict(HL.FROZEN, **SELF_FROZEN), threads=args.threads, workers=args.workers, seconds=time.time() - t0)
    if os.path.exists(out):
        raise SystemExit(f'{out} exists (never overwrite)')
    _jdump(T, out)
    s = sha(out)
    for ts in SEEDS:
        for wl in WLABS:
            t = tables[ts][wl]; st = t['train']
            print(f"  {ts} {wl:8s} w {t['w']:.4g} j {t['j_occ']}: strong cells {st['n_strong']} / nonempty {st['n_cells_nonempty']} (n >= 200: "
                  f"{st['n_cells_ge_nmin']}); coverage known {st['coverage_known_truth']:.4f}; gated share of decoder-occupied {st['gated_share_of_pred_occ']:.5f} "
                  f"({st['gated_voxels']} voxels, true free {st['gated_true_free_share']}); TRAIN pooled furn {st['train_pooled_furn_f1']['ungated']:.5f} -> "
                  f"{st['train_pooled_furn_f1']['gated']:.5f}; Lam_b max {t['lam_b_max']}", flush=True)
            print('      strong by Lam_b: ' + '; '.join(f"{l}: x [{v['x_lo']:+.2f}, {v['x_hi']:+.2f}) {len(v['xbins'])} bins {v['voxels']} vox"
                                                     for l, v in st['strong_by_lam'].items()), flush=True)
    print(f'gate_tables.json sha256 {s}  ({out})', flush=True)
    return T, s


# ================================================================ H1 panel (G5): HL.evaluate_file rewritten for ungated + gated
def _gate_stats_new(nfam):
    return dict(changed=0, truth=np.zeros(3, np.int64), cells={}, fam=np.zeros((nfam + 1, 3), np.int64), h=np.zeros((NB + 1, 3), np.int64),
                zone=np.zeros(3, np.int64), furn=np.zeros(2, np.int64))


def _gate_stats_add(st, ch, idx, T, H, zone=None, near_fam=None, nfam=0, layer1=None):
    """report only: changed voxels (truth = target free / occupied / unknown) per cell, family, height band (+ layer 1), new-object zone;
    furniture TP lost / FP removed (scope T.known & region)"""
    if not ch.any():
        return
    cls = [ch & T.free, ch & T.occ, ch & ~T.known]
    st['changed'] += int(ch.sum()); st['truth'] += [int(m.sum()) for m in cls]
    for j, m in enumerate(cls):
        if m.any():
            u, c = np.unique(idx[m], return_counts=True)
            for a, b in zip(u.tolist(), c.tolist()):
                k = f'{a // NX}_{a % NX}'
                st['cells'].setdefault(k, [0, 0, 0])[j] += int(b)
    for j, m in enumerate(cls):
        hb = G._band_sum(m, H.band_z, NB)
        st['h'][:NB, j] += hb
        if layer1 is not None:
            st['h'][NB, j] += int(m[:, :, layer1].sum())
        if near_fam is not None:
            f = near_fam[m]; f = np.where(f < 0, nfam, f)
            st['fam'][:, j] += np.bincount(f, minlength=nfam + 1)
        if zone is not None:
            st['zone'][j] += int((m & zone).sum())
    sc = ch & T.known & H.region
    st['furn'] += [int((sc & T.tr).sum()), int((sc & ~T.tr).sum())]


def _gate_stats_out(st, fams=None):
    o = dict(changed=st['changed'], truth=st['truth'].tolist(), truth_names=['target_free', 'target_occ', 'target_unknown'], cells=st['cells'],
             h={nm: st['h'][i].tolist() for i, nm in enumerate(H_NAMES)}, zone=st['zone'].tolist(),
             furn=dict(tp_lost=int(st['furn'][0]), fp_removed=int(st['furn'][1])))
    if fams is not None:
        o['fam'] = {nm: st['fam'][i].tolist() for i, nm in enumerate(list(fams) + ['none'])}
    return o


def h1_evaluate_file(path, M, cfgs, luts, gbp_check=True):
    """= HL.evaluate_file with each E3-L3 checkpoint encoded once and, per w in cfgs ({wl: {ts: cfg}}), one GaBP + decode feeding BOTH the
    ungated and the gated predictions -> {(wl, gate): res} in the H1L per-file format (+ 'gate' report rows on the gated res)"""
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
    zones = dict(zone=zone, rest=~zone)
    objzone = G.dil6(np.asarray(z['obj'], bool))
    fams = sorted({str(x) for x in z['obj_family']}); fidx = {f: i for i, f in enumerate(fams)}
    oid = np.full(occ.size, -1, np.int64); idx_, off = z['objvox_idx'], z['objvox_off']
    for i in range(len(off) - 1):
        oid[idx_[off[i]:off[i + 1]]] = i
    oid = oid.reshape(occ.shape); ofam = np.array([fidx[str(x)] for x in z['obj_family']], np.int64)
    if (oid >= 0).any():
        ind = ndimage.distance_transform_edt(oid < 0, return_distances=False, return_indices=True)
        near_fam = ofam[oid[ind[0], ind[1], ind[2]]]; del ind
        cheb1 = ndimage.distance_transform_cdt(oid < 0, metric='chessboard') == 1
    else:
        near_fam = np.full(occ.shape, -1, np.int64); cheb1 = np.zeros(occ.shape, bool)
    rstats = G.RelayStats()
    sL, _, _ = R.relay_sums(Lm, F, M['L'].table, rstats)
    sums = {h: R.relay_sums(Lm, F, M['b6'][h].table, rstats)[0] for h in 'AB'}
    nc = len(F['comps'])
    kcs = [sL[ci][1] for ci in range(nc)]
    known = [kcs[ci] > 0 for ci in range(nc)]
    for h in 'AB':
        for ci in range(nc):
            if not np.array_equal(sums[h][ci][1], kcs[ci]):
                raise RuntimeError('count sums differ between tables')
    preds = {'B4': [R.linear_readout(*sL[ci], 0) for ci in range(nc)]}
    for h in 'AB':
        preds[f'B6_{h}'] = [R.linear_readout(*sums[h][ci], M['b6'][h].theta) for ci in range(nc)]
    ctrl = {}
    # ---- controls (verbatim HL.evaluate_file / OP.evaluate R8 block)
    for arm in HL.CONTROLS:
        ps = preds[arm]
        if not all(np.array_equal(ps[ci] >= 0, known[ci]) for ci in range(nc)):
            raise RuntimeError(f'{arm}: known mask != B4')
        rows, acc, kacc, facc, ph = [], {}, {}, {}, np.zeros(4, np.float64)
        for ci, cm in enumerate(F['comps']):
            kc = kcs[ci]; w = len(cm)
            rows.append([R.scored(ps[ci], R8T[ci], H, kc), w])
            for zn, d in OP.zone_counts(ps[ci], R8T[ci], H, kc, zones).items():
                OP._addw(acc, zn, d, w)
            for kd, v in OP.kind_counts(ps[ci], R8T[ci], H, kzones).items():
                OP._addw(kacc, kd, v, w)
            for fk, v in OP.kind_counts(ps[ci], R8T[ci], H, fzones).items():
                OP._addw(facc, fk, v, w)
            ph += np.asarray(OP.phantom_blocks(ps[ci], kc, H, near, near_nf, layer1 + 1), np.float64) * w / n
        ctrl[arm] = dict(rows=rows, zones=OP._lists(acc), kinds=OP._lists(kacc), families=OP._lists(facc), phantoms=ph.tolist())
    out = {}
    for wl in cfgs:
        for g in GATES:
            out[(wl, g)] = dict(R8=dict(ctrl), flips={}, flips_u={}, flip_split={}, z={}, n_agents=n, components=F['comps'], families=fams,
                                layer1=layer1, e3={}, gate={})

    def named(s):
        fa, hh = np.asarray(s['fam']), np.asarray(s['h'])
        return dict(fam={nm: fa[i].tolist() for i, nm in enumerate(fams + ['none'])}, h={nm: hh[i].tolist() for i, nm in enumerate(OP.H_NAMES)})

    # ---- E3-L3 arms: every agent decodes its own belief (HL O1 / O2), then the same H1 layers on the ungated and on the gated predictions
    for ts in SEEDS:
        arm = f'E3L3_{ts}'; hp_ = PAIR[ts]; m0 = M['e3'][ts]
        t0 = time.time()
        with torch.no_grad():
            zp, lp = L.encode_maps(m0['model'], list(Lm), m0['tb'], 'cpu')
            hp, lq, zq = E3.source(zp, lp, 'round')
        zc = HL.z_zone_counts(zq.numpy(), lp.numpy() > 0, dict(zone=zone, objzone=objzone), occ)
        t_enc = time.time() - t0
        for wl, cf in cfgs.items():
            m = cf[ts]; lut = luts[ts][wl]; t1 = time.time()
            P = {'ungated': [], 'gated': []}; mism = 0; gst = _gate_stats_new(len(fams))
            chk = dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, steps=0, checked=bool(gbp_check))
            with torch.no_grad():
                for ci, cm in enumerate(F['comps']):
                    Fl = L.local_forest(F, cm)
                    hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, m['w'], 'round')
                    if gbp_check:
                        c = L.gbp_checks(zq[cm].unsqueeze(1), lq[cm], Fl, m['w'])
                        chk['cf_max'] = max(chk['cf_max'], c['cf_max']); chk['cf_rel_max'] = max(chk['cf_rel_max'], c['cf_rel'])
                        chk['violations'] += c['violations']; chk['steps'] += c['steps']
                    chk['components'] += 1
                    lg = L.decode_np(m0['model'], hb, lb, HL.DEC_CHUNK)
                    kc = kcs[ci]; lbn = lb.numpy(); hbn = hb.numpy(); pu, pg = [], []
                    for i in range(len(cm)):
                        mism += int(((lbn[i] > 0) != (kc > 0)).sum())
                        p = R.net_readout(lg[i], kc, m['jo'])
                        ix = cell_index(hbn[i, 0], lbn[i])
                        q, ch = apply_gate(p, ix, lut)
                        if not np.array_equal(q >= 0, p >= 0):
                            raise RuntimeError('gate changed the known mask')
                        _gate_stats_add(gst, ch, ix, R8T[ci], H, zone, near_fam, len(fams), layer1)
                        pu.append(p); pg.append(q)
                    P['ungated'].append(pu); P['gated'].append(pg)
            chk['pass'] = (not gbp_check) or (chk['cf_max'] <= L.CF_TOL and chk['violations'] == 0)
            t_model = t_enc + time.time() - t1
            for g in GATES:
                t2 = time.time()
                rows, acc, kacc, facc, ph = [], {}, {}, {}, np.zeros(4, np.float64)
                fl, flu, spw, spu = {}, {}, {}, {}
                reuse = 0
                for ci, cm in enumerate(F['comps']):
                    kc = kcs[ci]; cache = {}
                    for i, p in enumerate(P[g][ci]):
                        if not np.array_equal(p >= 0, known[ci]):
                            raise RuntimeError(f'{arm}: prediction known mask != k > 0')
                        key = hashlib.sha1(p.tobytes()).hexdigest()
                        if key in cache:
                            lay = cache[key]; reuse += 1
                        else:
                            lay = dict(row=R.scored(p, R8T[ci], H, kc), zc=OP.zone_counts(p, R8T[ci], H, kc, zones),
                                       kc=OP.kind_counts(p, R8T[ci], H, kzones), fc=OP.kind_counts(p, R8T[ci], H, fzones),
                                       ph=np.asarray(OP.phantom_blocks(p, kc, H, near, near_nf, layer1 + 1), np.float64),
                                       fl={f'B6_{h}': OP.flip_counts(p, preds[f'B6_{h}'][ci], sums[hp_][ci][0], R8T[ci], H, zone) for h in 'AB'},
                                       sp=OP.flip_split(p, preds[f'B6_{hp_}'][ci], sums[hp_][ci][0], R8T[ci], H, near_fam, len(fams), cheb1, layer1))
                            cache[key] = lay
                        rows.append([lay['row'], 1])
                        for zn, d in lay['zc'].items():
                            OP._addw(acc, zn, d, 1)
                        for kd, v in lay['kc'].items():
                            OP._addw(kacc, kd, v, 1)
                        for fk, v in lay['fc'].items():
                            OP._addw(facc, fk, v, 1)
                        ph += lay['ph'] / n
                        for b6, d in lay['fl'].items():
                            OP._addw(fl, b6, d, 1); HL._addf(flu, b6, d, 1.0 / len(cm))
                        OP._addw(spw, 's', lay['sp'], 1); HL._addf(spu, 's', lay['sp'], 1.0 / len(cm))
                res = out[(wl, g)]
                res['R8'][arm] = dict(rows=rows, zones=OP._lists(acc), kinds=OP._lists(kacc), families=OP._lists(facc), phantoms=ph.tolist())
                res['flips'][arm] = OP._lists(fl); res['flips_u'][arm] = HL._flists(flu)
                res['flip_split'][arm] = dict(w=named(OP._lists(spw['s'])), u=named(HL._flists(spu['s'])))
                res['z'][arm] = zc
                res['e3'][arm] = dict(known_mismatch=mism, gbp=dict(chk), w=m['w'], w_label=m['w_label'], j_occ=m['jo'], reused_agent_rows=reuse,
                                      seconds_model=t_model, seconds=t_model + time.time() - t2)
                if g == 'gated':
                    res['gate'][arm] = _gate_stats_out(gst, fams)
    for res in out.values():
        res['zone_voxels'] = int(zone.sum()); res['ood_voxels'] = int(np.asarray(z['ood_mask']).sum())
        res['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    return out


def _sub(root, stage, wl, g):
    return os.path.join(root, stage, WDIR[wl], g)


def _rec_done(p, opts):
    """resume rule (G9): an existing complete record of the same script and tables -> done; anything else -> refuse"""
    if not os.path.exists(p):
        return False
    x = _jload(p)
    if not x.get('failed') and x.get('script_sha256') == opts['script_sha256'] and x.get('tables_sha256') == opts['tables_sha256']:
        return True
    raise SystemExit(f'{p} exists (failed or another script / table version): never overwrite; move it away first')


def _h1_task(job):
    variant, split, panel, rooms, index, seeds, opts = job
    t0 = time.time()
    base = dict(format=HL.JOB_FORMAT, batch='G4-H1G', variant=variant, split=split, panel=panel, rooms=rooms, index=index, seeds={},
                script_sha256=opts['script_sha256'], frozen=HL.FROZEN, self_frozen=SELF_FROZEN, gbp_check=opts['gbp_check'],
                tables_sha256=opts['tables_sha256'], smoke=opts['smoke'])
    recs = {(wl, g): dict(copy.deepcopy(base), w=WREC[wl], w_thr_sha256=(CRITERIA_W_SHA256 if wl == 'w4' else None), gate=g)
            for wl in opts['ws'] for g in GATES}
    try:
        if my_sha() != opts['script_sha256']:
            raise RuntimeError('g4_h1_gate.py changed since the run started')
        check_self_frozen()
        M = HL.load_models(opts['threads']); thr = thresholds()
        _, tsha, luts = load_tables(opts['tables'], opts['tables_sha256'])
        cfgs = {wl: {ts: arm_cfg(M, thr, ts, wl) for ts in SEEDS} for wl in opts['ws']}
        hp = os.path.join(opts['h1_jobs'], HL.job_name(variant, panel, rooms, index) + '.json')
        if not os.path.exists(hp):
            raise RuntimeError(f'H1 record {os.path.basename(hp)} missing')
        h1 = _jload(hp)
        if h1.get('script_sha256') != HL.H1_SCRIPT_SHA256 or h1.get('failed') or h1.get('skipped'):
            raise RuntimeError('H1 record not a completed record of the H1 script 3a6959da')
        if (h1.get('variant'), h1.get('split'), h1.get('panel'), h1.get('rooms'), h1.get('index')) != (variant, split, panel, rooms, index):
            raise RuntimeError('H1 record identity != job')
        for (wl, g), r in recs.items():
            r['ckpt_sha256'] = {ts: M['e3'][ts]['ckpt_sha256'] for ts in SEEDS}; r['thr_sha256'] = {ts: M['e3'][ts]['thr_sha256'] for ts in SEEDS}
            r['w_used'] = {ts: cfgs[wl][ts]['w'] for ts in SEEDS}; r['j_occ'] = {ts: cfgs[wl][ts]['jo'] for ts in SEEDS}
            r['h1_record_sha256'] = sha(hp)
        for seed in seeds:
            ts0 = time.time()
            hs = h1['seeds'].get(str(seed))
            if hs is None:
                raise RuntimeError(f'H1 record has no seed {seed}')
            path = os.path.join(opts['export'], hs['file'])
            fs = sha(path)
            if fs != hs['file_sha256']:
                raise RuntimeError(f"{hs['file']}: sha {fs[:12]} != H1 record {hs['file_sha256'][:12]}")
            res = h1_evaluate_file(path, M, cfgs, luts, opts['gbp_check'])
            for key, r in res.items():
                ok, bad = HL.compare_controls(r, hs)
                r.update(file=hs['file'], file_sha256=fs, h1_controls_equal=ok, h1_controls_unequal=bad, seconds=time.time() - ts0)
                recs[key]['seeds'][str(seed)] = r
        for r in recs.values():
            r['seconds'] = time.time() - t0; r['peak_mb'] = G._peak_mb()
    except Exception as e:  # noqa: BLE001 (recorded; aggregation -> INCOMPLETE)
        for r in recs.values():
            r['failed'] = True; r['error'] = repr(e)[:500]; r['traceback'] = traceback.format_exc()[-3000:]
    for (wl, g), r in recs.items():
        d = os.path.join(_sub(opts['root'], 'h1', wl, g), 'jobs'); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, HL.job_name(variant, panel, rooms, index) + '.json')
        if os.path.exists(p):
            raise RuntimeError(f'{p} appeared during the run (never overwrite)')
        _jdump(r, p)
    any_r = next(iter(recs.values()))
    return dict(variant=variant, panel=panel, rooms=rooms, index=index, failed=bool(any_r.get('failed')), error=any_r.get('error'),
                seconds=any_r.get('seconds', time.time() - t0), peak_mb=any_r.get('peak_mb'))


def h1_cmd(args):
    check_self_frozen()
    assert args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} processes'
    assert args.formal != args.smoke, 'exactly one of --formal (H1 VAL panel) / --smoke (TRAIN)'
    ws = [w for w in args.ws.split(',') if w]
    assert ws and set(ws) <= set(WLABS), f'--ws from {WLABS}'
    if args.formal:
        assert args.tables_sha, '--formal needs --tables-sha (the coordinator confirms the TRAIN tables first)'
        _, tsha, _ = load_tables(args.tables, args.tables_sha, allow_smoke=False)
        root = OUT_ROOT; h1_jobs = HL.H1_JOBS; split, panel, seeds = 'VALIDATION', 'val', H1_SEEDS
        plan = [(v, r, i) for v in HL.VARIANTS for r, i in G.panel_specs('val')]
        print(f'pre-check (H1L review S3): sha of the {2 * len(plan)} H1 VAL npz against the H1 records ...', flush=True)
        bad = []
        for v, r, i in plan:
            hp = os.path.join(h1_jobs, HL.job_name(v, panel, r, i) + '.json')
            assert os.path.exists(hp), f'H1 record {v} r{r}_i{i} missing'
            h1 = _jload(hp)
            assert h1.get('script_sha256') == HL.H1_SCRIPT_SHA256 and not h1.get('failed')
            for s in seeds:
                hs = h1['seeds'][str(s)]
                if sha(os.path.join(HL.EXPORT_DIR, hs['file'])) != hs['file_sha256']:
                    bad.append(hs['file'])
        if bad:
            raise SystemExit(f'{len(bad)} H1 VAL npz differ from the H1 records: {bad[:5]}')
    else:
        _, tsha, _ = load_tables(args.tables, args.tables_sha or None, allow_smoke=True)
        root = SMOKE_ROOT; h1_jobs = H1_SMOKE_JOBS; split, panel = 'TRAIN', 'train'
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(h1_jobs, '*.json')))]
        assert recs and all(r.get('split') == 'TRAIN' and r.get('panel') == 'train' for r in recs), 'smoke: TRAIN H1 records only (never VAL)'
        plan = sorted({(r['variant'], int(r['rooms']), int(r['index'])) for r in recs})
        seeds = tuple(sorted({int(s) for r in recs for s in r['seeds']}))
    safe_out(root)
    opts = dict(root=root, export=HL.EXPORT_DIR, h1_jobs=h1_jobs, threads=args.threads, script_sha256=my_sha(), tables=os.path.abspath(args.tables),
                tables_sha256=tsha, ws=ws, gbp_check=not args.no_gbp_check, smoke=bool(args.smoke))
    jobs = []
    for v, r, i in plan:
        names = [os.path.join(_sub(root, 'h1', wl, g), 'jobs', HL.job_name(v, panel, r, i) + '.json') for wl in ws for g in GATES]
        done = [_rec_done(p, opts) for p in names]
        if all(done):
            continue
        if any(done):
            raise SystemExit(f'{v} r{r}_i{i}: some (w, gate) records exist, others not: never overwrite; move them away first')
        jobs.append((v, split, panel, r, i, seeds, opts))
    for wl in ws:
        for g in GATES:
            d = _sub(root, 'h1', wl, g); os.makedirs(os.path.join(d, 'jobs'), exist_ok=True)
            _jdump(dict(opts=opts, plan=plan, seeds=list(seeds), argv=sys.argv, started=utc(), w=wl, gate=g), os.path.join(d, f'run_{int(time.time())}.json'))
    print(f'G4-H1G h1: {len(jobs)} jobs ({len(plan) - len(jobs)} done), {args.workers} workers x {args.threads} threads, w {ws}, '
          f'tables {tsha[:12]} -> {root}/h1  {utc()}', flush=True)
    t0 = time.time()
    for k, rec in _pool_iter(args.workers, args.threads, _h1_task, jobs, maxtasks=4):
        st = 'FAILED ' + (rec.get('error') or '') if rec.get('failed') else 'ok'
        print(f"[{k}/{len(jobs)}] {HL.job_name(rec['variant'], rec['panel'], rec['rooms'], rec['index'])} {rec.get('seconds') or 0:.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB {st}  elapsed {time.time() - t0:.0f}s", flush=True)
    print(f'G4-H1G h1 done {utc()}; run aggregate next', flush=True)


# ================================================================ guard (G6): the frozen eval_arm + a recording wrapper of L.decode_np
class DecodeCapture:
    """records (h_b, Lam_b, logits) of every L.decode_np call made while active (values untouched); restores L.decode_np on exit"""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = L.decode_np
        orig = self._orig

        def rec(model, hb, lb, chunk=2):
            out = orig(model, hb, lb, chunk)
            self.calls.append((hb, lb, out))
            return out
        L.decode_np = rec
        return self

    def __exit__(self, *exc):
        L.decode_np = self._orig
        return False


def gate_eval_arm(a, calls, F, L_pub, R8T, H, jo, lut, check_rows):
    """the gated twin of one eval_arm result a: R8 rows re-scored on gated predictions from the captured (h_b, Lam_b, logits); every other
    field copied (wire, bytes, K0 untouched). Returns (gated arm, checks)"""
    nc = len(F['comps'])
    if len(calls) != nc + 1:
        raise RuntimeError(f'captured {len(calls)} decode calls, expected {nc} components + 1 (K0)')
    rows = []; wm = np.zeros(2, np.int64); st = _gate_stats_new(0); k = 0
    ok = dict(known_identical=True, fpr_identical=True, ungated_rows_reproduced=(True if check_rows else None))
    for ci, cm in enumerate(F['comps']):
        hb, lb, lg = calls[ci]
        if hb.shape[0] != len(cm) or len(lg) != len(cm):
            raise RuntimeError('captured decode call does not match the component')
        kc = sum((L_pub[x] != UNK).astype(np.int32) for x in cm); hbn = hb.numpy(); lbn = lb.numpy()
        for i in range(len(cm)):
            p = R.net_readout(lg[i], kc, jo)
            ix = cell_index(hbn[i, 0], lbn[i])
            q, ch = apply_gate(p, ix, lut)
            ok['known_identical'] &= bool(np.array_equal(q >= 0, p >= 0))
            if check_rows:
                ok['ungated_rows_reproduced'] &= bool(_jr(R.scored(p, R8T[ci], H, kc)) == _jr(a['R8'][k][0]))
            sq = R.scored(q, R8T[ci], H, kc)
            ok['fpr_identical'] &= bool(sq['unknown_known_fpr'] == a['R8'][k][0]['unknown_known_fpr'])
            rows.append([sq, 1]); wm += L2.whole_miss(q, R8T[ci]); _gate_stats_add(st, ch, ix, R8T[ci], H)
            k += 1
    if k != len(a['R8']):
        raise RuntimeError('agent rows != eval_arm R8 rows')
    ag = {key: copy.deepcopy(v) for key, v in a.items() if key not in ('R8', 'whole_miss')}
    ag['R8'] = rows; ag['whole_miss'] = dict(fn=int(wm[0]), fn_whole_object=int(wm[1]))
    ok['bytes_identical'] = bool(ag['bytes'] == a['bytes'] and ag['bytes_upper'] == a['bytes_upper'])
    ok['gate'] = _gate_stats_out(st)
    return ag, ok


def _guard_task(job):
    rec_e, opts = job
    t0 = time.time()
    base = dict(stage='guard', batch='G4-H1G', panel=rec_e['panel'], rooms=rec_e['rooms'], index=rec_e['index'], seed=rec_e['seed'],
                split=rec_e.get('split'), script_sha256=opts['script_sha256'], tables_sha256=opts['tables_sha256'], criteria_sha256=CRITERIA_W_SHA256,
                smoke=opts['smoke'], tiers={}, checks={})
    recs = {(wl, g): dict(copy.deepcopy(base), w=WREC[wl], gate=g) for wl in opts['ws'] for g in GATES}
    try:
        if my_sha() != opts['script_sha256']:
            raise RuntimeError('g4_h1_gate.py changed since the run started')
        check_self_frozen()
        torch.set_num_threads(max(1, opts['threads']))
        M = HL.load_models(opts['threads']); thr = thresholds()
        _, _, luts = load_tables(opts['tables'], opts['tables_sha256'])
        path = os.path.join(opts['exports'][str(rec_e['seed'])], rec_e['file'])
        if sha(path) != rec_e['file_sha256']:
            raise RuntimeError('export file changed')
        z = np.load(path)
        H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
        F = G.forest(R.edges_list(z['edges']), n)
        assert F['comps'] == [list(c) for c in rec_e['components']]
        K0T = [G.Target(private[a], H) for a in range(n)]
        R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
        for r in recs.values():
            r['ckpts'] = {ts: M['e3'][ts]['ckpt_sha256'] for ts in SEEDS}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            keep = tn == JUDGED_TIER
            for r in recs.values():
                r['tiers'][tn] = {}; r['checks'][tn] = {}
            for ts in SEEDS:
                for wl in opts['ws']:
                    c = arm_cfg(M, thr, ts, wl)
                    with DecodeCapture() as cap:
                        a = E3.eval_arm(c['model'], c['w'], c['thr'], c['tb'], L_pub, L_k0, F, H, K0T, R8T, opts['chunk'])
                    ag, ok = gate_eval_arm(a, cap.calls, F, L_pub, R8T, H, c['jo'], luts[ts][wl], check_rows=keep)
                    del cap
                    fr_u = [r_[0]['unknown_known_fpr'] for r_ in a['R8']]; fr_g = [r_[0]['unknown_known_fpr'] for r_ in ag['R8']]
                    ok['fpr_rows_identical'] = bool(fr_u == fr_g)
                    for x in (a, ag):                                   # = G4-E7 W3: strip strata from K0 / K0_sep and off-tier R8
                        x['K0'] = W._strip(x['K0']); x['K0_sep'] = W._strip(x['K0_sep'])
                        if not keep:
                            x['R8'] = W._strip(x['R8'])
                        x['thr'] = dict(R8=c['thr']['R8']['j_occ'], K0=c['thr']['K0']['j_occ'])
                    recs[(wl, 'ungated')]['tiers'][tn][f'E3L3_{ts}'] = a; recs[(wl, 'gated')]['tiers'][tn][f'E3L3_{ts}'] = ag
                    recs[(wl, 'gated')]['checks'][tn][ts] = ok
                    recs[(wl, 'ungated')]['checks'][tn][ts] = dict(known_mismatch=int(a['known_mismatch']))
        for r in recs.values():
            r['seconds'] = time.time() - t0; r['peak_mb'] = G._peak_mb()
    except Exception as e:  # noqa: BLE001
        for r in recs.values():
            r['failed'] = True; r['error'] = repr(e)[:500]; r['traceback'] = traceback.format_exc()[-3000:]
    for (wl, g), r in recs.items():
        d = os.path.join(_sub(opts['root'], 'guard', wl, g), 'jobs'); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, W.job_tag(rec_e) + '.json')
        if os.path.exists(p):
            raise RuntimeError(f'{p} appeared during the run (never overwrite)')
        _jdump(r, p)
    any_r = next(iter(recs.values()))
    return dict(tag=W.job_tag(rec_e), failed=bool(any_r.get('failed')), error=any_r.get('error'), seconds=any_r.get('seconds', time.time() - t0),
                peak_mb=any_r.get('peak_mb'))


def guard_records(seeds, panels, only=None):
    recs = W.export_records(list(seeds), list(panels))
    if only:
        recs = [r for r in recs if W.job_tag(r) in set(only)]
        assert {W.job_tag(r) for r in recs} == set(only), 'guard smoke task not found'
    return recs


def guard_cmd(args):
    check_self_frozen()
    assert args.workers <= MAX_WORKERS, f'at most {MAX_WORKERS} processes'
    assert args.formal != args.smoke, 'exactly one of --formal (VAL / fresh seeds 3, 4) / --smoke (val_r1_i0_s1)'
    ws = [w for w in args.ws.split(',') if w]; tiers = [t for t in args.tiers.split(',') if t]
    assert ws and set(ws) <= set(WLABS) and JUDGED_TIER in tiers and set(tiers) <= set(R.EVAL_TIERS)
    if args.formal:
        assert args.tables_sha, '--formal needs --tables-sha (the coordinator confirms the TRAIN tables first)'
        _, tsha, _ = load_tables(args.tables, args.tables_sha, allow_smoke=False)
        root = OUT_ROOT; recs = guard_records(GUARD_SEEDS, GUARD_PANELS)
    else:
        _, tsha, _ = load_tables(args.tables, args.tables_sha or None, allow_smoke=True)
        root = SMOKE_ROOT; recs = guard_records((1,), ('val',), SMOKE_GUARD_TAGS)
        assert all(int(r['seed']) == 1 for r in recs), 'the guard smoke never reads noise seeds 3, 4'
    safe_out(root)
    opts = dict(root=root, exports={str(s): W.EXPORTS[s] for s in {int(r['seed']) for r in recs}}, threads=args.threads, script_sha256=my_sha(),
                tables=os.path.abspath(args.tables), tables_sha256=tsha, ws=ws, tiers=tiers, chunk=2, smoke=bool(args.smoke))
    jobs = []
    for r in recs:
        if r.get('failed') or r.get('skipped'):
            raise SystemExit(f'export record {W.job_tag(r)} failed / skipped')
        names = [os.path.join(_sub(root, 'guard', wl, g), 'jobs', W.job_tag(r) + '.json') for wl in ws for g in GATES]
        done = [_rec_done(p, opts) for p in names]
        if all(done):
            continue
        if any(done):
            raise SystemExit(f'{W.job_tag(r)}: some (w, gate) records exist, others not: never overwrite')
        jobs.append((r, opts))
    for wl in ws:
        for g in GATES:
            d = _sub(root, 'guard', wl, g); os.makedirs(os.path.join(d, 'jobs'), exist_ok=True)
            _jdump(dict(opts=opts, tasks=[W.job_tag(r) for r in recs], argv=sys.argv, started=utc(), w=wl, gate=g), os.path.join(d, f'run_{int(time.time())}.json'))
    print(f'G4-H1G guard: {len(jobs)} jobs ({len(recs) - len(jobs)} done), tiers {tiers}, w {ws}, {args.workers} workers, tables {tsha[:12]} -> {root}/guard  {utc()}',
          flush=True)
    t0 = time.time()
    for k, rec in _pool_iter(args.workers, args.threads, _guard_task, jobs, maxtasks=2):
        st = 'FAILED ' + (rec.get('error') or '') if rec.get('failed') else 'ok'
        print(f"[{k}/{len(jobs)}] {rec['tag']} {rec.get('seconds') or 0:.0f}s peak {rec.get('peak_mb') or 0:.0f} MB {st}  elapsed {time.time() - t0:.0f}s",
              flush=True)
    print(f'G4-H1G guard done {utc()}; run aggregate next', flush=True)


# ================================================================ aggregation (G7 / G8)
def _sum_gate(items):
    o = dict(changed=0, truth=np.zeros(3, np.int64), cells={}, h={}, fam={}, zone=np.zeros(3, np.int64), furn=dict(tp_lost=0, fp_removed=0))
    for it in items:
        if not it:
            continue
        o['changed'] += it['changed']; o['truth'] += np.asarray(it['truth'], np.int64); o['zone'] += np.asarray(it.get('zone', [0, 0, 0]), np.int64)
        o['furn']['tp_lost'] += it['furn']['tp_lost']; o['furn']['fp_removed'] += it['furn']['fp_removed']
        for k, v in it['cells'].items():
            a = o['cells'].setdefault(k, [0, 0, 0])
            for j in range(3):
                a[j] += v[j]
        for part in ('h', 'fam'):
            for k, v in (it.get(part) or {}).items():
                a = o[part].setdefault(k, [0, 0, 0])
                for j in range(3):
                    a[j] += v[j]
    t = o['truth']
    o['truth'] = t.tolist(); o['zone'] = o['zone'].tolist()
    o['target_free_share'] = float(t[0] / (t[0] + t[1])) if (t[0] + t[1]) else None
    return o


def guard_aggregate(d):
    """per (panel, seed): tiers -> arm -> the frozen E3._summ_arm over the houses sorted by job tag (= G4-E7 W4) + per-house checks"""
    recs = sorted([_jload(p) for p in glob.glob(os.path.join(d, 'jobs', '*.json'))], key=W.job_tag)
    S = dict(panels={}, failed=[], n_jobs=len(recs), versions={}, checks_failed=[])
    for k in ('script_sha256', 'tables_sha256', 'criteria_sha256', 'ckpts', 'w', 'gate', 'smoke'):
        S['versions'][k] = sorted({json.dumps(r.get(k), sort_keys=True) for r in recs if not r.get('failed')})
    ok = []
    for r in recs:
        if r.get('failed'):
            S['failed'].append(W.job_tag(r)); continue
        ok.append(r)
        for tn, C in r['checks'].items():
            for ts, c in C.items():
                if 'known_mismatch' in c and c['known_mismatch']:
                    S['checks_failed'].append(f"{W.job_tag(r)} {tn} {ts}: known mismatch {c['known_mismatch']}")
                for key in ('known_identical', 'fpr_identical', 'fpr_rows_identical', 'bytes_identical'):
                    if key in c and not c[key]:
                        S['checks_failed'].append(f'{W.job_tag(r)} {tn} {ts}: {key} false')
                if c.get('ungated_rows_reproduced') is False:
                    S['checks_failed'].append(f'{W.job_tag(r)} {tn} {ts}: ungated rows from the captured logits != eval_arm rows')
    for panel in sorted({r['panel'] for r in ok}):
        P = dict(seeds={})
        for seed in sorted({r['seed'] for r in ok if r['panel'] == panel}):
            X = sorted([r for r in ok if r['panel'] == panel and r['seed'] == seed], key=W.job_tag)
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={}, per_house={}, gate={})
            for tn in R.EVAL_TIERS:
                if not X or not all(tn in r['tiers'] for r in X):
                    continue
                arms = sorted(a for a in X[0]['tiers'][tn] if all(a in r['tiers'][tn] for r in X))
                D['tiers'][tn] = {a: E3._summ_arm([r['tiers'][tn][a] for r in X]) for a in arms}
                D['per_house'][tn] = {f"r{r['rooms']}_i{r['index']}": {a: dict(furn=W._house_furn(r['tiers'][tn][a]['R8']), bytes=r['tiers'][tn][a]['bytes'],
                                                                               fpr=[x[0]['unknown_known_fpr'] for x in r['tiers'][tn][a]['R8']])
                                                                       for a in arms} for r in X}
                D['gate'][tn] = {a: _sum_gate([r['checks'][tn].get(a.split('_')[-1], {}).get('gate') for r in X]) for a in arms}
            P['seeds'][str(seed)] = D
        S['panels'][panel] = P
    return S


def compare_tree(mine, theirs, path='', out=None, limit=20):
    """exact equality after a JSON round trip; returns the list of differing paths (at most limit)"""
    out = [] if out is None else out
    a, b = _jr(mine), _jr(theirs)
    if a == b:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if len(out) >= limit:
                break
            if k not in a or k not in b:
                out.append(f'{path}/{k}: missing on one side')
            else:
                compare_tree(a[k], b[k], f'{path}/{k}', out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if len(out) >= limit:
                break
            compare_tree(x, y, f'{path}[{i}]', out, limit)
    else:
        out.append(f'{path}: {str(a)[:80]} != {str(b)[:80]}')
    return out


def gate_h1l(S_learned_ungated, ref, smoke=False):
    """G7 (H1): the learned-w ungated HL.aggregate summary == g4/h1l/summary.json (variants and verdict; smoke: variants of the smoke summary)"""
    diffs = compare_tree(S_learned_ungated.get('variants'), ref.get('variants'), 'variants')
    if not smoke:
        diffs += compare_tree(S_learned_ungated.get('verdict'), ref.get('verdict'), 'verdict')
    return dict(equal=not diffs, diffs=diffs[:20])


def gate_e7(Sg, E7S, wl, tiers=None):
    """G7 (guard): the ungated arm of w == the matching G4-E7 summary arm (E3L3W_{ts} for w = 4, E3L3_{ts} for the learned w), every
    (panel, seed, tier, checkpoint) of the E7 summary (tiers: only those run; cam_nf required)"""
    diffs, n = [], 0
    e7a = 'E3L3W_{}' if wl == 'w4' else 'E3L3_{}'
    for panel, P in E7S['panels'].items():
        for seed, D in P['seeds'].items():
            for tn, T in D['tiers'].items():
                if tiers is not None and tn not in tiers:
                    continue
                for ts in SEEDS:
                    try:
                        mine = Sg['panels'][panel]['seeds'][seed]['tiers'][tn][f'E3L3_{ts}']
                    except KeyError:
                        diffs.append(f'{panel}/{seed}/{tn}/{ts}: missing in this run'); continue
                    diffs += compare_tree(mine, T[e7a.format(ts)], f'{panel}/{seed}/{tn}/{ts}'); n += 1
    return dict(equal=bool(n) and not diffs, compared=n, diffs=diffs[:20])


def gate_e7_smoke(d, wl):
    """smoke G7 (guard): per-house ungated arm vs the G4-E7 smoke record of val_r1_i0_s1 (learned w only: the smoke w* there was 6)"""
    if sha(E7_SMOKE_JOB) != E7_SMOKE_JOB_SHA256:
        return dict(equal=False, diffs=['E7 smoke record sha != pinned'])
    ref = _jload(E7_SMOKE_JOB); diffs, n = [], 0
    if wl != 'learned':
        return dict(equal=None, note='no G4-E7 w = 4 per-house reference on seed 1 (E7 smoke ran w* = 6)')
    for p in glob.glob(os.path.join(d, 'jobs', '*.json')):
        r = _jload(p)
        if W.job_tag(r) != 'val_r1_i0_s1' or r.get('failed'):
            continue
        for tn, T in r['tiers'].items():
            for ts in SEEDS:
                mine = dict(T[f'E3L3_{ts}']); theirs = dict(ref['tiers'][tn][f'E3L3_{ts}'])
                diffs += compare_tree(mine, theirs, f'{tn}/{ts}'); n += 1
    return dict(equal=bool(n) and not diffs, compared=n, diffs=diffs[:20])


def guard_cells(Su, Sg, panel, seeds, tier=JUDGED_TIER):
    """(gated - ungated) per (checkpoint, noise seed) on one panel / tier"""
    cells = {}
    for ns in seeds:
        Du = Su['panels'].get(panel, {}).get('seeds', {}).get(str(ns)); Dg = Sg['panels'].get(panel, {}).get('seeds', {}).get(str(ns))
        if Du is None or Dg is None or tier not in Du['tiers'] or tier not in Dg['tiers']:
            continue
        for ts in SEEDS:
            arm = f'E3L3_{ts}'
            u, g = Du['tiers'][tier].get(arm), Dg['tiers'][tier].get(arm)
            if u is None or g is None:
                continue
            phu, phg = Du['per_house'][tier], Dg['per_house'][tier]
            hd = {h: phg[h][arm]['furn'] - phu[h][arm]['furn'] for h in phu if phu[h][arm]['furn'] is not None and phg[h][arm]['furn'] is not None}
            worst = min(hd, key=hd.get) if hd else None
            cells[f'{ts}|{ns}'] = dict(
                n_houses=len(Du['houses']), ungated_furn=u['R8']['furn_f1'], gated_furn=g['R8']['furn_f1'], d_furn=g['R8']['furn_f1'] - u['R8']['furn_f1'],
                d_free=g['R8']['free_recall'] - u['R8']['free_recall'], d_surface=g['R8']['surface_f1'] - u['R8']['surface_f1'],
                d_furn_tol1=g['R8']['furn_f1_tol1'] - u['R8']['furn_f1_tol1'],
                fpr_identical=bool(g['R8']['unknown_known_fpr'] == u['R8']['unknown_known_fpr']
                                   and all(phg[h][arm]['fpr'] == phu[h][arm]['fpr'] for h in phu)),
                bytes_identical=bool(g['bytes_median'] == u['bytes_median'] and all(phg[h][arm]['bytes'] == phu[h][arm]['bytes'] for h in phu)),
                worst_house=worst, worst_house_d=(hd[worst] if worst else None), houses_same=Du['houses'] == Dg['houses'],
                gate=Dg['gate'][tier].get(arm))
    return cells


def decide(reasons, h1_ungated, h1_gated, ctrl_ok_gated, val_cells, fresh_cells, C=D8_75):
    """D8-75 hard-coded. h1_*: {variant: SAFE / AMBIGUOUS / HARMFUL}; val_cells / fresh_cells: {'ts|ns': cell} (4 each, cam_nf, seeds 3, 4)"""
    eps = 1e-12; reasons = list(reasons)
    keys = [f'{ts}|{ns}' for ts in SEEDS for ns in GUARD_SEEDS]
    for v in HL.JUDGED:
        for nm, d in (('ungated', h1_ungated), ('gated', h1_gated)):
            if (d or {}).get(v) not in RANK:
                reasons.append(f'H1 {nm} {v}: verdict {(d or {}).get(v)}')
    for nm, st in (('VAL', val_cells), ('fresh', fresh_cells)):
        for k in keys:
            if k not in (st or {}):
                reasons.append(f'guard {nm} cell {k} missing')
            elif st[k].get('n_houses') != 24 or not st[k].get('houses_same', True):
                reasons.append(f"guard {nm} cell {k}: {st[k].get('n_houses')} houses")
    out = dict(criteria=C, reasons=reasons)
    if reasons:
        out['verdict'] = 'INCOMPLETE'
        return out
    fix = dict(floating_not_harmful=h1_gated['floating'] != 'HARMFUL',
               novel_scale_not_worse=all(RANK[h1_gated[v]] >= RANK[h1_ungated[v]] for v in ('novel', 'scale')),
               ctrl_ok=bool(ctrl_ok_gated))
    g = {}
    for k in keys:
        c = val_cells[k]
        g[k] = dict(furn=bool(c['d_furn'] >= C['guard_furn_margin'] - eps), free=bool(c['d_free'] >= -eps), fpr=bool(c['fpr_identical']),
                    bytes=bool(c['bytes_identical']))
    guard_ok = all(all(x.values()) for x in g.values())
    fresh_mean = float(np.mean([fresh_cells[k]['d_furn'] for k in keys]))
    fresh_ok = bool(fresh_mean >= C['fresh_demote_mean'] - eps)
    out.update(fix=fix, fix_ok=all(fix.values()), guard=g, guard_ok=guard_ok, fresh_mean_d=fresh_mean, fresh_ok=fresh_ok,
               val_mean_d=float(np.mean([val_cells[k]['d_furn'] for k in keys])))
    if not guard_ok:
        out['verdict'] = 'NO'
    elif all(fix.values()) and fresh_ok:
        out['verdict'] = 'LEVER'
    else:
        why = [k for k, v in fix.items() if not v] + ([] if fresh_ok else [f'fresh mean d {fresh_mean * 100:+.3f} pp < -0.3 pp'])
        out['verdict'] = 'AMBIGUOUS'; out['why_not_lever'] = why
    return out


def _h1_deltas(Su, Sg):
    """report only: per variant / cell, gated - ungated of the H1 verdict readings"""
    out = {}
    cu = (Su.get('verdict') or {}).get('variants', {}); cg = (Sg.get('verdict') or {}).get('variants', {})
    for v in HL.JUDGED:
        if v not in cu or v not in cg:
            continue
        o = {}
        for k, a in cu[v]['own']['cells'].items():
            b = cg[v]['own']['cells'].get(k)
            if b is None:
                continue
            o[k] = dict(d_zone=(b['d_zone'] - a['d_zone']) if (a['d_zone'] is not None and b['d_zone'] is not None) else None,
                        d_all=b['d_all'] - a['d_all'], flip_ungated=a['flip_prec'], flip_gated=b['flip_prec'],
                        flip_vox_ungated=a['flip_n_vox'], flip_vox_gated=b['flip_n_vox'], ph_ungated=a['ph_e3'], ph_gated=b['ph_e3'],
                        d_free=b['d_free'] - a['d_free'], harm_ungated=a['harm'], harm_gated=b['harm'], safe_gated=b['safe'])
        out[v] = o
    return out


def _h1_gate_report(d):
    """report only: summed gate statistics of the gated H1 records, per (variant, seed, arm)"""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, 'jobs', '*.json'))):
        r = _jload(p)
        if r.get('failed'):
            continue
        for sd, x in r['seeds'].items():
            for arm, gs in (x.get('gate') or {}).items():
                out.setdefault(r['variant'], {}).setdefault(sd, {}).setdefault(arm, []).append(gs)
    return {v: {sd: {arm: _sum_gate(items) for arm, items in A.items()} for sd, A in S.items()} for v, S in out.items()}


def aggregate_cmd(args):
    smoke = bool(args.smoke)
    root = SMOKE_ROOT if smoke else OUT_ROOT
    safe_out(root)
    T, tsha, _ = load_tables(args.tables, args.tables_sha or None, allow_smoke=smoke)
    me = my_sha(); reasons = []
    A = dict(batch='G4-H1G', prereg='D8-75', smoke=smoke, utc=utc(), script_sha256=me, tables=dict(path=os.path.abspath(args.tables), sha256=tsha),
             criteria=D8_75, h1={}, guard={}, gates={}, decision={})
    # ---- H1 (each subdir: the frozen HL.aggregate / verdict)
    H1 = {}
    for wl in WLABS:
        for g in GATES:
            d = _sub(root, 'h1', wl, g)
            if not os.path.isdir(os.path.join(d, 'jobs')):
                reasons.append(f'h1 {wl} {g}: no job directory'); continue
            if smoke:
                present = sorted({int(s) for p in glob.glob(os.path.join(d, 'jobs', '*.json')) for s in _jload(p).get('seeds', {})})
                S = HL.aggregate(d, smoke=True, seeds=tuple(present))
            else:
                S = HL.aggregate(d, smoke=False, seeds=H1_SEEDS)
            S['g4_h1g'] = dict(script_sha256=me, w=wl, gate=g)
            _jdump(S, os.path.join(d, 'summary.json'))
            open(os.path.join(d, 'report.txt'), 'w', encoding='utf-8').write(HL.report(S))
            H1[(wl, g)] = S
            vs = S['versions']
            for k, want in (('script_sha256', me),):
                if vs.get(k) != [json.dumps(want)]:
                    reasons.append(f'h1 {wl} {g}: {k} versions {vs.get(k)}')
            tv = sorted({_jload(p).get('tables_sha256') for p in glob.glob(os.path.join(d, 'jobs', '*.json'))})
            if tv != [tsha]:
                reasons.append(f'h1 {wl} {g}: tables sha in the records {tv} != {tsha}')
            vd = S['verdict']
            if vd.get('verdict') == 'INCOMPLETE' or (not smoke and vd.get('reasons')):
                reasons.append(f"h1 {wl} {g}: INCOMPLETE {vd.get('reasons')[:3]}")
            A['h1'][f'{wl}|{g}'] = dict(verdict=vd.get('verdict'), would_be=vd.get('would_be'), ctrl_ok=vd.get('ctrl_ok'), ctrl=vd.get('ctrl'),
                                        variants={v: x.get('verdict') for v, x in vd.get('variants', {}).items()}, n_jobs=S['n_jobs'])
    # ---- G7 H1 gate
    if ('learned', 'ungated') in H1:
        ref_p, ref_sha = (H1L_SMOKE_SUMMARY, H1L_SMOKE_SUMMARY_SHA256) if smoke else (H1L_SUMMARY, H1L_SUMMARY_SHA256)
        if sha(ref_p) != ref_sha:
            A['gates']['h1l'] = dict(equal=False, diffs=[f'{ref_p} sha != pinned'])
        else:
            A['gates']['h1l'] = gate_h1l(H1[('learned', 'ungated')], _jload(ref_p), smoke)
    else:
        A['gates']['h1l'] = dict(equal=False, diffs=['learned-w ungated H1 arm missing'])
    if not A['gates']['h1l']['equal']:
        reasons.append(f"gate: learned-w ungated H1 != {'h1l_smoke/train' if smoke else 'h1l'} summary {A['gates']['h1l']['diffs'][:3]}")
    for wl in WLABS:
        if (wl, 'ungated') in H1 and (wl, 'gated') in H1:
            A['h1'][f'{wl}|deltas'] = _h1_deltas(H1[(wl, 'ungated')], H1[(wl, 'gated')])
            A['h1'][f'{wl}|gate_stats'] = _h1_gate_report(_sub(root, 'h1', wl, 'gated'))
    # ---- guard
    GS = {}
    tiers_run = None
    for wl in WLABS:
        for g in GATES:
            d = _sub(root, 'guard', wl, g)
            if not os.path.isdir(os.path.join(d, 'jobs')):
                reasons.append(f'guard {wl} {g}: no job directory'); continue
            S = guard_aggregate(d)
            _jdump(S, os.path.join(d, 'summary.json'))
            GS[(wl, g)] = S
            if S['failed']:
                reasons.append(f"guard {wl} {g}: {len(S['failed'])} failed jobs {S['failed'][:4]}")
            if S['checks_failed']:
                reasons.append(f"guard {wl} {g}: checks failed {S['checks_failed'][:4]}")
            for k, want in (('script_sha256', me), ('tables_sha256', tsha), ('criteria_sha256', CRITERIA_W_SHA256)):
                if S['versions'].get(k) != [json.dumps(want)]:
                    reasons.append(f"guard {wl} {g}: {k} versions {S['versions'].get(k)}")
            if not smoke:
                want = {W.job_tag(r) for r in guard_records(GUARD_SEEDS, GUARD_PANELS)}
                have = {W.job_tag(_jload(p)) for p in glob.glob(os.path.join(d, 'jobs', '*.json'))}
                if want - have:
                    reasons.append(f'guard {wl} {g}: {len(want - have)} tasks missing, e.g. {sorted(want - have)[:3]}')
                if have - want:
                    reasons.append(f'guard {wl} {g}: {len(have - want)} foreign tasks, e.g. {sorted(have - want)[:3]}')
            tr = sorted({tn for P in S['panels'].values() for D in P['seeds'].values() for tn in D['tiers']})
            tiers_run = tr if tiers_run is None else sorted(set(tiers_run) & set(tr))
    # ---- G7 guard gates
    for wl in WLABS:
        if (wl, 'ungated') not in GS:
            A['gates'][f'e7_{wl}'] = dict(equal=False, diffs=['missing']); continue
        if smoke:
            A['gates'][f'e7_{wl}'] = gate_e7_smoke(_sub(root, 'guard', wl, 'ungated'), wl)
        elif sha(E7_SUMMARY) != E7_SUMMARY_SHA256:
            A['gates'][f'e7_{wl}'] = dict(equal=False, diffs=['e7 summary sha != pinned'])
        else:
            A['gates'][f'e7_{wl}'] = gate_e7(GS[(wl, 'ungated')], _jload(E7_SUMMARY), wl, tiers_run)
        if A['gates'][f'e7_{wl}'].get('equal') is False:
            reasons.append(f"gate: {wl} ungated guard arm != G4-E7 {A['gates'][f'e7_{wl}']['diffs'][:3]}")
    # ---- per w: cells, report-only rows, decision
    for wl in WLABS:
        Su, Sg = GS.get((wl, 'ungated')), GS.get((wl, 'gated'))
        cells = {}
        if Su and Sg:
            for panel in GUARD_PANELS if not smoke else ('val',):
                for tn in (tiers_run or [JUDGED_TIER]):
                    cells[f'{panel}|{tn}'] = guard_cells(Su, Sg, panel, GUARD_SEEDS if not smoke else (1,), tn)
        A['guard'][wl] = cells
        hu, hg = H1.get((wl, 'ungated')), H1.get((wl, 'gated'))
        vu = {v: x.get('verdict') for v, x in ((hu or {}).get('verdict', {}).get('variants') or {}).items()}
        vg = {v: x.get('verdict') for v, x in ((hg or {}).get('verdict', {}).get('variants') or {}).items()}
        ctrl_g = (hg or {}).get('verdict', {}).get('ctrl_ok')
        if smoke:
            A['decision'][wl] = dict(verdict='SMOKE (not a verdict)', h1_ungated=vu, h1_gated=vg, ctrl_ok_gated=ctrl_g, reasons=list(reasons))
        else:
            A['decision'][wl] = decide(reasons, vu, vg, ctrl_g, cells.get(f'val|{JUDGED_TIER}', {}), cells.get(f'fresh|{JUDGED_TIER}', {}))
            A['decision'][wl].update(h1_ungated=vu, h1_gated=vg, ctrl_ok_gated=ctrl_g)
    od = safe_out(os.path.join(root, 'aggregate')); os.makedirs(od, exist_ok=True)
    _jdump(A, os.path.join(od, 'summary.json'))
    txt = report(A)
    open(os.path.join(od, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
    return A


def _pp(x):
    return '   n/a' if x is None else f'{x * 100:+.3f}'


def report(A):
    L_ = [f"G4-H1G aggregate ({'SMOKE' if A['smoke'] else 'formal'}), tables {A['tables']['sha256'][:12]}, script {A['script_sha256'][:12]}, {A['utc']}"]
    for k, g in A['gates'].items():
        L_.append(f"gate {k}: equal {g.get('equal')} compared {g.get('compared')} {g.get('note') or ''} {g.get('diffs', [])[:3]}")
    for k, h in A['h1'].items():
        if k.endswith('|deltas') or k.endswith('|gate_stats'):
            continue
        L_.append(f"H1 {k}: {h['variants']} ctrl_ok {h['ctrl_ok']} ({h['n_jobs']} jobs)")
    for wl in WLABS:
        dl = A['h1'].get(f'{wl}|deltas') or {}
        for v, cells in dl.items():
            for k, c in cells.items():
                L_.append(f"  {wl} {v} {k}: flip {c['flip_ungated']} -> {c['flip_gated']} (vox {c['flip_vox_ungated']:.0f} -> {c['flip_vox_gated']:.0f}), "
                          f"zone d {_pp(c['d_zone'])} pp, all d {_pp(c['d_all'])} pp, phantoms {c['ph_ungated']:.2f} -> {c['ph_gated']:.2f}, "
                          f"harm {c['harm_ungated']} -> {c['harm_gated']}")
        gs = A['h1'].get(f'{wl}|gate_stats') or {}
        for v, S in gs.items():
            for sd, X in S.items():
                for arm, x in X.items():
                    L_.append(f"  {wl} gate stats {v} s{sd} {arm}: changed {x['changed']} (target free / occ / unknown {x['truth']}, free share "
                              f"{x['target_free_share']}), zone {x['zone']}, furniture TP lost {x['furn']['tp_lost']} FP removed {x['furn']['fp_removed']}")
        for key, cells in (A['guard'].get(wl) or {}).items():
            for k, c in cells.items():
                gt = c.get('gate') or {}
                L_.append(f"  guard {wl} {key} {k}: furn {c['ungated_furn']:.5f} -> {c['gated_furn']:.5f} (d {_pp(c['d_furn'])} pp, worst {c['worst_house']} "
                          f"{_pp(c['worst_house_d'])}), free d {_pp(c['d_free'])}, surf d {_pp(c['d_surface'])}, FPR same {c['fpr_identical']}, bytes same "
                          f"{c['bytes_identical']}; changed {gt.get('changed')} (free share {gt.get('target_free_share')})")
        d = A['decision'].get(wl, {})
        L_.append(f"DECISION w={wl}: {d.get('verdict')}  fix {d.get('fix')} guard_ok {d.get('guard_ok')} fresh mean {_pp(d.get('fresh_mean_d'))} "
                  f"why {d.get('why_not_lever')} reasons {d.get('reasons', [])[:4]}")
    return '\n'.join(L_)


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fit', 'h1', 'guard', 'aggregate'):
        p = sub.add_parser(name)
        p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=MAX_WORKERS); p.add_argument('--threads', type=int, default=1)
        if name == 'fit':
            p.add_argument('--only', default='')
        if name in ('h1', 'guard', 'aggregate'):
            p.add_argument('--tables', default=os.path.join(OUT_ROOT, 'fit', 'gate_tables.json')); p.add_argument('--tables-sha', default='')
        if name in ('h1', 'guard'):
            p.add_argument('--formal', action='store_true'); p.add_argument('--ws', default=','.join(WLABS))
        if name == 'h1':
            p.add_argument('--no-gbp-check', action='store_true')
        if name == 'guard':
            p.add_argument('--tiers', default=','.join(R.EVAL_TIERS))
    a = ap.parse_args(argv)
    G.setup_env()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if a.cmd == 'fit':
        return fit_cmd(a)
    if a.cmd == 'h1':
        return h1_cmd(a)
    if a.cmd == 'guard':
        return guard_cmd(a)
    return aggregate_cmd(a)


if __name__ == '__main__':
    main()
