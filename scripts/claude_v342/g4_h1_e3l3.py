"""g4_h1_e3l3.py (Claude, 2026-10-28): 批次 G4-H1L "E3-L3 过留出形状面板"（采纳前置；纯 CPU，0 个租约，冻结检查点，不重训）。
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-69（逐条照做；指标与判据照 D8-53，ctrl 自检区间改为 [+4.3, +8.6] pp）；
原文草案 review_E3L3/RI_accuracy.md §2.4 / §5.1，原型 review_E3L3/RI/ri_probe.py（job_ood）。
只 import 冻结代码，从不修改：g4_ood_panel.py（H1，sha 3a6959da：计分分层 zone_counts / kind_counts / flip_counts / flip_split /
phantom_blocks / object_zones、汇总小函数）、g4_e3l3.py（E3-L3，c7523cc8：load_model / source）及其 import 的 g4_e3l2 / g4_e3l
（encode_maps / gabp / local_forest / decode_np / gbp_checks）/ g4_e3r / g4_noise_panels，派生生成器 mapping_3d_storey_data_v420ood（9e1c7c2a，
只经 g4_ood_panel 间接 import，本脚本不生成任何世界）。不改任何已有文件。

面板（D8-69 = D8-53 原样）：ctrl / novel / scale / floating × VAL 24 房（G.panel_specs('val')）× 噪声种子 1、2，cam_nf，V3。
  复用 g4/h1_export/ 的 192 个 VAL npz，不重新导出：每个文件先算 sha256，与 g4/h1/jobs/{变体}_val_r{r}_i{i}.json 的 seeds[种子].file_sha256
  比对，不符则该任务失败（→ INCOMPLETE）。
臂：E3L3_342101 ↔ B6_A，E3L3_342202 ↔ B6_B。冻结检查点 review_E3R/inputs/e3l3_s{种子}/final.pt（完整 sha 钉住，前缀 1fa0cb39 / 3d332d40），
  阈值 = 同目录 threshold.json 的 R8 j_occ（完整 sha 钉住）；每台机器解码自己的 GaBP 信念（学到的 w，逐跳步长 1 取整 'round'）；
  已知 = Lam_b > 0，逐体素断言与 k > 0 相同（计不一致体素数，> 0 → INCOMPLETE）。
指标（D8-53，只把 NET 换成 E3-L3）：新物体区 = G.dil6(ood_mask)（OP.object_zones）；R8 精确家具 F1（zone 计数池化）；整体家具 F1 = E3-R 口径
  （分量展开逐行 furn_f1 均值，G.summarise_rows）；空闲召回同；翻转精确率 = {E3-L3 判占据 ∧ B6 判空闲 ∧ S <= 0 ∧ 计分范围} 中 T.tr 的比例，
  S = 配对表（342101 → A，342202 → B）的融合和；E3-L3 的接收方看不到 S，S 只用来定义"零学习证据认为空闲"；每房幻影块（OP.phantom_blocks）。
判据（D8-69，写死在 CRIT；novel / scale / floating 各判一次，2 检查点 × 2 噪声种子 = 4 格）：
  安全（4 格都满足）：zone E3L3 − B6 >= −1 pp，整体 E3L3 − B6 >= 0，翻转精确率 >= .70，每房幻影块 <= B6 + 0.5，空闲召回 >= B6 − 0.5 pp；
  有害：任一格 zone E3L3 − B6 < −1 pp，或翻转精确率 < .50，或每房幻影块 > B6 + 2；其余 AMBIGUOUS。
  ctrl 自检：4 格 E3L3 − max(B4, B6_A, B6_B)（整体 R8 家具）在 [+4.3, +8.6] pp 内，否则整批 INCONCLUSIVE。
只报告（D8-69）：E3L3 − NET（zone、整体、翻转精确率、幻影块、空闲召回；NET 读数引自 g4/h1/summary.json，sha 记入汇总）；按家族的 zone ΔF1
  与翻转精确率、按高度带与楼板上第 1 层的翻转；新物体区内 z 三级分布相对 ctrl 的变化；有限 w 读法（--w，见 O9）。

操作化（D8-69 没写死、在这里写定；写于看任何 VAL 读数之前）
  O1 每台机器一行：E3L3 的行 = [E.scored(p_i, R8T[分量], H, k), 1]，按机器逐个；B4 / B6 沿用 H1 的 [row, |分量|]。两者展开后都是"每台机器
     一行"，与 E3-L3 正式评估（g4_e3l3.eval_arm）和 NET 的分量展开等价。zone / kind / family / flip 计数同样每台机器权重 1；幻影块按
     OP.evaluate 的写法逐机器 / n 相加（= 逐机信念的均值）。
  O2 编码、GaBP、解码与 E3-L3 正式评估逐行相同：L.encode_maps（4 台一块）→ E3.source(.., 'round') → 每分量 L.local_forest + L.gabp(w, 'round')
     → L.decode_np(chunk 2) → R.net_readout(logit, k, j_occ)（已知 = k > 0，占据 = logit bin >= j_occ）。同一分量内两台机器的预测逐位相同时
     复用计分（结果逐位不变，只省时间）。
  O3 B4 / B6 读出与 OP.evaluate 逐行相同（relay_sums、linear_readout、E.scored、zone / kind / family 计数、幻影块），并作一致性闸门：
     每个 (变体, 房, 种子) 的 B4 / B6_A / B6_B 的 rows、zones、kinds、families、phantoms 必须与 g4/h1/jobs 的记录逐值相等（JSON 往返后比较），
     否则该任务记 h1_mismatch，汇总判 INCOMPLETE。这保证 E3-L3 与 H1 的 NET / B6 是在同一批地图、同一套计分上读的。
  O4 翻转：主判 = 自己的配对（342101 ↔ B6_A，S = 表 A 的和）；OP 的 'max' 配对（该 (变体, 种子) 里 R8 家具较高的 B6，S 仍取自己的表）只报告，
     判决不同时标出。翻转精确率分母为 0 记通过（P3）。"未加权"翻转体素数 = 每台机器权重 1 / |分量|（分量内取平均；浮点）。
  O5 z 三级分布（只报告）：源端码 z_i = round(tanh(u_i))（足迹内），在 zone = dil6(ood_mask) 与 objzone = dil6(obj) 内、按生成器真值
     occ == 1 / occ == 0 分开计 −1 / 0 / +1 的个数，逐机器相加。判决变体里 zone == objzone（H1 S6）；ctrl 没有新物体，用 objzone 作对照。
  O6 完整性：期望 = 4 变体 × 24 房 × 种子 {1, 2}；缺任务、失败、npz sha 不符、H1 记录缺失或不是 H1 正式脚本（3a6959da）写的、O3 不等、
     已知不一致 > 0、GaBP 核对失败（开启时）、脚本 / 冻结模块 / 检查点 / 阈值 sha 混用 → INCOMPLETE。正式汇总只收 split = VALIDATION 且
     panel = val 的任务；冒烟只收 TRAIN（OP 的 S4b 同一规则）。
  O7 GaBP 核对（默认开启，--no-gbp-check 关闭）：每个分量 L.gbp_checks（float64 GaBP 对闭式解、能量单调），cf_max <= L.CF_TOL 且违例 0，
     否则 INCOMPLETE（同 D8-63 的评估断言）。
  O8 冻结模块：worker 内重算本脚本与全部冻结模块的完整 sha 并与 FROZEN / 启动时比对；E3-L3 的证据表断言等于配对的 B6 表（E1c fit.json，
     完整 sha 钉住）。
  O9 --w（默认 learned = 检查点里学到的 w）：给出数值 w 时还必须给 --w-thr（JSON：{"342101": j_occ 或 {"R8": {"j_occ": ..}}, "342202": ..}，
     即 G4-E7 第 0 步在 TRAIN 上为该 w 重选的阈值），文件 sha 记入记录；有限 w 的运行只报告，汇总的 verdict 固定为 "REPORT ONLY (finite w)"，
     判据数字照算、照列。已知 = Lam_b > 0 的断言照旧（w <= 1 时会失败，判 INCOMPLETE）。
  O10 NET 读数只从 g4/h1/summary.json 引（完整 sha 钉住），不重算 NET。

用法（总是 python -B；CPU；本机 <= 10 进程）：
  run --formal [--workers 8] [--threads 1] [--out D] [--w learned | --w 4 --w-thr THR.json] [--no-gbp-check]     # 正式（VAL）
  run --smoke --h1-jobs g4/h1_smoke2/train/jobs --out g4/h1l_smoke/train [--workers 4]                          # 冒烟（只用 TRAIN）
  aggregate --out D [--smoke]
VAL 只在 --formal 下计分；冒烟不得读 VAL；不生成 TEST，不生成任何世界。
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, hashlib, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_ood_panel as OP  # noqa: E402   (frozen H1; imports g4_e3r and the derived generator)
import g4_e3l3 as E3  # noqa: E402        (frozen E3-L3; imports g4_e3l2 -> g4_e3l -> g4_e3r -> g4_noise_panels)

L = E3.L
R = E3.R
G = E3.G
assert OP.E is R and OP.G is G, 'g4_ood_panel and g4_e3l3 must share the same g4_e3r / g4_noise_panels'
torch = E3.torch
UNK = R.UNK

ROOT = G.ROOT
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
FIT_SHA256 = '8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2'
SEEDS = ('342101', '342202')
PAIR = {'342101': 'A', '342202': 'B'}
E3L3_CKPTS = {'342101': ('e3l3_s342101', '1fa0cb396f8f112ff14108691de992c195130513cbd86c83c84481b3155ecf6f',
                         'c583fa1ebad811db4f8e5ac2dc67fdbb7e2d6271a76b2fac0079b025cf50605e'),
              '342202': ('e3l3_s342202', '3d332d400813374869df58b7c19f6361899812e8d8b2fdc6d9305969c7d4111c',
                         '56f4bf7f9ecb33798dd5a14538f44632195a325ca8754f1399c53d122d2c8c73')}
FROZEN = dict(g4_ood_panel='3a6959da19eec5326f9a17e583d58e6ab1b12fd2a2c398172b61ace36a49ce23',
              mapping_3d_storey_data_v420ood='9e1c7c2aac5ddd3e53ed5035a3db707d87c1a1026987ece782796b39f4e1f92c',
              g4_e3l3='c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c',
              g4_e3l2='bab73d647655ecc6504d4ffd738409618bf95cab8089a489a7cb4da28dc3f314',
              g4_e3l='3dc970d8542f794082b28d265d68e0237f3670f2cf2dfd4f754c01c8ac24f6be',
              g4_e3r='1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970',
              g4_noise_panels='2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f')
H1_SCRIPT_SHA256 = FROZEN['g4_ood_panel']
H1_JOBS = os.path.join(AN, 'g4', 'h1', 'jobs')
H1_SUMMARY = os.path.join(AN, 'g4', 'h1', 'summary.json')
H1_SUMMARY_SHA256 = '2860724319d95d3f4daef737d25148c799dd22f50e211433727479175c6ef558'
EXPORT_DIR = os.path.join(AN, 'g4', 'h1_export')
OUT_DIR = os.path.join(AN, 'g4', 'h1l')
SMOKE_DIR = os.path.join(AN, 'g4', 'h1l_smoke')
TIER = 'cam_nf'
NOISE_SEEDS = (1, 2)
VARIANTS = OP.VARIANTS
JUDGED = OP.JUDGED
CONTROLS = ('B4', 'B6_A', 'B6_B')
ARMS = tuple(f'E3L3_{ts}' for ts in SEEDS)
# D8-69 = D8-53 criteria with the ctrl band lowered by 1 pp (E3-L3's non-inferiority tolerance)
CRIT = dict(zone_safe=-0.01, overall_safe=0.0, flip_safe=0.70, phantom_safe=0.5, free_safe=-0.005,
            zone_harm=-0.01, flip_harm=0.50, phantom_harm=2.0, ctrl_lo=0.043, ctrl_hi=0.086)
assert {k: v for k, v in CRIT.items() if not k.startswith('ctrl')} == {k: v for k, v in OP.CRIT.items() if not k.startswith('ctrl')}
DEC_CHUNK = 2                      # = the formal E3-L3 eval (--chunk 2)
JOB_FORMAT = 1
ZK = ('zone', 'objzone')


def sha(path):
    return G.file_sha256(path)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)


def _jr(x):
    """JSON round trip (the H1 records went through json.dump)"""
    return json.loads(json.dumps(x))


def frozen_files():
    import mapping_3d_storey_data_v420ood as WO
    return dict(g4_ood_panel=OP.__file__, mapping_3d_storey_data_v420ood=WO.__file__, g4_e3l3=E3.__file__, g4_e3l2=E3.L2.__file__,
                g4_e3l=L.__file__, g4_e3r=R.__file__, g4_noise_panels=G.__file__)


def check_frozen():
    for nm, p in frozen_files().items():
        s = sha(p)
        if s != FROZEN[nm]:
            raise RuntimeError(f'{nm}.py sha {s[:12]} != frozen {FROZEN[nm][:12]}')


# ================================================================ E3-L3 (shared with g4_e6_e3l3.py)
def parse_w_thr(path):
    """O9: {"342101": j or {"R8": {"j_occ": j}}, ...} -> {ts: int j}"""
    d = _jload(path); out = {}
    for ts in SEEDS:
        v = d[ts]
        out[ts] = int(v['R8']['j_occ']) if isinstance(v, dict) and 'R8' in v else (int(v['j_occ']) if isinstance(v, dict) else int(v))
    return out


def load_e3l3(threads=1, w=None, w_thr=None):
    """the two frozen E3-L3 checkpoints (full sha pinned) with their R8 thresholds; w = None -> the learned w, else the given finite w
    with the thresholds w_thr {ts: j_occ} (O9). Each also carries its evidence table (asserted = the paired B6 table by the callers)."""
    torch.set_num_threads(max(1, int(threads)))
    out = {}
    for ts in SEEDS:
        dn, ck_sha, thr_sha = E3L3_CKPTS[ts]
        p = os.path.join(INP, dn, 'final.pt'); tp = os.path.join(INP, dn, 'threshold.json')
        s1, s2 = sha(p), sha(tp)
        if s1 != ck_sha or s2 != thr_sha:
            raise RuntimeError(f'E3-L3 {ts}: checkpoint / threshold sha {s1[:12]} / {s2[:12]} != pinned {ck_sha[:12]} / {thr_sha[:12]}')
        ck, model = E3.load_model(p, 'cpu')
        if str(ck['seed']) != ts:
            raise RuntimeError(f'E3-L3 checkpoint seed {ck["seed"]} != {ts}')
        thr = _jload(tp)
        wl = float(model.w.detach())
        tb = np.asarray(ck['recipe']['evidence']['table'])
        if w is None:
            ww, jo, wlab = wl, int(thr['R8']['j_occ']), 'learned'
        else:
            if w_thr is None:
                raise RuntimeError('--w needs --w-thr (thresholds re-selected for that w on TRAIN, D8-67)')
            ww, jo, wlab = float(w), int(w_thr[ts]), f'finite:{float(w):g}'
        out[ts] = dict(model=model, w=ww, w_learned=wl, w_label=wlab, jo=jo, jo_learned=int(thr['R8']['j_occ']), table=tb,
                       tb=torch.as_tensor(tb, dtype=torch.int32), half=PAIR[ts], ckpt_sha256=s1, thr_sha256=s2)
    return out


@torch.no_grad()
def e3l3_preds(m, maps, F, kcs, gbp_check=True, chunk=DEC_CHUNK):
    """O2: every agent encodes its own map, source potentials (z, 1) on its footprint, per component two-pass GaBP with per-hop rounding at
    w, every member decodes ITS OWN belief; prediction = R.net_readout(logit, k, j_occ) (known = k > 0). Asserted (counted): Lam_b > 0 == k > 0.
    Returns (preds [component][member] int8 arrays, z [n, X, Y, Z] float, footprint [n, X, Y, Z] bool, info)."""
    zp, lp = L.encode_maps(m['model'], list(maps), m['tb'], 'cpu')
    hp, lq, zq = E3.source(zp, lp, 'round')
    out = []; mism = 0
    chk = dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, steps=0, checked=bool(gbp_check))
    for ci, cm in enumerate(F['comps']):
        Fl = L.local_forest(F, cm)
        hb, lb, _ = L.gabp(hp[cm], lq[cm], Fl, m['w'], 'round')
        if gbp_check:
            c = L.gbp_checks(zq[cm].unsqueeze(1), lq[cm], Fl, m['w'])
            chk['cf_max'] = max(chk['cf_max'], c['cf_max']); chk['cf_rel_max'] = max(chk['cf_rel_max'], c['cf_rel'])
            chk['violations'] += c['violations']; chk['steps'] += c['steps']
        chk['components'] += 1
        lg = L.decode_np(m['model'], hb, lb, chunk)
        kc = kcs[ci]; lbn = lb.numpy(); ps = []
        for i in range(len(cm)):
            mism += int(((lbn[i] > 0) != (kc > 0)).sum())
            ps.append(R.net_readout(lg[i], kc, m['jo']))
        out.append(ps)
    chk['pass'] = (not gbp_check) or (chk['cf_max'] <= L.CF_TOL and chk['violations'] == 0)
    return out, zq.numpy(), lp.numpy() > 0, dict(known_mismatch=mism, gbp=chk, w=m['w'], w_label=m['w_label'], j_occ=m['jo'])


# ================================================================ one exported (variant, house, seed) file
_M = None


def load_models(threads=1, w=None, w_thr=None):
    global _M
    if _M is None:
        if sha(FIT) != FIT_SHA256:
            raise RuntimeError('E1c fit.json sha mismatch')
        b6 = {h: R.Evidence.from_spec(f'c:{FIT}:{h}:cam_nf') for h in 'AB'}
        e3 = load_e3l3(threads, w, w_thr)
        for ts, m in e3.items():
            if not np.array_equal(m['table'], b6[PAIR[ts]].table):
                raise RuntimeError(f'E3L3_{ts} evidence table != B6_{PAIR[ts]} (O8)')
        _M = dict(b6=b6, L=R.Evidence.from_spec('L'), e3=e3)
    return _M


def _addf(acc, key, d, w):
    """float accumulation of nested count dicts / arrays (O4 'unweighted' rows use w = 1 / |component|)"""
    if isinstance(d, dict):
        a = acc.setdefault(key, {})
        for k, v in d.items():
            _addf(a, k, v, w)
    else:
        v = np.asarray(d, np.float64) * float(w)
        acc[key] = acc[key] + v if key in acc else v


def _flists(d):
    if isinstance(d, dict):
        return {k: _flists(v) for k, v in d.items()}
    return np.asarray(d).tolist()


def z_zone_counts(zq, fp, masks, occ):
    """O5: counts of z = -1 / 0 / +1 on the footprint inside each mask, split by generator truth occ == 1 / occ == 0 -> {mask: [[3], [3]]}"""
    out = {}
    for nm, Mk in masks.items():
        rows = []
        for t in (occ, ~occ):
            sel = fp & Mk & t; v = zq[sel]
            rows.append([int((v < -0.5).sum()), int((np.abs(v) < 0.5).sum()), int((v > 0.5).sum())])
        out[nm] = rows
    v = zq[fp]
    out['footprint'] = [[int((v < -0.5).sum()), int((np.abs(v) < 0.5).sum()), int((v > 0.5).sum())]]
    return out


def evaluate_file(path, M, gbp_check=True):
    """controls B4 / B6_A / B6_B exactly as OP.evaluate (O3), and the two E3-L3 arms per agent (O1 / O2) with the H1 layers"""
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
    oid = np.full(occ.size, -1, np.int64); idx, off = z['objvox_idx'], z['objvox_off']
    for i in range(len(off) - 1):
        oid[idx[off[i]:off[i + 1]]] = i
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
    res = dict(R8={}, flips={}, flips_u={}, flip_split={}, z={}, n_agents=n, components=F['comps'], families=fams, layer1=layer1, e3={})
    # ---- controls (O3: identical to OP.evaluate's R8 block)
    for arm in CONTROLS:
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
        res['R8'][arm] = dict(rows=rows, zones=OP._lists(acc), kinds=OP._lists(kacc), families=OP._lists(facc), phantoms=ph.tolist())
    # ---- E3-L3 arms, every agent decodes its own belief (O1 / O2)
    for ts in SEEDS:
        m = M['e3'][ts]; arm = f'E3L3_{ts}'; hp_ = PAIR[ts]
        t0 = time.time()
        ep, zq, fp, info = e3l3_preds(m, Lm, F, kcs, gbp_check)
        t_model = time.time() - t0
        rows, acc, kacc, facc, ph = [], {}, {}, {}, np.zeros(4, np.float64)
        fl, flu, spw, spu = {}, {}, {}, {}
        reuse = 0
        for ci, cm in enumerate(F['comps']):
            kc = kcs[ci]; cache = {}
            for i, p in enumerate(ep[ci]):
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
                    OP._addw(fl, b6, d, 1); _addf(flu, b6, d, 1.0 / len(cm))
                OP._addw(spw, 's', lay['sp'], 1); _addf(spu, 's', lay['sp'], 1.0 / len(cm))
        zc = z_zone_counts(zq, fp, dict(zone=zone, objzone=objzone), occ)

        def named(s):
            fa, hh = np.asarray(s['fam']), np.asarray(s['h'])
            return dict(fam={nm: fa[i].tolist() for i, nm in enumerate(fams + ['none'])}, h={nm: hh[i].tolist() for i, nm in enumerate(OP.H_NAMES)})
        res['R8'][arm] = dict(rows=rows, zones=OP._lists(acc), kinds=OP._lists(kacc), families=OP._lists(facc), phantoms=ph.tolist())
        res['flips'][arm] = OP._lists(fl); res['flips_u'][arm] = _flists(flu)
        res['flip_split'][arm] = dict(w=named(OP._lists(spw['s'])), u=named(_flists(spu['s'])))
        res['z'][arm] = zc
        res['e3'][arm] = dict(known_mismatch=info['known_mismatch'], gbp=info['gbp'], w=info['w'], w_label=info['w_label'], j_occ=info['j_occ'],
                              reused_agent_rows=reuse, seconds_model=t_model, seconds=time.time() - t0)
    res['zone_voxels'] = int(zone.sum()); res['ood_voxels'] = int(np.asarray(z['ood_mask']).sum())
    res['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    return res


def compare_controls(mine, rec_seed):
    """O3: my B4 / B6 blocks vs the H1 record (JSON round trip); -> (all equal, list of unequal 'arm:key')"""
    bad = []
    for arm in CONTROLS:
        theirs = rec_seed['R8'].get(arm)
        if theirs is None:
            bad.append(f'{arm}:missing'); continue
        a = _jr(mine['R8'][arm])
        for key in ('rows', 'zones', 'kinds', 'families', 'phantoms'):
            if a.get(key) != theirs.get(key):
                bad.append(f'{arm}:{key}')
    if _jr(mine['components']) != rec_seed.get('components'):
        bad.append('components')
    return not bad, bad


# ================================================================ jobs
def job_name(variant, panel, rooms, index):
    return OP.job_name(variant, panel, rooms, index)


def run_job(job):
    variant, split, panel, rooms, index, seeds, opts = job
    t0 = time.time()
    rec = dict(format=JOB_FORMAT, variant=variant, split=split, panel=panel, rooms=rooms, index=index, seeds={}, script_sha256=opts['script_sha256'],
               frozen=FROZEN, w=opts['w'], w_thr_sha256=opts.get('w_thr_sha256'), gbp_check=opts['gbp_check'])
    try:
        if sha(os.path.abspath(__file__)) != opts['script_sha256']:
            raise RuntimeError('g4_h1_e3l3.py changed since the run started')
        check_frozen()
        w_thr = parse_w_thr(opts['w_thr']) if opts.get('w_thr') else None
        M = load_models(opts['threads'], None if opts['w'] == 'learned' else float(opts['w']), w_thr)
        rec['ckpt_sha256'] = {ts: M['e3'][ts]['ckpt_sha256'] for ts in SEEDS}
        rec['thr_sha256'] = {ts: M['e3'][ts]['thr_sha256'] for ts in SEEDS}
        rec['w_used'] = {ts: M['e3'][ts]['w'] for ts in SEEDS}; rec['j_occ'] = {ts: M['e3'][ts]['jo'] for ts in SEEDS}
        hp = os.path.join(opts['h1_jobs'], job_name(variant, panel, rooms, index) + '.json')
        if not os.path.exists(hp):
            raise RuntimeError(f'H1 record {os.path.basename(hp)} missing')
        h1 = _jload(hp)
        if h1.get('script_sha256') != H1_SCRIPT_SHA256 or h1.get('failed') or h1.get('skipped'):
            raise RuntimeError('H1 record not a completed record of the H1 script 3a6959da')
        if (h1.get('variant'), h1.get('split'), h1.get('panel'), h1.get('rooms'), h1.get('index')) != (variant, split, panel, rooms, index):
            raise RuntimeError('H1 record identity != job')
        rec['h1_record_sha256'] = sha(hp)
        for seed in seeds:
            ts0 = time.time()
            hs = h1['seeds'].get(str(seed))
            if hs is None:
                raise RuntimeError(f'H1 record has no seed {seed}')
            path = os.path.join(opts['export'], hs['file'])
            fs = sha(path)
            if fs != hs['file_sha256']:
                raise RuntimeError(f"{hs['file']}: sha {fs[:12]} != H1 record {hs['file_sha256'][:12]}")
            r = evaluate_file(path, M, opts['gbp_check'])
            ok, bad = compare_controls(r, hs)
            r.update(file=hs['file'], file_sha256=fs, h1_controls_equal=ok, h1_controls_unequal=bad, seconds=time.time() - ts0)
            rec['seeds'][str(seed)] = r
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:  # noqa: BLE001 (recorded; aggregation -> INCOMPLETE)
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-3000:]
    return rec


def _run_and_save(job):
    rec = run_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    _jdump(rec, os.path.join(d, job_name(rec['variant'], rec['panel'], rec['rooms'], rec['index']) + '.json'))
    return rec


def _worker_init(threads):
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    G.setup_env()
    torch.set_num_threads(max(1, int(threads)))


# ================================================================ aggregation
def _summ_arm(X, arm):
    rows = [a for r in X for a in G._expand(r['R8'][arm]['rows'])]
    s = G.summarise_rows(rows)
    D = dict(**OP._row_keys(s), strata=s.get('strata'), zones=OP._zone_summary(OP._sum_tree([r['R8'][arm]['zones'] for r in X])),
             phantoms_per_house=float(np.mean([r['R8'][arm]['phantoms'][0] for r in X])),
             phantom_voxels_per_house=float(np.mean([r['R8'][arm]['phantoms'][1] for r in X])),
             blocks_per_house=float(np.mean([r['R8'][arm]['phantoms'][2] for r in X])),
             phantoms_nofloor_tall_per_house=float(np.mean([r['R8'][arm]['phantoms'][3] for r in X])), n_rows=len(rows))
    kin = OP._sum_tree([r['R8'][arm]['kinds'] for r in X])
    D['kinds'] = {kd: dict(f1=OP._f1(np.asarray(v, np.float64)), n_true=int(v[0] + v[2]), tp_fp_fn=np.asarray(v).tolist()) for kd, v in kin.items()}
    fam = {k: np.asarray(v) for k, v in OP._sum_tree([r['R8'][arm]['families'] for r in X]).items()}
    D['families'] = {fk: dict(f1=OP._f1(v.astype(np.float64)), n_true=int(v[0] + v[2]), tp_fp_fn=v.tolist()) for fk, v in fam.items()}
    return D


def _fsum(items):
    acc = {}
    for it in items:
        _addf(acc, 'x', it, 1.0)
    return acc.get('x', {})


def summarise(X):
    """X = per-(house, seed) eval dicts of one (variant, seed)"""
    D = dict(houses=len(X), R8={}, flips={}, flip_split={}, z={}, e3={})
    for arm in CONTROLS + ARMS:
        D['R8'][arm] = _summ_arm(X, arm)
    for arm in ARMS:
        D['flips'][arm] = {}
        for b6 in ('B6_A', 'B6_B'):
            acc = OP._sum_tree([r['flips'][arm][b6] for r in X]); accu = _fsum([r['flips_u'][arm][b6] for r in X])
            D['flips'][arm][b6] = {k: dict(n=int(v[0] + v[1]), prec=(v[0] / (v[0] + v[1])) if v[0] + v[1] else None, n_zone=int(v[2] + v[3]),
                                           prec_zone=(v[2] / (v[2] + v[3])) if v[2] + v[3] else None,
                                           n_vox=float(accu[k][0] + accu[k][1]), true_vox=float(accu[k][0])) for k, v in acc.items()}
        W_ = OP._sum_tree([r['flip_split'][arm]['w'] for r in X]); U_ = _fsum([r['flip_split'][arm]['u'] for r in X])
        sp = {}
        for part in ('fam', 'h'):
            sp[part] = {}
            for nm, w in W_[part].items():
                u = U_[part][nm]
                sp[part][nm] = dict(prec=(w[0] / (w[0] + w[1])) if w[0] + w[1] else None, true_vox=float(u[0]), false_vox=float(u[1]),
                                    false_cheb1_vox=(float(u[2]) if len(u) > 2 else None), true_w=int(w[0]), false_w=int(w[1]))
        D['flip_split'][arm] = sp
        zc = OP._sum_tree([r['z'][arm] for r in X]); zs = {}
        for nm, v in zc.items():
            v = np.asarray(v, np.float64)
            zs[nm] = [dict(counts=row.astype(np.int64).tolist(), share=(row / row.sum()).tolist() if row.sum() else None) for row in v]
        D['z'][arm] = zs
        D['e3'][arm] = dict(known_mismatch=int(sum(r['e3'][arm]['known_mismatch'] for r in X)),
                            gbp_cf_max=float(max(r['e3'][arm]['gbp']['cf_max'] for r in X)),
                            gbp_violations=int(sum(r['e3'][arm]['gbp']['violations'] for r in X)),
                            gbp_checked=all(r['e3'][arm]['gbp']['checked'] for r in X), gbp_pass=all(r['e3'][arm]['gbp']['pass'] for r in X),
                            w=sorted({r['e3'][arm]['w'] for r in X}), w_label=sorted({r['e3'][arm]['w_label'] for r in X}),
                            j_occ=sorted({r['e3'][arm]['j_occ'] for r in X}), reused_agent_rows=int(sum(r['e3'][arm]['reused_agent_rows'] for r in X)))
    return D


def _cell(V, ts, pairing, NET=None):
    """one (variant, seed, checkpoint) cell: E3L3 vs its paired B6 (D8-69); NET (report only, H1 summary) if given"""
    arm = f'E3L3_{ts}'
    b6 = f'B6_{PAIR[ts]}' if pairing == 'own' else max(('B6_A', 'B6_B'), key=lambda a: V['R8'][a]['furn_f1'])
    N, B = V['R8'][arm], V['R8'][b6]
    zN, zB = N['zones'].get('zone', {}), B['zones'].get('zone', {})
    fl = V['flips'][arm][b6]['le_-0']
    c = dict(b6=b6, e3_all=N['furn_f1'], b6_all=B['furn_f1'], d_all=N['furn_f1'] - B['furn_f1'],
             e3_zone=zN.get('f1'), b6_zone=zB.get('f1'),
             d_zone=(zN['f1'] - zB['f1']) if (zN.get('f1') is not None and zB.get('f1') is not None) else None,
             d_zone_tol1=(zN['f1_tol1'] - zB['f1_tol1']) if (zN.get('f1_tol1') is not None and zB.get('f1_tol1') is not None) else None,
             rest_fp_e3=N['zones']['rest']['fp'], rest_fp_b6=B['zones']['rest']['fp'], rest_n_true=N['zones']['rest'].get('n_true'),
             flip_prec=fl['prec'], flip_n=fl['n'], flip_n_vox=fl['n_vox'], flip_prec_zone=fl['prec_zone'], flip_n_zone=fl['n_zone'],
             ph_e3=N['phantoms_per_house'], ph_b6=B['phantoms_per_house'],
             ph_e3_nofloor_tall=N['phantoms_nofloor_tall_per_house'], ph_b6_nofloor_tall=B['phantoms_nofloor_tall_per_house'],
             free_e3=N['free_recall'], free_b6=B['free_recall'], d_free=N['free_recall'] - B['free_recall'],
             d_surface=N['surface_f1'] - B['surface_f1'], d_tol1_all=N['furn_f1_tol1'] - B['furn_f1_tol1'],
             families={fk: dict(e3=x['f1'], b6=B['families'].get(fk, {}).get('f1'), n_true=x['n_true'],
                                d=(x['f1'] - B['families'][fk]['f1']) if (x['f1'] is not None and B['families'].get(fk, {}).get('f1') is not None) else None)
                       for fk, x in N['families'].items()},
             flip_by_family=(V['flip_split'][arm]['fam'] if pairing == 'own' else None),
             flip_by_height=(V['flip_split'][arm]['h'] if pairing == 'own' else None))
    C = CRIT
    flip_ok = c['flip_prec'] is None or c['flip_prec'] >= C['flip_safe']
    c['safe'] = bool(c['d_zone'] is not None and c['d_zone'] >= C['zone_safe'] - 1e-12 and c['d_all'] >= C['overall_safe'] - 1e-12 and flip_ok
                     and c['ph_e3'] <= c['ph_b6'] + C['phantom_safe'] + 1e-12 and c['d_free'] >= C['free_safe'] - 1e-12)
    c['harm'] = bool((c['d_zone'] is not None and c['d_zone'] < C['zone_harm'] - 1e-12) or (c['flip_prec'] is not None and c['flip_prec'] < C['flip_harm'])
                     or c['ph_e3'] > c['ph_b6'] + C['phantom_harm'] + 1e-12)
    if NET is not None and pairing == 'own':                            # report only (D8-69): E3L3 - NET, NET from g4/h1/summary.json
        try:
            Nn = NET['R8'][f'NET_{ts}']; fn = NET['flips'][f'NET_{ts}'][b6]['le_-0']
            zn = Nn['zones'].get('zone', {}).get('f1')
            c['vs_net'] = dict(net_all=Nn['furn_f1'], d_all=N['furn_f1'] - Nn['furn_f1'], net_zone=zn,
                               d_zone=(zN['f1'] - zn) if (zn is not None and zN.get('f1') is not None) else None,
                               net_flip_prec=fn['prec'], d_flip_prec=(c['flip_prec'] - fn['prec']) if (c['flip_prec'] is not None and fn['prec'] is not None) else None,
                               net_flip_n_vox=fn.get('n_vox'), net_ph=Nn['phantoms_per_house'], d_ph=c['ph_e3'] - Nn['phantoms_per_house'],
                               net_free=Nn['free_recall'], d_free=N['free_recall'] - Nn['free_recall'],
                               families={fk: (x['f1'] - Nn['families'][fk]['f1']) if (x['f1'] is not None and Nn.get('families', {}).get(fk, {}).get('f1') is not None) else None
                                         for fk, x in N['families'].items()})
        except (KeyError, TypeError):
            c['vs_net'] = None
    return c


def _z_shift(V, Vc, ts):
    """O5 (report only): z share on true-free / true-occupied voxels near objects, variant zone vs ctrl objzone"""
    arm = f'E3L3_{ts}'
    try:
        zv = V['z'][arm]['zone']; zc = Vc['z'][arm]['objzone']
        out = {}
        for i, t in enumerate(('true_occ', 'true_free')):
            sv, sc = zv[i]['share'], zc[i]['share']
            out[t] = dict(variant=sv, ctrl=sc, d=[a - b for a, b in zip(sv, sc)] if (sv and sc) else None, n_variant=sum(zv[i]['counts']), n_ctrl=sum(zc[i]['counts']))
        return out
    except (KeyError, TypeError, IndexError):
        return None


def verdict(S, expected_houses, seeds, smoke=False, net_summary=None, finite_w=False):
    reasons = []
    if S['failed']:
        reasons.append(f"{len(S['failed'])} failed jobs: {S['failed'][:6]}")
    for k in ('script_sha256', 'ckpt_sha256', 'thr_sha256', 'frozen', 'w', 'gbp_check'):
        if len(S.get('versions', {}).get(k, [])) > 1:
            reasons.append(f'mixed {k} across job files')
    if S.get('foreign'):
        reasons.append(f"{len(S['foreign'])} job files of another split / panel: {S['foreign'][:6]}")
    if S.get('skipped'):
        reasons.append(f"{len(S['skipped'])} skipped: {S['skipped'][:6]}")
    if S.get('h1_mismatch'):
        reasons.append(f"{len(S['h1_mismatch'])} files: B4 / B6 != the H1 record (O3): {S['h1_mismatch'][:6]}")
    for v in VARIANTS:
        for sd in seeds:
            V = S['variants'].get(v, {}).get(str(sd))
            if V is None:
                reasons.append(f'{v}: noise seed {sd} missing'); continue
            if not smoke:
                miss = set(expected_houses) - set(V['house_list'])
                if miss:
                    reasons.append(f'{v}: seed {sd}: {len(miss)} of {len(expected_houses)} houses missing')
            for arm in ARMS:
                e = V['e3'][arm]
                if e['known_mismatch']:
                    reasons.append(f"{v} s{sd} {arm}: known (Lam_b > 0) != k > 0 on {e['known_mismatch']} voxels")
                if not e['gbp_pass']:
                    reasons.append(f"{v} s{sd} {arm}: GaBP check failed (cf_max {e['gbp_cf_max']}, violations {e['gbp_violations']})")
    out = dict(criteria=CRIT, reasons=reasons, cells={}, ctrl={}, variants={}, finite_w=bool(finite_w))
    if reasons and not smoke:
        out['verdict'] = 'INCOMPLETE'
        return out
    ctrl_ok = True
    for sd in seeds:
        V = S['variants'].get('ctrl', {}).get(str(sd))
        if V is None:
            ctrl_ok = False; continue
        for ts in SEEDS:
            ctl = max(CONTROLS, key=lambda a: V['R8'][a]['furn_f1'])
            g = V['R8'][f'E3L3_{ts}']['furn_f1'] - V['R8'][ctl]['furn_f1']
            ok = CRIT['ctrl_lo'] - 1e-12 <= g <= CRIT['ctrl_hi'] + 1e-12
            out['ctrl'][f'{ts}_s{sd}'] = dict(control=ctl, gain=g, ok=ok); ctrl_ok &= ok
    out['ctrl_ok'] = bool(ctrl_ok)
    for v in JUDGED:
        res = {}
        for pairing in ('own', 'max'):
            cells = {}
            for sd in seeds:
                V = S['variants'].get(v, {}).get(str(sd))
                if V is None:
                    continue
                Vc = S['variants'].get('ctrl', {}).get(str(sd))
                NET = (net_summary or {}).get('variants', {}).get(v, {}).get(str(sd)) if net_summary else None
                for ts in SEEDS:
                    c = _cell(V, ts, pairing, NET)
                    if Vc is not None:                                  # report only: ctrl, same checkpoint and seed (H1 S1)
                        cc = _cell(Vc, ts, pairing)
                        c['ctrl'] = dict(b6=cc['b6'], flip_prec=cc['flip_prec'], flip_n_vox=cc['flip_n_vox'], ph_e3=cc['ph_e3'], ph_b6=cc['ph_b6'])
                        if pairing == 'own':
                            c['z_shift'] = _z_shift(V, Vc, ts)
                    cells[f'{ts}_s{sd}'] = c
            if len(cells) < 2 * len(seeds):
                vv = 'INCOMPLETE'
            elif any(c['harm'] for c in cells.values()):
                vv = 'HARMFUL'
            elif all(c['safe'] for c in cells.values()):
                vv = 'SAFE'
            else:
                vv = 'AMBIGUOUS'
            res[pairing] = dict(verdict=vv, cells=cells)
        res['verdict'] = res['own']['verdict']; res['pairing_disagrees'] = res['own']['verdict'] != res['max']['verdict']
        out['variants'][v] = res
    if smoke:
        out['verdict'] = 'SMOKE (not a verdict)'
    elif finite_w:
        out['verdict'] = 'REPORT ONLY (finite w)'
        out['would_be'] = ({v: out['variants'][v]['verdict'] for v in JUDGED} if ctrl_ok else 'INCONCLUSIVE (ctrl self-check failed)')
    elif not ctrl_ok:
        out['verdict'] = 'INCONCLUSIVE (ctrl self-check failed)'
    else:
        out['verdict'] = {v: out['variants'][v]['verdict'] for v in JUDGED}
    return out


def aggregate(out_dir, smoke=False, seeds=NOISE_SEEDS, expected_houses=None):
    allrecs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]

    def foreign(r):
        return (r.get('split') == 'VALIDATION' or r.get('panel') == 'val') if smoke else (r.get('split') != 'VALIDATION' or r.get('panel') != 'val')
    recs = [r for r in allrecs if not foreign(r)]
    S = dict(variants={}, failed=[], skipped=[], h1_mismatch=[], n_jobs=len(recs),
             foreign=[job_name(r['variant'], r['panel'], r['rooms'], r['index']) for r in allrecs if foreign(r)], versions={})
    for k in ('script_sha256', 'ckpt_sha256', 'thr_sha256', 'frozen', 'w', 'gbp_check'):
        S['versions'][k] = sorted({json.dumps(r.get(k), sort_keys=True) for r in recs if not r.get('failed')})
    for r in recs:
        tag = job_name(r['variant'], r['panel'], r['rooms'], r['index'])
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append(tag)
        else:
            for sd, x in r['seeds'].items():
                if not x.get('h1_controls_equal'):
                    S['h1_mismatch'].append(f"{tag}_s{sd}:{x.get('h1_controls_unequal')}")
    ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
    S['timing'] = dict(job_seconds_median=float(np.median([r['seconds'] for r in ok])) if ok else None,
                       job_seconds_max=float(np.max([r['seconds'] for r in ok])) if ok else None,
                       file_seconds_median=float(np.median([x['seconds'] for r in ok for x in r['seeds'].values()])) if ok else None,
                       peak_mb_max=float(np.max([r.get('peak_mb') or 0 for r in ok])) if ok else None)
    present = sorted({int(s) for r in ok for s in r['seeds']})
    for v in VARIANTS:
        for sd in present:
            X = [r['seeds'][str(sd)] for r in ok if r['variant'] == v and str(sd) in r['seeds']]
            if not X:
                continue
            D = summarise(X)
            D['house_list'] = sorted(f"r{r['rooms']}_i{r['index']}" for r in ok if r['variant'] == v and str(sd) in r['seeds'])
            S['variants'].setdefault(v, {})[str(sd)] = D
    net = None
    if not smoke:
        S['net_summary'] = dict(path=H1_SUMMARY, sha256=sha(H1_SUMMARY) if os.path.exists(H1_SUMMARY) else None)
        if S['net_summary']['sha256'] == H1_SUMMARY_SHA256:
            net = _jload(H1_SUMMARY)
        else:
            S['net_summary']['note'] = 'sha != pinned: E3L3 - NET rows omitted'
    finite = any(json.loads(w) != 'learned' for w in S['versions']['w'])
    exp = expected_houses if expected_houses is not None else [f'r{r}_i{i}' for r, i in G.panel_specs('val')]
    S['verdict'] = verdict(S, exp, seeds if not smoke else present, smoke=smoke, net_summary=net, finite_w=finite)
    return S


def _f(x, d=4):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, (float, np.floating)) else str(x))


def _pp(x):
    return '   n/a' if x is None else f'{x * 100:+.2f}'


def report(S):
    L_ = [f"jobs {S['n_jobs']}, failed {len(S['failed'])}, skipped {len(S['skipped'])}, foreign {len(S.get('foreign', []))}, "
          f"H1-control mismatches {len(S.get('h1_mismatch', []))}; timing {S['timing']}"]
    L_.append(f"versions: w {S['versions'].get('w')} gbp_check {S['versions'].get('gbp_check')} script {[x[:14] for x in S['versions'].get('script_sha256', [])]}")
    for v, byseed in S['variants'].items():
        for sd, V in byseed.items():
            L_.append(f"\n== {v} seed {sd}: {V['houses']} houses")
            L_.append('  arm           R8furn  pooled  tol1    free    surf   | zone F1 tol1   n_true | rest fp | phantoms/house  S3-tall')
            for arm in CONTROLS + ARMS:
                a = V['R8'][arm]; zz = a['zones'].get('zone', {}); rr = a['zones'].get('rest', {})
                L_.append(f"  {arm:13s} {_f(a['furn_f1'])} {_f(a['furn_f1_pooled'])} {_f(a['furn_f1_tol1'])} {_f(a['free_recall'])} {_f(a['surface_f1'])} | "
                          f"{_f(zz.get('f1'))} {_f(zz.get('f1_tol1'))} {zz.get('n_true')} | {rr.get('fp')} | {_f(a['phantoms_per_house'], 2)} {_f(a['phantoms_nofloor_tall_per_house'], 2)}")
            for arm in ARMS:
                d = V['flips'][arm]
                L_.append(f"  flips {arm}: " + '; '.join(f"vs {b6} S<=0 n={x['le_-0']['n']} (voxels {x['le_-0']['n_vox']:.0f}) prec={_f(x['le_-0']['prec'], 3)}"
                                                       for b6, x in d.items()))
                sp = V['flip_split'][arm]
                L_.append(f"    by family (prec, true/false voxels): " + ', '.join(f"{k} {_f(x['prec'], 2)} {x['true_vox']:.0f}/{x['false_vox']:.0f}"
                                                                                for k, x in sp['fam'].items() if x['true_w'] + x['false_w']))
                L_.append(f"    by height: " + ', '.join(f"{k} {_f(x['prec'], 2)} {x['true_vox']:.0f}/{x['false_vox']:.0f}" for k, x in sp['h'].items()))
                zz = V['z'][arm]
                L_.append(f"    z -/0/+ share: zone occ {[_f(x, 3) for x in (zz.get('zone', [{}])[0].get('share') or [])]} free "
                          f"{[_f(x, 3) for x in (zz.get('zone', [{}, {}])[1].get('share') or [])]}; objzone occ "
                          f"{[_f(x, 3) for x in (zz.get('objzone', [{}])[0].get('share') or [])]} free {[_f(x, 3) for x in (zz.get('objzone', [{}, {}])[1].get('share') or [])]}")
                e = V['e3'][arm]
                L_.append(f"    w {e['w']} ({e['w_label']}) j_occ {e['j_occ']} known mismatch {e['known_mismatch']} GaBP cf_max {e['gbp_cf_max']:.2e} "
                          f"violations {e['gbp_violations']} checked {e['gbp_checked']}")
    vd = S['verdict']
    L_.append(f"\nVERDICT: {vd['verdict']}  reasons {vd['reasons']}")
    if vd.get('would_be'):
        L_.append(f"  (finite w, report only; would be {vd['would_be']})")
    L_.append(f"ctrl self-check [{CRIT['ctrl_lo'] * 100:+.1f}, {CRIT['ctrl_hi'] * 100:+.1f}] pp ok={vd.get('ctrl_ok')}: "
              + '; '.join(f"{k} {x['control']} {x['gain'] * 100:+.2f} pp {'ok' if x['ok'] else 'OUT'}" for k, x in vd.get('ctrl', {}).items()))
    for v, res in vd.get('variants', {}).items():
        L_.append(f"{v}: {res['verdict']} (max-pairing {res['max']['verdict']})")
        for k, c in res['own']['cells'].items():
            L_.append(f"   {k} vs {c['b6']}: zone {_pp(c['d_zone'])} pp, all {_pp(c['d_all'])} pp, flip {_f(c['flip_prec'], 3)} (voxels {c['flip_n_vox']:.0f}), "
                      f"phantoms {c['ph_e3']:.2f} vs {c['ph_b6']:.2f}, free {_pp(c['d_free'])} pp -> {'SAFE' if c['safe'] else ('HARM' if c['harm'] else 'amb')}")
            vn = c.get('vs_net')
            if vn:
                L_.append(f"      report only E3L3 - NET: zone {_pp(vn['d_zone'])} pp, all {_pp(vn['d_all'])} pp, flip {_f(vn['d_flip_prec'], 3)} "
                          f"(NET {_f(vn['net_flip_prec'], 3)}), phantoms {vn['d_ph']:+.2f}, free {_pp(vn['d_free'])} pp")
            cc = c.get('ctrl')
            if cc:
                L_.append(f"      report only ctrl same ckpt/seed: flip {_f(cc['flip_prec'], 3)}, phantoms {_f(cc['ph_e3'], 2)} vs {_f(cc['ph_b6'], 2)}")
            L_.append('      report only zone dF1 by family vs B6: ' + ', '.join(f"{fk} {_pp(x['d'])} ({x['n_true']})" for fk, x in sorted(c['families'].items()) if x['d'] is not None))
            zs = c.get('z_shift')
            if zs:
                L_.append('      report only z share shift vs ctrl objzone (true free voxels -/0/+): ' + str([_f(x, 3) for x in (zs['true_free']['d'] or [])])
                          + ' (true occupied): ' + str([_f(x, 3) for x in (zs['true_occ']['d'] or [])]))
    return '\n'.join(L_)


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'aggregate'):
        p = sub.add_parser(name)
        p.add_argument('--out', default=''); p.add_argument('--smoke', action='store_true'); p.add_argument('--seeds', default='')
        if name == 'run':
            p.add_argument('--formal', action='store_true'); p.add_argument('--export', default=EXPORT_DIR); p.add_argument('--h1-jobs', default=H1_JOBS)
            p.add_argument('--workers', type=int, default=8); p.add_argument('--threads', type=int, default=1)
            p.add_argument('--variants', default=','.join(VARIANTS)); p.add_argument('--w', default='learned')
            p.add_argument('--w-thr', default=''); p.add_argument('--no-gbp-check', action='store_true')
    a = ap.parse_args(argv)
    G.setup_env()
    if a.cmd == 'aggregate':
        seeds = tuple(int(s) for s in a.seeds.split(',')) if a.seeds else NOISE_SEEDS
        S = aggregate(a.out, smoke=a.smoke, seeds=seeds)
        _jdump(S, os.path.join(a.out, 'summary.json'))
        txt = report(S); open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
        return S
    assert a.workers <= 10, 'at most 10 processes on this machine'
    assert a.formal != a.smoke, 'exactly one of --formal (VAL) / --smoke (TRAIN)'
    if a.w != 'learned':
        float(a.w)
        assert a.w_thr and os.path.exists(a.w_thr), '--w needs --w-thr (O9)'
    else:
        assert not a.w_thr, '--w-thr only with a finite --w'
    h1_jobs = os.path.abspath(a.h1_jobs)
    if a.formal:
        assert os.path.normcase(h1_jobs) == os.path.normcase(os.path.abspath(H1_JOBS)), '--formal reads the H1 records of g4/h1/jobs'
        assert a.variants == ','.join(VARIANTS) and not a.seeds, '--formal: all four variants, noise seeds 1, 2'
        split, panel, seeds = 'VALIDATION', 'val', NOISE_SEEDS
        plan = [(v, r, i) for v in VARIANTS for r, i in G.panel_specs('val')]
        for v, r, i in plan:
            assert os.path.exists(os.path.join(h1_jobs, job_name(v, panel, r, i) + '.json')), f'H1 record {v} r{r}_i{i} missing'
    else:
        split, panel = 'TRAIN', 'train'
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(h1_jobs, '*.json')))]
        recs = [r for r in recs if r.get('variant') in a.variants.split(',')]
        assert recs and all(r.get('split') == 'TRAIN' and r.get('panel') == 'train' for r in recs), 'smoke: TRAIN H1 records only (never VAL)'
        plan = sorted({(r['variant'], int(r['rooms']), int(r['index'])) for r in recs})
        seeds = tuple(int(s) for s in a.seeds.split(',')) if a.seeds else tuple(sorted({int(s) for r in recs for s in r['seeds']}))
    out = a.out or (OUT_DIR if a.formal else SMOKE_DIR)
    if a.w != 'learned' and not a.out:
        out = out + f'_w{a.w}'
    os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
    check_frozen()
    opts = dict(out=out, export=os.path.abspath(a.export), h1_jobs=h1_jobs, threads=a.threads, script_sha256=sha(os.path.abspath(__file__)),
                w=a.w, w_thr=(os.path.abspath(a.w_thr) if a.w_thr else None), w_thr_sha256=(sha(a.w_thr) if a.w_thr else None),
                gbp_check=not a.no_gbp_check, split=split, seeds=list(seeds))
    _jdump(dict(opts=opts, plan=plan, frozen=FROZEN, argv=sys.argv, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), os.path.join(out, 'run.json'))
    jobs = []
    for v, r, i in plan:
        p = os.path.join(out, 'jobs', job_name(v, panel, r, i) + '.json')
        if os.path.exists(p):
            x = _jload(p)
            if not x.get('failed') and x.get('script_sha256') == opts['script_sha256'] and x.get('w') == a.w:
                continue
        jobs.append((v, split, panel, r, i, seeds, opts))
    print(f'{len(jobs)} jobs ({len(plan) - len(jobs)} done), {a.workers} workers x {a.threads} threads, w {a.w}, gbp_check {opts["gbp_check"]} -> {out}', flush=True)
    t0 = time.time()
    if a.workers <= 1:
        _worker_init(a.threads)
        it = ((k, _run_and_save(j)) for k, j in enumerate(jobs, 1))
    else:
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(a.workers, initializer=_worker_init, initargs=(a.threads,), maxtasksperchild=4)
        it = enumerate(pool.imap_unordered(_run_and_save, jobs), 1)
    for k, rec in it:
        st = 'FAILED ' + rec.get('error', '') if rec.get('failed') else 'ok'
        print(f"[{k}/{len(jobs)}] {job_name(rec['variant'], rec['panel'], rec['rooms'], rec['index'])} {rec.get('seconds', 0):.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB {st}  elapsed {time.time() - t0:.0f}s", flush=True)
    if a.workers > 1:
        pool.close(); pool.join()
    exp = [f'r{r}_i{i}' for r, i in sorted({(r, i) for _, r, i in plan})]
    S = aggregate(out, smoke=a.smoke, seeds=seeds, expected_houses=exp)
    _jdump(S, os.path.join(out, 'summary.json'))
    txt = report(S); open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
    return S


if __name__ == '__main__':
    main()
