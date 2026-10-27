"""g4_island_compose.py (Claude, 2026-10-27): 批次 G4-E6「按岛组合的位姿 stalk，接在冻结的 NET 读出前面」。
预注册：claude_handoff/OVERNIGHT_REPORT_20261021.md D8-51（背景 D8-45 / D8-49 / D8-50；原型 = R-A 探针
claude_handoff/analysis_20261027/review_E3R/RA/ra_regisl.py、ra_regisl_al.py、ra_regnf.py，RA_accuracy.md §2.2 / §5.1）。
纯 CPU（CUDA_VISIBLE_DEVICES= ），0 个租约。只 import 冻结代码，从不修改：
  g4_registration.py (E4, sha e9574c7e) : eval_arm 的流程、register_pair / evidence / planes_tern、compose / mean_transform / about /
                                          frame_matrix / _integrate / islands_K；
  g4_noise_panels.py (G4-E1, sha 2313d75b): 场景重建、噪声（公共随机数）、V3 积分、森林求和 fuse('sum')、House / Target / score；
  g4_e3r.py (E3-R, sha 1d65d9a8)         : Evidence、relay_sums、stalk、net_logits、load_ckpt、net_readout、linear_readout、scored。
边估计直接读 E4 冻结的 job 文件（claude_handoff/analysis_20261027/g4/e4/jobs/*.json），不重跑匹配器。

D8-51（约束本代码的部分，照录）：
  单变量：放进求和之前的每机 SE(2) 修正 K_a。对照 = 不修正（noreg），NET 读出。
  臂 G4-E6：1. 岛 = 被 E4 接受（corrected）的树边连成的连通块；2. 岛内从锚点（最小编号）沿接受边组合 E4 的边估计，得到 C_a；
    3. 每岛的规范 H_isl = mean_transform(成员的 C_a)，枢轴取成员**相信位置**的均值（不用真位置），K_a = H_isl^-1 · C_a；
    4. 被拒的边不传任何修正。
  固定：V3 积分，森林求和；E1c 表 342101 配半集 A、342202 配半集 B；冻结的 NET 检查点与阈值 j；计分 G.score，与 E3-R 同口径。
  判决行：E4 的整体对齐行（每个臂在自己融合后的 B4 信念上用 E4 匹配器对真值求一个整体 SE(2) G，NET、B6、B4 共用这个 G 重新积分后
    计分，流程同 E4 eval_arm）。原样行与真值规范行只报告。
  面板与格：VAL cam 24 房 × 噪声种子 1、2 × 读出 NET_342101 / NET_342202 = 4 格；fresh 同样 4 格，共同判决。
  判据：杠杆（全部满足）：VAL 4 格 NET(G4-E6) − NET(noreg) ≥ +3 pp；fresh 4 格均值 ≥ +3 pp 且没有一格 < +1 pp；
    cam_nf 上 NET(G4-E6) ≥ NET(noreg) − 1.0 pp（VAL 4 格）；cam 档空闲召回 ≥ −0.5 pp；未知→已知 FPR 只报告；
    每格「伤害房」（该房 NET 降超过 2 pp）≤ 3 / 24。
    不够：VAL 任一格 < +1 pp，或 cam_nf 保护不过。其余 ambiguous。
  只报告：B6、B4 的同一对比；原规则 E4 + NET；ceiling_islands + NET；每房岛数及其与 E4 可观测岛的一致率；规范后的残差 rms；harsh 档
    （E4 的 harsh 边估计）；按 k 与高度分层；位姿层字节 = E4 每条树边交换的 surfel 个数 × 6 B；原样行与对齐行的符号分歧。

预注册没写死、在这里写定的操作化（运行前写明）：
  O1 相信位置：智能体 a 的相信位置 = 它自己 3 个位姿的传感器原点经它自己的机间误差（to_public）之后、在它相信的公共帧里的 xy 均值，
     即它积分时用的 pub 扫描的原点 o 的均值（believed_positions）。只用本机数据；不读 positions_m、真机间误差或任何真值。
  O2 岛：每个无线分量内，job 中 corrected = True 的树边连成的连通块；没有接受边的智能体自成一岛。组合方向与 E4 compose 一致：边
     (i = 父, j = 子) 的估计 Delta 把 j 的帧映到 i 的帧，C_j = C_i · Delta，反向 C_i = C_j · Delta^-1。G4-E6 的 K 只读 job 边记录里的
     i、j、corrected、est 四个字段（e6_corrections 的白名单），不读 true / overlap_true / err_* / failed / observability。
  O3 参数往返：边估计与 E4 原规则的 K 都从 job 的 (yaw 度, tx, ty) 用 se2(radians(yaw), tx, ty) 重建（与 R-A 探针相同）。
  O4 判决行 = E4 eval_arm 的步骤（逐分量：B4 融合三态信念与 R8 真值各取 wall-column 平面 → register_pair(真值, 信念) 得 G_c；
     G_c = I 时沿用原图，否则每机用 G_c · K_a 重新 V3 积分其 pub 扫描），然后 B4 / B6_A / B6_B / NET_342101 / NET_342202 全部在同一组
     对齐后的地图上读出。B4 用 G.fuse('sum')（E1 / E4），B6 / NET 的 (S, k) 用 E3-R 的 relay_sums（中继 + 与直接和相等的断言）。
     B6、NET 的已知掩码断言与 B4 逐位相同。
  O5 计分：E3-R 的 scored（G.score strata=True，另加按噪声重数 k 的分层），每分量一行 [row, |分量|]，汇总时按智能体数展开后取均值
     （G.summarise_rows(G._expand(...))），即 E3-R / E4 的「分量展开均值」。
  O6 读出：B4 = sum L > 0；B6_A / B6_B = E1c fit.json（sha 8dd8b195）半集 A / B 的 cam_nf 表与 theta；NET_342101 = 检查点
     s342101/final.pt（sha 1b035fd1）+ 其 threshold.json 的 j，证据表断言等于半集 A；NET_342202 = s342202（sha a58b2b05）+ 半集 B。
  O7 格 = (噪声种子 1 / 2) × (NET_342101 / NET_342202)；差 = 判决行（cam、aligned）上 NET(e6) − NET(noreg) 的分量展开均值之差，pp。
  O8 空闲召回保护（NET 的 free_recall，判决行，e6 − noreg ≥ −0.5 pp）与伤害房（≤ 3 / 24）只在 VAL 4 格上判；fresh 只判幅度（4 格均值
     ≥ +3 pp、每格 ≥ +1 pp），fresh 的保护项读数只报告。cam_nf 保护按预注册只在 VAL 4 格判。
     （审查 S1，协调者 2026-10-27 决定：维持上述判法；fresh 4 格的空闲召回差、伤害房数与名单、cam_nf 差写进判决块
     verdict.fresh_guards_report_only 并列报告，标"只报告"，另给"若判则是否通过"的布尔值，不进判决。）
  O9 伤害房：该房（该任务）在判决行上 NET 家具 F1 的房均值（该房分量行按智能体展开后的均值，F1 无定义的行不计）
     e6 − noreg < −2 pp。分母 = 该格任务数（24）。
  O10 cam_nf 保护：cam_nf 档上用 E4 的 cam_nf 边估计按同一规则组合（相信位置 = cam_nf 的扫描原点），判决行同 O4。
  O11 真值规范行（只报告）：每分量 G_c = mean_transform({K_a · A_a}, 分量真位置均值)^-1，K_a ← G_c · K_a（A_a = 真机间误差）；
     规范后的残差 rms = 该行上每机 E_a = K_a · A_a 的偏航（°）与该机中心位移（cm）的 rms（真值只用于这一行与这一读数）。
  O12 ceiling_islands（只报告，只在 cam；另见 O24）：读 E4 cam job 的 arms.ceiling_islands.K；缺失时按 g4_registration.islands_K 重算（用真值），
     并在记录里标 ceiling_source = 'recomputed_with_truth'。
  O13 truepose（只用于 B4 / NET 收回等只报告读数）= cam_nf 档 noreg 臂（同一组地图与目标；断言 cam k0 sha = cam_nf pub sha）。
  O14 harsh（只报告）：只做 asis 与 aligned 两行（不做真值规范行）；臂 noreg / e6 / e4rule。
  O15 位姿层字节：每条树边 (n_src + n_tgt) × 6 B（双向交换；E4 job 的 surfel 计数），每房求和；另报单向 n_src × 6 B。
  O16 可观测岛一致率（cam）：接受边岛的划分与 E4 observability.islands 完全相同的任务比例；另报同分量智能体对上的一致比例。
  O17 一致性闸门：每档 noreg 臂的 B4 asis / aligned 行必须与 E4 job 的 noreg 行逐值相同（去掉本批新增的 k 分层键后比较），否则该任务
     记 e4_noreg_mismatch，汇总判 INCOMPLETE。E4 原规则、ceiling_islands 的 B4 行与 E4 的比较只报告（K 经参数往返可能差 1 ulp）。
  O18 复用：两个臂的 K 逐元素差 ≤ 1e-12 时复用读数（同 E4）。
  O19 INCOMPLETE：预期任务 = VAL / fresh 每房（G.panel_specs）× 种子 1、2 中 E4 job 存在且未跳过者（应为 96）；任一缺失、失败、
     地图 sha 不符（任务内直接抛错 → failed）、noreg 行不复现（O17）、或脚本 / 冻结代码 / 检查点 / 表 / 阈值的 sha 混用。
  O20 「原样行与对齐行的符号分歧」：每格比较 asis、aligned、truth_gauge 三行上 NET(e6) − NET(noreg) 的符号；也给出逐房分歧数。
  O21 读出复现 E3-R：凡有 R-A 独立复算（review_E3R/RA/out_recompute，VAL 48 个任务）的任务，cam / cam_nf 的 noreg 原样行
     NET_342101 / NET_342202 与 B6_A / B6_B 的家具 F1、空闲召回必须与之逐值相等，否则汇总判 INCOMPLETE；fresh 没有参照，不比较
     （fresh 的地图仍由任务内的地图 sha 断言与 O17 的 noreg B4 行比较保证）。
     审查 S2：O17 / O21 不等时，记录不等的分量数、主要指标的最大 |差|（一侧为 None 记 none_mismatch）与不等的键、两边的 G，
     以区分 1 体素的数值翻转与真错误；判 INCOMPLETE 的规则不变。
  O22 输入钉住（审查 S3）：E4 job 目录的清单（96 个文件按名排序的 "名 sha256" 行）的 sha256 = E4_MANIFEST_SHA256，运行前断言，
     每个任务再断言自己的 E4 job 文件 sha 等于清单条目；两个检查点 final.pt、两个 threshold.json、E1c fit.json 断言完整 sha256。
  O23 预期任务数（协调者编号 S5 = 审查 S7）：正式模式（VAL + fresh × 种子 1、2，未给 --jobs）断言预期任务数 == 96，汇总时不等即
     INCOMPLETE。
  O24 只报告臂 ceiling_acc_islands（审查 S4，只在 cam，用真值）：G4-E6 的接受边岛，岛内用真相对位姿（C_a = A_锚 A_a^-1），规范与
     G4-E6 相同（岛枢轴 = 成员相信位置均值）。e6 − 本行 = 岛内 E4 边估计误差的代价（E5-R 的目标量）；它不进判决，约多 10 % 时间。

资源（冒烟实测，2 进程并行）：每任务单线程约 380 s（机器空闲；cam_nf 110 s、cam 167 s、harsh 89 s），机器有其他批次时约 750 s；
加 O24 一臂后约多 10 %；峰值约 1.2 GB / 进程。96 个任务约 4.0–8.0 万 CPU 秒。
用法（总是 python -B；CPU；--workers ≤ 12）：
  python -B g4_island_compose.py --out DIR --jobs val_r1_i0_s1,val_r2_i3_s1 [--workers 2]      # 冒烟（只用 R-A 看过的 VAL 任务）
  python -B g4_island_compose.py --out DIR --panels val,fresh --seeds 1,2 --workers 12          # 正式
  python -B g4_island_compose.py --out DIR --aggregate-only [--panels val,fresh --seeds 1,2]
输出：DIR/jobs/*.json（每任务）、DIR/summary.json（汇总、格、判决）、DIR/run.json。
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True

import argparse, glob, hashlib, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3r as E  # noqa: E402            (frozen E3-R; imports g4_noise_panels and torch)
import g4_registration as R  # noqa: E402   (frozen E4)

os.environ['CUDA_VISIBLE_DEVICES'] = ''
G = E.G
W = G.W
torch = E.torch
ROOT = G.ROOT
A27 = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
E4_JOBS = os.path.join(A27, 'g4', 'e4', 'jobs')
INP = os.path.join(A27, 'review_E3R', 'inputs')
FIT = os.path.join(A27, 'g4', 'e1c', 'fit.json')
# S3 (review): full sha pins of every input (O22)
FIT_SHA256 = '8dd8b1952ead659ca49de1af0ce6b051943cf4aab1559704ab395acb152ea1f2'
CKPTS = (('342101', 's342101', 'A', '1b035fd1b6f0b01b136972505da834a384d61574e148f48e388e7fbbcdfb85cd',
          'c2dc9883968c9ce29fcbc95f2ac2728dc9ed29dbf8252c7971ef8b8fec3ef9bf'),
         ('342202', 's342202', 'B', 'a58b2b059d225ec95b438c56319a6945ac67ca779461db7d2cb059d99b5bb5c4',
          '775a0d4874e3d01bf32113395e2b34df74016181b33c3eb4c6e630410e8c14e3'))
E4_MANIFEST_SHA256 = '9d729e57fc2992ac47d6e703fda6fbfe57e1c08a59db771d66f25dd2a6c46702'   # sha256 of "name sha256" lines, sorted
N_EXPECTED = 96                                                                          # S5 / review S7 (O23)
FROZEN = dict(g4_registration='e9574c7eff5e1400b0bd901e0befc3abb65520d1612e8efbeade80307b5899fe',
              g4_noise_panels='2313d75bc3ee04515d87469ab9e89e49a3811e1125e39ca0bba839448cc8623f',
              g4_e3r='1d65d9a85f8ef0df4bcdb9f44feba9e1e6cae5f0057a6ec6b097d545ad883970')
JOB_FORMAT = 1
I3 = R.I3
CONDS = ('cam_nf', 'cam', 'harsh')
ARMS_BY_COND = dict(cam_nf=('noreg', 'e6', 'e4rule'), cam=('noreg', 'e6', 'e4rule', 'ceiling_islands', 'ceiling_acc_islands'),
                    harsh=('noreg', 'e6', 'e4rule'))
TG_CONDS = ('cam_nf', 'cam')                     # O14: harsh has no truth-gauge row
READOUTS = ('B4', 'B6_A', 'B6_B', 'NET_342101', 'NET_342202')
NETS = ('NET_342101', 'NET_342202')
ROWS = ('asis', 'aligned', 'truth_gauge')
JUDGED_COND, GUARD_COND, JUDGED_ROW = 'cam', 'cam_nf', 'aligned'
PANELS, SEEDS = ('val', 'fresh'), (1, 2)
# D8-51 criteria (pp)
LEVER_VAL_PP = 3.0
FRESH_MEAN_PP, FRESH_MIN_PP = 3.0, 1.0
NF_GUARD_PP = -1.0
FREE_GUARD_PP = -0.5
HARM_PP, HARM_MAX = -2.0, 3
INSUFF_PP = 1.0
SURFEL_BYTES = 6
SCORE_KEYS = ('furn_f1', 'furn_f1_tol1', 'surface_f1', 'var_occ_f1', 'free_recall', 'unknown_known_fpr', 'accuracy')


def _check(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def file_sha256(path):
    return G.file_sha256(path)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def e4_manifest(jobs_dir=E4_JOBS):
    """O22: {name: sha256} of every frozen E4 job file and the sha256 of the sorted 'name sha256' lines"""
    m = {os.path.basename(p)[:-5]: file_sha256(p) for p in sorted(glob.glob(os.path.join(jobs_dir, '*.json')))}
    txt = '\n'.join(f'{k} {m[k]}' for k in sorted(m))
    return m, hashlib.sha256(txt.encode()).hexdigest()


def pinned_e4_manifest():
    m, sha = e4_manifest()
    _check(sha == E4_MANIFEST_SHA256, f'E4 job directory manifest sha {sha} != pinned {E4_MANIFEST_SHA256} ({len(m)} files)')
    return m


def job_name(panel, rooms, index, seed):
    return f'{panel}_r{rooms}_i{index}_s{seed}'


def parse_name(name):
    panel, rest = name.split('_r', 1)
    rooms = int(rest.split('_i')[0]); index = int(rest.split('_i')[1].split('_s')[0]); seed = int(rest.split('_s')[1])
    return panel, rooms, index, seed


# ================================================================ SE(2) helpers and the composition rules (no truth below this line
# until truth_gauge_K / ceiling_K, which are labelled report-only)
def K_of(p):
    """(yaw deg, tx, ty) of the origin-based form -> 3 x 3 (as the R-A probe; O3)"""
    return R.se2(math.radians(p[0]), p[1], p[2])


def edge_view(e):
    """O2 whitelist: the only fields of an E4 edge record the G4-E6 rule may read"""
    return int(e['i']), int(e['j']), bool(e['corrected']), [float(x) for x in e['est']]


def accepted_islands(n, edges):
    """connected components of the accepted (corrected) tree edges; every agent belongs to exactly one island (singletons included)"""
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for e in edges:
        i, j, corr, _ = edge_view(e)
        if corr:
            par[find(i)] = find(j)
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(sorted(g) for g in groups.values())


def believed_positions(scans_pub):
    """O1: per agent the xy mean of its own believed public-frame sensor origins (the scans it integrates); no truth"""
    return [np.mean([np.asarray(s[0], np.float64)[:2] for s in scans], 0) for scans in scans_pub]


def e6_corrections(n, edges, bpos):
    """G4-E6 (D8-51): islands of accepted edges; inside an island compose E4's edge estimates from the anchor (smallest member) along
    accepted edges (C_j = C_i Delta_ij for parent i -> child j, reverse C_i = C_j Delta_ij^-1); per island the unit-prior gauge
    H_isl = mean_transform({C_a}, mean of the members' believed positions), K_a = H_isl^-1 C_a; rejected edges carry nothing.
    Inputs: the edge records (whitelisted fields only), the believed positions. Returns (K list, C dict, islands)."""
    nb = {a: [] for a in range(n)}
    for e in edges:
        i, j, corr, est = edge_view(e)
        if corr:
            M = K_of(est)
            nb[i].append((j, M)); nb[j].append((i, R.inv(M)))
    islands = accepted_islands(n, edges)
    C, K = {}, [None] * n
    for isl in islands:
        a0 = isl[0]; C[a0] = I3.copy(); q = [a0]
        while q:
            u = q.pop()
            for v, M in sorted(nb[u], key=lambda x: x[0]):
                if v not in C:
                    C[v] = C[u] @ M; q.append(v)
        _check(all(a in C for a in isl), 'island composition did not reach every member')
        piv = np.mean([np.asarray(bpos[a], np.float64)[:2] for a in isl], 0)
        Hinv = R.inv(R.mean_transform([C[a] for a in isl], piv))
        for a in isl:
            K[a] = Hinv @ C[a]
    return K, C, islands


def same_K(A, B, tol=1e-12):
    return all(np.allclose(x, y, rtol=0., atol=tol) for x, y in zip(A, B))


# ================================================================ report-only, truth-labelled helpers
def truth_gauge_K(Ks, MAs, positions, F):
    """O11 (report only): per component remove the mean residual SE(2) of K_a A_a about the component's true mean position; residual
    rms of the gauged row (yaw deg, displacement of each agent's centre cm)"""
    n = len(Ks); Kg = [None] * n; rs = []
    for cm in F['comps']:
        pv = np.mean([np.asarray(positions[a], np.float64)[:2] for a in cm], 0)
        Gc = R.inv(R.mean_transform([Ks[a] @ MAs[a] for a in cm], pv))
        for a in cm:
            Kg[a] = Gc @ Ks[a]; Ea = Kg[a] @ MAs[a]; cen = np.asarray(positions[a], np.float64)[:2]
            rs.append([math.degrees(R.yaw(Ea)), float(np.linalg.norm(R.apply2(Ea, cen) - cen) * 100)])
    rs = np.asarray(rs)
    return Kg, dict(yaw_rms_deg=float(np.sqrt((rs[:, 0] ** 2).mean())), disp_rms_cm=float(np.sqrt((rs[:, 1] ** 2).mean())),
                    per_agent=rs.tolist())


def acc_islands_true_K(islands, MAs, bpos):
    """O24 (review S4, report only, TRUTH): the accepted-edge islands of G4-E6 with the TRUE relative poses inside each island
    (C_a = A_anchor A_a^-1, as g4_registration.islands_K) and G4-E6's own gauge (island pivot = mean of the members' believed positions):
    e6 - this row isolates the error of E4's edge estimates inside the islands (the target quantity of E5-R)"""
    n = len(MAs); K = [None] * n
    for isl in islands:
        a0 = isl[0]
        C = {a: (I3.copy() if a == a0 else MAs[a0] @ R.inv(MAs[a])) for a in isl}
        piv = np.mean([np.asarray(bpos[a], np.float64)[:2] for a in isl], 0)
        Hinv = R.inv(R.mean_transform([C[a] for a in isl], piv))
        for a in isl:
            K[a] = Hinv @ C[a]
    return K


def ceiling_K(e4c, F, MAs, L_pub, cell, prm):
    """O12 (report only): E4's stored ceiling_islands K, else islands_K recomputed with the truth (labelled)"""
    arm = (e4c.get('arms') or {}).get('ceiling_islands')
    if arm and arm.get('K'):
        return [K_of(p) for p in arm['K']], 'e4_job'
    surf_m = [R.evidence(R.planes_logodds(L, cell, prm), cell, prm)['S']['m'] for L in L_pub]
    pivots = []
    for cm in F['comps']:
        P = np.concatenate([surf_m[a] for a in cm]) if any(len(surf_m[a]) for a in cm) else np.zeros((1, 2))
        pivots.append(P.mean(0))
    islands = [tuple(x) for x in e4c['observability']['islands']]
    return R.islands_K(F, islands, MAs, 'prior', pivots, surf_m), 'recomputed_with_truth'


def arm_Ks(e4c, cname, n, scans_pub, F, MAs=None, L_pub=None, cell=None, prm=None):
    """the corrections of every arm of one tier. noreg / e6 / e4rule use no truth (e6: edge whitelist + believed positions; e4rule:
    E4's stored K); ceiling_islands and ceiling_acc_islands (cam only) are truth-built and report-only: MAs is used ONLY for these two
    arms (ceiling_islands: only when the E4 job lacks its K)."""
    reg = e4c['registration']
    bpos = believed_positions(scans_pub)
    Ks = dict(noreg=[I3.copy() for _ in range(n)])
    Ks['e6'], C6, isl = e6_corrections(n, reg['edges'], bpos)
    Ks['e4rule'] = [K_of(p) for p in reg['K']]
    info = dict(islands=isl, n_islands=len(isl), C_e6=[R.params(C6[a]) for a in range(n)], believed_positions=[list(map(float, p)) for p in bpos])
    if cname == JUDGED_COND:
        Ks['ceiling_islands'], info['ceiling_source'] = ceiling_K(e4c, F, MAs, L_pub, cell, prm)
        Ks['ceiling_acc_islands'] = acc_islands_true_K(isl, MAs, bpos)
    return Ks, info


# ================================================================ read-outs (frozen E3-R pieces) and the arm evaluation (E4 eval_arm)
_RO = None


def load_readouts():
    """the frozen read-outs: B6_A / B6_B controls (E1c fit halves, cam_nf) and the two NET checkpoints with their own tables / j"""
    global _RO
    if _RO is not None:
        return _RO
    fit_sha = file_sha256(FIT)
    _check(fit_sha == FIT_SHA256, f'E1c fit sha {fit_sha} != pinned {FIT_SHA256}')
    b6 = {f'B6_{h}': E.Evidence.from_spec(f'c:{FIT}:{h}:cam_nf') for h in ('A', 'B')}
    nets, info = [], dict(fit_sha256=fit_sha, nets={})
    for sd, dn, half, ck_sha, thr_sha in CKPTS:
        p = os.path.join(INP, dn, 'final.pt'); tp = os.path.join(INP, dn, 'threshold.json')
        sha = file_sha256(p); tsha = file_sha256(tp)
        _check(sha == ck_sha, f'checkpoint {dn}/final.pt sha {sha} != pinned {ck_sha}')
        _check(tsha == thr_sha, f'{dn}/threshold.json sha {tsha} != pinned {thr_sha}')
        ck, model = E.load_ckpt(p, 'cpu')
        ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
        ref = b6[f'B6_{half}']
        _check(np.array_equal(ev.table, ref.table) and ev.theta == ref.theta, f'NET_{sd} evidence table != E1c half {half}')
        j = int(_jload(tp)['j'])
        nets.append((f'NET_{sd}', f'B6_{half}', model, j))
        info['nets'][f'NET_{sd}'] = dict(ckpt_sha256=sha, threshold_sha256=tsha, j=j, t=E.bin_edge(j), table=f'B6_{half}')
    _RO = dict(b6=b6, nets=nets, info=info)
    return _RO


def readouts(maps, F, R8T, H, RO, rstats, preds=None):
    """B4 (G.fuse 'sum' ternary), B6_A / B6_B (linear on the relayed sum of the half's table), NET_342101 / NET_342202 (frozen network on
    its own table's relayed (S, k), its own j). Rows = E3-R scored per component, [row, |component|]."""
    comps = F['comps']
    if preds is None:
        _, preds = G.fuse(maps, F, 'sum', rstats)
    sums = {tag: E.relay_sums(maps, F, ev.table, rstats)[0] for tag, ev in RO['b6'].items()}
    out = {r: [] for r in READOUTS}
    kcs = []
    for ci, cm in enumerate(comps):
        kc = sums['B6_A'][ci][1]; kcs.append(kc)
        _check(np.array_equal(sums['B6_B'][ci][1], kc), 'count sums differ between tables')
        _check(np.array_equal(preds[cm[0]] >= 0, kc > 0), 'B4 known mask != k > 0')
        out['B4'].append([E.scored(preds[cm[0]], R8T[ci], H, kc), len(cm)])
        for tag, ev in RO['b6'].items():
            out[tag].append([E.scored(E.linear_readout(sums[tag][ci][0], kc, ev.theta), R8T[ci], H, kc), len(cm)])
    for name, tag, model, j in RO['nets']:
        lg = E.net_logits(model, [x[0] for x in sums[tag]], [x[1] for x in sums[tag]], 'cpu', 1)
        for ci, cm in enumerate(comps):
            p = E.net_readout(lg[ci], kcs[ci], j)
            _check(np.array_equal(p >= 0, kcs[ci] > 0), 'network known mask != B4')
            out[name].append([E.scored(p, R8T[ci], H, kcs[ci]), len(cm)])
    return out


def eval_arm6(maps0, base_scans, Ks, F, R8T, T8, H, geo, prm, RO, rstats, tgK=None):
    """rows asis / aligned (E4 eval_arm: per component one G_c from register_pair(truth planes, fused B4 belief planes); every agent
    re-integrates its base scans under G_c K_a; every read-out on the same aligned maps) / truth_gauge (O11, report only)"""
    shape, cell, ds = geo
    n = len(maps0); comps = F['comps']; t0 = time.time()
    _, preds = G.fuse(maps0, F, 'sum', rstats)
    out = dict(asis=readouts(maps0, F, R8T, H, RO, rstats, preds), align=[])
    al = [None] * n
    for ci, cm in enumerate(comps):
        ev_t = R.evidence(R.planes_tern(T8[cm[0]], cell, prm), cell, prm)
        ev_b = R.evidence(R.planes_tern(preds[cm[0]], cell, prm), cell, prm)
        res = R.register_pair(ev_t, ev_b, cell, prm)
        Gc = res['M']
        out['align'].append(dict(G=R.params(Gc), E0=float(res['traj'][0]), E=float(res['E']), steps=len(res['traj']) - 1,
                                 coarse_accepted=bool(res['coarse_accepted']), icp_stop=res['icp_stop']))
        for a in cm:
            al[a] = maps0[a] if np.array_equal(Gc, I3) else R._integrate(shape, cell, ds, base_scans[a], Gc @ Ks[a])
    del preds
    out['aligned'] = readouts(al, F, R8T, H, RO, rstats)
    del al
    if tgK is not None:
        mt = [maps0[a] if np.array_equal(tgK[a], Ks[a]) else R._integrate(shape, cell, ds, base_scans[a], tgK[a]) for a in range(n)]
        out['truth_gauge'] = readouts(mt, F, R8T, H, RO, rstats)
        del mt
    out['seconds'] = time.time() - t0
    return out


def _strip_k(row):
    r = json.loads(json.dumps(row))
    if isinstance(r.get('strata'), dict):
        r['strata'].pop('k_furn', None); r['strata'].pop('k_free', None)
    return r


def rows_equal_e4(mine, theirs):
    """O17: my B4 rows (k strata removed) vs an E4 row list, value for value after a JSON round trip"""
    if theirs is None:
        return None
    a = [[_strip_k(r), k] for r, k in mine]
    return json.loads(json.dumps(a)) == json.loads(json.dumps(theirs))


DIAG_KEYS = ('furn_f1', 'free_recall', 'unknown_known_fpr', 'furn_f1_tol1', 'surface_f1')


def rows_diff(mine, theirs):
    """S2 (review): where two component-row lists differ -- the number of differing components, the max |difference| of the main
    metrics (None on one side only -> 'none_mismatch'), the differing keys; lets a 1-voxel numeric flip be told from a real error"""
    a = json.loads(json.dumps([[_strip_k(r), k] for r, k in mine])); b = json.loads(json.dumps(theirs))
    if len(a) != len(b):
        return dict(n_comp=[len(a), len(b)], structure='different component count')
    n_un, md, keys = 0, {k: 0.0 for k in DIAG_KEYS}, set()
    for (ra, ka), (rb, kb) in zip(a, b):
        if ra == rb and ka == kb:
            continue
        n_un += 1
        keys |= {k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k)}
        for k in DIAG_KEYS:
            x, y = ra.get(k), rb.get(k)
            if x is not None and y is not None:
                if md[k] != 'none_mismatch':
                    md[k] = max(md[k], abs(float(x) - float(y)))
            elif x != y:
                md[k] = 'none_mismatch'
    return dict(n_comp=len(a), n_unequal_comp=n_un, max_abs=md, keys_differing=sorted(keys))


def pair_agreement(isl_a, isl_b, comps):
    lab_a = {a: k for k, g in enumerate(isl_a) for a in g}; lab_b = {a: k for k, g in enumerate(isl_b) for a in g}
    tot = agree = 0
    for cm in comps:
        for x in range(len(cm)):
            for y in range(x + 1, len(cm)):
                a, b = cm[x], cm[y]; tot += 1
                agree += int((lab_a[a] == lab_a[b]) == (lab_b[a] == lab_b[b]))
    return (agree / tot) if tot else None, tot


# ================================================================ one (house, noise seed) task
def run_task(job):
    panel, rooms, index, seed, opts = job
    _check(__debug__, 'python -O strips the frozen code assertions: refused')
    _check(file_sha256(os.path.abspath(__file__)) == opts['script_sha256'], 'g4_island_compose.py changed since the run started')
    for nm, mod in (('g4_registration', R), ('g4_noise_panels', G), ('g4_e3r', E)):
        _check(file_sha256(mod.__file__) == FROZEN[nm], f'{nm}.py is not the frozen version')
    _check(index < 16, 'only VAL / fresh (scene_plan VALIDATION index < 16)')
    torch.set_num_threads(max(1, int(opts['threads'])))
    name = job_name(panel, rooms, index, seed); t_start = time.time(); timing = {}
    e4p = os.path.join(E4_JOBS, name + '.json')
    e4_sha = file_sha256(e4p)
    _check(e4_sha == opts['e4_sha256'][name], f'E4 job {name} sha {e4_sha} != the pinned manifest entry {opts["e4_sha256"][name]}')
    e4 = _jload(e4p)
    out = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, job=name, job_format=JOB_FORMAT,
               script_sha256=opts['script_sha256'], e4_job_sha256=e4_sha, e4_manifest_sha256=E4_MANIFEST_SHA256, conds={})
    if e4.get('skipped'):
        out['skipped'] = 'E4 job skipped: ' + str(e4['skipped']); return out
    _check(not e4.get('failed'), 'E4 job is a failed record')
    _check(e4['fingerprint']['script_sha256'] == FROZEN['g4_registration'] and e4['fingerprint']['g4_noise_panels_sha256'] == FROZEN['g4_noise_panels'],
           'E4 job written by another script version')
    prm = e4['fingerprint']['match']
    _check(prm == json.loads(json.dumps(R.MATCH)), 'E4 job matcher settings != frozen MATCH')
    RO = load_readouts(); out['readouts'] = RO['info']
    fx = W.make_fixture('VALIDATION', index, rooms)
    timing['fixture'] = time.time() - t_start
    occ = fx['occ']; cell = np.asarray(fx['cell'], np.float64); private = fx['private']; traj = fx['trajectory_m']
    positions = fx['positions_m']; n = private.shape[0]; shape = occ.shape; ds = 0.5 * float(cell.min()); geo = (shape, cell, ds)
    H = G.House(occ, fx['slab'], fx['obj'], cell); F = G.forest(fx['edges'], n)
    _check(W.array_hash(private) == e4['private_sha256'] and fx['world_id'] == e4['world_id'], 'fixture != E4 job')
    _check([list(c) for c in F['comps']] == e4['components'], 'components != E4 job')
    reached8 = np.asarray(fx['reached'][G.HOPS.index(8)], bool)
    for a in range(n):
        _check(set(np.nonzero(reached8[a])[0].tolist()) == set(F['comps'][F['comp_of'][a]]), 'reached8 != component')
    T8 = G.stream_targets(private, reached8)
    R8T = [G.Target(T8[cm[0]], H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    rays = W.fibonacci_sphere(W.BEAMS)
    t0 = time.time()
    traces = [[G.trace_pose(occ, cell, traj[j, a], rays) for j in range(traj.shape[0])] for a in range(n)]
    timing['trace'] = time.time() - t0
    rstats = G.RelayStats(); k0cache = {}
    out.update(agents=n, components=F['comps'], tree_edges=[list(e) for e in F['und']])
    for cname in opts['conds']:
        c = G.CONDITIONS[cname]; e4c = e4['conds'][cname]; tc = time.time()
        has_inter = c['inter_t'] > 0 or c['inter_yaw'] > 0; ck = G._ck(c); need_k0 = cname in ('cam_nf', 'cam')
        scans_pub, L_pub, MAs = [], [], []
        for a in range(n):
            frame = None
            if has_inter:
                z = G.frame_draws(seed, rooms, index, a)
                frame = (G.rotz(float(z[2] * c['inter_yaw'])), np.array([z[0] * c['inter_t'], z[1] * c['inter_t'], 0.]), np.asarray(positions[a], np.float64))
            MAs.append(R.frame_matrix(frame))
            scans = [G.measure(c, *traces[a][j], traj[j, a], G.pose_draws(seed, rooms, index, a, j, len(rays)), ds, rays)[0] for j in range(traj.shape[0])]
            pub = [G.to_public(s, frame) for s in scans]
            if (need_k0 or not has_inter) and (ck, a) not in k0cache:
                k0cache[(ck, a)] = G.integrate_agent(shape, cell, ds, scans, ('V3',))['V3']
            L_pub.append(G.integrate_agent(shape, cell, ds, pub, ('V3',))['V3'] if has_inter else k0cache[(ck, a)])
            scans_pub.append(pub)
        rec = dict(pub_sha256=W.array_hash(np.stack(L_pub)))
        _check(rec['pub_sha256'] == e4c['pub_sha256'], f'{cname}: rebuilt public-frame maps != E4 pub_sha256')
        if need_k0:
            rec['k0_sha256'] = W.array_hash(np.stack([k0cache[(ck, a)] for a in range(n)]))
            _check(rec['k0_sha256'] == e4c['k0_sha256'], f'{cname}: rebuilt K0 maps != E4 k0_sha256')
        rec['sha_match'] = True
        timing[f'{cname}_integrate'] = time.time() - tc
        Ks, info = arm_Ks(e4c, cname, n, scans_pub, F, MAs, L_pub, cell, prm)
        rec['e6'] = info
        arms, done = {}, []
        for arm in ARMS_BY_COND[cname]:
            K = Ks[arm]; ta = time.time()
            tgK, resid = truth_gauge_K(K, MAs, positions, F)
            prev = next((nm for nm, Kp in done if same_K(K, Kp)), None)
            if prev is not None:
                arms[arm] = dict(arms[prev], reused_from=prev, K=[R.params(x) for x in K], resid=resid)
            else:
                maps0 = [L_pub[a] if np.array_equal(K[a], I3) else R._integrate(shape, cell, ds, scans_pub[a], K[a]) for a in range(n)]
                res = eval_arm6(maps0, scans_pub, K, F, R8T, T8, H, geo, prm, RO, rstats, tgK if cname in opts['tg_conds'] else None)
                del maps0
                res.update(K=[R.params(x) for x in K], resid=resid)
                arms[arm] = res
            done.append((arm, K))
            timing[f'{cname}_{arm}'] = time.time() - ta
        if cname == JUDGED_COND and 'cam_nf' in out['conds']:
            _check(rec['k0_sha256'] == out['conds']['cam_nf']['pub_sha256'], 'cam truepose map != cam_nf map (O13)')
            arms['truepose'] = dict(ref='cam_nf.noreg')
        rec['arms'] = arms
        # ---- consistency with E4 (O17) and report-only diagnostics
        cmp = {}
        for mine, theirs in (('noreg', 'noreg'), ('e4rule', 'reg'), ('ceiling_islands', 'ceiling_islands')):
            if mine in arms and theirs in e4c['arms']:
                cmp[mine] = {row: rows_equal_e4(arms[mine][row]['B4'], e4c['arms'][theirs].get(row)) for row in ('asis', 'aligned')}
                cmp[mine]['G_equal'] = [list(x['G']) for x in arms[mine]['align']] == [list(x['G']) for x in e4c['arms'][theirs]['align']]
                for row in ('asis', 'aligned'):                         # S2: diagnostics when unequal
                    if cmp[mine][row] is False:
                        cmp[mine][f'{row}_diag'] = rows_diff(arms[mine][row]['B4'], e4c['arms'][theirs][row])
                if not cmp[mine]['G_equal']:
                    cmp[mine]['G_mine'] = [list(x['G']) for x in arms[mine]['align']]
                    cmp[mine]['G_e4'] = [list(x['G']) for x in e4c['arms'][theirs]['align']]
        rec['e4_rows_equal'] = cmp
        rec['e4_noreg_mismatch'] = not (cmp.get('noreg', {}).get('asis') and cmp.get('noreg', {}).get('aligned'))
        edges = e4c['registration']['edges']
        rec['pose_bytes'] = dict(per_edge_bidir=[(int(e['n_src']) + int(e['n_tgt'])) * SURFEL_BYTES for e in edges],
                                 per_edge_onedir=[int(e['n_src']) * SURFEL_BYTES for e in edges])
        rec['pose_bytes']['house_bidir'] = int(sum(rec['pose_bytes']['per_edge_bidir']))
        rec['pose_bytes']['house_onedir'] = int(sum(rec['pose_bytes']['per_edge_onedir']))
        rec['edges_accepted'] = int(sum(1 for e in edges if e['corrected'])); rec['edges_total'] = len(edges)
        if 'observability' in e4c:
            obs = [sorted(x) for x in e4c['observability']['islands']]
            rec['obs_islands'] = obs; rec['n_obs_islands'] = len(obs)
            rec['islands_equal_obs'] = sorted(info['islands']) == sorted(obs)
            rec['islands_pair_agreement'], rec['islands_pairs'] = pair_agreement(info['islands'], obs, F['comps'])
        timing[f'{cname}_total'] = time.time() - tc
        out['conds'][cname] = rec
        del scans_pub, L_pub
        later = {G._ck(G.CONDITIONS[c2]) for c2 in opts['conds'][opts['conds'].index(cname) + 1:]}
        for key in [k for k in k0cache if k[0] not in later]:
            del k0cache[key]
    out['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    timing['total'] = time.time() - t_start
    out['timing'] = timing; out['peak_mb'] = G._peak_mb()
    return out


def _job_path(out_dir, name):
    return os.path.join(out_dir, 'jobs', name + '.json')


def _worker_init():
    sys.dont_write_bytecode = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    G.setup_env()


def _run_and_save(job):
    panel, rooms, index, seed, opts = job
    name = job_name(panel, rooms, index, seed)
    try:
        res = run_task(job)
    except Exception:  # noqa: BLE001 (recorded; the aggregation turns it into INCOMPLETE)
        res = dict(panel=panel, house=f'r{rooms}_i{index}', rooms=rooms, index=index, seed=seed, job=name, failed=traceback.format_exc()[-2048:],
                   script_sha256=opts['script_sha256'])
    path = _job_path(opts['out'], name); tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh)
    os.replace(tmp, path)
    return name, (res.get('timing') or {}).get('total'), res.get('peak_mb'), res.get('skipped'), bool(res.get('failed'))


# ================================================================ aggregation and verdict (D8-51)
def expected_jobs(panels, seeds):
    """O19: every VAL / fresh house x seed whose E4 job exists and is not skipped"""
    exp, skipped, missing_e4 = [], [], []
    for panel in panels:
        for rooms, index in G.panel_specs(panel):
            for seed in seeds:
                nm = job_name(panel, rooms, index, seed); p = os.path.join(E4_JOBS, nm + '.json')
                if not os.path.exists(p):
                    missing_e4.append(nm)
                elif _jload(p).get('skipped'):
                    skipped.append(nm)
                else:
                    exp.append(nm)
    return exp, skipped, missing_e4


def _mean_rows(rows, keys=SCORE_KEYS, strata=False):
    s = G.summarise_rows(rows, keys=keys)
    if not strata:
        s.pop('strata', None)
    return s


def _house_mean(comp_rows, key='furn_f1'):
    xs = [r[key] for r in G._expand(comp_rows) if r.get(key) is not None]
    return float(np.mean(xs)) if xs else None


def _rows_of(job, cond, arm, row, ro):
    rec = job['conds'][cond]['arms']
    a = rec[arm]
    if 'ref' in a:                                    # truepose on cam = cam_nf noreg (O13)
        c2, a2 = a['ref'].split('.')
        a = job['conds'][c2]['arms'][a2]
        if row == 'truth_gauge' and 'truth_gauge' not in a:
            row = 'asis'
    if row not in a:
        return None
    return a[row][ro]


def _pp(x, y):
    return None if (x is None or y is None) else 100. * (x - y)


def cell_stats(J, cond, row, arm, ro, base='noreg'):
    """one cell: pooled means (agents expanded over houses) of arm vs base, deltas in pp, per-house deltas, harm count"""
    ra = [r for j in J for r in G._expand(_rows_of(j, cond, arm, row, ro))]
    rb = [r for j in J for r in G._expand(_rows_of(j, cond, base, row, ro))]
    ma, mb = _mean_rows(ra), _mean_rows(rb)
    per = {}
    for j in J:
        ha, hb = _house_mean(_rows_of(j, cond, arm, row, ro)), _house_mean(_rows_of(j, cond, base, row, ro))
        per[j['job']] = _pp(ha, hb)
    harm = sorted(k for k, v in per.items() if v is not None and v < HARM_PP)
    return dict(arm=ma['furn_f1'], base=mb['furn_f1'], delta_pp=_pp(ma['furn_f1'], mb['furn_f1']),
                free_arm=ma['free_recall'], free_base=mb['free_recall'], free_delta_pp=_pp(ma['free_recall'], mb['free_recall']),
                fpr_delta_pp=_pp(ma['unknown_known_fpr'], mb['unknown_known_fpr']), tol1_delta_pp=_pp(ma['furn_f1_tol1'], mb['furn_f1_tol1']),
                n_houses=len(J), n_agent_rows=len(ra), harm_houses=harm, n_harm=len(harm), per_house_pp=per)


def check_none_positions(job):
    """the arms of one tier must have furniture F1 undefined on the same component rows (as E4 review S6)"""
    for cname, rec in job['conds'].items():
        for row in ROWS:
            for ro in READOUTS:
                pats = set()
                for arm in rec['arms']:
                    rows = _rows_of(job, cname, arm, row, ro)
                    if rows is not None:
                        pats.add(tuple(k for k, (r, _) in enumerate(rows) if r.get('furn_f1') is None))
                _check(len(pats) <= 1, f"furn_f1 undefined on different rows across arms: {job['job']} {cname} {row} {ro}")


def verdict(cells, reasons):
    """D8-51, written before the run. cells[panel][seed][net] -> dict(delta_pp, nf_delta_pp, free_delta_pp, n_harm)"""
    if reasons:
        return dict(verdict='INCOMPLETE', reasons=reasons)
    val = [cells['val'][str(s)][nt] for s in SEEDS for nt in NETS]
    fr = [cells['fresh'][str(s)][nt] for s in SEEDS for nt in NETS]
    if any(c['delta_pp'] is None or c['nf_delta_pp'] is None or c['free_delta_pp'] is None for c in val) or any(c['delta_pp'] is None for c in fr):
        return dict(verdict='INCOMPLETE', reasons=['undefined cell'])
    fmean = float(np.mean([c['delta_pp'] for c in fr])); fmin = float(min(c['delta_pp'] for c in fr))
    checks = dict(val_all_ge_3=all(c['delta_pp'] >= LEVER_VAL_PP for c in val),
                  fresh_mean_ge_3=fmean >= FRESH_MEAN_PP, fresh_none_lt_1=fmin >= FRESH_MIN_PP,
                  nf_guard=all(c['nf_delta_pp'] >= NF_GUARD_PP for c in val),
                  free_guard=all(c['free_delta_pp'] >= FREE_GUARD_PP for c in val),
                  harm_le_3=all(c['n_harm'] <= HARM_MAX for c in val))
    insuff = any(c['delta_pp'] < INSUFF_PP for c in val) or not checks['nf_guard']
    if all(checks.values()):
        v = 'LEVER'
    elif insuff:
        v = 'INSUFFICIENT'
    else:
        v = 'AMBIGUOUS'
    return dict(verdict=v, checks=checks, fresh_mean_pp=fmean, fresh_min_pp=fmin,
                val_delta_pp=[c['delta_pp'] for c in val], val_nf_delta_pp=[c['nf_delta_pp'] for c in val],
                val_free_delta_pp=[c['free_delta_pp'] for c in val], val_n_harm=[c['n_harm'] for c in val],
                val_harm_houses=[c.get('harm_houses') for c in val],
                fresh_delta_pp=[c['delta_pp'] for c in fr],
                # S1 (coordinator, 2026-10-27): the fresh guard read-outs are listed next to the verdict, REPORT ONLY (not judged; O8)
                fresh_guards_report_only=dict(note='只报告 / report only: not part of the verdict (O8, S1)',
                                              free_delta_pp=[c.get('free_delta_pp') for c in fr], n_harm=[c.get('n_harm') for c in fr],
                                              harm_houses=[c.get('harm_houses') for c in fr], nf_delta_pp=[c.get('nf_delta_pp') for c in fr],
                                              free_guard_would_pass=all(c.get('free_delta_pp') is not None and c['free_delta_pp'] >= FREE_GUARD_PP for c in fr),
                                              harm_le_3_would_pass=all(c.get('n_harm') is not None and c['n_harm'] <= HARM_MAX for c in fr),
                                              nf_guard_would_pass=all(c.get('nf_delta_pp') is not None and c['nf_delta_pp'] >= NF_GUARD_PP for c in fr)),
                order='seed 1: NET_342101, NET_342202; seed 2: NET_342101, NET_342202')


def _sign(x):
    return None if x is None else (1 if x > 0 else (-1 if x < 0 else 0))


def e3r_reproduction(good):
    """report: the noreg as-is NET / B6 rows (cam, cam_nf) against R-A's independent recompute of E3-R (review_E3R/RA/out_recompute,
    VAL only; R8_net = NET of that checkpoint, R8_lin = its table's linear read-out = B6 of that half), furniture F1 and free recall"""
    d = os.path.join(A27, 'review_E3R', 'RA', 'out_recompute'); out = dict(equal=0, unequal=[], no_reference=0, diag={})

    def ad(x, y):
        return None if (x is None or y is None) else abs(float(x) - float(y))
    for nm, r in good.items():
        p = os.path.join(d, nm + '.json')
        if not os.path.exists(p):
            out['no_reference'] += 1; continue
        ref = _jload(p); ok = True; dg = dict(n_unequal_comp=0, max_abs_dF1=0.0, max_abs_dfree=0.0, where=[])
        for tn in ('cam', 'cam_nf'):
            if tn not in r['conds'] or tn not in ref['tiers']:
                continue
            a = r['conds'][tn]['arms']['noreg']['asis']
            for sd, b6 in (('342101', 'B6_A'), ('342202', 'B6_B')):
                T = ref['tiers'][tn][sd]
                for ro, key in ((f'NET_{sd}', 'R8_net'), (b6, 'R8_lin')):
                    if len(a[ro]) != len(T[key]):
                        ok = False; dg['where'].append(f'{tn}:{ro}:component count'); continue
                    for ci, (row, k) in enumerate(a[ro]):
                        e = row['furn_f1'] == T[key][ci][0] and row['free_recall'] == T[key][ci][2]
                        if not e:                                       # S2: how large is the difference?
                            ok = False; dg['n_unequal_comp'] += 1; dg['where'].append(f'{tn}:{ro}:{ci}')
                            for kk, x, y in (('max_abs_dF1', row['furn_f1'], T[key][ci][0]), ('max_abs_dfree', row['free_recall'], T[key][ci][2])):
                                v = ad(x, y)
                                dg[kk] = 'none_mismatch' if v is None else (dg[kk] if dg[kk] == 'none_mismatch' else max(dg[kk], v))
        if ok:
            out['equal'] += 1
        else:
            out['unequal'].append(nm); out['diag'][nm] = dg
    return out


def aggregate(out_dir, panels=PANELS, seeds=SEEDS, smoke=False, only=None):
    recs = {}
    for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json'))):
        r = _jload(p); recs[r.get('job') or os.path.basename(p)[:-5]] = r
    if only:
        exp = list(only); skipped_e4, missing_e4 = [], []
    else:
        exp, skipped_e4, missing_e4 = expected_jobs(panels, seeds)
    S = dict(expected=len(exp), skipped_in_e4=skipped_e4, missing_e4_job=missing_e4, failed=[], missing=[], skipped=[], e4_noreg_mismatch=[],
             script_sha256=sorted({r.get('script_sha256') for r in recs.values() if r.get('script_sha256')}),
             readouts=sorted({json.dumps(r.get('readouts'), sort_keys=True) for r in recs.values() if r.get('readouts')}), panels={})
    reasons = []
    good = {}
    for nm in exp:
        r = recs.get(nm)
        if r is None:
            S['missing'].append(nm)
        elif r.get('failed'):
            S['failed'].append(nm)
        elif r.get('skipped'):
            S['skipped'].append(nm)
        else:
            if any(r['conds'][c].get('e4_noreg_mismatch') for c in r['conds']):
                S['e4_noreg_mismatch'].append(nm)
                S.setdefault('e4_noreg_mismatch_diag', {})[nm] = {c: r['conds'][c]['e4_rows_equal'].get('noreg') for c in r['conds']
                                                                 if r['conds'][c].get('e4_noreg_mismatch')}
            if r.get('e4_manifest_sha256') != E4_MANIFEST_SHA256:
                S.setdefault('e4_unpinned', []).append(nm)
            try:
                check_none_positions(r)
            except RuntimeError as e:
                reasons.append(str(e)[:300])
            good[nm] = r
    S['e3r_reproduction'] = e3r_reproduction(good)
    if S['e3r_reproduction']['unequal']:
        reasons.append(f"{len(S['e3r_reproduction']['unequal'])} tasks: noreg read-outs != E3-R (R-A recompute) (O21)")
    if smoke:
        reasons.append('SMOKE: not a verdict (subset of tasks)')
    if not only and len(exp) != N_EXPECTED:
        reasons.append(f'expected {N_EXPECTED} tasks, the E4 directory gives {len(exp)} (O23)')
    try:
        pinned_e4_manifest()
    except RuntimeError as e:
        reasons.append(str(e)[:300])
    if missing_e4:
        reasons.append(f'{len(missing_e4)} E4 job files missing')
    for k in ('missing', 'failed', 'skipped', 'e4_noreg_mismatch', 'e4_unpinned'):
        if S.get(k):
            reasons.append(f'{len(S[k])} tasks {k}')
    if len(S['script_sha256']) > 1 or len(S['readouts']) > 1:
        reasons.append('mixed script / checkpoint / table versions in the task files')
    cells, report = {}, {}
    for panel in panels:
        cells[panel] = {}; report[panel] = {}
        for seed in seeds:
            J = [good[nm] for nm in exp if nm in good and parse_name(nm)[0] == panel and parse_name(nm)[3] == seed]
            if not J:
                continue
            cs, rp = {}, dict(houses=[j['job'] for j in J])
            # judged cells (+ guards)
            for nt in NETS:
                cj = cell_stats(J, JUDGED_COND, JUDGED_ROW, 'e6', nt)
                cg = cell_stats(J, GUARD_COND, JUDGED_ROW, 'e6', nt)
                cs[nt] = dict(delta_pp=cj['delta_pp'], net_e6=cj['arm'], net_noreg=cj['base'], free_delta_pp=cj['free_delta_pp'],
                              fpr_delta_pp=cj['fpr_delta_pp'], n_harm=cj['n_harm'], harm_houses=cj['harm_houses'], nf_delta_pp=cg['delta_pp'],
                              nf_e6=cg['arm'], nf_noreg=cg['base'], per_house_pp=cj['per_house_pp'], n_houses=cj['n_houses'])
            cells[panel][str(seed)] = cs
            # report-only table: every tier / row / arm / read-out pooled means and deltas vs noreg
            tab = {}
            for cond in CONDS:
                if not all(cond in j['conds'] for j in J):
                    continue
                tab[cond] = {}
                arms = list(ARMS_BY_COND[cond]) + (['truepose'] if cond == JUDGED_COND else [])
                for row in ROWS:
                    if _rows_of(J[0], cond, 'noreg', row, 'B4') is None:
                        continue
                    tab[cond][row] = {}
                    for arm in arms:
                        if any(_rows_of(j, cond, arm, row, 'B4') is None for j in J):
                            continue
                        tab[cond][row][arm] = {}
                        for ro in READOUTS:
                            m = _mean_rows([r for j in J for r in G._expand(_rows_of(j, cond, arm, row, ro))],
                                           strata=(row == JUDGED_ROW and arm in ('noreg', 'e6', 'truepose') and ro in ('B4',) + NETS))
                            if arm != 'noreg':
                                c = cell_stats(J, cond, row, arm, ro)
                                m.update(delta_pp=c['delta_pp'], free_delta_pp=c['free_delta_pp'], fpr_delta_pp=c['fpr_delta_pp'], n_harm=c['n_harm'],
                                         per_house_pp=c['per_house_pp'])
                            tab[cond][row][arm][ro] = m
                    # recovery of the gap to truepose (cam only, report)
                    if cond == JUDGED_COND and 'truepose' in tab[cond][row]:
                        for arm in ('e6', 'e4rule', 'ceiling_islands', 'ceiling_acc_islands'):
                            if arm in tab[cond][row]:
                                for ro in READOUTS:
                                    f0 = tab[cond][row]['noreg'][ro]['furn_f1']; f1 = tab[cond][row][arm][ro]['furn_f1']; ft = tab[cond][row]['truepose'][ro]['furn_f1']
                                    tab[cond][row][arm][ro]['recovery'] = ((f1 - f0) / (ft - f0)) if (ft - f0) > 0 else None
            rp['table'] = tab
            # sign disagreement between rows (O20), cam, e6 vs noreg and e4rule vs noreg, NET
            sd = {}
            for arm in ('e6', 'e4rule'):
                for nt in NETS:
                    d = {row: tab[JUDGED_COND][row][arm][nt]['delta_pp'] for row in ROWS if row in tab.get(JUDGED_COND, {}) and arm in tab[JUDGED_COND][row]}
                    sg = {row: _sign(v) for row, v in d.items()}
                    ph = {row: tab[JUDGED_COND][row][arm][nt]['per_house_pp'] for row in d}
                    house_dis = sum(1 for h in ph.get('aligned', {}) if 'asis' in ph and _sign(ph['asis'].get(h)) is not None
                                    and _sign(ph['aligned'][h]) is not None and _sign(ph['asis'][h]) != _sign(ph['aligned'][h]))
                    sd[f'{arm}:{nt}'] = dict(delta_pp=d, signs=sg, rows_disagree=len({v for v in sg.values() if v is not None}) > 1,
                                             houses_asis_vs_aligned_disagree=house_dis)
            rp['sign_disagreement'] = sd
            # islands, residuals, bytes, consistency with E4
            isl = {}
            for cond in CONDS:
                if not all(cond in j['conds'] for j in J):
                    continue
                cr = [j['conds'][cond] for j in J]
                d = dict(n_islands_mean=float(np.mean([c['e6']['n_islands'] for c in cr])),
                         edges_accepted=int(sum(c['edges_accepted'] for c in cr)), edges_total=int(sum(c['edges_total'] for c in cr)),
                         pose_bytes_house_bidir_median=float(np.median([c['pose_bytes']['house_bidir'] for c in cr])),
                         pose_bytes_house_bidir_mean=float(np.mean([c['pose_bytes']['house_bidir'] for c in cr])),
                         pose_bytes_house_onedir_median=float(np.median([c['pose_bytes']['house_onedir'] for c in cr])),
                         pose_bytes_edge_bidir_median=float(np.median([b for c in cr for b in c['pose_bytes']['per_edge_bidir']])),
                         resid={arm: dict(yaw_rms_deg=float(np.sqrt(np.mean([x[0] ** 2 for c in cr for x in c['arms'][arm]['resid']['per_agent']]))),
                                          disp_rms_cm=float(np.sqrt(np.mean([x[1] ** 2 for c in cr for x in c['arms'][arm]['resid']['per_agent']]))))
                                for arm in ARMS_BY_COND[cond]},
                         e4_rows_equal={arm: {row: sum(1 for c in cr if c['e4_rows_equal'].get(arm, {}).get(row)) for row in ('asis', 'aligned', 'G_equal')}
                                        for arm in ('noreg', 'e4rule', 'ceiling_islands') if arm in ARMS_BY_COND[cond]})
                if all('obs_islands' in c for c in cr):
                    d.update(n_obs_islands_mean=float(np.mean([c['n_obs_islands'] for c in cr])),
                             islands_equal_obs=int(sum(c['islands_equal_obs'] for c in cr)), n_tasks=len(cr),
                             islands_pair_agreement_mean=float(np.mean([c['islands_pair_agreement'] for c in cr if c['islands_pair_agreement'] is not None])))
                if cond == JUDGED_COND:
                    d['ceiling_source'] = sorted({c['e6'].get('ceiling_source') for c in cr})
                isl[cond] = d
            rp['diagnostics'] = isl
            rp['seconds'] = dict(median=float(np.median([j['timing']['total'] for j in J])), max=float(max(j['timing']['total'] for j in J)),
                                 sum=float(sum(j['timing']['total'] for j in J)))
            report[panel][str(seed)] = rp
    S['cells'] = cells
    S['report'] = report
    complete_panels = all(str(s) in cells.get(p, {}) for p in PANELS for s in SEEDS)
    if not complete_panels and not reasons:
        reasons.append('a judged panel / seed has no tasks')
    S['verdict'] = verdict(cells, reasons) if complete_panels else dict(verdict='INCOMPLETE', reasons=reasons)
    if smoke:
        S['verdict'] = dict(verdict='INCOMPLETE', reasons=reasons)
    S['criteria'] = dict(LEVER_VAL_PP=LEVER_VAL_PP, FRESH_MEAN_PP=FRESH_MEAN_PP, FRESH_MIN_PP=FRESH_MIN_PP, NF_GUARD_PP=NF_GUARD_PP,
                         FREE_GUARD_PP=FREE_GUARD_PP, HARM_PP=HARM_PP, HARM_MAX=HARM_MAX, INSUFF_PP=INSUFF_PP, judged_row=JUDGED_ROW,
                         judged_cond=JUDGED_COND, guard_cond=GUARD_COND, nets=list(NETS), seeds=list(SEEDS))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(S, fh, indent=1)
    return S


def _f(x, fmt):
    return 'None' if x is None else format(x, fmt)


def print_summary(S):
    print('verdict:', json.dumps({k: v for k, v in S['verdict'].items()}, default=str)[:1500])
    for panel, P in S['cells'].items():
        for seed, cs in P.items():
            for nt, c in cs.items():
                print(f"{panel} s{seed} {nt}: NET noreg {_f(c['net_noreg'], '.4f')} -> e6 {_f(c['net_e6'], '.4f')}  delta {_f(c['delta_pp'], '+.2f')} pp"
                      f" | cam_nf {_f(c['nf_delta_pp'], '+.2f')} pp | free {_f(c['free_delta_pp'], '+.2f')} pp | harm {c['n_harm']}/{c['n_houses']}")


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', default='', help='explicit task names (smoke), e.g. val_r1_i0_s1,val_r2_i3_s1')
    ap.add_argument('--panels', default='val,fresh'); ap.add_argument('--seeds', default='1,2')
    ap.add_argument('--conds', default=','.join(CONDS)); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--aggregate-only', action='store_true')
    a = ap.parse_args(argv)
    _check(__debug__, 'python -O refused')
    _check(a.workers <= 12, 'at most 12 processes')
    panels = tuple(x for x in a.panels.split(',') if x); seeds = tuple(int(x) for x in a.seeds.split(',') if x)
    _check(set(panels) <= set(PANELS), 'panels: val, fresh only (TEST is never generated)')
    os.makedirs(os.path.join(a.out, 'jobs'), exist_ok=True)
    only = [x for x in a.jobs.split(',') if x] or None
    if not a.aggregate_only:
        opts = dict(out=a.out, conds=[c for c in a.conds.split(',') if c], tg_conds=list(TG_CONDS), threads=a.threads,
                    script_sha256=file_sha256(os.path.abspath(__file__)), e4_sha256=pinned_e4_manifest())
        _check(opts['conds'][:2] == ['cam_nf', 'cam'] or opts['conds'] == ['cam_nf'], 'conds must start with cam_nf, cam (cam truepose = cam_nf map)')
        names = only if only else expected_jobs(panels, seeds)[0]
        if not only and (set(panels), set(seeds)) == (set(PANELS), set(SEEDS)):
            _check(len(names) == N_EXPECTED, f'expected {N_EXPECTED} tasks (VAL + fresh x seeds 1, 2), found {len(names)} (O23)')
        for nm in names:
            _check(nm.split('_')[0] in PANELS, f'{nm}: not a VAL / fresh task')
        todo = []
        for nm in names:
            p = _job_path(a.out, nm)
            if os.path.exists(p):
                r = _jload(p)
                if not r.get('failed') and r.get('script_sha256') == opts['script_sha256']:
                    continue
            panel, rooms, index, seed = parse_name(nm)
            todo.append((panel, rooms, index, seed, opts))
        run = dict(script='g4_island_compose.py', script_sha256=opts['script_sha256'], frozen=FROZEN, args=vars(a), n_todo=len(todo),
                   start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with open(os.path.join(a.out, 'run.json'), 'w') as fh:
            json.dump(run, fh, indent=1)
        t0 = time.time()
        if a.workers <= 1:
            for jb in todo:
                print(*_run_and_save(jb), f'{time.time() - t0:.0f}s', flush=True)
        else:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(a.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                for res in pool.imap_unordered(_run_and_save, todo):
                    print(*res, f'{time.time() - t0:.0f}s', flush=True)
        run.update(end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), wall_seconds=time.time() - t0)
        with open(os.path.join(a.out, 'run.json'), 'w') as fh:
            json.dump(run, fh, indent=1)
    S = aggregate(a.out, panels, seeds, smoke=bool(only), only=only)
    print_summary(S)
    return S


if __name__ == '__main__':
    main()
