"""g4_ood_panel.py (Claude, 2026-10-27): batch G4-H1 "留出家具形状面板，检验读出端学习先验 NET 的幻觉风险"，
预注册 claude_handoff/OVERNIGHT_REPORT_20261021.md D8-53（逐条照做；下面"操作化"一节写的是 D8-53 没写死、这里定下的部分）。
纯 CPU（CUDA_VISIBLE_DEVICES= ），0 个租约，冻结检查点与阈值，不重训。只 import 冻结的 g4_e3r.py（及其 import 的 g4_noise_panels.py）
与派生生成器 mapping_3d_storey_data_v420ood.py；不改任何已有文件。

世界与地图（其余照 E3-R）：
  4 个变体 ctrl / novel / scale / floating（定义见 mapping_3d_storey_data_v420ood.py 文件头）× VAL 24 房（scene_plan('VALIDATION', 16)
  里 index < 8，r1–r3，= G.panel_specs('val')）× 噪声种子 1、2，cam_nf 档，V3 积分器。导出 = E3-R 的 export 写法：E.house_ctx +
  E.tier_maps（同一组调用、同一套公共随机数），每 (变体, 房, 种子) 一个 npz（E3-R 的数组 occ / slab / obj / cell / private / edges /
  L_k0_cam_nf，另加计分分区数组 ood_mask / objvox_idx / objvox_off / obj_kinds / obj_family / obj_ood）。
  冻结模块的唯一替换（最小侵入）：每个任务开始时 E.W = mapping_3d_storey_data_v420ood.module_for(variant)；E.house_ctx 通过模块全局 W
  调 make_fixture，于是 E 的 house_ctx / tier_maps 原样在派生世界上运行。G.W 不被这条路径使用，不动。
读出（同一批地图、同一组和 S、k；已知掩码在所有臂之间逐位相同，断言）：
  B4（sum L > 0）、B6_A、B6_B（E1c 表 fit.json sha 8dd8b195 的半集 A / B，cam_nf，sum cq > theta）、
  NET_342101（读半集 A 的 S；检查点 sha 1b035fd1，冻结阈值 j）、NET_342202（半集 B；a58b2b05）。K0 另报（共享 j 与 K0_sep）。

指标（D8-53）：
  * R8 与 K0 家具 F1，精确与容差 1，两个区域：zone = G.dil6(ood_mask)（新物体 / 被缩放或离地的物体膨胀 1 格），rest = 其余；
    都在 E1 的计分范围（clean-known ∩ 家具区域 H.region）内，真值 = T.tr；
  * 空闲召回、表面 F1（G.score，E3-R 口径）；
  * 按 k（噪声重数 0..4, ≥5）与高度（[0, .3) / [.3, .9) / ≥ .9 m）分层（整体用 E.scored 的 strata；zone / rest 另算）；
  * 翻转精确率：NET 判占据、B6 判空闲、且 S ≤ 0（S = NET 读的那张表的融合和，单位 0.05 nat）的计分范围内体素中 T.tr 的比例；
  * 幻影块：NET 判占据的体素（k > 0 ∩ H.region）的 26 连通块中，3×3×3 邻域（Chebyshev ≤ 1 格）内没有任何生成器真值占据
    （occ == 1，含墙与楼板）的块数；每房 = 各连通分量的块数按分量大小加权（= 逐智能体信念的均值）。B6 用同一定义。
操作化（D8-53 未写死的部分）：
  P1 B6 配对：主判 = 每个 NET 配它自己读的那张表的线性读出（342101 ↔ B6_A，342202 ↔ B6_B）——同一 S、同一 k，唯一的变量是读出；
     敏感性行 = E3-R 的选法限于 B6（该 (变体, 种子) 里 R8 家具 F1 较高的那张），两种配对的判决不同时在报告里标出。
  P2 "新物体区 NET − B6"判在 R8 精确家具 F1 上，计数按连通分量大小加权后池化（= 分量展开，与 R-A 的 ra_kinds 相同）；
     "整体 NET − B6"用 E3-R 的口径（分量展开后逐行 furn_f1 的均值，G.summarise_rows 的 furn_f1）；空闲召回同为 E3-R 口径。
     K0、容差 1、rest 区、按类别的 zone F1 都只报告。
  P3 翻转精确率在分母为 0 时记为通过（没有逆证据翻转）并报告计数；阈值 S ≤ −2、−4 nat 只报告。
  P4 ctrl 自检 = E3-R 的增益定义：NET − max(B4, B6_A, B6_B)（整体 R8 家具，E3-R 口径），4 格都在 [+5.3, +8.6] pp 内，否则整批
     INCONCLUSIVE。ctrl 没有新物体，zone 为空，只做自检。
  P5 完整性：期望 = 4 变体 × 24 房 × 种子 {1, 2}。失败或缺少的任务 → INCOMPLETE。make_fixture 的确定性跳过（智能体放不下，
     ValueError；同一房两个种子一起跳过）照 E3-R 列出并从该变体剔除，不算缺任务（若出现，须协调者确认）。
审查补充（g4/code_review_g4_h1.md，LAUNCH OK；S1–S8 只加报告行与防呆，判据照 D8-53 原文不变）：
  S1 每个判决格并列 ctrl 同一检查点、同一种子的翻转精确率与每房幻影块；
  S2 翻转另报未加权体素数，并按最近生成器物体（EDT 索引）的家族、高度带与楼板上第 1 层拆分，另报错误翻转中离物体切比雪夫距离 1 的数；
  S3 幻影块另报一行：真值去掉地板楼板、且块伸到楼板以上第 2 层及以上（z 下标 ≥ 楼板顶层 + 2）的块数；判决仍用原定义；
  S4 --formal 拒绝 --plan 与不完整的 --houses；正式汇总只收 split = VALIDATION 且 panel = val 的任务（冒烟汇总不收 VAL 任务），
     其余列为 foreign 并判 INCOMPLETE；只把 v420 agent_positions 的三种 ValueError 记作跳过，正式判决中任何跳过都判 INCOMPLETE（P5，待协调者确认）；
  S5 worker 内重算本脚本、派生生成器、v420、g4_e3r、g4_noise_panels 的 sha 并与启动时比对；run.json 记录全部 sha；
  S6 判决变体里所有物体都带标志（ood_mask == obj），新物体区 = 全部家具膨胀 1 格，rest 区没有真值家具：rest F1 印 n/a，只报 rest FP；
  S7 报告写明 novel 新物体体素约 67 % 是轴对齐盒拼成的组合件（VAL 普查 169,431 / 252,163），并按格列出各家族的新物体区 ΔF1 与翻转精确率；
  S8 geom --png 拒绝写进仓库目录。
判据（novel、scale、floating 各判一次；2 检查点 × 2 噪声种子 = 4 格，D8-53 原文）：
  安全：4 格都满足 zone NET − B6 ≥ −1 pp，整体 NET − B6 ≥ 0，翻转精确率 ≥ .70，每房幻影块 NET ≤ B6 + 0.5，空闲召回 ≥ B6 − 0.5 pp；
  有害：任一格 zone NET − B6 < −1 pp，或翻转精确率 < .50，或每房幻影块 > B6 + 2；其余 AMBIGUOUS。

用法（python -B；一个任务 = 一个 (变体, 房)，内含两个噪声种子，导出后立即从 npz 读回计分）：
  run   --split VALIDATION --formal [--variants ctrl,novel,scale,floating] [--seeds 1,2] [--workers 8] [--threads 2] [--out D] [--export D]
  run   --split TRAIN --houses r1:9,r2:9 --plan ctrl:r1:9,novel:r2:9,... --seeds 203 --smoke --out D --export D     (冒烟，只用 TRAIN)
  aggregate --out D [--smoke]
  geom  --split VALIDATION --houses r1:0 [--census] [--fixture] [--png DIR] --out D         (只看几何，不计分)
VAL 是判决面板：run 在 VALIDATION 上必须带 --formal；冒烟不得在 VAL 上计分。
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, json, math, multiprocessing, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3r as E  # noqa: E402   (frozen; read only)
import torch  # noqa: E402
import mapping_3d_storey_data_v420ood as WO  # noqa: E402

G = E.G
ROOT = G.ROOT
AN = os.path.join(ROOT, 'claude_handoff', 'analysis_20261027')
INP = os.path.join(AN, 'review_E3R', 'inputs')
CKPTS = {'342101': os.path.join(INP, 's342101', 'final.pt'), '342202': os.path.join(INP, 's342202', 'final.pt')}
CKPT_SHA = {'342101': '1b035fd1', '342202': 'a58b2b05'}
PAIR = {'342101': 'A', '342202': 'B'}
FIT = os.path.join(AN, 'g4', 'e1c', 'fit.json')
FIT_SHA = '8dd8b195'
OUT_DIR = os.path.join(AN, 'g4', 'h1')
SMOKE_DIR = os.path.join(AN, 'g4', 'h1_smoke')
EXPORT_DIR = os.path.join(AN, 'g4', 'h1_export')
TIER = 'cam_nf'
NOISE_SEEDS = (1, 2)
VARIANTS = WO.VARIANTS
JUDGED = ('novel', 'scale', 'floating')
CRIT = dict(zone_safe=-0.01, overall_safe=0.0, flip_safe=0.70, phantom_safe=0.5, free_safe=-0.005,
            zone_harm=-0.01, flip_harm=0.50, phantom_harm=2.0, ctrl_lo=0.053, ctrl_hi=0.086)
FLIP_THR = (0.0, 2.0, 4.0)
CUBE = np.ones((3, 3, 3), bool)
NB = len(G.H_BANDS)
ARMS = ('B4', 'B6_A', 'B6_B', 'NET_342101', 'NET_342202')
AGENT_SKIPS = ('no free floor', 'not enough free floor with clearance', 'could not place agents')   # v420 agent_positions' ValueErrors
H_NAMES = ('h0-0.3m', 'h0.3-0.9m', 'h>=0.9m', 'layer1')               # S2 height split: G.H_BANDS + the first layer above the floor slab
NOTES = ('S6: in novel / scale / floating EVERY object carries the ood flag (ood_mask == obj), so the zone = all furniture dilated by one '
         'voxel and the rest holds no true furniture: rest F1 is n/a (empty set, not 0); only rest FP (false positives away from furniture) '
         'is informative. Zone and overall readouts differ only by those far false positives.',
         'S7: novel ~67 % of the new-object voxels are axis-aligned box composites (l_shape / step_shape; VAL census 169,431 / 252,163): '
         'O3 maps the big slots (wardrobe, fridge, counter, cabinets, bathtub) to them. The novel zone F1 is dominated by the least '
         'out-of-distribution parts; read the per-family zone dF1 and flip rows next to the verdict.',
         'S1-S3 rows (ctrl flips / phantoms of the same checkpoint and seed, unweighted flip voxels, flips by nearest-object family and '
         'height, phantoms without the floor slab reaching layer >= 2) are report only; the verdict is D8-53 verbatim.')
JOB_FORMAT = 2


def sha(path):
    return G.file_sha256(path)


def _jdump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


# ================================================================ models / tables (loaded once per worker)
_M = None


def load_models(threads=2):
    global _M
    if _M is None:
        torch.set_num_threads(max(1, int(threads)))
        assert sha(FIT).startswith(FIT_SHA), 'E1c fit.json sha mismatch'
        b6 = {h: E.Evidence.from_spec(f'c:{FIT}:{h}:cam_nf') for h in 'AB'}
        nets = {}
        for ts, p in CKPTS.items():
            assert sha(p).startswith(CKPT_SHA[ts]), f'checkpoint {ts} sha mismatch'
            ck, model = E.load_ckpt(p, 'cpu')
            thr = E._jload(os.path.join(os.path.dirname(p), 'threshold.json'))
            ev = E.Evidence.from_recipe(ck['recipe']['evidence'])
            assert str(ck['seed']) == ts
            assert np.array_equal(ev.table, b6[PAIR[ts]].table) and ev.theta == b6[PAIR[ts]].theta, f'NET_{ts} does not read B6_{PAIR[ts]}'
            nets[ts] = dict(model=model, ev=ev, j=int(thr['j']), j_sep=thr['k0'].get('j_sep'), theta_sep=thr['k0'].get('linear_theta_sep'),
                            thr_sha=sha(os.path.join(os.path.dirname(p), 'threshold.json')))
        _M = dict(b6=b6, nets=nets, L=E.Evidence.from_spec('L'))
    return _M


# ================================================================ scoring layers (only added, E3-R rows unchanged)
def zone_counts(p, T, H, kc, zones):
    """per zone: [tp, fp, fn, n_pred, pred_within1, n_true, true_within1, free_hit, free_n] (E1 scope: T.known & H.region), and the
    furniture tp / fp / fn by k (0..4, >= 5) and by height band"""
    pr = (p == 1) & T.known & H.region; tr = T.tr; dpr = G.dil6(pr)
    cls = np.clip(kc, 0, E.KMAX).astype(np.int64); fr = (p == 0) & T.free
    out = {}
    for name, Z in zones.items():
        a = pr & Z; b = tr & Z
        tpm, fpm, fnm = a & b, a & ~b, b & ~a
        c = [int(x.sum()) for x in (tpm, fpm, fnm, a, a & T.dil_tr, b, b & dpr, fr & Z, T.free & Z)]
        k = np.stack([np.bincount(cls[x], minlength=E.KMAX + 1) for x in (tpm, fpm, fnm)], 1)
        h = np.stack([G._band_sum(x, H.band_z, NB) for x in (tpm, fpm, fnm)], 1)
        out[name] = dict(c=np.asarray(c, np.int64), k=k.astype(np.int64), h=h.astype(np.int64))
    return out


def kind_counts(p, T, H, kzones):
    pr = (p == 1) & T.known & H.region; tr = T.tr
    return {kd: np.array([int((pr & Z & tr).sum()), int((pr & Z & ~tr).sum()), int((tr & Z & ~pr).sum())], np.int64) for kd, Z in kzones.items()}


def phantom_blocks(p, kc, H, near, near_nf=None, z_tall=2):
    """26-connected blocks of predicted occupancy (k > 0, furniture region) with no generator occupancy within one cell (3x3x3)
    -> (phantoms [the verdict definition], their voxels, all blocks, S3 row). S3 (review, report only): truth = occupancy WITHOUT the
    floor slab (near_nf), and the block reaches z index >= z_tall (= 2nd layer above the floor slab); catches floor-hugging blocks."""
    m = (p == 1) & (kc > 0) & H.region
    lab, nb = ndimage.label(m, structure=CUBE)
    if nb == 0:
        return 0, 0, 0, 0
    sizes = np.bincount(lab.ravel(), minlength=nb + 1)[1:]
    hit = np.bincount(lab[near & m], minlength=nb + 1)[1:]
    ph = hit == 0
    tall = 0
    if near_nf is not None:
        hit2 = np.bincount(lab[near_nf & m], minlength=nb + 1)[1:]
        zmax = np.full(nb + 1, -1, np.int64); np.maximum.at(zmax, lab[m], np.nonzero(m)[2]); zmax = zmax[1:]
        tall = int(((hit2 == 0) & (zmax >= z_tall)).sum())
    return int(ph.sum()), int(sizes[ph].sum()), int(nb), tall


def flip_split(pn, pb, S, T, H, near_fam, n_fam, cheb1, layer1):
    """S2 (report only): the S <= 0 flips (NET occupied, B6 free, scope) split by the family of the nearest generator object
    (EDT index) -> [n_fam + 1, 3] = (true, false, false at Chebyshev distance 1 from an object) with the last row = no object; and by
    height band (G.H_BANDS) + the first layer above the floor slab -> [NB + 1, 2] = (true, false)"""
    sel = (pn == 1) & (pb == 0) & T.known & H.region & (S <= 0)
    right, wrong = sel & T.tr, sel & ~T.tr
    def fam(m):
        f = near_fam[m]; f = np.where(f < 0, n_fam, f)
        return np.bincount(f, minlength=n_fam + 1).astype(np.int64)
    byf = np.stack([fam(right), fam(wrong), fam(wrong & cheb1)], 1)
    def hb(m):
        b = np.concatenate([G._band_sum(m, H.band_z, NB), [int(m[:, :, layer1].sum())]]); return b.astype(np.int64)
    return dict(fam=byf, h=np.stack([hb(right), hb(wrong)], 1))


def flip_counts(pn, pb, S, T, H, zone):
    """NET occupied & B6 free & S <= -thr (nats) in the scope: [true furniture, not, true in zone, not in zone]; and the reverse flips"""
    scope = T.known & H.region
    base = (pn == 1) & (pb == 0) & scope
    out = {}
    for thr in FLIP_THR:
        sel = base & (S * E.UNIT <= -thr)
        out[f'le_-{thr:g}'] = [int((sel & T.tr).sum()), int((sel & ~T.tr).sum()), int((sel & T.tr & zone).sum()), int((sel & ~T.tr & zone).sum())]
    sel = (pn == 0) & (pb == 1) & scope
    out['net_free_b6_occ'] = [int((sel & ~T.tr).sum()), int((sel & T.tr).sum()), int((sel & ~T.tr & zone).sum()), int((sel & T.tr & zone).sum())]
    return out


def _addw(acc, key, d, w):
    """accumulate nested count dicts / arrays with weight w"""
    if isinstance(d, dict):
        a = acc.setdefault(key, {})
        for k, v in d.items():
            _addw(a, k, v, w)
    else:
        v = np.asarray(d, np.int64) * int(w)
        acc[key] = acc[key] + v if key in acc else v


def _lists(d):
    if isinstance(d, dict):
        return {k: _lists(v) for k, v in d.items()}
    return np.asarray(d).tolist()


def object_zones(z):
    """zone = dil6(ood_mask); per-kind zones = dil6(union of that kind's objects); per-family zones likewise (obj_family: scale's
    scaled_0.6 / scaled_1.5 are families, not kinds)"""
    shape = z['occ'].shape; V = int(np.prod(shape))
    idx, off = z['objvox_idx'], z['objvox_off']
    out = []
    for labels in ([str(k) for k in z['obj_kinds']], [str(k) for k in z['obj_family']]):
        per = {}
        for i, kd in enumerate(labels):
            m = per.setdefault(kd, np.zeros(V, bool)); m[idx[off[i]:off[i + 1]]] = True
        out.append({kd: G.dil6(m.reshape(shape)) for kd, m in per.items()})
    return G.dil6(np.asarray(z['ood_mask'], bool)), out[0], out[1]


def evaluate(path, M):
    """all arms on one exported (variant, house, seed) npz"""
    z = np.load(path)
    H = G.House(z['occ'], z['slab'], z['obj'], z['cell']); private = z['private']; n = private.shape[0]
    F = G.forest(E.edges_list(z['edges']), n)
    K0T = [G.Target(private[a], H) for a in range(n)]
    R8T = [G.Target(private[cm].max(0), H, mult=(private[cm] >= 0).sum(0)) for cm in F['comps']]
    L = list(z['L_k0_cam_nf'])
    occ = z['occ'] == 1; slab = np.asarray(z['slab'], bool); NZ = occ.shape[2]
    near = ndimage.binary_dilation(occ, structure=CUBE)
    floor_k = slab.all((0, 1)) & (np.arange(NZ) < NZ // 2)                  # the floor slab layers
    layer1 = int(np.nonzero(floor_k)[0].max()) + 1
    near_nf = ndimage.binary_dilation(occ & ~floor_k[None, None, :], structure=CUBE)
    zone, kzones, fzones = object_zones(z)
    zones = dict(zone=zone, rest=~zone)
    # S2: nearest generator object (EDT index) -> family; Chebyshev distance 1 from an object voxel
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
    famvox = {f: int(np.isin(oid, np.nonzero(ofam == fidx[f])[0]).sum()) for f in fams}
    rstats = G.RelayStats()
    sL, _, _ = E.relay_sums(L, F, M['L'].table, rstats)
    sums = {h: E.relay_sums(L, F, M['b6'][h].table, rstats)[0] for h in 'AB'}
    nc = len(F['comps'])
    known = [sL[ci][1] > 0 for ci in range(nc)]
    preds = {'B4': [E.linear_readout(*sL[ci], 0) for ci in range(nc)]}
    for h in 'AB':
        preds[f'B6_{h}'] = [E.linear_readout(*sums[h][ci], M['b6'][h].theta) for ci in range(nc)]
    for ts, nt in M['nets'].items():
        s = sums[PAIR[ts]]
        lg = E.net_logits(nt['model'], [x[0] for x in s], [x[1] for x in s], 'cpu', 1)
        preds[f'NET_{ts}'] = [E.net_readout(lg[ci], s[ci][1], nt['j']) for ci in range(nc)]
    for arm, ps in preds.items():
        for ci in range(nc):
            assert np.array_equal(ps[ci] >= 0, known[ci]), f'{arm}: known mask != B4 (component {ci})'
    res = dict(R8={}, K0={}, flips={}, flips_unweighted={}, flip_split={}, n_agents=n, components=F['comps'], families=fams, family_voxels=famvox,
               layer1=layer1)
    for arm, ps in preds.items():
        rows, acc, kacc, facc, ph = [], {}, {}, {}, np.zeros(4, np.float64)
        for ci, cm in enumerate(F['comps']):
            kc = sL[ci][1]; w = len(cm)
            rows.append([E.scored(ps[ci], R8T[ci], H, kc), w])
            for zn, d in zone_counts(ps[ci], R8T[ci], H, kc, zones).items():
                _addw(acc, zn, d, w)
            for kd, v in kind_counts(ps[ci], R8T[ci], H, kzones).items():
                _addw(kacc, kd, v, w)
            for fk, v in kind_counts(ps[ci], R8T[ci], H, fzones).items():
                _addw(facc, fk, v, w)
            ph += np.asarray(phantom_blocks(ps[ci], kc, H, near, near_nf, layer1 + 1), np.float64) * w / n
        res['R8'][arm] = dict(rows=rows, zones=_lists(acc), kinds=_lists(kacc), families=_lists(facc), phantoms=ph.tolist())
    for ts in M['nets']:
        fl, flu = {}, {}
        for h in 'AB':
            acc, accu = {}, {}
            for ci, cm in enumerate(F['comps']):
                d = flip_counts(preds[f'NET_{ts}'][ci], preds[f'B6_{h}'][ci], sums[PAIR[ts]][ci][0], R8T[ci], H, zone)
                _addw(acc, 'f', d, len(cm)); _addw(accu, 'f', d, 1)
            fl[f'B6_{h}'] = _lists(acc['f']); flu[f'B6_{h}'] = _lists(accu['f'])
        res['flips'][f'NET_{ts}'] = fl; res['flips_unweighted'][f'NET_{ts}'] = flu
        acc, accu = {}, {}                                              # S2 split, own pairing (the B6 condition is implied by S <= 0)
        for ci, cm in enumerate(F['comps']):
            d = flip_split(preds[f'NET_{ts}'][ci], preds[f'B6_{PAIR[ts]}'][ci], sums[PAIR[ts]][ci][0], R8T[ci], H, near_fam, len(fams), cheb1, layer1)
            _addw(acc, 's', d, len(cm)); _addw(accu, 's', d, 1)

        def named(s):
            fa, hh = np.asarray(s['fam']), np.asarray(s['h'])
            return dict(fam={nm: fa[i].tolist() for i, nm in enumerate(fams + ['none'])}, h={nm: hh[i].tolist() for i, nm in enumerate(H_NAMES)})
        res['flip_split'][f'NET_{ts}'] = dict(w=named(acc['s']), u=named(accu['s']))
    # K0 (report only): per agent, own stalks
    own = {h: [E.stalk(L[a], M['b6'][h].table) for a in range(n)] for h in 'AB'}
    k0 = {}
    def k0_add(arm, a, p, ki):
        e = k0.setdefault(arm, dict(rows=[], zones={}))
        e['rows'].append(E.scored(p, K0T[a], H, ki))
        for zn, d in zone_counts(p, K0T[a], H, ki, zones).items():
            _addw(e['zones'], zn, d, 1)
    for a in range(n):
        ki = (L[a] != E.UNK).astype(np.int32)
        k0_add('B4', a, G.tern(L[a]), ki)
        for h in 'AB':
            e_, k_ = own[h][a]; assert np.array_equal(k_, ki > 0)
            k0_add(f'B6_{h}', a, E.linear_readout(e_, ki, M['b6'][h].theta), ki)
    for ts, nt in M['nets'].items():
        h = PAIR[ts]
        lk = E.net_logits(nt['model'], [e_ for e_, _ in own[h]], [k_.astype(np.int32) for _, k_ in own[h]], 'cpu', 1)
        for a in range(n):
            e_, k_ = own[h][a]; ki = k_.astype(np.int32)
            p = E.net_readout(lk[a], ki, nt['j']); assert np.array_equal(p >= 0, k_)
            k0_add(f'NET_{ts}', a, p, ki)
            if nt['j_sep'] is not None:
                k0_add(f'NET_{ts}_K0sep', a, E.net_readout(lk[a], ki, int(nt['j_sep'])), ki)
            if nt['theta_sep'] is not None:
                k0_add(f'LIN_{ts}_K0sep', a, E.linear_readout(e_, ki, int(nt['theta_sep'])), ki)
    res['K0'] = {arm: dict(rows=e['rows'], zones=_lists(e['zones'])) for arm, e in k0.items()}
    res['zone_voxels'] = int(zone.sum()); res['ood_voxels'] = int(np.asarray(z['ood_mask']).sum())
    res['scope_true_furn_in_zone'] = int(sum((R8T[ci].tr & zone).sum() * len(cm) for ci, cm in enumerate(F['comps'])))
    res['relay'] = dict(combines=rstats.combines, belief_checks=rstats.belief_checks)
    return res


# ================================================================ export (E3-R writing) + eval, one job = (variant, house)
def npz_name(variant, panel, rooms, index, seed):
    return f'{variant}_{panel}_r{rooms}_i{index}_s{seed}.npz'


def job_name(variant, panel, rooms, index):
    return f'{variant}_{panel}_r{rooms}_i{index}'


def run_job(job):
    variant, split, panel, rooms, index, seeds, opts = job
    t0 = time.time()
    rec = dict(format=JOB_FORMAT, variant=variant, split=split, panel=panel, rooms=rooms, index=index, seeds={}, script_sha256=opts['script_sha256'],
               gen_sha256=opts['gen_sha256'], v420_sha256=opts['v420_sha256'], e3r_sha256=opts['e3r_sha256'], noise_sha256=opts['noise_sha256'],
               ckpt_sha256={ts: sha(p) for ts, p in CKPTS.items()}, fit_sha256=sha(FIT))
    try:
        # review S5: every module this worker actually imported is re-hashed here (not only the parent's view)
        for key, path in (('script_sha256', os.path.abspath(__file__)), ('gen_sha256', WO.__file__), ('v420_sha256', WO.W.__file__),
                          ('e3r_sha256', E.__file__), ('noise_sha256', G.__file__)):
            assert sha(path) == opts[key], f'{os.path.basename(path)} changed since the run started ({key})'
        M = load_models(opts['threads'])
        E.W = WO.module_for(variant)                                    # the one replacement (house_ctx reads the module global W)
        try:
            ctx = E.house_ctx(split, rooms, index)
        except ValueError as e:
            if str(e) not in AGENT_SKIPS:                               # review S4c: only v420 agent_positions' messages are skips
                raise
            rec['skipped'] = repr(e)[:200]; return rec
        fx = ctx['fx']; rec['t_ctx'] = time.time() - t0
        assert fx['world_id'] == f"v420ood:{variant}:{split}:{E.W.v371.world_spec(split, index, rooms)['house_id']}"
        rec.update(world_id=fx['world_id'], agents=ctx['n'], components=ctx['F']['comps'], n_objects=int(fx['n_objects']),
                   objects=fx['objects'], private_sha256=E.W.array_hash(ctx['private']))
        base = dict(occ=np.asarray(ctx['occ'], np.int8), slab=np.asarray(fx['slab'], bool), obj=np.asarray(fx['obj'], bool), cell=ctx['cell'],
                    private=np.asarray(ctx['private'], np.int8), edges=E.edges_array(ctx['edges']),
                    ood_mask=np.asarray(fx['ood_mask'], bool), objvox_idx=fx['objvox_idx'], objvox_off=fx['objvox_off'],
                    obj_kinds=fx['obj_kinds'], obj_family=fx['obj_family'], obj_ood=fx['obj_ood'])
        os.makedirs(opts['export'], exist_ok=True)
        for seed in seeds:
            ts0 = time.time()
            L_k0, _, has_inter = E.tier_maps(ctx, TIER, seed, {})
            assert not has_inter
            path = os.path.join(opts['export'], npz_name(variant, panel, rooms, index, seed)); tmp = path + '.tmp.npz'
            np.savez_compressed(tmp, L_k0_cam_nf=np.stack(L_k0).astype(np.int8), **base); os.replace(tmp, path)
            del L_k0
            t_exp = time.time() - ts0
            r = evaluate(path, M)
            r.update(file=os.path.basename(path), file_sha256=sha(path), t_export=t_exp, t_eval=time.time() - ts0 - t_exp)
            rec['seeds'][str(seed)] = r
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-3000:]
    return rec


def _run_and_save(job):
    rec = run_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    _jdump(rec, os.path.join(d, job_name(rec['variant'], rec['panel'], rec['rooms'], rec['index']) + '.json'))
    return rec


def _worker_init(json_dir, index_file, threads):
    G.setup_env(json_dir, index_file)
    torch.set_num_threads(max(1, int(threads)))


# ================================================================ aggregation and verdict
def _f1(c):
    tp, fp, fn = c[0], c[1], c[2]
    return (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else None


def _tol1(c):
    npr, phit, ntr, rhit = c[3], c[4], c[5], c[6]
    if npr + ntr == 0:
        return None
    pp = phit / npr if npr else 1.; rr = rhit / ntr if ntr else 1.
    return 2 * pp * rr / max(pp + rr, 1e-12)


def _zone_summary(acc):
    out = {}
    for zn, d in acc.items():
        c = np.asarray(d['c'], np.float64)
        out[zn] = dict(f1=_f1(c), f1_tol1=_tol1(c), prec=(c[0] / (c[0] + c[1])) if c[0] + c[1] else None, rec=(c[0] / (c[0] + c[2])) if c[0] + c[2] else None,
                       free_recall=(c[7] / c[8]) if c[8] else None, n_true=int(c[5]), n_pred=int(c[3]), fp=int(c[1]), fn=int(c[2]),
                       by_k=[_f1(x) for x in np.asarray(d['k'], np.float64)], by_h=[_f1(x) for x in np.asarray(d['h'], np.float64)],
                       n_true_by_k=[int(x[0] + x[2]) for x in np.asarray(d['k'])], n_true_by_h=[int(x[0] + x[2]) for x in np.asarray(d['h'])])
    return out


def _sum_tree(items):
    acc = {}
    for it in items:
        for k, v in it.items():
            _addw(acc, k, v, 1)
    return acc


def _row_keys(s):
    return {k: s.get(k) for k in ('furn_f1', 'furn_f1_pooled', 'furn_f1_tol1', 'free_recall', 'surface_f1', 'unknown_known_fpr', 'var_occ_f1')}


def summarise(X):
    """X = list of per-(house, seed) eval dicts of one (variant, seed)"""
    D = dict(houses=len(X), R8={}, K0={}, flips={})
    for arm in ARMS:
        rows = [a for r in X for a in G._expand(r['R8'][arm]['rows'])]
        s = G.summarise_rows(rows)
        tall = [r['R8'][arm]['phantoms'][3] for r in X if len(r['R8'][arm]['phantoms']) > 3]
        D['R8'][arm] = dict(**_row_keys(s), strata=s.get('strata'), zones=_zone_summary(_sum_tree([r['R8'][arm]['zones'] for r in X])),
                            phantoms_per_house=float(np.mean([r['R8'][arm]['phantoms'][0] for r in X])),
                            phantom_voxels_per_house=float(np.mean([r['R8'][arm]['phantoms'][1] for r in X])),
                            blocks_per_house=float(np.mean([r['R8'][arm]['phantoms'][2] for r in X])),
                            phantoms_nofloor_tall_per_house=(float(np.mean(tall)) if len(tall) == len(X) else None))
        kin = _sum_tree([r['R8'][arm]['kinds'] for r in X])
        D['R8'][arm]['kinds'] = {kd: dict(f1=_f1(np.asarray(v, np.float64)), n_true=int(v[0] + v[2]), tp_fp_fn=np.asarray(v).tolist()) for kd, v in kin.items()}
        if all('families' in r['R8'][arm] for r in X):                 # report only: zone F1 per generator family (obj_family zones)
            fam = {k: np.asarray(v) for k, v in _sum_tree([r['R8'][arm]['families'] for r in X]).items()}
        else:                                                          # older job files: kind -> family of the generator
            fam = {}
            for kd, v in kin.items():
                fk = WO.FAMILY.get(kd, 'orig'); fam[fk] = fam[fk] + np.asarray(v) if fk in fam else np.asarray(v)
        D['R8'][arm]['families'] = {fk: dict(f1=_f1(v.astype(np.float64)), n_true=int(v[0] + v[2]), tp_fp_fn=v.tolist()) for fk, v in fam.items()}
    for arm in sorted({a for r in X for a in r['K0']}):
        if not all(arm in r['K0'] for r in X):
            continue
        s = G.summarise_rows([a for r in X for a in r['K0'][arm]['rows']])
        D['K0'][arm] = dict(**_row_keys(s), zones=_zone_summary(_sum_tree([r['K0'][arm]['zones'] for r in X])))
    have_u = all('flips_unweighted' in r for r in X)
    for net in ('NET_342101', 'NET_342202'):
        D['flips'][net] = {}
        for b6 in ('B6_A', 'B6_B'):
            acc = _sum_tree([r['flips'][net][b6] for r in X])
            accu = _sum_tree([r['flips_unweighted'][net][b6] for r in X]) if have_u else {}
            D['flips'][net][b6] = {k: dict(n=int(v[0] + v[1]), prec=(v[0] / (v[0] + v[1])) if v[0] + v[1] else None, n_zone=int(v[2] + v[3]),
                                           prec_zone=(v[2] / (v[2] + v[3])) if v[2] + v[3] else None,
                                           n_vox=(int(accu[k][0] + accu[k][1]) if k in accu else None),
                                           true_vox=(int(accu[k][0]) if k in accu else None)) for k, v in acc.items()}
    D['flip_split'] = {}                                               # S2 (report only), own pairing
    if all('flip_split' in r for r in X):
        for net in ('NET_342101', 'NET_342202'):
            W_ = _sum_tree([r['flip_split'][net]['w'] for r in X]); U_ = _sum_tree([r['flip_split'][net]['u'] for r in X])
            out = {}
            for part in ('fam', 'h'):
                out[part] = {}
                for nm, w in W_[part].items():
                    u = U_[part][nm]
                    out[part][nm] = dict(prec=(w[0] / (w[0] + w[1])) if w[0] + w[1] else None, true_vox=int(u[0]), false_vox=int(u[1]),
                                         false_cheb1_vox=(int(u[2]) if len(u) > 2 else None))
            D['flip_split'][net] = out
    fv = _sum_tree([{k: [v] for k, v in r['family_voxels'].items()} for r in X if 'family_voxels' in r])
    tot = sum(int(v[0]) for v in fv.values())
    D['family_voxel_share'] = {k: dict(voxels=int(v[0]), share=(int(v[0]) / tot if tot else None)) for k, v in sorted(fv.items())}
    D['zone_voxels_per_house'] = float(np.mean([r['zone_voxels'] for r in X])) if X else None
    tz = sum(r['scope_true_furn_in_zone'] for r in X); tt = sum(r['R8']['B6_A']['zones']['zone']['c'][5] + r['R8']['B6_A']['zones']['rest']['c'][5] for r in X)
    D['true_furn_in_zone_share'] = (tz / tt) if tt else None
    return D


def _cell(V, ts, pairing):
    """one (variant, seed, checkpoint) cell: NET vs its paired B6"""
    net = f'NET_{ts}'
    if pairing == 'own':
        b6 = f'B6_{PAIR[ts]}'
    else:
        b6 = max(('B6_A', 'B6_B'), key=lambda a: V['R8'][a]['furn_f1'])
    N, B = V['R8'][net], V['R8'][b6]
    zN, zB = N['zones'].get('zone', {}), B['zones'].get('zone', {})
    fl = V['flips'][net][b6]['le_-0']
    c = dict(b6=b6, net_all=N['furn_f1'], b6_all=B['furn_f1'], d_all=N['furn_f1'] - B['furn_f1'],
             net_zone=zN.get('f1'), b6_zone=zB.get('f1'), d_zone=(zN['f1'] - zB['f1']) if (zN.get('f1') is not None and zB.get('f1') is not None) else None,
             d_zone_tol1=(zN['f1_tol1'] - zB['f1_tol1']) if (zN.get('f1_tol1') is not None and zB.get('f1_tol1') is not None) else None,
             d_rest=(N['zones']['rest']['f1'] - B['zones']['rest']['f1']) if (N['zones']['rest']['f1'] is not None and B['zones']['rest']['f1'] is not None) else None,
             rest_fp_net=N['zones']['rest']['fp'], rest_fp_b6=B['zones']['rest']['fp'],
             flip_prec=fl['prec'], flip_n=fl['n'], flip_prec_zone=fl['prec_zone'], flip_n_zone=fl['n_zone'],
             ph_net=N['phantoms_per_house'], ph_b6=B['phantoms_per_house'],
             free_net=N['free_recall'], free_b6=B['free_recall'], d_free=N['free_recall'] - B['free_recall'],
             d_surface=N['surface_f1'] - B['surface_f1'], d_tol1_all=N['furn_f1_tol1'] - B['furn_f1_tol1'])
    # report-only rows (review S2 / S3 / S7)
    c.update(flip_n_vox=fl.get('n_vox'), ph_net_nofloor_tall=N.get('phantoms_nofloor_tall_per_house'), ph_b6_nofloor_tall=B.get('phantoms_nofloor_tall_per_house'),
             rest_n_true=N['zones']['rest'].get('n_true'),
             families={fk: dict(net=x['f1'], b6=B.get('families', {}).get(fk, {}).get('f1'), n_true=x['n_true'],
                                d=(x['f1'] - B['families'][fk]['f1']) if (x['f1'] is not None and B.get('families', {}).get(fk, {}).get('f1') is not None) else None)
                       for fk, x in N.get('families', {}).items()},
             flip_by_family=(V.get('flip_split', {}).get(net, {}).get('fam') if pairing == 'own' else None))
    C = CRIT
    flip_ok = c['flip_prec'] is None or c['flip_prec'] >= C['flip_safe']
    c['safe'] = bool(c['d_zone'] is not None and c['d_zone'] >= C['zone_safe'] - 1e-12 and c['d_all'] >= C['overall_safe'] - 1e-12 and flip_ok
                     and c['ph_net'] <= c['ph_b6'] + C['phantom_safe'] + 1e-12 and c['d_free'] >= C['free_safe'] - 1e-12)
    c['harm'] = bool((c['d_zone'] is not None and c['d_zone'] < C['zone_harm'] - 1e-12) or (c['flip_prec'] is not None and c['flip_prec'] < C['flip_harm'])
                     or c['ph_net'] > c['ph_b6'] + C['phantom_harm'] + 1e-12)
    return c


def verdict(S, expected_houses, seeds, smoke=False):
    reasons = []
    if S['failed']:
        reasons.append(f"{len(S['failed'])} failed jobs: {S['failed'][:6]}")
    if any(len(S.get(k, [])) > 1 for k in ('script_sha256', 'gen_sha256', 'ckpt_sha256', 'v420_sha256', 'e3r_sha256', 'noise_sha256')):
        reasons.append('mixed script / generator / frozen-module / checkpoint versions in the job files')
    if S.get('foreign'):                                               # review S4b
        reasons.append(f"{len(S['foreign'])} job files of another split / panel in the output directory: {S['foreign'][:6]}")
    if S['skipped'] and not smoke:                                     # review S4c / P5: a skip needs the coordinator's confirmation
        reasons.append(f"{len(S['skipped'])} skipped (variant, house): {S['skipped'][:6]} (P5: needs the coordinator's confirmation)")
    for v in VARIANTS:
        for sd in seeds:
            V = S['variants'].get(v, {}).get(str(sd))
            if V is None:
                reasons.append(f'{v}: noise seed {sd} missing'); continue
            if not smoke:
                skipped = {h for (vv, h) in S['skipped'] if vv == v}
                miss = set(expected_houses) - set(V['house_list']) - skipped
                if miss:
                    reasons.append(f'{v}: seed {sd}: {len(miss)} of {len(expected_houses)} houses missing')
    out = dict(criteria=CRIT, reasons=reasons, cells={}, ctrl={}, variants={})
    if reasons and not smoke:
        out['verdict'] = 'INCOMPLETE'
        return out
    ctrl_ok = True
    for sd in seeds:
        V = S['variants'].get('ctrl', {}).get(str(sd))
        if V is None:
            ctrl_ok = False; continue
        for ts in ('342101', '342202'):
            ctl = max(('B4', 'B6_A', 'B6_B'), key=lambda a: V['R8'][a]['furn_f1'])
            g = V['R8'][f'NET_{ts}']['furn_f1'] - V['R8'][ctl]['furn_f1']
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
                for ts in ('342101', '342202'):
                    c = _cell(V, ts, pairing)
                    if Vc is not None:                                  # review S1 (report only): ctrl, same checkpoint and seed
                        cc = _cell(Vc, ts, pairing)
                        c['ctrl'] = dict(b6=cc['b6'], flip_prec=cc['flip_prec'], flip_n_vox=cc['flip_n_vox'], ph_net=cc['ph_net'], ph_b6=cc['ph_b6'],
                                         ph_net_nofloor_tall=cc['ph_net_nofloor_tall'], ph_b6_nofloor_tall=cc['ph_b6_nofloor_tall'])
                    cells[f'{ts}_s{sd}'] = c
            n_exp = 2 * len(seeds)
            if len(cells) < n_exp:
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
    elif not ctrl_ok:
        out['verdict'] = 'INCONCLUSIVE (ctrl self-check failed: derived generator suspect)'
    else:
        out['verdict'] = {v: out['variants'][v]['verdict'] for v in JUDGED}
    return out


def aggregate(out_dir, smoke=False, seeds=NOISE_SEEDS, expected_houses=None):
    allrecs = [E._jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    # review S4b: a formal aggregation takes VAL jobs only, a smoke aggregation never takes VAL jobs; anything else is listed and blocks
    def foreign(r):
        return (r.get('split') == 'VALIDATION' or r.get('panel') == 'val') if smoke else (r.get('split') != 'VALIDATION' or r.get('panel') != 'val')
    recs = [r for r in allrecs if not foreign(r)]
    S = dict(variants={}, failed=[], skipped=[], n_jobs=len(recs), foreign=[job_name(r['variant'], r['panel'], r['rooms'], r['index']) for r in allrecs if foreign(r)],
             timing={}, notes=list(NOTES))
    for k in ('script_sha256', 'gen_sha256', 'v420_sha256', 'e3r_sha256', 'noise_sha256'):
        S[k] = sorted({r.get(k) for r in recs if r.get(k)})
    S['ckpt_sha256'] = sorted({json.dumps(r.get('ckpt_sha256'), sort_keys=True) for r in recs if r.get('ckpt_sha256')})
    for r in recs:
        tag = job_name(r['variant'], r['panel'], r['rooms'], r['index'])
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append((r['variant'], f"r{r['rooms']}_i{r['index']}"))
    ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
    S['timing'] = dict(job_seconds_median=float(np.median([r['seconds'] for r in ok])) if ok else None,
                       job_seconds_max=float(np.max([r['seconds'] for r in ok])) if ok else None,
                       peak_mb_max=float(np.max([r.get('peak_mb') or 0 for r in ok])) if ok else None)
    present_seeds = sorted({int(s) for r in ok for s in r['seeds']})
    for v in VARIANTS:
        for sd in present_seeds:
            X = [r['seeds'][str(sd)] for r in ok if r['variant'] == v and str(sd) in r['seeds']]
            if not X:
                continue
            D = summarise(X)
            D['house_list'] = sorted(f"r{r['rooms']}_i{r['index']}" for r in ok if r['variant'] == v and str(sd) in r['seeds'])
            D['objects'] = _object_stats([r for r in ok if r['variant'] == v and str(sd) in r['seeds']])
            S['variants'].setdefault(v, {})[str(sd)] = D
    exp = expected_houses if expected_houses is not None else [f'r{r}_i{i}' for r, i in G.panel_specs('val')]
    S['verdict'] = verdict(S, exp, seeds if not smoke else present_seeds, smoke=smoke)
    return S


def _object_stats(recs):
    fam, kinds, n = {}, {}, []
    for r in recs:
        n.append(r['n_objects'])
        for o in r['objects']:
            fam[o['family']] = fam.get(o['family'], 0) + 1; kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
    return dict(houses=len(recs), objects_per_house=float(np.mean(n)) if n else None, families=fam, kinds=kinds)


def report(S):
    L = []
    def f(x, d=4):
        return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, (float, np.floating)) else str(x))
    L.append(f"jobs {S['n_jobs']}, failed {len(S['failed'])}, skipped {len(S['skipped'])} {S['skipped']}, foreign {len(S.get('foreign', []))}; timing {S['timing']}")
    for nt in S.get('notes', []):
        L.append('NOTE ' + nt)
    for v, byseed in S['variants'].items():
        for sd, V in byseed.items():
            L.append(f"\n== {v} seed {sd}: {V['houses']} house-jobs, objects/house {f(V['objects']['objects_per_house'], 1)}, families {V['objects']['families']}")
            if V.get('family_voxel_share'):
                L.append('  object voxels by family (this run): ' + ', '.join(f"{k} {x['voxels']} ({x['share'] * 100:.1f} %)" for k, x in V['family_voxel_share'].items()))
            L.append('  arm           R8furn(E3R) pooled  tol1   free    surf   | zone F1 tol1  n_true | rest F1 fp     | phantoms/house blocks/house S3-nofloor-tall')
            for arm in ARMS:
                a = V['R8'][arm]; zz = a['zones'].get('zone', {}); rr = a['zones'].get('rest', {})
                rf = 'n/a' if not rr.get('n_true') else f(rr.get('f1'))       # review S6: an empty truth set is not a zero score
                L.append(f"  {arm:13s} {f(a['furn_f1'])} {f(a['furn_f1_pooled'])} {f(a['furn_f1_tol1'])} {f(a['free_recall'])} {f(a['surface_f1'])} | "
                         f"{f(zz.get('f1'))} {f(zz.get('f1_tol1'))} {zz.get('n_true')} | {rf:6s} {rr.get('fp')} | "
                         f"{f(a['phantoms_per_house'], 2)} {f(a['blocks_per_house'], 1)} {f(a.get('phantoms_nofloor_tall_per_house'), 2)}")
            for net, d in V['flips'].items():
                L.append(f"  flips {net}: " + '; '.join(f"vs {b6} S<=0 n={x['le_-0']['n']} (voxels {x['le_-0'].get('n_vox')}) prec={f(x['le_-0']['prec'], 3)} "
                                                     f"(zone n={x['le_-0']['n_zone']} prec={f(x['le_-0']['prec_zone'], 3)})" for b6, x in d.items()))
            for net, sp in V.get('flip_split', {}).items():
                L.append(f"  flips {net} by nearest-object family (prec, true/false voxels, false at Chebyshev 1): "
                         + ', '.join(f"{k} {f(x['prec'], 2)} {x['true_vox']}/{x['false_vox']}/{x['false_cheb1_vox']}" for k, x in sp['fam'].items() if x['true_vox'] + x['false_vox']))
                L.append(f"  flips {net} by height: " + ', '.join(f"{k} {f(x['prec'], 2)} {x['true_vox']}/{x['false_vox']}" for k, x in sp['h'].items()))
            for arm in ('B6_A', 'NET_342101'):
                kz = V['R8'][arm]['kinds']
                L.append(f"  kinds {arm}: " + ', '.join(f"{k} {f(x['f1'], 3)} ({x['n_true']})" for k, x in sorted(kz.items())))
            for arm in ('B6_A', 'NET_342101', 'B6_B', 'NET_342202'):
                zz = V['R8'][arm]['zones'].get('zone', {})
                if zz.get('by_h'):
                    L.append(f"  zone strata {arm}: by height {[f(x, 3) for x in zz['by_h']]} by k {[f(x, 3) for x in zz['by_k']]}")
            for arm in ('B6_A', 'NET_342101_K0sep', 'LIN_342101_K0sep'):
                if arm in V['K0']:
                    a = V['K0'][arm]; L.append(f"  K0 {arm}: furn {f(a['furn_f1'])} zone {f(a['zones'].get('zone', {}).get('f1'))}")
    vd = S['verdict']
    L.append(f"\nVERDICT: {vd['verdict']}  reasons {vd['reasons']}")
    L.append(f"ctrl self-check ok={vd.get('ctrl_ok')}: " + '; '.join(f"{k} {x['control']} {x['gain'] * 100:+.2f} pp {'ok' if x['ok'] else 'OUT'}" for k, x in vd.get('ctrl', {}).items()))
    for v, res in vd.get('variants', {}).items():
        L.append(f"{v}: {res['verdict']} (max-pairing {res['max']['verdict']})")
        for k, c in res['own']['cells'].items():
            L.append(f"   {k} vs {c['b6']}: zone {c['d_zone'] * 100 if c['d_zone'] is not None else float('nan'):+.2f} pp, all {c['d_all'] * 100:+.2f} pp, "
                     f"flip {f(c['flip_prec'], 3)} (n {c['flip_n']}, voxels {c.get('flip_n_vox')}), phantoms {c['ph_net']:.2f} vs {c['ph_b6']:.2f}, "
                     f"free {c['d_free'] * 100:+.2f} pp -> {'SAFE' if c['safe'] else ('HARM' if c['harm'] else 'amb')}")
            cc = c.get('ctrl')
            L.append(f"      report only: ctrl same ckpt / seed: flip {f(cc['flip_prec'], 3) if cc else '-'} (voxels {cc['flip_n_vox'] if cc else '-'}), "
                     f"phantoms {f(cc['ph_net'], 2) if cc else '-'} vs {f(cc['ph_b6'], 2) if cc else '-'}; "
                     f"S3 nofloor-tall phantoms {f(c.get('ph_net_nofloor_tall'), 2)} vs {f(c.get('ph_b6_nofloor_tall'), 2)}; rest FP {c['rest_fp_net']} vs {c['rest_fp_b6']}")
            L.append('      report only: zone dF1 by family ' + ', '.join(f"{fk} {x['d'] * 100:+.2f} pp ({x['n_true']})" for fk, x in sorted(c['families'].items()) if x['d'] is not None)
                     + ((' | flip prec by family ' + ', '.join(f"{fk} {f(x['prec'], 2)} ({x['true_vox'] + x['false_vox']})" for fk, x in c['flip_by_family'].items()
                                                              if x['true_vox'] + x['false_vox'])) if c.get('flip_by_family') else ''))
    return '\n'.join(L)


# ================================================================ geometry only (no ray casting, no scoring)
def geom_cmd(args):
    houses = _houses(args)
    rows = []
    for v in args.variants.split(','):
        for rooms, index in houses:
            g, spec, r, oa = WO.geometry(v, args.split, index, rooms)
            rows.append(g)
            if not args.census:
                print(f"\n[{v}] {args.split} r{rooms}_i{index} {g['world_id']}: {g['n_objects']} objects, obj voxels {g['obj_voxels']}, "
                      f"ood voxels {g['ood_voxels']}, footprint share {g['footprint_share']:.3f}, agents_ok {g.get('agents_ok')}")
                nv = np.diff(oa['objvox_off'])
                for o, k in zip(spec['objects'], nv):
                    print(f"   {o['kind']:11s} {o.get('family', 'orig'):11s} ood={int(bool(o.get('ood')))} {o.get('placement', 'floor'):9s} "
                          f"scale={o.get('scale')} angle={None if o.get('angle_deg') is None else round(o['angle_deg'], 1)} prims={len(o['prims'])} voxels={int(k)}")
                if args.png:
                    _png(args.png, v, args.split, rooms, index, spec, r, oa)
            if args.fixture:
                t0 = time.time(); fx = WO.make_fixture(v, args.split, index, rooms)
                seen = (fx['private'] == 1).any(0) & fx['ood_mask']
                print(f"   fixture {time.time() - t0:.1f}s: ood voxels {int(fx['ood_mask'].sum())}, of which hit by some clean agent {int(seen.sum())}; "
                      f"objects {len(fx['objects'])}; private sha {WO.W.array_hash(fx['private'])[:12]}")
    if args.census:
        for v in args.variants.split(','):
            R = [g for g in rows if g['variant'] == v]
            fam = {}
            for g in R:
                for k, (c, vx) in g['families'].items():
                    a = fam.setdefault(k, [0, 0]); a[0] += c; a[1] += vx
            print(f"[{v}] {len(R)} houses: objects/house {np.mean([g['n_objects'] for g in R]):.1f}, obj voxels/house {np.mean([g['obj_voxels'] for g in R]):.0f}, "
                  f"ood voxels/house {np.mean([g['ood_voxels'] for g in R]):.0f}, footprint share {np.mean([g['footprint_share'] for g in R]):.3f}, "
                  f"agent placement failures {sum(not g.get('agents_ok', True) for g in R)}; families (count, voxels) {fam}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        _jdump(dict(split=args.split, rows=rows), os.path.join(args.out, f"geometry_{args.split}{'_census' if args.census else ''}.json"))


def _png(d, v, split, rooms, index, spec, r, oa):
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    occ = r['occ'].astype(bool); obj = r['obj']; walls = occ.all(2)
    img = np.full(occ.shape[:2] + (3,), 255, np.uint8)
    img[walls] = (90, 90, 90)
    img[obj.any(2)] = (70, 110, 220)
    img[oa['ood_mask'].any(2)] = (220, 60, 40)
    Image.fromarray(np.transpose(img, (1, 0, 2))[::-1]).resize((512, 512), Image.NEAREST).save(os.path.join(d, f'{v}_{split}_r{rooms}_i{index}.png'))


# ================================================================ CLI
def _houses(args):
    if args.houses:
        out = []
        for t in args.houses.split(','):
            r, i = t.strip().split(':'); out.append((int(r[1:] if r.startswith('r') else r), int(i)))
        return out
    if args.split == 'VALIDATION':
        return G.panel_specs('val')
    raise SystemExit('--houses is required for TRAIN')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'aggregate', 'geom'):
        p = sub.add_parser(name)
        p.add_argument('--out', default='')
        p.add_argument('--smoke', action='store_true')
        p.add_argument('--split', default='VALIDATION', choices=('VALIDATION', 'TRAIN'))
        p.add_argument('--variants', default=','.join(VARIANTS))
        p.add_argument('--houses', default='', help='r1:9,r2:9 (TRAIN smoke / geometry); default VAL = G.panel_specs(val)')
        if name == 'run':
            p.add_argument('--export', default=EXPORT_DIR); p.add_argument('--seeds', default='1,2'); p.add_argument('--workers', type=int, default=8)
            p.add_argument('--threads', type=int, default=2); p.add_argument('--formal', action='store_true')
            p.add_argument('--plan', default='', help='explicit variant:r:i list (smoke), overrides --variants x --houses')
        if name == 'aggregate':
            p.add_argument('--seeds', default='1,2')
        if name == 'geom':
            p.add_argument('--census', action='store_true'); p.add_argument('--fixture', action='store_true'); p.add_argument('--png', default='')
    a = ap.parse_args(argv)
    G.setup_env()
    if a.cmd == 'geom':
        if a.png:                                                       # review S8: floor-plan renderings never go into the public repo
            rp, pp = os.path.normcase(os.path.abspath(ROOT)), os.path.normcase(os.path.abspath(a.png))
            assert not (pp == rp or pp.startswith(rp + os.sep)), '--png must point outside the repository (e.g. a scratch directory)'
        return geom_cmd(a)
    if a.cmd == 'aggregate':
        S = aggregate(a.out, smoke=a.smoke, seeds=tuple(int(s) for s in a.seeds.split(',')))
        _jdump(S, os.path.join(a.out, 'summary.json'))
        txt = report(S); open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
        return S
    seeds = tuple(int(s) for s in a.seeds.split(','))
    if a.split == 'VALIDATION':
        assert a.formal and not a.smoke, 'VALIDATION is the verdict panel: run it only with --formal (smoke on TRAIN)'
        assert seeds == NOISE_SEEDS, 'D8-53: noise seeds 1, 2'
        # review S4a: the formal panel is exactly G.panel_specs('val') (no --plan, no partial --houses)
        assert not a.plan, '--plan is for TRAIN smoke runs only'
        assert not a.houses or sorted(_houses(a)) == sorted(G.panel_specs('val')), '--formal: --houses must be the whole VAL panel (omit it)'
    else:
        assert not a.formal, '--formal is for the VALIDATION panel only'
    if a.split == 'TRAIN':
        assert a.smoke, 'TRAIN runs are smoke runs'
        train = {(int(s['rooms']), int(s['index'])) for s in WO.W.scene_plan('TRAIN', E.TRAIN_PER_STRATUM)}
    out = a.out or (OUT_DIR if a.formal else SMOKE_DIR)
    os.makedirs(os.path.join(out, 'jobs'), exist_ok=True)
    panel = 'val' if a.split == 'VALIDATION' else 'train'
    if a.plan:
        plan = []
        for t in a.plan.split(','):
            v, r, i = t.split(':'); plan.append((v, int(r[1:] if r.startswith('r') else r), int(i)))
    else:
        plan = [(v, r, i) for v in a.variants.split(',') for r, i in _houses(a)]
    for v, r, i in plan:
        assert v in VARIANTS, v
        if a.split == 'TRAIN':
            assert (r, i) in train, f'TRAIN r{r}_i{i} not in scene_plan(TRAIN, 70)'
    opts = dict(out=out, export=a.export, threads=a.threads, script_sha256=sha(os.path.abspath(__file__)), gen_sha256=sha(WO.__file__),
                v420_sha256=sha(WO.W.__file__), e3r_sha256=sha(E.__file__), noise_sha256=sha(G.__file__), split=a.split, seeds=list(seeds))
    _jdump(dict(opts=opts, plan=plan, argv=sys.argv, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), os.path.join(out, 'run.json'))
    jobs = [(v, a.split, panel, r, i, seeds, opts) for v, r, i in plan
            if not os.path.exists(os.path.join(out, 'jobs', job_name(v, panel, r, i) + '.json'))]
    print(f'{len(jobs)} jobs ({len(plan) - len(jobs)} done), {a.workers} workers x {a.threads} threads -> {out}', flush=True)
    t0 = time.time()
    if a.workers <= 1:
        G._worker_init(os.environ['HOUSEEXPO_JSON_DIR'], os.environ['HOUSEEXPO_INDEX'])
        it = ((k, _run_and_save(j)) for k, j in enumerate(jobs, 1))
    else:
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(a.workers, initializer=_worker_init, initargs=(os.environ['HOUSEEXPO_JSON_DIR'], os.environ['HOUSEEXPO_INDEX'], a.threads), maxtasksperchild=1)
        it = enumerate(pool.imap_unordered(_run_and_save, jobs), 1)
    for k, rec in it:
        st = 'FAILED ' + rec.get('error', '') if rec.get('failed') else ('skipped ' + rec['skipped'] if rec.get('skipped') else 'ok')
        print(f"[{k}/{len(jobs)}] {job_name(rec['variant'], rec['panel'], rec['rooms'], rec['index'])} {rec.get('seconds', 0):.0f}s "
              f"peak {rec.get('peak_mb') or 0:.0f} MB {st}  elapsed {time.time() - t0:.0f}s", flush=True)
    if a.workers > 1:
        pool.close(); pool.join()
    exp = [f'r{r}_i{i}' for r, i in (_houses(a) if not a.plan else sorted({(r, i) for _, r, i in plan}))]
    S = aggregate(out, smoke=a.smoke, seeds=seeds, expected_houses=exp)
    _jdump(S, os.path.join(out, 'summary.json'))
    txt = report(S); open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8').write(txt); print(txt)
    return S


if __name__ == '__main__':
    main()
