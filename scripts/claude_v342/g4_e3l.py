"""g4_e3l.py (Claude, 2026-10-27): 批次 E3-L "sheaf 隐码 GBP"（预注册 OVERNIGHT_REPORT D8-54；5090 第 2 个租约）。
冻结依赖只读 import：g4_e3r.py（E3-R，sha 1d65d9a8...）与 g4_noise_panels.py（G4-E1，sha 2313d75b...）。不改它们。

底线（D8-54，用户原话）："sheaf 隐码 + GBP 迭代下降能量 + 解码"；融合部分与任务无关。
  能量（每个隐码格子 c 独立，二次形式）：
      E(x) = sum_i sum_c 1/2 Lam_i(c) ||x_i(c) - z_i(c)||^2 + sum_{(i,j) in F} sum_c 1/2 w ||x_i(c) - x_j(c)||^2
  z_i, Lam_i = Enc_psi(本机证据)；F = 通信森林（与此前所有批次同一个 G.forest）；w > 0 可学习标量。
  GaBP 信息形式 (h, Lam)，节点势 (Lam_i z_i, Lam_i)，消息 j -> i：
      Lt = Lam_j + sum_{k in N(j)\\i} Lam_{k->j}，ht = h_j + sum h_{k->j}；Lam_{j->i} = w Lt / (w + Lt)，h_{j->i} = w ht / (w + Lt)。
  信念 Lam_b = Lam_i + sum_k Lam_{k->i}，x_i = h_b / Lam_b（Lam_b = 0 时 x_i = 0，无信息）。
  每台机器只解码自己的信念：Dec_phi(x_i, log1p(Lam_b), 高度) -> 全分辨率三态（已知头 + 占据头）。
  Enc = 恒等、Lam = 计数、w -> inf 时信念 = sum Lam z / sum Lam = E3-R 的 S / k（单测 test_e3r_is_special_case）。

操作化（D8-54 没写死、由本实现写死的部分；最后回复里逐条列出）
  O1 隐码网格：128 x 128 x 32 体素按 4 x 4 x 2 块 -> 32 x 32 x 16 格，D = 8。
  O2 编码器 Enc_psi（每机独立、权重共享）：输入 = E3-R 的 features_t(e_i, 1[触及])，即 (e x 0.05 / 4, log1p(1[触及]), z)，
     同一缩放；conv3(3->16) / conv3 s2(16->32) / conv3(32->32) / conv3 s(2,2,1)(32->64) / conv3(64->64)，ReLU，1x1 头 -> 9 通道
     （8 = z_i，1 = a_i）；Lam_i = softplus(a_i) x 1[该 4x4x2 块内有触及体素]（块内无触及时 Lam 恰为 0）。Lam 头偏置初值
     log(e^2 - 1)（初始 Lam 约 2），z 头权重 N(0, 0.1^2)。a_i 前向截在 >= -8（softplus(-8) = 3.4e-4，量化后仍是 0），反向为
     单侧直通（审查 S1）：a >= -8 处梯度原样通过；a < -8 处只放行使 a 增大的梯度（梯度 <= 0 的部分），把 a 往回拉、不再往下推。
     CPU 冒烟里 24 步内 a 漂到约 -90、Lam 失活；双向直通时 a 会在 -8 以下继续漂，压力解除后要先走回来；单侧直通下塌缩后能立即
     恢复。截断只防失活，不改变前向，也不改变线路上可表达的东西。训练日志与 rate_curve.json 记录支撑格数（Lam_q > 0）、
     触及块数（硬支撑上限）、R_meas、lam 的轨迹。
  O3 w = exp(omega)，omega 可学习（保证 w > 0），初值 w0 = 1000（初始 Lam 约 2，w0 / Lam 约 500，接近 w -> inf 的 E3-R 特例）。
  O4 训练样本的树 = 森林里的连通子树：|s| ~ U{1..n}（n = 分量大小）；从分量里均匀抽一个起点，然后每次在"与当前子树相邻、
     不在子树里"的节点中均匀抽一个加入，直到 |s|。子集在森林上的诱导子图因此是一棵树（G.forest 断言连通）。|s| = 1 即 K0。
  O5 两遍：消息按 G.forest 的 order（发送方一侧从小到大，每条有向边恰好算一次；与任何根的"收集 + 分发"给出同一组消息，
     因为每条消息只依赖发送方一侧）。可微，训练与评估同一函数 gabp()。
  O6 线路量化：步长写死 DELTA_H = DELTA_LAM = 1（消息直接是整数）。源端势 (Lam_i z_i, Lam_i) 与每一跳的消息都量化；
     Lam_q = 0 处强制 h_q = 0（零精度不携带信息）。训练：前向确定性取整、反向恒等（直通估计）；评估：确定性取整。
     评估另跑一遍不量化的浮点 GBP，报告信念差（量化信念 Lam_b > 0 的格子上 |x_q - x_f|（通道最大）的均值 / 最大值），
     另报"浮点 Lam_b >= 半个步长、量化后为 0"的格子数（被线路丢掉的信息）。
  O7 字节（判决口径，D8-54 补充 1）：每条有向消息 = 支撑掩码 (Lam_q > 0) 只编码一次 + 9 个通道（Lam 1 个、h 8 个）的值只在支撑内
     编码、掩码作为已知条件。掩码：G.ctx_tern_bytes(0/1 图)（G4-E1 的三元 ctx 编码器，9 个因果邻居、KT，图里只出现 0/1 两个
     符号，+ 2 B 头）。值：value_bytes_given_mask() = G.int_ctx_bytes 去掉掩码那一部分——符号位 (v > 0) 只在支撑内、用与
     G.ctx_tern_bits 相同的上下文（邻居的三元状态，支撑外 = 未知）做二值 KT，加 G.int_magnitude_events 的幅度流（本来就只
     编码支撑内），+ 4 B 头。全部有向边求和 = 每房字节（与 E3-R 的 relay 口径一致：每条有向边一次）。
     旧口径"9 个通道各自 G.int_ctx_bytes(通道, 掩码)"（掩码计 9 次，上界）改为只报告（bytes_upper）。
  O8 码率（D8-54 补充 2）：R* = 20,000 bit（2.5 KB）是"每机源端量化消息 (Lam_q, h_q) 在线路上的实际码长"（O7 口径）。
     R* 由预注册的 8,000 bit 改为 20,000 bit 是协调者在审查后的决定（D8-54 补充）：审查外推 1 KB/机约对应 0.03-0.05 倍对照字节，
     比杠杆字节线 0.15 倍严 3-6 倍；2.5 KB/机预计约 0.08-0.12 倍，把码率目标对到判据。训练时每
     RATE_EVERY = 50 次更新（及第 1 次）在当前 TRAIN batch 上实测：取每个样本被解码的那台机器（batch 8 -> 8 台）的源端量化消息，
     按 O7 口径算码长；R_meas = 最近 RATE_WINDOW = 4 次实测的均值（滑动平均，约 32 台机器 / 200 次更新）。对偶上升只用 R_meas：
     lam_dual <- max(0, lam_dual + ETA clip(R_meas / R* - 1, -1, +1))，ETA = 1e-3 / 次更新（每次更新都走一步，R_meas 在两次实测之间
     不变）；学习率预热期（前 --warmup = 300 次更新）lam_dual 保持 0（截断与预热冻结均经协调者确认）。
     可微代理码长（只提供梯度）：学习的因子化熵模型，支撑只计一次 = 每格一个支撑 Bernoulli 项（一个可学习的概率 q）+ 支撑内
     9 个通道的值项（每通道一个 3 分量离散化 logistic 混合，初值均值 0、尺度 1/4、1、4）。支撑指示用直通估计（前向 1[Lam_q > 0]，
     反向对 Lam / DELTA_LAM 恒等），值用加均匀噪声的松弛。损失项 lam_dual x R_proxy / R*；熵模型参数只由自己的似然项训练（输入
     detach），单独一组、学习率恒为 ENT_LR = 1e-2（不预热、不退火）。每次实测同时在同一批机器上算代理码长（离散、无噪声），
     实测 / 代理之比的曲线写进 rate_curve.json；评估时每台机器报告实测、代理与上界三种码长。
  O9 解码器 Dec_phi：E3-R 同结构 3 层 3D U-Net、基宽 16、无归一化，输入 10 通道 = x_i（8）、log1p(Lam_b)、高度 z，
     隐码按最近邻复制上采样（每个格子复制到它的 4x4x2 体素），输出 2 通道 logit（已知、占据）。
  O10 损失：已知头 = 全网格（"计分范围"：FPR 在全网格的目标未知区上计）上的 BCE，标签 = 目标已知（子集干净私有地图的并，
     = E3-R 的 scope）；占据头 = 目标已知区（"计分范围 ∧ 已知"）上的 BCE，标签 = 干净占据；两项各自取均值后相加，
     再加 lam_dual x R_proxy / R* 与熵模型似然项 R_proxy(detach 输入) / R*。训练时每个样本只解码子树里均匀抽的一台机器
     （对"子树里每台机器各解码一次"的损失是无偏估计；评估时每台机器都解码）。数据、AdamW、预热 + 余弦、裁剪 1.0、
     clean 概率 0.1、二面体增广（所有机器与标签同一变换）、留出 index % 10 == 9、deadline 都照 E3-R。
  O11 阈值（训练后在留出房 cam_nf 上，TF32 关）：先已知阈值，再占据阈值（顺序选）。
     已知：取最小的 t_k，使合并的"未知 -> 已知 FPR"<= 对照 FPR + 0.15 pp（D8-54 补充 3：判决保护仍是 +0.3 pp，给 VAL 抽样噪声
       留一半余量）；对照 FPR = E3-R 的已知掩码（k > 0，线性与 NET 相同）
       在同一批留出房、同一批抽样上的合并 FPR。
     占据：在 t_k 下，合并空闲召回 >= 对照、家具 F1 最大、同分取最小；对照空闲召回 = 配对的 E3-R NET 检查点用它冻结的
       阈值在同一批留出房上重算（R8 分量展开 / K0 用其 K0 阈值）。约束不可行时取空闲召回最大的最小阈值，标记 infeasible
       （只报告；判决里的保护项照常计）。K0 阈值同理（单机、K0 目标、对照 = 自己触及掩码的 FPR 与 E3-R NET K0_sep 的空闲召回）。
     配对断言：E3-L 编码器的证据表 = 配对 E3-R 检查点的证据表（342101 <-> 表 A，342202 <-> 表 B）。
  O12 评估（CPU、租约外）：复用 E3-R 的导出（不重新 export），VAL / fresh x 噪声种子 1、2 x cam_nf / cam / clean；R8 用公共系
     地图（cam 档 = L_pub_cam，不配准），K0 用机体系地图；每台机器解码自己的信念，对该分量的 R8 目标计分，行 = 每台机器
     一行（= E3-R 的分量展开均值）。对照读数直接取 E3-R 的 summary.json（同一导出、同一计分代码；链里 sha 钉住），
     对照每房字节 = NET 臂 rnet_relay_median 的 sum + count（约 753 KB）。
  O13 断言（评估时每个分量、每个档、每个检查点）：float64 GaBP 与闭式解 (diag Lam + w L)^{-1} diag Lam z 的最大绝对差
     <= 1e-5，只在 sum Lam > 1e-6 x max(1, w) 的格子上比（更小的 Lam 让 float64 求解奇异——冒烟里 softplus 给出过约 1e-40——
     且这些格子在线路上 Lam_q = 0、不携带信息；被排除的格子数写进 job；另记相对误差 cf_rel = 最大绝对差 / 闭式解的最大 |x|，
     只报告，用来区分"GaBP 错"与"闭式解病态"，审查 S6；判据仍是绝对差 <= 1e-5）；能量按"迭代顺序"单调不增，迭代 = 精确块坐标下降：x(0) = z；收集遍每条消息 v -> p
     发出后把 v 一侧子树设为"给定其余变量时的条件最优"（x_v = (ht_v + w x_p)/(Lt_v + w)，向下递归，只用已收到的消息）；
     根收齐后整棵树回代 = 全局最优；分发遍每到一台机器把它设为 GaBP 信念均值 h_b / Lam_b。每步都是精确块极小化，所以单调
     是定理，断言检查实现（容差 1e-9 x max(1, E0)）。结果写进 job，由判决的保护项计（不抛异常）。
     注：把"各机当前信念均值"当迭代值并不单调（反例见单测 test_naive_belief_sequence_not_monotone），故不用它。
  O14 判决（D8-54 写死在 D8_54 里，不读外部判据文件）：见 verdict()。fresh（D8-54 补充 4，与 G4-E5 同一处一致）：fresh 4 格均值
     d_furn（E3-L - 对照）>= -1.0 pp，不满足时只把 LEVER 降为 AMBIGUOUS，绝不升级；字面的"符号与 VAL 相反"仍计算并报告。

用法（总是 python -B；CPU 阶段 CUDA_VISIBLE_DEVICES= ）：
  train     --cache C --evidence E --seed S --out R --updates N [--device cuda] [--batch 8] [--max-updates M] [--deadline-epoch T]
            [--ctrl-ckpt E3R/final.pt --ctrl-threshold E3R/threshold.json]   (有对照才选阈值)
  select    --run R [--device cpu] [--ctrl-ckpt ... --ctrl-threshold ...] [--overwrite]
  eval      --export X --out O --ckpt R1/checkpoints/final.pt,R2/... [--workers 6] [--threads 1] [--only tag,...（--smoke）]
  aggregate --out O --runs R1,R2 --control-summary E3R/eval/summary.json
  selfcheck --pyroot P ;  fingerprint ...（= g4_e3r）
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, hashlib, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3r as R  # noqa: E402   (frozen; read only; restores CUDA_VISIBLE_DEVICES after g4_noise_panels)

G = R.G
W = R.W
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as Fnn  # noqa: E402

UNK = R.UNK
LMIN, NL = R.LMIN, R.NL
NBIN = R.NBIN
BLOCK = (4, 4, 2)                      # O1
D_LAT = 8
DELTA_H = 1.0                          # O6
DELTA_LAM = 1.0
W0 = 1000.0                            # O3
LAM_BIAS0 = math.log(math.e ** 2 - 1)  # softplus -> 2.0
A_FLOOR = -8.0                         # O2: softplus input floor (softplus(-8) = 3.4e-4: still Lam_q = 0; gradient never vanishes)
R_STAR = 20000.0                       # bits / agent: D8-54 supplement (coordinator, after review S2): 8,000 -> 20,000 (2.5 KB)
ETA = 1e-3                             # O8 (normalised dual step)
ENT_LR = 1e-2                          # O8: the entropy model's own constant learning rate (no warm-up, no cosine)
P_MIN = 1e-12                          # entropy model probability floor (<= 39.9 bits / symbol)
N_MIX = 3
GUARD_FPR = 0.003                      # verdict guard: unknown->known FPR <= control + 0.3 pp
GUARD_FPR_SELECT = 0.0015              # D8-54 supplement 3: the known threshold is selected at FPR <= control + 0.15 pp
RATE_EVERY = 50                        # D8-54 supplement 2: measure the wire code length every 50 updates (and at the first)
RATE_WINDOW = 4                        # R_meas = mean of the last 4 measurements
LN2 = math.log(2.0)
CF_TOL = 1e-5                          # GaBP vs closed form
CF_LAM_MIN = 1e-6                      # closed-form comparison on cells with sum Lam > 1e-6 x max(1, w) (O13)
EN_RTOL = 1e-9
MIN_UPDATES = R.MIN_UPDATES
RUN_FORMAT = 1
ARM = 'E3L'


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def file_sha256(p):
    return G.file_sha256(p)


# ================================================================ GaBP on a tree (information form), quantisation
class _FloorOneSided(torch.autograd.Function):
    """O2 (review S1): forward max(a, floor); backward: the gradient passes unchanged where a >= floor, and where a < floor only
    its non-positive part passes (a gradient-descent step can then only raise a back towards the floor, never push it further down)"""

    @staticmethod
    def forward(ctx, a, floor):
        ctx.save_for_backward(a); ctx.floor = float(floor)
        return a.clamp(min=floor)

    @staticmethod
    def backward(ctx, g):
        a, = ctx.saved_tensors
        return torch.where(a < ctx.floor, g.clamp(max=0.0), g), None


def floor_one_sided(a, floor=None):
    return _FloorOneSided.apply(a, A_FLOOR if floor is None else floor)


def quantize(h, lam, mode):
    """(h [.., D, *g], lam [.., *g]) -> quantised (multiples of the steps); Lam_q = 0 => h_q = 0.
    mode 'ste': forward = rounding, backward = identity (straight-through); 'round': deterministic; None: unchanged."""
    if mode is None:
        return h, lam
    lq = DELTA_LAM * torch.round(lam / DELTA_LAM)
    gate = (lq > 0).to(h.dtype).unsqueeze(-4)
    hq = DELTA_H * torch.round(h / DELTA_H) * gate
    if mode == 'ste':
        hg = h * gate
        return hg + (hq - hg).detach(), lam + (lq - lam).detach()
    assert mode == 'round', mode
    return hq, lq


def gabp(h, lam, Fl, w, qmode=None):
    """two-pass GaBP on the tree Fl (= G.forest of the local members): h [m, D, *g], lam [m, *g], w scalar (tensor or float).
    Every directed message is computed once in Fl['order'] (sender side size ascending) and quantised if qmode.
    Returns (h_b [m, D, *g], lam_b [m, *g], msgs {(s, r): (h, lam)})."""
    nb = Fl['nb']; msgs = {}
    for s, r in Fl['order']:
        lt = lam[s]; ht = h[s]
        for k in nb[s]:
            if k != r:
                lt = lt + msgs[(k, s)][1]; ht = ht + msgs[(k, s)][0]
        f = w / (w + lt)
        msgs[(s, r)] = quantize(ht * f.unsqueeze(0), lt * f, qmode)
    hb, lb = [], []
    for a in range(lam.shape[0]):
        l = lam[a]; hh = h[a]
        for k in nb[a]:
            l = l + msgs[(k, a)][1]; hh = hh + msgs[(k, a)][0]
        hb.append(hh); lb.append(l)
    return torch.stack(hb), torch.stack(lb), msgs


def belief_mean(hb, lb):
    """x = h_b / Lam_b where Lam_b > 0, else 0 (no information)"""
    pos = lb > 0
    return torch.where(pos.unsqueeze(-4), hb / torch.where(pos, lb, torch.ones_like(lb)).unsqueeze(-4), torch.zeros_like(hb))


def local_forest(F, members):
    """the tree of a connected member set inside the forest F (local indices 0..m-1, in the order of members)"""
    idx = {int(a): i for i, a in enumerate(members)}
    ed = [(idx[i], idx[j], 1.0) for i, j in F['und'] if i in idx and j in idx]
    Fl = G.forest(ed, len(members))
    assert len(Fl['comps']) == 1, f'members {list(members)} are not connected in the forest'
    return Fl


def sample_subtree(rng, nbs, comp, m):
    """O4: a connected subtree of size m of the component (uniform start, then a uniform frontier node at a time)"""
    start = int(comp[int(rng.integers(len(comp)))]); sub = {start}
    while len(sub) < m:
        fr = sorted({int(v) for u in sub for v in nbs[u] if v not in sub})
        sub.add(fr[int(rng.integers(len(fr)))])
    return tuple(sorted(sub))


# ---------------------------------------------------------------- float64 checks (closed form, energy descent)
def laplacian(Fl, m):
    Lp = np.zeros((m, m))
    for i, j in Fl['und']:
        Lp[i, i] += 1; Lp[j, j] += 1; Lp[i, j] -= 1; Lp[j, i] -= 1
    return Lp


def closed_form(z, lam, Fl, w):
    """(diag Lam + w L)^{-1} diag Lam z per cell; z [m, D, C], lam [m, C] float64 -> x [m, D, C], compared-cell mask [C].
    Only cells with sum Lam > CF_LAM_MIN x max(1, w) are solved: below that the system is numerically singular in float64
    (cond ~ 2 w m^2 / sum Lam; softplus can return ~1e-40) and the cell carries no information on the wire (Lam_q = 0)."""
    m, Dd, C = z.shape
    inf = lam.sum(0) > CF_LAM_MIN * max(1.0, float(w))
    x = np.zeros_like(z)
    if inf.any():
        A = np.zeros((int(inf.sum()), m, m)); A += w * laplacian(Fl, m)[None]
        li = lam[:, inf].T                                              # [Ci, m]
        A[:, np.arange(m), np.arange(m)] += li
        b = (li[:, :, None] * np.transpose(z[:, :, inf], (2, 0, 1)))   # [Ci, m, D]
        x[:, :, inf] = np.transpose(np.linalg.solve(A, b), (1, 2, 0))
    return x, inf


def energy(x, z, lam, Fl, w):
    e = 0.5 * float((lam[:, None, :] * (x - z) ** 2).sum())
    for i, j in Fl['und']:
        e += 0.5 * w * float(((x[i] - x[j]) ** 2).sum())
    return e


def descent_energies(z, lam, Fl, w, x_gbp, lb_gbp):
    """O13: energies of the exact block-coordinate-descent reading of the two passes (root = local 0). z [m, D, C], lam [m, C]
    float64; x_gbp / lb_gbp = the GaBP belief means / precisions (float64). Returns (energies, final x)."""
    m = z.shape[0]; nb = Fl['nb']
    order = [0]; parent = {0: None}
    for u in order:
        for v in nb[u]:
            if v not in parent:
                parent[v] = u; order.append(v)
    assert len(order) == m
    h = lam[:, None, :] * z
    Lt = lam.copy(); Ht = h.copy()
    for v in reversed(order[1:]):                                       # upward (collect) quantities, leaves first
        f = w / (w + Lt[v]); p = parent[v]
        Lt[p] = Lt[p] + Lt[v] * f; Ht[p] = Ht[p] + Ht[v] * f[None]
    sub = {v: [u for u in order if _is_desc(u, v, parent)] for v in order}

    def cond(x, v):                                                     # subtree(v) <- argmin given x[parent(v)]
        for u in sub[v]:
            xp = x[parent[u]]
            x[u] = (Ht[u] + w * xp) / (Lt[u] + w)[None]
    x = z.copy(); en = [energy(x, z, lam, Fl, w)]
    for v in reversed(order[1:]):                                       # collect pass: message v -> parent(v)
        cond(x, v); en.append(energy(x, z, lam, Fl, w))
    pos = Lt[0] > 0                                                     # root has everything: global optimum by back-substitution
    x[0] = np.where(pos[None], Ht[0] / np.where(pos, Lt[0], 1.0)[None], x[0])
    for v in order[1:]:
        x[v] = (Ht[v] + w * x[parent[v]]) / (Lt[v] + w)[None]
    en.append(energy(x, z, lam, Fl, w))
    for v in order:                                                     # distribute pass: belief means of the message passing
        x[v] = np.where((lb_gbp[v] > 0)[None], x_gbp[v], x[v]); en.append(energy(x, z, lam, Fl, w))
    return en, x


def _is_desc(u, v, parent):
    while u is not None:
        if u == v:
            return True
        u = parent[u]
    return False


def gbp_checks(z, lam, Fl, w):
    """float64 GaBP (no quantisation) vs closed form, and energy monotonicity; z [m, D, *g], lam [m, *g] (any float tensors)"""
    z64 = z.detach().double().cpu(); l64 = lam.detach().double().cpu()
    hb, lb, _ = gabp(l64.unsqueeze(1) * z64, l64, Fl, float(w), None)
    xg = belief_mean(hb, lb)
    m, Dd = z64.shape[:2]
    zf = z64.reshape(m, Dd, -1).numpy(); lf = l64.reshape(m, -1).numpy(); xgf = xg.reshape(m, Dd, -1).numpy(); lbf = lb.reshape(m, -1).numpy()
    xc, inf = closed_form(zf, lf, Fl, float(w))
    cf = float(np.abs(xgf[:, :, inf] - xc[:, :, inf]).max()) if inf.any() else 0.0
    scale = float(np.abs(xc[:, :, inf]).max()) if inf.any() else 0.0
    cf_rel = cf / max(scale, 1e-300) if inf.any() else 0.0          # review S6: relative to the largest |x| of the closed form
    en, _ = descent_energies(zf, lf, Fl, float(w), xgf, lbf)
    tol = EN_RTOL * max(1.0, abs(en[0]))
    viol = int(sum(1 for a, b in zip(en[:-1], en[1:]) if b > a + tol))
    tiny = (lf.sum(0) > 0) & ~inf
    return dict(cf_max=cf, cf_rel=cf_rel, cf_scale=scale, violations=viol, steps=len(en) - 1, e0=en[0], e_final=en[-1], cf_cells=int(inf.sum()),
                cf_excluded_tiny=int(tiny.sum()))


# ================================================================ networks
class Enc(nn.Module):
    """O2: per-agent encoder, full resolution -> 32 x 32 x 16 latent grid (strides (2,2,2) then (2,2,1))"""

    def __init__(self, cin=3, D=D_LAT, width=16):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.net = nn.Sequential(nn.Conv3d(cin, c1, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv3d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv3d(c2, c3, 3, stride=(2, 2, 1), padding=1), nn.ReLU(inplace=True),
                                 nn.Conv3d(c3, c3, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv3d(c3, D + 1, 1); self.D = D
        for mm in self.net:
            if isinstance(mm, nn.Conv3d):
                nn.init.kaiming_normal_(mm.weight, nonlinearity='relu'); nn.init.zeros_(mm.bias)
        nn.init.normal_(self.head.weight, std=0.1); nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[D] = LAM_BIAS0

    def forward(self, x):
        y = self.head(self.net(x))
        return y[:, :self.D], y[:, self.D]


class Dec(nn.Module):
    """O9: = E3-R UNet3 (3 levels, w / 2w / 4w, two 3x3x3 conv + ReLU per level, no normalisation) with a 2-channel head"""

    def __init__(self, cin, width=16, nout=2):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.e1 = R._block(cin, c1); self.e2 = R._block(c1, c2); self.b = R._block(c2, c3)
        self.u2 = nn.ConvTranspose3d(c3, c2, 2, stride=2); self.d2 = R._block(2 * c2, c2)
        self.u1 = nn.ConvTranspose3d(c2, c1, 2, stride=2); self.d1 = R._block(2 * c1, c1)
        self.head = nn.Conv3d(c1, nout, 1)
        for mm in self.modules():
            if isinstance(mm, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(mm.weight, nonlinearity='relu'); nn.init.zeros_(mm.bias)
        nn.init.normal_(self.head.weight, std=1e-3); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(Fnn.max_pool3d(x1, 2))
        xb = self.b(Fnn.max_pool3d(x2, 2))
        y2 = self.d2(torch.cat((self.u2(xb), x2), 1))
        y1 = self.d1(torch.cat((self.u1(y2), x1), 1))
        return self.head(y1)


class EntropyModel(nn.Module):
    """O8: the differentiable proxy code length (gradients only). Factorised over cells; the support (Lam_q > 0) counted ONCE: one
    Bernoulli(q) term per cell (q learnable) + inside the support, per channel a discretised mixture of N_MIX logistics."""

    def __init__(self, C=D_LAT + 1, K=N_MIX):
        super().__init__()
        self.logit_pi = nn.Parameter(torch.zeros(C, K))
        self.mu = nn.Parameter(torch.zeros(C, K))                                          # zero-centred, scales 1/4, 1, 4 (heavy tail)
        self.log_s = nn.Parameter(torch.log(torch.logspace(-1, 1, K, base=4.0)).repeat(C, 1))
        self.sup_logit = nn.Parameter(torch.tensor(0.0))                                   # support probability q = sigmoid(sup_logit)

    def value_bits(self, y, noisy=False, detach=False):
        """y [N, C, cells] in units of the quantisation step -> code length in bits of every value [N, C, cells]"""
        lp, mu, ls = self.logit_pi, self.mu, self.log_s
        if detach:
            lp, mu, ls = lp.detach(), mu.detach(), ls.detach()
        if noisy:
            y = y + torch.rand_like(y) - 0.5
        pi = torch.softmax(lp, -1)[None, :, :, None]; mu = mu[None, :, :, None]; s = torch.exp(ls)[None, :, :, None]
        yy = y[:, :, None, :]
        p = (pi * (torch.sigmoid((yy + 0.5 - mu) / s) - torch.sigmoid((yy - 0.5 - mu) / s))).sum(2)
        return -torch.log2(p.clamp_min(P_MIN))

    def bits_parts(self, y, m, noisy=False, detach=False):
        """y [N, C, cells] symbols, m [N, cells] support indicator (0 / 1; may carry a straight-through gradient)
        -> (support bits [N], in-support value bits per channel [N, C])"""
        sl = self.sup_logit.detach() if detach else self.sup_logit
        b1 = -Fnn.logsigmoid(sl) / LN2; b0 = -Fnn.logsigmoid(-sl) / LN2
        sup = (m * b1 + (1 - m) * b0).sum(-1)
        return sup, (self.value_bits(y, noisy, detach) * m.unsqueeze(1)).sum(-1)

    def bits(self, y, m, noisy=False, detach=False):
        sup, val = self.bits_parts(y, m, noisy, detach)
        return sup + val.sum(-1)


def upsample(t):
    """O9: nearest-neighbour replication of the latent grid onto the voxels ([..., x, y, z] -> [..., 4x, 4y, 2z])"""
    return t.repeat_interleave(BLOCK[0], -3).repeat_interleave(BLOCK[1], -2).repeat_interleave(BLOCK[2], -1)


class E3L(nn.Module):
    def __init__(self, D=D_LAT, width=16, w0=W0):
        super().__init__()
        self.D = D
        self.enc = Enc(3, D, width); self.dec = Dec(D + 2, width, 2); self.ent = EntropyModel(D + 1, N_MIX)
        self.log_w = nn.Parameter(torch.tensor(math.log(w0)))

    @property
    def w(self):
        return torch.exp(self.log_w)

    def encode(self, f, kn):
        """f [N, 3, X, Y, Z] features, kn [N, X, Y, Z] touched -> z [N, D, g], lam [N, g] (0 on blocks without a touched voxel)"""
        z, a = self.enc(f)
        sup = Fnn.max_pool3d(kn.to(f.dtype).unsqueeze(1), BLOCK, BLOCK)[:, 0] > 0
        a = floor_one_sided(a)                                          # O2: forward a >= -8, one-sided straight-through backward
        return z, Fnn.softplus(a) * sup.to(a.dtype)

    def source(self, z, lam, qmode):
        """the agents' potentials (h = Lam z, Lam), quantised; the proxy's symbols (continuous, gated) [N, D + 1, cells] and the
        support indicator [N, cells] (forward 1[Lam_q > 0], backward identity w.r.t. Lam / DELTA_LAM)"""
        h = lam.unsqueeze(1) * z
        hq, lq = quantize(h, lam, qmode)
        m = (DELTA_LAM * torch.round(lam / DELTA_LAM) > 0).to(h.dtype)
        sym = torch.cat([(lam / DELTA_LAM).unsqueeze(1), h * m.unsqueeze(1) / DELTA_H], 1).flatten(2)
        ls = lam / DELTA_LAM
        return hq, lq, sym, (m + (ls - ls.detach())).flatten(1)

    def dec_input(self, x, lb):
        """x [B, D, g], lb [B, g] -> [B, D + 2, X, Y, Z] (x, log1p(Lam_b), height)"""
        up = upsample(torch.cat([x, torch.log1p(lb).unsqueeze(1)], 1))
        Z = up.shape[-1]
        zp = torch.as_tensor(R.zplane(Z), device=up.device, dtype=up.dtype).expand(up.shape[0], 1, *up.shape[2:])
        return torch.cat([up, zp], 1)

    def decode(self, hb, lb):
        return self.dec(self.dec_input(belief_mean(hb, lb), lb))


def n_params(m):
    return R.n_params(m)


def weights_sha(m):
    return R.weights_sha(m)


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    r = ck['recipe']; model = E3L(r['D'], r['width'], r['w0']).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return ck, model


# ================================================================ data (E3-R cache, + the forest)
class LHouse(R.HouseData):
    def __init__(self, path, entry, seeds):
        super().__init__(path, entry, seeds)
        with np.load(path) as z:
            self.edges = R.edges_list(z['edges'])
        self.F = G.forest(self.edges, self.n)
        assert [list(c) for c in self.F['comps']] == self.comps, 'cache comp_of != forest components'
        self._fl = {}

    def local(self, members):
        key = tuple(members)
        if key not in self._fl:
            self._fl[key] = local_forest(self.F, key)
        return self._fl[key]


def load_cache(cache, manifest, which, device=None, verify=True):
    out = []
    for e in manifest['houses']:
        if (which == 'train' and e['holdout']) or (which == 'holdout' and not e['holdout']):
            continue
        p = os.path.join(cache, e['file'])
        if verify:
            assert file_sha256(p) == e['file_sha256'], f'{p}: sha256 differs from the manifest'
        h = LHouse(p, e, manifest['seeds'])
        out.append(h.to(device) if device is not None else h)
    return out


def sample_spec(rng, houses, p_clean=R.P_CLEAN):
    """(house, draw, connected subtree, aug, decoded agent)"""
    h = int(rng.integers(len(houses))); H = houses[h]
    d = 0 if rng.random() < p_clean else 1 + int(rng.integers(len(H.draws) - 1))
    cm = H.comps[int(rng.integers(len(H.comps)))]
    m = 1 + int(rng.integers(len(cm)))
    sub = sample_subtree(rng, H.F['nb'], cm, m)
    aug = int(rng.integers(8))
    dec = sub[int(rng.integers(m))]
    return h, d, sub, aug, dec


def aug_t(t, aug):
    """dihedral xy (E3-R order: transpose, flip x, flip y) on the last three dims [..., X, Y, Z]"""
    if aug & 1:
        t = t.transpose(-3, -2)
    if aug & 2:
        t = t.flip(-3)
    if aug & 4:
        t = t.flip(-2)
    return t


def evidence_t(L, table_t):
    kn = L != UNK
    e = torch.where(kn, table_t[(L.long() - LMIN).clamp(0, NL - 1)], torch.zeros((), dtype=table_t.dtype, device=L.device))
    return e, kn


def forward_batch(model, specs, houses, table_t, qmode):
    """encode every agent of every sample, GaBP on each sample's subtree, decode the sampled agent. Returns logits, labels and
    src = dict(sym, msk (proxy inputs, every encoded agent), hq, lq (quantised source potentials), dec (rows of the decoded agents))"""
    feats, kns, labs, sizes = [], [], [], []
    for h, d, sub, aug, dec in specs:
        H = houses[h]
        e, kn = evidence_t(H.t_maps[d][list(sub)], table_t)
        feats.append(aug_t(R.features_t(e, kn.to(torch.int32)), aug)); kns.append(aug_t(kn, aug))
        scope = (H.t_pk.int() & sum(1 << a for a in sub)) != 0
        labs.append((aug_t(scope, aug), aug_t(H.t_occ, aug))); sizes.append(len(sub))
    z, lam = model.encode(torch.cat(feats).contiguous(), torch.cat(kns))
    hq, lq, sym, msk = model.source(z, lam, qmode)
    w = model.w
    hbs, lbs, rows = [], [], []
    o = 0
    for (h, d, sub, aug, dec), m in zip(specs, sizes):
        Fl = houses[h].local(sub)
        hb, lb, _ = gabp(hq[o:o + m], lq[o:o + m], Fl, w, qmode)
        i = sub.index(dec); hbs.append(hb[i]); lbs.append(lb[i]); rows.append(o + i); o += m
    logits = model.decode(torch.stack(hbs), torch.stack(lbs))
    yk = torch.stack([s for s, _ in labs]); yo = torch.stack([oc for _, oc in labs])
    return logits, yk, yo, dict(sym=sym, msk=msk, hq=hq, lq=lq, lam=lam, dec=rows)


def bce_terms(logits, yk, yo):
    lk = Fnn.binary_cross_entropy_with_logits(logits[:, 0], yk.float())
    m = yk.float()
    lo = (Fnn.binary_cross_entropy_with_logits(logits[:, 1], yo.float(), reduction='none') * m).sum() / m.sum().clamp(min=1.0)
    return lk, lo


@torch.no_grad()
def holdout_bce(model, specs, houses, table_t, bs):
    model.eval(); tk = to = 0.; n = 0
    for i in range(0, len(specs), bs):
        lg, yk, yo, _ = forward_batch(model, [(h, d, s, 0, dc) for h, d, s, _, dc in specs[i:i + bs]], houses, table_t, 'round')
        lk, lo = bce_terms(lg, yk, yo); k = len(specs[i:i + bs])
        tk += float(lk) * k; to += float(lo) * k; n += k
    model.train()
    return tk / max(n, 1), to / max(n, 1)


def probe_check(model, houses, table_t, rng, n=3):
    """start-of-training probe (CPU float64): GaBP == closed form and energy descent on a few sampled subtrees"""
    out = []
    with torch.no_grad():
        for _ in range(n):
            h, d, sub, _, _ = sample_spec(rng, houses, 0.0)
            H = houses[h]
            e, kn = evidence_t(H.t_maps[d][list(sub)], table_t)
            z, lam = model.encode(R.features_t(e, kn.to(torch.int32)), kn)
            c = gbp_checks(z, lam, H.local(sub), float(model.w.detach())); out.append(c)
            assert c['cf_max'] <= CF_TOL and c['violations'] == 0, f'GaBP probe failed: {c}'
    return out


@torch.no_grad()
def measure_rate(model, src):
    """O8: wire code length (O7 口径) of the decoded agents' quantised source messages, and the proxy (discrete, no noise) on the same"""
    rows = src['dec']; hq = DELTA_H * torch.round(src['hq'][rows].detach() / DELTA_H); lq = DELTA_LAM * torch.round(src['lq'][rows].detach() / DELTA_LAM)
    wire = [8.0 * wire_bytes(*msg_ints(hq[i], lq[i])) for i in range(len(rows))]
    m = (lq > 0).to(hq.dtype)
    sym = torch.cat([(lq / DELTA_LAM).unsqueeze(1), hq / DELTA_H], 1).flatten(2)
    prox = model.ent.bits(sym, m.flatten(1)).float().cpu().numpy()
    return dict(measured_bits=float(np.mean(wire)), proxy_bits=float(np.mean(prox)), measured_over_proxy=float(np.mean(wire) / max(np.mean(prox), 1e-9)),
                support_cells=float(m.flatten(1).sum(1).mean()), touched_cells=float((src['lam'][rows].detach() > 0).flatten(1).sum(1).float().mean()),
                agents=len(rows))


def dual_step(lam_dual, r_hat):
    """O8: lam <- max(0, lam + ETA x clip(R_hat / R* - 1, -1, +1)). The clip (deviation from the literal additive rule, flagged in the
    report) bounds the step at ETA either way: with the literal rule an untrained entropy model (R_hat / R* ~ 10-50 in the first
    hundreds of updates) would drive lam to O(1-10) during the warm-up and take thousands of updates to come back."""
    return max(0.0, lam_dual + ETA * min(1.0, max(-1.0, r_hat / R_STAR - 1.0)))


def train(args):
    t_start = time.time()
    assert args.seed in R.TRAIN_SEEDS or args.smoke, f'training seed {args.seed} not in {R.TRAIN_SEEDS}'
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(max(1, args.threads))
    man_path = os.path.join(args.cache, 'manifest.json'); man_sha = file_sha256(man_path)
    if args.manifest_sha:
        assert man_sha == args.manifest_sha, f'cache manifest sha {man_sha} != pinned {args.manifest_sha}'
    man = _jload(man_path)
    assert man['format'] == R.CACHE_FORMAT and not (set(man['seeds']) & R.RESERVED_SEEDS), 'cache format / reserved noise seeds'
    ev = R.Evidence.from_spec(args.evidence, allow_incomplete=args.smoke)
    ctrl_paths = None
    if args.ctrl_ckpt:
        ck_r = torch.load(args.ctrl_ckpt, map_location='cpu', weights_only=False)
        assert np.array_equal(R.Evidence.from_recipe(ck_r['recipe']['evidence']).table, ev.table), 'evidence != paired E3-R checkpoint evidence'
        assert args.smoke or int(ck_r['seed']) == args.seed, f"paired E3-R checkpoint seed {ck_r['seed']} != {args.seed}"
        ctrl_paths = dict(ckpt=os.path.abspath(args.ctrl_ckpt), ckpt_sha256=file_sha256(args.ctrl_ckpt), threshold=os.path.abspath(args.ctrl_threshold),
                          threshold_sha256=file_sha256(args.ctrl_threshold), seed=int(ck_r['seed']))
        del ck_r
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    tr = load_cache(args.cache, man, 'train', dev); ho = load_cache(args.cache, man, 'holdout', dev)
    assert tr and ho, 'empty train or holdout set'
    load_s = time.time() - t_start
    table_t = torch.as_tensor(ev.table, dtype=torch.int32, device=dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 32))
    model = E3L(D_LAT, args.width, W0).to(dev)
    assert n_params(model) <= 1_500_000
    init_sha = weights_sha(model)
    probe = probe_check(model, tr, table_t, np.random.default_rng([args.seed, 5]))
    convs = [q for nme, q in model.named_parameters() if q.ndim > 1 and not nme.startswith('ent.')]
    rest = [q for nme, q in model.named_parameters() if q.ndim <= 1 and not nme.startswith('ent.')]
    entp = list(model.ent.parameters())
    assert len(convs) + len(rest) + len(entp) == len(list(model.parameters()))
    opt = torch.optim.AdamW([dict(params=convs, weight_decay=args.wd), dict(params=rest, weight_decay=0.0),
                             dict(params=entp, weight_decay=0.0, lr=ENT_LR, fixed_lr=True)], lr=args.lr)
    rng = np.random.default_rng([args.seed, 3]); hrng = np.random.default_rng([20261027, 9])
    hspecs = [sample_spec(hrng, ho) for _ in range(args.holdout_samples)]
    recipe = dict(D=D_LAT, width=args.width, w0=W0, block=list(BLOCK), delta_h=DELTA_H, delta_lam=DELTA_LAM, lam_bias0=LAM_BIAS0, r_star=R_STAR,
                  eta=ETA, dual_rule='lam <- max(0, lam + ETA clip(R_meas / R* - 1, -1, 1)) every update', rate_every=args.rate_every, rate_window=RATE_WINDOW,
                  rate_measured='wire bits (O7: mask once + in-support values) of the decoded agents (quantised source messages); R_meas = mean of the last RATE_WINDOW measurements',
                  rate_proxy='support Bernoulli + in-support per-channel logistic mixtures (gradients only)', ent_lr=ENT_LR, a_floor=A_FLOOR, dual_hold='lam = 0 for the first --warmup updates', n_mix=N_MIX, unit=R.UNIT, s_scale=R.S_SCALE, inputs='Enc(e x 0.05 / 4, log1p(1[touched]), z); Dec(x, log1p(Lam_b), z)',
                  evidence=ev.recipe(), p_clean=R.P_CLEAN, batch=args.batch, lr=args.lr, wd=args.wd, warmup=args.warmup, aug='dihedral xy',
                  train_decode='one uniform agent of the subtree per sample', subtree='connected (uniform start + uniform frontier growth)')
    run = dict(format=RUN_FORMAT, stage='train', batch_name='E3-L', prereg='D8-54', seed=args.seed, updates=args.updates, max_updates=args.max_updates,
               device=str(dev), recipe=recipe, parameters=n_params(model), parameters_by_part={k: n_params(getattr(model, k)) for k in ('enc', 'dec', 'ent')},
               initial_weights_sha256=init_sha, script_sha256=file_sha256(os.path.abspath(__file__)), e3r_sha256=file_sha256(R.__file__),
               noise_panels_sha256=file_sha256(G.__file__), cache=os.path.abspath(args.cache), manifest_sha256=man_sha, train_houses=len(tr),
               holdout_houses=len(ho), deadline_epoch=args.deadline_epoch, load_seconds=load_s, smoke=bool(args.smoke), control=ctrl_paths,
               probe_checks=probe, torch=torch.__version__, cudnn=(torch.backends.cudnn.version() if dev.type == 'cuda' else None),
               gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None), cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark), pythonpath=os.environ.get('PYTHONPATH'))
    with open(os.path.join(args.out, 'run.json'), 'w') as fh:
        json.dump(run, fh, indent=1)
    print(f'E3-L train seed {args.seed}: {len(tr)} train / {len(ho)} holdout houses, {n_params(model)} parameters {run["parameters_by_part"]}, '
          f'evidence {ev.desc.get("spec")}, load {load_s:.0f}s', flush=True)
    stop_at = min(args.updates, args.max_updates) if args.max_updates else args.updates
    history = []; acc = np.zeros(5); run_n = 0; marks = {}; truncated = False; u = 0; lam_dual = 0.0; gn = 0.
    meas = []; rate_curve = []; r_meas = None; last_rc = {}
    t_train = time.time()
    model.train()
    for u in range(stop_at):
        if args.deadline_epoch and time.time() > args.deadline_epoch - args.reserve:
            truncated = True; break
        for g in opt.param_groups:
            if not g.get('fixed_lr'):
                g['lr'] = R.lr_at(u, args.updates, args.lr, args.warmup)
        lg, yk, yo, src = forward_batch(model, [sample_spec(rng, tr) for _ in range(args.batch)], tr, table_t, 'ste')
        lk, lo = bce_terms(lg, yk, yo)
        sym, msk = src['sym'], src['msk']
        r_enc = model.ent.bits(sym, msk, noisy=True, detach=True).mean()                 # proxy: gradients into the encoder only
        r_ent = model.ent.bits(sym.detach(), msk.detach(), noisy=True).mean()           # the entropy model's own likelihood
        loss = lk + lo + lam_dual * r_enc / R_STAR + r_ent / R_STAR
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        done = u + 1
        if done == 1 or done % args.rate_every == 0:                    # O8: measured wire code length of the source messages
            rc = measure_rate(model, src)
            meas.append(rc['measured_bits']); r_meas = float(np.mean(meas[-RATE_WINDOW:])); last_rc = rc
            rc.update(updates=done, r_meas=r_meas, lam_dual=lam_dual); rate_curve.append(rc)
            with open(os.path.join(args.out, 'rate_curve.json'), 'w') as fh:
                json.dump(rate_curve, fh, indent=1)
        if done >= args.warmup:                                          # O8: lam held at 0 during the LR warm-up; R_meas only
            lam_dual = dual_step(lam_dual, r_meas)
        rh = float(r_enc.detach())
        acc += np.array([float(lk.detach()) + float(lo.detach()), float(lk.detach()), float(lo.detach()), rh, 1.0]); run_n += 1
        log_now = done % args.log_every == 0 or done == stop_at
        if log_now:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            marks[done] = time.time() - t_train
            a = acc / max(run_n, 1)
            print(f'upd{done} bce {a[0]:.5f} (known {a[1]:.5f} occ {a[2]:.5f}) R_meas {r_meas:.0f} bit (proxy {a[3]:.0f}) lam {lam_dual:.4f} '
                  f'support {last_rc.get("support_cells", 0):.0f}/{last_rc.get("touched_cells", 0):.0f} w {float(model.w.detach()):.2f} '
                  f'grad {gn:.3f} lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t_start:.0f}s', flush=True)
        if done % args.val_every == 0 or done == stop_at:
            hk, hoc = holdout_bce(model, hspecs, ho, table_t, args.batch)
            a = acc / max(run_n, 1)
            history.append(dict(updates=done, train_bce=a[0], train_bce_known=a[1], train_bce_occ=a[2], r_proxy_bits=a[3], r_meas_bits=r_meas, lam_dual=lam_dual,
                                support_cells=last_rc.get('support_cells'), touched_cells=last_rc.get('touched_cells'),
                                w=float(model.w.detach()), holdout_bce_known=hk, holdout_bce_occ=hoc, seconds=time.time() - t_start))
            print(f'  holdout bce known {hk:.5f} occ {hoc:.5f}', flush=True)
            with open(os.path.join(args.out, 'history.json'), 'w') as fh:
                json.dump(history, fh, indent=1)
        if log_now:
            acc[:] = 0; run_n = 0
    else:
        u = stop_at
    done = u if truncated else stop_at
    ks = sorted(marks); rate = None
    if len(ks) >= 2:
        k0 = ks[0] if len(ks) == 2 else ks[1]                        # skip the first window (cudnn autotune / warm-up)
        rate = (marks[ks[-1]] - marks[k0]) / max(ks[-1] - k0, 1)
    ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, recipe=recipe, seed=args.seed, updates_done=done,
              updates_planned=args.updates, truncated=truncated, lam_dual=lam_dual, w=float(model.w.detach()), run=run)
    ck_path = os.path.join(args.out, 'checkpoints', 'final.pt')
    torch.save(ck, ck_path)
    fin = dict(updates_done=done, updates_planned=args.updates, max_updates=args.max_updates, truncated=truncated,
               complete=(done == args.updates and not truncated), rate_s_per_update=rate, first_window_s=(marks[ks[0]] if ks else None),
               first_window_updates=(ks[0] if ks else None), load_seconds=load_s, train_seconds=time.time() - t_train, weights_sha256=weights_sha(model),
               w=float(model.w.detach()), lam_dual=lam_dual, r_meas_bits=r_meas, rate_measurements=len(rate_curve), rate_curve_last=(rate_curve[-1] if rate_curve else None),
               checkpoint=ck_path, checkpoint_sha256=file_sha256(ck_path), history_last=(history[-1] if history else None))
    fin_path = os.path.join(args.out, 'final.json')

    def write_fin():
        fin['seconds'] = time.time() - t_start
        with open(fin_path + '.tmp', 'w') as fh:
            json.dump(fin, fh, indent=1)
        os.replace(fin_path + '.tmp', fin_path)
    fin['threshold_status'] = 'pending' if (args.select and ctrl_paths) else 'not run'
    write_fin()                                                          # review S5: the checkpoint is recorded before the selection
    note = ''
    if args.select and ctrl_paths:
        ts = time.time()
        try:
            sel = select_run(model, ev, ho, dev, ctrl_paths['ckpt'], ctrl_paths['threshold'], chunk=2)
            sel['seconds'] = time.time() - ts
            with open(os.path.join(args.out, 'threshold.json'), 'w') as fh:
                json.dump(sel, fh, indent=1)
            fin['threshold'] = _thr_brief(sel); fin['threshold_status'] = 'ok'
            print('thresholds:', json.dumps(fin['threshold']), flush=True)
        except Exception as e:                                           # the seed keeps its checkpoint; no threshold.json -> INCOMPLETE
            fin['threshold_status'] = 'FAILED'; fin['threshold_error'] = repr(e)[:500]; fin['threshold_traceback'] = traceback.format_exc()[-2000:]
            note = ', threshold selection FAILED (checkpoint kept; run select later)'
            print(f'threshold selection FAILED: {e!r}', flush=True)
    elif args.select:
        print('no --ctrl-ckpt: threshold selection skipped (run select later)', flush=True)
    write_fin()
    print(f'E3-L train seed {args.seed} done: {done} updates ({"truncated" if truncated else "complete" if fin["complete"] else "partial"}), '
          f'rate {rate} s/update, w {float(model.w.detach()):.2f}, lam {lam_dual:.4f}, {fin["seconds"]:.0f}s{note}', flush=True)
    return fin


# ================================================================ read-out and threshold selection (held-out TRAIN houses, cam_nf)
def readout(klog, olog, jk, jo):
    """three-state: known iff known-logit bin >= jk; then occupied iff occupancy-logit bin >= jo"""
    kb = R.logit_bin(klog); ob = R.logit_bin(olog)
    return np.where(kb >= jk, (ob >= jo).astype(np.int8), np.int8(-1)).astype(np.int8)


def pick_known(U, u_tot, fpr_max):
    """smallest bin jk with pooled FPR('bin >= jk') <= fpr_max; infeasible -> NBIN - 1 (flagged)"""
    tail = np.cumsum(U[::-1])[::-1] / max(u_tot, 1)
    ok = tail <= fpr_max + 1e-15
    if ok.any():
        return int(np.flatnonzero(ok)[0]), True, tail
    return NBIN - 1, False, tail


def pick_occ(A, B, C, tr_tot, fr_tot, free_min):
    """E3-R rule on predicted-known voxels: pooled free recall >= free_min, furniture F1 maximal, ties -> smallest bin;
    infeasible -> the smallest bin of maximal free recall (flagged)"""
    sa = np.cumsum(A[::-1])[::-1]; sb = np.cumsum(B[::-1])[::-1]; pc = np.r_[0, np.cumsum(C)[:-1]]
    tp = sa; fp = sb; fn = tr_tot - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1); fr = pc / max(fr_tot, 1)
    ok = fr + 1e-15 >= free_min
    if ok.any():
        return int(np.flatnonzero(ok)[np.argmax(f1[ok])]), True, f1, fr
    return int(np.flatnonzero(fr >= fr.max() - 1e-15)[0]), False, f1, fr


@torch.no_grad()
def encode_maps(model, maps, table_t, device, chunk=4):
    """numpy int8 maps [n, X, Y, Z] -> z [n, D, g], lam [n, g] (float32, on device)"""
    zs, ls = [], []
    for i in range(0, len(maps), chunk):
        L = torch.from_numpy(np.ascontiguousarray(np.stack(maps[i:i + chunk]))).to(device)
        e, kn = evidence_t(L, table_t)
        z, lam = model.encode(R.features_t(e, kn.to(torch.int32)), kn)
        zs.append(z); ls.append(lam)
    return torch.cat(zs), torch.cat(ls)


@torch.no_grad()
def decode_np(model, hb, lb, chunk=2):
    out = []
    for i in range(0, hb.shape[0], chunk):
        out += list(model.decode(hb[i:i + chunk], lb[i:i + chunk]).float().cpu().numpy())
    return out


def agent_logits(model, houses, device, table_t, which, chunk=2):
    """yields (logits [2, X, Y, Z], Target, House, control known mask) for every agent of every cam_nf draw of the houses:
    R8 = each member's decoded GaBP belief on its component tree vs the component's R8 target (control known = k > 0 of the component);
    K0 = each agent's own quantised potential (no messages) vs its own target (control known = own touched)"""
    w = float(model.w.detach())
    for Hd in houses:
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        Ts = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps] if which == 'R8' else [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
        for d in range(1, len(Hd.draws)):
            maps = Hd.maps[d]
            z, lam = encode_maps(model, list(maps), table_t, device)
            hq, lq, _, _ = model.source(z, lam, 'round')
            if which == 'R8':
                for ci, cm in enumerate(Hd.comps):
                    hb, lb, _ = gabp(hq[cm], lq[cm], Hd.local(cm), w, 'round')
                    lg = decode_np(model, hb, lb, chunk); kc = (maps[cm] != UNK).any(0)
                    for i in range(len(cm)):
                        yield lg[i], Ts[ci], Hh, kc
            else:
                lg = decode_np(model, hq, lq, chunk)
                for a in range(Hd.n):
                    yield lg[a], Ts[a], Hh, maps[a] != UNK


@torch.no_grad()
def control_holdout(ck_path, thr_path, houses, device, chunk=2):
    """O11: the paired E3-R NET with its frozen thresholds on the same held-out houses / draws: pooled free recall, FPR (= its known
    mask k > 0), furniture F1; R8 component-expanded, K0 with its K0 threshold j_sep"""
    ck, net = R.load_ckpt(ck_path, device); thr = _jload(thr_path)
    ev = R.Evidence.from_recipe(ck['recipe']['evidence']); j = int(thr['j']); jk = int(thr['k0']['j_sep'])
    acc = {k: np.zeros(7, np.int64) for k in ('R8', 'K0')}                # free hits, free tot, fp-known, unk tot, tp, fp, fn
    with R.no_tf32():
        for Hd in houses:
            Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
            R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]; K0T = [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
            for d in range(1, len(Hd.draws)):
                maps = Hd.maps[d]
                sums = [R.direct_sums(maps, cm, ev.table) for cm in Hd.comps]
                lg = R.net_logits(net, [s for s, _ in sums], [k for _, k in sums], device, chunk)
                for ci, cm in enumerate(Hd.comps):
                    acc['R8'] += _counts(R.net_readout(lg[ci], sums[ci][1], j), R8T[ci], Hh) * len(cm)
                own = [R.stalk(maps[a], ev.table) for a in range(Hd.n)]
                lk = R.net_logits(net, [e for e, _ in own], [k.astype(np.int32) for _, k in own], device, chunk)
                for a in range(Hd.n):
                    acc['K0'] += _counts(R.net_readout(lk[a], own[a][1].astype(np.int32), jk), K0T[a], Hh)
    out = {}
    for k, c in acc.items():
        out[k] = dict(free=c[0] / max(c[1], 1), fpr=c[2] / max(c[3], 1), furn_f1=2 * c[4] / max(2 * c[4] + c[5] + c[6], 1), counts=c.tolist())
    out['stored'] = dict(free=thr.get('free'), k0_free_sep=thr['k0'].get('free_sep'), houses=thr.get('houses'))
    out['evidence_table'] = ev.table.tolist(); out['j'] = j; out['j_k0'] = jk; out['seed'] = ck.get('seed')
    return out


def _counts(p, T, H):
    kn = p >= 0; pr = (p == 1) & T.known & H.region; unk = ~T.known
    return np.array([int(((p == 0) & T.free).sum()), int(T.free.sum()), int((kn & unk).sum()), int(unk.sum()),
                     int((pr & T.tr).sum()), int((pr & ~T.tr).sum()), int((~pr & T.tr).sum())], np.int64)


def select_thresholds(model, houses, device, table_t, ctrl, chunk=2):
    """O11 (sequential: known threshold, then occupancy threshold), R8 and K0; two passes over the held-out houses each"""
    model.eval(); out = {}
    with R.no_tf32():
        for which in ('R8', 'K0'):
            c = ctrl[which]
            U = np.zeros(NBIN, np.int64); u_tot = 0; base = np.zeros(2, np.int64)
            for lg, T, Hh, kc in agent_logits(model, houses, device, table_t, which, chunk):
                unk = ~T.known
                U += np.bincount(R.logit_bin(lg[0])[unk], minlength=NBIN); u_tot += int(unk.sum())
                base += [int((kc & unk).sum()), int(unk.sum())]
            jk, feas_k, tail = pick_known(U, u_tot, c['fpr'] + GUARD_FPR_SELECT)
            A = np.zeros(NBIN, np.int64); B = np.zeros(NBIN, np.int64); C = np.zeros(NBIN, np.int64); tr_tot = fr_tot = 0; kfree = 0
            for lg, T, Hh, kc in agent_logits(model, houses, device, table_t, which, chunk):
                pk = R.logit_bin(lg[0]) >= jk; ob = R.logit_bin(lg[1])
                A += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NBIN)
                B += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NBIN)
                C += np.bincount(ob[pk & T.free], minlength=NBIN)
                tr_tot += int(T.tr.sum()); fr_tot += int(T.free.sum()); kfree += int((pk & T.free).sum())
            jo, feas_o, f1, fr = pick_occ(A, B, C, tr_tot, fr_tot, c['free'])
            out[which] = dict(j_known=jk, t_known=R.bin_edge(jk), fpr=float(tail[jk]), fpr_max=c['fpr'] + GUARD_FPR_SELECT, feasible_known=feas_k,
                              j_occ=jo, t_occ=R.bin_edge(jo), f1=float(f1[jo]), free=float(fr[jo]), free_min=c['free'], feasible_occ=feas_o,
                              free_known_coverage=kfree / max(fr_tot, 1), control=dict(free=c['free'], fpr=c['fpr'], furn_f1=c['furn_f1']),
                              control_fpr_check=base[0] / max(base[1], 1),
                              curve=dict(j=list(range(0, NBIN, 25)), f1=f1[::25].tolist(), free=fr[::25].tolist(), fpr=tail[::25].tolist()))
            assert abs(out[which]['control_fpr_check'] - c['fpr']) < 1e-12, 'control FPR (k > 0) differs between the two computations'
    model.train()
    return out


def select_run(model, ev, houses, device, ctrl_ckpt, ctrl_thr, chunk=2):
    dev = torch.device(device)
    ctrl = control_holdout(ctrl_ckpt, ctrl_thr, houses, dev, chunk)
    assert np.array_equal(np.asarray(ctrl['evidence_table']), ev.table), 'evidence != paired E3-R checkpoint evidence (O11 pairing)'
    table_t = torch.as_tensor(ev.table, dtype=torch.int32, device=dev)
    sel = select_thresholds(model, houses, dev, table_t, ctrl, chunk)
    names = [h.name for h in houses]
    stored = ctrl['stored']
    return dict(R8=sel['R8'], K0=sel['K0'], houses=names, n_houses=len(houses), draws=houses[0].draws[1:] if houses else [],
                control=dict(ckpt=os.path.abspath(ctrl_ckpt), ckpt_sha256=file_sha256(ctrl_ckpt), threshold=os.path.abspath(ctrl_thr),
                             threshold_sha256=file_sha256(ctrl_thr), seed=ctrl['seed'], j=ctrl['j'], j_k0=ctrl['j_k0'], R8=ctrl['R8'], K0=ctrl['K0'],
                             stored_free=stored['free'], stored_k0_free_sep=stored['k0_free_sep'],
                             same_houses_as_stored=(stored['houses'] == names) if stored['houses'] is not None else None,
                             d_free_vs_stored=((ctrl['R8']['free'] - stored['free']) if stored['free'] is not None else None)),
                rule='known: smallest t_k with pooled FPR <= control FPR (k > 0) + 0.15 pp (D8-54 supplement 3); occupancy (given t_k): pooled free recall >= paired '
                     'E3-R NET (its frozen thresholds, same houses), furniture F1 max, ties smallest; K0 the same on single agents',
                device=str(dev), tf32=False, gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None))


def _thr_brief(sel):
    return {w: dict(t_known=sel[w]['t_known'], j_known=sel[w]['j_known'], fpr=sel[w]['fpr'], fpr_max=sel[w]['fpr_max'], t_occ=sel[w]['t_occ'],
                    j_occ=sel[w]['j_occ'], f1=sel[w]['f1'], free=sel[w]['free'], free_min=sel[w]['free_min'], feasible=sel[w]['feasible_known'] and sel[w]['feasible_occ'],
                    control_furn_f1=sel[w]['control']['furn_f1']) for w in ('R8', 'K0')} | dict(holdout_houses=sel['n_houses'], device=sel['device'], tf32=sel['tf32'])


def select_cmd(args):
    dev = torch.device(args.device)
    if dev.type == 'cpu':
        torch.set_num_threads(max(1, args.threads))
    run = _jload(os.path.join(args.run, 'run.json'))
    ck, model = load_model(os.path.join(args.run, 'checkpoints', 'final.pt'), dev)
    ev = R.Evidence.from_recipe(ck['recipe']['evidence'])
    cache = args.cache or run['cache']
    man = _jload(os.path.join(cache, 'manifest.json'))
    ho = load_cache(cache, man, 'holdout')
    cc = args.ctrl_ckpt or (run.get('control') or {}).get('ckpt'); ct = args.ctrl_threshold or (run.get('control') or {}).get('threshold')
    assert cc and ct, 'no paired E3-R checkpoint / threshold given'
    tp = os.path.join(args.run, 'threshold.json')
    if os.path.exists(tp) and not args.overwrite:
        raise SystemExit(f'{tp} exists (use --overwrite to replace it)')
    ts = time.time()
    sel = select_run(model, ev, ho, dev, cc, ct)
    sel['seconds'] = time.time() - ts
    with open(tp, 'w') as fh:
        json.dump(sel, fh, indent=1)
    print(json.dumps(_thr_brief(sel)), flush=True)
    return sel


# ================================================================ eval (CPU, after the lease): E3-R export, every agent decodes its own belief
def msg_ints(hm, lm):
    li = torch.round(lm / DELTA_LAM).to(torch.int64).cpu().numpy(); hi = torch.round(hm / DELTA_H).to(torch.int64).cpu().numpy()
    assert (li >= 0).all() and not hi[:, li == 0].any(), 'zero-precision cell carries h != 0'
    return hi, li


def int_bytes(hi, li):
    """REPORT ONLY (upper bound, superseded by wire_bytes): 9 channels each by G.int_ctx_bytes(channel, Lam > 0), i.e. the mask 9 times"""
    kn = li > 0
    return float(G.int_ctx_bytes(li, kn) + sum(G.int_ctx_bytes(hi[d], kn) for d in range(hi.shape[0])))


def sign_bits_given_mask(v, known):
    """the sign stream (v > 0) of the known cells, the mask GIVEN: the contexts of G.ctx_tern_bits (ternary states of the 9 causal
    neighbours G.NB, out of grid / outside the mask = unknown, all known to the decoder) with a binary KT estimator per context,
    counted over the known cells only (= the ternary part of G.int_ctx_bytes without paying for the mask again)"""
    known = np.asarray(known, bool); v = np.asarray(v)
    t = np.where(known, (v > 0).astype(np.int8), np.int8(-1))
    X, Y, Z = t.shape
    s = (t.astype(np.int16) + 1).astype(np.uint16)
    c = np.zeros(t.shape, np.uint16)
    for dx, dy, dz in G.NB:
        c *= 3; c[dx:, dy:, dz:] += s[:X - dx, :Y - dy, :Z - dz]
    ck = c[known].astype(np.int64); sk = (v > 0)[known]; nc = 3 ** len(G.NB)
    n = np.bincount(ck, minlength=nc); n1 = np.bincount(ck[sk], minlength=nc)
    return G.kt_bin_bits(n - n1, n1)


def value_bytes_given_mask(v, known):
    """one channel's values inside the support, the mask given: sign stream (above) + G.int_magnitude_events streams, + 4 B header"""
    e0, e1, p0, p1, s0, s1 = G.int_magnitude_events(v, known)
    return (sign_bits_given_mask(v, known) + G.kt_bin_bits(e0, e1) + G.kt_bin_bits(p0, p1) + G.kt_bin_bits(s0, s1)) / 8 + 4


def mask_bytes(known):
    """the support mask once: G.ctx_tern_bytes of the 0 / 1 map (G4-E1 ternary ctx coder; only two symbols occur), + 2 B header"""
    return float(G.ctx_tern_bytes(np.asarray(known, bool).astype(np.int8)))


def wire_bytes(hi, li):
    """O7 (D8-54 supplement 1, the VERDICT bytes): mask (Lam_q > 0) once + the 9 channels' values inside the support given the mask"""
    kn = li > 0
    return mask_bytes(kn) + float(value_bytes_given_mask(li, kn) + sum(value_bytes_given_mask(hi[d], kn) for d in range(hi.shape[0])))


CH_LIM = 64


def chan_hist(hi, li):
    """per channel histogram of the integer symbols on the supported cells (Lam > 0), clipped to [-64, 64]"""
    kn = li > 0; out = np.zeros((hi.shape[0] + 1, 2 * CH_LIM + 1), np.int64)
    out[0] = np.bincount(np.clip(li[kn], -CH_LIM, CH_LIM) + CH_LIM, minlength=2 * CH_LIM + 1)
    for d in range(hi.shape[0]):
        out[d + 1] = np.bincount(np.clip(hi[d][kn], -CH_LIM, CH_LIM) + CH_LIM, minlength=2 * CH_LIM + 1)
    return out


@torch.no_grad()
def eval_arm(model, thr, table, L_k0, L_pub, F, H, K0T, R8T, chunk, want_bytes=True):
    table_t = torch.as_tensor(table, dtype=torch.int32); w = float(model.w.detach()); n = len(L_k0)
    zp, lp = encode_maps(model, L_pub, table_t, 'cpu')
    same = all(a is b for a, b in zip(L_pub, L_k0))
    zk, lk = (zp, lp) if same else encode_maps(model, L_k0, table_t, 'cpu')
    hq, lq, _, _ = model.source(zp, lp, 'round')
    hk, lkq = (hq, lq) if same else model.source(zk, lk, 'round')[:2]
    jk, jo = thr['R8']['j_known'], thr['R8']['j_occ']; jk0, jo0 = thr['K0']['j_known'], thr['K0']['j_occ']
    r8 = []; by = 0.0; byu = 0.0; chk = dict(cf_max=0.0, cf_rel_max=0.0, cf_scale_max=0.0, violations=0, components=0, steps=0, cf_cells=0, cf_excluded_tiny=0)
    qd = np.zeros(4); dis = np.zeros(2); lost = 0
    known_frac = []
    for ci, cm in enumerate(F['comps']):
        Fl = local_forest(F, cm)
        hb, lb, msgs = gabp(hq[cm], lq[cm], Fl, w, 'round')
        if want_bytes:
            for e in Fl['directed']:
                mi = msg_ints(*msgs[e]); by += wire_bytes(*mi); byu += int_bytes(*mi)
        hf, lf, _ = gabp(lp[cm].unsqueeze(1) * zp[cm], lp[cm], Fl, w, None)
        xq = belief_mean(hb, lb); xf = belief_mean(hf, lf)
        sel = lb > 0                                                      # cells the quantised wire informs
        if sel.any():
            dd = (xq - xf).abs().amax(1)[sel]
            qd += [float(dd.sum()), float(dd.max()), float(sel.sum()), float(xf.abs().amax(1)[sel].sum())]
        lost += int(((lb == 0) & (lf >= 0.5 * DELTA_LAM)).sum())       # float precision >= half a step, dropped by the wire
        c = gbp_checks(zp[cm], lp[cm], Fl, w)
        chk['cf_max'] = max(chk['cf_max'], c['cf_max']); chk['violations'] += c['violations']; chk['components'] += 1; chk['steps'] += c['steps']
        chk['cf_rel_max'] = max(chk['cf_rel_max'], c['cf_rel']); chk['cf_scale_max'] = max(chk['cf_scale_max'], c['cf_scale'])
        chk['cf_cells'] += c['cf_cells']; chk['cf_excluded_tiny'] += c['cf_excluded_tiny']
        for i in range(len(cm)):                                         # inter-agent belief disagreement (cells both informed)
            for j in range(i + 1, len(cm)):
                both = (lb[i] > 0) & (lb[j] > 0)
                if both.any():
                    dis += [float(((xq[i] - xq[j]) ** 2).sum(0)[both].sum()), float(both.sum())]
        lg = decode_np(model, hb, lb, chunk)
        kc = sum((L_pub[a] != UNK).astype(np.int32) for a in cm)
        for i, a in enumerate(cm):
            p = readout(lg[i][0], lg[i][1], jk, jo)
            r8.append([R.scored(p, R8T[ci], H, kc), 1]); known_frac.append(float((p >= 0).mean()))
    lgk = decode_np(model, hk, lkq, chunk)
    k0, k0s = [], []
    for a in range(n):
        ki = (L_k0[a] != UNK).astype(np.int32)
        k0.append(R.scored(readout(lgk[a][0], lgk[a][1], jk, jo), K0T[a], H, ki))
        k0s.append(R.scored(readout(lgk[a][0], lgk[a][1], jk0, jo0), K0T[a], H, ki))
    rate = []; ch = np.zeros((D_LAT + 1, 2 * CH_LIM + 1), np.int64)
    for a in range(n):                                                    # source code lengths: measured (O7) vs proxy vs upper bound (O8)
        hi, li = msg_ints(hq[a], lq[a])
        sym = torch.from_numpy(np.concatenate([li[None], hi]).reshape(1, D_LAT + 1, -1).astype(np.float32))
        sb, vb = model.ent.bits_parts(sym, torch.from_numpy((li > 0).reshape(1, -1).astype(np.float32)))
        prox = float(sb[0]) + float(vb[0].sum())
        rate.append(dict(measured_bits=8 * wire_bytes(hi, li), proxy_bits=prox, proxy_support_bits=float(sb[0]), proxy_bits_by_channel=vb[0].tolist(),
                         upper_bits=8 * int_bytes(hi, li), mask_bits=8 * mask_bytes(li > 0), cells=int((li > 0).sum())))
        ch += chan_hist(hi, li)
    return dict(R8=r8, K0=k0, K0_sep=k0s, bytes=(by if want_bytes else None), bytes_upper=(byu if want_bytes else None), gbp=chk, w=w,
                qdiff=dict(sum_absmax=qd[0], max=qd[1], cells=qd[2], sum_float_absmax=qd[3], lost_cells=lost), disagree=dict(sum_sq=dis[0], cells=dis[1]),
                rate=rate, channels=ch.tolist(), known_frac_r8=known_frac)


def eval_job(job):
    rec_e, opts = job
    t0 = time.time()
    rec = dict(stage='eval', panel=rec_e['panel'], rooms=rec_e['rooms'], index=rec_e['index'], seed=rec_e['seed'], split=rec_e.get('split'),
               e1_match=rec_e.get('e1_match'), script_sha256=opts['script_sha256'], ckpts=opts['ckpt_shas'])
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
        models = []
        for p in opts['ckpts']:
            ck, model = load_model(p, 'cpu'); thr = _jload(os.path.join(os.path.dirname(os.path.dirname(p)), 'threshold.json'))
            models.append((str(ck['seed']), model, thr, np.asarray(ck['recipe']['evidence']['table'], np.int64)))
        rec['tiers'] = {}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            T = {}
            for sd, model, thr, table in models:
                T[f'{ARM}_{sd}'] = eval_arm(model, thr, table, L_k0, L_pub, F, H, K0T, R8T, opts['chunk'])
            rec['tiers'][tn] = T
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _eval_and_save(job):
    rec = eval_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json")
    with open(p + '.tmp', 'w') as fh:
        json.dump(rec, fh)
    os.replace(p + '.tmp', p)
    return rec


# ================================================================ aggregation and the D8-54 verdict (criteria hard-coded)
D8_54 = dict(prereg='D8-54', judged_panel='val', report_panel='fresh', judged_tier='cam_nf', cam_tier='cam', clean_tier='clean',
             train_seeds=[str(s) for s in R.TRAIN_SEEDS], noise_seeds=[str(s) for s in R.EVAL_SEEDS], control_arm='NET_{seed} (E3-R summary)',
             lever_furn_margin=-0.010,          # R8 furniture F1 >= control - 1.0 pp (4 cells)
             lever_bytes_ratio=0.15,            # median bytes / house <= 0.15 x control (sum + count medians, ~753 KB)
             guard_free=0.005, guard_surface=0.005,   # free recall / surface F1 >= control - 0.5 pp
             guard_fpr=0.003,                   # unknown->known FPR <= control + 0.3 pp
             clean_min=0.99,                    # clean-tier R8 furniture >= .99
             k0_margin=-0.03,                   # K0 furniture (own K0 thresholds) >= control K0_sep - 3 pp
             gbp_cf_tol=CF_TOL,                 # GaBP == closed form, energy monotone: all checks pass
             cam_demote_margin=-0.03,           # cam tier 4-cell mean < control - 3 pp -> LEVER demoted to AMBIGUOUS
             not_enough_furn_margin=-0.03, not_enough_bytes_ratio=0.5,
             min_updates=MIN_UPDATES, truncated_blocks_lever=True, fresh_required=True,
             fresh_min_mean=-0.010,             # D8-54 supplement 4: fresh 4-cell mean d_furn >= -1.0 pp, else LEVER -> AMBIGUOUS (never upgrades)
             known_select_fpr=GUARD_FPR_SELECT,  # D8-54 supplement 3 (selection only; the guard stays guard_fpr)
             supplement='D8-54 supplement (coordinator, 2026-10-27): 1 bytes = mask once + in-support values; 2 R* on the measured wire '
                        'code length; 3 known threshold at control + 0.15 pp; 4 fresh mean >= -1 pp demotes LEVER only')


def _mean(v):
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None


def _ent(h):
    h = np.asarray(h, np.float64); s = h.sum()
    if s <= 0:
        return None
    p = h[h > 0] / s; return float(-(p * np.log2(p)).sum())


def aggregate(out_dir):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             ckpts=sorted({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}), gbp={})
    reasons = []
    if len(S['script_sha256']) > 1 or len(S['ckpts']) > 1:
        reasons.append('mixed script / checkpoint versions in the job files')
    for r in recs:
        tag = f"{r.get('panel')}_r{r.get('rooms')}_i{r.get('index')}_s{r.get('seed')}"
        if r.get('failed'):
            S['failed'].append(tag)
        elif r.get('skipped'):
            S['skipped'].append(tag)
        elif r.get('split') == 'VALIDATION' and (not r.get('e1_match') or any(v is not True for v in r['e1_match'].values())):
            S['e1_mismatch'].append(tag)
    ok = [r for r in recs if not r.get('failed') and not r.get('skipped')]
    for r in ok:                                                          # GaBP checks over every job / tier / arm
        for tn, T in r['tiers'].items():
            for arm, a in T.items():
                g = S['gbp'].setdefault(arm, dict(cf_max=0.0, cf_rel_max=0.0, cf_scale_max=0.0, violations=0, components=0, steps=0, jobs=0, cf_cells=0,
                                                  cf_excluded_tiny=0))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['violations'] += a['gbp']['violations']
                g['cf_rel_max'] = max(g['cf_rel_max'], a['gbp'].get('cf_rel_max', 0.0)); g['cf_scale_max'] = max(g['cf_scale_max'], a['gbp'].get('cf_scale_max', 0.0))
                g['components'] += a['gbp']['components']; g['steps'] += a['gbp']['steps']; g['jobs'] += 1
                g['cf_cells'] += a['gbp'].get('cf_cells', 0); g['cf_excluded_tiny'] += a['gbp'].get('cf_excluded_tiny', 0)
    for arm, g in S['gbp'].items():
        g['pass'] = bool(g['cf_max'] <= CF_TOL and g['violations'] == 0 and g['components'] > 0)
    for panel in sorted({r['panel'] for r in ok}):
        P = dict(seeds={})
        for seed in sorted({r['seed'] for r in ok if r['panel'] == panel}):
            X = [r for r in ok if r['panel'] == panel and r['seed'] == seed]
            D = dict(houses=sorted(f"r{r['rooms']}_i{r['index']}" for r in X), tiers={})
            for tn in R.EVAL_TIERS:
                if not X or not all(tn in r['tiers'] for r in X):
                    continue
                d = {}
                for arm in sorted(X[0]['tiers'][tn]):
                    if not all(arm in r['tiers'][tn] for r in X):
                        continue
                    A = [r['tiers'][tn][arm] for r in X]
                    e = dict(K0=R._summ([x for a in A for x in a['K0']]), K0_sep=R._summ([x for a in A for x in a['K0_sep']]),
                             R8=R._summ([x for a in A for x in G._expand(a['R8'])]))
                    if all(a.get('bytes') is not None for a in A):
                        e['bytes_median'] = float(np.median([a['bytes'] for a in A])); e['bytes_by_house'] = {f"r{r['rooms']}_i{r['index']}": r['tiers'][tn][arm]['bytes'] for r in X}
                        e['bytes_rule'] = 'D8-54 supplement 1: mask once (G.ctx_tern_bytes) + in-support values given the mask, all directed edges'
                    if all(a.get('bytes_upper') is not None for a in A):
                        e['report_bytes_upper_median'] = float(np.median([a['bytes_upper'] for a in A]))
                    q = np.sum([[a['qdiff']['sum_absmax'], a['qdiff']['cells'], a['qdiff']['sum_float_absmax']] for a in A], 0)
                    e['quant_belief_diff'] = dict(mean_abs=q[0] / max(q[1], 1), max_abs=max(a['qdiff']['max'] for a in A), mean_abs_float=q[2] / max(q[1], 1),
                                                  informed_cells=float(q[1]), lost_cells=int(sum(a['qdiff'].get('lost_cells', 0) for a in A)),
                                                  scope='cells with quantised Lam_b > 0; lost = float Lam_b >= 0.5 step but quantised 0')
                    ds = np.sum([[a['disagree']['sum_sq'], a['disagree']['cells']] for a in A], 0)
                    e['belief_disagreement_rms'] = float(math.sqrt(ds[0] / ds[1])) if ds[1] else None
                    rt = [x for a in A for x in a['rate']]
                    e['rate'] = dict(measured_bits_median=float(np.median([x['measured_bits'] for x in rt])), r_star=R_STAR,
                                     proxy_bits_median=float(np.median([x['proxy_bits'] for x in rt])),
                                     measured_over_proxy_median=float(np.median([x['measured_bits'] / max(x['proxy_bits'], 1e-9) for x in rt])),
                                     upper_bits_median=float(np.median([x['upper_bits'] for x in rt])), mask_bits_median=float(np.median([x['mask_bits'] for x in rt])),
                                     cells_median=float(np.median([x['cells'] for x in rt])),
                                     proxy_bits_by_channel_mean=np.mean([x['proxy_bits_by_channel'] for x in rt], 0).tolist())
                    ch = np.sum([a['channels'] for a in A], 0)
                    e['channels'] = [dict(name=('lam' if c == 0 else f'h{c - 1}'), entropy_bits=_ent(ch[c]), usage_nonzero=(1.0 - ch[c][CH_LIM] / max(ch[c].sum(), 1)))
                                     for c in range(ch.shape[0])]
                    e['w'] = float(np.mean([a['w'] for a in A])); e['gbp'] = dict(cf_max=max(a['gbp']['cf_max'] for a in A), violations=sum(a['gbp']['violations'] for a in A))
                    d[arm] = e
                D['tiers'][tn] = d
            P['seeds'][str(seed)] = D
        S['panels'][panel] = P
    S['agg_reasons'] = reasons
    return S


def _ctrl_arm(Sc, panel, ns, tier, ts):
    try:
        return Sc['panels'][panel]['seeds'][ns]['tiers'][tier][f'NET_{ts}'], Sc['panels'][panel]['seeds'][ns]['houses']
    except (KeyError, TypeError):
        return None, None


def _cell(S, Sc, panel, ts, ns, C, reasons, where, judged):
    P = S['panels'].get(panel); arm = f'{ARM}_{ts}'
    D = (P or {}).get('seeds', {}).get(ns)
    if D is None or C['judged_tier'] not in D['tiers'] or arm not in D['tiers'][C['judged_tier']]:
        reasons.append(f'{where}: {arm} {C["judged_tier"]} row missing'); return None
    a = D['tiers'][C['judged_tier']][arm]
    c, ch = _ctrl_arm(Sc, panel, ns, C['judged_tier'], ts)
    if c is None:
        reasons.append(f'{where}: control NET_{ts} missing in the E3-R summary'); return None
    if ch != D['houses']:
        reasons.append(f'{where}: house set differs from the control'); return None
    cb = c['rnet_relay_median']['sum'] + c['rnet_relay_median']['count']
    cell = dict(e3l_R8_furn=a['R8']['furn_f1'], control_R8_furn=c['R8']['furn_f1'], d_furn=a['R8']['furn_f1'] - c['R8']['furn_f1'],
                bytes_median=a.get('bytes_median'), control_bytes=cb, bytes_ratio=(a['bytes_median'] / cb) if a.get('bytes_median') is not None else None)
    if cell['bytes_ratio'] is None:
        reasons.append(f'{where}: bytes missing'); return None
    if not judged:
        return cell
    cell.update(d_furn_tol1=a['R8']['furn_f1_tol1'] - c['R8']['furn_f1_tol1'], free_delta=a['R8']['free_recall'] - c['R8']['free_recall'],
                surface_delta=a['R8']['surface_f1'] - c['R8']['surface_f1'], fpr_delta=a['R8']['unknown_known_fpr'] - c['R8']['unknown_known_fpr'],
                e3l_K0_sep=a['K0_sep']['furn_f1'], control_K0_sep=c['K0_sep']['furn_f1'], d_K0=a['K0_sep']['furn_f1'] - c['K0_sep']['furn_f1'],
                report_e3l_K0_shared=a['K0']['furn_f1'], w=a.get('w'), quant_belief_diff=a.get('quant_belief_diff'), rate=a.get('rate'))
    cl = D['tiers'].get(C['clean_tier'], {}).get(arm); cam = D['tiers'].get(C['cam_tier'], {}).get(arm)
    cc, _ = _ctrl_arm(Sc, panel, ns, C['cam_tier'], ts)
    if cl is None or cam is None or cc is None:
        reasons.append(f'{where}: clean / cam row (or the cam control) missing'); return None
    cell['clean_R8_furn'] = cl['R8']['furn_f1']; cell['cam_d_furn'] = cam['R8']['furn_f1'] - cc['R8']['furn_f1']; cell['cam_R8_furn'] = cam['R8']['furn_f1']
    g = dict(free=cell['free_delta'] >= -C['guard_free'] - 1e-12, surface=cell['surface_delta'] >= -C['guard_surface'] - 1e-12,
             fpr=cell['fpr_delta'] <= C['guard_fpr'] + 1e-12, clean=cl['R8']['furn_f1'] >= C['clean_min'], k0=cell['d_K0'] >= C['k0_margin'] - 1e-12)
    cell['guards'] = g; cell['guards_pass'] = all(g.values())
    return cell


def verdict(S, runs=None, control_summary=None, smoke=False):
    """D8-54: LEVER iff in all 4 VAL cam_nf cells d_furn >= -1 pp, bytes <= .15 x control and every guard passes, the GaBP checks all
    pass and no training seed is truncated; a LEVER whose cam-tier 4-cell mean d < -3 pp is demoted to AMBIGUOUS; NOT_ENOUGH iff any
    cell d_furn < -3 pp or bytes > .5 x control; else AMBIGUOUS. Fresh (D8-54 supplement 4, non-inferiority, demotion only): if the
    fresh 4-cell mean d_furn < -1.0 pp, a LEVER becomes AMBIGUOUS; nothing else changes and nothing is ever upgraded; the literal
    'fresh sign opposite to VAL' is computed and reported only.
    INCOMPLETE: failed jobs, E1 mismatch, missing run / final / threshold, < 2,000 updates, different update counts, any missing VAL /
    fresh cell, house, control row or tier, no GaBP checks."""
    C = dict(D8_54)
    reasons = list(S.get('agg_reasons', []))
    if S.get('failed'):
        reasons.append(f"failed jobs: {S['failed']}")
    if S.get('e1_mismatch'):
        reasons.append(f"E1 reproduction mismatch: {S['e1_mismatch']}")
    Sc = None
    if control_summary is None:
        reasons.append('control summary (E3-R) not given')
    else:
        Sc = _jload(control_summary)
    run_info = {}; truncated = []
    if runs is None:
        reasons.append('training runs not given')
    else:
        for r in runs:
            sd, info = R._run_info(r, C, reasons, truncated)
            if sd is not None and info is not None:
                run_info[sd] = info
        for sd in C['train_seeds']:
            if sd not in run_info:
                reasons.append(f'training seed {sd} missing')
        if len({i['updates_planned'] for i in run_info.values()}) > 1:
            reasons.append(f"training seeds have different update counts {sorted({i['updates_planned'] for i in run_info.values()})}")
    for ts in C['train_seeds']:
        g = S.get('gbp', {}).get(f'{ARM}_{ts}')
        if g is None:
            reasons.append(f'no GaBP checks for {ARM}_{ts}')
    cells, fresh = {}, {}
    if Sc is not None:
        for panel, store, judged in ((C['judged_panel'], cells, True), (C['report_panel'], fresh, False)):
            P = S['panels'].get(panel)
            if P is None:
                reasons.append(f'panel {panel} missing'); continue
            R._panel_houses(S, P, panel, C, reasons, smoke)
            for ts in C['train_seeds']:
                for ns in C['noise_seeds']:
                    c = _cell(S, Sc, panel, ts, ns, C, reasons, f'{panel} cell (train {ts}, noise {ns})', judged)
                    if c is not None:
                        store[f'{ts}|{ns}'] = c
    notes = []; fresh_info = None
    if reasons:
        v = 'INCOMPLETE: ' + '; '.join(reasons)
    else:
        gbp_ok = all(S['gbp'][f'{ARM}_{ts}']['pass'] for ts in C['train_seeds'])
        lever_cells = all(c['d_furn'] >= C['lever_furn_margin'] - 1e-12 and c['bytes_ratio'] <= C['lever_bytes_ratio'] + 1e-12 and c['guards_pass']
                          for c in cells.values())
        lever = lever_cells and gbp_ok and not (C['truncated_blocks_lever'] and truncated)
        if lever_cells and not gbp_ok:
            notes.append('lever blocked: a GaBP exactness / energy-monotonicity check failed')
        if lever_cells and gbp_ok and not lever:
            notes.append(f'lever blocked: truncated training seed(s) {truncated}')
        cam_mean = float(np.mean([c['cam_d_furn'] for c in cells.values()]))
        notenough = any(c['d_furn'] < C['not_enough_furn_margin'] or c['bytes_ratio'] > C['not_enough_bytes_ratio'] for c in cells.values())
        if lever and cam_mean < C['cam_demote_margin']:
            v = f'AMBIGUOUS (lever demoted: cam-tier 4-cell mean d {cam_mean:+.4f} < {C["cam_demote_margin"]})'
        else:
            v = 'LEVER' if lever else ('NOT_ENOUGH' if notenough else 'AMBIGUOUS')
        if fresh:                                                        # D8-54 supplement 4 (the literal sign is reported only)
            mv = float(np.mean([c['d_furn'] for c in cells.values()])); mf = float(np.mean([c['d_furn'] for c in fresh.values()]))
            fresh_info = dict(val_mean_d_furn=mv, fresh_mean_d_furn=mf, fresh_ok=bool(mf >= C['fresh_min_mean'] - 1e-12),
                              report_literal_sign_opposite=bool(mv * mf < 0))
            if v == 'LEVER' and not fresh_info['fresh_ok']:
                v = f'AMBIGUOUS (lever demoted: fresh 4-cell mean d_furn {mf:+.4f} < {C["fresh_min_mean"]})'
        notes.append(f'cam-tier 4-cell mean d_furn {cam_mean:+.4f}')
    return dict(verdict=v, criteria=C, cells=cells, fresh_cells=fresh, fresh=fresh_info, runs=run_info, truncated_seeds=truncated, notes=notes, incomplete_reasons=reasons,
                gbp=S.get('gbp'), control_summary=(dict(path=os.path.abspath(control_summary), sha256=file_sha256(control_summary)) if control_summary else None))


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'select', 'eval', 'aggregate', 'selfcheck', 'fingerprint'):
        p = sub.add_parser(name)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=1)
        if name == 'train':
            p.add_argument('--cache', required=True); p.add_argument('--evidence', required=True); p.add_argument('--seed', type=int, required=True)
            p.add_argument('--out', required=True); p.add_argument('--updates', type=int, required=True); p.add_argument('--max-updates', type=int, default=0)
            p.add_argument('--device', default='cuda'); p.add_argument('--batch', type=int, default=8); p.add_argument('--width', type=int, default=16)
            p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--wd', type=float, default=1e-4); p.add_argument('--warmup', type=int, default=300)
            p.add_argument('--clip', type=float, default=1.0); p.add_argument('--log-every', type=int, default=100); p.add_argument('--val-every', type=int, default=1000)
            p.add_argument('--holdout-samples', type=int, default=64); p.add_argument('--deadline-epoch', type=float, default=0)
            p.add_argument('--reserve', type=float, default=300.); p.add_argument('--select', type=int, default=1); p.add_argument('--manifest-sha', default='')
            p.add_argument('--ctrl-ckpt', default=''); p.add_argument('--ctrl-threshold', default='')
            p.add_argument('--rate-every', type=int, default=RATE_EVERY)
        if name == 'select':
            p.add_argument('--run', required=True); p.add_argument('--device', default='cpu'); p.add_argument('--cache', default='')
            p.add_argument('--ctrl-ckpt', default=''); p.add_argument('--ctrl-threshold', default=''); p.add_argument('--overwrite', action='store_true')
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--ckpt', required=True)
            p.add_argument('--chunk', type=int, default=2); p.add_argument('--only', default='', help='comma list of job tags (--smoke only)')
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--runs', default=''); p.add_argument('--control-summary', default='')
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True)
        if name == 'fingerprint':
            p.add_argument('--npz', required=True); p.add_argument('--write', default=''); p.add_argument('--expect', default='')
            p.add_argument('--arrays', default='L_k0_cam_nf,L_k0_cam,L_pub_cam')
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    sha = file_sha256(os.path.abspath(__file__))
    t0 = time.time()
    if args.cmd == 'train':
        return train(args)
    if args.cmd == 'select':
        return select_cmd(args)
    if args.cmd == 'selfcheck':
        assert os.path.dirname(os.path.abspath(R.__file__)) == HERE, f'g4_e3r imported from {R.__file__}, not {HERE}'
        return R.main(['selfcheck', '--pyroot', args.pyroot, '--houseexpo-json-dir', json_dir, '--houseexpo-index', index_file])
    if args.cmd == 'fingerprint':
        a = ['fingerprint', '--npz', args.npz, '--arrays', args.arrays] + (['--write', args.write] if args.write else []) + (['--expect', args.expect] if args.expect else [])
        return R.main(a)
    if args.cmd == 'eval':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        ckpts = [p for p in args.ckpt.split(',') if p]
        ck_shas = {p: file_sha256(p) for p in ckpts}
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', '*.json')))]
        assert recs, f'no export records in {args.export}/records'
        if args.only:
            assert args.smoke, '--only is for the smoke'
            want = set(args.only.split(','))
            recs = [r for r in recs if f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}" in want]
        for p in ckpts:
            assert os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'threshold.json')), f'{p}: no threshold.json'
        opts = dict(export=args.export, out=args.out, ckpts=ckpts, ckpt_shas=ck_shas, tiers=list(R.EVAL_TIERS), threads=args.threads, chunk=args.chunk,
                    script_sha256=sha, smoke=bool(args.smoke))
        os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
        for r in recs:
            if r.get('failed') or r.get('skipped'):
                with open(os.path.join(args.out, 'jobs', f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.json"), 'w') as fh:
                    json.dump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                                   skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))), fh)
        jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
        print(f'E3-L eval: {len(jobs)} jobs, checkpoints {ckpts}, {args.workers} workers x {args.threads} threads', flush=True)
        for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
                  f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
        return None
    if args.cmd == 'aggregate':
        S = aggregate(args.out)
        runs = [r for r in args.runs.split(',') if r] or None
        S['decision'] = verdict(S, runs, args.control_summary or None, smoke=args.smoke)
        with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
            json.dump(S, fh, indent=1)
        print('E3-L verdict:', S['decision']['verdict'], flush=True)
        for k, c in S['decision']['cells'].items():
            print(f"  cell {k}: E3-L R8 furn {c['e3l_R8_furn']:.4f} vs NET {c['control_R8_furn']:.4f} (d {c['d_furn']:+.4f}); bytes {c['bytes_median']:.0f} "
                  f"= {c['bytes_ratio']:.3f} x control; cam d {c['cam_d_furn']:+.4f}; guards {c['guards']}", flush=True)
        print('  fresh:', json.dumps(S['decision'].get('fresh')), flush=True)
        print('  gbp:', json.dumps(S['decision']['gbp']), flush=True)
        print('  runs:', json.dumps(S['decision']['runs']), flush=True)
        return S


if __name__ == '__main__':
    main()
