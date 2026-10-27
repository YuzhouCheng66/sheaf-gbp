"""g4_e3l2.py (Claude, 2026-10-27): 批次 E3-L2 "sheaf 隐码 GBP，硬容量码率"（预注册 OVERNIGHT_REPORT D8-59；5090 第 3 个租约）。
冻结依赖只读 import：g4_e3l.py（E3-L，sha 3dc970d8...，GaBP / 编解码器 / 量化 / O7 线路 / 计分 / 阈值原语 / 对照读数全部复用）、
g4_e3r.py（1d65d9a8...）、g4_noise_panels.py（2313d75b...）。本文件不改它们。

底线（D8-54 / D8-59，不变）：学习隐码 x_i；二次 sheaf 能量 E = sum_i sum_c 1/2 Lam_i(c)||x_i(c) - z_i(c)||^2 + sum_(i,j) in F sum_c
1/2 w ||x_i(c) - x_j(c)||^2 上的两遍 GaBP（L.gabp，消息公式与 E3-L 逐字相同）；每台机器只解码自己的信念；融合与任务无关。
单变量（相对 E3-L）：字节预算的施加方式 = 每机源端只保留 Lam 最大的 K 个触及格（硬容量），无 lambda、无熵模型。

操作化（D8-59 没写死、由本实现写死的部分；回复里逐条列出）
  P1 模型 = E3-L 的 Enc / Dec / w（同一类、同一初始化顺序：enc、dec、log_w；E3L2 不含熵模型），Lam = softplus(a) x 1[触及块]，
     a 用 E3-L 的单侧直通截断（>= -8）。块 4x4x2、D = 8、w = exp(omega)、w0 = 1000，都来自 L。
  P2 源端 top-K（source_topk）：每台机器在触及格（Lam > 0）里按连续 Lam 取最大的 K 个（torch.topk，同值时按 topk 的确定性次序），
     其余格 Lam = 0、h = 0（能量里就是"无数据项"）。选中格（审查 M1 修订）：Lam_c = clamp(Lam, 1, 15)（单侧直通），h = Lam_c z
     （信息形式 h = Lam_q z），Lam_q = round(Lam_c)（与 clamp(round(Lam), 1, 15) 逐位相同），h_q = clamp(round(h), -7, 7)，h 在
     +-7.5 处单侧直通。单侧直通 = 越界时只放行把值拉回区间的梯度（E3-L 审查 S1 的 A_FLOOR 先例）。未选的触及格：Lam 的反向
     恒等（= E3-L 的门直通，让格子学会"进不进 top-K"），h 不回传；选择本身是硬的。浮点势（qmode None：闭式核对、浮点信念）
     同样用 Lam_c。消息仍是 E3-L 的量化 GaBP，不限幅（"消息不截"）。clamp=False 且 K >= 触及数时与 E3-L 的源端逐位相同。
     修订理由（审查 M1）：原实现 h = Lam_raw z 且限幅两侧都恒等直通，没有回拉力，编码器会走进上饱和（Lam、h 外漂，梯度爆炸，
     隐码退化为符号位）或下塌缩（Lam_raw < 0.5 时 h_q 恒 0，编码器梯度为 0）；修订后选择规则与 Lam_q 逐位不变，h_q 只在
     原始 Lam 落在 [1, 15] 之外的选中格上不同。
  P3 训练的 K：每个样本 K = round(exp(U[log 128, log 2048]))，用数据 rng 抽，样本里所有机器同一个 K；留出曲线的样本也各带一个 K（hrng）。
  P4 固定形状编码器（程序修正）：一个 batch 的全部机器按 8 台一块送进编码器，最后一块补零（特征全 0、无触及体素，所以 Lam = 0），
     补的行在 top-K / GaBP 之前就丢掉、不进 GaBP、不贡献梯度；解码器 batch 固定为 --batch（8）。cudnn.benchmark 保持 True。
  P5 无 lambda、无熵模型损失；损失 = 已知头 BCE（全网格）+ 占据头 BCE（目标已知区），与 E3-L 相同。每 50 次更新（及第 1 次）
     在当前 batch 上实测被解码机器源端消息的 O7 码长（只报告，rate_curve.json：K、选中格、触及格、比特；审查 S6 另加：选中格
     h_q != 0 的占比（格 x 通道）、选中格原始 Lam > 15 / < 1 的占比、|Lam_c z| > 7.5 的占比、选中格原始 Lam 的中位、编码器梯度
     范数（裁剪前）），训练日志每行同样打印。
  P6 第 6,510 次更新另存 checkpoints/upd6510.pt（同格式；只报告，等步数归因）。
  P7 测速：每 --log-every 记一个时间点；每步耗时 = 起点 >= --time-from（200）的窗口合计；窗口耗时的变异系数 >= 10 % 时记警告并改用
     窗口中位数；所用窗口的起点 < --time-from（窗口不够时的退路）也记警告（审查 S5；进 final.json 的 rate_warning，链的预跑日志行
     原样打印它）。链的预跑：400 次更新、--log-every 50、--time-from 200（用 4 个窗口）；更新数 = floor(0.90 x avail / rate)，
     avail = 3000 - 60（余量）- 加载 - 首窗（50 次更新 + cudnn 调优）- 60（留出曲线）；训练器 --reserve 60（选阈值已移出租约）。
  P8 留出房对照（ctrlprep，只用 TRAIN，可在发射前算好并钉 sha）：19 个留出房 x 抽样 200-202（cam_nf，机体系 = 公共系）：
     每个 (房, 抽样) 的 E3-R 对照字节 = E3-R relay 的 sum + count 消息的 G.int_ctx_bytes 全部有向边之和（D8-54 口径，与 E3-R
     summary 的 rnet_relay_median 同一算法），按配对种子的证据表（342101 = 表 A，342202 = 表 B）各算一份；配对 E3-R NET（冻结
     检查点与阈值）的合并空闲召回 / FPR / 家具计数（L.control_holdout：R8 分量展开，K0 用 NET 的 K0 阈值）。
  P9 选 K（kselect，租约外 CPU；kselect 与 select 先断言对照文件的 manifest sha 等于当前 cache、其 E3-R 检查点与阈值文件的 sha
     未变、且等于训练时配对断言记下的检查点 sha，审查 S1）：网格 {128, 192, 256, 384, 512, 768, 1024, 1536, 2048}；每个 (房, 抽样) 的比 = E3-L2 每房字节
     （O7 线路口径，全部有向消息之和）/ 同房同抽样的配对对照字节；K* = 这些比的中位 <= 0.12 的最大 K；没有任何 K 满足 -> kselect 的
     K* 记为缺失（判 INCOMPLETE），select 改在最小 K 上选阈值并标 k_star_missing，评估照常读出最小 K（只报告）。另报每个 K 的字节中位。
  P10 阈值（select，租约外 CPU，只在 TRAIN 留出房 cam_nf 上）：每个 K 各自选 R8 阈值：已知 = 合并 FPR <= 对照 FPR + 0.15 pp 的最小 bin；
     占据 = 在已知阈值下，合并空闲召回 >= 对照 NET 空闲召回 - 0.25 pp、家具 F1 最大、同分取最小；不可行时取空闲召回最大的最小
     bin 并标记（L.pick_known / L.pick_occ）。K0 阈值只在 K* 上选（同规则、单机、对照 = 本机触及掩码 FPR 与 NET K0_sep 的
     空闲召回）；K0 的已知或占据阈值不可行时，K0 改用 K* 的 R8 阈值并标注 k0_fallback_r8。
  P11 评估（租约外 CPU，复用 E3-R 导出）：每个最终检查点在 K* 下完整评估（三档；R8 每机一行；K0；O7 字节 + 掩码计 9 次的上界；
     GaBP 闭式与能量断言在"选中格上的浮点势"上做；量化 / 浮点信念差；按 k 与高度分层；容差 1；整件漏报占比）；其余 K 只在
     cam_nf 上算 R8 与字节（整条曲线，只报告）；等步数检查点在它自己的 K* 下算 cam_nf 的 R8 与 K0（只报告）；零学习 Q3 / 一票
     按 GaBP 原生格式（R-E 的 FR1：全分辨率 D = 1、Lam = 1[已知]、z = 量化表值、w -> inf 子树和、O7 线路）对 CTRL（Sk，Qinf 表）
     的字节比（cam_nf，表 A，只报告）；E3-L 的读数取 E3-L summary（只报告）。
  P12 判决（D8-59 写死在 D8_59 里）：同 D8-54 与补充：杠杆 = 4 格 d_furn >= -1 pp 且每房字节中位 <= 0.15 x 对照且保护项全过且未截断
     且 GaBP 断言全过；cam 档 4 格均值 < -3 pp 或 fresh 4 格均值 < -1 pp 时 LEVER 降为 AMBIGUOUS（只降级）；不够 = 任一格
     < -3 pp 或字节 > 0.5 x；其余 AMBIGUOUS；INCOMPLETE 另含：K 选择失败或缺失、阈值缺失、K0 / R8 行缺失。

用法（总是 python -B；CPU 阶段 CUDA_VISIBLE_DEVICES= ）：
  train     --cache C --evidence E --seed S --out R --updates N [--device cuda] [--max-updates M] [--deadline-epoch T] [--ctrl-ckpt E3R/final.pt]
  ctrlprep  --cache C --e3r E3R_RUNS --out CTRL.json [--workers 10]            (TRAIN 留出房的对照字节与对照读数；发射前可先算、钉 sha)
  kselect   --run R --ctrl CTRL.json [--name final|upd6510] [--workers 10]
  select    --run R --ctrl CTRL.json [--name final|upd6510] [--kset all|kstar] [--workers 10]
  eval      --export X --out O --ckpt R1/checkpoints/final.pt,R2/... [--snap R1/checkpoints/upd6510.pt,...] [--qfile Q.json] [--workers 10]
  aggregate --out O --runs R1,R2 --control-summary E3R/eval/summary.json [--e3l-summary E3L/eval/summary.json]
  selfcheck --pyroot P ;  fingerprint ...（= g4_e3r）
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, ast, glob, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3l as L  # noqa: E402   (frozen; read only)

R = L.R
G = L.G
import torch  # noqa: E402
from torch import nn  # noqa: E402
from scipy import ndimage  # noqa: E402

UNK = R.UNK
NBIN = R.NBIN
K_GRID = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
K_MIN, K_MAX = 128, 2048
LAMQ_MIN, LAMQ_MAX = 1, 15
HQ_LIM = 7
ENC_CHUNK = 8
SNAP_UPDATES = 6510
BYTES_SELECT = 0.12                    # K* = the largest K with holdout median byte ratio <= 0.12
FREE_MARGIN_SELECT = 0.0025            # occupancy threshold: pooled free recall >= control - 0.25 pp
GUARD_FPR_SELECT = 0.0015              # known threshold: pooled FPR <= control + 0.15 pp
TIME_FROM = 200
CV_MAX = 0.10
RATE_EVERY = 50
ARM, ARM_SNAP = 'E3L2', 'E3L2s'
RUN_FORMAT = 1


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def file_sha256(p):
    return G.file_sha256(p)


# ================================================================ model and the hard-capacity source
class E3L2(nn.Module):
    """P1: E3-L's encoder, decoder and w (same classes, same construction order), no entropy model"""

    def __init__(self, D=L.D_LAT, width=16, w0=L.W0):
        super().__init__()
        self.D = D
        self.enc = L.Enc(3, D, width); self.dec = L.Dec(D + 2, width, 2)
        self.log_w = nn.Parameter(torch.tensor(math.log(w0)))

    w = L.E3L.w
    encode = L.E3L.encode
    dec_input = L.E3L.dec_input
    decode = L.E3L.decode


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    r = ck['recipe']; model = E3L2(r['D'], r['width'], r['w0']).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return ck, model


def topk_select(lam, K):
    """lam [N, *g] (0 outside the touched blocks), K int or [N] -> bool [N, *g]: the min(K, touched) touched cells of largest Lam"""
    N = lam.shape[0]; flat = lam.detach().flatten(1); touched = flat > 0
    if torch.is_tensor(K) or isinstance(K, (list, tuple, np.ndarray)):
        Ks = torch.as_tensor(K, device=lam.device).reshape(-1).long()
        Ks = Ks.expand(N) if Ks.numel() == 1 else Ks
    else:
        Ks = torch.full((N,), int(K), dtype=torch.long, device=lam.device)
    assert Ks.numel() == N and int(Ks.min()) >= 1
    kmax = int(min(int(Ks.max()), flat.shape[1]))
    score = torch.where(touched, flat, torch.full_like(flat, -float('inf')))
    idx = score.topk(kmax, dim=1).indices
    ok = torch.arange(kmax, device=lam.device)[None] < Ks[:, None]
    sel = torch.zeros_like(touched).scatter_(1, idx, ok) & touched
    return sel.view(lam.shape)


class _ClampOneSided(torch.autograd.Function):
    """P2 (review M1): forward clamp(x, lo, hi); backward identity inside [lo, hi]; above hi only g >= 0 passes (a descent step lowers x),
    below lo only g <= 0 passes (a descent step raises x) -- the one-sided straight-through of E3-L's A_FLOOR"""

    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x); ctx.lo = float(lo); ctx.hi = float(hi)
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = torch.where(x > ctx.hi, g.clamp(min=0.0), g)
        return torch.where(x < ctx.lo, g.clamp(max=0.0), g), None, None


def clamp_one_sided(x, lo, hi):
    return _ClampOneSided.apply(x, lo, hi)


def source_topk(z, lam, K, qmode, clamp=True):
    """P2: (h_q, Lam_q, sel). qmode 'ste' / 'round': selected cells quantised (and clamped), the rest 0; None: the float potentials of the
    selected cells (Lam_c sel, Lam_c z sel; no rounding) for the float GaBP / the checks. clamp=False keeps E3-L's semantics.
    Review M1: on the selected cells Lam_c = clamp(Lam, 1, 15) and h = Lam_c z, both clamps one-sided straight-through."""
    sel = topk_select(lam, K); s = sel.to(lam.dtype)
    if not clamp:                                                        # unclamped: E3-L semantics (Lam_q = 0 carries nothing)
        h = lam.unsqueeze(1) * z
        if qmode is None:
            return h * s.unsqueeze(1), lam * s, sel
        lr = L.DELTA_LAM * torch.round(lam / L.DELTA_LAM); hr = L.DELTA_H * torch.round(h / L.DELTA_H)
        gate = s * (lr > 0).to(lam.dtype)
        lq = lr * gate; hq = hr * gate.unsqueeze(1)
        if qmode == 'ste':
            hg = h * gate.unsqueeze(1)
            return hg + (hq - hg).detach(), lam + (lq - lam).detach(), sel
        assert qmode == 'round', qmode
        return hq, lq, sel
    lam_c = torch.where(sel, clamp_one_sided(lam, LAMQ_MIN, LAMQ_MAX), lam)
    h = lam_c.unsqueeze(1) * z                                           # the transmitted information form: h_q / Lam_q ~ z
    if qmode is None:
        return h * s.unsqueeze(1), lam_c * s, sel
    lq = L.DELTA_LAM * torch.round(lam_c / L.DELTA_LAM) * s
    hq = (L.DELTA_H * torch.round(h / L.DELTA_H)).clamp(-HQ_LIM, HQ_LIM) * s.unsqueeze(1)
    if qmode == 'round':
        return hq, lq, sel
    assert qmode == 'ste', qmode
    hg = clamp_one_sided(h, -HQ_LIM - 0.5, HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel


def encode_fixed(model, f, kn, chunk=ENC_CHUNK):
    """P4: the encoder on fixed-shape blocks of `chunk` agents (zero padding; padded rows dropped). chunk None = one batch."""
    N = f.shape[0]
    if not chunk:
        z, lam = model.encode(f, kn); return z, lam, 0
    P = (-N) % chunk
    if P:
        f = torch.cat([f, torch.zeros((P,) + tuple(f.shape[1:]), dtype=f.dtype, device=f.device)])
        kn = torch.cat([kn, torch.zeros((P,) + tuple(kn.shape[1:]), dtype=kn.dtype, device=kn.device)])
    zs, ls = [], []
    for i in range(0, N + P, chunk):
        z, lam = model.encode(f[i:i + chunk], kn[i:i + chunk]); zs.append(z); ls.append(lam)
    return torch.cat(zs)[:N], torch.cat(ls)[:N], P


def sample_k(rng):
    """P3: log K ~ U[log 128, log 2048]"""
    return int(min(K_MAX, max(K_MIN, round(math.exp(rng.uniform(math.log(K_MIN), math.log(K_MAX)))))))


def sample_spec(rng, houses, p_clean=R.P_CLEAN):
    h, d, sub, aug, dec = L.sample_spec(rng, houses, p_clean)
    return h, d, sub, aug, dec, sample_k(rng)


def forward_batch(model, specs, houses, table_t, qmode, clamp=True, chunk=ENC_CHUNK):
    """encode (fixed shape) every agent of every sample, top-K source with the sample's K, GaBP on each subtree, decode the sampled agent"""
    feats, kns, labs, sizes, Ks = [], [], [], [], []
    for h, d, sub, aug, dec, K in specs:
        H = houses[h]
        e, kn = L.evidence_t(H.t_maps[d][list(sub)], table_t)
        feats.append(L.aug_t(R.features_t(e, kn.to(torch.int32)), aug)); kns.append(L.aug_t(kn, aug))
        scope = (H.t_pk.int() & sum(1 << a for a in sub)) != 0
        labs.append((L.aug_t(scope, aug), L.aug_t(H.t_occ, aug))); sizes.append(len(sub)); Ks += [int(K)] * len(sub)
    z, lam, npad = encode_fixed(model, torch.cat(feats).contiguous(), torch.cat(kns), chunk)
    hq, lq, sel = source_topk(z, lam, torch.as_tensor(Ks, device=lam.device), qmode, clamp)
    w = model.w
    hbs, lbs, rows = [], [], []
    o = 0
    for (h, d, sub, aug, dec, K), m in zip(specs, sizes):
        hb, lb, _ = L.gabp(hq[o:o + m], lq[o:o + m], houses[h].local(sub), w, qmode)
        i = sub.index(dec); hbs.append(hb[i]); lbs.append(lb[i]); rows.append(o + i); o += m
    logits = model.decode(torch.stack(hbs), torch.stack(lbs))
    yk = torch.stack([s for s, _ in labs]); yo = torch.stack([oc for _, oc in labs])
    return logits, yk, yo, dict(hq=hq, lq=lq, lam=lam, z=z, sel=sel, dec=rows, K=Ks, npad=npad, n=len(Ks))


@torch.no_grad()
def holdout_bce(model, specs, houses, table_t, bs):
    model.eval(); tk = to = 0.; n = 0
    for i in range(0, len(specs), bs):
        lg, yk, yo, _ = forward_batch(model, [(h, d, s, 0, dc, K) for h, d, s, _, dc, K in specs[i:i + bs]], houses, table_t, 'round')
        lk, lo = L.bce_terms(lg, yk, yo); k = len(specs[i:i + bs])
        tk += float(lk) * k; to += float(lo) * k; n += k
    model.train()
    return tk / max(n, 1), to / max(n, 1)


@torch.no_grad()
def measure_rate(src):
    """P5 (report only): O7 wire bits of the decoded agents' source messages"""
    rows = src['dec']
    hq = L.DELTA_H * torch.round(src['hq'][rows].detach() / L.DELTA_H); lq = L.DELTA_LAM * torch.round(src['lq'][rows].detach() / L.DELTA_LAM)
    wire = [8.0 * L.wire_bytes(*L.msg_ints(hq[i], lq[i])) for i in range(len(rows))]
    out = dict(measured_bits=float(np.mean(wire)), K=[src['K'][r] for r in rows], support_cells=float((lq > 0).flatten(1).sum(1).float().mean()),
               touched_cells=float((src['lam'][rows].detach() > 0).flatten(1).sum(1).float().mean()), agents=len(rows), padded=int(src['npad']))
    out.update(source_diagnostics(src['z'][rows].detach(), src['lam'][rows].detach(), src['sel'][rows], hq))
    return out


@torch.no_grad()
def source_diagnostics(z, lam, sel, hq):
    """review S6 (report only), on the selected cells: share of h_q != 0 (cells x channels), raw Lam > 15 / < 1, |Lam_c z| > 7.5, raw Lam median"""
    if not bool(sel.any()):
        return dict(hq_nonzero=None, lam_clip_hi=None, lam_clip_lo=None, h_clip=None, lam_raw_median=None)
    lam_s = lam[sel]; h = (lam.clamp(LAMQ_MIN, LAMQ_MAX).unsqueeze(1) * z).movedim(1, -1)[sel]
    return dict(hq_nonzero=float((hq.movedim(1, -1)[sel] != 0).float().mean()), lam_clip_hi=float((lam_s > LAMQ_MAX).float().mean()),
                lam_clip_lo=float((lam_s < LAMQ_MIN).float().mean()), h_clip=float((h.abs() > HQ_LIM + 0.5).float().mean()),
                lam_raw_median=float(lam_s.float().median()))


def step_rate(marks, time_from):
    """P7: s / update over the windows that start at >= time_from; CV of the window rates; CV >= 10 % -> median and a warning"""
    ks = sorted(marks)
    win = [(a, b) for a, b in zip(ks[:-1], ks[1:]) if a >= time_from]
    if not win:                                                          # too short (tests / smoke): every window but the first
        win = list(zip(ks[1:-1], ks[2:])) or list(zip(ks[:-1], ks[1:]))
    if not win:
        return dict(rate=None, windows=[], cv=None, warning='no timing window', time_from=time_from)
    rates = [(marks[b] - marks[a]) / (b - a) for a, b in win]
    mean = (marks[win[-1][1]] - marks[win[0][0]]) / (win[-1][1] - win[0][0])
    cv = float(np.std(rates) / np.mean(rates)) if len(rates) > 1 and np.mean(rates) > 0 else 0.0
    warn = []; rate = mean
    if win[0][0] < time_from:                                            # review S5
        warn.append(f'timing windows start at update {win[0][0]} < {time_from} (fallback: too few windows)')
    if cv >= CV_MAX:
        rate = float(np.median(rates)); warn.append(f'window CV {cv:.3f} >= {CV_MAX}: rate = median of {len(rates)} windows')
    return dict(rate=float(rate), mean_rate=float(mean), windows=[[a, b, r] for (a, b), r in zip(win, rates)], cv=cv, warning=('; '.join(warn) or None),
                time_from=time_from, window_start=win[0][0], window_start_ok=bool(win[0][0] >= time_from))


def _f(x):
    return 'na' if x is None else f'{x:.3f}'


def _save_ckpt(model, recipe, args, done, truncated, run, path, snapshot=False):
    ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, recipe=recipe, seed=args.seed, updates_done=done,
              updates_planned=args.updates, truncated=truncated, w=float(model.w.detach()), run=run, snapshot=snapshot)
    torch.save(ck, path)
    return file_sha256(path)


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
    pair = None
    if args.ctrl_ckpt:
        ck_r = torch.load(args.ctrl_ckpt, map_location='cpu', weights_only=False)
        assert np.array_equal(R.Evidence.from_recipe(ck_r['recipe']['evidence']).table, ev.table), 'evidence != paired E3-R checkpoint evidence'
        assert args.smoke or int(ck_r['seed']) == args.seed, f"paired E3-R checkpoint seed {ck_r['seed']} != {args.seed}"
        pair = dict(ckpt=os.path.abspath(args.ctrl_ckpt), ckpt_sha256=file_sha256(args.ctrl_ckpt), seed=int(ck_r['seed'])); del ck_r
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    tr = L.load_cache(args.cache, man, 'train', dev); ho = L.load_cache(args.cache, man, 'holdout', dev)
    assert tr and ho, 'empty train or holdout set'
    load_s = time.time() - t_start
    table_t = torch.as_tensor(ev.table, dtype=torch.int32, device=dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 32))
    model = E3L2(L.D_LAT, args.width, L.W0).to(dev)
    init_sha = L.weights_sha(model)
    probe = L.probe_check(model, tr, table_t, np.random.default_rng([args.seed, 5]))
    convs = [q for _, q in model.named_parameters() if q.ndim > 1]; rest = [q for _, q in model.named_parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=convs, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)], lr=args.lr)
    rng = np.random.default_rng([args.seed, 3]); hrng = np.random.default_rng([20261027, 9])
    hspecs = [sample_spec(hrng, ho) for _ in range(args.holdout_samples)]
    recipe = dict(D=L.D_LAT, width=args.width, w0=L.W0, block=list(L.BLOCK), delta_h=L.DELTA_H, delta_lam=L.DELTA_LAM, lam_bias0=L.LAM_BIAS0,
                  a_floor=L.A_FLOOR, k_range=[K_MIN, K_MAX], k_sampling='round(exp(U[log 128, log 2048])) per sample', lam_q=[LAMQ_MIN, LAMQ_MAX],
                  h_q=[-HQ_LIM, HQ_LIM], source='review M1: selected Lam_c = clamp(Lam, 1, 15), h = Lam_c z, one-sided straight-through at 1 / 15 / +-7.5',
                  enc_chunk=ENC_CHUNK, unit=R.UNIT, s_scale=R.S_SCALE, evidence=ev.recipe(), p_clean=R.P_CLEAN,
                  batch=args.batch, lr=args.lr, wd=args.wd, warmup=args.warmup, aug='dihedral xy', loss='BCE known (grid) + BCE occupied (target known)',
                  train_decode='one uniform agent of the subtree per sample', subtree='connected (uniform start + uniform frontier growth)',
                  e3l_sha256=file_sha256(L.__file__))
    run = dict(format=RUN_FORMAT, stage='train', batch_name='E3-L2', prereg='D8-59', seed=args.seed, updates=args.updates, max_updates=args.max_updates,
               device=str(dev), recipe=recipe, parameters=L.n_params(model), initial_weights_sha256=init_sha, script_sha256=file_sha256(os.path.abspath(__file__)),
               e3l_sha256=file_sha256(L.__file__), e3r_sha256=file_sha256(R.__file__), noise_panels_sha256=file_sha256(G.__file__),
               cache=os.path.abspath(args.cache), manifest_sha256=man_sha, train_houses=len(tr), holdout_houses=len(ho), deadline_epoch=args.deadline_epoch,
               load_seconds=load_s, smoke=bool(args.smoke), pair=pair, probe_checks=probe, torch=torch.__version__,
               cudnn=(torch.backends.cudnn.version() if dev.type == 'cuda' else None), gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None),
               cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark), pythonpath=os.environ.get('PYTHONPATH'))
    _jdump(run, os.path.join(args.out, 'run.json'))
    print(f'E3-L2 train seed {args.seed}: {len(tr)} train / {len(ho)} holdout houses, {L.n_params(model)} parameters, evidence {ev.desc.get("spec")}, '
          f'load {load_s:.0f}s', flush=True)
    stop_at = min(args.updates, args.max_updates) if args.max_updates else args.updates
    history = []; acc = np.zeros(4); run_n = 0; marks = {}; truncated = False; u = 0; gn = genc = 0.; rate_curve = []; last = {}; snap = None
    t_train = time.time()
    model.train()
    for u in range(stop_at):
        if args.deadline_epoch and time.time() > args.deadline_epoch - args.reserve:
            truncated = True; break
        for g in opt.param_groups:
            g['lr'] = R.lr_at(u, args.updates, args.lr, args.warmup)
        lg, yk, yo, src = forward_batch(model, [sample_spec(rng, tr) for _ in range(args.batch)], tr, table_t, 'ste')
        lk, lo = L.bce_terms(lg, yk, yo)
        loss = lk + lo
        opt.zero_grad(set_to_none=True); loss.backward()
        genc = float(torch.sqrt(sum((q.grad.detach().float() ** 2).sum() for q in model.enc.parameters() if q.grad is not None)))
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        done = u + 1
        if done == 1 or done % args.rate_every == 0:
            last = measure_rate(src); last.update(updates=done, grad_total=gn, grad_encoder=genc); rate_curve.append(last)
            _jdump(rate_curve, os.path.join(args.out, 'rate_curve.json'))
        if args.snap and done == args.snap:                                # P6
            snap = dict(updates=done, path=os.path.join(args.out, 'checkpoints', f'upd{done}.pt'))
            snap['sha256'] = _save_ckpt(model, recipe, args, done, False, run, snap['path'], snapshot=True)
        acc += np.array([float(lk.detach()) + float(lo.detach()), float(lk.detach()), float(lo.detach()), 1.0]); run_n += 1
        log_now = done % args.log_every == 0 or done == stop_at
        if log_now:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            marks[done] = time.time() - t_train
            a = acc / max(run_n, 1)
            print(f'upd{done} bce {a[0]:.5f} (known {a[1]:.5f} occ {a[2]:.5f}) src {last.get("measured_bits", 0):.0f} bit support '
                  f'{last.get("support_cells", 0):.0f}/{last.get("touched_cells", 0):.0f} hq!=0 {_f(last.get("hq_nonzero"))} clipLam '
                  f'{_f(last.get("lam_clip_hi"))}/{_f(last.get("lam_clip_lo"))} clipH {_f(last.get("h_clip"))} Lam~ {_f(last.get("lam_raw_median"))} '
                  f'w {float(model.w.detach()):.2f} grad {gn:.3f} enc {genc:.3f} lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t_start:.0f}s', flush=True)
        if done % args.val_every == 0 or done == stop_at:
            hk, hoc = holdout_bce(model, hspecs, ho, table_t, args.batch)
            a = acc / max(run_n, 1)
            history.append(dict(updates=done, train_bce=a[0], train_bce_known=a[1], train_bce_occ=a[2], holdout_bce_known=hk, holdout_bce_occ=hoc,
                                w=float(model.w.detach()), src_bits=last.get('measured_bits'), support_cells=last.get('support_cells'),
                                touched_cells=last.get('touched_cells'), hq_nonzero=last.get('hq_nonzero'), lam_clip_hi=last.get('lam_clip_hi'),
                                lam_clip_lo=last.get('lam_clip_lo'), h_clip=last.get('h_clip'), lam_raw_median=last.get('lam_raw_median'),
                                grad_total=gn, grad_encoder=genc, seconds=time.time() - t_start))
            print(f'  holdout bce known {hk:.5f} occ {hoc:.5f}', flush=True)
            _jdump(history, os.path.join(args.out, 'history.json'))
        if log_now:
            acc[:] = 0; run_n = 0
    else:
        u = stop_at
    done = u if truncated else stop_at
    tm = step_rate(marks, args.time_from)
    ck_path = os.path.join(args.out, 'checkpoints', 'final.pt')
    ck_sha = _save_ckpt(model, recipe, args, done, truncated, run, ck_path)
    ks = sorted(marks)
    fin = dict(updates_done=done, updates_planned=args.updates, max_updates=args.max_updates, truncated=truncated, complete=(done == args.updates and not truncated),
               rate_s_per_update=tm['rate'], rate_mean_s_per_update=tm.get('mean_rate'), rate_cv=tm['cv'], rate_warning=tm['warning'], rate_windows=tm['windows'],
               time_from=args.time_from, rate_window_start=tm.get('window_start'), rate_window_start_ok=tm.get('window_start_ok'), first_window_s=(marks[ks[0]] if ks else None), first_window_updates=(ks[0] if ks else None), load_seconds=load_s,
               train_seconds=time.time() - t_train, weights_sha256=L.weights_sha(model), w=float(model.w.detach()), snapshot=snap,
               rate_measurements=len(rate_curve), rate_curve_last=(rate_curve[-1] if rate_curve else None), checkpoint=ck_path, checkpoint_sha256=ck_sha,
               history_last=(history[-1] if history else None), seconds=time.time() - t_start)
    _jdump(fin, os.path.join(args.out, 'final.json'))
    print(f'E3-L2 train seed {args.seed} done: {done} updates ({"truncated" if truncated else "complete" if fin["complete"] else "partial"}), '
          f'rate {tm["rate"]} s/update (timing windows from update {tm.get("window_start")}, >= {args.time_from}: {"ok" if tm.get("window_start_ok") else "NO"}; '
          f'cv {tm["cv"]}{", WARNING " + tm["warning"] if tm["warning"] else ""}), w {float(model.w.detach()):.2f}, '
          f'snapshot {snap["updates"] if snap else None}, {fin["seconds"]:.0f}s', flush=True)
    return fin


# ================================================================ off-lease CPU stages: control on the holdout, K selection, thresholds
def _e3r_paths(e3r, seed):
    for ck in (os.path.join(e3r, f's{seed}', 'checkpoints', 'final.pt'), os.path.join(e3r, f's{seed}', 'final.pt')):
        if os.path.exists(ck):
            return ck, os.path.join(e3r, f's{seed}', 'threshold.json')
    raise FileNotFoundError(f'no E3-R checkpoint for seed {seed} under {e3r}')


def _house(cache, entry, seeds):
    p = os.path.join(cache, entry['file'])
    assert file_sha256(p) == entry['file_sha256'], f'{p}: sha256 differs from the manifest'
    return L.LHouse(p, entry, seeds)


def _ctrl_task(job):
    """P8 for one holdout house: control bytes per draw and seed, and the paired NET readout counts"""
    entry, opts = job
    t0 = time.time(); rec = dict(house=entry['house'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        Hd = _house(opts['cache'], entry, opts['seeds'])
        rec['bytes'] = {}; rec['counts'] = {}
        for sd, (ckp, thp) in opts['e3r'].items():
            table = R.Evidence.from_recipe(torch.load(ckp, map_location='cpu', weights_only=False)['recipe']['evidence']).table
            rec['bytes'][sd] = {}
            for d in range(1, len(Hd.draws)):
                _, msgs, msgn = R.relay_sums(list(Hd.maps[d]), Hd.F, table)
                rec['bytes'][sd][Hd.draws[d]] = float(sum(G.int_ctx_bytes(*msgs[e]) for e in Hd.F['directed']) + sum(G.int_ctx_bytes(*msgn[e]) for e in Hd.F['directed']))
            c = L.control_holdout(ckp, thp, [Hd], 'cpu')
            rec['counts'][sd] = dict(R8=c['R8']['counts'], K0=c['K0']['counts'])
        rec['seconds'] = time.time() - t0
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def ctrlprep(args, json_dir, index_file):
    man = _jload(os.path.join(args.cache, 'manifest.json'))
    if args.manifest_sha:
        assert file_sha256(os.path.join(args.cache, 'manifest.json')) == args.manifest_sha, 'manifest sha'
    ents = [e for e in man['houses'] if e['holdout']]
    e3r = {str(sd): _e3r_paths(args.e3r, sd) for sd in (R.TRAIN_SEEDS if not args.smoke else args.seeds_smoke)}
    opts = dict(cache=args.cache, seeds=man['seeds'], e3r=e3r, threads=args.threads)
    recs = [r for _, r in R._pool(args.workers, _ctrl_task, [(e, opts) for e in ents], json_dir, index_file)]
    bad = [r['house'] for r in recs if r.get('failed')]
    assert not bad, f'ctrlprep failed on {bad}: {[r.get("error") for r in recs if r.get("failed")]}'
    out = dict(stage='ctrlprep', script_sha256=file_sha256(os.path.abspath(__file__)), e3l_sha256=file_sha256(L.__file__), cache=os.path.abspath(args.cache),
               manifest_sha256=file_sha256(os.path.join(args.cache, 'manifest.json')), houses=sorted(r['house'] for r in recs), draws=[f'cam_nf_s{s}' for s in man['seeds']],
               rule='E3-R relay: sum + count messages, G.int_ctx_bytes, all directed edges (D8-54); readouts = L.control_holdout (frozen NET + thresholds)',
               e3r={sd: dict(ckpt=ck, ckpt_sha256=file_sha256(ck), threshold=th, threshold_sha256=file_sha256(th),
                             table_sha256=_table_sha(R.Evidence.from_recipe(torch.load(ck, map_location='cpu', weights_only=False)['recipe']['evidence']).table))
                    for sd, (ck, th) in e3r.items()},
               bytes={sd: {r['house']: r['bytes'][sd] for r in recs} for sd in e3r}, readouts={})
    for sd in e3r:
        out['readouts'][sd] = {}
        for which in ('R8', 'K0'):
            c = np.sum([r['counts'][sd][which] for r in recs], 0)
            out['readouts'][sd][which] = dict(free=c[0] / max(c[1], 1), fpr=c[2] / max(c[3], 1), furn_f1=2 * c[4] / max(2 * c[4] + c[5] + c[6], 1), counts=c.tolist())
    _jdump(out, args.out)
    print(json.dumps({sd: dict(R8=out['readouts'][sd]['R8'], bytes_median=float(np.median([b for h in out['bytes'][sd].values() for b in h.values()])))
                      for sd in e3r}, default=float), flush=True)
    return out


def _table_sha(table):
    import hashlib
    return hashlib.sha256(np.asarray(table, np.int64).tobytes()).hexdigest()


def _ck_path(run, name):
    return os.path.join(run, 'checkpoints', f'{name}.pt')


def _sfx(name):
    return '' if name == 'final' else f'_{name}'


def _house_bytes_task(job):
    """P9 for one holdout house: E3-L2 house bytes for every K and draw"""
    entry, opts = job
    rec = dict(house=entry['house'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        ck, model = load_model(opts['ckpt'], 'cpu'); w = float(model.w.detach())
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        Hd = _house(opts['cache'], entry, opts['seeds'])
        rec['bytes'] = {str(K): {} for K in opts['ks']}
        with torch.no_grad():
            for d in range(1, len(Hd.draws)):
                z, lam = L.encode_maps(model, list(Hd.maps[d]), tb, 'cpu')
                for K in opts['ks']:
                    hq, lq, _ = source_topk(z, lam, K, 'round')
                    by = 0.0
                    for cm in Hd.comps:
                        _, _, msgs = L.gabp(hq[cm], lq[cm], Hd.local(cm), w, 'round')
                        by += sum(L.wire_bytes(*L.msg_ints(*msgs[e])) for e in Hd.local(cm)['directed'])
                    rec['bytes'][str(K)][Hd.draws[d]] = float(by)
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def check_ctrl(ctrl, cache, runj, sd):
    """review S1: the control file belongs to this cache, its E3-R checkpoint / threshold files are unchanged, and its checkpoint is the
    one the run was paired with at train time"""
    assert ctrl['manifest_sha256'] == file_sha256(os.path.join(cache, 'manifest.json')), 'control file manifest sha != the cache manifest'
    e = ctrl['e3r'][sd]
    assert file_sha256(e['ckpt']) == e['ckpt_sha256'], f"control E3-R checkpoint {e['ckpt']} changed"
    assert file_sha256(e['threshold']) == e['threshold_sha256'], f"control E3-R threshold {e['threshold']} changed"
    if runj.get('pair'):
        assert runj['pair']['ckpt_sha256'] == e['ckpt_sha256'], 'the E3-R checkpoint paired at train time != the control file checkpoint'


def choose_k(ratios, ks=K_GRID, cap=BYTES_SELECT):
    """P9: the largest K whose median ratio <= cap; None when no K qualifies"""
    ok = [K for K in ks if ratios.get(str(K)) is not None and ratios[str(K)] <= cap + 1e-12]
    return max(ok) if ok else None


def kselect(args, json_dir, index_file):
    ctrl = _jload(args.ctrl); runj = _jload(os.path.join(args.run, 'run.json')); sd = str(runj['seed'])
    ckp = _ck_path(args.run, args.name); ck = torch.load(ckp, map_location='cpu', weights_only=False)
    assert _table_sha(ck['recipe']['evidence']['table']) == ctrl['e3r'][sd]['table_sha256'], 'run evidence table != paired control table'
    cache = args.cache or runj['cache']; man = _jload(os.path.join(cache, 'manifest.json'))
    check_ctrl(ctrl, cache, runj, sd)
    ents = [e for e in man['houses'] if e['holdout']]
    assert sorted(e['house'] for e in ents) == ctrl['houses'], 'holdout houses != the control file'
    opts = dict(ckpt=ckp, cache=cache, seeds=man['seeds'], ks=list(K_GRID), threads=args.threads)
    recs = [r for _, r in R._pool(args.workers, _house_bytes_task, [(e, opts) for e in ents], json_dir, index_file)]
    bad = [r['house'] for r in recs if r.get('failed')]
    assert not bad, f'kselect failed on {bad}: {[r.get("error") for r in recs if r.get("failed")]}'
    cb = ctrl['bytes'][sd]
    ratios, med_b, rows = {}, {}, {}
    for K in K_GRID:
        rr = [r['bytes'][str(K)][dn] / cb[r['house']][dn] for r in recs for dn in r['bytes'][str(K)]]
        ratios[str(K)] = float(np.median(rr)); med_b[str(K)] = float(np.median([b for r in recs for b in r['bytes'][str(K)].values()]))
        rows[str(K)] = dict(median_ratio=ratios[str(K)], median_bytes=med_b[str(K)], n=len(rr), max_ratio=float(np.max(rr)))
    k_star = choose_k(ratios, cap=(args.bytes_max if args.smoke and args.bytes_max else BYTES_SELECT))
    out = dict(stage='kselect', name=args.name, seed=int(sd), ckpt=ckp, ckpt_sha256=file_sha256(ckp), ctrl=os.path.abspath(args.ctrl), ctrl_sha256=file_sha256(args.ctrl),
               cap=BYTES_SELECT, k_star=k_star, smallest_k=min(K_GRID), per_k=rows, control_bytes_median=float(np.median([b for h in cb.values() for b in h.values()])),
               houses=ctrl['houses'], rule='K* = largest K with median over (holdout house, draw) of E3-L2 / paired E3-R bytes <= 0.12',
               script_sha256=file_sha256(os.path.abspath(__file__)), smoke=bool(args.smoke),
               bytes={K: {r['house']: r['bytes'][K] for r in recs} for K in rows})
    _jdump(out, os.path.join(args.run, f'kselect{_sfx(args.name)}.json'))
    print(f"E3-L2 kselect seed {sd} {args.name}: K* {k_star} | " + ' '.join(f'{K}:{rows[str(K)]["median_ratio"]:.3f}' for K in K_GRID), flush=True)
    return out


def _sel_task(job):
    """P10 one holdout house, one pass. pass 1: FPR histograms (known head); pass 2: occupancy histograms given the known bins"""
    entry, opts = job
    rec = dict(house=entry['house'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        ck, model = load_model(opts['ckpt'], 'cpu'); w = float(model.w.detach())
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        Hd = _house(opts['cache'], entry, opts['seeds'])
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]; K0T = [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
        jobs = [('R8', K) for K in opts['ks_r8']] + [('K0', K) for K in opts['ks_k0']]
        acc = {f'{wh}|{K}': (np.zeros(NBIN, np.int64), np.zeros(2, np.int64)) if opts['pass'] == 1 else
               (np.zeros((3, NBIN), np.int64), np.zeros(2, np.int64)) for wh, K in jobs}
        with torch.no_grad():
            for d in range(1, len(Hd.draws)):
                maps = Hd.maps[d]
                z, lam = L.encode_maps(model, list(maps), tb, 'cpu')
                for wh, K in jobs:
                    hq, lq, _ = source_topk(z, lam, K, 'round'); key = f'{wh}|{K}'
                    items = []
                    if wh == 'R8':
                        for ci, cm in enumerate(Hd.comps):
                            hb, lb, _ = L.gabp(hq[cm], lq[cm], Hd.local(cm), w, 'round')
                            items += [(x, R8T[ci]) for x in L.decode_np(model, hb, lb)]
                    else:
                        items += [(x, K0T[a]) for a, x in enumerate(L.decode_np(model, hq, lq))]
                    for lg, T in items:
                        kb = R.logit_bin(lg[0])
                        if opts['pass'] == 1:
                            unk = ~T.known
                            acc[key][0][:] += np.bincount(kb[unk], minlength=NBIN); acc[key][1][:] += [int(unk.sum()), 0]
                        else:
                            pk = kb >= opts['jk'][key]; ob = R.logit_bin(lg[1])
                            acc[key][0][0] += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NBIN)
                            acc[key][0][1] += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NBIN)
                            acc[key][0][2] += np.bincount(ob[pk & T.free], minlength=NBIN)
                            acc[key][1][:] += [int(T.tr.sum()), int(T.free.sum())]
        rec['acc'] = {k: (a.tolist(), b.tolist()) for k, (a, b) in acc.items()}
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def k0_thresholds(k0, r8):
    """P10: the K0 thresholds at K*; when the K0 known or occupancy threshold is infeasible, the R8 thresholds of K* (flagged)"""
    k0 = dict(k0)
    if k0['feasible_known'] and k0['feasible_occ']:
        k0['k0_fallback_r8'] = False
    else:
        k0.update(k0_fallback_r8=True, own=dict(j_known=k0['j_known'], j_occ=k0['j_occ']), j_known=r8['j_known'], j_occ=r8['j_occ'],
                  t_known=r8['t_known'], t_occ=r8['t_occ'])
    return k0


def occ_free_min(ctrl_free):
    """P10: the occupancy threshold's pooled free-recall floor = control NET - 0.25 pp"""
    return ctrl_free - FREE_MARGIN_SELECT


def select(args, json_dir, index_file):
    ctrl = _jload(args.ctrl); runj = _jload(os.path.join(args.run, 'run.json')); sd = str(runj['seed'])
    ks = _jload(os.path.join(args.run, f'kselect{_sfx(args.name)}.json'))
    k_star = ks['k_star']; k_missing = k_star is None
    if k_missing:                                                        # P9: INCOMPLETE, but the smallest K is still read out (report only)
        k_star = min(K_GRID)
        print(f'kselect: no K with median ratio <= {BYTES_SELECT}; thresholds at the smallest K {k_star} for the report (the verdict is INCOMPLETE)', flush=True)
    ckp = _ck_path(args.run, args.name); ck = torch.load(ckp, map_location='cpu', weights_only=False)
    assert _table_sha(ck['recipe']['evidence']['table']) == ctrl['e3r'][sd]['table_sha256'], 'run evidence table != paired control table'
    cache = args.cache or runj['cache']; man = _jload(os.path.join(cache, 'manifest.json'))
    check_ctrl(ctrl, cache, runj, sd)
    ents = [e for e in man['houses'] if e['holdout']]
    assert sorted(e['house'] for e in ents) == ctrl['houses'], 'holdout houses != the control file'
    ks_r8 = list(K_GRID) if args.kset == 'all' else [k_star]
    cr = ctrl['readouts'][sd]
    base = dict(ckpt=ckp, cache=cache, seeds=man['seeds'], ks_r8=ks_r8, ks_k0=[k_star], threads=args.threads)
    t0 = time.time()
    recs = [r for _, r in R._pool(args.workers, _sel_task, [(e, dict(base, **{'pass': 1})) for e in ents], json_dir, index_file)]
    assert not [r for r in recs if r.get('failed')], f"select pass 1 failed: {[r.get('error') for r in recs if r.get('failed')]}"
    jk, sel = {}, {}
    for key in recs[0]['acc']:
        wh, K = key.split('|')
        U = np.sum([r['acc'][key][0] for r in recs], 0); ut = int(np.sum([r['acc'][key][1][0] for r in recs]))
        j, feas, tail = L.pick_known(U, ut, cr[wh]['fpr'] + GUARD_FPR_SELECT)
        jk[key] = j; sel[key] = dict(j_known=j, t_known=R.bin_edge(j), fpr=float(tail[j]), fpr_max=cr[wh]['fpr'] + GUARD_FPR_SELECT, feasible_known=feas)
    recs2 = [r for _, r in R._pool(args.workers, _sel_task, [(e, dict(base, **{'pass': 2, 'jk': jk})) for e in ents], json_dir, index_file)]
    assert not [r for r in recs2 if r.get('failed')], f"select pass 2 failed: {[r.get('error') for r in recs2 if r.get('failed')]}"
    for key in recs2[0]['acc']:
        wh, K = key.split('|')
        A = np.sum([r['acc'][key][0] for r in recs2], 0); tt = np.sum([r['acc'][key][1] for r in recs2], 0)
        fmin = occ_free_min(cr[wh]['free'])
        j, feas, f1, fr = L.pick_occ(A[0], A[1], A[2], int(tt[0]), int(tt[1]), fmin)
        sel[key].update(j_occ=j, t_occ=R.bin_edge(j), f1=float(f1[j]), free=float(fr[j]), free_min=fmin, feasible_occ=feas,
                        control=dict(free=cr[wh]['free'], fpr=cr[wh]['fpr'], furn_f1=cr[wh]['furn_f1']))
    out = dict(stage='select', name=args.name, seed=int(sd), k_star=k_star, k_star_missing=k_missing, ckpt=ckp, ckpt_sha256=file_sha256(ckp), kselect_sha256=file_sha256(os.path.join(args.run, f'kselect{_sfx(args.name)}.json')),
               ctrl_sha256=file_sha256(args.ctrl), kset=args.kset, houses=ctrl['houses'], n_houses=len(ents), draws=ctrl['draws'], device='cpu',
               rule='known: smallest bin with pooled FPR <= control + 0.15 pp; occupancy: pooled free >= control NET - 0.25 pp, max furniture F1, ties smallest; '
                    'infeasible -> smallest bin of max free (flag); K0 only at K*, infeasible -> the R8 thresholds of K* (k0_fallback_r8)',
               R8={K: sel[f'R8|{K}'] for K in map(str, ks_r8)}, seconds=None, script_sha256=file_sha256(os.path.abspath(__file__)))
    out['K0'] = k0_thresholds(sel[f'K0|{k_star}'], out['R8'][str(k_star)]); k0 = out['K0']; out['seconds'] = time.time() - t0
    _jdump(out, os.path.join(args.run, f'threshold{_sfx(args.name)}.json'))
    print(f"E3-L2 select seed {sd} {args.name}: K* {k_star} R8 " + json.dumps({K: [v['t_known'], v['t_occ'], round(v['f1'], 4), v['feasible_known'] and v['feasible_occ']]
                                                                                 for K, v in out['R8'].items()}) + f" K0 fallback {k0['k0_fallback_r8']} {out['seconds']:.0f}s", flush=True)
    return out


# ================================================================ eval (CPU, after the lease)
def whole_miss(p, T):
    """FN voxels (target furniture not predicted occupied) and the part of them in target objects (26-connected) with no hit at all"""
    lab, nl = ndimage.label(T.tr, structure=np.ones((3, 3, 3), bool))
    pr = (p == 1) & T.known & T.H.region
    hit = np.bincount(lab[pr & T.tr], minlength=nl + 1) > 0
    fn = T.tr & ~pr
    return int(fn.sum()), int((fn & ~hit[lab] & (lab > 0)).sum())


@torch.no_grad()
def eval_k(model, w, enc, K, thr_r8, thr_k0, F, H, K0T, R8T, L_pub, L_k0, chunk, full):
    """one checkpoint, one tier, one K. full: R8 + K0 + bytes (+ upper) + GaBP checks + belief diff + rates + whole-object misses;
    else R8 + bytes only (the curve)"""
    zp, lp, zk, lk, same = enc
    hq, lq, sel = source_topk(zp, lp, K, 'round')
    jk, jo = thr_r8['j_known'], thr_r8['j_occ']
    r8 = []; by = byu = 0.0; out = {}
    chk = dict(cf_max=0.0, cf_rel_max=0.0, cf_scale_max=0.0, violations=0, components=0, steps=0, cf_cells=0, cf_excluded_tiny=0)
    qd = np.zeros(4); dis = np.zeros(2); lost = 0; wm = np.zeros(2, np.int64)
    if full:
        hfs, lfs, _ = source_topk(zp, lp, K, None)
    for ci, cm in enumerate(F['comps']):
        Fl = L.local_forest(F, cm)
        hb, lb, msgs = L.gabp(hq[cm], lq[cm], Fl, w, 'round')
        for e in Fl['directed']:
            mi = L.msg_ints(*msgs[e]); by += L.wire_bytes(*mi)
            if full:
                byu += L.int_bytes(*mi)
        if full:
            hf, lf, _ = L.gabp(hfs[cm], lfs[cm], Fl, w, None)
            xq = L.belief_mean(hb, lb); xf = L.belief_mean(hf, lf); s = lb > 0
            if s.any():
                dd = (xq - xf).abs().amax(1)[s]; qd += [float(dd.sum()), float(dd.max()), float(s.sum()), float(xf.abs().amax(1)[s].sum())]
            lost += int(((lb == 0) & (lf >= 0.5 * L.DELTA_LAM)).sum())
            c = L.gbp_checks(zp[cm], lfs[cm], Fl, w)
            chk['cf_max'] = max(chk['cf_max'], c['cf_max']); chk['cf_rel_max'] = max(chk['cf_rel_max'], c['cf_rel']); chk['cf_scale_max'] = max(chk['cf_scale_max'], c['cf_scale'])
            chk['violations'] += c['violations']; chk['components'] += 1; chk['steps'] += c['steps']; chk['cf_cells'] += c['cf_cells']; chk['cf_excluded_tiny'] += c['cf_excluded_tiny']
            for i in range(len(cm)):
                for j in range(i + 1, len(cm)):
                    both = (lb[i] > 0) & (lb[j] > 0)
                    if both.any():
                        dis += [float(((xq[i] - xq[j]) ** 2).sum(0)[both].sum()), float(both.sum())]
        lg = L.decode_np(model, hb, lb, chunk)
        kc = sum((L_pub[a] != UNK).astype(np.int32) for a in cm)
        for i in range(len(cm)):
            p = L.readout(lg[i][0], lg[i][1], jk, jo)
            r8.append([R.scored(p, R8T[ci], H, kc), 1])
            if full:
                wm += whole_miss(p, R8T[ci])
    out.update(R8=r8, bytes=by, K=int(K))
    if not full:
        return out
    if same:
        hk, lkq = hq, lq
    else:
        hk, lkq, _ = source_topk(zk, lk, K, 'round')
    lgk = L.decode_np(model, hk, lkq, chunk)
    k0, k0s = [], []
    for a in range(len(L_k0)):
        ki = (L_k0[a] != UNK).astype(np.int32)
        k0.append(R.scored(L.readout(lgk[a][0], lgk[a][1], jk, jo), K0T[a], H, ki))
        k0s.append(R.scored(L.readout(lgk[a][0], lgk[a][1], thr_k0['j_known'], thr_k0['j_occ']), K0T[a], H, ki))
    rate = []; ch = np.zeros((L.D_LAT + 1, 2 * L.CH_LIM + 1), np.int64)
    for a in range(len(L_pub)):
        hi, li = L.msg_ints(hq[a], lq[a])
        rate.append(dict(measured_bits=8 * L.wire_bytes(hi, li), upper_bits=8 * L.int_bytes(hi, li), mask_bits=8 * L.mask_bytes(li > 0),
                         cells=int((li > 0).sum()), touched=int((lp[a] > 0).sum())))
        ch += L.chan_hist(hi, li)
    out.update(K0=k0, K0_sep=k0s, k0_fallback_r8=bool(thr_k0.get('k0_fallback_r8')), bytes_upper=byu, gbp=chk, w=w,
               qdiff=dict(sum_absmax=qd[0], max=qd[1], cells=qd[2], sum_float_absmax=qd[3], lost_cells=lost), disagree=dict(sum_sq=dis[0], cells=dis[1]),
               rate=rate, channels=ch.tolist(), whole_miss=dict(fn=int(wm[0]), fn_whole_object=int(wm[1])))
    return out


def zero_learning_bytes(L_maps, F, tables):
    """P11 (report only): R-E FR1 (GaBP native format) bytes of the zero-learning alphabets vs CTRL (Sk with Qinf)"""
    res = {}
    for nm, tb in tables.items():
        _, msgs, msgn = R.relay_sums(list(L_maps), F, tb)
        res[nm] = dict(ctrl=float(sum(G.int_ctx_bytes(*msgs[e]) + G.int_ctx_bytes(*msgn[e]) for e in F['directed'])),
                       fr1=float(sum(L.wire_bytes(msgs[e][0][None].astype(np.int64), msgn[e][0].astype(np.int64)) for e in F['directed'])))
    out = {nm: dict(fr1=v['fr1'], ratio=v['fr1'] / max(res['Qinf']['ctrl'], 1e-9)) for nm, v in res.items()}
    out['ctrl_qinf'] = res['Qinf']['ctrl']
    return out


def qtables(path):
    q = _jload(path)['quantizer']['A']
    return {k: np.asarray(ast.literal_eval(q[k])['table'] if isinstance(q[k], str) else q[k]['table'], np.int64) for k in ('Qinf', 'Q3', 'VOTE')}


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
        rec['tiers'] = {}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            T = {}
            for p, kind in [(p, 'final') for p in opts['ckpts']] + [(p, 'snap') for p in opts['snaps']]:
                if kind == 'snap' and tn != R.JUDGED_TIER:
                    continue
                ck, model = load_model(p, 'cpu'); w = float(model.w.detach()); sd = str(ck['seed'])
                thr = _jload(opts['thr'][p]); k_star = int(thr['k_star'])
                tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
                zp, lp = L.encode_maps(model, L_pub, tb, 'cpu'); same = all(a is b for a, b in zip(L_pub, L_k0))
                zk, lk = (zp, lp) if same else L.encode_maps(model, L_k0, tb, 'cpu')
                enc = (zp, lp, zk, lk, same)
                a = eval_k(model, w, enc, k_star, thr['R8'][str(k_star)], thr['K0'], F, H, K0T, R8T, L_pub, L_k0, opts['chunk'], full=True)
                a['k_star'] = k_star
                if kind == 'final' and tn == R.JUDGED_TIER:                  # the curve (report only)
                    a['curve'] = {K: eval_k(model, w, enc, int(K), thr['R8'][K], None, F, H, K0T, R8T, L_pub, L_k0, opts['chunk'], full=False)
                                  for K in thr['R8'] if int(K) != k_star}
                T[f'{ARM if kind == "final" else ARM_SNAP}_{sd}'] = a
            if tn == R.JUDGED_TIER and opts.get('qtables') is not None:
                T['zero_learning'] = zero_learning_bytes(L_pub, F, opts['qtables'])
            rec['tiers'][tn] = T
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _eval_and_save(job):
    rec = eval_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    _jdump(rec, os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json"))
    return rec


# ================================================================ aggregation and the D8-59 verdict (criteria hard-coded)
D8_59 = dict(L.D8_54, prereg='D8-59', bytes_select=BYTES_SELECT, free_margin_select=FREE_MARGIN_SELECT, known_select_fpr=GUARD_FPR_SELECT, k_grid=list(K_GRID),
             supplement='D8-59: hard-capacity top-K source, K* = largest K with holdout median byte ratio <= 0.12; judged at K*; '
                        'lever = 4 cells d >= -1 pp, bytes <= .15 x, guards; cam mean < -3 pp or fresh mean < -1 pp demote LEVER only; '
                        'not enough = any cell < -3 pp or bytes > .5 x')


def _summ_arm(A):
    e = dict(R8=R._summ([x for a in A for x in G._expand(a['R8'])]), bytes_median=float(np.median([a['bytes'] for a in A])),
             k_star=sorted({a.get('k_star') for a in A}))
    if 'K0' in A[0]:
        e.update(K0=R._summ([x for a in A for x in a['K0']]), K0_sep=R._summ([x for a in A for x in a['K0_sep']]),
                 k0_fallback_r8=any(a.get('k0_fallback_r8') for a in A), report_bytes_upper_median=float(np.median([a['bytes_upper'] for a in A])))
        q = np.sum([[a['qdiff']['sum_absmax'], a['qdiff']['cells'], a['qdiff']['sum_float_absmax']] for a in A], 0)
        e['quant_belief_diff'] = dict(mean_abs=q[0] / max(q[1], 1), max_abs=max(a['qdiff']['max'] for a in A), lost_cells=int(sum(a['qdiff']['lost_cells'] for a in A)))
        ds = np.sum([[a['disagree']['sum_sq'], a['disagree']['cells']] for a in A], 0)
        e['belief_disagreement_rms'] = float(math.sqrt(ds[0] / ds[1])) if ds[1] else None
        rt = [x for a in A for x in a['rate']]
        e['rate'] = dict(measured_bits_median=float(np.median([x['measured_bits'] for x in rt])), cells_median=float(np.median([x['cells'] for x in rt])),
                         touched_median=float(np.median([x['touched'] for x in rt])), upper_bits_median=float(np.median([x['upper_bits'] for x in rt])))
        ch = np.sum([a['channels'] for a in A], 0)
        e['channels'] = [dict(name=('lam' if c == 0 else f'h{c - 1}'), entropy_bits=L._ent(ch[c]), usage_nonzero=(1.0 - ch[c][L.CH_LIM] / max(ch[c].sum(), 1)))
                         for c in range(ch.shape[0])]
        wm = np.sum([[a['whole_miss']['fn'], a['whole_miss']['fn_whole_object']] for a in A], 0)
        e['whole_object_miss_share'] = float(wm[1] / wm[0]) if wm[0] else None
        e['w'] = float(np.mean([a['w'] for a in A]))
        e['gbp'] = dict(cf_max=max(a['gbp']['cf_max'] for a in A), cf_rel_max=max(a['gbp']['cf_rel_max'] for a in A), violations=sum(a['gbp']['violations'] for a in A))
    if 'curve' in A[0]:
        ks = sorted({K for a in A for K in a['curve']}, key=int)
        e['curve'] = {K: dict(R8=R._summ([x for a in A for x in G._expand(a['curve'][K]['R8'])]), bytes_median=float(np.median([a['curve'][K]['bytes'] for a in A])))
                      for K in ks if all(K in a['curve'] for a in A)}
    return e


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
    for r in ok:
        for tn, T in r['tiers'].items():
            for arm, a in T.items():
                if not arm.startswith(ARM + '_'):
                    continue
                g = S['gbp'].setdefault(arm, dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, steps=0, jobs=0))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['cf_rel_max'] = max(g['cf_rel_max'], a['gbp']['cf_rel_max'])
                g['violations'] += a['gbp']['violations']; g['components'] += a['gbp']['components']; g['steps'] += a['gbp']['steps']; g['jobs'] += 1
    for g in S['gbp'].values():
        g['pass'] = bool(g['cf_max'] <= L.CF_TOL and g['violations'] == 0 and g['components'] > 0)
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
                    if arm == 'zero_learning' or not all(arm in r['tiers'][tn] for r in X):
                        continue
                    d[arm] = _summ_arm([r['tiers'][tn][arm] for r in X])
                if all('zero_learning' in r['tiers'][tn] for r in X):
                    d['zero_learning'] = {nm: dict(ratio_median=float(np.median([r['tiers'][tn]['zero_learning'][nm]['ratio'] for r in X])))
                                          for nm in ('Qinf', 'Q3', 'VOTE')}
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
    cell = dict(k_star=a.get('k_star'), e3l2_R8_furn=a['R8']['furn_f1'], control_R8_furn=c['R8']['furn_f1'], d_furn=a['R8']['furn_f1'] - c['R8']['furn_f1'],
                bytes_median=a['bytes_median'], control_bytes=cb, bytes_ratio=a['bytes_median'] / cb)
    if 'curve' in a:
        cell['curve'] = {K: dict(d_furn=v['R8']['furn_f1'] - c['R8']['furn_f1'], bytes_ratio=v['bytes_median'] / cb) for K, v in a['curve'].items()}
    if not judged:
        return cell
    if 'K0_sep' not in a:
        reasons.append(f'{where}: K0 rows missing'); return None
    cell.update(d_furn_tol1=a['R8']['furn_f1_tol1'] - c['R8']['furn_f1_tol1'], free_delta=a['R8']['free_recall'] - c['R8']['free_recall'],
                surface_delta=a['R8']['surface_f1'] - c['R8']['surface_f1'], fpr_delta=a['R8']['unknown_known_fpr'] - c['R8']['unknown_known_fpr'],
                e3l2_K0_sep=a['K0_sep']['furn_f1'], control_K0_sep=c['K0_sep']['furn_f1'], d_K0=a['K0_sep']['furn_f1'] - c['K0_sep']['furn_f1'],
                k0_fallback_r8=a.get('k0_fallback_r8'), w=a.get('w'), whole_object_miss_share=a.get('whole_object_miss_share'))
    cl = D['tiers'].get(C['clean_tier'], {}).get(arm); cam = D['tiers'].get(C['cam_tier'], {}).get(arm)
    cc, _ = _ctrl_arm(Sc, panel, ns, C['cam_tier'], ts)
    if cl is None or cam is None or cc is None:
        reasons.append(f'{where}: clean / cam row (or the cam control) missing'); return None
    cell['clean_R8_furn'] = cl['R8']['furn_f1']; cell['cam_d_furn'] = cam['R8']['furn_f1'] - cc['R8']['furn_f1']
    g = dict(free=cell['free_delta'] >= -C['guard_free'] - 1e-12, surface=cell['surface_delta'] >= -C['guard_surface'] - 1e-12,
             fpr=cell['fpr_delta'] <= C['guard_fpr'] + 1e-12, clean=cl['R8']['furn_f1'] >= C['clean_min'], k0=cell['d_K0'] >= C['k0_margin'] - 1e-12)
    cell['guards'] = g; cell['guards_pass'] = all(g.values())
    return cell


def _stage_info(r, name, reasons, sd):
    ks = os.path.join(r, f'kselect{_sfx(name)}.json'); th = os.path.join(r, f'threshold{_sfx(name)}.json')
    info = {}
    if not os.path.exists(ks):
        reasons.append(f'training seed {sd}: no kselect{_sfx(name)}.json'); return info
    k = _jload(ks); info['k_star'] = k['k_star']; info['k_ratios'] = {K: v['median_ratio'] for K, v in k['per_k'].items()}
    if k['k_star'] is None:
        reasons.append(f'training seed {sd}: no K with holdout median byte ratio <= {BYTES_SELECT} (smallest K {k["smallest_k"]} ratio '
                       f'{k["per_k"][str(k["smallest_k"])]["median_ratio"]:.3f}; read out at the smallest K, report only)'); return info
    if not os.path.exists(th):
        reasons.append(f'training seed {sd}: no threshold{_sfx(name)}.json'); return info
    t = _jload(th)
    if t.get('k_star') != k['k_star'] and not (k['k_star'] is None and t.get('k_star_missing')):
        reasons.append(f'training seed {sd}: threshold K* {t.get("k_star")} != kselect K* {k["k_star"]}')
    info['threshold'] = dict(R8=t['R8'].get(str(k['k_star'])), K0=t.get('K0'))
    return info


def verdict(S, runs=None, control_summary=None, smoke=False, e3l_summary=None):
    """D8-59 (D8-54 and its supplements at K*): LEVER iff in all 4 VAL cam_nf cells d_furn >= -1 pp, median bytes <= .15 x control and
    every guard passes (free / surface -0.5 pp, FPR +0.3 pp, clean >= .99, K0 -3 pp), the GaBP checks pass and no seed is truncated. A LEVER
    is demoted to AMBIGUOUS when the cam-tier 4-cell mean d < -3 pp or the fresh 4-cell mean d < -1 pp (demotion only; the literal fresh
    sign is reported). NOT_ENOUGH iff any cell d < -3 pp or bytes > .5 x. Else AMBIGUOUS. INCOMPLETE: failed jobs, E1 mismatch, a missing
    run / final / kselect / threshold, no K <= 0.12 on the holdout, < 2,000 updates, different update counts, any missing cell / house /
    control row / tier / K0 row, no GaBP checks."""
    C = dict(D8_59)
    reasons = list(S.get('agg_reasons', []))
    if S.get('failed'):
        reasons.append(f"failed jobs: {S['failed']}")
    if S.get('e1_mismatch'):
        reasons.append(f"E1 reproduction mismatch: {S['e1_mismatch']}")
    Sc = _jload(control_summary) if control_summary else None
    if Sc is None:
        reasons.append('control summary (E3-R) not given')
    run_info = {}; truncated = []; stages = {}
    if runs is None:
        reasons.append('training runs not given')
    else:
        for r in runs:
            rr = []
            sd, info = R._run_info(r, C, rr, truncated)
            reasons += [x for x in rr if 'no threshold.json' not in x]           # E3-L2's threshold file is checked by _stage_info
            if sd is not None and info is not None:
                run_info[sd] = info; stages[sd] = _stage_info(r, 'final', reasons, sd)
        for sd in C['train_seeds']:
            if sd not in run_info:
                reasons.append(f'training seed {sd} missing')
        if len({i['updates_planned'] for i in run_info.values()}) > 1:
            reasons.append(f"training seeds have different update counts {sorted({i['updates_planned'] for i in run_info.values()})}")
    for ts in C['train_seeds']:
        if f'{ARM}_{ts}' not in S.get('gbp', {}):
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
        if fresh:
            mv = float(np.mean([c['d_furn'] for c in cells.values()])); mf = float(np.mean([c['d_furn'] for c in fresh.values()]))
            fresh_info = dict(val_mean_d_furn=mv, fresh_mean_d_furn=mf, fresh_ok=bool(mf >= C['fresh_min_mean'] - 1e-12), report_literal_sign_opposite=bool(mv * mf < 0))
            if v == 'LEVER' and not fresh_info['fresh_ok']:
                v = f'AMBIGUOUS (lever demoted: fresh 4-cell mean d_furn {mf:+.4f} < {C["fresh_min_mean"]})'
        notes.append(f'cam-tier 4-cell mean d_furn {cam_mean:+.4f}')
    rep = {}
    if e3l_summary:                                                          # report only: E3-L in the same cells
        Se = _jload(e3l_summary)
        for k, c in cells.items():
            ts, ns = k.split('|')
            try:
                rep[k] = dict(e3l_R8_furn=Se['panels']['val']['seeds'][ns]['tiers']['cam_nf'][f'E3L_{ts}']['R8']['furn_f1'])
                rep[k]['d_e3l2_minus_e3l'] = c['e3l2_R8_furn'] - rep[k]['e3l_R8_furn']
            except KeyError:
                rep[k] = None
    return dict(verdict=v, criteria=C, cells=cells, fresh_cells=fresh, fresh=fresh_info, runs=run_info, stages=stages, truncated_seeds=truncated, notes=notes,
                incomplete_reasons=reasons, gbp=S.get('gbp'), report_e3l=rep,
                control_summary=(dict(path=os.path.abspath(control_summary), sha256=file_sha256(control_summary)) if control_summary else None))


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'ctrlprep', 'kselect', 'select', 'eval', 'aggregate', 'selfcheck', 'fingerprint'):
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
            p.add_argument('--reserve', type=float, default=60.); p.add_argument('--manifest-sha', default=''); p.add_argument('--ctrl-ckpt', default='')
            p.add_argument('--rate-every', type=int, default=RATE_EVERY); p.add_argument('--time-from', type=int, default=TIME_FROM)
            p.add_argument('--snap', type=int, default=SNAP_UPDATES)
        if name == 'ctrlprep':
            p.add_argument('--cache', required=True); p.add_argument('--e3r', required=True); p.add_argument('--out', required=True)
            p.add_argument('--manifest-sha', default=''); p.add_argument('--seeds-smoke', default='342101,342202')
        if name in ('kselect', 'select'):
            p.add_argument('--run', required=True); p.add_argument('--ctrl', required=True); p.add_argument('--name', default='final'); p.add_argument('--cache', default='')
            p.add_argument('--bytes-max', type=float, default=0.0, help='smoke only: another byte cap for K*')
            if name == 'select':
                p.add_argument('--kset', default='all', choices=('all', 'kstar'))
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--ckpt', required=True)
            p.add_argument('--snap', default=''); p.add_argument('--qfile', default=''); p.add_argument('--chunk', type=int, default=2)
            p.add_argument('--only', default='', help='comma list of job tags (--smoke only)')
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--runs', default=''); p.add_argument('--control-summary', default='')
            p.add_argument('--e3l-summary', default='')
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True)
        if name == 'fingerprint':
            p.add_argument('--npz', required=True); p.add_argument('--write', default=''); p.add_argument('--expect', default='')
            p.add_argument('--arrays', default='L_k0_cam_nf,L_k0_cam,L_pub_cam')
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    if hasattr(args, 'seeds_smoke'):
        args.seeds_smoke = [int(s) for s in args.seeds_smoke.split(',') if s]
    if getattr(args, 'bytes_max', 0):
        assert args.smoke, '--bytes-max is for the smoke only'
    sha = file_sha256(os.path.abspath(__file__))
    t0 = time.time()
    if args.cmd == 'train':
        return train(args)
    if args.cmd == 'ctrlprep':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return ctrlprep(args, json_dir, index_file)
    if args.cmd == 'kselect':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return kselect(args, json_dir, index_file)
    if args.cmd == 'select':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return select(args, json_dir, index_file)
    if args.cmd == 'selfcheck':
        for m in (L, R):
            assert os.path.dirname(os.path.abspath(m.__file__)) == HERE, f'{m.__name__} imported from {m.__file__}, not {HERE}'
        return R.main(['selfcheck', '--pyroot', args.pyroot, '--houseexpo-json-dir', json_dir, '--houseexpo-index', index_file])
    if args.cmd == 'fingerprint':
        a = ['fingerprint', '--npz', args.npz, '--arrays', args.arrays] + (['--write', args.write] if args.write else []) + (['--expect', args.expect] if args.expect else [])
        return R.main(a)
    if args.cmd == 'eval':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        ckpts = [p for p in args.ckpt.split(',') if p]; snaps = [p for p in args.snap.split(',') if p]
        thr = {}
        for p in ckpts + snaps:
            nm = os.path.splitext(os.path.basename(p))[0]
            thr[p] = os.path.join(os.path.dirname(os.path.dirname(p)), f'threshold{_sfx(nm)}.json')
            assert os.path.exists(thr[p]), f'{p}: no {thr[p]}'
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', '*.json')))]
        assert recs, f'no export records in {args.export}/records'
        if args.only:
            assert args.smoke, '--only is for the smoke'
            want = set(args.only.split(','))
            recs = [r for r in recs if f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}" in want]
        opts = dict(export=args.export, out=args.out, ckpts=ckpts, snaps=snaps, thr=thr, ckpt_shas={p: file_sha256(p) for p in ckpts + snaps},
                    tiers=list(R.EVAL_TIERS), threads=args.threads, chunk=args.chunk, script_sha256=sha, smoke=bool(args.smoke),
                    qtables=(qtables(args.qfile) if args.qfile else None))
        os.makedirs(os.path.join(args.out, 'jobs'), exist_ok=True)
        for r in recs:
            if r.get('failed') or r.get('skipped'):
                _jdump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                            skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))),
                       os.path.join(args.out, 'jobs', f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.json"))
        jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
        print(f'E3-L2 eval: {len(jobs)} jobs, checkpoints {ckpts}, snapshots {snaps}, {args.workers} workers x {args.threads} threads', flush=True)
        for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
                  f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
        return None
    if args.cmd == 'aggregate':
        S = aggregate(args.out)
        runs = [r for r in args.runs.split(',') if r] or None
        S['decision'] = verdict(S, runs, args.control_summary or None, smoke=args.smoke, e3l_summary=(args.e3l_summary or None))
        _jdump(S, os.path.join(args.out, 'summary.json'))
        print('E3-L2 verdict:', S['decision']['verdict'], flush=True)
        for k, c in S['decision']['cells'].items():
            print(f"  cell {k}: K* {c['k_star']} E3-L2 R8 furn {c['e3l2_R8_furn']:.4f} vs NET {c['control_R8_furn']:.4f} (d {c['d_furn']:+.4f}); bytes "
                  f"{c['bytes_median']:.0f} = {c['bytes_ratio']:.3f} x control; cam d {c['cam_d_furn']:+.4f}; guards {c['guards']}", flush=True)
        print('  fresh:', json.dumps(S['decision'].get('fresh')), flush=True)
        print('  gbp:', json.dumps(S['decision']['gbp']), flush=True)
        print('  stages:', json.dumps(S['decision']['stages']), flush=True)
        return S


if __name__ == '__main__':
    main()
