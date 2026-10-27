"""g4_e3l3.py (Claude, 2026-10-27): 批次 E3-L3 "全分辨率、单通道三级的学习隐码 sheaf GBP"（预注册 OVERNIGHT_REPORT D8-63；
bigboy 两块 RTX 6000 Ada，两个种子并行）。
冻结依赖只读 import：g4_e3l.py（E3-L，sha 3dc970d8...：GaBP 两遍 / 逐跳量化 / 闭式与能量核对 / O7 线路字节 / 子树采样 / 阈值原语）、
g4_e3l2.py（E3-L2，sha bab73d64...：固定形状编码器分块、第 200-400 步计时、ctrlprep 对照文件的校验、整件漏报）、
g4_e3r.py（E3-R，sha 1d65d9a8...：TRAIN 缓存、UNet3、features_t、计分、对照读数、判决辅助）、g4_noise_panels.py（2313d75b...）。
本文件不改它们。

底线（D8-54 / D8-63，用户原话的形式）：学习隐码 x_i；二次 sheaf 能量
    E = sum_i sum_v 1/2 Lam_i(v) (x_i(v) - z_i(v))^2 + sum_(i,j) in F sum_v 1/2 w (x_i(v) - x_j(v))^2
上的两遍 GaBP（L.gabp，消息公式与 E3-L 逐字相同），然后每台机器只解码自己的信念；融合与任务无关。
臂（D8-63）：全分辨率（128 x 128 x 32 体素即隐码网格），D = 1，z_i = round(tanh(u_i)) in {-1, 0, +1}（直通取整），
Lam_i = 1[v in 足迹 tau_i]（不学习），已知 = Lam_b > 0（= k > 0，与 E3-R 逐位相同，断言），解码器只有占据头。

操作化（D8-63 没写死、由本实现写死的部分；回复里逐条列出）
  P1 模型 = 两个 E3-R UNet3（R.UNet3，3 层、基宽 16、无归一化、skip=False：不带 E3-R 的线性读出旁路，因为隐码 / 信念不是
     nats 尺度的证据）+ w = exp(omega)，omega0 = log(L.W0 = 1000)（同 E3-L）。构造顺序 enc、dec、log_w，torch.manual_seed(seed)。
     编码器输入 = R.features_t(e_i, 1[触及])（= E3-L：e x 0.05 / 4、log1p(1[触及])、高度），输出 u_i；z_hat = tanh(u_i) x 1[tau_i]
     （足迹外恰为 0）；z_i = round(z_hat)，训练时直通（前向取整、反向恒等），评估时确定性取整（torch.round，半数取偶：
     |z_hat| = 0.5 -> 0）。UNet3 的头初值 N(0, 1e-3^2)，所以开始时 z 全为 0（梯度经直通照常回传）。
     解码器输入 = (x_i, log1p(Lam_b), 高度)，x_i = h_b / Lam_b（Lam_b = 0 处 x = 0），输出 1 通道占据 logit。
  P2 源端势 (h_i, Lam_i) = (Lam_i z_i, Lam_i) = (z_i, 1) 在足迹内、(0, 0) 在足迹外；已经是整数，源端不再量化。线路上每一跳的消息
     用 E3-L 的逐跳量化（L.gabp 的 qmode：训练 'ste'，评估 'round'，步长 1；Lam_q = 0 处 h_q = 0）。"浮点 GBP"= 同一组整数源端势、
     消息不量化（qmode None），只用于闭式核对与信念差报告。
  P3 已知 = 量化信念的 Lam_b > 0。断言：Lam_b > 0 与该分量（或子树）k > 0 逐体素相同——eval 每个分量计不一致体素数（> 0 判
     INCOMPLETE），select 另断言留出房合并 FPR 计数与 ctrl_holdout.json 的对照计数逐个相等；训练探针只记录不一致数。w <= 1 时
     Lam 消息 w / (w + 1) 会取整为 0，断言会捕获（训练中记录 w）。
  P4 损失 = R.masked_bce(占据 logit, 干净占据, 计分范围 ∧ 已知)，计分范围 = 子树私有干净地图的并（E3-R scope），已知 = 解码那台机器
     的 Lam_b > 0（= 子树 k > 0，即 E3-R 的 'scope_known'）；batch 内合并求均值（同 E3-R）。
  P5 数据与日程 = E3-R / E3-L：TRAIN 缓存 cache-e3r-v1（186 训练 / 19 留出），clean 概率 0.1，cam_nf 种子 200-202 均匀，
     样本 = 森林里的连通子树（L.sample_spec：|s| ~ U{1..n}，均匀起点 + 均匀前沿生长），每样本只解码子树里均匀抽的一台机器，
     二面体增广；AdamW（lr 1e-3，wd 1e-4 只在 ndim > 1 的参数上），预热 300 + 余弦，裁剪 1.0，batch 8。
  P6 固定形状编码器（= E3-L2 P4，L2.encode_fixed）：一个 batch 的全部机器按 8 台一块，最后一块补零（补零行足迹为空、丢弃、不进 GaBP）；
     解码器 batch 固定为 8；cudnn.benchmark = True。
  P7 早停探针（只记录不中止；每 --probe-every = 1,000 次更新与最后一次更新各一次；预跑时 --probe-every 400 用来测探针耗时）：
     留出房 4 个（按名字排序后等距取 4 个）x 各一个 cam_nf 抽样（200, 201, 202, 200），整房所有分量走 GaBP（'round'），每台机器解码
     自己的信念，对分量 R8 目标计数：家具 F1 与空闲召回在 logit >= 0 处、以及 F1 最大的 bin（oracle，只报告）；z 的三级占比与熵
     （足迹内）；tanh 饱和率（足迹内 |z_hat| > 0.95）；w；已知不一致体素数；这 4 个 (房, 抽样) 的 O7 字节与 ctrl_holdout.json
     对照字节之比（给了 --ctrl-holdout 时）。同时算 32 个留出样本的 BCE（E3-L2 的 holdout_bce 口径）。距离 deadline 不足一次探针
     耗时时跳过该次探针（记 skipped）。每次探针的耗时进 history.json。
  P8 计时 = E3-L2 P7（L2.step_rate）：每 --log-every 记一个时间点；每步耗时 = 起点 >= --time-from（200）的窗口合计；变异系数 >= 10 %
     改用中位数并警告。链：两卡并行各预跑 400 步（--log-every 50），rate = max(两卡)；更新数 = floor(0.90 x avail / rate)，
     avail = 6000 - 60（余量）- 加载（两卡较大者）- 首窗（两卡较大者）- 探针耗时（两卡较大者）x 探针次数（floor(U0 / 1000) + 1，
     U0 = 未扣探针时的更新数）- 30（保存与收尾）。两种子相同；< 2,000 判 INCOMPLETE。
  P9 中间检查点：第 6,000 次更新另存 checkpoints/upd6000.pt（只在更新数 > 6,000 时；只存，不评估）。
  P10 阈值（select，CPU，只在 TRAIN 留出 19 房 x 抽样 200-202 的 cam_nf 上）：已知不选。占据阈值：合并空闲召回 >= 对照 NET 空闲召回 -
     0.25 pp、家具 F1 最大、同分取最小（L.pick_occ）；不可行时取空闲召回最大的最小 bin，标 feasible_occ = false。R8 = 分量展开
     （每台机器解码自己的信念，对分量 R8 目标）；K0 = 单机自己的势，对照 = ctrl_holdout 的 K0 空闲召回；K0 不可行时用 R8 阈值并标
     k0_fallback_r8。对照读数 = ctrl_holdout.json（g4_e3l2 ctrlprep 口径；L2.check_ctrl 校验 manifest、E3-R 检查点与阈值 sha、
     训练时配对的检查点 sha；运行的证据表 sha = 对照文件的表 sha）。只报告：留出房 O7 字节比的中位（每 (房, 抽样) 比对照字节）。
  P11 评估（CPU，复用 E3-R 导出）：VAL / fresh x 噪声种子 1、2 x cam_nf / cam / clean；R8 用公共系地图（cam 档 = L_pub_cam，不配准），
     K0 用机体系地图；每台机器解码自己的信念、对分量 R8 目标计分，一行一台机器（= E3-R 的分量展开均值）。字节（判决）= 全部有向消息
     的 O7（L.wire_bytes：掩码 Lam_q > 0 一次 + Lam 与 h 两个通道的支撑内值）；只报告：掩码计两次的上界（L.int_bytes）、各机自图
     字节（源端势的 O7）、量化与浮点 GBP 的信念差、机间信念差、z 三级分布与饱和率、整件漏报、按 k 与高度分层（R.scored）、容差 1。
     断言：每个分量 float64 GaBP 与闭式解差 <= 1e-5、能量单调（L.gbp_checks），已知不一致体素数。
  P12 判决（D8-63，写死在 D8_63 里）：杠杆 = VAL cam_nf 4 格都满足 d_furn >= -1 pp、每房字节中位 <= 0.15 x 对照（NET 的 rnet_relay_median
     sum + count）、空闲召回 / 表面 F1 >= 对照 - 0.5 pp、clean 档家具 >= .99、K0_sep >= 对照 K0_sep - 3 pp，且两种子未截断；cam 档 4 格
     均值 < -3 pp 或 fresh 4 格均值 d_furn < -1 pp 时 LEVER 降为 AMBIGUOUS（只降级）；不够 = 任一格 d_furn < -3 pp 或字节 > 0.5 x；
     其余 AMBIGUOUS。INCOMPLETE = 缺任务 / 失败任务 / E1 复现不符 / 缺 run、final、threshold / < 2,000 次 / 两种子更新数不同 /
     缺格、房、对照行、档、K0 行 / 混用脚本或检查点版本 / GaBP 断言失败（"断言失败 -> INCOMPLETE"，比 E3-L2 只挡杠杆更严）/
     已知不一致体素 > 0 或汇总 FPR 与对照之差 > 1e-12（体素级已知掩码已逐位断言；1e-12 只容许按行求均值的浮点次序差）。
     只报告：fresh 字面符号、E3-L / E3-L2 / QPOST4 同格读数（给了它们的 summary 时）。

用法（总是 python -B；CPU 阶段 CUDA_VISIBLE_DEVICES= ）：
  train     --cache C --evidence E --seed S --out R --updates N [--device cuda] [--max-updates M] [--deadline-epoch T] [--ctrl-ckpt E3R/final.pt]
            [--ctrl-holdout CTRL.json] [--expect-gpu 'RTX 6000 Ada']
  select    --run R --ctrl CTRL.json [--workers 10]
  eval      --export X --out O --ckpt R1/checkpoints/final.pt,R2/... [--workers 10]
  aggregate --out O --runs R1,R2 --control-summary E3R/eval/summary.json [--e3l-summary ..] [--e3l2-summary ..] [--e5-summary ..]
  selfcheck --pyroot P
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse, glob, json, math, time, traceback  # noqa: E401,E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g4_e3l2 as L2  # noqa: E402   (frozen; read only; imports g4_e3l, g4_e3r, g4_noise_panels)

L = L2.L
R = L.R
G = L.G
import torch  # noqa: E402
from torch import nn  # noqa: E402

UNK = R.UNK
NBIN = R.NBIN
W0 = L.W0
ENC_CHUNK = L2.ENC_CHUNK               # 8
SNAP_UPDATES = 6000                    # P9
FREE_MARGIN_SELECT = 0.0025            # P10: occupancy threshold, pooled free recall >= control - 0.25 pp
TIME_FROM = 200
PROBE_EVERY = 1000                     # P7
PROBE_HOUSES = 4
SAT = 0.95                             # P7: tanh saturation |z_hat| > 0.95
ARM = 'E3L3'
RUN_FORMAT = 1
BUDGET_S = 6000                        # per seed, load + compile included (D8-63)


def _jload(path):
    with open(path) as fh:
        return json.load(fh)


def _jdump(obj, path):
    with open(path + '.tmp', 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + '.tmp', path)


def file_sha256(p):
    return G.file_sha256(p)


# ================================================================ model, source, decoder input
class E3L3(nn.Module):
    """P1: encoder and decoder = R.UNet3 (3 levels, width 16, skip off), w = exp(omega)"""

    def __init__(self, width=16, w0=W0):
        super().__init__()
        self.enc = R.UNet3(3, width, skip=False); self.dec = R.UNet3(3, width, skip=False)
        self.log_w = nn.Parameter(torch.tensor(math.log(w0)))

    @property
    def w(self):
        return torch.exp(self.log_w)

    def encode(self, f, kn):
        """f [N, 3, X, Y, Z], kn [N, X, Y, Z] (touched) -> z_hat = tanh(u) x 1[tau] (bounded, 0 off the footprint), Lam = 1[tau]"""
        lam = kn.to(f.dtype)
        return torch.tanh(self.enc(f)) * lam, lam

    def decode(self, hb, lb):
        """hb [B, 1, X, Y, Z], lb [B, X, Y, Z] -> occupancy logit [B, X, Y, Z]"""
        return self.dec(dec_input(L.belief_mean(hb, lb)[:, 0], lb))


def dec_input(x, lb):
    """(x, log1p(Lam_b), height) -> [B, 3, X, Y, Z]"""
    zp = torch.as_tensor(R.zplane(x.shape[-1]), device=x.device, dtype=x.dtype).expand(x.shape)
    return torch.stack((x, torch.log1p(lb), zp), 1)


def source(zhat, lam, qmode):
    """P1 / P2: z = round(z_hat) in {-1, 0, +1} (straight-through for 'ste'); potentials h = Lam z [N, 1, *g], Lam [N, *g]; returns (h, Lam, z)"""
    zr = torch.round(zhat)
    z = zhat + (zr - zhat).detach() if qmode == 'ste' else zr
    return (lam * z).unsqueeze(1), lam, z


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    r = ck['recipe']; model = E3L3(r['width'], r['w0']).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return ck, model


def z_stats(z, zhat, lam):
    """P7: counts of z = -1 / 0 / +1 on the footprint, saturated |z_hat| > SAT, footprint size"""
    m = lam > 0
    zz = z.detach()[m]; zh = zhat.detach()[m]
    return np.array([int((zz < -0.5).sum()), int((zz.abs() < 0.5).sum()), int((zz > 0.5).sum()), int((zh.abs() > SAT).sum()), int(m.sum())], np.int64)


def z_summary(c):
    c = np.asarray(c, np.float64); n = max(c[4], 1.0); p = c[:3] / n
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum()) + 0.0
    return dict(z_minus=float(p[0]), z_zero=float(p[1]), z_plus=float(p[2]), z_entropy_bits=ent, tanh_saturation=float(c[3] / n), footprint=int(c[4]))


# ================================================================ training
def forward_batch(model, specs, houses, table_t, qmode, chunk=ENC_CHUNK):
    """encode (fixed shape) every agent of every sample, GaBP on each sample's subtree, decode the sampled agent;
    returns logits, clean occupancy, loss mask (scope & Lam_b > 0), src"""
    feats, kns, labs, sizes = [], [], [], []
    for h, d, sub, aug, dec in specs:
        H = houses[h]
        e, kn = L.evidence_t(H.t_maps[d][list(sub)], table_t)
        feats.append(L.aug_t(R.features_t(e, kn.to(torch.int32)), aug)); kns.append(L.aug_t(kn, aug))
        scope = (H.t_pk.int() & sum(1 << a for a in sub)) != 0
        labs.append((L.aug_t(scope, aug), L.aug_t(H.t_occ, aug))); sizes.append(len(sub))
    zhat, lam, npad = L2.encode_fixed(model, torch.cat(feats).contiguous(), torch.cat(kns), chunk)
    h, lq, z = source(zhat, lam, qmode)
    w = model.w
    hbs, lbs, rows = [], [], []
    o = 0
    for (hh, d, sub, aug, dec), m in zip(specs, sizes):
        hb, lb, _ = L.gabp(h[o:o + m], lq[o:o + m], houses[hh].local(sub), w, qmode)
        i = sub.index(dec); hbs.append(hb[i]); lbs.append(lb[i]); rows.append(o + i); o += m
    hb = torch.stack(hbs); lb = torch.stack(lbs)
    logits = model.decode(hb, lb)
    scope = torch.stack([s for s, _ in labs]); yo = torch.stack([oc for _, oc in labs])
    mask = scope & (lb.detach() > 0)
    return logits, yo, mask, dict(zhat=zhat, z=z, lam=lam, lb=lb.detach(), dec=rows, npad=npad, n=int(lam.shape[0]))


@torch.no_grad()
def holdout_bce(model, specs, houses, table_t, bs):
    model.eval(); tot = cnt = 0.
    for i in range(0, len(specs), bs):
        lg, yo, m, _ = forward_batch(model, [(h, d, s, 0, dc) for h, d, s, _, dc in specs[i:i + bs]], houses, table_t, 'round')
        l = torch.nn.functional.binary_cross_entropy_with_logits(lg, yo.float(), reduction='none')
        tot += float((l * m.float()).sum()); cnt += float(m.float().sum())
    model.train()
    return tot / max(cnt, 1.)


class Probe:
    """P7: the early-stopping probe (record only) on PROBE_HOUSES held-out houses, one cam_nf draw each, whole houses"""

    def __init__(self, ho, ctrl=None, seed=None, n=PROBE_HOUSES):
        order = sorted(range(len(ho)), key=lambda i: ho[i].name)
        pick = [order[int(round(i * len(order) / n))] for i in range(min(n, len(order)))]
        self.items = []
        for t, hi in enumerate(pick):
            Hd = ho[hi]; d = 1 + (t % (len(Hd.draws) - 1)) if len(Hd.draws) > 1 else 0
            Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
            Ts = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]
            kcs = [(Hd.maps[d][cm] != UNK).sum(0) for cm in Hd.comps]
            cb = None
            if ctrl is not None and seed is not None:
                cb = ctrl['bytes'].get(str(seed), {}).get(Hd.name, {}).get(Hd.draws[d])
            self.items.append((Hd, d, Hh, Ts, kcs, cb))

    @torch.no_grad()
    def run(self, model, table_t, want_bytes=True):
        was = model.training; model.eval(); t0 = time.time()
        w = model.w; wf = float(w)
        A = np.zeros(NBIN, np.int64); B = np.zeros(NBIN, np.int64); C = np.zeros(NBIN, np.int64); tr_tot = fr_tot = 0
        zc = np.zeros(5, np.int64); mism = 0; ratios = []; tot_bytes = 0.0
        for Hd, d, Hh, Ts, kcs, cb in self.items:
            e, kn = L.evidence_t(Hd.t_maps[d], table_t)
            zhat, lam, _ = L2.encode_fixed(model, R.features_t(e, kn.to(torch.int32)).contiguous(), kn)
            h, lq, z = source(zhat, lam, 'round')
            zc += z_stats(z, zhat, lam); by = 0.0
            for ci, cm in enumerate(Hd.comps):
                Fl = Hd.local(cm)
                hb, lb, msgs = L.gabp(h[cm], lq[cm], Fl, w, 'round')
                if want_bytes:
                    by += sum(L.wire_bytes(*L.msg_ints(*msgs[ed])) for ed in Fl['directed'])
                lg = L.decode_np(model, hb, lb, 2); lbn = lb.float().cpu().numpy(); T = Ts[ci]
                for i in range(len(cm)):
                    pk = lbn[i] > 0; mism += int((pk != (kcs[ci] > 0)).sum())
                    ob = R.logit_bin(lg[i])
                    A += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NBIN)
                    B += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NBIN)
                    C += np.bincount(ob[pk & T.free], minlength=NBIN)
                    tr_tot += int(T.tr.sum()); fr_tot += int(T.free.sum())
            tot_bytes += by
            if want_bytes and cb:
                ratios.append(by / cb)
        _, _, f1, fr = L.pick_occ(A, B, C, tr_tot, fr_tot, 0.0)
        j0 = int(R.logit_bin(0.0)); jm = int(np.argmax(f1))
        out = dict(furn_f1_at0=float(f1[j0]), free_at0=float(fr[j0]), furn_f1_oracle=float(f1[jm]), free_at_oracle=float(fr[jm]),
                   t_oracle=float(R.bin_edge(jm)), known_mismatch=int(mism), w=wf, bytes=(tot_bytes if want_bytes else None),
                   bytes_ratio_median=(float(np.median(ratios)) if ratios else None), houses=[(it[0].name, it[0].draws[it[1]]) for it in self.items])
        out.update(z_summary(zc)); out['seconds'] = time.time() - t0
        if was:
            model.train()
        return out


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
        if args.expect_gpu:
            assert args.expect_gpu in torch.cuda.get_device_name(dev), f'GPU {torch.cuda.get_device_name(dev)} is not "{args.expect_gpu}"'
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
    ctrl = None
    if args.ctrl_holdout:
        ctrl = _jload(args.ctrl_holdout)
        assert ctrl['manifest_sha256'] == man_sha, 'ctrl_holdout manifest sha != the cache manifest'
        assert args.smoke or L2._table_sha(ev.table) == ctrl['e3r'][str(args.seed)]['table_sha256'], 'evidence table != ctrl_holdout table'
    os.makedirs(os.path.join(args.out, 'checkpoints'), exist_ok=True)
    tr = L.load_cache(args.cache, man, 'train', dev); ho = L.load_cache(args.cache, man, 'holdout', dev)
    assert tr and ho, 'empty train or holdout set'
    load_s = time.time() - t_start
    table_t = torch.as_tensor(ev.table, dtype=torch.int32, device=dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed % (2 ** 32))
    model = E3L3(args.width, W0).to(dev)
    init_sha = L.weights_sha(model)
    convs = [q for _, q in model.named_parameters() if q.ndim > 1]; rest = [q for _, q in model.named_parameters() if q.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=convs, weight_decay=args.wd), dict(params=rest, weight_decay=0.0)], lr=args.lr)
    rng = np.random.default_rng([args.seed, 3]); hrng = np.random.default_rng([20261027, 9])
    hspecs = [L.sample_spec(hrng, ho) for _ in range(args.holdout_samples)]
    probe = Probe(ho, ctrl, args.seed)
    recipe = dict(width=args.width, w0=W0, D=1, alphabet=[-1, 0, 1], encoder='R.UNet3(3, width, skip=False) -> tanh -> round (straight-through) x 1[footprint]',
                  decoder='R.UNet3(3, width, skip=False) on (x, log1p(Lam_b), height), occupancy only', lam='1[footprint] (not learnt)',
                  known='Lam_b > 0 (= k > 0)', wire='E3-L per-hop quantised GaBP messages (step 1)', enc_chunk=ENC_CHUNK, unit=R.UNIT, s_scale=R.S_SCALE,
                  evidence=ev.recipe(), p_clean=R.P_CLEAN, batch=args.batch, lr=args.lr, wd=args.wd, warmup=args.warmup, clip=args.clip, aug='dihedral xy',
                  loss='BCE occupied on scope & Lam_b > 0 (E3-R scope_known)', train_decode='one uniform agent of the subtree per sample',
                  subtree='connected (uniform start + uniform frontier growth)', probe_every=args.probe_every,
                  e3l_sha256=file_sha256(L.__file__), e3l2_sha256=file_sha256(L2.__file__))
    run = dict(format=RUN_FORMAT, stage='train', batch_name='E3-L3', prereg='D8-63', seed=args.seed, updates=args.updates, max_updates=args.max_updates,
               device=str(dev), recipe=recipe, parameters=L.n_params(model), initial_weights_sha256=init_sha, script_sha256=file_sha256(os.path.abspath(__file__)),
               e3l_sha256=file_sha256(L.__file__), e3l2_sha256=file_sha256(L2.__file__), e3r_sha256=file_sha256(R.__file__), noise_panels_sha256=file_sha256(G.__file__),
               cache=os.path.abspath(args.cache), manifest_sha256=man_sha, train_houses=len(tr), holdout_houses=len(ho), deadline_epoch=args.deadline_epoch,
               load_seconds=load_s, smoke=bool(args.smoke), pair=pair, ctrl_holdout=(dict(path=os.path.abspath(args.ctrl_holdout), sha256=file_sha256(args.ctrl_holdout)) if ctrl else None),
               torch=torch.__version__, cudnn=(torch.backends.cudnn.version() if dev.type == 'cuda' else None),
               gpu=(torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'), cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark), pythonpath=os.environ.get('PYTHONPATH'), pid=os.getpid())
    _jdump(run, os.path.join(args.out, 'run.json'))
    print(f'E3-L3 train seed {args.seed}: {len(tr)} train / {len(ho)} holdout houses, {L.n_params(model)} parameters, evidence {ev.desc.get("spec")}, '
          f'{run["gpu"] or "cpu"}, load {load_s:.0f}s', flush=True)
    stop_at = min(args.updates, args.max_updates) if args.max_updates else args.updates
    history = []; acc = np.zeros(2); zc = np.zeros(5, np.int64); marks = {}; truncated = False; u = 0; gn = genc = 0.; snap = None; last_probe_s = 0.0
    t_train = time.time()
    model.train()

    def do_probe(done):
        nonlocal last_probe_s
        if args.deadline_epoch and time.time() + 1.5 * last_probe_s + 30 > args.deadline_epoch - args.reserve:
            history.append(dict(updates=done, probe_skipped='too close to the deadline', seconds=time.time() - t_start)); return
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        hb = holdout_bce(model, hspecs, ho, table_t, args.batch)
        pr = probe.run(model, table_t, want_bytes=not args.no_probe_bytes)
        last_probe_s = time.time() - t0
        rec = dict(updates=done, holdout_bce=hb, probe=pr, probe_seconds=last_probe_s, w=float(model.w.detach()), seconds=time.time() - t_start)
        history.append(rec); _jdump(history, os.path.join(args.out, 'history.json'))
        print(f"  probe upd{done}: holdout bce {hb:.5f} furn F1@0 {pr['furn_f1_at0']:.4f} (free {pr['free_at0']:.4f}) oracle {pr['furn_f1_oracle']:.4f} "
              f"z -/0/+ {pr['z_minus']:.3f}/{pr['z_zero']:.3f}/{pr['z_plus']:.3f} H {pr['z_entropy_bits']:.3f} sat {pr['tanh_saturation']:.3f} "
              f"w {pr['w']:.1f} known mismatch {pr['known_mismatch']} bytes ratio {L2._f(pr['bytes_ratio_median'])} {last_probe_s:.1f}s", flush=True)

    for u in range(stop_at):
        if args.deadline_epoch and time.time() > args.deadline_epoch - args.reserve:
            truncated = True; break
        for g in opt.param_groups:
            g['lr'] = R.lr_at(u, args.updates, args.lr, args.warmup)
        lg, yo, m, src = forward_batch(model, [L.sample_spec(rng, tr) for _ in range(args.batch)], tr, table_t, 'ste')
        loss = R.masked_bce(lg, yo.float(), m.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        genc = float(torch.sqrt(sum((q.grad.detach().float() ** 2).sum() for q in model.enc.parameters() if q.grad is not None)))
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        done = u + 1
        acc += [float(loss.detach()), 1.0]
        if done % args.zstat_every == 0 or done == 1:
            zc += z_stats(src['z'], src['zhat'], src['lam'])
        if args.snap and done == args.snap and args.snap < args.updates:          # P9
            snap = dict(updates=done, path=os.path.join(args.out, 'checkpoints', f'upd{done}.pt'))
            snap['sha256'] = _save_ckpt(model, recipe, args, done, False, run, snap['path'], snapshot=True)
        log_now = done % args.log_every == 0 or done == stop_at
        if log_now:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            marks[done] = time.time() - t_train
            zs = z_summary(zc)
            print(f'upd{done} bce {acc[0] / max(acc[1], 1):.5f} z -/0/+ {zs["z_minus"]:.3f}/{zs["z_zero"]:.3f}/{zs["z_plus"]:.3f} sat {zs["tanh_saturation"]:.3f} '
                  f'w {float(model.w.detach()):.2f} grad {gn:.3f} enc {genc:.3f} agents {src["n"]} pad {src["npad"]} lr {opt.param_groups[0]["lr"]:.2e} '
                  f'{time.time() - t_start:.0f}s', flush=True)
            acc[:] = 0; zc[:] = 0
        if args.probe_every and (done % args.probe_every == 0 or done == stop_at):
            do_probe(done)
            if dev.type == 'cuda':                                          # the probe time is not a training step: shift the timing marks
                torch.cuda.synchronize()
            t_train += last_probe_s if history and history[-1].get('updates') == done and 'probe_seconds' in history[-1] else 0.0
    else:
        u = stop_at
    done = u if truncated else stop_at
    tm = L2.step_rate(marks, args.time_from)
    ck_path = os.path.join(args.out, 'checkpoints', 'final.pt')
    ck_sha = _save_ckpt(model, recipe, args, done, truncated, run, ck_path)
    ks = sorted(marks)
    probes = [h for h in history if 'probe_seconds' in h]
    fin = dict(updates_done=done, updates_planned=args.updates, max_updates=args.max_updates, truncated=truncated, complete=(done == args.updates and not truncated),
               rate_s_per_update=tm['rate'], rate_mean_s_per_update=tm.get('mean_rate'), rate_cv=tm['cv'], rate_warning=tm['warning'], rate_windows=tm['windows'],
               time_from=args.time_from, rate_window_start=tm.get('window_start'), rate_window_start_ok=tm.get('window_start_ok'),
               first_window_s=(marks[ks[0]] if ks else None), first_window_updates=(ks[0] if ks else None), load_seconds=load_s,
               probe_seconds_max=(max(h['probe_seconds'] for h in probes) if probes else None), probes=len(probes),
               train_seconds=time.time() - t_train, weights_sha256=L.weights_sha(model), w=float(model.w.detach()), snapshot=snap,
               checkpoint=ck_path, checkpoint_sha256=ck_sha, probe_last=(probes[-1] if probes else None), seconds=time.time() - t_start)
    _jdump(fin, os.path.join(args.out, 'final.json'))
    print(f'E3-L3 train seed {args.seed} done: {done} updates ({"truncated" if truncated else "complete" if fin["complete"] else "partial"}), '
          f'rate {tm["rate"]} s/update (windows from update {tm.get("window_start")}, >= {args.time_from}: {"ok" if tm.get("window_start_ok") else "NO"}; '
          f'cv {tm["cv"]}{", WARNING " + tm["warning"] if tm["warning"] else ""}), probe max {fin["probe_seconds_max"]} s, w {fin["w"]:.2f}, '
          f'snapshot {snap["updates"] if snap else None}, {fin["seconds"]:.0f}s', flush=True)
    return fin


# ================================================================ thresholds on the held-out TRAIN houses (CPU)
def _sel_task(job):
    """P10 one held-out house: occupancy-logit histograms for R8 (component expansion) and K0, FPR counts, known mismatches, bytes"""
    entry, opts = job
    rec = dict(house=entry['house'])
    try:
        torch.set_num_threads(max(1, opts['threads']))
        ck, model = load_model(opts['ckpt'], 'cpu'); w = float(model.w.detach())
        tb = torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)
        Hd = L2._house(opts['cache'], entry, opts['seeds'])
        Hh = G.House(Hd.occ.astype(np.int8), Hd.slab, Hd.obj, Hd.cell)
        R8T = [G.Target(Hd.private[cm].max(0), Hh) for cm in Hd.comps]; K0T = [G.Target(Hd.private[a], Hh) for a in range(Hd.n)]
        acc = {wh: np.zeros((3, NBIN), np.int64) for wh in ('R8', 'K0')}; tot = {wh: np.zeros(4, np.int64) for wh in ('R8', 'K0')}
        mism = 0; rec['bytes'] = {}; zc = np.zeros(5, np.int64)

        def add(wh, lg, pk, T):
            ob = R.logit_bin(lg)
            acc[wh][0] += np.bincount(ob[pk & T.known & Hh.region & T.tr], minlength=NBIN)
            acc[wh][1] += np.bincount(ob[pk & T.known & Hh.region & ~T.tr], minlength=NBIN)
            acc[wh][2] += np.bincount(ob[pk & T.free], minlength=NBIN)
            unk = ~T.known
            tot[wh] += [int(T.tr.sum()), int(T.free.sum()), int((pk & unk).sum()), int(unk.sum())]
        with torch.no_grad():
            for d in range(1, len(Hd.draws)):
                maps = Hd.maps[d]
                zhat, lam = L.encode_maps(model, list(maps), tb, 'cpu')
                h, lq, z = source(zhat, lam, 'round'); zc += z_stats(z, zhat, lam)
                by = 0.0
                for ci, cm in enumerate(Hd.comps):
                    Fl = Hd.local(cm)
                    hb, lb, msgs = L.gabp(h[cm], lq[cm], Fl, w, 'round')
                    by += sum(L.wire_bytes(*L.msg_ints(*msgs[e])) for e in Fl['directed'])
                    kc = (maps[cm] != UNK).sum(0); lbn = lb.numpy()
                    for i, lg in enumerate(L.decode_np(model, hb, lb)):
                        pk = lbn[i] > 0; mism += int((pk != (kc > 0)).sum()); add('R8', lg, pk, R8T[ci])
                for a, lg in enumerate(L.decode_np(model, h, lq)):
                    add('K0', lg, maps[a] != UNK, K0T[a])
                rec['bytes'][Hd.draws[d]] = float(by)
        rec['acc'] = {k: v.tolist() for k, v in acc.items()}; rec['tot'] = {k: v.tolist() for k, v in tot.items()}
        rec['known_mismatch'] = mism; rec['z'] = zc.tolist()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def pick_r8_k0(A, tot, ctrl_r):
    """P10: R8 and K0 occupancy thresholds (free floor = control - 0.25 pp); K0 infeasible -> the R8 thresholds (k0_fallback_r8)"""
    out = {}
    for wh in ('R8', 'K0'):
        a = np.asarray(A[wh]); t = np.asarray(tot[wh]); fmin = ctrl_r[wh]['free'] - FREE_MARGIN_SELECT
        j, feas, f1, fr = L.pick_occ(a[0], a[1], a[2], int(t[0]), int(t[1]), fmin)
        out[wh] = dict(j_occ=int(j), t_occ=float(R.bin_edge(j)), f1=float(f1[j]), free=float(fr[j]), free_min=fmin, feasible_occ=bool(feas),
                       fpr=float(t[2] / max(t[3], 1)), control=dict(free=ctrl_r[wh]['free'], fpr=ctrl_r[wh]['fpr'], furn_f1=ctrl_r[wh]['furn_f1']))
    k0 = out['K0']
    if k0['feasible_occ']:
        k0['k0_fallback_r8'] = False
    else:
        k0.update(k0_fallback_r8=True, own=dict(j_occ=k0['j_occ'], t_occ=k0['t_occ']), j_occ=out['R8']['j_occ'], t_occ=out['R8']['t_occ'])
    return out


def select(args, json_dir, index_file):
    ctrl = _jload(args.ctrl); runj = _jload(os.path.join(args.run, 'run.json')); sd = str(runj['seed'])
    ckp = os.path.join(args.run, 'checkpoints', 'final.pt'); ck = torch.load(ckp, map_location='cpu', weights_only=False)
    assert L2._table_sha(ck['recipe']['evidence']['table']) == ctrl['e3r'][sd]['table_sha256'], 'run evidence table != paired control table'
    cache = args.cache or runj['cache']; man = _jload(os.path.join(cache, 'manifest.json'))
    L2.check_ctrl(ctrl, cache, runj, sd)
    tp = os.path.join(args.run, 'threshold.json')
    if os.path.exists(tp):
        raise SystemExit(f'{tp} exists (never overwrite)')
    ents = [e for e in man['houses'] if e['holdout']]
    assert sorted(e['house'] for e in ents) == ctrl['houses'], 'holdout houses != the control file'
    t0 = time.time()
    opts = dict(ckpt=ckp, cache=cache, seeds=man['seeds'], threads=args.threads)
    recs = [r for _, r in R._pool(args.workers, _sel_task, [(e, opts) for e in ents], json_dir, index_file)]
    bad = [r for r in recs if r.get('failed')]
    assert not bad, f"select failed: {[(r['house'], r.get('error')) for r in bad]}"
    A = {wh: np.sum([r['acc'][wh] for r in recs], 0) for wh in ('R8', 'K0')}; tot = {wh: np.sum([r['tot'][wh] for r in recs], 0) for wh in ('R8', 'K0')}
    mism = int(sum(r['known_mismatch'] for r in recs))
    cr = ctrl['readouts'][sd]
    assert mism == 0, f'known (Lam_b > 0) differs from k > 0 on {mism} voxels'
    for wh in ('R8', 'K0'):                                              # P3: FPR counts bit-identical to the control's (same known masks)
        assert [int(tot[wh][2]), int(tot[wh][3])] == [int(cr[wh]['counts'][2]), int(cr[wh]['counts'][3])], f'{wh} FPR counts differ from the control'
    sel = pick_r8_k0(A, tot, cr)
    cb = ctrl['bytes'][sd]
    ratios = [r['bytes'][dn] / cb[r['house']][dn] for r in recs for dn in r['bytes']]
    out = dict(stage='select', seed=int(sd), ckpt=ckp, ckpt_sha256=file_sha256(ckp), ctrl=os.path.abspath(args.ctrl), ctrl_sha256=file_sha256(args.ctrl),
               houses=ctrl['houses'], n_houses=len(ents), draws=ctrl['draws'], device='cpu', R8=sel['R8'], K0=sel['K0'], known_mismatch=mism,
               holdout_bytes=dict(median_ratio=float(np.median(ratios)), max_ratio=float(np.max(ratios)), n=len(ratios), report_only=True,
                                  median_bytes=float(np.median([b for r in recs for b in r['bytes'].values()]))),
               z=z_summary(np.sum([r['z'] for r in recs], 0)),
               rule='known = Lam_b > 0 (not selected); occupancy: pooled free recall >= control NET - 0.25 pp, max furniture F1, ties smallest; '
                    'infeasible -> smallest bin of max free (flag); K0 the same on single agents, infeasible -> the R8 threshold (k0_fallback_r8)',
               script_sha256=file_sha256(os.path.abspath(__file__)), seconds=time.time() - t0)
    _jdump(out, tp)
    print(f"E3-L3 select seed {sd}: R8 t {out['R8']['t_occ']} F1 {out['R8']['f1']:.4f} free {out['R8']['free']:.4f} (min {out['R8']['free_min']:.4f}, "
          f"feasible {out['R8']['feasible_occ']}) | K0 t {out['K0']['t_occ']} fallback {out['K0']['k0_fallback_r8']} | holdout bytes ratio "
          f"{out['holdout_bytes']['median_ratio']:.4f} | {out['seconds']:.0f}s", flush=True)
    return out


# ================================================================ eval (CPU): E3-R export, every agent decodes its own belief
@torch.no_grad()
def eval_arm(model, w, thr, table_t, L_pub, L_k0, F, H, K0T, R8T, chunk):
    zp, lp = L.encode_maps(model, L_pub, table_t, 'cpu')
    same = all(a is b for a, b in zip(L_pub, L_k0))
    zk, lk = (zp, lp) if same else L.encode_maps(model, L_k0, table_t, 'cpu')
    hp, lpq, zq = source(zp, lp, 'round')
    jo = thr['R8']['j_occ']
    r8 = []; by = byu = 0.0; mism = 0
    chk = dict(cf_max=0.0, cf_rel_max=0.0, cf_scale_max=0.0, violations=0, components=0, steps=0, cf_cells=0, cf_excluded_tiny=0)
    qd = np.zeros(4); dis = np.zeros(2); lost = 0; wm = np.zeros(2, np.int64)
    for ci, cm in enumerate(F['comps']):
        Fl = L.local_forest(F, cm)
        hb, lb, msgs = L.gabp(hp[cm], lpq[cm], Fl, w, 'round')
        for e in Fl['directed']:
            mi = L.msg_ints(*msgs[e]); by += L.wire_bytes(*mi); byu += L.int_bytes(*mi)
        hf, lf, _ = L.gabp(hp[cm], lpq[cm], Fl, w, None)
        xq = L.belief_mean(hb, lb); xf = L.belief_mean(hf, lf); s = lb > 0
        if s.any():
            dd = (xq - xf).abs().amax(1)[s]; qd += [float(dd.sum()), float(dd.max()), float(s.sum()), float(xf.abs().amax(1)[s].sum())]
        lost += int(((lb == 0) & (lf >= 0.5 * L.DELTA_LAM)).sum())
        c = L.gbp_checks(zq[cm].unsqueeze(1), lpq[cm], Fl, w)
        chk['cf_max'] = max(chk['cf_max'], c['cf_max']); chk['cf_rel_max'] = max(chk['cf_rel_max'], c['cf_rel']); chk['cf_scale_max'] = max(chk['cf_scale_max'], c['cf_scale'])
        chk['violations'] += c['violations']; chk['components'] += 1; chk['steps'] += c['steps']; chk['cf_cells'] += c['cf_cells']; chk['cf_excluded_tiny'] += c['cf_excluded_tiny']
        for i in range(len(cm)):
            for j in range(i + 1, len(cm)):
                both = (lb[i] > 0) & (lb[j] > 0)
                if both.any():
                    dis += [float(((xq[i] - xq[j]) ** 2).sum(0)[both].sum()), float(both.sum())]
        lg = L.decode_np(model, hb, lb, chunk)
        kc = sum((L_pub[a] != UNK).astype(np.int32) for a in cm); lbn = lb.numpy()
        for i in range(len(cm)):
            mism += int(((lbn[i] > 0) != (kc > 0)).sum())                    # P3 assertion (counted; > 0 -> INCOMPLETE)
            p = R.net_readout(lg[i], kc, jo)                                  # known = k > 0 (= Lam_b > 0, asserted above), occupied iff bin >= j
            r8.append([R.scored(p, R8T[ci], H, kc), 1]); wm += L2.whole_miss(p, R8T[ci])
    hk, lkq, zkq = (hp, lpq, zq) if same else source(zk, lk, 'round')
    lgk = L.decode_np(model, hk, lkq, chunk)
    k0, k0s = [], []
    for a in range(len(L_k0)):
        ki = (L_k0[a] != UNK).astype(np.int32)
        assert np.array_equal(lkq[a].numpy() > 0, ki > 0), 'K0 footprint != own known'
        k0.append(R.scored(R.net_readout(lgk[a], ki, jo), K0T[a], H, ki))
        k0s.append(R.scored(R.net_readout(lgk[a], ki, thr['K0']['j_occ']), K0T[a], H, ki))
    rate = []; zc = z_stats(zq, zp, lp)
    for a in range(len(L_pub)):                                               # each agent's own map (source potentials) on the O7 wire
        hi, li = L.msg_ints(hp[a], lpq[a])
        rate.append(dict(self_bytes=L.wire_bytes(hi, li), self_upper_bytes=L.int_bytes(hi, li), mask_bytes=L.mask_bytes(li > 0), footprint=int((li > 0).sum())))
    return dict(R8=r8, K0=k0, K0_sep=k0s, k0_fallback_r8=bool(thr['K0'].get('k0_fallback_r8')), bytes=by, bytes_upper=byu, gbp=chk, w=w,
                known_mismatch=mism, qdiff=dict(sum_absmax=qd[0], max=qd[1], cells=qd[2], sum_float_absmax=qd[3], lost_cells=lost),
                disagree=dict(sum_sq=dis[0], cells=dis[1]), rate=rate, z=zc.tolist(), whole_miss=dict(fn=int(wm[0]), fn_whole_object=int(wm[1])))


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
            ck, model = load_model(p, 'cpu')
            models.append((str(ck['seed']), model, float(model.w.detach()), _jload(opts['thr'][p]),
                           torch.as_tensor(np.asarray(ck['recipe']['evidence']['table']), dtype=torch.int32)))
        rec['tiers'] = {}
        for tn in opts['tiers']:
            L_k0 = list(z[f'L_k0_{tn}']); L_pub = list(z[f'L_pub_{tn}']) if f'L_pub_{tn}' in z else L_k0
            rec['tiers'][tn] = {f'{ARM}_{sd}': eval_arm(model, w, thr, tb, L_pub, L_k0, F, H, K0T, R8T, opts['chunk']) for sd, model, w, thr, tb in models}
        rec['seconds'] = time.time() - t0; rec['peak_mb'] = G._peak_mb()
    except Exception as e:
        rec['failed'] = True; rec['error'] = repr(e)[:500]; rec['traceback'] = traceback.format_exc()[-2000:]
    return rec


def _eval_and_save(job):
    rec = eval_job(job); opts = job[-1]
    d = os.path.join(opts['out'], 'jobs'); os.makedirs(d, exist_ok=True)
    _jdump(rec, os.path.join(d, f"{rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']}.json"))
    return rec


# ================================================================ aggregation and the D8-63 verdict (criteria hard-coded)
D8_63 = dict(L.D8_54, prereg='D8-63', guard_fpr=0.0, free_margin_select=FREE_MARGIN_SELECT, budget_seconds=BUDGET_S,
             supplement='D8-63: full-resolution D = 1 ternary latent, Lam = footprint, known = Lam_b > 0 (FPR asserted bit-identical to the control); '
                        'lever = 4 VAL cam_nf cells d >= -1 pp, bytes <= .15 x, free / surface >= -0.5 pp, clean >= .99, K0 >= -3 pp, not truncated; '
                        'cam 4-cell mean < -3 pp or fresh 4-cell mean < -1 pp demote LEVER only; not enough = any cell < -3 pp or bytes > .5 x; '
                        'INCOMPLETE also on any failed GaBP check or known-mask mismatch')
D8_63['known_select'] = 'none (Lam_b > 0)'


def _summ_arm(A):
    e = dict(R8=R._summ([x for a in A for x in G._expand(a['R8'])]), K0=R._summ([x for a in A for x in a['K0']]), K0_sep=R._summ([x for a in A for x in a['K0_sep']]),
             bytes_median=float(np.median([a['bytes'] for a in A])), report_bytes_upper_median=float(np.median([a['bytes_upper'] for a in A])),
             k0_fallback_r8=any(a.get('k0_fallback_r8') for a in A), known_mismatch=int(sum(a['known_mismatch'] for a in A)))
    q = np.sum([[a['qdiff']['sum_absmax'], a['qdiff']['cells'], a['qdiff']['sum_float_absmax']] for a in A], 0)
    e['quant_belief_diff'] = dict(mean_abs=q[0] / max(q[1], 1), max_abs=max(a['qdiff']['max'] for a in A), lost_cells=int(sum(a['qdiff']['lost_cells'] for a in A)))
    ds = np.sum([[a['disagree']['sum_sq'], a['disagree']['cells']] for a in A], 0)
    e['belief_disagreement_rms'] = float(math.sqrt(ds[0] / ds[1])) if ds[1] else None
    rt = [x for a in A for x in a['rate']]
    e['self_map'] = dict(bytes_median=float(np.median([x['self_bytes'] for x in rt])), upper_bytes_median=float(np.median([x['self_upper_bytes'] for x in rt])),
                         mask_bytes_median=float(np.median([x['mask_bytes'] for x in rt])), footprint_median=float(np.median([x['footprint'] for x in rt])))
    e['z'] = z_summary(np.sum([a['z'] for a in A], 0))
    wm = np.sum([[a['whole_miss']['fn'], a['whole_miss']['fn_whole_object']] for a in A], 0)
    e['whole_object_miss_share'] = float(wm[1] / wm[0]) if wm[0] else None
    e['w'] = float(np.mean([a['w'] for a in A]))
    e['gbp'] = dict(cf_max=max(a['gbp']['cf_max'] for a in A), cf_rel_max=max(a['gbp']['cf_rel_max'] for a in A), violations=sum(a['gbp']['violations'] for a in A))
    return e


def aggregate(out_dir):
    recs = [_jload(p) for p in sorted(glob.glob(os.path.join(out_dir, 'jobs', '*.json')))]
    S = dict(panels={}, failed=[], skipped=[], e1_mismatch=[], script_sha256=sorted({r.get('script_sha256') for r in recs if r.get('script_sha256')}),
             ckpts=sorted({json.dumps(r.get('ckpts'), sort_keys=True) for r in recs if r.get('ckpts')}), gbp={}, known_mismatch={})
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
                g = S['gbp'].setdefault(arm, dict(cf_max=0.0, cf_rel_max=0.0, violations=0, components=0, steps=0, jobs=0))
                g['cf_max'] = max(g['cf_max'], a['gbp']['cf_max']); g['cf_rel_max'] = max(g['cf_rel_max'], a['gbp']['cf_rel_max'])
                g['violations'] += a['gbp']['violations']; g['components'] += a['gbp']['components']; g['steps'] += a['gbp']['steps']; g['jobs'] += 1
                S['known_mismatch'][arm] = S['known_mismatch'].get(arm, 0) + int(a['known_mismatch'])
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
                D['tiers'][tn] = {arm: _summ_arm([r['tiers'][tn][arm] for r in X]) for arm in sorted(X[0]['tiers'][tn]) if all(arm in r['tiers'][tn] for r in X)}
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
    cell = dict(e3l3_R8_furn=a['R8']['furn_f1'], control_R8_furn=c['R8']['furn_f1'], d_furn=a['R8']['furn_f1'] - c['R8']['furn_f1'],
                bytes_median=a['bytes_median'], control_bytes=cb, bytes_ratio=a['bytes_median'] / cb)
    if not judged:
        return cell
    if 'K0_sep' not in a:
        reasons.append(f'{where}: K0 rows missing'); return None
    cell.update(d_furn_tol1=a['R8']['furn_f1_tol1'] - c['R8']['furn_f1_tol1'], free_delta=a['R8']['free_recall'] - c['R8']['free_recall'],
                surface_delta=a['R8']['surface_f1'] - c['R8']['surface_f1'], fpr_delta=a['R8']['unknown_known_fpr'] - c['R8']['unknown_known_fpr'],
                e3l3_K0_sep=a['K0_sep']['furn_f1'], control_K0_sep=c['K0_sep']['furn_f1'], d_K0=a['K0_sep']['furn_f1'] - c['K0_sep']['furn_f1'],
                k0_fallback_r8=a.get('k0_fallback_r8'), w=a.get('w'), z=a.get('z'), self_map=a.get('self_map'), quant_belief_diff=a.get('quant_belief_diff'),
                whole_object_miss_share=a.get('whole_object_miss_share'), known_mismatch=a.get('known_mismatch'))
    if abs(cell['fpr_delta']) > 1e-12:                                       # P3: FPR bit-identical is an assertion, not a margin
        reasons.append(f'{where}: unknown->known FPR differs from the control ({cell["fpr_delta"]:+.3e}; known masks must be identical)')
    cl = D['tiers'].get(C['clean_tier'], {}).get(arm); cam = D['tiers'].get(C['cam_tier'], {}).get(arm)
    cc, _ = _ctrl_arm(Sc, panel, ns, C['cam_tier'], ts)
    if cl is None or cam is None or cc is None:
        reasons.append(f'{where}: clean / cam row (or the cam control) missing'); return None
    cell['clean_R8_furn'] = cl['R8']['furn_f1']; cell['cam_d_furn'] = cam['R8']['furn_f1'] - cc['R8']['furn_f1']; cell['cam_R8_furn'] = cam['R8']['furn_f1']
    g = dict(free=cell['free_delta'] >= -C['guard_free'] - 1e-12, surface=cell['surface_delta'] >= -C['guard_surface'] - 1e-12,
             fpr=abs(cell['fpr_delta']) <= 1e-12, clean=cl['R8']['furn_f1'] >= C['clean_min'], k0=cell['d_K0'] >= C['k0_margin'] - 1e-12)
    cell['guards'] = g; cell['guards_pass'] = all(g.values())
    return cell


def _report_refs(cells, e3l=None, e3l2=None, e5=None):
    """report only: E3-L / E3-L2 / QPOST4 in the same VAL cam_nf cells"""
    rep = {}
    for k in cells:
        ts, ns = k.split('|'); r = {}
        for nm, path, arm in (('E3L', e3l, f'E3L_{ts}'), ('E3L2', e3l2, f'E3L2_{ts}')):
            if path:
                try:
                    a = _jload(path)['panels']['val']['seeds'][ns]['tiers']['cam_nf'][arm]
                    r[nm] = dict(R8_furn=a['R8']['furn_f1'], bytes_median=a.get('bytes_median'))
                except (KeyError, TypeError, OSError):
                    r[nm] = None
        if e5:
            try:
                q = _jload(e5)['panels']['val'][ns]['cells'][ts]['cam_nf']['QPOST4']
                r['QPOST4'] = dict(R8_furn=q['furn'], d_furn=q['d_furn'], bytes_ratio=q['ratio'], bytes_ratio_mask_once=q.get('ratio_mask_once'))
            except (KeyError, TypeError, OSError):
                r['QPOST4'] = None
        rep[k] = r
    return rep


def verdict(S, runs=None, control_summary=None, smoke=False, refs=None):
    """D8-63 (P12). LEVER iff in all 4 VAL cam_nf cells d_furn >= -1 pp, median bytes <= .15 x control and every guard passes (free /
    surface -0.5 pp, FPR identical, clean >= .99, K0 -3 pp), and no seed is truncated. A LEVER is demoted to AMBIGUOUS when the cam-tier
    4-cell mean d < -3 pp or the fresh 4-cell mean d < -1 pp (demotion only; the literal fresh sign is reported). NOT_ENOUGH iff any cell
    d < -3 pp or bytes > .5 x. Else AMBIGUOUS. INCOMPLETE: failed jobs, E1 mismatch, missing run / final / threshold, < 2,000 updates,
    different update counts, any missing cell / house / control row / tier / K0 row, mixed versions, a failed GaBP check (or none),
    a known-mask mismatch, an FPR that differs from the control."""
    C = dict(D8_63)
    reasons = list(S.get('agg_reasons', []))
    if S.get('failed'):
        reasons.append(f"failed jobs: {S['failed']}")
    if S.get('e1_mismatch'):
        reasons.append(f"E1 reproduction mismatch: {S['e1_mismatch']}")
    Sc = _jload(control_summary) if control_summary else None
    if Sc is None:
        reasons.append('control summary (E3-R) not given')
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
        elif not g.get('pass'):
            reasons.append(f'GaBP assertion failed for {ARM}_{ts} (cf_max {g.get("cf_max")}, violations {g.get("violations")})')
        if S.get('known_mismatch', {}).get(f'{ARM}_{ts}', 0):
            reasons.append(f"known mask (Lam_b > 0) != k > 0 on {S['known_mismatch'][f'{ARM}_{ts}']} voxels for {ARM}_{ts}")
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
        lever_cells = all(c['d_furn'] >= C['lever_furn_margin'] - 1e-12 and c['bytes_ratio'] <= C['lever_bytes_ratio'] + 1e-12 and c['guards_pass']
                          for c in cells.values())
        lever = lever_cells and not (C['truncated_blocks_lever'] and truncated)
        if lever_cells and not lever:
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
    rep = _report_refs(cells, **(refs or {})) if cells else {}
    return dict(verdict=v, criteria=C, cells=cells, fresh_cells=fresh, fresh=fresh_info, runs=run_info, truncated_seeds=truncated, notes=notes,
                incomplete_reasons=reasons, gbp=S.get('gbp'), known_mismatch=S.get('known_mismatch'), report_refs=rep,
                control_summary=(dict(path=os.path.abspath(control_summary), sha256=file_sha256(control_summary)) if control_summary else None))


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('train', 'select', 'eval', 'aggregate', 'selfcheck'):
        p = sub.add_parser(name)
        p.add_argument('--houseexpo-json-dir', default=None); p.add_argument('--houseexpo-index', default=None)
        p.add_argument('--smoke', action='store_true'); p.add_argument('--workers', type=int, default=6); p.add_argument('--threads', type=int, default=1)
        if name == 'train':
            p.add_argument('--cache', required=True); p.add_argument('--evidence', required=True); p.add_argument('--seed', type=int, required=True)
            p.add_argument('--out', required=True); p.add_argument('--updates', type=int, required=True); p.add_argument('--max-updates', type=int, default=0)
            p.add_argument('--device', default='cuda'); p.add_argument('--batch', type=int, default=8); p.add_argument('--width', type=int, default=16)
            p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--wd', type=float, default=1e-4); p.add_argument('--warmup', type=int, default=300)
            p.add_argument('--clip', type=float, default=1.0); p.add_argument('--log-every', type=int, default=100)
            p.add_argument('--probe-every', type=int, default=PROBE_EVERY); p.add_argument('--no-probe-bytes', action='store_true')
            p.add_argument('--holdout-samples', type=int, default=32); p.add_argument('--deadline-epoch', type=float, default=0)
            p.add_argument('--reserve', type=float, default=60.); p.add_argument('--manifest-sha', default=''); p.add_argument('--ctrl-ckpt', default='')
            p.add_argument('--ctrl-holdout', default=''); p.add_argument('--expect-gpu', default='')
            p.add_argument('--time-from', type=int, default=TIME_FROM); p.add_argument('--snap', type=int, default=SNAP_UPDATES)
            p.add_argument('--zstat-every', type=int, default=10)
        if name == 'select':
            p.add_argument('--run', required=True); p.add_argument('--ctrl', required=True); p.add_argument('--cache', default='')
        if name == 'eval':
            p.add_argument('--export', required=True); p.add_argument('--out', required=True); p.add_argument('--ckpt', required=True)
            p.add_argument('--chunk', type=int, default=2); p.add_argument('--only', default='', help='comma list of job tags (--smoke only)')
        if name == 'aggregate':
            p.add_argument('--out', required=True); p.add_argument('--runs', default=''); p.add_argument('--control-summary', default='')
            p.add_argument('--e3l-summary', default=''); p.add_argument('--e3l2-summary', default=''); p.add_argument('--e5-summary', default='')
        if name == 'selfcheck':
            p.add_argument('--pyroot', required=True)
    args = ap.parse_args(argv)
    json_dir = args.houseexpo_json_dir or R._HX[0] or G.HOUSEEXPO_JSON_DIR
    index_file = args.houseexpo_index or R._HX[1] or G.HOUSEEXPO_INDEX
    G.setup_env(json_dir, index_file)
    sha = file_sha256(os.path.abspath(__file__))
    t0 = time.time()
    if args.cmd == 'train':
        return train(args)
    if args.cmd == 'select':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return select(args, json_dir, index_file)
    if args.cmd == 'selfcheck':
        for m in (L2, L, R):
            assert os.path.dirname(os.path.abspath(m.__file__)) == HERE, f'{m.__name__} imported from {m.__file__}, not {HERE}'
        return R.main(['selfcheck', '--pyroot', args.pyroot, '--houseexpo-json-dir', json_dir, '--houseexpo-index', index_file])
    if args.cmd == 'eval':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        ckpts = [p for p in args.ckpt.split(',') if p]
        thr = {}
        for p in ckpts:
            thr[p] = os.path.join(os.path.dirname(os.path.dirname(p)), 'threshold.json')
            assert os.path.exists(thr[p]), f'{p}: no {thr[p]}'
        recs = [_jload(p) for p in sorted(glob.glob(os.path.join(args.export, 'records', '*.json')))]
        assert recs, f'no export records in {args.export}/records'
        if args.only:
            assert args.smoke, '--only is for the smoke'
            want = set(args.only.split(','))
            recs = [r for r in recs if f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}" in want]
        jd = os.path.join(args.out, 'jobs')
        if os.path.isdir(jd) and os.listdir(jd):
            raise SystemExit(f'{jd} is not empty (never overwrite)')
        opts = dict(export=args.export, out=args.out, ckpts=ckpts, thr=thr, ckpt_shas={p: file_sha256(p) for p in ckpts}, tiers=list(R.EVAL_TIERS),
                    threads=args.threads, chunk=args.chunk, script_sha256=sha, smoke=bool(args.smoke))
        os.makedirs(jd, exist_ok=True)
        for r in recs:
            if r.get('failed') or r.get('skipped'):
                _jdump(dict(stage='eval', panel=r['panel'], rooms=r['rooms'], index=r['index'], seed=r['seed'], failed=bool(r.get('failed')),
                            skipped=r.get('skipped'), error='export ' + str(r.get('error') or r.get('skipped'))),
                       os.path.join(jd, f"{r['panel']}_r{r['rooms']}_i{r['index']}_s{r['seed']}.json"))
        jobs = [(r, opts) for r in recs if not r.get('failed') and not r.get('skipped')]
        print(f'E3-L3 eval: {len(jobs)} jobs, checkpoints {ckpts}, {args.workers} workers x {args.threads} threads', flush=True)
        for i, rec in R._pool(args.workers, _eval_and_save, jobs, json_dir, index_file):
            print(f"[{i}/{len(jobs)}] {rec['panel']}_r{rec['rooms']}_i{rec['index']}_s{rec['seed']} {'FAILED ' + rec.get('error', '') if rec.get('failed') else ''}"
                  f"{rec.get('seconds', 0):.0f}s peak {rec.get('peak_mb') or 0:.0f} MB  {time.time() - t0:.0f}s", flush=True)
        return None
    if args.cmd == 'aggregate':
        S = aggregate(args.out)
        runs = [r for r in args.runs.split(',') if r] or None
        S['decision'] = verdict(S, runs, args.control_summary or None, smoke=args.smoke,
                                refs=dict(e3l=args.e3l_summary or None, e3l2=args.e3l2_summary or None, e5=args.e5_summary or None))
        _jdump(S, os.path.join(args.out, 'summary.json'))
        print('E3-L3 verdict:', S['decision']['verdict'], flush=True)
        for k, c in S['decision']['cells'].items():
            print(f"  cell {k}: E3-L3 R8 furn {c['e3l3_R8_furn']:.4f} vs NET {c['control_R8_furn']:.4f} (d {c['d_furn']:+.4f}); bytes {c['bytes_median']:.0f} = "
                  f"{c['bytes_ratio']:.3f} x control; cam d {c.get('cam_d_furn', float('nan')):+.4f}; guards {c.get('guards')}", flush=True)
        print('  fresh:', json.dumps(S['decision'].get('fresh')), flush=True)
        print('  gbp:', json.dumps(S['decision']['gbp']), '| known mismatch:', json.dumps(S['decision']['known_mismatch']), flush=True)
        return S


if __name__ == '__main__':
    main()
