# E3-L2 代码审查：sheaf 隐码 GBP，硬容量码率（D8-59；5090 第 3 个、也是最后一个租约）

- 审查者：独立审查（Claude），2026-10-27 17:45–18:05 UTC，只用本机 CPU（`CUDA_VISIBLE_DEVICES=`，同时最多 6 个进程）。没有碰 GPU 和 uk-linux，没有做 git，也没有改被审文件。
- 被审文件（sha256 已核对）：
  - `scripts/claude_v342/g4_e3l2.py`：799445520e0099e5…
  - `test_g4_e3l2.py`：5aaa0cc0c276d507…
  - `chain_day7_E3L2.sh`：d76eb97b20a93…
- 冻结依赖（只读，sha 已核对）：
  - `g4_e3l.py` 3dc970d8…
  - `g4_e3r.py` 1d65d9a8…
  - `g4_noise_panels.py` 2313d75b…
  - `chain_day7_E3L.sh` 4b88d913…
- 审查脚本与输出：`claude_handoff/analysis_20261027/g4/review_e3l2/`。

## 结论：**LAUNCH OK WITH CHANGES**

只有一个必改项 **M1**，而且是实质性的：

- 现在源端的"限幅 + 直通"会让编码器自己走进两种退化态。
  - **上饱和**：Λ 和 h 外漂，梯度爆炸，隐码退化成每通道一个符号位。
  - **下塌缩**：Λ 掉到 0.5 以下后 h_q 恒为 0，隐码不再携带任何信息，编码器梯度变成 0。
- 下塌缩正是 D8-57 那种"支撑塌缩"换了一个外形：前向 Λ_q 被下限托在 1，但 h_q = round(Λ_raw·z) 塌到 0。所以 D8-59 说的"选中格不会塌缩"，在现在的实现里不成立。
- 实现者自己的冒烟检查点已经处在上饱和态（见 §1.1）。

修正的范围：

- 约 15 行，选择规则和 Λ_q 逐位不变。
- 只有选中格里原始 Λ 落在 [1, 15] 之外时，h_q 才和原实现不同。
- 本机已验证：用修正版，同一个冒烟设置下两种退化都不再出现。

M1 改完后满足 §1.4 的验收条件，就可以重钉 sha 发射，不需要再审一轮。除此之外，D8-59 的一致性、补零、GaBP、K 选择与阈值、链脚本都没有发现阻断问题。

---

## 1. 必改项

### M1. 源端限幅直通没有回拉力：上饱和后下塌缩，隐码失活（`g4_e3l2.py` 第 150–167 行）

**现在的代码**

```
h  = lam * z                                  # 第 154 行：h 用原始的连续 Λ
lr = clamp(round(lam), 1, 15); hr = clamp(round(h), -7, 7)   # 第 157、159 行
ste: h·gate + (hq - h·gate).detach(),  lam + (lq - lam).detach()   # 第 163–165 行：两条路径都是恒等直通
```

**机制**（第一性原理，§1.2 的探针逐条印证）

1. **Λ 上限**：Λ > 15.5 时前向恒为 15。直通仍把"再加大 Λ"的梯度原样传回，没有任何量把它拉回来。AdamW 每步给 a 的位移约等于 lr，和梯度大小无关，所以 Λ 会持续外漂。
2. **h 限幅**：|h| > 7.5 时前向恒为 ±7。直通又经 h = Λz 把"再往外推"的梯度同时传给 z 和 Λ。
   - g_z = g_h·Λ，g_Λ = g_h·z，两者互相放大，编码器梯度范数随之爆炸。
   - 解码器只看到 x = ±7/Λ_q，也就是每通道一个符号位。
3. **Λ 下限**：选中格的原始 Λ < 0.5 时，前向 Λ_q 被托在 1，但 h = Λ_raw·z 取整后成了 0，于是 x = 0。
   - 这时 g_z = g_h·Λ_raw ≈ 0，Λ 也没有回拉力，编码器梯度变成 0。
   - 选中格只传"Λ_q = 1、h_q = 0"，剩下的信息只有支撑图样。
4. **梯度裁剪挡不住**：裁剪只改各参数之间的比例，AdamW 的步长本来就与梯度尺度无关，而漂移方向始终一致。尖峰时全局裁剪还会把解码器的梯度一起压到约 1/1000。

### 1.1 证据一：实现者自己的冒烟检查点

用 `implementer_smoke_ckpt_stats.txt` 读 `e3l2_smoke/runs/s342101/checkpoints/*.pt`，在留出 TRAIN 房、抽样 200 上统计选中格：

| 检查点 | K | 选中格原始 Λ 中位 | Λ > 15.5 占比 | \|h\| 被限幅占比 | 平均 \|x\| |
|---|---|---|---|---|---|
| upd8 | 128 / 512 / 2048 | 43.1 / 40.7 / 28.7 | 100 / 96 / 92 % | 100 % | 0.467–0.481（= 7/15，即符号码） |
| final（16 次） | 128 / 512 / 2048 | 5.0 / 4.5 / 3.4 | 0 % | 99 / 97 / 85 % | 1.5–2.3 |

实现者冒烟里记下的裁剪前梯度从 2.9 升到 107、245、327；BCE 从 1.38 升到 1.95；留出占据 BCE 是 2.78。对照：同一数据上 E3-L 的冒烟，梯度一直在 3–9，留出占据 BCE 是 0.61。

### 1.2 证据二：探针复现与修正对照

脚本 `review_e3l2/grad_probe.py` 是训练器的仪表化复刻，设置与实现者冒烟相同：E3-L 冒烟 cache（2 个 TRAIN 房）、batch 4、预热 4、lr 1e-3、24 次更新、种子 342101。逐步记录在 `probe_*.log`，汇总在 `probe_summary.txt`。

| 变体 | 编码器梯度最大 | 后期编码器梯度 | 选中格原始 Λ 的走向 | 结果 |
|---|---|---|---|---|
| **base**（被审代码） | **1,890**（第 12 次），总范数 2,283 | 第 19–24 次 **0.0** | 4.7 → 53（100 % > 15.5，第 7–10 次）→ 0.00（94–100 % < 0.5，第 14–24 次） | 第 7–12 次 \|h\| 100 % 被限幅，最大 4,410；第 13 次 loss 19.8（占据 12.5）；之后 h_q ≈ 0，隐码失活 |
| upper1s（只在上界单侧直通） | 15.6 | 第 10–15 次 0.0 | → 0.00（第 10 次起 100 % < 0.5） | 下塌缩 |
| both1s（前向不变，Λ 与 h 都单侧直通） | 3.5 | 第 10–12 次 0.0 | → 0.00（第 10 次起 100 % < 0.5） | 下塌缩：Λ 经 h = Λz 这条路仍被往下推 |
| noclamp（K 取极大、不限幅，等于 E3-L 去掉码率项） | 4.6 | 0.1–1.5 | 4–12 | 不发散 |
| **cons1s（推荐的 M1）** | **15.6** | **0.1–3.1，一直不为 0** | 原始 Λ 在 0.2–0.7 附近；前向 Λ_c 恒 ≥ 1，所以无害 | 不发散，隐码一直有梯度；\|z\| 最大从 12 回落到 10.5 |

读法：

- 只要 h 用的是原始 Λ，只修梯度（both1s）也挡不住下塌缩。
- 必须让 h 用限幅后的 Λ 来组：h = Λ_c·z，Λ_c ∈ [1, 15]。这也是 D8-59"源端势 (Λ_q z, Λ_q)、Λ_q ∈ [1, 15]"的直接读法。
- 探针只有 2 个房、24 步，学习率也比正式跑激进，所以它证明的是机制，不预测精度。
- 正式跑预热 300 步只会推迟漂移，不会消除漂移：没有回拉力是结构性的，AdamW 的步长与梯度尺度无关。

### 1.3 最小修正

本机已验证，代码见 `review_e3l2/fix_check.py` 里的 `source_topk_m1` 与 `_ClampOneSided`：

```python
class _ClampOneSided(torch.autograd.Function):           # 前向 clamp；越界时只放行把值拉回区间的梯度（同 E3-L 审查 S1 的单侧直通）
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x); ctx.lo = float(lo); ctx.hi = float(hi); return x.clamp(lo, hi)
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = torch.where(x > ctx.hi, g.clamp(min=0.0), g)
        return torch.where(x < ctx.lo, g.clamp(max=0.0), g), None, None

def source_topk(z, lam, K, qmode, clamp=True):           # clamp=False 分支保持原样（E3-L 语义）
    sel = topk_select(lam, K); s = sel.to(lam.dtype)    # 选择仍按原始连续 Λ，不变
    lam_c = torch.where(sel, _ClampOneSided.apply(lam, LAMQ_MIN, LAMQ_MAX), lam) if clamp else lam
    h = lam_c.unsqueeze(1) * z                           # h 由限幅后的 Λ 组成：h_q / Λ_q ≈ z
    if qmode is None:                                    # 浮点势（GaBP 闭式核对 / 浮点信念）同样用 Λ_c
        return h * s.unsqueeze(1), lam_c * s, sel
    lq = DELTA_LAM * torch.round(lam_c / DELTA_LAM) * s
    hq = (DELTA_H * torch.round(h / DELTA_H)).clamp(-HQ_LIM, HQ_LIM) * s.unsqueeze(1)
    if qmode == 'round':
        return hq, lq, sel
    hg = _ClampOneSided.apply(h, -HQ_LIM - 0.5, HQ_LIM + 0.5) * s.unsqueeze(1)
    return hg + (hq - hg).detach(), lam_c + (lq - lam_c).detach(), sel
```

`fix_check.py` 在 50 组随机输入、`round` 与 `ste` 两种模式下核对了：

- 选择与 Λ_q 和被审代码逐位相同；
- 选中格上原始 Λ ∈ [1, 15] 处 h_q 逐位相同；
- 选中格 Λ_q ≥ 1，|h_q| ≤ 7。

日志在 `fix_check_log.txt`。

**和 D8-59 的关系**

- 不改单变量、K 抽样、选择规则，也不改 Λ_q 的值域与 softplus 参数化。
- "限幅用直通"仍然成立。单侧直通是 E3-L 审查 S1（`A_FLOOR`）的先例。
- 前向只在原始 Λ 越界的选中格上改变 h，使它满足信息形式 h = Λ_q z。
- 协调者应把这一条作为 P2 的修订，在发射前写进 D8-59 的操作化记录。

### 1.4 M1 的验收（改完后由实现者 / 协调者执行；满足即可发射）

1. 改 `test_topk_and_clamp_ste`。它第 44–48 行断言 h_q = clamp(round(Λz))，并断言 Λ 的梯度处处恒等，这两条要换成新语义。
2. 加三个单测：
   - **死区回归**：选中格原始 Λ = 1e-3、z = 3 时，应得 Λ_q = 1、h_q = 3；
   - **单侧梯度**：Λ > 15.5 时只放行使 Λ 下降的梯度；Λ < 1 时只放行使 Λ 上升的梯度；|h| > 7.5 时只放行使 |h| 收缩的梯度；
   - **浮点势一致**：`qmode=None` 返回 (Λ_c z·s, Λ_c·s)。
3. 其余单测保持全过，尤其是 `test_forward_equals_e3l_when_unconstrained`（clamp=False 不受影响）和补零、GaBP、判决各项。
4. 用本审查的设置复跑 CPU 冒烟（`grad_probe.py` 的 cons1s 等价于修正版，也可以直接对修正后的文件跑 base 变体）。24 次更新内要同时满足：
   - 编码器梯度最大 < 50；
   - 第 12 次以后编码器梯度不为 0；
   - 选中格 h_q ≠ 0 的占比始终 > 0。
5. 重算 `g4_e3l2.py` 的 sha 并填进 `E3L2_SHA`。

---

## 2. 逐项检查

### 2.1 与 D8-59 的一致性

| 项 | 结论 | 证据 |
|---|---|---|
| 单变量：top-K 硬容量、无 λ、无熵模型损失 | ✓ | 损失 = `lk + lo`（第 322 行）；模型不含 `ent`（第 111–123 行）；优化器两组，没有熵模型组（第 291–292 行） |
| 编解码器结构、w 参数化与初值 | ✓ | 复用 `L.Enc` / `L.Dec` / `L.E3L.encode/decode/w`。同一种子下，E3L2 与 E3L 共有参数的初始化逐位相同（熵模型没有随机初始化，不改变 RNG 次序），参数 552,604 对 552,686，差的是熵模型的 82 个 |
| 数据、BCE、增广、子树、优化器与日程 | ✓ | `L.sample_spec` 之后再抽 K（第 190–192 行）；`L.bce_terms`；lr 1e-3、wd 1e-4、预热 300、裁剪 1.0，与 E3-L 默认值一致 |
| O7 字节 | ✓ | 训练、kselect、eval 都对全部有向消息求 `L.wire_bytes(L.msg_ints(..))`（第 468、640 行） |
| 评估 | ✓ | R8 每机一行、K0 / K0_sep、三档、容差 1、分层 strata、整件漏报，与 E3-L 的 `eval_arm` 逐项对应（第 625–686 行）。K0 用该机自己在 K* 下的 top-K 消息 |
| 固定形状 8 台一块、补零不进 GaBP | ✓ | 见 §2.3 |
| 预跑 400 步、只用第 200–400 步计时、CV ≥ 10 % 时警告并取中位 | ✓ | `step_rate`（第 238–252 行）；链第 108–109 行用 `--max-updates 400 --log-every 50 --time-from 200`，得到 4 个窗口 |
| 更新数 = floor(0.90·avail / rate) | ✓ | 链第 113–117 行 |
| K 与阈值在租约外 CPU 上选 | ✓ | 训练器不再选阈值；链第 139–153 行 |
| 第 6,510 次快照 | ✓ | 第 330–332 行，只报告 |
| 阈值规则：空闲召回 ≥ 对照 − 0.25 pp；FPR ≤ 对照 + 0.15 pp；K0 不可行时回落到 R8 阈值 | ✓ | 第 564–566、591、599 行；`k0_thresholds` 在第 553–561 行。冒烟复核：free_min = 0.99820 − 0.0025 = 0.99570，fpr_max = 0.00690 + 0.0015 = 0.00840，与 threshold.json 一致 |
| 判据：杠杆、保护、cam 主档降级、fresh 只降级、不够、INCOMPLETE | ✓ | `verdict`（第 899–981 行），常数直接取 `L.D8_54`（0.15、−1 pp、0.5、−3 pp、+0.3 pp、.99）。单测覆盖边界与各种缺失情形 |
| 只报告项 | ✓ | 整条曲线（VAL 与 fresh 的 cam_nf）、等步数检查点、E3-L 读数、零学习 FR1（只用表 A）、量化与浮点信念差、w、strata。冒烟里零学习比 Q3 = 0.53、VOTE = 0.32，与 R-E 的 0.51 / 0.33 吻合。E3-L summary 的取值路径核对过（.796 / .786 / .741 / .730，与 D8-57 一致） |

### 2.2 top-K 与直通

- **只在触及格内选**：未触及格的打分取 −inf，结果再与触及掩码相与（第 143–146 行）。单测 `test_topk_and_clamp_ste` 覆盖了 K 大于触及数和逐机不同 K 的情况。
- **K 的抽样**：`round(exp(U[log 128, log 2048]))`，每个样本一个 K，样本里所有机器相同（第 185–203 行）。单测里对数直方图 8 个桶的偏差都 < 1 %。
- **梯度路径**：
  - Λ 在全部触及格上恒等直通。这是 E3-L 的门直通，未选格靠它学"该不该被选"，这一点合理。
  - h 只在选中格上直通。
  - 问题出在限幅上，见 M1。
- **梯度裁剪与优化器**：全局范数裁剪 1.0 加 AdamW，挡不住漂移，理由见 M1 机制第 4 条。修正后探针里编码器梯度最大 15.6，被审代码是 1,890。

### 2.3 补零（第 170–182、204–205 行）

- 补的行特征全 0、kn 全 0，所以 Λ = 0。它们在 `[:N]` 处被切掉，然后才进 top-K 与 GaBP。
- 编码器没有归一化层，补零行不会影响真实行。
- 字节只在被解码的真实行（`dec` 下标 < N）上算。
- 单测核对过：分块与不分块的输出 allclose；GaBP 调用的机器数是 [3, 2, 1]，补的 2 行没有进去。

### 2.4 GaBP 与底线

- GaBP 原样调用冻结的 `L.gabp`，消息公式与 E3-L 逐字相同。
- 闭式核对与能量单调用 `L.gbp_checks`，作用在选中格的浮点势上。冒烟结果：cf_max 6.2e-13，违例 0，3 个分量。
- 推理只读 `L_pub` / `L_k0`。`R8T`、`K0T`、`kc` 只用于计分（第 657–663 行）。
- 解码器只读 (h_b, Λ_b)，也就是 belief_mean、log1p(Λ_b) 和高度。

### 2.5 K 选择与阈值（第 387–611 行）

- **只用 TRAIN 留出房**：留出标志取自 TRAIN cache 的 manifest。kselect 和 select 都断言房集合与 ctrl 文件相同，并断言证据表 sha 与配对种子相同。
- **ctrlprep 的对照字节**：sum + count 两种中继消息的 `G.int_ctx_bytes` 在全部有向边上求和（第 400 行），与 E3-R eval 的 `sums_for` 是同一算法（`g4_e3r.py` 第 911–912 行），也就是 D8-54 口径。
- **ctrlprep 的对照读数**：`L.control_holdout`，逐房计数后合并，等价于 E3-L 一次性合并。
- **复现**：本机重跑 ctrlprep，结果与实现者的 `ctrl_holdout.json` 在 bytes 和 readouts 上逐位相同。
- **复算 K 比值**：每 (房, 抽样) 比值的中位，在 K = 128 / 512 / 2048 上与 kselect.json 逐位相同（0.02210 / 0.05408 / 0.16696）。K* = 1024，因为 0.090 ≤ 0.12 < 0.124。
- **曲线**：全部 K 的阈值都只在 TRAIN 留出房上选。VAL 上其余 K 只算 R8 与字节，只报告，不进判决。

### 2.6 链脚本（与 `chain_day7_E3L.sh` 逐段对照）

- **安全性**：
  - sha 钉住了 3 个脚本、3 个映射模块、index、pyroot、manifest、fit、E3-R summary / 检查点 / 阈值、导出记录、ctrl_holdout、E3-L summary、E5 量化器；
  - FILL_ 检查覆盖 8 个占位；
  - 有锁，`[ -e $OUT ]` 保证不覆盖；
  - 自检、SOLO 等待都在；
  - 没有 pkill，`timeout -k 30` 只信号它自己的子进程组；
  - 只写 `runs/e3l2*` 与 `runs/chain_day7.log`。
  - 与 E3-L 链相同，没有发现新缺口。
- **预跑窗口确实排除了 cudnn 调优**：
  - 编码器每块都是 (8, 3, 128, 128, 32)；
  - 解码器 batch 固定为 8，留出评估 64 = 8 × 8；
  - 二面体增广不改形状（128 × 128）；
  - top-K 和 GaBP 不走 cudnn；
  - 所以调优只发生在第 1 步，落在第一窗口（第 50 步的计时点）里。第 200–400 步只含稳态，加上每 50 步一次 `measure_rate` 的 CPU 开销，正式跑里这个开销的密度也相同。
  - `probe_check` 里的可变形状调优发生在 `t_train` 之前。
- **avail 的扣除**：avail 扣了预跑的加载时间和整个第一窗口（50 步 + 调优），比 R-E 的"扣 20 s"更保守。没扣的是 probe_check、模型初始化和 ctrl 检查点加载，这几项靠 0.90 的安全系数吸收（见 S4）。
- **时长估算**：按 E3-L 实测，稳态约 0.155–0.17 s/步、加载约 4 s、固定形状下第一窗口约 30 s：
  - UPD ≈ 0.9 × (3000 − 60 − 4 − 30 − 60) / 0.165 ≈ 15,500；
  - 每种子约 70 + 2,563 ≈ 2,630 s，截断线是 2,910 s（SDE − reserve）；
  - 预跑约 110–150 s；
  - 租约合计约 5,400 s，< 7,150 s。
  - 两种子同一 UPD。租约结束后 GPU 不再使用：后续步骤都设了 `CUDA_VISIBLE_DEVICES=`，main 里也再设一次。
- **租约外 CPU**：
  - 顺序是 kselect → select（全部 K）→ 快照的 kselect / select → eval → aggregate；ctrlprep 在发射前做好并钉 sha。
  - 超时：kselect 3,600 s、select 10,800 s、快照各 3,600 s、eval 14,400 s、aggregate 1,800 s。
  - 估计实际耗时：
    - select 要对 9 个 K 各解码一遍，每种子约 15–20 分钟；
    - eval 每个任务要算 K* 全量 + 8 个 K 的曲线 + 快照，约为 E3-L 的 2.5–4 倍，E3-L 是 16 分钟，所以约 40–65 分钟；
    - 合计约 1.5–2 小时，比 D8-59 写的约 65 分钟多。
  - 每一步的超时都在估计值的 2 倍以上，不会误杀（见 S3）。

### 2.7 单测与本机冒烟

- **单测**：`test_g4_e3l2.py` 13/13 通过，17:46:17–17:46:49 UTC，日志 `review_e3l2/test_log.txt`。
- **端到端冒烟**：只用 TRAIN 房（E3-L 冒烟 cache，加上实现者已有的 TRAIN 导出 `train_r1_i0_s203`），17:52–17:59 UTC，脚本 `review_e3l2/run_smoke.sh`，输出在 `review_e3l2/smoke/`。
  - 依次跑了：train 6 次更新（快照在第 4 次）、ctrlprep、kselect、select（全部 K）、快照的 kselect / select、eval（含曲线、快照、零学习字节）、aggregate。各步 exit 0。
  - 判决按预期是 INCOMPLETE：6 < 2,000，并缺种子 342202、缺 VAL / fresh。GaBP 断言通过。
- **断言**（`neg_asserts.py`，日志 `neg_asserts_log.txt`）：下面 6 个护栏都按预期抛出：
  - manifest sha 不符；
  - 证据表 B 配 E3-R 342101 检查点（配对断言）；
  - 非训练种子；
  - kselect 的留出房集合与 ctrl 不符；
  - select 的证据表 sha 与 ctrl 不符；
  - 非冒烟时用了 `--bytes-max`。

---

## 3. 建议项（不阻断发射）

- **S1**：kselect / select 再断言两件事：
  - `ctrl['manifest_sha256']` 等于当前 cache 的 manifest；
  - `ctrl['e3r'][sd]['ckpt_sha256']` 等于链里钉住的 CK1 / CK2。

  现在只靠钉住 ctrl 文件的 sha 来间接保证。
- **S2**：只报告的"量化与浮点信念差"现在拿原始 Λ·z 当浮点参照，把限幅和量化混在一起。冒烟里 mean_abs 是 9.7–38，远大于量化本身。M1 让浮点势也用 Λ_c 后会好很多；报告里应写明这个参照只差"限幅 h 与取整"。
- **S3**：把租约外 CPU 阶段的预计时长改为约 1.5–2 小时，写进发射记录。D8-59 写的约 65 分钟偏乐观。超时设置不用改。
- **S4**：截断余量约 11 %。如果正式跑的稳态每步比预跑慢超过约 11 %，种子就会被 deadline 截断：判决仍会给出，但截断会阻止判 LEVER。可以在发射记录里注明；如果想更稳，可以把 probe_check 的耗时计入 avail。按 D8-59 原公式执行也可以接受。
- **S5**：`step_rate` 在不足 4 个 ≥ 200 的窗口时，会退回"除第一窗口外的全部窗口"（第 242–243 行）。这个退路是给测试和冒烟用的，正式预跑不应走到。建议预跑时在 updates.json 里记录所用窗口的起点，链里看到起点 < 200 时写警告。
- **S6**：给 `measure_rate` 加两个只报告字段：选中格 h_q ≠ 0 的占比，以及 |h| 被限幅的占比。这样万一训练里出现饱和或塌缩，rate_curve.json 里能直接看到。这是最后一个租约，诊断信息只有这一次机会记下来。

---

## 4. 文件

- 本报告：`claude_handoff/analysis_20261027/g4/code_review_e3l2.md`
- 单测日志：`g4/review_e3l2/test_log.txt`
- 梯度探针：
  - 脚本：`g4/review_e3l2/grad_probe.py`（变体 base / upper1s / both1s / noclamp / cons1s）
  - 日志：`probe_*.log`、`probe_*.json`
  - 汇总：`probe_summary.txt`（注意：cons1s 行的 \|h\| 统计按原始 Λ·z 计，不是修正版实际发送的 Λ_c·z）
- 实现者冒烟检查点的统计：`g4/review_e3l2/implementer_smoke_ckpt_stats.txt`
- M1 修正核对：`g4/review_e3l2/fix_check.py`、`fix_check_log.txt`
- 端到端冒烟：`g4/review_e3l2/run_smoke.sh`，输出在 `smoke/`
- 断言检查：`g4/review_e3l2/neg_asserts.py`、`neg_asserts_log.txt`、`neg_asserts_full.txt`；中间文件在 `smoke_neg/`
