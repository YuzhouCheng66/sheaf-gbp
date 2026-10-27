# E3-L3 代码审查：全分辨率、单通道三级的学习隐码 sheaf GBP（D8-63；bigboy 两块 RTX 6000 Ada 并行两个种子）

- 审查者：独立审查（Claude），2026-10-27 22:27–22:44 UTC（时间戳取自 `date -u`）。
- 被审查文件（sha 已逐个核对，bigboy 上 `~/claude-sheaf-gbp/work/scripts/` 里部署的版本也一致）：
  - `scripts/claude_v342/g4_e3l3.py`：c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c
  - `scripts/claude_v342/test_g4_e3l3.py`：8fb747a72a44232735763e7bdede27652e4f7af837248f7bdc08518e2f925ecd
  - `scripts/claude_v342/chain_bigboy_E3L3.sh`：e4827afc333198f6f4410a75169e13ff2bf8eb8731dd05466cb4808ddecc709f
- 冻结依赖（只读）：g4_e3l2.py bab73d64…、g4_e3l.py 3dc970d8…、g4_e3r.py 1d65d9a8…、g4_noise_panels.py 2313d75b…，与链里钉的值相同。
- 审查者的文件都在 `claude_handoff/analysis_20261027/g4/review_e3l3/`，bigboy 上只写了 `~/claude-sheaf-gbp/review_e3l3/`（一次 CPU 单测）。

## 结论：**LAUNCH OK WITH CHANGES**

- `g4_e3l3.py` 没有发现阻断问题。D8-63 的每一条规定都能在代码里找到对应，判据也写死了。GaBP 在全分辨率下与闭式解一致，量化消息可以独立重算，断言也确实会触发（§5）。
- 必改项只有一条，**M1，是运维问题，不是代码问题**：22:39 UTC 我看到 bigboy 上另一个项目（同一 Unix 用户 skim 下的 `~/stt_mem`）正在 **GPU 0** 上间歇地跑短作业。链在开跑前只采样两次空闲，挡不住这种作业。我没有在 bigboy 上跑 GPU 冒烟，原因也是这个。
- **发射前须知，属于预注册风险，不改代码**：线路上 Λ 那一路就是计数通道，它本身已经占掉约 **0.12 倍**对照字节。留给码值的只有约 0.03 倍，相当于每房约 22 KB。冒烟到第 100 次更新时已经是 0.257 倍。所以按 0.15 倍判杠杆，余地比协调者预测的（0.12–0.20 倍）窄得多（§2.2）。是否先告知用户，由协调者决定。

### 发射前例行步骤（协调者）

1. 处理 M1（确认 GPU 0/1 在约 1.8 h 内归本批独用，并加强空闲检查）。
2. 填写 `E3L3_SHA=c7523cc82f6da04497843e8b00559408d573e20377a8e7b99cc773771751928c`。
3. 填写 `E5_SUM_SHA=b33297e3c7e02c911cffe2b54c2d0e0352de5a7f48be2f0bc773c07fafccb46c`。bigboy 上 `data/e3l2-ref/e5_summary.json` 的 sha 已核对，与本机 `g4/e5/summary.json` 相同；`ctrl_holdout.json` 是 f0b55ed9…，与 CTRLH_SHA 相同。
4. 如果改了链，重跑 `test_chain_script_guards`，并重钉链的 sha。

---

## 1. 与 D8-63 的一致性

| 规定（D8-63） | 实现 | 结论 |
|---|---|---|
| 编码器：3 层 3D U-Net（基宽 16），输入 (e_i, 触及, 高度) | `E3L3.__init__` 129 行 `R.UNet3(3, width, skip=False)`；187 行 `R.features_t(e, kn)`，与 E3-L 相同 | ✓。协调者已定 skip=False。头的初值是 N(0, 1e-3)，所以开始时 z 全为 0 |
| ẑ = tanh(u) 有界；足迹外为 0 | 138 行 `torch.tanh(self.enc(f)) * lam` | ✓。冒烟的非平凡码里，足迹外的非零个数是 0（§5.3） |
| z = round(ẑ) ∈ {−1, 0, +1}，训练用直通 | 152–156 行：`'ste'` 前向取整、反向恒等；评估用 `torch.round` | ✓。单测里直通的梯度恒等。\|ẑ\| < 1 时取整结果只能是三级之一 |
| Λ_i = 1[v ∈ τ_i]，不学习 | 137–139 行 `lam = kn`；`kn = L != UNK`（g4_e3l 517–520 行） | ✓。我的脚本断言 Λ 与足迹逐位相等 |
| 二次能量；w = exp(ω) 可学，初值同 E3-L | 130 行 `log_w = log(1000)`，放在无权重衰减的参数组（313–314 行） | ✓ |
| 两遍 GaBP，复用 g4_e3l；断言与闭式一致、能量单调 | 196 / 528 行 `L.gabp`；eval 536 行 `L.gbp_checks`（float64 闭式 + 下降能量） | ✓。任一检查失败判 INCOMPLETE（770–771 行） |
| 逐跳量化，O7 字节（掩码一次 + 支撑内值） | 训练用 `'ste'`，评估用 `'round'`，步长 1；字节用 `L.wire_bytes`（529–530 行）：掩码一次，加 Λ 与 h 两个通道的支撑内值 | ✓。另报只读的上界 `int_bytes` |
| 解码器只读 (x_i, log1p Λ_b, 高度)，只有占据头 | 141–149 行：`dec_input(belief_mean(hb, lb), lb)`；`UNet3` 只出 1 个通道 | ✓ |
| 已知 = Λ_b > 0，与 k > 0 逐位相同 | 读出用 `R.net_readout(lg, kc, jo)`（548 行）。eval 逐分量统计不一致体素（547 行），select 断言不一致为 0、FPR 计数与对照逐个相等（494–496 行），verdict 中不一致或 FPR 差 > 1e-12 判 INCOMPLETE（702、772 行） | ✓。负向测试触发了（§5.2） |
| 损失掩码 = 计分范围 ∧ 已知 | 201 行 `mask = scope & (lb > 0)`；scope 是子树私有地图的并（188 行） | ✓。单测核对了掩码等于 E3-R 的 scope_known |
| 阈值：空闲召回 ≥ 对照 − 0.25 pp，F1 最大，同分取最小；不可行时取空闲召回最大的最小 bin；K0 不可行时回落到 R8 | `pick_r8_k0`（459–472 行）+ `L.pick_occ` | ✓。单测覆盖了可行、同分、不可行、K0 回落 |
| 杠杆：4 格 d ≥ −1 pp，字节 ≤ 0.15 倍，空闲召回 / 表面 F1 ≥ −0.5 pp，FPR 相同，clean ≥ .99，K0 ≥ −3 pp，未截断 | `_cell`（679–712 行）、`verdict`（790–792 行）、`D8_63`（605–610 行） | ✓ |
| 降级：cam 档 4 格均值 < −3 pp，或 fresh 均值 < −1 pp → AMBIGUOUS，只降不升 | 797 行、804 行 | ✓。单测证明 fresh 不会升级 |
| 不够：任一格 < −3 pp，或字节 > 0.5 倍 | 796 行 | ✓ |
| INCOMPLETE：缺任务、断言失败、sha 不符、< 2,000 次更新、两种子更新数不同 | 744–788 行，加上链里的 `check_shas` | ✓ |
| 训练数据、子树、只解码一台机器、批 8、编码器固定形状分块 | `L.sample_spec`、`L2.encode_fixed`（190 行），补零行丢弃 | ✓。单测证明补零不改变结果 |
| 预跑 400 步，只用第 200–400 步计时；更新数 = floor(0.90 × 可用 / 每步)，取两卡较慢者；< 2,000 判 INCOMPLETE | 链 126–149 行；`L2.step_rate` | ✓（见 §4） |
| 只报告：探针（F1、z 分布与熵、饱和率）、w、量化与浮点的信念差、各机自图字节、与 QPOST4 / E3-L / E3-L2 同格对比 | `Probe`（216–268 行）、eval_arm 的 qdiff / rate、`_report_refs` | ✓。E3-L、E3-L2 summary 里的臂名（`E3L_*`、`E3L2_*`）和 E5 的 `QPOST4` 字段都已核对存在 |

## 2. 训练会不会重蹈前两版的覆辙

### 2.1 tanh 饱和与塌缩

- **与 E3-L2 M1 的区别**。E3-L2 的失活来自两点：Λ 无界且没有回拉力；g_z = g_h·Λ 的乘法耦合让外推自我放大。E3-L3 这两点都没有：
  - Λ 固定为足迹指示；
  - ẑ 被 tanh 限在 (−1, 1)；
  - 没有限幅直通。
- **还剩一个较弱的同类机制**。|ẑ| > 0.5 以后，前向已经固定在 ±1，直通却仍把"再往外"的梯度传给 u。这就是虚梯度。
  - 但每个体素的梯度会乘上 1 − tanh²(u) ≈ 4e^(−2|u|)，所以推力随饱和指数衰减。
  - 卷积权重是所有体素共享的，梯度主要来自 \|ẑ\| 接近 0.5 的未饱和体素。
  - 所以饱和率会慢慢上升然后趋于平稳，不会像 E3-L2 那样失控。
- **即使全部饱和，前向仍然是合法的三级码**。代价只是后期可塑性下降，判决读数本身不会失效。
- **冒烟读数**（bigboy，旧版 sha 3dbd3f16，100 次更新）：
  - 第 40 步零码占 0.835；第 60 步回到全 0；第 80 步零码 0.564、饱和 0.317；第 100 步探针上零码 0.277、+1 占 0.723、熵 0.85 bit、饱和 0.299。
  - 早期有一次塌回全 0 又恢复。原因是第 40 步的梯度尖峰（总范数 48，编码器 10）。全 0 时编码器梯度照样能经直通回传，所以能自己恢复。
- **结论**：实现上没有放大因素。饱和率会被探针每 1,000 次更新记录一次，按预注册只记录、不中止。

### 2.2 没有码率项时的字节

- **零不是吸引子**。死区 (−0.5, 0.5) 里，直通把梯度原样传给 ẑ，所以只要解码器那一侧的梯度有一致的符号，体素就会越过 ±0.5。冒烟里 50 次更新内零码就从 100 % 掉到 27 %。
  - 这是"取整 + 直通、没有码率项"这个设计本身的性质（D8-63："码率只由字母表和熵编码决定"），**不是实现放大**。
  - 没有找到额外的放大因素：没有 Λ–h 耦合，支撑固定为足迹，足迹外为 0。码值用上下文自适应编码，常数或平滑的场很便宜。
- **结构性底座（新信息，请协调者注意）**：
  - 线路上 Λ 那一路的消息，就是发送侧的观测计数 k（w = 1000 时逐位精确，§3）。
  - 它和对照 NET 中继的 count 通道是同一个量。E3-R summary 里 VAL cam_nf 每房 count 的中位是 87,555 / 88,859 B，sum 是 664,785–677,607 B，count 约占对照字节的 **0.116–0.118**。
  - 实测：码全为 0 时，探针字节比是 **0.1202**（bigboy 冒烟第 50 步，4 个留出房抽样）。
  - 所以码值只剩 (0.15 − 0.12) × 752 KB ≈ **22.6 KB/房**的余地。
  - 作为参照：
    - 第 100 步时是 0.257 倍，码值约占 0.137 倍；
    - QPOST4（4 级，冻结编码器）总共 0.190–0.196 倍，码值约 0.075 倍。
  - E3-L3 的码值必须不到 QPOST4 码值的一半，而且没有任何码率压力。
- **对预测的影响**：协调者预测 0.12–0.20 倍，下端 0.12 就是全零码的底座。我认为 P(杠杆) 明显低于 .25；最可能的结局是字节在 0.2–0.35 倍之间，判 AMBIGUOUS（> 0.5 倍才判"不够"）。这不是代码缺陷，是否在发射前告知用户由协调者决定。
- **附带的检查**：我在本机把编码器头扰动成近似随机的三级码，在一个 8 机 TRAIN 房上，码值让字节从 116,235 B 增加到 349,302 B（§5.3）。底座在总量里的占比与上面的判断一致。

### 2.3 w 的漂移与已知掩码

- **什么时候会破坏已知掩码**。量化后的 Λ 消息是 round(lt·w/(w+lt))，lt 是 ≥ 1 的整数。lt = 1 时最小，为 w/(w+1)；只要 w > 1，它就取整到 ≥ 1。所以**只有 w ≤ 1 才会破坏已知掩码**（单测 `test_small_w_breaks_known_and_is_caught`）。
- **什么时候前向是精确的整数和**。发送侧最大计数为 s 时，w > 2s² − s 就够了：s = 7 只要 w > 91，s = 11 要 w > 231。w ≈ 1000 时，前向就是精确的 (Σz, k) 中继，w 只通过直通的虚梯度在动。
- **没有保护，但有监测**：
  - 训练日志每 200 次更新打印一次 w；
  - 探针每 1,000 次记录 w 与 `known_mismatch`；
  - select 断言不一致为 0，eval 统计不一致数，verdict 中不一致 > 0 判 INCOMPLETE。
  - 这些我都实测触发过（§5.2）。
- **最坏情形的界**。log_w 用 AdamW，lr 1e-3，无权重衰减。Adam 每步位移 ≲ lr，所以总漂移 ≤ lr 日程的总和（我的脚本 `neg_asserts.py` 算出）：

  | 更新数 | Σ lr | 最坏情形 w ≥ |
  |---|---|---|
  | 8,000 | 3.85 | 21.3 |
  | 10,000 | 4.85 | 7.8 |
  | 12,000 | 5.85 | 2.9 |

  按 bigboy 冒烟的 0.506 s/步，预计约 9,500–10,400 次更新，最坏情形 w 仍大于 7。要越过 1，需要约 13,800 次以上的更新，而且每一步都朝下。
- **经验上**：E3-L 学到的 w 是 1370 / 1525，E3-L2 是 843 / 931，漂移不到 0.5 nat。冒烟里 6 步漂移约 lr 总和的 0.9 倍（向上），100 步内有升有降。
- **结论**：风险很低，监测足够（S1）。

## 3. GaBP 与底线

- **推理不读真值或 scope**：
  - 编码器只读 L_pub / L_k0 的证据与触及（518–520 行），GaBP 只用势与森林，解码器只读 (x, log1p Λ_b, 高度)；
  - 读出的已知掩码 kc 来自输入地图（545 行）；
  - 真值（`R8T`、`K0T`、`H`）只出现在 `R.scored` 与 `whole_miss` 里；
  - 训练里 scope 和占据真值只进入损失。✓
- **解码器只读信念**：R8 读出时每台机器解码自己的 (h_b, Λ_b)，每台机器只解码一次（544 行）；K0 只解码自己的源端势（551 行）。✓
- **全分辨率下与闭式一致、消息可重算**（本机，`nontrivial.py`，8 机 TRAIN 导出房，128 × 128 × 32，非平凡码：−1 / 0 / +1 = 791,289 / 5,735 / 105,680）：
  - 14 条有向消息与我独立写的整数递推逐位相等；与发送侧的整数和 (Σz, k) 相比，不一致格数为 0（最大侧计数 7）；
  - x_b 与 Σz/k 的差 ≤ 2.6e-8；已知不一致 0；
  - float64 GaBP 与闭式解在 261,233 个格上的最大差 **6.2e-13**；能量违例 0 / 16 步；
  - 量化与浮点 GBP 的信念差最大 6.5e-3（就是 O(k/w)）。
  - bigboy 冒烟的 3 个档：cf_max 5.5e-13，违例 0。✓
- **一个提示（不阻断）**：码全为 0 时 cf_max 恰好是 0，GaBP 检查在数学上是空的。所以码的分布必须和 GaBP 结果一起报告，而 eval 已经报了 `z`。

## 4. bigboy 链

| 项 | 证据 | 结论 |
|---|---|---|
| 只用 GPU 0 / 1 | 126、128、155、157 行用 `CUDA_VISIBLE_DEVICES=0` / `1`；22 行 `CUDA_DEVICE_ORDER=PCI_BUS_ID`；102–105 行用 nvidia-smi 检查型号；训练器 `--expect-gpu 'RTX 6000 Ada'`（285 行）。bigboy 实测 GPU 2/3 是 RTX 3090，所以型号断言确实能区分 | ✓ |
| 开跑前两卡空闲 | 111–113 行：瞬时采样，间隔 30 s 再采一次；训练前再查一次（151 行） | **不够，见 M1** |
| 只写 `~/claude-sheaf-gbp/` | LOG、OUT、TMPDIR 都在其下；python 用 `-B`；没有 conda / pip；没有 torch.compile | ✓ |
| 磁盘 | 98–99 行要求 /home 剩余 ≥ 5,000 MB（实测 70,652 MB）。估计输出：每种子 3 个检查点（约 2.7 MB）、96 个 eval JSON、日志，合计几十 MB；191–193 行结束时与 2 GB 比较 | ✓ |
| sha 钉住 | `check_shas` 在开跑、空闲之后、训练前、训练后各跑一次（94、114、152、168 行）；FILL_ 占位会拒绝运行（65 行） | ✓ |
| 锁、不覆盖 | 95–97 行 `mkdir $OUT.lock` 与 `[ -e $OUT ]`；select 与 eval 也有自己的不覆盖检查（实测触发，§5） | ✓ |
| 预跑口径 | 两卡并行各 400 步，`--log-every 50`，`--time-from 200`，只在第 400 步做一次探针；探针在第 400 步的计时点之后，不进计时窗口；`t_train` 扣除探针时间（390 行） | ✓ |
| 更新数公式 | 137–146 行：base = 6000 − 60 − 加载 − 首窗 − 30，取两卡较大者；u0 = floor(0.9·base/rate)；avail = base − 探针 × (u0//1000 + 1)；UPD = floor(0.9·avail/rate)；rate = max(r1, r2)。代入冒烟读数（rate 0.506、加载 16 s、首窗约 31 s、探针 3.8 s）：UPD ≈ 10,350，训练墙钟约 5,320 s，比训练器的停点 5,910 s 早约 600 s（约 11 %）。注意 Python 的 import 时间不在"加载"里，但 60 s 余量与 0.90 系数足够覆盖 | ✓ |
| 每种子 ≤ 6,000 s | 训练器 deadline = TS + 5,970，留 60 s，停在 TS + 5,910；`timeout -k 30 6000` 在 TS + 6,000 发 SIGTERM，再过 30 s 发 SIGKILL | ✓ |
| deadline 与 kill 按 PID | 只有 `timeout`：它把自己放进新的进程组，只向这个组发信号。没有 pkill / killall / 按名字 kill。bigboy 上另一个项目跑在同一个 Unix 用户 skim 下，所以这一点很关键，已核对 | ✓ |
| CPU 阶段 ≤ 10 进程、超时 | select 与 eval 都是 `--workers 10 --threads 1`，加上 OMP 与 MKL 线程 = 1；超时分别是 10,800 s 和 43,200 s。严格数是 10 个 worker 加 1 个空闲的父进程 | ✓（S2） |
| 总时长约 2.6 h | 预跑约 6 min；训练约 100 min；select 2 × 约 4.5 min（bigboy 冒烟 261 s）；eval 96 个任务 × 2 个检查点，单任务单检查点约 96 s，10 个 worker 约 31 min；aggregate 约 1 min。合计约 2.5 h。bigboy 的 CPU 被另一个项目分用时会更长 | 合理 |
| 误伤其他文件或进程 | 没有 rm -rf，不写 work/scripts，不碰 `~/stt_mem` 或 /tmp。nvidia-smi 只读；`sha256sum` / `find` 只读 | ✓。但会和别人的 GPU 作业抢卡，见 M1 |

## 5. 单测与端到端冒烟

### 5.1 单测

- **本机**（CPU，`CUDA_VISIBLE_DEVICES=`）：**13/13 通过**，日志 `review_e3l3/unittest.log`。
- **bigboy**（CPU，nice 10，TMPDIR 指向 `~/claude-sheaf-gbp/review_e3l3/tmp`，与链相同的 PYTHONPATH 和 HouseExpo 环境变量，部署的 sha c7523cc8 / 8fb747a7）：**13/13 通过**，22:39:53–22:40:03 UTC，日志 `review_e3l3/unittest_bigboy.log`。

### 5.2 本机 CPU 端到端冒烟（只用 TRAIN 房，脚本 `review_e3l3/run_smoke.sh`，22:33–22:38 UTC）

- **train**：6 次更新，`--ctrl-ckpt` 与 `--ctrl-holdout` 的配对断言都在。2 次探针，记录了 `known_mismatch 0` 和字节比 0.141；第 4 步存了快照。
- **select**：可行，FPR 计数与对照逐个相等的断言通过。**再跑一次 select 会拒绝**（`threshold.json exists (never overwrite)`）。
- **eval**：3 个档的 GaBP 检查都通过，`known_mismatch 0`。aggregate 判 INCOMPLETE，理由是种子缺失和更新数不足，符合预期。
- **负向断言**（`neg_asserts.py`）：
  - 同一个检查点把 w 设成 0.9：eval 在 3 个档上统计到 1,187,160 / 1,204,984 / 1,149,170 个已知不一致体素 → verdict 判 INCOMPLETE（"known mask … on 3541314 voxels"）；select 抛出 `AssertionError: known (Lam_b > 0) differs from k > 0 on 967556 voxels`。
  - 把一个任务的 cf_max 改成 2e-5：verdict 判 INCOMPLETE（"GaBP assertion failed for E3L3_342101"）。

### 5.3 全分辨率非平凡码（`nontrivial.py`）

- 数字见 §3。cam_nf 档字节 349,302 B，其中全零码的底座 116,235 B。

### 5.4 bigboy GPU 冒烟

- **没有跑**。22:39:15 UTC，GPU 0 上正有另一个项目的作业（见 M1），我不想和它抢卡。
- 实现者的 bigboy GPU 冒烟用的是**旧 sha 3dbd3f16**，见 `e3l3_smoke/bigboy/run.json`；本地冒烟也是这个旧 sha。
- 发射时，链自己的预跑会在终版 sha 上走一遍 GPU 路径。万一失败，在预跑阶段就判 INCOMPLETE，损失只有几分钟（S3）。

---

## 必改项

### M1. GPU 0 / 1 正被另一个项目间歇使用，链的空闲检查挡不住（链 111–113、151 行）

**事实**（只读 ssh，记录在 `review_e3l3/bigboy_gpu_observation.txt`）：
- 22:38–22:40 UTC，bigboy 上同一 Unix 用户 skim 下有另一个项目 `/home/skim/stt_mem`：
  - 一个从约 17:10 UTC 起常驻的 forkserver 池（13 个 python 进程）；
  - 22:37 起，`timeout 1500 python sanity_fusion.py` 逐个启动 `eval_mem.py` / `diag_fixed.py` 短作业。
- 22:39:15 UTC，nvidia-smi 显示 **GPU 0**（GPU-c471e47c）上有 pid 573165 `diag_fixed.py`，占 2,310 MiB，利用率 23 %；22:40:42 UTC 又空了。
- 这些作业没有设 `CUDA_DEVICE_ORDER`，默认的 `cuda:0` 是最快的卡，也就是一块 6000 Ada，所以会落在本批要用的 GPU 0 或 1 上。

**后果**：
- 链只做两次瞬时采样（间隔 30 s），很可能正好落在两个短作业之间，判为空闲。
- 之后别人的作业和本批在同一张卡上并行：
  - 预跑时撞上，每步耗时偏大，更新数变少，但安全；
  - 训练时撞上，每步耗时可能超过 11 % 的余量，导致截断；按预注册，截断挡住杠杆，这一批的判决就被外部负载污染；
  - 同时也拖慢别人的作业。

**要做的**（任选其一，第 1 条必须做）：
1. 发射前向用户确认：stt_mem 那边在本批约 1.8 h 的 GPU 窗口（预跑加训练）里不使用 GPU 0 / 1，比如改用 GPU 2 / 3 或暂停。不能只靠链的检查。
2. 把开跑前与训练前的空闲检查改成**连续空闲窗口**，例如每 15 s 采样一次、连续 20 次（5 min）都没有外来进程才开始：
   ```bash
   gpus_idle_for(){ local n=0; while [ $n -lt $1 ]; do if gpus_busy; then n=0; else n=$((n+1)); fi; sleep 15; done; }
   ```
3. 在预跑与训练期间加一个只记录的监视器：每 60 s 把 U0 / U1 上的 `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory` 追加到 `$OUT/gpu_monitor.log`。循环用 `kill -0 $T1 || kill -0 $T2` 判断训练是否还在（只探测，不发信号）。报告里就能区分"慢是因为别人的作业"还是本身慢。
   - 不要 kill 别人的进程，也不要按名字匹配。

改完以后，重跑 `test_chain_script_guards`（它禁止 `pkill`、`killall`、`rm -rf`；`kill -0` 不在禁止之列），并重钉链的 sha。

## 建议项（不阻断发射）

- **S1（w 的监测）**：
  - 不建议加 w 下限，那会偏离预注册。
  - 建议协调者在训练期间看 `history.json` 每次探针的 `w` 与 `known_mismatch`。
  - w 一旦低于约 100，前向就不再是精确的整数和；低于 1 会破坏已知掩码，这一批就判 INCOMPLETE。
  - 最坏情形的界在 1.0 万次更新时是 w ≥ 7.8（§2.3）。如果预跑给出的更新数 > 13,000，要特别留意。
- **S2（CPU 阶段的礼让）**：
  - bigboy 只有 12 个核，另一个项目也在用 CPU（22:39 有一个进程占 327 % CPU）。
  - 建议 select / eval 前加 `nice -n 10`：不降低本批在空闲时的吞吐，又会把 CPU 让给别人。
  - "≤ 10 进程"严格数是 10 个 worker 加 1 个空闲父进程。如果协调者的上限是按进程总数算的，改成 `--workers 9`。
- **S3（终版 sha 的 GPU 冒烟）**：
  - 实现者的 GPU 冒烟与本地冒烟都是旧 sha 3dbd3f16。
  - 两版之间的差异我无法复核，因为旧文件不在了。CPU 路径已经在终版上跑通（§5）。
  - 如果 GPU 0 空出来，可以在 `~/claude-sheaf-gbp/review_e3l3/` 下用终版跑 ≤ 50 步（`--max-updates 50 --probe-every 50`），确认 cudnn 与固定形状这条路径；不跑也可以，预跑会兜底。
- **S4（训练器记录 GPU UUID）**：
  - 型号断言在 bigboy 上能区分 Ada 与 3090，但不能区分两块 Ada。
  - 可以在链里训练启动后记一行 `nvidia-smi --query-compute-apps=gpu_uuid,pid`，证明两个训练器的 PID 各在 U0 / U1 上。只记录。
- **S5（INCOMPLETE 文本的 JSON 转义）**：`incomplete()`（63 行）把 $1 直接放进 JSON 字符串，原因里如果带双引号，JSON 会失效。目前所有调用都不带引号，属于小问题。
- **S6（报告口径）**：判决报告里请把 z 的三级分布（eval 的 `z`）和 GaBP 检查放在一起。全零码会让闭式核对平凡通过（§3）。

## 附：审查者文件（`claude_handoff/analysis_20261027/g4/review_e3l3/`）

| 文件 | 内容 |
|---|---|
| `unittest.log` | 本机单测 13/13 |
| `unittest_bigboy.log` | bigboy 单测 13/13 |
| `bigboy_shas.txt` | bigboy 上部署脚本的 sha |
| `run_smoke.sh`、`run_smoke.log`、`smoke/` | 本机端到端冒烟：train / select / 再跑 select / eval / aggregate / 负向测试 |
| `neg_asserts.py`、`smoke/neg_asserts.log`、`smoke/neg_select_w09.log` | w = 0.9 与 GaBP 失败触发 INCOMPLETE；w 漂移的界 |
| `nontrivial.py`、`smoke/nontrivial.log` | 全分辨率非平凡码：消息独立重算、闭式解、字节底座 |
| `bigboy_gpu_observation.txt` | M1 的观测记录 |
