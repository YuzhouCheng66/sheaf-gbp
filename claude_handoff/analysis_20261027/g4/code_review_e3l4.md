# E3-L4 代码审查：在固定的有限 w* 下训练学习隐码 sheaf GBP（D8-68；uk-linux 5090，一个 7,200 s 租约）

- 审查者：独立审查（Claude），2026-10-28 03:40–04:13 UTC（时间戳取自 `date -u`）。
- 被审查文件（本机 sha 已逐个核对；uk-linux `/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/scripts/` 上部署的版本 03:52 UTC 只读核对，也一致）：
  - `scripts/claude_v342/g4_e3l4.py`：54428ba6b4ccddf0445c9e1f6fa63e35950dd3905046e89546c10f51aa9f857b
  - `scripts/claude_v342/test_g4_e3l4.py`：d28e60f46d53c705721cb67ab48769c3de9f2e828a46fefb5d4209feaafc0859
  - `scripts/claude_v342/chain_day8_E3L4.sh`：a051dbe5d2cae5555b62704286f3366b588733809e45cbc13a1872661eacf433（`W_STAR=FILL_W_STAR` 仍未填写，远端同样未填）
- 冻结依赖（只读，sha 与链里钉的值相同，远端也相同）：g4_e3l3.py c7523cc8…、g4_e3l2.py bab73d64…、g4_e3l.py 3dc970d8…、g4_e3r.py 1d65d9a8…、g4_noise_panels.py 2313d75b…，以及 mapping_3d_storey_data_v420 / houseexpo v365 / v371。
- 审查者自己的脚本与输出都在 `claude_handoff/analysis_20261027/g4/review_e3l4/`。没有修改任何被审查的文件。uk-linux 上只做了只读的 `sha256sum / ls / grep / nvidia-smi --query`，没有写文件，也没有在 GPU 上跑任何东西。没有碰 bigboy，没有做 git。

## 结论：**LAUNCH OK**（前提：协调者按 D8-67 第 0 步填入 W_STAR）

- 代码里没有发现阻断问题，也没有必改项。D8-68 的每一条都能在代码里找到对应。判据数值与 D8-63 逐项相同，只把噪声种子换成 3、4，且写死在代码里。单变量的实现是干净的：只有 ω 被冻结，其余部分逐位走 E3-L3 的冻结代码（§1）。
- **关键问题（有限 w 是否会让远处的已知信息在取整后消失）的答案是"不会"**，有推导，也有数值验证（§2）。
  - 只要 w > 1，Λ 消息取整后就 ≥ 1：wΛ̃/(w+Λ̃) ≥ w/(w+1) > ½。
  - 对网格 {3, 4, 6, 8}，下界分别是 0.75 / 0.80 / 0.86 / 0.89，离取整边界至少 0.25，float32 误差不会让它翻转。
  - 在长 400 的链、200 叶的星形树、随机深树上实测，不一致体素都是 0。w = 1 时有 35,025 个不一致，断言会把它判成 INCOMPLETE。
  - 用真实的 E3-L3 编码器在一个 TRAIN 房上走冻结的 eval_arm：w = 3、8 时不一致为 0，GaBP 断言全过，字节降到学到的 w 的 0.36–0.63 倍；w = 1 时不一致 1,187,160。
  - 本机单测 9/9。审查者另跑了一次冒烟（w* = 3，两个种子，只用 TRAIN 房），全链路通过。
- **发射前须知**。这两条都不是代码问题，但协调者应在发射前知道：
  1. **"第二个差异"很可能会触发**（§3.1）。要在 3,000 s 内跑完 9,562 次更新，5090 每步须 ≤ 约 0.266 s，也就是要比 bigboy 6000 Ada 实测的 0.548 s 快 2.05 倍以上。按显存带宽之比（1.79 / 0.96 TB/s ≈ 1.9 倍）估计，更可能落在 0.29–0.34 s/步，即约 7,500–8,800 次更新。预注册允许这种情况，代码处理也正确，但这样 E3-L4 的更新数会比 E3-L3 实际跑的 9,411 / 9,359 少 6–20 %。这个差异只影响和 E3-L3 的对比（只报告），不影响相对 NET 的判决。
  2. **取整给远处证据设了"精度下限 1"**（§2.3）。所以有限 w* 在线路上的效果是"把大计数饱和到 ≤ w"，而不是"远处证据按距离衰减"。这不是 bug：G4-E7 用的是同一个量化器，w* 的选择已经包含了这个效应。但 D8-66 第 4 条的读法要相应修正。

### 发射前例行步骤（协调者）

1. G4-E7 第 0 步出结果后，把 `W_STAR` 填成 3 / 4 / 6 / 8 之一；链会拒绝 FILL_ 和网格外的值。
   - 如果第 0 步判"没有 w 满足家具条件，本批不跑"，D8-68 的 w* 就没有定义，**E3-L4 也不能发射**（不要自行另选）。
   - 如果第 0 步判"没有 w 满足字节条件，取满足家具条件、字节最小的 w"，这个 w 仍然是合法的 w*。
2. 填完后，链的 sha 会变（链不钉自身，只记录在日志和 lease.json 里）。这时 `test_chain_script_guards` 里的 `fills == ['W_STAR']` 会失败，这是预期的，不要当作异常。
3. 建议（S1）在 lease.json 里写入 G4-E7 第 0 步判据文件的路径和 sha，作为 w* 的出处。

---

## 1. 单变量：只有 ω 冻结在 log w*

| 项 | E3-L3（c7523cc8） | E3-L4 | 证据 |
|---|---|---|---|
| 模型类 | `E3L3(width, W0)`：enc、dec = R.UNet3(3, 16, skip=False)，`log_w = Parameter(log w0)` | `E3L4(E3L3)` 子类，只多一行 `self.log_w.requires_grad_(False)` | g4_e3l4.py:109-116 |
| 构造与随机数 | `torch.manual_seed(seed)` 后依次构造 enc、dec、log_w | 相同。`torch.tensor(math.log(w))` 不消耗随机数 | 单测 `test_omega_frozen_and_gradient_zero`：同种子下 enc、dec 的 state_dict 逐位 `torch.equal`（L105-110） |
| w 的初值 | `W0 = L.W0 = 1000`，可学 | `configure()` 把 `L3.W0` 设为 w*，`L3.train` 里的 `E3L3(args.width, W0)` 和 `recipe['w0']` 都随之变化 | g4_e3l3.py:311、318；g4_e3l4.py:126-131 |
| 优化器 | AdamW，两组：ndim>1 带 wd，其余不带；log_w 在后一组 | 分组不变。log_w 的 grad 恒为 None，AdamW 跳过它（`if p.grad is None: continue`），clip_grad_norm_ 也忽略它 | 单测：3 步后 log_w 逐位不变、编码器在变（L99-103） |
| 编码器、三级字母表、Λ = 足迹、已知 = Λ_b > 0 | `encode`、`source`、`dec_input` | 继承，未改 | — |
| 逐跳取整 | `L.gabp(..., qmode)`：训练 'ste'，评估 'round'，步长 1 | 同一函数；w 换成冻结的张量 | g4_e3l3.py:196、246、444、528 |
| 数据与日程 | cache-e3r-v1（manifest 14e90b40）、证据表 A/B、子树采样、BCE、lr 1e-3、wd 1e-4、预热 300、裁剪 1.0、batch 8、探针每 1,000 次、第 6,000 次快照 | 训练器参数逐项相同：两条链的训练命令只差 `--expect-gpu`、deadline、`--second-note` | chain_bigboy_E3L3.sh:200 对照 chain_day8_E3L4.sh:159；MAN/FIT/CTRLH/TH/CK 的 sha 两链相同 |
| 阈值规则 | L3.select（空闲召回 ≥ 对照 − 0.25 pp，家具 F1 最大，同分取最小） | 直接调用 L3.main select，只在前面加了检查 | g4_e3l4.py:235-241 |
| 评估 | L3.eval_job / eval_arm | 同一函数；工作进程先 `configure`，于是臂名为 E3L4_<seed>，模型类为 E3L4 | g4_e3l4.py:245-248 |

- **w 是精确的**：CPU float32 下，`exp(float32(log w))` 对 3 / 4 / 6 / 8 恰好等于 w（单测 L55-57）。训练后、select 前、eval 前都断言检查点的 `exp(log_w) == w*`。评估时 GaBP 用的 w 也就恰好是 w*。
- **由单变量带来的必然差别**（不是第二个变量）：
  1. ω 的梯度不再计入 clip_grad_norm_ 的全局范数；
  2. w 冻结后 Λ 那一路不再需要梯度，autograd 少存一些张量，显存反而比 E3-L3 略省。
- **单变量之外，预注册本身带来的差异**（都已写在 D8-68 里，只影响与 E3-L3 的对比）：
  - 硬件从 6000 Ada 换成 5090；
  - torch 版本可能不同，run.json 里会记录；
  - 更新数可能少于 9,562，见 §3.1。

### configure() 的猴子补丁会不会泄漏

- 补丁只是对本进程内 `g4_e3l3` 模块的四个属性赋值（`E3L3`、`W0`、`ARM`，以及 verdict 时的 `D8_63`），不写任何文件。
- **其他进程不受影响**：
  - 对照 NET 由 `g4_e3r.py eval` 在独立进程里计算；
  - 进程池是 spawn（R._pool，g4_e3r.py:1236），子进程会重新 import，拿到未打补丁的 g4_e3l3。
- **eval 的工作进程**会自己调用 `configure`（g4_e3l4.py:247）。
- **select 的工作进程（L3._sel_task）没有打补丁**：它用原类 `E3L3(width, r['w0'] = w*)` 构造模型，然后 `load_state_dict` 读入冻结的 log_w。推理时 w 仍然恰好是 w*，结果与打补丁的版本相同（只是 log_w 的 requires_grad 为 True，在 no_grad 下没有影响）。
- **判决与 report12** 用 `_as_e3l4` 上下文，调用后复原；单测断言复原后 `L3.E3L3 is 原类`、`ARM == 'E3L3'`、`W0 == 1000`。
- train / select / eval 三个 CLI 路径里，`configure` 会一直生效到进程结束，这是有意的设计（每一步链调用都是单独进程）。
- **只是元数据的瑕疵**（S3）：L3.train 写的 run.json 仍然是 `batch_name='E3-L3'`、`prereg='D8-63'`，日志也打印 "E3-L3 train seed …"。判决只从 run.json 读 seed、recipe.evidence 和 pair，所以不影响结果。E3-L4 的身份由 `e3l4.json` 和 `final.json['e3l4']` 记录。

## 2. 有限 w 下的 GaBP

### 2.1 推导：已知区 = k > 0 在任意树、任意距离上都保持

- **记号**：消息 Λ_{j→i} = round(w·Λ̃ / (w+Λ̃))，其中 Λ̃ = Λ_j + Σ_{k≠i} Λ_{k→j}（L.gabp，g4_e3l.py:188-194；quantize 是 `torch.round`，取整规则是半数取偶）。
- **归纳**（从叶子向内）：
  - 源端 Λ_j ∈ {0, 1}，入消息都是整数，所以 Λ̃ 是非负整数。
  - Λ̃ = 0 ⇔ j 一侧的子树里没有任何机器观测到 v。此时消息为 0。
  - Λ̃ ≥ 1 时，因为 x ↦ wx/(w+x) 单调递增，消息 ≥ round(w/(w+1))。w > 1 时 w/(w+1) > ½，所以消息 ≥ 1。
  - 因此 Λ_{j→i} > 0 ⇔ 那一侧子树的 k > 0，于是 Λ_b(i) > 0 ⇔ 整个分量的 k > 0。这与链长、分支数无关。
- **边界**：w = 1 时 w/(w+1) = ½，半数取偶得 0，已知区丢失。这就是 check_w_star 拒绝 w ≤ 1 的原因（g4_e3l4.py:121）。
- **网格上的余量**：w ∈ {3, 4, 6, 8} 时，下界 w/(w+1) = 0.75 / 0.80 / 0.857 / 0.889，离 ½ 至少 0.25。'ste' 模式的前向值与 'round' 相同；GPU 上 w 即使差 1 ulp，也翻不过这个边界。
- **h 那一路**：用同一个 f 缩放，取整又是单调的，所以由 |h̃| ≤ Λ̃ 可归纳出 |h_q| ≤ Λ_q，信念均值 |x| ≤ 1。h 取整为 0 只会丢掉均值，不会丢掉已知区。

### 2.2 数值验证（`review_e3l4/gabp_finite_w_probe.py`，日志 `.log`，结果 `.json`）

- **用例**：每个 w 跑 52 个，都调用冻结的 L.gabp / L.gbp_checks：
  - 链长 2–400，只有一端观测；
  - 星形树 3–200 叶，只有一叶观测，或除中心外都观测；
  - 40 棵随机深树（2–120 个节点），足迹稀疏度 1 % / 5 % / 30 %。
- **三种口径**：'round' float64、'round' float32、'ste' float32。

| w | 已知区不一致体素（三种口径） | 消息 Λ 最大值 | 闭式差最大 | 能量违例 |
|---|---|---|---|---|
| 1.0 | **35,025 / 35,025 / 35,025** | 0 | 1.4e-13 | 0 |
| 1.0001、1.5、2 | 0 | 1 / 1 / 2 | ≤ 2.8e-13 | 0 |
| 3 / 4 / 6 / 8 | **0** | 3 / 4 / 6 / 8（≤ w） | ≤ 6.6e-13 | 0 |

- 取整后的 Λ 消息表（整数 Λ̃ = 0…20），w = 3：0, 1, 1, 2, 2, …, 2, 3（从 Λ̃ = 16 起为 3）；w = 8：0, 1, 2, 2, 3, …, 6。Λ̃ ≥ 1 时从不为 0。
- **真实编码器**（`review_e3l4/e3l3_ckpt_at_finite_w.py`）：
  - 做法：用冻结的 E3-L3 检查点（三级码，非平凡）走冻结的 `g4_e3l3.eval_arm`，在一个 TRAIN 冒烟房（`train_r1_i0_s203`，8 台机器构成一个分量，cam_nf）上，只把 w 换掉。
  - 阈值沿用 E3-L3 在学到的 w 下选的，没有重选。只用 1 个 TRAIN 房，结果只供参考。

| 检查点 | w | 已知区不一致 | 闭式差 | 违例 | 全网 O7 字节（相对学到的 w） | 量化与浮点信念差：均值 / 最大 | 机间信念差 rms | 家具 F1 |
|---|---|---|---|---|---|---|---|---|
| 342101 | 1240.37（学到的） | 0 | 6.1e-13 | 0 | 349,109（1） | .000 / .005 | 0 | .9918 |
| 342101 | 8 | 0 | 5.0e-15 | 0 | 219,938（.63） | .012 / .456 | .045 | .9922 |
| 342101 | 3 | 0 | 1.6e-15 | 0 | 132,261（.38） | .029 / .956 | .106 | .9903 |
| 342101 | 1 | **1,187,160** | 1.7e-15 | 0 | 493 | .033 / 1.23 | .295 | .2602 |
| 342202 | 1357.72（学到的） | 0 | 2.2e-13 | 0 | 260,228（1） | .000 / .004 | 0 | .9920 |
| 342202 | 8 | 0 | 5.0e-15 | 0 | 155,110（.60） | .005 / .456 | .027 | .9921 |
| 342202 | 3 | 0 | 1.6e-15 | 0 | 94,060（.36） | .012 / .956 | .067 | .9896 |
| 342202 | 1 | **1,187,160** | 1.7e-15 | 0 | 493 | .012 / 1.24 | .171 | .2857 |

  - 在真实码上，w* ∈ {3, 8} 时已知区逐位保持，GaBP 断言全过。字节降到学到的 w 的 0.36–0.63 倍，与 D8-66 的方向一致。
  - w = 1 时已知区几乎全部丢失，`known_mismatch` 这个断言能抓到。

### 2.3 断言会怎样反应，以及一个读法上的修正

- **如果某体素的已知区丢了**（例如误把 w 设成 1）：
  - **eval**：逐分量计数 `known_mismatch`（g4_e3l3.py:547），aggregate 求和，verdict 判 INCOMPLETE（L772-773）；FPR 不再逐位相同，也会触发 INCOMPLETE（L702-703）。
  - **select**：`assert mism == 0`（L494）会让 select 失败，于是没有 threshold.json，链判 "no trained checkpoint with thresholds" 或 verdict 判 "no threshold.json"，结果都是 INCOMPLETE。
  - **训练探针**：只记录，不中止。
  - 此外，在 w ∉ {3, 4, 6, 8} 时，`check_w_star` 在任何训练之前就拒绝。
- **读法修正（只是读法）**：取整给每条非零消息设了精度下限 1。
  - 链长 400、只有一端观测时，浮点 GaBP 在远端的 Λ_b 是 0.0075（按 3/(3+d) 衰减），而**取整后是 1**，均值 +1 被原样传了 399 跳。
  - 因此在线路上，有限 w* 只是把大计数饱和到 ≤ w，从而压缩字母表、省字节。稀疏的远处证据仍然以权重 1 参与，与本机自己的观测同权。D8-66 第 4 条说"远处证据按树上距离衰减"，那是浮点 GaBP 的性质。
  - 量化信念与浮点信念的差可以达到约 1.0（合成用例，`quant_mean_absdiff_max`）。解码器训练时看到的就是量化后的信念（'ste' 前向取整），所以训练和评估是一致的。这个差值按 D8-63 只报告。

### 2.4 闭式一致、能量单调

- `L.gbp_checks` 对一般的 w 有效：
  - 闭式解是 (diag Λ + wL)⁻¹ diag Λ z，只解 ΣΛ > 1e-6·max(1, w) 的格；
  - `descent_energies` 也对一般的 w 成立。
- 单测在 w ∈ {3, 4, 6, 8} × 25 棵随机树上，cf ≤ 1e-9、违例 0。审查者的 52 个用例 × 8 个 w 上，cf ≤ 6.6e-13、违例 0。
- 判决阈值 CF_TOL = 1e-5 不变；GaBP 断言失败判 INCOMPLETE（比 D8-68 写的"挡杠杆"更严，与 D8-63 / E3-L3 的实现相同）。

## 3. 更新数、显存、预跑口径

### 3.1 9,562 次与"第二个差异"

- **代码**：
  - 正式训练要求 `--updates ≤ 9562`；小于 9,562 时，必须给出 `--second-note`，否则拒绝（g4_e3l4.py:189-194）。冒烟实测：9,563 被拒；不给 note 也被拒。
  - 链在预跑后按 E3-L3 的规则计算（chain_day8_E3L4.sh:136-144）：
    - base = 3000 − 60 − 加载 − 首窗 − 30；
    - u0 = min(⌊0.9·base / r⌋, 9562)；
    - avail = base − 探针 × (u0 // 1000 + 1)；
    - fit = ⌊0.9·avail / r⌋；
    - UPD = min(fit, 9562)。
  - UPD < 9,562 时，把 note 同时写进日志、updates.json、e3l4.json 和 final.json；两个种子用同一个 UPD；少于 2,000 次判 INCOMPLETE。
  - 判决里，"未截断"指 updates_done == 计划数（L3 的 complete）；计划数 ≠ 9,562 又没有写明 note 时，判 INCOMPLETE（g4_e3l4.py:442-443）。这与 D8-68 一致。
- **复算**：
  - 用 E3-L3 在 bigboy 上的实测值代入：加载 17.0 s，首窗 31–32 s，探针 4.0–6.5 s，每步 0.5206 / 0.5478 s。D8-63 的 6,000 s 正好得到 9,562：base = 5861，avail = 5861 − 4.05 × 10 = 5820.5，⌊0.9 × 5820.5 / 0.5478⌋ = 9562。
  - 5090 上 3,000 s 要放下 9,562 次，需要 r ≤ 0.9 × (2910 − 17 − W − 10Q) / 9562。取 W = 20–32 s、Q = 4–6.5 s，得 **r ≤ 0.263–0.267 s/步**，相当于比 bigboy（取两卡较慢者）快 2.05–2.08 倍。
  - D8-63 写的是"bigboy 约比 5090 慢一倍"，正好落在临界点上。按带宽之比估计（这里是全分辨率 3D 卷积，宽 16，主要受带宽限制），5090 约 0.29–0.34 s/步，对应 UPD 约 7,500–8,800。
  - **P(第二个差异) 我估计约 0.7**。代码对它的处理正确。只是 E3-L4 会比 E3-L3 少练 6–20 %。D8-66 说 E3-L3 的留出探针最后 3,000 步还在涨，所以这个差异对 E3-L4 是不利的偏差：对"杠杆"判决来说偏保守，对"与 E3-L3 比"则会低估 E3-L4。
- **租约时间线**：
  - 预跑 ≤ 600 s（timeout），预计 150–250 s；每个种子 timeout = SDT − NOW ≤ 3,000 s；SDT ≤ START + 7150，再加 `-k 30`，硬上限是 START + 7180 < 7200。
  - 种子之间的"GPU 忙则等"最多等到 DE − 900。
  - 种子 2 剩余不足 900 s 时不跑，判决为 INCOMPLETE（缺种子）。

### 3.2 5090 显存估计

- uk-linux 上 RTX 5090 共 32,607 MiB，03:52 UTC 空闲（已用 344 MiB，没有计算进程）。
- TRAIN 缓存 205 房都是 8 台机器（远端 manifest 的 `"agents": 8` 计数为 205）。所以每批最多编码 8 × 8 = 64 台。
- 作者的 mem_probe 结果：
  - 编码器每台约 0.264 GiB，解码器每样本 0.262 GiB；
  - 64 台时，存给反向的激活约 19.0 GiB。
- 审查者补算：
  - GaBP 消息每条 2 MiB（128 × 128 × 32 × float32）。w 冻结后 Λ 那一路不存梯度，h 那一路每条消息约存 2–3 个张量，一批最多约 0.7 GiB；
  - 前向的临时张量一块约 2 GiB；
  - 再加上 cudnn.benchmark 的工作区和分配器碎片，峰值估计在 **22–26 GiB**，低于 31.8 GiB。
- 64 台出现的频率：Σ|s| 是 8 个 U{1..8} 之和（这是上界，按所有分量都是 8 台算），精确卷积得 P(Σ ≥ 57，补齐到 64) = 3.8e-4 每批，9,562 批里约 3.7 次。所以最坏情况会出现，但有约 6–10 GiB 的余量。
- 400 步预跑碰到 64 台批的概率只有 0.14，但几乎必然碰到 ≥ 49 台（补齐到 56，约 16.9 GiB）的批。
- 风险不大，但**没有在硬件上测过**：E3-L3 在 48 GB 卡上没有记录峰值。一旦 OOM，那个种子就没有 final.pt，判 INCOMPLETE，而租约只有一个。所以建议 S2：记下预跑峰值，就能按 64/56 外推最坏情况。

### 3.3 预跑口径

- 只用种子 342101 在同一块卡上预跑 400 步，`--log-every 50 --time-from 200`，探针在第 400 步测一次耗时；rate 取第 200–400 步的窗口（变异系数 ≥ 10 % 时取中位数）。E3-L3 是两卡取较慢者，这里两个种子在同一块卡上顺序跑，用同一个 rate 是合理的。
- 预跑如果没跑满 400 步，或 rate 读不出来，UPD 就是 0，链判 INCOMPLETE 并退出（只浪费约 10 分钟租约）。

## 4. 评估与判决

- **判决面板**：G4-E5 导出，VAL cam_nf × 噪声种子 3、4 × 训练种子 342101 / 342202，共 4 格；fresh 种子 3、4 只降级；种子 1、2 由 `report12` 只报告，从不写判决。
- **导出格式核对**（只看元数据，没有计分）：
  - 本机 `g4/e5_export/records` 共 96 条：val / fresh × 种子 3 / 4，每组 24 条；split 全是 VALIDATION；失败 0、跳过 0；e1_match 三档全是 None；脚本 a63ce7d2。
  - npz 的键（L_k0_{clean,cam_nf,cam}、L_pub_cam、occ、slab、obj、cell、private、edges）与 E3-R 导出相同。
  - 所以 `L3.eval_job` 和 `R.eval_job` 都能直接读。每个 npz 的 sha 由 record 钉住，eval 里断言 `file_sha256 == rec['file_sha256']`。
- **NET 对照在同一批地图上**：`g4_e3r.py eval` 用冻结的 E3-R 检查点和 threshold（链钉 CK1/CK2/TH1/TH2），对同一个 EXPORT34 计算（字节开，`--bytes 1` 是默认）；再用 `ctrlagg` = R.aggregate 汇总。
  - 判决时 `control_problems` 断言：对照没有失败或跳过的任务；g4_e3r 的 sha 与冻结版相同；对照检查点包含两个训练种子 run.json 里配对的 E3-R 检查点；val 和 fresh 两个面板的噪声种子都恰好是 3、4。
  - `L3._cell` 另外断言臂与对照的房间集合相同。
  - 只报告：与 G4-E5 Q∞ 行做同图一致性比对，并列出 QPOST4。
- **判据写死**：D8_68() = 冻结的 D8_63 加上 noise_seeds = 3、4，budget 3,000，updates 9,562。单测逐项断言与 D8_63 的数值相同。
  - 杠杆：Δ ≥ −1 pp、字节 ≤ 0.15 倍、空闲召回 / 表面 ≥ −0.5 pp、FPR 逐位相同、clean ≥ .99、K0_sep ≥ −3 pp、未截断。
  - 降级：cam 4 格均值 < −3 pp、fresh 4 格均值 < −1 pp。
  - 不够：任一格 < −3 pp 或字节 > 0.5 倍。
  - 这些都与 D8-68 / D8-63 相同；字节比的口径与 E3-L3 相同：臂每房 O7 字节的中位数除以 NET 的 rnet_relay_median 的 sum 与 count 之和。
  - `test_verdict_d8_68` 在边界两侧各测了一次（d = −.0099 判 LEVER、−.0105 判 AMBIGUOUS；ratio .15 判 LEVER、.151 判 AMBIGUOUS；等等）。
- **INCOMPLETE 条件**，以下任一出现即判：
  - 继承 L3 的条件：失败或缺失的任务、E1 不符（种子 3、4 只接受三档全为 None，Q7）、缺 run / final / threshold、少于 2,000 次、两个种子的更新数不同、缺格、缺房、缺对照、缺档、版本混用、GaBP 断言失败、已知区不一致、FPR 不同；
  - 本包装新增的条件：缺 e3l4.json，或其中 prereg / w* / omega_frozen 不符；final.json 里没有 e3l4 块，或最终 w ≠ w*；格内 w ≠ w*；计划数 ≠ 9,562 又没有写明 note；对照不合格。
  - 链层面：sha 不符就中止（不写 summary，等于缺任务）；预跑后算出的 UPD < 2,000 时写 `$OUT/summary.json`。
  - 注意：判决文件是 `$OUT/eval/summary.json`，链层面的 INCOMPLETE 写在 `$OUT/summary.json`。读结果时两处都要看（S4）。
- **E3-L3 在种子 3、4 上的同格读数**：本批不计算，因为它属于 G4-E7，两批互不读读数。种子 1、2 的 E3-L3 读数在 report12 里并列。

## 5. 链的安全性与冒烟

### 5.1 链（对照 chain_day7_E3L2.sh，它已在 5090 上成功跑过）

- **与 E3-L2 链一致的部分**：
  - 钉 sha，在开跑前、租约前、每个种子前、租约后各检查一次；FILL_ 检查；网格检查；
  - 锁 `$OUT.lock`，配 trap，并且在检查 `[ -e $OUT ]` 之前就加锁；不覆盖；
  - 磁盘 ≥ 5 GB（现有 1.5 TB）；导入自检；
  - SOLO 等待（项目里的其他链、trainer、diag，或任何 GPU 计算进程；隔 60 s 查两次）；
  - 只用 `timeout -k 30` 结束自己的子进程；没有 pkill、killall、kill、rm；
  - 只写 `/mnt/datasets`（OUT、LOG、TMPDIR），导出只读。
- **租约外的 CPU 阶段**：都设了 `CUDA_VISIBLE_DEVICES=`，nice 10，9 个 worker 加 1 个空闲的父进程，共 10 个进程；每个阶段都有 timeout。比 E3-L2 链（10 个 worker，没有 nice）更保守。
- **新增的 e3l4-ref 钉住是完整的**：
  - 整个 records 目录内容的 sha（E5REC_SHA），再由 record 钉住每个 npz 的 sha；
  - export_run_VALIDATION.json（E5RUN_SHA）、e5_summary.json（E5_SUM_SHA）、e3l3_summary.json（E3L3_SUM_SHA）；
  - 03:52 UTC 远端只读核对，全部相同：b33297e3、e1e964c5、fbf67c87、7554d1fd…（96 条 records、96 个 npz），以及 ctrl_holdout f0b55ed9、fit 8dd8b195、manifest 14e90b40、E3-R summary 8a0bd5dd、TH c2dc9883 / 775a0d48、CK 1b035fd1 / a58b2b05、index 7b4730cc、pyroot de96be65、EXPORT12 records 62527a54。
  - `e3l4-ref/e3l3_ckpt/` 只被单测使用，链不读它，不钉也可以。
- **小问题**（S，不阻断）：
  - 训练期间不像 bigboy 链那样记录 GPU 上的其他进程（S5）。
  - `~/.nv/ComputeCache` 可能被 CUDA 的 JIT 写入，这不在 /mnt/datasets 下；E3-L2 链也一样，算不上新问题（S6）。

### 5.2 单测（本机 CPU）

- `CUDA_VISIBLE_DEVICES= python -B test_g4_e3l4.py`：**9/9 通过**，03:42:40–03:43:00 UTC，日志 `review_e3l4/unittest_review.log`。
- `test_forward_equals_e3l3_at_learned_w` 用的是本机真实的 E3-L3 检查点（没有退回到随机权重）。
- 作者在 uk-linux 上的单测（`runs/e3l4_test.out`）也是 9/9。

### 5.3 审查者冒烟（本机 CPU，只用 TRAIN 房）

- **脚本**：`review_e3l4/run_review_smoke.sh`，日志 `run_review_smoke.log`，输出 `smoke/`，03:43:46–04:13:10 UTC。
  - 本机 CPU 同时还在跑别的批次，所以很慢。我的进程任何时候都不超过 4 个。
- **与作者的冒烟不同的地方**：
  - w* = 3，取网格的最小值，取整效应最强；
  - **两个训练种子**（证据表 A / B，各自配对 E3-R 检查点）；
  - 两个检查点在同一次 eval / aggregate 里一起跑；
  - 数据只用 TRAIN 房：E3-L 冒烟缓存、E3-L2 冒烟 ctrl_holdout、TRAIN 导出 `train_r1_i0_s203`。
- **结果**：

| 步骤 | 结果 |
|---|---|
| train --w-star 5 | 拒绝："w* = 5.0 is not in the G4-E7 step-0 grid" |
| train --updates 9563 | 拒绝："--updates 9563 > the pre-registered 9562" |
| train 两种子，各 8 次（附 note） | 都完整，8 / 8；final w = 3.0 = w*；e3l4.json 里 omega_frozen = True、second_difference = True；探针的 known_mismatch = 0 |
| select 两种子（w* = 3） | 通过：`assert mism == 0` 与"FPR 计数与对照逐个相等"的断言都过了；R8 阈值可行；留出房字节比 .0596 / .0587 |
| select --w-star 4（对 w* = 3 的 run） | 拒绝：e3l4.json 的 w* 不符 |
| eval（两个检查点，3 档） | exit 0；臂名 E3L4_342101 / E3L4_342202；每档 w = 3.0；known_mismatch = 0；GaBP pass（违例 0） |
| g4_e3r eval 对照 + ctrlagg | exit 0 |
| aggregate | 判 INCOMPLETE，原因只有预期的几条：8 次 < 2,000、没有 val / fresh 面板、对照没有种子 3 / 4。second_difference 正确列出两个种子和 note；**没有**"配对检查点不在对照里"这一条，说明两个种子的配对都核对通过了 |

- **局限**：训练 8 步后，z 仍然全是 0（UNet3 的头初值约为 0），所以冒烟只检验了管线。非平凡码在有限 w 下的行为由 §2.2 的真实编码器探针覆盖。
- **一个附带读数**（只供参考）：z 全为 0 时，w* = 3 的留出房字节比是 .06，E3-L3 审查用全零码实测是 .1202（房间与口径不完全相同）。也就是说，有限 w* 把"计数通道下限"大约减半。

## 必改项

无。

## 建议项（不阻断发射）

- **S1**（w* 的出处）：填 W_STAR 时，在链里加 `WSTAR_SRC=<G4-E7 第 0 步判据文件>` 和它的 sha，并写进 lease.json，这样 w* 能追溯到 TRAIN 上的选择（D8-67 要求第 0 步"写进判据文件并记 sha"）。改链后重跑 `test_chain_script_guards`；FILL_ 断言失败是预期的。
- **S2**（显存可见性）：在包装里训练结束后，把 `torch.cuda.max_memory_allocated()` 写进 `final.json['e3l4']`。只改 g4_e3l4.py，要重钉 E3L4_SHA。这样预跑后就能看到峰值；如果按 64/56 外推的峰值 > 30 GiB，可以考虑停下。也可以在链里给训练加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来减少碎片。这两条都不是必须。
- **S3**（元数据）：run.json 和训练日志里是 'E3-L3' / 'D8-63'，因为用的是冻结的 L3.train。读结果时以 e3l4.json 为准；报告里写明即可。
- **S4**（结果位置）：链层面的 INCOMPLETE 写在 `$OUT/summary.json`，判决在 `$OUT/eval/summary.json`。协调者收结果时两处都要看。
- **S5**（GPU 共用记录）：仿照 chain_bigboy_E3L3 的监视器，训练期间每 60 s 记录 5090 上的其他计算进程（只记录，不干预），以便截断时能归因。
- **S6**：设 `CUDA_CACHE_PATH=$TMPDIR/nvcache`，保证"只写 /mnt/datasets"不留例外。
- **S7**（解读）：结果报告里要写明 §2.3 的事实：取整让每条非零消息的精度 ≥ 1，有限 w* 的作用是饱和而不是衰减。另外，若触发了第二个差异，与 E3-L3 的对比（种子 1、2 的 report12）要注明 E3-L4 的更新数比 E3-L3 少多少。

## 附：审查者文件（`claude_handoff/analysis_20261027/g4/review_e3l4/`）

- `unittest_review.log`：本机单测 9/9。
- `gabp_finite_w_probe.py` / `.log` / `.json`：有限 w 下的取整探针（链、星形树、随机树；w = 1…8）。
- `e3l3_ckpt_at_finite_w.py` / `.log` / `.json`：真实 E3-L3 编码器在 w = 学到的值、8、3、1 下走冻结的 eval_arm（只用 1 个 TRAIN 房）。
- `run_review_smoke.sh` / `.log`，`smoke/`：审查者冒烟（w* = 3，两个种子，TRAIN 房）。
