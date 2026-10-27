# G4-E5 代码审查：噪声档诚实线路（源端量化 stalk + 冻结 NET 读出），对照预注册 D8-52

- 审查者：独立审查（Claude），写于 2026-10-27 13:15 UTC（`date -u`）。本机 CPU，`CUDA_VISIBLE_DEVICES=`，最多 4 个进程；没有用 GPU，没有碰 uk-linux，没有做任何 git 操作，没有修改被审查的文件。
- 审查对象：
  - `scripts/claude_v342/g4_quant_wire.py`：sha256 88916165…ecb1，1,225 行；
  - `scripts/claude_v342/test_g4_quant_wire.py`：b5560cc7…；
  - 判据与量化器文件 `g4/e5_quant/criteria_quantizer.json`：0df6e6d8…ae86；
  - 冒烟输出 `g4/e5_smoke/`。
- 冻结依赖已核对 sha：
  - g4_e3r.py 1d65d9a8，g4_noise_panels.py 2313d75b，g4_calib_stalk.py c905d9e6；
  - E1c fit.json 8dd8b195（`g4/e1c/fit.json` 与 `review_E3R/inputs/e1c_fit.json` 相同）；
  - 检查点 342101 1b035fd1 / 342202 a58b2b05，threshold.json c2dc9883 / 775a0d48，与判据文件记录的一致。
- 背景：按 D8-54，G4-E5 现在是"零学习的字节—精度参照线"，不再决定架构的状态定义。这一点不影响下面对"代码是否忠实于 D8-52"的判断。

## 结论：**LAUNCH OK WITH CHANGES**

核心计算全部正确：
- 量化器、级值、Q∞ 对照、NET 读出、计分口径、S+k 字节、级别计数编码器、每房一种格式 + 1 bit，都经独立复算**逐位一致**（见第 8 节）；
- Q∞ 在种子 1 VAL 全部 24 房上与 E3-R summary 的 NET 家具、中位字节完全相等；
- 单测 11/11 通过；
- 级别计数场在真实的整条消息上做真实算术编码往返成功，开销 0.9 bit。

上机前有两处必须改，都在 aggregate 的判决逻辑里，不涉及任何计算路径：
- **M1**：fresh 规则要定稿；
- **M2**：判决来源的护栏。

两处都改完、重跑 quantize 和冒烟 aggregate 并记下新 sha 后，可以上机。

### 必改项

**M1. fresh 检查：采用协调者的非劣口径，只允许降级，并在上机前写成 D8-52 补充。**

- **现状**（第 1129–1133 行）：VAL 判决是 LEVER 或 NO、且所选臂的 fresh 4 格均值 Δ 与 VAL 4 格均值 Δ 符号相反时，判 AMBIGUOUS。
- **问题一：符号检验和非劣检验不相容。** 本批是非劣检验，预期 Δ 是小的负数，接近 0。在这种情况下比较符号，等于对噪声掷硬币，而且惩罚的是"更好"的结果。
  - 例一：VAL −0.3 pp、fresh +0.05 pp，fresh 更好，却会被降级。
  - 例二：VAL −0.3 pp、fresh −0.95 pp，fresh 已经贴着非劣线，却照样通过。
  - 退路臂 Q4 的风险是实际存在的。R-B 的读数里，VAL 种子 2 / 342101 的 Q4 均值 Δ 约为 0.000 pp（三位小数）；fresh 种子 1、2 的 Q4 是 −0.14 到 −0.23 pp（`review_E3R/RB/rb_summary.json`，键 `a` 与 `a_fresh`）。所以 VAL 的 4 格均值完全可能落在 0 的任一侧。
- **问题二：代码对 NO 也会改判。** NO 被改成 AMBIGUOUS 是**升级**。D8-52 的原文是"降为 ambiguous"，这一行为不在字面里。它沿用了 E3-R 的写法（g4_e3r.py 第 1226 行对 NOT_ENOUGH 也改判），但 E3-R 是优越性检验，那里讲得通，这里讲不通。
- **评价协调者的提议**（fresh 4 格均值 Δ ≥ −1.0 pp）：**赞成**。理由：
  - 它把"同向检查"的本意（换一个没看过的面板，结论能否复现）翻译成非劣检验的对应物，也就是 fresh 也满足 VAL 的非劣边际；
  - 边际与判据 (1) 相同，没有引入新的自由数；
  - 方向中性：R-B 已看过的 fresh 种子 1、2 上，Q3 是 −0.24 到 −0.29 pp，Q4 是 −0.14 到 −0.23 pp。新旧两条规则在这些数上都通过，所以这次修改不是为了翻转一个已知的结果。补充里应当如实写明这一点。
- **改法**：
  1. 上机前（任何种子 3、4 的 eval 之前）在 OVERNIGHT_REPORT 里写"D8-52 补充"，带 UTC 时间、理由和上面的披露。
  2. 规则：最终臂判 LEVER，且该臂 fresh 4 格的 Δ 家具均值 < −1.0 pp 时，降为 AMBIGUOUS。NO 与 AMBIGUOUS 都不因 fresh 改变。fresh 缺格照旧判 INCOMPLETE（第 1096–1110 行已覆盖）。
  3. summary 同时报告：字面的符号检查结果（`fresh_sign_opposite`）、fresh 的逐格 Δ 和最差单房。
  4. `CRITERIA` 里把 `fresh_sign_check` 换成 `fresh_rule='noninf_mean_downgrade_only'` 和 `fresh_mean_min=-0.010`，重跑 `quantize`，在补充里记下新的判据 sha 与脚本 sha。量化表是确定性的，不会变；我的复算与现文件逐项相同。

**M2. 判决来源的护栏：不能静默回落，也不能在续接种子上出判决。**

- (a) 第 1057–1062 行：如果任务文件里的 criteria sha 与 `e5_quant/criteria_quantizer.json` 当前的 sha 不同，`crit` 会是 None，然后**静默**改用代码里的 `CRITERIA` 常量。重跑 quantize 会改变 `written_utc`，sha 跟着变，这种情况很容易发生。
  - 改法：sha 不同就加一条 reason（INCOMPLETE）；或者在判据里钉死预期的 sha 前缀。
- (b) aggregate 的 `--seeds` 可以随便传，第 1095–1116 行会照常在种子 1、2 上给出 LEVER / NO。D8-52 规定种子 1、2 只作连续性。
  - 改法：非冒烟时断言 `seeds == C_['noise_seeds']`，也就是 [3, 4]，否则拒绝运行。

### 建议项

- **S1. eval 的输入护栏。**
  - 非冒烟的 eval 应拒绝 `smoke: true` 的 qpost 文件。现在冒烟阈值文件 `qpost_threshold_smoke.json`（2 房 × 1 种子）可以被误传进正式运行（第 768 行没有检查）。也应拒绝 `n_houses != 19` 或文件数 ≠ 57 的阈值文件。
  - eval 开始时对照 `criteria['checkpoints']` 核对 final.pt 与 threshold.json 的 sha。目前只有 quantize 时核对过（第 280–285 行），eval 加载检查点时不核对（第 777–781 行）。
  - aggregate 发现 qpost sha 混用时写进 summary。
- **S2. 对照字节里已知掩码被编了两次，建议加一行只报告的敏感性读数。**
  - Q∞ 的 S+k 照 E3-R 口径（这一点是对的），已知掩码在 S 的三态遍和 k 的三态遍里各编一次。
  - 我在 2 个冒烟房上实测，第二次编码占 Q∞ S+k 的 4.0 %（28,715 B / 716,402 B）和 3.5 %（50,221 B / 1,448,622 B）。扣掉以后，Q3 的 B 比值从 0.362 / 0.349 变成 0.377 / 0.361（`review_e5/mask_sensitivity.txt`）。
  - 这不影响 0.5 这条线，但报告"约 0.35 倍"时应附上"掩码只编一次时约 0.37 倍"。
- **S3. 两臂都不是 LEVER 时如何选臂。**
  - 第 1124–1127 行取 Q3 / Q4 中较好的一个，例如 Q3 = NO、Q4 = AMBIGUOUS 时报 Q4 的 AMBIGUOUS。D8-52 只写了"Q3 不过而 Q4 过时取 Q4"。
  - 建议：两臂都不是 LEVER 时，批次判决取主臂 Q3 的判决，Q4 的判决并列报告。
  - 第 1045 行把"保护不过"和"字节 > 0.75"也映射成 NO，这是合理的操作化（后果同样是保留 Q∞）。两条都请写进 M1 的补充。
- **S4. 文字报告里标注只报告项。** `_fmt_cell`（第 1141–1151 行）打印 B_RO 时没有标"违反 C5，只报告"，QPOST4 那一行也没有标"源端先验，只报告"。summary.json 的 notes 里已有这两句（第 1135–1137 行），文字报告里也应写上。
- **S5. aggregate 的冒烟开关。** `aggregate --smoke` 会关掉 INCOMPLETE 和缺房检查（第 1102、1117 行）。在非冒烟的任务文件上应拒绝运行，或者在判决前加 "SMOKE" 前缀。
- **S6. QPOST4 缺失时提示。** 正式运行里任何格缺 QPOST4 时，打印警告并写进 summary。它是只报告项，不判 INCOMPLETE，但 D8-52 列了它。
- **S7. 避免 TypeError。** `_arm_verdict`（第 1033 行）里，`d_worst_house` 为 None 时（所有房的家具 F1 都是 None 的极端情形）会抛 TypeError。改成写一条 reason。
- **S8. 冒烟导出文件的去向。** 冒烟导出的 `e5_smoke/export_check/val_r1_i0_s3.npz` 是种子 3 的地图，只跑过 check-export，没有计分，并且已被 `.gitignore` 的 `claude_handoff/analysis_20261027/g4/**/*.npz` 忽略。正式导出写到别的目录（`g4/e5_export`），不会复用它。只需确认它不随任何拷贝流出。

## 逐条检查与证据

### 1. 是否忠实于 D8-52

- **单变量**：在 eval 的同一个循环里（第 791–811 行），四个臂只有 `tab` 不同。森林、`E.relay_sums`、`E.net_logits`、阈值 j（第 780–781 行从 threshold.json 读，`int(thr['j'])`，**没有重选**）、`E.net_readout`、R8 目标（第 776 行，带 mult，与 E3-R eval 相同）全部共用。
- **Q∞ 与 E3-R 逐位一致**：
  - 单房：冒烟 val_r1_i0_s1 / 342101 的 Q∞ NET 家具是 0.9708870875670739。这与 R-A 的逐房复算（`review_E3R/RA/out_recompute/val_r1_i0_s1.json`）一致，也与我的独立复算一致。
  - 全面板：我的独立代码在种子 1 VAL 全部 24 房上算出 NET_342101 = 0.972284969312721、NET_342202 = 0.972930500615567，**与 e3r/summary.json 完全相等**（差 0.0）。空闲召回也逐位相等。
  - 字节：每房中位 S 字节 664,784.595 / 673,892.168，k 字节 87,554.847，与 summary 的 `rnet_relay_median` 完全相等（`review_e5/compare_panel.txt`）。
- **计分口径**：
  - `_cell_stats`（第 979–988 行）取分量展开行的均值。冒烟 2 房的 Q∞ 是 (0.97089 + 0.97641) / 2 = 0.97365，与 aggregate 打印的 .9736 相符。我的面板复算也按这个口径做，与 E3-R headline 相等。
  - 最差单房：取该房展开行的均值后求差，再取最小值（第 993–994 行）。
- **判据边际**（第 142–146 行，以及判据文件）：
  - 家具 −1.0 pp、最差单房 −5.0 pp；
  - 空闲召回 −0.3 pp、表面 F1 −0.5 pp；
  - clean ≥ .99；
  - 字节 0.5 / 0.75；
  - 每面板 24 房。
  - 以上都与 D8-52 一致。`_arm_verdict`（第 1030–1046 行）要求 4 格全满足。
- **INCOMPLETE**：以下情形都会判 INCOMPLETE：失败任务（第 1074–1075 行）、缺种子（第 1100–1101 行）、缺房（第 1102–1106 行，VAL 与 fresh 都查）、缺档（第 1109–1110 行，clean 也必须有）、FPR 有差异（第 1113–1114 行）、脚本 / 判据 sha 混用（第 1055–1056 行）。漏洞见 M2。
- **种子**：判决在 3、4 上（默认 `--seeds 3,4`），续接种子 1、2 只在打印里标"continuity"（第 1168 行）。CLI 没有锁死，见 M2(b)。
- **fresh**：见 M1。

### 2. 量化器

- **DP 最优性**：`opt_partition`（第 194–219 行）在 c 序上对恰好 q 个非空连续段最小化 Σ N_b h(p_b)。
  - 单测有两项：与穷举所有连续切分逐个比较（40 例）；在二元标签下与所有集合划分的全局最优比较（12 例），也就是说 c 序上的连续切分本身就是互信息最优的。两项都通过。
  - 我用独立写的 DP 得到相同的表和相同的代价：A 1,003,539.302 nats，B 1,105,710.237 nats。
- **级值公式**：`level_values`（第 226–231 行）= `C.quantise(C.fit_table(合并计数))`，归一化项是 N + 0.5 × 111，与 E1c 的逐符号公式相同。单元素的段能逐位复现 E1c 的 cq（有单测）。我独立按公式算出：
  - 半集 A：Q3 = [−97, 5, 95]，各段符号数 [31, 56, 24]；
  - 半集 B：Q3 = [−93, 6, 93]，各段符号数 [29, 58, 24]；
  - Q4 的 A 是 [−97, 0, 50, 101]，B 是 [−112, −20, 24, 96]，取自判据文件。
  - Q3 与判据文件逐项相同，也与 R-B 探针一致（`rb_probe_match` 为 true）。
  - R-B 探针的归一化写法不同（用 log N_occ / N_free），但两者只差一个常数，排序相同，级值也相同。
- **半集与检查点的配对**：342101 配 A，342202 配 B（第 122–123 行）。quantize 时断言检查点 recipe 里的证据表等于该半集的 Q∞ 表（第 287–290 行）。我用半集 A 的表跑 342101，结果与 E3-R 逐位相同，旁证配对正确。
- **只用 TRAIN 计数**：quantize 只读 E1c fit.json（TRAIN 两个半集，种子 101 / 102，cam_nf 档，sha 前缀有断言，第 275–278 行）、检查点和阈值文件，不读任何 VAL / fresh 文件。

### 3. 字节

- **S + k 的 int-ctx**：`arm_bytes` 第 852–853 行是 `G.int_ctx_bytes(S 消息) + G.int_ctx_bytes(k 消息)`。计数通道断言与字母表无关（第 796–797 行）。这与 E3-R 的 `sums_for` 是同一个函数、同一种调用；面板中位数逐位相等，见第 1 节。
- **级别计数场**：
  - 编码顺序是"已知位 → k−1 一元码 → 除最常见级以外各级的计数（给定余数）"，上下文只用前面的遍或已经解码的因果邻居。
  - 码长是自适应 KT 的理想码长，与 E3-R、E0 的码同类。文档字符串第 41–54 行写明了这一点，满足 D8-52 "写明是哪一种"的要求。
  - 正确性证据有四层：
    1. 单测：向量化计数与逐个决策的计数相同，真实算术编码在小场上往返成功，码长 ≤ 理想值 + 16 bit。
    2. 我**按文档字符串的规格独立实现**了上下文和 KT 码长，在 2 房 × 2 检查点的全部有向边上与实现相差 ≤ 6.4e−10 B（`compare_house.txt`）。
    3. 我用实现者的真实算术编码器，在**真实的整条消息**上做往返：val_r1_i0_s1，边 0→3，7 个智能体，128 × 128 × 32，130,424 个已知体素，kmax 7。解码结果与原场相同；实际 36,923.0 B，理想 36,922.9 B，开销 0.9 bit（`roundtrip_real_crop.txt`）。
    4. `level_relay`（第 838–848 行）对每条边断言 Σ v_l N_l = S、Σ N_l = k，计数正确。
- **每房一种格式 + 1 bit**：第 862 行 B = min(Σ_e Sk, Σ_e lev) + 1/8 B，Q∞ 的 B = Σ Sk。我的复算与实现逐位相等：
  - r1_i0：Q3 B = 259,184.034 B，Q∞ 716,402.064 B；
  - r1_i1：Q3 B = 504,861.288 B，Q∞ 1,448,622.086 B；
  - 冒烟 2 房的中位比值是 382,022 / 1,082,512 = 0.353。
  - 逐边取最小（`B_edge_min`）只作报告，代码里写明了它不可解码的原因，处理正确。
- **B_RO**：
  - 计算（第 582–593 行）：每个分量取上行字节最少的根，再加 (|cm| − 1) × 三态读出的码长。单测用穷举核对了根的选择。
  - 标注：summary 的 notes 写了"NOT within the floor (violates C5)"，不进判决。文字打印缺标注，见 S4。

### 4. 新种子的导出

- **同一调用**：`export_cmd`（第 607–653 行）直接调用冻结的 `E._export_and_save` → `E.export_job`。opts 的键与 E3-R 相同（cache、out、split、tiers、e1_jobs、script_sha256），各档的顺序也按 `G.COND_ORDER` 排，与 E3-R 第 1338 行相同。
- **同一公共随机数**：仍是 `G.pose_draws` / `G.frame_draws`，用 `[SALT, seed, rooms, index, a, j]` 作种子。
- **与保留种子不相交**：`check_new_seeds`（第 169–173 行）排除 {1, 2, 7, 101, 102} ∪ {200, 201, 202}。单测核对了新旧种子的抽样逐项不同。
- **运行前没有读过种子 3、4 的计分**：
  - 冒烟 eval 只跑了种子 1（`eval/jobs`、`eval_noqpost/jobs` 里只有 `_s1`）。第 924–925 行禁止冒烟 eval 读取种子 3、4。
  - 种子 3 的冒烟导出只做过 check-export，打印的是数组形状和字母表，没有分数。
  - 在 `analysis_20261027` 下全树搜索 `_s3` / `_s4`，只有这一个冒烟导出文件；其它脚本里没有用种子 3、4 做噪声种子的地方。

### 5. QPOST4

- 阈值只在 TRAIN 留出房上选：
  - `qpost-select` 只读 `trainho_*.json` 记录（第 711 行），非冒烟时断言 19 房 × 种子 200–202 齐全且没有失败（第 716–720 行）；
  - 导出用 `E.train_plan()` 加 `E.is_holdout`，与 E3-R 缓存的 cam_nf 抽样是同一个 `tier_maps` 调用；
  - 规则是 E3-R 的 `E._select`，空闲召回下限取同一半集 B6 线性读出在同一批房上的值。
- 不碰 VAL。R-C 的读出（Σq ≥ 0）另作一行报告。
- 只报告。正式运行需要防止误用冒烟阈值文件，见 S1。

### 6. 掩码、FPR、中继、clean

- 已知掩码断言：每个臂的 NET 与 LIN 读出（第 802–803 行）、K0（第 820 行）、QPOST4（第 892 行）都断言等于 B4 的 k > 0。FPR 有任何差异就判 INCOMPLETE（第 1113 行）。冒烟 d FPR 全是 0。
- 中继断言：`E.relay_sums` 断言来源位掩码不相交、信念等于直接求和；`level_relay` 断言级别计数复现 (S, k)；QPOST4 也断言中继结果等于直接求和。
- clean 保护：第 1112 行取该臂 clean 档 R8 NET 家具，要求 ≥ .99。冒烟值为 .997–.998。

### 7. 运行完整性

- 失败任务、缺任务、缺档、缺房都判 INCOMPLETE。
- 限量参数（export 的 `--houses` / `--only`，eval 的 `--files` / `--tiers` / `--seeds`）都会表现为缺房、缺档或缺种子，从而判 INCOMPLETE。
- 任务文件记录 `script_sha256`、`criteria_sha256`、`qpost_sha256`（第 763–764 行）；`eval_run.json` 还记了 g4_e3r、g4_noise_panels、g4_calib_stalk 的 sha。续跑只复用三个 sha 都相同的任务（第 951 行）。
- 漏洞见 M2 与 S1。

### 8. 独立复算

脚本 `g4/review_e5/rev_e5_recompute.py`，sha 前缀 3aee036b。只读种子 1 的 VAL 文件，文件名不是 `_s1.npz` 结尾就断言失败。

- **自写部分**：量化 DP 和级值、连通分量与发送方一侧、直接求和、特征、logit 分箱、家具 F1 与空闲召回、级别计数编码器的理想码长（按文档规格重写）。
- **共用部分**：只有 `E.load_ckpt`（加载权重）和 `G.int_ctx_bytes`（对照字节的定义本身）。`G.score` 只作交叉核对，自写 F1 与它相等（有断言）。

| 项目（种子 1 VAL，cam_nf） | 我的值 | 实现 / E3-R | 差 |
|---|---|---|---|
| Q3 表（A / B） | [−97, 5, 95] / [−93, 6, 93] | 判据文件 | 相同 |
| r1_i0 342101：Q∞ / Q3 NET 家具 | .97088709 / .96870002 | 冒烟任务文件 | 0 |
| r1_i1 342202：Q∞ / Q3 NET 家具 | .97555749 / .97194764 | 冒烟任务文件 | 0 |
| 2 房 × 2 检查点逐边 [S, k, lev] 字节 | — | 冒烟任务文件 | ≤ 6.4e−10 B |
| B(Q3) / B(Q∞)，r1_i0 A / B，r1_i1 A / B | .3618 / .3582，.3485 / .3445 | 相同 | 0 |
| 24 房 Q∞ NET 家具，342101 / 342202 | .972284969312721 / .972930500615567 | e3r summary | 0 |
| 24 房中位 S / k 字节（342101） | 664,784.595 / 87,554.847 | e3r summary | 0 |

输出文件：
- `review_e5/compare_house.txt`、`compare_panel.txt`、`mask_sensitivity.txt`、`roundtrip_real_crop.txt`；
- `out/house.json`、`out/panel.json`（只有分数和字节，不含地图）。

### 9. 单测

`python -B scripts/claude_v342/test_g4_quant_wire.py`：**11/11 通过**，用时 3 s（`review_e5/test_log.txt`），与冒烟里的 `test_log.txt` 一致。

## 上机前的顺序（建议）

1. 按 M1、M2 修改 aggregate（顺手改 S1、S3–S7）。
2. 写 D8-52 补充（fresh 规则和选臂规则），带 UTC 时间。
3. 重跑 `quantize` 与单测，记下新的判据 sha 与脚本 sha。
4. 在已有的种子 1 冒烟任务文件上重跑 `aggregate --smoke`，确认新规则和新的 reason 都能触发。注意脚本 sha 变了以后，冒烟 eval 需要重跑。
5. 正式运行顺序：导出种子 3、4 → 导出 TRAIN 留出房 → `qpost-select`（非冒烟）→ eval（种子 3、4，再加续接种子 1、2）→ aggregate。
