# G4-H1G 代码审查（独立审查者，2026-10-28 11:52–12:05 UTC）

**结论：LAUNCH OK WITH CHANGES**。要改的只有一项（M1）：**D8-75 补充里新加的保护项"门控臂 R8 表面 F1 ≥ 不门控 − 0.5 pp，违反判 NO"，代码里没有实现**。除此之外，门控表、落格、门控、闸门、H1 判据和其余精度保护都与 D8-75 一致。我独立复算了一个 TRAIN 留出任务和一个 H1 TRAIN 冒烟文件，结果与记录逐数相等。单测 9/9 通过。

M1 只涉及判决函数里的两行代码和一处单测，不碰拟合代码，所以 `gate_tables.json`（0bcf0a55）照用，不用重新拟合，也不用重跑冒烟。改完后重跑单测，并把新的脚本 sha 记进报告，就可以正式运行。

## 审查对象（sha256 已核对）

| 文件 | sha256 |
|---|---|
| scripts/claude_v342/g4_h1_gate.py | f0da0b1523cb… ✓ |
| scripts/claude_v342/test_g4_h1_gate.py | 04edeb9bef55… ✓ |
| g4/h1g/fit/gate_tables.json | 0bcf0a556bd6… ✓（`smoke: false`，19 房，cam_nf_s200–202，57 个导出文件） |
| 冻结依赖 g4_h1_e3l3 / g4_e3l3_w / g4_e3l3 / g4_e3l2 / g4_e3l / g4_e3r / g4_noise_panels / g4_ood_panel | dc0f69d7 / 010785f9 / c7523cc8 / bab73d64 / 3dc970d8 / 1d65d9a8 / 2313d75b / 3a6959da，**都与 HL.FROZEN、SELF_FROZEN 相等** |
| 闸门参照 g4/h1l/summary.json、g4/e7/summary.json、g4/e7/criteria_w.json、h1l_smoke/train/summary.json、e7/smoke/jobs/val_r1_i0_s1.json、h1/summary.json | 2a6ae99a / c45eaca7 / 320317aa / dad4a22b / a398cd79 / 28607243，**与脚本里钉的值逐个相等** |

另外两点：
- `g4/h1g/` 下目前只有 `fit/`、`smoke/` 和 `fit_run.log`。正式运行的 `h1/`、`guard/`、`aggregate/` 都还不存在，所以没有会被覆盖的正式输出。
- 我没有修改被审查的文件，也没有做 git 操作。我的脚本和输出都在 `g4/review_h1g/` 下。

## 1. 与 D8-75 及补充的一致性

### 1.1 门控表拟合
- **落格**：`cell_index`（第 190–206 行）。
  - Λ_b 必须是 0..64 的精确整数，且 |h_b| ≤ Λ_b，否则断言失败。
  - xbin = ⌊20(h_b+Λ_b)/Λ_b⌋，上限截到 39。这就是 ⌊(x+1)/0.05⌋，x = h_b/Λ_b 与 `L.belief_mean` 的定义相同（g4_e3l.py 第 204–207 行），x = 1 归入第 39 格。
  - Λ_b = 0 的体素记为 −1，不落格，也不门控。
- **强空闲格的阈值**：`fit_strong`（第 209–215 行）的条件是 `n ≥ 200` 且 `1000·n_free ≥ 995·n`，用整数比较，Λ_b = 0 那一行强制为 False。0.995 和 200 写死在第 122–123 行，没有可调参数。
- **只用 TRAIN**：
  - `fit_cmd` 第 409–415 行读 `W.TRAIN_HO`，并断言房表等于 E3-L3 官方 threshold.json 的 houses、draws 等于 cam_nf_s200–202；
  - 目标的构造 `G.Target(private[cm].max(0), Hh)`（第 321 行）与 G4-E7 的 wselect（g4_e3l3_w.py 第 255 行）逐字相同；
  - 读法是 encode_maps → source('round') → gabp(w,'round') → decode_np(DEC_CHUNK) → net_readout(j_occ)，与 eval_arm 相同。
- **读数与补充的表一致**（fit_run.log）：

  | 表 | 强空闲格 / 非空格 | 覆盖已知体素 | 解码器判占据中被改判 | 被改判中真为空闲 |
  |---|---|---|---|---|
  | 342101 学到的 w | 6/80 | 0.3002 | 0.00209 | 0.0 |
  | 342101 w = 4 | 3/35 | 0.1776 | 0.00311 | 0.00021 |
  | 342202 学到的 w | 11/80 | 0.6220 | 0.00427 | 0.00457 |
  | 342202 w = 4 | 5/35 | 0.3808 | 0.00341 | 0.00323 |

### 1.2 门控只改"解码器判占据、且落在强空闲格"的 R8 体素
- **规则本身**：`apply_gate`（第 218–224 行）只把 `(p == 1) & (idx ≥ 0) & lut[idx]` 的体素改成 0。
- **H1 面板**：第 610–616 行先门控再断言 `q >= 0` 与 `p >= 0` 相同，即已知区不变。
- **guard 面板**（`gate_eval_arm`，第 817–847 行）：
  - 门控作用在截获的 decode 输出上；
  - 除 R8 与 whole_miss 外，所有字段原样从 eval_arm 结果复制，包括 bytes、bytes_upper、K0、K0_sep、rate、gbp；
  - 在 cam_nf 档，逐行核对"用截获的 logit 重算的不门控行 == eval_arm 的行"。
- **截获顺序**：eval_arm 先对各分量调用 decode_np，最后对 K0 调用一次（g4_e3l3.py 第 544、551 行）。截获代码断言调用次数是 `nc + 1`，按分量顺序取用，对齐正确。

### 1.3 H1 判据与"修好了幻觉"三条（`decide`，第 1121–1159 行）
1. **floating 不再有害**：实现为 `h1_gated['floating'] != 'HARMFUL'`。冻结的 HL.verdict 在 4 格中**任一格** harm 时就判 HARMFUL（g4_h1_e3l3.py 第 540–541、635–636 行），harm 的条件是翻转精确率 < .50、区内 Δ < −1 pp、幻影块 > B6 + 2 三者之一。所以"不是 HARMFUL"正好等于 4 格都满足这三条，与 D8-75 一致。
2. **novel、scale 不比不门控差**：用等级比较（SAFE > AMBIGUOUS > HARMFUL）。这比括号里"不从 AMBIGUOUS 变成有害"更严，SAFE → AMBIGUOUS 也算变差（见 S5）。
3. **ctrl 自检**：用门控臂的 `ctrl_ok`，区间是 HL.CRIT 的 [+4.3, +8.6] pp。
- **w = 4 臂**：HL.aggregate 会把它判为 finite_w（总判是 "REPORT ONLY"），但各变体的判决和 ctrl_ok 照常算出，decide 用的正是这两项，所以没有问题。

### 1.4 精度保护
已实现的部分（`decide` 第 1145–1146 行，数据来自 `guard_cells` 第 1094–1118 行，VAL cam_nf 种子 3、4 × 2 检查点 = 4 格）：
- 家具 Δ ≥ −0.3 pp（`guard_furn_margin = -0.003`）；
- 空闲召回 Δ ≥ −1e-12；
- FPR 每行和汇总都逐位相同；
- 字节每房和汇总中位都逐位相同；
- fresh 4 格家具均值 Δ < −0.3 pp 时，只把 LEVER 降为 AMBIGUOUS；
- 优先级是 INCOMPLETE > NO > LEVER > AMBIGUOUS；
- 4 格缺任何一格、或某格不是 24 房 → INCOMPLETE。e7 summary 里 VAL 和 fresh 每个种子都是 24 房，这条不会误触发。

**表面 F1 −0.5 pp 保护项：没有实现。** 这是 M1，见下文。
- `D8_75`（第 135–143 行）里没有表面 F1 的阈值；
- `decide` 的 `g[k]` 只含 furn、free、fpr、bytes 四项；
- `guard_cells` 第 1111 行算出了 `d_surface`，但只在 report 第 1339 行打印出来，不进判决；
- 后果：表面 F1 掉 0.5 pp 以上时，脚本仍可能输出 **LEVER**，而补充写明这种情况要判 NO；
- 单测 `test_decide_branches` 里也没有表面相关的分支（`_cells` 不带 `d_surface`）。

### 1.5 这条保护项会不会真的起作用（只用已允许的数据）
表面 F1 门控 − 不门控（cam_nf，各行平均）：

| 来源 | 342101 学到的 w | 342101 w = 4 | 342202 学到的 w | 342202 w = 4 |
|---|---|---|---|---|
| TRAIN 留出 TRAIN_r1_i19（我复算） | −0.13 pp | −0.29 pp | −0.09 pp | −0.12 pp |
| H1 TRAIN 冒烟 floating_r2_i9 s203（我复算） | −0.10 | −0.17 | −0.02 | −0.02 |
| VAL 种子 1、val_r1_i0（冒烟记录，R-I 看过的房） | −0.28 | −0.38 | −0.21 | −0.25 |

clean 档在冒烟里掉 0.55–2.09 pp，但这一档不进判决。

w = 4 的 342101 已经到 −0.38 pp，离 −0.5 pp 的线只差 0.1 pp 左右。所以这条保护项很可能真的影响判决，必须写进代码，不能靠人读报告去补。

## 2. 闸门
- **H1 闸门**（`gate_h1l`，第 1049–1054 行；aggregate 第 1232–1241 行）：
  - 学到的 w、不门控那一臂的 HL.aggregate 结果，用 variants 和 verdict 与 g4/h1l/summary.json（sha 已钉）逐值比较；
  - 冒烟时对 h1l_smoke/train 比较，`equal True`；
  - H1L 的 versions 是 `w: learned`、`gbp_check: true`，与本批默认（开启 gbp_check）相同，不会因为这一项造成假的不等。
- **guard 闸门**（`gate_e7`，第 1057–1073 行）：
  - w = 4 不门控对 E7 的 `E3L3W_{ts}`，学到的 w 不门控对 `E3L3_{ts}`；
  - 比较每个（面板、种子、档、检查点）的 `E3._summ_arm` 全部字段；
  - 若 compared = 0，或有任何一项不等 → INCOMPLETE；
  - G4-E7 的 w = 4 臂调用的是同一个 eval_arm，w = float(w_star) = 4.0，阈值也是同一个 `criteria_w thresholds[ts]['w4']`（g4_e3l3_w.py 第 608、523、529 行），与本批的 `arm_cfg` 相同；
  - 冒烟中学到的 w 对 E7 冒烟记录 `equal True compared 6`。w = 4 那一支在冒烟里没法比，原因见 S2。
- **npz sha**：
  - 正式 h1 先串行核对 192 个 VAL npz（第 748–760 行）；
  - 每个任务再核对一次（第 712–715 行）；
  - guard 核对导出文件的 sha（第 865 行）。
- **不覆盖已有输出**：
  - `safe_out` 保证所有写入都在 h1g 下（第 168–173 行）；
  - fit 遇到已存在的表或非空的 jobs 目录就拒绝（第 403–407、457 行）；
  - h1 / guard 的任务记录只在"同脚本 sha、同表 sha、且完整"时跳过，其余情况拒绝运行（第 670–677、729–730、905–906 行）；
  - 各子目录的 summary.json / report.txt 是派生文件，aggregate 时重写，这一点在 G9 已写明。

## 3. 独立复算（`review_h1g/indep_gate_check.py`，sha 1ba6fa1d）

复算脚本**没有 import g4_h1_gate**（脚本里有断言），只用冻结的 HL / W / E3 / L / R / G。落格我用 `Fraction` 精确计算 ⌊(h/Λ + 1)/(1/20)⌋，门控用我自己写的"强空闲格掩码"。没有读取 H1 VAL，也没有读取种子 3、4 的任何门控读数。

### fit 模式（`indep_fit.json` 69b4fb14；202 s，11:57–12:01 UTC）
- **gate_tables 的 counts** 等于 38 个正式 fit 任务记录之和，4 张表都是 True；
- **用精确分数重新推出的强空闲格** 与表中列出的完全相同，4/4 True；
  - n < 200 的非空格数为 0，也就是说 N_MIN = 200 在这批数据上没有起约束作用；
  - 卡在 0.995 下方的格，例如 342101 学到的 w 的 (Λ=3, bin 39)，q = 0.99469，正确地没有算作强空闲格。
- **TRAIN 留出房 TRAIN_r1_i19（cam_nf_s200–202，24 个机器行）**：
  - 每格 8 列计数与冒烟记录、正式记录都逐数相等（4/4）；
  - 合并的 totals 也逐数相等，例如 342101 学到的 w 为 [78464, 696, 1648, 4624688, 4628328]；
  - 用正式表门控后，改判数与"记录计数在强空闲格上求和"逐数相等：

  | 表 | 改判数 | 目标为空闲 | 目标为占据 | 目标未知 |
  |---|---|---|---|---|
  | 342101 学到的 w | 1816 | 0 | 1816 | 0 |
  | 342101 w = 4 | 3909 | 6 | 3896 | 7 |
  | 342202 学到的 w | 1576 | 24 | 1264 | 288 |
  | 342202 w = 4 | 1891 | 16 | 1698 | 177 |

  这印证了补充里的判断：被改判的几乎全是真占据的结构体素。

### h1 模式（`indep_h1.json` bbb9ca59；38 s）
- 文件是 floating_train_r2_i9 s203，属于 TRAIN；
- 2 个检查点 × 2 个 w，改判数 408 / 706 / 136 / 142、目标拆分（例如 342202 学到的 w 为 [24, 88, 24]）、每格计数、8 个门控行和 8 个不门控行（R.scored 的全部字段），都与冒烟记录逐值相等。

## 4. 单测
`review_h1g/unit_tests_review.log`：jax-gpu 环境，torch 2.11.0，`CUDA_VISIBLE_DEVICES=`，11:57–11:58 UTC，**9/9 通过**（70 s）。

## 必改项

**M1. 实现 D8-75 补充的表面 F1 保护项（违反判 NO）。** 最小改法如下。

`g4_h1_gate.py`，`D8_75`（第 136 行后）加一项：
```python
             guard_surface_margin=-0.005,         # D8-75 补充：gated R8 surface F1 - ungated >= -0.5 pp（4 个 VAL cam_nf 格，种子 3、4）；违反 -> NO
```
`decide()` 第 1145 行改为：
```python
        g[k] = dict(furn=bool(c['d_furn'] >= C['guard_furn_margin'] - eps),
                    surface=bool(c['d_surface'] >= C['guard_surface_margin'] - eps),
                    free=bool(c['d_free'] >= -eps), fpr=bool(c['fpr_identical']), bytes=bool(c['bytes_identical']))
```
不改 `guard_ok` 的逻辑，表面不过就自动判 NO。`guard_cells` 已经算出 `d_surface`（第 1111 行），不用另加计算。

`test_g4_h1_gate.py`：
- `_cells` 加参数 `surf=0.0`，字典里加 `d_surface=surf`；
- 在 `test_decide_branches` 里加三条断言：
  - `_cells(0.0, surf=-0.005)` 判 LEVER（恰好 −0.5 pp 算过）；
  - `_cells(0.0, surf=-0.0051)` 判 NO；
  - 只有一格 `d_surface = -0.006` 也判 NO。

改完之后：
- 重跑单测；
- 在报告补充里记录新的脚本 sha 和单测 sha；
- **不用重新拟合**：fit 代码不变，`load_tables` 不核对表内的 script_sha；
- **不用重跑冒烟**：冒烟的 aggregate 会因为脚本 sha 不同而记一条 versions 原因，但冒烟本来就不出判决。

## 建议项

- **S1 改完后把读数也写进判决输出**：建议 `decide` 的输出里加 `val_mean_d_surface`，DECISION 那一行打印每格的 surface 是否通过，方便核对。
- **S2 w = 4 的 E7 闸门要到正式运行才第一次真正比较**：E7 冒烟用的是 w* = 6，而 seed 1 上没有 w = 4 的逐房参照，种子 3、4 又不能用，所以现在没有合法的预比对。代码路径与学到的 w 相同，已逐行核对，风险低；万一不等，两个 w 都判 INCOMPLETE，代价是重跑一次。
- **S3 两条保护项按构造必然成立，结果报告里应写明**：
  - "字节逐位相同"：门控臂直接复制了不门控臂的 bytes（第 843–845 行），线路不变是由构造保证的，而不是测出来的；
  - "空闲召回不降"：门控只把已知区里的 1 改成 0，空闲召回只会升不会降。
- **S4 落格必须用整数算法，不要用浮点**：x = h/Λ 经常正好落在 0.05 的边界上。用 float32 实现的 ⌊(x+1)/0.05⌋，在 TRAIN_r1_i19 上每个（检查点, w）会有 1.2–4.1 万个体素落到不同的格（例如 x = 0.75 落到 34 而不是 35）。342202 学到的 w 的强空闲格 (Λ=8, bin 35) 正好就是这种边界格。本脚本用的是精确整数算法，没有问题；若门控以后要线上部署，必须沿用同一算法，并在结果里写明。
- **S5 写明"不比不门控差"的口径**：实现是"等级不低于不门控"，SAFE → AMBIGUOUS 也算变差，比 D8-75 括号里的例子更严。建议结果报告写明这一口径（运行前定下，不算改判据）。
- **S6 正式运行显式给 `--workers`**：默认是 6（第 132 行）。冒烟中每个进程峰值约 2.6–2.8 GB。粗估的单进程耗时：h1 约 96 任务 × 2 文件 × 57 s，guard 约 96 任务 × 330 s。

## 披露
- 为了核对结构，我读取了 g4/e7/summary.json 中**不门控**的 VAL / fresh 种子 3、4 的 R8 surface_f1 与 furn_f1。这些是 D8-74 已经公布的读数，我没有计算或看过任何门控读数，也没有接触 H1 VAL 面板。
- 第一次启动时用错了解释器（没有 torch，三个进程秒退，没有产出）。之后改用 `D:/anaconda3/envs/jax-gpu/python.exe`，全程同时最多 3 个 CPU 进程。
