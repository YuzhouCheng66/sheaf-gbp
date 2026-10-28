# G4-H1L / G4-E6L 代码审查：E3-L3 过留出形状面板（D8-69）、主档 cam 上 E3-L3 接在按岛组合位姿之后（D8-70）

- 审查者：独立代码审查（Claude），2026-10-28 03:39–04:01 UTC 审查与复算，04:02 UTC 写报告。
- 被审文件（审查前后 sha 未变）：
  - `scripts/claude_v342/g4_h1_e3l3.py`（`dc0f69d7…`），`test_g4_h1_e3l3.py`（`cde664cc…`）
  - `scripts/claude_v342/g4_e6_e3l3.py`（`eb083676…`），`test_g4_e6_e3l3.py`（`3de53333…`）
- 冻结依赖（只读；sha 与两脚本的 `FROZEN` 一致，单测的 `check_frozen` 通过）：g4_ood_panel 3a6959da、v420ood 9e1c7c2a、g4_e3l3 c7523cc8、g4_e3l2 bab73d64、g4_e3l 3dc970d8、g4_e3r 1d65d9a8、g4_noise_panels 2313d75b、g4_island_compose b7438b98、g4_registration e9574c7e。
- 审查者文件：`claude_handoff/analysis_20261027/g4/review_h1l_e6l/`
  - `rv_h1l_train.py`（9c42681e）/ `.log` / `.json`：H1L 在 1 个 TRAIN 文件上的独立复算与闸门实测；
  - `rv_e6l_val.py`（1da326a4）/ `.log` / `.json`：E6L 在 val_r1_i0_s1 上的独立复算；
  - `rv_sha_preflight.py`（91852dbb）/ `.log`：192 个 VAL npz 的 sha 核对（只算文件哈希，不读数组、不计分）；
  - `rv_vsnet_path.py`（7beec0bb）/ `.log`：只报告的 E3L3 − NET 代码路径（只打印键名，不打印 VAL 数值）；
  - `unit_tests_h1l.log`、`unit_tests_e6l.log`。
- **数据纪律**：H1L 的复算只用 TRAIN 户型（R-I 临时目录 `ri_ood_export` 里的 `floating_train_r2_i29_s203.npz`，加上 `h1_export` 里 H1 冒烟用过的 TRAIN 文件 `ctrl_train_r1_i9_s203.npz`）。E6L 只在 R-I 看过的 val_r1_i0_s1 上算。没有读任何 G4-H1L 的 VAL 文件内容；192 个 VAL npz 只算了 sha256。
- 资源：本机 CPU，`CUDA_VISIBLE_DEVICES=`，同时最多 3 个审查进程，解释器 `D:/anaconda3/envs/jax-gpu/python.exe`（base 环境没有 torch）。没有碰 GPU、uk-linux、bigboy，没有做 git，没有改被审文件或冻结文件。

## 结论

| 批次 | 结论 | 必改项 |
|---|---|---|
| **G4-H1L**（D8-69） | **LAUNCH OK** | 无 |
| **G4-E6L**（D8-70） | **LAUNCH OK** | 无 |

- 两个脚本对 D8-69 / D8-70 的面板、臂与配对、阈值、解码路径、指标、判据与 INCOMPLETE 条件都照做；没有发现会改变判决的错误。
- 独立复算**逐位相等**：
  - **H1L**（TRAIN floating 文件）：两检查点的 8 + 8 行 R8 读数与冻结的正式评估函数 `g4_e3l3.eval_arm` 逐位相等；新物体区 TP / FP / FN、自配对翻转（真 / 假）、每房幻影块与审查者自写的计数逐值相等。
  - **E6L**（val_r1_i0_s1）：用冻结的 `g4_island_compose.run_task` 本身重建对齐后的地图。在 cam_nf noreg、cam noreg、cam e6 三种行上，B4 / B6 / NET 与 `g4/e6/jobs` 逐行相等，E3L3 两臂与被审脚本冒烟记录逐行相等，该房的 E3L3(e6) − NET(e6) 也与冒烟汇总逐位相等（+0.20307291226174806 / −0.13720323935407785 pp）。
- 一致性闸门实测有效：
  - 同房同种子的 H1 记录：相等；
  - 同房换种子的记录：B4 / B6 的 15 个块全部不等，被挡下。
- 单测：H1L 8/8，E6L 7/7。
- 下面的建议项都不阻塞学到的 w 的正式运行。**S1 是跑有限 w 读法之前应改的。** 发射时请注意 §6 的内存余量。

---

## 1. 与 D8-69 的一致性（g4_h1_e3l3.py）

| D8-69 条目 | 实现 | 证据 |
|---|---|---|
| 面板：4 变体 × VAL 24 房 × 噪声种子 1、2，cam_nf，复用 `h1_export` 的 192 个 npz，逐个核对 sha | `--formal` 断言四变体、种子 1、2，H1 记录取 `g4/h1/jobs`（L792–798）。每个文件评估前算 sha，与 H1 记录 `seeds[s].file_sha256` 比较，不符则任务失败（L428–431）。 | 审查者核对 192 个文件 sha：**0 个不符**（`rv_sha_preflight.log`）。H1 正式记录 96 个，都由 3a6959da 写成（单测 `test_frozen_and_pins`）。 |
| 臂：E3L3_342101 ↔ B6_A，E3L3_342202 ↔ B6_B；检查点 1fa0cb39 / 3d332d40 | `E3L3_CKPTS` 钉完整 sha，阈值文件也钉住（L90–93）。加载时核对 sha 与 `ck['seed']`（L175–180）。证据表断言等于配对的 B6 表（L233–235）。 | 复算里 w = 1240.3678 / 1357.7167，j_occ = 1998 / 2006。 |
| 阈值用各自的 R8 threshold.json | `jo = thr['R8']['j_occ']`（L185） | 同上 |
| 每机解码自己的信念，学到的 w，逐跳取整 | `e3l3_preds`（L195–219）：encode_maps → source 'round' → 每分量 local_forest + gabp(w, 'round') → decode_np(chunk 2) → `net_readout(lg[i], k, j)`。逐语句对照 `g4_e3l3.eval_arm` 的 R8 段（g4_e3l3 L518–548），读出部分相同；eval_arm 多出的浮点 GaBP、字节、K0 不影响预测。 | 见 §3 |
| 已知 = Λ_b > 0，断言与 k > 0 逐位相同 | 不一致体素计数（L215），> 0 判 INCOMPLETE（L596–597）；另有 `p >= 0 == known` 的硬断言（L340–341）。 | 复算 known_mismatch = 0。有限 w = 0.9 时断言会触发（单测）。 |
| 指标：新物体区 = dil6(ood_mask)，整体 = 分量展开逐行均值，翻转 = E3L3 占据 ∧ B6 空闲 ∧ S ≤ 0 ∧ 范围，S 只取配对表的和，幻影块 | 新物体区用 `OP.object_zones`（L283）。行的权重：E3L3 每机 1，对照每分量 \|分量\|（L319、L352），展开后口径与 NET 相同。翻转用 `OP.flip_counts(p, B6, sums[自己的表][0], …)` 的 `le_-0`（L349、L520）。幻影块按每台机器 / n 相加（L359），对照是每分量 × \|分量\| / n（L326），两者等价。 | 复算逐值相等（§3） |
| 判据（安全 / 有害 / 其余 AMBIGUOUS） | `CRIT`（L115–116），并断言除 ctrl 外与 H1 的 `OP.CRIT` 相同（L117）。`_cell` 的安全 / 有害逻辑（L536–541）与 `OP._cell` 逐项相同。翻转分母为 0 记通过（P3）。 | `test_verdict_branches` 覆盖了边界：新物体区恰为 −1 pp、翻转恰为 .70、幻影块恰为 +0.5 都判安全。 |
| ctrl 自检：4 格 E3L3 − max(B4, B6) ∈ [+4.3, +8.6] pp，否则 INCONCLUSIVE | `ctrl_lo = 0.043`、`ctrl_hi = 0.086`；max 取 (B4, B6_A, B6_B)（L604–614），沿用 H1 的写法。不通过时整批判 INCONCLUSIVE（L649–650）。 | 单测：+4.5 判过，+4.1、+9.0 判 INCONCLUSIVE。 |
| 只报告：E3L3 − NET（NET 引 `h1/summary.json`），按家族与高度，z 三级分布相对 ctrl 的变化，有限 w | NET summary 钉了完整 sha（L104、L692–696）。`vs_net`（L542–554），家族 / 高度（L531–535），O5 的 z 计数（L257–268、L558–569）。 | TRAIN 冒烟（net = None）走不到 `vs_net` 这条路径。审查者用真实的 H1 summary 键结构走了一遍：4 个格的 12 个字段都有值，没有 KeyError 被吞掉（`rv_vsnet_path.log`）。 |
| INCOMPLETE | 下列任一条触发（L573–603）：失败、混用版本、外来 split、跳过、闸门不等（O3）、缺房或缺种子、已知不一致、GaBP 核对失败。 | 单测覆盖每一条。 |

小注：TRAIN 冒烟（`h1l_smoke/train`）的 ctrl 自检是 +1.03 到 +1.66 pp，"OUT"。这不是实现问题：唯一的冒烟房 r1_i9 很容易，B6 已经是 .987；H1 冒烟里 NET 在同一房上也只有 +1.0 到 +1.7 pp（`h1_smoke2/train/report.txt` L9–13、L33–37）。R-I 的 6 个 TRAIN 房上是 +8.10 / +8.12 pp（`h1l_smoke/ri_export/report.txt` L59）。按 D8-55，NET 在 VAL ctrl 上是 +6.30 到 +7.36；E3-L3 比 NET 低约 0.5 pp，所以 VAL 上应落在带内。

## 2. 与 D8-70 的一致性（g4_e6_e3l3.py）

| D8-70 条目 | 实现 | 证据 |
|---|---|---|
| 面板：VAL + fresh × 种子 1、2 × cam = G4-E6 的 96 个任务 | `OI.expected_jobs`。`--formal` 断言 96 个（L549–551）；每个任务名必须在钉住的 G4-E6 清单里（L552–553）。 | 单测 `test_pins_and_manifests`：E4、G4-E6 清单各 96 个，名单相同。 |
| 地图重建、E4 边估计、按岛组合、整体对齐行照冻结的 g4_island_compose | 用 OI 的函数（e6_corrections、believed_positions、readouts、same_K），对齐段逐行照 `OI.eval_arm6`（L264–272）。外层循环是复写的 `OI.run_task`（见 S4）。重建的地图与 E4 / G4-E6 job 的 pub_sha256、k0_sha256 比对，不符则任务失败（L241–243）。 | §3：用冻结的 `OI.run_task` 本身重建，三种行都与被审脚本的冒烟逐行相等。 |
| 两个 E3L3 读出，每机解码自己的信念 | `e3l3_rows` → `HL.e3l3_preds`（L165–171）。k 是对齐后地图的观测计数，与 E3-L3 正式评估相同。行为 `[scored, 1]`。 | 同上 |
| 一致性闸门：NET / B6 对齐行与 `g4/e6/jobs` 逐行相等，否则 INCOMPLETE | 每个计算的行，5 个读出都做 JSON 往返后逐值比较（L174–182、L280–282）；另外比岛（L250–251）、K（L252–253）、每分量的 G（L275–276）。复用臂也照样比（L258–262）。任一不等 → `gate.problems` → 汇总判 INCOMPLETE（L460–461、L474–476）。job 文件 sha 钉清单：E4 9d729e57，G4-E6 4c98930c（L88、L194–197）。 | 单测 `test_gate_rows`：1e-12 的扰动和 None 都会被抓住。冒烟两房的闸门全部相等。 |
| 判决行 = 整体对齐行、分量展开均值 | `cell_stats` 用 `G._expand` + `G.summarise_rows`，与 G4-E6 / E3-L3 同一口径（L362–371）。 | `test_cell_stats` |
| 非劣：VAL 4 格 ≥ NET(e6) − 1.0 pp | L410 | 单测：恰为 −1.0 判过 |
| 组合有效：E3L3(e6) − E3L3(noreg) ≥ +3 pp，4 格 | L411，noreg 取 cam 的对齐行 | 单测：+2.9 判 AMBIGUOUS |
| 保护：空闲召回 ≥ NET − 0.5 pp；伤害房（房均值差 < −2 pp）≤ 3 / 24 | L412–413；伤害房用 `_house_mean`，与 G4-E6 的 `OI._house_mean` 逐字相同（L341–343、L374–375）。 | 单测：3 个过、4 个不过 |
| fresh 4 格均值 ≥ −1.0 pp，只降级 | L409、L414–420：只把 NONINFERIOR 降为 AMBIGUOUS。fresh 的空闲召回与伤害房只报告，沿用 G4-E6 O8（S5）。 | 单测 |
| 结论：全过 → 非劣；任一 VAL 格 < −2 pp → 不非劣；其余 AMBIGUOUS | L417–424 | 单测：−2.1 判 NOT_NONINFERIOR，−1.2 判 AMBIGUOUS |
| 只报告：岛间损失（cam_nf − e6，E3L3 与 NET 各自）、按岛数、配对池化、有限 w | 岛间损失、按岛数、配对池化（L380–397、L346–359） | 冒烟里都能读出 |
| INCOMPLETE | 缺失、失败、跳过、闸门不等、已知不一致、GaBP 核对失败、版本混用、任务数 ≠ 96、清单 sha 不符（L444–480） | `test_aggregate_blocks` |

## 3. 独立复算（不用被审脚本的解码函数）

### 3.1 H1L：TRAIN 文件 `floating_train_r2_i29_s203.npz`（`rv_h1l_train.py`，03:44–03:54 UTC）

- **参照侧**：
  - R8 行：直接调用冻结的正式评估函数 `g4_e3l3.eval_arm`（learned w、R8 j_occ、chunk 2）。
  - 预测：自写最小循环，只用冻结原语（`E3.load_model / source`，`L.encode_maps / local_forest / gabp / decode_np`，`R.net_readout`），先断言其行与 eval_arm 逐位相等，再用于计数。
  - S 与 k：自己用 `R.stalk` 直接求和（不走中继）。B6 用 `S > θ`。
  - 自写新物体区 TP / FP / FN、翻转（`p = 1 ∧ B6 = 0 ∧ known ∧ region ∧ S ≤ 0`）、26 连通幻影块（3 × 3 × 3 邻域内没有真值占据）。
- **被审侧**：`HL.evaluate_file(path, HL.load_models(1), gbp_check=True)`。

| 臂 | 行（8 行 × 8 个键）对 eval_arm | 新物体区 TP / FP / FN | 翻转 真 / 假（精确率） | 每房幻影块 | 已知不一致 / GaBP |
|---|---|---|---|---|---|
| E3L3_342101 | 逐位相等 | 11,664 / 248 / 800，相等 | 88 / 144（.379），相等 | 2.0，相等 | 0 / 通过 |
| E3L3_342202 | 逐位相等 | 11,880 / 232 / 584，相等 | 104 / 120（.464），相等 | 0.0，相等 | 0 / 通过 |

（这是 TRAIN 户型上的单房读数，不是判决，只证明计数一致。）

- **O3 闸门实测**：H1 冒烟的 TRAIN 记录 `ctrl_train_r1_i9`。对 `ctrl_train_r1_i9_s203.npz` 跑 `evaluate_file`：
  - 与 s203 的记录比：**相等**；
  - 与 s204 的记录（同房、换噪声种子）比：B4 / B6_A / B6_B 的 rows、zones、kinds、families、phantoms **15 项全部不等**。

  加上单测：
  - npz sha 被改 → 任务失败；
  - H1 记录不是 3a6959da 写的 → 失败；
  - 记录缺失 → 失败；
  - 行上加 1e-15 → 不等。

  闸门挡得住文件、种子、房和计分代码层面的错位。

### 3.2 E6L：val_r1_i0_s1（`rv_e6l_val.py`，03:45–04:01 UTC）

- **地图侧**：直接调用冻结的 `g4_island_compose.run_task`，臂裁成 cam_nf {noreg}、cam {noreg, e6}，不做真值规范行。用包装 `OI.readouts` 截下 `eval_arm6` 对齐段送去打分的地图。重建与对齐都是 G4-E6 原代码，没有用被审脚本的任何部分。
- **E3-L3 侧**：在截下的地图上调用冻结的 `g4_e3l3.eval_arm`。

| 行 | B4 / B6_A / B6_B / NET×2 对 `g4/e6/jobs` | 同上对被审冒烟 | E3L3_342101 / 342202 对被审冒烟 | 房均值 E3L3 − NET（pp） |
|---|---|---|---|---|
| cam_nf noreg（对齐行） | 5 / 5 相等 | 相等 | 8 行 × 8 键，相等 / 相等 | +0.0955 / −0.0206 |
| cam noreg（对齐行） | 5 / 5 相等 | 相等 | 相等 / 相等 | −1.4953 / −2.0267 |
| **cam e6**（判决行） | 5 / 5 相等 | 相等 | 相等 / 相等 | **+0.2031 / −0.1372** |

- 被审冒烟汇总里该房的 `per_house_pp` 是 +0.20307291226174806 / −0.13720323935407785，与复算逐位相等。
- 岛划分相同（1 个接受边岛）。
- GaBP 与闭式解差 3.9e-13 / 1.9e-13，违例 0；已知不一致 0。
- 冒烟记录由 eb083676 写成。

## 4. 有限 w 读法（--w）：只报告、不判决

- **H1L**：
  - `--w` 为数值时必须给 `--w-thr`（L786–790）；`load_e3l3` 换 w 与阈值（L186–189）。记录里写 `w`、`w_thr_sha256`、`w_used`、`j_occ`（L404、L413）。
  - 汇总：只要有一条记录的 w 不是 learned，`finite = True`（L697）→ verdict 固定为 `'REPORT ONLY (finite w)'`，判据照算，写进 `would_be`（L646–648）。INCOMPLETE 仍优先。
  - 默认输出目录加后缀 `_w{w}`（L806–808）。
  - 已知断言照旧，w ≤ 1 时会触发（单测 w = 0.9）。
- **E6L**：
  - 同样的参数检查（L538–541）；汇总从 versions 判 finite → verdict 改为 REPORT ONLY，判决写进 `would_be`（L431–432、L496）；输出目录加后缀（L542）。
  - NET / B6 与闸门不受 w 影响。
- 结论：只报告的实现正确。但 w 与阈值文件的对应完全靠操作者，见 S1、S2。

## 5. 单测

- `test_g4_h1_e3l3.py`：**8 / 8 通过**，42 s（`unit_tests_h1l.log`）。其中：
  - `test_decode_reproduces_formal_task`：val_r1_i0_s1 上 E3L3 行与正式读数相等；
  - `test_matches_ri_prototype_on_train_ood`：与 R-I 原型逐值相等。
- `test_g4_e6_e3l3.py`：**7 / 7 通过**，18 s（`unit_tests_e6l.log`）。
- 都要用 `D:/anaconda3/envs/jax-gpu/python.exe`；base 环境的 python 3.12.4 没有 torch，import 就失败。

## 必改项

无。

## 建议项（不阻塞学到的 w 的正式运行）

- **S1（跑有限 w 读法之前应改）`--w-thr` 不认 G4-E7 的 `criteria_w.json`，w 与阈值之间也没有绑定。**
  - `parse_w_thr` 要的是顶层 `{"342101": …}`（H1L L158–164）。G4-E7 写的文件顶层是 `format / w_star / w_star_label / thresholds / …`，阈值在 `thresholds[ts][label]['R8']['j_occ']`（g4_e3l3_w.py L442–444；审查者查过 `g4/e7/smoke/criteria_w.json`，label 为 learned / w3 / w4 / w6 / w8）。
  - 所以操作者只能手抄一个中间文件，`--w 4 --w-thr <w6 的阈值>` 也会静默通过。
  - 建议直接接受 `criteria_w.json`：断言 `float(--w) == w_star`；取 `thresholds[ts][w_star_label]`，并核对其 `w` 字段；断言 `checkpoints[ts].sha256` 与本批钉的相同；criteria 的 sha 记进记录。这正是 D8-69 / D8-70 写的"若 G4-E7 第 0 步已选出 w*"。
- **S2 防止不同 w 的记录互相覆盖。** 续跑逻辑只在"同脚本 sha 且同 w"时跳过（H1L L820，E6L L559）。若有人用 `--w 4 --out g4/h1l`（或 `g4/e6l`），会**静默覆盖**学到的 w 的正式记录。建议：输出目录里已有不同 w（或不同 `w_thr_sha256`）的记录时拒绝运行；`w_thr_sha256` 也加入 H1L 的 versions 混用检查（H1L L576 目前没有；E6L L446 有）。
- **S3 H1L 的 192 个 npz sha 改成真正的"运行前"核对。** 现在是每个 worker 在评估各自文件之前核对（L428–431）；结局一样（不符 → 失败 → INCOMPLETE），但会先浪费算力。审查者已核对当前 192 个 0 不符。建议在 `main` 的 `--formal` 分支先串行核对一遍再起进程池（约十几秒）。
- **S4 E6L 的外层循环是复写，不是只 import。** D8-70 写的是"全部照冻结的 g4_island_compose（只 import）"。实现复用了 OI 的全部函数，但把 `OI.run_task` 的循环复写了（L211–297）；复写省了 OI 的 `reached8 == 分量` 检查（OI L473–474），别处逐行相同。地图 sha、岛、K、G 与 5 个读出逐行过闸门，审查者用冻结的 `run_task` 复算也逐位相同，所以不影响判决。建议在发射记录里写一句"闸门保证与冻结管线逐行相同"；或者以后改成包装 `OI.readouts` 的写法（`rv_e6l_val.py` 就是这样做的），做到字面上的"只 import"。
- **S5 写进运行前口径**：E6L 的空闲召回保护与伤害房只在 VAL 4 格判，fresh 的这两项只报告（O6，沿用 G4-E6 O8）。fresh 均值指的是 E3L3(e6) − NET(e6)。D8-70 原文的"每格""fresh 4 格均值"都有这层歧义，建议在 D8-70 补充里明写。
- **S6 H1L 并列配对行（R-I 草案 D）。** E6L 已有配对池化和配对逐行均值（O10），H1L 只有池化 F1 一列。D8-69 的"整体 E3L3 − B6 ≥ 0"用的是逐行均值；无家具分量上的零星误报会把行 F1 记成 0（R-I §1.2 的口径假象）。建议只报告地并列"配对逐行均值"，判决不变。
- **S7 E6L 的 `--aggregate-only` 不带 `--formal` 会被拒绝**（L536–537 要求 `--jobs` 与 `--formal` 二选一）。这没错，但 docstring 写成了 `[--formal]`，建议改为必填。

## 6. 发射注意（操作，不是代码问题）

- **内存**：04:00 UTC 本机 63.7 GB 内存只剩约 2.0 GB 空闲，有 15 个 python 进程（其他批次在跑）。
  - 峰值：H1L 每个 worker 约 2.7 GB（冒烟 `peak_mb_max` 2,655），E6L 约 3 GB。
  - 在这种负载下，审查者的单文件评估慢了一倍：H1L 64 s 对冒烟的 30 s；E6L 的冻结管线一房用了 601 s，被审冒烟是 197 s。
  - 建议发射时 `--workers` 不超过（空闲 GB − 4）/ 3，两批不要同时各开 8 个。
- **预计耗时**（空载时）：H1L 96 个任务 × 约 60 s / workers；E6L 96 个任务 × 约 200 s / workers（8 个进程约 40 分钟）。
- 解释器用 `D:/anaconda3/envs/jax-gpu/python.exe -B`，并设 `CUDA_VISIBLE_DEVICES=`。
- 正式输出目录 `g4/h1l/`、`g4/e6l/` 目前不存在，不会混入旧记录。
