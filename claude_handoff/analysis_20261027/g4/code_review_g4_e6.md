# G4-E6 代码审查：按岛组合的位姿 stalk + 冻结 NET 读出（D8-51）

- 审查者：独立代码审查（Claude），2026-10-27 13:06–13:17 UTC 审查与复算，13:17 UTC 写报告。
- 被审文件（审查前后 sha 未变）：
  - `scripts/claude_v342/g4_island_compose.py`（`13329332…`）
  - `scripts/claude_v342/test_g4_island_compose.py`（`fbfeead0…`）
- 冻结依赖（只读，sha 与脚本 `FROZEN` 一致）：`g4_registration.py` e9574c7e、`g4_noise_panels.py` 2313d75b、`g4_e3r.py` 1d65d9a8。
- 复算脚本与读数：`claude_handoff/analysis_20261027/g4/review_e6/`（`indep_e6.py`，sha 90620817；`indep_{cam,cam_nf,harsh}.json/.log`；`unittest.log`）。只在 val_r2_i3_s1 上计分；harsh 只算 K，不计分。没有碰其他 VAL 任务或任何 fresh 任务。
- 资源：最多 3 个 CPU 进程，`CUDA_VISIBLE_DEVICES=`，没有用 GPU 或 uk-linux，没有做 git。

## 结论：**LAUNCH OK**

没有必改项。组合规则、判决行、读出配对、计分口径和判据逻辑都忠实于 D8-51。独立复算在多岛房 val_r2_i3_s1 上与冒烟读数**逐数相等**：

- cam、cam_nf 两档，2 个臂 × 2 种行 × 5 种读出 × 3 个指标，共 120 个数，差全为 0.0；
- K 的最大差 1.1e-16；
- 岛划分相同，对齐 G 逐位相同。

单测 9 / 9 通过。下面的建议项（S1–S7）都不阻塞运行。其中 S1（O8 的读法）需要协调者在运行前确认一句。

---

## 1. 不用真值【通过】

- **K 的来源**：G4-E6 臂的 K 只来自 `e6_corrections(n, reg['edges'], believed_positions(scans_pub))`（L253）。
  - 边记录经 `edge_view` 白名单只读 `i / j / corrected / est`（L157–159）。
  - `believed_positions` 只读本机公共帧扫描原点 `s[0][:2]`（L180–182）。按 `G.measure` / `G.to_public`，`s[0]` 是带位姿噪声、经本机机间误差变换后的相信原点。
  - `MAs` 只在 `ceiling_K` 缺 E4 K 时才用（L247–258），冒烟的来源是 `e4_job`。`positions` 只进 `truth_gauge_K`（L218、L435），即只报告的行和残差读数。
- **E4 边记录本身不含真值**：`corrected = not array_equal(M, I3)`，`est = params(M)`（g4_registration L774–777），都由 `register_forest(L_pub, …)` 只从相信地图得出。
- **"接受"与"corrected"等价**：E4 的 96 个 job 里，未改正的边全部是门控停止且 ICP 0 步（cam_nf 186、cam 182、harsh 215，共 670 条树边 / 档）。没有"恒等过门但被当作拒绝"的歧义边。
- **单测**：`test_e6_K_never_reads_the_truth` 用有毒记录（读真值字段即抛错）、随机 MAs 和打乱真值字段三种方式，证明 e6 / e4rule / noreg 的 K 不变，并对源码做了禁词检查。
- **对齐 G**：`eval_arm6` 的 G 来自 `R.register_pair(真值平面, 本臂 B4 融合信念平面)`（L322–330），与 E4 `eval_arm`（g4_registration L626–645）逐行同构。它只用于不可观测的整体规范。
  - 单测 `test_judged_row_is_e4_eval_arm` 验证了 G 与 B4 行和 E4 相同；
  - 冒烟的 O17 比较为 noreg / e4rule / ceiling 三个臂在 asis / aligned / G 上 2 / 2 相等；
  - 我的复算 G 也逐位相同（见 §5）。
- **只报告的真值行有标注**：真值规范行键名为 `truth_gauge`（O11）；ceiling 记 `ceiling_source`。判决只读 `aligned` 行（`JUDGED_ROW`，L113）。

## 2. 组合规则【通过，方向经独立验证】

- **岛**：`accepted_islands` 用并查集求 corrected 树边的连通块，单点也成岛（L162–177）。
- **组合方向**：边 (i = 父, j = 子) 的 `est` 是 `register_pair(ev[父], ev[子])` 的 M，把 j 的帧映到 i 的帧（g4_registration L410–413、L496–498）。E4 `compose` 用 `C_child = C_parent · Δ`（L462–470）。实现取 `C_j = C_i·Δ`，反向取 `C_i = C_j·Δ⁻¹`（L195、L204），方向一致。
  - 接受边是无线树边的子集，所以岛内路径唯一，遍历顺序不影响结果。
  - 锚点取岛内最小编号。左乘规范对锚点不变：`mean_transform` 对左乘等变，所以 K 与锚点无关。单测 (b) 也验证了这一点。
- **规范与修正**：`H = mean_transform(成员 C, 相信位置均值)`，`K = H⁻¹C`（L206–209）。被拒边不进 `nb`，不传修正。
- **独立重算**（`indep_e6.py`：自写 SE(2)、BFS 岛、规范和扫描变换，不 import 被审脚本）。val_r2_i3_s1 是 8 机单分量，7 条树边接受 4 条：
  - 岛 `[[0,3],[1,6,7],[2,4],[5]]`，与实现相同，也与 E4 可观测岛相同；
  - K 与冒烟 job 的最大元素差：cam 1.1e-16、cam_nf 1.1e-16、harsh 1.1e-16。
- **方向的真值诊断**（只在审查里用真值）：岛内两两的修正后相对帧误差 `(K_x A_x)⁻¹(K_y A_y)`。

  | 档 | 正确方向 G4-E6 | 不配准 | 故意反向（C_j = C_i·Δ⁻¹） |
  |---|---|---|---|
  | cam：位移均值 / 最大，偏航最大 | 3.07 / 5.08 cm，0.91° | 6.65 / 10.85 cm，2.33° | 14.6 / 22.8 cm，4.17° |
  | harsh：位移均值 | 4.89 cm | 13.35 cm | 28.5 cm |

  所以方向与 E4 的边估计定义一致。
- **与 R-A 原型的差别只在枢轴**：
  - 相信位置与真位置差 1.5–12.0 cm（cam），K 的差 1.4e-5（cam）、6.5e-5（harsh）；
  - R-A 原型（真位置枢轴）在本房对齐行 NET_342101 = .89003，实现 .89028，差 +0.025 pp，属二阶影响，与 R-A §2.2 的判断一致。

## 3. 判决行【通过】

- **对齐行的流程**：每个臂先用 `G.fuse('sum')` 得到 B4 三态信念，逐分量做 `register_pair` 得到 G_c；每机按 `G_c·K_a` 从 pub 扫描重新做 V3 积分；B4、B6_A、B6_B、NET_342101、NET_342202 全部在同一组对齐地图上读出（L314–332）。与 E4 `eval_arm` 相同，G_c 恒等时同样沿用原图。
- **证据表与检查点的配对**（L265–286）：
  - NET_342101 的 recipe 证据表与 theta 断言等于 E1c fit 半集 A，NET_342202 等于半集 B（L281）；
  - NET 在自己那半集的 `relay_sums` 上推理（L304–306），阈值 j 读各自的 `threshold.json`；
  - B6 用对应半集的线性读出；
  - 检查点与 fit 都有 sha 前缀断言（L271、L277）。
- **已知掩码**：B6 / NET 的已知掩码断言与 B4 相同（L301、L309），所以未知→已知 FPR 在各读出间一致。
- **计分口径**：`E.scored`（即 `G.score(strata=True)` 加 k 分层），每分量一行 `[row, |分量|]`。汇总用 `G._expand` 按智能体数展开后取均值（`cell_stats` L553–566），与 E3-R（g4_e3r L1043）和 E4（L887、L958）相同。
- **未定义 F1**：`check_none_positions` 要求各臂未定义 F1 的行位置相同（L569–579）。实测 E4 的 2,352 行与 R-A 复算的 3,672 行里 furn_f1 = None 都是 0，所以不会误触发 INCOMPLETE。

## 4. 判据与汇总【通过】

`verdict`（L582–606）逐条对照 D8-51：

| D8-51 | 代码 | 判定 |
|---|---|---|
| VAL 4 格 ≥ +3 pp | `val_all_ge_3`：`>= 3.0` | ✓ |
| fresh 4 格均值 ≥ +3 pp | `fresh_mean_ge_3` | ✓ |
| fresh 无一格 < +1 pp | `fresh_none_lt_1`：`min >= 1.0` | ✓ |
| cam_nf：NET(e6) ≥ NET(noreg) − 1.0 pp，VAL 4 格 | `nf_guard`：`>= -1.0`，取 cam_nf 对齐行 | ✓ |
| cam 空闲召回 ≥ −0.5 pp | `free_guard`：NET free_recall，对齐行，VAL 4 格（O8） | ✓（见 S1） |
| 伤害房（房均 NET 降超过 2 pp）≤ 3 / 24 | `harm_le_3`：房均 < −2 pp 计数 ≤ 3，VAL 4 格（O8、O9） | ✓（见 S1） |
| 不够：VAL 任一格 < +1 pp，或 cam_nf 保护不过 | `insuff`（L596） | ✓ |
| 其余 ambiguous | L601–604 | ✓ |

- **INCOMPLETE 的触发条件**：
  - E4 job 缺失；
  - 任务缺失、失败或跳过；
  - noreg B4 行与 E4 不等（O17）；
  - noreg 的 NET / B6 与 R-A 复算不等（O21）；
  - 脚本、检查点或表的版本混用；
  - 未定义格；
  - 某面板 / 种子没有任务；
  - 冒烟。

  任务内的地图 sha、私有图哈希、分量、冻结代码 sha、E4 指纹和 MATCH 不符时直接抛错，记为 failed，也落到 INCOMPLETE。E4 的 96 个 job 都在，0 个跳过，所以预期任务数为 96。
- **单测**：`test_verdict_d8_51` 覆盖了全部边界，包括 −1.0 与 3 / 24 两条线本身判为通过。`test_aggregate_marks_missing_and_failed_incomplete` 覆盖缺失与失败两种情形。

## 5. 独立复算：val_r2_i3_s1【逐数相等】

`indep_e6.py` 的做法：

- 自写 K，然后重新积分；
- 用 `E.direct_sums` 直接求和（实现走 `relay_sums`）；
- B4 自写为 Σ L > 0；G_c 用冻结匹配器对准自写的 B4 信念；
- NET 与 B6 在直接和上读出，再自写按智能体展开的均值。

| cam，R8 家具 F1 | noreg 原样 | e6 原样 | noreg 对齐 | **e6 对齐** | **Δ 对齐（pp）** |
|---|---|---|---|---|---|
| B4 | .72266 | .76249 | .73298 | .79569 | +6.27 |
| B6_A | .76767 | .79144 | .79329 | .83583 | +4.25 |
| NET_342101 | .81080 | .83051 | .87058 | .89028 | **+1.97** |
| NET_342202 | .81683 | .82736 | .87056 | .88972 | **+1.92** |

- **cam_nf 保护**（对齐行）：NET_342101 −0.018 pp，NET_342202 −0.134 pp；B4 +0.76 pp。
- **与冒烟 job 比较**：
  - 每档 60 个数（noreg / e6 × asis / aligned × 5 读出 × furn_f1、free_recall、unknown_known_fpr），全部差 0.0；
  - G：cam noreg (0.14542°, 7.69 cm, −1.81 cm)，e6 (0.11226°, 5.96 cm, −1.97 cm)，与实现逐位相同；cam_nf 也相同；
  - pub 地图 sha 与 E4 相同。
- **连续性**：
  - 本房 e4rule 对齐行 NET_342101 = .752（冒烟），即 −11.9 pp，与 R-A §2.2 逐房列表的 −11.9 一致；
  - 冒烟 2 房（只是冒烟，不是判决）的 Δ 为 +3.13 / +3.46 pp，主要来自一岛房 val_r1_i0_s1。本多岛房只有 +1.9 pp。这与 R-A 的预测（多岛房受岛间平移限制）一致。

## 6. 多进程、续跑、sha、失败记录、单测【通过】

- **多进程**：spawn 池，`maxtasksperchild=4`，worker 数 ≤ 12（L837、L806）。子进程初始化时设 `CUDA_VISIBLE_DEVICES=''` 和 `setup_env`，torch 设 1 个线程。拒绝 `python -O`（L371、L805），所以冻结代码里的 `assert`（例如 `relay_sums`）保持有效。
- **续跑与原子写**：已有记录、未失败、且脚本 sha 相同的任务跳过（L823）；失败的任务会重跑。写 job 文件先写 `.tmp` 再 `os.replace`（L503）。
- **sha 断言**：每个任务断言脚本 sha 与 3 个冻结模块的完整 sha（L372–374）；E4 job 的指纹与 MATCH（L384–387）；fixture 与 E4 一致；pub / k0 地图 sha（L424、L427）；O13（cam k0 = cam_nf pub）。
- **失败记录**：`_run_and_save` 把 traceback 末 2 KB 写入记录（L493–503），汇总时判 INCOMPLETE。
- **单测**：`python -B test_g4_island_compose.py`，9 / 9 通过，约 1 分钟，日志见 `review_e6/unittest.log`。其中 `test_readouts_reproduce_e3r_noreg` 在本地 inputs 存在时实际执行，没有跳过（cam NET_342101 .8722、cam_nf .9709）。
- **资源**：冒烟每任务约 750 s，2 进程并行，峰值约 1.2 GB / 进程；本审查 cam 档 2 臂（不含真值规范行）用时 118 s。

---

## 对协调者已接受的操作化的评价

- **O1（相信位置 = 本机 3 个位姿的相信公共帧扫描原点 xy 均值）**：忠实。这是 D8-51 "成员相信位置"最字面的写法，只用本机数据。本房实测与真位置差 1.5–12 cm，K 只差 1e-5 量级，NET 差 0.03 pp。
  - 枢轴的另外两种选法：
    - surfel 质心：E4 的分量枢轴用它，但它不是"位置"；
    - 每机各用自己的相信位置：单位先验下，若每机误差绕它自己的中心转，这才是精确的 MAP。
  - 每机枢轴与单一枢轴的差是 O(偏航 × 岛内展开)，约厘米级。D8-51 已写死单一枢轴，本批不改；留给以后（S5）。
- **O8（空闲召回保护与伤害房只在 VAL 4 格判，fresh 只判幅度）**：与起草史一致，但不是唯一读法。
  - 支持 O8 的一面：R-A §5.1 草案的保护项只对 VAL 4 格；D8-51 的披露句写的是 fresh "达到同样的**幅度**"。
  - 反对 O8 的一面：D8-51 字面上 "cam 档空闲召回" 与 "每格伤害房" 没有限定面板，而 cam_nf 保护项明确写了 "VAL 4 格"，且 "fresh 同样 4 格，共同判决"。
  - 判断：可接受，不列为必改。但 fresh 正是用来抵消 VAL 被看过的面板，逐房伤害恰恰是 VAL 被看过后最可能被掩盖的信号。见 S1。
- **O10（cam_nf 保护在对齐行判）**：忠实。D8-51 的判决行定义适用于全部判据。R-A 的 −0.48 pp 基线是原样行，但 cam_nf 上对齐 G 很小（本房 noreg G 为 −0.003°、−0.65 cm），两行相差很小；原样行也已进报告表。
- **O17 / O21（noreg 行与 E4 job、与 R-A 复算不一致即 INCOMPLETE）**：比预注册更严，方向正确，属于"流程自检"闸门。剩余风险是非实质的浮点差：R-A 复算用 torch 2 线程，本实现用 1 线程；有 2 个任务（val_r3_i2_s1 / s2）是 2 分量，批组成可能不同。已有证据：冒烟 2 / 2 相等，单测相等，本审查（1 线程）相等。风险低，但一旦触发应先能分辨（S2）。
- **其余 O 项**：
  - O2–O7、O9、O18–O20 与代码一致，无异议；
  - O11–O16 是只报告的读数，标注清楚；
  - O15 的主数取双向 (n_src + n_tgt) × 6 B，符合"交换"的字义，另报了单向数。

## 必改项

无。

## 建议项（不阻塞）

- **S1（O8，运行前确认一句）**：
  - `verdict()` 的输出里加入 fresh 4 格的保护读数：`free_delta_pp`、`n_harm`、`nf_delta_pp`。`cells` 里已经算了，只是没放进判决块。
  - 协调者在 D8-51 补充里写明 fresh 保护项"只报告、但在结果节显著列出"。
  - 如果协调者把"共同判决"读作包括保护项，就在运行前把 `free_guard` / `harm_le_3` 扩到 8 格。代码只改 `verdict` 里的两行，以及对应的单测。
- **S2（O17 / O21 的可诊断性）**：不等时，除任务名外再记录不等的分量数和 max |ΔF1|。这样能区分 1 体素的数值翻转和真错误。保持 INCOMPLETE 规则不变。
- **S3（钉住输入）**：
  - E4 job 文件的 sha 目前只记录、不断言，建议在运行前生成 96 个 job 的 sha 清单，在任务内断言；
  - 检查点与 E1c fit 只断言 8 位前缀，`threshold.json` 不断言，建议改成完整 sha。风险低：版本混用在汇总时会报出。
- **S4（只报告行，可选，约多 10 % 时间）**：在 cam 上加一行"接受边岛 + 岛内真相对位姿 + 同一相信位置规范"。现在的 `ceiling_islands` 用的是 E4 可观测岛，枢轴是 surfel 质心，所以 e6 − ceiling 同时混进了三处差别：岛划分、枢轴、边误差。新行能把岛内边估计误差单独量出来，正是 E5-R 的目标量。
- **S5（以后，不在本批）**：做 E3-L 的位姿限制映射或后续组合时，规范可以改成每机枢轴，即上面说的精确单位先验 MAP。
- **S6（运行操作）**：
  - `--out` 用新目录；
  - 机器上有其他批次时，按每进程约 1.2 GB 和当前负载定 worker 数，建议 ≤ 8，96 个任务约 36–41 k CPU 秒；
  - 续跑会覆盖 `run.json`，建议把控制台日志 tee 到 `e6_run.log`；
  - 进程被硬杀时 Pool 可能挂住。直接重启即可，续跑逻辑会跳过已完成的任务。
- **S7**：正式模式（无 `--jobs`）下加一条断言 `len(expected) == 96`，防止 E4 目录被意外改动后分母悄悄变小。

## 附：复算命令

```
cd claude_handoff/analysis_20261027/g4/review_e6
CUDA_VISIBLE_DEVICES= D:/anaconda3/envs/jax-gpu/python.exe -B indep_e6.py --tier cam      # 118 s
CUDA_VISIBLE_DEVICES= D:/anaconda3/envs/jax-gpu/python.exe -B indep_e6.py --tier cam_nf   # 114 s
CUDA_VISIBLE_DEVICES= D:/anaconda3/envs/jax-gpu/python.exe -B indep_e6.py --tier harsh --k-only   # 45 s
cd scripts/claude_v342 && CUDA_VISIBLE_DEVICES= D:/anaconda3/envs/jax-gpu/python.exe -B test_g4_island_compose.py
```
