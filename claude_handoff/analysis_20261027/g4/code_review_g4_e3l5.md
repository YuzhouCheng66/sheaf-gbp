# 代码审查：批次 E3-L5（带形状多样性的 TRAIN 分布；预注册 D8-76 及其补充）

- 审查者：独立审查 agent，2026-10-28 12:00–12:48 UTC（中途因流中断续审一次）。
- 结论：**LAUNCH OK**（建议用 two-leases 模式）。没有必改项。发射前的两件事属于流程，不是代码问题：协调者填好 FILL_ 值（见第 7 节）；用户确认使用两个 5090 租约。
- 审查的 sha：

| 文件 | sha 前缀 | 备注 |
|---|---|---|
| mapping_3d_storey_data_v420div.py | 3286d879 | 与任务单一致 |
| g4_e3l5_gen.py | 64168499 | 与任务单一致 |
| run_e3l5_gen.sh | 5d5abeac | 与任务单一致（已经跑完） |
| g4_e3l5.py | d055eca7 | 与任务单一致；uk-linux 上部署的版本也是 d055eca7 |
| test_g4_e3l5.py | **ac66458a** | 已不是任务单里的 ee800dc2，加了 two-leases 的单测 |
| chain_day8_E3L5.sh | **287271c9** | two-leases 的定稿版（协调者 12:4x 通知），uk-linux 上部署的版本 sha 相同 |
| 冻结依赖 | g4_e3l3 c7523cc8、v420ood 9e1c7c2a、g4_ood_panel 3a6959da、g4_h1_e3l3 dc0f69d7 | 本机核对过，都没改动 |

- 审查者自己写的脚本与输出都在 `claude_handoff/analysis_20261027/g4/review_e3l5/`，没有写 .npz / .pt：
  - `indep_h1_y.py` 与 `indep_h1_y.out`：独立复算 H1 的 Y 区；
  - `uk_cache_check.py`：在 uk-linux 上只读核对缓存。uk-linux 上的副本与输出在 DIR/review_e3l5/。
- 没碰过任何 VAL 读数：没打开 H1 VAL 面板，也没打开 G4-E5 VAL 种子 3、4 的读数。对这些文件只算过 sha 与内容 sha。G4-E7 的 summary.json 只读了键名，用来核对闸门代码。

---

## 1. 单变量（检查 1）：通过

**训练参数**：拿 E3-L5 链交给训练器的旗标，与 E3-L3（chain_bigboy_E3L3.sh:200/202）和 E3-L4（chain_day8_E3L4.sh:159）逐项比较。正式训练的调用在 chain_day8_E3L5.sh:255（two-leases）和 :313（单租约）。

| 旗标 | E3-L3 | E3-L5 |
|---|---|---|
| --cache / --manifest-sha | cache-e3r-v1 / 14e90b40… | **cache-e3l5-div-v1 / 80b8cdc0…**（单变量） |
| --ctrl-holdout | e3l2-ref/ctrl_holdout.json | **e3l5-ref/ctrl_holdout_e3l5.json**（配套） |
| --evidence | c:e1c_fit.json:A / B:cam_nf | 相同 |
| --updates | 9,562（计划数） | 9,562（two-leases）；放不下时按 E3-L3 规则缩，并记为第二个差异 |
| --log-every 200、--time-from 200、--probe-every 1000、--snap 6000、--reserve 60、--ctrl-ckpt E3R/s{seed} | 有 | 相同 |
| --batch、--width、--lr、--wd、--warmup、--clip、--holdout-samples | 都用默认值 8 / 16 / 1e-3 / 1e-4 / 300 / 1.0 / 32 | 都用默认值，相同 |
| --expect-gpu、--deadline-epoch | RTX 6000 Ada，按预算 | RTX 5090，按预算（硬件不同，与 E3-L4 一样） |

- **包装层**：g4_e3l5.py:208–237 的 `train` 只读不吞地解析转发参数（`_sub_args` 用 parse_known_args）。它先做缓存前置检查，然后调 `L3.main(['train'] + rest)`，把参数原样交出去。只有 `--prereg-updates` 和 `--second-note` 是包装层自己消耗的。
  - 模型类、w0、w 可学（log_w 的 requires_grad）、优化器、日程、探针、阈值规则，都是冻结的 g4_e3l3 代码。
  - 单测 `test_train_forwards_arguments_unchanged` 覆盖了这一点。
- **阈值**（g4_e3l5.py:265–310）：
  - 学到的 w：原样调 `L3.main(['select'])`；
  - w = 4：`select_w` 逐行复制 L3.select（L3:476–509），只多了一处：在工作进程里把 `L3.load_model` 换成载入后令 `log_w ← log 4` 的包装，并断言 float32 下 exp 恰好等于 4.0。规则仍是 `L3.pick_r8_k0`（空闲召回 ≥ 对照 − 0.25 pp、家具 F1 最大、同分取最小）。
  - 两份阈值都只在新缓存的 19 个留出房上选（F2）。
- **只报告的细节**：
  - E3-L3 被 GPU 共用截断在 9,411 / 9,359 步，E3-L5 预计完整跑 9,562 步，多出 1.6–2.2 %；
  - 硬件从 6000 Ada 换成 5090。

  这两点与 E3-L4 的先例相同，不算第二个变量，结果里写一句即可（S6）。

## 2. 数据（检查 2）：通过

在 uk-linux 上只读核对（`uk_cache_check.py`，CPU，1 个进程，nice 10）：
- **manifest**：sha = `80b8cdc0bb6397ca1d2054e14d1f52fd1addf92b68c12c2e92aef56b52796434`，与预注册补充一致。
  - script_sha 64168499，data_module 3286d879；
  - 种子 [200, 201, 202]，p_clean 0.1。
- **房子集合**：205 房 = 186 训练 / 19 留出，skipped 与 failed 都为空。
  - 与 cache-e3r-v1 的 manifest（14e90b40…）房子集合相同，留出标记逐房相同；留出房都满足 index % 10 == 9。
- **分配**：按 world_id 前缀，train_div 100 房（训练 92 / 留出 8），v420 105 房（94 / 11）。
  - v420 那一半的 world_id 与 cache-e3r-v1 逐房相同；
  - verify_orig.json 的 ok 为 True：105 房逐数组相同，100 房 slab 相同、obj 不同。
- **Y 计数**：
  - 205 个 div_records 合计：Y 物体 0、体素 0、图元 0；train_div 全部在第 0 次尝试生成。
  - train_div 的类别表里没有 ball、ellipsoid、hanging、clutter、coat_rack；rot_table 有 155 件，它们的桌面杂物为 0。
- **家族总计**：与 D8-76 补充的表逐数相同。

| 家族 | 物体 | 体素 |
|---|---|---|
| rotated | 211 | 70,455 |
| scaled_0.6 | 1,918 | 144,980 |
| scaled_1.5 | 1,167 | 857,332 |
| wall_shelf | 1,346 | 104,413 |
| v420 orig | 4,524 | 1,647,450 |

- **3 房独立重算**：用 `D.world_spec` 重新生成世界，由审查者自己的代码按类别、尺度、图元归家族。

| 房 | 家族（物体数） | 图元 | Y / 异常 | npz 的 obj 与重新栅格化 | slab 与 e3r | 层板最低 z |
|---|---|---|---|---|---|---|
| TRAIN_r1_i0 | scaled_1.5 7、scaled_0.6 8、rotated 1 | box / cyl / obox | 无 | 相同 | 相同 | 无层板 |
| TRAIN_r2_i37 | scaled_1.5 17、scaled_0.6 37、wall_shelf 20、rotated 3 | box / cyl / obox | 无 | 相同 | 相同 | 1.015 m |
| TRAIN_r1_i9（留出） | wall_shelf 11、scaled_0.6 10、scaled_1.5 11、rotated 1 | box / cyl / obox | 无 | 相同 | 相同 | 1.115 m |

  三房都没有 'ell' 图元，也没有 X 以外的家族。
- **代码层面**：
  - v420div.py:198–208 每个槽位只在 X 的几种模式里抽一种；
  - rot_table 的桌面杂物照常消耗随机数，但在 :208 被丢掉（`payload = dict(payload, top=[])`），:222 断言没有附带物体；
  - 出口处 `check_div_objects`（:140–145）断言 kind、family、图元都不属于 Y；`family_stats` 再断言一次（:356–359）。

## 3. 判据（检查 3）：通过

`g4_e3l5.py` 的 `D8_76`（:89–106）与 `judge`（:968–1020），逐条对照 D8-76 判据与补充 F1–F5：

| 判据 | 实现（行） | 边界 | 结论 |
|---|---|---|---|
| 1 翻转 ≥ .50 | `fp >= 0.50 − 1e-12`，fp = Y 区内计数 [2]/([2]+[3])（:862、:983） | 分母为 0 记通过（同 D8-53 P3） | 对 |
| 1 Y 区 Δ ≥ −1 pp | `d_zone_Y >= −0.010 − eps`，对自己配对的 B6（:984） | 为 None 记不过 | 对 |
| 1 幻影块 ≤ B6 + 2 | `ph <= ph_B6 + 2 + eps`；整房计，取 48 张图的每房均值（F4） | 等号算过（D8-53 的"有害"是 > B6 + 2） | 对 |
| 2 翻转 ≥ E3L3W4 + .05 | :988–989 | 臂为 None 记过；比较对象为 None 而臂不为 None 记不过（F4 写明） | 对，偏保守 |
| 3 家具 ≥ E3L3W4 − 0.5 pp | `d_furn >= −0.005 − eps` | | 对 |
| 3 保护项（D8-63 原文，对 NET） | 空闲 ≥ −0.5 pp；表面 ≥ −0.5 pp；\|ΔFPR\| ≤ 1e-12；clean 家具 ≥ .99；K0_sep ≥ NET K0_sep − 3 pp（:990–995） | K0 口径与 E3-L3 的 g4_e3l3:699/710 相同 | 对 |
| 4 字节 ≤ 0.15 倍 | 臂每房 O7 中位 ÷（NET sum 中位 + count 中位）（:925–927） | 与 G4-E7 的 g4_e3l3_w:755–759、E3-L3 的 g4_e3l3:690 同一口径 | 对 |
| 判决 | 判据 1 任一格不过 → NOT_ENOUGH；1–4 全过且没截断 → LEVER；截断只挡 LEVER → AMBIGUOUS；其余 AMBIGUOUS（:1012–1018） | | 与 D8-76、D8-63 一致 |
| INCOMPLETE | 下列任一情况（:1066–1101、:1022–1047） | | 齐全 |

INCOMPLETE 的触发条件：
- 缺格、缺房、缺任务，或任务失败；
- H1 对照闸门不等；
- 已知区不一致；
- GaBP 核对失败，或某臂缺 GaBP 核对；
- G4-E7 闸门差 > 1e-3；
- < 2,000 步，或两种子更新数不同；
- 混用脚本或检查点；
- 计划步数 ≠ 9,562 又没写第二个差异；
- 缺 threshold_w4.json。

- **判决只读 w = 4**：`arm_key='arm'` 对应 E3L5W4。学到的 w 只进 `learned_w_would_be`，标明只报告（:1104–1106）。
- **配对（F5）**：H1 种子 1、2 与 VAL 种子 3、4 按序号配对，每条判据在它自己的 4 格上全过才算过，配对不影响结论。
- **G4-E7 闸门**：键名 `E3L3W_{ts}`、`NET_{ts}`、`bytes_median` 与 e7/summary.json 的实际结构相符（只看了键名）。e3l5-ref/e7_summary.json 的 sha c45eaca7 与本机 g4/e7/summary.json 相同。

## 4. H1 的 Y 区计分（检查 4）：通过，逐数一致

- **Y 区的定义**：Y 区 = 家族属于 {ellipsoid, hanging, clutter, pole} 的 `OP.object_zones` 家族区之并（g4_e3l5.py:637–639）。
  - 每个家族区是 dil6（6 邻域膨胀 1 格）。膨胀对并集可分配，所以它等于"Y 物体并集膨胀 1 格"。
  - 审查者直接对并集做膨胀，得到的体素数与记录相同：2,706 / 5,030。
- **翻转、幻影块、F1**：与 D8-53 / g4_ood_panel 的定义一致。
  - 翻转用 `OP.flip_counts`：臂判占据、B6 判空闲、S ≤ 0、在计分范围内；取 'le_-0' 在区内的计数 [2]、[3]。
  - 幻影块用 `OP.phantom_blocks`：26 连通、3×3×3 内没有任何生成器占据；臂每机权重 1/n，对照每分量权重 |分量|/n，与 H1L 相同。
- **比较对象 E3L3W4**：用冻结的检查点（arm_specs 断言 sha：1fa0cb39 / c583fa1e、3d332d40 / 56f4bf7f），w = 4，阈值取 G4-E7 criteria_w.json 的 w4 那一份；在同一个 `h1_file` 里、同一批图上与臂一起重算。
  - eval 一侧在同一批 G4-E5 地图上重算，并由 G4-E7 闸门核对。
- **独立复算**：`review_e3l5/indep_h1_y.py`。不 import g4_e3l5，也不 import g4_ood_panel / g4_h1_e3l3 的计分函数；只用冻结的 encode / gabp / decode 和 G.Target。只在 TRAIN H1 冒烟图（噪声种子 203）上算。

| 图 / 臂 | Y 区 tp / fp / fn | Y 区翻转（真 / 假） | 全范围翻转 | 每房幻影块 | 冒烟记录 |
|---|---|---|---|---|---|
| novel_train_r2_i9 · E3L3W4_342101（j = 1986） | 4828 / 504 / 796 | 53 / 281 | 365 / 400 | 0.25 | 逐数相同 |
| 同图 · B6_A | 5048 / 336 / 576 | — | — | 1.0 | 逐数相同 |
| novel · E3L3W4_342202（j = 1992） | 4912 / 531 / 712 | 55 / 318 | 387 / 554 | 0.0 | 逐数相同 |
| novel · B6_B | 5064 / 352 / 560 | — | — | 1.0 | 逐数相同 |
| floating_train_r2_i9 · E3L3W4_342101 | 8072 / 419 / 992 | 48 / 196 | 130 / 279 | 0.0 | 逐数相同 |
| floating · E3L3W4_342202 | 8315 / 356 / 749 | 54 / 163 | 134 / 257 | 1.0 | 逐数相同 |
| floating · B6_A / B6_B | 8576 / 432 / 488 | — | — | 0.0 | 逐数相同 |

- **合并两图（= 冒烟的 h1_cells）**，复算与 agg/summary.json 相同：
  - 342101：Y 区翻转精确率 101 / 578 = **.17474**；Y 区 F1 .90491，对 B6 .93700；幻影块 0.125，对 B6 0.5。
  - 342202：109 / 590 = **.18475**。
  - 这正是 D8-76 补充里披露的 .17 / .18。
- **已知区**：两图上已知区与 k > 0 不一致为 0。

## 5. 链的安全（检查 5）：审 chain_day8_E3L5.sh **287271c9**，通过

**基本的安全规则**
- **sha 钉住**：`check_shas`（:131–167）在这些时点调用：锁之前、SOLO 等待之后、每个租约的等待前后、单租约的每个种子之前、CPU 阶段之前。
  - 覆盖全部脚本、冻结依赖、pyroot、缓存的 manifest / e3l5_div / verify_orig、ctrl_holdout、E3-R / E3-L3 的检查点与阈值、G4-E7 的判据与 summary、G4-E5 导出、H1 VAL 的记录与 npz。
  - 审查者在 uk-linux 上预先核对了这些钉住的值，全部相符：h1_jobs 42899de8、h1_export 7cc07bd1、e5 records 7554d1fd、pyroot de96be65、export_run fbf67c87、E3-R 阈值与检查点、E3-L3 四个文件、criteria 320317aa、e7_summary c45eaca7。
- **拒绝 FILL_**：:127 检查 5 个 FILL_。APPROVAL 参数必填（:91）；第三参数只接受空或 two-leases（:93–94）。
- **锁与不覆盖**：
  - 先 `mkdir $OUT.lock`，并设 trap EXIT（:169–170）；`$OUT` 已存在就中止（:171）。
  - two-leases 下，lease_X / updates_X / lease_end_X / pf-s / s 中任一已存在就中止（:186–188）。函数里的 `exit` 会退出整个脚本，trap 照样执行。
  - g4_e3l5 自己也不覆盖：e3l5.json、threshold_w4.json、jobs/、summary.json 都会检查。
- **SOLO 等待**：每个租约各有一次 `busy`，看两遍、间隔 60 s（:191–193）。
  - busy 检查：其他 chain_ 链、train_ / diag_ / owned_ceiling、run_e3l5_gen，以及 nvidia-smi 上的任何计算进程。
  - 只等，不抢占。
- **结束进程**：只用 `timeout -k 30` 结束自己的子 PID。preflight（600 s）、训练（SDT − NOW）和各 CPU 阶段各有一个。全文没有 pkill、killall 或 kill。
- **CPU 阶段**：selfcheck、select、eval、h1、aggregate 都带 `CUDA_VISIBLE_DEVICES=`、`nice -n 10`，9 个工作进程 + 1 个父进程，按顺序执行（:176、:329、:340、:344、:347）。
- **只写 /mnt/datasets**：cd 到 DIR，OUT 是 runs/e3l5，LOG 是 runs/chain_day8.log，TMPDIR 是 tmp-claude；导出和参照只读。
  - 12:43 UTC 审查时，uk-linux 上 runs/e3l5 与锁都不存在，GPU 0 %、357 MiB。

**租约时间的复算**（two-leases，E3-L4 在同一块 5090 上的实测：0.35259 s/步，load 6.26 s，first window 26.14 s，probe 5.13 s）：
- base = 6000 − 60 − 6.26 − 26.14 − 30 = 5,877.6 s；
- u0 = min(⌊0.9 × 5877.6 / 0.35259⌋, 9562) = 9,562；
- avail = 5877.6 − 5.13 × 10 = 5,826.3 s；
- fit = ⌊0.9 × 5826.3 / 0.35259⌋ = **14,871 ≥ 9,562**，所以 UPD = 9,562，**不会出现第二个差异**。
- 训练约 9,562 × 0.357 ≈ 3,414 s。加上 preflight 约 200–250 s，每个租约约 3,650 s，在 7,200 s 内。
- 杀进程时刻 = NOW + 6,000（不晚于 START + 7,150）；训练器截止 = SDT − 30（不晚于 START + 6,600）。只有每步慢到 0.615 s（约 1.74 倍）才会截断。

**two-leases 下的步数与第二个差异**
- 租约 A：UPD_A = min(fit_A, 9,562)。不足 9,562 时写出第二个差异：日志、updates_A.json，经 `--second-note` 进 e3l5.json，再进 final.json 的 e3l5 块。
- 租约 B：从 updates_A.json 读回 UPD_A。
  - fit_B ≥ UPD_A 时用 UPD_A，并沿用 A 的注记；
  - 否则用 fit_B，注记写明并判 INCOMPLETE（由 aggregate 的"两种子更新数不同"保证，:1042）；
  - < 2,000 时不训练，直接 INCOMPLETE。
- **两个种子用同一更新数**：成立，否则就是 INCOMPLETE。
- 3 种情形的 bash 空转单测（假训练器）都通过：9562 / 9562；B 放不下 → INCOMPLETE；A 低于 9,562 → B 沿用 A 的数。
- 注记里的撇号（"lease A's"）在 Python 三引号里不出错，单测实际跑过这个分支。
- 两个租约都结束后，才进入 CPU 阶段。

**一处无法核对**：实现者说不带第三参数时与 3020fbf6 相比只有新增行。旧文件已被覆盖，我没有副本，这一点没有逐行核对。单租约分支本身已按上面的规则审过：结构同 E3-L4，每种子 3,000 s，预计 7,240 步，属于第二个差异。

## 6. 单测（检查 6）：通过

- 本机 CPU，1 个进程，`CUDA_VISIBLE_DEVICES=`：`pytest -q -p no:cacheprovider scripts/claude_v342/test_g4_e3l5.py`。
- 结果 **22 passed in 21.5 s**（12:43:50–12:44:15 UTC），测试时文件 sha 是 test ac66458a、chain 287271c9。

## 7. 协调者要填的 FILL_ 值（审查者在 uk-linux 上实测）

| 变量 | 值 |
|---|---|
| E3L5_SHA | d055eca77b799ad210504fc51a79ae12a11faba9291bd44023bdc1502b543e2a（部署版本与本机相同） |
| MAN5_SHA | 80b8cdc0bb6397ca1d2054e14d1f52fd1addf92b68c12c2e92aef56b52796434 |
| DIV5_SHA | 7043a5aa2a95d8bc0dfc4ae6391e0959151d374476a9de35ba708e5a85d19da6 |
| VER5_SHA | 2bd3d8da2c81852ec7da1b35727de7fbf634eaaa3a6b21df53e0951de625ae2c |
| CTRLH5_SHA | 2f7fbe1d3b4843427b13c8fa76fdb854d657bdec2a9f089ea724b37977b7f8ef |

填完后链的 sha 会变，请按新 sha 记录。

## 必改项（M）

无。

## 建议项（S）

- **S1 SOLO 等待没有上限**（:181、:192）。two-leases 下，租约 B 可能无限期等 5090，而锁和 CPU 阶段也会跟着一起等。
  - 建议协调者盯住 chain_day8.log；或者仿照 E3-L3 的 M1，设一个最长等待（例如 120 min），超时后照 INCOMPLETE 处理。
  - 不改也可以发射。
- **S2 preflight 与正式训练之间没有 GPU 空闲复查**。单租约分支的 run_seed 有 30 s 的轮询（:308），lease2 没有。两者之间只隔几秒，风险很小。建议补一条"GPU 上的其他进程"日志。
- **S3 进程数**：spawn 进程池另有一个空闲的 multiprocessing resource_tracker 进程，所以 python 进程实际是 9 + 1 + 1 个，多出的那个不占 CPU。以往批次也是这样。
- **S4** v420div 的 `check_div_objects` 里，`placement != 'floor'` 这一条是空检查：placement 由本模块自己写成 'floor'。真正起作用的是 kind、family、'ell' 图元、附带物体这几条断言，以及 family_stats。这只关系到文档的准确，不影响结果。
- **S5** 家族体素表按"每物体掩码扣楼板"求和，重叠体素会被重复计。v420 一半是 1,647,450，栅格是 1,647,347，差 103。只是报告上的区别。
- **S6 写结果时请注明两点**（不算第二个变量）：
  - E3-L3 被截断在 9,411 / 9,359 步，本批应完整跑 9,562 步；
  - 硬件从 RTX 6000 Ada 换成 5090。
- **S7** Y 区膨胀 1 格后，会盖到相邻 X 物体的体素。例如 novel 里 rot_table 上的杂物，膨胀后盖到桌面。这是"膨胀 1 格"定义本身带来的，H1 的口径也是这样。实现已按家族区拆分报告（`f:` 区）。读判据 1 的 F1 时请参考这些拆分。
- **S8** 判据 2 里"比较对象分母为 0 而臂不为 0 记不过"，是实现写定的保守口径（F4）。D8-76 原文没写这种情形，建议在发射记录里补一句。现在 E3L3W4 在 TRAIN 冒烟上的 Y 区翻转有 334–373 个体素，VAL 上分母为 0 的可能性很小。
- **S9** two-leases 需要用户确认**两个** 5090 租约。链只记录 APPROVAL 字符串，核实不了确认的内容；协调者要在发射记录里写明用户原话。
