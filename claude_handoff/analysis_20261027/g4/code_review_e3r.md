# 独立代码审查：批次 E3-R（读出端学习先验，5090，1 个租约）

审查者与实现者无关，2026-10-27 08:30–09:15 UTC。被审对象如下：

- `scripts/claude_v342/g4_e3r.py`（sha 17e04e6b…，与远端部署一致）
- `scripts/claude_v342/test_g4_e3r.py`（967918d3…）
- `scripts/claude_v342/chain_day7_E3R.sh`（45d0b5da…，还没部署到远端）

对照的预注册以 D8-44 原文为准，背景是 D8-41、D8-42（含补充）、D8-43，以及 `review_E1/RC_sheaf_plan.md` §2–§3。审查期间没有修改被审文件。临时脚本放在 `g4/review_e3r/`（只有 3 个探针 .py，不含数据）。

## 结论：**LAUNCH OK WITH CHANGES**

底线、数据隔离、训练实现和租约框架都对。必须先改两处：

- 判决代码有两处与 D8-44 字面不符：K0 保护项的比较对象错了；截断、缺 fresh、缺对照臂会被静默处理。
- 链式脚本没设 `PYTHONPATH`。不改的话，预跑在租约开始后就会 ImportError，白白耗掉 1 个租约。

M1–M5 改完、单测补齐，再加上发射前清单（E1c 正式结果、`e1_jobs` 等文件部署、FILL_* 钉死），就可以发射。

## 必改项

### M1　链式脚本没有设置 PYTHONPATH，也没有导入自检（`chain_day7_E3R.sh:13–14、51、60`）

**问题。** 远端的 `mapping_houseexpo_data_v365.py:30` 需要 `from scripts import mapping_multiroom_data_v293`，`occupancy_codec` 也在 `e3r-pyroot/exploration_grid` 下。远端日志 `runs/e3r_test.ABORTED-importpath*.out` 记录了两种失败：

- `No module named 'scripts'`；
- `scripts` 被解析到 `/home/yuzhou/Desktop/sheaf-gbp/scratch-scale-pair-v311-stage-20261018/scripts`，那是一个带 `__init__.py` 的常规包。

当前的 gen 进程 `/proc/998238/environ` 里显式带着 `PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid`，所以能跑。链式脚本里没有这一行，以 `setsid nohup` 启动后，每一步（预跑、训练、导出、评估）在 `import g4_noise_panels` 时都会失败。后果是：

- 预跑在 `START` 之后崩溃，`RATE=nan`，`UPD=0`，链判 INCOMPLETE 退出；
- 3 个租约里的 1 个就此作废。

还有一个隐患：`e3r-pyroot/scripts` 没有 `__init__.py`，是命名空间包。sys.path 上任何地方只要出现一个常规的 `scripts` 包，它就会优先被导入，与顺序无关。我查过：当前 site-packages 的 .pth 和几个已知目录里都没有这样的包，镜像中的 `mapping_multiroom_data_v293.py`、`prepare_houseexpo_pilot_v121.py` 与本机 sha 一致。

**改法。**

1. 第 13 行后加
   `export PYTHONPATH=/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot:/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/exploration_grid`。

2. 在 `check_shas` 里钉住镜像：
   `PYR_SHA=$(cd /mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot && find scripts exploration_grid -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-64)`，
   与填写的 `FILL_pyroot_sha` 比较。

3. 第一次 `check_shas` 之后、`busy` 等待之前（也就是**租约开始前**）加一道导入自检，失败就 `exit 2`：

   ```bash
   CUDA_VISIBLE_DEVICES= timeout -k 30 120 $P -B -c "import sys; sys.path.insert(0,'scripts'); import g4_e3r, scripts.mapping_multiroom_data_v293 as m, occupancy_codec as oc; \
   R='/mnt/datasets/yuzhou-sheaf-gbp-research/e3r-pyroot/'; assert m.__file__.startswith(R) and oc.__path__[0].startswith(R), (m.__file__, oc.__path__)" \
     || { log "ABORT: import self-check failed (PYTHONPATH / scripts package)"; exit 2; }
   ```

4. 把 PYTHONPATH 和 PYR_SHA 写进 lease.json。

### M2　K0 保护项与 D8-44 不符（`g4_e3r.py:627–667、875–887、1060–1064、1098`）

**D8-44 原文：** "K0 家具（网络与线性读出各用按同一规则在留出房上为 K0 选的阈值）不低于线性读出 − 1 pp"。

**现状。**

- 只有网络有 K0 专用阈值（`k0.j_sep`）。
- 线性读出的 K0 固定用默认的 θ（`e > ev.theta`，第 655、881 行），没有为 K0 选阈值。
- 判决里的 `d_K0_furn` 是"NET 的 K0 减去**对照臂 C**（B4/B6，R8 默认 θ）的 K0"（第 1062 行），不是减去"同一证据函数的线性读出（LIN_ts）、用 K0 专用阈值"。

探针 `review_e3r/probe_verdict.py` 的第 6 例：NET 的 K0_sep 为 .68，LIN 的 K0 为 .70，对照 B6_A 的 K0 为 .66。按 D8-44，−2 pp 低于 −1 pp 的下限，不应判杠杆；代码却得到 `d_K0 = +0.02`，判 **LEVER**。

**改法。**

1. 在 `select_threshold` 的 K0 循环里，对单机 e 的整数值做三类直方图（家具真值、非家具、空闲）。用同一个 `_select` 规则选 θ_K0：谓词 `e > θ`，约束是 K0 合并空闲召回不低于 `e > ev.theta` 的召回，取家具 F1 最大，同分取最小 θ。候选只需扫表 `ev.table` 中出现过的值。结果写进 `threshold.json['k0']['linear_theta_sep']`。
2. `eval_job` 为 `LIN_{sd}` 增加 `K0_sep` 行，用的是 `linear_readout(e, k, θ_K0)`。
3. `verdict` 的 K0 判据改为 `NET_ts.K0_sep.furn_f1 − LIN_ts.K0_sep.furn_f1 ≥ −0.01`。另外两行只报告：`d_K0_furn_shared` 和"对 C 的 K0 差"。
4. 单测：在合成数据上验证 θ_K0 等于暴力扫描的结果，并让探针第 6 例判 AMBIGUOUS。

### M3　判决在截断、缺格、缺对照臂时与 D8-44 不符（`g4_e3r.py:1021–1034、1040–1047、1054–1058、1070–1072、1080–1090`）

1. **截断。** D8-44 把"训练未截断"列为**杠杆**的条件，只有"少于 2,000 步"才判 INCOMPLETE。现在代码里任何 `complete=False` 都判 INCOMPLETE（第 1026 行）。探针第 5 例：种子 2 截断在 5,000/6,000 步，VAL 差值为 0，D8-44 下应判 NOT_ENOUGH，代码给出 INCOMPLETE。
   改法：判 INCOMPLETE 的情形为 `updates_done < 2000`、两种子 `updates_planned` 不同、缺 `threshold.json`、缺种子；`truncated` 只禁止 LEVER，NOT_ENOUGH 和 AMBIGUOUS 照常判。

2. **fresh 缺失。** fresh 面板整个缺失或缺格时，现在会静默跳过反号检查（第 1080–1090 行；探针第 1、3 例都给出 LEVER）。D8-44 写的是"fresh … 4 格均值的符号与 VAL 相反时降为 ambiguous"，任务口径是"缺格 INCOMPLETE"，E1c 的先例（D8-42 补充 3）也是"fresh 任何一格缺失都判 INCOMPLETE"。
   改法：fresh 的 4 格都要存在，每个种子的房子数也要齐（与 VAL 同一检查，第 1040–1047 行改成对 `val` 和 `fresh` 都做），否则 INCOMPLETE。

3. **对照臂缺失。** 第 1055 行 `ctrl_arms = [a for a in ctrl_arms if a in tier]` 会静默丢掉缺失的 B6 臂，于是 C 退化为 B4，NET − C 被高估。
   改法：criteria 里写死 `control_arms = ["B4","B6_A","B6_B"]`，缺任何一个都判 INCOMPLETE。fresh 一侧（第 1088 行）同样处理。

4. **clean 档缺失。** clean 档的 NET 行缺失时，现在只是保护项不过（第 1070–1071 行），应改判 INCOMPLETE。

5. 以上四条都写进 `test_verdict_interface`，覆盖探针第 1、3、5、6 例。

### M4　证据函数的选择规则没有钉进链（`chain_day7_E3R.sh:21–27、37、47、63–67`；`g4_e3r.py:131–137`）

**现状。** `EV_342101` 和 `EV_342202` 由协调者手填，链不检查它们是否符合 D8-44 的规则：E1c 判 LEVER 或 AMBIGUOUS 时用 B6，A 半集给 342101，B 半集给 342202；判 NOT_ENOUGH 或 INCOMPLETE 时用 L。链也不检查 fit.json 是否就是 E1c 正式判决所用的那一份。另外，`Evidence.from_spec` 只记录 `fit['complete']`，不拒绝不完整的拟合。

**改法。**

1. 把 E1c 正式的 `summary.json` 部署为 `$REF/e1c_summary.json`，在链里加 `E1C_SUM_SHA=FILL_…` 并放进 `check_shas`。
2. 第一次 `check_shas` 之后，用一小段 python 机械判定：
   - 读 `S['verdict']['verdict']`；
   - 如果 `startswith(('LEVER','AMBIGUOUS'))`，期望值是 `c:$REF/e1c_fit.json:A:cam_nf` 和 `c:$REF/e1c_fit.json:B:cam_nf`；否则两者都是 `L`；
   - 断言期望值与 `EV_*` 一致；
   - 断言 `S['fit_sha256'] == [FIT_SHA]` 且 `S['fit_sha256_file'] == FIT_SHA`；
   - 任何一条不符就 `exit 2`。
3. 把 E1c 的 verdict 字符串和 summary sha 写进 lease.json。
4. `Evidence.from_spec`：`fit.get('complete') is not True` 时拒绝，除非带 `--smoke`。
5. 如果 E1c 最终是 INCOMPLETE 且 fit 不可用，D8-44 没有定义此时 C 怎么取（它要求 B6_A、B6_B 用 E1c 的正式表）。这种情况下按 M3.3 会判 INCOMPLETE，应**暂缓发射**，由协调者另行决定。

这一条做完之后，选择规则在发射前就能确定，并且由 sha 钉住。

### M5　"不覆盖历史"没有覆盖 `$EXPORT`（`chain_day7_E3R.sh:17、54、102、105`）

**问题。** 导出写到 `/mnt/datasets/yuzhou-sheaf-gbp-research/cache-e3r-v1-export`（records 和 npz）。`eval` 会读取 `$EXPORT/records/*.json` 里的**全部**记录（`g4_e3r.py:1225`）。如果这个目录已经存在（比如重发或手工冒烟），旧记录会混进来，npz 也会被覆盖。第 54 行只检查 `$OUT` 的历史。

**改法。** 把 `[ -e $EXPORT ]` 加进第 54 行的中止条件。也可以干脆把导出放到 `$OUT/export-cache`，与 `$OUT` 一起受历史检查保护。

## 建议项

- **S1　cam 档的跨平台复现没有验证。** 我把远端 Linux 生成的 TRAIN 缓存（`TRAIN_r1_i0`、`TRAIN_r1_i9`，含 clean、cam_nf_s200、cam_nf_s201）与本机 Windows 冒烟缓存逐数组比较，结果**逐位相同**（`review_e3r/cmp_platform.py`）。两边版本不同：本机 numpy 2.2.6、scipy 1.15.3，远端 numpy 2.4.6、scipy 1.16.3。所以 cam_nf 与 clean 复现 E1 的风险低。
  cam 档多一步机间坐标变换（`to_public`），这一步没有跨平台验证过。而现在任何一档与 E1 不一致都会使整批判 INCOMPLETE（`g4_e3r.py:964`）。两种处理任选其一：
  - 发射前在远端 CPU 上跑一次 `export --split TRAIN --smoke --train-houses 1:0 --seeds 203`（约 1 分钟），与本机冒烟导出 `train_r1_i0_s203.npz` 逐数组比较；
  - 把 cam 档的不一致降为"只报告"，判决只要求 cam_nf 与 clean 复现。D8-44 的断言写在"VAL cam_nf 判决"之下，这样改仍在原文范围内。

- **S2　GPU 选阈值与 CPU 计分在数值上不逐位一致。** 两者的算法一致：分箱、掩码、特征、合并 F1 与空闲召回都由 `test_threshold_selection_matches_score` 在 CPU 上逐值验证过。但远端 `torch.backends.cudnn.allow_tf32 = True`（实测，torch 2.11.0+cu128，cuDNN 9.19），选阈值时卷积走 TF32，logit 的误差量级约 1e-3。CPU 上 fp32 与 fp64 的差只有约 1e-5（实测）。分箱宽 0.02，翻转的体素极少，对家具 F1 的影响远小于 0.1 pp，不影响判决，但文档里"selection and read-out agree exactly"只在同一设备上成立。
  改法任选：
  - 在 `select_threshold` 里临时关掉 `cudnn.allow_tf32` 和 `cuda.matmul.allow_tf32`；
  - 租约后在 CPU 上重选一次。成本约 513 次 CPU 前向（单线程每次约 1.2 s，开多线程 2–10 分钟），结果写到 `threshold_cpu.json` 供 eval 使用，也更贴合"推理在租约外用 CPU"。
  两种做法都应把设备记进 `threshold.json`。

- **S3　步数余量偏薄。** `UPD = 0.95 × avail / rate`，粗算每个种子的余量约 190 s，约合 7 %。按 M3，截断会挡住"杠杆"；5090 长时间运行降频就可能吃掉这点余量。建议改为 0.90。我在本机实测选阈值的 CPU 侧开销约 0.3 s / 房·抽样，留出约 19 房 × 3 次抽样约 20–30 s，所以 `--reserve 300` 可以降到 120，把时间让给训练。

- **S4　TRAIN 房子的确定性跳过。** 远端 gen 已处理 75 / 210 房，其中 5 房因放置失败被跳过（r1_i26、i29、i46、i49、i66），i29 和 i49 属于留出集。所以留出房会少于 21，训练房会少于 189。manifest 里已有 `n_train`、`n_holdout`、`skipped`。建议像 D8-36 补充那样，在运行前的口径里写明"放置失败的房子确定性跳过，实际数写进判决"，并让 summary 带上这两个数。

- **S5　FPR 应写成断言，而不是 0.3 pp 的保护项。** D8-44 说"未知→已知 FPR 与基线相同（断言）"。`G.score` 的 `unknown_known_fpr` 只取决于已知掩码，而掩码已被断言逐位相同。建议在 verdict 里断言 `fpr_delta == 0`，或在 criteria 里写 `guard_fpr = 0`。D8-44 没有列 +0.3 pp 这条保护，不宜带进来。

- **S6　早退时也写一份可机读的判决。** 链在第 81 行（UPD < 2000）和第 104 行（没有检查点）退出时，只在日志里写 INCOMPLETE。建议同时写 `$OUT/summary.json`，内容为 `{"verdict": "INCOMPLETE: …"}`。

- **S7　钉住 E1 参照文件的内容。** 第 49 行只数文件个数（96）。本机这 96 个 job 文件的内容哈希为 `181acc314d29cce7fe2c7b4ac7681e4df5f02b2e39f59a1f478b8a7fbb6ee559`，算法是 `LC_ALL=C ls *.json | sort | xargs cat | sha256sum`，与平台无关。建议改为比较这个值。

- **S8　缺房检查按面板计数。** 第 1045 行统计 skipped 时用 `t.endswith(f'_s{s}')`，会把另一个面板的跳过也算进来。改为同时匹配面板前缀。

- **S9　文档过时。**
  - `g4_e3r.py` 文件头仍写着"pre-registration NOT written yet"和"D8-43"；
  - 文件头写 logit 分箱是"[-20, 20]"，实际是 [−40, 40)；
  - 链的头注释也是"before the pre-registration"，启动示例里的 `"D8-4x"` 应为 `"D8-44"`。

- **S10　可复现性。** `cudnn.benchmark = True` 会选非确定性算法，GPU 训练不能逐位复现。这可以接受，但建议在 run.json 里补记 cuDNN 版本、GPU 名称和 `allow_tf32`。

- **S11　单测缺口**（M2、M3 的测试已在上面列出）：
  - 证据选择规则（M4 的 python 段）；
  - export 的 E1 复现比较路径：可以用合成的 E1 job JSON 测一次 True、一次扰动后的 False；
  - `train()` 在合成数据上跑 2 步的端到端测试（目前只有手工冒烟）。

- **S12** 先例 `chain_day6_3SZ2.sh:93` 在每个种子开始前都会再查一次 GPU 是否空闲，可选择照抄。

## 逐项核查记录

**1. 底线：通过。**
- 网络输入只有三路：`S×0.05/4`、`log(1+k)`、`z`（`features_np` / `features_t`，第 191–204 行）。没有 ℓ0、没有真值、没有 x/y。
- 线性跳连是 `α·S + β`，初值 α = 1、β = −θ×0.05，也就是从线性读出起步。ℓ0 只通过 β 进入读出端，计一次，不是输入。
- 已知掩码在 R8、K0 上与 B4 逐位断言相同（第 837、860、872、880 行）。`unknown_known_fpr` 只取决于掩码，所以与基线相同（S5 建议写成显式断言）。
- 网络只在 k > 0 处给出判断，不补全未知区。
- eval 中所有证据的求和都经过 E1 的 `G.relay`：来源位图两两不相交，并断言等于直接求和（`relay_sums`）。
- 训练与选阈值用直接求和；两者的等价已在 eval 中断言，单测 `test_relay_sums_equal_e1_fuse` 也覆盖。
- 单测还覆盖了：零观测机加入后不改变任何结果；两张只差 ℓ0 的表给出相同输入。

**2. 数据泄漏：通过。**
- gen 只用 `scene_plan('TRAIN', 70)`。
- 噪声种子 200–202，代码断言与 {1, 2, 7, 101, 102} 不相交（第 90 行、第 527 行、`_check_seeds`）。
- 留出规则 index % 10 == 9；阈值只在留出房的 cam_nf R8 分量和上选。
- 没有早停，也没有按留出房挑检查点。
- VAL / fresh 的导出在租约之后（链第 102 行）。
- `panel_specs` 只读 VALIDATION；`scene_plan` 遇到 TEST 会抛异常。
- 导出断言 K0 行和 R8 求和行与 E1 的 job 文件逐值相同，结构已核对：K0 用 strata=True；R8 带 mult 时 strata 含 h_*、m_*。E1c 用同一写法，并已在 VAL 上跑通。

**3. 判据与 D8-44 逐字对照。**

| 条目 | 结论 |
|---|---|
| C = max(B4, B6_A, B6_B)，逐格取 R8 家具较大者，B6 用 E1c 正式表的 `cq` 与 θ = tq − l0q，读出 `S > θ`，与 `g4_calib_stalk.py:125` 一致 | 对 |
| 杠杆 +3 pp（含等号） | 对 |
| 不够：任一格 < +1 pp | 对 |
| 判决顺序：杠杆 → 不够 → 其余 ambiguous | 对 |
| 空闲召回不低于 C − 0.5 pp、表面 F1 不低于 C − 0.5 pp | 对（需要在 criteria 里填 0.005） |
| clean 档 NET 的 R8 家具与空闲召回 ≥ .99 | 对 |
| fresh 4 格均值反号 → AMBIGUOUS | 对 |
| K0 保护 | 错，见 M2 |
| 截断、fresh 缺失、对照臂缺失、clean 缺失 | 错，见 M3 |
| 证据选择规则可在发射前确定并钉住 | 目前不行；按 M4 改后可以（需等 E1c 的 `summary.json` 定稿） |

**4. 租约：基本通过。**
- SOLO：`busy()` 会等其他链、项目里的 `train_` / `diag_` / `owned_ceiling` 进程、`nvidia-smi` 上的计算进程都退出。
- mkdir 锁，退出时 trap 清除。
- 租约 JSON：7,200 s，训练截止 +6,600 s，总截止 +7,150 s，`START` 记在预跑之前。
- 每一步都有 `timeout -k 30`。
- 不续租，不 resume。预跑 300 步，用 100–300 步的计时去掉 cuDNN 自动调优的影响；两种子用同一个 `UPD`；少于 2,000 步就退出。
- 链里没有 `pkill`。
- 写入路径都在 `/mnt/datasets` 下。
- 问题：M1（PYTHONPATH）、M5（$EXPORT），以及 S3、S6。

**5. PyTorch 训练：通过。**
- 训练实现：
  - 子集求和在 GPU 上用 int8 缓存完成，与 numpy 版逐值相等（有单测）；
  - 损失掩码 = 子集私有已知的并 ∧ k > 0（有单测）；
  - 二面体增广时标签与掩码同步变换；
  - 偏置、α、β 不做权重衰减；
  - 预热后余弦退火；
  - 梯度裁剪 1.0；
  - 参数 340,115 个；
  - 种子固定初值与数据顺序（逐位确定性见 S10）。
- 显存：本机测得每个样本保存给反向的张量约 274 MiB，batch 8 约 2.1 GiB；缓存放在显卡上约 195 房 × 17.8 MB ≈ 3.5 GB。合计远低于 32 GB。
- 吞吐：由预跑在真机上实测，可信。（推测）5090 上每步在 0.05–0.15 s 量级，2,000 步的下限只需几分钟，问题只在余量（S3）。
- 选阈值时间：本机实测 CPU 侧约 0.3 s / 房·抽样，GPU 上全部留出房约 20–30 s，远低于 300 s 的预留。
- GPU 与 CPU 的一致性：见 S2。

**6. 远端环境（只读核对）。**
- torch 2.11.0+cu128，CUDA 12.8，`cuda.is_available() = True`，cuDNN 91900。
- RTX 5090，32,607 MiB，审查时空闲：占用 344 MiB，没有计算进程。
- 远端 `g4_e3r.py`、`test_g4_e3r.py`、`g4_noise_panels.py`、v420、v365、v371 和 HouseExpo 索引的 sha 都与链中钉住的值一致。
- 本机 258 个 HouseExpo JSON 在远端 `houseexpo-v365/json`（共 35,126 个）里都在。
- HOUSEEXPO 的两个变量在链的顶部 export，每一步都会继承，这没问题；**PYTHONPATH 缺失（M1）**。
- 远端单测 14/14 已过。

**7. 单测：** 14 项覆盖了底线、求和、掩码、选阈值、校准、种子和学习率调度，还缺的见 M2、M3、S11。

## 发射前清单（不属于代码问题，由协调者完成）

1. E1c 正式批次出 `summary.json` 并定稿。审查时它的 eval 在本机跑到 4 / 96。
2. 远端 `e3r-ref/e1_jobs/` **当前是空的**，需要部署 96 个 E1 job 文件（S7 给出了内容哈希），另外部署 `e1c_fit.json`、`e1c_summary.json`（M4）、`criteria.json`。
3. gen 完成后读取 manifest 的 sha，填入 `MAN_SHA`。gen 在 08:3x 时进度为 75 / 210，预计约 18 分钟跑完。
4. 填写所有 FILL_*，包括 M1 新增的 `PYR_SHA`；`E3R_SHA` 填改完后的新 sha。
5. 部署链式脚本，启动参数用 `"D8-44"`。

M2、M3 改完后，建议的 `criteria.json` 如下（键名按实现者改后的接口调整）：

```json
{"judged_panel": "val", "judged_tier": "cam_nf", "train_seeds": ["342101", "342202"], "noise_seeds": ["1", "2"],
 "control_arms": ["B4", "B6_A", "B6_B"], "lever_r8_margin": 0.03, "not_enough_margin": 0.01,
 "guard_free": 0.005, "guard_surface": 0.005, "guard_fpr": 0.0, "clean_min": 0.99,
 "k0_rule": "NET K0_sep - LIN K0_sep (both K0-specific holdout thresholds)", "k0_margin": -0.01,
 "min_updates": 2000, "truncated_blocks_lever": true, "fresh_required": true, "fresh_sign_check": true,
 "calib_slope_range": null, "calib_ece_max": null, "min_houses": null}
```

## 我做过的操作

**本机。** 全部是 CPU，用 `/d/Anaconda3/envs/jax-gpu/python -B`，同时最多 1 个进程，只用 TRAIN 房子：
- `review_e3r/time_select.py`：选阈值的开销，以及 U-Net 在 CPU 上的前向时间与 fp32 / fp64 差异；
- `review_e3r/probe_verdict.py`：在合成 summary 上检查判决逻辑；
- `review_e3r/probe_mem.py`：估算显存；
- `review_e3r/cmp_platform.py`：本机与远端 TRAIN 缓存的逐位比较。

**远端。** 都用 `timeout 60 ssh`：
- 执行过 ls、sha256sum、tail / cat 日志、`nvidia-smi` 查询、读 `/proc/<gen pid>/environ`；
- 用 `cat` 取回两个 TRAIN 缓存 npz 和它们的 record。放在会话 scratchpad 里，不在仓库内；没有 HouseExpo 原始 JSON。
- 执行过允许的 torch / CUDA 可用性查询。另外多执行了一条同类的只读查询 `python -c "import numpy, scipy; print(versions)"`，事先没有获批，特此说明。
- 没有启动任何训练或计算进程，没有碰 GPU，没有写远端文件，也没有杀进程。
