VERDICT: LAUNCH OK WITH CHANGES (M1, M2, M3)

# 发射前代码审查：v386 解码器容量单变量（batch T，`chain_day4_T.sh`）

审查者：独立 Claude 子代理，2026-10-23。只读审查，未修改任何被审文件。本地无 GPU、无 `runs/` 目录，远端 `uk-linux` 上的文件（批次 Q 的 `run.json` / `initial.pt`、缓存 manifest、`scripts/analyze_v375.py` 是否已拷贝）无法在本地核实，凡属此类均标明"无法本地核实"。

## 范围

| 文件 | 行数 | sha256 |
|---|---|---|
| `scripts/claude_v342/train_v386.py`（被审） | 743 | `77e8bc73895bd9765765a9afebc6fe701603e826b99a83c30da3b2b7dd44a068`（与链脚本第 31 行硬编码值一致） |
| `scripts/claude_v342/train_v378.py`（基线） | 732 | `db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff`（= 批次 S 在远端核对通过的值，说明本地与远端字节一致） |
| `scripts/claude_v342/test_v386_width.py`（被审） | 73 | `bdc39a01ab9615ac289d2d790e963c426c14e41ab9411f7f1eb058eb2bdbc54a` |
| `scripts/claude_v342/analyze_v386.py`（被审） | 82 | `3f9d7885f3f4ba01455ad94143cef0640ea6cc2312456b5896e90ab3ba75ed5f` |
| `scripts/claude_v342/analyze_v375.py`（被导入） | 84 | `4a004073c622fc23b30430fe518180521871e3a0c71b8a4066bee3c556ef166e` |
| `scripts/claude_v342/chain_day4_T.sh`（被审） | 112 | `be5a120320e7649099c39cb342506d273fb2f39161d4a453537672d8fa17665b` |
| `scripts/claude_v342/chain_day4_S.sh`（惯例对照，已成功发射） | 66 | `9d9d69a7e45d43b3019d10af928a823fe8e44c11dc4c28809412621840458ac0` |
| `scripts/claude_v342/chain_day4_Q.sh`（对照臂的发射命令行） | 36 | `5c8d1e2db36bf6b3090d621a1785680c096cf1c09d942a1a1815c879ebc06999` |
| `scripts/claude_v342/chain_day4_N.sh`（预检惯例对照） | 90 | `ef58c78ef13a53877439128f4bcf2b215bf9efd5306379e0f0edc86bd1547ee6` |
| `mapping_houseexpo_data_v365.py`（分层键）、`OVERNIGHT_REPORT_20261021.md` D4-10 附注 / D5-1 / D5-2、`HANDOFF_20261023.md` 第 25 行 | 只读参考 |

方法：`diff -u train_v378.py train_v386.py`（`grep -c '^[<>]'` = 33 行，与声明一致），随后通读 `train_v386.py` 全文核对改动之外的每处 `DECODER_WIDTH` / RNG / 写文件位置；手工核算两种宽度的解码器参数量；逐行走查链脚本的每条 bash / 内嵌 Python 路径；用 `python -c` 验证 `max()` 对 `nan` 的行为；把 `analyze_v386.py` 读取的每个键对回训练器实际写出的 `meta` / `final` 字典；把 `RECIPE_ARGS` 逐个对回 `chain_day4_Q.sh` 第 29 行的对照臂命令行。

## 逐项核对

### 1. `train_v386.py` 与 v378 的差异

**全部 33 行差异落在 6 处，均属宽度旋钮及其 meta / 参数 / 热启动校验：**

| 处 | v386 行 | 内容 | 语义判断 |
|---|---|---|---|
| 文档串 | 1–6 | 说明文字 | 无代码影响。附注：写的是"three marked places"，实际是 6 处（下列），仅措辞 |
| `CnnTrunk.__init__` | 69–74 | 新增 `lift=128` 形参，`128` 常量改为 `lift`；`[32] * extra_stages` 改为 `[widths[-1]] * extra_stages` | 默认值不变；`DenseDecoder` 第 220 行用默认 `widths=(96,64,48,32)` ⇒ `widths[-1] = 32`，与 v378 逐字等价；owned 臂 W=1 时 `widths=(96,64,48,32)` ⇒ 同样 32。H256/H512（`extra_stages = 1/2`）在 W=1 下与 v378 等价；W=2 时额外级为 64（"跟随末级宽度"，设计意图）。批次 T 为 H128，`extra_stages = 0`，此行不生效 |
| 模块级常量 | 188 | `DECODER_WIDTH = 1.0` | 默认 1.0 |
| `OwnedDecoder.__init__` | 197–199 | `c = round(64W)`，`lift = round(128W)`，`widths = round(W·(96,64,48,32))` | W ∈ {1, 1.5, 2} 全为整数，`round` 无歧义。`CnnTrunk.forward`（82–90）未改，`in_planes + 4` / `incoming + 4` 的公共通道拼接逻辑对任意宽度成立 |
| 参数 / 全局 / 断言 | 449、453–454 | `--decoder-width`，写入全局，`dense` 臂或 W≤0 时断言 | 正确 |
| 热启动校验 | 576 | 校验元组加入 `('decoder_width', args.decoder_width)` | 见下方注意点 |
| meta | 588 | `decoder_width=args.decoder_width` | 同时 `args=vars(args)` 已含该键 |

通读全文确认：前向（201–211、222–229）、损失（398–405）、GBP 原语（266–373）、数据（483–563）、优化器与调度器（582–584）、保存 / 评估 / 训练循环（593–739）与 v378 无任何字节差异（diff 仅上述 6 处）。

**`DECODER_WIDTH` 读取时机**：只在 `OwnedDecoder.__init__` 第 197 行读取；`OwnedDecoder` 只由 `SheafModel.__init__` 第 240 行构造；`SheafModel` 只在 `main()` 第 565 行构造，而全局赋值在第 453 行、早于它；模块级没有任何模型构造（`grep` 确认）。测试脚本直接改模块属性 `v386.DECODER_WIDTH` 绕过 `main()`，因此 CLI → 全局 → 构造这条链在测试里**没有被覆盖**，只有远端预检会第一次走到它（见 M1 对预检 `run.json` 的核对要求）。

**参数量手工核算（posterior_only，2 输入平面，H128）**：W=1：pre 38,144 + lift 226,048 + 四级 197,184 / 94,592 / 50,208 / 24,256 + 输出 66 = **630,498**；W=2：150,016 + 894,464 + 781,440 / 373,504 / 197,184 / 94,592 + 130 = **2,491,330**（3.951×）。与测试脚本第 48、67 行断言值一致。

**热启动校验的一个缝隙（批次 T 不触发）**：第 577 行沿用 `payload.meta.get(mkey, mval) == mval` 的写法，对 v378 写出的检查点（没有 `decoder_width` 键）默认取 CLI 值本身 ⇒ 任意 W 都通过该校验；错配只会在第 579 行"部分热启动"断言被捕获，而 `--allow-partial-init` 会放过它。T 不用 `--init-from`，不影响本批（建议项 S2）。

**`test_v386_width.py` 证明了什么、没证明什么**：
- 证明：W=1 下 state_dict 键序 / 形状 / 数值逐张量相等、编码器输出与解码器输出逐位相等、`predict` 相等、v386 的 W=1 state_dict 能 `strict=True` 加载进 v378 模型（33–48 行）；W ∈ {1.5, 2} 下 `trunk.*` / `head.*` 张量与 v378 逐位相等、`source()` 输出相等、解码器输出形状不变且有限（52–68 行）；参数量 630,498 / 2,491,330；不同种子确实改变编码器（70–71 行）。
- 未覆盖：(a) `main()` 的参数管道（CLI → 全局、meta 的 `decoder_width` 键、热启动断言、dense 臂断言）；(b) `extra_stages > 0`（H256/H512）下 W=1 的等价（代码上显然，但无测试）；(c) dense 臂 W=1 等价（显然）；(d) `anchored` readout（4 平面）——测试强制 `posterior_only`，与 T 配方一致，可接受；(e) 构造后全局 torch RNG 状态在不同 W 下相同（见第 2 项，由 `fork_rng` 保证，但无断言）；(f) 训练一步（损失 / 梯度 / 优化器）的等价——由 diff 证明代码路径未改，可接受。(a) 由远端预检实际走到，建议在链脚本里顺手核对预检写出的 `run.json`（M1）。

### 2. "编码器初始权重相同 + 数据顺序相同"是否成立

- `SheafModel.__init__`（236–242）：`torch.random.fork_rng(devices=[])` + `manual_seed(seed)`，构造顺序 trunk → head → decoder。trunk / head 的随机抽取在 decoder 之前，与 W 无关 ⇒ 任意 W 下逐位相同（测试第 2 组已验证）。退出 `with` 块时 CPU 全局 RNG 恢复到进入前的状态 ⇒ **构造之后的全局 torch RNG 状态与 W 无关**。CUDA RNG 未 fork（`devices=[]`），但模型在 CPU 上构造、第 565 行才 `.to(device)`，构造不消耗 CUDA RNG。
- `main()`：第 478 行 `torch.manual_seed / np.random.seed / random.seed`；数据加载（483–563）走缓存（`np.load` + 哈希校验）与封存生成器，不依赖 W；第 661 行 `order_rng = np.random.default_rng(args.seed)` 独立于 torch，第 663 行 `permutation(len(train))` 只依赖种子与 `len(train) = 14,400` ⇒ 与批次 Q 的顺序逐位相同。
- 构造之后是否还有消耗 torch RNG 的东西：`grep` 全文无 `dropout / rand / multinomial / bernoulli`；`F.interpolate`、`truncated_dt`、森林 GBP、量化器、`AdamW`、`clip_grad_norm_` 均确定性；`evaluate` 在 `no_grad` 下。**结论：声明成立。**
- 两条需要写明的保留：(i) 训练器未设 `cudnn.deterministic`，卷积反向可能用非确定性核 ⇒ 即使 W=1 重跑也不会逐位复现 Q，"同初始 + 同顺序"不等于"同轨迹"；(ii) 第 692 行 `clip_grad_norm_(model.parameters(), 5.)` 是对**全部参数**的全局裁剪，解码器参数增 3.95× 会改变总范数，裁剪触发时编码器的有效步长随之变化 ⇒ "编码器不变"只对架构与初始权重成立，不对其优化轨迹成立（第 7 项）。

### 3. `analyze_v386.py` 读取的键与训练器写出的键

| 读取 | 训练器写出 | 一致 |
|---|---|---|
| `final.json`：`updates`、`epochs_completed`、`val`、`fresh`（分析 22–23、60 行） | 720、724、729 行 | 是；`val` 缺失（超 `deadline_total`）时 `complete` 为 False，60 行用 `.get` 安全 |
| `run.json`：`epochs`、`args`、`decoder_width`、`parameters{trunk,head,decoder}`、`train_cache{dir}`（24–25 行） | 585–588 行 | 是。对照由 v378 写出，无 `decoder_width`，`meta.get(..., meta['args'].get(..., 1.0))` ⇒ **1.0**，正确 |
| `manifest.json['per_stratum']`（25 行） | 训练器 530 行同样读该键 | 是（远端文件无法本地核实） |
| `summary['8']['all']['wall_f1' / 'unknown_known_fpr']`（60–62 行） | 637、652–653 行 | 是（`str(k)` 键） |
| `summary['8']['strata']['rooms3/N8' / 'rooms3/N16']`（63 行） | 652 行 `f'rooms{a}/N{b}'`；v365 `STRATA = {1: 2–4 室, 2: 5–7 室, 3: 8+ 室}`（第 37 行）、`AGENT_COUNTS = (8, 16)`（第 36 行） | 是，`rooms3` 正是预注册的"8+ 室分层" |
| `initial.pt['state_dict']` 的 `trunk.*` / `head.*`（34–38 行） | 591 行；`SheafModel` 属性名 trunk / head / decoder | 是 |
| `eq_weight(rows, k)`（16、58 行） | `analyze_v375.py` 第 19 行 `eq_weight(rows, k, metric='wall_f1', drop=frozenset())` → `(float|None, dict)` | 是。导入依赖远端 `scripts/analyze_v375.py` 存在（批次 N 第 89 行曾以 `$P scripts/analyze_v375.py` 运行，应已在位；无法本地核实），且 `$P scripts/analyze_v386.py` 的 `sys.path[0]` 即 `scripts/`，可解析 |

**`RECIPE_ARGS` 对回批次 Q 的命令行**（`chain_day4_Q.sh` 第 29 行 vs T 第 39 行 `COMMON`）：`lr`（Q 未传 ⇒ 默认 `3e-4`，T 传 `--lr 3e-4` ⇒ 同一浮点 0.0003）、`fusion max`、`readout posterior_only`、`precision learned`（默认）、`wire sealed`、`mu_bits 8`、`q_bits 8`、`dt_trunc 4`、`consistency 0.`、`edge_select 0.`、`size 128`、`arm owned`、`seed`、`world houseexpo_fit`、`epochs 4`、`train_cache`——Q 与 T 的 `$C`（第 13 / 26 行）与 `$S`（第 11 / 24 行）字符串逐字相同，`--train-cache $C/houseexpo_fit-k4800` 展开为同一绝对路径 ⇒ `train_cache` 入 `RECIPE_ARGS` **不是问题，反而是正确的身份守卫**；预期 `diffs = {}`。不在 `RECIPE_ARGS` 的差异只有 `decoder_width`（设计变量）；`val_every 1`、`refine_steps 0`、`expect_updates 57600` 两边相同。

阈值：`min(g) >= 1`、`max(|g|) < 1`、FPR `> 0.3`（66–70 行）与链脚本头部 10–14 行、D5-2 文本一致；单位 pp（×100）一致。链脚本第 111 行只把分析输出的**最后一行**写进日志，若分析抛异常，日志里只剩 traceback 末行（建议项 S1，可事后重跑，不影响数据）。

### 4. `chain_day4_T.sh` 的 shell 正确性与预检

- **`spu` 函数（46–54 行）**：函数体内 `$P - "$1" <<'PY'`，`PY` 结束符顶格、`}` 在其后一行，`$( )` 内调用函数带 heredoc 合法；`sys.argv[1]` 为日志路径。正则 `upd(\d+) .* (\d+)s$` 对训练器第 699 行格式（`... mask 0.38 57s`）匹配正确；`logging = (updates+1) % 50 == 0` ⇒ 打印 upd50/100/150/200，第 666 行在 `updates >= 200` 时退出 ⇒ `t[200]` 一定存在（除非崩溃）。**精度问题**：秒数是 `:.0f` 取整，`t[200]-t[50]` 误差 ±1 s，而该区间在 W=2 双任务下约 15–40 s ⇒ s/update 相对误差最大 ±7 %，**超过第 67 行的 5 % 余量**（M2）。
- **`MODE` heredoc（58–69 行）**：输出形如 `PARALLEL 5600 6300` / `SEQUENTIAL …` / `ABORT`（仅 `ValueError`）。**`float('nan')` 不抛 `ValueError`**，而 `max(0.1, nan)` 返回 `0.1`（本地 `python -c` 验证：`max(0.1, nan) = 0.1`，`max(nan, 0.1) = nan`）⇒ 若种子 342202 的预检崩溃（OOM、断言）而 342101 正常，第 62 行的 `spu` 就只取 S1，链脚本会按 PARALLEL / SEQUENTIAL 正常发射，并把 `[0.1, nan]` 写进 `lease.json`（非法 JSON）。反向顺序（S1 为 nan）才会碰巧走到 SEQUENTIAL 的 FIT=NO 中止。**预检失败可被静默忽略，这是必改项 M1。**
- **`case` 模式（71–109）**：`PARALLEL*` / `SEQUENTIAL*` / `*` 正确；Python 报错时 `MODE` 为空 ⇒ `*` 分支中止，正确。
- **顺序租约文件名（92 行）**：`lease$([ $seed = 342101 ] && echo "" || echo 2).json` ⇒ `lease.json` / `lease2.json`，正确。
- **pgrep 模式**：第 36 行 `^$P scripts/train_v3` 锚定真实训练器 argv，能拦住仍在跑的 `train_v378.py`（批次 S）；发射用的 ssh 外壳命令行以 `bash -c 'cd … && nohup bash scripts/chain_day4_T.sh …'` 开头，**不匹配**；链脚本内的 `$P -`、`$P -c`、`$P scripts/analyze_v386.py` 也不匹配；等待循环（85、103 行）锚定到 `--out runs/v386-decoder/wide…`，预检进程（`preflight-s…`）不匹配。全部正确。
- **预检隔离**：`--out runs/v386-decoder/preflight-s$seed`（43 行）与正式 `wide-s$seed`（81 / 100 行）不同目录；训练器第 474 行 `mkdir(exist_ok=False)`；分析只读 `wide-s*`。**预检不可能写进正式目录。**
- **`--max-updates 200` + `--expect-updates 57600`**：第 542 行断言 `epochs × len(train) = 4 × 14,400 = 57,600`，与 `max_updates` 无关 ⇒ 通过。
- **预检流程**：第 666 行在第 201 次迭代开头 `break`；第 702 行 `stopped = True` ⇒ 第 704 行**轮末验证会运行**（48 个 VAL 场景，W=1 时约 40–70 s，W=2 时更重）；写 `best.pt`、`latest.pt`、`history.json`；第 717 行 `break`；第 721 行写 `final.pt`；第 722 / 727 / 732 行因 `deadline_total = START` 已过而**跳过**三项最终评估；写 `final.json`，打印 `done`。预检总耗时 ≈ 加载（14,400 个 npz 的读入 + 逐文件哈希校验 536 行 + 8 行再生成校验 532 行 + 96 个 VAL/fresh fixture 生成与 DT 缓存）+ 200 步 + 一次验证 + 4 次检查点保存，估计 **3–6 分钟**；PARALLEL 模式下它落在租约内（`avail = DE − NOW` 已扣除）。
- **预检的"同等竞争"假设不可靠**：批次 N 的实录（`OVERNIGHT_REPORT` D4-10 附注）：预检估计 0.09 s/update，实跑 0.13 s/update，原因是预检时各任务处于加载 / 固定内存等不同阶段。T 的测量窗口 upd50→upd200 只有十几到几十秒，两个任务的加载时间只要相差这么多，测到的就是单任务速度。头部第 17 行"same contention as the real launch"没有被脚本验证（M2）。

### 5. 租约算术与回退规则

- **PARALLEL**：`START` 取在预检之前（38 行），`DE = +6600`、`DT = +7150`、`inclusive 7200`，与 Q/R/S 相同；预检包含在租约内（保守）；正式任务的训练窗口 = 6,600 − 预检耗时，第 66 行 `avail` 正是这个值，口径一致。但第 65 行 `need = 57600 × spu + 4 × 70` **漏掉**正式任务自己的启动 / 加载时间（与预检相同量级，约 1–4 分钟）与 W=2 下更重的验证（70 s 是 W=1 的经验值），再叠加 s/update 的 ±7 % 取整误差与 N 批那种低估，5 % 余量不足 ⇒ **PARALLEL 分支的装得下判断不保守**（M2）。后果不是错误结论而是跑不满 ⇒ INCONCLUSIVE，但会白费一个 2 小时租约。
- **SEQUENTIAL**：每个种子独立 `LS / LDE / LDT`（92 行），租约在发射前写出（93–98 → 100 行），第二个种子只在第一个进程退出后启动（103 行）⇒ **每个任务各在自己的包含式 7,200 s 租约内，符合 HANDOFF 第 25 行的规则**。装得下判断（89 行）`57600 × s_parallel + 280 ≤ 6400` 在两方面保守（用双任务速度估单任务；6,400 对 6,600），漏掉加载时间（约 100–300 s）后净效果仍保守。
- **头部与实现不一致**：头部第 19–20 行写"using the single-job preflight speed"，代码用的是**并行**速度（89 行 `max(S1, S2)`）。这不仅是记录问题，还很可能决定本批能否发射：Q 的 W=1 双任务速度 0.094 s/update（`chain_day4_Q.sh` 第 6 行），只要 W=2 的双任务速度超过约 0.106 s/update（增加 13 %），PARALLEL 不装、SEQUENTIAL 也判 NO ⇒ 整批中止；而单任务独占 GPU 的速度通常是双任务的 0.6–0.7 倍，顺序方案本可装下。按解码器 FLOPs 约 4× 估算，W=2 的每步耗时增幅落在 1.2–1.6× 区间的可能性很大（M3）。
- **无租约发射的路径**：PARALLEL / SEQUENTIAL 两条正式发射路径都先写租约再 `nohup`；不存在无租约的正式任务。预检在 SEQUENTIAL / ABORT 路径下没有任何租约记录（PARALLEL 下由 `lease_start_epoch = START` 追溯覆盖）（S3）。**超时未记录的路径**：截止由训练器自身执行（666、702、722–732 行），链脚本没有外部看门狗，训练器若挂死会无限等待（`wait` 45 行、`while pgrep` 85 / 103 行），与 Q/R/S 相同（S4）。
- **M1 的 nan 路径**是唯一能让"未经验证的种子"进入租约的路径。

### 6. 写入范围与历史覆盖

- 链脚本：`runs/chain_day3.log`（`tee -a` 追加）、`runs/v386-decoder/`（`mkdir -p` 37 行；`preflight-s*` 目录与 `.log`、`lease.json` / `lease2.json`、`wide-s*` 目录与 `.log`、`analysis.json`）。全部相对于第 23 行 `cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021`。
- 训练器：全部在 `--out` 之下（474–476、589、591、597、643、706、714、725、730、735、737 行）；缓存只读（535 行 `np.load`）；对照臂只被分析脚本 `torch.load` / `json.load` 读取。无 `rm` / `mv` / `shutil` / `unlink`。第 35 行目录守卫 + `exist_ok=False` 双重保证不覆盖。
- 副作用：`import` 会在 `scripts/` 与 `$S/scripts`（封存源码目录）下写 `__pycache__`，各批均如此，非新增。
- 需知悉：若在第 90 或 108 行中止，`runs/v386-decoder/` 已带着预检输出存在，再次发射会被第 35 行守卫拒绝，须手工归档（改名而非删除）后才能重发（S3）。

**结论：无 `/mnt/datasets/.../runs/v386-decoder` 与追加日志之外的写入，无删除或覆盖历史。**

### 7. 预注册的科学效度

等更新数、同种子、同配方、只改解码器宽度，是一个干净的单变量配对设计；初始权重逐位身份守卫与 `RECIPE_ARGS` 守卫使"唯一变量"可被事后审计。作为"解码器容量是否是杠杆"的检验，它是公平的，但应补写以下保留（头部已有第 15–16 行的速度 / 容量混淆一条）：

1. **学习率未随宽度重调**：标准参数化下 2× 宽度通常对应更小的最优 lr，3e-4 + 200 步 warm-up + 余弦照搬 ⇒ 阴性结果可能是"宽解码器在这个 lr 下学不好"，阳性结果也部分含"lr 对宽网络更合适"的成分。
2. **全局梯度裁剪耦合**（692 行）：裁剪阈值 5.0 作用于全部参数，解码器参数 3.95× ⇒ 总范数变化 ⇒ 裁剪触发时编码器有效步长改变。Q 的日志每 50 步打印各组梯度范数，可事后统计裁剪触发频率来定量这一耦合。
3. **等更新 ≠ 等算力**：WIDE 每步的解码器 FLOPs 约 4×，等更新数比较偏向 WIDE；本批不能回答"算力匹配下是否更优"。
4. **运行间噪声**：D5-1 自述运行间噪声 ±1–2 pp，1 pp 门槛处于噪声量级；"两种子都 ≥ 1 pp"减少但不消除假阳性；对照是早先跑过的单次运行（非同期重跑），cuDNN 非确定性意味着即使 W=1 也不能逐位复现 Q。建议附报两臂逐轮 VAL R8 曲线（`history.json`，每种子 4 对点）与 Q 的两种子离散度作为解读背景，判据本身不改。
5. **记忆容量**：参数 3.95× 也提高了对近重复户型的记忆能力（批次 N 的 M3 关切）。建议把 `neardup2_v375.json`（k4800 ∪ k200 标记集）的"无近重复"次要增益与 fresh 面板一并描述性报告，区分泛化与记忆。
6. **单点、均匀加宽**："杠杆"结论只对"均匀 2× 宽度、从头训练 57,600 步"成立，不定位容量在 pre / lift / 高分辨率级中的哪一处，也不覆盖深度或其他解码器家族；参数比 3.95× 与 FLOPs 比不同（高分辨率级占 FLOPs 大头）。
7. **预算层级**：对照是第 1 周期（Q），当前前沿是三周期（S）；在 Q 预算下的结论未必迁移到三周期日程（头部第 11 行已把它列为后续步骤，宜写明"迁移未经检验"）。

## 必改项

**M1 — 预检失败必须中止，而不是被 `max()` 吞掉（`chain_day4_T.sh` 58–69、89 行）。**
现状：`spu` 对缺失 `upd200` 输出 `nan`；`float('nan')` 不触发 `ValueError`，`max(S1, nan) = S1` ⇒ 一个种子的预检崩溃（OOM、断言、CUDA 错误）会被静默忽略并照常发射，`lease.json` 还会写入非法的 `nan`。要求：在写任何租约之前，两个种子都必须满足 (i) `S1`、`S2` 是有限正数（例如 `MODE` 的 Python 里 `if not all(math.isfinite(x) and x > 0 for x in (s1, s2)): print('ABORT')`，并在 SEQUENTIAL 分支的 FIT 中同样处理）；(ii) `runs/v386-decoder/preflight-s<seed>/history.json` 末项 `updates == 200` 且 `train_loss` 有限（照抄 `chain_day4_N.sh` 64–68 行）；(iii) `preflight-s<seed>/run.json` 的 `decoder_width == 2.0` 且 `parameters.decoder == 2491330`，`parameters.trunk` / `parameters.head` 与对照 `runs/v382-long/d4800e4-s<seed>/run.json` 相等（这一步同时补上测试未覆盖的 CLI → 全局 → 构造管道）。任一不满足 ⇒ `log "ABORT …"; exit 2`，不写租约、不发射。

**M2 — 装得下的投影必须保守（`chain_day4_T.sh` 46–69 行）。**
现状的三个漏洞：(a) s/update 取自取整到秒的两条日志行之差，±7 % 误差大于 5 % 余量；(b) `need` 漏掉正式任务的启动 / 加载时间，验证时长沿用 W=1 的 70 s；(c) 没有任何检查保证预检的测量窗口确实处于双任务竞争之下（批次 N 曾因此低估 44 %）。要求：
- 速率改用 `history.json` 的 `epoch_seconds / 200`（浮点，含首步开销，天然偏保守；N 批惯例），或把预检延长到 `--max-updates 500` 以上并取后段窗口；两者取大值亦可。
- `need = 57600 × spu + 4 × val_s + load_s`，其中 `val_s` 取预检日志 `done` 秒数减 `t[200]`（含检查点保存，偏保守），`load_s` 取 `t[50] − 50 × spu`（或 `gen` 行秒数加固定余量）。
- 竞争性检查：W=2 的双任务速度不可能快于 Q 实测的 W=1 双任务速度 0.094 s/update；若 `max(S1, S2) < 0.094`，判定预检未处于竞争状态，禁止走 PARALLEL（降级为 SEQUENTIAL 判定或中止），并把判定依据写进 `lease.json`。
- 余量保持 ≥ 5 %，并把 `need` 的三个分量分别写进 `lease.json`（`preflight_s_per_update`、`preflight_val_seconds`、`preflight_load_seconds`），便于事后核对投影误差。

**M3 — 头部声明与实现必须一致：顺序回退用什么速度（`chain_day4_T.sh` 19–20 行 vs 89 行）。**
头部与 D5-2 记录的是"用单任务预检速度判定顺序方案是否装得下"，代码用的是并行速度。二选一，推荐前者：当 PARALLEL 不装时，先单独跑一个种子的 200（或 500）步预检（`preflight-solo-s342101`，同样 `--deadline-total $START`），用其 `epoch_seconds` 速率按 M2 的公式对 6,600 s 窗口做 ≥ 5 % 余量的判定；装不下再 ABORT INCONCLUSIVE。理由见第 5 项：按并行速度判定，W=2 很可能在能装下的情况下被整批中止。若选择保留现实现，则必须改写头部第 19–20 行并在 `OVERNIGHT_REPORT` D5-2 加注"顺序判定使用并行速度（更保守）"，使预注册记录与代码一致。

## 建议项

- **S1**：第 111 行把分析完整输出重定向到 `runs/v386-decoder/analysis.txt`（Q 的惯例，第 35 行），日志只摘 `PRIMARY` 行；链脚本开头顺手守卫 `[ -f scripts/analyze_v386.py ] && [ -f scripts/analyze_v375.py ]`（远端是否已拷贝无法本地核实），并记录两者 sha。
- **S2**：`train_v386.py` 第 576–577 行对 `decoder_width` 的默认值改为 1.0（`payload.meta.get('decoder_width', 1.0) == args.decoder_width`），使 v378 检查点 + `--decoder-width 2.0 --allow-partial-init` 不能静默通过；本批不受影响。
- **S3**：SEQUENTIAL 分支的两个租约文件加 `preflight_start_epoch: $START` 与预检耗时，使链脚本的全部 GPU 占用都有记录；在 90 / 108 行中止时把 `runs/v386-decoder` 改名为 `runs/v386-decoder.aborted-$START`（改名不删除），避免重发被第 35 行守卫拦住后手工操作。
- **S4**：等待循环加看门狗：超过 `DT + 300` 仍有进程则 `log` 并 `kill`（训练器自行执行截止，看门狗只防挂死）；预检的 `wait` 也可加 `timeout`。
- **S5**：`test_v386_width.py` 补三条便宜断言：构造 W=1 与 W=2 模型后 `torch.get_rng_state()` 相等；`SIZE=256`（`extra_stages=1`）下 W=1 与 v378 state_dict 相等；在 CPU 上对同一批随机输入做一步 `stage_loss` 并比较损失值与各参数梯度（锁定"训练器其余部分未改"的声明）。
- **S6**：`analyze_v386.py` 加描述性次要项：两臂逐轮 VAL R8（`history.json`）、`neardup2_v375.json` 的无近重复增益、Q 两种子离散度；并把 `train_cache.manifest_sha256` 两边相等纳入 `ok`。
- **S7**：把第 7 项的保留 1–7 写进链脚本头部与 D5-2（预注册时写明，事后不能补）。
- **S8**：`train_v386.py` 文档串"three marked places"改为实际处数；`chain_day4_T.sh` 第 17 行"same contention as the real launch"改为 M2 的可验证表述。
- **S9**（沿袭 S 批审查 S3，仍未采纳）：第 36 行守卫只排除本项目训练器，可加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && ABORT`。

## 结论

**LAUNCH OK WITH CHANGES（M1、M2、M3）。**

训练器方面：v386 与 v378 的 33 行差异全部是宽度旋钮及其参数 / meta / 热启动校验，W=1 下与 v378 逐字节等价（含 DenseDecoder 与 H256/H512 的额外级），`DECODER_WIDTH` 只在 `main()` 赋值之后被读取；"编码器 / head 初始权重相同、数据顺序相同"的声明由 `fork_rng` + 构造顺序 + 独立的 `order_rng` 保证，构造后训练器没有任何 torch 随机消耗；参数量 630,498 / 2,491,330 手工核算一致。分析脚本读取的每个键都由训练器写出，`RECIPE_ARGS` 与批次 Q 的命令行逐项相同（含 `train_cache` 字符串），对照的 `decoder_width` 正确解析为 1.0，`rooms3` 即 8+ 室分层，`eq_weight` 签名匹配。链脚本的 pgrep 锚定、预检隔离、租约文件命名、顺序租约算术与写入范围均正确，无历史覆盖。

三处必改都在链脚本的预检 / 投影逻辑：一个种子的预检失败会被 `max()` 吞掉而照常发射（M1）；PARALLEL 的装得下判断用了 ±7 % 误差的速率、漏掉加载时间、也不验证测量窗口是否处于竞争之下——批次 N 已为此付过一次代价（M2）；顺序回退的判定依据与头部 / 预注册文本不一致，且按现实现 W=2 很可能在本可装下的情况下整批中止（M3）。三处改完后可发射；科学保留按 S7 在发射前补入记录。

---

## 审查后的修改（主代理，2026-10-23 发射前）

三条必改项全部落实，判定逻辑抽成独立脚本 `preflight_check_v386.py`（可在已有运行文件上自测；合法路径与三种非法路径均已在 uk-linux 上验证）：
- **M1**：预检有效性检查——两个预检都必须有且只有一条 `updates == 200`、损失有限的 history 记录；`run.json` 的 `decoder_width == 2.0`、`parameters.decoder == 2,491,330`、trunk/head 参数量与对照相同；日志必须有 `gen`/`done` 行；任一不满足 → `INVALID` → 链脚本 ABORT，不写 lease、不发射。
- **M2**：速率改用 `history.json` 的浮点 `epoch_seconds / 200`（含 warm-up，偏保守）；投影 `need = load + 57,600 × rate + 4 × val`（load、val 都从预检实测）；并行判定还要求 `rate ≥ 0.094 s/更新`（批次 Q 的双任务 W=1 速率），低于它说明两个预检没有重叠，判 INVALID；最终评估窗口另检查 `2 × val ≤ 550 s`。lease 里记录整条判定行。
- **M3**：顺序回退改为由**单任务预检**（种子 342101 单独 200 次更新）决定，`1.05 × need_solo ≤ 6,500 s` 才顺序发射（每个种子各一个 7,200 s 租约）；头部注释、D5-2 与代码已一致。
- 其他：预检目录加时间戳（`pf-par-<ts>-s<seed>`、`pf-solo-<ts>-s342101`），守卫只禁止 `wide-s*` 已存在，这样 ABORT 后可以重跑而不覆盖历史。
发射版本 sha256（前 16 位）：`chain_day4_T.sh 27d9f0d58a9571ed`、`preflight_check_v386.py 1add9aa737d43d5a`、`analyze_v386.py 3f9d7885f3f4ba01`、`train_v386.py 77e8bc73895bd976`。
