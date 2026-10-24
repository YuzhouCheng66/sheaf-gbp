VERDICT: LAUNCH OK WITH CHANGES (M1, M2)

# 发射前代码审查：v392 解码器的公共帧条件化单变量（batch V，`chain_day4_V.sh`）

审查者：独立 Claude 子代理，2026-10-24。只读审查，未修改任何被审文件。本地无 GPU、无 torch（`test_v392_frame.py` 无法本地复跑，只能逐行核对其断言逻辑；按主代理记录它已在本地与 uk-linux 通过）、无 `runs/` 目录；远端文件（批次 T 的 `run.json` / `initial.pt`、缓存 manifest）无法本地核实，凡属此类均标明。

## 范围

| 文件 | 行数 | sha256（前 16 位） | 备注 |
|---|---|---|---|
| `scripts/claude_v342/train_v392.py`（被审） | 795 | `ef4dc60082ef9f5f` | 与 `chain_day4_V.sh` 第 29 行硬编码值一致 |
| `scripts/claude_v342/train_v386.py`（基线） | 743 | `77e8bc73895bd976` | = 批次 T 发射版本（`code_review_v386_decoder.md` 末尾记录） |
| `scripts/claude_v342/test_v392_frame.py`（被审） | 77 | `f02afe557e74b2a7` | |
| `scripts/claude_v342/analyze_v392.py`（被审） | 92 | `e2e55ebc013e59b4` | 由 `analyze_v386.py`（`3f9d7885f3f4ba01`，T 发射版本）改出 |
| `scripts/claude_v342/analyze_v375.py`（被导入） | 84 | `4a004073c622fc23` | 提供 `eq_weight` |
| `scripts/claude_v342/preflight_check_v386.py`（被审，已重生成） | 64 | `fad8a9d8a8207ea7` | 含 `--code-embed` 与 `--frame-cond` 两个可选检查 |
| `scripts/claude_v342/chain_day4_V.sh`（被审） | 104 | `636ccbf11f34cc2d` | 由 `chain_day4_T.sh`（`27d9f0d58a9571ed`，已成功发射）改出 |
| `scripts/claude_v342/mapping_houseexpo_data_v371.py` / `_v365.py` | 122 / 208 | `6abdc221f01c323b` / `e1e6b5aab5105190` | 只读参考：`world_spec`、`make_fixture`、`public_frame` |
| `OVERNIGHT_REPORT_20261021.md` D4-8、D5-5、D5-8、D5-9；`fp_day5_decoder.md` 候选 2；`claude_handoff/README.md` 第 91 行 | 只读参考 | | |

方法：`diff train_v386.py train_v392.py`（`grep -c '^[<>]'` = **72 行**，与声明一致），逐处核对；通读 `train_v392.py` 的模型段（73–134、235–306）与 `main()` 全文（460–795），追踪 `CHART_M` 在训练 / 评估每条路径上的赋值与读取；把 `public_extra` 的坐标约定对回 `public()` 与 v371 的 `coordinate_convention`；手工核算参数增量；把 `analyze_v392.py` 读取的每个键对回训练器写出的 `meta` / `final`；逐行走查 `chain_day4_V.sh` 的每条路径（`bash -n` 通过）并与 `chain_day4_T.sh` 逐行比对。

## 逐项核对

### 1. `CHART_M` 在训练与评估的每条路径上是否正确且生效

**(a) 缓存 TRAIN 项（`load_cached`，`stream=True`）**
- `--world houseexpo_fit` 时 `data` 是 v371 模块：`train_v392.py` 506–510（先 import v365 再被 v371 覆盖）。缓存守卫 572–573 断言 `man['world'] == args.world` 且 `data_module_sha256` 等于当前 v371 文件的 sha，577–578 断言行 = `scene_plan(TRAIN)`、index sha 一致，580 行首尾 8 行在当前模块下复现 private hash ⇒ 缓存里的 private 栅格就是用同一条图幅规则生成的。
- `data.world_spec(split, index, rooms)`（584 行）= `v371.world_spec`（v371:32–39）：调用 `v365.world_spec`（v365:95–109，index 由模块级 `_INDEX` 只加载一次，v365:81–92；每户读一个小 JSON、算一次 contour sha256、减中心），再取 `np.ptp(v,0).max()`，`L = max(8, ceil(ext + 1))`，**返回的 dict 含 `chart_m=L`**（v371:37）。不调用 `rasterize` / `metric_trajectory`（纯多边形，无栅格）。
- 代价：14,400 行 × （一次小 JSON 打开 + sha256 + ptp），本地盘为秒级、`/mnt/datasets` 若是网络挂载约 1–2 分钟；它进入 597 行的 `gen <s>s` 时间，而预检脚本正是用这条 `gen` 行作 `load`（`preflight_check_v386.py` 45–49）⇒ **投影已自动计入**，不会低估。
- `chart_m=float(spec_r.get('chart_m', 20.))`（587 行）：v371 恒有该键，不会静默落到 20。

**(b) VAL / fresh 面板（`load()`）**
- 549 行 `float(inp['public_frame'].get('extent_m', [20., 20.])[0])`。v371 `make_fixture`：`frame = dict(..., extent_m=[L, L], ...)`（v371:114，`L = spec['chart_m']` 109 行）⇒ `[0]` 即 L ✓。v365:189 `[20., 20.]`、v293 `mapping_multiroom_data_v293.py`:162 `[20., 20.]` ⇒ 1.0 平面 ✓。`load_dynamic`（551–565）不写 `chart_m` ⇒ 默认 20，而 `--dynamic` 只允许 v293（513 行）✓。
- 潜在隐患（不影响本批）：v378 `houseexpo_rect` 的 `extent_m=[lx, ly]`（v378:202），`[0]` 只取 Lx；`world_spec` 用 `chart_xy_m`（v378:105）。见建议 S6。

**(c) Python 作用域**
- `main()` 第 498 行：`global PRECISION_MODE, ..., DECODER_WIDTH, FRAME_COND, CHART_M` —— **`CHART_M` 确在其中** ⇒ 训练循环 726 行 `CHART_M = sc.get('chart_m', 20.)` 写的是模块全局，不是局部变量。
- 嵌套函数 `evaluate` 第 650 行自带 `global CHART_M`（嵌套函数不继承外层 `global` 声明，这里正确处理）⇒ 654 行写模块全局。
- `public_extra` 第 91 行在**调用时**读模块级 `CHART_M`，不是默认参数或闭包捕获 ⇒ 逐场景更新生效。
- 顺序：赋值（654 / 726）在 `model.source`（655 / 727）、`rollout`、`stage_loss` / `predict` 之前；精修分支（656–670）与 `--oracle-union` 分支（672–681）也在赋值之后 ⇒ 每次解码器前向都用当前场景的 L。训练项经 `sc = dict(sc, own_t=..., tgt=...)`（721 / 724）复制时保留 `chart_m` 键 ✓。
- 以 `python scripts/train_v392.py` 运行时模块名为 `__main__`，`main()`、`evaluate`、`public_extra` 共享同一全局命名空间 ✓。

**结论：四条路径（缓存训练、内存训练、VAL/fresh、dynamic）上 `CHART_M` 均正确且生效；houseexpo_fit 路径上不存在静默默认为 20 的情形。** 唯一不足是运行产物里没有记录实际使用的 `chart_m`（见 S1）。

### 2. `public_extra` 的构造与栅格约定、设备 / dtype

- 85–92 行：`y=(arange(sy)+.5)/sy`、`x=(arange(sx)+.5)/sx`，`meshgrid(y, x, indexing='ij')` ⇒ `xx` 沿列（最后一维）变化 = x，`yy` 沿行变化 = y；与 `public()`（73–78）完全同构，与 v371 `coordinate_convention='pose=(x,y); raster=[y,x]'`（v371:115）一致。`u = frac(xx·8) − .5` 只依赖 x，`v` 只依赖 y（测试第 68 行断言）。文档串 "frac(8·(x+1)/2)" 与此等价（`(x+1)/2 = (i+.5)/sx`）。
- 相位精度：各级 extent 为 16, 16, 32, 64, 128（128–133 行：lift 在 16，之后每块先插值到 `extent` 再翻倍，封顶 `size`），均为 8 的倍数；`xx·8 = (i+.5)·8/sx` 对这些 sx 都是二进制有限小数，float32 下 `frac` 精确 ⇒ 恰好 8 个周期、7 次回绕（测试 65–66 行），取值 ∈ [−0.5, 0.5) ✓；16×16 级每块 2 px，相位 ±0.25，仍有意义。
- dtype：`arange` 为 int64，`+ .5` 升为默认 float32，与 `public()` 相同；`torch.full_like(yy, float(CHART_M)/20.)` 的 `float()` 兼容 numpy 标量（实际上 549 / 587 行已 `float()` 过，`sc.get` 返回 Python float）。device 取 `grid.device`（128 / 132 行）✓。`expand` 后 `torch.cat` 会物化，无 contiguity 问题（与 `public()` 同样写法）。

### 3. 初始恒等：加宽的卷积、原列与偏置、DenseDecoder

- 117–120 行：仅 `self.lift[0]` 与每个 `block[0]` 被加宽；W=2、H128 ⇒ `extra_stages = log2(128//128) = 0`，4 个 block ⇒ **5 个卷积**（测试第 47 行断言 `len(widened) == 1 + len(blocks)`）。`lift[1]`、`block[2]`、`output` 的输入不含公共平面，正确地不加宽。
- `widen_conv`（95–100）：新 `Conv2d(cin+3, cout, kernel_size, padding)`，stride / dilation / groups / bias 用默认 = 原卷积（原卷积都是 3×3、padding 1、stride 1）；`weight.zero_()` 后 `weight[:, :cin] = conv.weight`、`bias.copy_(conv.bias)` ⇒ 原列与偏置逐位相等、新列为零 ✓。`nn.Sequential.__setitem__` 替换不改 state_dict 键名（测试 29 行 `list(sa) == list(sb)`）。
- RNG：`widen_conv` 内 `nn.Conv2d(...)` 会消耗 RNG（kaiming 初始化），但发生在 `CnnTrunk.__init__` 的 v386 抽样之后，而 trunk 是 `OwnedDecoder` 的最后一个子模块（242–243），decoder 是 `SheafModel` 中最后构造的（282–286），且整个构造在 `fork_rng` 内（280–281）⇒ 所有 v386 张量逐位相同、训练用的全局 RNG 与数据顺序（`order_rng`，712 行）不受影响 ✓。测试 36–47 行与 "函数相等 for CHART_M ∈ {8,14,20}"（54–57）覆盖了这一点。
- `DenseDecoder`：264 行 `CnnTrunk(8, extra_stages=...)` 用默认 `frame_cond=False` ⇒ 未触碰；499 行断言禁止 `--frame-cond` 配 `dense` 臂 ✓。
- 参数增量：3·9·(256+192+128+96+64) = 27·736 = **19,872**；2,491,330 + 19,872 = **2,511,202** ✓（`analyze_v392.py` 头、`chain_day4_V.sh` 第 49/58 行 `--decoder-params 2511202`、测试第 50 行公式一致）。
- 热启动校验（625–626）：`frame_cond` 加入元组 ✓；但沿用 "检查点缺键则默认为 CLI 值" 的写法，v386 检查点 + `--frame-cond` 不会被这条拦下（会被形状不匹配 ⇒ `--allow-partial-init` 拦下）。本批从头训练，不受影响；见 S3。

### 4. 泄漏：新平面是否只是公共帧的函数；`chart_m` 是否为已声明的公共元数据

- `public_extra(count, sy, sx, device)` 的输入只有形状、设备与模块级 `CHART_M`；输出用 `expand(count, ...)` 对所有 agent 复制同一张（测试第 63 行 `p[0] == p[1]`）。`CHART_M` 来源：`public_frame.extent_m`（评估）或 `world_spec.chart_m`（训练）= 户型外接框的函数（v371:34–36），不读取任何 agent 的 `private` / 位姿 / 邻接 ⇒ **不含样本相关观测内容**。
- 公共声明：v371 文档串第 4–5 行 "L = max(8, ceil(max_extent_m + 1.0)) ... (public metadata: the building's bounding box)"；`OVERNIGHT_REPORT` D4-8 末条（第 445 行）如实写明 "图幅取自户型外接框，属于公共元数据假设"；`claude_handoff/README.md` 第 91 行要求公共几何输入必须披露。D5-9 已照此披露 ✓。
- 应如实指出的边界：L 虽是公共帧，但**与户型有关**（外接框尺寸与房间数分层强相关）。对照臂的栅格本身也由同一 L 定义（隐式），本臂则显式拿到 L。这不违反项目约束，但意味着 "8+ 室层增益最大" 既符合 "尺度感知" 也符合 "按户型大小学到的层先验"，机制核对不能二选一（见第 7 项与 S4）。

### 5. `analyze_v392.py` 的键与训练器写出的键

- 训练器 `meta` 顶层写 `frame_cond=bool(args.frame_cond)`（637 行）且 `args=vars(args)` 含 `frame_cond`；分析第 32 行 `bool(meta.get('frame_cond', meta['args'].get('frame_cond', False)))`：本臂 ⇒ `True`；批次 T 的 `run.json`（train_v386 写出，顶层与 args 均无该键）⇒ `False` ✓；身份守卫 58–60 行 `a['frame_cond'] is True and c['frame_cond'] is False`、两臂 `decoder_width == 2.0`（T 的顶层 `decoder_width=2.0`，v386:588）✓。
- 对照路径第 55 行 `runs/v386-decoder/wide-s{seed}`，预算 57,600 ✓；`RECIPE_ARGS`（26 行）不含 `decoder_width` / `frame_cond`，两者由显式条件覆盖 ✓；`train_cache` 字符串两臂同为 `$C/houseexpo_fit-k4800`（两链脚本 COMMON 相同）✓。
- `from analyze_v375 import eq_weight`：`analyze_v375.py` 第 19–27 行，签名 `(rows, k, metric='wall_f1', drop=frozenset())`，返回 `(均值, 计数)` 二元组，第 67 行按二元组解包 ✓。
- `encoder_init_identical`（42–46）比较 `trunk.*` / `head.*` ✓；新增 `val_R8_strata_gain_pp`（73 行）给出逐层增益 ✓；判据 1 pp、FPR 0.3 pp（77–80）✓。
- 注意：`decoder_params`（62 行）只报告不断言，2,511,202 / 2,491,330 没有进入 `ok`（见 S2）。

### 6. `chain_day4_V.sh`

- `bash -n` 通过；第 28–29 行训练器 sha 硬编码 `ef4dc600…` = 本地 `train_v392.py` 的 sha256 ✓（若 S1/S3/S6 改训练器须重新固定并重跑测试；M1/M2 只改链脚本，不影响该值）。
- **第 30–32 行对照存在守卫指向了错误目录（必改 M1）**：T 版第 36 行检查 `runs/v382-long/d4800e4-s$seed/$f`（对照），V 版改成了 `runs/v392-frame/frame-s$seed/$f` —— 这是**本臂自己的输出目录**，发射前必然不存在（若存在，第 33 行又会以 "already exists" 中止）。现状下链脚本**在任何情况下都会在第 31 行 ABORT，什么也不会发射**（失败方向安全，但批次无法进行）。应为 `runs/v386-decoder/wide-s$seed/$f`，与第 49/58 行 `--control-run runs/v386-decoder/wide-s342101/run.json` 和 `analyze_v392.py` 第 55 行一致。
- 第 33 行禁止 `frame-s*` 已存在 ✓；第 34 行 `pgrep -f "^$P scripts/train_v3"` 锚定 ✓（与 T 相同）。
- 第 37 行 COMMON：`--width`-对应的 `--decoder-width 2.0 --frame-cond`、`--expect-updates 57600`、其余与 T 逐字相同 ✓。
- 预检（43–66）：两种子并行 200 步，`--deadline-total $START` 使最终评估跳过但 `done` 行仍打印（训练器 791 行）、`gen` 行 597 ✓；`preflight_check_v386.py ... --mode parallel --width 2.0 --decoder-params 2511202 --frame-cond 1`，对照 `run.json` 为批次 T ✓（trunk/head 参数量与 T 相等是对的，编码器未变）；NOFIT ⇒ 单任务预检 ⇒ SEQUENTIAL / ABORT，与 T 相同 ✓。
- 等待模式：第 82 行 `^$P scripts/train_v392.py .*runs/v392-frame/frame`、第 97 行 `...frame-s$seed`，对应发射命令行第 78/94 行 `$P scripts/train_v392.py --source ... --out runs/v392-frame/frame-s$seed ...` ⇒ 匹配 ✓；预检目录 `pf-par-*` / `pf-solo-*` 不含 `frame-` 前缀的 `runs/v392-frame/frame` 子串 ✓（它们由 `wait` / 前台执行结束）。
- 租约：`runs/v392-frame/lease.json`（70）/ `lease.json`、`lease2.json`（86）✓，算术 6,600 / 7,150 / 7,200 与 T 相同 ✓。**但两处租约的 `"control"` 字段写的是 `runs/v382-long/d4800e4-s<seed> (batch V, ...)`（73、90 行）—— 路径是批次 Q、标签是 "batch V"，两个都错（必改 M2）**；头部第 9 行 "CONTROL = batch V itself" 与第 18 行 "(batch V: 0.1435 s/update ...)" 同属笔误（应为 batch T，数字出自 D5-5）。
- 写入范围：`runs/v392-frame/**`、`tee -a runs/chain_day3.log`（追加）；无 `rm` / 覆盖 ✓；分析输出 `runs/v392-frame/analysis.json`（103）✓。

`preflight_check_v386.py`（重生成版）：`--code-embed`（22、35–36）与 `--frame-cond`（23、37–38）均为可选、`None` 时跳过；`--frame-cond 1` 下 `int(bool(r.get('frame_cond', r['args'].get('frame_cond', False)))) != 1` ⇒ 若本臂 run.json 没写 `frame_cond=True` 即 INVALID ✓（这同时覆盖了 CLI → 全局 → 构造 → meta 的管道）。其余与 T 发射版一致。

### 7. 科学效度：是否公平的单变量对照；应补的保留

**成立的部分**：同种子 ⇒ `fork_rng` + 构造顺序保证编码器 / head 初始权重逐位相同、数据顺序（`order_rng`）相同；解码器初始**函数**与对照相同（零列）；配方参数逐项相同；两臂都是 Q 预算从头训练；预期两臂都是单任务（T 实际为顺序模式，D5-5），即使 V 预检选 PARALLEL，更新数固定、不改变结果口径。

**应在发射前写进 D5-9 的保留**（事后不能补）：
1. **三平面是一捆干预**：块相位 (u, v) 与尺度 L/20 同时加入；正结果不能归因到 "尺度" 还是 "相位"，需后续消融（只 u,v / 只 L）。D5-9 已把 "各层平坦 ⇒ 更像相位" 作为机制核对，但该核对**不具判别力**（见 2）。
2. **L 与分层相关**：8+ 室层 L 最大，但 L 平面也让解码器直接知道 "这是大户型"，可学层先验（墙密度、开口统计）；"8+ 室层增益最大" 与 "尺度感知" 和 "层先验" 都相容。更接近 fp_day5_decoder 候选 2 原话的核对是 "增益随 L 单调（按 L 三分位）"，可事后从 VAL 行的 spec 经 `world_spec` 算 L（S4），但仍只是描述性。
3. **噪声底**：判据 1 pp，而 T 的两种子增益 +2.16 / +1.56、W 约 +1 pp（D5-5、D5-8），CUDA 双线性插值反传不确定 ⇒ 同配置重跑也有 ±1–2 pp 量级波动；两种子通过 1 pp 只是 "可能是杠杆"，不是结论；建议报告时按 "两种子同号 + 8 组配对比较" 的 W 批口径一起读。
4. **容量微增**：+19,872 参数（解码器 +0.8 %），梯度裁剪范数（744 行）包含新列 ⇒ 与对照的有效步长有极小差异；可忽略，但应写明。
5. **零初始化 ≠ 零学习**：新列从第 1 步起有非零梯度且被 Adam 归一化（与 v388 嵌入噪声列不同）⇒ "无增益" 可以读作 "不是杠杆"（与 D5-2 对 v388 的附注相反），这一点对 V 是有利的，可写明。
6. **v293 世界**：L=20 ⇒ 1.0 常数平面只相当于偏置；若日后在 v293 上重复，只有 u,v 起作用。
7. **部署前提**：同 D4-8，L 来自户型外接框的公共元数据假设；可部署替代（观测外接框的 max/min 共识）尚未实现。

## 必改项

**M1 — 对照存在守卫指向本臂输出目录，链脚本必然中止（`chain_day4_V.sh` 30–32 行）。**
现状：`[ -f runs/v392-frame/frame-s$seed/$f ] || { log "ABORT: control runs/v392-frame/frame-s$seed/$f missing"; exit 2; }`。`frame-s*` 是本批将要创建的目录，发射前不存在 ⇒ 第 31 行 ABORT；若存在，第 33 行 ABORT。两种情况下都发不出去。要求：改为 `runs/v386-decoder/wide-s$seed/$f`（与第 49/58 行的 `--control-run` 及 `analyze_v392.py` 第 55 行一致），ABORT 文案同步改为 `control runs/v386-decoder/wide-s$seed/$f missing`。改后用 `bash -n` 并在远端以 "对照存在、`frame-s*` 不存在" 的状态干跑守卫段（例如临时 `exit 0` 于第 35 行前）确认通过第 33 行。

**M2 — 租约与头部的对照描述错误（`chain_day4_V.sh` 73、90 行；9、18 行）。**
现状：`lease.json` / `lease2.json` 的 `"control"` 为 `runs/v382-long/d4800e4-s<seed> (batch V, decoder width 2.0, no frame conditioning, same seed/recipe)`：路径是批次 Q（W=1）的运行，标签写成 "batch V"。租约是写入 `/mnt/datasets` 的溯源记录，事后分析与 OVERNIGHT 报告会引用。要求：改为 `runs/v386-decoder/wide-s<seed> (batch T, decoder width 2.0, no frame conditioning, same seed/recipe)`（90 行用 `$seed`）；头部第 9 行 "CONTROL = batch V itself" → "CONTROL = batch T"，第 18 行 "(batch V: 0.1435 s/update shared vs 0.0828 solo)" → "(batch T: …)"。

M1、M2 只改 `chain_day4_V.sh`，`train_v392.py` 的固定 sha 不变；发射前把新链脚本 sha 记入 D5-9。

## 建议项

- **S1**（溯源）：`train_v392.py` 在 `evaluate` 的每行记录加 `chart_m=sc.get('chart_m', 20.)`（692 行 `rows.append(...)`），`meta` 加训练集 `chart_m` 的 min / max / 直方图（637 行）；这样 "CHART_M 确实逐场景生效" 可在运行产物上事后核验，S4 的按-L 分析也不必再依赖 HouseExpo 环境变量。若采纳须重跑 `test_v392_frame.py` 并重新固定第 29 行 sha；也可留到下一版本。
- **S2**（身份守卫加强）：`analyze_v392.py` 在 `initial.pt` 上再比较解码器：对 5 个加宽卷积断言 `arm[:, :cin] == control` 且 `arm[:, cin:] == 0`，其余 `decoder.*` 张量逐位相等；并把 `decoder_params == 2,511,202 / 2,491,330` 纳入 `ok`。把 "解码器初始除零列外相同" 从单元测试的保证变成对实际运行产物的核验。
- **S3**：热启动校验（625–626）对 `frame_cond` 的缺键默认改为 `False`（`payload['meta'].get('frame_cond', False) == bool(args.frame_cond)`），使 v386 检查点 + `--frame-cond --allow-partial-init` 不能静默通过；本批不受影响。
- **S4**（预注册文本）：把第 7 项保留 1–5 写进 D5-9 与链脚本头部；机制核对补 "按 L 三分位的 R8 增益"（事后从 `final-val.json` 行的 spec 经 `v371.world_spec` 取 L），并明确它是描述性而非判别性。
- **S5**：`test_v392_frame.py` 补一条便宜断言：CPU 上对同一批输入做一步 `stage_loss`，W=2 开 / 关 `frame_cond` 的损失值相等、且所有 v386 张量的梯度逐位相等（新列为零 ⇒ 反传到上游的贡献为零），锁定 "恒等起点延伸到第一步梯度"；再加 `torch.get_rng_state()` 构造前后相等。
- **S6**：`--frame-cond` 与 `--world houseexpo_rect` 互斥断言（或把 L 定义为 `max(extent_m)` 并同步 `world_spec` 的 `chart_xy_m`）；v378 的 `extent_m=[lx, ly]` 下现写法只取 Lx。本批不受影响。
- **S7**（沿袭 CR386 S4 / S9，仍未采纳）：等待循环加 `DT + 300` 看门狗；第 34 行守卫可加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && ABORT`。
- **S8**：`train_v392.py` 文档串第 5 行 "0.4-1.0" 可注明来源（v365 `MAX_EXTENT = 19` ⇒ L ≤ 20，`L ≥ 8`）；链脚本第 103 行把分析完整输出另存 `runs/v392-frame/analysis.txt`（Q 批惯例）。

## 结论

**LAUNCH OK WITH CHANGES（M1、M2）。**

训练器方面：v392 与 v386 的 72 行差异全部属于 `--frame-cond` 旋钮及其 meta / 热启动校验 / 逐场景 `CHART_M`；关闭时与 v386 逐字节等价（`DenseDecoder` 未触碰）。`CHART_M` 在 `main()`（498 行）与 `evaluate`（650 行）都有 `global` 声明，赋值在每次解码器前向之前，`public_extra` 在调用时读模块全局；houseexpo_fit 路径上训练（`v371.world_spec().chart_m`，纯多边形，代价计入预检的 `gen` 时间）与评估（`public_frame.extent_m[0] = L`）取到同一条公共图幅规则的 L，不存在静默默认 20 的情形。三平面只依赖公共帧（形状 + L），对所有 agent 相同；块相位与 `public()` / v371 的 `raster=[y,x]` 约定一致，各级 extent 均为 8 的倍数。加宽恰为 `lift[0]` 与 4 个 `block[0]`，原列与偏置逐位复制、新列为零，加宽的 RNG 消耗发生在全部 v386 抽样之后且在 `fork_rng` 内；参数 +19,872 = 2,511,202 手工核算一致。分析脚本读取的键训练器都写出，对照的 `frame_cond` 正确解析为 `False`，`eq_weight` 签名匹配。预检脚本的 `--frame-cond 1` 同时覆盖了 CLI → meta 的管道。

两处必改都在链脚本的文本层：对照存在守卫被改成了本臂自己的输出目录，使链脚本在任何状态下都会在第 31 行中止、什么也发不出去（M1）；租约与头部把对照写成 "runs/v382-long（batch V）"，溯源记录错误（M2）。改完后可发射；第 7 项的科学保留（捆绑干预、L 与分层相关、1 pp 处于噪声底）请按 S4 在发射前补入 D5-9。

---

## 审查后的修改（主代理，2026-10-24 01:49 UTC，发射前）
- **M1 落实**：对照存在性守卫改回批次 T 的目录（`runs/v386-decoder/wide-s<seed>`）——此前生成脚本的全局替换把它改成了臂自己的输出目录，链脚本会永远中止；审查发现并阻止了一次空转。
- **M2 落实**：lease.json / lease2.json 的 `control` 字段与头部措辞改为批次 T（宽解码器、无条件化）。
- 预注册补充（审查科学告诫）：三个平面是捆绑干预，L 平面与房间数分层相关，所以"8+ 室增益最大"不能单独区分尺度与块相位；若通过，再做 u,v-only 与 L-only 的消融才能归因。
- 发射版本 `chain_day4_V.sh` sha256 前 16 位 `1feda7580d9cf55f`，01:49:12 UTC 启动；`train_v392.py` `ef4dc60082ef9f5f` 未变。
