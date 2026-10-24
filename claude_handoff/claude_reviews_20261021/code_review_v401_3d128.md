VERDICT: LAUNCH OK WITH CHANGES (M1, M2)

# 发射前代码审查：v401 批次 3D-B（128³、0.1 m 体素、固定 512 B 码，`chain_day5_3DB.sh`）

审查者：独立 Claude 子代理，2026-10-24。只读审查，未修改项目内任何文件；所有核对脚本写在本会话的临时目录。本地环境：`D:\anaconda3\envs\jax-gpu`（torch 2.11.0+cu128、numpy 2.2.6 / OpenBLAS）、RTX 4080 Laptop 12 GB（`torch.cuda` 报 11.99 GiB）。远端 uk-linux（RTX 5090 32 GB）不可用，其上的缓存、`runs/v400-3d/*`、`chain_day5_X.sh` 的运行状态无法核实；5090 的速度数字全部由 3D-A 的实测（D6-1：预检 0.34 s/update、实跑 0.36）乘以本地测得的 128/64 比值推算。

## 范围

| 文件 | 行数 | sha256 前 16 位 | 备注 |
|---|---|---|---|
| `scripts/claude_v342/mapping_3d_data_v401.py`（被审） | 261 | `01a9f26445732588` | 与 v400 的 diff：`beams(size)`、`PLACEMENT_SIZE`、`make_fixture` 的参考栅格、`scene_plan(size)`、文档串；栅格 / 传感器 / 森林函数体未动 |
| `scripts/claude_v342/train_3d_v401.py`（被审） | 492 | `b87240f417684f62` | 新增 `space_to_depth` / `depth_to_space` / `pool_ternary`（72–93）、`BASE = 64`、`dilate6`（255）、`surface_f1_tol1`（317–323）、`METRIC_KEYS`、fork 池（370–374）、pooled64 评估（408–431）、新日志行（471/482/486） |
| `scripts/claude_v342/gen_3d_cache_v401.py`（被审） | 39 | `89f82d1e2ab9e1d1` | v400 的顺序生成改为 fork 池 `imap`；manifest 新增 `hops`、`beams` |
| `scripts/claude_v342/test_3d_v401.py`（被审） | 140 | `84689f220f6ea4c4` | 本地通过：首跑全部输出见下，复跑 `time` = 63.4 s |
| `scripts/claude_v342/chain_day5_3DB.sh`（被审） | 107 | `44af0b65178bcb6d` | `bash -n` 通过 |
| `mapping_3d_data_v400.py` / `train_3d_v400.py`（只读参考） | 248 / 403 | `f11828b83c743a60` / `aa238ec3d41114b6` | 与上一审查附记中记录的发射版本一致 |
| `test_3d_v400.py` / `chain_day5_3D.sh`（只读参考） | 98 / 133 | `89a442b7b015dfae` / `e54199f3650a13ed` | 链脚本 sha 与发射记录（`3bf3f21d…`）不同：05:00 UTC 事故后加了 `FORCE_SOLO`，与本批无关 |
| `train_v378.py`（只读参考） | 732 | `db7bcc3528d58624` | `kbit_wire_quantize` 127–141；本地 `run.json` 记录的 `kbit_source_sha256` 与此一致 |
| `chain_day5_X.sh`（只看启动行） | — | — | 第 16 行 `P=/home/yuzhou/anaconda3/envs/jax-gpu/bin/python`，第 45 行 `$P scripts/train_v392.py $COMMON …` |

方法：diff v400→v401 逐行读；写 5 个核对脚本并实际运行：(A) 适配器索引映射 / 块局部性 / 覆盖率块序 / 解码器读出序 / v400 位级等价（含一步训练的梯度）/ 指标语义；(B) 束数、落点跨分辨率、栅格厚度、已知比例分解、泄漏、npz 体积；(C) 本地 GPU 上 64³ 与 128³ 的完整更新（5 hop 图同时存活）峰值显存与耗时、单场景评估耗时；(D) 链脚本 `rate()` 正则 / `UPD` 公式 / 租约 JSON；(E) 用 4 场景、4 agent 的 128³ 微型缓存在本地 CUDA 上**完整跑通训练器两次**（预检路径 100 步、正式路径含最终评估），再把链脚本的 `rate()` 与分析块**原文**套在真实输出上。

## 逐项核对

### 1. 适配器与块结构（`train_3d_v401.py` 72–93、106–182）

- `space_to_depth`（77）：对 s = 2、4 用三重显式循环核对文档串的索引映射 `channel = c·s³ + dx·s² + dy·s + dz ↔ voxel (s·i+dx, s·j+dy, s·k+dz)`，0 处不符；`depth_to_space(space_to_depth(x)) == x` 与 `space_to_depth(depth_to_space(z)) == z` 均成立；s = 1 时两者返回**同一对象**（脚本 A）。
- 解码器输出顺序：`depth_to_space(...)[:, 0].flatten(1)` 与目标 `own.reshape(n, -1)` 的 x-major 顺序逐元素相等，抽查平坦索引 `X·S² + Y·S + Z` 正确（脚本 A）。
- `PatchTrunk3D`：`bs = (BASE//BX, BASE//BY, BASE//BZ) = (8, 8, 16)` 个计算格，×s = (16, 16, 32) 体素 = 128/8 × 128/8 × 128/4 ✓；块边 16 是 s = 2 的倍数（256³ 时 32 是 4 的倍数），子体素组不会跨块。体素级局部性实测：块 k = 117 = (3,5,1) 六个面**外侧**各翻转一个体素，块 117 的 128 维特征逐位不变；六个面**内侧**各翻转一个体素，块 117 改变而其余 255 块与另一 agent 的全部特征逐位不变（脚本 A）。
- `OwnedHead` 的覆盖率（145）：`avg_pool3d` 核 (16,16,32) 在全分辨率上做，`flatten(1)` 的顺序 = `ix·BY·BZ + iy·BZ + iz`；单个已知体素 (63,95,63)→k=117、(64,95,63)→149、(63,96,63)→121、(63,95,64)→118、(0,0,0)→0、(127,127,127)→255，与 `unfold` 的块序一致（脚本 A）。
- 块读出（171、181）：输出通道 ch = c·8 + dx·4 + dy·2 + dz 的偏置加 100，恰好且只有奇偶性 (dx,dy,dz) 的 64³ 个体素的 logit c 变化；扰动一个计算格 (10,20,30) 的最后一级特征，只有 (20..21, 40..41, 60..61) 这 8 个体素变化（脚本 A）。`test_3d_v401.py` 第 116 行把这条写成注释而没有断言（S4）。
- 128³ 模型形状：`net[0].in_channels = 27`、`output.out_channels = 16`、解码器计划 [(32,32,16),(64,64,32),(64,64,64)]、末级 27 通道（与 v400 相同）。

### 2. "v401 @ 64³ == v400 位级相同"（自写脚本，不依赖测试 C）

- 两个种子（342101 / 342202）的 `weights_hash` 相等（`266b94cbf4ecee68` / `d362776667e78b24`），参数均为 1,124,867。
- TRAIN 11 的 64³ fixture 在 v401 与 v400 下 `private / reached / edges / positions` 完全相等；用同一 fixture 做**一次完整训练更新**（source → rollout → 5 hop 损失 → backward）：loss 1.386038 == 1.386038，**全部参数的梯度 `torch.equal`**，`feats / mu / q / mask` 相等；hop 0 / 8 的预测与 v400 五个指标键逐项相等；`skeleton_mask(64)` 相同（脚本 A）。测试 A1 另在 TRAIN 7 / VAL 1,000,000 上验证了 fixture 等价。
- 128³ 模型：trunk 609,504 + head 16,770 + decoder 517,129 = 1,143,403 参数，比 64³ 多 18,536（+1.65 %）：首卷积 27 入通道、末 1×1 卷积 16 出通道。这意味着"同一个卷积网络"只在**架构与计算网格**意义上成立，权重数与同种子的初始权重都与 3D-A 不同（M2 (i)）。

### 3. 指标（`train_3d_v401.py` 255–338）与输出键

- `surface_of`（262–267）：与显式 numpy 6-邻域（只认自由邻居，未知 / 出界不算）在随机三值体上逐元素相等；`dilate6`（255–259）与显式 6-邻域膨胀相等（脚本 A）。
- `pool_ternary`（88–93）：4³→2³ 的手工构造 `[[[0,1],[-1,-1]],[[-1,-1],[-1,0]]]` 相等，即 max(−1/0/1) = "任一子体素已知即已知、任一占据即占据"；s = 1 返回原对象。测试 B 另与显式循环比对了随机体。
- 池化骨架：native 128 占 .1767、pooled-128 占 .2310、native 64 占 .1483；楼板 z 行 native 64 [0,16,32,48]（各 1 层），pooled-128 [0,1,16,17,…]（各 2 层），native 128 [0,1,2,32,33,34,…]（各 3 层）；外墙 native 64 [0, 62, 63]，native 128 [0,1,2,125,126,127]，pooled-128 [0,1,62,63]。所以 pooled64 的可变结构 F1 剔除的骨架比 3D-A 厚一层——与头部"pooled64 永不与 3D-A 一比一"的声明一致。
- `metrics3d` 的 None / 零边界：目标全未知 → 五个 F1 / acc 为 None、FPR = 0（预测全未知）或 1（预测全已知）；目标全自由且预测全自由 → F1 全 None、acc 1、FPR None；一方为空另一方非空 → 各 F1 = 0.0、tol1 = 0.0（脚本 A）。`agg` 对 `[全 None 行, 部分行]` 给出 `{occ .5, surf .25, tol1 .75, var None, acc 1, fpr 0}`，键序 == `METRIC_KEYS`；`fmt` 对 None 输出 `n/a`。
- **tol1 的语义有一个方向性缺陷（M1）**。代码（317–323）先用目标已知范围裁掉预测（`po = (p==1) & known`），再剔除目标内部，然后才做一体素容差。真实的 stream target 是"首击层"：楼板顶层已知占据、其上自由已知、其下未知。本文构造该目标（z=12 占据、z≥13 自由、z≤11 未知）：
  | 预测 | occ F1 | surface F1 | **tol1（现行）** | tol1（已知范围先膨胀 1 体素） |
  |---|---|---|---|---|
  | 精确 {12} | 1 | 1 | 1 | 1 |
  | 向自由侧偏 1 体素 {13} | 0 | 0 | **1** | 1 |
  | 向实体侧偏 1 体素 {11} | 0 | 0 | **0** | 1 |
  | 加厚 {12,13} | .667 | .667 | 1 | 1 |
  | 偏 2 体素 {14} / 缺失 | 0 | 0 | 0 | 0 |
  即**向内的一体素抖动在 tol1 下与"结构缺失"不可区分**（预测体素落在目标未知区，在容差之前就被掩掉了）。头部第 17–20 行的读法"tol-1 R8 接近 3D-A 的原生 .916/.927 ⇒ 残差是抖动"只在一个方向上成立，而反向推论"tol1 不接近 ⇒ 信息丢失"不成立。附带：对完全已知的 3 层厚实体，"剔除目标内部"又让偏 1 体素的 tol1 只有 .667、只画内层 .000（脚本 A 的 3 层楼板表）；在 stream target 里内部体素只占目标占据的 3.5–6.3 %（TRAIN 7：64³ K0 6.3 % / R8 5.5 %，128³ K0 3.5 % / R8 3.7 %），所以这条是次要的。
- 键与链脚本：本地完整跑通后 `final.json` 键 = `[updates, weights_sha256, seconds, val, val_pooled64, fresh, fresh_pooled64]`，`val` / `val_pooled64` 各含 '0','1','2','4','8' × 六个 `METRIC_KEYS`，没有 None；`history.json` 每条含 `validation` 与 `validation_pooled64`；`validation/upd-*.json` 的 `rows[i]` 含 `hops` / `pooled64` / `mask_fraction` / `q_mean`；`run.json` 含 `updates`、`base 64`、`sub 2`、`beams 65536`、`version 401`、`kbit_source_sha256`。链脚本分析块（78–105）**原文**套在该 `final.json` + 伪造的 3D-A 参考上成功生成整行（`tol1_minus_ref_R8_surface_pp` 等），`round(None, 4)` 会抛 `TypeError` 落入 except → 该种子 INCONCLUSIVE（与 3D-A 相同，实际不会触发）。预检路径（`--deadline-total 0`）的 `final.json` 只有 `updates / weights_sha256 / seconds`，无 `val` ✓。

### 4. 数据（`mapping_3d_data_v401.py`）

- `beams(size)`：16384 / 65536 / 262144；非法尺寸抛 `ValueError`。5 m 处射线间距 .1385 / .0692 / .0346 m，恒为体素的 0.69 倍（脚本 B）。`observe` 在 128³：ds = 0.05 m、100 步、最远已知体素中心 5.082 m（64³ 为 5.157 m）。
- 落点：TRAIN 0 / 1 / 1599、VAL 1,000,000 / 1,000,001 / 1,000,031 / 1,000,063 七个场景，v401（参考栅格 64）的位置与 v400 逐位相等，`occ64` 与 v400 相等，`occ128` 与 `v400.rasterize(spec,128)` 相等（栅格函数未动）；若直接在 128 栅格上落点，7 个场景中 2 个会得到不同位置——`PLACEMENT_SIZE` 的设计是必要的。`world_id` 仍为 `v400:…` ✓。
- 128³ 栅格厚度（TRAIN 7，按薄轴统计）：0.2 m 的隔墙 / 走廊墙 36 个全部 2 层，0.3 m 的楼板 / 外墙 8 个全部 3 层；64³ 下 0.3 m 实体 6 个 1 层、2 个 2 层（上一审查记录的东 / 北墙不对称），0.2 m 实体全部 1 层。走廊中心柱楼板行 [0,1,2,32,33,34,64,65,66,96,97,98]；`skeleton_mask(128)` 楼板行相同、占 .1767。
- 泄漏：128³、65,536 束、4.4 m 外的轴对齐平面墙（2 层或 1 层）墙后自由体素 = 0；对角 1 体素薄墙（真实世界不存在）128³ 泄漏 22,357 体素 = 立方体的 1.07 %，64³ 为 2,751 = 1.05 %，两分辨率行为一致。
- **已知比例下降（.042 → .035）的分解**（脚本 B，4 个场景）：

  | 场景 | 64³ | pooled-128 | 128³ | 64 已知而 pooled 未知 | pooled 已知而 64 未知 | 孔洞/已知 64 → 128 |
  |---|---|---|---|---|---|---|
  | TRAIN 7 | .0422 | .0388 | .0347 | 9.8 % | 1.7 % | .0018 → .0010 |
  | TRAIN 0 | .0507 | .0460 | .0427 | 11.6 % | 2.5 % | .0011 → .0007 |
  | VAL 1,000,000 | .0489 | .0455 | .0410 | 8.4 % | 1.4 % | .0015 → .0008 |
  | VAL 1,000,001 | .0506 | .0446 | .0407 | 16.4 % | 5.2 % | .0020 → .0015 |

  ("孔洞" = 有 ≥ 5 个已知 6-邻居的未知体素数 / 已知体素数。) 两个机制各占一半：64 → pooled-128（−8 %）是**栅格几何**的可见性差异（64 栅格实体更薄、缝更粗，射线看得更远），pooled → native（再 −10 %）是**体素尺寸**本身的边界膨胀。孔洞率在 128³ 反而更低，说明束数 ×4 没有让私有地图更稀疏。R8 并集目标的已知比例 .2647 → .2260（TRAIN 7）。结论：不是隐藏变量，而是分辨率变量的固有后果；但它改变了指标的总体（目标已知体素少 15–18 %；occ 中 6-表面份额 64³ .79 → 128³ .51），应作为协变量写进预注册与报告（M2 (v)、S5）。
- 生成成本：`make_fixture` 64³ 0.7–0.8 s，128³ 8.3–12.0 s（本地单进程）；128³ 的 npz 压缩后仅 0.04 MiB（原始 16 MiB），`np.load` + 解压 0.014 s → 1,600 个文件约 22 s、磁盘 0.1 GiB；主存 1,600 × 16 MiB = 25.0 GiB。

### 5. 显存与速度（脚本 C，本地 4080，同一代码）

| | 64³ (= v400) | 128³ (v401) | 差 |
|---|---|---|---|
| 完整更新 N=8（5 hop 图存活）峰值 allocated / reserved | 10.90 / 12.97 GiB | **12.35 / 14.09 GiB** | +1.45 GiB |
| 完整更新 N=4（不溢出）耗时 | 0.497 s | 0.545 s | **×1.096** |
| 完整更新 N=8 耗时（12 GB 卡溢出到共享内存，不代表 5090） | 29.1 s | 30.7 s | ×1.057 |
| 解码器单 hop 前向+反向 N=8 | 0.186 s | 0.205 s | ×1.10 |
| 编码器前向+反向 N=8 | 0.030 s | 0.049 s | |
| 可分页主存 → GPU 一次 `to(device, non_blocking)` | 0.1 ms (2 MiB) | 0.9 ms (16 MiB) | |
| 评估一个场景 N=8（native + pooled64）| 0.37 s | 0.54 s | 32 场景 12 → 17 s |

- 64³ 的 10.90 GiB 与上一审查的 10.86 一致，说明测法可比。**128³ 单进程约 12.4 GiB allocated / 14 GiB reserved，在 32 GB 的 5090 上独跑绰绰有余**（两个并行也装得下，但本批按设计只独跑）。
- 速度：128³ 比 64³ 慢 6–10 %；3D-A 在 5090 上 0.34（预检）/ 0.36（实跑）s/update ⇒ 128³ 预期 **0.36–0.40 s/update**。
- 加载：68 个 128³ fixture（4 个缓存校验 + 64 个面板）在 8 进程池里约 9 轮 × 8–12 s ≈ 70–110 s，加 22 s 缓存解压 ⇒ "gen" ≈ 100–160 s；验证 32 场景在 5090 上估计 < 30 s ⇒ `rate()` 的 30 s 下限生效（本地真实预检日志里的估计值就是下限：`0.5800 37 30`）。
- 预算：`UPD = ⌊(6000/1.05 − load − 3·val)/rate⌋₅₀₀`，16,000 只在 rate ≤ 0.316–0.330 时可达（脚本 D，load 250–300、val 60–120）；按 0.36–0.40、load 130、val 30 ⇒ **预期 14,000–15,000**，高于 12,000 的"预算混杂"线，但**不会与 3D-A 的 16,000 对齐**（M2 (iii)）。训练循环实际含 4 次循环内验证 + 2 次最终评估 = 6 次而非 3 次（S2），128³ 下每次 < 30 s，被 1.05 的余量与 6,600 / 7,150 截止线覆盖：例如 UPD 14,500、rate 0.37 ⇒ 循环约 5,600 s，速率漂移容忍到 0.44（+18 %；3D-A 独跑漂移 +6 %）。
- 主存：25 GiB 可分页 + 面板 ≈ 1.4 GB 临时（68 × ~20 MiB 经管道回传后 `del gen`）在 125 GB 上无压力；可分页内存的 `non_blocking=True` 实际是经暂存区的同步拷贝，语义正确、每步 ~1 ms。

### 6. fork 池（`train_3d_v401.py` 370–383、`gen_3d_cache_v401.py` 31–36）

- 时序：池在 `torch.manual_seed` / `torch.device('cuda')` 之后、任何 CUDA 操作之前创建；本地验证这两个调用都**不**初始化 CUDA（`torch.cuda.is_initialized() == False`），且 26.8 GB 缓存在池之后才加载（fork 时父进程很小）✓。子进程只跑 numpy（`_fixture` 是模块级函数，fork 下直接继承）。`torch.set_num_threads(4)` 在 fork 前执行，但父进程在 fork 前没有跑过任何 torch 算子或大规模 BLAS，所以不存在已建好的 OpenMP 线程组，fork 后死锁的前提不成立；本地 numpy 是 OpenBLAS（自带 atfork 处理），服务器 conda 环境的 BLAS 未核实（S1）。
- 确定性：`make_fixture` 由 `(split,index,size,agents)` 决定，与进程无关；缓存四行 sha 校验在本地顺序路径（`--gen-workers 0`，Windows 无 fork）通过。`pool.map(..., chunksize=1)` 保序；`imap` 保序且第 36 行再断言 `[r['spec']] == plan`。manifest 含 `version/size/agents/scenes/hops/beams/data_module_sha256/rows`，本文按同一布局造的微型 manifest 被训练器的全部断言接受。
- 失败模式：子进程异常（如 `could not place agents`）会在 `pool.map` 处重新抛出 → 训练器崩溃 → 预检无 `history.json` → `rate()` 打印 `nan 0 0`（本文删掉 history.json 实测）→ `UPD = 0` → "ABORT INCONCLUSIVE" 跳到下一种子 ✓。**挂起**（任何 fork 相关死锁）则链脚本会无限等待预检——预检命令没有 `timeout`（S1）。
- 缓存生成时间估计：1,600 × 8–12 s / 8 进程 ≈ 27–40 min（`--workers 12` 约 20–27 min）。

### 7. `chain_day5_3DB.sh`

- `bash -n` 通过。等待循环（41）：`chain_day5_X[.]sh` 的方括号避免匹配自身 pgrep；`^$P scripts/train_v392.py ` 与 `chain_day5_X.sh` 第 45 行的启动命令行锚定一致；随后 `^$P scripts/train_` 兜底（42）。缓存检查（37–38）核 `scenes/size/version/beams/data_module_sha256`，sha 在运行时计算；3D-A 参考检查（39）核 `updates == 16000` 且含 `val`；历史保护（40）在等待前检查 `b128-s*`，训练器自身 `mkdir(exist_ok=False)`，`pf-solo-*` 带时间戳，租约只在 `UPD ≥ 4000` 后写 ⇒ 没有能覆盖历史的路径。
- `rate()`（45–56）：对训练器真实格式串（含 `grad {...}` 字典、`lr 7.50e-05`、`mask 0.31 355s`）的合成日志与**本地真实预检日志**都解析正确（`gen 37`、`t [68, 97]`、rate 0.5800、val 取 30 s 下限）；新的 `[3d] VAL u100 pooled64: …` 行不含 `upd`，不干扰；`upd500` 不会被当作 `upd50`；崩溃日志 → `nan 0 0`。
- `UPD`（61）：`math.isfinite` 保护、500 取整、16,000 封顶；500 的倍数必被 4 整除（`$((UPD/4))` 得 125 的倍数）。租约 heredoc 用 `RS=0.4123 LS=287 VS=95 UPD=12500` 等代入后 `json.loads` 合法，18 个键。
- 分析块（78–105）：读的键 `val['8']['surface_f1_tol1']`、`val_pooled64['8']['surface_f1']`、`run.json['updates']` 与训练器写的完全对应（第 3 项实测）；三分判读 .85/.88/.80/.84 与 .75/.65 在头部（11–15）、两处租约 `expectation`（69）与分析（98）三处一致；`budget_confounded = updates < 12000` 与头部 23–24 一致；任一种子缺文件 / 不完整 → INCONCLUSIVE 并仍写 `analysis.json`。
- 其余：`LOG=runs/chain_day3.log` 沿用旧名；预检与正式运行各自重新生成 68 个面板 fixture（各约 2 分钟），预检不计入租约 ✓。

### 8. 科学 / 预注册

- 判据在运行前写定且三处一致 ✓；"预算混杂 < 12,000" 规则有 3D-A 曲线（12k→16k 平坦）支撑 ✓；K0 / 融合核对 / FPR 与 3D-A 同口径 ✓。只画骨架的参照在 128³ TRAIN 7 上为表面 .683 / 可变 .000（测试输出），"不迁移" 的表面 F1 下限 .75 只高出 7 pp，但可变结构 F1 的 .65 / .80 门槛承担了判别力 ✓。
- 公平性：场景相同（`world_id` 不变、VAL 1,000,000+ 相同）、落点 / 森林 / reached 相同（7 个场景实测）、种子相同；**更新数不会相同**（预期 14–15k vs 16k）；**初始权重不同**（层形状改变使同种子的 RNG 消耗不同）；网络多 18.5k 参数。
- "beams ×4" 不是隐藏变量——头部已声明，且是让"射线间距 / 体素 = 0.69"保持不变的必要协变；实测 128³ 孔洞率更低。但"唯一变量是分辨率"应改为"分辨率，传感器角密度按体素匹配"。
- **负结果的可解释性**：v401 的解码器把每个 0.2 m 计算格的 27 维特征**线性**读成 2×2×2 子体素的 16 个 logit（无任何 128³ 卷积）。若结果是"不迁移"，无法区分是 512 B 码在 8,192 体素/坐标下信息不足，还是固定计算网格的块读出容量不足；头部第 13–14 行"next lever: code layout or semantics"把原因预先归到码上，是过度归因（M2 (ii)）。正结果则无歧义。
- tol1 读法见第 3 项（M1）；pooled64 的"仅报告、不比较"写法恰当 ✓。

## 必改项

**M1 — `surface_f1_tol1` 对向内的一体素抖动计 0，与"结构缺失"不可区分；二级读法因此只有单向有效（`train_3d_v401.py` 317–323、文档串 22–24；`chain_day5_3DB.sh` 17–20；`test_3d_v401.py` 65–71）。**
现状：先按目标已知范围裁掉预测再做容差，预测落在目标未知的实体层（首击层下方）时 tol1 = 0（本文构造的真实 stream target：向自由侧偏 1 → tol1 = 1，向实体侧偏 1 → tol1 = 0）。要求：(i) 把 tol1 的预测集合改为"已知范围膨胀一体素内的预测占据"——`known1 = dilate6(known[None], size)[0]; po1 = (p == 1) & known1`，precision = |po1 ∩ dilate6(s)| / |po1|，recall = |s ∩ dilate6(po1)| / |s|（不再剔除目标内部；`surface_f1` 本身**保持不变**以与 3D-A 可比）；本文验证该定义下向内 / 向外偏 1 都得 1、偏 2 与缺失得 0、单层墙的测试用例结果不变、完全已知的 3 层楼板偏 1 从 .667 变 1；(ii) `test_3d_v401.py` B 段加该首击层目标（顶层占据、上方自由、下方未知）的四个断言；(iii) 文档串与链脚本头部把 tol1 定义改写为上述口径，并把二级读法补成双向："tol1 R8 接近 3D-A 原生 R8 ⇒ 残差是 ±1 体素抖动；tol1 也明显低 ⇒ 结构缺失"。若主代理决定不改指标，则 (iii) 必须改为只保留单向读法并写明"向内抖动按缺失计"。

**M2 — 预注册文本需要四处修正，使结果可解释（`chain_day5_3DB.sh` 1–28、租约 `expectation` 69；D6-2 报告）。**
(i) "the SAME convolutional network" → "same architecture and schedule on the 64³ compute grid; first / last convolutions widen to 27 in / 16 out channels (+18,536 parameters, 1,143,403 vs 1,124,867; initial weights differ from 3D-A for the same seed)"。(ii) 把"does NOT scale … (next lever: code layout or semantics)"改为"does not scale under the fixed-compute block readout; the cause (512 B code vs. the 27-feature → 16-logit linear readout per 0.2 m cell) is NOT identified by this batch and needs a decoder-capacity control before touching the code"。(iii) 写明预算规则下 16,000 不可达（需 rate ≤ 0.33 s/update；预期 0.36–0.40 ⇒ 14,000–15,000），比较将在 ≤ 15,000 vs 16,000 之间进行，并引用 3D-A 曲线 12k→16k 平坦作为可接受的理由。(iv) "One variable: the voxel size" → "voxel size, with the sensor's angular density matched to the voxel (beams ×4); measured porosity does not increase"。(v) 记录协变量：每 agent 已知比例 .042 → .035（pooled .039）、R8 并集 .265 → .226、occ 中 6-表面份额 .79 → .51，说明原生 surface F1 与 3D-A 的差不能全归于码。全部为文本改动，改后 `bash -n` 并把新 sha 记入报告。

## 建议项

- **S1**（鲁棒性）：第 59 行预检加 `timeout -k 60 1800`，使任何 fork / 池挂起都落到 `nan 0 0 → ABORT INCONCLUSIVE` 而不是让链无限等待；可顺带在 `fixtures()` 里对子进程设 `OMP_NUM_THREADS=1`（无害）。
- **S2**（预算口径）：实际评估 6 次（4 次循环内 + final-val + final-fresh），公式用 `3·val`；128³ 下每次 < 30 s 无实害，但头部规则与公式应写 `6·val`（或 `4·val` 并说明最终两次由 7,150 截止线覆盖）。
- **S3**：`public3` 的第 4 通道 `scale = 1/size`（102 行）在 128³ 模型里是常数 1/128（64³ 为 1/64）；计算网格固定后它只标记"我是 128 模型"，无害，但应在文档串里注明或改为 `1/BASE`。
- **S4**（测试）：把 `test_3d_v401.py` 第 116 行的注释改成断言（扰动一个计算格只动 8 个体素）；加 M1 的首击层用例；加"6 个面内外翻转"的体素级局部性断言（现只测随机噪声）。
- **S5**（报告）：发射后用确定性 VAL fixture 离线计算 32 场景的已知比例（native / pooled）与只画骨架参照（现只有 TRAIN 7 的 .683），与 3D-A 的同类数字并排写入 D6-2。
- **S6**（溯源）：租约加 `expected_updates_range` 与 `preflight_gen_seconds`；`LOG` 改为 `runs/chain_day5.log`。
- **S7**（缓存生成）：`gen_3d_cache_v401.py` 用 `--workers 12`（14 CPU）可把 1,600 场景从 ~30–40 min 压到 ~20–27 min；manifest 已含 `beams/hops` ✓。
- **S8**（后续科学）：若结果为 partial / 负，先做解码器容量对照（同码、末级在 128³ 原生卷积，或 8×8×8 = 512 坐标）再动码；K0 相对 3D-A K0（.938/.945）的下降量可先行分离"码 + 读出"与"融合"。

## 结论

**LAUNCH OK WITH CHANGES（M1、M2）。**

代码层面没有发现错误：space_to_depth / depth_to_space 的索引映射、互逆与 s = 1 恒等经显式循环核对；编码器在 128³ 的块局部性到体素级成立，`unfold` 块尺寸、覆盖率池化与解码器读出三处块序一致；v401 在 64³ 与 v400 不仅权重相同，一次完整更新的 loss 与全部梯度也逐位相同；`surface_of` / `dilate6` / `pool_ternary` 与显式实现一致；训练器在本地 CUDA 上以 128³ 完整跑通，`history.json` / `final.json` / `run.json` 的键与链脚本分析块逐项对应，`rate()` 在真实日志上解析正确，租约 JSON 合法，历史不可覆盖。数据侧落点 / 森林 / reached 跨分辨率逐位相同，128³ 栅格厚度（楼板 3、隔墙 2、外墙 3）与骨架正确，平面墙无泄漏，孔洞率更低。显存实测 128³ 单进程 12.35 GiB（64³ 10.90），独跑 32 GB 无压力；速度慢 6–10 %，预期 5090 上 0.36–0.40 s/update、14,000–15,000 次更新。

两处必改均为低成本：M1 让 tol1 对向内抖动也宽容（两行代码 + 一个测试用例），否则这个专为"抖动 vs 丢失"设计的二级指标只能单向解读；M2 修正预注册文本（网络并非完全相同、负结果无法把原因归到码上、16,000 不可达、传感器协变、总体偏移协变量），否则结果的归因会在运行前就被写偏。改完后即可发射。
