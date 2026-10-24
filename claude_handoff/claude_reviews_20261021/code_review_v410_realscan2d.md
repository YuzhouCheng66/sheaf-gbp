VERDICT: LAUNCH OK WITH CHANGES (M1, M2, M3, M4)

# 发射前代码审查：v410 批次 RS-A（Replica 真实扫描楼层切片，ZS vs FT，`chain_day6_RS.sh`）

审查者：独立 Claude 子代理，2026-10-24（UTC 22:25 起）。只读审查：项目内与 uk-linux 上的被审文件均未修改；全部核对脚本与日志写在 uk-linux 的 `runs/review_v410/`（本地副本在本会话临时目录 `remote_logs/`），只用 CPU（`nice -n 10`，≤ 6 线程，`CUDA_VISIBLE_DEVICES=` 置空），未发信号、未动任何进程。审查期间 3D-B s2 于 22:28 UTC 结束（`chain_day3.log`），未与之冲突。

**一句话**：管线代码（训练器插线、缓存、链脚本机制）是对的，但当前版本**不能原样发射**：(1) 预处理把 Replica 网格的 z 轴当作铅垂，room_2 实际倾斜 8.7°（"0.62 m 楼面"即其伪影），apartment_0/1 倾斜约 1°；(2) VAL 的 office_3 墙体与图框成 ~52°，HouseExpo 基线在它上面 ZS 只有 .18，转正后是 .57–.61——FT 用的是均匀随机偏航训练，仅"学会旋转"就能让等权 G 超过 +3 pp；(3) 再多训一个 HouseExpo 周期本身就让 Replica ZS 涨 +0.8~+1.6 pp，FT − ZS 混入了"多训练"。四项必改都便宜（重跑预处理 + 缓存约 30 min，加一个对照臂约 25 min/种子，均在租约内）。

## 范围

| 文件 | 行数 | sha256 前 16 位 | 备注 |
|---|---|---|---|
| `scripts/claude_v342/prep_realscan_base_v410.py`（被审） | 119 | `8538b23105f86f4c` | uk-linux 同 sha；本文用它**逐位复现**了 9 个已存栅格（见 1.2） |
| `scripts/claude_v342/mapping_realscan2d_data_v410.py`（被审） | 176 | `5c2a3cf901bc5ed1` | uk-linux 同 sha；= 缓存 manifest 的 `data_module_sha256` |
| `scripts/claude_v342/train_v410.py`（被审） | 802 | `27878023c90cc0d5` | uk-linux 同 sha；与 v392 的 diff 见第 3 项 |
| `scripts/claude_v342/gen_realscan_cache_v410.py`（被审） | 74 | `0e9c2adb599b4e29` | uk-linux 同 sha |
| `scripts/claude_v342/test_realscan2d_v410.py`（被审） | 99 | `b573faa4ea522f4f` | uk-linux 同 sha；本文在 uk-linux 实跑通过（单线程 88 s） |
| `scripts/claude_v342/chain_day6_RS.sh`（被审） | 99 | `d64b01f6fa3e7bb7` | **尚未部署**到 uk-linux `scripts/`；`bash -n` 通过 |
| `train_v392.py` / `gen_hx_cache_v378.py`（参考） | 795 / 71 | `ef4dc60082ef9f5f` / `7fa0a45d552471b0` | 两端同 sha；v392 sha 与 `chain_day5_X.sh` 第 23 行钉死值一致 |
| `mapping_houseexpo_data_v365.py` / `_v371.py`（参考） | 208 / 122 | `e1e6b5aab5105190` / `6abdc221f01c323b` | 两端同 sha |
| `chain_day5_X.sh` / `chain_day5_3DB_s2.sh` / `fig_replica_preview.py`（参考） | 79 / 126 / 251 | `54df3fc67dc9001f` / `99e4a6e01030db8b` / `be2e118a6645bcbf` | |
| 数据：`realscan-v410/replica-base/index.json`、`replica-train-k4800/manifest.json` | — | `68543c7ada259aec` / `15ae800843c7395a` | 19 个楼层栅格；缓存 4,800 npz + manifest，29 MB，生成 839 s |

方法：逐行读 6 个被审文件与参考文件；本地 `diff`；在 uk-linux 上写并运行 10 个只读核对脚本：**P** `rv_prep_check.py`（PLY 解析对照 trimesh、18 个场景的竖直方向 / 重力 / 语义楼面、9 个楼层的栅格复现、下射线命中类别、仅由地面构成的障碍、密度复采样、上射线覆盖）；**M** `rv_map_check.py`（旋转往返、独立 `affine_transform` 交叉验证、膨胀 k 与面积、`metric_trajectory` 文本 diff、连通分量 / 主墙向、**原样执行训练器的缓存断言**、16 个随机缓存行跨进程复现、加载 / 生成计时）；**L** `rv_leak2/3.py`（栅格级泄漏检测，含正对照）；**Gl** `rv_glass.py`（`glass.sur` 玻璃面穿过自由空间的长度）；**B** `rv_boundary.py`（自由区边界是障碍还是扫描空洞）；**Z** `rv_zs_eval.py`（按 `train_v410.evaluate()` 逐行复刻 hop 0/8，在 CPU 上评估 c3/c2 两种子共 4 个 HouseExpo 检查点的 Replica VAL 与 fresh 全面板）；**R** `rv_zs_rot.py`（同一评估，把 VAL 场景转一个偏航）；**T** 原样跑 `test_realscan2d_v410.py`。Z 的正确性：它在 smoke 跑的 4 个 fixture 上逐值复现 `runs/v410-realscan/smoke-zs-cpu-s342101` 的结果（R8 .5201/.4704、K0 .5938/.5413；fresh .5189/.4734、K0 .5976/.5625）。

## 逐项核对

### 1. `prep_realscan_base_v410.py`

**1.1 PLY 解析（25–36）——正确。** room_2 与 office_3 的 `mesh.ply`：头部 `property list uint8 int vertex_indices`、顶点 27 B、全四边形；按头部算的字节预算 = 文件大小（31,788,451 / 52,214,939 B，逐字节吻合）；解析器的顶点与 `trimesh 4.12.2`（`process=False`）**逐位相等**，三角形集合相等（拆分后 1,444,796 / 2,371,984 个，唯一集合 1,444,621 / 2,371,430，两边相同），总面积相等（118.930 / 193.539 m²）。索引范围 [0, nv−1]。另外 18 个场景的 `habitat/mesh_semantic.ply` 与 `mesh.ply` 顶点逐位相同、四边形顶点序相同（P），因此可以把语义类别逐面对到被审网格上——下面的语义核对都用它。四边形拆成 (0,1,2)+(0,2,3) 保持绕向 ✓。

**1.2 已存栅格可复现 ✓。** 用被审代码（同一 sha256 播种的 RNG、同一下射线）重算 room_2、apartment_2、hotel_0、office_3、office_4、room_0、apartment_0 f0/f1、apartment_1 九个楼层，`obstacle` 与 `interior` 与 `replica-base/*.npz` **逐位相等**（P）。所以下面对已存栅格的结论就是对这份代码的结论。

**1.3 竖直方向：代码只判"符号"，但真正的问题是"倾斜"（M1）。**
- 符号启发式（89–99）在 18 个场景上**全部判 "kept"**（P），与数据自带的 `habitat/info_semantic.json` 的 `gravity_dir`（全部 ≈ (0,0,−1)）以及语义楼面在底、天花在顶（如 apartment_2 楼面中位 0.057 m、天花 2.758 m）一致。文档串"its sign varies"不成立；而且这个启发式对带天花的网格本身不可判别（倒置后最低的水平面是朝上的天花，同样"朝上"），只是从未触发。
- **倾斜**：`gravity_dir` 给出的倾角与语义楼面平面拟合一致（P）：

  | 场景 | `gravity_dir` 倾角 | 楼面平面拟合 | 楼面在其包围盒上的高差 | 语义楼面 z 5/50/95 % | prep `floor_z` |
  |---|---|---|---|---|---|
  | room_2（TRAIN） | **8.72°** | 8.72° | **−0.09 … 0.99 m** | 0.04 / 0.549 / 0.851 | **0.625** |
  | apartment_0 f0（TRAIN） | 1.31° | 1.45° | −0.02 … 0.23 m | 0.034 / 0.105 / 0.213 | 0.075 |
  | apartment_0 f1（TRAIN） | 1.31° | 1.11° | 2.79 … 2.96 m | 2.822 / 2.878 / 2.95 | 2.875 |
  | apartment_1（TRAIN） | 0.94° | 0.97° | 0.02 … 0.23 m | 0.048 / 0.129 / 0.193 | 0.125 |
  | apartment_2（**VAL**） | 0.50° | 0.50° | −0.01 … 0.11 m | 0.015 / 0.057 / 0.087 | 0.075 |
  | 其余 14 个场景 | 0.05–0.39° | 0.06–0.54°（office_1 的拟合 6.5° 被 0.48 m 处一块标为 floor 的面拉偏，其 `gravity_dir` 倾角只有 0.06°，非倾斜） | ≤ 0.09 m | | |

  所以 **room_2 的 0.62 m "楼面"不对**：它是倾斜 8.7° 的地面在 5 cm 直方图上的峰；网格最低角被平移到 0，地面横跨 1.08 m。后果（已存栅格，P）：room_2 自由格的局地地面相对 `zf` 的 1/50/99 % 分位 = −0.395 / −0.057 / +0.144 m，53 % 的自由格偏离 > 10 cm；自由区凸包内有 5.14 m² 被判为"非内部"，其中 95.5 % 的下射线在 1.2 m 之外**打到 floor / rug**（真实地面被当成虚空）；另有 1.72 m² 的障碍**只来自 floor/rug/mat 面**（地面抬进了 0.15–1.60 m 带）。对比其自由面积 10.6 m²，room_2 约 40 % 的可行区域丢失，图里那条左黑右灰的"带状房间"就是这个伪影。apartment_0 f0：1.79 m² 障碍只来自地面类面，自由格局地地面 99 % 分位 +0.144 m，紧贴带下沿 0.15 m；apartment_1 为 0.01 m²。VAL 四个场景的"仅地面障碍"为 0.00 / 0.00 / 0.00 / 0.09 m²，局地地面偏离 > 10 cm 的自由格占 0.1 % / 1.8 % / 0 / 0（apartment_2 / hotel_0 / office_3 / office_4）——**VAL 基本未受影响，受害的是 TRAIN 的 room_2（320 / 4,800 个 fixture）与 apartment_0**。

**1.4 楼层检测（39–51）。** apartment_0 两层 0.075 / 2.875 m 与语义楼面中位 0.105 / 2.878 m 吻合 ✓（f0 偏低 3 cm 是倾斜造成的直方图峰位）。15 %/2 m 规则在其余场景只出一层 ✓。room_2 见上。

**1.5 带内障碍采样（54–70）——充分。** hotel_0 / office_4 / room_0 用 16 点/cm²、不同种子复采样：只有 27 / 41 / 31 个**孤立**格子（占稠密障碍的 0.10 / 0.15 / 0.14 %）在 4 点/cm² 的已存栅格里缺失，反向多出 0 / 2 / 6 个（P）。带内三角形最长边 50 / 99 / 99.9 % = 1.89 / 2.3 / 2.7 cm，中位面积 0.89 cm²，"最长边 > 4 cm 且 < 4 个采样点"的狭长三角形 0 个 → 不存在细长面漏采；2 cm 腿、薄水平面（4 cm² 格平均 16 点）不会漏。按 20 万三角形分块、按带裁剪 ✓。

**1.6 内部定义（73–79）。** 自由格下射线命中的语义类别（P）：VAL 为 floor 72–99 % + rug（apartment_2 27.7 %）+ shower-stall（hotel_0 8.4 %），合理；"桌面当地面"不会产生假自由（桌面在带内本来就是障碍）。楼梯：apartment_0 f1 的楼梯井（下射线 1.2 m 外打到 `stair` 8,705 格 ≈ 3.5 m²）被正确排除 ✓；f0 只有 0.33 % 的自由格落在楼梯顶部台阶上。**但下射线接受任何朝向的面（M1 (ii)）**：apartment_0 f1 的自由格有 **39.0 % 打到的是 `ceiling` 类**——下层天花板的背面（离上层楼面 17–30 cm，在 0.4 m 容差内）——约 27 m²；与之吻合，f1 的自由区只有 60 % 在最大连通分量里（41.5 / 69.1 m²，M）。这些假区与主区被墙隔开，智能体不放置、也看不到，因此对训练的实际影响是放大了图框 L 与浪费分辨率，而不是错误目标；但它正是用户问到的"楼梯 / 挑空 / 天花"失效模式，在 HM3D 多层场景会更多。扫描空洞：apartment_2（VAL）凸包内 7.76 m² 下方**什么都没有**（未扫描区域，图中左上灰色弧形区）；"没有天花检查"的理由成立——FRL 的天花类面积只有 3.8–4.4 m²（对 39–41 m² 楼面），VAL 场景上射线 2.5 m 内命中 97–100 %（P）。
- 自由区边界的构成（B）：VAL apartment_2 有 **11.7 % 的自由边界（14.8 m）只挨着虚空**（扫描边缘 / 空洞，没有任何表面），hotel_0 2.8 %、office_3 0.5 %、office_4 0 %；TRAIN apartment_0 f1 17.3 %、room_2 19.1 %、apartment_1 10.9 %。即 VAL 场景 1 的"墙"里有一段是扫描覆盖的边界（S4）。
- 玻璃（Gl）：Replica 的玻璃不在 `mesh.ply` 里（在 `glass.sur`）。带内竖直玻璃面穿过**自由**空间的长度：office_4 1.56 m（一面 4.4 m 的玻璃隔断的 35 %）、apartment_2 0.84 m、hotel_0 0.60 m、office_3 0.32 m。所以 `mapping_realscan2d` 文档串第 9–10 行"glass … is wall"对室内玻璃不成立——室内玻璃在切片里是**自由**的（M4 (v)）。几何与传感器一致，不是泄漏，只是需要如实描述。

### 2. `mapping_realscan2d_data_v410.py`

- **旋转 / 逆旋转（78–83、100–103）✓。** TRAIN 3 / 17 / 101 / 1234（偏航 103.6° / 34.8° / 124.4° / 193.7°）：300 个随机自由底格心 base→chart→base 往返的 Chebyshev 误差 0、位移 ≤ 精格半对角线（0.47 cm vs 0.48 cm）；它们全部落在（不膨胀的）自由精格上；用 `scipy.ndimage.affine_transform`（独立写的矩阵，[y,x] 次序显式置换）重采样的 2048² 栅格与模块输出一致率 0.99987–1.00000（差异只在最近邻平局处）；det R = +1（无镜像）（M）。`[y, x]` 约定在 prep（`occ[iy, ix]`、`meshgrid` 后 `reshape(ny, nx)`）、`_blocked_fine`、`metric_trajectory`（`p = (col, row)·cell`）与 v371 传感器之间一致；测试 B 的"观测与栅格一致"也独立证明了这一点。偏航 0 的 VAL 场景 1 图框中心恰落在 2 cm 格边上（平移 −66.5 / −42.5 格），`floor()` 在平局处由浮点决定——确定性的，无害。
- **保守膨胀 k（94–96）。** k = ⌈(精格·√2/2)/2 cm⌉ 对"方形邻域查表"是能保证覆盖半对角线圆盘的**最小**半径（中心可在底格内任意位置）——规则本身正确。实测（M）：所有图框 L = 8–17 m，128 栅格对应的 512 精格 k = 1（L ≤ 14）或 2（L ≥ 15），512 栅格的 2048 精格恒为 k = 1。代价：

  | | L | 128 格 | 底图自由 | 128 无膨胀 | 128 实建 | 膨胀额外去掉 | 128 / 底图 |
  |---|---|---|---|---|---|---|---|
  | apartment_2 | 12 | 9.38 cm | 54.9 m² | 50.5 | 48.2 | 4.7 % | 87.8 % |
  | hotel_0 | 10 | 7.81 cm | 17.0 | 15.4 | 14.4 | 6.4 % | 84.8 % |
  | office_3 | 11 | 8.59 cm | 25.4 | 22.8 | 21.4 | 6.3 % | 84.2 % |
  | office_4 | 8 | 6.25 cm | 24.2 | 23.1 | 22.1 | 4.2 % | 91.2 % |
  | TRAIN 15 层（各一个偏航） | 8–17 | | | | | 3.4–10.6 % | 78.8–90.7 % |

  "全部超采样为自由"规则本身（= v371 第 50 行）去掉 4.5–12 %，膨胀再去 3–11 %（k = 2 的 apartment_0 / frl_4 / frl_5 最多）。对 G 无偏（两臂同一世界），属建议项 S3。
- **`metric_trajectory` / `make_fixture` 与 v371 等价 ✓。** 源码 diff（M）只多一个文档串和 `if nlab == 0: raise`；同一 `v365._seed(world_id + ':poses:N')` 构造。`make_fixture` 只差 `v371.observe_trajectory` 的模块前缀、`geometry_spec` 不再置 `verts_m=None`、门语义字符串。测试 A 对活的 v371 fixture 核对了键 / dtype / 形状 ✓。
- **划分 ✓。** TRAIN 15 层 / 14 场景（apartment_0 两层），VAL apartment_2 / hotel_0 / office_3 / office_4 各取 floor 0，场景不相交，FRL 六套全在 TRAIN，TEST 抛错（T）。**泄漏（L）**：在 8 cm 栅格、2° 步长旋转 + 全平移上求"自由→自由率与障碍→非自由率的均值"的最大值。正对照：VAL 中心 50×60 % 裁块放回原图 = **1.000**；同一套房子重新布置（frl_0 在 frl_1、frl_2 在 frl_3）= 0.763 / 0.732。TRAIN 楼层放进 VAL 楼层的最高分 0.896（room_2 在 office_3：自由 .872、障碍 .921），但同一个 room_2 在 office_4 也是 0.852、office_1 对四个 VAL 都是 0.84–0.85——小而障碍占多的图案在大图里任意处都能拿到 ~0.85，**不具特异性**。结论：**没有任何 TRAIN 楼层是 VAL 楼层的几何裁块**（裁块会得 ≈ 1.0）；同楼不同房间（office_0–2 vs office_3/4）这一级的相似此法无法排除，文档串已如实声明 ✓。（另试过按语义物体包围盒尺寸匹配家具，但正对照 FRL 重布置只配上 1/57，灵敏度不足，弃用、不作结论。）
- **`rooms` 语义与等权汇总 ✓。** VAL spec 的 `rooms` = 场景层 1..4，训练器 `evaluate()` 按 `(rooms, agents)` 分 8 层、`all` = 8 层等权，即"每 (场景, N) 等权"；链脚本分析块的 `eq()` 与 `all` 在无 None 时相同。
- **TRAIN 循环与偏航 ✓。** `index mod 15` 选楼层，N 按奇偶交替；15 为奇数，故每层 8/16 各半（CRT），缓存实测 8: 2,400 / 16: 2,400，前 30 行楼层序列逐一核对（M）。偏航 = sha256 哈希均匀 [0, 2π)。TRAIN 图框 L 每 7 个抽样：8–17 m（L/20 = 0.4–0.85，在 HouseExpo 训练见过的 0.4–1.0 内）。
- **确定性与 Pool / 模块全局缓存 ✓。** `_BASE` / `_UNITS` / `dil` 是进程内惰性缓存，fork 后各子进程独立填充，没有跨进程可变状态；从 manifest 随机抽 16 行在**新进程**里重生成：private 哈希 = manifest = 文件内容，`positions_m`、`door_mask` 逐位相等（M）。底图文件逐个核 sha256（62 行）✓。
- **主墙方向（M，影响第 5 项）。** 按梯度方向直方图（mod 90°，幅值加权）：VAL apartment_2 1.5°、hotel_0 88.5°、**office_3 51.5°**、office_4 3.5°；TRAIN apartment_0 9.5°、office_0 9.5°、office_1 35.5°、office_2 66.5°、room_1 24.5°，FRL 与 apartment_1 在 0.5–2.5° / 88.5°。

### 3. `train_v410.py` 与 `gen_realscan_cache_v410.py`

- **diff 只有世界插线 ✓。** 对 v392：文档串 5 行、`--world` 选项加 `realscan2d`（490）、HouseExpo 世界元组加一项（511）、导入分支 2 行（518–519）、缓存索引 sha 从 `data.INDEX_FILE` 读（585）。测试 F 断言 8 行代码变化全属插线（T）。缓存生成器对 v378：文档串、导入分支、选项、index sha 同样 4 处。
- **`--epochs 0 --init-from` ✓。** smoke 跑的 `final.json` 键 = `arm, updates(0), epochs_completed, weights_sha256, seconds, val, fresh`；`run.json` 的 `warm_start` = loaded 40、fresh []、skipped []、source_updates 57,600——未传 `--allow-partial-init`，任何部分初始化都会在 635 行断言失败 ✓。元数据核对含 readout / precision / bits / wire / decoder_width / frame_cond，**不含 fusion**（S2；链脚本传的 `--fusion max` 与 X 的 COMMON 一致，所以本批无害）。
- **残留的 HouseExpo 假设：无影响 ✓。** `edge_selection` 的 20 m / 2.5 m 网格只在 `--edge-select > 0` 时用；`--panel train` 的 index+200 只在诊断用；门掩码是启发式 `narrow_mask`，只进门指标（不进 wall F1），且 `node_metrics` 对"观测到的门格必须是目标自由格"的硬检查对 v410 仍成立（门格由 `walls == 0` 定义）；`CHART_M` 两条路径都取图框 L（`load` 从 `public_frame.extent_m`，`load_cached` 从 `world_spec`）；单层 `per_stratum = 4800` 的计划检查通过。
- **缓存 manifest 过训练器的全部断言 ✓。** 在 uk-linux 原样执行 579–587 行逻辑：world/size/hops、`data_module_sha256`、全 TRAIN、`rows == scene_plan('TRAIN', K=4800)`、`index_sha256`、首尾各 4 行重生成——**6 项全 True**；目录 4,800 npz 与 rows 一一对应；`load_cached` ×600 用 2.2 s → 4,800 约 18 s（M）。manifest 的 `script` 字段仍写 `gen_hx_cache_v375.py`（S6）。
- **测试 ✓。** 本文在 uk-linux 原样运行（单线程）：A 与活 v371 契约一致；B 传感器精确，8 个 agent 已知比例 .031–.194；C VAL 场景 1 128 栅格自由 48.2 / 54.9 m²；D 划分 / 确定性；E 128 个 VAL/fresh 与 240 个 TRAIN spec 全部可建；F 插线；`ALL TESTS PASSED`，real 88.9 s。注意：测试没有任何断言能发现 1.3 / 1.6 的问题（M1 (iv)）。

### 4. `chain_day6_RS.sh`

- `bash -n` 通过（本地 Git Bash）。
- 等待循环（39）：uk-linux `ps` 实际命令行为 `bash scripts/chain_day5_3DB_s2.sh` 与 `/home/yuzhou/anaconda3/envs/jax-gpu/bin/python scripts/train_3d_v401.py --out …`，两个模式都能匹配，方括号避免自匹配 ✓；40 行 `^$P scripts/train_` 兜底 ✓。3D-B 已于 22:28 UTC 结束，届时循环立即通过。
- 缓存检查（33）核 `scenes/world/size/data_module_sha256` ✓（manifest 实值 4800 / realscan2d / 128 / `5c2a3cf9…`）；基线检查（35–36）：两种子 `final.pt` 存在、`final.json` updates = 57,600 ✓。历史保护（38）：`zs-s* / ft-s*` 不匹配现存的 `smoke-zs-cpu-s342101` ✓；训练器 `mkdir(exist_ok=False)` ✓；但租约 `lease_s*.json`、`*.log` 与 `analysis.json` 用 `>` 覆盖（S5）。
- 租约 JSON：用 `START=1792890000` 等代入后 `json.loads` 合法，13 个键（本地）✓。
- `timeout -k 60 $((DT - now))`：差值为负时 `timeout` 报错不运行（安全）；**恰为 0 时 GNU timeout 视为"不限时"**（S1）。
- 预算（实测 / 引用）：Replica 面板 128 个 fixture 生成约 30 s（M：每个 0.23 s）；X 批每 14,400 次更新（含一次 48 场景验证）1,246–1,428 s（`c3-s*/history.json`）；缓存加载约 18 s。ZS ≈ 2 min、FT ≈ 24 min、HX ≈ 2 min，每种子 < 30 min，远在 6,600 / 7,150 s 截止线内——**加上 M3 的对照臂（≈ 25 min）后仍 < 1 h**。
- 分析块：读的键 `val/fresh → '8'/'0' → strata → wall_f1 / unknown_known_fpr`、`updates` 与训练器写出的结构一致（smoke `final.json` 实测；HouseExpo 基线 `final.json` 的 val 有 6 个 strata）；`ft.updates != 14400`、任何缺键 → INCONCLUSIVE ✓；HX 缺失 → `None` ✓。缺陷：FPR 保护只单独报 `fpr_ok`，没有进判词（X 是在判词后加 "BUT FPR protection violated"）；`*_per_scene_R8` 的键是 `rooms1/N8` 等，没有场景名（M4）。
- 环境变量：`HOUSEEXPO_*`（v365 导入需要）与 `REALSCAN_BASE_DIR` ✓；训练器 sha 只记录、不钉死（X 第 23 行钉死了，S2）。

### 5. 科学 / 预注册

**5.1 ZS 全面板（Z，CPU，与训练器 `evaluate()` 同口径；每 (场景, N) 等权）。**

| 模型 | VAL R8 | VAL K0 | VAL FPR8 | apartment_2 | hotel_0 | **office_3** | office_4 | fresh R8 / K0 |
|---|---|---|---|---|---|---|---|---|
| c3-s342101 | .4023 | .4374 | .0417 | .486 | .471 | **.179** | .474 | .4001 / .4440 |
| c3-s342202 | .4102 | .4407 | .0357 | .507 | .474 | **.177** | .483 | .4014 / .4433 |
| c2-s342101 | .3884 | .4311 | .0412 | .461 | .462 | .166 | .465 | .3924 / .4357 |
| c2-s342202 | .3939 | .4239 | .0351 | .485 | .475 | .155 | .461 | .3921 / .4226 |

smoke 的 .495 / .496 只是 apartment_2、N8 的 4 个 fixture；全面板是 .40 / .41，被 office_3 拉低。

**5.2 office_3 的崩溃是"朝向"，不是"真实扫描"（R，16 个 fixture/格，N8/N16 等权）。**

| 场景 | 偏航 | L | c3-s342101 R8 | c3-s342202 R8 |
|---|---|---|---|---|
| office_3（原 51.5°） | 0 | 11 | .179 | .177 |
| office_3 转正 | +38.5° | 10 | **.567** | **.615** |
| office_3 转正 | −51.5° | 10 | .613 | .599 |
| office_4（原 3.5°） | 0 → +40° | 8 → 10 | .474 → .310 | .483 → .325 |
| apartment_2（原 1.5°） | 0 → +40° | 12 → 15 | .486 → .181 | .507 → .201 |

旋转同时改变 L 与落点，但 16–44 pp 的效应远大于 L 的影响。HouseExpo 基线对墙向几乎没有不变性。而 FT 的 TRAIN 是**均匀随机偏航**——FT 会学到旋转，仅 office_3 一项就可能贡献 ≈ (.59 − .18)/4 ≈ **+10 pp** 的等权 G，单独越过 +3 pp 的"真实扫描数据是杠杆"门槛。换言之，v410 相对 v371 改了不止"几何来源"：还改了图框相对建筑的朝向（扫描坐标系任意 vs HouseExpo 户型对齐）以及加入了旋转增广（M2）。

**5.3 "多训练"本身的效应与噪声（Z，配对、同一面板）。**

| 对比 | VAL R8 差 | fresh R8 差 | 分层自助 SE（按落点） | 配对差的逐 fixture SD |
|---|---|---|---|---|
| c3 − c2，s342101（多一个 HouseExpo 周期，57,600 更新） | **+1.38 pp** | +0.77 pp | 0.22 / 0.24 pp | 2.0 / 2.2 pp |
| c3 − c2，s342202 | **+1.63 pp** | +0.93 pp | 0.22 / 0.26 pp | 2.1 / 2.7 pp |
| c3 s342101 − s342202（种子） | −0.79 pp | −0.13 pp | 0.43 / 0.44 pp | 3.7 / 4.1 pp |
| c2 s342101 − s342202（种子） | −0.54 pp | +0.03 pp | 0.39 / 0.36 pp | 3.6 / 3.4 pp |

（HouseExpo VAL 上同样的对比：c3 种子差 0.24 pp，c2 0.54 pp，c2→c3 +0.27 / +0.57 pp，来自各 `final.json`。）
- **阈值评估**：+3 pp 约为观测到的最大种子差（0.79 pp）的 4 倍、落点抽样 SE 的 7–14 倍，作为"明确效应"合理；面板只有 4 个场景，场景间异质性大（5.2），等权均值对单个场景很敏感，所以必须逐场景报告。
- **|G| < 1 pp 的"已迁移"档几乎不可达**：纯粹多训练（HouseExpo 一个周期）就让 Replica R8 涨 +0.8~+1.6 pp、两种子同号；FT 是 14,400 次更新、学习率重新升到 1e-4，G = FT − ZS 天然含这一项。没有"同样多训练、但用 HouseExpo 数据"的对照臂，"数据是杠杆"无法与"多训了"分开（M3）。
- **预算 / 过拟合**：14,400 更新在 15 层上 = 每层被看 960 次（3 epoch × 320），每个 fixture 3 次；FRL 六层是同一套房子（占 4,800 个 fixture 的 40 %），office_1 只有 4.2 m² 自由；不做 VAL 选择、报告最终权重是对的，但应在分析里同时列出每 epoch 的 VAL 曲线以便看出过拟合（S5）。头部"the cycle settings of batches W / X"不准确：W/X 的周期是 57,600 次（4 epoch × 14,400），这里是其四分之一，只有峰值学习率与 200 步预热相同（M4 (ii)）。
- **可解释性的其余威胁**：几何级泄漏已排除（2）；fresh 复用 VAL 场景，但不做选择，故无选择偏差，只是多一组落点 ✓；室内玻璃被当自由（1.6）、apartment_2 有 14.8 m 的扫描边缘被当墙（1.6）——两臂相同，不偏 G，但 apartment_2 的 G 有一部分会是"学会扫描边缘形状"，需逐场景解读（S4）；障碍带 0.15–1.60 m 内的面一律挡光束（桌面挡视线）是建模选择，应写明（S8）。

## 必改项

**M1 — 预处理必须做重力对齐，内部判定只认朝上的面；重跑底图与缓存（`prep_realscan_base_v410.py` 89–99、73–79、文档串 1–11；`test_realscan2d_v410.py`）。**
现状：只判 z 的符号（18 个场景全 "kept"），不处理倾斜；room_2 倾斜 8.72°（楼面高差 −0.09…0.99 m，"0.62 m 楼面"是伪影，约 40 % 可行区丢失：5.14 m² 真地面成虚空、1.72 m² 地面成障碍），apartment_0 1.31°（1.79 m² 地面成障碍）、apartment_1 0.94°、apartment_2（VAL）0.50°；下射线把下层天花背面当地面，apartment_0 f1 39 % 的自由格（≈ 27 m²，均不在最大连通分量）是假的。要求：(i) 在 `V = V - V.min(0)` 之前，按 `habitat/info_semantic.json` 的 `gravity_dir` 把网格旋到 −gravity = +z（Rodrigues；HM3D 无此字段时用楼面法向的面积加权平均），把 `gravity_dir` 与所用旋转写入 `index.json`；符号启发式删除或改为断言"语义楼面中位低于天花中位"；(ii) `interior_of` 用 `cast_rays` 返回的 `primitive_normals`，只有命中面法向 z ≥ 0.7（朝上）才算地面；(iii) 新目录重跑（现目录归档为 `ABORTED-tilt-replica-base`），重生成 4,800 缓存（上次 839 s），重跑测试与 smoke；(iv) 测试加三条与数据集无关的断言：每层 ≥ 99 % 的自由格局地地面（`zf + 0.8 − t_down`）偏离 `zf` ≤ 0.10 m；没有自由格的下射线命中朝下的面；底图逐位可复现（选一个 VAL 楼层）。预期：room_2、apartment_0 明显改变，VAL 四层只有细微变化（倾斜 ≤ 0.5°、仅地面障碍 ≤ 0.09 m²）。

**M2 — 去掉朝向混杂：图框按建筑主墙方向对齐（`mapping_realscan2d_data_v410.py` 75–83 的 `world_spec` 偏航；`chain_day6_RS.sh` 头部 3–20 与分析 64–97）。**
现状：VAL 用扫描坐标系（office_3 主墙 51.5°），TRAIN 用均匀随机偏航；实测 HouseExpo 基线在 office_3 上 .18，转正后 .57–.61，把对齐场景转 40° 则跌 16–31 pp（5.2）。FT 仅靠学会旋转就能让等权 G 超过 +3 pp。要求（首选 a）：(a) 像 v371 一样让图框与建筑对齐：在 prep 中为每层计算一次主墙方向（确定性方法，例如本文用的幅值加权梯度方向直方图 mod 90°，写入 `index.json`，属公开元数据），VAL 偏航 = −主墙角，TRAIN 偏航 = −主墙角 + k·90°（k 由 index 哈希定，可再加镜像），使 TRAIN / VAL 与 HouseExpo 的朝向分布一致；与 M1 一起重生成；测试 C 改为在对齐后的偏航上检查。(b) 若有意保留原始朝向，则预注册判词只用三个对齐场景（apartment_2、hotel_0、office_4，每 (场景, N) 等权）计算，office_3 单列为"朝向读数"，头部写明 FT 含 HouseExpo 训练所没有的旋转增广，"杠杆"结论不得建立在 office_3 上。

**M3 — 加"同样多训练、但用 HouseExpo 数据"的对照臂（`chain_day6_RS.sh` 53–60、64–97、38、头部）。**
现状：G = FT − ZS 混合了数据域与 +14,400 次更新（学习率重新升到 1e-4）；实测多一个 HouseExpo 周期就让 Replica ZS R8 +1.38 / +1.63 pp（VAL）、+0.77 / +0.93 pp（fresh），大于"已迁移"档的 1 pp。要求：每种子在 FT 之后加 FT-HX：同一初始化 `c3-s<seed>`、同一模型参数、同一调度（`--lr 1e-4`，200 步预热、余弦），数据为 `hx-cache-v375/houseexpo_fit-k4800`（14,400 个 fixture）`--world houseexpo_fit --epochs 1 --expect-updates 14400`，再用 `--world realscan2d --epochs 0 --init-from <FT-HX final.pt>` 评估 Replica VAL / fresh。主对比改为 **G_data = FT_RS − FT_HX**（Replica VAL R8，M2 之后的面板），阈值不变（两种子 ≥ +3 pp → 杠杆；两种子 |·| < 1 pp → 已迁移；否则 partial）；同时报告 G = FT_RS − ZS 与 FT_HX − ZS（多训练效应）。历史保护加 `fthx-s*`；租约 `runs` 列表与 `expectation` 同步改写。时间：每种子约 56 min，在 6,600 s 内。

**M4 — 预注册文本与分析接线（`chain_day6_RS.sh` 2–22、51、64–97；`mapping_realscan2d_data_v410.py` 8–12；`prep_realscan_base_v410.py` 11、89）。**
(i) 披露改正：smoke 是 4 个 fixture（VAL 落点 0、1 → R8 .495 / K0 .568，逐个 .520 / .470；fresh 落点 8、9 → .496 / .580），不是"2 VAL fixtures, placements 0 and 8"；并披露本审查在发射前测得的全面板 ZS（.402 / .410，office_3 .18）、朝向实验（5.2）与 c2→c3 效应（5.3），声明阈值不因此改变。(ii) "the cycle settings of batches W / X" 改为"W/X 的峰值学习率 1e-4 与 200 步预热；长度为其周期的四分之一（14,400 vs 57,600）"。(iii) FPR 保护进判词：任一种子 FT 相对对照的 FPR 升 > 0.3 pp 时，判词追加 "BUT FPR protection violated" 且不得判"杠杆"。(iv) 分析行用场景名（rooms1 = apartment_2、rooms2 = hotel_0、rooms3 = office_3、rooms4 = office_4），并给出每场景的 G。(v) 文档串"glass … is wall"改为"室内玻璃不在网格里，切片中为自由；外窗之外无地面，故为墙"（office_4 有 1.56 m 玻璃线穿过自由区）；prep 文档串"its sign varies"改为实测（18 个场景 `gravity_dir` 全 ≈ −z，room_2 倾斜 8.7°）。改后 `bash -n`，新 sha 记入报告。

## 建议项

- **S1**（链）：每个 `timeout` 前算 `R=$((DT - $(date +%s)))`，`R ≤ 120` 则跳过并记 INCONCLUSIVE，避免 `timeout 0` = 不限时。
- **S2**（溯源）：像 X 第 23 行那样把 `train_v410.py` / 数据模块的 sha 钉死；租约记录底图 `index.json` 与缓存 manifest 的 sha；训练器的热启动元数据核对补上 `fusion`。
- **S3**（栅格）：保守膨胀在 128 栅格上多去掉 3–11 % 的自由面积（k = 2 时最多）；可改为按 128 格的旋转足迹精确判定（任一阻塞底格与足迹相交即阻塞），或至少在文档串写明这一侵蚀；不影响 G。
- **S4**（解读）：apartment_2 有 14.8 m（11.7 %）自由边界是扫描边缘而非表面；逐场景报告 G，并在后续版本考虑把"只挨虚空"的边界格在评估中设为忽略。
- **S5**（分析）：`analysis.json` 附上 FT / FT-HX 每 epoch 的 VAL R8（`history.json`），只报告不选择；租约 / 日志 / `analysis.json` 改为带时间戳或先检查存在，避免覆盖。
- **S6**（缓存）：manifest 的 `script` 字段写成实际的 `gen_realscan_cache_v410.py` 及其 sha。
- **S7**（数据多样性）：在报告里写明 TRAIN 的有效布局约 10 个（FRL 六层为同一套房子、占 40 % fixture），office_1 只有 4.2 m² 自由；"15 floors"不等于 15 个独立样本。
- **S8**（模型假设）：写明障碍带 0.15–1.60 m 内的任何面都挡 2D 光束（桌面挡视线、桌下不可见），这是"可通行 = 可见"的投影约定。
- **S9**（测试）：把本文的旋转往返与 `affine_transform` 交叉检查、偏航 0 的平移等价、以及一个 VAL 楼层的底图逐位复现加入测试 C。
- **S10**（后续）：若选 M2 (b)，另做"旋转增广的 HouseExpo"对照，才能把"朝向覆盖"与"真实扫描几何"彻底分开；claims at scale 仍待 HM3D。

## 结论

**LAUNCH OK WITH CHANGES（M1、M2、M3、M4）——当前版本不得原样发射。**

已核实为正确的部分：PLY 解析与 trimesh 逐位一致、字节预算精确；被审 prep 代码逐位复现已存栅格；带内采样密度充分（16 点/cm² 复采样只多出 0.1–0.15 % 的孤立格，无狭长三角形）；旋转 / 逆旋转往返误差 0、独立重采样一致率 ≥ 0.99987、无镜像，`[y, x]` 约定贯通；`metric_trajectory` 与 v371 同构；划分不相交且无几何级泄漏（裁块正对照 1.0，最高 TRAIN-in-VAL 0.896 且非特异）；训练器对 v392 只改世界插线；`--epochs 0` 热启动全量加载 40 个张量并写出 `val` / `fresh`；真实缓存 manifest 过训练器全部 6 项断言，随机 16 行跨进程复现；测试在 uk-linux 88.9 s 通过；链脚本语法、等待模式、租约 JSON、分析键、INCONCLUSIVE 路径都成立，时间预算宽裕。

必须先改的是数据与设计，而不是管线：M1 修正 room_2 的 8.7° 倾斜（及 apartment_0/1 的 ~1°）与把下层天花当地面的问题；M2 消除朝向混杂——HouseExpo 基线在 office_3 上 .18、转正后 .57–.61，按现设计 FT 只靠学会旋转就可能拿到约 +10 pp 的等权 G；M3 加入"同样多训练、HouseExpo 数据"的对照，因为单是多一个 HouseExpo 周期就让 Replica R8 涨 +0.8~+1.6 pp；M4 修正披露与判词接线。M1 + M2 合起来只需重跑一次 prep 与缓存（约 30 min），M3 每种子加约 25 min，仍在单租约内。改完后，建议对新的 prep / 世界 diff 与重跑的测试、smoke 输出做一次简短的定向复核（不必全量重审），再发射。
