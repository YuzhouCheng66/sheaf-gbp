# Claude Code 研究交接：64 维 Sheaf-GBP 高保真多智能体地图融合

更新：2026-10-21。本文是**地图融合研究**的交接入口，不是仓库原有 PianoMovers 搬运 demo 的说明。

## 先读结论

用户希望取得的是**结构性突破**：仍使用 64 维 Sheaf-GBP，让墙体、实体占据区域和自身/已达邻居区域达到极高的复原准确率与 F1。请联合研究 encoder、decoder、restriction map 和目标能量，不要把瓶颈预设成 GBP，也不要继续无止境微调零点几个百分点。

- **交接质量锚：v310 完成 epoch 12 的完整多尺寸 codec（也是 v319 epoch 0 的冻结基线）。**它是有据可查的较强研究参考，不是所有尺度/指标都最好的模型，更没有通过全部历史保护门槛。原始 v299 control、较强的 native128 v304 control 也必须保留。
- **最新不等于最好。**v340 已正常结束，审计和三位独立复盘完成，候选未通过准入质量门槛；v341 只有尚未实现/未启动的设计，不能作为已训练模型交付。Claude 可以重新评估它，不必被它锁定。
- **尚无可信的整体 v299 突破结论；restriction 有真实梯度和权重变化，但尚未证明带来稳定、可归因的高保真增益。**数值正确不等于建模正确或复原准确。
- 用户已确认下一阶段优先考虑 **HouseExpo**，并要求不再只有墙：增加非墙实体、遮挡和复杂占据结构。现有 HouseExpo pilot 可复用，但它不是当前主训练集，也没有真实家具实例占据标签。
- 本次交接**没有启动、恢复或延长任何训练**，没有删除历史结果。完整模型与大数据留在原位置；本文和 [artifact_manifest.json](artifact_manifest.json) 是带校验信息的索引，不是包含所有权重的便携压缩包。

## 1. 给 Claude Code 的直接任务

请使用当前环境允许的**最多并行 agents**，而不是单 agent 顺着旧代码继续加层。建议由主 agent 统筹，下列角色独立读证据后再综合；如果并发不足，分轮执行：

| 角色 | 必须回答的问题 |
|---|---|
| Encoder / 信息保留 | 64 个全局变量到底保留了哪些墙、物体、已知支持与几何细节？K0 漏墙是编码丢失、优化失败，还是 decoder 读不出？ |
| Decoder / 融合可读性 | 如何用同一声明瓶颈同时读懂自身证据与外来互补证据？融合后 latent 是否落在不可读分布？如何保护可靠自身区域而不锁死错误预测？ |
| 能量 / restriction | 局部不完整地图的 code 能否当作完整公共地图变量的均值？观测因子应约束什么、哪些方向不应被约束？双端 restriction 到底比较哪个共同物理量？ |
| GBP / 统计 / 评价 | cavity、来源版本、重复证据、协方差闭包与集中式目标是否一致？实际到达 union、指标、数据分割及通信账是否诚实？ |
| 数据 / HouseExpo | 建立多样化且无户型泄漏的数据；区分结构墙、物体和全部 occupied；验证真实运动、遮挡、门洞与物理尺度。 |
| Runtime / 独立反方 | 分解真实端到端耗时，审查等价加速；主动寻找捷径、混淆变量、反例和不成立的突破主张。 |

每位 reviewer 必须先给出**假说、已有证据、反证、可证伪实验**，不能只赞成同一个直觉。主 agent 只选一个最有潜力的大改动及必要的匹配对照，不把全部方案排成自动 ablation 队列。

首要研究直觉：自身到访区域观测近乎无噪声，理应尽量保留；自身没到访但收到邻居证据的区域，应读出邻居的真实细节；未收到证据处保持 unknown。**restriction 和 decoder 是否能合作区分这三种情况，是问题，不是已实现的答案。**

## 2. 最佳参考、失败候选与当前状态

### 2.1 推荐保存/复现的多尺寸质量锚

远端完整模型：

```text
/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/
  checkpoints/u000021312-e0013-epoch-1792323462678299519.pt
```

- 文件 SHA256：`31da2f3962c1a102367088dbcdd4063d11bbcb77ac4f75f11d91f95efefeeb3a`
- 模型 tensor-state SHA256：`d1d397b654e6e108b23e1fdabf8f6bb2588d4756835cb9b9d66219be1a369203`
- 这是 **完成 E12**、额外训练 21312 updates 的 checkpoint；文件中的 `e0013` 表示下一训练 epoch，不能报告成 E13 的验证结果。
- 选择理由：在该 v299 warm-start 多尺寸系列中，有记录的较强等尺寸 R8 peer-wall-F1 研究锚；**不是全版本全指标的 Pareto 支配者**。
- v319 的 `epoch-0000.json` 是此锚的冻结基线；v319 的小型 restriction checkpoint 不能脱离这个完整 codec 单独加载。
- 下表使用原始 v310 E12 报告；v319 E0 复评有少量浮点/阈值差异，不声称两份预测逐像素完全一致。

下表来自 [v318 锚点审计](../configs/mapping_restriction_firstprinciples_v318.json)，为 VALIDATION 中每个尺寸的六个 rooms×N 分层等权平均。R8 是八轮实际通信后的已达 union，不是“每个源都恰好八跳”。数值均为百分数，未知误报越低越好：

| Native 尺寸 | K0 自身 accuracy | K0 自身 wall F1 | R8 union accuracy | R8 union wall F1 | R8 unknown FPR | R8 自身 wall F1 |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 92.55 | 83.11 | 89.38 | 62.56 | 6.52 | 59.90 |
| 256 | 90.16 | 55.08 | 90.15 | 46.64 | 8.81 | 41.58 |
| 512 | 85.24 | 34.57 | 89.14 | 43.20 | 12.29 | 37.66 |

这张表直接暴露了问题：部分尺寸 K0 已弱，128/256 融合后自身墙体保留也明显不足。不能仅引用接近 90% 的 accuracy 来掩盖墙体缺失。

### 2.2 保留的其他参考

| 版本 | 用途 | 不能误解为 |
|---|---|---|
| v299 `joint_control/validated-002000.pt` | 原始 global64 强基线；native128 R8 accuracy 88.87%、wall F1 60.25%、unknown FPR 5.44% | 所有新尺度/数据上的正式最终基线 |
| v304 `diagonal_recognition/validated-002000.pt` | 更强的 native128 K0/R1 机制参考；K0 wall F1 87.06%，R1 union wall F1 68.67% | v304 另一个失败候选 `observed_diagonal` |
| v310 E12 / v319 E0 | 本次推荐的多尺寸质量锚 | 已通过旧保护门槛或普遍最优 |
| v319 trained E8 | native128 R8 wall F1 63.51% 的研究结果，但保护失败；该运行 eligible-best 仍为 E0 | 可替换安全基线的最佳部署模型 |
| v340 E17 amplitude | 最新完成的端到端架构候选；有完整数学/包/梯度/原图审计 | 比上述锚更好，或证明 restriction 有因果收益 |
| v341 shared-key | 一个待证伪的 encoder/decoder 公共键对应设计 | 已实现、已准入或已训练的版本 |

v299 与后续不同运行的表格是**历史质量参照，不是同数据、同初始化、同预算的因果比较**。新增算法应另做公平匹配比较。

作为“并非所有尺寸最高分”的具体反例，旧 scratch 配对所选 E32 的 native512 R8：joint 为 **91.89% accuracy / 45.29% wall F1 / 12.74% unknown FPR**，only512 为 **90.72% / 44.97% / 15.07%**（[历史记录](../configs/mapping_current_program_v316.json)）。不应隐藏这些更强的部分指标，也不因此自动复活已停止的 scratch 运行。

v340 amplitude 的 R8 结果：native128 accuracy **76.65%** / wall F1 **50.65%** / unknown FPR **16.55%**；native256 **77.03% / 44.95% / 15.95%**。部分 F1 小幅上升但多项保护失败，没有突破。详见 [一次性结论](../configs/mapping_endpoint_v340_fp_synthesis.json)、[原始审计](../artifacts/v340-endpoint/audit.json) 和 [review ledger](../configs/mapping_endpoint_v340_review_ledger.json)。

当前状态末端为 [v341](../configs/mapping_current_program_v341.json)：`active_training=false`、`implementation_pending=true`、GPU 未准入。旧配置保留历史文字，不能把旧 PID、`registered`、计划或旧 Goal 字段当作实时状态。按 `successor` 明确链接追踪，再与实际进程和 endpoint ledger 核对。

## 3. 研究目标和真正硬约束

### 必须保留

1. **每个 agent 的地图表示为 64 个 Gaussian 变量，配 64 个 diagonal uncertainty/precision。**不要偷偷变成 64 个高维 token、每块地图一个 64D、或额外私有栅格特征。encoder 内部可以宽，但送到融合/读出的样本相关信息必须经过声明的瓶颈。
2. 保持有明确目标能量与消息语义的 **Sheaf-GBP**。不能把任意 attention、latent 平均或消息 MLP 换名当作 GBP。
3. 允许 decoder 同时读取保留的**自身 `(q,h)` 和已达 posterior `(q,h)`**，但自身分支也必须来自同一 64+64 瓶颈；不能保留 raw 图、真实 own mask、私有 feature raster、GT 修补或原图覆盖来制造高准确率。
4. 主目标是超高复原 accuracy、结构墙 F1 和全部 occupied F1，同时保护自身区域；邻居细节和 unknown 必须单独检查。目标可争取 99% accuracy，但当前没有达成证据；“超高 F1”不能被擅自改写成已有成功。
5. 真实运动/遮挡的局部观测、半径邻接、多 agent、多房间/走廊；分辨率、物理尺度、复杂度、数量分开控制。发现阶段优先 native128/256，最终仍需 native512、不同尺度、环路与不连通图。放大 128 不算 native512。
6. TRAIN 才能拟合和归一化，VALIDATION 选点，TEST 封存；实际到达的原始 source/version 决定监督。
7. 维度不是通信成本：完整记录数值位宽、量化尺度、header、支持/来源/版本、setup、转发、重传和无效发送；未知成本标未知。

当前实验假定地图已配准、公共坐标系/栅格边界和位姿可用，研究的是配准后的压缩与融合，**不是已经解决未知坐标变换或 SLAM**。公共几何输入及其获得成本必须披露，不能暗中给模型全图。

### 可以重新设计，但必须单独登记和公平测试

encoder/decoder 架构、全局因素的几何含义、观测能量、restriction 家族、读出协同机制、训练目标及数据分布都可以研究。**旧 v319“冻结 encoder/decoder/数据”不是今后的硬约束**：用户后来已要求端到端从头设计，并在本次要求数据多样化。

正对角 restriction、`.1 + softplus` 下界、当前 source-priority blend、周期场/残差 decoder、固定 20m、离线森林、具体 loss 权重和 scheduler 都是历史实现选择。不得在线修改封存运行，也不得为让新结果过关而事后放宽旧门槛；新科学设计须先声明变化和比较方式。

## 4. 第一性原理：到底要建模什么

### 4.1 局部地图 code 不是天然的全局地图似然

局部地图中的 unknown 不是 free，也不是“这里必须没有邻居的内容”。若直接使用

```text
E = 1/2 Σ_i ||x_i − μ_i||² + 1/2 Σ_(i,j) ||x_i − x_j||²
```

而 `μ_i` 是不同部分地图的完整重建 code，就可能同时要求“保持各自不同的部分图”和“全部隐变量相等”。这不是自动成立的观测模型。应区分：

- **观测因子** `A_i x ≈ b_i`：只约束真实观测所提供的内容；哪些方向应保留不确定性？
- **边相容性** `R_i,e x_i ≈ R_j,e x_j`：两端 restriction 必须落在具有相同物理意义的公共 edge space；不能把两个互不相同的区域强行相等。
- **decoder**：如何将后验和自身因子还原成已知支持与细节，并在未到达区域保持 unknown？

只发送 `R μ` 并不充分：还要正确变换 information/precision，说明公共坐标、可用元数据和接收端拉回方式。纯可逆坐标变换若完整保留均值与协方差，只是重参数化，不会凭空增加信息。冻结 code 后处理也不能恢复已彻底丢失的信息，但可能使已有信息更容易被 fusion/decoder 使用。

### 4.2 新架构当前实际能量（不是强锚 v310 的公式）

v329→v340 的模型使用：

```text
q_i > 0,  h_i = q_i ⊙ μ_i
E(x) = 1/2 Σ_i (x_i−μ_i)^T diag(q_i)(x_i−μ_i)
     + 1/2 Σ_e ||a_e ⊙ x_i − b_e ⊙ x_j||²
```

双端正对角 restriction 分别由自身 code 与公开端点位置生成，在一次固定证据解算中固定。若 j→i 的 cavity 为 `(Q,H)`，j 端 restriction 为 `b`，i 端为 `a`，未量化的接收消息是逐坐标：

```text
q_msg = a² Q / (Q + b²)
h_msg = a b H / (Q + b²)
```

发送端排除接收邻居的消息，传公共边坐标中的 Gaussian 参数，接收端用自己的 restriction 拉回；旧消息替换，不重复累计。源码入口见 [geometric source/restriction](../scripts/mapping_geometric_ownership_v329.py)、[GBP 和 wire replay](../scripts/mapping_local_extrinsic_v324.py)。当前实际 decoder 还继承 [v334](../scripts/mapping_geometric_ownership_v334.py)、[v335](../scripts/mapping_periodic_ownership_v335.py)、[v338](../scripts/mapping_residual_ownership_v338.py)、[v340](../scripts/mapping_amplitude_ownership_v340.py)，不要只读 v329 就认为看到了最终模型。

来源 ID 不重复只解决计算上的重复计数，**不证明不同 agent 的重叠观测统计独立**。多份高度相关的同一墙体观测仍可能引起过度自信；应另测试固定 union、不同来源分割/冗余的敏感性，不把 source-once 当成已经校准。

限制：source precision 只是**学习到的 surrogate**，没有证明是校准的传感似然；非空局部图几乎每个隐方向都有正 precision，未证明正确保留未观测 nullspace。`diag(JᵀJ)` 可以破坏真实的混合方向 nullspace。一般 dense 64×64 Gaussian covariance 有 **2080** 个独立数，不是 64 个；任意 dense restriction 后直接扔掉非对角项会改变目标/推断，必须诚实审计。

### 4.3 K0 和自身保护的关键教训

新架构无邻居时严格 `x=μ`，仍可能 K0 漏墙：编码、precision 输入和 decoder 本身都可能不够好。旧 v319 还有先验、坐标旋转、裁剪及 covariance 输入变化，冻结权重不意味着读入特征不变。因此不要把所有 K0 退化怪到多 hop。

当前 source-priority blend 用的是**预测**的自身 unknown 概率，不是真实观测 mask。它能保护可靠自身内容，也可能锁住“自信但错误的 free”。需要可证伪测试，不能假设一个 gate 就同时解决自己/邻居问题。

已有 GT latent/geometry 拟合只能作可达性诊断：成功不等于可部署的编码与传输成功；非凸优化失败也不证明 64D 理论容量不足。

## 5. 数据：保留历史回归，新增真正多样性

### 5.1 当前合成 bank 的实际局限

远端：`/home/yuzhou/Desktop/sheaf-gbp/support-native-pair-v306-5090/data-bank/manifest.json`，SHA256：
`3e450d0675814f8fbfc8b8028df796fc19d5a42404abac460e6b1310ddf6ac8a`。

**这不是自包含目录**：`data-bank/config.json` 的 `old_bank` 指向 `/home/yuzhou/Desktop/sheaf-gbp/multiroom-domain-v295-5090/data-bank`。H128 的 `NativeBank._old()` 仍从该目录读取 TRAIN/VALIDATION manifests 以及每个场景的原始 NPZ/JSON。只拷 v306 目录会缺依赖；两份旧 manifest 的 SHA 和 config SHA 已登记在交接清单中。旧依赖保持只读，迁移不能在线修改封存配置。

- 全 bank：**296 个 TRAIN 几何世界 × N8/16 × H128/256/512 = 1776 scene tuples**；24 个 VALIDATION 世界对应 144 tuples。128/256 子集为 1184/96，不能写成 1184 个独立户型。
- rooms 6/12/20；20m×20m；256 条全向光束；5m 感测、8m 通信半径。每个 agent 有三个很短位移的遮挡观测，然后进行静态通信。
- 实际形态主要为一条直走廊、两侧轴对齐矩形房间，free 补集为 occupied。缺乏家具、杂物、旋转/非矩形房间、弯曲/分支走廊等。当前名为 `wall F1` 的指标针对该生成器 occupied 类，不是已验证的一般结构墙分割。
- 同一世界在尺寸/N 上有别名，必须整组划分。旧 VALIDATION 被多轮设计使用，是 development panel；不能再称从未接触的最终泛化测试。TEST 仍封存。

依据：[数据计划](../configs/mapping_support_native_data_plan_v306.json)、[生成/训练入口](../scripts/train_mapping_support_native_v306.py)、[native bank](../scripts/mapping_native_bank_v306.py)。

### 5.2 用户已确认的公开 benchmark：HouseExpo

HouseExpo 官方报告 **35,126 户型、252,550 房间**，来自 SUNCG 的二维户型轮廓与房间信息；它不是包含真实家具实例占据标注的现成数据集。官方 PseudoSLAM 可以另外生成障碍物，这与源 JSON 含家具不同。[官方仓库](https://github.com/TeaganLi/HouseExpo)、[原论文](https://arxiv.org/abs/1903.09845)、[官方 JSON reader](https://github.com/TeaganLi/HouseExpo/blob/master/pseudoslam/envs/simulator/jsonReader.py)。

已有资产，不需要重新盲目下载：

```text
D:/sheaf-gbp-artifacts/occupancy_codec_20261014/
  houseexpo-source/json.tar.gz
  houseexpo-pilot-v121/manifest.json
  houseexpo-pilot-v121/train.npz
  houseexpo-pilot-v121/validation.npz
```

旧 pilot：200 TRAIN 户型×4轨迹×8 snapshots＝6400 graphs；40 VALIDATION 户型＝1280 graphs。N6、native128、0.1m/cell、3m 感测、5m 通信、256 beams。按 house ID 先分割，再生成轨迹；已排除精确归一化轮廓重复，但不是全面近重复去重。没有家具增强；TEST 未开。接入代码：[prepare_houseexpo_pilot_v121.py](../scripts/prepare_houseexpo_pilot_v121.py)、[mapping_houseexpo_admission_v122.py](../scripts/mapping_houseexpo_admission_v122.py)。旧 adapter 有 ≤128 限制，不能简单删 assert 或插值冒充 native256/512。

源包和 pilot 的 SHA 见 [manifest](artifact_manifest.json)。官方仓库有 [MIT LICENSE](https://github.com/TeaganLi/HouseExpo/blob/master/LICENSE)，同时应保留 SUNCG 衍生来源并单独核查数据使用/再分发条件；不能把软件许可证自动当成所有衍生数据权利都已审查。

### 5.3 Claude 的新数据任务

1. 历史 bank 保持不变，另建带版本、来源、分割和 SHA 的多样化数据集，不混写旧分数。
2. HouseExpo 增加户型连接、门洞、墙厚和尺度多样性；先 native128/256 做模型发现，随后真正原生 512 验证。
3. 增加明确标为**合成增强**的非墙 occupied 对象：柜体/桌椅近似体、圆/多边形障碍、斜物体、细杆、狭窄通道和遮挡组合。不要声称它们是 HouseExpo 真实家具。
4. 若接入真实家具 benchmark，另审计官方来源、许可、传感高度与二维投影；本次未指定第二个数据集，也未授权付费购买。
5. 建筑/户型级先切分；同户型全部尺寸、N、轨迹、裁剪和增强留在同 split；检查旋转、平移及近重复泄漏。
6. 分别维护结构墙、非墙物体、全部 occupied 的评估标签；这些标签不能变成 decoder 的未计费私有输入。若只有 ternary 输出，区分“按 GT 子区域统计的复原能力”与真正语义墙/物体分类能力。

## 6. 逐阶段监督：已经实现，仍必须检查

用户提出的 1-hop/2-hop union 思路已经落实为实际到达来源监督，而不是用全局 union 监督每一轮：

| 阶段 | 合法目标 | 重点验收 |
|---|---|---|
| K0 / R0 | 仅自身原始观测 | own accuracy/F1；自己没看到处保持 unknown |
| 第一次同步交换 | 自身 + 真的收到的直接邻居 source/version | peer-only 细节；没收到的邻居不能自动入 GT |
| 后续通信 | 按实际时间到达的原始 source/version union | 逐来源覆盖、细节、未知误报和累计字节 |
| 固定证据实验 | 固定完全相同的 source 集和版本 | 一次交换、多次真交换、无发送本地迭代分别比较 |

区分 `h` 图距离、`R` 实际通信轮数/事件、`L` 无发送本地计算。新转来的二跳证据不是“同样的一跳证据反复优化”。每条边一次交换也不是整个图一个包。

禁止 posterior/decode 当新观测、重复源似然反复相乘、用 variance 变小代替精度提升。要测试重复包/重复来源/过期版本/无新观测/图变化/环路/不连通及固定点稳定性。`peer_cells==0` 仅表示相对自身没有新增像素，不等价于两轮 source 集完全相同。

## 7. 指标、选点与突破标准

按 **尺寸×物理尺度×房间/对象复杂度×N×实际 R/source coverage** 报告，不只一个平均数：

- own、已达 union、peer-only accuracy；occupied/墙/物体 P、R、F1。
- unknown FPR＝target-unknown 中被预测成 known 的比例；unknown 误判成 free 同样计入。
- own 是原始自己已知；peer-only 是已达 known 且自己 unknown；union accuracy 在 target-known 内用原始三值预测，预测 unknown 也算错。
- 现有 wall F1 在目标已知范围统计，必须与 unknown FPR 一起读；空 scope 记 `None` 并保留数量。有真墙却预测完全无墙时 precision/F1 为 **0**，不能排除；真墙和预测墙都没有时 F1 才为 `None`，无真墙时 recall 为 `None`。
- 门洞是否被封、墙是否断裂、可达性/连通性、原始来源覆盖，以及 accuracy/F1—累计字节曲线。
- 固定、未挑选样例的 own observation、实际 reached GT、未经后处理 prediction；不要只发最漂亮的图。

历史保护规则供连续可比实验沿用：各 H/R 自身 F1 最多下降 0.5pp、accuracy 最多下降 0.2pp；通信 union accuracy 最多下降 0.2pp、unknown FPR 最多上升 0.5pp。通过这些保护后再比较平均通信 wall F1，至少 +2pp 才有旧协议所称“实质增益”的材料。**这不等于用户所希望的大突破，更不等于 99% 或高拓扑保真。**新数据实验须预先定义相应基线和门槛，不能事后改口径。

必须分开给四种结论：**数值/来源正确；像素复原；门洞/连通性；全过程压缩。**某一项通过不能代替其他项。

## 8. 64 维不自动等于 100× 压缩

历史 v299/v310/v319 的静态森林应用包为 172B；新 v321→v340 是 64 FP32 information + 64 FP32 precision，加 header/来源/版本，N8 为 **576B**、N16 为 **608B** 每个真实有向消息。两者不能混用。

最终目标仍保留：在**相同原始证据覆盖、同等复原质量**下，比较达到目标需要的最少发送次数与全过程总字节，相对公平 2-bit raw 争取 ≥100×，同时保留强无损/增量 raw 对照。不能虚增 raw 重复发送，或漏算我方重复包、尺度和初始化成本。

v340 已测的 profile/R source-once 2-bit / 本方法应用层字节比为 **3.376–51.848**；强无损 / 本方法为 **0.114–0.827**，即这个强 raw 对照更省。模型下发、分布式建树等成本仍有未知，且质量未匹配，**没有全过程 100× 的证据**。

当前森林使用离线公共调度和实际选边；它不解决分布式建树，也不代表一般有环推断。Claude 应保留一般环路与不连通图为后续真正验收范围。

## 9. 训练活性、数学测试与速度

新模型不只看 loss 下降或 `grad != None`。逐组件记录 encoder、mean、precision、restriction、lift、decoder 的梯度范数、初末权重差及饱和比例；固定非退化 TRAIN 见证验证梯度传播。持续零梯度应先查断图、量化反传、饱和和日志问题，不追加步数或宣称容量不足。

最低数学/实现检查：K0 闭式解、空观测、互补观测、重复证据幂等、tree cavity 对声明能量的集中式均值/方差/梯度参考、实际序列化包重放、合法来源时序。一般环路另验证，不从森林通过自动外推。

速度目标是缩短完整迭代周期。先量化 scene load/H2D、encoder、restriction/GBP、decoder、全像素 loss、backward、optimizer、同步/日志、VALIDATION 和 checkpoint I/O，按 H/N 报 wall-clock。已有 CUDA graph/manual VJP/执行顺序优化可以参考，但不能直接假设适配新架构。

**等价加速必须验证输出、loss、梯度、参数更新和 Adam 状态；至少多步固定见证，覆盖实际尺寸/N。**不得通过降分辨率、删监督、少算阶段、改变 batch/精度而不披露来冒充等价加速。新 decoder 本身是算法改变，不是旧模型十倍等价提速。此前数量级加速目标尚未兑现。

## 10. 研究节奏与已授权范围

[持续实验授权](../configs/mapping_continuous_authority_20261020.json) 记录用户“不必每轮再问授权，请持续实验”：普通新批默认 **7200 秒总计**，包含 GPU 准入、TRAIN、VALIDATION，使用新的不可延长 lease。旧预算不得重置，旧停止运行不得自动重启。选择从头训练还是继承权重/Adam/RNG/cursor必须在新协议中先写明。

- 新批前：登记单个干预和匹配对照、固定数据/选点/门槛/预算，完成数学测试、独立代码审查和 5090 准入。
- 运行时：源码不在线修改；至多每小时一个有界中间 reviewer 检查完整 VALIDATION 趋势、活性和安全，判断继续/安全停/实现错误。不持续消耗 token 空转做新研究。
- 每个真实结束的短批：先绑定 run id + endpoint SHA，审计包、原图、曲线和指标，再三位独立 reviewer **只做一次**复盘。
- 完整复盘交替：一次完全不参考文献的第一性原理；下一次原始论文/官方代码分析。v340 已完成第一性原理轮，**下一个真实完成的新训练批用文献轮**。不是每小时都开三位全面复盘。
- 保存独立结论、分歧、反证、选中方向及否定其他方向的依据；预算结束不等于收敛，继续也不是默认无限追加。

不得新增付费服务、硬件或更大范围任务。历史 `sheaf-gbp` 自动监控和 credits 暂停规则已被用户撤销，**不要从旧 heartbeat 复制回来**；当前没有自动每小时调度器。Claude 若实现自己的监控，需在其真实运行机制内明确说明，不假装存在后台监控。

## 11. SSH、5090 和安全边界

用户明确要求把远端连接交接给 Claude：

```text
SSH alias: uk-linux
SSH endpoint: yuzhou@100.69.198.56
GPU: NVIDIA RTX 5090
GPU UUID: GPU-238bdbd5-9f06-da6d-09f6-830fd831e817
Remote Python: /home/yuzhou/anaconda3/envs/jax-gpu/bin/python
Local project: C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill
Local Python: D:/anaconda3/envs/jax-gpu/python.exe
```

首选已有 SSH alias 和本机已配置的认证；README 不包含私钥/密码，不复制凭据。换机器时须先具备私有网络路由与自己的合法 SSH 认证，直连地址不是免认证接口。

只读检查示例（PowerShell）：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 uk-linux 'hostname; date -u; nvidia-smi --query-gpu=name,uuid,memory.used,memory.total --format=csv'
ssh uk-linux 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; df -B1 /home/yuzhou/Desktop/sheaf-gbp /mnt/datasets'
```

**2026-10-21 01:08:16 UTC 的交接快照**：没有匹配的本研究训练进程；5090 被另一个项目占用（约 9954MiB）。`/home` 文件系统空闲 **25,576,394,752B，低于 30GB 安全线**；`/mnt/datasets` 空闲约 **1.696TB**。这是快照，不保证接手时相同。

任何启动都先重新核对 UUID、进程命令和实际输出盘空间；不能杀其他任务、不能占满共享 GPU，不用本地 4080，不访问 bigboy。旧 `/home` 根当前不适合新写入训练；如改用 `/mnt/datasets` 新根，先检查所有写入路径/临时目录，明确只读旧依赖和新 lease，不能偷偷降低 30GB 安全线。不要因一次低 GPU 利用率判断卡住或重复启动。

安全停止保存 model、Adam、RNG、数据 cursor、scheduler、tracker 和恢复入口；checkpoint 保留 initial/best/latest、每 5 epoch 归档、两个 rolling recovery 和安全停止即可，不每步复制完整模型。不要借本次整理删除历史结果；已封存的清理记录另有审计。

## 12. 代码与证据导航

已经把 v310 的**完整 70 文件封存源码闭包**复制到 [reference_source_v310](reference_source_v310/)，合计 836899B；逐文件 SHA **70/70 一致**。它同时包含 v299 的加载依赖。校验清单为 [source-manifest-v310.json](source-manifest-v310.json)，SHA `a0154cb73488ab3bf7f3f34f15d193af6133d422587e46654103835c99185d8d`。快照没有修改；其中旧训练脚本仅用于阅读，不能直接运行恢复旧预算。

本轮已经在远端 CPU 实际加载 v299/v310 两个锚并核对 tensor-state SHA。提供独立只读工具 [verify_reference.py](verify_reference.py)：默认只校验本地 70 文件，不读 checkpoint、不训练、不写文件；带 `--anchor` 与 `--checkpoint` 才在 CPU 加载**精确匹配文件 SHA**的可信历史模型，且检查实际 import 来自封存源码。

```powershell
& 'D:/anaconda3/envs/jax-gpu/python.exe' claude_handoff/verify_reference.py
```

若在远端使用此工具，把这个小工具单独复制到自己的诊断目录，然后运行下述参数组合；不需要搬动大 checkpoint，也不要在占用中的 GPU 上做检查：

```text
/home/yuzhou/anaconda3/envs/jax-gpu/bin/python /YOUR/DIAGNOSTIC/DIR/verify_reference.py
  --source /home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source
  --source-manifest /home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/source-manifest.json
  --anchor v310_e12
  --checkpoint /home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090/checkpoints/u000021312-e0013-epoch-1792323462678299519.pt
```

上面是参数说明，执行时合成一条命令，或使用当地 shell 的续行符。v310 正确加载接口是 `MultiscaleV299.from_payload(payload["model"])`；v299 是 `ContentReadout64.from_payload(payload)`。**不要用 `from_parent_payload` 加载 E12，也不要用 trainer 的 `load_checkpoint(root,p)` 代替：后者会读 latest 并恢复 Adam/RNG，可能选错 checkpoint。**

目前没有独立的完整纯 evaluate CLI；原 trainer 的 `prepare/check/preflight/run/resume` 不是无副作用评分入口。需要重新评分时，Claude 应在新诊断目录做只读模型/数据薄封装，保留原 source 与输出，不运行 `resume`。CPU 加载通过也不等于完整指标已重跑。

不要把仓库根 `pip install -e .` 当作已安装全部地图实验依赖：根包主要是 manipulation demo，地图脚本还有 `scripts` 与 `exploration_grid` 的导入路径。只读检查或 CPU 测试可在项目根设置：

```powershell
Set-Location 'C:/Users/27118/Desktop/sheaf-gbp/experiments/sheaf-gbp-p6-local-skill'
$env:PYTHONPATH='.;exploration_grid'
& 'D:/anaconda3/envs/jax-gpu/python.exe' -m pytest -q scripts/test_mapping_multiscale_v299_v310.py
```

这是 CPU 测试入口，**不是复现历史分数或启动训练的命令**。封存 source snapshot 优先于同名可编辑脚本；先验证 SHA 再加载可信本项目 checkpoint。不要运行旧 `launch_*` / `train_*` 期待它自动变成新实验。

| 入口 | 内容 |
|---|---|
| [artifact manifest](artifact_manifest.json) | checkpoint、数据和小型审计的精确路径/哈希；本地与远端区别 |
| [v318 baseline/theory](../configs/mapping_restriction_firstprinciples_v318.json) | 推荐锚、不同参考的优劣、源/边因子区别、冻结 restriction 研究的反例 |
| [v310 model](../scripts/mapping_multiscale_v299_v310.py) / [trainer](../scripts/train_mapping_multiscale_v310.py) | v299 global64 的 native128/256/512 延续，不是任意新 sheaf 能量 |
| [v310 registered protocol](../configs/mapping_v299_multiscale_continuation_v310.json) | 原始训练、数据、选点和执行限制 |
| [v319 protocol](../configs/mapping_restriction_training_v319.json) / [endpoint](../configs/mapping_restriction_endpoint_v319.json) / [ledger](../configs/mapping_restriction_review_ledger_v319.json) | 冻结 codec restriction 学习及失败范围；best 保留 E0 |
| [metrics](../scripts/mapping_recognition_metrics_v277.py) | 指标 scope/空分母/原始预测定义 |
| [v340 audit](../artifacts/v340-endpoint/audit.json) / [synthesis](../configs/mapping_endpoint_v340_fp_synthesis.json) | 最近一次完整训练的实证失败和下一步假说 |
| [v341 design](../configs/mapping_shared_key_design_v341.json) | 未实现的公共 read/write keys 对应诊断；是候选，不是必走路线 |
| [current program v341](../configs/mapping_current_program_v341.json) | 交接时明确状态链末端；并非运行状态的永久保证 |
| [priority contract](../configs/mapping_hop_fidelity_priority_20261017.json) | 长期逐到达监督/真实通信/独立结论约束；注意其中旧历史段落 |

固定原图示例（未后处理、不是挑最好）：[v340 control 128](../artifacts/v340-endpoint/control-native128-rooms20-N16-scene0-agent0.png)、[candidate 128](../artifacts/v340-endpoint/amplitude-native128-rooms20-N16-scene0-agent0.png)、[control 256](../artifacts/v340-endpoint/control-native256-rooms20-N16-scene0-agent0.png)、[candidate 256](../artifacts/v340-endpoint/amplitude-native256-rooms20-N16-scene0-agent0.png)。

## 13. 接手后的第一个交付物

请先交一份简短的事实清单和一个**可证伪的大方向**，而不是立刻续训最新弱模型：

1. 核实推荐锚可加载、完整 source/data 可定位、旧实验确实停止；保留准确 hash 和执行环境。
2. 并行分析上述 codec / 能量 / restriction / 推断 / 数据问题，明确哪里是证据、哪里只是猜想。
3. 选择能同时解释 K0 与融合后墙体缺失的一项高收益改变，给出强反例、匹配对照、完整训练暴露/VALIDATION成本和失败停止条件。
4. 将 HouseExpo 与非墙 occupied 多样性纳入版本化计划，先防数据泄漏，不覆盖历史基准。
5. 数学和代码审查通过、硬件/磁盘安全时再登记新批；在原生监督和真实信息瓶颈下报告结果。不承诺“24小时必有突破”。

交接目的不是维护某个版本号，而是找到**64 维可通信证据、restriction 的相容性、正确目标能量和高保真 decoder 能协同成立的模型**。
