# 多机器人 SLAM 与协同感知：传输内容、带宽-精度权衡、对位姿/配准误差的鲁棒性

> 调研日期：2026-10-27。范围：3D、scan/feature 共享、带宽曲线、位姿误差鲁棒性（2D 地图合并与一般性"通信高效地图共享"已由本项目早先报告覆盖，这里不重复）。
> 记号：kB/KB 按原文照抄；"AP@0.5/0.7" 指 3D box AP@IoU；"σt/σr" 指平移 (m) / 偏航 (°) 高斯噪声标准差。
> 来源可靠性：凡标注"(仅搜索摘要)"的数字，来自搜索引擎对原文的摘要，没有打开全文核对，请降级使用。

## 问题 1：分布式 / 多机器人 SLAM 交换什么、带宽多少、精度如何

### Takeaway
主流 C-SLAM（DOOR-SLAM、Kimera-Multi、Swarm-SLAM、D2SLAM、DCL-SLAM、COVINS）几乎都用"两阶段"通信：先广播紧凑的 global descriptor（BoW / NetVLAD / CosPlace / ScanContext / LiDAR-Iris，约 1–30 kB/keyframe），匹配上后才按需发送 local features 或点云（约 20–100 kB）做几何验证，后端只交换位姿图变量（每条约 0.1 kB）。稠密地图本身通常**不传**：Kimera-Multi 每台机器人只用收到的位姿约束在本地变形自己的 mesh。实测带宽在几十到几百 kB/s 量级（集中式 COVINS 约 0.5 MB/s 上行），任务总量在几十 MB 到几 GB 之间；轨迹精度为厘米级（EuRoC 室内）到米级（KITTI / 校园）。

### Cited Findings
**Kimera-Multi (Tian et al., T-RO 2022)**
- 通信流程：机器人在通信范围内互发新 keyframe 的 DBoW2 bag-of-words 描述子；一旦匹配，再请求该 keyframe 的 3D keypoints + descriptors 做几何验证。分布式 PGO（RBCD）只交换"public poses"（参与机器人间回环的位姿）；GNC 的回环权重由一方算好后发给另一方。论文描述的通信只覆盖回环检测和 PGO，**没有 mesh 传输**，mesh 校正在每台机器人本地完成 — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)
- 声称相对集中式基线最多减少 70% 通信 — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)
- Table II 的通信量（抽取文本缺表头，列含义和单位是根据"三列相加等于总量"推断的，推测单位 MB）：Medfield 为 PR 22.6 / GV 41.5 / PGO 1.8 / 总计 65.9；City 69.5；Camp 59.4；EuRoC Vicon Room 1 27.8；Vicon Room 2 24.4；Machine Hall（5 机）145.7。同表还有未标注的大数（如 Medfield 的 2918、3104），可能是"传全部图像 / 全部 keypoints"的集中式基线 — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)
- ATE（m，对 GT 轨迹）：D-GNC 在 Medfield 3.92 / City 0.85 / Camp 0.96 / V1 0.35 / V2 0.47 / MH 0.41，与集中式 GNC（3.88 / 1.00 / 1.33 / 0.36 / 0.43 / 0.52）相当；L2（不做鲁棒化）在 Medfield 为 64.2 m，PCM 为 12.5 m — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)
- 语义 mesh 精度（% 正确标注点，Kimera-Semantics → 本地 mesh 优化 LMO）：Camp 合并地图 79.4 → 95.2；City 合并地图 76.1 → 78.8 — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)
- 合成 2D PGO（INTEL，70% 外点回环）中有一个实例：最终 ATE 为 PCM 2.24 m、naive 初始化 GNC 11.59 m、PCM+GNC 0.09 m、鲁棒初始化 GNC 0.003 m — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)

**Swarm-SLAM (Lajoie & Beltrame, RA-L 2024)**
- 每个 keyframe 提取紧凑 global descriptor 并广播给邻居：LiDAR 用 ScanContext，图像用 CosPlace。相遇时先做 bookkeeping，只补发对方没有的描述子。候选回环按 budget 选择，可用 greedy 或新的 spectral（最大化位姿图 algebraic connectivity）优先级；local features 的交换用 vertex-cover 形式最小化 — [Swarm-SLAM arXiv](https://arxiv.org/abs/2301.06230)
- 后端：在去中心化选举出的单个机器人上求解 GNC 位姿图。在 KITTI-360 09（5 机，10 km）上，对照的 DGS+PCM 和 D-GNC "需要超过 5 倍的数据传输"，且精度不如它 — [Swarm-SLAM arXiv](https://arxiv.org/abs/2301.06230)
- 真实实验：3 台机器人（Spot + 2 台 Scout），ad-hoc 网络，475.42 m，3103 keyframes，67 条机器人间回环（其中 10 条外点），**总通信 94.95 MB**（不含可视化），主要来自前端，与 keyframe 数成正比 — [Swarm-SLAM arXiv](https://arxiv.org/abs/2301.06230)
- 评测口径：Table II 的 ATE 对 GPS 经纬度；但 spectral vs greedy 的 Fig. 3 中，ATE 是相对"包含所有回环的最终估计"计算的，不是相对外部 GT — [Swarm-SLAM arXiv](https://arxiv.org/abs/2301.06230)

**D2SLAM (Xu et al., T-RO 2024)**
- 按距离分模式：near 模式（约 3–5 m 内）广播完整 keyframe（SuperPoint 特征 + 全局描述子）；far 模式只发 compact keyframe（仅 NetVLAD，"to save bandwidth"）。回环时发 compact keyframe，匹配成功后对方回传完整 keyframe。描述子用 PCA 降维，但未报告压缩比 — [D2SLAM arXiv](https://arxiv.org/html/2211.01538)
- 消息大小（Table VII）：complete keyframe 20.1 kB（stereo）/ 56.3 kB（omni）@5 Hz；compact keyframe 1.2 kB / 4.8 kB @5 Hz；D²VINS 更新 0.188 + 0.036·p kB @5 Hz；D²PGO 更新 0.056 + 0.064·e kB @1 Hz（TUM Corr 5 上约 81 kB/次）— [D2SLAM arXiv](https://arxiv.org/html/2211.01538)
- 总量（Table VIII，MB，前端 / D²VINS / D²PGO）：OmniLongYaw 5 机 984.2 / 70.8 / 2179.4（基线 1546.7 / 75.9 / 2340.9）；TUM ROOM 5 为 244.2 / 1.2 / 31.0；HKUST RI 3 为 105.4 / 9.6 / 11.9（基线前端 231.8）— [D2SLAM arXiv](https://arxiv.org/html/2211.01538)
- ATE 对比 DOOR-SLAM（pos m / rot °）：TUM ROOM 5 为 0.032/0.72 vs 0.064/1.47；TUM Corr 5 为 0.275/3.93 vs 0.068/1.26（DOOR-SLAM 更好，作者归因于 GT 不完整）；OmniLongYaw 5 为 0.168/3.33 vs 0.149/3.47 — [D2SLAM arXiv](https://arxiv.org/html/2211.01538)

**DCL-SLAM (Zhong et al., 2022/2023，LiDAR)；DiSCo-SLAM 的数字来自该文对比**
- 交换内容：LiDAR-Iris 全局描述子约 29.12 kB/keyframe（80×360 二值图 + row key）；有候选回环时才发过滤后的点云，约 99.92 kB/帧（VLP-16）；回环测量约 0.11 kB；相对位姿约 104 B/keyframe；分布式 PGO 消息约 0.10 kB。对照：原始点云约 385 kB，feature cloud 约 160 kB，单帧 VLP-16 scan 约 1 MB — [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)
- **DiSCo-SLAM 每 keyframe 约 200 kB**（VLP-16），DCL-SLAM 约 29 kB — [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)
- 每机平均发送：KITTI 81–188 kB/s；校园数据集 34–68 kB/s（平均 53.6 kB/s）；9 机实验每机平均发送 34.36 kB/s、接收 193.57 kB/s（接收随队伍规模增长）。分布式 PGO 约 3.46 MB，集中式 PGO 约 10.06 MB（Library 序列）— [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)
- 精度：KITTI 08 上 DCL-FAST-LIO2 ATE 4.58/4.54 m，DiSCo-SLAM 5.16/5.73 m；DiSCo-SLAM 在 KITTI 05、09 上合并失败；9 机校园实验平均 ATE 1.83 m — [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)

**COVINS (Schmuck et al., 3DV 2021，集中式，沿用 CCM-SLAM 的消息格式)**
- agent → server 发 keyframe 和 landmark，分"full"（含 2D keypoints + descriptors 等静态数据）和"update"（只发状态变化）两种消息；agent 以 5 Hz 发送，server 以 2 Hz 回传 — [COVINS arXiv](https://arxiv.org/html/2108.05756)
- 消息大小：KF full 97 kB，KF update 273 B，LM full 162 B，LM update 65 B。EuRoC 8 条序列上行平均 **493.36 kB/s**（402–609 kB/s），下行约 2.3 kB/s — [COVINS arXiv](https://arxiv.org/html/2108.05756)
- ATE：EuRoC MH01–03 为 0.024、MH01–05 为 0.036、V101–103 为 0.042（论文表注写 mm，但上下文显示应为 m）；12 架 UAV 村庄场景平均 ATE 0.050 m，尺度误差 0.44% — [COVINS arXiv](https://arxiv.org/html/2108.05756)

**DOOR-SLAM (Lajoie et al., RA-L 2020)**
- 点对点、不要求全连通；前端在不交换原始传感器数据的情况下检测机器人间回环；后端为分布式 PGO + 分布式 PCM 拒绝外点；摘要只说"low communication bandwidth"，没给数字 — [DOOR-SLAM arXiv](https://arxiv.org/abs/1909.12198)

### Inferences
- C-SLAM 的"消息"本质上是**用于配准的证据**（描述子 → 特征/点云 → 相对位姿约束）加**位姿图变量**，而不是地图本身。这与本项目"在消息里传占据信息再迭代融合"的设计不同。最接近的对照是 Kimera-Multi：每台机器人本地保留稠密地图，只传位姿约束，再本地变形地图。
- 带宽主要来自前端（Swarm-SLAM 明说"mostly attributable to the front-end"；D2SLAM 前端 MB 数比 D²VINS 大 1–2 个数量级）。在 8 机无线树拓扑下，如果要做跨机器人配准，配准证据的流量很可能比占据消息本身更大，需要单独做预算。
- 外点回环是 C-SLAM 精度的头号杀手（Kimera-Multi 中 L2 vs GNC 在 Medfield 相差 16 倍）。本项目若引入在线配准，需要 PCM/GNC 类的外点处理，否则一次错误配准的伤害远大于 5 cm/1° 的小误差。

### Gaps
- CCM-SLAM、LAMP 2.0、CoLRIO、DiSCo-SLAM 自身论文报告的带宽数字未能直接取得（DiSCo-SLAM 只有 DCL-SLAM 转述的约 200 kB/keyframe）。LAMP 2.0 把 keyed scans 发往基站的数据量没有查到。
- Kimera-Multi Table II 的列名和单位缺失，上面的列含义是推断的，需对照 PDF 核实。
- Swarm-SLAM Table II（各数据集 kB 与 ATE）因 PDF 排版无法可靠解析，只取到真实实验的 94.95 MB 和"对照方法 >5 倍数据"的结论。
- 这些 SLAM 论文都没有报告"地图（占据 / mesh）精度随带宽变化"的曲线，只有轨迹 ATE 和总通信量。

## 问题 2：协同感知（V2X / 多智能体）：带宽-精度曲线、压缩比、对位姿噪声的鲁棒性

### Takeaway
中间融合（传 BEV 特征）是主流。压缩比的报告口径是相对**未压缩 float32 特征图**：32×（V2X-ViT / V2V4Real 默认），4096× 时 AP 只掉约 3%（OPV2V），稀疏选择可以再降 5,000–100,000×（Where2comm），codebook 索引相对 Where2comm 还能再降约 1,200–1,300×（CodeFilling）。位姿噪声的通用设定是在 x、y、yaw 上加零均值高斯噪声，σ = 0.2 m / 0.2° 为"正常"水平，扫描到 0.6 m / 0.6°（有的到 1.0 m 或 2.0 m）。精细融合方法在 AP@0.7 上非常脆弱（FPV-RCNN 在 OPV2V 上 0.840 → 0.200），带显式位姿校正的方法（CoAlign）在 0.6 m/0.6° 下 AP@0.7 只掉约 4 个点。

### Cited Findings
**带宽-精度 / 压缩**
- Where2comm (NeurIPS 2022)：消息 = 空间置信图 R（作为下一轮的请求）+ 由二值选择矩阵稀疏化的特征 Z；只发非零特征及其索引。单个模型可在不同带宽下评估，覆盖整条曲线 — [Where2comm PDF](https://arxiv.org/pdf/2209.12836)
- Where2comm 在 OPV2V 上"通信量低 100,000 倍以上"，同时仍优于 DiscoNet 和 V2X-ViT；在 CoPerception-UAVs 上通信量比 When2com 少 5,000 倍且精度更高。横轴为 AP@0.5 vs 通信量 — [Where2comm arXiv](https://arxiv.org/abs/2209.12836)；[Where2comm PDF](https://arxiv.org/pdf/2209.12836)
- CodeFilling (CVPR 2024)：发送整数 codebook 索引而不是特征。字节数为 H×W×|M|×log2(nL)×nR/8（特征形式为 ×C×32/8）。相对 Where2comm 通信量降低 **1,333×（DAIR-V2X）/ 1,206×（OPV2VH+）**；其中仅"码索引表示"一项就带来 208× 而精度不变；在 100K 带宽约束下比此前 SOTA 高 5–38%（各设置的具体映射原文不清）— [CodeFilling arXiv](https://arxiv.org/html/2405.04966)
- OPV2V (ICRA 2022)：中间融合在 **4096× 压缩**下性能"只轻微下降（约 3%）"，仍优于 early/late fusion。按 27 Mbps @300 m 的 V2V 广播速率，4096× 压缩后单条消息约 5 ms — [OPV2V arXiv](https://arxiv.org/html/2109.07644)
- V2X-ViT (ECCV 2022)：所有中间融合方法默认用 1×1 conv 做 32× 通道压缩；V2X-ViT 在 128× 压缩下仍保持高性能；传输时间 t_c = f_s/v（v = 27 Mbps）+ 0–200 ms 均匀异步项 — [V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- FPV-RCNN (RA-L 2022)：基于 PV-RCNN keypoints 的特征融合，协同感知消息（CPM）**< 0.3 KB**，约比共享 BEV 特征图小 50 倍；通道 128 → 32 无明显掉点；COMAP 数据集上 AP@0.7 比 BEV keypoint 融合高约 9%，与两种无损原始数据融合基线相当；内含基于 maximum consensus 的定位误差校正模块 — [FPV-RCNN arXiv](https://arxiv.org/abs/2109.11615)
- 综述中的"Transmission (MB)"列（未压缩特征）：V2VNet 12.58（OPV2V/V2XSet）、3.15（V2X-Sim）；OPV2V/AttFuse 11.01；F-Cooper 6.29；CoBEVT 6.29；V2X-ViT 6.29（V2XSet）；Where2comm 11.01（V2XSet）；DiscoNet、Who2com/When2com 1.05（V2X-Sim）— [V2X collaborative perception survey (arXiv 2308.16714)](https://arxiv.org/html/2308.16714)

**位姿噪声鲁棒性（噪声模型 + AP 下降）**
- V2X-ViT：Noisy Setting = 发送方位置和朝向加高斯噪声，σ_xyz = 0.2 m、σ_heading = 0.2°，并加 100 ms 时延。Perfect → Noisy 的 AP@0.5 / AP@0.7：Late Fusion 0.727/0.620 → 0.549/0.307；Early Fusion 0.819/0.710 → 0.720/0.384；F-Cooper 0.840/0.680 → 0.715/0.469；OPV2V(AttFuse) 0.807/0.664 → 0.709/0.487；V2VNet 0.845/0.677 → 0.791/0.493；DiscoNet 0.844/0.695 → 0.798/0.541；V2X-ViT 0.882/0.712 → 0.836/0.614；No Fusion 0.606/0.402 — [V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- V2X-ViT 扫描：σ_xyz 0–0.5 m，σ_heading 0–1.0°。在 σ_xyz ≤ 0.2 m、σ_heading ≤ 0.4° 内，V2X-ViT 掉点 < 3%，其他中间融合至少掉 6%，early/late fusion 的 AP@0.7 掉近 20%；在 0.5 m / 1° 时 V2X-ViT 约 60%，其他方法显著退化 — [V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- CoAlign (ICRA 2023)：训练时加 σt = 0.2 m、σr = 0.2° 高斯噪声（x、y 和 yaw）；测试 σt/σr = 0/0、0.2/0.2、0.4/0.4、0.6/0.6；DAIR-V2X 还额外测试 Laplace 噪声。AP@0.7（0 → 0.6）在 **OPV2V** 上：CoAlign 0.912 → 0.868；V2X-ViT 0.856 → 0.823；V2VNet(robust) 0.854 → 0.826；V2VNet 0.740 → 0.504；**FPV-RCNN 0.840 → 0.200**；MASH 恒为 0.198 — [CoAlign arXiv](https://arxiv.org/html/2211.07214)
- CoAlign 在 V2X-Sim 2.0 上 AP@0.7：0.765 → 0.684（FPV-RCNN 0.838 → 0.282）；在 DAIR-V2X 上 AP@0.7：0.604 → 0.570（V2X-ViT 0.531 → 0.522；V2VNet 0.402 → 0.350）。CoAlign 的机制：agent-object 位姿图优化，用检测框一致性修正相对位姿。消融中 late fusion 在 OPV2V 的 AP@0.7 约为 0.907/0.490/0.275/0.239（表格抽取有错位风险）— [CoAlign arXiv](https://arxiv.org/html/2211.07214)
- Where2comm：沿用 V2VNet / V2X-ViT 的设定，零均值高斯噪声，σ 0–0.6 m。在 CoPerception-UAVs 上，V2VNet 在 > 0.4 m、DiscoNet 在 > 0.5 m 时低于 No Collaboration，Where2comm 始终高于单机。作者归因于置信图滤掉噪声特征 + transformer 选择性融合 — [Where2comm PDF](https://arxiv.org/pdf/2209.12836)
- CodeFilling：沿用 CoAlign 设定，σ 0–1.0 m；基线在约 0.6 m 以上失效（低于不协同），CodeFilling 始终高于不协同 — [CodeFilling arXiv](https://arxiv.org/html/2405.04966)
- 综述的鲁棒性实验（只在干净位姿上训练，加噪测试；噪声加在 LiDAR 位姿的 roll/pitch/yaw 和 X/Y/Z 上）：DAIR-V2X-C 上 **early fusion 在位置 σ = 0.6 m 时 AP@0.5 下降 70.5%**；中间融合和 late fusion 退化较少；DiscoNet 峰值精度低但退化很小；在 V2XSet 上，heading 误差的影响小于位置误差 — [V2X collaborative perception survey](https://arxiv.org/html/2308.16714)
- FeaKM (2025)：DAIR-V2X，x、y、θ 高斯噪声，测试到 "2.0/2.0"（原文未定义单位，推测 m/°）。AP@0.5 从 0 到 2.0/2.0：CoAlign 0.762 → 0.654；V2X-ViT 0.713 → 0.637；F-Cooper 0.740 → 0.650；FeaKM(8 pairs) 0.768 → 0.668；FeaKM(4 pairs) 0.726 → 0.692（最平，但起点更低）。AP@0.7：CoAlign 0.625 → 0.555 — [FeaKM arXiv](https://arxiv.org/html/2502.11003v1)
- DATA (2025)：在 0.6 m 定位噪声 + 1.0° 朝向噪声下，其他方法的 AP70 在 V2XSIM 上下降超过 30%、在 V2XSet 上下降约 9%；DATA 在 V2XSet 0.6 m 时保持 67.22% AP70（仅搜索摘要）— [DATA arXiv](https://arxiv.org/pdf/2507.18237)
- V2VLoc (2025)：用 LiDAR 定位替代 GNSS，按 GNSS 式噪声 0–4 m/° 扫描，报告 AP@0.3/0.5/0.7，指出不同模型的鲁棒程度差异很大（仅搜索摘要）— [V2VLoc arXiv](https://arxiv.org/pdf/2511.14247)

### Inferences
- **精细融合 + 严格指标 = 对小配准误差极敏感。** FPV-RCNN 只传最少的 keypoint 特征（< 0.3 KB），在干净位姿下 AP@0.7 很高（OPV2V 0.840），但 0.2 m/0.2° 就掉到 0.568，0.6 m 时只剩 0.200；同一噪声下 AP@0.5 掉得少得多。本项目"5 cm / 1° 让家具区域 F1 掉约 16 点"与此一致：家具尺度的 F1 相当于严格 IoU，要求的配准精度与体素尺寸同量级，而车辆 AP@0.5 能容忍分米级误差。所以文献里"0.2 m/0.2° 只掉 1 个点"的数字不能直接搬到室内细粒度占据上。
- 1° 朝向误差在杠杆臂 r 处产生约 0.0175·r 的位移（5 m 处约 9 cm，50 m 处约 0.9 m）。V2X 文献在 100 m 级范围内评测，旋转误差被放大，却仍得出"heading 影响小于 position"的结论，说明 heading 噪声取值（≤ 1°）相对其 box 尺度偏小。室内 8 机场景的杠杆臂短，平移和旋转的相对重要性需要用本项目自己的数字判断。
- 能"恢复精度"的方法分三类：(a) 显式位姿校正（CoAlign 的 agent-object pose graph；FPV-RCNN 的 maximum consensus；FeaKM 的特征 keypoint 匹配；V2VLoc 的 LiDAR 定位）；(b) 带噪训练 + attention / 置信度选择（V2X-ViT、Where2comm、V2VNet robust 版本）；(c) 完全不依赖位姿（MASH：完全平坦，但精度低）。(a) 带来的增益最大：OPV2V 0.6 m 时，CoAlign 0.868 vs V2VNet 0.504（AP@0.7）。这支持本项目"求和 log-odds + 位姿配准"的路线，学习部分更适合放在配准和不确定性加权上，而不是消息编码本身。
- late fusion 和 early fusion 对位姿噪声都比中间融合更脆弱（V2X-ViT：late AP@0.7 0.620 → 0.307，early 0.710 → 0.384）。与本项目对照：求和 log-odds 属于"late / map-level fusion"，如果没有配准修正，文献预期它会严重退化。

### Gaps
- How2comm（NeurIPS 2023）、What2comm（ACM MM 2023）、CoBEVT、DiscoNet 原文的带宽曲线具体数值和位姿噪声表格没有取到；Where2comm 附录 Table 3 的精确 AP-带宽数值也未抽取（PDF 中有数字但列名不可靠）。
- FeaKM 的噪声单位、DATA / V2VLoc 的完整表格未从全文核实。
- 未找到统一的"位姿鲁棒性 benchmark"（如 RCP-Bench 类）的可靠数字；各论文的数据集、噪声尺度、AP 阈值不同，不可横向直接比较。
- 几乎所有工作只在 x、y、yaw（SE(2)）上加噪；6-DoF（含 z、roll、pitch）噪声只在综述的实验中出现。

## 问题 3：传点云（early）vs 传特征（intermediate）vs 传地图/检测（late）：字节数与精度

### Takeaway
真实数据集上的典型量级：late fusion（box）约 **3 KB/帧**，early fusion（原始点云）约 **1 MB/帧**，32× 压缩后的中间特征约 **0.2 MB/帧**。在干净或近乎干净的位姿下，中间融合 ≥ early fusion > late fusion；early fusion 是信息上的"上界"，但对位姿误差最敏感。

### Cited Findings
- V2V4Real (CVPR 2023，真实双车，GPS 初始化位姿)，"AM (MB)" 列：No Fusion 0；Late 0.003；Early 0.96；F-Cooper / V2VNet / AttFuse / V2X-ViT / CoBEVT 均为 0.20（32× auto-encoder 压缩）。同步 AP@0.5/0.7：Late 55.0/26.7；Early 59.7/32.1；F-Cooper 60.7/31.8；V2VNet 64.5/34.3；AttFuse 64.7/33.6；V2X-ViT 64.9/36.9；CoBEVT 66.5/36.0；No Fusion 39.8/22.0。异步（100 ms 延迟）时 Early 降至 52.1/25.8，CoBEVT 58.6/29.7。AM 定义为"算法指定的传输数据大小"，没有给出公式 — [V2V4Real arXiv](https://arxiv.org/html/2303.07601)
- OPV2V：四种 detector 下，中间融合在 Default Towns 上普遍 ≥ early fusion（如 VoxelNet：Intermediate 0.906/0.864 vs Early 0.852/0.758 vs Late 0.801/0.738 vs None 0.688/0.526，AP@0.5/0.7）；PointPillar：0.908/0.815 vs 0.891/0.800 vs 0.858/0.781。论文定性认为 early fusion"需要大带宽、难以实时"，late fusion 带宽小但受单机性能限制 — [OPV2V arXiv](https://arxiv.org/html/2109.07644)
- V2X-ViT（Perfect）：Early 0.819/0.710 vs Late 0.727/0.620 vs V2X-ViT 0.882/0.712；文中提到大带宽下 early fusion AP 降到约 28%，作者以此说明压缩的必要性 — [V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- LiDAR 原始数据量参照：DCL-SLAM 报告原始点云约 385 kB、feature cloud 约 160 kB、单帧 VLP-16 scan 约 1 MB、全局描述子约 29 kB — [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)
- 极端稀疏的中间表示：FPV-RCNN 的 CPM < 0.3 KB，约为 BEV 特征共享的 1/50，干净位姿下与两种原始数据融合基线相当 — [FPV-RCNN arXiv](https://arxiv.org/abs/2109.11615)
- 综述实验：early fusion 在位置 σ = 0.6 m 时 AP@0.5 下降 70.5%，中间和 late fusion 退化较少 — [V2X collaborative perception survey](https://arxiv.org/html/2308.16714)

### Inferences
- 量级比例大致为 late : intermediate(32×) : early ≈ 1 : 70 : 300（V2V4Real：0.003 : 0.20 : 0.96 MB）。中间融合用约 1/5 的 early 字节取得更高 AP，这是"学习消息优于原始数据"在 V2X 中最常被引用的证据。但要注意，它的对照对象是**原始点云 + 无配准修正**，而不是本项目那样的"精确几何码"。在无噪声仿真中，early fusion 理论上应是上界；它在 OPV2V 上反而略输，很可能来自 detector 训练分布的差异，而不是信息论意义上的优势。
- 本项目的"精确几何码"更接近 early / map-level 共享。文献表明这类方案在位姿噪声下退化最严重，与本项目"5 cm/1° 掉 16 点"的观察方向一致。

### Gaps
- 未找到在**同一场景、同一 detector** 下给出 early/intermediate/late 的完整 AP-vs-字节 Pareto 曲线并同时扫位姿噪声的论文（V2V4Real 只给单点；V2X-ViT 有噪声但没有字节数）。
- 点云压缩（如 Draco / octree / G-PCC）后再做 early fusion 的系统性对比没有取到（EMP、AutoCast 等系统论文未检索）。

## 问题 4：带显式带宽预算的多机器人占据 / 语义地图共享（3D）

### Takeaway
3D 地图共享的压缩主要走参数化路线（GMM 子图：相对点云地图省 96–98% 的数据，相对稠密栅格最多少两个数量级），或走学习特征 + 稀疏化路线（协同语义占据 CoHFF：97× 压缩只损失约 5% IoU、约 14–15% mIoU；2026 年的 VQ 方法相对稠密 3D 表示最多 82×）。神经隐式多机建图（Di-NeRF、MNE-SLAM、MCN-SLAM）声称通信友好，但在已查到的材料中缺少绝对字节数。

### Cited Findings
- MR-GMMapping (Dong et al., RA-L 2022)：机器人之间只用 GMM 子图做相对位姿估计，并按需自适应选择高斯分量数。相对点云地图**数据传输减少 98%**；相对点云和 GMM-frame 基线，相对位姿估计的平移精度提升 11%、旋转精度提升 30%（摘要，经搜索确认）— [MR-GMMapping (Tsinghua NICS PDF)](https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2Fd375e8c2-f81f-451a-a5dc-efe2442b3af3.pdf)；[代码](https://github.com/efc-robot/gmm_map_python)
- MR-GMMExplore（后续工作）：探索仿真中通信负载比点云约低 96%（仅搜索摘要）— [Tsinghua NICS PDF](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/3f18206c-0703-4ad6-9e46-1d61efbca467.pdf)
- Corah, O'Meadhra, Goel, Michael (RA-L 2019)：以 GMM 作全局地图，通过低带宽链路共享，本地用 Monte Carlo ray tracing 从 GMM 生成占据栅格供规划；摘要只说"低通信率"。O'Meadhra 2018 学位论文报告 GMM 重建占据所需数据约为 Bayesian kernel 占据方法的 1/50；Goel 2024 学位论文（Self-Organizing GMM）称相对稠密栅格方法最多减少两个数量级带宽（仅搜索摘要）— [CMU RI 页面](https://www.ri.cmu.edu/publications/communication-efficient-planning-and-mapping-for-multi-robot-exploration-in-large-environments)；[O'Meadhra thesis](https://www.ri.cmu.edu/app/uploads/2018/08/Thesis-1.pdf)；[Goel thesis](https://www.ri.cmu.edu/app/uploads/2024/12/thesis.pdf)
- CoHFF (Song et al., 2024，协同 3D 语义占据，CARLA / Semantic-OPV2V)：只传两个正交平面（x-z、y-z）的 128 维特征，ego 保留 BEV 平面，用可学习 mask 稀疏化。稀疏率 → 通信量：0 → 16.53 MB；0.5 → 8.27 MB；0.8 → 3.31 MB；0.95 → 0.83 MB；0.99 → 0.17 MB。IoU/mIoU：协同 50.46/34.16 → 0.99 稀疏（约 97×）时 48.02/29.48；单车约 38.52/24.85（表格排版混乱，此映射为推断，但与文中 +31% IoU / +37% mIoU 一致）。正文另给出 100×100×8×128 网格从 39.05 MB 降至 0.78 MB（50×），与表 3 的 16.53 MB 基线不一致，原文未解释。附录用 0–0.6 m GPS 高斯噪声测试，称 IoU 稳定，但未给数值。GT 由 CARLA 语义 LiDAR（每车额外 18 个，覆盖遮挡区域）生成 — [CoHFF arXiv](https://arxiv.org/html/2402.07635v2)
- SAVQ (2026，Sparse-Aware Vector Quantization)：利用 3D 场景稀疏性只编码信息区域，相对传输稠密 3D 表示通信最多减少 82×；摘要未给 IoU 和位姿噪声结果 — [SAVQ arXiv](https://arxiv.org/abs/2607.01928)
- 纯视觉 Gaussian Splatting 协同语义占据 (2025)：以 CoHFF 34.6% 的带宽取得更好的占据预测（仅搜索摘要）— [arXiv 2508.10936](https://arxiv.org/pdf/2508.10936)
- 神经隐式多机建图：分布式 NeRF 只交换模型权重（无字节数）；MNE-SLAM (CVPR 2025) 指出 CP-SLAM 需在集中阶段传原始 keyframe，不满足带宽约束；RAMEN 指出 DiNNO / Di-NeRF 的共识法假设同步通信，消息延迟或丢失时可能发散；格拉斯哥一篇 edge-assisted ViT-NeRF SLAM 声称相对基于图像的多机 SLAM 带宽减少 > 90%（均仅搜索摘要）— [Distributed NeRF arXiv](https://arxiv.org/html/2409.20289v1)；[MNE-SLAM CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_MNE-SLAM_Multi-Agent_Neural_SLAM_for_Mobile_Robots_CVPR_2025_paper.pdf)；[RAMEN arXiv](https://www.arxiv.org/pdf/2502.19592)；[Glasgow eprint](https://eprints.gla.ac.uk/384373/2/384373.pdf)
- DCL-SLAM 的后端对比：分布式 PGO 约 3.46 MB vs 集中式 PGO 约 10.06 MB — [DCL-SLAM arXiv](https://arxiv.org/html/2210.11978)

### Inferences
- 与本项目最相关的两个数字：(1) GMM 子图相对点云省约 98% 且**配准反而更准**，说明紧凑的几何参数表示可以同时服务"地图共享"和"配准"两个用途；(2) CoHFF 97× 压缩时 IoU 只掉约 2.4 个点（50.46 → 48.02），但 mIoU 掉约 4.7 个点。细类别（类似本项目的家具区域）比整体占据对压缩更敏感，与本项目把家具区域 F1 作为主指标的做法一致。
- 协同语义占据（CoHFF / SAVQ / GS 方法）都基于 CARLA 仿真 GT，位姿噪声要么没测，要么只给定性结论。所以"学习消息在真实噪声下的价值"在占据任务上**没有被文献量化过**，这是本项目可以填补的空白。

### Gaps
- MR-GMMapping 的绝对字节数 / KB/s、Corah 2019 的具体压缩比，没有从全文核实。
- MNE-SLAM、MCN-SLAM、Di-NeRF 等神经隐式多机建图的 MB/轮或 KB/s 未查到；MCN-SLAM 摘要没有数字。
- Hydra-Multi（3D scene graph 共享）的带宽数字未检索。
- 未找到"信息论驱动的 3D 占据子图选择性共享 + 明确 KB/s 预算 + 地图 F1/IoU"三者同时报告的论文（2D 部分见本项目早先报告）。

## 问题 5：评测惯例：是否对干净 GT 打分？压缩比相对什么计算？

### Takeaway
协同感知几乎都对**干净 GT**（仿真器真值或人工标注的 ego 坐标系 box）打分，噪声只加在传输方位姿上，且多为 SE(2) 高斯噪声。通信量的事实标准是 **log2(字节数)**，字节数按 **float32 未压缩特征** × 选中位置数计算（DiscoNet 用 log10）。压缩比几乎都相对"未压缩 float32 特征图"或相对上一代方法（如 CodeFilling 对 Where2comm），而**不是**相对无损熵编码基线。C-SLAM 报告总 MB 或 kB/s，精度用 ATE 对 GPS / 动捕 GT，且通常不报告地图精度随带宽变化的曲线。

### Cited Findings
- Where2comm：通信量 = log2(|M| × D × 32/8)，|M| 为选中的空间格子数，D 为通道数，32 表示 float32，8 表示换算成字节。与 DiscoNet 唯一的区别是对数底（DiscoNet 用 10），因此数值约为后者的 3.32 倍 — [Where2comm PDF](https://arxiv.org/pdf/2209.12836)
- CodeFilling："遵循既往工作的标准"，用 log2(字节) 报告通信量；特征形式和码索引形式的字节公式见问题 2；压缩倍数（1,333× / 1,206×）是**相对 Where2comm** 而不是相对原始数据 — [CodeFilling arXiv](https://arxiv.org/html/2405.04966)
- OPV2V 的 4096× 和 V2X-ViT 的 32× / 128× 都是相对未压缩中间特征（1×1 conv 通道压缩 / auto-encoder）— [OPV2V arXiv](https://arxiv.org/html/2109.07644)；[V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- V2V4Real：AM 为"算法指定的传输数据大小"，无公式；位姿用 GPS 初始化（真实噪声），没有说明是否额外加噪；GT 为各车在自身 LiDAR 坐标系中标注，测试时变换到 ego，冲突时取 ego 的标注 — [V2V4Real arXiv](https://arxiv.org/html/2303.07601)
- 噪声施加方式：CoAlign 在训练全程加 σ = 0.2 m/0.2° 噪声，并在 DAIR-V2X 上用 Laplace 噪声测试分布外情况；综述实验则"干净训练、加噪测试、不微调"；V2X-ViT 把位姿噪声和 100 ms 时延绑在一起作为"Noisy Setting"，不分开报告 — [CoAlign arXiv](https://arxiv.org/html/2211.07214)；[V2X collaborative perception survey](https://arxiv.org/html/2308.16714)；[V2X-ViT arXiv](https://arxiv.org/html/2203.10638)
- 协同语义占据 GT 来自 CARLA 语义 LiDAR（含额外 18 个用于覆盖遮挡），作者承认真实世界性能未验证 — [CoHFF arXiv](https://arxiv.org/html/2402.07635v2)
- C-SLAM：Kimera-Multi 的 ATE 对 GT 轨迹，合成实验对"无外点位姿图的 MLE"；Swarm-SLAM 的 ATE 对 GPS，但其优先级实验对"含全部回环的最终估计"；COVINS 对 EuRoC 动捕真值，表注单位前后不一致 — [Kimera-Multi arXiv](https://arxiv.org/html/2106.14386)；[Swarm-SLAM arXiv](https://arxiv.org/abs/2301.06230)；[COVINS arXiv](https://arxiv.org/html/2108.05756)

### Inferences
- 本项目若想让结果能与文献对接，建议同时报告：(1) log2(bytes)，以 float32 为参照，便于和 Where2comm / CodeFilling 曲线对齐；(2) 相对**无损编码**的几何码（如熵编码后的 log-odds / octree）的压缩比，因为文献普遍缺这一项，而本项目"精确几何码占优"的结论正好依赖它；(3) 位姿噪声按 σt/σr 网格（建议覆盖文献的 0.2/0.2 和本项目的 0.05/1° 两种尺度），区分训练时是否带噪。
- 文献的"相对 float32 特征图的压缩倍数"会系统性夸大学习消息的优势：float32 特征本身就远大于等价的几何信息。把"精确几何码 + 无损熵编码"作为基线，正好补上这个评测上的空白。

### Gaps
- 没有找到在协同感知中把 learned codec 和 lossless / 经典点云编码（G-PCC、Draco、octree 熵编码）在同一 rate-distortion 坐标下比较的论文。
- 没有找到用"带传感器噪声 GT"或"多次扫描去噪 GT"评测的协同占据工作，全部为仿真干净 GT 或人工标注 box。
