# 协同（多智能体）预测、预报与世界模型：V2X 自动驾驶与多机器人（2023–2026 研究笔记）

> 检索日期：2026-10-28。凡 arXiv 编号 ≥ 2601 且未列会议者均为**预印本**。"（检索摘要）"表示该条只来自搜索结果摘要或聚合页、未读原文，可信度较低。数字一律按原文抄录，不做换算。笔记目的：判断"去中心化 + 固定尺寸潜变量消息 + 显式不确定性 + GBP 共识"在哪里新颖、在哪里有用。

## 问题 1：主流协同感知+预测/预报方法有哪些？各自传输什么、带宽多少、融合在哪里、不确定性如何表示、做过哪些鲁棒性实验、headline 数字是多少？

### Takeaway
2023–2026 的协同预测/预报方法**几乎全部是 ego 中心融合**：邻居把 BEV 特征、query、轨迹或目标框发给 ego，由 ego 端 Transformer/图网络/最大值融合；**没有一篇用共识或 GBP 做去中心化融合**。显式不确定性很少见，只有 CMP（GMM 轨迹）、CoAnchor（每个锚点带协方差 + Kalman 门控）、V2X-WAM（学习得到的可靠性图 × 时间指数衰减）和 Double-M/MOT-CUP 一类工作。鲁棒性实验大多只测时延（100–500 ms）和位姿高斯噪声（0.2 m/0.2° 到 1 m/1°），丢包实验很少（V2X-Graph、UniV2X 的 query 丢弃），异构实验集中在单帧检测（HEAL/GT-Space/HeteroPROMPT/ALF），而不在预测上。

### Cited Findings

**A. 协同感知 + 预测（感知到轨迹）**

- **V2XPnP**（Zhou, Ma 等，ICCV 2025；arXiv 2412.01812）：传输**中间 BEV 特征图**；每个 agent 先把自己的多帧历史 BEV 融合成一张图再发（"一步通信"），所以包大小与单帧协同感知相当 — [arXiv v3](https://arxiv.org/html/2412.01812v3)；[ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/1744)
  - 带宽：沿通道用 1×1 卷积压缩，压缩率 32×，多步传输从 5×0.269 Mb 降到 0.269 Mb；估计 C-V2X 一步传输时延约 10–20 ms；压缩到 ×128 仍优于全尺寸 V2X-ViT* — [arXiv v3](https://arxiv.org/html/2412.01812v3)
  - 融合：ego 中心（点云/特征变换到 ego 坐标，ego 解压后做多智能体空间注意力）；**不建模不确定性** — [arXiv v3](https://arxiv.org/html/2412.01812v3)
  - 鲁棒性：时延 100–500 ms，位姿高斯噪声 (0.2 m, 0.2°)–(1 m, 1°)；**没有丢包实验** — [arXiv v3](https://arxiv.org/html/2412.01812v3)
  - 指标：AP@0.5、ADE、FDE、MR（2 m 阈值）、EPA（τ=2 m，α=0.5）；VC 模式 V2XPnP 为 AP 71.6 / ADE 1.35 / FDE 2.36 / MR 31.7 / EPA 48.2，最强基线 V2X-ViT* 为 69.6 / 1.39 / 2.56 / 35.2 / 44.7；No Fusion 为 43.9 / 1.87 / 3.24 / 33.8 / 24.3 — [arXiv v3](https://arxiv.org/html/2412.01812v3)
  - 消融：一步通信优于多步通信（AP 71.6 vs 68.2；EPA 48.2 vs 43.0）；只用时间融合时 AP 57.2 / EPA 33.8，加空间融合后为 71.3 / 44.4，再加地图后为 71.6 / 48.2 — [arXiv v3](https://arxiv.org/html/2412.01812v3)
  - 基线：No Fusion、FaF*、Late/Early Fusion、CoBEVFlow*、FFNet*、V2X-ViT*（补充材料里还有 V2VNet*、F-Cooper*、DiscoNet*） — [arXiv v3](https://arxiv.org/html/2412.01812v3)
- **CMP: Cooperative Motion Prediction with Multi-Agent Communication**（arXiv 2403.17916，v3 已被 IEEE RA-L 接收，2025 年 2 月） — [arXiv v3](https://arxiv.org/html/2403.17916v3)
  - 传输分两阶段：(1) LiDAR BEV 特征经卷积自编码器压缩后连同位姿一起广播；256× 压缩时 OPV2V 从 82.5 降到 0.32 MB/s，V2V4Real 从 60.0 降到 0.23 MB/s；(2) 各车再广播自己的**预测轨迹假设（GMM 分量）**，延迟一帧 — [arXiv v3](https://arxiv.org/html/2403.17916v3)
  - 融合：感知在 ego 端（FuseBEVT）；预测聚合**在每辆 CAV 上各自做**（对自己的和收到的预测做 self-attention），这是本笔记里少有的"每个 agent 都聚合"的设计，但它不是共识 — [arXiv v3](https://arxiv.org/html/2403.17916v3)
  - 不确定性：每个目标的未来用 K=64 分量的高斯混合表示（权重、均值、方差、相关系数），用 L1 + NLL 训练；不建模检测或跟踪的不确定性 — [arXiv v3](https://arxiv.org/html/2403.17916v3)
  - 鲁棒性：固定 100 ms 时间窗，迟到的消息直接丢弃，所有结果都在 100 ms 时延下得到；**未测位姿噪声和丢包** — [arXiv v3](https://arxiv.org/html/2403.17916v3)
  - 数字（5 s）：OPV2V minADE6 1.858 vs V2VNet 2.117（低 12.3%），minFDE6 4.163 vs 4.904；V2V4Real minADE6 3.910 vs 4.443；单张 RTX 6000 Ada 上平均约 67.3 ms — [arXiv v3](https://arxiv.org/html/2403.17916v3)
- **UniV2X: End-to-End Autonomous Driving through V2X Cooperation**（Yu 等，AAAI 2025，pp. 9598–9606） — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33040)；[arXiv v3](https://arxiv.org/html/2404.00717v3)
  - 传输（路侧 → 车）：256 维 **agent query**（带 track ID 和参考点，只发有效 query）+ 256 维 **lane query** + 200×200 的 **BEV 占据概率图 + 概率流图**。未来占据按线性模型 P_future(t)=P0+t·P1 外推，所以只需 "2*200*200 floats"，不必发 T 张 — [arXiv v3](https://arxiv.org/html/2404.00717v3)
  - 时延补偿：QueryFlowNet（三层 MLP）预测 query flow — [arXiv v3](https://arxiv.org/html/2404.00717v3)
  - 带宽（论文记作 "BPS"）：规划表 8.09×10⁵，感知表 6.96×10⁴，建图表 1.47×10⁵；早期融合/BEV 特征融合等基线为 8.19×10⁷，V2X-ViT/CoBEVT 为 2.56×10⁶ — [arXiv v3](https://arxiv.org/html/2404.00717v3)。**单位有歧义**：V2X-VLM 把 8.09×10⁵ 理解为 bits per second — [V2X-VLM](https://arxiv.org/pdf/2408.09251)；V2X-WAM 给出的同类 UniMM-V2X 数字用的是 B/s — [V2X-WAM](https://arxiv.org/html/2609.37098v1)
  - 融合：ego 端（query 匹配后融合，占据用逐元素 max）；**不量化不确定性** — [arXiv v3](https://arxiv.org/html/2404.00717v3)
  - 鲁棒性：时延 0 vs 500 ms，有 flow 时 mAP 只掉 1.2（29.5→28.5），无 flow 掉 3.6；随机丢弃 10–100% 路侧 query，性能平滑下降，100% 时接近 No Fusion；带宽 0–1 Mb/s 扫描；**位姿误差只讨论未测试** — [arXiv v3](https://arxiv.org/html/2404.00717v3)
  - 数字（DAIR-V2X）：规划碰撞率 0.34%（No Fusion 0.89%），L2 3.48 m（反而略差于 No Fusion 的 3.44 m），检测 mAP 0.295（No Fusion 0.165），AMOTA 0.239（低于 V2X-ViT 的 0.287），占据 IoU 近/远 22.0/26.5 — [arXiv v3](https://arxiv.org/html/2404.00717v3)
- **UniMM-V2X**（arXiv 2511.09013，预印本，未列会议）：agent 之间**共享 query**，在感知和预测两级做多级融合，并用 MoE；在 DAIR-V2X 上相对 UniV2X 声称感知 +39.7%、预测误差 −7.2%、规划 +33.2%；abstract 未给带宽 — [arXiv](https://arxiv.org/abs/2511.09013)。V2X-WAM 报告它在 V2X-Seq-SPD 上平均 L2 为 1.49 m、传输 9.32×10⁵ B/s — [V2X-WAM](https://arxiv.org/html/2609.37098v1)
- **2024 年统一端到端 V2X 框架**（arXiv 2405.03971）：把车端和路侧的时序 BEV 特征融合成协同特征，并做运动/事故预测（检索摘要） — [arXiv](https://arxiv.org/pdf/2405.03971)

**B. 纯协同轨迹预报（目标级 / 轨迹级）**

- **V2X-Seq**（Yu 等，CVPR 2023；arXiv 2305.05938）：定义 VIC3D Tracking、Online-VIC Forecasting、Offline-VIC Forecasting 三个任务 — [arXiv](https://arxiv.org/abs/2305.05938)
- **V2X-Graph: Learning Cooperative Trajectory Representations for Motion Forecasting**（Ruan, Yu 等；**NeurIPS 2024**，arXiv 2023 年 11 月发布） — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1812042b83f20707a898ff6f8af7db84-Abstract.html)
  - 传输：**感知结果和轨迹**（track ID、位置、朝向、检测框、类别），**不传特征**，理由是通信受限 — [arXiv v2](https://arxiv.org/html/2311.00371v2)
  - 融合：在模型内部做轨迹特征级图融合，从 ego 视角解码；不确定性为 K=6 的 Laplace 混合 — [arXiv v2](https://arxiv.org/html/2311.00371v2)
  - 鲁棒性（V2X-Seq，minADE/minFDE/MR）：时延 0/100/200 ms 时分别为 1.0458/1.7855/0.2527、1.0468/1.7879/0.2562、1.0779/1.8385/0.2688；路侧数据随机丢失 0/10/30/50% 时 minADE 为 1.05/1.05/1.08/1.10；未见位姿误差实验 — [arXiv v2](https://arxiv.org/html/2311.00371v2)
  - V2X-Traj（V2V&I）：V2X-Graph 0.72/1.13/0.11，HDGT 0.94/1.56/0.17；但在 vehicle-only 设置下 HDGT 的 minFDE/MR 更好 — [arXiv v2](https://arxiv.org/html/2311.00371v2)
- **Co-MTP**（Zhang, Zhou 等，同济；arXiv 2502.16589 v3，页面未写会议）：路侧发送**历史轨迹和对其他目标的预测结果**；在车端做历史域 + 未来域（含 AV 自身规划轨迹）的异构图融合；K=6 模态 + 概率头 — [arXiv](https://arxiv.org/html/2502.16589)
  - 鲁棒性：0.2 m 位置噪声和 0.5 s 时延；作者称各模型对时延都稳健，噪声会削弱 Co-MTP 但它仍领先；无丢包扫描 — [arXiv](https://arxiv.org/html/2502.16589)
  - 数字（V2X-Seq，minADE/minFDE/MR）：Co-MTP 0.76/1.15/0.16，V2X-Graph 1.17/2.03/0.29（minADE −35%），HiVT 1.25/2.33/0.35 — [arXiv](https://arxiv.org/html/2502.16589)。注意：此表中 V2X-Graph 为 1.17，而 V2X-Graph 原文自报 1.05（[arXiv v2](https://arxiv.org/html/2311.00371v2)），两者设置可能不同，引用时需注明来源

**C. 时延/异步补偿（"预测未来特征"一系，是潜变量预测的前身）**

- **FFNet: Flow-Based Feature Fusion for VIC 3D Detection**（Yu 等，NeurIPS 2023）：发送压缩后的**特征流（feature flow）**，而不是单帧特征；路侧特征按流外推到车端时间，用于补偿异步；流用自监督方式训练；在 200 ms 时延下 mAP@BEV 62.87%，传输量约为早期融合的 1/100 — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ca5d2665de83394f437dad0c3746907-Abstract.html)；[arXiv](https://arxiv.org/pdf/2311.01682)（检索摘要）
- **CoBEVFlow: Asynchrony-Robust Collaborative Perception via BEV Flow**（Wei 等，NeurIPS 2023）：用 BEV flow 把异步特征重新放到正确位置；构建了含多种异步的合成数据集 IRV2V；在 IRV2V 和 DAIR-V2X 上评测 — [arXiv](https://arxiv.org/pdf/2309.16940)；[NeurIPS](https://neurips.cc/virtual/2023/poster/71517)（检索摘要）

**D. 协同占据预测 / 占据预报**

- **CoHFF: Collaborative Semantic Occupancy Prediction with Hybrid Feature Fusion in Connected Automated Vehicles**（Song 等，CVPR 2024；arXiv 2402.07635）：车辆之间共享"compressed orthogonal attention features"；在扩充了 3D 语义占据标签的协同数据集上，协同结果比单车好 30% 以上；abstract 未给带宽 — [arXiv](https://arxiv.org/abs/2402.07635)
- **Co3SOP: A Synthetic Benchmark for Collaborative 3D Semantic Occupancy Prediction in V2X-Enabled Autonomous Driving**（Wu 等，东京大学/NII；arXiv 2506.17004；项目页称 ICRA 2026 接收）：在 CARLA 中回放 OPV2V，配高分辨率语义体素传感器；基线在空间对齐后做注意力融合；协同范围越大收益越大 — [arXiv v2](https://arxiv.org/html/2506.17004v2)；[项目页](https://tlab.hongo.wide.ad.jp/portfolio/co3sop-a-collaborative-3d-semantic-occupancy-prediction-dataset-and-benchmark-for-autonomous-driving/)（检索摘要）
- **VQSOP: Sparse-Aware Vector Quantization for Bandwidth-Efficient Collaborative 3D Semantic Occupancy Prediction**（Li, Zhang, Chen；arXiv 2607.01928；注释写 "Accepted by ECCV26"）：利用 3D 场景稀疏性做向量量化，通信量最多降 82×；abstract 没写传的是码本索引还是量化特征 — [arXiv](https://arxiv.org/abs/2607.01928)
- **Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction**（Chen, Huang, Bagchi；AAAI 2026 oral；arXiv 2508.10936）：共享**稀疏 3D 语义高斯基元**；相对单车 +8.42 mIoU / +5.11 IoU，相对协同基线 +3.28 mIoU / +22.41 IoU；只用 34.6% 的通信量时仍有 +1.9 mIoU — [arXiv](https://arxiv.org/abs/2508.10936)
- **BEV-V2X: Cooperative Birds-Eye-View Fusion and Grid Occupancy Prediction via V2X-Based Data Sharing**（Chang 等；ResearchGate 标注为 IEEE T-IV 稿件）：用注意力融合 CAV 与路侧数据，**预测未来全局 BEV 占据栅格** — [ResearchGate PDF](https://www.researchgate.net/profile/Cheng_Chang52/publication/372296620_BEV-V2X_Cooperative_Birds-Eye-View_Fusion_and_Grid_Occupancy_Prediction_via_V2X-Based_Data_Sharing/links/64aeb26b8de7ed28ba937dc1/BEV-V2X-Cooperative-Birds-Eye-View-Fusion-and-Grid-Occupancy-Prediction-via-V2X-Based-Data-Sharing.pdf)（检索摘要）。它的扩展版 arXiv 2312.05104（Li 等）**已被作者撤回**（"has not received unanimous consent from all the members"） — [arXiv](https://arxiv.org/abs/2312.05104)
- 未来占据 + 流的预报还出现在 UniV2X（概率流）和 V2X-WAM（10 步×0.5 s 的占据和动态流，见问题 2） — [UniV2X](https://arxiv.org/html/2404.00717v3)；[V2X-WAM](https://arxiv.org/html/2609.37098v1)

**E. 异构与目标级共享（多为单帧检测，不是预测）**

- **HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception**（Lu 等，ICLR 2024）：先用初始 agent 和多尺度 Pyramid Fusion 建立共享特征空间，新类型 agent 通过"backward alignment"在本地单独训练编码器映射进来；在 OPV2V-H 上连续加入 3 种 agent 时可训练参数比 SOTA 少 91.5%；要求 BEV 特征 — [arXiv](https://arxiv.org/html/2401.13964v3)；[ICLR](https://iclr.cc/virtual/2024/poster/18889)
- **GT-Space**（Wang, Xu, Tan；ICLR 2026；arXiv 2603.19308）：用 GT 标签编码的 BEV 构造公共特征空间，每个 agent 只需一个轻量 projector，GT 空间只在训练时使用；传输压缩 BEV 特征，未报告带宽；AP@50：OPV2V 0.891 vs HEAL 0.887 / STAMP 0.876，V2XSet 0.874 vs HEAL 0.854，RCooper(A1) 0.477 vs HEAL 0.478；称在各位姿误差等级下保持领先，500 ms 时延下仍优于基线 — [arXiv](https://arxiv.org/html/2603.19308v1)；[ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/380a0b16a7e6f8c5010f798c9f2d3c61-Paper-Conference.pdf)
- **HeteroPROMPT**（arXiv 2607.26283 v2，abs 页标题拼成 "HeteroPROPMT"；Maleki, Radha，MSU；未列会议）：共享中间 BEV 特征，用 PARAFAC 分解的视觉 prompt + FiLM（DRIFT 模块）把新模态对齐到 ego 中心空间，主干冻结，可训练参数 0.064M；OPV2V-H 上 +m2 AP50/70 为 89.0/78.6 vs HEAL 87.2/78.3；未报告带宽和推理时延；**明确把时延、丢包、定位误差列为未来工作** — [arXiv](https://arxiv.org/html/2607.26283)
- **ALF: Adaptation-Free Heterogeneous Collaborative Perception with Unseen Agent Configurations**（Bae, Ahn；arXiv 2605.26642）：只传**框级消息**，约 **120 字节/agent/帧（10 Hz 时约 9.6 kbps）**；在 ego 端把框转成伪 BEV 图，再结合 ego 场景上下文合成兼容的潜特征；V2X-Real 上 64 组零样本设置中，mAP@0.7 相对最强基线提升 35.91% — [arXiv](https://arxiv.org/abs/2605.26642)
- **CodeFilling**（Hu 等，CVPR 2024）：用**共享码本的整数码**代替高维特征图，并按"信息填充"选择要发的消息；相对 Where2comm 通信量在 DAIR-V2X 上低 1,333 倍，在 OPV2VH+ 上低 1,206 倍；码本也可作为异构模型的统一特征空间 — [arXiv](https://arxiv.org/abs/2405.04966)
- **Which2comm**（arXiv 2503.17175）：传目标级稀疏特征而不是稠密特征图（检索摘要） — [arXiv](https://arxiv.org/abs/2503.17175v2)；**CoopDETR**（arXiv 2502.19313）：智能体之间共享目标级 query（检索摘要） — [arXiv](https://arxiv.org/pdf/2502.19313)

**F. 时空错位鲁棒**

- **CoAnchor: Robust Collaborative Perception under Spatio-Temporal Misalignment via Object-Level Anchors**（Li 等，BUPT；arXiv 2608.21055；ACM MM 2026） — [arXiv](https://arxiv.org/html/2608.21055)
  - 传输：仍发时延后的**稠密 BEV 特征**，另加稀疏目标状态锚点（位置、平面速度、朝向）；未报告带宽 — [arXiv](https://arxiv.org/html/2608.21055)
  - 不确定性：**每个锚点带协方差矩阵**，用当前帧 ego 观测做 Kalman 式更新，并用 NIS χ² 门限 7.815 做门控。这是本笔记中 V2X 侧最接近"显式高斯不确定性 + 概率融合"的方法，但仍是 ego 中心 — [arXiv](https://arxiv.org/html/2608.21055)
  - 鲁棒性：时延 100/200/300 ms，位姿噪声最高 (0.9 m, 0.9°)；V2V4Real Joint-Hard 下 AP@0.5/0.7 为 67.04/44.05，cascade 基线 baseST（CoAlign→TraF-Align）为 61.04/37.69；OPV2V Joint-Hard 下为 95.15/87.96 vs 90.19/81.21；干净设置下只是"competitive" — [arXiv](https://arxiv.org/html/2608.21055)

**G. 不确定性量化**

- **Double-M**（ICRA 2023）：协同检测的单次推理不确定性，同时给出认知和偶然不确定性（用 moving block bootstrap，并对每个框角点直接建模多元高斯）；在 V2X-Sim 上不确定性分数提升 4 倍以上 — [arXiv](https://arxiv.org/abs/2209.08162v2)（检索摘要）
- **MOT-CUP**（RA-L 2024）：用直接建模 + conformal prediction 量化协同检测的不确定性，并传播到运动预测和数据关联；在 V2X-Sim 上准确率 +2%，不确定性降低 2.67 倍 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2303.14346)（检索摘要）
- MAEAD 综述指出，可靠性研究正在走向 "uncertainty-aware fusion"，并以 Hyper-V2X（为协同 BEV 分割估计认知和偶然不确定性）为例 — [MAEAD](https://arxiv.org/html/2606.13840)

### Inferences
- **融合拓扑方面的空白很明确**：上面所有 V2X 方法都由 ego（或路侧→ego）做一次性前向融合。CMP 的"每辆车各自聚合预测"最接近分布式，但它不迭代，也不保证一致。所以"每个 agent 维护自己的信念，通过多轮 GBP 消息收敛到一致的共享未来"在 V2X 预测文献里没有对应工作（依据：A–F 中每个方法的融合描述）。
- **固定尺寸潜变量消息已有先例，但都只是压缩手段，不是概率对象**：V2X-WAM 用 12×64×64 INT8（49,200 B），V2XPnP 用 1×1 卷积做 32× 压缩，CodeFilling 发整数码，VQSOP 用 VQ（最多 82×），ALF 每帧 120 B。这些消息都没有附带精度/协方差，所以"消息 = (均值, 精度) 的固定尺寸潜变量"是一个可辩护的新点。
- **可用作参照的带宽量级**：从约 9.6 kbps（ALF 框级）、约 98 kB/s（V2X-WAM），到 0.23–0.32 MB/s（CMP 256×）、约 0.27 Mb/帧（V2XPnP），再到 8.19×10⁷ "BPS"（UniV2X 表中的稠密 BEV 基线）。新方法应在同一张"性能–带宽"图上与这些点对比。
- **鲁棒性的事实标准协议**：时延 100–500 ms，位姿 (0.2 m, 0.2°)–(1 m, 1°)，丢包 10–50%（V2X-Graph）或 query 丢弃 10–100%（UniV2X）。V2X-Graph 已给出逐档数字，可直接作为轨迹级去中心化方法的对照表。
- **预测级的异构问题几乎没人做**：HEAL/GT-Space/HeteroPROMPT/ALF 都是检测。"不同模型的智能体通过共享潜空间 + 共识做预测"是第二个空白。

### Gaps
- **"HERO"（目标级共享）**：用三种措辞检索都没有找到对应论文，无法确认它是否存在、标题是什么。检索到的近似工作是 ALF（框级消息）、Which2comm（目标级稀疏特征）和 CoopDETR（目标 query）。
- CoHFF、Co3SOP、VQSOP 的具体 mIoU 数值、带宽和鲁棒性实验没有读到（只看了 abstract）。
- BEV-V2X 原文（Chang 等）的最终发表信息、预测时域和指标没有核实。
- UniV2X 的 "BPS" 究竟是 bit 还是 byte，原文说明没有读到；两篇第三方论文的解读互相矛盾。
- FFNet 和 CoBEVFlow 的数字来自检索摘要，未读原文表格。
- Co-MTP 的会议归属没有在 arXiv 页面上确认。

## 问题 2：是否已有人做出协同*世界模型*（多车或多机器人的共享潜在未来预测/规划）？它们怎样处理消息时延或丢失？

### Takeaway
截至 2026-10，**只有一篇明确称为"协同世界（动作）模型"的 V2X 工作：V2X-WAM**（2026-09 预印本）。它预测的是**显式 BEV 占据 + 动态流**而非纯潜变量，融合仍以 ego 为中心，时延只用时间指数衰减权重处理，**没有测试时的时延、丢包或位姿鲁棒性实验**。HanoiWorld 本身是**单智能体** JEPA 世界模型，只在相关工作 §2.5 中**呼吁**协同世界模型应从"当前状态融合"转向"未来状态预测"，并且在消息延迟或缺失时能平滑退化。MARL 领域有去中心化世界模型（CoDreamer、Decentralized Collective World Model），但都**没有测试消息丢失或时延**，也都不是概率共识。

### Cited Findings
- **V2X-WAM: A Cooperative World Action Model for End-to-End Autonomous Driving**（You 等；arXiv 2609.37098v1，2026-09-29，预印本） — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 消息：路侧特征投影为 **12×64×64**，逐通道对称 **INT8** 量化；每条 49,200 B（含 12 个通道 scale），2 Hz，共 98,400 B/s — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 融合：ego 中心；可靠性 = 学习得到的空间可靠性图（由 ego BEV 和解码后的消息预测）× 按时间戳差计算的**指数衰减时间置信度**，逐元素作用在路侧消息上 — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 世界模型：显式 BEV，以规划动作/航点为条件，预测未来 10 步×0.5 s（最长 5 s）的占据和动态流；用"flow-guided occupancy transport"把上一时刻占据按流 warp，再与直接预测融合 — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 鲁棒性：**没有报告时延、位姿噪声或丢包的测试**；训练时有 BEV dropout/噪声和 3% 的路侧消息 dropout — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 数据与数字（V2X-Seq-SPD）：平均 L2 0.97 m（1/2/3 s 为 0.53/0.92/1.45），平均碰撞率 0.01%；相对 UniMM-V2X，L2 低 34.9%（基线 1.49 m），传输少 89.4%（基线 9.32×10⁵ B/s）；占据 IoU 23.07% vs StreamingFlow 20.61%，flow EPE 1.08 vs 1.06 m（略差） — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 消融：Ego Only 为 1.11 m / 0.05% / 19.80% / 1.15；去掉可靠性模块为 1.22 m / 0.10% / 22.15% / 1.15 — [arXiv](https://arxiv.org/html/2609.37098v1)
  - 相关工作：作者称"action-conditioned cooperative world modeling ... remain[s] largely unexplored"，列举的 Drive-WM、LAW、World4Drive 等均为单车 — [arXiv](https://arxiv.org/html/2609.37098v1)
- **HanoiWorld: A Joint Embedding Predictive Architecture Based World Model for Autonomous Vehicle Controller**（Tran Tien Dat 等；arXiv 2601.01577，2026-01-04，预印本） — [arXiv abs](https://arxiv.org/abs/2601.01577)
  - 方法是**单智能体**：JEPA + RNN 世界模型，actor-critic 沿用 DreamerV3 的想象 rollout 训练；环境为 HighwayEnv（highway-v0、merge-v0、roundabout-v0），64×64 RGB，每个环境 100 个评测回合 — [arXiv html](https://arxiv.org/html/2601.01577v1)
  - 只有相关工作 §2.5 讨论协同，原句为 "Cooperative world models must go beyond "current-state fusion" to "future-state prediction" under shared information"，并提到需要 "models that degrade gracefully when messages are delayed or missing"；方法、实验和未来工作中都没有涉及 — [arXiv html](https://arxiv.org/html/2601.01577v1)
  - 碰撞率（DreamerV3 / VQ-VAE / HanoiWorld）：highway 0.550/1.000/0.200；merge 0.030/0.290/**0.970**；roundabout 0.500/0.570/0.340。平均回报在 highway 和 merge 上 DreamerV3 更高 — [arXiv html](https://arxiv.org/html/2601.01577v1)
- **MAEAD: Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models**（Hu, Fang 等；arXiv 2606.13840，v4 为 2026-08-19，综述，约 400 篇） — [arXiv](https://arxiv.org/abs/2606.13840)
  - 提出 Shared World Models（SWM，跨智能体的预测性表示）这一框架；优先方向是 "verifiable shared-state maintenance"、意图/计划对齐、通信和算力约束下的安全协同；并指出评测集中在仿真、精选基准和离线协议上 — [arXiv](https://arxiv.org/abs/2606.13840)
  - 明确写道 "persistent edge-hosted SWM maintenance and closed-loop multi-agent planning remain open system problems"，以及 "cross-agent alignment and safety-critical coordination remain open" — [arXiv html](https://arxiv.org/html/2606.13840)
  - 文中"共识"只出现在 What2Keep（"consensus-based feature selection"，CVIU 2026）一处；**没有提到 belief propagation** — [arXiv html](https://arxiv.org/html/2606.13840)
- **OmniV2X: A Generative Foundation Planner for Efficient End-to-End Cooperative Driving**（Peng 等；arXiv 2606.21165；IROS 2026 接收）：使用 "lightweight, standard-compliant V2X tokens"，带宽不到 1%，微调只用不到 10% 的 V2X 数据；在 DAIR-V2X-Seq 上评测；abstract 未给时延/丢包实验 — [arXiv](https://arxiv.org/abs/2606.21165)
- **UniV2X 的"共享未来"雏形**：传输占据概率图 + 流，接收方按 P0+t·P1 线性外推未来；QueryFlowNet 补偿时延（500 ms 下 mAP 只掉 1.2） — [arXiv v3](https://arxiv.org/html/2404.00717v3)
- **FFNet**：传特征流，在车端把路侧特征外推到当前时刻（"预测未来特征"），在 200 ms 时延下评测 — [arXiv](https://arxiv.org/pdf/2311.01682)（检索摘要）
- **CoDreamer: Communication-Based Decentralised World Models**（Toledo, Prorok；arXiv 2406.13600；RLC 2024 CoCoMARL workshop） — [arXiv](https://arxiv.org/html/2406.13600v1)
  - 用 GATv2 在世界模型的**潜变量**（观测、随机后验/先验、512 维 GRU 循环状态、32×32 类别潜变量）之间以及策略之间传消息；通信范围 C 以内建边；作者称是 "a fully decentralised framework" — [arXiv](https://arxiv.org/html/2406.13600v1)
  - **未测试消息丢失或时延**；只提到想象 rollout 中边特征可能过时 — [arXiv](https://arxiv.org/html/2406.13600v1)
  - 结果：VMAS IQM 0.84 vs IDreamer 0.81 / IPPO 0.79（置信区间部分重叠）；Melting Pot 0.48 vs 0.30 / 0.05 — [arXiv](https://arxiv.org/html/2406.13600v1)
- **Decentralized Collective World Model for Emergent Communication and Coordination**（Nomura, Aoki, Taniguchi, Horii；arXiv 2504.03353 v3，2025-10） — [arXiv](https://arxiv.org/html/2504.03353v3)
  - 两智能体玩具任务；每步交换一个**二维高斯消息的采样**；每个 agent 的消息先验取自对方的后验，用 **Product-of-Experts** 合并；用 InfoNCE 对比损失做表示对齐；**没有共识协议，也没有消息丢失/时延实验** — [arXiv](https://arxiv.org/html/2504.03353v3)
  - 去中心化 EC 的协同效果第二好，仅次于集中式 BC，最差的是无通信的 NC — [arXiv](https://arxiv.org/html/2504.03353v3)
- **Dreaming Of Others: Latent Teammate Modeling In World Models For MARL**（arXiv 2605.31361，2026 World Modeling Workshop poster，5 页概念文）：把 RSSM 潜变量分解为环境部分和队友部分，**没有实验结果** — [arXiv](https://arxiv.org/abs/2605.31361)
- **V2XCrafter**（arXiv 2605.29471）：多智能体扩散模型以邻车的潜状态为参考，用跨智能体注意力保持跨视角一致，用于协同检测的数据增强（检索摘要，未读原文） — [聚合页](https://awesomepapers.io/similarity-search/papers/2605.29471)

### Inferences
- **"协同潜在世界模型 + 去中心化共识"仍是空白**：V2X-WAM 是显式 BEV 且以 ego 为中心；CoDreamer 用潜变量去中心化，但消息是确定性的 GNN 聚合，没有不确定性，也没有一致性保证；DCWM 用高斯消息和 PoE，在精神上最接近（PoE 等于高斯因子相乘），但只有 2 个 agent、二维消息、单轮，且没有鲁棒性实验。
- **时延/丢失的处理手段目前只有四类**：(i) 指数衰减权重（V2X-WAM）；(ii) 学习的 flow 外推（UniV2X、FFNet、CoBEVFlow）；(iii) 超时丢弃（CMP 的 100 ms 窗口）；(iv) 插值（V2X-Graph）。都没有"按消息年龄把精度降低"并在推断中自动反映出来的概率机制。GBP 的消息自带精度，天然可以做"陈旧消息 → 精度衰减"，这是一个可以对外讲清楚的卖点（多机器人侧 RAMEN 也采用了相同思想，见问题 5）。
- HanoiWorld 的价值只在于它是一份"立场陈述"可供引用，不能作为协同世界模型的基线。

### Gaps
- 没有读到 V2X-WAM 的后 15% 正文（附录），无法确认是否有补充的鲁棒性实验。
- "CoDrive（跨车世界一致视频生成）"在一个 awesome-list 中被提及，但检索不到原文，无法确认是否存在。
- 没有找到多车**闭环**世界模型（例如在 CARLA 中用共享潜在想象做 MARL）的同行评审工作；这可能是真实的空白，也可能是检索没有覆盖到。

## 问题 3：哪些数据集/基准支持带序列的协同预报？规模和许可如何？哪些适合在单 GPU 上做第一个实验？

### Takeaway
真实数据中，**V2X-Seq（SPD 用于感知+预测，TFD 用于轨迹预报）**和 **V2XPnP-Seq / V2X-Real** 是主力；V2X-Traj（V2V&I 轨迹）随 V2X-Graph 一起发布。仿真数据有 OPV2V 及其衍生集（OPV2V-H、Co3SOP、IRV2V）。**几乎没有一个数据集在页面上写明数据许可**（V2X-Real 页脚为 "All Rights Reserved"，V2X-Seq 只写了"可在中国大陆不限量下载"），这需要在使用前单独核实。单 GPU 的首个实验，最现实的选择是 **V2X-Seq-TFD 或 V2X-Traj 的轨迹级（目标级）协同预报**：不需要处理 LiDAR，已有 V2X-Graph/Co-MTP 的逐档时延/丢包数字可以对照。

### Cited Findings
| 数据集 | 年份/会议 | 规模 | 智能体 | 任务/特点 | 许可（已核实部分） | 来源 |
|---|---|---|---|---|---|---|
| V2X-Seq-SPD | CVPR 2023 | >15,000 帧，95 个场景，10 Hz | 1 车 + 1 路侧 | VIC3D 跟踪；UniV2X、V2X-WAM 的端到端评测 | GitHub 未写数据许可，只写"can be unlimitedly downloaded within mainland China"；arXiv 论文为 CC BY 4.0 | [arXiv](https://arxiv.org/abs/2305.05938)；[GitHub](https://github.com/AIR-THU/DAIR-V2X-Seq)；[V2X-WAM](https://arxiv.org/html/2609.37098v1) |
| V2X-Seq-TFD | CVPR 2023 | 约 80k 路侧视角 + 80k 车视角 + 50k 协同视角场景；28 个路口；672 小时；含矢量地图和信号灯 | 车 + 路侧 | Online/Offline-VIC 预报 | 同上 | [arXiv](https://arxiv.org/abs/2305.05938)；[GitHub](https://github.com/AIR-THU/DAIR-V2X-Seq) |
| V2X-Seq（V2X-Graph 中的切分） | — | 51,146 个场景，10 s，10 Hz，5 s 历史预测 5 s | 车 + 路侧 | minADE/minFDE/MR，K=6 | — | [arXiv](https://arxiv.org/html/2311.00371v2) |
| V2X-Traj | NeurIPS 2024 | 10,102 个场景（6,062/2,020/2,020），8 s，10 Hz，4 s 预测 4 s，北京路口 | 2 AV + 1 路侧 | V2V、V2I、V2V&I 轨迹预报 | 论文页未写许可 | [arXiv](https://arxiv.org/html/2311.00371v2) |
| DAIR-V2X | CVPR 2022 | 71,254 图像帧 + 71,254 点云帧 | 车 + 路侧 | 检测；UniV2X、UniMM-V2X 的主评测集 | 代码 Apache-2.0；页面未写数据许可 | [GitHub](https://github.com/AIR-THU/DAIR-V2X) |
| V2XPnP-Seq | ICCV 2025 | 论文：96 个场景（49 个 2V+2I、38 个 V2V、9 个 V+2I），每个 95–283 帧，10 Hz，约 40k 帧，145 万 3D 框，20.8 万 RGB 图像，10 类 | 2 CAV + 2 路侧 | 感知+预测；VC/IC/V2V/I2I 四种模式 | 论文和 GitHub 均未写许可；GitHub Dataset 1.0 只有 68 个场景，"v2.0 共 96 个场景，计划 2026/01" | [arXiv](https://arxiv.org/html/2412.01812v3)；[GitHub](https://github.com/Zewei-Zhou/V2XPnP) |
| V2X-Real | ECCV 2024 | 33K LiDAR 帧，171K 相机帧，>120 万 3D 框，10 类 | 2 CAV + 2 路侧 | 检测（VC/IC/V2V/I2I），OPV2V 格式 | 项目页页脚 "All Rights Reserved"（© 2023 UCLA Mobility Lab），未见开放许可 | [arXiv](https://arxiv.org/abs/2403.16034v2)；[项目页](https://mobility-lab.seas.ucla.edu/v2x-real) |
| OPV2V | ICRA 2022 | 73 个场景，每个约 25 s | 2–7 CAV（CARLA） | 检测；CMP 用它做预测 | OpenCOOD 代码 MIT；页面未写数据许可 | [CMP](https://arxiv.org/html/2403.17916v3)；[OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) |
| V2V4Real | CVPR 2023 | 67 个真实场景（美国），每个 10–20 s | 2 车 | 检测；CMP 用它做预测 | — | [CMP](https://arxiv.org/html/2403.17916v3) |
| TUMTraf V2X | CVPR 2024 | 2,000 帧标注点云，5,000 张标注图像，3 万个带 track ID 的 3D 框，5 个路侧 + 4 个车载传感器 | 车 + 路侧 | 检测/跟踪 | arXiv 论文为 CC BY 4.0；数据许可未核实 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zimmer_TUMTraf_V2X_Cooperative_Perception_Dataset_CVPR_2024_paper.html)；[arXiv](https://arxiv.org/html/2403.01316v1) |
| Co3SOP | ICRA 2026 | OPV2V 在 CARLA 中回放，[1000,1000,70] 体素，0.1 m | V2V | 协同语义占据 | 代码在 GitHub tlab-wide/Co3SOP | [arXiv](https://arxiv.org/html/2506.17004v2)（检索摘要） |
| IRV2V | NeurIPS 2023 | 合成数据，含多种时间异步 | V2V | 异步鲁棒检测 | — | [arXiv](https://arxiv.org/pdf/2309.16940)（检索摘要） |
| OPV2V-H / RCooper | ICLR 2024 / — | 异构 agent 基准 / 真实路侧 | — | 异构检测 | — | [HEAL](https://arxiv.org/html/2401.13964v3)；[GT-Space](https://arxiv.org/html/2603.19308v1) |
| CoPeD | RA-L 2024 | 3 个地面机器人（Warthog/Jackal）+ 2 个空中机器人，室内外 | 异构多机器人 | 多机器人协同感知 | 数据在 github.com/arplaboratory/CoPeD | [arXiv](https://arxiv.org/abs/2405.14731)（检索摘要） |

- OpenCOOD 内置位姿噪声（`loc_err`，`xyz_std`/`rpy_std`=0.2）和异步（`async_mode`='real'，`async_overhead` 200 或 300）配置，示例为 4 GPU 训练，未写最低硬件 — [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- 已知的单卡资源量级：CMP 推理约 67.3 ms（RTX 6000 Ada） — [CMP](https://arxiv.org/html/2403.17916v3)；HeteroPROMPT 训练峰值显存 3.8 GB，HEAL 为 7.0 GB（A6000，batch 1） — [HeteroPROMPT](https://arxiv.org/html/2607.26283)
- 2025 ICCV Workshop 的 V2X 端到端挑战赛基于 UniV2X 框架和 V2X-Seq-SPD，有 30 多支队伍参加，分协同时序感知和协同端到端规划两个赛道 — [arXiv](https://arxiv.org/abs/2507.21610)
- Griffin（空地协同检测跟踪）指出 CoPeD 面向低空场景且标注是自动生成的，认为这是局限 — [Griffin](https://arxiv.org/html/2503.06983v2)（检索摘要）

### Inferences
- **首个实验（单 GPU）建议**：在 V2X-Seq-TFD 或 V2X-Traj 上做**目标/轨迹级**的去中心化潜变量预报。输入是轨迹，不需要 LiDAR 主干网络，与 V2X-Graph 的逐档时延表（0/100/200 ms）和丢包表（0/10/30/50%）可以一一对应，Co-MTP 也给出了 0.2 m 噪声和 0.5 s 时延的协议。V2X-Traj 有 3 个智能体（2 AV + 1 路侧），是能体现"多于两方共识"的最小真实设定。
- **第二步**：V2XPnP-Seq（4 个智能体，含 I2I/V2V）用于端到端感知+预测，但需要 LiDAR 主干网络且 v2.0 发布状态不确定；单卡可行性取决于是否复用 OpenCOOD 的预计算 BEV 特征（这一点未验证）。
- **许可风险**：V2X-Real 页脚为 "All Rights Reserved"，V2X-Seq 只写了下载地域。在公开仓库中**不能提交任何数据或派生文件**（与用户既有的"公开仓库不放原始数据"规则一致）。

### Gaps
- V2X-Seq、DAIR-V2X、V2XPnP-Seq、OPV2V、TUMTraf 的**数据许可条款原文**都没有读到（GitHub/项目页未写，可能在注册/下载页内）。
- 各数据集的下载体积（GB）均未找到。
- 没有找到"多机器人 + 序列 + 预测"的专门基准；CoPeD 是感知数据集，是否有可用于预报的时序标注未确认。

## 问题 4：近期综述和论文把哪些问题列为开放问题（位姿误差、时延、带宽、异构、可扩展性、智能体间一致性、去中心化）？

### Takeaway
各综述一致把**位姿误差、异步/时延、带宽、异构**列为四大障碍；2026 年的 MAEAD 综述进一步把**跨智能体对齐、可验证的共享状态维护、边缘端持久 SWM、闭环多智能体规划**列为开放问题。**去中心化融合在综述中只是零散提及，没有被系统研究**；"共识"一词主要出现在安全领域（剔除恶意智能体），而不是状态估计。

### Cited Findings
- MAEAD（2026）原句：位姿 "fusion is sensitive to time synchronization and pose estimation errors"；异步 "residual asynchrony can still degrade performance in fast-changing scenes"；带宽 "raw point clouds and images impose heavy spectrum bandwidth and latency demands"；规模 "raw or dense feature exchange can exceed practical communication budgets in crowded scenes"；一致性 "must be updated and harmonized across the fleet"；去中心化 "persistent edge-hosted SWM maintenance and closed-loop multi-agent planning remain open system problems" — [MAEAD html](https://arxiv.org/html/2606.13840)
- MAEAD 还指出 "Foundation-model-based coordination also lacks verifiable real-time safety guarantees in open traffic" — [MAEAD abs](https://arxiv.org/abs/2606.13840)
- Huang 等《V2X Cooperative Perception for Autonomous Driving: Recent Advances and Challenges》（arXiv 2310.03525 v4，2024-11，已投 IEEE）：位姿 "Accurate relative transformation matrices are foundational"，且部分方法 "rely on ground-truth poses during the training phase"；带宽 "Excessive network load increases the risk of important data packets being delayed or lost"；一致性 "Variability in feature quality may lead to perception inconsistencies across agents"；对协同预测只做了简短讨论（接收邻车意图或期望轨迹）；去中心化只零散提及（DSRC 的去中心化设计、Where2comm 的稀疏通信图） — [arXiv](https://arxiv.org/html/2310.03525v4/)
- V2X 端到端挑战赛（ICCVW 2025）把带宽感知融合、鲁棒多智能体规划、异构传感器集成列为挑战 — [arXiv](https://arxiv.org/abs/2507.21610)
- HEAL：支持新 agent 类型的代价可能随类型数线性甚至平方增长，从而限制可扩展性 — [HEAL](https://arxiv.org/pdf/2401.13964)（检索摘要）
- 2026 年的 "General Collaborative Intelligence" 论文认为，大多数方法假设智能体集合固定，并把**agent churn（智能体加入/退出）**列为开放问题；还指出编码、传输、融合的端到端时延在紧急制动等场景下可能超过决策预算，使协同反而不如单车 — [arXiv](https://arxiv.org/pdf/2609.22967)（检索摘要）
- 2026 年的一篇瑞典学位论文把模型异构、时延、有限带宽列为部署的主要障碍 — [DiVA](https://www.diva-portal.org/smash/get/diva2:2052321/FULLTEXT01.pdf)（检索摘要）
- HeteroPROMPT 把对时延、丢包、定位误差的鲁棒性列为未来工作 — [arXiv](https://arxiv.org/html/2607.26283)
- 安全领域的"共识"：MADE 让每个 agent 用 match loss 和重建损失检验与协作者的一致性，从而剔除恶意 agent；ROBOSAC 随机采样协作者子集来验证共识，但需要先验的恶意概率 — [MADE](https://arxiv.org/abs/2310.11901)（检索摘要）

### Inferences
- 把这些开放问题和拟议方法对应起来：(1) 一致性/对齐 → GBP 收敛后各 agent 的边缘一致；(2) agent churn → Robot Web 式的局部因子图，可以随时加入或退出（见问题 5）；(3) 时延/丢包 → 消息精度随消息年龄衰减，加上异步 GBP；(4) 带宽 → 固定尺寸潜变量（均值+精度）；(5) 异构 → 共享潜空间（与 HEAL/GT-Space 思路一致，但要用在预测上）。这五项都有文献明确称为"开放"，可以直接引用。
- 需要提防的反方论点：GBP 迭代次数会带来额外时延（"General Collaborative Intelligence" 的时延预算论点），所以必须报告"通信轮数 × 每轮字节"的总开销，以及在固定墙钟时间下的性能。

### Gaps
- 没有找到 2025 年的专门协同感知综述全文（只看到 2026 年 IEEE T-ITS 车辆协同感知系统综述被引用）。
- MAEAD 第 8.3 节（数据集与基准）和关于 SWM 一致性的详细段落在抓取范围之外，没有读到。

## 问题 5：多机器人（非驾驶）场景下有哪些共享世界状态 / 分布式估计方法（Robot Web、分布式 GBP 规划、分布式神经场）？有没有使用学习到的潜状态？

### Takeaway
基于 GBP 的多机器人分布式估计和规划已经相当成熟（Robot Web：最多约 1000 台机器人，丢包 50% 仍稳定；GBPplanner；探索与共识；DistGP），但**状态全部是手工定义的几何量**（位姿、轨迹点、GP 参数）。分布式神经场（DiNNO、Di-NeRF、RAMEN）共享的是**网络权重**，用 ADMM 做共识，RAMEN 已经采用"消息越旧不确定性越大"的加权共识。**没有找到"学习得到的潜状态 + GBP 共识"用于多机器人预测的工作**。最接近的是 GBP Learning（ICML 2024，把参数作为 GBP 变量）和 DistGP（ICRA 2026，用 GBP 训练分布式 GP 地图）。

### Cited Findings
- **A Robot Web for Distributed Many-Device Localisation**（Murai, Ortiz, Saeedi, Kelly, Davison；IEEE T-RO vol. 40，2024；arXiv 2202.03314） — [Spiral](https://spiral.imperial.ac.uk/entities/publication/ac087e1d-6aff-4dc0-b17d-908540e034c1)；[arXiv](https://arxiv.org/abs/2202.03314v2)
  - 每个机器人只存自己的位姿变量、里程计/先验因子，以及它对其他机器人的距离-方位观测因子；消息是高斯边缘（N×1 信息向量 + N×N 精度矩阵，李群变量另带点估计）；**异步**，任意调度下都收敛；机器人不需要知道总数；假设全局时钟同步 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03314)
  - 鲁棒性：随机丢包至少到 50% 时 ATE 仍低；使用 DCS 鲁棒因子（Φ=10）时，即使 70–80% 的观测被破坏 ATE 仍低；初始化噪声 0.5 m/0.5 rad 时误差"爆炸" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03314)
  - 精度（ATE RMSE，GTSAM vs GBP）：N=128、30 m 量程时 0.006 vs 0.006；N=64、10 m 量程时 0.154 vs 0.290；GBP 收敛到精确的边缘均值，但**方差可能过度自信**；规模实验最多 1024 台；实物实验为 9 台 TurtleBot3（Raspberry Pi 3B+）全机载运行 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03314)
  - 后续：用 GBP 联合估计位姿和外参标定（arXiv 2401.15036） — [arXiv](https://arxiv.org/abs/2401.15036v1)；Murai 2025 年博士论文《Distributed Spatial AI》 — [PDF](https://www.doc.ic.ac.uk/~ajd/Publications/Riku-M-2025-PhD-Thesis.pdf)
- **Distributing Collaborative Multi-Robot Planning with Gaussian Belief Propagation**（Patwardhan, Murai, Davison；RA-L 2023 / ICRA 2023）：把多机器人规划写成机器人状态上的因子图，每台机器人存一个片段，用 GBP 求解；即使通信失败，轨迹也比其他分布式规划更短、更快、更平滑；GBPplanner 可免费用于非商业用途 — [arXiv](https://arxiv.org/abs/2203.11618v3)；[项目页](https://www.imperial.ac.uk/dyson-robotics-lab/projects/gbpplanner/)（检索摘要）
- **A Distributed Multi-Robot Framework for Exploration, Information Acquisition and Consensus**（Patwardhan 等，arXiv 2310.01930）：把 GBP 用于团队探索和建图共识 — [arXiv](https://arxiv.org/abs/2310.01930v1)（检索摘要）
- 2025–2026 年的 GBP 扩展：DD-GBP（多臂、异步消息、局部树形因子图，arXiv 2610.20460） — [arXiv](https://arxiv.org/abs/2610.20460)；结合全局路径搜索的 GBP 多智能体路径规划（arXiv 2502.20369） — [arXiv](https://arxiv.org/pdf/2502.20369)；巡逻机器人群用 GBP 对环境信号做集体排序（arXiv 2608.17690） — [arXiv](https://arxiv.org/pdf/2608.17690)；学习得到的双臂解耦任务用 GBP 协调（IROS 2025） — [UC3M](https://e-archivo.uc3m.es/entities/publication/4f8c47a3-b50c-46fb-9070-83e7aca670af)；机器人群仅凭局部观测和消息建立共享空间框架（Autonomous Robots 2025） — [Springer](https://link.springer.com/article/10.1007/s10514-025-10228-1)（以上均为检索摘要）
- **Learning in Deep Factor Graphs with Gaussian Belief Propagation**（Nabarro, van der Wilk, Davison；ICML 2024，PMLR 235:37141–37163）：因子图结构仿照神经网络，**参数作为变量**，训练和预测都用 GBP；由于 BP 是局部的，训练可以分布式、异步进行；可以把上一个任务的后验作为下一个任务的先验来做持续学习 — [PMLR](https://proceedings.mlr.press/v235/nabarro24a.html)；[arXiv](https://arxiv.org/abs/2311.14649)
- **DistGP: A Distributed Gaussian Process Model for Multi-Robot Mapping**（Nabarro, van der Wilk, Davison；arXiv 2603.07351；ICRA 2026）：稀疏 GP 的分解方式与多机器人结构一致，用 GBP 在 loopy 模型上做分布式异步训练；能达到集中式批训练的性能，但收敛更慢；相对 DiNNO 精度更高、对稀疏通信更稳健，并支持动态连通 — [arXiv](https://arxiv.org/abs/2603.07351)
- **DiNNO: Distributed Neural Network Optimization for Multi-Robot Collaborative Learning**（Yu, Vincent, Schwager；RA-L 7(2) 2022）：基于 ADMM，每台机器人持有一份网络副本，只交换权重；在 2D 密度场神经隐式建图中，每台机器人都能重建全图；收敛保证只对凸目标成立 — [arXiv](https://arxiv.org/abs/2109.08665v1)（检索摘要）
- **RAMEN: Real-time Asynchronous Multi-agent Neural Implicit Mapping**（Zhao, Ivanovic, Mehr；arXiv 2502.19592，预印本）：在网络参数层面共识；断连时保留邻居的过时副本，其**不确定性随时间增长**；用梯度信息估计逐参数不确定性，做加权 C-ADMM；作者指出 DiNNO 和 Di-NeRF 使用的 C-ADMM 对异步通信不稳健 — [arXiv](https://arxiv.org/abs/2502.19592)
- CoDreamer（见问题 2）是本组中唯一在多智能体之间传递**学习得到的潜状态**的方法，但用的是 GNN，而不是概率共识 — [arXiv](https://arxiv.org/html/2406.13600v1)

### Inferences
- **空白定位（核心结论）**：
  1. V2X 预测：有学习的潜变量，但融合在 ego 端，几乎没有不确定性，也没有共识；
  2. 多机器人 GBP：有共识、异步、丢包鲁棒和 agent 加入/退出，但状态是手工定义的几何量；
  3. MARL 世界模型：有去中心化的学习潜变量，但没有概率消息、没有一致性保证，也没有丢包实验；
  4. 分布式神经场：有不确定性加权共识，但共享的是权重，不是逐时刻的状态信念。

  **"学习得到的固定尺寸潜变量信念（均值+精度）+ GBP 迭代共识 + 解码为未来轨迹/占据"正好落在四者的交集之外**，目前没有直接竞争者。
- 可直接借用的设计依据：Robot Web 的消息格式（信息向量 + 精度）、丢包协议（0–50%）、鲁棒因子（DCS）和"方差过度自信"的已知缺陷；RAMEN 的"陈旧度 → 不确定性增长"；GBP Learning 中"参数也作为变量、可以分布式异步训练"的思路，可以用来论证编码器/解码器与 GBP 的联合训练是可行的。
- 必须正面回应的风险：Robot Web 已报告 GBP 的方差可能过度自信，所以"显式不确定性"需要单独做校准评测（例如 NLL、覆盖率），不能只报 ADE/FDE。

### Gaps
- MACIM 和 Di-NeRF 的原文没有读到，无法给出它们的方法细节和数字。
- RAMEN 和 DistGP 只读了 abstract，没有拿到定量的鲁棒性数字和实验平台。
- 没有找到"多机器人共享世界模型做未来预测"（非 MARL 玩具环境）的工作；COMBO 等具身多智能体世界模型这次没有检索，属于覆盖缺口。
