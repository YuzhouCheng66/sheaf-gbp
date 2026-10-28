# 学习潜在状态上的去中心化推断与通信：最接近的先例（Prior Art）调研笔记

> 调研日期：2026-10-28。被检验的设计（下称"本设计"）：每个智能体 i 的编码器输出固定维度潜向量 z_i 与对角精度 Λ_i；各智能体的潜信念 x_i 通过在二次 sheaf 能量
> E(x) = Σ_i ½(x_i − z_i)ᵀΛ_i(x_i − z_i) + Σ_(i,j) ½ w‖F_ij x_i − F_ji x_j‖²
> 上做精确高斯置信传播（GBP）融合（限制映射 F、固定大小消息、树上精确、异步），可能再沿时间经学习的潜动力学模型传播，最后由解码器/策略读取融合后的潜变量。
> 说明：每条"Cited Findings"仅记录本次会话中实际读到的内容（摘要页、检索摘要或全文片段）；凡来自检索摘要而非原文的，已标注"检索摘要"。"Inferences"为基于引文的推理，未经文献直接证实。

## 问题 1：MARL 中的学习式通信（CommNet、TarMAC、IC3Net、MAGIC、带宽受限/信息瓶颈通信）对固定大小潜消息、可扩展性与消息丢失鲁棒性说明了什么？

### Takeaway
MARL 学习式通信早已确立"固定维度连续消息 + 端到端反传 + 聚合（均值/注意力/GAT）"范式，并有方差/KL 约束控制带宽（VBC、变分消息编码）与训练期丢消息（Message-Dropout）、执行期消融集成（AME）等鲁棒性手段；但这些方法的融合算子都是学习的神经聚合，而非具有概率语义的精度加权推断，也没有"树上精确"或"消息丢失下仍收敛到同一后验"的保证——这正是 GBP 类方法（问题 2）能补上的空白。

### Cited Findings
- **CommNet**：Sukhbaatar, Szlam, Fergus, "Learning Multiagent Communication with Backpropagation", NIPS 2016（pp. 2244–2252），arXiv 1605.07736；面向完全合作任务，连续通信通道与各智能体策略一起端到端反传训练。— [NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html)；[arXiv](https://www.arxiv.org/pdf/1605.07736)
- **IC3Net**：arXiv 1812.09755，arXiv 标题为 "Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks"（2018 年 12 月提交）；提出 Individualized Controlled Continuous Communication Model，用门控机制控制是否通信，并使用个体化奖励，从而扩展到混合/竞争场景；作者指出 CommNet 仅限合作场景。— [arXiv](https://arxiv.org/pdf/1812.09755)
- **TarMAC**："TarMAC: Targeted Multi-Agent Communication"，arXiv 1810.11187；使用软的发送者-接收者注意力，智能体学会"何时通信、发给谁"；与 IC3Net 结合可用于混合/竞争环境。— [arXiv](https://arxiv.org/pdf/1810.11187)。注：检索摘要称其为 "ICML 2018"，本人先验记忆为 ICML 2019，**会议年份存在冲突，未核实**。
- 2022 年的 "Universally Expressive Communication in Multi-Agent Reinforcement Learning"（arXiv 2206.06758）把基于 IC3Net 奖励/通信结构的 TarMAC 称为 T-IC3Net，并指出 TarMAC 的软注意力隐式地产生完全通信图，可由 GAT 建模。— [arXiv](https://arxiv.org/pdf/2206.06758)
- **MAGIC**（Niu, Paleja, Gombolay, AAMAS 2021）：Scheduler 决定何时/向谁通信，Message Processor 基于动态图上的图注意力网络（GAT）；通信图是学习得到的；作者自报通信效率平均比基线高 27.4%。— [AAMAS 2021 PDF](https://archive.illc.uva.nl/AAMAS-2021/pdfs/p964.pdf)；扩展版见 [Georgia Tech repository: "Adaptable and Scalable Multi-Agent Graph-Attention Communication"](https://repository.gatech.edu/entities/publication/5df09d4a-615b-431f-b276-537e94e70c1a)
- **VBC**：Zhang, Zhang, Lin, "Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control", NeurIPS 2019，arXiv 1909.02682；训练时限制消息方差以过滤噪声、保留有用内容；在多个 MARL 基准上通信开销比 SOTA 低 2–10×，同时总体性能更好。— [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html)；[arXiv](https://arxiv.org/abs/1909.02682v2)
- **带宽受限变分消息编码**："Bandwidth-constrained Variational Message Encoding for Cooperative Multi-agent Reinforcement Learning"，arXiv 2512.11179；按信息瓶颈原则，用 KL 散度把每个智能体的消息分布拉向共享先验，先验方差直接控制每个消息维度的信息容量；在带宽比 0.05 下于 SMACv1、SMACv2、MPE Tag 上评测（检索摘要）。— [arXiv](https://arxiv.org/pdf/2512.11179)
- **Message-Dropout**（Kim, Cho, Sung, AAAI 2019，arXiv 1902.06527）：训练期以一定概率按块丢弃收到的他人消息，并用校正概率乘以被丢块的权重；报告训练速度和执行期稳态性能显著提升；不是对抗防御。— [arXiv](https://arxiv.org/abs/1902.06527v1)
- **AME**（Sun et al., "Certifiably Robust Policy Learning against Adversarial Multi-Agent Communication", ICLR 2023，arXiv 2206.10158）：在多个"消融消息子集"上得到多个基动作并集成（离散动作多数投票、连续动作取中位数），对最多 C < (N−1)/2 条被任意扰动的通信给出可证鲁棒性。— [arXiv](https://arxiv.org/abs/2206.10158v2)；[ICLR](https://iclr.cc/virtual/2023/poster/11377)
- **综述**：Zhu, Dastani, Wang, "A survey of multi-agent deep reinforcement learning with communication", JAAMAS（2024-01-06 在线发表），亦见 AAMAS 2024；提出 9 个分析维度（含通信约束），鲁棒性只顺带涉及。— [Springer](https://link.springer.com/article/10.1007/s10458-023-09633-6)；[AAMAS 2024](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p2845.pdf)
- 综述 "Robust and efficient communication in multi-agent reinforcement learning"（Chaos 第 36 卷第 2 期）指出大多数方法假设通信是即时、可靠、带宽无限的；系统回顾消息扰动、传输延迟、有限带宽，并以每条消息丢失概率 P_loss 建模消息丢失（检索摘要；出版年未核实）。— [AIP](https://pubs.aip.org/aip/cha/article/36/2/022101/3380329/Robust-and-efficient-communication-in-multi-agent)；[ResearchGate](https://www.researchgate.net/publication/397663741_Robust_and_Efficient_Communication_in_Multi-Agent_Reinforcement_Learning)
- 2025–2026 鲁棒通信相关工作（仅核实到标题）："Robust Multi-agent Communication Based on Decentralization-Oriented Adversarial Training"（arXiv 2504.21278）；"Decoupling Communication from Policy: Robust MARL under Bandwidth Constraints"（arXiv 2605.21085）；"Value-Aware Prediction for Robust Multi-Agent Coordination Under Communication Loss"（arXiv 2607.17914）；2026 综述 "A Survey of Multi-Agent Deep Reinforcement Learning with Graph Neural Network-Based Communication"（arXiv 2604.25972）。— [2504.21278](https://arxiv.org/pdf/2504.21278)；[2605.21085](https://arxiv.org/pdf/2605.21085)；[2607.17914](https://arxiv.org/pdf/2607.17914)；[2604.25972](https://arxiv.org/pdf/2604.25972)
- **多智能体世界模型中的通信**：Zeng & Zhang, "Efficient Information Sharing for Training Decentralized Multi-Agent World Models", RLC 2025 / Reinforcement Learning Journal vol. 6, pp. 909–922；指出独立训练世界模型常导致策略欠佳，而已有工作多依赖需过量通信的集中训练；先给无限带宽找出哪些组件受益于哪种通信，再限制到预定带宽；在 SMAC 上评测（检索摘要，PDF 抓取失败）。— [RLJ](https://rlj.cs.umass.edu/2025/papers/Paper103.html)
- CALL（arXiv 2506.08149，2025 年 6 月）：自我中心 MARL 中智能体通过轻量通信共享世界模型的低维潜表示（检索摘要）。— [arXiv](https://arxiv.org/pdf/2506.08149)
- "Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models"（TMLR 2025，arXiv 2406.15836）：去中心化局部动力学 + 集中式表示聚合，SMAC、MAMuJoCo（检索摘要）。— [arXiv](https://arxiv.org/abs/2406.15836v2)
- "Generative Emergent Communication: Large Language Model is a Collective World Model"（arXiv 2501.00226；期刊版 Advanced Robotics 2026）：把共享潜变量（消息 m）纳入每个智能体的世界模型（检索摘要）。— [arXiv](https://arxiv.org/pdf/2501.00226)；[T&F](https://www.tandfonline.com/doi/full/10.1080/01691864.2026.2661958)
- "Dreaming of Others: Latent Teammate Modeling in World Models for MARL"（arXiv 2605.31361）：把 Dreamer 式潜状态分解为环境与队友成分并加 ToM 头（检索摘要）。— [arXiv](https://arxiv.org/html/2605.31361)
- **协同感知（中间特征融合）**：Where2comm（arXiv 2209.12836）以空间置信图决定传输哪些特征位置，作者报告在 OPV2V 上通信量降低超过 100,000× 且优于 DiscoNet、V2X-ViT；并称 DiscoNet 使用基于 MLP 的融合权重学习。— [arXiv](https://arxiv.org/pdf/2209.12836)。2025 年 UAV 协同感知论文（arXiv 2504.21774）批评 Where2comm 仅以 objectness 为准则、忽略模型预测的不确定性，转而用一致性构造逐位置不确定性图。— [arXiv](https://arxiv.org/pdf/2504.21774)；中间融合综述 [arXiv 2404.16139](https://arxiv.org/html/2404.16139v2)

### Inferences
- 消息大小与 N 无关的性质在 MARL 中通常通过对邻居消息做均值/注意力加权和得到（CommNet/TarMAC/MAGIC 的共性，细节来自先验知识，未在本次逐篇核实）；本设计的 GBP 消息同样与 N 无关，但其"聚合"是信息形式（η, Λ）的**加法**，具有明确的概率语义——这一点是与 MARL 通信最本质的区别，可作为新颖性陈述的核心。
- VBC 与变分消息编码已经把"消息方差/先验方差"当作带宽与可靠性的控制量，但方差不进入接收端的贝叶斯融合；本设计把编码器方差（Λ_i）直接用作融合权重，是把这两条线（带宽 IB 与精度加权）统一的一种自然方式，但需要说明 Λ_i 的校准如何训练。
- 鲁棒性：MARL 中丢消息鲁棒性靠训练期随机化（Message-Dropout）或执行期集成（AME）获得，属于经验/投票型；GBP 在丢包时只是收敛变慢、固定点不变（见问题 2 的 Robot Web / DD-GBP 证据），这是本设计可以主张的结构性优势，但需实验验证"学习到的 Λ_i 下仍成立"。
- 协同感知（Where2comm/DiscoNet）与本设计在"多智能体中间潜特征融合 + 置信度/不确定性"上最接近，但融合是集中式/单跳注意力，且置信图不是精度。

### Gaps
- MAGIC 的 AAMAS 2021 正式标题与 TarMAC 的会议年份未在原文核实。
- 未找到在 MARL 通信中显式使用 GBP/卡尔曼式精度加权融合学习消息并做多跳推断的论文（检索有限，不排除存在）。
- RLC 2025 论文（Zeng & Zhang）的消息维度、融合算子与带宽结果数字未读到（PDF 证书错误）。
- 未读到关于 CommNet/TarMAC 在消息丢失下性能退化的定量数据。

## 问题 2：多机器人与学习中的高斯置信传播（Robot Web、分布式 GBP 规划、深度因子图中的 GBP 学习、GPU/异步 GBP，以及 2024–2026 年 GBP 与学习/潜变量结合的工作）

### Takeaway
Davison 组已系统证明 GBP 适合异步、去中心化、对丢包鲁棒的多机推断（Robot Web 1000 机器人仿真接近集中式精度；DD-GBP 在 90% 丢包下仍有 86.0% 成功率），并已把 GBP 用于学习（Nabarro et al. ICML 2024 的"GBP Learning"）与端到端可微的神经-GBP 混合（2026 年 GBPN 深度补全）；但现有多机 GBP 的变量都是物理状态（位姿、轨迹、地图/GP），没有一篇在**学习的潜空间**上以编码器输出精度作一元因子、以学习的 sheaf 限制映射作智能体间因子来做 GBP。

### Cited Findings
- **Robot Web**：Murai, Ortiz, Saeedi, Kelly, Davison, "A Robot Web for Distributed Many-Device Localisation", IEEE T-RO vol. 40, 2024, pp. 121–138；arXiv 2202.03314（2022 年 2 月）。在描述所有机器人内部及相互观测的**非线性因子图**上运行 GBP；定义可运行于发布/读取网页等**异步通道**上的轻量协议；鲁棒因子使其能容忍大量错误测量与**丢包**；最多 1000 台机器人的仿真中全局精度接近集中式非线性因子图求解器；并在算力有限的真实机器人上运行。— [arXiv](https://arxiv.org/abs/2202.03314v2)；[Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/ac087e1d-6aff-4dc0-b17d-908540e034c1)
- **GBP Planning**：Patwardhan, Murai, Davison, "Distributing Collaborative Multi-Robot Planning with Gaussian Belief Propagation", arXiv 2203.11618，IEEE RA-L 2023；纯分布式多机规划。— [arXiv](https://arxiv.org/abs/2203.11618)
- Murai et al., "Distributed Simultaneous Localisation and Auto-Calibration using Gaussian Belief Propagation", arXiv 2401.15036（2024 年 1 月）：可扩展、全分布、在线的多机定位与外参标定。— [arXiv](https://arxiv.org/abs/2401.15036)；Murai 2025 年博士论文。— [PDF](https://www.doc.ic.ac.uk/~ajd/Publications/Riku-M-2025-PhD-Thesis.pdf)
- **GBP Learning**：Nabarro, van der Wilk, Davison, "Learning in Deep Factor Graphs with Gaussian Belief Propagation", ICML 2024（PMLR 235, pp. 37141–37163），arXiv 2311.14649；把网络结构写成因子图、可学习参数作为图中变量，将因子近似为高斯，用 GBP 同时做训练（给定观测对参数推断）与预测；可扩展到深层网络，并自然支持持续学习（上一任务的 BP 后验作下一任务先验）；视频去噪上可学习参数优于经典因子图，MNIST 单 epoch 训练做持续图像分类。— [PMLR](https://proceedings.mlr.press/v235/nabarro24a.html)；[arXiv](https://www.arxiv.org/abs/2311.14649)
- **教程/硬件**：Ortiz, Evans, Davison, "A visual introduction to Gaussian Belief Propagation", arXiv 2107.02308（2021），将 GBP 定位为面向新兴硬件的分布式推断框架。— [arXiv](https://arxiv.org/abs/2107.02308v1)。Ortiz, Pupilli, Leutenegger, Davison, "Bundle Adjustment on a Graph Processor", CVPR 2020：在 Graphcore IPU 上 125 关键帧、1919 点的 BA 在 40 ms 内求解，对比 Ceres CPU 1450 ms；GBP 可处理增量 SLAM 与鲁棒代价。— [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Ortiz_Bundle_Adjustment_on_a_Graph_Processor_CVPR_2020_paper.html)；[arXiv](https://arxiv.org/abs/2003.03134v2)。Ortiz 博士论文 "Gaussian belief propagation for real-time decentralised inference" 主张 GBP 作为异步、增量估计框架。— [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/92cca8ac-2ffc-4e9a-b88e-e5e7db83aa24)
- **DD-GBP**（Jin, Dai, Yin, Wang, Chen, Hu, arXiv 2610.20460，2026-10-14 提交）："Each robot maintains a local tree-structured factor graph"，并 "exchange messages asynchronously over active inter-arm links"；动态子网重构与事件触发更新；30 臂无障碍测试成功率 99.87%、平均热启动求解 15.80 ms（>60 Hz）；**90% 丢包下成功率仍为 86.0%**；50 个四臂动障碍场景无碰撞全目标完成率 92.5%（六个基线中最高）；页面提到"局部推断精度的理论分析"但未说明树/环精确性。— [arXiv](https://arxiv.org/abs/2610.20460)
- **DistGP**："A Distributed Gaussian Process Model for Multi-Robot Mapping"（arXiv 2603.07351，2026 年 3 月）：稀疏 GP 的分解镜像多机结构，可经 GBP 分布式训练；其**有环（loopy）模型优于树结构 GP**，可在线训练并适应动态连通性（检索摘要）。— [arXiv](https://arxiv.org/html/2603.07351)
- 其他 GBP 多机工作（仅核实标题）："Multi-Agent Path Planning in Complex Environments using Gaussian Belief Propagation with Global Path Finding"（arXiv 2502.20369）；"Collective Ranking of Environmental Signals through Gaussian Belief Propagation in a Patrolling Robot Swarm"（arXiv 2608.17690）。— [2502.20369](https://arxiv.org/pdf/2502.20369)；[2608.17690](https://arxiv.org/pdf/2608.17690)
- **神经网络 + GBP（2025–2026）**："Gaussian Belief Propagation Network for Depth Completion"（arXiv 2601.21291）：神经网络构建深度变量上的 MRF、GBP 推断稠密深度；网络还预测图结构（含非局部边），并**通过 GBP 输出端到端训练**（检索摘要）。— [arXiv](https://arxiv.org/html/2601.21291v1)。"A Spiking Neural Network Implementation of Gaussian Belief Propagation"（arXiv 2512.10638）：将 Forney 因子图映射到 LIF 神经元群体。— [arXiv](https://arxiv.org/html/2512.10638v1)。"Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional Systems"（arXiv 2402.08193）。— [arXiv](https://arxiv.org/abs/2402.08193)
- **有环 GBP 的理论**：Weiss & Freeman, "Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology"（NIPS 1999）：若收敛，则对任意拓扑给出正确的后验**均值**，并给出真实边缘与 loopy BP 输出之间的精确关系式。— [NeurIPS](https://proceedings.neurips.cc/paper/1999/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)；Malioutov, Johnson, Willsky（JMLR 2006）提出 walk-summable 高斯图模型，给出更强的 loopy GBP 收敛结果。— [JMLR PDF](https://jmlr2020.csail.mit.edu/papers/volume7/malioutov06a/malioutov06a.pdf)；Giscard 的 path-sum 工作指出 LBP 均值正确但协方差估计一般不正确。— [ar5iv 1410.7165](https://ar5iv.arxiv.org/html/1410.7165)
- "Consensus Propagation"（arXiv cs/0603078）：共识传播是 BP、更具体说是高斯 BP 的特例（检索摘要）。— [arXiv](https://arxiv.org/pdf/cs/0603078)

### Inferences
- 本设计的能量可写为 E(x) = ½xᵀ(Λ + wL_F)x − xᵀΛz + const，其中 L_F 为 sheaf Laplacian（L_F = δᵀδ），Λ = blockdiag(Λ_i)。只要所有 Λ_i ≻ 0，J = Λ + wL_F 正定，MAP 为 x* = J⁻¹Λz，后验协方差 J⁻¹。这是一个标准高斯 MRF，故 Weiss–Freeman / walk-summability 结论直接适用：树上 GBP 精确（均值与协方差），有环图上若收敛则均值精确、方差一般过度自信。
- F_ij = I 时 L_F 为（Kronecker 扩张的）图 Laplacian，J 的非对角元非正，属"吸引型"模型；按我对 Malioutov et al. 的记忆，正定的吸引型模型是 walk-summable 的，loopy GBP 会收敛（**需对原文核实**）。一般学习得到的 F_ij 含符号混合项，walk-summability 不再自动成立——这是学习限制映射时需要正则化/约束（如正交性）或用阻尼的理论理由。
- 即使 Λ_i 为对角，F_ij 混合维度时，精确 GBP 消息的信息矩阵一般是稠密的 d×d（消息大小 O(d²)，与 N 无关但与潜维度平方成正比）；若只传对角精度，则是近似（类似 RKN 的因子化假设，见问题 4）。这需在"固定大小消息"的表述中写清楚。
- Robot Web / DD-GBP 的丢包鲁棒性证据针对物理状态与非线性因子；把它迁移到学习潜空间是本设计需自己证明的部分。
- Nabarro et al. 与 GBPN 表明"通过 GBP 训练/反传"已可行，但都是单机、集中式；DistGP 是"GBP 分布式训练学习模型"的最近例子，但其变量是 GP 诱导点而非编码器潜变量。

### Gaps
- 未找到 2024–2026 年把 GBP 用于**多智能体学习潜状态**（encoder 输出均值+精度作一元因子）融合的论文；检索覆盖有限，但多组关键词均未命中。
- 未找到"GPU 上异步 GBP"在 2024–2026 年的专门论文（仅有 2020 年 IPU 工作与 Ortiz 论文）。
- DD-GBP 的"局部推断精度理论分析"具体内容未读到；DistGP 的定量结果未读到。
- 向量值（块）高斯模型的 walk-summability 条件与 sheaf Laplacian 的关系，未找到文献。

## 问题 3：Sheaf 方法（sheaf Laplacian、Hansen & Ghrist 意见动力学、sheaf 神经网络/神经 sheaf 扩散、2023–2026 基于 sheaf 的多智能体/分布式估计）——有没有在异构智能体潜空间之间学习限制映射的工作？

### Takeaway
有，且集中在 2025–2026 年：SFRL（2026-08）在异构（含不同潜维度）智能体之间学习正交/等距限制映射并用 sheaf Laplacian 二次粘合项对齐；Sheaf-ADMM（ICML 2026）在神经编码器参数化的局部凸子问题之间用 sheaf 约束 + 展开 ADMM 端到端训练；LatentToM（CoRL 2025 oral）学习"共识嵌入"并在推断时用 sheaf Laplacian 做"共识修复"；SIGMA（ICRA 2025）用 sheaf 一致性学习潜共识特征。但它们使用的都是 sheaf 扩散、ADMM 或训练期正则，**没有一个把编码器精度作一元项、在 sheaf 能量上做精确（树上）GBP 并读取后验**。Hansen–Ghrist 的 discourse sheaf 能量与 Bosca–Ghrist 的"方向性固执（directional stubbornness）"在数学上最接近本设计的二次能量。

### Cited Findings
- Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves"，arXiv 1808.01513，Journal of Applied and Computational Topology（DOI 10.1007/s41468-019-00038-7）：sheaf Laplacian 对常值 sheaf 退化为图 Laplacian / Hodge Laplacian。— [arXiv](https://arxiv.org/pdf/1808.01513)；[Springer](https://link.springer.com/article/10.1007/s41468-019-00038-7)
- Hansen & Ghrist, "Opinion Dynamics on Discourse Sheaves"，arXiv 2005.12798，SIAM J. Appl. Math.：每个智能体有私有意见空间，每条连接有话语（discourse）空间，线性映射描述私有观点如何公开表达；引入新的 Laplacian 与扩散动力学，研究可控性、可达性、有界信任、调和扩张；扩散收敛，并可在无外部影响时再现极化（检索摘要）。— [alphaxiv](https://alphaxiv.org/abs/2005.12798)；[ar5iv](https://ar5iv.arxiv.org/html/2005.12798)
- Bosca & Ghrist, "Selective Adaptation of Beliefs and Communication on Cellular Sheaves"（arXiv 2601.22431，2026）：扩展到**方向性固执**——智能体在意见的某些方向上固定、其他方向上灵活（检索摘要）。— [arXiv](https://arxiv.org/html/2601.22431v2)
- Hansen & Gebhart, "Sheaf Neural Networks"，arXiv 2012.06333（2020-12-08），NeurIPS 2020 "Topological Data Analysis and Beyond" workshop：用 sheaf Laplacian 推广 GCN 的扩散算子，在带符号图的合成半监督分类上显著优于 Kipf–Welling GCN。— [arXiv](https://arxiv.org/pdf/2012.06333)
- Bodnar, Di Giovanni, Chamberlain, Liò, Bronstein, "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"，NeurIPS 2022，arXiv 2202.04579：标准 GNN 隐式使用平凡 sheaf；非平凡 sheaf 让离散参数化扩散对渐近行为有更多控制；**从数据中学习 sheaf（限制映射）**，在异配基准上有竞争力。— [NeurIPS](https://papers.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html)；[arXiv](https://arxiv.org/abs/2202.04579v2)
- "Bayesian Sheaf Neural Networks"（arXiv 2410.09590）：把 sheaf Laplacian 当作潜随机变量，其分布由输入数据的函数参数化（检索摘要）。— [arXiv](https://arxiv.org/html/2410.09590)
- 综述 "Sheaf theory: from deep geometry to deep learning"（Ayzenberg, Magai 等，arXiv 2502.15476）：sheaf 扩散即对 sheaf Dirichlet 能量（二次函数）做梯度下降，核空间 H⁰(G,F) 为共识态，可为高维、非均匀的结构化一致状态（检索摘要）。— [arXiv](https://arxiv.org/html/2502.15476v1)
- Sergeant-Perthuis, "Sheaves in Machine Learning"（2026-01-15 讲义）：把广义置信传播（Generalized BP）推广到有限偏序集上的任意 presheaf（非高斯特化）（检索摘要）。— [PDF](https://www.normalesup.org/~sergeant/sheaf.pdf)
- **LatentToM**：He, Sznaier Camps, Liu, Schwager, Sartoretti, "Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation"，CoRL 2025（Oral），arXiv 2505.09144。每个机器人学习私有 ego 嵌入与跨机对齐的 consensus 嵌入；集中训练时用 sheaf 启发的"一阶上同调损失"对齐共识嵌入，并加 ToM 重建与方向性共识约束；推断时若有通信，每步交换**一个紧凑嵌入**，用 sheaf-Laplacian 更新拉近局部共识嵌入（"consensus repair"，轻量、模型无关、无需重训）；无通信时也能仅凭观测隐式协调；真实双臂任务中与集中式策略相当并对临时故障与延迟鲁棒（项目页未给数值与限制映射细节）。— [项目页](https://stanfordmsl.github.io/LatentToM/)；[arXiv](https://arxiv.org/abs/2505.09144)
- **SIGMA**：Liao, Xia, Cao, Dai, He, Wu, Sartoretti, "SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding"，ICRA 2025，arXiv 2502.06440：把 sheaf 理论引入去中心化深度 RL，以局部共识学习几何交叉依赖；神经网络在潜空间近似共识，自监督训练。— [arXiv](https://arxiv.org/abs/2502.06440)。检索摘要称其智能体同构，故限制映射相同并被共享（未在原文核实）。— [arXiv HTML](https://arxiv.org/html/2502.06440)
- **Sheaf-ADMM**：Seely, Cupiał, Jones, "Learning Multi-Agent Coordination via Sheaf-ADMM"，arXiv 2605.31005（2026-05-29），ICML 2026：输入划分为重叠局部视图，每个智能体求解由**神经编码器参数化的凸子问题**；智能体间约束由 cellular sheaf 指定（哪些方面须一致）；通过 ADMM 协调，**反传穿过展开的优化联合训练全部组件**；ADMM 暴露原始、共识、对偶变量便于分析干预；任务为迷宫寻路、MNIST、数独（数独上解出率"markedly higher"于参数匹配的 MPNN）。— [arXiv](https://arxiv.org/abs/2605.31005)。检索摘要另称其限制映射为"每个方向一个基础映射、全体智能体共享，加输入相关的低秩更新"（摘要页未提及，未核实）。
- **SFRL**：D'Acunto, Grimaldi, Avino, Pandolfo, Di Nino, Barbarossa, Di Lorenzo, "Sheaf-Based Federated Representation Learning"，arXiv 2608.10016（2026-08-08）：**不假设共享全局潜空间**，异构性包括模型、数据与**潜维度**；边上可学习的限制映射约束为正交变换与等距嵌入，以 Procrustes 闭式解更新，与局部模型梯度步交替（Sheaf-FRL）；对齐正则为 sheaf Laplacian 诱导的二次粘合惩罚，仅在少量共享 pilot 样本上计算，邻居交换的是 pilot 样本的潜表示；证明确定/随机情形下收敛到一阶驻点；在语义通信下的协作分类中优于基线，并对潜维度压缩更鲁棒（摘要未给数字）。— [arXiv](https://arxiv.org/abs/2608.10016)
- Hanks, Nino, Bou Barcelo, Copeland, Dixon, Fairbanks, "Heterogeneous Multi-Agent Multi-Target Tracking using Cellular Sheaves"，arXiv 2512.24886（2025-12-31）：状态维度不同的异构智能体，把未知目标跟踪写成 cellular sheaf 上的**调和扩张问题**，基于 sheaf Laplacian 的去中心化控制律 + Lyapunov 收敛分析；检索摘要称作者把非线性限制映射列为未来工作。— [arXiv](https://arxiv.org/abs/2512.24886)
- Anwer, Riess, Hale, "Multi-Agent System Identification with Nonlinear Sheaf Diffusion"，arXiv 2605.11204（2026-05-11）：由非线性 sheaf Laplacian 支配的多智能体系统，从轨迹恢复交互律；**恢复的障碍是拓扑性的、由 sheaf 上同调度量**：无约束函数类下当且仅当该上同调消失才可唯一恢复；实验表明"accurate trajectory reproduction need not certify recovery of the underlying interaction law"。— [arXiv](https://arxiv.org/abs/2605.11204)
- "Applied Sheaf Theory For Multi-agent Artificial Intelligence (Reinforcement Learning) Systems: A Prospectus"（arXiv 2504.17700）：建议从协调行为演示中学习线性限制映射（检索摘要）。— [arXiv](https://arxiv.org/pdf/2504.17700)
- Hernández & Sánchez-Soto, "The Sheaf Laplacian: A Topological Framework for Data Fusion and Consensus in Distributed Sensing Networks"，arXiv 2606.19529（2026-06-17）：主张 sheaf Laplacian 用于分布式传感网络的数据融合与共识；摘要未涉及卡尔曼/信息滤波、GBP、精度加权或学习限制映射。— [arXiv](https://arxiv.org/abs/2606.19529)
- 仅核实标题："A Sheaf Framework for Strategic Multi-Agent Systems: From Consensus to Nash Equilibria"（arXiv 2606.01663）。— [awesomepapers](https://awesomepapers.io/ai-agents/papers/2606.01663)

### Inferences
- 本设计的耦合项 Σ½w‖F_ij x_i − F_ji x_j‖² 正是 discourse sheaf / sheaf Dirichlet 能量；一元项 ½(x_i−z_i)ᵀΛ_i(x_i−z_i) 相当于"带固执的意见动力学"，且对角 Λ_i 对应 Bosca–Ghrist 的"方向性固执"——区别在于本设计的固执度由编码器逐样本预测（异方差），而非固定参数。能量的 MAP 解是 sheaf 上带锚定的调和扩张（w→∞ 时投影到全局截面 H⁰，w→0 时 x=z），与 Hanks et al. 的调和扩张表述同源。
- 方法学对照：NSD/SNN 用 sheaf Laplacian 做层间扩散（固定步数、无概率语义）；LatentToM 用 sheaf Laplacian 梯度步做推断期修复；Sheaf-ADMM 用 ADMM 求硬约束一致；SFRL 只在训练期正则。本设计在同一能量上做**精确 GBP**，可在树上一遍（两遍消息）得到精确后验均值和协方差，并天然异步、对丢包鲁棒——这是可主张的新组合，但单项都有先例。
- SFRL 的正交/等距约束与 StateBridge（问题 5）的闭式正交对齐是同一思想：限制正交性既能避免 F→0 的平凡解，又能使精度在映射下按协方差规则变换。本设计若学习 F，应考虑类似约束。
- Anwer–Riess–Hale 的结论提示：若 F_ij 与编码器一起学习，限制映射存在规范自由度（任一智能体潜空间的可逆重参数化可被 F 吸收），其可辨识性由上同调/数据信息矩阵决定；仅凭下游任务性能不能证明学到的 F 是"有意义的对齐"。

### Gaps
- 未找到在 sheaf 能量上做 GBP（而非扩散/ADMM）的论文；Sergeant-Perthuis 讲义是 presheaf 上的广义 BP，非高斯、非多智能体学习。
- LatentToM 的损失公式与是否学习限制映射、Sheaf-ADMM 限制映射的确切参数化、SFRL 如何处理不同潜维度（等距嵌入细节）均未读到全文。
- 未检索 Michael Robinson 的 sheaf 传感器融合工作（consistency radius 等），可能是"sheaf 作为传感器融合数据结构"的更早先例。

## 问题 4：潜空间滤波与融合（深度卡尔曼滤波、Kalman VAE、循环卡尔曼网络、PoE/精度加权融合、协方差交叉与共识滤波）——什么条件下对学习潜变量做加性/精度加权融合是正确的？如何训练？

### Takeaway
"编码器输出潜观测 + 方差，再做高斯条件化/乘积"在单智能体/集中式设定中已很成熟：RKN（ICML 2019）用高维因子化潜状态使卡尔曼更新退化为标量运算并端到端训练；Bayesian Context Aggregation（ICLR 2021）把每个上下文编码视为潜高斯的噪声观测、做精度加权；MVAE（NeurIPS 2018）在条件独立假设下把联合后验写成 PoE。其正确性依赖：(i) 各来源对潜变量条件独立，(ii) 精度校准，(iii) 所有专家处于同一潜坐标系，(iv) 先验只计一次，(v) 单峰高斯近似成立。违反 (i) 即"双重计数"，经典答案是协方差交叉 / KL 平均共识（保守的凸组合）或树拓扑上的信道滤波器——本设计的加法式融合隐含假设 (i)。

### Cited Findings
- **Deep Kalman Filters**：Krishnan, Shalit, Sontag, arXiv 1511.05121（2015-11-16）：用变分推断学习一大类卡尔曼滤波器，以深度网络实现任意复杂的转移与发射分布；引入 Healing MNIST，并用于 8,000 名患者 4.5 年的电子病历。— [arXiv](https://arxiv.org/abs/1511.05121v1)
- **Kalman VAE**：Fraccaro, Kamronn, Paquet, Winther, "A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning"，NIPS 2017，arXiv 1710.05741：解耦"识别模型给出的对象表示"与"描述其动力学的潜状态"；代码中识别模型为卷积 VAE、潜动力学为线性高斯状态空间模型（LGSSM）。— [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)；[arXiv](https://arxiv.org/abs/1710.05741v1)；[代码](https://github.com/simonkamronn/kvae)
- **Recurrent Kalman Networks**：Becker et al., "Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces"，ICML 2019（PMLR 97），arXiv 1905.07357：高维**因子化**潜状态使卡尔曼更新简化为标量运算，避免难以反传且不稳定的矩阵求逆；局部线性动力学传播潜状态；端到端反传、无需额外近似；优于 LSTM/GRU，且不确定性估计更准确。— [PMLR](https://proceedings.mlr.press/v97/becker19a.html)；[arXiv](https://arxiv.org/abs/1905.07357)
- **Bayesian Context Aggregation**：Volpp et al., "Bayesian Context Aggregation for Neural Processes"，ICLR 2021：把上下文聚合表述为贝叶斯推断——每个上下文编码被视作潜高斯变量的噪声观测，另一个网络估计逐点观测方差，后验为编码的**精度加权组合**；无信息先验 + 等方差时退化为均值聚合；低方差点权重更大，且权重的计算不看其他点；主要收益是更好地处理任务歧义（检索摘要 + 全文镜像）。— [ICLR](https://iclr.cc/virtual/2021/poster/3148)；[全文文本](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2021/bayesian_context_aggregation_for_neural_processes__8a1485c5.txt)。后续：Tailor, Khan, Nalisnick, "Exploiting Inferential Structure in Neural Processes"（arXiv 2306.15169）围绕该聚合构建代理潜变量模型。— [arXiv](https://arxiv.org/pdf/2306.15169)；层次世界模型论文 "Learning World Models With Hierarchical Temporal Abstractions: A Probabilistic Perspective"（arXiv 2404.16078）推导了放宽 Volpp 的恒等观测模型假设的闭式更新（检索摘要）。— [arXiv](https://arxiv.org/pdf/2404.16078)
- **MVAE / PoE**：Wu & Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning"，NeurIPS 2018，arXiv 1802.05335：在假定的**条件独立**结构下，推断网络成为专家乘积（PoE），每个模态只需一个推断网络；子采样训练方案把完整样本同时当作完整与部分观测样本；以更少参数达到 SOTA，并对不完整监督鲁棒。— [arXiv](https://arxiv.org/pdf/1802.05335)；[NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html)
- **PoE 的已知缺陷**（检索摘要综合）：高斯专家下精度更高的专家对融合结果影响更大，MoE 不受过度自信专家影响但其联合精度不随模态数增加；PoE-VAE 倾向于依赖某些模态而忽视其他；专家精度失校准会使结果有偏。— [MMVAE, Shi et al., arXiv 1911.03393](https://arxiv.org/pdf/1911.03393)；[多模态深度生成模型综述 arXiv 2207.02127](https://arxiv.org/pdf/2207.02127)。高斯乘积不能唯一确定其组成部分，缺失数据下各推断网络不唯一。— [arXiv 2403.06338](https://arxiv.org/pdf/2403.06338)。CoDE-VAE 称"专家独立"是过于乐观的假设并建模专家间依赖。— ["Aggregation of Dependent Expert Distributions", arXiv 2505.01134](https://arxiv.org/html/2505.01134v1)。Kutuzova et al. 反驳：PoE 适合合取式组合，MoE 适合析取式融合（检索摘要，原文未核实）。
- **双重计数与协方差交叉**：同一观测信息被隐式重复使用会导致过度自信估计；协方差交叉（CI）在任意拓扑下保证一致但常过于保守；精确消去公共信息需要全局拓扑与各节点估计的知识；最优消息传递方法依赖特殊拓扑（全连接、树）。**信道滤波器（channel filter）**把网络约束为树（任意两节点间唯一路径），每条链路上的滤波器跟踪并减去公共信息（检索摘要，来自 Julier/UDRC 材料）。— [UDRC 讲义](https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/publications/julier_udrc_SS20151.pdf)；[UCL Julier 论文](https://discovery-pp.ucl.ac.uk/15814/1/15814.pdf)；[KIT Fusion 2011 Noack](https://isas.iar.kit.edu/pdf/Fusion11_Noack.pdf)
- **KL 平均共识**：Battistelli, Chisci 等（IFAC 2011）证明 CI 融合规则可解释为对高斯密度的 Kullback–Leibler 平均做共识；Automatica 2014 将其作为分布式估计的融合步；2018 年工作表明指数族上的 KL 融合化为**自然参数的加权平均**，并提出事件触发共识以权衡通信与收敛速度；2024 年 Automatica 工作用加权 KL 平均处理不同质量的传感器（检索摘要）。— [UNIFI 仓库](https://flore.unifi.it/handle/2158/849929)；[hybrid consensus PDF](https://flore.unifi.it/bitstream/2158/911148/5/hybrid-consensus_revised.pdf)

### Inferences
- **本设计 = 分布式的 Bayesian Context Aggregation / RKN 观测更新 + 智能体间软约束**：当 F_ij = I 且 w→∞ 时，x_i 全相等，MAP 为 (ΣΛ_i)⁻¹ΣΛ_i z_i，即 Volpp 式精度加权（无先验）/ PoE 均值；当 w 有限时，x_i 是带软一致约束的部分池化。因此"精度加权平均学习潜变量何时有意义"的条件可直接从 PoE/卡尔曼文献导出：
  1. **同一潜坐标系**：z_i 必须表达同一隐变量（或经 F_ij 可对齐）；对独立训练的编码器这不成立，需共享编码器、联合训练或学习 F（问题 3、5）。
  2. **条件独立**：各智能体观测在给定潜变量时独立；视野重叠或共享历史时违反，导致过度自信（PoE 文献的 CoDE-VAE 论点、DDF 的双重计数）。加法式信息融合（GBP、PoE）隐含此假设；CI/KL 平均是凸组合（保守）的替代。
  3. **精度校准**：Λ_i 需反映真实误差；PoE 文献显示失校准专家会主导结果。训练上可采用 RKN/Volpp 式通过融合后输出的似然（NLL）端到端反传，使 Λ_i 获得"作为融合权重"的语义。
  4. **先验只计一次**：若每个编码器输出的是含先验的后验，加法融合会重复计入先验（MVAE 的处理是把先验作为单独专家；此为我对该论文公式的记忆，未在本次重读）。
  5. **单峰高斯**：编码器后验若多峰（如对称性/歧义），PoE 会给出错误的单峰融合；Volpp 的"任务歧义"讨论与 MoE/MoPoE 文献表明这是已知风险。
- **沿时间的 GBP 带来新的环**：即使智能体通信图是树，加入每个智能体的潜动力学链后，时空因子图出现环（i_t—j_t—j_{t+1}—i_{t+1}），"树上精确"不再成立；若改为滤波（边缘化旧状态），被边缘化的先验会把智能体耦合成稠密联合，按智能体近似即回到 DDF 的双重计数问题，需要信道滤波器式的公共信息簿记或 CI 式保守化。这是本设计时序扩展的核心理论障碍。
- RKN 的因子化（对角）假设是"对角 Λ_i"可行的先例，但 RKN 的观测模型是恒等映射；一旦 F_ij 混合维度，后验精度就不再对角（见问题 2 推论）。

### Gaps
- 未找到直接讨论"多智能体学习潜变量 + 精度加权融合 + 未知相关性"的论文（DDF 文献针对物理状态）。
- Volpp et al. 的训练目标细节、RKN 的观测方差参数化细节未在本次读原文。
- CI 与 KL 平均等价的确切条件、事件触发共识的收敛结果未读全文。
- 未检索 Haarnoja et al. "Backprop KF" 与 Kloss et al. "How to train your differentiable filter"（可微滤波训练的经验教训），应补。

## 问题 5：AI 智能体之间的潜空间通信（Interlat、LatentMAS、Thought Communication、StateBridge、2026 年"Beyond Tokens"综述）——哪些开放问题（跨架构对齐、安全）与限制映射和共识相关？

### Takeaway
2025–2026 年 LLM 智能体之间的潜通信（隐藏状态、KV-cache）迅速增长，"Beyond Tokens"综述（2026-07）归纳了 18 项代表工作，并把**跨架构对齐**、**潜通道安全/可审计性**、**压缩**列为开放问题；其融合算子是拼接、前缀、加法/线性映射、交叉注意力或缓存注入，**完全没有高斯/精度加权融合，也不把共识当作融合算子**。StateBridge（闭式正交对齐）与 SFRL（Procrustes 正交限制映射）在数学上就是同一类"限制映射"，而 2026 年的因果审计与可解释性工作表明潜通道的增益常来自非样本特异内容——这为本设计提供了"可审计、具概率语义的潜通信"的定位空间。

### Cited Findings
- **Interlat**：Du et al., "Enabling Agents to Communicate Entirely in Latent Space"，arXiv 2511.09149，ACL 2026（long paper 1248）；以模型最后一层隐藏状态作为消息、跳过解码为 token；接收端通过可训练的轻量自注意力 + 投影层（通信适配器）解读；进一步学习压缩，推断加速最高 24×且性能有竞争力；优于微调 CoT 与单智能体基线，包括跨异构模型；作者自称可行性研究。— [arXiv](https://arxiv.org/abs/2511.09149v3)；[ACL Anthology](https://aclanthology.org/2026.acl-long.1248.pdf)
- **LatentMAS**：Zou et al., "Latent Collaboration in Multi-Agent Systems"，arXiv 2511.20639（2025-11-25），ICML 2026（PMLR v306）：免训练框架，智能体在隐藏状态上推理，共享**潜工作记忆**保存并传递内部表示；准确率较单模型与文本基线最高提升 14.6%，输出 token 更少。— [arXiv](https://arxiv.org/abs/2511.20639v1)；[PMLR](https://proceedings.mlr.press/v306/zou26k.html)；[ICML 2026](https://icml.cc/virtual/2026/poster/61180)
- **Thought Communication**：Zheng et al.（CMU、Meta AI、MBZUAI），"Thought Communication in Multiagent Collaboration"，NeurIPS 2025，arXiv 2510.20733：提出"心对心"的思想通信范式；理论上证明任意两智能体之间**共享与私有潜思想**在非参数设定下可辨识。— [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/b2b502c3629beadda06311386d2c6f73-Abstract-Conference.html)；[arXiv](https://arxiv.org/pdf/2510.20733)
- **StateBridge**：Peng et al., "StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems"，arXiv 2608.13317（2026-08-13）：用**闭式正交变换**把发送者最后一层隐藏状态对齐到接收者输入空间（三阶段：消息提取、对齐、前缀注入）；在 26 个模型-任务对中 22 个最佳或并列最佳；OLMo3-7B-Think 上 LatentMAS 平均 55.1%、TextMAS 73.9%、StateBridge 76.7%；Qwen3 的 GSM8K 上落后，作者归因于连续前缀影响精确匹配下的输出格式（检索摘要）。— [arXiv](https://arxiv.org/pdf/2608.13317)
- **"Beyond Tokens" 综述**：Yingzhuo Liu, "Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent Systems"，arXiv 2606.05711（v3，2026-07-15；文献截止 2026-07-15）。三轴分类：WHAT（嵌入/隐藏状态/KV-cache/状态差分/工作区）、WHICH（不对齐、同骨干学习投影、异构骨干的通用编解码器或交互层；层映射 last→first、all→corresponding、selected→selected）、HOW（拼接、前缀、加法/学习线性映射、交叉注意力、缓存恢复/直接注入）。18 项代表工作包括 CIPHER、AC、Interlat、SDE、KVComm、Cache-to-Cache、LatentMAS、ThoughtComm、Q-KVComm、LRAgent、RelayCaching、Agent Memory、Agent Primitives、Edge LLM Handover、Vision Wormhole、BIGMAS、Mixture of Thoughts、Five Ws 综述。开放问题：**跨架构对齐**（宽度/深度/分词器不同，张量维度匹配不等于保留任务信息；建议 hub-and-spoke 编解码器与共享概念子空间；评测应区分"成对记忆"与向未见过的配对迁移）；**安全与鲁棒**（潜通道是"an opaque, high-privilege interface"，威胁含篡改、重放、跨会话缓存混淆与泄露；"Integrity is necessary but not sufficient"；混合协议应绑定可见承诺、模型身份、会话与载荷摘要）；**压缩**（核心指标应为"task utility per transmitted byte"）；**可审计性**（纯潜通道应被视为"untrusted binary interface"，需溯源、完整性与回退）。不确定性仅作为"采样单 token 丢失备选与概率"讨论；**共识未被视为融合算子；无高斯/精度加权融合**（KVComm 的高斯先验仅用于层选择）。— [arXiv](https://arxiv.org/html/2606.05711v3)；[GitHub 列表见论文](https://huggingface.co/papers/2606.05711)
- "See What I See, Know What I Think: Dense Latent Communication Across Heterogeneous Agents"（arXiv 2606.13594，2026-06-12）：异构模型 KV-cache 因深度、头结构、通道几何、位置编码不同而不对齐；压缩感知分析发现"latent communication is sparse in reasoning but dense in knowledge"（接收者可见问题时约 10/288 个 KV 组即够，不可见时约需 250 组）；学习的变换含 RoPE 去除/恢复、单调层映射、逐 KV 组 MLP 与可学习门控；在六组 Qwen3-4B/8B/14B 发送-接收对上匹配或超过文本通信，TFLOPs 少约 2–3×（如 4B→14B 为 21.5 vs 56.2）；局限：**每个发送-接收对需单独训练**，且存在隐私问题（可从通道推断任务内容）。— [Hugging Face](https://huggingface.co/papers/2606.13594)
- Zhang & Emu, "Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication"，arXiv 2607.26773（2026-07-29）：在潜表示进入接收者处做受控消息替换；"aggregate accuracy does not identify how a latent message affects the receiver"；Qwen3-4B GSM8K 上总效应 −1.00 个百分点分解为"他例消息"保留的 −6.17 与样本特异内容的 +5.17，8B 上两者方向反转；MATH-500（4B）的 15.00 点增益中 8.33 来自他例消息、6.67 来自样本特异内容。— [arXiv](https://arxiv.org/abs/2607.26773)
- ICML 2026 "What Do Latent Agents Actually Represent? Interpreting Hidden-State Communication in Multi-Agent Systems"：检索摘要称其发现潜通信表面上的丰富性反映的是基座模型几何，而非学到的组合式协调（未读原文）。— [ICML 2026](https://icml.cc/virtual/2026/69328)
- NeurIPS 2026 "Selective and Hybrid-State Communication for Latent Multi-Agent Systems"（仅核实标题）。— [NeurIPS 2026](https://neurips.cc/virtual/2026/169808)

### Inferences
- **跨架构对齐 ≈ 学习限制映射**：StateBridge 的闭式正交对齐、SFRL 的 Procrustes 正交限制映射、"See What I See" 的逐对学习变换，本质上都是在估计 F_ij（或 F_ji⁻¹F_ij）。sheaf 视角的增益是：(a) 限制映射映到**边上的公共（话语）空间**而非直接映到对方空间，天然支持不同维度与"只需在部分方向一致"；(b) 多智能体时把逐对对齐的 O(N²) 训练问题变为图上局部问题，并可用上同调（H¹）度量全局不一致（Anwer–Riess–Hale 的可辨识性结论适用）；(c) "Beyond Tokens"建议的 hub-and-spoke 编解码器 / 共享概念子空间就是星形图上的 sheaf。
- **共识作为融合算子是空白**：综述明确不存在精度加权/共识型融合；本设计把融合定义为二次能量的后验，可直接输出不确定性（后验协方差）与每条边的残差 ‖F_ij x_i − F_ji x_j‖（sheaf 不一致度），后者可作为可审计信号——回应综述对"可审计性"与"安全"的要求（检测被篡改或异常智能体的消息）。这一主张需实验支撑。
- 因果审计结果提示：本设计的评测也应包含受控消息替换（他例消息/随机消息/本例消息），以证明融合增益来自样本特异信息，而不仅是"多一个上下文"。
- 安全：精度加权融合下，恶意智能体只需上报极大 Λ_i 就能主导融合（PoE"过度自信专家主导"的对抗版本）；Robot Web 的鲁棒因子（Huber 类核）与 AME 的消融集成是可借鉴的对策。

### Gaps
- StateBridge 的数字来自检索摘要，未读全文；"What Do Latent Agents Actually Represent?" 的结论未读原文。
- 未找到 LLM 潜通信与 sheaf/限制映射的直接联系文献；"Beyond Tokens"综述后 41,990 字符（约 30%）未读。
- Interlat 是否在异构模型间学习对齐、其压缩机制细节未核实。

## 综合（目标问题）：最接近的先例、真正新颖之处与已知理论障碍

### Takeaway
本设计的每个组件都有直接先例——精度加权融合学习潜变量（Volpp 2021、RKN 2019、MVAE 2018）、异步去中心化 GBP（Robot Web 2024、DD-GBP 2026）、学习限制映射/sheaf 共识（NSD 2022、SFRL 2026、Sheaf-ADMM 2026、LatentToM 2025）、潜通信（Interlat/LatentMAS/StateBridge）——但在本次检索范围内，**没有找到把"编码器输出 (z_i, 对角 Λ_i) 作为一元因子 + 学习的 sheaf 限制映射作为智能体间二次因子 + 精确/异步 GBP 求后验 + 解码器读取融合潜变量"组合在一起的工作**。最接近的三项是 LatentToM（学习共识嵌入 + sheaf Laplacian 推断期修复）、Sheaf-ADMM（神经编码器 + sheaf 约束 + 展开优化端到端训练）、SFRL（异构潜空间间学习正交限制映射 + sheaf Laplacian 二次粘合）。主要理论障碍是：潜坐标系对齐与规范自由度、条件独立/双重计数（尤其加入时间维后时空图成环）、精度校准与对抗性过度自信、非高斯（多峰）潜后验、学习 F 时的 walk-summability/收敛性与消息稠密化。

### Cited Findings
- 最接近先例（按相似度）：LatentToM（CoRL 2025）— [项目页](https://stanfordmsl.github.io/LatentToM/)；Sheaf-ADMM（ICML 2026）— [arXiv](https://arxiv.org/abs/2605.31005)；SFRL（2026）— [arXiv](https://arxiv.org/abs/2608.10016)；SIGMA（ICRA 2025）— [arXiv](https://arxiv.org/abs/2502.06440)；Bayesian Context Aggregation（ICLR 2021）— [ICLR](https://iclr.cc/virtual/2021/poster/3148)；RKN（ICML 2019）— [PMLR](https://proceedings.mlr.press/v97/becker19a.html)；Robot Web（T-RO 2024）— [arXiv](https://arxiv.org/abs/2202.03314v2)；DD-GBP（2026）— [arXiv](https://arxiv.org/abs/2610.20460)；GBP Learning（ICML 2024）— [PMLR](https://proceedings.mlr.press/v235/nabarro24a.html)；Opinion Dynamics on Discourse Sheaves — [ar5iv](https://ar5iv.arxiv.org/html/2005.12798)；Selective Adaptation on Cellular Sheaves（2026）— [arXiv](https://arxiv.org/html/2601.22431v2)
- 双重计数/有环理论：Weiss & Freeman（若收敛则均值正确）— [NeurIPS](https://proceedings.neurips.cc/paper/1999/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)；walk-summability — [JMLR](https://jmlr2020.csail.mit.edu/papers/volume7/malioutov06a/malioutov06a.pdf)；LBP 协方差一般不正确 — [ar5iv 1410.7165](https://ar5iv.arxiv.org/html/1410.7165)；DDF 中精确消去公共信息需全局知识，最优方法依赖全连接/树拓扑 — [UDRC](https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/publications/julier_udrc_SS20151.pdf)
- 精度加权的失效模式：过度自信专家主导、模态失衡、独立性过于乐观 — [MMVAE](https://arxiv.org/pdf/1911.03393)；[CoDE-VAE](https://arxiv.org/html/2505.01134v1)
- 可辨识性：sheaf 上同调度量交互律恢复的障碍，轨迹复现不保证恢复 — [arXiv 2605.11204](https://arxiv.org/abs/2605.11204)；共享/私有潜思想的非参数可辨识性 — [Thought Communication](https://arxiv.org/pdf/2510.20733)
- 潜通信开放问题（跨架构对齐、安全、可审计、压缩）及缺少精度加权/共识融合 — [Beyond Tokens](https://arxiv.org/html/2606.05711v3)

### Inferences
- **可主张的新颖点（需逐条用实验/证明支撑）**：
  1. 以编码器预测的异方差精度 Λ_i 作为 sheaf 能量一元项（"学习的方向性固执"），把 Bosca–Ghrist 式固执与 Volpp/RKN 式精度加权统一；
  2. 在学习的潜空间上用**精确 GBP**（树上一遍、异步、丢包下固定点不变）代替 sheaf 扩散（NSD、LatentToM）、ADMM（Sheaf-ADMM）或注意力聚合（TarMAC/MAGIC/LLM 潜通信）；
  3. 学习的限制映射 F_ij 在**推断期**参与融合（而非仅训练期正则，如 SFRL），并可输出每边不一致残差作为可审计信号；
  4. 消息大小与智能体数 N 无关（MARL 共性），同时具有概率语义与可证的树上精确性（MARL 通信缺失）。
- **理论障碍清单**：
  1. *坐标/规范问题*：独立训练的潜空间只在可逆变换意义下确定；F 可吸收任意规范变换，且存在 F→0 的平凡最小能量解，需正交/等距约束（SFRL、StateBridge 的做法）或固定一侧。
  2. *条件独立与双重计数*：加法式信息融合假设智能体观测条件独立；重叠视野、共享历史、时间递归融合都会导致过度自信；CI/KL 平均（凸组合）是保守替代，但会丢失树上精确性。
  3. *时间维成环*：智能体图是树时，加入各自潜动力学链后时空因子图有环；滤波式边缘化会产生跨智能体稠密耦合。树上精确的主张只对"单时刻"或"时空图仍为树"的特殊结构成立。
  4. *消息稠密化*：对角 Λ_i 下，只要 F_ij 混合维度，精确消息/后验精度即稠密（O(d²)）；只传对角是近似，需量化误差。
  5. *收敛性*：有环时 GBP 收敛不被保证；F=I 的吸引型情形（推测）walk-summable，学习的一般 F 不保证，可能需阻尼或谱约束。
  6. *非高斯/多峰潜后验*：PoE 融合多峰后验会产生错误单峰结果；需要混合（MoE/MoPoE）或非高斯 BP。
  7. *精度校准与对抗*：精度失校准或恶意上报大 Λ_i 会主导融合；需鲁棒核（Robot Web）或消融集成（AME）。
  8. *训练*：解码器读取的是融合后潜变量，Λ_i 只有通过"穿过 GBP 反传"的端到端损失（如 GBPN、Sheaf-ADMM、RKN 的做法）才获得作为融合权重的语义；单独训练编码器 VAE 方差不保证这一点。

### Gaps
- 检索并非穷尽（约 30 次检索/抓取）；不能排除存在未被索引的"GBP + 学习 sheaf + 多智能体潜融合"工作，尤其是 2026 年下半年的 workshop 论文。
- 未核实向量值吸引型/sheaf 型高斯模型的 walk-summability 充分条件。
- 未找到关于"学习潜变量的精度加权融合在跨智能体相关观测下的校准"的实证研究。
