# 潜在世界模型基础与多视角/多智能体部分可观测动力学的低成本决定性测试平台

> 调研日期：2026-10-28。范围：(a) 哪类潜在世界模型的 latent 适合多观测者 precision-weighted 融合；(b) 能在几 GPU 小时内跑出决定性结论的环境/数据集，要求能控制带宽、丢包/延迟、位姿噪声和智能体数。
> 约定：每条事实都附来源；"仅摘要/二手"表示只读到摘要、搜索摘要或第三方综述，没读到正文表格。数字均照读到的原文转录，没读到的放入 Gaps。

## Q1. 潜在世界模型族：latent 怎么定义（全局向量 / patch token / object slot）？是否表示不确定性？状态多大？哪些天然是"每实体固定维向量"？

### Takeaway
主流 JEPA 系（DINO-WM、V-JEPA 2、PLDM、EB-JEPA、C-JEPA）和 TD-MPC2 的 latent 都是确定性点估计，没有可直接用作融合权重的方差/精度。Dreamer/RSSM 系有显式随机 latent，但 DreamerV3 起是离散 categorical 分布，不是 Gaussian。天然"每实体固定维向量"的只有 object-centric/slot 系（SlotFormer、SOLD、C-JEPA，C-JEPA 的 slot 数为 CLEVRER 7、Push-T 4）。真正把精度加权融合写进 latent 的是 RKN（因子化 Gaussian、标量 Kalman 更新）和 MVAE（Gaussian product-of-experts）这两条较老的路线，但它们都不是当下的 SOTA 世界模型。

### Cited Findings

**JEPA 系（确定性，patch / 向量 / slot）**
- EB-JEPA（"A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures"，Terver, Balestriero, …, Rabbat, LeCun, Bar；arXiv 2602.03604，2026-02-03）：教学型库，含图像 SSL、视频 latent 预测、动作条件世界模型三个示例。每个示例都按"单 GPU 几小时内训练完"设计。— [arXiv HTML](https://arxiv.org/html/2602.03604v1)
  - ICLR 2026 虚拟站点有该文条目，但读到的材料无法确认它是 workshop 还是主会论文。— [ICLR 2026 listing](https://www.iclr.cc/virtual/2026/10018036)
  - Latent 尺寸：Video-JEPA 的 latent 维度 d=16（hidden 32，ResUNet predictor，64×64 图像）；AC-Video-JEPA 的 d=32（IMPALA encoder，GRU predictor，65×65 图像，轨迹长 T=17，12 epochs，batch 384）。Image-JEPA 的 encoder 输出 512 维（ResNet-18）或 384 维（ViT-S）。— [arXiv HTML](https://arxiv.org/html/2602.03604v1)
  - Two Rooms 规划结果：MPPI（full cost）97±2%，CEM 96±2%，MPPI（只用 last state）89±2%，每次约 37 s。去掉逆动力学（IDM）项后成功率降到 1±1%。CIFAR-10 linear probe：SIGReg 91.02%，VICReg 90.12%。— [arXiv HTML](https://arxiv.org/html/2602.03604v1)
  - Latent 中没有概率或不确定性表示，VICReg 的 "variance" 指 batch 内的特征方差正则。论文为 CC BY 4.0，代码在 github.com/facebookresearch/eb_jepa，但正文没写代码许可证。— [arXiv HTML](https://arxiv.org/html/2602.03604v1)
- DINO-WM（"DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"，Zhou, Pan, LeCun, Pinto；ICML 2025，PMLR v267；arXiv 2411.04983）：— [PMLR](https://proceedings.mlr.press/v267/zhou25t.html)；[arXiv HTML](https://arxiv.org/html/2411.04983v2)
  - Latent 是冻结 DINOv2 的 14×14=196 个 patch token，每个 384 维（每帧约 196×384）。正文对输入分辨率的说法不一致：Sec 4.1 写 224×224，附录 A.9 写 resize 到 196×196。— [arXiv HTML](https://arxiv.org/html/2411.04983v2)
  - Predictor 为 ViT（depth 6，16 heads，MLP 2048，约 19M 参数），帧级 causal attention，action 经 MLP 变为 10 维后拼接到每个 patch。— [arXiv HTML](https://arxiv.org/html/2411.04983v2)
  - CLS（全局向量）与 patch 的消融，成功率依次为 Maze/Wall/Reach/Push-T：CLS 0.96/0.58/0.60/0.44，Patch 0.98/0.96/0.92/0.90。Chamfer distance（越低越好）：Rope CLS 0.84 vs Patch 0.41；Granular 0.79 vs 0.26。— [arXiv HTML](https://arxiv.org/html/2411.04983v2)
  - 规划用 CEM，每轮 100 个样本、10 次迭代。在 A6000 上一次完整 CEM 规划调用 53.0 s。没有不确定性表示（MSE 训练、确定性规划）。— [arXiv HTML](https://arxiv.org/html/2411.04983v2)
- PLDM（"Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models"，Sobal 等，含 LeCun；arXiv 2502.14819，v4 2025-10-29；搜索结果标为 NeurIPS 2025）：用无重建的 JEPA 式目标训练 latent dynamics，测试时用 MPPI 规划；换任务只需改 cost。在未见布局上的泛化和数据效率优于 model-free RL，但在大量高质量数据下 model-free 最好。代码在 github.com/vladisai/PLDM。（仅摘要）— [arXiv](https://arxiv.org/abs/2502.14819v4)
- V-JEPA 2（arXiv 2506.09985，2025-06，Meta FAIR 与 Mila）：冻结视频 encoder，用不到 62 小时无标注 Droid 机器人视频后训练出动作条件世界模型 V-JEPA 2-AC，这是一个 300M 参数的 transformer，自回归预测下一帧的表示。论文报告在两个实验室的 Franka 机械臂上零样本完成 pick-and-place（图像目标 + MPC 规划）。规划用 CEM 的说法来自第三方。— [arXiv](https://arxiv.org/abs/2506.09985)；[Meta publication page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)；CEM 说法来自 [LearnOpenCV（二手）](https://learnopencv.com/v-jepa-2-meta-world-model-robotics-guide/)
- C-JEPA（"Causal-JEPA: Learning World Models through Object-Level Latent Interventions"，v2 标题为 "…Object-Level Latent Masking"；Nam, Le Lidec, Maes, LeCun, Balestriero；arXiv 2602.11389；ICML 2026 有 poster 条目）：— [arXiv HTML](https://arxiv.org/html/2602.11389)；[ICML 2026 poster](https://icml.cc/virtual/2026/poster/63623)
  - 每帧 latent 是 N 个 object slot：CLEVRER 上 N=7，Push-T 上 N=4。主 encoder 是 VideoSAUR（聚合冻结的 DINOv2 特征），CLEVRER 上另用 SAVi 对比。Predictor 是双向注意力的 masked ViT，action 和本体感觉作为独立的 entity token 输入。— [arXiv HTML](https://arxiv.org/html/2602.11389)
  - 输入特征量只有 patch 式世界模型的 "1.02% of the total input feature size"。Push-T 成功率：DINO-WM 91.33%，OC-DINO-WM 60.67%，OC-JEPA 76.00%，C-JEPA 88.67%。规划快 8 倍以上：单张 L40s 上评估 50 条轨迹，C-JEPA 平均 673 s，DINO-WM 5,763 s。— [arXiv HTML](https://arxiv.org/html/2602.11389)
  - Counterfactual VQA（VideoSAUR）：OC-JEPA 47.68，C-JEPA 68.81（+21.13）。SAVi 版为 41.10 和 60.19。损失是 latent L2 点估计，没有不确定性。代码在 github.com/galilai-group/cjepa，权重在 HF 的 HazelNam/CJEPA。— [arXiv HTML](https://arxiv.org/html/2602.11389)；[HF](https://huggingface.co/HazelNam/CJEPA)

**Dreamer/RSSM 系（显式随机 latent）**
- DreamerV3 的 RSSM：TorchRL 中 RSSMPosteriorV3/RSSMPriorV3 默认 stochastic state 为 32 个 categorical 变量 × 32 类（展平为 1024），确定性 GRU 状态 512 维。一篇基于 DreamerV3 的论文也报告了 512 recurrent units 和 32×32 stochastic。注意这两处都是二手来源，原论文按模型规模给出的尺寸没有核对。— [TorchRL RSSMPosteriorV3](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.RSSMPosteriorV3.html)；[TorchRL RSSMPriorV3](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.modules.RSSMPriorV3.html)；[arXiv 2405.06263](https://arxiv.org/pdf/2405.06263)
- Dreamer 4（"Training Agents Inside of Scalable World Models"，Hafner, Yan, Lillicrap；arXiv 2509.24527，2025-09）：由 causal tokenizer 和 interactive dynamics model 组成，两者共用 block-causal transformer 架构。在单 GPU 上以 21 FPS 实时交互推理。是第一个只用离线数据在 Minecraft 中拿到钻石的智能体，数据量比 VPT 少 100 倍。钻石 0.7% 和 21 FPS 两个数来自二手摘要。— [项目页](https://danijar.com/project/dreamer4/)；[arXiv](https://arxiv.org/abs/2509.24527v1)
- R2I（"Mastering Memory Tasks with World Models"，Samsami 等；ICLR 2024；arXiv 2403.04253）：在 DreamerV3 中用 S4 变体（状态空间模型）替换 recurrent 部分，报告在 POPGym、BSuite、Memory Maze 上达到 SOTA，计算比 DreamerV3 最多快 9 倍。— [arXiv HTML](https://arxiv.org/html/2403.04253v1)；[mlanthology](https://mlanthology.org/iclr/2024/samsami2024iclr-mastering)

**TD-MPC2（确定性向量）**
- TD-MPC2（"TD-MPC2: Scalable, Robust World Models for Continuous Control"，arXiv 2310.16828，ICLR 2024 poster）：模型规模表中 MLP dim 为 384/512/1024/1792/4096 时，对应 latent state dim 为 128/512/768/768/1376，配 5 个 Q 函数。Latent 用 SimNorm 处理，即切成多个固定维 simplex 后各自 softmax，参考实现默认 simplex 维度 V=8。— [arXiv](https://arxiv.org/pdf/2310.16828)；V=8、τ=1 也见 [Dream-MPC（二手）](https://arxiv.org/pdf/2605.04568)

**Object-centric / slot 系（每实体固定维）**
- SlotFormer（Wu, Dvornik, Greff, Kipf, Garg；ICLR 2023；arXiv 2210.05861）：在无监督 slot 表示上用 Transformer 自回归预测未来 slot。在 CLEVRER、Physion、OBJ3D、PHYRE 上评估。CLEVRER 上把预测动力学交给 Aloe 推理，predictive 题准确率每选项 +3.4%、每题 +6.0%；Physion +1.9%。代码 MIT 许可，修正后 CLEVRER mIoU 为 49.42。— [arXiv](https://arxiv.org/pdf/2210.05861)；[GitHub](https://github.com/pairlab/SlotFormer)
- SOLD（Mosbach, Ewertz, Villar-Corrales, Behnke；ICML 2025）：从像素无监督学习 object-centric 动力学并用于 model-based RL，在需要关系推理的多物体操作任务上与 DreamerV3、TD-MPC2 对比。具体数字未核对。— [ICML 2025 PDF](https://www.ais.uni-bonn.de/papers/ICML_2025_Mosbach.pdf)
- Slot-MPC（arXiv 2605.14937，2026-05，同一 Bonn 组）：slot 表示 + 动作条件 object-centric 动力学 + 基于梯度的 MPC。在离线、覆盖有限的设定下，梯度 MPC 优于采样 MPC。（仅摘要）— [arXiv](https://arxiv.org/pdf/2605.14937)
- "Better Slots, Better Worlds"（arXiv 2608.12078，2026-08）：规划成功率与 slot 质量（FG-ARI、mBO）相关，但 slot 质量高到一定程度后收益饱和；规划用 CEM。（仅摘要）— [alphaXiv](https://www.alphaxiv.org/abs/2608.12078)

**显式精度加权融合的 latent（较老，但与本问题直接对应）**
- RKN（"Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces"，Becker 等；ICML 2019）：学一个高维特征空间，并假设 belief 可分解为独立 Gaussian，使 Kalman 更新退化为标量运算、不需要矩阵求逆，可端到端反传。报告其不确定性估计比 LSTM/GRU 准得多。— [PMLR](https://proceedings.mlr.press/v97/becker19a.html)；[arXiv](https://arxiv.org/abs/1905.07357)
- MVAE（"Multimodal Generative Models for Scalable Weakly-Supervised Learning"，Wu & Goodman；NeurIPS 2018；arXiv 1802.05335）：联合后验正比于各单模态后验之积再除以先验，所以推断网络取 product-of-experts，Gaussian 专家之积有闭式的精度加权均值和方差。训练时用子采样 ELBO 处理缺失模态。已知问题：Gaussian 之积不能唯一确定各个专家。— [arXiv](https://arxiv.org/html/1802.05335)；[NeurIPS](https://papers.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html)
- 后续工作有 MMVAE（mixture-of-experts）、MoPoE，以及 HELVAE（Hellinger 聚合，用矩匹配投影回单个 Gaussian，免去子采样；AISTATS 2026）。— [HELVAE alphaXiv](https://www.alphaxiv.org/abs/2601.06572v4)；[AISTATS 2026 slides](https://virtual.aistats.org/media/aistats-2026/Slides/13434.pdf)

### Inferences
- 按"每实体固定维 + 可附精度"来排：slot 系（C-JEPA、SlotFormer、SOLD）在结构上最贴合 sheaf 节点的定义（一个实体对应一个固定维向量），但都没有方差输出。要做 precision-weighted / GBP 融合，需要自己加一个 per-slot 的对数精度头，可以按 RKN 的因子化 Gaussian 或 MVAE 的 PoE 来做。这是最小改动：只换融合层，不动 encoder 和 predictor。
- DINO-WM 的 196×384 patch latent 体量最大，而且 patch 绑定在相机像素坐标上。多个观测者视角不同时，patch 之间没有对应关系，融合前必须先做配准或 lifting，这会把位姿噪声直接引入融合误差。全局向量（DINO CLS）没有配准问题，但在 Wall/Push-T 上已明显掉点（0.58、0.44 vs 0.96、0.90），说明单个全局向量会丢失空间细节。
- DreamerV3 的 categorical latent 可以通过对 logits 求和（即独立 categorical 的乘积）来融合，这是 PoE 的离散版本，但它不是"精度加权"的连续形式。TD-MPC2 的 SimNorm latent 是 simplex 上的确定点，没有现成的不确定性。
- C-JEPA 只用 1.02% 的特征量就把 Push-T 做到 88.67%（DINO-WM 为 91.33%），说明 slot latent 在带宽受限的通信中天然占优：每个智能体只需发 N×d 个数，而不是 196×384。

### Gaps
- C-JEPA 的 slot 维度 d、SOLD 的 slot 维度和数量、V-JEPA 2 encoder 输出 token 的维度、Dreamer 4 tokenizer latent 的维度与数量，正文中均没有读到具体数值。
- DreamerV3 原论文中不同模型规模对应的 RSSM 尺寸，只从 TorchRL 文档和二手论文得到 32×32 + 512，没有核对原文。
- PLDM 的 latent 形状（空间特征图还是向量）未读到。
- 没有找到任何把 JEPA latent 本身做成 Gaussian（mean + precision）并用于多观测者融合的已发表工作。
- EB-JEPA 是 ICLR 2026 workshop 还是主会，无法确认。

## Q2. 多视角/多智能体动力学数据集与模拟器：哪些提供每智能体的局部视角、可用于评分的真值状态，并且跑起来便宜？

### Takeaway
最便宜、可控性最强的是 VMAS（PyTorch 向量化 2D 物理，可开数万个并行环境，自带 LIDAR 传感器和通信动作，GPL-3.0）和 JaxMARL（MPE、Overcooked、SMAX 的 JAX 版，SMAX 比 SMAC 快约 40,000 倍）。视觉多相机方向，Kubric/MOVi 有多相机版本 MOVi-MC-AC（2041 个场景，每个相机都有 RGB、实例掩码和深度），CLEVR-MultiView 是多视角动态场景生成器。真实多智能体时序数据以 V2XPnP（4 个智能体，100 个场景）为代表，但成本高。SMACv2、Melting Pot、COMBO（TDW）可用作补充，但都不直接提供"每智能体局部视角 + 全局实体真值 + 可控通信"这三者的组合。

### Cited Findings

**向量化 2D 多机器人模拟器**
- VMAS（"VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning"，arXiv 2207.03530）：— [arXiv](https://arxiv.org/pdf/2207.03530)；[GitHub](https://github.com/proroklab/VectorizedMultiAgentSimulator)
  - PyTorch 向量化 2D 物理引擎，在加速硬件上可扩展到数万个并行环境，支持 `device="cuda"`。许可证 GPL-3.0。
  - 与部分可观测相关的场景：`discovery`（对智能体和目标的 LIDAR）、`flocking`（LIDAR）、`navigation`（可选 LIDAR）、`sampling`（每个智能体只看自身周围 3×3 格的样本值）、`road_traffic`（支持 ego-view 和 BEV，支持部分观测）。`dropout` 场景"需要通信才能解"。
  - 已移植全部 MPE 场景，其中 `simple_crypto`、`simple_reference`、`simple_speaker_listener`、`simple_world_comm` 为通信类。
  - 智能体可配置连续或离散的通信动作，但这种通信不可微。每个场景自定义 `observation()`，可输出 tensor 或嵌套 dict。场景代码直接访问实体状态（例如 `agent.state.pos`），README 没有明确说明 wrapper 是否暴露全局状态。
  - 集成了 RLlib、TorchRL、BenchMARL、Gym/Gymnasium，README 未提 PettingZoo。
- BenchMARL（arXiv 2312.01472）：基于 TorchRL 的 MARL 训练库，支持 VMAS 任务。MASAC 的 critic 使用全局观测，ISAC 不使用。— [arXiv](https://arxiv.org/pdf/2312.01472)

**JAX 加速的 MARL 环境**
- JaxMARL（"JaxMARL: Multi-Agent RL Environments and Algorithms in JAX"，NeurIPS 2024 Datasets & Benchmarks；arXiv 2311.10090）：— [NeurIPS D&B PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5aee125f052c90e326dcf6f380df94f6-Paper-Datasets_and_Benchmarks_Track.pdf)；[arXiv v5](https://arxiv.org/html/2311.10090v5)
  - 包含 MPE、MABrax、Hanabi、Overcooked、SMAX，覆盖全观测和部分观测。
  - 训练流水线最多快 14 倍；把多次训练一起向量化时最多快 12,500 倍。
  - SMAX 在单张 NVIDIA 2080 上比等价的 SMAC 实现快 40,000 倍，而且不依赖 StarCraft II 引擎。
- SMACv2（"SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning"，Ellis 等；NeurIPS 2023 D&B；arXiv 2212.07489）：每个 episode 随机生成队伍组成和起始位置；用 SC2 的真实视野范围取代 SMAC 的固定值，最小攻击距离设为 2。作者认为原 SMAC 视野过大，部分可观测性不足，因此提出 EPO（extended partial observability），只有第一个发现敌人的智能体能确定看到它。— [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2023/file/764c18ad230f9e7bf6a77ffc2312c55e-Paper-Datasets_and_Benchmarks.pdf)；[arXiv](https://arxiv.org/pdf/2212.07489)
- Melting Pot（Google DeepMind；基于 DeepMind Lab2D；PyPI 包名 dm-meltingpot）：仓库 README 描述为 21 个 substrate 和 85 个以上测试场景，自带 RLlib 和 PettingZoo + SB3 训练示例。这组数字可能对应 v1，Melting Pot 2.0 技术报告的数字可能不同，未核对。— [GitHub README](https://github.com/google-deepmind/meltingpot/blob/main/README.md)；[PyPI](https://pypi.org/project/dm-meltingpot/)；[DeepMind blog](https://deepmind.google/discover/blog/melting-pot-an-evaluation-suite-for-multi-agent-reinforcement-learning/)

**视觉多相机物体动力学**
- Kubric（"Kubric: A scalable dataset generator"，arXiv 2203.03570）：基于 PyBullet（物理）和 Blender（渲染），代码 Apache-2.0。输出实例分割、深度和光流。— [arXiv](https://arxiv.org/pdf/2203.03570)；[ecosyste.ms](https://science.ecosyste.ms/projects/278199)
  - MOVi-A 到 C：最多 10 个运动物体，每段 24 帧。MOVi-D/E：最多 23 个物体，只有少数在动；MOVi-E 的相机随机移动。— [arXiv 2203.03570](https://arxiv.org/pdf/2203.03570)
- MOVi-MC-AC（MOVi Multi-Camera Amodal Content，arXiv 2507.00339）：2041 个场景（训练 1651，测试 390），每个相机提供 RGB 视频、可见物体的实例掩码和场景深度。— [arXiv](https://arxiv.org/pdf/2507.00339)
- CLEVR-MultiView：多视角、相机可移动的动态场景生成器，包含碰撞，来自 DyMON/MulMON 的作者。— [GitHub](https://github.com/NanboLi/CLEVR-MultiView)
- CLEVRER：单视角，带四类模板生成的 VQA（descriptive、explanatory、predictive、counterfactual），可从官网获得真值分割掩码。Physion（NeurIPS 2021）：用 readout 模型判断 agent 物体是否接触 patient 物体来评估；能直接访问物理状态的 GNN 明显更好。— [arXiv 2106.08261 / HF](https://huggingface.co/papers/2106.08261)；[Physion NeurIPS 2021](https://neurips.cc/virtual/2021/22771)；[arXiv 2410.15728](https://arxiv.org/pdf/2410.15728)

**多智能体具身模拟与真实协同感知数据**
- COMBO（"COMBO: Compositional World Models for Embodied Multi-Agent Cooperation"，ICLR 2025；arXiv 2404.10775）：每个智能体只有 egocentric 局部视图，先从局部观测估计完整世界状态，再用组合式视频生成世界模型模拟联合动作。基于 ThreeDWorld（TDW-Game、TDW-Cook），2 到 4 个智能体。— [arXiv](https://arxiv.org/abs/2404.10775v2)；[ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7d03c6bf9f07acb4038eea96c63db52d-Abstract-Conference.html)
- V2XPnP（"V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction"，Zhou, Ma 等；ICCV 2025；arXiv 2412.01812）：真实 4 智能体时序数据集，共 100 个场景，支持 VC/IC/V2V/I2I 四种协作模式。比较了一步与多步通信，以及 early/late/intermediate 三种融合。VC 基准上为 69.2 AP@0.5、1.26 m ADE，FFNet 为 66.1、1.41。— [项目页](https://mobility-lab.seas.ucla.edu/v2xpnp/)；[arXiv](https://arxiv.org/abs/2412.01812v2)
- 协同感知标准数据集：OPV2V、V2X-Sim、DAIR-V2X，以及 Where2comm 自建的 CoPerception-UAVs（车与无人机两类智能体，相机和 LiDAR）。— [Where2comm arXiv](https://arxiv.org/abs/2209.12836)

### Inferences
- 决定性实验首选 VMAS。理由一：GPU 向量化，几 GPU 小时足够训练加扫参。理由二：真值实体状态可以直接从场景对象读出，适合给每个实体打分。理由三：LIDAR 或自定义局部 FOV 传感器可以精确控制"每个观测者看到哪部分"。理由四：通信是显式动作，便于插入量化（控制带宽）、伯努利丢包、k 步延迟和位姿高斯噪声。注意 GPL-3.0 对代码分发的约束。
- 需要像素输入时，用 Kubric 自渲染多相机 MOVi 场景（MOVi-MC-AC 已有 2041 个现成场景）。每个相机就是一个观测者，真值物体状态可以从 Kubric 导出，用于 slot 级打分。渲染成本远高于 VMAS，建议只做第二阶段验证。
- JaxMARL 的 MPE/SMAX 适合"规划/控制"侧评分，但 MPE 观测本身就是低维向量，部分可观测性弱；SMACv2 的 EPO 才有较强的部分可观测性，但其中的通信与融合需要自己实现。
- V2XPnP、OPV2V 这类数据集离"几 GPU 小时"很远，而且对象是 3D 检测，不是 latent 世界模型。更适合借用它们的协议（带宽、位姿、延迟扫描），而不是直接拿来当测试平台。

### Gaps
- JaxMARL、SMACv2、Melting Pot、CLEVRER、Physion 以及 MOVi 数据本身的许可证都没有读到原文（只确认了 Kubric 代码为 Apache-2.0、VMAS 为 GPL-3.0、SlotFormer 为 MIT）。
- VMAS 的最新版本号和发布日期在 GitHub 页面上没有显示。论文中的吞吐基准（每秒步数）没有读到具体数值。
- Melting Pot 2.0 的 substrate 数量、观测分辨率和 egocentric 窗口尺寸没有读到。
- MOVi-MC-AC 每个场景的相机数量没有读到。
- 没有找到"多相机 + 实体真值 + 内建通信/带宽控制"三者齐备的现成基准，需要自己组装。

## Q3. 关于"部分可观测下的共享状态估计"和"预测质量-通信带宽曲线"，现有哪些评估协议？

### Takeaway
最成熟的协议来自协同感知：精度对通信量（带宽）的曲线（Where2comm 声称在 OPV2V 上通信量降低超过 100,000 倍）、位姿噪声扫描（零均值 Gaussian，std 最高 1.0 m，沿用 CoAlign）、延迟扫描（0 到 500 ms，沿用 V2X-ViT）、丢包率扫描（V2X-INCOP）。多智能体潜在世界模型方面，CS-JEPA（2026）提出同时报告集体状态误差和智能体间一致性，并在拓扑和群体规模偏移下测试；但没有覆盖丢包、延迟和噪声。MARL 通信方面，IMAC 把带宽约束形式化为消息熵约束。目前没有找到把带宽、丢包、延迟、位姿噪声和 N 统一起来、并面向 latent 世界模型的标准协议。

### Cited Findings

**协同感知协议**
- Where2comm（Hu, Fang, Lei, Zhong, Chen；NeurIPS 2022；arXiv 2209.12836）：把"带宽 vs 感知精度"定为协同感知的核心矛盾，靠调整参与通信的空间区域来适配不同带宽。声称在 OPV2V 上通信量降低超过 100,000 倍，并且仍优于 DiscoNet 和 V2X-ViT。在 OPV2V、V2X-Sim、DAIR-V2X、CoPerception-UAVs 上做 3D 检测。— [arXiv abs](https://arxiv.org/abs/2209.12836)；[NeurIPS 2022 PDF](https://proceedings.nips.cc/paper_files/paper/2022/file/1f5c5cd01b864d53cc5fa0a3472e152e-Paper-Conference.pdf)
- 鲁棒性方面的分歧：Where2comm 自称对位姿噪声鲁棒，并通过选择性通信降低延迟影响；Select2Drive 则报告 Where2comm 在高延迟下性能下降，原因是融合了过时信息。— [Where2comm](https://arxiv.org/pdf/2209.12836)；矛盾见 [Select2Drive arXiv 2501.12040](https://arxiv.org/pdf/2501.12040)
- 常用扰动协议（CoLC 的描述）：位姿噪声为零均值 Gaussian，std 最高 1.0 m（沿用 CoAlign）；延迟扫描 0 到 500 ms（沿用 V2X-ViT）。— [CoLC arXiv 2603.00682](https://arxiv.org/pdf/2603.00682)
- CodeFilling 报告基线在位姿误差超过 0.6 m 或延迟超过 100 ms 后失效（搜索摘要级，属于该文自己的设定）。— [arXiv 2405.04966](https://arxiv.org/pdf/2405.04966)
- CoBEVFlow（"Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow"）专门处理异步和延迟。— [arXiv 2309.16940](https://arxiv.org/pdf/2309.16940)
- V2X-INCOP（IEEE T-IV 2024；arXiv 2304.11821）：对通信中断（丢包）建模，利用历史协作信息恢复缺失内容。在不同丢包率下取平均，协同感知相对单车感知的增益在 OPV2V、V2X-Sim、DAIR-V2X 上分别最高 14.06%、13.9%、12.07%。— [arXiv](https://arxiv.org/abs/2304.11821v2)
- 2025 年的一项实证研究在 32 条测试路线上在线评估协同感知，覆盖正常和异常通信（延迟、位姿误差）两种条件。— [arXiv 2509.24927](https://arxiv.org/pdf/2509.24927)
- V2XPnP 把"感知 + 预测"一起评估（AP@0.5 和 ADE），并比较一步与多步通信。— [项目页](https://mobility-lab.seas.ucla.edu/v2xpnp/)

**多智能体世界模型与共享状态**
- CS-JEPA（"One Future, Every Robot: Label-Efficient Collective-State Prediction with Decentralized JEPA"，Gazzaev, Gavrilov, Muravyov；arXiv 2607.28443，v3 2026-08-06，投稿 ICRA 2027）：每个机器人根据自身历史和"有界的邻居消息"，预测同一个固定宽度的集体未来 latent。训练不用 agreement loss，部署时不汇总预测。— [arXiv](https://arxiv.org/abs/2607.28443)
  - 指标为集体状态误差和机器人间一致性。相对容量匹配的原始未来重建基线，误差在分布内降低 28.4%，在拓扑和群体规模偏移下降低 64.4% 到 75.6%。一致性在每个 seed、每个评估 split 上都有提升。
  - 摘要未提丢包、延迟和噪声。
- DCWM（"Decentralized Collective World Model for Emergent Communication and Coordination"，Nomura, Aoki, Taniguchi, Horii；IEEE ICDL 2025；arXiv 2504.03353）：在时间维上扩展 collective predictive coding，用对比学习对齐消息；任务是两智能体轨迹绘制。智能体感知能力不同时，有通信的模型优于无通信模型，协调性能仅次于集中式模型。限制智能体直接访问彼此内部状态时，涌现符号更能反映环境状态。— [arXiv](https://arxiv.org/abs/2504.03353)
- Physion 的 readout 协议：在冻结表示上训练接触预测分类器。可以借来做"共享 latent 是否编码了真值状态"的探针评估。— [Physion](https://neurips.cc/virtual/2021/22771)

**MARL 通信中的带宽形式化**
- IMAC（"Learning Efficient Multi-agent Communication: An Information Bottleneck Approach"，Wang 等；ICML 2020；arXiv 1911.06992）：证明带宽限制可以转化为消息熵约束；用变分信息瓶颈加定制 batchnorm 控制消息熵来模拟带宽。— [arXiv](https://arxiv.org/abs/1911.06992)；[ICML 2020](https://icml.cc/virtual/2020/poster/5864)

### Inferences
- 一个可直接复用的协议组合：
  - x 轴取比特/消息（量化位宽 × 维度），也可按 IMAC 的思路取消息熵上界。
  - y 轴取未来 k 步实体状态误差（解码到真值）和规划成功率（参考 DINO-WM、C-JEPA 的 MPC 成功率）。
  - 另加一条 CS-JEPA 式的"智能体间一致性"曲线，并检查一致性提升没有以精度为代价（排除所有智能体塌缩到同一个错误预测）。
  - 扰动沿用协同感知惯例：位姿 std 取 0 到 1.0（按环境尺度归一），延迟取 0 到若干步，丢包率取 0 到 0.5 或更高；另外扫描 N。
- 预先登记判据的建议（与"两 seed 必须都通过"的习惯一致）：精度加权融合在中等带宽下的未来预测误差，要显著低于朴素平均和无通信，并且至少恢复"无通信到集中式上界"差距的一个预定比例。这个比例需在实验前写死，本文不建议具体数值。

### Gaps
- Where2comm 原文中通信量的具体单位和曲线采样方式（例如是否取 log2 字节）没有读到正文。
- V2X-ViT、CoAlign 原文中噪声协议的完整参数（是否包含航向角噪声）没有核对。
- 没有找到面向 latent 世界模型、同时扫描带宽、丢包、延迟、位姿噪声和 N 的公开基准或协议论文。
- CS-JEPA 的实验环境、机器人数量和消息尺寸在摘要页没有给出。

## Q4. 决定性实验需要哪些基线（无通信、集中式 attention/拼接上界、朴素 latent 平均、TarMAC 式学习通信），已有论文如何实现？

### Takeaway
已有文献中的基线大致分为五类：(1) 无通信或独立局部模型（MA-JEPA 的"matched independent local model"，DCWM 的非通信模型）；(2) 集中式或联合预测（DCWM 的 centralized、MWM-MARL 的 attention 全局融合、MA-JEPA 训练时的 joint predictor）；(3) 早期/后期/中间融合（V2XPnP）；(4) 学习通信（CommNet、IC3Net 门控、TarMAC 签名注意力、IMAC 信息瓶颈、MAMBA 的离散邻居消息）；(5) 概率融合（MVAE 的 PoE、RKN 的 Kalman 更新）。朴素平均作为显式基线很少单独报告。

### Cited Findings

**学习通信基线**
- TarMAC（"TarMAC: Targeted Multi-Agent Communication"；ICML 2019，arXiv 1810.11187）：发送方在消息之外广播一个 key（签名），接收方据此做软注意力，支持多轮通信。与 IC3Net 结合后称为 T-IC3Net，在 Predator-Prey 上收敛更快（3 个 seed）。注意力发送会以低概率泄露状态，在竞争场景中不利。有一则第三方综述误写为 ICML 2018，ICML 幻灯片 URL 显示为 2019。— [arXiv](https://arxiv.org/pdf/1810.11187)；[ICML 2019 slides](https://ICML.cc/media/icml-2019/Slides/5140.pdf)
- CommNet 用单一可学习通道，IC3Net 在完全图上加门控来决定何时通信。在 I2C（NeurIPS 2020）中，审稿人指出 IC3Net 和 TarMAC 在协作导航和 predator-prey 上不如 full communication。— [arXiv 2006.06455（I2C）](https://arxiv.org/pdf/2006.06455)；[NeurIPS 2020 review](https://papers.nips.cc/paper_files/paper/2020/file/fb2fcd534b0ff3bbed73cc51df620323-Review.html)
- IMAC：信息瓶颈约束消息熵，用来模拟受限带宽。— [arXiv](https://arxiv.org/abs/1911.06992)

**世界模型类基线**
- MAMBA（"Scalable Multi-Agent Model-Based Reinforcement Learning"，Egorov & Shpilman；AAMAS 2022；arXiv 2205.15023）：基于 DreamerV2 的 latent 世界模型，用 attention。执行时智能体与邻居交换离散消息来维持各自的世界模型。在 SMAC 和 Flatland 上评估，在低数据量下有优势。— [arXiv](https://arxiv.org/pdf/2205.15023)
- arXiv 2508.02912 比较了基于世界模型的工程化通信与涌现学习通信，报告前者随任务复杂度增加扩展更好（仅搜索摘要）。— [arXiv](https://arxiv.org/html/2508.02912v3)
- MA-JEPA（"Learning World Models with Joint-Embedding Prediction for Single- and Multi-Agent Reinforcement Learning"，Kaplowitz, Obahor, Schroeder de Witt；NeurIPS 2026 workshop poster）：categorical latent，causal Transformer。训练专用的 joint predictor 用全部智能体的局部状态和联合动作，预测每个智能体下一步局部观测的表示；执行保持去中心化。在 SMAC 上优于匹配的独立局部模型，在视觉 DMC 上与 DreamerV3 相当。页面没有给出数值。— [NeurIPS 2026](https://neurips.cc/virtual/2026/159678)
- MWM-MARL（"From Pixels to Cooperation: Multi-Agent Reinforcement Learning based on Multimodal World Models"，arXiv 2511.01310）：共享的多模态世界模型用 attention 把各智能体 embedding 融合为一个全局上下文。报告在智能体感官不对称时融合必不可少。— [arXiv](https://arxiv.org/abs/2511.01310)
- SeqWM（arXiv 2509.13095）：每个智能体有自己的 latent 世界模型，按顺序串联，训练损失不穿过通信通道反传。IWoL（arXiv 2509.25550）：训练时用全局状态学一个"交互世界 latent"，执行只用局部观测加通信。（均仅摘要）— [SeqWM](https://arxiv.org/html/2509.13095v2)；[IWoL](https://arxiv.org/html/2509.25550v3)
- "Dreaming of Others"（arXiv 2605.31361）的相关工作指出，以往共享想象的方法常假设智能体能访问共享 latent 状态，从而回避了部分可观测问题（仅摘要）。— [arXiv](https://arxiv.org/pdf/2605.31361)
- CS-JEPA 的主基线是"容量匹配的原始未来重建"，并加了 translation-free 和 crossed-pretraining 两个对照。— [arXiv](https://arxiv.org/abs/2607.28443)

**融合方式基线**
- V2XPnP 系统比较了 early（原始数据）、late（结果级）和 intermediate（特征级）三种融合，以及一步和多步通信，主模型为 intermediate + 一步。— [arXiv](https://arxiv.org/abs/2412.01812v2)
- 概率融合基线：MVAE 的 Gaussian PoE 给出闭式精度加权均值和方差；RKN 用因子化 Gaussian 做标量 Kalman 更新。— [MVAE](https://arxiv.org/html/1802.05335)；[RKN](https://proceedings.mlr.press/v97/becker19a.html)
- DCWM 的基线是非通信模型和集中式模型，集中式最好，DCWM 次之。— [arXiv](https://arxiv.org/abs/2504.03353)

### Inferences
- 最小基线集，在同一 encoder、predictor 和参数预算下比较：
  - B0 无通信（每个智能体只用自己的局部 latent）。
  - B1 朴素平均（对每个实体 slot 等权取均值）。
  - B2 PoE/Kalman 精度加权（每个 slot 带对数精度，闭式融合；这就是 GBP 单轮、无环情形的特例）。
  - B3 学习通信（TarMAC 签名注意力，同等比特预算）。
  - B4 集中式上界（全部原始观测拼接或 attention 输入单个模型，即 MWM-MARL 或 MA-JEPA joint predictor 的形态）。
  - 被测方法 Sheaf-GBP：多轮消息传递加 restriction map。
- 关键对照是 B1 对 B2：如果精度加权在位姿噪声和遮挡不对称的情况下不能明显优于朴素平均，那么"精度"这条主张就不成立。B2 对 Sheaf-GBP 检验的是 restriction map 和多轮迭代的增益。
- TarMAC 在部分任务上不如 full communication（I2C 审稿意见），所以 B3 不能当作学习通信的上界。应当同时报告 B4。

### Gaps
- 没有找到在 latent 世界模型设定下把"朴素平均"作为显式基线并报告数值的论文。
- BenchMARL 是否内建 TarMAC/CommNet 等通信算法没有核对，读到的只有 MASAC/ISAC 的 critic 配置。
- MA-JEPA、MWM-MARL、IWoL、SeqWM 的具体数值和实现细节没有读到正文。

## Q5. 单智能体世界模型在部分可观测或多视角输入下有哪些已知结果，能否给多智能体版本的可能增益定界？

### Takeaway
现有证据都是间接的。(1) 空间信息丢失代价很大：DINO-WM 中全局 CLS 换成 patch 后，Wall 从 0.58 升到 0.96，Push-T 从 0.44 升到 0.90。(2) Object-centric 压缩的代价很小：C-JEPA 用 1.02% 的特征量达到 88.67%，DINO-WM 为 91.33%。(3) 多视角训练主要带来视角鲁棒性（MV-MWM）。(4) 记忆能补偿部分可观测（R2I）。(5) 协同或集中融合相对单体的增益在协同感知中约 12% 到 14%（V2X-INCOP），在 DCWM 中呈"集中式 > 通信 > 无通信"的排序。没有找到关于"视角数或智能体数"的 scaling law。

### Cited Findings
- MV-MWM（"Multi-View Masked World Models for Visual Robotic Manipulation"，Seo, Kim, James, Lee, Shin, Abbeel；ICML 2023，PMLR v202 pp. 30613–30632；arXiv 2302.02408）：多视角 MAE 重建被随机遮挡视角的像素，再在其表示上学世界模型。用随机视角训练后，策略能承受强视角随机化，并且不需要相机标定就能迁移到真实机器人。在 RLBench 上同时测了多视角控制和"单视角控制 + 辅助相机做表示学习"两种设置。— [PMLR](https://proceedings.mlr.press/v202/seo23a.html)；[arXiv](https://arxiv.org/pdf/2302.02408)
- DINO-WM 的 CLS 与 patch 对比见 Q1：全局向量在需要空间细节的任务上大幅掉点。— [arXiv HTML](https://arxiv.org/html/2411.04983v2)
- C-JEPA：object-level masking 迫使模型从其他物体推断被遮挡物体的状态，相当于在 latent 中模拟"某观测者看不到某实体"。Counterfactual 准确率提高 21.13 个点。— [arXiv HTML](https://arxiv.org/html/2602.11389)
- R2I：在 POPGym、BSuite、Memory Maze 等记忆/部分可观测任务上达到 SOTA，在 Atari/DMC 等标准任务上与 DreamerV3 相当。— [arXiv HTML](https://arxiv.org/html/2403.04253v1)；[mlanthology](https://mlanthology.org/iclr/2024/samsami2024iclr-mastering)
- V2X-INCOP：在不同丢包率下取平均，协同感知相对单车最高增益为 14.06%（OPV2V）、13.9%（V2X-Sim）、12.07%（DAIR-V2X）。— [arXiv](https://arxiv.org/abs/2304.11821v2)
- DCWM：通信模型协调性能仅次于集中式，在感知能力不对称时优于无通信。— [arXiv](https://arxiv.org/abs/2504.03353)
- MWM-MARL：感官不对称时 attention 融合是必要的（仅摘要）。— [arXiv](https://arxiv.org/abs/2511.01310)
- Physion：能直接访问物理状态的 GNN 明显优于基于视觉的模型，说明"拿到真值状态"本身就是一个强上界。— [Physion](https://neurips.cc/virtual/2021/22771)
- EB-JEPA：只用最终状态做 cost 时成功率 89±2%，用全程 cost 时 97±2%（同一 latent）；去掉 IDM 后降到 1±1%，说明小 latent 世界模型的成败可能由正则项而不是信息量决定。— [arXiv HTML](https://arxiv.org/html/2602.03604v1)

### Inferences
- 定界思路：多智能体融合的最大增益不超过"集中式上界减去无通信"的差距（B4 − B0）。可以先用单智能体实验测出两端：用全局视角（或所有相机拼接）训练的单模型作为上界，用单个局部视角训练的模型作为下界。如果这个差距本身很小，例如环境太容易或记忆已经足够补偿，那么任何融合方法都不可能显示出决定性差异。因此 testbed 必须先设计成 B4 − B0 足够大：遮挡强、视角互补、动态不可从单视角历史推断。
- C-JEPA 的结果（用 1% 的特征量达到相近的规划成功率）提示：在带宽受限的场景中，slot latent 加融合有可能在低比特区间逼近集中式 patch 上界。这需要实验验证。
- 训练正则（IDM、SIGReg/VICReg）对小 latent 世界模型影响极大（EB-JEPA 中去掉 IDM 从 97% 降到 1%）。决定性实验必须在所有基线上固定同一组正则，否则融合的效果会被训练方面的差异淹没。

### Gaps
- 没有找到"视角数或相机数 → 预测误差"或"智能体数 → 预测误差"的 scaling 曲线，无论单智能体还是多智能体世界模型。
- MV-MWM 的具体成功率数字没有读到正文。
- 对 POPGym 上 DreamerV3 与 R2I 的具体分数差距，没有读到表格。
- V-JEPA 2 和 Dreamer 4 在部分可观测或多视角下的表现，没有找到相关报告。
