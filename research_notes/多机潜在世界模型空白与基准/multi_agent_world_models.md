# 多智能体世界模型（2022–2026）：潜在状态如何定义、共享、融合与对齐

> 调研日期 2026-10-28。范围：model-based MARL 与具身多智能体的世界模型。目的：判断"每个智能体持有固定维潜变量 + 精度（不确定性），在二次 sheaf 能量上用高斯置信传播（GBP）达成共识（去中心化、定长消息、对丢包/延迟鲁棒）"这一方法，哪些已被解决、哪些仍空白。
> 标注约定：【同行评审】= 已确认会议/期刊；【预印本】= 仅 arXiv/未确认录用；【工作坊】= workshop。"未读到"= 我没有在原文中读到该信息，不做猜测。

## Q1. 主要的多智能体世界模型论文有哪些？各自如何定义和共享潜在状态？

### Takeaway
2022–2026 的主线是 Dreamer/Transformer 式潜变量世界模型向多智能体的扩展：潜在状态几乎都是"每个智能体一份"（离散类别潜变量或 VQ token），跨智能体信息靠注意力/Transformer/Perceiver/GNN 在训练或想象中聚合；只有 MAMBA、CoDreamer 以及两篇小规模集体预测编码工作（DCWM、Dreamer-CPC）在执行期真正交换潜变量消息。2026 年出现的生成式视频世界模型（Solaris、Gamma-World）则是单一集中式模型联合生成所有智能体视角。

### Cited Findings

#### A. 逐篇记录（model-based MARL 主线）

**MAMBA — "Scalable Multi-Agent Model-Based Reinforcement Learning"，Egorov & Shpilman，AAMAS 2022【同行评审，arXiv 2205.15023】**
- 潜在状态：每个智能体一个 DreamerV2 式 RSSM；随机状态为 32 个类别分布 × 32 类，执行与想象阶段"只允许传输随机状态 z"，每条消息 32·log 32 = 160 bit — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 通信内容：每个智能体广播上一步的 z_{t-1}，并假设能观察到他人上一步动作 a_{t-1}；(z_{t-1}, a_{t-1}) 经 3 层 Transformer "Communication Block" 变为特征 e_t^i，用于更新本智能体世界模型并预测动作 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 去中心化：摘要称"智能体间通信足以在执行期为每个智能体维持一个世界模型"，想象 rollout 用于训练 — [arXiv abs](https://arxiv.org/abs/2205.15023)
- 一致性问题：作者明确指出用注意力独立预测下一潜变量会造成智能体对环境当前视图的不一致；不采用自回归（会拖慢训练且不允许去中心化执行），而是加互信息项 L_info = −ln p(a_{t-1}^i | h_t^i, z_t^i)，使各自潜变量主要依赖自身动作，从而"可分开派发" — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 局部性：可在注意力中屏蔽非邻居；Flatland 执行期只接收邻居（铁路上邻近）的消息；SMAC 中未测试局部性 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 计算/通信权衡：广播 (z,a) 时每个智能体需 O(n²) 计算单层注意力，交换 q_iK^T 等中间向量可降到 O(n) — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- "reward-agnostic communication"：消息由世界模型（非奖励）学得，作者认为因此不适用于混合（竞争）环境 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 不确定性：随机离散潜变量 + KL balancing；未读到用不确定性做融合加权的机制 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 鲁棒性：Communication Block 内 dropout p=0.1 用于防过拟合（不是信道丢包实验）；未读到消息丢失/延迟实验；智能体数量在 Flatland 中为 5/10/15 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- 结果（SMAC，3 次运行均值(std) 胜率%，Table 3）：2s_vs_1sc@100k 98(1) vs MAPPO 78(9)；8m@100k 80(3) vs MAPPO 45(7)；3s_vs_4z@100k 26(4) vs 全部基线 0；corridor@500k 39(10) vs 0；3s5z_vs_3s6z@500k 44(25)；反例 bane_vs_bane@100k 39(20) vs MAPPO 93(2)；27m_vs_30m@500k 仅 1(1) — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- Flatland：5 智能体在 300k 样本达到良好表现；10 智能体时优势缩小；15 智能体时样本效率低于 1RL 基线，但 1.5M 样本时表现相当，作者归因于离散潜空间更"噪"、需更强世界模型 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)

**MA-Dreamer — Lobos-Tsunekawa et al., 2022【预印本，arXiv 2204.04687】**
- 仅二手信息：被描述为"在智能体之间引入共享想象，通过潜在 rollout 进行通信" — [Dreaming of Others, arXiv 2605.31361](https://arxiv.org/pdf/2605.31361)

**MABL — "MABL: Bi-Level Latent-Variable World Model for Sample-Efficient Multi-Agent Reinforcement Learning"，Venugopal, Milani, Fang, Ravindran，AAMAS 2024【同行评审，arXiv 2304.06011】**
- 潜在状态：上层"全局潜变量"编码全局状态与他人观测，仅训练期使用；下层"智能体潜变量"由自身观测和 agent embedding 计算并以全局潜变量为条件，训练与执行都用 — [arXiv HTML](https://arxiv.org/html/2304.06011v2)；[AAMAS 2024 PDF](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1865.pdf)
- 执行：去中心化，"智能体独立行动"，执行期无需通信 — [arXiv HTML](https://arxiv.org/html/2304.06011v2)
- 基准与结果（3 次运行 mean±std）：SMAC 3s_vs_4z 83±18 vs MAG 30±32；3s_vs_5z 31±30 vs MAG 5±13；Flatland 5 智能体 51±27 vs MAG 19±19；MAMuJoCo Humanoid 550±90 vs MAPPO 429±36；HumanoidStandup 106k±29k vs MAMBA 83k±30k；corridor 52±31 与 MAG 55±27 持平；2s_vs_1sc 92±7 落后 MAPPO 98±3 — [arXiv HTML](https://arxiv.org/html/2304.06011v2)
- 不确定性：随机潜变量 + ELBO/KL，未用于决策或融合 — [arXiv HTML](https://arxiv.org/html/2304.06011v2)

**MARIE — "Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models"，Zhang, Bai, Zhao, Yan, Li, Li【arXiv 2406.15836；搜索摘要称 v2 标注 TMLR 录用，且 ICLR 2025 虚拟站点有同名条目；我在 v2 HTML 中只读到"Reviewed on OpenReview"，未直接核实 TMLR 字样】**
- 潜在状态：共享 VQ-VAE 把每个智能体的观测离散为 K=16 个 token（码本 512）；共享 Transformer 对每个智能体的局部动力学建模（去中心化），Perceiver 把所有智能体的 token 与动作聚合成每个智能体一个全局特征，插入其局部序列（集中式聚合） — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)；[ICLR 2025 条目](https://iclr.cc/virtual/2025/37527)
- 执行：策略只用 VQ-VAE 重建的局部观测，执行不需要世界模型；未读到执行期智能体间通信 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- 结果（SMAC 13 张图，4 个种子，Table 1 均值胜率%）：3s_vs_5z 78.4 vs MAMBA 13.4；corridor 71.0 vs 26.5；3s_vs_4z 73.0 vs 29.3；2c_vs_64zg 25.9 vs 9.8；作者称在 >3 智能体的所有地图上最强 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- SMACv2（3 张图）：zerg_5_vs_5 与 MAMBA 相当，另两张图优于全部基线（正文无数值）；MAMuJoCo 3 个场景中 2 个最佳，HalfCheetah-v2-3x2 上 MAMBA 更好 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- 智能体数量扩展消融（2–7 个智能体）：集中式联合动力学变体在 >3 智能体后样本效率急剧下降，去中心化的 MARIE 保持稳定；去掉聚合模块的损失随智能体数增加而增大 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- 计算：单张 RTX 3090（36 核 CPU、128GB 内存），100K 环境步约 1 天、200K 约 2–3 天、400K 约 4 天；显存 22GB；参数约 55.35M（MAMBA 约 23.72M）；>5 智能体时想象步长降为 8 以适应单卡 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- 局限：自回归生成在长视野下推理慢；仅 4 个种子；仅低数据量设定 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)

**CoDreamer — "CoDreamer: Communication-Based Decentralised World Models"，Toledo & Prorok, 2024【预印本，arXiv 2406.13600】**
- 潜在状态：每个智能体一个 DreamerV3 式 RSSM（离散随机后验 + 循环状态），这些状态被表示为图的节点 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 两级通信：(1) 世界模型内部——在 RSSM 以及奖励/终止预测头中加入 GNN 层（解码器保持独立）；(2) 策略内部——actor/critic 各有 GNN torso。GNN 为 GATv2；在通信半径内按欧氏距离建边，边特征为相对距离 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 消息维度/大小：未给出 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 想象：遵循 CTDE；想象中沿用起始时刻的邻接矩阵，不重新预测；作者把"边特征可能过时"列为局限/未来工作 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 不确定性：只有随机离散潜变量，无显式不确定性估计或加权 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 基准与结果（归一化 IQM，95% CI）：Melting Pot（Daycare、Cooperative Mining、Collaborative Cooking 两种）CoDreamer 0.48 [0.45,0.50] vs IDreamer 0.30 [0.17,0.44] vs IPPO 0.05；VMAS（Flocking、Discovery、Buzz Wire）0.84 vs 0.81 vs 0.79，作者承认 VMAS 提升边际且 CI 重叠；消融显示 Melting Pot 增益主要来自世界模型内通信 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 计算：140 次运行（5 算法 × 4 种子 × 7 任务）每次约 9 小时，合计约 1,260 TPU v2/v3 小时，约 181 kgCO₂eq；每次 500k 环境步；世界模型约 16M（向量）/22M（视觉）参数 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- 鲁棒性：未做消息丢失、延迟或智能体数量变化实验，每个任务智能体数固定 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)

**GAWM — "GAWM: Global-Aware World Model for Multi-Agent Reinforcement Learning"，Shi, Liu, Zhang, Zheng, Dong, Wei, 2025【预印本，arXiv 2501.10116】**
- 潜在状态：每个智能体一个 z_t^i，外加一个共享的循环确定性状态 h_t；Transformer 的 obs-fusion 模块对所有智能体的观测 o_t^i 与历史做跨智能体融合后送入后验；另有 act-fusion Transformer；奖励/折扣由全局 h_t, z_t 预测 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- 执行：去中心化，π^i(a^i|o^i) 仅用本地观测 + GRU；未描述执行期通信（世界模型只在训练中用） — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- 一致性度量：提出 Global Consistency Index（各智能体预测的全局状态与跨智能体均值之间的 L2 距离 + 奖励/折扣偏离阈值的指示惩罚）与 Global Prediction Error；GCI：GAWM 0.63–1.29，MAG 2.43–3.48，MAMBA 2.91–4.14 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- 结果（SMAC，3 种子，胜率% (std)）：2s_vs_1sc@15k 93(3) vs MAMBA 64(15)；3s_vs_5z@300k 93(2) vs 53(8)；8m@40k 90(2) vs 37(7)；corridor@400k 86(3) vs 39(9)；MAPPO/QMIX 在这些低样本预算下为 0–38 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- 不确定性：仅 RSSM 式随机潜变量 + KL；无集成/显式不确定性 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- 计算：仅说额外 Transformer "略微增加"世界模型每次迭代训练时间，无具体数字 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)

**MATWM — "Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning"，Deihim, Alonso, Apostolopoulou, 2025【预印本，arXiv 2506.18537；HTML 显示 Elsevier "Artificial Intelligence" 期刊字样，录用未确认】**
- 潜在状态：每个智能体观测编码为 32×32 类别随机潜变量；actor-critic 状态 = 自身潜变量 + 隐状态 + 队友动作 logits 预测 — [arXiv HTML](https://arxiv.org/html/2506.18537v1)
- 想象：去中心化——世界模型参数共享，但只用焦点智能体自己的信息计算其轨迹；队友行为由"teammate predictor"从焦点智能体潜变量历史推断；作者称训练不需要集中信息或通信 — [arXiv HTML](https://arxiv.org/html/2506.18537v1)
- 结果：SMAC 12 张图（9 张 easy@50K，3 张 hard@200K）平均胜率 MATWM 61.1% vs MARIE 57.1% vs MAMBA 45.4% vs MAPPO 17.1%；PettingZoo Pistonball（20 智能体）92.6；Melting Pot 7 个环境全部领先；去掉 teammate predictor 后 8m 与 so_many_baneling 降为 0.0 — [arXiv HTML](https://arxiv.org/html/2506.18537v1)
- 局限：去中心化世界模型受非平稳性影响，世界模型跟不上快速变化的策略时性能会下滑；无计算开销测量 — [arXiv HTML](https://arxiv.org/html/2506.18537v1)

**DIMA — "Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective"，Zhang, Li, Ye, Qiu, Qu, Li, Zhang, Bai，NeurIPS 2025【arXiv 页面写 "Accepted at NIPS'25"，arXiv 2505.20922】**
- 建模：不直接建模联合状态-动作转移，而是每步只在状态空间上"逐个揭示"各智能体的影响，类比扩散模型的反向过程；基准 MAMuJoCo 与 Bi-DexHands；称达到 SOTA；代码 github.com/breez3young/DIMA — [arXiv abs](https://arxiv.org/abs/2505.20922)

**MMSA — "Multi-Agent Model-Based Reinforcement Learning with Joint State-Action Learned Embeddings"，Wang & Meger, 2026【arXiv 2602.12520；搜索结果指向 AAMAS 2026 论文集 PDF，arXiv 页面本身未写会议】**
- 建模：VAE 世界模型 + 状态-动作学习嵌入（SALE），SALE 同时用于想象模块与带混合网络的联合智能体网络；基准 SMAC、MAMuJoCo、Level-Based Foraging；摘要无数值；SALE 是逐智能体还是全局未读到 — [arXiv abs](https://arxiv.org/abs/2602.12520)；[AAMAS 2026 PDF（搜索结果）](https://ifaamas.org/Proceedings/aamas2026/pdfs/IXMJ8234.pdf)

**MWM-MARL — "From Pixels to Cooperation: Multi Agent Reinforcement Learning based on Multimodal World Models"【预印本，arXiv 2511.01310；仅读到搜索摘要】**
- 先训练共享的生成式多模态世界模型（基于 RSSM），再训练去中心化 actor，把表征学习与策略学习解耦 — [arXiv PDF](https://arxiv.org/pdf/2511.01310)

**DMAWM — "Learning Disentangled Multi-Agent World Model for Decentralized Control"，Xue, Jiang, Zhang, Guo, Yuan, Zhang, Yu，ICML 2026 poster【同行评审】**
- 动机：在潜空间建模联合动力学会在智能体潜状态间产生"虚假相关"；方法：独立的 agent 模块 + 共享的 environment 模块；真实执行中每个 agent 模块只处理本地信息形成分解（factorized）潜表示，environment 模块被训练成"镜像"这一分解结构，使想象更贴近去中心化执行 — [ICML 2026 poster 页](https://icml.cc/virtual/2026/poster/61713)
- 执行期：页面的通俗摘要称完全去中心化，无通信、无实时访问队友内部模型 — [ICML 2026 poster 页](https://icml.cc/virtual/2026/poster/61713)
- 基准：SMAC、SMACv2、基于视觉的 Melting Pot；页面只声称收敛速度与最终性能优于 model-based/model-free 方法，无数值；OpenReview 链接 https://openreview.net/forum?id=nYyfpPubnW（抓取被浏览器验证拦截） — [ICML 2026 poster 页](https://icml.cc/virtual/2026/poster/61713)

**DCWM — "Decentralized Collective World Model for Emergent Communication and Coordination"，Nomura, Aoki, Taniguchi, Horii【预印本，arXiv 2504.03353 v3, 2025-10】**
- 潜在状态：每个智能体独立初始化的 RSSM，z 为 4×4 unimix 类别，h 为 32 维 GRU；各自最小化分布式自由能 — [arXiv HTML v3](https://arxiv.org/html/2504.03353v3)
- 通信：每步一个连续消息 m_t（2 维高斯），交换的是采样值而非分布参数，滑动窗口 W=10；InfoNCE 对比损失对齐跨智能体消息；消息先验用"另一智能体后验"的 Product-of-Experts 近似（集体预测编码） — [arXiv HTML v3](https://arxiv.org/html/2504.03353v3)
- 实验：2 智能体点轨迹追踪（各观测一个轴），行为克隆策略，3 个种子；无消息丢失实验；更大群体、RL、高维任务列为未来工作 — [arXiv HTML v3](https://arxiv.org/html/2504.03353v3)

**Dreamer-CPC — "Dreamer-CPC: Message Learning with World Models for Decentralized Multi-agent Reinforcement Learning"，Takayama, Yoshida, Taniguchi【预印本，arXiv 2607.19809，注明投稿 ICONIP 2026】**
- 每个智能体独立维护世界模型与消息模块，用集体预测编码（Collective Predictive Coding）学消息；环境 Observer 与自建 CatchApple（任务相关观测会暂时缺失）；CatchApple 上回报为 IPPO-CPC 的 4–5 倍；摘要未给消息大小与丢包实验 — [arXiv abs](https://arxiv.org/abs/2607.19809)

**Dreaming of Others — "Dreaming Of Others: Latent Teammate Modeling In World Models For Multi-Agent Reinforcement Learning"，Tomas Leroy-Stone【arXiv 2605.31361，WMW 2026 Workshop poster，立场论文，无实验结果】**
- 潜在状态：单智能体 RSSM，z_t = [z_t^env, z_t^team]；观测解码器只用 z^env，队友策略解码器从 z^team 预测队友下一动作；ToM 损失 = 交叉熵 + α·KL 时间一致性项 — [arXiv PDF](https://arxiv.org/pdf/2605.31361)
- 明确不需要共享想象、集中式策略或显式通信；提议用 MPE、Overcooked-AI、Melting Pot（经 JaxMARL/BenchMARL）评估零样本/少样本协作 — [arXiv PDF](https://arxiv.org/pdf/2605.31361)
- 作者对前人的定性：MA-Dreamer/CoDreamer/GAWM/MACD 等常假设共享潜状态访问或集中式想象，实际上回避了部分可观测性 — [arXiv PDF](https://arxiv.org/pdf/2605.31361)

#### B. 具身/生成式多智能体世界模型

**COMBO — "COMBO: Compositional World Models for Embodied Multi-Agent Cooperation"，Zhang et al.，ICLR 2025【同行评审，arXiv 2404.10775】**
- 世界状态：把各智能体的第一人称 RGBD 视图转成部分点云并合并为俯视正交图（按时间叠加，噪声大且不完整），再用扩散模型 inpaint 成估计的世界状态 — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)
- 组合式世界模型：联合动作按智能体拆成文本分量，每个分量一个条件视频扩散 score，求和组合（Eq. 5、7）；可从 4 智能体数据迁移到 3/2 智能体 — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)；[ICLR 2025 论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7d03c6bf9f07acb4038eea96c63db52d-Abstract-Conference.html)
- 规划：形式化为 Dec-POMDP，Algorithm 1 为"agent i 的规划过程"，每步重规划；VLM 意图跟踪器预测他人动作（2 智能体 75.8%、3 智能体 64.6% 正确）+ 树搜索；论文未描述通信协议或执行期是否共享原始观测 — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)
- 结果：TDW-Game 成功率 1.00（平均 17.5/17.4 步）vs CoELA 0.90/0.60；TDW-Cook 0.90/1.00 vs CoELA 0.15/0.05；MAPPO 与 Recurrent World Models 在两个 TDW 任务均为 0%；2D-FetchQ 81.3% vs Co-GAIL 53.3% — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)
- 计算：世界模型阶段一 192×V100 约 1 天，阶段二 120×V100 约 1 天；每个 VLM 子模块 LLaVA-1.5-7B LoRA 在 18×V100 上约 3 小时；推理因多次大模型调用而慢 — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)

**Solaris — "Solaris: Building a Multiplayer Video World Model in Minecraft"，Savva/Michel et al.（NYU, Saining Xie），NeurIPS 2026 poster【同行评审，arXiv 2602.22208】**
- 表示：单一联合模型；每个玩家的潜帧组成联合张量，在 DiT 中沿序列维交错，共享"Multiplayer Self-Attention"跨玩家交换信息，加玩家 ID embedding 与逐玩家 3D RoPE — [arXiv HTML v2](https://arxiv.org/html/2602.22208v2)
- 玩家数：arXiv v2 正文训练与评估 P=2 — [arXiv HTML v2](https://arxiv.org/html/2602.22208v2)；NeurIPS 摘要称训练了 3 玩家概念验证模型，并可在其硬件上"模拟最多 16 名玩家" — [NeurIPS 2026 poster 页](https://neurips.cc/virtual/2026/poster/148491)（两版本表述不一致；16 是模拟能力而非训练规模）
- 数据与结果：12.64M 帧（每玩家 6.32M，9,240 段）；VLM-judge/FID：Movement 68.2/38.5、Consistency 71.4/99.4、Building 20.8/83.6；计算 TPU v5p-128/v5p-64，无总时数 — [arXiv HTML v2](https://arxiv.org/html/2602.22208v2)
- 局限："缺乏持久记忆"，玩家离开彼此视野后共享上下文丢失、轨迹开始分歧；世界没有底层持久状态，仅由两张初始帧指定 — [arXiv HTML v2](https://arxiv.org/html/2602.22208v2)

**Gamma-World (γ-World) — "Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players"，Liu et al.（NVIDIA/清华/多伦多/Vector），2026-05【预印本，arXiv 2605.28816】**
- 表示：每个智能体一条独立的潜视频 token 流（张量带显式 agent 轴 P×T×H×W×Cz）；身份由无参数的 Simplex Rotary Agent Encoding（智能体位于旋转角空间正单纯形顶点，置换等价）给出；动作经共享编码器作为逐块偏置 — [arXiv HTML](https://arxiv.org/html/2605.28816v1)；[arXiv abs](https://arxiv.org/abs/2605.28816)
- 跨智能体：Sparse Hub Attention，默认每个潜帧 K=8 个可学习 hub token，跨智能体代价从二次降为线性；K=1/8/32/128 时 FVD 250.9/223.4/221.8/220.5 — [arXiv HTML](https://arxiv.org/html/2605.28816v1)
- 集中式：一个模型输入所有智能体的历史观测与动作，联合预测每个智能体的下一观测 — [arXiv HTML](https://arxiv.org/html/2605.28816v1)
- 数据与结果：同步多智能体 Minecraft 轨迹（2 智能体为主，4 智能体扩展）+ 双臂机器人数据集（抽取文本中拼写为 "RealOmin-Open"，左右臂视为两个智能体）；完整模型 FVD 223.4、FID 30.2、LPIPS 0.269、PSNR 27.7、SSIM 0.836；对 Solaris：Consistency 类 FVD 280.0 vs 443.1；2→4 玩家无需额外训练但只有定性结果 — [arXiv HTML](https://arxiv.org/html/2605.28816v1)
- 计算/速度：基座 Cosmos-Predict2.5-2B，教师与因果学生各在 32×GB200 上训练，无总 GPU 时；蒸馏后 KV-cache 因果学生达 24 FPS — [arXiv HTML](https://arxiv.org/html/2605.28816v1)；[项目页](https://research.nvidia.com/labs/sil/projects/gamma-world/)
- 局限：超大群体可能需要更大的旋转频带或层级分组；未显式约束 3D 几何/物理，长 rollout 会累积不一致 — [arXiv HTML](https://arxiv.org/html/2605.28816v1)

#### C. 汇总表（执行期通信与潜状态）

| 方法 | 潜状态粒度 | 想象/rollout | 执行期通信 & 大小 | 显式不确定性用于融合？ | 来源 |
|---|---|---|---|---|---|
| MAMBA (AAMAS'22) | 每智能体 RSSM，32×32 类别 | 每智能体模型 + Transformer 通信块 | 广播 z_{t-1}（160 bit）+ 观察他人 a_{t-1} | 否 | [PDF](https://arxiv.org/pdf/2205.15023) |
| MABL (AAMAS'24) | 全局潜变量（仅训练）+ 智能体潜变量 | 训练期用全局 | 无 | 否 | [HTML](https://arxiv.org/html/2304.06011v2) |
| MARIE (TMLR?/ICLR'25 列表) | 每智能体 16 个 VQ token | 局部 Transformer + Perceiver 集中聚合 | 无（策略不需世界模型） | 否 | [HTML](https://arxiv.org/html/2406.15836v2) |
| CoDreamer (预印本) | 每智能体 RSSM（图节点） | CTDE，GNN 消息，邻接固定 | GATv2 消息，大小未报告 | 否 | [HTML](https://arxiv.org/html/2406.13600v1) |
| GAWM (预印本) | 每智能体 z + 共享 h | 集中式 Transformer 融合 | 无 | 否（但提出 GCI 一致性指标） | [HTML](https://arxiv.org/html/2501.10116v1) |
| MATWM (预印本) | 每智能体 32×32 类别 | 去中心化 + 队友预测器 | 未描述/称不需要 | 否 | [HTML](https://arxiv.org/html/2506.18537v1) |
| DMAWM (ICML'26) | 分解的每智能体潜变量 + 共享环境模块 | 环境模块镜像分解结构 | 无 | 未读到 | [poster](https://icml.cc/virtual/2026/poster/61713) |
| DCWM (预印本) | 每智能体 RSSM（4×4 类别 + 32 维 GRU） | 去中心化 | 2 维高斯消息采样值 | 部分：PoE 消息先验 | [HTML](https://arxiv.org/html/2504.03353v3) |
| COMBO (ICLR'25) | 合并点云的俯视图像 | 每智能体规划，组合式视频扩散 | 未描述 | 否 | [HTML](https://arxiv.org/html/2404.10775v2) |
| Solaris (NeurIPS'26) | 每玩家潜帧 | 单一联合 DiT | 不适用（模拟器） | 否 | [HTML](https://arxiv.org/html/2602.22208v2) |
| Gamma-World (预印本) | 每智能体潜视频 token | 单一联合模型 + hub token | 不适用（模拟器） | 否 | [HTML](https://arxiv.org/html/2605.28816v1) |

### Inferences
- "每个智能体一份潜状态 + 某种跨智能体聚合"是绝对主流；聚合算子依次是注意力（MAMBA）、GNN（CoDreamer）、Transformer 融合（GAWM）、Perceiver（MARIE）、hub token（Gamma-World）、全联合注意力（Solaris）。这些都是学习得到的单次前向聚合，没有一篇把聚合写成"带精度的能量最小化/置信传播"。
- 与目标方法最近的先行工作是 MAMBA：它已经实现"执行期用定长离散消息（160 bit）维持每个智能体的世界模型"，并可局部化到邻居。因此"去中心化 + 定长消息"本身不是新点；新点必须落在"精度加权的一致性（GBP/sheaf）+ 对丢包/延迟的鲁棒性 + 一致性保证/度量"上。
- DCWM/Dreamer-CPC（Taniguchi 组的集体预测编码）在概率上最接近（PoE 消息先验、高斯消息），但只在 2 智能体玩具任务或小环境上验证，未与 SMAC 系 MBRL 方法对比。
- 名称易混：MATWM ≠ MARIE；DMAWM（ICML 2026 解耦多智能体世界模型）≠ DMWM（NeurIPS 2025 单智能体 Dual-Mind 世界模型，[NeurIPS 条目](https://nips.cc/virtual/2025/poster/119328)）。
- 跨论文的同名基线数值不可直接比较（例如 MAMBA 在 corridor 上自报 39(10)@500k，而 MARIE 表中 MAMBA 为 26.5），因为样本预算与种子不同（推断）。

### Gaps
- MAMBA 消息"160 bit"只计 z，不含需要观察的他人动作 a_{t-1} 的比特数；论文未给总带宽。
- CoDreamer 消息维度、MATWM 执行期是否交换任何信息、COMBO 执行期是否共享观测（合并点云是否需要传输）均未在原文中读到。
- DMAWM 的完整论文、数值结果与代码未能获取（OpenReview 被浏览器验证拦截，ICML 页无表格）。
- MACD（Chai et al., AAMAS 2024，反事实想象做信用分配）、MAG、MA-Dreamer 只读到二手描述，未核实其潜状态与通信细节。
- MMSA 与 MWM-MARL 只读到摘要级信息，无数值。
- MARIE 是否为 TMLR 正式录用：搜索摘要与 ICLR 2025 列表支持，但我未在论文页直接读到 "TMLR" 字样。
- 未检索到 Multiverse（被 Solaris 用作"frame concatenation"基线来源）的一手资料。

## Q2. 哪些方法在执行期真正去中心化（仅本地观测 + 消息），哪些依赖集中训练/想象或共享潜状态？

### Takeaway
几乎所有 MBRL 方法都遵循 CTDE：世界模型（含跨智能体聚合）只在训练/想象中使用，执行期策略仅用本地观测且无通信（MABL、MARIE、GAWM、MATWM、DMAWM）。在执行期"用消息维持世界模型"的只有 MAMBA、CoDreamer、DCWM、Dreamer-CPC；生成式视频世界模型（Solaris、Gamma-World）是完全集中的联合模拟器。没有任何一篇在执行期做多轮迭代共识。

### Cited Findings
- MAMBA 明确把"智能体基于本地观测和他人消息独立决策"列为目标，执行与想象中只传随机状态 z — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- MAMBA 不使用训练期额外全局信息（critic 仅用所有预测潜状态），作者把这称为 model-based 方法的局限 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- MABL：全局潜变量只在训练期使用，执行期无需通信；作者指出 CTCE 方法需要持续传输，而 MABL 避免了这一点 — [arXiv HTML](https://arxiv.org/html/2304.06011v2)
- MARIE：Perceiver 在想象的每一时刻聚合所有智能体信息（集中式聚合），而部署的策略只用重建的局部观测、不需要世界模型 — [arXiv HTML v2](https://arxiv.org/html/2406.15836v2)
- GAWM：世界模型的 obs-fusion 需要所有智能体观测（集中），执行策略只用本地观测 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- MATWM：世界模型只用焦点智能体自身信息，用队友预测器替代他人信息，训练不需集中信息或通信 — [arXiv HTML](https://arxiv.org/html/2506.18537v1)
- DMAWM：执行期完全去中心化、无通信；用环境模块镜像去中心化执行时的分解潜结构 — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61713)
- CoDreamer：遵循 CTDE，训练用所有智能体经验的图；执行期世界模型可由智能体簇独立使用，通过通信半径内的 GNN 消息交换 — [arXiv HTML](https://arxiv.org/html/2406.13600v1)
- DCWM：每个智能体独立参数的世界模型，执行中交换 2 维高斯消息采样 — [arXiv HTML v3](https://arxiv.org/html/2504.03353v3)；Dreamer-CPC：每个智能体独立维护世界模型与消息模块 — [arXiv abs](https://arxiv.org/abs/2607.19809)
- Dreaming of Others 对领域的判断：已有工作常假设共享潜状态访问或集中式想象 — [arXiv PDF](https://arxiv.org/pdf/2605.31361)
- COMBO 的 Algorithm 1 是每个智能体自己的规划流程（去中心化规划），但世界状态由所有智能体视图合并的点云生成 — [arXiv HTML v2](https://arxiv.org/html/2404.10775v2)
- Solaris 与 Gamma-World 都是一个模型联合生成全部智能体的观测 — [Solaris HTML](https://arxiv.org/html/2602.22208v2)；[Gamma-World HTML](https://arxiv.org/html/2605.28816v1)

### Inferences
- 文献里的"去中心化世界模型"至少有三种不同含义：(a) 执行期不用世界模型、无通信（CTDE，MABL/MARIE/GAWM/MATWM/DMAWM）；(b) 每个智能体有自己的世界模型，执行期靠单步消息更新（MAMBA/CoDreamer/DCWM/Dreamer-CPC）；(c) 想象本身是逐智能体的（MATWM/DMAWM 的动机）。目标方法属于 (b)，且将是 (b) 中唯一显式做"迭代共识 + 精度"的；写作时需明确区分这三种含义，避免与 (a) 类比较时被认为不公平。
- 各方法的聚合是一次前向传播（单轮），而 GBP 是迭代消息传递；单轮聚合在多跳通信图上无法得到全局一致解，这是一个可检验的差异点（推断，未见论文直接比较）。

### Gaps
- 没有找到在执行期做多轮/异步消息传递来对齐潜状态的多智能体世界模型。
- COMBO、MATWM 在执行期到底传输什么，原文未写清。

## Q3. 是否有工作用显式不确定性加权（Kalman/GBP/PoE 式）把多个智能体的部分观测融合成一致的共享潜状态，或给出智能体间一致性保证？

### Takeaway
在多智能体世界模型里，没有找到用精度/协方差加权（Kalman/GBP 式）融合潜状态、或给出跨智能体一致性保证的工作。最接近的是 DCWM 的 PoE 消息先验（2 智能体玩具任务）和 GAWM 的经验一致性指标 GCI；精度加权融合与共识保证只出现在非世界模型文献中（Robot Web 的 GBP、CroMAC 的 PoE、T2MAC 的证据融合、SIGMA 的 sheaf 共识）。

### Cited Findings
- MAMBA 承认独立预测会造成智能体间视图不一致，解决方法是 L_info 解耦而非融合 — [arXiv PDF](https://arxiv.org/pdf/2205.15023)
- GAWM 用 Transformer 融合并定义 Global Consistency Index（各智能体预测全局状态偏离均值的 L2 + 奖励/折扣偏离惩罚）衡量一致性，属于经验指标而非保证；GCI 在作者自己的离线数据对上评估 — [arXiv HTML](https://arxiv.org/html/2501.10116v1)
- CoDreamer、MATWM、MABL、GAWM 的不确定性都只体现在随机潜变量/KL 项，未用于融合加权 — [CoDreamer](https://arxiv.org/html/2406.13600v1)；[MATWM](https://arxiv.org/html/2506.18537v1)；[MABL](https://arxiv.org/html/2304.06011v2)；[GAWM](https://arxiv.org/html/2501.10116v1)
- DCWM：消息先验近似为另一智能体的后验（Product-of-Experts），并有集体正则项；但交换的是采样值而非分布参数，且未显式用不确定性做决策 — [arXiv HTML v3](https://arxiv.org/html/2504.03353v3)
- 非世界模型对照 1（GBP）：Robot Web 用 GBP 做多机器人分布式定位，最多 1000 个机器人的模拟中收敛到与集中式非线性因子图求解器相当的精度；完美通信下集中式与分布式 GBP 等价；鲁棒因子使其能容忍高比例错误测量或丢包；后发表于 IEEE T-RO 2024 — [arXiv 2202.03314](https://arxiv.org/pdf/2202.03314)；[Imperial Spiral 条目](https://spiral.imperial.ac.uk/entities/publication/ac087e1d-6aff-4dc0-b17d-908540e034c1)
- GBP 也被用于多机器人协同规划（动力学与碰撞约束因子图） — [arXiv 2203.11618](https://arxiv.org/pdf/2203.11618)
- 非世界模型对照 2（sheaf）：SIGMA（ICRA 2025）把 sheaf 理论用于去中心化深度 RL 的多智能体路径规划，用自监督训练网络在潜空间近似共识；stalk 收集智能体数据、restriction map 描述智能体关系；作者称在大规模、多智能体场景中优势最大 — [arXiv 2502.06440](https://arxiv.org/html/2502.06440v2)
- 非世界模型对照 3（PoE/证据融合，仅搜索摘要）：CroMAC 把每条消息视为状态的一个"视图"，用 PoE 多视图 VAE 得到联合消息表示以抵抗扰动消息 — [SCIS 2024](http://scis.scichina.com/en/2024/142102.pdf)；T2MAC 用 Dempster 组合规则融合消息，低不确定性视角主导结果 — [arXiv 2401.10973](https://arxiv.org/pdf/2401.10973)
- 非世界模型对照 4：KL 信念门控（单作者预印本，2026）让每个智能体维护 K=8 的 softmax "信念"，KL 超阈值才通信；但信念头无独立监督、消息为 128 维 LSTM 隐状态的门控平均，融合不携带不确定性，且不预测动力学（不是世界模型）；MPE 上门控并未选择性触发（通信率恒为 1.0） — [arXiv 2608.14559](https://arxiv.org/html/2608.14559)

### Inferences
- 目标方法的最强新颖点：在"学习到的潜在世界状态"上做"带精度的 GBP 共识"，把 Robot Web 一类几何状态估计的性质（完美通信下等价于集中解、对丢包鲁棒）迁移到学习型多智能体世界模型。现有文献中这两条线（MBRL 世界模型 vs GBP/sheaf 共识）没有交汇。
- GAWM 的 GCI 可以直接作为跨智能体一致性的现成评估指标（与 GBP 后的 disagreement / sheaf 能量对比），有利于和 GAWM/MAMBA 正面比较。
- 风险：SIGMA 已把"sheaf + 去中心化 RL + 潜空间共识"用于 MAPF，审稿人可能据此质疑 sheaf 部分的新颖性；需强调 SIGMA 的共识由网络近似、无精度加权，且不是世界模型（推断，基于 SIGMA 摘要层信息）。
- 风险：DCWM 已经把 PoE（高斯精度相乘的本质）用于消息先验；目标方法需说明与 PoE 单步融合的区别——多跳迭代、sheaf 限制映射、对丢包/延迟的处理（推断）。

### Gaps
- 未找到任何多智能体世界模型给出跨智能体潜状态一致性的理论保证。
- CroMAC 与 T2MAC 的细节只来自搜索摘要，未读全文。
- 未读协同感知（V2X）中带不确定性的 BEV 特征融合全文；搜索只显示该领域以中间特征融合 + 注意力为主，"共享潜在世界模型"仍是新兴方向，MAEAD 综述把共享世界模型作为方向提出 — [MAEAD 综述](https://arxiv.org/abs/2606.13840)

## Q4. 这些论文使用哪些基准、指标，运行成本多高？对丢包/延迟/部分可观测/智能体数的鲁棒性测过吗？

### Takeaway
MBRL 线以 SMAC（低样本预算胜率）为绝对核心，辅以 SMACv2、MAMuJoCo、Flatland、Melting Pot、VMAS、PettingZoo、LBF；具身/生成线用 ThreeDWorld（TDW）与 Minecraft。单次 MBRL 运行约需 1–4 个 GPU 日（MARIE 单 3090）或约 9 TPU 小时（CoDreamer），生成式视频世界模型需要数十至近 200 张加速卡。没有任何一篇多智能体世界模型论文测试消息丢失或延迟；智能体数量扩展只在少数论文里做到 ≤20。

### Cited Findings
- 基准使用：
  - SMAC：MAMBA（23 张图，正文 8 张）— [PDF](https://arxiv.org/pdf/2205.15023)；MABL — [HTML](https://arxiv.org/html/2304.06011v2)；MARIE（13 张）— [HTML](https://arxiv.org/html/2406.15836v2)；GAWM（8 张）— [HTML](https://arxiv.org/html/2501.10116v1)；MATWM（12 张）— [HTML](https://arxiv.org/html/2506.18537v1)；MMSA — [abs](https://arxiv.org/abs/2602.12520)；DMAWM — [poster](https://icml.cc/virtual/2026/poster/61713)
  - SMACv2：MARIE（3 张，附录）— [HTML](https://arxiv.org/html/2406.15836v2)；DMAWM — [poster](https://icml.cc/virtual/2026/poster/61713)；MARIE 作者指出 SMACv1 有严重缺陷是使用 SMACv2 的动机 — [HTML](https://arxiv.org/html/2406.15836v2)
  - MAMuJoCo：MABL、MARIE、DIMA（+Bi-DexHands）、MMSA — [MABL](https://arxiv.org/html/2304.06011v2)；[MARIE](https://arxiv.org/html/2406.15836v2)；[DIMA](https://arxiv.org/abs/2505.20922)；[MMSA](https://arxiv.org/abs/2602.12520)
  - Flatland（5/10/15 智能体）：MAMBA、MABL（5/10）— [MAMBA](https://arxiv.org/pdf/2205.15023)；[MABL](https://arxiv.org/html/2304.06011v2)
  - Melting Pot：CoDreamer、MATWM（7 个环境）、DMAWM（视觉版）；VMAS：CoDreamer；PettingZoo：MATWM；LBF：MMSA — [CoDreamer](https://arxiv.org/html/2406.13600v1)；[MATWM](https://arxiv.org/html/2506.18537v1)；[DMAWM](https://icml.cc/virtual/2026/poster/61713)；[MMSA](https://arxiv.org/abs/2602.12520)
  - MPE / Overcooked-AI：仅在 Dreaming of Others 中作为提议的评估协议（无结果）— [PDF](https://arxiv.org/pdf/2605.31361)；KL 信念门控用 MPE simple_spread 与 Predator-Prey — [HTML](https://arxiv.org/html/2608.14559)
  - ThreeDWorld（TDW-Game、TDW-Cook）+ 2D-FetchQ + 真实人机试验：COMBO — [HTML](https://arxiv.org/html/2404.10775v2)
  - Minecraft 多人：Solaris、Gamma-World；双臂机器人数据：Gamma-World — [Solaris](https://arxiv.org/html/2602.22208v2)；[Gamma-World](https://arxiv.org/html/2605.28816v1)
- 指标：SMAC 胜率（低样本预算 15k–500k 步）；CoDreamer 用归一化 IQM 与 95% 置信区间；COMBO 用 30 步视野内成功率与成功 episode 的平均步数（每设置 20 个 episode）；Gamma-World 用 FVD/FID/LPIPS/PSNR/SSIM；Solaris 用 FID + VLM-as-judge；GAWM 自定义 GCI/GPE — [CoDreamer](https://arxiv.org/html/2406.13600v1)；[COMBO](https://arxiv.org/html/2404.10775v2)；[Gamma-World](https://arxiv.org/html/2605.28816v1)；[Solaris](https://arxiv.org/html/2602.22208v2)；[GAWM](https://arxiv.org/html/2501.10116v1)
- 种子数普遍很少：MAMBA 3、MABL 3、GAWM 3、MARIE 4（作者自承需更多种子）、CoDreamer 4、DCWM 3、MATWM 报告 4 种子中位数 — [MAMBA](https://arxiv.org/pdf/2205.15023)；[MABL](https://arxiv.org/html/2304.06011v2)；[GAWM](https://arxiv.org/html/2501.10116v1)；[MARIE](https://arxiv.org/html/2406.15836v2)；[CoDreamer](https://arxiv.org/html/2406.13600v1)；[DCWM](https://arxiv.org/html/2504.03353v3)；[MATWM](https://arxiv.org/html/2506.18537v1)
- 成本：
  - MARIE：单 RTX 3090，100K 步约 1 天、400K 步约 4 天，显存 22GB — [HTML](https://arxiv.org/html/2406.15836v2)
  - CoDreamer：每次运行约 9 小时，140 次合计约 1,260 TPU v2/v3 小时 — [HTML](https://arxiv.org/html/2406.13600v1)
  - MAMBA：作者称每条采集轨迹比 model-free 需要更多计算时间，墙钟对比在附录 — [PDF](https://arxiv.org/pdf/2205.15023)
  - COMBO：192×V100 约 1 天 + 120×V100 约 1 天（世界模型）+ VLM 微调 — [HTML](https://arxiv.org/html/2404.10775v2)
  - Gamma-World：教师、学生各在 32×GB200 上训练，蒸馏 400 iter — [HTML](https://arxiv.org/html/2605.28816v1)；Solaris：TPU v5p-128 / v5p-64 — [HTML](https://arxiv.org/html/2602.22208v2)
  - GAWM、MATWM 未给计算量 — [GAWM](https://arxiv.org/html/2501.10116v1)；[MATWM](https://arxiv.org/html/2506.18537v1)
- 鲁棒性测试：
  - 消息丢失/延迟：MAMBA、CoDreamer、DCWM、KL 信念门控均未测试 — [MAMBA](https://arxiv.org/pdf/2205.15023)；[CoDreamer](https://arxiv.org/html/2406.13600v1)；[DCWM](https://arxiv.org/html/2504.03353v3)；[KL 门控](https://arxiv.org/html/2608.14559)
  - 唯一与"丢包"相关的鲁棒性结论来自非世界模型的 Robot Web（GBP 鲁棒因子容忍丢包） — [arXiv 2202.03314](https://arxiv.org/pdf/2202.03314)
  - 观测缺失：Dreamer-CPC 的 CatchApple 设计为任务相关观测暂时缺失 — [abs](https://arxiv.org/abs/2607.19809)；DCWM 用观测离散化 bins=1 构造互补观测的 Dec-POMDP — [HTML](https://arxiv.org/html/2504.03353v3)
  - 智能体数：MAMBA Flatland 5/10/15；MARIE 2–7 消融；MATWM Pistonball 20、Melting Pot 2–8（无专门扫描）；COMBO 2–4；Gamma-World 2→4（零样本，定性）；Solaris 2（NeurIPS 摘要称 3 人概念验证、硬件可模拟 16 人） — [MAMBA](https://arxiv.org/pdf/2205.15023)；[MARIE](https://arxiv.org/html/2406.15836v2)；[MATWM](https://arxiv.org/html/2506.18537v1)；[COMBO](https://arxiv.org/html/2404.10775v2)；[Gamma-World](https://arxiv.org/html/2605.28816v1)；[Solaris poster](https://neurips.cc/virtual/2026/poster/148491)

### Inferences
- 丢包/延迟鲁棒性是多智能体世界模型文献中完全空白的评估维度——目标方法若在 SMAC/SMACv2 或 Melting Pot/VMAS 上系统扫描丢包率与延迟步数，并与 MAMBA（单步注意力通信）和 CoDreamer（GNN）对比，很可能是首个此类结果。
- 低成本可复现路线：以 MAMBA/MARIE 代码基座在少量 SMAC 图上运行（单卡 1–4 天/运行）；按"两个种子都必须通过"的规则，至少 2 种子 × 若干丢包率，预计数十 GPU 日（推断，基于 MARIE 报告的墙钟）。CoDreamer 的 VMAS/Melting Pot 设置约 9 TPU 小时/运行，也较可承受。
- 生成式视频世界模型（Solaris、Gamma-World、COMBO）的成本在 30–200 张加速卡量级，不适合作为本方法的主基准，但其"跨智能体一致性"评测（Solaris 的 Consistency 类别、Gamma-World 的 inter-agent consistency）可作为动机引用。

### Gaps
- MAMBA 附录中的具体墙钟数字、MABL/DMAWM/DIMA 的计算量未读到。
- 没有找到专门为多智能体世界模型设计的、带通信信道模型（丢包/延迟/带宽）的标准基准。

## Q5. 论文与综述明确提出的开放问题有哪些（智能体数扩展、通信带宽、非平稳性、部分可观测、异质性）？

### Takeaway
被明确点名的开放问题集中在：智能体数扩展（世界模型需预测所有智能体，样本效率随数目下降）、非平稳性（世界模型跟不上队友策略变化）、跨智能体状态一致性/持久共享状态（视频模型轨迹分歧、"可验证的共享状态维护"）、通信受限（带宽/延迟常被理想化）以及多智能体交互数据与基准稀缺。异质性与通信鲁棒性几乎只在综述中被提及，没有方法论文给出解决方案。

### Cited Findings
- 扩展性：MAMBA 在 Flatland 中智能体从 5 增到 15 时样本效率优势消失，作者归因于需要更强表达力的世界模型 — [MAMBA PDF](https://arxiv.org/pdf/2205.15023)；MARIE 的集中式联合动力学变体在 >3 智能体后急剧退化 — [MARIE HTML](https://arxiv.org/html/2406.15836v2)；Gamma-World 指出超大群体需更大旋转频带或层级分组 — [Gamma-World HTML](https://arxiv.org/html/2605.28816v1)
- 非平稳性：MATWM 明确说去中心化世界模型受非平稳性影响，世界模型落后于快速变化的策略时性能下降，并列为未来工作 — [MATWM HTML](https://arxiv.org/html/2506.18537v1)；Dreaming of Others 主张把队友建成结构化潜过程以减少表观非平稳性 — [PDF](https://arxiv.org/pdf/2605.31361)
- 联合建模的虚假相关：DMAWM 指出在潜空间联合建模会在智能体潜状态间产生虚假相关 — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61713)
- 一致性/持久共享状态：Solaris "缺乏持久记忆"、无底层持久世界状态，玩家离开彼此视野后轨迹分歧 — [Solaris HTML](https://arxiv.org/html/2602.22208v2)；Gamma-World 长 rollout 会累积不一致 — [Gamma-World HTML](https://arxiv.org/html/2605.28816v1)；MAMBA 指出独立预测导致视图不一致 — [MAMBA PDF](https://arxiv.org/pdf/2205.15023)
- 通信图动态：CoDreamer 想象中邻接矩阵固定、边特征过时，列为未来工作 — [CoDreamer HTML](https://arxiv.org/html/2406.13600v1)
- 综述：
  - MAEAD 综述（约 400 篇，预印本 arXiv 2606.13840，Hu et al.）把共享世界模型定义为跨车辆、基础设施和其他交通参与者维护的预测性跨智能体表征，并把"可验证的共享状态维护、稳健的意图与计划对齐"、通信与计算约束下的安全协同列为优先方向；指出评估集中在仿真、精选基准与离线协议 — [arXiv abs](https://arxiv.org/abs/2606.13840)
  - "A Comprehensive Survey on World Models for Embodied AI"（arXiv 2510.16732）：多智能体交互数据尤其欠缺，限制了社会感知与协作世界模型；建议基准显式覆盖多智能体交互与安全关键事件（仅搜索摘要） — [arXiv PDF](https://arxiv.org/pdf/2510.16732)
  - Cavolowsky（UMD 学术论文，2025）提出多智能体世界模型（MAWM）选型框架，指出扩展、分布式一致性、任务无关迁移等挑战，以及设计选择对可扩展性/通信效率/协调效果的权衡（仅搜索摘要；PDF 证书错误未能读取） — [UMD PDF](https://www.cs.umd.edu/sites/default/files/scholarly_papers/202501_Cavolowsky,_Mark_Scholarly_Paper.pdf)；[UMD 页面](https://www.cs.umd.edu/content/building-better-multi-agent-systems-theory-and-practice-multi-agent-world-models)
  - "From 2D to 3D Cognition" 综述：多个智能体在共享 3D 空间中带来可扩展性、实时性与全局一致性维护的挑战（仅搜索摘要） — [arXiv 2506.20134](https://arxiv.org/pdf/2506.20134)
  - 具身生成式多智能体协作系统综述：协作系统中的智能体数量仍然很少；基于基础模型的方法缺乏理论基础（仅搜索摘要） — [arXiv 2502.11518](https://arxiv.org/pdf/2502.11518)
  - 机器人学习世界模型综述：剩余挑战在长视野鲁棒性、跨基准泛化与缺乏标准化报告（仅搜索摘要） — [arXiv 2605.00080](https://arxiv.org/html/2605.00080v1)
- 带宽：MAMBA 把离散消息与局部性作为降低带宽、提高去中心化的两项设计原则 — [MAMBA PDF](https://arxiv.org/pdf/2205.15023)；KL 信念门控试图降低通信，但论文未报告平均通信节省，且在 MPE 上通信率恒为 1.0 — [arXiv 2608.14559](https://arxiv.org/html/2608.14559)
- 异质性：Dreaming of Others 把人类伙伴视为部分可观测、异质的自适应过程，作为未来方向（无实验） — [PDF](https://arxiv.org/pdf/2605.31361)；Gamma-World 将更复杂、异质、长视野的场景留作未来工作 — [Gamma-World HTML](https://arxiv.org/html/2605.28816v1)

### Inferences
- 对目标方法而言"已解决"的部分：每智能体潜变量 + 想象训练（MAMBA/MARIE/MATWM）、执行期定长离散消息维持世界模型（MAMBA，160 bit）、逐智能体分解以减少虚假相关（MAMBA 的 L_info、DMAWM）。
- "仍开放"的部分（与目标方法高度对齐）：(1) 显式精度/不确定性加权的跨智能体潜状态融合；(2) 多轮、可异步、对丢包与延迟有可证明或经验鲁棒性的共识；(3) 跨智能体潜状态一致性的保证或标准度量（GAWM 的 GCI 是唯一现成指标）；(4) 在线通信图变化下的世界模型（CoDreamer 承认未解决）；(5) 智能体数 >20 的扩展。
- 综述层面"可验证的共享状态维护"（MAEAD）与"分布式一致性"（Cavolowsky）正好对应"sheaf 能量为零即全局一致"这一可检查判据，可作为论文动机引用（推断）。

### Gaps
- 未找到一篇专门的、同行评审的 2025–2026 "多智能体世界模型"综述；Cavolowsky 文档的发表状态与全文未能核实。
- 2510.16732、2506.20134、2502.11518、2605.00080 的开放问题表述仅来自搜索摘要，未读全文。
- 异质智能体（不同观测/动作空间）的多智能体世界模型实验在本次检索中未发现。
