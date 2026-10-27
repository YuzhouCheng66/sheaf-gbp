# 噪声与未知相关下的多机 occupancy 融合、GBP 联合位姿-地图推断与 sheaf 先例（文献至 2026-10）

范围说明：本笔记接续 `HouseExpo 多机地图融合文献综述/distributed_gbp_sheaf_fusion.md`。那份笔记已经覆盖了 Robot Web / DistGP / RAMEN 的规模与带宽、Tarski Laplacian 与 max-consensus 的有限步收敛、CI/ICI/channel filter 的定性性质，以及 Robinson、Hansen–Ghrist 和 Bodnar 的基础工作。本文不重复这些内容，只补充四个方向：(a) 噪声与相关性下各融合规则的假设和适用边界；(b) 配准误差如何建模与恢复；(c) GBP 与这些融合规则的精确对应；(d) learned prior 与 factor graph / GBP 的结合方式。

关于"已读"的范围：每条 Cited Finding 都在本次会话中从所链接的页面读过。其中三份 PDF（Bailey 2012、Makarenko 2009、Saeedi 2014）由本地脚本抽取全文后检索关键段落；带"检索摘要"标记的条目只读过搜索结果摘要，没有读全文。推导出的结论一律放在 Inferences。

背景（内部 probe，本文未复核）：无噪声时 max-selection 是精确的。加入 range noise 后，radio tree 上的 log-odds 求和（每个 agent 只计一次）在 surfaces/furniture 上比 max-selection 高 9–24 F1。观测同一 voxel 的 agent 增多时，max-selection 从 .84 降到 .56，求和从 .85 升到 .99。5 cm / 1° 的配准误差会损失约 16–29 个点。

---

## 1. 多机 occupancy 融合规则：log-odds 求和、opinion pools、double counting / rumour propagation、channel filter、CI/ICI、consensus——各自何时适用，有无定量比较

### Takeaway
文献给出的是一个由假设决定的谱系，没有通用最优规则：
- **独立性成立（每条测量只计一次）**：log-odds 求和（independent opinion pool / 标准 Bayes）最优。在树形拓扑上，channel filter 能把它做成精确的分布式算法（Grime & Durrant-Whyte 1994；Makarenko et al. 2009 证明它就是 junction tree 上的 BP）。
- **相关性完全未知**：用 geometric mean density（GMD，又称 logarithmic opinion pool / exponential mixture / Chernoff fusion），即 CI 的非高斯推广，权重和为 1，不会 double count（Bailey, Julier & Agamennoni 2012）。
- **per-coordinate max-precision selection**：就是标量 CI 在熵/行列式最优 ω 下的解。Bailey 等明确指出一维高斯的最优 ω 总在 0 或 1。所以它属于 CI 族的极端点：保守、幂等，但会丢掉独立证据带来的信息增益。

没有找到在多机 occupancy 上把 sum、max、CI 三者直接做定量对比的论文。现有定量比较只出现在 Bayes 与 Dempster–Shafer 之间，差异很小（0.001–0.022），而且结论依赖传感器模型的匹配方式。

### Cited Findings

GMD / CI / opinion pools（Bailey, Julier, Agamennoni, "On Conservative Fusion of Information with Unknown Non-Gaussian Dependence", Fusion 2012, pp. 1876–1883）：
- 论文研究的融合规则是 GMD，即分量 pdf 的加权几何平均 p_c ∝ p_a^ω p_b^(1−ω)，作者称其为 CI 在非高斯 pdf 上的推广。同一规则在文献中也叫 logarithmic opinion pool、Chernoff fusion 或 exponential mixture densities。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf); [IEEE Xplore](https://ieeexplore.ieee.org/document/6290529)
- 权重与独立性假设：两个权重都取 1 就是标准 Bayes 更新，只有在分量条件独立时才成立。权重"和为 1"的约束能避免 common information 被 double count（Theorem 3: "The GMD does not double count common information"）。权重和大于 1 则隐含"已知部分依赖程度"，作者引用了 Julier 2009, "Estimating and exploiting the degree of independent information in distributed data fusion"。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf)
- 精确去除公共信息的恒等式：若 Z_a、Z_b 的公共部分为 Z_a∩Z_b，且数据条件独立，则 p(x|Z_a∪Z_b) ∝ p(x|Z_a)p(x|Z_b)/p(x|Z_a∩Z_b)。这正是 channel filter 的"除以公共信息"。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf)
- 非归一化的加权几何平均逐点夹在两个分量之间（min ≤ GMD ≤ max）。对一维高斯，GMD/CI 无法降低熵，最优 ω 总在 0 或 1（"the optimum will always be ω=0 or ω=1"）。即使是独立数据的最优 Bayes 融合，在证据冲突时熵也可能上升。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf)
- linear opinion pool（AMD，算术平均）同样不会 double count，但作者称它做的是 "amalgamation not fusion"，结果是混合分布。GMD 要求每个分量本身是一致的：任何一方若在真值处概率为 0，GMD 就不一致。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf)
- ω 的选择：传统做法是最小化行列式或迹，也可以用 Chernoff information 或 KL 散度来选。作者承认"如何正当地选 ω"仍是开放问题。— [ACFR PDF](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf)

Opinion pools 的经典性质（多传感器 Bayes 融合学位论文）：
- Independent Opinion Pool（乘积规则）假设各源条件独立。当各节点共享一个公共先验时，它对观点的强化"极端"（"extreme in its reinforcement of opinion when the prior information at each node is common"）。Linear Opinion Pool 即使 M 很大，只要有一个传感器持异议，结果也可能出错。— [Bayesian Approaches to Multi-Sensor Data Fusion (Pitt 镜像)](https://people.cs.pitt.edu/~chang/231/y08/y08sem/semkul/mphil.pdf)

Channel filter / 树拓扑 / rumour：
- Makarenko, Brooks, Kaupp, Durrant-Whyte, "Decentralised Data Fusion: A Graphical Model Approach"（Fusion 2009）有以下要点。
  - 核心问题是防止两个相关的局部后验在融合时 double count。
  - Channel Filter (CF) 靠两点做到这一点：坚持树形通信拓扑，并在每条通信边上维护一个 channel filter，记录这条边上已经传过的公共信息。
  - CF "is exact for static problems or for dynamic problems when full-rate communications are used"。
  - 论文用分布式 Junction Tree 算法（Paskin & Guestrin）重新推导了 CF：底层在物理 mesh 网络上建 spanning tree，中层强制 running intersection property，顶层做推断。
  - 替代方案有两种：Information Graph（精确，但要传 pedigree），以及对高斯始终使用 Covariance Intersection 做保守融合。
  - 来源：[Makarenko et al. 2009 PDF](https://dellaert.github.io/files/Makarenko09fusion.pdf)
- CF 的源头文献：S. Grime, H. F. Durrant-Whyte, "Data fusion in decentralized sensing networks"（Control Engineering Practice, 1994）。本次只看到书目记录，没有读全文。— [ResearchGate 记录](https://www.researchgate.net/publication/243772218_Data_fusion_in_decentralized_sensing_networks)
- 拓扑可以替代簿记：在 hypercube gossip 拓扑上，只要各节点的估计误差互不相关，就可以直接利用网络拓扑避免 double counting，不需要 channel filter 或 pedigree（检索摘要）。— [Scalable decentralised data fusion using hypercube gossiping](https://www.researchgate.net/publication/254049797_Scalable_decentralised_data_fusion_using_hypercube_gossiping)
- "data incest" 的定义是同一信息被多次使用、却被当作彼此独立。这是 CI 类保守融合存在的理由。— [Interval Split CI (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11124889/)

多机 occupancy 地图实践中的求和：
- Saeedi et al.（IEEE RAM 2014，Group mapping / probabilistic GVD）把收到的对方地图视为"a batch of sensor data"，用 log-odds 的可加性直接相加：l_fused = l_1 + l'_2。之后再用 entropy filter 拒绝会让熵上升的更新。— [Saeedi_RAM_2014 PDF](https://people.csail.mit.edu/lpaull/publications/Saeedi_RAM_2014.pdf)
- 一项多机 SLAM 专利在每次相遇时对对应 cell 的 log-odds 求和（检索摘要）。— [US 11347238](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11347238)

Consensus 类方法：
- Ramachandran, Kakish, Berman（IEEE T-RO 2020, arXiv 1903.04836）中，每个机器人反复把自己的激光测量与邻居广播的 occupancy 地图融合，基于时变图上的 average consensus。作者证明所有机器人的地图最终收敛到真实地图，前提是全局定位已知、通信受限。摘要页没有讨论 double counting。— [arXiv](https://arxiv.org/abs/1903.04836)
- 一篇多机 semantic octree 工作把分布式建图写成多类 log-odds 上的带 consensus 约束的优化问题（检索摘要）。— [arXiv 2402.08867](https://arxiv.org/pdf/2402.08867)
- Battistelli, Chisci 等人的工作把 consensus 分成两类，并提出了混合方案：
  - consensus on information 是在 KL-average 意义下对概率密度做共识。线性高斯情形下，它等价于对 information matrix 和 information vector 做 consensus，也就是 CI 的共识版本（IFAC 2011）。— [Curtin 记录](https://espace.curtin.edu.au/handle/20.500.11937/4438)
  - "Consensus-based linear and nonlinear filtering"（IEEE TAC 60, 2015, pp. 1410–1415）提出混合的 HCMCI，把 consensus on measurements 与 consensus on information 结合起来，在集体可观且网络连通时对线性系统证明了稳定性。— [DOI 10.1109/TAC.2014.2357135](https://unpaywall.org/10.1109%2FTAC.2014.2357135)
  - 二手综述对两者取舍的说法：CI 把节点间相关性视为完全未知，偏保守，consensus 步数被截断时精度受损；CM 只融合新信息，不保守，但需要足够多的 consensus 步才能保证稳定。— [TAMU T-RO 2021 PDF](https://cse-robotics.engr.tamu.edu/dshell/papers/tro2021dse.pdf)

定量比较（仅限 Bayes 与 DS，未找到 sum/max/CI 的对比）：
- Berlenko & Krinkin（arXiv 2602.18872，2026-02）用 BetP 匹配单次观测的判决概率，以此隔离融合规则本身的影响。结果是 Bayes log-odds 在 15/15 个方向比较中占优（p = 3.1e-5），但绝对差只有 0.001–0.022；换成 normalized plausibility 匹配时方向反转。— [arXiv](https://arxiv.org/abs/2602.18872)
- Sensors 2026 的 controlled comparison 发现，Bayes 与 belief-function 的比较结论高度依赖传感器模型参数化，用社区默认值独立参数化甚至会反转边界锐度。Bayes 一侧对 log-odds 做了 |L| ≤ L_max 截断。— [MDPI Sensors 26(13) 4266](https://www.mdpi.com/1424-8220/26/13/4266)
- 一项车载建图专利把 maximum pool 与 Bayesian、Dempster-Shafer、linear/independent pools 并列为可选融合方法，但没有评估（检索摘要）。— [US 12442928](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12442928)

### Inferences
- **把四种规则放进同一个参数化。** 对单个 Bernoulli cell，归一化 GMD 有闭式解 l_c = ω_a·l_a + ω_b·l_b，其中 l 为 log-odds。这是由 p^ω 的 log-odds 线性性直接推出的。不同取值对应不同规则：
  - ω_a = ω_b = 1 是求和，即 independent opinion pool / Bayes。
  - ω_a + ω_b = 1 是 CI/GMD（保守）。
  - 1 < ω_a + ω_b < 2 是"部分独立"（Julier 2009 的思路）。
  - 逐 cell 取 ω ∈ {0,1} 中使 |l_c| 最大者，就是 per-coordinate max-confidence selection。由于 Bernoulli 熵随 |l| 单调递减，这个选择恰好是逐 cell 的熵最优 CI。这与 Bailey 等"一维高斯最优 ω 在端点"的结论相同。
  - 结论：项目当前的 "scalar CI / max-precision selection" 在文献里有准确定位，它是 CI 族在标量坐标上的最优解，并不是一种临时规则。
- **为什么 agent 越多 max 越差（.84 → .56）、求和越好（.85 → .99）。**
  - max-selection 取的是 N 个噪声证据的极端次序统计量。只要每个 agent 有概率 p 产生一个"高置信但错误"的 cell，被选中的概率约为 1 − (1 − p)^N，随 N 单调上升。CI 族从不累积独立证据，所以这种误差不会随 N 衰减。
  - 求和则是 N 个独立对数似然比相加：均值按 N 线性增长，噪声按 √N 增长，错误率按 Chernoff 界指数下降。
  - probe 的方向与上述推理和文献完全一致（文献本身没有给出 occupancy 上的数值）。
- **何时用什么。**
  - (i) 测量条件独立、每条测量只计一次（radio tree + 每 agent 计一次 = channel filter 的前提）时，用求和。它最优，而且是精确的分布式解。
  - (ii) 存在重复转发或环路、无法追踪来源时，用 CI/GMD（权重和为 1）或 max-selection。保守，但要付出"独立信息被丢弃"的代价，并且噪声下随 N 变差。
  - (iii) 能估计重叠程度时，用 1 < Σω < 2 的部分独立融合。
  - (iv) 拓扑可控时，可以靠拓扑避免 double counting（树 / hypercube），比 CI 更紧。
- **在不放弃 lattice/幂等前提下拿到求和语义。**
  - 做法是让每个 agent 的 log-odds 贡献按 agent-ID 分槽保存，每槽带版本号。这是 last-writer-wins register 的乘积，是一个 join-semilattice（CRDT 的 G-Counter 模式：按副本取 max，读出时求和；见前一份笔记引用的 Shapiro et al. 2011）。融合读数取各槽之和。
  - 好处：join 幂等、可交换、可结合，能在任意图上收敛到"每个 agent 恰好计一次"的和。这与 Information Graph / pedigree 的思路相同。
  - 代价：状态随 agent 数 N 线性增长。在树上只需每条边维护一个 channel filter，不需要 O(N) 状态。
  - 这条可以作为 "Tarski Laplacian 硬约束" 与 "求和精度" 之间的折中，但尚未在文献中看到 occupancy 上的实现。
- **公共先验也会被重复计数。** 如果每个 agent 发送的是"后验"而不是"似然"（例如 log-odds 初值非 0，或 learned code 的后验已包含 N(0, I) 先验），直接求和会把先验计 N 次。按 Bailey 的恒等式，应该发送 posterior ÷ prior，或在融合后减去 (N−1) 份先验。

### Gaps
- 没有找到在多机 occupancy grid 上定量比较 log-odds sum、max/selection、CI/GMD 和 consensus 的论文。CI 用于 occupancy/evidential grid 的工作也没有找到，两次检索都为空。
- Julier & Uhlmann 1997 CI 原文、Grime & Durrant-Whyte 1994 原文、Julier 2009（degree of independent information）都没能打开。Battistelli TAC 2015 和 unifi 仓库的全文返回 403，HCMCI 的数值结果没有拿到。
- Ramachandran et al. T-RO 2020 的融合细节（权重、log-odds 还是概率、是否处理重复计数）只读到了摘要。
- ICI（Noack 2017）相对 CI 的数值增益见前一份笔记的定性描述，本次没有补到数值。

---

## 2. 不确定相对位姿下的多机地图合并（Birk & Carpin、Saeedi 等）：配准误差如何建模、如何纠正

### Takeaway
文献中处理配准误差有三条路线：
- **把位姿不确定性"摊进"地图。** 线性化变换协方差，对每个 cell 做位置相关的高斯扩散（Saeedi et al. 2014）。这只能让融合变得保守、减少假表面，无法恢复丢失的精度。
- **用图/扫描匹配重新估计变换。** 包括 Birk & Carpin 的 overlap/acceptance 搜索、Hough/GVD 匹配、ICP 精修。
- **把相对位姿作为变量放进 factor graph 联合优化，再用优化后的轨迹重建或形变地图。** 例如 Kimera-Multi 的 D-GNC + mesh deformation，Robot Web 的 GBP 跨机因子。

只有后两条能真正收回配准造成的损失，这与内部 probe"只有配准能恢复 16–29 点"的结论一致。学习式融合网络在合成数据上能容忍 5 m / 20° 的错位，但只是"补偿"错位，并不估计位姿。

### Cited Findings
- Birk & Carpin, "Merging occupancy grid maps from multiple robots"（Proceedings of the IEEE, 2006）把多机地图合并建模为：在相对位姿未知时，搜索使局部地图重叠区域一致的变换。这是该方向的经典参考（检索摘要；PDF 返回 403）。— [UC Merced PDF](https://robotics.ucmerced.edu/sites/g/files/ufvvjh1576/f/page/documents/pro2006.pdf)
- Saeedi et al.（IEEE RAM 2014，probabilistic GVD map merging）的流程如下：
  - 把变换 f(θ, x, y) 用一阶 Taylor 线性化，给每个被变换的点赋一个高斯分布。其协方差 R_r 是 cell 位置 q 的函数，离旋转中心越远扩散越大（Algorithm 1 逐 cell 施加以原点为中心的高斯核）。
  - 先在带不确定性的旋转下对齐，再求平移，然后再次传播平移不确定性，"We now have a fully probabilistic representation of the required transformation"。
  - 随后用 log-odds 加和融合，并用 entropy filter 拒绝使熵增大的更新。
  - 与 adaptive random walk map merging 和 map segmentation 相比"至少快 8 倍"。
  - 文中认为两大难点是"无全局位置参考的合并"和"包含地图不确定性的融合"。
  - 来源：[Saeedi_RAM_2014 PDF](https://people.csail.mit.edu/lpaull/publications/Saeedi_RAM_2014.pdf)
- van Kempen et al.,"Combined Registration and Fusion of Evidential Occupancy Grid Maps for Live Digital Twins of Traffic"（IEEE IV 2023）：
  - 训练一个 DNN，输入两车各自的 evidential OGM，直接输出融合后的 evidential OGM 及一阶、二阶不确定性。
  - 合成实验中可补偿最多 5 m、20° 的空间错位。
  - 只用合成数据训练，在真实数据上优于"坐标变换 + 组合规则"基线。摘要页没有给出数值。
  - 来源：[arXiv 2304.03578](https://arxiv.org/abs/2304.03578)
- Kimera-Multi（Tian et al., T-RO 2022）：
  - 局部感知相似导致 perceptual aliasing，会产生错误的跨机关联，从而破坏标准后端。为此用分布式 GNC（D-GNC）做鲁棒 PGO 来剔除错误回环。
  - 每个机器人用改进后的轨迹对本地 mesh 做 mesh deformation，得到全局一致的 metric-semantic mesh。
  - 单机轨迹最长约 800 m（检索摘要）。
  - 来源：[arXiv](https://arxiv.org/abs/2106.14386v2); [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.14386)
- 一篇基于 octree occupancy 的多机 3D 建图工作用 ICP 精修不确定的地图间变换，并指出机器人无法把相对位姿测到给定误差界以内（检索摘要）。— [ResearchGate](https://www.researchgate.net/publication/290875273_Robust_and_efficient_multi-robot_3D_mapping_with_octree_based_occupancy_grids)
- Andersone 系列工作：
  - 错误匹配最常发生在相似区域很多的环境（例如长宽相近的走廊）。— [SciTePress 2021](https://www.scitepress.org/Papers/2021/103424/103424.pdf)
  - 局部不准会增加合并难度，"globally inaccurate maps are virtually impossible to merge"，相似度评价需要对局部差异鲁棒。— [LLU 2013](https://iitf.lbtu.lv/conference/proceedings2013/Papers/084_Andersone_I.pdf)
  - 提出无真值的 CNN 地图质量评价，建议在合并时给评分高的区域更高权重。— [SciTePress 2020](https://www.scitepress.org/Papers/2020/91755/91755.pdf)
- Yu et al.（Sensors 2020）的综述按 occupancy grid、feature 和 topological 三类地图组织合并方法，并认为局部图合并为全局图的步骤在很大程度上决定多机 SLAM 的性能（检索摘要）。— [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7730201/)
- 多车 OGM 融合（IET Communications 2022）先做坐标变换，再用 kernel density estimation 估计占用概率（检索摘要）。— [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cmu2.12314)
- Robot Web（T-RO 2024）与 Murai et al.（RA-L 2024 auto-calibration）用 GBP 在跨机测量因子上联合估计位姿和外参。具体精度数字见前一份笔记：九台 TurtleBot3 的 ATE 从 0.071 m 降到 0.028 m；N=64 时 extrinsics 误差为 0.025 m / 1.214°，集中式为 0.020 m / 0.969°。— [ar5iv 2202.03314](https://ar5iv.labs.arxiv.org/html/2202.03314); [arXiv 2401.15036](https://arxiv.org/html/2401.15036v1)

### Inferences
- **量级估算（推导）。** 5 cm / 1° 的配准误差在距离 r 处的横向位移约为 0.05 + r·0.01745 m：r = 3 m 时约 0.10 m，r = 5 m 时约 0.14 m，r = 8 m 时约 0.19 m。这已经超过常用的 5–10 cm voxel。对薄的 surface/furniture 来说，等于把同一表面错开一到两个 voxel，F1 损失 16–29 点在量级上合理。
- **配准误差下不同融合规则的表现不同。**
  - max/selection 对 occupied 近似取并集，错位会生成"双层表面"（假阳性）。
  - 求和会把错位证据摊开，部分相互抵消，表面变厚、变模糊。
  - Saeedi 式高斯扩散相当于主动做模糊，换取一致性。
  - 三者都无法恢复锐利的几何，只有重新估计变换（路线 2/3）可以。
- **与 GBP 及硬约束的兼容性。** 路线 3 天然兼容 GBP：相对位姿（SE(2)/SE(3)）作为变量节点，跨机重叠证据作为因子，消息在切空间中以 (η, Λ) 传递，与 Robot Web 相同。地图项可以与位姿项交替更新（EM / block-coordinate），也可以联合线性化。Kimera-Multi 的教训是：跨机关联必须配鲁棒核（GNC/DCS），否则走廊类环境中的错误匹配会破坏整个图。
- **学习式融合的位置。** van Kempen 的网络只学会了"在错位存在时输出合理的融合图"，并不产出位姿。它适合作为 learned prior 或初始化，不能替代配准。

### Gaps
- 没有找到"注入相对位姿噪声（cm/°）→ occupancy F1/IoU"的受控扫描研究。现有文献只有定性说法，无法与内部 probe 的 16–29 点直接对照。
- Birk & Carpin 2006 和 Carpin 2008（Hough spectra map merging）原文返回 403，没有读到；Saeedi et al. 的 JFR 2016 综述 "Multiple-robot SLAM: A review" 也没有检索到全文。
- Lee et al.（Robotica 2012）"probabilistic map merging for multi-robot RBPF-SLAM with unknown initial poses" 只在检索摘要中出现。
- Kimera-Multi 的 mesh 精度数值没有拿到。

---

## 3. Gaussian Belief Propagation：SLAM / BA / 分布式估计、深度 factor graph 学习、环图收敛、量化与压缩消息

### Takeaway
- **树上精确，等价于 channel filter。** GBP 与 channel filter 在树上是同一个算法（Makarenko 2009 用 junction tree 推断推导了 CF）。因此"radio tree 上 log-odds 求和、每个 agent 计一次"本质上就是树上精确的 (G)BP。它满足"GBP 式迭代、下降能量"的硬约束，只是不满足幂等性。
- **环图收敛。** Du et al.（JMLR 2018）对向量 GBP 证明：任意 PSD 初始化下 information matrix 指数收敛到唯一极限；"forest 加单个环"总是收敛；其条件严格弱于 walk-summability。
- **工程与学习侧。** GBP 已用于 BA，IPU 上 125 帧 / 1919 点用时 <40 ms，Ceres 为 1450 ms（Ortiz et al. CVPR 2020）。Nabarro et al.（ICML 2024）把整个 CNN 写成 factor graph 用 GBP 训练，单遍 MNIST 98.16%、CIFAR-10 53.1%，但需要 damping 0.9 + dropout 0.5 才能稳定。
- **没有找到量化或按比特预算压缩 GBP 消息的论文**，与前一份笔记一致。现有的"压缩"只有低秩消息（GEnBP）和广播式减少消息数量（Bickson）。

### Cited Findings
- Makarenko et al. 2009：CF 是在 spanning tree 上运行的分布式 Junction Tree 推断。CF 链路构成 spanning tree，每对相邻平台各存一个 channel filter，初值为 1；收到消息后用"消息 ÷ channel 中的公共信息"更新本地 belief。— [PDF](https://dellaert.github.io/files/Makarenko09fusion.pdf)
- Du, Ma, Wu, Kar, Moura,"Convergence Analysis of Distributed Inference with Vector-Valued Gaussian Belief Propagation"（JMLR 18(172), 2018）：
  - 在宽泛条件下，任意 PSD 初始化的消息 information matrix 都收敛到唯一的正定极限，速率为指数级。
  - 给出了 belief mean 收敛到集中式最优估计的充要条件。
  - factor graph 为 forest 加单个环时 GBP 总收敛。
  - 条件严格弱于 walk-summability。
  - 来源：[JMLR](https://jmlr.org/papers/v18/16-556.html)
- Ortiz, Pupilli, Leutenegger, Davison,"Bundle Adjustment on a Graph Processor"（CVPR 2020）：
  - 在 Graphcore IPU 的 1216 核上用 GBP 求解 BA，125 keyframes / 1919 points 时 <40 ms，Ceres CPU 为 1450 ms。
  - 作者承认 GBP 在 CPU 上未必比全局优化方法有竞争力，并认为它更适合动态、增量问题。实验包含增量 SLAM、鲁棒代价和多种因子类型。
  - 来源：[arXiv 2003.03134](https://arxiv.org/pdf/2003.03134); [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Ortiz_Bundle_Adjustment_on_a_Graph_Processor_CVPR_2020_paper.html)
- Nabarro, van der Wilk, Davison,"Learning in Deep Factor Graphs with Gaussian Belief Propagation"（ICML 2024, arXiv 2311.14649）：
  - **建模方式。** inputs、outputs、parameters 和 latents 都作为随机变量，训练和预测都是推断。conv、transposed conv、max-pool、upsampling、dense 和 softmax 都写成二次能量因子，内含非线性 g(·)。用低秩观测结构和 Woodbury 恒等式把 factor→variable 的复杂度从 O(V(V−1)³) 降到 O(VM³)。
  - **视频去噪（DAVIS bear，10% 像素损坏）。** 第 5 帧 PSNR：pairwise 平滑 31.1，单层 32.6/32.9（逐帧/continual），五层 33.4/34.3。鲁棒因子采用类 Tukey 形式，N_rob = 1.4 / 0.2。
  - **分类。** 单遍 MNIST 98.16 ± 0.03%。CIFAR-10 53.1 ± 0.3，对比 lucibello2022deep 的 41.3（后者训练 100 epoch）。只有当 Adam + replay buffer 达到约 6×10³ 例时才能追平 GBP。
  - **异步调度。** 随机层顺序得 98.11 ± 0.04%，同步为 98.16%，但只是单机上的调度实验。
  - **作者列出的局限。** 只用了标量变量；有环或非线性因子时无收敛保证，需要 damping 0.9、dropout 0.5；在现有硬件上远慢于反向传播（MNIST 约 3 小时，RTX 3090）。
  - 来源：[arXiv HTML](https://arxiv.org/html/2311.14649v3); [arXiv abs](https://arxiv.org/abs/2311.14649)
- Bickson 的 GaBP 专著：在稠密图上，广播式 GaBP 把需要传输的不同消息数从 O(n²) 降到 O(n)，适合无线广播网络（检索摘要）。— [arXiv 0811.2518](https://arxiv.org/pdf/0811.2518)
- GEnBP（2024）把 Ensemble Kalman Filter 与 GaBP 结合，沿边传递低秩消息，可扩展到比 GaBP 更高的因子维度（检索摘要）。— [arXiv 2402.08193](https://arxiv.org/pdf/2402.08193)
- GaBP 的脉冲神经网络实现（2025）基于 Forney-style factor graph（检索摘要）。— [arXiv 2512.10638](https://arxiv.org/pdf/2512.10638)
- 前一份笔记已经核实、此处不重复展开的结论：Weiss & Freeman 1999 证明环图 GBP 收敛时均值精确；Robot Web 指出环图上方差过度自信、无一般收敛保证；DistGP（ICRA 2026）用 GBP 训练分布式稀疏 GP。

### Inferences
- **把 occupancy 融合写成 GBP。** 每个 voxel 的 log-odds 是一个变量，每个 agent 的观测是挂在该变量上的 unary 因子，跨 agent 的通信图决定消息路径。
  - 在树上，GBP 的 "variable→neighbour 消息 = 除去该邻居后其余所有信息之和" 正好就是 channel filter 的"减去公共信息"。它在直径 d 轮后精确，与 max-consensus / Tarski flow 的 d 轮界相同。因此在树上从 max 切换到求和，不增加通信轮数。
  - 在环上，同一算法会把环上回传的信息重复计入。这就是 loopy BP 的方差过度自信（rumour），在 log-odds 中表现为置信过高。补救办法有三种：限定在 spanning tree 上传播（CF）、使用 pedigree / per-agent 槽（见第 1 节），或退回 CI（Σω = 1）。
- **能量下降。** 高斯 / information 形式下，树上 GBP 是对二次能量的精确块消元；在环上 damping 有助收敛（Robot Web 用 0.2，Nabarro 用 0.9）。这与"GBP 式迭代下降能量"的硬约束兼容。lattice join（Tarski）是另一种单调流：在 lattice 序意义下单调，但不对应二次能量。
- **比特预算。** 量化 GBP 消息没有现成文献。前一份笔记引用的 quantised-innovation KF 显示 2–3 bit 每标量已经接近无损，可以作为 8+8 bit 的旁证。注意：求和型消息的量化误差会随轮数累积，幂等的 max 型则不会。这是选择求和后需要单独验证的点（推断）。

### Gaps
- 没有找到 GBP 消息量化、按比特预算压缩或 event-triggered GBP 的论文，三次不同措辞的检索都为空。
- Ortiz, Evans, Sucar, Davison,"Incremental Abstraction in Distributed Probabilistic SLAM Graphs"（ICRA 2022）没有检索到，它可能是"GBP + 学习式抽象"的直接先例。
- 没有找到用 GBP 同时估计跨机配准与 occupancy/latent 地图的系统。Robot Web 和 Murai 只估计位姿/外参，DistGP 假设位姿已知。

---

## 4. 传感融合与分布式估计中的 sheaf 理论：Robinson、Hansen–Ghrist、Ghrist–Riess、sheaf NN，以及用于位姿/坐标系对齐的 connection sheaf（SE(2)/SE(3) synchronization）

### Takeaway
"配准 = 让 sheaf 存在 global section"在数学上有清晰的先例：
- synchronization 问题等价于图上的 flat principal G-bundle；可同步当且仅当 holonomy 平凡，障碍由一个 twisted 上同调群刻画；graph connection Laplacian 就是 0 阶 Hodge/sheaf Laplacian（Gao, Brodzki, Mukherjee, DCG；Hansen–Ghrist 2019）。
- 2026 年已有从噪声信号联合学习"一致的 connection graph"（含局部坐标系）的方法 SCGL。

但在 SLAM 或多机建图中，没有找到把 SE(2)/SE(3) 配准写成 sheaf 并与地图 sheaf 耦合的论文。位姿同步仍然在 SE-Sync / 分布式 PGO 的框架下研究。

### Cited Findings
- Gao, Brodzki, Mukherjee,"The Geometry of Synchronization Problems and Learning Group Actions"（arXiv 1610.09051，Discrete & Computational Geometry）：
  - 图上关于拓扑群 G 的 synchronization 问题对应 flat principal G-bundle。
  - 可同步性由 holonomy 是否平凡决定，障碍是一个 twisted de Rham cochain complex 的上同调群。
  - graph connection Laplacian 是 0 阶 Hodge Laplacian。
  - 论文还给出按局部可同步性划分对象的算法。
  - 来源：[arXiv](https://arxiv.org/abs/1610.09051)
- Hansen & Ghrist,"Toward a Spectral Theory of Cellular Sheaves"（J. Appl. Comput. Topol. 2019, arXiv 1808.01513）：
  - graph connection Laplacian（Singer–Wu）可以看作 O(n) 向量丛上的 sheaf Laplacian。
  - 文中提出 "homological programs"：在 C^k(X; F) 上优化目标，约束为最优解位于 H^k(X; F)。
  - 来源：[arXiv PDF](https://arxiv.org/pdf/1808.01513); [Springer](https://link.springer.com/article/10.1007/s41468-019-00038-7)
  - Ghrist 主页把这条线概括为 "homological programming as a topological version of linear programming"。— [Ghrist research page](https://www2.math.upenn.edu/~ghrist/research.html)
- Hanks et al. 2025 用 ADMM 把 homological programming 扩展到 nonlinear sheaf Laplacian。— [arXiv 2504.02049](https://arxiv.org/html/2504.02049v2)
- Di Nino, D'Acunto, Barbarossa, Di Lorenzo,"Structured Sheaf Learning of Consistent Connection Graphs"（arXiv 2608.08710，2026-08）：
  - connection graph 即节点上的向量信号加上边上的正交 transport map。
  - 从噪声向量信号联合估计去噪信号、拓扑和逐节点的局部参考系，并把 connection Laplacian 的谱与组合 Laplacian 的谱耦合，以保证 global section 空间非平凡。
  - 块坐标算法 SCGL 收敛到非凸问题的驻点。摘要中没有数值，也没有提到位姿应用。
  - 来源：[arXiv](https://arxiv.org/abs/2608.08710)
- 一篇 2026 预印本（Bochner–Weitzenböck decomposition）把 Bandeira–Singer–Spielman 的 connection Laplacian Cheeger 不等式推广到 Bochner 矩阵，覆盖球面和 O(d) 上的部分同步与完全同步（检索摘要）。— [arXiv 2610.28455](https://arxiv.org/html/2610.28455)
- Barbero et al.,"Sheaf Neural Networks with Connection Laplacians"（2022）用 connection Laplacian 构造 sheaf NN 的 restriction maps。本次只确认了标题和 arXiv 号，没有读正文细节。— [arXiv 2206.08702](https://arxiv.org/pdf/2206.08702)
- 位姿同步的主流框架没有使用 sheaf 语言。一篇 2026 年的去中心化 PGO 论文的相关工作部分列出了：
  - Rosen et al. 的 SE-Sync，具有可验证的正确性。
  - Briales et al. 把状态空间扩展到 Cartan motion group。
  - Choudhary et al. 的两阶段分布式 Gauss–Seidel，先解耦旋转再解平移，只交换涉及跨机回环的位姿。
  - 来源：[arXiv 2606.24489](https://arxiv.org/pdf/2606.24489)
- 以 "sheaf + pose graph / SLAM" 检索没有得到相关论文。另有一篇 "A homological approach to consensus and fault tolerance in decentralized sensor networks"（Bol. Soc. Paran. Mat.），只看到标题。— [记录](https://periodicos.uem.br/ojs/index.php/BSocParanMat/article/view/80121)
- 以 "sheaf Laplacian + Gaussian belief propagation" 检索，只返回 GaBP 与 Kalman 的经典文献，没有 sheaf 与 GBP 结合的论文。— [检索所见 GaBP 分布式 KF 文献 arXiv 0810.1628](https://arxiv.org/pdf/0810.1628)
- Robinson（consistency radius、nearest global section）、Hansen–Ghrist（opinion dynamics）、Ghrist–Riess（Tarski Laplacian）、Bodnar et al.（Neural Sheaf Diffusion）的结论见前一份笔记。

### Inferences
- **把配准和地图写成两层 sheaf。**
  - 底层是 pose / connection sheaf：stalk 为各 agent 的坐标系，边上的 restriction map 为相对 SE(2)/SE(3) 变换。global section 存在当且仅当环路 holonomy 平凡，即 cycle consistency。在切空间线性化后，这就是 Robot Web 式 GBP 的消息空间。
  - 上层是 map sheaf：stalk 为局部 log-odds 或 latent code。restriction map 由底层位姿参数化，是重采样或投影算子。
  - 联合能量 = 位姿图能量 + 地图在重叠区域的不一致能量（sheaf Laplacian 二次型）。对 restriction map 求梯度，就是"用地图一致性来配准"（等价于 scan-to-map / photometric 配准）。
  - 这个结构可以严格落在"sheaf 向量 + GBP 式迭代 + 能量下降"的硬约束内。文献中没有现成实现。
- **诊断量。** Robinson 的 consistency radius 在上层 sheaf 上可以直接用作"配准残差"指标：融合前重叠区的不一致度高，说明 restriction map（位姿）有误，而不是传感器噪声。这能把内部 probe 中"噪声导致的损失"和"配准导致的损失"分开报告。
- **Tarski Laplacian 的适用边界。** 它在 lattice 序下精确、有限步收敛，但它的 "global section" 是 lattice 意义上的一致。restriction map 一旦是带误差的几何变换，lattice join 会把错位证据当作"更确定"而保留（例如 max 取到错位的 occupied）。因此在配准不确定时，用 vector-valued sheaf Laplacian（二次能量）比 lattice sheaf 更自然。

### Gaps
- 没有找到在 SLAM 或多机建图中把相对位姿写成 sheaf restriction map 并联合优化的论文，也没有找到 sheaf Laplacian 与 GBP 显式结合的论文。
- SCGL（2608.08710）和 Bochner 预印本（2610.28455）只读到摘要，不清楚是否涉及 SE(d) 或平移分量。
- Singer 2011 angular synchronization、Bandeira–Singer–Spielman 2013 原文、SE-Sync（IJRR 2019）本次没有打开，只通过二手引用确认。

---

## 5. learned prior 与 GBP / factor graph 的结合（learned factors、neural potentials）用于建图

### Takeaway
learned prior 进入 factor graph 有三种成熟形态：
1. **learned latent manifold。** CodeSLAM 和 DeepFactors 把稠密深度写成"图像 + 低维 code"的解码，在标准 factor graph 中联合优化 code 和位姿。
2. **learned factor / potential。** Nabarro et al. 2024 把 CNN 层写成 GBP 因子；FGNN 从数据学习因子势；Yi et al. 2021 学习可微 factor graph smoother 中的噪声与动力学模型。
3. **learned message correction。** Neural Enhanced BP 在 BP 旁边并联一个 GNN 来修正消息。

把 learned Gaussian code 在多机之间用 GBP 融合的唯一近邻仍然是 DistGP（ICRA 2026），见前一份笔记。没有找到"learned code + 多机 + 处理公共先验重复计数"的工作。

### Cited Findings
- CodeSLAM（Bloesch et al., CVPR 2018）：每个 keyframe 的深度由图像和一个学到的紧凑 code 共同决定，于是可以在重叠的 keyframe 之间联合估计位姿和稠密深度。DeepFactors 的相关工作部分指出它是窗口式、非实时的，也不能泛化到手持视频。— [arXiv 1804.00874](https://arxiv.org/pdf/1804.00874); [DeepFactors arXiv 2001.05049](https://arxiv.org/pdf/2001.05049)
- DeepFactors（Czarnowski et al., RA-L 2020）：在 CodeSLAM 的 code 表示上补齐了完整 SLAM，包括回环、重定位和全图批优化。photometric、reprojection 和 geometric 误差都写成标准 factor graph 因子，保证实时。— [arXiv 2001.05049](https://arxiv.org/pdf/2001.05049); [Imperial 项目页](https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors)
- Nabarro et al.（ICML 2024）：参数的 BP 后验边缘作为下一个任务的先验（continual learning）。数值和局限见第 3 节。— [arXiv HTML](https://arxiv.org/html/2311.14649v3)
- Neural Enhanced Belief Propagation（Garcia Satorras & Welling, AISTATS 2021）：一个 factor-graph GNN 与 BP 并行运行，接收 BP 消息并修正输出，在模型近似或图有环时有帮助（检索摘要）。— [PMLR PDF](https://proceedings.mlr.press/v130/garcia-satorras21a/garcia-satorras21a.pdf)
- Factor Graph Neural Networks（Zhang, Wu, Lee, NeurIPS 2020；JMLR 2023）：从数据学习因子势，可以精确表示 max-product BP，在真实势未知时学习近似（检索摘要）。— [NeurIPS PDF](https://proceedings.neurips.cc/paper/2020/file/61c66a2f4e6e10dc9c16ddf9d19745d6-Paper.pdf); [JMLR](https://www.jmlr.org/papers/volume24/21-0434/21-0434.pdf)
- "Differentiable Factor Graph Optimization for Learning Smoothers"（Yi et al., IROS 2021）只在检索列表中出现，没有读正文。— [arXiv 2105.08257](https://arxiv.org/pdf/2105.08257)
- DynoSLAM（2026）把 socially-aware 生成式 GNN 直接放进 factor graph 优化，把行人运动建模为随机 world model（检索摘要）。— [arXiv 2605.02759](https://arxiv.org/html/2605.02759)
- 一篇 LLM 先验增强的实时室内物体 SLAM 在 TUM RGB-D 和 3RScan 上报告建图精度提升 36.8%（检索摘要）。— [arXiv 2509.21602](https://arxiv.org/html/2509.21602)
- van Kempen et al. 2023 用 DNN 直接学"配准 + 融合"两个 evidential OGM（见第 2 节）。— [arXiv 2304.03578](https://arxiv.org/abs/2304.03578)
- 以 "diffusion prior + factor graph SLAM" 检索没有得到直接论文。最接近的是 UnPose（2025），它用不确定性引导的 diffusion prior 做零样本位姿估计。— [arXiv 2508.15972](https://arxiv.org/pdf/2508.15972)

### Inferences
- **对项目的 learned Gaussian code（64 均值 + 64 对角精度）。**
  - 若各 agent 的 encoder 输出的是"后验"（已包含 N(0, I) 先验），则多机求和会把先验计 N 次。这就是 Bailey 等所说的公共先验 double count，也是 independent opinion pool 的"极端强化"问题。
  - 正确的 GBP 写法是：先验作为挂在 code 变量上的一个 unary 因子，只出现一次；每个 agent 发送似然因子，即 posterior ÷ prior，在 information 形式下是 Λ_post − Λ_prior 和 η_post − η_prior。
  - 这一点在 max-selection 下不出现（幂等），但切换到求和后必须处理。
- **code 空间求和 ≠ voxel 空间求和。** 解码器是非线性的，所以 code 空间的 product-of-Gaussians 不等于对解码后 log-odds 求和。如果关心的是 surface/furniture 的 F1，可以考虑两种做法：在 voxel log-odds 空间做求和融合，再用 learned prior 作为单独的 unary/pairwise 因子（DeepFactors 式）；或者验证 code 空间求和在内部 probe 上的增益是否与 voxel 空间一致。
- **learned prior 与噪声。** Nabarro 和 FGNN 的经验是，学到的因子需要鲁棒核（类 Tukey）和 damping 才能在环图上稳定。learned prior 应当在"降低 range noise 假阳性"的方向上帮助求和（相当于更好的 inverse sensor model），而不是替代多机证据的累积。van Kempen 的学习式融合说明网络可以"吸收"大错位，但代价是不可解释、不估计位姿。

### Gaps
- 没有找到在多机设定下用 GBP 融合 learned latent code 并处理公共先验与 double counting 的论文。DistGP 是最近邻，但它是 GP inducing point，位姿已知。
- Ortiz et al. ICRA 2022（Incremental Abstraction）、iMAP/NICE-SLAM 一类神经隐式 SLAM 与 factor graph 的结合，本次都没有核实。
- 没有找到用 diffusion / score-based 先验作为 factor graph 因子做 occupancy 建图的论文。只确认了位姿估计方向的 UnPose，不确定是否已有 2026 年的工作。
