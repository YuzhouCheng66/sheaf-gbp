# 生成先验退居幕后，配准决定成败

文献整体支持项目的内部结论：八个可检验的判断里七个站得住，只有"学习熵模型能把每机消息从 200–260 B 再压到 130–180 B"这一条偏乐观。第一，**少步生成模型（MeanFlow、consistency、shortcut）根本不给似然，不能充当 arithmetic coding 的熵模型**。连续 diffusion 只给 ELBO，在 CIFAR-10 上 10 步时码长比无穷步差 63%；bits-back 用于单条消息时，开销在 ARDM 的表里记为 ∞。真正能当"生成式熵模型"的是固定顺序、组内并行的离散分解，ARDM 从 3072 步降到 50 步只多约 2% bpd。第二，**后验采样在理论上让失真翻倍**。扩散 SSC、扩散分割和 SceneSense 的实证都显示采样输给判别式或边缘阈值化，项目内部"采样比阈值化低 6–22 个家具 F1 点"正是这个规律，与模型好坏无关。生成器因此只能做三件事：做熵模型，做输出边缘概率的去噪器，或做一个打上"推断"标记、不计分的补全层。第三，**在无损设定下对标 MPEG G-PCC，学习熵模型自报的增益是 4%–52%**：致密表面约三成，稀疏点云约一成，没有一篇超过约 2.1 倍。文献里所有"几十倍"的数字都来自改变"发什么"，比如以 float32 特征为分母、OctoMap 从 .ot 换成 .bt（19–42 倍）、或者不传 free space。所以协议因子和编码因子必须分开报告；"相对最强无损几十倍"在全保真干净世界里不可能。第四，**有位姿噪声时，决定精度的是显式配准校正，而不是融合规则**。OPV2V 上 0.6 m/0.6° 噪声下，带位姿图校正的 CoAlign 的 AP@0.7 仍有 0.868；只传不到 0.3 KB keypoint 的 FPV-RCNN 跌到 0.200。log-odds 求和在树上等价于 channel filter，也就是树上精确的 BP，是独立证据下的最优规则。文献的空白正好落在项目要做的地方：没人在多机占据上定量比较过 sum、max 和 CI；没人做过 cm/° 级位姿噪声对占据 F1 的受控扫描；没人把 SE(3) 配准写成与地图 sheaf 耦合的 sheaf；协同感知压缩也没人用强无损几何码作基线。

## 八个内部判断七个站得住，只有学习增益的预期偏乐观

项目内部结论可以拆成八个可检验的判断，下表逐一对应文献中最硬的证据。这套结论能成立，靠的是三个文献社区各自的惯例，项目各取了一部分：

| 社区 | 惯例 | 项目借用的部分 |
|---|---|---|
| 点云几何压缩（PCC） | 以 G-PCC 这样的强无损编码器为锚，学习增益通常为 5%–50% | 码率怎么记账 |
| 协同感知 | 以未压缩 float32 特征为分母，倍数为 32 倍到 10 万倍；只在位姿上加噪，按干净真值评分 | 噪声协议 |
| 生成建模 | 优化 FID 一类分布指标，不看逐体素精度；rate–distortion–perception 理论给出采样的代价 | 生成器该放在哪里 |

| 内部判断 | 文献裁决 | 最硬的证据 | 需要补的限定 |
|---|---|---|---|
| 干净世界以精确几何码（平面 + 盒 + 光线序残差 + 位姿）作参照线 | 支持 | 无损码在它描述的体素上失真为 0，perception–distortion 权衡不出现；同等字节预算（<153 B）下，生成式图像编码只能保证"像" ([PerCo](https://arxiv.org/abs/2310.10325)) | 参照线要同时对标 OctoMap .bt 和 G-PCC，并注明 TMC13 版本 |
| 生成先验不直接出图 | 支持 | 后验采样使失真翻倍 ([Blau & Michaeli 2019](https://arxiv.org/abs/1901.07821))；从噪声生成的 SSC 为 30.5 mIoU，判别式为 36.2–36.7 ([GSSC](https://huggingface.co/papers/2608.26737)) | 去噪器输出边缘概率；补全另开通道，不计入 F1 |
| 生成先验做熵模型 | 有条件支持 | 只有离散分组分解给出精确码长；MeanFlow 原文不涉及似然 ([MeanFlow](https://arxiv.org/html/2505.13447v1)) | "生成先验"必须是似然模型；8 台机器之间概率须逐位一致 |
| 学习熵模型把 200–260 B 压到 130–180 B（−31% 到 −35%） | 偏乐观 | 致密表面 −29% 到 −36%，稀疏点云 −4% 到 −10%，并行化后 −17.5% ([VoxelDNN 扩展版](https://arxiv.org/html/2107.00400)) | 预注册门槛定为 ≥10–15%，两个 seed 都过 |
| 加固基准：测距噪声、位姿与机间配准误差、按干净真值评分 | 支持，且填补空白 | 协同感知的惯例就是干净 GT、只在位姿上加噪 ([CoAlign](https://arxiv.org/html/2211.07214))；没有人对占据 F1 做过 cm/° 级扫描 | 噪声尺度要比 V2X 默认的 0.2 m/0.2° 细，并加 6-DoF 噪声与外点 |
| 森林上的 log-odds 求和 | 支持，前提要写明 | 条件独立时求和最优；channel filter 在树上精确，等价于 junction-tree BP ([Makarenko 2009](https://dellaert.github.io/files/Makarenko09fusion.pdf)) | 公共先验只计一次；相关误差下需 1<Σω<2 或截断 |
| 位姿 sheaf 配准体现 sheaf-GBP 的价值 | 数学先例齐全，工程先例为零 | 同步问题即平坦主丛 ([Gao et al.](https://arxiv.org/abs/1610.09051))；位姿校正收回大部分噪声损失 ([CoAlign](https://arxiv.org/html/2211.07214)) | 用向量值 sheaf 而非格值 sheaf；必须带鲁棒核 |
| "相对 gzip 几十倍"拆成协议因子与编码因子；"相对最强无损几十倍"在全保真下不可能 | 强支持 | 文献中的"几十倍"全部来自改变发送内容；无损对标 G-PCC 的学习增益最高约 2.1 倍 ([SparseVoxelDNN](https://arxiv.org/html/2204.05043)) | "最强无损"必须已包含平面 + 盒这一表示 |

## 少步生成器给不出码长：熵模型只能用离散分组分解

熵模型只需要给出 q(x)，码长就是 −log₂q(x)，和样本质量无关。按这个标准，少步生成模型可以分成三档。

**第一档：连续 Gaussian diffusion。** 它只给 ELBO 上界，而且步数一减就崩。VDM 在 CIFAR-10 上无穷步为 2.65 bpd；离散 10 步训练并评估为 **4.31 bpd，多 63%**；连续时间训练的模型取 10 步评估达到 7.54 bpd ([VDM](https://arxiv.org/abs/2107.00630))。即使步数足够，实际编码还得走 bits-back：VDM 用 BB-ANS 在 100 步时实际码长为 2.91 bpd，与 ELBO 仍有差距 ([VDM](https://arxiv.org/abs/2107.00630))。更要命的是 bits-back 需要一段初始比特。ARDM 论文的表把 VDM 和 LBB 的"逐图压缩"码长都记为 **∞**，作者明说这类方法用作逐图压缩器时"have a large overhead" ([ARDM](https://arxiv.org/abs/2110.02037))。项目每台机器人每次只发一条消息，正好落在这一列。

**第二档：flow matching。** 它的似然要靠 uniform dequantization 加 Hutchinson 随机迹估计算出来 ([Flow Matching](https://ar5iv.labs.arxiv.org/html/2210.02747))，不是确定的 PMF，接收端无法逐位复现。

**第三档：MeanFlow、sCM、shortcut 等一步模型。** MeanFlow 原文只讨论一步采样（ImageNet 256 上 1-NFE FID 3.43），完全没有涉及似然、密度或压缩 ([MeanFlow](https://arxiv.org/html/2505.13447v1))；sCM 和 shortcut 也一样。这类模型在熵模型这个角色上贡献为零。它们最多能当特征提取器，后面再接一个用交叉熵训练的逐体素概率头，而那时真正起作用的是这个判别式的头。

能用的是离散分解 p(x)=∏ₖ∏_{i∈Gₖ}p(xᵢ|x_{G<k}, context)：分组顺序固定，组内各体素条件独立。它的码长精确，步数减少时退化也平缓。ARDM 在 CIFAR-10 上从 3072 步降到 50 步，码长从 **2.69 变为 2.74 bpd，只多约 2%**；逐图压缩时 100 步为 2.75 bpd，而且不需要 bits-back ([ARDM](https://arxiv.org/abs/2110.02037))。图像压缩工程界已经在用同一结构。checkerboard context model 两遍解码，速度提高 40 倍以上，率失真性能"almost the same" ([Checkerboard](https://arxiv.org/abs/2103.15306))。M2T 发现，用预定义的确定性 schedule 驱动 masked transformer 熵模型，效果不比自适应 schedule 差，还能快约 4 倍 ([M2T](https://arxiv.org/abs/2304.07313))。SparsePCGC 的 SOPA 在点云占据上做多尺度分组 ([SparsePCGC](https://arxiv.org/abs/2111.10633))。VAR 的 next-scale 结构可以借鉴，但它算的是有损 VQ token 的似然，不是体素本身的似然 ([VAR](https://arxiv.org/html/2404.02905v2))。

落到 128×128×32 的三值栅格上，现成的候选有三个：2×2×2 奇偶八分组（K=8）；建在无损 max-pool 占据金字塔上的 octree 式多尺度模型；固定 schedule 的 masked diffusion。离散 diffusion 家族里只有最后一种能进入熵模型角色，而它本质上就是 ARDM。训练成本也说明没必要上完整 diffusion：PDD 称原始离散扩散在 256×256×16 上要 4 张 A100 训练 16 天以上 ([PDD](https://arxiv.org/html/2311.12085v2))。分组交叉熵模型只需要 K 次前向，训练成本应与同尺寸的判别式 SSC 网络相当（推断）。

多机互通还有一条硬约束：range coding 要求收发两端算出的概率逐位一致。Ballé、Johnston、Minnen 指出，跨硬件或软件的任何浮点舍入差异都会让解码灾难性失败，并为此提出 integer networks ([Integer Networks](https://research.google/pubs/integer-networks-for-data-compression-with-latent-variable-models/))。后续工作发现，带 context model 的复杂模型很难在 integer network 下保持性能，于是改用 post-training quantization ([PTQ for LIC](https://arxiv.org/pdf/2202.07513))，或者只量化关键输出并附带 safeguarding bitstream ([Reproducible LIC](https://arxiv.org/pdf/2410.09872))。只要 8 台机器人里有两种计算平台，"异构设备逐位一致解码"就必须写进预注册判据。

另一个低成本的保险是 1 bit escape 标志：每条消息在学习编码和手写编码之间取较小者，把先验失配时的最坏代价限定在约 1 bit。这是工程推断，文献里没有找到针对这一做法的数字。先验失配本身只会多花比特，不会改变重建；例如 VoxelDNN 在 Arco Valentino 上比 G-PCC 多用 3.7% ([VoxelDNN 扩展版](https://arxiv.org/html/2107.00400))。

## 采样输给阈值化是理论上的必然：生成器应放在码流之外

理论上，这件事有明确答案：

- Blau 与 Michaeli 2018 年证明，失真与感知质量不可兼得 ([Perception–Distortion](https://arxiv.org/abs/1711.06077))。
- 他们 2019 年的 rate–distortion–perception 构造表明，追求完美感知只需让失真"翻倍" ([RDP](https://arxiv.org/abs/1901.07821))。
- Yan 等证明，在没有共同随机性时，完美感知的代价恰好是最低可达 MSE 的两倍 ([Yan et al.](https://arxiv.org/abs/2106.02782))。
- Ohayon、Michaeli、Elad 的 PMRF 指出，后验采样的 MSE 恰为 MMSE 的两倍，并给出更好的做法：先取后验均值，再做最优传输 ([PMRF](https://arxiv.org/html/2410.00418v3))。

落到二值体素上（以下为推导）。设接收端对某体素的后验占据概率为 p：取阈值 0.5 时期望错误率是 min(p,1−p)，后验采样是 2p(1−p)。在接近确定的区域，两者之比趋近 2。对 F1，由期望的线性性，即使从真实后验做联合采样，E[TP] 也只有 Σp²，所以 **F1_sample ≈ Σp²/Σp**。举例：一批家具体素的占据概率都是 0.6，采样的期望 F1 是 0.60，全部判为占据是 0.75，差 15 点。

最大化 F1 的最优阈值等于最优 F1 的一半 ([Lipton et al.](https://arxiv.org/abs/1402.1892))。项目内部"采样比边缘阈值化低 6–22 个家具 F1 点"正落在上面推导给出的区间里。差距来自决策规则与指标不匹配，换一个更好的生成模型也消除不了。

实证与理论一致：

- Wolleb 等的扩散分割在 BRATS2020 上，单次采样 Dice 0.866，5 次采样平均后再阈值化 0.881，判别式 nnU-Net 0.891。单个样本耗时 48 s，判别式只要 13 ms ([Wolleb et al.](https://ar5iv.labs.arxiv.org/html/2112.03145))。
- GSSC 从噪声生成的条件模型 mIoU 为 30.5，判别式 base 为 36.2–36.7 ([GSSC](https://huggingface.co/papers/2608.26737))。
- DiffSSC 的几何 IoU 63.4 是表中最高，但 mIoU 27.4 比 TALoS 的 37.9 低约 10 点 ([DiffSSC](https://arxiv.org/html/2409.18092v3))。
- SceneSense 的一个厨房例子中，IoU 0.52 低于直接用局部占据的 0.61 ([SceneSense](https://arxiv.org/html/2403.11985v1))。
- DiffC 中，ancestral sampling 比 probability-flow 重建低约 3 dB，这正是"失真翻倍"在 PSNR 上的体现 ([DiffC](https://arxiv.org/abs/2206.08889))。

"压缩再生成"在图像领域已经做到了项目的字节量级。PerCo 在 0.003 bpp 下，一张 512×768 的 Kodak 图**不到 153 B** ([PerCo](https://arxiv.org/abs/2310.10325))，恰好落在项目 130–260 B 的消息预算内。但它的 PSNR、MS-SSIM、LPIPS 都明显差于失真导向的编码器（二手转述）([PerCo ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/file/43e9d647ccd3e4b7b5baab53f0368686-Paper-Conference.pdf))。码率再往下，问题更明显：

- 低于约 0.05 bpp 时出现"semantic deviation"，例如招牌上的文字被改写 ([2512.06344 摘要](https://awesomepapers.io/generative-models/papers/2512.06344))。
- 在 0.001–0.008 bpp 区间，代表性编码器不是平滑退化，而是突然"semantic collapse"，生成畸形或无法辨认的内容 ([Rethinking GIC](https://arxiv.org/abs/2609.39315))。
- 一步扩散编码器 OneDC 声称码率降低 39% 以上、解码快 20 倍 ([OneDC](https://arxiv.org/html/2505.16687v2))，但这个收益是在感知指标下取得的。

把这类方法搬到占据地图上，一把被"合理地"挪了位置的椅子，会同时贡献 FP 和 FN。精确几何码则相反：在它描述的体素上 X̂=X，失真和真实感同时为零，RDP 权衡根本不出现。权衡只出现在两处：接收端用光线投射推出的 free space，以及未观测区。

文献给生成器留下四个位置，都在码流之外或旁边：

1. **熵模型。** 收发两端各持一份，只输出概率，不改变重建。
2. **输出边缘概率的去噪器。** GSSC 的 S²D² 以冻结判别网络的 argmax 为起点，只做一次 Euler 更新，SemanticKITTI hidden test 的 mIoU **从 36.7 升到 38.8**；1 步与 100 步的差异在单 seed 噪声以内，每次耗时 107 ms。但在弱 base（JS3C-Net）上，person 类 IoU 在 74.7% 的帧上下降，bicyclist 类在 92.5% 的帧上下降（2026 预印本，摘要页数字）([GSSC](https://huggingface.co/papers/2608.26737))。这条证据直接要求把"稀有家具类的 F1 不降"写进预注册门槛。去噪先验进入融合时，可参照 Achey 等的权重惯例：他们把扩散预测作为单独的 log-odds 项，并明确设定 P(m|d) < P(m|z)，理由是"我们更信任传感器" ([Achey et al.](https://arxiv.org/html/2506.20049v2))。
3. **显式标记为"推断"的补全层。** 它不进 F1，也不进 frontier 判定（下一节有失败案例）。如果要做，Hamdi、Gündüz、Wagner 的结果提示：接收端的侧信息和共享随机种子都能降低追求真实感的代价 ([Hamdi et al.](https://arxiv.org/abs/2507.14825))，而多机系统天然具备这两样。
4. **合成训练数据。** GSSC 用同一个扩散模型造出了 32,039 对配对数据 ([GSSC](https://huggingface.co/papers/2608.26737))。

MeanFlow 一类的一步模型在这里最多能当后验均值的近似器或数据生成器。把它用于一步后验采样的 NullFlow 等工作都还是 2026 年的预印本，没有标准基准上的头对头比较 ([NullFlow](https://arxiv.org/pdf/2606.22696))。

## 三维场景生成先验：当细化器有小收益，直接出图会产生幻觉

2023–2026 年的三维扩散补全工作呈现同一模式：分布指标上赢，逐体素指标上输。

- SSD 在 CarlaSC 128×128×8 上，相对同架构的判别式基线 mIoU 提高 5.89（39.94 → 45.83），IoU 却低 0.09，也没有超过 MotionSC 的 46.31 ([SSD](https://arxiv.org/html/2301.00527v1))。
- LiDiff 在 0.2 m 体素上 IoU 为 33.12，refine 后 38.92，**低于判别式 LODE 的 47.88**；只有到 0.1 m 才领先（24.75，MID 为 16.98）([LiDiff](https://arxiv.org/html/2403.13470))。

生成先验最有价值的用法是当细化器，而且输入越弱，收益越大：

| 被细化的 base | IoU（细化前 → 后） | mIoU（细化前 → 后） |
|---|---|---|
| MonoScene（RGB 输入） | 37.12 → 50.44 | 11.50 → 17.08 |
| SSA-SC（LiDAR 输入） | 58.25 → 60.71 | 24.54 → 25.58 |
| SCPNet（LiDAR 输入） | — | 只多 0.64 |

（数据来源：[SemCity](https://arxiv.org/html/2403.07773)）

对项目而言，这意味着生成先验在覆盖稀疏、单机观测弱的区域可能有正收益，在 8 机充分观测的区域收益趋近于零（推断）。

这些论文的评估惯例本身就不适合项目：

- PDD 先用分割网络给生成场景打标，再算 mIoU(V)，得到 68.02，**高于真值场景自身的 52.19**。可见这个指标衡量的是"好不好分割"，不是保真度 ([PDD](https://arxiv.org/html/2311.12085v2))。
- LT3SD、XCube 这些大场景生成器只报告 COV、MMD、1-NNA、FID 等分布指标，补全只给定性图 ([LT3SD](https://arxiv.org/html/2409.08215v2); [XCube](https://arxiv.org/html/2312.03806v2))。
- 与项目最接近的 Octree Latent Diffusion 恰好用 Replica 室内数据、0.05 m 体素、**128×128×32 网格**，但只报告"给定真值粗占据"条件下的语义 mIoU 32.22%，没有补全 IoU ([Octree LD](https://arxiv.org/html/2509.16483v1))。
- SceneSense 直言 IoU 会"重罚猜测的几何"，于是改报 FID/KID ([SceneSense](https://arxiv.org/html/2403.11985v1))。
- 没有一篇扩散 SSC 论文报告体素级校准（ECE），也没有报告样本方差与误差的相关性。

记忆化需要单独检查。PDD 生成场景与最近训练场景的 SSIM 为 0.72 和 0.74，高于验证集与训练集之间的 0.65 和 0.67 ([PDD](https://arxiv.org/html/2311.12085v2))。项目的参数化家具库分布更窄，"背模板"的风险更高（推断）。

机器人建图里记录最清楚的失败来自 SceneSense 的后续工作。Achey 等把 frontier 处的生成预测写进规划用的地图后，预测"erases the boundary between free and unknown"，规划器"prematurely concludes that the environment is fully explored"，该配置因此达不到 95% 覆盖。作者在局限部分承认："a single, highly confident but incorrect prediction can still adversely affect the map's quality" ([Achey et al.](https://arxiv.org/html/2506.20049v2))。Map Space Belief Prediction 把"物体不存在却被报告为存在"定义为 Hallucinated 类别，三种方法都持续出现这类错误 ([2502.20606](https://arxiv.org/pdf/2502.20606))。能给出保证的只有 conformal 路线：PwC 对 scene completion 做 conformal 校准，再配合非确定性滤波，使预测的 free 空间以高概率不与真实占据相交，端到端碰撞概率不超过 ε ([PwC](https://arxiv.org/html/2403.08185v3))。SceneSense 的 occupancy inpainting 在每一步把已观测体素重新注入，因此生成结果永远不会覆盖观测 ([SceneSense](https://arxiv.org/html/2403.11985v1))。

综合起来，观测与预测需要三层隔离（推断）：生成时硬约束不改观测；存储时分通道或降权；使用时 coverage 与 frontier 只读观测通道。项目"不直接出图"的设计同时满足这三层。

还有一条值得借的线索是对象级表示。DiffuScene 把场景表示为最多 13 个对象的集合，每个对象是 [location, size, cos θ, sin θ, class, shape code]，并用 pairwise box IoU（PIoU）衡量物体穿插 ([DiffuScene](https://arxiv.org/html/2303.14207))。ATISS 只用带标签的 3D bounding box 训练 ([ATISS](https://research.nvidia.com/labs/toronto-ai/publication/neurips_2021_atiss))；LEGO-Net 在同一表示上做"去噪式重排" ([LEGO-Net](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_LEGO-Net_Learning_Regular_Rearrangements_of_Objects_in_Rooms_CVPR_2023_paper.pdf))。这与项目的"平面 + 盒"几何码同构，而且对象 token 加离散家具库索引，比 LT3SD 每个房间约 33 KB 的 latent tree 紧凑得多 ([LT3SD](https://arxiv.org/html/2409.08215v2))。所以如果要学先验，最该学的是盒参数的分布（类别、尺寸、朝向、与墙的关系），而不是体素 latent（推断）。

## 学习熵模型只比强手写编码再省一到三成，"几十倍"全部来自协议

### 自然图像：每加一层学习，只多省一成左右

在自然图像上，学习式熵模型的递进增益都在一成左右：

- 在 hyperprior 上加空间自回归上下文，平均文件大小 −15.8% ([Minnen et al.](https://arxiv.org/abs/1809.02736))。
- channel-wise 熵模型相对 context-adaptive 基线，Kodak −6.7%，Tecnick −11.4% ([Minnen & Singh](https://arxiv.org/abs/2007.08739))。
- Contextformer 相对 VTM 最多 −11% ([Contextformer](https://arxiv.org/abs/2203.02452))。

VQ 与扩散编码器在低码率下的数字都是在感知指标下取得的：

- GLC 在 CLIC 2020 上达到与 MS-ILLM 相同的 FID 时，少用 45% 码率 ([GLC](https://arxiv.org/abs/2505.16177))。
- MS-ILLM 达到与 HiFiC 相同的 FID 时，少用 30–40% 比特 ([MS-ILLM](https://arxiv.org/abs/2301.11189))。
- VQGAN 方案在 ≤0.04 bpp 时丢掉 20% 的索引仍能恢复 ([VQGAN](https://arxiv.org/abs/2307.08265))。

在失真或精确度量下，这些方法相对手写上下文编码器能省多少，文献没有给出。

### 二值 3D 占据：对标 G-PCC 时最高约 2.1 倍

与项目最可比的设定是"二值 3D 占据 + 手写邻域上下文算术编码"，也就是 MPEG G-PCC 的 octree 几何编码 ([Nguyen et al.](https://arxiv.org/html/2107.00400))。

**VoxelDNN 扩展版（对标 G-PCC）** ([VoxelDNN 扩展版](https://arxiv.org/html/2107.00400))：

| 数据集类型 | 数据集 | 码率变化 |
|---|---|---|
| 致密 | MVUB | −29.01% |
| 致密 | 8i | −35.77% |
| 稀疏 | CAT1 | −3.72% |
| 稀疏 | USP | −9.91% |
| 单个样本 | Arco Valentino | **+3.70%**（变差） |

代价是解码比 G-PCC 慢 3 个数量级。作者还写道，块不够稠密时码率会"rapidly grows by an order of magnitude"。

**其他学习方法：**

- MSVoxelDNN 用分组并行换来两个数量级的解码提速，增益降到 −17.5% ([MSVoxelDNN](https://arxiv.org/abs/2104.09859))。
- SparseVoxelDNN 自报四个数据集平均 −52%，**约合 2.1 倍，是本次找到的最大无损增益** ([SparseVoxelDNN](https://arxiv.org/html/2204.05043))。

**8iVFB 上的 bpp 与 G-PCC 版本：**

| 方法 | bpp | 对照的 G-PCC 版本与 bpp | 相对变化 |
|---|---|---|---|
| ECM-OPCC ([ECM-OPCC](https://arxiv.org/abs/2211.10916)) | 0.58 | v14，0.99 | −41.9% |
| UniPCGC ([UniPCGC](https://arxiv.org/abs/2503.18541)) | 0.51 | v23，0.72 | −29.2% |

同一序列上，G-PCC 自己从 v14 到 v23 就降了约 25–30%，所以"相对 G-PCC"必须注明版本。

**稀疏 LiDAR：**

- SparsePCGC 自报：KITTI 2 cm −19.7%，Ford 2 cm −12.4%，KITTI 1 mm −10.8%，Ford 1 mm −5.1% ([SparsePCGC](https://arxiv.org/abs/2111.10633))。
- 第三方重跑 SparsePCGC，只剩 −2% 到 −7% ([NPA](https://arxiv.org/abs/2208.12573))。
- 实时编码器 RENO 相对 G-PCC v23 约 −12.3% ([RENO](https://arxiv.org/abs/2503.12382))。

室内最接近的证据是 VoxelContext-Net 在 ScanNet 上的 −43.66%，但那是有损的截断 octree ([VoxelContext-Net](https://arxiv.org/abs/2105.02158))。

### 对照项目：−31% 到 −35% 只能当乐观上限

项目内部，手写 3D 上下文编码为 **1,384 B，gzip 为 2,733 B**，即约 0.021 对 0.042 bit/体素，省 49%。BB-ANS 在二值 MNIST 上相对 gzip 省约 42%（0.19 对 0.33 bits/dim，二手转述）([BB-ANS](https://arxiv.org/abs/1901.04866))。可见"上下文建模对通用 LZ"这一档的增益，手写编码已经拿到了。学习先验要争的是下一档，而文献在这一档的数字是：致密表面三成上下，稀疏一成左右，并行化后不到两成。

一条 200–260 B 的消息只有几百个符号，消息内部可用的上下文很少，正是 VoxelDNN 作者指出的卷积上下文变弱的区间。学习增益只能主要来自场景统计，也就是墙和家具形状的分布。位姿比特几乎不受占据先验影响，还会稀释整条消息的相对增益。因此"130–180 B"（−31% 到 −35%）只能当乐观上限。预注册门槛宜定为：在留出房屋上，两个 seed 都相对"同一几何码 + 手写上下文编码器"稳定省下 ≥10–15%。基线必须是"手写编码作用在同一几何描述上"，否则会把上下文建模本身的增益算到学习先验头上。

### 噪声给码率设了地板

噪声给码率设了一道学习也越不过的地板。

- **精度越细，码率线性上涨，学习增益同步缩小。** KITTI 上量化精度从 2 cm 细到 1 mm（多 4 bit），G-PCC 无损码率从 7.637 涨到 20.154 bpp，SparsePCGC 从 6.130 涨到 17.971 bpp。两者每多 1 bit 精度都多约 3 bpp，学习增益同时从 −19.7% 缩到 −10.8% ([SparsePCGC](https://arxiv.org/abs/2111.10633))。
- **采集噪声越大，码率越高。** 采集更嘈杂的 MVUB 比 8iVFB 贵 10–20%，学习增益从 −42.4% 缩到 −33.6% ([ECM-OPCC](https://arxiv.org/abs/2211.10916))。

i.i.d. 噪声的熵不可压缩（推导）。在 6–16 cm 的网格上，cm 级测距噪声大多被量化吸收，剩下的是表面增厚和孤立伪击中。对命中集先做"多次观测才确认、剔除孤立体素"这类去噪，省下的码率可能比换一个更强的熵模型还多。但那是有损操作，必须用下游 F1 来评判。反过来，若拿带隐式去噪的学习编码器去比严格无损的命中集，所谓"增益"里有相当一部分其实只是删掉了噪声（推断）。

### 深度生成模型做无损压缩，上限也只有几倍

用深度生成模型做无损压缩，上限也只有几倍。Delétang 等 v2 版表格中，enwik9 上 gzip 为 32.3%，LZMA2 为 23.0%，Chinchilla 70B 为 8.3%，即**对 gzip 约 3.9 倍、对 LZMA2 约 2.8 倍**；把 70B 参数计入码长后，adjusted rate 高达 14,008% ([Delétang et al.](https://arxiv.org/html/2309.10668v2))。在 enwik8 上训练的小 Transformer 遇到 ImageNet 时，输出变成原始大小的 194%–216%，也就是分布外直接膨胀 ([Delétang et al.](https://arxiv.org/html/2309.10668v2))。多机系统里模型预先部署、两端共享，参数开销可以摊销；但分布外崩溃无法摊销，所以 escape 回退是必要的。

### 文献里的大倍数都是协议因子

下表把文献中的倍数按来源分类。

| 倍数来源 | 文献实例 | 倍数 | 分母 | 类型 |
|---|---|---|---|---|
| 全概率 .ot → 最大似然 .bt | FR-079 5 cm：15.76 → 0.67 MB；freiburg1_360 2 cm：21.59 → 0.52 MB ([OctoMap](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf)) | 19–42 倍 | 同一地图的概率文件 | 协议（丢概率、剪枝） |
| 不传 free space | .bt 外推到同等观测范围约 1.7–4 KB → 命中集 + 位姿 200–260 B | 7–20 倍（推算） | OctoMap .bt | 协议 |
| 参数化子图 | GMM 子图 4,138 B，点云子图 95,836 B ([MR-GMMExplore](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/3f18206c-0703-4ad6-9e46-1d61efbca467.pdf)) | 约 23 倍 | 点云子图 | 表示 |
| 中间特征压缩 | OPV2V 4096 倍时只掉约 3% ([OPV2V](https://arxiv.org/html/2109.07644))；Where2comm 超过 10 万倍 ([Where2comm](https://arxiv.org/abs/2209.12836)) | 32 倍至 10⁵ 倍 | 未压缩 float32 特征 | 协议 |
| 码本索引 | CodeFilling 仅换成索引表示就 208 倍，总计 1,333 倍 ([CodeFilling](https://arxiv.org/html/2405.04966)) | 208–1,333 倍 | Where2comm | 表示 |
| 上下文编码对通用 LZ | 项目 1,384 B 对 2,733 B；BB-ANS 0.19 对 0.33 | 1.7–2.0 倍 | gzip | 编码 |
| 学习上下文对标准手写上下文 | 对标 G-PCC：−4% 到 −52%，个别为正 | 1.04–2.1 倍 | G-PCC | 编码 |
| 70B LLM 对最强通用无损 | enwik9：8.3% 对 23.0% | 2.8 倍 | LZMA2 | 编码 |

规律很清楚：**凡是十倍以上的数字，分母都换成了另一种"发什么"；凡是在同一内容上比较编码器，倍数都落在 1–4 倍之间**。所以"相对 gzip 几十倍"应写成协议因子 × 表示因子 × 编码因子三项连乘，每一项都注明分母。

"相对最强无损几十倍在全保真干净世界不可能"这一判断成立，但有一个前提：最强无损基线必须已经包含平面 + 盒这一参数化表示。如果基线是体素级的 G-PCC，换成参数化表示照样能拿到几十倍，只不过那是表示因子，不是学习带来的。还有一个反方向的风险：在程序化生成的干净世界里，学习模型可能直接"学会了生成器"，在同分布测试上拿到虚高的编码增益。Delétang 的分布外膨胀和 PDD 的记忆化信号都提示，这类增益必须在留出的生成参数或真实扫描上复核（推断）。

### 接收端已有的地图是最值得利用的上下文

学习熵模型最值得利用、也最少被讨论的上下文，是接收端已经有的地图。

- **两端共享上下文（conditional coding）。** DCVC 系列在视频上相对强经典编码器仍省约 20%–26%：DCVC 对 x265 veryslow 为 −26.0% ([DCVC](https://arxiv.org/abs/2109.15047))；DCVC-RT 对 VTM 为 −21.0%，A100 上 1080p 解码约 113 fps ([DCVC-RT](https://arxiv.org/pdf/2502.20762))。
- **只有解码端有侧信息。** 学习式立体图像压缩可省 10%–50% ([Ayzik & Avidan](https://arxiv.org/abs/2001.04753))。
- **增益强烈依赖协议。** DCVC-DC 相对 VTM 在一种测试协议下为 −16.1%，在另一种协议下为 +12.8% ([DCVC-FM](https://arxiv.org/html/2402.17414v1))。这提醒我们，项目里的增益会随重叠率剧烈变化。

对约 150 B 的无损短消息，Slepian–Wolf 的"渐近无损失"在短块长下未必成立。更稳妥的做法是：用确认机制把已同步的公共历史变成两端共享的上下文，再做条件熵编码（推断）。

## 命中集加位姿：最小的早融合消息，也最怕配准误差

### 与几何编码比：小两到三个数量级

三维几何压缩的公开数字都来自 10⁵–10⁶ 个点的大点云：

| 编码方式 | 数据与精度 | 码率 | 每帧大小 |
|---|---|---|---|
| G-PCC octree 无损（TMC13 v14） | 稠密 10-bit 人体点云 | 平均 0.952 bpp | — |
| G-PCC octree 无损 | KITTI，2 cm | 7.637 bpp | 约 115 KB（按每帧约 12 万点推算） |
| G-PCC octree 无损 | KITTI，1 mm | 20.154 bpp | — |
| RENO（实时神经编码） | 12-bit | 约 2.2 bpp | 约 34 KB |
| RIDDLE（range image + 预测） | Waymo，Chamfer distance 约 0.005 | 3.65 bpp（G-PCC 为 10.78 bpp） | 不超过约 77 KB |
| RVL（深度图无损） | — | 每个有效像素至少 4 bit | — |

来源：G-PCC 数字见 [SparsePCGC](https://arxiv.org/abs/2111.10633)，RENO 见 [RENO supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/You_RENO_Real-Time_Neural_CVPR_2025_supplemental.pdf)，RIDDLE 见 [RIDDLE](https://arxiv.org/abs/2206.01738)，RVL 见 [Fischer et al.](https://dspace5old.zcu.cz/bitstream/11025/38439/1/Fischer.pdf)。

OctoMap 的最大似然二进制格式 .bt 每个内部节点 16 bit。FR-079 走廊在 5 cm 分辨率下为 0.67 MB，freiburg1_360 在 5 cm 下为 0.07 MB ([OctoMap](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf))，折算约 0.13–0.43 bit/包围盒体素（推算）。

把项目的数字放进来（以下均为推算）：

- 每台机器人 200–260 B，即 1,600–2,080 bit。摊到 524,288 个网格体素上约 0.003–0.004 bit/体素；摊到 5.2–7.9 万个观测体素上约 0.02–0.04 bit/观测体素。
- 按 .bt 的码率外推到同样的观测范围，约 1.7–4 KB，**是命中集 + 位姿的 7–20 倍**。差距主要来自 .bt 显式编码了 free space，而命中集让接收端用光线投射自己算出来。
- 比 LiDAR 帧编码小两到三个数量级，原因是那些编码保留厘米或毫米级的原始测量，而命中集只保留 6–16 cm 网格上的第一个击中体素。
- 设命中集有 N 个占据体素，则每体素约 1,600/N 到 2,080/N bit。N≈1,000 时约 1.6–2.1 bit，与 G-PCC 在致密表面上的 0.7–1.0 bpp 同一量级，学习熵模型还有空间；N≈300 时约 5–7 bit，头部开销、位姿和上下文不足会占主导。N 需要用项目自己的数据核对。

光线序残差的思路既有文献支撑，也有边界。RIDDLE 说明，在传感器坐标系里组织编码很有价值。但在高质量区间，range image 类方法反而比 octree 和 sparse 类方法差很多：RENO 对 R-PCC 和 RTST 的 BD-rate 分别为 −84.80% 和 −90.66% ([RENO](https://arxiv.org/abs/2503.12382))。6–16 cm 属于粗量化，落在光线序编码占优的一侧（推断）。另外，没有任何论文报告学习型编解码器在 10⁴ 点以下小点云上的 bpp，而这正是多机增量消息的规模。

### 多机 SLAM 实际传什么：配准证据和位姿图，几乎不传地图

主流系统先广播紧凑的全局描述子，匹配上后再按需发送局部特征或点云：

| 系统 | 传输内容与大小 | 来源 |
|---|---|---|
| D2SLAM | 完整 keyframe 20.1 kB（stereo）/ 56.3 kB（omni）；compact keyframe 1.2 / 4.8 kB | [D2SLAM](https://arxiv.org/html/2211.01538) |
| DCL-SLAM | LiDAR-Iris 描述子约 29.12 kB/帧；验证用点云约 99.92 kB；相对位姿约 104 B；每机平均发送 34–188 kB/s | [DCL-SLAM](https://arxiv.org/html/2210.11978) |
| COVINS（集中式） | 每个 keyframe 的 full 消息 97 kB；EuRoC 上行平均 493.36 kB/s | [COVINS](https://arxiv.org/html/2108.05756) |
| Kimera-Multi | 只传 DBoW2 向量，按需传 keypoints 和 public poses；mesh 由每台机器人用收到的位姿约束在本地变形，不传输 | [Kimera-Multi](https://arxiv.org/html/2106.14386) |
| Swarm-SLAM | 3 机真实实验总通信 94.95 MB，"mostly attributable to the front-end" | [Swarm-SLAM](https://arxiv.org/abs/2301.06230) |

这些论文都没有报告"地图精度随带宽变化"的曲线。项目的单条地图消息比一个 keyframe 描述子还小两个数量级，所以一旦引入在线配准，配准证据的流量很可能反超地图本身。

文献给出的出路是让紧凑几何同时服务于地图共享和配准。MR-GMMapping 只用 GMM 子图做相对位姿估计，数据传输比点云地图少 98%，平移和旋转精度还分别提高 11% 和 30% ([MR-GMMapping](https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2Fd375e8c2-f81f-451a-a5dc-efe2442b3af3.pdf))。平面 + 盒几何码天然是好的配准基元，不增加流量就能直接喂给位姿 sheaf（推断）。

### 协同感知：信息量属于 early，字节数比 late 还小

协同感知把消息分成三档。V2V4Real 的真实双车数据上：

| 融合方式 | 传输内容 | 每帧大小 | 同步时 AP@0.5 / AP@0.7 |
|---|---|---|---|
| late fusion | 检测框 | 0.003 MB | 55.0 / 26.7 |
| 中间融合（CoBEVT） | 32 倍压缩后的中间特征 | 0.20 MB | 66.5 / 36.0 |
| early fusion | 原始点云 | 0.96 MB | 59.7 / 32.1 |

（数据来源：[V2V4Real](https://arxiv.org/html/2303.07601)）

按信息内容，命中集 + 位姿属于 early 或地图级共享，但字节数比 late fusion 还小一个数量级。字节数上最接近的是 FPV-RCNN 的 keypoint 消息：不到 0.3 KB，约为 BEV 特征的 1/50 ([FPV-RCNN](https://arxiv.org/abs/2109.11615))。而它恰恰是对位姿噪声最脆弱的方法（见下一节）。

协同 3D 占据方面：

- CoHFF 稀疏到 0.99（约 97 倍压缩）时，IoU 从 50.46 降到 48.02，mIoU 从 34.16 降到 29.48，细类比整体占据更敏感 ([CoHFF](https://arxiv.org/html/2402.07635v2))。
- SAVQ 相对稠密 3D 表示最多省 82 倍 ([SAVQ](https://arxiv.org/abs/2607.01928))。
- MCOP 每次传输 0.23 MB，Where2comm 式基线为 17.50 MB ([MCOP](https://arxiv.org/html/2510.12679))。

### 量级排序

| 消息类型 | 实例 | 每条或每帧大小 |
|---|---|---|
| 相对位姿或回环测量 | DCL-SLAM ([DCL-SLAM](https://arxiv.org/html/2210.11978)) | 约 0.1 kB |
| **项目：命中集 + 位姿** | 8 机，128×128×32 | **200–260 B** |
| 协同检测 keypoint | FPV-RCNN ([FPV-RCNN](https://arxiv.org/abs/2109.11615)) | <0.3 KB |
| late fusion 检测框 | V2V4Real ([V2V4Real](https://arxiv.org/html/2303.07601)) | 约 3 KB |
| OctoMap .bt（外推到同等观测范围） | 推算 | 约 1.7–4 KB |
| GMM 子图（80 个分量） | MR-GMMExplore | 约 4.1 KB |
| compact keyframe | D2SLAM ([D2SLAM](https://arxiv.org/html/2211.01538)) | 1.2–4.8 kB |
| LiDAR 全局描述子 | DCL-SLAM LiDAR-Iris | 约 29 kB |
| 场景 latent | LT3SD 每房间 ([LT3SD](https://arxiv.org/html/2409.08215v2)) | 约 33 KB |
| LiDAR 帧神经编码 | RENO 12-bit（推算） | 约 34 KB |
| LiDAR 帧无损标准编码 | G-PCC 2 cm（推算） | 约 115 KB |
| 中间特征（32 倍压缩） | V2V4Real | 0.20 MB |
| 原始点云 early fusion | V2V4Real | 0.96 MB |

### 压缩倍数的分母五花八门

压缩倍数的分母在文献中极不统一，这正是项目必须拆分报告的原因：

- 协同感知的事实标准是 log2(字节)，字节数按"选中格子数 × 通道数 × float32"计算 ([Where2comm](https://arxiv.org/pdf/2209.12836))。
- OPV2V 的 4096 倍、V2X-ViT 的 32 倍和 128 倍，都是相对未压缩的中间特征 ([V2X-ViT](https://arxiv.org/html/2203.10638))。
- CodeFilling 的 1,333 倍是相对上一代方法 Where2comm。
- GMM 的 96%–98% 是相对点云地图。
- Kimera-Multi 的"最多少 70%"是相对集中式基线。
- 只有点云压缩社区用 G-PCC 作 BD-rate 锚点；即便如此，在有损设定下对上较弱的锚点时，也能报出约 −95% 这样的数字 ([SparsePCGC](https://arxiv.org/abs/2111.10633))。

没有一篇协同感知论文把学习消息和无损经典几何编码放在同一个 rate–distortion 坐标里比较。项目如果同时报告 log2(bytes)（便于和 Where2comm 曲线对齐）和"相对强无损几何码"的倍数，就补上了这块空白。

## 有位姿误差时，决定精度的是配准校正而不是融合规则

### 位姿噪声的惯例与跌幅

协同感知的噪声惯例是：只在发送方位姿的 x、y、yaw 上加零均值高斯噪声，σ=0.2 m/0.2° 算"正常"水平，扫描到 0.6 m/0.6°，个别工作扫到 2.0 ([FeaKM](https://arxiv.org/html/2502.11003v1))。V2X-ViT 把 0.2 m/0.2° 噪声与 100 ms 时延绑在一起，作为 Noisy Setting：

| 方法 | AP@0.7（无噪声 → Noisy Setting） |
|---|---|
| late fusion | 0.620 → 0.307 |
| early fusion | 0.710 → 0.384 |
| V2X-ViT 自身 | 0.712 → 0.614 |

（数据来源：[V2X-ViT](https://arxiv.org/html/2203.10638)）

一项综述实验中，early fusion 在位置 σ=0.6 m 时 AP@0.5 下降 70.5% ([V2X survey](https://arxiv.org/html/2308.16714))。

精细融合加严格指标最脆弱。OPV2V 上从无噪声到 0.6 m/0.6° 噪声：

| 方法 | AP@0.7（0 → 0.6 m/0.6°） |
|---|---|
| FPV-RCNN | **0.840 → 0.200** |
| V2VNet | 0.740 → 0.504 |
| CoAlign（带 agent-object 位姿图校正） | **0.912 → 0.868** |

（数据来源：[CoAlign](https://arxiv.org/html/2211.07214)）

项目内部"5 cm/1° 让表面和家具损失 16–29 个 F1 点"与此同向：家具 F1 相当于严格 IoU，要求的配准精度与体素同量级。从几何上算，5 cm/1° 在距离 r 处造成的横向错位约为 0.05+0.01745r m：3 m 处约 0.10 m，8 m 处约 0.19 m，已经超过 6–16 cm 的体素（推算）。所以 V2X 文献里"0.2 m/0.2° 只掉 1 个点"的结论不能直接搬到室内细粒度占据上。加固基准的噪声网格应覆盖 cm/° 级，并加入 z、roll、pitch，因为几乎所有协同感知工作只在 SE(2) 上加噪。

### 能收回精度的三类方法

文献中能收回精度的方法分三类，第一类增益最大：

1. **显式位姿校正**：CoAlign 的位姿图、FPV-RCNN 的 maximum consensus、FeaKM 的特征匹配。
2. **带噪训练加注意力或置信度选择**：V2X-ViT、Where2comm。
3. **完全不依赖位姿**：MASH，它在 OPV2V 上的 AP@0.7 恒为 0.198 ([CoAlign](https://arxiv.org/html/2211.07214))。

多机地图合并得出同一结论：

- Saeedi 等把变换的不确定性线性化，对每个 cell 做与位置相关的高斯扩散，再做 log-odds 求和，并用熵滤波拒绝会使熵增大的更新 ([Saeedi et al. 2014](https://people.csail.mit.edu/lpaull/publications/Saeedi_RAM_2014.pdf))。这只能让融合变保守、减少假表面，收不回锐利的几何。
- van Kempen 等的网络能补偿最多 5 m/20° 的错位，直接输出融合后的栅格，但不输出位姿 ([van Kempen et al.](https://arxiv.org/abs/2304.03578))。

只有重新估计变换，或者把相对位姿作为变量放进因子图联合优化，才能真正收回损失。Kimera-Multi 的教训在外点：不做鲁棒化的 L2 位姿图在 Medfield 上 ATE 为 **64.2 m，D-GNC 为 3.92 m** ([Kimera-Multi](https://arxiv.org/html/2106.14386))。在走廊多、家具相似的室内，跨机关联必须配 GNC 或 PCM 一类的鲁棒核。

### 融合规则：一个由假设决定的谱系

Bailey、Julier、Agamennoni 把 CI 推广到非高斯情形，即 geometric mean density（GMD，也就是 logarithmic opinion pool）：p_c∝p_a^ω p_b^(1−ω)。要点有四条 ([Bailey et al. 2012](https://www-personal.acfr.usyd.edu.au/tbailey/papers/fusion2012.pdf))：

- 两个权重都取 1 就是标准 Bayes 更新，只在条件独立时成立。
- 权重和为 1 时，"does not double count common information"。
- 一维高斯的最优 ω 总在 0 或 1。
- 公共信息可以用 p(x|Z_a)p(x|Z_b)/p(x|Z_a∩Z_b) 精确去掉。

对单个 Bernoulli 体素，归一化的 GMD 有闭式解 l_c=ω_a·l_a+ω_b·l_b（以下为推导）：

| 权重取值 | 对应的规则 |
|---|---|
| ω_a=ω_b=1 | log-odds 求和 |
| Σω=1 | CI |
| 1<Σω<2 | 部分独立 |
| 逐体素在 {0,1} 中取使 \|l_c\| 最大者 | max-confidence 选择，也是逐体素熵最优的 CI |

这把项目先前用的 max 选择和现在的求和放进了同一个参数化，也解释了内部探针的结果：

- max 选择取的是 N 个带噪证据中的极端次序统计量，"高置信但错误"的体素被选中的概率随 N 上升（.84→.56）。
- 求和则是累加 N 个独立的对数似然比，错误率按 Chernoff 界指数下降（.85→.99）。

文献里唯一的定量比较只发生在 Bayes 与 Dempster–Shafer 之间：Bayes 在 15/15 次比较中胜出，但绝对差只有 0.001–0.022 ([Berlenko & Krinkin](https://arxiv.org/abs/2602.18872))；而且结论会随传感器模型的参数化方式反转 ([Sensors 2026](https://www.mdpi.com/1424-8220/26/13/4266))。sum、max、CI 在多机占据上的直接对比，以及 CI 在占据栅格上的应用，本次两轮检索都没有找到。

### 求和在森林上的正当性，以及三处代价

求和在森林上的正当性来自 channel filter。Makarenko 等证明 channel filter 对静态问题是精确的，并用分布式 junction tree 推断重新推导了它：在 spanning tree 的每条边上维护一个记录"已传公共信息"的滤波器，收到消息后除以公共信息 ([Makarenko 2009](https://dellaert.github.io/files/Makarenko09fusion.pdf))。树上 GBP 中，发给某个邻居的消息等于"除该邻居外其余信息之和"，这正是同一操作；它在直径 d 轮后精确，与 max-consensus 的轮数界相同。所以在树上从 max 换成求和，不增加通信轮数（推导）。

代价有三处：

1. **公共先验。** 当各节点共享先验时，独立意见池对观点的强化会"extreme in its reinforcement of opinion" ([Bayesian fusion thesis](https://people.cs.pitt.edu/~chang/231/y08/y08sem/semkul/mphil.pdf))。如果各机发送的是后验而不是似然（例如 log-odds 初值不为 0，或学习先验已经写进后验），求和会把先验计 N 次。应发送 posterior÷prior。
2. **环路。** 同一算法在环上会把回传的信息重复计入，也就是 loopy BP 的过度自信。补救办法有三种：限定在 spanning tree 上传播；按 agent-ID 分槽的 G-Counter 式 CRDT，join 幂等、读出时求和，但状态随 N 线性增长 ([Shapiro et al.](https://webarchive.di.uminho.pt/haslab.uminho.pt/cbm/publications/conflict-free-replicated-data-types.html))；或者退回 CI。
3. **量化。** 求和型消息的量化误差随轮数累积，幂等的 max 则不会。量化 Kalman 文献显示，每个标量新息用 2–3 bit 已接近无量化的效果 ([Msechu et al.](https://www-users.cse.umn.edu/~stergios/papers/TSP08-QKF.pdf))。

此外，同型号传感器的系统性测距偏差会违反条件独立。这时应考虑 1<Σω<2 的部分独立权重，或像 OctoMap 那样对 log-odds 做 clamping ([OctoMap](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf))（推断）。在配准误差下，max 会生成双层表面，求和会把错位的证据摊薄，两者都收不回锐利的几何（推断）。

### GBP 的工程基础已经够用

- Du 等对向量值 GBP 证明：任意 PSD 初始化下，消息的 information matrix 都指数收敛到唯一极限；"forest 加单个环"的图总能收敛；这个条件严格弱于 walk-summability ([Du et al., JMLR 2018](https://jmlr.org/papers/v18/16-556.html))。
- Ortiz 等在 Graphcore IPU 上用 GBP 做 BA，125 个 keyframe、1919 个点时不到 40 ms，Ceres 为 1450 ms ([Ortiz et al.](https://arxiv.org/pdf/2003.03134))。
- Robot Web 在 9 台 TurtleBot3 真机上把 ATE 从 0.071 m 降到 0.028 m ([Robot Web](https://ar5iv.labs.arxiv.org/html/2202.03314))。
- Murai 等在 64 机上自标定，外参误差为 0.025 m/1.214°，集中式为 0.020 m/0.969° ([Murai et al.](https://arxiv.org/html/2401.15036v1))。

学习先验进入因子图也有成熟的形态。CodeSLAM 和 DeepFactors 在标准 factor graph 中联合优化学习得到的 code 与位姿 ([DeepFactors](https://arxiv.org/pdf/2001.05049))。Nabarro 等把整个 CNN 写成 GBP 因子，但需要 damping 0.9 和 dropout 0.5 才能稳定 ([Nabarro et al.](https://arxiv.org/abs/2311.14649))。

缺的是两件事：

- 没有系统用 GBP 同时估计跨机配准与占据地图。Robot Web 和 Murai 只估计位姿或外参，DistGP 假设位姿已知 ([DistGP](https://arxiv.org/html/2603.07351v1))。
- 没有论文量化或按比特预算压缩 GBP 消息。

### Sheaf：数学先例完整，工程先例为零

数学上的先例：

- Gao、Brodzki、Mukherjee 证明：图上关于群 G 的 synchronization 问题对应 flat principal G-bundle；可同步当且仅当 holonomy 平凡；graph connection Laplacian 就是 0 阶 Hodge Laplacian ([Gao et al.](https://arxiv.org/abs/1610.09051))。
- Hansen 与 Ghrist 把 connection Laplacian 解释为 O(n) 向量丛上的 sheaf Laplacian，并提出 homological programming ([Hansen & Ghrist](https://arxiv.org/pdf/1808.01513))。
- 2026 年的 SCGL 从带噪的向量信号中，联合估计去噪后的信号、拓扑和逐节点的局部参考系 ([SCGL](https://arxiv.org/abs/2608.08710))。
- Robinson 把融合定义为"计算离观测最近的 global section"，并指出 consistency radius 是融合畸变的下界 ([Robinson](https://arxiv.org/pdf/1603.01446))。

工程上，在 SLAM 与多机建图里，没有找到任何论文把 SE(2) 或 SE(3) 配准写成 sheaf 并与地图 sheaf 耦合。一篇 2026 年去中心化 PGO 论文的相关工作部分列出了 SE-Sync、Cartan motion group 和两阶段 Gauss–Seidel，没有用到 sheaf 语言 ([2606.24489](https://arxiv.org/pdf/2606.24489))。多智能体 sheaf 工作到 2025 年底仍然只有仿真 ([Hanks et al.](https://arxiv.org/html/2504.02049v2))。

文献还给项目的 sheaf 选型提了个醒。Ghrist–Riess 的 Tarski Laplacian 在格序下能有限步精确收敛 ([Ghrist & Riess](https://ar5iv.labs.arxiv.org/html/2007.04099))。但限制映射一旦是带误差的几何变换，lattice join 会把错位的占据当作"更确定"而保留下来。配准不确定时，二次能量的向量值 sheaf Laplacian 更自然（推断）。

可行的结构分两层（推断）：

- **底层 pose/connection sheaf**：限制映射是相对 SE(3) 变换，global section 存在当且仅当所有环路的 holonomy 平凡。
- **上层 map sheaf**：限制映射由底层位姿参数化，是重采样算子。

联合能量等于位姿图能量加上重叠区地图的不一致能量。对限制映射求梯度，就是"用地图一致性来配准"；在切空间线性化之后，这正是 Robot Web 式 GBP 的消息空间。上层 sheaf 的 consistency radius 还能把"噪声造成的损失"和"配准造成的损失"分开报告（推断）。

## 加固基准可直接借用的部件，以及留给我们的空白

加固基准的多数部件可以直接借用：

| 用途 | 借用对象 | 具体做法 |
|---|---|---|
| 场景多样性 | 3D-FRONT（6,813 栋、14,629 个房间）([DiffuScene](https://arxiv.org/html/2303.14207))；ProcTHOR-10K ([ProcTHOR](https://arxiv.org/abs/2206.06994))；Infinigen Indoors ([Infinigen](https://arxiv.org/html/2406.11824v1)) | 加入非轴对齐家具（cos θ, sin θ）和更丰富的布局，同时保留精确真值 |
| 外部效度 | HM3D，1,000 栋重建 ([HM3D](https://arxiv.org/abs/2109.08238))；Replica 0.05 m、128×128×32 ([Octree LD](https://arxiv.org/html/2509.16483v1)) | 做迁移测试；真值 mesh 的缺失区排除在评分之外 |
| 深度噪声 | Redwood 深度噪声模型 ([检索摘要](https://arxiv.org/pdf/2010.13439)) | 噪声随距离增长，并带随机缺失 |
| 里程计与执行噪声 | Habitat-sim `PyRobotNoisyActuationSpec`（LoCoBot 实测，带 noise multiplier）([Habitat](https://aihabitat.org/docs/habitat-sim/habitat_sim.agent.PyRobotNoisyActuationSpec.html)) | 用 noise multiplier 做扫描 |
| 机间位姿噪声 | CoAlign 的 σt/σr 网格、"训练时是否带噪"两种设定、Laplace 分布外测试 ([CoAlign](https://arxiv.org/html/2211.07214)) | 网格下延到 cm/° 级，并加入 6-DoF |
| 外点 | Kimera-Multi 的外点回环实验 ([Kimera-Multi](https://arxiv.org/html/2106.14386)) | 在走廊、相似家具处注入错误关联 |
| 熵模型结构 | ARDM、checkerboard、M2T、SparsePCGC | K=2–8 组的固定顺序分解，用交叉熵训练 |
| 跨机确定性 | integer networks、PTQ | 在异构设备上做逐位一致解码测试 |
| 码率记账 | log2(bytes) ([Where2comm](https://arxiv.org/pdf/2209.12836))；bits per occupied voxel 加 BD-rate，注明 TMC13 版本 ([UniPCGC](https://arxiv.org/abs/2503.18541)) | 协议、表示、编码三个因子分别报告 |
| 去噪器 | GSSC S²D² 一步细化；Achey 的 P(m\|d)<P(m\|z) | 输出边缘概率，稀有类不降作为门槛 |
| 补全隔离 | SceneSense inpainting；PwC conformal | 预测单独成层，不进 F1 和 coverage |
| 配准 | CoAlign 位姿图；MR-GMMapping 几何双用；D-GNC；Robot Web GBP | 由同一份几何码提供配准证据 |
| 融合与诊断 | channel filter；posterior÷prior；分槽 CRDT；Robinson consistency radius | 噪声损失与配准损失分开报告 |

评估惯例方面：

- 按干净真值评分；"训练时干净、测试时加噪"与"训练时带噪"两种设定都要报告。
- 观测区 F1 和预测区 F1 分开报告；coverage 只读观测通道。
- F1 阈值按验证集选取。
- 位姿精度同时报告 ATE 与 ARE。
- 生成先验附带记忆化检查。
- 所有门槛预注册，两个 seed 都通过才算数。

文献的空白，也就是项目可以贡献的地方，有九处：

1. **没有面向三值 3D 体素地图或多机 3D 占据消息的学习熵编码。** 10⁴ 点以下小点云的 bpp 也没有任何公开数字；hyperprior 的 side information 在 100–200 B 短消息中占多少，同样没有量化。
2. **协同感知没有把学习消息与无损经典几何码放在同一个 rate–distortion 坐标里。** 分母一直是 float32 特征或上一代方法。
3. **没有"cm/° 级位姿噪声 → 占据 F1/IoU"的受控扫描**，也没有在传感器噪声下、按干净真值评分的协同占据评测。CoHFF 对 0–0.6 m GPS 噪声只给了定性结论。
4. **多机占据上没有 sum、max、CI 三者的定量对比。**
5. **没有用 GBP 联合估计跨机配准与占据的系统**，也没有 GBP 消息的量化研究。
6. **没有把位姿 sheaf 与地图 sheaf 耦合的工作**，没有 sheaf 与 GBP 的显式结合，也没有硬件上的多机 sheaf 融合结果。
7. **3D 占据上没有直接比较"后验采样与边缘阈值化"F1 的研究**；生成式建图的 FP/FN 率与校准曲线没人量化；RDP 理论在二值数据和 F1 指标上没有专门结果。
8. **MeanFlow 和 consistency 模型没有在离散体素上做过任何实验**；masked diffusion 在少于 10 步时的码长曲线也没有。
9. **室内带家具房屋"部分观测 → 占据补全"没有逐体素 F1 的结果**，LT3SD 和 Octree LD 都只报生成指标。

## 结论

这次综述最重要的变化，是改写了"学习和 sheaf-GBP 的价值在哪里"的答案。在干净全保真的世界里，文献留给学习的空间只是一个 1.1–1.5 倍左右的编码因子；十倍以上的收益都来自"发什么"这一协议选择，而精确几何码已经把这部分拿走了。相反，在噪声与配准误差下，文献记录的损失是几十个百分点：FPV-RCNN 从 0.840 跌到 0.200，early fusion 掉 70.5%，项目内部损失 16–29 个 F1 点。能收回这些损失的，恰好是显式位姿校正和对独立证据的正确累加。所以学习和 sheaf-GBP 的主战场应该从"省字节"移到"收回噪声与配准造成的 F1"，后者的杠杆比前者大一个数量级。

再往前推一步：字节最少的消息最怕配准误差。FPV-RCNN 与命中集同在 0.3 KB 以下的量级，这把"压缩"和"配准"绑成了同一个问题。同一份平面 + 盒几何码既是地图消息，也应该是配准证据；位姿 sheaf 与地图 sheaf 共享一个能量，在森林上用求和型 GBP 求解。文献里没有人这样做过，也没有人给出 cm/° 噪声对占据 F1 的曲线、多机占据上 sum、max 与 CI 的对比，或者以强无损几何码为基线的协同感知压缩比。这四项可以合成一篇自带基线的工作。至于生成器，设计规则可以压成一句话：只让它给概率，不让它给样本。
