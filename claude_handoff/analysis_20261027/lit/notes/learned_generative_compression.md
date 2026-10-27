# 学习式与生成式低码率压缩：对"精确几何码 + 学习熵先验"结论的文献对照

> 标注约定：除非特别注明，文中所有数字都是**原论文作者自报**（claimed），不是独立复现（measured）。标 [二手] 的数字取自第三方摘要或其他论文对原表的转述，未逐字核对原文。项目内部数字（1,384 B / 2,733 B / 200–260 B / 130–180 B）是用户自测，仅在 Inferences 中用作对照。检索日期：2026-10-27。

## Q1：学习式熵模型（hyperprior、自回归上下文、checkerboard、channel-wise、transformer）相对经典上下文编码（CABAC、JBIG2、G-PCC 式手工上下文）能多省多少？二值图像、分割图、占据栅格上有没有数字？

### Takeaway
自然图像有损编码中，各项增益大致是：在 hyperprior 上加空间自回归上下文，平均 −15.8%；channel-wise 相对 context-adaptive 基线，−6.7% 到 −11.4%；transformer 熵模型相对 VTM 最多 −11%。与本项目最接近的设定是"二值 3D 占据 + 手工邻域上下文算术编码"，即 MPEG G-PCC。在这个设定下，学习式上下文模型在致密表面点云上自报 −29% 到 −36%（SparseVoxelDNN 跨数据集平均 −52%）。但在稀疏点云上只有 −4% 到 −10%，个别样本还会变差（+3.7% 到 +8%）。代价是解码比 G-PCC 慢 2 到 3 个数量级。我没有找到学习式方法与 JBIG2，或与语义分割图上的 CABAC 直接对比的数字。

### Cited Findings
**A. 自然图像（有损，学习式熵模型之间的递进增益）**
- Ballé et al., "Variational image compression with a scale hyperprior"（ICLR 2018）。摘要只声称在 MS-SSIM 上达到 state of the art、PSNR 超过已发表的 ANN 方法，没有给具体百分比 — [arXiv 1802.01436](https://arxiv.org/abs/1802.01436)
- Minnen, Ballé, Toderici, "Joint Autoregressive and Hierarchical Priors for Learned Image Compression"（NeurIPS 2018）。在 hyperprior 基线上加入自回归上下文后，平均文件大小 −15.8%。相对 JPEG −59.8%，相对 WebP 和 JPEG2000 均超过 −35%，比 BPG 小 8.4%。作者称这是第一个在 PSNR 和 MS-SSIM 上都超过 BPG 的学习式方法 — [arXiv 1809.02736](https://arxiv.org/abs/1809.02736)
- 同一论文承认，自回归组件带来显著的计算代价（串行解码） — [NeurIPS 2018 PDF](https://papers.neurips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf)
- He et al., "Checkerboard Context Model for Efficient Learned Image Compression"（CVPR 2021）。解码加速超过 40 倍，率失真性能"almost the same" — [arXiv 2103.15306](https://arxiv.org/abs/2103.15306)
- Minnen & Singh, "Channel-wise Autoregressive Entropy Models for Learned Image Compression"（ICIP 2020）。相对 context-adaptive 基线，Kodak 平均 −6.7%，Tecnick −11.4%。低码率下相对基线最多 −18%，相对 BPG 最多 −25% — [arXiv 2007.08739](https://arxiv.org/abs/2007.08739)
- Koyuncu et al., "Contextformer: A Transformer with Spatio-Channel Attention for Context Modeling in Learned Image Compression"（ECCV 2022）。相对 VTM 16.2 最多 −11% — [arXiv 2203.02452](https://arxiv.org/abs/2203.02452)
- Informer（"Joint Global and Local Hierarchical Priors for Learned Image Compression"，CVPR 2022）。MSE 优化模型相对 BPG −11.84% 到 −19.27% [二手] — [arXiv 2112.04487](https://arxiv.org/html/2112.04487v2)
- Mentzer et al., "M2T: Masking Transformers Twice for Faster Decoding"（ICCV 2023）。推理约快 4 倍，代价是"a small increase in bitrate"，未量化 — [arXiv 2304.07313](https://arxiv.org/abs/2304.07313)
- Entroformer（ICLR 2022）、ELIC（CVPR 2022）、TCM（Liu et al., CVPR 2023）的摘要只声称 state of the art，没有给 BD-rate 数字 — [Entroformer 2202.05492](https://arxiv.org/abs/2202.05492)；[ELIC 2203.10886](https://arxiv.org/abs/2203.10886)；[TCM 2303.14978](https://arxiv.org/abs/2303.14978)

**B. 无损自然图像（学习式 vs 手工上下文：JPEG-XL / FLIF）**
- DLPR（Bai et al., "Deep Lossy Plus Residual Coding for Lossless and Near-lossless Image Compression"）。作者声称无损和近无损都达到 SOTA，编码速度"competitive"（不是最快） — [arXiv 2209.04847](https://arxiv.org/abs/2209.04847)
- DLPR 论文的运行时间表 [二手]。768×512 图像：FLIF 解码 0.16 s，JPEG-XL 0.08 s，DLPR 无损 1.80 s。2040×1356 图像：JPEG-XL 0.42 s，DLPR 11.91 s — [arXiv 2209.04847 PDF](https://arxiv.org/pdf/2209.04847)
- L3C（Mentzer et al., "Practical Full Resolution Learned Lossless Image Compression"，CVPR 2019）。比最快的 PixelCNN 变体（Multiscale-PixelCNN）快两个数量级以上，预测全部像素概率只需 3 次前向 — [arXiv 1811.12817](https://arxiv.org/abs/1811.12817)

**C. 二值 / 占据数据（最可比：学习式上下文 vs G-PCC 手工上下文算术编码）**
- G-PCC 的八叉树几何编码本身就是手工上下文算术编码，工具包括 Neighbour-Dependent Entropy Context、intra prediction、planar/angular mode，以及孤立点的 Direct Coding Mode — [Nguyen et al., arXiv 2107.00400](https://arxiv.org/html/2107.00400)
- VoxelDNN（Nguyen, Quach, Valenzise, Duhamel，"Learning-based lossless compression of 3D point cloud geometry"）是 3D masked-conv 自回归模型，逐体素预测占据概率。以 bits per occupied voxel 计，平均比 G-PCC −28% — [arXiv 2011.14700](https://arxiv.org/abs/2011.14700)
- VoxelDNN 扩展版 Table V（最佳配置 Baseline+DA+CE）的数据集均值。致密表面点云：MVUB 0.7912 bpov（−29.01%），8i 0.6405 bpov（−35.77%）。较稀疏点云：CAT1 3.4035 bpov（−3.72%），USP 3.4855 bpov（−9.91%）。单样本 Arco Valentino 为 +3.70%。不加 DA/CE 时，CAT1 平均 +0.91%（比 G-PCC 更差）。正文给出的总平均是 −20.17% — [arXiv 2107.00400](https://arxiv.org/html/2107.00400)
- 同一论文说明了代价。编码/解码时间：G-PCC 2.91 s / 2.85 s，VoxelDNN Baseline 3282 s / 6783 s，作者称复杂度比 G-PCC 高 3 个数量级。每个模型约 3.5 MB 参数。作者还写道，块不够稠密时方法变得低效，非常稀疏的块码率会"rapidly grows by an order of magnitude"，处理低密度点云"would require to rethink the network architecture" — [arXiv 2107.00400](https://arxiv.org/html/2107.00400)
- MSVoxelDNN（多尺度、分组并行）。平均比 G-PCC −17.5%，比 VoxelDNN 少了一部分压缩增益，换来解码复杂度降低两个数量级以上；在稀疏点云上可能比 G-PCC 差 — [arXiv 2104.09859](https://arxiv.org/abs/2104.09859)
- SparseVoxelDNN（Nguyen & Kaup 2022，sparse tensor）。四个数据集上平均比 MPEG 标准 −52%，比 VoxelDNN 再 −29%，但解码仍远慢于 G-PCC — [arXiv 2204.05043](https://arxiv.org/html/2204.05043)
- OctAttention（Fu et al., AAAI 2022，八叉树 + 大范围注意力上下文）。在 SemanticKITTI（LiDAR，稀疏）上平均比 G-PCC −25.4%，而 OctSqueeze 不到 −4%。物体点云上，MPEG 8i 约 −35%，MVUB 约 −31%。编码时间比体素式基线省 95% — [AAAI PDF](https://cdn.aaai.org/ojs/19942/19942-13-23955-1-2-20220628.pdf)；[arXiv 2202.06028](https://arxiv.org/abs/2202.06028)
- ECM-OPCC 在 LiDAR 上五个失真级别平均比 G-PCC −31%，比 OctAttention 再 −4.4% 到 −8.9%。它的 MPEG 8i 计时表 [二手]：G-PCC 解码约 1.0 s，OctAttention 约 948 s，ECM-OPCC 约 19.5 s；G-PCC 主要跑 CPU，学习式方法依赖 GPU — [arXiv 2211.10916](https://arxiv.org/pdf/2211.10916)
- 一项较早的广播场景对比得出相反结论：G-PCC 压缩率最好。但它对比的是较老的八叉树方法 [二手] — [arXiv 2202.00719](https://arxiv.org/pdf/2202.00719)
- 二值 MNIST（真正的二值图像）：BB-ANS（VAE + bits-back）0.19 bits/dim，bz2 0.25，gzip 0.33，PNG 0.78 [二手] — [arXiv 1901.04866](https://arxiv.org/abs/1901.04866)；[liner 摘要](https://liner.com/review/practical-lossless-compression-with-latent-variables-using-bits-back-coding)

**D. 分割图 / 语义图 / 机器人占据栅格**
- "Context Adaptive Extended Chain Coding for Semantic Map Compression"（Yang, Liao, Choi, Racapé, Bajić，2026，非学习式）。在语义图数据集上比一个未具名的 SOTA 基准平均 −18%。编码时间最多 −98%、解码最多 −50%（对比一个未具名的现代通用无损编码器）。在占据图上"consistent compression gains across the majority of tested scenarios"，未给数字 — [arXiv 2603.03073](https://arxiv.org/abs/2603.03073)
- 有一份检索摘要称上面这篇论文与 JBIG1、HEVC-SCC、FLIF 对比，并称"未找到近期专门做无损语义图压缩的学习式方法"。但我抓取到的摘要页没有出现这些内容，**未核实** — [arXiv 2603.03073 PDF](https://arxiv.org/pdf/2603.03073)
- 二值文档图像：一篇多级字典方法（2024）称总体优于 JBIG2 和一个学习式基线 MGT，但承认 JBIG2 在部分二值图上更好 [二手，未读全文] — [arXiv 2406.03087](https://arxiv.org/pdf/2406.03087)
- 多机 2D 激光 SLAM 的学习式地图通信（Sensors 2024，有损）。CNN 下采样后做 Huffman + RLE，接收端用 CNN 恢复。称通信开销 −99%，恢复图 PSNR 50.649 dB、SSIM 0.975；只做了两机测试 — [PMC11124910](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124910)
- 带宽背景：传一张占据栅格通常需要约 2 Mbps，而水下通信带宽很少超过 100 kbps（DARS 2024） — [marmotlab PDF](https://www.marmotlab.org/publications/67-DARS2024-CommsConstrMAExploration.pdf)
- 时变概率占据栅格的分层压缩（2025，非学习式，多分辨率层级编码器，做 distortion–description-size 权衡） — [arXiv 2504.10751 摘要](https://bytez.com/docs/arxiv/2504.10751/paper)

### Inferences
- **项目数字换算。** 128×128×32 = 524,288 体素。1,384 B ≈ 0.0211 bit/voxel；gzip 2,733 B ≈ 0.0417 bit/voxel。手工 3D 上下文编码比 gzip 小约 49%。BB-ANS 在二值 MNIST 上比 gzip 小约 42%（0.19 vs 0.33），两者同一量级。这说明"上下文建模 vs 通用 LZ"的主要增益已经被手工 3D 上下文编码拿到了。学习式先验要争的是下一档：在强手工上下文编码器之上再省多少。
- **预测增益与文献对照。** 200–260 B 降到 130–180 B，即约 −31% 到 −35%。文献中学习式模型相对强手工上下文编码器（G-PCC）的增益是：致密表面 −29% 到 −36%；稀疏 −4% 到 −10%，个别为正；并行化（MSVoxelDNN）后 −17.5%。图像域里单加一层上下文是 −15.8%，channel-wise 是 −6.7% 到 −11.4%。所以 −31% 到 −35% 落在文献增益区间的**上沿**，而且文献中只有"致密、同分布、重型串行自回归模型"同时满足时才达到过。
- **几何描述码可能处在"稀疏"区间。** 一条 200 B 级消息只有几百个符号，模型能用的消息内上下文很少；VoxelDNN 作者指出，这正是学习式卷积上下文变弱的区间。学习式先验的增益只能主要来自"场景统计先验"（家具/墙面的形状分布），而不是消息内上下文。建议预注册的成功判据定为：在留出房屋上，两个 seed 都相对"同一几何码 + 手工上下文编码器"稳定 ≥10–15%。−30% 以上应视为乐观上限，不宜作为默认预期。
- **对比基线要选对。** 学习式增益必须对照"手工上下文编码器作用在同一几何描述上"来计算，不能对照未编码的几何描述，否则会把上下文建模本身的增益也算到学习先验头上。
- **增益会被稀释。** 位姿比特与体素坐标的统计不同，占据先验对位姿几乎无效。学习先验只作用于体素部分，整条消息的相对增益会被稀释。
- **失配只多花比特，不改重建。** 无损学习式熵编码不会改变重建结果，先验失配只会多花比特（例如 VoxelDNN 在 Arco Valentino 上 +3.7%）。可以加 1 bit 的 escape 标志位，在学习编码与手工编码之间取较小者，把最坏情况限制在约 +1 bit。这是工程推断，文献中没有找到针对此做法的数字。
- **速度不是瓶颈。** VoxelDNN 那 3 个数量级的慢速对应的是约百万体素的点云。对一条只有几百个符号的消息，即使串行自回归也很快。这是推断，未测。

### Gaps
- 没有找到学习式熵模型与 JBIG2 / JBIG1 在二值图像上的直接对比数字，也没有找到学习式方法与 CABAC 在语义分割图上的直接对比。
- 没有找到针对**三元（free/occupied/unknown）3D 体素图**或多机 3D 占据消息的学习式熵编码论文。现有工作是 2D 有损（Sensors 2024）或非学习式。
- 在约 100–200 B 的极短消息上，hyperprior 的 side information 开销占比多少，文献中没有找到量化数字。
- 本节几乎所有数字都是作者自报，学习式点云压缩方法之间的独立复现很少。不同论文的 G-PCC 版本和测试集不同，−28%、−17.5%、−52%、−25.4% 不能直接横向比较。

## Q2：用深度生成模型做无损压缩（bits-back：BB-ANS、Bit-Swap、HiLLoC；Integer Discrete Flows；PixelCNN 类；LLM-as-compressor）。增益是多少，代价是什么（速度、两端都要部署的模型大小）？

### Takeaway
各类生成模型无损编码器的实际码率都接近模型的 NLL / ELBO。在同分布数据上，它们比 gzip、bz2、PNG 这类通用或手工编码器好 10% 到 40% 左右。代价有三：（1）速度慢一到两个数量级以上；（2）收发两端都必须持有同一模型；（3）一旦分布外，码率会崩溃。Delétang et al. 的 LLM 结果最能说明这一点：Chinchilla 70B 的原始压缩率最好（enwik9 8.3%），但把 70B 参数计入码长后，"adjusted rate"高达约 14,008%。小 Transformer 在不匹配的模态上甚至会把数据膨胀到原始大小的约 2 倍。

### Cited Findings
- BB-ANS（Townsend, Bird, Barber, "Practical Lossless Compression with Latent Variables using Bits Back Coding"，ICLR 2019）。二值 MNIST 0.19 bpd，bz2 0.25、gzip 0.33、PNG 0.78。全灰度 MNIST 1.41 bpd，bz2 1.42。实际码率接近测试 ELBO（0.19 / 1.39）[二手] — [arXiv 1901.04866](https://arxiv.org/abs/1901.04866)；[liner 摘要](https://liner.com/review/practical-lossless-compression-with-latent-variables-using-bits-back-coding)
- 另一篇论文（ICLR 2022 的 HCLT / probabilistic circuits）在其表中列 BB-ANS 在 MNIST 上为 1.42（ELBO 1.39），与原文的 1.41 略有出入，可能是设置不同 — [Liu et al. ICLR 2022 PDF](http://starai.cs.ucla.edu/papers/LiuICLR22.pdf)
- Bit-Swap（Kingma, Abbeel, Ho，ICML 2019）称对层级隐变量模型取得"strictly better compression rates"，摘要没有给数字 — [arXiv 1905.06845](https://arxiv.org/abs/1905.06845)
- HiLLoC（Townsend, Bird, Kunze, Barber，ICLR 2020）。VAE 在 32×32 ImageNet 上训练，能泛化到 64×64 和全尺寸照片，称在全尺寸 ImageNet 上达到 state of the art。我没有找到具体 bpd 数字 — [arXiv 1912.09953](https://arxiv.org/abs/1912.09953)
- Monte Carlo bits-back（2021）专门研究如何改进 bits-back 码率 — [arXiv 2102.11086](https://arxiv.org/pdf/2102.11086)
- Integer Discrete Flows（Hoogeboom et al., NeurIPS 2019）称在 CIFAR10、ImageNet32、ImageNet64 上达到 SOTA — [arXiv 1905.07376](https://arxiv.org/abs/1905.07376)。IDF++ 给出 IDF 原文 CIFAR-10 测试集 3.32 bpd，用全训练集重训后为 3.30 bpd — [arXiv 2006.12459](https://arxiv.org/pdf/2006.12459)
- L3C 比 Multiscale-PixelCNN 快两个数量级以上（3 次前向，而不是逐像素一次前向） — [arXiv 1811.12817](https://arxiv.org/abs/1811.12817)。学习式无损编码器 DLPR 在 768×512 图像上解码 1.80 s，JPEG-XL 为 0.08 s [二手] — [arXiv 2209.04847](https://arxiv.org/pdf/2209.04847)
- **Delétang et al., "Language Modeling Is Compression"（ICLR 2024），v2 表格（压缩后/原始，%，raw / adjusted）**
  - enwik9：gzip 32.3，LZMA2 23.0，Chinchilla 1B 11.3 / 211.3，7B 10.2 / 1410.2，70B 8.3 / 14008.3。
  - ImageNet patch：PNG 58.5，gzip 70.7，Chinchilla 70B 48.0 / 14048.0。
  - LibriSpeech：FLAC 30.3（2048-byte 分块），Chinchilla 70B 21.0 / 14021.0。
  - 来源：[arXiv 2309.10668v2](https://arxiv.org/html/2309.10668v2)
- **数字冲突。** 论文摘要和 ICLR 页面写的是 Chinchilla 70B 把 ImageNet patch 压到 43.4%、LibriSpeech 压到 16.4%（对比 PNG 58.5%、FLAC 30.3%）。但 v2 表格是 48.0% 和 21.0%。差异原因未查明，可能是版本或计数口径不同（例如被丢弃的 LSB 是否追加计入） — [ICLR 2024 页面](https://iclr.cc/virtual/2024/poster/17997)；[arXiv 2309.10668v2](https://arxiv.org/html/2309.10668v2)
- Delétang 的预处理与开销。图像转灰度，切成 32×64 patch（2048 bytes）。每字节除以 2 映射到 [0,127]，被丢弃的 LSB 追加到输出。上下文限 2048 token，数据按 2048 字节分块。论文没有给吞吐数字，只说 Transformer 运行时间"already very long"，prequential coding"very costly" — [arXiv 2309.10668v2](https://arxiv.org/html/2309.10668v2)
- **分布外崩溃。** 在 enwik8 上训练的小 Transformer（200K / 800K / 3.2M 参数）在 enwik9 上能到 30.9% / 21.7% / 17.0%，优于 gzip。但在 ImageNet 上是 194.0% / 185.1% / 215.8%，在 LibriSpeech 上是 146.6% / 131.1% / 228.2%，即比不压缩还大 — [arXiv 2309.10668v2](https://arxiv.org/html/2309.10668v2)
- 计入模型大小后，每个数据集都存在一个临界模型规模，超过后 adjusted rate 反而上升；数据集越大，能"负担"的参数越多 — [arXiv 2309.10668v2](https://arxiv.org/html/2309.10668v2)；[alphaxiv 概览](https://www.alphaxiv.org/overview/2309.10668v2)
- LLMZip（Valmeekam et al. 2023）用 LLaMA-7B，称优于 BSC、ZPAQ、paq8h，摘要没有给数字 — [arXiv 2306.04050](https://arxiv.org/abs/2306.04050)
- P²-LLM（"Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need"，2024）称超过 SOTA 经典与学习式编码器，摘要没有给数字 — [arXiv 2411.12448](https://arxiv.org/abs/2411.12448)
- 综述：Yang, Mandt, Theis, "An Introduction to Neural Data Compression"（2022），系统覆盖熵模型、bits-back 等 — [arXiv 2202.06533](https://arxiv.org/abs/2202.06533)

### Inferences
- **模型开销可摊销。** 多机系统里模型是预先部署、两端共享的，所以 Delétang 意义上的"模型计入码长"是一次性成本，可以摊销到所有消息上。真正的约束是第二点：同分布。在合成家具房屋上训练的先验，遇到真实扫描时可能像小 Transformer 遇到图像那样失效。Q1 中的 escape 回退机制因此是必要的。
- **bits-back 不适合单条短消息。** bits-back 类方法（BB-ANS、HiLLoC）需要一个初始比特缓冲来"借"比特，只有在长序列上才能摊销。对单条约 150 B 的独立消息，这部分开销无法摊销（推断，见 Gaps）。全观测的自回归或上下文模型直接配算术编码更合适，它们不需要隐变量，没有 bits-back 开销。
- **LLM-as-compressor 只作理论参照。** 70B 级模型不可能部署在机载端。它说明的是"更强的预测器带来更低的码率"，但也说明码率增益随模型规模递减，且需要海量算力。对本项目而言，正确的规模是 MB 级领域专用模型（VoxelDNN 每个模型约 3.5 MB 已是一个参照点，见 Q1）。

### Gaps
- 没有在原文中核实 bits-back 的"initial bits"开销在单条短消息上的具体大小。这是领域常识，本次检索没有拿到带数字的出处。
- HiLLoC 的全尺寸 ImageNet bpd、BB-ANS 和 HiLLoC 的编解码速度、LLMZip 的具体 bpc，本次均未拿到原文数字。
- 没有找到 LLM 或生成式无损编码器在二值体素 / 占据数据上的结果。
- 学习式熵编码的跨平台确定性问题（浮点不一致会导致解码失败，Ballé 等人提出过 integer networks 方案）本次没有检索核实，部署前需要专门查。

## Q3：生成式 / 感知式编码器在超低码率下的表现（HiFiC、MS-ILLM、CDC、DiffC、PerCo、文本/语义条件编码、VQ tokenizer）：码率、失真指标、已知的幻觉失效模式

### Takeaway
生成式编码器已经能在 0.003–0.04 bpp 下给出"看起来真实"的图像。PerCo 在 0.003 bpp 时，一张 512×768 的 Kodak 图只需不到 153 B，与本项目消息同一字节量级。但这些系统优化的是 FID/KID 等分布指标；PSNR、MS-SSIM、LPIPS 往往比失真导向的编码器更差。低于约 0.05 bpp 时，有多篇论文报告"semantic deviation"（例如招牌上的字被改写）；低于约 0.01 bpp 时，有论文报告"semantic collapse"（物体变形、消失）。它们能保证的是"像"，不是"对"。

### Cited Findings
**GAN / 似然类感知编码器**
- HiFiC（Mentzer, Toderici, Tschannen, Agustsson, "High-Fidelity Generative Image Compression"，NeurIPS 2020）。用户研究中，即使对比方法用了超过 2 倍的码率，HiFiC 仍更受偏好 — [arXiv 2006.09965](https://arxiv.org/abs/2006.09965)
- MS-ILLM（Muckley et al., "Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models"，ICML 2023）。在 CLIC2020 上用少 30–40% 的比特达到与 HiFiC 相同的 FID。作者指出，单纯优化 PSNR / MS-SSIM 会让低码率重建发糊 — [arXiv 2301.11189](https://arxiv.org/abs/2301.11189)

**扩散类**
- CDC（Yang & Mandt, "Lossy Image Compression with Conditional Diffusion Models"，NeurIPS 2023）。解码器是条件扩散模型：传输"content"潜变量，纹理在反向扩散中合成。FID 优于 GAN 基线，失真指标与 VAE 类方法"competitive"，只需"a handful of decoding steps" — [arXiv 2209.06950](https://arxiv.org/abs/2209.06950)
- DiffC（Theis, Salimans, Hoffman, Mentzer, "Lossy Compression with Gaussian Diffusion"，2022）。不用编码器变换，而是用 reverse channel coding 传输加噪像素。在 ImageNet 64×64 上优于 HiFiC。高码率下，基于 probability-flow 的重建比 ancestral sampling 高约 3 dB。支持渐进解码 — [arXiv 2206.08889](https://arxiv.org/abs/2206.08889)
- Vonderfecht & Liu, "Lossy Compression with Pretrained Diffusion Models"（ICLR 2025）。称这是 DiffC 的第一个完整实现，用于 SD 1.5 / 2.1 / XL 和 Flux-dev，无需额外训练，压缩加解压在 10 秒以内 — [arXiv 2501.09815](https://arxiv.org/abs/2501.09815)
- 后续工作（2026）指出，扩散编码器的主要运行时瓶颈在迭代去噪解码，提出用 few-step 模型替代 — [arXiv 2606.10450](https://arxiv.org/html/2606.10450)

**PerCo 系列（文本 + 局部特征条件）**
- PerCo（Careil, Muckley, Verbeek, Lathuilière, "Towards image compression with perfect realism at ultra-low bitrates"，ICLR 2024）。评估码率 0.1 到 0.003 bpp。0.003 bpp 时，512×768 Kodak 图不到 153 bytes。声称 FID/KID 达到 SOTA，且视觉质量对码率不敏感 — [arXiv 2310.10325](https://arxiv.org/abs/2310.10325)
- 在 PerCo 论文的对比中，MS-ILLM 在其工作点上重建明显模糊，PerCo 更真实，但 PerCo 的 PSNR、MS-SSIM、LPIPS 都显著更差 [二手] — [PerCo ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/43e9d647ccd3e4b7b5baab53f0368686-Paper-Conference.pdf)；[liner 摘要](https://liner.com/review/towards-image-compression-with-perfect-realism-at-ultralow-bitrates)
- PerCo (SD) 的开源复现：0.0036 bpp 时 FID 4.49，优于 PerCo 在 0.0041 bpp 的 5.49 — [arXiv 2409.20255](https://arxiv.org/html/2409.20255)
- PerCoV2 在 0.003–0.03 bpp 上全面改进 PerCo 系列；v2 版本称在 MSCOCO-30k 上 0.0015–0.025 bpp 范围内 FID/KID 达到 SOTA。两个版本的标题和结果不同，引用时需注明版本 — [arXiv 2503.09368v1](https://arxiv.org/html/2503.09368v1)；[v2](https://arxiv.org/html/2503.09368v2)
- Text + Sketch（Lei et al. 2023）用文本加草图，不做端到端训练，称在极低码率下语义和感知保真度显著优于学习式编码器，摘要没有给数字 — [arXiv 2307.01944](https://arxiv.org/abs/2307.01944)

**VQ tokenizer 类**
- Mao et al., "Extreme Image Compression using Fine-tuned VQGANs"（DCC 2024）。把图像编码为 VQGAN 码本索引，再无损编码索引。在 ≤0.04 bpp 时感知指标优于 SOTA 编码器。丢失多达 20% 的索引后，仍能由 Transformer 恢复，感知损失很小 — [arXiv 2307.08265](https://arxiv.org/abs/2307.08265)
- GLC（Qi, Jia, Li, Li, Li, Lu, "Generative Latent Coding for Ultra-Low Bitrate Image and Video Compression"）在生成式 VQ-VAE 潜空间里做变换编码，码率低于 0.04 bpp。在 CLIC 2020 上达到与 MS-ILLM 相同的 FID 时少用 45% 码率；视频版在 DISTS 上比 PLVC 省 65.3% — [arXiv 2505.16177](https://arxiv.org/abs/2505.16177)
- RDVQ（2026）给 VQ 压缩加上可微 RD 优化，Kodak 上测到 0.01 bpp — [arXiv 2604.10546](https://arxiv.org/html/2604.10546)

**已知幻觉 / 失效模式**
- "Semantic deviation"：在 bpp < 0.05 时，重建看起来可信但细节错误，例如招牌文字被改写、纹理被替换 — [arXiv 2512.06344 摘要](https://awesomepapers.io/generative-models/papers/2512.06344)
- "Semantic collapse"（Zhang, Jia, Li, Liu, "Rethinking Generative Image Compression at Extremely Low Bitrates"，2026-09）。研究 256×256 图、低至 16 bits、0.001–0.008 bpp 的区间。代表性编码器在低于其正常工作码率时不是平滑退化，而是突然崩溃，产生"malformed or unrecognizable content"。作者把原因归为重建损失与语义目标冲突，以及像素空间和重建导向的 VAE 扩散模型保语义效率低。他们提出的 RAE-CoD 相对最佳对手：VFM 特征 MSE 至少 −25.7%，Fréchet Distance ratio 至少 −69.1% — [arXiv 2609.39315](https://arxiv.org/abs/2609.39315)
- 指代式压缩论文指出，粗粒度的视觉-语义对齐加上内在随机性，会重建出语义一致但实例完全不同的物体 [二手] — [arXiv 2606.01608](https://arxiv.org/html/2606.01608)
- GLC 的扩展论文称，在 0.01 bpp 或更低时，现有方法因信息严重失真而无法给出满意重建 [二手] — [arXiv 2512.20194](https://bytez.com/docs/arxiv/2512.20194/paper)
- 纯隐式扩散方案在超低码率下常出现语义错位 — [arXiv 2602.05213](https://arxiv.org/html/2602.05213)
- 综述：Yang & Mandt, "Advances in Diffusion-Based Generative Compression"（2026）。称扩散方法能在极低码率下给出"realistic reconstructions"，并列出若干 open challenges — [arXiv 2601.18932](https://arxiv.org/abs/2601.18932)

### Inferences
- **同预算下的对照。** PerCo 在 0.003 bpp 下每张图不到 153 B，恰好落在本项目的 130–260 B 消息预算内。这个对照说明：在这一字节预算下，图像领域的最佳生成式编码器只能保证分布真实性和粗语义，不能保证几何。如果把同样的思路搬到占据地图，会出现"椅子换位置"或"凭空长出一面墙"，每处都同时贡献 FP 和 FN。
- **0.003 bpp 有欺骗性。** 一张 Kodak 图约 40 万像素（512×768 = 393,216），但图像本身信息冗余极高，0.003 bpp 只是"图像看起来很像"的码率。本项目的几何码是精确无损的，两者目标不同，不能用 bpp 直接类比。
- **生成式编码可以另开一层。** 如果系统需要"补全未观测区域"的可视化，可以作为与精确几何码分离、并显式标记为"推断"的附加层；不应进入 F1 计算的那部分地图。
- **VQ tokenizer 的启示在于丢包鲁棒性。** VQGAN 方案在丢失 20% 索引时仍可恢复，说明离散 token 加学习先验可以带来传输层面的鲁棒性。这与"精确几何码 + 学习熵先验"并不矛盾：先验既能用于熵编码，也能在丢包时做补全，但补全部分必须标记。

### Gaps
- CDC 的具体 bpp 和 FID 数字没有拿到。一份检索摘要称有后续论文把 CDC 描述为工作在约 0.2 bpp 及以上的较高码率，但没有确认是哪篇论文（候选之一是 [arXiv 2404.18820](https://arxiv.org/pdf/2404.18820)），因此未列入 Cited Findings。
- 没有找到在 3D 占据或点云上用检测类指标（F1、IoU）评估生成式编码幻觉率的工作。
- 幻觉目前没有统一的量化指标；上述论文多用 VFM 特征距离或 VLM 打分，不可直接迁移到体素 F1。
- HiFiC、MS-ILLM 的具体工作码率（bpp 值）本次没有从原文核实。

## Q4：Rate–distortion–perception（RDP）理论：何时追求真实感会牺牲失真？对 F1 这类检测指标意味着什么？

### Takeaway
在固定码率下，要求完美真实感（重建分布等于源分布）、且没有共同随机性时，最小 MSE 恰好翻倍，即 3 dB，最优做法是在解码端做后验采样。有共同随机性时代价可以低于 2 倍。Theis et al. 在 DiffC 中观察到，ancestral sampling 比 probability-flow 重建低约 3 dB，与 2 倍理论一致。对本项目，精确无损的几何码在它描述的体素上 D = 0，RDP 权衡不存在。权衡只出现在没传的部分（射线推断出的 free space、未观测区）；在那里，F1 最优的做法是对后验边缘概率取阈值，而不是采样。

### Cited Findings
- Blau & Michaeli, "The Perception-Distortion Tradeoff"（CVPR 2018）。失真与感知质量之间存在不可兼得的权衡 — [ResearchGate](https://www.researchgate.net/publication/321124804_The_Perception-Distortion_Tradeoff)
- Blau & Michaeli, "Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff"（ICML 2019）。构造出一种编解码器：达到完美感知质量，失真只有 MMSE 编码器的两倍 — [arXiv 1901.07821](https://arxiv.org/pdf/1901.07821)
- Yan, Wen, Ying, Ma, Liu, "On Perceptual Lossy Compression: The Cost of Perceptual Reconstruction and An Optimal Training Framework"（ICML 2021）。完美感知质量的代价是"exactly a doubling of the lowest achievable MSE distortion"；为经典 RD 最优的编码器同时也适用于完美感知压缩（失真为两倍） — [arXiv 2106.02782](https://arxiv.org/abs/2106.02782)
- 完美真实感约束下 D(0) = 2D(∞)，即 MMSE 的两倍，由满足 P_{X̂|Y} = P_{X|Y} 的后验采样解码器达到；没有共同随机性时必须恰好翻倍 [二手] — [Wagner, arXiv 2202.04147](https://arxiv.org/pdf/2202.04147)；[RDP 综述, PMC12025864](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025864/)
- Wagner, "The Rate-Distortion-Perception Tradeoff: The Role of Common Randomness"（2022）。刻画了有限共同随机性下的 RDP 函数；共同随机性无限时退化回原 RDP 函数 — [arXiv 2202.04147](https://arxiv.org/abs/2202.04147)
- Zhang et al., "Universal Rate-Distortion-Perception Representations for Lossy Compression"（NeurIPS 2021）。同一个编码表示可以服务多个 (D, P) 工作点，只需更换解码器 — [NeurIPS 2021 PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/5fde40544cff0001484ecae2466ce96e-Paper.pdf)
- Hamdi, Gündüz, Wagner, "Rate-Distortion-Perception Trade-off with Strong Realism Constraints: Role of Side Information and Common Randomness"（2025）。研究解码端侧信息与共同随机性在强真实感约束下的作用 — [arXiv 2507.14825](https://arxiv.org/abs/2507.14825)
- 一项 2025 年的工作（algorithmic realism，批量设定）指出：批大小随 n 次指数增长时不需要共同随机性；指数增长时需要。这不是标准设定 — [arXiv 2511.15255](https://arxiv.org/html/2511.15255)
- 实证对应一：DiffC 中 probability-flow 重建比 ancestral sampling 高约 3 dB（高码率） — [arXiv 2206.08889](https://arxiv.org/abs/2206.08889)
- 实证对应二：PerCo 比 MS-ILLM 更真实，但 PSNR、MS-SSIM、LPIPS 显著更差 [二手] — [PerCo ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/43e9d647ccd3e4b7b5baab53f0368686-Paper-Conference.pdf)
- 实证对应三：MS-ILLM 作者指出，失真与统计保真度需要联合权衡 — [arXiv 2301.11189](https://arxiv.org/abs/2301.11189)

### Inferences
- **二值体素 + Hamming 失真的直接推论（推导，非文献数字）。** 设接收端对某体素的后验占据概率为 p。MAP（阈值 0.5）的期望错误率是 min(p, 1−p)；后验采样的期望错误率是 2p(1−p)。因为 2p(1−p) ≤ 2·min(p, 1−p)，在 p 接近 0.5 时两者之比趋近 1，在 p 接近 0 或 1 时趋近 2。这正是 RDP 的"至多翻倍"在二值情形的体现：追求真实感的解码在不确定区域最多让体素错误率翻倍。
- **对 F1 的含义。** 后验采样会在"可能为空"的区域随机放置占据（FP），在"可能占据"的区域随机挖空（FN），precision 和 recall 两头受损。F1 不能按体素逐项分解，F1 最优阈值一般不是 0.5，需要按验证集选取。但无论阈值取多少，都应是对后验边缘概率取阈值的确定性决策，而不是采样。
- **精确几何码绕开了 RDP 权衡。** 几何码对它描述的体素是无损的（X̂ = X），失真与真实感同时为 0，RDP 权衡不出现。权衡只出现在两处：（1）接收端用射线投射推断 free space 时的系统性误差；（2）未观测区域。只要不对这两部分做"生成式补全"，F1 就不会因追求真实感而受损。这从理论上支持"精确几何码 + 学习熵先验"，即用学习先验降码率，而不是用生成解码器补内容。
- **侧信息与共同随机性的关联。** 如果将来确实想做未观测区补全，Hamdi–Gündüz–Wagner 的结果提示：接收端自己的地图（侧信息）和收发两端共享的随机种子（共同随机性）都能降低真实感的代价。多智能体系统天然具备这两样（共享模型和种子、共享历史消息）。

### Gaps
- 没有找到专门针对 F1、IoU 或集合型检测指标的 RDP 理论结果；上面的二值推导是自行推导，未见文献。
- 没有找到"二值源 + Hamming 失真 + 完美真实感"的闭式 RDP 曲线文献（一般理论覆盖 MSE 较多）。

## Q5：分布式 / 侧信息学习编码（neural Wyner–Ziv、distributed source coding、学习式视频编码中的 conditional coding，如 DCVC 系列）：接收端侧信息能带来多少增益？

### Takeaway
接收端（decoder-only）侧信息在学习式立体图像压缩中带来 10%–50% 的带宽节省（Ayzik & Avidan，KITTI Stereo）。学习式 Wyner–Ziv 编码器在合成信源上能自发学到 binning，部分情形逼近信息论界。如果侧信息两端都有，即 conditional coding，DCVC 系列相对 VTM 自报约 −21% 到 −25%（相对 x265 veryslow −26%），而且 DCVC-RT 已在 1080p 上达到约 113–125 fps。我没有找到量化"decoder-only 侧信息"与"两端共享侧信息"之间差距的学习式实验，也没有找到点云或占据数据上的学习式分布式编码。

### Cited Findings
**Decoder-only 侧信息（Wyner–Ziv / DSC）**
- Ayzik & Avidan, "Deep Image Compression using Decoder Side Information"（ECCV 2020）。用接收端的相关图像做侧信息，带宽节省 10%–50%，具体取决于失真水平和侧信息相关性。KITTI Stereo 上 MS-SSIM 0.93 时，码率约从 0.05 bpp 降到 0.03 bpp（约 −40%）[二手] — [ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620681.pdf)；[arXiv 2001.04753](https://arxiv.org/abs/2001.04753)
- Mital, Ozyilkan, Garjani, Gündüz, "Neural Distributed Image Compression with Cross-Attention Feature Alignment"（WACV 2023）。在 KITTI 和 Cityscapes 立体数据上明显优于前作 NDIC，低码率下增益更明显；摘要没有给数字 — [arXiv 2207.08489](https://arxiv.org/abs/2207.08489)
- Mital et al. 的 NDIC（DCC 2022）利用 Wyner common information 做分布式图像压缩 — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/Neural-distributed-image-compression-using-common-information.pdf)
- 一篇 2024 年的论文用 cross-attention 融合侧信息，在 KITTI Stereo 和 Cityscapes 上称超过 SOTA，摘要中没有数字 — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352864824000683)
- Whang, Nagle, Acharya, Kim, Dimakis, "Neural Distributed Source Coding"（2021，JSAIT 2024）。基于条件 VQ-VAE，不依赖相关结构的先验形式，可扩展到高维；称多个数据集上 PSNR 达到 SOTA，摘要没有给数字 — [arXiv 2106.02797](https://arxiv.org/abs/2106.02797)
- Özyılkan, Ballé, Erkip, "Learned Wyner–Ziv Compressors Recover Binning"（ISIT 2023）。在二次高斯情形下，学习式编码器在没有任何先验分布知识的情况下自发学到源空间 binning，并在每个量化索引内学到线性解码 — [arXiv 2305.04380](https://arxiv.org/abs/2305.04380)。扩展版（JSAIT 2024）用变分向量量化，同样学到 binning 和索引与侧信息的最优组合 — [arXiv 2310.16961](https://arxiv.org/abs/2310.16961)
- 综述（"Distributed Compression in the Era of Machine Learning"，2024）。称学习式 WZ 编码器配高阶熵编码时，在抽象信源上优于点对点 RD 函数，部分情形达到有侧信息时的信息论最优界。该综述**没有**给出 decoder-only 与两端共享（conditional）之间差距的数字 — [arXiv 2402.07997](https://arxiv.org/html/2402.07997)
- 一篇 2026 年的工作研究极低码率下的多模态侧信息分布式图像压缩（未读全文） — [arXiv 2605.22061](https://arxiv.org/html/2605.22061)

**Conditional coding（两端共享上下文：DCVC 系列）**
- DCVC（Li, Li, Lu, "Deep Contextual Video Compression"，NeurIPS 2021）。从残差（predictive）编码转向特征域条件编码；作者认为 predictive coding 对压缩率"only suboptimal"。1080p 标准测试序列上相对 x265 veryslow 省 26.0% — [arXiv 2109.15047](https://arxiv.org/abs/2109.15047)
- DCVC-FM（"Neural Video Compression with Feature Modulation"，CVPR 2024）。intra-period −1 设定下相对 VTM −25.5%；YUV420、intra-period 32 下相对 VTM −24.1%。在同一张 YUV420 / IP −1 表中，DCVC-DC 为 +12.8%（比 VTM 差）；而在 RGB / IP32 设定下，DCVC-DC 相对 VTM −16.1%。结论是同一方法的增益强烈依赖测试协议 — [arXiv 2402.17414](https://arxiv.org/html/2402.17414v1)
- DCVC-RT（"Towards Practical Real-Time Neural Video Compression"，CVPR 2025）。相对 VTM −21.0%（ECM 为 −20.5%）。A100 fp16 上 1080p 编码 125.2 fps、解码 112.8 fps。int16 版本 −18.3%，Large 版本 −30.8% — [arXiv 2502.20762](https://arxiv.org/pdf/2502.20762)
- 有其他论文报告 DCVC-RT 在 random-access（B 帧）设定下表现较弱，属于不同配置，不能与上面的 low-delay 数字直接比较 [二手] — [arXiv 2506.07709](https://arxiv.org/pdf/2506.07709)
- 有一篇第三方评测文章（Streaming Learning Center）讨论 DCVC-RT 的速度与压缩表现，我没有核实其中的数字 — [streaminglearningcenter.com](https://streaminglearningcenter.com/articles/evaluating-dcvc-rt-a-real-time-neural-video-codec-that-delivers-on-speed-and-compression.html)

### Inferences
- **两种设定。** 本项目里接收端自己的地图就是侧信息，分两种情形：
  - **情形 A，conditional coding。** 发送端确切知道接收端已有什么，例如双方都持有之前已确认收到的消息，或共享的公共地图版本。这时可以像 DCVC 一样，让熵模型以共享上下文为条件，只编码"新信息"。文献中这一路线的增益在视频上是相对强经典编码器再省约 20%–26%，且已有实时实现。
  - **情形 B，Wyner–Ziv。** 发送端不知道接收端地图的细节。学习式 WZ 在立体图像上有 10%–50% 的增益，但这是有损、长块（整幅图）的结果。
- **短消息下应优先选情形 A。** 本项目是约 150 B 的无损短消息。Slepian–Wolf 的"无损分布式编码无渐近损失"只在长块长下成立；短块长下的有限长度惩罚可能吃掉大部分增益（推断，未找到短块长数字）。因此更稳妥的设计是：用确认机制让双方都知道已同步的公共历史，把它变成"共同侧信息"，然后对新几何信息做条件熵编码（情形 A），而不是依赖 decoder-only 侧信息（情形 B）。
- **"接收端射线投射 free space"本身就是一种侧信息利用。** 发送端只传占据和位姿，free space 由接收端根据共享的几何规则重建，这等价于把部分信息"移到解码端的确定性计算"。学习式条件熵模型可以进一步以"接收端已知的占据 / free 体素"为上下文，给出新占据体素的更尖锐概率。
- **预注册判据建议。** 对条件编码的增益也应预注册，例如在不同重叠率分档下测量"条件熵编码 vs 非条件熵编码"的字节差。DCVC-DC 与 DCVC-FM 的协议敏感性提醒我们：增益会随重叠率或协议剧烈变化。

### Gaps
- 没有找到学习式分布式 / 条件编码在点云、占据栅格或多机地图消息上的结果。
- 没有找到量化"decoder-only 侧信息 vs 两端共享侧信息"差距的学习式实验；综述也没有给出。
- 短块长（约 1000–2000 bits）下 Slepian–Wolf 或 Wyner–Ziv 实际编码的有限长度惩罚，本次没有找到带数字的来源。
- Whang et al. NDSC、Mital et al. 各方法的具体码率节省数字，本次只拿到摘要层面的定性描述。
