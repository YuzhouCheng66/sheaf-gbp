# 少步（few-step / one-step）生成模型能否充当三维占据地图的条件熵模型或去噪先验

> 背景：8 机器人，128×128×32 体素（524,288 voxels），ternary（unknown/free/occupied），带家具的室内房屋。当前设计：消息 = 精确几何码（观测到的 occupied 体素 / floor plan + boxes + 传感器位姿，接收端用 ray casting 重新推出 free）；学到的生成先验只用作 arithmetic coding 的 entropy model，不作为地图输出；带噪传感器下可兼作 denoiser。内部探针：从生成模型采样出地图，比对边缘概率阈值化，家具 F1 低 6–22 点。
> 检索日期：2026-10-27。"实测"指论文表格中的数值；"声称"指摘要或二手摘要中的表述，未逐表核对。

---

## Q1. 哪些少步/一步生成模型族能给出可用于 arithmetic coding 的精确或有界似然？代价如何？

### Takeaway
对 arithmetic coding 有用的，只有"按固定分组顺序、组内因子化的离散条件分布"这一族：AR、order-agnostic ARDM / absorbing（masked）diffusion 取固定顺序、checkerboard / M2T 式分组 context model、多尺度（octree / next-scale）预测。这一族的码长是精确的，并且步数减少时退化平缓：CIFAR-10 上从 3072 步降到 50 步，2.69 → 2.74 bpd，约多 2%。连续 Gaussian diffusion 只给 ELBO，少步时急剧变差（10 步时 4.31 vs 2.65 bpd，多约 63%），而且编码必须走 bits-back，单条消息的开销是无穷大。flow matching 的 ODE 似然要经过 dequantization 和随机 trace 估计，不能直接当 PMF 用。Consistency models、MeanFlow、shortcut models 完全不提供似然，不能直接当熵模型。

### Cited Findings

**连续 Gaussian diffusion / VDM（ELBO，界）**
- VDM（Kingma, Salimans, Poole, Ho, NeurIPS 2021）测试集 bits/dim：CIFAR-10 无增广 2.65、有增广 2.49；ImageNet 32×32 3.72；ImageNet 64×64 3.40。2021 年时均为 SOTA，后续是否已被超越本次未核对。— [VDM, arXiv 2107.00630](https://arxiv.org/abs/2107.00630)
- VDM 证明 continuous-time VLB 对 noise schedule 不变，只依赖两端的 SNR。— [VDM](https://arxiv.org/abs/2107.00630)
- **少步惩罚（实测，VDM Table 2，CIFAR-10）**：
  - 离散时间训练与评估都取 T 步时：T=10 为 4.31，T=100 为 2.84，T=250 为 2.73，T=500 为 2.68，T=1000 为 2.67，T=10000 为 2.66 bpd。
  - 连续时间训练的模型取 T_eval=10 评估时为 7.54 bpd。
  - 结论：10 步时界比 ∞ 步（2.65）差约 63%，按连续时间训练的模型差约 185%。— [VDM](https://arxiv.org/abs/2107.00630)
- **bits-back 实际码长（实测）**：连续时间模型用 BB-ANS 编码，T_eval=100 时净码长 2.91 bpd，T_eval=500 和 T_eval=1000 时都是 2.72 bpd，与理论 ELBO（2.69 / 2.67）存在差距。作者自述"for large T_eval a gap remains……and compression becomes computationally expensive"。— [VDM](https://arxiv.org/abs/2107.00630)
- VDM 的训练成本：CIFAR-10 在 8 块 TPUv3 上以 batch 128 训练 1000 万步，用 9 天达到 2.65 bpd；2.5 小时时已越过 Sparse Transformer 的 2.80。— [VDM](https://arxiv.org/abs/2107.00630)

**bits-back 的单条消息问题（与"每个机器人发一条地图消息"直接相关）**
- ARDM 论文 Table 3（CIFAR-10，bpd）按"逐图压缩 / 整数据集压缩"两列报告。原表符号为 ∞，pdftotext 渲染成 "8"，按上下文判断为 ∞。
  - VDM bits-back：逐图 ∞，整集 2.72（1000 步）、2.72（500 步）、2.91（100 步）。
  - LBB（flow + local bits-back）：逐图 ∞，整集 3.12。
  - HiLLoC：逐图 4.19（与 FLIF 相同，因为初始比特由 FLIF 填充），整集 3.56。
  - 作者原话："bits-back coding methods……have a large overhead when used as per-image compressors……caused by the large number of initial bits"，且数据流是串联的，取中间一张须先解码前面所有图。
  - 来源：[Hoogeboom et al., Autoregressive Diffusion Models, ICLR 2022, arXiv 2110.02037](https://arxiv.org/abs/2110.02037)

**离散 diffusion / masked（absorbing）diffusion / ARDM（精确码长）**
- ARDM 的定位：OA-ARDM 等价于 D3PM-absorbing 的连续时间极限。它可直接配 rANS 做逐图无损压缩，不需要 bits-back。实现上固定一个编码顺序；不同随机排列之间差别 < 0.02 bpd。— [ARDM](https://arxiv.org/abs/2110.02037)
- **ARDM 少步实测（CIFAR-10，bpd）**：
  - ARDM-OA：3072 步 2.69；Parallel ARDM-OA 50 步 2.74。
  - ARDM-Upscale4：4×3072 步 2.64；4×50 步 2.68。
  - 逐图压缩：OA-ARDM 500 步 2.73，100 步 2.75；ARDM-Upscale4 500 步 2.71，100 步 2.76。
  - 对照：IDF++ 3.26、IDF 3.34、FLIF 4.19。
  - 来源：[ARDM](https://arxiv.org/abs/2110.02037)
- **text8（bpc）**：
  - OA-ARDM 250 步 1.43；Parallelized OA-ARDM 20 步 1.51。
  - D3PM-absorbing 1000 步 1.45、256 步 1.47、20 步 1.56；D3PM-uniform 1000 步 1.61。
  - 来源：[ARDM](https://arxiv.org/abs/2110.02037)
- **D3PM 在 CIFAR-10 上的似然**（引自 ARDM Table 2）：D3PM-absorbing 1000 步 4.40 bpd；D3PM-Gaussian 1000 步 3.44 bpd。该结果已被 ARDM、MD4 超越。— [ARDM](https://arxiv.org/abs/2110.02037)
- ARDM 的训练成本：CIFAR-10 在 8 块 TPUv4 上训练约 2 周（3000 epochs），"a couple of days" 后已达约 2.8 bpd。— [ARDM](https://arxiv.org/abs/2110.02037)
- **MD4**（Shi et al., NeurIPS 2024，masked diffusion）：连续时间 ELBO 是加权的交叉熵积分；像素级 CIFAR-10 2.75 bpd，ImageNet 64×64 3.40 bpd，声称优于同规模 AR。本次未找到把 MD4 实际跑成压缩器的结果。— [MD4 代码](https://github.com/google-deepmind/md4)；[MD4 摘要页](https://liner.com/ko/review/simplified-and-generalized-masked-diffusion-for-discrete-data)

**Flow matching / CNF（ODE 似然）**
- Flow Matching（Lipman et al., ICLR 2023）Table 1，FM w/ OT 的 NLL（bpd）/ FID / 采样 NFE：
  - CIFAR-10：2.99 / 6.35 / 142
  - ImageNet 32：3.53 / 5.02 / 122
  - ImageNet 64：3.31 / 14.45 / 138
  - ImageNet 128：2.90 / 20.9 / NFE 未报告
  - 同表 DDPM 在 CIFAR-10 上为 3.12 / 7.48 / 274。
  - 来源：[Flow Matching, arXiv 2210.02747](https://ar5iv.labs.arxiv.org/html/2210.02747)
- 上述 NLL 的计算方式：uniform dequantization 加 importance-weighted 估计（K 取 1–50）；ODE 用 dopri5 求解，atol/rtol = 1e-5；divergence 用 Hutchinson trace estimator（无偏随机估计）。似然评估所需 NFE 未单独报告。— [Flow Matching](https://ar5iv.labs.arxiv.org/html/2210.02747)

**AR / next-scale prediction（VAR）**
- VAR（Tian et al., NeurIPS 2024 best paper）：ImageNet 256 上用 10 个 scale step；每步对一个 scale 的全部 token "generated in parallel"。
  - FID：d16（310M）3.30，d20（600M）2.57，d24（1.0B）2.09，d30（2.0B）1.92，d30-re（拒绝采样）1.73。
  - 推理速度约为 VQGAN / ViT-VQGAN 的 20 倍。
  - 不报告 perplexity / NLL，只在 scaling 图里给出 test cross-entropy。
  - 来源：[VAR, arXiv 2404.02905](https://arxiv.org/html/2404.02905v2)

**Consistency / shortcut / MeanFlow（无似然）**
- MeanFlow 原文不讨论 likelihood、density、compression 或 inverse problems。— [MeanFlow HTML](https://arxiv.org/html/2505.13447v1)
- 相关一步模型的数值（引自 MeanFlow Table 2）：
  - iCT-XL/2（consistency）ImageNet 256：1-NFE FID 34.24，2-NFE 20.30。
  - Shortcut-XL/2：1-NFE 10.60。
  - iMM-XL/2：1×2 NFE 7.77。
  - 来源：[MeanFlow](https://arxiv.org/html/2505.13447v1)
- sCM（Lu & Song, ICLR 2025）2 步 FID：ImageNet 512 1.88（1.5B 参数）、CIFAR-10 2.06、ImageNet 64 1.48。OpenAI 称单个样本在单张 A100 上 0.11 s。sCM 已取代 2023 年原版 CM 的数字。— [sCM, arXiv 2410.11081](https://arxiv.org/abs/2410.11081)；[OpenAI blog](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)

**分组并行 context model：工程界已在用的"少步精确熵模型"**
- Checkerboard context model（He et al., CVPR 2021）：两遍解码，anchor 只依赖 hyperprior，其余半数用 checkerboard context。声称解码加速 >40 倍，"almost the same rate-distortion performance"。具体 BD-rate 损失本次未取得。— [Checkerboard, arXiv 2103.15306](https://arxiv.org/abs/2103.15306)
- M2T（Mentzer, Agustsson, Tschannen, ICCV 2023）：把 MaskGIT 式 masked transformer 用作熵模型。发现预定义的确定性 schedule 在压缩上与 MaskGIT 的不确定性自适应 schedule 一样好或更好。固定 schedule 允许 masked attention 和 activation caching，推理约快 4 倍，码率略增。— [M2T, arXiv 2304.07313](https://arxiv.org/abs/2304.07313)
- SparsePCGC（点云几何 / 占据压缩）：多尺度 sparse tensor 占据编码。SOPA 概率模型可以单阶段（只用 cross-scale prior），也可以多阶段（逐步利用自回归邻居）。声称在 KITTI、Ford 等数据集上，无损和有损模式均为 SOTA（对比 G-PCC）。具体分组数与 bpp 本次未取得。— [SparsePCGC, arXiv 2111.10633](https://arxiv.org/abs/2111.10633)

**确定性（多机互通时的硬约束）**
- Ballé, Johnston, Minnen（ICLR 2019）指出：只要发送端与接收端算出的先验有任何浮点差异（跨硬件/软件的 round-off），range coding 就会灾难性失败。他们提出 integer networks 来保证跨平台一致。— [Integer Networks, Google Research](https://research.google/pubs/integer-networks-for-data-compression-with-latent-variable-models/)
- 后续工作的结论：
  - 对带 context model 的较复杂模型，integer network 难以保持性能损失可忽略，因此改用 post-training quantization。— [PTQ for cross-platform LIC, arXiv 2202.07513](https://arxiv.org/pdf/2202.07513)
  - 另有方案改为量化关键输出并附带 safeguarding bitstream。— [Towards Reproducible Learning-based Compression, arXiv 2410.09872](https://arxiv.org/pdf/2410.09872)

### Inferences
- **熵模型只看 NLL，不看样本质量。** 码长为 −log₂ q(x)。FID 再好的一步生成器，如果给不出 q(x)，就不能直接用于 arithmetic coding。MeanFlow、CM、shortcut 只输出样本（Dirac 映射），不提供密度，因此对"熵模型"这一角色帮助为零。它们顶多能当特征/预测器，前提是再接一个用交叉熵训练的逐体素概率头；那时有效的部分其实就是一个判别式条件模型。
- **"少步精确熵模型" = K 组固定顺序、组内条件独立的分解** p(x)=∏_k ∏_{i∈G_k} p(x_i | x_{G_<k}, context)。这对应 K 次网络前向和 K 次 arithmetic coding 流程，码长精确，而且编码端和解码端只需在概率上逐位一致。ARDM 的 50 步对 3072 步只多约 1.9% bpd，checkerboard 两遍、M2T 固定 schedule 也显示 K 很小时损失很小。对 128×128×32 的占据栅格，现成形式有三种：
  - 2–8 组 checkerboard / 3D 八分组（2×2×2 parity）；
  - octree / 多尺度 next-scale（对应 VAR 和 SparsePCGC 的思路，但必须用无损的 max-pool 占据金字塔，不能用有损 VQ tokenizer）；
  - 固定 schedule 的 masked diffusion（M2T / ARDM）。
- **不宜采用的路线：连续 diffusion/flow + bits-back。** 理由有四点：
  1. 每台机器人每次只发一条消息，bits-back 的初始比特开销无法摊销，ARDM 表中逐图码长为 ∞；
  2. 少步时 ELBO 大幅变差（10 步 +63%）；
  3. ODE 似然依赖随机 Hutchinson 估计与 dequantization，不能作为确定的 PMF；
  4. 跨 8 台机器的浮点确定性要求还会额外放大工程难度。
- VAR 本身的似然是 VQ token 的似然，不是原始体素的似然；它的 tokenizer 有损，因此不能直接用于精确几何码。可借用的只是"next-scale、组内并行"这一因子化结构。

### Gaps
- 未找到 MD4 / MDLM / discrete flow matching 实际跑成无损压缩器的结果，也未找到它们在少于 10 步下的码长曲线（只有 ARDM、VDM 的数据）。
- Checkerboard、M2T 在无损（而非有损 latent）场景下相对全串行 AR 的精确码率损失百分比，以及 SparsePCGC 的组数与 KITTI bpp，本次都没有取到原表。
- 未核实 2024–2026 年 CIFAR-10 / ImageNet64 似然 SOTA 是否已低于 VDM 2.65 和 ARDM-Upscale 2.64。
- 未找到关于 "consistency model / flow map 似然估计" 的可靠论文（检索无果）。

---

## Q2. MeanFlow 实际提出了什么？报告的数字是多少？有没有人把 MeanFlow / consistency 模型用于压缩、似然估计或条件去噪？

### Takeaway
MeanFlow（Geng, Deng, Bai, Kolter, He；NeurIPS 2025 oral）用平均速度 u 代替瞬时速度 v，通过 MeanFlow identity 从零训练一步生成器。ImageNet 256 上 1-NFE FID 为 3.43（XL/2，676M，240 epochs）。这个数字已被 iMF 的 1.72（CVPR 2026）取代。原文不涉及似然或压缩。现有"一步扩散用于压缩"的工作全是把生成器当有损感知解码器，与我们"生成先验只当熵模型"的设计正好相反。MeanFlow 用于逆问题/条件去噪的工作只有 2026 年几篇预印本，尚无标准基准上的头对头比较。

### Cited Findings
- **定义（实测原文）**：
  - 平均速度 u(z_t, r, t) = (1/(t−r)) ∫_r^t v(z_τ, τ) dτ。
  - MeanFlow identity：u = v − (t−r)·(d/dt)u，其中 d/dt u = v·∂_z u + ∂_t u。
  - 训练损失：‖u_θ − sg(u_tgt)‖²，u_tgt = v_t − (t−r)(v_t ∂_z u_θ + ∂_t u_θ)。
  - 所需 JVP 用 jvp 计算，tangent 取 (v, 0, 1)；stop-gradient 避免二阶反传。
  - 来源：[MeanFlow HTML](https://arxiv.org/html/2505.13447v1)
- **训练开销**：B/4 消融（JAX，TPU v4-8）中，FM 为 0.045 s/iter，MeanFlow 为 0.052 s/iter，约多 16%；作者称"less than 20%"。无需 pre-training、distillation 或 curriculum。— [MeanFlow](https://arxiv.org/html/2505.13447v1)
- **ImageNet 256（从零训练，240 epochs）**：
  - B/2（131M）1-NFE 6.17；M/2 1-NFE 5.01（参数量在 Table 2 写作 308M，Table 4 写作 497.8M，原文自相矛盾）；L/2（459M）1-NFE 3.84；XL/2（676M）1-NFE 3.43，2-NFE 2.93。
  - XL/2+ 训练 1000 epochs，2-NFE 2.20。
  - 对照：DiT-XL/2 250×2 NFE 2.27；SiT-XL/2 2.06。
  - 来源：[MeanFlow](https://arxiv.org/html/2505.13447v1)
- **CIFAR-10 无条件**：MeanFlow 1-NFE FID 2.92，约 55M U-Net，不用 preconditioner。同表 iCT 2.83、ECT 3.60、sCT 2.97、IMM 3.20。— [MeanFlow](https://arxiv.org/html/2505.13447v1)
- **后续（取代原数字）**：
  - iMF（Improved Mean Flows, arXiv 2512.02012, CVPR 2026）：改成以瞬时速度为回归目标，guidance 作为条件输入。1-NFE FID 1.72（最大模型）；118M 模型 1-NFE 2.74。二手中文媒体称 2-NFE 为 1.54，未在原文核实。— [iMF](https://arxiv.org/html/2512.02012v1)；[CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/37834)
  - AlphaFlow（arXiv 2510.20771）：最大 vanilla-DiT 模型 1-NFE 2.58、2-NFE 2.15（数字来自检索摘要）。— [AlphaFlow](https://arxiv.org/abs/2510.20771)
  - Re-MeanFlow（arXiv 2511.23342）：先做一次 reflow 拉直轨迹再学 mean velocity，声称在 ImageNet 64/256/512 上优于既往一步蒸馏方法。— [Re-MeanFlow](https://arxiv.org/abs/2511.23342)
- **Rectified flow 系**：2-rectified flow++（arXiv 2405.20320）声称 CIFAR-10 1-NFE FID 比原 2-rectified flow 降低最多 72%（v1）或 75%（v2），两版不一致。二手摘要给出 3.38，未核实。SlimFlow 的一步 CIFAR-10 结果：15.7M 参数 FID 5.02，27.9M 参数 4.53。— [RF++](https://arxiv.org/abs/2405.20320)；[SlimFlow](https://arxiv.org/html/2407.12718v2)
- **一步扩散用于压缩（均为有损、感知导向，生成器作解码器）**：
  - OneDC（MSRA, arXiv 2505.16687）：latent codec 加一步扩散生成器，hyperprior 作语义信号；声称码率降低 >39%，解码比多步扩散 codec 快 20 倍。— [OneDC](https://arxiv.org/html/2505.16687v2)
  - OSCAR（单模型支持多码率）。— [OSCAR](https://arxiv.org/pdf/2505.16091v2)
  - StableCodec（ICCV 2025，<0.05 bpp 的极低码率）。— [StableCodec](https://iccv.thecvf.com/virtual/2025/poster/2634)
  - DiffO：去噪强度随码率自适应，解码约快 50 倍。— [DiffO](https://arxiv.org/html/2506.16572v1)
- **MeanFlow / 一步模型用于逆问题与条件去噪（2026 预印本，结果均未核实）**：
  - NullFlow（arXiv 2606.22696）：在 subspace-restricted MeanFlow 上做一步后验采样，从 pseudo-inverse 加 null-space 噪声出发。— [NullFlow](https://arxiv.org/pdf/2606.22696)
  - Cheng et al.（arXiv 2603.14798）：conditional MeanFlow 加 neural operator，用于函数空间 Bayesian 逆问题。— [2603.14798](https://arxiv.org/html/2603.14798v1)
  - Physics-Informed MeanFlow（NeurIPS 2026 listing）。— [PIMF](https://neurips.cc/virtual/2026/166462)
  - Amortized Posterior Sampling（arXiv 2407.17907）：把扩散先验蒸馏成条件 flow，1-NFE 出后验样本。— [APS](https://arxiv.org/html/2407.17907v1)
  - FlowDPS：多步，不是一步。— [FlowDPS](https://arxiv.org/pdf/2503.08136)
- 检索未发现任何把 consistency model 用于似然估计的工作。

### Inferences
- MeanFlow 的价值在于"一步出样本"，这恰好是我们设计中不需要的那一部分：我们既不输出生成地图，也需要 q(x) 而不是样本。可能的用途只剩两种：
  1. 作为条件去噪器 / 后验均值的近似器。一步模型的 1-NFE 输出在 t→0 时接近 E[x|z]（但 MeanFlow 是对整个区间求平均，并不保证输出后验均值）。
  2. 作为合成训练数据的生成器，类似 GSSC 用生成模型造配对数据（见 Q3）。
- 一步有损生成式 codec 的"解码器幻觉"风险，正是我们选择精确几何码的原因。这类 codec 的 >39% 码率收益是在感知指标（LPIPS/FID）下取得的，不能迁移到"占据体素必须精确"的任务上。
- 训练开销方面：MeanFlow 比 FM 多约 16% 的 JVP 开销，这对小规模实验不构成障碍；但 3D U-Net 的 JVP 显存在 128×128×32 上有多大，没有公开数据。

### Gaps
- 没有找到 MeanFlow、iMF、sCM 在离散/类别数据（如体素占据）上的任何实验。
- NullFlow、PIMF 等 2026 预印本的定量结果未逐表核实，也没有它们相对 DPS/DDRM 的标准对比。
- 一步模型能否近似给出 posterior mean 或逐体素边缘概率，理论上不明确，本次未找到研究。

---

## Q3. diffusion / flow 在体素、占据、点云上的应用：部分观测条件生成、后验采样、去噪，质量与算力

### Takeaway
场景级 3D diffusion 在 2024–2026 年已能做到生成和补全，但凡涉及逐体素精度（mIoU/IoU），从噪声生成的模型普遍不如判别式 SSC 网络。最好的做法是把判别网络的 argmax 当作起点，再做 1 步生成式 refinement（GSSC 的 S²D²，+2.1–2.4 mIoU）。这一结构与"阈值化边缘概率 + 先验修正"一致，而与"从先验采样"相反。数据规模与我们的计划相当（约 5k–50k 场景），训练成本在单机 2–4 卡、数天到两周量级。

### Cited Findings
- **PDD（Pyramid Discrete Diffusion, ECCV 2024）**：
  - 设置：尺度 32×32×4 → 64×64×8 → 256×256×16（s4 拆成 136×136×16 子场景）；T=100（训练和采样）；3D U-Net；每个模型用 4×A100；AdamW，最多 800 epochs；可在 SemanticKITTI 上再微调 200 epochs。
  - 作者称：原始 discrete diffusion 在 256×256×16 上训练需要超过 16 天（4×A100）。
  - 数据：CarlaSC 共 32,400 scans（18/3/3 场景划分）。
  - 无条件生成（CarlaSC，用分割网络在生成场景上测 mIoU(V)）：GT 52.19，DiscreteDiff 40.05，LatentDiff 38.01，PDD 68.02（PDD 高于 GT，说明这一指标衡量的是"可分割性"而不是保真度）。
  - 以粗场景为条件：DiscreteDiff 52.52，PDD 55.75；以点云为条件：DiscreteDiff 38.55。
  - 推理时间未报告。
  - 来源：[PDD, arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)
- **LT3SD（CVPR 2025，室内带家具房屋，与我们场景最接近）**：3D-FRONT 过滤后剩 6,479 栋房屋（80/5/15 划分）。latent tree 编解码器每个分辨率层训练 5 小时到 1 天；扩散阶段在 2 张 RTX A6000 上约 6 天收敛。— [LT3SD, arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- **DiffSSC（Cao & Behnke 2024，LiDAR 语义补全，点级 DDPM）**：IoU 63.4（表中最高），mIoU 27.4。对照：SCPNet 56.1/36.7，TALoS 60.2/37.9，S3CNet 45.0/29.5，JS3C-Net 56.6/23.8，LMSCNet 55.3/17.0。即几何 IoU 好，语义 mIoU 比最强判别式方法低约 10 点。— [DiffSSC, arXiv 2409.18092](https://arxiv.org/pdf/2409.18092)
- **GSSC（Generative Semantic Scene Completion, arXiv 2608.26737, 2026 预印本，Fudan/CMU）**：
  - 方法：一套 multinomial discrete diffusion 承担三个角色：合成配对数据（PS³，32,039 对）、从噪声生成（SGSC）、一步 refinement（S²D²：源为冻结判别网络的 argmax，直线插值，一次 Euler 更新）。去噪器为约 35M 参数的 4 层 3D U-Net。
  - mIoU 数字：
    - SCPNet 公开 hidden test 36.7 → S²D² 一步、无 TTA 38.8（+2.1）；4 步 + 8 视角 TTA 39.2。
    - val：36.17 → 38.54（+2.36），19 类中 18 类提升。
    - 从噪声生成（SGSC，无判别 base）只有 30.5，经 S²D² 修正后 32.8。
  - 步数：N=1 为 38.54，N=4 为 38.65，N=100 为 38.2，差异在单 seed 噪声内。
  - 代价：每次修正 107 ms；整条管线 3.23 FPS，判别网络单独为 4.95 FPS。
  - 负面：在弱 base（JS3C-Net）上一步修正会抹掉稀有类，person IoU 在 74.7% 的帧上下降，bicyclist 在 92.5% 的帧上下降。motorcyclist +8.3 在重训后只剩 +0.3（不可复现）。
  - 以上来自 HF 论文页摘要，未逐表核实。— [GSSC (HF papers)](https://huggingface.co/papers/2608.26737)
- **LiDiff（Nunes et al., CVPR 2024）**：点级加噪，不受体素分辨率限制。在场景级补全上，形状级扩散基线 PVD 表现最差，作者认为形状补全扩散不能直接迁移到场景尺度。— [LiDiff, arXiv 2403.13470](https://arxiv.org/pdf/2403.13470)
- **其他**：
  - Octree Latent Diffusion（arXiv 2509.16483）：先 structure diffusion 生成粗占据 octree，再 latent semantic diffusion，补全通过推理期 inpainting 实现。— [Octree diffusion](https://arxiv.org/html/2509.16483v1)
  - FlowSSC（arXiv 2601.15250）：单目 SSC，triplane latent 上做一步 latent diffusion；SemanticKITTI test mIoU 19.52 / IoU 56.97。— [FlowSSC](https://arxiv.org/pdf/2601.15250)
  - PatchScene（arXiv 2606.03915）：patch 级体素扩散，用于大场景补全。— [PatchScene](https://arxiv.org/pdf/2606.03915)
- **离散先验的后验采样**：SGDD（Split Gibbs Discrete Diffusion Posterior Sampling, NeurIPS 2025）为离散扩散先验提供 plug-and-play 后验采样，在合成基准上收敛到目标后验；早期版本声称"up to 2x"提升，NeurIPS 版改为">30%"，两版不一致。— [SGDD, arXiv 2503.01161](https://arxiv.org/html/2503.01161v2)

### Inferences
- 对我们的"去噪先验"角色，GSSC 的 S²D² 是目前最贴近的证据：判别式 argmax 作为源，只用 1 步，增益 +2 mIoU 量级，代价是每次约 100 ms 加一个约 35M 的 3D U-Net。即使在这种结构下，弱 base 上的稀有类（相当于我们的稀有家具类别）仍会被抹掉。因此若采用，应把"稀有类 / 家具 F1 不降"写成预注册门槛。
- 从噪声生成的条件模型（SGSC 30.5）比判别 base（36.2–36.7）低约 6 点，量级与我们的探针（家具 F1 低 6–22 点）一致。
- 数据规模：
  - 我们 5k–50k 个 128×128×32 场景，共 2.6e9–2.6e10 个体素标签。LT3SD 的 3D-FRONT 训练集约 5.2k 栋房屋，GSSC 约 19k 真实帧加 32k 合成，PDD 用 CarlaSC 32.4k scans，三者都在同一量级。
  - 训练成本粗估（按上述论文换算，不是实测）：LT3SD 扩散阶段约 2 卡 × 6 天 ≈ 290 GPU-h（A6000）；PDD 所述原始离散扩散在 256×256×16 上 >4 卡 × 16 天 ≈ >1,500 A100-h。我们的栅格为 524k 体素，是 256×256×16（1.05M）的一半，单卡训练一个 discrete diffusion 先验大约需要数百 GPU-h。
  - 若只训练用于熵编码的分组条件模型（交叉熵、单次或 K 次前向），训练成本应接近同尺寸判别式 SSC 网络，远低于完整扩散。

### Gaps
- 未找到在室内带家具房屋（3D-FRONT / HouseExpo 类）上做"部分观测 → 占据补全"并报告逐体素 F1/IoU 的扩散工作；LT3SD 只报告生成质量。
- 未取得 DPS（Chung et al., ICLR 2023）、DDRM 在体素或占据数据上的 NFE 与精度；DPS 每步需对网络反传的具体成本本次未核验。
- 多机器人 / 分布式场景下用生成先验融合多视角占据的工作：本次未检索到。

---

## Q4. 采样与 MAP / 边缘阈值化在检测型指标（F1、IoU）上的已知结果：perception–distortion 与"生成式采样损害逐体素精度"的证据

### Takeaway
理论上，posterior sampling 换来分布匹配（perception），代价是失真恰好翻倍：MSE = 2×MMSE（Blau & Michaeli）。对 0/1 占据，逐体素期望准确率从 max(p,1−p) 降到 p²+(1−p)²。按期望计数近似，F1 降到 Σp²/Σp。实证方面，医学分割中单个扩散样本的 Dice 低于 5 样本平均，后者又低于判别式 nnU-Net；扩散 SSC 的 mIoU 低于判别 SSC。我们 6–22 点的探针结果符合理论预期，不是实现缺陷。

### Cited Findings
- Blau & Michaeli（CVPR 2018）"The Perception-Distortion Tradeoff"：失真与感知质量（输出分布到数据分布的散度）此消彼长，存在不可达区域。— [arXiv 1711.06077](https://arxiv.org/abs/1711.06077)
- Rate–Distortion–Perception（Blau & Michaeli, ICML 2019）：构造一个从"给定解码输出的源后验"中采样的后处理，可以得到完美感知质量，失真只翻倍（"only twice the distortion"）。— [arXiv 1901.07821](https://arxiv.org/abs/1901.07821)
- Ohayon, Michaeli, Elad（PMRF, ICLR 2025）：posterior sampling 达到完美感知指标（p_X̂ = p_X），但 MSE 恰为 MMSE 的两倍。在完美感知约束下，最小 MSE 通常严格小于 2×MMSE，可通过把 posterior mean 最优传输到数据分布来得到，即 PMRF 的做法。— [PMRF, arXiv 2410.00418](https://arxiv.org/html/2410.00418v3)
- Freirich et al.（NeurIPS 2021）：在 Wasserstein 空间中刻画 D-P tradeoff，并指出 posterior sampling 往往不是最优策略。— [arXiv 2107.02555](https://ar5iv.arxiv.org/html/2107.02555)
- **实证：扩散分割的单样本与平均（Wolleb et al., MIDL 2022, BRATS2020）**：
  - Dice：单次采样 0.866，5 次采样平均再阈值化 0.881，nnU-Net（5-fold 集成）0.891。
  - HD95：6.05 / 5.18 / 5.00。
  - 速度：单个样本 48 s（T=1000），SegNet 13 ms。
  - 来源：[Wolleb et al., arXiv 2112.03145](https://ar5iv.labs.arxiv.org/html/2112.03145)
- **实证：扩散 SSC 与判别 SSC**：
  - DiffSSC mIoU 27.4，TALoS 37.9。— [DiffSSC](https://arxiv.org/pdf/2409.18092)
  - GSSC：从噪声生成 30.5，判别 base 36.2–36.7；以判别输出为源、一步修正后 38.5–38.8。— [GSSC](https://huggingface.co/papers/2608.26737)
- **F1 的最优决策**：对校准良好的概率，最大化 F1 的阈值等于最优 F1 的一半（Lipton, Elkan, Narayanaswamy, ECML PKDD 2014）。— [arXiv 1402.1892](https://arxiv.org/abs/1402.1892)

### Inferences
- **一阶推导（适用于任何校准良好的联合后验采样器，与模型族无关）**：设第 i 个体素占据的边缘概率为 p_i。
  - 由期望线性性，联合采样（哪怕从真后验采样）得到 E[TP]=Σp_i²，E[FP]=E[FN]=Σp_i(1−p_i)。
  - 因此 F1_sample ≈ 2ΣTP/(2ΣTP+FP+FN) = Σp_i²/Σp_i。
  - 阈值化在 τ 处时，TP = Σ_{p>τ}p，FP = Σ_{p>τ}(1−p)，FN = Σ_{p≤τ}p。
  - 例：一批家具体素都是 p=0.6。采样的期望 F1 = 0.60；全部判为占据（τ<0.6）的 F1 = 1.2/1.6 = 0.75，差 15 点。p 越接近 0.5，差距越大。这正好落在我们观测到的 6–22 点区间。
  - 结论：差距来自 perception–distortion 本身，不能靠更好的生成模型消除。
- 推论 1：任何"要输出地图"的环节都应使用边缘概率加按 F1 优化的阈值（Lipton 规则：τ* = F1*/2），或 PMRF 式的"先取后验均值、再做最小必要修正"；不应采样。这支持现有设计："生成先验只作熵模型，地图输出用精确码 + 阈值化"。
- 推论 2：作为熵模型时，perception–distortion 不起作用。码长只取决于 q 对真分布的交叉熵，不涉及采样，所以 Q1 的分组因子化熵模型不会受采样惩罚影响。
- 推论 3：作为去噪器时，应输出边缘概率（或 MMSE），而不是样本。GSSC 的单步确定性 Euler 更新、Wolleb 的多样本平均，本质上都是在逼近边缘或众数。

### Gaps
- 未找到直接在 3D 占据上比较"后验采样 vs 边缘阈值化"的 F1/IoU 研究（只有分割和 SSC 的旁证）。
- 未找到 rate–distortion–perception 理论在离散 / 二值占据数据上的专门结果。

---

## Q5. 小规模（单 GPU，约 5k–50k 场景，128×128×32）下的训练成本与数据需求

### Takeaway
公开的 3D 场景扩散工作在约 5k–50k 场景量级上训练，硬件为 2–4 张数据中心级 GPU，耗时数天到两周。2D 似然模型达到 SOTA 级 bpd 需要上千 TPU 芯片小时，但达到"接近 SOTA"只需要其 1–10%。对我们来说，最省的路线是用交叉熵训练一个 K 组（K≈2–8）分组条件占据模型作为熵模型。它的成本与判别 SSC 网络同级（单卡大约 1–3 天，属推测）。完整的 discrete diffusion 或 MeanFlow 先验要多一个数量级，而且对熵编码没有额外收益。

### Cited Findings
- VDM（CIFAR-10，50k 张 × 3072 维）：8×TPUv3 用 9 天达到 2.65 bpd；8 芯片 × 2.5 小时即越过 2.80。— [VDM](https://arxiv.org/abs/2107.00630)
- ARDM（CIFAR-10）：8×TPUv4 约 2 周（3000 epochs）；"a couple of days" 达约 2.8 bpd。网络为 VDM 式 U-Net，32×32 分辨率下 32+32 个 ResBlock，256 通道。— [ARDM](https://arxiv.org/abs/2110.02037)
- MeanFlow ImageNet 256：B/2（131M）训练 240 epochs 即达 1-NFE FID 6.17；CIFAR-10 用约 55M U-Net。B/4 消融为 80 epochs（400K iter，TPU v4-8）。— [MeanFlow](https://arxiv.org/html/2505.13447v1)
- PDD：4×A100，最多 800 epochs；作者称原始离散扩散在 256×256×16 上需 >16 天（4×A100）；数据为 CarlaSC 32.4k scans。— [PDD](https://arxiv.org/html/2311.12085v2)
- LT3SD：3D-FRONT 6,479 栋房屋；编解码器每层 5 h–1 天；扩散阶段 2×A6000 约 6 天。— [LT3SD](https://arxiv.org/html/2409.08215v2)
- GSSC：去噪器约 35M 参数 3D U-Net；训练数据 19,130 真实帧加 32,039 合成对；每次推理 107 ms。— [GSSC](https://huggingface.co/papers/2608.26737)
- VAR：200–350 epochs，batch 768–1024，最大一次训练达 305B tokens，GPU 总数未报告。— [VAR](https://arxiv.org/html/2404.02905v2)

### Inferences
- **维度对比**：我们每个场景 524,288 体素，是 CIFAR-10 单张图（3,072 维）的约 170 倍。5k–50k 场景合计 2.6e9–2.6e10 个标签，是 CIFAR-10 训练集（1.5e8）的 17–170 倍。但占据数据高度稀疏，而且以三值或二值为主，有效熵远低于像素；只要条件上下文足够强，较小的网络（约 10–40M，参照 GSSC 的 35M）应可拟合。这是推断，需要预注册实验验证。
- **建议的预注册比较（供报告撰写者参考，均为推断）**：在同一测试集上比较以下几种熵模型的码长（bits/scene）：
  - (a) 逐体素独立 Bernoulli（只用上下文）；
  - (b) 2 组 checkerboard；
  - (c) 8 组 2×2×2 parity 分组；
  - (d) octree 多尺度（next-scale）；
  - (e) 可选：K 步固定 schedule 的 masked diffusion。
  - 判据示例：(c) 相对 (a) 码长至少降 X%，且编码和解码在两台不同 GPU 或 CPU 上逐位一致（参照 integer network / PTQ 文献），两个 seed 都通过。
- **单 GPU 可行性**：参照 LT3SD（2 卡 × 6 天）与 PDD（4 卡、>16 天对应 2 倍体素量），单卡训练完整离散扩散先验约需 1–3 周；分组交叉熵模型只需一次前向和标准 CE 损失，训练时长应与一个 3D U-Net 分割网络相当。MeanFlow / sCM 的 JVP 开销（约 +16%）不是主要瓶颈，但它们对熵编码没有用（见 Q1）。

### Gaps
- 未找到在 128×128×32（或同等规模）占据栅格上，单 GPU 训练扩散、flow 或分组熵模型的 GPU-h 实测。
- 未找到"训练场景数 vs 码长 / 补全 F1"的 scaling 曲线（例如 5k 与 50k 场景的差距）。
- 机器人端（如 Jetson 类嵌入式 GPU）上 3D U-Net 熵模型的编解码延迟与确定性实测：本次未检索到。
