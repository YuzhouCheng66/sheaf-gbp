# 三维场景生成式先验：补全/去噪、室内布局生成与生成式建图的幻觉风险

> 截至 2026-10-27 的检索笔记。范围限于 3D，不重复早先关于 2D 平面图 map prediction 的报告。
> 标注约定：
> - 【表内】= 论文表格中的数值（作者自测，非独立复现）
> - 【声称】= 摘要或正文中的定性或汇总说法，未见对应表格，或与表格不一致
> - 【检索摘要】= 只来自搜索引擎摘要或二手综述，未读到原文对应段落，可信度较低
> - 【推断】= 笔者基于引文的推理，不是文献结论
>
> 所有数值都是原作者在自己设定下的报告，本次未找到独立复现。

---

## Q1. 用 diffusion 做 semantic scene completion（SSC）与 LiDAR scene completion：输入、输出分辨率、IoU/mIoU，以及相对判别式 SSC 的增益

### Takeaway
在 voxel 级 IoU/mIoU 指标上，diffusion SSC 相对判别式 SSC 的增益不稳定。

- 有几处正增益：
  - SSD 在同一骨干上 mIoU +5.9，但 IoU 略降。
  - DiffSSC 的几何 IoU 最高，但 mIoU 比 SCPNet/TALoS 低约 10 个点。
  - SemCity 作为"后处理 refiner"对弱模型（RGB 输入的 MonoScene）提升很大（IoU +13.3），对强 LiDAR 模型只有 +1～2.5 IoU。
- 有反例：LiDiff 在 0.2 m voxel 上的 IoU（33.1，refine 后 38.9）低于判别式的 LODE（47.9），只在 0.1 m 的细分辨率上领先。

文献中 diffusion 的主要价值在于两点：生成质量指标（FID/KID/MMD），以及作为先验来细化或补全；直接拿单个 sample 当最终地图并不占优。这和本项目"采样比阈值化低 6–22 F1"的发现方向一致。

### Cited Findings

**SSD**（Lee et al., "Diffusion Probabilistic Models for Scene-Scale 3D Categorical Data", arXiv 2301.00527, 2023）
- 数据与输入：CarlaSC 数据集（合成户外驾驶），24 个场景，10 个语义类加 free 类，每个场景 128×128×8 voxel。 — [arXiv 2301.00527](https://arxiv.org/html/2301.00527v1)
- 【表内】SSC 测试集 IoU / mIoU（Table 3）：
  - LMSCNet SS：85.98 / 42.53
  - SSCNet Full：80.69 / 41.91
  - MotionSC (T=1)：86.46 / 46.31
  - 同架构、无 diffusion 的判别式网络：80.70 / 39.94
  - Discrete Diffusion（multinomial diffusion）：80.61 / 45.83
  - 即相对同架构判别式基线 mIoU +5.89，IoU −0.09；但没有超过 MotionSC。 — [arXiv 2301.00527](https://arxiv.org/html/2301.00527v1)
- 【表内】计算开销（2×RTX 3090，T=100 步）：
  - 128×128×8 的 discrete diffusion：19m48s/epoch 训练，0.883 s/样本采样。
  - 32×32×2 的 latent diffusion：7m37s/epoch，0.499 s/样本。
  - 正文有一处把 32×32×2 写成 "32×32×32"，与表格自相矛盾。 — [arXiv 2301.00527](https://arxiv.org/html/2301.00527v1)

**LiDiff**（Nunes et al., "Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion", CVPR 2024, arXiv 2403.13470）
- 输入与输出：
  - 输入是单帧 LiDAR 点云（50 m 范围内，FPS 采样 18,000 点）。
  - 输出是补全点云；GT 为聚合并滤除动态物体后的 SemanticKITTI 地图，采样到 180,000 点。
  - 训练用 T=1000 步，推理用 DPMSolver 降到 50 步。 — [arXiv 2403.13470](https://arxiv.org/html/2403.13470)
- 【表内】SemanticKITTI seq 08 上的 Chamfer distance（m）：
  - LMSCNet 0.641，LODE 1.029，Make-It-Dense 0.503，PVD 1.256
  - LiDiff 0.434，refine 后 0.375 — [arXiv 2403.13470](https://arxiv.org/html/2403.13470)
- 【表内】不同 voxel 尺寸下的 IoU（%），依次为 0.5 m / 0.2 m / 0.1 m：
  - LMSCNet：32.23 / 23.05 / 3.48
  - LODE：43.56 / **47.88** / 6.06
  - MID：45.02 / 41.01 / 16.98
  - LiDiff：42.49 / 33.12 / 11.02
  - LiDiff refined：40.71 / 38.92 / **24.75**
  - 即粗分辨率下判别式更好，只有 0.1 m 时 diffusion 领先。 — [arXiv 2403.13470](https://arxiv.org/html/2403.13470)
- 作者自述局限：不能做无条件生成，必须有输入扫描来引导。 — [arXiv 2403.13470](https://arxiv.org/html/2403.13470)

**DiffSSC**（Cao & Behnke, IROS 2025, arXiv 2409.18092）
- 方法：
  - 输入是 Cylinder3D 分割后的语义点云（每点为坐标加语义概率向量）。
  - 在点空间和语义空间分别做 diffusion，输出稠密语义点云。
  - 评测时再体素化到 256×256×32、0.2 m 的网格。作者承认这一步体素化会引入量化误差。 — [arXiv 2409.18092v3](https://arxiv.org/html/2409.18092v3)
- 【表内】SemanticKITTI 单帧前半视野 IoU / mIoU：
  - LMSCNet 55.3 / 17.0
  - SSA-SC 58.8 / 23.5
  - JS3C-Net 56.6 / 23.8
  - LODE 51.2 / 23.4
  - SCPNet 56.1 / 36.7
  - TALoS 60.2 / 37.9
  - **DiffSSC 63.4 / 27.4**
  - 即 IoU 最高，但 mIoU 明显低于 SCPNet/TALoS。作者把差距部分归因于后两者使用了多帧信息。 — [arXiv 2409.18092v3](https://arxiv.org/html/2409.18092v3)
- 【表内】全景（panoramic）设定，IoU / mIoU：
  - SemanticKITTI：DiffSSC 60.3 / 26.7，JS3C-Net 51.3 / 21.4
  - SSCBench-KITTI360：DiffSSC 47.3 / 20.4，LODE 38.2 / 15.4 — [arXiv 2409.18092v3](https://arxiv.org/html/2409.18092v3)
- 【表内】消融：
  - 去掉 diffusion、只留 refinement：23.4 / 7.6
  - 去掉 global regularization：40.3 / 10.9
  - 线性 noise schedule：52.6 / 20.7
  - 即性能强烈依赖正则项和 schedule。 — [arXiv 2409.18092v3](https://arxiv.org/html/2409.18092v3)
- 计算与讨论：训练用 1×A6000、20 epochs、T=1000；未报告推理时间。论文没有讨论 hallucination。 — [arXiv 2409.18092v3](https://arxiv.org/html/2409.18092v3)
- 【声称】版本差异：v1 摘要称"超过所有 LiDAR 方法"，v3 改为"surpassing most existing methods"，措辞更保守。 — [检索摘要](https://arxiv.org/abs/2409.18092)

**PDD**（Liu et al., "Pyramid Diffusion for Fine 3D Large Scene Generation", ECCV 2024 Oral, arXiv 2311.12085）
- 方法：多尺度 discrete diffusion，尺度为 s1 32×32×4 → s2 64×64×8 → s4 256×256×16，每个尺度一个 3D-UNet。 — [arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)
- 评测方式：报告的是生成质量，而不是 SSC 任务。生成场景先用分割网络打标，再算 mIoU，另报 F3D 和 MMD。
- 【表内】CarlaSC 无条件生成，依次为 mIoU(V) / F3D / MMD：
  - GT：52.19 / 0.246 / 0.108
  - DiscreteDiff：40.05 / 1.361 / 0.599
  - LatentDiff：38.01 / 0.331 / 0.211
  - PDD：68.02 / 0.315 / 0.200
- 注意 PDD 的 mIoU(V) 高于 GT 本身的 52.19。这说明该指标衡量的是"生成场景对分割网络有多容易"，并不是补全准确率。 — [arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)
- 【表内】SemanticKITTI 迁移（微调 s1/s2/s4）：mIoU(V) 44.7，F3D 0.21，MMD 0.11。 — [arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)
- 【表内】记忆化检查：生成场景与最近训练场景的 SSIM 为 0.72（CarlaSC）和 0.74（SemanticKITTI），高于验证集与训练集之间的 0.65 和 0.67。即生成样本比真实新场景更接近训练集，存在一定记忆倾向；作者未这样解读。 — [arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)
- 【表内】计算：每个 PDD 模型 4×A100，800 epochs，T=100。作者称单尺度 256×256×16 的 DD 基线需要"在 4 张 A100 上超过 16 天"。 — [arXiv 2311.12085](https://arxiv.org/html/2311.12085v2)

**SemCity**（Lee et al., CVPR 2024, arXiv 2403.07773）
- 表示与分辨率：
  - triplane 表示，xy/xz/yz 三个平面，triplane 分辨率 128×128×32，C=16。
  - SemanticKITTI 为 256×256×32（20 类），CarlaSC 为 128×128×8。
  - 单张 RTX 3090，100 个 diffusion 步。 — [arXiv 2403.07773](https://arxiv.org/html/2403.07773)
- 【表内】生成质量（渲染图上的 FID）：SemanticKITTI 上 SSD 112.82，SemCity 56.55；CarlaSC 上 SSD 87.39，SemCity 40.63。 — [arXiv 2403.07773](https://arxiv.org/html/2403.07773)
- 【表内】作为 SSC refiner 的增益（SemanticKITTI val，IoU / mIoU，refine 前 → 后）：
  - MonoScene：37.12 / 11.50 → 50.44 / 17.08
  - OccDepth：41.60 / 12.84 → 50.20 / 16.79
  - SSA-SC：58.25 / 24.54 → 60.71 / 25.58
  - SCPNet：50.24 / 37.55 → 59.25 / 38.19
  - 即对弱输入增益大；对 LiDAR 方法 mIoU 只增加 0.6–1.0。 — [arXiv 2403.07773](https://arxiv.org/html/2403.07773)

**LiDM**（Ran et al., "Towards Realistic Scene Generation with LiDAR Diffusion Models", CVPR 2024）
- 【声称】在 range image 的 latent 空间加入 curve-wise compression、point-wise coordinate supervision 和 patch-wise encoding，比 point-based DM 快达 107 倍。
- 吞吐量：curve-based 1.603 samples/s，LiDARGen 0.015 samples/s。patch-based latent 方法更快（2.171 samples/s），所以 107 倍只是相对 point-based 方法而言。 — [检索摘要/CVPR 页面](https://openaccess.thecvf.com/content/CVPR2024/html/Ran_Towards_Realistic_Scene_Generation_with_LiDAR_Diffusion_Models_CVPR_2024_paper.html)

**后继工作**
- **LiDPM**（Martyniuk et al., IV 2025, arXiv 2504.17791）
  - 【声称】LiDiff 的 "local diffusion" 近似并不必要；只要选好起点，vanilla DDPM 就足以完成场景级补全，并且在 SemanticKITTI 上更好。
  - 还可以通过关闭条件做无条件生成。 — [arXiv 2504.17791](https://arxiv.org/abs/2504.17791)
- **Octree Latent Diffusion**（Zhang, Crowe, Heckman, arXiv 2509.16483, 2025）
  - 方法：先用 structure diffusion 生成粗占据八叉树，再用 latent semantic diffusion 生成语义。补全靠推理时 inpainting 实现，无需重训。
  - 数据设置：Replica 室内 0.05 m voxel、**128×128×32** 网格、92 类；SemanticKITTI 0.2 m、256×256×32。
  - 【表内】在给定 GT 粗占据的条件下，语义 mIoU 为 SemanticKITTI 16.9%、Replica 32.22%。没有报告补全 IoU。
  - 【声称】单扫描补全的质量"与用完整 oracle 结构作条件相当"，但没有给出数字。
  - 自述局限：mask dilation，即 latent mask 让条件输入与输出无法完全对齐。 — [arXiv 2509.16483](https://arxiv.org/html/2509.16483v1)

### Inferences
- 【推断】文献在三个独立设定里都出现"diffusion 赢分布、输逐体素"的模式：
  - LiDiff 在 0.2 m 下 IoU 低于 LODE；
  - DiffSSC 的 mIoU 低于判别式 SOTA；
  - SceneSense 的例子 IoU 0.52 < 0.61（见 Q4）。
  
  第一性原理解释：IoU/F1 这类逐体素指标的最优决策是对后验边际取阈值；从后验采样会引入额外方差，期望逐体素得分因此必然不高于边际阈值化。这与本项目"采样比阈值化低 6–22 F1"一致，说明那不是实现缺陷，而是指标与决策规则不匹配。
- 【推断】SemCity 的数据显示：生成先验在作为 refiner、且输入很弱时最有价值。对本项目，这意味着在 coverage 低、或单机观测稀疏的区域，生成先验可能有正收益；在已充分观测的区域收益趋近于零。
- 【推断】PDD 的 SSIM 记忆化信号（0.72 > 0.65）提示：在训练分布很窄的数据上（例如本项目的参数化盒子家具库），生成模型容易"背家具模板"。补全结果会看起来合理，但可能是模板复制。

### Gaps
- 没有找到在 NYUv2 或 ScanNet 等室内 SSC 基准上、与判别式方法同表比较的 diffusion SSC 结果。Octree LD 在 Replica 上只报告了生成指标和给定 GT 结构时的 mIoU。
- LiDPM、LiDM 的具体表格数值本次没有读到全文。
- 没有找到任何 diffusion SSC 论文报告 voxel 级校准（calibration/ECE）或 sample 方差与误差相关性的量化结果。

---

## Q2. 大场景稀疏体素 / latent 3D diffusion（XCube, LT3SD, BlockFusion, NFD, triplane diffusion）：分辨率与计算量

### Takeaway
大场景 3D 生成已经能做到有效分辨率 1024³（XCube），或 2.2 cm voxel、90 m×45 m 的室内多房间场景（LT3SD）。代价是：

- 训练需要多卡 A100/A6000，持续数天至数周；
- 推理需要秒级到小时级；
- 补全能力在这些论文里基本只有定性展示，没有补全 IoU。

对 128×128×32 这种量级的网格，SSD/SemCity 这类单卡方法已经足够；XCube/LT3SD 属于"超配"。

### Cited Findings
**XCube**（Ren et al., CVPR 2024 Highlight, arXiv 2312.03806）
- 方法与分辨率：
  - 基于 VDB 稀疏体素的分层 latent diffusion，有效分辨率最高 1024³。
  - Waymo 和 Karton City 的层级为 32³ → 256³ → 1024³。
  - 演示了 100 m×100 m、10 cm voxel 的户外场景。 — [arXiv 2312.03806](https://arxiv.org/html/2312.03806v2)
- 【表内/声称】计算：
  - ShapeNet 用 8×V100，其余数据集用 8×A100，但没有给出训练天数。
  - 生成物体或驾驶场景"不到 30 秒"。
  - 稀疏框架下 1024³ 网格内存 8.4 MB（TorchSparse 为 104.6 MB）；3.4M voxels 占 11 MB。 — [arXiv 2312.03806](https://arxiv.org/html/2312.03806v2)
- Waymo 单扫描 LiDAR 补全只有定性图（Fig. 11），没有 CD 或 IoU。 — [arXiv 2312.03806](https://arxiv.org/html/2312.03806v2)

**LT3SD**（Meng, Li, Nießner, Dai, CVPR 2025, arXiv 2409.08215）
- 表示与数据：
  - 3 层 latent tree（TUDF 加 4 通道 latent），UDF voxel 为 2.2 cm，截断 10 cm。
  - 训练数据为 3D-FRONT 的 6,479 个 house。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- 生成规模与推理时间：
  - 无限场景输出分辨率 [4096, 2048, 128]，对应 90.3 m×45.1 m×2.8 m，约 170 个房间。
  - 单 GPU 推理约 2 小时；对比 BlockFusion 生成 7 个房间需要 3 小时。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- 【表内】3D-FRONT 上的对比，依次为 COV↑ / MMD-CD↓ / 1-NNA-CD↓ / FID↓：
  - PVD：43.82 / 3.69 / 70.83 / 237.85
  - NFD：44.65 / 3.65 / 62.86 / 266.27
  - BlockFusion：24.32 / 5.10 / 89.01 / 45.55
  - XCube：48.60 / **3.35** / 56.45 / 55.35
  - LT3SD：**53.10** / 3.51 / **53.22** / **13.39**
  - 作者称"所有指标显著超越"，但 XCube 的 MMD 其实更好，属于夸大。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- 【表内】压缩相关：latent tree 每个房间样本约 33 KB；latent cascaded 基线为 41 KB。重建 ℓ2 误差分别为 3.20e-4 和 4.91e-4。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- 【声称，数值存疑】diffusion 训练用 RTX A6000，抽取文本写的是"约 66 天"收敛，GPU 数也是乱码。需要查 PDF 核实，不建议引用。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)
- 补全与记忆化：
  - 场景补全只有定性展示（Fig. 6）：保持观测几何不变，采样出多样补全。
  - 记忆化检查用 Chamfer 最近邻检索，只有定性结论。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)

**BlockFusion**（Wu et al., SIGGRAPH 2024, arXiv 2401.17053）
- 方法：
  - 把场景切成随机裁剪的 block，每个 block 拟合为 triplane 加 SDF MLP。
  - 用 VAE 压缩到 latent triplane，在上面做 diffusion。
  - 扩展场景时对重叠区做条件化 extrapolation；可用 2D layout 作条件。 — [arXiv 2401.17053](https://arxiv.org/html/2401.17053v4)
- 预训练 checkpoint 约 10 GB。训练硬件本次没有核实到。 — [检索摘要](https://www.gitlink.org.cn/mirrors/BlockFusion)

**NFD / triplane diffusion**
- 本次只在 LT3SD 的对比表中拿到 NFD 在 3D-FRONT 上的数值：FID 266.27，COV 44.65。 — [arXiv 2409.08215](https://arxiv.org/html/2409.08215v2)

### Inferences
- 【推断】对本项目 128×128×32 的整屋网格（约 52 万 voxel）而言：
  - SSD 的 latent 版本（压到 32×32×2 后采样 0.5 s/样本）或 SemCity 的 triplane（128×128×32，单张 3090）在规模上直接适用；
  - XCube/LT3SD 的分层稀疏设计主要解决百万到十亿级 voxel 的问题，在本基准上收益不大。
- 【推断】压缩视角：LT3SD 每房间约 33 KB 的 latent、XCube 11 MB/3.4M voxel，都远大于"对象 token"表示（见 Q3）。如果目标是多机通信压缩，latent 网格不是最省的码。

### Gaps
- XCube、LT3SD、BlockFusion 都没有给出补全任务的 IoU/F1，所以无法判断它们作为 completion prior 在逐体素指标上的表现。
- BlockFusion、NFD 的原始训练 GPU-hours 没有核实到；LT3SD 的训练天数提取文本有乱码。

---

## Q3. 室内布局与家具生成（ATISS, DiffuScene, LEGO-Net, HouseDiffusion/House-GAN, ProcTHOR, Infinigen-Indoors, 3D-FRONT）；是否有对象级（box/token）紧凑场景表示

### Takeaway
室内家具布局生成的主流表示正是"对象集合"：每个对象用类别、位置、尺寸、朝向（和可选的形状码）表示。

- ATISS 只用带标签的 3D bounding box 监督；
- DiffuScene 的对象向量为 [location, size, cos θ, sin θ, class, shape code]，卧室最多 13 个对象；
- LEGO-Net 在同一表示上做"去噪式重排"。

这类表示本身就是极紧凑的场景码，而且与本项目"盒子加圆柱参数化家具库"天然同构。

### Cited Findings
**3D-FRONT**
- 原始数据为 6,813 个 house、14,629 个房间。
- 按 ATISS/DiffuScene 的预处理过滤后：4,041 个卧室、900 个餐厅、813 个客厅。
- 卧室每屋 3–13 个对象，餐厅和客厅为 3–21 个。 — [DiffuScene arXiv 2303.14207](https://arxiv.org/html/2303.14207)

**DiffuScene**（Tang et al., arXiv 2303.14207）
- 表示：
  - 场景为 N 个对象的无序集合，用 "empty" 对象补齐到固定数量。
  - 每个对象为 o_i = [location(3), size(3), cos θ, sin θ, class(C), shape code]。
  - 卧室 C=22，N=13；形状码由预训练 VAE 给出，维度 32。 — [arXiv 2303.14207](https://arxiv.org/html/2303.14207)
- 【表内】卧室无条件生成，依次为 FID / KID×1e-3 / SCA% / CKL×0.01：
  - DepthGAN：40.15 / 18.54 / 96.04 / 5.04
  - Sync2Gen：31.07 / 11.21 / 82.97 / 2.24
  - ATISS：18.60 / 1.72 / 61.71 / 0.78
  - DiffuScene：17.21 / 0.70 / 52.15 / 0.35 — [arXiv 2303.14207](https://arxiv.org/html/2303.14207)
- 【表内】物体穿插（pairwise box IoU，PIoU）：GT 0.43，ATISS 0.50，DiffuScene 0.43。去掉 IoU loss 后性能变差；作者把 DepthGAN 和 Sync2Gen 描述为易产生物体相交。 — [arXiv 2303.14207](https://arxiv.org/html/2303.14207)
- 【表内】重排任务（rearrangement，依次为 FID / PIoU）：ATISS 27.14 / 0.84，LEGO-Net 23.73 / 0.89，DiffuScene 22.16 / 0.61。场景补全任务只有定性对比。 — [arXiv 2303.14207](https://arxiv.org/html/2303.14207)

**ATISS**（Paschalidou et al., NeurIPS 2021）
- 自回归 transformer，把场景视为无序对象集合，只用带标签的 3D bounding box 端到端训练。在 3D-FRONT 的 4 类房间上评测。 — [NVIDIA 项目页](https://research.nvidia.com/labs/toronto-ai/publication/neurips_2021_atiss)

**LEGO-Net**（Wei et al., CVPR 2023）
- 在 3D-FRONT 干净场景上加扰动，训练 transformer 迭代地把物体从凌乱布局移回规整布局（"受 diffusion 启发"），同时限制移动距离。预处理沿用 ATISS 的协议。 — [CVPR 2023 论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_LEGO-Net_Learning_Regular_Rearrangements_of_Objects_in_Rooms_CVPR_2023_paper.pdf)

**HouseDiffusion / House-GAN++**（户型平面图生成）
- 【表内，跨论文】在 GSDiff 的同表比较中，RPLAN 拓扑约束生成的 FID 为 House-GAN++ 48.40、HouseDiffusion 11.87；参数量分别为 2M 和 27M。
- 不同论文给出的 FID 不一致，例如另一篇报告 GSDiff 为 4.83，而 GSDiff 自报约 6.6–6.8。
- 【声称】HouseDiffusion 可以生成非 Manhattan 布局。 — [GSDiff arXiv 2408.16258](https://arxiv.org/pdf/2408.16258)；[HouseDiffusion](https://ar5iv.labs.arxiv.org/html/2211.13287)
- 一篇 2026 年的评测论文警告：长时间训练会导致 FID 无法诊断的多样性坍缩。 — [arXiv 2602.01949](https://arxiv.org/pdf/2602.01949v1)

**ProcTHOR**（Deitke et al., NeurIPS 2022）
- 用 10,000 个程序生成的房屋训练简单模型（ProcTHOR-10K）。
- 【声称】在 6 个具身 AI 基准上达到 SOTA，零样本结果常常优于使用下游训练数据的方法。 — [arXiv 2206.06994](https://arxiv.org/abs/2206.06994)

**Infinigen Indoors**（Raistrick et al., CVPR 2024）
- 100% 程序化生成，不依赖外部资产，开源 BSD 许可。
- 用约束 DSL 加求解器安排物体，约束针对 "furniture""storage" 这类抽象类别。
- 其 Table 1 把 ProcTHOR 列为"程序规则、1.6K 场景"，与 ProcTHOR-10K 的说法不一致，可能是统计口径不同。 — [arXiv 2406.11824](https://arxiv.org/html/2406.11824v1)

### Inferences
- 【推断，数量级估算】按 DiffuScene 的表示，一个卧室约 13 对象 ×（3+3+2+22+32 ≈ 62 维）≈ 800 个浮点数；若去掉 32 维形状码而改用离散的家具库索引，约 13×（8 + 类别 ID）。
  - 相比之下，128×128×32 二值网格为 524,288 bit（约 64 KB）。
  - 对于本项目"参数化盒子/圆柱"家具，对象 token 几乎是无损的，可以作为远比 voxel latent 紧凑的跨机通信码与先验。
- 【推断】DiffuScene 的 PIoU 指标与 LEGO-Net 的"去噪到规整布局"思路，可以直接迁移为本项目的家具先验：
  - 用 box 穿插和对齐等物理约束作 energy；
  - 再配合本项目"盒子轴对齐、过于简单"的评估，引入旋转（cos θ, sin θ）与非轴对齐家具，能自然增加难度。
- 【推断】HouseExpo 平面图对应 HouseDiffusion 这一类 vector floorplan 生成；家具布局对应 ATISS/DiffuScene 这一类。两层可以分别建模（户型 → 每房间家具集合）。

### Gaps
- 没有核实到 ATISS 的原始表格数值、参数量和训练时间。也没有核实 3D-FRONT 原论文的家具数量（常见说法是约 13k 件家具，但本次未验证）。
- 没有找到"把对象 token 用作多机地图压缩码"的直接文献；上述只是推断。
- 没有找到 DiffuScene/ATISS 在 completion 任务上的定量结果（只有定性图）。

---

## Q4. 机器人探索/建图中的生成式占据预测（SceneSense 等）：如何报告不确定性、已记录的幻觉失败与安全问题、如何区分"观测"与"预测"空间

### Takeaway
3D 生成式占据预测在机器人上的代表是 SceneSense 系列（CU Boulder，Heckman 组）。它们分隔观测与预测的标准做法是 occupancy inpainting：每一步去噪都把已观测的 free/occupied voxel 重新注入，因此生成结果永不覆盖观测。

但已有文献记录了关键失败：把预测写回规划用的地图后，free 与 unknown 的边界被抹掉，frontier 规划器"过早认为已探索完毕"，导致无法达到 95% 覆盖。

不确定性方面，文献只用多次采样的分布作定性代理；校准的不确定性需要 conformal prediction（PwC）一类方法，才能给出碰撞概率保证。

### Cited Findings
**SceneSense**（Reed et al., arXiv 2403.11985, 2024）
- 输入与数据：
  - 输入为运行中的 Octomap 加单个 RGB-D（256×256）；条件特征由 PointNet++ 经 cross-attention 注入。
  - voxel 为 0.1 m；训练数据为 Habitat 中 HM3D 的第 3–12 号房屋（约 9,000 个位姿），测试为第 1–2 号。 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)
- Occupancy inpainting 的做法：
  - 由 Octomap 的占据概率构造 occupied 与 unoccupied 两个 mask，只允许未观测 voxel 改变。
  - 每步加入当前步噪声后重新施加已知 voxel（SDEdit/RePaint 风格）。 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)
- 【表内】FID / KID：
  - House 1：基线（running Octomap）26.18 / 18.91，SceneSense 17.81 / 7.93
  - House 2：22.55 / 14.06 → 20.94 / 6.93 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)
- 【表内/正文】IoU 反例：在一个厨房例子中，SceneSense 的 IoU 为 0.52，而局部占据数据为 0.61。
  - 预测的墙与真实墙偏差 0.1–0.2 m。
  - 作者明确认为 IoU "重罚猜测的几何"，因此改用 FID/KID。 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)
- 推理设置与失效原因：
  - 推理 30 步时效果最好；步数更多会出现 sampling drift，导致 KID 变差。
  - guidance 过强时泛化变差。
  - 在走廊和拐角多的 House 2 上误差更大，作者归因于它与训练分布不同。 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)
- 速度与训练：RTX 4090 上每步约 0.0633 s；训练用 1×A100、250 epochs。 — [arXiv 2403.11985](https://arxiv.org/html/2403.11985v1)

**Achey et al.**（"Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis", arXiv 2506.20049, 2025；SceneSense 的后续）
- 融合规则：
  - 采用 OctoMap 式 log-odds 更新，传感器项 P(m|z) 与扩散项 P(m|d) 分开。
  - 作者明确设定 P(m|d) < P(m|z)，"因为我们更信任传感器"，但没有给出具体数值。
  - 每个位姿融合 3–5 个 SceneSense 样本。 — [arXiv 2506.20049](https://arxiv.org/html/2506.20049v2)
- 【表内】FID，依次为 Env 1 / Env 2：
  - 机器人中心预测：基线 36.0 / 30.3，SceneSense（PMM）29.2 / 21.0
  - frontier 中心预测：基线 116.9 / 150.6，SceneSense（PMM）30.1 / 34.5
  - 摘要汇总为 FID 改善 24.44%（近处）和 75.59%（远处）。
- 目标区域中未知 voxel 占 59–71%。 — [arXiv 2506.20049](https://arxiv.org/html/2506.20049v2)
- 【表内】下游规划效果：
  - 走廊穿越平均用时：GBPlanner 25.98 s（1 次失败），加 Frontier SS 后 21.10 s（0 次失败）。
  - 全探索 Env 2：GB 106.46 s，GB + Local SS 81.68 s。 — [arXiv 2506.20049](https://arxiv.org/html/2506.20049v2)
- **已记录的幻觉失败（关键）**：
  - Frontier SS 把未观测 voxel 写成占据概率，"erases the boundary between free and unknown"，规划器"prematurely concludes that the environment is fully explored"。
  - 因此该配置未达到 95% 覆盖，表中记为 N.A.
  - 作者局限部分写道："a single, highly confident but incorrect prediction can still adversely affect the map's quality"。
  - 透明和反光表面上可能产生不可预测行为（未量化）。
  - 没有报告 precision/recall 或 false positive 率，也没有校准的不确定性；样本离散度只以 IoU 分布图定性展示。 — [arXiv 2506.20049](https://arxiv.org/html/2506.20049v2)
- 硬件：Boston Dynamics Spot + Ouster OS1-64；LIO-SAM + OctoMap + GBPlanner；每次预测约 0.5 s（RTX 4070 Ti Super，30 步）。 — [arXiv 2506.20049](https://arxiv.org/html/2506.20049v2)

**Map Prediction and Generative Entropy for Multi-Agent Exploration**（arXiv 2501.13189, 2025）
- 用微调的 latent diffusion inpainting 模型，在多智能体 2D 占据图上补全未知区域。
- 由"迭代推断场景的多种解释"构造 generative entropy，用于给任务分配加权；实验为 3 辆车、模拟城市环境。 — [arXiv 2501.13189](https://arxiv.org/abs/2501.13189)
- 【检索摘要】作者称预测"只用于任务加权、不直接用于导航，因此幻觉或误预测不影响机器人安全"。这是一种以架构隔离预测与安全的做法。 — [ResearchGate 摘要](https://www.researchgate.net/publication/388354104_Map_Prediction_and_Generative_Entropy_for_Multi-Agent_Exploration)

**Perceive with Confidence (PwC)**（arXiv 2403.08185；后发表于 IJRR/SAGE）
- 方法：
  - 对 scene completion 模型 NU-MCC（输出 UDF）做 conformal 校准。
  - 每个校准环境的非一致性分数，取使阈值化后的预测占据完全覆盖 GT 占据的最小阈值 q。
  - 再用非确定性滤波：占据估计随时间取交，free 估计取并。
- 保证：预测的 free 空间以高概率不与真实占据相交；配合安全规划器，端到端碰撞概率 ≤ ε。 — [arXiv 2403.08185](https://arxiv.org/html/2403.08185v3)
- 【表内/声称】结果：
  - occupancy 版 PwC 的成功率比 bounding-box 版高 40%（摘要写 46%，前后不一致）。
  - bounding-box 版在仿真中漏检降低 70%，在硬件（Unitree Go1）上降低 93.3%。
  - 场景为 3D-FRONT 椅子放入 8×8 m 房间。 — [arXiv 2403.08185](https://arxiv.org/html/2403.08185v3)

**其他不确定性相关工作**
- α-OCC：对相机 3D 语义占据做深度不确定性传播加 hierarchical conformal prediction。【声称】在保持覆盖保证的同时，预测集比基线小最多 92%（TMLR 2026）。 — [arXiv 2406.11021](https://arxiv.org/html/2406.11021v4)
- IJCAI 2025 的可靠性评测发现：相机语义占据模型相对 LiDAR 方法存在持续的可靠性和校准差距。 — [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0220.pdf)
- Map Space Belief Prediction（arXiv 2502.20606）把 "Hallucinated" 定义为一种结果类别（物体不存在却被报告为存在），并报告 3 种方法都存在持续的幻觉，作者归因于语义分割不可靠和相机噪声。 — [arXiv 2502.20606](https://arxiv.org/pdf/2502.20606)
- 2D 先驱（作背景）：Katyal et al., ICRA 2019，"Uncertainty-aware occupancy map prediction using generative networks for robot navigation"。 — [检索摘要（被引于 PIPE 等）](https://arxiv.org/pdf/2503.07504)

### Inferences
- 【推断】综合 SceneSense 和 Achey et al.，"观测 vs 预测"需要三层隔离：
  1. 生成时，用 inpainting 硬约束不覆盖观测；
  2. 存储时，让预测通道与 log-odds 观测通道分开，或者设更低的权重；
  3. 使用时，frontier/coverage 判定只读观测通道。
  
  Achey et al. 的失败正是违反第 3 层所致。本项目若用生成模型补全未见空间，评估时应分别报告"观测区 F1"和"预测区 F1"，coverage 指标不得计入预测体素。
- 【推断】文献中的"不确定性"大多是样本离散度，没有校准。PwC 的 conformal 方法是唯一给出有限样本保证的路线，它的代价是保守（扩张占据）。对多机地图融合，可以把 conformal 阈值视为"预测通道的最大信任半径"。
- 【推断】本项目把生成模型定位为 entropy model、denoiser 或 completion prior，而不是直接输出地图。这与 2501.13189（预测只用于任务分配）、Achey et al.（P(m|d) < P(m|z)）、SceneSense（只改未观测体素）的设计哲学一致。

### Gaps
- 没有找到 3D 生成式建图中 false-positive 或 false-negative 障碍率的系统量化，也没有校准曲线（ECE）。SceneSense 系列只报 FID/KID 和 IoU 分布。
- 没有找到"因生成式地图幻觉导致碰撞"的真实事故报告；PwC 只在受控仿真和硬件上统计碰撞率。
- Occupancy Anticipation（Ramakrishnan et al., ECCV 2020）是 2D 俯视图占据预测，赢得 2020 Habitat PointNav Challenge。本次只读到摘要，未拿到其 map accuracy 或 IoU 数值，也没有拿到其对预测区域的处理方式。 — [arXiv 2008.09285](https://arxiv.org/abs/2008.09285)

---

## Q5. 用于多机建图的真实感室内基准（ScanNet, Replica, HM3D, Matterport3D）：噪声模型（深度噪声、位姿漂移）与 GT 定义

### Takeaway
现有多机室内基准分为两类，各有短板：

- 合成或重渲染基准（Replica 系）提供稠密 GT mesh/深度，但智能体数少（2–4），且默认没有传感器噪声；
- 真实数据（AriaMultiagent、ETH3D、ScanNet）有真实噪声，但缺少稠密 3D GT，多只有轨迹 GT。

Habitat 生态里的标准噪声模型是 Redwood 深度噪声和 LoCoBot 实测的执行噪声，可作为本项目"无噪声传感器"问题的直接补丁。

### Cited Findings
**HM3D**（Ramakrishnan et al., NeurIPS 2021 Datasets）
- 规模：1,000 个建筑级 3D 重建，可导航面积 112.5k m²，是 MP3D 和 Gibson 的 1.4–3.7 倍。
- 【声称】质量：相对 Replica、MP3D、Gibson、ScanNet，渲染图像的视觉保真度高 20–85%，因重建不完整导致的 mesh 缺陷少 34–91%。 — [arXiv 2109.08238](https://arxiv.org/abs/2109.08238)
- 其含义：GT 是带缺陷的重建 mesh，而不是 CAD 真值。

**多智能体基准**
- ReplicaMultiagent：每序列 2 个智能体，评测 ATE RMSE（cm）与 PSNR/SSIM/LPIPS。
- AriaMultiagent：真实数据，每房间 3 个智能体。
- ReplicaMultiagent Plus（MAGS-SLAM 提出）：在 Habitat-Sim 中重渲染 Replica，最多 4 个智能体、长轨迹，提供 GT 几何和语义标注。
- 本次读到的正文没有说明是否模拟传感器噪声。 — [MAGS-SLAM arXiv 2605.10760](https://arxiv.org/html/2605.10760v1)
- 【检索摘要】真实数据集（AriaMultiagent、ETH3D 多机划分、MNE/MCN-SLAM 数据集）有真实传感器噪声，但没有稠密 mesh GT。 — [MAGS-SLAM](https://arxiv.org/pdf/2605.10760)
- 【检索摘要】MCN-SLAM 称 ScanNet "只提供轨迹 GT，没有 3D GT"。这与 ScanNet 实际发布了 BundleFusion 重建 mesh 的事实存在张力，更准确的表述应是"重建 mesh 不完整、不宜作 GT"。 — [MCN-SLAM arXiv 2506.18678](https://arxiv.org/html/2506.18678v2)
- 漂移评测：MNE-SLAM 对齐初始位姿后，对每个位姿算 ATE，并报告 RMSE、Mean、Median。 — [MNE-SLAM 补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Deng_MNE-SLAM_Multi-Agent_Neural_CVPR_2025_supplemental.pdf)
- 真实多机数据集 S3E：13 条户外加 5 条室内序列（UGV）。 — [arXiv 2210.13723](https://arxiv.org/pdf/2210.13723)

**噪声模型**
- 深度噪声：Redwood 深度噪声模型是 Habitat 中模拟深度的标准选择。也有工作对 RGB 加 Gaussian/Speckle/Poisson 噪声。 — [检索摘要，例如 arXiv 2010.13439](https://arxiv.org/pdf/2010.13439)
- 执行噪声：来自 LoCoBot 的实测平移和旋转噪声（动捕测得）。Habitat-sim 的 `PyRobotNoisyActuationSpec` 默认 robot='LoCoBot'、controller='ILQR'，并带 noise multiplier 用于消融。 — [Habitat-sim 文档](https://aihabitat.org/docs/habitat-sim/habitat_sim.agent.PyRobotNoisyActuationSpec.html)；[Active Neural SLAM arXiv 2004.05155](https://arxiv.org/pdf/2004.05155)
- 真实 LoCoBot 的 RealSense 深度存在伪影和缺失区域，仿真中的噪声模型无法完全覆盖。 — [检索摘要](https://arxiv.org/pdf/2010.13439)

### Inferences
- 【推断】本项目可以低成本升级为"有噪声版"：
  - 深度用 Redwood 模型（或按距离平方增长的高斯噪声加随机缺失）；
  - 里程计用 LoCoBot 的执行噪声乘以 noise multiplier 做扫描；
  - GT 仍由 HouseExpo 挤出与参数化家具精确生成。

  这样保留了本基准相对 HM3D/ScanNet 的最大优势，即精确、完整的 GT，同时去掉"无噪声"这一不现实之处。
- 【推断】若要外部效度，可加一个 HM3D 子集（1,000 场景）作为迁移测试。但其 GT 是重建 mesh，本身带孔洞和缺陷（作者称比其他数据集少 34–91%），评测时需要把 GT 缺失区排除在外。
- 【推断】难度升级方向：
  - 非轴对齐家具（加入 yaw）；
  - 更多智能体（ReplicaMultiagent Plus 已到 4 个）；
  - 加入位姿漂移，使多机融合必须做 loop closure 或对齐。

  这些与 Q3 的对象 token 表示可以共用。

### Gaps
- 没有找到一个同时具备稠密 GT、可控深度噪声与位姿漂移、并且有 4 个以上智能体的公开室内多机建图基准；ReplicaMultiagent Plus 是否含噪声未能确认。
- Matterport3D 的 GT 定义和缺陷率，本次只在 HM3D 摘要的对比中间接出现，没有读原文。
- 2020 Habitat PointNav Challenge 的具体噪声设定（是否移除 GPS+Compass、深度噪声强度）没有核实到原文。
