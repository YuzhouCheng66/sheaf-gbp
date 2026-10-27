# 三维几何压缩（点云 / 体素占据 / LiDAR 与深度）现状：给“hit set + poses”基线定位

> 调研时间：2026-10-27。数字只摘自能读到原文或原表的来源；只来自检索摘要、未核对全文的条目会标“未核实”。
> 记号：bpp = bits per (input) point。对体素化点云（每个点就是一个占据体素），bpp 等于 bits per occupied voxel。“vox10 / 10-bit” 指每轴 2^10 格。LiDAR 的 “1 mm / 18-bit”、“2 cm / 14-bit” 指量化精度。BD-rate 为负表示省码率。
> 我方场景：128×128×32 = 524,288 个体素，每个机器人观测 10–15%（约 5.2–7.9 万个体素），格子 6–16 cm。不学习时每个机器人的地图消息约 200–260 B，即 1,600–2,080 bit。

---

## Q1. MPEG G-PCC（octree / trisoup / predictive）、V-PCC 与 Google Draco：稠密点云和稀疏 LiDAR 上的典型 bpp

### Takeaway
G-PCC octree 无损：稠密 10-bit 人体点云约 0.9–1.1 bpp（TMC13 v14）。新版 TMC13 v23 在部分同名序列上的报告值低到约 0.65–0.82 bpp，但两组数据来自不同论文，未必可比。稀疏 LiDAR 上，2 cm 精度约 7.6–10 bpp，1 mm 精度约 20–22 bpp。Draco 在 LiDAR 上明显弱于 G-PCC 和学习型编解码器：同等 D1 下 RENO 比它省约 45–48% 码率。V-PCC 适合稠密、实心的内容，不适合稀疏内容；trisoup 只有有损模式。

### Cited Findings
- **G-PCC octree 无损，稠密点云（TMC13 v14，MPEG CTC，`positionQuantizationScale=1`）**：longdress vox10 1.015、loot vox10 0.970、red&black vox10 1.100、soldier vox10 1.030、queen 0.773、basketball player vox11 0.898、dancer vox11 0.880，平均 0.952 bpp。每帧编/解码 5–24 s / 3–15 s。— [SparsePCGC, arXiv 2111.10633, Table 4](https://arxiv.org/abs/2111.10633)
- **G-PCC octree 无损，稀疏 LiDAR（TMC13 v14，angular mode 关闭）**：KITTI 2 cm 为 7.637 bpp，Ford 2 cm 为 10.032，KITTI 1 mm 为 20.154，Ford 1 mm 为 22.298。编/解码约 1–1.8 s / 0.6–1.0 s。KITTI 每帧约 12 万点；1 mm 量化需要 18 bit，2 cm 需要 14 bit。— [SparsePCGC, Table 4 与 §5](https://arxiv.org/abs/2111.10633)
- 第三方重跑的 G-PCC（TMC13 v14）LiDAR 无损数字为 KITTI_vox2cm 7.62、Ford_vox2cm 9.95、KITTI_vox1mm 20.17、Ford_vox1mm 22.31 bpp，平均 15.01，与上一条基本一致。— [Xue et al., Efficient LiDAR PCGC through Neighborhood Point Attention, arXiv 2208.12573, Table 1](https://arxiv.org/abs/2208.12573)
- **G-PCC 在 MPEG 8i 上的无损值（另一组帧，TMC13 稳定版）**：loot 0.95、redandblack 1.09、boxer 0.94、thaidancer 0.99，平均 0.99 bpp。MVUB（phil / ricardo，9/10-bit）平均 1.10 bpp。— [ECM-OPCC, arXiv 2211.10916, Table 3/4](https://arxiv.org/abs/2211.10916)
- **G-PCC v23**：在 8iVFB（Red&black ×300 帧、Loot ×300 帧、Thaidancer ×1、Boxer ×1）上的无损值为 0.82 / 0.69 / 0.70 / 0.65，平均 0.72 bpp，比上面 v14 时代的数字低约 25–30%。— [UniPCGC, arXiv 2503.18541, Table 1](https://arxiv.org/abs/2503.18541)。这与 ECM-OPCC 和 SparsePCGC 报告的约 0.95–1.0 bpp 不一致。原因可能是 TMC13 版本（v14 对 v23）、配置或帧选择不同，我没有核实。
- **G-PCC v23 在 LiDAR 上的速度**：KITTI 12-bit 编/解码 0.564 / 0.174 s，14-bit 为 0.973 / 0.343 s。— [RENO, arXiv 2503.12382, Table 1](https://arxiv.org/abs/2503.12382)
- **G-PCC predictive geometry + angular mode**：面向 Category 3（LiDAR 采集）、低延迟场景。它在 (r, φ, laser index) 域里做预测；这个域里的编码本身不是无损的，所以还要再编一组 Cartesian 残差。— [USPTO 专利文本（转述 G-PCC 设计），检索摘要，未核实全文](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12430809)
- **Draco（LiDAR，有损）**：在 KITTI 上，同等 D1 时 RENO 比 Draco 省 48.34%（D2 省 48.30%）；在 Ford 上省 45.01% / 44.97%。Draco 很快：KITTI 12-bit 编/解码 0.065 / 0.028 s。— [RENO, Table 1](https://arxiv.org/abs/2503.12382)
- **Draco 与 OctSqueeze 对照**：KITTI 上，Draco 2.17 bpp 时 PSNR 51.52，OctSqueeze 2.06 bpp 时 PSNR 58.54。NorthAmerica 上，Draco 12.53 bpp / PSNR 79.38，OctSqueeze 11.36 bpp / 80.06。OctSqueeze 声称在同等重建质量下比 Draco 和 MPEG anchor 省 10–20% 码率。— [OctSqueeze, CVPR 2020, arXiv 2005.07178, Fig. 4 / §4.4](https://arxiv.org/abs/2005.07178)
- **G-PCC 与 Draco 在 Waymo 上（有损）**：Chamfer distance 约 0.005 时，G-PCC 需要 10.78 bpp，RIDDLE 只要 3.65 bpp。— [RIDDLE, CVPR 2022, arXiv 2206.01738, §5](https://arxiv.org/abs/2206.01738)
- **V-PCC**：一篇 2026 年论文在“无损几何”设定下，以 G-PCC octree 为 anchor，报告 V-PCC 的平均 BD-rate 为 +146.4%（如 Loot +243.4%），原因是 raw / EOM patch 里的高频冗余去不掉。— [arXiv 2607.03576（检索摘要，未核实全文）](https://arxiv.org/pdf/2607.03576)
- **trisoup**：只编三角面信息，不编点坐标，本质是有损的，不能作为无损候选。— [检索摘要中引用的 octree / trisoup 描述，arXiv 2209.02226 等](https://arxiv.org/pdf/2209.02226)

### Inferences
- 精度每多 1 bit，G-PCC 的无损码率几乎线性增加约 3 bpp：KITTI 从 2 cm 到 1 mm 多 4 bit，码率从 7.64 涨到 20.15，即约 3.1 bpp/bit，相当于每个坐标每多 1 bit 精度就多 1 bit 码率。低位比特基本是不可压的传感器噪声（推论，见 Q5）。
- 换算成每帧大小（以 KITTI 每帧约 12 万点为假设）：G-PCC 2 cm 无损约 115 KB/帧，1 mm 无损约 300 KB/帧。原始 float32 xyz 是 96 bpp，约 1.4 MB/帧。
- 衡量我们的 hit set 时，最直接可比的标准编码器是 G-PCC octree 无损（体素网格本身就是整数坐标，等价于“已量化、lossless”）。但公开数字都来自 10^5–10^6 点的大点云，见 Gaps。

### Gaps
- 没找到 MPEG CTC 官方文档里 V-PCC 无损几何的 bpp 表，也没找到 G-PCC trisoup 的 bpp 数字（trisoup 只有有损模式）。
- 没找到 Draco 在稠密或 LiDAR 数据上的**无损** bpp，只有它和其他方法在有损设置下的相对比较。
- 没找到 G-PCC predictive geometry 开启 angular mode 时的 LiDAR 无损 bpp 公开表。上面的 LiDAR 数字都是 angular mode 关闭的结果。
- G-PCC v14 与 v23 的无损差距（约 0.95 对 0.72 bpp）只有一个来源，帧也不完全相同，需要用 TMC13 自己复现。
- **小点云**（几百到几万点，正是我们的规模）上 G-PCC 和 Draco 的 bpp 与头部开销，没有找到任何公开数据。

---

## Q2. 学习型点云几何编解码器（2019–2026）：无损 bpp 与相对 G-PCC 的收益

### Takeaway
稠密 10-bit 人体点云上，学习型无损编码已经把码率从 G-PCC v14 的约 0.95–1.0 bpp 压到 0.51–0.65 bpp，即 −34% 到 −48%。相对 v23 的报告值（约 0.72 bpp）只省 10–29%。在稀疏 LiDAR 上收益明显缩小：2 cm 精度时无损约 −12% 到 −20%，1 mm 精度时只有 −5% 到 −18%；有损 BD-rate 约 −12% 到 −30%。精度越高、点越稀，学习带来的收益越小。实时性方面（RENO，约 10–26 fps），实时版的码率通常比最强的自回归模型差约 10 个百分点。

### Cited Findings
**稠密物体点云（8iVFB / MVUB，无损）**
- 8iVFB 平均 bpp：G-PCC 0.99，SparsePCGC 0.66（−33.8%），VoxelDNN 0.62（−37.6%，但编/解码 885 s / 640 s），MSVoxelDNN 0.79（−20.5%），OctAttention 0.65（−34.6%，解码 948 s），ECM-OPCC 0.58（−41.9%，解码 19.5 s）。— [ECM-OPCC, arXiv 2211.10916, Table 3](https://arxiv.org/abs/2211.10916)
- MVUB（phil / ricardo，9/10-bit）平均：G-PCC 1.10，OctAttention 0.76，ECM-OPCC 0.73 bpp（−33.6%）；同表中 8i 上 ECM-OPCC 为 −42.4%。— [ECM-OPCC, Table 4](https://arxiv.org/abs/2211.10916)
- SparsePCGC 无损，7 条稠密序列平均 0.589 bpp，G-PCC v14 为 0.952，即 −38.2%，单序列最好 −42.1%。每帧编/解码约 2–6 s，比 G-PCC 快约 2–3 倍。— [SparsePCGC, Table 4](https://arxiv.org/abs/2111.10633)
- EHEM（CVPR 2023）在 8iVFB 上无损平均 0.61 bpp，G-PCC 0.99，OctAttention 0.65。— [EHEM supplement, Table 3](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Song_Efficient_Hierarchical_Entropy_CVPR_2023_supplemental.pdf)
- UniPCGC（AAAI 2025）在 8iVFB 上无损平均 0.51 bpp：比 G-PCC v23（0.72）省 29.2%，比 SparsePCGC（0.56）省 8.1%，比 ECM-OPCC（0.58）省 11.3%，比 OctAttention（0.65）省 21.2%。编/解码 0.56 / 0.57 s（RTX 4080）。— [UniPCGC, arXiv 2503.18541, Table 1](https://arxiv.org/abs/2503.18541)
- VoxelDNN（ICASSP 2021）在体素化几何上平均比 G-PCC 省 28%。— [IEEE Xplore 摘要（检索摘要）](https://ieeexplore.ieee.org/abstract/document/9414763)
- OctAttention（AAAI 2022）自报：在 LiDAR（SemanticKITTI）和物体点云（MPEG 8i、MVUB）上 BD-rate 增益 10%–35%，编码时间比 voxel-based baseline 少 95%。— [OctAttention, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/19942)

**稀疏 LiDAR（无损）**
- SparsePCGC 对 G-PCC v14：KITTI 2 cm 6.130 对 7.637（−19.7%），Ford 2 cm 8.784 对 10.032（−12.4%），KITTI 1 mm 17.971 对 20.154（−10.8%），Ford 1 mm 21.168 对 22.298（−5.1%），平均 −12.0%。原文称 LiDAR 无损模式下它比 G-PCC 慢约 2–3 倍（表中 G-PCC 约 1–1.8 s，SparsePCGC 约 1.6–2.9 s）。作者把收益缩小归因于稀疏点云中点与点之间的相关性难以捕捉。— [SparsePCGC, Table 4 与 §6](https://arxiv.org/abs/2111.10633)
- 第三方重跑的 SparsePCGC 只有 −2% 到 −7%：KITTI 2 cm 7.06，Ford 2 cm 9.69，KITTI 1 mm 19.73，Ford 1 mm 22.27。同一论文的 NPA 方法得到 6.18 / 8.53 / 16.62 / 19.86 bpp，平均比 G-PCC 省 14.72%。— [Xue et al., arXiv 2208.12573, Table 1](https://arxiv.org/abs/2208.12573)。与 SparsePCGC 自报值不一致，差异可能来自训练或测试划分不同。

**稀疏 LiDAR（有损 / 量化 octree，BD-rate 对 G-PCC）**
- SparsePCGC 有损 D1 BD-rate：KITTI 2 cm −33.4%，Ford 2 cm −29.8%，KITTI 1 mm −29.1%，Ford 1 mm −22.0%，平均 −28.6%；D2 平均 −36.2%。稠密点云上 D1 BD-rate 约 −95%，但这是相对 G-PCC octree 的有损模式，而那个模式本身很弱。— [SparsePCGC, Table 4](https://arxiv.org/abs/2111.10633)
- OctSqueeze（CVPR 2020，Uber）在 NorthAmerica（内部数据）上：octree 深度 12 / 14 / 16 时约 3.17–3.48 / 8.59–8.91 / 14.33–14.97 bpp（随上下文特征变化）。量化误差从深度 16 的 0.3 cm 到深度 11 的 9.75 cm，深度每减 1 误差翻倍。它比 Draco 和 MPEG anchor 省 10–20%。— [OctSqueeze, Tables 1–2 与 §4](https://arxiv.org/abs/2005.07178)
- VoxelContext-Net（CVPR 2021）：在 **ScanNet（室内 RGB-D 重建，最大深度 9）** 上比 G-PCC 省 43.66% 码率（OctSqueeze 省 15.00%）；在 SemanticKITTI（深度 12）上省 31.15%（OctSqueeze 省 2.13%）。示例帧：VoxelContext-Net 0.36 bpp / PSNR 55.51，OctSqueeze 0.52 / 54.35，G-PCC 0.45 / 48.46。— [VoxelContext-Net, arXiv 2105.02158, §4](https://arxiv.org/abs/2105.02158)
- ECM-OPCC 在 SemanticKITTI（深度 8–12）上的 bpp：最低档（D1 52.9 dB，CD 0.0037）时 G-PCC 0.185、OctSqueeze 0.177、OctAttention 0.137、ECM 0.131；最高档（D1 77.0 dB，CD 0.00029）时 G-PCC 4.874、VoxelContext-Net 4.497、OctAttention 3.711、ECM 3.380。ECM 在各档比 G-PCC 省 29–32%。— [ECM-OPCC, Table 9](https://arxiv.org/abs/2211.10916)
- EHEM（CVPR 2023）在 SemanticKITTI（16-bit）上平均比 G-PCC 省 28.89%，比 OctAttention 省 19.47%。深度 16 时编/解码 2.53 / 3.01 s，OctAttention 为 0.66 / 708 s，G-PCC 为 1.07 / 0.54 s。示例帧码率约 7.2–7.8 bpp：同等码率下 EHEM 的 D1 为 90.6 dB，G-PCC 为 80.6 dB。— [EHEM, CVPR 2023, Table 1 / Fig. 9 / §5.2](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Efficient_Hierarchical_Entropy_Model_for_Learned_Point_Cloud_Compression_CVPR_2023_paper.pdf)
- RENO（CVPR 2025）对 G-PCC v23 的 BD-rate：KITTI D1 −12.26%，D2 −12.23%；Ford −12.50% / −12.69%。对 Draco 为 −48.34%（KITTI D1）。14-bit 约 10 fps，12-bit 约 20 fps（RTX 3090），模型约 1 MB。12-bit 约为 ±3.2 cm，14-bit 约为 ±8 mm。— [RENO, arXiv 2503.12382](https://arxiv.org/abs/2503.12382)
- RENO 补充材料（SemanticKITTI，对 G-PCC v23，D1 BD-rate）：EHEM −28.89%，Unicorn −25.62%，Unicorn 单阶段版 +6.56%，RENO-L −20.63%，RENO −12.47%。第 11 序列首帧总编码时间（12-bit / 14-bit）：EHEM 1.187 / 4.414 s，Unicorn 0.598 / 0.849 s，RENO 0.038 / 0.073 s。同一帧码率约 2.24–2.28（12-bit）和 6.55–6.58（14-bit），单位未标注，按上下文应为 bpp。正文还写到 Unicorn 编一帧 14-bit 约需 2 s。— [RENO supplement, Tables 4–6](https://openaccess.thecvf.com/content/CVPR2025/supplemental/You_RENO_Real-Time_Neural_CVPR_2025_supplemental.pdf)
- 模型体积：SparsePCGC 稠密无损模型 4.9 MB，稀疏无损模型 21.7 MB（5×5×5 卷积核），稀疏有损模型 31.6 MB。— [SparsePCGC, §7](https://arxiv.org/abs/2111.10633)
- 2025–2026 新工作：
  - Re-Densification + Cross-Scale Propagation（arXiv 2508.20466，2025-08）：KITTI 12-bit 编/解码 26 fps，自称压缩率达到 SOTA，摘要中没有给 bpp。— [arXiv 2508.20466](https://arxiv.org/abs/2508.20466)
  - PACE（Zhu, You, Ding, Ma，arXiv 2605.01320，2026-05）：post-causal entropy modeling，自回归模式下解码延迟降低 90% 以上，自称 SOTA，摘要中没有具体 BD-rate。— [arXiv 2605.01320](https://arxiv.org/abs/2605.01320)
  - LPCM：基于学习的 predictive coding，对标 G-PCC predictive geometry。— [arXiv 2505.20059（检索摘要）](https://arxiv.org/html/2505.20059)

### Inferences
- **“相对 G-PCC 的增益”依赖 TMC13 版本。** 多数学习型论文用 v14 做 anchor。如果 UniPCGC 报的 v23 值（8iVFB 平均 0.72 bpp）可信，OctAttention（0.65）相对 v23 只省约 10%，ECM-OPCC（0.58）约 19%，UniPCGC（0.51）约 29%。我们若要对标，应当用最新 TMC13 自己跑。
- **点越稀疏、精度越高，学习收益越小**：稠密 −35% 到 −48%，LiDAR 2 cm −12% 到 −20%，LiDAR 1 mm −5% 到 −18%。我们的体素集是 6–16 cm 网格上的室内表面，**在体素单位下更像“稠密 2-流形表面”**，而不像稀疏 LiDAR。所以学习型熵模型在这个设定下可能拿到 20–40% 的收益（推论）。前提是点数足够多，模型才能学到并利用上下文。
- 室内数据最接近的证据是 VoxelContext-Net 在 ScanNet 上的 −43.66%（有损截断 octree，深度 9 即 512³）。这说明室内表面的学习先验比 LiDAR 更有效，但它不是无损数字。
- 换算到我们的预算：200–260 B 等于 1,600–2,080 bit。如果 hit set 有 N 个占据体素，且位姿开销可以忽略，那么每个体素约 1600/N 到 2080/N bit。若 N 约 1,000，就是约 1.6–2.1 bit/体素，和 G-PCC 在稠密表面上的约 0.7–1.0 bpp 同一量级甚至略高，学习型熵模型还有空间。若 N 约 300，就是约 5–7 bit/体素，这时头部、位姿开销和小样本上下文不足会占主导（推论，N 需要用我们自己的数据核对）。

### Gaps
- PCGCv2、Unicorn 原文的无损 bpp 表没有拿到（Unicorn 只有 RENO 补充材料里的 BD-rate）。PACE 和 Re-Densification 的具体 bpp 也没有拿到。
- OctAttention 原文的逐序列 bpp 没有直接核对，目前用的是 ECM-OPCC 和 UniPCGC 表中的引用值 0.64–0.65 / 0.76。
- 没有任何论文报告学习型编解码器在 **<10⁴ 点** 的小点云上的 bpp，而这正是多机器人增量消息的规模。
- 没有找到在室内体素占据（如 ScanNet 或 Replica 在 5–10 cm 体素下）上做**无损**学习压缩的公开 bpp。

---

## Q3. LiDAR 扫描 / range image 压缩与深度图压缩：每帧比特数

### Takeaway
range image 加学习型预测（RIDDLE）在 Waymo 上、Chamfer distance 约 0.005 时只要约 3.65 bpp，比 G-PCC 的 10.78 bpp 省 65% 以上。按每帧最多约 17 万点估算，约为每帧几十 KB。实时神经编解码器（RENO）在 12-bit（约 ±3 cm）时约 2.2 bpp，按每帧约 12 万点估算约 34 KB/帧。但在高质量区间，range image 类方法（R-PCC、RTST）反而落后于 octree / sparse 类方法 60–90%。深度图无损编码 RVL 很快（约 1 ms），但每个有效像素至少要 4 bit。

### Cited Findings
- **RIDDLE（Waymo，CVPR 2022）**：range image 大小为 64×2650。Chamfer distance 约 0.005 时码率比 G-PCC 低 65% 以上（10.78 降到 3.65 bpp）。码率约 4 bpp 时，distortion 降低 85% 以上。原始数据按 32 bpp 计（图注 “Groundtruth (32bpp)”）。— [RIDDLE, arXiv 2206.01738, §5](https://arxiv.org/abs/2206.01738)
- RIDDLE 用在 Waymo 训练的模型直接压 SemanticKITTI（伪 range image）：同 Chamfer distance（约 0.005）下码率约 4.3 bpp，比 OctSqueeze、MuSCLE 等此前方法低 50% 以上。— [RIDDLE, §5](https://arxiv.org/abs/2206.01738)
- RIDDLE 在 0.1 m 深度精度下：64 线为 2.23 bpp，下采样到 32 线为 2.55 bpp。LASzip 为 10.62 bpp（PSNR 67.6，CD 0.0048），RIDDLE 为 4.51 bpp（PSNR 72.39，CD 0.0026）。加上 intensity 和 elongation 后，range 通道约占总码率的 70%（range 4.04 bpp，intensity 0.88，elongation 0.78）。— [RIDDLE, 附录](https://arxiv.org/abs/2206.01738)
- 延迟：G-PCC 压缩 1594.5 ms、解压 1052.1 ms（CPU，WOD）。OctSqueeze 为 106.0 / 902.3 ms（GPU，KITTI）。RIDDLE 为 532.5 / 966.3 ms（GPU，WOD），其中网络推理 487 ms。— [RIDDLE, Table 5–6](https://arxiv.org/abs/2206.01738)
- **range image 方法在高质量区间不占优**：KITTI 上 RENO 相对 R-PCC 的 D1 BD-rate 为 −84.80%，相对 RTST 为 −90.66%。R-PCC 和 RTST 只在约 3 bpp 以下的 12-bit 设置下能跑。— [RENO, Table 1](https://arxiv.org/abs/2503.12382)
- **RENO 每帧码率**：SemanticKITTI 第 11 序列首帧，12-bit 约 2.24–2.28，14-bit 约 6.55–6.58（单位按上下文为 bpp）。— [RENO supplement, Table 6](https://openaccess.thecvf.com/content/CVPR2025/supplemental/You_RENO_Real-Time_Neural_CVPR_2025_supplemental.pdf)。解码后的点云上 CenterPoint 检测 mAP 为 65.64，原始点云为 65.97（12-bit）。— [RENO](https://arxiv.org/abs/2503.12382)
- KITTI（HDL-64E）每帧约 12 万点，Ford 每帧约 10 万点。— [SparsePCGC, §5](https://arxiv.org/abs/2111.10633)
- **深度图（RVL，ISS 2017）**：Kinect v2 深度图为 512×424、16-bit（有效值 80–8000 mm，即 13 bit），原始约 424 kB/帧。RVL 的压缩比与 PNG、JPEG-LS、RLGR 相当或略好；压缩 1.2 ms、解压 0.99 ms，大约是次快方法的 4 倍和 2 倍。— [Wilson, Fast Lossless Depth Image Compression](https://www.microsoft.com/en-us/research/uploads/prod/2018/09/p100-wilson.pdf)
- RVL 每个有效像素至少需要 4 bit；Fischer 等（WSCG 2020）的 span-wise 预测方法压缩比明显更高。— [Fischer et al., Improved Lossless Depth Image Compression](https://dspace5old.zcu.cz/bitstream/11025/38439/1/Fischer.pdf)

### Inferences
- 每帧大小估算：
  - RIDDLE（Waymo）：3.65 bpp × 最多约 16.96 万像素（64×2650），不超过约 77 KB/帧。
  - RENO（KITTI 12-bit）：约 2.25 bpp × 12 万点，约 34 KB/帧。
  - G-PCC（KITTI 2 cm 无损）：约 115 KB/帧。
  - RVL 深度图：下限为 4 bit × 有效像素数，满幅 Kinect v2 约 108 kB/帧。

  这些都比我们的 200–260 B/机器人地图大 2–3 个数量级。原因是这些编解码器都保留厘米或毫米级的**原始测量**，而我们的消息只保留 6–16 cm 网格上的**第一击中体素**，并依赖接收端 ray-casting 重建 free space。
- 和 range image 类方法一样，“hit set + poses” 利用了**传感器几何**：射线起点就是位姿，所以 free space 可以由接收端推出，不必传输。RIDDLE 的结果（range image 加预测远好于 octree）说明，把编码组织在传感器坐标系里很有价值。我们的 hit set 若改成按射线 / 扫描顺序做 delta 编码，而不是按无序体素坐标编码，可能还有可观收益（推论）。

### Gaps
- Waymo 每帧实际点数（而非 range image 像素上限）没有核对。
- RVL 在 Kinect 数据上的具体压缩比数字没有从原文图表中提取到，只有“与 PNG / JPEG-LS 相当”的定性结论和 4 bit/像素下限。
- 学习型深度图压缩（如基于 learned image codec 的 16-bit 深度）在机器人数据上的 bits/frame，没有找到可靠数字。

---

## Q4. 机器人占据地图格式：OctoMap .bt / .ot、UFOMap、Voxblox / TSDF、子图压缩；室内地图的实际大小

### Takeaway
OctoMap 的 maximum-likelihood 二进制格式（.bt）每个内部节点用 2 bit/child，即 16 bit/节点。原论文实测：室内走廊 FR-079（43.7×18.2×3.3 m）在 5 cm 下为 0.67 MB，10 cm 下为 0.14 MB；RGB-D 办公室 freiburg1_360（7.9×7.3×4.6 m）在 2 cm 下为 0.52 MB，5 cm 下为 0.07 MB。完整概率格式（.ot）大 20–40 倍。折算约为包围盒内每体素 0.13–0.43 bit。UFOMap 的内存约为 OctoMap 的 38%，但没有给文件或传输大小。Voxblox / TSDF 和多机器人子图传输没有找到可靠的公开大小数字。

### Cited Findings
- **OctoMap Table 2**（Hornung et al., Autonomous Robots 2013），列依次为：3D grid 内存 / 无压缩 octree / pruned / ML 内存 / Full 文件 / Lossy (.bt) 文件，单位 MB：
  - FR-079 corridor（43.7×18.2×3.3 m）5 cm：78.88 / 73.55 / 41.62 / 24.72 / **15.76 / 0.67**
  - FR-079 corridor 10 cm：10.01 / 10.87 / 7.22 / 5.02 / **2.70 / 0.14**
  - freiburg1_360 RGB-D（7.9×7.3×4.6 m，内存数字含颜色）2 cm：252.99 / 159.97 / 45.52 / 20.05 / **21.59 / 0.52**
  - freiburg1_360 RGB-D 5 cm：16.19 / 11.24 / 4.55 / 2.52 / **2.11 / 0.07**
  - Freiburg campus（292×167×28 m）20 cm：Full 49.68，Lossy 2.00；10 cm：Full 379.70，Lossy 13.82
  - New College（250×161×33 m）20 cm：Full 18.65，Lossy 0.99；10 cm：Full 148.75，Lossy 6.40

  — [Hornung et al. 2013, OctoMap preprint, Table 2](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf)
- .bt 编码方式：每个节点只存 8 个子节点的标签（free / occupied / inner / unknown），每个 2 bit，共 16 bit/节点；叶子不需要再写，由标签恢复。作者称即使大型室外环境、细分辨率，文件也从未超过 15 MB。完整格式存节点数据（occupancy float 等）外加每节点 8 bit 的子节点存在位，所以文件大得多。内存方面，32-bit 架构下每个内部节点 40 B、每个叶子 8 B。— [OctoMap preprint, §4.1–4.3, 式 (8)](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf)
- 作者指出 map 文件还能再用通用文件压缩工具压一层。— [OctoMap preprint, §5.4](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf)
- ROS `octomap_msgs`：binary 格式只存 max-likelihood 的 free / occupied；先调用 `toMaxLikelihood()` 和 `prune()`，序列化结果会小很多。— [octomap_msgs 文档](https://docs.ros.org/en/indigo/api/octomap_msgs/html/namespaceoctomap__msgs.html)
- **UFOMap 与 OctoMap 内存对比**（MiB）：FR-078 tidyup 5 cm 为 7.42 对 21.49；FR-079 corridor 5 cm 为 19.70 对 51.86；freiburg1_360 2 cm 为 15.98 对 42.05；Freiburg campus 20 cm 为 58.71 对 155.46；New College 20 cm 为 29.41 对 75.40。— [UFOMap wiki: Performance](https://github.com/UnknownFreeOccupied/ufomap/wiki/Performance)。论文称 UFOMap 约为 OctoMap 的 38%，原因是叶子结构更小、子节点内联存储，而且显式表示 unknown。— [UFOMap, arXiv 2003.04749](https://arxiv.org/abs/2003.04749)
- 多机器人场景：一篇综述提到某 UAV 系统在新发现体素超过阈值时触发子图共享，也提到高精度占据地图传输需要大带宽，可能失败或残缺。— [Multi-Robot Cooperative Exploration survey, arXiv 2503.07278（检索摘要）](https://arxiv.org/pdf/2503.07278)
- 面向通信的地图压缩（2D）：Psomiadis et al. 用多分辨率 abstraction 压缩占据栅格，比特数为 n = k·n_m + n_i；该工作只针对 2D 栅格和路径规划。— [Communication-Aware Map Compression for Online Path-Planning, arXiv 2309.13451](https://arxiv.org/abs/2309.13451)
- 稠密点云地图的学习压缩：Wiesmann et al.（RA-L 2021）用 KPConv 自编码器压缩 KITTI 和 nuScenes 地图块，称同码率下重建优于其他方法；具体 bpp 没有提取到。— [Deep Compression for Dense Point Cloud Maps](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wiesmann2021ral.pdf)

### Inferences
- **按包围盒体素数折算 .bt 码率**（我自己的计算）：
  - FR-079 10 cm：437×182×33 ≈ 262 万体素，0.14 MB ≈ 1.12 Mbit，约 **0.43 bit/体素**
  - FR-079 5 cm：约 2,100 万体素，0.67 MB，约 **0.26 bit/体素**
  - freiburg1_360 5 cm：约 212 万体素，0.07 MB，约 **0.26 bit/体素**
  - freiburg1_360 2 cm：约 3,316 万体素，0.52 MB，约 **0.13 bit/体素**

  这些 .bt 文件显式编码了 free space。
- **与我们的数字对比**：我们 1,600–2,080 bit 对应 524,288 个网格体素，约 0.003–0.004 bit/网格体素；按观测体素（5.2–7.9 万）算约 0.02–0.04 bit/观测体素。按 .bt 的约 0.26–0.43 bit/体素粗略外推到这 5.2–7.9 万个观测体素，OctoMap .bt 消息约 1.7–4 KB，是我们 hit set + poses 的 **7–20 倍**。差距主要来自 .bt 显式传 free space，而我们让接收端 ray-casting 自己算出来。这个外推忽略了 unknown 区域剪枝和尺度差异，只能看作量级估计。
- OctoMap 的 ML 量化加剪枝本身就是一种“有损但地图语义无损”的压缩：只要下游只用 free / occupied 二值，它对规划就是无损的。我们的 hit set 方案把这个思路推得更远，连 free 也不传。
- 所以任何学习型 mapping codec 要声称有优势，至少要同时打赢两条基线：(a) OctoMap .bt（再加 gzip 或 zstd）；(b) G-PCC / 学习型无损编码器压 hit set，再加位姿。在我们的数量级上，(b) 才是硬基线。

### Gaps
- Voxblox / TSDF 的子图文件或消息大小（KB/子图），以及 Voxblox 与 OctoMap 的对比数字，都没有找到。
- UFOMap 只有内存数字，没有序列化或压缩文件大小。
- 没有找到多机器人系统（如 Kimera-Multi、Swarm-SLAM、DOOR-SLAM）中**室内 3D 占据子图**每次传输的实际 KB 数和带宽统计。
- OctoMap .bt 再经 gzip / zstd 后的大小没有公开数字（原文只说还能再压）。

---

## Q5. 传感器噪声对（学习型）无损码率的影响，以及边压缩边去噪

### Takeaway
没有找到直接系统测量“噪声幅度和无损 bpp 关系”的论文。间接证据有三条，都指向噪声会线性抬高码率、而且学习也压不掉。第一，量化精度每细 1 bit（低于传感器噪声的位基本是噪声），G-PCC 和 SparsePCGC 的无损码率都增加约 3 bpp，即每个坐标约 1 bit。第二，学习型方法相对 G-PCC 的收益从 2 cm 的约 20% 降到 1 mm 的约 5–11%。第三，较嘈杂的采集数据（MVUB）比干净数据（8iVFB）贵约 10–20%。联合压缩和去噪目前只有零星工作（PCGCD，DCC 2025），以及面向 G-PCC 的 RD 优化预处理（报告 −38.84% BD-rate），都没有可直接引用的 LiDAR 或室内数字。

### Cited Findings
- **精度与码率**：KITTI 上 G-PCC 无损从 2 cm（14-bit）到 1 mm（18-bit），码率从 7.637 增到 20.154 bpp；SparsePCGC 从 6.130 增到 17.971。Ford 上 G-PCC 从 10.032 增到 22.298，SparsePCGC 从 8.784 增到 21.168。学习收益同步从 −19.7% / −12.4% 缩到 −10.8% / −5.1%。— [SparsePCGC, Table 4](https://arxiv.org/abs/2111.10633)
- 第三方结果同样显示 1 mm 时收益更小：NPA 方法在 KITTI 2 cm 上为 −18.90%，在 Ford 1 mm 上为 −10.98%。— [arXiv 2208.12573, Table 1](https://arxiv.org/abs/2208.12573)
- SparsePCGC 作者把稀疏点云上收益偏小解释为点的空间分布更稀疏，点间相关性难以刻画。— [SparsePCGC, §6.2](https://arxiv.org/abs/2111.10633)
- 采集数据之间的差异：G-PCC 在 8iVFB 上 0.99 bpp，在 MVUB 上 1.10 bpp；ECM-OPCC 分别为 0.57 和 0.73，学习收益从 −42.4% 降到 −33.6%。— [ECM-OPCC, Table 4](https://arxiv.org/abs/2211.10916)
- **联合压缩与去噪**：Meng, Ou, Zhang, Liu, Li，“PCGCD: Joint point cloud geometry compression and denoising”，DCC 2025，pp. 23–32。我只在其他论文的参考文献里看到它，方法和数字都没拿到。— [被 arXiv 2508.20466 引用](https://arxiv.org/abs/2508.20466)
- **面向 G-PCC 的 RD 优化预处理**：Ma et al. 2025 在编码前做点云预处理，报告平均比 G-PCC 省 38.84% BD-rate；摘要没有说明数据集，也没有明说是去噪。— [arXiv 2508.01633](https://arxiv.org/abs/2508.01633)
- 纯去噪方向（不含压缩）的近期工作包括 PCDNF（TVCG 2023）、GeoGCN，以及 2025 年的去噪 benchmark 综述。— [PCDNF](https://doi.org/10.1109/tvcg.2023.3292464)；[Survey arXiv 2508.17011](https://alphaxiv.org/overview/2508.17011v1)
- OctoMap 用 clamping 和 ML 阈值化做有损压缩，其作用相当于把概率噪声“截断”到离散状态，换来可剪枝的结构。— [OctoMap preprint, §3.4](https://www.hrl.uni-bonn.de/papers/hornung13auro-preprint.pdf)

### Inferences
- **第一性原理**：无损编码的码率下限是数据熵，i.i.d. 噪声的熵不可压缩。精度每细 1 bit，码率约增加 3 bpp（三轴各约 1 bit），而且对 G-PCC 和学习型模型几乎一样（3.1 对 2.96 bpp/bit）。这说明**低于噪声水平的比特对任何编解码器都是纯成本**，学习先验只在高于噪声的结构尺度上起作用。
- 在我们 6–16 cm 的体素网格上，cm 级测距噪声大多被量化吸收。剩下的噪声主要表现为：(a) 表面“增厚”或抖动，同一面墙落在相邻两层体素；(b) 孤立的伪击中体素（多径、玻璃、动态物体）。在 octree 和上下文编码里，孤立点特别贵（G-PCC 为此专门有 isolated-point 工具，见 SparsePCGC 对 LiDAR 运行时间的讨论）。因此对 hit set 先做“地图一致性”去噪（例如要求多次观测才确认、剔除孤立体素），可能比换更强的熵模型更省码率。但这已经是**有损**操作，必须用下游指标来衡量，比如 ray-cast 重建后的 free / occ 一致率和规划可行性（推论）。
- 学习型 mapping codec 要声称优于“hit set + poses”，应在同样的噪声和去噪设定下比较。否则带隐式去噪的 learned codec 和严格无损的 hit set 之间的码率差，有相当一部分其实是“删掉噪声”的收益，而不是“更好的先验”。

### Gaps
- 没有找到受控实验：给干净点云逐级加高斯噪声或离群点，测 G-PCC 和学习型 codec 的无损 bpp 曲线。
- PCGCD（DCC 2025）以及其他联合压缩去噪工作的具体数据集和码率收益没有拿到。
- 没有找到在室内 RGB-D 或 LiDAR 体素地图上量化“噪声 → 码率”的研究。
