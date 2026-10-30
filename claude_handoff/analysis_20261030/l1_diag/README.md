# L1 诊断（2026-10-30，非预注册；CPU；不改批次 L1 的设计与判决）

S1-hetero-s342101 与 B0 无差别（D8-84）之后，为区分"预算不够 / 共识混合太弱 / 接口缺陷"所做的两类诊断。全部在 uk-linux 的 CPU 上运行（`CUDA_VISIBLE_DEVICES=`，nice 10），不占 5090；路径 `/mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021/toy_l1/`。

- `toy_train.diff`：对钉定版 `l1_train.py`（sha256 前缀 5f8eee78）的唯一改动——加 `--full-ft` 开关（玩具小模型全参训练）；`l1_model.py` / `l1_fusion.py` / `l1_data.py` 用钉定版原文件（aa3a4eec / a0f56eb6 / 41d0db81）。
- `toy_gen.py`：玩具数据生成器（l1_data 记录格式）。题目 "what is the code ?"，答案词（40 个候选）只在一方分片的金段里（`the code is <A> .`），另一方金段是干扰事实（`the box is <B> .`），每方再各一段填充；`answer_owner` = 持有答案的分片。train 4,000 / val 500 / fresh 300，种子 11 / 22 / 33。
- `toy_run.sh`：启动一个玩具臂（B0 / S1 / U / S1L1，异构 tiny-a + tiny-b，batch 8，warm-up 20，`--no-grad-ckpt`）。
- `toy_runs/*.stdout`：各次运行的日志（含每步损失、VAL / fresh、owner 桶、缺席曲线、C 断言）。`k8`：LoRA-only 800 步；`lr*r32*`：LoRA-only 3,000 / 15,000 步；`ft`：全参 3,000 步；`ft20k`：全参 20,000 步；`ftS013`：全参 3,000 步 + 读者输出 ×0.13（软 token 缩到词嵌入尺度，见 `l1_model_reader_scale.diff`，环境变量 `L1_TOY_READER_SCALE`）；`ftS013_20k`：同上 20,000 步。
- `toy_probe.py`：载入一次运行的 `trainable.pt`，在 VAL 上统计学到的 λ（分位、夹持占比）、融合态里"自己 z"的份额、软 token 范数与词嵌入范数之比。`probe_real_S1s1.txt` 是它在真实 S1-hetero-s342101 权重上的输出。
- `toy_probe2.py`：软 token 替换测试——同一批题，把读者的输入换成 融合态 / 只有自己的 z（p=1）/ 全零 / 随机 LayerNorm 向量 / 0.03 倍随机，比较答案损失（按 answer_owner 分桶）。输出 `probe2_real_S1s1.txt`。

读法与结论见 `claude_handoff/OVERNIGHT_REPORT_20261021.md` 的 D8-84 诊断注记。
