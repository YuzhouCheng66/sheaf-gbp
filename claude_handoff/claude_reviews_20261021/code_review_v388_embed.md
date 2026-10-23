VERDICT: LAUNCH OK WITH CHANGES (M1, M2)

# 发射前代码审查：v388 解码器输入端码级嵌入单变量（batch U，`chain_day4_U.sh`）

审查者：独立 Claude 子代理，2026-10-23。只读审查，未修改任何被审文件。本地无 GPU、无 `runs/`；远端 `uk-linux` 上的文件（`runs/v388-gate/gate-s342202.json`、批次 Q 的 `run.json` / `initial.pt`、批次 T 的 `lease.json`、`scripts/analyze_v375.py` 是否已拷贝）无法在本地核实，凡属此类均标明"无法本地核实"。本地用 `D:/Anaconda3/envs/jax-gpu/python.exe`（torch 2.11.0）复跑了 `test_v388_embed.py`（ALL TESTS PASSED），并另写了一段检查脚本核对参数分组、边界插值、查表精确性与 AdamW 首步动力学（结果引用于下）。

## 范围

| 文件 | 行数 | sha256 |
|---|---|---|
| `scripts/claude_v342/train_v388.py`（被审） | 780 | `1e34ce3d60d47ae4a57576508ac6ae603cbf71ca126eb2d5098769cb08f44dcb`（与链脚本第 32 行硬编码值一致） |
| `scripts/claude_v342/train_v386.py`（基线，批次 T 已发射） | 743 | `77e8bc73895bd9765765a9afebc6fe701603e826b99a83c30da3b2b7dd44a068`（= 批次 T 链脚本核对值） |
| `scripts/claude_v342/test_v388_embed.py`（被审） | 69 | `c20e1b5d502d139abd17af2fb9f03351c6f7afa2e08e759f81fd8f93133a8283` |
| `scripts/claude_v342/analyze_v388.py`（被审） | 94 | `12ab0fe526c9c48381f716b719dc447e6d4b4e351a39ede863948f977ea88043` |
| `scripts/claude_v342/analyze_v375.py`（被导入） | 84 | `4a004073c622fc23b30430fe518180521871e3a0c71b8a4066bee3c556ef166e`（与 T 批审查记录一致） |
| `scripts/claude_v342/preflight_check_v386.py`（被审，新增 `--code-embed`） | 61 | `e8d2666e5f17d45e602e29aa19650c30b8a985dfed18aa7219a0a30124dce37e` |
| `scripts/claude_v342/chain_day4_U.sh`（被审） | 112 | `d2a757375ec69164b27baebfc7ca2904f4ea44230318b8a156d46f9ae713b9d7` |
| `scripts/claude_v342/chain_day4_T.sh`（对照，今日已成功运行） | 109 | `27d9f0d58a9571edb17122d964dc36edf8f6a27c0fdb7e26e12896ebf8dccd92`（前 16 位 = T 批发射记录值） |
| `scripts/claude_v342/gate_v388_codeaxis.py`（被审，已在 v384 权重上运行） | 123 | `bb841db32a2f33713b6e87c7f0166e22a46db8c42211d3eb3bc813a552bf17c8` |
| `scripts/mapping_implicit_model_v192.py`（`latent_quantize`）、`scripts/mapping_multiroom_forest_v295.py`（profile-4 线路量化器） | 只读参考（线路噪声估算） |

方法：`diff train_v386.py train_v388.py`（`grep -c '^[<>]'` = 47 行，与声明一致）；通读 `train_v388.py` 全文核对改动之外每处 `CODE_EMBED` / `embed` / RNG 消耗 / 写文件位置；`diff analyze_v386.py analyze_v388.py`、`diff chain_day4_T.sh chain_day4_U.sh` 逐行走查；`bash -n chain_day4_U.sh` 通过；在本地用 Python 生成与 `gate_v388_codeaxis.py` 第 121–122 行同构的 `indent=1` JSON，实跑链脚本第 33 行的 `grep -o` 与第 78–83 行的 heredoc，再用 `json.load` 验证租约文件；按 v295 `quantize` 的 profile-4 码本（`RANGES[4] = (1e-8, 128)`，第 35、189–192 行）推算线路重取整噪声的级数。

## 逐项核对

### 1. `train_v388.py` 与 v386 的 47 行差异

| 处 | v388 行 | 内容 | 判断 |
|---|---|---|---|
| 文档串 | 1–11 | 说明 | 无代码影响 |
| 模块常量 | 199 | `CODE_EMBED = 0` | 默认 0 ⇒ v386 逐字节等价（测试 1 验证 state_dict 与解码器输出 `torch.equal`） |
| `OwnedDecoder.__init__` | 211–226 | 表 `self.embed`（255×D；第 0 列 = `arange(-127,128)/127`，第 1.. 列 = `Generator(388)` 的 0.05 高斯噪声经 3 点环形平滑）；`new = Conv2d(D+1, c)`，权重置零后 `new.weight[:,0] = old.weight[:,0]`、`new.weight[:,D] = old.weight[:,1]`、`new.bias.copy_(old.bias)`；`self.pre[0] = new` | 见第 1 项结论 |
| `embed_plane` | 228–233 | `u = (clamp(mean)+1)·127`，`i0 = floor(u).clamp(0,253).long()`，`w = u − i0`，线性插值，`permute(0,2,1).reshape(b, D, 8, 8)` | 与 `post[:, :64].reshape(b,1,8,8)` 的行主序一致 |
| `forward` | 239–240 | posterior_only 分支下均值平面替换为 D 个平面，对数方差平面不变 | anchored 分支（242–243）不受影响；`CODE_EMBED>0` 时第 213 行断言强制 posterior_only |
| argparse / 全局 / 断言 | 485、489、491 | `--code-embed`；写入全局；`D ≥ 0 且 (D == 0 或 owned+posterior_only+mu_bits 8)` | 正确 |
| 热启动校验 | 613 | 校验元组加入 `('code_embed', args.code_embed)` | 见第 4 项 |
| meta | 625 | `code_embed=args.code_embed`（同时 `args=vars(args)` 含该键） | 正确 |

通读确认：损失（433–447）、GBP 原语（301–408）、数据（520–600）、优化器 / 调度器（618–621）、训练循环（696–754）、评估与最终阶段（636–775）与 v386 无字节差异；`CODE_EMBED` 只在 `main()` 第 489 行赋值之后被 `OwnedDecoder.__init__`（第 212 行）读取。

**检查项 1（D=8 是否从恰好 v378/v386 的函数出发）：是，函数等价到 1e-6，其余张量逐位相同。**
- `pre[0]` 的偏置：第 225 行 `copy_`，测试第 48 行 `equal` 验证；两列权重：第 225 行按切片赋值，测试第 50 行 `torch.equal` 逐位验证，其余列 `abs().sum() == 0`。
- `old` 的去向：第 226 行 `self.pre[0] = new` 替换 `Sequential` 内的子模块，`old` 为局部变量，不再被任何模块引用。本地核实：D=8 时 `decoder.named_modules()` 中只有一个 `pre.0`，state_dict 多出的键仅 `decoder.embed`，`parameter_groups()['decoder']` = **636,570** = 630,498 + 2,040 + 4,032（trunk 218,976 / head 16,770 不变），总参数 872,316（测试 3 的 +6,072 一致）。优化器建在 `model.parameters()`（第 619 行）上，无悬挂参数。
- `parameter_groups()`（第 290–291 行）用 `self.decoder.parameters()` ⇒ 表在 `decoder` 组内（本地核实 `any(p is m.decoder.embed ...)` = True），第 728 行的梯度范数日志把它并入 `decoder` 项；没有单独的 `embed` 范数（见建议 S4）。
- RNG：第 223 行的 `nn.Conv2d(D+1, c)` 在 `fork_rng` 种子流内多消耗一次随机数，但它是 `OwnedDecoder` 中最后一个构造的模块，且随即被清零 / 覆盖，`fork_rng` 退出时恢复全局状态 ⇒ 编码器、head、`pre[1..]`、trunk 与 v386 逐位相同（测试 2 第 44–47 行），数据顺序由独立的 `order_rng`（第 698 行）决定，与 Q 相同。
- 注意（非缺陷）：表的噪声列用固定种子 388 的 CPU 生成器，**两个种子（342101 / 342202）的表初值完全相同**，这是设计选择，但意味着两种子在该分量上不是独立复制（见第 7 项）。
- "函数等价"只到 `allclose(atol=1e-6)` 而非逐位：原因见第 2 项（查表并非严格精确）。

**检查项 2（边界插值、STE 路径、dtype/device）：边界正确；查表在 255 级中的 183 级严格精确、72 级差 1 ulp；dtype/device 无问题。**
- 边界：本地核实 `mean=+1 ⇒ u=254, i0=253, w=1.0, e == E[254]`（`torch.equal`）；`mean=−1 ⇒ i0=0, w=0, e == E[0]`；`mean=126/127 ⇒ i0=253, w=0, e == E[253]`；`mean=0 ⇒ E[127]`。`clamp(0,253)` 保证 `i0+1 ≤ 254`。
- 精确性：`latent_quantize`（第 284–288 行，与 v192 第 37–46 行同一算法 `round(z·127)/127`）输出的 float32 级值 `k/127` 经第 230 行 `(mean+1)·127` 后，**72 个级（k = −126, −125, …）落在整数下方 1 ulp**（`u = k+126.9999995`），于是 `i0 = k−1, w = 0.9999995`，输出 = `5e−7·E[k−1] + (1−5e−7)·E[k]`，与 `E[k]` 之差 ≤ 5e−7·|E[k]−E[k−1]|（第 0 列 4e−9，噪声列约 1e−8）。前向上可忽略（解码器输入是 float32，conv 本身误差同量级），但严格说前向不是"纯查表"，且**反向对均值的斜率在这 72 级用左侧差分 `E[k]−E[k−1]`、其余 183 级用右侧差分 `E[k+1]−E[k]`**，取决于浮点舍入而非设计（见建议 S3）。
- STE：rollout 的 `belief()`（max 融合第 382–391 行；sum 融合第 350–352 行）都以 FP64 的 `E/P` 经 v192 `latent_quantize` 重取整后 `.float()` 输出 ⇒ 送入解码器的均值恒在 255 级上（第 391 行），梯度经 `hard.detach() + (bounded − bounded.detach())` 流回编码器，再经 `embed_plane` 的斜率传递；测试 3 第 65 行验证了到均值输入的梯度非零。
- dtype/device：表是 float32 `nn.Parameter`，随 `model.to(device)`（第 602 行）上卡；`mean` 来自 `.float()`；`.long()` 索引与 `self.embed[i0]` 的高级索引在 CUDA 上合法；反向用 `index_put_(accumulate=True)`（原子加，非确定次序）——与已有 cuDNN 卷积 / 双线性插值反向同属非确定类别，不引入新类别。无 autocast / AMP。

**检查项 3（泄漏与诊断路径）：无泄漏；`--oracle-union` 与 `--refine-steps` 可用。**
- `self.embed` 只在 `OwnedDecoder.forward` 第 239 行被读取；`model.source`（第 279–282 行）、rollout（第 301–408 行）、线路量化器、一致性损失（第 719–725 行用 `feats[k][:, :64]` 与 `mu_u`）均不接触它。表是解码器参数，不随样本变化，不上线路。
- `--oracle-union`（第 658–667 行）构造的 `oracle_features` 与 K0 同布局 `[hard8 mean 64, log var 64]`，经 `predict(model.decoder, cat(feats[0], feats[k]))`（第 671 行）走同一 `forward` ⇒ 均值平面照常查表。
- `--refine-steps`（第 642–656 行）对 `z` 求导：`latent_quantize(z)·mask → own_f → stage_loss(model.decoder, cat(own_f, own_f))`，梯度经表的斜率回到 `z`，可用；语义变化（斜率改由表决定）属本干预的一部分。本批 `--refine-steps 0`。

**检查项 4（热启动校验）：对本批无影响；对未来"从 D=0 检查点热启 D=8"存在静默陷阱。**
- 第 613–614 行 `payload.meta.get(mkey, mval) == mval`：v384/v378/v386 检查点无 `code_embed` 键 ⇒ 取 CLI 值 ⇒ 恒通过。D=0 ← D=8 检查点（meta 8 ≠ 0）与 D=8 ← v388 D=0 检查点（meta 0 ≠ 8）都会正确报错。
- D=8 ← v384 检查点：`decoder.pre.0.weight` 形状 (64,2,3,3) ≠ (64,9,3,3) 被 `skipped`，`decoder.embed` 为 `missing`；无 `--allow-partial-init` 时第 616 行断言拦截（安全）。**若加 `--allow-partial-init`，`pre.0` 保持的是"新鲜随机 Conv2d(2,c) 的恒等重排"而不是检查点里已训练的 `pre.0`**——解码器第一层随机、其余层已训练，函数与检查点完全不同，却不报错。本批从零训练，不触发；但链脚本第 13–14 行写明的后续（"与 T 的赢家和三周期计划合并"）若走热启动，必须先加"提升路径"（把检查点 `pre.0.weight[:,0]→new[:,0]`、`[:,1]→new[:,D]`）——见建议 S2。

**检查项 5（`analyze_v388.py` 的键）：全部存在，对照正确解析为 0，导入可解析。**
- `code_embed=meta.get('code_embed', meta['args'].get('code_embed', 0))`（第 32 行）：臂的 `run.json` 顶层（训练器第 625 行）与 `args` 都写 8；对照（train_v378/v382 写出）两处皆无 ⇒ 0；`decoder_width` 同理 ⇒ 1.0。第 58–60 行 `ok` 要求 `float(a/c decoder_width) == 1.0`、`int(a code_embed) == --embed (8)`、`int(c code_embed) == 0`、`enc_same`（`initial.pt` 的 `trunk.`/`head.` 张量逐位相等，第 42–46 行）。
- `RECIPE_ARGS`（第 26 行）不含 `decoder_width`/`code_embed`，故二者不进 `recipe_diffs`；`train_cache` 字符串须与 Q 的命令行相同（沿袭 T 批审查已核对的 `$C/houseexpo_fit-k4800`）。
- 判定（第 74–83 行）与文档串 / 链脚本头部一致：`min(g) ≥ 1` ⇒ 杠杆；`max|g| < 1` ⇒ 非杠杆；否则 ambiguous；STRUCTURAL 标签要求 VAL 与 fresh 上 `min ≥ 2 且 mean ≥ 3`；FPR `> 0.3` 追加违例。`fresh_R8` 缺失时 `gf` 为空 ⇒ 不加标签（保守）。
- `from analyze_v375 import eq_weight`：本地 `analyze_v375.py` 第 19 行签名 `eq_weight(rows, k, metric='wall_f1', drop=frozenset())`，返回 `(float, dict)`，与第 67 行的解包匹配；链脚本以 `$P scripts/analyze_v388.py` 启动，`sys.path[0]` = `scripts/` ⇒ 远端须有 `scripts/analyze_v375.py`（T 批同样依赖，无法本地核实）。

**检查项 6（`chain_day4_U.sh`）：语法正确、守卫正确、写入范围正确；两处必改（M1 租约 JSON 被引号破坏；M2 与批次 T 的等待竞态）。**
- `bash -n` 通过。
- GATE 抓取（第 33 行）：`gate_v388_codeaxis.py` 第 121–122 行写 `out['verdict'] = dict(verdict=..., ...)`、`indent=1` ⇒ 文件中出现 `"verdict": {` 与嵌套的 `"verdict": "LAUNCH with expectation downgraded to +1 pp"`。`grep -o '"verdict": "[^"]*"'` 只匹配后者（前者冒号后是 `{`），本地同构 JSON 实跑结果：`GATE=["verdict": "LAUNCH with expectation downgraded to +1 pp"]`，`case *LAUNCH*` 命中 ⇒ 守卫逻辑正确。
- **但 `$GATE` 带着双引号被写进第 80 / 97 行的 `"gate": "$GATE"`** ⇒ `"gate": ""verdict": "LAUNCH ...""`，本地 `json.load` 报 `JSONDecodeError: Expecting ',' delimiter`。两种模式的租约文件都会是非法 JSON（M1）。
- 等待批次 T（第 37 行）`pgrep -f "^$P scripts/train_v386.py "`，锚定且含尾随空格，能匹配 T 的预检与正式任务。**竞态**：若批次 T 走的是 SEQUENTIAL（`chain_day4_T.sh` 第 93–107 行），种子 1 退出到种子 2 `nohup` 之间最长约 60 s 空档（T 的轮询周期 60 s），U 以 120 s 轮询，命中空档的概率最高约 50 %；命中后第 42 行 `pgrep -f "^$P scripts/train_v3"` 在同一瞬间也看不到进程，随后两个预检（以及其后的正式任务）会与 T 的种子 2 同卡运行：预检速率失真，T 的种子 2 可能错过租约（T ⇒ INCONCLUSIVE）。T 若是 PARALLEL 则无此空档（M2；本地无法核实 T 的模式）。
- pgrep 模式：并行等待 `"^$P scripts/train_v388.py .*runs/v388-embed/embed"`（第 90 行）匹配 `--out runs/v388-embed/embed-s<seed>`，不匹配预检目录 `pf-par-*`（预检由第 55 行 `wait` 等待）；顺序等待带 `-s$seed`（第 105 行）。
- 租约命名 `lease.json` / `lease2.json`（第 78、94 行）与 T 相同；`$CHK` / `$CHK2` 不含引号。
- `--decoder-params 636570`：= 630,498 + 255×8 + 7×64×9 = 630,498 + 2,040 + 4,032，本地实例化核实（见第 1 项）；`preflight_check_v386.py` 第 37 行对 `parameters.decoder` 做等值检查，第 34–35 行对 `code_embed` 做等值检查（`r.get('code_embed', r['args'].get('code_embed', 0))`，臂写 8），第 39–40 行 trunk/head 与对照相等。
- 写入范围：`mkdir -p runs/v388-embed`（第 43 行）；预检目录与日志、租约、`embed-s*` 与日志、`analysis.json` 全在 `runs/v388-embed/` 下；`LOG=runs/chain_day3.log` 只 `tee -a`（第 28–29 行）。训练器只写 `--out`。无其他写入。
- 其他：第 41 行只禁止 `embed-s*` 已存在（可重跑预检），第 38–40 行检查对照四个文件存在；`--min-rate` 沿用默认 0.094（见建议 S1）；头部第 13–14 行"combine with batch U's winner"应为 batch T；第 48 行注释仍写"W=2 decoder"。

**检查项 7（科学有效性）：是一个干净的单变量对照，但要带以下保留（详见"建议项"S5 与结论）。**

### `test_v388_embed.py` 覆盖了什么、没覆盖什么

覆盖（本地复跑通过）：D=0 逐位等价（state_dict + 解码器输出）；D=8 除 `pre.0.weight`/`embed` 外逐位等价；偏置 / 两列权重 / 零列 / 表第 0 列 / 噪声幅度；初始化函数等价 `atol 1e-6`；任意连续均值下第 0 通道插值线性（能捕获索引偏移）；梯度到表与到均值输入；参数增量 6,072。

未覆盖：
- 边界 `mean = ±1`（`torch.rand*2−1` 几乎不取端点，`clamp(0,253)` 分支未被测试触发）——本审查已手工核实。
- 查表在真实级值上的精确性（183/255 精确、72 级差 1 ulp）——本审查已核实，无害。
- 文档串第 7 行承诺的**第 4 项"热启动 meta 校验"在测试体内不存在**（代码止于第 3 项）。
- CUDA dtype/device、完整一步训练（优化器含表）、两种子噪声表相同——由远端 200 步预检（有限损失、参数量等值）部分覆盖。
- D=8 的 state_dict 键集合恰为 v386 + `decoder.embed`（由参数量等式间接覆盖）。

### `gate_v388_codeaxis.py` 三项测量与判定规则

- (a) 级扫描（第 73–89 行）：`base = k0_feats(...)` 是该场景全部智能体的 K0 特征；取一个智能体 `a` 的一行 `repeat(255,1)`，只改第 `k` 列（`k < 64`，即均值平面；`64+k` 的对数方差不动），`k` 只从 `mask[a] > 0` 的自观测块里抽；解码后按 `(by, bx) = divmod(k, 8)`、`p = 16` 切出该块，统计 `== 1` 图案的精确去重数与 Hamming ≤ 2 的贪心聚类数。**按说明实现**。细节：贪心聚类与行序相关（上界性质），`pred0` 未使用，均无碍。
- (b) 扰动（第 60–72 行）：`torch.where(mask > 0, (mu·127 + s).clamp(−127, 127)/127, mu)` 只移自观测坐标并钳位；重新 `k0_feats`（经 `latent_quantize` 重取整）后对 `private` 评 K0 wall-F1；`c1 = −min(Δ(+1), Δ(−1))` 取较差一侧。**按说明实现**（按场景等权而非按分层等权，作为门槛可接受）。
- (c) 残差解剖（第 97–112 行）：`pred_k8` vs `target_k8`，`scope = tgt ≥ 0`，精确 F1；容差版对两侧各做 `binary_dilation(iterations=1)`：`tp_t = pw & dilate(tw)`、`fp_t = pw & ~dilate(tw)`、`fn_t = tw & ~dilate(pw)`。**按说明实现**。注意 `ndimage.binary_dilation` 默认结构元是十字（4 邻域），所以"1 px"是 4 邻域容差，53 % 的抖动占比是**下界**（8 邻域会更高，余量更少）。文件键 `final-val:r3` 的拆分（第 108 行）对 `final-val-panel-r3-n8-i0.npz` 正确。
- 判定（第 114–120 行）：`d2 ≥ 200 或 c1 ≥ 1 ⇒ ABORT`；`d2 < 128 且 c1 < 1 且 js < 0.5 ⇒ LAUNCH`；否则 `LAUNCH 降级到 +1 pp`——与文档串第 11–13 行逐条一致；报告值 (37, 0.2, 0.53) 落入第三支，与记录的结论一致。
- 复用限制：第 36–37 行只从 meta 设置 `READOUT_MODE` / `PRECISION_MODE`，**不设 `DECODER_WIDTH` / `CODE_EMBED`**，因此对 v388 D=8 的训练后权重跑同一扰动测试（链脚本头部第 19 行预注册的事后 ±1/±3 级测试）会在 `load_state_dict` 处失败（多出 `decoder.embed`、`pre.0.weight` 形状不符）——见建议 S6。

## 必改项

**M1 — 租约文件会被 `$GATE` 里的双引号写成非法 JSON（`chain_day4_U.sh` 第 33、80、97 行）。**
现状：`grep -o` 的匹配文本连同引号进入 `$GATE`（`"verdict": "LAUNCH with expectation downgraded to +1 pp"`），heredoc 写出 `"gate": ""verdict": "LAUNCH …""`，本地 `json.load` 失败。租约是预注册审计记录，两种模式都受影响。要求：改为不带引号的纯文本，例如
`GATE=$($P -c "import json; print(json.load(open('runs/v388-gate/gate-s342202.json'))['verdict']['verdict'])" 2>/dev/null)`，并把第 34 行改为 `case "$GATE" in LAUNCH*) ;; *) … ;; esac`（文件缺失时 `$GATE` 为空 ⇒ 仍 ABORT）。改后用一份本地同构 JSON 跑一次第 78–83 行的 heredoc 并 `json.load` 验证。

**M2 — 与批次 T 的等待存在最长 60 s 的竞态窗口（`chain_day4_U.sh` 第 37、42 行）。**
现状：只等 `train_v386.py` 进程；T 若以 SEQUENTIAL 运行，其种子 1 退出到种子 2 启动之间（`chain_day4_T.sh` 第 105–106、94–102 行，轮询 60 s）U 的 120 s 轮询有最高约 50 % 概率命中空档，随后 U 的预检 / 正式任务与 T 的种子 2 同卡：U 的预检速率失真，T 的种子 2 可能错过租约（T ⇒ INCONCLUSIVE）。要求（任选其一，推荐前者，一行改动且无副作用）：
- 第 37 行改为同时等待链进程：`while pgrep -f "chain_day4_T.sh" >/dev/null || pgrep -f "^$P scripts/train_v386.py " >/dev/null; do sleep 120; done`（U 自身命令行不含该串，不会自锁）；
- 或在启动 U 前核实 `runs/v386-decoder/lease.json` 的 `"mode": "parallel"` 且不存在 `lease2.json`（T 已无第二次启动），并把这一核实写进 `chain_day3.log`。
若 T 已完全结束，此项为零成本的无操作，但仍应改，因为链脚本是可复用模板。

（两项都只改 `chain_day4_U.sh`，不触及 `train_v388.py`，第 32 行的训练器 sha 无需更新。）

## 建议项

- **S1（预检假中止风险）**：`preflight_check_v386.py` 的 `--min-rate 0.094` 是 Q 的 W=1 双任务实测值，U 的算力 ≈ Q（解码器 +0.96 % 参数，且都在 8×8 输入层，FLOPs 增量可忽略），所以并行预检的速率正好落在阈值上；正常波动（例如 0.092）会被判 INVALID ⇒ 链脚本第 73 行 ABORT，白耗一个租约时段。单任务独占速率约 0.06 s/更新，阈值降到 0.085 仍能分辨"未重叠"。建议第 57 行显式加 `--min-rate 0.085` 并在租约 `preflight` 行里带上判定依据（阈值来源：HANDOFF L38 的 0.094 × 0.9）。
- **S2（热启动提升路径）**：`train_v388.py` 第 613–614 行对 `code_embed` 的缺省改为 0（`payload.meta.get('code_embed', 0) == args.code_embed`），使 v384 + `--code-embed 8 --allow-partial-init` 不能静默通过；若后续真要从 D=0 检查点热启 D=8，加一条显式的"嵌入提升"分支（检查点 `pre.0.weight[:,0]→new[:,0]`、`[:,1]→new[:,D]`、偏置照抄，表保持恒等初始化），并写一条测试断言提升后的解码器函数与检查点 `allclose`。本批不受影响；改动训练器须重跑测试并更新链脚本第 32 行 sha。
- **S3（查表精确与单侧斜率一致）**：第 230–232 行可改为 `i = (mean.detach()·127).round().long() + 127`（精确整数索引，forward 严格查表）并用 STE 给斜率：`e = E[i] + (u − u.detach())·(E[min(i+1,254)] − E[i])`——前向逐位等于 `E[k]`，反向统一为右侧差分，`k = 127` 处用左侧差分。当前实现在 72/255 级上用左侧、其余用右侧，取决于浮点舍入而非设计；前向误差 ≤ 1e−8，可以不改。若改，同 S2 的测试 / sha 要求。
- **S4（可观测性）**：`parameter_groups()` 增加一个只用于日志的 `embed` 组（不改优化器分组），或在第 736 行日志里加 `|E[:,1:]|` 的范数与 `pre.0.weight[:, 1:D]` 的范数，使"表是否真的离开恒等初始化"在训练中可见（见 S5 的操纵检验）。改动训练器同样须更新 sha；若不想动训练器，可在事后从 `final.pt` 读出。
- **S5（预注册补充：操纵检验与零结果解释）**：本地核实 AdamW 首步后 `pre.0` 的 7 个新列幅度 ≈ 调度后的 lr（Adam 归一化，**零初始化不等于"门控渐入"**），而表的噪声列在第一步梯度恒为零（conv 新列为零 ⇒ 无梯度到表）；"符号读取"只能在 conv 新列先长起来之后二阶地学习。因此 `|gain| < 1 pp` 的预注册读法"平滑读取不是限制"过强：零结果也可能是"该预算 / 该初始化下没学出来"。建议在跑完之前把以下写入 `analyze_v388.py` 或单独脚本作为预注册的操纵检验：(i) `final.pt` 中 `E[:,1:]` 与初值的差范数、`pre.0.weight[:,1:D]` 的范数；(ii) 用 `gate_v388_codeaxis.py`（按 S6 修好后）在训练后的臂上重跑级扫描——`mean_distinct_h2` 相对 v384 的 37 是否上升，这是"解码器是否开始按符号读码"的直接证据；(iii) 预注册的 ±1/±3 级扰动成本（线路鲁棒性）。若 (i)(ii) 显示表几乎未动，判定应写成"本预算下未学出 / 无信息"而非"非杠杆"。
- **S6（门脚本复用）**：`gate_v388_codeaxis.py` 第 36–37 行补 `trainer.DECODER_WIDTH = meta.get('decoder_width', 1.0); trainer.CODE_EMBED = meta.get('code_embed', 0)`，否则不能在 v388 权重上做事后测试；同时把第 105 行的膨胀结构元写明（`structure=np.ones((3,3))` 为 8 邻域，或在输出里注明当前是 4 邻域），并删除未用的 `pred0`。
- **S7（测试补全）**：`test_v388_embed.py` 补：`mean = ±1` 的端点断言（`e == E[254]` / `E[0]`）；文档串承诺的第 4 项（构造 D=8 模型后对 v386 的 `initial.pt`-同构 payload 走第 604–617 行逻辑，断言无 `--allow-partial-init` 时抛错）；D=8 的 state_dict 键集合 = v386 键集合 ∪ {`decoder.embed`}。
- **S8（文案）**：`chain_day4_U.sh` 第 13–14 行"batch U's winner"→"batch T's winner"；第 48 行"W=2 decoder"→"code-embed 8 decoder"；头部第 19 行预注册的事后扰动测试写明用哪个脚本（S6 后的门脚本）。
- **S9（沿袭 T 批 S1/S9，仍未采纳）**：分析完整输出重定向到 `runs/v388-embed/analysis.txt`；第 42 行守卫可加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && ABORT`。

## 结论

**LAUNCH OK WITH CHANGES（M1、M2）。**

训练器方面：v386 → v388 的 47 行差异全部是嵌入旋钮及其参数 / meta / 热启动校验；D=0 逐字节等价（测试 + 本地复跑）；D=8 时编码器、head、数据顺序与 Q 相同，解码器除 `pre.0.weight`（两列照抄、其余零）与新表外逐位相同，初始函数与 Q 等价到 1e−6（差异来自 72/255 级的 1 ulp 插值，前向误差 ≤ 1e−8，无害）；边界 `±1` 的查表、`clamp(0,253)`、`.long()` 索引、行主序重排、参数分组（表在 `decoder` 组、636,570）均核实正确；表只在解码器前向被读取，不进融合、不上线路；`--oracle-union`、`--refine-steps` 路径布局不变、可用。分析脚本读取的键都由训练器写出，对照的 `code_embed`/`decoder_width` 正确解析为 0/1.0，STRUCTURAL 规则与预注册文本一致。门脚本的三项测量按说明实现，判定规则按文档串执行，(37, 0.2 pp, 0.53) 落入"降级到 +1 pp"一支。

两处必改都在链脚本：`$GATE` 带引号写入租约使两种模式的 `lease.json` 成为非法 JSON（M1，本地实跑验证）；与批次 T 的等待只盯训练器进程，T 若为顺序模式则有最长 60 s 的空档可被 U 命中并同卡（M2）。两处都不改训练器，sha 不变。

科学保留（建议随 S5 在发射前写入预注册记录）：
1. **线路重取整噪声**：profile-4 线路把 `(p, η)` 分别量化（精度对数码本步比 1.096 ⇒ p 相对误差 ≤ 4.7 % ⇒ 送达均值相对误差 ≤ 2.3 % ⇒ |μ|≈1 处 ≤ 2.9 级，另加块指数下的 1/127 取整），接收端再 `latent_quantize` ⇒ k>0 阶段读表的级可比发送端偏 ≤ 3 级。表若学成"尖锐符号"会在 k>0（权重 4×0.125）被惩罚，K0（权重 0.5）读的是干净级值——学到的表会被线路噪声迫使保持局部平滑，这正是预注册 ±1/±3 事后测试要量的东西。
2. **53 % 抖动占比是下界**（4 邻域膨胀），可寻址残差 < 47 % 的 (1−F1)，+1 pp 的期望值合理；门测的是 v384（三周期、更强的模型），对 Q 预算模型的余量只是近似迁移。
3. **恒等初始化只在第 0 步成立**：Adam 从第 1 步起以调度 lr 整体推动零列；表的噪声列前期梯度为零，机制是二阶学出来的 ⇒ 零结果需配操纵检验（S5）才能区分"非杠杆"与"未学出"。
4. **容量 / 特征混淆**：+6,072 参数与 7 个额外输入通道，正结果可能是"输入端多了随机特征"而非"符号读取"；冻结表（仅学 conv 新列）是便宜的后续对照。
5. **干预同时改变了编码器的梯度**（STE 斜率改由表决定），这是"解码器输入端嵌入及其对编码器训练信号的影响"的联合效应，解释时不能说成纯解码器改动。
6. **两种子共享同一张噪声表**（Generator 388），n = 2 且 CUDA 非确定 ⇒ "两种子都 ≥ +1 pp"的功效未量化；Q 自身的种子离散度与批次 S 的每周期 +1.0/+1.3 pp 是比较尺度。
7. 远端事项无法本地核实：`scripts/train_v388.py` 字节与本地一致（sha 1e34ce3d…）、`scripts/analyze_v375.py` 已拷贝、`runs/v388-gate/gate-s342202.json` 存在且结构同第 121 行。

---

## 审查后的修改（主代理，2026-10-23 16:09 UTC，发射前）

- **M1 落实**：lease 的 `gate` 字段改为用 python 取出不带引号的判定字符串（`json.load(...)['verdict']['verdict']`），`case "$GATE" in LAUNCH*)`；本地用 `json.loads` 验证了 lease 模板为合法 JSON。
- **M2 落实**：等待循环同时监视 `chain_day4_T.sh` 进程与 `train_v386.py` 进程，T 顺序模式两个种子之间的空档不再被误判为"T 已结束"；通过后写一条日志。
- **S1 采纳**：U 的算力与现有模型相当，并行预检的"重叠"下限由 0.094 改为 0.075 s/更新（批次 Q 双任务 0.089–0.094；W=2 单任务 0.083），避免误判 INVALID。
- 其余建议（S2–S9）记录在案，未改代码：热启动 D=8 自 D=0 需 `--allow-partial-init` 且 pre.0 形状不同（本批从头训练，不涉及）；post-hoc 扰动测试脚本需从 meta 读 `CODE_EMBED`（结果出来后做）；零列在第一步就被 Adam 移动、表的噪声列初始梯度为零——"无增益"不能自动读作"不是杠杆"，分析时附报表的学习量（‖E − E₀‖）。
- 发射版本 `chain_day4_U.sh` sha256 前 16 位 `12797618c6b04d8b`；`train_v388.py` `1e34ce3d60d47ae4` 未变。16:09:36 UTC 启动，等待批次 T 结束后自动预检并发射。
