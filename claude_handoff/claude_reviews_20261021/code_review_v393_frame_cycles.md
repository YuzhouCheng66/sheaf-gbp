VERDICT: LAUNCH OK

# 发射前代码审查：v393 宽解码器 + 公共帧条件化的第 2、3 周期（batch X，`chain_day5_X.sh`）

审查者：独立 Claude 子代理，2026-10-24。只读审查，未修改任何被审文件。本地无 GPU、无 `runs/` 目录：远端的批次 V 检查点、v391 对照文件、3D 批次的实际进度均无法本地核实，凡属此类在下文标明"无法本地核实"。本地做了四件可做的事：`diff chain_day4_W.sh chain_day5_X.sh` 逐行核对；`sha256sum` 核对训练器与三个链脚本；把 X 的 lease heredoc 用占位值渲染后 `json.load`；把 X 第 57–76 行的内嵌分析原样抽出，在四组构造的 `final.json` 上运行（全部完整 / 某种子第 3 周期缺目录 / 某种子 `updates` 未跑满 / 0.99 pp 边界）。

## 范围

| 文件 | 行数 | sha256（前 16 位） | 备注 |
|---|---|---|---|
| `scripts/claude_v342/chain_day5_X.sh`（被审，新） | 79 | `a95889714f42ac34` | LF 行尾、无 CR；`bash -n` 通过 |
| `scripts/claude_v342/chain_day4_W.sh`（模板，已成功执行） | 82 | `936d1b02c56fec8c` | **= D5-8 记录的发射版本**，所以 diff 是对着真正跑过的脚本做的 |
| `scripts/claude_v342/chain_day5_3D.sh`（X 要等待的批次） | 133 | `3bf3f21d38e2497c` | = `code_review_v400_3d.md` 记录的发射版本（M1–M3 落实后） |
| `scripts/claude_v342/chain_day4_V.sh`（热启动来源批次） | | `1feda7580d9cf55f` | = D5-9 记录的发射版本；第 37 行 COMMON 用于核对 meta 断言 |
| `scripts/claude_v342/train_v392.py`（训练器） | 795 | `ef4dc60082ef9f5f…00d2` | 本地计算值 = X 第 23 行硬编码值 = V 发射版本（V 第 29 行、CR392 §范围） |
| `scripts/claude_v342/train_3d_v400.py` | | `aa238ec3d41114b6` | 只用于核对 pgrep 模式；= 3D 发射版本 |
| `OVERNIGHT_REPORT_20261021.md` D5-8 / D5-10 / D5-11 / D6-0，`HANDOFF_20261023.md` §2，`code_review_v391_wide_cycles.md`，`code_review_v392_frame.md`，`code_review_v400_3d.md` | | | 只读参考 |

diff 共 17 个 hunk，全部落在：头部注释（2–14 行）、启动日志与训练器 sha（21–23）、存在性守卫（25–26）、永不覆盖（28）、等待循环（30）、日志措辞与输出目录（32–33）、COMMON（34）、`run_cycle` 的输出目录 / lease 路径 / lease 字段 / 日志 / 训练器调用（36、38、40–42、44–45、47–48）、两次 `run_cycle` 的实参（52–53）、分析的对照路径 / 臂路径 / 判定字串 / 输出路径 / 日志（59、63、70、75、79）。**没有任何 diff 之外的改动**（W 的 S1 未舍入判定、S2 措辞、`return 1 || continue`、前台训练器、lease 先于发射等结构全部原样继承）。

## 逐项核对

### 1. 训练器、`--frame-cond`、热启动

- **sha 固定**：第 22–23 行 `train_v392.py` 的 sha 硬编码 `ef4dc600…00d2`，本地 `sha256sum` 一致，且与批次 V 发射时固定的值相同 ⇒ X 用的是产生热启动检查点的同一训练器 ✓。
- **COMMON**（第 34 行）：与 W 第 37 行相比只多 `--frame-cond`；与 V 第 37 行相比只有 `--lr 1e-4`（V 从头训练用 3e-4，周期配方用 1e-4，= R/S/W）。其余（`--world houseexpo_fit`、`--train-cache …/houseexpo_fit-k4800`、`--arm owned --size 128`、`--epochs 4`、`--val-every 1 --refine-steps 0`、`--readout posterior_only --fusion max`、`--expect-updates 57600`、`--decoder-width 2.0`）与 W、V 逐字相同 ✓。`--precision` / `--mu-bits` / `--q-bits` / `--wire` 两边都走默认（learned / 8 / 8 / sealed）。
- **热启动路径**（`train_v392.py` 616–628 行）：`loaded` = 同名同形状张量；`load_state_dict({**own, **loaded}, strict=True)`；断言 `arm` 相同（624）；第 625–626 行对 `readout_mode, precision_mode, mu_bits, q_bits, wire, decoder_width, frame_cond` **七项** meta 与 CLI 逐一断言（`frame_cond` 与 `decoder_width` 都在元组里 ✓）；第 627–628 行在没有 `--allow-partial-init` 时断言 `missing == skipped == []`（X 没有传该旗标 ⇒ 任何形状不符都会立即致命，而不是静默部分加载）。
- **形状是否全量匹配**：模型形状是 `(arm, size, READOUT_MODE, DECODER_WIDTH, FRAME_COND)` 的确定函数——`OwnedDecoder.__init__`（237–243 行：`planes = 2`，`c = 128`，`lift = 256`，`widths = (192,128,96,64)`）+ `CnnTrunk`（104–120 行：`frame_cond` 时对 `lift[0]` 与四个 `block[0]` 做 `widen_conv(+3)`）。V 的 final 与 X 的模型在这五个量上完全相同 ⇒ 每个张量同名同形状 ⇒ `loaded` = 全部、`missing = skipped = []` ✓。第 3 周期从 X 自己的 `c2-s<seed>/checkpoints/final.pt` 热启动，同理 ✓。
- **检查点是否带 meta**：`save('final.pt', …)`（642–646、773 行）写入 `meta=meta`；`meta`（634–637 行）含 `arm, readout_mode, precision_mode, mu_bits, q_bits, wire, decoder_width=2.0, frame_cond=True` ⇒ 七项断言有真实的键可比（不是 CR392 S3 指出的"缺键默认通过"情形）。V 的 COMMON 与 X 的模型定义旗标逐项相同 ⇒ **七项断言全部通过，无需 `--allow-partial-init`**。
- 优化器 / 余弦日程：`--init-from` 只加载 `state_dict`，AdamW 与 `LambdaLR`（629–632 行）按 57,600 次更新、200 步 warmup 重新开始 ⇒ 与 R/S/W 的"周期"语义相同 ✓。

### 2. 守卫、等待循环、pgrep

| 守卫 | 行 | 核对 |
|---|---|---|
| 训练器 sha | 22–23 | 见第 1 项 ✓ |
| 存在性 | 25 | `runs/v392-frame/frame-s<seed>/checkpoints/final.pt` + `final.json`（批次 V 两个种子），`runs/v391-wide-cycles/c2-s<seed>/final.json` 与 `c3-s<seed>/final.json`（W 的两个周期）——与第 52–53 行实际用到的热启动与对照路径一一对应 ✓ |
| 批次 V 完整性 | 26 | `final.json['updates'] == 57600`，否则 ABORT；D5-10 记录两个种子都跑满 ✓（v391 对照的完整性未重查，D5-8 记录四个运行 4,956–5,004 s 全部跑满） |
| 永不覆盖 | 28 | `ls -d runs/v393-frame-cycles/c2-s*` 存在即 ABORT；训练器 `out.mkdir(exist_ok=False)`（520 行）双重保证；`c3-s*` 不可能先于 `c2-s*` 存在 ✓ |
| 等待 3D 批次 | 30 | `pgrep -f "chain_day5_3D.sh"` **或** `pgrep -f "^$P scripts/train_3d_v400.py "`，120 s 轮询。见下文 |
| 无训练器 | 31 | `^$P scripts/train_v3`——**不匹配 `train_3d_v400.py`**，见下文 |

**等待循环是否覆盖 3D 批次的全部阶段**：3D 链脚本（已发射版本）调用训练器的四处——并行预检（42 行，`&` + `wait`）、并行正式运行（83 行，`nohup $P scripts/train_3d_v400.py …`，`nohup` 会 exec，argv[0] 仍是 python 路径）、顺序预检（91 行）、顺序正式运行（103 行）——命令行都以 `$P scripts/train_3d_v400.py ` 开头，本地用这几条 argv 字符串测试锚定模式 `^$P scripts/train_3d_v400.py `：全部命中 ✓。3D 链脚本里没有训练器进程的时段（预检到发射之间、顺序模式两个种子之间、训练完成后的内嵌分析）其 bash 仍然活着，命令行 `bash scripts/chain_day5_3D.sh`（无论是否套 `setsid nohup`）被第一个子句命中 ⇒ X 继续等待 ✓。按任务说明 3D 走的是并行分支：两个训练器结束后链脚本还有一次 `sleep 60` 轮询 + 分析，X 最晚在链脚本退出后 120 s 内启动。**没有竞争空档；无自匹配**（X 自身命令行 `bash scripts/chain_day5_X.sh` 不含 `chain_day5_3D.sh`；`tee`、`sha256sum`、`$P -c`、`$P -` 都不匹配任何模式）。

**第 31 行的通用训练器守卫是否覆盖 `train_3d_v400.py`**：不覆盖——`scripts/train_v3` 与 `scripts/train_3d_v400.py` 在第 15 个字符处就分叉（本地测试：`train_v3` 对 3D argv 为 N，对 `train_v386/v388/v392` 为 Y）。它仍能拦住任何残留的 2D 训练器（T/U/V/W 的 v386/v388/v392）。**对预注册的日程而言前面的等待循环已经足够**：循环退出的那一刻，3D 链脚本和 3D 训练器都不存在，而 3D-A 批次只有链脚本会发射 3D 训练器 ⇒ 不可能在 31 行之前冒出一个 3D 训练器。守卫漏掉的只是"有人在那 ≤ 120 s 的窗口里手工启动 `train_3d_v400.py`，或后续另一个 3D 链脚本（例如 3D-B）在 X 等待期间被发射"这类计划外情形。3D 链脚本自己第 32 行用的是更宽的 `^$P scripts/train_`，与 X 头部第 13 行"never co-runs with any trainer"的措辞一致；建议 X 也改成这一形式（S1，一个 token，方向只会更安全），本审查不把它列为必改。

**启动顺序的运行约束（非代码问题，与 W 审查相同）**：X 第 30 行只在 3D 链脚本 **已经在跑** 时才会等待。任务说明 3D 的两个并行运行正在跑、约 06:20 UTC 结束，所以发射 X 时第 30 行会正常进入等待；发射前用 `pgrep -af chain_day5_3D.sh` 确认一次即可。反方向：若之后再发射任何"等 `train_v392.py` 退出"的链脚本，它应同时等 `chain_day5_X.sh` 的 bash（X 两个周期之间有 1–2 s 没有训练器进程的空档，120 s 轮询有小概率落进去，导致 X 的下一周期被挤成双任务而 INCONCLUSIVE）。

### 3. 租约与对照

- **lease JSON**（38–43 行）：本地用占位值渲染同一 heredoc 并 `json.load` ⇒ 合法；`"frame_cond": true` 解析为 Python `True`，`"lr": 1e-4` → `0.0001`，`"decoder_width": 2.0`，`warm_start_sha256` 为 64 位十六进制字符串（`sha256sum | cut -d' ' -f1`），`"trainer": "scripts/train_v392.py"`、`"trainer_sha256": "$TSHA"` ✓。`control` 字段 = `runs/v391-wide-cycles/c<cyc>-s<seed>/final.json (wide decoder without frame conditioning, same seed, same cycle)`——路径与标签都对（CR392 M2 那类"路径是 Q、标签是 V"的错误没有重现）✓。`expectation` 与头部第 9–11 行、D5-11 判据逐字对应 ✓。lease 在训练器发射前写出；`LS` 取在 `sha256sum` 之前（保守）✓。
- **第 2 / 3 周期的对照传参**（52–53 行）：第 2 周期热启动 `runs/v392-frame/frame-s$seed/checkpoints/final.pt`、对照 `runs/v391-wide-cycles/c2-s$seed/final.json`；第 3 周期热启动 `runs/v393-frame-cycles/c2-s$seed/checkpoints/final.pt`、对照 `runs/v391-wide-cycles/c3-s$seed/final.json` ✓（`$4` 只进入 lease 的 `control` 字段，分析块另行硬编码同一路径，两处一致）。
- **内嵌分析**（56–79 行）：`for cyc, ctrl in ((2, "runs/v391-wide-cycles/c2-s{s}"), (3, "runs/v391-wide-cycles/c3-s{s}"))`，臂路径 `runs/v393-frame-cycles/c{cyc}-s{s}/final.json` ✓；读取键 `updates`、`val['8']['all']['wall_f1']`、`val['8']['all']['unknown_known_fpr']`——臂侧由 `train_v392.py` 第 772–790 行写出（`final['val'] = ev_val['summary']`），对照侧 v391 由 `train_v386.py` 写出且 W 的同一分析块已成功消费（D5-8 的表出自 `runs/v391-wide-cycles/analysis.json`）✓。判定字串：`conditioning transfers (new best)` / `faster learning only` / `ambiguous` / `INCONCLUSIVE`，FPR > 0.3 pp 只追加 ` BUT FPR protection violated`；主判定取 `cycle3`，`cycle2` 标为 secondary ✓。比较用未舍入的差值、输出时才 `round(…, 3)`（W 的 S1 已继承）。
- **本地运行四组构造数据**：(A) 两种子完整、第 3 周期 +1.2 / +0.7 → `ambiguous`，第 2 周期 +1.5 / +1.1 且一侧 FPR +0.4 → `conditioning transfers (new best) BUT FPR protection violated`；(B) 删除 `c3-s342202` 目录 → `cycle3 = INCONCLUSIVE`，`cycle2` 照常判定；(C) `c3-s342202` 的 `updates = 51000` → 同 (B)；(D) 第 3 周期 +0.99 / +1.00 → `ambiguous`（0.99 不被舍入成 1.0）。`analysis.json` 写到 `runs/v393-frame-cycles/` ✓。四种路径都不抛异常。

### 4. 残留引用

- **可执行行**：枚举 X 中全部 `runs/` 路径（25、26、28、33、36、38、52、53、59、63、75 行）：热启动只指向 `v392-frame` 与 `v393-frame-cycles`，对照只指向 `v391-wide-cycles`，输出只指向 `v393-frame-cycles`；没有 `v386-decoder`、`v383-cont`、`v384-cycle3`、`train_v386.py`、`train_v388.py`、`chain_day4_U.sh` 的任何残留 ✓。`v391` 出现的每一处都是作为对照，符合设计。
- **注释行**：第 29 行 `# wait for batch U (its chain and trainer); never co-run` 是 W 的旧注释，未随第 30 行一起改（S2，纯文本）。头部第 5 行引用的 v391 数字（c2 .899/.899，c3 .905/.909）与 D5-8 一致 ✓；第 3 行的批次 V 数字（+1.82 / +0.99）与 D5-10 一致 ✓。

### 5. 预算与排程

- **单次运行耗时**：W 的四个宽解码器周期实测 4,956–5,004 s（D5-8，`final.json.seconds`，含加载 + 4 轮 + 4 次轮末验证，不含两次最终评估）；批次 V（同一架构 + `--frame-cond`，从头训练）单任务约 85 min ≈ 5,100 s（任务说明；D5-10 未给秒数，无法本地核实）。`--frame-cond` 只给 5 个卷积各加 3 个输入通道（+19,872 参数），算力几乎不变，与 V ≈ W 的实测一致。
- **租约余量**：每个运行 `LDE = LS + 6600`、`LDT = LS + 7150`、`inclusive_seconds 7200`（37 行，= W/R/S/T/U/V），满足 HANDOFF §2 第 25 行。训练截止余量 ≈ 6,600 − 5,100 ≈ 1,500 s（约 23 %），与 W（20–25 %）同量级。最终评估窗口 550 s：批次 V 的单任务预检已按 `--mode solo --final-window 550 --frame-cond 1`（V 第 58 行）验证过本架构 2 × val 放得下 ✓。到达训练截止时训练器写出 `updates < 57600` 的 `final.json`（772–790 行一定写出），链脚本第 48 行判 INCONCLUSIVE、不发第 3 周期 ✓。
- **排程**：四个运行严格顺序（前台 `wait`），约 4 × 5,200 s + 切换 ≈ 5.8 h。3D 并行运行约 06:20 UTC 结束、X 最晚 06:22 UTC 开始 ⇒ **预计约 12:10 UTC 完成**（无法本地核实 3D 的实际进度）。与 W 不同的是这段时间落在 uk-linux 的白天：如有其他用户的 GPU 任务出现，"宽解码器单任务"的前提不再成立，速度掉到共享的 ~0.14 s/update 就会错过 6,600 s 截止 ⇒ 该种子 INCONCLUSIVE（失败方向安全，但浪费一个租约）。X 第 31 行只排除本项目训练器、不检查其他 GPU 用户（S 审查 S3 / CR386 S9 / CR391 S5 / CR392 S7 沿袭未采纳），见 S3。
- **写入范围**：`mkdir -p runs/v393-frame-cycles`、四个 lease、四个运行目录 + `.log`、`analysis.json`、追加 `runs/chain_day3.log`；不删除、不覆盖历史 ✓（HANDOFF §2 第 23–24 行）。

## 必改项

无。17 个 hunk 逐行核对后没有发现会让批次跑错对照、跑错模型、覆盖历史或在干净机器上误中止的改动。

## 建议项

- **S1（一个 token，建议发射前顺手改）**：第 31 行 `^$P scripts/train_v3` → `^$P scripts/train_`，与 `chain_day5_3D.sh` 第 32 行一致，使守卫真正覆盖 `train_3d_v400.py` 及任何后续训练器，兑现头部第 13 行"never co-runs with any trainer"。对本批预注册日程不是必需（等待循环已覆盖），但它是唯一能拦住"X 等待期间另有 3D 训练器被手工 / 另一链脚本启动"的地方。
- **S2（注释）**：第 29 行 `# wait for batch U …` → `# wait for the 3D batch (its chain and trainer); never co-run`。
- **S3（运行前提机器化，沿袭 CR386 S9 / CR391 S5 / CR392 S7）**：第 31 行之后或 `run_cycle` 入口加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "ABORT: GPU busy"; exit 2; }`（或至少 `log` 一条警告）。X 在白天运行，这是本批唯一与 W 不同的风险来源。
- **S4（附报项）**：头部第 11 行与 D5-11 承诺的 K0、fresh、8+ 室分层、第 2 周期次级比较，内嵌分析只算 R8 与 FPR；请像 D5-8 那样预先登记事后脚本（数据都在 `final.json` / `validation/final-val.json`），避免事后挑选。
- **S5（后续链脚本的约定）**：之后任何要排在 X 之后的链脚本，等待条件应为 `pgrep -f chain_day5_X.sh || pgrep -f "^$P scripts/train_v392.py "`（两个子句），不要只等训练器 argv（X 两周期之间有 1–2 s 空档）。

## 结论

**LAUNCH OK。**

`chain_day5_X.sh` 是对已成功执行的 `chain_day4_W.sh`（本地 sha = D5-8 记录的发射版本）的机械派生，17 个 hunk 全部是预期中的替换：训练器 `train_v386.py` → `train_v392.py`（sha `ef4dc600…` 与本地文件及批次 V 发射版本一致）、COMMON 加 `--frame-cond`、热启动 `v386-decoder/wide` → `v392-frame/frame`、对照 `v383-cont` / `v384-cycle3` → `v391-wide-cycles/c2` / `c3`、输出 `v391-wide-cycles` → `v393-frame-cycles`、等待对象 `chain_day4_U.sh` / `train_v388.py` → `chain_day5_3D.sh` / `train_3d_v400.py`、lease 加 `"frame_cond": true` 并把 `control` 与 `expectation` 改写为 v391 口径、判定字串 `capacity` → `conditioning`。可执行行没有任何 v386 / v383 / v384 / 批次 U 的残留。

热启动在训练器 616–628 行走全量加载：V 的 final 与 X 的模型在 `(arm, size, readout, decoder_width, frame_cond)` 五个决定形状的量上相同，七项 meta 断言（含 `decoder_width`、`frame_cond`）都有真实的键可比且全部通过，不需要 `--allow-partial-init`；优化器与余弦日程按 R/S/W 的周期语义重启。守卫在干净机器上不会误中止；等待循环同时看 3D 链脚本与锚定的 3D 训练器 argv，覆盖并行 / 顺序两个分支的所有阶段，没有竞争空档，也不会自匹配。第 31 行的通用守卫确实不匹配 `train_3d_v400.py`，但对预注册日程而言等待循环已足够，改成 `train_` 只是更稳妥（S1）。lease 渲染后是合法 JSON；内嵌分析在完整 / 缺目录 / 未跑满 / 0.99 pp 边界四种情形下都给出正确的判定且不崩溃。每个运行 6,600 / 7,150 s 的截止线相对 W 实测 ~5,000 s、V 实测 ~5,100 s 有约 23 % 余量，最终评估窗口已由 V 的单任务预检验证；四个租约约 5.8 h，排在 3D 批次之后，预计约 12:10 UTC 完成。

发射前建议顺手采纳 S1（一个 token）与 S2（注释）；S3 是本批白天运行带来的唯一新增风险的机器化检查，强烈建议但不阻塞。发射时请确认 `chain_day5_3D.sh` 仍在运行（否则第 30 行不会等待），并把发射版本的 sha 记入 D5-11。

---

## 审查后的修改（主代理，2026-10-24 05:15 UTC，发射前）
- 结论 LAUNCH OK，无必改项。采纳 S1（通用训练器守卫由 `train_v3` 扩为 `train_`，覆盖 3D 训练器）与 S2（陈旧注释）。
- 发射版本 `chain_day5_X.sh` sha256 前 16 位见 chain 日志；在 3D 链脚本仍在运行时启动，等 3D 结束后顺序执行 4 个租约（约 06:20 → 12:10 UTC）。
