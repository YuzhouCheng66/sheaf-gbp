VERDICT: LAUNCH OK

# 发射前代码审查：v384 第三余弦周期（batch S，`chain_day4_S.sh`）

审查者：独立 Claude 子代理，2026-10-23。只读审查，未修改任何被审文件；本地无 GPU、无 `runs/` 目录，所以远端检查点、batch R 的耗时与 lease 文件无法在本地核实，凡属此类均在下文标明"无法本地核实"。

## 范围

| 文件 | sha256 |
|---|---|
| `scripts/claude_v342/chain_day4_S.sh`（被审，62 行） | `9e7ed9251ea632d515cbf82675e38f5619095d018cc8f67fe36593eb5786fda5` |
| `scripts/claude_v342/chain_day4_R.sh`（对照基线，46 行） | `f83e477f9ed49af9e596d8603eb1aaf6076c297e8a8e227299912ef54a98f768` |
| `scripts/claude_v342/train_v378.py`（训练器，732 行） | `db7bcc3528d58624a7ec34863f2be492fbd5aea9b87025239c662678fb875bff`（与 S 第 26 行硬编码值一致） |
| `claude_handoff/HANDOFF_20261023.md` 第 6 节第 1 条 | 只读参考 |

审查方法：对 R/S 做逐行 `diff`；对照训练器中 `--init-from`（556–569 行）、调度器（570–573 行）、`final.json` 写出（709–727 行）、聚合键（626–642 行）；手工走查 bash 守卫与内嵌 Python 的各条路径。

## 逐项核对

### 1. 与 chain_day4_R.sh 的全部差异

`diff chain_day4_R.sh chain_day4_S.sh` 的结果只落在以下几处，训练命令行（S 第 40 行 vs R 第 26 行）除 `--out` 与 `--init-from` 两个参数外**逐字相同**（`--world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --seed $seed --deadline-epoch $DE --deadline-total $DT --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600`）。环境变量、`$P`、`$S`、`$C`、`LOG`、`log()`（S 17–22 行 = R 9–14 行）完全一致。

预期之内的差异：
- 热启动路径：`runs/v383-cont/cont-s$seed/checkpoints/final.pt`（S 27、34、40、51 行）。
- 输出目录：`runs/v384-cycle3/c3-s$seed`（S 28、30、32、40、44、51 行）。
- 守卫：S 24–29 行（训练器 sha、热启动存在、输出目录不存在、无 `train_v378.py` 进程）。
- lease 内容：新增 `warm_start_sha256`、`control` 两个键，`trainer_sha256` 改用变量 `$TSHA`（S 33–36 行）。
- 分析中的 FPR 行：S 48、54、58、59 行。

**预期之外、但属无害的差异（须知悉）**：
- (a) **R 第 17 行的"等待前序批次结束"循环（`while pgrep -f "runs/v382-long/d4800e4"`）在 S 中被删除**，改为 S 第 29 行"若有任何 `train_v378.py` 进程则中止（exit 2）"。行为语义从"排队等待"变为"拒绝启动"。这意味着 S **不能**预先排在别的训练任务后面挂起；必须在 GPU 已空时启动。对本实验（头部第 16 行声明 GPU 已空）无影响。
- (b) 日志文本：`"cycle3 seed $seed pid $!"`（S 41 行）vs `"cont seed ..."`；`"batch S start/done/analysis"`。仅文字。
- (c) 分析输出格式：S 59 行打印 `"R8 gain pp", [...], "FPR delta pp", [...]`，R 43 行只打印列表。仅格式。
- (d) 头部注释 2–16 行全部重写。仅注释。

除此之外没有任何差异。结论：**"唯一变量是热启动点"的声明成立**（Adam 状态在 571 行新建、`LambdaLR` 在 573 行以 200 步 warm-up + 余弦重新开始、`--init-from` 在 557–561 行只加载 `state_dict`，与头部第 6–7 行描述一致）。

### 2. 路径一致性与 pgrep 等待模式

| 用途 | 行 | 路径 |
|---|---|---|
| 存在性守卫 | 28 | `runs/v384-cycle3` |
| mkdir | 30 | `runs/v384-cycle3` |
| lease | 32 | `runs/v384-cycle3/lease.json` |
| `--out` | 40 | `runs/v384-cycle3/c3-s$seed` |
| stdout/stderr 日志 | 40 | `runs/v384-cycle3/c3-s$seed.log` |
| 等待模式 | 44 | `pgrep -f "runs/v384-cycle3/c3"` |
| 分析读取（S） | 51 | `runs/v384-cycle3/c3-s{s}/final.json` |
| 分析读取（对照 R） | 51 | `runs/v383-cont/cont-s{s}/final.json` |
| 热启动守卫 / `--init-from` | 27 / 40 | `runs/v383-cont/cont-s$seed/checkpoints/final.pt` |

全部一致。`pgrep -f "runs/v384-cycle3/c3"` 匹配两条训练命令行中的 `--out runs/v384-cycle3/c3-s3421xx`；`>` 重定向不在 argv 中，不会产生额外匹配；R 批（`runs/v383-cont/...`）、Q 批（`runs/v382-long/...`）均不匹配。唯一会误匹配的是用户自己运行的、命令行含该字符串的进程（例如 `tail -f runs/v384-cycle3/c3-s342101.log`），后果只是分析被推迟到该进程退出，不会丢失结果（见建议项 S2）。训练器 463 行 `mkdir(exist_ok=False)`，与第 28 行守卫双重保证不覆盖历史。

### 3. 守卫在干净机器上是否会误中止

- 第 25–26 行 sha 守卫：本地计算的 `train_v378.py` sha256 与硬编码值一致；远端若不一致则中止是正确行为。`$(sha256sum ...)` 子进程在第 29 行执行前已退出，不会被 pgrep 命中。
- 第 27 行热启动存在性：`-f` 测试，无副作用。
- 第 28 行目录不存在：首次运行时目录不存在，正常通过。
- 第 29 行 `pgrep -f "train_v378.py"`：链脚本自身的命令行是 `bash scripts/chain_day4_S.sh`，不含 `train_v378.py`；`pgrep` 自身被排除；`log` 函数的 `tee` 子进程命令行是 `tee -a runs/chain_day3.log`，也不含。**在干净机器上不会误中止。** 会误中止的只有用户自己打开的、命令行含该文件名的进程（如 `less scripts/train_v378.py`、`vim`、`grep train_v378.py ...`），误中止方向是安全侧（exit 2，除 `chain_day3.log` 一行外无任何写入）。
- 所有守卫均在 `cd` 之后、`mkdir` 与 lease 写入之前执行，中止时不会留下半成品目录。

### 4. 内嵌 Python 读取的键 vs 训练器实际写出的键

训练器 709 行：`final = dict(arm=..., updates=updates, ...)` → `a.get("updates")` 存在，对应 S 52 行。
训练器 713 行：`final['val'] = ev_val['summary']`；642 行 `summary[str(k)] = dict(all={...}, strata=...)`，`k ∈ HOPS = (0,1,2,4,8)`；626 行聚合键包含 `'wall_f1'` 与 `'unknown_known_fpr'`。因此 `a["val"]["8"]["all"]["wall_f1"]` 与 `a["val"]["8"]["all"]["unknown_known_fpr"]`（S 53–54 行）均存在。`"8"` 是字符串键（`str(k)`），与脚本一致。对照 R 的 `final.json` 由同一训练器同一版本写出，键相同。

INCONCLUSIVE 路径：
- `updates != 57600` → 52 行 `print("INCONCLUSIVE"); raise SystemExit` → 55 行 `except SystemExit: raise` 重新抛出 → 解释器以状态 0 退出，57–59 行**不执行**。`RES` 捕获到 `INCONCLUSIVE`，62 行写入日志。正确。
- 任何其他异常（文件缺失、`val` 键缺失——训练器 711 行超过 `deadline_total` 时不写 `val`、指标为 `None` 时的 `TypeError`）→ 56 行 `print("INCONCLUSIVE", e); raise SystemExit` → 同上。正确。
- 第二个种子失败时第一个种子的增益已入 `g`，但因提前退出不会计算判定。正确。
- heredoc 使用 `<<'PY'`（带引号），不发生 shell 展开。正确。

### 5. 预注册规则与代码一致性

| 头部规则 | 代码 | 一致 |
|---|---|---|
| 第 10 行：两个种子都 ≥ R + 1 pp ⇒ more training helps | 57 行 `min(g) >= 1` | 是 |
| 第 11 行：两个种子都 \|S−R\| < 1 pp ⇒ converged | 57 行 `max(abs(x) for x in g) < 1` | 是 |
| 第 12 行：其余 ⇒ ambiguous | 57 行 else 分支 | 是 |
| 第 13 行：FPR 上升 > 0.3 pp 为次要保护规则，不决定判定 | 58 行 `if max(fpr) > 0.3: v += " BUT FPR protection violated"`，只追加标注 | 是 |
| 第 14 行：updates ≠ 57,600 ⇒ INCONCLUSIVE | 52 行 | 是 |

增益单位：53–54 行 `100 * (a - c)`，F1 与 FPR 均在 [0,1]，乘 100 即 pp；阈值 1 与 0.3 单位一致。lease 第 36 行 `expectation` 文本与之相同。`--expect-updates 57600` 由训练器 531 行断言 `epochs × len(train) = 4 × 14,400`，与 `updates_each` 一致。

### 6. Lease 算术

第 31 行：`DE = START + 6600`，`DT = START + 7150`，`inclusive_seconds: 7200`，与 R 第 18 行完全相同。R 在相同参数、相同双任务并行下耗时 5,331 / 5,378 s（头部第 15 行声明，**无法本地核实**，`runs/` 不在本地）。按该数据：训练截止余量 ≈ 1,220 s（≈ 23 %），总截止余量 ≈ 1,770 s；最后写 `final.json` 距 7,200 s 边界 50 s。S 与 R 的热启动点不同不会改变每步耗时。风险点仅有两个，均已被训练器处理成 INCONCLUSIVE 而非错误结论：(i) GPU 被外部任务共享导致变慢——第 29 行守卫只检查本项目训练器，不检查其他 GPU 用户，头部第 16 行依赖手工检查；(ii) 若训练恰好在 6,600 s 被截断，`final-val` + `final-fresh` 评估只剩 550 s（711、716 行各自检查 `deadline_total`），评估耗时无本地数据可核。两种情况下 `updates != 57600` 或 `val` 缺失都会触发 INCONCLUSIVE。可接受。

### 7. 写入范围与历史覆盖

链脚本写入：`runs/chain_day3.log`（追加，`tee -a`）、`runs/v384-cycle3/lease.json`、`runs/v384-cycle3/c3-s*.log`；全部相对于第 17 行 `cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021`。训练器写入：全部在 `out`（`--out`）之下——463–465 行 mkdir，578 行 `run.json`，580/583 行检查点，632 行 npz 面板，695/703/714/719/724/726 行 json；`--train-cache` 只读（524 行 `np.load`）。没有 `rm`/`shutil`/`unlink`。第 28 行守卫 + 训练器 `exist_ok=False` 保证不覆盖已有目录；热启动文件只被 `torch.load` 读取（557 行）。`chain_day3.log` 为追加写。**无任何 `/mnt/datasets` 之外的写入，无删除或覆盖历史。**

## 必改项

无。

## 建议项

- **S1（强烈建议，一行）**：第 34 行把两个热启动文件的 sha256 作为硬编码事实写进 lease，但脚本没有核对它。建议在第 27 行的循环里顺带核对：`[ "$(sha256sum runs/v383-cont/cont-s$seed/checkpoints/final.pt | cut -d' ' -f1)" = "<对应值>" ] || { log "ABORT: warm start sha mismatch for $seed"; exit 2; }`。这样预注册记录与实际加载的文件保证一致（训练器 562 行会把真实 `init_file_sha256` 写入 `run.json`，事后也可交叉核对，所以即便不改也可追溯）。
- **S2**：第 44 行等待模式可收窄为 `pgrep -f "train_v378.py.*runs/v384-cycle3/c3"`，避免用户 `tail -f` 日志时把分析无限推迟；第 29 行守卫同理可收窄为 `pgrep -f "python.*train_v378.py"`，避免编辑器/`grep` 进程触发误中止。两者均非正确性问题。
- **S3**：第 29 行只排除本项目训练器，不检查 GPU 上的其他进程；若希望守卫覆盖头部第 16 行的手工检查，可加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "ABORT: GPU busy"; exit 2; }`。
- **S4（记录性）**：S 不再像 R 那样排队等待前序批次（差异 1(a)）。若后续批次沿用 S 的模板又需要排队，须把等待循环加回来。

## 结论

**LAUNCH OK。** 与 R 的差异严格限于热启动路径、输出目录、四条守卫、lease 新增键、FPR 行，以及日志/注释文字和 R 中"排队等待"改为"拒绝启动"这一无害语义变化。所有路径一致；守卫在干净机器上不会误中止；分析读取的键均由 `train_v378.py` 写出；INCONCLUSIVE 路径在计算判定前退出；预注册阈值（1 pp、0.3 pp）与代码一致；lease 算术与 R 相同且余量约 23 %；所有写入限于 `/mnt/datasets/.../runs/v384-cycle3` 与追加日志，不覆盖历史。S1 建议在启动前加上，但不构成阻塞。

---

## 审查后的修改（主代理，2026-10-23 13:13 UTC，发射前）

被审版本 sha256 `9e7ed925…`；发射版本 sha256 **`9d9d69a7e45d43b3019d10af928a823fe8e44c11dc4c28809412621840458ac0`**（远端 `scripts/chain_day4_S.sh` 核对一致）。只做了两处改动，均为审查建议项：
1. **S1 采纳**：守卫里增加两条热启动权重的 sha256 核对（与 lease 中记录的 `27bc72e7…` / `da0f00f5…` 相同），不一致则中止。
2. **S2 采纳（被实际事故证实）**：13:12:17 UTC 第一次发射时，守卫 `pgrep -f "train_v378.py"` 匹配到了发射用的 ssh 外壳命令行（其中含有状态检查的同名文字），脚本在 mkdir 之前正确中止（`chain_day3.log` 有 ABORT 记录，未写任何输出）。两处 pgrep 模式改为锚定真实训练器 argv：`^$P scripts/train_v378.py `（守卫）与 `^$P scripts/train_v378.py .*runs/v384-cycle3/c3`（等待循环）。
第二次发射 13:14:16 UTC 成功，pid 483616 / 483620。
