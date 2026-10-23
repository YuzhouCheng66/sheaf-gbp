VERDICT: LAUNCH OK

# 发射前代码审查：v391 宽解码器第 2、3 周期（batch W，`chain_day4_W.sh`）

审查者：独立 Claude 子代理，2026-10-23。只读审查，未修改任何被审文件。本地无 GPU、无 `runs/` 目录：远端的批次 T 检查点、v383 / v384 对照文件、W=2 的验证耗时均无法在本地核实，凡属此类在下文标明"无法本地核实"。

## 范围

| 文件 | 行数 | sha256 | 备注 |
|---|---|---|---|
| `scripts/claude_v342/chain_day4_W.sh`（被审，新） | 82 | `e34e33f8ff464be49b7269f2c7418d4b63a76ae36e1f7f429d5a85d15e4e44e8` | LF 行尾，无 CR |
| `scripts/claude_v342/chain_day4_R.sh`（窄解码器第 2 周期，已成功） | 46 | `f83e477f9ed49af9e596d8603eb1aaf6076c297e8a8e227299912ef54a98f768` | = S 审查记录值 |
| `scripts/claude_v342/chain_day4_S.sh`（窄解码器第 3 周期，已成功） | 66 | `9d9d69a7e45d43b3019d10af928a823fe8e44c11dc4c28809412621840458ac0` | = 发射版本 |
| `scripts/claude_v342/chain_day4_T.sh`（守卫 / 租约 / 锚定 pgrep 惯例） | 109 | `27d9f0d58a9571edb17122d964dc36edf8f6a27c0fdb7e26e12896ebf8dccd92` | = 发射版本 |
| `scripts/claude_v342/chain_day4_U.sh`（W 要等待的批次） | 113 | `12797618c6b04d8b5760bd648aaafd648fb53673482ad29337abe34db469f8ca` | = 发射版本 |
| `scripts/claude_v342/train_v386.py`（训练器） | 743 | `77e8bc73895bd9765765a9afebc6fe701603e826b99a83c30da3b2b7dd44a068` | 本地计算值 = W 第 26 行硬编码值 = T 发射版本 |
| `OVERNIGHT_REPORT_20261021.md` D5-2 / D5-5 / D5-6，`HANDOFF_20261023.md` §2，`code_review_v386_decoder.md`，`code_review_v384_cycle3.md`，`code_review_v388_embed.md` 的落实记录 | | | 只读参考 |

方法：把 W 的训练命令行（第 37、48 行）逐参数对回 R 第 26 行 / S 第 44 行；对照训练器的 `--init-from`（567–580 行）、meta（585–591 行）、截止线（666、702、716–737 行）、`final.json` 键（720–738 行）、信号处理（39–44、452 行）；逐行走查 bash 的守卫、等待循环、`run_cycle` 的各条路径与内嵌 Python 的各条异常路径；用 D5-5 的实测耗时核算租约余量；在本地核对各文件 sha256 与行尾。

## 逐项核对

### 1. 训练命令行与热启动是否就是 R/S 的配方

**命令行**：W 第 37 行 `COMMON` + 第 48 行逐参数展开为
`--source $S --world houseexpo_fit --train-cache $C/houseexpo_fit-k4800 --arm owned --size 128 --device cuda --epochs 4 --lr 1e-4 --val-every 1 --refine-steps 0 --readout posterior_only --fusion max --expect-updates 57600 --decoder-width 2.0 --out <out> --seed <seed> --init-from <warm> --deadline-epoch <LDE> --deadline-total <LDT>`。
与 R 第 26 行 / S 第 44 行相比，除 `--decoder-width 2.0` 与路径类参数（`--out`、`--init-from`、截止线）外**逐字相同**（lr 1e-4、epochs 4、同缓存 `houseexpo_fit-k4800`、同 world、`--fusion max`、`--readout posterior_only`、`--val-every 1`、`--refine-steps 0`、`--expect-updates 57600`、同 `$S` / `$C` / 环境变量，第 19–21 行 = R 10–12 行 = S 18–20 行）。训练器文件从 `train_v378.py` 换成 `train_v386.py`，而 v386 在 W=1 下与 v378 逐字节等价（`test_v386_width.py`、CR386 §1），所以"除宽度外全部等于 R/S"成立。

**热启动能否全量加载、meta 断言能否通过**（训练器 567–580 行）：
- 第 453 行先把 CLI 的 `--decoder-width 2.0` 写入全局 `DECODER_WIDTH`，第 565 行才构造 `SheafModel`，`OwnedDecoder.__init__`（197 行）据此得到与批次 T 完全相同的通道数 ⇒ 两边 `state_dict` 的键集与每个张量的形状一一相同 ⇒ 第 570 行 `loaded` 覆盖全部键，`missing = []`、`skipped = []`，第 578–579 行的"禁止部分热启动"断言**通过**，且链脚本**没有**传 `--allow-partial-init`（正确：传了会削弱守卫）。
- 第 575 行 `arm`：T 的 meta 为 `owned`（T 第 42 行 `--arm owned`）= CLI ✓。第 576–577 行六个键：`readout_mode = posterior_only`（两边 CLI 都传）✓；`precision_mode = learned`（两边都用默认值，433 行）✓；`mu_bits / q_bits = 8`（默认，437–438 行）✓；`wire = sealed`（默认，439 行）✓；`decoder_width = 2.0`（T `COMMON` 第 42 行传了 2.0，meta 第 588 行记录为浮点 2.0；W 传 2.0）✓。T 的 `final.pt` 由第 594–597 行 `save()` 写出，带 `meta`。
- 第 3 周期的热启动是 W 自己的第 2 周期 `final.pt`，同一训练器、同一 CLI，meta 必然一致 ✓。
- 第 569 行只取 `payload['state_dict']`；AdamW（582 行）与 `LambdaLR`（584 行：200 步 warm-up + 对 57,600 步的余弦）重新开始，与 S 头部第 6–7 行声明的"权重热启动、优化器与日程重启"完全一致 ✓。
- `--expect-updates 57600` 的断言（541–542 行）只看 `epochs × len(train) = 4 × 14,400`，与热启动无关 ✓。

**结论：命令行 = R/S 配方 + 宽度旋钮；width-2 → width-2 的热启动全量加载、六项 meta 断言全部通过，不需要也没有使用 `--allow-partial-init`。**

### 2. 守卫、等待循环、pgrep

| 守卫 | 行 | 判断 |
|---|---|---|
| 训练器 sha | 25–26 | 硬编码 `77e8bc73…` = T 发射版本 = 本地文件实际哈希 ✓ |
| 对照 / 热启动存在 | 27–28 | 每个种子检查 T 的 `checkpoints/final.pt` 与 `final.json`、v383 `final.json`、v384 `final.json` 四个文件 ✓ |
| 批次 T 完整性 | 29 | `final.json['updates'] == 57600`，否则 ABORT ✓（v383 / v384 的完整性未重查，但 R / S 的分析已各自以同一规则判定过，D5-0 / D5-1 记录为完整；见 S4） |
| 永不覆盖 | 31 | `ls -d runs/v391-wide-cycles/c2-s*` 存在即 ABORT；训练器第 474 行 `mkdir(exist_ok=False)` 双重保证；`c3-s*` 不可能先于 `c2-s*` 存在 ✓。只有"上一次尝试在写完 lease 后、训练器 mkdir 之前就死掉"（例如 import 失败）这种情形会让 `mkdir -p`（36 行）+ `cat > lease_*.json`（41 行）覆盖旧 lease，概率很低（S3） |
| 等待 U | 33 | `pgrep -f "chain_day4_U.sh"` **或** `pgrep -f "^$P scripts/train_v388.py "`，每 120 s 轮询。见下文 |
| 无训练器 | 34 | `^$P scripts/train_v3` 锚定真实 argv，能拦住任何本项目训练器（含 T / S 的残留）✓ |

**等待 U 的两个子句为什么都需要、有没有空档**：U 的链脚本有两段没有训练器进程的时间：顺序模式下两个种子之间（U 第 106–108 行，≤ 60 s）以及预检结束到正式发射之间（U 第 56–86 行，`preflight_check_v386.py` 几秒）。这两段里 U 的 **bash 进程仍然活着**，其命令行（`bash scripts/chain_day4_U.sh`，无论相对 / 绝对路径都含该字符串）被第一个子句命中 ⇒ W 继续等待。U 结束后链脚本内再无任何动作（第 111–113 行分析完即退出）。第二个子句是防"U 的 bash 被人工杀掉而训练器还在"的后备：U 的四处训练器调用（54、66、87、103 行）都是 `$P scripts/train_v388.py …`（`nohup` 会 exec，argv[0] 仍是 python 路径）⇒ 锚定模式 `^$P scripts/train_v388.py ` 命中 ✓。**没有竞争空档。**

第一个子句故意不锚定是对的：等待循环里误报（例如用户开着 `less scripts/chain_day4_U.sh` 或 `watch "pgrep -af chain_day4_U"`）只会让 W 推迟，而漏报会导致共用 GPU；两个方向里推迟是安全方向。代价是这类进程会让 W 无限等待（S5）。

**自匹配**：W 自身的 bash 命令行 `bash scripts/chain_day4_W.sh` 不含 `chain_day4_U.sh`；ssh 外壳 `bash -c 'cd … && setsid nohup bash scripts/chain_day4_W.sh …'` 以 `bash` 开头，锚定了 `^$P` 的两条训练器模式都不会命中；`pgrep` 排除自身；`tee -a runs/chain_day3.log`、`$P -c …`、`sha256sum …` 这些子进程的命令行都不含任何模式 ⇒ **不会自匹配，也不会误中止**（批次 S 13:12 UTC 那次误中止的模式已被锚定化）。

**启动顺序的运行约束（非代码问题）**：W 只等 U，不等 T 的链脚本；按 D5-3 / D5-5 的时间线，T 已于 18:17 UTC 结束，U 自 16:09 UTC 起一直活着（先等 T、再预检 / 训练），所以 W 发射时第 33 行会正常进入等待。反过来，如果 U 的链脚本在 W 之后才启动，U 第 37 行只认 `train_v386.py` 进程、每 120 s 轮询，有可能恰好落进 W 两个周期之间约 1–2 s 的空档而开始预检 ⇒ W 的下一周期被挤慢、INCONCLUSIVE（安全方向）。**只要保持"U 先于 W 启动"这一现状即可**，本审查把它记为运行约束而非必改。

### 3. `run_cycle` 函数（38–53 行）

- **局部变量**：第 39 行 `local seed cyc warm ctrl out`、第 40 行 `local LS LDE LDT`、第 49 行 `local upd` 全部局部；使用的全局只有 `P`、`COMMON`、`TSHA`、`log`，均为设计意图 ✓。`out=runs/v391-wide-cycles/c$2-s$1`、`$out.log`、`lease_c${cyc}_s${seed}.json` 的展开都正确（`-`、`.` 不是变量名字符）。
- **lease 先于发射**：第 41–46 行写 lease，第 48 行才发射 ✓。`warm_start_sha256` 由未加引号的 heredoc 内的命令替换 `$(sha256sum $warm | cut -d' ' -f1)` 在写入时计算（第 2 周期的 `$warm` 已由第 28 行保证存在；第 3 周期的 `$warm` 只在第 2 周期返回 0 后才会用到，而 `final.pt` 在训练器第 721 行、早于 `final.json` 写出）✓。JSON 合法性：所有键加引号，`1e-4` 与 `2.0` 是合法 JSON 数字，`"warm_start_sha256"`、`"control"`、`"expectation"` 都是加引号的字符串，`$seed` / `$cyc` 是整数 ⇒ 合法（U 审查 M1 的问题没有重现）。
- **前台运行训练器**（第 48 行，无 `nohup` / `&`）：链脚本以 `setsid nohup bash scripts/chain_day4_W.sh` 启动，bash 已无控制终端（setsid）且 SIGHUP 为忽略（nohup）；训练器经 exec 继承 SIGHUP 忽略（Python 只重设 SIGINT / SIGPIPE，第 452 行另装 SIGTERM / SIGINT → `STOP`），所以 ssh 断开影响不到它。bash 在 `wait` 子进程上阻塞，**不需要** T/U 那种 `pgrep` 轮询 ⇒ 少一处可能误匹配的模式，且周期之间的切换只有 1–2 s；链脚本 pid 全程不变，后续批次可以像 W 等 U 那样 `pgrep -f chain_day4_W.sh` 等它。两种干预场景：`pkill -f chain_day4_W` 只杀 bash，训练器成为孤儿按自身截止线跑完（数据不丢，但第 3 周期不会再发射）；`pkill -f train_v386` → `STOP` → 第 666 行在下一更新边界退出 → 轮末验证 → `final.json` 的 `updates < 57600` → 链脚本判 INCONCLUSIVE ✓。**90 分钟前台运行在这种启动方式下没有问题。**
- **完成判定**（第 49–51 行）：读 `$out/final.json` 的 `updates`；文件缺失（训练器崩溃）时 `2>/dev/null` 使 `upd` 为空，键缺失时为 `-1`，两者都 ≠ `57600` ⇒ `return 1` ✓。
- **`return 1` + `|| continue`**（第 55–56 行）：种子 A 第 2 周期未完成 ⇒ 第 55 行 `continue` 跳过 A 的第 3 周期，直接开始种子 B 的第 2 周期 ✓；第 3 周期未完成 ⇒ 第 56 行 `continue` 进入下一种子 ✓；`||` 列表也使返回 1 不会触发任何 errexit（脚本没有 `set -e`）。行为与 D5-6"该种子 INCONCLUSIVE、不再续跑"一致。头部第 16 行写的"makes the chain stop"措辞不准确（实际是"该种子停止、另一种子照跑"），见 S2。

### 4. 截止线算术

- 第 40 行：每个运行在**函数入口**取 `LS`，`LDE = LS + 6600`、`LDT = LS + 7150`、`inclusive_seconds 7200`，与 R 18 / S 35 / T 91 / U 95 行相同，满足 HANDOFF §2 第 25 行"每批训练是一个包含式 7,200 s 租约"；四个运行各一个 lease 文件，严格顺序（前台），与 D5-6 和 T 的顺序模式先例（`lease.json` / `lease2.json`）一致。`LS` 取在 `sha256sum` 热启动文件（约 1 s）之前，租约从保守一侧计。
- **余量**：批次 T 顺序模式实测 5,256 / 4,973 s（D5-5；这是 `final.json.seconds` = 第 720 行 `time.time() − t0`，`t0` 在第 480 行、数据加载之前 ⇒ 含加载 + 4 轮 + 4 次轮末验证 + 检查点，不含两次最终评估）。热启动只多一次 `torch.load` + 哈希（约 40 MB，可忽略），速度与 T 相同 ⇒ 训练截止余量 1,344–1,627 s（20–25 %），与 S 的 23 % 同量级。最终评估窗口 550 s：T 的单任务预检已按 `--final-window 550 --mode solo`（T 第 63 行）验证过 W=2 下 `2 × val ≤ 550 s`（CR386 M2 的落实记录）⇒ 正常情况下训练在约 5.3 ks 结束、两次最终评估在约 5.5 ks 完成，远早于 7,150 s。**余量充分**，前提是 GPU 确实只有这一个任务：第 34 行只排除本项目训练器，不检查其他 GPU 用户（S 审查 S3 / CR386 S9 仍未采纳；S5）。
- **到达训练截止时会发生什么**：第 666 行在下一更新边界 `break`；第 702 行 `stopped = True` ⇒ 第 703–711 行照常做轮末验证并可能写 `best.pt`；第 712 行 `latest.pt`；第 717 行跳出轮循环；第 720 行 `final['updates'] < 57600`；第 721 行写 `final.pt`；第 722 / 727 行的两次最终评估各自检查 `deadline_total`；第 737 行**一定**写 `final.json` ⇒ 链脚本第 51 行判 INCONCLUSIVE、不发射该种子的第 3 周期 ✓，与预注册一致。训练器若挂死，链脚本（前台 `wait`）会无限等待，没有外部看门狗，与 T/U 相同（S6）。
- 总时长：4 × 约 5.3–5.5 ks ≈ 6 h，在 U 结束之后开始。

### 5. 内嵌分析（59–82 行）

- 语法：`RES=$($P - <<'PY' … PY )`，heredoc 带引号 ⇒ 无 shell 展开；结束符 `PY` 顶格，`)` 在其后一行；与 R 32–45 / S 50–66 行同一写法 ✓。
- **键**：`final.json` 的 `updates`（训练器 720 行）、`val`（724 行 `= ev_val['summary']`）、`val['8']['all']['wall_f1']` 与 `['unknown_known_fpr']`（`summary` 以 `str(k)` 为键，第 657 行 `fmt` 用的正是这两个键）全部由训练器写出 ✓；对照 v383 / v384 由 `train_v378.py` 写出，`final` / `summary` 结构相同（S 审查 §4 已核对）✓。路径 `runs/v391-wide-cycles/c{cyc}-s{s}/final.json` = 第 39 行 `out` ✓；对照路径 = 第 55–56 行传入的 `ctrl` ✓。
- **阈值**：`min(g) >= 1` → capacity transfers；`max(|g|) < 1` → faster learning only；其余 ambiguous；`max(fpr) > 0.3` 只追加标注、不改判定 —— 与头部第 10–12 行、lease `expectation`、D5-6 一致；单位 pp（×100）一致 ✓。第 79 行先打印 `cycle3`（主判定）再打印 `cycle2 (secondary)` ✓。
- **缺失 / 不完整路径**：某种子被跳过（无 `c3-s<seed>` 目录）⇒ 第 66 行 `open` 抛 `FileNotFoundError` ⇒ 第 70 行 `except` ⇒ `ok = False` ⇒ 第 76 行 `INCONCLUSIVE`，`g` 里保留另一种子的值作信息 ✓；`updates != 57600` ⇒ 第 67 行 `ok = False; continue` ✓；超 `deadline_total` 导致 `val` 缺失 ⇒ `KeyError` ⇒ except ✓；指标为 `None` ⇒ `TypeError` ⇒ except ✓。`out["cycle2"]` / `out["cycle3"]` 在循环里**无条件**赋值，第 79 行不可能 `KeyError` ✓。`analysis.json` 在打印之前写出；若写入失败只丢日志一行，原始数据不受影响。**不会崩溃。**
- 两处可改进但不影响发射：(i) 第 68–69 行先 `round(…, 2)` 再比阈值，等效阈值变成 0.995 / 0.295 pp，R（39 / 42 行）与 S（57–58 / 61–62 行）都是用未取整值判定、只在打印时取整（S1）；(ii) 头部第 12–13 行与 D5-6 承诺附报 K0、fresh、8+ 室分层，内嵌分析只算 R8 与 FPR，附报项需事后从 `final.json` / `final-val.json` 另算（数据都在，S7）。

### 6. 科学效度

"同日程下宽解码器第 3 周期 vs 窄解码器第 3 周期"是一个干净的**种子配对、等日程**检验，判定规则与 D5-2 的"通过"分支、D5-6 的预注册一致。应与结果一并写明的保留：

1. **配对成立的理由**：同种子 ⇒ 编码器 / head 初始权重与数据顺序相同（CR386 §2：`fork_rng` + 独立 `order_rng`）；每个周期都在第 661 行重新播种 `order_rng`，所以两臂的第 2、3 周期重放同样的 4 个排列；宽臂的第 1 周期（T）与 Q 只差宽度（`analyze_v386.py` 身份守卫，D5-5 通过）。配对是真实的。
2. **日程是窄解码器选的**：lr 1e-4、每周期 4 轮、共 3 周期全部来自 R/S 在窄解码器上的结果；宽解码器的最优续训 lr 未必相同（CR386 保留 1 在此再次适用）。因此"faster learning only"的含义是"在这个日程下增益被吸收"，不是"容量不能迁移"；胜出也部分含"该日程恰好更适合宽臂"的成分。
3. **噪声与功效**：运行间噪声 ±1–2 pp（D5-1），门槛 1 pp 处于噪声量级；S 本身的第 3 周期增益只有 +1.0 / +1.3，说明窄前沿仍在移动。"两种子都 ≥ 1"压低假阳性，但一个真实的 +1.5 pp 迁移落入"ambiguous"的概率不小；两种子不能支撑更细的量化陈述。
4. **三分法之外的情形**：两臂在 3 周期时都未饱和、只是斜率不同，会落入"ambiguous"；任何关于渐近上限的说法都不在本检验的支持范围内（CR386 D5-2 保留照旧）。
5. **等更新 ≠ 等算力**与**全局梯度裁剪耦合**（692 行）从 CR386 保留 2–3 延续；三周期把这两点各放大三倍。
6. **记忆 vs 泛化**：12 遍同样的 14,400 套户型、同样的顺序，解码器参数 3.95× ⇒ 记忆近重复户型的能力更强；且 VAL 已被 Q→R→S→T→W 的每一次决策使用（分叉路径）。"新 HouseExpo 最佳"是 VAL 陈述；fresh 面板（索引 8–15，从未被选择）与 8+ 室分层应作为确认性附报，`neardup2_v375.json` 的无近重复切分（CR386 S6）宜一并描述。
7. **采用成本**：判定"adopted"隐含接受解码器 3.95× 参数、约 4× 推理 FLOPs；这是检验之外的工程取舍，应在结论里分开写。
8. **非混杂的差异**：对照 R/S 是双任务并行跑的、W 单任务跑——只影响时钟不影响结果（cuDNN 非确定性除外）；训练器文件 v378 vs v386 在 W=1 下逐字节等价，不构成混杂。

### 7. 写入范围与历史覆盖

- 链脚本：`runs/chain_day3.log`（第 23 行 `tee -a`，追加）；`runs/v391-wide-cycles/`（第 36 行 `mkdir -p`）下的 `lease_c{2,3}_s<seed>.json`（41 行）、`c{2,3}-s<seed>.log`（48 行）、`analysis.json`（78 行）。全部相对于第 18 行 `cd /mnt/datasets/yuzhou-sheaf-gbp-research/claude-diag-20261021`。
- 训练器：全部在 `--out` = `runs/v391-wide-cycles/c{2,3}-s<seed>/` 之下（474–476 行 mkdir，`exist_ok=False`；589 `run.json`；591 / 594–597 检查点；643 行 npz 面板在 `out/validation/`；706 / 714 / 725 / 730 / 735 / 737 行 json）。热启动文件只被 `torch.load`（568 行）与 `sha256sum`（43 行）读取；对照只被 `json.load`（66 行）读取；缓存只读（535 行 `np.load`）。
- 链脚本与训练器均无 `rm` / `mv` / `shutil` / `unlink` / `os.remove`（grep 确认）。副作用仅 `scripts/` 与 `$S/scripts` 下的 `__pycache__`，各批均如此。
- **结论：除 `/mnt/datasets/.../runs/v391-wide-cycles` 与追加日志外无任何写入；不删除、不覆盖历史。** 满足 HANDOFF §2 第 23–24 行。

## 必改项

无。

## 建议项

- **S1（一行，建议发射前顺手改）**：第 68–69 行改为 `g.append(100 * (...))`、`fpr.append(100 * (...))`，只在第 77 行写出 / 打印时 `round(x, 2)`，使判定使用未取整值，与 R/S 及预注册的 1 pp / 0.3 pp 完全一致（现写法等效阈值 0.995 / 0.295 pp）。
- **S2（措辞）**：头部第 16 行"makes the chain stop"改为"stops further cycles for that seed; the other seed still runs"，与第 51 行日志、D5-6 一致。
- **S3**：第 31 行守卫改为 `[ -e runs/v391-wide-cycles ]`（W 没有需要重跑进同一目录的预检阶段），杜绝 lease 文件被覆盖的边角情形；中止后如需重发，改名归档而非删除。
- **S4（溯源）**：若 T 的 `final.pt` sha256 已在远端记录（`runs/v386-decoder/wide-s<seed>/checkpoints/final.pt`），可仿 S 第 29–30 行硬编码核对；现方案（lease 记录 + 训练器 `run.json['warm_start']['init_file_sha256']`）已可事后交叉核对，故不构成必改。分析里可顺带检查对照的 `updates == 57600`。
- **S5**：第 34 行之后加 `nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "ABORT: GPU busy"; exit 2; }`（S 审查 S3 / CR386 S9，仍未采纳）；"宽解码器必须独占 GPU"是本批租约余量的前提，这是唯一能把它变成机器检查的地方。发射后避免在 uk-linux 上长期开着命令行含 `chain_day4_U.sh` 的进程（编辑器、`watch`），否则 W 会一直等待。
- **S6**：前台训练器加看门狗：`timeout -k 60 $((LDT - LS + 300)) $P scripts/train_v386.py …`，训练器自身执行截止线，`timeout` 只防挂死（与 CR386 S4 同义）。
- **S7**：把头部承诺的附报项（K0、fresh R8/K0、8+ 室分层 `rooms3/N8`、`rooms3/N16`，第 2 周期 vs v383）写进分析，或预先登记一个事后脚本；数据全部在 `final.json` / `validation/final-val.json`，不影响发射。
- **S8（预注册文本）**：把第 6 项保留 2–7 写进 D5-6 或链脚本头部（事前写明，事后不能补），特别是"日程取自窄解码器"与"VAL 陈述 vs fresh 确认"两条。

## 结论

**LAUNCH OK。**

训练命令行除 `--decoder-width 2.0` 外与 R/S 逐字相同；width-2 → width-2 的热启动在训练器 567–580 行全量加载（无 fresh / skipped 张量，不需要 `--allow-partial-init`），六项 meta 断言与 `arm` 断言全部通过，优化器与余弦日程按 R/S 的语义重启。守卫（训练器 sha、四个文件的存在、T 完整性、永不覆盖、锚定的训练器 pgrep）在干净机器上不会误中止；等待 U 的循环同时看链脚本与锚定的训练器 argv，覆盖了 U 顺序模式与预检之后的两段空档，不会自匹配发射用的 ssh 外壳。`run_cycle` 的 lease 先于发射、变量局部、前台运行在 `setsid nohup` 下安全、完成判定与 `return 1 || continue` 正确实现"该种子 INCONCLUSIVE、另一种子照跑"。每个运行各自 +6,600 / +7,150 s 的截止线相对 T 实测 4,973–5,256 s 有 20–25 % 余量，最终评估窗口已由 T 的单任务预检验证；到达训练截止时训练器写出 `updates < 57,600` 的 `final.json`，走 INCONCLUSIVE 路径。内嵌分析读取的每个键都由训练器写出，缺失第 3 周期、`val` 缺失、指标为 `None` 都落到 INCONCLUSIVE 而不崩溃。写入限于 `runs/v391-wide-cycles` 与追加日志，不删除历史。

建议在发射前顺手采纳 S1（取整先后，一行）与 S2（措辞）；其余建议项不阻塞。科学上这是一个公平的"同日程"配对检验，但结论的措辞应受第 6 项保留约束——尤其是日程取自窄解码器、1 pp 门槛处于噪声量级、"新最佳"是 VAL 陈述而 fresh 面板才是确认。

---

## 审查后的修改（主代理，2026-10-23 18:32 UTC，发射前）
- S1 采纳：判定改用未舍入的差值（输出时再保留 3 位），阈值不再被四舍五入放宽。
- S2 采纳：头部措辞改为"某周期未跑满 → 该种子停止续跑、另一种子继续"，与代码一致。
- 发射版本 `chain_day4_W.sh` sha256 前 16 位 `936d1b02c56fec8c`，18:32:24 UTC 启动，正在等批次 U（链脚本与训练器）退出后顺序执行 4 个租约。
