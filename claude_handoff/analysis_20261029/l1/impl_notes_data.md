# L1 批次（D8-82）数据侧与判决侧实现笔记（Claude，2026-10-29）

依据：`claude_handoff/DESIGN_LLM_LATENT_SHEAF_GBP.md` 第 2、4、5、6 节；`OVERNIGHT_REPORT_20261021.md` D8-82。
预注册数字（分片 / 划分种子 342101；TRAIN 20,000 / VAL 1,000 / fresh 1,000；P0 15 pt 与 5 × |ΔB0|；P1 +2.0；P2 +1.0；P3 +1.0；
降级 1.0 pt；训练种子 342101 / 342202）全部写死为模块常量，未改动。未提交、未推送、未碰 GPU、未动 `l1_fusion.py`。

## 文件（均在 `scripts/claude_v342/`）

| 文件 | 作用 | sha256（前 16 位） |
|---|---|---|
| `l1_data.py` | HotpotQA 读取、分片、划分、提示模板、EM / F1（纯 python，无 torch） | `41d0db811e710579` |
| `l1_aggregate.py` | D8-82 判决逻辑（P0–P3、LEVER / NO / AMBIGUOUS / INCOMPLETE、降级、同模型对诊断）；`summary_schema()` 可被模型侧 import | `e9c3ba51c5b24b0b` |
| `test_l1_data.py` | 数据侧单测（tmp_path 合成 HotpotQA 格式夹具：34 train / 14 dev，含 9 段、单支撑标题、支撑标题缺失、bridge / comparison、yes-no、答案在两段、答案在第二段） | — |
| `test_l1_aggregate.py` | 判决侧单测（合成 summary.json 树） | — |
| `run_l1_data.sh` | Linux 盒子上的数据准备（下载 → 镜像回退 → sha / 大小 → `l1_data.py` 默认大小），本机不运行 | `4c559ecaf68b0304` |

行尾均为 LF（与 `l1_fusion.py` 等同目录文件一致；`.sh` 若为 CRLF 在 Linux 上会炸，单测断言了这一点）。

## API（名字与签名按协调者给定，模型侧按此调用）

`l1_data.py`
- `load_hotpot_json(path) -> list[dict]`
- `filter_reason(ex) -> str | None`（`n_paragraphs` / `duplicate_titles` / `n_sup_titles` / `sup_title_missing`）
- `shard_example(ex, rng) -> dict | None`，键：`id, question, answer, type, level, all_paragraphs, shards, gold_title, answer_owner, sup`
- `make_splits(train_json, dev_json, out_dir, seed=342101, n_train=20000, n_val=1000, n_fresh=1000) -> meta`；写 `train.jsonl / val.jsonl / fresh.jsonl / splits_meta.json`
- `read_jsonl(path)`、`write_jsonl(path, rows)`
- `paragraphs_text(paragraphs)`、`latent_prompt(ex, agent) -> (prefix, suffix)`、`full_prompt(ex)`、`no_comm_prompt(ex, agent)`、
  `text_sender_prompt(ex, agent)`、`text_sender_target(ex, agent)`、`text_receiver_prompt(ex, agent, message)`、`answer_target(ex)`
- `normalize_answer(s)`、`exact_match(pred, gold) -> 0/1`、`f1_score(pred, gold) -> float`（hotpot_evaluate_v1 规则，含 yes/no/noanswer 零分规则）
- CLI：`python l1_data.py --train-json P --dev-json P --out DIR [--seed --n-train --n-val --n-fresh]`，把 meta 打印为 JSON
- 常量：`SEED, N_TRAIN, N_VAL, N_FRESH, N_PARAGRAPHS, N_GOLD, SHARD_SIZE, SPLITS`

`l1_aggregate.py`
- 常量：`ARMS, PAIRS, SEEDS, MODELS, LATENT_ARMS, GBP_ARMS, REQUIRED_ARMS, STAGE0_ARMS, P0_MIN_GAIN, P0_SEED_FACTOR, CRITERIA, DEGRADE_PT`
- `run_dir_name(arm, pair, seed) -> "<arm>-<pair>-s<seed>"`（模型侧建目录请用它）
- `summary_schema() -> dict`、`validate_summary(summary, arm, pair, seed) -> list[str]`（空 = 合法；模型侧写完可自检）
- `load_runs(runs_dir)`、`p0_block`、`criterion_block`、`degrade_block`、`same_pair_block`、`decide`、`aggregate(runs_dir, stage0=False) -> dict`、`to_markdown(agg)`
- CLI：`python l1_aggregate.py --runs DIR [--stage0]` → `DIR/aggregate.json`、`DIR/aggregate.md`；打印 `P0: ...` 与（非 stage0 时）`VERDICT: ...`；
  INCOMPLETE 原因逐行打印为 `INCOMPLETE: ...`

summary.json 模式（模型侧必须写，EM / F1 一律**百分点 0–100**）：
```
{"arm", "pair", "seed", "val": {"em", "f1", "em_agent": [a0, a1]（B4: [a]）, "em_model": {"Qwen/Qwen3-1.7B": em, "meta-llama/Llama-3.2-1B-Instruct": em}},
 "fresh": {同上}, "c_assert": true/false（S1、S1L1 必须有；U 可选）, "scripts_sha": {"<script>": "<sha256>"}}
```

## 决定（设计稿 / 简报未写死、由本实现写死的部分）

数据侧（`l1_data.py`）
1. 额外过滤：10 段标题必须两两不同（`duplicate_titles`），否则"金标段"不唯一；单独计数在 `splits_meta.json.counts.filtered_out`。
2. 掷硬币的对象：两段金标按上下文顺序取，`rng.randrange(2) == 1` 则对调（第二段进 shard 0）。rng 调用顺序固定：掷硬币 → 打乱 8 段干扰 → shard 0 内打乱 → shard 1 内打乱。
3. `answer_owner` 的子串检验在 `' '.join(sentences)` 上做（与 `paragraphs_text` 同一种拼接），答案先 `.strip().lower()`。
4. `sup` 句子原文保留（HotpotQA 句首空格不去）；`supporting_facts` 重复项去重、越界索引跳过、按句号序排列；空则 `text_sender_target` 给 `" (none)"`。
5. `sealed_test_count` = dev 可用题数 − n_val − n_fresh；被过滤掉的 dev 题不算封存 TEST，另计。
6. `make_splits` 的 rng 顺序：分片全部 train → 分片全部 dev → `rng.sample(train, n_train)` → `rng.sample(dev, n_val + n_fresh)`。
7. jsonl 用 `ensure_ascii=False`、LF、键保持插入序；meta 不含时间戳，另记输入文件 sha256 与脚本自身 sha256 → Windows / Linux 逐字节一致（单测比对两次运行字节）。
8. n_train 或 n_val + n_fresh 超过可用数、或一份划分内 id 重复 → `ValueError`。
9. `type` / `level` 用 `.get`（缺则 None）。

判决侧（`l1_aggregate.py`）
1. 第 1 阶段判决必需的 run：hetero × {B0, B4, T, U, S1, S1L1} × 两种子（12 个）。T0 与整个同模型对只报告，缺了不算 INCOMPLETE。
2. 完整聚合里 P0 不过 → INCOMPLETE（原因 "P0 gate failed (stage 1 was not supposed to run)"；设计稿：不过就停）。简报的判决规则没提 P0，这是我加的最严格读法，可商榷。
3. `--stage0` 除 PASS / FAIL 外还会打印 `P0: INCOMPLETE`（缺 B0 / B4 run、缺 em_model 键、stage-0 run 间 sha 冲突）。stage0 写的也是 `aggregate.json / aggregate.md`（内含 `"stage": 0`），之后的完整运行覆盖之。
4. `c_assert`：S1、S1L1 必须有（缺 → INCOMPLETE）；U 可选；任何 run（含同模型对）为 false → INCOMPLETE。
5. `scripts_sha` 一致性在读到的所有 run 上检查（含 T0、同模型对）；任一脚本 sha 不一致 → INCOMPLETE，并列出哪些 run 持哪个 sha。
6. summary 的 arm / pair / seed 与目录名不符 → INCOMPLETE。
7. 差值先四舍五入到 6 位小数再比门槛（42.3 − 40.3 在二进制里是 1.9999999999999964，按 2.0 算过）。
8. 降级是严格"低 1.0 pt 以上"（差 1.0 不触发）；只把 LEVER 压到 AMBIGUOUS，NO 不变。
9. 单位保护：读到的所有 EM 都 ≤ 1.0 → INCOMPLETE（怀疑写成了比例而非百分点）。`validate_summary` 要求 em / f1 ∈ [0, 100]。
10. P0 读 B0 / B4 hetero 的 `val.em_model[model]`；P1–P3 读 `val.em`（两智能体平均）。
11. 完整模式也打印 `P0: ...` 行（在 `VERDICT:` 之前）。

启动脚本（`run_l1_data.sh`）
1. 协调者 2026-10-29 18:34 UTC 的改动：先 curl 官方 `http://curtis.ml.cmu.edu/datasets/hotpot`；若任一文件仍缺失或为空 → 运行 `l1_hotpot_from_hf.py --out "$DATA_DIR"`，
   `HF_HUB_CACHE` / `HF_DATASETS_CACHE` 设在 `$HF_CACHE_ROOT/{hub,datasets}`（默认 `/mnt/datasets/yuzhou-sheaf-gbp-research/hf-cache`）；打印 `source: existing | official | hf-mirror`
   与 `hf_source.json` 内容。转换器未改动。
2. 下载写到 `<name>.part`，完整且非空才改名；失败则删 `.part`。
3. 不覆盖：`$DATA_DIR/splits_l1/splits_meta.json` 已存在 → 打印之并 exit 0（可重复运行）。
4. 环境变量（默认值即正式运行）：`DATA_DIR`、`HF_CACHE_ROOT`、`PYTHON`（默认 `python3`；只有走镜像回退时才需要装了 `datasets` 的解释器）、`CURL`、`HF_CONVERTER`、
   `L1_DATA_EXTRA_ARGS`（**只给单测用**，正式运行必须为空；静态单测断言脚本本身不传 `--n-train / --n-val / --n-fresh / --seed`）。
5. 只写 `$DATA_DIR` 与 `$HF_CACHE_ROOT`；单测用 Git Bash 真跑了两条路径（文件已存在 → 不下载 → 拒绝覆盖；curl 失败 → 桩转换器 → 继续）。

## 测试

```
python -B -m pytest scripts/claude_v342/test_l1_data.py scripts/claude_v342/test_l1_aggregate.py -q
33 passed in 6.82s
```
解释器：系统 python 3.12.4（Anaconda）+ pytest 7.4.4（`C:\Users\27118\AppData\Local\Temp\claude\venv-l1` 当时还没装 pytest）。CPU，无网络，无原始数据；
夹具只在 tmp_path 生成，仓库里没有任何 jsonl / 夹具。

## 待模型侧核对的接口点

- `latent_prompt` 返回 `(prefix, suffix)`，软 token 插在两者之间；`no_comm_prompt = prefix + suffix`。
- `paragraphs_text` 用 `' '.join(sentences)`，HotpotQA 句首自带空格会出现双空格（按简报原样实现，未改）。
- summary.json 的 EM / F1 必须是百分点；目录名请用 `run_dir_name()`；写完可用 `validate_summary()` 自检。
