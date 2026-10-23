# Claude Code entry point

本目录的当前交接任务是 **64 维 Sheaf-GBP 高保真多智能体地图融合研究**，不是根 README 下方保留的 PianoMovers 操作任务。

先完整阅读 [claude_handoff/README.md](claude_handoff/README.md) 和其 [artifact manifest](claude_handoff/artifact_manifest.json)。用户要求使用当前可用的最多并行 agents，独立研究 encoder、decoder、restriction map、目标能量、推断和数据；不能把最新版本当成最好版本。

SSH：`uk-linux`，备用明确端点 `yuzhou@100.69.198.56`；只允许指定 RTX5090 UUID `GPU-238bdbd5-9f06-da6d-09f6-830fd831e817`。不含凭据，启动前重新检查占用与磁盘；不碰他人任务、4080或bigboy。

本交接本身没有启动训练。所有旧运行保持停止，不得重启已过期 lease。v310 E12 是有保留条件的质量参考，v340失败，v341只是未实现设计；没有已确认的全范围高精度/100×突破。保留原demo和所有历史结果。新工作遵守交接说明中的瓶颈、真实证据监督、数据分割、运行预算和验收约束。
