# 融合系统改进：公开证据

说明见[改进报告](../../fusion-improvements-2026-10-18.md)。

`calibration/` 保存比较前冻结的计划、三个对照的代码、评价器、协议和全部逐条件结果。`calibration_plan.json` 中的哈希可核对每个公开输入。原始 `search.npz` 不复制到 Git；其 SHA-256 已记录，可按[案例说明](../../../examples/pathmnist_mil/README.md)构建。

校准只使用开发训练与验证 bags。旧留出集未打开。这里的均值和 seed 变化用于训练预算诊断，不是独立确认、新颖性证据或系统间性能排名。历史首次案例的证据保持不变。

计划中使用相对路径。公开材料不含 API key、原始图像、论文 PDF 或完整 agent 对话。

`development_run/` 保存六轮真实搜索的基线与候选源码、指标、训练轨迹、协议、逐轮预算和搜索树摘要。[summary.json](development_run/summary.json)记录最终选择及逐 seed 分数。所有候选按原始宿主回执验证后导出；只读源码审计核对了实现与 idea 的一致性。

公开 `metrics.json` 只将 `protocol_source` 的本机绝对路径改为 `task/evaluation_protocol.json`，所有数值不变。各程序的 `receipt_summary.json` 区分原始结果哈希与公开指标哈希，并记录该转换。搜索树摘要移除本机程序目录。idea 只导出摘要与原始完整文件哈希，不重新分发论文引文和完整对话。回执摘要是可检查的开发记录，不是强隔离或独立第三方认证。

[SHA256.json](SHA256.json)列出公开包全部文件的 SHA-256（不含哈希清单自身）。最终候选为第 5 轮，分数 0.58993；第 6 轮 0.58673 未替换它。这是复用开发验证 bags 的搜索结果，未打开旧留出数据。
