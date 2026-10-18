# 方法学接入验证记录

日期：2026-10-18。此记录验证软件接线与证据边界，不评价科研结论或论文水平。

## 真实模型调用

通过实际 `draft_research_agenda` 入口，执行了一次作者模型调用，反思轮数为 1。使用 `codex/gpt-6.1-sol` 与已配置的 native profile。输入为明确标注“软件集成验证”的假设性医疗 AI 问题，没有真实患者数据、实验结果或文献来源。

| 检查项 | 观察结果 |
|---|---|
| 调用结果 | 完成，约 64.7 秒 |
| profile | `medical` |
| 规则版本 | `1` |
| 规则 SHA-256 | `2ee77dfcd20f9b604f6b4ee33024b220cace8173644bc5d7d11eb744bbbb99a9` |
| 输入绑定 | 回执 `frozen_evidence.methodology` 包含同一 profile、版本与哈希 |
| 输出绑定 | 模型返回相同版本、哈希及 `stage: agenda` |
| 方法卡 | 返回问题、范围、替代解释、判别实验、`study_design`、`evidence_plan` 与 `medical_context` |
| 未知事实 | 数据使用依据、受试者隔离、偏倚与人群信息均保留为未知 |
| 结构检查结果 | `needs_evidence`，保留上述三个医学证据缺口 |

工具报告的 token 用量如下。这是完整调用的用量，包含 agent 上下文和输出，不能解释为方法学规则单独增加的成本。

| 工具返回字段 | 数值 |
|---|---:|
| `inputTokens` | 11,935 |
| `cacheReadTokens` | 13,824 |
| `outputTokens` | 3,094 |
| `reasoningOutputTokens` | 921 |
| `totalTokens` | 28,853 |

工具未返回可核验的金额或定价依据。本次不估算费用。原始提示、响应、传输回执和验证 JSON 保留在执行环境的 `scratch/harness-audit/methodology-live-smoke/`；其中的计划不作为已完成科研结果。

## 自动检查

一次 Linux 联合回归完成 **569 passed、2 skipped**。覆盖研究入口、冻结与恢复、实际 shell 提示传递、方法卡与论文材料边界、运行时打包，以及 broker/final evaluation 相关检查。另有 Ruff 与 23 个源文件的 mypy 检查通过。

关键检查包括：

- 新研究在初始议程调用前冻结方法学规则。
- 旧工作区恢复不自动加入规则，也不改写原输入。
- 删除已启用研究的规则或论文投影，不能使检查静默关闭。
- 提示包含对应阶段的规则与哈希；idea 和 coding 不重复注入同一规则块。
- 论文审查材料从宿主冻结记录重新核验；修改声明、规则或论文上下文会保留证据缺口。
- 初始议程的字段完整性，不被标记为最终候选方法或临床有效性认证。

这次真实调用只覆盖议程阶段。其他阶段的提示测试使用模拟 agent 或捕获接口；它们不等于所有 provider 的在线验收。预算匹配的科研质量比较、完整病例流程和论文验收另行记录。

配置与操作见[方法学接入说明](research-methodology.zh-CN.md)。
