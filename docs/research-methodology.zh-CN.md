# 研究方法学如何进入实际模型调用

方法学集成把问题定义、实验设计、贡献判断、最小验证、医学研究边界和证据约束写作接入现有流程。新研究默认启用。系统将规则冻结在工作区，再按阶段加入实际模型提示词。规则会占用输入 token；此功能本身不增加模型调用轮次。

它帮助 agent 明确研究问题、设计可区分实验和报告证据缺口。规则进入提示词、模型返回绑定信息、检查项全部填写，是三个可检查事实。它们都不证明研究结论正确。

## 配置与运行

在项目的 `research.yaml` 顶层设置：

```yaml
schema_version: 1
methodology:
  enabled: true
  profile: general
```

这是需合并到现有文件的片段。保留文献、模型、搜索和 Harness 配置。`general` 适合通用研究；医学、临床或医疗 AI 研究使用 `medical`。后者增加数据使用依据、受试者隔离、偏倚和人群边界规则。新准备的研究默认使用 `enabled: true`、`profile: general`。如需关闭，在准备工作区前设置 `enabled: false`。

以下命令从仓库根目录执行。先创建一个带已有基线的 CPU 项目：

```bash
./scripts/research-agent init playground/methodology-demo --template circpack \
  --provider openrouter --model '填入可用模型ID' \
  --query "circle packing unequal circles unit square optimization"
```

编辑生成的 `research.yaml`，确认上面的配置片段。按[检索配置说明](model-search.zh-CN.md)设置模型和凭据，并准备运行 agent 所需的环境。使用 `--brief` 可从研究说明创建项目；复杂任务仍需补齐数据、评价器和任务契约。

```bash
# 只检查配置组合；不创建研究工作区，也不调用模型。
./scripts/research-agent harness playground/methodology-demo \
  --workspace results/harness/methodology-demo --dry-run agent.name=codex

# 完整流程：准备文献与规则，再执行议程、创新性审查、实验、写作和审稿。
./scripts/research-agent harness playground/methodology-demo \
  --workspace results/harness/methodology-demo agent.name=codex

# 中断后，使用同一工作区恢复。
./scripts/research-agent harness --resume results/harness/methodology-demo
```

完整运行的前提、容器环境和预算见[Harness 使用指南](research-harness.zh-CN.md)。`high_impact` 是配置目标；它不表示运行后一定达到高水平发表标准。

如只需准备输入，可另用一个工作区：

```bash
./scripts/research-agent prepare playground/methodology-demo \
  --workspace results/unified/methodology-prepared agent.name=codex
```

`prepare` 保存文献和方法学规则，不启动实验，也不单独生成 Harness 议程。它通常会调用文献检索模型；加 `--offline` 只使用已配置的本地来源。`run --resume` 可从这种工作区启动原有整合流程。若需要完整 Harness 的初始议程、独立审查和就绪报告，应使用上面的 `harness` 命令创建独立工作区。

## 冻结与恢复

规则保存在 `input_task/research_context/methodology.json`。准备阶段先写入规则，再调用初始议程生成和审查。该文件与任务、文献一起冻结。

| 字段 | 内容 |
|---|---|
| `policy_version` | 当前规则契约版本，目前为 `1` |
| `profile` | `general` 或 `medical` |
| `stage_rules` | 六个阶段分别使用的规则 |
| `sources` | 参考技能及附属材料的名称、SHA-256；已记录的许可证信息 |
| `rules_sha256` | 除哈希字段外，整个规则对象的规范化 SHA-256 |

加载时检查结构、版本、哈希和已知规则内容。恢复不能静默替换规则或改变 profile。需要改变配置、设计或输入时，在原始项目中修改，再建立新工作区。不要手工修改冻结文件，也不要通过 `.agentic_task_ignore` 排除它。

旧工作区没有方法学标记时，相关提示路径保持原有行为；恢复不会自动加入新规则。新代码读取旧 manifest 时保留这个边界。启用后的工作区若丢失标记，或标记与冻结配置不一致，会报错。

## 六个阶段实际接入什么

| 阶段 | 进入模型的内容 | 实际入口 |
|---|---|---|
| `agenda` | 有边界的问题、替代解释、不同预测、最小实验、停止或转向条件 | `draft_research_agenda` 的初始生成与反思提示 |
| `novelty` | 最近工作的方法与贡献边界、实质差异、混淆因素、尚缺证据 | `review_novelty` 的独立审查提示 |
| `idea` | 议程、拟议差异、判别实验、匹配的数据与计算预算 | `idea.sh` 通过 `render_context` 生成的研究上下文 |
| `coding` | 实现干预与对照、检查数据划分、保存种子与配置、先做最小测试 | `run.sh` 通过 `render_campaign_budget` 生成的执行上下文 |
| `paper` | 主张到证据的对应、可复现方法、来源核验、结论边界 | 写作和修改提示中的论文研究上下文 |
| `review` | 独立检查研究问题、对照、数据划分、引用与主张依据 | 论文内部审稿、证据审计及独立审稿的宿主核验材料 |

每次只提供当前阶段的简明规则，并附上相关议程字段。方法学提示块最多 8,000 字符；这不是 token 数。过长的议程不会被截成不完整 JSON，提示会要求读取完整的 `research_agenda.json`。规则本身不被截断。

现有议程反思、独立审查和 agent 操作仍按原预算执行。增加此配置不会自动增加检索次数、反思次数、实验数或审稿轮次。需要更多研究投入时，分别调整[研究预算](research-quality-guide.zh-CN.md)。

## 方法卡：需要声明哪些事实

方法卡是 `research_agenda.json` 中的一组字段，不是新增的独立文件。已知事实应附证据；缺失事实保留未知。下面是字段含义，不是已完成的研究结果。

| 位置 | 内容 |
|---|---|
| `research_question`、`scope` | 研究对象、条件、可测关系和适用范围 |
| `closest_work` | 最近工作的真实问题、核心方法、贡献边界、重合与差异；引用保存的来源片段 |
| `alternatives[].explanation` | 能解释同一观察的竞争解释 |
| `alternatives[].discriminating_experiment` | `intervention`、`controls`、`outcome_if_claim`、`outcome_if_alternative`、`decision_rule`；两种解释应产生不同的可观察预测 |
| `study_design.observation_unit`、`split_unit` | 测量单位与数据划分单位，例如图像和患者；二者可能不同 |
| `study_design.intervention`、`fixed_factors`、`matched_budget` | 改变什么、固定什么、如何匹配模型容量和训练、调参等预算 |
| `study_design.outcomes`、`falsifier` | 观测指标，以及什么结果会反驳主张 |
| `study_design.minimal_test`、`stop_or_pivot` | 足以改变下一步决策的小实验，以及不利结果下的行动 |
| `evidence_plan.strong_baselines`、`robustness_checks`、`independent_confirmation` | 强基线、稳健性检查和独立确认计划；各字段为文本列表 |

`study_design` 各字段通常为文本；`fixed_factors` 和 `outcomes` 也接受文本列表。声明一个实验计划，不表示实验已经执行。某个改动取得更高分，也不单独证明所提机制或新颖性。

医疗 profile 还使用 `medical_context.data_use_basis`、`subject_isolation`、`bias_and_population`。每项接受文本，或下面这样的事实对象：

```json
{
  "medical_context": {
    "data_use_basis": {"status": "unknown", "value": "尚未取得可核验的使用依据", "evidence": []},
    "subject_isolation": {"status": "unknown", "value": "尚未核对受试者跨划分重合", "evidence": []},
    "bias_and_population": {"status": "unknown", "value": "尚未核对来源人群与站点差异", "evidence": []}
  }
}
```

`known` 应提供实际内容及来源引用；`unknown` 是有效输入，但会保留证据缺口；`not_applicable` 需要说明理由。不能由公开下载、文件存在或模型判断推断伦理批准、同意或使用权限。研究指标也不能直接推出临床获益、诊断效力、治疗适用性或监管批准。

## 如何检查调用与结果

先检查冻结标记的 `policy_version`、`profile` 和 `rules_sha256`。再检查 `novelty/initial/agenda_draft/`、议程反思目录和 `reviewer_*/` 中的 `prompt.txt` 与 `receipt.json`。提示包含阶段与规则哈希；议程和创新性审查回执的 `frozen_evidence.methodology` 绑定同一规则版本。provider 原始响应与提示分别保存哈希，便于检查恢复时是否仍使用同一输入。

idea、coding 和 paper 走现有 agent 提示路径。查看对应运行保存的提示与 agent 输出，确认实际阶段和规则哈希。提示要求模型在结构化输出的 `methodology_binding` 中，或在没有元数据字段时的执行报告中，返回版本、哈希与阶段。该信息不进入论文正文。模型回显不等于遵守了规则；真实执行仍须看宿主评价、数据和来源记录。

完整 Harness 的 `paper_readiness.json` 增加 `checks.methodology`；`paper_readiness.md` 和 `research_next_actions.json` 也给出规则标识与后续行动。方法学检查报告声明的结构性缺口：

| 状态 | 含义 |
|---|---|
| `disabled` | 没有启用标记；旧工作区不增加方法学就绪条件 |
| `needs_evidence` | 必要声明缺失、事实未知，或两个竞争解释的预测完全相同 |
| `declared` | 被检查的必要字段已有声明；不是证据真实性、因果效力或临床有效性认证 |

当前覆盖检查针对冻结的初始议程，标记为 `assessment_scope: frozen_initial_agenda`、`selected_candidate_design_assessed: false`。最终候选的独立创新性审查另有候选绑定和议程；它不自动变成完整的方法学复核。即使初始方法卡为 `declared`，仍须核对最终实现是否保留了声明的对照和设计。

该检查不能代替实验、基线、消融、独立确认、引用审计、创新性审查和人工判断。最终确认数据一旦打开，修改设计或方法需要新研究；不能根据确认结果继续调参。`ready_for_human_review` 仅表示可以进入人工科研核查，不表示可自动投稿。

## 技能来源与集成边界

| 参考技能 | 本项目采用的原则 |
|---|---|
| `research-case-problem` | 先界定问题，再区分观察、解释和可判别预测 |
| `research-case-design` | 明确干预、对照、固定因素和残余混淆 |
| `research-case-innovation` | 发展有意义的增量贡献；核心高度重合时不强造 novelty |
| `research-case-validation` | 先做能改变决策的最小验证，并定义停止或转向条件 |
| `paper-writer` | 主张随证据强度收缩；引用需核验；保留检索失败和验证不足的状态 |
| `openclaw-medical-skills` | 医学 profile 下的数据权限、受试者隔离、人群偏倚和临床外推边界 |

实现位于 [`src/research/methodology.py`](../src/research/methodology.py)。它使用本项目编写的简明规则，并记录所参考材料的版本哈希；运行时不读取用户机器上的完整技能目录。`paper-writer` 及所参考附属材料的来源许可证记录为 `CC-BY-NC-SA-4.0`。来源记录不代表把该许可证套用于本项目全部代码。

这次集成没有安装整套医疗工具库，也不表示其中所有工具可用。方法学单元测试、保存提示检查或模拟 agent 路径检查，只验证实现与传递关系。它们不能替代真实 provider 调用，也不能证明研究质量已提高；需要另做预算匹配的实际案例比较。

本次接入的真实调用与自动检查结果见[验证记录](research-methodology-validation.zh-CN.md)。
