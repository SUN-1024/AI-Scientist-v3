# 快速弱监督病理实验与集成检查

日期：2026-10-18。入口：`scripts/research-agent run`。结论：真实模型检索、idea 生成、代码生成和宿主评分已连通。两个有效候选均未超过基线，因此保留基线。此次结果支持流程验证，不支持新方法有效性或融合系统优于 V1/V2/V3 的结论。

## 实际使用的模型与流程

- 文献检索：本机 Magpie Responses 网关，`http://127.0.0.1:3425/v1`，模型 `codex/gpt-6.1-sol`，原生 `web_search`。
- 实验 agent：Headless CLI 调用 Codex，专用 `agsci-magpie` profile 使用同一网关和模型，reasoning effort 为 `low`。
- 运行链路：`research run` → 冻结输入与文献 → V3 investigation → idea/rubric → 候选代码 → 宿主评价器 → 搜索树与最佳程序记录。没有使用另一个实验框架代替该链路。
- 搜索后：独立脚本冻结比较对象，以两个新 seed 检查留出集。留出结果没有回传给搜索。
- 本次只运行 investigation。论文、完整消融和 V3 sealed final evaluation 未运行。运行边界为本机开发模式，没有容器文件隔离。

在线检索与三个预先准备的本地全文合并后，共得到 6 篇去重来源，其中 4 篇有全文。另两篇只保存元数据：一篇下载为空，一篇返回 HTTP 418。状态保持 `partial`，没有将模型概述当作原文，也没有自动切回 Semantic Scholar API。来源及哈希见 [source_index.json](research/pathmnist-mil-pilot-2026-10-18/source_index.json)。

## 问题与小预算设置

问题：保持肿瘤实例不变，加入非肿瘤背景后，bag 分类是否变差？修改聚合方式能否减少这种退化，并承受 15% bag 标签噪声？

真实图像来自 [PathMNIST 官方发布](https://zenodo.org/records/10519652)，由图像构造合成 bags。Patch 标签仅用于构造 bag，不传给训练模型。此设置不是临床 WSI，也没有证明患者独立性。

| 项目 | 本次设置 |
|---|---|
| 数据 | 600 个训练 bags、100 个验证 bags、80 个外部留出 bags |
| 配对干预 | 16 个原始 patch → 加入 48 个非肿瘤 patch；原有 1–2 个阳性实例不变 |
| 特征 | 固定颜色/纹理特征；仅训练分区拟合标准化和 PCA(48) |
| 监督 | Bag 标签；无 patch 标签训练、无 backbone 训练 |
| 噪声 | 干净标签、固定 15% 对称翻转；方法间共享 seed 和翻转位置 |
| 训练 | CPU 2 线程，batch 32，最多 160 steps，每 20 steps 验证，patience 3 |
| 上限 | 20 秒训练软上限、20,000 个参数；本次模型均定义 2,673 个参数 |
| 验证 seed | 7 |
| 留出 seed | 11、19；checkpoint 仍由原验证集选择，留出集不用于早停 |

官方图像分区共 89,996 / 10,004 / 7,180 张。本次精确去重检查未发现需移除的重复图像。不同 bags 在各自分区内不复用 patch；配对视图有意复用原始 bag。缺少患者 ID，因此精确去重不能代替患者划分。[数据与哈希记录](research/pathmnist-mil-pilot-2026-10-18/data_manifest.json)

## 搜索结果

选择分数为：先对每个噪声条件计算 `min(AUROC_16, AUROC_64)`，再取均值。不能用两列均值的较小值替代这个计算。

| 程序 | 验证 AUROC，16 patch | 验证 AUROC，64 patch | 验证选择分数 | 留出选择分数 |
|---|---:|---:|---:|---:|
| Gated attention 基线 | 0.6412 | 0.5582 | **0.5582** | **0.5795** |
| Log-mean-exp 实例聚合 | 0.6606 | 0.5448 | 0.5448 | 未评估 |
| Top-2 gated attention，保留边界并列项 | 0.5620 | 0.5502 | 0.5456 | 0.5272 |

验证两列为两个噪声条件的均值。留出分数为两个新 seed × 两个噪声条件，共四项最差视图 AUROC 的均值。它们是小样本描述统计，没有显著性结论。

搜索对象在查看留出结果前固定：基线、搜索选中的程序，以及验证分数最高的有效候选。搜索选中程序就是基线，因此复用同一次评估。最高分候选是 Top-2 gated attention。没有根据留出结果重新选择或修改方法。[预先固定的比较对象](research/pathmnist-mil-pilot-2026-10-18/holdout_comparison_plan.json)；[留出逐条件结果](research/pathmnist-mil-pilot-2026-10-18/holdout_summary.json)

两个候选的接口与宿主评价都通过，但改进分数的 rubric 未通过。V3 因此显示 `completed no winner`：没有通过改进门槛的新赢家。最佳程序记录仍指向有效基线。不能把这个状态写成“发现了更好的方法”。

Log-mean-exp 候选保留了未使用的 attention 参数。其定义参数数与基线相同，但参与前向计算的容量较小，不能称为严格容量匹配。Top-2 gated attention 保留原有 encoder、attention 和 classifier，并让所有模块参与前向计算。所有程序都通过 16/64 实例的置换不变性检查；Top-2 另检查了所有 attention 分数相同的情况。[接口检查](research/pathmnist-mil-pilot-2026-10-18/interface_checks.json)

## 如何解释负结果

验证集上，基线加入背景后的 AUROC 下降，但阳性 bag 的平均概率没有下降。干净标签下，阳性概率增加约 0.0168，阴性概率增加约 0.0723；噪声条件下分别增加约 0.0227 和 0.0993。这说明仅看“阳性得分是否下降”不足以判断背景影响。Top-2 的概率变化更稳定，也没有转化成更好的区分能力。[搜索结束后的概率变化分析](research/pathmnist-mil-pilot-2026-10-18/posthoc_score_change.json)

当前证据不能排除特征不足、背景极值、优化差异和标签噪声等解释。下一项研究应预先确定更强的冻结特征和匹配容量对照，再使用新的留出划分。不要用本次留出分数继续调 top-k。

这些候选作为已知聚合对照保留。相关工作包括 [ABMIL，ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html)、[Attention2Minority，2023](https://arxiv.org/abs/2301.07700)、[PseMix，TMI 2024](https://arxiv.org/abs/2306.16180) 和 [LNPL-MIL，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shao_LNPL-MIL_Learning_from_Noisy_Pseudo_Labels_for_Promoting_Multiple_Instance_ICCV_2023_paper.html)。本次不主张 top-k、稀疏注意力或 log-mean-exp 的算法新颖性。配对背景干预是后续机制研究的起点，不是已经建立的论文贡献。

## 实际发现的问题与处理

| 问题 | 原因与处理 | 验证状态 |
|---|---|---|
| 原生模式无法开始第一个实验 | 缺少宿主评估过的基线；新增独立基线副本、一次评估与代码/评价器/结果哈希校验 | 真实基线完成；恢复复用已验证结果；更改字节会被拒绝 |
| 配置显示 tree，实际执行仍为 best | investigation 的配置加载器遗漏三个搜索字段 | 已补齐；真实恢复运行写出树节点；配置传递回归通过 |
| 自定义 task 标题导致目标缺失 | 提示词只提取 Goal/Rubrics；现在 idea 与 coding 提示均附完整任务 | idea 真实读取接口与全文；两个模板的自定义标题回归通过 |
| 禁止所有命令，导致无法读本地论文 | 旧 idea 提示禁止必要文件读取；改为限定范围的文件 I/O 与输出验证 | 修复后真实读取保存全文；仍禁止训练与评价 |
| 实验 profile 暴露不需要的桌面工具 | 默认 MCP/功能继承；专用 profile 关闭桌面、浏览器、手机、跨 agent 和记忆发现能力 | 配置读取检查通过；后续使用限定文件读写；全局用户配置未改 |
| Mac Git 检查点失败 | 上游固定目录描述符路径依赖 Linux | 本例关闭自动 Git 检查点；保留 JSON 证据。这是规避措施，未宣称修复跨平台实现 |
| Magpie 上游临时 502 | 第二个 idea 的 coding 调用在生成候选前失败 | 保留为执行失败；专用 profile 的请求/流重试上限改为 3。尚未用故障注入验证恢复效果 |

初始配置是两个 idea cycles。调试中增加到三个调度轮次，以验证修复；实际只有两个候选完成评分，另一个轮次因 502 没有生成代码。第三轮的 idea 文本仍把旧轮数当作限制，说明预算的结构化传递还应改进。不能把一次失败调用当作一次方法比较。

树配置修复发生在首次候选之后，因此此次树记录覆盖后续轮次，并不是一次完整的分支质量比较。较低分分支可继续被探索的行为，另由确定性 CPU 集成测试验证。

## 时间、检查与下一步优化

- 本次验证集训练每个条件约 0.059–0.115 秒，实际 120–160 steps。留出训练约 0.059–0.111 秒。评价函数耗时约 0.55–2.00 秒；这些数字不包括外部进程启动和全部导入时间。
- 最后一次恢复运行耗时约 6 分 19 秒，主要用于模型阅读和生成。它不是下载、数据准备、前置检索及此前失败尝试的总耗时。没有进行全量 epoch 或 GPU backbone 训练。
- 最终回归检查：主命令 292 项通过；基线检查点 5 项通过，共 297 项。macOS 排除一项 Linux 描述符测试，该项在此前 Linux 容器检查中通过。Ruff 和新源模块的 mypy 通过。
- 真实调用验证了本机 Magpie 的这个模型及搜索路径。其余 provider 模板仍以 HTTP fixture 验证为主，不代表所有账户和模型都已实测。

仍值得优先改进：运行前先核验基线、agent profile 和数据，再消费在线检索额度；将 API 失败作为独立、可恢复的状态；给 idea/coding 两个阶段传同一份结构化预算；缓存已核对的方法卡，减少每轮重复阅读全文。正式研究应恢复容器隔离与封闭最终评估，并做多任务、同预算比较。

## 可复现材料

- [运行说明与命令](../examples/pathmnist_mil/README.md)
- [本次固定任务](https://github.com/SUN-1024/AI-Scientist-v3/blob/3ff1951/examples/pathmnist_mil/task.md)、[评价器](https://github.com/SUN-1024/AI-Scientist-v3/blob/3ff1951/examples/pathmnist_mil/evaluate.py)与[小预算 preset](https://github.com/SUN-1024/AI-Scientist-v3/blob/3ff1951/configs/preset/pathmnist_mil.yaml)；链接固定到首次案例提交，避免与后续改进混用。
- [程序、逐条件指标、idea 与检查结果](research/pathmnist-mil-pilot-2026-10-18/README.md)
- [多模型 API 配置](model-search.zh-CN.md)

公开材料不包含 API key、原始图像、论文 PDF 或完整 agent 日志。数据和全文按说明从原始来源取得。
