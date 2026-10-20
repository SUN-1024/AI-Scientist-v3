# 快速弱监督病理案例

问题：保持肿瘤 patch 不变，只加入正常组织背景，MIL 分类器是否变差？小幅修改聚合方法能否在 15% bag 标签噪声下保持效果？

本例通过融合项目的 `research run` 运行。默认模型为 DeepSeek 官方 API 的
`deepseek-flash`，直接使用 `https://api.deepseek.com/`，不需要 Magpie。
用途是检查研究流程及短时实验能力。它使用真实 PathMNIST 图像构造合成 bags，
不能当作临床全切片或患者级验证。

本页先给出最小连通配置。增加预算的开发研究见下方“多 seed 与更大搜索”；各层预算的含义见[研究质量指南](../../docs/research-quality-guide.zh-CN.md)。

## 实验预算

| 项目 | 设置 |
|---|---|
| 图像特征 | 固定颜色/纹理统计，训练集拟合 StandardScaler 和 PCA(48)；一次缓存 |
| Bag 监督 | 600 个训练 bags；只有 bag 标签参与模型训练 |
| 配对验证 | 100 个 bags；16 个原始 patch 与加入 48 个非肿瘤 patch 后的 64-patch 视图 |
| 留出评估 | 80 个 bags，放在搜索任务目录之外；搜索后才使用 |
| 标签噪声 | 0%、15%；同一个 seed 使用相同翻转位置 |
| 小模型 | 基线为 gated attention MIL；最多 20,000 个参数 |
| 单条件训练 | 最多 160 steps；每 20 steps 验证；patience=3；20 秒训练上限 |
| 搜索 | 2 个 idea cycles；每次只生成一个候选，由宿主评估一次 |

20 秒限制是训练循环的软上限。Python 导入、模型生成、搜索与文件准备另计。预处理和原始数据下载只做一次。

## 准备

从仓库根目录执行。先按仓库安装文档安装 `.venv`、GNU Bash、Codex 和 Headless CLI，再安装案例依赖：

```bash
uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple -r examples/pathmnist_mil/requirements.txt
mkdir -p scratch/weak-supervision/data
curl -fL --retry 2 -C - --proxy http://127.0.0.1:7890 \
  -o scratch/weak-supervision/data/pathmnist.npz \
  'https://zenodo.org/records/10519652/files/pathmnist.npz?download=1'
.venv/bin/python examples/pathmnist_mil/prepare_data.py \
  --archive scratch/weak-supervision/data/pathmnist.npz \
  --holdout scratch/weak-supervision/holdout.npz
.venv/bin/python examples/pathmnist_mil/download_papers.py
```

`prepare_data.py` 核验官方 MD5，移除跨分区完全重复图像，并保存构造记录和 SHA-256。不会覆盖已有特征或 holdout。数据和论文缓存不进入 Git。

为实验 Codex 创建专用 `~/.codex/agsci-deepseek.config.toml`，可从
[模板](../../configs/research-providers/agsci-deepseek.config.toml.example)复制，
并从该模板目录复制 `deepseek-models.json` 到同一 `.codex` 目录。配置文件不含
API key。设置 `DEEPSEEK_API_KEY` 后，Codex 直接访问 DeepSeek Responses API。

## 执行融合流程

```bash
./scripts/research-agent run examples/pathmnist_mil \
  --workspace results/unified/pathmnist-mil-quick \
  preset=pathmnist_mil --dry-run

# 调用 DeepSeek Flash 完成真实文献检索、idea 生成、代码生成和实验。
./scripts/research-agent run examples/pathmnist_mil \
  --workspace results/unified/pathmnist-mil-quick \
  preset=pathmnist_mil
```

本例是显式启用的本机开发模式。搜索 agent 没有容器文件隔离。它应只操作自己的任务目录；不要把这种运行描述为 sealed evaluation。正式研究采用 V3 容器与任务契约流程。

已有工作区通过 `--resume` 继续；新的实验使用新工作区。不要修改冻结的 `input_task`。本机如使用独立构建的 Headless CLI，可设置 `HEADLESS_BIN` 为其可执行文件绝对路径。

## 多 seed 与更大搜索

先比较基线和已有对照的共享训练预算，再选 `--max-steps`。新任务支持冻结的 `evaluation_protocol.json`，评价器记录其内容和 SHA-256。默认未提供协议时仍使用 seed 7、160 steps 的快速设置。

```bash
# 三个已归档对照 × 三个训练预算 × 三个 seed × 两个噪声条件。
# 输出目录必须不存在；只读取 search.npz，不打开旧留出数据。
.venv/bin/python examples/pathmnist_mil/calibrate.py \
  --search-data examples/pathmnist_mil/resources/search.npz \
  --output scratch/weak-supervision/calibration

# 本次共享预算校准后使用 640 steps；新数据需重新校准。
.venv/bin/python examples/pathmnist_mil/create_study.py \
  --output scratch/weak-supervision/research-task \
  --max-steps 640 --seeds 7 11 19

./scripts/research-agent run scratch/weak-supervision/research-task \
  --workspace results/unified/pathmnist-mil-research \
  preset=pathmnist_mil_research --dry-run

./scripts/research-agent run scratch/weak-supervision/research-task \
  --workspace results/unified/pathmnist-mil-research \
  preset=pathmnist_mil_research
```

研究 preset 设为 24 个调度轮次、高推理强度、单 worker。任务创建脚本提高文献检索预算，并冻结三个 seed、相同学习率与标签平滑的评价协议。此开发协议关闭早停，以共同步数上限比较模型；仍保留每条件 20 秒软上限。每个候选的一次宿主评价运行全部 seed 和噪声条件。临时编码 API 错误最多尝试三次，仍属于同一个 idea；低分实验不会因此重新抽样。

已有留出结果只作为历史记录。这个开发续跑不打开旧留出集，也不声称完成独立确认。新工作区内的验证提升仍是开发结果。正式确认需要新数据和预先确定的比较计划。

评价器同时检查实例置换不变性与 batch 排列等变性。它记录逐条件 AUROC、阳性与阴性概率变化、类间差距、训练预算是否用尽，以及总参数、可训练参数和观察到梯度连接的参数。声明了但未参与计算的参数不再被当作已证实的有效容量。

## 文献和贡献边界

- [Attention-based Deep Multiple Instance Learning, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html)：attention/gated attention 聚合基线。
- [Attention2Minority, 2023](https://arxiv.org/abs/2301.07700)：使用正常组织特征推断显著实例，针对小病灶选择实例。
- [PseMix, IEEE TMI 2024](https://arxiv.org/abs/2306.16180)：pseudo-bag mixup，包含噪声与遮挡相关实验。
- [Ranking-Aware MIL, JMIR Medical Informatics 2026](https://medinform.jmir.org/2026/1/e84417/)：利用部分专家标签引导排序和阈值重归一化，与稀疏注意力思路接近。

本例的价值是固定阳性证据的配对干预、快速诊断和流程验证。仅改 top-k/attention 温度不构成可信的新方法贡献。候选可以作为 replication/control 保留。完整论文结论仍需更强特征、真实 WSI、患者划分、重复实验及匹配预算的对照。

## 搜索后留出检查

以下命令只适用于尚未查看留出集的首次快速试验。确认搜索完成后运行一次。它先冻结基线和已选程序，再使用两个新 seed；留出结果不回传给搜索。本仓库已记录的留出集已查看，不应用于上方开发续跑的独立确认；创建脚本会标记该限制，评价脚本会拒绝此用途。

```bash
.venv/bin/python examples/pathmnist_mil/evaluate_holdout.py \
  --workspace results/unified/pathmnist-mil-quick \
  --holdout scratch/weak-supervision/holdout.npz \
  --output scratch/weak-supervision/holdout-evaluation \
  --include-best-challenger
```

这是本机的开发检查。它与正式 V3 的容器封闭评估阶段不同。已有输出不会被覆盖。
可选的 `--include-best-challenger` 在查看留出结果前，按验证分数固定最高分的有效候选，
作为诊断对照。如果搜索选中的程序就是基线，两者复用同一次评估。它不会根据留出分数重新选方法。
