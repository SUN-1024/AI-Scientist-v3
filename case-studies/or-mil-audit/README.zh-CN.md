# 案例研究：OR-MIL 证书审计（case-studies/or-mil-audit）

这是一个真实的端到端科研案例：以 FocusMIL 的 A1–A5 为起点，做有限见证、证书审计、
版本空间与查询成本分析，并顺带审计了标准 MIL benchmark 的实例标签来源。产出两篇论文
（手稿与完整证据在工作区的 `scratch/` 下，不入库）。本目录只保留**可复现的实验脚本**。

## 数据

归档：Andrews, Tsochantaridis & Hofmann, MIL-Data-2002（Corel + TREC9 + MUSK），
`http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9.tgz`，
SHA-256 `590d7e6269cae7ff3ce013a7d11125e75b80721e3ae3bd62826d0158a49667a9`。

```bash
CASE_DATA=/tmp/or-mil-data HTTPS_PROXY=http://host:port bash fetch_archive.sh
```

脚本会校验归档 SHA-256，并只解出本案例使用的文件（Corel/TREC9 svm-light 与生成脚本）。

## 脚本

所有脚本都用环境变量定位输入输出：

- `CASE_DATA`：数据**根目录**，脚本读取其中的 `mil2002/*.svm`（`fetch_archive.sh` 生成的结构）
- `CASE_OUT`：结果目录（JSON/图）
- `CASE_REPO`：仓库根（默认按脚本位置推断）
- `CASE_DATASETS`：可选，逗号分隔的数据集名，默认全部十个（便于小样本测试）

| 脚本 | 作用 | 预期结果 |
|---|---|---|
| `ambiguity_analysis.py` | 版本空间大小与未决实例边缘概率 | 构造 7^4（log10=3.38）、边缘 4/7≈0.571；数据集 log10 161–545 |
| `identification_bounds.py` | 识别全部标签的查询上下界 | 单例上界比信息论下界高 2–10 次查询 |
| `benchmark_models.py` | 10 数据集 × 3 模型族的实例 F1 | linear/max-pool 全低于 prevalence；attention 7/10 高于 |
| `recompute_baseline.py` | 用评估集袋标签重算传播基线 | 纯数据上传播 F1=1（标签即袋标签） |
| `mixed_bag_experiment.py` | 40% 混合袋重建实验 | 传播 F1=0.75，仍高于所有模型 |
| `mixed_bag_sweep.py` | 混合比例 0/0.2/0.4/0.6 扫描 | 传播 F1 = 2(1−m)/(2−m)，attention 在 m=0.6 反超 |
| `plot_mixing_curve.py` | 画 Figure 1 | `mixing-curve.pdf` |
| `verify_note_numbers.py` | 手稿数字与证据 JSON 对账 | 41 项检查、0 失败 |

示例：

```bash
CASE_DATA=/tmp/or-mil-data CASE_OUT=/tmp/or-mil-out \
  python case-studies/or-mil-audit/benchmark_models.py
```

一键流程（`--skip-torch` 可跳过模型对照）：

```bash
CASE_DATA=/tmp/or-mil-data CASE_OUT=/tmp/or-mil-out bash run_all.sh --skip-torch
```

`benchmark_models.py`、`mixed_bag_*.py` 需要 PyTorch（CPU 版即可；论文实验用 2.11.0+cpu，
在固定容器镜像内运行）。其余脚本只用标准库与仓库内的 `src.research.audit.mil_assumption_audit`。

## 相关仓库工具

- `python -m src.research.audit.certificate_tables --root <data> --records <records>`：
  从原始 svm/CSV 重算全部证书表并与记录比对；
- `python -m src.research.audit.bundle_build ...`：确定性复现包构建；
- `agsci research verify-bundle ...`：复现包与手稿哈希校验（含陈旧哈希检测）；
- `python -m src.research.paper_claims ...`：手稿声明规则检查。

## 结论摘要

1. FocusMIL 印刷的 A1–A5 条件不足以确定实例真值：4 袋 8 点构造有 9 个相容映射；
2. 证书强制所有负例、几乎不强制正例；认证 k 个正例至少需 k 次查询；
3. 精确识别全部标签的成本被夹在 Σlog₂(2^r−1) 与 Σr 之间，单例查询近最优；
4. 标准 Corel/TREC9 归档的"实例标签"由生成器写入袋标签，标签传播基线在纯数据上 F1=1；
5. 受控混合后传播 F1 = 2(1−m)/(2−m)，attention 只在混合足够高时反超。
