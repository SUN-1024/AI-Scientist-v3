# 原文定位工具

作者或独立审查器可通过只读 MCP 工具导航已经冻结的完整原文。工具返回现有校验器接受的 passage_id 和逐字 quote，避免人工计算长文的字符偏移。

工具只处理宿主准备的 source_packet(task_dir) 结果，不联网，不运行程序，不接收文件路径。原文内容仍是待分析的来源材料。定位正确不代表内容真实、支持当前主张或具有创新性。

## 准备与启动

宿主在新的工作区中保存完整 source packet，计算该 JSON 文件原始字节的 SHA-256。每个来源的 full_text 也必须与其 text_sha256 一致。JSON 文件哈希与 source_packet 的内容哈希是不同的绑定，不能混用。

~~~bash
python -m src.research.source_lookup \
  --packet /ABS/source_packet.json \
  --packet-sha256 YOUR_RAW_FILE_SHA256 \
  --audit-log /ABS/private-call/source-lookup.jsonl
~~~

服务使用 MCP 2025-06-18 stdio；stdout 仅含 JSON-RPC 消息。可选日志记录工具参数、结果哈希和来源包哈希。它不记录模型隐藏推理。

## 连接一次 Codex 调用

在本次运行的私有 CODEX_HOME 中，为所选 profile 增加以下配置。保留该 profile 原有的模型和 provider 配置；不要用示例覆盖全局配置。

~~~toml
[mcp_servers.agsci_sources]
command = "/ABS/trusted/venv/bin/python"
args = [
  "-m", "src.research.source_lookup",
  "--packet", "/ABS/source_packet.json",
  "--packet-sha256", "YOUR_RAW_FILE_SHA256",
  "--audit-log", "/ABS/private-call/source-lookup.jsonl",
]
cwd = "/ABS/trusted/harness"
~~~

原生 agent review 的 environment 已支持为单次调用指定 CODEX_HOME。原文包和工具代码应来自宿主冻结的输入；每次调用使用自己的日志目录。Codex 的 [stdio MCP 配置](https://learn.chatgpt.com/docs/extend/mcp) 由其原生客户端加载。

该连接不修改 run_fresh_agent_json 的 read-only 终端策略。若 Linux 内层命令沙箱无法启动，MCP 原文工具与终端命令须分别做运行验证。

## 四个工具

| 工具 | 输入 | 输出 |
|---|---|---|
| list_sources | 无 | 冻结的来源 ID、标题、可用性与文本哈希 |
| find_source_passages | source_id、query、可选 limit | 字面匹配附近的原文窗口及合法区间 ID |
| locate_source_quote | source_id、quote、可选 limit | exact quote 的所有匹配计数和前几个区间 |
| read_source_span | source_id、start、end | 最多 4000 个字符的原文及区间 ID |

字符偏移使用 Python Unicode code points。工具不改变换行、Unicode、空格或 NUL 字符。搜索区分大小写。重复匹配标记为 ambiguous；缺失标记为 not_found，不通过模糊匹配制造引用。片段仍需由作者或审查器阅读，并判断它能支持什么结论。

不完整来源可被列出，但不能通过详细引用工具取得支持。源文件哈希变化、正文哈希不符或路径中存在符号链接时，服务启动失败。工具参数不提供扩大文件访问或执行命令的入口。

## 验证范围

单元测试覆盖 Unicode 偏移与既有 _passage_span 校验器的往返、重复和缺失匹配、完整原文要求、原始文件哈希、能力边界，以及真实 stdio 子进程。测试不调用模型。

对具体研究 case，仍应在所选模型、profile 和容器中验证实际 MCP 调用，保留工具日志与 transport receipt。旧评审记录保持原样；新生成的定位建议不追溯改变旧判定。
