# 用可配置模型检索论文

在线文献检索现在通过模型 API 执行，不再调用 Semantic Scholar API。使用者选择服务、模型、API 地址和 key 的环境变量。检索模型与 V3 的实验 coding agent 分开配置，可以使用不同服务。

这适用于新 `research` 项目的前置文献检索、论文初稿引用检索和初稿后的补充检索。通过项目中的 `research.yaml` 识别这一流程。上游没有该配置文件的旧任务仍保留原有引用流程。

## 两条执行路径

| 配置 | 执行过程 | 需要填写 |
|---|---|---|
| `native` | 调用模型及服务端联网工具 → 保存返回文本和结构化来源 → 解析论文记录 | 模型 ID、模型 API key；部分服务还需区域或工作空间地址 |
| `external` | 模型生成检索词 → 本机请求搜索服务 → 把结果交给同一模型解析 | 模型 ID、模型 API key，以及 SearXNG / Tavily / Brave / 智谱搜索配置 |

普通聊天 API 不一定有联网工具。仅让模型“列出论文”不能证明它做过搜索。`native` 必须返回结构化来源；`external` 必须取得搜索服务的结果。没有这些记录时，检索状态为 `unavailable`，不会用模型记忆补成成功结果。

外部搜索不要求模型支持 function calling。因此兼容普通 Chat Completions 的模型也可以参与。外部搜索服务仍可能限流；错误会保存，系统不会宣称已完整检索。

## 最短配置流程

在仓库根目录查看服务和配置模板：

```bash
./scripts/research-agent providers
./scripts/research-agent providers --config openrouter
./scripts/research-agent providers --config deepseek
```

[27 份 YAML 模板](../configs/research-providers/README.md)也已保存。把所选模板的 `literature.model_search` 合入项目的 `research.yaml`，保留已有的 `queries`、`sources` 和实验搜索配置。

新项目可以直接选择服务：

```bash
./scripts/research-agent init playground/my-study --template circpack \
  --provider openrouter --model '填入服务商支持的模型ID' \
  --query 'circle packing optimization closest methods'
```

模型 ID 由使用者填写。模板不锁定“最新模型”。同一品牌的聊天网页、Coding Plan 和通用 API 可能有不同地址、额度和工具权限；请使用 key 对应的 API 地址。

key 通过环境变量传入。程序不自动读取或复制其他电脑的凭据，也不把 key 写进 `research.yaml`。例如在 zsh 中用隐藏输入设置 key：

```bash
read -rs 'OPENROUTER_API_KEY?OpenRouter API key: '
export OPENROUTER_API_KEY
```

配置后先预览，再单独运行检索：

```bash
./scripts/research-agent prepare playground/my-study \
  --workspace results/unified/my-study --dry-run

# 这一步会使用已配置的模型 API 和搜索额度。
./scripts/research-agent prepare playground/my-study \
  --workspace results/unified/my-study
```

`--dry-run` 不联网，也不检验 key。检索结果在 `input_task/research_context/`。需要更换配置或重新检索时，请使用新工作区；`--resume` 会复用冻结的前置来源，不重新消费这部分检索额度。论文阶段可以为自己的新查询调用同一模型配置；同一查询和配置的成功引用结果会缓存。

## 有内置联网工具的服务

默认原生模板包括 OpenAI、OpenRouter、Anthropic、Gemini、阿里云百炼、火山方舟、智谱、xAI 和 Perplexity。Gemini 另有 `generateContent` 模板。它们通过六类 HTTP 协议适配；是否能使用所选模型，还取决于服务商账户和工具支持。

OpenRouter 示例：

```yaml
literature:
  queries:
    - "your research question and closest prior methods"
  model_search:
    provider: openrouter
    model: "填入模型ID"
    api_key_env: OPENROUTER_API_KEY
    max_output_tokens: 6000
    max_search_calls: 4
    timeout_seconds: 180
```

几个服务的具体差异：

| 服务 | 本实现使用的接口 | 配置注意点 |
|---|---|---|
| OpenAI | Responses + `web_search` | 保存 `web_search_call` 来源与 URL citations。[官方说明](https://developers.openai.com/api/docs/guides/tools-web-search) |
| OpenRouter | Chat Completions + `openrouter:web_search` | 使用当前 server tool，不使用旧 `:online` 后缀。[官方说明](https://openrouter.ai/docs/guides/features/server-tools/web-search) |
| Anthropic | Messages + `web_search_20250305` | 保存服务端搜索结果；`pause_turn` 最多续接 `max_continuations` 次。[官方说明](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) |
| Gemini | Interactions + `google_search` | 读取 `steps` 和 URL annotations；另支持 generateContent 的 grounding metadata。[官方说明](https://ai.google.dev/gemini-api/docs/google-search) |
| 阿里云百炼 | DashScope 原生 generation | 使用 `enable_search`、`forced_search`、`enable_source`；填写控制台提供的区域/工作空间 API 根地址，通常以 `/api/v1` 结尾。[官方说明](https://help.aliyun.com/zh/model-studio/web-search) |
| 火山方舟 | Responses + `web_search` | 默认工具来源 `doubao`；模型和联网权限须在账户中可用。[官方说明](https://docs.volcengine.com/docs/ark/web-search?lang=en) |
| 智谱 | Chat Completions + `web_search` | 请求返回搜索结果；如果账户接口只返回正文，可改用 `external` + `zhipu` 搜索后端。[官方说明](https://docs.bigmodel.cn/cn/guide/tools/web-search) |
| xAI | Responses + `web_search` | 以返回的搜索来源和 citations 为依据。[官方说明](https://docs.x.ai/developers/tools/web-search) |
| Perplexity | Agent API 的 `/v1/responses` 别名 | 解析 `search_results`；不依赖旧 Sonar Chat Completions 路径。[官方说明](https://docs.perplexity.ai/docs/agent-api/quickstart) |

服务接口会变化。自定义地址和参数可以覆盖模板；本地 HTTP 测试不代表这些服务已逐个用真实账户验证。

## 普通模型加外部搜索

已提供 DeepSeek、Kimi、MiniMax、硅基流动、Groq、Together、Fireworks、Mistral、NVIDIA NIM、百度千帆、腾讯混元、阶跃星辰、Yi、Ollama、LM Studio 和 vLLM 的配置入口。

以 DeepSeek + Tavily 为例：

```yaml
literature:
  model_search:
    provider: deepseek
    model: "填入模型ID"
    api_key_env: DEEPSEEK_API_KEY
    mode: external
    external_search:
      provider: tavily
      api_key_env: TAVILY_API_KEY
      max_results: 5
```

Tavily 默认使用其官方搜索地址。[Tavily API 文档](https://docs.tavily.com/documentation/api-reference/endpoint/search)

已有 SearXNG 实例时，不需要再购买一个搜索 key：

```yaml
    external_search:
      provider: searxng
      base_url: "http://127.0.0.1:8080"
      max_results: 5
```

实例必须已启用 JSON 搜索响应。模板没有指定公共实例，也不会自动安装 SearXNG。[SearXNG Search API](https://docs.searxng.org/dev/search_api.html)

另可选择 `brave`（默认环境变量 `BRAVE_SEARCH_API_KEY`）或 `zhipu`（默认 `ZHIPU_API_KEY`）。模型 key 与搜索 key 分开配置。外部模式每批最多执行 `max_search_calls` 条模型生成的查询，每条取回数量是 `external_search.max_results` 与 `literature.results_per_query` 的较小值。

本地模型选择 `ollama`、`lmstudio` 或 `vllm`。这些模板默认不要求模型 key，但仍需配置搜索后端。远程 API 使用 HTTPS；本机回环地址允许 HTTP。

## 自定义服务、网关和参数

`custom` 支持兼容网关。Azure 或其他使用不同路径/认证头的服务可配置如下字段；示例值须按服务文档替换：

```yaml
literature:
  model_search:
    provider: custom
    model: "your-model-or-deployment"
    base_url: "https://your-api.example/v1"
    api_format: chat_completions
    api_path: /chat/completions
    api_key_env: MY_RESEARCH_API_KEY
    api_key_header: Authorization
    api_key_prefix: "Bearer "
    mode: external
    request_overrides:
      temperature: 0.2
    external_search:
      provider: tavily
      api_key_env: TAVILY_API_KEY
```

支持的 `api_format`：`chat_completions`、`responses`、`anthropic`、`gemini`、`gemini_generate_content`、`dashscope`。空字符串表示使用服务模板默认值。`api_path` 可以含非凭据查询参数，例如 API version。API key 只能通过环境变量传入。

`request_overrides` 用于模型特有的请求字段。对象递归合并，数组整体替换，`null` 删除默认字段。例如仅接受 `max_completion_tokens` 的兼容服务可设置：

```yaml
    request_overrides:
      max_tokens: null
      max_completion_tokens: 6000
```

不允许覆盖模型输入或认证字段。解析与查询规划阶段会关闭联网工具。若原生工具使用未知的返回结构，系统可能无法识别来源；此时选择 `external`，或扩展 `source_receipts()` 并增加协议测试。

`proxy_url: http://127.0.0.1:7890` 可让模型和搜索请求直接走本地代理。未指定时采用系统代理配置；仅在 DNS 失败或连接被拒绝、请求未送达时尝试 `literature.proxy_fallback`。限流、认证失败和可能已计费的超时不会自动重发。论文下载也支持该备用代理。

## 返回文本、来源与费用记录

```text
research_context/
├── catalog.json                  # 论文元数据、全文状态、来源链接
├── text/                         # 实际取得的论文文本
└── model_search/
    ├── 01_native_search.json     # 原始服务响应，已做 key 脱敏
    ├── ...                      # 查询规划、搜索、解析或续接响应
    ├── parsed_answer.txt         # 用于最终解析的模型文本
    └── trace.json                # 服务、模型、调用阶段、usage、来源与错误
```

模型可返回 JSON、代码块或普通文字。普通文字最多再用一次模型调用转换成 JSON。来源链接必须与实际搜索记录对应；未对应的候选不会成为可靠检索结果。模型摘要保存在 `model_summary`，元数据标记为 `model_extracted_unverified`。它们不会变成 `abstract` 或 `full_text`。有来源链接也不保证标题、年份或贡献描述正确，后续仍要核对原文。

`max_output_tokens` 是每次模型调用的请求上限；服务端搜索的调用上限只在接口支持时生效。一次原生检索可能续接并追加一次格式解析。外部路径通常包含两次模型调用和若干搜索请求。`usage` 按服务原样保存；它不是跨平台的费用估算，也不是硬性金额上限。

收到限流、缺少 key、截断输出、错误 JSON 或没有来源时，查看 `trace.json`。已有的本地全文仍可使用。`prepare --offline` 跳过模型和网络检索，并使该工作区的论文引用阶段也只使用本地来源。实验和论文写作 agent 本身的运行仍遵循 V3 的设置。

## 论文阶段的引用核对

模型找到候选后，主机从论文页面的 `citation_*` 标签或 DOI 的 CSL 元数据读取标题、作者和年份，再生成 BibTeX。元数据不足时保留错误，不用模型记忆补齐。DOI 取回的是单篇文献记录，不用于论文发现。[DOI 内容协商说明](https://www.crossref.org/documentation/retrieve-metadata/content-negotiation/)

原始元数据、哈希和模型响应保存在论文工作区的 `reference_search/`；`references_provenance.json` 记录每条引用的来源。缓存文件和原始元数据均会检查哈希。达到论文配置的最低引用数量后才写出 `references.bib`。不足时停止该阶段，已有引用文件保留，不退回旧检索 API。

离线项目若要写参考文献，可在本地全文来源中填写从原文核对的作者信息：

```yaml
literature:
  sources:
    - id: closest_method
      title: "从原文核对的真实标题"
      authors: ["Surname, Given name"]
      year: 2025
      venue: "从原文核对的期刊或会议"
      url: "https://example.org/replace-with-actual-paper"
      path: papers/closest_method.pdf
```

这些字段标记为 `user_supplied`。在线页面或 DOI 记录标记为 `source_metadata`。两者都与模型提取但尚未核对的元数据区分开来。引用格式和字段有出处，仍不代表论文中的某个论断已被该来源支持。

开发验证和实际账户验证的边界见[验证记录](unified-research-validation.md)。
