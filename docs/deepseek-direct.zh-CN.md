# DeepSeek Flash 直连

本项目默认使用 Codex 客户端连接 DeepSeek 官方 Responses API。调用链为：

```text
Codex CLI -> https://api.deepseek.com/ -> deepseek-flash
```

Magpie 不再是默认路径。

## 配置文件

- profile：`configs/research-providers/agsci-deepseek.config.toml.example`
- 模型目录：`configs/research-providers/deepseek-models.json`
- 目录来源：`configs/research-providers/deepseek-models.provenance.json`
- 当前 checkout 的私有 profile：
  `scratch/harness-audit/container-auth/.codex/agsci-deepseek.config.toml`
- 当前 checkout 的私有模型目录：
  `scratch/harness-audit/container-auth/.codex/deepseek-models.json`

profile 使用 `env_key = "DEEPSEEK_API_KEY"`。仓库和 profile 都不保存 API key。

## 设置密钥

在运行 Harness 的进程环境中设置：

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'
```

也可以创建私有 env 文件，并交给外层 Docker runner：

```bash
export RESEARCH_RUNNER_ENV_FILE=/private/path/deepseek.env
```

该文件必须包含：

```dotenv
DEEPSEEK_API_KEY=YOUR_KEY
```

不要把该文件放入 Git、任务目录或导出的研究快照。

## 运行

直接运行：

```bash
./scripts/research-agent harness playground/my-study \
  --workspace results/harness/my-study \
  agent.name=codex \
  agent.profile=agsci-deepseek \
  agent.model=deepseek-flash \
  review.agent_profile=agsci-deepseek \
  review.agent_model=deepseek-flash
```

Linux coordinator 已把 `agsci-deepseek` 设为默认 profile。设置
`DEEPSEEK_API_KEY` 后直接执行现有容器命令即可。

## 搜索边界

DeepSeek Flash 支持 Responses API，但 Codex 内置的服务端 `web_search` 仅适用于
`deepseek-v4-pro`。因此 profile 明确设置：

```toml
web_search = "disabled"
```

本项目的研究来源检索继续使用 DeepSeek 独立 Search API：

- `provider: deepseek`
- `model: deepseek-flash`
- `mode: external`
- `api_key_env: DEEPSEEK_API_KEY`

模型调用和搜索调用是两条独立路径。仅更换模型 ID 不会自动启用联网。

## 保留的工具

作者和审稿器的来源定位工具、受限 Python 工具仍通过 Codex 客户端的 MCP 接口运行。
这些工具的上游模型现在是 DeepSeek Flash；私有 profile、工具配置、回执和隔离策略
保持不变。

## 核验

```bash
.venv/bin/python -m pytest -q tests/research/test_linux_runner.py --tb=short
python3 -m py_compile \
  src/docker/research-runner/native_state.py \
  src/docker/research-runner/preflight.py
bash -n scripts/run-research-harness-container.sh
```

在没有设置 `DEEPSEEK_API_KEY` 时，可以验证 profile、模型目录和 staging，但不能
完成真实模型调用。密钥设置后，再分别验证 `/models`、一次小型 Responses 请求，
以及带来源工具的短 Harness 调用。

来源：

- [DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/)
- [DeepSeek Codex integration](https://api-docs.deepseek.com/quick_start/agent_integrations/codex/)
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
