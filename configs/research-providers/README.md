# Model search provider templates

Copy the selected `literature.model_search` into your project research.yaml.
Fill in a model ID and export the named key. These are configuration templates, not live account certifications.
See [the Chinese guide](../../docs/model-search.zh-CN.md) for setup and evidence handling.

| Provider | Mode | API format | Model key environment variable | Reference |
|---|---|---|---|---|
| [Magpie local gateway](magpie.yaml) | native | `responses` | optional; gateway configured locally | [local setup](../../docs/model-search.zh-CN.md#本机-magpie) |
| [OpenAI](openai.yaml) | native | `responses` | `OPENAI_API_KEY` | [docs](https://developers.openai.com/api/docs/guides/tools-web-search) |
| [OpenRouter](openrouter.yaml) | native | `chat_completions` | `OPENROUTER_API_KEY` | [docs](https://openrouter.ai/docs/guides/features/server-tools/web-search) |
| [Anthropic / Claude](anthropic.yaml) | native | `anthropic` | `ANTHROPIC_API_KEY` | [docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) |
| [Google Gemini Interactions](gemini.yaml) | native | `gemini` | `GEMINI_API_KEY` | [docs](https://ai.google.dev/gemini-api/docs/google-search) |
| [Google Gemini generateContent](gemini_generate_content.yaml) | native | `gemini_generate_content` | `GEMINI_API_KEY` | [docs](https://ai.google.dev/api/generate-content) |
| [阿里云百炼 / Qwen](dashscope.yaml) | native | `dashscope` | `DASHSCOPE_API_KEY` | [docs](https://help.aliyun.com/zh/model-studio/web-search) |
| [火山方舟 / Doubao](doubao.yaml) | native | `responses` | `ARK_API_KEY` | [docs](https://docs.volcengine.com/docs/ark/web-search?lang=en) |
| [智谱 / GLM](zhipu.yaml) | native | `chat_completions` | `ZHIPU_API_KEY` | [docs](https://docs.bigmodel.cn/cn/guide/tools/web-search) |
| [xAI / Grok](xai.yaml) | native | `responses` | `XAI_API_KEY` | [docs](https://docs.x.ai/developers/tools/web-search) |
| [Perplexity Agent API](perplexity.yaml) | native | `responses` | `PERPLEXITY_API_KEY` | [docs](https://docs.perplexity.ai/docs/agent-api/quickstart) |
| [DeepSeek](deepseek.yaml) | external | `chat_completions` | `DEEPSEEK_API_KEY` | [docs](https://api-docs.deepseek.com/) |
| [月之暗面 / Kimi](moonshot.yaml) | external | `chat_completions` | `MOONSHOT_API_KEY` | [docs](https://platform.moonshot.cn/docs/guide/start-using-kimi-api) |
| [MiniMax](minimax.yaml) | external | `chat_completions` | `MINIMAX_API_KEY` | [docs](https://platform.minimax.cn/docs/api-reference/text-openai-api) |
| [硅基流动](siliconflow.yaml) | external | `chat_completions` | `SILICONFLOW_API_KEY` | [docs](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions) |
| [Groq](groq.yaml) | external | `chat_completions` | `GROQ_API_KEY` | [docs](https://console.groq.com/docs/openai) |
| [Together AI](together.yaml) | external | `chat_completions` | `TOGETHER_API_KEY` | [docs](https://docs.together.ai/docs/openai-api-compatibility) |
| [Fireworks AI](fireworks.yaml) | external | `chat_completions` | `FIREWORKS_API_KEY` | [docs](https://docs.fireworks.ai/api-reference/post-chatcompletions) |
| [Mistral](mistral.yaml) | external | `chat_completions` | `MISTRAL_API_KEY` | [docs](https://docs.mistral.ai/api/) |
| [NVIDIA NIM](nvidia.yaml) | external | `chat_completions` | `NVIDIA_API_KEY` | [docs](https://docs.api.nvidia.com/nim/reference/llm-apis) |
| [百度千帆 / ERNIE](baidu.yaml) | external | `chat_completions` | `QIANFAN_API_KEY` | [docs](https://cloud.baidu.com/doc/qianfan-api/index.html) |
| [腾讯混元](hunyuan.yaml) | external | `chat_completions` | `HUNYUAN_API_KEY` | [docs](https://cloud.tencent.com/document/product/1729/111007) |
| [阶跃星辰 / Step](stepfun.yaml) | external | `chat_completions` | `STEPFUN_API_KEY` | [docs](https://platform.stepfun.com/docs/overview/concept) |
| [零一万物 / Yi](yi.yaml) | external | `chat_completions` | `YI_API_KEY` | [docs](https://platform.lingyiwanwu.com/docs) |
| [Ollama](ollama.yaml) | external | `chat_completions` | `` | [docs](https://docs.ollama.com/api/openai-compatibility) |
| [LM Studio](lmstudio.yaml) | external | `chat_completions` | `` | [docs](https://lmstudio.ai/docs/developer/openai-compat) |
| [vLLM](vllm.yaml) | external | `chat_completions` | `` | [docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) |
| [Custom / Azure / LiteLLM / other gateway](custom.yaml) | external | `chat_completions` | `AGSCI_LITERATURE_API_KEY` | user configured |
