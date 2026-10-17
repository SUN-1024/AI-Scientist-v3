"""Editable provider presets, not a claim that every model supports web search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    label: str
    api_format: str
    mode: str
    base_url: str
    api_key_env: str
    docs: str


# Model IDs are deliberately supplied by the user: model access and tool support vary.
PROVIDERS = {
    "openai": Provider("OpenAI", "responses", "native", "https://api.openai.com/v1", "OPENAI_API_KEY", "https://developers.openai.com/api/docs/guides/tools-web-search"),
    "openrouter": Provider("OpenRouter", "chat_completions", "native", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "https://openrouter.ai/docs/guides/features/server-tools/web-search"),
    "anthropic": Provider("Anthropic / Claude", "anthropic", "native", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool"),
    "gemini": Provider("Google Gemini Interactions", "gemini", "native", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY", "https://ai.google.dev/gemini-api/docs/google-search"),
    "gemini_generate_content": Provider("Google Gemini generateContent", "gemini_generate_content", "native", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY", "https://ai.google.dev/api/generate-content"),
    "dashscope": Provider("阿里云百炼 / Qwen", "dashscope", "native", "", "DASHSCOPE_API_KEY", "https://help.aliyun.com/zh/model-studio/web-search"),
    "doubao": Provider("火山方舟 / Doubao", "responses", "native", "https://ark.cn-beijing.volces.com/api/v3", "ARK_API_KEY", "https://docs.volcengine.com/docs/ark/web-search?lang=en"),
    "zhipu": Provider("智谱 / GLM", "chat_completions", "native", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "https://docs.bigmodel.cn/cn/guide/tools/web-search"),
    "xai": Provider("xAI / Grok", "responses", "native", "https://api.x.ai/v1", "XAI_API_KEY", "https://docs.x.ai/developers/tools/web-search"),
    "perplexity": Provider("Perplexity Agent API", "responses", "native", "https://api.perplexity.ai/v1", "PERPLEXITY_API_KEY", "https://docs.perplexity.ai/docs/agent-api/quickstart"),
    "deepseek": Provider("DeepSeek", "chat_completions", "external", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "https://api-docs.deepseek.com/"),
    "moonshot": Provider("月之暗面 / Kimi", "chat_completions", "external", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "https://platform.moonshot.cn/docs/guide/start-using-kimi-api"),
    "minimax": Provider("MiniMax", "chat_completions", "external", "https://api.minimax.cn/v1", "MINIMAX_API_KEY", "https://platform.minimax.cn/docs/api-reference/text-openai-api"),
    "siliconflow": Provider("硅基流动", "chat_completions", "external", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY", "https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions"),
    "groq": Provider("Groq", "chat_completions", "external", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "https://console.groq.com/docs/openai"),
    "together": Provider("Together AI", "chat_completions", "external", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "https://docs.together.ai/docs/openai-api-compatibility"),
    "fireworks": Provider("Fireworks AI", "chat_completions", "external", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "https://docs.fireworks.ai/api-reference/post-chatcompletions"),
    "mistral": Provider("Mistral", "chat_completions", "external", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "https://docs.mistral.ai/api/"),
    "nvidia": Provider("NVIDIA NIM", "chat_completions", "external", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "https://docs.api.nvidia.com/nim/reference/llm-apis"),
    "baidu": Provider("百度千帆 / ERNIE", "chat_completions", "external", "https://qianfan.baidubce.com/v2", "QIANFAN_API_KEY", "https://cloud.baidu.com/doc/qianfan-api/index.html"),
    "hunyuan": Provider("腾讯混元", "chat_completions", "external", "https://api.hunyuan.cloud.tencent.com/v1", "HUNYUAN_API_KEY", "https://cloud.tencent.com/document/product/1729/111007"),
    "stepfun": Provider("阶跃星辰 / Step", "chat_completions", "external", "https://api.stepfun.com/v1", "STEPFUN_API_KEY", "https://platform.stepfun.com/docs/overview/concept"),
    "yi": Provider("零一万物 / Yi", "chat_completions", "external", "https://api.lingyiwanwu.com/v1", "YI_API_KEY", "https://platform.lingyiwanwu.com/docs"),
    "ollama": Provider("Ollama", "chat_completions", "external", "http://127.0.0.1:11434/v1", "", "https://docs.ollama.com/api/openai-compatibility"),
    "lmstudio": Provider("LM Studio", "chat_completions", "external", "http://127.0.0.1:1234/v1", "", "https://lmstudio.ai/docs/developer/openai-compat"),
    "vllm": Provider("vLLM", "chat_completions", "external", "http://127.0.0.1:8000/v1", "", "https://docs.vllm.ai/en/latest/serving/openai_compatible_server/"),
    "custom": Provider("Custom / Azure / LiteLLM / other gateway", "chat_completions", "external", "", "AGSCI_LITERATURE_API_KEY", ""),
}

API_FORMATS = {provider.api_format for provider in PROVIDERS.values()}
