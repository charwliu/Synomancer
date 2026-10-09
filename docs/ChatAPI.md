# 本地 MLX API 与 Gemini 模型

默认聊天后端为本地 MLX。模型从聊天、ASR 和 TTS 共用的 HF_HOME 加载，
不连接 Ollama。切换模型会复用现有会话历史。

## App 调用 Google Gemini

设置 → **Gemini**：

1. 开启 **Enable Gemini models**。
2. 从 [Google AI Studio](https://aistudio.google.com/apikey) 获取 API Key，填入密钥框。
3. 在 **Model IDs** 每行填写一个文本生成模型 ID，或点击 **Choose API Models…**
   获取账户模型列表后选择。模型可用性由 Google 账户决定，列表可能包含非聊天模型。
4. 点击 **Apply Gemini Settings**，在聊天模型菜单选择 `Gemini · 模型ID`。

请求固定发送到 Google 官方 Gemini 兼容端点
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`，
使用 Bearer 认证、Chat Completions SSE 和最终 token 用量。
模型列表使用同一端点下的 `/models`。默认不发送 temperature，使用模型默认值。

Gemini API Key 存入 Keychain 的独立 `gemini` 项，不写入聊天数据库或 UserDefaults。
旧 OpenAI 密钥、模型设置不会自动迁移或发送给 Google。旧 OpenAI 会话保留，
继续聊天时需要选择 Gemini 或本地 MLX 模型；旧提供方请求会明确报错。
Gemini 内部模型 ID 带 `gemini:` 前缀，请求中发送原始模型 ID。

主聊天和 macOS 快捷补全均按所选模型路由。选择 Gemini 后会发送会话文本至 Google；
失败不会自动回退，停止回答会取消网络请求。
本次迁移保持原远程入口的文本范围，不启用图片、工具调用或音频。

## 向其他程序提供本地模型 API

设置 → **Local MLX API**：

1. 选择端口，默认 `8080`。
2. 生成或填写 **Local API Access Token**；这是本地服务的密钥，与 Gemini API Key 独立。
3. 点击 **Start Local API**。必要时通过目录选择器授权 HF_HOME。

地址为 `http://127.0.0.1:8080/v1`，固定只绑定本机 IPv4 loopback。
每个请求都需要 `Authorization: Bearer <本地访问令牌>`。
启用后会在下次启动 App 时恢复服务；停止服务会取消活动请求。
修改 HF_HOME 后需要重新启动服务。

支持：

- `GET /v1/models`：仅列出本机完整、受支持的 MLX 模型。
- `POST /v1/chat/completions`：普通 JSON 回答和 `stream: true` SSE 回答。
- 多轮字符串消息，角色 `system`、`developer`、`user`、`assistant`。
- `temperature`、`top_p`、`max_tokens` 或 `max_completion_tokens`（1–8192）、
  `stop` 字符串或至多四个字符串、`n: 1`、`stream_options.include_usage`。
- 真实 token 用量、`finish_reason: stop/length`，流结束标记 `[DONE]`。

仅暴露本地 MLX，不代理云端模型，也不产生 Gemini API 费用。
本地服务使用 Chat Completions，未实现 `/v1/responses`、工具调用、媒体输入、
结构化输出或 embedding。未支持的参数返回 HTTP 400，而不是被静默忽略。
客户端须发送带 Content-Length 的 HTTP/1.x JSON 请求，单次正文至多 1 MiB。
聊天、快捷补全及 API 共用串行 MLX 生成器，忙时请求排队；关闭客户端或服务会取消推理。

列出模型：

```sh
# LOCAL_API_TOKEN 请在本机从设置取得，不要提交到仓库。
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $LOCAL_API_TOKEN"
```

使用 OpenAI Python SDK 调用本地模型（把 model 替换为模型列表中的 ID）：

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ["LOCAL_API_TOKEN"],
)
stream = client.chat.completions.create(
    model="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    messages=[{"role": "user", "content": "解释一下 MLX。"}],
    max_completion_tokens=512,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    elif chunk.usage:
        print("\n", chunk.usage)
```

## 验证

构建 Release 后运行：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --cache /path/to/workspace/.cache/huggingface/hub --chat-api

python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --cache /path/to/workspace/.cache/huggingface/hub --chat-store
```

`--chat-api` 使用真实缓存模型和真实 loopback TCP/HTTP 测试本地模型列表、认证、
错误响应、普通/流式推理、stop、用量、客户端断开取消及监听器重启。
OpenAI Responses/Chat Completions 客户端使用受控 HTTP 夹具验证请求格式、中文 SSE、
多行事件、CRLF、401/429、生成失败、断流和取消；不访问真实账户或读取个人 API Key。
Gemini 回归额外验证 Google 端点、独立配置和密钥账户、消息历史与最终用量。
真实 Google 账户调用需在 App 设置中配置 Gemini API Key 后验证。
仅运行离线协议测试可使用 `--remote-api-only`，无需本地权重或个人密钥。
`--remote-store-only --cache <HF缓存>` 独立验证 Gemini 模型选择和旧 OpenAI 会话保留，
只扫描缓存，不加载权重推理。

`--chat-store` 使用内存 SwiftData 和独立设置验证旧会话保留、主聊天/快捷补全的真实 MLX 推理、
停止后重启，以及混合模型列表、远程模型选择和远程调用禁用时不会回退至本地模型。
这些测试共用一个测试可执行文件，请顺序运行。

远程协议依据：[Google Gemini 兼容接口](https://ai.google.dev/gemini-api/docs/openai)。

缓存中兼容的 Qwen OptiQ 模型在请求 `temperature: 0`、App 设置启用
**Use Local Qwen MTP** 时自动使用本地 MTP sidecar。其他温度保持普通采样；
缺失 sidecar 时使用主模型，最终 `usage` 使用真实生成 token 数。
详细格式与限制见 [MLXModelLibrary.md](MLXModelLibrary.md)。
