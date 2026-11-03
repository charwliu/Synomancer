# 本地 MLX API 与 OpenAI / Gemini 模型

默认聊天后端为本地 MLX。模型从聊天、ASR 和 TTS 共用的 HF_HOME 加载，
不连接 Ollama。切换模型会复用现有会话历史。

## App 调用 OpenAI 及兼容接口

设置 → **OpenAI** 标签 → **OpenAI-compatible Services** 支持同时保存多个服务，每项独立设置。服务名称、Base URL 和密钥使用独立的带边框输入框；全局保存失败时，原因会显示在设置窗口顶部：

1. 选择已有服务，或用 **Add Service** 新增；填写互不重复的 **Service name**，
   开启 **Enable this service**。同一服务的名称、地址、密钥、模型和 API 模式均独立保存。
2. **Base URL** 默认是 `https://api.openai.com/v1`，留空也使用官方地址。
   兼容服务填入自己的 API 根地址，例如 `https://api.example.com/proxy/v1`
   或 `http://127.0.0.1:8080/v1`；保留服务要求的路径前缀，不填 `/responses`、
   `/chat/completions` 或 `/models` 后缀，App 不会自动添加 `/v1`。
3. 官方接口从 [OpenAI Platform](https://platform.openai.com/api-keys) 获取密钥；
   兼容服务填写该服务自己的 API Key，无需认证的服务可以留空。
4. 在 **Model IDs** 中每行填写一个模型 ID，或点击 **Choose API Models…** 从当前服务获取列表。
   请选择支持文本生成及所选 API 模式的模型。选择器中的 **Done** 返回设置，选择仍为草稿。
5. 默认启用 **Use Responses API**；只支持 Chat Completions 的服务关闭此选项。
   **Send temperature parameter** 也按服务独立设置，默认关闭。
6. 点击 **Apply OpenAI Services**（或设置页的 Save）保存全部草稿。
   聊天模型菜单显示 `服务名称 · 模型ID`；不同服务可以配置相同模型 ID，路由不会混淆。
   **Choose API Models…** 使用当前服务草稿中的地址与密钥读取列表，不会顺带应用其他配置。

模型列表和生成请求使用同一服务的 Base URL。远程地址要求 HTTPS；仅 `localhost`、
IPv4 `127/8` 和 IPv6 `::1` loopback 服务允许 HTTP。地址不能包含用户名、密码、
查询参数或片段；任何服务的无效草稿都会阻止应用，不会写入偏好设置或改动密钥。
服务必须提供最终 API 根地址：请求不跟随 HTTP 重定向，避免转发认证或聊天内容。

API Key 仅保存在 Keychain，不写入 UserDefaults、聊天数据库、导出会话或源码。
每个新服务的密钥按固定服务 ID 和规范化 Base URL（含端口及路径）隔离：
即使两个服务使用同一地址，也不会共享密钥。切换地址立即清空当前密钥框；
离开地址框、按回车或应用设置时，才读取该服务在目标地址的已存密钥，避免每次按键访问 Keychain。
若已为目标地址输入新密钥，读取过程不会覆盖该输入；域名大小写、默认端口和尾部斜杠会规范化。
无密钥的兼容请求不发送 Authorization，服务器错误消息中的完整当前密钥会被脱敏。
读取密钥失败时，不会因未修改的空输入框删除原密钥。

旧单服务配置在首次成功保存时迁移为 **OpenAI**，沿用原来官方地址的 `openai`
或自定义端点的 Keychain 项，保留 API 模式、模型列表、选中模型及旧会话身份。
服务改名不会改变模型身份；更换同一服务的 Base URL 后，该服务的会话使用新地址。
若要同时保留两个目的地，应新增独立服务。
**Remove Service** 后应用设置会移除该服务的配置及其所有端点密钥，保留历史会话。
停用或删除服务后，其旧会话请求明确报错，不会自动发送给其他服务。
全部服务都可删除；新增服务不会接管旧会话。

OpenAI 兼容服务、本地模型与 Gemini 可同时使用。旧会话仍保留其提供方和服务身份。

主聊天和快捷补全均支持多轮文本、流式回答、最终用量和取消。
Responses 请求使用 `store: false` 并携带本地消息历史；达到输出上限时保留部分回答、
用量和 `length` 结束原因。默认不发送 temperature，按模型兼容性手动开启。
当前远程入口提供文字对话；图片生成、图片理解和语音使用各自已适配的本地模型功能。
选择 OpenAI 模型会把会话文本发送到配置的端点，调用费用和数据处理由该服务及用户账户决定。

协议与认证依据：[OpenAI Docs API 概览](https://developers.openai.com/api/reference/overview)、
[Responses 流式接口](https://developers.openai.com/api/docs/guides/streaming-responses)。

## App 调用 Google Gemini

设置 → **Gemini** 标签：

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
OpenAI 密钥、模型设置不会迁移或发送给 Google。旧 OpenAI 会话保留，
启用并配置 OpenAI 后可选择对应模型继续；已停用的提供方请求会明确报错。
Gemini 内部模型 ID 带 `gemini:` 前缀，请求中发送原始模型 ID。

主聊天和 macOS 快捷补全均按所选模型路由。选择 Gemini 后会发送会话文本至 Google；
失败不会自动回退，停止回答会取消网络请求。
本次迁移保持原远程入口的文本范围，不启用图片、工具调用或音频。

## 向其他程序提供本地模型 API

设置 → **Local API** 标签 → **Local MLX API**：

1. 选择端口，默认 `8080`。
2. 点击 **Generate Access Token**，成功后显示 **New access token saved in Keychain.**。
   每次生成新的 64 位十六进制随机令牌；掩码长度相同，不能据此判断令牌没有变化。
   可用 **Show Token / Hide Token** 查看或隐藏，用 **Copy Access Token** 配置客户端。
   这是本地服务的密钥，与 OpenAI / Gemini API Key 独立。
3. 点击 **Start Local API**。必要时通过目录选择器授权 HF_HOME。
   运行或启动中不能替换令牌，需先 **Stop Local API**。已运行时仍可查看和复制。

生成的令牌立即保存到 Keychain，保存失败会显示错误并保留原输入值。
手动填写的令牌至少 16 个字符，不能含内部空白或控制字符；在启动服务时校验并保存。
令牌不会写入偏好设置、源码或日志。

人工验证：停止 API → 生成并确认成功提示 → 显示/复制令牌 → 再次生成，确认内容变化 →
重开设置确认保留最后生成的令牌。随后启动 API，用当前令牌访问 `/v1/models`；
缺少令牌或使用旧令牌应得到 HTTP 401。请勿在反馈中提供实际令牌。

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

仅暴露本地 MLX，不代理云端模型，也不产生 OpenAI / Gemini API 费用。
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
  --configuration Release --cache ~/.cache/huggingface/hub --chat-api

python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --cache ~/.cache/huggingface/hub --chat-store
```

`--chat-api` 使用真实缓存模型和真实 loopback TCP/HTTP 测试本地模型列表、认证、
错误响应、普通/流式推理、stop、用量、客户端断开取消及监听器重启。
OpenAI Responses/Chat Completions 客户端使用受控 HTTP 夹具验证请求格式、中文 SSE、
多行事件、CRLF、401/429、生成失败、断流和取消；不访问真实账户或读取个人 API Key。
Gemini 回归额外验证 Google 端点、独立配置和密钥账户、消息历史与最终用量。
OpenAI 回归验证官方默认地址、Responses 默认值、Chat Completions 切换、独立密钥账户、
输出上限用量，以及开关和模型列表与 Gemini 的隔离。Base URL 回归额外验证地址校验、
端点切换与密钥隔离、Keychain 读取失败时不删除原密钥、无效草稿不落盘和错误密钥脱敏。
真实 loopback HTTP 夹具覆盖自定义路径下的模型列表、两种流式协议、最终用量、
有/无 Bearer 认证，并验证 301/302/303/307/308 重定向的目标收到零请求。
多服务回归检查旧配置迁移、重载、同地址不同密钥、同名模型、服务改名和停用/删除后不回退；
两台 loopback HTTP 服务使用不同密钥和 API 模式，验证生产路由发送原始模型 ID 到正确端点。
这些远程协议检查使用 HTTP 夹具；真实账户调用需在 App 设置中配置各自的 API Key 后验证。
仅运行协议测试可使用 `--remote-api-only`，需要 loopback 网络权限，无需本地权重或个人密钥。
`--remote-store-only --cache <HF缓存>` 独立验证 OpenAI / Gemini 模型选择和旧会话保留，
只扫描缓存，不加载权重推理；还验证服务名称显示、同名模型身份、主聊天/快捷补全路由与删除后的会话保留。

`--chat-store` 使用内存 SwiftData 和独立设置验证旧会话保留、主聊天/快捷补全的真实 MLX 推理、
停止后重启，以及混合模型列表、远程模型选择和远程调用禁用时不会回退至本地模型。
还检查同名远程模型的提供方隔离、HF_HOME 不可用时仅使用远程模型，以及关闭提供方后不回退。
这些测试共用一个测试可执行文件，请顺序运行。

远程协议依据：[Google Gemini 兼容接口](https://ai.google.dev/gemini-api/docs/openai)。

缓存中兼容的 Qwen OptiQ 模型在请求 `temperature: 0`、App 设置启用
**Use Local Qwen MTP** 时自动使用本地 MTP sidecar。其他温度保持普通采样；
缺失 sidecar 时使用主模型，最终 `usage` 使用真实生成 token 数。
详细格式与限制见 [MLXModelLibrary.md](MLXModelLibrary.md)。

## 人工验证设置与真实兼容服务

以下步骤由使用者在 App 中操作；合成 HTTP 回归不等同于真实服务通过。

1. 退出并重新打开新构建。设置应有 General、Models、OpenAI、Gemini、Speech、Local API 六页；
   各页可独立滚动，Cancel / Save 固定在顶部。
2. 在 OpenAI 页配置服务 A，再用 Add Service 配置服务 B。为两者设置不同名称、
   各自的 Base URL 和 API Key，并添加同一个真实可用的文本模型 ID。
   Base URL 填 API 根地址，不带 `/responses` 或 `/chat/completions`；仅支持 Chat Completions 的服务关闭 Use Responses API。
3. 确认名称和地址可以全选替换、粘贴和逐字修改，API Key 显示为掩码。切到其他页再回来，
   草稿应保留。保存并重新打开设置，服务及各自参数应恢复；错误原因应在窗口中显示。
4. 在聊天模型菜单分别选择 `A · 同名模型`、`B · 同名模型`，各新建会话。
   先要求“记住代码 PINE-472，并从 1 数到 40”，确认文字逐步出现；
   再问“上轮的代码是什么？只回复代码”，应得到 PINE-472。
5. 对需要认证的服务，若可恢复原密钥，可将 B 的密钥临时替换成无效测试串并应用：
   应只有 B 认证失败，A 仍能推理；随后恢复 B 并确认成功。不要将 A 的真实密钥发送给 B。
6. 改动 A 的 Base URL 时，旧地址密钥应立即清空，不应带到新目的地。
   使用 Cancel 放弃此草稿，再打开应恢复原设置。不要将未知测试地址应用到真实会话。
7. 停用 B 并应用，B 的旧会话应明确报错，A 继续正常；重新启用 B 后恢复。
   反馈结果时只需说明服务代号、模型和失败步骤，不要提供密钥或含凭证的日志。
