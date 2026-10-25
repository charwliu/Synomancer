# Synomancer · 思诺曼

以 SwiftUI 和 Swift + MLX 构建的 Apple Silicon 本地 AI 客户端。
聊天、图片生成与编辑、语音识别和朗读在设备上运行；也可选用 Google Gemini 文本模型。

## 功能

- 本地多轮聊天、流式回答、思考区、图片理解及 macOS 快捷补全。
- Muse Glimmer MXFP8、受支持的 Qwen/Gemma 模型，以及原生 Qwen GGUF 推理。
  模型兼容性以[聊天文档](docs/LocalChat.md)和[模型库说明](docs/MLXModelLibrary.md)为准。
- FLUX / Qwen 图像生成与编辑，Qwen3-ASR 和 Qwen3-TTS 本地语音管线。
- 原生 Hugging Face 模型下载，支持续传、分片校验及共享 HF 缓存。
- 可选 Gemini 文本聊天，以及仅绑定 loopback、需要访问令牌的本地模型 API。

不同模型需要不同的内存容量、权重格式和访问权限；下载完成不代表该架构已受支持。
仓库不包含模型权重、个人配置、访问凭证、录音或验收日志。

## 构建

当前 macOS target 要求 Apple Silicon、macOS 27 和具有相应 SDK 的 Xcode。
使用 Swift 6；在 Xcode 中打开 `Synomancer.xcodeproj`，选择 **Synomancer-Release**。
签名和开发团队在自己的 Xcode 设置中配置。

无需签名的本机 Release 构建：

```sh
git clone git@github.com:charwliu/Synomancer.git
cd Synomancer
xcodebuild -project Synomancer.xcodeproj -scheme Synomancer-Release \
  -configuration Release -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  CODE_SIGNING_ALLOWED=NO build
```

首次构建需要解析和下载 Swift Package 依赖。构建产物位于
`.build/DerivedData/Build/Products/Release/Synomancer.app`。
`Libraries/` 中包含本地包及历史适配；当前聊天模型库来自 Xcode 固定的 SPM 依赖，
不要把历史 MLX 目录当作当前运行时入口。

## 初次使用

1. 在设置 → Hugging Face 中选择 **HF_HOME** 并授权访问。
   未配置时使用当前用户主目录下的 `.cache/huggingface/hub`；可自行选择其他磁盘。
2. 下载兼容模型，或选择已有的完整 HF 缓存，刷新模型列表后选择模型。
3. Gemini API Key 在设置中输入并保存在 Keychain；HF Token 仅供本次下载使用。
   不要将凭证写进代码、配置示例、日志或提交信息。

本地模型推理不需要外部推理服务。Hugging Face 下载会访问网络；启用 Gemini 后，
远程请求会把会话文本发送到 Google。

## 文档与测试

| 内容 | 文档 |
| --- | --- |
| 本地聊天与 Muse Native | [LocalChat.md](docs/LocalChat.md) |
| MLX 模型库和量化适配 | [MLXModelLibrary.md](docs/MLXModelLibrary.md) |
| 图像生成与编辑 | [LocalImages.md](docs/LocalImages.md) |
| HF 下载与缓存 | [HuggingFaceDownloads.md](docs/HuggingFaceDownloads.md) |
| Gemini 与本地 API | [ChatAPI.md](docs/ChatAPI.md) |
| 语音识别 | [LocalSpeechRecognition.md](docs/LocalSpeechRecognition.md) |
| 朗读 | [LocalTextToSpeech.md](docs/LocalTextToSpeech.md) |

先构建 Release，再按各文档运行 `Tests/` 下的独立测试。多个聊天/图像测试共用
`SynomancerASRSmoke`，须顺序运行。合成检查不能代替真实模型推理；后者需要
兼容硬件和完整权重。输出、临时数据库及测试日志统一保存在忽略 Git 的 `.build/`。

贡献约定见 [AGENTS.md](AGENTS.md)。各本地包及第三方适配保留各自的许可与来源说明。
