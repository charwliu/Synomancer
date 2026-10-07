# 本地 MLX 聊天

主聊天和 macOS 快捷补全面板直接使用 Swift + MLX 加载本机模型。
App 不再连接 Ollama，不再请求服务器模型列表、生成接口或健康检查。
OllamaKit 已从 App 的包依赖和链接目标中移除。

设置 → Hugging Face → **HF_HOME** 与 ASR/TTS 共用，默认：
`/path/to/workspace/.cache/huggingface/hub`。
使用 **Choose HF_HOME Folder…** 授权缓存根目录，然后点击
**Refresh Models** 或 **Save**。可在 **Chat Models → Default Model**
指定默认模型，也可在聊天模型菜单中切换。

扫描标准 HF 缓存父目录、hub、仓库目录、snapshot 和导出的模型目录。
只有本项目模型工厂支持的架构、完整 tokenizer 和权重分片会出现在列表中。
优先使用 refs/main 指向的完整 snapshot，其他完整 snapshot 作为备选。
不下载模型、不复制或改写缓存；模型通过本地 directory 配置加载。
默认选择权重体积较小的缓存模型，后续复用已加载的模型。

当前本机默认模型为 `mlx-community/Qwen2-VL-2B-Instruct-4bit`，使用其文本能力。
也发现了 DeepSeek-R1-Distill-Qwen-32B 的 4bit/8bit 缓存。
模型库已升级为官方 mlx-swift-lm 3.32.3，版本和适配说明见 [MLXModelLibrary.md](MLXModelLibrary.md)。
仅下载权重不会增加未实现的架构或模型扩展支持。
超过本机大部分物理内存的权重在加载前拒绝，避免误选超大模型导致分配失败。

现有 Qwen2-VL 视觉实现未通过颜色识别测试，因此该模型暂时不开放图片输入。
该本地模型的图像输入会在推理前报明确错误。

可选的 OpenAI 远程模型和本地 OpenAI 兼容 API 见 [ChatAPI.md](ChatAPI.md)。

旧会话和它们关联的旧模型记录保留。打开旧会话时，模型选择器回退到本地模型；
发送下一条消息时，继续使用原有消息历史并更新该会话的模型关联。
枚举中的旧 provider 值只用于解码历史记录，没有对应运行后端。

普通回答每四个 token 更新一次 UI，MTP 按上游文本 chunk 更新，完成时保存完整输出，停止时保留已生成文字。
请求标识防止旧任务回调修改新回答，取消按 token 步进检查。
主聊天和快捷补全共享一个模型服务，生成串行执行，模型仅在切换或缓存设置变化时重载。

Qwen 的提示模板可能已经包含 `<think>`，生成内容只有 `</think>`。
App 根据实际渲染的提示模板补全这一开头，流式思考和正式答案分开显示；
思考内容默认折叠，可展开查看。停止在思考阶段的消息也保留在折叠区。
旧记录中单独成行的 `</think>` 同样按思考结束解析，无需修改数据库或删除会话。
复制、选取文字、朗读和快捷补全只使用正式答案。
Markdown 代码块、行内代码和转义内容中的示例标签保留原样。
本地 HTTP API 继续返回原始模型输出；App 补全的标记不计入生成 token 用量。

## 验证

聊天请求会在加载模型权重之前验证本地分词器的聊天模板，并复用已经解析的分词器。
`gemma-3-27b-pt-bf16` 是预训练基座模型，官方不提供聊天模板，不能直接作为指令聊天模型使用。
请选择同系列的 `-it` 模型；缺少模板的指令缓存需要补全官方模板文件。
App 不会给基座模型注入指令模板，也不会修改 HF 缓存或自动下载替代模型。
模板格式回归可运行下面测试命令，将 `--chat` 替换成 `--chat-tokenizer`。

使用 `Synomancer-Release` scheme 构建后：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat

python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat-store
```

`--chat` 执行真实缓存模型推理，验证多轮上下文和取消后立即重启。
可用 `--model owner/repo` 指定缓存模型。
`--chat-store` 使用内存 SwiftData 数据库和独立 UserDefaults，验证旧会话保留、
主聊天完整回答保存、停止后重启及快捷补全，不访问用户聊天数据库。
`--chat-reasoning` 使用真实 Qwen3.8 27B 缓存，验证普通/MTP 流式思考分离和最终答案。
`--chat-moe` 使用 `Qwen3.5-122B-A10B-oQ4-MTP` 真实缓存验证内嵌草稿参数排除后，
普通生成和 App 思考分离都能得到完整答案；该 MoE 的 MTP 开关回退到普通生成。
`Tests/AssistantMessageContentTests.swift` 覆盖历史孤立标签、标签分块和 Markdown 字面值。
不要同时运行这些测试：它们共用测试可执行文件，性能测量也应独占推理任务。

`Tests/LocalChatCacheTests.swift` 是无需模型的缓存完整性测试，覆盖缺失分片、
断开的 blob 符号链接、未支持架构和缺失视觉处理器配置。

Qwen3.8 OptiQ 的 swish/MTP 适配和设置见 [MLXModelLibrary.md](MLXModelLibrary.md)。
主聊天的 **Local Chat Temperature** 可设为 0 启用原生 MTP；其他温度保持普通采样。
