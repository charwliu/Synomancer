# 本地 MLX 聊天

主聊天和 macOS 快捷补全面板直接使用 Swift + MLX 加载本机模型。
App 不再连接 Ollama，不再请求服务器模型列表、生成接口或健康检查。
OllamaKit 已从 App 的包依赖和链接目标中移除。

## Muse Glimmer 30B MXFP8

`mlx-community/Muse-Glimmer-30B-mxfp8` 使用 App 内的 Swift + MLX Native 后端，
支持多轮文本对话和图片理解。复用固定的 mlx-swift-lm 3.32.3 中的
`MuseGlimmer` / `MuseGlimmerProcessor`，按原始 MXFP8（8-bit、group size 32）
加载，无需转换权重、Python 运行时或外部推理服务。

在设置 → Hugging Face → 下载 HF 模型中点击 **Muse Glimmer 30B MXFP8 · Native**，
再检查模型并下载到 HF_HOME。仓库权重约 32.5 GB；完整的 150 个权重分片、
分片索引、tokenizer、聊天模板和 `processor_config.json` 就绪后，刷新聊天模型，
选择 `mlx-community/Muse-Glimmer-30B-mxfp8`。不完整的缓存不会出现在模型列表。
模型来源：[MLX Community](https://huggingface.co/mlx-community/Muse-Glimmer-30B-mxfp8)。

App 在加载权重之前验证 ATEM 控制 token。生成时按 token ID 解析消息边界，
隐藏 assistant/recipient 头和工具帧；主聊天的 `to=self` 内容映射到现有可折叠思考区，
API 默认只输出 `to=user` 的正式答案。多轮重放使用正式答案，避免把思考区当作
普通 assistant 内容。token 用量保留真实生成数量，包含协议和思考 token。
工具帧只隐藏，不执行工具调用；当前不加载 Muse assistant drafter，使用普通解码。
图片沿用 448 像素预算，由原生 processor 对齐 patch/merge 网格；暂不复用提示前缀缓存。

Release 构建后执行以下回归（顺序运行，测试可执行文件与其他聊天测试共用）：

```sh
python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release --muse-components

python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat-muse
```

`--muse-components` 不需要下载权重：检查 ATEM 分帧、思考、Unicode、字面标记、
无效 tokenizer、合成 MXFP8 checkpoint 的严格加载和 logits 一致性，以及图片
patch/token 数量和缓存发现。它不代表真实 30B 模型推理通过。
`--chat-muse` 还要求完整真实缓存，检查多轮口令、最终流式用量、红/蓝两种图片、
生成回答的下一轮历史重放、停止字符串、取消和立即重启；缺失权重时失败，
不会跳过真实推理。设置 `SYNOMANCER_MUSE_IMAGE=/absolute/path/to/red-mug.png`
可增加红色杯子测试图的真实描述检查。测试为模板默认的高思考模式预留 1024
个生成 token，包含隐藏的思考和协议 token。

只验证真实 tokenizer/模板时，可在 `--muse-components` 前设置
`SYNOMANCER_MUSE_TOKENIZER_DIR=/path/to/snapshot`，目录须包含完整的
`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和
`processor_config.json`。该检查不加载 30B 权重。

2026-10-21 本机真实验证通过：固定 revision
`205eb41208cca89cd6eb9409ac0c2cc1fbbe1548` 的全部 150 个分片
（32,503,464,693 字节）逐片 SHA-256 与下载清单匹配。以下结果来自 App 的
`LocalChatService`，使用完整 30B MXFP8 权重和 Metal，未连接外部推理服务：

- 多轮口令返回 `REDWOOD-472`；将真实生成的 assistant 回答重放后，追问返回 `472`。
- ATEM 思考区和正式答案正确分离，最终流式文本及 token 用量一致；口令生成
  239 token，本机两次实测约 13.6–15.3 token/s（包含思考和协议 token）。
- 红、蓝两张图片分别回答 `red`、`blue`。红杯测试图描述为：
  “A glossy red ceramic mug sits on a white surface against a plain light gray background.”
- `WORLD` 停止字符串与完整回答的前缀一致；流式生成中取消后立即重启成功。

完整日志位于 `.build/MuseNativeReal.log`，分片复核结果位于
`.build/MuseSHA256Verification.json`。此前的 Release 构建、合成 MXFP8/ATEM/
图片预处理、真实 tokenizer/模板检查，以及现有 Qwen3.8 27B 回归也已通过。

## 本地缓存与聊天

设置 → Hugging Face → **HF_HOME** 与 ASR/TTS 共用，默认：
`/path/to/workspace/.cache/huggingface/hub`。
使用 **Choose HF_HOME Folder…** 授权缓存根目录，然后点击
**Refresh Models** 或 **Save**。可在 **Chat Models → Default Model**
指定默认模型，也可在聊天模型菜单中切换。

扫描标准 HF 缓存父目录、hub、仓库目录、snapshot 和导出的模型目录。
只有本项目模型工厂支持的架构、完整 tokenizer 和权重分片会出现在列表中。
优先使用 refs/main 指向的完整 snapshot，其他完整 snapshot 作为备选。
不下载模型、不复制或改写缓存；模型通过本地 directory 配置加载。
权重索引支持 `optiq/optiq_vision.safetensors` 等相对子目录路径；所有引用的
分片都必须完整可读，拒绝绝对路径和 `..` 等路径段。HF blob 符号链接仍受支持。
默认选择权重体积较小的缓存模型，后续复用已加载的模型。

当前本机默认模型为 `mlx-community/Qwen2-VL-2B-Instruct-4bit`，使用其文本能力。
也发现了 DeepSeek-R1-Distill-Qwen-32B 的 4bit/8bit 缓存。
模型库已升级为官方 mlx-swift-lm 3.32.3，版本和适配说明见 [MLXModelLibrary.md](MLXModelLibrary.md)。
仅下载权重不会增加未实现的架构或模型扩展支持。
超过本机大部分物理内存的权重在加载前拒绝，避免误选超大模型导致分配失败。

现有 Qwen2-VL 视觉实现未通过颜色识别测试，因此该模型暂时不开放图片输入。
该本地模型的图像输入会在推理前报明确错误。

可选的 Gemini 远程模型和本地 OpenAI 兼容 API 见 [ChatAPI.md](ChatAPI.md)。

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

## 流式输出与速度显示

已加载的同一模型复用发现结果和配置校验；模型/缓存配置切换时重新发现
和加载。普通生成使用上游按换行分段的 StreamingDetokenizer，避免每次
刷新解码完整回答。首 token 立即通知，之后至少积累 4 token 且间隔
100 ms 才刷新；结束时强制发送完整的权威解码结果。带 stop 字符串的请求
仍逐 token 完整解码以保留精确停止语义。MTP 的界面更新及用于临时计数的
重新分词也限制到约 10 Hz，最终计数仍取 worker 的实际 token ledger。

本地助手回复下方常驻显示速度：
- 生成中 `≈ … tokens/s` 是首次流式更新之后的平均估算，不包括加载/预填充；
  MTP 的临时计数来自分词器，因此不是草稿接受率或精确吞吐统计。
- 完成后显示推理引擎提供的 tokens/s 和输出 token 数，包含思考 token；
  可选 SwiftData 字段保留统计，旧消息显示为空，取消时清除临时估算。
- 远程 Gemini 请求暂不显示此指标，避免将网络/首字等待时间与本地指标混用。

除了减少目录扫描、CPU 解码/分词和界面刷新开销，普通文本生成现在还复用
Qwen3.5/3.8、Gemma-4 的提示前缀快照，不改变权重精度或用户采样参数。

## 提示前缀缓存

- 仅保存一份生成前的前缀快照及模型位置状态，不缓存生成中的回答。
  Qwen 前缀长度按 256 token 对齐，末尾至少 128 token 每次重算；Gemma-4
  按 128 token 对齐，末尾至少 32 token 重算。仍逐 token 验证前缀，模板尾部
  变化时只在确实匹配的范围内复用。少于 384 token 时不建立快照。
  Qwen 的更细边界实验未通过回答正确性回归，因此保留已验证的边界。
- 每次先渲染完整聊天模板，再逐 token 检查快照是否为新提示的严格前缀。
  匹配时只计算剩余部分；不匹配则重建。不会回滚或截断 Qwen 的循环状态。
  修改末尾消息仍可复用未变的前缀，修改系统提示等前缀内容则失效。
- 快照使用上游 cache.copy() 并在启动生成 worker 前求值，取消生成不会修改
  已保存的快照。切换模型、HF_HOME、使用图片或进入 MTP 路径时清除。
- 缓存张量逻辑大小上限为物理内存的 1/64，最多 2 GiB；MLX 分配器、模型
  权重和当前生成的工作缓存另计。第一次建立快照有额外边界计算和保存成本。
- MTP 仍走原有路径，不复用此缓存；Qwen 温度 0 且有兼容 drafter 时通常进入 MTP。
  当前适用架构为 qwen3_5、qwen3_5_text、gemma4、gemma4_text 的纯文本输入。
  本机实测覆盖 Qwen3.8 OptiQ 4bit 与 Gemma-4 31B 8bit。
- LocalChatResult.cachedPromptTokens 提供实际命中数，promptTokens 仍是完整输入
  数量，promptSeconds 包含本次新建/延长快照的时间。界面的 tokens/s 仍衡量
  输出吞吐，前缀缓存主要缩短首 token 等待。

`--chat` 对上述架构额外验证系统提示变化、取消后重试及前缀扩展。
缓存与禁用缓存都必须完成正确答案；同一快照重复、取消重试和扩展后重复
必须逐 token 一致。缓存划分改变浮点运算批次；
Qwen 长续聊样例出现了不同的思考措辞，不保证所有提示与禁用缓存时逐 token 相同。

首轮前缀缓存基线（两种模型均为 256/128 边界）：
2026-10-07，M4 Max 128 GiB，Release，模型已加载的续聊样例，
对照为同一服务显式关闭前缀缓存；首 token 耗时包含准备输入和本次预填充：

| 模型 | 命中 / 输入 token | 无缓存 | 命中缓存 |
| --- | --- | --- | --- |
| Qwen3.8-27B-Uncensored-OptiQ-4bit | 1024 / 1273 | 7.11 秒 | 1.75 秒 |
| Gemma-4-31B-it-MLX-8bit | 1024 / 1203 | 8.47 秒 | 1.54 秒 |

这是固定测试样例的单次对照，不是所有请求的提速保证；第一次请求、前缀失配、
MTP 和图片请求不会得到这项命中收益。该表衡量首 token 等待，不是解码 tokens/s。

2026-10-07，M4 Max 128 GiB，Release 本地短提示多轮记忆回归：
Qwen3.8-27B-Uncensored-OptiQ-4bit 约 25.5 tokens/s；
Gemma-4-31B-it-MLX-8bit 约 14.1 tokens/s。它们是该样例的运行读数，
不是前后提速对比，也不代表长回答、长上下文或其他量化版本。
当时 Gemma-4 OptiQ 4bit 本地目录缺少处理器配置，未参与推理测试。
2026-10-08 已从其 Google QAT 基础模型补齐 `processor_config.json`。
测试覆盖最终流式刷新、有效速度统计、取消/立即重启，以及真实 Store 的
统计保存、实时估算与取消清理；Store 测试使用隔离数据库和模拟朗读回调。

## 预填充分块对照

`SYNOMANCER_BENCH_PREFILL=1` 配合上述 `--chat --model owner/repo` 命令，
在模型预热后以 512 → 1024 → 1024 → 512 顺序关闭前缀缓存测试，并检查完整答案。
M4 Max 128 GiB、Release、约 1,800 token 提示，首 token 耗时均值：
Qwen 512/1024 为 8.20/7.75 秒，Gemma 为 9.55/9.99 秒。单组耗时范围重叠，
未证明稳定收益，因此保留上游默认分块，不自动扩大为 1024。

`SYNOMANCER_BENCH_PREFIX=1` 则固定 512 分块，对同一长提示连续运行三次，
第一轮建立快照，后两轮报告命中数量和首 token 耗时；每轮都检查最终答案。

Gemma-4 31B 8bit 的旧/新边界直接对照（1,759 token，固定 512 分块）：
旧边界命中 1,536 token，两次热请求首 token 为 1.254/1.266 秒；
新边界命中 1,664 token，为 0.615/0.614 秒。两组冷请求分别为 8.733/8.781 秒，
最终答案均为预期口令。此固定样例热请求约缩短一半，不能推广为所有请求的
固定提升，也不是解码吞吐翻倍。较细边界在未多命中 token 时不保证更快。

## 长回答的界面解析

NativeMarkdownView 保存仅属于当前视图的解析缓存：代码围栏外的完整空行
之前的块可以复用，正在生成的尾部继续解析。使用 UTF-8 严格前缀检查，
编辑、截断或分词器修正之前文字时重新解析；代码块中的空行不会提前提交。
没有空行的长段落或未闭合代码块仍需要解析整个未完成的尾部。

每个 Markdown 块通过 Equatable 视图跳过未变化内容的行内格式解析；
折叠的思考区只在展开时构建 Markdown 内容。不会建立全局聊天文本缓存。

2026-10-07，M4 Max，Swift 优化构建：22,560 字节的混合 Markdown，346 次
流式快照，ABBA 顺序完整解析为 136.4/134.1 ms，增量解析为 8.55/7.88 ms。
这是块级解析的 CPU 时间，未包含行内样式、SwiftUI 布局或模型推理，不能据此
推算 tokens/s。测试包含 682 次流式/编辑对照、Unicode 规范化及原有块解析回归。
实际 NSHostingView 验证浅色/深色的完整 Markdown、未闭合代码、折叠思考和
思考中状态，共 8 组与优化前逐像素一致（含滚动区域的代码与表格文字）。

独立解析回归及基准：

```sh
xcrun swiftc -O -module-cache-path /private/tmp/SynomancerMarkdownModuleCache \
  Applications/Synomancer/UI/Shared/Chat/Components/ChatMessages/NativeMarkdownParser.swift \
  Tests/NativeMarkdownParserTests.swift -o /private/tmp/SynomancerMarkdownTests
/private/tmp/SynomancerMarkdownTests
```

## Native Qwen4 GGUF / IQ3_S

`SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` now has a dedicated
Swift + MLX implementation in `Services/GGUF`. It reads the two original IQ3_S
shards without converting them to affine MLX quantization. GGUF is a container:
this checkpoint actually mixes ten quantization formats. Packed Metal kernels
handle Q8_0, Q6_K, Q5_K, Q4_K, Q2_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS and IQ2_S.
The decoder includes Qwen4 hyper-connections, PLE n-gram hashing, GatedDeltaNet,
QSA and routed/shared experts. The projector uses a native Swift Qwen3-VL vision
tower and multimodal rotary positions.

Inference runs inside Synomancer. No Python interpreter, llama.cpp runtime,
external service, port, or API key is involved. The Metal layout definitions are
adapted from MIT-licensed llama.cpp commit
`8a1a9b5126126e5228b95fa909d4b08fac65e8b3`; model equations follow MIT-licensed
mlx-vlm commit `c24454765c3177a385185c185b4957567c5fdf89`. The Swift vision tower is
adapted from mlx-swift-lm 3.32.3. License notices are retained in the source files.

An exported directory needs the official `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `preprocessor_config.json`, and `synomancer-gguf.json`:

```json
{
  "firstShard": "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
  "projector": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "mtp": "mtp-Qwen3.8-Flash-Next-Q8_0-complete.gguf"
}
```

Asset paths are relative and may contain subdirectories. Both shards must exist.
The projector is optional; without it the model is text-only. HF blob symlinks
are supported. Shape and tensor completeness are checked before allocating
weights. `Tools/prepare-qwen4-gguf.sh tokenizer-assets gguf-snapshot export-directory`
creates an offline export by copying metadata/tokenizer files and linking the
existing weights. An optional fourth argument copies a complete MTP sidecar into
the export and adds its relative `mtp` path. It refuses to overwrite an existing export.

The prepared local entry is `Synomancer/Qwen3.8-Flash-Next-IQ3_S` under the selected
HF cache. Refresh models in Synomancer and select it. PLE embeddings are read by
row from the mapped GGUF, with a packed-row LRU capped at 16 MiB and 8,192 rows. Routed expert weights are
also mapped, with a global packed-buffer LRU controlled by **Flash-Next 专家缓存**
in Chat Models settings (default 16 GiB; capped at a quarter of physical RAM,
up to 64 GiB). A budget of 0 disables retention, not expert computation. Changes
apply on the next request and reload the target model. The loader requires at
least 32 GiB; the tested machine has 128 GiB, so this is not a capacity guarantee
for smaller machines or long/image contexts. Switching to another local chat model
releases these weights. Generation supports history, image attachments,
temperature/top-p, stop strings, streaming, cancellation and restart.

Native MTP uses the complete 37-tensor Q8_0 sidecar, including the repaired
`blk.48.nextn.hc_head_{norm,down,up}.weight` tensors. The original incomplete
sidecar is rejected before allocating draft weights. Set **Local Chat Temperature**
to **0** and enable **MTP**; missing sidecars, other temperatures, or a disabled
MTP setting use ordinary decoding with an explicit fallback reason. The local
export already includes the complete sidecar. Only target-verified draft tokens
are emitted. On rejection, the target commits the accepted prefix by trimming
attention/index caches, selecting the saved PLE state, and recomputing only GDN
recurrence from retained verifier projections. It does not rerun embeddings,
attention or experts. Draft KV is trimmed separately. After warmup (two rounds
for short prompts, eight for bulk prompts), up to three drafts are used when
at least 65% of the most recent 32 rounds accepted the first draft; otherwise
only one is proposed. Generation limits, stop strings and cancellation apply
to emitted tokens. The final permitted pending token needs no extra forward pass.

Long prompts use chunks of up to 1,024 tokens (512 for prompts below 1,024 tokens).
Dense projections reuse lazily
materialized FP16 weights; routed experts use a packed grouped Metal GEMM without
expanding the full expert weights. QSA applies per-query causal block selection,
and GDN/PLE carry state between chunks. Only the final prompt row is projected
to vocabulary logits. `prefillStepSize` can override the chunk size (1–1,024).
Short prompts default to 16-token chunks. For bulk prompts (128 tokens or more),
the last prompt token is processed separately, matching Python `generate_step`.
The draft initializes from the final target hidden row instead of reprocessing
the entire prompt; the target still attends to the complete prompt.

The native path caches converted floating-point matrices, GDN constants and
completed QSA index blocks. Packed GDN rows are restored to native head order
once at load; the output projection indexes its input without changing the
packed weight reduction order. Pointwise graphs are compiled, GDN uses grouped
MLX convolution, and decode submits logits/state asynchronously until sampling.
Singleton decode submits every two layers to reduce CPU/Metal scheduling overhead;
prefill and multi-row verification retain per-layer submission.
Large prefill chunks use tiled masked attention; short verifier/decode blocks
use indexed QSA beyond its 2,048-token dense-attention budget.
Eligible floating-point projections evaluate verifier rows together in a Metal
GEMV that preserves singleton reduction order. Narrow projections retain the
row-wise fallback; packed quantization and weight precision stay unchanged.
MoE decode graphs reuse captured weight references and receive activations as
explicit inputs. Routing is recomputed for every call; KV, convolution and
recurrent state remain request-owned. Bulk prefill retains grouped expert GEMM.

Current limits: prompt-cache reuse is disabled. Context is capped at 32,768
prompt + output tokens; images are resized to the chat service's 448-pixel
budget. Cold model loading is separate from prefill time.

Validation commands (after a Release build):

```sh
Tests/run-gguf-reader-tests.sh
Tests/run-gguf-tests.sh --kernels
Tests/run-gguf-tests.sh /path/to/first-shard.gguf "$PWD/Tests/Fixtures/qwen4-iq3s-greedy.json"
SYNOMANCER_GGUF_PREFILL=512 Tests/run-gguf-tests.sh /path/to/first-shard.gguf "$PWD/Tests/Fixtures/qwen4-iq3s-long.json"
SYNOMANCER_GGUF_MTP=/path/to/complete.gguf Tests/run-gguf-tests.sh /path/to/first-shard.gguf "$PWD/Tests/Fixtures/qwen4-iq3s-long-api-128.json"
Tests/run-gguf-tests.sh --integration /path/to/export /path/to/blue-right-handled-mug.png
```

The fixture records greedy token IDs from the independently verified MLX / Metal
baseline. Integration checks exercise the actual `LocalChatService`, including
vision, history, cancellation, retry, stop sequences and sampling. These are
real-model tests, not mocked inference.

Native optimization regressions can enable `SYNOMANCER_GGUF_MTP=/path/to/complete.gguf`
to compare MTP off, on and forced rejection against the same reference tokens.
`SYNOMANCER_GGUF_COMPARE_PREFILL=1` compares chunks 1, 512 and warm 512 in one
loaded process, then tests MTP with chunk 512 when a sidecar is provided.
The synthetic kernel test covers all ten quantization formats, both SIMD lane
widths, exact small-batch GEMV, duplicate expert routes and partial GEMM tiles.

Initial baseline after native MTP and batch-invariance fixes (same loaded M4 Max 128 GB process,
47-token prompt, 256 greedy output tokens), ordinary decoding measured **26.84
tokens/s**, MTP **47.36 tokens/s** (**1.76×**). It accepted 182 of 203 proposed
tokens (**89.7%**); all 256 output tokens matched the independent reference.
Small verifier blocks preserve singleton reductions for floating projections,
PLE convolutions and attention as well as packed quantized GEMV. The optional
`SYNOMANCER_GGUF_TRACE=1` test asserts per-layer hidden-state equality against
singleton decode before running the output regression.

For a 4,037-token prompt in one loaded process, token-at-a-time prefill took
**193.20 s**, 512-token chunks took **12.16 s** on the first pass and **11.18 s**
warm (**15.9–17.3× faster**). These exclude model loading; the first sequential
pass includes kernel warmup. All three runs matched the 32 checked reference
output tokens. MTP prefill took **11.53 s** and its 32-token output also matched,
as did forced rejection of an otherwise accepted proposal. This long-context
sample accepted only 14/26 draft tokens (53.8%): decode stayed around **20–21
tokens/s**, so MTP did not provide an additional decode speedup on this prompt.
Acceptance and acceleration depend on the prompt; full 32K parity is not claimed.

That baseline's service regression also passed history, vision MTP on/off text and
token-count equality ("杯身是蓝色的，把手在右侧。"), cancellation/restart, stop strings,
nonzero-temperature fallback and a one-token generation limit. The short first
vision response includes cold dispatch overhead and is not used as a steady-state
throughput benchmark. Release build and GGUF/cache-reader checks passed. Local
logs and `validation.json` are in `.build/GGUFTests/` (generated, not versioned).

Prefill layout matters for exact greedy comparisons. The original
`qwen4-iq3s-long-128.json` records Python's 512-step direct-model prefill.
`qwen4-iq3s-long-1024-128.json` records direct-model 1,024-step prefill with the
entire final chunk; Python and native agree on all 128 tokens, but differ from
the older fixture at token 35 (`Record` versus `Color`). Both are retained.

The current application uses the generation API layout: 1,024-step prefill with
the final prompt token separate, and final-hidden-row draft initialization.
`qwen4-iq3s-long-api-128.json` records this configuration and independent Python
MTP `generate_step` output. Native ordinary/MTP/forced rejection match all 128 tokens;
Python's repeated MTP runs also agree. Do not mix these distinct prefill layouts
when asserting parity. `SYNOMANCER_GGUF_DIAGNOSE_PARITY=1` remains a diagnostic
escape hatch only; all final checks run with strict reference assertions.

`SYNOMANCER_GGUF_REPETITIONS=3` repeats ordinary/MTP runs with fresh request
state in one loaded process. `SYNOMANCER_GGUF_EXTRA_FIXTURE=/path/to/fixture.json`
adds a second prompt without reloading weights. Decode throughput excludes
the first token supplied by prefill and excludes result printing. Cold load,
prefill, and decode must be reported separately; do not compare system-memory
pressure runs to an earlier uncontended baseline as evidence of a speedup.

2026-10-09 optimization measurements, M4 Max 128 GB, Release, original
mixed-quantization GGUF weights. The app uses Swift + MLX with no Python or
llama.cpp runtime. Cached MoE/GDN graphs, GPU draft chaining, prefix-state commit
and final-token early return are enabled. The adaptive policy uses first-draft
hits in the most recent 32 rounds (65% threshold), with two warmup rounds for
short prompts and eight for bulk prompts.

Each benchmark keeps weights loaded and resets request state. Warm means exclude
one initial run and include three subsequent runs; loading, prefill and decode
are separate, and printing is outside decode timing. Short results come from
`/tmp/qwen4-coalesced-short.log`; the standalone long run is
`/tmp/qwen4-coalesced-final.log`. Python runs are sequential, not concurrent.

| Measurement | Native Swift + MLX | Python MLX reference |
| --- | --- | --- |
| 47 input / 256 output, ordinary decode, warm mean | 46.42 tokens/s | 43.68 tokens/s |
| Same short prompt, MTP decode, warm mean | 83.31 tokens/s | 73.74 tokens/s |
| Same short prompt, ordinary prefill, warm mean | 0.423 s | 0.482 s |
| Same short prompt, MTP prefill, warm mean | 0.438 s | 0.493 s |
| 4,037 input / 128 output, MTP decode, warm mean | 56.29 tokens/s | 54.60 tokens/s |
| Same long prompt, MTP prefill, warm mean | 6.84 s | 6.92 s |
| Same long continuation, ordinary decode, warm mean | 42.90 tokens/s | 41.32 tokens/s (controlled continuation) |
| Same long continuation, ordinary prefill, warm mean | 6.24 s | 6.90 s (controlled continuation) |

These are individual benchmark sessions, not a universal speedup claim. Earlier
Python short runs measured 46.89 / 79.08 tokens/s ordinary/MTP: native ordinary
short decode is comparable (about 1% below that earlier session), while MTP is
faster in these samples. Native long MTP warm means varied from 54.59 to 58.02
across sessions. The initial native long prefill baseline above was 11.18 s warm.

Per-layer submission had intermittent long ordinary slowdowns to 31–37 tokens/s.
A forward/reverse submission-interval sweep and an alternating 1/2-layer test
retained exact outputs. With the draft resident, the alternating two-layer runs
measured 44.23, 43.93, 43.27 and 42.89 tokens/s; the last per-layer run fell to
38.73. The final default changes only singleton non-prefill submission to every
two layers. Its three warm ordinary runs measured 43.19, 43.10 and 42.42 tokens/s.
This supports the selected scheduling change; it does not prove the underlying
system variability has been eliminated. Cold loading and compilation remain
outside steady-state comparisons.

Python's ordinary generation API predicts `Record` at zero-based output index
35, whereas its MTP API predicts `Color`; native ordinary and MTP both predict
`Color`. This is a numerical output divergence, not full Python ordinary/MTP
parity. The ordinary comparison in the table feeds Python the same complete
128-token continuation used by native, still computing full logits and argmax
at every step. Only prediction 35 differs in all four Python runs. That benchmark
compares identical forward inputs, not unconstrained Python API generation.
The short Python comparison passes all reference tokens; its subsequent ordinary
long assertion failed at index 35 and is not counted as a passing full suite.

Native ordinary/MTP/forced rejection match all 256 short and 128 long fixture
tokens. Long MTP also matches Python's 135 proposals, 76 accepted tokens and 28
rejected rounds. Prefix-state commit is checked against full replay: 14 short
and 24 long rejection cases have exactly equal hidden states and every cache.
Synthetic GDN prefix lengths 1/2/3 also match, with input snapshots unchanged.
The final-token regression checks that returning the pending token does not
advance target history.

`SYNOMANCER_GGUF_MODE_ORDER=0,0,0,0,1,1,1,1,2` reproduces short benchmark order
and forced rejection; `SYNOMANCER_GGUF_LONG_MODE_ORDER` controls long order.
`SYNOMANCER_GGUF_CHECK_PREFIX_COMMIT=1` enables full-state comparisons in mode 2
only, excluding their overhead from performance modes. Graph comparisons use
`SYNOMANCER_GGUF_EXPERT_VARIANTS` with `plain`, `compiled`, `gdn`, `chain` or
`early`; do not combine these with layer tracing.
`SYNOMANCER_GGUF_PREFILL_CHUNKS=512,1024,1024,512` compares chunk sizes.
`SYNOMANCER_GGUF_SUBMISSION_STRIDES=1,2,1,2,1,2,1,2` compares singleton
decode submission intervals with strict token checks in one loaded process.
These switches are test harness controls.

Static projection caching and four-token drafts were tested and removed because
they did not improve throughput. Generated evidence is in
`.build/GGUFTests/optimization-progress.json`. Final strict-reference, kernel,
Release and service logs use `/tmp/qwen4-coalesced-{short,final,kernels,release,integration}.log`.
Release build and service integration passed history, visual MTP on/off output
and token-count equality, cancellation/restart, stop sequences, sampling fallback
and one-token termination. Quantized/floating kernel and GGUF reader/cache checks
also passed. The short vision request is a correctness check, not a steady-state
vision throughput benchmark.


### Bounded GGUF weight residency

The target's PLE table and routed gate/up/down expert tensors are excluded from
initial MLX allocation. Loading progress counts only resident tensor bytes.
Routing selects the same top-10 experts as the resident path; selected records
are compacted and IDs remapped without changing their order or quantization.
Each paged MoE result is evaluated before staging the next layer so deferred
GPU work cannot keep a full forward's evicted weights alive. The paged expert
path bypasses compiled graphs that capture weight arrays. Metal dispatch objects
are reused without retaining staged arrays. MTP's separate draft remains resident.

The expert budget covers retained packed MLX buffers only. Base weights, MTP,
vision, recurrent/KV state, temporary per-layer staging, and macOS clean file
pages are additional memory. The OS can reclaim mapped pages; a file mapping's
virtual size is not a physical allocation. This first implementation uses exact
routing and LRU, without speculative disk prefetch or CPU expert execution.
Lower budgets and cold file pages can substantially reduce throughput.

Expanded FP16 prefill matrices are released at the decode boundary and on request
completion/cancellation/error. The service also drains GPU work before cleanup
and clears the allocator's unused buffers; persistent packed expert cache entries
survive between requests.

Validation switches for `Tests/run-gguf-tests.sh`:
- `SYNOMANCER_GGUF_RESIDENT=1`: original fully resident weights, for comparison.
- `SYNOMANCER_GGUF_EXPERT_GIB=N`: paging with N GiB expert retention (default 16).
- `SYNOMANCER_GGUF_EXPERT_GIB=-1`: PLE paging with fully resident experts.

The kernel suite checks exact PLE rows (including duplicate and boundary IDs),
expert remapping for singleton, verifier and grouped-prefill paths, zero-budget
and LRU eviction, and release/reconstruction of expanded prefill weights. Model
smoke runs assert reference token equality and report MLX allocations, expert
cache bytes/hits/misses, and the process physical footprint separately.

2026-10-09 validation on M4 Max 128 GiB (single runs, original IQ3_S weights;
short fixture 47 prompt tokens / 256 output tokens, draft loaded in both modes):

| Mode | Post-decode physical footprint | Ordinary decode | MTP decode |
| --- | --- | --- | --- |
| Fully resident reference | 90.3–90.9 GB | 26.8 token/s | 48.5 token/s |
| PLE paging + 16 GiB expert cache | 29.3–29.9 GB | 11.7 token/s | 22.1 token/s |

GB in this table is decimal; the cache setting is binary GiB. File cache warmth,
first-use kernel compilation, prompts and background load affect timings. Both
modes matched all 256 reference tokens. The paged forced-rejection path also
matched. A 4,037-token prompt (1,024-token chunks) matched all 128 reference
outputs in ordinary/MTP/forced-rejection modes. It released 5,819,596,800 bytes
of expanded prefill matrices; the combined short/long paged run peaked at
38,200,270,544 bytes physical footprint. Expert retained bytes stayed below
17,179,869,184 throughout. These measurements preceded the additional 8,192-row
PLE metadata cap; that cap was covered by the kernel/service regressions.
Logs: `/tmp/synomancer-paging-{buffer-model,resident-baseline,kernels,integration}.log`.

The final service integration passed history recall, image descriptions with MTP
on/off equality, cancellation, restart, stop sequences, sampling fallback and
one-token limits. The Release build passed. The Mac was locked during the final
UI check, so the new settings control was compiled but not visually verified.

### Native decode fusion (strata-mlx PR #9)

The native Swift/MLX GGUF backend adapts four optimizations from
[strata-mlx PR #9](https://github.com/yibie/strata-mlx/pull/9): SIMD top-10 routing,
GDN history/conv/SiLU fusion, GDN output RMSNorm/gate fusion, and routed/shared
expert combination. The Metal source retains the upstream MIT notice. There is
no Python inference dependency. Windows above eight tokens retain the original
paths; ordinary decoding and the native MTP verifier use the same helpers.

The port preserves this backend's ascending expert order (including tied
scores), FP16 SiLU and reduction rounding, FP32 GDN normalization weights, and
convolution history used for speculative rollback. It deliberately retains the
existing Q/K normalization, text/multimodal RoPE, and hyper-connections. Upstream
MLX-version-specific numerical formulas and M5 Ultra thresholds are not copied
blindly. PLE warming from PR #10 is not included.

Set `SYNOMANCER_GGUF_PR9=0` before launch to use the old path. Tests additionally
support `SYNOMANCER_GGUF_PR9_ORDER=b0,b1,o0,o1,...`: `b`/`o` select baseline or
optimized kernels and `0`/`1`/`2` select ordinary/MTP/forced rejection. Both graph
variants are retained separately so switching does not force retracing each run.
This switch is changed only serially by the benchmark, between completed runs.

Reproduce the comparison with local weights (paths may contain spaces):

```sh
Tests/run-gguf-tests.sh --kernels
Tests/run-gguf-decode-benchmark.sh /path/to/first-shard.gguf /path/to/mtp.gguf \
  Tests/Fixtures/qwen4-iq3s-greedy.json resident
Tests/run-gguf-decode-benchmark.sh /path/to/first-shard.gguf /path/to/mtp.gguf \
  Tests/Fixtures/qwen4-iq3s-greedy.json paged
```

The benchmark shares one loaded model per residency mode, warms both kernel
variants and generation modes, then takes four paired, order-reversed samples per
configuration. Every run checks all reference tokens; the first token belongs
to prefill and is excluded from decode throughput. Paged mode uses a 16 GiB
expert cache. Compare results within a process, not against historical runs.

Measured on 2026-10-15, Apple M4 Max / 128 GB / macOS 27.0.1, IQ3_S target
and complete Q8_0 MTP sidecar. Short fixture: 47 input / 256 output tokens.
Three measured runs per cell, after the four warmup runs; values are medians:

| Residency / generation | Baseline token/s | Fused token/s | Change |
| --- | ---: | ---: | ---: |
| Full resident, ordinary | 41.34 | 39.65 | -4.1% |
| Full resident, MTP | 81.80 | 67.65 | -17.3% |
| PLE paging + 16 GiB experts, ordinary | 11.58 | 11.65 | +0.6% |
| PLE paging + 16 GiB experts, MTP | 21.67 | 22.99 | +6.1% |

The full-resident generation run drifted substantially: baseline ordinary ranged
33.08–47.39 token/s and fused ordinary 25.40–40.68; baseline MTP 61.47–83.47
and fused MTP 43.25–82.68. Both variants slowed later in the run. These measured
regressions must not be omitted, but do not establish the kernel contribution in
isolation. No stable full-resident end-to-end speedup is claimed.

A separate paired-forward experiment restores the same state and feeds identical
tokens to both variants, alternating which goes first. It includes host dispatch
and GPU completion, and compares the full logits outside timing. Each row has
24 measured pairs after four warmups; times below are medians, not end-to-end
MTP throughput. Context is the initial prompt length; it grows during the test.

| Residency / initial context / window | Baseline ms | Fused ms | Median paired difference (fused - baseline) |
| --- | ---: | ---: | ---: |
| Full resident / 47 / 1 | 21.02 | 19.95 | -1.04 ms |
| Full resident / 47 / 4 | 34.02 | 32.99 | -1.09 ms |
| Full resident / 4,037 / 1 | 23.91 | 23.03 | -0.90 ms |
| Full resident / 4,037 / 4 | 38.85 | 37.61 | -1.19 ms |
| Paged / 47 / 1 | 83.90 | 82.79 | -2.57 ms |
| Paged / 47 / 4 | 147.26 | 151.48 | +0.01 ms |

The full-resident paired results show about 3–5% lower forward latency. Paged
four-token pairs show no clear gain: the first variant pays more cache/staging
work, and reversing order reverses much of that difference. These results do
not imply that kernel fusion removes expert-paging overhead. Full-resident
short-run physical footprint stayed about 91.2 GB; paged mode stayed around
30 GB. No residency policy, expert budget, or PLE warming behavior was changed.

Validation: the synthetic kernel suite passed, including exact FP16 operations,
FP32 normalization weights, tie order, history writes and long-window fallbacks.
All 32 end-to-end A/B runs matched the 256-token reference, with unchanged MTP
proposed/accepted counts (206/185). All 168 paired forwards (including warmups)
had zero full-logit difference. Forced-rejection MTP matched the short 256-token
and 4K-context 128-token references, with 14/16 exact prefix-cache/hidden checks.
The Swift 6 Release application build passed.

Enable the paired experiment with `SYNOMANCER_GGUF_PR9_PAIRED=1`. For the long
fixture use `Tests/Fixtures/qwen4-iq3s-long-1024-128.json`; its prefill step is
1,024. Generated evidence is under `.build/GGUFTests/`: `pr9-resident.log`,
`pr9-paged.log`, `pr9-resident-paired.log`, `pr9-kernels.log`, `pr9-release.log`,
`pr9-results.json` and `pr9-source-sha256.txt`. The interrupted
`pr9-resident-preliminary.log` is excluded from every result above.


### Per-kernel default selection (2026-10-15 follow-up)

The four ports are independently selectable with `SYNOMANCER_GGUF_PR9_MASK`:
`1` = top-10, `2` = convolution/SiLU/history, `4` = gated RMSNorm,
`8` = MoE combination. Add bits to combine them. `0` is the original path and
`15` is the previous all-enabled path. `SYNOMANCER_GGUF_PR9=0` overrides this
setting and disables every port. The default is now `11` (top-10, conv, combine);
gated RMSNorm remains available for experiments but is not enabled by default.

Compiled MoE graphs are keyed by the top-10/combine bits; GDN graphs are keyed
by the conv/norm bits and prefix-capture mode. Changing an unrelated bit does
not create another copy of the same graph. The mask is immutable during normal
app execution; only the serial test harness changes it between completed calls.

The ablation holds model state and input fixed, rotates and reverses the order
of configurations, and checks full-logit equality. It tests single features,
all-enabled, and leave-one-out configurations at windows of 1, 2 and 4 tokens.
The first four samples warm each graph; the initial sweep measures 60 samples
per cell. Timing includes CPU graph dispatch and GPU completion. Thermal state
is recorded with each observation. Only the within-sample paired differences
are used to attribute a change to a feature.

To run that sweep using the benchmark helper, add these environment settings:

```sh
SYNOMANCER_GGUF_PR9_ABLATION=1 SYNOMANCER_GGUF_PR9_ABLATION_ONLY=1 \
SYNOMANCER_GGUF_PR9_PAIRED=0 \
Tests/run-gguf-decode-benchmark.sh /path/to/first-shard.gguf /path/to/mtp.gguf \
  Tests/Fixtures/qwen4-iq3s-greedy.json resident
```

`SYNOMANCER_GGUF_PR9_ABLATION_MASKS` and `SYNOMANCER_GGUF_PR9_ABLATION_SAMPLES`
customize the sweep. The default mask list is `0,1,2,4,8,15,14,13,11,7`.
`SYNOMANCER_GGUF_PR9_MASK_ORDER` specifies a mask for each end-to-end run.
`SYNOMANCER_GGUF_PR9_LONG_ORDER` and `SYNOMANCER_GGUF_PR9_LONG_MASK_ORDER`
can override the generation/mask sequence for a long extra fixture, so a long
forced-rejection regression need not repeat the entire throughput matrix.

Initial full-resident sweep: median paired latency change in ms (negative is
faster), 60 measured states per window. All configurations use identical inputs
and restored state within a sample:

| Enabled alone | 1 token | 2 tokens | 4 tokens | Default |
| --- | ---: | ---: | ---: | --- |
| Top-10 (`1`) | -0.320 | -0.330 | -0.352 | On |
| Conv/SiLU/history (`2`) | -0.195 | -0.174 | -0.211 | On |
| Gated RMSNorm (`4`) | -0.043 | -0.117 | +0.160 | Off |
| MoE combination (`8`) | -0.746 | -0.320 | -0.636 | On |
| Selected combination (`11`) | -1.022 | -1.070 | -0.976 | On |
| Previous all-enabled (`15`) | -1.048 | -1.121 | -1.015 | Opt-in |

The leave-one-out comparison is particularly weak for gated RMSNorm: adding it
back to mask 11 wins only 53%, 52%, and 52% of samples at windows 1, 2, and 4.
Its median marginal savings are just 0.022, 0.015, and 0.059 ms. The actual Q/K
normalization output is FP16, so this is not explained by the gate-norm path
being entirely skipped due to a dtype mismatch. It has no reliable incremental
benefit on this machine and is disabled by default.

An independent continuous-run repeat exposed substantial common-mode drift.
Baseline MTP fell from 77.58 to 56.10 token/s across the measured rounds; mask 11
fell from 78.44 to 59.70. Later, the two-token all-disabled forward median reached
83.09 ms, compared with 25.65 ms in the initial sweep. Both paths were affected.
Thermal state changed from nominal to fair; AC power and normal power mode were
confirmed. No power/thermal warning was reported by `pmset`, and hardware
frequency counters were unavailable without privileged authentication. These
observations establish run-state drift, but do not prove thermal throttling as
its sole cause. The four kernels do not reproduce a per-pass slowdown large
enough to explain the previous 17% end-to-end result.

For transparency, the continuous-repeat throughput medians (four measured runs)
were 40.00 -> 41.93 token/s ordinary and 73.09 -> 71.66 MTP. Adjacent paired speed
changes ranged from -5.6% to +8.6% ordinary and -4.0% to +6.4% MTP. This is not a
stable MTP speedup claim. The noisy repeat is retained, not pooled into the
initial sweep or silently discarded.

`SYNOMANCER_GGUF_BENCH_PAUSE_MS=5000` adds five seconds *between* generation runs
(up to 60,000 ms), outside timing, for a separate paced-request comparison.
It never inserts a delay in application inference. The benchmark helper also
preserves Hugging Face snapshot symlink names when making paths absolute; a
canonical blob hash loses the shard naming needed to find the second file.

The paced diagnostic did **not** produce a valid final speed estimate and was
stopped. Its first all-disabled run was only 3.87 token/s while thermal state was
nominal (mask 11 was 3.18); thus thermal state alone does not explain the slow
state. A short process sample showed waits in `mlx::core::scheduler::wait_for_one`
from `eval_impl` and in Metal/IOGPU command submission. This is evidence of
scheduler/driver waiting, not proof of a particular GPU-frequency, page-fault,
or background-task cause. A process-local `taskpolicy -B` diagnostic attempt did
not promptly restore normal speed; that also makes the interrupted run unsuitable
for inclusion in the benchmark. No user application or system power setting was
changed. The test process was terminated and its model allocation released.

The default selection is therefore based on the balanced per-forward ablation,
including the leave-one-out check, rather than the disturbed paced run. It is a
conservative selection of demonstrated kernel savings, not a claim that the
remaining system-level throughput variability has been fixed. Hardware-frequency
and per-process GPU contention attribution remain unresolved.

Validation for this follow-up: both ablation sweeps completed (1,920 + 1,260
configuration evaluations), all with zero full-logit difference. The independent
repeat matched all 256 reference tokens in each of its 20 ordinary/MTP runs;
proposal/acceptance counts remained 206/185. The selected defaults also matched
the 4K-prompt 128-token forced-rejection reference and 16 exact prefix-cache/hidden
comparisons. The full kernel suite and Swift 6 Release build passed.

Evidence under `.build/GGUFTests/`: `pr9-ablation-resident.log`/`.json`,
`pr9-selected-resident.log`/`.json`, `pr9-selected-kernels.log`,
`pr9-selected-release.log`, and `pr9-selected-source-sha256.txt`.
`pr9-selected-paced.log` and `pr9-slow-sample.txt` are diagnostic evidence only;
the interrupted paced run is excluded from performance conclusions.
