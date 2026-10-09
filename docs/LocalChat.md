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
- 远程 OpenAI 请求暂不显示此指标，避免将网络/首字等待时间与本地指标混用。

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
Gemma-4 OptiQ 4bit 本地目录缺少 preprocessor_config.json，未参与推理测试。
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
HF cache. Refresh models in Synomancer and select it. This resident implementation
requires 128 GB unified memory. The first load takes roughly 1–2 minutes and
subsequent requests reuse the weights. Switching to another local chat model
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
