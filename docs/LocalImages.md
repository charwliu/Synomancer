# 本地图像生成与编辑

App 顶部工具栏的照片按钮打开“图像生成与编辑”。输入描述可以文生图，
选择参考照片后输入指令可以编辑；“继续编辑这张图”将当前结果作为下一次参考图。
支持进度、停止、随机种子、尺寸、步数和 PNG 导出。界面仅使用 SwiftUI。

运行时完全使用 Swift + MLX，不启动 Python、服务器或云端 API。
模型通过与聊天、ASR、TTS 相同的 HF_HOME 及文件夹授权读取，
默认 /path/to/workspace/.cache/huggingface/hub。缓存只读，不自动下载。
默认优先选择完整的小型量化缓存；缺失分片的快照不会显示。

推荐现有 mlx-community/Qwen-Image-2.1-MLX-4bit（约 10.5GB 权重）。
支持官方 Diffusers 格式，以及保留视觉编码器的 MLX affine 量化格式。
仅导出文本编码器、缺少视觉塔的版本不能用于参考照片编辑。

## 负面提示词

在主提示词下填写 **负面提示词（可选）**，例如“模糊、水印、多余手指”。
输入框随内容增高，最多四行。填写后显示独立的 **负面 CFG 强度**，范围 1.1–10。
Qwen 默认 2，FLUX.1 默认 1.1；切换模型时恢复对应默认值。
它与 FLUX 的 Guidance 嵌入参数不同。留空或只输入空白时完全沿用原单分支推理。

| 模型 | 负面提示词 |
| --- | --- |
| Qwen-Image 2.1 | 文生图、参考图编辑及结构控制生成 |
| FLUX.1-dev | 文生图 |
| FLUX.2-dev / Klein 9B | 当前不支持；界面提示，直接带负面词的请求会被拒绝 |

启用后分别编码正、负提示词，每一步组合 `negative + CFG × (positive - negative)`。
Qwen 编辑时两个分支使用同一参考图，但各自建立文本布局及去噪缓存；负面提示词变化会使
会话条件缓存失效。ControlNet 的两个分支也使用独立缓存，在控制结束时一起失效。
FLUX.1 两个分支分别使用 CLIP pooled 和 T5 特征。
实测 FLUX.1 在 512×512、28 步、CFG 2、负面词包含 blurry 等词时明显模糊，
因此采用较低默认强度，界面同时提示风险；负面词并不保证改善画质。
同一提示词/负面词/种子在 CFG 1.1、28 步下实测恢复清晰杯口、把手和高光，
耗时约 159.36 秒，取消及取消后再次生成也通过。图片保存在
`.build/NegativePrompt/flux1-cfg11/flux1-dev-red-mug.png`。
每步增加一次 DiT 计算，速度和内存开销增加；高强度不保证更好，Qwen 2.1 原本默认无 CFG。
切换至 FLUX.2 时保留负面词草稿，但不发送；切回支持的模型后可继续编辑。

CPU Diffusers 小模型对照覆盖生成/参考图编辑、不同正负文本长度及缓存首步/后续步，
最大绝对误差小于 `8e-7`；同时验证空白负面词、无效强度和不支持的模型。
完整 Qwen MLX 4bit 权重已实测 512×512、8 步、CFG 2 的文生图与参考图编辑：
分别约 35.61 秒和 48.16 秒，MLX 分配峰值约 10.62 GB。检查输出红杯与蓝杯，
改色成功且构图保留；8 步图像偏软，仅作功能回归，不作为最终画质评估。

```sh
python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --image --image-size 512 \
  --image-steps 8 --image-negative-prompt 'blurry, watermark, text, distorted handle' \
  --image-output .build/NegativePrompt/images
```

协议参考：[Qwen-Image 2.1](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage21)、
[FLUX.1](https://huggingface.co/docs/diffusers/api/pipelines/flux)。

## FLUX.1-dev 文生图

支持 `black-forest-labs/FLUX.1-dev` 的官方分组件 Diffusers 缓存，运行时为
纯 Swift + MLX。共用 HF_HOME 与文件夹授权，不下载或修改模型权重。
本次支持原始 BF16 导出；不将未验证的量化、Schnell、Kontext 或 Fill 当作 Dev 加载。

在图像面板点击“刷新”，选择 `black-forest-labs/FLUX.1-dev`，输入提示词并生成。
切换到此模型时采用默认 28 步；Guidance 默认为 3.5，可在 0–10 之间调节。
尺寸、Seed、停止、PNG 导出和生成后的超分沿用现有界面。此入口仅做文生图，
切换时清除参考图/控制图，并停用参考照片选择和“继续编辑”按钮。

加载所需目录：`transformer/`、`text_encoder/`、`text_encoder_2/`、`vae/`、
`tokenizer/`、`tokenizer_2/`、`scheduler/` 和根目录 `model_index.json`。
CLIP 使用原仓库的 `vocab.json`/`merges.txt`，不要求额外下载 `tokenizer.json`；
T5 使用缓存的 tokenizer.json。根目录 `flux1-dev.safetensors` 是与 transformer
分片重复的整包权重，不参与加载或内存估算。所有分片必须完整可读。

实现要点：

- CLIP 的 77 token 因果注意力、QuickGELU 和 EOS pooled 向量；UTF-8 byte BPE
  支持中文及 emoji，截断保留 EOS。
- T5-XXL 的 512 token EOS/right padding、双向相对位置桶、无缩放注意力及 gated GELU。
  与官方 FluxPipeline 一样，T5 padding 不传 attention mask。
- Dev 的逐层双流/单流调制、三轴 RoPE、独立 timestep/guidance/pooled embeddings。
- 根据输出 token 数计算动态 mu 的 FlowMatch Euler，FP32 更新后回到模型 dtype。
- 16 通道 latent 的 2×2 packing，VAE 解码前执行 `latent / 0.3611 + 0.1159`。
  VAE 使用 FP32，PNG 为不透明 RGB；文本编码器求值后释放，避免与 DiT 同时驻留。

源码依据为官方 Diffusers
[`122b1e11fd497c3eeef14b3b98ca26a60166e48b`](https://github.com/huggingface/diffusers/tree/122b1e11fd497c3eeef14b3b98ca26a60166e48b/src/diffusers)
的 FluxPipeline、FluxTransformer2DModel、AutoencoderKL 和官方 Transformers
[`v5.18.0`](https://github.com/huggingface/transformers/tree/v5.18.0/src/transformers/models)
的 CLIP/T5（Apache-2.0，见 [Apache-2.0](licenses/QwenImage21-Apache-2.0.txt)）。
VAE 的共享卷积实现保留 MFLUX MIT 版权，见 [MFLUX MIT](licenses/MFLUX-MIT.txt)。
模型遵循其仓库 [LICENSE.md](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)。

数值验证以现有 Python 开发环境运行 CPU oracle；App 本身不调用 Python。

~~~sh
/path/to/workspace/aiml/transformer/.venv/bin/python Tests/make-flux1-parity.py
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release --flux1-parity \
  --cache /path/to/workspace/.cache/huggingface/hub --metal-validation
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release --image-flux1 \
  --cache /path/to/workspace/.cache/huggingface/hub --image-output .build/Flux1Images \
  --metal-validation
~~~

2026-10-08，CPU Diffusers 0.41.0.dev0 / Transformers 5.18.0 对照的最大绝对误差：
DiT `1.3e-6`，CLIP `8.4e-7`，T5 `1.1e-6`，VAE `5.6e-6`，采样日程 `6.0e-8`。
12 组真实 tokenizer 对照（中文、emoji、HTML 字面文本、空文本、超长文本）一致。
完整缓存发现、缺失 T5 分片拒绝和不支持媒体/guidance 的请求拒绝测试通过。
完整权重实测：M4 Max 128 GiB，Release，开启 Metal API 验证，512×512、28 步、
Guidance 3.5、Seed 42，共约 79.4 秒（含加载及本次 shader 编译）。生成的红杯 PNG
已目视检查，主体、右侧把手和浅灰背景符合提示词，没有随机彩色噪点。
真实去噪取消和取消后的新请求均通过；这是单次样例耗时，不是所有尺寸的速度保证。
输出位于 `.build/Flux1Images/flux1-dev-red-mug.png`，不纳入 Git。

### FLUX.1 / Qwen-Image-2.1 性能对比

2026-10-08，M4 Max 128 GiB，Release，关闭 Metal API 验证。
使用 App 的原生 Swift + MLX 管线，768×768，Seed 42，相同英文红杯产品摄影
提示词，无参考图、ControlNet 或超分。FLUX 使用默认 28 步、Guidance 3.5，
Qwen 使用默认 40 步；Qwen 管线不使用 FLUX 的 Guidance 参数。

首次请求指独立服务会话的第一次生成，并非清除 macOS 文件缓存的磁盘冷启动。
顺序为 FLUX BF16、Qwen BF16、Qwen 4bit，然后反向各重复一次；相同输入的
Qwen 重复请求允许现有的提示词/前缀缓存命中，FLUX 当前没有跨请求编码缓存。
各次请求前清理 MLX 空闲分配缓存，保留服务已有的条件缓存。测试期间未发现
其他 Synomancer 推理或 Swift 编译进程，系统未报告温控告警。

| 模型权重 | 默认步数 | 首次端到端 | 重复端到端 | 分组件权重大小 | MLX 活跃分配峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLUX.1-dev BF16 | 28 | 111.39 s | 167.97 s | 33.74 GB | 31.51 GB |
| Qwen-Image-2.1 BF16 | 40 | 189.11 s | 161.85 s | 33.12 GB | 24.40 GB |
| Qwen-Image-2.1 MLX 4bit | 40 | 220.72 s | 212.80 s | 10.48 GB | 14.36 GB |

端到端计时覆盖缓存发现、加载、提示词编码、去噪、VAE 解码和 PNG 编码，
不含编译、自动下载或将已编码 PNG 写入文件的时间。分组件大小不包括 FLUX
根目录重复的整包权重；它不是 App 总内存。首次请求每步去噪（排除第一步）
分别约为 3.63、4.29、5.38 秒，受执行时的机器状态波动影响。

重复请求阶段计时：

| 模型 | 加载/编码 | 去噪 | 解码/PNG | 去噪每步（排除第一步） |
| --- | ---: | ---: | ---: | ---: |
| FLUX.1-dev BF16 | 6.22 s | 160.97 s | 0.78 s | 5.39 s |
| Qwen-Image-2.1 BF16 | 0.09 s | 160.77 s | 0.99 s | 3.90 s |
| Qwen-Image-2.1 MLX 4bit | 0.04 s | 211.39 s | 1.37 s | 5.32 s |

本例 FLUX 比 Qwen 4bit 更快，但相对 Qwen BF16 未显示稳定的速度优势：
FLUX 首次明显更快，重复时 BF16 Qwen 略快。差异同时包含模型实现和默认步数不同。
4bit 主要减少权重大小，在这次 768×768 测试中没有转化为速度优势。
单个提示词不能用于判断整体画质、文字生成能力或其他分辨率的速度。
这也不是 Python Diffusers、MFLUX 或其他后端的性能排名。
首次和重复耗时的差值还受共享 Metal shader 缓存及机器运行状态影响，
不能全部归因于提示词/前缀缓存；两者不是独立控制缓存开关的因果实验。
FLUX 两轮波动约 51%，原因尚未通过算子或 GPU 性能分析定位，不能将首轮
结果解释为架构固有的固定加速倍数。各模型目前只有一组首次/重复样本。
128 GiB 内存允许优先尝试 Qwen BF16，尤其是需要参考图编辑或 ControlNet 时；
4bit 适合减少内存占用。FLUX 当前入口仅做文生图。

六次实际生成均完成，三个模型的首次/重复 PNG 完全一致。三张首次生成的
红色杯子图已目视检查，构图与提示词一致，没有大量随机彩色噪点；不据此
对通用画质作排名。

复现（必须已有完整的三个缓存，不会下载）：

~~~sh
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub \
  --image-model-benchmark --image-size 768 \
  --image-output .build/ImageModelComparison
~~~

阶段计时、缓存命中信息、MLX 活跃分配峰值和 PNG 文件名写入
`.build/ImageModelComparison/results-768.json`。峰值由 MLX allocator 统计，
包含进程中其他服务保留的条件缓存，不代表 RSS 或 App 全部统一内存。

## FLUX.2-dev（官方 BF16 / MLX 8bit）

支持 `black-forest-labs/FLUX.2-dev` 的原始 Diffusers 分组件 BF16 缓存及其本地 MLX affine 8bit 副本，
使用纯 Swift + MLX，不启动 Python 或远程文本编码服务。共用设置中的 HF_HOME。
在图像面板“刷新”后选择此模型；支持文生图及一张参考照片编辑。
默认 28 步、Guidance 4，步数可提高到 50；28 步是速度优先的初始值，
不等同于 Klein 的 4 步蒸馏模型。此入口不使用 Qwen ControlNet。

实现基于官方 [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
快照 `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` 的配置和 tokenizer；
Diffusers `122b1e11fd497c3eeef14b3b98ca26a60166e48b` 的 Flux2Pipeline/
Flux2Transformer2DModel，以及 Transformers v5.18.0 的 Mistral 层（Apache-2.0）。
系统提示词和原始模板来自 BFL flux2
`5a5d316b1b42f6b59a8c9194b77c8256be848432`。
共用 Klein 的原生 VAE、四轴 RoPE、参考图 packing 和 empirical-mu Euler。
授权正文沿用 [Apache-2.0](licenses/QwenImage21-Apache-2.0.txt) 和
[MFLUX MIT](licenses/MFLUX-MIT.txt)，模型权重遵循其仓库许可证。

- 原始 Dev 的 8 个双流 / 48 个单流块、48 头、128 head dim、15360 维文本输入。
  补齐 timestep + Guidance 嵌入；原始配置省略 `guidance_embeds` 时按默认 true。
- 本地 Mistral 24B 文本编码器读取第 10/20/30 层隐藏状态并拼接；不执行后十层、
  最终 RMSNorm、词表投影或视觉塔。普通提示词编码不启用可选 prompt upsampling。
- 严格采用原始 byte-level BPE、系统模板和 512 token 左侧补齐，过滤字面 `[IMG]`。
  空 padding 查询产生零注意力输出，避免 MLX 的全掩码 softmax 产生 NaN。
  不让旧版 `LlamaTokenizerFast` 类名将字节词表改写成 SentencePiece/Metaspace。
- 文本编码求值后释放编码器，再加载 DiT。原始分组件权重大小为：
  DiT 64.45 GB、文本编码器 48.02 GB、VAE 0.34 GB，合计约 112.81 GB。
  此数值是磁盘文件大小，不是同时占用的 GPU 内存。
- 分阶段预算使用 `max(文本编码器, DiT + 2 × VAE) + 16 GiB`，加上已有 MLX
  活跃分配后需低于物理内存的 75%；在加载 DiT 前再次检查。预算预留 VAE
  FP32 转换和大图中间数组，但不是实测峰值或系统内存保证。建议 128 GiB 起，
  其他大型聊天模型驻留时可能被提前拒绝，需先释放模型。
- 缺少任何分片的缓存不显示；不将 BNB、FP8 或 MLX 4bit 导出作为本次 Dev 支持。

### 本地 MLX 8bit 转换

`Tools/quantize-flux2-dev.py` 使用 MLX affine Q8，默认 group size 64。
Transformer 与完整 Mistral 导出的二维权重量化（含 embedding），一维参数和 VAE
保留原精度；运行时仍使用 BF16 激活和 FP32 VAE。不是 GGUF Q8_0，也不是 FP8。
转换按原始分片依次处理，不会同时展开整个 112.81 GB 模型，不修改原始缓存。
输出先写入暂存目录，全部完成后发布；已有输出或未完成暂存目录会拒绝覆盖。

~~~sh
/path/to/workspace/aiml/qwen3-asr/.venv/bin/python Tools/quantize-flux2-dev.py \
  --source /path/to/workspace/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --output /path/to/workspace/.cache/huggingface/hub/models--local--FLUX.2-dev-MLX-8bit/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c-q8-g64
~~~

App 图像面板刷新后选择 **`local/FLUX.2-dev-MLX-8bit`**。
`local/` 表示本机派生缓存，不是 Hugging Face 远端仓库；无需下载或发布到该 ID。
程序也支持直接选择转换后的模型目录。

每个量化组件的 `config.json` 写入
`quantization: {bits: 8, group_size: 64, mode: "affine"}`，原始模型类及架构配置保留。
矩阵保存为 uint32 的 `.weight`，以及同名前缀的 `.scales`、`.biases`；索引包含三个张量。
加载端支持 affine 8bit 的 32/64/128 分组，拒绝未知量化方法。
Mistral embedding 先选取提示词使用的行，再解量化，不展开完整词表。

本机转换结果（2026-10-08，MLX 0.32.2）：139 秒，DiT **34.24 GB**、
Mistral **25.51 GB**、VAE **0.34 GB**，权重合计约 **60.09 GB**。
`quantization_report.json` 保留源路径、参数、矩阵数量及抽样权重误差；
权重误差不等同于图像质量损失。完整原始权重仍保留在原模型目录。

小模型回归对照 packed Q8 执行与显式解量化执行，覆盖 Dev Guidance、参考图注意力、
Mistral 第 10/20/30 层、左侧 padding 和量化 embedding；同时保留原精度数值回归。

下载完整权重到共用缓存（需要已有模型仓库访问权限）：

~~~sh
hf download black-forest-labs/FLUX.2-dev \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --cache-dir /path/to/workspace/.cache/huggingface/hub \
  --exclude 'flux2-dev.safetensors' 'ae.safetensors'
~~~

根目录两个整包与 `transformer/`、`vae/` 重复，不参与加载；保留
`model_index.json`、`transformer/`、`text_encoder/`、`vae/`、`tokenizer/`、`scheduler/`。

### Dev 验证状态（2026-10-08）

小模型 CPU 对照（Diffusers 0.41.0.dev0、Transformers 5.18.0）开启 Metal API 验证：
Guidance 0/4 与参考图注意力最大误差 `1.4e-6`，Mistral 10/20/30 层特征 `9e-8`，
四种新旧权重前缀均通过。六组原始 Rust tokenizer 对照覆盖英文、中文、emoji、
空文本、字面图像标记和长文本截断；模型发现、缺失分片拒绝、请求参数及
64/128 GiB 内存预算检查通过。

分词 oracle 直接加载序列化 tokenizer.json：本机 Transformers 5.18 自动 Llama
路径会将原始 ByteLevel 改为 Metaspace，解码后丢失空格，不用该路径作为金标准。
使用原始模板及 PreTrainedTokenizerFast 保留文件定义，逐 token 比较原生实现。
Klein 共用代码的数值回归和真实 768×768 文生图/编辑回归通过，两个 PNG 与
上一版本逐字节一致。

本机 M4 Max 128 GiB 的完整 Dev MLX 8bit 已通过实际文生图及单参考图编辑，
均为 512×512、28 步、Guidance 4：

| 验证 | 种子 | 本次耗时 |
| --- | --- | --- |
| 红色陶瓷杯文生图 | 42 | 526.74 秒 |
| 将参考图红杯改为蓝杯 | 43 | 945.48 秒 |

人工检查两张 PNG：改色成功，杯形、把手、构图、背景和光照基本保留。
全流程 MLX 分配峰值 **37.97 GB**，不是整个进程或系统总内存。
耗时含加载且单步速度存在波动，仅代表本次运行，不是独占设备基准。
结果与日志保存在 `.build/Flux2Q8/images/` 和 `.build/Flux2Q8/inference.log`。
转换工具两项回归、原生 Q8 数值与模型发现回归、Release App 构建均通过。
尚未进行原始 BF16 的整模型推理及同种子画质对照。

开发验证所需元数据（不会下载大型权重）：

~~~sh
hf download black-forest-labs/FLUX.2-dev \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --local-dir .build/Flux2DevMetadata \
  --include 'model_index.json' 'transformer/config.json' 'text_encoder/config.json' \
    'vae/config.json' 'scheduler/scheduler_config.json' 'tokenizer/*'
/path/to/workspace/aiml/transformer/.venv/bin/python Tests/make-flux2-dev-parity.py
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --flux2-dev-parity --metal-validation
~~~

MLX 8bit 完整缓存的实际生成/编辑（M4 Max 128 GiB）：

~~~sh
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --image-flux2-dev --image-size 512 \
  --cache /path/to/workspace/.cache/huggingface/hub/models--local--FLUX.2-dev-MLX-8bit \
  --image-output .build/Flux2Q8/images
~~~

## FLUX.2 Klein 9B

支持 `mlx-community/FLUX.2-klein-9B` 的分组件 safetensors 缓存，
`model_index.json` 必须为 `Flux2KleinPipeline` 且 `is_distilled=true`。
切换到 Klein 后默认 4 步，使用蒸馏模型的 guidance 1；支持文生图和单张参考图编辑。
此入口不识别 base、dev 或 KV 变体，也不使用 Qwen ControlNet。
在图像面板点击“刷新”，选择 `mlx-community/FLUX.2-klein-9B`；不选择参考图时
生成照片，选择参考照片后输入修改指令即可编辑。共用设置中的 HF_HOME。

缓存位置为 `/path/to/workspace/.cache/huggingface/hub`。下载命令：

~~~sh
hf download mlx-community/FLUX.2-klein-9B \
  --revision b0f0826a36667ec7c58253e50557ba76f8c0255e \
  --cache-dir /path/to/workspace/.cache/huggingface/hub \
  --exclude 'flux-2-klein-9b.safetensors'
~~~

根目录的同名整包权重与 `transformer/` 重复，不参与加载。
所需组件是 `transformer/`、`text_encoder/`、`vae/`、`tokenizer/` 和 `scheduler/`。
该仓库为非 4bit 权重；App 不在下载或加载时自动量化。
本地分组件权重约 34.71 GB。该快照实际存储 FP16；Klein 的文本编码器和 DiT
在内存中将 FP16 权重转为 BF16，以采用参考用法的计算范围，VAE 保持 FP32。
全尺寸文本特征可达到约 2.3 万，直接采用 FP16 时第一步去噪会产生非有限值；
BF16 计算已通过完整权重生成/编辑验证。不会改写 HF 缓存文件。
模型参考用法见 [mlx-community/FLUX.2-klein-9B](https://huggingface.co/mlx-community/FLUX.2-klein-9B)。

- Qwen3 使用关闭 thinking 的模板、512 token 右侧 padding，拼接第 9、18、27 层隐藏状态。
  不计算词表投影，也不执行第 28 层之后的网络；文本编码完成后释放编码器。
- DiT 使用双流与单流块、四轴 RoPE 和共享调制；参考图 token 的时间坐标为 10，
  目标图为 0。普通 Klein 每步重新计算参考图注意力，不使用 KV 变体专属缓存。
- VAE 使用 RGB、32 通道 latent、2×2 patch 和训练时的 128 通道 BN 统计。
  参考图保持自己的比例，面积超过 1024² 时等比缩小，再居中裁去不足 16 像素的边缘；
  不强制缩放至输出画布。输出是带不透明 alpha 的 PNG。
- FlowMatch Euler 使用 Klein 官方 empirical mu；以 FP32 更新后转回模型 dtype。

数值回归使用 CPU Diffusers/Transformers 创建小模型夹具：

~~~sh
/path/to/workspace/aiml/transformer/.venv/bin/python Tests/make-flux2-parity.py
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release --flux2-parity --metal-validation
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release --image-flux2 \
  --cache /path/to/workspace/.cache/huggingface/hub --image-output .build/Flux2Images \
  --image-size 768 --metal-validation
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release --flux2-checks \
  --cache /path/to/workspace/.cache/huggingface/hub --metal-validation
~~~

Swift 实现参考 [MFLUX](https://github.com/mflux-community/mflux/tree/02bbe7bc36b2e32edca8f255a1ec494878c77fcc/src/mflux/models/flux2)
（MIT，正文见 [MFLUX-MIT](licenses/MFLUX-MIT.txt)）与 Hugging Face Diffusers 的
`Flux2KleinPipeline`、`Flux2Transformer2DModel`、`AutoencoderKLFlux2`（Apache-2.0，
正文见 [Apache-2.0](licenses/QwenImage21-Apache-2.0.txt)）。
模型权重遵循仓库内的 [LICENSE.md](https://huggingface.co/mlx-community/FLUX.2-klein-9B/blob/b0f0826a36667ec7c58253e50557ba76f8c0255e/LICENSE.md)。

### Klein 验证状态（2026-10-08）

Release App 构建通过。小模型 CPU Diffusers/Transformers 对照：Qwen3 第 9/18/27 层
最大绝对误差 `5.6e-8`，DiT 生成与参考图注意力 `1.1e-6`，VAE 编解码 `1.6e-5`，
采样日程 `6.0e-8`。已下载的真实 FP16 VAE 权重也通过 CPU/Metal 对照，
编码 `6.5e-6`、解码 `2.9e-5`。模型发现、缺失分片、非蒸馏 base 拒绝及参考图尺寸测试通过。
原有 Qwen 256–2048 尺寸对齐回归通过。
新增大幅度文本特征回归覆盖 FP16 导出的去噪溢出问题；实际第一步去噪完成后的
取消、取消后重新生成、拒绝不支持的 Qwen ControlNet 请求均通过 Metal API 验证。

完整 `mlx-community/FLUX.2-klein-9B` 快照现已完成下载，并通过原生 Swift/MLX
实际文生图和参考图编辑：M4 Max 128 GiB，Release，Metal API 验证开启，
768×768，4 步，Seed 42。红杯文生图约 20.59 秒，改成蓝杯的参考图编辑约
44.13 秒；包含加载、编码、去噪和 PNG 编码。MLX 活跃分配峰值约 25.96 GB，
不代表 App 的全部内存。两张 PNG 已目视检查，杯子颜色正确，构图与背景保持，
没有大量随机噪点。该数据为各一个样例，不是所有输入的速度保证，也不是与
前节默认步数不同的 FLUX.1/Qwen 测试在相同执行条件下的速度排名。
输出为 `.build/Flux2Images/generated-red-mug-768.png` 和
`.build/Flux2Images/edited-blue-mug-768.png`，不纳入 Git。

## Qwen-Image-2.1 实现

Services/ImageGeneration 独立于聊天模型注册表：

- 官方 MLXVLM.Qwen3VL 提供视觉、M-RoPE 和 DeepStack 编码。
  在本次请求的私有模型中替换最后的 RMSNorm 和词表投影，
  得到图像模型训练使用的末层归一化之前的隐藏状态。
  使用模型的原始文本模板，不应用通用聊天模板。
- DiT 实现单流块因果注意力、零中心 RMSNorm 文本投影、共享 scale/gate
  调制和二维图像 RoPE。第一次去噪计算提示词/参考图 K/V，后续步骤只计算目标图。
  相同条件的文本/参考图 K/V 可在图像面板会话中限额复用；ControlNet K/V 只用于当前请求。
- 单帧 RGBA VAE 以参考图 posterior mean 编码，支持 residual
  average/down 和 duplicate/up 支路。原始因果卷积取最后一个时间切片；
  已转换的 MLX OHWI 权重保持原布局。VAE 使用 float32。
- FlowMatch Euler 使用缓存中的动态时间偏移和 shift_terminal，以 FP32 累加后
  转回 latent dtype。时间步先在 0–1000 范围按模型 dtype 舍入，再除以 1000，
  按 Qwen 参考管线的顺序计算 FP32 正弦/余弦，避免 BF16/FP16 相位偏差。
- 图像任务由 actor 串行执行；每个 DiT 层和 VAE block 检查取消，
  在退出前完成已提交的 GPU 工作。所有随机噪声使用请求的独立 MLX key。

当前版本支持一张参考照片，PNG 可保留模型生成的 alpha。Qwen-Image-2.1
Fun ControlNet Union 可直接使用控制图；也可从照片自动提取相对深度图或
Vision 人体姿态骨架图。自动深度使用 Apple 的
[Depth Anything V2 Small Core ML](https://huggingface.co/apple/coreml-depth-anything-v2-small)
（Apache-2.0，约 50 MB），姿态识别使用系统 Vision；模型放在同一 HF_HOME，
不会自动下载。Union 权重来自
[alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)，
需单独获取并遵循其 Qwen Research License。控制图模式按结构重新生成画面，
参考照片用于提取结构，不会作为像素级编辑底图。界面可调整控制强度与生效步数；
两条控制分支会缓存文本前缀，控制结束时重新建立无控制前缀缓存。
控制图等比适配画布时新增的空白区域保持透明，预览、PNG 导出与 VAE 编码
使用相同补边；姿态图只在原照片覆盖范围内保留骨架编码所需的黑底。
参考照片按选定画布等比缩放并以白色补边，不剪掉照片内容；
图像编辑显式将视觉处理器的像素预算设为画布面积，避免 Qwen3-VL 默认的
1280 个视觉 token 限制再次缩小参考图；视觉网格与 VAE 网格保持对应。
选择参考照片或“继续编辑这张图”时，根据图片尺寸及 EXIF 方向自动设置宽高，
按模型范围 256–2048 缩放，并对齐到最近的 32 像素倍数；仍可手动调整。
极端长宽比受最小短边限制，可能增加补边。编辑不保证所有未指定内容逐像素不变。
默认 768×768、40 步；提高尺寸会增加注意力和 VAE 内存。
512×512 适合先检查指令效果。未接入 LoRA、自动蒙版或 Turbo 蒸馏。

## 超分辨率与 4K 导出

生成图片后，在“超分后端”选择 Apple VideoToolbox 或 RealESRGAN · MLX，
选择 2×、4×或 4K，然后点击“超分生成图”。保留生成原图；“显示超分结果”
切换预览和保存版本。每次超分均使用原图，避免反复放大累积伪影。
4K 定义为保留宽高比、长边 3840；16:9 对应 3840×2160，正方形对应
3840×3840。模型先按原生倍数超分，再以 Lanczos 调整到目标尺寸；
建议用 1024 原图，768 或 512 原图到 4K 包含额外插值放大。

- Apple 使用 macOS 26+ `VTSuperResolutionScalerConfiguration` 的 `.image`
  模式，运行时查询设备支持和放大倍数。首次点“下载 Apple 超分模型”明确启动
  系统模型下载；模型由 Apple 管理，不位于 HF_HOME。未就绪时不会静默改用
  普通插值。停止等待不一定中止系统后台下载。
- RealESRGAN 使用 FP32 的 23 个 RRDB、64 特征、32 增长通道的
  `RealESRGAN_x4plus`。App 内完全使用 Swift + MLX，不启动 Python。
  读取 `HF_HOME/synomancer/upscalers/RealESRGAN_x4plus.safetensors`；
  HF_HOME 指向父目录时也会检查 `hub/` 下相同路径。
- 两种后端以 256 像素核心块、32 像素上下文边缘处理，裁掉上下文后拼接，
  限制峰值显存；透明度由原图独立缩放并恢复。输入每边最多 2048，支持停止。

现有 PyTorch 权重只需转换一次，转换脚本依赖 torch/numpy，App 运行时不需要：

~~~sh
/path/to/workspace/aiml/real-esrgan/.venv/bin/python Tools/convert-realesrgan.py \
  /path/to/workspace/aiml/real-esrgan/weights/RealESRGAN_x4plus.pth \
  /path/to/workspace/.cache/huggingface/hub/synomancer/upscalers/RealESRGAN_x4plus.safetensors
~~~

脚本严格检查官方网络形状，优先使用 EMA 权重，将卷积 OIHW 转为 MLX 的
OHWI，保留 FP32，并以原子替换写入独立派生文件，不修改原始权重。
RealESRGAN 模型和 BasicSR 架构分别参考
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）与
[BasicSR RRDBNet](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py)
（Copyright 2018-2022 BasicSR Authors，Apache-2.0）。Swift 改写加入原生张量布局、
模型校验、分块和取消处理；Apache-2.0 正文见本目录的
[许可证](licenses/QwenImage21-Apache-2.0.txt)。模型文件不纳入 Git。

## 来源与许可证

Swift 移植参考：

- [Qwen-Image-2.1 官方实现](https://github.com/QwenLM/Qwen-Image-2.1)。
- Hugging Face Diffusers，本机锁定提交
  8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd：
  transformer_qwenimage21.py、autoencoder_kl_qwenimage21.py、
  pipeline_qwenimage21.py，Apache-2.0。
- [MFLUX](https://github.com/mflux-community/mflux)，参考提交
  9b9fd43aae32909dec9d313b7ecf5342aa3d05f2：
  qwen21/model 的 reference DiT、layout、residual VAE，以及 prompt encoding；
  外层代码 MIT，适配的 Qwen/Diffusers 代码保留 Apache-2.0 归属。
- Qwen-Image-2.1 Fun ControlNet Union 的并行控制块参考本地适配的
  VideoX-Fun `transformer/qwen_fun/control.py`（上游 Apache-2.0）；Apple 深度模型
  为 Apache-2.0。姿态骨架由系统 Vision 生成。
- 官方 mlx-swift-lm 3.32.3 的 Qwen3-VL 实现原样链接，没有修改 SPM checkout。

代码许可证见 [MFLUX MIT](licenses/MFLUX-MIT.txt)、
[Qwen/Diffusers Apache-2.0](licenses/QwenImage21-Apache-2.0.txt)。
模型权重单独遵循其仓库中的 Qwen Research License。本项目不附带模型权重。

## 验证

参考图尺寸回归（不加载模型权重，运行真实视觉预处理与 DiT 网格检查）：

~~~sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --image-reference-sizes
~~~

覆盖 256–2048 像素、横竖画布及 1280 个视觉 token 边界，并复现旧路径在
1152×1152 下的尺寸不一致错误。该检查使用合成提示 token，不执行模型推理。

完成 Release 构建后：

~~~sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --image --metal-validation
~~~

上述 Python 仅调用 Swift 编译器并启动原生测试程序，不参与模型推理。
测试读取真实缓存，生成红色杯子的照片，再将照片编辑为蓝色杯子；
加 --image-size 768 可覆盖 App 的默认画布。图像测试使用 Swift 6 编译。
PNG 保存在忽略的 .build/ImageTestOutput。需查看图片确认语义和视觉质量，
不能仅凭 PNG 格式正确就认为编辑成功。

本机 Release 实测（MLX 4bit，512×512，40 步，一次完整请求含加载/编码/解码）：
文生图约 62 秒，单图编辑约 90 秒。红杯生成、红杯改蓝杯均完成视觉检查；
杯柄、构图和背景基本保留。这不是其他尺寸或设备的性能保证。

取消和缓存检查使用 --image-checks，覆盖真实去噪期间取消后立即重启、
缺失分片、断开的权重链接、缺少视觉塔的导出，以及矩阵尺寸错误提前报错。
重启测试仅运行两步检查任务恢复，不用于验证图像质量。

数值参考夹具由开发期 CPU Diffusers 产生：

~~~sh
/path/to/workspace/aiml/transformer/.venv/bin/python \
  Tests/create-image-parity-fixtures.py .build/ImageParity
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --cache /path/to/workspace/.cache/huggingface/hub \
  --image-parity --metal-validation
~~~

两层 DiT（含参考图及缓存前后）、原始格式残差 VAE 编码/解码和 Euler
调度与独立 CPU 参考的最大绝对误差不超过约 2.1e-6。
该测试检查相同权重/输入，不依赖 MLX 与 PyTorch 的随机种子相同。
另外检查 BF16、FP16、FP32 的时间步舍入，以及 40 次 FP32 Euler 更新与
独立 CPU 参考的对照；模型激活 dtype 在更新后保持不变。
App 界面另用原生 AppKit/SwiftUI 离屏渲染检查参考图、结果和控件布局。

复现参考照片编辑时，可给 --image --image-size 768 加上
--image-reference /path/to/photo.png 和 --image-prompt '编辑指令'。
固定 40 步、Seed 43，输出 .build/ImageTestOutput/reference-edit-768.png。
随机彩色噪点不能仅凭症状认定为 VAE 网格纹理，需要原始参考照片和指令对照。
时间步舍入与 FP32 更新修正后，真实 MLX 4bit、768×768、40 步红杯改蓝杯
回归完成并通过视觉检查，未见大量随机彩色噪点；这不代表已复现其他参考照片的故障。

`hf cache ls --show-warnings` 可能报告普通文件（如 version.txt）不是仓库，
或只剩 refs 的仓库没有 snapshots。该提示来自 Python Hugging Face CLI，
表示这些条目被忽略；Synomancer 的 Swift 扫描器只列出完整的图像快照。
其他模型仓库的此类扫描告警不能据此归因为当前图像模型的彩色噪点。

Metal 校验回归：MLX 0.32.3 对非连续张量的标量行索引会执行 Gather，
空的索引形状缓冲触发 `bytes argument cannot be nil` 断言。
DiT 调制行改用范围切片并保留单例维度，避免该内核且保持数值结果一致。
测试显式开启 Metal API validation，不能仅以关闭校验时通过作为验证。
修复后在 Swift 6 + Metal API validation 下，使用真实 MLX 4bit 缓存完成
768×768、40 步红杯生成及红杯改蓝杯编辑，两张 PNG 均通过视觉检查。
Release 和 Debug App 构建均无错误、无告警。

### 重复生成的编码复用

图像服务保留最近一次已求值的提示词特征、参考图 VAE latent 和控制图
latent，总元素数限制为 16M（FP32 最多 64 MiB）。缓存键包含模型快照
路径、提示词、图片内容 SHA-256 和输出尺寸；修改 Seed、步数、控制强度
或控制结束比例不会重复编码。条件改变时替换缓存，超限条件不保留。
缓存不保留视觉编码器权重；无 ControlNet 的去噪前缀使用独立的限额缓存，
不改变采样精度或步数。
此优化主要减少相同条件下重复尝试的准备时间，首次请求和去噪耗时不变，
尚未测量本次改动的端到端提速比例。大尺寸编辑可手动降低生成尺寸，
再使用超分获得高分辨率结果；这会改变细节，应由用户选择。

### 去噪调制广播

首步建立文本/参考图 K/V 后，后续步骤只处理目标 token。此时四组
scale/gate 及输出调制均使用相同目标行，直接保留 `[1, 1, D]` 广播，
不再通过 `which` 展开为 `[1, tokenCount, D]`。首步仍保留原来的
按 token 选择逻辑。适用于普通生成、参考图编辑和 ControlNet；不改变
精度或采样步数。原生数值与 CPU Diffusers 参考、Metal API validation
以及同 Seed 真实图片对比已通过；性能实测见下文。

### 会话内去噪前缀复用

无 ControlNet 时，同一模型快照、提示词、参考图内容及尺寸的文本/参考图
前缀状态不依赖 Seed 或 timestep。完成去噪后保存已求值的 K/V，下一次
匹配请求从首步开始只计算目标图。最多保留一组，预算为物理内存的 1/64，
且不超过 2 GiB；超过预算时仍使用原来的请求内缓存。输入条件改变后的
下一次请求会替换缓存，关闭图像面板会释放前缀及编码缓存。清理修订号
防止正在等待编码的请求在关闭面板后重新写入缓存。

这主要优化连续调整 Seed、步数的重复生成/编辑，不缩减采样步数，不改变
精度。ControlNet 的前缀与控制强度/结束时间有关，仍保持请求内生命周期。
缓存命中时也避免在每层重新切片、复制已有的 K/V 前缀。

对比实验中逐层异步提交、每 4 层分组求值、合并 Q/K/V 与 MLP 投影均未
显示稳定收益，后两者还增加了内存使用，因此未纳入实现。

### 2026-10-07 实测（M4 Max，128 GiB）

Release，`mlx-community/Qwen-Image-2.1-MLX-4bit`，512×512。
先分别预热两个服务的提示词/参考图编码，只有前缀缓存开关不同。
8 步照片编辑按“关闭—开启—开启—关闭”执行：

| 前缀复用 | 第一次 | 第二次 | 平均 |
| --- | ---: | ---: | ---: |
| 关闭 | 11.8944 s | 12.5286 s | 12.2115 s |
| 开启 | 10.9939 s | 11.2715 s | 11.1327 s |

此例省约 1.08 秒（8.8%），属于相同条件连续试图的收益。首次请求无此
收益；不能把 8 步比例外推为 40 步加速比例。独立进程的首轮编码和机器
运行状态有明显波动，因此不使用不同时间的整次请求耗时作为提速证据。
CLI 基准不运行 UI 流光效果。

40 步文生图和首次编辑与基线 PNG 的 SHA-256 完全一致。跨请求前缀复用
会改变首步矩阵运算的序列长度，浮点舍入可产生微小像素差异：40 步重复
编辑在 0…255 RGBA 范围的 RMS 误差为 0.2910，平均绝对误差为 0.0724；
8 步对照 RMS 为 0.2602、平均绝对误差为 0.0572。已检查杯子图片，构图、
颜色和细节无可见变化。**不保证缓存开启/关闭之间逐字节相同**；同一路径
两次运行的 PNG 完全相同。回归阈值为 RMS ≤ 1、平均绝对误差 ≤ 0.25。

未采用的实验：
- 每层异步提交、每四层同步：未见稳定去噪收益。
- 每四层合并求值：ABBA 每八步平均 12.8569 s，逐层为 12.8413 s；
  峰值活跃分配增加约 0.743 GB。
- 合并 Q/K/V 和 MLP 两路投影：平均 10.0385 s，独立投影为 9.9258 s；
  打包权重增加约 3.31 GB 活跃内存。

复现保留方案的隔离对比（参考图使用上面图像测试生成的红色杯子）：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub \
  --image --image-prefix-benchmark \
  --image-reference .build/ImagePerformance/baseline/generated-red-mug.png
```

`--image --image-repeat` 额外运行完整重复编辑和不同 Seed/步数对照；
`--image-checks --metal-validation` 检查取消后重启、不同 Seed 和清理会话缓存；
`--image-parity --metal-validation` 比较 DiT/VAE/Euler/ControlNet CPU 参考。


### 从参考图生成提示词

选择参考图片后，在“识图模型”中选择本地视觉模型，点击“从参考图生成提示词”。
生成的中文描述先作为草稿显示，点击“填入提示词”或“替换当前提示词”才会写入
输入框；已有的编辑指令不会自动被覆盖。更换参考图或停止任务会丢弃迟到结果，
模型的思考内容、空回复和达到长度上限的未完成回复不会作为提示词应用。

此功能复用本地 MLX 聊天服务和统一 HF_HOME，不上传图片，不自动下载模型。
缓存需要具有视觉权重及处理器配置；支持 `preprocessor_config.json` 和
`processor_config.json` 两种文件名，与当前 MLX VLM 加载器一致。
本机 Gemma-4 31B 8bit 使用后者；仅缺少前者不能据此判断为纯文本模型。
缺少两种处理器配置的缓存暂不列入识图模型。

生命周期与真实识图验证：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub \
  --reference-prompt --model lmstudio-community/gemma-4-31B-it-MLX-8bit \
  --image-reference .build/ImagePerformance/prefix-cache/generated-red-mug.png
```

省略 `--model` 和 `--image-reference` 可仅运行模型发现、草稿、取消和错误处理测试。
2026-10-08，M4 Max 128 GiB，Qwen3.5-122B-A10B-oQ4-MTP 对 512×512 红杯
图片生成中文描述通过，包含首次加载约 54.9 秒；这是单个样例的端到端耗时。
同一图片使用 Gemma-4 31B 8bit 的实际识图也通过，包含首次加载约 31.6 秒，
正确描述红色陶瓷杯、右侧把手和浅灰背景；该数据不是持续生成速度基准。

### Chat model residency during image work

Local image generation and editing reserve the chat service's generation gate.
They wait for the current chat to finish, then release its model, MTP draft,
vision weights, prompt checkpoint and allocator cache before loading image
weights. New local chats wait until image work and GPU cleanup finish, including
on cancellation or error. The next chat lazily reloads its selected model.
Merely opening Image Studio does not unload chat. Remote Gemini requests do not
use this local model gate.
