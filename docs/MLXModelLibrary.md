# MLX 模型库版本

Synomancer 的 Xcode 构建使用官方 `ml-explore/mlx-swift-lm` **3.32.3**：

- URL：<https://github.com/ml-explore/mlx-swift-lm>
- 固定 release：`3.32.3`
- 上游 revision：`3b339ad6e3b3f44c8121ecff5131c7fd55e075e6`
- 使用产品：`MLXLLM`、`MLXVLM`、`MLXLMCommon`
- 底层 `mlx-swift`：`0.32.3`

旧 `Libraries/MLXLLM`、`Libraries/MLXVLM`、`Libraries/MLXLMCommon` 是历史源码，
已从 App 的 package references 中移除，不参与 App 编译。升级及模型架构修复应以
上游 SPM 依赖为准，不要再修改这些旧文件来改变 App 的行为。

## 本地适配

新版模型库将 tokenizer 加载与模型实现解耦。`LocalChatTokenizerLoader.swift` 将
现有 `swift-transformers` 的 `Tokenizers.Tokenizer` 桥接到 `MLXLMCommon.Tokenizer`，
使用 `AutoTokenizer.from(modelFolder:)` 仅加载本地目录。它与 ASR/TTS 共用已存在的
tokenizer 依赖，不调用 Hub 下载接口。

桥接实现改编自官方 3.32.3 中的
`Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift`
（`TokenizerAdaptorMacro`、`TokenizerLoaderMacro`）。保留 Apple 的版权标记；
上游 Apache-2.0 许可证见下方 `MLXModelLibrary-LICENSE`。

模型发现先检查完整的本地 snapshots，再通过上游 actor registry 的 `contains` 查询
架构支持情况，不创建模型或下载文件。共享 LLM/VLM 架构名称的模型依据视觉配置和
本地 preprocessor 是否齐全选择加载路径；文本导出不因缺少视觉处理器而被漏掉。

普通聊天使用 `generateTokensTask`；MTP 使用公开的 `MTPSpeculativeTokenIterator`
和泛型 `generateTask`。两种路径均保留流式输出、停止字符串、采样参数和实际 token 用量。
模型与草稿头由同一 actor 独占；取消、停止或完成后等待 worker 结束并同步 GPU，
再允许下一次请求运行。不新增模型的 `@unchecked Sendable`。

`LocalQwenMoEModel.swift` 是 App 对 Qwen3.5 MoE 的局部加载桥接。
`Qwen3.5-122B-A10B-oQ4-MTP` 把 `language_model.mtp.*` 放在主权重分片内；
上游 3.32.3 遇到 safetensors 的 `format=mlx` 元数据会跳过原本的 MTP 清理，
导致严格参数更新报 `unhandledKeys: mtp`。桥接保留上游对已转换权重的处理，
只排除完整路径段为 `mtp` 的草稿参数，不再次偏移 RMSNorm，不改变主权重或混合量化。
模型的模块树、前向计算、视觉预处理和 KV cache 仍委托官方实现。
该 MoE 模型目前使用普通生成；名称中的 MTP 不代表 App 已启用其草稿头。
HF 缓存保持只读，官方依赖版本保持不变。加载/生成失败时，聊天界面显示实际错误。

## Qwen swish 和 OptiQ MTP

`Qwen3.8-27B-Uncensored-OptiQ-4bit` 使用 `qwen3_5` 架构。新版上游已正确实现
GDN 的 `RMSNormGated` **swish/SiLU**；完整注意力的输出门控仍是 **sigmoid**。
`output_gate_type: "swish"` 描述 GDN，不能据此将完整注意力改为 swish。
已移除上一版误拒绝 swish 的检查；明确声明的其他 GDN 门控仍会报错。
参考：[Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。

`LocalQwenMTP.swift` 读取同一 snapshot 中 `mtp_file` 或
`mlx_lm_extra_tensors.mtp_file` 指定的 sidecar。当前支持文本 dense Qwen3.5 系列
（含上述 Qwen3.8 导出）的 OptiQ affine 4-bit、group size 64 草稿头，要求
`mtp_policy: "optiq-int4-prequantized-gs64"`，避免把已转换 norms 的其他导出当作原始权重：

- 主模型仍按其逐层 4/8-bit 配置加载；sidecar 单独加载，不混入主分片索引。
- 根据 `.scales` 只转换打包投影；`mtp.fc` 和 norms 保持浮点精度。
- OptiQ 转换器保留原始 HF zero-centered norm 权重；加载 sidecar 时统一 `+1` 一次。
  主模型的 norms 已转换，不重复修改。不能把 `prequantized` 理解为 norms 已转换，
  也不能用 tensor mean 是否低于 0.5 来猜测是否转换。
- 使用上游 Qwen MTP draft model，共享主模型的 embedding/head，block size 2
  （每轮一个草稿 token），由上游完成目标验证和 GDN/KV cache 回滚。
- 当前上游 Qwen MTP 仅支持温度 0；其他温度直接使用普通生成，不创建 MTP iterator。
  MTP 缺失或下载未完成时也使用主模型。已存在但格式不匹配的 sidecar 会报错，
  可关闭 MTP 后继续使用主模型。
- 返回实际 proposed/accepted draft tokens、fallback reason 和 prefill/decode 时间。
  MTP 流式中间 token 数由文本重新编码估算，最终 usage 来自上游真实 token 账本。
- 不下载、复制或改写 HF_HOME 模型权重。只新增 Swift 适配层，不引入 Python 运行时。

sidecar 原始 norms 的契约依据官方 [OptiQ 0.5.19](https://pypi.org/project/mlx-optiq/0.5.19/)
发布包中的 `optiq/runtime/mtp_convert.py`（提取原始 `mtp.*`，norms 原样保留为 BF16）。
上游原生实现见 [Qwen35MTP.swift](https://github.com/ml-explore/mlx-swift-lm/blob/3.32.3/Libraries/MLXLLM/Models/Qwen35MTP.swift)。

设置 → Chat Models：启用 **Use Local Qwen MTP**，点击 **Use Temperature 0 for MTP**。
默认温度仍保留 0.6；快捷补全面板保留自己的温度。本地 HTTP API 的温度由请求决定，
发送 `"temperature": 0` 并启用上述 MTP 开关即可使用草稿头。

## 验证命令

Release 构建后，顺序执行（共用同一个测试可执行文件）：

```sh
python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release --chat-library

python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat
```

`--chat-library` 使用小型合成 MLX checkpoint 检查 4/8-bit 混合量化 logits、
主分片索引排除 MTP sidecar，以及 GDN swish 与完整注意力 sigmoid 的独立数值参考、MTP raw norms 转换、
混合缓存贪心 tokens 一致性和 sidecar 路径检查。
`--chat` 使用真实缓存模型验证多轮消息和取消后立即重启。
还可分别运行 `--chat-store`、`--chat-api`、`--tts-streaming` 验证 App Store、
本地 HTTP API 与实际 TTS/ASR 回转录。

真实模型 MTP 验证（需完整主权重和 sidecar）：

```sh
python3 Tests/run-asr-tests.py --derived-data .build/DerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat-mtp \
  --model mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit
```

`--chat-mtp` 预热两条路径后比较 128-token 贪心输出和实际吞吐，检查草稿接受率、
停止字符串、取消后立即重启和非零温度 fallback。速度与提示词和接受率有关，
测试打印实测值，不将其他运行时的 MTP 宣传速度当作 Swift App 的速度。

### 本机验证记录（2026-10-07，Release）

真实 `mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit`、128-token 编程提示，
两条路径预热后：普通生成约 24.2 tokens/s，MTP 约 35.0 tokens/s（约 1.45x），
草稿接受 56/70（80%），MTP prefill 约 0.41s。这是一次短测，不是通用性能保证。
多轮口令提取的完整输出和 token 用量一致；混合缓存微型 checkpoint 贪心 tokens 一致。

**数值限制：**真实 BF16 长回答可能与普通贪心回答逐字不同。目标模型的单 token
编译解码和双 token 批量验证使用不同计算路径；同一初始 cache 的比较中，最大 logit
差异为 0.15625。接近决策边界时，这种差异可能改变后续 continuation。
每个草稿仍由目标模型验证，但不能宣称真实 BF16 长生成与普通路径 bitwise 等价。
需要逐字复现普通路径时关闭 MTP。

流式停止会暂存可能属于 stop string 的后缀，避免分块输出泄漏停止标记的部分前缀。


## Metal 验证下的 Gemma-4 崩溃修复

2026-10-08 的 App 崩溃报告和开启 `MTL_DEBUG_LAYER=1` 的真实模型复现均指向
`Gather::eval_gpu` → `CommandEncoder::set_vector_bytes`。标量索引的形状/步幅向量
为空，旧实现把空指针交给 `setBytes`，触发 `bytes argument cannot be nil` 后 SIGABRT。
这是底层编码错误，Swift 的 `catch` 无法拦截，也不是窗口关闭退出流程。

当前官方 mlx-swift 0.32.3（底层 MLX v0.32.2）及核查时的 mlx-swift main
仍未携带该修复。`Scripts/backport-mlx-metal.py` 回移植官方 MLX 的空向量处理：
空向量绑定一个初始化为零的元素，非空向量保留原实现；kernel 的维度计数不变。
来源：[MLX device.h](https://github.com/ml-explore/mlx/blob/6553c47ba6500e877d7348dd576c021c43a22cf7/mlx/backend/metal/device.h)。

共享 Synomancer scheme 的 Build Pre-action 在编译前应用补丁，覆盖项目的
`.build/SourcePackages` 和标准 DerivedData 的 `SourcePackages`；额外的命令行
checkout 可通过脚本 `--source-packages` 指定。只修改 MLX 的源头和 framework
导出两个头文件，保留权限；重复构建不重复修改。依赖仍来自官方固定 release，
HF_HOME、权重及运行时模型实现不变。上游头文件若出现未知变化会拒绝补丁，
需要审查后调整；正式 release 带入同一修复后可以移除此构建预处理。

验证命令（先通过共享 scheme 完成 Release 构建，再顺序运行）：

```sh
python3 Tests/MLXMetalBackportTests.py
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --chat-library --metal-validation
python3 Tests/run-asr-tests.py --derived-data /private/tmp/SynomancerDerivedData \
  --source-packages .build/SourcePackages --configuration Release \
  --cache /path/to/workspace/.cache/huggingface/hub --chat --metal-validation \
  --model lmstudio-community/gemma-4-31B-it-MLX-8bit
```

`--chat-library` 包含非首轴标量 Gather 的形状和数值回归；真实 `--chat` 验证
多轮聊天、取消和立即重启。补丁脚本测试检查幂等、只读权限恢复，以及未知头文件
在任何写入前被拒绝。保留 Metal API 验证，不以关闭验证作为修复。

Gemma-4 VLM processor 的纯文本输出为 `[1, L]`，而现有聊天前缀缓存要求 `[L]`。
App 在没有图像、视频、音频且文本 mask 缺失或全部为 1 时将其归一化为 `[L]`；官方 Gemma
text-only prepare 会自行恢复批次维度。这样开启视觉支持后仍保留文本前缀复用，
有媒体的输入继续使用完整处理器输出。

本机开启 Metal API 验证的 Gemma-4 31B 8bit 聊天回归通过：多轮口令提取、
1024/1204 tokens 前缀命中、热/冷回答一致、修改系统提示词失效、缓存扩展、
取消后的相同请求重试和立即发送新请求均完成。关闭窗口行为未参与此崩溃复现。
同样开启验证的真实红杯识图通过，中文提示词与修复前未开启验证时的结果一致。
