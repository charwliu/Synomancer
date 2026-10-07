# 本地 Qwen-Image-2.1

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

## 实现

Services/ImageGeneration 独立于聊天模型注册表：

- 官方 MLXVLM.Qwen3VL 提供视觉、M-RoPE 和 DeepStack 编码。
  在本次请求的私有模型中替换最后的 RMSNorm 和词表投影，
  得到图像模型训练使用的末层归一化之前的隐藏状态。
  使用模型的原始文本模板，不应用通用聊天模板。
- DiT 实现单流块因果注意力、零中心 RMSNorm 文本投影、共享 scale/gate
  调制和二维图像 RoPE。第一次去噪计算提示词/参考图 K/V，后续步骤只计算目标图。
  缓存属于当前请求，不会用于下一次请求。
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
缓存不保留视觉编码器权重，也不跨请求复用去噪 KV；不改变采样精度或步数。
此优化主要减少相同条件下重复尝试的准备时间，首次请求和去噪耗时不变，
尚未测量本次改动的端到端提速比例。大尺寸编辑可手动降低生成尺寸，
再使用超分获得高分辨率结果；这会改变细节，应由用户选择。

### 去噪调制广播

首步建立文本/参考图 K/V 后，后续步骤只处理目标 token。此时四组
scale/gate 及输出调制均使用相同目标行，直接保留 `[1, 1, D]` 广播，
不再通过 `which` 展开为 `[1, tokenCount, D]`。首步仍保留原来的
按 token 选择逻辑。适用于普通生成、参考图编辑和 ControlNet；不改变
精度或采样步数。Release 编译验证之外尚未进行本次 GPU 性能及图像对比。
