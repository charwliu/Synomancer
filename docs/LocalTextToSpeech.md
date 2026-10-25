# 本地朗读

设置 → Read Aloud → Speech Model 选择 **Local Qwen3-TTS 1.7B**。
缓存使用设置 → Hugging Face → **HF_HOME**，与 ASR 共用，默认
`~/.cache/huggingface/hub`。首次需要通过 **Choose HF_HOME Folder…**
授权外部目录；若已授权同一目录，无需重选。
若首次朗读/录音无法读取 HF_HOME，macOS 会弹出授权文件夹选择器，选择缓存根目录
后继续原操作。取消选择器会取消当前操作；已授权且可读取时不重复弹出。

当前接入 **Qwen3-TTS 1.7B CustomVoice**，支持其九个预设音色。
Automatic 语言根据文字选择中文或英文；自动音色分别使用 Vivian / Ryan。
可指定中文、英文或预设音色。系统语音可通过 Speech Model 切换回来。

## 自动朗读模式

聊天工具栏的 **自动朗读** 开关，或设置 → Read Aloud → **自动朗读**，
开启后会在新回答完成并保存后自动朗读正式答案，沿用所选 Speech Model、
语言和音色。开关默认关闭，选择会保留到下次启动。
思考段不朗读；打开旧会话、取消生成、失败的响应及快捷补全不会自动播放。
发送新问题、新建或切换会话、关闭模式时会停止当前自动朗读。
macOS 的朗读状态与停止按钮位于 App 顶部工具栏，
在“自动朗读”和“新会话”按钮左侧；聊天区保留各条消息的朗读入口。
停止本次播放会保留自动朗读模式，供下一次回答使用。
TTS 音频仍按下面的流式方式播放；开始合成前会等待文字回答完成。

## 流式播放

点击消息朗读按钮后，使用 `synthesizeStream` 持续接收音频块，直接排入
AVAudioEngine / AVAudioPlayerNode 的 24 kHz 单声道 Float32 PCM 队列。
App 首块生成 6 个真实 codec 帧，后续每 12 帧生成音频；每帧代表约 80 ms，
首块包含正常的因果启动裁剪。播放前预缓冲至少 0.6 秒；最后不足预缓冲长度的
短句也会在生成结束时开始播放。实际首声延迟还包括模型准备、文本预填充和设备启动。
语音 codec 的注意力 K/V、前置卷积及波形卷积历史跨音频块保留，避免每隔几个词
重新初始化解码器。启动不足四帧时等待真实帧，不补虚构的零编号 codec 帧。
样本位置根据实际波形长度计算，每个有效样本只输出一次。
无需等待完整句子或整篇答案，也不写临时 WAV 文件。

朗读前清理 Markdown 标记，长文本按句尾/空白切分，每段最多 256 个字符，
减少短段之间反复预填充及音调重置。播放队列限制为约四秒 PCM加一个音频块，
为解码抖动和下一文本段的预填充保留余量。当前段的模型输出由音频流缓冲，
播放器仍使用同一个节点顺序排队。若队列确实耗尽，暂停并重新预缓冲后继续播放。
macOS/visionOS App 顶部工具栏显示准备/生成状态和停止按钮；iOS 在聊天区显示。
生成和播放均可停止，错误会显示提示。
播放器保留整段回答最后 20 毫秒 PCM；生成完成后只对这段末尾音频做余弦淡出，
从真实末尾振幅平滑降到零，再排入 50 毫秒静音后释放音频设备。
中间的音频块和文本段不单独淡出，不会因这一处理新增句内停顿；
模型用空音频块标记 EOS 时也会正确释放保留的尾部样本。
停止会取消生产任务、清空播放队列；旧 GPU 任务退出后才能复用模型开启新请求。
生成进度超过 120 秒没有更新会自动结束并提示错误。

采用模型默认采样参数；贪心采样在实测中可能漏读，不用于 App 的朗读。

## 模型与运行配置

只读加载 Hugging Face 缓存中的 CustomVoice 模型和语音 tokenizer。
优先使用模型 snapshot 的 `speech_tokenizer`，也支持独立的
`Qwen/Qwen3-TTS-Tokenizer-12Hz` 缓存。无需复制缓存文件，不自动下载模型。
本机已有 `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`，以及配套 codec。
请选择包含全部模型和 blobs 的 hub 根目录。

模型通过 Swift + MLX 在本机执行，不上传朗读文本，不改写源模型，
不修改聊天推理的全局 GPU 内存设置。加载 BF16 权重时，在内存中将 Talker
和 Code Predictor 的线性层量化到 8bit；语音 codec 保持原精度。
首次需要加载、量化和预热，后续复用模型。
较长中文回读中，Talker 4bit 出现内容缺失；当前使用 8bit 保留完整朗读。

Xcode 中选择 **Synomancer-Release** scheme 运行。该 scheme 使用 Release 配置，
不附加调试器；原来的 Synomancer scheme 保留 Debug 用于开发。
命令行构建可使用：

```sh
xcodebuild -project Synomancer.xcodeproj -scheme Synomancer-Release \
  -configuration Release -destination 'platform=macOS' \
  -derivedDataPath .build/DerivedData \
  -clonedSourcePackagesDirPath .build/SourcePackages \
  -disableAutomaticPackageResolution -onlyUsePackageVersionsFromResolvedFile build
```

构建产物为 `.build/DerivedData/Build/Products/Release/Synomancer.app`。

## 验证

`Tests/AutomaticResponseSpeechTests.swift` 验证模式开关、思考过滤、取消和替换任务。
`Tests/SpeechPlaybackTailTests.swift` 验证末尾淡出、空 EOS、短音频、跨块样本连续性，
并可使用 `.build/TTSTestAudio` 中的真实中英文 TTS 音频验证末尾处理。
`--chat-store` 用真实聊天推理和模拟语音输出验证完成回调只触发一次、历史不重播、
取消或错误不朗读；`--playback` 用真实本地 TTS 和静音输出验证自动启动和停止播放。

构建后，运行真实离线中英文流式合成、时间测量与 ASR 内容回读：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release \
  --cache ~/.cache/huggingface/hub --tts-streaming
```

此测试只读加载 TTS 和 ASR 缓存，测量非流式 RTF、流式首块/首个非静音块延迟
及 RTF，验证连续 codec 帧、有限数值、最终块和 ASR 内容关键字，不播放声音。
测试音频位于被 Git 忽略的 `.build/TTSTestAudio`。
`--tts` 保留非流式合成、音频解码及取消检查。

使用相同参数将 `--tts-streaming` 换为 `--playback`，可通过真实 App 服务和静音
AVAudioEngine 验证首段完成前开始播放、停止后立即重启、完整播放及完成回调。
`--components-only` 验证缓存布局和 ASR 缓冲区回归。
实际扬声器听感和目录授权需手动验证。
上游版本与适配记录见 `Libraries/Qwen3ASR/SOURCE.md`。

## 接缝回归

```sh
python3 Tests/run-asr-tests.py \
  --derived-data .build/DerivedData --source-packages .build/SourcePackages \
  --configuration Release --cache ~/.cache/huggingface/hub \
  --tts-continuity
```

用真实 speech tokenizer 权重和同一组确定性 codec IDs，对比整段和分块解码，
包括启动不足四帧、不规则块和卷积上下文窗口切换。此数值测试不代表语音内容测试。
96 帧参考波形为 181,440 个样本；原路径额外产生 1,920 个启动样本，
对齐后的相对 RMS 误差约 0.73。修复后样本数完全一致，相对误差约 `4e-6`，
不通过重复样本、插入静音或淡化处理掩盖解码错误。
实际中英文内容使用 `--tts-streaming` 的本地 ASR 回读检查；
`--playback` 静音播放完整多段答案，并检查播放队列 underrun 次数。
