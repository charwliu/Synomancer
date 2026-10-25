# 本地语音输入

Synomancer 使用 Swift + MLX 运行 Qwen3-ASR 1.7B。
点击输入框的波形按钮，授予麦克风权限，模型准备完毕后开始录音。
文字在录音过程中持续填入输入框，不需要点击停止才显示；用户仍需自行发送。
停止按钮用于结束录音并完成最后一个片段，关闭输入视图会取消当前工作。

每次转写完成后等待约 2 秒，再识别新的缓冲音频，因此实际更新时间是
这个间隔加上推理时间。片段末尾的暂定文字会修正；检测到约 0.8 秒静音，
或片段达到 15 秒后固定当前片段，再识别下一段。这是周期重识别，不是
模型逐 token 的原生流式推理。持续无停顿语音在固定分段边界附近可能降低准确度。

音频仅保留在有上限的内存缓冲区中，不保存录音文件，也不上传录音。
推理前混合成单声道并重采样到 16 kHz，不使用 Apple Speech。

## 使用已有 Hugging Face 缓存

设置中的 **Hugging Face → HF_HOME** 是聊天/ASR/TTS 共用的唯一缓存设置，
默认 `~/.cache/huggingface/hub`。通过 **Choose HF_HOME Folder…** 可以选择
`huggingface`、`hub`、模型仓库目录或可独立读取的模型目录。
macOS 建议选择 `hub` 根目录：snapshot 中的权重通常通过符号链接指向
仓库的 `blobs`，沙盒需要同时获得这两部分的读取授权。选择的目录通过
只读 security-scoped bookmark 保存。

本机缓存目录为 `~/.cache/huggingface/hub`，已有
`mlx-community/Qwen3-ASR-1.7B-bf16`。
读取这份缓存时，将文本解码器矩阵在内存中量化为 4bit，以降低推理成本；
音频编码器保留原始 dtype。缓存文件保持不变，不写回量化权重，不重复下载。

支持标准 `models--owner--repo/refs/main`、`snapshots/<revision>` 和 `blobs`
布局，检查 tokenizer 和分片完整性。当前支持 1.7B 的 mlx-community BF16、
aufklarer MLX 4bit/8bit 格式；其他导出格式可能不兼容。
用户指定的缓存失效或不完整时会报错，不会自动下载另一个模型。
更换缓存设置会在下一次语音输入时重新加载模型。

ASR 和 TTS 都只读取配置的 HF_HOME，不自动搜索其他路径或下载模型。
该设置既可指向 HF 缓存父目录，也可直接指向 hub。
输入路径不会自动授予沙盒权限，首次使用外部目录需通过文件夹选择器授权。
修改路径后，两者会在下次使用时重新加载；旧目录的 bookmark 不会用于新路径。
之前已选择的共享缓存目录与授权会迁移到统一设置。
模型常驻进程内，不更改聊天推理的全局 MLX 内存限制。

## 验证

先用 Xcode 构建 macOS Debug。无需模型的缓存与缓冲区回归：

```sh
python3 Tests/run-asr-tests.py --derived-data /path/to/Synomancer-DerivedData --components-only
```

使用本机缓存进行实际离线转写：

```sh
python3 Tests/run-asr-tests.py \
  --derived-data /path/to/Synomancer-DerivedData \
  --cache ~/.cache/huggingface/hub \
  --audio /path/to/speech.wav --offline
```

加 `--expected-words word1 word2` 校验已知录音中的关键字。
使用 speech-swift 上游的 20 秒 `Tests/Qwen3ASRTests/Resources/test_audio.wav`
时可加 `--streaming`，验证音频结束前的多次部分结果修正。
测试也验证已取消推理任务抛出 `CancellationError`。
测试不申请麦克风权限；实际采集、设置目录选择和授权持久化需在 App 中验证。
不要将模型权重或私人录音提交到 Git。

语音库的来源、版本与适配说明见 `Libraries/Qwen3ASR/SOURCE.md`。

统一 HF_HOME 配置回归（独立 UserDefaults 域，不修改 App 设置）：

```sh
swiftc -swift-version 6 -parse-as-library \
  -module-cache-path /private/tmp/synomancer-hf-home-module-cache \
  Applications/Synomancer/Services/SpeechCacheAccess.swift \
  Tests/HFHomeConfigurationTests.swift -o /private/tmp/synomancer-hf-home-tests
/private/tmp/synomancer-hf-home-tests
```

若首次朗读/录音无法读取 HF_HOME，macOS 会直接弹出授权文件夹选择器，
选择缓存根目录后继续原操作。取消选择器会取消当前操作；已授权且可读取时不重复弹出。
