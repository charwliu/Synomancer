# Hugging Face 模型下载

设置 → Hugging Face → **下载 HF 模型**。输入 `namespace/model` 和 Revision（默认 `main`），先点击“检查模型”查看文件数量、总大小和固定的 commit，再点击“下载到 HF_HOME”。默认缓存为 `~/.cache/huggingface/hub`，与聊天、图像、ASR、TTS 共用。

首次下载需要在 macOS 文件夹选择器中选择 HF_HOME，授予写入权限；此前的模型读取授权仅允许读取。可以选择 `huggingface` 根目录或 `hub` 目录，前者的模型保存到其 `hub` 子目录。后续下载复用保存的授权。

受限或私有仓库可以填写有访问权限的 HF Token。Token 仅保存在 App 本次运行的内存中，不写入设置或日志。受限仓库需要先在 Hugging Face 网页接受访问条款。

FLUX.2-dev 等受限仓库在未授权的文件列表中可能将 LFS 哈希隐藏为 `********`。App 会用不跟随 CDN 重定向的 HEAD 请求读取真实哈希、文件大小和 commit；无权限时提示填写 Token 和申请仓库访问，不把隐藏哈希误报为损坏文件，也不会使用 Git LFS 指针的哈希校验权重。若 App 无法读取终端保存的 HF 凭据，请在下载区域填写 Token。参考 [HF 受限模型访问说明](https://huggingface.co/docs/hub/models-gated)。

下载采用原生 Swift 和 Hugging Face Swift SDK 获取文件列表，串行流式写入缓存；不调用 Python 或 `hf` CLI。关闭设置窗口后下载继续，点击“停止”保留当前文件的 `.incomplete`。网络中断后也保留收到的字节，再次点击下载时通过 HTTP Range 续传；若服务器不支持 Range，则安全重下当前文件。已完成的文件直接复用。若仓库 commit 已改变，新版本的不同文件可能需要重新下载。

文件完成时检查大小和哈希：LFS 文件使用 SHA-256，普通文件使用 Git blob SHA-1。校验失败的当前文件不会发布为完成文件；其他完成的文件保留。缓存使用标准 `models--namespace--model/{blobs,snapshots,refs}` 结构和相对符号链接，可与 HF CLI 共用。

默认筛选常见模型文件。官方 FLUX.1-dev、FLUX.2-dev 和 mlx-community/FLUX.2-klein-9B 使用分组件文件，排除根目录重复整包；“下载全部仓库文件”可覆盖筛选。官方 FLUX.2-dev 的 BF16 分组件权重约 112.81 GB，应预留足够空间。下载器可以下载 App 尚不支持的架构；下载完成不代表该模型能用于推理。

完成后自动刷新设置和空闲图像工作室的模型列表。FLUX.2-dev 原生管线的支持与验证范围见 [LocalImages.md](LocalImages.md)。

## 验证

```sh
python3 Tests/run-hf-download-tests.py --derived-data /private/tmp/SynomancerDerivedData
.build/HFModelDownloadTests --real .build/HFDownloadRealCache
```

第一条使用真实本机 HTTP 连接，覆盖中途断流后的准确 Range 续传、已完成文件复用、忽略 Range 的服务器、校验失败、实时取消，以及残留 partial 不覆盖完整 blob。第二条仅从公开 `hf-internal-testing/tiny-random-gpt2` 下载 `config.json` 和 `tokenizer.json`，验证 HF 元数据、HTTPS 重定向、哈希和缓存复用，不下载模型权重。测试输出和缓存位于忽略 Git 的 `.build` 目录。

HTTP 回归还覆盖隐藏/缺失 LFS 哈希的 HEAD 回退、403 权限提示、HEAD 哈希/大小/commit 不匹配，以及检查阶段不下载权重或跟随 CDN。可在已有 `HF_TOKEN` 或标准 HF 凭据的环境中运行 `.build/HFModelDownloadTests --real-flux-metadata`，检查官方 FLUX.2-dev 的实际分片元数据（含超过 4 GB 的编码器分片），不下载权重。
