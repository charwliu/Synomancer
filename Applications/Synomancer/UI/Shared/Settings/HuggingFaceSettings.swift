import SwiftUI
import UniformTypeIdentifiers

struct HuggingFaceSettings: View {
    @AppStorage(SpeechCacheAccess.homeKey) private var home = SpeechCacheAccess.defaultPath
    @State private var selecting = false
    @State private var errorMessage: String?
    @StateObject private var downloads = HFModelDownloadStore.shared
    var onDownloadComplete: () -> Void = {}

    var body: some View {
        Section("Hugging Face") {
            TextField("HF_HOME", text: $home)
                .autocorrectionDisabled()
                .disabled(downloads.busy)
            Button("Choose HF_HOME Folder…") { selecting = true }.disabled(downloads.busy)
            Button("Use Default HF_HOME") { home = SpeechCacheAccess.defaultPath }.disabled(downloads.busy)
            Text("聊天、图像、ASR 和 TTS 共用此缓存。选择一次文件夹以授予访问权限；支持 huggingface 或 hub 根目录。")
                .font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("下载 HF 模型") {
                TextField("仓库名（namespace/model）", text: $downloads.repository).autocorrectionDisabled().disabled(downloads.busy)
                TextField("Revision", text: $downloads.revision).autocorrectionDisabled().disabled(downloads.busy)
                SecureField("HF Token（可选）", text: $downloads.token).disabled(downloads.busy)
                Text("Token 仅保留在本次 App 会话中。受限仓库还需在 Hugging Face 网页接受访问条款。")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("下载全部仓库文件", isOn: $downloads.allFiles).disabled(downloads.busy)
                Text("默认下载模型文件；已知 FLUX 仓库只下载分组件，避免重复整包。完成的文件会复用，中断可继续下载。模型需与 App 支持的格式兼容。")
                    .font(.caption).foregroundStyle(.secondary)
                if let plan = downloads.plan {
                    Text("\(plan.files.count) 个文件 · \(ByteCountFormatter.string(fromByteCount: plan.bytes, countStyle: .file))")
                    Text("Commit \(String(plan.commit.prefix(12)))").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Button("检查模型") { downloads.inspect() }.disabled(downloads.busy)
                    Button("下载到 HF_HOME") { downloads.download() }.disabled(downloads.busy || downloads.plan == nil)
                    if downloads.busy { Button(downloads.stopping ? "正在停止…" : "停止") { downloads.stop() }.disabled(downloads.stopping) }
                }
                if downloads.busy {
                    ProgressView(value: downloads.fraction)
                    if downloads.total > 0 {
                        Text("\(ByteCountFormatter.string(fromByteCount: downloads.received, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: downloads.total, countStyle: .file))")
                            .font(.caption).monospacedDigit()
                    }
                }
                if !downloads.status.isEmpty { Text(downloads.status).font(.caption).textSelection(.enabled) }
                if let error = downloads.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .hfModelDownloadCompleted)) { _ in onDownloadComplete() }
        .onAppear {
            // Preserve directory selections made before the unified setting existed.
            if UserDefaults.standard.string(forKey: SpeechCacheAccess.homeKey) == nil {
                home = SpeechCacheAccess.configuration().path
            }
        }
        .fileImporter(isPresented: $selecting, allowedContentTypes: [.folder]) { result in
            do {
                let root = try result.get()
                let accessing = root.startAccessingSecurityScopedResource()
                defer { if accessing { root.stopAccessingSecurityScopedResource() } }
                try SpeechCacheAccess.save(root)
                home = SpeechCacheAccess.normalized(root.path)
            } catch { errorMessage = error.localizedDescription }
        }
        .fileDialogDefaultDirectory(URL(fileURLWithPath: SpeechCacheAccess.configuration().path, isDirectory: true))
        .alert("HF_HOME", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { errorMessage = nil }
        } message: { Text(errorMessage ?? "") }
    }
}
