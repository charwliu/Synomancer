import SwiftUI
import UniformTypeIdentifiers

struct HuggingFaceSettings: View {
    @AppStorage(SpeechCacheAccess.homeKey) private var home = SpeechCacheAccess.defaultPath
    @State private var selecting = false
    @State private var errorMessage: String?

    var body: some View {
        Section("Hugging Face") {
            TextField("HF_HOME", text: $home)
                .autocorrectionDisabled()
            Button("Choose HF_HOME Folder…") { selecting = true }
            Button("Use Default HF_HOME") { home = SpeechCacheAccess.defaultPath }
            Text("聊天、图像、ASR 和 TTS 共用此缓存。选择一次文件夹以授予访问权限；支持 huggingface 或 hub 根目录。")
                .font(.caption).foregroundStyle(.secondary)
        }
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
