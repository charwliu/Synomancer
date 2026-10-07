import SwiftUI
import UniformTypeIdentifiers

struct ImageStudioView: View {
    @StateObject private var store: ImageStudioStore
    @AppStorage(SpeechCacheAccess.homeKey) private var home = SpeechCacheAccess.defaultPath
    @Environment(\.dismiss) private var dismiss
    @State private var importing = false
    @State private var exporting = false
    @State private var importingControl = false
    @State private var exportingControl = false

    @MainActor init(store: ImageStudioStore? = nil) {
        _store = StateObject(wrappedValue: store ?? ImageStudioStore())
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("图像生成与编辑", systemImage: "photo.badge.plus").font(.title2.bold())
                Spacer()
                Button("关闭") { dismiss() }.disabled(store.busy)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
            HStack {
                Picker("本地模型", selection: $store.modelID) {
                    ForEach(store.models) { Text($0.id).tag($0.id) }
                }
                .disabled(store.busy)
                Button("刷新") { Task { await store.reload() } }.disabled(store.busy)
            }
            TextEditor(text: $store.prompt)
                .font(.body).frame(height: 90)
                .overlay(alignment: .topLeading) {
                    if store.prompt.isEmpty {
                        Text(store.reference == nil ? "描述想生成的照片…" : "描述对参考照片的修改…")
                            .foregroundStyle(.secondary).padding(6).allowsHitTesting(false)
                    }
                }
                .disabled(store.busy)
                .accessibilityLabel("图片描述或编辑指令")
            HStack {
                Button("选择参考照片…") { importingControl = false; importing = true }
                if store.reference != nil {
                    Button("移除参考照片") { store.reference = nil }
                }
                Spacer()
                Picker("尺寸", selection: $store.width) {
                    ForEach(Array(stride(from: 256, through: 2048, by: 32)), id: \.self) { size in
                        Text(verbatim: String(size)).tag(size)
                    }
                }.frame(width: 100)
                Text("×")
                Picker("高度", selection: $store.height) {
                    ForEach(Array(stride(from: 256, through: 2048, by: 32)), id: \.self) { size in
                        Text(verbatim: String(size)).tag(size)
                    }
                }.labelsHidden().frame(width: 75)
                Stepper("步数 \(store.steps)", value: $store.steps, in: 10...60, step: 5).frame(width: 140)
                TextField("随机种子", text: $store.seed).frame(width: 90)
            }.disabled(store.busy)
            controlOptions
            upscaleOptions
            HStack(alignment: .top, spacing: 16) {
                if let reference = store.reference { preview(reference, title: "参考照片") }
                if let map = store.control { preview(map.png, title: "控制图") }
                if let image = store.displayedPNG {
                    preview(image, title: store.showUpscaled && store.upscaled != nil ? "超分结果" : "生成原图")
                }
                if store.reference == nil && store.control == nil && store.result == nil {
                    ContentUnavailableView("本地图像", systemImage: "photo",
                        description: Text("输入描述生成照片，或选择照片并输入编辑指令。"))
                        .frame(maxWidth: .infinity, minHeight: 220)
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            if store.busy {
                if store.generating || store.upscaling { ProgressView(value: store.progress) }
                else { ProgressView().controlSize(.small) }
                Text(store.status).font(.callout).foregroundStyle(.secondary)
            } else if let result = store.result {
                Text("耗时 \(result.seconds.formatted(.number.precision(.fractionLength(1)))) 秒 · Seed \(result.seed)")
                    .font(.caption).foregroundStyle(.secondary)
                if store.showUpscaled, let output = store.upscaled {
                    Text("\(output.backend.title) · \(output.width)×\(output.height) · 超分耗时 \(output.seconds.formatted(.number.precision(.fractionLength(1)))) 秒")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = store.error {
                Text(error).foregroundStyle(.red).font(.callout).textSelection(.enabled)
            }
            HStack {
                Text("Qwen-Image-2.1 · Swift + MLX · 本地 HF_HOME").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if store.busy {
                    Button(store.stopping ? "正在停止…" : "停止") { store.stop() }.disabled(store.stopping)
                } else {
                    if let image = store.displayedPNG {
                        Button("继续编辑这张图") { store.reference = image; store.prompt = "" }
                        Button("保存 PNG…") { exportingControl = false; exporting = true }
                    }
                    Button(store.control != nil ? "按控制图生成" : store.reference == nil ? "生成照片" : "编辑照片") { store.generate() }
                        .buttonStyle(.borderedProminent)
                        .disabled(store.modelID.isEmpty || store.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || (store.control != nil && store.controlScale > 0 && !store.controlAvailable))
                }
            }
        }
        .padding(20)
#if os(macOS)
        .frame(minWidth: 820, idealWidth: 1000, minHeight: 700, idealHeight: 850)
#endif
        .task(id: home) { await store.reload() }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image]) { result in
            do {
                let url = try result.get()
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let data = try Data(contentsOf: url)
                guard Image(data: data) != nil else { throw LocalImageError.invalid("无法读取参考照片。") }
                if importingControl { store.setControlSource(data) } else { store.reference = data }; store.error = nil
            } catch { store.error = error.localizedDescription }
        }
        .fileExporter(isPresented: $exporting, document: GeneratedPNG(data: exportingControl ? store.control?.png ?? Data() : store.displayedPNG ?? Data()),
            contentType: .png, defaultFilename: exportingControl ? "Synomancer-control" : store.showUpscaled && store.upscaled != nil ? "Synomancer-upscaled" : "Synomancer-image") { result in
                if case .failure(let error) = result { store.error = error.localizedDescription }
            }
        .onChange(of: store.controlKind) { _, _ in store.refreshControl() }
        .onChange(of: store.width) { _, _ in store.refreshControl() }
        .onChange(of: store.height) { _, _ in store.refreshControl() }
        .onDisappear { store.stop() }
    }

    private var upscaleOptions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Picker("超分后端", selection: $store.upscaleBackend) {
                    ForEach(ImageUpscaleBackend.allCases) { Text($0.title).tag($0) }
                }.frame(maxWidth: 290)
                Picker("输出尺寸", selection: $store.upscaleSize) {
                    ForEach(ImageUpscaleSize.allCases) { Text($0.title).tag($0) }
                }.frame(maxWidth: 220)
                Button("超分生成图") { store.upscale() }
                    .disabled(store.result == nil || (store.upscaleBackend == .apple
                        ? !store.upscaleAvailability.appleReady : !store.upscaleAvailability.realESRGANReady))
                if store.upscaled != nil { Toggle("显示超分结果", isOn: $store.showUpscaled) }
            }
            if store.upscaleBackend == .apple {
                if !store.upscaleAvailability.appleSupported {
                    Text("当前系统或设备不支持 Apple 图片超分，请选择 RealESRGAN。")
                } else if !store.upscaleAvailability.appleReady {
                    HStack {
                        Text("首次使用需下载 Apple 系统模型；准备完成后在本地处理。")
                        Button("下载 Apple 超分模型") { store.prepareUpscaleModel() }
                    }
                }
            } else if !store.upscaleAvailability.realESRGANReady {
                Text("HF_HOME 中尚未找到 RealESRGAN_x4plus 超分模型。")
            }
            Text("始终从生成原图超分，建议使用 1024 原图。4K 按原比例输出长边 3840 像素；保存 PNG 导出当前显示的版本。")
        }
        .font(.caption)
        .disabled(store.busy)
    }

    private var controlOptions: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Picker("控制图处理", selection: $store.controlKind) {
                    ForEach(LocalControlMapKind.allCases) { Text($0.title).tag($0) }
                }.frame(maxWidth: 260)
                Button("选择控制图或照片…") { importingControl = true; importing = true }
                if let reference = store.reference {
                    Button("使用参考照片") { store.setControlSource(reference) }
                }
                if store.controlSource != nil { Button("移除控制图") { store.clearControl() } }
                if store.control != nil {
                    Button("保存控制图…") { exportingControl = true; exporting = true }
                }
            }
            if store.controlSource != nil {
                HStack {
                    Text("控制强度 \(store.controlScale.formatted(.number.precision(.fractionLength(2))))")
                    Slider(value: $store.controlScale, in: 0...2, step: 0.05).frame(maxWidth: 180)
                    Text("控制结束 \(store.controlEnd.formatted(.percent.precision(.fractionLength(0))))")
                    Slider(value: $store.controlEnd, in: 0.05...1, step: 0.05).frame(maxWidth: 180)
                }
                Text("启用控制图后按结构生成照片；参考照片用于提取深度或姿态。提示词请描述完整目标画面。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !store.controlAvailable {
                Text("HF_HOME 中尚未找到 Qwen-Image-2.1-Fun-Controlnet-Union 权重。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if store.controlKind == .depth && !store.depthAvailable {
                Text("深度提取需要 HF_HOME 中的 apple/coreml-depth-anything-v2-small 模型。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(store.busy)
    }

    private func preview(_ data: Data, title: String) -> some View {
        VStack(spacing: 8) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            if let image = Image(data: data) {
                image.resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: 270)
                    .background(.quaternary).clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }.frame(maxWidth: .infinity)
    }
}

private struct GeneratedPNG: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }
    let data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws { data = configuration.file.regularFileContents ?? Data() }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper { FileWrapper(regularFileWithContents: data) }
}
