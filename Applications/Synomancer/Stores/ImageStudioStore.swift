import Foundation
import Combine
import ImageIO

@MainActor
final class ImageStudioStore: ObservableObject {
    @Published var models: [LocalImageModel] = []
    @Published var modelID = "" {
        didSet {
            guard oldValue != modelID else { return }
            steps = selectedFamily.defaultSteps
            guidance = selectedFamily.defaultGuidance
            trueCFGScale = selectedFamily.defaultTrueCFGScale
            if !selectedFamily.supportsControl { clearControl() }
            if !selectedFamily.supportsReference { reference = nil }
        }
    }
    var selectedFamily: LocalImageFamily { models.first(where: { $0.id == modelID })?.family ?? .qwenImage21 }
    @Published var prompt = ""
    @Published var negativePrompt = ""
    @Published var trueCFGScale = 2.0
    @Published var reference: Data? {
        didSet {
            if describingReference {
                promptRequest = nil; task?.cancel(); status = ""
            }
            promptDraft = ""; promptError = nil
            if let reference { matchReferenceSize(reference) }
        }
    }
    @Published var promptModels: [LocalChatModel] = []
    @Published var promptModelID = ""
    @Published var promptDraft = ""
    @Published private(set) var promptError: String?
    @Published private(set) var describingReference = false
    @Published var result: LocalImageResult?
    @Published var upscaleBackend: ImageUpscaleBackend = .apple
    @Published var upscaleSize: ImageUpscaleSize = .fourK
    @Published var showUpscaled = true
    @Published private(set) var upscaled: ImageUpscaleResult?
    @Published private(set) var upscaleAvailability = ImageUpscaleAvailability()
    @Published private(set) var upscaling = false
    @Published private(set) var preparingUpscale = false
    var displayedPNG: Data? { showUpscaled ? upscaled?.png ?? result?.png : result?.png }
    @Published var width = 768
    @Published var height = 768
    @Published var steps = 40
    @Published var guidance = 3.5
    @Published var seed = "42"
    @Published var controlSource: Data?
    @Published var control: LocalControlMapResult?
    @Published var controlKind: LocalControlMapKind = .supplied
    @Published var controlScale = 1.0
    @Published var controlEnd = 1.0
    @Published private(set) var controlAvailable = false
    @Published private(set) var depthAvailable = false
    @Published private(set) var extracting = false
    @Published private(set) var generating = false
    @Published private(set) var stopping = false
    @Published private(set) var status = ""
    @Published private(set) var progress = 0.0
    @Published var error: String?
    private var task: Task<Void, Never>?
    private var revision = UUID()
    private var extraction: Task<Void, Never>?
    private var extractionRevision = UUID()
    private var promptRequest: UUID?
    typealias PromptGenerator = @Sendable (String, Data, SpeechCacheAccess.Configuration) async throws -> String
    private let promptGenerator: PromptGenerator

    init(promptGenerator: @escaping PromptGenerator = { model, image, cache in
        try await LocalReferencePromptService.generate(modelID: model, image: image, cache: cache)
    }) {
        self.promptGenerator = promptGenerator
    }

    var busy: Bool { generating || extracting || upscaling || preparingUpscale || describingReference }

    func describeReference() {
        guard !busy, let image = reference else { return }
        guard promptModels.contains(where: { $0.id == promptModelID && $0.supportsImages }) else {
            promptError = LocalReferencePromptService.PromptError.modelUnavailable.localizedDescription
            return
        }
        let id = UUID(), model = promptModelID, cache = SpeechCacheAccess.configuration()
        promptRequest = id; describingReference = true; stopping = false
        promptDraft = ""; promptError = nil; status = "正在识别参考图并生成提示词…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let draft = try await promptGenerator(model, image, cache)
                try Task.checkCancellation()
                if promptRequest == id {
                    promptDraft = draft; status = "提示词已生成，请确认后使用"
                }
            } catch is CancellationError {
                if promptRequest == id { status = "已停止生成提示词" }
            } catch {
                if promptRequest == id { promptError = error.localizedDescription; status = "提示词生成失败" }
            }
            // No replacement request can start until this worker has settled.
            describingReference = false; stopping = false; task = nil; promptRequest = nil
        }
    }

    func usePromptDraft() {
        guard !busy, reference != nil, !promptDraft.isEmpty else { return }
        prompt = promptDraft
        promptDraft = ""
    }

    private func matchReferenceSize(_ data: Data) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let pixelWidth = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let pixelHeight = properties[kCGImagePropertyPixelHeight] as? NSNumber else { return }
        var w = pixelWidth.doubleValue, h = pixelHeight.doubleValue
        guard w.isFinite, h.isFinite, w > 0, h > 0 else { return }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        if (5...8).contains(orientation) { swap(&w, &h) }
        // Fit within model limits; retain the reference dimensions when possible.
        // Very narrow panoramas still require the model's minimum short edge.
        let scale = min(2048 / max(w, h), max(1, 256 / min(w, h)))
        func aligned(_ value: Double) -> Int {
            min(2048, max(256, Int((value * scale / 32).rounded()) * 32))
        }
        width = aligned(w)
        height = aligned(h)
    }

    func reload() async {
        guard !busy else { return }
        do {
            let available = try await LocalChatService.shared.models(cache: SpeechCacheAccess.configuration())
            try Task.checkCancellation()
            promptModels = available.filter(\.supportsImages)
            if !promptModels.contains(where: { $0.id == promptModelID }) {
                promptModelID = promptModels.first?.id ?? ""
            }
            promptError = nil
        } catch is CancellationError { return }
        catch { promptModels = []; promptModelID = ""; promptError = error.localizedDescription }
        upscaleAvailability = await LocalImageUpscaleService.shared.availability(cache: SpeechCacheAccess.configuration())
        do {
            models = try await LocalImageService.shared.models(cache: SpeechCacheAccess.configuration())
            if !models.contains(where: { $0.id == modelID }) { modelID = models.first?.id ?? "" }
            let availability = try await LocalImageService.shared.controlAvailability(cache: SpeechCacheAccess.configuration())
            controlAvailable = availability.union; depthAvailable = availability.depth
            error = models.isEmpty ? LocalImageError.unavailable.errorDescription : nil
        } catch { self.error = error.localizedDescription }
    }

    func generate() {
        guard !busy else { return }
        guard let value = UInt64(seed) else { error = "随机种子必须是非负整数。"; return }
        let request = LocalImageRequest(modelID: modelID, prompt: prompt, reference: reference,
            width: width, height: height, steps: steps, seed: value,
            control: control?.png, controlScale: Float(controlScale), controlEnd: Float(controlEnd), guidance: Float(guidance),
            negativePrompt: selectedFamily.supportsNegativePrompt ? negativePrompt : "", trueCFGScale: Float(trueCFGScale))
        let cache = SpeechCacheAccess.configuration()
        let generation = UUID()
        revision = generation
        generating = true; stopping = false; progress = 0; error = nil; status = "准备生成…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let output = try await LocalImageService.shared.generate(request, cache: cache) { [weak self] text, amount in
                    Task { @MainActor [weak self] in
                        guard let self, self.revision == generation, self.generating, !self.stopping else { return }
                        self.status = text; self.progress = amount
                    }
                }
                try Task.checkCancellation()
                guard revision == generation else { return }
                result = output; upscaled = nil; status = "完成"; progress = 1
            } catch is CancellationError {
                if revision == generation { status = "已停止" }
            } catch {
                if revision == generation { self.error = error.localizedDescription; status = "生成失败" }
            }
            if revision == generation { generating = false; stopping = false; task = nil }
        }
    }

    func stop() {
        if extracting {
            extraction?.cancel(); stopping = true; status = "正在停止提取…"
            return
        }
        guard generating || upscaling || preparingUpscale || describingReference else { return }
        stopping = true; status = "正在停止…"; task?.cancel()
    }

    func prepareUpscaleModel() {
        guard !busy else { return }
        preparingUpscale = true; stopping = false; error = nil
        status = "正在下载 Apple 超分模型…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try await LocalImageUpscaleService.shared.prepareAppleModel()
                status = "Apple 超分模型已就绪"
            } catch is CancellationError { status = "已停止等待模型下载" }
            catch { self.error = error.localizedDescription; status = "模型准备失败" }
            upscaleAvailability = await LocalImageUpscaleService.shared.availability(cache: SpeechCacheAccess.configuration())
            preparingUpscale = false; stopping = false; task = nil
        }
    }

    func upscale() {
        guard !busy, let original = result else { return }
        let backend = upscaleBackend, size = upscaleSize, cache = SpeechCacheAccess.configuration()
        let operation = UUID(); revision = operation
        upscaling = true; stopping = false; error = nil; progress = 0; status = "准备超分…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let output = try await LocalImageUpscaleService.shared.upscale(original.png, backend: backend,
                    size: size, cache: cache) { [weak self] text, amount in
                        Task { @MainActor [weak self] in
                            guard let self, revision == operation, upscaling, !stopping else { return }
                            status = text; progress = amount
                        }
                    }
                try Task.checkCancellation()
                upscaled = output; showUpscaled = true; status = "超分完成"; progress = 1
            } catch is CancellationError { status = "已停止超分" }
            catch { self.error = error.localizedDescription; status = "超分失败" }
            upscaling = false; stopping = false; task = nil
        }
    }

    func setControlSource(_ data: Data) {
        guard !generating && !describingReference else { return }
        controlSource = data
        refreshControl()
    }

    func refreshControl() {
        guard !generating && !describingReference, let source = controlSource else { return }
        extraction?.cancel()
        let id = UUID(); extractionRevision = id
        let kind = controlKind, w = width, h = height, cache = SpeechCacheAccess.configuration()
        control = nil; extracting = true; stopping = false; error = nil
        status = kind == .supplied ? "读取控制图…" : "正在\(kind.title)…"
        extraction = Task { [weak self] in
            guard let self else { return }
            do {
                let map = try await LocalControlMapService.shared.extract(source, kind: kind, width: w, height: h, cache: cache)
                try Task.checkCancellation()
                guard extractionRevision == id else { return }
                control = map
                status = map.people.map { "已提取 \($0) 人的人体姿态" } ?? "控制图已就绪"
            } catch is CancellationError {
                if extractionRevision == id { status = "已停止提取" }
            } catch {
                if extractionRevision == id { self.error = error.localizedDescription; status = "控制图提取失败" }
            }
            if extractionRevision == id { extracting = false; stopping = false; extraction = nil }
        }
    }

    func clearControl() {
        guard !generating && !describingReference else { return }
        extraction?.cancel(); extractionRevision = UUID(); extraction = nil
        extracting = false; stopping = false; control = nil; controlSource = nil
        status = ""; error = nil
    }
}
