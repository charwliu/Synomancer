import Foundation
import Combine

extension Notification.Name {
    static let hfModelDownloadCompleted = Notification.Name("Synomancer.HFModelDownloadCompleted")
}

@MainActor final class HFModelDownloadStore: ObservableObject {
    static let shared = HFModelDownloadStore()
    @Published var repository = "black-forest-labs/FLUX.2-dev" { didSet { if oldValue != repository { plan = nil } } }
    @Published var revision = "main" { didSet { if oldValue != revision { plan = nil } } }
    @Published var token = "" { didSet { if oldValue != token { plan = nil } } }
    @Published var allFiles = false { didSet { if oldValue != allFiles { plan = nil } } }
    @Published private(set) var plan: HFDownloadPlan?
    @Published private(set) var busy = false
    @Published private(set) var stopping = false
    @Published private(set) var status = ""
    @Published private(set) var error: String?
    @Published private(set) var fraction = 0.0
    @Published private(set) var received: Int64 = 0
    @Published private(set) var total: Int64 = 0
    private var task: Task<Void, Never>?
    private var requestID = UUID()

    func inspect() {
        guard !busy else { return }
        let repo = repository.trimmingCharacters(in: .whitespacesAndNewlines)
        let revision = revision.trimmingCharacters(in: .whitespacesAndNewlines)
        let token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let allFiles = allFiles
        let id = UUID(); requestID = id
        busy = true; stopping = false; error = nil; plan = nil; status = "正在读取模型文件列表…"; fraction = 0
        task = Task { [self] in
            defer { if requestID == id { busy = false; stopping = false; task = nil } }
            do {
                let result = try await HFModelDownloadService().prepare(repository: repo, revision: revision, token: token, allFiles: allFiles)
                try Task.checkCancellation()
                guard requestID == id else { return }
                plan = result; status = "文件列表已就绪，请点击下载"
            } catch {
                if requestID == id { if Task.isCancelled { status = "已停止" } else { self.error = error.localizedDescription; status = "检查失败" } }
            }
        }
    }
    func download() {
        guard !busy, let plan else { return }
        let token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let id = UUID(); requestID = id
        busy = true; stopping = false; error = nil; received = 0; total = plan.bytes; fraction = 0; status = "准备下载…"
        task = Task { [self] in
            defer { if requestID == id { busy = false; stopping = false; task = nil } }
            do {
                let cache = try await SpeechCacheAccess.authorizedWritableConfiguration()
                _ = try await HFModelDownloadService().download(plan, cache: cache, token: token) { [weak self] received, total, file, done, count in
                    guard let self, self.requestID == id, self.busy, !self.stopping else { return }
                    self.received = received; self.total = total
                    self.fraction = total > 0 ? min(1, Double(received) / Double(total)) : 0
                    self.status = "\(done)/\(count) · \(file)"
                }
                try Task.checkCancellation()
                if requestID == id {
                    fraction = 1; received = plan.bytes; status = "下载完成，已保存到 HF_HOME"
                    NotificationCenter.default.post(name: .hfModelDownloadCompleted, object: nil)
                }
            } catch {
                if requestID == id {
                    if Task.isCancelled || error is CancellationError { status = "已停止；再次点击下载可继续" }
                    else { self.error = error.localizedDescription; status = "下载中断；再次点击下载可继续" }
                }
            }
        }
    }
    func stop() { guard busy else { return }; stopping = true; task?.cancel() }
}
