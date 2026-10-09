import Foundation
import HuggingFace
import CryptoKit
import Darwin

struct HFDownloadFile: Codable, Sendable {
    let path: String
    let size: Int64
    let blob: String
    let lfs: Bool
}

struct HFDownloadPlan: Sendable {
    let repository: String
    let revision: String
    let commit: String
    let files: [HFDownloadFile]
    var bytes: Int64 { files.reduce(0) { $0 + $1.size } }
}

enum HFDownloadError: LocalizedError {
    case invalid(String), http(Int), integrity(String), accessRequired(String)
    var errorDescription: String? {
        switch self {
        case .invalid(let text): text
        case .http(let status): "HF 下载请求失败（HTTP \(status)）。401/403 时请填写有访问权限的 Token，并在 HF 网页接受仓库条款。"
        case .integrity(let path): "文件校验失败：\(path)。已保留其他完成的文件，请重试。"
        case .accessRequired(let repository): "无法访问 \(repository) 的模型权重。请在 HF 网页接受仓库访问条款，并填写有该仓库读取权限的 HF Token 后重新检查模型。"
        }
    }
}

struct HFModelDownloadService: Sendable {
    let host: URL
    let session: URLSession
    init(host: URL = URL(string: "https://huggingface.co")!, session: URLSession = .shared) {
        self.host = host; self.session = session
    }
    static func validPath(_ path: String) -> Bool {
        !path.isEmpty && !path.contains("\\") && !path.hasPrefix("/")
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static func repository(_ value: String) throws -> HuggingFace.Repo.ID {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ part in
            !part.isEmpty && part != "." && part != ".." && part.count <= 96
                && part.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "-_.".contains($0)) }
        }), let repo = HuggingFace.Repo.ID(rawValue: value) else {
            throw HFDownloadError.invalid("仓库名需为 namespace/model，例如 black-forest-labs/FLUX.2-dev。")
        }
        return repo
    }
    static func cacheRoot(_ selected: URL) -> URL {
        if selected.lastPathComponent == "huggingface" ||
            ((try? selected.appendingPathComponent("hub").resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true) {
            return selected.appendingPathComponent("hub", isDirectory: true)
        }
        return selected
    }
    static func wanted(_ path: String, repository: String, allFiles: Bool) -> Bool {
        if allFiles { return true }
        let flux = ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.2-dev", "mlx-community/FLUX.2-klein-9B"]
        if flux.contains(repository) {
            return path == "model_index.json" || path.hasPrefix("README") || path.hasPrefix("LICENSE") ||
                ["transformer/", "vae/", "text_encoder/", "text_encoder_2/", "tokenizer/", "tokenizer_2/", "scheduler/", "processor/"].contains(where: path.hasPrefix)
        }
        return ["json", "safetensors", "model", "txt", "jinja", "tiktoken", "bin", "mlmodel", "gguf", "md"].contains((path as NSString).pathExtension.lowercased())
    }
    private func client(token: String, cache: HubCache? = nil) -> HubClient {
        HubClient(session: session, host: host, tokenProvider: token.isEmpty ? .environment : .fixed(token: token), cache: cache)
    }
    private static func validHash(_ value: String?, lfs: Bool) -> Bool {
        guard let value, value.count == (lfs ? 64 : 40) else { return false }
        return value.allSatisfy { $0.isASCII && $0.isHexDigit }
    }
    private func fileMetadata(_ entry: Git.TreeEntry, repository: String, commit: String, token: String) async throws -> HFDownloadFile {
        guard Self.validPath(entry.path), let size = entry.effectiveSize, size >= 0 else {
            throw HFDownloadError.invalid("HF 文件元数据无效：\(entry.path)")
        }
        let lfs = entry.lfs != nil
        // A gated tree may contain a masked LFS hash. Its ordinary Git oid
        // identifies the pointer, not the model bytes, and cannot replace it.
        let treeHash = lfs ? entry.lfs?.oid : entry.oid
        if Self.validHash(treeHash, lfs: lfs), let treeHash {
            return .init(path: entry.path, size: Int64(size), blob: treeHash.lowercased(), lfs: lfs)
        }
        var request = URLRequest(url: host.appendingPathComponent(repository).appendingPathComponent("resolve")
            .appendingPathComponent(commit).appendingPathComponent(entry.path))
        request.httpMethod = "HEAD"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let provider: TokenProvider = token.isEmpty ? .environment : .fixed(token: token)
        if let resolved = try await provider.getToken() { request.setValue("Bearer \(resolved)", forHTTPHeaderField: "Authorization") }
        // Keep the linked hash/size on the original response; do not request
        // the weight or follow its potentially signed CDN URL.
        let (_, response) = try await session.data(for: request, delegate: HFMetadataRedirectDelegate())
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        if [401, 403].contains(http.statusCode) { throw HFDownloadError.accessRequired(repository) }
        guard (200..<400).contains(http.statusCode) else { throw HFDownloadError.http(http.statusCode) }
        let hash = (http.value(forHTTPHeaderField: "X-Linked-Etag") ?? http.value(forHTTPHeaderField: "ETag"))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "W/", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        let length = http.value(forHTTPHeaderField: "X-Linked-Size") ?? http.value(forHTTPHeaderField: "Content-Length")
        guard Self.validHash(hash, lfs: lfs), let hash, let length, Int64(length) == Int64(size),
              http.value(forHTTPHeaderField: "X-Repo-Commit") == commit else {
            throw HFDownloadError.invalid("HF 文件元数据无效：\(entry.path)")
        }
        return .init(path: entry.path, size: Int64(size), blob: hash.lowercased(), lfs: lfs)
    }
    func prepare(repository: String, revision: String, token: String, allFiles: Bool) async throws -> HFDownloadPlan {
        let id = try Self.repository(repository)
        guard Self.validPath(revision) else { throw HFDownloadError.invalid("Revision 不能为空或包含路径跳转。") }
        let client = client(token: token)
        let model = try await client.getModel(id, revision: revision)
        guard let commit = model.sha, [40, 64].contains(commit.count), commit.allSatisfy(\.isHexDigit) else {
            throw HFDownloadError.invalid("HF 没有返回有效的模型 commit。")
        }
        let pages = try await client.listAllTree(in: id, revision: commit, recursive: true)
        var files: [HFDownloadFile] = []
        for try await page in pages {
            for entry in page.items where entry.type == .file && Self.wanted(entry.path, repository: repository, allFiles: allFiles) {
                files.append(try await fileMetadata(entry, repository: repository, commit: commit, token: token))
            }
        }
        guard !files.isEmpty else { throw HFDownloadError.invalid("此仓库没有匹配的模型文件。") }
        // Publish shard indexes first so incomplete multi-shard snapshots remain detectable.
        files.sort { a, b in
            let ai = a.path.hasSuffix(".safetensors.index.json"), bi = b.path.hasSuffix(".safetensors.index.json")
            return ai != bi ? ai : a.path < b.path
        }
        return .init(repository: repository, revision: revision, commit: commit, files: files)
    }

    func download(_ plan: HFDownloadPlan, cache configuration: SpeechCacheAccess.Configuration, token: String,
                  progress: @escaping @MainActor @Sendable (Int64, Int64, String, Int, Int) -> Void) async throws -> URL {
        let root = try configuration.resolve()
        let access = root.startAccessingSecurityScopedResource()
        defer { if access { root.stopAccessingSecurityScopedResource() } }
        let cache = HubCache(cacheDirectory: Self.cacheRoot(root))
        let id = try Self.repository(plan.repository)
        try FileManager.default.createDirectory(at: cache.cacheDirectory, withIntermediateDirectories: true)
        var completed: Int64 = 0
        for (index, file) in plan.files.enumerated() {
            try Task.checkCancellation()
            if let cached = cache.cachedFilePath(repo: id, kind: .model, revision: plan.commit, filename: file.path),
               (try? cached.resourceValues(forKeys: [.fileSizeKey]).fileSize) == Int(file.size) {
                completed += file.size
                await progress(completed, plan.bytes, file.path, index + 1, plan.files.count)
                continue
            }
            let blob = try cache.blobPath(repo: id, kind: .model, etag: file.blob)
            let partial = try cache.incompleteBlobPath(repo: id, kind: .model, etag: file.blob)
            let lock = try HFDownloadFileLock(cache.lockPath(for: blob))
            try await lock.acquire()
            defer { lock.release() }
            if (try? blob.resourceValues(forKeys: [.fileSizeKey]).fileSize) != Int(file.size) {
                let base = completed
                var request = URLRequest(url: host.appendingPathComponent(plan.repository).appendingPathComponent("resolve")
                    .appendingPathComponent(plan.commit).appendingPathComponent(file.path))
                request.cachePolicy = .reloadIgnoringLocalCacheData
                request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
                let provider: TokenProvider = token.isEmpty ? .environment : .fixed(token: token)
                if let resolved = try await provider.getToken() { request.setValue("Bearer \(resolved)", forHTTPHeaderField: "Authorization") }
                try await HFStreamingDownload.transfer(request: request, file: file, partial: partial, session: session) { bytes in
                    Task { @MainActor in progress(base + bytes, plan.bytes, file.path, index, plan.files.count) }
                }
                // Only publish the partial verified by this transfer. A stale partial
                // must never replace an already complete blob shared with another client.
                guard Darwin.rename(partial.path, blob.path) == 0 else { throw POSIXError(.EIO) }
            }
            let snapshot = try cache.snapshotPath(repo: id, kind: .model, commitHash: plan.commit)
            let target = snapshot.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let temporary = target.deletingLastPathComponent().appendingPathComponent(".hf-link-" + UUID().uuidString)
            let relative = String(repeating: "../", count: file.path.split(separator: "/").count + 1) + "blobs/" + file.blob
            try FileManager.default.createSymbolicLink(atPath: temporary.path, withDestinationPath: relative)
            guard Darwin.rename(temporary.path, target.path) == 0 else {
                try? FileManager.default.removeItem(at: temporary); throw POSIXError(.EIO)
            }
            completed += file.size
            await progress(completed, plan.bytes, file.path, index + 1, plan.files.count)
        }
        try Task.checkCancellation()
        try cache.updateRef(repo: id, kind: .model, ref: plan.revision, commit: plan.commit)
        return try cache.snapshotPath(repo: id, kind: .model, commitHash: plan.commit)
    }
}

private final class HFMetadataRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

private final class HFDownloadFileLock: @unchecked Sendable {
    private let fd: Int32
    init(_ path: URL) throws {
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        fd = Darwin.open(path.path, O_CREAT | O_RDWR, 0o600)
        guard fd >= 0 else { throw POSIXError(.EACCES) }
    }
    func acquire() async throws {
        while flock(fd, LOCK_EX | LOCK_NB) != 0 {
            guard errno == EWOULDBLOCK || errno == EAGAIN else { throw POSIXError(.EIO) }
            try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(100))
        }
    }
    func release() { _ = flock(fd, LOCK_UN) }
    deinit { Darwin.close(fd) }
}
