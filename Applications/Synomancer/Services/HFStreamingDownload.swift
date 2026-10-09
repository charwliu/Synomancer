import Foundation
import CryptoKit

/// Serial URLSession delegate writes directly to a stable cache partial, so
/// cancellation and network failures preserve received bytes across launches.
final class HFStreamingDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var task: URLSessionDataTask?
    private var ownedSession: URLSession?
    private var cancelled = false
    private var handle: FileHandle?
    private let file: HFDownloadFile
    private let partial: URL
    private var received: Int64
    private var sha256 = SHA256()
    private var gitSHA = Insecure.SHA1()
    private var accepted = false
    private var failure: Error?
    private var lastProgress = Date.distantPast
    private let progress: @Sendable (Int64) -> Void

    private init(file: HFDownloadFile, partial: URL, progress: @escaping @Sendable (Int64) -> Void) throws {
        self.file = file; self.partial = partial; self.progress = progress
        try FileManager.default.createDirectory(at: partial.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: partial.path) { _ = FileManager.default.createFile(atPath: partial.path, contents: nil) }
        received = Int64((try partial.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        super.init()
        if received > file.size {
            let h = try FileHandle(forWritingTo: partial); try h.truncate(atOffset: 0); try h.close(); received = 0
        }
        resetHash()
        let reader = try FileHandle(forReadingFrom: partial)
        defer { try? reader.close() }
        while let chunk = try reader.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            try Task.checkCancellation(); updateHash(chunk)
        }
        handle = try FileHandle(forWritingTo: partial)
        try handle?.seekToEnd()
    }
    private func resetHash() {
        sha256 = SHA256(); gitSHA = Insecure.SHA1()
        if !file.lfs { gitSHA.update(data: Data("blob \(file.size)\0".utf8)) }
    }
    private func updateHash(_ data: Data) {
        if file.lfs { sha256.update(data: data) } else { gitSHA.update(data: data) }
    }
    private func verify() throws {
        guard received == file.size else { throw HFDownloadError.integrity(file.path) }
        let digest = file.lfs ? sha256.finalize().map { String(format: "%02x", $0) }.joined()
            : gitSHA.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == file.blob.lowercased() else {
            try? handle?.close(); handle = nil
            try? FileManager.default.removeItem(at: partial)
            throw HFDownloadError.integrity(file.path)
        }
        try handle?.synchronize()
    }
    static func transfer(request: URLRequest, file: HFDownloadFile, partial: URL, session: URLSession,
                         progress: @escaping @Sendable (Int64) -> Void) async throws {
        let delegate = try HFStreamingDownload(file: file, partial: partial, progress: progress)
        if delegate.received == file.size {
            do { try delegate.verify(); try delegate.handle?.close(); delegate.handle = nil; progress(file.size); return }
            catch {
                // A complete-looking but invalid partial must restart from zero.
                if file.size == 0 || FileManager.default.fileExists(atPath: partial.path) { throw error }
                return try await transfer(request: request, file: file, partial: partial, session: session, progress: progress)
            }
        }
        var request = request
        if delegate.received > 0 { request.setValue("bytes=\(delegate.received)-", forHTTPHeaderField: "Range") }
        let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
        let configuration = session.configuration
        configuration.timeoutIntervalForRequest = 60; configuration.timeoutIntervalForResource = 86400
        let owned = URLSession(configuration: configuration, delegate: delegate, delegateQueue: queue)
        delegate.ownedSession = owned
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.lock.lock()
                delegate.continuation = continuation
                let task = owned.dataTask(with: request); delegate.task = task
                let cancelled = delegate.cancelled
                delegate.lock.unlock()
                if cancelled { task.cancel() } else { task.resume() }
            }
        } onCancel: { delegate.cancel() }
    }
    private func cancel() {
        lock.lock(); cancelled = true; let task = task; lock.unlock()
        task?.cancel()
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        do {
            guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
            if http.statusCode == 200 {
                if received > 0 {
                    try handle?.truncate(atOffset: 0); try handle?.seek(toOffset: 0)
                    received = 0; resetHash()
                }
            } else if http.statusCode == 206 {
                guard let range = http.value(forHTTPHeaderField: "Content-Range"),
                      range.hasPrefix("bytes \(received)-"), range.hasSuffix("/\(file.size)") else {
                    throw HFDownloadError.invalid("服务器返回了不匹配的续传范围：\(file.path)")
                }
            } else { throw HFDownloadError.http(http.statusCode) }
            accepted = true; progress(received); completionHandler(.allow)
        } catch { failure = error; completionHandler(.cancel) }
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard accepted, failure == nil else { return }
        do {
            guard received + Int64(data.count) <= file.size else { throw HFDownloadError.integrity(file.path) }
            try handle?.write(contentsOf: data); updateHash(data); received += Int64(data.count)
            if Date().timeIntervalSince(lastProgress) >= 0.2 {
                lastProgress = Date(); progress(received)
            }
        } catch { failure = error; dataTask.cancel() }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        var result: Result<Void, Error>
        do {
            if let error = failure ?? error { throw error }
            guard accepted else { throw URLError(.badServerResponse) }
            try verify(); progress(received); result = .success(())
        } catch { result = .failure(error) }
        try? handle?.close(); handle = nil
        lock.lock(); let continuation = continuation; self.continuation = nil; lock.unlock()
        continuation?.resume(with: result)
        session.finishTasksAndInvalidate(); ownedSession = nil
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        var redirected = request
        if task.originalRequest?.url?.scheme == "https" && request.url?.scheme != "https" {
            failure = URLError(.secureConnectionFailed); completionHandler(nil); return
        }
        if request.url?.host != task.originalRequest?.url?.host { redirected.setValue(nil, forHTTPHeaderField: "Authorization") }
        completionHandler(redirected)
    }
    deinit { try? handle?.close() }
}
