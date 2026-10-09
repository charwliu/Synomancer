import Foundation
import HuggingFace
import CryptoKit

@main struct HFModelDownloadTests {
    static func main() async throws {
        setbuf(stdout, nil)
        if CommandLine.arguments[1] == "--real-flux-metadata" {
            let plan = try await HFModelDownloadService().prepare(repository: "black-forest-labs/FLUX.2-dev", revision: "main", token: "", allFiles: false)
            let shard = plan.files.first { $0.path == "text_encoder/model-00001-of-00010.safetensors" }!
            precondition(shard.size == 4_883_550_696 && shard.lfs && shard.blob.count == 64)
            precondition(plan.files.filter { $0.path.hasSuffix(".safetensors") }.allSatisfy { $0.lfs && $0.blob.count == 64 })
            print("PASS: real gated FLUX.2-dev preflight (\(plan.files.count) files, \(plan.bytes) bytes, first encoder shard \(shard.size) bytes); no weights downloaded")
            return
        }
        if CommandLine.arguments[1] == "--real" {
            let root = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
            let service = HFModelDownloadService()
            let complete = try await service.prepare(repository: "hf-internal-testing/tiny-random-gpt2", revision: "main", token: "", allFiles: false)
            let files = complete.files.filter { ["config.json", "tokenizer.json"].contains($0.path) }
            precondition(files.count == 2 && files.reduce(0) { $0 + $1.size } < 100_000)
            let plan = HFDownloadPlan(repository: complete.repository, revision: complete.revision, commit: complete.commit, files: files)
            let configuration = SpeechCacheAccess.Configuration(path: root.path, bookmark: nil)
            let snapshot = try await service.download(plan, cache: configuration, token: "") { _, _, _, _, _ in }
            for file in files {
                let data = try Data(contentsOf: snapshot.appendingPathComponent(file.path))
                _ = try JSONSerialization.jsonObject(with: data)
                precondition(data.count == file.size)
            }
            let reused = try await service.download(plan, cache: configuration, token: "") { _, _, _, _, _ in }
            precondition(reused == snapshot)
            print("PASS: real HF metadata, HTTPS redirect, Git hash and cache reuse (\(plan.bytes) bytes, \(plan.commit))")
            return
        }
        let host = URL(string: CommandLine.arguments[1])!
        let root = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let service = HFModelDownloadService(host: host)
        let config = SpeechCacheAccess.Configuration(path: root.path, bookmark: nil)
        do {
            _ = try await service.prepare(repository: "fixture/model", revision: "main", token: "invalid-token", allFiles: true)
            preconditionFailure("Masked gated metadata without authorization accepted")
        } catch HFDownloadError.accessRequired(let repo) { precondition(repo == "fixture/model") }
        let plan = try await service.prepare(repository: "fixture/model", revision: "main", token: "fixture-key", allFiles: true)
        precondition(plan.files.count == 2 && plan.bytes > 0)
        let missing = try await service.prepare(repository: "fixture/missing-hash", revision: "main", token: "fixture-key", allFiles: true)
        precondition(missing.files.first { $0.lfs }!.blob == plan.files.first { $0.lfs }!.blob)
        for repo in ["fixture/bad-hash", "fixture/bad-size", "fixture/bad-commit"] {
            do {
                _ = try await service.prepare(repository: repo, revision: "main", token: "fixture-key", allFiles: true)
                preconditionFailure("Mismatched HEAD metadata accepted")
            } catch HFDownloadError.invalid { }
        }
        let initialStats = try await URLSession.shared.data(from: host.appendingPathComponent("stats")).0
        let initial = try JSONSerialization.jsonObject(with: initialStats) as! [String: Any]
        precondition(initial["redirect_gets"] as! Int == 0 && initial["weight_gets"] as! Int == 0)
        print("PASS: masked/missing LFS hash resolved by HEAD, auth error and hash/size/commit mismatches checked; no redirect or weight GET")
        do {
            _ = try await service.download(plan, cache: config, token: "fixture-key") { _, _, _, _, _ in }
            preconditionFailure("Interrupted transfer must fail")
        } catch { print("PASS: interrupted HTTP transfer returns an error") }
        let cache = HubCache(cacheDirectory: root)
        let weight = plan.files.first { $0.path == "model.safetensors" }!
        let partial = try cache.incompleteBlobPath(repo: .init(rawValue: plan.repository)!, kind: .model, etag: weight.blob)
        let retained = try partial.resourceValues(forKeys: [.fileSizeKey]).fileSize!
        precondition(retained > 0 && retained < weight.size)
        let snapshot = try await service.download(plan, cache: config, token: "fixture-key") { received, total, _, _, _ in
            precondition(received >= 0 && received <= total)
        }
        let downloaded = try Data(contentsOf: snapshot.appendingPathComponent("model.safetensors"))
        precondition(downloaded.count == Int(weight.size))
        let stats = try await URLSession.shared.data(from: host.appendingPathComponent("stats")).0
        let observed = try JSONSerialization.jsonObject(with: stats) as! [String: Any]
        precondition(observed["range"] as! Int == retained)
        print("PASS: interrupted bytes retained and resumed with exact HTTP Range \(retained)")
        let repeated = try await service.download(plan, cache: config, token: "fixture-key") { _, _, _, _, _ in }
        precondition(repeated == snapshot)
        let after = try await URLSession.shared.data(from: host.appendingPathComponent("stats")).0
        precondition(after == stats, "Complete cached files must not be fetched again")
        print("PASS: completed cache reuse, relative snapshots and refs")
        let ref = try String(contentsOf: root.appendingPathComponent("models--fixture--model/refs/main"), encoding: .utf8)
        precondition(ref == plan.commit)
        let link = try FileManager.default.destinationOfSymbolicLink(atPath: snapshot.appendingPathComponent("config.json").path)
        precondition(!link.hasPrefix("/"))
        // A leftover partial must not overwrite a complete blob when rebuilding links.
        try FileManager.default.removeItem(at: snapshot.appendingPathComponent(weight.path))
        try Data("stale partial".utf8).write(to: partial)
        _ = try await service.download(plan, cache: config, token: "fixture-key") { _, _, _, _, _ in }
        let restored = try Data(contentsOf: snapshot.appendingPathComponent(weight.path))
        precondition(restored == downloaded)
        print("PASS: rebuilding snapshot preserves complete blob despite stale partial")
        for id in ["../model", "user/../../bad", "/bad", "bad/name/extra"] {
            do { _ = try HFModelDownloadService.repository(id); preconditionFailure("Unsafe repo accepted") }
            catch HFDownloadError.invalid { }
        }
        precondition(!HFModelDownloadService.validPath("a/../bad") && !HFModelDownloadService.validPath("/bad"))
        precondition(!HFModelDownloadService.wanted("flux2-dev.safetensors", repository: "black-forest-labs/FLUX.2-dev", allFiles: false))
        precondition(HFModelDownloadService.wanted("transformer/model.safetensors", repository: "black-forest-labs/FLUX.2-dev", allFiles: false))
        // A server that ignores Range must restart rather than append duplicates.
        let payload = Data(repeating: 65, count: 4096)
        let oid = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        let direct = root.appendingPathComponent("ignore-range.incomplete")
        try payload.prefix(1024).write(to: direct)
        try await HFStreamingDownload.transfer(request: URLRequest(url: host.appendingPathComponent("ignore")),
            file: .init(path: "ignore", size: 4096, blob: oid, lfs: true), partial: direct, session: .shared) { _ in }
        let restarted = try Data(contentsOf: direct)
        precondition(restarted == payload)
        print("PASS: Range-ignored 200 restarts safely; checksum and size verified")
        let corruptPath = root.appendingPathComponent("corrupt.incomplete")
        let wrongOID = String(repeating: "0", count: 64)
        do {
            try await HFStreamingDownload.transfer(request: URLRequest(url: host.appendingPathComponent("ignore")),
                file: .init(path: "corrupt", size: 4096, blob: wrongOID, lfs: true), partial: corruptPath, session: .shared) { _ in }
            preconditionFailure("Corrupted file accepted")
        } catch HFDownloadError.integrity { }
        precondition(!FileManager.default.fileExists(atPath: corruptPath.path))
        print("PASS: checksum mismatch rejects and removes invalid partial")
        // Cancel a slow live transfer and ensure its partial remains available.
        let cancelPath = root.appendingPathComponent("cancel.incomplete")
        let cancelled = Task {
            try await HFStreamingDownload.transfer(request: URLRequest(url: host.appendingPathComponent("slow")),
                file: .init(path: "slow", size: 4096, blob: oid, lfs: true), partial: cancelPath, session: .shared) { _ in }
        }
        for _ in 0..<100 {
            if ((try? cancelPath.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 0 { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        cancelled.cancel()
        do { try await cancelled.value; preconditionFailure("Cancelled transfer returned success") }
        catch { }
        let cancelledBytes = try cancelPath.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        precondition(cancelledBytes > 0)
        print("PASS: live cancellation retains partial; path and FLUX filter checks")
    }
}
