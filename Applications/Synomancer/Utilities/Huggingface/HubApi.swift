//
//  HubApi.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//



import Foundation

public class HubApi {
    nonisolated(unsafe) public static let shared = HubApi()
    
    var downloadBase: URL
    var hfToken: String?
    var endpoint: String
    var useBackgroundSession: Bool

    private var currentTask: Task<URL, Error>?

    public typealias RepoType = Hub.RepoType
    public typealias Repo = Hub.Repo

    public init(
        downloadBase: URL? = nil, hfToken: String? = nil,
        endpoint: String = "https://huggingface.co", useBackgroundSession: Bool = false
    ) {
        self.hfToken = hfToken
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        self.endpoint = endpoint
        self.useBackgroundSession = useBackgroundSession
    }

    
}

/// File retrieval
extension HubApi {
    /// Model data for parsed filenames
    public struct Sibling: Codable {
        let rfilename: String
    }

    public struct SiblingsResponse: Codable {
        let siblings: [Sibling]
    }

    /// Throws error if the response code is not 20X
    public func httpGet(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        if let hfToken = hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw Hub.HubClientError.unexpectedError
        }

        switch response.statusCode {
        case 200 ..< 300: break
        case 400 ..< 500: throw Hub.HubClientError.authorizationRequired
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return (data, response)
    }

    public func getFilenames(from repo: Repo, matching globs: [String] = []) async throws
        -> [String]
    {
        // Read repo info and only parse "siblings"
        let url = URL(string: "\(endpoint)/api/\(repo.type)/\(repo.id)")!
        let (data, _) = try await httpGet(for: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard globs.count > 0 else { return filenames }

        var selected: Set<String> = []
        for glob in globs {
            selected = selected.union(filenames.matching(glob: glob))
        }
        return Array(selected)
    }

    public func getFilenames(from repoId: String, matching globs: [String] = []) async throws
        -> [String]
    {
        return try await getFilenames(from: Repo(id: repoId), matching: globs)
    }

    public func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        return try await getFilenames(from: repo, matching: [glob])
    }

    public func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        return try await getFilenames(from: Repo(id: repoId), matching: [glob])
    }
}

/// Configuration loading helpers
extension HubApi {
    /// Assumes the file has already been downloaded.
    /// `filename` is relative to the download base.
    public func configuration(from filename: String, in repo: Repo) throws -> Config {
        let fileURL = localRepoLocation(repo).appending(path: filename)
        return try configuration(fileURL: fileURL)
    }

    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    public func configuration(fileURL: URL) throws -> Config {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [String: Sendable] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Whoami
extension HubApi {
    public func whoami() async throws -> Config {
        guard hfToken != nil else { throw Hub.HubClientError.authorizationRequired }

        let url = URL(string: "\(endpoint)/api/whoami-v2")!
        let (data, _) = try await httpGet(for: url)

        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [String: Sendable] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Snaphsot download
extension HubApi {
    public func localRepoLocation(_ repo: Repo) -> URL {
        downloadBase.appending(component: repo.type.rawValue).appending(component: repo.id)
    }

    public struct HubFileDownloader {
        let repo: Repo
        let repoDestination: URL
        let relativeFilename: String
        let hfToken: String?
        let endpoint: String?
        let backgroundSession: Bool

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: endpoint ?? "https://huggingface.co")!
            if repo.type != .models {
                url = url.appending(component: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve/main")  // TODO: revisions
            url = url.appending(path: relativeFilename)
            return url
        }

        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }

        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }

        func prepareDestination() throws {
            let directoryURL = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        }

        // We'll probably need to support Combine as well to play well with Swift UI
        // (See for example PipelineLoader in swift-coreml-diffusers)
        @discardableResult
        func download(progressHandler: @escaping (Double) throws -> Void) async throws -> URL {
            guard !downloaded else { return destination }

            try prepareDestination()
            let downloader = Downloader(
                from: source, to: destination, using: hfToken, inBackground: backgroundSession)
            try await downloader.download(from: source, with: hfToken)
            
            let downloadSubscriber = downloader.downloadState.sink { state in
                if case .downloading(let progress) = state {
                    do {
                        try progressHandler(progress)
                    } catch {
                        downloader.cancel()
                    }
                }
            }
            return try await withTaskCancellationHandler {
                try withExtendedLifetime(downloadSubscriber) {
                    try downloader.waitUntilDone()
                }
            } onCancel: {
                downloader.cancel()
            }
        }
    }
    
    @discardableResult
    public func snapshot(
        from repo: sending Repo, matching globs: sending [String] = [],
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
//        self.currentTask = Task {
            let filenames = try await getFilenames(from: repo, matching: globs)
            let progress = Progress(totalUnitCount: Int64(filenames.count))
            let repoDestination = localRepoLocation(repo)
            for filename in filenames {
                if Task.isCancelled { throw CancellationError() }
                let fileProgress = Progress(
                    totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
                let downloader = HubFileDownloader(
                    repo: repo,
                    repoDestination: repoDestination,
                    relativeFilename: filename,
                    hfToken: hfToken,
                    endpoint: endpoint,
                    backgroundSession: useBackgroundSession
                )

                try await downloader.download { fractionDownloaded in
                    do {
                        if Task.isCancelled { throw CancellationError() }
                        fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                        progressHandler(progress)
                    } catch {
                        throw error
                    }
                }
                fileProgress.completedUnitCount = 100
            }
            progressHandler(progress)
            return repoDestination
//        }
//       return try await currentTask!.value
    }


    @discardableResult
    public func snapshot(
        from repoId: String, matching globs: [String] = [],
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await snapshot(
            from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    public func snapshot(
        from repo: Repo, matching glob: String,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await snapshot(from: repo, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    public func snapshot(
        from repoId: String, matching glob: String,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await snapshot(
            from: Repo(id: repoId), matching: [glob], progressHandler: progressHandler)
    }
}

/// Stateless wrappers that use `HubApi` instances
extension Hub {
    public static func getFilenames(from repo: Hub.Repo, matching globs: [String] = []) async throws
        -> [String]
    {
        return try await HubApi.shared.getFilenames(from: repo, matching: globs)
    }

    public static func getFilenames(from repoId: String, matching globs: [String] = []) async throws
        -> [String]
    {
        return try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: globs)
    }

    public static func getFilenames(from repo: Repo, matching glob: String) async throws -> [String]
    {
        return try await HubApi.shared.getFilenames(from: repo, matching: glob)
    }

    public static func getFilenames(from repoId: String, matching glob: String) async throws
        -> [String]
    {
        return try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: glob)
    }

    public static func snapshot(
        from repo: Repo, matching globs: [String] = [],
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await HubApi.shared.snapshot(
            from: repo, matching: globs, progressHandler: progressHandler)
    }

    public static func snapshot(
        from repoId: String, matching globs: [String] = [],
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await HubApi.shared.snapshot(
            from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    public static func snapshot(
        from repo: Repo, matching glob: String,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await HubApi.shared.snapshot(
            from: repo, matching: glob, progressHandler: progressHandler)
    }

    public static func snapshot(
        from repoId: String, matching glob: String,
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> URL {
        return try await HubApi.shared.snapshot(
            from: Repo(id: repoId), matching: glob, progressHandler: progressHandler)
    }

    public static func whoami(token: String) async throws -> Config {
        return try await HubApi(hfToken: token).whoami()
    }
}

extension [String] {
    public func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

extension HubApi {
    public func cancelCurrentDownload() {
        currentTask?.cancel()
    }
}


extension HubApi {
    func getAverageTemperature() async {
        let fetchTask = Task { () -> Double in
            let url = URL(string: "https://hws.dev/readings.json")!
            let (data, _) = try await URLSession.shared.data(from: url)
            let readings = try JSONDecoder().decode([Double].self, from: data)
            let sum = readings.reduce(0, +)
            return sum / Double(readings.count)
        }
        
        do {
            let result = try await fetchTask.value
            print("Average temperature: \(result)")
        } catch {
            print("Failed to get data.")
        }
    }
}


