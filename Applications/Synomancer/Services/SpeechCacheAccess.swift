import Foundation
#if os(macOS)
import AppKit
#endif

/// Chat, images, ASR and TTS share one HF_HOME and directory grant.
/// Model downloads upgrade the grant to read/write through the native picker.
enum SpeechCacheAccess {
    static let homeKey = "HF_HOME"
    static let defaultPath = "/path/to/workspace/.cache/huggingface/hub"
    static let bookmarkKey = "localSpeechCacheBookmark"
    // Retained for migration and to associate an existing bookmark with its path.
    static let pathKey = "localSpeechCachePath"
    static let writablePathKey = "hfCacheWritablePath"

    struct Configuration: Equatable, Sendable {
        let path: String
        let bookmark: Data?

        func resolve() throws -> URL {
            if let bookmark { return try SpeechCacheAccess.resolveBookmark(bookmark) }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
    }

    static func configuration(in defaults: UserDefaults = .standard) -> Configuration {
        let saved = defaults.string(forKey: homeKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let legacy = defaults.string(forKey: pathKey)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let selected: String
        if let saved {
            selected = saved.isEmpty ? defaultPath : saved
        } else if let legacy, !legacy.isEmpty {
            selected = legacy
        } else {
            selected = defaultPath
        }
        let path = normalized(selected)
        let authorizedPath = legacy.map(normalized)
        let bookmark = authorizedPath == path ? defaults.data(forKey: bookmarkKey) : nil
        return Configuration(path: path, bookmark: bookmark)
    }

    /// A typed default path does not grant sandbox access. Ask through the
    /// native picker at first use, before reporting any missing model files.
    @MainActor
    static func authorizedConfiguration(in defaults: UserDefaults = .standard) async throws -> Configuration {
        try Task.checkCancellation()
        let config = configuration(in: defaults)
        do {
            let root = try config.resolve()
            let accessing = root.startAccessingSecurityScopedResource()
            defer { if accessing { root.stopAccessingSecurityScopedResource() } }
            _ = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            return config
        } catch {
#if os(macOS)
            let panel = NSOpenPanel()
            panel.title = "Authorize HF_HOME"
            panel.message = "Allow Synomancer to read the shared chat/ASR/TTS model cache. Select the HF_HOME folder, including its snapshots and blobs."
            panel.prompt = "Use HF_HOME"
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.allowsMultipleSelection = false
            panel.showsHiddenFiles = true
            panel.directoryURL = URL(fileURLWithPath: config.path, isDirectory: true)
            let selected: URL? = await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    panel.begin { response in
                        continuation.resume(returning: response == .OK ? panel.url : nil)
                    }
                }
            } onCancel: {
                Task { @MainActor in panel.cancel(nil) }
            }
            try Task.checkCancellation()
            guard let selected else { throw CancellationError() }
            let accessing = selected.startAccessingSecurityScopedResource()
            defer { if accessing { selected.stopAccessingSecurityScopedResource() } }
            try save(selected, in: defaults)
            return configuration(in: defaults)
#else
            throw AccessError.unavailable(config.path)
#endif
        }
    }

    enum AccessError: LocalizedError {
        case unavailable(String)
        var errorDescription: String? {
            switch self {
            case .unavailable(let path): return "HF_HOME is not readable: \(path). Choose its folder in Settings to authorize access."
            }
        }
    }

    static func normalized(_ path: String) -> String {
        let expanded = (path.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
    }

    static func resolveBookmark(_ data: Data) throws -> URL {
        var stale = false
#if os(macOS)
        let options: URL.BookmarkResolutionOptions = [.withSecurityScope]
#else
        let options: URL.BookmarkResolutionOptions = []
#endif
        return try URL(resolvingBookmarkData: data, options: options, bookmarkDataIsStale: &stale)
    }

    static func save(_ url: URL, in defaults: UserDefaults = .standard, writable: Bool = false) throws {
#if os(macOS)
        let options: URL.BookmarkCreationOptions = writable ? [.withSecurityScope] : [.withSecurityScope, .securityScopeAllowOnlyReadAccess]
#else
        let options: URL.BookmarkCreationOptions = [.minimalBookmark]
#endif
        let data = try url.bookmarkData(options: options, includingResourceValuesForKeys: nil, relativeTo: nil)
        defaults.set(data, forKey: bookmarkKey)
        defaults.set(normalized(url.path), forKey: pathKey)
        defaults.set(normalized(url.path), forKey: homeKey)
        if writable { defaults.set(normalized(url.path), forKey: writablePathKey) }
        else { defaults.removeObject(forKey: writablePathKey) }
    }

    @MainActor static func authorizedWritableConfiguration(in defaults: UserDefaults = .standard) async throws -> Configuration {
        try Task.checkCancellation()
        let current = configuration(in: defaults)
        if defaults.string(forKey: writablePathKey) == current.path, current.bookmark != nil {
            return current
        }
#if os(macOS)
        let panel = NSOpenPanel()
        panel.title = "允许写入 HF_HOME"
        panel.message = "选择 HF_HOME 文件夹，允许 Synomancer 保存下载的模型。现有只读授权不能用于下载。"
        panel.prompt = "允许下载到此文件夹"
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false; panel.canCreateDirectories = true; panel.showsHiddenFiles = true
        panel.directoryURL = URL(fileURLWithPath: current.path, isDirectory: true)
        let url: URL? = await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                panel.begin { response in continuation.resume(returning: response == .OK ? panel.url : nil) }
            }
        } onCancel: { Task { @MainActor in panel.cancel(nil) } }
        try Task.checkCancellation()
        guard let url else { throw CancellationError() }
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        try save(url, in: defaults, writable: true)
        return configuration(in: defaults)
#else
        throw AccessError.unavailable(current.path)
#endif
    }
}
