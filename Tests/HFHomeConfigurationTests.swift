import Foundation

@main
struct HFHomeConfigurationTests {
    static func main() throws {
        let name = "Synomancer.HFHomeTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let initial = SpeechCacheAccess.configuration(in: defaults)
        precondition(initial.path == "\(NSHomeDirectory())/.cache/huggingface/hub" && initial.bookmark == nil)
        let initialURL = try initial.resolve()
        precondition(initialURL.path == initial.path)
        print("PASS: default HF_HOME used without a stored preference")

        let bookmark = Data([1, 2, 3])
        defaults.set("/tmp/legacy/hub", forKey: SpeechCacheAccess.pathKey)
        defaults.set(bookmark, forKey: SpeechCacheAccess.bookmarkKey)
        let migrated = SpeechCacheAccess.configuration(in: defaults)
        precondition(migrated.path == "/tmp/legacy/hub" && migrated.bookmark == bookmark)
        print("PASS: existing shared path and authorization migrate")

        defaults.set(" /tmp/new-cache/hub/ ", forKey: SpeechCacheAccess.homeKey)
        let changed = SpeechCacheAccess.configuration(in: defaults)
        precondition(changed.path == "/tmp/new-cache/hub" && changed.bookmark == nil)
        precondition(changed != migrated)
        let changedURL = try changed.resolve()
        precondition(changedURL.path == changed.path)
        print("PASS: changing HF_HOME changes model identity and rejects the old directory grant")

        defaults.set("/tmp/legacy/./hub", forKey: SpeechCacheAccess.homeKey)
        precondition(SpeechCacheAccess.configuration(in: defaults).bookmark == bookmark)
        defaults.set("  ", forKey: SpeechCacheAccess.homeKey)
        let empty = SpeechCacheAccess.configuration(in: defaults)
        precondition(empty.path == SpeechCacheAccess.defaultPath && empty.bookmark == nil)
        print("PASS: equivalent paths preserve the grant; empty configured paths use the default")
    }
}
