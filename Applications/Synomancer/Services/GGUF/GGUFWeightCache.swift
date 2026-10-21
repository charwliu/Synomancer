import Foundation
import MLX

/// Serial-owner LRU of evaluated packed arrays. File mappings are not counted as
/// resident allocations; their clean pages remain reclaimable by macOS.
final class GGUFWeightCache {
    struct Key: Hashable { let tensor: String; let index: Int }
    private final class Entry {
        let key: Key
        let data: MLXArray
        let byteCount: Int
        weak var previous: Entry?
        var next: Entry?
        init(_ key: Key, _ data: MLXArray, _ byteCount: Int) { self.key = key; self.data = data; self.byteCount = byteCount }
    }
    let budget: Int
    let maximumEntries: Int
    private(set) var bytes = 0
    private(set) var hits = 0
    private(set) var misses = 0
    private var entries: [Key: Entry] = [:]
    private var first: Entry?
    private var last: Entry?

    init(budget: Int, maximumEntries: Int = .max) {
        self.budget = max(0, budget); self.maximumEntries = max(0, maximumEntries)
    }
    private func unlink(_ entry: Entry) {
        if let previous = entry.previous { previous.next = entry.next } else { first = entry.next }
        if let next = entry.next { next.previous = entry.previous } else { last = entry.previous }
        entry.previous = nil; entry.next = nil
    }
    private func append(_ entry: Entry) {
        entry.previous = last
        last?.next = entry
        last = entry
        if first == nil { first = entry }
    }
    func read(key: Key, byteCount: Int, load: () -> MLXArray) -> MLXArray {
        if let entry = entries[key] {
            hits += 1; unlink(entry); append(entry)
            return entry.data
        }
        misses += 1
        // Evict before allocating the replacement. Oversized entries bypass
        // the cache instead of flushing useful smaller entries.
        if byteCount <= budget && maximumEntries > 0 {
            while bytes > budget - byteCount || entries.count >= maximumEntries, let oldest = first {
                unlink(oldest); entries.removeValue(forKey: oldest.key); bytes -= oldest.byteCount
            }
        }
        let data = load()
        eval(data)
        precondition(data.nbytes == byteCount)
        if byteCount <= budget && maximumEntries > 0 {
            let entry = Entry(key, data, byteCount)
            entries[key] = entry; append(entry); bytes += byteCount
        }
        return data
    }
    func removeAll() {
        // Break the forward chain iteratively, even for millions of PLE rows.
        while let entry = first { unlink(entry) }
        entries.removeAll(); bytes = 0
    }
    deinit { removeAll() }
}

enum GGUFMemoryPolicy {
    static let expertBudgetKey = "localQwenGGUFExpertCacheGiB"
    static let defaultExpertGiB = 16
    static var maximumExpertGiB: Int {
        max(1, min(64, Int(ProcessInfo.processInfo.physicalMemory / (4 * 1024 * 1024 * 1024))))
    }
    static func expertBudget(preferences: UserDefaults = .standard) -> Int {
        let requested = preferences.object(forKey: expertBudgetKey) as? Int ?? defaultExpertGiB
        return max(0, min(maximumExpertGiB, requested)) * 1024 * 1024 * 1024
    }
}
