import Foundation

/// Bounded mono PCM shared by the audio callback and async decoder.
/// All mutable storage is protected by `lock`; snapshots own their sample arrays.
public final class StreamingSpeechBuffer: @unchecked Sendable {
    public struct Snapshot: Sendable {
        public let samples: [Float]
        public let endFrame: Int
        public let sampleRate: Int

        public var hasSpeech: Bool {
            !samples.isEmpty && samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count) > 0.000001
        }
        public var hasSilentTail: Bool {
            let count = Int(Double(sampleRate) * 0.8)
            guard samples.count >= count else { return false }
            let tail = samples.suffix(count)
            return tail.reduce(Float(0)) { $0 + $1 * $1 } / Float(count) <= 0.000001
        }
    }

    private let lock = NSLock()
    private var samples: [Float] = []
    private var endFrame = 0
    private let sampleRate: Int
    private let capacity: Int

    public init(sampleRate: Int, maximumSeconds: Int = 30) {
        precondition(sampleRate > 0 && maximumSeconds > 0)
        self.sampleRate = sampleRate
        capacity = sampleRate * maximumSeconds
    }

    public func append(_ values: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        samples.append(contentsOf: values)
        endFrame += values.count
        if samples.count > capacity { samples.removeFirst(samples.count - capacity) }
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(samples: samples, endFrame: endFrame, sampleRate: sampleRate)
    }

    /// Preserve audio that arrived while the snapshot was being decoded.
    public func discard(through frame: Int) {
        lock.lock()
        defer { lock.unlock() }
        let startFrame = endFrame - samples.count
        let count = max(0, min(samples.count, frame - startFrame))
        samples.removeFirst(count)
    }
}

/// Revises the current utterance while preserving previously committed segments.
public struct StreamingSpeechTranscript: Sendable {
    public private(set) var text = ""
    private var committed = ""

    public init() {}

    @discardableResult
    public mutating func update(partial: String, commit: Bool) -> String {
        let partial = partial.trimmingCharacters(in: .whitespacesAndNewlines)
        if !partial.isEmpty {
            text = [committed, partial].filter { !$0.isEmpty }.joined(separator: " ")
        }
        if commit { committed = text }
        return text
    }
}
