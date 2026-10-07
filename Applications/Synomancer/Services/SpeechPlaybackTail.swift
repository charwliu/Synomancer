import Foundation

/// Holds just the end of an utterance so EOS can fade real PCM to silence.
/// Intermediate chunk boundaries are untouched, including empty EOS sentinels.
struct SpeechPlaybackTail {
    private let fadeSamples: Int
    private let silenceSamples: Int
    private var pending: [Float] = []

    init(sampleRate: Int = 24000) {
        precondition(sampleRate > 0)
        fadeSamples = max(1, sampleRate / 50) // 20 ms
        silenceSamples = sampleRate / 20 // 50 ms for the output device to drain
    }

    mutating func append(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }
        let combined = pending + samples
        let emitted = max(0, combined.count - fadeSamples)
        pending = Array(combined.dropFirst(emitted))
        return Array(combined.prefix(emitted))
    }

    mutating func finish() -> [Float] {
        guard !pending.isEmpty else { return [] }
        var result = pending
        pending.removeAll(keepingCapacity: true)
        let count = result.count
        for index in result.indices {
            let phase = count > 1 ? Double(index) / Double(count - 1) : 1
            let gain = Float(0.5 * (1 + cos(.pi * phase)))
            result[index] *= gain
        }
        // Guarantee an exact zero endpoint, independent of cosine rounding.
        result[count - 1] = 0
        result.append(contentsOf: repeatElement(Float(0), count: silenceSamples))
        return result
    }
}
