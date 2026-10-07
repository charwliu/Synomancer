import Foundation
import MLX
import AudioCommon
import Qwen3TTS

@main
struct TTSCodecContinuityTests {
    static func main() throws {
        setbuf(stdout, nil)
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let selection = LocalTTSCache.resolve(in: root) else { fatalError("Missing local TTS cache") }
        let model = try Qwen3TTSModel.fromLocal(modelDirectory: selection.model, tokenizerDirectory: selection.tokenizer,
            wiredMemoryPolicy: .none, runtimeTalkerBits: 8, runtimePredictorBits: 8)
        // Real codec weights, deterministic synthetic codec IDs: compare identical
        // codes rather than two independently sampled TTS utterances.
        var values: [Int32] = []
        let frames = 96
        for index in 0..<(16 * frames) { values.append(Int32((index * 131 + (index / frames) * 17) % 1024)) }
        let codes = MLXArray(values).reshaped([1, 16, frames])
        let full = model.codecDecoder.executeDecoder(codes).reshaped([-1]).asArray(Float.self)
        for sizes in [[4, 6], [6, 12], [1, 2, 3, 7, 12]] {
            var state = SpeechTokenizerDecoder.StreamingState()
            var actual: [Float] = []
            var offset = 0, index = 0
            while offset < frames {
                let width = sizes.count == 2 ? (index == 0 ? sizes[0] : sizes[1]) : sizes[index % sizes.count]
                let end = min(frames, offset + width)
                let chunk = model.codecDecoder.decodeStreamingChunk(codes: codes[0..., 0..., offset..<end], state: &state)
                    .reshaped([-1]).asArray(Float.self)
                precondition(chunk.allSatisfy(\.isFinite))
                actual.append(contentsOf: chunk)
                offset = end
                index += 1
            }
            precondition(actual.count == full.count, "Duplicate/lost PCM samples: \(actual.count) != \(full.count)")
            var error = 0.0, energy = 0.0, maximum = 0.0
            for (a, b) in zip(actual, full) {
                let difference = Double(a - b)
                error += difference * difference
                energy += Double(b) * Double(b)
                maximum = max(maximum, abs(difference))
            }
            let normalized = sqrt(error / energy)
            print("CODEC sizes=\(sizes) samples=\(actual.count) normalized RMS error=\(normalized) max error=\(maximum)")
            precondition(normalized < 0.001 && maximum < 0.001, "Streaming waveform differs from full causal decode")
        }
        print("PASS: cached codec matches full decode across regular/irregular chunks without duplicate or missing samples")
    }
}
