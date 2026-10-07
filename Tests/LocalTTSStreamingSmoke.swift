import Foundation
import AVFoundation
import AudioCommon
import Qwen3ASR
import Qwen3TTS
import MLX

@main
struct TTSStreamingSmoke {
    static func main() async throws {
        setbuf(stdout, nil)
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let selection = LocalTTSCache.resolve(in: root),
              let asrCache = LocalASRCache.resolve(in: root) else { fatalError("Missing local speech caches") }
        let loading = Date()
        let tts = try Qwen3TTSModel.fromLocal(modelDirectory: selection.model, tokenizerDirectory: selection.tokenizer,
                                             wiredMemoryPolicy: .none, runtimeTalkerBits: 8, runtimePredictorBits: 8)
        print("PASS: offline TTS loading/warmup=\(Date().timeIntervalSince(loading))s")
        let outputs = [
            ("This is a streaming speech test. Audio starts before the entire sentence has finished generating.", "english", "ryan", ["streaming", "test", "finished"]),
            ("你好，这是本地流式语音合成测试。第一段音频生成后就会开始播放，不需要等待整个回答。", "chinese", "vivian", ["语音", "测试", "回答"]),
        ]
        let destination = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for (index, entry) in outputs.enumerated() {
            MLX.seed(42)
            let batchStart = Date()
            let batch = try tts.synthesizeCheckingCancellation(text: entry.0, language: entry.1, speaker: entry.2,
                                                                languageExplicit: true)
            let batchTime = Date().timeIntervalSince(batchStart)
            try WAVWriter.write(samples: batch, sampleRate: 24000, to: destination.appendingPathComponent("tts-batch-\(index).wav"))
            print("BENCH: \(entry.1) batch audio=\(Double(batch.count) / 24000)s total=\(batchTime)s RTF=\(batchTime / (Double(batch.count) / 24000))")
            MLX.seed(42)
            var samples: [Float] = []
            var first: Double?
            var firstAudible: Double?
            var chunks = 0
            var finals = 0
            var expectedFrame = 0
            let start = Date()
            for try await chunk in tts.synthesizeStream(
                text: entry.0, language: entry.1, speaker: entry.2,
                streaming: .init(firstChunkFrames: 3, chunkFrames: 6, decoderLeftContext: 10), languageExplicit: true) {
                precondition(chunk.sampleRate == 24000 && chunk.samples.allSatisfy(\.isFinite))
                // frameIndex addresses codec frames, not PCM sample offsets. The
                // decoder's causal startup makes the first PCM block shorter.
                precondition(chunk.frameIndex == expectedFrame, "Non-contiguous codec frames")
                if chunk.isFinal { finals += 1 }
                if !chunk.samples.isEmpty {
                    expectedFrame += chunks == 0 ? 4 : 6
                    chunks += 1
                    if first == nil { first = Date().timeIntervalSince(start) }
                    let energy = chunk.samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(chunk.samples.count)
                    if energy > 0.000001, firstAudible == nil { firstAudible = Date().timeIntervalSince(start) }
                    samples.append(contentsOf: chunk.samples)
                }
            }
            let total = Date().timeIntervalSince(start)
            precondition(chunks > 2 && finals == 1 && firstAudible != nil)
            precondition(first! < total)
            let url = destination.appendingPathComponent("tts-stream-\(index).wav")
            try WAVWriter.write(samples: samples, sampleRate: 24000, to: url)
            let player = try AVAudioPlayer(contentsOf: url)
            precondition(player.duration > 0)
            print("PASS: \(entry.1) stream chunks=\(chunks) first=\(first!)s first audible=\(firstAudible!)s audio=\(player.duration)s total=\(total)s RTF=\(total / player.duration)")
        }
        let asr = try await Qwen3ASRModel.fromPretrained(modelId: asrCache.modelId,
                                                        cacheDir: asrCache.directory, offlineMode: true)
        for (index, entry) in outputs.enumerated() {
            let url = destination.appendingPathComponent("tts-stream-\(index).wav")
            let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
            let text = try asr.transcribeCheckingCancellation(audio: audio)
            let batchAudio = try AudioFileLoader.load(url: destination.appendingPathComponent("tts-batch-\(index).wav"), targetSampleRate: 16000)
            print("TRANSCRIPT batch: \(entry.1): \(try asr.transcribeCheckingCancellation(audio: batchAudio))")
            print("TRANSCRIPT: \(entry.1): \(text)")
            for word in entry.3 {
                precondition(text.lowercased().contains(word), "Missing expected word \(word): \(text)")
            }
            print("PASS: \(entry.1) streaming TTS → local ASR: \(text)")
        }
    }
}
