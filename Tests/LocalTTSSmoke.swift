import Foundation
import AVFoundation
import AudioCommon
import Qwen3ASR
import Qwen3TTS
import MLX

@main
struct TTSSmoke {
    static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let selection = LocalTTSCache.resolve(in: root),
              let asrCache = LocalASRCache.resolve(in: root) else { fatalError("Missing local speech caches") }
        MLX.seed(42)
        let tts = try Qwen3TTSModel.fromLocal(modelDirectory: selection.model, tokenizerDirectory: selection.tokenizer,
                                             wiredMemoryPolicy: .none, runtimeTalkerBits: 8, runtimePredictorBits: 8)
        print("PASS: offline CustomVoice and codec loading")
        let outputs = [
            ("Hello. This is a local speech test.", "english", "ryan", ["local", "test"]),
            ("你好，这是本地语音合成测试。", "chinese", "vivian", ["语音", "测试"]),
        ]
        let destination = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for (index, entry) in outputs.enumerated() {
            let start = Date()
            let samples = try tts.synthesizeCheckingCancellation(text: entry.0, language: entry.1, speaker: entry.2,
                                                                 sampling: .default, languageExplicit: true)
            precondition(!samples.isEmpty && samples.allSatisfy(\.isFinite))
            let energy = samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count)
            precondition(energy > 0.000001, "Silent TTS output")
            let url = destination.appendingPathComponent("tts-\(index).wav")
            try WAVWriter.write(samples: samples, sampleRate: 24000, to: url)
            let player = try AVAudioPlayer(contentsOf: url)
            precondition(player.duration > 0 && Int(player.format.sampleRate) == 24000)
            print("PASS: \(entry.1) finite audible PCM, WAV/player decode; audio=\(player.duration)s synthesis=\(Date().timeIntervalSince(start))s")
        }
        let asr = try await Qwen3ASRModel.fromPretrained(modelId: asrCache.modelId,
                                                        cacheDir: asrCache.directory, offlineMode: true)
        for (index, entry) in outputs.enumerated() {
            let url = destination.appendingPathComponent("tts-\(index).wav")
            let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
            let text = try asr.transcribeCheckingCancellation(audio: audio)
            for word in entry.3 {
                precondition(text.lowercased().contains(word), "Missing expected word \(word): \(text)")
            }
            print("PASS: \(entry.1) TTS → local ASR: \(text)")
        }
        withUnsafeCurrentTask { $0?.cancel() }
        do {
            _ = try tts.synthesizeCheckingCancellation(text: "Cancelled speech.", language: "english", speaker: "ryan")
            fatalError("Cancelled TTS generation did not stop")
        } catch is CancellationError { print("PASS: TTS cancellation") }
    }
}
