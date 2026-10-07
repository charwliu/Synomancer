import Foundation
import Qwen3ASR
import AudioCommon

@main
struct ASRSmoke {
    static func main() async throws {
        let cache = URL(fileURLWithPath: CommandLine.arguments[1])
        let audioURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let selection = LocalASRCache.resolve(in: cache)
        let model = try await Qwen3ASRModel.fromPretrained(
            modelId: selection?.modelId ?? "aufklarer/Qwen3-ASR-1.7B-MLX-4bit",
            cacheDir: selection?.directory ?? cache,
            offlineMode: CommandLine.arguments.contains("--offline"),
            progressHandler: { value, status in print("\(Int(value * 100))% \(status)") })
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let start = Date()
        let text = try model.transcribeCheckingCancellation(audio: audio)
        guard !text.isEmpty, !text.contains("Text decoder not loaded") else {
            fatalError("No transcription")
        }
        for word in CommandLine.arguments.dropFirst(3).filter({ !$0.hasPrefix("--") }) {
            guard text.lowercased().contains(word.lowercased()) else {
                fatalError("Missing expected word: \(word)")
            }
        }
        print("TRANSCRIPT: \(text)")
        print("SECONDS: \(Date().timeIntervalSince(start)) AUDIO: \(Double(audio.count) / 16000)")
        if CommandLine.arguments.contains("--streaming") {
            let buffer = StreamingSpeechBuffer(sampleRate: 16000)
            var transcript = StreamingSpeechTranscript()
            var previous = 0
            var updates = 0
            // The public fixture has speech in seconds 5–9, before its 20s end.
            for seconds in [2, 4, 6, 8, 10] {
                let end = min(seconds * 16000, audio.count)
                buffer.append(Array(audio[previous..<end]))
                previous = end
                let snapshot = buffer.snapshot()
                if !snapshot.hasSpeech { buffer.discard(through: snapshot.endFrame); continue }
                let partial = try model.transcribeCheckingCancellation(audio: snapshot.samples)
                let value = transcript.update(partial: partial, commit: snapshot.hasSilentTail)
                if snapshot.hasSilentTail { buffer.discard(through: snapshot.endFrame) }
                if !value.isEmpty { updates += 1 }
                print("PARTIAL @\(seconds)s: \(value)")
            }
            guard updates >= 2, transcript.text.lowercased().contains("guarantee"),
                  transcript.text.lowercased().contains("tomorrow") else {
                fatalError("Expected live revisions before the end of the fixture")
            }
            print("PASS: live revisions before stop")
        }
        withUnsafeCurrentTask { $0?.cancel() }
        do {
            _ = try model.transcribeCheckingCancellation(audio: audio)
            fatalError("Cancelled inference did not stop")
        } catch is CancellationError {
            print("PASS: cancellation")
        }
    }
}
