import Foundation
import AudioCommon
@preconcurrency import Qwen3TTS

/// The model is confined to this actor; generation never runs on the UI actor.
actor LocalTTSService {
    static let shared = LocalTTSService()
    private var model: Qwen3TTSModel?
    private var cacheIdentity: SpeechCacheAccess.Configuration?
    private var streaming = false

    private func prepare(cache: SpeechCacheAccess.Configuration, progress: @escaping @Sendable (Double) -> Void) throws -> Qwen3TTSModel {
        try Task.checkCancellation()
        if model == nil || cacheIdentity != cache {
            model = nil
            let root = try cache.resolve()
            let accessing = root.startAccessingSecurityScopedResource()
            defer { if accessing { root.stopAccessingSecurityScopedResource() } }
            let selection = LocalTTSCache.resolve(in: root)
            guard let selection else {
                throw TTSError.cacheMissing(cache.path)
            }
            model = try Qwen3TTSModel.fromLocal(modelDirectory: selection.model, tokenizerDirectory: selection.tokenizer,
                                               wiredMemoryPolicy: .none, runtimeTalkerBits: 8, runtimePredictorBits: 8, progressHandler: { value, _ in progress(value) })
            cacheIdentity = cache
        }
        try Task.checkCancellation()
        guard let model else { throw TTSError.cacheMissing(cache.path) }
        progress(1)
        return model
    }

    /// Keep a cancelled producer alive until its final GPU step exits, before
    /// allowing the next request to reuse the model's compiled functions/caches.
    func stream(text: String, language: String, speaker: String, cache: SpeechCacheAccess.Configuration,
                progress: @escaping @Sendable (Double) -> Void,
                consume: @escaping @Sendable (AudioChunk) async throws -> Void) async throws {
        while streaming {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(20))
        }
        try Task.checkCancellation()
        streaming = true
        defer { streaming = false }
        let model = try prepare(cache: cache, progress: progress)
        let voice = speaker == "auto" ? (language == "chinese" ? "vivian" : "ryan") : speaker
        let consumer = Task {
            let completion = SpeechStreamCompletion()
            var iterator = model.synthesizeStream(
                text: text, language: language, speaker: voice,
                streaming: .init(firstChunkFrames: 6, chunkFrames: 12, decoderLeftContext: 16),
                languageExplicit: true,
                onCompletion: { Task { await completion.finish() } }).makeAsyncIterator()
            var receivedAudio = false
            do {
                while let chunk = try await iterator.next() {
                    try Task.checkCancellation()
                    guard chunk.sampleRate == 24000, chunk.samples.allSatisfy(\.isFinite) else {
                        throw TTSError.invalidAudio
                    }
                    if !chunk.samples.isEmpty { receivedAudio = true }
                    try await consume(chunk)
                }
                guard receivedAudio else { throw TTSError.invalidAudio }
            } catch {
                // Cancel this child and advance its iterator to propagate cancellation
                // to synthesizeStream, then wait for actual producer termination.
                withUnsafeCurrentTask { $0?.cancel() }
                _ = try? await iterator.next()
                await completion.wait()
                throw error
            }
            await completion.wait()
        }
        try await withTaskCancellationHandler {
            try await consumer.value
        } onCancel: {
            consumer.cancel()
        }
    }

    enum TTSError: LocalizedError {
        case cacheMissing(String), invalidAudio
        var errorDescription: String? {
            switch self {
            case .cacheMissing(let path): return "No complete Qwen3-TTS 1.7B CustomVoice model and speech tokenizer found in HF_HOME: \(path). Choose the huggingface/hub root in Settings."
            case .invalidAudio: return "The local TTS model did not produce valid audio."
            }
        }
    }
}

private actor SpeechStreamCompletion {
    private var finished = false
    private var waiter: CheckedContinuation<Void, Never>?

    func finish() {
        finished = true
        waiter?.resume()
        waiter = nil
    }

    func wait() async {
        if finished { return }
        await withCheckedContinuation { waiter = $0 }
    }
}
