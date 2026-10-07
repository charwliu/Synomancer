import AVFoundation
import Combine
import Foundation
@preconcurrency import Qwen3ASR
import AudioCommon

/// Keeps the MLX model and inference off the main actor, and serializes its use.
private actor LocalSpeechModel {
    static let shared = LocalSpeechModel()
    private var model: Qwen3ASRModel?
    private var loading = false
    private var cacheIdentity: SpeechCacheAccess.Configuration?

    func prepare(configuration: SpeechCacheAccess.Configuration, progress: @escaping @Sendable (Double) -> Void) async throws {
        if model != nil, cacheIdentity == configuration { return }
        guard !loading else { throw SpeechRecognizer.RecordingError.modelBusy }
        loading = true
        defer { loading = false }
        model = nil
        let root = try configuration.resolve()
        let accessing = root.startAccessingSecurityScopedResource()
        defer { if accessing { root.stopAccessingSecurityScopedResource() } }
        guard let local = LocalASRCache.resolve(in: root) else {
            throw SpeechRecognizer.RecordingError.invalidCache
        }
        model = try await Qwen3ASRModel.fromPretrained(
            modelId: local.modelId, cacheDir: local.directory, offlineMode: true,
            progressHandler: { value, _ in progress(value) }
        )
        cacheIdentity = configuration
        try Task.checkCancellation()
    }

    func transcribe(samples input: [Float], sampleRate: Int) throws -> String {
        guard let model else { throw SpeechRecognizer.RecordingError.modelUnavailable }
        let samples = sampleRate == 16000 ? input : AudioFileLoader.resample(input, from: sampleRate, to: 16000)
        // Bound encoder memory and decode long recordings in 20-second windows.
        let window = 20 * 16000
        var parts: [String] = []
        for offset in stride(from: 0, to: samples.count, by: window) {
            try Task.checkCancellation()
            let audio = Array(samples[offset..<min(offset + window, samples.count)])
            guard audio.count >= 1600 else { continue }
            // Avoid hallucinated text on a silent recording.
            let energy = audio.reduce(Float(0)) { $0 + $1 * $1 } / Float(audio.count)
            guard energy > 0.000001 else { continue }
            let text = try model.transcribeCheckingCancellation(audio: audio)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { parts.append(text) }
        }
        return parts.joined(separator: " ")
    }
}

@MainActor
final class SpeechRecognizer: ObservableObject {
    enum Phase { case idle, preparing, recording, transcribing }
    enum RecordingError: LocalizedError {
        case microphoneDenied, couldNotRecord, modelBusy, modelUnavailable, invalidCache
        var errorDescription: String? {
            switch self {
            case .microphoneDenied: return "Microphone access is required for voice input. Enable it in System Settings."
            case .couldNotRecord: return "Unable to start microphone recording."
            case .modelBusy: return "The speech model is loading. Please try again shortly."
            case .invalidCache: return "The selected speech model cache is unavailable or incomplete. Check HF_HOME and authorize its folder in Settings."
            case .modelUnavailable: return "The local speech model is not ready."
            }
        }
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var transcript = ""
    @Published private(set) var preparationProgress: Double = 0
    @Published var errorMessage: String?
    private var audioEngine: AVAudioEngine?
    private var audioBuffer: StreamingSpeechBuffer?
    private var streamingTask: Task<Void, Never>?
    private var transcriptState = StreamingSpeechTranscript()
    private var onUpdate: (@Sendable @MainActor (String) -> Void)?

    func startRecording(onUpdate: @escaping @Sendable @MainActor (String) -> Void) async -> Bool {
        guard phase == .idle else { return false }
        phase = .preparing
        preparationProgress = 0
        transcript = ""
        transcriptState = StreamingSpeechTranscript()
        self.onUpdate = onUpdate
        errorMessage = nil
        do {
            guard await AVCaptureDevice.requestAccess(for: .audio) else {
                throw RecordingError.microphoneDenied
            }
            try Task.checkCancellation()
            let cache = try await SpeechCacheAccess.authorizedConfiguration()
            try Task.checkCancellation()
            try await LocalSpeechModel.shared.prepare(configuration: cache) { [weak self] value in
                Task { @MainActor in self?.preparationProgress = value }
            }
            try Task.checkCancellation()
#if os(iOS) || os(visionOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement)
            try session.setActive(true)
#endif
            let engine = AVAudioEngine()
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate > 0, format.channelCount > 0 else {
                throw RecordingError.couldNotRecord
            }
            let buffer = StreamingSpeechBuffer(sampleRate: Int(format.sampleRate))
            let tap: AVAudioNodeTapBlock = { pcm, _ in
                guard let channels = pcm.floatChannelData else { return }
                let frames = Int(pcm.frameLength)
                let count = Int(pcm.format.channelCount)
                var mono = [Float](repeating: 0, count: frames)
                for channel in 0..<count {
                    for frame in 0..<frames { mono[frame] += channels[channel][frame] / Float(count) }
                }
                buffer.append(mono)
            }
            if #available(macOS 27, iOS 27, visionOS 27, *) {
                try input.__installTap(onBus: 0, bufferSize: 4096, format: format, error: (), block: tap)
            } else {
                input.installTap(onBus: 0, bufferSize: 4096, format: format, block: tap)
            }
            audioEngine = engine
            audioBuffer = buffer
            engine.prepare()
            try engine.start()
            phase = .recording
            streamingTask = Task { [weak self] in
                while !Task.isCancelled {
                    do {
                        try await Task.sleep(for: .seconds(2))
                        guard let self, self.phase == .recording else { return }
                        try await self.updateTranscript(final: false)
                    } catch {
                        guard let self else { return }
                        if !(error is CancellationError) {
                            self.cancelRecording()
                            self.errorMessage = error.localizedDescription
                        }
                        return
                    }
                }
            }
            return true
        } catch {
            cancelRecording()
            if !(error is CancellationError) { errorMessage = error.localizedDescription }
            return false
        }
    }

    private func updateTranscript(final: Bool) async throws {
        guard let buffer = audioBuffer else { return }
        let snapshot = buffer.snapshot()
        guard !snapshot.samples.isEmpty else { return }
        if !snapshot.hasSpeech {
            buffer.discard(through: snapshot.endFrame)
            return
        }
        // Short initial fragments wait for more context, except when stopped.
        guard final || snapshot.samples.count >= snapshot.sampleRate else { return }
        let text = try await LocalSpeechModel.shared.transcribe(
            samples: snapshot.samples, sampleRate: snapshot.sampleRate)
        try Task.checkCancellation()
        guard phase == .recording || phase == .transcribing else { return }
        let commit = final || snapshot.hasSilentTail || snapshot.samples.count >= 15 * snapshot.sampleRate
        transcript = transcriptState.update(partial: text, commit: commit)
        if !text.isEmpty { onUpdate?(transcript) }
        if commit { buffer.discard(through: snapshot.endFrame) }
    }

    func finishRecording() async -> String? {
        guard phase == .recording else { return nil }
        phase = .transcribing
        stopCapture()
        let pending = streamingTask
        pending?.cancel()
        streamingTask = nil
        // Drain the previous decode before finalizing to avoid duplicate commits.
        await pending?.value
        defer {
            audioBuffer = nil
            onUpdate = nil
            phase = .idle
        }
        do {
            try await updateTranscript(final: true)
            return transcript
        } catch {
            if !(error is CancellationError) { errorMessage = error.localizedDescription }
            return nil
        }
    }

    func cancelRecording() {
        streamingTask?.cancel()
        streamingTask = nil
        stopCapture()
        audioBuffer = nil
        onUpdate = nil
        phase = .idle
    }

    private func stopCapture() {
        if let audioEngine {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        audioEngine = nil
        deactivateAudioSession()
    }

    private func deactivateAudioSession() {
#if os(iOS) || os(visionOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
#endif
    }
}
