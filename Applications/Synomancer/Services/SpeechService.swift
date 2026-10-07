import Foundation
import AVFoundation
import SwiftUI
import AudioCommon

@MainActor
private final class SpeechEventsDelegate: NSObject, AVSpeechSynthesizerDelegate {
    var onFinished: ((ObjectIdentifier) -> Void)?
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let identity = ObjectIdentifier(utterance)
        Task { @MainActor [weak self] in self?.onFinished?(identity) }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let identity = ObjectIdentifier(utterance)
        Task { @MainActor [weak self] in self?.onFinished?(identity) }
    }
}

@MainActor
final class SpeechSynthesizer: NSObject, ObservableObject {
    static let shared = SpeechSynthesizer()
    private let preferences: UserDefaults
    private let playbackVolume: Float
    private let synthesizer = AVSpeechSynthesizer()
    private let speechDelegate = SpeechEventsDelegate()
    private let audioTeardownQueue = DispatchQueue(label: "SpeechSynthesizer.audioTeardown", qos: .userInteractive)
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var pendingBuffers = 0
    private var bufferedSeconds = 0.0
    private var generationFinished = false
    private var playbackStarted = false
    private var underruns = 0
    private var maximumBufferedSeconds = 0.0
    private var playbackTail = SpeechPlaybackTail()
    private let prebufferSeconds = 0.6
    private let maximumQueueSeconds = 4.0
    struct PlaybackMetrics {
        let underruns: Int
        let maximumBufferedSeconds: Double
    }
    private(set) var lastPlaybackMetrics: PlaybackMetrics?
    private var watchdog: Task<Void, Never>?
    private var lastActivity = Date()
    private var generationTask: Task<Void, Never>?
    private var requestID: UUID?
    private var utteranceID: ObjectIdentifier?
    private var completion: (() -> Void)?

    @Published private(set) var isSpeaking = false
    @Published private(set) var isGenerating = false
    @Published private(set) var generatedAudioChunks = 0
    @Published private(set) var playedAudioChunks = 0
    @Published private(set) var generatedSegments = 0
    @Published private(set) var segmentCount = 0
    @Published private(set) var generationStatus = ""
    @Published var errorMessage: String?
    @Published var voices: [AVSpeechSynthesisVoice] = []

    override convenience init() {
        self.init(preferences: .standard)
    }

    init(preferences: UserDefaults, playbackVolume: Float = 1) {
        self.preferences = preferences
        self.playbackVolume = playbackVolume
        super.init()
        synthesizer.delegate = speechDelegate
        speechDelegate.onFinished = { [weak self] identity in
            guard let self, self.utteranceID == identity else { return }
            self.finish()
        }
        fetchVoices()
    }

    func getVoiceIdentifier() -> String? {
        let selected = preferences.string(forKey: "voiceIdentifier")
        return voices.first(where: { $0.identifier == selected })?.identifier ?? voices.first?.identifier
    }

    func speak(text: String, onFinished: @escaping () -> Void = {}) async {
        await stopSpeaking()
        guard !Task.isCancelled else { onFinished(); return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { onFinished(); return }
        let request = UUID()
        requestID = request
        completion = onFinished
        errorMessage = nil
        isSpeaking = true
        underruns = 0
        maximumBufferedSeconds = 0
        lastPlaybackMetrics = nil
        if preferences.string(forKey: "speechBackend") == "local" {
            isGenerating = true
            generationFinished = false
            generationStatus = "Preparing Local Speech…"
            lastActivity = Date()
            watchdog = Task { [weak self] in
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .seconds(1)) } catch { return }
                    guard let self, self.requestID == request else { return }
                    if self.isGenerating, Date().timeIntervalSince(self.lastActivity) > 120 {
                        self.errorMessage = "Local speech synthesis made no progress for two minutes. Try again or select a system voice."
                        self.finish()
                        return
                    }
                }
            }
            let settings = preferences
            let preferredLanguage = settings.string(forKey: "localTTSLanguage") ?? "auto"
            let language = preferredLanguage == "auto"
                ? (text.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) } ? "chinese" : "english")
                : preferredLanguage
            let speaker = settings.string(forKey: "localTTSSpeaker") ?? "auto"
            generationTask = Task { [weak self] in
                do {
                    let cache = try await SpeechCacheAccess.authorizedConfiguration(in: settings)
                    try Task.checkCancellation()
                    let chunks = SpeechTextChunks.split(text, firstLimit: 256, limit: 256)
                    guard !chunks.isEmpty else { self?.finish(); return }
                    self?.segmentCount = chunks.count
                    for (index, chunk) in chunks.enumerated() {
                        try Task.checkCancellation()
                        guard let self, self.requestID == request else { return }
                        self.generationStatus = "Generating Speech \(index + 1) / \(chunks.count)…"
                        self.lastActivity = Date()
                        try await LocalTTSService.shared.stream(
                            text: chunk, language: language, speaker: speaker, cache: cache,
                            progress: { [weak self] value in
                                Task { @MainActor in
                                    guard let self, self.requestID == request else { return }
                                    self.lastActivity = Date()
                                    self.generationStatus = value < 1
                                        ? "Loading Speech Model \(Int(value * 100))%…"
                                        : "Generating Speech \(index + 1) / \(chunks.count)…"
                                }
                            }, consume: { [weak self] chunk in
                                try await self?.enqueue(chunk, request: request)
                            })
                        try Task.checkCancellation()
                        guard self.requestID == request else { return }
                        self.lastActivity = Date()
                        self.generatedSegments = index + 1
                    }
                    guard let self, self.requestID == request else { return }
                    // Flush once for the whole answer, including when the model
                    // signalled EOS with an empty final chunk. Keep chunk and text
                    // segment boundaries unchanged throughout the utterance.
                    try await self.enqueueSamples(self.playbackTail.finish(), sampleRate: 24000, request: request)
                    try Task.checkCancellation()
                    guard self.requestID == request else { return }
                    self.generationFinished = true
                    if self.pendingBuffers == 0 { self.finish() }
                    else { try self.startPlaybackIfReady(force: true) }

                } catch {
                    guard let self, self.requestID == request else { return }
                    if !(error is CancellationError) { self.errorMessage = error.localizedDescription }
                    self.finish()
                }
            }
        } else {
            guard let identifier = getVoiceIdentifier() else {
                errorMessage = "No system voice is available."
                finish()
                return
            }
            do { try activateAudioSession() } catch {
                errorMessage = error.localizedDescription
                finish()
                return
            }
            let utterance = AVSpeechUtterance(string: text)
            utterance.voice = AVSpeechSynthesisVoice(identifier: identifier)
            utterance.rate = 0.5
            utteranceID = ObjectIdentifier(utterance)
            synthesizer.speak(utterance)
        }
    }

    func stopSpeaking() async {
        generationTask?.cancel()
        utteranceID = nil
        synthesizer.stopSpeaking(at: .immediate)
        finish()
    }

    private func enqueue(_ chunk: AudioChunk, request: UUID) async throws {
        try Task.checkCancellation()
        guard requestID == request else { throw CancellationError() }
        try await enqueueSamples(playbackTail.append(chunk.samples), sampleRate: chunk.sampleRate, request: request)
    }

    private func enqueueSamples(_ samples: [Float], sampleRate: Int, request: UUID) async throws {
        try Task.checkCancellation()
        guard requestID == request else { throw CancellationError() }
        lastActivity = Date()
        guard !samples.isEmpty else { return }
        // Keep enough queued PCM to cover decoder and next-segment preparation jitter.
        while bufferedSeconds >= maximumQueueSeconds {
            try await Task.sleep(for: .milliseconds(20))
            guard requestID == request else { throw CancellationError() }
        }
        try Task.checkCancellation()
        guard requestID == request else { throw CancellationError() }
        if audioEngine == nil {
            try activateAudioSession()
            let engine = AVAudioEngine()
            let node = AVAudioPlayerNode()
            guard let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1) else {
                throw LocalTTSService.TTSError.invalidAudio
            }
            engine.attach(node)
            if #available(macOS 27, iOS 27, visionOS 27, *) {
                try engine.connectNode(node, to: engine.mainMixerNode, format: format)
            } else {
                engine.connect(node, to: engine.mainMixerNode, format: format)
            }
            node.volume = playbackVolume
            engine.prepare()
            try engine.start()
            audioEngine = engine
            playerNode = node
        }
        guard let node = playerNode,
              let format = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { throw LocalTTSService.TTSError.invalidAudio }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { samples in
            channel.update(from: samples.baseAddress!, count: samples.count)
        }
        let duration = Double(samples.count) / Double(sampleRate)
        pendingBuffers += 1
        bufferedSeconds += duration
        maximumBufferedSeconds = max(maximumBufferedSeconds, bufferedSeconds)
        generatedAudioChunks += 1
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.requestID == request else { return }
                self.pendingBuffers -= 1
                self.bufferedSeconds = max(0, self.bufferedSeconds - duration)
                self.playedAudioChunks += 1
                self.lastActivity = Date()
                if self.pendingBuffers == 0 {
                    if self.generationFinished { self.finish() }
                    else {
                        self.underruns += 1
                        self.playerNode?.pause()
                        self.playbackStarted = false
                        self.isGenerating = true
                    }
                }
            }
        }
        try startPlaybackIfReady(force: generationFinished)
    }

    private func startPlaybackIfReady(force: Bool = false) throws {
        guard !playbackStarted, pendingBuffers > 0,
              force || bufferedSeconds >= prebufferSeconds, let node = playerNode else { return }
        if #available(macOS 27, iOS 27, visionOS 27, *) { try node.playAudio() }
        else { node.play() }
        playbackStarted = true
        isGenerating = false
    }

    private func finish() {
        if requestID != nil {
            lastPlaybackMetrics = .init(underruns: underruns, maximumBufferedSeconds: maximumBufferedSeconds)
        }
        generationTask?.cancel()
        watchdog?.cancel()
        watchdog = nil
        tearDownLocalPlayback()
        playbackTail = SpeechPlaybackTail()
        pendingBuffers = 0
        bufferedSeconds = 0
        generationFinished = false
        playbackStarted = false
        generationStatus = ""
        generatedSegments = 0
        segmentCount = 0
        generatedAudioChunks = 0
        playedAudioChunks = 0
        requestID = nil
        utteranceID = nil
        generationTask = nil
        isSpeaking = false
        isGenerating = false
#if os(iOS) || os(visionOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
#endif
        let callback = completion
        completion = nil
        callback?()
    }

    private func tearDownLocalPlayback() {
        let node = playerNode
        let engine = audioEngine
        playerNode = nil
        audioEngine = nil

        audioTeardownQueue.async {
            node?.stop()
            engine?.stop()
        }
    }

    private func activateAudioSession() throws {
#if os(iOS) || os(visionOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: .duckOthers)
        try session.setActive(true)
#endif
    }

    func fetchVoices() {
        let fetched = AVSpeechSynthesisVoice.speechVoices().sorted { $0.quality.rawValue > $1.quality.rawValue }
        if !voices.elementsEqual(fetched, by: { $0.identifier == $1.identifier }) { voices = fetched }
    }
}
