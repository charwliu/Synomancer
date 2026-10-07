import Foundation
import AudioCommon

@main
@MainActor
struct SpeechPlaybackTests {
    static func main() async throws {
        setbuf(stdout, nil)
        let domain = "Synomancer.PlaybackTests.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: domain)!
        defer { preferences.removePersistentDomain(forName: domain) }
        preferences.set("local", forKey: "speechBackend")
        preferences.set("english", forKey: "localTTSLanguage")
        preferences.set("ryan", forKey: "localTTSSpeaker")
        preferences.set(CommandLine.arguments[1], forKey: SpeechCacheAccess.homeKey)
        let speaker = SpeechSynthesizer(preferences: preferences, playbackVolume: 0)
        let temp = FileManager.default.temporaryDirectory
        func files() -> Set<URL> {
            Set(((try? FileManager.default.contentsOfDirectory(at: temp, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.lastPathComponent.hasPrefix("synomancer-tts-") })
        }
        let before = files()
        var completions = 0
        let longAnswer = "# Answer\n" + String(repeating: "This is a local speech test. The next paragraph explains another point. ", count: 8)
        preferences.set(true, forKey: AutomaticResponseSpeech.enabledKey)
        var automaticText: String?
        let automatic = AutomaticResponseSpeech(preferences: preferences,
            speak: { text in
                automaticText = text
                await speaker.speak(text: text) { completions += 1 }
            }, stop: { await speaker.stopSpeaking() })
        automatic.responseCompleted("<think>Do not read this reasoning.</think>" + longAnswer)
        for _ in 0..<100 where !speaker.isSpeaking { try await Task.sleep(for: .milliseconds(10)) }
        precondition(automaticText == longAnswer.trimmingCharacters(in: .whitespacesAndNewlines) && speaker.isSpeaking)
        let start = Date()
        while speaker.isGenerating, Date().timeIntervalSince(start) < 90 {
            try await Task.sleep(for: .milliseconds(50))
        }
        precondition(speaker.errorMessage == nil, speaker.errorMessage ?? "")
        precondition(speaker.isSpeaking && !speaker.isGenerating)
        precondition(speaker.generatedAudioChunks > 0 && speaker.generatedSegments == 0 && speaker.segmentCount > 0)
        print("PASS: actual muted PCM playback begins before even the first text segment finishes generating; first playback=\(Date().timeIntervalSince(start))s")
        automatic.cancel()
        for _ in 0..<100 where speaker.isSpeaking { try await Task.sleep(for: .milliseconds(10)) }
        precondition(!speaker.isSpeaking && !speaker.isGenerating && completions == 1)
        for _ in 0..<100 where !files().subtracting(before).isEmpty {
            try await Task.sleep(for: .milliseconds(100))
        }
        precondition(files().subtracting(before).isEmpty, "Cancelled audio files were not removed")
        print("PASS: stop cancels generation/playback, finishes once and removes temporary files")
        print("PASS: auto-read routes only the completed answer into real local TTS and stops muted playback")

        // Start immediately after stop; the old GPU producer must finish before
        // a new stream reuses the cached model. Old buffer callbacks are ignored.
        await speaker.speak(text: "This is a speech playback test.") { completions += 1 }
        let second = Date()
        var received = 0
        var played = 0
        var firstPlayback: Double?
        while speaker.isSpeaking, Date().timeIntervalSince(second) < 60 {
            received = max(received, speaker.generatedAudioChunks)
            played = max(played, speaker.playedAudioChunks)
            if !speaker.isGenerating, firstPlayback == nil { firstPlayback = Date().timeIntervalSince(second) }
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(received > 1 && played > 0 && firstPlayback != nil)
        print("BENCH: warm App first PCM playback=\(firstPlayback!)s chunks=\(received), played before completion=\(played)")
        precondition(speaker.errorMessage == nil, speaker.errorMessage ?? "")
        precondition(!speaker.isSpeaking && completions == 2)
        precondition(files().subtracting(before).isEmpty)
        print("PASS: actual muted PCM playback completion resets state and cleans up")

        let multiSegment = String(repeating: "This is a local speech test. ", count: 12)
        precondition(SpeechTextChunks.split(multiSegment, firstLimit: 256, limit: 256).count > 1)
        await speaker.speak(text: multiSegment) { completions += 1 }
        let third = Date()
        var completedSegments = 0
        var playedChunks = 0
        while speaker.isSpeaking, Date().timeIntervalSince(third) < 90 {
            completedSegments = max(completedSegments, speaker.generatedSegments)
            playedChunks = max(playedChunks, speaker.playedAudioChunks)
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(speaker.errorMessage == nil, speaker.errorMessage ?? "")
        precondition(!speaker.isSpeaking && completions == 3 && completedSegments > 0 && playedChunks > 5)
        precondition(files().subtracting(before).isEmpty)
        precondition(speaker.lastPlaybackMetrics?.underruns == 0, "Playback queue ran dry inside a multi-segment answer")
        print("BENCH: complete playback underruns=\(speaker.lastPlaybackMetrics!.underruns), max queued PCM=\(speaker.lastPlaybackMetrics!.maximumBufferedSeconds)s")
        print("PASS: complete multi-segment answer drains PCM playback continuously and finishes once")
    }
}
