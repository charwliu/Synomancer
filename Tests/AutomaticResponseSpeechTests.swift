import Foundation

@main
@MainActor
struct AutomaticResponseSpeechTests {
    static func main() async throws {
        let domain = "Synomancer.AutomaticSpeechTests.\(UUID())"
        let preferences = UserDefaults(suiteName: domain)!
        defer { preferences.removePersistentDomain(forName: domain) }
        var spoken: [String] = []
        var stops = 0
        let automatic = AutomaticResponseSpeech(preferences: preferences,
            speak: { spoken.append($0) }, stop: { stops += 1 })
        func settle() async throws { try await Task.sleep(for: .milliseconds(20)) }

        automatic.responseCompleted("Default is silent")
        try await settle()
        precondition(spoken.isEmpty)
        preferences.set(true, forKey: AutomaticResponseSpeech.enabledKey)
        automatic.responseCompleted("<think>Private reasoning</think>正式答案")
        try await settle()
        precondition(spoken == ["正式答案"])
        automatic.cancel()
        try await settle()
        precondition(stops == 1)

        automatic.responseCompleted("<think>Still thinking")
        automatic.responseCompleted("\n</think>\n")
        try await settle()
        precondition(spoken.count == 1)
        automatic.responseCompleted("Cancelled before speaking")
        automatic.cancel()
        try await settle()
        precondition(spoken.count == 1)

        automatic.responseCompleted("Mode was disabled")
        preferences.set(false, forKey: AutomaticResponseSpeech.enabledKey)
        try await settle()
        precondition(spoken.count == 1)
        preferences.set(true, forKey: AutomaticResponseSpeech.enabledKey)
        automatic.responseCompleted("Superseded answer")
        automatic.responseCompleted("Legacy reasoning\n</think>\nReplacement answer")
        try await settle()
        precondition(spoken == ["正式答案", "Replacement answer"])
        automatic.cancel()
        try await settle()
        precondition(stops == 2)
        print("PASS: default silence, answer-only speech, thinking-only responses, disable/cancel and immediate replacement")
    }
}
