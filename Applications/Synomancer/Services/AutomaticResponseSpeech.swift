import Foundation

/// Triggered by a newly completed request, never by loading saved messages.
@MainActor
final class AutomaticResponseSpeech {
    static let enabledKey = "automaticallyReadResponses"
    private let preferences: UserDefaults
    private let speak: @MainActor (String) async -> Void
    private let stop: @MainActor () async -> Void
    private var task: Task<Void, Never>?
    private var revision = 0
    private var ownsPlayback = false

    init(preferences: UserDefaults, speak: @escaping @MainActor (String) async -> Void,
         stop: @escaping @MainActor () async -> Void) {
        self.preferences = preferences
        self.speak = speak
        self.stop = stop
    }

    func responseCompleted(_ content: String) {
        guard preferences.bool(forKey: Self.enabledKey) else { return }
        let answer = AssistantMessageContent(content).answer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else { return }
        cancel()
        revision += 1
        let request = revision
        task = Task { [weak self] in
            guard let self, !Task.isCancelled, self.revision == request,
                  self.preferences.bool(forKey: Self.enabledKey) else { return }
            self.ownsPlayback = true
            await self.speak(answer)
            if self.revision == request { self.task = nil }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        revision += 1
        guard ownsPlayback else { return }
        ownsPlayback = false
        let request = revision
        Task { [weak self] in
            guard let self, self.revision == request else { return }
            await self.stop()
        }
    }
}
