import SwiftUI

@Observable @MainActor
final class CompletionsPanelVM {
    var selectedText: String?
    var onReceiveText: (String) -> Void
    var messageResponse = ""
    var isReady = false
    let sentenceQueue = AsyncQueue<String>()
    private let preferences: UserDefaults
    private var generation: Task<Void, Never>?
    private var requestID: UUID?
    private var receivedTokens = 0
    private var delivered = ""

    init(preferences: UserDefaults = .standard, onReceiveText: @escaping (String) -> Void = { _ in }) {
        self.preferences = preferences
        self.onReceiveText = onReceiveText
    }

    static func constructPrompt(completion: CompletionInstructionSD, selectedText: String) -> String {
        if completion.instruction.contains("{{text}}") {
            return completion.instruction.replacingOccurrences(of: "{{text}}", with: selectedText)
        }
        return completion.instruction + " " + selectedText
    }

    func sendPrompt(completion: CompletionInstructionSD, model: LanguageModelSD) {
        guard let selectedText else { return }
        cancel()
        let prompt = Self.constructPrompt(completion: completion, selectedText: selectedText)
        let request = UUID()
        let modelID = model.requestModelID
        let provider = model.modelProvider
        let temperature = Float(completion.modelTemperature ?? 0.8)
        requestID = request
        isReady = true
        messageResponse = ""
        delivered = ""
        receivedTokens = 0
        generation = Task {
            do {
                _ = await sentenceQueue.dequeueAll()
                let result = try await ChatGenerationService.generate(modelID: modelID, provider: provider,
                    messages: [.init(role: "user", content: prompt)], preferences: preferences, temperature: temperature) { [self] text, count in
                        Task { @MainActor in
                            guard self.requestID == request, count > self.receivedTokens else { return }
                            self.receivedTokens = count
                            await self.receive(text, request: request)
                        }
                    }
                try Task.checkCancellation()
                guard requestID == request else { return }
                await receive(result.text, request: request)
                requestID = nil
                generation = nil
                isReady = false
            } catch {
                guard requestID == request else { return }
                requestID = nil
                generation = nil
                isReady = false
                if !(error is CancellationError) { AppStore.shared.uiLog(message: error.localizedDescription, status: .error) }
            }
        }
    }

    private func receive(_ text: String, request: UUID) async {
        guard requestID == request else { return }
        messageResponse = AssistantMessageContent(text, streaming: true).answer
        // Do not paste incomplete UTF-8 fragments emitted during token decoding.
        var stable = messageResponse
        while stable.last == "�" { stable.removeLast() }
        guard stable.utf8.starts(with: delivered.utf8) else { return }
        let delta = String(decoding: stable.utf8.dropFirst(delivered.utf8.count), as: UTF8.self)
        delivered = stable
        if !delta.isEmpty { await sentenceQueue.enqueue(delta) }
    }

    func cancel() {
        generation?.cancel()
        generation = nil
        requestID = nil
        isReady = false
    }
}
