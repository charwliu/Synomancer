import Foundation
import SwiftData

@main
@MainActor
struct LocalChatStoreTests {
    static func main() async throws {
        setbuf(stdout, nil)
        let domain = "Synomancer.ChatStoreTests.\(UUID())"
        let preferences = UserDefaults(suiteName: domain)!
        defer { preferences.removePersistentDomain(forName: domain) }
        preferences.set(CommandLine.arguments[1], forKey: SpeechCacheAccess.homeKey)
        let schema = Schema([LanguageModelSD.self, ConversationSD.self, MessageSD.self, CompletionInstructionSD.self])
        let database = try ModelContainer(for: schema, configurations: [ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)])
        let data = SwiftDataService(modelContainer: database)
        let legacy = LanguageModelSD(name: "legacy-ollama:latest", modelProvider: .ollama)
        let oldConversation = ConversationSD(name: "Legacy conversation")
        oldConversation.model = legacy
        let originalUser = MessageSD(content: "The passphrase is REDWOOD-472.", role: "user")
        originalUser.conversation = oldConversation
        let originalAssistant = MessageSD(content: "Remember the user's passphrase.\n</think>\n\nI will remember REDWOOD-472.", role: "assistant", done: true)
        originalAssistant.conversation = oldConversation
        try data.saveModels(models: [legacy])
        try data.createConversation(oldConversation)
        try data.createMessage(originalUser)
        try data.createMessage(originalAssistant)
        precondition(originalAssistant.displayContent == "I will remember REDWOOD-472.")
        precondition(MessageSD(content: "Use `</think>`.", role: "user").displayContent == "Use `</think>`.")
        let savedLegacy = try data.fetchMessages(oldConversation.id)
        precondition(savedLegacy.last!.content == originalAssistant.content)
        print("PASS: saved legacy orphan-tag history is preserved; assistant copy/speech excludes thinking; user literals are unchanged")
        preferences.set(CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : legacy.name, forKey: "defaultMLXModel")
        let models = LanguageModelStore(swiftDataService: data, preferences: preferences)
        try await models.loadModels()
        precondition(!models.models.isEmpty && models.models.allSatisfy { $0.modelProvider == .mlx })
        let savedModels = try data.fetchModels()
        let savedConversations = try data.fetchConversations()
        precondition(savedModels.contains { $0.name == legacy.name && $0.modelProvider == .ollama })
        precondition(savedConversations.contains { $0.id == oldConversation.id })
        models.setModel(model: legacy)
        let selected = models.selectedModel!
        precondition(selected.modelProvider == .mlx && models.supportsImages == selected.imageSupport)
        precondition(preferences.string(forKey: "defaultMLXModel") == selected.name)
        print("PASS: MLX discovery/selection preserves old provider rows and conversations")

        var spoken: [String] = []
        let automatic = AutomaticResponseSpeech(preferences: preferences,
            speak: { spoken.append($0) }, stop: {})
        let chat = ConversationStore(swiftDataService: data, preferences: preferences, automaticSpeech: automatic)
        try await chat.selectConversation(oldConversation)
        chat.sendPrompt(userPrompt: "Reply only with the exact passphrase I gave you.", model: selected)
        try await waitForCompletion(chat)
        let messages = try data.fetchMessages(oldConversation.id)
        precondition(messages.count == 4 && messages.last!.content.contains("REDWOOD-472") && messages.last!.done)
        precondition(messages.last!.displayContent.contains("REDWOOD-472") && !messages.last!.displayContent.contains("</think>"))
        precondition(oldConversation.model?.modelProvider == .mlx)
        precondition(spoken.isEmpty)
        print("PASS: real ConversationStore continues old history through MLX and saves the complete final answer")

        preferences.set(true, forKey: AutomaticResponseSpeech.enabledKey)
        chat.selectedConversation = nil
        chat.messages = []
        chat.sendPrompt(userPrompt: "Reply only with OK.", model: selected, systemPrompt: "You are a helpful assistant.")
        try await waitForCompletion(chat)
        try await Task.sleep(for: .milliseconds(20))
        let fresh = chat.selectedConversation!
        let freshMessages = try data.fetchMessages(fresh.id)
        precondition(fresh.id != oldConversation.id && freshMessages.count == 3)
        precondition(freshMessages.first!.role == "system" && freshMessages.last!.content.uppercased().contains("OK"))
        precondition(spoken == [freshMessages.last!.displayContent.trimmingCharacters(in: .whitespacesAndNewlines)])
        precondition(!spoken[0].contains("</think>"))
        print("PASS: new conversations save system/user/assistant messages without duplicates")
        try await chat.selectConversation(fresh)
        try await Task.sleep(for: .milliseconds(20))
        precondition(spoken.count == 1)
        print("PASS: enabled auto speech receives exactly the completed answer; opening history does not replay it (mock speech output)")

        chat.sendPrompt(userPrompt: "Write a detailed story of at least 2000 words about exploring the ocean.", model: selected)
        let start = Date()
        while chat.messages.last?.content.isEmpty != false && Date().timeIntervalSince(start) < 30 {
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(chat.conversationState == .loading && chat.messages.last?.content.isEmpty == false)
        let cancelled = chat.messages.last!
        chat.stopGenerate()
        let preserved = cancelled.content
        chat.sendPrompt(userPrompt: "Reply only with OK.", model: selected)
        try await waitForCompletion(chat)
        try await Task.sleep(for: .milliseconds(20))
        precondition(chat.messages.last!.content.uppercased().contains("OK") && chat.messages.last!.done)
        precondition(cancelled.content == preserved && cancelled.done)
        if AssistantMessageContent(cancelled.content).isThinking { precondition(cancelled.displayContent.isEmpty) }
        precondition(spoken.count == 2 && spoken.last == chat.messages.last!.displayContent.trimmingCharacters(in: .whitespacesAndNewlines))
        print("PASS: stop/restart isolates response updates and preserves the cancelled answer")
        print("PASS: cancelled generation stays silent; replacement speaks once (mock speech output)")

        let completion = CompletionInstructionSD(name: "Test", keyboardCharacterStr: "t", instruction: "Reply only with OK. Ignore this text: {{text}}", order: 0, modelTemperature: 0)
        let panel = CompletionsPanelVM(preferences: preferences)
        panel.selectedText = "fixture"
        panel.sendPrompt(completion: completion, model: selected)
        let panelStart = Date()
        while panel.isReady && Date().timeIntervalSince(panelStart) < 30 {
            try await Task.sleep(for: .milliseconds(20))
        }
        let queued = await panel.sentenceQueue.dequeueAll().joined()
        precondition(!panel.isReady && panel.messageResponse.uppercased().contains("OK") && queued == panel.messageResponse)
        print("PASS: shortcut completion uses local MLX and queues the complete response")
        precondition(spoken.count == 2)

        preferences.set(true, forKey: "openAIEnabled")
        preferences.set("fixture-model\nfixture-model\nsecond-model", forKey: "openAIModels")
        try await models.loadModels()
        let remote = models.models.first { $0.name == "openai:fixture-model" }!
        precondition(models.models.filter { $0.modelProvider == .openai }.count == 2)
        precondition(remote.requestModelID == "fixture-model" && remote.displayName == "OpenAI · fixture-model")
        models.setModel(model: remote)
        precondition(models.selectedModel === remote && !models.supportsImages)
        try await models.loadModels()
        precondition(models.selectedModel?.modelProvider == .openai)
        // Remote discovery must work when an external HF volume is missing, without a picker.
        preferences.set("/private/tmp/nonexistent-HF-\(UUID())", forKey: SpeechCacheAccess.homeKey)
        try await models.loadModels()
        precondition(models.models.count == 2 && models.models.allSatisfy { $0.modelProvider == .openai })
        print("PASS: local/remote model lists, namespaced IDs and OpenAI selection with missing HF_HOME")

        preferences.set(false, forKey: "openAIEnabled")
        chat.selectedConversation = nil; chat.messages = []
        chat.sendPrompt(userPrompt: "Remote route fixture", model: remote)
        let remoteStart = Date()
        while chat.conversationState == .loading && Date().timeIntervalSince(remoteStart) < 5 {
            try await Task.sleep(for: .milliseconds(20))
        }
        if case .error(let error) = chat.conversationState { precondition(error.contains("OpenAI is disabled")) }
        else { preconditionFailure("Disabled OpenAI must not fall back to MLX") }
        precondition(spoken.count == 2)
        let savedRemote = chat.selectedConversation!
        preferences.set(CommandLine.arguments[1], forKey: SpeechCacheAccess.homeKey)
        try await models.loadModels()
        precondition(models.selectedModel?.modelProvider == .mlx)
        let remoteHistory = try data.fetchConversations()
        precondition(remoteHistory.contains { $0.id == savedRemote.id && $0.model?.modelProvider == .openai })
        print("PASS: provider routing rejects disabled OpenAI without MLX fallback; refresh preserves remote conversation")
    }

    static func waitForCompletion(_ chat: ConversationStore) async throws {
        let start = Date()
        while chat.conversationState == .loading && Date().timeIntervalSince(start) < 60 {
            try await Task.sleep(for: .milliseconds(20))
        }
        if case .error(let error) = chat.conversationState { fatalError(error) }
        precondition(chat.conversationState == .completed)
    }
}
