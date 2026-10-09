import Foundation

enum ChatGenerationService {
    @MainActor static func generate(modelID: String, provider: ModelProvider, messages: [LocalChatMessage],
                                   preferences: UserDefaults = .standard, temperature: Float? = nil,
                                   onUpdate: @escaping @Sendable (String, Int) -> Void) async throws -> LocalChatResult {
        switch provider {
        case .mlx:
            let cache = try await SpeechCacheAccess.authorizedConfiguration(in: preferences)
            return try await LocalChatService.shared.generate(modelID: modelID, messages: messages, cache: cache,
                temperature: temperature ?? Float(preferences.object(forKey: LocalQwenMTP.temperatureKey) as? Double ?? 0.6),
                useMTP: preferences.object(forKey: LocalQwenMTP.enabledKey) as? Bool ?? true, frameReasoning: true, onUpdate: onUpdate)
        case .gemini:
            let configuration = try GeminiConfiguration.load(in: preferences)
            return try await OpenAIChatService.shared.generate(modelID: modelID, messages: messages,
                configuration: configuration, temperature: temperature ?? 0.6, onUpdate: onUpdate)
        case .ollama, .openai:
            throw OpenAIError.message("This saved model is no longer supported. Select a local MLX or Gemini model.")
        }
    }
}
