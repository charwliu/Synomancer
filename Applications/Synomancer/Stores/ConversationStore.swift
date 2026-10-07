//
//  ConversationStore.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation
import SwiftData
import SwiftUI

@Observable @MainActor
final class ConversationStore {
    static let shared = ConversationStore(swiftDataService: SwiftDataService.shared)
    
    private var swiftDataService: SwiftDataService
    private let preferences: UserDefaults
    private var generation: Task<Void, Never>?
    private var requestID: UUID?
    private var activeMessage: MessageSD?
    private var receivedTokens = 0
    var automaticSpeech: AutomaticResponseSpeech?

    @MainActor var conversationState: ConversationState = .completed
    @MainActor var conversations: [ConversationSD] = []
    @MainActor var selectedConversation: ConversationSD?
    @MainActor var messages: [MessageSD] = []
    
    init(swiftDataService: SwiftDataService, preferences: UserDefaults = .standard,
         automaticSpeech: AutomaticResponseSpeech? = nil) {
        self.swiftDataService = swiftDataService
        self.preferences = preferences
        self.automaticSpeech = automaticSpeech
    }
    
    func loadConversations() async throws {
        conversations = try swiftDataService.fetchConversations()
    }
    
    func deleteAllConversations() {
        stopGenerate()
        Task { [weak self] in
            guard let self else { return }
            DispatchQueue.main.async { [weak self] in
                self?.messages = []
                self?.selectedConversation = nil
            }
            try? swiftDataService.deleteConversations()
            try? swiftDataService.deleteMessages()
            try? await loadConversations()
        }
    }
    
    func deleteDailyConversations(_ date: Date) {
        stopGenerate()
        Task {
            DispatchQueue.main.async { [self] in
                selectedConversation = nil
                messages = []
            }
            try? swiftDataService.deleteConversations()
            try? await loadConversations()
        }
    }
    
    
    func create(_ conversation: ConversationSD) async throws {
        try swiftDataService.createConversation(conversation)
    }
    
    func reloadConversation(_ conversation: ConversationSD) async throws {
        let (messages, selectedConversation) = try (
            swiftDataService.fetchMessages(conversation.id),
            swiftDataService.getConversation(conversation.id)
        )
        
        self.messages = messages
        self.selectedConversation = selectedConversation
    }
    
    func selectConversation(_ conversation: ConversationSD) async throws {
        stopGenerate()
        try await reloadConversation(conversation)
    }
    
    func delete(_ conversation: ConversationSD) async throws {
        if selectedConversation?.id == conversation.id { stopGenerate() }
        try swiftDataService.deleteConversation(conversation)
        let fetchedConversations = try swiftDataService.fetchConversations()
        DispatchQueue.main.async {
            self.selectedConversation = nil
            self.conversations = fetchedConversations
        }
    }
    
    @MainActor func stopGenerate() {
        automaticSpeech?.cancel()
        guard requestID != nil else { return }
        generation?.cancel()
        requestID = nil
        generation = nil
        if let message = activeMessage {
            message.done = true
            try? swiftDataService.updateMessage(message)
        }
        activeMessage = nil
        conversationState = .completed
    }

    @MainActor
    func sendPrompt(userPrompt: String, model: LanguageModelSD, image: Image? = nil, systemPrompt: String = "", trimmingMessageId: String? = nil) {
        guard !userPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        stopGenerate()
        let conversation = selectedConversation ?? ConversationSD(name: userPrompt)
        let isNew = selectedConversation == nil
        conversation.updatedAt = .now
        conversation.model = model
        if let trimmingMessageId {
            conversation.messages = conversation.messages.sorted { $0.createdAt < $1.createdAt }
                .prefix(while: { $0.id.uuidString != trimmingMessageId })
        }
        if !systemPrompt.isEmpty && conversation.messages.isEmpty {
            let system = MessageSD(content: systemPrompt, role: "system")
            system.conversation = conversation
            if !conversation.messages.contains(where: { $0.id == system.id }) { conversation.messages.append(system) }
        }
        let user = MessageSD(content: userPrompt, role: "user", image: image?.render()?.compressImageData())
        user.conversation = conversation
        if !conversation.messages.contains(where: { $0.id == user.id }) { conversation.messages.append(user) }
        let history = conversation.messages.sorted { $0.createdAt < $1.createdAt }
            .map { LocalChatMessage(role: $0.role, content: $0.content, image: $0.image) }
        let assistant = MessageSD(content: "", role: "assistant")
        assistant.conversation = conversation
        let request = UUID()
        let modelID = model.requestModelID
        let provider = model.modelProvider
        requestID = request
        activeMessage = assistant
        receivedTokens = 0
        conversationState = .loading
        generation = Task {
            do {
                if isNew { try swiftDataService.createConversation(conversation) }
                try swiftDataService.updateConversation(conversation)
                try swiftDataService.createMessage(user)
                try swiftDataService.createMessage(assistant)
                // Update immediately on the owning actor; deferred UI assignments could
                // otherwise overwrite a user's newer conversation selection.
                messages = try swiftDataService.fetchMessages(conversation.id)
                selectedConversation = conversation
                conversations = try swiftDataService.fetchConversations()
                let result = try await ChatGenerationService.generate(modelID: modelID, provider: provider, messages: history, preferences: preferences) { [self] text, count in
                    Task { @MainActor in
                        guard self.requestID == request, count > self.receivedTokens else { return }
                        self.receivedTokens = count
                        self.activeMessage?.content = text
                    }
                }
                try Task.checkCancellation()
                guard requestID == request else { return }
                assistant.content = result.text
                assistant.done = true
                assistant.error = false
                try swiftDataService.updateMessage(assistant)
                requestID = nil
                activeMessage = nil
                generation = nil
                conversationState = .completed
                automaticSpeech?.responseCompleted(assistant.content)
            } catch {
                guard requestID == request else { return }
                assistant.error = !(error is CancellationError)
                assistant.done = true
                try? swiftDataService.updateMessage(assistant)
                requestID = nil
                activeMessage = nil
                generation = nil
                if error is CancellationError { conversationState = .completed }
                else { conversationState = .error(message: (error as? LocalizedError)?.errorDescription ?? String(describing: error)) }
            }
        }
    }
}
