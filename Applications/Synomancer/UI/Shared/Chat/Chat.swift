//
//  Chat.swift
//  Synomancer
//
//  Created by Wei LIU on 2/3/25.
//


import SwiftUI

struct Chat: View, Sendable {
    @State private var languageModelStore: LanguageModelStore
    @State private var conversationStore: ConversationStore
    @State private var appStore: AppStore
    @AppStorage("systemPrompt") private var systemPrompt: String = ""
    @AppStorage("appUserInitials") private var userInitials: String = ""
    @AppStorage("defaultMLXModel") private var defaultMLXModel: String = ""
    @State var showMenu = false
    
    init(languageModelStore: LanguageModelStore, conversationStore: ConversationStore, appStore: AppStore) {
        _languageModelStore = State(initialValue: languageModelStore)
        _conversationStore = State(initialValue: conversationStore)
        _appStore = State(initialValue: appStore)
    }
    
    func toggleMenu() {
        withAnimation(.spring) {
            showMenu.toggle()
        }
        Task {
            await Haptics.shared.mediumTap()
        }
    }
    
    @MainActor
    func updateSelectedModel() {
        if languageModelStore.selectedModel == nil {
            if defaultMLXModel != "" {
                languageModelStore.setModel(modelName: defaultMLXModel)
            } else {
                languageModelStore.setModel(model: languageModelStore.models.first)
            }
        }
    }
    
    @MainActor
    func sendMessage(prompt: String, model: LanguageModelSD, image: Image?, trimmingMessageId: String?) {
        conversationStore.sendPrompt(
            userPrompt: prompt,
            model: model,
            image: image,
            systemPrompt: systemPrompt,
            trimmingMessageId: trimmingMessageId
        )
    }
    
    func onConversationTap(_ conversation: ConversationSD) {
        Task {
            do {
                try await conversationStore.selectConversation(conversation)
                languageModelStore.setModel(model: conversation.model)
                await Haptics.shared.mediumTap()
            } catch {
                AppStore.shared.uiLog(message: error.localizedDescription, status: .error)
            }
        }
        withAnimation {
            showMenu.toggle()
        }
    }
    
    @MainActor func onStopGenerateTap() {
        conversationStore.stopGenerate()
        Task {
           await Haptics.shared.mediumTap()
        }
    }
    
    func onConversationDelete(_ conversation: ConversationSD) {
        Task {
            await Haptics.shared.mediumTap()
            try? await conversationStore.delete(conversation)
        }
    }
    
    func newConversation() {
        conversationStore.stopGenerate()
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.3)) {
                self.conversationStore.selectedConversation = nil
            }
        }
        
        Task {
            await Haptics.shared.mediumTap()
            try? await languageModelStore.loadModels()
        }
        
#if os(iOS)
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
#endif
    }
    
    func copyChat(_ json: Bool) {
        Task {
            let messages = ConversationStore.shared.messages
            
            if messages.count == 0 {
                return
            }
            
            if json {
                let jsonArray = messages.map { message in
                    return [
                        "role": message.role,
                        "content": message.content
                    ]
                }
                let jsonEncoder = JSONEncoder()
                jsonEncoder.outputFormatting = [.withoutEscapingSlashes]

                if let jsonData = try? jsonEncoder.encode(jsonArray),
                   let jsonString = String(data: jsonData, encoding: .utf8) {
                    Clipboard.shared.setString(jsonString)
                }
            } else {
                let body = messages.map{"\($0.role.capitalized): \($0.content)"}.joined(separator: "\n\n")
                Clipboard.shared.setString(body)
            }
        }
    }
    
    var body: some View {
        Group {
#if os(macOS) || os(visionOS)
            ChatView(
                selectedConversation: conversationStore.selectedConversation,
                conversations: conversationStore.conversations,
                messages: conversationStore.messages,
                modelsList: languageModelStore.models,
                onMenuTap: toggleMenu,
                onNewConversationTap: newConversation,
                onSendMessageTap: sendMessage,
                onConversationTap:onConversationTap,
                conversationState: conversationStore.conversationState,
                onStopGenerateTap: onStopGenerateTap,
                reachable: appStore.isReachable,
                modelSupportsImages: languageModelStore.supportsImages,
                selectedModel: languageModelStore.selectedModel,
                onSelectModel: languageModelStore.setModel,
                onConversationDelete: onConversationDelete,
                onDeleteDailyConversations: conversationStore.deleteDailyConversations,
                userInitials: userInitials,
                copyChat: copyChat
            )
#else
            SideBarStack(sidebarWidth: 300,showSidebar: $showMenu, sidebar: {
                SidebarView(
                    selectedConversation: conversationStore.selectedConversation,
                    conversations: conversationStore.conversations,
                    onConversationTap: onConversationTap,
                    onConversationDelete: onConversationDelete,
                    onDeleteDailyConversations: conversationStore.deleteDailyConversations
                )
            }) {
                ChatView(
                    conversation: conversationStore.selectedConversation,
                    messages: conversationStore.messages,
                    modelsList: languageModelStore.models,
                    selectedModel: languageModelStore.selectedModel,
                    onSelectModel: languageModelStore.setModel,
                    onMenuTap: toggleMenu,
                    onNewConversationTap: newConversation,
                    onSendMessageTap: sendMessage,
                    conversationState: conversationStore.conversationState,
                    onStopGenerateTap: onStopGenerateTap,
                    reachable: appStore.isReachable,
                    modelSupportsImages: languageModelStore.supportsImages,
                    userInitials: userInitials
                )
            }
#endif
        }
        .safeAreaInset(edge: .bottom) {
            if let progress = conversationStore.loadingProgress {
                HStack(spacing: 10) {
                    if let fraction = progress.fraction {
                        ProgressView(value: fraction).frame(width: 120)
                        Text(fraction, format: .percent.precision(.fractionLength(0)))
                            .monospacedDigit()
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    Text(progress.label)
                    Spacer()
                }
                .font(.callout)
                .padding(.horizontal).padding(.vertical, 8)
                .background(.regularMaterial)
            }
        }
        .appleIntelligenceEffect(active: conversationStore.conversationState == .loading)
        .onChange(of: languageModelStore.models, { _, modelsList in
            if languageModelStore.selectedModel == nil {
                updateSelectedModel()
            }
        })
        .onChange(of: conversationStore.selectedConversation, initial: true, { _, newConversation in
            if let conversation = newConversation {
                languageModelStore.setModel(model: conversation.model)
            } else {
                updateSelectedModel()
            }
        })
    }
}
