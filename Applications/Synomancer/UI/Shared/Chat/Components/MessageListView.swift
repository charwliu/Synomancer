//
//  MessageListView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//

import SwiftUI

#if os(macOS)
import AppKit
#endif

struct MessageListView: View {
    var messages: [MessageSD]
    var conversationState: ConversationState
    var userInitials: String
    @Binding var editMessage: MessageSD?
    @State private var messageSelected: MessageSD?
    @StateObject private var speechSynthesizer = SpeechSynthesizer.shared
    
    func onEditMessageTap() -> (MessageSD) -> Void {
        return { message in
            editMessage = message
        }
    }
    
    func onReadAloud(_ message: String) {
        Task {
            await speechSynthesizer.speak(text: message)
        }
    }
    
    func stopReadingAloud() {
        Task {
            await speechSynthesizer.stopSpeaking()
        }
    }
    
    var body: some View {
        ZStack(alignment: .top) {
            ScrollViewReader { scrollViewProxy in
                ScrollView {
                    ForEach(messages) { message in
                        let contextMenu = ContextMenu(menuItems: {
                            Button(action: {Clipboard.shared.setString(message.displayContent)}) {
                                Label("Copy", systemImage: "doc.on.doc")
                            }
                            
#if os(iOS) || os(visionOS)
                            Button(action: { messageSelected = message }) {
                                Label("Select Text", systemImage: "selection.pin.in.out")
                            }
                            
                            Button(action: {
                                onReadAloud(message.displayContent)
                            }) {
                                Label("Read Aloud", systemImage: "speaker.wave.3.fill")
                            }
                            .disabled(message.displayContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
#endif
                            
                            if message.role == "user" {
                                Button(action: {
                                    withAnimation { editMessage = message }
                                }) {
                                    Label("Edit", systemImage: "pencil")
                                }
                            }
                            
                            if editMessage?.id == message.id {
                                Button(action: {
                                    withAnimation { editMessage = nil }
                                }) {
                                    Label("Unselect", systemImage: "pencil")
                                }
                            }
                        })
                        
                        ChatMessageView(
                            message: message,
                            showLoader: conversationState == .loading && messages.last == message,
                            userInitials: userInitials,
                            editMessage: $editMessage
                        )
                        .listRowInsets(EdgeInsets())
                        .listRowSeparator(.hidden)
                        .padding(.vertical, 10)
                        .padding(.horizontal, 10)
                        .contentShape(Rectangle())
                        .contextMenu(contextMenu)
                        .runningBorder(animated: message.id == editMessage?.id)
                        .id(message)
                    }
                }
                .onAppear {
                    scrollViewProxy.scrollTo(messages.last, anchor: .bottom)
                }
                .onChange(of: messages) { oldMessages, newMessages in
                    scrollViewProxy.scrollTo(messages.last, anchor: .bottom)
                }
                .onChange(of: messages.last?.content) {
                    scrollViewProxy.scrollTo(messages.last, anchor: .bottom)
                }
#if os(iOS) || os(visionOS)
                .sheet(item: $messageSelected) { message in
                    SelectTextSheet(message: message)
                }
#endif
            }
            
#if os(iOS)
            ReadingAloudView(isGenerating: speechSynthesizer.isGenerating, generationStatus: speechSynthesizer.generationStatus, onStopTap: stopReadingAloud)
                .frame(maxWidth: 400)
                .showIf(speechSynthesizer.isSpeaking)
                .transition(.asymmetric(
                    insertion: AnyTransition.opacity.combined(with: .scale(scale: 0.7, anchor: .top)),
                    removal: AnyTransition.opacity.combined(with: .scale(scale: 0.7, anchor: .top)))
                )
#endif
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if case .error = conversationState {
                ConversationStatusView(state: conversationState)
                    .textSelection(.enabled)
                    .padding()
                    .background(.regularMaterial)
            }
        }
        .alert("Read Aloud", isPresented: Binding(
            get: { speechSynthesizer.errorMessage != nil },
            set: { if !$0 { speechSynthesizer.errorMessage = nil } }
        )) {
            Button("OK") { speechSynthesizer.errorMessage = nil }
        } message: { Text(speechSynthesizer.errorMessage ?? "") }
    }
}

#Preview {
    MessageListView(
        messages: MessageSD.sample,
        conversationState: .loading,
        userInitials: "AM",
        editMessage: .constant(MessageSD.sample[0])
    )
}
