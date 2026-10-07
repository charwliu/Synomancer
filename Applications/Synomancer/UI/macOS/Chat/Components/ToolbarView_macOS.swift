//
//  ToolbarView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



#if os(macOS) || os(visionOS)
import SwiftUI

struct ToolbarView: View {
    @StateObject private var speechSynthesizer = SpeechSynthesizer.shared
    @State private var showingImages = false
    var modelsList: [LanguageModelSD]
    var selectedModel: LanguageModelSD?
    var onSelectModel: @MainActor (_ model: LanguageModelSD?) -> ()
    var onNewConversationTap: () -> ()
    var copyChat: (_ json: Bool) -> ()
    
    var body: some View {
        ModelSelectorView(
            modelsList: modelsList,
            selectedModel: selectedModel,
            onSelectModel: onSelectModel,
            showChevron: false
        )
        .frame(height: 20)
        
        MoreOptionsMenuView(copyChat: copyChat)
        Button { showingImages = true } label: {
            Image(systemName: "photo.badge.plus").font(.system(size: 19)).padding(5)
        }
        .buttonStyle(GrowingButton())
        .help("图像生成与编辑")
        .accessibilityLabel("图像生成与编辑")
        .sheet(isPresented: $showingImages) { ImageStudioView() }
        if speechSynthesizer.isSpeaking {
            ReadingAloudToolbarView(
                isGenerating: speechSynthesizer.isGenerating,
                generationStatus: speechSynthesizer.generationStatus,
                onStopTap: { Task { await speechSynthesizer.stopSpeaking() } }
            )
        }
        AutomaticReadAloudToggle()
        
        Button(action: onNewConversationTap) {
            Image(systemName: "square.and.pencil")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(height: 20)
                .padding(5)
        }
        .buttonStyle(PlainButtonStyle())
        .keyboardShortcut(KeyEquivalent("n"), modifiers: .command)
    }
}

#Preview {
    ToolbarView(
        modelsList: LanguageModelSD.sample,
        selectedModel: LanguageModelSD.sample[0],
        onSelectModel: {_ in},
        onNewConversationTap: {}, 
        copyChat: {_ in}
    )
}

#endif
