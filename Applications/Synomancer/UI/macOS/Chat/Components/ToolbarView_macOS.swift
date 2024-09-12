//
//  ToolbarView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



#if os(macOS) || os(visionOS)
import SwiftUI

struct ToolbarView: View {
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
