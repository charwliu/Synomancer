//
//  Voice.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



import SwiftUI

struct Voice: View {
    @State var languageModelStore: LanguageModelStore
    @State var conversationStore: ConversationStore
    @State var appStore: AppStore
    
    var body: some View {
        VoiceView()
    }
}
