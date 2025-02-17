//
//  Voice.swift
//  Synomancer
//
//  Created by Wei LIU on 1/31/25.
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
