//
//  SynomancerApp.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//

import Defaults
import SwiftUI
import SwiftData

#if os(macOS)
import KeyboardShortcuts
extension KeyboardShortcuts.Name {
    static let togglePanelMode = Self("togglePanelMode1", default: .init(.k, modifiers: [.command, .option]))
}
#endif


@main
struct SynomancerApp: App {
    @State private var conversationViewModel: ConversationView.ViewModel = .init()
    @State private var settingsViewModel: SettingsView.ViewModel = .init()
    
    @Default(.language) var language
    @State private var runner = LLMRunner()
    
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self, CompletionInstructionSD.self, ConversationSD.self, LanguageModelSD.self, MessageSD.self, Conversation.self, Message.self
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

//    var body: some Scene {
//        WindowGroup {
//            ContentView()
//        }
//        .modelContainer(sharedModelContainer)
//        .database(BackgroundDatabase(modelContainer: sharedModelContainer))
//    }
    var body: some Scene {
        WindowGroup {
            ConversationView()
                .environment(conversationViewModel)
                .environment(
                    \.locale, .init(identifier: language.rawValue)
                )
                .environment(runner)
                .frame(minWidth: 900, minHeight: 580)
        }
        .modelContainer(sharedModelContainer)
        
        Settings {
            SettingsView()
                .environment(conversationViewModel)
                .environment(settingsViewModel)
                .environment(
                    \.locale, .init(identifier: language.rawValue)
                )
                .environment(runner)
                .frame(minWidth: 620, minHeight: 380)
        }
        .modelContainer(sharedModelContainer)
    }
}
