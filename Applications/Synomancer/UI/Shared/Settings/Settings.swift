//
//  Settings.swift
//  Synomancer
//
//  Created by Wei LIU on 1/31/25.
//


import SwiftUI
import Combine

struct Settings: View {
    var languageModelStore = LanguageModelStore.shared
    var conversationStore = ConversationStore.shared
    var swiftDataService = SwiftDataService.shared

    @AppStorage("systemPrompt") private var systemPrompt: String = ""
    @AppStorage("vibrations") private var vibrations: Bool = true
    @AppStorage("colorScheme") private var colorScheme = AppColorScheme.system
    @AppStorage("defaultMLXModel") private var defaultMLXModel: String = ""
    @AppStorage("appUserInitials") private var appUserInitials: String = ""
    @AppStorage("voiceIdentifier") private var voiceIdentifier: String = ""

    @StateObject private var speechSynthesiser = SpeechSynthesizer.shared

    @Environment(\.presentationMode) var presentationMode

    private let timer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    @State private var cancellable: AnyCancellable?
    @State private var geminiKey = ""
    @State private var originalGeminiKey: String?

    private func save() {
        do {
            // A failed Keychain read must not turn an untouched empty field into deletion.
            if geminiKey != (originalGeminiKey ?? "") {
                try ChatSecrets.save(geminiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: GeminiConfiguration.keychainAccount)
            }
            if UserDefaults.standard.bool(forKey: "geminiEnabled") { _ = try GeminiConfiguration.load() }
        } catch {
            AppStore.shared.uiLog(message: error.localizedDescription, status: .error)
            return
        }
        Task {
            await Haptics.shared.mediumTap()
            do { try await languageModelStore.loadModels() }
            catch { AppStore.shared.uiLog(message: error.localizedDescription, status: .error) }
        }
        presentationMode.wrappedValue.dismiss()
    }

    private func refreshModels() {
        Task {
            do { try await languageModelStore.loadModels() }
            catch { AppStore.shared.uiLog(message: error.localizedDescription, status: .error) }
        }
    }

    private func deleteAll() {
        Task {
            conversationStore.deleteAllConversations()
            try? await languageModelStore.deleteAllModels()
        }
    }

    var body: some View {
        SettingsView(
            systemPrompt: $systemPrompt,
            vibrations: $vibrations,
            colorScheme: $colorScheme,
            defaultModel: $defaultMLXModel,
            appUserInitials: $appUserInitials,
            voiceIdentifier: $voiceIdentifier,
            geminiKey: $geminiKey,
            save: save,
            refreshModels: refreshModels,
            deleteAll: deleteAll,
            localModels: languageModelStore.models,
            voices: speechSynthesiser.voices
        )
        .frame(maxWidth: 700)
        .task {
            do {
                let stored = try ChatSecrets.read(GeminiConfiguration.keychainAccount)
                geminiKey = stored
                originalGeminiKey = stored
            }
            catch { AppStore.shared.uiLog(message: error.localizedDescription, status: .error) }
        }
        #if os(visionOS)
        .frame(minWidth: 600, minHeight: 800)
        #endif
        .onChange(of: defaultMLXModel) { _, modelName in
            languageModelStore.setModel(modelName: modelName)
        }
        .onAppear {
            /// refresh voices in the background
            cancellable = timer.sink { _ in
                speechSynthesiser.fetchVoices()
            }
        }
        .onDisappear {
            cancellable?.cancel()
        }
    }
}

#Preview {
    Settings()
}
