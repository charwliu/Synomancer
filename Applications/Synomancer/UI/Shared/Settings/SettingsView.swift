//
//  SettingsView.swift
//  Synomancer
//
//  Created by Wei LIU on 1/31/25.
//


import SwiftUI
import AVFoundation

struct SettingsView: View {
    @Environment(\.presentationMode) var presentationMode
    @Binding var systemPrompt: String
    @Binding var vibrations: Bool
    @Binding var colorScheme: AppColorScheme
    @Binding var defaultModel: String
    @Binding var appUserInitials: String
    @Binding var voiceIdentifier: String
    @Binding var geminiKey: String
    var openAISettings: OpenAISettingsState
    var save: () -> ()
    var refreshModels: () -> ()
    var deleteAll: () -> ()
    var localModels: [LanguageModelSD]
    var voices: [AVSpeechSynthesisVoice]
    
    @AppStorage("speechBackend") private var speechBackend = "system"
    @AppStorage(GGUFMemoryPolicy.expertBudgetKey) private var expertCacheGiB = GGUFMemoryPolicy.defaultExpertGiB
    @AppStorage(LocalQwenMTP.enabledKey) private var mtpEnabled = true
    @AppStorage(LocalQwenMTP.temperatureKey) private var localTemperature = 0.6
    @AppStorage("appleIntelligenceEffectEnabled") private var intelligenceEffectEnabled = true
    @State private var deleteConversationsDialog = false
    
    var saveError: String?
    @State private var selectedTab = SettingsTab.general

    private enum SettingsTab: Hashable {
        case general, models, openAI, gemini, speech, localAPI
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Cancel") { presentationMode.wrappedValue.dismiss() }
                    .frame(width: 70, alignment: .leading)
                Spacer()
                Text("Settings").font(.headline)
                Spacer()
                Button("Save", action: save).frame(width: 70, alignment: .trailing)
            }.padding()

            if let saveError {
                Text(saveError).font(.caption).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal).padding(.bottom, 8)
            }

            TabView(selection: $selectedTab) {
                generalSettings
                    .tabItem { Label("General", systemImage: "gearshape") }.tag(SettingsTab.general)
                modelSettings
                    .tabItem { Label("Models", systemImage: "cpu") }.tag(SettingsTab.models)
                Form { OpenAISettings(refreshModels: refreshModels, state: openAISettings) }
                    .formStyle(.grouped)
                    .tabItem { Label("OpenAI", systemImage: "network") }.tag(SettingsTab.openAI)
                Form { GeminiSettings(refreshModels: refreshModels, apiKey: $geminiKey) }
                    .formStyle(.grouped)
                    .tabItem { Label("Gemini", systemImage: "sparkles") }.tag(SettingsTab.gemini)
                speechSettings
                    .tabItem { Label("Speech", systemImage: "waveform") }.tag(SettingsTab.speech)
                Form { LocalChatAPISettings() }
                    .formStyle(.grouped)
                    .tabItem { Label("Local API", systemImage: "server.rack") }.tag(SettingsTab.localAPI)
            }
        }
        .preferredColorScheme(colorScheme.toiOSFormat)
        .confirmationDialog("Delete All Conversations?", isPresented: $deleteConversationsDialog) {
            Button("Delete", role: .destructive) { deleteAll() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Delete All Conversations?")
        }
    }

    private var generalSettings: some View {
        Form {
            Section("Appearance") {
                Picker("Appearance", selection: $colorScheme) {
                    ForEach(AppColorScheme.allCases, id: \.self) { scheme in
                        Text(scheme.toString).tag(scheme)
                    }
                }
                Toggle("生成时显示彩色流光", isOn: $intelligenceEffectEnabled)
                Text("聊天生成和图像处理时，在内容边缘显示流光。遵循系统减少动态效果设置。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("App") {
                TextField("Initials", text: $appUserInitials).autocorrectionDisabled()
                #if os(iOS)
                Toggle("Vibrations", isOn: $vibrations)
                #endif
            }
            Section("Data") {
                Button("Clear All Data", role: .destructive) { deleteConversationsDialog = true }
            }
        }.formStyle(.grouped)
    }

    private var modelSettings: some View {
        Form {
            HuggingFaceSettings(onDownloadComplete: refreshModels)
            Section("Chat Models") {
                Button("Refresh Models", action: refreshModels)
                Text("Models are loaded directly from HF_HOME on this device.")
                    .font(.caption).foregroundStyle(.secondary)
                if localModels.isEmpty {
                    Text("No chat model found. Choose HF_HOME or configure OpenAI or Gemini.")
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Default Model", selection: $defaultModel) {
                        ForEach(localModels, id: \.self) { model in
                            Text(model.displayName).tag(model.name)
                        }
                    }
                }
                Stepper("Flash-Next 专家缓存：\(min(expertCacheGiB, GGUFMemoryPolicy.maximumExpertGiB)) GiB",
                        value: $expertCacheGiB, in: 0...GGUFMemoryPolicy.maximumExpertGiB)
                Text("下次请求生效。0 表示不保留专家缓存；较大的缓存通常更快。此预算不包含基础权重、上下文、视觉模型和系统文件缓存。PLE 查找表始终按需读取。")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Use Local Qwen MTP", isOn: $mtpEnabled)
                HStack {
                    Text("Local Chat Temperature")
                    Slider(value: $localTemperature, in: 0...2, step: 0.1)
                    Text(localTemperature, format: .number.precision(.fractionLength(1))).monospacedDigit()
                }
                Button("Use Temperature 0 for MTP") { localTemperature = 0 }
                Text("MTP uses a compatible Qwen model’s local draft weights at temperature 0. Other models and temperatures use ordinary generation. Completion shortcuts and API requests keep their own temperature.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("MTP can produce a different continuation from ordinary greedy generation. Turn it off when exact reproducibility matters.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("System prompt") {
                TextEditor(text: $systemPrompt).font(.system(size: 13))
                    .frame(minHeight: 120).multilineTextAlignment(.leading)
            }
        }.formStyle(.grouped)
    }

    private var speechSettings: some View {
        Form {
            LocalSpeechSettings()
            LocalTTSSettings()
            if speechBackend == "system" {
                Section("System Voice") {
                    Picker("Voice", selection: $voiceIdentifier) {
                        ForEach(voices, id: \.identifier) { voice in
                            Text(voice.prettyName).tag(voice.identifier)
                        }
                    }
                    #if os(macOS)
                    Text("Download voices by going to Settings > Accessibility > Spoken Content > System Voice > Manage Voices.")
                        .font(.caption).foregroundStyle(.secondary)
                    #else
                    Text("Download voices by going to Settings > Accessibility > Spoken Content > Voices.")
                        .font(.caption).foregroundStyle(.secondary)
                    #endif
                    Button("Open Settings", action: openVoiceSettings)
                }
            }
        }.formStyle(.grouped)
    }

    private func openVoiceSettings() {
        #if os(macOS)
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpeakableItems") {
            NSWorkspace.shared.open(url)
        }
        #else
        if let url = URL(string: "App-Prefs:root=General&path=ACCESSIBILITY"), UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url, options: [:], completionHandler: nil)
        }
        #endif
    }
}

#Preview {
    SettingsView(
        systemPrompt: .constant("You are an intelligent assistant solving complex problems. You are an intelligent assistant solving complex problems. You are an intelligent assistant solving complex problems."),
        vibrations: .constant(true),
        colorScheme: .constant(.light),
        defaultModel: .constant("mlx-community/Qwen2-VL-2B-Instruct-4bit"),
        appUserInitials: .constant("AM"),
        voiceIdentifier: .constant("sample"),
        geminiKey: .constant(""),
        openAISettings: OpenAISettingsState(),
        save: {},
        refreshModels: {},
        deleteAll: {},
        localModels: LanguageModelSD.sample,
        voices: []
    )
}
