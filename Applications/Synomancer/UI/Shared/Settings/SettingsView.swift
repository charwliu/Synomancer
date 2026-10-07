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
    @Binding var openAIKey: String
    var save: () -> ()
    var refreshModels: () -> ()
    var deleteAll: () -> ()
    var localModels: [LanguageModelSD]
    var voices: [AVSpeechSynthesisVoice]
    
    @AppStorage("speechBackend") private var speechBackend = "system"
    @AppStorage(LocalQwenMTP.enabledKey) private var mtpEnabled = true
    @AppStorage(LocalQwenMTP.temperatureKey) private var localTemperature = 0.6
    @AppStorage("appleIntelligenceEffectEnabled") private var intelligenceEffectEnabled = true
    @State private var deleteConversationsDialog = false
    
    var body: some View {
        VStack {
            ZStack {
                HStack {
                    Button {
                        presentationMode.wrappedValue.dismiss()
                    } label: {
                        Text("Cancel")
                            .font(.system(size: 16))
                            .foregroundStyle(Color(.label))
                    }
                    
                    
                    Spacer()
                    
                    Button(action: save) {
                        Text("Save")
                            .font(.system(size: 16))
                            .foregroundStyle(Color(.label))
                    }
                }
                
                HStack {
                    Spacer()
                    Text("Settings")
                        .font(.system(size: 16))
                        .fontWeight(.medium)
                        .foregroundStyle(Color(.label))
                    Spacer()
                }
            }
            .padding()
            
            Form {
                Section("视觉效果") {
                    Toggle("生成时显示彩色流光", isOn: $intelligenceEffectEnabled)
                    Text("聊天生成和图像处理时，在内容边缘显示流光。遵循系统减少动态效果设置。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HuggingFaceSettings()
                LocalSpeechSettings()
                LocalTTSSettings()
                OpenAISettings(refreshModels: refreshModels, apiKey: $openAIKey)
                LocalChatAPISettings()
                Section(header: Text("Chat Models").font(.headline)) {
                    
                    Button("Refresh Models", action: refreshModels)
                    Text("Models are loaded directly from HF_HOME on this device.")
                        .font(.caption).foregroundStyle(.secondary)

                    Toggle("Use Local Qwen MTP", isOn: $mtpEnabled)
                    HStack {
                        Text("Local Chat Temperature")
                        Slider(value: $localTemperature, in: 0...2, step: 0.1)
                        Text(localTemperature, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                    }
                    Button("Use Temperature 0 for MTP") { localTemperature = 0 }
                    Text("MTP uses a compatible Qwen model’s local draft weights at temperature 0. Other models and temperatures use ordinary generation. Completion shortcuts and API requests keep their own temperature.")
                        .font(.caption).foregroundStyle(.secondary)

                    Text("MTP can produce a different continuation from ordinary greedy generation. Turn it off when exact reproducibility matters.")
                        .font(.caption).foregroundStyle(.secondary)

                    VStack(alignment: .leading) {
                        Text("System prompt")
                        TextEditor(text: $systemPrompt)
                            .font(.system(size: 13))
                            .cornerRadius(4)
                            .multilineTextAlignment(.leading)
                            .frame(minHeight: 100)
                    }
                    
                    if localModels.isEmpty {
                        Text("No chat model found. Choose HF_HOME or configure OpenAI.")
                            .foregroundStyle(.secondary)
                    } else {
                        Picker(selection: $defaultModel) {
                            ForEach(localModels, id: \.self) { model in
                                Text(model.displayName).tag(model.name)
                            }
                        } label: {
                            Label("Default Model", systemImage: "cpu")
                        }
                    }

                    Section(header: Text("APP").font(.headline).padding(.top, 20)) {
                        
#if os(iOS)
                        Toggle(isOn: $vibrations, label: {
                            Label("Vibrations", systemImage: "water.waves")
                                .foregroundStyle(Color.label)
                        })
#endif
                    }
                    
                    
                    Picker(selection: $colorScheme) {
                        ForEach(AppColorScheme.allCases, id:\.self) { scheme in
                            Text(scheme.toString).tag(scheme.id)
                        }
                    } label: {
                        Label("Appearance", systemImage: "sun.max")
                            .foregroundStyle(Color.label)
                    }
                    
                    Picker(selection: $voiceIdentifier) {
                        ForEach(voices, id:\.self.identifier) { voice in
                            Text(voice.prettyName).tag(voice.identifier)
                        }
                    } label: {
                        Label("Voice", systemImage: "waveform")
                            .foregroundStyle(Color.label)
                        
#if os(macOS)
                        Text("Download voices by going to Settings > Accessibility > Spoken Content > System Voice > Manage Voices.")
#else
                        Text("Download voices by going to Settings > Accessibility > Spoken Content > Voices.")
#endif
                        
                        Button(action: {
#if os(macOS)
                            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpeakableItems") {
                                NSWorkspace.shared.open(url)
                            }
#else
                            let url = URL(string: "App-Prefs:root=General&path=ACCESSIBILITY")
                            if let url = url, UIApplication.shared.canOpenURL(url) {
                                UIApplication.shared.open(url, options: [:], completionHandler: nil)
                            }
#endif
                            
                        }) {
                            
                            Text("Open Settings")
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    .showIf(speechBackend == "system")

                    TextField("Initials", text: $appUserInitials)
                        .disableAutocorrection(true)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
#if os(iOS)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
#endif
                    
                    Button(action: {deleteConversationsDialog.toggle()}) {
                        HStack {
                            Spacer()
                            
                            Text("Clear All Data")
                                .foregroundStyle(Color(.systemRed))
                                .padding(.vertical, 6)
                            
                            Spacer()
                        }
                    }
                }
            }
            .formStyle(.grouped)
        }
        .preferredColorScheme(colorScheme.toiOSFormat)
        .confirmationDialog("Delete All Conversations?", isPresented: $deleteConversationsDialog) {
            Button("Delete", role: .destructive) { deleteAll() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Delete All Conversations?")
        }
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
        openAIKey: .constant(""),
        save: {},
        refreshModels: {},
        deleteAll: {},
        localModels: LanguageModelSD.sample,
        voices: []
    )
}
