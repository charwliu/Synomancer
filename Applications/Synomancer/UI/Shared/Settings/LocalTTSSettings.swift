import SwiftUI

struct LocalTTSSettings: View {
    @AppStorage("speechBackend") private var backend = "system"
    @AppStorage("localTTSSpeaker") private var speaker = "auto"
    @AppStorage("localTTSLanguage") private var language = "auto"
    @AppStorage(AutomaticResponseSpeech.enabledKey) private var automaticReadAloud = false

    var body: some View {
        Section("Read Aloud") {
            Toggle("自动朗读", isOn: $automaticReadAloud)
            Text("回答完成后自动朗读正式答案")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Speech Model", selection: $backend) {
                Text("System Voice").tag("system")
                Text("Local Qwen3-TTS 1.7B").tag("local")
            }
            if backend == "local" {
                Picker("Language", selection: $language) {
                    Text("Automatic (Chinese / English)").tag("auto")
                    Text("Chinese").tag("chinese")
                    Text("English").tag("english")
                }
                Picker("Speaker", selection: $speaker) {
                    Text("Automatic").tag("auto")
                    ForEach(["vivian", "serena", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"], id: \.self) {
                        Text($0).tag($0)
                    }
                }
                Text("Uses CustomVoice model and speech-tokenizer weights from the shared HF_HOME cache. No automatic model download.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
