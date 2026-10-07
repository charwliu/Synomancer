import SwiftUI

struct LocalSpeechSettings: View {
    var body: some View {
        Section("Local Voice Input") {
            Text("Qwen3-ASR 1.7B")
            Text("Uses the shared HF_HOME cache. BF16 decoder weights are quantized to 4bit in memory; cached files stay unchanged.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
