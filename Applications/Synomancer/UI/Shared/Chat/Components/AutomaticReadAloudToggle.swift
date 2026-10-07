import SwiftUI

struct AutomaticReadAloudToggle: View {
    @AppStorage(AutomaticResponseSpeech.enabledKey) private var enabled = false

    var body: some View {
        Toggle(isOn: $enabled) {
            Label("自动朗读", systemImage: "speaker.wave.2")
        }
        .toggleStyle(.button)
        .help("回答完成后自动朗读正式答案")
    }
}
