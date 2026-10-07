import SwiftUI

struct ReadingAloudToolbarView: View {
    var isGenerating = false
    var generationStatus = "Generating Local Speech…"
    var onStopTap: () -> Void

    private var status: String {
        isGenerating ? generationStatus : String(localized: "Reading Aloud")
    }

    var body: some View {
        HStack(spacing: 8) {
            Label(status, systemImage: "speaker.wave.2")
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: 220)
                .help(status)
            Button(action: onStopTap) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)
                    .frame(width: 14, height: 14)
                    .padding(5)
            }
            .buttonStyle(GrowingButton())
            .help("Stop Reading Aloud")
            .accessibilityLabel("Stop Reading Aloud")
        }
    }
}

#Preview {
    ReadingAloudToolbarView(onStopTap: {})
        .padding()
}
