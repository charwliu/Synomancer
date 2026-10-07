import SwiftUI

struct RecordingView: View {
    @StateObject var speechRecognizer = SpeechRecognizer()
    @Binding var isRecording: Bool
    var onComplete: @Sendable @MainActor (_ transcription: String) -> Void = { _ in }
    @State private var operation: Task<Void, Never>?

    private var isBusy: Bool {
        speechRecognizer.phase == .preparing || speechRecognizer.phase == .transcribing
    }

    private func toggleRecord() {
        operation = Task { @MainActor in
            if speechRecognizer.phase == .recording {
                let text = await speechRecognizer.finishRecording()
                isRecording = false
                if let text, !text.isEmpty, !Task.isCancelled { onComplete(text) }
            } else {
                isRecording = await speechRecognizer.startRecording(onUpdate: onComplete)
            }
        }
    }

    var body: some View {
        Button(action: toggleRecord) {
            if speechRecognizer.phase == .preparing {
                ProgressView(value: speechRecognizer.preparationProgress)
                    .progressViewStyle(.circular)
                    .controlSize(.small).frame(width: 20, height: 20)
            } else if isBusy {
                ProgressView().controlSize(.small).frame(width: 20, height: 20)
            } else if speechRecognizer.phase == .recording {
                Image(systemName: "stop.circle.fill")
                    .font(.system(size: 20))
                    .foregroundStyle(.blue)
            } else {
                Image(systemName: "waveform")
                    .resizable().scaledToFit().frame(height: 20)
                    .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.plain)
        .disabled(isBusy)
        .help(speechRecognizer.phase == .preparing
              ? "Preparing Qwen3-ASR 1.7B from HF_HOME"
              : "Local voice input — Qwen3-ASR 1.7B")
        .onChange(of: speechRecognizer.phase) { _, phase in
            if phase == .idle { isRecording = false }
        }
        .accessibilityLabel(speechRecognizer.phase == .recording ? "Stop recording" : "Local voice input")
        .alert("Voice Input", isPresented: Binding(
            get: { speechRecognizer.errorMessage != nil },
            set: { if !$0 { speechRecognizer.errorMessage = nil } }
        )) {
            Button("OK") { speechRecognizer.errorMessage = nil }
        } message: {
            Text(speechRecognizer.errorMessage ?? "")
        }
        .onChange(of: isRecording) { _, newValue in
            if !newValue {
                if speechRecognizer.phase == .recording {
                    operation?.cancel()
                    speechRecognizer.cancelRecording()
                } else if speechRecognizer.phase == .transcribing {
                    operation?.cancel()
                }
            }
        }
        .onDisappear {
            operation?.cancel()
            speechRecognizer.cancelRecording()
            isRecording = false
        }
    }
}

#Preview {
    RecordingView(isRecording: .constant(false))
}
