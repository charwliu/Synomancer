import SwiftUI

struct AssistantResponseView: View {
    let content: String
    var streaming = false
    @State private var expanded = false

    var body: some View {
        let parsed = AssistantMessageContent(content, streaming: streaming)
        VStack(alignment: .leading, spacing: 12) {
            if let reasoning = parsed.reasoning, !reasoning.isEmpty || (streaming && parsed.isThinking) {
                DisclosureGroup(isExpanded: $expanded) {
                    NativeMarkdownView(content: reasoning)
                        .foregroundStyle(.secondary)
                        .padding(.top, 6)
                } label: {
                    if streaming && parsed.isThinking {
                        Label("思考中...", systemImage: "brain")
                    } else {
                        Label("思考", systemImage: "brain")
                    }
                }
                .font(.callout)
                .foregroundStyle(.secondary)
            }
            if !parsed.answer.isEmpty {
                NativeMarkdownView(content: parsed.answer)
            }
        }
    }
}
