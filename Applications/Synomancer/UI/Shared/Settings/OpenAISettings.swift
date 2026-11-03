import SwiftUI

struct OpenAISettings: View {
    var refreshModels: () -> Void
    @AppStorage("openAIEnabled") private var enabled = false
    @AppStorage("openAIModels") private var modelIDs = ""
    @AppStorage("openAIUseResponses") private var useResponses = true
    @AppStorage("openAISendTemperature") private var sendTemperature = false
    @Binding var apiKey: String
    @State private var status = ""
    @State private var fetching = false
    @State private var availableIDs: [String] = []
    @State private var showModels = false

    var body: some View {
        Section("OpenAI") {
            Toggle("Enable OpenAI models", isOn: $enabled)
            Text("Selecting an OpenAI model sends conversation text to OpenAI. Local MLX and Gemini remain available.")
                .font(.caption).foregroundStyle(.secondary)
            Link("Get an OpenAI API key", destination: URL(string: "https://platform.openai.com/api-keys")!)
            SecureField("OpenAI API Key", text: $apiKey).autocorrectionDisabled()
            Toggle("Use Responses API", isOn: $useResponses)
            Text("Responses is the default. Turn it off for models that require Chat Completions.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Send temperature parameter", isOn: $sendTemperature)
            Text("Keep temperature disabled for models that do not support it.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Model IDs (one per line)")
            TextField("Model IDs (one per line)", text: $modelIDs, axis: .vertical)
                .labelsHidden().font(.system(.body, design: .monospaced))
                .lineLimit(1...8).autocorrectionDisabled()
            HStack {
                Button("Apply OpenAI Settings") { apply() }
                Button(fetching ? "Loading…" : "Choose API Models…") { fetch() }
                    .disabled(!enabled || fetching)
            }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
            Text("The API key is saved separately in Keychain. Requests use api.openai.com. Choose text models available to your API account and selected API mode.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .sheet(isPresented: $showModels) {
            VStack(alignment: .leading, spacing: 12) {
                Text("OpenAI Models").font(.headline)
                Text("Select text generation models to add to the chat menu.").font(.caption)
                ScrollView {
                    LazyVStack(alignment: .leading) {
                        ForEach(availableIDs, id: \.self) { id in
                            Toggle(id, isOn: Binding(get: { selectedIDs.contains(id) }, set: { selected in
                                var ids = selectedIDs
                                if selected { ids.insert(id) } else { ids.remove(id) }
                                modelIDs = ids.sorted().joined(separator: "\n")
                            }))
                        }
                    }
                }
                Button("Done") { showModels = false; refreshModels() }
            }.padding().frame(minWidth: 400, minHeight: 400)
        }
    }

    private var selectedIDs: Set<String> {
        Set(modelIDs.components(separatedBy: .newlines).map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty })
    }

    private func apply() {
        do {
            try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: OpenAIConfiguration.keychainAccount)
            if enabled { _ = try OpenAIConfiguration.load() }
            status = "Settings applied."
            refreshModels()
        } catch { status = error.localizedDescription }
    }

    private func fetch() {
        fetching = true
        Task {
            defer { fetching = false }
            do {
                try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: OpenAIConfiguration.keychainAccount)
                availableIDs = try await OpenAIChatService.shared.models(configuration: OpenAIConfiguration.load())
                showModels = true
                status = ""
            } catch { status = error.localizedDescription }
        }
    }
}
