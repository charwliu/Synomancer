import SwiftUI

struct OpenAISettings: View {
    var refreshModels: () -> Void
    @AppStorage("openAIEnabled") private var enabled = false
    @AppStorage("openAIModels") private var modelIDs = ""
    @AppStorage("openAIUseResponses") private var useResponses = true
    @AppStorage("openAISendTemperature") private var sendTemperature = false
    @Bindable var state: OpenAISettingsState
    @State private var status = ""
    @State private var fetching = false
    @State private var availableIDs: [String] = []
    @State private var showModels = false

    var body: some View {
        Section("OpenAI") {
            Toggle("Enable OpenAI models", isOn: $enabled)
            Text("Selecting an OpenAI model sends conversation text and the endpoint key to the configured service.")
                .font(.caption).foregroundStyle(.secondary)
            Link("Get an OpenAI API key", destination: URL(string: "https://platform.openai.com/api-keys")!)
            TextField("Base URL", text: Binding(get: { state.baseURLText }, set: { state.setBaseURL($0) }))
                .autocorrectionDisabled()
            Text("Default: https://api.openai.com/v1. Include the API path prefix (such as /v1), without /responses or /chat/completions. Remote APIs require HTTPS; loopback servers may use HTTP.")
                .font(.caption).foregroundStyle(.secondary)
            SecureField("API Key (optional for compatible servers)", text: $state.apiKey).autocorrectionDisabled()
            if let error = state.keyLoadError { Text(error).font(.caption).foregroundStyle(.red) }
            Toggle("Use Responses API", isOn: $useResponses)
            Text("Responses is the default. Turn it off for compatible servers or models that require Chat Completions.")
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
            Text("Keys are saved in Keychain separately for each Base URL. Changing the URL loads that endpoint’s key. Only apply URLs you trust. Requests do not follow redirects. Choose text models supported by the selected API mode.")
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
            try state.apply()
            status = "Settings applied."
            refreshModels()
        } catch { status = error.localizedDescription }
    }

    private func fetch() {
        fetching = true
        Task {
            defer { fetching = false }
            do {
                try state.apply()
                let configuration = try OpenAIConfiguration.load()
                let ids = try await OpenAIChatService.shared.models(configuration: configuration)
                guard (try? OpenAIConfiguration.normalizedBaseURL(state.baseURLText)) == configuration.baseURL else { return }
                availableIDs = ids
                showModels = true
                status = ""
            } catch { status = error.localizedDescription }
        }
    }
}
