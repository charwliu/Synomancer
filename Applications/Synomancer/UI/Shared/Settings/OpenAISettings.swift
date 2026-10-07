import SwiftUI

struct OpenAISettings: View {
    var refreshModels: () -> Void
    @AppStorage("openAIEnabled") private var enabled = false
    @AppStorage("openAIBaseURL") private var baseURL = "https://api.openai.com/v1"
    @AppStorage("openAIProtocol") private var apiProtocol = "responses"
    @AppStorage("openAIModels") private var modelIDs = ""
    @AppStorage("openAISendTemperature") private var sendTemperature = false
    @Binding var apiKey: String
    @State private var status = ""
    @State private var fetching = false
    @State private var availableIDs: [String] = []
    @State private var showModels = false

    var body: some View {
        Section("OpenAI") {
            Toggle("Enable OpenAI models", isOn: $enabled)
            Text("Selecting an OpenAI model sends conversation text to the configured API. Local MLX remains available.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("API Base URL", text: $baseURL)
                .autocorrectionDisabled()
            SecureField("API Key", text: $apiKey)
                .autocorrectionDisabled()
            Picker("API", selection: $apiProtocol) {
                Text("OpenAI Responses").tag("responses")
                Text("Chat Completions (compatible)").tag("chat")
            }
            Toggle("Send temperature parameter", isOn: $sendTemperature)
            Text("Keep temperature disabled for models that do not support it.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Model IDs (one per line)")
            TextEditor(text: $modelIDs).font(.system(.body, design: .monospaced)).frame(minHeight: 65)
            HStack {
                Button("Apply OpenAI Settings") { apply() }
                Button(fetching ? "Loading…" : "Choose API Models…") { fetch() }
                    .disabled(!enabled || fetching)
            }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
            Text("The API key is saved in Keychain. Model availability depends on your API account; choose a model that supports text generation.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .sheet(isPresented: $showModels) {
            VStack(alignment: .leading, spacing: 12) {
                Text("API Models").font(.headline)
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

    private var selectedIDs: Set<String> { Set(modelIDs.components(separatedBy: .newlines).filter { !$0.isEmpty }) }
    private func apply() {
        do {
            try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: "openai")
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
                try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: "openai")
                availableIDs = try await OpenAIChatService.shared.models(configuration: OpenAIConfiguration.load())
                showModels = true
                status = ""
            } catch { status = error.localizedDescription }
        }
    }
}
