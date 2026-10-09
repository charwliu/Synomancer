import SwiftUI

struct GeminiSettings: View {
    var refreshModels: () -> Void
    @AppStorage("geminiEnabled") private var enabled = false
    @AppStorage("geminiModels") private var modelIDs = ""
    @AppStorage("geminiSendTemperature") private var sendTemperature = false
    @Binding var apiKey: String
    @State private var status = ""
    @State private var fetching = false
    @State private var availableIDs: [String] = []
    @State private var showModels = false

    var body: some View {
        Section("Gemini") {
            Toggle("Enable Gemini models", isOn: $enabled)
            Text("Selecting a Gemini model sends conversation text to Google Gemini. Local MLX remains available.")
                .font(.caption).foregroundStyle(.secondary)
            Link("Get a Gemini API key", destination: URL(string: "https://aistudio.google.com/apikey")!)
            SecureField("API Key", text: $apiKey)
                .autocorrectionDisabled()
            Toggle("Send temperature parameter", isOn: $sendTemperature)
            Text("Keep temperature disabled for models that do not support it.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Model IDs (one per line)")
            TextField("Model IDs (one per line)", text: $modelIDs, axis: .vertical)
                .labelsHidden()
                .font(.system(.body, design: .monospaced))
                .lineLimit(1...8)
                .autocorrectionDisabled()
            HStack {
                Button("Apply Gemini Settings") { apply() }
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
            try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: GeminiConfiguration.keychainAccount)
            if enabled { _ = try GeminiConfiguration.load() }
            status = "Settings applied."
            refreshModels()
        } catch { status = error.localizedDescription }
    }
    private func fetch() {
        fetching = true
        Task {
            defer { fetching = false }
            do {
                try ChatSecrets.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), account: GeminiConfiguration.keychainAccount)
                availableIDs = try await OpenAIChatService.shared.models(configuration: GeminiConfiguration.load())
                showModels = true
                status = ""
            } catch { status = error.localizedDescription }
        }
    }
}
