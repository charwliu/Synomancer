import SwiftUI

struct OpenAISettings: View {
    var refreshModels: () -> Void
    @Bindable var state: OpenAISettingsState
    @State private var selectedID: String?
    @State private var status = ""

    var body: some View {
        Section("OpenAI-compatible Services") {
            if let error = state.loadError { Text(error).foregroundStyle(.red) }
            if !state.services.isEmpty {
                Picker("Service", selection: Binding(get: { selectedID ?? state.services.first!.id }, set: { selectedID = $0 })) {
                    ForEach(state.services) { service in
                        Text(service.profile.name).tag(service.id)
                    }
                }
            }
            HStack {
                Button("Add Service") { selectedID = state.add() }
                if let selected = selected {
                    Button("Remove Service", role: .destructive) {
                        state.remove(selected.id)
                        selectedID = state.services.first?.id
                    }
                }
            }.disabled(state.loadError != nil)
            if let selected {
                OpenAIServiceEditor(state: state, editor: selected)
                    .id(selected.id)
            }
            Button("Apply OpenAI Services") {
                do {
                    try state.apply()
                    refreshModels()
                    status = "Services applied."
                } catch { status = error.localizedDescription }
            }.disabled(state.loadError != nil)
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
            Text("Each service has its own models and Keychain credentials, even when Base URLs match. Changing a URL loads only that service’s key for the destination. Removing a service and applying deletes its saved keys; existing conversations remain.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var selected: OpenAIServiceSettingsState? {
        state.services.first { $0.id == selectedID } ?? state.services.first
    }
}

private struct OpenAIServiceEditor: View {
    @Bindable var state: OpenAISettingsState
    @Bindable var editor: OpenAIServiceSettingsState
    @State private var status = ""
    @State private var fetching = false
    @State private var availableIDs: [String] = []
    @State private var showModels = false

    var body: some View {
        Group {
            TextField("Service name", text: $editor.profile.name).autocorrectionDisabled()
            Toggle("Enable this service", isOn: $editor.profile.enabled)
            Text("Selecting a model sends conversation text and this service’s key to its configured endpoint. Only use services you trust.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Base URL", text: Binding(get: { editor.profile.baseURL }, set: { editor.setBaseURL($0) }))
                .autocorrectionDisabled()
            Text("Default: https://api.openai.com/v1. Include the API path prefix (such as /v1), without /responses or /chat/completions. Remote APIs require HTTPS; loopback servers may use HTTP. Requests do not follow redirects.")
                .font(.caption).foregroundStyle(.secondary)
            SecureField("API Key (optional for compatible servers)", text: $editor.apiKey).autocorrectionDisabled()
            if let error = editor.keyLoadError { Text(error).font(.caption).foregroundStyle(.red) }
            Link("Get an OpenAI API key", destination: URL(string: "https://platform.openai.com/api-keys")!)
            Toggle("Use Responses API", isOn: $editor.profile.useResponses)
            Text("Turn it off for services or models that require Chat Completions.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Send temperature parameter", isOn: $editor.profile.sendTemperature)
            TextField("Model IDs (one per line)", text: $editor.profile.models, axis: .vertical)
                .font(.system(.body, design: .monospaced)).lineLimit(1...8).autocorrectionDisabled()
            Text("Model selections remain drafts until Apply or Save.").font(.caption).foregroundStyle(.secondary)
            Button(fetching ? "Loading…" : "Choose API Models…") { fetch() }
                .disabled(!editor.profile.enabled || fetching)
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
        }
        .sheet(isPresented: $showModels) {
            VStack(alignment: .leading, spacing: 12) {
                Text(editor.profile.name + " Models").font(.headline)
                Text("Select text models supported by this service and API mode.").font(.caption)
                ScrollView {
                    LazyVStack(alignment: .leading) {
                        ForEach(availableIDs, id: \.self) { id in
                            Toggle(id, isOn: Binding(get: { editor.profile.modelIDs.contains(id) }, set: { selected in
                                var ids = Set(editor.profile.modelIDs)
                                if selected { ids.insert(id) } else { ids.remove(id) }
                                editor.profile.models = ids.sorted().joined(separator: "\n")
                            }))
                        }
                    }
                }
                Button("Done") { showModels = false }
            }.padding().frame(minWidth: 400, minHeight: 400)
        }
    }

    private func fetch() {
        fetching = true
        Task {
            defer { fetching = false }
            do {
                try editor.validateKey()
                let configuration = try OpenAIConfiguration.transport(apiKey: editor.apiKey, baseURL: editor.profile.baseURL,
                    useResponses: editor.profile.useResponses, sendTemperature: editor.profile.sendTemperature)
                let ids = try await OpenAIChatService.shared.models(configuration: configuration)
                guard state.services.contains(where: { $0.id == editor.id }),
                      (try? OpenAIConfiguration.normalizedBaseURL(editor.profile.baseURL)) == configuration.baseURL else { return }
                availableIDs = ids
                showModels = true
                status = ""
            } catch { status = error.localizedDescription }
        }
    }
}
