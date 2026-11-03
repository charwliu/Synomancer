import SwiftUI

struct LocalChatAPISettings: View {
    @AppStorage("localAPIEnabled") private var enabled = false
    @AppStorage("localAPIPort") private var port = "8080"
    @AppStorage("HF_HOME") private var home = SpeechCacheAccess.defaultPath
    @State private var token = ""
    @State private var status = ""
    @State private var starting = false
    @State private var showToken = false
    @State private var statusIsError = false
    private var server = LocalChatAPIServer.shared

    var body: some View {
        Section("Local MLX API") {
            Text("Serve local MLX text models to OpenAI-compatible clients on this Mac.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Port", text: $port).disabled(server.isRunning || server.isStarting || starting)
            VStack(alignment: .leading, spacing: 6) {
                Text("Local API Access Token").font(.caption).foregroundStyle(.secondary)
                Group {
                    if showToken { TextField("Local API Access Token", text: $token) }
                    else { SecureField("Local API Access Token", text: $token) }
                }
                .labelsHidden().textFieldStyle(.roundedBorder).autocorrectionDisabled()
                .accessibilityLabel("Local API Access Token").privacySensitive()
                .disabled(server.isRunning || server.isStarting || starting)
                HStack {
                    Button(showToken ? "Hide Token" : "Show Token") { showToken.toggle() }
                    Button("Copy Access Token") {
                        Clipboard.shared.setString(token)
                        statusIsError = false
                        status = "Access token copied."
                    }.disabled(token.isEmpty)
                }
            }
            HStack {
                Button("Generate Access Token") {
                    do {
                        token = try LocalAPIAccessToken.generate()
                        statusIsError = false
                        status = "New access token saved in Keychain."
                    } catch { statusIsError = true; status = error.localizedDescription }
                }.disabled(server.isRunning || server.isStarting || starting)
                Button(server.isRunning ? "Stop Local API" : (server.isStarting || starting) ? "Starting…" : "Start Local API") {
                    if server.isRunning { server.stop(); enabled = false; statusIsError = false; status = "" }
                    else { start() }
                }.disabled(server.isStarting || starting)
            }
            if server.isRunning || server.isStarting || starting {
                Text("Stop the Local API before generating or editing its access token.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if server.isRunning {
                Text("Running: http://127.0.0.1:\(server.port)/v1").textSelection(.enabled)
                Text("Endpoints: GET /models, POST /chat/completions (streaming supported).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = server.errorMessage { Text(error).font(.caption).foregroundStyle(.red) }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(statusIsError ? Color.red : Color.secondary) }
            Text("Generated tokens are saved immediately. Custom tokens are saved when you start the API. Use Show Token or Copy Access Token to configure your client.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Listening on localhost only. Use the access token as the client API key. The server exposes local models and does not forward requests to OpenAI. Restart after changing HF_HOME. While enabled, it starts again when Synomancer opens.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task { do { token = try ChatSecrets.read(LocalAPIAccessToken.keychainAccount) } catch { statusIsError = true; status = error.localizedDescription } }
        .onChange(of: home) { _, _ in server.stop(); enabled = false }
    }

    private func start() {
        starting = true
        Task {
            defer { starting = false }
            do {
                guard let number = UInt16(port), number > 0 else { throw OpenAIError.message("Port must be between 1 and 65535.") }
                token = try token.isEmpty ? LocalAPIAccessToken.generate() : LocalAPIAccessToken.persist(token)
                let cache = try await SpeechCacheAccess.authorizedConfiguration()
                try await server.start(port: number, token: token, cache: cache)
                enabled = true; statusIsError = false; status = ""
            } catch { enabled = false; statusIsError = true; status = error.localizedDescription }
        }
    }
}
