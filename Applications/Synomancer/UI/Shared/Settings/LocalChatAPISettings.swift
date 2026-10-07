import SwiftUI

struct LocalChatAPISettings: View {
    @AppStorage("localAPIEnabled") private var enabled = false
    @AppStorage("localAPIPort") private var port = "8080"
    @AppStorage("HF_HOME") private var home = SpeechCacheAccess.defaultPath
    @State private var token = ""
    @State private var status = ""
    @State private var starting = false
    private var server = LocalChatAPIServer.shared

    var body: some View {
        Section("Local MLX API") {
            Text("Serve local MLX text models to OpenAI-compatible clients on this Mac.")
                .font(.caption).foregroundStyle(.secondary)
            TextField("Port", text: $port).disabled(server.isRunning || server.isStarting || starting)
            SecureField("Local API Access Token", text: $token).disabled(server.isRunning || server.isStarting || starting)
            HStack {
                Button("Generate Access Token") {
                    do { token = try ChatSecrets.newToken(); try ChatSecrets.save(token, account: "local-server") }
                    catch { status = error.localizedDescription }
                }.disabled(server.isRunning || server.isStarting || starting)
                Button(server.isRunning ? "Stop Local API" : (server.isStarting || starting) ? "Starting…" : "Start Local API") {
                    if server.isRunning { server.stop(); enabled = false; status = "" }
                    else { start() }
                }.disabled(server.isStarting || starting)
            }
            if server.isRunning {
                Text("Running: http://127.0.0.1:\(server.port)/v1").textSelection(.enabled)
                Text("Endpoints: GET /models, POST /chat/completions (streaming supported).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = server.errorMessage { Text(error).font(.caption).foregroundStyle(.red) }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
            Text("Listening on localhost only. Use the access token as the client API key. The server exposes local models and does not forward requests to OpenAI. Restart after changing HF_HOME. While enabled, it starts again when Synomancer opens.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task { do { token = try ChatSecrets.read("local-server") } catch { status = error.localizedDescription } }
        .onChange(of: home) { _, _ in server.stop(); enabled = false }
    }

    private func start() {
        starting = true
        Task {
            defer { starting = false }
            do {
                guard let number = UInt16(port), number > 0 else { throw OpenAIError.message("Port must be between 1 and 65535.") }
                if token.isEmpty { token = try ChatSecrets.newToken() }
                try ChatSecrets.save(token, account: "local-server")
                let cache = try await SpeechCacheAccess.authorizedConfiguration()
                try await server.start(port: number, token: token, cache: cache)
                enabled = true; status = ""
            } catch { enabled = false; status = error.localizedDescription }
        }
    }
}
