import Foundation
@preconcurrency import Network

// Real loopback TCP, synthetic responses; never reads Keychain or calls a cloud account.
@MainActor
private final class OpenAIEndpointFixture {
    private var listener: NWListener?
    private(set) var requests: [APIHTTPRequest] = []
    private(set) var baseURL: URL!
    var redirectURL: URL?
    var redirectStatus = 302

    func start() async throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        listener.newConnectionHandler = { connection in
            connection.start(queue: .main)
            Task { @MainActor in await self.serve(APIConnection(connection)) }
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            listener.stateUpdateHandler = { state in
                Task { @MainActor in
                    switch state {
                    case .ready:
                        listener.stateUpdateHandler = nil
                        self.baseURL = URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/proxy/v1")!
                        continuation.resume()
                    case .failed(let error):
                        listener.stateUpdateHandler = nil
                        continuation.resume(throwing: error)
                    default: break
                    }
                }
            }
            listener.start(queue: .main)
        }
    }

    func stop() { listener?.cancel(); listener = nil }

    private func serve(_ connection: APIConnection) async {
        defer { connection.close() }
        do {
            let request = try await connection.request()
            requests.append(request)
            if let redirectURL {
                try await connection.send(Data("HTTP/1.1 \(redirectStatus) Redirect\r\nLocation: \(redirectURL.absoluteString)\r\nContent-Length: 0\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".utf8))
            } else if request.path.hasSuffix("/models") {
                try await connection.json(["data": [["id": "compatible-model"]]])
            } else {
                try await connection.startSSE()
                if request.path.hasSuffix("/responses") {
                    try await connection.event(["type": "response.output_text.delta", "delta": "loopback 回复"])
                    try await connection.event(["type": "response.completed", "response": ["usage": ["input_tokens": 7, "output_tokens": 3]]])
                } else {
                    try await connection.event(["choices": [["delta": ["content": "loopback 回复"], "finish_reason": "stop"]]])
                    try await connection.event(["choices": [], "usage": ["prompt_tokens": 7, "completion_tokens": 3]])
                }
                try await connection.endSSE()
            }
        } catch { /* A client may close after receiving a complete Responses event. */ }
    }
}

@MainActor
enum OpenAIEndpointTests {
    private static func check(_ condition: @autoclosure () throws -> Bool) throws {
        let value = try condition()
        precondition(value)
    }

    static func configurationTests() throws {
        let normalize = OpenAIConfiguration.normalizedBaseURL
        try check(try normalize(" \n") == OpenAIConfiguration.officialBaseURL)
        try check(try normalize("HTTPS://API.OPENAI.COM:443/v1///") == OpenAIConfiguration.officialBaseURL)
        try check(try normalize("https://api.fixture.invalid:8443/proxy/v1/").absoluteString == "https://api.fixture.invalid:8443/proxy/v1")
        for url in ["http://localhost:8080/v1", "http://127.0.0.1:8080/v1", "http://127.2.3.4/v1", "http://[::1]:8080/v1"] {
            _ = try normalize(url)
        }
        let credentialURL = "https://" + ["fixture-user", "fixture-password"].joined(separator: ":") + "@api.fixture.invalid/v1"
        let invalid = ["/v1", "api.fixture.invalid/v1", "file:///tmp/v1", "ftp://api.fixture.invalid/v1", "https:///v1",
                       "https://api.fixture.invalid:0/v1", "https://api.fixture.invalid:65536/v1",
                       "https://api.fixture.invalid:bad/v1", "https://api.fixture.invalid/v1?token=fixture-value",
                       "https://api.fixture.invalid/v1#fragment", credentialURL, "https://api.fixture.invalid/a b",
                       "http://api.fixture.invalid/v1", "http://localhost.fixture.invalid/v1", "http://127.0.0.256/v1", "http://127.-0.0.1/v1"]
        for url in invalid {
            do { _ = try normalize(url); preconditionFailure("Unsafe URL accepted") }
            catch { precondition(!error.localizedDescription.contains("fixture-password") && !error.localizedDescription.contains("fixture-value")) }
        }
        let official = OpenAIConfiguration.keychainAccount(for: try normalize("https://API.OPENAI.COM:443/v1/"))
        precondition(official == "openai")
        let custom = try normalize("https://api.fixture.invalid/proxy/v1")
        let account = OpenAIConfiguration.keychainAccount(for: custom)
        precondition(account != official && account != GeminiConfiguration.keychainAccount)
        try check(account == OpenAIConfiguration.keychainAccount(for: try normalize(custom.absoluteString + "/")))
        for other in ["https://api.fixture.invalid/other/v1", "https://api.fixture.invalid:8443/proxy/v1", "https://other.fixture.invalid/proxy/v1"] {
            try check(account != OpenAIConfiguration.keychainAccount(for: try normalize(other)))
        }
        let suite = "Synomancer.EndpointTest." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        preferences.set(true, forKey: "openAIEnabled")
        preferences.set(credentialURL, forKey: "openAIBaseURL")
        var keyReads = 0
        do {
            _ = try OpenAIConfiguration.load(in: preferences, readKey: { _ in keyReads += 1; return "fixture-key" })
            preconditionFailure("Invalid saved URL accepted")
        } catch { precondition(keyReads == 0) }
        preferences.removeObject(forKey: "openAIBaseURL")
        var keys = [official: "official-fixture", account: "custom-fixture"]
        var writes = 0
        let state = OpenAISettingsState(preferences: preferences, readKey: { keys[$0] ?? "" }, writeKey: { key, account in keys[account] = key; writes += 1 })
        state.load()
        precondition(state.apiKey == "official-fixture")
        state.setBaseURL(custom.absoluteString)
        precondition(state.apiKey == "custom-fixture")
        try state.apply()
        precondition(writes == 0 && preferences.string(forKey: "openAIBaseURL") == custom.absoluteString)
        state.setBaseURL("http://localhost:8080/v1")
        precondition(state.apiKey.isEmpty)
        try state.apply()
        let anonymous = try OpenAIConfiguration.load(in: preferences, readKey: { keys[$0] ?? "" })
        precondition(anonymous.apiKey.isEmpty && anonymous.baseURL.host == "localhost")
        state.apiKey = "unsaved-fixture"
        state.setBaseURL(custom.absoluteString)
        precondition(state.apiKey == "custom-fixture", "Unsaved key must not move to another endpoint")
        state.apiKey = " updated-fixture "
        try state.apply()
        precondition(keys[account] == "updated-fixture" && keys[official] == "official-fixture" && writes == 1)
        state.setBaseURL(credentialURL)
        precondition(state.apiKey.isEmpty)
        do { try state.apply(); preconditionFailure("Invalid draft saved") } catch { }
        precondition(preferences.string(forKey: "openAIBaseURL") == custom.absoluteString && writes == 1)
        state.setBaseURL("")
        precondition(state.apiKey == "official-fixture")
        try state.apply()
        precondition(preferences.string(forKey: "openAIBaseURL") == OpenAIConfiguration.officialBaseURL.absoluteString)
        let failing = OpenAISettingsState(preferences: preferences, readKey: { _ in throw OpenAIError.message("fixture Keychain failure") }, writeKey: { _, _ in writes += 1 })
        failing.load()
        do { try failing.apply(); preconditionFailure("Failed key read silently enabled") } catch { }
        preferences.set(false, forKey: "openAIEnabled")
        try failing.apply()
        precondition(writes == 1, "Failed read deleted an untouched key")
        preferences.set(true, forKey: "openAIEnabled")
        failing.apiKey = "replacement-fixture"
        try failing.apply()
        precondition(writes == 2 && failing.keyLoadError == nil)
        print("PASS: Base URL validation/canonicalization, draft-only persistence, scoped Keychain accounts, endpoint switching and failed-read safety")
    }

    static func loopbackTests() async throws {
        let server = OpenAIEndpointFixture(), destination = OpenAIEndpointFixture()
        try await server.start()
        try await destination.start()
        defer { server.stop(); destination.stop() }
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        sessionConfiguration.urlCache = nil
        let session = URLSession(configuration: sessionConfiguration)
        defer { session.invalidateAndCancel() }
        let client = OpenAIChatService(session: session)
        let history = [LocalChatMessage(role: "user", content: "loopback fixture")]
        for key in ["fixture-key", ""] {
            let base = server.baseURL!.appendingPathComponent(key.isEmpty ? "anonymous" : "authenticated")
            for responses in [true, false] {
                let config = try OpenAIConfiguration.transport(apiKey: key, baseURL: base.absoluteString, useResponses: responses)
                let models = try await client.models(configuration: config)
                precondition(models == ["compatible-model"])
                let result = try await client.generate(modelID: models[0], messages: history, configuration: config) { _, _ in }
                precondition(result.text == "loopback 回复" && result.promptTokens == 7 && result.completionTokens == 3)
                let request = server.requests.last!
                precondition(request.method == "POST" && request.path == base.path + (responses ? "/responses" : "/chat/completions"))
                let body = try JSONSerialization.jsonObject(with: request.body) as! [String: Any]
                precondition(body[responses ? "input" : "messages"] != nil)
                precondition(body["temperature"] == nil)
                if responses { precondition(body["store"] as? Bool == false) }
                else { precondition((body["stream_options"] as? [String: Bool])?["include_usage"] == true) }
            }
        }
        for request in server.requests {
            precondition(request.headers["authorization"] == (request.path.contains("anonymous") ? nil : "Bearer fixture-key"))
        }
        print("PASS: real loopback HTTP, custom path prefix, models, Responses/Chat streaming and usage, with/without Bearer auth (synthetic output)")
        server.redirectURL = destination.baseURL
        let initialCount = server.requests.count
        for status in [301, 302, 303, 307, 308] {
            print("Checking redirect HTTP \(status)")
            server.redirectStatus = status
            let config = try OpenAIConfiguration.transport(apiKey: "fixture-key", baseURL: server.baseURL.absoluteString)
            do { _ = try await client.models(configuration: config); preconditionFailure("Model redirect followed") }
            catch let error as OpenAIError { precondition(error.localizedDescription.contains("Could not list")) }
            for responses in [true, false] {
                let config = try OpenAIConfiguration.transport(apiKey: "fixture-key", baseURL: server.baseURL.absoluteString, useResponses: responses)
                do {
                    _ = try await client.generate(modelID: "fixture-model", messages: history, configuration: config) { _, _ in }
                    preconditionFailure("Generation redirect followed")
                } catch let error as OpenAIError {
                    print("Redirect generation: \(error.localizedDescription)")
                    precondition(error.localizedDescription.contains("HTTP \(status)"))
                }
            }
        }
        print("Redirect source requests: \(server.requests.count - initialCount); destination requests: \(destination.requests.count)")
        precondition(server.requests.count == initialCount + 15)
        precondition(destination.requests.isEmpty, "Redirect destination received credentials or conversation")
        print("PASS: 301/302/303/307/308 redirects blocked for models, Responses and Chat; destination received zero requests")
    }
}
