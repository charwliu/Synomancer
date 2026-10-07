import Foundation

// This fixture never contacts OpenAI or reads the user's API key.
final class OpenAIFixtureProtocol: URLProtocol, @unchecked Sendable {
    static let lock = NSLock()
    static var observed: [String: Data] = [:]
    private let stateLock = NSLock()
    private var stopped = false
    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "api.fixture.invalid" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let path = request.url!.path
        let body = request.httpBody ?? request.httpBodyStream.map { stream in
            stream.open(); defer { stream.close() }
            var data = Data(), buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable { let count = stream.read(&buffer, maxLength: buffer.count); if count <= 0 { break }; data.append(contentsOf: buffer.prefix(count)) }
            return data
        } ?? Data()
        Self.lock.lock(); Self.observed[path] = body; Self.lock.unlock()
        precondition(request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-key")
        let status = path.contains("unauthorized") ? 401 : path.contains("rate-limit") ? 429 : 200
        let type = status == 200 && !path.hasSuffix("models") ? "text/event-stream" : "application/json"
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": type])!, cacheStoragePolicy: .notAllowed)
        if path.contains("slow") {
            DispatchQueue.global().asyncAfter(deadline: .now() + 10) { [self] in
                stateLock.lock(); defer { stateLock.unlock() }
                if !stopped { client?.urlProtocol(self, didLoad: Data("data: [DONE]\n\n".utf8)); client?.urlProtocolDidFinishLoading(self) }
            }
            return
        }
        let payload: String
        if status != 200 { payload = "{\"error\":{\"message\":\"fixture API failure\"}}" }
        else if path.hasSuffix("models") { payload = "{\"data\":[{\"id\":\"fixture-model\"},{\"id\":\"another-model\"}]}" }
        else if path.contains("failed") { payload = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"fixture generation failure\"}}}\n\n" }
        else if path.contains("truncated") { payload = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n" }
        else if path.hasSuffix("responses") {
            payload = ": heartbeat\r\n\r\nevent: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\r\ndata: \"delta\":\"你好\"}\r\n\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"，MLX\"}\r\n\r\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":5}}}\r\n\r\n"
        } else {
            payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" MLX\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n"
        }
        // Split UTF-8 inside a scalar as a transport would; SSE must preserve it.
        let bytes = Data(payload.utf8), midpoint = bytes.count / 2
        client?.urlProtocol(self, didLoad: bytes.prefix(midpoint))
        client?.urlProtocol(self, didLoad: bytes.suffix(bytes.count - midpoint))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { stateLock.lock(); stopped = true; stateLock.unlock() }
    static func body(_ path: String) -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return (try! JSONSerialization.jsonObject(with: observed[path]!)) as! [String: Any]
    }
}

@main @MainActor
struct ChatAPITests {
    static func main() async throws {
        setbuf(stdout, nil)
        try parserTests()
        try await remoteClientTests()
        let cache = SpeechCacheAccess.Configuration(path: CommandLine.arguments[1], bookmark: nil)
        let models = try await LocalChatService.shared.models(cache: cache)
        let model = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : models.first!.id
        try await serverTests(cache: cache, model: model)
    }

    static func parserTests() throws {
        let valid = Data("{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"stop\":\"END\",\"max_completion_tokens\":8}".utf8)
        let decoded = try LocalChatAPIRequest.decode(valid)
        precondition(decoded.stop == ["END"] && decoded.maxTokens == 8)
        for body in ["{", "{}", "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[]}",
                     "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":0}"] {
            do { _ = try LocalChatAPIRequest.decode(Data(body.utf8)); preconditionFailure("Expected 400") }
            catch let error as APIHTTPError { precondition(error.status == 400) }
        }
        for header in ["POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2",
                       "POST / HTTP/1.1\r\nTransfer-Encoding: chunked", "POST / HTTP/1.1\r\nContent-Length: -1"] {
            do { _ = try APIHTTPRequest.parseHeaders(Data(header.utf8)); preconditionFailure("Expected bad headers") }
            catch let error as APIHTTPError { precondition(error.status == 400) }
        }
        var delta = APITextDelta(stop: ["STOP"])
        precondition(delta.next("你好ST") == "你好")
        precondition(delta.next("你好STOPignored") == "")
        precondition(delta.next("你好", final: true) == "")
        var unicode = APITextDelta(stop: [])
        precondition(unicode.next("e") == "e")
        precondition(unicode.next("e\u{301}🧑") == "\u{301}🧑")
        precondition(unicode.next("e\u{301}🧑\u{200D}💻", final: true) == "\u{200D}💻")
        print("PASS: request validation, HTTP framing, stop prefixes and Unicode deltas")
    }

    static func remoteClientTests() async throws {
        let settings = URLSessionConfiguration.ephemeral
        settings.protocolClasses = [OpenAIFixtureProtocol.self]
        let session = URLSession(configuration: settings)
        defer { session.invalidateAndCancel() }
        let client = OpenAIChatService(session: session)
        func config(_ prefix: String = "", responses: Bool = true) -> OpenAIConfiguration {
            .init(baseURL: URL(string: "https://api.fixture.invalid/\(prefix)v1")!, apiKey: "fixture-key", useResponses: responses, sendTemperature: false)
        }
        let messages: [LocalChatMessage] = [.init(role: "system", content: "test"), .init(role: "user", content: "hello")]
        let responses = try await client.generate(modelID: "fixture-model", messages: messages, configuration: config()) { _, _ in }
        precondition(responses.text == "你好，MLX" && responses.promptTokens == 12 && responses.completionTokens == 5)
        let sent = OpenAIFixtureProtocol.body("/v1/responses")
        precondition(sent["model"] as? String == "fixture-model" && sent["temperature"] == nil && sent["store"] as? Bool == false)
        precondition((sent["input"] as? [[String: String]])?.count == 2)
        let chat = try await client.generate(modelID: "fixture-model", messages: messages, configuration: config(responses: false)) { _, _ in }
        precondition(chat.text == "hello MLX" && chat.promptTokens == 12 && chat.completionTokens == 5)
        precondition((OpenAIFixtureProtocol.body("/v1/chat/completions")["messages"] as? [[String: String]])?.count == 2)
        let models = try await client.models(configuration: config())
        precondition(models == ["another-model", "fixture-model"])
        for prefix in ["unauthorized/", "rate-limit/", "truncated/", "failed/"] {
            do {
                _ = try await client.generate(modelID: "fixture-model", messages: messages, configuration: config(prefix, responses: prefix != "truncated/")) { _, _ in }
                preconditionFailure("Expected API failure")
            } catch let error as OpenAIError { precondition(!error.localizedDescription.isEmpty) }
        }
        let slow = Task { try await client.generate(modelID: "fixture-model", messages: messages, configuration: config("slow/")) { _, _ in } }
        try await Task.sleep(for: .milliseconds(100)); slow.cancel()
        do { _ = try await slow.value; preconditionFailure("Expected cancelled API call") }
        catch { precondition(error is CancellationError || (error as? URLError)?.code == .cancelled) }
        print("PASS: OpenAI Responses/Chat SSE, request schemas, model list, 401/429, failed/truncated stream and cancellation (HTTP fixture)")
    }

    static func serverTests(cache: SpeechCacheAccess.Configuration, model: String) async throws {
        let server = LocalChatAPIServer()
        let token = "local-test-token-00000000"
        // Retry ports to avoid depending on a developer's existing API server.
        var chosen: UInt16 = 0
        for port in UInt16(18980)...UInt16(18999) {
            do { try await server.start(port: port, token: token, cache: cache); chosen = port; break } catch { continue }
        }
        precondition(chosen != 0 && server.isRunning)
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        let base = URL(string: "http://127.0.0.1:\(chosen)/v1")!
        func request(_ path: String, body: [String: Any]? = nil, authorized: Bool = true) throws -> URLRequest {
            var request = URLRequest(url: base.appendingPathComponent(path))
            request.timeoutInterval = 60
            if authorized { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            return request
        }
        let (catalogData, catalogResponse) = try await session.data(for: request("models"))
        let catalog = try JSONSerialization.jsonObject(with: catalogData) as! [String: Any]
        precondition((catalogResponse as! HTTPURLResponse).statusCode == 200 && (catalog["data"] as! [[String: Any]]).contains { $0["id"] as? String == model })
        let (_, unauthorized) = try await session.data(for: request("models", authorized: false))
        precondition((unauthorized as! HTTPURLResponse).statusCode == 401)
        let (_, missing) = try await session.data(for: request("responses", body: [:]))
        precondition((missing as! HTTPURLResponse).statusCode == 404)
        var body: [String: Any] = ["model": model, "messages": [["role": "user", "content": "Reply only with OK."]], "temperature": 0, "max_tokens": 32]
        var bad = body; bad["tools"] = []
        let (_, unsupported) = try await session.data(for: request("chat/completions", body: bad))
        precondition((unsupported as! HTTPURLResponse).statusCode == 400)
        bad = body; bad["model"] = "missing-local-model"
        let (_, missingModel) = try await session.data(for: request("chat/completions", body: bad))
        precondition((missingModel as! HTTPURLResponse).statusCode == 404)
        print("PASS: real localhost listener, local-only model catalog, auth and HTTP 400/404 errors")

        let (data, response) = try await session.data(for: request("chat/completions", body: body))
        let answer = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        precondition((response as! HTTPURLResponse).statusCode == 200)
        let choice = (answer["choices"] as! [[String: Any]])[0]
        let message = choice["message"] as! [String: String]
        let usage = answer["usage"] as! [String: Int]
        precondition(message["content"]!.uppercased().contains("OK") && usage["prompt_tokens"]! > 0 && usage["completion_tokens"]! > 0)
        precondition(answer["object"] as? String == "chat.completion")
        print("PASS: non-streaming API generates with actual cached MLX weights and reports token usage")

        let client = OpenAIChatService(session: session)
        let clientConfig = OpenAIConfiguration(baseURL: base, apiKey: token, useResponses: false, sendTemperature: true)
        let streamed = try await client.generate(modelID: model, messages: [.init(role: "user", content: "Reply only with OK.")], configuration: clientConfig, temperature: 0) { _, _ in }
        precondition(streamed.text.uppercased().contains("OK") && streamed.promptTokens > 0 && streamed.completionTokens > 0)
        print("PASS: OpenAI client consumes real MLX HTTP chunked SSE and final usage")

        body["messages"] = [["role": "user", "content": "Reply only with REDWOOD-472."]]
        body["stop"] = "WOOD"
        body["stream"] = true
        let (stopBytes, stopResponse) = try await session.bytes(for: request("chat/completions", body: body))
        precondition((stopResponse as! HTTPURLResponse).statusCode == 200)
        var parser = ChatSSEDecoder(), text = "", finish = "", done = false
        for try await byte in stopBytes {
            guard let event = try parser.consume(byte) else { continue }
            if event == "[DONE]" { done = true; break }
            let value = try JSONSerialization.jsonObject(with: Data(event.utf8)) as! [String: Any]
            let choice = (value["choices"] as! [[String: Any]])[0]
            text += (choice["delta"] as? [String: String])?["content"] ?? ""
            if let reason = choice["finish_reason"] as? String { finish = reason }
        }
        precondition(done && !text.contains("WOOD") && text.contains("RED") && finish == "stop")
        print("PASS: real streaming stop sequence excludes the stop string and its partial prefixes")

        body.removeValue(forKey: "stop")
        body["messages"] = [["role": "user", "content": "Write a very long story of at least 2000 words about exploring the ocean."]]
        body["max_tokens"] = 2048
        let longRequest = try request("chat/completions", body: body)
        let long = Task {
            let (bytes, _) = try await session.bytes(for: longRequest)
            for try await _ in bytes { try Task.checkCancellation() }
        }
        try await Task.sleep(for: .milliseconds(500))
        long.cancel()
        _ = try? await long.value
        // A new short request must complete promptly after a client disconnect.
        let restarted = try await client.generate(modelID: model, messages: [.init(role: "user", content: "Reply only with OK.")], configuration: clientConfig, temperature: 0) { _, _ in }
        precondition(restarted.text.uppercased().contains("OK"))
        let interrupted = Task {
            let (bytes, _) = try await session.bytes(for: longRequest)
            for try await _ in bytes { try Task.checkCancellation() }
        }
        try await Task.sleep(for: .milliseconds(500))
        server.stop(); precondition(!server.isRunning)
        _ = try? await interrupted.value
        try await server.start(port: chosen, token: token, cache: cache)
        let (_, afterRestart) = try await session.data(for: request("models"))
        precondition((afterRestart as! HTTPURLResponse).statusCode == 200)
        let afterStop = try await client.generate(modelID: model, messages: [.init(role: "user", content: "Reply only with OK.")], configuration: clientConfig, temperature: 0) { _, _ in }
        precondition(afterStop.text.uppercased().contains("OK"))
        print("PASS: client disconnect/server stop cancel MLX generation; replacement request and listener restart succeed")
    }
}
