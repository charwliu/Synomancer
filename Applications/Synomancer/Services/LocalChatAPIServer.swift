import Foundation
@preconcurrency import Network

struct LocalChatAPIRequest: Decodable, Sendable {
    struct Message: Decodable, Sendable { let role: String; let content: String }
    struct StreamOptions: Decodable, Sendable { let include_usage: Bool? }
    let model: String
    let messages: [Message]
    let stream: Bool?
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let temperature: Float?
    let top_p: Float?
    let n: Int?
    let stream_options: StreamOptions?
    let stop: [String]?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, max_tokens, max_completion_tokens, temperature, top_p, n, stream_options, stop
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        model = try values.decode(String.self, forKey: .model)
        messages = try values.decode([Message].self, forKey: .messages)
        stream = try values.decodeIfPresent(Bool.self, forKey: .stream)
        max_tokens = try values.decodeIfPresent(Int.self, forKey: .max_tokens)
        max_completion_tokens = try values.decodeIfPresent(Int.self, forKey: .max_completion_tokens)
        temperature = try values.decodeIfPresent(Float.self, forKey: .temperature)
        top_p = try values.decodeIfPresent(Float.self, forKey: .top_p)
        n = try values.decodeIfPresent(Int.self, forKey: .n)
        stream_options = try values.decodeIfPresent(StreamOptions.self, forKey: .stream_options)
        if let single = try? values.decode(String.self, forKey: .stop) { stop = [single] }
        else { stop = try values.decodeIfPresent([String].self, forKey: .stop) }
    }

    var maxTokens: Int { max_completion_tokens ?? max_tokens ?? 2048 }

    static func decode(_ data: Data) throws -> Self {
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw APIHTTPError(status: 400, message: "Expected a JSON object.")
        }
        let allowed: Set<String> = ["model", "messages", "stream", "max_tokens", "max_completion_tokens", "temperature", "top_p", "n", "stream_options", "stop"]
        if let field = json.keys.sorted().first(where: { !allowed.contains($0) }) {
            throw APIHTTPError(status: 400, message: "Unsupported parameter: \(field). This server supports text chat only.")
        }
        if let options = json["stream_options"] as? [String: Any], options.keys.contains(where: { $0 != "include_usage" }) {
            throw APIHTTPError(status: 400, message: "Unsupported stream_options parameter.")
        }
        let request: Self
        do { request = try JSONDecoder().decode(Self.self, from: data) }
        catch { throw APIHTTPError(status: 400, message: "Invalid chat request. Messages must have a role and string content.") }
        guard !request.model.isEmpty, !request.messages.isEmpty, request.messages.count <= 128,
              request.messages.allSatisfy({ ["system", "developer", "user", "assistant"].contains($0.role) }),
              request.messages.reduce(0, { $0 + $1.content.utf8.count }) <= 262144,
              (1...8192).contains(request.maxTokens), request.n == nil || request.n == 1,
              (0...2).contains(request.temperature ?? 0.6), (0...1).contains(request.top_p ?? 1),
              !(request.max_tokens != nil && request.max_completion_tokens != nil),
              (request.stop ?? []).count <= 4,
              (request.stop ?? []).allSatisfy({ !$0.isEmpty && $0.utf8.count <= 256 }) else {
            throw APIHTTPError(status: 400, message: "Invalid model, messages, sampling or token limit (1–8192). Only n=1 is supported.")
        }
        return request
    }
}

struct APIHTTPError: LocalizedError, Sendable {
    let status: Int
    let message: String
    var errorDescription: String? { message }
}

struct APIHTTPRequest: Sendable {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    // Length-framed HTTP/1.x; pipelining, upgrades and chunked request bodies are unsupported.
    static func parseHeaders(_ data: Data) throws -> (String, String, [String: String], Int) {
        guard let text = String(data: data, encoding: .utf8) else { throw APIHTTPError(status: 400, message: "Invalid HTTP headers.") }
        let lines = text.components(separatedBy: "\r\n")
        let parts = (lines.first ?? "").split(separator: " ")
        guard parts.count == 3, ["HTTP/1.0", "HTTP/1.1"].contains(String(parts[2])) else { throw APIHTTPError(status: 400, message: "Invalid request line.") }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":"), !line.hasPrefix(" "), !line.hasPrefix("\t") else { throw APIHTTPError(status: 400, message: "Invalid header.") }
            let name = line[..<colon].lowercased()
            guard !name.isEmpty, headers[name] == nil else { throw APIHTTPError(status: 400, message: "Duplicate HTTP header.") }
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        guard headers["transfer-encoding"] == nil, headers["upgrade"] == nil else { throw APIHTTPError(status: 400, message: "Chunked request bodies and upgrades are unsupported.") }
        let length: Int
        if let value = headers["content-length"] {
            guard let parsed = Int(value), parsed >= 0 else { throw APIHTTPError(status: 400, message: "Invalid Content-Length.") }
            length = parsed
        } else { length = 0 }
        guard length <= 1048576 else { throw APIHTTPError(status: 413, message: "Request body exceeds 1 MiB.") }
        return (String(parts[0]), String(parts[1].split(separator: "?", maxSplits: 1)[0]), headers, length)
    }
}

// NWConnection is thread safe; all sends in a response are awaited in order.
final class APIConnection: @unchecked Sendable {
    let connection: NWConnection
    init(_ connection: NWConnection) { self.connection = connection }
    func close() { connection.cancel() }

    func read(closeOnCancellation: Bool = true) async throws -> Data? {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) { data, _, complete, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let data, !data.isEmpty { continuation.resume(returning: data) }
                    else if complete { continuation.resume(returning: nil) }
                    else { continuation.resume(throwing: APIHTTPError(status: 400, message: "Empty HTTP input.")) }
                }
            }
        } onCancel: { if closeOnCancellation { self.close() } }
    }

    func request() async throws -> APIHTTPRequest {
        let timeout = Task { try await Task.sleep(for: .seconds(30)); close() }
        defer { timeout.cancel() }
        var buffer = Data()
        let separator = Data("\r\n\r\n".utf8)
        while buffer.range(of: separator) == nil {
            guard let part = try await read() else { throw CancellationError() }
            buffer.append(part)
            if buffer.range(of: separator) == nil && buffer.count > 16384 { throw APIHTTPError(status: 431, message: "Headers exceed 16 KiB.") }
        }
        let range = buffer.range(of: separator)!
        guard range.lowerBound <= 16384 else { throw APIHTTPError(status: 431, message: "Headers exceed 16 KiB.") }
        let (method, path, headers, length) = try APIHTTPRequest.parseHeaders(Data(buffer[..<range.lowerBound]))
        var body = Data(buffer[range.upperBound...])
        if method == "POST", headers["content-length"] == nil { throw APIHTTPError(status: 411, message: "Content-Length is required.") }
        while body.count < length {
            guard let part = try await read() else { throw CancellationError() }
            body.append(part)
        }
        guard body.count == length else { throw APIHTTPError(status: 400, message: "HTTP pipelining is unsupported.") }
        return APIHTTPRequest(method: method, path: path, headers: headers, body: body)
    }

    func send(_ data: Data) async throws {
        let timeout = Task { try await Task.sleep(for: .seconds(30)); close() }
        defer { timeout.cancel() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                connection.send(content: data, completion: .contentProcessed { error in
                    if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                })
            }
        } onCancel: { self.close() }
    }

    func json(_ value: [String: Any], status: Int = 200) async throws {
        let body = try JSONSerialization.data(withJSONObject: value)
        let reason = HTTPURLResponse.localizedString(forStatusCode: status)
        var data = Data("HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
        data.append(body)
        try await send(data)
    }

    func startSSE() async throws {
        try await send(Data("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".utf8))
    }
    func event(_ value: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: value)
        var frame = Data("data: ".utf8); frame.append(data); frame.append(Data("\n\n".utf8))
        try await chunk(frame)
    }
    func chunk(_ body: Data) async throws {
        var data = Data("\(String(body.count, radix: 16))\r\n".utf8)
        data.append(body); data.append(Data("\r\n".utf8))
        try await send(data)
    }
    func endSSE() async throws {
        try await chunk(Data("data: [DONE]\n\n".utf8))
        try await send(Data("0\r\n\r\n".utf8))
    }
}

// Hold back incomplete UTF-8 and possible stop-sequence prefixes until they are stable.
struct APITextDelta {
    var delivered = ""
    let stop: [String]
    mutating func next(_ snapshot: String, final: Bool = false) -> String {
        var stable = snapshot
        while stable.last == "�" { stable.removeLast() }
        if let end = stop.compactMap({ stable.range(of: $0)?.lowerBound }).min() { stable = String(stable[..<end]) }
        if !final {
            var withheld = 0
            for sequence in stop {
                for count in 1..<sequence.count where stable.hasSuffix(sequence.prefix(count)) { withheld = max(withheld, count) }
            }
            if withheld > 0 { stable.removeLast(withheld) }
        }
        guard stable.utf8.starts(with: delivered.utf8) else { return "" }
        let delta = String(decoding: stable.utf8.dropFirst(delivered.utf8.count), as: UTF8.self)
        delivered = stable
        return delta
    }
}

@Observable @MainActor
final class LocalChatAPIServer {
    static let shared = LocalChatAPIServer()
    private(set) var isRunning = false
    private(set) var isStarting = false
    private(set) var port: UInt16 = 8080
    private(set) var errorMessage: String?
    @ObservationIgnored private var listener: NWListener?
    @ObservationIgnored private var connections: [UUID: APIConnection] = [:]
    @ObservationIgnored private var tasks: [UUID: Task<Void, Never>] = [:]
    @ObservationIgnored private var epoch = UUID()
    @ObservationIgnored private var shutdown: Task<Void, Never>?
    @ObservationIgnored private var startingContinuation: AsyncThrowingStream<Void, Error>.Continuation?

    func start(port: UInt16, token: String, cache: SpeechCacheAccess.Configuration) async throws {
        stop()
        let identifier = epoch
        await shutdown?.value
        try Task.checkCancellation()
        guard epoch == identifier else { throw CancellationError() }
        guard port > 0, token.count >= 16 else { throw OpenAIError.message("Choose a valid port and an access token of at least 16 characters.") }
        isStarting = true
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        let newListener: NWListener
        do { newListener = try NWListener(using: parameters) }
        catch { isStarting = false; throw error }
        listener = newListener
        let states = AsyncThrowingStream<Void, Error> { continuation in
            self.startingContinuation = continuation
            newListener.stateUpdateHandler = { state in
                switch state {
                case .ready: continuation.yield(()); continuation.finish()
                case .failed(let error): continuation.finish(throwing: error)
                case .cancelled: continuation.finish(throwing: CancellationError())
                default: break
                }
            }
        }
        newListener.newConnectionHandler = { [weak self] connection in
            connection.start(queue: .global(qos: .userInitiated))
            Task { @MainActor in
                guard let self, self.epoch == identifier, self.connections.count < 8 else { connection.cancel(); return }
                let id = UUID(), wire = APIConnection(connection)
                self.connections[id] = wire
                self.tasks[id] = Task {
                    await Self.handle(wire, token: token, cache: cache)
                    wire.close()
                    self.connections.removeValue(forKey: id)
                    self.tasks.removeValue(forKey: id)
                }
            }
        }
        newListener.start(queue: .global(qos: .userInitiated))
        do {
            try await withTaskCancellationHandler {
                for try await _ in states { try Task.checkCancellation(); break }
            } onCancel: { newListener.cancel() }
            guard epoch == identifier else { throw CancellationError() }
            startingContinuation = nil
            self.port = port; isStarting = false; isRunning = true; errorMessage = nil
            newListener.stateUpdateHandler = { [weak self] state in
                if case .failed(let error) = state {
                    Task { @MainActor in
                        guard let self, self.epoch == identifier else { return }
                        self.stop(); self.errorMessage = error.localizedDescription
                    }
                }
            }
        } catch {
            if epoch == identifier { stop(); errorMessage = error.localizedDescription }
            throw error
        }
    }

    func restore(in preferences: UserDefaults = .standard) async {
        guard preferences.bool(forKey: "localAPIEnabled"), !isRunning, !isStarting else { return }
        do {
            guard let port = UInt16(preferences.string(forKey: "localAPIPort") ?? "8080") else {
                throw OpenAIError.message("Invalid local API port.")
            }
            let token = try ChatSecrets.read("local-server")
            // Restoring uses the saved grant; it never opens a folder picker at launch.
            let cache = SpeechCacheAccess.configuration(in: preferences)
            _ = try await LocalChatService.shared.models(cache: cache)
            try await start(port: port, token: token, cache: cache)
        } catch { errorMessage = error.localizedDescription; preferences.set(false, forKey: "localAPIEnabled") }
    }

    func stop() {
        epoch = UUID()
        startingContinuation?.finish(throwing: CancellationError()); startingContinuation = nil
        if let listener {
            let states = AsyncStream<Void> { continuation in
                listener.stateUpdateHandler = { state in
                    if case .cancelled = state { continuation.finish() }
                }
            }
            let previous = shutdown
            shutdown = Task {
                for await _ in states { }
                await previous?.value
            }
            listener.cancel()
        }
        listener = nil
        tasks.values.forEach { $0.cancel() }
        connections.values.forEach { $0.close() }
        tasks.removeAll(); connections.removeAll()
        isRunning = false; isStarting = false
    }

    nonisolated static func errorBody(_ message: String, status: Int) -> [String: Any] {
        ["error": ["message": message, "type": status == 401 ? "authentication_error" : status < 500 ? "invalid_request_error" : "server_error",
                   "param": NSNull(), "code": NSNull()]]
    }

    nonisolated private static func handle(_ wire: APIConnection, token: String, cache: SpeechCacheAccess.Configuration) async {
        var streaming = false
        do {
            let request = try await wire.request()
            guard request.headers["authorization"] == "Bearer " + token else { throw APIHTTPError(status: 401, message: "Invalid local API access token.") }
            guard request.method == "GET" || request.method == "POST" else { throw APIHTTPError(status: 405, message: "Unsupported HTTP method.") }
            if request.method == "GET", request.path == "/v1/models" {
                let models = try await LocalChatService.shared.models(cache: cache)
                // Remote OpenAI models are never proxied through the local server.
                try await wire.json(["object": "list", "data": models.map { ["id": $0.id, "object": "model", "created": 0, "owned_by": "local"] as [String: Any] }])
                return
            }
            guard request.method == "POST", request.path == "/v1/chat/completions" else { throw APIHTTPError(status: 404, message: "Supported endpoints: /v1/models and /v1/chat/completions.") }
            guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true else { throw APIHTTPError(status: 415, message: "Content-Type must be application/json.") }
            let chat = try LocalChatAPIRequest.decode(request.body)
            let useMTP = UserDefaults.standard.object(forKey: LocalQwenMTP.enabledKey) as? Bool ?? true
            let models = try await LocalChatService.shared.models(cache: cache)
            guard models.contains(where: { $0.id == chat.model }) else { throw APIHTTPError(status: 404, message: "Local MLX model not found. Use GET /v1/models.") }
            let id = "chatcmpl-" + UUID().uuidString
            let created = Int(Date().timeIntervalSince1970)
            let events = AsyncThrowingStream<LocalAPIEvent, Error>.makeStream()
            let generation = Task {
                do {
                    let result = try await LocalChatService.shared.generate(modelID: chat.model,
                        messages: chat.messages.map { .init(role: $0.role == "developer" ? "system" : $0.role, content: $0.content) },
                        cache: cache, temperature: chat.temperature ?? 0.6, maxTokens: chat.maxTokens,
                        topP: chat.top_p ?? 1, stop: chat.stop ?? [], useMTP: useMTP) { text, _ in
                            if chat.stream == true { events.continuation.yield(.update(text)) }
                        }
                    events.continuation.yield(.result(result)); events.continuation.finish()
                } catch { events.continuation.finish(throwing: error) }
            }
            // Detect client disconnects even while a model is loading or generating.
            let disconnected = Task {
                _ = try? await wire.read(closeOnCancellation: false)
                generation.cancel()
            }
            defer { generation.cancel(); disconnected.cancel() }
            try await withTaskCancellationHandler {
                var delta = APITextDelta(stop: chat.stop ?? [])
                for try await event in events.stream {
                    try Task.checkCancellation()
                    switch event {
                    case .update(let text):
                        if !streaming {
                            try await wire.startSSE(); streaming = true
                            try await wire.event(chunk(id: id, created: created, model: chat.model, delta: ["role": "assistant", "content": ""]))
                        }
                        let content = delta.next(text)
                        if !content.isEmpty { try await wire.event(chunk(id: id, created: created, model: chat.model, delta: ["content": content])) }
                    case .result(let result):
                        let usage: [String: Int] = ["prompt_tokens": result.promptTokens, "completion_tokens": result.completionTokens, "total_tokens": result.promptTokens + result.completionTokens]
                        if chat.stream == true {
                            if !streaming {
                                try await wire.startSSE(); streaming = true
                                try await wire.event(chunk(id: id, created: created, model: chat.model, delta: ["role": "assistant", "content": ""]))
                            }
                            let content = delta.next(result.text, final: true)
                            if !content.isEmpty { try await wire.event(chunk(id: id, created: created, model: chat.model, delta: ["content": content])) }
                            try await wire.event(chunk(id: id, created: created, model: chat.model, delta: [:], finish: result.finishReason))
                            if chat.stream_options?.include_usage == true {
                                try await wire.event(["id": id, "object": "chat.completion.chunk", "created": created, "model": chat.model, "choices": [], "usage": usage])
                            }
                            try await wire.endSSE()
                        } else {
                            try await wire.json(["id": id, "object": "chat.completion", "created": created, "model": chat.model,
                                "choices": [["index": 0, "message": ["role": "assistant", "content": result.text], "finish_reason": result.finishReason]], "usage": usage])
                        }
                    }
                }
            } onCancel: { generation.cancel(); wire.close() }
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            let status = (error as? APIHTTPError)?.status ?? 500
            let body = errorBody(error.localizedDescription, status: status)
            if streaming { try? await wire.event(body); try? await wire.endSSE() }
            else { try? await wire.json(body, status: status) }
        }
    }

    nonisolated private static func chunk(id: String, created: Int, model: String, delta: [String: String], finish: String? = nil) -> [String: Any] {
        ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
         "choices": [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]]]
    }
    private enum LocalAPIEvent: Sendable { case update(String), result(LocalChatResult) }
}
