import Foundation

// Both OpenAI Responses and Chat Completions use SSE; a line is not an event.
struct ChatSSEParser {
    private var lines: [String] = []
    mutating func consume(_ line: String) -> String? {
        if line.isEmpty {
            defer { lines.removeAll(keepingCapacity: true) }
            return lines.isEmpty ? nil : lines.joined(separator: "\n")
        }
        if line.hasPrefix("data:") {
            var value = String(line.dropFirst(5))
            if value.first == " " { value.removeFirst() }
            lines.append(value)
        }
        return nil
    }
}

// AsyncBytes.lines skips blank lines on some Foundation versions. SSE requires them.
struct ChatSSEDecoder {
    private var line = Data()
    private var parser = ChatSSEParser()
    private var eventBytes = 0
    private var firstLine = true
    mutating func consume(_ byte: UInt8) throws -> String? {
        if byte != 10 {
            line.append(byte)
            guard line.count + eventBytes <= 1048576 else { throw OpenAIError.message("API stream event exceeds 1 MiB.") }
            return nil
        }
        if line.last == 13 { line.removeLast() }
        guard var value = String(data: line, encoding: .utf8) else { throw OpenAIError.message("Invalid UTF-8 in API stream.") }
        if firstLine { firstLine = false; if value.first == "\u{FEFF}" { value.removeFirst() } }
        eventBytes += line.count
        line.removeAll(keepingCapacity: true)
        if value.isEmpty { eventBytes = 0 }
        return parser.consume(value)
    }
}

actor OpenAIChatService {
    static let shared = OpenAIChatService()
    private let session: URLSession
    init(session: URLSession = .shared) { self.session = session }

    func generate(modelID: String, messages: [LocalChatMessage], configuration: OpenAIConfiguration,
                  temperature: Float = 0.6,
                  onUpdate: @escaping @Sendable (String, Int) -> Void) async throws -> LocalChatResult {
        guard !messages.contains(where: { $0.image != nil }) else {
            throw OpenAIError.message("OpenAI image input is not enabled in this version.")
        }
        let history = messages.map { ["role": $0.role, "content": $0.content] }
        var body: [String: Any] = ["model": modelID, "stream": true]
        if configuration.useResponses {
            body["input"] = history
            body["store"] = false
        } else {
            body["messages"] = history
            body["stream_options"] = ["include_usage": true]
        }
        // Reasoning models can reject temperature. Only send it when enabled explicitly.
        if configuration.sendTemperature { body["temperature"] = temperature }
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent(configuration.useResponses ? "responses" : "chat/completions"))
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("Bearer " + configuration.apiKey, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw OpenAIError.message("Invalid API response.") }
        guard (200..<300).contains(http.statusCode) else {
            var data = Data()
            for try await byte in bytes {
                data.append(byte)
                if data.count >= 65536 { break }
            }
            let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let detail = (json?["error"] as? [String: Any])?["message"] as? String
            throw OpenAIError.message("API HTTP \(http.statusCode): \(detail ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode))")
        }
        guard http.value(forHTTPHeaderField: "Content-Type")?.lowercased().contains("text/event-stream") == true else {
            throw OpenAIError.message("The API did not return an SSE stream.")
        }
        var parser = ChatSSEDecoder()
        var text = "", updates = 0, promptTokens = 0, completionTokens = 0, finishReason = "stop"
        var completed = false
        let started = Date()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard let event = try parser.consume(byte) else { continue }
            if event == "[DONE]" { completed = true; break }
            guard let data = event.data(using: .utf8),
                  let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw OpenAIError.message("Malformed API stream event.")
            }
            if let error = value["error"] as? [String: Any] {
                throw OpenAIError.message(error["message"] as? String ?? "API stream failed.")
            }
            if configuration.useResponses {
                let type = value["type"] as? String ?? ""
                switch type {
                case "response.output_text.delta", "response.refusal.delta": text += value["delta"] as? String ?? ""
                case "response.completed":
                    completed = true
                    let response = value["response"] as? [String: Any]
                    let usage = response?["usage"] as? [String: Any]
                    promptTokens = usage?["input_tokens"] as? Int ?? 0
                    completionTokens = usage?["output_tokens"] as? Int ?? 0
                case "response.failed", "error":
                    let response = value["response"] as? [String: Any]
                    let error = response?["error"] as? [String: Any]
                    throw OpenAIError.message(error?["message"] as? String ?? value["message"] as? String ?? "API response failed.")
                case "response.incomplete":
                    let response = value["response"] as? [String: Any]
                    let details = response?["incomplete_details"] as? [String: Any]
                    if details?["reason"] as? String == "max_output_tokens" { completed = true; finishReason = "length" }
                    else { throw OpenAIError.message("The API response was incomplete.") }
                default: break
                }
            } else {
                if let usage = value["usage"] as? [String: Any] {
                    promptTokens = usage["prompt_tokens"] as? Int ?? 0
                    completionTokens = usage["completion_tokens"] as? Int ?? 0
                }
                if let choice = (value["choices"] as? [[String: Any]])?.first {
                    let delta = choice["delta"] as? [String: Any]
                    text += delta?["content"] as? String ?? delta?["refusal"] as? String ?? ""
                    if let reason = choice["finish_reason"] as? String { finishReason = reason }
                }
            }
            updates += 1
            onUpdate(text, updates)
            if completed { break }
        }
        try Task.checkCancellation()
        guard completed else { throw OpenAIError.message("The API stream ended before completion. Please retry.") }
        return LocalChatResult(text: text, tokensPerSecond: Double(completionTokens) / max(Date().timeIntervalSince(started), 0.001),
            promptTokens: promptTokens, completionTokens: completionTokens, finishReason: finishReason)
    }

    func models(configuration: OpenAIConfiguration) async throws -> [String] {
        var request = URLRequest(url: configuration.baseURL.appendingPathComponent("models"))
        request.setValue("Bearer " + configuration.apiKey, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 30
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw OpenAIError.message("Could not list API models. Check the endpoint and API key.")
        }
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return ((object?["data"] as? [[String: Any]]) ?? []).compactMap { $0["id"] as? String }.sorted()
    }
}
