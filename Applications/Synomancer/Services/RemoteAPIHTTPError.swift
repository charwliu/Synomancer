import Foundation

// Decode only known error fields, never dump an HTML page, request or raw response body.
enum RemoteAPIHTTPError {
    static func make(response: HTTPURLResponse, data: Data, configuration: OpenAIConfiguration,
                     now: Date = Date()) -> OpenAIError {
        let error = decode(Data(data.prefix(65536)))
        var parts: [String] = []
        let message = (error?["message"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let message, !message.isEmpty { parts.append(message) }
        else if response.statusCode == 429 { parts.append("Too many requests or API quota exhausted.") }
        else { parts.append(HTTPURLResponse.localizedString(forStatusCode: response.statusCode)) }

        for field in ["status", "code", "type"] {
            if let value = error?[field] as? String, !value.isEmpty { parts.append("\(field): \(value)") }
        }
        let details = error?["details"] as? [[String: Any]] ?? []
        for detail in details where (detail["@type"] as? String)?.hasSuffix("google.rpc.QuotaFailure") == true {
            for violation in (detail["violations"] as? [[String: Any]] ?? []).prefix(4) {
                var fields = ["quotaMetric", "quotaId", "quotaValue"].compactMap { key -> String? in
                    if let value = violation[key] as? String { return "\(key): \(value)" }
                    if let value = violation[key] as? NSNumber { return "\(key): \(value.stringValue)" }
                    return nil
                }
                if let model = (violation["quotaDimensions"] as? [String: Any])?["model"] as? String {
                    fields.append("model: " + model)
                }
                if !fields.isEmpty { parts.append(fields.joined(separator: ", ")) }
            }
        }
        var delays: [Double] = []
        if let header = response.value(forHTTPHeaderField: "Retry-After"), let delay = retryAfter(header, now: now) {
            delays.append(delay)
        }
        for detail in details where (detail["@type"] as? String)?.hasSuffix("google.rpc.RetryInfo") == true {
            if let value = detail["retryDelay"] as? String, value.hasSuffix("s"),
               let delay = seconds(String(value.dropLast())) { delays.append(delay) }
        }
        if let delay = delays.max() {
            parts.append("Server retry guidance: wait at least \(String(format: "%.0f", ceil(delay))) seconds.")
        }
        if response.statusCode == 429 {
            if configuration.baseURL.host?.lowercased() == "generativelanguage.googleapis.com" {
                parts.append("Check this model's project rate limits, quota and billing in Google AI Studio.")
            } else {
                parts.append("Check this API project's rate limits, quota and billing.")
            }
        }
        var text = "API HTTP \(response.statusCode): " + parts.joined(separator: "\n")
        // Redact before limiting the UI message so truncation cannot expose part of the key.
        if !configuration.apiKey.isEmpty { text = text.replacingOccurrences(of: configuration.apiKey, with: "[redacted]") }
        return .message(String(text.prefix(8192)))
    }

    private static func decode(_ data: Data) -> [String: Any]? {
        if let value = try? JSONSerialization.jsonObject(with: data), let error = object(value) { return error }
        // Streaming gateways can return an SSE error envelope even with a non-2xx status.
        var decoder = ChatSSEDecoder()
        var framed = data
        framed.append(Data("\n\n".utf8))
        for byte in framed {
            guard let payload = try? decoder.consume(byte), let bytes = payload.data(using: .utf8),
                  let value = try? JSONSerialization.jsonObject(with: bytes) else { continue }
            if let error = object(value) { return error }
        }
        return nil
    }

    private static func object(_ value: Any, depth: Int = 0) -> [String: Any]? {
        guard depth < 5 else { return nil }
        if let array = value as? [Any] {
            return array.lazy.compactMap { object($0, depth: depth + 1) }.first
        }
        guard let value = value as? [String: Any] else { return nil }
        if let error = value["error"] as? [String: Any] { return error }
        if let message = value["error"] as? String { return ["message": message] }
        if value["message"] is String || value["status"] is String { return value }
        return nil
    }

    private static func seconds(_ value: String) -> Double? {
        guard let seconds = Double(value.trimmingCharacters(in: .whitespacesAndNewlines)),
              seconds.isFinite, seconds >= 0 else { return nil }
        return seconds
    }

    private static func retryAfter(_ value: String, now: Date) -> Double? {
        if let delay = seconds(value) { return delay }
        let format = DateFormatter()
        format.locale = Locale(identifier: "en_US_POSIX")
        format.timeZone = TimeZone(secondsFromGMT: 0)
        format.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        return format.date(from: value).map { max(0, $0.timeIntervalSince(now)) }
    }
}
