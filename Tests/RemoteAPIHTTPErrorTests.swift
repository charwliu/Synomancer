import Foundation

enum RemoteAPIHTTPErrorTests {
    static func run() throws {
        let config = try GeminiConfiguration.transport(apiKey: "fixture-key")
        let now = Date(timeIntervalSince1970: 0)
        func message(_ body: String, status: Int = 429, headers: [String: String] = [:], configuration: OpenAIConfiguration? = nil) -> String {
            let response = HTTPURLResponse(url: config.baseURL, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
            return RemoteAPIHTTPError.make(response: response, data: Data(body.utf8), configuration: configuration ?? config, now: now).localizedDescription
        }
        let body = """
        [{"error":{"code":429,"message":"Fixture quota exceeded fixture-key","status":"RESOURCE_EXHAUSTED","details":[
          {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaMetric":"generate_content_free_tier_requests","quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier","quotaValue":"0","quotaDimensions":{"model":"fixture-model","project":"must-not-dump-this-field"}}]},
          {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"12.2s"}
        ]}}]
        """
        let quota = message(body, headers: ["Retry-After": "7"])
        for expected in ["API HTTP 429", "Fixture quota exceeded [redacted]", "RESOURCE_EXHAUSTED", "generate_content_free_tier_requests",
                         "GenerateRequestsPerDay", "quotaValue: 0", "fixture-model", "13 seconds", "Google AI Studio"] {
            precondition(quota.contains(expected))
        }
        precondition(!quota.contains("fixture-key") && !quota.contains("must-not-dump-this-field"))
        let object = message("{\"error\":{\"message\":\"fixture quota\",\"code\":\"insufficient_quota\",\"type\":\"quota_error\"}}")
        precondition(object.contains("fixture quota") && object.contains("insufficient_quota") && object.contains("quota_error"))
        let sse = message("event: error\r\ndata: {\"error\":{\"message\":\"SSE quota fixture-key\",\"status\":\"RESOURCE_EXHAUSTED\"}}\r\n\r\n")
        precondition(sse.contains("SSE quota [redacted]") && sse.contains("RESOURCE_EXHAUSTED"))
        for raw in ["", "<html>must-not-dump fixture-key</html>", "{bad-json", "[]"] {
            let fallback = message(raw)
            precondition(fallback.contains("Too many requests or API quota exhausted") && fallback.contains("Google AI Studio"))
            precondition(!fallback.contains("client error") && !fallback.contains("must-not-dump") && !fallback.contains("fixture-key"))
        }
        precondition(message("", headers: ["Retry-After": "Thu, 01 Jan 1970 00:00:30 GMT"]).contains("30 seconds"))
        precondition(message("", headers: ["Retry-After": "20"]).contains("20 seconds"))
        for delay in ["-3", "NaN", "inf", "nonsense"] {
            precondition(!message("", headers: ["Retry-After": delay]).contains("Server retry guidance"))
        }
        let custom = try OpenAIConfiguration.transport(apiKey: "fixture-key", baseURL: "https://api.fixture.invalid/v1")
        let generic = message("", configuration: custom)
        precondition(generic.contains("project's rate limits") && !generic.contains("Google AI Studio"))
        let forbidden = message("{\"error\":\"Fixture permission denied fixture-key\"}", status: 403)
        precondition(forbidden.contains("HTTP 403") && forbidden.contains("permission denied [redacted]") && !forbidden.contains("rate limits"))
        let long = message("{\"error\":{\"message\":\"" + String(repeating: "x", count: 7900) + "fixture-key" + String(repeating: "y", count: 1000) + "\"}}")
        precondition(long.count <= 8192 && !long.contains("fixture-key") && long.contains("[redacted]"))
        print("PASS: 429 object/array/SSE errors, quota metrics, RetryInfo/Retry-After seconds/date, safe empty-body fallback, non-429 errors and secret redaction")
    }
}
