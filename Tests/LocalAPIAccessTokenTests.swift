import Foundation

@MainActor
enum LocalAPIAccessTokenTests {
    private enum FixtureError: Error { case failed }

    static func run() async throws {
        var saved = "", writes = 0
        let tokenA = try LocalAPIAccessToken.generate(save: { saved = $0; writes += 1 })
        precondition(tokenA == saved && writes == 1 && tokenA.count == 64)
        precondition(tokenA.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) })
        let tokenB = try LocalAPIAccessToken.generate(save: { saved = $0; writes += 1 })
        precondition(tokenB == saved && tokenB != tokenA && writes == 2)
        var displayed = tokenA
        do {
            displayed = try LocalAPIAccessToken.generate(save: { _ in throw FixtureError.failed })
            preconditionFailure("Failed save published a token")
        } catch FixtureError.failed { }
        precondition(displayed == tokenA && saved == tokenB)
        do {
            _ = try LocalAPIAccessToken.generate(save: { _ in preconditionFailure("Failed RNG wrote a token") },
                                               random: { throw FixtureError.failed })
            preconditionFailure("RNG failure ignored")
        } catch FixtureError.failed { }
        for invalid in ["short", "fixture token with spaces", "fixture-token-123\u{00}4"] {
            do {
                _ = try LocalAPIAccessToken.persist(invalid, save: { _ in preconditionFailure("Invalid token persisted") })
                preconditionFailure("Invalid token accepted")
            } catch let error as OpenAIError { precondition(!error.localizedDescription.contains(invalid)) }
        }
        let custom = try LocalAPIAccessToken.persist(" fixture-custom-access-token ", save: { saved = $0 })
        precondition(custom == "fixture-custom-access-token" && saved == custom)
        precondition(LocalAPIAccessToken.keychainAccount != OpenAIConfiguration.keychainAccount && LocalAPIAccessToken.keychainAccount != GeminiConfiguration.keychainAccount)
        print("PASS: secure random token generation/rotation, save-before-publish, RNG/Keychain failure preservation and custom token validation (in-memory persistence)")

        let server = LocalChatAPIServer()
        var port: UInt16 = 0
        let cache = SpeechCacheAccess.Configuration(path: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path, bookmark: nil)
        for candidate in UInt16(19030)...UInt16(19049) {
            do { try await server.start(port: candidate, token: tokenA, cache: cache); port = candidate; break }
            catch { continue }
        }
        precondition(port != 0)
        defer { server.stop() }
        let session = URLSession(configuration: .ephemeral)
        defer { session.invalidateAndCancel() }
        // An unknown route distinguishes successful auth (404) from rejection (401), without touching models.
        func status(_ token: String?) async throws -> Int {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/token-regression")!)
            request.timeoutInterval = 5
            if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
            let (body, response) = try await session.data(for: request)
            let text = String(decoding: body, as: UTF8.self)
            precondition(!text.contains(tokenA) && !text.contains(tokenB))
            return (response as! HTTPURLResponse).statusCode
        }
        let initial = try await status(tokenA), missing = try await status(nil), wrong = try await status(tokenB)
        precondition(initial == 404 && missing == 401 && wrong == 401)
        try await server.start(port: port, token: tokenB, cache: cache)
        let rotated = try await status(tokenB), previous = try await status(tokenA)
        precondition(rotated == 404 && previous == 401)
        print("PASS: generated tokens authenticate on the production loopback server; restart activates the new token and rejects the old one; no model inference or personal credentials")
    }
}
