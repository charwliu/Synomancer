import Foundation
import Security
import CryptoKit

// Secrets are kept out of UserDefaults, SwiftData and exported conversations.
enum ChatSecrets {
    static let service = "Synomancer.ChatAPI"

    static func read(_ account: String) throws -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: account,
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var value: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &value)
        if status == errSecItemNotFound { return "" }
        guard status == errSecSuccess, let data = value as? Data, let text = String(data: data, encoding: .utf8) else {
            throw KeychainError(status: status)
        }
        return text
    }

    static func save(_ value: String, account: String) throws {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecAttrAccount as String: account]
        if value.isEmpty {
            let status = SecItemDelete(query as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else { throw KeychainError(status: status) }
            return
        }
        let attributes: [String: Any] = [kSecValueData as String: Data(value.utf8)]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var insertion = query.merging(attributes) { _, new in new }
            insertion[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            let added = SecItemAdd(insertion as CFDictionary, nil)
            guard added == errSecSuccess else { throw KeychainError(status: added) }
        } else if status != errSecSuccess { throw KeychainError(status: status) }
    }

    static func deleteOpenAIService(_ id: String) throws {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service, kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll]
        var value: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &value)
        if status == errSecItemNotFound { return }
        guard status == errSecSuccess, let items = value as? [[String: Any]] else { throw KeychainError(status: status) }
        for item in items {
            guard let account = item[kSecAttrAccount as String] as? String else { continue }
            let matches = id == OpenAIService.legacyID
                ? account == OpenAIConfiguration.keychainAccount || account.hasPrefix("openai.custom.")
                : account.hasPrefix("openai.service." + id + ".")
            if matches { try save("", account: account) }
        }
    }

    static func newToken() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        guard status == errSecSuccess else { throw KeychainError(status: status) }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    struct KeychainError: LocalizedError {
        let status: OSStatus
        var errorDescription: String? { "Keychain operation failed (\(status))." }
    }
}

struct OpenAIConfiguration: Sendable {
    let baseURL: URL
    let apiKey: String
    let useResponses: Bool
    let sendTemperature: Bool

    static let officialBaseURL = URL(string: "https://api.openai.com/v1")!
    static let keychainAccount = "openai"

    // Store only a validated API root. Credentials and query tokens must never enter preferences.
    static func normalizedBaseURL(_ text: String) throws -> URL {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return officialBaseURL }
        guard !text.unicodeScalars.contains(where: { CharacterSet.whitespacesAndNewlines.contains($0) || CharacterSet.controlCharacters.contains($0) }),
              var parts = URLComponents(string: text),
              let scheme = parts.scheme?.lowercased(), ["https", "http"].contains(scheme),
              let host = parts.host?.lowercased(), !host.isEmpty,
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              parts.port == nil || (1...65535).contains(parts.port!) else {
            throw OpenAIError.message("Enter an absolute HTTP(S) Base URL without credentials, query or fragment.")
        }
        let octets = host.split(separator: ".", omittingEmptySubsequences: false)
        let loopback = host == "localhost" || host == "[::1]" || host == "::1" ||
            (octets.count == 4 && octets[0] == "127" && octets.allSatisfy { part in Int(part).map { (0...255).contains($0) && String($0) == part } ?? false })
        guard scheme == "https" || loopback else {
            throw OpenAIError.message("Use HTTPS for remote APIs. HTTP is allowed only for loopback servers.")
        }
        parts.scheme = scheme
        parts.host = host
        if (scheme == "https" && parts.port == 443) || (scheme == "http" && parts.port == 80) { parts.port = nil }
        while parts.percentEncodedPath.hasSuffix("/") { parts.percentEncodedPath.removeLast() }
        guard let url = parts.url else { throw OpenAIError.message("Invalid Base URL.") }
        return url
    }

    static func keychainAccount(for baseURL: URL) -> String {
        if baseURL == officialBaseURL { return keychainAccount }
        let digest = SHA256.hash(data: Data(baseURL.absoluteString.utf8))
        return "openai.custom." + digest.map { String(format: "%02x", $0) }.joined()
    }

    static func transport(apiKey: String, baseURL: String = officialBaseURL.absoluteString,
                          useResponses: Bool = true, sendTemperature: Bool = false) throws -> Self {
        let url = try normalizedBaseURL(baseURL)
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard url != officialBaseURL || !key.isEmpty else { throw OpenAIError.message("Save an OpenAI API key in Settings.") }
        return Self(baseURL: url, apiKey: key, useResponses: useResponses, sendTemperature: sendTemperature)
    }

    static func keychainAccount(serviceID: String, baseURL: URL) -> String {
        if serviceID == OpenAIService.legacyID { return keychainAccount(for: baseURL) }
        let digest = SHA256.hash(data: Data(baseURL.absoluteString.utf8))
        return "openai.service." + serviceID + "." + digest.map { String(format: "%02x", $0) }.joined()
    }

    @MainActor static func load(serviceID: String? = nil, in preferences: UserDefaults = .standard,
        readKey: (String) throws -> String = { try ChatSecrets.read($0) }) throws -> Self {
        let id = serviceID ?? OpenAIService.legacyID
        guard let service = try OpenAIServices.load(in: preferences).first(where: { $0.id == id }) else {
            throw OpenAIError.message("This OpenAI service was removed. Select an available model.")
        }
        guard service.enabled else { throw OpenAIError.message("OpenAI is disabled for this service in Settings.") }
        let url = try normalizedBaseURL(service.baseURL)
        return try transport(apiKey: readKey(keychainAccount(serviceID: id, baseURL: url)), baseURL: url.absoluteString,
                             useResponses: service.useResponses, sendTemperature: service.sendTemperature)
    }

    @MainActor static func modelIDs(in preferences: UserDefaults = .standard) -> [String] {
        // Retained for callers of the original single-service entry point.
        guard let service = try? OpenAIServices.load(in: preferences).first(where: { $0.id == OpenAIService.legacyID }),
              service.enabled else { return [] }
        return service.modelIDs
    }

}

enum OpenAIError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { text } else { nil } }
}

// Google owns this endpoint; the wire format is Chat Completions compatible.
// Separate preferences and Keychain account prevent sending an old OpenAI key to Google.
enum GeminiConfiguration {
    static let baseURL = URL(string: "https://generativelanguage.googleapis.com/v1beta/openai")!
    static let keychainAccount = "gemini"

    static func transport(apiKey: String, sendTemperature: Bool = false) throws -> OpenAIConfiguration {
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { throw OpenAIError.message("Save a Gemini API key in Settings.") }
        return OpenAIConfiguration(baseURL: baseURL, apiKey: key, useResponses: false,
                                   sendTemperature: sendTemperature)
    }

    @MainActor static func load(in preferences: UserDefaults = .standard) throws -> OpenAIConfiguration {
        guard preferences.bool(forKey: "geminiEnabled") else {
            throw OpenAIError.message("Gemini is disabled in Settings.")
        }
        return try transport(apiKey: ChatSecrets.read(keychainAccount),
                             sendTemperature: preferences.bool(forKey: "geminiSendTemperature"))
    }

    @MainActor static func modelIDs(in preferences: UserDefaults = .standard) -> [String] {
        guard preferences.bool(forKey: "geminiEnabled") else { return [] }
        return Array(Set((preferences.string(forKey: "geminiModels") ?? "")
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty })).sorted()
    }
}
