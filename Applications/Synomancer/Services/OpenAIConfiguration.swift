import Foundation
import Security

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

    @MainActor static func load(in preferences: UserDefaults = .standard) throws -> Self {
        guard preferences.bool(forKey: "openAIEnabled") else { throw OpenAIError.message("OpenAI is disabled in Settings.") }
        let address = preferences.string(forKey: "openAIBaseURL") ?? "https://api.openai.com/v1"
        guard let url = URL(string: address), let host = url.host, url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil,
              url.scheme == "https" || (url.scheme == "http" && ["127.0.0.1", "localhost", "::1"].contains(host)) else {
            throw OpenAIError.message("Use an HTTPS API base URL, or HTTP on localhost.")
        }
        let key = try ChatSecrets.read("openai")
        guard !key.isEmpty else { throw OpenAIError.message("Save an OpenAI API key in Settings.") }
        return Self(baseURL: url, apiKey: key,
                    useResponses: (preferences.string(forKey: "openAIProtocol") ?? "responses") == "responses",
                    sendTemperature: preferences.bool(forKey: "openAISendTemperature"))
    }

    @MainActor static func modelIDs(in preferences: UserDefaults = .standard) -> [String] {
        guard preferences.bool(forKey: "openAIEnabled") else { return [] }
        let raw = preferences.string(forKey: "openAIModels") ?? ""
        return Array(Set(raw.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty })).sorted()
    }
}

enum OpenAIError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { text } else { nil } }
}
