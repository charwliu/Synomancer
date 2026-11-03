import Foundation
import Security
import CryptoKit
import Observation

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

    @MainActor static func load(in preferences: UserDefaults = .standard,
        readKey: (String) throws -> String = { try ChatSecrets.read($0) }) throws -> Self {
        guard preferences.bool(forKey: "openAIEnabled") else {
            throw OpenAIError.message("OpenAI is disabled in Settings.")
        }
        let url = try normalizedBaseURL(preferences.string(forKey: "openAIBaseURL") ?? "")
        return try transport(apiKey: readKey(keychainAccount(for: url)), baseURL: url.absoluteString,
            useResponses: preferences.object(forKey: "openAIUseResponses") as? Bool ?? true,
            sendTemperature: preferences.bool(forKey: "openAISendTemperature"))
    }

    @MainActor static func modelIDs(in preferences: UserDefaults = .standard) -> [String] {
        guard preferences.bool(forKey: "openAIEnabled") else { return [] }
        return Array(Set((preferences.string(forKey: "openAIModels") ?? "")
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty })).sorted()
    }
}

// The URL is a draft until Apply/Save. Switching endpoints never carries a key along.
@Observable @MainActor
final class OpenAISettingsState {
    private(set) var baseURLText = OpenAIConfiguration.officialBaseURL.absoluteString
    var apiKey = ""
    private(set) var keyLoadError: String?
    @ObservationIgnored private let preferences: UserDefaults
    @ObservationIgnored private let readKey: (String) throws -> String
    @ObservationIgnored private let writeKey: (String, String) throws -> Void
    @ObservationIgnored private var keyEndpoint: URL?
    @ObservationIgnored private var originalKey: String?

    init(preferences: UserDefaults = .standard,
         readKey: @escaping (String) throws -> String = { try ChatSecrets.read($0) },
         writeKey: @escaping (String, String) throws -> Void = { try ChatSecrets.save($0, account: $1) }) {
        self.preferences = preferences
        self.readKey = readKey
        self.writeKey = writeKey
    }

    func load() {
        setBaseURL(preferences.string(forKey: "openAIBaseURL") ?? "")
    }

    func setBaseURL(_ text: String) {
        baseURLText = text
        let endpoint = try? OpenAIConfiguration.normalizedBaseURL(text)
        guard endpoint != keyEndpoint || endpoint == nil else { return }
        apiKey = ""
        originalKey = nil
        keyLoadError = nil
        keyEndpoint = endpoint
        guard let endpoint else { return }
        do {
            let stored = try readKey(OpenAIConfiguration.keychainAccount(for: endpoint))
            apiKey = stored
            originalKey = stored
        } catch {
            keyLoadError = "Could not read this endpoint's API key from Keychain. Re-enter the key before enabling it."
        }
    }

    func apply() throws {
        let endpoint = try OpenAIConfiguration.normalizedBaseURL(baseURLText)
        guard keyEndpoint == endpoint else { throw OpenAIError.message("Reload the endpoint settings before saving.") }
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if preferences.bool(forKey: "openAIEnabled") {
            if keyLoadError != nil && key.isEmpty { throw OpenAIError.message(keyLoadError!) }
            _ = try OpenAIConfiguration.transport(apiKey: key, baseURL: endpoint.absoluteString)
        }
        // A failed read must not turn an untouched empty field into a Keychain deletion.
        if apiKey != (originalKey ?? "") {
            try writeKey(key, OpenAIConfiguration.keychainAccount(for: endpoint))
            originalKey = key
            apiKey = key
            keyLoadError = nil
        }
        preferences.set(endpoint.absoluteString, forKey: "openAIBaseURL")
        baseURLText = endpoint.absoluteString
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
