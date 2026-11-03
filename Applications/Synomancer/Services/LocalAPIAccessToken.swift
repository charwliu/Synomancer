import Foundation

enum LocalAPIAccessToken {
    static let keychainAccount = "local-server"

    static func generate(save: (String) throws -> Void = { try ChatSecrets.save($0, account: keychainAccount) },
                         random: () throws -> String = { try ChatSecrets.newToken() }) throws -> String {
        // Publish the replacement to the UI only after it is durably saved.
        try persist(random(), save: save)
    }

    static func persist(_ token: String,
                        save: (String) throws -> Void = { try ChatSecrets.save($0, account: keychainAccount) }) throws -> String {
        let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.count >= 16,
              value.rangeOfCharacter(from: .whitespacesAndNewlines.union(.controlCharacters)) == nil else {
            throw OpenAIError.message("Use an access token of at least 16 characters without spaces or control characters.")
        }
        try save(value)
        return value
    }
}
