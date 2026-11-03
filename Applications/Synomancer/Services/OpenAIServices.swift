import Foundation
import Observation

// This persisted record intentionally has no API key. IDs survive renames and endpoint edits.
struct OpenAIService: Codable, Equatable, Identifiable, Sendable {
    static let legacyID = "00000000-0000-0000-0000-000000000001"
    let id: String
    var name = "OpenAI"
    var baseURL = OpenAIConfiguration.officialBaseURL.absoluteString
    var models = ""
    var enabled = false
    var useResponses = true
    var sendTemperature = false

    init(id: String = UUID().uuidString.lowercased(), name: String = "OpenAI",
         baseURL: String = OpenAIConfiguration.officialBaseURL.absoluteString, models: String = "",
         enabled: Bool = false, useResponses: Bool = true, sendTemperature: Bool = false) {
        self.id = id
        self.name = name
        self.baseURL = baseURL
        self.models = models
        self.enabled = enabled
        self.useResponses = useResponses
        self.sendTemperature = sendTemperature
    }

    var modelIDs: [String] {
        Array(Set(models.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })).sorted()
    }

    func modelName(_ modelID: String) -> String {
        id == Self.legacyID ? "openai:" + modelID : "openai-service:" + id + ":" + modelID
    }
}

@MainActor
enum OpenAIServices {
    static let preferencesKey = "openAIServices"

    static func load(in preferences: UserDefaults = .standard) throws -> [OpenAIService] {
        if preferences.object(forKey: preferencesKey) != nil {
            guard let data = preferences.data(forKey: preferencesKey),
                  let services = try? JSONDecoder().decode([OpenAIService].self, from: data) else {
                throw OpenAIError.message("Could not read OpenAI services. Existing configuration was preserved.")
            }
            return try validated(services)
        }
        // Read the old settings until the first successful save; never touch their key during migration.
        return try validated([OpenAIService(id: OpenAIService.legacyID, name: "OpenAI",
            baseURL: preferences.string(forKey: "openAIBaseURL") ?? OpenAIConfiguration.officialBaseURL.absoluteString,
            models: preferences.string(forKey: "openAIModels") ?? "",
            enabled: preferences.bool(forKey: "openAIEnabled"),
            useResponses: preferences.object(forKey: "openAIUseResponses") as? Bool ?? true,
            sendTemperature: preferences.bool(forKey: "openAISendTemperature"))])
    }

    static func validated(_ services: [OpenAIService]) throws -> [OpenAIService] {
        var ids = Set<String>(), names = Set<String>()
        return try services.map { service in
            var service = service
            guard UUID(uuidString: service.id)?.uuidString.lowercased() == service.id,
                  ids.insert(service.id).inserted else { throw OpenAIError.message("Invalid or duplicate OpenAI service ID.") }
            service.name = service.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !service.name.isEmpty, service.name.count <= 80,
                  !service.name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
                  names.insert(service.name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))).inserted else {
                throw OpenAIError.message("Give each OpenAI service a distinct name (1–80 characters).")
            }
            service.baseURL = try OpenAIConfiguration.normalizedBaseURL(service.baseURL).absoluteString
            service.models = service.modelIDs.joined(separator: "\n")
            return service
        }
    }

    static func save(_ services: [OpenAIService], in preferences: UserDefaults = .standard) throws {
        let data = try JSONEncoder().encode(validated(services))
        preferences.set(data, forKey: preferencesKey)
        for key in ["openAIEnabled", "openAIBaseURL", "openAIModels", "openAIUseResponses", "openAISendTemperature"] {
            preferences.removeObject(forKey: key)
        }
    }
}

// Each editor owns an independent draft and key. Invalid URL drafts never enter UserDefaults.
@Observable @MainActor
final class OpenAIServiceSettingsState: Identifiable {
    var profile: OpenAIService
    let id: String
    var apiKey = ""
    private(set) var keyLoadError: String?
    @ObservationIgnored private let readKey: (String) throws -> String
    @ObservationIgnored private let writeKey: (String, String) throws -> Void
    @ObservationIgnored private var keyEndpoint: URL?
    @ObservationIgnored private var originalKey: String?
    @ObservationIgnored private var needsKeyLoad = false

    init(profile: OpenAIService, readKey: @escaping (String) throws -> String,
         writeKey: @escaping (String, String) throws -> Void) {
        self.profile = profile
        self.id = profile.id
        self.readKey = readKey
        self.writeKey = writeKey
        setBaseURL(profile.baseURL)
    }

    func setBaseURL(_ text: String) {
        editBaseURL(text)
        finishBaseURLEditing()
    }

    // Keystrokes clear the old key immediately but never block on Keychain I/O.
    func editBaseURL(_ text: String) {
        profile.baseURL = text
        let endpoint = try? OpenAIConfiguration.normalizedBaseURL(text)
        guard endpoint != keyEndpoint || endpoint == nil else { return }
        apiKey = ""
        originalKey = nil
        keyLoadError = nil
        keyEndpoint = endpoint
        needsKeyLoad = endpoint != nil
    }

    func finishBaseURLEditing() {
        guard needsKeyLoad, let endpoint = keyEndpoint,
              (try? OpenAIConfiguration.normalizedBaseURL(profile.baseURL)) == endpoint else { return }
        needsKeyLoad = false
        // A newly entered key belongs to this destination; do not replace it on blur.
        guard apiKey.isEmpty else { return }
        do {
            let stored = try readKey(OpenAIConfiguration.keychainAccount(serviceID: id, baseURL: endpoint))
            apiKey = stored
            originalKey = stored
        } catch {
            keyLoadError = "Could not read this service's API key from Keychain. Re-enter the key before enabling it."
        }
    }

    func validateKey() throws {
        finishBaseURLEditing()
        let endpoint = try OpenAIConfiguration.normalizedBaseURL(profile.baseURL)
        guard profile.id == id, keyEndpoint == endpoint else { throw OpenAIError.message("Reload the service settings before saving.") }
        if profile.enabled {
            if keyLoadError != nil && apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { throw OpenAIError.message(keyLoadError!) }
            _ = try OpenAIConfiguration.transport(apiKey: apiKey, baseURL: endpoint.absoluteString)
        }
    }

    func saveKey() throws {
        let endpoint = try OpenAIConfiguration.normalizedBaseURL(profile.baseURL)
        // A failed read must not turn an untouched empty field into a Keychain deletion.
        if apiKey != (originalKey ?? "") {
            let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            try writeKey(key, OpenAIConfiguration.keychainAccount(serviceID: id, baseURL: endpoint))
            originalKey = key
            apiKey = key
            keyLoadError = nil
        }
    }
}

@Observable @MainActor
final class OpenAISettingsState {
    private(set) var services: [OpenAIServiceSettingsState] = []
    private(set) var loadError: String?
    @ObservationIgnored private let preferences: UserDefaults
    @ObservationIgnored private let readKey: (String) throws -> String
    @ObservationIgnored private let writeKey: (String, String) throws -> Void
    @ObservationIgnored private let deleteKeys: (String) throws -> Void
    @ObservationIgnored private var savedIDs = Set<String>()
    @ObservationIgnored private var loaded = false

    init(preferences: UserDefaults = .standard,
         readKey: @escaping (String) throws -> String = { try ChatSecrets.read($0) },
         writeKey: @escaping (String, String) throws -> Void = { try ChatSecrets.save($0, account: $1) },
         deleteKeys: @escaping (String) throws -> Void = { try ChatSecrets.deleteOpenAIService($0) }) {
        self.preferences = preferences
        self.readKey = readKey
        self.writeKey = writeKey
        self.deleteKeys = deleteKeys
    }

    func load() {
        guard !loaded else { return }
        do {
            let profiles = try OpenAIServices.load(in: preferences)
            services = profiles.map { OpenAIServiceSettingsState(profile: $0, readKey: readKey, writeKey: writeKey) }
            savedIDs = Set(profiles.map(\.id))
            loaded = true
        } catch { loadError = error.localizedDescription }
    }

    @discardableResult func add() -> String? {
        guard loaded, loadError == nil else { return nil }
        var index = services.count + 1
        while services.contains(where: { $0.profile.name == "Service \(index)" }) { index += 1 }
        let profile = OpenAIService(name: "Service \(index)")
        services.append(OpenAIServiceSettingsState(profile: profile, readKey: readKey, writeKey: writeKey))
        return profile.id
    }

    func remove(_ id: String) { services.removeAll { $0.id == id } }

    func apply() throws {
        guard loaded, loadError == nil else {
            throw OpenAIError.message(loadError ?? "OpenAI services have not finished loading.")
        }
        let profiles = try OpenAIServices.validated(services.map(\.profile))
        // Validate every draft before any Keychain mutation or metadata publication.
        for service in services { try service.validateKey() }
        for service in services { try service.saveKey() }
        for removed in savedIDs.subtracting(profiles.map(\.id)) { try deleteKeys(removed) }
        try OpenAIServices.save(profiles, in: preferences)
        for (editor, profile) in zip(services, profiles) { editor.profile = profile }
        savedIDs = Set(profiles.map(\.id))
    }
}
