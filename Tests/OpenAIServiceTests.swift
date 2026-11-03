import Foundation

@MainActor
enum OpenAIServiceTests {
    static func configurationTests() throws {
        let suite = "Synomancer.ServiceTest." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        preferences.set(true, forKey: "openAIEnabled")
        preferences.set("https://api.fixture.invalid/v1", forKey: "openAIBaseURL")
        preferences.set(" shared-model \nshared-model\n", forKey: "openAIModels")
        preferences.set(false, forKey: "openAIUseResponses")
        preferences.set(true, forKey: "openAISendTemperature")
        let base = try OpenAIConfiguration.normalizedBaseURL("https://api.fixture.invalid/v1")
        let legacyAccount = OpenAIConfiguration.keychainAccount(for: base)
        var keys = [legacyAccount: "legacy-fixture-credential"]
        var writes = 0, deletes: [String] = []
        let state = OpenAISettingsState(preferences: preferences, readKey: { keys[$0] ?? "" },
            writeKey: { key, account in keys[account] = key; writes += 1 }, deleteKeys: { id in
                deletes.append(id)
                keys = keys.filter { !$0.key.hasPrefix("openai.service." + id + ".") }
            })
        state.load()
        precondition(state.services.count == 1)
        let legacy = state.services[0]
        precondition(legacy.id == OpenAIService.legacyID && legacy.apiKey == "legacy-fixture-credential")
        precondition(legacy.profile.modelIDs == ["shared-model"] && !legacy.profile.useResponses && legacy.profile.sendTemperature)
        precondition(legacy.profile.modelName("shared-model") == "openai:shared-model")
        try state.apply()
        precondition(preferences.object(forKey: "openAIEnabled") == nil && preferences.object(forKey: "openAIBaseURL") == nil)
        precondition(writes == 0 && keys[legacyAccount] == "legacy-fixture-credential")
        state.add(); state.add()
        let a = state.services[1], b = state.services[2]
        a.profile.name = "Service A"; b.profile.name = "Service B"
        for editor in [a, b] {
            editor.setBaseURL(base.absoluteString)
            precondition(editor.apiKey.isEmpty, "New services must not inherit any other key")
            editor.profile.enabled = true
            editor.profile.models = "shared-model\nvendor/model:revision"
        }
        a.apiKey = "service-a-fixture-credential"; b.apiKey = "service-b-fixture-credential"
        a.profile.useResponses = false; b.profile.sendTemperature = true
        let aAccount = OpenAIConfiguration.keychainAccount(serviceID: a.id, baseURL: base)
        let bAccount = OpenAIConfiguration.keychainAccount(serviceID: b.id, baseURL: base)
        precondition(aAccount != bAccount && aAccount != legacyAccount)
        try state.apply()
        precondition(writes == 2 && keys[aAccount] == a.apiKey && keys[bAccount] == b.apiKey)
        let saved = preferences.data(forKey: OpenAIServices.preferencesKey)!
        let json = String(decoding: saved, as: UTF8.self)
        for key in keys.values { precondition(!json.contains(key)) }
        precondition(!json.lowercased().contains("apikey"))
        for editor in [a, b] {
            let config = try OpenAIConfiguration.load(serviceID: editor.id, in: preferences, readKey: { keys[$0] ?? "" })
            precondition(config.apiKey == editor.apiKey && config.baseURL == base)
            precondition(config.useResponses == editor.profile.useResponses && config.sendTemperature == editor.profile.sendTemperature)
        }
        a.profile.name = "Renamed A"
        try state.apply()
        precondition(a.id != b.id && a.profile.modelName("shared-model") != b.profile.modelName("shared-model"))
        precondition(writes == 2)
        a.setBaseURL("https://other.fixture.invalid/v1")
        precondition(a.apiKey.isEmpty && b.apiKey == "service-b-fixture-credential")
        a.apiKey = "unsaved-fixture-credential"
        a.setBaseURL(base.absoluteString)
        precondition(a.apiKey == "service-a-fixture-credential")
        let beforeInvalid = preferences.data(forKey: OpenAIServices.preferencesKey)
        a.apiKey = "replacement-fixture-credential"
        b.setBaseURL("https://api.fixture.invalid/v1?token=fixture-only")
        do { try state.apply(); preconditionFailure("Invalid profile published") } catch { }
        precondition(writes == 2 && preferences.data(forKey: OpenAIServices.preferencesKey) == beforeInvalid)
        b.setBaseURL(base.absoluteString)
        b.profile.name = "renamed a"
        do { try state.apply(); preconditionFailure("Ambiguous service names accepted") } catch { }
        precondition(writes == 2)
        b.profile.name = "Service B"
        a.apiKey = "service-a-fixture-credential"
        b.profile.enabled = false
        try state.apply()
        var reads = 0
        do {
            _ = try OpenAIConfiguration.load(serviceID: b.id, in: preferences, readKey: { _ in reads += 1; return "fixture-key" })
            preconditionFailure("Disabled service accepted")
        } catch { precondition(error.localizedDescription.contains("disabled") && reads == 0) }
        state.remove(a.id)
        try state.apply()
        precondition(deletes == [a.id] && keys[aAccount] == nil && keys[bAccount] == "service-b-fixture-credential")
        do {
            _ = try OpenAIConfiguration.load(serviceID: a.id, in: preferences, readKey: { _ in reads += 1; return "fixture-key" })
            preconditionFailure("Removed service fell back")
        } catch { precondition(error.localizedDescription.contains("removed") && reads == 0) }
        let reopened = OpenAISettingsState(preferences: preferences, readKey: { keys[$0] ?? "" }, writeKey: { _, _ in preconditionFailure("Read rewrote key") }, deleteKeys: { _ in preconditionFailure("Read deleted key") })
        reopened.load()
        precondition(reopened.services.count == 2 && reopened.services[1].id == b.id)
        precondition(reopened.services[1].apiKey == "service-b-fixture-credential")
        // A corrupt registry is an error, never a reason to silently use old defaults.
        let corrupt = Data("not a services list".utf8)
        preferences.set(corrupt, forKey: OpenAIServices.preferencesKey)
        preferences.set(true, forKey: "openAIEnabled")
        let invalid = OpenAISettingsState(preferences: preferences, readKey: { _ in preconditionFailure("Corrupt registry read a key") }, writeKey: { _, _ in preconditionFailure("Corrupt registry wrote a key") }, deleteKeys: { _ in preconditionFailure("Corrupt registry deleted a key") })
        invalid.load()
        precondition(invalid.loadError != nil && invalid.add() == nil)
        do { try invalid.apply(); preconditionFailure("Corrupt registry replaced") } catch { }
        precondition(preferences.data(forKey: OpenAIServices.preferencesKey) == corrupt)
        print("PASS: multi-service migration/reload, same-URL independent keys, renames, invalid/duplicate drafts, disable/remove without fallback, no credentials in preferences")
    }
}
