//
//  LanguageModelStore.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//



import Foundation
import SwiftData

@Observable @MainActor
final class LanguageModelStore {
    static let shared = LanguageModelStore(swiftDataService: SwiftDataService.shared)
    
    private var swiftDataService: SwiftDataService
    private let preferences: UserDefaults
    private var discoveryID = UUID()
    @MainActor var models: [LanguageModelSD] = []
    @MainActor var supportsImages = false
    @MainActor var selectedModel: LanguageModelSD?
    
    init(swiftDataService: SwiftDataService, preferences: UserDefaults = .standard) {
        self.swiftDataService = swiftDataService
        self.preferences = preferences
    }
    
    @MainActor
    func setModel(model: LanguageModelSD?) {
        if let model, let available = models.first(where: { $0.name == model.name && $0.modelProvider == model.modelProvider }) {
            selectedModel = available
        } else {
            let preferred = preferences.string(forKey: "defaultMLXModel")
            selectedModel = models.first(where: { $0.name == preferred }) ?? models.first
        }
        supportsImages = selectedModel?.imageSupport ?? false
    }

    @MainActor
    func setModel(modelName: String) {
        setModel(model: models.first(where: { $0.name == modelName }))
    }

    func loadModels() async throws {
        let request = UUID()
        discoveryID = request
        do {
            let remoteIDs = GeminiConfiguration.modelIDs(in: preferences)
            var localModels: [LocalChatModel] = []
            do {
                let cache = remoteIDs.isEmpty
                    ? try await SpeechCacheAccess.authorizedConfiguration(in: preferences)
                    : SpeechCacheAccess.configuration(in: preferences)
                localModels = try await LocalChatService.shared.models(cache: cache)
            } catch {
                // A remote model remains usable when a removable local cache is absent.
                if remoteIDs.isEmpty { throw error }
            }
            try Task.checkCancellation()
            guard discoveryID == request else { return }
            let stored = try swiftDataService.fetchModels()
            var available: [LanguageModelSD] = []
            for local in localModels {
                let model = stored.first(where: { $0.name == local.id && $0.modelProvider == .mlx })
                    ?? LanguageModelSD(name: local.id, imageSupport: local.supportsImages, modelProvider: .mlx)
                model.imageSupport = local.supportsImages
                model.isAvailable = true
                available.append(model)
            }
            for id in remoteIDs {
                let name = "gemini:" + id
                let model = stored.first(where: { $0.name == name && $0.modelProvider == .gemini })
                    ?? LanguageModelSD(name: name, imageSupport: false, modelProvider: .gemini)
                model.isAvailable = true
                available.append(model)
            }
            // Old provider rows remain attached to their saved conversations.
            // Never delete them during discovery: their relationship cascades.
            try swiftDataService.saveModels(models: available)
            models = available
            setModel(model: selectedModel)
            let preferred = preferences.string(forKey: "defaultMLXModel")
            if !available.contains(where: { $0.name == preferred }), let selectedModel {
                preferences.set(selectedModel.name, forKey: "defaultMLXModel")
            }
            AppStore.shared.isReachable = !available.isEmpty
        } catch {
            guard discoveryID == request else { return }
            models = []
            selectedModel = nil
            supportsImages = false
            AppStore.shared.isReachable = false
            throw error
        }
    }

    func deleteAllModels() async throws {
        await MainActor.run {
            self.models = []
            self.selectedModel = nil
            self.supportsImages = false
            AppStore.shared.isReachable = false
        }
        try swiftDataService.deleteModels()
    }
}
