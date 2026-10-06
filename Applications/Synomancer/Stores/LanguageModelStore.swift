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
    @MainActor var models: [LanguageModelSD] = []
    @MainActor var supportsImages = false
    @MainActor var selectedModel: LanguageModelSD?
    
    init(swiftDataService: SwiftDataService) {
        self.swiftDataService = swiftDataService
    }
    
    @MainActor
    func setModel(model: LanguageModelSD?) {
        if let model = model {
            // check if model still exists
            if models.contains(model) {
                selectedModel = model
            }
        } else {
            selectedModel = nil
        }
    }
    
    @MainActor
    func setModel(modelName: String) {
        for model in models {
            if model.name == modelName {
                setModel(model: model)
                return
            }
        }
        if let lastModel = models.last {
            setModel(model: lastModel)
        }
    }
    
    func loadModels() async throws {
        let remoteModels = try await OllamaService.shared.getModels()
        try swiftDataService.saveModels(models: remoteModels.map{LanguageModelSD(name: $0.name, imageSupport: $0.imageSupport, modelProvider: .ollama)})
        
        let storedModels = (try? swiftDataService.fetchModels()) ?? []
        
        await MainActor.run {
            let remoteModelNames = remoteModels.map { $0.name }
            let models = storedModels.filter{remoteModelNames.contains($0.name)}
            self.models = models
        }
    }
    
    func deleteAllModels() async throws {
        await MainActor.run {
            self.models = []
        }
        try swiftDataService.deleteModels()
    }
}
