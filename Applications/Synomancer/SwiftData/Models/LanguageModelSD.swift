//
//  LanguageModelSD.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//

import Foundation
import SwiftData

@Model
final class LanguageModelSD: Identifiable {
    @Attribute(.unique) var name: String
    var isAvailable: Bool = false
    var imageSupport: Bool = false
    // Persist a primitive string. Legacy Codable enum storage can trap during
    // both reads and saves; existing rows recover their provider from the name.
    var providerID: String? = nil

    var provider: ModelProvider {
        get {
            switch providerID {
            case "mlx": return .mlx
            case "gemini": return .gemini
            case "openai": return .openai
            case "ollama": return .ollama
            default:
                if name.hasPrefix("gemini:") { return .gemini }
                if name.hasPrefix("openai:") { return .openai }
                return name.contains("/") ? .mlx : .ollama
            }
        }
        set {
            switch newValue {
            case .mlx: providerID = "mlx"
            case .gemini: providerID = "gemini"
            case .openai: providerID = "openai"
            case .ollama: providerID = "ollama"
            }
        }
    }
    
    @Relationship(deleteRule: .cascade, inverse: \ConversationSD.model)
    var conversations: [ConversationSD]? = []
    
    
    init(name: String, imageSupport: Bool = false, modelProvider: ModelProvider) {
        self.name = name
        self.imageSupport = imageSupport
        self.provider = modelProvider
    }
    
    @Transient var isNotAvailable: Bool {
        isAvailable == false
    }
}

// MARK: - Helpers
extension LanguageModelSD {
    var requestModelID: String {
        switch provider {
        case .gemini: return name.hasPrefix("gemini:") ? String(name.dropFirst(7)) : name
        case .openai: return name.hasPrefix("openai:") ? String(name.dropFirst(7)) : name
        default: return name
        }
    }

    var displayName: String {
        switch provider {
        case .gemini: return "Gemini · " + requestModelID
        case .openai: return "OpenAI · " + requestModelID
        default: return name
        }
    }

    var prettyName: String {
        if provider != .ollama { return displayName }
        guard let modelName = name.components(separatedBy: ":").first else {
            return name
        }
        
        return modelName.capitalized
    }
    
    var prettyVersion: String {
        let components = name.components(separatedBy: ":")
        if components.count >= 2 {
            return components[1]
        }
        return ""
    }
    
    var supportsImages: Bool {
        if imageSupport {
            return true
        }
        
        /// older technique to detect image modality
        /// @deprecated
        let imageSupportedModels = ["llava"]
        for modelName in imageSupportedModels {
            if name.contains(modelName) {
                return true
            }
        }
        return false
    }
    
    static var sample: [LanguageModelSD] {
        [
            .init(name: "mlx-community/Qwen2-VL-2B-Instruct-4bit", imageSupport: false, modelProvider: .mlx),
            .init(name: "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit", modelProvider: .mlx)
        ]
    }
}
