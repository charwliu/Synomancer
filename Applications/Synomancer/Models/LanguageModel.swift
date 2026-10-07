//
//  LanguageModel.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation

struct LanguageModel: Sendable {
    var name: String
    var provider: ModelProvider
    var imageSupport: Bool
}

enum ModelProvider: Codable, Sendable {
    // Retained to decode previously saved conversations; no runtime backend.
    case ollama
    case mlx
    case openai
}
