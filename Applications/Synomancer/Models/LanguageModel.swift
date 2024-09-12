//
//  LanguageModel.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation

struct LanguageModel {
    var name: String
    var provider: ModelProvider
    var imageSupport: Bool
}

enum ModelProvider: Codable {
    case ollama
}
