//
//  CompletionsStore.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation
import SwiftUI

@Observable
final class CompletionsStore: @unchecked Sendable {
    static let shared = CompletionsStore(swiftDataService: SwiftDataService.shared)
    private let swiftDataService: SwiftDataService
    
    var completions: [CompletionInstructionSD] = []
    
    init(swiftDataService: SwiftDataService) {
        self.swiftDataService = swiftDataService
        load()
    }
    
    func save() {
        Task {
            try? await swiftDataService.updateCompletionInstructions(completions)
        }
    }
    
    func delete(_ completion: CompletionInstructionSD) {
        Task {
            try? await swiftDataService.deleteCompletionInstruction(completion)
            load()
        }
    }
    
    func load() {
        Task {
            var loadedCompletions: [CompletionInstructionSD] = []
            loadedCompletions = (try? await SwiftDataService.shared.fetchCompletionInstructions()) ?? []
            
            if loadedCompletions.count == 0 {
                try? await SwiftDataService.shared.updateCompletionInstructions(CompletionInstructionSD.sample)
                loadedCompletions = (try? await SwiftDataService.shared.fetchCompletionInstructions()) ?? []
            }
            
            withAnimation {
                completions = loadedCompletions
            }
        }
    }
}
