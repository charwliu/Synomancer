//
//  FetchDescriptor.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

public import Foundation

public import SwiftData

extension FetchDescriptor {
    public init(predicate: Predicate<T>? = nil, sortBy: [SortDescriptor<T>] = [], fetchLimit: Int?) {
        self.init(predicate: predicate, sortBy: sortBy)
        
        self.fetchLimit = fetchLimit
    }
    
    public init(model: ModelID<T>) {
        let persistentIdentifier = model.persistentIdentifier
        self.init(
            predicate: #Predicate<T> { $0.persistentModelID == persistentIdentifier },
            fetchLimit: 1
        )
    }
}
