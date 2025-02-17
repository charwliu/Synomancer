//
//  ModelContext.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

import Foundation
import SwiftData
import Synogging

extension ModelContext: @retroactive Synogging.Loggable {
    public typealias LoggingSystemType = SynoLogging
    public static var loggingCategory: SynoLogging.Category { .data }
}

extension ModelContext: Loggable {
    func existingModel<T>(for objectID: PersistentIdentifier) throws -> T? where T: PersistentModel {
        if let registered: T = registeredModel(for: objectID) {
            return registered
        }
        if let notRegistered: T = model(for: objectID) as? T {
            return notRegistered
        }
        
        let fetchDescriptor = FetchDescriptor<T>(
            predicate: #Predicate { $0.persistentModelID == objectID }
        )
        
        return try fetch(fetchDescriptor).first
    }
}
