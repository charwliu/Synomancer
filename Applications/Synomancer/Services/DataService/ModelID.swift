//
//  ModelID.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//


import SwiftData

public struct ModelID<T: PersistentModel>: Sendable, Identifiable {
    public var id: PersistentIdentifier.ID { return persistentIdentifier.id }
    public let persistentIdentifier: PersistentIdentifier
    
    enum Error: Swift.Error { case notFound(PersistentIdentifier) }
}

extension ModelID where T: PersistentModel {
    public init(_ model: T) { self.init(persistentIdentifier: model.persistentModelID) }
    
    internal static func ifMap(_ model: T?) -> ModelID? { model.map(self.init) }
}
