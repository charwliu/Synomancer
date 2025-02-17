//
//  ManagedObjectMetadata.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//


public import SwiftData

public struct ManagedObjectMetadata: Sendable, Hashable {
    public let entityName: String
    public let persistentIdentifier: PersistentIdentifier
    public init(entityName: String, persistentIdentifier: PersistentIdentifier) {
        self.entityName = entityName
        self.persistentIdentifier = persistentIdentifier
    }
}
