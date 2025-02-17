//
//  ModelActorDatabase.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

public import Foundation
public import SwiftData

@ModelActor
public actor ModelActorDatabase: Database {
    
    public func delete<T: PersistentModel>(_: T.Type, withID id: PersistentIdentifier) async -> Bool {
        guard let model: T = self.modelContext.registeredModel(for: id) else { return false }
        self.modelContext.delete(model)
        return true
    }

    public func delete<T>(where predicate: Predicate<T>?) async throws where T: PersistentModel {
        try self.modelContext.delete(model: T.self, where: predicate)
    }
    
    public func insert(_ closuer: @escaping @Sendable () -> some PersistentModel) async -> PersistentIdentifier {
        let model = closuer()
        self.modelContext.insert(model)
        return model.persistentModelID
    }
    
    public func fetch<T, U>(
        _ selectDescriptor: @escaping @Sendable () -> FetchDescriptor<T>,
        with closure: @escaping @Sendable ([T]) throws -> U
    ) async throws -> U where T: PersistentModel, U: Sendable {
        let models = try self.modelContext.fetch(selectDescriptor())
        return try closure(models)
    }
    
    public func fetch<T: PersistentModel, U: PersistentModel, V: Sendable>(
        _ selectDescriptorA: @escaping @Sendable () -> FetchDescriptor<T>,
        _ selectDescriptorB: @escaping @Sendable () -> FetchDescriptor<U>,
        with closure: @escaping @Sendable ([T], [U]) throws -> V
    ) async throws -> V {
        let a = try self.modelContext.fetch(selectDescriptorA())
        let b = try self.modelContext.fetch(selectDescriptorB())
        return try closure(a, b)
    }
    
    public func get<T, U>(
        for objectID: PersistentIdentifier,
        with closure: @escaping @Sendable (T?) throws -> U
    ) async throws -> U where T: PersistentModel, U: Sendable {
        let model: T? = try self.modelContext.existingModel(for: objectID)
        return try closure(model)
    }
    
    public static var loggingCategory: SynoLogging.Category { .data }
    
    public func transaction(_ block: @escaping @Sendable (ModelContext) throws -> Void) async throws
    {
        assert(isMainThread: false)
        
        try self.modelContext.transaction {
            assert(isMainThread: false)
            try block(modelContext)
        }
    }
}
