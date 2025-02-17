//
//  DatabaseKey.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

import Foundation
import SwiftData
public import SwiftUI


fileprivate struct DefaultDatabase: Database {
    
    func delete(_: (some PersistentModel).Type, withID _: PersistentIdentifier) async -> Bool {
        assertionFailure("No Database Set.")
        return false
    }
    
    func delete(where _: Predicate<some PersistentModel>?) async throws {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
    
    func insert(_ closuer: @escaping @Sendable () -> some PersistentModel) async -> PersistentIdentifier {
        assertionFailure("No Database Set.")
        fatalError("No Database Set.")
    }

    func fetch<T, U>(_ : @escaping @Sendable () -> FetchDescriptor<T>, with _: @escaping @Sendable ([T]) throws -> U) async throws -> U where T : PersistentModel, U : Sendable {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
    
    func fetch<T, U, V>(
        _ selectDescriptorA: @escaping @Sendable () -> FetchDescriptor<T>,
        _ selectDescriptorB: @escaping @Sendable () -> FetchDescriptor<U>,
        with closure: @escaping @Sendable ([T], [U]) throws -> V
    ) async throws -> V where T : PersistentModel, U : PersistentModel, V : Sendable {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
    
    func get<T, U>(for _: PersistentIdentifier, with _: @escaping @Sendable (T?) throws -> U) async throws -> U where T : PersistentModel, U : Sendable {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
    
    private struct NotImplmentedError: Error { static let instance = NotImplmentedError() }
    
    static let instance = DefaultDatabase()
    
    func transaction(_: @escaping @Sendable (ModelContext) throws -> Void) async throws {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
}


extension EnvironmentValues {
    @Entry public var database: any Database = DefaultDatabase.instance
}

extension Scene {
    public func database(_ database: any Database) -> some Scene {
        environment(\.database, database)
    }
}

