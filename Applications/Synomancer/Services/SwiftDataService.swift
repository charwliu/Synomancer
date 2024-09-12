//
//  SwiftDataService.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation
import SwiftData
import SwiftUI

public protocol Database: Sendable {
    func delete<T: Sendable>(_ model: T) async where T: PersistentModel
    func insert<T: Sendable>(_ model: T) async where T: PersistentModel
    func save() async throws
    func fetch<T>(_ descriptor: FetchDescriptor<T>) async throws -> [T] where T: PersistentModel
    
    func delete<T: PersistentModel>(
        where predicate: Predicate<T>?
    ) async throws
}

public extension Database {
    func fetch<T: PersistentModel>(
        where predicate: Predicate<T>?,
        sortBy: [SortDescriptor<T>]
    ) async throws -> [T] {
        try await self.fetch(FetchDescriptor<T>(predicate: predicate, sortBy: sortBy))
    }
    
    func fetch<T: PersistentModel>(
        _ predicate: Predicate<T>,
        sortBy: [SortDescriptor<T>] = []
    ) async throws -> [T] {
        try await self.fetch(where: predicate, sortBy: sortBy)
    }
    
    func fetch<T: PersistentModel>(
        _: T.Type,
        predicate: Predicate<T>? = nil,
        sortBy: [SortDescriptor<T>] = []
    ) async throws -> [T] {
        try await self.fetch(where: predicate, sortBy: sortBy)
    }
    
    func delete<T: PersistentModel>(
        model _: T.Type,
        where predicate: Predicate<T>? = nil
    ) async throws {
        try await self.delete(where: predicate)
    }
}

struct DefaultDatabase: Database {
    struct NotImplmentedError: Error {
        static let instance = NotImplmentedError()
    }
    
    static let instance = DefaultDatabase()
    
    func fetch<T>(_: FetchDescriptor<T>) async throws -> [T] where T: PersistentModel {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
    
    func delete(_: some PersistentModel) async {
        assertionFailure("No Database Set.")
    }
    
    func delete<T>(where predicate: Predicate<T>?) async throws where T : PersistentModel {
        assertionFailure("No Database Set.")
    }

    func insert(_: some PersistentModel) async {
        assertionFailure("No Database Set.")
    }
    
    func save() async throws {
        assertionFailure("No Database Set.")
        throw NotImplmentedError.instance
    }
}

private struct DatabaseKey: EnvironmentKey {
  static var defaultValue: any Database {
    DefaultDatabase.instance
  }
}

public extension EnvironmentValues {
  var database: any Database {
    get { self[DatabaseKey.self] }
    set { self[DatabaseKey.self] = newValue }
  }
}

public extension Scene {
  func database(
    _ database: any Database
  ) -> some Scene {
    self.environment(\.database, database)
  }
}

@ModelActor
public actor ModelActorDatabase: Database {
    public func delete<T>(where predicate: Predicate<T>?) async throws where T : PersistentModel {
        try self.modelContext.delete(model: T.self, where: predicate)
    }
    
    public func fetch<T: Sendable>(_ descriptor: FetchDescriptor<T>) async throws -> [T] where T : PersistentModel {
        return try self.modelContext.fetch(descriptor)
    }
    
    public func delete<T>(_ model: T) async where T : PersistentModel {
        self.modelContext.delete(model)
    }
    
    public func insert<T>(_ model: T) async where T : PersistentModel {
        self.modelContext.insert(model)
    }
    
    public func save() async throws {
        try self.modelContext.save()
    }
    
    
}

public class BackgroundDatabase: Database, @unchecked Sendable {

    convenience init(modelContainer: ModelContainer) {
        self.init {
            return ModelActorDatabase(modelContainer: modelContainer)
        }
    }
    
    private actor DatabaseContainer {
        private let factory: @Sendable () -> any Database
        private var wrappedTask: Task<any Database, Never>?
        
        fileprivate init(factory: @escaping @Sendable () -> any Database) {
            self.factory = factory
        }
        
        fileprivate var database: any Database {
            get async {
                if let wrappedTask {
                    return await wrappedTask.value
                }
                let task = Task {
                    factory()
                }
                self.wrappedTask = task
                return await task.value
            }
        }
    }
    
    private let container: DatabaseContainer
    
    private var database: any Database {
        get async {
            await container.database
        }
    }
    
    internal init(_ factory: @Sendable @escaping () -> any Database) {
        self.container = .init(factory: factory)
    }
    
    public func delete<T>(_ model: T) async where T : Sendable, T : PersistentModel {
        await self.database.delete(model)
    }
    
    public func delete(where predicate: Predicate<some PersistentModel>?) async throws {
        return try await self.database.delete(where: predicate)
    }
    
    public func fetch<T>(_ descriptor: FetchDescriptor<T>) async throws -> [T] where T: PersistentModel {
        return try await self.database.fetch(descriptor)
    }
    
    public func insert<T>(_ model: T) async where T : Sendable, T : PersistentModel {
        return await self.database.insert(model)
    }
    
    public func save() async throws {
        return try await self.database.save()
    }
}

public struct SharedDatabase : Sendable {
    
    public static let shared: SharedDatabase = .init()
    
    public let schemas: [any PersistentModel.Type]
    public var modelContainer: ModelContainer
    public let database: any Database
    
    private init(
        schemas: [any PersistentModel.Type] = [LanguageModelSD.self, ConversationSD.self, MessageSD.self, CompletionInstructionSD.self],
        modelContainer: ModelContainer? = nil,
        database: (any Database)? = nil
    ) {
        self.schemas = schemas
        let modelContainer = modelContainer ?? .forTypes(schemas)
        self.modelContainer = modelContainer
        self.database = BackgroundDatabase(modelContainer: modelContainer)
    }
}

extension ModelContainer {
    public static func forTypes(_ schemas: [any PersistentModel.Type]) -> ModelContainer {
        let schema = Schema(schemas)
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }
}

final actor SwiftDataService: Sendable {
    nonisolated let modelExecutor: any SwiftData.ModelExecutor
    nonisolated let modelContainer: SwiftData.ModelContainer
    private let modelContext: ModelContext
    
    static let shared = SwiftDataService()
    
    init() {
        let sharedModelContainer: ModelContainer = {
            let schema = Schema([
                LanguageModelSD.self,
                ConversationSD.self,
                MessageSD.self,
                CompletionInstructionSD.self
            ])
            let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)
            
            do {
                return try ModelContainer(for: schema, configurations: [modelConfiguration])
            } catch {
                fatalError("Could not create ModelContainer: \(error)")
            }
        }()
        
        self.modelContext = ModelContext(sharedModelContainer)
        self.modelContext.autosaveEnabled = false
        modelContainer = sharedModelContainer
        modelExecutor = DefaultSerialModelExecutor(modelContext: modelContext)
    }
}

// MARK: - Language Models
extension SwiftDataService {
    func fetchModels() throws -> [LanguageModelSD] {
        let sortDescriptor = SortDescriptor(\LanguageModelSD.name)
        let fetchDescriptor = FetchDescriptor<LanguageModelSD>(sortBy: [sortDescriptor])
        let models = try modelContext.fetch(fetchDescriptor)
        
        return models
    }
    
    func saveModels(models: [LanguageModelSD]) throws {
        for model in models {
            modelContext.insert(model)
        }
        
        try modelContext.saveChanges()
    }
    
    func deleteModels() throws {
        try modelContext.delete(model: LanguageModelSD.self)
        try modelContext.saveChanges()
    }
}

// MARK: - Conversations
extension SwiftDataService {
    func createConversation(_ conversation: ConversationSD) throws {
        self.modelContext.insert(conversation)
        try modelContext.saveChanges()
    }
    
    func renameConversation(_ conversation: ConversationSD) throws {
        try modelContext.saveChanges()
    }
    
    func deleteConversation(_ conversation: ConversationSD) throws {
        self.modelContext.delete(conversation)
        try modelContext.saveChanges()
    }
    
    func updateConversation(_ conversation: ConversationSD) throws {
        conversation.updatedAt = .now
        try modelContext.saveChanges()
    }
    
    func fetchConversations() throws -> [ConversationSD] {
        let sortDescriptor = SortDescriptor(\ConversationSD.updatedAt, order: .reverse)
        let fetchDescriptor = FetchDescriptor<ConversationSD>(sortBy: [sortDescriptor])
        return try modelContext.fetch(fetchDescriptor)
    }
    
    func getConversation(_ conversationId: UUID) throws -> ConversationSD? {
        let predicate = #Predicate<ConversationSD>{ $0.id == conversationId }
        let fetchDescriptor = FetchDescriptor<ConversationSD>(predicate: predicate)
        let conversations = try modelContext.fetch(fetchDescriptor)
        return conversations.first
    }
    
    func deleteConversations() throws {
        try modelContext.delete(model: ConversationSD.self)
        try modelContext.saveChanges()
    }
    
    func deleteMessages() throws {
        try modelContext.delete(model: MessageSD.self)
        try modelContext.saveChanges()
    }
    
    func deleteConversations(_ date: Date) throws {
        let predicate = #Predicate<ConversationSD>{ $0.createdAt >=  date && $0.createdAt <= date}
        try modelContext.delete(model: ConversationSD.self, where: predicate)
    }
}


// MARK: - Messages
extension SwiftDataService {
    func fetchMessages(_ conversationId: UUID) throws -> [MessageSD] {
        let predicate = #Predicate<MessageSD>{ $0.conversation?.id == conversationId }
        let sortDescriptor = SortDescriptor(\MessageSD.createdAt)
        let fetchDescriptor = FetchDescriptor<MessageSD>(predicate: predicate, sortBy: [sortDescriptor])
        return try modelContext.fetch(fetchDescriptor)
    }
    
    func updateMessage(_ message: MessageSD) throws {
        try modelContext.saveChanges()
    }
    
    func createMessage(_ mesasge: MessageSD) throws {
        self.modelContext.insert(mesasge)
        try modelContext.saveChanges()
    }
}

// MARK: - CompletionInstruction
extension SwiftDataService {
    func fetchCompletionInstructions() throws -> [CompletionInstructionSD] {
        let sortDescriptor = SortDescriptor(\CompletionInstructionSD.order, order: .forward)
        let fetchDescriptor = FetchDescriptor<CompletionInstructionSD>(sortBy: [sortDescriptor])
        return try modelContext.fetch(fetchDescriptor)
    }
    
    func updateCompletionInstructions(_ instructions: [CompletionInstructionSD]) throws {
        for index in instructions.indices {
            instructions[index].order = index
            modelContext.insert(instructions[index])
        }
        try modelContext.saveChanges()
    }
    
    func deleteCompletionInstruction(_ instruction: CompletionInstructionSD) throws {
        self.modelContext.delete(instruction)
        try modelContext.saveChanges()
    }
}

// MARK: - General
extension SwiftDataService {
    func deleteEverything() throws {
        try modelContext.delete(model: ConversationSD.self)
        try modelContext.delete(model: LanguageModelSD.self)
        try modelContext.delete(model: MessageSD.self)
        try modelContext.delete(model: CompletionInstructionSD.self)
        try modelContext.saveChanges()
    }
}
