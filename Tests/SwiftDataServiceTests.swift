import Foundation
import SwiftData

@main
struct SwiftDataServiceTests {
    @MainActor
    static func main() throws {
        if CommandLine.arguments.count == 3 {
            let schema = Schema(.all)
            let disk = ModelConfiguration(schema: schema, url: URL(fileURLWithPath: CommandLine.arguments[2]), cloudKitDatabase: .none)
            let container = try ModelContainer(for: schema, configurations: disk)
            let context = ModelContext(container)
            if CommandLine.arguments[1] == "--seed-providers" {
                for (name, provider) in [("gemini:test", ModelProvider.gemini), ("openai:legacy", .openai),
                                         ("mlx-community/test", .mlx), ("llama:latest", .ollama)] {
                    let model = LanguageModelSD(name: name, modelProvider: provider)
                    model.providerID = nil // Model rows from before the stable string field.
                    let conversation = ConversationSD(name: name)
                    conversation.model = model
                    context.insert(model); context.insert(conversation)
                }
                context.insert(LanguageModelSD(name: "local-folder", modelProvider: .mlx))
                try context.save()
            } else {
                let models = try context.fetch(FetchDescriptor<LanguageModelSD>())
                precondition(models.count == 5)
                for model in models {
                    let expected: ModelProvider = model.name.hasPrefix("gemini:") ? .gemini
                        : model.name.hasPrefix("openai:") ? .openai
                        : model.name == "llama:latest" ? .ollama : .mlx
                    precondition(model.provider == expected)
                    _ = model.displayName
                    model.provider = expected
                }
                let conversations = try context.fetch(FetchDescriptor<ConversationSD>())
                precondition(conversations.count == 4)
                try context.save()
                print("PASS: providers and conversations survive a separate process without reading legacy enum")
            }
            return
        }
        let schema = Schema(.all)
        let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: schema, configurations: configuration)
        let service = SwiftDataService(modelContainer: container)
        let model = LanguageModelSD(name: "test-model", modelProvider: .ollama)
        try service.saveModels(models: [model])
        expect(try service.fetchModels().count == 1, "model creation")

        let conversation = ConversationSD(name: "test conversation")
        conversation.model = model
        try service.createConversation(conversation)
        let later = MessageSD(content: "second", role: "assistant")
        later.createdAt = Date(timeIntervalSince1970: 2)
        later.conversation = conversation
        let earlier = MessageSD(content: "first", role: "user")
        earlier.createdAt = Date(timeIntervalSince1970: 1)
        earlier.conversation = conversation
        try service.createMessage(later)
        try service.createMessage(earlier)
        expect(try service.fetchMessages(conversation.id).map(\.content) == ["first", "second"], "message ordering")
        expect(try service.getConversation(conversation.id)?.model?.name == model.name, "relationships")

        later.content = "updated"
        try service.updateMessage(later)
        let freshContext = ModelContext(container)
        let savedMessages = try freshContext.fetch(FetchDescriptor<MessageSD>())
        expect(savedMessages.contains { $0.content == "updated" }, "saved edits visible to a fresh context")

        let first = CompletionInstructionSD(name: "first", keyboardCharacterStr: "f", instruction: "one", order: 0)
        let second = CompletionInstructionSD(name: "second", keyboardCharacterStr: "s", instruction: "two", order: 1)
        try service.updateCompletionInstructions([second, first])
        expect(try service.fetchCompletionInstructions().map(\.name) == ["second", "first"], "completion reordering")
        try service.deleteCompletionInstruction(second)
        expect(try service.fetchCompletionInstructions().count == 1, "completion deletion")
        expect(CompletionInstructionSD.sample.first !== CompletionInstructionSD.sample.first, "independent sample objects")

        try service.deleteConversation(conversation)
        expect(try service.fetchConversations().isEmpty, "conversation deletion")
        expect(try service.fetchMessages(conversation.id).isEmpty, "message cascade")
        try service.deleteEverything()
        expect(try service.fetchModels().isEmpty, "model cleanup")
        expect(try service.fetchCompletionInstructions().isEmpty, "completion cleanup")

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("synomancer-store-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let disk = ModelConfiguration(schema: schema, url: directory.appendingPathComponent("test.store"), cloudKitDatabase: .none)
        do {
            let writer = SwiftDataService(modelContainer: try ModelContainer(for: schema, configurations: disk))
            try writer.createConversation(ConversationSD(name: "persisted"))
        }
        let reopened = SwiftDataService(modelContainer: try ModelContainer(for: schema, configurations: disk))
        expect(try reopened.fetchConversations().map(\.name) == ["persisted"], "disk reopen")
        print("12 SwiftData persistence regressions passed")
    }

    private static func expect(_ condition: Bool, _ description: String) {
        precondition(condition, description)
    }
}
