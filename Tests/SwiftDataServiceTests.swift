import Foundation
import SwiftData

@main
struct SwiftDataServiceTests {
    @MainActor
    static func main() throws {
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
