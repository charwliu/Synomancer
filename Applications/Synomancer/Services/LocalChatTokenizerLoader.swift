// Adapted from Apple's MLXHuggingFace TokenizerAdaptorMacro/TokenizerLoaderMacro.
// Copyright © 2026 Apple Inc. Apache-2.0; see docs/MLXModelLibrary.md.
import Foundation
import MLXLMCommon
import Tokenizers

/// Directory-only bridge to the tokenizer already shared with local speech.
/// Does not fetch a tokenizer or any other model files from the Hub.
struct LocalChatTokenizerLoader: MLXLMCommon.TokenizerLoader {
    private var preparedDirectory: URL?
    private var preparedTokenizer: (any MLXLMCommon.Tokenizer)?

    init() {}

    /// Validate chat capability before the factory allocates model weights.
    /// Reuse the same tokenizer so preflight does not parse the vocabulary twice.
    static func validated(for model: LocalChatModel) async throws -> Self {
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: model.directory)
        guard tokenizer.hasChatTemplate else {
            let pretrained = model.modelType.hasPrefix("gemma")
                && model.id.lowercased().split(whereSeparator: { "-_/".contains($0) }).contains("pt")
            throw ChatTemplateError.missing(modelID: model.id, pretrained: pretrained)
        }
        var loader = Self()
        loader.preparedDirectory = model.directory.standardizedFileURL
        loader.preparedTokenizer = Bridge(upstream: tokenizer)
        return loader
    }

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        if directory.standardizedFileURL == preparedDirectory, let preparedTokenizer {
            return preparedTokenizer
        }
        let tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return Bridge(upstream: tokenizer)
    }

    enum ChatTemplateError: LocalizedError {
        case missing(modelID: String, pretrained: Bool)

        var errorDescription: String? {
            switch self {
            case .missing(let modelID, true):
                "\(modelID) 是预训练基座模型（PT），没有聊天模板。请在 HF_HOME 中添加并选择同系列的 -it 指令模型；基座模型需要文本续写模式。"
            case .missing(let modelID, false):
                "\(modelID) 的本地缓存没有聊天模板。请选择完整的指令模型，确认缓存包含 chat_template.jinja、chat_template.json，或 tokenizer_config.json 中的 chat_template。"
            }
        }
    }

    private struct Bridge: MLXLMCommon.Tokenizer {
        let upstream: any Tokenizers.Tokenizer
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }
        func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
        func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
        var bosToken: String? { upstream.bosToken }
        var eosToken: String? { upstream.eosToken }
        var unknownToken: String? { upstream.unknownToken }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                               additionalContext: [String: any Sendable]?) throws -> [Int] {
            do { return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext) }
            catch Tokenizers.TokenizerError.missingChatTemplate { throw MLXLMCommon.TokenizerError.missingChatTemplate }
        }
    }
}
