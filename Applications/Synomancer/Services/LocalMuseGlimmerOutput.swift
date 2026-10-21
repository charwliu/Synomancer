// ATEM frame rules adapted from mlx-swift-lm 3.32.3 OnyxStreamAdapter.
// Copyright © 2026 Apple Inc. Apache-2.0; see docs/MLXModelLibrary-LICENSE.
import Foundation
import MLXLMCommon

/// Decode control token IDs before detokenization. Header/tool frames never
/// enter the answer; App reasoning uses its existing collapsible think block.
struct LocalMuseGlimmerOutput {
    private enum Recipient { case reasoning, user, hidden }
    private struct Frame {
        let recipient: Recipient
        var tokens: [Int] = []
        var closed = false
    }
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let start: Int
    private let message: Int
    private let endings: Set<Int>
    private var header: [Int]? = []
    private var primed = true
    private var frames: [Frame] = []

    init(tokenizer: any MLXLMCommon.Tokenizer) throws {
        self.tokenizer = tokenizer
        func control(_ name: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(name),
                  tokenizer.encode(text: name, addSpecialTokens: false) == [id] else {
                throw OutputError.invalidControlToken(name)
            }
            return id
        }
        start = try control("<|start|>")
        message = try control("<|message|>")
        endings = try Set(["<|eom|>", "<|eot|>", "<|end_of_text|>"].map(control))
    }

    mutating func append(_ token: Int) {
        if token == start {
            closeFrame()
            header = []
            primed = false
        } else if endings.contains(token) {
            closeFrame()
            header = nil
        } else if let tokens = header {
            if token == message {
                let text = (primed ? "assistant" : "") + tokenizer.decode(tokenIds: tokens)
                let fields = text.split(whereSeparator: \.isWhitespace)
                let target = fields.dropFirst().first { $0.hasPrefix("to=") }.map { String($0.dropFirst(3)) }
                let recipient: Recipient = fields.first != "assistant" ? .hidden
                    : target == "self" ? .reasoning : target == nil || target == "user" ? .user : .hidden
                frames.append(Frame(recipient: recipient))
                header = nil
            } else if tokens.count < 256 {
                header?.append(token)
            } else {
                header = nil
            }
        } else if !frames.isEmpty, !frames[frames.count - 1].closed {
            frames[frames.count - 1].tokens.append(token)
        }
    }

    func text(includeReasoning: Bool) -> String {
        var answer = ""
        var reasoning: [String] = []
        var thinking = false
        for frame in frames {
            switch frame.recipient {
            case .user:
                answer += tokenizer.decode(tokenIds: frame.tokens)
            case .reasoning where includeReasoning:
                reasoning.append(tokenizer.decode(tokenIds: frame.tokens))
                thinking = !frame.closed
            default: break
            }
        }
        guard !reasoning.isEmpty else { return answer }
        return "<think>\n" + reasoning.joined(separator: "\n")
            + (thinking ? "" : "\n</think>\n") + answer
    }

    private mutating func closeFrame() {
        if !frames.isEmpty { frames[frames.count - 1].closed = true }
    }

    enum OutputError: LocalizedError {
        case invalidControlToken(String)
        var errorDescription: String? {
            switch self {
            case .invalidControlToken(let token): "Muse Glimmer 分词器缺少有效的 ATEM 控制 token：\(token)"
            }
        }
    }
}
