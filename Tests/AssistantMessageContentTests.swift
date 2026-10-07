import Foundation

@main
struct AssistantMessageContentTests {
    static func main() {
        func check(_ raw: String, reasoning: String?, answer: String, thinking: Bool = false, streaming: Bool = false) {
            let parsed = AssistantMessageContent(raw, streaming: streaming)
            precondition(parsed.reasoning == reasoning, "Reasoning mismatch: \(raw)")
            precondition(parsed.answer == answer, "Answer mismatch: \(raw)")
            precondition(parsed.isThinking == thinking, "State mismatch: \(raw)")
        }
        check("普通回答", reasoning: nil, answer: "普通回答")
        check(" \n<think>分析问题。\n</think>\n\n正式答案。", reasoning: "分析问题。", answer: "正式答案。")
        check("<think>analysis</think>answer", reasoning: "analysis", answer: "answer")
        check("旧的思考内容\n</think>\n\n旧答案", reasoning: "旧的思考内容", answer: "旧答案")
        check("</think>\n答案", reasoning: "", answer: "答案")
        check("<think>unfinished", reasoning: "unfinished", answer: "", thinking: true)
        check("<think>\n\n</think>\nAnswer", reasoning: "", answer: "Answer")
        for length in 1..<"<think>".count {
            check(String("<think>".prefix(length)), reasoning: "", answer: "", thinking: true, streaming: true)
        }
        for length in 1..<"</think>".count {
            check("<think>analysis\n" + "</think>".prefix(length), reasoning: "analysis", answer: "", thinking: true, streaming: true)
        }
        for literal in [
            "The closing tag is </think>.", "Use `<think>` and `</think>`.",
            "```html\n</think>\n```\nExample", "~~~html\n</think>\n~~~",
            "    </think>\nIndented code", "\\</think>\nEscaped tag",
            "``a `\n</think>\n``\nLiteral code", "<thinkable>example"
        ] {
            check(literal, reasoning: nil, answer: literal, streaming: true)
        }
        check("<think>Use `</think>` in code.\n```html\n</think>\n```\nDone.\n</think>\nAnswer",
              reasoning: "Use `</think>` in code.\n```html\n</think>\n```\nDone.", answer: "Answer")
        // After the initial protocol block, literal examples belong to the answer.
        check("<think>analysis</think>Use `</think>`.", reasoning: "analysis", answer: "Use `</think>`.")
        let continuation = "分析问题。\n</think>\n\n正式答案。"
        let framed = AssistantMessageContent.framed(continuation, primed: true)
        check(framed, reasoning: "分析问题。", answer: "正式答案。")
        precondition(AssistantMessageContent.framed("", primed: true).isEmpty)
        precondition(AssistantMessageContent.framed(continuation, primed: false) == continuation)
        precondition(AssistantMessageContent.framed(framed, primed: true) == framed)
        // Simulate every split of a full decoded snapshot, including UTF-8 text.
        for length in 1...continuation.count {
            let snapshot = AssistantMessageContent.framed(String(continuation.prefix(length)), primed: true)
            let parsed = AssistantMessageContent(snapshot, streaming: true)
            precondition(!(parsed.reasoning ?? "").contains("</think>"))
            precondition(!parsed.answer.contains("</think>"))
            if parsed.isThinking { precondition(parsed.answer.isEmpty) }
        }
        print("PASS: paired/legacy reasoning, all delimiter splits, cancelled thinking, Markdown literals and answer-only text")
    }
}
