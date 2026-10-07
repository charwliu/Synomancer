import Foundation

@main
struct SpeechTextChunksTests {
    static func main() {
        func compact(_ value: String) -> String { String(value.filter { !$0.isWhitespace }) }
        let chinese = String(repeating: "这是生成的中文答案，需要完整朗读，不能遗漏任何段落。", count: 30)
        let english = String(repeating: "This generated answer should be read aloud without losing words or waiting for the entire answer. ", count: 20)
        for text in [chinese, english, "emoji 👨‍👩‍👧‍👦 中文和 English " + chinese] {
            let parts = SpeechTextChunks.split(text)
            precondition(parts.count > 1 && parts[0].count <= 24)
            precondition(parts.dropFirst().allSatisfy { $0.count <= 64 })
            precondition(compact(parts.joined(separator: " ")) == compact(text))
        }
        let markdown = "# 标题\n- **重点**\n[说明](https://example.com)\n```swift\nprint(1)\n```"
        precondition(compact(SpeechTextChunks.split(markdown).joined()) == "标题重点说明print(1)")
        precondition(SpeechTextChunks.split("   \n").isEmpty)
        print("PASS: long Chinese/English answers, short first segment, Unicode and Markdown cleanup")
    }
}
