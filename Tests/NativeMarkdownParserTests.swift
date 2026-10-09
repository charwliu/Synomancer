import Foundation

@main
@MainActor
struct NativeMarkdownParserTests {
    static func main() {
        precondition(NativeMarkdownParser.parse("hello\nworld") == [.paragraph("hello\nworld")])
        precondition(NativeMarkdownParser.parse("```swift\nlet x = 1\n```\nnext") == [.code("let x = 1", "swift"), .paragraph("next")])
        precondition(NativeMarkdownParser.parse("~~~\nstreaming") == [.code("streaming", nil)])
        precondition(NativeMarkdownParser.parse("# Title\n- item\n2. second") == [.heading("Title", 1), .listItem("item", "•"), .listItem("second", "2.")])
        precondition(NativeMarkdownParser.parse("> **quoted**\n> next") == [.quote([.paragraph("**quoted**\nnext")])])
        precondition(NativeMarkdownParser.parse("| A | B |\n| --- | :---: |\n| x\\|y | z |") == [.table([["A", "B"], ["x|y", "z"]])])
        precondition(NativeMarkdownParser.parse("````python\n```\n````") == [.code("```", "python")])
        precondition(NativeMarkdownParser.parse("#not a heading") == [.paragraph("#not a heading")])
        print("8 native Markdown parser regressions passed")
        let fixtures = [
            "# Title\n\nParagraph **bold** and 中文 e\u{301} 👨‍👩‍👧‍👦.\n\nNext paragraph\nline two\n\n",
            "| A | B |\n| --- | :---: |\n| x\\|y | z |\n\nAfter table\n",
            "````swift\nlet value = 1\n\n```\nnot closed yet\n````\n\nAfter code\n\n",
            "> quote\n> ```swift\n> let a = 1\n>\n> ```\n\n> new quote\n\nTail",
            "- item\n2. another\n\n---\n\n~~~python\n\nprint('open fence')",
            "paragraph\r\n\r\n```\r\ncode\r\n```\r\n",
            "A\n  \n# heading\n\n| pending | table |\n|---|---|\n",
        ]
        var checks = 0
        for fixture in fixtures {
            for chunkSize in [1, 4, 13, 47] {
                let cache = NativeMarkdownParseCache()
                let scalars = Array(fixture.unicodeScalars)
                var source = ""
                for start in stride(from: 0, to: scalars.count, by: chunkSize) {
                    for scalar in scalars[start..<min(start + chunkSize, scalars.count)] {
                        source.unicodeScalars.append(scalar)
                    }
                    precondition(cache.parse(source) == NativeMarkdownParser.parse(source), "Streaming Markdown changed")
                    checks += 1
                }
                // Completed views, edits, tokenizer corrections and empty resets.
                for changed in [fixture, String(fixture.dropLast(7)), "replacement\n\n" + fixture,
                                fixture.replacingOccurrences(of: "\n", with: " "), "", fixture] {
                    precondition(cache.parse(changed) == NativeMarkdownParser.parse(changed), "Edited Markdown changed")
                    checks += 1
                }
            }
        }
        // Canonically equivalent Strings can have different byte offsets.
        let unicodeCache = NativeMarkdownParseCache()
        for text in ["é\n\nA", "e\u{301}\n\nA", "e\u{301}\n\nA\n\nB"] {
            precondition(unicodeCache.parse(text) == NativeMarkdownParser.parse(text))
        }
        print("PASS: \(checks) streaming/edit checks plus Unicode normalization match full parsing")

        let section = "# Results\n\nA paragraph with **bold**, `code`, [link](https://example.com) and 中文.\n\n"
            + "| A | B |\n| --- | --- |\n| one | two |\n\n```swift\nlet n = 42\n\nprint(n)\n```\n\n> Quoted text\n> next line\n\n"
        let longAnswer = String(repeating: section, count: 120)
        let characters = Array(longAnswer)
        let updates = stride(from: 64, through: characters.count, by: 64).map { String(characters.prefix($0)) }
            + [longAnswer]
        var expectedCount: Int?
        // ABBA: measure parsing only; input snapshots are prepared before timing.
        for incremental in [false, true, true, false] {
            let cache = NativeMarkdownParseCache()
            var count = 0
            let start = ContinuousClock.now
            for source in updates {
                count += incremental ? cache.parse(source).count : NativeMarkdownParser.parse(source).count
            }
            let elapsed = start.duration(to: .now).components
            let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            if let expectedCount { precondition(count == expectedCount) }
            else { expectedCount = count }
            if incremental { precondition(cache.parse(longAnswer) == NativeMarkdownParser.parse(longAnswer)) }
            print("MARKDOWN incremental=\(incremental) updates=\(updates.count) bytes=\(longAnswer.utf8.count) seconds=\(seconds) checksum=\(count)")
        }
    }
}
