import Foundation

@main
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
    }
}
