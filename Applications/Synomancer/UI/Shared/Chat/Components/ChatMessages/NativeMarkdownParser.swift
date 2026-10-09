import Foundation

indirect enum NativeMarkdownBlock: Equatable {
    case paragraph(String)
    case heading(String, Int)
    case code(String, String?)
    case listItem(String, String)
    case quote([NativeMarkdownBlock])
    case table([[String]])
    case divider
}

/// Block structure stays separate from inline formatting, which Foundation handles.
enum NativeMarkdownParser {
    static func parse(_ source: String) -> [NativeMarkdownBlock] {
        let lines = source.components(separatedBy: "\n")
        var blocks: [NativeMarkdownBlock] = []
        var paragraph: [String] = []
        var index = 0
        func flush() {
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: "\n"))) }
            paragraph.removeAll(keepingCapacity: true)
        }
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let fence = fenceStart(trimmed) {
                flush()
                index += 1
                var code: [String] = []
                while index < lines.count {
                    let closing = lines[index].trimmingCharacters(in: .whitespaces)
                    if closing.count >= fence.count && closing.allSatisfy({ $0 == fence.character }) {
                        index += 1
                        break
                    }
                    code.append(lines[index])
                    index += 1
                }
                blocks.append(.code(code.joined(separator: "\n"), fence.language))
                continue
            }
            if trimmed.isEmpty {
                flush()
            } else if index + 1 < lines.count && isTableSeparator(lines[index + 1]) && line.contains("|") {
                flush()
                var rows = [tableCells(line)]
                index += 2
                while index < lines.count && lines[index].contains("|") && !lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                    rows.append(tableCells(lines[index]))
                    index += 1
                }
                blocks.append(.table(rows))
                continue
            } else if trimmed.hasPrefix(">") {
                flush()
                var quoted: [String] = []
                while index < lines.count && lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(">") {
                    var value = lines[index].trimmingCharacters(in: .whitespaces)
                    value.removeFirst()
                    if value.first == " " { value.removeFirst() }
                    quoted.append(value)
                    index += 1
                }
                blocks.append(.quote(parse(quoted.joined(separator: "\n"))))
                continue
            } else if let heading = heading(trimmed) {
                flush()
                blocks.append(.heading(heading.text, heading.level))
            } else if ["---", "***", "___"].contains(trimmed) {
                flush()
                blocks.append(.divider)
            } else if let item = listItem(trimmed) {
                flush()
                blocks.append(.listItem(item.text, item.marker))
            } else {
                paragraph.append(line)
            }
            index += 1
        }
        flush()
        return blocks
    }

    fileprivate static func fenceStart(_ line: String) -> (character: Character, count: Int, language: String?)? {
        guard let character = line.first, character == "`" || character == "~" else { return nil }
        let count = line.prefix(while: { $0 == character }).count
        guard count >= 3 else { return nil }
        let language = String(line.dropFirst(count)).trimmingCharacters(in: .whitespaces)
        return (character, count, language.isEmpty ? nil : language)
    }

    private static func heading(_ line: String) -> (text: String, level: Int)? {
        let level = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level), line.dropFirst(level).first == " " else { return nil }
        return (String(line.dropFirst(level + 1)), level)
    }

    private static func listItem(_ line: String) -> (text: String, marker: String)? {
        if let first = line.first, "-*+".contains(first), line.dropFirst().first == " " {
            return (String(line.dropFirst(2)), "•")
        }
        let digits = line.prefix(while: { $0.isNumber })
        let rest = line.dropFirst(digits.count)
        guard !digits.isEmpty, let delimiter = rest.first, delimiter == "." || delimiter == ")", rest.dropFirst().first == " " else { return nil }
        return (String(rest.dropFirst(2)), "\(digits).")
    }

    private static func tableCells(_ line: String) -> [String] {
        var value = line.trimmingCharacters(in: .whitespaces)
        if value.hasPrefix("|") { value.removeFirst() }
        if value.hasSuffix("|") { value.removeLast() }
        var cells: [String] = []
        var cell = ""
        var escaped = false
        for character in value {
            if escaped {
                cell.append(character)
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "|" {
                cells.append(cell.trimmingCharacters(in: .whitespaces))
                cell = ""
            } else {
                cell.append(character)
            }
        }
        if escaped { cell.append("\\") }
        cells.append(cell.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let cells = tableCells(line)
        return !cells.isEmpty && cells.allSatisfy { cell in
            let value = cell.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            return value.count >= 3 && value.allSatisfy { $0 == "-" }
        }
    }
}

/// Per-view cache. Only blank lines outside fenced code seal a prefix: tables,
/// paragraphs and quoted blocks can still change shape while their tail grows.
/// Edits, truncation and non-append tokenizer corrections rebuild from source.
@MainActor
final class NativeMarkdownParseCache {
    private var source = ""
    private var committedBytes = 0
    private var committed: [NativeMarkdownBlock] = []
    private var latest: [NativeMarkdownBlock] = []

    func parse(_ next: String) -> [NativeMarkdownBlock] {
        let appending = next.utf8.starts(with: source.utf8)
        if appending && next.utf8.count == source.utf8.count { return latest }
        if !appending {
            committedBytes = 0
            committed.removeAll(keepingCapacity: true)
        }
        source = next
        var tail = String(decoding: next.utf8.dropFirst(committedBytes), as: UTF8.self)
        let sealedBytes = Self.sealedPrefixBytes(tail)
        if sealedBytes > 0 {
            committed += NativeMarkdownParser.parse(String(decoding: tail.utf8.prefix(sealedBytes), as: UTF8.self))
            committedBytes += sealedBytes
            tail = String(decoding: tail.utf8.dropFirst(sealedBytes), as: UTF8.self)
        }
        latest = committed + NativeMarkdownParser.parse(tail)
        return latest
    }

    private static func sealedPrefixBytes(_ text: String) -> Int {
        let bytes = text.utf8
        var start = bytes.startIndex
        var sealed = bytes.startIndex
        var fence: (character: Character, count: Int)?
        for index in bytes.indices where bytes[index] == 10 {
            let line = String(decoding: bytes[start..<index], as: UTF8.self)
                .trimmingCharacters(in: .whitespaces)
            let next = bytes.index(after: index)
            if let active = fence {
                if line.count >= active.count && line.allSatisfy({ $0 == active.character }) {
                    fence = nil
                }
            } else if let opening = NativeMarkdownParser.fenceStart(line) {
                fence = (opening.character, opening.count)
            } else if line.isEmpty {
                sealed = next
            }
            start = next
        }
        return bytes.distance(from: bytes.startIndex, to: sealed)
    }
}
