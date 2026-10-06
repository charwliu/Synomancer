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

    private static func fenceStart(_ line: String) -> (character: Character, count: Int, language: String?)? {
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
