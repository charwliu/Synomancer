import Foundation

/// Produces short speakable segments, including Chinese text without spaces.
public enum SpeechTextChunks {
    public static func split(_ markdown: String, firstLimit: Int = 24, limit: Int = 64) -> [String] {
        precondition(firstLimit > 0 && limit > 0)
        var text = markdown.replacingOccurrences(of: "(?m)^\\s*```[^\\n]*$", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "\\[([^\\]]+)\\]\\([^)]*\\)", with: "$1", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?m)^\\s{0,3}(?:#{1,6}\\s+|>\\s*|[-+*]\\s+)", with: "", options: .regularExpression)
        for marker in ["**", "__", "`"] { text = text.replacingOccurrences(of: marker, with: "") }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var chunks: [String] = []
        while !text.isEmpty {
            let capacity = chunks.isEmpty ? firstLimit : limit
            let end = text.index(text.startIndex, offsetBy: capacity, limitedBy: text.endIndex) ?? text.endIndex
            var boundary = end
            if end != text.endIndex {
                let window = text[..<end]
                if let sentence = window.lastIndex(where: { ".!?。！？；;\n".contains($0) }),
                   text.distance(from: text.startIndex, to: sentence) >= min(8, capacity / 2) {
                    boundary = text.index(after: sentence)
                } else if let space = window.lastIndex(where: \.isWhitespace),
                          text.distance(from: text.startIndex, to: space) >= capacity / 2 {
                    boundary = text.index(after: space)
                }
            }
            let chunk = String(text[..<boundary]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty { chunks.append(chunk) }
            text = String(text[boundary...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return chunks
    }
}
