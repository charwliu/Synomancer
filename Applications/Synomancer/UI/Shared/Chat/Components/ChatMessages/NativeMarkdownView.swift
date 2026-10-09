import SwiftUI

struct NativeMarkdownView: View {
    let content: String
    @State private var parser = NativeMarkdownParseCache()

    var body: some View {
        NativeMarkdownBlocks(blocks: parser.parse(content))
    }
}

private struct NativeMarkdownBlocks: View {
    let blocks: [NativeMarkdownBlock]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                NativeMarkdownBlockView(block: block).equatable()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }

}

private struct NativeMarkdownBlockView: View, Equatable {
    let block: NativeMarkdownBlock

    private func inline(_ value: String) -> Text {
        let text = try? AttributedString(
            markdown: value,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
        return Text(text ?? AttributedString(value))
    }

    @ViewBuilder
    var body: some View {
        switch block {
        case .paragraph(let value):
            inline(value)
        case .heading(let value, let level):
            inline(value).font(headingFont(level)).fontWeight(.semibold)
        case .code(let value, let language):
            CodeBlockView(code: value, language: language)
        case .listItem(let value, let marker):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(marker).frame(minWidth: 18, alignment: .trailing)
                inline(value)
            }
        case .quote(let nested):
            HStack(alignment: .top, spacing: 12) {
                RoundedRectangle(cornerRadius: 2).fill(.secondary).frame(width: 3)
                NativeMarkdownBlocks(blocks: nested).foregroundStyle(.secondary)
            }.fixedSize(horizontal: false, vertical: true)
        case .table(let rows):
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                inline(cell).fontWeight(rowIndex == 0 ? .semibold : .regular)
                            }
                        }
                    }
                }.padding(12)
            }.background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
        case .divider:
            Divider()
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: .title
        case 2: .title2
        case 3: .title3
        default: .headline
        }
    }
}
