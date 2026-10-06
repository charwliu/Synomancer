//
//  CodeBlockView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



import SwiftUI

struct CodeBlockView: View {
    var code: String
    var language: String?
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(language.flatMap { $0.isEmpty ? nil : $0 } ?? "code")
                    .font(.system(size: 13, design: .monospaced))
                    .fontWeight(.semibold)
                Spacer()
                
                Button(action: {
                    Clipboard.shared.setString(code)
                }) {
                    Image(systemName: "doc.on.doc")
                        .padding(7)
                }
                .buttonStyle(GrowingButton())
            }
            .padding(.horizontal)
            .padding(.vertical, 4)
            .background(Color.secondary.opacity(0.1))
            
            Divider()
            
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(16)
            }
        }
        .background(Color.secondary.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))

    }
}
