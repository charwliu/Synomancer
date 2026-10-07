//
//  ChatMessageView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



import SwiftUI

struct ChatMessageView: View {
    @Environment(\.colorScheme) var colorScheme
    @StateObject private var speechSynthesizer = SpeechSynthesizer.shared
    var message: MessageSD
    var showLoader: Bool = false
    var userInitials: String
    @Binding var editMessage: MessageSD?
    @State private var mouseHover = false
    
    var roleName: String  {
        let userInitialsNotEmpty = userInitials != "" ? userInitials : "AM"
        return message.role == "user" ? userInitialsNotEmpty.uppercased() : "AI"
    }
    
    var image: Image? {
        message.image != nil ? Image(data: message.image!) : nil
    }
    
    var body: some View {
        VStack(alignment: .trailing, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Group {
                    if message.role == "user" {
                        Spacer()
                    } else {
                        if showLoader {
                            ProgressView().controlSize(.small)
                                .frame(width: 24, height: 24)
                        } else {
                            Image("logo-nobg")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 24, height: 24)
                        }
                    }
                }
                .offset(CGSize(width: 0, height: 6))
                
                VStack(alignment: .leading) {
                    Group {
                        if message.role == "user" {
                            NativeMarkdownView(content: message.content)
                        } else {
                            AssistantResponseView(content: message.content, streaming: showLoader)
                        }
                    }
#if os(macOS)
                        .textSelection(.enabled)
#endif
                    
                    if let uiImage = image {
                        uiImage
                            .resizable()
                            .scaledToFit()
                            .frame(width: 100)
                            .clipShape(RoundedRectangle(cornerRadius: 5))
                        
                    }
                }
                .if(message.role == "user", transform: { v in
                    v.padding()
                        .background(RoundedRectangle(cornerRadius: 25).fill(.regularMaterial))
                })
                
                if message.role != "user" {
                    Spacer()
                }
            }
#if os(macOS)
            HStack(spacing: 0) {
                /// Copy button
                Button(action: {Clipboard.shared.setString(message.displayContent)}) {
                    Image(systemName: "doc.on.doc")
                        .padding(8)
                }
                .buttonStyle(GrowingButton())
                .clipShape(RoundedRectangle(cornerRadius: 10))
                
                /// Play button
                Button(action: {
                    Task {
                        await speechSynthesizer.speak(text: message.displayContent)
                    }
                }) {
                    Image(systemName: "speaker.wave.2.fill")
                        .frame(width: 10)
                        .padding(8)
                }
                .buttonStyle(GrowingButton())
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .disabled(message.displayContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                
                /// Edit button
                Button(action: {editMessage = message}) {
                    Image(systemName: "pencil")
                        .padding(8)
                }
                .buttonStyle(GrowingButton())
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .showIf(message.role == "user")
            }
            .opacity(mouseHover ? 1 : 0.0001)
            
#endif
        }
#if os(macOS)
        .onHover { over in
            withAnimation(.easeInOut(duration: 0.3)) {
                mouseHover = over
            }
        }
#endif
    }
}

#Preview(traits: .sizeThatFitsLayout) {
    VStack {
        ChatMessageView(message: MessageSD.sample[0], userInitials: "AM", editMessage: .constant(nil))
            
        ChatMessageView(message: MessageSD.sample[1], userInitials: "AM", editMessage: .constant(nil))
            
        ChatMessageView(message: MessageSD(content: "```python \nprint(5+5)\n```", role: "ai"), showLoader: true, userInitials: "AM", editMessage: .constant(nil))
            
    }
}
