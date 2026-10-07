//
//  MessageSD.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation
import SwiftData

@Model
final class MessageSD: Identifiable {
    @Attribute(.unique) var id: UUID = UUID()
    
    var content: String
    var role: String
    var done: Bool = false
    var error: Bool = false
    var createdAt: Date = Date.now
    @Attribute(.externalStorage) var image: Data?
    
    @Relationship var conversation: ConversationSD?
        
    
    init(content: String, role: String, done: Bool = false, error: Bool = false, image: Data? = nil) {
        self.content = content
        self.role = role
        self.done = done
        self.error = error
        self.conversation = conversation
        self.image = image
    }

    @Transient var model: String {
        conversation?.model?.name ?? ""
    }
}

extension MessageSD {
    var displayContent: String {
        role == "user" || role == "system" ? content : AssistantMessageContent(content, streaming: !done).answer
    }

    static var sample: [MessageSD] {
        [
            .init(content: "How many quarks there are in SM?", role: "user"),
            .init(content: "There are 6 quarks in SM, each of them has an antiparticle and colour.", role: "assistant"),
            .init(content: "How elementary particle is defined in mathematics?", role: "user"),
            .init(content: "Elementary particle is defined as an irreducible representation of the poincase group.", role: "assistant")
        ]
    }
}
