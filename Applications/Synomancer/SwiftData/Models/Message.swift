//
//  Message.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//



import Foundation
import SwiftData

@Model
class Message {
    enum Role: String, Codable {
        case user
        case assistant
        case system
    }

    var role: Role
    var content: String
    var isComplete: Bool
    var timestamp: Date
    var error: String?

    init(
        role: Role,
        content: String = "",
        isComplete: Bool = false,
        timestamp: Date = Date()
    ) {
        self.role = role
        self.content = content
        self.isComplete = isComplete
        self.timestamp = timestamp
    }
}
