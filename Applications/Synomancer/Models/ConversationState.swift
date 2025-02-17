//
//  ConversationState.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation

enum ConversationState: Equatable, Sendable {
    case loading
    case completed
    case error(message: String)
}
