//
//  AppStore.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//
import Foundation
import Combine
import SwiftUI

enum AppState {
    case chat
    case voice
}

@Observable
final class AppStore: @unchecked Sendable {
    @MainActor static let shared = AppStore()
    
    @MainActor var isReachable: Bool = false
    @MainActor var notifications: [NotificationMessage] = []
    @MainActor var menuBarIcon: String? = nil
    var appState: AppState = .chat
    
    @MainActor init() {}

    @MainActor func uiLog(message: String, status: NotificationMessage.Status) {
        notifications = [NotificationMessage(message: message, status: status)] + notifications.suffix(5)
    }
}
