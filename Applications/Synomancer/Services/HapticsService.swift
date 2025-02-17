//
//  HapticsService.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


#if os(iOS)
import UIKit

final class Haptics: Sendable {
    static let shared = Haptics()
    
    private init() { }

    private func play(_ feedbackStyle: UIImpactFeedbackGenerator.FeedbackStyle) async {
        let vibrations = UserDefaults.standard.bool(forKey: "vibrations")
        if vibrations {
            await UIImpactFeedbackGenerator(style: feedbackStyle).impactOccurred()
        }
    }
    
    private func notify(_ feedbackType: UINotificationFeedbackGenerator.FeedbackType) async {
        let vibrations = UserDefaults.standard.bool(forKey: "vibrations")
        if vibrations {
            await UINotificationFeedbackGenerator().notificationOccurred(feedbackType)
        }
    }
    
    func lightTap() {
        Task {
            await play(.light)
        }
    }
    
    func mediumTap() async {
        Task {
            await play(.medium)
        }
    }
}
#else
final class Haptics: Sendable {
    static let shared = Haptics()
    func lightTap() {}
    func mediumTap() async {}
}
#endif

