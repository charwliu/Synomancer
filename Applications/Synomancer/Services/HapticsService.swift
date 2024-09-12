//
//  HapticsService.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


#if os(iOS)
import UIKit

class Haptics: @unchecked Sendable {
    static let shared = Haptics()
    
    private init() { }

    private func play(_ feedbackStyle: UIImpactFeedbackGenerator.FeedbackStyle) {
        let vibrations = UserDefaults.standard.bool(forKey: "vibrations")
        if vibrations {
            UIImpactFeedbackGenerator(style: feedbackStyle).impactOccurred()
        }
    }
    
    private func notify(_ feedbackType: UINotificationFeedbackGenerator.FeedbackType) {
        let vibrations = UserDefaults.standard.bool(forKey: "vibrations")
        if vibrations {
            UINotificationFeedbackGenerator().notificationOccurred(feedbackType)
        }
    }
    
    func lightTap() {
        play(.light)
    }
    
    func mediumTap() {
        play(.medium)
    }
}
#else
class Haptics: @unchecked Sendable {
    static let shared = Haptics()
    func lightTap() {}
    func mediumTap() async {}
}
#endif

