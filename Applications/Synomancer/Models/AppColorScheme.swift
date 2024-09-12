//
//  AppColorScheme.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//

import Foundation
import SwiftUI

enum AppColorScheme: String, Identifiable, CaseIterable {
    case light, dark, system
    
    var id: String {
        self.rawValue
    }
    
    var toString: String {
        switch self {
        case .system:
            "System"
        case .light:
            "Light"
        case .dark:
            "Dark"
        }
    }
    
    var toiOSFormat: ColorScheme? {
        switch self {
        case .system:
            return nil
        case .light:
            return ColorScheme.light
        case .dark:
            return ColorScheme.dark
        }
    }
}
