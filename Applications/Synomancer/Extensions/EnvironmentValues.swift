//
//  EnvironmentValues.swift
//  Synomancer
//
//  Created by Liu Wei on 9/20/24.
//
import SwiftUI

public extension EnvironmentValues {
    @Entry var tintColor: Color = .accentColor
    @Entry var hoveringOverLuminareItem: Bool = false
    @Entry var clickedOutsideFlag: Bool = false
    @Entry var currentlyScrolling: Bool  = false
}
