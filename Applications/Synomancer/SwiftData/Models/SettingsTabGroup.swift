//
//  SettingsTabGroup.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//



import SwiftUI

struct SettingsTabGroup: Identifiable {
    public var id: UUID = .init()

    let title: LocalizedStringKey?
    let tabs: [SettingsTab]

    init(_ title: LocalizedStringKey, _ tabs: [SettingsTab]) {
        self.title = title
        self.tabs = tabs
    }

    init(_ tabs: [SettingsTab]) {
        self.title = nil
        self.tabs = tabs
    }
}
