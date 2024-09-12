//
//  DisplayStyle.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//


import Foundation

enum DisplayStyle: String, CaseIterable, Identifiable {
    case plain, markdown
    var id: Self { self }
}
