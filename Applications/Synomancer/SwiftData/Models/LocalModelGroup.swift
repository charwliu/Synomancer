//
//  LocalModelGroup.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//



import Foundation

struct LocalModelGroup: Identifiable {
    let id = UUID()
    let name: String
    var models: [LocalModel]
}
