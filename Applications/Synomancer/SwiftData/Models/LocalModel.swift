//
//  LocalModel.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//



import Defaults
import Foundation

struct LocalModel: Identifiable {
    let id = UUID()
    let group: String
    let name: String
    let url: URL

    var origin: String {
        "\(group)/\(name)"
    }
}
