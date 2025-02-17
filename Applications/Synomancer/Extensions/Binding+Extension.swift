//
//  Binding+Extension.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import SwiftUI

extension Binding {
    @MainActor
    func onChange(_ handler: @escaping (Value) -> Void) -> Binding<Value> {
        Binding(
            get: { self.wrappedValue },
            set: { newValue in
                self.wrappedValue = newValue
                handler(newValue)
            }
        )
    }
}
