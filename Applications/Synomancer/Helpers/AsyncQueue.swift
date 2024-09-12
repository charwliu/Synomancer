//
//  AsyncQueue.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import Foundation

actor AsyncQueue<T> {
    private var items: [T] = []

    func enqueue(_ item: T) {
        items.append(item)
    }

    func dequeueAll() -> [T] {
        defer { items.removeAll() }
        return items
    }
}

