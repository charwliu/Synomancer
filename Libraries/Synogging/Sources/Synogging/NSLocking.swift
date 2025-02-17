//
//  NSLocking.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

import Foundation

#if !canImport(os)
extension NSLocking {
    internal func withLock<R>(_ body: () throws -> R) rethrows -> R {
        lock()
        defer {
            self.unlock()
        }
        return try body()
    }
}
#endif
