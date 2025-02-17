//
//  Assert.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

public import Foundation

@inlinable internal func assert(isMainThread: Bool) {
    assert(isMainThread == Thread.isMainThread)
}

@inlinable internal func assertionFailure(
    error: any Error,
    file: StaticString = #file,
    line: UInt = #line
) { assertionFailure(error.localizedDescription, file: file, line: line) }
