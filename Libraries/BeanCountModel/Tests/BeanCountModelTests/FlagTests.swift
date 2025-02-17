//
//  FlagTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing

struct FlagTests {
    @Test func testDescription() async throws {
        let complete = Flag.complete
        #expect(String(describing: complete) == "*")
        let incomplete = Flag.incomplete
        #expect(String(describing: incomplete) == "!")
    }
}
