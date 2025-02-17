//
//  CollectionTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

@testable import BeanCountParser
import Testing

class CollectionTests {
    @Test func safeArray()  async throws{
        var array = [String]()
        #expect(array[safe: 0] == nil)
        array.append("value")
        #expect(array[safe: 0] == "value")
        #expect(array[safe: 1] == nil)
    }
}
