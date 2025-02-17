//
//  DateParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

struct DateParserTests {
    
    @Test(
        arguments: zip(
            ["2017-06-09", "2017-00-09", "2017-02-30"], [TestUtils.date20170609, nil, nil]
        )
    )
    func parseFrom(string: String, result: Date?) async throws {
        let date = DateParser.parseFrom(string: string)
        #expect(date == result)
    }
}
