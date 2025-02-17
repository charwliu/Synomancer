//
//  EventParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

struct EventParserTests {
    static private let basicString = "2017-06-09 event \"ABC\" \"DEF\""
    static private let whitespaceString = "2017-06-09 event    \"  A B C  \"       \"  D E F  \"     "
    static private let endOfLineCommentString = "2017-06-09 event \"ABC\" \"DEF\";gfsdt     "
    static private let specialCharacterString = "2017-06-09 event \"ABC💵\" \"DEF💵\""
    static private let invalidDateString = "2017-02-30 event \"ABC\" \"DEF\""
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString, Self.invalidDateString],
            [
                Event(date: TestUtils.date20170609, name: "ABC", value: "DEF"),
                Event(date: TestUtils.date20170609, name: "  A B C  ", value: "  D E F  "),
                Event(date: TestUtils.date20170609, name: "ABC", value: "DEF"),
                Event(date: TestUtils.date20170609, name: "ABC💵", value: "DEF💵"),
                nil
            ])
    )
    func parseFrom(line: String, result: Event?) async throws {
        let event = EventParser.parseFrom(line: line)
        if let result = result {
            let e = try #require(event)
            #expect(e.date == result.date)
            #expect(e.name == result.name)
            #expect(e.value == result.value)
        } else {
            #expect(event == nil)
        }
    }
}
