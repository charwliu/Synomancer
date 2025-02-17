//
//  OptionParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

struct OptionParserTests {
    static private let basicString = "option \"ABC\" \"DEF\""
    static private let whitespaceString = "option    \"  A B C  \"       \"  D E F  \"     "
    static private let endOfLineCommentString = "option \"ABC\" \"DEF\";gfsdt     "
    static private let specialCharacterString = "option \"ABC💵\" \"DEF💵\""
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString],
            [
                Option(name: "ABC", value: "DEF"),
                Option(name: "  A B C  ", value: "  D E F  "),
                Option(name: "ABC", value: "DEF"),
                Option(name: "ABC💵", value: "DEF💵"),
            ])
    )
    func parseFrom(line: String, result: Option?) async throws {
        let option = OptionParser.parseFrom(line: line)
        if let result = result {
            let o = try #require(option)
            #expect(o.name == result.name)
            #expect(o.value == result.value)
        } else {
            #expect(option == nil)
        }
    }
    
}
