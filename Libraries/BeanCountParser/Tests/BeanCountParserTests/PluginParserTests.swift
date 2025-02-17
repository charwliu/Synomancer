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

struct PluginParserTests {
    static private let basicString = "plugin \"ABC\""
    static private let whitespaceString = "plugin    \"  A B C  \"        "
    static private let endOfLineCommentString = "plugin \"ABC\";gfsdt     "
    static private let specialCharacterString = "plugin \"ABC💵\""
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString],
            [
                "ABC",
                "  A B C  ",
                "ABC",
                "ABC💵"
            ])
    )
    func parseFrom(line: String, result: String?) async throws {
        let plugin = PluginParser.parseFrom(line: line)
        if let result = result {
            let o = try #require(plugin)
            #expect(o == result)
        } else {
            #expect(plugin == nil)
        }
    }
    
}
