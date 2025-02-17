//
//  MetaDataParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

struct MetaDataParserTests {
    static private let basicString = "  test:\"ABC\""
    static private let whitespaceString = "  test:    \"A B C\"        "
    static private let whitespaceBeginningString = "    test:    \"  A B C  \"        "
    static private let whitespaceNonMatchingString = " test:\"ABC\""
    static private let endOfLineCommentString = "  test: \"ABC\";gfsdt     "
    static private let specialCharacterString = "  test💵: \"ABC💵\""
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.whitespaceBeginningString, Self.whitespaceNonMatchingString, Self.endOfLineCommentString, Self.specialCharacterString],
            [
                ["test": "ABC"],
                ["test": "A B C"],
                ["test": "  A B C  "],
                nil,
                ["test": "ABC"],
                ["test💵": "ABC💵"]
            ])
    )
    func parseFrom(line: String, result: [String: String]?) async throws {
        let metaData = MetaDataParser.parseFrom(line: line)
        if let result = result {
            let md = try #require(metaData)
            #expect(md == result)
        } else {
            #expect(metaData == nil)
        }
    }
}
