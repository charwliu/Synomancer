//
//  CustomsParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
import Testing
@testable import BeanCountParser
import BeanCountModel

struct CustomsParserTests {
    static private let basicString = "2017-06-09 custom \"ABC\" \"DEF\""
    static private let multipleValuesString = "2017-06-09 custom \"ABC\" \"DEF\" \"GHI\" \"JKL\" \"MNO\""
    static private let whitespaceString = "2017-06-09 custom    \"  A B C  \"       \"  D E F  \"      \"G H I\"       "
    static private let endOfLineCommentString = "2017-06-09 custom \"ABC\" \"DEF\"  \"GHI\";gfsdt     "
    static private let specialCharacterString = "2017-06-09 custom \"ABC💵\" \"DEF💵\" \"GHI💵\""
    static private let invalidDateString = "2017-02-30 custom \"ABC\" \"DEF\""
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.multipleValuesString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString, Self.invalidDateString],
            [
                Custom(date: TestUtils.date20170609, name: "ABC", values: ["DEF"]),
                Custom(date: TestUtils.date20170609, name: "ABC", values: ["DEF", "GHI", "JKL", "MNO"]),
                Custom(date: TestUtils.date20170609, name: "  A B C  ", values: ["  D E F  ", "G H I"]),
                Custom(date: TestUtils.date20170609, name: "ABC", values: ["DEF", "GHI"]),
                Custom(date: TestUtils.date20170609, name: "ABC💵", values: ["DEF💵", "GHI💵"]),
                nil
            ])
    )
    func parseFrom(line: String, result: Custom?) async throws {
        let custom = CustomsParser.parseFrom(line: line)
        if let result = result {
            let c = try #require(custom)
            #expect(c.date == result.date)
            #expect(c.name == result.name)
            #expect(c.values == result.values)
        } else {
            #expect(custom == nil)
        }
    }
}
