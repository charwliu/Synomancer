//
//  CommodityParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
import Testing
@testable import BeanCountParser
import BeanCountModel

struct CommodityParserTests {
    
    static private let basicString = "2017-06-09 commodity CAD"
    static private let whitespaceString = "2017-06-09    commodity        CAD"
    static private let endOfLineCommentString = "2017-06-09 commodity CAD ;gfsdt     "
    static private let specialCharacterString = "2017-06-09 commodity CAD💵"
    static private let invalidDateString = "2017-02-30 commodity CAD"
    static private let commodity = Commodity(symbol: "CAD", opening: TestUtils.date20170609)
    static private let special = Commodity(symbol: "CAD💵", opening: TestUtils.date20170609)
    
    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString, Self.invalidDateString],
            [Self.commodity, Self.commodity, Self.commodity, Self.special, nil]
        )
    )
    func parseFrom(line: String, result: Commodity?) {
        let commodity = CommodityParser.parseFrom(line: line)
        #expect(commodity == result)
    }
}
