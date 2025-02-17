//
//  PriceParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation


struct PriceParserTests {
    static let twoP11 = Decimal(211) / Decimal(100)
    static private let price = try! Price(date: TestUtils.date20170609,
                                   commoditySymbol: "EUR",
                                          amount: Amount(number: twoP11, commoditySymbol: "CAD", decimalDigits: 2))
    
    static private let basicPrice = "2017-06-09 price EUR 2.11 CAD"
    static private let priceComment = "2017-06-09 price EUR 2.11 CAD ;fsajfdsanfjsak"
    static private let priceWhitespace = "2017-06-09       price        EUR        2.11           CAD"
    
    static private let priceSpecialCharacter = "2017-06-09 price 💵 2.11 💸"
    static private let priceWholeNumber = "2017-06-09 price EUR 2 CAD"
    
    static private let invalidPriceMissingNumber = "2017-06-09 price EUR  CAD"
    static private let invalidPriceMissingFirstCurrency = "2017-06-09 price  2.11 CAD"
    static private let invalidPriceMissingSecondCurrency = "2017-06-09 price EUR 2.11"
    static private let invalidPriceMissingCurrencies = "2017-06-09 price 2.11"
    
    @Test(arguments: zip([
        Self.basicPrice,
        Self.priceComment,
        Self.priceWhitespace,
        Self.priceSpecialCharacter,
        Self.priceWholeNumber,
        Self.invalidPriceMissingNumber,
        Self.invalidPriceMissingCurrencies,
        Self.invalidPriceMissingFirstCurrency,
        Self.invalidPriceMissingSecondCurrency
    ], [
        Self.price,
        Self.price,
        Self.price,
        try Price(date: TestUtils.date20170609, commoditySymbol: "💵", amount:  Amount(number: Self.twoP11, commoditySymbol: "💸", decimalDigits: 2)),
        try Price(date: TestUtils.date20170609, commoditySymbol: "EUR", amount:  Amount(number: Decimal(2), commoditySymbol: "CAD", decimalDigits: 0)),
        nil,
        nil,
        nil,
        nil,
    ])
    )
    func parseFrom(line: String, result: Price?) async throws {
        let parsedPrice = PriceParser.parseFrom(line: line)
        if let result = result {
            let price = try #require(parsedPrice)
            #expect(price == result)
        } else {
            #expect(parsedPrice == nil)
        }
    }
    
}
