//
//  PostingParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

class PostingParserTests {
    
    private let transaction = Transaction(metaData: TransactionMetaData(date: Date(), payee: "Payee", narration: "Narration"), postings: [])
    
    static private let basicPosting: Posting = Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(1.23), commoditySymbol: "EUR", decimalDigits: 2))
    
    static private let basicPostingString = "  Assets:Checking 1.23 EUR"
    static private let integerPostingString = "  Assets:Checking 1 EUR"
    static private let noThousandsSeparatorPostingString = "  Assets:Checking 100000 EUR"
    static private let thousandsSeparatorPostingString = "  Assets:Checking 100,000 EUR"
    static private let negativePostingString = "  Assets:Checking -1.2 EUR"
    static private let positivePostingString = "  Assets:Checking +1.23 EUR"
    static private let separatorPostingString = "  Assets:Checking -1,000.23 EUR"
    static private let whitespacePostingString = "         Assets:Checking        1.23    EUR     "
    static private let invalidAccountPostingString = "  Invalid:Checking 1.23 EUR"
    static private let endOfLineCommentPostingString = " Assets:Checking 1.23 EUR    ;gfdsg f gfds   "
    static private let specialCharacterPostingString = "  Assets:💰 1.00 💵"
    static private let totalPricePostingString = "  Assets:💰 -2.00 💵 @@ 2.0 EUR"
    static private let unitPricePostingString = "  Assets:💰 2.0 💵 @ 1.003 EUR"
    static private let costPostingString = "  Assets:💰 2.0 💵 {2017-06-09, 1.003 EUR, \"TEST\"}"
    static private let invalidCostPostingString = "  Assets:💰 2.0 💵 {2017-06-09, -1.003 EUR, \"TEST\"}"
    static private let costAndUnitPricePostingString = "  Assets:💰 2.0 💵 {2017-06-09, 1.003 EUR} @ 1.003 EUR"
    static private let costAndTotalPricePostingString = "  Assets:💰 2.0 💵 {1.003 EUR, \"TEST\"} @@ 2.0 EUR"
    
    @Test(arguments: zip([
        Self.basicPostingString,
        Self.integerPostingString,
        Self.noThousandsSeparatorPostingString,
        Self.thousandsSeparatorPostingString,
        Self.negativePostingString,
        Self.positivePostingString,
        Self.separatorPostingString,
        Self.whitespacePostingString,
        Self.invalidAccountPostingString,
        Self.endOfLineCommentPostingString,
        Self.specialCharacterPostingString,
        Self.totalPricePostingString,
        Self.unitPricePostingString,
        Self.costPostingString,
//        Self.invalidCostPostingString,
        Self.costAndUnitPricePostingString,
        Self.costAndTotalPricePostingString,
    ], [
        Self.basicPosting,
        Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(1), commoditySymbol: "EUR", decimalDigits: 0)),
        Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(100_000), commoditySymbol: "EUR", decimalDigits: 0)),
        Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(100_000), commoditySymbol: "EUR", decimalDigits: 0)),
        Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(-1.2), commoditySymbol: "EUR", decimalDigits: 1)),
        Self.basicPosting,
        Posting(accountName: TestUtils.checking, amount: Amount(number: Decimal(-1_000.23), commoditySymbol: "EUR", decimalDigits: 2)),
        Self.basicPosting,
        nil,
        Self.basicPosting,
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(1), commoditySymbol: "💵", decimalDigits: 2)),
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(-2), commoditySymbol: "💵", decimalDigits: 2), price: Amount(number: Decimal(1), commoditySymbol: "EUR", decimalDigits: 1)),
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(2), commoditySymbol: "💵", decimalDigits: 1), price: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3)),
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(2), commoditySymbol: "💵", decimalDigits: 1), cost: try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: TestUtils.date20170609, label: "TEST")),
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(2), commoditySymbol: "💵", decimalDigits: 1), price: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), cost: try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: TestUtils.date20170609, label: nil)),
        Posting(accountName: TestUtils.special, amount: Amount(number: Decimal(2), commoditySymbol: "💵", decimalDigits: 1), price: Amount(number: Decimal(1), commoditySymbol: "EUR", decimalDigits: 1), cost: try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: nil, label: "TEST")),

    ])
    )  func parseFrom(line: String, result: Posting?) async throws {
        let posting = try PostingParser.parseFrom(line: line)
        if let result = result {
            #expect(result == posting)
        } else {
            #expect(posting == nil)
        }
    }
    
    @Test func pareeFromWithError() async throws {
        // "  Assets:💰 2.0 💵 {2017-06-09, -1.003 EUR, \"TEST\"}"
        // negativeAmount
        #expect(throws: (any Error).self) { try PostingParser.parseFrom(line: Self.invalidCostPostingString) }
    }
}
