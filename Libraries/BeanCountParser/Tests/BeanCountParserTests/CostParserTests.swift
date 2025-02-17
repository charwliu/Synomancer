//
//  CostParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
@testable import BeanCountParser
import BeanCountModel
import Testing
import Foundation

class CostParserTests {
    private static let regex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: "\(CostParser.costGroup)", options: [])
    }()
    
    private func cost(from line: String) throws -> Cost? {
        let postingMatches = line.matchingStrings(regex: Self.regex)
        guard let match = postingMatches[safe: 0] else {
            return nil
        }
        return try CostParser.parseFrom(match: match, startIndex: 1)
    }
    
    @Test("cost") func cost() async throws {
        let cost = try cost(from: "{2017-06-09, 1.003 EUR, \"TEST\"}")
        #expect( try Cost(amount: Amount(number: Decimal(1.003),
                                         commoditySymbol: "EUR",
                                         decimalDigits: 3),
                          date: TestUtils.date20170609, label: "TEST") == cost)
    }
    
    @Test func invalid() async {
        let postingMatches = "{2017-06-09, -1.003 EUR, \"TEST\"}".matchingStrings(regex: Self.regex)
        guard let match = postingMatches[safe: 0] else {
            Issue.record("Invalid string")
            return
        }
        #expect(throws: (any Error).self)  {try CostParser.parseFrom(match: match, startIndex: 1)}
    }
    
    @Test func negativeAmount() async throws {
        #expect(try cost(from: "2017-06-09, 1.003 EUR, \"TEST\"}") == nil)
        #expect(try cost(from: "{2017-06-09, 1.003 EUR, \"TEST\"") == nil)
        #expect(try cost(from: "2017-06-09, 1.003 EUR, \"TEST\"") == nil)
    }
    
    @Test func empty() async throws {
        let parsedCost = try cost(from: "{}")
        #expect(try Cost(amount: nil, date: nil, label: nil) == parsedCost)
    }
    
    @Test func emptyStringLabel() async throws {
        let parsedCost = try cost(from: "{\"\"}")
        #expect(try Cost(amount: nil, date: nil, label: "") == parsedCost)
        #expect(try Cost(amount: nil, date: nil, label: nil) != parsedCost)
        
    }
    
    @Test func withoutDate() async throws {
        let parsedCost = try cost(from: "{1.003 EUR, \"TEST\"}")
        #expect(try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: nil, label: "TEST") == parsedCost)
    }
    
    @Test func withoutLabel() async throws {
        let parsedCost = try cost(from: "{2017-06-09, 1.003 EUR}")
        #expect(try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: TestUtils.date20170609, label: nil) == parsedCost)
    }
    
    @Test func onlyDate() async throws {
        let parsedCost = try cost(from: "{2017-06-09}")
        #expect(try Cost(amount: nil, date: TestUtils.date20170609, label: nil) == parsedCost)
    }
    
    @Test func onlyLabel() async throws {
        let parsedCost = try cost(from: "{\"TEST\"}")
        #expect(try Cost(amount: nil, date: nil, label: "TEST") == parsedCost)
    }
    
    @Test func onlyAmount() async throws {
        let parsedCost = try cost(from: "{1.003 EUR}")
        #expect(try Cost(amount: Amount(number: Decimal(1.003), commoditySymbol: "EUR", decimalDigits: 3), date: nil, label: nil) == parsedCost)
    }
    
    @Test func order() async throws {
        let result = try Cost(amount: Amount(number: Decimal(1.003),
                                             commoditySymbol: "EUR",
                                             decimalDigits: 3),
                              date: TestUtils.date20170609,
                              label: "TEST")
        #expect(try cost(from: "{2017-06-09, 1.003 EUR, \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09, \"TEST\", 1.003 EUR}") == result)
        #expect(try cost(from: "{1.003 EUR, 2017-06-09, \"TEST\"}") == result)
        #expect(try cost(from: "{1.003 EUR, \"TEST\", 2017-06-09}") == result)
        #expect(try cost(from: "{\"TEST\", 2017-06-09, 1.003 EUR}") == result)
        #expect(try cost(from: "{\"TEST\", 1.003 EUR, 2017-06-09}") == result)
    }
    
    @Test func whitespace() async throws {
        let result = try Cost(amount: Amount(number: Decimal(1.003),
                                             commoditySymbol: "EUR",
                                             decimalDigits: 3),
                              date: TestUtils.date20170609,
                              label: "TEST")
        // Note: Because a commodity may contain commas there must be a space a either before or after the comma which follows the commodity
        #expect(try cost(from: "{2017-06-09, 1.003 EUR, \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09,1.003 EUR, \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09,    1.003 EUR,     \"TEST\"}") == result)
        #expect(try cost(from: "{   2017-06-09, 1.003 EUR, \"TEST\"    }") == result)
        #expect(try cost(from: "{ 2017-06-09,1.003 EUR, \"TEST\" }") == result)
        #expect(try cost(from: "{2017-06-09 , 1.003 EUR , \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09 ,1.003 EUR , \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09    ,    1.003 EUR ,     \"TEST\"}") == result)
        #expect(try cost(from: "{   2017-06-09    ,    1.003 EUR   ,  \"TEST\"    }") == result)
        #expect(try cost(from: "{2017-06-09, 1.003 EUR ,\"TEST\" }") == result)
        #expect(try cost(from: "{2017-06-09,1.003 EUR ,\"TEST\"}") == result)
    }
    
    @Test func commaCommodity() async throws {
        let result = try Cost(amount: Amount(number: Decimal(1.003),
                                             commoditySymbol: "EUR,AB",
                                             decimalDigits: 3),
                              date: TestUtils.date20170609,
                              label: "TEST")
        #expect(try cost(from: "{2017-06-09, 1.003 EUR,AB, \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09, 1.003 EUR,AB , \"TEST\"}") == result)
        #expect(try cost(from: "{2017-06-09, 1.003 EUR,AB ,\"TEST\"}") == result)
    }
    
    @Test func specialCharacters() async throws {
        let result = try Cost(amount: Amount(number: Decimal(1.003),
                                             commoditySymbol: "💰",
                                             decimalDigits: 3),
                              date: TestUtils.date20170609,
                              label: "TES😀")
        #expect(try cost(from: "{2017-06-09, 1.003 💰, \"TES😀\"}") == result)
    }
}
