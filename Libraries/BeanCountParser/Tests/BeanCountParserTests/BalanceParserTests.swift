//
//  BalanceParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
@testable import BeanCountParser
import BeanCountModel
import Testing


struct BalanceParserTests {
    static private let basicString = "2017-06-09 balance Assets:Cash 10.00 CAD"
    static private let whitespaceString = "2017-06-09    balance    Assets:Cash     10.00      CAD"
    static private let endOfLineCommentString = "2017-06-09 balance Assets:Cash 10.00 CAD ;gfsdt     "
    static private let specialCharacterString = "2017-06-09 balance Assets:💰 10.00 💰"
    static private let invalidDateString = "2017-02-30 balance Assets:Cash 10.00 CAD"
    
    static private let amount = Amount( number: 10, commoditySymbol: "CAD", decimalDigits: 2 )
    static private let balance = Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: amount)
    static private let special = Balance( date:TestUtils.date20170609, accountName: TestUtils.special, amount: Amount(number: 10, commoditySymbol: "💰", decimalDigits: 2))

    @Test(
        arguments: zip(
            [Self.basicString, Self.whitespaceString, Self.endOfLineCommentString, Self.specialCharacterString, Self.invalidDateString],
            [ Self.balance, Self.balance, Self.balance,Self.special, nil])
    )
    func parseFrom(line: String, result: Balance?) {
        let balance = BalanceParser.parseFrom(line: line)
        #expect(balance == result)
    }
    
}
