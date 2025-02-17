//
//  BalanceTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing
import Foundation

struct BalanceTests {
    @Test func testDescription() async throws {
        let account = Account(name: TestUtils.cash)
        let amount = Amount(number: Decimal(1), commoditySymbol: TestUtils.cad)
        
        var balance = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount)
        #expect(String(describing: balance) == "2017-06-08 balance \(account.name) \(amount)")
        
        balance = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount, metaData: ["A": "B"])
        #expect(String(describing: balance) == "2017-06-08 balance \(account.name) \(amount)\n  A: \"B\"")
    }
    
    @Test func testEqual() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: TestUtils.cad)
        var balance = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount)
        var balance2 = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount)
        #expect(balance == balance2)
        
        // Meta Data
        balance = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount, metaData: ["A": "B"])
        balance2 = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount, metaData: ["A": "C"])
        #expect(balance != balance2)
        balance2 = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount, metaData: ["A": "B"])
        #expect(balance == balance2)

        // Date different
        let balance3 = Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: amount)
        #expect(balance != balance3)
        
        // Account different
        let balance4 = Balance(date: TestUtils.date20170608, accountName: TestUtils.chequing, amount: amount)
        #expect(balance != balance4)

        // Amount commodity different
        let amount2 = Amount(number: Decimal(1), commoditySymbol: TestUtils.usd)
        let balance5 = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount2)
        #expect(balance != balance5)
        
        // Amount number different
        let amount3 = Amount(number: Decimal(2), commoditySymbol: TestUtils.cad)
        let balance6 = Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: amount3)
        #expect(balance != balance6)
    }
}
