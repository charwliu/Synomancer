//
//  Test.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//
@testable import BeanCountModel

import Testing

struct AccountNameTests {
    private let invalidNames = ["Assets", "Liabilities", "Income", "Expenses", "Equity", "Assets:", "Assets:Test:", "Assets:Test:", "Assets:Test::Test", "💰", ""]
    private let validNames = ["Assets:Cash", "Assets:Cash:Test:Test:A", "Assets:Cash:Ca💰h:Test:💰", "Liabilities:Test", "Income:Test", "Expenses:Test", "Equity:Test"]

    
    @Test func testInitNames() async throws {
        for name in validNames {
            #expect(throws: Never.self ) { try AccountName(name)}
        }
        for name in invalidNames {
            #expect { try AccountName(name) } throws: { error in
                error.localizedDescription == "Invalid Account name: \(name)"
            }
        }
    }
    
    @Test func testIsAccountNameValid() async throws {
        for name in validNames {
            #expect(AccountName.isNameValid(name))
        }
        for name in invalidNames {
            #expect(AccountName.isNameValid(name) == false)
        }
    }
    
    @Test func testNameItem() async throws {
        #expect(TestUtils.cash.nameItem == "Cash")
        #expect(try AccountName("Assets:A:B:C:D:E:Cash").nameItem == "Cash")
        #expect(try AccountName("Assets:💰").nameItem == "💰")
    }
    
    @Test func testAccountType() async throws {
        #expect(try AccountName("Assets:Test").accountType == AccountType.asset)
        #expect(try AccountName("Liabilities:Test").accountType == AccountType.liability)
        #expect(try AccountName("Income:Test").accountType == AccountType.income)
        #expect(try AccountName("Expenses:Test").accountType == AccountType.expense)
        #expect(try AccountName("Equity:Test").accountType == AccountType.equity)
    }
    
    @Test func testAccountNameEqual() async throws {
        let name1 = try AccountName("Assets:Test")
        let name2 = try AccountName("Assets:Test")
        let name3 = try AccountName("Assets:Test:Test")
        #expect(name1 == name2)
        #expect(name1 != name3)

    }

}
