//
//  TransactionPostingTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/27/24.
//

@testable import BeanCountModel
import Testing
import Foundation

struct TransactionPostingTests {
    private var posting1 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
    
    @Test func testInitTransactionPosting() async throws {
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []),
                                      postings: [])
        let posting = Posting(accountName: TestUtils.cash, amount: TestUtils.amount, metaData: ["A": "B"])
        let transactionPosting = TransactionPosting(posting: posting, transaction: transaction)
        #expect(posting.accountName == transactionPosting.accountName)
        #expect(posting.amount == transactionPosting.amount)
        #expect(posting.metaData == transactionPosting.metaData)
        #expect(transaction == transactionPosting.transaction)
    }
    
    @Test func testDescription() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: "💵")
        let posting = Posting(accountName: TestUtils.chequing, amount: amount)
        #expect(String(describing: posting) == "  \(String(describing: TestUtils.chequing)) \(String(describing: amount))")
    }
    
    @Test func testDescriptionMetaData() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: "💵")
        let posting = Posting(accountName: TestUtils.chequing, amount: amount, metaData: ["A": "B"])
        
        #expect(String(describing: posting) == "  \(String(describing: TestUtils.chequing)) \(String(describing: amount))\n    A: \"B\"")
    }
    
    @Test func testDescriptionPrice() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: "💵")
        let price = Amount(number: Decimal(1.555), commoditySymbol: TestUtils.eur)
        let posting = Posting(accountName: TestUtils.chequing, amount: amount, price: price)
        
        #expect(String(describing: posting) == "  \(String(describing: TestUtils.chequing)) \(String(describing: amount)) @ \(price)")
    }
    
    @Test func testDescriptionCost() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: "💵")
        let cost = try Cost(amount: amount, date: nil, label: "label")
        let posting = Posting(accountName: TestUtils.chequing, amount: amount, price: nil, cost: cost)
        
        #expect(String(describing: posting) == "  \(String(describing: TestUtils.chequing)) \(String(describing: amount)) \(String(describing: cost))")
    }
    
    @Test func testDescriptionCostAndPrice() async throws {
        let amount = Amount(number: Decimal(1), commoditySymbol: "💵")
        let price = Amount(number: Decimal(1.555), commoditySymbol: TestUtils.eur)
        let cost = try Cost(amount: amount, date: nil, label: "label")
        let posting = Posting(accountName: TestUtils.chequing, amount: amount, price: price, cost: cost)
        
        #expect(String(describing: posting) == "  \(String(describing: TestUtils.chequing)) \(String(describing: amount)) \(String(describing: cost)) @ \(price)")
    }
    
    @Test func testEqual() async throws {
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        #expect(posting1 == posting2)
    }
    
    @Test func testEqualRespectsMetaData()  async throws {
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount, metaData: ["A": "B"])
        #expect(posting1 != posting2)
    }
    
    @Test func testEqualRespectsAccount() async throws {
        let posting2 = Posting(accountName: try AccountName("\(String(describing: TestUtils.cash)):💰"), amount: TestUtils.amount)
        #expect(posting1 != posting2)
    }
    
    @Test func testEqualRespectsAmount() async throws{
        let posting2 = Posting(accountName: TestUtils.cash,
                               amount: Amount(number: posting1.amount.number + posting1.amount.number,
                                              commoditySymbol: "\(TestUtils.eur)1"))
        #expect(posting1 != posting2)
    }
    
    @Test func testEqualRespectsPrice() async throws {
        let price = Amount(number: Decimal(1.555), commoditySymbol: TestUtils.eur)
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount, price: price)
        #expect(posting1 != posting2)
    }
    
    @Test func testEqualRespectsCost() async throws {
        let amount = Amount(number: Decimal(1.555), commoditySymbol: TestUtils.eur)
        let cost = try Cost(amount: amount, date: nil, label: "label")
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount, price: nil, cost: cost)
        #expect(posting1 != posting2)
    }
}
