//
//  ParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//

import Testing
@testable import BeanCountParser
import BeanCountModel
import Foundation

struct ParserTests {
    enum TestFile: String {
        case minimal = "Minimal"
        case postingWithoutTransaction = "PostingWithoutTransaction"
        case transactionWithoutPosting = "TransactionWithoutPosting"
        case comments = "Comments"
        case commentsEndOfLine = "CommentsEndOfLine"
        case whitespace = "Whitespace"
        case big = "Big"
        case invalidCost = "InvalidCost"
        case metaData = "MetaData"
        
        static let withoutError = [minimal, comments, commentsEndOfLine, whitespace, big, metaData]
    }
    
    static private let basicAccountOpeningString = "2017-06-09 open Assets:Cash"
    static private let accountOpeningStringCommodity = "2017-06-09 open Assets:Cash EUR"
    static private let basicAccountClosingString = "2017-06-09 close Assets:Cash"
    
    static private let commodityString = "2017-06-09 commodity EUR"
    static private let priceString = "2017-06-09 price EUR 1.50 CAD"
    static private let balanceString = "2017-06-09 balance Assets:Cash 0.00 CAD"
    static private let invalidBalanceString = "2017-06-09 balance TEST:Cash 0.00 CAD"
    static private let optionString = "option \"ABC\" \"DEF\""
    static private let pluginString = "plugin \"ABC\""
    static private let eventString = "2017-06-09 event \"ABC\" \"DEF\""
    static private let customString = "2017-06-09 custom \"ABC\" \"DEF\""
    
    static private let metaDataString = "  metaData: \"TestString\""
    static private let metaDataString2 = "  metaData2: \"TestString2\""
    static private let metaData = ["metaData": "TestString"]
    static private let metaData2 = ["metaData": "TestString", "metaData2": "TestString2"]
    static private let comment = "; TEST comment"
    
    
    
    @Test("parse(contentOf:) minimal, whitespace and end of line comments", arguments: [TestFile.minimal, TestFile.whitespace, TestFile.commentsEndOfLine])
    func minimal(testFile: TestFile) async throws {
        ensureMinimal(testFile: testFile)
    }
    
    @Test("parse(contentOf:) posting without transaction") func postingWithoutTransaction() async throws {
        let ledger = ensureEmpty(testFile: .postingWithoutTransaction)
        #expect(ledger.errors.count == 1)
    }
    
    @Test("parse(contentOf:) transaction without posting") func transactionWithoutPosting() async throws {
        let ledger = ledgerFor(testFile: .transactionWithoutPosting)
        #expect(ledger.errors.count == 1)
        #expect(ledger.parsingErrors.count == 0)
    }
    
    @Test("parse(contentOf:) comments") func comments() async throws {
        let ledger = ensureEmpty(testFile: .comments)
        #expect(ledger.errors.count == 0)
    }
    
    @Test("parse(contentOf:) invalid cost") func invalidCost() async throws {
        var errorMessage = ""
        do {
            _ = try Cost(amount: Amount(number: Decimal(-1), commoditySymbol: "EUR", decimalDigits: 2), date: nil, label: nil)
        } catch {
            errorMessage = error.localizedDescription
        }
        let ledger = ledgerFor(testFile: .invalidCost)
        #expect(ledger.errors.count == 2)
        #expect(ledger.errors[0] == "\(errorMessage) (line 7)")
    }
    
    @Test("parse(string:) accounts") func accounts() async throws {
        // open and close is ok
        var ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)\n\(Self.basicAccountClosingString)")
        #expect(ledger.errors.isEmpty)
        
        // only open is ok
        ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)")
        #expect(ledger.errors.isEmpty)
        
        // only close is not ok
        ledger = Parser.parse(string: "\(Self.basicAccountClosingString)")
        #expect(!ledger.errors.isEmpty)
        
        // open twice is not ok
        ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)\n\(Self.basicAccountOpeningString)")
        #expect(!ledger.errors.isEmpty)
        ledger = Parser.parse(string: "\(Self.accountOpeningStringCommodity)\n\(Self.accountOpeningStringCommodity)")
        #expect(!ledger.errors.isEmpty)
        
        // close twice is not ok
        ledger = Parser.parse(string: "\(Self.basicAccountClosingString)\n\(Self.basicAccountClosingString)")
        #expect(!ledger.errors.isEmpty)
        
        // close twice (+ normal open) is not ok
        ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)\n\(Self.basicAccountClosingString)\n\(Self.basicAccountClosingString)")
        #expect(!ledger.errors.isEmpty)
        
        // meta data
        ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)\n\(Self.metaDataString)")
        #expect(ledger.accounts.first?.metaData == Self.metaData)
        
        // no meta data on closing
        ledger = Parser.parse(string: "\(Self.basicAccountOpeningString)\n\(Self.basicAccountClosingString)\n\(Self.metaDataString)")
        #expect(!ledger.errors.isEmpty)
    }
    
    @Test("parse(string:) commodity") func commodity() async throws {
        var ledger = Parser.parse(string: "\(Self.commodityString)")
        #expect(ledger.errors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.commodityString)\n\(Self.commodityString)")
        #expect(!ledger.errors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.commodityString)\n\(Self.metaDataString)")
        #expect(ledger.commodities.first?.metaData == Self.metaData)
    }
    
    @Test("parse(string:) price") func price() async throws {
        var ledger = Parser.parse(string: "\(Self.priceString)")
        #expect(ledger.parsingErrors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.priceString)\n\(Self.priceString)")
        #expect(!ledger.parsingErrors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.priceString)\n\(Self.metaDataString)")
        #expect(ledger.prices.first?.metaData == Self.metaData)
    }
    
    @Test("parse(string:) balance") func balance() async throws {
        var ledger = Parser.parse(string: "\(Self.balanceString)")
        #expect(ledger.parsingErrors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.invalidBalanceString)")
        #expect(!ledger.parsingErrors.isEmpty)

        ledger = Parser.parse(string: "\(Self.balanceString)\n\(Self.metaDataString)")
        #expect(ledger.accounts.first?.balances.first?.metaData == Self.metaData)
    }
    
    @Test("parse(string:) option") func option() async throws {
        let ledger = Parser.parse(string: "\(Self.optionString)")
        #expect(ledger.option.first == Option(name: "ABC", value: "DEF"))
    }
    
    @Test("parse(string:) plugin") func plugin() async throws {
        let ledger = Parser.parse(string: "\(Self.pluginString)")
        #expect(ledger.plugins.first == "ABC")
    }
    
    @Test("parse(string:) event") func event() async throws {
        var ledger = Parser.parse(string: "\(Self.eventString)")
        #expect(ledger.events.first == Event(date: TestUtils.date20170609, name: "ABC", value: "DEF"))
        
        ledger = Parser.parse(string: "\(Self.eventString)\n\(Self.metaDataString)")
        #expect(ledger.events.first?.metaData == Self.metaData)
    }
    
    @Test("parse(string:) custom") func custom() async throws {
        var ledger = Parser.parse(string: "\(Self.customString)")
        #expect(ledger.custom.first == Custom(date: TestUtils.date20170609, name: "ABC", values: ["DEF"]))
        
        ledger = Parser.parse(string: "\(Self.customString)\n\(Self.metaDataString)")
        #expect(ledger.custom.first?.metaData == Self.metaData)
    }
    
    @Test("parse(string:) comments meta data") func commentsMetaData() async throws {
        var ledger = Parser.parse(string: "\(Self.comment)")
        #expect(ledger.errors.isEmpty)
        
        ledger = Parser.parse(string: "\(Self.comment)\n\(Self.eventString)")
        #expect(ledger.events.first == Event(date: TestUtils.date20170609, name: "ABC", value: "DEF"))
        
        ledger = Parser.parse(string: "\(Self.comment)\n\(Self.eventString)\n\(Self.metaDataString)")
        #expect(ledger.events.first?.metaData == Self.metaData)
        
        ledger = Parser.parse(string: "\(Self.comment)\n\(Self.eventString)\n\(Self.metaDataString)\n\(Self.comment)\n\(Self.metaDataString2)")
        #expect(ledger.events.first?.metaData == Self.metaData2)
    }
    
    @Test("parse(string:) meta data") func metaData() async throws {
        let ledger = ledgerFor(testFile: .metaData)
        #expect(ledger.accounts[0].metaData == Self.metaData2)
        #expect(ledger.accounts[1].metaData == Self.metaData2)
        #expect(ledger.commodities.first?.metaData == Self.metaData2)
        #expect(ledger.transactions[0].metaData.metaData == Self.metaData2)
        #expect(ledger.transactions[0].postings[0].metaData == Self.metaData2)
        #expect(ledger.transactions[0].postings[1].metaData == Self.metaData2)
        #expect(ledger.transactions[1].metaData.metaData == Self.metaData2)
        #expect(ledger.transactions[1].postings[0].metaData == Self.metaData2)
        #expect(ledger.transactions[1].postings[1].metaData == Self.metaData2)
    }

    @Test("parse(contentOf:) and parse(string:) round trip") func roundTrip() async throws {
        for testFile in TestFile.withoutError {
            let ledger1 = ledgerFor(testFile: testFile)
            let ledger2 = Parser.parse(string: String(describing: ledger1))
            #expect(ledger1 == ledger2)
        }
    }
    
    // Helper
    
    private func ensureEmpty(testFile: TestFile) -> Ledger {
        let ledger = ledgerFor(testFile: testFile)
        #expect(ledger.transactions.count == 0)
        return ledger
    }
    
    
    private func ensureMinimal(testFile: TestFile) {
        let ledger = ledgerFor(testFile: testFile)
        #expect(ledger.transactions.count == 1)
        #expect(ledger.errors.isEmpty)
        #expect(ledger.commodities.count == 1)
        #expect(ledger.accounts.count == 2)
        let transaction = ledger.transactions[0]
        #expect(transaction.postings.count == 2)
        #expect(transaction.metaData.payee == "Payee")
        #expect(transaction.metaData.narration == "Narration")
        #expect(transaction.metaData.date == TestUtils.date20170608)
        let posting1 = transaction.postings.first {$0.amount.number == Decimal(-1)}!
        #expect(posting1.accountName.fullName == "Equity:OpeningBalance")
        #expect(posting1.amount.commoditySymbol == "EUR")
        let posting2 = transaction.postings.first {$0.amount.number == Decimal(1)}!
        #expect(posting2.accountName.fullName == "Assets:Checking")
        #expect(posting2.amount.commoditySymbol == "EUR")
    }
    
    private func ledgerFor(testFile: TestFile) -> Ledger{
        let url: URL
        switch testFile {
        case .minimal:
            url = Bundle.module.url(forResource: "Minimal", withExtension: "beancount", subdirectory: "Resources")!
        case .postingWithoutTransaction:
            url = Bundle.module.url(forResource: "PostingWithoutTransaction", withExtension: "beancount", subdirectory: "Resources")!
        case .transactionWithoutPosting:
            url = Bundle.module.url(forResource: "TransactionWithoutPosting", withExtension: "beancount", subdirectory: "Resources")!
        case .comments:
            url = Bundle.module.url(forResource: "Comments", withExtension: "beancount", subdirectory: "Resources")!
        case .commentsEndOfLine:
            url = Bundle.module.url(forResource: "CommentsEndOfLine", withExtension: "beancount", subdirectory: "Resources")!
        case .whitespace:
            url = Bundle.module.url(forResource: "Whitespace", withExtension: "beancount", subdirectory: "Resources")!
        case .big:
            url = Bundle.module.url(forResource: "Big", withExtension: "beancount", subdirectory: "Resources")!
        case .invalidCost:
            url = Bundle.module.url(forResource: "InvalidCost", withExtension: "beancount", subdirectory: "Resources")!
        case .metaData:
            url = Bundle.module.url(forResource: "MetaData", withExtension: "beancount", subdirectory: "Resources")!
        }
        guard let ledger = try? Parser.parse(contentOf: url) else {
            Issue.record("Failed to parse ledger at URL: \(url.absoluteString)")
            return Ledger()
        }
        return ledger
    }
}
