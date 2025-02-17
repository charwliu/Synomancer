//
//  TransactionMetaDataTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/27/24.
//

@testable import BeanCountModel
import Testing
import Foundation

struct TransactionMetaDataTests {
    private let payee = "Payee"
    private let narration = "Narration"
    private let flag = Flag.complete
    private let date = TestUtils.date20170608
    private let dateString = "2017-06-08"
    private var transactionMetaData: TransactionMetaData?
    
    init() async throws {
        transactionMetaData = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [])
    }
    
    @Test func testDescription() async throws {
        #expect(String(describing: transactionMetaData!) == "\(dateString) \(String(describing: flag)) \"\(payee)\" \"\(narration)\"")
    }
    
    @Test func testDescriptionMetaData() async throws {
        let transactionMetaData = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [], metaData: ["A": "B"])
        #expect(String(describing: transactionMetaData) == "\(dateString) \(String(describing: flag)) \"\(payee)\" \"\(narration)\"\n  A: \"B\"")
    }
    
    @Test func testDescriptionSpecialCharacters() async throws {
        let payee = "🏫"
        let narration = "🎓"
        let transactionMetaData = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [])
        #expect(String(describing: transactionMetaData) == "\(dateString) \(String(describing: flag)) \"\(payee)\" \"\(narration)\"")
    }
    
    @Test func testDescriptionTag() async throws{
        let tag = Tag(name: "🎁")
        let transactionMetaData = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [tag])
        #expect(String(describing: transactionMetaData) == "\(dateString) \(String(describing: flag)) \"\(payee)\" \"\(narration)\" \(String(describing: tag))")
    }
    
    @Test func testDescriptionTags() async throws{
        let tag1 = Tag(name: "tag1")
        let tag2 = Tag(name: "tag2")
        let transactionMetaData = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [tag1, tag2])
        #expect(String(describing: transactionMetaData) ==
                       "\(dateString) \(String(describing: flag)) \"\(payee)\" \"\(narration)\" \(String(describing: tag1)) \(String(describing: tag2))")
    }
    
    @Test func testEqual() async throws {
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [])
        #expect(transactionMetaData == transactionMetaData1)
    }
    
    @Test func testEqualWithTags()  async throws {
        let tag1 = Tag(name: "tag1")
        let tag2 = Tag(name: "tag2")
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [tag1, tag2])
        let transactionMetaData2 = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [tag1, tag2])
        #expect(transactionMetaData1 == transactionMetaData2)
    }
    
    @Test func testEqualRespectsMetaData() async throws{
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee, narration: narration, flag: flag, tags: [], metaData: ["A": "B"])
        #expect(transactionMetaData != transactionMetaData1)
    }
    
    @Test func testEqualRespectsDate() async throws {
        let transactionMetaData1 = TransactionMetaData(date: date.addingTimeInterval(TimeInterval(1)), payee: payee, narration: narration, flag: flag, tags: [])
        #expect(transactionMetaData != transactionMetaData1)
    }
    
    @Test func testEqualRespectsPayee() async throws {
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee + "1", narration: narration, flag: flag, tags: [])
        #expect(transactionMetaData != transactionMetaData1)
    }
    
    @Test func testEqualRespectsNarration() async throws {
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee, narration: narration + "1", flag: flag, tags: [])
        #expect(transactionMetaData != transactionMetaData1)
    }
    
    @Test func testEqualRespectsTags() async throws {
        let tag1 = Tag(name: "tag1")
        let transactionMetaData1 = TransactionMetaData(date: date, payee: payee, narration: narration, flag: Flag.incomplete, tags: [tag1])
        #expect(transactionMetaData != transactionMetaData1)
    }
    
}
