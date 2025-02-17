//
//  TransactionMetaDataParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
import BeanCountModel
@testable import BeanCountParser
import Testing

struct TransactionMetaDataParserTests {
    
    private let basicTransactionMetaDataString = "2017-06-09 * \"Payee\" \"Narration\""
    private let whitespaceTransactionMetaDataString = "2017-06-09   *    \"Payee\"   \"Narration\""
    private let endOfLineCommentTransactionMetaDataString = "2017-06-09 * \"Payee\" \"Narration\" ;TESTgfdsgds      "
    private let specialCharacterTransactionMetaDataString = "2017-06-09 * \"öøuß´@🇩🇪🇨🇦💵\" \"🎉😊💵Test⚅℃⁒♾\" #🇨🇦"
    private let incompleteTransactionMetaDataString = "2017-06-09 ! \"Payee\" \"Narration\""
    private let tagsTransactionMetaDataString = "2017-06-09 * \"Payee\" \"Narration\" #1 #two"
    private let invalidDateTransactionMetaDataString = "2017-02-30 * \"Payee\" \"Narration\""

    @Test func basic() async throws{
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: basicTransactionMetaDataString)!
        expectBasicTransactionMetaData(transactionMetaData)
    }
    
    @Test func testWhitespace() async throws {
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: whitespaceTransactionMetaDataString)!
        expectBasicTransactionMetaData(transactionMetaData)
    }
    
    @Test func testEndOfLineCommentTransactionMetaDataString() async throws {
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: endOfLineCommentTransactionMetaDataString)!
        expectBasicTransactionMetaData(transactionMetaData)
    }
    
    @Test func testSpecialCharacterTransactionMetaDataString() async throws {
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: specialCharacterTransactionMetaDataString)!
        #expect(transactionMetaData.narration == "🎉😊💵Test⚅℃⁒♾")
        #expect(transactionMetaData.payee == "öøuß´@🇩🇪🇨🇦💵")
        #expect(transactionMetaData.flag == Flag.complete)
        #expect(transactionMetaData.date == TestUtils.date20170609)
        #expect(transactionMetaData.tags.count == 1)
        #expect(transactionMetaData.tags[0].name == "🇨🇦")
        #expect(String(describing: transactionMetaData) == specialCharacterTransactionMetaDataString)
    }
    
    
    @Test func testIncompleteTransactionMetaDataString() async throws {
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: incompleteTransactionMetaDataString)!
        #expect(transactionMetaData.narration == "Narration")
        #expect(transactionMetaData.payee == "Payee")
        #expect(transactionMetaData.flag == Flag.incomplete)
        #expect(transactionMetaData.date == TestUtils.date20170609)
        #expect(transactionMetaData.tags.count == 0)
        #expect(String(describing: transactionMetaData) == incompleteTransactionMetaDataString)
    }
    
    @Test func testTags() async throws {
        let transactionMetaData = TransactionMetaDataParser.parseFrom(line: tagsTransactionMetaDataString)!
        #expect(transactionMetaData.narration == "Narration")
        #expect(transactionMetaData.payee == "Payee")
        #expect(transactionMetaData.flag == Flag.complete)
        #expect(transactionMetaData.date == TestUtils.date20170609)
        #expect(transactionMetaData.tags.count == 2)
        #expect(transactionMetaData.tags[0].name == "1")
        #expect(transactionMetaData.tags[1].name == "two")
        #expect(String(describing: transactionMetaData) == tagsTransactionMetaDataString)
    }
    
    @Test func testInvalidDate() async throws {
        #expect(TransactionMetaDataParser.parseFrom(line: invalidDateTransactionMetaDataString) == nil)
    }
    
    @Test func testPerformance() async throws {
        
    }
    // Helper
    
    private func expectBasicTransactionMetaData(_ transactionMetaData: TransactionMetaData) {
        #expect(transactionMetaData.narration == "Narration")
        #expect(transactionMetaData.payee == "Payee")
        #expect(transactionMetaData.flag == Flag.complete)
        #expect(transactionMetaData.tags.count == 0)
        #expect(transactionMetaData.date == TestUtils.date20170609)
        #expect(String(describing: transactionMetaData) == basicTransactionMetaDataString)
    }
}
