//
//  AccountParserTests.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//
@testable import BeanCountParser
import Testing
import Foundation
import BeanCountModel


struct AccountParserTests {
    private let basicOpeningString = "2017-06-09 open Assets:Cash"
    private let basicClosingString = "2017-06-09 close Assets:Cash"
    
    private let whitespaceOpeningString = "2017-06-09    open    Assets:Cash      CAD"
    private let whitespaceClosingString = "2017-06-09   close      Assets:Cash"
    
    private let endOfLineCommentOpeningString = "2017-06-09 open Assets:Cash EUR ;gfsdt     "
    private let endOfLineCommentClosingString = "2017-06-09 close Assets:Cash   ;gfd "
    
    private let specialCharacterOpeningString = "2017-06-09 open Assets:💵 💵"
    private let specialCharacterClosingString = "2017-06-09 close Assets:💵"
    
    private let invalidCloseWithCommodityString = "2017-06-09 close Assets:Cash CAD"
    private let invalidCloseDateString = "2017-02-30 close Assets:Cash CAD"
    
    private let commodityWithSemicolonOpeningString = "2017-06-09 open Assets:Cash EUR;test ;gfsd"
    private let commodityWithSemicolonClosingString = "2017-06-09 close Assets:Cash ;gfsd"
    
    private let invalidNameOpeningString = "2017-06-09 open Assets::Cash"
    
    private let bookingMethodStrictOpeningString = "2017-06-09 open Assets:Cash EUR;test \"STRICT\" ;gfsd" // Commodity with Semicolon
    private let bookingMethodLifoOpeningString = "2017-06-09    open    Assets:Cash    EUR      \"LIFO\"     ;gfsd" // Whitespace
    private let bookingMethodFifoOpeningString = "2017-06-09 open Assets:Cash 💵 \"FIFO\" ;gfsd" // Special Character
    private let bookingMethodClosingString = "2017-06-09 close Assets:Cash ;gfsd"
    private let bookingMethodInClosingString = "2017-06-09 close Assets:Cash \"FIFO\" ;gfsd"
    
    @Test func basic() async {
        testWith(openingString: basicOpeningString, closingString: basicClosingString, commoditySymbol: nil)
    }
    
    @Test func invalidName() async {
        let account = AccountParser.parseFrom(line: invalidNameOpeningString)
        #expect(account == nil)
    }
    
    @Test func whitespace() async throws {
        testWith(openingString: whitespaceOpeningString, closingString: whitespaceClosingString, commoditySymbol: "CAD")
    }
    
    @Test func endOfLineComment() async throws {
        testWith(openingString: endOfLineCommentOpeningString, closingString: endOfLineCommentClosingString, commoditySymbol: "EUR")
    }
    
    @Test func specialCharacter() async throws {
        testWith(openingString: specialCharacterOpeningString, closingString: specialCharacterClosingString, commoditySymbol: "💵")
    }
    
    @Test(arguments: ["2017-06-09 close Assets:Cash CAD", "2017-02-30 close Assets:Cash CAD"])  func invalidClose(line: String) async throws {
        #expect(AccountParser.parseFrom(line: line) == nil)
    }
    
    @Test func commodityWithSemicolon() {
        testWith(openingString: commodityWithSemicolonOpeningString, closingString: commodityWithSemicolonClosingString, commoditySymbol: "EUR;test")
    }
    
    @Test func bookingMethodStrict() {
        testWith(openingString: bookingMethodStrictOpeningString, closingString: bookingMethodClosingString, commoditySymbol: "EUR;test", bookingMethod: .strict)
    }
    
    @Test func bookingMethodLifo() {
        testWith(openingString: bookingMethodLifoOpeningString, closingString: bookingMethodClosingString, commoditySymbol: "EUR", bookingMethod: .lifo)
    }
    
    @Test func bookingMethodFifo() {
        testWith(openingString: bookingMethodFifoOpeningString, closingString: bookingMethodClosingString, commoditySymbol: "💵", bookingMethod: .fifo)
    }
    
    @Test func bookingMethodInClosingString() async throws {
        let account = AccountParser.parseFrom(line: bookingMethodInClosingString)
        #expect(account == nil)
    }
    
//    let benchmarks = {
//        Benchmark("AcountParser.parseFrom") { benchmark in
//            performance()
//        }
//    }

    func performance() {
        for _ in 0...1_000 {
            _ = AccountParser.parseFrom(line: basicOpeningString)
            _ = AccountParser.parseFrom(line: basicClosingString)
            
            _ = AccountParser.parseFrom(line: whitespaceOpeningString)
            _ = AccountParser.parseFrom(line: whitespaceClosingString)
            
            _ = AccountParser.parseFrom(line: endOfLineCommentOpeningString)
            _ = AccountParser.parseFrom(line: endOfLineCommentClosingString)
            
            _ = AccountParser.parseFrom(line: specialCharacterOpeningString)
            _ = AccountParser.parseFrom(line: specialCharacterClosingString)
        }
    }
    
    
    // Helper
    private func testWith(openingString: String, closingString: String, commoditySymbol: CommoditySymbol?, bookingMethod: BookingMethod? = nil) {
        let account1 = AccountParser.parseFrom(line: openingString)
        
        #expect(account1 != nil)
        #expect(account1!.opening! == TestUtils.date20170609)
        #expect(account1!.closing == nil)
        #expect(account1!.commoditySymbol == commoditySymbol)
        
        if let bookingMethod {
            #expect(account1!.bookingMethod == bookingMethod)
        }
        
        let account2 = AccountParser.parseFrom(line: closingString)
        #expect(account2 != nil)
        #expect(account2!.opening == nil)
        #expect(account2!.closing! == TestUtils.date20170609)
    }
}
